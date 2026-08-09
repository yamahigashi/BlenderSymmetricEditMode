# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import copy
import time
import traceback
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import bmesh
import bpy
from bpy.app.handlers import persistent
from bpy.props import StringProperty

from . import backup, core, rip
from ._types import (
    Coordinate3D,
    FaceId,
    HiddenFaceMap,
    HistoryRecord,
    KnifeSession,
    MeshSelectionMode,
    MirrorFaceMap,
    OperatorResult,
    PathEdgeSignature,
    PathSignature,
    RipSignature,
    SymmetryAxes,
    ViewState,
    WindowContext,
)


@dataclass(frozen=True)
class ToolProfile:
    kind: str
    label: str
    wm_operator_names: tuple[str, ...]
    primary_wm_operator: str
    tool_idnames: tuple[str, ...]
    keymap_operator: str
    passthrough_handoff_grace: float
    passthrough_stable_ticks: int
    supports_nested_offset: bool
    # Adjust Last Operation re-executes the native operator via exec.  Rip's
    # macro cannot repeat (MESH_OT_rip has no exec), so only tools whose
    # native repeat is sound take part in the F9 baseline flow.
    supports_adjust_repeat: bool


TOOL_PROFILES: dict[str, ToolProfile] = {
    "KNIFE": ToolProfile(
        kind="KNIFE",
        label="Knife",
        wm_operator_names=("KNIFE_TOOL",),
        primary_wm_operator="MESH_OT_KNIFE_TOOL",
        tool_idnames=("3D View Tool: Edit Mesh, Knife",),
        keymap_operator="mesh.knife_tool",
        passthrough_handoff_grace=0.01,
        passthrough_stable_ticks=2,
        supports_nested_offset=False,
        supports_adjust_repeat=False,
    ),
    "LOOP_CUT": ToolProfile(
        kind="LOOP_CUT",
        label="Loop Cut",
        wm_operator_names=("LOOPCUT_SLIDE", "MESH_OT_LOOPCUT", "EDGE_SLIDE"),
        primary_wm_operator="MESH_OT_LOOPCUT_SLIDE",
        tool_idnames=("3D View Tool: Edit Mesh, Loop Cut",),
        keymap_operator="mesh.loopcut_slide",
        passthrough_handoff_grace=0.04,
        passthrough_stable_ticks=3,
        supports_nested_offset=False,
        supports_adjust_repeat=True,
    ),
    "OFFSET_LOOP_CUT": ToolProfile(
        kind="OFFSET_LOOP_CUT",
        label="Offset Edge Loop Cut",
        wm_operator_names=("OFFSET_EDGE_LOOPS_SLIDE", "MESH_OT_OFFSET_EDGE_LOOPS", "EDGE_SLIDE"),
        primary_wm_operator="MESH_OT_OFFSET_EDGE_LOOPS_SLIDE",
        tool_idnames=("3D View Tool: Edit Mesh, Offset Edge Loop Cut",),
        keymap_operator="mesh.offset_edge_loops_slide",
        passthrough_handoff_grace=0.04,
        passthrough_stable_ticks=3,
        supports_nested_offset=True,
        supports_adjust_repeat=True,
    ),
    "RIP": ToolProfile(
        kind="RIP",
        label="Rip",
        wm_operator_names=("RIP_MOVE",),
        primary_wm_operator="MESH_OT_RIP_MOVE",
        tool_idnames=(),
        keymap_operator="mesh.rip_move",
        passthrough_handoff_grace=0.04,
        passthrough_stable_ticks=3,
        supports_nested_offset=False,
        supports_adjust_repeat=False,
    ),
}

# Public compatibility views retained for existing imports.
TOOL_LABELS = {profile.kind: profile.label for profile in TOOL_PROFILES.values()}
MODAL_IDENTIFIER_TOKENS = {profile.kind: profile.wm_operator_names for profile in TOOL_PROFILES.values()}
_PASSTHROUGH_HANDOFF_GRACE = {profile.kind: profile.passthrough_handoff_grace for profile in TOOL_PROFILES.values()}
_PASSTHROUGH_STABLE_TICKS = {profile.kind: profile.passthrough_stable_ticks for profile in TOOL_PROFILES.values()}
_WM_OPERATOR_TO_TOOL = {
    profile.primary_wm_operator: profile.kind for profile in TOOL_PROFILES.values() if profile.supports_adjust_repeat
}

_SESSIONS: dict[int, KnifeSession] = {}
_HISTORY_RECORDS: OrderedDict[int, HistoryRecord] = OrderedDict()
_NEXT_HISTORY_TOKEN = max(1, int(time.time_ns() & 0x7FFFFFFF))
_HISTORY_REPAIR_QUEUED = False
_HISTORY_REPAIR_BUSY = False
_MAX_HISTORY_RECORDS = 256
_HISTORY_SEQUENCE = 0
# Last WARNING/INFO/ERROR messages recorded via _finish_report (tests inspect;
# Operator.report is C-bound and not patchable). Instrumentation is intentional
# and limited to the two call sites that tests assert on today: session-lost
# ERROR at finish entry, and the post-native decline/fatal classification
# (mirror-failure WARNING/ERROR after rollback). Other success INFO/WARNING
# paths still use Operator.report directly.
_FINISH_REPORTS: list[tuple[str, str]] = []

_PASSTHROUGH_POLL_INTERVAL = 0.01
_PASSTHROUGH_START_GRACE = 0.75


class SymmetricKnifeError(RuntimeError):
    """Expected validation or Blender-context failure shown in the UI."""


def _window_key(context: WindowContext) -> int:
    return context.window.as_pointer() if context.window is not None else 0


def _new_history_token() -> int:
    global _NEXT_HISTORY_TOKEN

    while True:
        token = _NEXT_HISTORY_TOKEN
        _NEXT_HISTORY_TOKEN = (_NEXT_HISTORY_TOKEN + 1) & 0x7FFFFFFF
        if _NEXT_HISTORY_TOKEN == 0:
            _NEXT_HISTORY_TOKEN = 1
        if token and token not in _HISTORY_RECORDS:
            return token


def _remember_history_session(session: KnifeSession, context) -> None:
    global _HISTORY_SEQUENCE

    _HISTORY_SEQUENCE += 1
    history_session = copy.copy(session)
    # Undo/Redo snapshots retain compact face CustomData mappings. Keeping the
    # equivalent Python dictionaries for every undo step is prohibitively
    # expensive on dense meshes, so they are reconstructed only when needed.
    history_session.mirror_face_ids = {}
    history_session.hidden_by_face_id = {}
    # Native Offset temporarily suspends Mesh Symmetry only while its Edge
    # Slide is live.  A later Undo/Redo repair must never rewrite user settings.
    history_session.symmetry_suspended = False
    _HISTORY_RECORDS[session.history_token] = HistoryRecord(
        session=history_session,
        sequence=_HISTORY_SEQUENCE,
    )
    _HISTORY_RECORDS.move_to_end(session.history_token)
    configured_limit = int(getattr(context.preferences.edit, "undo_steps", 32)) + 4
    history_limit = min(_MAX_HISTORY_RECORDS, max(8, configured_limit))
    while len(_HISTORY_RECORDS) > history_limit:
        _HISTORY_RECORDS.popitem(last=False)


def clear_history_records() -> None:
    _HISTORY_RECORDS.clear()


def _find_window(pointer: int):
    window_manager = bpy.context.window_manager
    if window_manager is None:
        return None
    return next(
        (window for window in window_manager.windows if window.as_pointer() == pointer),
        None,
    )


def _find_saved_view(session: KnifeSession):
    window = _find_window(session.window_pointer)
    area = (
        next(
            (candidate for candidate in window.screen.areas if candidate.as_pointer() == session.area_pointer),
            None,
        )
        if window is not None
        else None
    )
    if area is None and window is not None:
        area = next(
            (candidate for candidate in window.screen.areas if candidate.type == "VIEW_3D"),
            None,
        )
    region = (
        next(
            (candidate for candidate in area.regions if candidate.as_pointer() == session.region_pointer),
            None,
        )
        if area is not None
        else None
    )
    if region is None and area is not None:
        region = next(
            (candidate for candidate in area.regions if candidate.type == "WINDOW"),
            None,
        )
    return window, area, region


def _capture_view_state(area) -> ViewState | None:
    if area is None or area.type != "VIEW_3D":
        return None
    region_3d = area.spaces.active.region_3d
    return ViewState(
        view_rotation=region_3d.view_rotation.copy(),
        view_location=region_3d.view_location.copy(),
        view_distance=float(region_3d.view_distance),
        view_perspective=region_3d.view_perspective,
        view_camera_offset=tuple(region_3d.view_camera_offset),
        view_camera_zoom=float(region_3d.view_camera_zoom),
    )


def _apply_view_state(area, state: ViewState) -> None:
    region_3d = area.spaces.active.region_3d
    region_3d.view_rotation = state.view_rotation
    region_3d.view_location = state.view_location
    region_3d.view_distance = state.view_distance
    region_3d.view_perspective = state.view_perspective
    region_3d.view_camera_offset = state.view_camera_offset
    region_3d.view_camera_zoom = state.view_camera_zoom
    region_3d.update()


def _cleanup_object_layers(session: KnifeSession) -> None:
    obj = bpy.data.objects.get(session.object_name)
    if obj is None or obj.type != "MESH" or obj.data.name != session.mesh_name:
        return
    try:
        if obj.mode == "EDIT":
            bm = bmesh.from_edit_mesh(obj.data)
            if core.remove_temporary_layers(bm):
                bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
        else:
            core.remove_temporary_mesh_attributes(obj.data)
    except (ReferenceError, RuntimeError):
        # The object may be in the middle of an Undo or mode transition.  Stale
        # layers are also removed before the next session and before saving.
        pass


def _restore_mesh_symmetry(session: KnifeSession) -> None:
    """Restore axes temporarily suspended for native Offset Edge Slide."""

    if not session.symmetry_suspended:
        return
    try:
        obj = bpy.data.objects.get(session.object_name)
        if obj is not None and obj.type == "MESH" and obj.data.name == session.mesh_name:
            for attribute, enabled in zip(
                ("use_mesh_mirror_x", "use_mesh_mirror_y", "use_mesh_mirror_z"),
                session.symmetry_flags.as_tuple(),
                strict=True,
            ):
                setattr(obj, attribute, enabled)
    except (ReferenceError, RuntimeError):
        # Teardown also runs mid-Undo and during mode transitions; the flag
        # write must never abort the remaining layer and record cleanup.
        pass
    session.symmetry_suspended = False


def _suspend_mesh_symmetry(session: KnifeSession, obj) -> None:
    """Prevent Offset's built-in Edge Slide from moving the target half."""

    profile = TOOL_PROFILES.get(session.tool_kind)
    if profile is None or not profile.supports_nested_offset:
        return
    session.symmetry_suspended = True
    for attribute in (
        "use_mesh_mirror_x",
        "use_mesh_mirror_y",
        "use_mesh_mirror_z",
    ):
        setattr(obj, attribute, False)
    view_layer = bpy.context.view_layer
    if view_layer is not None:
        view_layer.update()


def _teardown_session_state(
    session: KnifeSession,
    *,
    keep_history_record: bool,
) -> None:
    """Release resources owned by a live or detached session snapshot."""

    _restore_mesh_symmetry(session)
    _cleanup_object_layers(session)
    if session.history_token and not keep_history_record:
        _HISTORY_RECORDS.pop(session.history_token, None)


def cleanup_session(
    window_pointer: int,
    *,
    keep_history_record: bool = False,
) -> None:
    session = _SESSIONS.pop(window_pointer, None)
    if session is not None:
        _teardown_session_state(
            session,
            keep_history_record=keep_history_record,
        )


def _cleanup_repair_session(session: KnifeSession) -> None:
    """Tear down a history-repair session whether or not finish detached it."""

    active = _SESSIONS.get(session.window_pointer)
    if active is not None and active.history_token == session.history_token:
        cleanup_session(
            session.window_pointer,
            keep_history_record=True,
        )
        return
    _teardown_session_state(session, keep_history_record=True)


def cleanup_all_sessions() -> None:
    for window_pointer in tuple(_SESSIONS):
        cleanup_session(window_pointer)


@persistent
def cleanup_stale_attributes(_dummy=None) -> None:
    """Save/load handler: temporary detection layers must never reach disk."""

    cleanup_all_sessions()
    for mesh in bpy.data.meshes:
        try:
            core.remove_temporary_mesh_attributes(mesh)
        except RuntimeError:
            pass


@persistent
def cleanup_after_load(_dummy=None) -> None:
    clear_history_records()
    cleanup_stale_attributes()


def _modal_operator_identifiers(window) -> set[str]:
    identifiers = set()
    for operator in window.modal_operators:
        for identifier in (
            getattr(getattr(operator, "bl_rna", None), "identifier", ""),
            getattr(operator, "bl_idname", ""),
        ):
            if identifier:
                identifiers.add(identifier.upper())
    return identifiers


def _native_tool_is_active(window, tool_kind: str) -> bool:
    tokens = MODAL_IDENTIFIER_TOKENS.get(tool_kind, ())
    return any(token in identifier for identifier in _modal_operator_identifiers(window) for token in tokens)


def _single_edit_mesh_poll(context) -> bool:
    obj = context.edit_object
    if context.area is None or context.area.type != "VIEW_3D":
        return False
    if context.region is None or context.region.type != "WINDOW":
        return False
    if context.mode != "EDIT_MESH" or obj is None or obj.type != "MESH":
        return False
    return len(context.objects_in_mode_unique_data) == 1


def _prepare_session(
    context,
    report,
    *,
    tool_kind: str = "KNIFE",
) -> bool:
    if tool_kind not in TOOL_LABELS:
        raise ValueError(f"Unsupported native tool kind: {tool_kind!r}")
    settings = context.scene.ydd_symmetric_edit
    obj = context.edit_object
    window_pointer = _window_key(context)

    conflicting_session = next(
        (
            session
            for pointer, session in _SESSIONS.items()
            if pointer != window_pointer and session.mesh_name == obj.data.name
        ),
        None,
    )
    if conflicting_session is not None:
        return False

    # A previous cancelled run is harmless, but clean it before reusing the
    # fixed temporary layer names.
    cleanup_session(window_pointer)
    bm = bmesh.from_edit_mesh(obj.data)
    core.remove_temporary_layers(bm)

    enabled_axes = core.enabled_mesh_symmetry_axes(obj)
    if len(enabled_axes) != 1:
        core.remove_temporary_layers(bm)
        bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
        return False

    _axis_name, axis_index = enabled_axes[0]
    if tool_kind == "RIP":
        guard = rip.prepare_guard_reason(context, bm, axis_index, settings.tolerance)
        if guard is not None:
            level, reason = guard
            bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
            if level == "WARNING":
                report({"WARNING"}, f"Rip is not mirrored: {reason}")
            return False

    history_token = _new_history_token()
    topology = core.prepare_topology(
        bm,
        axis_index,
        settings.tolerance,
        history_token,
        mark_vertex_ids=tool_kind == "RIP",
    )
    if topology.matched_faces == 0:
        core.remove_temporary_layers(bm)
        bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
        return False

    rip_snapshot = None
    if tool_kind == "RIP":
        rip_snapshot = rip.build_snapshot(bm, axis_index, settings.tolerance)
        if rip_snapshot is None:
            core.remove_temporary_layers(bm)
            bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
            return False

    bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
    session = KnifeSession(
        window_pointer=window_pointer,
        area_pointer=context.area.as_pointer(),
        region_pointer=context.region.as_pointer(),
        object_name=obj.name,
        mesh_name=obj.data.name,
        axis_index=axis_index,
        source_side=settings.source_side,
        tolerance=settings.tolerance,
        mirror_face_ids=topology.mirror_face_ids,
        hidden_by_face_id=topology.hidden_by_face_id,
        carrier_frames=topology.carrier_frames,
        mesh_select_mode=MeshSelectionMode(
            vertices=bool(context.tool_settings.mesh_select_mode[0]),
            edges=bool(context.tool_settings.mesh_select_mode[1]),
            faces=bool(context.tool_settings.mesh_select_mode[2]),
        ),
        started_at=time.monotonic(),
        tool_kind=tool_kind,
        history_token=history_token,
        symmetry_flags=SymmetryAxes(
            x=bool(obj.use_mesh_mirror_x),
            y=bool(obj.use_mesh_mirror_y),
            z=bool(obj.use_mesh_mirror_z),
        ),
        rip=rip_snapshot,
    )
    _SESSIONS[window_pointer] = session
    _suspend_mesh_symmetry(session, obj)
    _remember_history_session(session, context)
    _schedule_passthrough_watcher(window_pointer, history_token)

    if topology.matched_faces < topology.total_faces:
        report(
            {"WARNING"},
            f"Only {topology.matched_faces} of {topology.total_faces} faces have an exact mirrored counterpart",
        )
    return True


class MESH_OT_ydd_symmetric_edit_intercept(bpy.types.Operator):
    """Prepare symmetry, then let the exact native tool event continue."""

    bl_idname = "mesh.ydd_symmetric_edit_intercept"
    bl_label = "Prepare ydd Symmetric Edit for Native Cut Tool"
    bl_description = "Attach symmetric post-processing to this native tool route"
    bl_options = {"INTERNAL"}

    if TYPE_CHECKING:
        route_key: str
    else:
        route_key: StringProperty(options={"HIDDEN", "SKIP_SAVE"})

    @classmethod
    def poll(cls, context):
        return _single_edit_mesh_poll(context)

    def invoke(
        self,
        context: bpy.types.Context,
        event: bpy.types.Event,
    ) -> OperatorResult:
        del event
        from . import keymaps

        if not keymaps.route_is_current(self.route_key):
            return {"PASS_THROUGH"}
        tool_kind = keymaps.route_tool_kind(self.route_key)
        if tool_kind is None:
            return {"PASS_THROUGH"}
        try:
            _prepare_session(
                context,
                self.report,
                tool_kind=tool_kind,
            )
        except Exception:
            traceback.print_exc()
            cleanup_session(_window_key(context))
        # Do not invoke the native operator here. Passing through the original
        # physical event preserves toolbar first-click/drag behavior and all
        # native operator properties.
        return {"PASS_THROUGH"}


def _create_cutter_object(context, target, coordinates, edges):
    mesh = bpy.data.meshes.new("YSE_TemporaryCutter")
    mesh.from_pydata(coordinates, edges, [])
    mesh.update()
    cutter = bpy.data.objects.new("YSE_TemporaryCutter", mesh)
    context.scene.collection.objects.link(cutter)
    cutter.matrix_world = target.matrix_world.copy()
    cutter.display_type = "WIRE"
    cutter.hide_render = True
    context.view_layer.update()
    return cutter, mesh


def _remove_cutter(cutter, mesh) -> None:
    """Best-effort temporary cutter teardown.

    Called from finish ``finally`` blocks: must not raise. Any exception is
    logged and swallowed so later stages (backup rollback, selection restore,
    layer removal, result return) still run.
    """

    try:
        if cutter is not None and cutter.name in bpy.data.objects:
            bpy.data.objects.remove(cutter, do_unlink=True)
        if mesh is not None and mesh.name in bpy.data.meshes and mesh.users == 0:
            bpy.data.meshes.remove(mesh)
    except Exception:
        traceback.print_exc()


def _finish_rip_session(
    operator: bpy.types.Operator,
    session: KnifeSession,
    obj,
    window_pointer: int,
) -> OperatorResult:
    """Mirror a confirmed native Rip.  All-or-nothing.

    The native result already changed the mesh, so this always returns
    FINISHED; a mirror failure restores the pre-mirror state and reports a
    WARNING while keeping the native rip intact (undo stays one step).
    Backup creation failure is fatal ERROR, same as the cut-tool finish path.
    """

    backup_mesh = None
    mirrored_count = 0
    reason: str | None = None
    backup_creation_failed = False
    rollback_failed = False
    try:
        if session.rip is None:
            reason = "the pre-rip snapshot was lost"
        else:
            bm = bmesh.from_edit_mesh(obj.data)
            reason = rip.preflight_reason(bm, session.rip, session.mirror_face_ids)
            if reason is None:
                try:
                    backup_mesh = backup.create_topology_backup(bm)
                except Exception as exc:
                    traceback.print_exc()
                    backup_creation_failed = True
                    reason = f"Could not create topology backup for rollback: {exc}"
                if backup_mesh is not None:
                    bm = bmesh.from_edit_mesh(obj.data)
                    mirrored_count, reason = rip.apply_mirrored_rip(bm, session.rip, session.mirror_face_ids)
                    if reason is None:
                        bmesh.update_edit_mesh(obj.data, loop_triangles=True, destructive=True)
                    else:
                        mirrored_count = 0
                        try:
                            backup.restore_topology_backup(obj.data, backup_mesh)
                        except Exception:
                            traceback.print_exc()
                            rollback_failed = True
                            reason = f"Rip mirror failed and rollback failed: {reason}"
    except Exception as exc:
        traceback.print_exc()
        reason = str(exc)
        if backup_mesh is not None:
            try:
                backup.restore_topology_backup(obj.data, backup_mesh)
            except Exception:
                traceback.print_exc()
                rollback_failed = True
                reason = f"Rip mirror failed and rollback failed: {exc}"
    finally:
        if obj is not None and obj.mode == "EDIT":
            try:
                bm = bmesh.from_edit_mesh(obj.data)
                core.remove_temporary_layers(bm)
                bmesh.update_edit_mesh(obj.data, loop_triangles=True, destructive=True)
            except (ReferenceError, RuntimeError):
                pass
        backup.remove_backup(backup_mesh)
        cleanup_session(window_pointer, keep_history_record=True)

    if reason is not None:
        level = {"ERROR"} if backup_creation_failed or rollback_failed else {"WARNING"}
        _finish_report(operator, level, f"ydd Symmetric Edit: Rip was not mirrored: {reason}")
    else:
        _finish_report(operator, {"INFO"}, f"Mirrored Rip across {mirrored_count} seam edge(s)")
        _maybe_extend_selection_to_mirror(obj, session.axis_index, session.tolerance)
    return {"FINISHED"}


def _finish_report(operator, level: set[str], message: str) -> None:
    """Report and record for tests (``Operator.report`` itself is not patchable)."""

    kind = "WARNING" if "WARNING" in level else "ERROR" if "ERROR" in level else "INFO"
    _FINISH_REPORTS.append((kind, message))
    operator.report(level, message)


def _maybe_extend_selection_to_mirror(obj, axis_index: int, tolerance: float) -> None:
    """When Scene ``select_mirrored`` is on, add-select ρ(S) after a success.

    Best-effort: selection restore / layer cleanup must not fail because of
    this.  Never mutates ``select_history`` (delegated to core).
    """

    try:
        scene = bpy.context.scene
        settings = getattr(scene, "ydd_symmetric_edit", None)
        if settings is None or not bool(getattr(settings, "select_mirrored", False)):
            return
        if obj is None or obj.mode != "EDIT":
            return
        bm = bmesh.from_edit_mesh(obj.data)
        core.extend_selection_to_mirror(bm, axis_index, tolerance)
        bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
    except Exception:
        traceback.print_exc()


class MESH_OT_ydd_symmetric_edit_finish(bpy.types.Operator):
    bl_idname = "mesh.ydd_symmetric_edit_finish"
    bl_label = "Apply Mirrored ydd Symmetric Edit Cut"
    bl_options = {"INTERNAL"}

    @classmethod
    def poll(cls, context):
        return _single_edit_mesh_poll(context)

    def execute(self, context):
        _FINISH_REPORTS.clear()
        window_pointer = _window_key(context)
        session = _SESSIONS.get(window_pointer)
        if session is None:
            _finish_report(self, {"ERROR"}, "ydd Symmetric Edit session data was lost")
            return {"CANCELLED"}

        tool_label = TOOL_LABELS.get(session.tool_kind, "Cut Tool")
        _restore_mesh_symmetry(session)
        obj = bpy.data.objects.get(session.object_name)
        if obj is None or obj != context.edit_object or obj.type != "MESH" or obj.data.name != session.mesh_name:
            cleanup_session(window_pointer)
            self.report({"ERROR"}, f"The edited mesh changed during {tool_label}")
            return {"CANCELLED"}

        if session.tool_kind == "RIP":
            return _finish_rip_session(self, session, obj, window_pointer)

        cutter = None
        cutter_mesh = None
        backup_mesh = None
        projection_committed = False
        preexisting_vertex_keys = set()
        result = {"CANCELLED"}
        mirrored_segment_count = 0
        crossing_count = 0
        warning = ""
        selection_state = None
        side: str | None = None
        mirror_failure: str | None = None
        rollback_failed = False
        backup_creation_failed = False
        mesh_select_mode = MeshSelectionMode(
            vertices=bool(context.tool_settings.mesh_select_mode[0]),
            edges=bool(context.tool_settings.mesh_select_mode[1]),
            faces=bool(context.tool_settings.mesh_select_mode[2]),
        )

        try:
            bm = bmesh.from_edit_mesh(obj.data)
            edge_layer, face_layer = core.get_required_layers(bm)
            if edge_layer is None or face_layer is None:
                raise SymmetricKnifeError("Temporary topology markers are missing")

            def _create_backup(edit_bm):
                """Create topology backup; classify create failure as fatal."""
                nonlocal backup_creation_failed
                try:
                    return backup.create_topology_backup(edit_bm)
                except Exception as exc:
                    traceback.print_exc()
                    backup_creation_failed = True
                    raise SymmetricKnifeError(f"Could not create topology backup for rollback: {exc}") from exc

            if session.tool_kind == "KNIFE":
                # Both-sides mirror. CROSSES are p-stitched first, then
                # half-edges join the POSITIVE/NEGATIVE mirror path inside
                # one backup transaction.
                by_side, total_path_edges = core.collect_knife_path_edges_by_side(
                    bm,
                    session.axis_index,
                    session.tolerance,
                )
                if total_path_edges == 0:
                    result = {"FINISHED"}
                    self.report({"INFO"}, f"{tool_label} made no new cut")
                    return result

                def _all_path_edges(sides: dict) -> list:
                    return (
                        list(sides.get("POSITIVE", ()))
                        + list(sides.get("NEGATIVE", ()))
                        + list(sides.get("CROSSES", ()))
                        + list(sides.get("PLANE", ()))
                    )

                crossing_count = len(by_side["CROSSES"])
                side = "BOTH"
                live_mirror_face_ids = core.resolve_live_mirror_face_map(
                    bm,
                    session.mirror_face_ids,
                    session.axis_index,
                    session.tolerance,
                    path_edges=_all_path_edges(by_side),
                )

                if crossing_count:
                    # Single backup covers p-stitch + mirror.
                    selection_state = core.add_selection_layers(bm)
                    backup_mesh = _create_backup(bm)
                    bm = bmesh.from_edit_mesh(obj.data)
                    by_side, total_path_edges = core.collect_knife_path_edges_by_side(
                        bm,
                        session.axis_index,
                        session.tolerance,
                    )
                    live_mirror_face_ids = core.resolve_live_mirror_face_map(
                        bm,
                        session.mirror_face_ids,
                        session.axis_index,
                        session.tolerance,
                        path_edges=_all_path_edges(by_side),
                    )
                    stitched, stitch_reason = core.apply_crosses_p_stitch(
                        bm,
                        by_side["CROSSES"],
                        session.axis_index,
                        session.tolerance,
                    )
                    if stitch_reason:
                        raise SymmetricKnifeError(stitch_reason)
                    # Half-edges reclassify as POSITIVE/NEGATIVE after the split.
                    by_side, total_path_edges = core.collect_knife_path_edges_by_side(
                        bm,
                        session.axis_index,
                        session.tolerance,
                    )
                    live_mirror_face_ids = core.resolve_live_mirror_face_map(
                        bm,
                        session.mirror_face_ids,
                        session.axis_index,
                        session.tolerance,
                        path_edges=_all_path_edges(by_side),
                    )
                    crossing_count = len(by_side["CROSSES"])
                    # Remaining CROSSES are self-mirrored (skipped by stitch) or
                    # still straddling after a failed reclassification — the
                    # latter is a bug; treat as decline via unmatched mirror.
                    if stitched:
                        # Persist p-stitch into the edit mesh so subsequent
                        # BMesh rebuilds (layer add / backup refresh) see it.
                        bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=True)

                        bm = bmesh.from_edit_mesh(obj.data)
                        _edge_layer, face_layer = core.get_required_layers(bm)
                        by_side, total_path_edges = core.collect_knife_path_edges_by_side(
                            bm,
                            session.axis_index,
                            session.tolerance,
                        )
                        live_mirror_face_ids = core.resolve_live_mirror_face_map(
                            bm,
                            session.mirror_face_ids,
                            session.axis_index,
                            session.tolerance,
                            path_edges=_all_path_edges(by_side),
                        )

                crossing_plan, crossing_reason = core.plan_mirrored_path_crossings(
                    bm,
                    by_side,
                    session.axis_index,
                    session.tolerance,
                    live_mirror_face_ids,
                    session.carrier_frames,
                )
                if crossing_reason:
                    raise SymmetricKnifeError(crossing_reason)
                if crossing_plan:
                    if backup_mesh is None:
                        selection_state = core.add_selection_layers(bm)
                        backup_mesh = _create_backup(bm)
                        bm = bmesh.from_edit_mesh(obj.data)
                        _edge_layer, face_layer = core.get_required_layers(bm)
                        by_side, total_path_edges = core.collect_knife_path_edges_by_side(
                            bm,
                            session.axis_index,
                            session.tolerance,
                        )
                        live_mirror_face_ids = core.resolve_live_mirror_face_map(
                            bm,
                            session.mirror_face_ids,
                            session.axis_index,
                            session.tolerance,
                            path_edges=_all_path_edges(by_side),
                        )
                        crossing_plan, crossing_reason = core.plan_mirrored_path_crossings(
                            bm,
                            by_side,
                            session.axis_index,
                            session.tolerance,
                            live_mirror_face_ids,
                            session.carrier_frames,
                        )
                        if crossing_reason:
                            raise SymmetricKnifeError(crossing_reason)
                        if not crossing_plan:
                            raise SymmetricKnifeError("The mirrored path crossing plan changed before apply")

                    _crossings_stitched, crossing_reason = core.apply_mirrored_path_crossings(
                        bm,
                        crossing_plan,
                    )
                    if crossing_reason:
                        raise SymmetricKnifeError(crossing_reason)
                    by_side, total_path_edges = core.collect_knife_path_edges_by_side(
                        bm,
                        session.axis_index,
                        session.tolerance,
                    )
                    live_mirror_face_ids = core.resolve_live_mirror_face_map(
                        bm,
                        session.mirror_face_ids,
                        session.axis_index,
                        session.tolerance,
                        path_edges=_all_path_edges(by_side),
                    )
                    _edge_layer, face_layer = core.get_required_layers(bm)

                source_edges = by_side["POSITIVE"] + by_side["NEGATIVE"]
                if not source_edges:
                    # Only PLANE and/or self-mirrored CROSSES: already symmetric.
                    if backup_mesh is not None:
                        # No further mutation; drop the unused backup cleanly.
                        projection_committed = True
                    result = {"FINISHED"}
                    self.report(
                        {"INFO"},
                        f"{tool_label} cut lies on the mirror plane",
                    )
                    return result

                target_face_ids, unmatched = core.target_face_ids_for_edges(
                    source_edges,
                    face_layer,
                    live_mirror_face_ids,
                )
                if not target_face_ids:
                    raise SymmetricKnifeError(
                        "The cut faces have no exact mirrored counterpart; adjust axis or tolerance"
                    )
                if unmatched:
                    raise SymmetricKnifeError(f"{len(unmatched)} cut face(s) have no exact mirrored counterpart")

                use_direct_topology = core.reflected_path_uses_only_target_boundaries(
                    bm,
                    source_edges,
                    session.axis_index,
                    session.tolerance,
                    live_mirror_face_ids,
                )
                if use_direct_topology:
                    if backup_mesh is None:
                        selection_state = core.add_selection_layers(bm)
                        backup_mesh = _create_backup(bm)
                        bm = bmesh.from_edit_mesh(obj.data)
                        by_side, total_path_edges = core.collect_knife_path_edges_by_side(
                            bm,
                            session.axis_index,
                            session.tolerance,
                        )
                        live_mirror_face_ids = core.resolve_live_mirror_face_map(
                            bm,
                            session.mirror_face_ids,
                            session.axis_index,
                            session.tolerance,
                            path_edges=_all_path_edges(by_side),
                        )
                        source_edges = by_side["POSITIVE"] + by_side["NEGATIVE"]
                    if not source_edges:
                        raise SymmetricKnifeError("The native cut path was lost before mirroring")
                    created, already_present, direct_reason = core.apply_reflected_path_topology(
                        bm,
                        source_edges,
                        session.axis_index,
                        session.tolerance,
                        live_mirror_face_ids,
                    )
                    if direct_reason:
                        raise SymmetricKnifeError(f"Could not rebuild the mirrored {tool_label}: {direct_reason}")
                    if created + already_present != len(source_edges):
                        raise SymmetricKnifeError(f"The mirrored {tool_label} topology did not match the source")

                    projection_committed = True
                    result = {"FINISHED"}
                    if not created:
                        self.report(
                            {"INFO"},
                            f"The opposite side already contains this {tool_label}",
                        )
                        return result

                    warning_parts = []
                    if already_present:
                        warning_parts.append(f"{already_present} segment(s) already existed")
                    suffix = f"; {'; '.join(warning_parts)}" if warning_parts else ""
                    self.report(
                        {"WARNING"} if warning_parts else {"INFO"},
                        f"Mirrored {created} {tool_label} segment(s) to both sides{suffix}",
                    )
                    return result

                coordinates, cutter_edges, already_present = core.build_reflected_cutter(
                    bm,
                    source_edges,
                    session.axis_index,
                    session.tolerance,
                )
                if not cutter_edges:
                    if backup_mesh is not None:
                        # p-stitch may have run; keep those splits if any (they
                        # are part of the X scaffold). Commit when CROSSES were
                        # stitched even if the mirror cutter is empty.
                        projection_committed = True
                    result = {"FINISHED"}
                    self.report({"INFO"}, "The opposite side already contains this cut")
                    return result

                core.reserve_source_path_marker(bm)
                if backup_mesh is None:
                    selection_state = core.add_selection_layers(bm)
                    backup_mesh = _create_backup(bm)
                    preexisting_vertex_keys = {hash(vertex) for vertex in bm.verts}
                else:
                    # Backup already covers pre-p-stitch native; refresh the
                    # vertex key set after the stitch so snap only moves new
                    # projection geometry.
                    preexisting_vertex_keys = {hash(vertex) for vertex in bm.verts}
                _edge_layer, face_layer = core.get_required_layers(bm)
                for face in bm.faces:
                    face.hide = FaceId(int(face[face_layer])) not in target_face_ids
                bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)

                cutter, cutter_mesh = _create_cutter_object(context, obj, coordinates, cutter_edges)

                window, area, region = _find_saved_view(session)
                if window is None or area is None or region is None:
                    raise SymmetricKnifeError("The original 3D Viewport is no longer available")

                current_view = _capture_view_state(area)
                try:
                    if session.projection_view is not None:
                        _apply_view_state(area, session.projection_view)
                    with context.temp_override(
                        window=window,
                        area=area,
                        region=region,
                        selected_objects=[cutter],
                    ):
                        knife_project = getattr(bpy.ops.mesh, "knife_project")
                        if not cast(bool, knife_project.poll()):
                            raise SymmetricKnifeError("Knife Project has no valid 3D View context")
                        knife_result = cast(
                            set[str],
                            knife_project(cut_through=True),
                        )
                finally:
                    if current_view is not None:
                        _apply_view_state(area, current_view)
                if "FINISHED" not in knife_result:
                    raise SymmetricKnifeError("Blender's Knife Project did not finish")

                mirrored_segment_count = len(cutter_edges)
                bm = bmesh.from_edit_mesh(obj.data)
                snapped, _projection_error, snap_reason = core.snap_projected_graph(
                    bm,
                    coordinates,
                    cutter_edges,
                    session.tolerance,
                    preexisting_vertex_keys,
                )
                if not snapped:
                    raise SymmetricKnifeError(f"Projected cut could not be snapped to exact symmetry: {snap_reason}")
                projection_committed = True
                if already_present:
                    warning += f"; {already_present} segment(s) already existed"
                result = {"FINISHED"}
            else:
                # Loop Cut / Offset keep the one-side source selection.
                # Straddling rings symmetrize through the ordinary reflection
                # only because their carrier faces map to themselves or their
                # pairs; when pairing fails the counterpart check declines.
                source_edges, side, total_path_edges, crossing_count = core.collect_source_path_edges(
                    bm,
                    session.axis_index,
                    session.tolerance,
                    session.source_side,
                    selected_only=session.tool_kind
                    in {
                        "LOOP_CUT",
                        "OFFSET_LOOP_CUT",
                    },
                )
                if total_path_edges == 0:
                    result = {"FINISHED"}
                    self.report({"INFO"}, f"{tool_label} made no new cut")
                    return result
                if side is None or not source_edges:
                    raise SymmetricKnifeError(
                        "Could not determine the source side; keep the new topology on one side of the mirror plane"
                    )

                # Native loopcut skips hidden ring edges, leaving an open
                # partial ring whose mirror is not well-defined.
                if core.path_ring_includes_pre_hidden_edges(bm):
                    raise SymmetricKnifeError("the cut ring includes hidden edges; partial ring cuts are not mirrored")

                target_face_ids, unmatched = core.target_face_ids_for_edges(
                    source_edges,
                    face_layer,
                    session.mirror_face_ids,
                )
                if not target_face_ids:
                    raise SymmetricKnifeError(
                        "The cut faces have no exact mirrored counterpart; adjust axis or tolerance"
                    )
                if unmatched:
                    raise SymmetricKnifeError(f"{len(unmatched)} cut face(s) have no exact mirrored counterpart")

                collapsed_target_markers = set()
                profile = TOOL_PROFILES.get(session.tool_kind)
                if profile is not None and profile.supports_nested_offset:
                    collapsed_target_markers, _collapsed_reason = core.collapsed_offset_target_edge_markers(
                        bm,
                        source_edges,
                        session.axis_index,
                        session.tolerance,
                    )
                if collapsed_target_markers:
                    # Esc cancels only Offset's Edge Slide child; Blender keeps the
                    # two new loops at factor zero. A coincident Knife Project is
                    # impossible, so reproduce the same topology with BMesh.
                    core.reserve_source_path_marker(bm)
                    selection_state = core.add_selection_layers(bm)
                    backup_mesh = _create_backup(bm)
                    mirrored_segment_count, collapsed_reason = core.apply_collapsed_offset_topology(
                        bm,
                        collapsed_target_markers,
                        use_cap_endpoint=session.offset_use_cap_endpoint,
                    )
                    if not mirrored_segment_count:
                        raise SymmetricKnifeError(collapsed_reason)
                    if mirrored_segment_count != len(source_edges):
                        raise SymmetricKnifeError("The mirrored zero-offset topology did not match the source")
                    projection_committed = True
                    result = {"FINISHED"}
                    self.report(
                        {"INFO"},
                        f"Mirrored {mirrored_segment_count} zero-factor Offset "
                        f"Edge Loop Cut segment(s) from the {side.lower()} side",
                    )
                    return result

                use_direct_topology = session.tool_kind in {
                    "LOOP_CUT",
                    "OFFSET_LOOP_CUT",
                }
                if use_direct_topology:
                    # A viewport projection is not one-to-one on curved or
                    # self-occluding surfaces. Rebuild the native cut topology on
                    # paired target faces instead, using exact reflected points.
                    selection_state = core.add_selection_layers(bm)
                    backup_mesh = _create_backup(bm)
                    bm = bmesh.from_edit_mesh(obj.data)
                    source_edges, side, total_path_edges, crossing_count = core.collect_source_path_edges(
                        bm,
                        session.axis_index,
                        session.tolerance,
                        session.source_side,
                        selected_only=session.tool_kind
                        in {
                            "LOOP_CUT",
                            "OFFSET_LOOP_CUT",
                        },
                    )
                    if side is None or not source_edges:
                        raise SymmetricKnifeError("The native cut path was lost before mirroring")
                    created, already_present, direct_reason = core.apply_reflected_path_topology(
                        bm,
                        source_edges,
                        session.axis_index,
                        session.tolerance,
                        session.mirror_face_ids,
                    )
                    if direct_reason:
                        raise SymmetricKnifeError(f"Could not rebuild the mirrored {tool_label}: {direct_reason}")
                    if created + already_present != len(source_edges):
                        raise SymmetricKnifeError(f"The mirrored {tool_label} topology did not match the source")

                    projection_committed = True
                    result = {"FINISHED"}
                    if not created:
                        _finish_report(
                            self,
                            {"INFO"},
                            f"The opposite side already contains this {tool_label}",
                        )
                        return result

                    warning_parts = []
                    if crossing_count:
                        warning_parts.append(f"skipped {crossing_count} segment(s) crossing the mirror plane")
                    if already_present:
                        warning_parts.append(f"{already_present} segment(s) already existed")
                    suffix = f"; {'; '.join(warning_parts)}" if warning_parts else ""
                    _finish_report(
                        self,
                        {"WARNING"} if warning_parts else {"INFO"},
                        f"Mirrored {created} {tool_label} segment(s) from the {side.lower()} side{suffix}",
                    )
                    return result

                coordinates, cutter_edges, already_present = core.build_reflected_cutter(
                    bm,
                    source_edges,
                    session.axis_index,
                    session.tolerance,
                )
                if not cutter_edges:
                    result = {"FINISHED"}
                    self.report({"INFO"}, "The opposite side already contains this cut")
                    return result

                core.reserve_source_path_marker(bm)
                selection_state = core.add_selection_layers(bm)
                backup_mesh = _create_backup(bm)
                preexisting_vertex_keys = {hash(vertex) for vertex in bm.verts}
                # Layer creation invalidates held element wrappers; retrieve layers
                # and iterate faces again before changing visibility.
                _edge_layer, face_layer = core.get_required_layers(bm)
                for face in bm.faces:
                    face.hide = FaceId(int(face[face_layer])) not in target_face_ids
                bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)

                cutter, cutter_mesh = _create_cutter_object(context, obj, coordinates, cutter_edges)

                window, area, region = _find_saved_view(session)
                if window is None or area is None or region is None:
                    raise SymmetricKnifeError("The original 3D Viewport is no longer available")

                current_view = _capture_view_state(area)
                try:
                    if session.projection_view is not None:
                        _apply_view_state(area, session.projection_view)
                    with context.temp_override(
                        window=window,
                        area=area,
                        region=region,
                        selected_objects=[cutter],
                    ):
                        knife_project = getattr(bpy.ops.mesh, "knife_project")
                        if not cast(bool, knife_project.poll()):
                            raise SymmetricKnifeError("Knife Project has no valid 3D View context")
                        knife_result = cast(
                            set[str],
                            knife_project(cut_through=True),
                        )
                finally:
                    if current_view is not None:
                        _apply_view_state(area, current_view)
                if "FINISHED" not in knife_result:
                    raise SymmetricKnifeError("Blender's Knife Project did not finish")

                mirrored_segment_count = len(cutter_edges)
                bm = bmesh.from_edit_mesh(obj.data)
                snapped, _projection_error, snap_reason = core.snap_projected_graph(
                    bm,
                    coordinates,
                    cutter_edges,
                    session.tolerance,
                    preexisting_vertex_keys,
                )
                if not snapped:
                    raise SymmetricKnifeError(f"Projected cut could not be snapped to exact symmetry: {snap_reason}")
                projection_committed = True
                if crossing_count:
                    warning += f"; skipped {crossing_count} segment(s) crossing the mirror plane"
                if already_present:
                    warning += f"; {already_present} segment(s) already existed"
                result = {"FINISHED"}
        except SymmetricKnifeError as exc:
            # Defer report until after finally so rollback success/failure can
            # classify the outcome.
            mirror_failure = str(exc)
        except Exception as exc:
            traceback.print_exc()
            mirror_failure = str(exc)
        finally:
            # Each stage is isolated: a failure here must not skip later stages
            # or prevent returning a post-native result.
            _remove_cutter(cutter, cutter_mesh)

            if backup_mesh is not None and not projection_committed:
                try:
                    backup.restore_topology_backup(obj.data, backup_mesh)
                except Exception:
                    traceback.print_exc()
                    rollback_failed = True

            # Knife Project replaces selection and temporary hiding even when a
            # later step fails. Best-effort restoration keeps the native source
            # result usable (and the whole operation remains one Undo step).
            if obj is not None and obj.mode == "EDIT":
                try:
                    bm = bmesh.from_edit_mesh(obj.data)
                    context.tool_settings.mesh_select_mode = mesh_select_mode.as_tuple()
                    if selection_state is not None:
                        core.restore_visibility_and_selection(
                            bm,
                            session.hidden_by_face_id,
                            selection_state,
                        )
                    else:
                        face_layer = bm.faces.layers.int.get(core.FACE_ID_LAYER)
                        if face_layer is not None:
                            for face in bm.faces:
                                face_id = FaceId(int(face[face_layer]))
                                face.hide = bool(session.hidden_by_face_id.get(face_id, False))
                    core.remove_temporary_layers(bm)
                    # Select Mirrored runs after selection restore + layer cleanup
                    # so it sees the final native selection and permanent topology.
                    # Early returns still execute this finally (Python semantics).
                    if mirror_failure is None:
                        try:
                            settings = getattr(context.scene, "ydd_symmetric_edit", None)
                            if settings is not None and bool(getattr(settings, "select_mirrored", False)):
                                core.extend_selection_to_mirror(
                                    bm,
                                    session.axis_index,
                                    session.tolerance,
                                )
                        except Exception:
                            traceback.print_exc()
                    bmesh.update_edit_mesh(obj.data, loop_triangles=True, destructive=True)
                except Exception:
                    traceback.print_exc()

            try:
                backup.remove_backup(backup_mesh)
            except Exception:
                traceback.print_exc()
            try:
                cleanup_session(window_pointer, keep_history_record=True)
            except Exception:
                traceback.print_exc()

        # After the native cut mutated the mesh, always return FINISHED so
        # its undo push survives.
        # Successful rollback (or no backup / mirror not started) → WARNING
        # decline that keeps the native result. Backup create failure or
        # rollback exception → fatal ERROR. Pre-mirror decline that never
        # attempted backup stays WARNING.
        if mirror_failure is not None:
            if rollback_failed or backup_creation_failed:
                _finish_report(
                    self,
                    {"ERROR"},
                    f"ydd Symmetric Edit: {mirror_failure}",
                )
            else:
                _finish_report(
                    self,
                    {"WARNING"},
                    f"ydd Symmetric Edit: {mirror_failure}",
                )
            result = {"FINISHED"}

        # Success INFO only when the mirror stage completed (not after a
        # WARNING decline that still returns FINISHED). Routed through
        # _finish_report so tests can observe dual-report suppression.
        if result == {"FINISHED"} and mirrored_segment_count and mirror_failure is None:
            assert side is not None
            if side == "BOTH":
                message = f"Mirrored {mirrored_segment_count} {tool_label} segment(s) to both sides{warning}"
            else:
                message = (
                    f"Mirrored {mirrored_segment_count} {tool_label} segment(s) from the {side.lower()} side{warning}"
                )
            _finish_report(self, {"WARNING"} if warning else {"INFO"}, message)
        return result


def _session_new_path_signature(session: KnifeSession) -> PathSignature | RipSignature | None:
    """Return a stable signature for topology created by the native tool."""

    obj = bpy.data.objects.get(session.object_name)
    if obj is None or obj.type != "MESH" or obj.data.name != session.mesh_name:
        return None
    try:
        if obj.mode == "EDIT":
            bm = bmesh.from_edit_mesh(obj.data)
            if session.tool_kind == "RIP":
                return rip.rip_result_signature(bm)
            marker_layer = bm.edges.layers.int.get(core.EDGE_ORIGINAL_LAYER)
            if marker_layer is None:
                return None
            path = []
            for edge in bm.edges:
                if int(edge[marker_layer]) > 0:
                    continue
                endpoints = sorted(
                    Coordinate3D(
                        x=round(float(vertex.co[0]), 9),
                        y=round(float(vertex.co[1]), 9),
                        z=round(float(vertex.co[2]), 9),
                    )
                    for vertex in edge.verts
                )
                path.append(
                    PathEdgeSignature(
                        first=endpoints[0],
                        second=endpoints[1],
                    )
                )
            return tuple(sorted(path)) if path else None
    except (ReferenceError, RuntimeError):
        return None
    return None


def _session_has_new_path(session: KnifeSession) -> bool:
    """Return as soon as one edge created by the native tool is found."""

    obj = bpy.data.objects.get(session.object_name)
    if obj is None or obj.type != "MESH" or obj.data.name != session.mesh_name:
        return False
    try:
        if obj.mode == "EDIT":
            bm = bmesh.from_edit_mesh(obj.data)
            if session.tool_kind == "RIP":
                return rip.has_rip_result(bm)
            marker_layer = bm.edges.layers.int.get(core.EDGE_ORIGINAL_LAYER)
            if marker_layer is None:
                return False
            return any(int(edge[marker_layer]) <= 0 for edge in bm.edges)
    except (ReferenceError, RuntimeError):
        return False
    return False


def _capture_projection_view(session: KnifeSession) -> None:
    _window, area, _region = _find_saved_view(session)
    state = _capture_view_state(area)
    if state is None:
        return
    session.projection_view = state
    record = _HISTORY_RECORDS.get(session.history_token)
    if record is not None:
        record.session.projection_view = copy.deepcopy(state)


def _capture_native_result_options(session: KnifeSession, context) -> None:
    """Retain native macro options needed by a topology-only fallback."""

    profile = TOOL_PROFILES.get(session.tool_kind)
    expected_identifier = profile.primary_wm_operator if profile is not None else ""
    operator = context.active_operator
    if operator is None or getattr(operator, "bl_idname", "").upper() != (expected_identifier):
        operator = next(
            (
                candidate
                for candidate in reversed(context.window_manager.operators)
                if getattr(candidate, "bl_idname", "").upper() == expected_identifier
            ),
            None,
        )
    if operator is not None:
        session.native_operator_pointer = int(operator.as_pointer())
    record = _HISTORY_RECORDS.get(session.history_token)
    if record is not None:
        record.session.native_operator_pointer = session.native_operator_pointer

    if profile is None or not profile.supports_nested_offset:
        return
    nested = getattr(operator, "MESH_OT_offset_edge_loops", None)
    if nested is not None:
        session.offset_use_cap_endpoint = bool(getattr(nested, "use_cap_endpoint", False))
    if record is not None:
        record.session.offset_use_cap_endpoint = session.offset_use_cap_endpoint


def _invoke_finish_operator() -> set[str]:
    finish_operator = getattr(bpy.ops.mesh, "ydd_symmetric_edit_finish")
    return cast(set[str], finish_operator("EXEC_DEFAULT"))


def _watch_passthrough_session(window_pointer: int, history_token: int):
    session = _SESSIONS.get(window_pointer)
    if session is None or session.history_token != history_token:
        return None

    window = _find_window(window_pointer)
    if window is None:
        cleanup_session(window_pointer)
        return None

    if _native_tool_is_active(window, session.tool_kind):
        session.saw_modal = True
        session.modal_absent_since = None
        session.path_signature = None
        session.stable_path_ticks = 0
        return _PASSTHROUGH_POLL_INTERVAL

    now = time.monotonic()
    if not session.saw_modal and not _session_has_new_path(session):
        if now - session.started_at < _PASSTHROUGH_START_GRACE:
            return _PASSTHROUGH_POLL_INTERVAL
        cleanup_session(window_pointer)
        return None

    # Loop Cut and Offset are native macros.  Their topology child and Edge
    # Slide child can leave a brief interval with no relevant modal operator.
    # Require both a quiet interval and an unchanged result before postprocess.
    path_signature = _session_new_path_signature(session)
    if session.modal_absent_since is None:
        session.modal_absent_since = now
        session.path_signature = path_signature
        session.stable_path_ticks = 1
        return _PASSTHROUGH_POLL_INTERVAL
    if path_signature == session.path_signature:
        session.stable_path_ticks += 1
    else:
        session.modal_absent_since = now
        session.path_signature = path_signature
        session.stable_path_ticks = 1
    handoff_grace = _PASSTHROUGH_HANDOFF_GRACE.get(session.tool_kind, 0.04)
    stable_ticks = _PASSTHROUGH_STABLE_TICKS.get(session.tool_kind, 3)
    if now - session.modal_absent_since < handoff_grace or session.stable_path_ticks < stable_ticks:
        return _PASSTHROUGH_POLL_INTERVAL

    if path_signature is None:
        # Knife/Loop Cut cancellation and confirmed no-ops leave no topology.
        cleanup_session(window_pointer)
        return None

    record = _HISTORY_RECORDS.get(history_token)
    _capture_projection_view(session)
    window, area, region = _find_saved_view(session)
    if window is None or area is None or region is None:
        if record is not None:
            record.status = "FAILED"
        cleanup_session(window_pointer, keep_history_record=True)
        return None

    try:
        with bpy.context.temp_override(window=window, area=area, region=region):
            _capture_native_result_options(session, bpy.context)
            result = _invoke_finish_operator()
    except Exception:
        traceback.print_exc()
        result = {"CANCELLED"}
        cleanup_session(window_pointer, keep_history_record=True)

    if record is not None:
        record.status = "COMMITTED" if "FINISHED" in result else "FAILED"
    return None


def _schedule_passthrough_watcher(
    window_pointer: int,
    history_token: int,
) -> None:
    def callback():
        return _watch_passthrough_session(window_pointer, history_token)

    bpy.app.timers.register(callback, first_interval=0.02)


def _read_history_tokens(obj) -> set[int]:
    """Read raw history tokens; read failures propagate to the caller."""

    if obj.type != "MESH":
        return set()
    if obj.mode == "EDIT":
        bm = bmesh.from_edit_mesh(obj.data)
        layer = bm.faces.layers.int.get(core.HISTORY_TOKEN_LAYER)
        if layer is None:
            return set()
        return {int(face[layer]) for face in bm.faces if int(face[layer])}
    attribute = obj.data.attributes.get(core.HISTORY_TOKEN_LAYER)
    if attribute is None:
        return set()
    return {int(item.value) for item in attribute.data if int(item.value)}


def _object_history_tokens(obj) -> set[int]:
    try:
        return _read_history_tokens(obj)
    except (AttributeError, ReferenceError, RuntimeError):
        return set()


def _object_has_temporary_layers(obj) -> bool:
    if obj.type != "MESH":
        return False
    try:
        if obj.mode == "EDIT":
            bm = bmesh.from_edit_mesh(obj.data)
            return any(
                layers.get(name) is not None
                for layers, name in (
                    (bm.edges.layers.int, core.EDGE_ORIGINAL_LAYER),
                    (bm.faces.layers.int, core.FACE_ID_LAYER),
                    (bm.faces.layers.int, core.HISTORY_TOKEN_LAYER),
                )
            )
        return any(obj.data.attributes.get(name) is not None for name in core.TEMP_LAYER_NAMES)
    except (ReferenceError, RuntimeError):
        return False


def _restore_session_face_maps(session: KnifeSession, obj) -> bool:
    if obj.mode != "EDIT":
        return False
    try:
        bm = bmesh.from_edit_mesh(obj.data)
        face_id_layer = bm.faces.layers.int.get(core.FACE_ID_LAYER)
        mirror_id_layer = bm.faces.layers.int.get(core.FACE_MIRROR_ID_LAYER)
        hidden_layer = bm.faces.layers.int.get(core.FACE_HIDDEN_LAYER)
        if face_id_layer is None or mirror_id_layer is None or hidden_layer is None:
            return False

        mirror_face_ids: MirrorFaceMap = {}
        hidden_by_face_id: HiddenFaceMap = {}
        for face in bm.faces:
            face_id = FaceId(int(face[face_id_layer]))
            if face_id <= 0:
                continue
            raw_mirror_id = int(face[mirror_id_layer])
            if raw_mirror_id > 0:
                mirror_face_ids.setdefault(face_id, FaceId(raw_mirror_id))
            hidden_by_face_id.setdefault(face_id, bool(face[hidden_layer]))
        session.mirror_face_ids = mirror_face_ids
        session.hidden_by_face_id = hidden_by_face_id
        return bool(mirror_face_ids)
    except (ReferenceError, RuntimeError):
        return False


def _cleanup_object_temporary_layers(obj) -> None:
    try:
        if obj.mode == "EDIT":
            bm = bmesh.from_edit_mesh(obj.data)
            if core.remove_temporary_layers(bm):
                bmesh.update_edit_mesh(
                    obj.data,
                    loop_triangles=False,
                    destructive=False,
                )
        else:
            core.remove_temporary_mesh_attributes(obj.data)
    except (ReferenceError, RuntimeError):
        pass


def _history_marker_objects():
    result = []
    seen_meshes = set()
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        mesh = cast(bpy.types.Mesh, obj.data)
        if mesh.as_pointer() in seen_meshes:
            continue
        if not _object_has_temporary_layers(obj):
            continue
        seen_meshes.add(mesh.as_pointer())
        result.append((obj, _object_history_tokens(obj)))
    return result


def _repair_history_state():
    global _HISTORY_REPAIR_BUSY, _HISTORY_REPAIR_QUEUED

    _HISTORY_REPAIR_QUEUED = False
    if _HISTORY_REPAIR_BUSY:
        return None

    marker_objects = _history_marker_objects()
    if not marker_objects:
        return None

    live_tokens = {session.history_token for session in _SESSIONS.values() if session.history_token}
    live_mesh_names = {session.mesh_name for session in _SESSIONS.values()}
    repairable_marker_objects = [
        (obj, tokens)
        for obj, tokens in marker_objects
        if not tokens.intersection(live_tokens) and obj.data.name not in live_mesh_names
    ]
    if not repairable_marker_objects:
        return None

    candidates = []
    for obj, tokens in repairable_marker_objects:
        # Stale debris tokens may coexist with the repairable one; only a
        # unique COMMITTED token is authoritative.  Finish removes every
        # temporary layer, so foreign leftovers are swept by the repair itself.
        committed_tokens = [
            (token, _HISTORY_RECORDS[token])
            for token in tokens
            if token in _HISTORY_RECORDS and _HISTORY_RECORDS[token].status == "COMMITTED"
        ]
        if len(committed_tokens) != 1:
            continue
        token, record = committed_tokens[0]
        candidates.append((obj, token, record))

    # A native Knife Undo/Redo exposes exactly one recognized raw snapshot.
    # Ambiguous or unknown marker states are cleaned rather than projected.
    if len(candidates) != 1:
        for obj, _tokens in repairable_marker_objects:
            _cleanup_object_temporary_layers(obj)
        return None

    obj, token, record = candidates[0]
    if obj.mode != "EDIT" or bpy.context.edit_object is not obj:
        _cleanup_object_temporary_layers(obj)
        return None

    session = copy.deepcopy(record.session)
    session.object_name = obj.name
    session.mesh_name = obj.data.name
    profile = TOOL_PROFILES.get(session.tool_kind)
    if profile is not None and profile.supports_nested_offset:
        current_symmetry = SymmetryAxes(
            x=bool(obj.use_mesh_mirror_x),
            y=bool(obj.use_mesh_mirror_y),
            z=bool(obj.use_mesh_mirror_z),
        )
        # A raw Offset undo snapshot can contain the deliberately suspended
        # flags. Treat that state like the tail of the original modal operation
        # so finish restores the exact native setting captured by the session.
        session.symmetry_suspended = current_symmetry != session.symmetry_flags
    if not _restore_session_face_maps(session, obj):
        record.status = "FAILED"
        _cleanup_repair_session(session)
        return None
    window, area, region = _find_saved_view(session)
    if window is None or area is None or region is None:
        record.status = "FAILED"
        _cleanup_repair_session(session)
        return None

    # A live session may have started in this window between the undo_post
    # queueing and this timer tick.  Storing the repair session would silently
    # replace that entry, killing its watcher and leaking its suspended
    # symmetry flags.  Leave everything untouched; the record stays COMMITTED
    # so the next undo_post can retry the repair.
    if session.window_pointer in _SESSIONS:
        return None

    _HISTORY_REPAIR_BUSY = True
    try:
        _SESSIONS[session.window_pointer] = session
        with bpy.context.temp_override(window=window, area=area, region=region):
            result = _invoke_finish_operator()
        if "FINISHED" not in result:
            record.status = "FAILED"
            _cleanup_repair_session(session)
    except Exception:
        traceback.print_exc()
        record.status = "FAILED"
        _cleanup_repair_session(session)
    finally:
        _HISTORY_REPAIR_BUSY = False
    _HISTORY_RECORDS.move_to_end(token)
    return None


def _queue_history_repair(_dummy=None) -> None:
    global _HISTORY_REPAIR_QUEUED

    if _HISTORY_REPAIR_BUSY or _HISTORY_REPAIR_QUEUED:
        return
    _HISTORY_REPAIR_QUEUED = True
    if not bpy.app.timers.is_registered(_repair_history_state):
        bpy.app.timers.register(_repair_history_state, first_interval=0.01)


@persistent
def repair_after_undo(_dummy=None) -> None:
    if _prepare_adjust_last_operation_repeat():
        return
    _queue_history_repair()


@persistent
def repair_after_redo(_dummy=None) -> None:
    _queue_history_repair()


def _prepare_adjust_last_operation_repeat() -> bool:
    """Prepare the native re-execution phase of Adjust Last Operation.

    Blender implements an F9 property change as Undo followed by a direct
    re-execution of the registered native operator.  That second phase does not
    traverse a keymap, so its ``undo_post`` is the only point where the baseline
    can be marked without replacing Blender's own operator or redo panel.
    """

    context = bpy.context
    active_operator = context.active_operator
    if active_operator is None:
        return False
    identifier = getattr(active_operator, "bl_idname", "").upper()
    tool_kind = _WM_OPERATOR_TO_TOOL.get(identifier)
    if tool_kind is None or not _single_edit_mesh_poll(context):
        return False

    obj = context.edit_object
    if obj is None or obj.type != "MESH" or not isinstance(obj.data, bpy.types.Mesh):
        return False
    window_pointer = _window_key(context)
    prior_record = next(
        (
            record
            for record in reversed(tuple(_HISTORY_RECORDS.values()))
            if record.status == "COMMITTED"
            and record.session.tool_kind == tool_kind
            and record.session.window_pointer == window_pointer
            and record.session.object_name == obj.name
            and record.session.mesh_name == obj.data.name
            and record.session.native_operator_pointer == int(active_operator.as_pointer())
        ),
        None,
    )
    if prior_record is None:
        return False
    # Token presence alone cannot separate a raw snapshot from a baseline
    # poisoned by lazy undo encoding; only path-edge evidence can.  Read
    # failures and indeterminate layer states must stay on the repair side.
    try:
        tokens = _read_history_tokens(obj)
    except Exception:
        return False
    if tokens:
        try:
            bm = bmesh.from_edit_mesh(obj.data)
            path_state = core.native_path_edge_state(bm)
        except Exception:
            return False
        if path_state != "ABSENT":
            return False
        _queue_history_repair()

    profile = TOOL_PROFILES.get(tool_kind)
    if profile is not None and profile.supports_nested_offset:
        for attribute, enabled in zip(
            ("use_mesh_mirror_x", "use_mesh_mirror_y", "use_mesh_mirror_z"),
            prior_record.session.symmetry_flags.as_tuple(),
            strict=True,
        ):
            setattr(obj, attribute, enabled)

    try:
        return _prepare_session(
            context,
            lambda _levels, _message: None,
            tool_kind=tool_kind,
        )
    except Exception:
        traceback.print_exc()
        cleanup_session(window_pointer)
        return False


def register_history_handlers() -> None:
    for handlers, callback in (
        (bpy.app.handlers.undo_post, repair_after_undo),
        (bpy.app.handlers.redo_post, repair_after_redo),
    ):
        if callback not in handlers:
            handlers.append(callback)


def unregister_history_handlers() -> None:
    global _HISTORY_REPAIR_QUEUED

    for handlers, callback in (
        (bpy.app.handlers.undo_post, repair_after_undo),
        (bpy.app.handlers.redo_post, repair_after_redo),
    ):
        if callback in handlers:
            handlers.remove(callback)
    if bpy.app.timers.is_registered(_repair_history_state):
        bpy.app.timers.unregister(_repair_history_state)
    _HISTORY_REPAIR_QUEUED = False
    clear_history_records()


CLASSES = (
    MESH_OT_ydd_symmetric_edit_intercept,
    MESH_OT_ydd_symmetric_edit_finish,
)
