from __future__ import annotations

import time
import traceback
from dataclasses import dataclass

import bmesh
import bpy

from . import extrude, matching, rip, session_state, snapshot
from ._types import (
    KnifeSession,
    MeshSelectionMode,
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
    "EXTRUDE_NORMAL": ToolProfile(
        kind="EXTRUDE_NORMAL",
        label="Extrude",
        wm_operator_names=("MESH_OT_extrude_region_move",),
        primary_wm_operator="MESH_OT_extrude_region_move",
        tool_idnames=(),
        keymap_operator="view3d.edit_mesh_extrude_move_normal",
        passthrough_handoff_grace=0.04,
        passthrough_stable_ticks=3,
        supports_nested_offset=False,
        supports_adjust_repeat=False,
    ),
    "EXTRUDE_CONTEXT": ToolProfile(
        kind="EXTRUDE_CONTEXT",
        label="Extrude Region",
        wm_operator_names=("MESH_OT_extrude_context_move",),
        primary_wm_operator="MESH_OT_extrude_context_move",
        tool_idnames=("3D View Tool: Edit Mesh, Extrude Region",),
        keymap_operator="mesh.extrude_context_move",
        passthrough_handoff_grace=0.04,
        passthrough_stable_ticks=3,
        supports_nested_offset=False,
        supports_adjust_repeat=False,
    ),
    "EXTRUDE_SHRINK_FATTEN": ToolProfile(
        kind="EXTRUDE_SHRINK_FATTEN",
        label="Extrude Along Normals",
        wm_operator_names=("MESH_OT_extrude_region_shrink_fatten",),
        primary_wm_operator="MESH_OT_extrude_region_shrink_fatten",
        tool_idnames=("3D View Tool: Edit Mesh, Extrude Along Normals",),
        keymap_operator="mesh.extrude_region_shrink_fatten",
        passthrough_handoff_grace=0.04,
        passthrough_stable_ticks=3,
        supports_nested_offset=False,
        supports_adjust_repeat=False,
    ),
    "EXTRUDE_FACES_INDIV": ToolProfile(
        kind="EXTRUDE_FACES_INDIV",
        label="Extrude Individual",
        wm_operator_names=("MESH_OT_extrude_faces_move",),
        primary_wm_operator="MESH_OT_extrude_faces_move",
        tool_idnames=("3D View Tool: Edit Mesh, Extrude Individual",),
        keymap_operator="mesh.extrude_faces_move",
        passthrough_handoff_grace=0.04,
        passthrough_stable_ticks=3,
        supports_nested_offset=False,
        supports_adjust_repeat=False,
    ),
    "EXTRUDE_EDGES_INDIV": ToolProfile(
        kind="EXTRUDE_EDGES_INDIV",
        label="Extrude Edges",
        wm_operator_names=("MESH_OT_extrude_edges_move",),
        primary_wm_operator="MESH_OT_extrude_edges_move",
        tool_idnames=(),
        keymap_operator="mesh.extrude_edges_move",
        passthrough_handoff_grace=0.04,
        passthrough_stable_ticks=3,
        supports_nested_offset=False,
        supports_adjust_repeat=False,
    ),
    "EXTRUDE_VERTS_INDIV": ToolProfile(
        kind="EXTRUDE_VERTS_INDIV",
        label="Extrude Vertices",
        wm_operator_names=("MESH_OT_extrude_vertices_move",),
        primary_wm_operator="MESH_OT_extrude_vertices_move",
        tool_idnames=(),
        keymap_operator="mesh.extrude_vertices_move",
        passthrough_handoff_grace=0.04,
        passthrough_stable_ticks=3,
        supports_nested_offset=False,
        supports_adjust_repeat=False,
    ),
    "EXTRUDE_MANIFOLD": ToolProfile(
        kind="EXTRUDE_MANIFOLD",
        label="Extrude Manifold",
        wm_operator_names=("MESH_OT_extrude_manifold",),
        primary_wm_operator="MESH_OT_extrude_manifold",
        tool_idnames=("3D View Tool: Edit Mesh, Extrude Manifold",),
        keymap_operator="mesh.extrude_manifold",
        passthrough_handoff_grace=0.04,
        passthrough_stable_ticks=3,
        supports_nested_offset=False,
        supports_adjust_repeat=False,
    ),
}
EXTRUDE_TOOL_KINDS = frozenset(
    {
        "EXTRUDE_NORMAL",
        "EXTRUDE_CONTEXT",
        "EXTRUDE_SHRINK_FATTEN",
        "EXTRUDE_FACES_INDIV",
        "EXTRUDE_EDGES_INDIV",
        "EXTRUDE_VERTS_INDIV",
        "EXTRUDE_MANIFOLD",
    }
)
TOOL_LABELS = {profile.kind: profile.label for profile in TOOL_PROFILES.values()}
MODAL_IDENTIFIER_TOKENS = {profile.kind: profile.wm_operator_names for profile in TOOL_PROFILES.values()}
_PASSTHROUGH_HANDOFF_GRACE = {profile.kind: profile.passthrough_handoff_grace for profile in TOOL_PROFILES.values()}
_PASSTHROUGH_STABLE_TICKS = {profile.kind: profile.passthrough_stable_ticks for profile in TOOL_PROFILES.values()}
_WM_OPERATOR_TO_TOOL = {
    profile.primary_wm_operator: profile.kind for profile in TOOL_PROFILES.values() if profile.supports_adjust_repeat
}


class SymmetricKnifeError(RuntimeError):
    """Expected validation or Blender-context failure shown in the UI."""


def _window_key(context: WindowContext) -> int:
    return context.window.as_pointer() if context.window is not None else 0


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
            try:
                snapshot.remove_temporary_layers(bm)
            except Exception:
                traceback.print_exc()
            finally:
                try:
                    bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
                except Exception:
                    traceback.print_exc()
        else:
            snapshot.remove_temporary_mesh_attributes(obj.data)
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
        session_state._HISTORY_RECORDS.pop(session.history_token, None)


def cleanup_session(
    window_pointer: int,
    *,
    keep_history_record: bool = False,
) -> None:
    session = session_state._SESSIONS.pop(window_pointer, None)
    if session is not None:
        _teardown_session_state(
            session,
            keep_history_record=keep_history_record,
        )


def _cleanup_repair_session(session: KnifeSession) -> None:
    """Tear down a history-repair session whether or not finish detached it."""

    active = session_state._SESSIONS.get(session.window_pointer)
    if active is not None and active.history_token == session.history_token:
        cleanup_session(
            session.window_pointer,
            keep_history_record=True,
        )
        return
    _teardown_session_state(session, keep_history_record=True)


def cleanup_all_sessions() -> None:
    for window_pointer in tuple(session_state._SESSIONS):
        cleanup_session(window_pointer)


def _single_edit_mesh_poll(context) -> bool:
    obj = context.edit_object
    if context.area is None or context.area.type != "VIEW_3D":
        return False
    if context.region is None or context.region.type != "WINDOW":
        return False
    if context.mode != "EDIT_MESH" or obj is None or obj.type != "MESH":
        return False
    return len(context.objects_in_mode_unique_data) == 1


def gizmo_session_is_modal(context) -> bool:
    from . import watcher

    obj = context.edit_object
    if obj is None or obj.type != "MESH":
        return False
    adopted = next(
        (
            active
            for active in session_state._SESSIONS.values()
            if active.mesh_name == obj.data.name and active.route == "GIZMO_ADOPTED"
        ),
        None,
    )
    if adopted is None:
        return False
    window = _find_window(adopted.window_pointer)
    return window is not None and watcher._native_tool_is_active(window, adopted.tool_kind)


def _prepare_session(
    context,
    report,
    *,
    tool_kind: str = "KNIFE",
    route_kmi_properties: tuple[tuple[str, object], ...] = (),
) -> bool:
    from .history import _remember_history_session
    from .watcher import _schedule_passthrough_watcher

    if tool_kind not in TOOL_LABELS:
        raise ValueError(f"Unsupported native tool kind: {tool_kind!r}")
    settings = context.scene.ydd_symmetric_edit
    obj = context.edit_object
    window_pointer = _window_key(context)

    adopted_session = next(
        (
            active
            for active in session_state._SESSIONS.values()
            if active.mesh_name == obj.data.name and active.route == "GIZMO_ADOPTED"
        ),
        None,
    )
    if adopted_session is not None:
        from . import operators, watcher

        adopted_window = _find_window(adopted_session.window_pointer)
        if adopted_window is not None and watcher._native_tool_is_active(adopted_window, adopted_session.tool_kind):
            return False
        if not watcher._capture_confirmed_extrude_result(adopted_session):
            return False
        saved_window, saved_area, saved_region = _find_saved_view(adopted_session)
        if saved_window is None or saved_area is None or saved_region is None:
            return False
        try:
            with bpy.context.temp_override(window=saved_window, area=saved_area, region=saved_region):
                result = operators._invoke_finish_operator()
        except Exception:
            traceback.print_exc()
            return False
        record = session_state._HISTORY_RECORDS.get(adopted_session.history_token)
        if record is not None:
            record.status = "COMMITTED" if "FINISHED" in result else "FAILED"
        if "FINISHED" not in result or any(
            active.mesh_name == obj.data.name for active in session_state._SESSIONS.values()
        ):
            return False

    conflicting_session = next(
        (
            session
            for pointer, session in session_state._SESSIONS.items()
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
    snapshot.remove_temporary_layers(bm)

    enabled_axes = matching.enabled_mesh_symmetry_axes(obj)
    if len(enabled_axes) != 1:
        snapshot.remove_temporary_layers(bm)
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
    if tool_kind in EXTRUDE_TOOL_KINDS and len(bm.faces) == 0:
        snapshot.remove_temporary_layers(bm)
        bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
        return False

    history_token = session_state._new_history_token()
    registered = False
    layers_may_exist = False
    try:
        layers_may_exist = True
        topology = snapshot.prepare_topology(
            bm,
            axis_index,
            settings.tolerance,
            history_token,
            mark_vertex_ids=tool_kind == "RIP" or tool_kind in EXTRUDE_TOOL_KINDS,
            mesh_object=obj,
        )

        rip_snapshot = None
        extrude_snapshot = None
        prepare_disposition = "APPLY"
        prepare_disposition_reason = ""
        if tool_kind == "RIP":
            # The bulk capture no longer refreshes BMesh indices, but the rip
            # snapshot keys its region and one-ring by vertex.index.
            bm.verts.ensure_lookup_table()
            bm.verts.index_update()
            rip_snapshot = rip.build_snapshot(
                bm,
                axis_index,
                settings.tolerance,
                lookup=topology.topology_resolution.vertex_lookup_unresolved,
            )
            if rip_snapshot is None:
                snapshot.remove_temporary_layers(bm)
                bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
                return False
        elif tool_kind in EXTRUDE_TOOL_KINDS:
            extrude.stamp_all_vertex_ids(bm)
            extrude_snapshot = extrude.build_snapshot(
                bm,
                axis_index,
                settings.tolerance,
                tool_kind=tool_kind,
                route_kmi_properties=route_kmi_properties,
                mesh_select_mode=MeshSelectionMode(
                    vertices=bool(context.tool_settings.mesh_select_mode[0]),
                    edges=bool(context.tool_settings.mesh_select_mode[1]),
                    faces=bool(context.tool_settings.mesh_select_mode[2]),
                ),
                mesh_object=obj,
            )
            if extrude_snapshot is None:
                snapshot.remove_temporary_layers(bm)
                bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
                return False
            prepare_disposition, prepare_disposition_reason = extrude.evaluate_prepare_gates(extrude_snapshot)
            if prepare_disposition == "DECLINE" and prepare_disposition_reason:
                report(
                    {"WARNING"},
                    f"Extrude was not mirrored: {prepare_disposition_reason} (native kept; mirror manually or undo)",
                )

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
            mirror_face_ids={},
            hidden_by_face_id=topology.hidden_by_face_id,
            carrier_frames={},
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
            topology_resolution=topology.topology_resolution,
            extrude=extrude_snapshot,
            prepare_disposition=prepare_disposition,
            prepare_disposition_reason=prepare_disposition_reason,
        )
        session_state._SESSIONS[window_pointer] = session
        registered = True
        _suspend_mesh_symmetry(session, obj)
        _remember_history_session(session, context)
        _schedule_passthrough_watcher(window_pointer, history_token)
        return True
    except Exception:
        if registered:
            cleanup_session(window_pointer)
        if layers_may_exist:
            try:
                snapshot.remove_temporary_layers(bm)
                bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
            except Exception:
                traceback.print_exc()
        raise
