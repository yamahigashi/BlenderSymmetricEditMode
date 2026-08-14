# ruff: noqa: F401
from __future__ import annotations

import traceback
from typing import TYPE_CHECKING, cast

import bmesh
import bpy
from bpy.app.handlers import persistent
from bpy.props import BoolProperty, StringProperty

from . import (
    backup,
    extrude,
    face_mapping,
    layer_names,
    rip,
    selection,
    session_state,
    snapshot,
    stitch_common,
    stitch_crossings,
    stitch_offset,
    stitch_pathedges,
    stitch_pstitch,
    stitch_reflect,
)
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
    WindowContext,
)
from .gc_gate import gc_disabled_during_execute
from .history import (
    _cleanup_object_temporary_layers,
    _history_marker_objects,
    _object_has_temporary_layers,
    _object_history_tokens,
    _prepare_adjust_last_operation_repeat,
    _queue_history_repair,
    _read_history_tokens,
    _remember_history_session,
    _repair_history_state,
    _restore_session_face_maps,
    clear_history_records,
    register_history_handlers,
    repair_after_redo,
    repair_after_undo,
    unregister_history_handlers,
)  # noqa: F401
from .session import (
    _PASSTHROUGH_HANDOFF_GRACE,
    _PASSTHROUGH_STABLE_TICKS,
    _WM_OPERATOR_TO_TOOL,
    EXTRUDE_TOOL_KINDS,
    MODAL_IDENTIFIER_TOKENS,
    TOOL_LABELS,
    TOOL_PROFILES,
    SymmetricKnifeError,
    ToolProfile,
    _cleanup_object_layers,
    _cleanup_repair_session,
    _prepare_session,
    _restore_mesh_symmetry,
    _single_edit_mesh_poll,
    _suspend_mesh_symmetry,
    _teardown_session_state,
    _window_key,
    cleanup_all_sessions,
    cleanup_session,
)  # noqa: F401
from .watcher import (
    _capture_native_result_options,
    _modal_operator_identifiers,
    _native_tool_is_active,
    _schedule_passthrough_watcher,
    _session_has_new_path,
    _session_new_path_signature,
    _watch_passthrough_session,
)  # noqa: F401

_SESSION_STATE_EXPORTS = {
    "_FINISH_REPORTS",
    "_HISTORY_RECORDS",
    "_HISTORY_REPAIR_BUSY",
    "_HISTORY_REPAIR_QUEUED",
    "_HISTORY_SEQUENCE",
    "_MAX_HISTORY_RECORDS",
    "_NEXT_HISTORY_TOKEN",
    "_PASSTHROUGH_POLL_INTERVAL",
    "_PASSTHROUGH_START_GRACE",
    "_SESSIONS",
    "_new_history_token",
}


def __getattr__(name: str):
    if name in _SESSION_STATE_EXPORTS:
        return getattr(session_state, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | _SESSION_STATE_EXPORTS)


@persistent
def cleanup_stale_attributes(_dummy=None) -> None:
    """Save/load handler: temporary detection layers must never reach disk."""

    cleanup_all_sessions()
    for mesh in bpy.data.meshes:
        try:
            snapshot.remove_temporary_mesh_attributes(mesh)
        except RuntimeError:
            pass


@persistent
def cleanup_after_load(_dummy=None) -> None:
    clear_history_records()
    cleanup_stale_attributes()


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
        if tool_kind in EXTRUDE_TOOL_KINDS and keymaps.live_route_has_dissolve_and_intersect(self.route_key):
            return {"PASS_THROUGH"}
        try:
            _prepare_session(
                context,
                self.report,
                tool_kind=tool_kind,
                route_kmi_properties=keymaps.route_kmi_properties(self.route_key),
            )
        except Exception:
            traceback.print_exc()
            cleanup_session(_window_key(context))
        # Do not invoke the native operator here. Passing through the original
        # physical event preserves toolbar first-click/drag behavior and all
        # native operator properties.
        return {"PASS_THROUGH"}


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
                snapshot.remove_temporary_layers(bm)
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


def _write_session_disposition(session: KnifeSession, disposition: str, reason: str) -> None:
    session.prepare_disposition = disposition
    session.prepare_disposition_reason = reason
    record = session_state._HISTORY_RECORDS.get(session.history_token)
    if record is not None:
        record.session.prepare_disposition = disposition
        record.session.prepare_disposition_reason = reason


def _write_extrude_freeze(session: KnifeSession, freeze) -> None:
    session.extrude_freeze = freeze
    record = session_state._HISTORY_RECORDS.get(session.history_token)
    if record is not None:
        record.session.extrude_freeze = freeze


def _finish_extrude_session(
    operator: bpy.types.Operator,
    session: KnifeSession,
    obj,
    window_pointer: int,
    context,
) -> OperatorResult:
    """Mirror a confirmed native region extrude. All-or-nothing, four states."""

    backup_mesh = None
    reason: str | None = None
    backup_creation_failed = False
    rollback_failed = False
    applied = False
    mutated = False
    selection_state = None
    write_decline = False
    mesh_select_mode = session.mesh_select_mode

    try:
        if session.extrude is None:
            reason = "the pre-extrude snapshot was lost"
            write_decline = True
        else:
            intervening = extrude.intervening_operator_reason(session, context)
            if intervening is not None:
                reason = intervening
                write_decline = True
            elif session.prepare_disposition == "DECLINE":
                reason = session.prepare_disposition_reason or "the extrusion selection cannot be mirrored"
            elif not session.extrude_options_captured:
                reason = "native extrude options could not be captured"
            else:
                bm = bmesh.from_edit_mesh(obj.data)
                if session.extrude_freeze is not None:
                    classified, classify_reason = extrude.reconnect_freeze(
                        bm,
                        session.extrude,
                        session.extrude_freeze,
                    )
                else:
                    classified, classify_reason = extrude.classify_live(bm, session.extrude)
                if classified is None:
                    reason = classify_reason or "the native extrude result could not be classified"
                    write_decline = True
                else:
                    if session.extrude_freeze is None:
                        _write_extrude_freeze(session, classified.freeze)
                    description, describe_reason = extrude.describe_source(bm, session.extrude, classified)
                    if description is None:
                        reason = describe_reason or "the native extrude could not be described"
                        write_decline = True
                    else:
                        selection_state = selection.add_selection_layers(bm)
                        selection.snapshot_live_hidden(bm)
                        try:
                            backup_mesh = backup.create_topology_backup(bm)
                        except Exception as exc:
                            traceback.print_exc()
                            backup_creation_failed = True
                            reason = "backup_creation_failed"
                            del exc
                        if backup_mesh is not None:
                            bm = bmesh.from_edit_mesh(obj.data)
                            if session.extrude_freeze is not None:
                                classified, classify_reason = extrude.reconnect_freeze(
                                    bm,
                                    session.extrude,
                                    session.extrude_freeze,
                                )
                            else:
                                classified, classify_reason = extrude.classify_live(bm, session.extrude)
                            if classified is None or classify_reason is not None:
                                reason = classify_reason or "classification was lost after backup"
                                write_decline = True
                            else:
                                description, describe_reason = extrude.describe_source(
                                    bm,
                                    session.extrude,
                                    classified,
                                )
                                if description is None:
                                    reason = describe_reason or "source description was lost after backup"
                                    write_decline = True
                                else:
                                    stationarity_reason = extrude.check_origin_stationarity(
                                        bm,
                                        session.extrude,
                                        classified,
                                    )
                                    if stationarity_reason is not None:
                                        reason = stationarity_reason
                                        write_decline = True
                                    else:
                                        source_copy_coords = {
                                            vertex_id: (
                                                float(copy.co.x),
                                                float(copy.co.y),
                                                float(copy.co.z),
                                            )
                                            for vertex_id, copy in classified.copies.items()
                                        }
                                        source_origin_coords = {
                                            vertex_id: (
                                                float(origin.co.x),
                                                float(origin.co.y),
                                                float(origin.co.z),
                                            )
                                            for vertex_id, origin in classified.origins.items()
                                        }
                                        mirror_copies, apply_reason, apply_audit = extrude.apply_mirror(
                                            bm,
                                            session.extrude,
                                            classified,
                                            description,
                                        )
                                        mutated = True
                                        if apply_reason is not None or apply_audit is None:
                                            reason = apply_reason or "the mirrored extrude could not be applied"
                                            write_decline = True
                                        else:
                                            verify_reason = extrude.verify_mirror(
                                                bm,
                                                session.extrude,
                                                classified,
                                                description,
                                                mirror_copies,
                                                source_copy_coords,
                                                source_origin_coords,
                                                apply_audit,
                                            )
                                            if verify_reason is not None:
                                                reason = verify_reason
                                                write_decline = True
                                            else:
                                                applied = True
                                                bmesh.update_edit_mesh(
                                                    obj.data,
                                                    loop_triangles=True,
                                                    destructive=True,
                                                )
    except Exception as exc:
        traceback.print_exc()
        reason = str(exc)
        write_decline = True
        if backup_mesh is not None and mutated:
            try:
                backup.restore_topology_backup(obj.data, backup_mesh)
                applied = False
                mutated = False
            except Exception:
                traceback.print_exc()
                rollback_failed = True
                reason = f"Extrude mirror failed and rollback failed: {exc}"
    finally:
        try:
            if backup_mesh is not None and mutated and not applied:
                try:
                    backup.restore_topology_backup(obj.data, backup_mesh)
                    mutated = False
                except Exception:
                    traceback.print_exc()
                    rollback_failed = True
                    reason = f"Extrude mirror failed and rollback failed: {reason}"
                    _finish_report(
                        operator,
                        {"WARNING"},
                        "ydd Symmetric Edit: extrude backup restore failed",
                    )
        finally:
            try:
                if obj is not None and obj.mode == "EDIT":
                    try:
                        bm = bmesh.from_edit_mesh(obj.data)
                        context.tool_settings.mesh_select_mode = mesh_select_mode.as_tuple()
                        if selection_state is not None:
                            selection.restore_visibility_and_selection(
                                bm,
                                session.hidden_by_face_id,
                                selection_state,
                                use_live_hidden=True,
                            )
                    except Exception:
                        traceback.print_exc()
                        _finish_report(
                            operator,
                            {"WARNING"},
                            "ydd Symmetric Edit: extrude visibility/selection restore failed",
                        )
            finally:
                try:
                    if obj is not None and obj.mode == "EDIT":
                        try:
                            bm = bmesh.from_edit_mesh(obj.data)
                            snapshot.remove_temporary_layers(bm)
                        except Exception:
                            traceback.print_exc()
                            _finish_report(
                                operator,
                                {"WARNING"},
                                "ydd Symmetric Edit: temporary extrude layers could not be removed",
                            )
                finally:
                    try:
                        if obj is not None and obj.mode == "EDIT":
                            try:
                                bmesh.update_edit_mesh(obj.data, loop_triangles=True, destructive=True)
                            except Exception:
                                traceback.print_exc()
                                _finish_report(
                                    operator,
                                    {"WARNING"},
                                    "ydd Symmetric Edit: extrude mesh update failed",
                                )
                    finally:
                        try:
                            backup.remove_backup(backup_mesh)
                        except Exception:
                            traceback.print_exc()
                            _finish_report(
                                operator,
                                {"WARNING"},
                                "ydd Symmetric Edit: extrude backup release failed",
                            )
                        finally:
                            try:
                                cleanup_session(window_pointer, keep_history_record=True)
                            except Exception:
                                traceback.print_exc()
                                _finish_report(
                                    operator,
                                    {"WARNING"},
                                    "ydd Symmetric Edit: extrude session cleanup failed",
                                )

    if write_decline and reason is not None and reason != "native extrude options could not be captured":
        _write_session_disposition(session, "DECLINE", reason)

    kept_hint = " (native kept; mirror manually or undo)"
    if reason is not None:
        if rollback_failed:
            _finish_report(
                operator,
                {"ERROR"},
                f"ydd Symmetric Edit: Extrude mirror failed and rollback failed: {reason}. Undo to recover.",
            )
        elif backup_creation_failed:
            _finish_report(
                operator,
                {"WARNING"},
                f"ydd Symmetric Edit: Extrude was not mirrored: {reason}{kept_hint}",
            )
        else:
            _finish_report(
                operator,
                {"WARNING"},
                f"ydd Symmetric Edit: Extrude was not mirrored: {reason}{kept_hint}",
            )
    else:
        _finish_report(operator, {"INFO"}, "Mirrored Extrude")
        _maybe_extend_selection_to_mirror(obj, session.axis_index, session.tolerance)
    return {"FINISHED"}


def _finish_report(operator, level: set[str], message: str) -> None:
    """Report and record for tests (``Operator.report`` itself is not patchable)."""

    kind = "WARNING" if "WARNING" in level else "ERROR" if "ERROR" in level else "INFO"
    session_state._FINISH_REPORTS.append((kind, message))
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
        selection.extend_selection_to_mirror(bm, axis_index, tolerance, mesh_object=obj)
        bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
    except Exception:
        traceback.print_exc()


class MESH_OT_ydd_symmetric_edit_finish(bpy.types.Operator):
    bl_idname = "mesh.ydd_symmetric_edit_finish"
    bl_label = "Apply Mirrored ydd Symmetric Edit Cut"
    bl_options = {"INTERNAL"}

    if TYPE_CHECKING:
        preserve_history_layers: bool
    else:
        preserve_history_layers: BoolProperty(options={"HIDDEN", "SKIP_SAVE"})

    @classmethod
    def poll(cls, context):
        return _single_edit_mesh_poll(context)

    @gc_disabled_during_execute
    def execute(self, context):
        session_state._FINISH_REPORTS.clear()
        window_pointer = _window_key(context)
        session = session_state._SESSIONS.get(window_pointer)
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
            if session.topology_resolution is not None:
                try:
                    resolution = session.topology_resolution
                    rip_face_scope = (
                        {FaceId(face_id) for vertex in session.rip.vertices for face_id in vertex.face_ids}
                        if session.rip is not None
                        else set()
                    )
                    resolution.resolve_faces(rip_face_scope)
                    session.mirror_face_ids = {
                        face_id: target
                        for face_id, target in resolution.scoped_mirror_face_ids.items()
                        if target is not None
                    }
                    session.carrier_frames = resolution.scoped_carrier_frames
                    bm = bmesh.from_edit_mesh(obj.data)
                    resolution.materialize_faces(bm, rip_face_scope)
                except Exception as exc:
                    traceback.print_exc()
                    record = session_state._HISTORY_RECORDS.get(session.history_token)
                    if record is not None:
                        record.status = "FAILED"
                    cleanup_session(window_pointer, keep_history_record=True)
                    _finish_report(self, {"ERROR"}, f"ydd Symmetric Edit resolution failed: {exc}")
                    return {"FINISHED"}
            return _finish_rip_session(self, session, obj, window_pointer)

        if session.tool_kind in EXTRUDE_TOOL_KINDS:
            return _finish_extrude_session(self, session, obj, window_pointer, context)

        backup_mesh = None
        mirror_committed = False
        result = {"CANCELLED"}
        crossing_count = 0
        selection_state = None
        restore_mutation_summaries = []
        direct_topology_success = False
        restore_summary_complete = True
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
            edge_layer, face_layer = snapshot.get_required_layers(bm)
            if edge_layer is None or face_layer is None:
                raise SymmetricKnifeError("Temporary topology markers are missing")

            materialized_face_scope: set[FaceId] = set()
            knife_path_edge_cache = None

            def _resolve_scope_overlay_and_materialize(edit_bm, path_edges):
                current_face_layer = edit_bm.faces.layers.int.get(layer_names.FACE_ID_LAYER)
                if current_face_layer is None:
                    raise SymmetricKnifeError("Temporary face markers are missing")
                scope = {
                    FaceId(int(face[current_face_layer]))
                    for edge in path_edges
                    if edge.is_valid
                    for face in edge.link_faces
                    if face.is_valid
                }
                resolution = session.topology_resolution
                if resolution is None:
                    lazy_face_map = session.mirror_face_ids
                else:
                    resolution.resolve_faces(scope)
                    lazy_face_map = resolution.scoped_mirror_face_ids
                    session.carrier_frames = resolution.scoped_carrier_frames
                overlay = face_mapping.resolve_live_mirror_face_map(
                    edit_bm,
                    lazy_face_map,
                    session.axis_index,
                    session.tolerance,
                    path_edges=path_edges,
                )
                if resolution is not None:
                    pending = scope - materialized_face_scope
                    if pending:
                        resolution.materialize_faces(edit_bm, pending)
                        materialized_face_scope.update(pending)
                return overlay

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
                def _all_path_edges(sides: dict) -> list:
                    return (
                        list(sides.get("POSITIVE", ()))
                        + list(sides.get("NEGATIVE", ()))
                        + list(sides.get("CROSSES", ()))
                        + list(sides.get("PLANE", ()))
                    )

                def _collect_knife_path_edges(edit_bm, *, topology_changed: bool = False):
                    nonlocal knife_path_edge_cache
                    if not topology_changed:
                        cached = stitch_pathedges.reclassify_knife_path_edge_cache(
                            edit_bm,
                            session.axis_index,
                            session.tolerance,
                            knife_path_edge_cache,
                        )
                        if cached is not None:
                            return cached
                    result = stitch_pathedges.collect_knife_path_edges_by_side(
                        edit_bm,
                        session.axis_index,
                        session.tolerance,
                    )
                    knife_path_edge_cache = stitch_pathedges.capture_knife_path_edge_cache(
                        edit_bm,
                        _all_path_edges(result[0]),
                    )
                    return result

                by_side, total_path_edges = _collect_knife_path_edges(bm, topology_changed=True)
                if total_path_edges == 0:
                    result = {"FINISHED"}
                    self.report({"INFO"}, f"{tool_label} made no new cut")
                    return result

                crossing_count = len(by_side["CROSSES"])
                live_mirror_face_ids = _resolve_scope_overlay_and_materialize(bm, _all_path_edges(by_side))
                _edge_layer, face_layer = snapshot.get_required_layers(bm)
                by_side, total_path_edges = _collect_knife_path_edges(bm)
                live_mirror_face_ids = _resolve_scope_overlay_and_materialize(bm, _all_path_edges(by_side))

                if crossing_count:
                    # Single backup covers p-stitch + mirror.
                    selection_state = selection.add_selection_layers(bm)
                    backup_mesh = _create_backup(bm)
                    bm = bmesh.from_edit_mesh(obj.data)
                    by_side, total_path_edges = _collect_knife_path_edges(bm)
                    live_mirror_face_ids = _resolve_scope_overlay_and_materialize(bm, _all_path_edges(by_side))
                    stitched, stitch_reason, stitch_summary = stitch_pstitch.apply_crosses_p_stitch(
                        bm,
                        by_side["CROSSES"],
                        session.axis_index,
                        session.tolerance,
                        return_summary=True,
                    )
                    restore_mutation_summaries.append(stitch_summary)
                    restore_summary_complete &= stitch_summary.complete
                    if stitch_reason:
                        raise SymmetricKnifeError(stitch_reason)
                    # Half-edges reclassify as POSITIVE/NEGATIVE after the split.
                    by_side, total_path_edges = _collect_knife_path_edges(bm, topology_changed=True)
                    live_mirror_face_ids = _resolve_scope_overlay_and_materialize(bm, _all_path_edges(by_side))
                    crossing_count = len(by_side["CROSSES"])
                    # Remaining CROSSES are self-mirrored (skipped by stitch) or
                    # still straddling after a failed reclassification — the
                    # latter is a bug; treat as decline via unmatched mirror.
                    if stitched:
                        # Persist p-stitch into the edit mesh so subsequent
                        # BMesh rebuilds (layer add / backup refresh) see it.
                        bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=True)

                        bm = bmesh.from_edit_mesh(obj.data)
                        _edge_layer, face_layer = snapshot.get_required_layers(bm)
                        by_side, total_path_edges = _collect_knife_path_edges(bm, topology_changed=True)
                        live_mirror_face_ids = _resolve_scope_overlay_and_materialize(bm, _all_path_edges(by_side))

                crossing_plan, crossing_reason = stitch_crossings.plan_mirrored_path_crossings(
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
                        selection_state = selection.add_selection_layers(bm)
                        backup_mesh = _create_backup(bm)
                        bm = bmesh.from_edit_mesh(obj.data)
                        _edge_layer, face_layer = snapshot.get_required_layers(bm)
                        by_side, total_path_edges = _collect_knife_path_edges(bm)
                        live_mirror_face_ids = _resolve_scope_overlay_and_materialize(bm, _all_path_edges(by_side))
                        crossing_plan, crossing_reason = stitch_crossings.plan_mirrored_path_crossings(
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

                    previous_crossing_by_side = {
                        side_name: list(by_side.get(side_name, ()))
                        for side_name in ("POSITIVE", "NEGATIVE", "CROSSES", "PLANE")
                    }
                    previous_crossing_edges = [
                        edge
                        for side_name in ("POSITIVE", "NEGATIVE", "CROSSES", "PLANE")
                        for edge in previous_crossing_by_side[side_name]
                    ]
                    crossing_cache_positions = None
                    if (
                        knife_path_edge_cache is not None
                        and len(knife_path_edge_cache) == len(previous_crossing_edges)
                        and len({hash(edge) for edge in previous_crossing_edges}) == len(previous_crossing_edges)
                        and len(set(knife_path_edge_cache)) == len(knife_path_edge_cache)
                    ):
                        crossing_cache_positions = {
                            hash(edge): position for position, edge in enumerate(previous_crossing_edges)
                        }
                    _crossings_stitched, crossing_reason, crossing_summary = (
                        stitch_crossings.apply_mirrored_path_crossings(
                            bm,
                            crossing_plan,
                            cache_positions=crossing_cache_positions,
                            return_summary=True,
                        )
                    )
                    restore_mutation_summaries.append(crossing_summary.selection_mutations)
                    restore_summary_complete &= crossing_summary.selection_mutations.complete
                    if crossing_reason:
                        raise SymmetricKnifeError(crossing_reason)
                    patched = stitch_pathedges.patch_knife_path_edges_by_side(
                        bm,
                        previous_crossing_by_side,
                        knife_path_edge_cache,
                        crossing_summary,
                        session.axis_index,
                        session.tolerance,
                    )
                    if patched is None:
                        by_side, total_path_edges = _collect_knife_path_edges(bm, topology_changed=True)
                    else:
                        by_side, total_path_edges = patched
                    live_mirror_face_ids = _resolve_scope_overlay_and_materialize(bm, _all_path_edges(by_side))
                    _edge_layer, face_layer = snapshot.get_required_layers(bm)

                source_edges = by_side["POSITIVE"] + by_side["NEGATIVE"]
                if not source_edges:
                    # Only PLANE and/or self-mirrored CROSSES: already symmetric.
                    if backup_mesh is not None:
                        # No further mutation; drop the unused backup cleanly.
                        mirror_committed = True
                    result = {"FINISHED"}
                    self.report(
                        {"INFO"},
                        f"{tool_label} cut lies on the mirror plane",
                    )
                    return result

                _target_face_ids, unmatched = stitch_pathedges.target_face_ids_for_edges(
                    source_edges,
                    face_layer,
                    live_mirror_face_ids,
                )
                if unmatched:
                    raise SymmetricKnifeError(f"{len(unmatched)} cut face(s) have no exact mirrored counterpart")

                use_direct_topology = stitch_reflect.reflected_path_uses_only_target_boundaries(
                    bm,
                    source_edges,
                    session.axis_index,
                    session.tolerance,
                    live_mirror_face_ids,
                    session.carrier_frames,
                )
                if not use_direct_topology:
                    raise SymmetricKnifeError("the mirrored cut cannot be rebuilt directly on the opposite side")

                if backup_mesh is None:
                    selection_state = selection.add_selection_layers(bm)
                    backup_mesh = _create_backup(bm)
                    # Layer creation invalidates the held BMesh proxies, so
                    # reclassify the unchanged path against a fresh view.
                    bm = bmesh.from_edit_mesh(obj.data)
                    by_side, total_path_edges = _collect_knife_path_edges(bm, topology_changed=True)
                    live_mirror_face_ids = _resolve_scope_overlay_and_materialize(bm, _all_path_edges(by_side))
                    _edge_layer, face_layer = snapshot.get_required_layers(bm)
                    source_edges = by_side["POSITIVE"] + by_side["NEGATIVE"]
                if not source_edges:
                    raise SymmetricKnifeError("The native cut path was lost before mirroring")

                created, already_present, direct_reason, reflected_summary = (
                    stitch_reflect.apply_reflected_path_topology(
                        bm,
                        source_edges,
                        session.axis_index,
                        session.tolerance,
                        live_mirror_face_ids,
                        session.carrier_frames,
                        return_summary=True,
                    )
                )
                if direct_reason:
                    raise SymmetricKnifeError(f"direct mirror declined: {direct_reason}")
                if created + already_present != len(source_edges):
                    reason = (
                        f"the mirrored {tool_label} topology did not match the source "
                        f"({created + already_present}/{len(source_edges)})"
                    )
                    raise SymmetricKnifeError(f"direct mirror declined: {reason}")

                restore_mutation_summaries.append(reflected_summary)
                restore_summary_complete &= reflected_summary.complete
                direct_topology_success = True
                mirror_committed = True
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

            else:
                # Loop Cut / Offset keep the one-side source selection.
                # Straddling rings symmetrize through the ordinary reflection
                # only because their carrier faces map to themselves or their
                # pairs; when pairing fails the counterpart check declines.
                source_edges, side, total_path_edges, crossing_count = stitch_pathedges.collect_source_path_edges(
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
                if stitch_pathedges.path_ring_includes_pre_hidden_edges(bm):
                    raise SymmetricKnifeError("the cut ring includes hidden edges; partial ring cuts are not mirrored")

                live_mirror_face_ids = _resolve_scope_overlay_and_materialize(bm, source_edges)
                _edge_layer, face_layer = snapshot.get_required_layers(bm)
                source_edges, side, total_path_edges, crossing_count = stitch_pathedges.collect_source_path_edges(
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
                live_mirror_face_ids = _resolve_scope_overlay_and_materialize(bm, source_edges)
                target_face_ids, unmatched = stitch_pathedges.target_face_ids_for_edges(
                    source_edges,
                    face_layer,
                    live_mirror_face_ids,
                )
                if unmatched:
                    raise SymmetricKnifeError(f"{len(unmatched)} cut face(s) have no exact mirrored counterpart")
                if not target_face_ids:
                    raise SymmetricKnifeError("The native cut path has no carrier faces")

                collapsed_target_markers = set()
                profile = TOOL_PROFILES.get(session.tool_kind)
                if profile is not None and profile.supports_nested_offset:
                    collapsed_target_markers, _collapsed_reason = stitch_offset.collapsed_offset_target_edge_markers(
                        bm,
                        source_edges,
                        session.axis_index,
                        session.tolerance,
                    )
                if collapsed_target_markers:
                    # Esc cancels only Offset's Edge Slide child; Blender keeps the
                    # two new loops at factor zero. Reproduce the same topology
                    # with BMesh.
                    selection_state = selection.add_selection_layers(bm)
                    backup_mesh = _create_backup(bm)
                    collapsed_segment_count, collapsed_reason = stitch_offset.apply_collapsed_offset_topology(
                        bm,
                        collapsed_target_markers,
                        use_cap_endpoint=session.offset_use_cap_endpoint,
                    )
                    if not collapsed_segment_count:
                        raise SymmetricKnifeError(collapsed_reason)
                    if collapsed_segment_count != len(source_edges):
                        raise SymmetricKnifeError("The mirrored zero-offset topology did not match the source")
                    mirror_committed = True
                    result = {"FINISHED"}
                    self.report(
                        {"INFO"},
                        f"Mirrored {collapsed_segment_count} zero-factor Offset "
                        f"Edge Loop Cut segment(s) from the {side.lower()} side",
                    )
                    return result

                # Rebuild the native cut topology on paired target faces using
                # exact reflected points.
                selection_state = selection.add_selection_layers(bm)
                backup_mesh = _create_backup(bm)
                bm = bmesh.from_edit_mesh(obj.data)
                source_edges, side, total_path_edges, crossing_count = stitch_pathedges.collect_source_path_edges(
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
                created, already_present, direct_reason = stitch_reflect.apply_reflected_path_topology(
                    bm,
                    source_edges,
                    session.axis_index,
                    session.tolerance,
                    live_mirror_face_ids,
                    session.carrier_frames,
                )
                if direct_reason:
                    raise SymmetricKnifeError(f"Could not rebuild the mirrored {tool_label}: {direct_reason}")
                if created + already_present != len(source_edges):
                    raise SymmetricKnifeError(f"The mirrored {tool_label} topology did not match the source")

                mirror_committed = True
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
            if backup_mesh is not None and not mirror_committed:
                try:
                    backup.restore_topology_backup(obj.data, backup_mesh)
                except Exception:
                    traceback.print_exc()
                    rollback_failed = True

            # Mirror-stage operations replace selection and temporary hiding
            # even when a later step fails. Best-effort restoration keeps the
            # native source result usable (and the whole operation remains one
            # Undo step).
            if obj is not None and obj.mode == "EDIT":
                try:
                    bm = bmesh.from_edit_mesh(obj.data)
                    context.tool_settings.mesh_select_mode = mesh_select_mode.as_tuple()
                    if selection_state is not None:
                        summary = stitch_common.combine_selection_mutation_summaries(restore_mutation_summaries)
                        if session.tool_kind == "KNIFE" and restore_mutation_summaries:
                            scoped_direct_success = (
                                direct_topology_success and mirror_committed and mirror_failure is None
                            )
                            selection.restore_selection_for_route(
                                bm,
                                session.hidden_by_face_id,
                                selection_state,
                                summary,
                                direct_topology_success=scoped_direct_success,
                                summary_complete=restore_summary_complete and summary.complete,
                            )
                        else:
                            selection.restore_visibility_and_selection(
                                bm,
                                session.hidden_by_face_id,
                                selection_state,
                            )
                    else:
                        face_layer = bm.faces.layers.int.get(layer_names.FACE_ID_LAYER)
                        if face_layer is not None:
                            for face in bm.faces:
                                face_id = FaceId(int(face[face_layer]))
                                face.hide = bool(session.hidden_by_face_id.get(face_id, False))
                    if not self.preserve_history_layers:
                        snapshot.remove_temporary_layers(bm)
                    # Select Mirrored normally runs after layer cleanup so it sees
                    # permanent topology. History's first F9 repair stage retains
                    # the layers for its immediately following adjusted stage.
                    # Early returns still execute this finally (Python semantics).
                    if mirror_failure is None:
                        try:
                            settings = getattr(context.scene, "ydd_symmetric_edit", None)
                            if settings is not None and bool(getattr(settings, "select_mirrored", False)):
                                selection.extend_selection_to_mirror(
                                    bm,
                                    session.axis_index,
                                    session.tolerance,
                                    mesh_object=obj,
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
                if self.preserve_history_layers:
                    session_state._SESSIONS.pop(window_pointer, None)
                else:
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
                # The disposition hint holds only here: the ERROR branch cannot
                # promise the mirror side was left untouched.
                _finish_report(
                    self,
                    {"WARNING"},
                    f"ydd Symmetric Edit: {mirror_failure} (native cut kept; mirror manually or undo)",
                )
            result = {"FINISHED"}

        return result


def _invoke_finish_operator(*, preserve_history_layers: bool = False) -> set[str]:
    finish_operator = getattr(bpy.ops.mesh, "ydd_symmetric_edit_finish")
    if preserve_history_layers:
        return cast(set[str], finish_operator("EXEC_DEFAULT", preserve_history_layers=True))
    return cast(set[str], finish_operator("EXEC_DEFAULT"))


CLASSES = (
    MESH_OT_ydd_symmetric_edit_intercept,
    MESH_OT_ydd_symmetric_edit_finish,
)
