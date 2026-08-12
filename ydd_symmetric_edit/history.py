from __future__ import annotations

import copy
import traceback
from typing import cast

import bmesh
import bpy
from bpy.app.handlers import persistent

from . import layer_names, matching, session_state, snapshot, stitch
from ._types import (
    FaceId,
    HiddenFaceMap,
    HistoryRecord,
    KnifeSession,
    MirrorFaceMap,
    SymmetryAxes,
)
from .session import (
    _WM_OPERATOR_TO_TOOL,
    TOOL_PROFILES,
    _cleanup_repair_session,
    _find_saved_view,
    _prepare_session,
    _single_edit_mesh_poll,
    _window_key,
    cleanup_session,
)


def _remember_history_session(session: KnifeSession, context) -> None:

    session_state._HISTORY_SEQUENCE += 1
    history_session = copy.copy(session)
    # Native Offset temporarily suspends Mesh Symmetry only while its Edge
    # Slide is live.  A later Undo/Redo repair must never rewrite user settings.
    history_session.symmetry_suspended = False
    session_state._HISTORY_RECORDS[session.history_token] = HistoryRecord(
        session=history_session,
        sequence=session_state._HISTORY_SEQUENCE,
    )
    session_state._HISTORY_RECORDS.move_to_end(session.history_token)
    configured_limit = int(getattr(context.preferences.edit, "undo_steps", 32)) + 4
    history_limit = min(session_state._MAX_HISTORY_RECORDS, max(8, configured_limit))
    while len(session_state._HISTORY_RECORDS) > history_limit:
        session_state._HISTORY_RECORDS.popitem(last=False)


def clear_history_records() -> None:
    session_state._HISTORY_RECORDS.clear()


def _read_history_tokens(obj) -> set[int]:
    """Read raw history tokens; read failures propagate to the caller."""

    if obj.type != "MESH":
        return set()
    if obj.mode == "EDIT":
        bm = bmesh.from_edit_mesh(obj.data)
        layer = bm.faces.layers.int.get(layer_names.HISTORY_TOKEN_LAYER)
        if layer is None:
            return set()
        return {int(face[layer]) for face in bm.faces if int(face[layer])}
    attribute = obj.data.attributes.get(layer_names.HISTORY_TOKEN_LAYER)
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
                    (bm.edges.layers.int, layer_names.EDGE_ORIGINAL_LAYER),
                    (bm.faces.layers.int, layer_names.FACE_ID_LAYER),
                    (bm.faces.layers.int, layer_names.HISTORY_TOKEN_LAYER),
                )
            )
        return any(obj.data.attributes.get(name) is not None for name in layer_names.TEMP_LAYER_NAMES)
    except (ReferenceError, RuntimeError):
        return False


def _restore_session_face_maps(session: KnifeSession, obj) -> bool:
    if obj.mode != "EDIT":
        return False
    try:
        bm = bmesh.from_edit_mesh(obj.data)
        face_id_layer = bm.faces.layers.int.get(layer_names.FACE_ID_LAYER)
        mirror_id_layer = bm.faces.layers.int.get(layer_names.FACE_MIRROR_ID_LAYER)
        hidden_layer = bm.faces.layers.int.get(layer_names.FACE_HIDDEN_LAYER)
        history_layer = bm.faces.layers.int.get(layer_names.HISTORY_TOKEN_LAYER)
        if face_id_layer is None:
            return False

        mirror_face_ids: MirrorFaceMap = {}
        hidden_by_face_id: HiddenFaceMap = {}
        face_ids_valid = True
        mirror_values_complete = mirror_id_layer is not None
        mirror_values_consistent = True
        for face in bm.faces:
            face_id = FaceId(int(face[face_id_layer]))
            if face_id <= 0:
                face_ids_valid = False
                continue
            raw_mirror_id = int(face[mirror_id_layer]) if mirror_id_layer is not None else 0
            if raw_mirror_id > 0:
                mirror_id = FaceId(raw_mirror_id)
                previous = mirror_face_ids.setdefault(face_id, mirror_id)
                if previous != mirror_id:
                    mirror_values_consistent = False
            else:
                mirror_values_complete = False
            hidden_by_face_id.setdefault(face_id, bool(face[hidden_layer]) if hidden_layer is not None else False)
        face_id_domain = set(hidden_by_face_id)
        layer_complete = (
            face_ids_valid
            and mirror_values_complete
            and mirror_values_consistent
            and bool(face_id_domain)
            and set(mirror_face_ids) == face_id_domain
            and set(mirror_face_ids.values()).issubset(face_id_domain)
            and len(mirror_face_ids) == len(set(mirror_face_ids.values()))
        )
        resolution = session.topology_resolution
        if resolution is not None:
            tokens = (
                {int(face[history_layer]) for face in bm.faces if int(face[face_id_layer]) > 0}
                if history_layer is not None
                else set()
            )
            token_matches = resolution.history_token == session.history_token and session.history_token in tokens
            handle_domain_matches = face_ids_valid and face_id_domain == set(range(1, resolution.face_count + 1))
            if token_matches and handle_domain_matches:
                expected = resolution.resolve().mirror_face_ids
                session.mirror_face_ids = expected
                session.carrier_frames = resolution.carrier_frames
                resolution.materialize(bm)
            elif layer_complete:
                session.mirror_face_ids = mirror_face_ids
            else:
                return False
        elif layer_complete:
            session.mirror_face_ids = mirror_face_ids
        else:
            return False
        session.hidden_by_face_id = hidden_by_face_id
        return bool(session.mirror_face_ids)
    except (ReferenceError, RuntimeError):
        return False


def _cleanup_object_temporary_layers(obj) -> None:
    try:
        if obj.mode == "EDIT":
            bm = bmesh.from_edit_mesh(obj.data)
            if snapshot.remove_temporary_layers(bm):
                bmesh.update_edit_mesh(
                    obj.data,
                    loop_triangles=False,
                    destructive=False,
                )
        else:
            snapshot.remove_temporary_mesh_attributes(obj.data)
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


def _edge_coordinate_signature(edge) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    coordinates = sorted((float(vertex.co[0]), float(vertex.co[1]), float(vertex.co[2])) for vertex in edge.verts)
    return coordinates[0], coordinates[1]


def _select_path_signatures(obj, signatures, tolerance: float, *, mark_as_path: bool = False) -> bool:
    """Replace the edit selection with the edges matching *signatures*."""

    bm = bmesh.from_edit_mesh(obj.data)
    matches = []
    for signature in signatures:
        candidates = [
            edge
            for edge in bm.edges
            if all(
                max(abs(a - b) for a, b in zip(actual, expected, strict=True)) <= tolerance
                for actual, expected in zip(_edge_coordinate_signature(edge), signature, strict=True)
            )
        ]
        if len(candidates) != 1 or candidates[0] in matches:
            return False
        matches.append(candidates[0])

    for vertex in bm.verts:
        vertex.select = False
    for edge in bm.edges:
        edge.select = False
    for face in bm.faces:
        face.select = False
    for edge in matches:
        edge.select = True
        for vertex in edge.verts:
            vertex.select = True
    if mark_as_path:
        marker_layer = bm.edges.layers.int.get(layer_names.EDGE_ORIGINAL_LAYER)
        if marker_layer is None:
            return False
        for edge in matches:
            edge[marker_layer] = 0
    bm.select_flush_mode()
    bm.select_history.clear()
    for edge in matches:
        if edge.is_valid and edge.select:
            bm.select_history.add(edge)
    bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
    return True


def _adjust_last_operation_repair_plan(obj, prior_record: HistoryRecord):
    """Describe an F9 repeat that swallowed a prior selected-path repair."""

    selected_path_tools = {"LOOP_CUT", "OFFSET_LOOP_CUT"}
    if prior_record.session.tool_kind not in selected_path_tools:
        return None

    active_operator = bpy.context.active_operator
    if active_operator is None:
        return None
    try:
        active_pointer = int(active_operator.as_pointer())
    except (AttributeError, ReferenceError, RuntimeError):
        return None
    active_tool_kind = _WM_OPERATOR_TO_TOOL.get(getattr(active_operator, "bl_idname", "").upper())
    if active_tool_kind not in selected_path_tools:
        return None
    window_pointer = _window_key(bpy.context)
    if prior_record.session.window_pointer != window_pointer:
        return None
    adjusted_records = [
        record
        for record in session_state._HISTORY_RECORDS.values()
        if record is not prior_record
        and record.status == "COMMITTED"
        and record.sequence > prior_record.sequence
        and record.session.tool_kind == active_tool_kind
        and record.session.window_pointer == window_pointer
        and record.session.object_name == obj.name
        and record.session.mesh_name == obj.data.name
        and record.session.native_operator_pointer == active_pointer
    ]
    if not adjusted_records:
        return None
    adjusted_record = max(adjusted_records, key=lambda record: record.sequence)

    try:
        bm = bmesh.from_edit_mesh(obj.data)
        marker_layer = bm.edges.layers.int.get(layer_names.EDGE_ORIGINAL_LAYER)
        if marker_layer is None:
            return None
        prior_edges = [edge for edge in bm.edges if int(edge[marker_layer]) == 0 and not edge.select]
        adjusted_edges, _side, total_path_edges, _crossing_count = stitch.collect_source_path_edges(
            bm,
            adjusted_record.session.axis_index,
            adjusted_record.session.tolerance,
            adjusted_record.session.source_side,
            selected_only=True,
        )
    except (AttributeError, ReferenceError, RuntimeError):
        return None
    if not prior_edges or not adjusted_edges or total_path_edges == 0:
        return None
    return (
        adjusted_record,
        tuple(_edge_coordinate_signature(edge) for edge in prior_edges),
        tuple(_edge_coordinate_signature(edge) for edge in adjusted_edges),
    )


def _live_adjusted_path_face_map(bm, session: KnifeSession):
    """Map the adjusted source carriers onto stage 1's current target faces."""

    face_layer = bm.faces.layers.int.get(layer_names.FACE_ID_LAYER)
    if face_layer is None:
        return None
    source_edges, _side, total_path_edges, _crossing_count = stitch.collect_source_path_edges(
        bm,
        session.axis_index,
        session.tolerance,
        session.source_side,
        selected_only=True,
    )
    if not source_edges or total_path_edges == 0:
        return None

    candidate_faces = {face for face in bm.faces if face.is_valid}
    mirror_face_ids: MirrorFaceMap = {}
    for source_edge in source_edges:
        endpoint_face_sets = []
        for vertex in source_edge.verts:
            expected = matching.mirror_coordinate(vertex.co, session.axis_index)
            kind, exact_vertex, boundary_edge, _factor, _reason = stitch._resolve_reflected_vertex_on_target(
                expected,
                candidate_faces,
                session.tolerance,
            )
            if kind == "exact" and exact_vertex is not None:
                endpoint_faces = {face for face in exact_vertex.link_faces if face.is_valid}
            elif kind == "boundary" and boundary_edge is not None:
                endpoint_faces = {face for face in boundary_edge.link_faces if face.is_valid}
            else:
                return None
            endpoint_face_sets.append(endpoint_faces)

        target_faces = endpoint_face_sets[0].intersection(*endpoint_face_sets[1:])
        if len(target_faces) != 1:
            return None
        target_face = next(iter(target_faces))
        target_id = FaceId(int(target_face[face_layer]))
        if target_id <= 0:
            return None
        for source_face in source_edge.link_faces:
            if not source_face.is_valid:
                continue
            source_id = FaceId(int(source_face[face_layer]))
            if source_id <= 0:
                return None
            # Native cuts split one prepared carrier into several descendants;
            # all of those source IDs intentionally share its unsplit target.
            previous = mirror_face_ids.setdefault(source_id, target_id)
            if previous != target_id:
                return None
    return mirror_face_ids or None


def _prepare_adjusted_session_face_maps(session: KnifeSession, obj, path_signatures) -> bool:
    """Build stage 2's carrier IDs from the mesh produced by stage 1."""

    bm = bmesh.from_edit_mesh(obj.data)
    topology = snapshot.prepare_topology(
        bm,
        session.axis_index,
        session.tolerance,
        session.history_token,
        mesh_object=obj,
    )
    # The adjusted native path makes this snapshot asymmetric, so its resolver
    # cannot supply the mirror map. Keep only the freshly stamped ID/marker
    # domain and derive the selected carriers geometrically below.
    session.topology_resolution = None
    session.mirror_face_ids = {}
    session.carrier_frames = {}
    session.hidden_by_face_id = topology.hidden_by_face_id
    bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
    if not _select_path_signatures(
        obj,
        path_signatures,
        session.tolerance,
        mark_as_path=True,
    ):
        return False
    bm = bmesh.from_edit_mesh(obj.data)
    live_map = _live_adjusted_path_face_map(bm, session)
    if live_map is None:
        return False
    session.mirror_face_ids = live_map
    return True


def _restore_history_record_session(obj, record: HistoryRecord, *, adjusted_path_signatures=None):
    try:
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
        restored = (
            _restore_session_face_maps(session, obj)
            if adjusted_path_signatures is None
            else _prepare_adjusted_session_face_maps(session, obj, adjusted_path_signatures)
        )
    except Exception:
        traceback.print_exc()
        record.status = "FAILED"
        _cleanup_object_temporary_layers(obj)
        return None
    if not restored:
        record.status = "FAILED"
        _cleanup_repair_session(session)
        return None
    window, area, region = _find_saved_view(session)
    if window is None or area is None or region is None:
        record.status = "FAILED"
        _cleanup_repair_session(session)
        return None
    return session, window, area, region


def _repair_history_state():
    from .operators import _invoke_finish_operator

    session_state._HISTORY_REPAIR_QUEUED = False
    if session_state._HISTORY_REPAIR_BUSY:
        return None

    marker_objects = _history_marker_objects()
    if not marker_objects:
        return None

    live_tokens = {session.history_token for session in session_state._SESSIONS.values() if session.history_token}
    live_mesh_names = {session.mesh_name for session in session_state._SESSIONS.values()}
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
            (token, session_state._HISTORY_RECORDS[token])
            for token in tokens
            if token in session_state._HISTORY_RECORDS and session_state._HISTORY_RECORDS[token].status == "COMMITTED"
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

    repeat_plan = _adjust_last_operation_repair_plan(obj, record)
    restored_record = _restore_history_record_session(obj, record)
    if restored_record is None:
        return None
    session, window, area, region = restored_record

    # A live session may have started in this window between the undo_post
    # queueing and this timer tick.  Storing the repair session would silently
    # replace that entry, killing its watcher and leaking its suspended
    # symmetry flags.  Leave everything untouched; the record stays COMMITTED
    # so the next undo_post can retry the repair.
    if session.window_pointer in session_state._SESSIONS:
        return None

    session_state._HISTORY_REPAIR_BUSY = True  # type: ignore
    failed_record = record
    cleanup_snapshot = session
    try:
        if repeat_plan is not None:
            adjusted_record, prior_signatures, adjusted_signatures = repeat_plan
            if not _select_path_signatures(obj, prior_signatures, session.tolerance):
                raise RuntimeError("The prior raw cut path changed before history repair")
        session_state._SESSIONS[session.window_pointer] = session
        with bpy.context.temp_override(window=window, area=area, region=region):
            # F9 has already moved the selection to the adjusted native path.
            # Repair the unselected prior path under its own session first, but
            # retain its marker maps long enough to interpret the adjusted path.
            result = _invoke_finish_operator(preserve_history_layers=repeat_plan is not None)
        if "FINISHED" not in result:
            record.status = "FAILED"
            _cleanup_repair_session(session)
        elif repeat_plan is not None:
            restored_adjusted = _restore_history_record_session(
                obj,
                adjusted_record,
                adjusted_path_signatures=adjusted_signatures,
            )
            if restored_adjusted is None:
                # Stage 1 committed; the adjusted operation stays unmirrored and
                # unrepairable because its layers were wiped with the failure.
                print(
                    "ydd Symmetric Edit: F9 repair stage 2 failed; "
                    "the adjusted operation was left unmirrored"
                )
                session_state._HISTORY_RECORDS.move_to_end(token)
                return None
            adjusted_session, adjusted_window, adjusted_area, adjusted_region = restored_adjusted
            failed_record = adjusted_record
            cleanup_snapshot = adjusted_session
            if adjusted_session.window_pointer in session_state._SESSIONS:
                # A live session appeared during stage 1.  The mesh keeps the
                # adjusted token layers, so the next repair tick can retry this
                # record as an ordinary single-token candidate.
                session_state._HISTORY_RECORDS.move_to_end(token)
                return None
            session_state._SESSIONS[adjusted_session.window_pointer] = adjusted_session
            with bpy.context.temp_override(
                window=adjusted_window,
                area=adjusted_area,
                region=adjusted_region,
            ):
                # Normal finish performs the final layer cleanup.  Moving the
                # adjusted record last preserves chronological bookkeeping.
                adjusted_result = _invoke_finish_operator()
            if "FINISHED" not in adjusted_result:
                adjusted_record.status = "FAILED"
                _cleanup_repair_session(adjusted_session)
            session_state._HISTORY_RECORDS.move_to_end(token)
            session_state._HISTORY_RECORDS.move_to_end(adjusted_session.history_token)
    except Exception:
        traceback.print_exc()
        failed_record.status = "FAILED"
        _cleanup_repair_session(cleanup_snapshot)
    finally:
        session_state._HISTORY_REPAIR_BUSY = False
    if repeat_plan is None:
        session_state._HISTORY_RECORDS.move_to_end(token)
    return None


def _queue_history_repair(_dummy=None) -> None:

    if session_state._HISTORY_REPAIR_BUSY or session_state._HISTORY_REPAIR_QUEUED:
        return
    session_state._HISTORY_REPAIR_QUEUED = True  # type: ignore
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
            for record in reversed(tuple(session_state._HISTORY_RECORDS.values()))
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
            path_state = stitch.native_path_edge_state(bm)
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

    for handlers, callback in (
        (bpy.app.handlers.undo_post, repair_after_undo),
        (bpy.app.handlers.redo_post, repair_after_redo),
    ):
        if callback in handlers:
            handlers.remove(callback)
    if bpy.app.timers.is_registered(_repair_history_state):
        bpy.app.timers.unregister(_repair_history_state)
    session_state._HISTORY_REPAIR_QUEUED = False
    clear_history_records()
