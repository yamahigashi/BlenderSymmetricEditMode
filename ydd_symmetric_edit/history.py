from __future__ import annotations

import copy
import traceback
from typing import cast

import bmesh
import bpy
from bpy.app.handlers import persistent

from . import core, session_state
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
        history_layer = bm.faces.layers.int.get(core.HISTORY_TOKEN_LAYER)
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
        restored = _restore_session_face_maps(session, obj)
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

    # A live session may have started in this window between the undo_post
    # queueing and this timer tick.  Storing the repair session would silently
    # replace that entry, killing its watcher and leaking its suspended
    # symmetry flags.  Leave everything untouched; the record stays COMMITTED
    # so the next undo_post can retry the repair.
    if session.window_pointer in session_state._SESSIONS:
        return None

    session_state._HISTORY_REPAIR_BUSY = True  # type: ignore
    try:
        session_state._SESSIONS[session.window_pointer] = session
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
        session_state._HISTORY_REPAIR_BUSY = False
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
