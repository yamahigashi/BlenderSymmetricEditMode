from __future__ import annotations

import time
import traceback

import bmesh
import bpy

from . import extrude, layer_names, rip, session_state
from ._types import (
    Coordinate3D,
    KnifeSession,
    PathEdgeSignature,
    PathSignature,
    RipSignature,
)
from .session import (
    _PASSTHROUGH_HANDOFF_GRACE,
    _PASSTHROUGH_STABLE_TICKS,
    EXTRUDE_TOOL_KINDS,
    MODAL_IDENTIFIER_TOKENS,
    TOOL_PROFILES,
    _find_saved_view,
    _find_window,
    cleanup_session,
)


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
    return any(token.upper() in identifier for identifier in _modal_operator_identifiers(window) for token in tokens)


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
            if session.tool_kind in EXTRUDE_TOOL_KINDS:
                return extrude.extrude_result_signature(bm)
            marker_layer = bm.edges.layers.int.get(layer_names.EDGE_ORIGINAL_LAYER)
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
            if session.tool_kind in EXTRUDE_TOOL_KINDS:
                return extrude.has_extrude_result(bm)
            marker_layer = bm.edges.layers.int.get(layer_names.EDGE_ORIGINAL_LAYER)
            if marker_layer is None:
                return False
            return any(int(edge[marker_layer]) <= 0 for edge in bm.edges)
    except (ReferenceError, RuntimeError):
        return False
    return False


def _selection_signature(session: KnifeSession) -> tuple[tuple[int, bool], ...]:
    obj = bpy.data.objects.get(session.object_name)
    if obj is None or obj.type != "MESH" or obj.mode != "EDIT":
        return ()
    try:
        bm = bmesh.from_edit_mesh(obj.data)
        vertex_layer = bm.verts.layers.int.get(layer_names.VERT_SESSION_ID_LAYER)
        if vertex_layer is None:
            return ()
        return tuple(
            sorted(
                (int(vertex[vertex_layer]), bool(vertex.select)) for vertex in bm.verts if int(vertex[vertex_layer]) > 0
            )
        )
    except (ReferenceError, RuntimeError):
        return ()


def _capture_confirmed_extrude_operator(session: KnifeSession, window) -> None:
    """Remember the macro that just left the modal stack."""

    del window
    profile = TOOL_PROFILES.get(session.tool_kind)
    expected = (profile.primary_wm_operator if profile is not None else "").upper()
    saved_window, area, region = _find_saved_view(session)
    operator = None
    if saved_window is not None and area is not None and region is not None:
        try:
            with bpy.context.temp_override(window=saved_window, area=area, region=region):
                operator = getattr(bpy.context, "active_operator", None)
        except Exception:
            traceback.print_exc()
            operator = None
    if operator is None or getattr(operator, "bl_idname", "").upper() != expected:
        return
    try:
        pointer = int(operator.as_pointer())
        idname = str(operator.bl_idname)
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        return
    if not pointer or not idname:
        return
    session.confirmed_operator_idname = idname
    session.confirmed_operator_pointer = pointer
    session.confirmed_selection_signature = _selection_signature(session)
    record = session_state._HISTORY_RECORDS.get(session.history_token)
    if record is not None:
        record.session.confirmed_operator_idname = session.confirmed_operator_idname
        record.session.confirmed_operator_pointer = session.confirmed_operator_pointer
        record.session.confirmed_selection_signature = session.confirmed_selection_signature


def _confirmed_extrude_fully_captured(session: KnifeSession) -> bool:
    return bool(
        session.confirmed_operator_idname
        and session.confirmed_operator_pointer
        and session.extrude_options_captured
    )


def _capture_confirmed_extrude_result(session: KnifeSession) -> bool:
    """Capture confirmed macro, selection signature, and native options.

    Returns True only when idname, pointer, and extrude_options_captured are set.
    An empty selection signature is a valid snapshot.
    """

    if _confirmed_extrude_fully_captured(session):
        return True

    already_confirmed = bool(session.confirmed_operator_idname and session.confirmed_operator_pointer)
    window, area, region = _find_saved_view(session)
    if window is None or area is None or region is None:
        return False
    try:
        with bpy.context.temp_override(window=window, area=area, region=region):
            if not already_confirmed:
                _capture_confirmed_extrude_operator(session, window)
                if not (session.confirmed_operator_idname and session.confirmed_operator_pointer):
                    session.confirmed_selection_signature = _selection_signature(session)
                    record = session_state._HISTORY_RECORDS.get(session.history_token)
                    if record is not None:
                        record.session.confirmed_selection_signature = session.confirmed_selection_signature
            _capture_native_result_options(session, bpy.context)
    except Exception:
        traceback.print_exc()
        return False
    return _confirmed_extrude_fully_captured(session)


def _capture_native_result_options(session: KnifeSession, context) -> None:
    """Retain native macro options needed by a topology-only fallback."""

    profile = TOOL_PROFILES.get(session.tool_kind)
    expected_identifier = (profile.primary_wm_operator if profile is not None else "").upper()
    operator = context.active_operator
    if operator is None or getattr(operator, "bl_idname", "").upper() != expected_identifier:
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
    record = session_state._HISTORY_RECORDS.get(session.history_token)
    if record is not None:
        record.session.native_operator_pointer = session.native_operator_pointer

    if session.tool_kind in EXTRUDE_TOOL_KINDS:
        options = extrude.capture_native_options(operator, session.tool_kind)
        if options is not None and session.tool_kind == "EXTRUDE_SHRINK_FATTEN":
            options = extrude.ensure_observed_shrink_value(options, session)
        captured = options is not None
        session.extrude_options = options
        session.extrude_options_captured = captured
        if record is not None:
            record.session.extrude_options = options
            record.session.extrude_options_captured = captured
        return

    if profile is None or not profile.supports_nested_offset:
        return
    nested = getattr(operator, "MESH_OT_offset_edge_loops", None)
    if nested is not None:
        session.offset_use_cap_endpoint = bool(getattr(nested, "use_cap_endpoint", False))
    if record is not None:
        record.session.offset_use_cap_endpoint = session.offset_use_cap_endpoint


def _watch_passthrough_session(window_pointer: int, history_token: int):
    session = session_state._SESSIONS.get(window_pointer)
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
        return session_state._PASSTHROUGH_POLL_INTERVAL

    now = time.monotonic()
    if not session.saw_modal and not _session_has_new_path(session):
        if now - session.started_at < session_state._PASSTHROUGH_START_GRACE:
            return session_state._PASSTHROUGH_POLL_INTERVAL
        cleanup_session(window_pointer)
        return None

    # Loop Cut and Offset are native macros.  Their topology child and Edge
    # Slide child can leave a brief interval with no relevant modal operator.
    # Require both a quiet interval and an unchanged result before postprocess.
    path_signature = _session_new_path_signature(session)
    if session.modal_absent_since is None:
        if session.tool_kind in EXTRUDE_TOOL_KINDS:
            _capture_confirmed_extrude_operator(session, window)
        session.modal_absent_since = now
        session.path_signature = path_signature
        session.stable_path_ticks = 1
        if session.tool_kind in EXTRUDE_TOOL_KINDS and tuple(window.modal_operators):
            return _invoke_passthrough_finish(window_pointer, history_token, session)
        return session_state._PASSTHROUGH_POLL_INTERVAL
    if session.tool_kind in EXTRUDE_TOOL_KINDS and tuple(window.modal_operators):
        # Grab/other modal during grace keeps moving verts, so the signature
        # never stabilizes. Finish immediately as an intervening-operator decline.
        return _invoke_passthrough_finish(window_pointer, history_token, session)
    if path_signature == session.path_signature:
        session.stable_path_ticks += 1
    else:
        session.modal_absent_since = now
        session.path_signature = path_signature
        session.stable_path_ticks = 1
    handoff_grace = _PASSTHROUGH_HANDOFF_GRACE.get(session.tool_kind, 0.04)
    stable_ticks = _PASSTHROUGH_STABLE_TICKS.get(session.tool_kind, 3)
    if now - session.modal_absent_since < handoff_grace or session.stable_path_ticks < stable_ticks:
        return session_state._PASSTHROUGH_POLL_INTERVAL

    if path_signature is None:
        # Knife/Loop Cut cancellation and confirmed no-ops leave no topology.
        cleanup_session(window_pointer)
        return None

    return _invoke_passthrough_finish(window_pointer, history_token, session)


def _invoke_passthrough_finish(window_pointer: int, history_token: int, session: KnifeSession):
    from .operators import _invoke_finish_operator

    record = session_state._HISTORY_RECORDS.get(history_token)
    window, area, region = _find_saved_view(session)
    if window is None or area is None or region is None:
        if record is not None:
            record.status = "FAILED"
        cleanup_session(window_pointer, keep_history_record=True)
        return None

    try:
        if session.tool_kind in EXTRUDE_TOOL_KINDS:
            if not _capture_confirmed_extrude_result(session):
                if record is not None:
                    record.status = "FAILED"
                cleanup_session(window_pointer, keep_history_record=True)
                return None
        with bpy.context.temp_override(window=window, area=area, region=region):
            if session.tool_kind not in EXTRUDE_TOOL_KINDS:
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
