# SPDX-License-Identifier: GPL-3.0-or-later

"""Inset Faces / Bevel EXPAND_PASSTHROUGH intercept, poller, and pure helpers."""

from __future__ import annotations

import time
import traceback
from collections.abc import Iterable
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import TYPE_CHECKING, Literal, Protocol

if TYPE_CHECKING:
    import bmesh
    import bpy
    from bpy.app.handlers import persistent
    from bpy.props import StringProperty
    from bpy.stub_internal.rna_enums import OperatorReturnItems
else:
    try:
        import bmesh
        import bpy
        from bpy.app.handlers import persistent
        from bpy.props import StringProperty
    except ImportError:
        bmesh = None
        bpy = None

        def persistent(function):
            return function

        StringProperty = None

TARGET_OPERATOR_IDNAMES = frozenset({"mesh.inset", "mesh.bevel"})
FAMILY_OPERATOR_IDNAMES = TARGET_OPERATOR_IDNAMES
REPLAY_OPERATOR_IDNAMES = frozenset({"mesh.vert_connect_path", "mesh.dissolve_mode"})
REPLAY_CALL_MENU_NAMES = frozenset(
    {
        "VIEW3D_MT_edit_mesh_merge",
        "VIEW3D_MT_edit_mesh_delete",
        "VIEW3D_MT_edit_mesh_extrude",
    }
)
DEFAULT_BEVEL_AFFECT = "EDGES"
ONSET_TIMEOUT_S = 2.0
POLLER_FIRST_INTERVAL = 0.02
POLLER_INTERVAL = 0.05
# Bevel's native cancel can stomp selection writes for a short window after
# the modal disappears (deferred cleanup, measured; exact timing racy).  The
# verify loop rewrites with growing intervals until the write survives.
RESTORE_VERIFY_ATTEMPTS = 8
RESTORE_VERIFY_BASE_INTERVAL = 0.05
RESTORE_VERIFY_MAX_INTERVAL = 0.4
DIAGNOSTIC_VERTEX_LIMIT = 500_000
MAX_ARM_ATTEMPTS = 2
UNDO_PUSH_MESSAGE = "Symmetric select (inset/bevel)"
HIDDEN_WARNING = "Inset/Bevel declined: {count} hidden counterpart(s)"
UNMATCHED_WARNING = "Inset/Bevel will run on one side only; the mesh is not symmetric"
PUSH_FAILED_WARNING = "Inset/Bevel could not record an undo step; leaving the native route unexpanded"
ARM_FAILED_WARNING = "Inset/Bevel poller failed; the expanded selection may remain"
RESTORE_FAILED_WARNING = "Inset/Bevel could not restore the pre-expansion selection"
DIAGNOSTIC_WARNING = "Inset/Bevel result is not mirror-symmetric ({count} vertices)"
DIAGNOSTIC_SKIP_INFO = "Inset/Bevel symmetry check skipped (vertex count exceeds 500000)"
TOOL_KEYMAP_PREFIX = "3D View Tool:"


class PollerVerdict(StrEnum):
    WAIT = "WAIT"
    CONFIRMED = "CONFIRMED"
    CANCELLED = "CANCELLED"
    UNKNOWN = "UNKNOWN"


class TickResult(StrEnum):
    NO_OP = "NO_OP"
    DISCARD = "DISCARD"
    WAIT = "WAIT"
    CONFIRMED = "CONFIRMED"
    CANCELLED = "CANCELLED"
    UNKNOWN = "UNKNOWN"


class TransactionEvent(StrEnum):
    EXCEPTION = "exception"
    TOKEN_REGISTER_FAILED = "token_register_failed"
    PUSH_FAILED = "push_failed"
    ARM_FAILED = "arm_failed"
    RESTORE_FAILED = "restore_failed"


class TransactionAction(StrEnum):
    PASS_THROUGH = "pass_through"
    DISCARD_TOKEN_PASS_THROUGH = "discard_token_pass_through"
    DISCARD_S0_PASS_THROUGH = "discard_s0_pass_through"
    RESTORE_S0_DISCARD_TOKEN_PASS_THROUGH = "restore_s0_discard_token_pass_through"
    RESTORE_S0_DISCARD_TOKEN_WARN_PASS_THROUGH = "restore_s0_discard_token_warn_pass_through"
    ATTEMPT_ARM = "attempt_arm"
    WARN_PASS_THROUGH = "warn_pass_through"
    LEAVE_EXPANDED_WARN_DISCARD_TOKEN = "leave_expanded_warn_discard_token"
    WARN_ONLY = "warn_only"


class PreambleDecision(StrEnum):
    NO_PRIOR = "NO_PRIOR"
    CONSUME_EVENT = "CONSUME_EVENT"
    SETTLED = "SETTLED"


class _KmiEvent(Protocol):
    type: str
    value: str
    any: bool
    shift: int
    ctrl: int
    alt: int
    oskey: int
    hyper: int
    key_modifier: str
    direction: str
    repeat: bool


@dataclass(frozen=True, slots=True)
class OperatorIdentity:
    idname: str | None
    pointer: int | None


UNOBSERVABLE_OPERATOR = OperatorIdentity("__unobservable__", None)


@dataclass(frozen=True, slots=True)
class MeshCounts:
    verts: int
    edges: int
    faces: int


@dataclass(frozen=True, slots=True)
class SelectionSnapshot:
    verts: frozenset[int]
    edges: frozenset[int]
    faces: frozenset[int]
    counts: MeshCounts


@dataclass(frozen=True, slots=True)
class TransactionState:
    snapshot_exists: bool = False
    token_registered: bool = False
    selection_mutated: bool = False
    push_finished: bool = False
    poller_armed: bool = False
    arm_attempts: int = 0


@dataclass(frozen=True, slots=True)
class PriorTokenView:
    onset: bool
    modal_alive: bool
    recorded_op: OperatorIdentity
    current_op: OperatorIdentity
    invoke_counts: MeshCounts
    current_counts: MeshCounts
    selection_matches_s1: bool
    elapsed_s: float
    s0: frozenset[int]
    discard: bool = False


@dataclass(frozen=True, slots=True)
class PreambleResult:
    decision: PreambleDecision
    verdict: PollerVerdict | None
    restored_s0: bool
    token_cleared: bool
    selection: frozenset[int]


def normalize_operator_idname(raw: str | None) -> str | None:
    if raw is None:
        return None
    text = str(raw)
    if not text:
        return None
    if text in TARGET_OPERATOR_IDNAMES:
        return text
    lowered = text.lower()
    if lowered.startswith("mesh_ot_"):
        return "mesh." + lowered[8:]
    return text


def _values_overlap(left: str, right: str) -> bool:
    if left == right:
        return True
    return left == "ANY" or right == "ANY"


def _modifier_pair_overlaps(left: int, right: int) -> bool:
    if left == -1 or right == -1:
        return True
    return left == right


def kmi_events_overlap(left: _KmiEvent, right: _KmiEvent) -> bool:
    if str(left.type) != str(right.type):
        return False
    if not _values_overlap(str(left.value), str(right.value)):
        return False
    if not (bool(left.any) or bool(right.any)):
        # hyper is missing on pre-4.4 KeyMapItems; absent means 0 (not held).
        for name in ("shift", "ctrl", "alt", "oskey", "hyper"):
            if not _modifier_pair_overlaps(int(getattr(left, name, 0)), int(getattr(right, name, 0))):
                return False
    if str(left.key_modifier) != str(right.key_modifier):
        return False
    left_direction = str(left.direction)
    right_direction = str(right.direction)
    if left_direction != right_direction and left_direction != "ANY" and right_direction != "ANY":
        return False
    return bool(left.repeat) == bool(right.repeat)


def is_collision_consumer(
    idname: str,
    *,
    menu_name: str = "",
    session_reflect_idnames: frozenset[str] = frozenset(),
) -> bool:
    if idname in FAMILY_OPERATOR_IDNAMES:
        return True
    if idname in session_reflect_idnames:
        return True
    if idname in REPLAY_OPERATOR_IDNAMES:
        return True
    return idname == "wm.call_menu" and menu_name in REPLAY_CALL_MENU_NAMES


def count_overlapping_consumers(
    target: _KmiEvent,
    items: Iterable[tuple[_KmiEvent, str, str]],
    *,
    session_reflect_idnames: frozenset[str] = frozenset(),
) -> int:
    total = 0
    for event, idname, menu_name in items:
        if not is_collision_consumer(
            idname,
            menu_name=menu_name,
            session_reflect_idnames=session_reflect_idnames,
        ):
            continue
        if kmi_events_overlap(target, event):
            total += 1
    return total


def confirm_evidence(recorded_op: OperatorIdentity, current_op: OperatorIdentity) -> bool:
    if current_op == recorded_op:
        return False
    return current_op.idname in TARGET_OPERATOR_IDNAMES


def classify_poller_tick(
    *,
    recorded_op: OperatorIdentity,
    current_op: OperatorIdentity,
    invoke_counts: MeshCounts,
    current_counts: MeshCounts,
    modal_alive: bool,
    onset: bool,
    elapsed_s: float,
    selection_matches_s1: bool,
) -> PollerVerdict:
    if confirm_evidence(recorded_op, current_op):
        return PollerVerdict.CONFIRMED
    if modal_alive:
        return PollerVerdict.WAIT
    if onset:
        if current_op == recorded_op:
            if invoke_counts == current_counts:
                return PollerVerdict.CANCELLED
            return PollerVerdict.UNKNOWN
        return PollerVerdict.UNKNOWN
    if elapsed_s >= ONSET_TIMEOUT_S:
        if selection_matches_s1 and invoke_counts == current_counts:
            return PollerVerdict.CANCELLED
        return PollerVerdict.UNKNOWN
    return PollerVerdict.WAIT


def poller_generation_is_stale(token_generation: int, current_generation: int) -> bool:
    return token_generation != current_generation


def discard_reason(
    *,
    addon_unregistered: bool = False,
    edit_mode: bool = True,
    object_exists: bool = True,
    mesh_exists: bool = True,
    window_exists: bool = True,
    load_pre: bool = False,
) -> str | None:
    if addon_unregistered:
        return "unregister"
    if load_pre:
        return "load_pre"
    if not edit_mode:
        return "edit_mode_exit"
    if not object_exists:
        return "object_missing"
    if not mesh_exists:
        return "mesh_missing"
    if not window_exists:
        return "window_missing"
    return None


def evaluate_poller_tick(
    *,
    token_generation: int,
    current_generation: int,
    addon_unregistered: bool = False,
    edit_mode: bool = True,
    object_exists: bool = True,
    mesh_exists: bool = True,
    window_exists: bool = True,
    load_pre: bool = False,
    recorded_op: OperatorIdentity,
    current_op: OperatorIdentity,
    invoke_counts: MeshCounts,
    current_counts: MeshCounts,
    modal_alive: bool,
    onset: bool,
    elapsed_s: float,
    selection_matches_s1: bool,
) -> TickResult:
    if poller_generation_is_stale(token_generation, current_generation):
        return TickResult.NO_OP
    if (
        discard_reason(
            addon_unregistered=addon_unregistered,
            edit_mode=edit_mode,
            object_exists=object_exists,
            mesh_exists=mesh_exists,
            window_exists=window_exists,
            load_pre=load_pre,
        )
        is not None
    ):
        return TickResult.DISCARD
    return TickResult(
        classify_poller_tick(
            recorded_op=recorded_op,
            current_op=current_op,
            invoke_counts=invoke_counts,
            current_counts=current_counts,
            modal_alive=modal_alive,
            onset=onset,
            elapsed_s=elapsed_s,
            selection_matches_s1=selection_matches_s1,
        ).value
    )


def next_transaction_action(state: TransactionState, event: TransactionEvent) -> TransactionAction:
    if event is TransactionEvent.RESTORE_FAILED:
        return TransactionAction.WARN_ONLY
    if event is TransactionEvent.TOKEN_REGISTER_FAILED:
        return TransactionAction.DISCARD_S0_PASS_THROUGH
    if event is TransactionEvent.PUSH_FAILED:
        return TransactionAction.RESTORE_S0_DISCARD_TOKEN_WARN_PASS_THROUGH
    if event is TransactionEvent.ARM_FAILED:
        if state.arm_attempts < MAX_ARM_ATTEMPTS:
            return TransactionAction.ATTEMPT_ARM
        return TransactionAction.LEAVE_EXPANDED_WARN_DISCARD_TOKEN
    if event is TransactionEvent.EXCEPTION:
        if not state.selection_mutated:
            if state.token_registered:
                return TransactionAction.DISCARD_TOKEN_PASS_THROUGH
            return TransactionAction.PASS_THROUGH
        if not state.push_finished:
            return TransactionAction.RESTORE_S0_DISCARD_TOKEN_PASS_THROUGH
        if not state.poller_armed:
            return TransactionAction.ATTEMPT_ARM
        return TransactionAction.PASS_THROUGH
    raise ValueError(event)


def action_after_arm_result(*, succeeded: bool, failed_attempts: int, warn_on_success: bool) -> TransactionAction:
    if succeeded:
        return TransactionAction.WARN_PASS_THROUGH if warn_on_success else TransactionAction.PASS_THROUGH
    if failed_attempts < MAX_ARM_ATTEMPTS:
        return TransactionAction.ATTEMPT_ARM
    return TransactionAction.LEAVE_EXPANDED_WARN_DISCARD_TOKEN


def apply_invoke_preamble(prior: PriorTokenView | None, current_selection: frozenset[int]) -> PreambleResult:
    """Settle a prior token before any new-invoke work (contract section 4-0).

    Never returns a non-terminal verdict: with the target modal dead, elapsed
    time is treated as past the onset timeout so WAIT cannot leak a token.
    """
    if prior is None:
        return PreambleResult(PreambleDecision.NO_PRIOR, None, False, False, current_selection)
    if prior.discard:
        return PreambleResult(PreambleDecision.SETTLED, PollerVerdict.UNKNOWN, False, True, current_selection)
    if prior.modal_alive:
        return PreambleResult(PreambleDecision.CONSUME_EVENT, None, False, False, current_selection)
    verdict = classify_poller_tick(
        recorded_op=prior.recorded_op,
        current_op=prior.current_op,
        invoke_counts=prior.invoke_counts,
        current_counts=prior.current_counts,
        modal_alive=False,
        onset=prior.onset,
        elapsed_s=max(prior.elapsed_s, ONSET_TIMEOUT_S),
        selection_matches_s1=prior.selection_matches_s1,
    )
    if verdict is PollerVerdict.CANCELLED:
        return PreambleResult(PreambleDecision.SETTLED, verdict, True, True, prior.s0)
    return PreambleResult(PreambleDecision.SETTLED, verdict, False, True, current_selection)


def leading_domains_for_route(native_operator: str, affect: str | None) -> tuple[str, ...] | None:
    if native_operator == "mesh.inset":
        return ("FACE",)
    if native_operator != "mesh.bevel":
        return None
    if affect == "EDGES":
        return ("EDGE",)
    if affect == "VERTICES":
        return ("VERT",)
    return None


def resolve_saved_bevel_affect(kmi_properties: tuple[tuple[str, object], ...]) -> str:
    for name, value in kmi_properties:
        if name == "affect":
            return str(value)
    return DEFAULT_BEVEL_AFFECT


def is_space_tool_keymap(keymap_name: str) -> bool:
    return keymap_name.startswith(TOOL_KEYMAP_PREFIX)


@dataclass
class _InsetBevelToken:
    generation: int
    window_pointer: int
    object_name: str
    mesh_name: str
    window: bpy.types.Window | None
    area: bpy.types.Area | None
    region: bpy.types.Region | None
    s0: SelectionSnapshot
    s1: SelectionSnapshot | None
    invoke_counts: MeshCounts
    recorded_op: OperatorIdentity
    native_idname: str
    axis_index: int
    tolerance: float
    onset: bool = False
    t0: float | None = None
    armed: bool = False
    restoring: bool = False
    restore_attempts: int = 0


_ACTIVE_TOKEN: _InsetBevelToken | None = None
_ARMED_GENERATION: int | None = None
_NEXT_GENERATION = 1
_LOAD_PRE = False
_UNREGISTERED = False
_REPORTS: list[tuple[str, str]] = []


def _next_generation() -> int:
    global _NEXT_GENERATION
    generation = _NEXT_GENERATION
    _NEXT_GENERATION += 1
    return generation


def _record_report(level: str, message: str) -> None:
    _REPORTS.append((level, message))
    # Poller-side findings have no operator to report through; the console
    # line is the user-visible trace (same limitation as the knife warnings).
    print(f"ydd_symmetric_edit inset/bevel {level}: {message}")


def _operator_report(operator, level: set[str], message: str) -> None:
    kind = "WARNING" if "WARNING" in level else "ERROR" if "ERROR" in level else "INFO"
    _record_report(kind, message)
    try:
        operator.report(level, message)
    except Exception:
        traceback.print_exc()


def _identity_from_operator(operator) -> OperatorIdentity:
    if operator is None:
        return OperatorIdentity(None, None)
    raw = getattr(operator, "bl_idname", None) or getattr(operator, "idname", None)
    try:
        pointer = int(operator.as_pointer())
    except Exception:
        pointer = None
    return OperatorIdentity(normalize_operator_idname(str(raw) if raw else None), pointer)


def _capture_selection(bm) -> SelectionSnapshot:
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    return SelectionSnapshot(
        verts=frozenset(i for i in range(len(bm.verts)) if bm.verts[i].select),
        edges=frozenset(i for i in range(len(bm.edges)) if bm.edges[i].select),
        faces=frozenset(i for i in range(len(bm.faces)) if bm.faces[i].select),
        counts=MeshCounts(len(bm.verts), len(bm.edges), len(bm.faces)),
    )


def _write_selection(bm, snapshot: SelectionSnapshot) -> None:
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    # Deselect writes propagate: `face.select = False` also deselects the
    # face's edges and verts unless a selected face shares them.  A single
    # per-element pass therefore erases an edge/vert-only snapshot when the
    # trailing face pass clears every face.  Clear everything first, then
    # apply the snapshot with select-only writes (which propagate downward
    # harmlessly).
    for face in bm.faces:
        face.select = False
    for edge in bm.edges:
        edge.select = False
    for vertex in bm.verts:
        vertex.select = False
    for i in snapshot.verts:
        bm.verts[i].select = True
    for i in snapshot.edges:
        bm.edges[i].select = True
    for i in snapshot.faces:
        bm.faces[i].select = True
    bm.select_flush_mode()


def _count_selected_leading(bm, domains: tuple[str, ...]) -> int:
    total = 0
    if "VERT" in domains:
        total += sum(1 for vertex in bm.verts if vertex.select and not vertex.hide)
    if "EDGE" in domains:
        total += sum(1 for edge in bm.edges if edge.select and not edge.hide)
    if "FACE" in domains:
        total += sum(1 for face in bm.faces if face.select and not face.hide)
    return total


def _reference_alive(value: bpy.types.bpy_struct | None) -> bool:
    if value is None:
        return False
    try:
        value.as_pointer()
    except (ReferenceError, RuntimeError, AttributeError):
        return False
    return True


def _edit_mesh_for_token(token: _InsetBevelToken):
    obj = bpy.data.objects.get(token.object_name)
    if obj is None or obj.type != "MESH" or obj.mode != "EDIT":
        return None, None, None
    mesh = obj.data
    if not isinstance(mesh, bpy.types.Mesh) or mesh.name != token.mesh_name:
        return None, None, None
    bm = bmesh.from_edit_mesh(mesh)
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    return obj, mesh, bm


def _current_counts_from_token(token: _InsetBevelToken) -> MeshCounts | None:
    _obj, _mesh, bm = _edit_mesh_for_token(token)
    if bm is None:
        return None
    return MeshCounts(len(bm.verts), len(bm.edges), len(bm.faces))


def _selection_matches(token: _InsetBevelToken, snapshot: SelectionSnapshot | None) -> bool:
    if snapshot is None:
        return False
    _obj, _mesh, bm = _edit_mesh_for_token(token)
    if bm is None:
        return False
    current = _capture_selection(bm)
    return current.verts == snapshot.verts and current.edges == snapshot.edges and current.faces == snapshot.faces


def _selection_matches_s1(token: _InsetBevelToken) -> bool:
    return _selection_matches(token, token.s1)


def _read_active_operator(token: _InsetBevelToken) -> OperatorIdentity:
    window = token.window if _reference_alive(token.window) else None
    if window is None:
        return OperatorIdentity(None, None)
    area = token.area if _reference_alive(token.area) else None
    region = token.region if _reference_alive(token.region) else None
    try:
        with bpy.context.temp_override(window=window, area=area, region=region):
            return _identity_from_operator(getattr(bpy.context, "active_operator", None))
    except Exception:
        # Never fall back to the global context: another window's operator
        # would read as confirm evidence.  Unobservable classifies as UNKNOWN.
        return UNOBSERVABLE_OPERATOR


def _target_modal_alive(token: _InsetBevelToken) -> bool:
    window = token.window if _reference_alive(token.window) else None
    if window is None:
        return False
    try:
        for operator in window.modal_operators:
            identity = _identity_from_operator(operator)
            if identity.idname in TARGET_OPERATOR_IDNAMES:
                return True
    except (ReferenceError, RuntimeError, AttributeError):
        return False
    return False


def _token_discard_flags(token: _InsetBevelToken) -> dict[str, bool]:
    obj = bpy.data.objects.get(token.object_name)
    mesh = bpy.data.meshes.get(token.mesh_name)
    object_exists = obj is not None
    mesh_exists = mesh is not None
    if object_exists and mesh_exists:
        data = getattr(obj, "data", None)
        mesh_exists = data is not None and getattr(data, "name", None) == token.mesh_name
    edit_mode = bool(object_exists and getattr(obj, "mode", None) == "EDIT")
    return {
        "addon_unregistered": _UNREGISTERED,
        "edit_mode": edit_mode,
        "object_exists": object_exists,
        "mesh_exists": mesh_exists,
        "window_exists": _reference_alive(token.window),
        "load_pre": _LOAD_PRE,
    }


def _stop_poller_timer() -> None:
    global _ARMED_GENERATION
    _ARMED_GENERATION = None
    if bpy is None:
        return
    try:
        if bpy.app.timers.is_registered(_poller_timer):
            bpy.app.timers.unregister(_poller_timer)
    except Exception:
        traceback.print_exc()


def _discard_active_token() -> None:
    global _ACTIVE_TOKEN
    _ACTIVE_TOKEN = None
    _stop_poller_timer()


def _restore_s0(token: _InsetBevelToken) -> bool:
    obj, mesh, bm = _edit_mesh_for_token(token)
    if obj is None or mesh is None or bm is None:
        return False
    current = MeshCounts(len(bm.verts), len(bm.edges), len(bm.faces))
    if current != token.invoke_counts:
        return False
    _write_selection(bm, token.s0)
    # No update_edit_mesh / redraw tag here: triggering a redraw in the
    # post-bevel-cancel state stomps the freshly written select flags
    # (measured).  The verify loop tags a redraw only after the write
    # has been observed to survive.
    return True


def _tag_view3d_redraw(token: _InsetBevelToken) -> None:
    window = token.window if _reference_alive(token.window) else None
    if window is None:
        return
    try:
        for area in window.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()
    except (ReferenceError, RuntimeError, AttributeError):
        pass


def _count_unmirrored_vertices(bm, axis_index: int, tolerance: float) -> int:
    from . import matching

    coords = [vertex.co.copy() for vertex in bm.verts]
    # Mirror-existence check (many-to-one allowed): the involutive pair table
    # drops ambiguous assignments near duplicated coordinates and reported
    # symmetric results as unmirrored.
    lookup = matching.build_vertex_mirror_lookup(coords, axis_index, tolerance)
    return sum(1 for co in coords if lookup.find(co) is None)


def _run_confirm_diagnostic(token: _InsetBevelToken, operator=None) -> None:
    _obj, _mesh, bm = _edit_mesh_for_token(token)
    if bm is None:
        return
    if len(bm.verts) > DIAGNOSTIC_VERTEX_LIMIT:
        if operator is not None:
            _operator_report(operator, {"INFO"}, DIAGNOSTIC_SKIP_INFO)
        else:
            _record_report("INFO", DIAGNOSTIC_SKIP_INFO)
        return
    try:
        missing = _count_unmirrored_vertices(bm, token.axis_index, token.tolerance)
    except Exception:
        traceback.print_exc()
        return
    if missing <= 0:
        return
    message = DIAGNOSTIC_WARNING.format(count=missing)
    if operator is not None:
        _operator_report(operator, {"WARNING"}, message)
    else:
        _record_report("WARNING", message)


def _apply_settled_verdict(token: _InsetBevelToken, verdict: PollerVerdict, *, staged_restore: bool = False) -> None:
    if verdict is PollerVerdict.CONFIRMED:
        _run_confirm_diagnostic(token)
        _discard_active_token()
        return
    if verdict is PollerVerdict.CANCELLED:
        try:
            _restore_s0(token)
        except Exception:
            traceback.print_exc()
            _record_report("WARNING", RESTORE_FAILED_WARNING)
            _discard_active_token()
            return
        if staged_restore:
            # Keep the token: verify on later ticks that the write survived
            # the native cancel cleanup, rewriting if it was stomped.
            token.restoring = True
            token.restore_attempts = 0
            return
        _discard_active_token()
        return
    _discard_active_token()


def _prior_token_view(token: _InsetBevelToken) -> PriorTokenView:
    flags = _token_discard_flags(token)
    discard = discard_reason(**flags) is not None
    counts = _current_counts_from_token(token)
    if counts is None:
        discard = True
        counts = token.invoke_counts
    return PriorTokenView(
        onset=token.onset,
        modal_alive=False if discard else _target_modal_alive(token),
        recorded_op=token.recorded_op,
        current_op=UNOBSERVABLE_OPERATOR if discard else _read_active_operator(token),
        invoke_counts=token.invoke_counts,
        current_counts=counts,
        selection_matches_s1=False if discard else _selection_matches_s1(token),
        elapsed_s=0.0 if token.t0 is None else time.monotonic() - token.t0,
        s0=token.s0.verts,
        discard=discard,
    )


def _settle_prior_token_sync() -> Literal["none", "consume", "settled"]:
    token = _ACTIVE_TOKEN
    if token is None:
        return "none"
    result = apply_invoke_preamble(_prior_token_view(token), frozenset())
    if result.decision is PreambleDecision.CONSUME_EVENT:
        return "consume"
    if result.verdict is PollerVerdict.CONFIRMED:
        _run_confirm_diagnostic(token)
    if result.restored_s0:
        try:
            _restore_s0(token)
        except Exception:
            traceback.print_exc()
            _record_report("WARNING", RESTORE_FAILED_WARNING)
    _discard_active_token()
    return "settled"


def _register_dormant_token(
    *,
    context,
    obj,
    s0: SelectionSnapshot,
    recorded_op: OperatorIdentity,
    native_idname: str,
    axis_index: int,
    tolerance: float,
) -> _InsetBevelToken | None:
    global _ACTIVE_TOKEN
    window = getattr(context, "window", None)
    if window is None:
        return None
    try:
        window_pointer = int(window.as_pointer())
    except Exception:
        return None
    token = _InsetBevelToken(
        generation=_next_generation(),
        window_pointer=window_pointer,
        object_name=obj.name,
        mesh_name=obj.data.name,
        window=window,
        area=getattr(context, "area", None),
        region=getattr(context, "region", None),
        s0=s0,
        s1=None,
        invoke_counts=s0.counts,
        recorded_op=recorded_op,
        native_idname=native_idname,
        axis_index=axis_index,
        tolerance=tolerance,
    )
    _ACTIVE_TOKEN = token
    return token


def _arm_poller(token: _InsetBevelToken) -> bool:
    global _ARMED_GENERATION
    token.t0 = time.monotonic()
    token.armed = True
    _ARMED_GENERATION = token.generation
    try:
        if not bpy.app.timers.is_registered(_poller_timer):
            bpy.app.timers.register(_poller_timer, first_interval=POLLER_FIRST_INTERVAL)
        return True
    except Exception:
        traceback.print_exc()
        token.armed = False
        _ARMED_GENERATION = None
        return False


def _poller_timer() -> float | None:
    # Timer callbacks that raise are dropped by Blender; an unguarded
    # exception would leak the token with no observer.
    try:
        token = _ACTIVE_TOKEN
        if token is None or _ARMED_GENERATION is None or token.generation != _ARMED_GENERATION:
            return None
        if token.restoring:
            if discard_reason(**_token_discard_flags(token)) is not None:
                _discard_active_token()
                return None
            if _selection_matches(token, token.s0):
                _tag_view3d_redraw(token)
                _discard_active_token()
                return None
            if token.restore_attempts >= RESTORE_VERIFY_ATTEMPTS:
                # Leave whatever selection survived: the expanded selection is
                # still symmetric, so not restoring is the safe degradation.
                _record_report("WARNING", RESTORE_FAILED_WARNING)
                _discard_active_token()
                return None
            token.restore_attempts += 1
            if not _restore_s0(token):
                _record_report("WARNING", RESTORE_FAILED_WARNING)
                _discard_active_token()
                return None
            return min(
                RESTORE_VERIFY_BASE_INTERVAL * token.restore_attempts,
                RESTORE_VERIFY_MAX_INTERVAL,
            )
        flags = _token_discard_flags(token)
        counts = _current_counts_from_token(token)
        elapsed = 0.0 if token.t0 is None else time.monotonic() - token.t0
        if counts is None:
            flags["mesh_exists"] = False
            counts = token.invoke_counts
            modal_alive = False
            current_op = UNOBSERVABLE_OPERATOR
            sel_matches = False
        else:
            modal_alive = _target_modal_alive(token)
            current_op = _read_active_operator(token)
            # The S1 comparison walks the whole mesh; only the no-onset
            # timeout branch consumes it, so skip it everywhere else.
            sel_matches = (
                _selection_matches_s1(token)
                if (not modal_alive and not token.onset and elapsed >= ONSET_TIMEOUT_S)
                else False
            )
        result = evaluate_poller_tick(
            token_generation=token.generation,
            current_generation=_ARMED_GENERATION,
            recorded_op=token.recorded_op,
            current_op=current_op,
            invoke_counts=token.invoke_counts,
            current_counts=counts,
            modal_alive=modal_alive,
            onset=token.onset,
            elapsed_s=elapsed,
            selection_matches_s1=sel_matches,
            **flags,
        )
        if result is TickResult.WAIT:
            if modal_alive:
                token.onset = True
            return POLLER_INTERVAL
        if result is TickResult.CONFIRMED:
            _apply_settled_verdict(token, PollerVerdict.CONFIRMED)
            return None
        if result is TickResult.CANCELLED:
            _apply_settled_verdict(token, PollerVerdict.CANCELLED, staged_restore=True)
            return POLLER_INTERVAL if _ACTIVE_TOKEN is token else None
        _discard_active_token()
        return None
    except Exception:
        traceback.print_exc()
        try:
            _discard_active_token()
        except Exception:
            traceback.print_exc()
        return None


def _read_only_symmetry_parameters(context) -> tuple[bpy.types.Object, int, float] | None:
    """replay._symmetry_parameters minus the touched-mesh recording.

    This route creates no temporary layers, so it must not enlarge the
    save-time attribute sweep scope (isolation rule, contract section 2).
    """
    from . import matching

    obj = context.edit_object
    if obj is None or obj.type != "MESH":
        return None
    if len(context.objects_in_mode_unique_data) != 1:
        return None
    axes = matching.enabled_mesh_symmetry_axes(obj)
    if len(axes) != 1:
        return None
    _axis_name, axis_index = axes[0]
    settings = context.scene.ydd_symmetric_edit
    return obj, axis_index, float(settings.tolerance)


def _resolve_tool_bevel_affect(context) -> str | None:
    try:
        workspace = getattr(context, "workspace", None)
        if workspace is None:
            return None
        tools = getattr(workspace, "tools", None)
        if tools is None:
            return None
        tool = tools.from_space_view3d_mode("EDIT_MESH", create=False)
        if tool is None:
            return None
        properties = tool.operator_properties("mesh.bevel")
        if properties is None:
            return None
        affect = getattr(properties, "affect", None)
        if affect is None:
            return None
        return str(affect)
    except Exception:
        return None


def _undo_push_finished(result: object) -> bool:
    if isinstance(result, (set, frozenset)):
        return set(result) == {"FINISHED"}
    return False


@persistent
def _on_load_pre(_dummy) -> None:
    global _LOAD_PRE
    _LOAD_PRE = True
    _discard_active_token()


@persistent
def _on_load_post(_dummy) -> None:
    # Without this release the load_pre latch would discard every future
    # token until the addon is re-registered.
    global _LOAD_PRE
    _LOAD_PRE = False


def install_runtime_hooks() -> None:
    global _UNREGISTERED, _LOAD_PRE
    _UNREGISTERED = False
    _LOAD_PRE = False
    if bpy is None:
        return
    if _on_load_pre not in bpy.app.handlers.load_pre:
        bpy.app.handlers.load_pre.append(_on_load_pre)
    if _on_load_post not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(_on_load_post)


def cleanup_runtime() -> None:
    global _UNREGISTERED, _LOAD_PRE
    _UNREGISTERED = True
    _discard_active_token()
    if bpy is None:
        _LOAD_PRE = False
        _REPORTS.clear()
        return
    for handler_list, handler in (
        (bpy.app.handlers.load_pre, _on_load_pre),
        (bpy.app.handlers.load_post, _on_load_post),
    ):
        try:
            if handler in handler_list:
                handler_list.remove(handler)
        except Exception:
            traceback.print_exc()
    _LOAD_PRE = False
    _REPORTS.clear()


_OperatorBase = bpy.types.Operator if bpy is not None else object


class MESH_OT_ydd_symmetric_edit_inset_bevel_intercept(_OperatorBase):
    """Expand a symmetric selection, then let the native Inset or Bevel event continue."""

    bl_idname = "mesh.ydd_symmetric_edit_inset_bevel_intercept"
    bl_label = "Prepare ydd Symmetric Edit for Inset/Bevel"
    bl_description = "Expand the mirrored selection, then pass the native Inset or Bevel event through"
    bl_options = {"INTERNAL"}

    if TYPE_CHECKING:
        route_key: str
    elif StringProperty is not None:
        route_key: StringProperty(options={"HIDDEN", "SKIP_SAVE"})

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        return context.mode == "EDIT_MESH"

    def invoke(self, context: bpy.types.Context, event: bpy.types.Event) -> set[OperatorReturnItems]:
        del event
        if _settle_prior_token_sync() == "consume":
            return {"CANCELLED"}
        state = TransactionState()
        try:
            return self._invoke_body(context, state)
        except Exception:
            traceback.print_exc()
            return self._apply_transaction_action(
                context,
                state,
                next_transaction_action(state, TransactionEvent.EXCEPTION),
            )

    def _invoke_body(self, context, state: TransactionState):
        from . import element_pairs, keymaps, matching, session_state

        route = keymaps.inset_bevel_route(self.route_key)
        if route is None or not keymaps.inset_bevel_route_is_current(self.route_key):
            return {"PASS_THROUGH"}
        if session_state.sessions_active():
            return {"PASS_THROUGH"}
        objects_in_mode = getattr(context, "objects_in_mode_unique_data", None)
        if objects_in_mode is None or len(objects_in_mode) != 1:
            return {"PASS_THROUGH"}
        obj = context.edit_object
        if obj is None or obj.type != "MESH":
            return {"PASS_THROUGH"}
        if len(matching.enabled_mesh_symmetry_axes(obj)) != 1:
            return {"PASS_THROUGH"}

        affect: str | None = None
        if route.native_operator == "mesh.bevel":
            if route.is_tool:
                affect = _resolve_tool_bevel_affect(context)
                if affect is None:
                    return {"PASS_THROUGH"}
            else:
                affect = resolve_saved_bevel_affect(route.kmi_properties)
        domains = leading_domains_for_route(route.native_operator, affect)
        if domains is None:
            return {"PASS_THROUGH"}

        symmetry = _read_only_symmetry_parameters(context)
        if symmetry is None:
            return {"PASS_THROUGH"}
        obj, axis_index, tolerance = symmetry
        mesh = obj.data
        if not isinstance(mesh, bpy.types.Mesh):
            return {"PASS_THROUGH"}
        bm = bmesh.from_edit_mesh(mesh)
        pair_maps = element_pairs.build_element_pair_maps(bm, axis_index, tolerance, mesh_object=obj)
        plan = element_pairs.plan_leading_domain_expansion(bm, pair_maps, domains=domains)

        if plan.hidden_counterpart_count > 0:
            _operator_report(
                self,
                {"WARNING"},
                HIDDEN_WARNING.format(count=plan.hidden_counterpart_count),
            )
            return {"CANCELLED"}
        if plan.unmatched_count > 0:
            _operator_report(self, {"WARNING"}, UNMATCHED_WARNING)
            return {"PASS_THROUGH"}
        if _count_selected_leading(bm, domains) == 0:
            return {"PASS_THROUGH"}
        if not plan.add_vert_indices and not plan.add_edge_indices and not plan.add_face_indices:
            return {"PASS_THROUGH"}

        try:
            s0 = _capture_selection(bm)
            state = replace(state, snapshot_exists=True)
            token = _register_dormant_token(
                context=context,
                obj=obj,
                s0=s0,
                recorded_op=_identity_from_operator(getattr(context, "active_operator", None)),
                native_idname=route.native_operator,
                axis_index=axis_index,
                tolerance=tolerance,
            )
            if token is None:
                return self._apply_transaction_action(
                    context,
                    state,
                    next_transaction_action(state, TransactionEvent.TOKEN_REGISTER_FAILED),
                )
            state = replace(state, token_registered=True)

            # A partially applied plan must already restore S0, so flag the
            # mutation before the first select write, not after the flush.
            state = replace(state, selection_mutated=True)
            element_pairs.apply_expansion_plan(bm, plan)
            bm.select_flush_mode()
            bmesh.update_edit_mesh(mesh, loop_triangles=False, destructive=False)
            token.s1 = _capture_selection(bm)

            push_result = bpy.ops.ed.undo_push(message=UNDO_PUSH_MESSAGE)
            if not _undo_push_finished(push_result):
                return self._apply_transaction_action(
                    context,
                    state,
                    next_transaction_action(state, TransactionEvent.PUSH_FAILED),
                )
            state = replace(state, push_finished=True)

            return self._arm_or_leave(context, state, token, warn_on_success=False)
        except Exception:
            traceback.print_exc()
            action = next_transaction_action(state, TransactionEvent.EXCEPTION)
            return self._apply_transaction_action(context, state, action)

    def _arm_or_leave(self, context, state: TransactionState, token: _InsetBevelToken, *, warn_on_success: bool):
        failed_attempts = state.arm_attempts
        while True:
            if _arm_poller(token):
                state = replace(state, poller_armed=True)
                action = action_after_arm_result(
                    succeeded=True,
                    failed_attempts=failed_attempts,
                    warn_on_success=warn_on_success,
                )
                if action is TransactionAction.WARN_PASS_THROUGH:
                    _operator_report(self, {"WARNING"}, ARM_FAILED_WARNING)
                return {"PASS_THROUGH"}
            failed_attempts += 1
            state = replace(state, arm_attempts=failed_attempts)
            action = next_transaction_action(state, TransactionEvent.ARM_FAILED)
            if action is TransactionAction.ATTEMPT_ARM:
                continue
            _operator_report(self, {"WARNING"}, ARM_FAILED_WARNING)
            _discard_active_token()
            return {"PASS_THROUGH"}

    def _apply_transaction_action(self, context, state: TransactionState, action: TransactionAction):
        token = _ACTIVE_TOKEN
        if action is TransactionAction.PASS_THROUGH:
            if state.token_registered and not state.poller_armed:
                _discard_active_token()
            return {"PASS_THROUGH"}
        if action is TransactionAction.DISCARD_TOKEN_PASS_THROUGH:
            _discard_active_token()
            return {"PASS_THROUGH"}
        if action is TransactionAction.DISCARD_S0_PASS_THROUGH:
            _discard_active_token()
            return {"PASS_THROUGH"}
        if action in {
            TransactionAction.RESTORE_S0_DISCARD_TOKEN_PASS_THROUGH,
            TransactionAction.RESTORE_S0_DISCARD_TOKEN_WARN_PASS_THROUGH,
        }:
            if action is TransactionAction.RESTORE_S0_DISCARD_TOKEN_WARN_PASS_THROUGH:
                _operator_report(self, {"WARNING"}, PUSH_FAILED_WARNING)
            if token is not None:
                try:
                    if not _restore_s0(token):
                        _operator_report(self, {"WARNING"}, RESTORE_FAILED_WARNING)
                except Exception:
                    traceback.print_exc()
                    _operator_report(self, {"WARNING"}, RESTORE_FAILED_WARNING)
            _discard_active_token()
            return {"PASS_THROUGH"}
        if action is TransactionAction.ATTEMPT_ARM:
            if token is None:
                return {"PASS_THROUGH"}
            return self._arm_or_leave(context, state, token, warn_on_success=True)
        if action is TransactionAction.WARN_PASS_THROUGH:
            _operator_report(self, {"WARNING"}, ARM_FAILED_WARNING)
            return {"PASS_THROUGH"}
        if action is TransactionAction.LEAVE_EXPANDED_WARN_DISCARD_TOKEN:
            _operator_report(self, {"WARNING"}, ARM_FAILED_WARNING)
            _discard_active_token()
            return {"PASS_THROUGH"}
        if action is TransactionAction.WARN_ONLY:
            _operator_report(self, {"WARNING"}, RESTORE_FAILED_WARNING)
            return {"PASS_THROUGH"}
        return {"PASS_THROUGH"}


CLASSES = (MESH_OT_ydd_symmetric_edit_inset_bevel_intercept,)
