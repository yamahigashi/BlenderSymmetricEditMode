# SPDX-License-Identifier: GPL-3.0-or-later

"""Inset Faces / Bevel EXPAND_PASSTHROUGH intercept, poller, and pure helpers."""

from __future__ import annotations

import time
import traceback
from collections.abc import Iterable, Mapping
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
BEVEL_FINGERPRINT_PROPS = (
    "offset", "offset_pct", "segments", "profile", "spread", "affect",
    "offset_type", "profile_type", "miter_outer", "miter_inner", "vmesh_method",
    "face_strength_mode", "material", "use_clamp_overlap", "loop_slide",
    "mark_seam", "mark_sharp", "harden_normals",
)
INSET_REPLAY_PROPS = (
    "thickness",
    "depth",
    "use_boundary",
    "use_even_offset",
    "use_relative_offset",
    "use_edge_rail",
    "use_outset",
    "use_select_inset",
    "use_individual",
    "use_interpolate",
)
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
HIDDEN_WARNING = "Inset/Bevel declined: {count} hidden counterpart(s)"
UNMATCHED_WARNING = "Inset/Bevel will run on one side only; the mesh is not symmetric"
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
    ARM_FAILED = "arm_failed"
    RESTORE_FAILED = "restore_failed"
    S1_CAPTURE_FAILED = "s1_capture_failed"


class TransactionAction(StrEnum):
    PASS_THROUGH = "pass_through"
    DISCARD_TOKEN_PASS_THROUGH = "discard_token_pass_through"
    DISCARD_S0_PASS_THROUGH = "discard_s0_pass_through"
    RESTORE_S0_DISCARD_TOKEN_PASS_THROUGH = "restore_s0_discard_token_pass_through"
    RESTORE_S0_DISCARD_TOKEN_WARN_PASS_THROUGH = "restore_s0_discard_token_warn_pass_through"
    ATTEMPT_ARM = "attempt_arm"
    WARN_PASS_THROUGH = "warn_pass_through"
    WARN_ONLY = "warn_only"
    FAIL_CLOSED = "fail_closed"


class PreambleDecision(StrEnum):
    NO_PRIOR = "NO_PRIOR"
    CONSUME_EVENT = "CONSUME_EVENT"
    SETTLED = "SETTLED"


class SelectionRelation(StrEnum):
    SELF_MIRRORED = "SELF_MIRRORED"
    DISJOINT = "DISJOINT"
    PARTIAL = "PARTIAL"
    UNMATCHED = "UNMATCHED"


class InsetReplayState(StrEnum):
    WATCHING = "WATCHING"
    REPLAY_READY = "REPLAY_READY"
    EXECUTING = "EXECUTING"
    EXEC_COMMITTED = "EXEC_COMMITTED"
    PUSHED = "PUSHED"
    ABORTED_ONE_SIDED = "ABORTED_ONE_SIDED"
    DEGRADED_SYMMETRIC = "DEGRADED_SYMMETRIC"


class InsetReplayEvent(StrEnum):
    CONFIRMED = "CONFIRMED"
    PROPS_READY = "PROPS_READY"
    EXEC_FINISHED = "EXEC_FINISHED"
    REPLAY_FAILED = "REPLAY_FAILED"
    POSTPROCESS_FAILED = "POSTPROCESS_FAILED"
    PUSH_FINISHED = "PUSH_FINISHED"


class CompletionStatus(StrEnum):
    ACTIVE = "ACTIVE"
    SUSPENDED = "SUSPENDED"


class F9Decision(StrEnum):
    NO_OP = "NO_OP"
    INTERVENE = "INTERVENE"
    SUSPEND = "SUSPEND"


@dataclass(frozen=True, slots=True)
class CompletionRecord:
    window_pointer: int
    object_name: str
    mesh_name: str
    route: str
    operator: OperatorIdentity
    fingerprint: tuple
    status: CompletionStatus = CompletionStatus.ACTIVE


@dataclass(frozen=True, slots=True)
class F9Classification:
    decision: F9Decision
    record: CompletionRecord | None
    warning: str | None = None


def _enum_identifier(value):
    identifier = getattr(value, "identifier", None)
    return str(identifier) if identifier is not None else str(value)


def _operator_property_value(operator, name: str):
    properties = getattr(operator, "properties", None)
    if properties is None:
        return None, False
    try:
        rna = getattr(properties, "bl_rna", None)
        definition = rna.properties.get(name) if rna is not None else None
        if definition is not None and bool(getattr(definition, "is_readonly", False)):
            return None, False
        if definition is None and not hasattr(properties, name):
            return None, False
        value = getattr(properties, name)
    except (AttributeError, KeyError, TypeError, RuntimeError):
        return None, False
    if isinstance(value, bool) or isinstance(value, int):
        return value, True
    if isinstance(value, float):
        return round(value, 6), True
    if isinstance(value, str):
        return value, True
    return _enum_identifier(value), True


def fingerprint(operator, *, route: str | None = None, override=None) -> tuple:
    """Return the single v5.2 normalized operator fingerprint.

    ``override`` is a saved-window context manager supplied by the runtime;
    pure callers leave it as ``None``.
    """
    if operator is None:
        return ()
    idname = normalize_operator_idname(
        getattr(operator, "bl_idname", None) or getattr(operator, "idname", None)
    )
    names = INSET_REPLAY_PROPS if (route or idname) == "mesh.inset" else BEVEL_FINGERPRINT_PROPS
    def read():
        values = []
        for name in names:
            value, present = _operator_property_value(operator, name)
            if present:
                values.append((name, value))
        return tuple(values)
    if override is None:
        return read()
    try:
        with override:
            return read()
    except Exception:
        return ()


def classify_f9_intervention(
    active_operator,
    record: CompletionRecord | None,
    *,
    window_pointer: int | None,
    object_name: str | None,
    mesh_name: str | None,
    route: str | None = None,
) -> F9Classification:
    """Pure §5C-2 discriminator; no mesh/token mutation is performed."""
    if record is None or record.status is CompletionStatus.SUSPENDED:
        return F9Classification(F9Decision.NO_OP, record)
    if active_operator is None:
        return F9Classification(F9Decision.SUSPEND, replace(record, status=CompletionStatus.SUSPENDED))
    active_id = normalize_operator_idname(
        getattr(active_operator, "bl_idname", None) or getattr(active_operator, "idname", None)
    )
    if active_id not in TARGET_OPERATOR_IDNAMES:
        return F9Classification(F9Decision.NO_OP, record)
    if (window_pointer, object_name, mesh_name, active_id) != (
        record.window_pointer, record.object_name, record.mesh_name, record.operator.idname
    ):
        return F9Classification(F9Decision.NO_OP, record)
    current = fingerprint(active_operator, route=route or active_id)
    if current == record.fingerprint:
        return F9Classification(F9Decision.SUSPEND, replace(record, status=CompletionStatus.SUSPENDED))
    return F9Classification(F9Decision.INTERVENE, record)


def reactivate_completion_record(record: CompletionRecord | None, active_operator, *, window_pointer: int | None, object_name: str | None, mesh_name: str | None) -> CompletionRecord | None:
    if record is None or active_operator is None:
        return record
    active_id = normalize_operator_idname(
        getattr(active_operator, "bl_idname", None) or getattr(active_operator, "idname", None)
    )
    if record.object_name != object_name or record.mesh_name != mesh_name or record.window_pointer != window_pointer or active_id != record.operator.idname:
        return record
    return replace(record, status=CompletionStatus.ACTIVE, operator=_identity_from_operator(active_operator))


def supersede_decision(*, repeat_origin: bool, restore_only: bool, mode: str, replay_state: InsetReplayState, same_context: bool = True) -> str:
    if not same_context:
        return "NO_OP_WARNING"
    if restore_only:
        return "RESTORE_BEFORE_SUPERSEDE"
    if repeat_origin and replay_state is InsetReplayState.WATCHING:
        return "SUPERSEDE"
    if mode == "inset" and replay_state in {InsetReplayState.REPLAY_READY, InsetReplayState.EXECUTING, InsetReplayState.EXEC_COMMITTED}:
        return "NO_OP_WARNING"
    return "NO_OP_WARNING"


def restore_only_required(*, restore_only: bool, verdict: PollerVerdict) -> bool:
    return bool(restore_only)


def completion_record_valid(record: CompletionRecord | None, *, window_exists: bool, object_exists: bool, mesh_exists: bool) -> bool:
    return record is not None and window_exists and object_exists and mesh_exists


def transaction_failure_action(*, snapshot_exists: bool, token_registered: bool, selection_mutated: bool, poller_armed: bool, arm_attempts: int = 0) -> TransactionAction:
    state = TransactionState(snapshot_exists, token_registered, selection_mutated, poller_armed, arm_attempts)
    event = TransactionEvent.ARM_FAILED if arm_attempts else TransactionEvent.EXCEPTION
    return next_transaction_action(state, event)


# Contract-facing aliases keep the pure API stable for Blender-free tests.
fingerprint_operator = fingerprint
classify_undo_post = classify_f9_intervention
redo_reactivate = reactivate_completion_record
supersede_token = supersede_decision
restore_only_should_restore = restore_only_required


def counts_strictly_increased(before: MeshCounts, after: MeshCounts) -> bool:
    """Return true only when no topology count shrank and one count grew."""

    previous = (before.verts, before.edges, before.faces)
    current = (after.verts, after.edges, after.faces)
    return all(new >= old for old, new in zip(previous, current, strict=True)) and any(
        new > old for old, new in zip(previous, current, strict=True)
    )


def _pair_for(pair_maps, domain: str):
    return {
        "VERT": getattr(pair_maps, "vert_pairs", {}),
        "EDGE": getattr(pair_maps, "edge_pair_by_index", {}),
        "FACE": getattr(pair_maps, "face_pair_by_index", {}),
    }[domain]


def classify_selection_relation(
    selected: Mapping[str, Iterable[int]],
    pair_maps,
    *,
    domains: tuple[str, ...],
) -> tuple[SelectionRelation, bool]:
    """Classify S versus rho(S), also reporting self-mirror/off-plane mixing.

    The second result is true when an on-plane (self-mirror) element and an
    off-plane selected element coexist.  Unmatched elements are reported
    before overlap classification, as required by the intercept guard order.
    """

    self_mirror = False
    off_plane: dict[str, set[int]] = {domain: set() for domain in domains}
    mirror: dict[str, set[int]] = {domain: set() for domain in domains}
    for domain in domains:
        pairs = _pair_for(pair_maps, domain)
        for raw_index in selected.get(domain, ()):
            index = int(raw_index)
            partner = pairs.get(index)
            if partner is None:
                return SelectionRelation.UNMATCHED, False
            if int(partner) == index:
                self_mirror = True
                continue
            off_plane[domain].add(index)
            mirror[domain].add(int(partner))
    # ``off_plane`` contains S; compare only the off-plane part so plane
    # elements do not force every selection into PARTIAL.
    any_off_plane = any(off_plane.values())
    if not any_off_plane:
        return SelectionRelation.SELF_MIRRORED, self_mirror
    intersection = any(
        bool(off_plane[domain] & mirror[domain]) for domain in domains
    )
    complete = all(off_plane[domain] == mirror[domain] for domain in domains)
    if complete:
        relation = SelectionRelation.SELF_MIRRORED
    elif not intersection:
        relation = SelectionRelation.DISJOINT
    else:
        relation = SelectionRelation.PARTIAL
    return relation, bool(self_mirror and any_off_plane)


def inset_replay_relation(
    selected: Mapping[str, Iterable[int]], pair_maps, *, domains: tuple[str, ...] = ("FACE",)
) -> tuple[SelectionRelation, bool]:
    return classify_selection_relation(selected, pair_maps, domains=domains)


def next_inset_replay_state(state: InsetReplayState, event: InsetReplayEvent) -> InsetReplayState:
    """Pure state transition table for §5A; committed work is never one-sided."""

    transitions = {
        (InsetReplayState.WATCHING, InsetReplayEvent.CONFIRMED): InsetReplayState.REPLAY_READY,
        (InsetReplayState.REPLAY_READY, InsetReplayEvent.PROPS_READY): InsetReplayState.EXECUTING,
        (InsetReplayState.REPLAY_READY, InsetReplayEvent.REPLAY_FAILED): InsetReplayState.ABORTED_ONE_SIDED,
        (InsetReplayState.EXECUTING, InsetReplayEvent.EXEC_FINISHED): InsetReplayState.EXEC_COMMITTED,
        (InsetReplayState.EXECUTING, InsetReplayEvent.REPLAY_FAILED): InsetReplayState.ABORTED_ONE_SIDED,
        (InsetReplayState.EXEC_COMMITTED, InsetReplayEvent.PUSH_FINISHED): InsetReplayState.PUSHED,
        (InsetReplayState.EXEC_COMMITTED, InsetReplayEvent.POSTPROCESS_FAILED): InsetReplayState.DEGRADED_SYMMETRIC,
    }
    try:
        return transitions[(state, event)]
    except KeyError:
        raise ValueError(f"invalid inset replay transition: {state} + {event}") from None


def classify_element_side(
    coordinates: Iterable[Iterable[float]], axis_index: int, tolerance: float, user_sign: int
) -> str:
    values = [float(tuple(coordinate)[axis_index]) for coordinate in coordinates]
    if not values or all(abs(value) <= tolerance for value in values):
        return "PLANE"
    has_positive = any(value > tolerance for value in values)
    has_negative = any(value < -tolerance for value in values)
    if has_positive and has_negative:
        return "SPAN"
    if user_sign >= 0:
        if all(value >= -tolerance for value in values):
            return "USER"
        if all(value <= tolerance for value in values):
            return "MIRROR"
    else:
        if all(value <= tolerance for value in values):
            return "USER"
        if all(value >= -tolerance for value in values):
            return "MIRROR"
    return "SPAN"


def normalize_bevel_selection(
    selected: Mapping[str, Iterable[int]],
    element_classes: Mapping[tuple[str, int], str],
    *,
    select_mirrored: bool,
    manual_both_sides: bool = False,
) -> dict[str, frozenset[int]]:
    if select_mirrored or manual_both_sides:
        return {domain: frozenset(map(int, selected.get(domain, ()))) for domain in ("VERT", "EDGE", "FACE")}
    return {
        domain: frozenset(
            index
            for index in map(int, selected.get(domain, ()))
            if element_classes.get((domain, index)) in {"USER", "SPAN", "PLANE"}
        )
        for domain in ("VERT", "EDGE", "FACE")
    }


# Descriptive aliases keep the pure contract vocabulary discoverable to tests
# and callers without duplicating any classification logic.
classify_selection_overlap = classify_selection_relation
classify_bevel_element_side = classify_element_side
inset_replay_state_transition = next_inset_replay_state
bevel_selection_normalize = normalize_bevel_selection


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
class SelectionContextSnapshot:
    selection: SelectionSnapshot
    history: tuple[tuple[str, int], ...] = ()
    active: tuple[str, int] | None = None


@dataclass(frozen=True, slots=True)
class TransactionState:
    snapshot_exists: bool = False
    token_registered: bool = False
    selection_mutated: bool = False
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
    onset_op: OperatorIdentity | None = None
    mode: Literal["inset", "bevel"] = "bevel"
    replay_state: InsetReplayState = InsetReplayState.WATCHING


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


def confirm_evidence(
    recorded_op: OperatorIdentity,
    current_op: OperatorIdentity,
    *,
    onset: bool = False,
    onset_op: OperatorIdentity | None = None,
    invoke_counts: MeshCounts | None = None,
    current_counts: MeshCounts | None = None,
) -> bool:
    if current_op.idname not in TARGET_OPERATOR_IDNAMES or current_op == recorded_op:
        return False
    if onset:
        # After an observed modal onset, lineage is the onset identity, not
        # merely the target idname (which another operator may reuse).
        return onset_op is None or current_op == onset_op
    if invoke_counts is None or current_counts is None:
        return False
    return counts_strictly_increased(invoke_counts, current_counts)


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
    onset_op: OperatorIdentity | None = None,
    mode: str = "bevel",
) -> PollerVerdict:
    if mode == "inset" and not counts_strictly_increased(invoke_counts, current_counts):
        if invoke_counts != current_counts:
            return PollerVerdict.UNKNOWN
    if modal_alive:
        return PollerVerdict.WAIT
    if mode == "inset":
        # Counts are the primary discriminator for inset: a confirmed inset
        # always adds geometry (RG8, even zero-drag) and a cancel restores the
        # invoke counts.  Operator identity comparisons are ambiguous here:
        # consecutive same-type runs can reuse a freed operator pointer, so
        # run N's active operator can compare equal to run N-1's recorded_op
        # and a real confirm would silently classify as CANCELLED.
        if counts_strictly_increased(invoke_counts, current_counts):
            if current_op.idname == "mesh.inset":
                return PollerVerdict.CONFIRMED
            return PollerVerdict.UNKNOWN
        if onset or elapsed_s >= ONSET_TIMEOUT_S:
            if invoke_counts == current_counts:
                return PollerVerdict.CANCELLED
            return PollerVerdict.UNKNOWN
        return PollerVerdict.WAIT
    # Bevel: identity comparisons stay (zero-drag confirms keep the counts
    # unchanged, RG9), but a count increase with the target operator active
    # is decisive on its own — consecutive runs can reuse a freed operator
    # pointer, and identity equality must not veto a real confirm (the
    # normalization duty makes a silent UNKNOWN harmful here).
    if (
        counts_strictly_increased(invoke_counts, current_counts)
        and current_op.idname == "mesh.bevel"
    ):
        return PollerVerdict.CONFIRMED
    if confirm_evidence(
        recorded_op,
        current_op,
        onset=onset,
        onset_op=onset_op,
        invoke_counts=invoke_counts,
        current_counts=current_counts,
    ):
        return PollerVerdict.CONFIRMED
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
    onset_op: OperatorIdentity | None = None,
    mode: str = "bevel",
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
            onset_op=onset_op,
            mode=mode,
        ).value
    )


def next_transaction_action(state: TransactionState, event: TransactionEvent) -> TransactionAction:
    if event is TransactionEvent.RESTORE_FAILED:
        return TransactionAction.WARN_ONLY
    if event is TransactionEvent.TOKEN_REGISTER_FAILED:
        return TransactionAction.DISCARD_S0_PASS_THROUGH
    if event is TransactionEvent.ARM_FAILED:
        if state.arm_attempts < MAX_ARM_ATTEMPTS:
            return TransactionAction.ATTEMPT_ARM
        return (
            TransactionAction.RESTORE_S0_DISCARD_TOKEN_WARN_PASS_THROUGH
            if state.selection_mutated
            else TransactionAction.WARN_PASS_THROUGH
        )
    if event is TransactionEvent.S1_CAPTURE_FAILED:
        return (
            TransactionAction.RESTORE_S0_DISCARD_TOKEN_WARN_PASS_THROUGH
            if state.selection_mutated
            else TransactionAction.WARN_PASS_THROUGH
        )
    if event is TransactionEvent.EXCEPTION:
        if not state.selection_mutated:
            if state.token_registered:
                return TransactionAction.DISCARD_TOKEN_PASS_THROUGH
            return TransactionAction.PASS_THROUGH
        if not state.poller_armed:
            # A partial expansion failure is user-visible state churn; the
            # rollback must not be silent (contract: warn on every
            # post-mutation failure).
            return TransactionAction.RESTORE_S0_DISCARD_TOKEN_WARN_PASS_THROUGH
        return TransactionAction.PASS_THROUGH
    raise ValueError(event)


def action_after_arm_result(*, succeeded: bool, failed_attempts: int, warn_on_success: bool) -> TransactionAction:
    if succeeded:
        return TransactionAction.WARN_PASS_THROUGH if warn_on_success else TransactionAction.PASS_THROUGH
    if failed_attempts < MAX_ARM_ATTEMPTS:
        return TransactionAction.ATTEMPT_ARM
    return TransactionAction.RESTORE_S0_DISCARD_TOKEN_WARN_PASS_THROUGH


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
        onset_op=prior.onset_op,
        mode=prior.mode,
    )
    # A non-WATCHING replay guard is defensive; synchronous completion keeps
    # this pending state unreachable during ordinary invocation.
    if verdict is PollerVerdict.CONFIRMED or (
        prior.mode == "inset" and prior.replay_state is not InsetReplayState.WATCHING
    ):
        return PreambleResult(PreambleDecision.CONSUME_EVENT, verdict, False, False, current_selection)
    if verdict is PollerVerdict.CANCELLED:
        # Inset cancellation is native-owned; its token is discarded without
        # touching the selection or mesh.  This guard is defensive because
        # synchronous completion normally settles the token before preamble.
        restored = prior.mode != "inset"
        return PreambleResult(PreambleDecision.SETTLED, verdict, restored, True, prior.s0 if restored else current_selection)
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
    mode: Literal["inset", "bevel"] = "bevel"
    m0: SelectionSnapshot | None = None
    relation: SelectionRelation = SelectionRelation.DISJOINT
    self_mixed: bool = False
    mesh_select_mode: tuple[bool, bool, bool] = (False, True, False)
    user_sign: int = 1
    manual_both_sides: bool = False
    onset_op: OperatorIdentity | None = None
    replay_state: InsetReplayState = InsetReplayState.WATCHING
    f_user: SelectionContextSnapshot | None = None
    f_mirror: SelectionSnapshot | None = None
    select_mirrored: bool | None = None
    topology_backup: object | None = None
    live_layer_cleaned: bool = False
    onset: bool = False
    t0: float | None = None
    armed: bool = False
    restoring: bool = False
    restore_attempts: int = 0
    normalizing: bool = False
    normalize_attempts: int = 0
    pending_selection: SelectionSnapshot | None = None
    repeat_origin: bool = False
    restore_only: bool = False
    props_fingerprint: tuple = ()
    s0_prime: SelectionSnapshot | None = None


_ACTIVE_TOKEN: _InsetBevelToken | None = None
_ARMED_GENERATION: int | None = None
_NEXT_GENERATION = 1
_LOAD_PRE = False
_UNREGISTERED = False
_REPORTS: list[tuple[str, str]] = []
_COMPLETION_RECORD: CompletionRecord | None = None
_UNDO_POST_GUARD = False
_REDO_POST_GUARD = False


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


def _element_domain(element) -> str | None:
    name = type(element).__name__.upper()
    if "VERT" in name:
        return "VERT"
    if "EDGE" in name:
        return "EDGE"
    if "FACE" in name:
        return "FACE"
    return None


def _capture_selection_context(bm) -> SelectionContextSnapshot:
    history: list[tuple[str, int]] = []
    active: tuple[str, int] | None = None
    try:
        for element in bm.select_history:
            domain = _element_domain(element)
            if domain is not None:
                history.append((domain, int(element.index)))
        element = bm.select_history.active
        domain = _element_domain(element)
        if domain is not None:
            active = (domain, int(element.index))
    except (AttributeError, ReferenceError, RuntimeError):
        pass
    return SelectionContextSnapshot(_capture_selection(bm), tuple(history), active)


def _restore_selection_history(bm, context_snapshot: SelectionContextSnapshot) -> None:
    try:
        bm.select_history.clear()
        tables = {"VERT": bm.verts, "EDGE": bm.edges, "FACE": bm.faces}
        for domain, index in context_snapshot.history:
            table = tables[domain]
            if 0 <= index < len(table):
                bm.select_history.add(table[index])
        if context_snapshot.active is not None:
            domain, index = context_snapshot.active
            table = tables[domain]
            if 0 <= index < len(table):
                bm.select_history.active = table[index]
    except (AttributeError, ReferenceError, RuntimeError, KeyError):
        # History restoration is best effort; selection and topology remain
        # authoritative for the replay transaction.
        pass


def _remove_live_backup_layer(bm) -> None:
    if bm is None:
        return
    try:
        from . import layer_names

        layer = bm.verts.layers.int.get(layer_names.VERT_BACKUP_ID_LAYER)
        if layer is not None:
            bm.verts.layers.int.remove(layer)
    except Exception:
        traceback.print_exc()


def _write_selection(bm, snapshot: SelectionSnapshot, *, flush: bool = True) -> None:
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
    if flush:
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


def _selection_matches_snapshot(bm, snapshot: SelectionSnapshot) -> bool:
    current = _capture_selection(bm)
    return (
        current.verts == snapshot.verts
        and current.edges == snapshot.edges
        and current.faces == snapshot.faces
    )


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


def _clear_all_selection(token: _InsetBevelToken) -> bool:
    obj, mesh, bm = _edit_mesh_for_token(token)
    if obj is None or mesh is None or bm is None:
        return False
    try:
        for face in bm.faces:
            face.select = False
        for edge in bm.edges:
            edge.select = False
        for vertex in bm.verts:
            vertex.select = False
        bmesh.update_edit_mesh(mesh, loop_triangles=False, destructive=False)
        return True
    except Exception:
        traceback.print_exc()
        return False


def _snapshot_as_mapping(snapshot: SelectionSnapshot | None) -> dict[str, frozenset[int]]:
    if snapshot is None:
        return {"VERT": frozenset(), "EDGE": frozenset(), "FACE": frozenset()}
    return {"VERT": snapshot.verts, "EDGE": snapshot.edges, "FACE": snapshot.faces}


def _element_coordinates(element) -> tuple[tuple[float, float, float], ...]:
    if hasattr(element, "co"):
        coordinate = element.co
        return (tuple(float(value) for value in coordinate),)
    return tuple(tuple(float(value) for value in vertex.co) for vertex in element.verts)


def _capture_native_selection_context(token: _InsetBevelToken, bm) -> SelectionContextSnapshot:
    del token
    return _capture_selection_context(bm)


def _classify_bevel_sides(
    bm, domains: tuple[str, ...], axis_index: int, tolerance: float
) -> tuple[int, bool]:
    tables = {"VERT": bm.verts, "EDGE": bm.edges, "FACE": bm.faces}
    candidate_sign = 1
    for domain in domains:
        for element in tables[domain]:
            if not element.select or element.hide:
                continue
            values = [coordinate[axis_index] for coordinate in _element_coordinates(element)]
            off_plane = [value for value in values if abs(value) > tolerance]
            if off_plane and all(value >= -tolerance for value in values):
                candidate_sign = 1
                break
            if off_plane and all(value <= tolerance for value in values):
                candidate_sign = -1
                break
        else:
            continue
        break
    user_count = 0
    mirror_count = 0
    for domain in domains:
        for element in tables[domain]:
            if not element.select or element.hide:
                continue
            category = classify_element_side(
                _element_coordinates(element), axis_index, tolerance, candidate_sign
            )
            if category == "USER":
                user_count += 1
            elif category == "MIRROR":
                mirror_count += 1
    return candidate_sign, bool(user_count and mirror_count)


def _normalize_bevel_selection_in_mesh(token: _InsetBevelToken, bm) -> bool:
    if token.select_mirrored or token.relation is SelectionRelation.SELF_MIRRORED or token.manual_both_sides:
        return True
    tables = {"VERT": bm.verts, "EDGE": bm.edges, "FACE": bm.faces}
    selected = _capture_selection(bm)
    selected_map = _snapshot_as_mapping(selected)
    classes: dict[tuple[str, int], str] = {}
    for domain, table in tables.items():
        table.ensure_lookup_table()
        for element in table:
            if not element.select:
                continue
            category = classify_element_side(_element_coordinates(element), token.axis_index, token.tolerance, token.user_sign)
            classes[(domain, int(element.index))] = category
    normalized = normalize_bevel_selection(
        selected_map,
        classes,
        select_mirrored=bool(token.select_mirrored),
        manual_both_sides=token.manual_both_sides,
    )
    try:
        pending = SelectionSnapshot(
            verts=normalized["VERT"],
            edges=normalized["EDGE"],
            faces=normalized["FACE"],
            counts=selected.counts,
        )
        _write_selection(bm, pending, flush=False)
        # §5B explicitly forbids select_flush_mode on this path.
        _obj, mesh, _ = _edit_mesh_for_token(token)
        if mesh is None:
            return False
        bmesh.update_edit_mesh(mesh, loop_triangles=False, destructive=False)
        token.pending_selection = pending
        token.normalizing = True
        token.normalize_attempts = 0
        return True
    except Exception:
        traceback.print_exc()
        return False


def _token_override(token: _InsetBevelToken):
    window = token.window if _reference_alive(token.window) else None
    area = token.area if _reference_alive(token.area) else None
    region = token.region if _reference_alive(token.region) else None
    if window is None:
        raise RuntimeError("saved Blender window is unavailable")
    return bpy.context.temp_override(window=window, area=area, region=region)


def _active_operator_for_token(token: _InsetBevelToken):
    window = token.window if _reference_alive(token.window) else None
    area = token.area if _reference_alive(token.area) else None
    region = token.region if _reference_alive(token.region) else None
    if window is None:
        return None
    try:
        with bpy.context.temp_override(window=window, area=area, region=region):
            return getattr(bpy.context, "active_operator", None)
    except Exception:
        return None


def _save_completion_record(token: _InsetBevelToken, operator=None) -> None:
    global _COMPLETION_RECORD
    operator = operator or _active_operator_for_token(token)
    identity = _identity_from_operator(operator) if operator is not None else token.onset_op
    if identity is None:
        return
    if identity.idname not in TARGET_OPERATOR_IDNAMES:
        return
    if operator is None:
        props = token.props_fingerprint
    else:
        try:
            with _token_override(token):
                props = fingerprint(operator, route=token.native_idname)
        except Exception:
            props = fingerprint(operator, route=token.native_idname)
    _COMPLETION_RECORD = CompletionRecord(
        window_pointer=token.window_pointer,
        object_name=token.object_name,
        mesh_name=token.mesh_name,
        route=token.native_idname,
        operator=identity,
        fingerprint=props,
    )


def _completion_context(record: CompletionRecord | None):
    if record is None or bpy is None:
        return None, None, None, None
    window = next((w for w in getattr(bpy.context, "window_manager", ()).windows if _reference_alive(w) and int(w.as_pointer()) == record.window_pointer), None)
    if window is None:
        return None, None, None, None
    obj = bpy.data.objects.get(record.object_name)
    mesh = bpy.data.meshes.get(record.mesh_name)
    try:
        with bpy.context.temp_override(window=window):
            operator = getattr(bpy.context, "active_operator", None)
    except Exception:
        operator = None
    return window, obj, mesh, operator


def _mark_completion_suspended() -> None:
    global _COMPLETION_RECORD
    if _COMPLETION_RECORD is not None:
        _COMPLETION_RECORD = replace(_COMPLETION_RECORD, status=CompletionStatus.SUSPENDED)


def _f9_current_operator(record: CompletionRecord):
    if bpy is None:
        return None, None
    try:
        for window in bpy.context.window_manager.windows:
            if not _reference_alive(window) or int(window.as_pointer()) != record.window_pointer:
                continue
            with bpy.context.temp_override(window=window):
                return getattr(bpy.context, "active_operator", None), window
    except Exception:
        return None, None
    return None, None


def _discard_repeat_token_without_restore() -> None:
    """Invalidate a prior F9 token; undo_post already restored its step."""
    global _ACTIVE_TOKEN
    token = _ACTIVE_TOKEN
    if token is not None and token.repeat_origin:
        try:
            _cleanup_inset_backup(token)
        except Exception:
            traceback.print_exc()
        _discard_active_token()


def _neutralize_for_f9(token: _InsetBevelToken) -> bool:
    obj, mesh, bm = _edit_mesh_for_token(token)
    if obj is None or mesh is None or bm is None:
        return False
    try:
        for face in bm.faces:
            face.select = False
        for edge in bm.edges:
            edge.select = False
        for vertex in bm.verts:
            vertex.select = False
        bmesh.update_edit_mesh(mesh, loop_triangles=False, destructive=False)
        token.restore_only = True
        token.s0_prime = token.s0
        return True
    except Exception:
        traceback.print_exc()
        return False


def _f9_gate_and_plan(operator, window, record: CompletionRecord):
    """Return (token prerequisites, plan) or ``None`` for fail-closed."""
    if bpy is None or window is None:
        return None


    try:
        from . import element_pairs, matching, session_state
        with bpy.context.temp_override(window=window):
            context = bpy.context
            if context.mode != "EDIT_MESH" or session_state.sessions_active():
                return None
            obj = context.edit_object
            if obj is None or obj.type != "MESH" or len(context.objects_in_mode_unique_data) != 1:
                return None
            axes = matching.enabled_mesh_symmetry_axes(obj)
            if len(axes) != 1:
                return None
            _axis_name, axis_index = axes[0]
            tolerance = float(context.scene.ydd_symmetric_edit.tolerance)
            mesh = obj.data
            bm = bmesh.from_edit_mesh(mesh)
            try:
                from . import layer_names
                # History token layers persist on committed meshes long after
                # the cut settled; their mere presence must not veto F9.  The
                # mutation-owner conflict exists only while a repair is
                # queued/running, or while our own backup layer marks an
                # unfinished replay transaction.
                if session_state.history_repair_active():
                    return None
                if bm.verts.layers.int.get(layer_names.VERT_BACKUP_ID_LAYER) is not None:
                    return None
            except Exception:
                # Marker inspection is fail-closed: an unreadable marker API
                # must not permit a second mutation owner to run.
                return None
            affect = None
            if record.route == "mesh.bevel":
                affect = getattr(getattr(operator, "properties", None), "affect", DEFAULT_BEVEL_AFFECT)
                affect = _enum_identifier(affect)
            domains = leading_domains_for_route(record.route, affect)
            if domains is None:
                return None
            pair_maps = element_pairs.build_element_pair_maps(bm, axis_index, tolerance, mesh_object=obj)
            plan = element_pairs.plan_leading_domain_expansion(bm, pair_maps, domains=domains)
            if plan.hidden_counterpart_count or plan.unmatched_count:
                return None
            selected = {
                "VERT": frozenset(int(v.index) for v in bm.verts if v.select),
                "EDGE": frozenset(int(e.index) for e in bm.edges if e.select),
                "FACE": frozenset(int(f.index) for f in bm.faces if f.select),
            }
            relation, self_mixed = classify_selection_relation(selected, pair_maps, domains=domains)
            if relation is not SelectionRelation.DISJOINT:
                return None
            s0 = _capture_selection(bm)
            m0 = SelectionSnapshot(
                frozenset(plan.add_vert_indices), frozenset(plan.add_edge_indices),
                frozenset(plan.add_face_indices), s0.counts,
            )
            try:
                mesh_select_mode = tuple(bool(value) for value in context.tool_settings.mesh_select_mode)
            except Exception:
                mesh_select_mode = (False, True, False)
            sign, manual = _classify_bevel_sides(bm, domains, axis_index, tolerance)
            token = _register_dormant_token(
                context=context, obj=obj, s0=s0, recorded_op=_identity_from_operator(operator),
                native_idname=record.route, axis_index=axis_index, tolerance=tolerance,
                mode="inset" if record.route == "mesh.inset" else "bevel", m0=m0,
                relation=relation, self_mixed=self_mixed, mesh_select_mode=mesh_select_mode,
                user_sign=sign, manual_both_sides=manual, repeat_origin=True,
                props_fingerprint=fingerprint(operator, route=record.route),
            )
            return token, plan, bm, mesh, context
    except Exception:
        traceback.print_exc()
        return None


def _f9_current_edit_names(window) -> tuple[str | None, str | None]:
    """Names of the mesh actually being edited in *window* right now.

    The record-match in classify/reactivate must compare against the live
    context, not the record's own values, or the object/mesh conditions
    become tautologies.
    """

    try:
        with bpy.context.temp_override(window=window):
            obj = bpy.context.edit_object
            if obj is None or obj.type != "MESH":
                return None, None
            return str(obj.name), str(obj.data.name)
    except Exception:
        traceback.print_exc()
        return None, None


def _schedule_deferred_f9_restore(token: _InsetBevelToken) -> None:
    """Restore the pre-F9 selection after the imminent native replay.

    Only used when the poller could not be armed inside the undo_post
    handler: the replay is about to run synchronously, so the restore has to
    happen on a later timer tick.  Falls back to a warning (selection left
    neutralized) when the timer cannot be registered either.
    """

    snapshot = token.s0_prime if token.s0_prime is not None else token.s0
    object_name = token.object_name
    mesh_name = token.mesh_name
    landing_counts = token.invoke_counts

    def _deferred_restore():
        try:
            obj = bpy.data.objects.get(object_name)
            if obj is None or obj.mode != "EDIT" or obj.data.name != mesh_name:
                return None
            bm = bmesh.from_edit_mesh(obj.data)
            counts = MeshCounts(len(bm.verts), len(bm.edges), len(bm.faces))
            if counts != landing_counts:
                _record_report("WARNING", "F9 neutralization did not hold; selection restore skipped")
                return None
            _write_selection(bm, snapshot, flush=False)
            bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
        except Exception:
            traceback.print_exc()
            _record_report("WARNING", RESTORE_FAILED_WARNING)
        return None

    try:
        bpy.app.timers.register(_deferred_restore, first_interval=0.3)
    except Exception:
        traceback.print_exc()
        _record_report("WARNING", "F9 restore could not be scheduled; selection left cleared")


def _create_restore_only_token(operator, window, record: CompletionRecord):
    if bpy is None or window is None:
        return None
    try:
        with bpy.context.temp_override(window=window):
            context = bpy.context
            obj = context.edit_object
            if obj is None or obj.type != "MESH":
                return None
            mesh = obj.data
            bm = bmesh.from_edit_mesh(mesh)
            axes = _read_only_symmetry_parameters(context)
            if axes is None:
                return None
            _obj, axis_index, tolerance = axes
            token = _register_dormant_token(
                context=context, obj=obj, s0=_capture_selection(bm),
                recorded_op=_identity_from_operator(operator), native_idname=record.route,
                axis_index=axis_index, tolerance=tolerance,
                mode="inset" if record.route == "mesh.inset" else "bevel",
                relation=SelectionRelation.PARTIAL, repeat_origin=True,
                restore_only=True,
                props_fingerprint=fingerprint(operator, route=record.route),
            )
            if token is None:
                return None
            token.s0_prime = token.s0
            _neutralize_for_f9(token)
            if not _arm_poller(token):
                # The native replay runs right after this handler returns;
                # restoring the selection now would hand it that selection
                # and defeat the neutralization.  Restore must wait until
                # after the replay.
                _record_report("WARNING", ARM_FAILED_WARNING)
                _schedule_deferred_f9_restore(token)
                _discard_active_token()
                return None
            return token
    except Exception:
        traceback.print_exc()
        return None


def _capture_inset_props(token: _InsetBevelToken) -> dict[str, object] | None:
    try:
        with _token_override(token):
            operator = getattr(bpy.context, "active_operator", None)
            if _identity_from_operator(operator).idname != "mesh.inset":
                return None
            properties = getattr(operator, "properties", None)
            if properties is None:
                return None
            result: dict[str, object] = {}
            rna_properties = getattr(getattr(properties, "bl_rna", None), "properties", None)
            for name in INSET_REPLAY_PROPS:
                if not hasattr(properties, name):
                    return None
                rna_property = rna_properties.get(name) if rna_properties is not None else None
                if rna_property is not None and bool(getattr(rna_property, "is_readonly", False)):
                    return None
                result[name] = getattr(properties, name)
            return result
    except Exception:
        traceback.print_exc()
        return None


def _read_select_mirrored(token: _InsetBevelToken) -> bool | None:
    try:
        with _token_override(token):
            settings = getattr(getattr(bpy.context, "scene", None), "ydd_symmetric_edit", None)
            if settings is None:
                return False
            return bool(getattr(settings, "select_mirrored", False))
    except Exception:
        traceback.print_exc()
        return None


def _cleanup_inset_backup(token: _InsetBevelToken, bm=None) -> None:
    try:
        if bm is None:
            _obj, _mesh, bm = _edit_mesh_for_token(token)
        _remove_live_backup_layer(bm)
        token.live_layer_cleaned = True
    finally:
        if token.topology_backup is not None:
            try:
                from . import backup

                backup.remove_backup(token.topology_backup)
            except Exception:
                traceback.print_exc()
            finally:
                token.topology_backup = None


def _inset_replay_abort(token: _InsetBevelToken, native_context: SelectionContextSnapshot | None) -> None:
    """Restore native topology/selection after a pre-commit EXEC failure."""

    try:
        _obj, mesh, bm = _edit_mesh_for_token(token)
        if mesh is not None and token.topology_backup is not None:
            from . import backup

            backup.restore_topology_backup(mesh, token.topology_backup)
            _obj, mesh, bm = _edit_mesh_for_token(token)
            _remove_live_backup_layer(bm)
            if bm is not None and native_context is not None:
                restored = False
                for _attempt in range(RESTORE_VERIFY_ATTEMPTS):
                    _write_selection(bm, native_context.selection, flush=False)
                    if _selection_matches_snapshot(bm, native_context.selection):
                        restored = True
                        break
                if not restored:
                    _record_report("WARNING", RESTORE_FAILED_WARNING)
                _restore_selection_history(bm, native_context)
                bmesh.update_edit_mesh(mesh, loop_triangles=False, destructive=False)
    except Exception:
        traceback.print_exc()
        _record_report("WARNING", "Inset mirror replay rollback failed")
    finally:
        _cleanup_inset_backup(token)


def _run_inset_replay(token: _InsetBevelToken) -> InsetReplayState:
    """Execute the §5A post-confirm replay and return its terminal state."""

    # The replay issues bpy.ops calls (EXEC + undo_push); keep the F9 handler
    # disabled for the whole sequence so it can never observe our own ops.
    global _UNDO_POST_GUARD
    previous_guard = _UNDO_POST_GUARD
    _UNDO_POST_GUARD = True
    try:
        return _run_inset_replay_guarded(token)
    finally:
        _UNDO_POST_GUARD = previous_guard


def _run_inset_replay_guarded(token: _InsetBevelToken) -> InsetReplayState:
    token.replay_state = next_inset_replay_state(token.replay_state, InsetReplayEvent.CONFIRMED)
    props = _capture_inset_props(token)
    if props is not None:
        token.props_fingerprint = tuple(
            (name, round(value, 6) if isinstance(value, float) else value)
            for name, value in props.items()
        )
    token.select_mirrored = _read_select_mirrored(token)
    obj, mesh, bm = _edit_mesh_for_token(token)
    if props is None or token.select_mirrored is None or bm is None or mesh is None:
        token.replay_state = next_inset_replay_state(token.replay_state, InsetReplayEvent.REPLAY_FAILED)
        _cleanup_inset_backup(token, bm)
        _record_report("WARNING", "Inset mirror replay prerequisites were unavailable")
        return token.replay_state
    native_context = _capture_native_selection_context(token, bm)
    token.f_user = native_context
    m0 = token.m0
    if m0 is None:
        token.replay_state = next_inset_replay_state(token.replay_state, InsetReplayEvent.REPLAY_FAILED)
        _cleanup_inset_backup(token, bm)
        _record_report("WARNING", "Inset mirror replay selection was unavailable")
        return token.replay_state
    for indices, count in ((m0.verts, len(bm.verts)), (m0.edges, len(bm.edges)), (m0.faces, len(bm.faces))):
        if any(index < 0 or index >= count for index in indices):
            token.replay_state = next_inset_replay_state(token.replay_state, InsetReplayEvent.REPLAY_FAILED)
            _cleanup_inset_backup(token, bm)
            _record_report("WARNING", "Inset mirror replay selection was out of range")
            return token.replay_state
    try:
        from . import backup

        token.topology_backup = backup.create_topology_backup(bm)
    except Exception:
        traceback.print_exc()
        _remove_live_backup_layer(bm)
        token.replay_state = next_inset_replay_state(token.replay_state, InsetReplayEvent.REPLAY_FAILED)
        _cleanup_inset_backup(token, bm)
        _record_report("WARNING", "Inset mirror replay backup could not be created")
        return token.replay_state
    try:
        token.replay_state = next_inset_replay_state(token.replay_state, InsetReplayEvent.PROPS_READY)
        _write_selection(bm, m0, flush=True)
        bmesh.update_edit_mesh(mesh, loop_triangles=False, destructive=False)
        with _token_override(token):
            result = bpy.ops.mesh.inset("EXEC_DEFAULT", **props)
        if not _undo_push_finished(result):
            raise RuntimeError("mesh.inset replay did not finish")
        token.replay_state = next_inset_replay_state(token.replay_state, InsetReplayEvent.EXEC_FINISHED)
    except Exception:
        traceback.print_exc()
        _inset_replay_abort(token, native_context)
        token.replay_state = next_inset_replay_state(token.replay_state, InsetReplayEvent.REPLAY_FAILED)
        _record_report("WARNING", "Inset mirror replay failed; the result is one-sided")
        return token.replay_state

    # EXEC_COMMITTED: no rollback to one-sided is permitted from here on.
    try:
        _obj, mesh, bm = _edit_mesh_for_token(token)
        if bm is None or mesh is None:
            raise RuntimeError("mesh disappeared after inset replay")
        token.f_mirror = _capture_selection(bm)
        if token.select_mirrored:
            merged = SelectionSnapshot(
                verts=frozenset(native_context.selection.verts | token.f_mirror.verts),
                edges=frozenset(native_context.selection.edges | token.f_mirror.edges),
                faces=frozenset(native_context.selection.faces | token.f_mirror.faces),
                counts=token.f_mirror.counts,
            )
        else:
            merged = native_context.selection
        _write_selection(bm, merged, flush=True)
        _restore_selection_history(bm, native_context)
        _remove_live_backup_layer(bm)
        bmesh.update_edit_mesh(mesh, loop_triangles=False, destructive=False)
        token.live_layer_cleaned = True
        _cleanup_inset_backup(token, bm)
        with _token_override(token):
            push_result = bpy.ops.ed.undo_push(message="Symmetric inset")
        if not _undo_push_finished(push_result):
            raise RuntimeError("undo_push did not finish")
        token.replay_state = next_inset_replay_state(token.replay_state, InsetReplayEvent.PUSH_FINISHED)
    except Exception:
        traceback.print_exc()
        try:
            _obj, mesh, bm = _edit_mesh_for_token(token)
            if bm is None or mesh is None:
                raise RuntimeError("mesh disappeared while restoring native selection")
            _write_selection(bm, native_context.selection, flush=False)
            _restore_selection_history(bm, native_context)
            bmesh.update_edit_mesh(mesh, loop_triangles=False, destructive=False)
        except Exception:
            traceback.print_exc()
            _record_report("WARNING", "Symmetric inset could not restore the native selection context")
        _record_report("WARNING", "Symmetric inset applied; selection/undo may be degraded")
        token.replay_state = next_inset_replay_state(token.replay_state, InsetReplayEvent.POSTPROCESS_FAILED)
        _cleanup_inset_backup(token)
    return token.replay_state


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
    global _COMPLETION_RECORD
    if verdict is PollerVerdict.CONFIRMED:
        observed = _read_active_operator(token)
        if observed.idname in TARGET_OPERATOR_IDNAMES:
            token.onset_op = observed
        if token.mode == "inset":
            replay_state = _run_inset_replay(token)
            if replay_state in {InsetReplayState.PUSHED, InsetReplayState.DEGRADED_SYMMETRIC}:
                _run_confirm_diagnostic(token)
                _save_completion_record(token)
            elif token.repeat_origin:
                _COMPLETION_RECORD = None
            _discard_active_token()
            return
        token.select_mirrored = _read_select_mirrored(token)
        active_operator = _active_operator_for_token(token)
        if active_operator is not None:
            token.props_fingerprint = fingerprint(active_operator, route="mesh.bevel", override=_token_override(token))
        _obj, _mesh, bm = _edit_mesh_for_token(token)
        if token.select_mirrored is None:
            _record_report("WARNING", "Bevel selection normalization skipped: select_mirrored was unreadable")
        elif bm is not None and token.select_mirrored is False:
            if not _normalize_bevel_selection_in_mesh(token, bm):
                _record_report("WARNING", "Bevel selection normalization failed")
        if token.normalizing:
            return
        _run_confirm_diagnostic(token)
        _save_completion_record(token)
        _discard_active_token()
        return
    if verdict is PollerVerdict.CANCELLED:
        if token.mode == "inset":
            if token.repeat_origin:
                _COMPLETION_RECORD = None
            _discard_active_token()
            return
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
    if token.repeat_origin:
        _COMPLETION_RECORD = None
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
        onset_op=token.onset_op,
        mode=token.mode,
        replay_state=token.replay_state,
    )


def _settle_prior_token_sync() -> Literal["none", "consume", "settled"]:
    token = _ACTIVE_TOKEN
    if token is None:
        return "none"
    if token.normalizing:
        # A pending normalization is owned by the poller; synchronous
        # completion makes this defensive consume path unreachable normally.
        return "consume"
    result = apply_invoke_preamble(_prior_token_view(token), frozenset())
    if result.decision is PreambleDecision.CONSUME_EVENT:
        return "consume"
    if result.verdict is PollerVerdict.CONFIRMED or (
        token.mode == "inset" and token.replay_state is not InsetReplayState.WATCHING
    ):
        # Confirmed post-processing is exclusively owned by the poller; the
        # new event is consumed and must not abandon replay or normalization.
        # The replay-pending branch is defensive; synchronous completion
        # makes it unreachable in normal operation.
        return "consume"
    if result.verdict is PollerVerdict.CONFIRMED:
        _run_confirm_diagnostic(token)
    if result.restored_s0 and token.mode != "inset":
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
    mode: Literal["inset", "bevel"] = "bevel",
    m0: SelectionSnapshot | None = None,
    relation: SelectionRelation = SelectionRelation.DISJOINT,
    self_mixed: bool = False,
    mesh_select_mode: tuple[bool, bool, bool] = (False, True, False),
    user_sign: int = 1,
    manual_both_sides: bool = False,
    repeat_origin: bool = False,
    props_fingerprint: tuple = (),
    restore_only: bool = False,
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
        mode=mode,
        m0=m0,
        relation=relation,
        self_mixed=self_mixed,
        mesh_select_mode=mesh_select_mode,
        user_sign=user_sign,
        manual_both_sides=manual_both_sides,
        repeat_origin=repeat_origin,
        props_fingerprint=props_fingerprint,
        restore_only=restore_only,
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
        if token.restore_only:
            if token.restore_attempts == 0:
                token.restore_attempts = 1
                return POLLER_INTERVAL
            current = _current_counts_from_token(token)
            if current is not None and current != token.invoke_counts:
                _record_report("WARNING", "F9 neutral replay changed topology; restoring selection best-effort")
            if token.s0_prime is not None:
                if not _restore_s0(token):
                    _record_report("WARNING", RESTORE_FAILED_WARNING)
                    _record_report("WARNING", "F9 restore-only failed; clearing all selection")
                    _clear_all_selection(token)
            global _COMPLETION_RECORD
            _COMPLETION_RECORD = None
            _discard_active_token()
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
                _record_report("WARNING", RESTORE_FAILED_WARNING)
                _record_report("WARNING", "Inset/Bevel restore failed; clearing all selection")
                _clear_all_selection(token)
                _discard_active_token()
                return None
            token.restore_attempts += 1
            if not _restore_s0(token):
                _record_report("WARNING", RESTORE_FAILED_WARNING)
                _record_report("WARNING", "Inset/Bevel restore failed; clearing all selection")
                _clear_all_selection(token)
                _discard_active_token()
                return None
            return min(
                RESTORE_VERIFY_BASE_INTERVAL * token.restore_attempts,
                RESTORE_VERIFY_MAX_INTERVAL,
            )
        if token.normalizing:
            if discard_reason(**_token_discard_flags(token)) is not None:
                _discard_active_token()
                return None
            if _selection_matches(token, token.pending_selection):
                _tag_view3d_redraw(token)
                token.normalizing = False
                token.pending_selection = None
                _run_confirm_diagnostic(token)
                _save_completion_record(token)
                _discard_active_token()
                return None
            if token.normalize_attempts >= RESTORE_VERIFY_ATTEMPTS:
                _record_report("WARNING", "Bevel selection normalization could not be verified")
                _discard_active_token()
                return None
            token.normalize_attempts += 1
            _obj, mesh, bm = _edit_mesh_for_token(token)
            if mesh is None or bm is None or token.pending_selection is None:
                _record_report("WARNING", "Bevel selection normalization failed")
                _discard_active_token()
                return None
            _write_selection(bm, token.pending_selection, flush=False)
            return min(
                RESTORE_VERIFY_BASE_INTERVAL * token.normalize_attempts,
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
            if (
                token.repeat_origin
                and token.mode == "bevel"
                and not modal_alive
                and current_op.idname == "mesh.bevel"
                and counts == token.invoke_counts
            ):
                active_object = _active_operator_for_token(token)
                if active_object is not None and fingerprint(active_object, route="mesh.bevel", override=_token_override(token)) == token.props_fingerprint:
                    _apply_settled_verdict(token, PollerVerdict.CONFIRMED)
                    return POLLER_INTERVAL if _ACTIVE_TOKEN is token else None
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
            onset_op=token.onset_op,
            mode=token.mode,
            **flags,
        )
        if result is TickResult.WAIT:
            if modal_alive and current_op.idname in TARGET_OPERATOR_IDNAMES and current_op.pointer is not None:
                token.onset = True
                if token.onset_op is None and current_op.idname in TARGET_OPERATOR_IDNAMES:
                    token.onset_op = current_op
            return POLLER_INTERVAL
        if result is TickResult.CONFIRMED:
            _apply_settled_verdict(token, PollerVerdict.CONFIRMED)
            return POLLER_INTERVAL if _ACTIVE_TOKEN is token else None
        if result is TickResult.CANCELLED:
            _apply_settled_verdict(token, PollerVerdict.CANCELLED, staged_restore=True)
            return POLLER_INTERVAL if _ACTIVE_TOKEN is token else None
        _discard_active_token()
        return None
    except Exception:
        traceback.print_exc()
        token = _ACTIVE_TOKEN
        if token is not None and token.mode == "inset":
            try:
                _cleanup_inset_backup(token)
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
    global _LOAD_PRE, _COMPLETION_RECORD
    _LOAD_PRE = True
    _COMPLETION_RECORD = None
    _discard_active_token()


@persistent
def _on_load_post(_dummy) -> None:
    # Without this release the load_pre latch would discard every future
    # token until the addon is re-registered.
    global _LOAD_PRE
    _LOAD_PRE = False


@persistent
def _on_undo_post(_dummy) -> None:
    """Re-enter native Inset/Bevel F9 between undo and operator replay."""
    global _UNDO_POST_GUARD, _COMPLETION_RECORD
    if _UNDO_POST_GUARD or bpy is None or _UNREGISTERED or _COMPLETION_RECORD is None:
        return
    _UNDO_POST_GUARD = True
    try:
        record = _COMPLETION_RECORD
        recorded_obj = bpy.data.objects.get(record.object_name)
        if recorded_obj is None or recorded_obj.data.name != record.mesh_name:
            # The recorded object/mesh no longer exists: the record can never
            # match a live context again.
            _COMPLETION_RECORD = None
            return
        operator, window = _f9_current_operator(record)
        if operator is None:
            _mark_completion_suspended()
            return
        try:
            window_pointer = int(window.as_pointer())
        except Exception:
            window_pointer = None
        current_object_name, current_mesh_name = _f9_current_edit_names(window)
        classification = classify_f9_intervention(
            operator,
            record,
            window_pointer=window_pointer,
            object_name=current_object_name,
            mesh_name=current_mesh_name,
            route=record.route,
        )
        _COMPLETION_RECORD = classification.record
        if classification.decision is F9Decision.SUSPEND:
            return
        if classification.decision is not F9Decision.INTERVENE:
            return
        token = _ACTIVE_TOKEN
        if token is not None:
            decision = supersede_decision(
                repeat_origin=token.repeat_origin,
                restore_only=token.restore_only,
                mode=token.mode,
                replay_state=token.replay_state,
                same_context=(
                    token.window_pointer == window_pointer
                    and token.object_name == current_object_name
                    and token.mesh_name == current_mesh_name
                ),
            )
            if decision == "SUPERSEDE":
                _discard_repeat_token_without_restore()
            elif token.restore_only:
                if not _restore_s0(token):
                    _record_report("WARNING", RESTORE_FAILED_WARNING)
                    _record_report("WARNING", "F9 restore-only supersede failed; clearing all selection")
                    _clear_all_selection(token)
                _discard_active_token()
            else:
                _record_report("WARNING", "F9 intervention skipped while another token is active")
                return
        prepared = _f9_gate_and_plan(operator, window, record)
        if prepared is None:
            _record_report("WARNING", "F9 symmetry gate failed; native replay neutralized")
            if _create_restore_only_token(operator, window, record) is None:
                _COMPLETION_RECORD = None
            return
        token, plan, bm, mesh, context = prepared
        if token is None:
            return
        try:
            if token.mode == "bevel":
                token.s0_prime = token.s0
                _write_selection(bm, SelectionSnapshot(
                    frozenset(plan.add_vert_indices) | token.s0.verts,
                    frozenset(plan.add_edge_indices) | token.s0.edges,
                    frozenset(plan.add_face_indices) | token.s0.faces,
                    token.s0.counts,
                ))
                bm.select_flush_mode()
                bmesh.update_edit_mesh(mesh, loop_triangles=False, destructive=False)
                token.s1 = _capture_selection(bm)
            armed = _arm_poller(token)
        except Exception:
            traceback.print_exc()
            armed = False
        if not armed:
            # The replay runs right after this handler; a synchronous restore
            # (or a partially expanded selection) would hand it a wrong
            # selection.  Neutralize and defer the restore instead
            # (fail-closed); the token must not stay registered un-armed.
            _record_report("WARNING", ARM_FAILED_WARNING)
            token.s0_prime = token.s0_prime if token.s0_prime is not None else token.s0
            try:
                _neutralize_for_f9(token)
                _schedule_deferred_f9_restore(token)
            except Exception:
                traceback.print_exc()
                _record_report("WARNING", RESTORE_FAILED_WARNING)
            _discard_active_token()
    except Exception:
        traceback.print_exc()
    finally:
        _UNDO_POST_GUARD = False


@persistent
def _on_redo_post(_dummy) -> None:
    global _REDO_POST_GUARD, _COMPLETION_RECORD
    if _REDO_POST_GUARD or bpy is None or _UNREGISTERED or _COMPLETION_RECORD is None:
        return
    _REDO_POST_GUARD = True
    try:
        record = _COMPLETION_RECORD
        operator, window = _f9_current_operator(record)
        if operator is None:
            return
        try:
            pointer = int(window.as_pointer())
        except Exception:
            pointer = None
        current_object_name, current_mesh_name = _f9_current_edit_names(window)
        _COMPLETION_RECORD = reactivate_completion_record(
            record,
            operator,
            window_pointer=pointer,
            object_name=current_object_name,
            mesh_name=current_mesh_name,
        )
    except Exception:
        traceback.print_exc()
    finally:
        _REDO_POST_GUARD = False


def install_runtime_hooks() -> None:
    global _UNREGISTERED, _LOAD_PRE
    _UNREGISTERED = False
    _LOAD_PRE = False
    if bpy is None:
        return
    hooks = (
        (bpy.app.handlers.load_pre, _on_load_pre),
        (bpy.app.handlers.load_post, _on_load_post),
        (bpy.app.handlers.undo_post, _on_undo_post),
        (bpy.app.handlers.redo_post, _on_redo_post),
    )
    for handler_list, handler in hooks:
        if handler not in handler_list:
            handler_list.append(handler)


def cleanup_runtime() -> None:
    global _UNREGISTERED, _LOAD_PRE, _COMPLETION_RECORD
    _UNREGISTERED = True
    _COMPLETION_RECORD = None
    _discard_active_token()
    if bpy is None:
        _LOAD_PRE = False
        _REPORTS.clear()
        return
    for handler_list, handler in (
        (bpy.app.handlers.load_pre, _on_load_pre),
        (bpy.app.handlers.load_post, _on_load_post),
        (bpy.app.handlers.undo_post, _on_undo_post),
        (bpy.app.handlers.redo_post, _on_redo_post),
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
    """Intercept native Inset/Bevel routes and apply their contract-specific postprocess."""

    bl_idname = "mesh.ydd_symmetric_edit_inset_bevel_intercept"
    bl_label = "Prepare ydd Symmetric Edit for Inset/Bevel"
    bl_description = "Prepare a symmetric Inset or Bevel route and pass the native event through"
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
        selected = {
            "VERT": frozenset(int(vertex.index) for vertex in bm.verts if vertex.select and not vertex.hide),
            "EDGE": frozenset(int(edge.index) for edge in bm.edges if edge.select and not edge.hide),
            "FACE": frozenset(int(face.index) for face in bm.faces if face.select and not face.hide),
        }
        relation, self_mixed = classify_selection_relation(selected, pair_maps, domains=domains)
        if relation is SelectionRelation.UNMATCHED:
            _operator_report(self, {"WARNING"}, UNMATCHED_WARNING)
            return {"PASS_THROUGH"}
        if relation is SelectionRelation.SELF_MIRRORED or relation is SelectionRelation.PARTIAL:
            if relation is SelectionRelation.PARTIAL or (self_mixed and route.native_operator == "mesh.inset"):
                _operator_report(self, {"WARNING"}, "Inset/Bevel selection overlap is not guaranteed symmetric")
            return {"PASS_THROUGH"}
        if self_mixed and route.native_operator == "mesh.inset":
            _operator_report(self, {"WARNING"}, "Inset/Bevel selection overlap is not guaranteed symmetric")
            return {"PASS_THROUGH"}
        if not plan.add_vert_indices and not plan.add_edge_indices and not plan.add_face_indices:
            return {"PASS_THROUGH"}

        try:
            s0 = _capture_selection(bm)
            state = replace(state, snapshot_exists=True)
            try:
                mesh_select_mode = tuple(bool(value) for value in context.tool_settings.mesh_select_mode)
            except Exception:
                mesh_select_mode = (False, True, False)
            m0 = SelectionSnapshot(
                verts=frozenset(plan.add_vert_indices),
                edges=frozenset(plan.add_edge_indices),
                faces=frozenset(plan.add_face_indices),
                counts=s0.counts,
            )
            bevel_user_sign, manual_both_sides = _classify_bevel_sides(
                bm, domains, axis_index, tolerance
            )
            token = _register_dormant_token(
                context=context,
                obj=obj,
                s0=s0,
                recorded_op=_identity_from_operator(getattr(context, "active_operator", None)),
                native_idname=route.native_operator,
                axis_index=axis_index,
                tolerance=tolerance,
                mode="inset" if route.native_operator == "mesh.inset" else "bevel",
                m0=m0,
                relation=relation,
                self_mixed=self_mixed,
                mesh_select_mode=mesh_select_mode,
                user_sign=bevel_user_sign,
                manual_both_sides=manual_both_sides if route.native_operator == "mesh.bevel" else False,
            )
            if token is None:
                return self._apply_transaction_action(
                    context,
                    state,
                    next_transaction_action(state, TransactionEvent.TOKEN_REGISTER_FAILED),
                )
            state = replace(state, token_registered=True)

            if route.native_operator == "mesh.inset":
                return self._arm_or_leave(context, state, token, warn_on_success=False)

            # A partially applied plan must already restore S0, so flag the
            # mutation before the first select write, not after the flush.
            state = replace(state, selection_mutated=True)
            element_pairs.apply_expansion_plan(bm, plan)
            bm.select_flush_mode()
            bmesh.update_edit_mesh(mesh, loop_triangles=False, destructive=False)
            token.s1 = _capture_selection(bm)
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
            return self._apply_transaction_action(
                context,
                state,
                TransactionAction.RESTORE_S0_DISCARD_TOKEN_WARN_PASS_THROUGH,
            )

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
                _operator_report(self, {"WARNING"}, ARM_FAILED_WARNING)
            if token is not None:
                try:
                    if not _restore_s0(token):
                        _operator_report(self, {"WARNING"}, RESTORE_FAILED_WARNING)
                        _operator_report(self, {"WARNING"}, "Inset/Bevel restore failed; clearing all selection")
                        _clear_all_selection(token)
                except Exception:
                    traceback.print_exc()
                    _operator_report(self, {"WARNING"}, RESTORE_FAILED_WARNING)
                    _operator_report(self, {"WARNING"}, "Inset/Bevel restore failed; clearing all selection")
                    try:
                        _clear_all_selection(token)
                    except Exception:
                        traceback.print_exc()
            _discard_active_token()
            return {"PASS_THROUGH"}
        if action is TransactionAction.ATTEMPT_ARM:
            if token is None:
                return {"PASS_THROUGH"}
            return self._arm_or_leave(context, state, token, warn_on_success=True)
        if action is TransactionAction.WARN_PASS_THROUGH:
            _operator_report(self, {"WARNING"}, ARM_FAILED_WARNING)
            return {"PASS_THROUGH"}
        if action is TransactionAction.WARN_ONLY:
            _operator_report(self, {"WARNING"}, RESTORE_FAILED_WARNING)
            return {"PASS_THROUGH"}
        return {"PASS_THROUGH"}


CLASSES = (MESH_OT_ydd_symmetric_edit_inset_bevel_intercept,)
