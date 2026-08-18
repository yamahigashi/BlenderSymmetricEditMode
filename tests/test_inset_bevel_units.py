# SPDX-License-Identifier: GPL-3.0-or-later

"""Pure-Python unit tests for Inset/Bevel EXPAND_PASSTHROUGH helpers."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from unittest import mock

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _load_inset_bevel():
    path = REPOSITORY_ROOT / "ydd_symmetric_edit" / "inset_bevel.py"
    spec = importlib.util.spec_from_file_location("yse_inset_bevel", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ib = _load_inset_bevel()


@dataclass
class FakeKmi:
    type: str
    value: str
    any: bool = False
    shift: int = 0
    ctrl: int = 0
    alt: int = 0
    oskey: int = 0
    hyper: int = 0
    key_modifier: str = "NONE"
    direction: str = "ANY"
    repeat: bool = False
    idname: str = "mesh.inset"


def _op(idname: str | None = None, pointer: int | None = None) -> object:
    return ib.OperatorIdentity(idname, pointer)


def _counts(verts: int = 8, edges: int = 12, faces: int = 6) -> object:
    return ib.MeshCounts(verts, edges, faces)


def _classify(**overrides):
    payload = {
        "recorded_op": _op(None, None),
        "current_op": _op(None, None),
        "invoke_counts": _counts(),
        "current_counts": _counts(),
        "modal_alive": False,
        "onset": False,
        "elapsed_s": 0.0,
        "selection_matches_s1": True,
    }
    payload.update(overrides)
    return ib.classify_poller_tick(**payload)


def _tick(**overrides):
    payload = {
        "token_generation": 1,
        "current_generation": 1,
        "recorded_op": _op(None, None),
        "current_op": _op(None, None),
        "invoke_counts": _counts(),
        "current_counts": _counts(),
        "modal_alive": False,
        "onset": False,
        "elapsed_s": 0.0,
        "selection_matches_s1": True,
    }
    payload.update(overrides)
    return ib.evaluate_poller_tick(**payload)


# --- 1. §5 判別純関数の全遷移 -------------------------------------------------


def test_classify_wait_before_onset_without_evidence():
    assert _classify() is ib.PollerVerdict.WAIT


def test_classify_wait_while_target_modal_is_alive():
    assert _classify(modal_alive=True, elapsed_s=10.0) is ib.PollerVerdict.WAIT
    assert _classify(onset=True, modal_alive=True, elapsed_s=10.0) is ib.PollerVerdict.WAIT


def test_classify_confirmed_when_active_operator_becomes_target():
    assert (
        _classify(
            recorded_op=_op(None, 1),
            current_op=_op("mesh.inset", 2),
            current_counts=_counts(9, 14, 7),
        )
        is ib.PollerVerdict.CONFIRMED
    )
    assert (
        _classify(
            recorded_op=_op("mesh.inset", 1),
            current_op=_op("mesh.bevel", 2),
            onset=True,
            modal_alive=False,
        )
        is ib.PollerVerdict.CONFIRMED
    )


def test_classify_cancelled_when_onset_disappears_with_same_identity_and_counts():
    recorded = _op("mesh.knife_tool", 7)
    assert (
        _classify(
            recorded_op=recorded,
            current_op=recorded,
            onset=True,
            modal_alive=False,
        )
        is ib.PollerVerdict.CANCELLED
    )


def test_classify_unknown_when_onset_disappears_onto_foreign_operator():
    assert (
        _classify(
            recorded_op=_op(None, None),
            current_op=_op("mesh.knife_tool", 9),
            onset=True,
            modal_alive=False,
        )
        is ib.PollerVerdict.UNKNOWN
    )


# --- 2. quick-release (onset 前確定証拠) → CONFIRMED --------------------------


def test_quick_release_requires_strict_count_increase_before_onset():
    assert (
        _classify(
            recorded_op=_op(None, None),
            current_op=_op("mesh.inset", 11),
            onset=False,
            modal_alive=False,
            elapsed_s=0.0,
            current_counts=_counts(9, 14, 7),
        )
        is ib.PollerVerdict.CONFIRMED
    )


# --- 3. 同一 tick で timeout と onset/確定証拠が並立 → onset/証拠が勝つ --------


def test_same_tick_timeout_without_lineage_is_cancelled_when_selection_unchanged():
    assert (
        _classify(
            recorded_op=_op(None, None),
            current_op=_op("mesh.bevel", 3),
            elapsed_s=2.0,
            selection_matches_s1=True,
        )
        is ib.PollerVerdict.CANCELLED
    )


def test_same_tick_timeout_loses_to_onset_modal():
    assert (
        _classify(
            modal_alive=True,
            elapsed_s=2.5,
            selection_matches_s1=False,
        )
        is ib.PollerVerdict.WAIT
    )


# --- 4. onset なし 2.0s + 選択 ≠ S1 → UNKNOWN --------------------------------


def test_onset_timeout_without_s1_match_is_unknown():
    assert (
        _classify(
            elapsed_s=2.0,
            selection_matches_s1=False,
            current_counts=_counts(),
        )
        is ib.PollerVerdict.UNKNOWN
    )


def test_onset_timeout_with_s1_and_counts_match_is_cancelled():
    assert (
        _classify(
            elapsed_s=2.0,
            selection_matches_s1=True,
            current_counts=_counts(),
        )
        is ib.PollerVerdict.CANCELLED
    )


# --- 5. ポインタ再利用相当 + counts 降格 ---------------------------------------


def test_same_identity_unchanged_counts_is_cancelled():
    recorded = _op("mesh.bevel", 42)
    assert (
        _classify(
            recorded_op=recorded,
            current_op=recorded,
            onset=True,
            modal_alive=False,
            invoke_counts=_counts(8, 12, 6),
            current_counts=_counts(8, 12, 6),
        )
        is ib.PollerVerdict.CANCELLED
    )


def test_same_identity_changed_counts_confirms_bevel_via_counts():
    # Pointer reuse across consecutive bevel runs makes current == recorded
    # on a genuine confirm; the count increase plus the target idname is the
    # decisive evidence (normalization must not be skipped via UNKNOWN).
    recorded = _op("mesh.bevel", 42)
    assert (
        _classify(
            recorded_op=recorded,
            current_op=recorded,
            onset=True,
            modal_alive=False,
            invoke_counts=_counts(8, 12, 6),
            current_counts=_counts(16, 24, 12),
        )
        is ib.PollerVerdict.CONFIRMED
    )
    # A foreign operator with grown counts is interference, not a confirm.
    assert (
        _classify(
            recorded_op=recorded,
            current_op=_op("mesh.knife_tool", 42),
            onset=True,
            modal_alive=False,
            invoke_counts=_counts(8, 12, 6),
            current_counts=_counts(16, 24, 12),
        )
        is ib.PollerVerdict.UNKNOWN
    )


# --- 6. generation ラッパ + §4-0 同期決着 -------------------------------------


def test_stale_generation_is_noop():
    assert _tick(token_generation=1, current_generation=2) is ib.TickResult.NO_OP


def test_live_generation_still_classifies():
    assert _tick(token_generation=4, current_generation=4) is ib.TickResult.WAIT


def _prior_view(**overrides):
    payload = {
        "onset": True,
        "modal_alive": False,
        "recorded_op": _op(None, None),
        "current_op": _op(None, None),
        "invoke_counts": _counts(),
        "current_counts": _counts(),
        "selection_matches_s1": True,
        "elapsed_s": 0.1,
        "s0": frozenset({1}),
        "discard": False,
    }
    payload.update(overrides)
    return ib.PriorTokenView(**payload)


def test_invoke_preamble_restores_prior_cancel_before_new_snapshot():
    prior_s0 = frozenset({1})
    result = ib.apply_invoke_preamble(_prior_view(s0=prior_s0), frozenset({1, 2}))
    assert result.decision is ib.PreambleDecision.SETTLED
    assert result.restored_s0 is True
    assert result.token_cleared is True
    assert result.selection == prior_s0


def test_invoke_preamble_never_returns_wait_before_timeout():
    # No onset yet, 0.1s elapsed, no evidence: the live poller would WAIT,
    # but the preamble must force a terminal verdict (contract section 4-0).
    result = ib.apply_invoke_preamble(
        _prior_view(onset=False, elapsed_s=0.1, selection_matches_s1=True),
        frozenset(),
    )
    assert result.decision is ib.PreambleDecision.SETTLED
    assert result.verdict is ib.PollerVerdict.CANCELLED
    assert result.restored_s0 is True
    mismatch = ib.apply_invoke_preamble(
        _prior_view(onset=False, elapsed_s=0.1, selection_matches_s1=False),
        frozenset(),
    )
    assert mismatch.verdict is ib.PollerVerdict.UNKNOWN
    assert mismatch.restored_s0 is False


def test_invoke_preamble_consumes_when_modal_alive_even_without_onset():
    result = ib.apply_invoke_preamble(_prior_view(onset=False, modal_alive=True), frozenset())
    assert result.decision is ib.PreambleDecision.CONSUME_EVENT
    assert result.token_cleared is False


def test_invoke_preamble_discard_clears_without_restore():
    result = ib.apply_invoke_preamble(
        _prior_view(discard=True, selection_matches_s1=True),
        frozenset({7}),
    )
    assert result.decision is ib.PreambleDecision.SETTLED
    assert result.verdict is ib.PollerVerdict.UNKNOWN
    assert result.restored_s0 is False
    assert result.token_cleared is True


def test_invoke_preamble_consumes_event_when_onset_and_modal_alive():
    current = frozenset({3, 4})
    result = ib.apply_invoke_preamble(
        _prior_view(
            modal_alive=True,
            recorded_op=_op("mesh.inset", 1),
            current_op=_op("mesh.inset", 1),
            elapsed_s=0.4,
            s0=frozenset({3}),
        ),
        current,
    )
    assert result.decision is ib.PreambleDecision.CONSUME_EVENT
    assert result.token_cleared is False
    assert result.restored_s0 is False
    assert result.selection == current


def test_invoke_preamble_consumes_confirmed_postprocess_for_both_routes():
    prior = _prior_view(
        mode="inset",
        current_op=_op("mesh.inset", 8),
        invoke_counts=_counts(),
        current_counts=_counts(9, 14, 7),
        onset=False,
    )
    result = ib.apply_invoke_preamble(prior, frozenset())
    assert result.decision is ib.PreambleDecision.CONSUME_EVENT
    bevel = replace(prior, mode="bevel", current_op=_op("mesh.bevel", 8), current_counts=_counts(9, 14, 7))
    assert ib.apply_invoke_preamble(bevel, frozenset()).decision is ib.PreambleDecision.CONSUME_EVENT


def test_invoke_preamble_without_prior_keeps_current_selection():
    current = frozenset({9})
    result = ib.apply_invoke_preamble(None, current)
    assert result.decision is ib.PreambleDecision.NO_PRIOR
    assert result.selection == current


# --- 7. transaction 状態機械 ---------------------------------------------------


def test_transaction_exception_before_mutation_passthrough():
    state = ib.TransactionState()
    assert ib.next_transaction_action(state, ib.TransactionEvent.EXCEPTION) is ib.TransactionAction.PASS_THROUGH
    registered = replace(state, token_registered=True, snapshot_exists=True)
    assert (
        ib.next_transaction_action(registered, ib.TransactionEvent.EXCEPTION)
        is ib.TransactionAction.DISCARD_TOKEN_PASS_THROUGH
    )


def test_transaction_exception_after_mutation_restores_s0_with_warning():
    # A partial expansion failure churned user-visible selection; the
    # rollback must not be silent.
    state = ib.TransactionState(snapshot_exists=True, token_registered=True, selection_mutated=True)
    assert (
        ib.next_transaction_action(state, ib.TransactionEvent.EXCEPTION)
        is ib.TransactionAction.RESTORE_S0_DISCARD_TOKEN_WARN_PASS_THROUGH
    )


def test_transaction_s1_failure_restores_and_warns():
    state = ib.TransactionState(snapshot_exists=True, token_registered=True, selection_mutated=True)
    assert ib.next_transaction_action(state, ib.TransactionEvent.S1_CAPTURE_FAILED) is ib.TransactionAction.RESTORE_S0_DISCARD_TOKEN_WARN_PASS_THROUGH


def test_transaction_arm_failure_retries_once_then_restores():
    state = ib.TransactionState(
        snapshot_exists=True,
        token_registered=True,
        selection_mutated=True,
        arm_attempts=1,
    )
    assert ib.next_transaction_action(state, ib.TransactionEvent.ARM_FAILED) is ib.TransactionAction.ATTEMPT_ARM
    exhausted = replace(state, arm_attempts=2)
    assert (
        ib.next_transaction_action(exhausted, ib.TransactionEvent.ARM_FAILED)
        is ib.TransactionAction.RESTORE_S0_DISCARD_TOKEN_WARN_PASS_THROUGH
    )


def test_transaction_exception_after_arm_passes_through():
    state = ib.TransactionState(
        snapshot_exists=True,
        token_registered=True,
        selection_mutated=True,
        poller_armed=True,
    )
    assert ib.next_transaction_action(state, ib.TransactionEvent.EXCEPTION) is ib.TransactionAction.PASS_THROUGH


# --- 8. §3-3 衝突述語 ----------------------------------------------------------


def test_any_true_overlaps_four_modifiers_only():
    wildcard = FakeKmi(type="I", value="PRESS", any=True)
    ctrl = FakeKmi(type="I", value="PRESS", ctrl=1)
    assert ib.kmi_events_overlap(wildcard, ctrl)
    left_shift_mod = FakeKmi(type="I", value="PRESS", any=True, key_modifier="LEFT_SHIFT")
    none_mod = FakeKmi(type="I", value="PRESS", ctrl=1, key_modifier="NONE")
    assert not ib.kmi_events_overlap(left_shift_mod, none_mod)


def test_key_modifier_is_independent_of_any():
    inset = FakeKmi(type="I", value="PRESS", key_modifier="NONE")
    with_mod = FakeKmi(type="I", value="PRESS", key_modifier="LEFT_SHIFT")
    assert not ib.kmi_events_overlap(inset, with_mod)


def test_direction_and_repeat_participate_in_overlap():
    north = FakeKmi(type="NDOF_MOTION", value="NOTHING", direction="NORTH")
    south = FakeKmi(type="NDOF_MOTION", value="NOTHING", direction="SOUTH")
    any_dir = FakeKmi(type="NDOF_MOTION", value="NOTHING", direction="ANY")
    assert not ib.kmi_events_overlap(north, south)
    assert ib.kmi_events_overlap(north, any_dir)
    press = FakeKmi(type="I", value="PRESS", repeat=False)
    repeat = FakeKmi(type="I", value="PRESS", repeat=True)
    assert not ib.kmi_events_overlap(press, repeat)


def test_i_key_coexists_with_key_modifier_variant():
    inset = FakeKmi(type="I", value="PRESS", idname="mesh.inset")
    modified = FakeKmi(type="I", value="PRESS", key_modifier="LEFT_CTRL", idname="mesh.bevel")
    items = (
        (inset, "mesh.inset", ""),
        (modified, "mesh.bevel", ""),
    )
    assert ib.count_overlapping_consumers(inset, items) == 1
    assert ib.count_overlapping_consumers(modified, items) == 1


def test_hyper_modifier_participates_in_overlap():
    plain = FakeKmi(type="I", value="PRESS")
    hyper = FakeKmi(type="I", value="PRESS", hyper=1)
    assert not ib.kmi_events_overlap(plain, hyper)
    hyper_any = FakeKmi(type="I", value="PRESS", hyper=-1)
    assert ib.kmi_events_overlap(plain, hyper_any)


def test_value_any_overlaps_press():
    any_value = FakeKmi(type="I", value="ANY")
    press = FakeKmi(type="I", value="PRESS")
    click_drag = FakeKmi(type="LEFTMOUSE", value="CLICK_DRAG")
    press_mouse = FakeKmi(type="LEFTMOUSE", value="PRESS")
    assert ib.kmi_events_overlap(any_value, press)
    assert not ib.kmi_events_overlap(click_drag, press_mouse)


def test_collision_consumer_includes_family_session_reflect_and_replay():
    assert ib.is_collision_consumer("mesh.inset")
    assert ib.is_collision_consumer("mesh.bevel")
    assert ib.is_collision_consumer("mesh.knife_tool", session_reflect_idnames=frozenset({"mesh.knife_tool"}))
    assert ib.is_collision_consumer("mesh.vert_connect_path")
    assert ib.is_collision_consumer("mesh.dissolve_mode")
    assert ib.is_collision_consumer("wm.call_menu", menu_name="VIEW3D_MT_edit_mesh_delete")
    assert not ib.is_collision_consumer("wm.call_menu", menu_name="VIEW3D_MT_view")
    assert not ib.is_collision_consumer("mesh.select_all")


# --- 9. §5 破棄条件 -----------------------------------------------------------


def test_discard_reason_enumerates_all_contract_conditions():
    assert ib.discard_reason(addon_unregistered=True) == "unregister"
    assert ib.discard_reason(edit_mode=False) == "edit_mode_exit"
    assert ib.discard_reason(object_exists=False) == "object_missing"
    assert ib.discard_reason(mesh_exists=False) == "mesh_missing"
    assert ib.discard_reason(window_exists=False) == "window_missing"
    assert ib.discard_reason(load_pre=True) == "load_pre"
    assert ib.discard_reason() is None


def test_discard_conditions_win_over_classification():
    assert _tick(addon_unregistered=True, current_op=_op("mesh.inset", 2)) is ib.TickResult.DISCARD
    assert _tick(edit_mode=False) is ib.TickResult.DISCARD
    assert _tick(object_exists=False) is ib.TickResult.DISCARD
    assert _tick(mesh_exists=False) is ib.TickResult.DISCARD
    assert _tick(window_exists=False) is ib.TickResult.DISCARD
    assert _tick(load_pre=True) is ib.TickResult.DISCARD


def test_stale_generation_wins_over_discard():
    assert _tick(token_generation=1, current_generation=9, load_pre=True) is ib.TickResult.NO_OP


class _Pairs:
    vert_pairs = {}
    edge_pair_by_index = {}
    face_pair_by_index = {0: 1, 1: 0, 2: 2}


def test_v43_selection_three_way_and_inset_self_mixed_guard():
    relation, mixed = ib.classify_selection_relation({"FACE": {0}}, _Pairs(), domains=("FACE",))
    assert relation is ib.SelectionRelation.DISJOINT and not mixed
    relation, mixed = ib.classify_selection_relation({"FACE": {0, 1}}, _Pairs(), domains=("FACE",))
    assert relation is ib.SelectionRelation.SELF_MIRRORED and not mixed
    relation, mixed = ib.classify_selection_relation({"FACE": {0, 2}}, _Pairs(), domains=("FACE",))
    assert relation is ib.SelectionRelation.DISJOINT and mixed


def test_v43_lineage_onset_identity_and_strict_counts():
    recorded = _op(None, 1)
    onset = _op("mesh.inset", 9)
    assert _classify(
        recorded_op=recorded,
        current_op=onset,
        onset=True,
        onset_op=onset,
        current_counts=_counts(9, 14, 7),
    ) is ib.PollerVerdict.CONFIRMED
    assert _classify(
        recorded_op=recorded,
        current_op=_op("mesh.inset", 10),
        onset=True,
        onset_op=onset,
        current_counts=_counts(9, 14, 7),
    ) is ib.PollerVerdict.UNKNOWN
    assert _classify(
        recorded_op=recorded,
        current_op=_op("mesh.inset", 10),
        current_counts=_counts(),
        elapsed_s=2.0,
    ) is ib.PollerVerdict.CANCELLED


def test_v44_inset_counts_primary_survives_pointer_reuse():
    # Consecutive same-type runs can hand run N the freed pointer of run
    # N-1's confirmed operator: current == recorded must not veto a confirm
    # whose counts increased (measured: the ON case after an OFF case
    # silently classified CANCELLED and lost the replay).
    reused = _op("mesh.inset", 7)
    assert _classify(
        mode="inset",
        recorded_op=reused,
        current_op=reused,
        onset=True,
        onset_op=reused,
        current_counts=_counts(9, 14, 7),
    ) is ib.PollerVerdict.CONFIRMED
    # ESC with the same reused pointer: counts restored -> CANCELLED.
    assert _classify(
        mode="inset",
        recorded_op=reused,
        current_op=reused,
        onset=True,
        onset_op=reused,
        current_counts=_counts(),
    ) is ib.PollerVerdict.CANCELLED
    # Counts grew but the active operator is not mesh.inset: interference,
    # never a replay.
    assert _classify(
        mode="inset",
        recorded_op=reused,
        current_op=_op("mesh.bevel", 8),
        onset=True,
        onset_op=reused,
        current_counts=_counts(9, 14, 7),
    ) is ib.PollerVerdict.UNKNOWN
    # No-onset quick confirm accepts on counts alone (v4.2 rule, now also
    # immune to pointer reuse).
    assert _classify(
        mode="inset",
        recorded_op=reused,
        current_op=reused,
        current_counts=_counts(9, 14, 7),
    ) is ib.PollerVerdict.CONFIRMED
    # Counts decreased after onset: never confirm, never restore.
    assert _classify(
        mode="inset",
        recorded_op=reused,
        current_op=reused,
        onset=True,
        onset_op=reused,
        current_counts=_counts(7, 10, 5),
    ) is ib.PollerVerdict.UNKNOWN


def test_v43_replay_state_machine_never_aborts_after_exec_commit():
    state = ib.InsetReplayState.WATCHING
    state = ib.next_inset_replay_state(state, ib.InsetReplayEvent.CONFIRMED)
    state = ib.next_inset_replay_state(state, ib.InsetReplayEvent.PROPS_READY)
    state = ib.next_inset_replay_state(state, ib.InsetReplayEvent.EXEC_FINISHED)
    assert state is ib.InsetReplayState.EXEC_COMMITTED
    try:
        ib.next_inset_replay_state(state, ib.InsetReplayEvent.REPLAY_FAILED)
    except ValueError:
        pass
    else:
        raise AssertionError("EXEC_COMMITTED must not transition to one-sided abort")


def test_v43_bevel_side_classes_and_three_domain_normalization():
    assert ib.classify_element_side([(0.0, 0.0, 0.0)], 0, 1e-3, 1) == "PLANE"
    assert ib.classify_element_side([(1.0, 0.0, 0.0)], 0, 1e-3, 1) == "USER"
    assert ib.classify_element_side([(-1.0, 0.0, 0.0)], 0, 1e-3, 1) == "MIRROR"
    assert ib.classify_element_side([(-1.0, 0.0, 0.0), (1.0, 0.0, 0.0)], 0, 1e-3, 1) == "SPAN"
    selected = {"VERT": {1, 2}, "EDGE": {3}, "FACE": {4}}
    classes = {("VERT", 1): "USER", ("VERT", 2): "MIRROR", ("EDGE", 3): "PLANE", ("FACE", 4): "SPAN"}
    normalized = ib.normalize_bevel_selection(selected, classes, select_mirrored=False)
    assert normalized == {"VERT": frozenset({1}), "EDGE": frozenset({3}), "FACE": frozenset({4})}
    assert ib.normalize_bevel_selection(selected, classes, select_mirrored=True)["VERT"] == frozenset({1, 2})


class _FakeElement:
    def __init__(self, index, coordinates):
        self.index = index
        if len(coordinates) == 1:
            self.co = tuple(coordinates[0])
        else:
            self.verts = [type("V", (), {"co": tuple(coordinate)})() for coordinate in coordinates]
        self.select = True
        self.hide = False


class _FakeTable(list):
    def ensure_lookup_table(self):
        return None


class _FakeBMesh:
    def __init__(self, elements):
        self.verts = _FakeTable()
        self.edges = _FakeTable(elements)
        self.faces = _FakeTable()


def test_bevel_side_scan_ignores_span_before_user_and_detects_manual_both_sides():
    bm = _FakeBMesh(
        [
            _FakeElement(0, ((-2.0, 0.0, 0.0), (1.0, 0.0, 0.0))),
            _FakeElement(1, ((1.0, 0.0, 0.0), (1.0, 1.0, 0.0))),
            _FakeElement(2, ((-1.0, 0.0, 0.0), (-1.0, 1.0, 0.0))),
        ]
    )
    user_sign, manual_both = ib._classify_bevel_sides(bm, ("EDGE",), 0, 1e-3)
    assert user_sign == 1
    assert manual_both is True


def test_inset_counts_decrease_before_onset_is_immediately_unknown():
    assert _classify(
        mode="inset",
        onset=False,
        elapsed_s=0.1,
        invoke_counts=_counts(8, 12, 6),
        current_counts=_counts(7, 10, 5),
    ) is ib.PollerVerdict.UNKNOWN


def test_inset_count_increase_under_foreign_operator_is_immediately_unknown():
    assert _classify(
        mode="inset",
        current_op=_op("mesh.bevel", 9),
        invoke_counts=_counts(8, 12, 6),
        current_counts=_counts(9, 14, 7),
    ) is ib.PollerVerdict.UNKNOWN


def test_inset_cancelled_does_not_restore_selection_or_mesh():
    token = ib._InsetBevelToken(
        generation=1,
        window_pointer=1,
        object_name="obj",
        mesh_name="mesh",
        window=None,
        area=None,
        region=None,
        s0=ib.SelectionSnapshot(frozenset({1}), frozenset(), frozenset(), _counts()),
        s1=None,
        invoke_counts=_counts(),
        recorded_op=_op(),
        native_idname="mesh.inset",
        axis_index=0,
        tolerance=1e-3,
        mode="inset",
    )
    with mock.patch.object(ib, "_restore_s0", side_effect=AssertionError("must not restore")) as restore, mock.patch.object(
        ib, "_discard_active_token"
    ) as discard:
        ib._apply_settled_verdict(token, ib.PollerVerdict.CANCELLED, staged_restore=True)
    restore.assert_not_called()
    discard.assert_called_once()


def test_inset_preamble_cancel_marks_no_restore():
    result = ib.apply_invoke_preamble(_prior_view(mode="inset"), frozenset({7}))
    assert result.verdict is ib.PollerVerdict.CANCELLED
    assert result.restored_s0 is False
    assert result.selection == frozenset({7})


class _FakeProps:
    def __init__(self, **values):
        self.__dict__.update(values)


class _FakeOperator:
    def __init__(self, idname="mesh.inset", pointer=1, **props):
        self.bl_idname = idname
        self._pointer = pointer
        self.properties = _FakeProps(**props)

    def as_pointer(self):
        return self._pointer


def test_v52_fingerprint_allowlist_and_float_quantization():
    op = _FakeOperator(
        thickness=0.123456789,
        depth=0.25,
        release_confirm=True,
        ignored="x",
    )
    value = ib.fingerprint(op, route="mesh.inset")
    assert ("thickness", 0.123457) in value
    assert ("depth", 0.25) in value
    assert all(name != "release_confirm" for name, _ in value)


def test_v52_bevel_fingerprint_reacts_to_non_offset_props():
    # An F9 that only touches segments/profile/affect must still count as a
    # property change (offset alone is not the fingerprint).
    base = _FakeOperator(offset=0.5, segments=1, profile=0.5, affect="EDGES")
    tweaked_segments = _FakeOperator(offset=0.5, segments=4, profile=0.5, affect="EDGES")
    tweaked_profile = _FakeOperator(offset=0.5, segments=1, profile=0.7, affect="EDGES")
    tweaked_affect = _FakeOperator(offset=0.5, segments=1, profile=0.5, affect="VERTICES")
    reference = ib.fingerprint(base, route="mesh.bevel")
    assert ib.fingerprint(tweaked_segments, route="mesh.bevel") != reference
    assert ib.fingerprint(tweaked_profile, route="mesh.bevel") != reference
    assert ib.fingerprint(tweaked_affect, route="mesh.bevel") != reference
    assert ib.fingerprint(
        _FakeOperator(offset=0.5, segments=1, profile=0.5, affect="EDGES"),
        route="mesh.bevel",
    ) == reference


def test_v52_f9_discriminator_matrix_and_suspend():
    op = _FakeOperator(thickness=0.1)
    identity = ib.OperatorIdentity("mesh.inset", 1)
    record = ib.CompletionRecord(7, "Obj", "Mesh", "mesh.inset", identity, ib.fingerprint(op))
    assert ib.classify_f9_intervention(None, record, window_pointer=7, object_name="Obj", mesh_name="Mesh").decision is ib.F9Decision.SUSPEND
    changed = _FakeOperator(thickness=0.2)
    result = ib.classify_f9_intervention(changed, record, window_pointer=7, object_name="Obj", mesh_name="Mesh")
    assert result.decision is ib.F9Decision.INTERVENE
    assert ib.classify_f9_intervention(changed, None, window_pointer=7, object_name="Obj", mesh_name="Mesh").decision is ib.F9Decision.NO_OP
    suspended = ib.replace(record, status=ib.CompletionStatus.SUSPENDED)
    reactivated = ib.reactivate_completion_record(suspended, _FakeOperator(pointer=99, thickness=0.1), window_pointer=7, object_name="Obj", mesh_name="Mesh")
    assert reactivated.status is ib.CompletionStatus.ACTIVE and reactivated.operator.pointer == 99


def test_v52_supersede_and_restore_only_matrix():
    assert ib.supersede_decision(repeat_origin=True, restore_only=False, mode="bevel", replay_state=ib.InsetReplayState.WATCHING) == "SUPERSEDE"
    assert ib.supersede_decision(repeat_origin=False, restore_only=True, mode="bevel", replay_state=ib.InsetReplayState.WATCHING) == "RESTORE_BEFORE_SUPERSEDE"
    assert ib.supersede_decision(repeat_origin=False, restore_only=False, mode="bevel", replay_state=ib.InsetReplayState.WATCHING) == "NO_OP_WARNING"
    assert ib.restore_only_required(restore_only=True, verdict=ib.PollerVerdict.UNKNOWN)
