# SPDX-License-Identifier: GPL-3.0-or-later

"""Pure-Python unit tests for Inset/Bevel EXPAND_PASSTHROUGH helpers."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass, replace
from pathlib import Path

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


def test_quick_release_confirm_evidence_before_onset_is_confirmed():
    assert (
        _classify(
            recorded_op=_op(None, None),
            current_op=_op("mesh.inset", 11),
            onset=False,
            modal_alive=False,
            elapsed_s=0.0,
        )
        is ib.PollerVerdict.CONFIRMED
    )


# --- 3. 同一 tick で timeout と onset/確定証拠が並立 → onset/証拠が勝つ --------


def test_same_tick_timeout_loses_to_confirm_evidence():
    assert (
        _classify(
            recorded_op=_op(None, None),
            current_op=_op("mesh.bevel", 3),
            elapsed_s=2.0,
            selection_matches_s1=True,
        )
        is ib.PollerVerdict.CONFIRMED
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


def test_same_identity_changed_counts_downgrades_to_unknown():
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


def test_transaction_exception_after_mutation_restores_s0():
    state = ib.TransactionState(snapshot_exists=True, token_registered=True, selection_mutated=True)
    assert (
        ib.next_transaction_action(state, ib.TransactionEvent.EXCEPTION)
        is ib.TransactionAction.RESTORE_S0_DISCARD_TOKEN_PASS_THROUGH
    )


def test_transaction_push_failure_restores_and_warns():
    state = ib.TransactionState(snapshot_exists=True, token_registered=True, selection_mutated=True)
    assert (
        ib.next_transaction_action(state, ib.TransactionEvent.PUSH_FAILED)
        is ib.TransactionAction.RESTORE_S0_DISCARD_TOKEN_WARN_PASS_THROUGH
    )


def test_transaction_arm_failure_retries_once_then_leaves_expanded():
    state = ib.TransactionState(
        snapshot_exists=True,
        token_registered=True,
        selection_mutated=True,
        push_finished=True,
        arm_attempts=1,
    )
    assert ib.next_transaction_action(state, ib.TransactionEvent.ARM_FAILED) is ib.TransactionAction.ATTEMPT_ARM
    exhausted = replace(state, arm_attempts=2)
    assert (
        ib.next_transaction_action(exhausted, ib.TransactionEvent.ARM_FAILED)
        is ib.TransactionAction.LEAVE_EXPANDED_WARN_DISCARD_TOKEN
    )


def test_transaction_exception_after_push_attempts_arm():
    state = ib.TransactionState(
        snapshot_exists=True,
        token_registered=True,
        selection_mutated=True,
        push_finished=True,
    )
    assert ib.next_transaction_action(state, ib.TransactionEvent.EXCEPTION) is ib.TransactionAction.ATTEMPT_ARM


def test_transaction_exception_after_arm_passes_through():
    state = ib.TransactionState(
        snapshot_exists=True,
        token_registered=True,
        selection_mutated=True,
        push_finished=True,
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
