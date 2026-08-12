# SPDX-License-Identifier: GPL-3.0-or-later

"""Headless unit checks for the finish/delete/dissolve GC gate (perf epoch §I-1).

Each of the 6 gated ``execute`` methods is invoked through its unbound class
attribute with a lightweight ``SimpleNamespace`` stand-in operator (same
pattern as ``test_perf_equivalence.py``), so no real mesh/session is needed.
A module-level function called early in each ``execute`` body is monkeypatched
as an observation/exception hook and to force the native/early-return path.
"""

from __future__ import annotations

import gc
import math
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

import bpy

PACKAGE_PARENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_PARENT))

from ydd_symmetric_edit import delete_dissolve, operators, session_state  # noqa: E402

_FINISH_WINDOW_SENTINEL = 0x5E1F_C0DE


class _ProbeError(Exception):
    """Marker exception raised from test hooks to verify finally-restore."""


def _check_gc_gate(invoke, expected_result) -> None:
    # (a) prior enabled -> restored enabled.
    gc.enable()
    result = invoke()
    assert result == expected_result, result
    assert gc.isenabled() is True

    # (b) prior disabled -> restored disabled.
    gc.disable()
    result = invoke()
    assert result == expected_result, result
    assert gc.isenabled() is False
    gc.enable()

    # (c) gc observed disabled during the execute body.
    observed = []
    gc.enable()
    result = invoke(observe=lambda: observed.append(gc.isenabled()))
    assert observed == [False], observed
    assert result == expected_result, result
    assert gc.isenabled() is True

    # (d) exception during execute body still restores prior state.
    gc.disable()
    try:
        invoke(raise_exc=True)
    except _ProbeError:
        pass
    else:
        raise AssertionError("expected _ProbeError to propagate")
    assert gc.isenabled() is False
    gc.enable()

    # (e) native passthrough / early-return path restores as well.
    gc.enable()
    result = invoke()
    assert result == expected_result, result
    assert gc.isenabled() is True


def _invoke_finish(*, observe=None, raise_exc=False):
    session_state._SESSIONS.pop(_FINISH_WINDOW_SENTINEL, None)
    fake_operator = SimpleNamespace(
        report=lambda level, message: None,
        preserve_history_layers=False,
    )

    def hook(context):
        if raise_exc:
            raise _ProbeError("boom")
        if observe is not None:
            observe()
        return _FINISH_WINDOW_SENTINEL

    original = operators._window_key
    operators._window_key = hook
    try:
        return operators.MESH_OT_ydd_symmetric_edit_finish.execute(fake_operator, bpy.context)
    finally:
        operators._window_key = original


def _invoke_delete_like(operator_cls, *, observe=None, raise_exc=False):
    fake_operator = SimpleNamespace(_native=lambda: {"FINISHED"})

    def hook(context):
        if raise_exc:
            raise _ProbeError("boom")
        if observe is not None:
            observe()
        return None

    original_sessions = dict(session_state._SESSIONS)
    session_state._SESSIONS.clear()
    original = delete_dissolve._symmetry_parameters
    delete_dissolve._symmetry_parameters = hook
    try:
        return operator_cls.execute(fake_operator, bpy.context)
    finally:
        delete_dissolve._symmetry_parameters = original
        session_state._SESSIONS.clear()
        session_state._SESSIONS.update(original_sessions)


class _FakeMeshOps:
    def __init__(self, dissolve_stub) -> None:
        self._dissolve_stub = dissolve_stub

    def __getattr__(self, name):
        if name == "ydd_symmetric_edit_dissolve":
            return self._dissolve_stub
        raise AttributeError(name)


class _FakeBpyModule:
    def __init__(self, dissolve_stub) -> None:
        self.ops = SimpleNamespace(mesh=_FakeMeshOps(dissolve_stub))


def _invoke_dissolve_mode(*, observe=None, raise_exc=False):
    fake_operator = SimpleNamespace(
        use_verts=False,
        use_face_split=False,
        use_boundary_tear=False,
        angle_threshold=math.pi,
        use_preserve_quads=True,
    )

    def hook(select_mode):
        if raise_exc:
            raise _ProbeError("boom")
        if observe is not None:
            observe()
        return "VERTS"

    def dissolve_stub(**kwargs):
        return {"FINISHED"}

    original_mode_fn = delete_dissolve._dissolve_mode_from_select_mode
    original_bpy = delete_dissolve.bpy
    delete_dissolve._dissolve_mode_from_select_mode = hook
    delete_dissolve.bpy = _FakeBpyModule(dissolve_stub)
    try:
        return delete_dissolve.MESH_OT_ydd_symmetric_edit_dissolve_mode.execute(fake_operator, bpy.context)
    finally:
        delete_dissolve._dissolve_mode_from_select_mode = original_mode_fn
        delete_dissolve.bpy = original_bpy


def check_finish_gc_gate() -> None:
    _check_gc_gate(_invoke_finish, {"CANCELLED"})


def check_delete_gc_gate() -> None:
    _check_gc_gate(
        lambda **kw: _invoke_delete_like(delete_dissolve.MESH_OT_ydd_symmetric_edit_delete, **kw),
        {"FINISHED"},
    )


def check_dissolve_gc_gate() -> None:
    _check_gc_gate(
        lambda **kw: _invoke_delete_like(delete_dissolve.MESH_OT_ydd_symmetric_edit_dissolve, **kw),
        {"FINISHED"},
    )


def check_edge_collapse_gc_gate() -> None:
    _check_gc_gate(
        lambda **kw: _invoke_delete_like(delete_dissolve.MESH_OT_ydd_symmetric_edit_edge_collapse, **kw),
        {"FINISHED"},
    )


def check_delete_edgeloop_gc_gate() -> None:
    _check_gc_gate(
        lambda **kw: _invoke_delete_like(delete_dissolve.MESH_OT_ydd_symmetric_edit_delete_edgeloop, **kw),
        {"FINISHED"},
    )


def check_dissolve_mode_gc_gate() -> None:
    _check_gc_gate(_invoke_dissolve_mode, {"FINISHED"})


def run() -> None:
    assert gc.isenabled(), "test process must start with gc enabled"
    check_finish_gc_gate()
    check_delete_gc_gate()
    check_dissolve_gc_gate()
    check_edge_collapse_gc_gate()
    check_delete_edgeloop_gc_gate()
    check_dissolve_mode_gc_gate()
    assert gc.isenabled()
    print("YSE_GC_GATE_TEST_OK", flush=True)


if __name__ == "__main__":
    try:
        run()
    except Exception:
        traceback.print_exc()
        print("YSE_GC_GATE_TEST_FAILED", flush=True)
        sys.exit(1)
