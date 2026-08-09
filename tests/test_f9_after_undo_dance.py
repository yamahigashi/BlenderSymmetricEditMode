# SPDX-License-Identifier: GPL-3.0-or-later

"""GUI acceptance: F9 after undo/redo dance + second ring cut (§C-3).

Promoted from tmp/repro_f9_straddle.py (dance + cut2 + single F9).

Run::

    cmd.exe /c run_gui_test.bat 52 test_f9_after_undo_dance.py
    cmd.exe /c run_gui_test.bat 42 test_f9_after_undo_dance.py

Continuous F9 is intentionally not simulated (ed.undo_redo collapses the
stack on a second call; contract §C-3).
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from pathlib import Path

import bmesh
import bpy
from bpy_extras import view3d_utils
from mathutils import Quaternion, Vector

bpy.context.preferences.view.show_splash = False
bpy.context.preferences.view.smooth_view = 0
bpy.context.preferences.view.use_save_prompt = False
bpy.context.preferences.filepaths.use_auto_save_temporary_files = False

PACKAGE_PARENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_PARENT))

import ydd_symmetric_edit as addon  # noqa: E402
from ydd_symmetric_edit import keymaps, operators  # noqa: E402

OBJECT_NAME = "Grid"
STATE: dict = {}
MARK_OK = "YSE_F9_DANCE_OK"
MARK_FAIL = "YSE_F9_DANCE_FAILED"


def fail(msg=""):
    if msg:
        print(f"YSE_F9_DANCE_ERROR={msg}", flush=True)
    traceback.print_exc()
    for name, status in STATE.get("case_results", {}).items():
        print(f"YSE_F9_DANCE_RESULT={name}:{status}", flush=True)
    print(MARK_FAIL, flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(1)


def record_case(name: str, passed: bool) -> None:
    status = "PASS" if passed else "FAIL"
    STATE.setdefault("case_results", {})[name] = status
    print(f"YSE_F9_DANCE_RESULT={name}:{status}", flush=True)


def make_grid():
    for old in tuple(bpy.data.objects):
        bpy.data.objects.remove(old, do_unlink=True)
    mesh = bpy.data.meshes.new("Grid")
    verts = []
    faces = []
    xs = [-2.0, -1.0, 0.0, 1.0, 2.0]
    ys = [-1.0, 1.0]
    index = {}
    for xi, x in enumerate(xs):
        for yi, y in enumerate(ys):
            index[(xi, yi)] = len(verts)
            verts.append((x, y, 0.0))
    for xi in range(len(xs) - 1):
        faces.append(
            (
                index[(xi, 0)],
                index[(xi + 1, 0)],
                index[(xi + 1, 1)],
                index[(xi, 1)],
            )
        )
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(OBJECT_NAME, mesh)
    bpy.context.scene.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    obj.use_mesh_mirror_x = True
    obj.use_mesh_mirror_y = False
    obj.use_mesh_mirror_z = False
    return obj


def window_coordinate(coordinate):
    local = view3d_utils.location_3d_to_region_2d(
        STATE["region"],
        STATE["area"].spaces.active.region_3d,
        Vector(coordinate),
    )
    assert local is not None
    return (
        int(round(STATE["region"].x + local.x)),
        int(round(STATE["region"].y + local.y)),
    )


def mirror_flags():
    obj = bpy.data.objects[OBJECT_NAME]
    return (obj.use_mesh_mirror_x, obj.use_mesh_mirror_y, obj.use_mesh_mirror_z)


def symmetric_state():
    bm = bmesh.from_edit_mesh(bpy.data.objects[OBJECT_NAME].data)
    verts = sorted(tuple(round(float(c), 5) for c in v.co) for v in bm.verts)
    mirrored = sorted((round(-x, 5), y, z) for x, y, z in verts)
    edges = sorted(
        tuple(sorted(tuple(round(float(c), 5) for c in v.co) for v in e.verts)) for e in bm.edges
    )
    mirrored_edges = sorted(
        tuple(
            sorted(
                (round(-x, 5), y, z)
                for x, y, z in (tuple(round(float(c), 5) for c in v.co) for v in e.verts)
            )
        )
        for e in bm.edges
    )
    return verts == mirrored and edges == mirrored_edges, len(bm.verts), len(bm.edges)


def busy():
    native_modal = any(session.saw_modal for session in operators._SESSIONS.values())
    return bool(
        native_modal
        or operators._SESSIONS
        or operators._HISTORY_REPAIR_QUEUED
        or operators._HISTORY_REPAIR_BUSY
    )


def route_ready():
    route_keys = {
        route.route_key
        for route in keymaps._ROUTES_BY_KEY.values()
        if route.native_operator == "mesh.loopcut_slide"
        and route.keymap_name == "Mesh"
        and route.event.type == "R"
        and route.event.value == "PRESS"
        and route.event.ctrl
    }
    if not route_keys:
        return False
    return any(
        item.active
        and item.idname == keymaps.INTERCEPT_OPERATOR
        and getattr(item.properties, "route_key", "") in route_keys
        for _keymap, item in keymaps._REGISTERED_ITEMS
    )


def active_loopcut_children():
    operator = bpy.context.active_operator
    assert operator is not None
    macros = getattr(operator, "macros", None)
    if not macros:
        return None
    return {child.bl_idname: child.properties for child in macros}


def instrument_f9_prepare():
    """Capture whether F9 discrimination prepared a session (not bare native)."""
    original = operators._prepare_adjust_last_operation_repeat

    def logged():
        result = original()
        STATE["f9_prepare_result"] = result
        STATE["f9_sessions_after_prepare"] = len(operators._SESSIONS)
        print(
            f"YSE_F9_DANCE_DISCRIMINATOR result={result} "
            f"sessions={STATE['f9_sessions_after_prepare']}",
            flush=True,
        )
        return result

    operators._prepare_adjust_last_operation_repeat = logged


def start():
    try:
        STATE["case_results"] = {}
        addon.register()
        instrument_f9_prepare()
        addon.sync_persistent_keymap(True)
        window = bpy.context.window_manager.windows[0]
        area = next(a for a in window.screen.areas if a.type == "VIEW_3D")
        region = next(r for r in area.regions if r.type == "WINDOW")
        STATE.update(window=window, area=area, region=region)
        region_3d = area.spaces.active.region_3d
        region_3d.view_perspective = "ORTHO"
        region_3d.view_rotation = Quaternion((1.0, 0.0, 0.0, 0.0))
        region_3d.view_location = (0.0, 0.0, 0.0)
        region_3d.view_distance = 7.0
        region_3d.update()

        obj = make_grid()
        STATE["obj"] = obj
        with bpy.context.temp_override(window=window, area=area, region=region):
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.context.tool_settings.mesh_select_mode = (False, True, False)
            bpy.context.tool_settings.workspace_tool_type = "DEFAULT"
            bpy.ops.mesh.select_all(action="DESELECT")
            bpy.ops.wm.tool_set_by_id(name="builtin.select_box")
            # Dense undo stack so ed.undo_redo after F9 does not collapse.
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.mesh.select_all(action="DESELECT")
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.mesh.select_all(action="DESELECT")
            bpy.ops.ed.undo_push(message="YSE F9 dance baseline")

        STATE["deadline"] = time.monotonic() + 10.0
        bpy.app.timers.register(wait_route, first_interval=0.2)
    except BaseException:
        fail()
    return None


def wait_route():
    try:
        if route_ready():
            bpy.app.timers.register(
                lambda: begin_cut((-0.7, 0.0, 0.0), (-0.7, 0.2, 0.0), wait_cut),
                first_interval=0.3,
            )
            return None
        if time.monotonic() > STATE["deadline"]:
            raise RuntimeError("intercept route not ready")
        return 0.05
    except BaseException:
        fail()
    return None


def begin_cut(hover_3d, confirm_3d, next_wait):
    try:
        hover = window_coordinate(hover_3d)
        confirm = window_coordinate(confirm_3d)
        STATE["after_events"] = next_wait
        STATE["events"] = [
            {"type": "ESC", "value": "PRESS", "x": hover[0], "y": hover[1]},
            {"type": "ESC", "value": "RELEASE", "x": hover[0], "y": hover[1]},
            {"type": "MOUSEMOVE", "value": "NOTHING", "x": hover[0], "y": hover[1]},
            {"type": "R", "value": "PRESS", "ctrl": True, "x": hover[0], "y": hover[1]},
            {"type": "R", "value": "RELEASE", "ctrl": True, "x": hover[0], "y": hover[1]},
            {"type": "WHEELUPMOUSE", "value": "PRESS", "x": hover[0], "y": hover[1]},
            {"type": "LEFTMOUSE", "value": "PRESS", "x": hover[0], "y": hover[1]},
            {"type": "LEFTMOUSE", "value": "RELEASE", "x": hover[0], "y": hover[1]},
            {"type": "MOUSEMOVE", "value": "NOTHING", "x": confirm[0], "y": confirm[1]},
            {"type": "LEFTMOUSE", "value": "PRESS", "x": confirm[0], "y": confirm[1]},
            {"type": "LEFTMOUSE", "value": "RELEASE", "x": confirm[0], "y": confirm[1]},
        ]
        bpy.app.timers.register(send_events, first_interval=0.05)
    except BaseException:
        fail()
    return None


def send_events():
    try:
        if STATE["events"]:
            STATE["window"].event_simulate(**STATE["events"].pop(0))
            return 0.15
        STATE["deadline"] = time.monotonic() + 15.0
        bpy.app.timers.register(STATE.get("after_events") or wait_cut, first_interval=0.1)
    except BaseException:
        fail()
    return None


def wait_cut():
    try:
        state_now = symmetric_state()
        if busy() or state_now[1] <= 10 or not state_now[0]:
            if time.monotonic() > STATE["deadline"]:
                raise RuntimeError(
                    f"timed out waiting for straddling cut: verts={symmetric_state()[1]}"
                    f" sessions={len(operators._SESSIONS)}"
                )
            return 0.05
        sym, nverts, nedges = symmetric_state()
        print(
            f"YSE_F9_DANCE_AFTER_CUT1 sym={sym} verts={nverts} edges={nedges} flags={mirror_flags()}",
            flush=True,
        )
        assert sym, f"initial straddling cut is not symmetric ({nverts})"
        assert mirror_flags() == (True, False, False), mirror_flags()
        record_case("straddle_cut1_symmetric", True)
        bpy.app.timers.register(do_undo_redo_dance, first_interval=0.5)
    except BaseException:
        record_case("straddle_cut1_symmetric", False)
        fail()
    return None


def do_undo_redo_dance():
    try:
        with bpy.context.temp_override(
            window=STATE["window"], area=STATE["area"], region=STATE["region"]
        ):
            r1 = bpy.ops.ed.undo()
        print(f"YSE_F9_DANCE_UNDO={r1}", flush=True)
        bpy.app.timers.register(dance_redo, first_interval=1.0)
    except BaseException:
        fail()
    return None


def dance_redo():
    try:
        if busy():
            return 0.1
        with bpy.context.temp_override(
            window=STATE["window"], area=STATE["area"], region=STATE["region"]
        ):
            r2 = bpy.ops.ed.redo()
        print(f"YSE_F9_DANCE_REDO={r2}", flush=True)
        bpy.app.timers.register(dance_settle, first_interval=1.0)
    except BaseException:
        fail()
    return None


def dance_settle():
    try:
        if busy():
            return 0.1
        sym, nverts, nedges = symmetric_state()
        print(f"YSE_F9_DANCE_AFTER_DANCE sym={sym} verts={nverts}", flush=True)
        assert sym, "post-dance state is not symmetric"
        assert mirror_flags() == (True, False, False), mirror_flags()
        record_case("undo_redo_dance", True)
        STATE["cut2_base"] = nverts
        with bpy.context.temp_override(
            window=STATE["window"], area=STATE["area"], region=STATE["region"]
        ):
            bpy.ops.ed.undo_push(message="YSE between dance and cut2")
        bpy.app.timers.register(
            lambda: begin_cut((-1.5, -1.0, 0.0), (-1.68, 0.0, 0.0), wait_cut2),
            first_interval=0.5,
        )
    except BaseException:
        record_case("undo_redo_dance", False)
        fail()
    return None


def wait_cut2():
    try:
        state_now = symmetric_state()
        if busy() or state_now[1] <= STATE["cut2_base"] or not state_now[0]:
            if time.monotonic() > STATE["deadline"]:
                raise RuntimeError(f"timed out waiting for cut2: {state_now}")
            return 0.05
        print(
            f"YSE_F9_DANCE_CUT2 sym={state_now[0]} verts={state_now[1]}",
            flush=True,
        )
        assert state_now[0], "cut2 is not symmetric"
        assert mirror_flags() == (True, False, False), mirror_flags()
        record_case("side_cut2_symmetric", True)
        STATE["initial"] = (state_now[1], state_now[2])
        bpy.app.timers.register(adjust_f9, first_interval=0.5)
    except BaseException:
        record_case("side_cut2_symmetric", False)
        fail()
    return None


def adjust_f9():
    try:
        children = active_loopcut_children()
        assert children is not None, "no active macro operator for F9"
        loopcut = children.get("MESH_OT_loopcut")
        assert loopcut is not None, list(children)
        print(f"YSE_F9_DANCE_BEFORE_F9 number_cuts={loopcut.number_cuts}", flush=True)
        STATE["f9_prepare_result"] = None
        STATE["f9_sessions_after_prepare"] = None
        loopcut.number_cuts = loopcut.number_cuts + 1
        with bpy.context.temp_override(
            window=STATE["window"],
            area=STATE["area"],
            region=STATE["region"],
        ):
            result = bpy.ops.ed.undo_redo()
        print(f"YSE_F9_DANCE_UNDO_REDO={result}", flush=True)
        STATE["deadline"] = time.monotonic() + 15.0
        bpy.app.timers.register(wait_adjusted, first_interval=0.1)
    except BaseException:
        fail()
    return None


def wait_adjusted():
    try:
        if busy():
            if time.monotonic() > STATE["deadline"]:
                raise RuntimeError("timed out waiting for F9 adjusted loop cut")
            return 0.05
        if not STATE.get("settled"):
            STATE["settled"] = True
            return 0.5
        sym, nverts, nedges = symmetric_state()
        flags = mirror_flags()
        print(
            f"YSE_F9_DANCE_AFTER_F9 sym={sym} verts={nverts} edges={nedges} flags={flags}",
            flush=True,
        )
        assert flags == (True, False, False), f"mirror flags changed: {flags}"
        assert sym, f"F9 adjust is not symmetric (verts={nverts})"
        record_case("f9_symmetric_and_flags", True)

        prepare_result = STATE.get("f9_prepare_result")
        sessions_after = STATE.get("f9_sessions_after_prepare")
        assert prepare_result is True, (
            f"F9 discriminator did not prepare a session (result={prepare_result})"
        )
        assert sessions_after and sessions_after > 0, (
            f"F9 prepare returned True but no session was created ({sessions_after})"
        )
        record_case("f9_prepared_session", True)

        print(MARK_OK, flush=True)
        sys.stdout.flush()
        addon.unregister()
        bpy.ops.wm.quit_blender()
    except BaseException:
        if "f9_symmetric_and_flags" not in STATE.get("case_results", {}):
            record_case("f9_symmetric_and_flags", False)
        if "f9_prepared_session" not in STATE.get("case_results", {}):
            record_case("f9_prepared_session", False)
        fail()
    return None


try:
    bpy.app.timers.register(start, first_interval=0.5)
except BaseException:
    fail()
