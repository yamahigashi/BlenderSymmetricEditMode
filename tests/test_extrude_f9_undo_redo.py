# SPDX-License-Identifier: GPL-3.0-or-later

"""GUI representative: plain Undo restore + Redo repair_after_redo.

Plan: .agents/doc/f9_extrude_plan_2026-08-15.md v3.1 §4-2 row 12.
Does not call ed.undo_redo.

Run::

    cmd.exe /c "tmp\\run_menu_reg.bat 42 test_extrude_f9_undo_redo.py"
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from collections import Counter
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
from ydd_symmetric_edit import history as history_module  # noqa: E402
from ydd_symmetric_edit import layer_names, operators  # noqa: E402

OBJECT_NAME = "YSE_F9UndoRedo"
MESH_NAME = "YSE_F9UndoRedoMesh"
MARKER_OK = "YSE_EXTRUDE_F9_UNDO_REDO_OK"
MARKER_FAILED = "YSE_EXTRUDE_F9_UNDO_REDO_FAILED"
TOOL_KIND = "EXTRUDE_NORMAL"
NX, NY = 6, 4
PRECISION = 5
EXPECTED_NET = (8, 16, 8)
FAILSAFE_S = 90.0
STATE: dict = {"phase": "startup", "finished": False}


def fail(message=""):
    if message:
        print(f"YSE_EXTRUDE_F9_UNDO_REDO_ERROR={message}", flush=True)
    traceback.print_exc()
    print(f"YSE_EXTRUDE_F9_UNDO_REDO_PHASE={STATE.get('phase')}", flush=True)
    print(MARKER_FAILED, flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(1)


def test_object():
    return bpy.data.objects[OBJECT_NAME]


def override():
    return bpy.context.temp_override(window=STATE["window"], area=STATE["area"], region=STATE["region"])


def click_drag_threshold_px():
    inputs = bpy.context.preferences.inputs
    if hasattr(inputs, "drag_threshold_mouse"):
        base = int(inputs.drag_threshold_mouse)
    elif hasattr(inputs, "drag_threshold"):
        base = int(inputs.drag_threshold)
    else:
        raise RuntimeError("preferences.inputs exposes neither drag_threshold_mouse nor drag_threshold")
    scale = float(getattr(bpy.context.preferences.system, "ui_scale", 1.0))
    return max(1, int(round(base * scale)) + 1)


def configure_view(area):
    region_3d = area.spaces.active.region_3d
    region_3d.view_perspective = "ORTHO"
    region_3d.view_rotation = Quaternion((1.0, 0.0, 0.0, 0.0))
    region_3d.view_location = (0.0, 0.0, 0.0)
    region_3d.view_distance = 8.0
    region_3d.update()


def window_coordinate(coordinate):
    region = STATE["region"]
    region_3d = STATE["area"].spaces.active.region_3d
    local = view3d_utils.location_3d_to_region_2d(region, region_3d, Vector(coordinate))
    if local is None:
        raise RuntimeError(f"could not project {coordinate}")
    return int(round(region.x + local.x)), int(round(region.y + local.y))


def grid_xy(i, j):
    return (i - NX / 2, j - NY / 2)


def build_mesh():
    for old in tuple(bpy.data.objects):
        bpy.data.objects.remove(old, do_unlink=True)
    for old_mesh in tuple(bpy.data.meshes):
        if old_mesh.users == 0:
            bpy.data.meshes.remove(old_mesh)
    mesh = bpy.data.meshes.new(MESH_NAME)
    coords, faces = [], []
    for j in range(NY + 1):
        for i in range(NX + 1):
            coords.append((i - NX / 2, j - NY / 2, 0.0))
    stride = NX + 1
    for j in range(NY):
        for i in range(NX):
            a = j * stride + i
            faces.append((a, a + 1, a + stride + 1, a + stride))
    mesh.from_pydata(coords, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(OBJECT_NAME, mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.use_mesh_mirror_x = True
    obj.use_mesh_mirror_y = False
    obj.use_mesh_mirror_z = False
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    with override():
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.context.tool_settings.mesh_select_mode = (False, False, True)
        bpy.ops.ed.undo_push(message="YSE F9 undo/redo baseline")
    return obj


def grid_face(bm, i, j):
    wanted = {
        grid_xy(i, j),
        grid_xy(i + 1, j),
        grid_xy(i + 1, j + 1),
        grid_xy(i, j + 1),
    }
    for face in bm.faces:
        have = {(round(float(vertex.co.x), 6), round(float(vertex.co.y), 6)) for vertex in face.verts}
        expect = {(round(x, 6), round(y, 6)) for x, y in wanted}
        if have == expect:
            return face
    raise AssertionError(f"grid face {i},{j} not found")


def select_single_face(bm):
    bpy.context.tool_settings.mesh_select_mode = (False, False, True)
    for face in bm.faces:
        face.select = False
    for edge in bm.edges:
        edge.select = False
    for vertex in bm.verts:
        vertex.select = False
    bm.select_history.clear()
    grid_face(bm, 4, 1).select = True
    bm.select_flush_mode()


def coordinate_key(co):
    return tuple(round(float(value), PRECISION) for value in co)


def assert_x_symmetric(bm):
    live = Counter(coordinate_key(vertex.co) for vertex in bm.verts)
    mirrored = Counter((-x, y, z) for x, y, z in live.elements())
    assert live == mirrored, "vertex coordinates are not X-symmetric"


def assert_layers_removed(bm):
    for name in layer_names.TEMP_LAYER_NAMES:
        for sequence in (bm.verts, bm.edges, bm.faces):
            assert sequence.layers.int.get(name) is None, f"temporary layer leaked: {name}"


def topology_counts(bm):
    return len(bm.verts), len(bm.edges), len(bm.faces)


def live_net(bm):
    base = STATE["baseline"]
    return (
        len(bm.verts) - base[0],
        len(bm.edges) - base[1],
        len(bm.faces) - base[2],
    )


def latest_record():
    records = [
        record
        for record in operators._HISTORY_RECORDS.values()
        if record.session.object_name == OBJECT_NAME and record.session.tool_kind == TOOL_KIND
    ]
    return max(records, key=lambda record: record.sequence) if records else None


def send_events(events, done, index=0, interval=0.09, done_delay=0.2):
    def step():
        try:
            if index < len(events):
                STATE["window"].event_simulate(**events[index])
                send_events(events, done, index + 1, interval=interval, done_delay=done_delay)
            else:
                bpy.app.timers.register(done, first_interval=done_delay)
        except BaseException:
            fail()
        return None

    bpy.app.timers.register(step, first_interval=interval)


def wait_settled(done, started=None):
    started = time.monotonic() if started is None else started

    def poll():
        try:
            busy = bool(
                STATE["window"].modal_operators
                or operators._SESSIONS
                or operators._HISTORY_REPAIR_QUEUED
                or operators._HISTORY_REPAIR_BUSY
            )
            if busy:
                if time.monotonic() - started > 12.0:
                    raise RuntimeError(
                        f"flow never settled; modal={[op.bl_idname for op in STATE['window'].modal_operators]} "
                        f"sessions={list(operators._SESSIONS)}"
                    )
                return 0.1
            bpy.app.timers.register(done, first_interval=0.25)
        except BaseException:
            fail()
        return None

    bpy.app.timers.register(poll, first_interval=0.1)


def instrument_repair():
    original = history_module.repair_after_redo

    def logged(dummy=None):
        STATE["redo_repair_called"] = STATE.get("redo_repair_called", 0) + 1
        print(f"YSE_EXTRUDE_F9_UNDO_REDO_REPAIR_AFTER_REDO={STATE['redo_repair_called']}", flush=True)
        return original(dummy)

    history_module.repair_after_redo = logged
    handlers = bpy.app.handlers.redo_post
    if original in handlers:
        handlers[handlers.index(original)] = logged


def extrude_events(cursor_xyz, drag=(0, 80)):
    x, y = window_coordinate(cursor_xyz)
    tx, ty = x + drag[0], y + drag[1]
    return [
        {"type": "MOUSEMOVE", "value": "NOTHING", "x": x, "y": y},
        {"type": "E", "value": "PRESS", "x": x, "y": y},
        {"type": "E", "value": "RELEASE", "x": x, "y": y},
        {"type": "MOUSEMOVE", "value": "NOTHING", "x": tx, "y": ty},
        {"type": "LEFTMOUSE", "value": "PRESS", "x": tx, "y": ty},
        {"type": "LEFTMOUSE", "value": "RELEASE", "x": tx, "y": ty},
    ]


def after_extrude():
    try:
        STATE["phase"] = "after_extrude"
        obj = test_object()
        bm = bmesh.from_edit_mesh(obj.data)
        got = live_net(bm)
        print(f"YSE_EXTRUDE_F9_UNDO_REDO_NET_AFTER_EXTRUDE={got}", flush=True)
        assert got == EXPECTED_NET, f"net {got} != {EXPECTED_NET}"
        assert_x_symmetric(bm)
        assert_layers_removed(bm)
        record = latest_record()
        assert record is not None, "missing COMMITTED extrude record"
        STATE["old_token"] = record.session.history_token
        bpy.app.timers.register(do_undo, first_interval=0.4)
    except BaseException:
        fail()
    return None


def do_undo():
    try:
        STATE["phase"] = "undo"
        with override():
            result = bpy.ops.ed.undo()
        print(f"YSE_EXTRUDE_F9_UNDO_REDO_UNDO={result}", flush=True)
        assert result == {"FINISHED"}, result
        wait_settled(after_undo)
    except BaseException:
        fail()
    return None


def after_undo():
    try:
        STATE["phase"] = "after_undo"
        obj = test_object()
        bm = bmesh.from_edit_mesh(obj.data)
        counts = topology_counts(bm)
        print(f"YSE_EXTRUDE_F9_UNDO_REDO_AFTER_UNDO={counts}", flush=True)
        assert counts == STATE["baseline"], f"one undo did not restore baseline: {counts}"
        assert_x_symmetric(bm)
        assert_layers_removed(bm)
        bpy.app.timers.register(do_redo, first_interval=0.5)
    except BaseException:
        fail()
    return None


def do_redo():
    try:
        STATE["phase"] = "redo"
        with override():
            result = bpy.ops.ed.redo()
        print(f"YSE_EXTRUDE_F9_UNDO_REDO_REDO={result}", flush=True)
        assert result == {"FINISHED"}, result
        wait_settled(after_redo)
    except BaseException:
        fail()
    return None


def after_redo():
    try:
        STATE["phase"] = "after_redo"
        obj = test_object()
        bm = bmesh.from_edit_mesh(obj.data)
        got = live_net(bm)
        print(f"YSE_EXTRUDE_F9_UNDO_REDO_NET_AFTER_REDO={got}", flush=True)
        assert got == EXPECTED_NET, f"redo remirror net {got} != {EXPECTED_NET}"
        assert_x_symmetric(bm)
        assert_layers_removed(bm)
        record = latest_record()
        assert record is not None, "missing record after redo"
        assert record.session.history_token == STATE["old_token"], "plain redo must reuse the committed record"
        assert STATE.get("redo_repair_called", 0) >= 1, "repair_after_redo did not run"
        STATE["finished"] = True
        print(MARKER_OK, flush=True)
        sys.stdout.flush()
        addon.unregister()
        bpy.ops.wm.quit_blender()
    except BaseException:
        fail()
    return None


def failsafe():
    if STATE.get("finished"):
        return None
    fail(f"failsafe quit timer expired in phase={STATE.get('phase')}")
    return None


def start():
    try:
        addon.register()
        instrument_repair()
        addon.sync_persistent_keymap(True)
        window = bpy.context.window_manager.windows[0]
        area = next(item for item in window.screen.areas if item.type == "VIEW_3D")
        region = next(item for item in area.regions if item.type == "WINDOW")
        STATE.update(window=window, area=area, region=region)
        configure_view(area)
        build_mesh()
        obj = test_object()
        bm = bmesh.from_edit_mesh(obj.data)
        select_single_face(bm)
        bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
        with override():
            push = bpy.ops.ed.undo_push(message="YSE F9 undo/redo prepared")
        assert push == {"FINISHED"}, push
        STATE["baseline"] = topology_counts(bmesh.from_edit_mesh(obj.data))
        print(
            f"YSE_EXTRUDE_F9_UNDO_REDO_BASELINE={STATE['baseline']} drag_threshold={click_drag_threshold_px()}",
            flush=True,
        )
        events = extrude_events((1.5, -0.5, 0.0), drag=(0, 80))
        send_events(events, lambda: wait_settled(after_extrude))
    except BaseException:
        fail()
    return None


bpy.app.timers.register(failsafe, first_interval=FAILSAFE_S)
bpy.app.timers.register(start, first_interval=0.4)
