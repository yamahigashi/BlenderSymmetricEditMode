# SPDX-License-Identifier: GPL-3.0-or-later

"""GUI representative: loopcut COMMITTED then extrude raw; F9/repair do not mix.

Plan: .agents/doc/f9_extrude_plan_2026-08-15.md v3.1 §4-2 row 13.

Run::

    cmd.exe /c "tmp\\run_menu_reg.bat 42 test_extrude_f9_hetero_a.py"
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
from ydd_symmetric_edit import keymaps, layer_names, operators  # noqa: E402

OBJECT_NAME = "YSE_F9HeteroA"
MESH_NAME = "YSE_F9HeteroAMesh"
MARKER_OK = "YSE_EXTRUDE_F9_HETERO_A_OK"
MARKER_FAILED = "YSE_EXTRUDE_F9_HETERO_A_FAILED"
FAILSAFE_S = 120.0
STATE: dict = {"phase": "startup", "finished": False, "undo_redo_used": 0}


def fail(message=""):
    if message:
        print(f"YSE_EXTRUDE_F9_HETERO_A_ERROR={message}", flush=True)
    traceback.print_exc()
    print(f"YSE_EXTRUDE_F9_HETERO_A_PHASE={STATE.get('phase')}", flush=True)
    print(MARKER_FAILED, flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(1)


def test_object():
    return bpy.data.objects[OBJECT_NAME]


def override():
    return bpy.context.temp_override(window=STATE["window"], area=STATE["area"], region=STATE["region"])


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


def build_mesh():
    for old in tuple(bpy.data.objects):
        bpy.data.objects.remove(old, do_unlink=True)
    mesh = bpy.data.meshes.new(MESH_NAME)
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
        faces.append((index[(xi, 0)], index[(xi + 1, 0)], index[(xi + 1, 1)], index[(xi, 1)]))
    mesh.from_pydata(verts, [], faces)
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
        bpy.context.tool_settings.mesh_select_mode = (False, True, False)
        bpy.ops.mesh.select_all(action="DESELECT")
        bpy.ops.wm.tool_set_by_id(name="builtin.select_box")
        bpy.ops.ed.undo_push(message="YSE F9 hetero A baseline")
    return obj


def coordinate_key(co):
    return tuple(round(float(value), 5) for value in co)


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


def records_of(kind):
    return [
        record
        for record in operators._HISTORY_RECORDS.values()
        if record.session.object_name == OBJECT_NAME and record.session.tool_kind == kind
    ]


def busy():
    return bool(
        STATE["window"].modal_operators
        or operators._SESSIONS
        or operators._HISTORY_REPAIR_QUEUED
        or operators._HISTORY_REPAIR_BUSY
    )


def loopcut_route_ready():
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


def send_events(events, done, index=0, interval=0.15, done_delay=0.2):
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
            if busy():
                if time.monotonic() - started > 15.0:
                    raise RuntimeError("flow never settled")
                return 0.1
            bpy.app.timers.register(done, first_interval=0.25)
        except BaseException:
            fail()
        return None

    bpy.app.timers.register(poll, first_interval=0.1)


def instrument_f9():
    original = history_module._prepare_adjust_last_operation_repeat

    def logged():
        result = original()
        STATE["f9_prepare_result"] = result
        STATE["f9_kind"] = None
        operator = bpy.context.active_operator
        if operator is not None:
            STATE["f9_kind"] = operator.bl_idname
        print(
            f"YSE_EXTRUDE_F9_HETERO_A_DISCRIMINATOR result={result} active={STATE['f9_kind']}",
            flush=True,
        )
        return result

    history_module._prepare_adjust_last_operation_repeat = logged


def after_loopcut():
    try:
        STATE["phase"] = "after_loopcut"
        obj = test_object()
        bm = bmesh.from_edit_mesh(obj.data)
        print(f"YSE_EXTRUDE_F9_HETERO_A_AFTER_LOOPCUT={topology_counts(bm)}", flush=True)
        assert_x_symmetric(bm)
        assert_layers_removed(bm)
        loop_records = records_of("LOOP_CUT")
        assert loop_records, "loopcut did not commit a history record"
        STATE["loopcut_token"] = max(loop_records, key=lambda record: record.sequence).session.history_token
        STATE["after_loopcut"] = topology_counts(bm)
        bpy.app.timers.register(do_raw_extrude, first_interval=0.4)
    except BaseException:
        fail()
    return None


def do_raw_extrude():
    try:
        STATE["phase"] = "raw_extrude"
        obj = test_object()
        bm = bmesh.from_edit_mesh(obj.data)
        bpy.context.tool_settings.mesh_select_mode = (False, False, True)
        for face in bm.faces:
            face.select = False
        for edge in bm.edges:
            edge.select = False
        for vertex in bm.verts:
            vertex.select = False
        chosen = None
        for face in bm.faces:
            center = face.calc_center_median()
            if center.x > 0.5 and abs(center.z) < 1e-4:
                chosen = face
                break
        assert chosen is not None, "no +X floor face for raw extrude"
        chosen.select = True
        bm.select_flush_mode()
        bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
        with override():
            result = bpy.ops.mesh.extrude_region_move(
                "EXEC_DEFAULT",
                TRANSFORM_OT_translate={
                    "value": (0.0, 0.0, 0.35),
                    "orient_type": "GLOBAL",
                    "orient_matrix": ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
                    "orient_matrix_type": "GLOBAL",
                    "constraint_axis": (False, False, True),
                },
            )
        assert result == {"FINISHED"}, result
        print(f"YSE_EXTRUDE_F9_HETERO_A_RAW_EXTRUDE={result}", flush=True)
        wait_settled(after_raw)
    except BaseException:
        fail()
    return None


def after_raw():
    try:
        STATE["phase"] = "after_raw"
        obj = test_object()
        bm = bmesh.from_edit_mesh(obj.data)
        print(f"YSE_EXTRUDE_F9_HETERO_A_AFTER_RAW={topology_counts(bm)}", flush=True)
        assert topology_counts(bm) != STATE["after_loopcut"], "raw extrude did not change topology"
        assert not records_of("EXTRUDE_NORMAL"), "raw extrude must not create an extrude record"
        assert records_of("LOOP_CUT"), "loopcut record disappeared before F9"
        bpy.app.timers.register(do_f9, first_interval=0.4)
    except BaseException:
        fail()
    return None


def do_f9():
    try:
        STATE["phase"] = "f9"
        STATE["f9_prepare_result"] = None
        assert STATE["undo_redo_used"] == 0
        with override():
            result = bpy.ops.ed.undo_redo()
        STATE["undo_redo_used"] = 1
        print(f"YSE_EXTRUDE_F9_HETERO_A_UNDO_REDO={result}", flush=True)
        assert result == {"FINISHED"}, result
        wait_settled(after_f9)
    except BaseException:
        fail()
    return None


def after_f9():
    try:
        STATE["phase"] = "after_f9"
        prepare = STATE.get("f9_prepare_result")
        active = STATE.get("f9_kind")
        print(f"YSE_EXTRUDE_F9_HETERO_A_PREPARE={prepare} active={active}", flush=True)
        # Raw EXEC extrude does not replace last-operator. undo_redo therefore
        # F9s the still-active loopcut, which must stay LOOP_CUT and must not
        # spawn an extrude record.
        assert active == "MESH_OT_loopcut_slide", active
        assert prepare is True, f"loopcut F9 should prepare its own session (result={prepare})"
        assert not records_of("EXTRUDE_NORMAL"), "extrude record mixed into loopcut F9"
        loop_records = records_of("LOOP_CUT")
        assert loop_records, "loopcut record lost after undo_redo"
        assert all(record.session.tool_kind == "LOOP_CUT" for record in loop_records)
        obj = test_object()
        bm = bmesh.from_edit_mesh(obj.data)
        assert_layers_removed(bm)
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


def begin_loopcut():
    try:
        hover = window_coordinate((-0.7, 0.0, 0.0))
        confirm = window_coordinate((-0.7, 0.2, 0.0))
        events = [
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
        send_events(events, lambda: wait_settled(after_loopcut))
    except BaseException:
        fail()
    return None


def wait_route():
    try:
        if loopcut_route_ready():
            bpy.app.timers.register(begin_loopcut, first_interval=0.3)
            return None
        if time.monotonic() > STATE["deadline"]:
            raise RuntimeError("loopcut intercept not ready")
        return 0.05
    except BaseException:
        fail()
    return None


def start():
    try:
        addon.register()
        instrument_f9()
        addon.sync_persistent_keymap(True)
        window = bpy.context.window_manager.windows[0]
        area = next(item for item in window.screen.areas if item.type == "VIEW_3D")
        region = next(item for item in area.regions if item.type == "WINDOW")
        STATE.update(window=window, area=area, region=region)
        configure_view(area)
        build_mesh()
        STATE["deadline"] = time.monotonic() + 10.0
        bpy.app.timers.register(wait_route, first_interval=0.2)
    except BaseException:
        fail()
    return None


bpy.app.timers.register(failsafe, first_interval=FAILSAFE_S)
bpy.app.timers.register(start, first_interval=0.4)
