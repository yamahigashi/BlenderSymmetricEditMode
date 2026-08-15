# SPDX-License-Identifier: GPL-3.0-or-later

"""GUI representative: history-limit eviction leftover hetero token → cleanup-only.

Plan: .agents/doc/f9_extrude_plan_2026-08-15.md v3.1 §4-2 row 15.

Run::

    cmd.exe /c "tmp\\run_menu_reg.bat 42 test_extrude_f9_hetero_c.py"
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

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
from ydd_symmetric_edit import layer_names, operators, session_state  # noqa: E402
from ydd_symmetric_edit._types import HistoryRecord  # noqa: E402

OBJECT_NAME = "YSE_F9HeteroC"
MESH_NAME = "YSE_F9HeteroCMesh"
MARKER_OK = "YSE_EXTRUDE_F9_HETERO_C_OK"
MARKER_FAILED = "YSE_EXTRUDE_F9_HETERO_C_FAILED"
LEFTOVER_TOKEN = 777001
NX, NY = 6, 4
EXPECTED_NET = (8, 16, 8)
FAILSAFE_S = 120.0
STATE: dict = {"phase": "startup", "finished": False, "undo_redo_used": 0}


def fail(message=""):
    if message:
        print(f"YSE_EXTRUDE_F9_HETERO_C_ERROR={message}", flush=True)
    traceback.print_exc()
    print(f"YSE_EXTRUDE_F9_HETERO_C_PHASE={STATE.get('phase')}", flush=True)
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


def grid_xy(i, j):
    return (i - NX / 2, j - NY / 2)


def build_mesh():
    for old in tuple(bpy.data.objects):
        bpy.data.objects.remove(old, do_unlink=True)
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
        bpy.ops.ed.undo_push(message="YSE F9 hetero C baseline")
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


def live_net(bm):
    base = STATE["baseline"]
    return (
        len(bm.verts) - base[0],
        len(bm.edges) - base[1],
        len(bm.faces) - base[2],
    )


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
            if busy():
                if time.monotonic() - started > 15.0:
                    raise RuntimeError("flow never settled")
                return 0.1
            bpy.app.timers.register(done, first_interval=0.25)
        except BaseException:
            fail()
        return None

    bpy.app.timers.register(poll, first_interval=0.1)


def snapshot_undo_post_layers():
    obj = test_object()
    bm = bmesh.from_edit_mesh(obj.data)
    token_layer = bm.faces.layers.int.get(layer_names.HISTORY_TOKEN_LAYER)
    tokens = set()
    if token_layer is not None:
        tokens = {int(face[token_layer]) for face in bm.faces if int(face[token_layer])}
    STATE["undo_post_tokens"] = tokens
    print(f"YSE_EXTRUDE_F9_HETERO_C_UNDO_POST_TOKENS={sorted(tokens)}", flush=True)


def instrument_f9():
    original = history_module._prepare_adjust_last_operation_repeat

    def logged():
        snapshot_undo_post_layers()
        result = original()
        STATE["f9_prepare_result"] = result
        print(f"YSE_EXTRUDE_F9_HETERO_C_DISCRIMINATOR result={result}", flush=True)
        return result

    history_module._prepare_adjust_last_operation_repeat = logged


def paint_leftover_token():
    obj = test_object()
    bm = bmesh.from_edit_mesh(obj.data)
    old = bm.faces.layers.int.get(layer_names.HISTORY_TOKEN_LAYER)
    if old is not None:
        bm.faces.layers.int.remove(old)
    layer = bm.faces.layers.int.new(layer_names.HISTORY_TOKEN_LAYER)
    for face in bm.faces:
        face[layer] = LEFTOVER_TOKEN
    bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
    with override():
        push = bpy.ops.ed.undo_push(message="leftover-hetero")
    assert push == {"FINISHED"}, push


def evict_extrude_record():
    records = records_of("EXTRUDE_NORMAL")
    assert records, "no extrude record to evict"
    evicted = max(records, key=lambda record: record.sequence)
    STATE["evicted_token"] = evicted.session.history_token
    window = STATE["window"]
    configured_limit = int(getattr(bpy.context.preferences.edit, "undo_steps", 32)) + 4
    history_limit = min(session_state._MAX_HISTORY_RECORDS, max(8, configured_limit))
    filler = 0
    while evicted.session.history_token in session_state._HISTORY_RECORDS:
        filler += 1
        token = 800000 + filler
        fake = SimpleNamespace(
            tool_kind="LOOP_CUT",
            window_pointer=window.as_pointer(),
            object_name=OBJECT_NAME,
            mesh_name=test_object().data.name,
            native_operator_pointer=filler,
            history_token=token,
            extrude=None,
        )
        session_state._HISTORY_RECORDS[token] = HistoryRecord(session=fake, status="COMMITTED", sequence=1000 + filler)
        session_state._HISTORY_RECORDS.move_to_end(token)
        while len(session_state._HISTORY_RECORDS) > history_limit:
            session_state._HISTORY_RECORDS.popitem(last=False)
        if filler > history_limit + 4:
            break
    assert evicted.session.history_token not in session_state._HISTORY_RECORDS, "failed to evict extrude record"
    print(
        f"YSE_EXTRUDE_F9_HETERO_C_EVICTED token={STATE['evicted_token']} leftover={LEFTOVER_TOKEN} "
        f"limit={history_limit} records={len(session_state._HISTORY_RECORDS)}",
        flush=True,
    )


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
        print(f"YSE_EXTRUDE_F9_HETERO_C_NET_AFTER_EXTRUDE={got}", flush=True)
        assert got == EXPECTED_NET, f"net {got} != {EXPECTED_NET}"
        assert_x_symmetric(bm)
        assert_layers_removed(bm)
        evict_extrude_record()
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
        print(f"YSE_EXTRUDE_F9_HETERO_C_UNDO_REDO={result}", flush=True)
        assert result == {"FINISHED"}, result
        wait_settled(after_f9)
    except BaseException:
        fail()
    return None


def after_f9():
    try:
        STATE["phase"] = "after_f9"
        prepare = STATE.get("f9_prepare_result")
        print(f"YSE_EXTRUDE_F9_HETERO_C_PREPARE={prepare}", flush=True)
        assert prepare is False, f"evicted leftover must be cleanup-only (result={prepare})"
        assert LEFTOVER_TOKEN in STATE.get("undo_post_tokens", set()), STATE.get("undo_post_tokens")
        obj = test_object()
        bm = bmesh.from_edit_mesh(obj.data)
        leftover = bm.faces.layers.int.get(layer_names.HISTORY_TOKEN_LAYER)
        tokens = set()
        if leftover is not None:
            tokens = {int(face[leftover]) for face in bm.faces if int(face[leftover])}
        print(f"YSE_EXTRUDE_F9_HETERO_C_AFTER_CLEANUP_TOKENS={sorted(tokens)}", flush=True)
        assert LEFTOVER_TOKEN not in tokens, "leftover evicted token was not cleaned"
        assert_layers_removed(bm)
        assert not records_of("EXTRUDE_NORMAL"), "evicted extrude record must stay gone"
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
        instrument_f9()
        addon.sync_persistent_keymap(True)
        window = bpy.context.window_manager.windows[0]
        area = next(item for item in window.screen.areas if item.type == "VIEW_3D")
        region = next(item for item in area.regions if item.type == "WINDOW")
        STATE.update(window=window, area=area, region=region)
        configure_view(area)
        bpy.context.preferences.edit.undo_steps = 8
        build_mesh()
        obj = test_object()
        bm = bmesh.from_edit_mesh(obj.data)
        select_single_face(bm)
        bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
        with override():
            push = bpy.ops.ed.undo_push(message="YSE F9 hetero C prepared")
        assert push == {"FINISHED"}, push
        paint_leftover_token()
        STATE["baseline"] = topology_counts(bmesh.from_edit_mesh(obj.data))
        print(f"YSE_EXTRUDE_F9_HETERO_C_BASELINE={STATE['baseline']}", flush=True)
        events = extrude_events((1.5, -0.5, 0.0), drag=(0, 80))
        send_events(events, lambda: wait_settled(after_extrude))
    except BaseException:
        fail()
    return None


bpy.app.timers.register(failsafe, first_interval=FAILSAFE_S)
bpy.app.timers.register(start, first_interval=0.4)
