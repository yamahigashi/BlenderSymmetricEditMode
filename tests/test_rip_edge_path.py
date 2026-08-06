# SPDX-License-Identifier: GPL-3.0-or-later

"""GUI regression for the symmetric Rip (V) route: edge paths.

Run with Blender's real window/event loop::

    blender --factory-startup --enable-event-simulate \
        --python test_rip_edge_path.py

Serialized cases on an X-symmetric grid plane:

1. edgepath   V on an interior 2-edge path, drag, LMB: both sides rip and the
              result stays X-symmetric; one undo restores the baseline.
2. multiisland  two disjoint paths ripped by one V: both mirrored.
3. zeromove   V then immediate LMB: mirrored zero-width slit.
4. esc        V then ESC: native keeps the rip (R0 §5-5), mirror follows.
5. onplane    selection touches the mirror plane: no session, native-only
              result passes through unchanged (WARNING path).
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
from ydd_symmetric_edit import core, operators  # noqa: E402

MARKER_OK = "YSE_RIP_EDGE_PATH_TEST_OK"
MARKER_FAILED = "YSE_RIP_EDGE_PATH_TEST_FAILED"
NX, NY = 6, 4
PRECISION = 5
STATE = {}


def fail(message=""):
    if message:
        print(f"YSE_RIP_EDGE_PATH_ERROR={message}", flush=True)
    traceback.print_exc()
    print(MARKER_FAILED, flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(1)


def viewport_context():
    window = bpy.context.window_manager.windows[0]
    area = next(area for area in window.screen.areas if area.type == "VIEW_3D")
    region = next(region for region in area.regions if region.type == "WINDOW")
    return window, area, region


def configure_view(area):
    region_3d = area.spaces.active.region_3d
    region_3d.view_perspective = "ORTHO"
    region_3d.view_rotation = Quaternion((1.0, 0.0, 0.0, 0.0))
    region_3d.view_location = (0.0, 0.0, 0.0)
    region_3d.view_distance = 8.0
    region_3d.update()


def override():
    return bpy.context.temp_override(window=STATE["window"], area=STATE["area"], region=STATE["region"])


def window_coordinate(coordinate):
    region = STATE["region"]
    region_3d = STATE["area"].spaces.active.region_3d
    local = view3d_utils.location_3d_to_region_2d(region, region_3d, Vector(coordinate))
    if local is None:
        raise RuntimeError(f"could not project {coordinate}")
    return int(round(region.x + local.x)), int(round(region.y + local.y))


def build_mesh(name):
    for old in tuple(bpy.data.objects):
        bpy.data.objects.remove(old, do_unlink=True)
    for old_mesh in tuple(bpy.data.meshes):
        if old_mesh.users == 0:
            bpy.data.meshes.remove(old_mesh)
    mesh = bpy.data.meshes.new(f"YSE_RipMesh_{name}")
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
    obj = bpy.data.objects.new(f"YSE_RipObject_{name}", mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.use_mesh_mirror_x = True
    obj.use_mesh_mirror_y = False
    obj.use_mesh_mirror_z = False
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    with override():
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.context.tool_settings.mesh_select_mode = (True, False, False)
        bpy.ops.ed.undo_push(message=f"YSE rip baseline {name}")
    STATE["object"] = obj
    return obj


def grid_vert(bm, i, j):
    for vertex in bm.verts:
        if (
            abs(vertex.co.x - (i - NX / 2)) < 1e-4
            and abs(vertex.co.y - (j - NY / 2)) < 1e-4
            and abs(vertex.co.z) < 1e-4
        ):
            return vertex
    raise AssertionError(f"grid vert {i},{j} not found")


def select_verts(bm, ij_list):
    for face in bm.faces:
        face.select = False
    for edge in bm.edges:
        edge.select = False
    for vertex in bm.verts:
        vertex.select = False
    bm.select_history.clear()
    for ij in ij_list:
        grid_vert(bm, *ij).select = True
    bm.select_flush_mode()


def coordinate_key(co):
    return tuple(round(float(value), PRECISION) for value in co)


def vertex_multiset(bm):
    return Counter(coordinate_key(vertex.co) for vertex in bm.verts)


def mirrored_multiset(bm):
    return Counter(coordinate_key((-vertex.co.x, vertex.co.y, vertex.co.z)) for vertex in bm.verts)


def assert_x_symmetric(bm):
    assert vertex_multiset(bm) == mirrored_multiset(bm), "vertex coordinates are not X-symmetric"


def assert_layers_removed(bm):
    assert bm.verts.layers.int.get(core.VERT_RIP_ID_LAYER) is None, "rip vertex layer leaked"
    assert bm.edges.layers.int.get(core.EDGE_ORIGINAL_LAYER) is None, "edge layer leaked"
    assert bm.faces.layers.int.get(core.FACE_ID_LAYER) is None, "face layer leaked"


def topology_counts(bm):
    return len(bm.verts), len(bm.edges), len(bm.faces)


def send_events(events, done, index=0):
    def step():
        try:
            if index < len(events):
                STATE["window"].event_simulate(**events[index])
                send_events(events, done, index + 1)
            else:
                bpy.app.timers.register(done, first_interval=0.2)
        except BaseException:
            fail()
        return None

    bpy.app.timers.register(step, first_interval=0.09)


def wait_settled(done, started=None):
    started = time.monotonic() if started is None else started

    def poll():
        try:
            busy = bool(STATE["window"].modal_operators) or bool(operators._SESSIONS)
            if busy:
                if time.monotonic() - started > 12.0:
                    raise RuntimeError(
                        f"rip flow never settled; modal={[op.bl_idname for op in STATE['window'].modal_operators]} "
                        f"sessions={list(operators._SESSIONS)}"
                    )
                return 0.1
            bpy.app.timers.register(done, first_interval=0.25)
        except BaseException:
            fail()
        return None

    bpy.app.timers.register(poll, first_interval=0.1)


def rip_events(cursor_ij, drag=(40, 0), confirm="LMB"):
    x, y = window_coordinate((cursor_ij[0] - NX / 2, cursor_ij[1] - NY / 2, 0.0))
    events = [
        {"type": "MOUSEMOVE", "value": "NOTHING", "x": x, "y": y},
        {"type": "V", "value": "PRESS", "x": x, "y": y},
        {"type": "V", "value": "RELEASE", "x": x, "y": y},
    ]
    tx, ty = x + drag[0], y + drag[1]
    if drag != (0, 0):
        events.append({"type": "MOUSEMOVE", "value": "NOTHING", "x": tx, "y": ty})
    if confirm == "LMB":
        events.append({"type": "LEFTMOUSE", "value": "PRESS", "x": tx, "y": ty})
        events.append({"type": "LEFTMOUSE", "value": "RELEASE", "x": tx, "y": ty})
    else:
        events.append({"type": "ESC", "value": "PRESS", "x": tx, "y": ty})
        events.append({"type": "ESC", "value": "RELEASE", "x": tx, "y": ty})
    return events


def run_case(name, select_ij, cursor_ij, verify, *, drag=(40, 0), confirm="LMB"):
    def start(next_case):
        try:
            print(f"YSE_RIP_CASE={name}", flush=True)
            obj = build_mesh(name)
            bm = bmesh.from_edit_mesh(obj.data)
            STATE["baseline"] = topology_counts(bm)
            select_verts(bm, select_ij)
            bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)

            def settled():
                try:
                    bm2 = bmesh.from_edit_mesh(STATE["object"].data)
                    verify(bm2)
                    next_case()
                except BaseException:
                    fail()

            send_events(rip_events(cursor_ij, drag=drag, confirm=confirm), lambda: wait_settled(settled))
        except BaseException:
            fail()

    return start


def verify_edgepath(bm):
    dv = len(bm.verts) - STATE["baseline"][0]
    de = len(bm.edges) - STATE["baseline"][1]
    assert dv == 6, f"expected 6 new vertices (3 source + 3 mirror), got {dv}"
    assert de == 8, f"expected 8 new edges (4 source + 4 mirror), got {de}"
    assert_x_symmetric(bm)
    assert_layers_removed(bm)


def verify_and_undo_edgepath(bm):
    verify_edgepath(bm)
    with override():
        undo_result = bpy.ops.ed.undo()
    assert undo_result == {"FINISHED"}, undo_result
    obj = STATE["object"]
    bm2 = bmesh.from_edit_mesh(obj.data)
    counts = topology_counts(bm2)
    assert counts == STATE["baseline"], f"one undo did not restore baseline: {counts} != {STATE['baseline']}"
    assert_x_symmetric(bm2)
    assert_layers_removed(bm2)


def verify_multiisland(bm):
    dv = len(bm.verts) - STATE["baseline"][0]
    de = len(bm.edges) - STATE["baseline"][1]
    # Native Rip only rips the connected island under the path heuristic; the
    # disjoint extra vertex stays untouched (measured, 4.2/5.2).  The mirror
    # must reproduce exactly what native ripped: 2 verts / 3 seam edges.
    assert dv == 4, f"expected 4 new vertices (2 source + 2 mirror), got {dv}"
    assert de == 6, f"expected 6 new edges (3 source + 3 mirror), got {de}"
    assert_x_symmetric(bm)
    assert_layers_removed(bm)


def verify_zero_width(bm):
    dv = len(bm.verts) - STATE["baseline"][0]
    assert dv == 2, f"expected 2 new vertices (1 source + 1 mirror), got {dv}"
    assert vertex_multiset(bm)[coordinate_key((1.0, 0.0, 0.0))] == 2, "zero-width slit vertex not duplicated"
    assert vertex_multiset(bm)[coordinate_key((-1.0, 0.0, 0.0))] == 2, "mirror slit vertex not duplicated"
    assert_x_symmetric(bm)
    assert_layers_removed(bm)


def verify_onplane_passthrough(bm):
    # Native rip ran unmirrored: source side ripped, mirror side untouched,
    # so the vertex multiset is NOT X-symmetric and no layer remains.
    dv = len(bm.verts) - STATE["baseline"][0]
    assert dv == 2, f"expected native-only rip to add 2 vertices, got {dv}"
    assert vertex_multiset(bm) != mirrored_multiset(bm), "on-plane rip should not be mirrored"
    assert_layers_removed(bm)


def run_all(cases, index=0):
    if index >= len(cases):
        print(MARKER_OK, flush=True)
        sys.stdout.flush()
        addon.unregister()
        bpy.ops.wm.quit_blender()
        return
    cases[index](lambda: run_all(cases, index + 1))


def start_test():
    try:
        addon.register()
        addon.sync_persistent_keymap(True)
        window, area, region = viewport_context()
        configure_view(area)
        STATE.update(window=window, area=area, region=region)
        cases = [
            run_case("edgepath", [(4, 1), (4, 2), (4, 3)], (3.6, 2), verify_and_undo_edgepath),
            run_case("multiisland", [(5, 1), (4, 2), (4, 3)], (3.6, 2), verify_multiisland),
            run_case("zeromove", [(4, 2)], (3.6, 2), verify_zero_width, drag=(0, 0)),
            run_case("esc", [(4, 2)], (3.6, 2), verify_zero_width, confirm="ESC"),
            run_case(
                "onplane",
                [(3, 2), (4, 2)],
                (3.6, 2),
                verify_onplane_passthrough,
            ),
        ]
        run_all(cases)
    except BaseException:
        fail()
    return None


bpy.app.timers.register(start_test, first_interval=0.3)
