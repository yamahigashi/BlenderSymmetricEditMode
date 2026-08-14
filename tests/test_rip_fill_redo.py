# SPDX-License-Identifier: GPL-3.0-or-later

"""GUI regression for the symmetric Rip route: Rip Fill (Alt+V) and undo/redo.

Run with Blender's real window/event loop::

    blender --factory-startup --enable-event-simulate \
        --python test_rip_fill_redo.py

Serialized cases on an X-symmetric grid plane:

1. fill       Alt+V on an interior 2-edge path: bridge faces mirrored with
              face/loop CustomData (UV, smooth) copied from the source fill.
2. undoredo   plain V rip, then undo to baseline and redo back to the full
              mirrored result (repair handlers stay quiet on clean states).
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
from ydd_symmetric_edit import layer_names, operators  # noqa: E402

MARKER_OK = "YSE_RIP_FILL_REDO_TEST_OK"
MARKER_FAILED = "YSE_RIP_FILL_REDO_TEST_FAILED"
NX, NY = 6, 4
PRECISION = 5
TEST_VID_LAYER = "yse_test_vid"
STATE = {}


def fail(message=""):
    if message:
        print(f"YSE_RIP_FILL_REDO_ERROR={message}", flush=True)
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


def build_mesh(name, *, with_uv=False):
    for old in tuple(bpy.data.objects):
        bpy.data.objects.remove(old, do_unlink=True)
    for old_mesh in tuple(bpy.data.meshes):
        if old_mesh.users == 0:
            bpy.data.meshes.remove(old_mesh)
    mesh = bpy.data.meshes.new(f"YSE_RipFill_{name}")
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
    obj = bpy.data.objects.new(f"YSE_RipFillObj_{name}", mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.use_mesh_mirror_x = True
    obj.use_mesh_mirror_y = False
    obj.use_mesh_mirror_z = False
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    with override():
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.context.tool_settings.mesh_select_mode = (True, False, False)

    bm = bmesh.from_edit_mesh(obj.data)
    bm.verts.layers.int.new(TEST_VID_LAYER)
    if with_uv:
        bm.loops.layers.uv.new("UVMap")
    bm = bmesh.from_edit_mesh(obj.data)
    vid = bm.verts.layers.int.get(TEST_VID_LAYER)
    for index, vertex in enumerate(bm.verts, start=1):
        vertex[vid] = index
    if with_uv:
        uv = bm.loops.layers.uv.get("UVMap")
        for face in bm.faces:
            face.smooth = True
            for loop in face.loops:
                loop[uv].uv = (loop.vert.co.x * 0.1 + 0.5, loop.vert.co.y * 0.1 + 0.5)
    bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
    with override():
        bpy.ops.ed.undo_push(message=f"YSE rip fill baseline {name}")
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
    assert bm.verts.layers.int.get(layer_names.VERT_SESSION_ID_LAYER) is None, "rip vertex layer leaked"
    assert bm.edges.layers.int.get(layer_names.EDGE_ORIGINAL_LAYER) is None, "edge layer leaked"
    assert bm.faces.layers.int.get(layer_names.FACE_ID_LAYER) is None, "face layer leaked"


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
                    raise RuntimeError("rip flow never settled")
                return 0.1
            bpy.app.timers.register(done, first_interval=0.25)
        except BaseException:
            fail()
        return None

    bpy.app.timers.register(poll, first_interval=0.1)


def rip_events(cursor_ij, *, alt=False, drag=(40, 0)):
    x, y = window_coordinate((cursor_ij[0] - NX / 2, cursor_ij[1] - NY / 2, 0.0))
    events = [{"type": "MOUSEMOVE", "value": "NOTHING", "x": x, "y": y}]
    if alt:
        events.append({"type": "LEFT_ALT", "value": "PRESS", "x": x, "y": y})
        events.append({"type": "V", "value": "PRESS", "x": x, "y": y, "alt": True})
        events.append({"type": "V", "value": "RELEASE", "x": x, "y": y, "alt": True})
        events.append({"type": "LEFT_ALT", "value": "RELEASE", "x": x, "y": y})
    else:
        events.append({"type": "V", "value": "PRESS", "x": x, "y": y})
        events.append({"type": "V", "value": "RELEASE", "x": x, "y": y})
    tx, ty = x + drag[0], y + drag[1]
    events.append({"type": "MOUSEMOVE", "value": "NOTHING", "x": tx, "y": ty})
    events.append({"type": "LEFTMOUSE", "value": "PRESS", "x": tx, "y": ty})
    events.append({"type": "LEFTMOUSE", "value": "RELEASE", "x": tx, "y": ty})
    return events


def fill_faces_by_side(bm):
    """Fill faces (repeated test-vid corners) split by mirror side."""

    vid = bm.verts.layers.int.get(TEST_VID_LAYER)
    assert vid is not None, "test vid layer disappeared"
    positive, negative = [], []
    for face in bm.faces:
        ids = [int(loop.vert[vid]) for loop in face.loops]
        if len(ids) != len(set(ids)):
            center_x = sum(loop.vert.co.x for loop in face.loops) / len(face.loops)
            (positive if center_x > 0 else negative).append(face)
    return positive, negative


def verify_fill(bm):
    dv = len(bm.verts) - STATE["baseline"][0]
    df = len(bm.faces) - STATE["baseline"][2]
    assert dv == 6, f"expected 6 new vertices, got {dv}"
    assert df == 8, f"expected 8 new faces (4 source fill + 4 mirror fill), got {df}"
    assert_x_symmetric(bm)
    assert_layers_removed(bm)

    source_fill, mirror_fill = fill_faces_by_side(bm)
    assert len(source_fill) == 4, f"expected 4 source fill faces, got {len(source_fill)}"
    assert len(mirror_fill) == 4, f"expected 4 mirrored fill faces, got {len(mirror_fill)}"

    uv = bm.loops.layers.uv.get("UVMap")
    assert uv is not None, "UV layer disappeared"

    def face_key(face):
        return tuple(
            sorted(coordinate_key((abs(loop.vert.co.x), loop.vert.co.y, loop.vert.co.z)) for loop in face.loops)
        )

    def uv_multiset(face):
        return Counter(tuple(round(float(value), 5) for value in loop[uv].uv) for loop in face.loops)

    mirror_by_key = {face_key(face): face for face in mirror_fill}
    for face in source_fill:
        counterpart = mirror_by_key.get(face_key(face))
        assert counterpart is not None, "a source fill face has no mirrored counterpart"
        assert counterpart.smooth == face.smooth, "smooth flag was not copied to the mirrored fill"
        assert counterpart.material_index == face.material_index, "material index was not copied"
        assert uv_multiset(counterpart) == uv_multiset(face), "loop UVs were not copied to the mirrored fill"


def verify_undoredo(bm):
    dv = len(bm.verts) - STATE["baseline"][0]
    assert dv == 4, f"expected 4 new vertices, got {dv}"
    assert_x_symmetric(bm)
    assert_layers_removed(bm)

    with override():
        undo_result = bpy.ops.ed.undo()
    assert undo_result == {"FINISHED"}, undo_result
    obj = STATE["object"]
    bm2 = bmesh.from_edit_mesh(obj.data)
    assert topology_counts(bm2) == STATE["baseline"], "one undo did not restore the baseline"
    assert_layers_removed(bm2)

    with override():
        redo_result = bpy.ops.ed.redo()
    assert redo_result == {"FINISHED"}, redo_result

    def after_redo():
        try:
            bm3 = bmesh.from_edit_mesh(STATE["object"].data)
            dv3 = len(bm3.verts) - STATE["baseline"][0]
            assert dv3 == 4, f"redo did not restore the mirrored rip, dv={dv3}"
            assert_x_symmetric(bm3)
            assert_layers_removed(bm3)
            STATE["next_case"]()
        except BaseException:
            fail()
        return None

    # Redo may queue the history repair timer; let it settle before verifying.
    bpy.app.timers.register(after_redo, first_interval=0.6)
    return "ASYNC"


def run_case(name, select_ij, cursor_ij, verify, *, alt=False, with_uv=False):
    def start(next_case):
        try:
            print(f"YSE_RIP_FILL_CASE={name}", flush=True)
            obj = build_mesh(name, with_uv=with_uv)
            bm = bmesh.from_edit_mesh(obj.data)
            STATE["baseline"] = topology_counts(bm)
            select_verts(bm, select_ij)
            bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)

            def settled():
                try:
                    STATE["next_case"] = next_case
                    bm2 = bmesh.from_edit_mesh(STATE["object"].data)
                    if verify(bm2) != "ASYNC":
                        next_case()
                except BaseException:
                    fail()

            send_events(rip_events(cursor_ij, alt=alt), lambda: wait_settled(settled))
        except BaseException:
            fail()

    return start


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
            run_case("fill", [(4, 1), (4, 2), (4, 3)], (3.6, 2), verify_fill, alt=True, with_uv=True),
            run_case("undoredo", [(4, 1), (4, 2)], (3.6, 2), verify_undoredo),
        ]
        run_all(cases)
    except BaseException:
        fail()
    return None


bpy.app.timers.register(start_test, first_interval=0.3)
