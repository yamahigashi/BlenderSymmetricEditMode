# SPDX-License-Identifier: GPL-3.0-or-later

"""End-to-end test for the persistent plain-K mode.

This test deliberately starts both Knife sessions through a simulated ``K``
event.  Calling the add-on operator directly would not exercise the persistent
keymap that this regression test is intended to cover.
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

# Prevent Blender's startup splash from consuming the first simulated K event.
bpy.context.preferences.view.show_splash = False
bpy.context.preferences.view.smooth_view = 0
bpy.context.preferences.view.use_save_prompt = False
bpy.context.preferences.filepaths.use_auto_save_temporary_files = False

PACKAGE_PARENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_PARENT))

import ydd_symmetric_edit as addon  # noqa: E402
from ydd_symmetric_edit import core, keymaps, operators  # noqa: E402

CUT_X_COORDINATES = (-1.65, -1.30)
STATE = {
    "cut_index": 0,
    "deadline": 0.0,
    "events": [],
}


def modal_identifiers():
    try:
        return [operator.bl_rna.identifier for operator in STATE["window"].modal_operators]
    except Exception:
        return []


def fail(message=""):
    if message:
        print(f"PERSISTENT_MODE_ERROR={message}", flush=True)
    traceback.print_exc()
    print(f"PERSISTENT_MODE_PHASE={STATE.get('phase')}", flush=True)
    print(f"PERSISTENT_MODE_MODAL_IDS={modal_identifiers()}", flush=True)
    print(f"PERSISTENT_MODE_SESSIONS={list(operators._SESSIONS)}", flush=True)
    print(
        f"PERSISTENT_MODE_KMI_ACTIVE={[item.active for _keymap, item in keymaps._REGISTERED_ITEMS]}",
        flush=True,
    )
    print("YSE_PERSISTENT_MODE_TEST_FAILED", flush=True)
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
    region_3d.view_distance = 5.0
    region_3d.update()


def make_mesh():
    for old in tuple(bpy.data.objects):
        bpy.data.objects.remove(old, do_unlink=True)

    mesh = bpy.data.meshes.new("YSE_PersistentMesh")
    mesh.from_pydata(
        [
            (-2.0, -1.0, 0.0),
            (-1.0, -1.0, 0.0),
            (-1.0, 1.0, 0.0),
            (-2.0, 1.0, 0.0),
            (1.0, -1.0, 0.0),
            (2.0, -1.0, 0.0),
            (2.0, 1.0, 0.0),
            (1.0, 1.0, 0.0),
        ],
        [],
        [(0, 1, 2, 3), (4, 5, 6, 7)],
    )
    mesh.update()
    obj = bpy.data.objects.new("YSE_PersistentObject", mesh)
    bpy.context.scene.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    obj.use_mesh_mirror_x = True
    obj.use_mesh_mirror_y = False
    obj.use_mesh_mirror_z = False
    return obj


def window_coordinate(region, region_3d, coordinate):
    local = view3d_utils.location_3d_to_region_2d(region, region_3d, Vector(coordinate))
    if local is None:
        raise RuntimeError(f"Could not project test point {coordinate}")
    return int(round(region.x + local.x)), int(round(region.y + local.y))


def internal_vertical_cut_coordinates(bm):
    coordinates = []
    for edge in bm.edges:
        x0, x1 = (vertex.co.x for vertex in edge.verts)
        if abs(x0 - x1) > 1.0e-6:
            continue
        x = (x0 + x1) * 0.5
        if not 1.0 + 1.0e-6 < abs(x) < 2.0 - 1.0e-6:
            continue
        ys = sorted(vertex.co.y for vertex in edge.verts)
        if abs(ys[0] + 1.0) <= 1.0e-6 and abs(ys[1] - 1.0) <= 1.0e-6:
            coordinates.append(x)
    return sorted(coordinates)


def assert_completed_cuts(count):
    bm = bmesh.from_edit_mesh(STATE["object"].data)
    actual = internal_vertical_cut_coordinates(bm)
    expected = sorted(coordinate for source in CUT_X_COORDINATES[:count] for coordinate in (source, -source))
    assert len(actual) == len(expected), (actual, expected)
    # The source Knife location is screen-space and therefore lands on the
    # nearest pixel.  Its generated counterpart, however, must be exact.
    assert all(
        abs(actual_coordinate - expected_coordinate) <= 5.0e-3
        for actual_coordinate, expected_coordinate in zip(actual, expected, strict=True)
    ), (actual, expected)
    negative = [coordinate for coordinate in actual if coordinate < 0.0]
    positive = [coordinate for coordinate in actual if coordinate > 0.0]
    assert len(negative) == len(positive) == count, actual
    assert all(abs(left + right) <= 1.0e-7 for left, right in zip(negative, reversed(positive), strict=True)), actual


def assert_no_temporary_layers(bm):
    assert bm.edges.layers.int.get(core.EDGE_ORIGINAL_LAYER) is None
    assert bm.faces.layers.int.get(core.FACE_ID_LAYER) is None
    assert bm.verts.layers.int.get(core.VERT_SELECTION_LAYER) is None
    assert bm.edges.layers.int.get(core.EDGE_SELECTION_LAYER) is None
    assert bm.faces.layers.int.get(core.FACE_SELECTION_LAYER) is None
    assert bm.verts.layers.int.get(core.VERT_HIDDEN_LAYER) is None
    assert bm.edges.layers.int.get(core.EDGE_HIDDEN_LAYER) is None
    assert bm.verts.layers.int.get(core.VERT_BACKUP_ID_LAYER) is None


def finish_test():
    try:
        STATE["phase"] = "final verification"
        bm = bmesh.from_edit_mesh(STATE["object"].data)
        assert len(bm.verts) == 16, len(bm.verts)
        assert len(bm.edges) == 20, len(bm.edges)
        assert len(bm.faces) == 6, len(bm.faces)
        assert_completed_cuts(2)
        assert_no_temporary_layers(bm)
        assert not operators._SESSIONS, operators._SESSIONS
        assert not any(
            data.name.startswith(("YSE_TemporaryCutter", "YSE_TemporaryBackup"))
            for data in (*bpy.data.objects, *bpy.data.meshes)
        )

        items = keymaps._REGISTERED_ITEMS
        assert items and all(item.active for _keymap, item in items)

        print("YSE_PERSISTENT_MODE_TEST_OK", flush=True)
        addon.unregister()
        bpy.ops.wm.quit_blender()
    except BaseException:
        fail()
    return None


def wait_for_cut_finish():
    try:
        active = any(
            "NATIVE_SYMMETRIC_KNIFE_SESSION" in identifier.upper() or "KNIFE_TOOL" in identifier.upper()
            for identifier in modal_identifiers()
        )
        if operators._SESSIONS or active:
            if time.monotonic() > STATE["deadline"]:
                fail("Timed out waiting for the Knife session to finish")
            return 0.05

        completed = STATE["cut_index"] + 1
        assert_completed_cuts(completed)
        items = keymaps._REGISTERED_ITEMS
        assert items and all(item.active for _keymap, item in items)

        STATE["cut_index"] = completed
        if completed < len(CUT_X_COORDINATES):
            bpy.app.timers.register(begin_cut, first_interval=0.2)
        else:
            bpy.app.timers.register(finish_test, first_interval=0.2)
    except BaseException:
        fail()
    return None


def send_next_knife_event():
    try:
        if STATE["events"]:
            STATE["window"].event_simulate(**STATE["events"].pop(0))
            return 0.08

        STATE["phase"] = f"waiting for cut {STATE['cut_index'] + 1} to finish"
        STATE["deadline"] = time.monotonic() + 8.0
        bpy.app.timers.register(wait_for_cut_finish, first_interval=0.25)
    except BaseException:
        fail()
    return None


def wait_for_session_start():
    try:
        if operators._SESSIONS:
            STATE["phase"] = f"drawing cut {STATE['cut_index'] + 1}"
            start, end = STATE["stroke"]
            STATE["events"] = [
                {"type": "MOUSEMOVE", "value": "NOTHING", "x": start[0], "y": start[1]},
                {"type": "LEFTMOUSE", "value": "PRESS", "x": start[0], "y": start[1]},
                {"type": "LEFTMOUSE", "value": "RELEASE", "x": start[0], "y": start[1]},
                {"type": "MOUSEMOVE", "value": "NOTHING", "x": end[0], "y": end[1]},
                {"type": "LEFTMOUSE", "value": "PRESS", "x": end[0], "y": end[1]},
                {"type": "LEFTMOUSE", "value": "RELEASE", "x": end[0], "y": end[1]},
                {"type": "RET", "value": "PRESS", "x": end[0], "y": end[1]},
                {"type": "RET", "value": "RELEASE", "x": end[0], "y": end[1]},
            ]
            bpy.app.timers.register(send_next_knife_event, first_interval=0.1)
            return None

        if time.monotonic() > STATE["deadline"]:
            fail("The simulated K event did not start ydd Symmetric Edit")
        return 0.05
    except BaseException:
        fail()
    return None


def send_k_event():
    try:
        STATE["phase"] = f"sending K for cut {STATE['cut_index'] + 1}"
        start = STATE["stroke"][0]
        STATE["window"].event_simulate(type="K", value="PRESS", x=start[0], y=start[1])
        STATE["window"].event_simulate(type="K", value="RELEASE", x=start[0], y=start[1])
        STATE["deadline"] = time.monotonic() + 4.0
        bpy.app.timers.register(wait_for_session_start, first_interval=0.1)
    except BaseException:
        fail()
    return None


def begin_cut():
    try:
        cut_x = CUT_X_COORDINATES[STATE["cut_index"]]
        region = STATE["region"]
        region_3d = STATE["area"].spaces.active.region_3d
        start = window_coordinate(region, region_3d, (cut_x, -1.0, 0.0))
        end = window_coordinate(region, region_3d, (cut_x, 1.0, 0.0))
        STATE["stroke"] = (start, end)
        STATE["phase"] = f"positioning for cut {STATE['cut_index'] + 1}"
        STATE["window"].event_simulate(type="MOUSEMOVE", value="NOTHING", x=start[0], y=start[1])
        bpy.app.timers.register(send_k_event, first_interval=0.15)
    except BaseException:
        fail()
    return None


def start_test():
    try:
        STATE["phase"] = "setup"
        addon.register()
        addon.sync_persistent_keymap(True)

        items = keymaps._REGISTERED_ITEMS
        assert items and all(item.active for _keymap, item in items)

        window, area, region = viewport_context()
        configure_view(area)
        obj = make_mesh()
        with bpy.context.temp_override(window=window, area=area, region=region):
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.context.tool_settings.mesh_select_mode = (True, False, False)

        assert core.enabled_mesh_symmetry_axes(obj) == (("X", 0),)
        STATE.update(window=window, area=area, region=region, object=obj)
        bpy.app.timers.register(begin_cut, first_interval=0.25)
    except BaseException:
        fail()
    return None


bpy.app.timers.register(start_test, first_interval=0.25)
