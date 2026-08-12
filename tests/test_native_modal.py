# SPDX-License-Identifier: GPL-3.0-or-later

"""End-to-end test: drive Blender's real modal Knife through simulated events."""

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

# Prevent Blender's startup UI from consuming simulated keyboard events.
bpy.context.preferences.view.show_splash = False
bpy.context.preferences.view.smooth_view = 0
bpy.context.preferences.view.use_save_prompt = False
bpy.context.preferences.filepaths.use_auto_save_temporary_files = False

PACKAGE_PARENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_PARENT))

import ydd_symmetric_edit as addon  # noqa: E402
from ydd_symmetric_edit import layer_names, operators  # noqa: E402

STATE = {
    "events": [],
    "started": 0.0,
}


def fail():
    traceback.print_exc()
    modal_ids = []
    try:
        modal_ids = [operator.bl_rna.identifier for operator in STATE["window"].modal_operators]
    except Exception:
        pass
    print(f"MODAL_IDS={modal_ids}", flush=True)
    print("YSE_NATIVE_MODAL_TEST_FAILED", flush=True)
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
    mesh = bpy.data.meshes.new("YSE_ModalMesh")
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
    obj = bpy.data.objects.new("YSE_ModalObject", mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.use_mesh_mirror_x = True
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    return obj


def window_coordinate(region, region_3d, coordinate):
    local = view3d_utils.location_3d_to_region_2d(region, region_3d, Vector(coordinate))
    if local is None:
        raise RuntimeError(f"Could not project test point {coordinate}")
    return int(round(region.x + local.x)), int(round(region.y + local.y))


def symmetric_cut_x_coordinates(bm):
    return sorted(
        sum(vertex.co.x for vertex in edge.verts) * 0.5
        for edge in bm.edges
        if {round(vertex.co.y, 6) for vertex in edge.verts} == {-1.0, 1.0}
        and all(1.0 < abs(vertex.co.x) < 2.0 for vertex in edge.verts)
    )


def verify_finished():
    try:
        window = STATE["window"]
        running = any(
            "NATIVE_SYMMETRIC_KNIFE" in operator.bl_rna.identifier.upper()
            or "KNIFE_TOOL" in operator.bl_rna.identifier.upper()
            for operator in window.modal_operators
        )
        if running or operators._SESSIONS:
            if time.monotonic() - STATE["started"] > 8.0:
                raise RuntimeError("Timed out waiting for the Knife macro to finish")
            return 0.1

        bm = bmesh.from_edit_mesh(STATE["object"].data)
        assert len(bm.verts) == 12, len(bm.verts)
        assert len(bm.edges) == 14, len(bm.edges)
        assert len(bm.faces) == 4, len(bm.faces)
        cut_x = symmetric_cut_x_coordinates(bm)
        assert len(cut_x) == 2 and cut_x[0] < 0.0 < cut_x[1], cut_x
        assert abs(cut_x[0] + cut_x[1]) <= 1.0e-7, cut_x
        assert bm.edges.layers.int.get(layer_names.EDGE_ORIGINAL_LAYER) is None
        assert bm.faces.layers.int.get(layer_names.FACE_ID_LAYER) is None

        with bpy.context.temp_override(window=STATE["window"], area=STATE["area"], region=STATE["region"]):
            undo_result = bpy.ops.ed.undo()
        assert undo_result == {"FINISHED"}, undo_result
        undone_object = bpy.data.objects.get("YSE_ModalObject")
        assert undone_object is not None and undone_object.mode == "EDIT"
        bm = bmesh.from_edit_mesh(undone_object.data)
        assert len(bm.verts) == 8 and len(bm.edges) == 8 and len(bm.faces) == 2
        assert bm.edges.layers.int.get(layer_names.EDGE_ORIGINAL_LAYER) is None
        assert bm.faces.layers.int.get(layer_names.FACE_ID_LAYER) is None
        print("YSE_NATIVE_MODAL_TEST_OK", flush=True)
        addon.unregister()
        bpy.ops.wm.quit_blender()
    except BaseException:
        fail()
    return None


def send_next_event():
    try:
        if STATE["events"]:
            event = STATE["events"].pop(0)
            STATE["window"].event_simulate(**event)
            return 0.08
        STATE["started"] = time.monotonic()
        bpy.app.timers.register(verify_finished, first_interval=0.4)
    except BaseException:
        fail()
    return None


def wait_for_session_start():
    try:
        if operators._SESSIONS:
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
            bpy.app.timers.register(send_next_event, first_interval=0.1)
            return None
        if time.monotonic() - STATE["started"] > 4.0:
            raise RuntimeError("Timed out waiting for the simulated K route")
        return 0.05
    except BaseException:
        fail()
    return None


def send_k_event():
    try:
        start = STATE["stroke"][0]
        STATE["window"].event_simulate(type="K", value="PRESS", x=start[0], y=start[1])
        STATE["window"].event_simulate(type="K", value="RELEASE", x=start[0], y=start[1])
        STATE["started"] = time.monotonic()
        bpy.app.timers.register(wait_for_session_start, first_interval=0.1)
    except BaseException:
        fail()
    return None


def start_test():
    try:
        addon.register()
        addon.sync_persistent_keymap(True)
        window, area, region = viewport_context()
        configure_view(area)
        obj = make_mesh()
        with bpy.context.temp_override(window=window, area=area, region=region):
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.context.tool_settings.mesh_select_mode = (True, False, False)
            bpy.ops.ed.undo_push(message="YSE clean baseline")

        region_3d = area.spaces.active.region_3d
        start = window_coordinate(region, region_3d, (-1.5, -1.0, 0.0))
        end = window_coordinate(region, region_3d, (-1.5, 1.0, 0.0))
        STATE.update(
            window=window,
            area=area,
            region=region,
            object=obj,
            stroke=(start, end),
        )
        window.event_simulate(type="MOUSEMOVE", value="NOTHING", x=start[0], y=start[1])
        bpy.app.timers.register(send_k_event, first_interval=0.15)
    except BaseException:
        fail()
    return None


bpy.app.timers.register(start_test, first_interval=0.25)
