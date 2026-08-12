# SPDX-License-Identifier: GPL-3.0-or-later

"""End-to-end cancellation test for temporary-layer cleanup."""

from __future__ import annotations

import os
import sys
import time
import traceback
from pathlib import Path

import bmesh
import bpy

# Prevent Blender's startup UI from consuming simulated keyboard events.
bpy.context.preferences.view.show_splash = False
bpy.context.preferences.view.smooth_view = 0
bpy.context.preferences.view.use_save_prompt = False
bpy.context.preferences.filepaths.use_auto_save_temporary_files = False

PACKAGE_PARENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_PARENT))

import ydd_symmetric_edit as addon  # noqa: E402
from ydd_symmetric_edit import layer_names, operators  # noqa: E402

STATE = {"started": 0.0}


def fail():
    traceback.print_exc()
    print("YSE_NATIVE_CANCEL_TEST_FAILED", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(1)


def verify_cancelled():
    try:
        window = STATE["window"]
        running = any(
            "NATIVE_SYMMETRIC_KNIFE" in operator.bl_rna.identifier.upper()
            or "KNIFE_TOOL" in operator.bl_rna.identifier.upper()
            for operator in window.modal_operators
        )
        if running or operators._SESSIONS:
            if time.monotonic() - STATE["started"] > 5.0:
                raise RuntimeError("Timed out waiting for cancel cleanup")
            return 0.1

        bm = bmesh.from_edit_mesh(STATE["object"].data)
        assert len(bm.verts) == 8 and len(bm.edges) == 8 and len(bm.faces) == 2
        assert bm.edges.layers.int.get(layer_names.EDGE_ORIGINAL_LAYER) is None
        assert bm.faces.layers.int.get(layer_names.FACE_ID_LAYER) is None
        print("YSE_NATIVE_CANCEL_TEST_OK", flush=True)
        addon.unregister()
        bpy.ops.wm.quit_blender()
    except BaseException:
        fail()
    return None


def send_escape():
    try:
        window = STATE["window"]
        x, y = STATE["mouse"]
        window.event_simulate(type="ESC", value="PRESS", x=x, y=y)
        window.event_simulate(type="ESC", value="RELEASE", x=x, y=y)
        STATE["started"] = time.monotonic()
        bpy.app.timers.register(verify_cancelled, first_interval=0.4)
    except BaseException:
        fail()
    return None


def wait_for_session_start():
    try:
        if operators._SESSIONS:
            bpy.app.timers.register(send_escape, first_interval=0.1)
            return None
        if time.monotonic() - STATE["started"] > 4.0:
            raise RuntimeError("Timed out waiting for the simulated K route")
        return 0.05
    except BaseException:
        fail()
    return None


def send_k_event():
    try:
        x, y = STATE["mouse"]
        STATE["window"].event_simulate(type="K", value="PRESS", x=x, y=y)
        STATE["window"].event_simulate(type="K", value="RELEASE", x=x, y=y)
        STATE["started"] = time.monotonic()
        bpy.app.timers.register(wait_for_session_start, first_interval=0.1)
    except BaseException:
        fail()
    return None


def start_test():
    try:
        addon.register()
        addon.sync_persistent_keymap(True)
        window = bpy.context.window_manager.windows[0]
        area = next(area for area in window.screen.areas if area.type == "VIEW_3D")
        region = next(region for region in area.regions if region.type == "WINDOW")
        for old in tuple(bpy.data.objects):
            bpy.data.objects.remove(old, do_unlink=True)

        mesh = bpy.data.meshes.new("YSE_CancelMesh")
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
        obj = bpy.data.objects.new("YSE_CancelObject", mesh)
        bpy.context.scene.collection.objects.link(obj)
        obj.use_mesh_mirror_x = True
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)

        with bpy.context.temp_override(window=window, area=area, region=region):
            bpy.ops.object.mode_set(mode="EDIT")

        mouse = (region.x + region.width // 2, region.y + region.height // 2)
        STATE.update(window=window, object=obj, mouse=mouse)
        window.event_simulate(type="MOUSEMOVE", value="NOTHING", x=mouse[0], y=mouse[1])
        bpy.app.timers.register(send_k_event, first_interval=0.15)
    except BaseException:
        fail()
    return None


bpy.app.timers.register(start_test, first_interval=0.25)
