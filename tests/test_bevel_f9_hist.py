# SPDX-License-Identifier: GPL-3.0-or-later

"""Bevel F9 on a mesh carrying a residual history token layer (v5.2 §5C-4).

Knife/loop-cut/connect leave their committed token layers on the mesh long
after the cut settled; their mere presence must not veto the F9
re-intervention (measured regression: every F9 failed closed on such
meshes).

One ``ed.undo_redo`` per Blender process: confirm an expanded bevel through
the intercept route, adjust ``offset`` on the native active operator, run F9
once, and assert the undo_post re-expansion kept both sides symmetric and the
OFF normalization kept only the user side selected.

Run with Blender's real window/event loop::

    blender --factory-startup --enable-event-simulate --python test_bevel_f9.py
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
from mathutils import Quaternion

bpy.context.preferences.view.show_splash = False
bpy.context.preferences.view.smooth_view = 0
bpy.context.preferences.view.use_save_prompt = False
bpy.context.preferences.filepaths.use_auto_save_temporary_files = False

PACKAGE_PARENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_PARENT))

import ydd_symmetric_edit as addon  # noqa: E402
from ydd_symmetric_edit import inset_bevel, keymaps, ui  # noqa: E402

MARKER_OK = "YSE_BEVEL_F9_HIST_TEST_OK"
MARKER_FAILED = "YSE_BEVEL_F9_HIST_TEST_FAILED"
PRECISION = 5
STATE: dict = {"gen": None, "wait_modal": False, "window": None, "area": None, "region": None}


def fail(message=""):
    if message:
        print(f"YSE_BEVEL_F9_ERROR={message}", flush=True)
    traceback.print_exc()
    print(f"YSE_BEVEL_F9_REPORTS={list(inset_bevel._REPORTS)}", flush=True)
    print(f"YSE_BEVEL_F9_TOKEN={inset_bevel._ACTIVE_TOKEN}", flush=True)
    print(MARKER_FAILED, flush=True)
    sys.stdout.flush()
    os._exit(1)


def override():
    return bpy.context.temp_override(window=STATE["window"], area=STATE["area"], region=STATE["region"])


def configure_view(area):
    space = area.spaces.active
    space.show_gizmo = False
    space.show_gizmo_tool = False
    region_3d = space.region_3d
    region_3d.view_perspective = "ORTHO"
    region_3d.view_rotation = Quaternion((1.0, 0.0, 0.0, 0.0))
    region_3d.view_location = (0.0, 0.0, 0.0)
    region_3d.view_distance = 8.0
    region_3d.update()


def build_grid():
    if bpy.context.mode != "OBJECT":
        with override():
            bpy.ops.object.mode_set(mode="OBJECT")
    for old in tuple(bpy.data.objects):
        bpy.data.objects.remove(old, do_unlink=True)
    with override():
        bpy.ops.mesh.primitive_grid_add(x_subdivisions=4, y_subdivisions=4, size=2.0)
        bpy.ops.wm.tool_set_by_id(name="builtin.select_box")
    obj = bpy.context.active_object
    obj.use_mesh_mirror_x = True
    obj.use_mesh_mirror_y = False
    obj.use_mesh_mirror_z = False
    with override():
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_mode(type="EDGE")
    configure_view(STATE["area"])
    bm = bmesh.from_edit_mesh(obj.data)
    for sequence in (bm.faces, bm.edges, bm.verts):
        for element in sequence:
            element.select = False
    bm.edges.ensure_lookup_table()
    target = None
    for edge in bm.edges:
        first, second = edge.verts
        if (
            abs(first.co.x - 0.5) < 1e-6
            and abs(second.co.x - 0.5) < 1e-6
            and -0.1 < first.co.y < 0.6
        ):
            target = edge
            break
    if target is None:
        raise AssertionError("fixture edge not found")
    target.select = True
    for vertex in target.verts:
        vertex.select = True
    bm.select_flush_mode()
    from ydd_symmetric_edit import layer_names
    bm.faces.layers.int.new(layer_names.HISTORY_TOKEN_LAYER)
    bmesh.update_edit_mesh(obj.data)
    with override():
        bpy.ops.ed.undo_push(message="user select")
    return obj


def vertex_multiset(obj):
    bm = bmesh.from_edit_mesh(obj.data)
    return Counter(tuple(round(float(value), PRECISION) for value in vertex.co) for vertex in bm.verts)


def assert_symmetric(obj, *, label):
    counts = vertex_multiset(obj)
    mirrored = Counter({(-x, y, z): n for (x, y, z), n in counts.items()})
    assert counts == mirrored, f"{label}: not X-symmetric"


def topology_counts(obj):
    bm = bmesh.from_edit_mesh(obj.data)
    return len(bm.verts), len(bm.edges), len(bm.faces)


def wait_until(predicate, timeout, message):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError(message)
        yield 0.1


def route_ready():
    addon_config = bpy.context.window_manager.keyconfigs.addon
    if addon_config is None:
        return False
    for keymap in addon_config.keymaps:
        for item in keymap.keymap_items:
            if (
                item.idname == keymaps.INSET_BEVEL_INTERCEPT_OPERATOR
                and item.type == "B"
                and bool(item.ctrl)
                and not bool(item.shift)
                and item.active
            ):
                return True
    return False


def token_settled():
    return inset_bevel._ACTIVE_TOKEN is None


def main_gen():
    window = STATE["window"]
    region = STATE["region"]
    obj = build_grid()
    yield from wait_until(route_ready, 10.0, "bevel intercept route never appeared")
    cx = region.x + region.width // 2
    cy = region.y + region.height // 2
    window.event_simulate(type="MOUSEMOVE", value="NOTHING", x=cx, y=cy)
    yield 0.1
    window.event_simulate(type="B", value="PRESS", ctrl=True, x=cx, y=cy)
    window.event_simulate(type="B", value="RELEASE", ctrl=True, x=cx, y=cy)
    yield 0.3
    window.event_simulate(type="MOUSEMOVE", value="NOTHING", x=cx + 60, y=cy + 40)
    yield 0.15
    window.event_simulate(type="MOUSEMOVE", value="NOTHING", x=cx + 110, y=cy + 75)
    yield 0.15
    window.event_simulate(type="LEFTMOUSE", value="PRESS", x=cx + 110, y=cy + 75)
    window.event_simulate(type="LEFTMOUSE", value="RELEASE", x=cx + 110, y=cy + 75)
    STATE["wait_modal"] = True
    yield 0.3
    yield from wait_until(token_settled, 8.0, "replay token did not settle after confirm")
    yield 0.3

    obj = bpy.context.active_object
    confirmed_counts = topology_counts(obj)
    assert_symmetric(obj, label="after confirm")
    before_f9 = vertex_multiset(obj)

    active = bpy.context.active_operator
    assert active is not None and "bevel" in active.bl_idname.lower(), (
        f"active_operator is not the native bevel: {getattr(active, 'bl_idname', None)}"
    )
    active.properties.offset = 0.3

    with override():
        assert bpy.ops.ed.undo_redo.poll(), "ed.undo_redo.poll() is false after bevel confirm"
        result = bpy.ops.ed.undo_redo()
    assert result == {"FINISHED"}, result
    yield 0.4
    yield from wait_until(token_settled, 8.0, "F9 re-intervention token did not settle")
    yield 0.3

    obj = bpy.context.active_object
    assert topology_counts(obj) == confirmed_counts, "F9 changed the bevel topology"
    after_f9 = vertex_multiset(obj)
    assert after_f9 != before_f9, "adjusted offset produced identical geometry"
    assert_symmetric(obj, label="after F9")

    bm = bmesh.from_edit_mesh(obj.data)
    assert not any(
        vertex.select and vertex.co.x < -1e-4 for vertex in bm.verts
    ), "OFF normalization left mirror-side selection after F9"
    assert any(vertex.select for vertex in bm.verts), "F9 left nothing selected"

    assert inset_bevel._ACTIVE_TOKEN is None, "token left active after F9"
    print(MARKER_OK, flush=True)
    sys.stdout.flush()


def tick():
    try:
        if STATE["wait_modal"]:
            if STATE["window"].modal_operators:
                return 0.15
            STATE["wait_modal"] = False
        if STATE["gen"] is None:
            return None
        try:
            delay = next(STATE["gen"])
        except StopIteration:
            finish()
            return None
        return delay
    except BaseException:
        fail()
        return None


def finish():
    if bpy.context.mode != "OBJECT":
        with override():
            bpy.ops.object.mode_set(mode="OBJECT")
    for old in tuple(bpy.data.objects):
        bpy.data.objects.remove(old, do_unlink=True)

    def quit_now():
        bpy.ops.wm.quit_blender()
        return None

    bpy.app.timers.register(quit_now, first_interval=0.2)


def start_test():
    try:
        addon.register()
        preferences = ui.get_addon_preferences(bpy.context)
        if preferences is not None:
            preferences.enabled = True
        addon.sync_persistent_keymap(True)
        window = bpy.context.window_manager.windows[0]
        area = next(area for area in window.screen.areas if area.type == "VIEW_3D")
        region = next(region for region in area.regions if region.type == "WINDOW")
        STATE.update(window=window, area=area, region=region)
        STATE["gen"] = main_gen()
        bpy.app.timers.register(tick, first_interval=0.4)
    except BaseException:
        fail()
    return None


bpy.app.timers.register(start_test, first_interval=0.2)
