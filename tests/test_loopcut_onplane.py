# SPDX-License-Identifier: GPL-3.0-or-later

"""GUI regression for all-PLANE Loop Cut (contract v3.1 §4 / §6 Wave 1).

A ring that lands entirely on x=0 is already symmetric: AUTO and POSITIVE
source_side must both succeed (native kept, no decline). A CROSSES-only path
whose source side cannot be determined stays on the current decline.

Run with Blender's real window as documented in ``docs/testing.md``::

    blender --factory-startup --enable-event-simulate --no-window-focus \
        --disable-crash-handler -p 40 40 960 600 --python test_loopcut_onplane.py
"""

from __future__ import annotations

import os
import sys
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
from ydd_symmetric_edit import layer_names, operators  # noqa: E402

MARKER_OK = "YSE_LOOPCUT_ONPLANE_TEST_OK"
MARKER_FAILED = "YSE_LOOPCUT_ONPLANE_TEST_FAILED"
COORD_PRECISION = 5
PLANE_INFO = "cut lies on the mirror plane"
SOURCE_SIDE_WARNING = "Could not determine the source side"
CUBE_BASELINE = (8, 12, 6)
CUBE_AFTER_CUT = (12, 20, 10)
QUAD_BASELINE = (4, 4, 1)


def fail(message=""):
    if message:
        print(f"YSE_LOOPCUT_ONPLANE_ERROR={message}", flush=True)
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
    region_3d.view_rotation = Quaternion((0.70710678, 0.70710678, 0.0, 0.0))
    region_3d.view_location = (0.0, 0.0, 0.0)
    region_3d.view_distance = 8.0
    region_3d.update()


def override(window, area, region):
    return bpy.context.temp_override(window=window, area=area, region=region)


def coordinate_key(coordinate, precision=COORD_PRECISION):
    return tuple(round(float(value), precision) for value in coordinate)


def assert_x_symmetric(bm):
    live = Counter(coordinate_key(vertex.co) for vertex in bm.verts)
    mirrored = Counter((-x, y, z) for x, y, z in live.elements())
    assert live == mirrored, (live - mirrored, mirrored - live)


def assert_layers_removed(bm):
    for name in layer_names.TEMP_LAYER_NAMES:
        for sequence in (bm.verts, bm.edges, bm.faces):
            assert sequence.layers.int.get(name) is None, f"temporary layer leaked: {name}"


def warning_messages():
    return [message for kind, message in operators._FINISH_REPORTS if kind == "WARNING"]


def info_messages():
    return [message for kind, message in operators._FINISH_REPORTS if kind == "INFO"]


def error_messages():
    return [message for kind, message in operators._FINISH_REPORTS if kind == "ERROR"]


def clear_scene():
    for old in tuple(bpy.data.objects):
        bpy.data.objects.remove(old, do_unlink=True)
    for mesh in tuple(bpy.data.meshes):
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)


def make_object(name, coords, faces):
    mesh = bpy.data.meshes.new(name + "Mesh")
    mesh.from_pydata(coords, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    obj.use_mesh_mirror_x = True
    obj.use_mesh_mirror_y = False
    obj.use_mesh_mirror_z = False
    return obj


def cube_coords():
    coords = [
        (-1.0, -1.0, -1.0),
        (1.0, -1.0, -1.0),
        (1.0, 1.0, -1.0),
        (-1.0, 1.0, -1.0),
        (-1.0, -1.0, 1.0),
        (1.0, -1.0, 1.0),
        (1.0, 1.0, 1.0),
        (-1.0, 1.0, 1.0),
    ]
    faces = [
        (0, 1, 2, 3),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7),
    ]
    return coords, faces


def crosses_quad_coords():
    coords = [
        (-1.0, -1.0, 0.0),
        (1.0, -1.0, 0.0),
        (1.0, 1.0, 0.0),
        (-1.0, 1.0, 0.0),
    ]
    return coords, [(0, 1, 2, 3)]


def set_source_side(value):
    settings = bpy.context.scene.ydd_symmetric_edit
    settings.source_side = value
    assert settings.source_side == value, settings.source_side


def enter_edit(obj, window, area, region):
    with override(window, area, region):
        # Push while the object exists in object mode so one later undo cannot
        # drop the datablock (EXEC_DEFAULT loopcut sometimes shares a step).
        bpy.ops.ed.undo_push(message=f"YSE loopcut onplane object {obj.name}")
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.context.tool_settings.mesh_select_mode = (False, True, False)
        bpy.ops.mesh.select_all(action="DESELECT")
        bpy.ops.ed.undo_push(message=f"YSE loopcut onplane baseline {obj.name}")
    return bmesh.from_edit_mesh(obj.data)


def refetch_edit_object(name, window, area, region):
    obj = bpy.data.objects.get(name)
    assert obj is not None, f"object {name!r} disappeared"
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    if obj.mode != "EDIT":
        with override(window, area, region):
            result = bpy.ops.object.mode_set(mode="EDIT")
            assert result == {"FINISHED"}, result
    return obj


def find_x_aligned_edge_index(bm):
    for edge in bm.edges:
        a, b = edge.verts
        if abs(a.co.y - b.co.y) < 1e-6 and abs(a.co.z - b.co.z) < 1e-6 and abs(a.co.x - b.co.x) > 0.5:
            return edge.index
    raise AssertionError("no X-aligned cube edge")


def find_y_aligned_negative_edge_index(bm):
    for edge in bm.edges:
        a, b = edge.verts
        if abs(a.co.x - b.co.x) < 1e-6 and abs(a.co.z - b.co.z) < 1e-6 and abs(a.co.y - b.co.y) > 0.5:
            if a.co.x < -0.5 and b.co.x < -0.5:
                return edge.index
    raise AssertionError("no Y-aligned edge on the negative side")


def run_loopcut(window, area, region, *, edge_index, number_cuts=1, value=0.0):
    operators._FINISH_REPORTS.clear()
    with override(window, area, region):
        prepared = operators._prepare_session(
            bpy.context,
            lambda _level, _message: None,
            tool_kind="LOOP_CUT",
        )
        assert prepared, "failed to prepare LOOP_CUT session"
        result = bpy.ops.mesh.loopcut_slide(
            "EXEC_DEFAULT",
            MESH_OT_loopcut={
                "number_cuts": number_cuts,
                "object_index": 0,
                "edge_index": edge_index,
                "mesh_select_mode_init": (False, True, False),
            },
            TRANSFORM_OT_edge_slide={"value": value},
        )
        assert "FINISHED" in result, result
        finished = bpy.ops.mesh.ydd_symmetric_edit_finish("EXEC_DEFAULT")
        assert finished == {"FINISHED"}, finished
    return finished


def assert_all_new_edges_on_plane(bm, baseline_vert_count, *, tol=1e-4):
    on_plane = 0
    off_plane = 0
    for edge in bm.edges:
        if all(abs(vertex.co.x) <= tol for vertex in edge.verts):
            # Count only the new mid-ring (both verts not in the original cube
            # corners at |x|=1). New ring verts sit at x=0.
            if all(abs(vertex.co.x) <= tol for vertex in edge.verts):
                on_plane += 1
        elif any(abs(vertex.co.x) <= tol for vertex in edge.verts):
            off_plane += 1
    assert baseline_vert_count == CUBE_BASELINE[0]
    new_plane_verts = [vertex for vertex in bm.verts if abs(vertex.co.x) <= tol]
    assert len(new_plane_verts) == 4, len(new_plane_verts)
    assert all(abs(abs(vertex.co.y) - 1.0) < 1e-4 or abs(abs(vertex.co.z) - 1.0) < 1e-4 for vertex in new_plane_verts)


def case_all_plane(window, area, region, *, source_side):
    print(f"YSE_LOOPCUT_ONPLANE_CASE=all_plane_{source_side.lower()}", flush=True)
    clear_scene()
    set_source_side(source_side)
    coords, faces = cube_coords()
    obj = make_object(f"OnPlane_{source_side}", coords, faces)
    bm = enter_edit(obj, window, area, region)
    edge_index = find_x_aligned_edge_index(bm)
    bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)

    run_loopcut(window, area, region, edge_index=edge_index, number_cuts=1, value=0.0)
    assert not error_messages(), operators._FINISH_REPORTS
    warnings = warning_messages()
    infos = info_messages()
    assert not any(SOURCE_SIDE_WARNING in message for message in warnings), warnings
    assert any(PLANE_INFO in message for message in infos), infos

    bm = bmesh.from_edit_mesh(obj.data)
    counts = (len(bm.verts), len(bm.edges), len(bm.faces))
    print(f"YSE_LOOPCUT_ONPLANE_NET all_plane_{source_side.lower()}={counts}", flush=True)
    assert counts == CUBE_AFTER_CUT, counts
    assert_x_symmetric(bm)
    assert_all_new_edges_on_plane(bm, CUBE_BASELINE[0])
    assert_layers_removed(bm)

    name = obj.name
    with override(window, area, region):
        undo = bpy.ops.ed.undo()
    assert undo == {"FINISHED"}, undo
    obj = refetch_edit_object(name, window, area, region)
    bm = bmesh.from_edit_mesh(obj.data)
    assert (len(bm.verts), len(bm.edges), len(bm.faces)) == CUBE_BASELINE
    assert_x_symmetric(bm)
    assert_layers_removed(bm)
    print(
        f"YSE_LOOPCUT_ONPLANE_OK_CASE=all_plane_{source_side.lower()} infos={infos} warnings={warnings}",
        flush=True,
    )


def case_crosses_undetermined(window, area, region):
    print("YSE_LOOPCUT_ONPLANE_CASE=crosses_undetermined", flush=True)
    clear_scene()
    set_source_side("AUTO")
    coords, faces = crosses_quad_coords()
    obj = make_object("CrossesUndetermined", coords, faces)
    bm = enter_edit(obj, window, area, region)
    edge_index = find_y_aligned_negative_edge_index(bm)
    bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)

    run_loopcut(window, area, region, edge_index=edge_index, number_cuts=1, value=0.0)
    assert not error_messages(), operators._FINISH_REPORTS
    warnings = warning_messages()
    infos = info_messages()
    assert any(SOURCE_SIDE_WARNING in message for message in warnings), (warnings, infos)
    assert not any(PLANE_INFO in message for message in infos), infos

    bm = bmesh.from_edit_mesh(obj.data)
    counts = (len(bm.verts), len(bm.edges), len(bm.faces))
    print(f"YSE_LOOPCUT_ONPLANE_NET crosses_undetermined={counts}", flush=True)
    # Native keeps the single CROSSES cut (two new verts, three new edges, one
    # extra face). Side is undetermined so nothing is remirrored.
    assert counts[0] > QUAD_BASELINE[0], counts
    assert_layers_removed(bm)

    name = obj.name
    with override(window, area, region):
        undo = bpy.ops.ed.undo()
    assert undo == {"FINISHED"}, undo
    obj = refetch_edit_object(name, window, area, region)
    bm = bmesh.from_edit_mesh(obj.data)
    assert (len(bm.verts), len(bm.edges), len(bm.faces)) == QUAD_BASELINE, (
        len(bm.verts),
        len(bm.edges),
        len(bm.faces),
    )
    print(
        f"YSE_LOOPCUT_ONPLANE_OK_CASE=crosses_undetermined infos={infos} warnings={warnings}",
        flush=True,
    )


def start_test():
    try:
        addon.register()
        addon.sync_persistent_keymap(True)
        window, area, region = viewport_context()
        configure_view(area)
        case_all_plane(window, area, region, source_side="AUTO")
        case_all_plane(window, area, region, source_side="POSITIVE")
        case_crosses_undetermined(window, area, region)
        print(MARKER_OK, flush=True)
        sys.stdout.flush()
        addon.unregister()
        bpy.ops.wm.quit_blender()
    except BaseException:
        fail()
    return None


bpy.app.timers.register(start_test, first_interval=0.3)
