# SPDX-License-Identifier: GPL-3.0-or-later

"""GUI regression for REPLAY Merge By Distance redo/threshold re-run.

Run with Blender's real window/event loop::

    blender --factory-startup --enable-event-simulate \
        --python test_merge_redo.py

Uses the factory Cube so non-modal ``ed.undo`` memfiles stay coherent.
Sequence (F9 re-run is approximated by calling the operator again with new
parameters, per contract §4.6):
  1. By Distance with a tiny threshold (cube unit spacing → no merges).
  2. undo once → baseline restored.
  3. By Distance with threshold covering the X-span of one selected corner
     and its mirror.
  4. One pair merges; one more undo fully restores baseline.
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

PACKAGE_PARENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_PARENT))

import ydd_symmetric_edit as addon  # noqa: E402

MARKER_OK = "YSE_MERGE_REDO_TEST_OK"
MARKER_FAILED = "YSE_MERGE_REDO_TEST_FAILED"
COORD_PRECISION = 5
SMALL_THRESHOLD = 0.01
# Factory cube corners are 2.0 apart along each axis.  Threshold just above 2
# merges a selected corner with its X-mirror only when mirrors are added.
LARGE_THRESHOLD = 2.01
# Single +X corner; the operator adds the X-mirror before remove_doubles.
SELECT_COORDS = ((1.0, -1.0, -1.0),)
STATE = {}


def fail(message: str = "") -> None:
    if message:
        print(f"YSE_MERGE_REDO_ERROR={message}", flush=True)
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


def configure_view(area) -> None:
    region_3d = area.spaces.active.region_3d
    region_3d.view_perspective = "ORTHO"
    region_3d.view_rotation = Quaternion((1.0, 0.0, 0.0, 0.0))
    region_3d.view_location = (0.0, 0.0, 0.0)
    region_3d.view_distance = 8.0
    region_3d.update()


def coordinate_key(coordinate, precision: int = COORD_PRECISION):
    return tuple(round(float(value), precision) for value in coordinate)


def mirror_key(coordinate, precision: int = COORD_PRECISION):
    x, y, z = coordinate
    return coordinate_key((-float(x), float(y), float(z)), precision)


def vertex_coord_multiset(bm, precision: int = COORD_PRECISION) -> Counter:
    return Counter(coordinate_key(vertex.co, precision) for vertex in bm.verts)


def edge_coord_multiset(bm, precision: int = COORD_PRECISION) -> Counter:
    return Counter(tuple(sorted(coordinate_key(vertex.co, precision) for vertex in edge.verts)) for edge in bm.edges)


def topology_signature(bm):
    return (
        len(bm.verts),
        len(bm.edges),
        len(bm.faces),
        vertex_coord_multiset(bm),
        edge_coord_multiset(bm),
    )


def find_vertex(bm, expected, precision: int = COORD_PRECISION):
    key = coordinate_key(expected, precision)
    for vertex in bm.verts:
        if coordinate_key(vertex.co, precision) == key:
            return vertex
    raise AssertionError(f"vertex not found: {expected}")


def select_vertices(bm, coordinates) -> None:
    for face in bm.faces:
        face.select = False
    for edge in bm.edges:
        edge.select = False
    for vertex in bm.verts:
        vertex.select = False
    bm.select_history.clear()
    for coordinate in coordinates:
        vertex = find_vertex(bm, coordinate)
        vertex.select = True
        bm.select_history.add(vertex)


def active_mesh_object():
    obj = bpy.context.view_layer.objects.active
    if obj is not None and obj.type == "MESH":
        return obj
    for candidate in bpy.data.objects:
        if candidate.type == "MESH":
            return candidate
    return None


def phase_run():
    """One timer body: small → undo → large → undo, so each op's undo step is live."""

    try:
        window, area, region = STATE["window"], STATE["area"], STATE["region"]
        obj = active_mesh_object()
        assert obj is not None
        with bpy.context.temp_override(window=window, area=area, region=region):
            bm = bmesh.from_edit_mesh(obj.data)
            select_vertices(bm, SELECT_COORDS)
            bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)

            # 1) Small threshold: cube unit spacing → zero merges.
            result_small = bpy.ops.mesh.ydd_symmetric_edit_merge(
                mode="BY_DISTANCE",
                threshold=SMALL_THRESHOLD,
            )
            assert result_small == {"FINISHED"}, result_small
            bm = bmesh.from_edit_mesh(obj.data)
            assert topology_signature(bm) == STATE["baseline"]
            print("YSE_MERGE_REDO_SMALL_OK", flush=True)

            # 2) Undo the small (no-op) step so the next call is a clean re-run.
            undo_small = bpy.ops.ed.undo()
            assert undo_small == {"FINISHED"}, undo_small
            obj = active_mesh_object()
            assert obj is not None
            if obj.mode != "EDIT":
                bpy.context.view_layer.objects.active = obj
                bpy.ops.object.mode_set(mode="EDIT")
                # mode_set after undo creates its own undo step; push a fresh
                # baseline so the subsequent merge is a single undo unit.
                bpy.ops.ed.undo_push(message="YSE Merge Redo re-baseline")
            bm = bmesh.from_edit_mesh(obj.data)
            assert topology_signature(bm)[0] == STATE["baseline"][0]

            # 3) F9-style re-run with a larger threshold.
            select_vertices(bm, SELECT_COORDS)
            bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
            result_large = bpy.ops.mesh.ydd_symmetric_edit_merge(
                mode="BY_DISTANCE",
                threshold=LARGE_THRESHOLD,
            )
            assert result_large == {"FINISHED"}, result_large

            bm = bmesh.from_edit_mesh(obj.data)
            assert len(bm.verts) == STATE["baseline"][0] - 1, (len(bm.verts), STATE["baseline"][0])
            print(f"YSE_MERGE_REDO_LARGE_TOPO={topology_signature(bm)[:3]}", flush=True)

            # Survivor of the (1,-1,-1)/(-1,-1,-1) pair: either an endpoint
            # (4.x remove_doubles keep-one) or the midpoint on the plane (5.x
            # may average).  Exactly one vertex may remain in that YZ column.
            near_pair = [
                vertex for vertex in bm.verts if abs(vertex.co.y + 1.0) < 1.0e-4 and abs(vertex.co.z + 1.0) < 1.0e-4
            ]
            assert len(near_pair) == 1, [coordinate_key(v.co) for v in near_pair]
            assert abs(near_pair[0].co.x) <= 1.0 + 1.0e-4
            print(f"YSE_MERGE_REDO_SURVIVOR={coordinate_key(near_pair[0].co)}", flush=True)

            verts = vertex_coord_multiset(bm)
            mirrored = Counter(mirror_key(key) for key in verts.elements())
            if verts != mirrored:
                print(
                    f"YSE_MERGE_REDO_SYMMETRY_SOFT=coords not X-symmetric after remove_doubles; verts={verts}",
                    flush=True,
                )

            # 4) One undo restores the pre-large-merge baseline.
            undo_large = bpy.ops.ed.undo()
            assert undo_large == {"FINISHED"}, undo_large
            obj = active_mesh_object()
            assert obj is not None, f"missing after final undo; have={list(bpy.data.objects.keys())}"
            if obj.mode != "EDIT":
                bpy.context.view_layer.objects.active = obj
                bpy.ops.object.mode_set(mode="EDIT")
            bm = bmesh.from_edit_mesh(obj.data)
            assert topology_signature(bm)[0] == STATE["baseline"][0], topology_signature(bm)[:3]
            assert topology_signature(bm)[3] == STATE["baseline"][3], "vertex multiset not restored"

        print(MARKER_OK, flush=True)
        addon.unregister()
        bpy.ops.wm.quit_blender()
    except BaseException:
        fail()
    return None


def phase_setup():
    try:
        addon.register()
        window, area, region = viewport_context()
        configure_view(area)

        obj = bpy.data.objects["Cube"]
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        obj.use_mesh_mirror_x = True
        obj.use_mesh_mirror_y = False
        obj.use_mesh_mirror_z = False
        for name in ("Camera", "Light"):
            other = bpy.data.objects.get(name)
            if other is not None:
                other.select_set(False)

        with bpy.context.temp_override(window=window, area=area, region=region):
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.context.tool_settings.mesh_select_mode = (True, False, False)
            bpy.ops.ed.undo_push(message="YSE Merge Redo baseline")

            bm = bmesh.from_edit_mesh(obj.data)
            baseline = topology_signature(bm)
            assert baseline[0] == 8, baseline[0]

        STATE.update(
            window=window,
            area=area,
            region=region,
            baseline=baseline,
        )
        bpy.app.timers.register(phase_run, first_interval=0.15)
    except BaseException:
        fail()
    return None


bpy.context.preferences.view.show_splash = False
bpy.context.preferences.view.smooth_view = 0
bpy.context.preferences.view.use_save_prompt = False
bpy.context.preferences.filepaths.use_auto_save_temporary_files = False
bpy.app.timers.register(phase_setup, first_interval=0.25)
