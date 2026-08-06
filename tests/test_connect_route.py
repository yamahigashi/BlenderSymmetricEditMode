# SPDX-License-Identifier: GPL-3.0-or-later

"""GUI regression for REPLAY symmetric Vertex Connect (J).

Run with Blender's real window/event loop::

    blender --factory-startup --enable-event-simulate \
        --python test_connect_route.py

Uses the factory Cube (X-symmetric) so non-modal operator undo memfiles stay
coherent.  Calls ``mesh.ydd_symmetric_edit_connect`` after building
select_history on the +X face diagonal.  Passes when both sides gain the new
edge, coordinates stay X-symmetric, and one undo restores baseline topology.
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

MARKER_OK = "YSE_CONNECT_ROUTE_TEST_OK"
MARKER_FAILED = "YSE_CONNECT_ROUTE_TEST_FAILED"
COORD_PRECISION = 5
STATE = {}


def fail(message: str = "") -> None:
    if message:
        print(f"YSE_CONNECT_ROUTE_ERROR={message}", flush=True)
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
    region_3d.view_distance = 6.0
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


def mirrored_vertex_multiset(bm, precision: int = COORD_PRECISION) -> Counter:
    return Counter(mirror_key(vertex.co, precision) for vertex in bm.verts)


def mirrored_edge_multiset(bm, precision: int = COORD_PRECISION) -> Counter:
    return Counter(tuple(sorted(mirror_key(vertex.co, precision) for vertex in edge.verts)) for edge in bm.edges)


def assert_x_symmetric(bm) -> None:
    verts = vertex_coord_multiset(bm)
    edges = edge_coord_multiset(bm)
    assert verts == mirrored_vertex_multiset(bm), f"vertex coords not X-symmetric: {verts}"
    assert edges == mirrored_edge_multiset(bm), f"edges not X-symmetric: {edges}"


def has_edge_between(bm, a, b, precision: int = COORD_PRECISION) -> bool:
    key = tuple(sorted((coordinate_key(a, precision), coordinate_key(b, precision))))
    return key in edge_coord_multiset(bm)


def topology_counts(bm) -> tuple[int, int, int]:
    return len(bm.verts), len(bm.edges), len(bm.faces)


def find_vertex(bm, expected, precision: int = COORD_PRECISION):
    key = coordinate_key(expected, precision)
    for vertex in bm.verts:
        if coordinate_key(vertex.co, precision) == key:
            return vertex
    raise AssertionError(f"vertex not found: {expected}")


def select_path(bm, coordinates) -> None:
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


def ensure_edit(window, area, region, obj):
    if obj.mode != "EDIT":
        bpy.context.view_layer.objects.active = obj
        with bpy.context.temp_override(window=window, area=area, region=region):
            bpy.ops.object.mode_set(mode="EDIT")
    return bmesh.from_edit_mesh(obj.data)


def phase_verify_and_undo():
    try:
        window, area, region = STATE["window"], STATE["area"], STATE["region"]
        obj = active_mesh_object()
        assert obj is not None, f"no mesh object after connect; have={list(bpy.data.objects.keys())}"
        bm = ensure_edit(window, area, region, obj)

        # +2 edges / +2 faces: source diagonal and mirrored diagonal.
        assert topology_counts(bm) == (8, 14, 8), topology_counts(bm)
        assert has_edge_between(bm, (1.0, -1.0, -1.0), (1.0, 1.0, 1.0)), "missing source diagonal"
        assert has_edge_between(bm, (-1.0, -1.0, -1.0), (-1.0, 1.0, 1.0)), "missing mirrored diagonal"
        assert_x_symmetric(bm)

        with bpy.context.temp_override(window=window, area=area, region=region):
            undo_result = bpy.ops.ed.undo()
        assert undo_result == {"FINISHED"}, undo_result

        obj = active_mesh_object()
        assert obj is not None, f"object missing after one undo; have={list(bpy.data.objects.keys())}"
        bm = ensure_edit(window, area, region, obj)
        counts = topology_counts(bm)
        print(f"YSE_CONNECT_ROUTE_AFTER_UNDO={counts}", flush=True)
        assert counts == STATE["baseline"], (
            f"undo did not fully restore baseline in 1 step: {counts} != {STATE['baseline']}; "
            f"source diagonal still present={has_edge_between(bm, (1.0, -1.0, -1.0), (1.0, 1.0, 1.0))}; "
            f"mirror diagonal still present={has_edge_between(bm, (-1.0, -1.0, -1.0), (-1.0, 1.0, 1.0))}"
        )
        assert vertex_coord_multiset(bm) == STATE["baseline_verts"]
        assert edge_coord_multiset(bm) == STATE["baseline_edges"]
        assert not has_edge_between(bm, (1.0, -1.0, -1.0), (1.0, 1.0, 1.0))
        assert not has_edge_between(bm, (-1.0, -1.0, -1.0), (-1.0, 1.0, 1.0))
        assert_x_symmetric(bm)

        print(MARKER_OK, flush=True)
        addon.unregister()
        bpy.ops.wm.quit_blender()
    except BaseException:
        fail()
    return None


def phase_connect():
    try:
        window, area, region = STATE["window"], STATE["area"], STATE["region"]
        with bpy.context.temp_override(window=window, area=area, region=region):
            result = bpy.ops.mesh.ydd_symmetric_edit_connect()
        assert result == {"FINISHED"}, result
        bpy.app.timers.register(phase_verify_and_undo, first_interval=0.15)
    except BaseException:
        fail()
    return None


def phase_setup():
    try:
        addon.register()
        window, area, region = viewport_context()
        configure_view(area)

        # Factory Cube is X-symmetric about the origin and already lives in the
        # undo memfile chain.  RNA-built meshes make non-modal ed.undo jump back
        # to factory startup; keep the Cube datablock instead.
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
            bpy.ops.ed.undo_push(message="YSE Connect Route baseline")

            bm = bmesh.from_edit_mesh(obj.data)
            baseline = topology_counts(bm)
            baseline_verts = vertex_coord_multiset(bm)
            baseline_edges = edge_coord_multiset(bm)
            assert baseline == (8, 12, 6), baseline
            assert_x_symmetric(bm)

            # +X face diagonal.  Mirror path is the -X face diagonal.
            select_path(bm, ((1.0, -1.0, -1.0), (1.0, 1.0, 1.0)))
            bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)

        STATE.update(
            window=window,
            area=area,
            region=region,
            baseline=baseline,
            baseline_verts=baseline_verts,
            baseline_edges=baseline_edges,
        )
        bpy.app.timers.register(phase_connect, first_interval=0.15)
    except BaseException:
        fail()
    return None


bpy.context.preferences.view.show_splash = False
bpy.context.preferences.view.smooth_view = 0
bpy.context.preferences.view.use_save_prompt = False
bpy.context.preferences.filepaths.use_auto_save_temporary_files = False
bpy.app.timers.register(phase_setup, first_interval=0.25)
