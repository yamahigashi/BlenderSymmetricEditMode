# SPDX-License-Identifier: GPL-3.0-or-later

"""Interactive-window integration for direct and fallback Knife processing."""

from __future__ import annotations

import math
import os
import sys
import traceback
from pathlib import Path

import bmesh
import bpy
from mathutils import Quaternion

PACKAGE_PARENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_PARENT))

import ydd_symmetric_edit as addon  # noqa: E402
from ydd_symmetric_edit import layer_names, operators, stitch_pathedges, stitch_reflect  # noqa: E402


def viewport_context():
    window = bpy.context.window_manager.windows[0]
    area = next(area for area in window.screen.areas if area.type == "VIEW_3D")
    region = next(region for region in area.regions if region.type == "WINDOW")
    return window, area, region


def configure_top_view(area):
    space = area.spaces.active
    region_3d = space.region_3d
    region_3d.view_perspective = "ORTHO"
    region_3d.view_rotation = Quaternion((1.0, 0.0, 0.0, 0.0))
    region_3d.view_location = (0.0, 0.0, 0.0)
    region_3d.view_distance = 6.0
    region_3d.update()


def make_symmetric_quads():
    mesh = bpy.data.meshes.new("YSE_TestMesh")
    vertices = [
        (-2.0, -1.0, 0.0),
        (-1.0, -1.0, 0.0),
        (-1.0, 1.0, 0.0),
        (-2.0, 1.0, 0.0),
        (1.0, -1.0, 0.0),
        (2.0, -1.0, 0.0),
        (2.0, 1.0, 0.0),
        (1.0, 1.0, 0.0),
    ]
    mesh.from_pydata(vertices, [], [(0, 1, 2, 3), (4, 5, 6, 7)])
    mesh.update()
    for layer_name, offset in (("UV_Main", 0.0), ("UV_Detail", 0.125)):
        uv_layer = mesh.uv_layers.new(name=layer_name)
        for loop in mesh.loops:
            coordinate = mesh.vertices[loop.vertex_index].co
            uv_layer.data[loop.index].uv = (
                coordinate.x * 0.2 + 0.5 + offset,
                coordinate.y * 0.25 + 0.5,
            )
    color = mesh.color_attributes.new(name="CornerColor", type="FLOAT_COLOR", domain="CORNER")
    for datum in color.data:
        datum.color = (0.2, 0.4, 0.8, 1.0)
    obj = bpy.data.objects.new("YSE_TestObject", mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.use_mesh_mirror_x = True
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    obj.shape_key_add(name="Basis")
    key = obj.shape_key_add(name="Key1")
    key.data[0].co.z = 0.6
    key.data[4].co.z = -0.4
    return obj


def simulate_source_knife(obj):
    bm = bmesh.from_edit_mesh(obj.data)
    bottom = next(
        edge
        for edge in bm.edges
        if all(vertex.co.x < 0.0 for vertex in edge.verts)
        and all(abs(vertex.co.y + 1.0) < 1.0e-8 for vertex in edge.verts)
    )
    _edge, bottom_vertex = bmesh.utils.edge_split(bottom, bottom.verts[0], 0.5)
    top = next(
        edge
        for edge in bm.edges
        if all(vertex.co.x < 0.0 for vertex in edge.verts)
        and all(abs(vertex.co.y - 1.0) < 1.0e-8 for vertex in edge.verts)
    )
    _edge, top_vertex = bmesh.utils.edge_split(top, top.verts[0], 0.5)
    source_face = next(face for face in bm.faces if bottom_vertex in face.verts and top_vertex in face.verts)
    bmesh.utils.face_split(source_face, bottom_vertex, top_vertex)
    path = bm.edges.get((bottom_vertex, top_vertex))
    path.select = True
    bottom_vertex.select = True
    top_vertex.select = True
    bm.select_history.clear()
    bm.select_history.add(path)
    bmesh.update_edit_mesh(obj.data, loop_triangles=True, destructive=True)


def simulate_source_bent_knife(obj):
    """Make the topology produced by a boundary/interior/boundary stroke."""

    bm = bmesh.from_edit_mesh(obj.data)
    bottom = next(
        edge
        for edge in bm.edges
        if all(vertex.co.x < 0.0 for vertex in edge.verts)
        and all(abs(vertex.co.y + 1.0) < 1.0e-8 for vertex in edge.verts)
    )
    _edge, bottom_vertex = bmesh.utils.edge_split(bottom, bottom.verts[0], 0.5)
    top = next(
        edge
        for edge in bm.edges
        if all(vertex.co.x < 0.0 for vertex in edge.verts)
        and all(abs(vertex.co.y - 1.0) < 1.0e-8 for vertex in edge.verts)
    )
    _edge, top_vertex = bmesh.utils.edge_split(top, top.verts[0], 0.5)
    source_face = next(face for face in bm.faces if bottom_vertex in face.verts and top_vertex in face.verts)
    bmesh.utils.face_split(
        source_face,
        bottom_vertex,
        top_vertex,
        coords=[(-1.2, 0.0, 0.0)],
    )
    bmesh.update_edit_mesh(obj.data, loop_triangles=True, destructive=True)


def has_exact_edge(bm, a, b, tolerance=1.0e-7):
    def close(co, expected):
        return all(abs(co[index] - expected[index]) <= tolerance for index in range(3))

    return any(
        (close(edge.verts[0].co, a) and close(edge.verts[1].co, b))
        or (close(edge.verts[0].co, b) and close(edge.verts[1].co, a))
        for edge in bm.edges
    )


def run_test():
    addon.register()
    window, area, region = viewport_context()
    configure_top_view(area)

    for old in tuple(bpy.data.objects):
        bpy.data.objects.remove(old, do_unlink=True)
    obj = make_symmetric_quads()

    with bpy.context.temp_override(window=window, area=area, region=region):
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.context.tool_settings.mesh_select_mode = (True, False, False)
        bm = bmesh.from_edit_mesh(obj.data)
        right_face = next(face for face in bm.faces if face.calc_center_median().x > 0.0)
        right_face.hide_set(True)
        bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
        prepared = operators._prepare_session(
            bpy.context,
            lambda _severity, _message: None,
            tool_kind="KNIFE",
        )
        assert prepared
        simulate_source_knife(obj)
        finished = bpy.ops.mesh.ydd_symmetric_edit_finish("EXEC_DEFAULT")
        assert finished == {"FINISHED"}, finished

    bm = bmesh.from_edit_mesh(obj.data)
    assert len(bm.verts) == 12, len(bm.verts)
    assert len(bm.edges) == 14, len(bm.edges)
    assert len(bm.faces) == 4, len(bm.faces)
    assert has_exact_edge(bm, (1.5, -1.0, 0.0), (1.5, 1.0, 0.0))
    assert bm.edges.layers.int.get(layer_names.EDGE_ORIGINAL_LAYER) is None
    assert bm.faces.layers.int.get(layer_names.FACE_ID_LAYER) is None
    assert all(face.hide for face in bm.faces if face.calc_center_median().x > 0.0)
    assert all(edge.hide for edge in bm.edges if all(vertex.co.x > 0.0 for vertex in edge.verts))
    assert all(vertex.hide for vertex in bm.verts if vertex.co.x > 0.0)
    active = bm.select_history.active
    assert isinstance(active, bmesh.types.BMEdge)
    assert abs((active.verts[0].co.x + active.verts[1].co.x) * 0.5 + 1.5) < 1.0e-7
    for layer_name in ("UV_Main", "UV_Detail"):
        uv_layer = bm.loops.layers.uv.get(layer_name)
        assert uv_layer is not None
        assert all(
            math.isfinite(component) for face in bm.faces for loop in face.loops for component in loop[uv_layer].uv
        )
    assert bm.loops.layers.float_color.get("CornerColor") is not None
    assert list(bm.verts.layers.shape.keys()) == ["Basis", "Key1"]
    assert tuple(bpy.context.tool_settings.mesh_select_mode) == (True, False, False)
    assert not any(obj.name.startswith("YSE_TemporaryCutter") for obj in bpy.data.objects)

    with bpy.context.temp_override(window=window, area=area, region=region):
        bpy.ops.object.mode_set(mode="OBJECT")
    assert list(obj.data.shape_keys.key_blocks.keys()) == ["Basis", "Key1"]
    assert len(obj.data.shape_keys.key_blocks["Basis"].data) == 12

    # Interior multi-click waypoints are intentionally outside the current
    # direct builder and must continue through the Knife Project fallback.
    for old in tuple(bpy.data.objects):
        bpy.data.objects.remove(old, do_unlink=True)
    bent_obj = make_symmetric_quads()
    with bpy.context.temp_override(window=window, area=area, region=region):
        bpy.ops.object.mode_set(mode="EDIT")
        assert operators._prepare_session(
            bpy.context,
            lambda _severity, _message: None,
            tool_kind="KNIFE",
        )
        simulate_source_bent_knife(bent_obj)
        bent_bm = bmesh.from_edit_mesh(bent_obj.data)
        bent_session = next(iter(operators._SESSIONS.values()))
        bent_source, _side, _total, _crossing = stitch_pathedges.collect_source_path_edges(
            bent_bm,
            bent_session.axis_index,
            bent_session.tolerance,
            bent_session.source_side,
        )
        assert len(bent_source) == 2
        assert not stitch_reflect.reflected_path_uses_only_target_boundaries(
            bent_bm,
            bent_source,
            bent_session.axis_index,
            bent_session.tolerance,
            bent_session.mirror_face_ids,
        )
        finished = bpy.ops.mesh.ydd_symmetric_edit_finish("EXEC_DEFAULT")
        assert finished == {"FINISHED"}, finished

    bent_bm = bmesh.from_edit_mesh(bent_obj.data)
    assert (len(bent_bm.verts), len(bent_bm.edges), len(bent_bm.faces)) == (
        14,
        16,
        4,
    )
    assert has_exact_edge(bent_bm, (1.5, -1.0, 0.0), (1.2, 0.0, 0.0))
    assert has_exact_edge(bent_bm, (1.2, 0.0, 0.0), (1.5, 1.0, 0.0))
    assert bent_bm.edges.layers.int.get(layer_names.EDGE_ORIGINAL_LAYER) is None
    assert bent_bm.faces.layers.int.get(layer_names.FACE_ID_LAYER) is None
    assert not any(obj.name.startswith("YSE_TemporaryCutter") for obj in bpy.data.objects)

    print("YSE_INTEGRATION_TEST_OK", flush=True)
    addon.unregister()
    bpy.ops.wm.quit_blender()


def guarded_run():
    try:
        run_test()
    except BaseException:
        traceback.print_exc()
        print("YSE_INTEGRATION_TEST_FAILED", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    return None


bpy.app.timers.register(guarded_run, first_interval=0.25)
