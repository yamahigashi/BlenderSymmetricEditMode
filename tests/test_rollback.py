# SPDX-License-Identifier: GPL-3.0-or-later

"""Verify that a failed direct-topology validation rolls back the target cut."""

from __future__ import annotations

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
from ydd_symmetric_edit import core, operators  # noqa: E402


def make_object():
    mesh = bpy.data.meshes.new("YSE_RollbackMesh")
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
    obj = bpy.data.objects.new("YSE_RollbackObject", mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.use_mesh_mirror_x = True
    obj.shape_key_add(name="Basis")
    key = obj.shape_key_add(name="Key1")
    key.data[0].co.z = 0.75
    key.data[4].co.z = -0.5
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    return obj


def source_cut(obj):
    bm = bmesh.from_edit_mesh(obj.data)
    bottom = next(
        edge
        for edge in bm.edges
        if all(vertex.co.x < 0.0 for vertex in edge.verts)
        and all(abs(vertex.co.y + 1.0) < 1.0e-8 for vertex in edge.verts)
    )
    _edge, a = bmesh.utils.edge_split(bottom, bottom.verts[0], 0.5)
    top = next(
        edge
        for edge in bm.edges
        if all(vertex.co.x < 0.0 for vertex in edge.verts)
        and all(abs(vertex.co.y - 1.0) < 1.0e-8 for vertex in edge.verts)
    )
    _edge, b = bmesh.utils.edge_split(top, top.verts[0], 0.5)
    face = next(face for face in bm.faces if a in face.verts and b in face.verts)
    bmesh.utils.face_split(face, a, b)
    bmesh.update_edit_mesh(obj.data, loop_triangles=True, destructive=True)


def run():
    addon.register()
    for old in tuple(bpy.data.objects):
        bpy.data.objects.remove(old, do_unlink=True)
    window = bpy.context.window_manager.windows[0]
    area = next(area for area in window.screen.areas if area.type == "VIEW_3D")
    region = next(region for region in area.regions if region.type == "WINDOW")
    region_3d = area.spaces.active.region_3d
    region_3d.view_perspective = "ORTHO"
    region_3d.view_rotation = Quaternion((1.0, 0.0, 0.0, 0.0))
    region_3d.view_location = (0.0, 0.0, 0.0)
    region_3d.view_distance = 6.0
    region_3d.update()
    obj = make_object()

    original_apply = core.apply_reflected_path_topology
    with bpy.context.temp_override(window=window, area=area, region=region):
        bpy.ops.object.mode_set(mode="EDIT")
        assert operators._prepare_session(
            bpy.context,
            lambda _severity, _message: None,
            tool_kind="KNIFE",
        )
        source_cut(obj)

        def forced_apply(*args, **kwargs):
            result = original_apply(*args, **kwargs)
            return (*result[:2], "forced rollback test", *result[3:])

        core.apply_reflected_path_topology = forced_apply
        try:
            result = bpy.ops.mesh.ydd_symmetric_edit_finish("EXEC_DEFAULT")
        finally:
            core.apply_reflected_path_topology = original_apply
        # Contract §2.2: successful rollback after mirror failure is a
        # legitimate decline → WARNING + FINISHED (native result kept).
        assert result == {"FINISHED"}, result
        warnings = [message for kind, message in operators._FINISH_REPORTS if kind == "WARNING"]
        assert any("forced rollback test" in message for message in warnings), (
            warnings,
            operators._FINISH_REPORTS,
        )

    bm = bmesh.from_edit_mesh(obj.data)
    assert len(bm.verts) == 10, len(bm.verts)
    assert len(bm.edges) == 11, len(bm.edges)
    assert len(bm.faces) == 3, len(bm.faces)
    assert any(
        all(-2.0 < vertex.co.x < -1.0 for vertex in edge.verts)
        and {round(vertex.co.y, 6) for vertex in edge.verts} == {-1.0, 1.0}
        for edge in bm.edges
    )
    assert not any(
        all(1.0 < vertex.co.x < 2.0 for vertex in edge.verts)
        and {round(vertex.co.y, 6) for vertex in edge.verts} == {-1.0, 1.0}
        for edge in bm.edges
    )
    assert bm.edges.layers.int.get(core.EDGE_ORIGINAL_LAYER) is None
    assert list(bm.verts.layers.shape.keys()) == ["Basis", "Key1"]
    assert not any(mesh.name.startswith("YSE_TemporaryBackup") for mesh in bpy.data.meshes)
    assert not any(obj.name.startswith("YSE_TemporaryCutter") for obj in bpy.data.objects)

    with bpy.context.temp_override(window=window, area=area, region=region):
        bpy.ops.object.mode_set(mode="OBJECT")
    shape_keys = obj.data.shape_keys
    assert shape_keys is not None
    assert list(shape_keys.key_blocks.keys()) == ["Basis", "Key1"]
    basis = shape_keys.key_blocks["Basis"]
    key = shape_keys.key_blocks["Key1"]
    outer_left = next(
        index
        for index, point in enumerate(basis.data)
        if abs(point.co.x + 2.0) < 1.0e-7 and abs(point.co.y + 1.0) < 1.0e-7
    )
    inner_right = next(
        index
        for index, point in enumerate(basis.data)
        if abs(point.co.x - 1.0) < 1.0e-7 and abs(point.co.y + 1.0) < 1.0e-7
    )
    assert abs(key.data[outer_left].co.z - 0.75) < 1.0e-7
    assert abs(key.data[inner_right].co.z + 0.5) < 1.0e-7
    print("YSE_ROLLBACK_TEST_OK", flush=True)
    addon.unregister()
    bpy.ops.wm.quit_blender()


def guarded():
    try:
        run()
    except BaseException:
        traceback.print_exc()
        print("YSE_ROLLBACK_TEST_FAILED", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    return None


bpy.app.timers.register(guarded, first_interval=0.25)
