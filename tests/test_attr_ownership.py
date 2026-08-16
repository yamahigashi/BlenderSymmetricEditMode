# SPDX-License-Identifier: GPL-3.0-or-later

"""Headless checks: the save-time attribute sweep only touches owned meshes."""

from __future__ import annotations

import sys
from pathlib import Path

import bpy

PACKAGE_PARENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_PARENT))

import ydd_symmetric_edit as addon  # noqa: E402
from ydd_symmetric_edit import layer_names, operators, replay, session_state  # noqa: E402


def make_mesh(name):
    mesh = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    return obj


def add_attr(mesh, name):
    mesh.attributes.new(name=name, type="INT", domain="POINT")


def check_sweep_is_scoped_to_touched_meshes():
    session_state._TOUCHED_MESH_NAMES.clear()
    touched = make_mesh("yse_touched")
    foreign = make_mesh("yse_foreign")
    add_attr(touched.data, layer_names.HISTORY_TOKEN_LAYER)
    add_attr(foreign.data, layer_names.VERT_SESSION_ID_LAYER)
    session_state.record_touched_mesh(touched.data.name)

    operators.cleanup_stale_attributes()

    assert touched.data.attributes.get(layer_names.HISTORY_TOKEN_LAYER) is None, "owned attribute must be swept"
    assert foreign.data.attributes.get(layer_names.VERT_SESSION_ID_LAYER) is not None, (
        "foreign same-named attribute must survive the save sweep"
    )


def check_load_recovery_sweeps_everything():
    foreign = bpy.data.objects["yse_foreign"]
    assert foreign.data.attributes.get(layer_names.VERT_SESSION_ID_LAYER) is not None

    operators.cleanup_after_load()

    assert foreign.data.attributes.get(layer_names.VERT_SESSION_ID_LAYER) is None, (
        "load recovery must sweep stale layers regardless of ownership"
    )
    assert not session_state._TOUCHED_MESH_NAMES, "registry must reset on load"


def check_symmetry_gate_records_touched_mesh():
    obj = make_mesh("yse_gate")
    import bmesh

    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=2.0)
    bm.to_mesh(obj.data)
    bm.free()
    obj.use_mesh_mirror_x = True
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    try:
        session_state._TOUCHED_MESH_NAMES.clear()
        parameters = replay._symmetry_parameters(bpy.context)
        assert parameters is not None, "symmetry gate should accept the mirrored cube"
        assert obj.data.name in session_state._TOUCHED_MESH_NAMES, (
            "the replay/delete entry gate must record the mesh as touched"
        )
    finally:
        bpy.ops.object.mode_set(mode="OBJECT")


def clear_startup_objects():
    # The factory-startup Cube would otherwise join edit mode and trip the
    # multi-object guard in _symmetry_parameters.
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)


def run():
    addon.register()
    try:
        clear_startup_objects()
        check_sweep_is_scoped_to_touched_meshes()
        check_load_recovery_sweeps_everything()
        check_symmetry_gate_records_touched_mesh()
    finally:
        addon.unregister()
    print("YSE_ATTR_OWNERSHIP_OK", flush=True)


if __name__ == "__main__":
    run()
