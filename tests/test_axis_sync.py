# SPDX-License-Identifier: GPL-3.0-or-later

"""Verify that sessions read Blender's own Mesh Symmetry axis settings."""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

import bmesh
import bpy

PACKAGE_PARENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_PARENT))

import ydd_symmetric_edit as addon  # noqa: E402
from ydd_symmetric_edit import core, operators  # noqa: E402


def set_axes(obj, x=False, y=False, z=False):
    obj.use_mesh_mirror_x = x
    obj.use_mesh_mirror_y = y
    obj.use_mesh_mirror_z = z


def assert_clean(obj):
    bm = bmesh.from_edit_mesh(obj.data)
    assert bm.edges.layers.int.get(core.EDGE_ORIGINAL_LAYER) is None
    assert bm.faces.layers.int.get(core.FACE_ID_LAYER) is None
    assert not operators._SESSIONS


def ignore_report(_severity, _message):
    pass


def expect_prepare_failure():
    prepared = operators._prepare_session(
        bpy.context,
        ignore_report,
        tool_kind="KNIFE",
    )
    assert not prepared


def run():
    addon.register()
    window = bpy.context.window_manager.windows[0]
    area = next(area for area in window.screen.areas if area.type == "VIEW_3D")
    region = next(region for region in area.regions if region.type == "WINDOW")

    for old in tuple(bpy.data.objects):
        bpy.data.objects.remove(old, do_unlink=True)
    bpy.ops.mesh.primitive_cube_add()
    obj = bpy.context.object

    with bpy.context.temp_override(window=window, area=area, region=region):
        bpy.ops.object.mode_set(mode="EDIT")

        set_axes(obj)
        assert core.enabled_mesh_symmetry_axes(obj) == ()
        expect_prepare_failure()
        assert_clean(obj)

        set_axes(obj, x=True, y=True)
        assert core.enabled_mesh_symmetry_axes(obj) == (("X", 0), ("Y", 1))
        expect_prepare_failure()
        assert_clean(obj)

        for axis, index, values in (
            ("X", 0, (True, False, False)),
            ("Y", 1, (False, True, False)),
            ("Z", 2, (False, False, True)),
        ):
            set_axes(obj, *values)
            assert core.enabled_mesh_symmetry_axes(obj) == ((axis, index),)
            assert operators._prepare_session(
                bpy.context,
                ignore_report,
                tool_kind="KNIFE",
            )
            session = next(iter(operators._SESSIONS.values()))
            assert session.axis_index == index
            operators.cleanup_all_sessions()
            assert_clean(obj)

    print("YSE_AXIS_SYNC_TEST_OK", flush=True)
    addon.unregister()
    bpy.ops.wm.quit_blender()


def guarded():
    try:
        run()
    except BaseException:
        traceback.print_exc()
        print("YSE_AXIS_SYNC_TEST_FAILED", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    return None


bpy.app.timers.register(guarded, first_interval=0.25)
