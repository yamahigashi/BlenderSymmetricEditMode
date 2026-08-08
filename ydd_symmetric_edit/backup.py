# SPDX-License-Identifier: GPL-3.0-or-later

"""Whole-mesh topology backup shared by the mirrored post-processes.

Lives in its own module: the one-shot Connect/Merge replay (``replay``) needs
the same transaction primitives as the cut/rip sessions, and importing
``operators`` from ``replay`` would be circular.  Not in ``core`` because
these functions depend on ``bpy.data``.
"""

from __future__ import annotations

import traceback

import bmesh
import bpy

from . import core
from ._types import TopologyBackup


def create_topology_backup(bm: bmesh.types.BMesh) -> TopologyBackup:
    old_id_layer = bm.verts.layers.int.get(core.VERT_BACKUP_ID_LAYER)
    if old_id_layer is not None:
        bm.verts.layers.int.remove(old_id_layer)
    id_layer = bm.verts.layers.int.new(core.VERT_BACKUP_ID_LAYER)
    for vertex_id, vertex in enumerate(bm.verts, start=1):
        vertex[id_layer] = vertex_id

    shape_values = {}
    for shape_layer in bm.verts.layers.shape.values():
        shape_values[shape_layer.name] = [vertex[shape_layer].copy() for vertex in bm.verts]

    backup_mesh = bpy.data.meshes.new("YSE_TemporaryBackup")
    try:
        bm.to_mesh(backup_mesh)
    except BaseException:
        # The caller never receives the TopologyBackup, so the datablock would
        # leak on every failed attempt; drop it before re-raising.
        try:
            bpy.data.meshes.remove(backup_mesh)
        except Exception:
            traceback.print_exc()
        raise
    return TopologyBackup(mesh=backup_mesh, shape_values=shape_values)


def restore_topology_backup(mesh, backup: TopologyBackup) -> None:
    bm = bmesh.from_edit_mesh(mesh)
    # Deleting elements keeps the live BMesh CustomData layer definitions and,
    # crucially, shape-layer UIDs.  bm.clear()/from_mesh() would silently detach
    # existing KeyBlocks and destroy shape-key deformations.
    if len(bm.verts):
        bmesh.ops.delete(bm, geom=list(bm.verts), context="VERTS")
    bm.from_mesh(backup.mesh)

    id_layer = bm.verts.layers.int.get(core.VERT_BACKUP_ID_LAYER)
    if id_layer is None:
        raise RuntimeError("Topology backup vertex IDs are missing")
    for shape_name, values_by_id in backup.shape_values.items():
        shape_layer = bm.verts.layers.shape.get(shape_name)
        if shape_layer is None:
            raise RuntimeError(f"Shape layer {shape_name!r} was lost during rollback")
        for vertex in bm.verts:
            vertex[shape_layer] = values_by_id[int(vertex[id_layer]) - 1]
    bm.normal_update()
    bmesh.update_edit_mesh(mesh, loop_triangles=True, destructive=True)


def remove_backup(backup: TopologyBackup | None) -> None:
    """Best-effort removal; must never raise.

    Callers run this from ``finally`` blocks whose contract is "always return
    the native result" — an exception here would skip the return and lose the
    undo push, which is exactly what the transaction exists to prevent.
    """

    try:
        if backup is not None and backup.mesh.name in bpy.data.meshes and backup.mesh.users == 0:
            bpy.data.meshes.remove(backup.mesh)
    except Exception:
        traceback.print_exc()
