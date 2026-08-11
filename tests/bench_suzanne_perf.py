# SPDX-License-Identifier: GPL-3.0-or-later

"""Measurement-only benchmark for the large-mesh Suzanne preparation path.

Run with Blender in background mode, for example::

    blender --factory-startup --background --python tests/bench_suzanne_perf.py

The fixture is built from Blender's deterministic Suzanne primitive and one
``subdivide_edges`` operation.  ``cuts=10`` is intentionally one operation,
not ten sequential subdivision passes.
"""

from __future__ import annotations

import statistics
import sys
import time
import traceback
from pathlib import Path

import bmesh
import bpy

PACKAGE_PARENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_PARENT))

from ydd_symmetric_edit import core  # noqa: E402

TOLERANCE = 1.0e-5
SUBDIVIDE_CUTS = 10
REPEATS = 5


def build_suzanne():
    """Build a fresh edit-mesh Suzanne object with one ten-cut subdivision."""

    mesh = bpy.data.meshes.new("YSE_SuzannePerfMesh")
    obj = bpy.data.objects.new("YSE_SuzannePerfObject", mesh)
    bpy.context.scene.collection.objects.link(obj)
    source_bm = bmesh.new()
    try:
        bmesh.ops.create_monkey(source_bm)
        bmesh.ops.subdivide_edges(
            source_bm,
            edges=list(source_bm.edges),
            cuts=SUBDIVIDE_CUTS,
            use_grid_fill=True,
        )
        source_bm.to_mesh(mesh)
        mesh.update()
    except Exception:
        bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.meshes.remove(mesh)
        raise
    finally:
        source_bm.free()

    for selected in bpy.context.selected_objects:
        selected.select_set(False)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    try:
        bpy.ops.object.mode_set(mode="EDIT")
    except Exception:
        bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.meshes.remove(mesh)
        raise
    return obj, bmesh.from_edit_mesh(mesh)


def destroy_suzanne(obj) -> None:
    """Flush and remove one benchmark-owned Suzanne object and its mesh."""

    mesh = obj.data
    if obj.mode == "EDIT":
        bmesh.update_edit_mesh(mesh, loop_triangles=False, destructive=False)
        bpy.ops.object.mode_set(mode="OBJECT")
    bpy.data.objects.remove(obj, do_unlink=True)
    if mesh.users == 0:
        bpy.data.meshes.remove(mesh)


def _prepare_topology(obj, bm) -> None:
    core.prepare_topology(bm, 0, TOLERANCE, mark_vertex_ids=True, mesh_object=obj)


def _mesh_counts(bm) -> str:
    return f"verts={len(bm.verts)} edges={len(bm.edges)} faces={len(bm.faces)}"


def _timed_prepare() -> float:
    obj, bm = build_suzanne()
    try:
        started = time.perf_counter_ns()
        _prepare_topology(obj, bm)
        return (time.perf_counter_ns() - started) / 1.0e6
    finally:
        destroy_suzanne(obj)


def main() -> None:
    print("YSE_SUZANNE_PERF_START", flush=True)
    print(
        f"blender_version={bpy.app.version_string} blender_build={bpy.app.build_hash}",
        flush=True,
    )
    print(
        f"fixture=suzanne operation=single_subdivide_edges cuts={SUBDIVIDE_CUTS} repeats={REPEATS}",
        flush=True,
    )

    fixture_obj, fixture_bm = build_suzanne()
    try:
        print(f"mesh_counts {_mesh_counts(fixture_bm)}", flush=True)
    finally:
        destroy_suzanne(fixture_obj)

    # Prime Blender/Python allocation and import paths, but do not include it
    # in the measured samples.
    warmup_obj, warmup_bm = build_suzanne()
    try:
        _prepare_topology(warmup_obj, warmup_bm)
    finally:
        destroy_suzanne(warmup_obj)
    print("warmup=excluded", flush=True)

    samples = [_timed_prepare() for _ in range(REPEATS)]
    formatted = ", ".join(f"{sample:.3f}" for sample in samples)
    print(f"prepare_topology samples_ms=[{formatted}]", flush=True)
    print(f"prepare_topology median_ms={statistics.median(samples):.3f}", flush=True)
    print("YSE_SUZANNE_PERF_DONE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
