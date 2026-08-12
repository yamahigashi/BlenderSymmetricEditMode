# SPDX-License-Identifier: GPL-3.0-or-later

"""Differential checks for the §C7-3 crossings vertex-bin index."""

from __future__ import annotations

import math
import sys
import traceback
import types
from pathlib import Path

import bmesh
import numpy

PACKAGE_PARENT = Path(__file__).resolve().parents[1]
TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE_PARENT))
sys.path.insert(0, str(TESTS_DIR))

import test_crossings_vertex_index as legacy_index_test  # noqa: E402

from ydd_symmetric_edit import stitch_crossings  # noqa: E402

TOLERANCE = 1.0e-5
MARKER = "YSE_CROSSINGS_NUMPY_TEST_OK"


def _fixture():
    """Build the small indexed fixture with layers before any geometry."""

    bm = bmesh.new()
    # CustomData layers must precede all geometry creation (Blender 4.2/5.2).
    bm.verts.layers.int.new("yse_vertex_marker")
    bm.edges.layers.int.new("yse_edge_marker")
    bm.faces.layers.int.new("yse_face_marker")
    coordinates = (
        (0.10, 0.10, 0.10),
        (0.20, 0.10, 0.10),
        (1.10, 0.10, 0.10),
        (-0.10, 0.10, 0.10),
        (0.10, 1.10, 0.10),
        (0.10, -0.10, 0.10),
        (0.10, 0.10, 1.10),
        (0.10, 0.10, -0.10),
    )
    vertices = [bm.verts.new(coordinate) for coordinate in coordinates]
    for first, second in zip(vertices, vertices[1:], strict=False):
        bm.edges.new((first, second))
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    return bm


def _index_signature(index):
    return {key: tuple(id(vertex) for vertex in vertices) for key, vertices in index.items()}


def check_numpy_matches_python():
    bm, _plan = legacy_index_test._build_two_cluster_finite_plan()
    try:
        expected, expected_fallback = stitch_crossings._build_crossings_vertex_bin_index_python(bm, TOLERANCE)
        actual, actual_fallback = stitch_crossings._build_crossings_vertex_bin_index(bm, TOLERANCE)
        assert not expected_fallback and not actual_fallback
        assert _index_signature(actual) == _index_signature(expected)
        assert all(isinstance(key, tuple) and len(key) == 3 for key in actual)
    finally:
        bm.free()


def check_nonfinite_coordinate_returns_fallback():
    bm = _fixture()
    try:
        bm.verts.new((math.nan, 0.0, 0.0))
        assert stitch_crossings._build_crossings_vertex_bin_index(bm, TOLERANCE) == (None, True)
    finally:
        bm.free()


def check_int64_guard_matches_python():
    bm = _fixture()
    original_python = stitch_crossings._build_crossings_vertex_bin_index_python
    calls = {"count": 0}

    def observed_python(*args, **kwargs):
        calls["count"] += 1
        return original_python(*args, **kwargs)

    stitch_crossings._build_crossings_vertex_bin_index_python = observed_python
    try:
        bm.verts[0].co.x = 1.0e30
        expected = original_python(bm, 1.0e-6)
        actual = stitch_crossings._build_crossings_vertex_bin_index(bm, 1.0e-6)
        assert _index_signature(actual[0]) == _index_signature(expected[0])
        assert actual[1] is False
        assert calls["count"] == 1
    finally:
        stitch_crossings._build_crossings_vertex_bin_index_python = original_python
        bm.free()


def check_order_guard_detects_reordered_keys():
    bm = _fixture()
    original_floor = numpy.floor
    original_python = stitch_crossings._build_crossings_vertex_bin_index_python
    calls = {"count": 0}

    def observed_python(*args, **kwargs):
        calls["count"] += 1
        return original_python(*args, **kwargs)

    stitch_crossings._build_crossings_vertex_bin_index_python = observed_python
    try:

        def swapped_floor(values, *args, **kwargs):
            result = original_floor(values, *args, **kwargs)
            if getattr(result, "ndim", 0) == 2 and len(result) >= 2:
                result = result.copy()
                result[[0, 1]] = result[[1, 0]]
            return result

        numpy.floor = swapped_floor
        expected = original_python(bm, TOLERANCE)
        actual = stitch_crossings._build_crossings_vertex_bin_index(bm, TOLERANCE)
        assert _index_signature(actual[0]) == _index_signature(expected[0])
        assert actual[1] is False
        assert calls["count"] == 1
    finally:
        stitch_crossings._build_crossings_vertex_bin_index_python = original_python
        numpy.floor = original_floor
        bm.free()


def check_count_mismatch_falls_back():
    original = stitch_crossings._build_crossings_vertex_bin_index_python
    calls = {"count": 0}

    def observed(*args, **kwargs):
        calls["count"] += 1
        return {}, False

    stitch_crossings._build_crossings_vertex_bin_index_python = observed
    previous_bpy = sys.modules.get("bpy")

    class FakeBMesh:
        verts = (object(),)

        def to_mesh(self, _mesh):
            return None

    class FakeMesh:
        vertices = ()

    fake_bpy = types.SimpleNamespace(
        data=types.SimpleNamespace(
            meshes=types.SimpleNamespace(
                new=lambda _name: FakeMesh(),
                remove=lambda _mesh: None,
            )
        )
    )
    sys.modules["bpy"] = fake_bpy
    try:
        actual = stitch_crossings._build_crossings_vertex_bin_index(FakeBMesh(), TOLERANCE)
        assert actual == ({}, False)
        assert calls["count"] == 1
    finally:
        stitch_crossings._build_crossings_vertex_bin_index_python = original
        if previous_bpy is None:
            sys.modules.pop("bpy", None)
        else:
            sys.modules["bpy"] = previous_bpy


def check_numpy_exception_falls_back():
    bm = _fixture()
    original = stitch_crossings._build_crossings_vertex_bin_index_python
    previous_bpy = sys.modules.get("bpy")
    calls = {"count": 0}

    def observed(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    class FailingMesh:
        class Vertices:
            def __len__(self):
                return len(bm.verts)

            def foreach_get(self, _attribute, _buffer):
                raise RuntimeError("synthetic foreach_get failure")

        vertices = Vertices()

    class ProxyBMesh:
        verts = tuple(bm.verts)

        def to_mesh(self, _mesh):
            return None

    removed = {"count": 0}
    fake_bpy = types.SimpleNamespace(
        data=types.SimpleNamespace(
            meshes=types.SimpleNamespace(
                new=lambda _name: FailingMesh(),
                remove=lambda _mesh: removed.__setitem__("count", removed["count"] + 1),
            )
        )
    )
    stitch_crossings._build_crossings_vertex_bin_index_python = observed
    sys.modules["bpy"] = fake_bpy
    try:
        proxy = ProxyBMesh()
        expected = original(proxy, TOLERANCE)
        actual = stitch_crossings._build_crossings_vertex_bin_index(proxy, TOLERANCE)
        assert _index_signature(actual[0]) == _index_signature(expected[0])
        assert calls["count"] == 1
        assert removed["count"] == 1
    finally:
        stitch_crossings._build_crossings_vertex_bin_index_python = original
        if previous_bpy is None:
            sys.modules.pop("bpy", None)
        else:
            sys.modules["bpy"] = previous_bpy
        bm.free()


def run():
    checks = (
        check_numpy_matches_python,
        check_nonfinite_coordinate_returns_fallback,
        check_int64_guard_matches_python,
        check_order_guard_detects_reordered_keys,
        check_count_mismatch_falls_back,
        check_numpy_exception_falls_back,
    )
    for check in checks:
        check()
        print(f"PASS {check.__name__}", flush=True)
    print(MARKER, flush=True)


if __name__ == "__main__":
    try:
        run()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
