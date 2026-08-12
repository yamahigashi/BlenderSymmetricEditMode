# SPDX-License-Identifier: GPL-3.0-or-later

"""Headless differential checks for the §C7-2 inlined collect predicate.

Contract: .agents/doc/perf_epoch_finish_plan7_2026-08-12.md, §C7-2.
The oracle is the untouched ``_is_path_edge_by_markers`` predicate applied to
every edge; it is compared against ``_discover_path_edges`` (both the
full-scan and ``selected_only`` forms), which §C7-2 replaces with an inlined
equivalent. Layers are always created before any geometry is built, since
adding a CustomData layer after verts/faces exist invalidates prior BMEdge/
BMFace references (ReferenceError).

Marker: YSE_COLLECT_PREDICATE_TEST_OK
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import bmesh

PACKAGE_PARENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_PARENT))

from ydd_symmetric_edit import layer_names, stitch_pathedges  # noqa: E402

MARKER = "YSE_COLLECT_PREDICATE_TEST_OK"


# ---------------------------------------------------------------------------
# Fixture / oracle helpers
# ---------------------------------------------------------------------------


def _new_bm(*, with_face_layer=True):
    """Create a BMesh with marker layers already present (before geometry)."""

    bm = bmesh.new()
    edge_layer = bm.edges.layers.int.new(layer_names.EDGE_ORIGINAL_LAYER)
    face_layer = bm.faces.layers.int.new(layer_names.FACE_ID_LAYER) if with_face_layer else None
    return bm, edge_layer, face_layer


def _oracle_full(bm, edge_layer, face_layer):
    return {edge for edge in bm.edges if stitch_pathedges._is_path_edge_by_markers(edge, edge_layer, face_layer)}


def _oracle_selected(bm, edge_layer, face_layer):
    return {
        edge for edge in bm.edges if edge.select and stitch_pathedges._is_path_edge_by_markers(edge, edge_layer, face_layer)
    }


def _assert_matches_oracle(bm, edge_layer, face_layer, label):
    expected_full = _oracle_full(bm, edge_layer, face_layer)
    expected_selected = _oracle_selected(bm, edge_layer, face_layer)
    actual_full = set(stitch_pathedges._discover_path_edges(bm, selected_only=False))
    actual_selected = set(stitch_pathedges._discover_path_edges(bm, selected_only=True))
    assert actual_full == expected_full, (label, "full", actual_full, expected_full)
    assert actual_selected == expected_selected, (label, "selected_only", actual_selected, expected_selected)


def _shared_edge(vert_a, vert_b):
    shared = set(vert_a.link_edges) & set(vert_b.link_edges)
    assert len(shared) == 1, shared
    return next(iter(shared))


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def check_two_face_match():
    """Manifold shared edge, tag != 0, both faces share one FACE_ID -> path edge."""

    bm, edge_layer, face_layer = _new_bm()
    v0 = bm.verts.new((0.0, 0.0, 0.0))
    v1 = bm.verts.new((1.0, 0.0, 0.0))
    v2 = bm.verts.new((0.0, 1.0, 0.0))
    v3 = bm.verts.new((0.0, -1.0, 0.0))
    face_a = bm.faces.new((v0, v1, v2))
    face_b = bm.faces.new((v1, v0, v3))
    edge = _shared_edge(v0, v1)
    edge[edge_layer] = 1
    face_a[face_layer] = 5
    face_b[face_layer] = 5
    for e in bm.edges:
        e.select = True

    _assert_matches_oracle(bm, edge_layer, face_layer, "two_face_match")
    assert edge in stitch_pathedges._discover_path_edges(bm, selected_only=False)
    bm.free()


def check_two_face_mismatch():
    """Manifold shared edge, tag != 0, FACE_ID differs -> not a path edge."""

    bm, edge_layer, face_layer = _new_bm()
    v0 = bm.verts.new((0.0, 0.0, 0.0))
    v1 = bm.verts.new((1.0, 0.0, 0.0))
    v2 = bm.verts.new((0.0, 1.0, 0.0))
    v3 = bm.verts.new((0.0, -1.0, 0.0))
    face_a = bm.faces.new((v0, v1, v2))
    face_b = bm.faces.new((v1, v0, v3))
    edge = _shared_edge(v0, v1)
    edge[edge_layer] = 1
    face_a[face_layer] = 5
    face_b[face_layer] = 7
    for e in bm.edges:
        e.select = True

    _assert_matches_oracle(bm, edge_layer, face_layer, "two_face_mismatch")
    assert edge not in stitch_pathedges._discover_path_edges(bm, selected_only=False)
    bm.free()


def check_three_face_fan_match():
    """Non-manifold 3-face fan edge, tag != 0, all FACE_IDs equal -> path edge."""

    bm, edge_layer, face_layer = _new_bm()
    v0 = bm.verts.new((0.0, 0.0, 0.0))
    v1 = bm.verts.new((1.0, 0.0, 0.0))
    v2 = bm.verts.new((0.0, 1.0, 0.0))
    v3 = bm.verts.new((0.0, -1.0, 0.0))
    v4 = bm.verts.new((0.0, 0.0, 1.0))
    face_a = bm.faces.new((v0, v1, v2))
    face_b = bm.faces.new((v0, v1, v3))
    face_c = bm.faces.new((v0, v1, v4))
    edge = _shared_edge(v0, v1)
    assert len(edge.link_faces) == 3, len(edge.link_faces)
    edge[edge_layer] = 1
    for face in (face_a, face_b, face_c):
        face[face_layer] = 9
    for e in bm.edges:
        e.select = True

    _assert_matches_oracle(bm, edge_layer, face_layer, "three_face_fan_match")
    assert edge in stitch_pathedges._discover_path_edges(bm, selected_only=False)
    bm.free()


def check_three_face_fan_mismatch():
    """Non-manifold 3-face fan edge, tag != 0, one FACE_ID differs -> not a path edge."""

    bm, edge_layer, face_layer = _new_bm()
    v0 = bm.verts.new((0.0, 0.0, 0.0))
    v1 = bm.verts.new((1.0, 0.0, 0.0))
    v2 = bm.verts.new((0.0, 1.0, 0.0))
    v3 = bm.verts.new((0.0, -1.0, 0.0))
    v4 = bm.verts.new((0.0, 0.0, 1.0))
    face_a = bm.faces.new((v0, v1, v2))
    face_b = bm.faces.new((v0, v1, v3))
    face_c = bm.faces.new((v0, v1, v4))
    edge = _shared_edge(v0, v1)
    assert len(edge.link_faces) == 3, len(edge.link_faces)
    edge[edge_layer] = 1
    face_a[face_layer] = 9
    face_b[face_layer] = 9
    face_c[face_layer] = 11
    for e in bm.edges:
        e.select = True

    _assert_matches_oracle(bm, edge_layer, face_layer, "three_face_fan_mismatch")
    assert edge not in stitch_pathedges._discover_path_edges(bm, selected_only=False)
    bm.free()


def check_wire_edge():
    """Edges with 0 or 1 link faces, tag != 0 -> never a path edge."""

    bm, edge_layer, face_layer = _new_bm()
    # 0 link faces (isolated edge).
    w0 = bm.verts.new((5.0, 0.0, 0.0))
    w1 = bm.verts.new((6.0, 0.0, 0.0))
    wire_edge = bm.edges.new((w0, w1))
    wire_edge[edge_layer] = 1

    # 1 link face (boundary edge of an unshared triangle).
    t0 = bm.verts.new((0.0, 0.0, 0.0))
    t1 = bm.verts.new((1.0, 0.0, 0.0))
    t2 = bm.verts.new((0.0, 1.0, 0.0))
    face = bm.faces.new((t0, t1, t2))
    face[face_layer] = 3
    boundary_edge = _shared_edge(t0, t1)
    boundary_edge[edge_layer] = 1

    for e in bm.edges:
        e.select = True

    _assert_matches_oracle(bm, edge_layer, face_layer, "wire_edge")
    result = set(stitch_pathedges._discover_path_edges(bm, selected_only=False))
    assert wire_edge not in result
    assert boundary_edge not in result
    bm.free()


def check_missing_face_layer():
    """No FACE_ID layer at all: tag==0 is still path; tag!=0 is never path."""

    bm, edge_layer, face_layer = _new_bm(with_face_layer=False)
    assert face_layer is None

    v0 = bm.verts.new((0.0, 0.0, 0.0))
    v1 = bm.verts.new((1.0, 0.0, 0.0))
    v2 = bm.verts.new((0.0, 1.0, 0.0))
    v3 = bm.verts.new((0.0, -1.0, 0.0))
    bm.faces.new((v0, v1, v2))
    bm.faces.new((v1, v0, v3))
    tag_zero_edge = _shared_edge(v0, v1)
    tag_zero_edge[edge_layer] = 0

    o0 = bm.verts.new((10.0, 0.0, 0.0))
    o1 = bm.verts.new((11.0, 0.0, 0.0))
    o2 = bm.verts.new((10.0, 1.0, 0.0))
    o3 = bm.verts.new((10.0, -1.0, 0.0))
    bm.faces.new((o0, o1, o2))
    bm.faces.new((o1, o0, o3))
    tag_nonzero_edge = _shared_edge(o0, o1)
    tag_nonzero_edge[edge_layer] = 1

    for e in bm.edges:
        e.select = True

    _assert_matches_oracle(bm, edge_layer, face_layer, "missing_face_layer")
    result = set(stitch_pathedges._discover_path_edges(bm, selected_only=False))
    assert tag_zero_edge in result
    assert tag_nonzero_edge not in result
    bm.free()


def check_tag_zero_and_nonzero_mixed():
    """A single mesh mixing tag==0 edges with tag!=0 matching/mismatching edges."""

    bm, edge_layer, face_layer = _new_bm()

    # tag == 0 -> always path, regardless of face state.
    z0 = bm.verts.new((0.0, 0.0, 0.0))
    z1 = bm.verts.new((1.0, 0.0, 0.0))
    z2 = bm.verts.new((0.0, 1.0, 0.0))
    face_z = bm.faces.new((z0, z1, z2))
    face_z[face_layer] = 1
    tag_zero_edge = _shared_edge(z0, z1)
    tag_zero_edge[edge_layer] = 0

    # tag != 0, matching FACE_ID -> path.
    m0 = bm.verts.new((10.0, 0.0, 0.0))
    m1 = bm.verts.new((11.0, 0.0, 0.0))
    m2 = bm.verts.new((10.0, 1.0, 0.0))
    m3 = bm.verts.new((10.0, -1.0, 0.0))
    face_m1 = bm.faces.new((m0, m1, m2))
    face_m2 = bm.faces.new((m1, m0, m3))
    face_m1[face_layer] = 2
    face_m2[face_layer] = 2
    tag_match_edge = _shared_edge(m0, m1)
    tag_match_edge[edge_layer] = 7

    # tag != 0, mismatching FACE_ID -> not path.
    x0 = bm.verts.new((20.0, 0.0, 0.0))
    x1 = bm.verts.new((21.0, 0.0, 0.0))
    x2 = bm.verts.new((20.0, 1.0, 0.0))
    x3 = bm.verts.new((20.0, -1.0, 0.0))
    face_x1 = bm.faces.new((x0, x1, x2))
    face_x2 = bm.faces.new((x1, x0, x3))
    face_x1[face_layer] = 3
    face_x2[face_layer] = 4
    tag_mismatch_edge = _shared_edge(x0, x1)
    tag_mismatch_edge[edge_layer] = 9

    for e in bm.edges:
        e.select = True

    _assert_matches_oracle(bm, edge_layer, face_layer, "tag_zero_and_nonzero_mixed")
    result = set(stitch_pathedges._discover_path_edges(bm, selected_only=False))
    assert tag_zero_edge in result
    assert tag_match_edge in result
    assert tag_mismatch_edge not in result
    bm.free()


def check_selected_only_excludes_unselected_tag_zero():
    """An unselected tag==0 edge is in the full form but not selected_only."""

    bm, edge_layer, face_layer = _new_bm()

    s0 = bm.verts.new((0.0, 0.0, 0.0))
    s1 = bm.verts.new((1.0, 0.0, 0.0))
    s2 = bm.verts.new((0.0, 1.0, 0.0))
    face_s = bm.faces.new((s0, s1, s2))
    face_s[face_layer] = 1
    selected_tag_zero_edge = _shared_edge(s0, s1)
    selected_tag_zero_edge[edge_layer] = 0

    u0 = bm.verts.new((10.0, 0.0, 0.0))
    u1 = bm.verts.new((11.0, 0.0, 0.0))
    u2 = bm.verts.new((10.0, 1.0, 0.0))
    face_u = bm.faces.new((u0, u1, u2))
    face_u[face_layer] = 1
    unselected_tag_zero_edge = _shared_edge(u0, u1)
    unselected_tag_zero_edge[edge_layer] = 0

    for e in bm.edges:
        e.select = False
    selected_tag_zero_edge.select = True

    _assert_matches_oracle(bm, edge_layer, face_layer, "selected_only_excludes_unselected_tag_zero")

    full_result = set(stitch_pathedges._discover_path_edges(bm, selected_only=False))
    selected_result = set(stitch_pathedges._discover_path_edges(bm, selected_only=True))
    assert selected_tag_zero_edge in full_result
    assert unselected_tag_zero_edge in full_result
    assert selected_tag_zero_edge in selected_result
    assert unselected_tag_zero_edge not in selected_result
    bm.free()


def check_missing_edge_layer_returns_empty():
    """No EDGE_ORIGINAL_LAYER at all -> both forms return an empty list."""

    bm = bmesh.new()
    v0 = bm.verts.new((0.0, 0.0, 0.0))
    v1 = bm.verts.new((1.0, 0.0, 0.0))
    v2 = bm.verts.new((0.0, 1.0, 0.0))
    bm.faces.new((v0, v1, v2))
    for e in bm.edges:
        e.select = True

    assert stitch_pathedges._discover_path_edges(bm, selected_only=False) == []
    assert stitch_pathedges._discover_path_edges(bm, selected_only=True) == []
    bm.free()


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run() -> None:
    check_two_face_match()
    check_two_face_mismatch()
    check_three_face_fan_match()
    check_three_face_fan_mismatch()
    check_wire_edge()
    check_missing_face_layer()
    check_tag_zero_and_nonzero_mixed()
    check_selected_only_excludes_unselected_tag_zero()
    check_missing_edge_layer_returns_empty()
    print(MARKER, flush=True)


if __name__ == "__main__":
    try:
        run()
    except Exception:
        traceback.print_exc()
        print("YSE_COLLECT_PREDICATE_TEST_FAILED", flush=True)
        sys.exit(1)
