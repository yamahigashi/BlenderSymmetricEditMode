# SPDX-License-Identifier: GPL-3.0-or-later

"""Headless unit checks for delete/dissolve leading-domain expansion."""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import bmesh
import bpy

PACKAGE_PARENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_PARENT))

from ydd_symmetric_edit import delete_dissolve  # noqa: E402

_AXIS = 0
_TOL = 1.0e-5


def _build_symmetric_grid() -> bmesh.types.BMesh:
    """2x2 quads mirrored across X=0.

    Vertex layout (index by creation order):
      0=(-1,-1,0)  1=(0,-1,0)  2=(1,-1,0)
      3=(-1, 1,0)  4=(0, 1,0)  5=(1, 1,0)

    Faces: left (0,1,4,3), right (1,2,5,4).
    """

    bm = bmesh.new()
    coords = (
        (-1.0, -1.0, 0.0),
        (0.0, -1.0, 0.0),
        (1.0, -1.0, 0.0),
        (-1.0, 1.0, 0.0),
        (0.0, 1.0, 0.0),
        (1.0, 1.0, 0.0),
    )
    verts = [bm.verts.new(co) for co in coords]
    bm.faces.new((verts[0], verts[1], verts[4], verts[3]))
    bm.faces.new((verts[1], verts[2], verts[5], verts[4]))
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    bm.verts.index_update()
    bm.edges.index_update()
    bm.faces.index_update()
    return bm


def _clear_select(bm: bmesh.types.BMesh) -> None:
    for vertex in bm.verts:
        vertex.select = False
    for edge in bm.edges:
        edge.select = False
    for face in bm.faces:
        face.select = False


def _edge_between(bm: bmesh.types.BMesh, a: int, b: int) -> bmesh.types.BMEdge:
    key = frozenset((a, b))
    for edge in bm.edges:
        if frozenset((edge.verts[0].index, edge.verts[1].index)) == key:
            return edge
    raise AssertionError(f"edge {a}-{b} not found")


def check_symmetric_grid_domains() -> None:
    """(a) VERT / EDGE / FACE one-sided selection expands to the mirror."""

    bm = _build_symmetric_grid()
    try:
        maps = delete_dissolve.build_element_pair_maps(bm, _AXIS, _TOL)

        assert maps.vert_pairs[0] == 2
        assert maps.vert_pairs[2] == 0
        assert maps.vert_pairs[1] == 1
        assert maps.vert_pairs[4] == 4

        # VERT: select left-bottom only → add right-bottom.
        _clear_select(bm)
        bm.verts[0].select = True
        plan = delete_dissolve.plan_leading_domain_expansion(bm, maps, domains=("VERT",))
        assert set(plan.add_vert_indices) == {2}
        assert plan.add_edge_indices == ()
        assert plan.add_face_indices == ()
        assert plan.unmatched_count == 0
        assert plan.hidden_counterpart_count == 0

        # EDGE: select left vertical edge 0-3 → add right vertical edge 2-5.
        _clear_select(bm)
        left_edge = _edge_between(bm, 0, 3)
        right_edge = _edge_between(bm, 2, 5)
        left_edge.select = True
        plan = delete_dissolve.plan_leading_domain_expansion(bm, maps, domains=("EDGE",))
        assert set(plan.add_edge_indices) == {right_edge.index}
        assert plan.unmatched_count == 0

        # FACE: select left face → add right face.
        _clear_select(bm)
        bm.faces[0].select = True
        plan = delete_dissolve.plan_leading_domain_expansion(bm, maps, domains=("FACE",))
        assert set(plan.add_face_indices) == {1}
        assert maps.face_pair_by_index[0] == 1
        assert maps.face_pair_by_index[1] == 0
        assert plan.unmatched_count == 0
    finally:
        bm.free()


def check_on_plane_self_pairs() -> None:
    """(b) On-plane verts and straddling edges are self-paired (no add)."""

    bm = _build_symmetric_grid()
    try:
        maps = delete_dissolve.build_element_pair_maps(bm, _AXIS, _TOL)

        assert maps.vert_pairs[1] == 1
        assert maps.vert_pairs[4] == 4
        _clear_select(bm)
        bm.verts[1].select = True
        plan = delete_dissolve.plan_leading_domain_expansion(bm, maps, domains=("VERT",))
        assert plan.add_vert_indices == ()
        assert plan.unmatched_count == 0

        center = _edge_between(bm, 1, 4)
        assert maps.edge_pair_by_index[center.index] == center.index
        _clear_select(bm)
        center.select = True
        plan = delete_dissolve.plan_leading_domain_expansion(bm, maps, domains=("EDGE",))
        assert plan.add_edge_indices == ()
        assert plan.unmatched_count == 0
    finally:
        bm.free()


def check_unmatched_asymmetric_region() -> None:
    """(c) Unpaired verts count as unmatched; paired verts still expand."""

    bm = bmesh.new()
    try:
        left = bm.verts.new((-1.0, 0.0, 0.0))
        right = bm.verts.new((1.0, 0.0, 0.0))
        outlier = bm.verts.new((-2.0, 1.0, 0.0))
        bm.edges.new((left, right))
        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        bm.verts.index_update()
        bm.edges.index_update()

        maps = delete_dissolve.build_element_pair_maps(bm, _AXIS, _TOL)
        assert maps.vert_pairs[left.index] == right.index
        assert maps.vert_pairs[right.index] == left.index
        assert outlier.index not in maps.vert_pairs

        _clear_select(bm)
        left.select = True
        outlier.select = True
        plan = delete_dissolve.plan_leading_domain_expansion(bm, maps, domains=("VERT",))
        assert set(plan.add_vert_indices) == {right.index}
        assert plan.unmatched_count == 1
        assert plan.hidden_counterpart_count == 0
    finally:
        bm.free()


def check_duplicate_face_keys() -> None:
    """(d) Coincident faces sharing a vertex set are both unmatched.

    ``bmesh.faces.new`` rejects a second face on the same verts; load a mesh
    datablock built with ``from_pydata`` so the collision case is reachable.
    """

    mesh = bpy.data.meshes.new("yse_delete_dup_faces")
    try:
        # Left quads at x=-2..-1 (two coincident), right mirrors at x=2..1.
        verts = (
            (-2.0, -1.0, 0.0),
            (-2.0, 1.0, 0.0),
            (-1.0, 1.0, 0.0),
            (-1.0, -1.0, 0.0),
            (2.0, -1.0, 0.0),
            (2.0, 1.0, 0.0),
            (1.0, 1.0, 0.0),
            (1.0, -1.0, 0.0),
        )
        faces = (
            (0, 1, 2, 3),
            (0, 1, 2, 3),
            (4, 5, 6, 7),
            (4, 5, 6, 7),
        )
        mesh.from_pydata(verts, [], faces)
        mesh.update()

        bm = bmesh.new()
        try:
            bm.from_mesh(mesh)
            bm.verts.ensure_lookup_table()
            bm.faces.ensure_lookup_table()
            bm.verts.index_update()
            bm.faces.index_update()
            assert len(bm.faces) == 4

            maps = delete_dissolve.build_element_pair_maps(bm, _AXIS, _TOL)
            for face in bm.faces:
                assert maps.face_pair_by_index[face.index] is None

            _clear_select(bm)
            bm.faces[0].select = True
            plan = delete_dissolve.plan_leading_domain_expansion(bm, maps, domains=("FACE",))
            assert plan.add_face_indices == ()
            assert plan.unmatched_count == 1
        finally:
            bm.free()
    finally:
        bpy.data.meshes.remove(mesh)


def check_hidden_counterpart() -> None:
    """(e) Hidden counterparts increment hidden_counterpart_count, not add."""

    bm = _build_symmetric_grid()
    try:
        maps = delete_dissolve.build_element_pair_maps(bm, _AXIS, _TOL)
        _clear_select(bm)
        bm.verts[0].select = True
        bm.verts[2].hide = True
        plan = delete_dissolve.plan_leading_domain_expansion(bm, maps, domains=("VERT",))
        assert plan.add_vert_indices == ()
        assert plan.hidden_counterpart_count == 1
        assert plan.unmatched_count == 0
    finally:
        bm.free()


def check_edge_face_composite_domain() -> None:
    """(f) EDGE+FACE domains expand both independently."""

    bm = _build_symmetric_grid()
    try:
        maps = delete_dissolve.build_element_pair_maps(bm, _AXIS, _TOL)
        _clear_select(bm)
        left_edge = _edge_between(bm, 0, 3)
        right_edge = _edge_between(bm, 2, 5)
        # face.select cascades to boundary edges/verts; strip those so EDGE
        # domain only sees the intentional left vertical edge.
        bm.faces[0].select = True
        for edge in bm.edges:
            edge.select = False
        for vertex in bm.verts:
            vertex.select = False
        left_edge.select = True
        assert bm.faces[0].select

        plan = delete_dissolve.plan_leading_domain_expansion(bm, maps, domains=("EDGE", "FACE"))
        assert set(plan.add_edge_indices) == {right_edge.index}
        assert set(plan.add_face_indices) == {1}
        assert plan.add_vert_indices == ()
        assert plan.unmatched_count == 0
    finally:
        bm.free()


def check_apply_expansion_plan_no_flush() -> None:
    """(g) apply sets select flags only; no flush side effects on other domains."""

    bm = _build_symmetric_grid()
    try:
        maps = delete_dissolve.build_element_pair_maps(bm, _AXIS, _TOL)
        _clear_select(bm)
        bm.verts[0].select = True

        edge_select_before = {edge.index: edge.select for edge in bm.edges}
        face_select_before = {face.index: face.select for face in bm.faces}

        plan = delete_dissolve.plan_leading_domain_expansion(bm, maps, domains=("VERT",))
        assert set(plan.add_vert_indices) == {2}
        delete_dissolve.apply_expansion_plan(bm, plan)

        assert bm.verts[0].select
        assert bm.verts[2].select
        for edge in bm.edges:
            assert edge.select == edge_select_before[edge.index]
        for face in bm.faces:
            assert face.select == face_select_before[face.index]
    finally:
        bm.free()


def check_already_selected_counterpart() -> None:
    """(h) Both sides already selected → zero additions."""

    bm = _build_symmetric_grid()
    try:
        maps = delete_dissolve.build_element_pair_maps(bm, _AXIS, _TOL)
        _clear_select(bm)
        bm.verts[0].select = True
        bm.verts[2].select = True
        plan = delete_dissolve.plan_leading_domain_expansion(bm, maps, domains=("VERT",))
        assert plan.add_vert_indices == ()
        assert plan.unmatched_count == 0
        assert plan.hidden_counterpart_count == 0

        _clear_select(bm)
        left_edge = _edge_between(bm, 0, 3)
        right_edge = _edge_between(bm, 2, 5)
        left_edge.select = True
        right_edge.select = True
        plan = delete_dissolve.plan_leading_domain_expansion(bm, maps, domains=("EDGE",))
        assert plan.add_edge_indices == ()

        _clear_select(bm)
        bm.faces[0].select = True
        bm.faces[1].select = True
        plan = delete_dissolve.plan_leading_domain_expansion(bm, maps, domains=("FACE",))
        assert plan.add_face_indices == ()
    finally:
        bm.free()


def run() -> None:
    check_symmetric_grid_domains()
    check_on_plane_self_pairs()
    check_unmatched_asymmetric_region()
    check_duplicate_face_keys()
    check_hidden_counterpart()
    check_edge_face_composite_domain()
    check_apply_expansion_plan_no_flush()
    check_already_selected_counterpart()
    print("YSE_DELETE_UNITS_OK", flush=True)


if __name__ == "__main__":
    try:
        run()
    except Exception:
        traceback.print_exc()
        print("YSE_DELETE_UNITS_FAILED", flush=True)
        sys.exit(1)
