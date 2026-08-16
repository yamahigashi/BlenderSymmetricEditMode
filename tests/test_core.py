# SPDX-License-Identifier: GPL-3.0-or-later

"""Headless Blender checks for topology marking and cutter reflection."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import bmesh
from mathutils import Vector

PACKAGE_PARENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_PARENT))

from ydd_symmetric_edit import (  # noqa: E402
    extrude,
    keymaps,
    layer_names,
    matching,
    selection,
    snapshot,
    stitch_common,
    stitch_offset,
    stitch_pathedges,
    stitch_reflect,
)
from ydd_symmetric_edit._types import ExtrudeSnapshot, MeshSelectionMode  # noqa: E402


def build_two_symmetric_quads():
    bm = bmesh.new()
    left = [
        bm.verts.new(co)
        for co in (
            (-2.0, -1.0, 0.0),
            (-1.0, -1.0, 0.0),
            (-1.0, 1.0, 0.0),
            (-2.0, 1.0, 0.0),
        )
    ]
    right = [bm.verts.new(co) for co in ((1.0, -1.0, 0.0), (2.0, -1.0, 0.0), (2.0, 1.0, 0.0), (1.0, 1.0, 0.0))]
    bm.faces.new(left)
    bm.faces.new(right)
    return bm


def split_left_face_like_native_knife(bm):
    bottom = next(
        edge
        for edge in bm.edges
        if all(vertex.co.x < 0.0 for vertex in edge.verts)
        and all(abs(vertex.co.y + 1.0) < 1.0e-8 for vertex in edge.verts)
    )
    _new_edge, bottom_vertex = bmesh.utils.edge_split(bottom, bottom.verts[0], 0.5)
    top = next(
        edge
        for edge in bm.edges
        if all(vertex.co.x < 0.0 for vertex in edge.verts)
        and all(abs(vertex.co.y - 1.0) < 1.0e-8 for vertex in edge.verts)
    )
    _new_edge, top_vertex = bmesh.utils.edge_split(top, top.verts[0], 0.5)
    source_face = next(face for face in bm.faces if bottom_vertex in face.verts and top_vertex in face.verts)
    bmesh.utils.face_split(source_face, bottom_vertex, top_vertex)
    return bm.edges.get((bottom_vertex, top_vertex))


def check_injective_component_counterexample():
    """The design's mutual-nearest counterexample must yield the complete
    assignment q1->t2, q2->t1, q3->t3 (1-D, tolerance 1.0)."""

    targets = [Vector((0.0, 0.0, 0.0)), Vector((1.2, 0.0, 0.0)), Vector((-1.5, 0.0, 0.0))]
    queries = [Vector((0.4, 0.0, 0.0)), Vector((-0.9, 0.0, 0.0)), Vector((-2.0, 0.0, 0.0))]
    lookup = matching.build_vertex_mirror_lookup(targets, 0, 1.0)
    assert lookup.find_all_direct(queries) == (1, 0, 2)


def check_injective_tie_rejection_is_order_independent():
    """Two complete assignments whose distance multisets are equal must be a
    tie even when sequential float addition would compare them unequal
    (distances [1, eps, eps] vs [eps, eps, 1])."""

    eps = 2.0**-53
    candidate_lists = [
        [(eps, 11), (1.0, 10)],
        [(eps, 11), (eps, 12)],
        [(eps, 12), (1.0, 10)],
    ]
    result = matching._solve_injective_component([0, 1, 2], candidate_lists)
    assert result is None, result

    # A strictly better unique optimum must still be found.
    unique_lists = [
        [(0.1, 10), (0.4, 11)],
        [(0.1, 11), (0.4, 10)],
    ]
    result = matching._solve_injective_component([0, 1], unique_lists)
    assert result == {0: 10, 1: 11}, result


def check_injective_step_limit_rejects_component():
    # Costs 0.4 vs 0.5: a unique optimum, no tie.
    candidate_lists = [
        [(0.1, 10), (0.3, 11)],
        [(0.2, 10), (0.3, 11)],
    ]
    # Exhausting the search takes 3 trial assignments; capping at 2 must
    # reject the whole component even though the optimum was already seen.
    assert matching._solve_injective_component([0, 1], candidate_lists, step_limit=2) is None
    assert matching._solve_injective_component([0, 1], candidate_lists) == {0: 10, 1: 11}


def check_on_plane_vertices_never_serve_off_plane_queries():
    """Partial-batch regression: an off-plane query whose reflection lands
    near an on-plane vertex must stay unresolved, even when no on-plane query
    is present in the batch to reserve that vertex."""

    tolerance = 0.001
    registered = [Vector((-0.0002, 0.0, 0.0)), Vector((5.0, 0.0, 0.0))]
    lookup = matching.build_vertex_mirror_lookup(registered, 0, tolerance)
    # Reflection of +0.0011 is -0.0011: Chebyshev 0.0009 from the on-plane
    # vertex, i.e. inside tolerance — but the plane partition must reject it.
    assert lookup.find_all_mirrored([Vector((0.0011, 0.0, 0.0))]) == (None,)
    # The on-plane vertex still self-corresponds when queried directly.
    assert lookup.find_all_mirrored([Vector((-0.0002, 0.0, 0.0))]) == (0,)


def check_pair_table_involution_and_partial_pairs():
    coords = [
        Vector((-1.0, 0.0, 0.0)),
        Vector((2.0, 0.0, 0.0)),
        Vector((1.0, 0.0, 0.0)),
        Vector((-2.0, 0.0, 0.0)),
        Vector((0.0, 3.0, 0.0)),
        Vector((7.0, 0.0, 0.0)),  # no counterpart
    ]
    pairs = matching.build_vertex_pair_table(coords, 0, 0.001)
    assert all(pairs[pairs[vertex]] == vertex for vertex in pairs)
    assert pairs.get(0) == 2 and pairs.get(2) == 0
    assert pairs.get(1) == 3 and pairs.get(3) == 1
    assert pairs.get(4) == 4  # on-plane self-pair
    assert 5 not in pairs


def check_quantized_coordinate_bin_boundary():
    """Contract E: pairs within tolerance must match across exclusive bin edges."""

    tolerance = 1.0e-5
    inverse = 1.0 / tolerance  # 1e5 scale referenced by the fix contract
    assert abs(inverse - 1.0e5) < 1.0e-9

    # mathutils.Vector is float32, so a pure 2e-10 gap collapses at unit scale.
    # Use ~1e-6 (still << tolerance, float32-stable) across a floor bin edge.
    boundary = 1.0
    delta = 1.0e-6
    left = Vector((boundary - delta, 0.0, 0.0))
    right = Vector((boundary + delta, 0.0, 0.0))
    assert abs(left.x - right.x) <= tolerance
    assert abs(left.x - right.x) < 5.0e-6

    primary_left = matching._quantized_coordinate(left, tolerance)
    primary_right = matching._quantized_coordinate(right, tolerance)
    assert primary_left != primary_right, (
        primary_left,
        primary_right,
        left.x * inverse,
        right.x * inverse,
    )
    neighborhood_left = set(matching._iter_quantized_neighborhood(left, tolerance))
    assert primary_right in neighborhood_left
    assert primary_left in set(matching._iter_quantized_neighborhood(right, tolerance))

    far = Vector((boundary + 2.0 * tolerance, 0.0, 0.0))
    assert primary_left not in set(matching._iter_quantized_neighborhood(far, tolerance))
    assert matching._quantized_coordinate(far, tolerance) not in neighborhood_left

    # Historical round() bug at half-bin boundaries (float64 sample; inverse=1e5).
    round_left_x = 0.5 * tolerance - 1.0e-10
    round_right_x = 0.5 * tolerance + 1.0e-10
    assert round(round_left_x * inverse) != round(round_right_x * inverse)
    round_left = Vector((round_left_x, 0.0, 0.0))
    round_right = Vector((round_right_x, 0.0, 0.0))
    assert matching._quantized_coordinate(round_left, tolerance) in set(
        matching._iter_quantized_neighborhood(round_right, tolerance)
    )

    # Symmetric quads whose mirrored verts sit on opposite sides of a bin edge
    # must still prepare a full face map (exact dict lookup alone would miss).
    bm = bmesh.new()
    try:
        inner = boundary
        outer = boundary + 1.0
        left_face = [
            bm.verts.new(co)
            for co in (
                (-outer, -1.0, 0.0),
                (-(inner + delta), -1.0, 0.0),
                (-(inner + delta), 1.0, 0.0),
                (-outer, 1.0, 0.0),
            )
        ]
        right_face = [
            bm.verts.new(co)
            for co in (
                (inner - delta, -1.0, 0.0),
                (outer, -1.0, 0.0),
                (outer, 1.0, 0.0),
                (inner - delta, 1.0, 0.0),
            )
        ]
        bm.faces.new(left_face)
        bm.faces.new(right_face)
        # Mirrored left inner x is +(inner+delta); right stores +(inner-delta).
        # Real gap is ~2e-6, but floor bins differ across the integer boundary.
        topology = snapshot.prepare_topology(bm, matching.AXIS_INDEX["X"], tolerance)
        assert topology.matched_faces == 2 and topology.total_faces == 2, (
            topology.matched_faces,
            topology.total_faces,
            topology.mirror_face_ids,
        )

        existing: dict = {}
        stitch_common._register_edge_endpoint_pair(
            existing,
            Vector((inner + delta, -1.0, 0.0)),
            Vector((inner + delta, 1.0, 0.0)),
            tolerance,
        )
        assert stitch_common._edge_coordinate_key_matches(
            Vector((inner - delta, -1.0, 0.0)),
            Vector((inner - delta, 1.0, 0.0)),
            tolerance,
            existing,
        )
    finally:
        bm.free()


def check_interior_edge_factor_uses_absolute_distance():
    """Contract F: long edges accept end-near splits by absolute distance."""

    edge_length = 1000.0
    tolerance = 1.0e-5
    # Old dimensionless check rejected factor 4e-6; absolute distance is 0.004.
    assert stitch_common._is_interior_edge_factor(4.0e-6, edge_length, tolerance)
    assert not stitch_common._is_interior_edge_factor(1.0e-9, edge_length, tolerance)
    assert not stitch_common._is_interior_edge_factor(0.5, 0.0, tolerance)


def check_choose_source_side_ignores_coordinate_tolerance():
    """Contract G: path length must not be compared to coordinate tolerance."""

    bm = bmesh.new()
    try:
        # Length 8e-4 < coordinate tolerance 1e-3, but still a real negative cut.
        a = bm.verts.new((-0.0020, 0.0, 0.0))
        b = bm.verts.new((-0.0028, 0.0, 0.0))
        edge = bm.edges.new((a, b))
        side, crossing = stitch_pathedges.choose_source_side([edge], matching.AXIS_INDEX["X"], 1.0e-3, "AUTO")
        assert side == "NEGATIVE" and crossing == 0
        assert edge.calc_length() < 1.0e-3
        assert edge.calc_length() > stitch_pathedges._MIN_SIDE_LENGTH
    finally:
        bm.free()


def check_collapsed_offset_prefers_geometrically_closer_marker():
    """R2-1: ambiguous-band competitors pick the nearer original marker."""

    tolerance = 1.0e-3
    bm = bmesh.new()
    try:
        marker_layer = bm.edges.layers.int.new(layer_names.EDGE_ORIGINAL_LAYER)

        # Far competitor first (~1.5×tol): old bin-only markers[0] would adopt it.
        far_shift = 1.5 * tolerance
        far_a = bm.verts.new((1.0 + far_shift, 0.0, 0.0))
        far_b = bm.verts.new((1.0 + far_shift, 1.0, 0.0))
        far_edge = bm.edges.new((far_a, far_b))
        far_edge[marker_layer] = 20

        # Near original (~0.25×tol): geometrically valid and closer.
        near_shift = 0.25 * tolerance
        near_a = bm.verts.new((1.0 + near_shift, 0.0, 0.0))
        near_b = bm.verts.new((1.0 + near_shift, 1.0, 0.0))
        near_edge = bm.edges.new((near_a, near_b))
        near_edge[marker_layer] = 10

        # Also keep a second in-tolerance competitor to exercise min-max selection.
        mid_shift = 0.8 * tolerance
        mid_a = bm.verts.new((1.0 + mid_shift, 0.0, 0.0))
        mid_b = bm.verts.new((1.0 + mid_shift, 1.0, 0.0))
        mid_edge = bm.edges.new((mid_a, mid_b))
        mid_edge[marker_layer] = 30

        source_a = bm.verts.new((-1.0, 0.0, 0.0))
        source_b = bm.verts.new((-1.0, 1.0, 0.0))
        source_edge = bm.edges.new((source_a, source_b))
        source_edge[marker_layer] = 0

        markers, reason = stitch_offset.collapsed_offset_target_edge_markers(
            bm,
            [source_edge],
            matching.AXIS_INDEX["X"],
            tolerance,
        )
        assert reason == "", reason
        assert markers == {10}, markers
    finally:
        bm.free()


def check_reflected_path_lazy_existing_edge_store():
    """All identity hits must avoid building the geometric edge store."""

    bm = build_two_symmetric_quads()
    try:
        topology = snapshot.prepare_topology(bm, matching.AXIS_INDEX["X"], 1.0e-5)
        path_edge = split_left_face_like_native_knife(bm)
        marker = bm.edges.layers.int.get(layer_names.EDGE_ORIGINAL_LAYER)
        assert marker is not None
        path_edge[marker] = 0
        first = stitch_reflect.apply_reflected_path_topology(
            bm,
            [path_edge],
            matching.AXIS_INDEX["X"],
            1.0e-5,
            topology.mirror_face_ids,
        )
        assert first == (1, 0, "")

        registrations = 0
        original_register = stitch_common._register_edge_endpoint_pair

        def count_registration(*args, **kwargs):
            nonlocal registrations
            registrations += 1
            return original_register(*args, **kwargs)

        stitch_common._register_edge_endpoint_pair = count_registration
        try:
            second = stitch_reflect.apply_reflected_path_topology(
                bm,
                [path_edge],
                matching.AXIS_INDEX["X"],
                1.0e-5,
                topology.mirror_face_ids,
            )
        finally:
            stitch_common._register_edge_endpoint_pair = original_register
        assert second == (0, 1, "")
        assert registrations == 0
    finally:
        bm.free()


def check_reflected_path_lazy_store_falls_back_after_identity_miss():
    """A later geometric miss builds the store scoped to target-face edges."""

    bm = build_two_symmetric_quads()
    try:
        topology = snapshot.prepare_topology(bm, matching.AXIS_INDEX["X"], 1.0e-5)

        def add_vertical_cut(x):
            bottom = next(
                edge
                for edge in bm.edges
                if all(abs(vertex.co.y + 1.0) < 1.0e-8 for vertex in edge.verts)
                and min(vertex.co.x for vertex in edge.verts) < x < max(vertex.co.x for vertex in edge.verts)
            )
            _edge, bottom_vertex = bmesh.utils.edge_split(
                bottom,
                bottom.verts[0],
                (x - bottom.verts[0].co.x) / (bottom.verts[1].co.x - bottom.verts[0].co.x),
            )
            top = next(
                edge
                for edge in bm.edges
                if all(abs(vertex.co.y - 1.0) < 1.0e-8 for vertex in edge.verts)
                and min(vertex.co.x for vertex in edge.verts) < x < max(vertex.co.x for vertex in edge.verts)
            )
            _edge, top_vertex = bmesh.utils.edge_split(
                top,
                top.verts[0],
                (x - top.verts[0].co.x) / (top.verts[1].co.x - top.verts[0].co.x),
            )
            face = next(face for face in bm.faces if bottom_vertex in face.verts and top_vertex in face.verts)
            bmesh.utils.face_split(face, bottom_vertex, top_vertex)
            return bm.edges.get((bottom_vertex, top_vertex))

        first = add_vertical_cut(-1.6)
        second = add_vertical_cut(-1.3)
        marker = bm.edges.layers.int.get(layer_names.EDGE_ORIGINAL_LAYER)
        assert marker is not None
        first[marker] = 0
        second[marker] = 0

        initial = stitch_reflect.apply_reflected_path_topology(
            bm,
            [first],
            matching.AXIS_INDEX["X"],
            1.0e-5,
            topology.mirror_face_ids,
        )
        assert initial == (1, 0, "")

        # Materialize the later target endpoints without creating their edge.
        # This keeps the mixed call's lazy-store boundary free of endpoint
        # splits, making the pre-call BMesh snapshot its exact scoped oracle.
        target_vertices = []
        for y in (-1.0, 1.0):
            boundary = next(
                edge
                for edge in bm.edges
                if all(abs(vertex.co.y - y) < 1.0e-8 for vertex in edge.verts)
                and min(vertex.co.x for vertex in edge.verts) < 1.3 < max(vertex.co.x for vertex in edge.verts)
                and all(vertex.co.x > 0.0 for vertex in edge.verts)
            )
            _new_edge, target_vertex = bmesh.utils.edge_split(
                boundary,
                boundary.verts[0],
                (1.3 - boundary.verts[0].co.x) / (boundary.verts[1].co.x - boundary.verts[0].co.x),
            )
            target_vertices.append(target_vertex)

        face_layer = bm.faces.layers.int.get(layer_names.FACE_ID_LAYER)
        assert face_layer is not None
        # The store only registers edges linked to a face in the pending
        # records' target-id union (the positive quad); registration order
        # is non-normative.
        scoped_ids = {int(face[face_layer]) for face in bm.faces if all(vertex.co.x > 0.0 for vertex in face.verts)}
        expected_bulk = [
            (
                tuple(float(component) for component in edge.verts[0].co),
                tuple(float(component) for component in edge.verts[1].co),
                frozenset(int(face[face_layer]) for face in edge.link_faces),
            )
            for edge in bm.edges
            if edge.is_valid and any(int(face[face_layer]) in scoped_ids for face in edge.link_faces)
        ]

        registrations = []
        original_register = stitch_common._register_edge_endpoint_pair

        def count_registration(*args, **kwargs):
            registrations.append(
                (
                    tuple(float(component) for component in args[1]),
                    tuple(float(component) for component in args[2]),
                    frozenset(int(face_id) for face_id in kwargs["face_ids"]),
                )
            )
            return original_register(*args, **kwargs)

        stitch_common._register_edge_endpoint_pair = count_registration
        try:
            mixed = stitch_reflect.apply_reflected_path_topology(
                bm,
                [first, second],
                matching.AXIS_INDEX["X"],
                1.0e-5,
                topology.mirror_face_ids,
            )
        finally:
            stitch_common._register_edge_endpoint_pair = original_register
        assert mixed == (1, 1, "")
        assert len(registrations) == len(expected_bulk) + 1
        assert set(registrations[: len(expected_bulk)]) == set(expected_bulk)
        created_edge = bm.edges.get(target_vertices)
        assert created_edge is not None
        assert {registrations[-1][0], registrations[-1][1]} == {
            tuple(float(component) for component in vertex.co) for vertex in created_edge.verts
        }
        assert registrations[-1][2] == frozenset(int(face[face_layer]) for face in created_edge.link_faces)
    finally:
        bm.free()


def check_large_ngon_coords_match_without_recursion_error():
    """R2-2: large n-gon multiset match must not raise RecursionError."""

    vertex_count = 1100
    assert vertex_count > sys.getrecursionlimit() // 2
    tolerance = 1.0e-4
    first = tuple(
        (
            math.cos(2.0 * math.pi * index / vertex_count),
            math.sin(2.0 * math.pi * index / vertex_count),
            0.0,
        )
        for index in range(vertex_count)
    )
    # Same multiset, rotated and slightly perturbed within tolerance.
    second = tuple(
        (
            first[(index + 17) % vertex_count][0] + 1.0e-5,
            first[(index + 17) % vertex_count][1] - 1.0e-5,
            first[(index + 17) % vertex_count][2],
        )
        for index in range(vertex_count)
    )
    assert matching._coords_match_chebyshev(first, second, tolerance)
    broken = second[:-1] + ((100.0, 100.0, 100.0),)
    assert not matching._coords_match_chebyshev(first, broken, tolerance)


def check_vertex_mirror_lookup_bin_boundary():
    """Phase 2a: find hits across exclusive floor-bin edges within tolerance."""

    tolerance = 1.0e-5
    # float32-stable gap across the integer floor boundary at 1.0.
    boundary = 1.0
    delta = 1.0e-6
    registered = Vector((boundary + delta, 0.25, -0.5))
    query = Vector((-(boundary - delta), 0.25, -0.5))
    # Real mirror gap is ~2e-6 << tolerance; primary bins differ.
    assert abs(registered.x - (-query.x)) <= tolerance
    primary_reg = matching._quantized_coordinate(registered, tolerance)
    primary_mirror = matching._quantized_coordinate(matching.mirror_coordinate(query, 0), tolerance)
    assert primary_reg != primary_mirror

    lookup = matching.build_vertex_mirror_lookup([registered], matching.AXIS_INDEX["X"], tolerance)
    assert lookup.find(query) == 0

    # Invariant: any axis gap ≥ 2·tolerance must never hit.
    far_query = Vector((-(boundary + 2.0 * tolerance), 0.25, -0.5))
    assert lookup.find(far_query) is None


def check_vertex_mirror_lookup_is_on_plane_boundary():
    """Phase 2a: is_on_plane is true on |axis| ≤ tolerance, false outside."""

    tolerance = 1.0e-3
    lookup = matching.build_vertex_mirror_lookup([], matching.AXIS_INDEX["Y"], tolerance)
    assert lookup.is_on_plane(Vector((1.0, 0.0, 0.0)))
    # mathutils.Vector is float32; assign through the component so the stored
    # value is what is_on_plane actually sees (1e-3 is not binary-exact).
    for sign in (1.0, -1.0):
        on = Vector((0.0, 0.0, 0.0))
        on.y = sign * tolerance
        if abs(float(on.y)) > tolerance:
            on.y = sign * tolerance * (1.0 - 1.0e-6)
        assert abs(float(on.y)) <= tolerance
        assert lookup.is_on_plane(on)

        off = Vector((0.0, 0.0, 0.0))
        off.y = sign * (tolerance + 1.0e-6)
        assert abs(float(off.y)) > tolerance
        assert not lookup.is_on_plane(off)


def check_vertex_mirror_lookup_asymmetric_returns_none():
    """Phase 2a: coordinates without a mirror counterpart yield None."""

    tolerance = 1.0e-4
    # Only the negative half is registered; positive-only query has no mirror.
    coords = [
        Vector((-2.0, 0.0, 0.0)),
        Vector((-1.0, 1.0, 0.0)),
        Vector((-1.5, -0.5, 0.5)),
    ]
    lookup = matching.build_vertex_mirror_lookup(coords, matching.AXIS_INDEX["X"], tolerance)
    assert lookup.find(Vector((-2.0, 0.0, 0.0))) is None
    assert lookup.find(Vector((3.0, 0.0, 0.0))) is None
    # Exact mirror of coords[1] is present via reflection of a positive query.
    assert lookup.find(Vector((1.0, 1.0, 0.0))) == 1


def check_vertex_mirror_lookup_nearest_among_candidates():
    """Phase 2a: multiple in-tolerance candidates pick the nearest Chebyshev."""

    tolerance = 1.0e-3
    # Query (+1, 0, 0) mirrors to (-1, 0, 0). Register three competitors.
    far = Vector((-1.0 - 0.9 * tolerance, 0.0, 0.0))
    near = Vector((-1.0 - 0.1 * tolerance, 0.0, 0.0))
    mid = Vector((-1.0 - 0.5 * tolerance, 0.0, 0.0))
    # Far first so order-of-registration alone would prefer the wrong index.
    lookup = matching.build_vertex_mirror_lookup(
        [far, near, mid],
        matching.AXIS_INDEX["X"],
        tolerance,
    )
    assert lookup.find(Vector((1.0, 0.0, 0.0))) == 1


def check_edge_side_tol_boundary_classification():
    """_edge_side boundary cases: ±tol endpoints, one-end-in-tol, tol±ε.

    Tolerance and sample coords use binary-exact floats so BMesh's float32
    storage does not nudge endpoints across the comparison boundary.
    """

    tolerance = 0.125  # exact in float32
    axis = matching.AXIS_INDEX["X"]
    bm = bmesh.new()
    try:

        def classify(a_x: float, b_x: float) -> str:
            a = bm.verts.new((a_x, 0.0, 0.0))
            b = bm.verts.new((b_x, 1.0, 0.0))
            edge = bm.edges.new((a, b))
            return stitch_pathedges._edge_side(edge, axis, tolerance)

        # Both endpoints exactly at +tol / beyond → POSITIVE (max > tol).
        assert classify(tolerance, 1.0) == "POSITIVE"
        # Both endpoints exactly at -tol / beyond → NEGATIVE (min < -tol).
        assert classify(-tolerance, -1.0) == "NEGATIVE"
        # Both exactly on plane bounds → PLANE.
        assert classify(tolerance, -tolerance) == "PLANE"
        assert classify(0.0, 0.0) == "PLANE"
        assert classify(tolerance, tolerance) == "PLANE"
        assert classify(-tolerance, -tolerance) == "PLANE"
        # One endpoint in the tol band, other outside same side → that side.
        assert classify(0.0, 1.0) == "POSITIVE"
        assert classify(0.0, -1.0) == "NEGATIVE"
        assert classify(-0.5 * tolerance, 1.0) == "POSITIVE"
        assert classify(0.5 * tolerance, -1.0) == "NEGATIVE"
        # Straddling with both ends outside the band → CROSSES.
        assert classify(-1.0, 1.0) == "CROSSES"
        # tol±ε: just outside the PLANE slab on opposite sides → CROSSES.
        eps = 0.001  # still exact as float32 relative to 0.125
        assert classify(-(tolerance + eps), tolerance + eps) == "CROSSES"
        # Just inside: both within ±tol → PLANE.
        assert classify(-(tolerance - eps), tolerance - eps) == "PLANE"
        # One end at +tol, other at -tol-ε → still NEGATIVE (both ≤ tol, min < -tol).
        assert classify(tolerance, -(tolerance + eps)) == "NEGATIVE"
        # Both ends strictly outside opposite sides → CROSSES.
        assert classify(tolerance + eps, -(tolerance + eps)) == "CROSSES"
        # One end only in tol: other deep positive → POSITIVE.
        assert classify(0.0, 2.0) == "POSITIVE"
        # One end only in tol: other deep negative → NEGATIVE.
        assert classify(0.0, -2.0) == "NEGATIVE"
    finally:
        bm.free()


class _FakeKeymapProps:
    bl_rna = type("RNA", (), {"properties": ()})()
    name = ""

    def is_property_set(self, identifier):
        return False


class _FakeKeymapItem:
    def __init__(self, idname, type="E", value="PRESS"):
        self.idname = idname
        self.active = True
        self.type = type
        self.value = value
        self.any = False
        self.shift = 0
        self.ctrl = 0
        self.alt = 0
        self.oskey = 0
        self.hyper = 0
        self.key_modifier = "NONE"
        self.direction = "ANY"
        self.repeat = False
        self.properties = _FakeKeymapProps()


class _FakeKeymapItems(list):
    def new(self, idname, *, type, value, head=False, **event_arguments):
        item = _FakeKeymapItem(idname, type=type, value=value)
        for name, value in event_arguments.items():
            setattr(item, name, value)
        if head:
            self.insert(0, item)
        else:
            self.append(item)
        return item


class _FakeKeymap:
    def __init__(
        self,
        items,
        *,
        name="Mesh",
        space_type="EMPTY",
        region_type="WINDOW",
        is_modal=False,
    ):
        self.name = name
        self.space_type = space_type
        self.region_type = region_type
        self.is_modal = is_modal
        self.keymap_items = _FakeKeymapItems(items)


class _FakeKeymaps(list):
    def new(self, *, name, space_type, region_type, modal, tool=False):
        for keymap in self:
            if (
                keymap.name == name
                and keymap.space_type == space_type
                and keymap.region_type == region_type
                and keymap.is_modal == modal
            ):
                return keymap
        keymap = _FakeKeymap(
            [],
            name=name,
            space_type=space_type,
            region_type=region_type,
            is_modal=modal,
        )
        self.append(keymap)
        return keymap


class _FakeWindowManager:
    def __init__(self, items, addon_items=None):
        keymap = _FakeKeymap(items)
        configs = {
            "user": type("User", (), {"keymaps": _FakeKeymaps([keymap])})(),
            "active": type("Active", (), {"name": "Blender"})(),
        }
        if addon_items is not None:
            configs["addon"] = type("Addon", (), {"keymaps": _FakeKeymaps([_FakeKeymap(addon_items)])})()
        self.keyconfigs = type("KeyConfigs", (), configs)()


def _fake_window_manager_with_addon(user_items, addon_items):
    return _FakeWindowManager(user_items, addon_items=addon_items)


def check_extrude_keymap_layers():
    knife = "mesh.knife_tool"
    extrude_e = "view3d.edit_mesh_extrude_move_normal"
    extrude_context = "mesh.extrude_context_move"
    original_operators = keymaps.OPERATOR_TOOL_KINDS
    original_extrude = keymaps.EXTRUDE_TOOL_KINDS
    keymaps.OPERATOR_TOOL_KINDS = {**original_operators, extrude_context: "EXTRUDE_CONTEXT"}
    keymaps.EXTRUDE_TOOL_KINDS = frozenset({*original_extrude, "EXTRUDE_CONTEXT"})
    try:
        routes, _fingerprint = keymaps._native_routes(
            _FakeWindowManager([_FakeKeymapItem(knife), _FakeKeymapItem(extrude_e)])
        )
        kinds = {route.tool_kind for route in routes}
        assert "KNIFE" in kinds, kinds
        assert "EXTRUDE_NORMAL" not in kinds, kinds

        routes, _fingerprint = keymaps._native_routes(
            _FakeWindowManager(
                [
                    _FakeKeymapItem(extrude_e, type="E"),
                    _FakeKeymapItem(extrude_context, type="E"),
                    _FakeKeymapItem(knife, type="K"),
                ]
            )
        )
        kinds = {route.tool_kind for route in routes}
        assert "KNIFE" in kinds, kinds
        assert "EXTRUDE_NORMAL" not in kinds, kinds
        assert "EXTRUDE_CONTEXT" not in kinds, kinds

        routes, _fingerprint = keymaps._native_routes(
            _FakeWindowManager(
                [
                    _FakeKeymapItem(knife, type="K"),
                    _FakeKeymapItem(extrude_e, type="E"),
                ]
            )
        )
        kinds = {route.tool_kind for route in routes}
        assert "KNIFE" in kinds, kinds
        assert "EXTRUDE_NORMAL" in kinds, kinds
    finally:
        keymaps.OPERATOR_TOOL_KINDS = original_operators
        keymaps.EXTRUDE_TOOL_KINDS = original_extrude


def check_replay_keymap_routes():
    connect = _FakeKeymapItem("mesh.vert_connect_path", type="K")
    connect.ctrl = 1
    connect_routes, merge_routes, fingerprint = keymaps._replay_keymap_routes(_FakeWindowManager([connect]))
    assert len(connect_routes) == 1, connect_routes
    assert merge_routes == [], merge_routes
    assert connect_routes[0].event.type == "K"
    assert connect_routes[0].event.ctrl == 1
    assert keymaps._event_arguments(connect_routes[0].event) == {
        "type": "K",
        "value": "PRESS",
        "ctrl": 1,
    }

    reassigned = _FakeKeymapItem("mesh.vert_connect_path", type="L")
    _, _, reassigned_fingerprint = keymaps._replay_keymap_routes(_FakeWindowManager([reassigned]))
    assert reassigned_fingerprint != fingerprint

    no_connect, _, _ = keymaps._replay_keymap_routes(_FakeWindowManager([_FakeKeymapItem("mesh.select_all", type="J")]))
    assert no_connect == [], no_connect

    merge = _FakeKeymapItem("wm.call_menu", type="N")
    merge.properties.name = "VIEW3D_MT_edit_mesh_merge"
    merge.shift = 1
    merge.alt = 1
    merge.oskey = 1
    merge.key_modifier = "SPACE"
    connect_routes, merge_routes, merge_fingerprint = keymaps._replay_keymap_routes(_FakeWindowManager([merge]))
    assert connect_routes == [], connect_routes
    assert len(merge_routes) == 1, merge_routes
    assert keymaps._event_arguments(merge_routes[0].event) == {
        "type": "N",
        "value": "PRESS",
        "shift": 1,
        "alt": 1,
        "oskey": 1,
        "key_modifier": "SPACE",
    }

    default_merge = _FakeKeymapItem("wm.call_menu", type="M")
    default_merge.properties.name = "VIEW3D_MT_edit_mesh_merge"
    _, _, default_merge_fingerprint = keymaps._replay_keymap_routes(_FakeWindowManager([default_merge]))
    assert merge_fingerprint != default_merge_fingerprint

    registered_before = list(keymaps._REGISTERED_ITEMS)
    try:
        window_manager = _fake_window_manager_with_addon([merge], [])
        keymaps._register_replay_keymaps(window_manager, [], merge_routes)
        addon_items = window_manager.keyconfigs.addon.keymaps[0].keymap_items
        assert all(item.idname != keymaps.CONNECT_OPERATOR for item in addon_items), addon_items
        registered_merge = [
            item for item in addon_items if item.idname == "wm.call_menu" and item.properties.name == keymaps.MERGE_MENU
        ]
        assert len(registered_merge) == 1, registered_merge
    finally:
        keymaps._REGISTERED_ITEMS.clear()
        keymaps._REGISTERED_ITEMS.extend(registered_before)

    # A Connect route outside the "Mesh" keymap is still discovered (the scan
    # is unconditional, like the delete/extrude menu scans).
    foreign_connect = _FakeKeymapItem("mesh.vert_connect_path", type="P")
    foreign_wm = _FakeWindowManager([])
    foreign_wm.keyconfigs.user.keymaps.append(
        _FakeKeymap([foreign_connect], name="3D View Tool: Edit Mesh, Poly Build")
    )
    foreign_routes, _, _ = keymaps._replay_keymap_routes(foreign_wm)
    assert len(foreign_routes) == 1, foreign_routes
    assert foreign_routes[0].keymap_name == "3D View Tool: Edit Mesh, Poly Build"

    # Inactive items are ignored.
    inactive = _FakeKeymapItem("mesh.vert_connect_path", type="J")
    inactive.active = False
    inactive_routes, _, _ = keymaps._replay_keymap_routes(_FakeWindowManager([inactive]))
    assert inactive_routes == [], inactive_routes

    # One event bound to both Connect and the Merge menu is ambiguous: drop both.
    conflict_connect = _FakeKeymapItem("mesh.vert_connect_path", type="Y")
    conflict_merge = _FakeKeymapItem("wm.call_menu", type="Y")
    conflict_merge.properties.name = "VIEW3D_MT_edit_mesh_merge"
    conflict_connects, conflict_merges, _ = keymaps._replay_keymap_routes(
        _FakeWindowManager([conflict_connect, conflict_merge])
    )
    assert conflict_connects == [], conflict_connects
    assert conflict_merges == [], conflict_merges

    # The Connect clone registers head-first with the cloned event arguments.
    registered_before = list(keymaps._REGISTERED_ITEMS)
    try:
        connect_item = _FakeKeymapItem("mesh.vert_connect_path", type="K")
        connect_item.ctrl = 1
        cloned_routes, _, _ = keymaps._replay_keymap_routes(_FakeWindowManager([connect_item]))
        assert len(cloned_routes) == 1, cloned_routes
        window_manager = _fake_window_manager_with_addon([connect_item], [])
        keymaps._register_replay_keymaps(window_manager, cloned_routes, [])
        addon_items = window_manager.keyconfigs.addon.keymaps[0].keymap_items
        registered_connect = [item for item in addon_items if item.idname == keymaps.CONNECT_OPERATOR]
        assert len(registered_connect) == 1, addon_items
        assert registered_connect[0].type == "K"
        assert registered_connect[0].ctrl == 1
        assert addon_items[0] is registered_connect[0], "connect clone must be inserted head-first"
    finally:
        keymaps._REGISTERED_ITEMS.clear()
        keymaps._REGISTERED_ITEMS.extend(registered_before)


class _ExplicitFakeProps:
    def __init__(self, values, *, set_identifiers=None):
        self._values = dict(values)
        self._set_identifiers = set(self._values) if set_identifiers is None else set(set_identifiers)
        self.bl_rna = type(
            "RNA",
            (),
            {"properties": [type("P", (), {"identifier": name})() for name in self._values]},
        )()

    def is_property_set(self, identifier):
        return identifier in self._set_identifiers

    def __getattr__(self, identifier):
        try:
            return self._values[identifier]
        except KeyError as exc:
            raise AttributeError(identifier) from exc


class _NonScalarChild:
    pointer = object()


class _FakeMacroChild:
    def __init__(self):
        self.use_normal_flip = True
        self.use_dissolve_ortho_edges = False
        self.mirror = True


class _FakeExtrudeMacro:
    def __init__(self, *, topology_name, transform_name, omit=()):
        if topology_name not in omit:
            setattr(self, topology_name, _FakeMacroChild())
        if transform_name not in omit:
            setattr(
                self,
                transform_name,
                _ExplicitFakeProps(
                    {"value": 0.0, "use_even_offset": True},
                    set_identifiers={"use_even_offset"},
                ),
            )


def check_extrude_option_capture():
    expected_children = {
        "EXTRUDE_NORMAL": ("MESH_OT_extrude_region", "TRANSFORM_OT_translate"),
        "EXTRUDE_CONTEXT": ("MESH_OT_extrude_context", "TRANSFORM_OT_translate"),
        "EXTRUDE_SHRINK_FATTEN": ("MESH_OT_extrude_region", "TRANSFORM_OT_shrink_fatten"),
        "EXTRUDE_FACES_INDIV": ("MESH_OT_extrude_faces_indiv", "TRANSFORM_OT_shrink_fatten"),
        "EXTRUDE_EDGES_INDIV": ("MESH_OT_extrude_edges_indiv", "TRANSFORM_OT_translate"),
        "EXTRUDE_VERTS_INDIV": ("MESH_OT_extrude_verts_indiv", "TRANSFORM_OT_translate"),
        "EXTRUDE_MANIFOLD": ("MESH_OT_extrude_region", "TRANSFORM_OT_translate"),
    }
    for tool_kind, (topology_name, transform_name) in expected_children.items():
        # Each required macro child is independently fail-closed.
        assert (
            extrude.capture_native_options(
                _FakeExtrudeMacro(
                    topology_name=topology_name,
                    transform_name=transform_name,
                    omit=(topology_name,),
                ),
                tool_kind,
            )
            is None
        )
        assert (
            extrude.capture_native_options(
                _FakeExtrudeMacro(
                    topology_name=topology_name,
                    transform_name=transform_name,
                    omit=(transform_name,),
                ),
                tool_kind,
            )
            is None
        )

        options = extrude.capture_native_options(
            _FakeExtrudeMacro(topology_name=topology_name, transform_name=transform_name),
            tool_kind,
        )
        assert options is not None, tool_kind
        assert options.use_normal_flip is True
        assert options.use_dissolve_ortho_edges is False
        assert options.mirror is True
        assert dict(options.transform_props) == {"value": 0.0, "use_even_offset": True}


def check_kmi_scalar_capture():
    pointer = object()
    item = _FakeKeymapItem("mesh.extrude_context_move")
    item.properties = _ExplicitFakeProps(
        {
            "bool_value": True,
            "int_value": 7,
            "float_value": 2.5,
            "str_value": "native",
            "pointer_value": pointer,
        }
    )
    captured = keymaps._capture_set_kmi_properties(item)
    assert dict(captured) == {
        "bool_value": True,
        "int_value": 7,
        "float_value": 2.5,
        "str_value": "native",
    }, captured
    assert all(isinstance(value, (bool, int, float, str)) for _, value in captured)


def check_extrude_keymap_regressions():
    extrude_e = "view3d.edit_mesh_extrude_move_normal"
    original_routes = dict(keymaps._ROUTES_BY_KEY)
    original_running = keymaps._RUNNING
    original_enabled = keymaps._ENABLED
    original_window_manager = keymaps._window_manager
    try:
        child = _NonScalarChild()
        scalar_item = _FakeKeymapItem(extrude_e, type="E")
        scalar_item.properties = _ExplicitFakeProps(
            {
                "dissolve_and_intersect": False,
                "MESH_OT_extrude_region": child,
            }
        )
        routes, _fingerprint = keymaps._native_routes(_FakeWindowManager([scalar_item]))
        extrude_routes = [route for route in routes if route.tool_kind == "EXTRUDE_NORMAL"]
        assert len(extrude_routes) == 1, extrude_routes
        props = dict(extrude_routes[0].kmi_properties)
        assert props.get("dissolve_and_intersect") is False, props
        assert "MESH_OT_extrude_region" not in props, props
        assert all(isinstance(value, (bool, int, float, str)) for value in props.values()), props

        routes, _fingerprint = keymaps._native_routes(
            _FakeWindowManager(
                [
                    _FakeKeymapItem(extrude_e, type="E"),
                    _FakeKeymapItem(extrude_e, type="E"),
                ]
            )
        )
        assert all(route.tool_kind != "EXTRUDE_NORMAL" for route in routes), routes

        live_item = _FakeKeymapItem(extrude_e, type="E")
        live_item.properties = _ExplicitFakeProps({"dissolve_and_intersect": False})
        live_wm = _FakeWindowManager([live_item])
        routes, _fingerprint = keymaps._native_routes(live_wm)
        route = next(candidate for candidate in routes if candidate.tool_kind == "EXTRUDE_NORMAL")
        keymaps._ROUTES_BY_KEY.clear()
        keymaps._ROUTES_BY_KEY[route.route_key] = route
        keymaps._RUNNING = True
        keymaps._ENABLED = True
        keymaps._window_manager = lambda: live_wm
        assert keymaps.route_is_current(route.route_key) is True
        # Stage 1: every supported operator kind on the saved key/event is
        # counted, so an additional supported KMI invalidates the route.
        duplicate = _FakeKeymapItem(extrude_e, type="E")
        live_wm.keyconfigs.user.keymaps[0].keymap_items.append(duplicate)
        assert keymaps.route_is_current(route.route_key) is False
        live_wm.keyconfigs.user.keymaps[0].keymap_items.remove(duplicate)
        assert keymaps.route_is_current(route.route_key) is True
        # Stage 1 must count every supported kind, not only the saved idname.
        # Keep the saved KMI and add another supported kind on the same event.
        context_item = _FakeKeymapItem("mesh.extrude_context_move", type="E")
        live_wm.keyconfigs.user.keymaps[0].keymap_items.append(context_item)
        assert keymaps.route_is_current(route.route_key) is False
        live_wm.keyconfigs.user.keymaps[0].keymap_items.remove(context_item)
        assert keymaps.route_is_current(route.route_key) is True
        # Stage 2: a different supported operator replacing the saved KMI is
        # stale even though the keymap still has exactly one supported item.
        live_item.idname = "mesh.extrude_context_move"
        assert keymaps.route_is_current(route.route_key) is False
        live_item.idname = extrude_e
        live_item.properties = _ExplicitFakeProps({"dissolve_and_intersect": True})
        assert keymaps.route_is_current(route.route_key) is False

        dissolve_item = _FakeKeymapItem(extrude_e, type="E")
        dissolve_item.properties = _ExplicitFakeProps({"dissolve_and_intersect": True})
        dissolve_wm = _FakeWindowManager([dissolve_item])
        routes, _fingerprint = keymaps._native_routes(dissolve_wm)
        dissolve_route = next(candidate for candidate in routes if candidate.tool_kind == "EXTRUDE_NORMAL")
        keymaps._ROUTES_BY_KEY.clear()
        keymaps._ROUTES_BY_KEY[dissolve_route.route_key] = dissolve_route
        keymaps._window_manager = lambda: dissolve_wm
        assert keymaps.live_route_has_dissolve_and_intersect(dissolve_route.route_key) is True
    finally:
        keymaps._ROUTES_BY_KEY.clear()
        keymaps._ROUTES_BY_KEY.update(original_routes)
        keymaps._RUNNING = original_running
        keymaps._ENABLED = original_enabled
        keymaps._window_manager = original_window_manager


def check_extrude_menu_fail_closed():
    """Call-menu + supported operator on the same event hooks neither route."""

    menu = _FakeKeymapItem("wm.call_menu", type="E")
    menu.properties.name = "VIEW3D_MT_edit_mesh_extrude"
    extrude_e = _FakeKeymapItem("view3d.edit_mesh_extrude_move_normal", type="E")

    routes, _fingerprint = keymaps._native_routes(_FakeWindowManager([menu, extrude_e]))
    assert all(route.tool_kind != "EXTRUDE_NORMAL" for route in routes), routes
    menu_routes, _menu_fp = keymaps._extrude_menu_routes(_FakeWindowManager([menu, extrude_e]))
    assert menu_routes == [], menu_routes

    only_menu, _only_fp = keymaps._extrude_menu_routes(_FakeWindowManager([menu]))
    assert len(only_menu) == 1, only_menu
    assert only_menu[0].menu_name == "VIEW3D_MT_edit_mesh_extrude"

    alias = _FakeKeymapItem("wm.call_menu", type="E")
    alias.properties.name = "VIEW3D_MT_view"
    alias_routes, _alias_fp = keymaps._extrude_menu_routes(_FakeWindowManager([menu, alias]))
    assert alias_routes == [], alias_routes

    second_native = _FakeKeymapItem("wm.call_menu", type="E")
    second_native.properties.name = "VIEW3D_MT_edit_mesh_extrude"
    two_native, _two_fp = keymaps._extrude_menu_routes(_FakeWindowManager([menu, second_native]))
    assert two_native == [], two_native

    original_menu_routes = dict(keymaps._EXTRUDE_MENU_ROUTES_BY_KEY)
    original_running = keymaps._RUNNING
    original_enabled = keymaps._ENABLED
    original_window_manager = keymaps._window_manager
    try:
        opener = _FakeKeymapItem(keymaps.EXTRUDE_MENU_OPENER, type="E")
        live_wm = _fake_window_manager_with_addon([menu], [opener])
        live_routes, _live_fp = keymaps._extrude_menu_routes(live_wm)
        assert len(live_routes) == 1, live_routes
        route = live_routes[0]
        keymaps._EXTRUDE_MENU_ROUTES_BY_KEY.clear()
        keymaps._EXTRUDE_MENU_ROUTES_BY_KEY[route.route_key] = route
        keymaps._RUNNING = True
        keymaps._ENABLED = True
        keymaps._window_manager = lambda: live_wm
        assert keymaps.extrude_menu_route_is_current(route.route_key) is True
        print("YSE_CORE_CASE=t4_route_is_current_false", flush=True)
        assert keymaps.extrude_menu_route_is_current("yse:stale-route") is False
        live_wm.keyconfigs.addon.keymaps[0].keymap_items.append(_FakeKeymapItem(keymaps.EXTRUDE_MENU_OPENER, type="E"))
        assert keymaps.extrude_menu_route_is_current(route.route_key) is False
        live_wm.keyconfigs.addon.keymaps[0].keymap_items.pop()
        assert keymaps.extrude_menu_route_is_current(route.route_key) is True
        live_alias = _FakeKeymapItem("wm.call_menu", type="E")
        live_alias.properties.name = "VIEW3D_MT_view"
        live_wm.keyconfigs.user.keymaps[0].keymap_items.append(live_alias)
        assert keymaps.extrude_menu_route_is_current(route.route_key) is False
    finally:
        keymaps._EXTRUDE_MENU_ROUTES_BY_KEY.clear()
        keymaps._EXTRUDE_MENU_ROUTES_BY_KEY.update(original_menu_routes)
        keymaps._RUNNING = original_running
        keymaps._ENABLED = original_enabled
        keymaps._window_manager = original_window_manager


def _dummy_extrude_snapshot(tool_kind, selected_vertex_ids, face_corners, edge_endpoints):
    return ExtrudeSnapshot(
        axis_index=0,
        tolerance=1e-4,
        tool_kind=tool_kind,
        route_kmi_properties=(),
        mesh_select_mode=MeshSelectionMode(False, False, False),
        selected_vertex_ids=frozenset(selected_vertex_ids),
        selected_edge_markers=frozenset(),
        selected_face_ids=frozenset(),
        vertex_preop=(),
        vertex_pairs=(),
        edge_pairs=(),
        face_pairs=(),
        hidden_vertex_ids=frozenset(),
        hidden_edge_markers=frozenset(),
        hidden_face_ids=frozenset(),
        face_corners=tuple(face_corners),
        edge_endpoints=tuple(edge_endpoints),
        vertex_count=0,
        edge_count=0,
        face_count=0,
    )


def check_derive_expected_census_stage3c():
    """Empty F_r FACES_INDIV, non-empty F_r EDGES/VERTS_INDIV, and F12 edge."""

    face_corners = ((1, (10, 11, 12, 13)),)
    uncovered = _dummy_extrude_snapshot(
        "EXTRUDE_FACES_INDIV",
        {10, 11},
        face_corners,
        ((100, (10, 11)),),
    )
    assert extrude._derive_expected_census(uncovered) is None

    covering_vids = {10, 11, 12, 13}
    covering_edges = (
        (100, (10, 11)),
        (101, (11, 12)),
        (102, (12, 13)),
        (103, (13, 10)),
    )
    for kind in ("EXTRUDE_EDGES_INDIV", "EXTRUDE_VERTS_INDIV"):
        covered = _dummy_extrude_snapshot(kind, covering_vids, face_corners, covering_edges)
        assert extrude._derive_expected_census(covered) is None, kind

    edge_only = _dummy_extrude_snapshot(
        "EXTRUDE_EDGES_INDIV",
        {10, 11},
        face_corners,
        ((100, (10, 11)),),
    )
    assert extrude._derive_expected_census(edge_only) == ((2, 3, 1), (0, 0, 0), (2, 3, 1))


def run():
    bm = build_two_symmetric_quads()
    topology = snapshot.prepare_topology(bm, matching.AXIS_INDEX["X"], 1.0e-5)
    assert topology.matched_faces == 2 and topology.total_faces == 2
    assert len(topology.hidden_by_face_id) == 2

    path_edge = split_left_face_like_native_knife(bm)
    marker = bm.edges.layers.int.get(layer_names.EDGE_ORIGINAL_LAYER)
    face_ids = bm.faces.layers.int.get(layer_names.FACE_ID_LAYER)
    assert path_edge is not None and path_edge[marker] == 0
    assert all(edge[marker] > 0 for edge in bm.edges if edge != path_edge)

    source_edges, side, total, crossing = stitch_pathedges.collect_source_path_edges(
        bm, matching.AXIS_INDEX["X"], 1.0e-5, "AUTO"
    )
    assert source_edges == [path_edge]
    assert side == "NEGATIVE" and total == 1 and crossing == 0

    targets, unmatched = stitch_pathedges.target_face_ids_for_edges(source_edges, face_ids, topology.mirror_face_ids)
    assert len(targets) == 1 and not unmatched
    assert stitch_reflect.reflected_path_uses_only_target_boundaries(
        bm,
        source_edges,
        matching.AXIS_INDEX["X"],
        1.0e-5,
        topology.mirror_face_ids,
    )

    # Some native loop routes can propagate a positive CustomData value onto
    # a newly created internal edge. Selection plus the shared original face
    # ID must still recover the complete native result.
    path_edge[marker] = 999
    path_edge.select = True
    selected_source, selected_side, selected_total, selected_crossing = stitch_pathedges.collect_source_path_edges(
        bm,
        matching.AXIS_INDEX["X"],
        1.0e-5,
        "AUTO",
        selected_only=True,
    )
    assert selected_source == [path_edge]
    assert selected_side == "NEGATIVE"
    assert selected_total == 1 and selected_crossing == 0

    created, present, direct_reason = stitch_reflect.apply_reflected_path_topology(
        bm,
        selected_source,
        matching.AXIS_INDEX["X"],
        1.0e-5,
        topology.mirror_face_ids,
    )
    assert (created, present, direct_reason) == (1, 0, "")
    assert (len(bm.verts), len(bm.edges), len(bm.faces)) == (12, 14, 4)
    reflected_edge = next(edge for edge in bm.edges if all(abs(vertex.co.x - 1.5) < 1.0e-8 for vertex in edge.verts))
    assert not reflected_edge.select

    snapshot.remove_temporary_layers(bm)
    assert bm.edges.layers.int.get(layer_names.EDGE_ORIGINAL_LAYER) is None
    assert bm.faces.layers.int.get(layer_names.FACE_ID_LAYER) is None
    bm.free()

    # Direct loop reconstruction must retain source selection and the exact
    # hidden state of a target half, including newly created target elements.
    hidden_bm = build_two_symmetric_quads()
    target_face = next(face for face in hidden_bm.faces if face.calc_center_median().x > 0.0)
    target_face.hide_set(True)
    hidden_topology = snapshot.prepare_topology(
        hidden_bm,
        matching.AXIS_INDEX["X"],
        1.0e-5,
    )
    hidden_path = split_left_face_like_native_knife(hidden_bm)
    hidden_path.select = True
    for vertex in hidden_path.verts:
        vertex.select = True
    selection_snapshot = selection.add_selection_layers(hidden_bm)
    hidden_source, _side, _total, _crossing = stitch_pathedges.collect_source_path_edges(
        hidden_bm,
        matching.AXIS_INDEX["X"],
        1.0e-5,
        "AUTO",
        selected_only=True,
    )
    created, present, reason = stitch_reflect.apply_reflected_path_topology(
        hidden_bm,
        hidden_source,
        matching.AXIS_INDEX["X"],
        1.0e-5,
        hidden_topology.mirror_face_ids,
    )
    assert (created, present, reason) == (1, 0, "")
    selection.restore_visibility_and_selection(
        hidden_bm,
        hidden_topology.hidden_by_face_id,
        selection_snapshot,
    )
    target_faces = [face for face in hidden_bm.faces if face.calc_center_median().x > 0.0]
    assert len(target_faces) == 2 and all(face.hide for face in target_faces)
    target_path = next(
        edge for edge in hidden_bm.edges if all(abs(vertex.co.x - 1.5) < 1.0e-8 for vertex in edge.verts)
    )
    assert target_path.hide and not target_path.select
    source_path = next(
        edge for edge in hidden_bm.edges if all(abs(vertex.co.x + 1.5) < 1.0e-8 for vertex in edge.verts)
    )
    assert source_path.select and all(vertex.select for vertex in source_path.verts)
    snapshot.remove_temporary_layers(hidden_bm)
    hidden_bm.free()

    # A multi-click Knife can place an intentional waypoint inside a face.
    # The direct builder realizes it as a face-interior chain, so the
    # boundary preflight must accept it instead of declining to Knife Project.
    bend_bm = build_two_symmetric_quads()
    bend_topology = snapshot.prepare_topology(
        bend_bm,
        matching.AXIS_INDEX["X"],
        1.0e-5,
    )
    bottom = next(
        edge
        for edge in bend_bm.edges
        if all(vertex.co.x < 0.0 for vertex in edge.verts)
        and all(abs(vertex.co.y + 1.0) < 1.0e-8 for vertex in edge.verts)
    )
    _edge, bend_bottom = bmesh.utils.edge_split(bottom, bottom.verts[0], 0.5)
    top = next(
        edge
        for edge in bend_bm.edges
        if all(vertex.co.x < 0.0 for vertex in edge.verts)
        and all(abs(vertex.co.y - 1.0) < 1.0e-8 for vertex in edge.verts)
    )
    _edge, bend_top = bmesh.utils.edge_split(top, top.verts[0], 0.5)
    bend_face = next(face for face in bend_bm.faces if bend_bottom in face.verts and bend_top in face.verts)
    bmesh.utils.face_split(
        bend_face,
        bend_bottom,
        bend_top,
        coords=[(-1.2, 0.0, 0.0)],
    )
    bend_source, bend_side, bend_total, bend_crossing = stitch_pathedges.collect_source_path_edges(
        bend_bm,
        matching.AXIS_INDEX["X"],
        1.0e-5,
        "AUTO",
    )
    assert bend_side == "NEGATIVE" and bend_total == 2 and bend_crossing == 0
    assert stitch_reflect.reflected_path_uses_only_target_boundaries(
        bend_bm,
        bend_source,
        matching.AXIS_INDEX["X"],
        1.0e-5,
        bend_topology.mirror_face_ids,
        bend_topology.carrier_frames,
    )
    bend_created, bend_already, bend_reason = stitch_reflect.apply_reflected_path_topology(
        bend_bm,
        bend_source,
        matching.AXIS_INDEX["X"],
        1.0e-5,
        bend_topology.mirror_face_ids,
        bend_topology.carrier_frames,
    )
    assert bend_reason == "", bend_reason
    assert bend_created == 2, (bend_created, bend_already)
    bend_bm.free()

    check_injective_component_counterexample()
    check_injective_tie_rejection_is_order_independent()
    check_injective_step_limit_rejects_component()
    check_on_plane_vertices_never_serve_off_plane_queries()
    check_pair_table_involution_and_partial_pairs()
    check_quantized_coordinate_bin_boundary()
    check_interior_edge_factor_uses_absolute_distance()
    check_choose_source_side_ignores_coordinate_tolerance()
    check_collapsed_offset_prefers_geometrically_closer_marker()
    check_reflected_path_lazy_existing_edge_store()
    check_reflected_path_lazy_store_falls_back_after_identity_miss()
    check_large_ngon_coords_match_without_recursion_error()
    check_vertex_mirror_lookup_bin_boundary()
    check_vertex_mirror_lookup_is_on_plane_boundary()
    check_vertex_mirror_lookup_asymmetric_returns_none()
    check_vertex_mirror_lookup_nearest_among_candidates()
    check_edge_side_tol_boundary_classification()
    check_extrude_keymap_layers()
    check_replay_keymap_routes()
    check_kmi_scalar_capture()
    check_extrude_option_capture()
    check_extrude_keymap_regressions()
    check_extrude_menu_fail_closed()
    check_derive_expected_census_stage3c()
    print("YSE_CORE_TEST_OK", flush=True)


if __name__ == "__main__":
    run()
