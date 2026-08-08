# SPDX-License-Identifier: GPL-3.0-or-later

"""Headless Blender checks for topology marking and cutter reflection."""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import bmesh
from mathutils import Vector

PACKAGE_PARENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_PARENT))

from ydd_symmetric_edit import core  # noqa: E402


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


def build_marked_graph(coordinates, edges):
    bm = bmesh.new()
    marker = bm.edges.layers.int.new(core.EDGE_ORIGINAL_LAYER)
    vertices = [bm.verts.new(coordinate) for coordinate in coordinates]
    for a, b in edges:
        edge = bm.edges.new((vertices[a], vertices[b]))
        edge[marker] = 0
    return bm


def coordinate_signature(bm):
    return tuple(sorted(tuple(round(float(component), 7) for component in vertex.co) for vertex in bm.verts))


def check_radius_search_is_the_only_candidate_source():
    """The KDTree radius search is a complete enumeration; there is no
    all-pairs fallback anymore.  An empty candidate set must fail cleanly,
    and the normal path must snap to the exact expected coordinates."""

    expected_vertices = [
        Vector((0.0, 0.0, 0.0)),
        Vector((1.0, 0.0, 0.0)),
        Vector((1.0, 1.0, 0.0)),
        Vector((0.0, 1.0, 0.0)),
    ]
    expected_edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    projected_vertices = [
        (0.0010, -0.0004, 0.0),
        (1.0006, 0.0003, 0.0),
        (0.9995, 1.0008, 0.0),
        (-0.0007, 0.9996, 0.0),
    ]

    fast_bm = build_marked_graph(projected_vertices, expected_edges)
    starved_bm = build_marked_graph(projected_vertices, expected_edges)
    try:
        fast_result = core.snap_projected_graph(
            fast_bm,
            expected_vertices,
            expected_edges,
            1.0e-5,
        )
        assert fast_result[0]
        assert fast_result[2] == ""
        assert coordinate_signature(fast_bm) == tuple(sorted(tuple(coordinate) for coordinate in expected_vertices))

        nearby_candidates = core._nearby_projection_candidates
        try:
            core._nearby_projection_candidates = lambda *_args: []
            starved_result = core.snap_projected_graph(
                starved_bm,
                expected_vertices,
                expected_edges,
                1.0e-5,
            )
        finally:
            core._nearby_projection_candidates = nearby_candidates

        assert not starved_result[0]
        assert starved_result[2] == "could not match every projected graph vertex"
        # The failed attempt must not have moved anything.
        assert coordinate_signature(starved_bm) == tuple(sorted(tuple(coordinate) for coordinate in projected_vertices))
    finally:
        fast_bm.free()
        starved_bm.free()


def check_constrained_matching_resolves_near_coincident_vertices():
    """The distance-greedy assignment provably swapped near-coincident
    vertices and then rejected a solvable graph; the constrained matching
    must construct the identity assignment instead."""

    expected_vertices = [
        Vector((-10.0, 0.0, 0.0)),
        Vector((0.0, 0.0, 0.0)),
        Vector((0.001, 0.0, 0.0)),
        Vector((10.0, 0.0, 0.0)),
    ]
    expected_edges = [(0, 1), (1, 2), (2, 3)]
    projected_vertices = [
        (-10.0, 0.0, 0.0),
        (0.0009, 0.0, 0.0),
        (0.0001, 0.0, 0.0),
        (10.0, 0.0, 0.0),
    ]

    bm = build_marked_graph(projected_vertices, expected_edges)
    try:
        snapped, _error, reason = core.snap_projected_graph(
            bm,
            expected_vertices,
            expected_edges,
            1.0e-4,
        )
        assert snapped, reason
        assert reason == ""
        assert coordinate_signature(bm) == tuple(
            sorted(tuple(round(float(component), 7) for component in coordinate) for coordinate in expected_vertices)
        )
    finally:
        bm.free()


def check_injective_component_counterexample():
    """The design's mutual-nearest counterexample must yield the complete
    assignment q1->t2, q2->t1, q3->t3 (1-D, tolerance 1.0)."""

    targets = [Vector((0.0, 0.0, 0.0)), Vector((1.2, 0.0, 0.0)), Vector((-1.5, 0.0, 0.0))]
    queries = [Vector((0.4, 0.0, 0.0)), Vector((-0.9, 0.0, 0.0)), Vector((-2.0, 0.0, 0.0))]
    lookup = core.build_vertex_mirror_lookup(targets, 0, 1.0)
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
    result = core._solve_injective_component([0, 1, 2], candidate_lists)
    assert result is None, result

    # A strictly better unique optimum must still be found.
    unique_lists = [
        [(0.1, 10), (0.4, 11)],
        [(0.1, 11), (0.4, 10)],
    ]
    result = core._solve_injective_component([0, 1], unique_lists)
    assert result == {0: 10, 1: 11}, result


def check_injective_step_limit_rejects_component():
    # Costs 0.4 vs 0.5: a unique optimum, no tie.
    candidate_lists = [
        [(0.1, 10), (0.3, 11)],
        [(0.2, 10), (0.3, 11)],
    ]
    # Exhausting the search takes 3 trial assignments; capping at 2 must
    # reject the whole component even though the optimum was already seen.
    assert core._solve_injective_component([0, 1], candidate_lists, step_limit=2) is None
    assert core._solve_injective_component([0, 1], candidate_lists) == {0: 10, 1: 11}


def check_on_plane_vertices_never_serve_off_plane_queries():
    """Partial-batch regression: an off-plane query whose reflection lands
    near an on-plane vertex must stay unresolved, even when no on-plane query
    is present in the batch to reserve that vertex."""

    tolerance = 0.001
    registered = [Vector((-0.0002, 0.0, 0.0)), Vector((5.0, 0.0, 0.0))]
    lookup = core.build_vertex_mirror_lookup(registered, 0, tolerance)
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
    pairs = core.build_vertex_pair_table(coords, 0, 0.001)
    assert all(pairs[pairs[vertex]] == vertex for vertex in pairs)
    assert pairs.get(0) == 2 and pairs.get(2) == 0
    assert pairs.get(1) == 3 and pairs.get(3) == 1
    assert pairs.get(4) == 4  # on-plane self-pair
    assert 5 not in pairs


def check_projection_backtracking_and_failure_reasons():
    """The constrained matcher must survive a first-candidate contradiction
    (undo + next candidate), report unsolvable adjacency, and report the step
    limit as ambiguity instead of raising."""

    # d0-d1 adjacent; e0/e1 isolated in the expected graph, e2-e3 adjacent.
    # d0's nearest candidate e0 forces d1's options empty -> undo -> e2 wins.
    candidates = [(0.1, 0, 0), (0.2, 0, 2), (0.1, 1, 1), (0.2, 1, 3)]
    destination_pairs = [(0, 1)]
    expected_edge_set = {(2, 3)}
    assignment, _distances, reason = core._assign_projection_candidates(
        candidates, 2, destination_pairs, expected_edge_set
    )
    assert reason == "", reason
    assert assignment == {0: 2, 1: 3}, assignment

    # Unsolvable: both destinations forced onto non-adjacent expected verts.
    _assignment, _distances, reason = core._assign_projection_candidates([(0.0, 0, 0), (0.0, 1, 1)], 2, [(0, 1)], set())
    assert reason == "graph adjacency mismatch", reason

    # Step limit: with the limit forced to zero, the first real branching
    # trial must fail as ambiguity, not raise.
    original_limit = core._PROJECTION_STEP_LIMIT
    try:
        core._PROJECTION_STEP_LIMIT = 0
        _assignment, _distances, reason = core._assign_projection_candidates(
            candidates, 2, destination_pairs, expected_edge_set
        )
        assert reason == "ambiguous projection correspondence", reason
    finally:
        core._PROJECTION_STEP_LIMIT = original_limit


def check_long_graph_fast_path():
    segment_count = 3000
    expected_vertices = [Vector((float(index), 0.0, 0.0)) for index in range(segment_count + 1)]
    expected_edges = [(index, index + 1) for index in range(segment_count)]
    projected_vertices = [
        (
            float(index),
            5.0e-5 if index % 2 else -5.0e-5,
            0.0,
        )
        for index in range(segment_count + 1)
    ]
    bm = build_marked_graph(projected_vertices, expected_edges)
    try:
        started_at = time.perf_counter()
        snapped, error, reason = core.snap_projected_graph(
            bm,
            expected_vertices,
            expected_edges,
            1.0e-5,
        )
        elapsed = time.perf_counter() - started_at
        assert snapped, reason
        assert not reason
        assert 0.0 < error < 1.0e-4
        assert coordinate_signature(bm) == tuple(tuple(coordinate) for coordinate in expected_vertices)
        # This intentionally generous ceiling catches accidental restoration
        # of the legacy nine-million-candidate path without being CI-fragile.
        assert elapsed < 3.0, elapsed
        print(f"YSE_CORE_LONG_GRAPH_SECONDS={elapsed:.4f}", flush=True)
    finally:
        bm.free()


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

    primary_left = core._quantized_coordinate(left, tolerance)
    primary_right = core._quantized_coordinate(right, tolerance)
    assert primary_left != primary_right, (
        primary_left,
        primary_right,
        left.x * inverse,
        right.x * inverse,
    )
    neighborhood_left = set(core._iter_quantized_neighborhood(left, tolerance))
    assert primary_right in neighborhood_left
    assert primary_left in set(core._iter_quantized_neighborhood(right, tolerance))

    far = Vector((boundary + 2.0 * tolerance, 0.0, 0.0))
    assert primary_left not in set(core._iter_quantized_neighborhood(far, tolerance))
    assert core._quantized_coordinate(far, tolerance) not in neighborhood_left

    # Historical round() bug at half-bin boundaries (float64 sample; inverse=1e5).
    round_left_x = 0.5 * tolerance - 1.0e-10
    round_right_x = 0.5 * tolerance + 1.0e-10
    assert round(round_left_x * inverse) != round(round_right_x * inverse)
    round_left = Vector((round_left_x, 0.0, 0.0))
    round_right = Vector((round_right_x, 0.0, 0.0))
    assert core._quantized_coordinate(round_left, tolerance) in set(
        core._iter_quantized_neighborhood(round_right, tolerance)
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
        topology = core.prepare_topology(bm, core.AXIS_INDEX["X"], tolerance)
        assert topology.matched_faces == 2 and topology.total_faces == 2, (
            topology.matched_faces,
            topology.total_faces,
            topology.mirror_face_ids,
        )

        existing: dict = {}
        core._register_edge_endpoint_pair(
            existing,
            Vector((inner + delta, -1.0, 0.0)),
            Vector((inner + delta, 1.0, 0.0)),
            tolerance,
        )
        assert core._edge_coordinate_key_matches(
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
    assert core._is_interior_edge_factor(4.0e-6, edge_length, tolerance)
    assert not core._is_interior_edge_factor(1.0e-9, edge_length, tolerance)
    assert not core._is_interior_edge_factor(0.5, 0.0, tolerance)


def check_choose_source_side_ignores_coordinate_tolerance():
    """Contract G: path length must not be compared to coordinate tolerance."""

    bm = bmesh.new()
    try:
        # Length 8e-4 < coordinate tolerance 1e-3, but still a real negative cut.
        a = bm.verts.new((-0.0020, 0.0, 0.0))
        b = bm.verts.new((-0.0028, 0.0, 0.0))
        edge = bm.edges.new((a, b))
        side, crossing = core.choose_source_side([edge], core.AXIS_INDEX["X"], 1.0e-3, "AUTO")
        assert side == "NEGATIVE" and crossing == 0
        assert edge.calc_length() < 1.0e-3
        assert edge.calc_length() > core._MIN_SIDE_LENGTH
    finally:
        bm.free()


def check_collapsed_offset_prefers_geometrically_closer_marker():
    """R2-1: ambiguous-band competitors pick the nearer original marker."""

    tolerance = 1.0e-3
    bm = bmesh.new()
    try:
        marker_layer = bm.edges.layers.int.new(core.EDGE_ORIGINAL_LAYER)

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

        markers, reason = core.collapsed_offset_target_edge_markers(
            bm,
            [source_edge],
            core.AXIS_INDEX["X"],
            tolerance,
        )
        assert reason == "", reason
        assert markers == {10}, markers
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
    assert core._coords_match_chebyshev(first, second, tolerance)
    broken = second[:-1] + ((100.0, 100.0, 100.0),)
    assert not core._coords_match_chebyshev(first, broken, tolerance)


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
    primary_reg = core._quantized_coordinate(registered, tolerance)
    primary_mirror = core._quantized_coordinate(core.mirror_coordinate(query, 0), tolerance)
    assert primary_reg != primary_mirror

    lookup = core.build_vertex_mirror_lookup([registered], core.AXIS_INDEX["X"], tolerance)
    assert lookup.find(query) == 0

    # Invariant: any axis gap ≥ 2·tolerance must never hit.
    far_query = Vector((-(boundary + 2.0 * tolerance), 0.25, -0.5))
    assert lookup.find(far_query) is None


def check_vertex_mirror_lookup_is_on_plane_boundary():
    """Phase 2a: is_on_plane is true on |axis| ≤ tolerance, false outside."""

    tolerance = 1.0e-3
    lookup = core.build_vertex_mirror_lookup([], core.AXIS_INDEX["Y"], tolerance)
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
    lookup = core.build_vertex_mirror_lookup(coords, core.AXIS_INDEX["X"], tolerance)
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
    lookup = core.build_vertex_mirror_lookup(
        [far, near, mid],
        core.AXIS_INDEX["X"],
        tolerance,
    )
    assert lookup.find(Vector((1.0, 0.0, 0.0))) == 1


def check_edge_side_tol_boundary_classification():
    """_edge_side boundary cases: ±tol endpoints, one-end-in-tol, tol±ε.

    Tolerance and sample coords use binary-exact floats so BMesh's float32
    storage does not nudge endpoints across the comparison boundary.
    """

    tolerance = 0.125  # exact in float32
    axis = core.AXIS_INDEX["X"]
    bm = bmesh.new()
    try:

        def classify(a_x: float, b_x: float) -> str:
            a = bm.verts.new((a_x, 0.0, 0.0))
            b = bm.verts.new((b_x, 1.0, 0.0))
            edge = bm.edges.new((a, b))
            return core._edge_side(edge, axis, tolerance)

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


def run():
    bm = build_two_symmetric_quads()
    topology = core.prepare_topology(bm, core.AXIS_INDEX["X"], 1.0e-5)
    assert topology.matched_faces == 2 and topology.total_faces == 2
    assert len(topology.hidden_by_face_id) == 2

    path_edge = split_left_face_like_native_knife(bm)
    marker = bm.edges.layers.int.get(core.EDGE_ORIGINAL_LAYER)
    face_ids = bm.faces.layers.int.get(core.FACE_ID_LAYER)
    assert path_edge is not None and path_edge[marker] == 0
    assert all(edge[marker] > 0 for edge in bm.edges if edge != path_edge)

    source_edges, side, total, crossing = core.collect_source_path_edges(bm, core.AXIS_INDEX["X"], 1.0e-5, "AUTO")
    assert source_edges == [path_edge]
    assert side == "NEGATIVE" and total == 1 and crossing == 0

    targets, unmatched = core.target_face_ids_for_edges(source_edges, face_ids, topology.mirror_face_ids)
    assert len(targets) == 1 and not unmatched
    assert core.reflected_path_uses_only_target_boundaries(
        bm,
        source_edges,
        core.AXIS_INDEX["X"],
        1.0e-5,
        topology.mirror_face_ids,
    )

    coordinates, edges, already_present = core.build_reflected_cutter(bm, source_edges, core.AXIS_INDEX["X"], 1.0e-5)
    assert len(coordinates) == 2 and edges == [(0, 1)]
    assert already_present == 0
    assert all(abs(co.x - 1.5) < 1.0e-8 for co in coordinates)

    # Some native loop routes can propagate a positive CustomData value onto
    # a newly created internal edge. Selection plus the shared original face
    # ID must still recover the complete native result.
    path_edge[marker] = 999
    path_edge.select = True
    selected_source, selected_side, selected_total, selected_crossing = core.collect_source_path_edges(
        bm,
        core.AXIS_INDEX["X"],
        1.0e-5,
        "AUTO",
        selected_only=True,
    )
    assert selected_source == [path_edge]
    assert selected_side == "NEGATIVE"
    assert selected_total == 1 and selected_crossing == 0

    created, present, direct_reason = core.apply_reflected_path_topology(
        bm,
        selected_source,
        core.AXIS_INDEX["X"],
        1.0e-5,
        topology.mirror_face_ids,
    )
    assert (created, present, direct_reason) == (1, 0, "")
    assert (len(bm.verts), len(bm.edges), len(bm.faces)) == (12, 14, 4)
    reflected_edge = next(edge for edge in bm.edges if all(abs(vertex.co.x - 1.5) < 1.0e-8 for vertex in edge.verts))
    assert not reflected_edge.select

    core.remove_temporary_layers(bm)
    assert bm.edges.layers.int.get(core.EDGE_ORIGINAL_LAYER) is None
    assert bm.faces.layers.int.get(core.FACE_ID_LAYER) is None
    bm.free()

    # Direct loop reconstruction must retain source selection and the exact
    # hidden state of a target half, including newly created target elements.
    hidden_bm = build_two_symmetric_quads()
    target_face = next(face for face in hidden_bm.faces if face.calc_center_median().x > 0.0)
    target_face.hide_set(True)
    hidden_topology = core.prepare_topology(
        hidden_bm,
        core.AXIS_INDEX["X"],
        1.0e-5,
    )
    hidden_path = split_left_face_like_native_knife(hidden_bm)
    hidden_path.select = True
    for vertex in hidden_path.verts:
        vertex.select = True
    selection_snapshot = core.add_selection_layers(hidden_bm)
    hidden_source, _side, _total, _crossing = core.collect_source_path_edges(
        hidden_bm,
        core.AXIS_INDEX["X"],
        1.0e-5,
        "AUTO",
        selected_only=True,
    )
    created, present, reason = core.apply_reflected_path_topology(
        hidden_bm,
        hidden_source,
        core.AXIS_INDEX["X"],
        1.0e-5,
        hidden_topology.mirror_face_ids,
    )
    assert (created, present, reason) == (1, 0, "")
    core.restore_visibility_and_selection(
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
    core.remove_temporary_layers(hidden_bm)
    hidden_bm.free()

    # A multi-click Knife can place an intentional waypoint inside a face.
    # The boundary-only direct builder must decline it before editing so the
    # production operator can retain Knife Project as a compatibility fallback.
    bend_bm = build_two_symmetric_quads()
    bend_topology = core.prepare_topology(
        bend_bm,
        core.AXIS_INDEX["X"],
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
    bend_source, bend_side, bend_total, bend_crossing = core.collect_source_path_edges(
        bend_bm,
        core.AXIS_INDEX["X"],
        1.0e-5,
        "AUTO",
    )
    assert bend_side == "NEGATIVE" and bend_total == 2 and bend_crossing == 0
    assert not core.reflected_path_uses_only_target_boundaries(
        bend_bm,
        bend_source,
        core.AXIS_INDEX["X"],
        1.0e-5,
        bend_topology.mirror_face_ids,
    )
    bend_bm.free()

    # Existing target vertices are validation anchors, never snap targets.
    guard = bmesh.new()
    a = guard.verts.new((0.0, 0.0, 0.0))
    b = guard.verts.new((1.0, 0.0, 0.0))
    guard.edges.new((a, b))
    marker = guard.edges.layers.int.new(core.EDGE_ORIGINAL_LAYER)
    edge = next(iter(guard.edges))
    edge[marker] = 0
    before = (a.co.copy(), b.co.copy())
    snapped, _error, _reason = core.snap_projected_graph(
        guard,
        [Vector((0.1, 0.0, 0.0)), Vector((1.1, 0.0, 0.0))],
        [(0, 1)],
        1.0e-5,
        {hash(a), hash(b)},
    )
    assert not snapped
    assert a.co == before[0] and b.co == before[1]
    guard.free()

    check_radius_search_is_the_only_candidate_source()
    check_constrained_matching_resolves_near_coincident_vertices()
    check_injective_component_counterexample()
    check_injective_tie_rejection_is_order_independent()
    check_injective_step_limit_rejects_component()
    check_on_plane_vertices_never_serve_off_plane_queries()
    check_pair_table_involution_and_partial_pairs()
    check_projection_backtracking_and_failure_reasons()
    check_long_graph_fast_path()
    check_quantized_coordinate_bin_boundary()
    check_interior_edge_factor_uses_absolute_distance()
    check_choose_source_side_ignores_coordinate_tolerance()
    check_collapsed_offset_prefers_geometrically_closer_marker()
    check_large_ngon_coords_match_without_recursion_error()
    check_vertex_mirror_lookup_bin_boundary()
    check_vertex_mirror_lookup_is_on_plane_boundary()
    check_vertex_mirror_lookup_asymmetric_returns_none()
    check_vertex_mirror_lookup_nearest_among_candidates()
    check_edge_side_tol_boundary_classification()
    print("YSE_CORE_TEST_OK", flush=True)


if __name__ == "__main__":
    run()
