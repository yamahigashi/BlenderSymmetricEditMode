from __future__ import annotations

from collections.abc import Iterable, Sequence

import bmesh
from mathutils import Vector

from . import stitch_common
from .matching import coordinates_match


def plane_intersection_of_edge(
    edge: bmesh.types.BMEdge,
    axis_index: int,
) -> tuple[Vector, float] | None:
    """Return ``(p, factor_from_vert0)`` where the edge meets the mirror plane.

    *p* is snapped so ``p[axis_index] == 0``. Returns ``None`` when the edge is
    parallel to the plane or the intersection is outside the segment.
    """

    a = edge.verts[0].co
    b = edge.verts[1].co
    ax = float(a[axis_index])
    bx = float(b[axis_index])
    denom = ax - bx
    if abs(denom) <= 1.0e-30:
        return None
    # a + t*(b-a) has axis component 0 ⇒ t = ax / (ax - bx)
    factor = ax / denom
    if factor < 0.0 or factor > 1.0:
        return None
    point = a.lerp(b, factor)
    point[axis_index] = 0.0
    return point, factor


def apply_crosses_p_stitch(
    bm: bmesh.types.BMesh,
    crosses_edges: Iterable[bmesh.types.BMEdge],
    axis_index: int,
    tolerance: float,
    *,
    return_summary: bool = False,
) -> tuple[int, str] | tuple[int, str, stitch_common.SelectionMutationSummary]:
    """Split non-self-mirrored CROSSES edges at their plane intersection *p*.

    All intersections are collected before any mutation, clustered, then
    applied with priority (existing vertex → existing edge split → new via
    edge_split); mutating mid-collection would shift later intersections.
    Self-mirrored CROSSES (endpoints are a mirror pair) are left untouched.

    Returns ``(stitched_edge_count, failure_reason)``. On failure the mesh may
    already be partially mutated; callers must roll back the whole stage.
    """

    tracker = stitch_common._SelectionMutationTracker()

    def _result(count: int, reason: str):
        if return_summary:
            return count, reason, tracker.finish(complete=not reason)
        return count, reason

    edges = [edge for edge in crosses_edges if edge.is_valid]
    for edge in edges:
        tracker.add_edge(edge)
    if not edges:
        return _result(0, "")

    # (i) Collect plane intersections before any mutation.
    records: list[tuple[bmesh.types.BMEdge, Vector, float]] = []
    for edge in edges:
        if stitch_common.is_self_mirrored_edge(edge, axis_index, tolerance):
            continue
        intersection = plane_intersection_of_edge(edge, axis_index)
        if intersection is None:
            return _result(0, "a cross-plane cut segment has no plane intersection")
        point, factor = intersection
        # Degenerate: intersection already at an endpoint within tol → treat as
        # already on-plane (no split); skip so POSITIVE/NEGATIVE reclassification
        # after a previous stitch can re-bucket cleanly.
        if any(coordinates_match(vertex.co, point, tolerance) for vertex in edge.verts):
            continue
        records.append((edge, point, factor))

    if not records:
        return _result(0, "")

    points = [point for _edge, point, _factor in records]
    clusters = stitch_common.cluster_points_by_tolerance(points, tolerance)

    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()

    stitched = 0
    for cluster_indices in clusters:
        representative = points[cluster_indices[0]]
        member_edges = [records[index][0] for index in cluster_indices]

        plan_vertex, host_split, reason = _plan_plane_stitch_vertex(
            bm,
            representative,
            member_edges,
            axis_index,
            tolerance,
        )
        if reason:
            return _result(stitched, reason)

        vertex = plan_vertex
        if host_split is not None:
            host_edge, factor = host_split
            try:
                _new_edge, vertex = bmesh.utils.edge_split(host_edge, host_edge.verts[0], factor)
            except (RuntimeError, ValueError) as exc:
                return _result(stitched, f"could not split a host edge at the knife stitch point: {exc}")
            tracker.add_edge(host_edge)
            tracker.add_edge(_new_edge)
            tracker.add_vertex(vertex)
        if vertex is None:
            # Priority (3): create p by splitting the lex-first member edge.
            seed = _lex_first_edge(member_edges, axis_index)
            intersection = plane_intersection_of_edge(seed, axis_index)
            if intersection is None:
                return _result(stitched, "a cross-plane cut segment has no plane intersection")
            _point, factor = intersection
            try:
                _new_edge, vertex = bmesh.utils.edge_split(seed, seed.verts[0], factor)
            except (RuntimeError, ValueError) as exc:
                return _result(stitched, f"could not split a cross-plane cut at the mirror plane: {exc}")
            tracker.add_edge(seed)
            tracker.add_edge(_new_edge)
            tracker.add_vertex(vertex)
            stitched += 1

        assert vertex is not None
        tracker.add_vertex(vertex)
        vertex.co = representative.copy()
        vertex.co[axis_index] = 0.0
        vertex.select = False

        for edge in member_edges:
            if not edge.is_valid:
                return _result(stitched, "a cross-plane cut edge was lost during p-stitch")
            tracker.add_edge(edge)
            if any(endpoint == vertex for endpoint in edge.verts):
                continue
            if any(coordinates_match(endpoint.co, vertex.co, tolerance) for endpoint in edge.verts):
                for endpoint in edge.verts:
                    if coordinates_match(endpoint.co, vertex.co, tolerance) and endpoint != vertex:
                        tracker.add_vertex(endpoint)
                        try:
                            # pointmerge keeps verts[0] as the survivor
                            # (bmo_pointmerge_exec); put the cluster representative first.
                            bmesh.ops.pointmerge(bm, verts=[vertex, endpoint], merge_co=vertex.co)
                        except (RuntimeError, ValueError) as exc:
                            return _result(stitched, f"could not merge plane-stitch vertices: {exc}")
                        break
                continue

            recomputed = plane_intersection_of_edge(edge, axis_index)
            if recomputed is None:
                return _result(stitched, "a cross-plane cut segment lost its plane intersection")
            _point, factor = recomputed
            try:
                _new_edge, new_vertex = bmesh.utils.edge_split(edge, edge.verts[0], factor)
            except (RuntimeError, ValueError) as exc:
                return _result(stitched, f"could not split a cross-plane cut at the mirror plane: {exc}")
            tracker.add_edge(edge)
            tracker.add_edge(_new_edge)
            tracker.add_vertex(new_vertex)
            new_vertex.co = vertex.co.copy()
            new_vertex.select = False
            if new_vertex != vertex:
                try:
                    # Survivor first: keeps *vertex* valid across ≥3 merges in
                    # one cluster (multi-segment X at p).
                    bmesh.ops.pointmerge(bm, verts=[vertex, new_vertex], merge_co=vertex.co)
                except (RuntimeError, ValueError) as exc:
                    return _result(stitched, f"could not unify plane-stitch vertices: {exc}")
                tracker.add_vertex(new_vertex)
            stitched += 1

    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.normal_update()
    return _result(stitched, "")


def _lex_first_edge(
    edges: Sequence[bmesh.types.BMEdge],
    axis_index: int = 0,
) -> bmesh.types.BMEdge:
    """Deterministic edge pick by mirror-invariant endpoint keys.

    Keys use |axis| on the mirror component so a seed and its mirror image
    sort equivalently. Remaining complete-orbit ties are acceptable best-effort.
    """

    return min(
        edges,
        key=lambda edge: (
            min(
                stitch_common._mirror_invariant_endpoint_key(edge.verts[0].co, axis_index),
                stitch_common._mirror_invariant_endpoint_key(edge.verts[1].co, axis_index),
            ),
            max(
                stitch_common._mirror_invariant_endpoint_key(edge.verts[0].co, axis_index),
                stitch_common._mirror_invariant_endpoint_key(edge.verts[1].co, axis_index),
            ),
        ),
    )


def _plan_plane_stitch_vertex(
    bm: bmesh.types.BMesh,
    representative: Vector,
    member_edges: Sequence[bmesh.types.BMEdge],
    axis_index: int,
    tolerance: float,
) -> tuple[
    bmesh.types.BMVert | None,
    tuple[bmesh.types.BMEdge, float] | None,
    str,
]:
    """Priority plan: (1) existing vertex (2) host edge split (3) member seed.

    Returns ``(existing_vertex, host_split_or_None, error)``. When both vertex
    and host_split are None and error is empty, the caller creates *p* by
    splitting a member CROSSES edge.
    """

    del axis_index  # reserved; host multi-candidate no longer uses mirror pairing

    # (1) Existing vertex within tol of the representative.
    exact_vertices = sorted(
        (
            ((vertex.co - representative).length, vertex.index, vertex)
            for vertex in bm.verts
            if vertex.is_valid and coordinates_match(vertex.co, representative, tolerance)
        ),
        key=lambda item: (item[0], item[1]),
    )
    if len(exact_vertices) > 1:
        return None, None, "ambiguous on-plane vertices within tolerance at a knife stitch point"
    if exact_vertices:
        return exact_vertices[0][2], None, ""

    # (2) Existing edges whose interior contains the representative.
    # Match Tolerance is the only threshold; a wider edge limit would adopt
    # unrelated edges.
    edge_limit = max(tolerance, 1.0e-9)
    host_edges: list[tuple[float, int, bmesh.types.BMEdge, float]] = []
    member_ids = {id(edge) for edge in member_edges}
    for edge in bm.edges:
        if not edge.is_valid or id(edge) in member_ids:
            continue
        distance, factor = stitch_common._point_segment_distance_and_factor(representative, edge)
        if not stitch_common._is_interior_edge_factor(factor, edge.calc_length(), tolerance):
            continue
        if distance > edge_limit:
            continue
        host_edges.append((distance, edge.index, edge, factor))

    if host_edges:
        host_edges.sort(key=lambda item: (item[0], item[1]))
        if len(host_edges) > 1:
            # Multi-candidate host edges are ambiguous.
            # - Nearest fallback is forbidden.
            # - A mirror pair both within tol is equidistant from on-plane p
            #   (degenerate); edge.index tie-break is also forbidden → decline.
            return None, None, "ambiguous host edges for a knife plane-stitch point"
        _distance, _index, host_edge, factor = host_edges[0]
        return None, (host_edge, factor), ""

    # (3) Caller creates via member edge_split.
    return None, None, ""
