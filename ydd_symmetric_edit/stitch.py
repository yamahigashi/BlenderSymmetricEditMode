from __future__ import annotations

import logging
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import bmesh
import mathutils.geometry
import numpy  # type: ignore
from mathutils import Vector
from mathutils.kdtree import KDTree

from ._types import (
    CarrierFrameMap,
    CarrierFrameSnapshot,
    FaceId,
    MirrorFaceMap,
    QuantizedCoordinate,
)
from .face_mapping import _canonical_carrier_frames
from .matching import (
    _chebyshev_distance_3d,
    _iter_quantized_neighborhood,
    _quantized_coordinate,
    coordinates_match,
    mirror_coordinate,
)
from .snapshot import EDGE_HIDDEN_LAYER, EDGE_ORIGINAL_LAYER, EDGE_SELECTION_LAYER, FACE_ID_LAYER, VERT_SELECTION_LAYER

_MIN_SIDE_LENGTH = 1.0e-9
_LOGGER = logging.getLogger(__name__)


def _edge_side(
    edge: bmesh.types.BMEdge,
    axis_index: int,
    tolerance: float,
) -> str:
    """Classify an edge relative to the mirror plane.

    Always returns one of POSITIVE / NEGATIVE / PLANE / CROSSES.
    """

    a = edge.verts[0].co[axis_index]
    b = edge.verts[1].co[axis_index]
    if a >= -tolerance and b >= -tolerance and max(a, b) > tolerance:
        return "POSITIVE"
    if a <= tolerance and b <= tolerance and min(a, b) < -tolerance:
        return "NEGATIVE"
    if abs(a) <= tolerance and abs(b) <= tolerance:
        return "PLANE"
    return "CROSSES"


def choose_source_side(
    path_edges: Iterable[bmesh.types.BMEdge],
    axis_index: int,
    tolerance: float,
    requested: str,
) -> tuple[str | None, int]:
    """Resolve AUTO and return ``(side, crossing_edge_count)``.

    Used by Loop Cut / Offset Edge Loop Cut (one-side selection). Knife no
    longer chooses a source side; see :func:`collect_knife_path_edges_by_side`.
    """

    positive_length = 0.0
    negative_length = 0.0
    crossing = 0
    for edge in path_edges:
        side = _edge_side(edge, axis_index, tolerance)
        if side == "POSITIVE":
            positive_length += edge.calc_length()
        elif side == "NEGATIVE":
            negative_length += edge.calc_length()
        elif side == "CROSSES":
            crossing += 1

    if requested in {"POSITIVE", "NEGATIVE"}:
        return requested, crossing
    if positive_length <= _MIN_SIDE_LENGTH and negative_length <= _MIN_SIDE_LENGTH:
        return None, crossing
    if positive_length >= negative_length:
        return "POSITIVE", crossing
    return "NEGATIVE", crossing


def _is_path_edge_by_markers(edge: bmesh.types.BMEdge, edge_layer, face_layer) -> bool:
    """True when *edge* is a native cut fragment.

    Tag==0 is the primary signal. Existing-edge splits inherit a non-zero parent
    tag, so also accept internal edges whose link faces all share one original
    FACE_ID (FACE_ID complement). Selection is intentionally not consulted.
    """

    if edge[edge_layer] == 0:
        return True
    return (
        face_layer is not None
        and len(edge.link_faces) >= 2
        and len({FaceId(int(face[face_layer])) for face in edge.link_faces}) == 1
    )


def native_path_edge_state(bm: bmesh.types.BMesh) -> Literal["PRESENT", "ABSENT", "UNKNOWN"]:
    """Classify whether *bm* carries evidence of an unprocessed native cut.

    ABSENT is only returned when both marker layers exist and a full scan
    found no path edge, so callers can treat missing layers or read failures
    as indeterminate rather than as a clean prepared baseline.
    """

    try:
        edge_layer = bm.edges.layers.int.get(EDGE_ORIGINAL_LAYER)
        face_layer = bm.faces.layers.int.get(FACE_ID_LAYER)
        if edge_layer is None or face_layer is None:
            return "UNKNOWN"
        for edge in bm.edges:
            if _is_path_edge_by_markers(edge, edge_layer, face_layer):
                return "PRESENT"
        return "ABSENT"
    except (AttributeError, ReferenceError, RuntimeError):
        return "UNKNOWN"


def _discover_path_edges(
    bm: bmesh.types.BMesh,
    *,
    selected_only: bool = False,
) -> list[bmesh.types.BMEdge]:
    """Discover native path edges created by the last cut tool.

    Loop Cut and Offset Edge Loop Cut expose their complete native result as
    the current edge selection. This is more authoritative for those tools
    than CustomData inheritance on complex rings. Knife strokes can honor
    "Select Result" being disabled, so their marker-based path stays intact.

    Both branches use the same novelty test (tag==0 or FACE_ID complement).
    *selected_only* only adds the selection filter for Loop Cut / Offset;
    Knife (selected_only=False) must still recover inherited-tag CROSSES
    fragments so the whole-stage decline cannot be skipped.
    """

    edge_layer = bm.edges.layers.int.get(EDGE_ORIGINAL_LAYER)
    if edge_layer is None:
        return []

    face_layer = bm.faces.layers.int.get(FACE_ID_LAYER)
    result: list[bmesh.types.BMEdge] = []
    for edge in bm.edges:
        if selected_only and not edge.select:
            continue
        if edge[edge_layer] == 0:
            result.append(edge)
            continue
        if face_layer is None:
            continue
        faces = edge.link_faces
        count = len(faces)
        if count < 2:
            continue
        first = faces[0][face_layer]
        if count == 2:
            if faces[1][face_layer] == first:
                result.append(edge)
            continue
        if all(face[face_layer] == first for face in faces):
            result.append(edge)
    return result


def path_ring_includes_pre_hidden_edges(bm: bmesh.types.BMesh) -> bool:
    """True when the Loop Cut / Offset ring includes a pre-hidden edge.

    Native ``loopcut`` skips hidden ring edges and yields a *partial* (open)
    path. A closed selected path means the cut ring was complete, so unrelated
    hidden geometry on another ring must not decline. An open path together
    with a pre-hidden edge in the face-neighbourhood of a path endpoint is the
    partial-ring case that must decline.
    """

    edge_hidden = bm.edges.layers.int.get(EDGE_HIDDEN_LAYER)
    edge_layer = bm.edges.layers.int.get(EDGE_ORIGINAL_LAYER)
    if edge_hidden is None or edge_layer is None:
        return False

    face_layer = bm.faces.layers.int.get(FACE_ID_LAYER)
    path_edges = [edge for edge in bm.edges if edge.select and _is_path_edge_by_markers(edge, edge_layer, face_layer)]
    if not path_edges:
        return False

    degree: dict[int, int] = defaultdict(int)
    vert_by_key: dict[int, bmesh.types.BMVert] = {}
    for edge in path_edges:
        for vertex in edge.verts:
            key = hash(vertex)
            degree[key] += 1
            vert_by_key[key] = vertex
    # Closed loop: every path vertex is incident to exactly two path edges.
    if degree and all(count == 2 for count in degree.values()):
        return False

    endpoint_verts = [vert_by_key[key] for key, count in degree.items() if count == 1]
    if not endpoint_verts:
        return False

    # BFS over faces around path endpoints: the skipped (hidden) ring edges
    # sit in the gap adjacent to the open ends.
    seen_faces: set[int] = set()
    face_queue: list[bmesh.types.BMFace] = []
    for vertex in endpoint_verts:
        for face in vertex.link_faces:
            if face.is_valid and face.index not in seen_faces:
                seen_faces.add(face.index)
                face_queue.append(face)

    # Expand one adjacency step so a one-edge gap still reaches the hidden edge.
    for face in list(face_queue):
        for edge in face.edges:
            for other in edge.link_faces:
                if other.is_valid and other.index not in seen_faces:
                    seen_faces.add(other.index)
                    face_queue.append(other)

    for face in face_queue:
        for edge in face.edges:
            if edge.is_valid and edge[edge_hidden]:
                return True
    return False


def classify_path_edges_by_side(
    path_edges: Iterable[bmesh.types.BMEdge],
    axis_index: int,
    tolerance: float,
) -> dict[str, list[bmesh.types.BMEdge]]:
    """Bucket path edges into POSITIVE / NEGATIVE / PLANE / CROSSES."""

    by_side: dict[str, list[bmesh.types.BMEdge]] = {
        "POSITIVE": [],
        "NEGATIVE": [],
        "PLANE": [],
        "CROSSES": [],
    }
    for edge in path_edges:
        by_side[_edge_side(edge, axis_index, tolerance)].append(edge)
    return by_side


_KnifePathEdgeCacheEntry = tuple[
    int,
    tuple[float, float, float],
    tuple[float, float, float],
]
_KnifePathEdgeCache = tuple[_KnifePathEdgeCacheEntry, ...]


def capture_knife_path_edge_cache(
    bm: bmesh.types.BMesh,
    path_edges: Iterable[bmesh.types.BMEdge],
) -> _KnifePathEdgeCache | None:
    """Capture metadata for reclassifying an unchanged Knife path.

    No ``BMEdge`` proxy is retained: the edit BMesh may be rebuilt between
    calls.  Reuse verifies both the edge index and endpoint coordinates.
    """

    bm.edges.ensure_lookup_table()
    bm.edges.index_update()
    entries: list[_KnifePathEdgeCacheEntry] = []
    for edge in path_edges:
        index = int(edge.index)
        if index < 0:
            return None
        first = _coordinate_tuple(edge.verts[0].co)
        second = _coordinate_tuple(edge.verts[1].co)
        if second < first:
            first, second = second, first
        entries.append((index, first, second))
    return tuple(entries)


def reclassify_knife_path_edge_cache(
    bm: bmesh.types.BMesh,
    axis_index: int,
    tolerance: float,
    cache: _KnifePathEdgeCache | None,
) -> tuple[dict[str, list[bmesh.types.BMEdge]], int] | None:
    """Reclassify cached path edges, or return ``None`` when unverifiable."""

    if cache is None:
        return None
    bm.edges.ensure_lookup_table()
    path_edges: list[bmesh.types.BMEdge] = []
    for index, expected_first, expected_second in cache:
        if index >= len(bm.edges):
            return None
        edge = bm.edges[index]
        if not edge.is_valid:
            return None
        first = _coordinate_tuple(edge.verts[0].co)
        second = _coordinate_tuple(edge.verts[1].co)
        if second < first:
            first, second = second, first
        if first != expected_first or second != expected_second:
            return None
        path_edges.append(edge)
    by_side = classify_path_edges_by_side(path_edges, axis_index, tolerance)
    return by_side, len(path_edges)


def collect_knife_path_edges_by_side(
    bm: bmesh.types.BMesh,
    axis_index: int,
    tolerance: float,
) -> tuple[dict[str, list[bmesh.types.BMEdge]], int]:
    """Classify every new Knife path edge without choosing a source side.

    Both POSITIVE and NEGATIVE buckets are mirrored toward each other. PLANE
    edges are shared. CROSSES edges are p-stitched before the half-edges join
    the POSITIVE/NEGATIVE mirror path.

    Returns ``(by_side, total_path_edge_count)``.
    """

    all_path_edges = _discover_path_edges(bm, selected_only=False)
    by_side = classify_path_edges_by_side(all_path_edges, axis_index, tolerance)
    return by_side, len(all_path_edges)


def patch_knife_path_edges_by_side(
    bm: bmesh.types.BMesh,
    previous_by_side: Mapping[str, Sequence[bmesh.types.BMEdge]],
    cache: _KnifePathEdgeCache | None,
    summary: CrossingMutationSummary,
    axis_index: int,
    tolerance: float,
) -> tuple[dict[str, list[bmesh.types.BMEdge]], int] | None:
    """Patch a Knife collect result from a crossings mutation summary.

    ``None`` means the strict §I-3b conditions were not met and callers must
    perform the ordinary full collect.  The returned bucket order is formed
    by replacing each source edge in-place and appending descendants to its
    original bucket.
    """

    if cache is None or not summary.edges:
        return None
    if summary.removed_edge_count or summary.removed_face_count:
        return None
    bm.edges.ensure_lookup_table()
    previous = [
        edge for side in ("POSITIVE", "NEGATIVE", "CROSSES", "PLANE") for edge in previous_by_side.get(side, ())
    ]
    if len(previous) != len(cache):
        return None
    if len(set(cache)) != len(cache):
        return None
    cache_positions = {hash(edge): position for position, edge in enumerate(previous)}
    replacement: dict[str, list[bmesh.types.BMEdge]] = {
        side: list(previous_by_side.get(side, ())) for side in ("POSITIVE", "NEGATIVE", "CROSSES", "PLANE")
    }
    bucket_positions: dict[tuple[str, int], int] = {}
    for side, bucket in replacement.items():
        for local_position, edge in enumerate(bucket):
            cache_position = cache_positions.get(hash(edge))
            if cache_position is None or (side, cache_position) in bucket_positions:
                return None
            bucket_positions[(side, cache_position)] = local_position
    edge_layer = bm.edges.layers.int.get(EDGE_ORIGINAL_LAYER)
    face_layer = bm.faces.layers.int.get(FACE_ID_LAYER)
    if edge_layer is None:
        return None
    previous_by_id = {
        hash(edge): (edge, side, position)
        for side, bucket in replacement.items()
        for position, edge in enumerate(bucket)
    }
    expected_side_by_edge: dict[int, tuple[bmesh.types.BMEdge, str]] = {}
    for side, bucket in replacement.items():
        for edge in bucket:
            if not edge.is_valid or not _is_path_edge_by_markers(edge, edge_layer, face_layer):
                _LOGGER.warning("incremental Knife collect declined: cached path-edge membership changed")
                return None
            if _edge_side(edge, axis_index, tolerance) != side:
                _LOGGER.warning("incremental Knife collect declined: cached path-edge side changed")
                return None
            expected_side_by_edge[hash(edge)] = (edge, side)
    for mutation in summary.edges:
        if mutation.endpoint_reused or mutation.pointmerged:
            return None
        source_info = previous_by_id.get(mutation.source_edge_id)
        source = source_info[0] if source_info is not None else None
        source_position = mutation.cache_position
        if (
            source is None
            or source_position is None
            or source_position < 0
            or source_position >= len(previous)
            or previous[source_position] is not source
        ):
            return None
        side = source_info[1] if source_info is not None else None
        if side is None or not mutation.final_edges:
            return None
        final_edges = [edge for edge in mutation.final_edges if edge.is_valid]
        if len(final_edges) != len(mutation.final_edges):
            return None
        if any(not _is_path_edge_by_markers(edge, edge_layer, face_layer) for edge in final_edges):
            _LOGGER.warning("incremental Knife collect declined: split path-edge membership changed")
            return None
        if any(_edge_side(edge, axis_index, tolerance) != side for edge in final_edges):
            return None
        if source not in final_edges or len({hash(edge) for edge in final_edges}) != len(final_edges):
            return None
        for edge in final_edges:
            edge_id = hash(edge)
            prior = expected_side_by_edge.get(edge_id)
            if prior is not None and prior[1] != side:
                _LOGGER.warning("incremental Knife collect declined: mutation side assignment conflicts")
                return None
            expected_side_by_edge[edge_id] = (edge, side)
        # Remove the old record by its flattened cache position.  Descendants
        # are appended below in live BMesh order after all source replacements.
        bucket = replacement[side]
        local_position = bucket_positions.get((side, source_position))
        if local_position is None or bucket[local_position] is not source:
            return None
        bucket[local_position] = source

    if summary.unexpected_topology_change:
        return None
    tail = [bm.edges[index] for index in range(summary.pre_apply_edge_count, len(bm.edges))]
    source_ids = set(previous_by_id)
    mutation_non_sources = [
        edge
        for mutation in summary.edges
        for edge in mutation.final_edges
        if edge.is_valid and hash(edge) != mutation.source_edge_id
    ]
    if any(edge not in tail for edge in mutation_non_sources):
        _LOGGER.warning("incremental Knife collect declined: a split edge is not after pre-apply edges")
        return None
    final_non_sources = [edge for edge, _side in expected_side_by_edge.values() if hash(edge) not in source_ids]
    if len(tail) != len(final_non_sources) or any(edge not in tail for edge in final_non_sources):
        _LOGGER.warning("incremental Knife collect declined: a split edge is not after pre-apply edges")
        return None
    if len(expected_side_by_edge) != len(previous) + len(tail):
        _LOGGER.warning("incremental Knife collect declined: mutation edge set is incomplete")
        return None
    # FACE_ID complement closure is checked after all source mutations so
    # cross-source new half-edges have their expected side available.
    tracked_ids = set(expected_side_by_edge)
    closure_faces = []
    closure_face_ids: set[int] = set()

    def add_closure_face(face: bmesh.types.BMFace) -> None:
        face_id = hash(face)
        if face_id not in closure_face_ids:
            closure_face_ids.add(face_id)
            closure_faces.append(face)

    for mutation in summary.edges:
        for face in mutation.pre_faces:
            if not face.is_valid:
                _LOGGER.warning("incremental Knife collect declined: a pre-split FACE_ID face disappeared")
                return None
            add_closure_face(face)
        for edge in mutation.final_edges:
            if edge.is_valid:
                for face in edge.link_faces:
                    if face.is_valid:
                        add_closure_face(face)
    if face_layer is not None:
        for face in closure_faces:
            for closed_edge in face.edges:
                if not closed_edge.is_valid:
                    continue
                closed_id = hash(closed_edge)
                is_path = _is_path_edge_by_markers(closed_edge, edge_layer, face_layer)
                if closed_id not in tracked_ids:
                    if is_path:
                        _LOGGER.warning(
                            "incremental Knife collect declined: FACE_ID closure contains an untracked path edge"
                        )
                        return None
                    continue
                expected = expected_side_by_edge[closed_id][1]
                if not is_path or _edge_side(closed_edge, axis_index, tolerance) != expected:
                    _LOGGER.warning("incremental Knife collect declined: FACE_ID closure membership or side changed")
                    return None

    for side in ("POSITIVE", "NEGATIVE", "CROSSES", "PLANE"):
        replacement[side].extend(
            edge for edge in tail if expected_side_by_edge.get(hash(edge), (None, None))[1] == side
        )

    patched = {side: list(replacement[side]) for side in ("POSITIVE", "NEGATIVE", "CROSSES", "PLANE")}
    all_edges = [edge for side in ("POSITIVE", "NEGATIVE", "CROSSES", "PLANE") for edge in patched[side]]
    if len({hash(edge) for edge in all_edges}) != len(all_edges):
        return None
    return patched, len(all_edges)


def is_self_mirrored_edge(
    edge: bmesh.types.BMEdge,
    axis_index: int,
    tolerance: float,
) -> bool:
    """True when the edge endpoints are a mirror pair (ρ(s) = s, no X needed)."""

    a = edge.verts[0].co
    b = edge.verts[1].co
    return coordinates_match(a, mirror_coordinate(b, axis_index), tolerance) and coordinates_match(
        b,
        mirror_coordinate(a, axis_index),
        tolerance,
    )


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


def cluster_points_by_tolerance(
    points: Sequence[Vector],
    tolerance: float,
) -> list[list[int]]:
    """Lex-order scan clustering: absorb within tol of the representative.

    Returns clusters as lists of indices into *points*. The first index of each
    cluster is the representative (lex-first member). Member-to-rep distance is
    at most *tolerance*; diameter among members can reach 2·tol.
    """

    order = sorted(
        range(len(points)),
        key=lambda index: (
            float(points[index][0]),
            float(points[index][1]),
            float(points[index][2]),
        ),
    )
    clusters: list[list[int]] = []
    for index in order:
        placed = False
        for cluster in clusters:
            representative = points[cluster[0]]
            if coordinates_match(points[index], representative, tolerance):
                cluster.append(index)
                placed = True
                break
        if not placed:
            clusters.append([index])
    return clusters


def apply_crosses_p_stitch(
    bm: bmesh.types.BMesh,
    crosses_edges: Iterable[bmesh.types.BMEdge],
    axis_index: int,
    tolerance: float,
) -> tuple[int, str]:
    """Split non-self-mirrored CROSSES edges at their plane intersection *p*.

    All intersections are collected before any mutation, clustered, then
    applied with priority (existing vertex → existing edge split → new via
    edge_split); mutating mid-collection would shift later intersections.
    Self-mirrored CROSSES (endpoints are a mirror pair) are left untouched.

    Returns ``(stitched_edge_count, failure_reason)``. On failure the mesh may
    already be partially mutated; callers must roll back the whole stage.
    """

    edges = [edge for edge in crosses_edges if edge.is_valid]
    if not edges:
        return 0, ""

    # (i) Collect plane intersections before any mutation.
    records: list[tuple[bmesh.types.BMEdge, Vector, float]] = []
    for edge in edges:
        if is_self_mirrored_edge(edge, axis_index, tolerance):
            continue
        intersection = plane_intersection_of_edge(edge, axis_index)
        if intersection is None:
            return 0, "a cross-plane cut segment has no plane intersection"
        point, factor = intersection
        # Degenerate: intersection already at an endpoint within tol → treat as
        # already on-plane (no split); skip so POSITIVE/NEGATIVE reclassification
        # after a previous stitch can re-bucket cleanly.
        if any(coordinates_match(vertex.co, point, tolerance) for vertex in edge.verts):
            continue
        records.append((edge, point, factor))

    if not records:
        return 0, ""

    points = [point for _edge, point, _factor in records]
    clusters = cluster_points_by_tolerance(points, tolerance)

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
            return stitched, reason

        vertex = plan_vertex
        if host_split is not None:
            host_edge, factor = host_split
            try:
                _new_edge, vertex = bmesh.utils.edge_split(host_edge, host_edge.verts[0], factor)
            except (RuntimeError, ValueError) as exc:
                return stitched, f"could not split a host edge at the knife stitch point: {exc}"
        if vertex is None:
            # Priority (3): create p by splitting the lex-first member edge.
            seed = _lex_first_edge(member_edges, axis_index)
            intersection = plane_intersection_of_edge(seed, axis_index)
            if intersection is None:
                return stitched, "a cross-plane cut segment has no plane intersection"
            _point, factor = intersection
            try:
                _new_edge, vertex = bmesh.utils.edge_split(seed, seed.verts[0], factor)
            except (RuntimeError, ValueError) as exc:
                return stitched, f"could not split a cross-plane cut at the mirror plane: {exc}"
            stitched += 1

        assert vertex is not None
        vertex.co = representative.copy()
        vertex.co[axis_index] = 0.0
        vertex.select = False

        for edge in member_edges:
            if not edge.is_valid:
                return stitched, "a cross-plane cut edge was lost during p-stitch"
            if any(endpoint == vertex for endpoint in edge.verts):
                continue
            if any(coordinates_match(endpoint.co, vertex.co, tolerance) for endpoint in edge.verts):
                for endpoint in edge.verts:
                    if coordinates_match(endpoint.co, vertex.co, tolerance) and endpoint != vertex:
                        try:
                            # pointmerge keeps verts[0] as the survivor
                            # (bmo_pointmerge_exec); put the cluster representative first.
                            bmesh.ops.pointmerge(bm, verts=[vertex, endpoint], merge_co=vertex.co)
                        except (RuntimeError, ValueError) as exc:
                            return stitched, f"could not merge plane-stitch vertices: {exc}"
                        break
                continue

            recomputed = plane_intersection_of_edge(edge, axis_index)
            if recomputed is None:
                return stitched, "a cross-plane cut segment lost its plane intersection"
            _point, factor = recomputed
            try:
                _new_edge, new_vertex = bmesh.utils.edge_split(edge, edge.verts[0], factor)
            except (RuntimeError, ValueError) as exc:
                return stitched, f"could not split a cross-plane cut at the mirror plane: {exc}"
            new_vertex.co = vertex.co.copy()
            new_vertex.select = False
            if new_vertex != vertex:
                try:
                    # Survivor first: keeps *vertex* valid across ≥3 merges in
                    # one cluster (multi-segment X at p).
                    bmesh.ops.pointmerge(bm, verts=[vertex, new_vertex], merge_co=vertex.co)
                except (RuntimeError, ValueError) as exc:
                    return stitched, f"could not unify plane-stitch vertices: {exc}"
            stitched += 1

    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.normal_update()
    return stitched, ""


def _mirror_invariant_endpoint_key(
    co,
    axis_index: int,
) -> tuple[float, float, float]:
    """Canonical endpoint key with |axis| so ρ-related seeds share the key.

    Absolute value on the mirror axis makes (M,I) and (ρM,ρI) order-equivalent
    for seed selection. Full orbit ties are allowed as best-effort attribute
    inheritance (shape keys etc.).
    """

    components = [float(co[0]), float(co[1]), float(co[2])]
    components[axis_index] = abs(components[axis_index])
    return (components[0], components[1], components[2])


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
                _mirror_invariant_endpoint_key(edge.verts[0].co, axis_index),
                _mirror_invariant_endpoint_key(edge.verts[1].co, axis_index),
            ),
            max(
                _mirror_invariant_endpoint_key(edge.verts[0].co, axis_index),
                _mirror_invariant_endpoint_key(edge.verts[1].co, axis_index),
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
        distance, factor = _point_segment_distance_and_factor(representative, edge)
        if not _is_interior_edge_factor(factor, edge.calc_length(), tolerance):
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


@dataclass(frozen=True, slots=True)
class _MirroredSegmentIntersection:
    kind: Literal["NONE", "PROPER", "ENDPOINT_INTERIOR", "ENDPOINT_ENDPOINT", "COLLINEAR"]
    factor_a: float = 0.0
    factor_b: float = 0.0
    point_a: Vector | None = None
    point_b: Vector | None = None
    coordinate: Vector | None = None
    endpoint_a: int | None = None
    endpoint_b: int | None = None


@dataclass(frozen=True, slots=True)
class _MirroredPathOccurrence:
    edge: bmesh.types.BMEdge
    edge_id: int
    factor: float
    endpoint_index: int | None
    edge_key: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
    ]


@dataclass(frozen=True, slots=True)
class _RawMirroredPathCrossing:
    coordinate: Vector
    positive: tuple[_MirroredPathOccurrence, ...]
    negative: tuple[_MirroredPathOccurrence, ...]


@dataclass(frozen=True, slots=True)
class _MirroredPathCrossingCluster:
    positive_coordinate: Vector
    negative_coordinate: Vector
    positive: tuple[_MirroredPathOccurrence, ...]
    negative: tuple[_MirroredPathOccurrence, ...]
    tolerance: float


def _edge_survivor_key(
    edge: bmesh.types.BMEdge,
    axis_index: int,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    first = _mirror_invariant_endpoint_key(edge.verts[0].co, axis_index)
    second = _mirror_invariant_endpoint_key(edge.verts[1].co, axis_index)
    return (first, second) if first <= second else (second, first)


def _project_to_carrier(
    coordinate: Vector,
    frame: CarrierFrameSnapshot,
) -> tuple[float, float]:
    assert frame.normal is not None and frame.basis_u is not None
    origin = Vector(frame.origin.as_tuple())
    normal = Vector(frame.normal.as_tuple())
    basis_u = Vector(frame.basis_u.as_tuple())
    basis_w = normal.cross(basis_u)
    delta = coordinate - origin
    return float(delta.dot(basis_u)), float(delta.dot(basis_w))


def _cross_2d(first: tuple[float, float], second: tuple[float, float]) -> float:
    return first[0] * second[1] - first[1] * second[0]


def _distance_to_line_2d(
    point: tuple[float, float],
    first: tuple[float, float],
    second: tuple[float, float],
) -> float:
    delta = (second[0] - first[0], second[1] - first[1])
    length = math.hypot(delta[0], delta[1])
    if length == 0.0:
        return math.hypot(point[0] - first[0], point[1] - first[1])
    offset = (point[0] - first[0], point[1] - first[1])
    return abs(_cross_2d(offset, delta)) / length


def _matching_endpoint_index(
    point: Vector,
    endpoints: tuple[Vector, Vector],
    tolerance: float,
) -> int | None:
    matches = [index for index, endpoint in enumerate(endpoints) if coordinates_match(point, endpoint, tolerance)]
    if not matches:
        return None
    return min(matches, key=lambda index: _coordinate_tuple(endpoints[index]))


def _classify_segment_contact(
    factor_a: float,
    factor_b: float,
    endpoints_a: tuple[Vector, Vector],
    endpoints_b: tuple[Vector, Vector],
    tolerance: float,
) -> _MirroredSegmentIntersection:
    point_a = endpoints_a[0].lerp(endpoints_a[1], factor_a)
    point_b = endpoints_b[0].lerp(endpoints_b[1], factor_b)
    endpoint_a = _matching_endpoint_index(point_a, endpoints_a, tolerance)
    endpoint_b = _matching_endpoint_index(point_b, endpoints_b, tolerance)
    if endpoint_a is not None and endpoint_b is not None:
        kind = "ENDPOINT_ENDPOINT"
    elif endpoint_a is not None or endpoint_b is not None:
        kind = "ENDPOINT_INTERIOR"
    else:
        kind = "PROPER"
    return _MirroredSegmentIntersection(
        kind=kind,
        factor_a=factor_a,
        factor_b=factor_b,
        point_a=point_a,
        point_b=point_b,
        coordinate=(point_a + point_b) * 0.5,
        endpoint_a=endpoint_a,
        endpoint_b=endpoint_b,
    )


def _intersect_segments_on_carrier(
    endpoints_a: tuple[Vector, Vector],
    endpoints_b: tuple[Vector, Vector],
    frame: CarrierFrameSnapshot,
    tolerance: float,
) -> _MirroredSegmentIntersection:
    projected_a = tuple(_project_to_carrier(point, frame) for point in endpoints_a)
    projected_b = tuple(_project_to_carrier(point, frame) for point in endpoints_b)
    a0, a1 = projected_a
    b0, b1 = projected_b
    delta_a = (a1[0] - a0[0], a1[1] - a0[1])
    delta_b = (b1[0] - b0[0], b1[1] - b0[1])
    length_a = math.hypot(delta_a[0], delta_a[1])
    length_b = math.hypot(delta_b[0], delta_b[1])

    if length_a == 0.0 and length_b == 0.0:
        if math.hypot(a0[0] - b0[0], a0[1] - b0[1]) > tolerance:
            return _MirroredSegmentIntersection("NONE")
        return _classify_segment_contact(0.0, 0.0, endpoints_a, endpoints_b, tolerance)
    if length_a == 0.0:
        distance, factor_b = _point_segment_distance_2d(a0, b0, b1)
        if distance > tolerance:
            return _MirroredSegmentIntersection("NONE")
        return _classify_segment_contact(0.0, factor_b, endpoints_a, endpoints_b, tolerance)
    if length_b == 0.0:
        distance, factor_a = _point_segment_distance_2d(b0, a0, a1)
        if distance > tolerance:
            return _MirroredSegmentIntersection("NONE")
        return _classify_segment_contact(factor_a, 0.0, endpoints_a, endpoints_b, tolerance)

    collinear = (
        abs(_cross_2d(delta_a, delta_b)) <= tolerance * max(length_a, length_b)
        and _distance_to_line_2d(a0, b0, b1) <= tolerance
        and _distance_to_line_2d(a1, b0, b1) <= tolerance
        and _distance_to_line_2d(b0, a0, a1) <= tolerance
        and _distance_to_line_2d(b1, a0, a1) <= tolerance
    )
    if collinear:
        direction = delta_a if length_a >= length_b else delta_b
        direction_length = math.hypot(direction[0], direction[1])
        unit = (direction[0] / direction_length, direction[1] / direction_length)

        def scalar(point: tuple[float, float]) -> float:
            return point[0] * unit[0] + point[1] * unit[1]

        a_values = (scalar(a0), scalar(a1))
        b_values = (scalar(b0), scalar(b1))
        overlap_start = max(min(a_values), min(b_values))
        overlap_end = min(max(a_values), max(b_values))
        if overlap_end < overlap_start:
            return _MirroredSegmentIntersection("NONE")
        if overlap_end > overlap_start:
            return _MirroredSegmentIntersection("COLLINEAR")
        denominator_a = a_values[1] - a_values[0]
        denominator_b = b_values[1] - b_values[0]
        if denominator_a == 0.0 or denominator_b == 0.0:
            return _MirroredSegmentIntersection("COLLINEAR")
        factor_a = (overlap_start - a_values[0]) / denominator_a
        factor_b = (overlap_start - b_values[0]) / denominator_b
        return _classify_segment_contact(factor_a, factor_b, endpoints_a, endpoints_b, tolerance)

    denominator = _cross_2d(delta_a, delta_b)
    if denominator == 0.0:
        return _MirroredSegmentIntersection("NONE")
    offset = (b0[0] - a0[0], b0[1] - a0[1])
    factor_a = _cross_2d(offset, delta_b) / denominator
    factor_b = _cross_2d(offset, delta_a) / denominator
    if factor_a < 0.0 or factor_a > 1.0 or factor_b < 0.0 or factor_b > 1.0:
        return _MirroredSegmentIntersection("NONE")
    return _classify_segment_contact(factor_a, factor_b, endpoints_a, endpoints_b, tolerance)


def _point_segment_distance_2d(
    point: tuple[float, float],
    first: tuple[float, float],
    second: tuple[float, float],
) -> tuple[float, float]:
    delta = (second[0] - first[0], second[1] - first[1])
    length_squared = delta[0] * delta[0] + delta[1] * delta[1]
    if length_squared == 0.0:
        return math.hypot(point[0] - first[0], point[1] - first[1]), 0.0
    factor = max(
        0.0,
        min(
            1.0,
            ((point[0] - first[0]) * delta[0] + (point[1] - first[1]) * delta[1]) / length_squared,
        ),
    )
    nearest = (first[0] + factor * delta[0], first[1] + factor * delta[1])
    return math.hypot(point[0] - nearest[0], point[1] - nearest[1]), factor


def _edge_carrier_ids(edge: bmesh.types.BMEdge, face_layer) -> set[FaceId]:
    return {FaceId(int(face[face_layer])) for face in edge.link_faces if face.is_valid}


def _mirrored_carrier_ids(
    carrier_ids: set[FaceId],
    mirror_face_ids: MirrorFaceMap,
) -> set[FaceId]:
    return {mirrored for face_id in carrier_ids if (mirrored := mirror_face_ids.get(face_id)) is not None}


def _intersection_results_match(
    first: _MirroredSegmentIntersection,
    second: _MirroredSegmentIntersection,
    tolerance: float,
) -> bool:
    if first.kind != second.kind:
        return False
    if first.kind in {"NONE", "COLLINEAR"}:
        return True
    if first.coordinate is None or second.coordinate is None:
        return first.coordinate is second.coordinate
    return (
        first.endpoint_a == second.endpoint_a
        and first.endpoint_b == second.endpoint_b
        and coordinates_match(first.coordinate, second.coordinate, tolerance)
    )


def _occurrence(
    edge: bmesh.types.BMEdge,
    factor: float,
    endpoint_index: int | None,
    axis_index: int,
) -> _MirroredPathOccurrence:
    return _MirroredPathOccurrence(
        edge=edge,
        edge_id=hash(edge),
        factor=float(factor),
        endpoint_index=endpoint_index,
        edge_key=_edge_survivor_key(edge, axis_index),
    )


def _deduplicate_occurrences(
    occurrences: Iterable[_MirroredPathOccurrence],
) -> tuple[_MirroredPathOccurrence, ...]:
    by_edge: dict[int, _MirroredPathOccurrence] = {}
    for occurrence in occurrences:
        key = occurrence.edge_id
        current = by_edge.get(key)
        if current is None:
            by_edge[key] = occurrence
            continue
        if current.endpoint_index is None and occurrence.endpoint_index is not None:
            by_edge[key] = occurrence
        elif (
            current.endpoint_index is None and occurrence.endpoint_index is None and occurrence.factor < current.factor
        ):
            by_edge[key] = occurrence
    return tuple(sorted(by_edge.values(), key=lambda occurrence: (occurrence.edge_key, occurrence.factor)))


def plan_mirrored_path_crossings(
    bm: bmesh.types.BMesh,
    by_side: dict[str, Sequence[bmesh.types.BMEdge]],
    axis_index: int,
    tolerance: float,
    mirror_face_ids: MirrorFaceMap,
    carrier_frames: CarrierFrameMap,
) -> tuple[list[_MirroredPathCrossingCluster], str]:
    """Plan mirrored path intersections without modifying *bm*."""

    face_layer = bm.faces.layers.int.get(FACE_ID_LAYER)
    if face_layer is None:
        return [], "temporary face markers are missing"

    candidates: dict[int, bmesh.types.BMEdge] = {}
    for side in ("POSITIVE", "NEGATIVE", "CROSSES"):
        for edge in by_side.get(side, ()):
            if edge.is_valid:
                candidates[hash(edge)] = edge

    positive: list[bmesh.types.BMEdge] = []
    negative: list[bmesh.types.BMEdge] = []
    fixed: list[bmesh.types.BMEdge] = []
    for edge in candidates.values():
        if is_self_mirrored_edge(edge, axis_index, tolerance):
            fixed.append(edge)
            continue
        midpoint = (float(edge.verts[0].co[axis_index]) + float(edge.verts[1].co[axis_index])) * 0.5
        if midpoint > 0.0:
            positive.append(edge)
        elif midpoint < 0.0:
            negative.append(edge)

    raw_crossings: list[_RawMirroredPathCrossing] = []

    def evaluate_pair(
        canonical_a: tuple[Vector, Vector],
        canonical_b: tuple[Vector, Vector],
        canonical_carriers: set[FaceId],
    ) -> tuple[_MirroredSegmentIntersection | None, str]:
        frames, reason = _canonical_carrier_frames(
            canonical_carriers,
            mirror_face_ids,
            carrier_frames,
            axis_index,
        )
        if reason:
            return None, reason
        if not frames:
            return None, ""

        accepted: _MirroredSegmentIntersection | None = None
        for frame in frames:
            result = _intersect_segments_on_carrier(
                canonical_a,
                canonical_b,
                frame,
                tolerance,
            )
            if result.kind == "COLLINEAR":
                complete = (
                    coordinates_match(canonical_a[0], canonical_b[0], tolerance)
                    and coordinates_match(canonical_a[1], canonical_b[1], tolerance)
                ) or (
                    coordinates_match(canonical_a[0], canonical_b[1], tolerance)
                    and coordinates_match(canonical_a[1], canonical_b[0], tolerance)
                )
                if not complete:
                    return None, "mirrored cut segments partially overlap on a carrier"
            if result.coordinate is not None:
                assert result.point_a is not None and result.point_b is not None
                if (result.point_a - result.point_b).length > 2.0 * frame.deviation + 2.0 * tolerance:
                    return None, "a mirrored cut intersection exceeds its carrier non-planarity guard"
            if accepted is None:
                accepted = result
            elif not _intersection_results_match(accepted, result, tolerance):
                return None, "mirrored cut carriers produce ambiguous intersection results"
        return accepted, ""

    for positive_edge in positive:
        positive_endpoints = (positive_edge.verts[0].co.copy(), positive_edge.verts[1].co.copy())
        positive_carriers = _edge_carrier_ids(positive_edge, face_layer)
        for negative_edge in negative:
            negative_endpoints = (
                mirror_coordinate(negative_edge.verts[0].co, axis_index),
                mirror_coordinate(negative_edge.verts[1].co, axis_index),
            )
            if (
                coordinates_match(positive_endpoints[0], negative_endpoints[0], tolerance)
                and coordinates_match(positive_endpoints[1], negative_endpoints[1], tolerance)
            ) or (
                coordinates_match(positive_endpoints[0], negative_endpoints[1], tolerance)
                and coordinates_match(positive_endpoints[1], negative_endpoints[0], tolerance)
            ):
                continue
            negative_carriers = _mirrored_carrier_ids(
                _edge_carrier_ids(negative_edge, face_layer),
                mirror_face_ids,
            )
            result, reason = evaluate_pair(
                positive_endpoints,
                negative_endpoints,
                positive_carriers.intersection(negative_carriers),
            )
            if reason:
                return [], reason
            if result is None or result.kind in {"NONE", "COLLINEAR"}:
                continue
            assert result.coordinate is not None
            raw_crossings.append(
                _RawMirroredPathCrossing(
                    coordinate=result.coordinate,
                    positive=(_occurrence(positive_edge, result.factor_a, result.endpoint_a, axis_index),),
                    negative=(_occurrence(negative_edge, result.factor_b, result.endpoint_b, axis_index),),
                )
            )

    for moving_edges, moving_is_positive in ((positive, True), (negative, False)):
        for moving_edge in moving_edges:
            if moving_is_positive:
                moving_endpoints = (moving_edge.verts[0].co.copy(), moving_edge.verts[1].co.copy())
                moving_carriers = _edge_carrier_ids(moving_edge, face_layer)
            else:
                moving_endpoints = (
                    mirror_coordinate(moving_edge.verts[0].co, axis_index),
                    mirror_coordinate(moving_edge.verts[1].co, axis_index),
                )
                moving_carriers = _mirrored_carrier_ids(
                    _edge_carrier_ids(moving_edge, face_layer),
                    mirror_face_ids,
                )
            for fixed_edge in fixed:
                fixed_endpoints = (
                    mirror_coordinate(fixed_edge.verts[0].co, axis_index),
                    mirror_coordinate(fixed_edge.verts[1].co, axis_index),
                )
                fixed_carriers = _mirrored_carrier_ids(
                    _edge_carrier_ids(fixed_edge, face_layer),
                    mirror_face_ids,
                )
                result, reason = evaluate_pair(
                    moving_endpoints,
                    fixed_endpoints,
                    moving_carriers.intersection(fixed_carriers),
                )
                if reason:
                    return [], reason
                if result is None or result.kind in {"NONE", "COLLINEAR"}:
                    continue
                assert result.coordinate is not None
                moving_occurrence = _occurrence(
                    moving_edge,
                    result.factor_a,
                    result.endpoint_a,
                    axis_index,
                )
                fixed_positive = _occurrence(
                    fixed_edge,
                    1.0 - result.factor_b,
                    None if result.endpoint_b is None else 1 - result.endpoint_b,
                    axis_index,
                )
                fixed_negative = _occurrence(
                    fixed_edge,
                    result.factor_b,
                    result.endpoint_b,
                    axis_index,
                )
                raw_crossings.append(
                    _RawMirroredPathCrossing(
                        coordinate=result.coordinate,
                        positive=(moving_occurrence, fixed_positive) if moving_is_positive else (fixed_positive,),
                        negative=(fixed_negative,) if moving_is_positive else (moving_occurrence, fixed_negative),
                    )
                )

    for first_index, first_edge in enumerate(fixed):
        first_endpoints = (
            mirror_coordinate(first_edge.verts[0].co, axis_index),
            mirror_coordinate(first_edge.verts[1].co, axis_index),
        )
        first_carriers = _mirrored_carrier_ids(
            _edge_carrier_ids(first_edge, face_layer),
            mirror_face_ids,
        )
        for second_edge in fixed[first_index + 1 :]:
            second_endpoints = (
                mirror_coordinate(second_edge.verts[0].co, axis_index),
                mirror_coordinate(second_edge.verts[1].co, axis_index),
            )
            second_carriers = _mirrored_carrier_ids(
                _edge_carrier_ids(second_edge, face_layer),
                mirror_face_ids,
            )
            result, reason = evaluate_pair(
                first_endpoints,
                second_endpoints,
                first_carriers.intersection(second_carriers),
            )
            if reason:
                return [], reason
            if result is None or result.kind in {"NONE", "COLLINEAR"}:
                continue
            assert result.coordinate is not None
            positive_occurrences = []
            negative_occurrences = []
            for edge, factor, endpoint in (
                (first_edge, result.factor_a, result.endpoint_a),
                (second_edge, result.factor_b, result.endpoint_b),
            ):
                positive_occurrences.append(
                    _occurrence(
                        edge,
                        1.0 - factor,
                        None if endpoint is None else 1 - endpoint,
                        axis_index,
                    )
                )
                negative_occurrences.append(_occurrence(edge, factor, endpoint, axis_index))
            raw_crossings.append(
                _RawMirroredPathCrossing(
                    coordinate=result.coordinate,
                    positive=tuple(positive_occurrences),
                    negative=tuple(negative_occurrences),
                )
            )

    if not raw_crossings:
        return [], ""

    points = [crossing.coordinate for crossing in raw_crossings]
    clusters: list[_MirroredPathCrossingCluster] = []
    for indices in cluster_points_by_tolerance(points, tolerance):
        representative = points[indices[0]].copy()
        if abs(float(representative[axis_index])) <= tolerance:
            representative[axis_index] = 0.0
        mirrored = mirror_coordinate(representative, axis_index)
        positive_occurrences = [occurrence for index in indices for occurrence in raw_crossings[index].positive]
        negative_occurrences = [occurrence for index in indices for occurrence in raw_crossings[index].negative]
        if representative[axis_index] == 0.0:
            combined = _deduplicate_occurrences(positive_occurrences + negative_occurrences)
            positive_occurrences = list(combined)
            negative_occurrences = []
        clusters.append(
            _MirroredPathCrossingCluster(
                positive_coordinate=representative,
                negative_coordinate=mirrored,
                positive=_deduplicate_occurrences(positive_occurrences),
                negative=_deduplicate_occurrences(negative_occurrences),
                tolerance=tolerance,
            )
        )
    return clusters, ""


_VertexBinKey = tuple[int, int, int]
_VertexBinIndex = dict[_VertexBinKey, list[bmesh.types.BMVert]]


def _coordinate_components_finite(co) -> bool:
    return math.isfinite(float(co[0])) and math.isfinite(float(co[1])) and math.isfinite(float(co[2]))


def _crossings_quantized_coordinate(co, tolerance: float) -> _VertexBinKey:
    """Return the tuple bin key used exclusively by the crossings index."""

    inverse = 1.0 / max(tolerance, 1.0e-12)
    return (
        math.floor(co[0] * inverse),
        math.floor(co[1] * inverse),
        math.floor(co[2] * inverse),
    )


def _iter_crossings_quantized_neighborhood(
    co,
    tolerance: float,
) -> Iterable[_VertexBinKey]:
    """Yield the 27 tuple bins around a crossings query coordinate."""

    primary = _crossings_quantized_coordinate(co, tolerance)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                yield (primary[0] + dx, primary[1] + dy, primary[2] + dz)


def _build_crossings_vertex_bin_index_python(
    bm: bmesh.types.BMesh,
    tolerance: float,
) -> tuple[_VertexBinIndex | None, bool]:
    """Build a crossings vertex index with the scalar Python loop.

    Returns ``(index, use_fallback)``. When any live vertex coordinate or the
    tolerance is non-finite, returns ``(None, True)`` so callers fall back to
    the pre-U6-3a full ``bm.verts`` scan.
    """

    if not math.isfinite(tolerance):
        return None, True
    index: _VertexBinIndex = {}
    for vertex in bm.verts:
        if not vertex.is_valid:
            continue
        if not _coordinate_components_finite(vertex.co):
            return None, True
        bin_key = _crossings_quantized_coordinate(vertex.co, tolerance)
        index.setdefault(bin_key, []).append(vertex)
    return index, False


def _build_crossings_vertex_bin_index(
    bm: bmesh.types.BMesh,
    tolerance: float,
) -> tuple[_VertexBinIndex | None, bool]:
    """Build a crossings vertex index using a NumPy bulk coordinate read."""

    if not math.isfinite(tolerance):
        return None, True

    try:
        import bpy

        tmp = bpy.data.meshes.new(".yse_tmp_index")
        try:
            bm.to_mesh(tmp)
            vertex_count = len(tmp.vertices)
            if vertex_count != len(bm.verts):
                _LOGGER.warning("crossings vertex numpy index fallback: vertex count mismatch")
                return _build_crossings_vertex_bin_index_python(bm, tolerance)

            buf = numpy.empty(vertex_count * 3, dtype=numpy.float32)
            tmp.vertices.foreach_get("co", buf)
            coords = buf.reshape(vertex_count, 3).astype(numpy.float64)
            if not numpy.isfinite(coords).all():
                return None, True

            inverse = 1.0 / max(tolerance, 1.0e-12)
            scaled = coords * inverse
            if numpy.abs(scaled).max() >= 2**62:
                _LOGGER.warning("crossings vertex numpy index fallback: int64 safety range")
                return _build_crossings_vertex_bin_index_python(bm, tolerance)
            bins = numpy.floor(scaled).astype(numpy.int64)
            keys = bins.tolist()

            bm.verts.ensure_lookup_table()
            if vertex_count < 8:
                sample_indices = range(vertex_count)
            else:
                sample_indices = {
                    0,
                    vertex_count - 1,
                    *(int(round(step * (vertex_count - 1) / 7)) for step in range(1, 7)),
                }
            for sample_index in sample_indices:
                coordinate = bm.verts[sample_index].co
                key = keys[sample_index]
                if any(math.floor(coordinate[axis] * inverse) != key[axis] for axis in range(3)):
                    _LOGGER.warning("crossings vertex numpy index fallback: vertex order mismatch")
                    return _build_crossings_vertex_bin_index_python(bm, tolerance)

            index: _VertexBinIndex = {}
            for vertex, key in zip(bm.verts, keys, strict=True):
                index.setdefault((key[0], key[1], key[2]), []).append(vertex)
            return index, False
        finally:
            bpy.data.meshes.remove(tmp)
    except Exception as exc:
        _LOGGER.warning("crossings vertex numpy index fallback: %s", exc)
        return _build_crossings_vertex_bin_index_python(bm, tolerance)


def _register_crossings_vertex(
    index: _VertexBinIndex,
    vertex: bmesh.types.BMVert,
    tolerance: float,
) -> bool:
    """Register *vertex* under its primary bin. False when non-finite."""

    if not math.isfinite(tolerance) or not _coordinate_components_finite(vertex.co):
        return False
    bin_key = _crossings_quantized_coordinate(vertex.co, tolerance)
    index.setdefault(bin_key, []).append(vertex)
    return True


def _unregister_crossings_vertex(
    index: _VertexBinIndex,
    vertex: bmesh.types.BMVert,
    coordinate,
    tolerance: float,
) -> bool:
    """Remove *vertex* from the bin of *coordinate*. False when non-finite."""

    if not math.isfinite(tolerance) or not _coordinate_components_finite(coordinate):
        return False
    bin_key = _crossings_quantized_coordinate(coordinate, tolerance)
    bucket = index.get(bin_key)
    if not bucket:
        return True
    for position, candidate in enumerate(bucket):
        if candidate is vertex:
            del bucket[position]
            break
    if not bucket:
        del index[bin_key]
    return True


def _rebin_crossings_vertex(
    index: _VertexBinIndex,
    vertex: bmesh.types.BMVert,
    old_coordinate,
    new_coordinate,
    tolerance: float,
) -> bool:
    """Move *vertex* from the bin of *old_coordinate* to *new_coordinate*."""

    if not _unregister_crossings_vertex(index, vertex, old_coordinate, tolerance):
        return False
    if not math.isfinite(tolerance) or not _coordinate_components_finite(new_coordinate):
        return False
    bin_key = _crossings_quantized_coordinate(new_coordinate, tolerance)
    index.setdefault(bin_key, []).append(vertex)
    return True


def _scan_crossings_vertices_full(
    bm: bmesh.types.BMesh,
    coordinate,
    tolerance: float,
    *,
    exclude: set[bmesh.types.BMVert] | None = None,
    exclude_vertex: bmesh.types.BMVert | None = None,
) -> list[bmesh.types.BMVert]:
    """Pre-U6-3a full-mesh listcomp path retained for non-finite fallback."""

    if exclude is not None:
        return [
            vertex
            for vertex in bm.verts
            if vertex.is_valid and vertex not in exclude and coordinates_match(vertex.co, coordinate, tolerance)
        ]
    return [
        vertex
        for vertex in bm.verts
        if vertex.is_valid and vertex != exclude_vertex and coordinates_match(vertex.co, coordinate, tolerance)
    ]


def _scan_crossings_vertices_indexed(
    index: _VertexBinIndex,
    coordinate,
    tolerance: float,
    *,
    exclude: set[bmesh.types.BMVert] | None = None,
    exclude_vertex: bmesh.types.BMVert | None = None,
) -> list[bmesh.types.BMVert] | None:
    """27-bin candidate generation + full predicate. None when non-finite."""

    if not math.isfinite(tolerance) or not _coordinate_components_finite(coordinate):
        return None
    seen: set[int] = set()
    matches: list[bmesh.types.BMVert] = []
    for bin_key in _iter_crossings_quantized_neighborhood(coordinate, tolerance):
        for vertex in index.get(bin_key, ()):
            vertex_id = id(vertex)
            if vertex_id in seen:
                continue
            seen.add(vertex_id)
            if not vertex.is_valid:
                continue
            if exclude is not None:
                if vertex in exclude:
                    continue
            elif exclude_vertex is not None and vertex == exclude_vertex:
                continue
            if coordinates_match(vertex.co, coordinate, tolerance):
                matches.append(vertex)
    return matches


@dataclass(frozen=True, slots=True)
class CrossingEdgeMutation:
    """Mutation facts for one source edge in a crossings transaction."""

    source_edge_id: int
    final_edges: tuple[bmesh.types.BMEdge, ...]
    cache_position: int | None
    endpoint_reused: bool
    pointmerged: bool
    pre_faces: tuple[bmesh.types.BMFace, ...] = ()


@dataclass(frozen=True, slots=True)
class CrossingMutationSummary:
    """Source-edge mutation summary consumed by incremental collect."""

    edges: tuple[CrossingEdgeMutation, ...]
    removed_edge_count: int = 0
    removed_face_count: int = 0
    pre_apply_edge_count: int = 0
    pre_apply_face_count: int = 0
    unexpected_topology_change: bool = False


def apply_mirrored_path_crossings(
    bm: bmesh.types.BMesh,
    plan: Sequence[_MirroredPathCrossingCluster],
    *,
    cache_positions: Mapping[int, int] | None = None,
    return_summary: bool = False,
) -> tuple[int, str] | tuple[int, str, CrossingMutationSummary]:
    """Apply a mirrored crossing plan on its live BMesh transaction.

    The historical ``(count, reason)`` result remains the default.  Callers
    that own the Knife collect cache may opt into a third result containing
    source-edge mutation facts via ``return_summary=True``.
    """

    def _result(
        count: int,
        reason: str,
        summary: CrossingMutationSummary | None = None,
    ) -> tuple[int, str] | tuple[int, str, CrossingMutationSummary]:
        if return_summary:
            return count, reason, summary or CrossingMutationSummary(())
        return count, reason

    if not plan:
        return _result(0, "")
    marker_layer = bm.edges.layers.int.get(EDGE_ORIGINAL_LAYER)
    vertex_selection_layer = bm.verts.layers.int.get(VERT_SELECTION_LAYER)
    edge_selection_layer = bm.edges.layers.int.get(EDGE_SELECTION_LAYER)
    if marker_layer is None or vertex_selection_layer is None or edge_selection_layer is None:
        return _result(0, "temporary topology or selection markers are missing")

    bm.edges.ensure_lookup_table()
    pre_apply_edge_count = len(bm.edges)
    pre_apply_face_count = len(bm.faces)

    applications: list[tuple[Vector, tuple[_MirroredPathOccurrence, ...]]] = []
    for cluster in plan:
        if cluster.positive:
            applications.append((cluster.positive_coordinate, cluster.positive))
        if cluster.negative:
            applications.append((cluster.negative_coordinate, cluster.negative))

    native_edge_selection: dict[int, bool] = {}
    native_vertex_selection: dict[int, bool] = {}
    edge_marker: dict[int, int] = {}
    endpoint_vertex_by_occurrence: dict[int, bmesh.types.BMVert] = {}
    participant_endpoints: list[set[bmesh.types.BMVert]] = []
    for _coordinate, occurrences in applications:
        endpoints: set[bmesh.types.BMVert] = set()
        for occurrence in occurrences:
            edge = occurrence.edge
            if not edge.is_valid:
                return _result(0, "a mirrored crossing source edge was lost before stitching")
            edge_key = occurrence.edge_id
            native_edge_selection.setdefault(edge_key, bool(edge.select))
            edge_marker.setdefault(edge_key, int(edge[marker_layer]))
            if occurrence.endpoint_index is not None:
                endpoint = edge.verts[occurrence.endpoint_index]
                endpoints.add(endpoint)
                endpoint_vertex_by_occurrence[id(occurrence)] = endpoint
                native_vertex_selection.setdefault(hash(endpoint), bool(endpoint.select))
        participant_endpoints.append(endpoints)

    tolerance = _plan_tolerance(plan)
    use_fallback = not math.isfinite(tolerance)
    if not use_fallback:
        for coordinate, _occurrences in applications:
            if not _coordinate_components_finite(coordinate):
                use_fallback = True
                break
    vertex_index: _VertexBinIndex | None = None
    if not use_fallback:
        vertex_index, use_fallback = _build_crossings_vertex_bin_index(bm, tolerance)

    def _collect_extras(
        application_index: int,
        coordinate,
    ) -> list[bmesh.types.BMVert]:
        nonlocal use_fallback, vertex_index
        exclude = participant_endpoints[application_index]
        if not use_fallback and vertex_index is not None:
            indexed = _scan_crossings_vertices_indexed(
                vertex_index,
                coordinate,
                tolerance,
                exclude=exclude,
            )
            if indexed is not None:
                return indexed
            use_fallback = True
            vertex_index = None
        return _scan_crossings_vertices_full(
            bm,
            coordinate,
            tolerance,
            exclude=exclude,
        )

    def _collect_ambiguous(
        coordinate,
        survivor: bmesh.types.BMVert,
    ) -> list[bmesh.types.BMVert]:
        nonlocal use_fallback, vertex_index
        if not use_fallback and vertex_index is not None:
            indexed = _scan_crossings_vertices_indexed(
                vertex_index,
                coordinate,
                tolerance,
                exclude_vertex=survivor,
            )
            if indexed is not None:
                return indexed
            use_fallback = True
            vertex_index = None
        return _scan_crossings_vertices_full(
            bm,
            coordinate,
            tolerance,
            exclude_vertex=survivor,
        )

    def _index_register(vertex: bmesh.types.BMVert) -> None:
        nonlocal use_fallback, vertex_index
        if use_fallback or vertex_index is None:
            return
        if not _register_crossings_vertex(vertex_index, vertex, tolerance):
            use_fallback = True
            vertex_index = None

    def _index_unregister(vertex: bmesh.types.BMVert, coordinate) -> None:
        nonlocal use_fallback, vertex_index
        if use_fallback or vertex_index is None:
            return
        if not _unregister_crossings_vertex(vertex_index, vertex, coordinate, tolerance):
            use_fallback = True
            vertex_index = None

    def _index_rebin(vertex: bmesh.types.BMVert, old_coordinate, new_coordinate) -> None:
        nonlocal use_fallback, vertex_index
        if use_fallback or vertex_index is None:
            return
        if not _rebin_crossings_vertex(
            vertex_index,
            vertex,
            old_coordinate,
            new_coordinate,
            tolerance,
        ):
            use_fallback = True
            vertex_index = None

    # reusable_vertex (extras): resolve for every application before any
    # topology mutation (§I-3a evaluation order).
    reusable_vertex: list[bmesh.types.BMVert | None] = []
    for application_index, (coordinate, _occurrences) in enumerate(applications):
        extras = _collect_extras(application_index, coordinate)
        if len(extras) > 1:
            return _result(0, "multiple existing vertices are ambiguous at a mirrored cut intersection")
        reusable_vertex.append(extras[0] if extras else None)
        if extras:
            native_vertex_selection.setdefault(hash(extras[0]), bool(extras[0].select))

    split_entries_by_edge: dict[
        int,
        list[tuple[float, int, _MirroredPathOccurrence]],
    ] = defaultdict(list)
    edge_by_key: dict[int, bmesh.types.BMEdge] = {}
    pre_faces_by_edge: dict[int, tuple[bmesh.types.BMFace, ...]] = {}
    endpoint_reused_by_edge: dict[int, bool] = defaultdict(bool)
    pointmerged_by_edge: dict[int, bool] = defaultdict(bool)
    created_edges_by_edge: dict[int, list[bmesh.types.BMEdge]] = defaultdict(list)
    for application_index, (_coordinate, occurrences) in enumerate(applications):
        for occurrence in occurrences:
            key = occurrence.edge_id
            edge_by_key[key] = occurrence.edge
            pre_faces_by_edge.setdefault(
                key,
                tuple(face for face in occurrence.edge.link_faces if face.is_valid),
            )
            endpoint_reused_by_edge[occurrence.edge_id] |= occurrence.endpoint_index is not None
            if occurrence.endpoint_index is not None:
                continue
            split_entries_by_edge[key].append((occurrence.factor, application_index, occurrence))

    vertex_by_occurrence: dict[int, bmesh.types.BMVert] = dict(endpoint_vertex_by_occurrence)
    for edge_key, entries in split_entries_by_edge.items():
        original_edge = edge_by_key[edge_key]
        if not original_edge.is_valid:
            return _result(0, "a mirrored crossing source edge was lost during stitching")
        original_start = original_edge.verts[0]
        original_end = original_edge.verts[1]
        descendant = original_edge
        descendant_start = original_start
        interval_start = 0.0
        selected = native_edge_selection[edge_key]
        marker = edge_marker[edge_key]
        entries.sort(key=lambda entry: entry[0])
        for factor, _application_index, occurrence in entries:
            if factor <= interval_start or factor >= 1.0:
                return _result(0, "a mirrored crossing split factor is not interior to its descendant edge")
            local_factor = (factor - interval_start) / (1.0 - interval_start)
            try:
                new_edge, new_vertex = bmesh.utils.edge_split(
                    descendant,
                    descendant_start,
                    local_factor,
                )
            except (RuntimeError, ValueError) as exc:
                return _result(0, f"could not split a mirrored path crossing edge: {exc}")

            for half_edge in (descendant, new_edge):
                half_edge[marker_layer] = marker
                half_edge.select = selected
                half_edge[edge_selection_layer] = int(selected)
            new_vertex.select = selected
            new_vertex[vertex_selection_layer] = int(selected)
            native_vertex_selection[hash(new_vertex)] = selected
            vertex_by_occurrence[id(occurrence)] = new_vertex
            _index_register(new_vertex)
            created_edges_by_edge[edge_key].append(new_edge)

            descendants = [edge for edge in (descendant, new_edge) if original_end in edge.verts]
            if len(descendants) != 1:
                return _result(0, "could not track a mirrored crossing descendant edge")
            descendant = descendants[0]
            descendant_start = new_vertex
            interval_start = factor

    for application_index, (coordinate, occurrences) in enumerate(applications):
        vertices: list[bmesh.types.BMVert] = []
        edge_key_by_vertex: dict[int, tuple[tuple[float, float, float], tuple[float, float, float]]] = {}
        for occurrence in occurrences:
            vertex = vertex_by_occurrence.get(id(occurrence))
            if vertex is None or not vertex.is_valid:
                return _result(0, "a mirrored crossing vertex was lost before cluster unification")
            if vertex not in vertices:
                vertices.append(vertex)
            vertex_key = hash(vertex)
            current_key = edge_key_by_vertex.get(vertex_key)
            if current_key is None or occurrence.edge_key < current_key:
                edge_key_by_vertex[vertex_key] = occurrence.edge_key

        existing = reusable_vertex[application_index]
        if existing is not None:
            if not existing.is_valid:
                return _result(0, "an existing mirrored crossing vertex was lost before reuse")
            if existing not in vertices:
                vertices.append(existing)
            survivor = existing
        else:
            survivor = min(vertices, key=lambda vertex: edge_key_by_vertex[hash(vertex)])

        selected = any(native_edge_selection[occurrence.edge_id] for occurrence in occurrences)
        selected |= any(native_vertex_selection.get(hash(vertex), bool(vertex.select)) for vertex in vertices)
        snapshot_selected = selected or any(
            bool(vertex[vertex_selection_layer]) for vertex in vertices if vertex.is_valid
        )
        old_survivor_co = survivor.co.copy()
        survivor.co = coordinate.copy()
        _index_rebin(survivor, old_survivor_co, coordinate)
        for vertex in list(vertices):
            if vertex == survivor:
                continue
            discard_co = vertex.co.copy() if vertex.is_valid else None
            try:
                bmesh.ops.pointmerge(
                    bm,
                    verts=[survivor, vertex],
                    merge_co=coordinate,
                )
            except (RuntimeError, ValueError) as exc:
                return _result(0, f"could not unify mirrored crossing vertices: {exc}")
            for occurrence in occurrences:
                pointmerged_by_edge[occurrence.edge_id] = True
            if discard_co is not None:
                _index_unregister(vertex, discard_co)
            if not survivor.is_valid:
                return _result(0, "the mirrored crossing survivor was lost during point merge")
        old_survivor_co = survivor.co.copy()
        survivor.co = coordinate.copy()
        _index_rebin(survivor, old_survivor_co, coordinate)
        survivor.select = selected
        survivor[vertex_selection_layer] = int(snapshot_selected)

        ambiguous = _collect_ambiguous(coordinate, survivor)
        if ambiguous:
            return _result(0, "a separate existing vertex remains within tolerance of a mirrored cut intersection")

    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.normal_update()
    bm.edges.ensure_lookup_table()
    created_count = sum(len(edges) for edges in created_edges_by_edge.values())
    expected_edge_count = pre_apply_edge_count + created_count
    unexpected_topology_change = len(bm.edges) != expected_edge_count or len(bm.faces) != pre_apply_face_count
    removed_edge_count = max(0, expected_edge_count - len(bm.edges))
    removed_face_count = max(0, pre_apply_face_count - len(bm.faces))
    tail = tuple(bm.edges[index] for index in range(pre_apply_edge_count, len(bm.edges)))
    if len(tail) != created_count:
        unexpected_topology_change = True
    final_edges_by_id = {
        edge_key: (edge_by_key[edge_key], *created_edges_by_edge.get(edge_key, ())) for edge_key in edge_by_key
    }
    summary = CrossingMutationSummary(
        tuple(
            CrossingEdgeMutation(
                source_edge_id=edge_key,
                final_edges=final_edges_by_id.get(edge_key, ()),
                cache_position=None if cache_positions is None else cache_positions.get(edge_key),
                endpoint_reused=endpoint_reused_by_edge.get(edge_key, False),
                pointmerged=pointmerged_by_edge.get(edge_key, False),
                pre_faces=pre_faces_by_edge.get(edge_key, ()),
            )
            for edge_key in edge_by_key
        ),
        removed_edge_count=removed_edge_count,
        removed_face_count=removed_face_count,
        pre_apply_edge_count=pre_apply_edge_count,
        pre_apply_face_count=pre_apply_face_count,
        unexpected_topology_change=unexpected_topology_change,
    )
    return _result(len(applications), "", summary)


def _plan_tolerance(plan: Sequence[_MirroredPathCrossingCluster]) -> float:
    return plan[0].tolerance if plan else 0.0


def collect_source_path_edges(
    bm: bmesh.types.BMesh,
    axis_index: int,
    tolerance: float,
    requested_side: str,
    *,
    selected_only: bool = False,
) -> tuple[list[bmesh.types.BMEdge], str | None, int, int]:
    """Return path edges on one source half (Loop Cut / Offset; one-side).

    Knife uses :func:`collect_knife_path_edges_by_side` instead. The two final
    integers are the total number of new path edges and the number that cross
    the mirror plane.
    """

    all_path_edges = _discover_path_edges(bm, selected_only=selected_only)
    if not all_path_edges and bm.edges.layers.int.get(EDGE_ORIGINAL_LAYER) is None:
        return [], None, 0, 0

    side, crossing = choose_source_side(all_path_edges, axis_index, tolerance, requested_side)
    if side is None:
        return [], None, len(all_path_edges), crossing

    source_edges = [edge for edge in all_path_edges if _edge_side(edge, axis_index, tolerance) == side]
    return source_edges, side, len(all_path_edges), crossing


def target_face_ids_for_edges(
    source_edges: Iterable[bmesh.types.BMEdge],
    face_layer,
    mirror_face_ids: MirrorFaceMap,
) -> tuple[set[FaceId], set[FaceId]]:
    """Return matching target IDs and source IDs without a counterpart."""

    targets: set[FaceId] = set()
    unmatched: set[FaceId] = set()
    for edge in source_edges:
        for face in edge.link_faces:
            source_id = FaceId(int(face[face_layer]))
            target_id = mirror_face_ids.get(source_id)
            if target_id is None:
                unmatched.add(source_id)
            else:
                targets.add(target_id)
    return targets, unmatched


def _point_segment_distance_and_factor(
    coordinate: Vector,
    edge: bmesh.types.BMEdge,
) -> tuple[float, float]:
    # Intentionally Euclidean: this is a geometric point-to-segment distance,
    # not a coordinate-identity test (which is coordinates_match / Chebyshev).
    a = edge.verts[0].co
    delta = edge.verts[1].co - a
    length_squared = delta.length_squared
    if length_squared <= 1.0e-30:
        return (coordinate - a).length, 0.0
    factor = max(
        0.0,
        min(1.0, (coordinate - a).dot(delta) / length_squared),
    )
    return (coordinate - (a + factor * delta)).length, factor


def _is_interior_edge_factor(factor: float, edge_length: float, tolerance: float) -> bool:
    """True when the split sits more than *tolerance* from both endpoints (contract F)."""

    if edge_length <= 0.0:
        return False
    distance_along = factor * edge_length
    return tolerance < distance_along < (edge_length - tolerance)


def _collect_reflected_path_context(
    source_edges: list[bmesh.types.BMEdge],
    face_layer,
    mirror_face_ids: MirrorFaceMap,
    *,
    require_all_mirrored: bool,
) -> tuple[
    dict[int, bmesh.types.BMVert],
    dict[int, set[FaceId]],
    list[tuple[int, int, set[FaceId]]],
    set[FaceId],
    str,
]:
    """Build source-vertex / target-face maps shared by preflight and apply."""

    source_vertex_by_key: dict[int, bmesh.types.BMVert] = {}
    target_ids_by_vertex: dict[int, set[FaceId]] = defaultdict(set)
    edge_records: list[tuple[int, int, set[FaceId]]] = []
    unmatched_face_ids: set[FaceId] = set()

    for edge in source_edges:
        endpoint_keys: list[int] = []
        edge_target_ids: set[FaceId] = set()
        for vertex in edge.verts:
            key = hash(vertex)
            source_vertex_by_key[key] = vertex
            endpoint_keys.append(key)
        for face in edge.link_faces:
            source_face_id = FaceId(int(face[face_layer]))
            target_face_id = mirror_face_ids.get(source_face_id)
            if target_face_id is None:
                unmatched_face_ids.add(source_face_id)
                if require_all_mirrored:
                    return {}, {}, [], unmatched_face_ids, "unmatched"
                continue
            edge_target_ids.add(target_face_id)
            for key in endpoint_keys:
                target_ids_by_vertex[key].add(target_face_id)
        edge_records.append((endpoint_keys[0], endpoint_keys[1], edge_target_ids))

    return (
        source_vertex_by_key,
        target_ids_by_vertex,
        edge_records,
        unmatched_face_ids,
        "",
    )


def _target_faces_by_id(
    bm: bmesh.types.BMesh,
    face_layer,
    needed_target_ids: set[FaceId],
) -> dict[FaceId, list[bmesh.types.BMFace]]:
    target_faces_by_id: dict[FaceId, list[bmesh.types.BMFace]] = defaultdict(list)
    for face in bm.faces:
        face_id = FaceId(int(face[face_layer]))
        if face_id in needed_target_ids:
            target_faces_by_id[face_id].append(face)
    return target_faces_by_id


def _resolve_reflected_vertex_on_target(
    expected: Vector,
    candidate_faces: set[bmesh.types.BMFace],
    tolerance: float,
) -> tuple[
    Literal["exact", "boundary", "missing", "ambiguous"],
    bmesh.types.BMVert | None,
    bmesh.types.BMEdge | None,
    float,
    str,
]:
    """Map a reflected position to an existing target vertex or boundary split.

    Shared by preflight (dry-run) and apply so both use identical E/F semantics.
    Returns ``(kind, exact_vertex, split_edge, factor, error_reason)``.

    Multiple distinct vertices within *tolerance* of *expected* are ambiguous:
    callers must decline rather than pick an arbitrary vertex that could
    invent a duplicate edge.
    """

    if not candidate_faces:
        return "missing", None, None, 0.0, "a mirrored target face was lost"

    # Acceptance uses coordinates_match; the Euclidean length ranks candidates
    # only to detect a unique nearest match when exactly one is accepted.
    candidate_vertices = {vertex for face in candidate_faces for vertex in face.verts}
    exact_vertices = sorted(
        (
            ((vertex.co - expected).length, vertex.index, vertex)
            for vertex in candidate_vertices
            if coordinates_match(vertex.co, expected, tolerance)
        ),
        key=lambda item: (item[0], item[1]),
    )
    if len(exact_vertices) > 1:
        return (
            "ambiguous",
            None,
            None,
            0.0,
            "ambiguous mirrored target vertices within tolerance",
        )
    if exact_vertices:
        return "exact", exact_vertices[0][2], None, 0.0, ""

    edge_limit = max(tolerance * 2.0, 1.0e-9)
    candidate_edges = {edge for face in candidate_faces for edge in face.edges if edge.is_valid}
    edge_candidates: list[tuple[float, int, bmesh.types.BMEdge, float]] = []
    for edge in candidate_edges:
        distance, factor = _point_segment_distance_and_factor(expected, edge)
        if not _is_interior_edge_factor(factor, edge.calc_length(), tolerance):
            continue
        if distance > edge_limit:
            continue
        edge_candidates.append((distance, edge.index, edge, factor))
    if not edge_candidates:
        return (
            "missing",
            None,
            None,
            0.0,
            "a reflected cut vertex is not on a target boundary edge",
        )

    edge_candidates.sort(key=lambda item: (item[0], item[1]))
    _distance, _edge_index, target_edge, factor = edge_candidates[0]
    return "boundary", None, target_edge, factor, ""


def _face_surface_triangles(face: bmesh.types.BMFace):
    """Valid tessellation triangles covering the (possibly curved) face.

    Only tris and quads are covered.  A quad diagonal is valid only when its
    two triangles wind consistently — on a concave or folded quad the other
    diagonal spans area outside the real surface and must not be offered.
    N-gons are not covered at all: their interior-chain acceptance is declined
    so those strokes keep using the projection fallback.
    """

    vertices = face.verts
    count = len(vertices)
    if count == 3:
        yield vertices[0].co, vertices[1].co, vertices[2].co
        return
    if count == 4:
        for a, b, c, d in ((0, 1, 2, 3), (1, 2, 3, 0)):
            first = (vertices[b].co - vertices[a].co).cross(vertices[c].co - vertices[a].co)
            second = (vertices[c].co - vertices[a].co).cross(vertices[d].co - vertices[a].co)
            if first.dot(second) <= 0.0:
                continue
            yield vertices[a].co, vertices[b].co, vertices[c].co
            yield vertices[a].co, vertices[c].co, vertices[d].co
        return
    # Boundary splits routinely turn carrier quads into n-gons.  Ear clipping
    # respects concavity, so no triangle spans area outside the real polygon.
    coordinates = [vertex.co for vertex in vertices]
    for tri_a, tri_b, tri_c in mathutils.geometry.tessellate_polygon([coordinates]):
        yield coordinates[tri_a], coordinates[tri_b], coordinates[tri_c]


def _point_strictly_inside_face(
    point: Vector,
    face: bmesh.types.BMFace,
    tolerance: float,
) -> bool:
    """True when *point* lies on the face surface, away from its boundary.

    Curved-surface faces are not planar, so containment is measured as the
    distance to the face's tessellation triangles instead of a single median
    plane with a normal-projected polygon test.
    """

    if not face.is_valid:
        return False
    surface_limit = max(tolerance * 2.0, 1.0e-9)
    if any(coordinates_match(vertex.co, point, tolerance) for vertex in face.verts):
        return False
    edge_limit = max(tolerance * 2.0, 1.0e-9)
    for edge in face.edges:
        if not edge.is_valid:
            continue
        distance, factor = _point_segment_distance_and_factor(point, edge)
        if distance <= edge_limit and _is_interior_edge_factor(factor, edge.calc_length(), tolerance):
            return False
    for corner_a, corner_b, corner_c in _face_surface_triangles(face):
        closest = mathutils.geometry.closest_point_on_tri(point, corner_a, corner_b, corner_c)
        if (closest - point).length <= surface_limit:
            return True
    return False


def _interior_faces_for_point(
    point: Vector,
    candidate_faces: set[bmesh.types.BMFace],
    tolerance: float,
) -> list[bmesh.types.BMFace]:
    return sorted(
        (face for face in candidate_faces if face.is_valid and _point_strictly_inside_face(point, face, tolerance)),
        key=lambda face: face.index,
    )


def _path_adjacency(edge_records: list[tuple[int, int, set[FaceId]]]) -> dict[int, set[int]]:
    adjacency: dict[int, set[int]] = defaultdict(set)
    for source_a, source_b, _target_ids in edge_records:
        adjacency[source_a].add(source_b)
        adjacency[source_b].add(source_a)
    return adjacency


@dataclass(frozen=True, slots=True)
class _InteriorChain:
    """A maximal interior terminal chain accepted by the direct topology path."""

    members: tuple[int, ...]
    end_a: int
    end_b: int
    target_face_id: FaceId


def _faces_supporting_resolution(
    kind: str,
    exact_vertex: bmesh.types.BMVert | None,
    target_edge: bmesh.types.BMEdge | None,
    interior_face: bmesh.types.BMFace | None,
    candidate_faces: set[bmesh.types.BMFace],
) -> set[bmesh.types.BMFace]:
    if kind == "exact":
        assert exact_vertex is not None
        return {face for face in exact_vertex.link_faces if face.is_valid and face in candidate_faces}
    if kind == "boundary":
        assert target_edge is not None
        return {face for face in target_edge.link_faces if face.is_valid and face in candidate_faces}
    if kind == "interior":
        assert interior_face is not None
        return {interior_face}
    return set()


def _classify_reflected_vertices(
    source_vertex_by_key: dict[int, bmesh.types.BMVert],
    target_ids_by_vertex: dict[int, set[FaceId]],
    target_faces_by_id: dict[FaceId, list[bmesh.types.BMFace]],
    axis_index: int,
    tolerance: float,
) -> tuple[
    dict[int, tuple[str, bmesh.types.BMVert | None, bmesh.types.BMEdge | None, float, bmesh.types.BMFace | None, str]],
    str,
]:
    """Classify each source vertex for the direct path (exact/boundary/interior).

    Returns ``(classification, failure_reason)``. On failure classification may
    be incomplete; *failure_reason* is non-empty.
    """

    classification: dict[
        int,
        tuple[str, bmesh.types.BMVert | None, bmesh.types.BMEdge | None, float, bmesh.types.BMFace | None, str],
    ] = {}
    for source_key, source_vertex in source_vertex_by_key.items():
        expected = mirror_coordinate(source_vertex.co, axis_index)
        candidate_faces = {
            face
            for target_id in target_ids_by_vertex[source_key]
            for face in target_faces_by_id.get(target_id, ())
            if face.is_valid
        }
        kind, exact_vertex, target_edge, factor, reason = _resolve_reflected_vertex_on_target(
            expected,
            candidate_faces,
            tolerance,
        )
        if kind in {"exact", "boundary"}:
            classification[source_key] = (kind, exact_vertex, target_edge, factor, None, reason)
            continue
        if kind == "ambiguous":
            return classification, reason
        interior_faces = _interior_faces_for_point(expected, candidate_faces, tolerance)
        if len(interior_faces) == 1:
            classification[source_key] = ("interior", None, None, 0.0, interior_faces[0], "")
            continue
        if len(interior_faces) > 1:
            return (
                classification,
                "ambiguous mirrored target faces for interior point",
            )
        return classification, reason
    return classification, ""


def _find_interior_chains(
    classification: dict[
        int,
        tuple[str, bmesh.types.BMVert | None, bmesh.types.BMEdge | None, float, bmesh.types.BMFace | None, str],
    ],
    adjacency: dict[int, set[int]],
    source_vertex_by_key: dict[int, bmesh.types.BMVert],
    target_ids_by_vertex: dict[int, set[FaceId]],
    target_faces_by_id: dict[FaceId, list[bmesh.types.BMFace]],
    face_layer,
    axis_index: int,
    tolerance: float,
) -> tuple[list[_InteriorChain], str]:
    """Return accepted interior chains, or a non-empty reason on decline."""

    interior_keys = {key for key, (kind, *_rest) in classification.items() if kind == "interior"}
    if not interior_keys:
        return [], ""

    for key in interior_keys:
        if len(adjacency.get(key, ())) != 2:
            # Degree-1 tip or branch (degree >= 3): not a simple chain.
            return [], "a reflected cut vertex is not on a target boundary edge"

    visited: set[int] = set()
    chains: list[_InteriorChain] = []

    def _ordered_component(start: int) -> list[int] | None:
        """Return a path-ordered list of a connected interior component, or None."""

        stack = [start]
        component: list[int] = []
        seen: set[int] = set()
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            component.append(node)
            for neighbour in adjacency[node]:
                if neighbour in interior_keys and neighbour not in seen:
                    stack.append(neighbour)
        # Component must be a path: exactly two endpoints with one interior neighbour,
        # or a single vertex.
        if len(component) == 1:
            return component
        endpoint_count = 0
        endpoints: list[int] = []
        for node in component:
            interior_degree = sum(1 for neighbour in adjacency[node] if neighbour in interior_keys)
            if interior_degree == 1:
                endpoint_count += 1
                endpoints.append(node)
            elif interior_degree != 2:
                return None
        if endpoint_count != 2:
            return None  # cycle or branch among interiors
        # Walk from one endpoint along interior edges.
        ordered = [endpoints[0]]
        previous = None
        while len(ordered) < len(component):
            current = ordered[-1]
            nxt = None
            for neighbour in adjacency[current]:
                if neighbour in interior_keys and neighbour != previous:
                    nxt = neighbour
                    break
            if nxt is None:
                return None
            ordered.append(nxt)
            previous = current
        return ordered

    for start in sorted(interior_keys, key=lambda key: source_vertex_by_key[key].index):
        if start in visited:
            continue
        ordered = _ordered_component(start)
        if ordered is None:
            return [], "a reflected cut vertex is not on a target boundary edge"
        visited.update(ordered)

        # Chain ends: the non-interior neighbours of the path endpoints.
        def _chain_end(member: int, other_member: int | None) -> int | None:
            exteriors = [
                neighbour
                for neighbour in adjacency[member]
                if neighbour not in interior_keys and neighbour != other_member
            ]
            if len(exteriors) != 1:
                return None
            return exteriors[0]

        if len(ordered) == 1:
            ends = [neighbour for neighbour in adjacency[ordered[0]] if neighbour not in interior_keys]
            if len(ends) != 2:
                return [], "a reflected cut vertex is not on a target boundary edge"
            end_a, end_b = ends[0], ends[1]
            # Stable order by source vertex index.
            if source_vertex_by_key[end_a].index > source_vertex_by_key[end_b].index:
                end_a, end_b = end_b, end_a
        else:
            end_a = _chain_end(ordered[0], ordered[1])
            end_b = _chain_end(ordered[-1], ordered[-2])
            if end_a is None or end_b is None:
                return [], "a reflected cut vertex is not on a target boundary edge"

        if end_a == end_b:
            # A closed loop anchored on one non-interior vertex would need
            # face_split(face, A, A), which BMesh rejects; decline instead.
            return [], "a reflected cut vertex is not on a target boundary edge"

        if classification[end_a][0] == "interior" or classification[end_b][0] == "interior":
            return [], "a reflected cut vertex is not on a target boundary edge"
        if classification[end_a][0] in {"missing", "ambiguous"}:
            return [], classification[end_a][5] or "a reflected cut vertex is not on a target boundary edge"
        if classification[end_b][0] in {"missing", "ambiguous"}:
            return [], classification[end_b][5] or "a reflected cut vertex is not on a target boundary edge"

        # Common target face among chain members and both ends.
        common_faces: set[bmesh.types.BMFace] | None = None
        for key in (*ordered, end_a, end_b):
            kind, exact_vertex, target_edge, _factor, interior_face, _reason = classification[key]
            candidate_faces = {
                face
                for target_id in target_ids_by_vertex[key]
                for face in target_faces_by_id.get(target_id, ())
                if face.is_valid
            }
            supported = _faces_supporting_resolution(
                kind,
                exact_vertex,
                target_edge,
                interior_face,
                candidate_faces,
            )
            if common_faces is None:
                common_faces = set(supported)
            else:
                common_faces &= supported
        if not common_faces or len(common_faces) != 1:
            return [], "a reflected cut vertex is not on a target boundary edge"

        common_face = next(iter(common_faces))
        target_face_id = FaceId(int(common_face[face_layer]))
        # Every interior member must actually use this same face instance.
        for key in ordered:
            interior_face = classification[key][4]
            if interior_face is None or interior_face != common_face:
                return [], "a reflected cut vertex is not on a target boundary edge"

        chains.append(
            _InteriorChain(
                members=tuple(ordered),
                end_a=end_a,
                end_b=end_b,
                target_face_id=target_face_id,
            )
        )

    if len(visited) != len(interior_keys):
        return [], "a reflected cut vertex is not on a target boundary edge"
    return chains, ""


def _chain_source_edge_keys(chains: list[_InteriorChain]) -> set[frozenset[int]]:
    keys: set[frozenset[int]] = set()
    for chain in chains:
        sequence = (chain.end_a, *chain.members, chain.end_b)
        for left, right in zip(sequence, sequence[1:], strict=False):
            keys.add(frozenset((left, right)))
    return keys


def _realize_interior_chain(
    bm: bmesh.types.BMesh,
    chain: _InteriorChain,
    source_vertex_by_key: dict[int, bmesh.types.BMVert],
    target_vertex_by_source_key: dict[int, bmesh.types.BMVert],
    axis_index: int,
    tolerance: float,
    face_layer,
    marker_layer,
    existing_edges: dict | None,
    realized_face_ids: set[FaceId],
) -> tuple[int, int, str, dict | None]:
    """face_split one accepted interior chain. Returns created/already_present delta."""

    end_a = target_vertex_by_source_key[chain.end_a]
    end_b = target_vertex_by_source_key[chain.end_b]
    reflected_coords = [mirror_coordinate(source_vertex_by_key[member].co, axis_index) for member in chain.members]
    candidate_faces = sorted(
        (
            face
            for face in set(end_a.link_faces).intersection(end_b.link_faces)
            if face.is_valid and FaceId(int(face[face_layer])) == chain.target_face_id
        ),
        key=lambda face: face.index,
    )
    # Descendants of prior splits inherit the parent face id, so several
    # candidates can share the chain's target id.  Only the face that strictly
    # contains every reflected interior coordinate is the correct host.
    #
    # The strict re-test cannot be applied to the untouched single-candidate
    # case: the boundary end splits have already turned the host quad into an
    # n-gon whose ear-clip triangulation deviates from the evaluated quad
    # surface by more than the tolerance-scale limit on curved meshes, so it
    # would falsely decline (classification accepted these coordinates on the
    # pre-split ancestor during this same apply call).  Once an earlier chain
    # has face_split this target id, a lone shared candidate may be the wrong
    # descendant, so only then is the strict containment test decisive.
    if len(candidate_faces) == 1 and chain.target_face_id not in realized_face_ids:
        target_face = candidate_faces[0]
    else:
        containing_faces = [
            face
            for face in candidate_faces
            if all(_point_strictly_inside_face(coordinate, face, tolerance) for coordinate in reflected_coords)
        ]
        if len(containing_faces) != 1:
            return 0, 0, "could not place mirrored interior chain on a target face", existing_edges
        target_face = containing_faces[0]

    try:
        new_face, _new_loop = bmesh.utils.face_split(
            target_face,
            end_a,
            end_b,
            coords=[tuple(coordinate) for coordinate in reflected_coords],
        )
    except (RuntimeError, ValueError) as exc:
        return 0, 0, f"could not split a target face for interior chain: {exc}", existing_edges
    if new_face is None or not new_face.is_valid or not target_face.is_valid:
        return 0, 0, "could not split a target face for interior chain", existing_edges
    realized_face_ids.add(chain.target_face_id)

    # The coords vertices lie exactly on the cut path both descendants share,
    # so they are the shared vertices minus the chain ends.  This stays local
    # to the two faces; snapshotting bm.verts/bm.edges around the split costs
    # seconds of proxy iteration on dense meshes.
    end_hashes = {hash(end_a), hash(end_b)}
    target_vert_hashes = {hash(vertex) for vertex in target_face.verts}
    new_verts = [
        vertex for vertex in new_face.verts if hash(vertex) in target_vert_hashes and hash(vertex) not in end_hashes
    ]
    if len(new_verts) < len(chain.members):
        return 0, 0, "interior chain face_split created too few vertices", existing_edges

    remaining = list(new_verts)
    for member, expected in zip(chain.members, reflected_coords, strict=False):
        remaining.sort(key=lambda vertex: (vertex.co - expected).length)
        chosen = remaining.pop(0)
        chosen.co = expected.copy()
        chosen.select = False
        target_vertex_by_source_key[member] = chosen

    # Count and mark every chain segment end_a–v1–…–vn–end_b.  Every segment
    # touches at least one vertex the face_split just created, so none can
    # predate this call: they all count as created.
    created = 0
    already = 0
    sequence_keys = (chain.end_a, *chain.members, chain.end_b)
    for left_key, right_key in zip(sequence_keys, sequence_keys[1:], strict=False):
        left = target_vertex_by_source_key[left_key]
        right = target_vertex_by_source_key[right_key]
        edge = bm.edges.get([left, right])
        if edge is None:
            return created, already, "interior chain face_split missed a chain edge", existing_edges
        edge[marker_layer] = 0
        edge.select = False
        for face in edge.link_faces:
            face.select = False
        created += 1
        if existing_edges is not None:
            _register_edge_endpoint_pair(
                existing_edges,
                left.co,
                right.co,
                tolerance,
                face_ids={FaceId(int(face[face_layer])) for face in edge.link_faces},
            )
    return created, already, "", existing_edges


def reflected_path_uses_only_target_boundaries(
    bm: bmesh.types.BMesh,
    source_edges: Iterable[bmesh.types.BMEdge],
    axis_index: int,
    tolerance: float,
    mirror_face_ids: MirrorFaceMap,
) -> bool:
    """Return whether the direct path builder supports every reflected vertex.

    A committed straight Knife segment and the loop-based tools terminate on
    existing face boundaries. Multi-click Knife strokes may also place an
    intentional face-interior terminal chain (degree-2 interior vertices whose
    ends resolve on one common target face). Those chains are accepted here;
    interior networks with branches still use the Knife Project fallback.
    """

    source_edges = list(source_edges)
    face_layer = bm.faces.layers.int.get(FACE_ID_LAYER)
    if not source_edges or face_layer is None:
        return False

    (
        source_vertex_by_key,
        target_ids_by_vertex,
        edge_records,
        unmatched_face_ids,
        _status,
    ) = _collect_reflected_path_context(
        source_edges,
        face_layer,
        mirror_face_ids,
        require_all_mirrored=True,
    )
    if unmatched_face_ids or not source_vertex_by_key:
        return False

    needed_target_ids = {target_id for target_ids in target_ids_by_vertex.values() for target_id in target_ids}
    target_faces_by_id = _target_faces_by_id(bm, face_layer, needed_target_ids)
    classification, reason = _classify_reflected_vertices(
        source_vertex_by_key,
        target_ids_by_vertex,
        target_faces_by_id,
        axis_index,
        tolerance,
    )
    if reason:
        return False

    adjacency = _path_adjacency(edge_records)
    _chains, chain_reason = _find_interior_chains(
        classification,
        adjacency,
        source_vertex_by_key,
        target_ids_by_vertex,
        target_faces_by_id,
        face_layer,
        axis_index,
        tolerance,
    )
    return not chain_reason


def apply_reflected_path_topology(
    bm: bmesh.types.BMesh,
    source_edges: Iterable[bmesh.types.BMEdge],
    axis_index: int,
    tolerance: float,
    mirror_face_ids: MirrorFaceMap,
) -> tuple[int, int, str]:
    """Rebuild a native cut path exactly on its mirrored faces.

    Unlike Knife Project, this is independent of the viewport and cannot cut a
    second front/back surface when a curved loop overlaps itself on screen.
    The native source path supplies exact endpoint coordinates and inherited
    original-face IDs. Target boundary edges are split at the reflected points,
    then the corresponding target faces are split between those vertices.

    Face-interior terminal chains (degree-2 interior vertices between two
    normally-resolved ends on one common target face) are realized with a
    single ``bmesh.utils.face_split(..., coords=...)`` per chain.

    Existing segments are detected by BMVert identity *and* by endpoint
    coordinate tolerance (same store as :func:`build_reflected_cutter`) so a
    near-self-mirrored stroke does not invent a geometric duplicate. Multiple
    tol-local vertex candidates decline the whole apply (all-or-nothing).

    Straddling Loop Cut rings symmetrize through this same path: reflection
    onto self-mirrored or paired carrier faces creates the counterpart ring
    or counts it as already present.  The face correspondence is load-bearing;
    without it the counterpart check below declines.

    Returns ``(created_edges, already_present_edges, failure_reason)``. Callers
    must provide rollback because a late validation error can occur after an
    earlier target edge has already been split.
    """

    source_edges = list(source_edges)
    if not source_edges:
        return 0, 0, "no source cut edges were supplied"

    marker_layer = bm.edges.layers.int.get(EDGE_ORIGINAL_LAYER)
    face_layer = bm.faces.layers.int.get(FACE_ID_LAYER)
    if marker_layer is None or face_layer is None:
        return 0, 0, "temporary topology markers are missing"

    # Capture every source-side relationship before modifying the target. This
    # also makes faces which touch the symmetry plane safe to process.
    (
        source_vertex_by_key,
        target_ids_by_vertex,
        edge_records,
        unmatched_face_ids,
        _status,
    ) = _collect_reflected_path_context(
        source_edges,
        face_layer,
        mirror_face_ids,
        require_all_mirrored=False,
    )

    if unmatched_face_ids:
        return (
            0,
            0,
            f"{len(unmatched_face_ids)} source face(s) have no mirrored counterpart",
        )
    if any(not target_ids for _a, _b, target_ids in edge_records):
        return 0, 0, "a source cut edge has no mirrored target face"

    needed_target_ids = {target_id for target_ids in target_ids_by_vertex.values() for target_id in target_ids}
    target_faces_by_id = _target_faces_by_id(bm, face_layer, needed_target_ids)
    classification, classify_reason = _classify_reflected_vertices(
        source_vertex_by_key,
        target_ids_by_vertex,
        target_faces_by_id,
        axis_index,
        tolerance,
    )
    if classify_reason:
        return 0, 0, classify_reason

    adjacency = _path_adjacency(edge_records)
    chains, chain_reason = _find_interior_chains(
        classification,
        adjacency,
        source_vertex_by_key,
        target_ids_by_vertex,
        target_faces_by_id,
        face_layer,
        axis_index,
        tolerance,
    )
    if chain_reason:
        return 0, 0, chain_reason

    chain_edge_keys = _chain_source_edge_keys(chains)
    target_vertex_by_source_key: dict[int, bmesh.types.BMVert] = {}

    # Resolve non-interior vertices first so chain ends exist before face_split.
    # Resolution must happen per vertex against the CURRENT topology: an
    # earlier split can place a second reflected vertex on one of the new
    # halves, so the up-front classification's edge/factor would be stale.
    for source_key, source_vertex in source_vertex_by_key.items():
        if classification[source_key][0] == "interior":
            continue
        expected = mirror_coordinate(source_vertex.co, axis_index)
        candidate_faces = {
            face
            for target_id in target_ids_by_vertex[source_key]
            for face in target_faces_by_id.get(target_id, ())
            if face.is_valid
        }
        kind, exact_vertex, target_edge, factor, reason = _resolve_reflected_vertex_on_target(
            expected,
            candidate_faces,
            tolerance,
        )
        if kind == "exact":
            assert exact_vertex is not None
            target_vertex_by_source_key[source_key] = exact_vertex
            continue
        if kind in {"missing", "ambiguous"}:
            return 0, 0, reason

        assert target_edge is not None
        try:
            _new_edge, target_vertex = bmesh.utils.edge_split(
                target_edge,
                target_edge.verts[0],
                factor,
            )
        except (RuntimeError, ValueError) as exc:
            return 0, 0, f"could not split a mirrored target edge: {exc}"
        target_vertex.co = expected
        target_vertex.select = False
        target_vertex_by_source_key[source_key] = target_vertex

    existing_edges: _EdgeEndpointStore | None = None
    created_edges = 0
    already_present = 0

    realized_face_ids: set[FaceId] = set()
    for chain in chains:
        created_delta, already_delta, fail_reason, existing_edges = _realize_interior_chain(
            bm,
            chain,
            source_vertex_by_key,
            target_vertex_by_source_key,
            axis_index,
            tolerance,
            face_layer,
            marker_layer,
            existing_edges,
            realized_face_ids,
        )
        if fail_reason:
            return created_edges, already_present, fail_reason
        created_edges += created_delta
        already_present += already_delta

    # Endpoint-tol store matches build_reflected_cutter so geometric duplicates
    # (different BMVert pairs within tol) count as already_present.  Keep it
    # lazy: the common native-topology case resolves every segment by BMEdge
    # identity and never needs a geometric index.  Freeze the complete target
    # FaceId scope before processing; deferred/retry must not shrink it.  After
    # target vertices have been resolved, the edge loop does not mutate
    # topology before its first identity miss, so constructing the scoped store
    # at that boundary observes the same mesh as the eager path.
    initial_pending = [record for record in edge_records if frozenset((record[0], record[1])) not in chain_edge_keys]
    scoped_target_ids = frozenset().union(*(record[2] for record in initial_pending))
    pending = initial_pending
    while pending:
        deferred = []
        progress = False
        for source_a, source_b, possible_target_ids in pending:
            target_a = target_vertex_by_source_key[source_a]
            target_b = target_vertex_by_source_key[source_b]
            existing = bm.edges.get([target_a, target_b])
            if existing is not None:
                existing_target_ids = {FaceId(int(face[face_layer])) for face in existing.link_faces}
                if not existing_target_ids.intersection(possible_target_ids):
                    return (
                        created_edges,
                        already_present,
                        "an existing mirrored edge is outside its target face",
                    )
                already_present += 1
                progress = True
                continue

            if existing_edges is None:
                existing_edges = {}
                scoped_faces_by_id: dict[FaceId, list[bmesh.types.BMFace]] = defaultdict(list)
                for face in bm.faces:
                    if not face.is_valid:
                        continue
                    face_id = FaceId(int(face[face_layer]))
                    if face_id in scoped_target_ids:
                        scoped_faces_by_id[face_id].append(face)

                seen_edges: set[bmesh.types.BMEdge] = set()
                for scoped_faces in scoped_faces_by_id.values():
                    for face in scoped_faces:
                        for edge in face.edges:
                            if not edge.is_valid or edge in seen_edges:
                                continue
                            seen_edges.add(edge)
                            _register_edge_endpoint_pair(
                                existing_edges,
                                edge.verts[0].co,
                                edge.verts[1].co,
                                tolerance,
                                face_ids={FaceId(int(face[face_layer])) for face in edge.link_faces},
                            )

            endpoint_match = _match_edge_endpoint_pair_for_faces(
                target_a.co,
                target_b.co,
                tolerance,
                existing_edges,
                possible_target_ids,
            )
            if endpoint_match == "ambiguous":
                return (
                    created_edges,
                    already_present,
                    "multiple coordinate-matching edges are ambiguous across target faces",
                )
            if endpoint_match == "match":
                already_present += 1
                progress = True
                continue

            candidate_faces = sorted(
                (
                    face
                    for face in set(target_a.link_faces).intersection(target_b.link_faces)
                    if face.is_valid and FaceId(int(face[face_layer])) in possible_target_ids
                ),
                key=lambda face: face.index,
            )
            if not candidate_faces:
                deferred.append((source_a, source_b, possible_target_ids))
                continue

            try:
                bmesh.utils.face_split(candidate_faces[0], target_a, target_b)
            except (RuntimeError, ValueError) as exc:
                return (
                    created_edges,
                    already_present,
                    f"could not split a target face: {exc}",
                )
            new_edge = bm.edges.get([target_a, target_b])
            if new_edge is None:
                return created_edges, already_present, "target face split made no edge"
            new_edge[marker_layer] = 0
            new_edge.select = False
            for face in new_edge.link_faces:
                face.select = False
            assert existing_edges is not None
            _register_edge_endpoint_pair(
                existing_edges,
                new_edge.verts[0].co,
                new_edge.verts[1].co,
                tolerance,
                face_ids={FaceId(int(face[face_layer])) for face in new_edge.link_faces},
            )
            created_edges += 1
            progress = True

        if deferred and not progress:
            return (
                created_edges,
                already_present,
                f"could not place {len(deferred)} mirrored cut segment(s)",
            )
        pending = deferred

    bm.normal_update()
    return created_edges, already_present, ""


apply_reflected_loop_topology = apply_reflected_path_topology


def _edge_coordinate_key(
    a: Vector,
    b: Vector,
    tolerance: float,
) -> tuple[QuantizedCoordinate, QuantizedCoordinate]:
    qa = _quantized_coordinate(a, tolerance)
    qb = _quantized_coordinate(b, tolerance)
    return (qa, qb) if qa <= qb else (qb, qa)


def _coordinate_tuple(co: Vector) -> tuple[float, float, float]:
    return (float(co[0]), float(co[1]), float(co[2]))


def _canonical_edge_endpoints(
    a: Vector,
    b: Vector,
    tolerance: float,
) -> tuple[QuantizedCoordinate, tuple[float, float, float], tuple[float, float, float]]:
    """Return primary bin of the first endpoint plus real coords in canonical order."""

    coord_a = _coordinate_tuple(a)
    coord_b = _coordinate_tuple(b)
    qa = _quantized_coordinate(a, tolerance)
    qb = _quantized_coordinate(b, tolerance)
    if qa < qb or (qa == qb and coord_a <= coord_b):
        return qa, coord_a, coord_b
    return qb, coord_b, coord_a


_EdgeEndpointEntry = tuple[
    int | None,
    tuple[float, float, float],
    tuple[float, float, float],
    frozenset[FaceId] | None,
]
_EdgeEndpointStore = dict[QuantizedCoordinate, list[_EdgeEndpointEntry]]


def _register_edge_endpoint_pair(
    store: _EdgeEndpointStore,
    a: Vector,
    b: Vector,
    tolerance: float,
    marker: int | None = None,
    face_ids: Iterable[FaceId] | None = None,
) -> None:
    """Store real endpoint coords under the primary bin of the canonical first endpoint."""

    primary_a, coord_a, coord_b = _canonical_edge_endpoints(a, b, tolerance)
    stored_face_ids = frozenset(face_ids) if face_ids is not None else None
    store.setdefault(primary_a, []).append((marker, coord_a, coord_b, stored_face_ids))


def _match_edge_endpoint_pair(
    a: Vector,
    b: Vector,
    tolerance: float,
    store: _EdgeEndpointStore,
) -> tuple[float, int | None] | None:
    """Best geometric edge match: min of max endpoint Chebyshev distances (R2-1).

    Probes the 27-bin neighborhood of each query endpoint orientation (not the
    27x27 product). Accepts only candidates with both endpoints within
    *tolerance* Chebyshev distance.
    """

    best: tuple[float, int | None] | None = None
    for query_first, query_second in ((a, b), (b, a)):
        first_coord = _coordinate_tuple(query_first)
        second_coord = _coordinate_tuple(query_second)
        for bin_key in _iter_quantized_neighborhood(query_first, tolerance):
            for marker, stored_a, stored_b, _face_ids in store.get(bin_key, ()):
                distance_a = _chebyshev_distance_3d(first_coord, stored_a)
                distance_b = _chebyshev_distance_3d(second_coord, stored_b)
                if distance_a > tolerance or distance_b > tolerance:
                    continue
                score = max(distance_a, distance_b)
                if best is None or score < best[0]:
                    best = (score, marker)
    return best


def _match_edge_endpoint_pair_for_faces(
    a: Vector,
    b: Vector,
    tolerance: float,
    store: _EdgeEndpointStore,
    possible_target_ids: set[FaceId],
) -> Literal["no_match", "match", "ambiguous"]:
    matches: set[tuple[QuantizedCoordinate, int]] = set()
    for query_first, query_second in ((a, b), (b, a)):
        first_coord = _coordinate_tuple(query_first)
        second_coord = _coordinate_tuple(query_second)
        for bin_key in _iter_quantized_neighborhood(query_first, tolerance):
            for entry_index, (_marker, stored_a, stored_b, face_ids) in enumerate(store.get(bin_key, ())):
                if face_ids is None or not face_ids.intersection(possible_target_ids):
                    continue
                if _chebyshev_distance_3d(first_coord, stored_a) > tolerance:
                    continue
                if _chebyshev_distance_3d(second_coord, stored_b) > tolerance:
                    continue
                matches.add((bin_key, entry_index))
    if not matches:
        return "no_match"
    if len(matches) > 1:
        return "ambiguous"
    return "match"


def _edge_coordinate_key_matches(
    a: Vector,
    b: Vector,
    tolerance: float,
    store: _EdgeEndpointStore,
) -> bool:
    """True when *(a, b)* geometrically matches a stored edge within *tolerance*."""

    return _match_edge_endpoint_pair(a, b, tolerance, store) is not None


def _edge_keys_matching_lookup(
    a: Vector,
    b: Vector,
    tolerance: float,
    store: _EdgeEndpointStore,
) -> int | None:
    """Return the geometrically nearest original marker, or None if no match."""

    match = _match_edge_endpoint_pair(a, b, tolerance, store)
    if match is None:
        return None
    return match[1]


def build_reflected_cutter(
    bm: bmesh.types.BMesh,
    source_edges: Iterable[bmesh.types.BMEdge],
    axis_index: int,
    tolerance: float,
) -> tuple[list[Vector], list[tuple[int, int]], int]:
    """Build reflected loose edges, omitting segments already in the mesh."""

    existing_edges: _EdgeEndpointStore = {}
    for edge in bm.edges:
        _register_edge_endpoint_pair(
            existing_edges,
            edge.verts[0].co,
            edge.verts[1].co,
            tolerance,
        )
    vertex_indices: dict[int, int] = {}
    vertices: list[Vector] = []
    edges: list[tuple[int, int]] = []
    already_present = 0

    bm.verts.ensure_lookup_table()
    bm.verts.index_update()

    for edge in source_edges:
        reflected = (
            mirror_coordinate(edge.verts[0].co, axis_index),
            mirror_coordinate(edge.verts[1].co, axis_index),
        )
        if _edge_coordinate_key_matches(reflected[0], reflected[1], tolerance, existing_edges):
            already_present += 1
            continue

        cutter_edge = []
        for source_vertex, coordinate in zip(edge.verts, reflected, strict=False):
            source_index = source_vertex.index
            cutter_index = vertex_indices.get(source_index)
            if cutter_index is None:
                cutter_index = len(vertices)
                vertex_indices[source_index] = cutter_index
                vertices.append(coordinate)
            cutter_edge.append(cutter_index)
        if cutter_edge[0] != cutter_edge[1]:
            edges.append((cutter_edge[0], cutter_edge[1]))

    return vertices, edges, already_present


def collapsed_offset_target_edge_markers(
    bm: bmesh.types.BMesh,
    source_edges: Iterable[bmesh.types.BMEdge],
    axis_index: int,
    tolerance: float,
) -> tuple[set[int], str]:
    """Find original target edges for an Offset Edge Slide cancelled at zero.

    Blender's Offset macro commits its topology child before Edge Slide.  Esc
    cancels only the slide, leaving two new source loops exactly coincident with
    the selected original loop.  Knife Project cannot cut a coincident edge, so
    this identifies the reflected original target loop for a matching BMesh
    ``offset_edgeloops`` operation.
    """

    marker_layer = bm.edges.layers.int.get(EDGE_ORIGINAL_LAYER)
    if marker_layer is None:
        return set(), "edge marker layer is missing"

    originals_by_endpoint: _EdgeEndpointStore = {}
    new_edges_by_endpoint: _EdgeEndpointStore = {}
    for edge in bm.edges:
        marker = int(edge[marker_layer])
        if marker <= 0:
            _register_edge_endpoint_pair(
                new_edges_by_endpoint,
                edge.verts[0].co,
                edge.verts[1].co,
                tolerance,
            )
        else:
            _register_edge_endpoint_pair(
                originals_by_endpoint,
                edge.verts[0].co,
                edge.verts[1].co,
                tolerance,
                marker=marker,
            )

    target_markers = set()
    matched_nonzero_segments = 0
    for edge in source_edges:
        reflected_a = mirror_coordinate(edge.verts[0].co, axis_index)
        reflected_b = mirror_coordinate(edge.verts[1].co, axis_index)
        if (reflected_a - reflected_b).length <= tolerance:
            # Endpoint-cap output can collapse to a point at factor zero.  The
            # target BMesh op will recreate it from the non-degenerate loop.
            # (Intentionally Euclidean: an edge-length degeneracy test, not a
            # coordinate-identity test.)
            continue
        if _edge_coordinate_key_matches(reflected_a, reflected_b, tolerance, new_edges_by_endpoint):
            return set(), "the target already contains native zero-offset topology"
        marker = _edge_keys_matching_lookup(
            reflected_a,
            reflected_b,
            tolerance,
            originals_by_endpoint,
        )
        if marker is None:
            return set(), "a reflected zero-offset segment has no original target edge"
        target_markers.add(marker)
        matched_nonzero_segments += 1

    if not target_markers or not matched_nonzero_segments:
        return set(), "no reflected original target loop was found"
    return target_markers, ""


def apply_collapsed_offset_topology(
    bm: bmesh.types.BMesh,
    target_edge_markers: set[int],
    *,
    use_cap_endpoint: bool,
) -> tuple[int, str]:
    """Create the target-side topology for a zero-factor Offset operation."""

    marker_layer = bm.edges.layers.int.get(EDGE_ORIGINAL_LAYER)
    if marker_layer is None:
        return 0, "edge marker layer is missing"
    target_edges = [edge for edge in bm.edges if int(edge[marker_layer]) in target_edge_markers]
    if len(target_edges) != len(target_edge_markers):
        return 0, "one or more target loop edges were lost"

    result = bmesh.ops.offset_edgeloops(
        bm,
        edges=target_edges,
        use_cap_endpoint=use_cap_endpoint,
    )
    output_edges = list(result.get("edges", ()))
    if not output_edges:
        return 0, "Blender did not create the target offset topology"
    for edge in output_edges:
        edge.select = False
    bm.normal_update()
    return len(output_edges), ""


def reserve_source_path_marker(bm: bmesh.types.BMesh) -> int:
    """Move source path edges away from zero before Knife Project runs.

    Knife Project creates its new through-face edges with the default integer
    value zero.  Marking the already-created native Knife graph as -1 makes the
    projected destination graph unambiguous, including closed-loop bridge edges.
    """

    marker_layer = bm.edges.layers.int.get(EDGE_ORIGINAL_LAYER)
    if marker_layer is None:
        return 0
    count = 0
    for edge in bm.edges:
        if edge[marker_layer] == 0:
            edge[marker_layer] = -1
            count += 1
    return count


_PROJECTION_STEP_LIMIT = 10_000


def _assign_projection_candidates(
    candidates: list[tuple[float, int, int]],
    destination_count: int,
    destination_pairs: list[tuple[int, int]],
    expected_edge_set: set[tuple[int, int]],
) -> tuple[dict[int, int], dict[int, float], str]:
    """Adjacency-constrained matching of destination vertices to expected ones.

    Replaces the earlier distance-greedy assignment, which provably swapped
    near-coincident vertices and then failed the global adjacency check even
    though a valid solution existed.  The search combines unit propagation
    (a destination with one remaining candidate is fixed and its target is
    removed everywhere), adjacency propagation (the candidates of a fixed
    destination's unassigned neighbors shrink to the expected-graph neighbors
    of its target), and depth-first backtracking over the rest, trying
    candidates in ascending distance order.

    The returned solution is the deterministic first accepted one; total or
    maximum distance optimality is not guaranteed.  Trying candidates
    nearest-first biases the search toward short-distance solutions, which in
    practice suppresses embedding twists via expected-graph automorphisms.
    Extension point if strict optimality ever becomes necessary:
    branch-and-bound over the same propagation core.

    Returns ``(assignment, distances, failure_reason)``; a non-empty reason
    means no assignment was produced.
    """

    allowed: list[dict[int, float]] = [{} for _ in range(destination_count)]
    for distance, destination_id, expected_id in candidates:
        previous = allowed[destination_id].get(expected_id)
        if previous is None or distance < previous:
            allowed[destination_id][expected_id] = distance
    if any(not options for options in allowed):
        return {}, {}, "could not match every projected graph vertex"
    initial_allowed = [dict(options) for options in allowed]

    destination_adjacency: list[set[int]] = [set() for _ in range(destination_count)]
    for a, b in destination_pairs:
        destination_adjacency[a].add(b)
        destination_adjacency[b].add(a)
    expected_adjacency: dict[int, set[int]] = defaultdict(set)
    for a, b in expected_edge_set:
        expected_adjacency[a].add(b)
        expected_adjacency[b].add(a)

    assignment: dict[int, int] = {}
    used: set[int] = set()
    steps = 0

    def _sorted_options(destination_id: int) -> list[tuple[float, int]]:
        return sorted(
            (distance, expected_id)
            for expected_id, distance in allowed[destination_id].items()
            if expected_id not in used
        )

    def _assign_and_propagate(destination_id: int, expected_id: int, trail: list) -> bool:
        """Fix one pair, then propagate; False on contradiction."""

        queue = [(destination_id, expected_id)]
        while queue:
            current, target = queue.pop()
            if current in assignment:
                if assignment[current] != target:
                    return False
                continue
            if target in used:
                return False
            for neighbor in destination_adjacency[current]:
                fixed = assignment.get(neighbor)
                if fixed is not None and fixed not in expected_adjacency[target]:
                    return False
            assignment[current] = target
            used.add(target)
            trail.append(current)
            for neighbor in destination_adjacency[current]:
                if neighbor in assignment:
                    continue
                options = allowed[neighbor]
                restricted = {
                    expected: distance
                    for expected, distance in options.items()
                    if expected in expected_adjacency[target]
                }
                if len(restricted) != len(options):
                    trail.append((neighbor, options))
                    allowed[neighbor] = restricted
                available = [(distance, expected) for expected, distance in restricted.items() if expected not in used]
                if not available:
                    return False
                if len(available) == 1:
                    queue.append((neighbor, min(available)[1]))
        return True

    def _undo(trail: list) -> None:
        while trail:
            item = trail.pop()
            if isinstance(item, tuple):
                neighbor, previous_options = item
                allowed[neighbor] = previous_options
            else:
                used.discard(assignment.pop(item))

    # Root pass: destinations that are unique from the start.  A contradiction
    # here means no adjacency-consistent complete assignment exists at all.
    root_trail: list = []
    for destination_id in range(destination_count):
        if destination_id in assignment:
            continue
        options = _sorted_options(destination_id)
        if not options:
            return {}, {}, "could not match every projected graph vertex"
        if len(options) == 1 and not _assign_and_propagate(destination_id, options[0][1], root_trail):
            return {}, {}, "graph adjacency mismatch"

    def _choose_destination() -> tuple[int, list[tuple[float, int]]] | None:
        best: tuple[int, list[tuple[float, int]]] | None = None
        for destination_id in range(destination_count):
            if destination_id in assignment:
                continue
            options = _sorted_options(destination_id)
            if best is None or len(options) < len(best[1]):
                best = (destination_id, options)
                if len(options) <= 1:
                    break
        return best

    frames: list[list] = []
    advancing = len(assignment) < destination_count
    while len(assignment) < destination_count:
        if advancing:
            chosen = _choose_destination()
            assert chosen is not None
            frames.append([chosen[0], chosen[1], 0, []])
        if not frames:
            return {}, {}, "graph adjacency mismatch"
        frame = frames[-1]
        destination_id, options, _index, trail = frame
        _undo(trail)
        placed = False
        while frame[2] < len(options):
            _distance, expected_id = options[frame[2]]
            frame[2] += 1
            if expected_id in used:
                continue
            if len(options) > 1:
                steps += 1
                if steps > _PROJECTION_STEP_LIMIT:
                    return {}, {}, "ambiguous projection correspondence"
            if _assign_and_propagate(destination_id, expected_id, trail):
                placed = True
                break
            _undo(trail)
        if placed:
            advancing = True
        else:
            frames.pop()
            advancing = False

    distances = {
        destination_id: initial_allowed[destination_id][expected_id]
        for destination_id, expected_id in assignment.items()
    }
    return dict(assignment), distances, ""


def _nearby_projection_candidates(
    destination_vertices: list[bmesh.types.BMVert],
    destination_degree: list[int],
    expected_vertices: list[Vector],
    expected_degree: list[int],
    snap_limit: float,
    existing_limit: float,
    preexisting_vertex_keys: set[int],
) -> list[tuple[float, int, int]]:
    """Return only degree-compatible expected vertices within snapping range."""

    expected_ids_by_degree: dict[int, list[int]] = defaultdict(list)
    for expected_id, degree in enumerate(expected_degree):
        expected_ids_by_degree[degree].append(expected_id)

    trees = {}
    for degree, expected_ids in expected_ids_by_degree.items():
        tree = KDTree(len(expected_ids))
        for expected_id in expected_ids:
            tree.insert(expected_vertices[expected_id], expected_id)
        tree.balance()
        trees[degree] = tree

    candidates = []
    for destination_id, vertex in enumerate(destination_vertices):
        tree = trees.get(destination_degree[destination_id])
        if tree is None:
            continue
        limit = existing_limit if hash(vertex) in preexisting_vertex_keys else snap_limit
        # Include points on the numerical boundary of the accepted range.
        # Intentionally Euclidean: the KDTree radius only collects candidates
        # and decides nothing.  It shares the metric and the per-vertex limit
        # with the final distance validation in snap_projected_graph, so the
        # radius is a complete bound on the acceptable search space.
        search_radius = limit * (1.0 + 1.0e-12) + 1.0e-15
        for _coordinate, expected_id, distance in tree.find_range(
            vertex.co,
            search_radius,
        ):
            candidates.append((distance, destination_id, expected_id))
    return candidates


def _mapped_projection_edge_set(
    destination_pairs: list[tuple[int, int]],
    assignment: dict[int, int],
) -> set[tuple[int, int]]:
    mapped = set()
    for a, b in destination_pairs:
        ma, mb = assignment[a], assignment[b]
        mapped.add((ma, mb) if ma <= mb else (mb, ma))
    return mapped


def snap_projected_graph(
    bm: bmesh.types.BMesh,
    expected_vertices: list[Vector],
    expected_edges: list[tuple[int, int]],
    tolerance: float,
    preexisting_vertex_keys: set[int] | None = None,
) -> tuple[bool, float, str]:
    """Snap Knife Project's destination graph to exact reflected coordinates.

    Knife Project is screen-space, so even a cutter that lies exactly on the
    destination surface can return small projection errors.  The new graph is
    identified by its zero edge marker, matched by degree and proximity, checked
    for graph isomorphism, and only then snapped.
    """

    marker_layer = bm.edges.layers.int.get(EDGE_ORIGINAL_LAYER)
    if marker_layer is None:
        return False, 0.0, "edge marker layer is missing"

    destination_edges = [edge for edge in bm.edges if edge[marker_layer] == 0]
    destination_vertices = []
    destination_index = {}
    destination_pairs = []
    bm.verts.ensure_lookup_table()
    bm.verts.index_update()
    for edge in destination_edges:
        pair = []
        for vertex in edge.verts:
            key = vertex.index
            index = destination_index.get(key)
            if index is None:
                index = len(destination_vertices)
                destination_index[key] = index
                destination_vertices.append(vertex)
            pair.append(index)
        destination_pairs.append((pair[0], pair[1]))

    if len(destination_edges) != len(expected_edges):
        return (
            False,
            0.0,
            f"projected edge count {len(destination_edges)} != expected {len(expected_edges)}",
        )
    if len(destination_vertices) != len(expected_vertices):
        return (
            False,
            0.0,
            f"projected vertex count {len(destination_vertices)} != expected {len(expected_vertices)}",
        )
    if not expected_vertices:
        return True, 0.0, ""

    expected_degree = [0] * len(expected_vertices)
    for a, b in expected_edges:
        expected_degree[a] += 1
        expected_degree[b] += 1
    destination_degree = [0] * len(destination_vertices)
    for a, b in destination_pairs:
        destination_degree[a] += 1
        destination_degree[b] += 1

    expected_edge_set = {(a, b) if a <= b else (b, a) for a, b in expected_edges}
    expected_lengths = [
        (expected_vertices[a] - expected_vertices[b]).length
        for a, b in expected_edges
        if (expected_vertices[a] - expected_vertices[b]).length > tolerance
    ]
    minimum_edge_length = min(expected_lengths, default=max(tolerance, 1.0e-6))
    snap_limit = max(tolerance * 20.0, minimum_edge_length * 0.02)
    existing_limit = max(tolerance * 2.0, 1.0e-9)
    preexisting_vertex_keys = preexisting_vertex_keys or set()

    # Long Loop Cut graphs often contain thousands of same-degree vertices.
    # Searching only their local KDTree neighborhood keeps the normal path near
    # O(n log n).  The radius search is a *complete* candidate enumeration: it
    # uses the same Euclidean metric and the same per-vertex limits as the
    # final distance validation below, so an assignment using any vertex
    # outside the radius would necessarily be rejected there.  No wider
    # fallback can add an acceptable solution.  Degree compatibility is a
    # necessary condition of the final graph isomorphism check and therefore
    # never narrows the acceptable space either.
    candidates = _nearby_projection_candidates(
        destination_vertices,
        destination_degree,
        expected_vertices,
        expected_degree,
        snap_limit,
        existing_limit,
        preexisting_vertex_keys,
    )
    assignment, distances, assignment_reason = _assign_projection_candidates(
        candidates,
        len(destination_vertices),
        destination_pairs,
        expected_edge_set,
    )
    if assignment_reason:
        return False, 0.0, assignment_reason

    if len(assignment) != len(destination_vertices):
        return False, 0.0, "could not match every projected graph vertex"

    mapped_edge_set = _mapped_projection_edge_set(destination_pairs, assignment)
    if mapped_edge_set != expected_edge_set:
        return False, max(distances.values(), default=0.0), "graph adjacency mismatch"

    # Intentionally Euclidean, and it must stay that way: sharing this metric
    # and these limits with the KDTree radius search above is exactly what
    # makes that search a complete candidate enumeration.
    maximum_distance = max(distances.values(), default=0.0)
    existing_error = max(
        (
            distances[destination_id]
            for destination_id, vertex in enumerate(destination_vertices)
            if hash(vertex) in preexisting_vertex_keys
        ),
        default=0.0,
    )
    movable_error = max(
        (
            distances[destination_id]
            for destination_id, vertex in enumerate(destination_vertices)
            if hash(vertex) not in preexisting_vertex_keys
        ),
        default=0.0,
    )
    if existing_error > existing_limit:
        return (
            False,
            existing_error,
            f"existing endpoint mismatch {existing_error:.6g} exceeds {existing_limit:.6g}",
        )
    if movable_error > snap_limit:
        return (
            False,
            movable_error,
            f"projection error {movable_error:.6g} exceeds safe snap limit {snap_limit:.6g}",
        )

    for destination_id, expected_id in assignment.items():
        vertex = destination_vertices[destination_id]
        if hash(vertex) not in preexisting_vertex_keys:
            vertex.co = expected_vertices[expected_id]
    bm.normal_update()
    return True, maximum_distance, ""
