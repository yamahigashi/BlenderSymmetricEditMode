from __future__ import annotations

import logging
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal, overload

import bmesh
import numpy
from mathutils import Vector

from . import stitch_common
from ._types import CarrierFrameMap, CarrierFrameSnapshot, FaceId, MirrorFaceMap
from .face_mapping import _canonical_carrier_frames
from .layer_names import EDGE_ORIGINAL_LAYER, EDGE_SELECTION_LAYER, FACE_ID_LAYER, VERT_SELECTION_LAYER
from .matching import coordinates_match, mirror_coordinate

_LOGGER = logging.getLogger(__name__)


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
    first = stitch_common._mirror_invariant_endpoint_key(edge.verts[0].co, axis_index)
    second = stitch_common._mirror_invariant_endpoint_key(edge.verts[1].co, axis_index)
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
    return min(matches, key=lambda index: stitch_common._coordinate_tuple(endpoints[index]))


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
        if stitch_common.is_self_mirrored_edge(edge, axis_index, tolerance):
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
                    coordinate=result.coordinate,  # ty: ignore[invalid-argument-type]  # fake-bpy/ty stub limitation
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
                        coordinate=result.coordinate,  # ty: ignore[invalid-argument-type]  # fake-bpy/ty stub limitation
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
                    coordinate=result.coordinate,  # ty: ignore[invalid-argument-type]  # fake-bpy/ty stub limitation
                    positive=tuple(positive_occurrences),
                    negative=tuple(negative_occurrences),
                )
            )

    if not raw_crossings:
        return [], ""

    points = [crossing.coordinate for crossing in raw_crossings]
    clusters: list[_MirroredPathCrossingCluster] = []
    for indices in stitch_common.cluster_points_by_tolerance(points, tolerance):
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
                positive_coordinate=representative,  # ty: ignore[invalid-argument-type]  # fake-bpy/ty stub limitation
                negative_coordinate=mirrored,  # ty: ignore[invalid-argument-type]  # fake-bpy/ty stub limitation
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
    selection_mutations: stitch_common.SelectionMutationSummary = field(
        default_factory=stitch_common.SelectionMutationSummary
    )


@overload
def apply_mirrored_path_crossings(
    bm: bmesh.types.BMesh,
    plan: Sequence[_MirroredPathCrossingCluster],
    *,
    cache_positions: Mapping[int, int] | None = ...,
    return_summary: Literal[False] = ...,
) -> tuple[int, str]: ...
@overload
def apply_mirrored_path_crossings(
    bm: bmesh.types.BMesh,
    plan: Sequence[_MirroredPathCrossingCluster],
    *,
    cache_positions: Mapping[int, int] | None = ...,
    return_summary: Literal[True],
) -> tuple[int, str, CrossingMutationSummary]: ...
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

    selection_tracker = stitch_common._SelectionMutationTracker()

    def _result(
        count: int,
        reason: str,
        summary: CrossingMutationSummary | None = None,
    ) -> tuple[int, str] | tuple[int, str, CrossingMutationSummary]:
        if return_summary:
            return (
                count,
                reason,
                summary
                or CrossingMutationSummary((), selection_mutations=selection_tracker.finish(complete=not reason)),
            )
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
            selection_tracker.add_edge(edge)
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
                selection_tracker.add_edge(half_edge)
                half_edge[marker_layer] = marker
                half_edge.select = selected
                half_edge[edge_selection_layer] = int(selected)
            new_vertex.select = selected
            selection_tracker.add_vertex(new_vertex)
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
        selection_tracker.add_vertex(survivor)
        survivor.co = coordinate.copy()
        _index_rebin(survivor, old_survivor_co, coordinate)
        for vertex in list(vertices):
            if vertex == survivor:
                continue
            selection_tracker.add_vertex(vertex)
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
            selection_tracker.add_vertex(vertex)
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
        selection_mutations=selection_tracker.finish(complete=not unexpected_topology_change),
    )
    return _result(len(applications), "", summary)


def _plan_tolerance(plan: Sequence[_MirroredPathCrossingCluster]) -> float:
    return plan[0].tolerance if plan else 0.0
