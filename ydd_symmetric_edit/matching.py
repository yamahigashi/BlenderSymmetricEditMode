from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import Literal

import numpy  # type: ignore
from mathutils import Vector
from mathutils.kdtree import KDTree

from ._types import (
    Coordinate3D,
    MirrorOverlap,
    OverlapClassification,
    QuantizedCoordinate,
)

AXIS_INDEX = {"X": 0, "Y": 1, "Z": 2}
MESH_SYMMETRY_PROPERTIES = (
    ("X", 0, "use_mesh_mirror_x"),
    ("Y", 1, "use_mesh_mirror_y"),
    ("Z", 2, "use_mesh_mirror_z"),
)


def enabled_mesh_symmetry_axes(obj) -> tuple[tuple[str, int], ...]:
    """Read Blender's own Edit Mesh symmetry toggles from *obj*."""

    return tuple(
        (axis, axis_index)
        for axis, axis_index, property_name in MESH_SYMMETRY_PROPERTIES
        if bool(getattr(obj, property_name, False))
    )


def mirror_coordinate(co: Vector, axis_index: int) -> Vector:
    """Return *co* reflected around an object-local coordinate plane."""

    result = co.copy()
    result[axis_index] = -result[axis_index]
    return result


def _quantized_coordinate(co: Vector, tolerance: float) -> QuantizedCoordinate:
    """Primary floor-bin key for *co*.

    Floor bins of width ``tolerance`` are the storage key. Callers that need
    tolerance-robust equality must probe the neighborhood (see
    ``_iter_quantized_neighborhood``) or use geometric verification; exclusive
    ``round`` bins previously rejected pairs whose real difference was far
    below tolerance but straddled a bin boundary.
    """

    inverse = 1.0 / max(tolerance, 1.0e-12)
    return QuantizedCoordinate(
        x=math.floor(co[0] * inverse),
        y=math.floor(co[1] * inverse),
        z=math.floor(co[2] * inverse),
    )


def _iter_quantized_neighborhood(
    co: Vector,
    tolerance: float,
) -> Iterator[QuantizedCoordinate]:
    """Yield primary floor bin and its 26 Chebyshev neighbors (27 total).

    Invariant: if every component of two coordinates differs by at most
    ``tolerance``, each primary bin lies in the other's neighborhood. If any
    component differs by at least ``2 * tolerance``, the neighborhoods cannot
    contain the other primary bin.
    """

    primary = _quantized_coordinate(co, tolerance)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                yield QuantizedCoordinate(
                    x=primary.x + dx,
                    y=primary.y + dy,
                    z=primary.z + dz,
                )


def _coordinate_3d(co: Vector) -> Coordinate3D:
    return Coordinate3D(
        x=float(co[0]),
        y=float(co[1]),
        z=float(co[2]),
    )


def _chebyshev_distance_3d(
    first: tuple[float, float, float],
    second: tuple[float, float, float],
) -> float:
    return max(
        abs(first[0] - second[0]),
        abs(first[1] - second[1]),
        abs(first[2] - second[2]),
    )


def coordinates_match(first, second, tolerance: float) -> bool:
    """The single definition of coordinate identity: per-component (Chebyshev).

    Every "are these the same point" acceptance test in the add-on must go
    through this predicate.  Geometric quantities (segment distances, edge
    lengths, KDTree radii) intentionally stay Euclidean and are marked so at
    their use sites.
    """

    return (
        _chebyshev_distance_3d(
            (float(first[0]), float(first[1]), float(first[2])),
            (float(second[0]), float(second[1]), float(second[2])),
        )
        <= tolerance
    )


class VertexMirrorLookup:
    """Index of registered coordinates for mirror-plane counterpart lookup.

    Built by :func:`build_vertex_mirror_lookup`. KDTree supplies candidates;
    stored double-precision tuples decide acceptance and ordering.
    """

    def __init__(
        self,
        *,
        axis_index: int,
        tolerance: float,
        coords: tuple[tuple[float, float, float], ...],
        tree: KDTree | None = None,
    ) -> None:
        self._axis_index = axis_index
        self._tolerance = tolerance
        self._coords = coords
        self._tree = tree
        self._on_plane_indices: frozenset[int] | None = None
        self._batch_index: dict | Literal[False] | None = None
        self._batch_path_count = 0

    def _ensure_tree(self) -> KDTree:
        tree = self._tree
        if tree is None:
            tree = KDTree(len(self._coords))
            for index, stored in enumerate(self._coords):
                tree.insert(Vector(stored), index)
            tree.balance()
            self._tree = tree
        return tree

    def _on_plane_registered(self) -> frozenset[int]:
        """Registered indices whose stored coordinate lies on the plane."""

        cached = self._on_plane_indices
        if cached is None:
            axis_index = self._axis_index
            tolerance = self._tolerance
            cached = frozenset(
                index for index, stored in enumerate(self._coords) if abs(stored[axis_index]) <= tolerance
            )
            self._on_plane_indices = cached
        return cached

    def find(self, co: Vector) -> int | None:
        """Return the index of the registered coord matching ``mirror(co)``.

        Accepts only candidates whose Chebyshev distance is at most
        *tolerance*. When several qualify, returns the nearest candidate.
        """

        expected = mirror_coordinate(co, self._axis_index)
        expected_coord = (float(expected[0]), float(expected[1]), float(expected[2]))
        candidates = self._candidates_for(expected_coord)
        return None if not candidates else candidates[0][1]

    def is_on_plane(self, co: Vector) -> bool:
        """True when *co* lies on the mirror plane within *tolerance*."""

        return abs(co[self._axis_index]) <= self._tolerance

    def find_all_mirrored(self, coords: Sequence[Vector]) -> tuple[int | None, ...]:
        """Injective assignment of registered vertices to each coord's mirror image.

        Unlike per-query :meth:`find`, no two queries can resolve to the same
        registered vertex: assignments are solved per connected candidate
        component for the minimum-total-distance complete cover, and ties or
        incomplete covers reject the whole component (``None``) rather than
        silently picking one interpretation.

        Fixed rule: the plane partitions both sides of the assignment.
        On-plane queries resolve by *direct* self-correspondence against
        on-plane registered vertices only, and off-plane queries never match
        an on-plane registered vertex — regardless of whether that vertex
        also appears as a query in this batch.  (A per-batch reservation
        would leak on partial batches: a history-only query list contains no
        on-plane queries, so nothing would be reserved, and a near-plane
        reflection could silently steal an on-plane vertex.)
        """

        results: list[int | None] = [None] * len(coords)
        query = self._query_matrix(coords)
        if query is not None:
            # The float64 image of every float32 coordinate is exact, so the
            # vectorized plane predicate and axis negation match is_on_plane
            # and mirror_coordinate value-for-value.
            on_plane_mask = numpy.abs(query[:, self._axis_index]) <= self._tolerance
            on_plane_queries = numpy.flatnonzero(on_plane_mask)
            if len(on_plane_queries):
                resolved = self._resolve_injective(query[on_plane_mask], plane_side=True)
                for query_index, target in zip(on_plane_queries.tolist(), resolved, strict=True):
                    results[query_index] = target
            off_queries = numpy.flatnonzero(~on_plane_mask)
            if len(off_queries):
                mirrored = query[~on_plane_mask]
                mirrored[:, self._axis_index] = -mirrored[:, self._axis_index]
                resolved = self._resolve_injective(mirrored, plane_side=False)
                for query_index, target in zip(off_queries.tolist(), resolved, strict=True):
                    results[query_index] = target
            return tuple(results)

        on_plane_entries = [(index, co) for index, co in enumerate(coords) if self.is_on_plane(co)]
        if on_plane_entries:
            resolved = self._resolve_injective(
                [co for _index, co in on_plane_entries],
                plane_side=True,
            )
            for (query_index, _co), target in zip(on_plane_entries, resolved, strict=True):
                results[query_index] = target
        off_entries = [
            (index, mirror_coordinate(co, self._axis_index))
            for index, co in enumerate(coords)
            if not self.is_on_plane(co)
        ]
        if off_entries:
            resolved = self._resolve_injective(
                [position for _index, position in off_entries],
                plane_side=False,
            )
            for (query_index, _position), target in zip(off_entries, resolved, strict=True):
                results[query_index] = target
        return tuple(results)

    def find_all_direct(self, coords: Sequence[Vector]) -> tuple[int | None, ...]:
        """Injective assignment of registered vertices to each coordinate itself.

        No reflection is applied; this replaces the historical
        ``find(mirror_coordinate(co))`` double-reflection idiom.
        """

        query = self._query_matrix(coords)
        return tuple(self._resolve_injective(query if query is not None else list(coords)))

    @staticmethod
    def _query_matrix(coords: Sequence) -> numpy.ndarray | None:
        """Vectorized float64 image of *coords*, or ``None`` when unconvertible."""

        if len(coords) == 0:
            return numpy.zeros((0, 3), dtype=numpy.float64)
        if isinstance(coords, numpy.ndarray):
            matrix = numpy.asarray(coords, dtype=numpy.float64)
        else:
            # Vector.to_tuple() is a C call and ~4x faster than letting numpy
            # iterate each Vector through the sequence protocol.
            try:
                matrix = numpy.array([position.to_tuple() for position in coords], dtype=numpy.float64)
            except AttributeError:
                try:
                    matrix = numpy.array(
                        [(float(position[0]), float(position[1]), float(position[2])) for position in coords],
                        dtype=numpy.float64,
                    )
                except (TypeError, ValueError, IndexError):
                    return None
        if matrix.ndim != 2 or matrix.shape != (len(coords), 3):
            return None
        return matrix

    def _candidates_for(self, position) -> list[tuple[float, int]]:
        """Distance-ascending registered candidates within *tolerance*."""

        position_coord = (float(position[0]), float(position[1]), float(position[2]))
        tolerance = self._tolerance
        radius = math.sqrt(3.0) * tolerance * (1.0 + 1.0e-3)
        found = []
        for _coordinate, index, _distance in self._ensure_tree().find_range(Vector(position_coord), radius):
            stored = self._coords[index]
            distance = _chebyshev_distance_3d(position_coord, stored)
            if distance <= tolerance:
                found.append((distance, index))
        found.sort()
        return found

    def _resolve_injective(
        self,
        positions: Sequence,
        plane_side: bool | None = None,
    ) -> list[int | None]:
        """Solve an injective nearest assignment for a list of query positions.

        ``plane_side`` restricts the registered candidates: ``True`` keeps
        only on-plane vertices, ``False`` only off-plane vertices, ``None``
        allows all (the direct search).  Queries are grouped into connected
        components of the bipartite query/candidate graph; components never
        share candidates, so solving each component independently keeps the
        whole result injective.
        """

        arrays = self._batch_candidate_arrays(positions, plane_side)
        results: list[int | None] = [None] * len(positions)
        if arrays is not None:
            query_indices, registered_indices, distances = arrays
            if len(query_indices) == 0:
                return results
            query_counts = numpy.bincount(query_indices, minlength=len(positions))
            target_counts = numpy.bincount(registered_indices, minlength=len(self._coords))
            starts = numpy.cumsum(query_counts) - query_counts
            single = query_counts == 1
            single_queries = numpy.flatnonzero(single)
            single_targets = registered_indices[starts[single_queries]]
            trivial = target_counts[single_targets] == 1
            assigned = numpy.full(len(positions), -1, dtype=numpy.int64)
            assigned[single_queries[trivial]] = single_targets[trivial]
            remainder_mask = query_counts > 0
            remainder_mask[single_queries[trivial]] = False
            remainder = numpy.flatnonzero(remainder_mask).tolist()
            results = [None if target < 0 else target for target in assigned.tolist()]
            if not remainder:
                return results
            candidate_lists: dict[int, list[tuple[float, int]]] = {}
            registered_list = registered_indices.tolist()
            distance_list = distances.tolist()
            for query in remainder:
                begin = int(starts[query])
                end = begin + int(query_counts[query])
                candidate_lists[query] = [
                    (float(distance_list[position]), int(registered_list[position])) for position in range(begin, end)
                ]
            queue = remainder
        else:
            if plane_side is None:
                candidate_lists = {
                    query: list(self._candidates_for(position)) for query, position in enumerate(positions)
                }
            else:
                on_plane_targets = self._on_plane_registered()
                candidate_lists = {
                    query: [
                        (distance, index)
                        for distance, index in self._candidates_for(position)
                        if (index in on_plane_targets) == plane_side
                    ]
                    for query, position in enumerate(positions)
                }
            queue = list(range(len(positions)))

        queries_by_target: dict[int, list[int]] = defaultdict(list)
        for query in queue:
            for _distance, target in candidate_lists[query]:
                queries_by_target[target].append(query)

        visited: dict[int, bool] = dict.fromkeys(queue, False)
        for query in queue:
            candidates = candidate_lists[query]
            if len(candidates) == 1 and queries_by_target[candidates[0][1]] == [query]:
                results[query] = candidates[0][1]
                visited[query] = True
        for start in queue:
            if visited[start]:
                continue
            visited[start] = True
            if not candidate_lists[start]:
                continue
            component = []
            component_targets: set[int] = set()
            pending = [start]
            while pending:
                query = pending.pop()
                component.append(query)
                for _distance, target in candidate_lists[query]:
                    if target in component_targets:
                        continue
                    component_targets.add(target)
                    for other in queries_by_target[target]:
                        if not visited[other]:
                            visited[other] = True
                            pending.append(other)
            assignment = _solve_injective_component(component, candidate_lists)
            if assignment is not None:
                for query, target in assignment.items():
                    results[query] = target
        return results

    def _registered_batch_index(self):
        """Registered-side packed-bin index, built once; ``False`` when unpackable."""

        cached = self._batch_index
        if cached is not None:
            return cached
        matrix = numpy.array(self._coords, dtype=numpy.float64).reshape(len(self._coords), 3)
        inverse = 1.0 / max(self._tolerance, 1.0e-12)
        scaled = matrix * inverse
        if not numpy.isfinite(scaled).all() or numpy.any(numpy.abs(scaled) >= 2**62):
            self._batch_index = False
            return False
        bins = numpy.floor(scaled).astype(numpy.int64)
        mins = bins.min(axis=0)
        span_x, span_y, span_z = (int(size) for size in (bins.max(axis=0) - mins + 1))
        # Collision-free 1D packing: shift bins into the registered bounding
        # box and refuse the batch path when the span product would overflow
        # the int64 key space.
        if span_x * span_y * span_z >= 2**63:
            self._batch_index = False
            return False
        stride_y = span_z
        stride_x = span_y * span_z
        shifted = bins - mins
        keys = shifted[:, 0] * stride_x + shifted[:, 1] * stride_y + shifted[:, 2]
        order = numpy.argsort(keys, kind="stable")
        cached = {
            "matrix": matrix,
            "mins": mins,
            "spans": numpy.asarray((span_x, span_y, span_z), dtype=numpy.int64),
            "stride_x": stride_x,
            "stride_y": stride_y,
            "order": order,
            "sorted_keys": keys[order],
            "on_plane": numpy.abs(matrix[:, self._axis_index]) <= self._tolerance,
        }
        self._batch_index = cached
        return cached

    def _batch_candidate_arrays(self, positions: Sequence, plane_side: bool | None):
        """Sorted (query, registered, distance) candidate arrays, or ``None``."""

        if isinstance(positions, numpy.ndarray) and positions.dtype == numpy.float64:
            query_coords = positions.reshape(len(positions), 3)
        else:
            query_coords = numpy.asarray(
                [(float(position[0]), float(position[1]), float(position[2])) for position in positions],
                dtype=numpy.float64,
            ).reshape(len(positions), 3)
        inverse = 1.0 / max(self._tolerance, 1.0e-12)
        scaled_queries = query_coords * inverse
        if not numpy.isfinite(scaled_queries).all() or numpy.any(numpy.abs(scaled_queries) >= 2**62):
            return None

        empty = (
            numpy.empty(0, dtype=numpy.int64),
            numpy.empty(0, dtype=numpy.int64),
            numpy.empty(0, dtype=numpy.float64),
        )
        if len(self._coords):
            index = self._registered_batch_index()
            if index is False:
                return None
        if len(positions) == 0 or len(self._coords) == 0:
            self._batch_path_count += 1
            return empty

        query_bins = numpy.floor(scaled_queries).astype(numpy.int64)
        # The z-neighborhood is contiguous in packed-key space, so the 27
        # neighbor bins collapse into 9 (dx, dy) windows of a clamped
        # three-bin z range; clamping keeps ranges from bleeding into the
        # neighboring (x, y) row.
        offsets = numpy.asarray([(x, y) for x in (-1, 0, 1) for y in (-1, 0, 1)], dtype=numpy.int64)
        shifted_xy = (query_bins[:, None, :2] + offsets[None, :, :]).reshape(-1, 2) - index["mins"][:2]
        shifted_z = numpy.repeat(query_bins[:, 2] - index["mins"][2], len(offsets))
        span_z = int(index["spans"][2])
        valid = numpy.flatnonzero(
            (shifted_xy >= 0).all(axis=1)
            & (shifted_xy < index["spans"][:2]).all(axis=1)
            & (shifted_z >= -1)
            & (shifted_z <= span_z)
        )
        if len(valid) == 0:
            self._batch_path_count += 1
            return empty
        valid_xy = shifted_xy[valid]
        low_z = numpy.clip(shifted_z[valid] - 1, 0, span_z - 1)
        high_z = numpy.clip(shifted_z[valid] + 1, 0, span_z - 1)
        row_keys = valid_xy[:, 0] * index["stride_x"] + valid_xy[:, 1] * index["stride_y"]
        sorted_keys = index["sorted_keys"]
        left = numpy.searchsorted(sorted_keys, row_keys + low_z, side="left")
        right = numpy.searchsorted(sorted_keys, row_keys + high_z, side="right")
        counts = right - left
        total = int(counts.sum())
        if total == 0:
            self._batch_path_count += 1
            return empty

        repeated_windows = numpy.repeat(numpy.arange(len(counts), dtype=numpy.int64), counts)
        window_starts = numpy.repeat(numpy.cumsum(counts) - counts, counts)
        offsets_in_window = numpy.arange(total, dtype=numpy.int64) - window_starts
        registered_indices = index["order"][left[repeated_windows] + offsets_in_window]
        query_indices = valid[repeated_windows] // len(offsets)
        matrix = index["matrix"]
        distances = numpy.max(numpy.abs(matrix[registered_indices] - query_coords[query_indices]), axis=1)
        within = distances <= self._tolerance
        if plane_side is not None:
            within &= index["on_plane"][registered_indices] == plane_side
        query_indices = query_indices[within]
        registered_indices = registered_indices[within]
        distances = distances[within]
        # Window traversal is already query-grouped; a per-query (distance,
        # index) sort is only needed when some query has several candidates.
        if len(query_indices) and int(numpy.bincount(query_indices).max()) > 1:
            candidate_order = numpy.lexsort((registered_indices, distances, query_indices))
            query_indices = query_indices[candidate_order]
            registered_indices = registered_indices[candidate_order]
            distances = distances[candidate_order]
        self._batch_path_count += 1
        return query_indices, registered_indices, distances

    def _batch_candidates(
        self,
        positions: Sequence,
        plane_side: bool | None = None,
    ) -> list[list[tuple[float, int]]] | None:
        """Return all registered candidates for a batch, or ``None`` to fallback."""

        arrays = self._batch_candidate_arrays(positions, plane_side)
        if arrays is None:
            return None
        query_indices, registered_indices, distances = arrays
        candidates: list[list[tuple[float, int]]] = [[] for _ in positions]
        for query_index, registered_index, distance in zip(
            query_indices.tolist(), registered_indices.tolist(), distances.tolist(), strict=True
        ):
            candidates[query_index].append((float(distance), int(registered_index)))
        return candidates


class VertexRegistry:
    """Resolution-free float64 vertex index for the plane-split oracle.

    The registry captures only coordinates, side masks, and the packed-bin
    search index. Candidate and claimant methods return parallel arrays of
    ``(query index, target index, Chebyshev distance)`` and never retain an
    assignment or any other resolution result. They return ``None`` when the
    vectorized index cannot represent non-finite or out-of-range coordinates.
    """

    def __init__(self, coords64: numpy.ndarray, axis_index: int, tolerance: float) -> None:
        coords = numpy.asarray(coords64, dtype=numpy.float64)
        if coords.ndim != 2 or coords.shape[1:] != (3,):
            raise ValueError("coords64 must have shape (count, 3)")
        if axis_index not in (0, 1, 2):
            raise ValueError("axis_index must be 0, 1, or 2")

        self.coords64 = coords
        self.axis_index = int(axis_index)
        self.tolerance = float(tolerance)
        axis = coords[:, self.axis_index]
        self._on_plane = numpy.abs(axis) <= self.tolerance
        self._positive = ~self._on_plane & (axis > self.tolerance)
        self._negative = ~self._on_plane & (axis < -self.tolerance)
        self.on_plane_indices = numpy.flatnonzero(self._on_plane)
        self.positive_indices = numpy.flatnonzero(self._positive)
        self.negative_indices = numpy.flatnonzero(self._negative)
        self._offsets = numpy.asarray(
            [(x, y) for x in (-1, 0, 1) for y in (-1, 0, 1)],
            dtype=numpy.int64,
        )
        self._index = self._build_index()

    @staticmethod
    def _empty_candidate_arrays():
        empty_indices = numpy.empty(0, dtype=numpy.int64)
        return empty_indices, empty_indices.copy(), numpy.empty(0, dtype=numpy.float64)

    def _build_index(self):
        count = len(self.coords64)
        if count == 0:
            return {}
        if not numpy.isfinite(self.coords64).all():
            return None
        inverse = 1.0 / max(self.tolerance, 1.0e-12)
        scaled = self.coords64 * inverse
        if numpy.any(numpy.abs(scaled) >= 2**62):
            return None
        bins = numpy.floor(scaled).astype(numpy.int64)
        mins = bins.min(axis=0)
        spans = bins.max(axis=0) - mins + 1
        span_x, span_y, span_z = (int(span) for span in spans)
        if span_x * span_y * span_z >= 2**63:
            return None
        stride_y = span_z
        stride_x = span_y * span_z
        shifted = bins - mins
        keys = shifted[:, 0] * stride_x + shifted[:, 1] * stride_y + shifted[:, 2]
        order = numpy.argsort(keys, kind="stable")
        return {
            "inverse": inverse,
            "mins": mins,
            "spans": numpy.asarray((span_x, span_y, span_z), dtype=numpy.int64),
            "stride_x": stride_x,
            "stride_y": stride_y,
            "order": order,
            "sorted_keys": keys[order],
        }

    def _query_indices(self, indices) -> numpy.ndarray:
        query_indices = numpy.asarray(indices, dtype=numpy.int64).reshape(-1)
        if len(query_indices) and (numpy.any(query_indices < 0) or numpy.any(query_indices >= len(self.coords64))):
            raise IndexError("vertex query index out of range")
        return query_indices

    @staticmethod
    def _sorted_candidate_arrays(queries, targets, distances):
        if len(queries):
            order = numpy.lexsort((targets, distances, queries))
            queries = queries[order]
            targets = targets[order]
            distances = distances[order]
        return queries, targets, distances

    def _probe(self, query_indices: numpy.ndarray, query_points: numpy.ndarray, target_mask: numpy.ndarray):
        if len(query_indices) == 0:
            return self._empty_candidate_arrays()
        index = self._index
        if index is None:
            return None
        if not index:
            return self._empty_candidate_arrays()

        query_bins = numpy.floor(query_points * index["inverse"]).astype(numpy.int64)
        offsets = self._offsets
        shifted_xy = (query_bins[:, None, :2] + offsets[None, :, :]).reshape(-1, 2) - index["mins"][:2]
        shifted_z = numpy.repeat(query_bins[:, 2] - index["mins"][2], len(offsets))
        span_z = int(index["spans"][2])
        valid = numpy.flatnonzero(
            (shifted_xy >= 0).all(axis=1)
            & (shifted_xy < index["spans"][:2]).all(axis=1)
            & (shifted_z >= -1)
            & (shifted_z <= span_z)
        )
        if len(valid) == 0:
            return self._empty_candidate_arrays()

        low = numpy.clip(shifted_z[valid] - 1, 0, span_z - 1)
        high = numpy.clip(shifted_z[valid] + 1, 0, span_z - 1)
        row = shifted_xy[valid][:, 0] * index["stride_x"] + shifted_xy[valid][:, 1] * index["stride_y"]
        left = numpy.searchsorted(index["sorted_keys"], row + low, side="left")
        right = numpy.searchsorted(index["sorted_keys"], row + high, side="right")
        counts = right - left
        total = int(counts.sum())
        if total == 0:
            return self._empty_candidate_arrays()

        windows = numpy.repeat(numpy.arange(len(counts), dtype=numpy.int64), counts)
        starts = numpy.repeat(numpy.cumsum(counts) - counts, counts)
        within = numpy.arange(total, dtype=numpy.int64) - starts
        targets = index["order"][left[windows] + within]
        local_queries = valid[windows] // len(offsets)
        queries = query_indices[local_queries]
        distances = numpy.max(numpy.abs(self.coords64[targets] - query_points[local_queries]), axis=1)
        keep = (distances <= self.tolerance) & target_mask[targets]
        return self._sorted_candidate_arrays(queries[keep], targets[keep], distances[keep])

    def candidates_on_plane(self, query_indices):
        """Return C0(q) for the on-plane members of *query_indices*."""

        queries = self._query_indices(query_indices)
        queries = queries[self._on_plane[queries]]
        return self._probe(queries, self.coords64[queries], self._on_plane)

    def candidates_off_plane(self, query_indices):
        """Return Coff(q) for the off-plane members of *query_indices*."""

        queries = self._query_indices(query_indices)
        queries = queries[~self._on_plane[queries]]
        points = self.coords64[queries].copy()
        points[:, self.axis_index] = -points[:, self.axis_index]
        return self._probe(queries, points, ~self._on_plane)

    # Both claimant methods rely on Chebyshev mirror symmetry
    # d∞(ρ(q), t) == d∞(ρ(t), q): the reverse relation is the forward
    # relation with columns swapped, so no separate reverse index exists.
    def claimants_on_plane(self, target_indices):
        """Return on-plane queries whose C0 set contains each requested target."""

        arrays = self.candidates_on_plane(target_indices)
        if arrays is None:
            return None
        targets, queries, distances = arrays
        return self._sorted_candidate_arrays(queries, targets, distances)

    def claimants_off_plane(self, target_indices):
        """Return off-plane queries whose Coff set contains each requested target."""

        arrays = self.candidates_off_plane(target_indices)
        if arrays is None:
            return None
        targets, queries, distances = arrays
        return self._sorted_candidate_arrays(queries, targets, distances)


_INJECTIVE_STEP_LIMIT = 2_000


def _solve_injective_component(
    queries: Sequence[int],
    candidate_lists: Mapping[int, Sequence[tuple[float, int]]] | Sequence[Sequence[tuple[float, int]]],
    step_limit: int = _INJECTIVE_STEP_LIMIT,
) -> dict[int, int] | None:
    """Minimum-total-Chebyshev complete assignment for one candidate component.

    Mutual-nearest pre-fixing is deliberately not used: fixing an apparently
    safe pair can destroy the only complete assignment (1-D counterexample
    with tolerance 1.0 — candidates q1→{t1:0.4, t2:0.8}, q2→{t1:0.9, t3:0.6},
    q3→{t3:0.5}: fixing mutual-nearest q1→t1 leaves q2 and q3 fighting over
    t3, yet the complete injective solution q1→t2, q2→t1, q3→t3 exists).
    The component is therefore always solved as a whole.

    Returns ``None`` when no complete cover of the queries exists, when the
    optimum is tied between two distinct assignments (ambiguous), or when the
    search exceeds *step_limit*.
    """

    order = sorted(queries, key=lambda query: (len(candidate_lists[query]), query))
    order_count = len(order)
    if order_count == 0:
        return {}
    if order_count > step_limit:
        return None

    best_cost: float | None = None
    best_assignment: dict[int, int] | None = None
    tie = False
    steps = 0
    used: set[int] = set()
    chosen_targets = [-1] * order_count
    chosen_distances = [0.0] * order_count
    candidate_positions = [0] * order_count
    prefix_costs = [0.0] * (order_count + 1)
    depth = 0
    while depth >= 0:
        if depth == order_count:
            # Leaf costs are compared via fsum with a small relative margin:
            # sequential float addition is order-dependent, so two genuinely
            # tied assignments (e.g. distances [1, ε, ε] vs [ε, ε, 1]) would
            # otherwise compare unequal and the tie would go undetected.
            cost = math.fsum(chosen_distances)
            margin = _injective_tie_margin(cost, best_cost)
            if best_cost is None or cost < best_cost - margin:
                best_cost = cost
                best_assignment = {order[level]: chosen_targets[level] for level in range(order_count)}
                tie = False
            elif abs(cost - best_cost) <= margin:
                tie = True
            depth -= 1
            continue
        if chosen_targets[depth] >= 0:
            used.discard(chosen_targets[depth])
            chosen_targets[depth] = -1
        candidates = candidate_lists[order[depth]]
        advanced = False
        while candidate_positions[depth] < len(candidates):
            distance, target = candidates[candidate_positions[depth]]
            candidate_positions[depth] += 1
            if target in used:
                continue
            cost = prefix_costs[depth] + distance
            # Candidates are distance-ascending, so once the accumulated cost
            # exceeds the best one (beyond the tie margin) it stays that way;
            # equal-within-margin cost continues so ties are still discovered.
            if best_cost is not None and cost > best_cost + _injective_tie_margin(cost, best_cost):
                break
            steps += 1
            if steps > step_limit:
                return None
            used.add(target)
            chosen_targets[depth] = target
            chosen_distances[depth] = distance
            prefix_costs[depth + 1] = cost
            if depth + 1 < order_count:
                candidate_positions[depth + 1] = 0
            depth += 1
            advanced = True
            break
        if not advanced:
            candidate_positions[depth] = 0
            depth -= 1

    if tie or best_assignment is None:
        return None
    return best_assignment


def _injective_tie_margin(cost: float, best_cost: float | None) -> float:
    """Float-noise margin for cost comparison, scaled to the magnitudes."""

    reference = max(abs(cost), abs(best_cost) if best_cost is not None else 0.0, 1.0e-12)
    return reference * 1.0e-9


def _one_sided_candidate_arrays(coords64: numpy.ndarray, axis_index: int, tolerance: float):
    """Build sorted mirror candidate arrays with one off-plane probe."""

    count = len(coords64)
    if count == 0 or not numpy.isfinite(coords64).all():
        return None
    axis = coords64[:, axis_index]
    on_plane = numpy.abs(axis) <= tolerance
    positive = ~on_plane & (axis > tolerance)
    negative = ~on_plane & (axis < -tolerance)
    inverse = 1.0 / max(tolerance, 1.0e-12)
    scaled = coords64 * inverse
    if numpy.any(numpy.abs(scaled) >= 2**62):
        return None
    bins = numpy.floor(scaled).astype(numpy.int64)
    mins = bins.min(axis=0)
    spans = bins.max(axis=0) - mins + 1
    if int(spans[0]) * int(spans[1]) * int(spans[2]) >= 2**63:
        return None
    stride_y = int(spans[2])
    stride_x = int(spans[1]) * stride_y
    shifted = bins - mins
    keys = shifted[:, 0] * stride_x + shifted[:, 1] * stride_y + shifted[:, 2]
    order = numpy.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    offsets = numpy.asarray([(x, y) for x in (-1, 0, 1) for y in (-1, 0, 1)], dtype=numpy.int64)
    span_z = int(spans[2])
    span_xy = numpy.asarray((int(spans[0]), int(spans[1])), dtype=numpy.int64)

    def probe(query_indices, query_points, side_mask):
        if len(query_indices) == 0:
            empty_i = numpy.empty(0, dtype=numpy.int64)
            return empty_i, empty_i.copy(), numpy.empty(0, dtype=numpy.float64)
        query_bins = numpy.floor(query_points * inverse).astype(numpy.int64)
        shifted_xy = (query_bins[:, None, :2] + offsets[None, :, :]).reshape(-1, 2) - mins[:2]
        shifted_z = numpy.repeat(query_bins[:, 2] - mins[2], len(offsets))
        valid = numpy.flatnonzero(
            (shifted_xy >= 0).all(axis=1)
            & (shifted_xy < span_xy).all(axis=1)
            & (shifted_z >= -1)
            & (shifted_z <= span_z)
        )
        if len(valid) == 0:
            empty_i = numpy.empty(0, dtype=numpy.int64)
            return empty_i, empty_i.copy(), numpy.empty(0, dtype=numpy.float64)
        low = numpy.clip(shifted_z[valid] - 1, 0, span_z - 1)
        high = numpy.clip(shifted_z[valid] + 1, 0, span_z - 1)
        row = shifted_xy[valid][:, 0] * stride_x + shifted_xy[valid][:, 1] * stride_y
        left = numpy.searchsorted(sorted_keys, row + low, side="left")
        right = numpy.searchsorted(sorted_keys, row + high, side="right")
        counts = right - left
        total = int(counts.sum())
        if total == 0:
            empty_i = numpy.empty(0, dtype=numpy.int64)
            return empty_i, empty_i.copy(), numpy.empty(0, dtype=numpy.float64)
        windows = numpy.repeat(numpy.arange(len(counts), dtype=numpy.int64), counts)
        starts = numpy.repeat(numpy.cumsum(counts) - counts, counts)
        within = numpy.arange(total, dtype=numpy.int64) - starts
        targets = order[left[windows] + within]
        local_queries = valid[windows] // len(offsets)
        queries = query_indices[local_queries]
        distances = numpy.max(numpy.abs(coords64[targets] - query_points[local_queries]), axis=1)
        keep = (distances <= tolerance) & side_mask[targets]
        return queries[keep], targets[keep], distances[keep]

    positive_indices = numpy.flatnonzero(positive)
    mirrored_positive = coords64[positive].copy()
    mirrored_positive[:, axis_index] = -mirrored_positive[:, axis_index]
    positive_queries, positive_targets, positive_distances = probe(positive_indices, mirrored_positive, negative)
    plane_indices = numpy.flatnonzero(on_plane)
    plane_queries, plane_targets, plane_distances = probe(plane_indices, coords64[on_plane], on_plane)
    queries = numpy.concatenate((positive_queries, positive_targets, plane_queries))
    targets = numpy.concatenate((positive_targets, positive_queries, plane_targets))
    distances = numpy.concatenate((positive_distances, positive_distances, plane_distances))
    if len(queries):
        order_all = numpy.lexsort((targets, distances, queries))
        queries = queries[order_all]
        targets = targets[order_all]
        distances = distances[order_all]
    return queries, targets, distances


def _one_sided_pair_table(coords64: numpy.ndarray, axis_index: int, tolerance: float):
    """Build mirror candidates with one off-plane probe and its transpose."""

    arrays = _one_sided_candidate_arrays(coords64, axis_index, tolerance)
    if arrays is None:
        return None
    count = len(coords64)
    queries, targets, distances = arrays
    query_counts = numpy.bincount(queries, minlength=count)
    target_counts = numpy.bincount(targets, minlength=count)
    starts = numpy.cumsum(query_counts) - query_counts
    assigned = numpy.full(count, -1, dtype=numpy.int64)
    singles = numpy.flatnonzero(query_counts == 1)
    if len(singles):
        single_targets = targets[starts[singles]]
        unique = target_counts[single_targets] == 1
        assigned[singles[unique]] = single_targets[unique]
    remainder_mask = query_counts > 0
    remainder_mask[assigned >= 0] = False
    candidate_lists: dict[int, list[tuple[float, int]]] = {}
    for query in numpy.flatnonzero(remainder_mask).tolist():
        begin = int(starts[query])
        end = begin + int(query_counts[query])
        candidate_lists[query] = [
            (float(distance), int(target))
            for distance, target in zip(distances[begin:end], targets[begin:end], strict=True)
        ]
    if candidate_lists:
        queries_by_target: dict[int, list[int]] = defaultdict(list)
        for query, candidates in candidate_lists.items():
            for _distance, target in candidates:
                queries_by_target[target].append(query)
        visited: set[int] = set()
        for start_query in candidate_lists:
            if start_query in visited or not candidate_lists[start_query]:
                visited.add(start_query)
                continue
            component = []
            component_targets: set[int] = set()
            pending = [start_query]
            visited.add(start_query)
            while pending:
                query = pending.pop()
                component.append(query)
                for _distance, target in candidate_lists[query]:
                    if target in component_targets:
                        continue
                    component_targets.add(target)
                    for other in queries_by_target[target]:
                        if other not in visited:
                            visited.add(other)
                            pending.append(other)
            assignment = _solve_injective_component(component, candidate_lists)
            if assignment is not None:
                for query, target in assignment.items():
                    assigned[query] = target
    pairs = {
        source: int(target)
        for source, target in enumerate(assigned.tolist())
        if target >= 0 and int(assigned[target]) == source
    }
    return pairs


def build_vertex_mirror_lookup(
    coords: Sequence[Vector],
    axis_index: int,
    tolerance: float,
) -> VertexMirrorLookup:
    """Build a :class:`VertexMirrorLookup` over *coords*.

    Coordinates are stored in registration order and indexed by a KDTree.
    Chebyshev verification remains against the stored coordinate tuples.
    """

    stored_coords = tuple((float(co[0]), float(co[1]), float(co[2])) for co in coords)
    return VertexMirrorLookup(axis_index=axis_index, tolerance=tolerance, coords=stored_coords)


def _vertex_pair_table_from_lookup(
    lookup: VertexMirrorLookup,
    coords: Sequence[Vector],
) -> dict[int, int]:
    assigned = lookup.find_all_mirrored(coords)
    pairs: dict[int, int] = {}
    for source, target in enumerate(assigned):
        if target is None:
            continue
        if target == source or assigned[target] == source:
            pairs[source] = target
    return pairs


def build_vertex_pair_table(
    coords: Sequence[Vector],
    axis_index: int,
    tolerance: float,
) -> dict[int, int]:
    """Involutive vertex pair table over *coords*: ``pairs[pairs[v]] == v``.

    Built from the injective batch assignment.  Injectivity alone does not
    imply ``mirror(mirror(v)) == v``, so one-way assignments (v→w without
    w→v) are discarded and both vertices stay unpaired.  On-plane vertices
    pair with themselves.
    """

    return _vertex_pair_table_from_lookup(build_vertex_mirror_lookup(coords, axis_index, tolerance), coords)


def classify_selection_overlap(
    coords: Sequence[Vector],
    selected_indices: Iterable[int],
    *,
    axis_index: int,
    tolerance: float,
) -> OverlapClassification:
    """Classify how a selection relates to its own mirror image.

    ``selection_crosses_mirror`` used to answer only "would replaying the
    mirror double-apply?" (ρ(S) ∩ S ≠ ∅).  This keeps that boundary — a
    both-sides selection whose mirror image does not intersect the selection
    is still DISJOINT and replays correctly — and refines the intersecting
    case into SELF_MIRRORED versus PARTIAL.
    """

    pairs = build_vertex_pair_table(coords, axis_index, tolerance)
    selected = frozenset(selected_indices)
    shared = frozenset(index for index in selected if abs(coords[index][axis_index]) <= tolerance)
    off = selected - shared
    complete = all(index in pairs for index in off)
    mirrors = {pairs[index] for index in off if index in pairs}
    crossing = mirrors & selected
    if not crossing:
        # An all-on-plane selection is trivially its own mirror image; ρ has
        # degenerated to the identity, and one native run is already symmetric.
        overlap = MirrorOverlap.SELF_MIRRORED if not off else MirrorOverlap.DISJOINT
    elif complete and mirrors == off:
        overlap = MirrorOverlap.SELF_MIRRORED
    else:
        overlap = MirrorOverlap.PARTIAL
    return OverlapClassification(overlap=overlap, complete=complete, pairs=pairs)


def _coords_match_chebyshev(
    first: Sequence[tuple[float, float, float]],
    second: Sequence[tuple[float, float, float]],
    tolerance: float,
) -> bool:
    """Return whether the two coordinate multisets pair within *tolerance* per axis.

    Uses iterative Kuhn bipartite matching (explicit BFS augmenting paths) so
    large n-gons cannot hit RecursionError (contract R2-2).
    """

    count = len(first)
    if count != len(second):
        return False
    if count == 0:
        return True

    adjacency: list[list[int]] = [[] for _ in range(count)]
    for left_index, left_coord in enumerate(first):
        for right_index, right_coord in enumerate(second):
            if coordinates_match(left_coord, right_coord, tolerance):
                adjacency[left_index].append(right_index)

    match_left = [-1] * count
    match_right = [-1] * count

    for start_left in range(count):
        if match_left[start_left] >= 0:
            continue
        parent_right = [-1] * count
        seen_right = [False] * count
        queue = [start_left]
        queue_head = 0
        free_right = -1
        while queue_head < len(queue) and free_right < 0:
            left_index = queue[queue_head]
            queue_head += 1
            for right_index in adjacency[left_index]:
                if seen_right[right_index]:
                    continue
                seen_right[right_index] = True
                parent_right[right_index] = left_index
                matched_left = match_right[right_index]
                if matched_left < 0:
                    free_right = right_index
                    break
                queue.append(matched_left)
        if free_right < 0:
            return False
        right_index = free_right
        while right_index >= 0:
            left_index = parent_right[right_index]
            previous_right = match_left[left_index]
            match_left[left_index] = right_index
            match_right[right_index] = left_index
            right_index = previous_right

    return True
