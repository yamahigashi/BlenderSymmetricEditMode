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
