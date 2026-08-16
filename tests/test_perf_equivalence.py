# SPDX-License-Identifier: GPL-3.0-or-later

"""Headless equivalence checks for the large-mesh preparation changes."""

# Wrapper fixtures intentionally capture each fresh BMesh and spy state.
# ruff: noqa: B010, B023

from __future__ import annotations

import copy
import math
import random
import sys
import traceback
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import bmesh
import bpy
import numpy
from mathutils import Vector
from mathutils.kdtree import KDTree

PACKAGE_PARENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_PARENT))
sys.path.insert(0, str(PACKAGE_PARENT / "tests"))

from perf_fixtures import DENSE_CANDIDATE_COORDS  # noqa: E402

from ydd_symmetric_edit import (  # noqa: E402
    _types,
    face_mapping,
    layer_names,
    matching,
    operators,
    selection,
    stitch_common,
    stitch_crossings,
    stitch_pathedges,
    stitch_reflect,
)
from ydd_symmetric_edit import replay as replay_module  # noqa: E402
from ydd_symmetric_edit import rip as rip_module  # noqa: E402
from ydd_symmetric_edit import session as session_module  # noqa: E402
from ydd_symmetric_edit import snapshot as snapshot_module  # noqa: E402
from ydd_symmetric_edit import watcher as watcher_module  # noqa: E402
from ydd_symmetric_edit._types import (  # noqa: E402
    Coordinate3D,
    FaceId,
    KnifeSession,
    MeshSelectionMode,
    MirrorOverlap,
    OverlapClassification,
)

TOLERANCE = 1.0e-5
SEEDS = (3, 17, 41)


def _reference_tie_margin(cost, best_cost):
    reference = max(abs(cost), abs(best_cost) if best_cost is not None else 0.0, 1.0e-12)
    return reference * 1.0e-9


def _reference_solve_injective_component(queries, candidate_lists, step_limit=2000):
    order = sorted(queries, key=lambda query: (len(candidate_lists[query]), query))
    if len(order) == 0 or len(order) > step_limit:
        return {} if not order else None
    best_cost = None
    best_assignment = None
    tie = False
    steps = 0
    used = set()
    chosen_targets = [-1] * len(order)
    chosen_distances = [0.0] * len(order)
    candidate_positions = [0] * len(order)
    prefix_costs = [0.0] * (len(order) + 1)
    depth = 0
    while depth >= 0:
        if depth == len(order):
            cost = math.fsum(chosen_distances)
            margin = _reference_tie_margin(cost, best_cost)
            if best_cost is None or cost < best_cost - margin:
                best_cost = cost
                best_assignment = {order[level]: chosen_targets[level] for level in range(len(order))}
                tie = False
            elif abs(cost - best_cost) <= margin:
                tie = True
            depth -= 1
            continue
        if chosen_targets[depth] >= 0:
            used.discard(chosen_targets[depth])
            chosen_targets[depth] = -1
        advanced = False
        candidates = candidate_lists[order[depth]]
        while candidate_positions[depth] < len(candidates):
            distance, target = candidates[candidate_positions[depth]]
            candidate_positions[depth] += 1
            if target in used:
                continue
            cost = prefix_costs[depth] + distance
            if best_cost is not None and cost > best_cost + _reference_tie_margin(cost, best_cost):
                break
            steps += 1
            if steps > step_limit:
                return None
            used.add(target)
            chosen_targets[depth] = target
            chosen_distances[depth] = distance
            prefix_costs[depth + 1] = cost
            if depth + 1 < len(order):
                candidate_positions[depth + 1] = 0
            depth += 1
            advanced = True
            break
        if not advanced:
            candidate_positions[depth] = 0
            depth -= 1
    return None if tie or best_assignment is None else best_assignment


class _ReferenceBinLookup:
    """The pre-KDTree 27-bin lookup retained as a test oracle."""

    def __init__(self, coords, axis_index: int, tolerance: float):
        self._axis_index = axis_index
        self._tolerance = tolerance
        self._coords = tuple((float(co[0]), float(co[1]), float(co[2])) for co in coords)
        self._bins = defaultdict(list)
        for index, coordinate in enumerate(self._coords):
            self._bins[matching._quantized_coordinate(Vector(coordinate), tolerance)].append((index, coordinate))
        self._on_plane_indices = None

    def is_on_plane(self, co) -> bool:
        return abs(float(co[self._axis_index])) <= self._tolerance

    def _on_plane_registered(self):
        if self._on_plane_indices is None:
            self._on_plane_indices = frozenset(
                index
                for entries in self._bins.values()
                for index, coordinate in entries
                if abs(coordinate[self._axis_index]) <= self._tolerance
            )
        return self._on_plane_indices

    def _candidates_for(self, position):
        position = (float(position[0]), float(position[1]), float(position[2]))
        candidates = []
        for bin_key in matching._iter_quantized_neighborhood(Vector(position), self._tolerance):
            for index, stored in self._bins.get(bin_key, ()):
                distance = matching._chebyshev_distance_3d(position, stored)
                if distance <= self._tolerance:
                    candidates.append((distance, index))
        candidates.sort()
        return candidates

    def find(self, co):
        expected = matching.mirror_coordinate(co, self._axis_index)
        expected = (float(expected[0]), float(expected[1]), float(expected[2]))
        best = None
        for bin_key in matching._iter_quantized_neighborhood(Vector(expected), self._tolerance):
            for index, stored in self._bins.get(bin_key, ()):
                distance = matching._chebyshev_distance_3d(expected, stored)
                if distance <= self._tolerance and (best is None or distance < best[0]):
                    best = (distance, index)
        return None if best is None else best[1]

    def _resolve_injective(self, positions, plane_side=None):
        if plane_side is None:
            candidate_lists = [self._candidates_for(position) for position in positions]
        else:
            on_plane = self._on_plane_registered()
            candidate_lists = [
                [
                    (distance, index)
                    for distance, index in self._candidates_for(position)
                    if (index in on_plane) == plane_side
                ]
                for position in positions
            ]
        by_target = defaultdict(list)
        for query, candidates in enumerate(candidate_lists):
            for _distance, target in candidates:
                by_target[target].append(query)
        results = [None] * len(positions)
        visited = [False] * len(positions)
        for start in range(len(positions)):
            if visited[start]:
                continue
            visited[start] = True
            if not candidate_lists[start]:
                continue
            component = []
            targets = set()
            pending = [start]
            while pending:
                query = pending.pop()
                component.append(query)
                for _distance, target in candidate_lists[query]:
                    if target in targets:
                        continue
                    targets.add(target)
                    for other in by_target[target]:
                        if not visited[other]:
                            visited[other] = True
                            pending.append(other)
            assignment = _reference_solve_injective_component(component, candidate_lists)
            if assignment is not None:
                for query, target in assignment.items():
                    results[query] = target
        return results

    def find_all_mirrored(self, coords):
        result = [None] * len(coords)
        on_plane = [(index, co) for index, co in enumerate(coords) if self.is_on_plane(co)]
        if on_plane:
            resolved = self._resolve_injective([co for _index, co in on_plane], plane_side=True)
            for (index, _co), target in zip(on_plane, resolved, strict=True):
                result[index] = target
        off_plane = [
            (index, matching.mirror_coordinate(co, self._axis_index))
            for index, co in enumerate(coords)
            if not self.is_on_plane(co)
        ]
        if off_plane:
            resolved = self._resolve_injective([position for _index, position in off_plane], plane_side=False)
            for (index, _position), target in zip(off_plane, resolved, strict=True):
                result[index] = target
        return tuple(result)

    def find_all_direct(self, coords):
        return tuple(self._resolve_injective(list(coords)))


def _axis_coordinate(values, axis_index: int):
    return Vector(values)


def _vector_tuple(vector) -> tuple[float, float, float]:
    return float(vector[0]), float(vector[1]), float(vector[2])


def _make_point(axis_index: int, axis_value: float, transverse, offset: bool, offset_magnitude: float = 10000.0):
    values = [0.0, 0.0, 0.0]
    values[axis_index] = axis_value
    transverse_values = list(transverse)
    for coordinate in range(3):
        if coordinate != axis_index:
            values[coordinate] = transverse_values.pop(0)
    if offset:
        if axis_value < 0.0:
            values[axis_index] -= offset_magnitude
        elif axis_value > 0.0:
            values[axis_index] += offset_magnitude
        for coordinate in range(3):
            if coordinate != axis_index:
                values[coordinate] += offset_magnitude
    return _axis_coordinate(values, axis_index)


def _make_lookup_case(
    axis_index: int,
    seed: int,
    offset: bool = False,
    tolerance: float = TOLERANCE,
    offset_magnitude: float = 10000.0,
):
    rng = random.Random(seed)
    coords = []
    for _index in range(9):
        magnitude = 0.2 + 3.0 * rng.random()
        transverse = (rng.uniform(-2.0, 2.0), rng.uniform(-2.0, 2.0))
        source = _make_point(axis_index, -magnitude, transverse, offset, offset_magnitude)
        counterpart = source.copy()
        counterpart[axis_index] = -counterpart[axis_index]
        for coordinate in range(3):
            counterpart[coordinate] += rng.uniform(-0.22, 0.22) * tolerance
        coords.extend((source, counterpart))

    boundary_source = _make_point(axis_index, -(1.0 + 0.2 * tolerance), (0.3, -0.8), offset, offset_magnitude)
    boundary_target = boundary_source.copy()
    boundary_target[axis_index] = -boundary_target[axis_index] + 0.4 * tolerance
    coords.extend((boundary_source, boundary_target))

    sphere_source = _make_point(axis_index, -4.0, (1.3, -1.1), offset, offset_magnitude)
    sphere_target = sphere_source.copy()
    sphere_target[axis_index] = -sphere_target[axis_index] + 0.95 * tolerance
    for coordinate in range(3):
        if coordinate != axis_index:
            sphere_target[coordinate] += 0.95 * tolerance
    coords.extend((sphere_source, sphere_target))
    sphere_outer = sphere_source.copy()
    sphere_outer[axis_index] = -sphere_outer[axis_index] + 1.05 * tolerance
    for coordinate in range(3):
        if coordinate != axis_index:
            sphere_outer[coordinate] += 1.05 * tolerance
    coords.extend((sphere_source.copy(), sphere_outer))

    plane = _make_point(axis_index, 0.35 * tolerance, (0.2, -0.4), offset, offset_magnitude)
    plane[axis_index] = 0.0
    if not offset:
        coords.append(plane)
    missing = _make_point(axis_index, 8.0, (0.1, 0.2), offset, offset_magnitude)
    coords.append(missing)

    duplicate_source = _make_point(axis_index, -6.0, (-0.2, 0.4), offset, offset_magnitude)
    duplicate_target = duplicate_source.copy()
    duplicate_target[axis_index] = -duplicate_target[axis_index]
    duplicate_target_2 = duplicate_target.copy()
    duplicate_target_2[axis_index] += 0.4 * tolerance
    coords.extend((duplicate_source, duplicate_target, duplicate_target_2))
    return coords


def _assert_lookup_case(coords, axis_index: int, tolerance: float = TOLERANCE):
    reference = _ReferenceBinLookup(coords, axis_index, tolerance)
    lookup = matching.build_vertex_mirror_lookup(coords, axis_index, tolerance)
    assert not hasattr(lookup, "_bins")
    assert tuple(lookup._coords) == reference._coords
    assert matching.build_vertex_pair_table(coords, axis_index, tolerance) == _pair_table(reference, coords)
    assert lookup.find_all_mirrored(coords) == tuple(reference.find_all_mirrored(coords))
    assert lookup.find_all_direct(coords) == tuple(reference.find_all_direct(coords))
    assert lookup._batch_path_count >= 2
    for query in coords:
        candidates = reference._candidates_for(matching.mirror_coordinate(query, axis_index))
        if not candidates or len([distance for distance, _index in candidates if distance == candidates[0][0]]) > 1:
            continue
        assert lookup.find(query) == reference.find(query)


def _assert_bin_and_sphere_fixture(axis_index: int):
    source = _make_point(axis_index, -1.0, (0.25, -0.5), False)
    expected = matching.mirror_coordinate(source, axis_index)
    inside = expected.copy()
    outside = expected.copy()
    # expected (=1.0) sits at the very top of its floor bin, so the straddling
    # candidates must sit above it to land in a different bin while staying
    # inside tol. Factors leave room for float32 rounding (~1.2% of tol here).
    inside[axis_index] += 0.99 * TOLERANCE
    outside[axis_index] += 1.03 * TOLERANCE
    expected_key = matching._quantized_coordinate(expected, TOLERANCE)
    inside_key = matching._quantized_coordinate(inside, TOLERANCE)
    outside_key = matching._quantized_coordinate(outside, TOLERANCE)
    assert expected_key != inside_key and expected_key != outside_key
    assert (
        0.98 * TOLERANCE <= matching._chebyshev_distance_3d(_vector_tuple(expected), _vector_tuple(inside)) <= TOLERANCE
    )
    assert matching._chebyshev_distance_3d(_vector_tuple(expected), _vector_tuple(outside)) > TOLERANCE
    reference = _ReferenceBinLookup((inside, outside), axis_index, TOLERANCE)
    lookup = matching.build_vertex_mirror_lookup((inside, outside), axis_index, TOLERANCE)
    assert reference.find(source) == lookup.find(source) == 0
    assert reference.find_all_mirrored((source,)) == lookup.find_all_mirrored((source,)) == (0,)

    sphere_source = _make_point(axis_index, -4.0, (1.3, -1.1), False)
    sphere_expected = matching.mirror_coordinate(sphere_source, axis_index)
    sphere_inside = sphere_expected.copy()
    sphere_outside = sphere_expected.copy()
    for coordinate in range(3):
        sphere_inside[coordinate] -= 0.95 * TOLERANCE
        sphere_outside[coordinate] -= 1.05 * TOLERANCE
    inside_distance = matching._chebyshev_distance_3d(_vector_tuple(sphere_expected), _vector_tuple(sphere_inside))
    outside_distance = matching._chebyshev_distance_3d(_vector_tuple(sphere_expected), _vector_tuple(sphere_outside))
    assert 0.9 * TOLERANCE <= inside_distance <= TOLERANCE
    assert TOLERANCE < outside_distance <= 1.1 * TOLERANCE
    reference = _ReferenceBinLookup((sphere_inside, sphere_outside), axis_index, TOLERANCE)
    lookup = matching.build_vertex_mirror_lookup((sphere_inside, sphere_outside), axis_index, TOLERANCE)
    assert reference.find(sphere_source) == lookup.find(sphere_source) == 0
    assert reference.find_all_mirrored((sphere_source,)) == lookup.find_all_mirrored((sphere_source,)) == (0,)


def _pair_table(reference, coords):
    assigned = reference.find_all_mirrored(coords)
    pairs = {}
    for source, target in enumerate(assigned):
        if target is not None and (target == source or assigned[target] == source):
            pairs[source] = target
    return pairs


def _reference_coordinates_match(first, second, tolerance):
    if len(first) != len(second):
        return False
    adjacency = [
        [index for index, right in enumerate(second) if matching.coordinates_match(left, right, tolerance)]
        for left in first
    ]
    match_left = [-1] * len(first)
    match_right = [-1] * len(second)
    for start in range(len(first)):
        parent = [-1] * len(second)
        seen = [False] * len(second)
        queue = [start]
        free = -1
        while queue and free < 0:
            left = queue.pop(0)
            for right in adjacency[left]:
                if seen[right]:
                    continue
                seen[right] = True
                parent[right] = left
                matched = match_right[right]
                if matched < 0:
                    free = right
                    break
                queue.append(matched)
        if free < 0:
            return False
        right = free
        while right >= 0:
            left = parent[right]
            previous = match_left[left]
            match_left[left] = right
            match_right[right] = left
            right = previous
    return True


def _check_vertex_lookup_equivalence():
    for axis_index in (matching.AXIS_INDEX["X"], matching.AXIS_INDEX["Y"], matching.AXIS_INDEX["Z"]):
        _assert_bin_and_sphere_fixture(axis_index)
        for seed in SEEDS:
            _assert_lookup_case(_make_lookup_case(axis_index, seed), axis_index)
        # tol=1e-5 needs a moderate offset: float32 ULP at 1e4 (9.77e-4) would
        # collapse every sub-tol jitter to zero; at 10 the ULP is ~0.095*tol so
        # the boundary/sphere sub-cases keep their intended distances.
        small_radius_offset = _make_lookup_case(axis_index, SEEDS[0], offset=True, offset_magnitude=10.0)
        assert all(abs(float(component)) > 7.0 for coordinate in small_radius_offset for component in coordinate)
        _assert_lookup_case(small_radius_offset, axis_index)
        offset_coords = _make_lookup_case(axis_index, SEEDS[0], offset=True, tolerance=1.0e-2)
        assert all(abs(float(component)) > 9990.0 for coordinate in offset_coords for component in coordinate)
        _assert_lookup_case(offset_coords, axis_index, tolerance=1.0e-2)

    _assert_a_reference_case([], 0, TOLERANCE)
    empty_lookup = matching.build_vertex_mirror_lookup([], 0, TOLERANCE)
    empty_query = Vector((1.0, 2.0, 3.0))
    assert empty_lookup.find(empty_query) is None
    assert empty_lookup.find_all_direct((empty_query,)) == (None,)
    assert empty_lookup.find_all_mirrored((empty_query,)) == (None,)

    plane_coords = [Vector((0.0, float(index), -float(index))) for index in range(5)]
    _assert_a_reference_case(plane_coords, 0, TOLERANCE)
    lookup = matching.build_vertex_mirror_lookup(plane_coords, 0, TOLERANCE)
    assert lookup.find_all_mirrored(plane_coords) == tuple(range(5))

    batch_positions = tuple(plane_coords)
    batch_lookup = matching.build_vertex_mirror_lookup(batch_positions, 0, TOLERANCE)
    batch_lists = batch_lookup._batch_candidates(batch_positions)
    assert batch_lists is not None
    assert batch_lists == [batch_lookup._candidates_for(position) for position in batch_positions]
    assert batch_lookup._batch_path_count == 1

    tie_lookup = matching.build_vertex_mirror_lookup(
        [Vector((0.5 * TOLERANCE, 0.0, 0.0)), Vector((-0.5 * TOLERANCE, 0.0, 0.0))],
        0,
        TOLERANCE,
    )
    assert tie_lookup.find(Vector((0.0, 0.0, 0.0))) == 0

    # Query 0 (-1) forms its own component over targets {1, 2} and resolves to
    # the nearer index 1; queries 1 and 2 contest target 0 and are both
    # rejected as an incomplete cover.
    ambiguous = [Vector((-1.0, 0.0, 0.0)), Vector((1.0, 0.0, 0.0)), Vector((1.0 + 0.4 * TOLERANCE, 0.0, 0.0))]
    old_result, new_result = _assert_a_reference_case(ambiguous, 0, TOLERANCE)
    assert old_result == new_result == (1, None, None)


def _check_batch_candidate_contract_edges():
    tolerance = TOLERANCE

    def assert_candidates(lookup, positions, plane_side=None):
        actual = lookup._batch_candidates(positions, plane_side)
        assert actual is not None
        expected = [lookup._candidates_for(position) for position in positions]
        if plane_side is not None:
            on_plane = lookup._on_plane_registered()
            expected = [
                [(distance, index) for distance, index in candidates if (index in on_plane) == plane_side]
                for candidates in expected
            ]
        assert actual == expected

    boundary_coords = tuple(
        Vector(co)
        for co in (
            (1.0, 0.0, 0.0),
            (1.0 + 0.4 * tolerance, 0.0, 0.0),
            (1.0 - 0.4 * tolerance, 0.0, 0.0),
            (-1.0, -0.25, 0.5),
            (-1.0 + 0.4 * tolerance, -0.25, 0.5),
            (-1.0 - 0.4 * tolerance, -0.25, 0.5),
        )
    )
    boundary_lookup = matching.build_vertex_mirror_lookup(boundary_coords, 0, tolerance)
    assert_candidates(boundary_lookup, boundary_coords)
    boundary_candidates = boundary_lookup._batch_candidates((Vector((1.0, 0.0, 0.0)),))
    assert boundary_candidates is not None
    assert boundary_candidates[0] == sorted(boundary_lookup._candidates_for(Vector((1.0, 0.0, 0.0))))

    # span_z == 1 (planar registered set): the clamped z range degenerates to
    # one bin, and queries whose z bin sits one step outside the registered
    # box on either side must still resolve through the neighborhood.
    planar_coords = tuple(
        Vector(co)
        for co in (
            (-1.0, 0.5, 0.0),
            (1.0, 0.5, 0.0),
            (0.25, -0.75, 0.0),
            (-0.25, -0.75, 0.0),
        )
    )
    planar_lookup = matching.build_vertex_mirror_lookup(planar_coords, 0, tolerance)
    planar_index = planar_lookup._registered_batch_index()
    assert planar_index is not False and int(planar_index["spans"][2]) == 1
    planar_queries = tuple(
        Vector((float(co[0]), float(co[1]), float(co[2]) + offset))
        for co in planar_coords
        for offset in (-0.9 * tolerance, 0.0, 0.9 * tolerance, -1.9 * tolerance, 1.9 * tolerance)
    )
    assert_candidates(planar_lookup, planar_queries)
    assert_candidates(planar_lookup, planar_queries, plane_side=False)
    reference_planar = _ReferenceBinLookup(planar_coords, 0, tolerance)
    assert planar_lookup.find_all_direct(planar_queries) == tuple(reference_planar.find_all_direct(planar_queries))

    # Box-corner clamps: registered box spans 2x2x1 bins at tolerance=1;
    # queries sit diagonally outside every axis at once.
    corner_coords = (Vector((0.0, 0.0, 0.0)), Vector((1.0, 1.0, 0.0)))
    corner_lookup = matching.build_vertex_mirror_lookup(corner_coords, 0, 1.0)
    corner_queries = (Vector((-1.0, -1.0, -1.0)), Vector((2.0, 2.0, 1.0)))
    actual_corner = corner_lookup._batch_candidates(corner_queries)
    assert actual_corner is not None
    reference_corner = _ReferenceBinLookup(corner_coords, 0, 1.0)
    assert actual_corner == [reference_corner._candidates_for(position) for position in corner_queries]

    # Span-product overflow: |scaled| stays under 2**62 per axis, yet the
    # packed span product overflows int64 and must fall back to the KDTree
    # path with identical public results.
    huge_span_coords = (Vector((0.0, 0.0, 0.0)), Vector((float(2**21), float(2**21), float(2**21))))
    huge_span_lookup = matching.build_vertex_mirror_lookup(huge_span_coords, 0, 1.0)
    assert huge_span_lookup._registered_batch_index() is False
    assert huge_span_lookup._batch_candidates(huge_span_coords) is None
    reference_huge = _ReferenceBinLookup(huge_span_coords, 0, 1.0)
    assert huge_span_lookup.find_all_direct(huge_span_coords) == tuple(reference_huge.find_all_direct(huge_span_coords))
    assert huge_span_lookup.find_all_mirrored(huge_span_coords) == tuple(
        reference_huge.find_all_mirrored(huge_span_coords)
    )

    # Non-finite queries abandon the batch path; the public API answers via
    # the per-query KDTree route without raising.
    finite_lookup = matching.build_vertex_mirror_lookup((Vector((1.0, 0.0, 0.0)),), 0, tolerance)
    nan_query = (Vector((float("nan"), 0.0, 0.0)),)
    assert finite_lookup._batch_candidates(nan_query) is None
    assert finite_lookup.find_all_direct(nan_query) == (None,)

    # Caller-owned float64 arrays must come back untouched (the mirror
    # negation operates on a fancy-indexed copy).
    owned = numpy.asarray([(0.5, 0.25, 0.0), (-0.5, 0.25, 0.0)], dtype=numpy.float64)
    owned_snapshot = owned.copy()
    owned_lookup = matching.build_vertex_mirror_lookup(
        (Vector((0.5, 0.25, 0.0)), Vector((-0.5, 0.25, 0.0))), 0, tolerance
    )
    owned_lookup.find_all_mirrored(owned)
    owned_lookup.find_all_direct(owned)
    assert (owned == owned_snapshot).all()

    clamp_coords = (Vector((0.0, 0.0, 0.0)), Vector((1.0e-13, 0.0, 0.0)))
    clamp_lookup = matching.build_vertex_mirror_lookup(clamp_coords, 0, 0.0)
    assert_candidates(clamp_lookup, clamp_coords)

    empty_registered = matching.build_vertex_mirror_lookup([], 0, tolerance)
    assert empty_registered._batch_candidates((Vector((float("nan"), 0.0, 0.0)),)) is None
    assert empty_registered._batch_candidates((Vector((float("inf"), 0.0, 0.0)),)) is None
    invalid_tree = KDTree(1)
    invalid_tree.insert(Vector((0.0, 0.0, 0.0)), 0)
    invalid_tree.balance()
    invalid_registered = matching.VertexMirrorLookup(
        axis_index=0,
        tolerance=tolerance,
        coords=((float("nan"), 0.0, 0.0),),
        tree=invalid_tree,
    )
    assert invalid_registered._batch_candidates(()) is None
    huge_registered = matching.VertexMirrorLookup(
        axis_index=0,
        tolerance=tolerance,
        coords=((2**63 * tolerance, 0.0, 0.0),),
        tree=invalid_tree,
    )
    assert huge_registered._batch_candidates(()) is None

    rounding_coords = (Vector((0.09375, 0.0, 0.0)), Vector((-0.09375, 0.0, 0.0)))
    rounding_lookup = matching.build_vertex_mirror_lookup(rounding_coords, 0, 1.0e-5)
    assert_candidates(rounding_lookup, rounding_coords)

    plane_coords = (Vector((0.0, 0.0, 0.0)), Vector((2.0 * tolerance, 0.0, 0.0)))
    plane_lookup = matching.build_vertex_mirror_lookup(plane_coords, 0, tolerance)
    plane_positions = (Vector((0.0, 0.0, 0.0)), Vector((2.0 * tolerance, 0.0, 0.0)))
    assert_candidates(plane_lookup, plane_positions, True)
    assert_candidates(plane_lookup, plane_positions, False)
    assert plane_lookup.find_all_mirrored((Vector((0.0, 0.0, 0.0)), Vector((-2.0 * tolerance, 0.0, 0.0)))) == (0, 1)
    near_plane_target = matching.build_vertex_mirror_lookup((Vector((0.5 * tolerance, 0.0, 0.0)),), 0, tolerance)
    near_plane_queries = (Vector((0.5 * tolerance, 0.0, 0.0)), Vector((-1.25 * tolerance, 0.0, 0.0)))
    near_plane_candidate_positions = (near_plane_queries[0], Vector((1.25 * tolerance, 0.0, 0.0)))
    near_plane_candidates = near_plane_target._batch_candidates(near_plane_candidate_positions)
    assert near_plane_candidates is not None
    assert near_plane_candidates[0][0][1] == 0
    assert math.isclose(near_plane_candidates[0][0][0], 0.0, abs_tol=1.0e-12)
    assert near_plane_candidates[1][0][1] == 0
    assert math.isclose(near_plane_candidates[1][0][0], 0.75 * tolerance, rel_tol=1.0e-6)
    near_plane_off_candidates = near_plane_target._batch_candidates(near_plane_candidate_positions, plane_side=False)
    assert near_plane_off_candidates is not None
    assert near_plane_off_candidates == [[], []]
    assert near_plane_target.find_all_mirrored(near_plane_queries) == (0, None)
    near_plane_reference = _ReferenceBinLookup((Vector((0.5 * tolerance, 0.0, 0.0)),), 0, tolerance)
    assert near_plane_target.find_all_mirrored(near_plane_queries) == tuple(
        near_plane_reference.find_all_mirrored(near_plane_queries)
    )

    fallback_lookup = matching.build_vertex_mirror_lookup((Vector((0.0, 0.0, 0.0)),), 0, tolerance)
    fallback_calls = 0
    original_candidates = fallback_lookup._candidates_for

    def count_fallback(position):
        nonlocal fallback_calls
        fallback_calls += 1
        return original_candidates(position)

    method_name = "_candidates_for"
    setattr(fallback_lookup, method_name, count_fallback)
    try:
        out_of_range = Vector((2**63 * tolerance, 0.0, 0.0))
        assert fallback_lookup._batch_candidates((out_of_range,)) is None
        assert fallback_lookup.find_all_direct((out_of_range,)) == (None,)
        assert fallback_calls == 1
    finally:
        setattr(fallback_lookup, method_name, original_candidates)

    contested = matching.build_vertex_mirror_lookup((Vector((1.0, 0.0, 0.0)),), 0, tolerance)
    assert contested._resolve_injective((Vector((1.0, 0.0, 0.0)), Vector((1.0 + 0.4 * tolerance, 0.0, 0.0)))) == [
        None,
        None,
    ]

    two_target_coords = (Vector((1.0, 0.0, 0.0)), Vector((1.0 + 0.8 * tolerance, 0.0, 0.0)))
    two_target_lookup = matching.build_vertex_mirror_lookup(two_target_coords, 0, tolerance)
    two_target_queries = (Vector((1.0 + 0.1 * tolerance, 0.0, 0.0)), Vector((1.0 + 0.2 * tolerance, 0.0, 0.0)))
    assert two_target_lookup._resolve_injective(two_target_queries) == [0, 1]

    rng = random.Random(113)
    large_coords = []
    for index in range(1100):
        magnitude = 0.01 * (index + 1)
        source = Vector((-magnitude, rng.uniform(-5.0, 5.0), rng.uniform(-5.0, 5.0)))
        if index % 17 == 0:
            large_coords.append(source)
            continue
        target = source.copy()
        target[0] = -target[0] + rng.uniform(-0.2, 0.2) * tolerance
        target[1] += rng.uniform(-0.2, 0.2) * tolerance
        target[2] += rng.uniform(-0.2, 0.2) * tolerance
        large_coords.extend((source, target))
    assert len(large_coords) >= 2000
    large_lookup = matching.build_vertex_mirror_lookup(large_coords, 0, tolerance)
    large_batch = large_lookup._batch_candidates(large_coords)
    assert large_batch is not None
    large_oracle = [large_lookup._candidates_for(position) for position in large_coords]
    assert large_batch == large_oracle
    reference = _ReferenceBinLookup(large_coords, 0, tolerance)
    assert large_lookup.find_all_mirrored(large_coords) == tuple(reference.find_all_mirrored(large_coords))


def _check_extend_selection_matrix():
    bm = bmesh.new()
    left = [bm.verts.new(co) for co in ((-1.0, 0.0, 0.0), (-1.0, 1.0, 0.0), (-1.0, 0.0, 1.0))]
    right = [bm.verts.new(co) for co in ((1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (1.0, 0.0, 1.0))]
    left_face = bm.faces.new(left)
    right_face = bm.faces.new(right)
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    left_edge = next(edge for edge in bm.edges if set(edge.verts) == {left[0], left[1]})
    right_edge = next(edge for edge in bm.edges if set(edge.verts) == {right[0], right[1]})

    def reset():
        for vertex in bm.verts:
            vertex.select = False
        for edge in bm.edges:
            edge.select = False
        for face in bm.faces:
            face.select = False

    try:
        reset()
        history_before = tuple(cast(Any, bm.select_history))
        assert selection.extend_selection_to_mirror(bm, 0, TOLERANCE) == 0
        assert not any(vertex.select for vertex in bm.verts)
        assert tuple(cast(Any, bm.select_history)) == history_before

        reset()
        left[0].select = True
        bm.select_history.add(left[0])
        history_before = tuple(cast(Any, bm.select_history))
        assert selection.extend_selection_to_mirror(bm, 0, TOLERANCE) == 1
        assert right[0].select
        assert tuple(cast(Any, bm.select_history)) == history_before

        # Selecting a BMesh edge/face flushes selection down to its verts
        # (and edges), so the mirrored additions count those too.
        reset()
        left_edge.select = True
        bm.select_history.add(left_edge)
        history_before = tuple(cast(Any, bm.select_history))
        assert selection.extend_selection_to_mirror(bm, 0, TOLERANCE) == 3
        assert right_edge.select
        assert tuple(cast(Any, bm.select_history)) == history_before

        reset()
        left_face.select = True
        bm.select_history.add(left_face)
        history_before = tuple(cast(Any, bm.select_history))
        assert selection.extend_selection_to_mirror(bm, 0, TOLERANCE) == 7
        assert right_face.select
        assert tuple(cast(Any, bm.select_history)) == history_before
    finally:
        bm.free()


def _check_extend_selection_lazy_indices():
    class CountingSequence:
        def __init__(self, values):
            self.values = list(values)
            self.iterations = 0

        def __iter__(self):
            self.iterations += 1
            return iter(self.values)

        def __len__(self):
            return len(self.values)

        def __getitem__(self, index):
            return self.values[index]

    class FakeVertex:
        def __init__(self, index, coordinate):
            self.index = index
            self.co = Vector(coordinate)
            self.select = False
            self.is_valid = True
            self.link_edges = []
            self.link_faces = []

    class FakeEdge:
        def __init__(self, index, vertices):
            self.index = index
            self.verts = vertices
            self.select = False
            self.is_valid = True
            for vertex in vertices:
                vertex.link_edges.append(self)

        def other_vert(self, vertex):
            return self.verts[1] if self.verts[0] is vertex else self.verts[0]

    class FakeFace:
        def __init__(self, index, vertices):
            self.index = index
            self.verts = vertices
            self.select = False
            self.is_valid = True
            for vertex in vertices:
                vertex.link_faces.append(self)

    for selection_mode in ("empty", "vertex", "edge", "face"):
        vertices = [
            FakeVertex(0, (-1.0, 0.0, 0.0)),
            FakeVertex(1, (1.0, 0.0, 0.0)),
            FakeVertex(2, (-1.0, 1.0, 0.0)),
            FakeVertex(3, (1.0, 1.0, 0.0)),
            FakeVertex(4, (-1.0, 0.0, 1.0)),
            FakeVertex(5, (1.0, 0.0, 1.0)),
        ]
        edges = [FakeEdge(0, (vertices[0], vertices[2])), FakeEdge(1, (vertices[1], vertices[3]))]
        faces = [
            FakeFace(0, (vertices[0], vertices[2], vertices[4])),
            FakeFace(1, (vertices[1], vertices[3], vertices[5])),
        ]
        if selection_mode == "vertex":
            vertices[0].select = True
        elif selection_mode == "edge":
            edges[0].select = True
        elif selection_mode == "face":
            faces[0].select = True
        vertex_sequence = CountingSequence(vertices)
        edge_sequence = CountingSequence(edges)
        face_sequence = CountingSequence(faces)
        bm = SimpleNamespace(
            verts=vertex_sequence,
            edges=edge_sequence,
            faces=face_sequence,
            select_history=(),
        )
        for sequence in (vertex_sequence, edge_sequence, face_sequence):
            setattr(cast(Any, sequence), "ensure_lookup_table", lambda: None)
            setattr(cast(Any, sequence), "index_update", lambda: None)
        cast(Any, selection.extend_selection_to_mirror)(bm, 0, TOLERANCE)
        assert edge_sequence.iterations == 1
        assert face_sequence.iterations == 1


def _check_extend_selection_wrapper_matrix():
    wrapper_specs = (
        (operators, "operator"),
        (replay_module, "replay"),
    )
    selection_modes = ("empty", "vertex", "edge", "face")
    original_extend = selection.extend_selection_to_mirror
    original_operator_bpy = operators.bpy
    original_operator_bmesh = operators.bmesh
    original_replay_bpy = replay_module.bpy
    original_replay_bmesh = replay_module.bmesh

    for wrapper_module, wrapper_name in wrapper_specs:
        for selection_mode in selection_modes:
            bm = bmesh.new()
            left = [bm.verts.new(co) for co in ((-1.0, 0.0, 0.0), (-1.0, 1.0, 0.0), (-1.0, 0.0, 1.0))]
            right = [bm.verts.new(co) for co in ((1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (1.0, 0.0, 1.0))]
            left_face = bm.faces.new(left)
            right_face = bm.faces.new(right)
            bm.verts.ensure_lookup_table()
            bm.edges.ensure_lookup_table()
            bm.faces.ensure_lookup_table()
            left_edge = next(edge for edge in bm.edges if set(edge.verts) == {left[0], left[1]})
            right_edge = next(edge for edge in bm.edges if set(edge.verts) == {right[0], right[1]})
            if selection_mode == "vertex":
                left[0].select = True
            elif selection_mode == "edge":
                left_edge.select = True
            elif selection_mode == "face":
                left_face.select = True
            history_element = (
                left[0] if selection_mode == "vertex" else left_edge if selection_mode == "edge" else left_face
            )
            if selection_mode != "empty":
                bm.select_history.add(history_element)
            history_before = tuple(cast(Any, bm.select_history))
            calls = []
            updates = []

            def spy_extend(mesh, axis_index, tolerance, *, mesh_object=None):
                calls.append((mesh, axis_index, tolerance, mesh_object))
                return original_extend(mesh, axis_index, tolerance, mesh_object=mesh_object)

            setattr(selection, "extend_selection_to_mirror", spy_extend)
            settings = SimpleNamespace(select_mirrored=True)
            fake_context = SimpleNamespace(scene=SimpleNamespace(ydd_symmetric_edit=settings))
            fake_bpy = SimpleNamespace(context=fake_context)
            fake_mesh_module = SimpleNamespace(
                from_edit_mesh=lambda _mesh: bm,
                update_edit_mesh=lambda _mesh, **kwargs: updates.append(kwargs),
            )
            setattr(wrapper_module, "bpy", fake_bpy)
            setattr(wrapper_module, "bmesh", fake_mesh_module)
            obj = SimpleNamespace(mode="EDIT", data=SimpleNamespace())
            try:
                if wrapper_name == "operator":
                    wrapper_module._maybe_extend_selection_to_mirror(obj, 0, TOLERANCE)
                else:
                    wrapper_module._maybe_extend_selection_to_mirror(obj.data, 0, TOLERANCE, mesh_object=obj)
                assert len(calls) == 1, (wrapper_name, selection_mode)
                assert calls[0][0] is bm
                assert calls[0][1:3] == (0, TOLERANCE)
                assert calls[0][3] is obj, (wrapper_name, selection_mode)
                assert len(updates) == 1, (wrapper_name, selection_mode)
                assert tuple(cast(Any, bm.select_history)) == history_before, (wrapper_name, selection_mode)
                if selection_mode == "empty":
                    assert not any(element.select for element in (*bm.verts, *bm.edges, *bm.faces))
                elif selection_mode == "vertex":
                    assert right[0].select
                elif selection_mode == "edge":
                    assert right_edge.select
                else:
                    assert right_face.select
            finally:
                bm.free()

    setattr(selection, "extend_selection_to_mirror", original_extend)
    setattr(operators, "bpy", original_operator_bpy)
    setattr(operators, "bmesh", original_operator_bmesh)
    setattr(replay_module, "bpy", original_replay_bpy)
    setattr(replay_module, "bmesh", original_replay_bmesh)


def _assert_a_reference_case(coords, axis_index: int, tolerance: float):
    reference = _ReferenceBinLookup(coords, axis_index, tolerance)
    lookup = matching.build_vertex_mirror_lookup(coords, axis_index, tolerance)
    expected_pairs = _pair_table(reference, coords)
    assert matching.build_vertex_pair_table(coords, axis_index, tolerance) == expected_pairs
    batch_count_before = lookup._batch_path_count
    old_mirrored = tuple(reference.find_all_mirrored(coords))
    new_mirrored = lookup.find_all_mirrored(coords)
    assert old_mirrored == new_mirrored
    old_direct = tuple(reference.find_all_direct(coords))
    new_direct = lookup.find_all_direct(coords)
    assert old_direct == new_direct
    assert lookup._batch_path_count > batch_count_before
    return old_mirrored, new_mirrored


@dataclass(frozen=True)
class _ReferenceFaceRecord:
    key: Any
    mirrored_key: Any
    coords: tuple[tuple[float, float, float], ...]
    centroid: tuple[float, float, float]
    vertex_ids: tuple[int, ...]


def _face_records(bm, axis_index: int, tolerance: float):
    records = {}
    for raw_id, face in enumerate(bm.faces, start=1):
        face_id = FaceId(raw_id)
        coords = tuple((float(vertex.co[0]), float(vertex.co[1]), float(vertex.co[2])) for vertex in face.verts)
        key = _types.FaceKey(
            vertex_count=len(coords),
            coordinates=tuple(
                sorted(matching._quantized_coordinate(Vector(coordinate), tolerance) for coordinate in coords)
            ),
        )
        mirrored_coords = tuple(
            tuple(float(component) for component in matching.mirror_coordinate(Vector(coordinate), axis_index))
            for coordinate in coords
        )
        mirrored_key = _types.FaceKey(
            vertex_count=len(coords),
            coordinates=tuple(
                sorted(matching._quantized_coordinate(Vector(coordinate), tolerance) for coordinate in mirrored_coords)
            ),
        )
        centroid = _vector_tuple(face.calc_center_median())
        records[face_id] = _ReferenceFaceRecord(
            key,
            mirrored_key,
            coords,
            cast(tuple[float, float, float], centroid),
            tuple(vertex.index for vertex in face.verts),
        )
    return records


def _reference_eager_face_map(bm, axis_index: int, tolerance: float):
    bm.verts.ensure_lookup_table()
    bm.verts.index_update()
    coords = tuple(vertex.co.copy() for vertex in bm.verts)
    pairs = _pair_table(_ReferenceBinLookup(coords, axis_index, tolerance), coords)
    face_vertex_ids = {
        FaceId(raw_id): tuple(vertex.index for vertex in face.verts) for raw_id, face in enumerate(bm.faces, start=1)
    }
    by_vertices = defaultdict(list)
    for face_id, vertex_ids in face_vertex_ids.items():
        by_vertices[frozenset(vertex_ids)].append(face_id)
    result = {}
    fallback = []
    for face_id, vertex_ids in face_vertex_ids.items():
        mapped = [pairs.get(vertex_id) for vertex_id in vertex_ids]
        own = by_vertices[frozenset(vertex_ids)]
        counterpart = by_vertices[frozenset(mapped)] if all(value is not None for value in mapped) else ()
        if len(mapped) == len(vertex_ids) and len(own) == 1 and len(counterpart) == 1:
            result[face_id] = counterpart[0]
        else:
            fallback.append(face_id)
    if not fallback:
        return result
    records = _face_records(bm, axis_index, tolerance)
    key_to_face_ids = defaultdict(list)
    face_coords = {}
    faces_by_count_centroid = defaultdict(list)
    fallback_ready = False

    def build_records():
        for face_id, record in records.items():
            key_to_face_ids[record.key].append(face_id)

    def ensure_fallback_index():
        nonlocal fallback_ready
        if fallback_ready:
            return
        for face_id, record in records.items():
            face_coords[face_id] = record.coords
            centroid = Vector(record.centroid)
            faces_by_count_centroid[(len(record.coords), matching._quantized_coordinate(centroid, tolerance))].append(
                face_id
            )
        fallback_ready = True

    build_records()
    fallback_assignments = {}
    for face_id in fallback:
        record = records[face_id]
        exact = key_to_face_ids.get(record.mirrored_key)
        if exact:
            candidates = list(exact)
        else:
            ensure_fallback_index()
            mirrored_centroid = matching.mirror_coordinate(Vector(record.centroid), axis_index)
            mirrored_coords = tuple(
                tuple(float(component) for component in matching.mirror_coordinate(Vector(coordinate), axis_index))
                for coordinate in record.coords
            )
            candidates = []
            seen = set()
            found_self = False
            found_other = False
            for centroid_key in matching._iter_quantized_neighborhood(mirrored_centroid, tolerance):
                for candidate_id in faces_by_count_centroid.get((len(record.coords), centroid_key), ()):
                    if candidate_id in seen:
                        continue
                    seen.add(candidate_id)
                    candidate_coords = face_coords[candidate_id]
                    if not _reference_coordinates_match(mirrored_coords, candidate_coords, tolerance):
                        continue
                    candidates.append(candidate_id)
                    found_self = found_self or candidate_id == face_id
                    found_other = found_other or candidate_id != face_id
                    if found_self and found_other:
                        break
                if found_self and found_other:
                    break
        if not candidates:
            continue
        counterpart = candidates[0]
        if abs(record.centroid[axis_index]) > tolerance and counterpart == face_id:
            counterpart = next((candidate for candidate in candidates if candidate != face_id), counterpart)
        fallback_assignments[face_id] = counterpart

    pair_table_targets = set(result.values())
    fallback_target_counts = defaultdict(int)
    for counterpart in fallback_assignments.values():
        fallback_target_counts[counterpart] += 1
    for face_id, counterpart in fallback_assignments.items():
        if counterpart in pair_table_targets or fallback_target_counts[counterpart] > 1:
            continue
        result[face_id] = counterpart

    final_target_counts = defaultdict(int)
    for counterpart in result.values():
        final_target_counts[counterpart] += 1
    return {face_id: counterpart for face_id, counterpart in result.items() if final_target_counts[counterpart] == 1}


def _build_deformed_grid(segments: int, asymmetric: bool = False):
    bm = bmesh.new()
    bmesh.ops.create_grid(bm, x_segments=segments, y_segments=segments, size=2.0)
    for vertex in bm.verts:
        vertex.co.z = 0.08 * math.cos(float(vertex.co.x) * 2.0) * math.cos(float(vertex.co.y))
    if asymmetric:
        bm.verts.ensure_lookup_table()
        bm.verts[0].co.x += 0.123
    return bm


def _build_duplicate_faces():
    bm = bmesh.new()
    left = [bm.verts.new(co) for co in ((-2.0, -1.0, 0.0), (-1.0, -1.0, 0.0), (-1.0, 1.0, 0.0), (-2.0, 1.0, 0.0))]
    right = [bm.verts.new(co) for co in ((2.0, -1.0, 0.0), (1.0, -1.0, 0.0), (1.0, 1.0, 0.0), (2.0, 1.0, 0.0))]
    duplicate = [bm.verts.new(co) for co in ((2.0, -1.0, 0.0), (1.0, -1.0, 0.0), (1.0, 1.0, 0.0), (2.0, 1.0, 0.0))]
    bm.faces.new(left)
    bm.faces.new(right)
    bm.faces.new(duplicate)
    return bm


def _build_loose_ngon_fixture():
    bm = bmesh.new()
    left = [
        bm.verts.new(co)
        for co in ((-3.0, -1.0, 0.0), (-2.0, -1.0, 0.0), (-1.5, 0.0, 0.0), (-2.0, 1.0, 0.0), (-3.0, 1.0, 0.0))
    ]
    right = [
        bm.verts.new((-co[0], co[1], co[2]))
        for co in ((-3.0, -1.0, 0.0), (-2.0, -1.0, 0.0), (-1.5, 0.0, 0.0), (-2.0, 1.0, 0.0), (-3.0, 1.0, 0.0))
    ]
    bm.faces.new(left)
    bm.faces.new(list(reversed(right)))
    bm.verts.new((0.25, 3.0, 0.0))
    return bm


def _build_fallback_bin_order_fixture():
    bm = bmesh.new()
    source_values = ((-2.0, 0.5, 0.0), (-3.0, 2.5, 0.0), (-2.0, 0.5, 1.0))
    source = [bm.verts.new(value) for value in source_values]
    earlier = [bm.verts.new((-value[0], value[1] + 0.6, value[2])) for value in source_values]
    later = [bm.verts.new((-value[0], value[1] - 0.6, value[2])) for value in source_values]
    bm.faces.new(source)
    bm.faces.new(earlier)
    bm.faces.new(later)
    return bm


def _check_topology_equivalence():
    bm = _build_deformed_grid(7)
    original_face_key = face_mapping._face_key
    frame_name = (
        "_carrier_frame_from_coords"
        if hasattr(snapshot_module, "_carrier_frame_from_coords")
        else "_carrier_frame_snapshot"
    )
    original_frame = getattr(snapshot_module, frame_name)
    counters = {"key": 0, "frame": 0}

    def count_key(*args, **kwargs):
        counters["key"] += 1
        return original_face_key(*args, **kwargs)

    def count_frame(*args, **kwargs):
        counters["frame"] += 1
        return original_frame(*args, **kwargs)

    face_key_name = "_face_key"
    setattr(face_mapping, face_key_name, count_key)
    setattr(snapshot_module, frame_name, count_frame)
    try:
        topology = snapshot_module.prepare_topology(bm, 0, TOLERANCE)
    finally:
        setattr(face_mapping, face_key_name, original_face_key)
        setattr(snapshot_module, frame_name, original_frame)
    assert topology.mirror_face_ids == _reference_eager_face_map(bm, 0, TOLERANCE)
    assert topology.matched_faces == topology.total_faces
    assert counters == {"key": 0, "frame": 0}, counters
    carrier_frames = cast(Any, topology.carrier_frames)
    assert not carrier_frames._cache
    bm.free()

    perturbed = _build_deformed_grid(7, asymmetric=True)
    perturbed.faces.ensure_lookup_table()
    perturbed.faces[0].hide = True
    original_face_key = face_mapping._face_key
    fallback_calls = 0

    def count_fallback_key(*args, **kwargs):
        nonlocal fallback_calls
        fallback_calls += 1
        return original_face_key(*args, **kwargs)

    try:
        face_key_name = "_face_key"
        setattr(face_mapping, face_key_name, count_fallback_key)
        topology = snapshot_module.prepare_topology(perturbed, 0, TOLERANCE)
        # The geometric fallback runs at lazy resolve time, so force it
        # while the counting patch is still installed.
        _ = topology.mirror_face_ids
    finally:
        setattr(face_mapping, face_key_name, original_face_key)
    try:
        expected = _reference_eager_face_map(perturbed, 0, TOLERANCE)
        # Exact geometry fallback is resolved by FaceRegistry's NumPy row
        # index; Python FaceKey construction is reserved for the compatibility
        # property and must not run on this path.
        assert fallback_calls == 0
        assert topology.mirror_face_ids == expected
        assert topology.hidden_by_face_id[FaceId(1)] is True
        assert topology.matched_faces == len(expected) < topology.total_faces
        assert topology.total_faces == len(perturbed.faces)
    finally:
        perturbed.free()

    asymmetric = _build_duplicate_faces()
    try:
        asymmetric.faces.ensure_lookup_table()
        asymmetric.faces[0].hide = True
        topology = snapshot_module.prepare_topology(asymmetric, 0, TOLERANCE)
        expected = _reference_eager_face_map(asymmetric, 0, TOLERANCE)
        assert expected == {FaceId(1): FaceId(2)}
        assert topology.mirror_face_ids == expected
        assert topology.matched_faces == len(expected)
        assert topology.total_faces == len(asymmetric.faces)
        assert dict(topology.hidden_by_face_id) == {
            FaceId(1): True,
            FaceId(2): False,
            FaceId(3): False,
        }
        assert topology.matched_faces < topology.total_faces
    finally:
        asymmetric.free()

    fallback_order = _build_fallback_bin_order_fixture()
    try:
        expected = _reference_eager_face_map(fallback_order, 0, 1.0)
        topology = snapshot_module.prepare_topology(fallback_order, 0, 1.0)
        assert expected[FaceId(1)] == FaceId(3)
        assert topology.mirror_face_ids == expected
    finally:
        fallback_order.free()


def _reference_carrier_frame(vertices):
    if not vertices:
        zero = Coordinate3D(0.0, 0.0, 0.0)
        return _types.CarrierFrameSnapshot(vertices, zero, None, None, 0.0)
    count = float(len(vertices))
    origin_vector = Vector(
        (
            sum(vertex.x for vertex in vertices) / count,
            sum(vertex.y for vertex in vertices) / count,
            sum(vertex.z for vertex in vertices) / count,
        )
    )
    newell = Vector((0.0, 0.0, 0.0))
    for index, current in enumerate(vertices):
        following = vertices[(index + 1) % len(vertices)]
        newell.x += (current.y - following.y) * (current.z + following.z)
        newell.y += (current.z - following.z) * (current.x + following.x)
        newell.z += (current.x - following.x) * (current.y + following.y)
    origin = matching._coordinate_3d(origin_vector)
    if newell.length <= 1.0e-12:
        return _types.CarrierFrameSnapshot(vertices, origin, None, None, 0.0)
    normal_vector = newell.normalized()
    basis_u = None
    for vertex in sorted(vertices):
        delta = Vector(vertex.as_tuple()) - origin_vector
        projected = delta - normal_vector * delta.dot(normal_vector)
        if projected.length > 1.0e-12:
            basis_u = projected.normalized()
            break
    if basis_u is None:
        return _types.CarrierFrameSnapshot(vertices, origin, matching._coordinate_3d(normal_vector), None, 0.0)
    deviation = max(abs((Vector(vertex.as_tuple()) - origin_vector).dot(normal_vector)) for vertex in vertices)
    return _types.CarrierFrameSnapshot(
        vertices=vertices,
        origin=origin,
        normal=matching._coordinate_3d(normal_vector),
        basis_u=matching._coordinate_3d(basis_u),
        deviation=float(deviation),
    )


def _check_carrier_frame_lifecycle():
    base = 10000.0
    epsilon = 1.0e-4
    basis_case = (
        Coordinate3D(base, base, base),
        Coordinate3D(base + epsilon, base, base),
        Coordinate3D(base, base + epsilon, base + epsilon),
    )
    collapsed = tuple(Vector(vertex.as_tuple()) for vertex in basis_case)
    assert collapsed[0] == collapsed[1] == collapsed[2]
    basis_snapshot = _reference_carrier_frame(basis_case)
    assert basis_snapshot.normal is not None and basis_snapshot.basis_u is None
    raw_cases = (
        (),
        (Coordinate3D(-1.0, -1.0, 0.0), Coordinate3D(1.0, -1.0, 0.0), Coordinate3D(1.0, 1.0, 0.0)),
        (
            Coordinate3D(-1.0, -1.0, 0.0),
            Coordinate3D(1.0, -1.0, 0.3),
            Coordinate3D(1.0, 1.0, -0.2),
            Coordinate3D(-1.0, 1.0, 0.1),
        ),
        (Coordinate3D(0.0, 0.0, 0.0), Coordinate3D(1.0, 0.0, 0.0), Coordinate3D(2.0, 0.0, 0.0)),
        (Coordinate3D(0.0, 0.0, 0.0), Coordinate3D(0.0, 0.0, 0.0), Coordinate3D(0.0, 0.0, 0.0)),
        (Coordinate3D(0.0, 0.0, 0.0), Coordinate3D(1.0, 0.0, 0.0), Coordinate3D(0.0, 1.0, 0.0)),
        (Coordinate3D(1.0, 0.0, 0.0), Coordinate3D(0.0, 1.0, 0.0), Coordinate3D(0.0, 0.0, 0.0)),
        basis_case,
    )
    vertex_coords = tuple(vertex.as_tuple() for vertices in raw_cases for vertex in vertices)
    face_vertex_ids = {}
    offset = 0
    for index, vertices in enumerate(raw_cases):
        face_id = FaceId(index + 1)
        face_vertex_ids[face_id] = tuple(range(offset, offset + len(vertices)))
        offset += len(vertices)
    lazy_type_name = "LazyCarrierFrameMap"
    lazy_type = cast(Any, getattr(snapshot_module, lazy_type_name))
    lazy = lazy_type(vertex_coords, face_vertex_ids)
    assert lazy._cache == {}
    keys = tuple(lazy)
    assert len(lazy) == len(face_vertex_ids)
    assert keys == tuple(face_vertex_ids)
    assert FaceId(1) in lazy
    assert lazy._cache == {}
    same = lazy_type(vertex_coords, dict(face_vertex_ids))
    different_coords = ((99.0, 99.0, 99.0),) + vertex_coords[1:]
    different = lazy_type(different_coords, dict(face_vertex_ids))
    assert lazy == same
    assert lazy != different
    assert lazy._cache == same._cache == different._cache == {}
    single = lazy_type(vertex_coords[:3], {FaceId(1): (0, 1, 2)})
    loose = lazy_type((vertex_coords[2], vertex_coords[1], vertex_coords[0]), {FaceId(1): (2, 1, 0)})
    assert single == loose
    assert single._cache == loose._cache == {}
    shared = copy.copy(lazy)
    assert shared._vertex_coords is lazy._vertex_coords
    assert shared._face_vertex_ids is lazy._face_vertex_ids
    assert shared._cache is lazy._cache
    cloned = copy.deepcopy(lazy)
    assert cloned._vertex_coords == lazy._vertex_coords
    assert cloned._face_vertex_ids == lazy._face_vertex_ids
    assert cloned._face_vertex_ids is not lazy._face_vertex_ids
    assert cloned._cache == {}
    partial = lazy_type(vertex_coords, dict(face_vertex_ids))
    partial.get(FaceId(1))
    original_factory = snapshot_module._carrier_frame_from_coords
    factory_calls = 0

    def count_factory(vertices):
        nonlocal factory_calls
        factory_calls += 1
        return original_factory(vertices)

    setattr(snapshot_module, "_carrier_frame_from_coords", count_factory)
    try:
        factory_calls = 0
        partial_clone = copy.deepcopy(partial)
    finally:
        setattr(snapshot_module, "_carrier_frame_from_coords", original_factory)
    assert factory_calls == 0
    assert tuple(partial._cache) == tuple(partial_clone._cache) == (FaceId(1),)
    assert len(partial_clone._cache) < len(face_vertex_ids)
    assert partial_clone._cache is not partial._cache
    partial_clone._cache.clear()
    assert tuple(partial._cache) == (FaceId(1),)
    for face_id, vertex_ids in face_vertex_ids.items():
        vertices = tuple(Coordinate3D(*vertex_coords[index]) for index in vertex_ids)
        assert lazy.get(face_id) == _reference_carrier_frame(vertices)
    value = lazy.get(FaceId(2))
    assert lazy[FaceId(2)] is value
    assert lazy.get(FaceId(999)) is None
    assert len(lazy._cache) == len(face_vertex_ids)
    assert lazy[FaceId(6)] != lazy[FaceId(7)]
    cloned_value = cloned.get(FaceId(2))
    assert cloned_value == value
    assert cloned_value is not value


def _check_history_and_single_object_guard():
    vertex_coords = ((0.0, 0.0, 0.0),)
    face_vertex_ids = {FaceId(1): (0,)}
    lazy_type_name = "LazyCarrierFrameMap"
    lazy_type = cast(Any, getattr(snapshot_module, lazy_type_name))
    carrier_frames = lazy_type(vertex_coords, face_vertex_ids)
    history_token = operators._new_history_token()
    resolution = snapshot_module.LazyTopologyResolution(
        numpy.asarray(vertex_coords, dtype=numpy.float64),
        numpy.asarray((0,), dtype=numpy.int64),
        numpy.asarray((0,), dtype=numpy.int64),
        numpy.asarray((1,), dtype=numpy.int64),
        numpy.zeros(1, dtype=bool),
        numpy.empty(0, dtype=bool),
        numpy.zeros(1, dtype=bool),
        0,
        TOLERANCE,
        history_token,
    )
    session = KnifeSession(
        window_pointer=1,
        area_pointer=2,
        region_pointer=3,
        object_name="object",
        mesh_name="mesh",
        axis_index=0,
        source_side="NEGATIVE",
        tolerance=TOLERANCE,
        mirror_face_ids={FaceId(1): FaceId(1)},
        hidden_by_face_id={FaceId(1): False},
        carrier_frames=carrier_frames,
        mesh_select_mode=MeshSelectionMode(True, False, False),
        started_at=1.0,
        history_token=history_token,
        topology_resolution=resolution,
    )
    context = SimpleNamespace(preferences=SimpleNamespace(edit=SimpleNamespace(undo_steps=32)))
    try:
        operators._remember_history_session(session, context)
        record = operators._HISTORY_RECORDS[session.history_token]
        assert record.session.carrier_frames is carrier_frames
        assert record.session.topology_resolution is resolution
        assert resolution.resolve_count == 0
        deep_session = copy.deepcopy(record.session)
        assert deep_session.carrier_frames is not carrier_frames
        assert deep_session.carrier_frames == carrier_frames
        assert deep_session.topology_resolution is not resolution
        assert deep_session.topology_resolution == resolution
        assert resolution.resolve_count == deep_session.topology_resolution.resolve_count == 0
        expected = carrier_frames.get(FaceId(1))
        assert deep_session.carrier_frames.get(FaceId(1)) == expected
        eviction_context = SimpleNamespace(preferences=SimpleNamespace(edit=SimpleNamespace(undo_steps=0)))
        for _ in range(8):
            later = copy.copy(session)
            later.history_token = operators._new_history_token()
            operators._remember_history_session(later, eviction_context)
        assert history_token not in operators._HISTORY_RECORDS
        assert len(operators._HISTORY_RECORDS) == 8
    finally:
        operators.clear_history_records()

    obj = SimpleNamespace(type="MESH")
    area = SimpleNamespace(type="VIEW_3D")
    region = SimpleNamespace(type="WINDOW")
    one = SimpleNamespace(
        edit_object=obj,
        area=area,
        region=region,
        mode="EDIT_MESH",
        objects_in_mode_unique_data=(obj,),
    )
    two = SimpleNamespace(
        edit_object=obj,
        area=area,
        region=region,
        mode="EDIT_MESH",
        objects_in_mode_unique_data=(obj, SimpleNamespace(type="MESH")),
    )
    assert operators._single_edit_mesh_poll(one)
    assert not operators._single_edit_mesh_poll(two)


def _check_multi_object_native_passthrough():
    objects = []
    meshes = []
    session_count = len(operators._SESSIONS)
    try:
        if bpy.context.object is not None and bpy.context.object.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.select_all(action="DESELECT")
        for index in range(2):
            mesh = bpy.data.meshes.new(f"YSEPerfMultiMesh{index}")
            mesh.from_pydata([(float(index), 0.0, 0.0), (float(index), 1.0, 0.0)], [(0, 1)], [])
            obj = bpy.data.objects.new(f"YSEPerfMultiObject{index}", mesh)
            bpy.context.scene.collection.objects.link(obj)
            obj.select_set(True)
            objects.append(obj)
            meshes.append(mesh)
        bpy.context.view_layer.objects.active = objects[0]
        bpy.ops.object.mode_set(mode="EDIT")
        assert len(bpy.context.objects_in_mode_unique_data) == 2
        assert not operators.MESH_OT_ydd_symmetric_edit_intercept.poll(bpy.context)
        bpy.ops.mesh.select_all(action="DESELECT")
        bpy.ops.mesh.select_all(action="SELECT")
        assert all(all(vertex.select for vertex in bmesh.from_edit_mesh(mesh).verts) for mesh in meshes)
        assert len(operators._SESSIONS) == session_count
    finally:
        if bpy.context.object is not None and bpy.context.object.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        for obj in objects:
            bpy.data.objects.remove(obj, do_unlink=True)
        for mesh in meshes:
            if mesh.name in bpy.data.meshes:
                bpy.data.meshes.remove(mesh)


def _check_rip_lookup_validation():
    bm = bmesh.new()
    bmesh.ops.create_grid(bm, x_segments=2, y_segments=1, size=2.0)
    try:
        topology = snapshot_module.prepare_topology(bm, 0, TOLERANCE, mark_vertex_ids=True)
        assert topology.topology_resolution.resolve_count == 0
        bm.verts.ensure_lookup_table()
        bm.verts[0].select = True
        coords = tuple(vertex.co.copy() for vertex in bm.verts)
        valid = topology.vertex_lookup
        assert topology.topology_resolution.resolve_count == 1
        variants = {
            "axis": matching.build_vertex_mirror_lookup(coords, 1, TOLERANCE),
            "tolerance": matching.build_vertex_mirror_lookup(coords, 0, TOLERANCE * 2.0),
            "count": matching.build_vertex_mirror_lookup(coords[:-1], 0, TOLERANCE),
        }
        first_bad_coords = list(coords)
        first_bad_coords[0] = Vector((first_bad_coords[0][0] + 1.0, *first_bad_coords[0][1:]))
        first_bad = matching.build_vertex_mirror_lookup(first_bad_coords, 0, TOLERANCE)
        variants["first"] = first_bad
        last_bad_coords = list(coords)
        last_bad_coords[-1] = Vector((last_bad_coords[-1][0] + 1.0, *last_bad_coords[-1][1:]))
        last_bad = matching.build_vertex_mirror_lookup(last_bad_coords, 0, TOLERANCE)
        variants["last"] = last_bad

        original_match = rip_module._lookup_matches_mesh
        original_builder = matching.build_vertex_mirror_lookup
        validation_calls = 0
        rebuild_calls = 0

        def count_validation(*args, **kwargs):
            nonlocal validation_calls
            validation_calls += 1
            return original_match(*args, **kwargs)

        def count_rebuild(*args, **kwargs):
            nonlocal rebuild_calls
            rebuild_calls += 1
            return original_builder(*args, **kwargs)

        validation_name = "_lookup_matches_mesh"
        builder_name = "build_vertex_mirror_lookup"
        setattr(rip_module, validation_name, count_validation)
        setattr(matching, builder_name, count_rebuild)
        try:
            assert rip_module.build_snapshot(bm, 0, TOLERANCE, lookup=valid) is not None
            assert validation_calls == 1 and rebuild_calls == 0
            for name, variant in variants.items():
                validation_calls = 0
                rebuild_calls = 0
                assert rip_module.build_snapshot(bm, 0, TOLERANCE, lookup=variant) is not None, name
                assert validation_calls == 1 and rebuild_calls == 1, name

            validation_calls = 0
            rebuild_calls = 0
            vertex_layer = bm.verts.layers.int.get(layer_names.VERT_SESSION_ID_LAYER)
            assert vertex_layer is not None
            bm.verts.layers.int.remove(vertex_layer)
            assert rip_module.build_snapshot(bm, 0, TOLERANCE, lookup=valid) is None
            assert validation_calls == 1 and rebuild_calls == 0

            topology = snapshot_module.prepare_topology(bm, 0, TOLERANCE, mark_vertex_ids=True)
            for vertex in bm.verts:
                vertex.select = False
            validation_calls = 0
            rebuild_calls = 0
            assert rip_module.build_snapshot(bm, 0, TOLERANCE, lookup=topology.vertex_lookup) is None
            assert validation_calls == 1 and rebuild_calls == 0
        finally:
            setattr(rip_module, validation_name, original_match)
            setattr(matching, builder_name, original_builder)
    finally:
        bm.free()


def _check_rip_scoped_vertex_ids():
    """RIP IDs are zero until snapshot resolution and scoped to its region."""

    bm = _build_deformed_grid(4)
    try:
        bm.verts.ensure_lookup_table()
        bm.verts.index_update()
        for vertex in bm.verts:
            vertex.select = False
        bm.verts[1].select = True

        topology = snapshot_module.prepare_topology(bm, 0, TOLERANCE, mark_vertex_ids=True)
        vertex_layer = bm.verts.layers.int.get(layer_names.VERT_SESSION_ID_LAYER)
        assert vertex_layer is not None
        assert all(int(vertex[vertex_layer]) == 0 for vertex in bm.verts)

        lookup = topology.topology_resolution.vertex_lookup_unresolved
        # This 5x5 grid has row-major indices. Vertex 1's selected source
        # region is {0, 1, 2, 6}; its X mirrors are {4, 3, 2, 8}.
        expected_region_indices = {0, 1, 2, 6}
        expected_mirror_indices = {2, 3, 4, 8}
        expected_marked_indices = {0, 1, 2, 3, 4, 6, 8}

        snapshot = rip_module.build_snapshot(bm, 0, TOLERANCE, lookup=lookup)
        assert snapshot is not None
        marked = {vertex.index for vertex in bm.verts if int(vertex[vertex_layer]) > 0}
        assert marked == expected_marked_indices
        assert 0 < len(marked) < len(bm.verts)
        assert all(int(bm.verts[index][vertex_layer]) == index + 1 for index in expected_marked_indices)
        assert all(int(bm.verts[index][vertex_layer]) == 0 for index in {5, 7, 9, 10, 24})
        assert expected_region_indices | expected_mirror_indices == expected_marked_indices
        assert {record.vertex_id for record in snapshot.vertices} == {index + 1 for index in expected_region_indices}
        assert {record.mirror_vertex_id for record in snapshot.vertices if record.mirror_vertex_id is not None} == {
            index + 1 for index in expected_mirror_indices
        }

        # The lookup=None compatibility path resolves the same selected
        # region and must produce the same snapshot IDs.
        topology = snapshot_module.prepare_topology(bm, 0, TOLERANCE, mark_vertex_ids=True)
        snapshot_without_lookup = rip_module.build_snapshot(bm, 0, TOLERANCE)
        assert snapshot_without_lookup == snapshot

        # Re-preparing must not retain positive IDs from the prior scoped run.
        topology = snapshot_module.prepare_topology(bm, 0, TOLERANCE, mark_vertex_ids=True)
        vertex_layer = bm.verts.layers.int.get(layer_names.VERT_SESSION_ID_LAYER)
        assert vertex_layer is not None
        assert all(int(vertex[vertex_layer]) == 0 for vertex in bm.verts)
        for vertex in bm.verts:
            vertex.select = False
        assert rip_module.build_snapshot(bm, 0, TOLERANCE) is None
        assert all(int(vertex[vertex_layer]) == 0 for vertex in bm.verts)
        del topology
    finally:
        bm.free()


def _check_rip_resolution_free_lookup_equivalence():
    """RIP's captured-array lookup stays lazy and bit-identical to the eager path."""

    bm = _build_deformed_grid(4)
    try:
        bm.verts.ensure_lookup_table()
        bm.verts.index_update()
        for vertex in bm.verts:
            vertex.select = False
        for index in (1, 2, 3):
            bm.verts[index].select = True

        topology = snapshot_module.prepare_topology(bm, 0, TOLERANCE, mark_vertex_ids=True)
        resolution = topology.topology_resolution
        lazy_lookup = resolution.vertex_lookup_unresolved
        assert resolution.resolve_count == 0
        lazy_snapshot = rip_module.build_snapshot(bm, 0, TOLERANCE, lookup=lazy_lookup)
        assert lazy_snapshot is not None
        assert resolution.resolve_count == 0

        eager_lookup = resolution.vertex_lookup
        assert resolution.resolve_count == 1
        eager_snapshot = rip_module.build_snapshot(bm, 0, TOLERANCE, lookup=eager_lookup)
        assert eager_snapshot is not None
        assert lazy_snapshot == eager_snapshot
    finally:
        bm.free()


class _BulkCollection:
    def __init__(self, values, *, by_name=None):
        self._values = values
        self._by_name = by_name or {}
        self.calls = []

    def __len__(self):
        return len(self._values)

    def foreach_get(self, name, target):
        self.calls.append(name)
        if name in self._by_name:
            data = self._by_name[name]
            if name == "co":
                target[:] = numpy.asarray(data, dtype=numpy.float32).reshape(-1)
            else:
                target[:] = numpy.asarray(data, dtype=target.dtype).reshape(-1)
            return
        if name == "co":
            target[:] = numpy.asarray(self._values, dtype=numpy.float32).reshape(-1)
            return
        target[:] = numpy.asarray(self._values, dtype=target.dtype)


class _BulkVertexCollection(_BulkCollection):
    def foreach_get(self, name, target):
        if name == "hide":
            self.calls.append(name)
            target[:] = False
            values = self._by_name.get("hide")
            if values is not None:
                values_array = numpy.asarray(values, dtype=target.dtype).reshape(-1)
                target[: min(len(target), len(values_array))] = values_array[: len(target)]
            return
        super().foreach_get(name, target)


class _BulkLoopCollection(_BulkCollection):
    """Loop domain: foreach_get vertex_index + single-element .vertex_index."""

    def __getitem__(self, index):
        return SimpleNamespace(vertex_index=int(self._values[index]))


class _BulkPolygonCollection(_BulkCollection):
    """Polygon domain with loop_start/loop_total for §L-3 single-element guards."""

    def __init__(self, hides, starts, totals, *, by_name=None):
        super().__init__(hides, by_name=by_name)
        self._starts = starts
        self._totals = totals

    def __getitem__(self, index):
        return SimpleNamespace(
            loop_start=int(self._starts[index]),
            loop_total=int(self._totals[index]),
        )

    def foreach_get(self, name, target):
        if name == "loop_start":
            self.calls.append(name)
            target[:] = numpy.asarray(self._starts, dtype=target.dtype)
            return
        if name == "loop_total":
            self.calls.append(name)
            target[:] = numpy.asarray(self._totals, dtype=target.dtype)
            return
        super().foreach_get(name, target)


class _BulkMeshData:
    def __init__(self, bm):
        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
        starts = []
        totals = []
        loops = []
        offset = 0
        for face in bm.faces:
            values = tuple(vertex.index for vertex in face.verts)
            loops.extend(values)
            starts.append(offset)
            totals.append(len(values))
            offset += len(values)
        cos = [tuple(float(value) for value in vertex.co) for vertex in bm.verts]
        vert_select = [bool(vertex.select) for vertex in bm.verts]
        vert_hide = [bool(vertex.hide) for vertex in bm.verts]
        edge_select = [bool(edge.select) for edge in bm.edges]
        edge_hide = [bool(edge.hide) for edge in bm.edges]
        face_select = [bool(face.select) for face in bm.faces]
        face_hide = [bool(face.hide) for face in bm.faces]
        self.shape_keys = None
        # vertices: co lives in _values (guard tests mutate length); hide and
        # select are explicit columns so capture tests have an independent oracle.
        self.vertices = _BulkVertexCollection(cos, by_name={"hide": vert_hide, "select": vert_select})
        # edges/polygons: hide stays on _values for existing topology tests;
        # select is optional by_name for §S bulk selection capture.
        self.edges = _BulkCollection(edge_hide, by_name={"select": edge_select})
        # Mutable lists so guard tests can corrupt loop order / totals in place.
        self._starts = starts
        self._totals = totals
        self.polygons = _BulkPolygonCollection(face_hide, self._starts, self._totals, by_name={"select": face_select})
        self.loops = _BulkLoopCollection(loops)

    def sync_selection_from(self, bm):
        """Refresh select flags after free-BMesh mutations (test-only)."""

        self.vertices._by_name["select"] = [bool(vertex.select) for vertex in bm.verts]
        self.edges._by_name["select"] = [bool(edge.select) for edge in bm.edges]
        self.polygons._by_name["select"] = [bool(face.select) for face in bm.faces]

    def polygon_starts(self, name):
        return self._starts if name == "loop_start" else self._totals


class _BulkMeshObject:
    def __init__(self, data):
        self.data = data
        self.update_calls = 0

    def update_from_editmode(self):
        self.update_calls += 1


def _install_bulk_polygon_loop_hooks(bulk_data):
    """No-op compatibility shim: polygons now expose loop_start/loop_total natively.

    Returns a restore callable so existing try/finally sites keep working.
    """

    return _BulkCollection.foreach_get


def _clone_bmesh(source):
    clone = bmesh.new()
    for vertex in source.verts:
        clone.verts.new(tuple(vertex.co))
    clone.verts.ensure_lookup_table()
    clone.verts.index_update()
    for edge in source.edges:
        clone.edges.new((clone.verts[edge.verts[0].index], clone.verts[edge.verts[1].index]))
    clone.edges.ensure_lookup_table()
    for face in source.faces:
        clone.faces.new([clone.verts[vertex.index] for vertex in face.verts])
    clone.faces.ensure_lookup_table()
    for source_vertex, clone_vertex in zip(source.verts, clone.verts, strict=True):
        clone_vertex.select = bool(source_vertex.select)
        clone_vertex.hide = bool(source_vertex.hide)
    for source_edge, clone_edge in zip(source.edges, clone.edges, strict=True):
        clone_edge.select = bool(source_edge.select)
        clone_edge.hide = bool(source_edge.hide)
    for source_face, clone_face in zip(source.faces, clone.faces, strict=True):
        clone_face.select = bool(source_face.select)
        clone_face.hide = bool(source_face.hide)
    return clone


def _selection_state(bm):
    return (
        tuple(bool(vertex.select) for vertex in bm.verts),
        tuple(bool(edge.select) for edge in bm.edges),
        tuple(bool(face.select) for face in bm.faces),
    )


def _reference_classify_selection_overlap(coords, selected_indices, axis_index=0, tolerance=TOLERANCE):
    pairs = matching.build_vertex_pair_table(coords, axis_index, tolerance)
    selected = frozenset(selected_indices)
    shared = frozenset(index for index in selected if abs(coords[index][axis_index]) <= tolerance)
    off = selected - shared
    complete = all(index in pairs for index in off)
    mirrors = {pairs[index] for index in off if index in pairs}
    crossing = mirrors & selected
    if not crossing:
        overlap = MirrorOverlap.SELF_MIRRORED if not off else MirrorOverlap.DISJOINT
    elif complete and mirrors == off:
        overlap = MirrorOverlap.SELF_MIRRORED
    else:
        overlap = MirrorOverlap.PARTIAL
    return OverlapClassification(overlap=overlap, complete=complete, pairs=pairs)


def _reference_extend_selection_to_mirror(bm, axis_index=0, tolerance=TOLERANCE):
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    bm.verts.index_update()
    bm.edges.index_update()
    bm.faces.index_update()
    pairs = matching.build_vertex_pair_table(tuple(vertex.co.copy() for vertex in bm.verts), axis_index, tolerance)
    if not pairs:
        return 0

    selected_verts = [vertex for vertex in bm.verts if vertex.select]
    selected_edges = [edge for edge in bm.edges if edge.select]
    selected_faces = [face for face in bm.faces if face.select]
    edge_by_verts = {frozenset((edge.verts[0].index, edge.verts[1].index)): edge for edge in bm.edges if edge.is_valid}
    face_by_verts = {frozenset(vertex.index for vertex in face.verts): face for face in bm.faces if face.is_valid}
    added = 0
    for vertex in selected_verts:
        partner_index = pairs.get(vertex.index)
        if partner_index is None or partner_index == vertex.index:
            continue
        if partner_index < 0 or partner_index >= len(bm.verts):
            continue
        partner = bm.verts[partner_index]
        if partner.is_valid and not partner.select:
            partner.select = True
            added += 1
    for edge in selected_edges:
        if not edge.is_valid:
            continue
        source_indices = (edge.verts[0].index, edge.verts[1].index)
        partner_indices = [pairs[index] for index in source_indices if index in pairs]
        if len(partner_indices) != len(source_indices):
            continue
        partner_set = frozenset(partner_indices)
        if partner_set == frozenset(source_indices):
            continue
        partner = edge_by_verts.get(partner_set)
        if partner is not None and partner.is_valid and not partner.select:
            partner.select = True
            added += 1
    for face in selected_faces:
        if not face.is_valid:
            continue
        source_indices = tuple(vertex.index for vertex in face.verts)
        partner_indices = [pairs[index] for index in source_indices if index in pairs]
        if len(partner_indices) != len(source_indices):
            continue
        partner_set = frozenset(partner_indices)
        if partner_set == frozenset(source_indices):
            continue
        partner = face_by_verts.get(partner_set)
        if partner is not None and partner.is_valid and not partner.select:
            partner.select = True
            added += 1
    return added


def _build_u6_scoped_fixture(*, perturb=False):
    bm = bmesh.new()
    bmesh.ops.create_grid(bm, x_segments=12, y_segments=12, size=2.0)
    bm.verts.ensure_lookup_table()
    for vertex in bm.verts:
        vertex.co.z = 0.05 * math.cos(float(vertex.co.x) * 2.0) * math.cos(float(vertex.co.y))
    dense_indices = []
    for coordinate in DENSE_CANDIDATE_COORDS:
        dense_indices.append(bm.verts.new(coordinate).index)
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    bm.verts.index_update()
    bm.edges.index_update()
    bm.faces.index_update()
    dense_indices = list(range(len(bm.verts) - len(DENSE_CANDIDATE_COORDS), len(bm.verts)))
    if perturb:
        bm.verts[0].co.x += 0.37
    return bm, dense_indices


def _select_u6_pattern(bm, dense_indices, pattern):
    rng = random.Random(60_006)
    if pattern == "random":
        for vertex in rng.sample(list(bm.verts), 18):
            vertex.select = True
        for edge in rng.sample(list(bm.edges), 12):
            edge.select = True
        for face in rng.sample(list(bm.faces), 8):
            face.select = True
    elif pattern == "on_plane":
        for vertex in bm.verts:
            if abs(float(vertex.co.x)) <= TOLERANCE:
                vertex.select = True
        for face in list(bm.faces)[::31]:
            face.select = True
    elif pattern == "negative":
        for vertex in [vertex for vertex in bm.verts if float(vertex.co.x) < -TOLERANCE][::7]:
            vertex.select = True
        for edge in [edge for edge in bm.edges if max(float(vertex.co.x) for vertex in edge.verts) < -TOLERANCE][::11]:
            edge.select = True
        for face in [face for face in bm.faces if max(float(vertex.co.x) for vertex in face.verts) < -TOLERANCE][::13]:
            face.select = True
    elif pattern == "dense":
        for index in dense_indices[1::2]:
            bm.verts[index].select = True
    else:
        raise AssertionError(pattern)


def _check_u6_scoped_selection_equivalence():
    for pattern, perturb in (
        ("random", False),
        ("on_plane", False),
        ("negative", False),
        ("random", True),
        ("dense", False),
    ):
        source, dense_indices = _build_u6_scoped_fixture(perturb=perturb)
        try:
            _select_u6_pattern(source, dense_indices, pattern)
            expected_bm = _clone_bmesh(source)
            actual_bm = _clone_bmesh(source)
            expected_added = _reference_extend_selection_to_mirror(expected_bm)

            full_build_calls = 0
            original_selection_builder = selection.build_vertex_pair_table

            def count_selection_full_build(*args, **kwargs):
                nonlocal full_build_calls
                full_build_calls += 1
                return original_selection_builder(*args, **kwargs)

            setattr(selection, "build_vertex_pair_table", count_selection_full_build)
            try:
                actual_added = selection.extend_selection_to_mirror(actual_bm, 0, TOLERANCE)
            finally:
                setattr(selection, "build_vertex_pair_table", original_selection_builder)
            assert full_build_calls == 0, (pattern, full_build_calls)
            assert actual_added == expected_added, pattern
            assert _selection_state(actual_bm) == _selection_state(expected_bm), pattern

            coords = numpy.asarray([tuple(vertex.co) for vertex in source.verts], dtype=numpy.float64)
            selected = tuple(vertex.index for vertex in source.verts if vertex.select)
            expected = _reference_classify_selection_overlap(coords, selected)
            registry = matching.VertexRegistry(coords, 0, TOLERANCE)
            resolved = registry.resolve_closure(selected)
            assert resolved is not None
            closure, expected_scoped_pairs = resolved
            assert expected_scoped_pairs == {
                index: partner for index, partner in expected.pairs.items() if index in set(closure.tolist())
            }

            full_build_calls = 0
            original_matching_builder = matching.build_vertex_pair_table

            def count_matching_full_build(*args, **kwargs):
                nonlocal full_build_calls
                full_build_calls += 1
                return original_matching_builder(*args, **kwargs)

            setattr(matching, "build_vertex_pair_table", count_matching_full_build)
            try:
                actual = matching.classify_selection_overlap(
                    coords,
                    selected,
                    axis_index=0,
                    tolerance=TOLERANCE,
                )
            finally:
                setattr(matching, "build_vertex_pair_table", original_matching_builder)
            assert full_build_calls == 0, (pattern, full_build_calls)
            assert actual.overlap == expected.overlap, pattern
            assert actual.complete == expected.complete, pattern
            assert dict.__len__(actual.pairs) == len(expected_scoped_pairs), pattern
            assert all(actual.pairs[index] == partner for index, partner in expected_scoped_pairs.items()), pattern
            assert actual.pairs == expected.pairs, pattern

            replay_actual = replay_module.classify_mirror_selection(
                coords,
                selected,
                axis_index=0,
                tolerance=TOLERANCE,
            )
            expected_off = frozenset(index for index in selected if abs(coords[index][0]) > TOLERANCE)
            expected_mirror_by_source = {
                index: expected.pairs[index] for index in expected_off if index in expected.pairs
            }
            assert replay_actual.selected == frozenset(selected)
            assert replay_actual.off == expected_off
            assert replay_actual.shared == frozenset(selected) - expected_off
            assert replay_actual.mirror_by_source == expected_mirror_by_source
            assert replay_actual.missing == frozenset(expected_off - expected_mirror_by_source.keys())
            assert replay_actual.overlap == expected.overlap
            assert replay_actual.complete == expected.complete
            assert dict.__len__(replay_actual.pairs) == len(expected_scoped_pairs)
            assert replay_actual.pairs == expected.pairs
        finally:
            expected_bm.free()
            actual_bm.free()
            source.free()


def _check_u6_face_duplicate_vertex_set_tie():
    class FakeVertex:
        def __init__(self, index):
            self.index = index
            self.is_valid = True
            self.link_faces = []

    class FakeFace:
        def __init__(self, index, vertices, is_valid=True):
            self.index = index
            self.verts = vertices
            self.is_valid = is_valid
            for vertex in vertices:
                vertex.link_faces.append(self)

    vertices = [FakeVertex(index) for index in range(4)]
    lower = FakeFace(3, vertices)
    higher = FakeFace(9, tuple(reversed(vertices)))
    FakeFace(12, vertices, is_valid=False)
    bm = SimpleNamespace(verts=vertices)
    assert selection._find_face_by_verts(bm, frozenset(range(4))) is higher
    assert higher.index > lower.index


def _check_u6_registry_fallback():
    class UnavailableRegistry:
        def __init__(self, *_args, **_kwargs):
            pass

        def resolve_closure(self, _seeds):
            return None

        def resolve_closure_arrays(self, _seeds):
            return None

    coords = numpy.asarray(((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0)), dtype=numpy.float64)
    expected = _reference_classify_selection_overlap(coords, (0,))
    original_registry = matching.VertexRegistry
    original_builder = matching.build_vertex_pair_table
    build_calls = 0

    def count_build(*args, **kwargs):
        nonlocal build_calls
        build_calls += 1
        return original_builder(*args, **kwargs)

    setattr(matching, "VertexRegistry", UnavailableRegistry)
    setattr(matching, "build_vertex_pair_table", count_build)
    try:
        actual = matching.classify_selection_overlap(coords, (0,), axis_index=0, tolerance=TOLERANCE)
    finally:
        setattr(matching, "build_vertex_pair_table", original_builder)
        setattr(matching, "VertexRegistry", original_registry)
    assert build_calls == 1
    assert actual == expected

    bm = bmesh.new()
    try:
        left = bm.verts.new((-1.0, 0.0, 0.0))
        bm.verts.new((1.0, 0.0, 0.0))
        bm.verts.ensure_lookup_table()
        left.select = True
        original_registry = selection.VertexRegistry
        original_builder = selection.build_vertex_pair_table
        build_calls = 0
        setattr(selection, "VertexRegistry", UnavailableRegistry)
        setattr(selection, "build_vertex_pair_table", count_build)
        try:
            added = selection.extend_selection_to_mirror(bm, 0, TOLERANCE)
        finally:
            setattr(selection, "build_vertex_pair_table", original_builder)
            setattr(selection, "VertexRegistry", original_registry)
        assert build_calls == 1
        assert added == 1
        assert bm.verts[1].select
    finally:
        bm.free()


def _check_selection_snapshot_bulk_equivalence():
    """capture_selection_snapshot bulk and BMesh paths agree bit-for-bit."""

    from ydd_symmetric_edit import element_pairs

    bm = bmesh.new()
    try:
        left = [bm.verts.new(co) for co in ((-1.0, 0.0, 0.0), (-1.0, 1.0, 0.0), (-1.0, 0.0, 1.0))]
        right = [bm.verts.new(co) for co in ((1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (1.0, 0.0, 1.0))]
        left_face = bm.faces.new(left)
        right_face = bm.faces.new(right)
        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
        bm.verts.index_update()
        bm.edges.index_update()
        bm.faces.index_update()
        left_edge = next(edge for edge in bm.edges if set(edge.verts) == {left[0], left[1]})
        left[0].select = True
        left_edge.select = True
        left_face.select = True
        bm.select_history.clear()
        bm.select_history.add(left[0])
        bm.select_history.add(left_edge)

        bulk = _clone_bmesh(bm)
        bulk.select_history.clear()
        bulk.select_history.add(bulk.verts[left[0].index])
        bulk.select_history.add(bulk.edges[left_edge.index])
        bulk_data = _BulkMeshData(bulk)
        mesh_object = _BulkMeshObject(bulk_data)
        original_foreach = _install_bulk_polygon_loop_hooks(bulk_data)
        try:
            compat = snapshot_module.capture_selection_snapshot(
                bm, domains=("VERT", "EDGE", "FACE"), include_history=True
            )
            captured = snapshot_module.capture_selection_snapshot(
                bulk,
                mesh_object=mesh_object,
                domains=("VERT", "EDGE", "FACE"),
                include_history=True,
            )
            assert mesh_object.update_calls == 1
            assert captured.coords.dtype == numpy.float64
            assert captured.coords.shape == (len(bm.verts), 3)
            # Bulk promotes Mesh float32; compare at capture precision.
            assert numpy.array_equal(
                compat.coords.astype(numpy.float32),
                captured.coords.astype(numpy.float32),
            )
            assert tuple(int(value) for value in compat.selected_verts) == tuple(
                int(value) for value in captured.selected_verts
            )
            assert tuple(int(value) for value in compat.selected_edges) == tuple(
                int(value) for value in captured.selected_edges
            )
            assert tuple(int(value) for value in compat.selected_faces) == tuple(
                int(value) for value in captured.selected_faces
            )
            assert compat.history_indices == captured.history_indices
            assert len(compat.history_coords) == len(captured.history_coords) == 1
            assert tuple(float(value) for value in compat.history_coords[0]) == tuple(
                float(value) for value in captured.history_coords[0]
            )

            # extend_selection_to_mirror: bulk vs compat selection outcomes match.
            for source in (bm, bulk):
                for element in (*source.verts, *source.edges, *source.faces):
                    element.select = False
                source.verts[left[0].index].select = True
                source.edges[left_edge.index].select = True
                source.faces[left_face.index].select = True
            bulk_data.sync_selection_from(bulk)
            added_compat = selection.extend_selection_to_mirror(bm, 0, TOLERANCE, mesh_object=None)
            added_bulk = selection.extend_selection_to_mirror(bulk, 0, TOLERANCE, mesh_object=mesh_object)
            assert added_compat == added_bulk
            assert _selection_state(bm) == _selection_state(bulk)
            assert right[0].select and right_face.select

            # build_element_pair_maps: vert/edge/face pair tables match.
            maps_compat = element_pairs.build_element_pair_maps(bm, 0, TOLERANCE, mesh_object=None)
            maps_bulk = element_pairs.build_element_pair_maps(bulk, 0, TOLERANCE, mesh_object=mesh_object)
            assert maps_compat.vert_pairs == maps_bulk.vert_pairs
            assert maps_compat.edge_pair_by_index == maps_bulk.edge_pair_by_index
            assert maps_compat.face_pair_by_index == maps_bulk.face_pair_by_index

            # classify via _vertex_snapshot bulk/compat agreement.
            bulk_data.sync_selection_from(bulk)
            coords_c, selected_c, history_c, history_i_c = replay_module._vertex_snapshot(bm)
            coords_b, selected_b, history_b, history_i_b = replay_module._vertex_snapshot(bulk, mesh_object=mesh_object)
            assert selected_c == selected_b
            assert history_i_c == history_i_b
            snap_c = replay_module.classify_mirror_selection(coords_c, selected_c, axis_index=0, tolerance=TOLERANCE)
            snap_b = replay_module.classify_mirror_selection(coords_b, selected_b, axis_index=0, tolerance=TOLERANCE)
            assert snap_c.selected == snap_b.selected
            assert snap_c.pairs == snap_b.pairs
            assert snap_c.mirror_by_source == snap_b.mirror_by_source
            assert snap_c.overlap == snap_b.overlap
        finally:
            _BulkCollection.foreach_get = original_foreach
            bulk.free()
    finally:
        bm.free()


def _check_capture_resolve_contract():
    """Bulk and compatibility snapshots agree before/after lazy resolution."""

    fixtures = []
    for builder in (_build_deformed_grid, _build_duplicate_faces, _build_loose_ngon_fixture):
        source = builder(5) if builder is _build_deformed_grid else builder()
        fixtures.append(source)
    for source in fixtures:
        compat = source
        compat.verts.ensure_lookup_table()
        compat.verts.index_update()
        bulk = _clone_bmesh(compat)
        bulk_data = _BulkMeshData(bulk)
        mesh_object = _BulkMeshObject(bulk_data)
        original_foreach = _install_bulk_polygon_loop_hooks(bulk_data)
        try:
            compatibility = snapshot_module.prepare_topology(compat, 0, TOLERANCE)
            captured = snapshot_module.prepare_topology(bulk, 0, TOLERANCE, mesh_object=mesh_object)
            assert mesh_object.update_calls == 1
            assert compatibility.total_faces == captured.total_faces
            assert compatibility.mirror_face_ids == captured.mirror_face_ids
            assert compatibility.hidden_by_face_id == captured.hidden_by_face_id
            assert compatibility.topology_resolution.coords64.dtype == numpy.float64
            assert captured.topology_resolution.loop_verts.dtype == numpy.int64
            assert captured.topology_resolution.vertex_count == len(bulk.verts)
            assert captured.topology_resolution.edge_count == len(bulk.edges)
            assert captured.topology_resolution.face_count == len(bulk.faces)
            coords_matrix = captured.topology_resolution.coords64
            assert matching._one_sided_pair_table(coords_matrix, 0, TOLERANCE) == matching.build_vertex_pair_table(
                tuple(Vector(tuple(row)) for row in coords_matrix.tolist()), 0, TOLERANCE
            )
            bulk_data.shape_keys = object()
            shape_key_object = _BulkMeshObject(bulk_data)
            shape_key_topology = snapshot_module.prepare_topology(bulk, 0, TOLERANCE, mesh_object=shape_key_object)
            assert shape_key_object.update_calls == 0
            assert shape_key_topology.total_faces == captured.total_faces
            bulk_data.shape_keys = None
            shared_object = _BulkMeshObject(bulk_data)
            shared = snapshot_module.prepare_topology(bulk, 0, TOLERANCE, mesh_object=shared_object)
            assert shared.mirror_face_ids == compatibility.mirror_face_ids
        finally:
            _BulkCollection.foreach_get = original_foreach
            bulk.free()
            compat.free()

    empty = bmesh.new()
    try:
        data = _BulkMeshData(empty)
        compatibility = snapshot_module.prepare_topology(empty, 0, TOLERANCE)
        captured = snapshot_module.prepare_topology(empty, 0, TOLERANCE, mesh_object=_BulkMeshObject(data))
        assert compatibility.topology_resolution.coords64.shape == (0, 3)
        assert captured.topology_resolution.coords64.shape == (0, 3)
        assert compatibility.mirror_face_ids == captured.mirror_face_ids == {}
    finally:
        empty.free()


def _check_rip_capture_omits_vertex_edge_hides():
    """RIP capture skips vertex/edge hide reads but keeps face hide semantics."""

    bm = _build_deformed_grid(3)
    try:
        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
        for index, vertex in enumerate(bm.verts):
            vertex.hide = index % 3 == 1
        for index, edge in enumerate(bm.edges):
            edge.hide = index % 3 == 2
        for index, face in enumerate(bm.faces):
            face.hide = index % 2 == 1

        non_rip = snapshot_module.prepare_topology(bm, 0, TOLERANCE)
        assert non_rip.topology_resolution.hide_vertices.any()
        assert non_rip.topology_resolution.hide_edges.any()
        assert non_rip.topology_resolution.hide_faces.any()

        hide_before_rip = (
            tuple(bool(vertex.hide) for vertex in bm.verts),
            tuple(bool(edge.hide) for edge in bm.edges),
            tuple(bool(face.hide) for face in bm.faces),
        )
        for vertex in bm.verts:
            vertex.select = False
        for index in (0, 2, 3):
            bm.verts[index].select = True

        data = _BulkMeshData(bm)
        rip_topology = snapshot_module.prepare_topology(
            bm,
            0,
            TOLERANCE,
            mark_vertex_ids=True,
            mesh_object=_BulkMeshObject(data),
        )
        resolution = rip_topology.topology_resolution
        assert "hide" not in data.vertices.calls
        assert "hide" not in data.edges.calls
        assert "hide" in data.polygons.calls
        assert numpy.array_equal(resolution.hide_vertices, numpy.zeros(len(bm.verts), dtype=bool))
        assert numpy.array_equal(resolution.hide_edges, numpy.zeros(len(bm.edges), dtype=bool))
        assert numpy.array_equal(
            resolution.hide_faces,
            numpy.asarray([bool(face.hide) for face in bm.faces], dtype=bool),
        )
        assert bm.verts.layers.int.get(layer_names.VERT_HIDDEN_LAYER) is None
        assert bm.edges.layers.int.get(layer_names.EDGE_HIDDEN_LAYER) is None
        rip_snapshot = rip_module.build_snapshot(
            bm,
            0,
            TOLERANCE,
            lookup=resolution.vertex_lookup_unresolved,
        )
        assert rip_snapshot is not None
        snapshot_module.remove_temporary_layers(bm)
        hide_after_rip = (
            tuple(bool(vertex.hide) for vertex in bm.verts),
            tuple(bool(edge.hide) for edge in bm.edges),
            tuple(bool(face.hide) for face in bm.faces),
        )
        assert hide_after_rip == hide_before_rip

        fallback_bm = _clone_bmesh(bm)
        try:
            fallback_data = _BulkMeshData(fallback_bm)
            fallback_data.shape_keys = object()
            fallback_object = _BulkMeshObject(fallback_data)
            fallback_topology = snapshot_module.prepare_topology(
                fallback_bm,
                0,
                TOLERANCE,
                mark_vertex_ids=True,
                mesh_object=fallback_object,
            )
            assert fallback_object.update_calls == 0
            assert fallback_data.vertices.calls == []
            assert fallback_data.edges.calls == []
            assert fallback_data.polygons.calls == []
            fallback_resolution = fallback_topology.topology_resolution
            assert numpy.array_equal(
                fallback_resolution.hide_vertices,
                numpy.zeros(len(fallback_bm.verts), dtype=bool),
            )
            assert numpy.array_equal(
                fallback_resolution.hide_edges,
                numpy.zeros(len(fallback_bm.edges), dtype=bool),
            )
            assert numpy.array_equal(
                fallback_resolution.hide_faces,
                numpy.asarray([bool(face.hide) for face in fallback_bm.faces], dtype=bool),
            )
            assert fallback_bm.verts.layers.int.get(layer_names.VERT_HIDDEN_LAYER) is None
            assert fallback_bm.edges.layers.int.get(layer_names.EDGE_HIDDEN_LAYER) is None
        finally:
            fallback_bm.free()

        fallback = snapshot_module._capture_bmesh_snapshot(bm, skip_vertex_edge_hides=True)
        assert numpy.array_equal(fallback[4], numpy.zeros(len(bm.verts), dtype=bool))
        assert numpy.array_equal(fallback[5], numpy.zeros(len(bm.edges), dtype=bool))
        assert numpy.array_equal(
            fallback[6],
            numpy.asarray([bool(face.hide) for face in bm.faces], dtype=bool),
        )
        compatibility = snapshot_module._capture_bmesh_snapshot(bm)
        assert compatibility[4].any()
        assert compatibility[5].any()
    finally:
        bm.free()


def _assert_bulk_snapshot_matches_edit_bmesh(obj, bm):
    for index, vertex in enumerate(bm.verts):
        vertex.hide = index % 3 == 1
    for index, edge in enumerate(bm.edges):
        edge.hide = index % 3 == 2
    for index, face in enumerate(bm.faces):
        face.hide = index % 2 == 1
    bulk = snapshot_module._capture_mesh_snapshot(obj, bm)
    compatibility = snapshot_module._capture_bmesh_snapshot(bm)
    assert bulk is not None
    assert len(bulk) == len(compatibility)
    for actual, expected in zip(bulk, compatibility, strict=True):
        assert numpy.array_equal(actual, expected)


def _check_bulk_edit_mesh_order_variants():
    builders = (_build_duplicate_faces, _build_loose_ngon_fixture, lambda: _build_deformed_grid(3))
    for fixture_index, builder in enumerate(builders):
        source = builder()
        mesh = bpy.data.meshes.new(f"YSEPerfBulkOrderMesh{fixture_index}")
        source.to_mesh(mesh)
        source.free()
        obj = bpy.data.objects.new(f"YSEPerfBulkOrderObject{fixture_index}", mesh)
        shared = None
        try:
            if bpy.context.object is not None and bpy.context.object.mode != "OBJECT":
                bpy.ops.object.mode_set(mode="OBJECT")
            bpy.ops.object.select_all(action="DESELECT")
            bpy.context.scene.collection.objects.link(obj)
            if fixture_index == len(builders) - 1:
                shared = bpy.data.objects.new("YSEPerfBulkOrderShared", mesh)
                bpy.context.scene.collection.objects.link(shared)
                assert mesh.users >= 2
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.mode_set(mode="EDIT")
            edit_bm = bmesh.from_edit_mesh(mesh)
            _assert_bulk_snapshot_matches_edit_bmesh(obj, edit_bm)
            if fixture_index == len(builders) - 1:
                bmesh.ops.create_cube(edit_bm, size=0.25)
                _assert_bulk_snapshot_matches_edit_bmesh(obj, edit_bm)
                bmesh.update_edit_mesh(mesh, loop_triangles=False, destructive=True)
                edit_bm = bmesh.from_edit_mesh(mesh)
                _assert_bulk_snapshot_matches_edit_bmesh(obj, edit_bm)
        finally:
            if obj.mode != "OBJECT":
                bpy.ops.object.mode_set(mode="OBJECT")
            if shared is not None:
                bpy.data.objects.remove(shared, do_unlink=True)
            bpy.data.objects.remove(obj, do_unlink=True)
            if mesh.name in bpy.data.meshes:
                bpy.data.meshes.remove(mesh)


def _finish_scope_fixture(face_coordinates, path_face_index):
    source = bmesh.new()
    for coordinates in face_coordinates:
        source.faces.new([source.verts.new(value) for value in coordinates])
    mesh = bpy.data.meshes.new("YSEPerfFinishScopeMesh")
    source.to_mesh(mesh)
    source.free()
    obj = bpy.data.objects.new("YSEPerfFinishScopeObject", mesh)
    window_pointer = operators._window_key(bpy.context)
    if bpy.context.object is not None and bpy.context.object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="DESELECT")
    bpy.context.scene.collection.objects.link(obj)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    bm = bmesh.from_edit_mesh(mesh)
    token = operators._new_history_token()
    topology = snapshot_module.prepare_topology(bm, 0, TOLERANCE, token, mesh_object=obj)
    bm.faces.ensure_lookup_table()
    path_face = bm.faces[path_face_index]
    edge_layer = bm.edges.layers.int.get(layer_names.EDGE_ORIGINAL_LAYER)
    assert edge_layer is not None
    path_face.edges[0][edge_layer] = 0
    bmesh.update_edit_mesh(mesh, loop_triangles=False, destructive=False)
    session = KnifeSession(
        window_pointer=window_pointer,
        area_pointer=0,
        region_pointer=0,
        object_name=obj.name,
        mesh_name=mesh.name,
        axis_index=0,
        source_side="NEGATIVE",
        tolerance=TOLERANCE,
        mirror_face_ids={},
        hidden_by_face_id=topology.hidden_by_face_id,
        carrier_frames={},
        mesh_select_mode=MeshSelectionMode(True, False, False),
        started_at=1.0,
        history_token=token,
        topology_resolution=topology.topology_resolution,
    )
    return obj, mesh, window_pointer, session, topology.topology_resolution


def _check_finish_scope_warning_matrix():
    left = ((-2.0, -1.0, 0.0), (-1.0, -1.0, 0.0), (-1.5, 1.0, 0.0))
    right = tuple((-x, y, z) for x, y, z in reversed(left))
    unmatched = ((3.0, 2.0, 0.0), (4.0, 2.0, 0.0), (3.5, 3.0, 0.0))
    cases = (
        ("matched-scope-with-distant-unmatched", (left, right, unmatched), 0, False, True),
        ("unmatched-in-scope", (left, right, unmatched), 2, True, False),
        ("zero-match-mesh", (unmatched,), 0, True, False),
    )
    for label, faces, path_face_index, expect_warning, expect_apply in cases:
        obj, mesh, window_pointer, session, resolution = _finish_scope_fixture(faces, path_face_index)
        original_crossing_plan = stitch_crossings.plan_mirrored_path_crossings
        original_boundary_check = stitch_reflect.reflected_path_uses_only_target_boundaries
        original_apply = stitch_reflect.apply_reflected_path_topology
        original_backup_create = operators.backup.create_topology_backup
        original_backup_remove = operators.backup.remove_backup
        apply_calls = 0

        def count_apply(_bm, source_edges, *_args, **_kwargs):
            nonlocal apply_calls
            apply_calls += 1
            if _kwargs.get("return_summary"):
                return len(source_edges), 0, "", stitch_common.SelectionMutationSummary()
            return len(source_edges), 0, ""

        setattr(stitch_crossings, "plan_mirrored_path_crossings", lambda *_args, **_kwargs: ([], ""))
        setattr(stitch_reflect, "reflected_path_uses_only_target_boundaries", lambda *_args, **_kwargs: True)
        setattr(stitch_reflect, "apply_reflected_path_topology", count_apply)
        setattr(operators.backup, "create_topology_backup", lambda _bm: object())
        setattr(operators.backup, "remove_backup", lambda _value: None)
        operators._SESSIONS[window_pointer] = session
        reports = []
        fake_operator = SimpleNamespace(
            report=lambda level, message: reports.append((level, message)),
            preserve_history_layers=False,
        )
        try:
            result = operators.MESH_OT_ydd_symmetric_edit_finish.execute(fake_operator, bpy.context)
            assert result == {"FINISHED"}, label
            warnings = [message for level, message in reports if level == {"WARNING"}]
            if expect_warning:
                assert warnings == [
                    "ydd Symmetric Edit: 1 cut face(s) have no exact mirrored counterpart"
                    " (native cut kept; mirror manually or undo)"
                ], label
            else:
                assert warnings == [], label
            assert bool(apply_calls) is expect_apply, label
            if label == "matched-scope-with-distant-unmatched":
                assert resolution.resolve_count == 0
                assert resolution.partial_face_resolve_count >= 1
        finally:
            setattr(stitch_crossings, "plan_mirrored_path_crossings", original_crossing_plan)
            setattr(stitch_reflect, "reflected_path_uses_only_target_boundaries", original_boundary_check)
            setattr(stitch_reflect, "apply_reflected_path_topology", original_apply)
            setattr(operators.backup, "create_topology_backup", original_backup_create)
            setattr(operators.backup, "remove_backup", original_backup_remove)
            operators._SESSIONS.pop(window_pointer, None)
            if obj.mode != "OBJECT":
                bpy.ops.object.mode_set(mode="OBJECT")
            bpy.data.objects.remove(obj, do_unlink=True)
            if mesh.name in bpy.data.meshes:
                bpy.data.meshes.remove(mesh)


def _check_scoped_face_overlay_and_materialize():
    bm = _build_deformed_grid(3)
    try:
        topology = snapshot_module.prepare_topology(bm, 0, TOLERANCE, 808)
        resolution = topology.topology_resolution
        face_id_layer = bm.faces.layers.int.get(layer_names.FACE_ID_LAYER)
        edge_layer = bm.edges.layers.int.get(layer_names.EDGE_ORIGINAL_LAYER)
        assert face_id_layer is not None and edge_layer is not None
        bm.faces.ensure_lookup_table()
        path_edge = bm.faces[0].edges[0]
        path_edge[edge_layer] = 0
        scope = {FaceId(int(face[face_id_layer])) for face in path_edge.link_faces}
        resolution.resolve_faces(scope)

        class GetOnlyFaceMap:
            def get(self, key, default=None):
                return resolution.scoped_mirror_face_ids.get(key, default)

            def __iter__(self):
                raise AssertionError("scope overlay must not iterate the lazy face map")

            def __len__(self):
                raise AssertionError("scope overlay must not resolve the lazy face map")

        overlay = face_mapping.resolve_live_mirror_face_map(
            bm,
            GetOnlyFaceMap(),
            0,
            TOLERANCE,
            path_edges=(path_edge,),
        )
        assert set(overlay).issuperset(scope)
        assert resolution.resolve_count == 0

        resolution.materialize_faces(bm, scope)
        mirror_layer = bm.faces.layers.int.get(layer_names.FACE_MIRROR_ID_LAYER)
        assert mirror_layer is not None
        for face in bm.faces:
            face_id = FaceId(int(face[face_id_layer]))
            expected = int(resolution.scoped_mirror_face_ids.get(face_id) or 0) if face_id in scope else 0
            assert int(face[mirror_layer]) == expected
        assert resolution.resolve_count == 0
    finally:
        bm.free()


def _check_finish_zero_match_decline():
    source = bmesh.new()
    vertices = [source.verts.new(value) for value in ((1.0, 0.0, 0.0), (2.0, 0.0, 0.0), (1.0, 1.0, 0.0))]
    source.faces.new(vertices)
    mesh = bpy.data.meshes.new("YSEPerfZeroMatchMesh")
    source.to_mesh(mesh)
    source.free()
    obj = bpy.data.objects.new("YSEPerfZeroMatchObject", mesh)
    window_pointer = operators._window_key(bpy.context)
    try:
        if bpy.context.object is not None and bpy.context.object.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.select_all(action="DESELECT")
        bpy.context.scene.collection.objects.link(obj)
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode="EDIT")
        bm = bmesh.from_edit_mesh(mesh)
        token = operators._new_history_token()
        topology = snapshot_module.prepare_topology(bm, 0, TOLERANCE, token, mesh_object=obj)
        bmesh.update_edit_mesh(mesh, loop_triangles=False, destructive=False)
        bm.edges.ensure_lookup_table()
        split = bmesh.ops.subdivide_edges(bm, edges=(bm.edges[0],), cuts=1, use_grid_fill=False)
        marker_layer = bm.edges.layers.int.get(layer_names.EDGE_ORIGINAL_LAYER)
        assert marker_layer is not None
        # subdivide interpolates int layers from the parent edge; a real knife
        # cut leaves the marker at 0, so zero the split edges explicitly.
        for element in split["geom_split"]:
            if isinstance(element, bmesh.types.BMEdge):
                element[marker_layer] = 0
        native_vertex_count = len(bm.verts)
        assert native_vertex_count > 3
        bmesh.update_edit_mesh(mesh, loop_triangles=False, destructive=True)
        session = KnifeSession(
            window_pointer=window_pointer,
            area_pointer=0,
            region_pointer=0,
            object_name=obj.name,
            mesh_name=mesh.name,
            axis_index=0,
            source_side="NEGATIVE",
            tolerance=TOLERANCE,
            mirror_face_ids={},
            hidden_by_face_id=topology.hidden_by_face_id,
            carrier_frames={},
            mesh_select_mode=MeshSelectionMode(True, False, False),
            started_at=1.0,
            history_token=token,
            topology_resolution=topology.topology_resolution,
        )
        operators._SESSIONS[window_pointer] = session
        reports = []
        fake_operator = SimpleNamespace(
            report=lambda level, message: reports.append((level, message)),
            preserve_history_layers=False,
        )
        result = operators.MESH_OT_ydd_symmetric_edit_finish.execute(fake_operator, bpy.context)
        assert result == {"FINISHED"}
        assert reports == [
            (
                {"WARNING"},
                "ydd Symmetric Edit: 1 cut face(s) have no exact mirrored counterpart"
                " (native cut kept; mirror manually or undo)",
            )
        ]
        restored = bmesh.from_edit_mesh(mesh)
        assert len(restored.verts) == native_vertex_count
        assert restored.faces.layers.int.get(layer_names.FACE_ID_LAYER) is None
    finally:
        operators._SESSIONS.pop(window_pointer, None)
        if obj.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        bpy.data.objects.remove(obj, do_unlink=True)
        if mesh.name in bpy.data.meshes:
            bpy.data.meshes.remove(mesh)


def _check_hide_layer_omission_and_consumers():
    bm = _build_deformed_grid(3)
    try:
        topology = snapshot_module.prepare_topology(bm, 0, TOLERANCE)
        assert bm.verts.layers.int.get(layer_names.VERT_HIDDEN_LAYER) is None
        assert bm.edges.layers.int.get(layer_names.EDGE_HIDDEN_LAYER) is None
        assert bm.faces.layers.int.get(layer_names.FACE_HIDDEN_LAYER) is None
        assert not stitch_pathedges.path_ring_includes_pre_hidden_edges(bm)
        snapshot = _types.SelectionSnapshot(False, False, False, [])
        selection.restore_visibility_and_selection(bm, topology.hidden_by_face_id, snapshot)
    finally:
        bm.free()


def _check_bulk_guard_fallbacks():
    for kind in ("counts", "coordinates", "faces"):
        bm = _build_deformed_grid(2)
        try:
            data = _BulkMeshData(bm)
            expected = snapshot_module.prepare_topology(bm, 0, TOLERANCE)
            expected_map = expected.mirror_face_ids
            if kind == "counts":
                data.vertices._values.append((9.0, 9.0, 9.0))
            elif kind == "coordinates":
                first = list(data.vertices._values[0])
                first[0] += 0.5
                data.vertices._values[0] = tuple(first)
            else:
                # Corrupt first-face loop vertex_index order (starts alone are
                # recomputed from loop_total and never trigger fallback).
                assert data._totals and data._totals[0] >= 2
                start = int(data._starts[0])
                data.loops._values[start], data.loops._values[start + 1] = (
                    data.loops._values[start + 1],
                    data.loops._values[start],
                )
            obj = _BulkMeshObject(data)
            topology = snapshot_module.prepare_topology(bm, 0, TOLERANCE, mesh_object=obj)
            assert obj.update_calls == 1
            assert topology.total_faces == len(bm.faces)
            assert topology.topology_resolution.vertex_count == len(bm.verts)
            assert topology.mirror_face_ids == expected_map
            expected_resolution = expected.topology_resolution
            actual_resolution = topology.topology_resolution
            assert numpy.array_equal(actual_resolution.coords64, expected_resolution.coords64)
            assert numpy.array_equal(actual_resolution.loop_verts, expected_resolution.loop_verts)
            assert numpy.array_equal(actual_resolution.loop_starts, expected_resolution.loop_starts)
            assert numpy.array_equal(actual_resolution.loop_totals, expected_resolution.loop_totals)
        finally:
            bm.free()


def _assert_selection_capture_equals(actual, expected):
    assert numpy.array_equal(actual.coords, expected.coords)
    assert numpy.array_equal(actual.selected_verts, expected.selected_verts)
    assert numpy.array_equal(actual.selected_edges, expected.selected_edges)
    assert numpy.array_equal(actual.selected_faces, expected.selected_faces)
    assert actual.history_indices == expected.history_indices
    assert len(actual.history_coords) == len(expected.history_coords)
    for left, right in zip(actual.history_coords, expected.history_coords, strict=True):
        assert tuple(float(value) for value in left) == tuple(float(value) for value in right)


def _check_selection_bulk_guard_fallbacks():
    """_capture_mesh_selection falls back to BMesh for each §L-3 integrity failure."""

    domains = ("VERT", "EDGE", "FACE")
    for kind in ("shape_keys", "counts", "coordinates", "faces"):
        bm = _build_deformed_grid(2)
        try:
            bm.verts.ensure_lookup_table()
            bm.edges.ensure_lookup_table()
            bm.faces.ensure_lookup_table()
            bm.verts.index_update()
            bm.edges.index_update()
            bm.faces.index_update()
            for index, vertex in enumerate(bm.verts):
                vertex.select = index % 3 == 0
            for index, edge in enumerate(bm.edges):
                edge.select = index % 4 == 0
            for index, face in enumerate(bm.faces):
                face.select = index % 2 == 0
            if bm.verts:
                bm.select_history.clear()
                bm.select_history.add(bm.verts[0])

            expected = snapshot_module.capture_selection_snapshot(bm, domains=domains, include_history=True)
            data = _BulkMeshData(bm)
            if kind == "shape_keys":
                data.shape_keys = object()
            elif kind == "counts":
                data.vertices._values.append((9.0, 9.0, 9.0))
            elif kind == "coordinates":
                first = list(data.vertices._values[0])
                first[0] += 0.5
                data.vertices._values[0] = tuple(first)
            else:
                assert data._totals and data._totals[0] >= 2
                start = int(data._starts[0])
                data.loops._values[start], data.loops._values[start + 1] = (
                    data.loops._values[start + 1],
                    data.loops._values[start],
                )
            obj = _BulkMeshObject(data)
            captured = snapshot_module.capture_selection_snapshot(
                bm,
                mesh_object=obj,
                domains=domains,
                include_history=True,
            )
            if kind == "shape_keys":
                assert obj.update_calls == 0
            else:
                assert obj.update_calls == 1
            _assert_selection_capture_equals(captured, expected)
        finally:
            bm.free()


def _check_nonfinite_one_sided_fallback():
    coords = numpy.asarray(((numpy.nan, 0.0, 0.0), (1.0, 0.0, 0.0)), dtype=numpy.float64)
    assert matching._one_sided_pair_table(coords, 0, TOLERANCE) is None
    empty = numpy.empty(0, dtype=numpy.int64)
    handle = snapshot_module.LazyTopologyResolution(
        coords,
        empty,
        empty,
        empty,
        numpy.zeros(2, dtype=bool),
        empty,
        empty,
        0,
        TOLERANCE,
        1,
    )

    def outcome(call):
        try:
            return "ok", call()
        except Exception as exc:
            return "error", type(exc)

    expected = outcome(
        lambda: matching.build_vertex_pair_table(tuple(Vector(tuple(row)) for row in coords.tolist()), 0, TOLERANCE)
    )
    actual = outcome(handle.resolve)
    assert actual[0] == expected[0]
    if actual[0] == "ok":
        assert actual[1].pairs == expected[1]


def _expected_mirror_candidate_arrays(coords, axis_index: int):
    lookup = matching.build_vertex_mirror_lookup(coords, axis_index, TOLERANCE)
    matrix = numpy.asarray([_vector_tuple(coordinate) for coordinate in coords], dtype=numpy.float64)
    on_plane = numpy.abs(matrix[:, axis_index]) <= TOLERANCE
    parts = []
    plane_indices = numpy.flatnonzero(on_plane)
    if len(plane_indices):
        arrays = lookup._batch_candidate_arrays(matrix[on_plane], True)
        assert arrays is not None
        queries, targets, distances = arrays
        parts.append((plane_indices[queries], targets, distances))
    off_indices = numpy.flatnonzero(~on_plane)
    if len(off_indices):
        mirrored = matrix[~on_plane].copy()
        mirrored[:, axis_index] = -mirrored[:, axis_index]
        arrays = lookup._batch_candidate_arrays(mirrored, False)
        assert arrays is not None
        queries, targets, distances = arrays
        parts.append((off_indices[queries], targets, distances))
    if not parts:
        empty_i = numpy.empty(0, dtype=numpy.int64)
        return empty_i, empty_i.copy(), numpy.empty(0, dtype=numpy.float64)
    queries = numpy.concatenate([part[0] for part in parts])
    targets = numpy.concatenate([part[1] for part in parts])
    distances = numpy.concatenate([part[2] for part in parts])
    order = numpy.lexsort((targets, distances, queries))
    return queries[order], targets[order], distances[order]


def _check_one_sided_candidate_arrays_contract():
    for axis_index in (matching.AXIS_INDEX["X"], matching.AXIS_INDEX["Y"], matching.AXIS_INDEX["Z"]):
        for seed in SEEDS:
            coords = _make_lookup_case(axis_index, seed)
            expected = _expected_mirror_candidate_arrays(coords, axis_index)
            actual = matching._one_sided_candidate_arrays(
                numpy.asarray([_vector_tuple(coordinate) for coordinate in coords], dtype=numpy.float64),
                axis_index,
                TOLERANCE,
            )
            assert actual is not None
            assert actual[0].tolist() == expected[0].tolist()
            assert actual[1].tolist() == expected[1].tolist()
            assert actual[2].tolist() == expected[2].tolist()


def _check_phase4_u2_dependency_direction():
    ast = __import__("ast")
    package_dir = PACKAGE_PARENT / "ydd_symmetric_edit"
    module_names = ("snapshot", "face_mapping", "matching", "layer_names")
    sources = {
        module_name: (package_dir / f"{module_name}.py").read_text(encoding="utf-8") for module_name in module_names
    }

    face_mapping_source = sources["face_mapping"]
    assert "from .snapshot import" not in face_mapping_source
    assert "import .snapshot" not in face_mapping_source

    for module_name, forbidden_imports in (
        ("face_mapping", {"snapshot"}),
        ("matching", {"snapshot", "face_mapping"}),
        ("layer_names", {"snapshot", "face_mapping"}),
    ):
        tree = ast.parse(sources[module_name])
        imported_modules = {
            node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module is not None
        }
        imported_modules.update(
            alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names
        )
        imported_modules.update(
            alias.name for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) for alias in node.names
        )
        assert forbidden_imports.isdisjoint(imported_modules)

    for module_name, source in sources.items():
        tree = ast.parse(source)
        for function in (node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))):
            delayed_imports = [node for node in ast.walk(function) if isinstance(node, (ast.Import, ast.ImportFrom))]
            assert not delayed_imports, f"{module_name}.{function.name} contains a delayed import"


def _candidate_multiset(arrays):
    assert arrays is not None
    queries, targets, distances = arrays
    return sorted(
        (int(query), int(target), float(distance))
        for query, target, distance in zip(queries, targets, distances, strict=True)
    )


def _check_vertex_registry_plane_split_oracle():
    tolerance = 0.125
    coords = numpy.asarray(
        (
            (1.0, 0.5, -0.5),
            (-1.0, 0.5, -0.5),
            (0.0, 0.0, 0.0),
            (tolerance, tolerance, 0.0),
            (-tolerance, 0.0, 0.0),
            (-2.0, 2.0, 0.0),
            (3.0, 1.0, 0.0),
            (-3.0 - tolerance, 1.0, 0.0),
            (2.0 * tolerance, 0.0, 0.0),
        ),
        dtype=numpy.float64,
    )
    registry = matching.VertexRegistry(coords, 0, tolerance)

    assert registry.on_plane_indices.tolist() == [2, 3, 4]
    assert _candidate_multiset(registry.candidates_on_plane(numpy.asarray((2,), dtype=numpy.int64))) == [
        (2, 2, 0.0),
        (2, 3, tolerance),
        (2, 4, tolerance),
    ]
    assert _candidate_multiset(registry.candidates_off_plane(numpy.asarray((0, 5, 6, 8), dtype=numpy.int64))) == [
        (0, 1, 0.0),
        (6, 7, tolerance),
    ]
    assert _candidate_multiset(registry.claimants_on_plane(numpy.asarray((3,), dtype=numpy.int64))) == [
        (2, 3, tolerance),
        (3, 3, 0.0),
    ]
    assert _candidate_multiset(registry.claimants_off_plane(numpy.asarray((1, 7), dtype=numpy.int64))) == [
        (0, 1, 0.0),
        (6, 7, tolerance),
    ]


def _check_vertex_registry_dense_candidates():
    tolerance = TOLERANCE
    positive = [(1.0 + 0.3 * tolerance * index, 0.0, 0.0) for index in range(4)]
    negative = [(-coordinate[0], coordinate[1], coordinate[2]) for coordinate in positive]
    registry = matching.VertexRegistry(numpy.asarray(positive + negative, dtype=numpy.float64), 0, tolerance)

    forward = _candidate_multiset(registry.candidates_off_plane(registry.positive_indices))
    assert len(forward) == 16
    assert {(query, target) for query, target, _distance in forward} == {
        (query, target) for query in range(4) for target in range(4, 8)
    }
    reverse = _candidate_multiset(registry.claimants_off_plane(registry.negative_indices))
    assert reverse == forward


def _check_vertex_registry_matches_one_sided_candidate_multiset():
    tolerance = 0.125
    coords = numpy.asarray(
        (
            (1.0, 0.0, 0.0),
            (1.0 + 0.25 * tolerance, 0.0, 0.0),
            (-1.0, 0.0, 0.0),
            (-1.0 - 0.25 * tolerance, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (tolerance, tolerance, 0.0),
            (-tolerance, 0.0, 0.0),
            (-4.0, 2.0, 0.0),
        ),
        dtype=numpy.float64,
    )
    for axis_index in range(3):
        axis_coords = numpy.roll(coords, axis_index, axis=1)
        registry = matching.VertexRegistry(axis_coords, axis_index, tolerance)
        parts = (
            registry.candidates_off_plane(registry.positive_indices),
            registry.claimants_off_plane(registry.positive_indices),
            registry.candidates_on_plane(registry.on_plane_indices),
        )
        candidate_rows = [row for arrays in parts for row in _candidate_multiset(arrays)]

        expected = matching._one_sided_candidate_arrays(axis_coords, axis_index, tolerance)
        assert sorted(candidate_rows) == _candidate_multiset(expected)


def _partial_resolution_handle(coords, tolerance=TOLERANCE):
    coords64 = numpy.asarray(coords, dtype=numpy.float64).reshape((-1, 3))
    empty = numpy.empty(0, dtype=numpy.int64)
    return snapshot_module.LazyTopologyResolution(
        coords64,
        empty,
        empty,
        empty,
        numpy.zeros(len(coords64), dtype=bool),
        empty,
        empty,
        0,
        tolerance,
        1,
    )


def _full_vertex_results(coords, tolerance=TOLERANCE):
    handle = _partial_resolution_handle(coords, tolerance)
    handle.resolve()
    return {vertex_id: handle._pairs.get(vertex_id) for vertex_id in range(handle.vertex_count)}


def _phase4_u3_coords():
    return numpy.asarray(
        DENSE_CANDIDATE_COORDS
        + (
            (3.0, 2.0, 0.0),
            (-3.0, 2.0, 0.0),
            (0.0, 4.0, 0.0),
            (0.0, 4.0 + 0.5 * TOLERANCE, 0.0),
            (7.0, 7.0, 0.0),
        ),
        dtype=numpy.float64,
    )


def _check_partial_vertex_resolution_matches_global():
    coords = _phase4_u3_coords()
    expected = _full_vertex_results(coords)

    rng = random.Random(830_021)
    random_ids = rng.sample(range(len(coords)), 7)
    random_handle = _partial_resolution_handle(coords)
    assert random_handle.resolve_vertices(random_ids) == {
        vertex_id: expected[vertex_id] for vertex_id in sorted(random_ids)
    }

    negative_handle = _partial_resolution_handle(coords)
    assert negative_handle.resolve_vertices((1,)) == {1: expected[1]}
    assert negative_handle._vertex_resolved[: len(DENSE_CANDIDATE_COORDS)].all()
    assert 0 in negative_handle._vertex_cache

    plane_handle = _partial_resolution_handle(coords)
    assert plane_handle.resolve_vertices((10,)) == {10: expected[10]}
    assert plane_handle._vertex_resolved[10:12].all()

    dense_handle = _partial_resolution_handle(coords)
    assert dense_handle.resolve_vertices((0,)) == {0: expected[0]}
    assert dense_handle._vertex_resolved[: len(DENSE_CANDIDATE_COORDS)].all()

    rejection_coords = numpy.asarray(
        ((1.0, 0.0, 0.0), (-1.0, 0.0, 0.0), (1.0 + 0.5 * TOLERANCE, 0.0, 0.0)),
        dtype=numpy.float64,
    )
    rejection_expected = _full_vertex_results(rejection_coords)
    assert rejection_expected == {0: None, 1: None, 2: None}
    rejection_handle = _partial_resolution_handle(rejection_coords)
    assert rejection_handle.resolve_vertices((1,)) == {1: None}
    assert rejection_handle._vertex_resolved.all()
    assert rejection_handle._vertex_cache == {0: None, 1: None, 2: None}


def _check_partial_vertex_resolution_sequence_and_negative_cache():
    coords = _phase4_u3_coords()
    expected = _full_vertex_results(coords)
    first = (1, 8)
    second = (10, 12)

    sequential = _partial_resolution_handle(coords)
    assert sequential.resolve_vertices(first) == {vertex_id: expected[vertex_id] for vertex_id in first}
    assert sequential.resolve_vertices(second) == {vertex_id: expected[vertex_id] for vertex_id in second}

    combined = _partial_resolution_handle(coords)
    union = tuple(sorted(set(first) | set(second)))
    assert combined.resolve_vertices(union) == {vertex_id: expected[vertex_id] for vertex_id in union}
    assert sequential._vertex_cache == combined._vertex_cache
    assert numpy.array_equal(sequential._vertex_resolved, combined._vertex_resolved)

    negative_count = sequential.partial_resolve_count
    registry = sequential._vertex_registry
    assert sequential.resolve_vertices((12,)) == {12: None}
    assert sequential.resolve_vertices((12,)) == {12: None}
    assert sequential.partial_resolve_count == negative_count
    assert sequential._vertex_registry is registry


def _check_partial_vertex_resolution_deepcopy_and_full_resolve():
    coords = _phase4_u3_coords()
    expected = _full_vertex_results(coords)
    original = _partial_resolution_handle(coords)
    original.resolve_vertices((1,))
    assert original.resolve_count == 0

    cloned = copy.deepcopy(original)
    assert original.resolve_count == cloned.resolve_count == 0
    assert cloned == original
    assert cloned._vertex_cache == original._vertex_cache
    assert cloned._vertex_cache is not original._vertex_cache
    assert cloned._vertex_resolved is not original._vertex_resolved
    assert cloned._vertex_registry is None or cloned._vertex_registry is not original._vertex_registry

    cloned.resolve_vertices((8,))
    assert cloned._vertex_resolved[8]
    assert not original._vertex_resolved[8]
    assert 8 not in original._vertex_cache

    original.resolve()
    assert original.resolve_count == 1
    assert original._pairs == {vertex_id: mirror for vertex_id, mirror in expected.items() if mirror is not None}
    assert original.resolve_vertices(range(len(coords))) == expected


def _check_partial_vertex_resolution_non_power_tolerance_boundary():
    tolerance = 1.0e-5
    source_axis = 21.0 * 2.0**-20
    target_axis = -source_axis + tolerance
    coords = numpy.asarray(
        (
            (source_axis, 0.25, -0.5),
            (target_axis, 0.25, -0.5),
            (0.0, 3.0, 0.0),
        ),
        dtype=numpy.float64,
    )
    assert abs(-coords[0, 0] - coords[1, 0]) == tolerance
    expected = _full_vertex_results(coords, tolerance)
    handle = _partial_resolution_handle(coords, tolerance)
    assert handle.resolve_vertices((0, 1, 2)) == expected


def _check_partial_vertex_resolution_registry_fallback_and_empty_guard():
    empty_registry = matching.VertexRegistry(numpy.empty((0, 3), dtype=numpy.float64), 0, TOLERANCE)
    empty_closure = empty_registry.resolve_closure(())
    assert empty_closure is not None
    assert len(empty_closure[0]) == 0 and empty_closure[1] == {}

    coords = numpy.asarray(((numpy.nan, 0.0, 0.0),), dtype=numpy.float64)
    invalid_registry = matching.VertexRegistry(coords, 0, TOLERANCE)
    assert invalid_registry.resolve_closure(()) is None
    full = _partial_resolution_handle(coords)
    partial = _partial_resolution_handle(coords)

    def outcome(call):
        try:
            return "ok", call()
        except Exception as exc:
            return "error", type(exc)

    expected = outcome(lambda: full.resolve()._pairs.get(0))
    actual = outcome(lambda: partial.resolve_vertices((0,))[0])
    assert actual == expected
    assert partial.resolve_count == 1


def _check_lazy_restore_state_matrix():
    bm = _build_deformed_grid(3)
    try:
        topology = snapshot_module.prepare_topology(bm, 0, TOLERANCE, 19)
        session = KnifeSession(
            window_pointer=1,
            area_pointer=2,
            region_pointer=3,
            object_name="object",
            mesh_name="mesh",
            axis_index=0,
            source_side="NEGATIVE",
            tolerance=TOLERANCE,
            mirror_face_ids={},
            hidden_by_face_id=topology.hidden_by_face_id,
            carrier_frames={},
            mesh_select_mode=MeshSelectionMode(True, False, False),
            started_at=1.0,
            history_token=19,
            topology_resolution=topology.topology_resolution,
        )
        obj = SimpleNamespace(mode="EDIT", data=SimpleNamespace())
        original_from_edit = operators.bmesh.from_edit_mesh
        setattr(operators.bmesh, "from_edit_mesh", lambda _data: bm)
        try:
            unresolved_materialize = snapshot_module.LazyTopologyResolution(
                topology.topology_resolution.coords64,
                topology.topology_resolution.loop_verts,
                topology.topology_resolution.loop_starts,
                topology.topology_resolution.loop_totals,
                topology.topology_resolution.hide_vertices,
                topology.topology_resolution.hide_edges,
                topology.topology_resolution.hide_faces,
                0,
                TOLERANCE,
                19,
            )
            try:
                unresolved_materialize.materialize(bm)
            except RuntimeError:
                pass
            else:
                raise AssertionError("materialize must reject unresolved handles")
            assert operators._restore_session_face_maps(session, obj)
            assert topology.topology_resolution.resolve_count == 1
            assert session.carrier_frames is topology.topology_resolution.carrier_frames
            assert session.mirror_face_ids == topology.mirror_face_ids
            assert copy.deepcopy(topology.topology_resolution) == topology.topology_resolution
            assert not any(
                isinstance(value, matching.VertexMirrorLookup)
                for value in topology.topology_resolution.__dict__.values()
            )
            assert not any(name in {"bm", "mesh_object", "callback"} for name in topology.topology_resolution.__dict__)
            mirror_layer = bm.faces.layers.int.get(layer_names.FACE_MIRROR_ID_LAYER)
            face_id_layer = bm.faces.layers.int.get(layer_names.FACE_ID_LAYER)
            token_layer = bm.faces.layers.int.get(layer_names.HISTORY_TOKEN_LAYER)
            assert mirror_layer is not None and face_id_layer is not None and token_layer is not None
            first_face = next(iter(bm.faces))
            for face in bm.faces:
                face[mirror_layer] = int(face[face_id_layer])
            assert operators._restore_session_face_maps(session, obj)
            assert topology.topology_resolution.resolve_count == 1
            assert int(first_face[mirror_layer]) == int(
                topology.mirror_face_ids[FaceId(int(first_face[face_id_layer]))]
            )
            first_face[mirror_layer] = 0
            last_face = list(bm.faces)[-1]
            if last_face is not first_face:
                last_face[token_layer] = 77
            assert operators._restore_session_face_maps(session, obj)
            assert topology.topology_resolution.resolve_count == 1
            assert int(first_face[mirror_layer]) == int(
                topology.mirror_face_ids[FaceId(int(first_face[face_id_layer]))]
            )
            no_handle = copy.copy(session)
            no_handle.topology_resolution = None
            for face in bm.faces:
                face[mirror_layer] = 0
            assert not operators._restore_session_face_maps(no_handle, obj)
            first_face[mirror_layer] = int(topology.mirror_face_ids.get(FaceId(1), FaceId(1)))
            assert not operators._restore_session_face_maps(no_handle, obj)
            for face in bm.faces:
                face[mirror_layer] = int(topology.mirror_face_ids.get(FaceId(int(face[face_id_layer])), FaceId(0)))
            assert operators._restore_session_face_maps(no_handle, obj)

            for face in bm.faces:
                face[token_layer] = 77
                face[mirror_layer] = 0
            foreign_handle = copy.copy(session)
            foreign_handle.topology_resolution = snapshot_module.LazyTopologyResolution(
                topology.topology_resolution.coords64,
                topology.topology_resolution.loop_verts,
                topology.topology_resolution.loop_starts,
                topology.topology_resolution.loop_totals,
                topology.topology_resolution.hide_vertices,
                topology.topology_resolution.hide_edges,
                topology.topology_resolution.hide_faces,
                0,
                TOLERANCE,
                19,
            )
            assert not operators._restore_session_face_maps(foreign_handle, obj)
            assert foreign_handle.topology_resolution.resolve_count == 0

            for face in bm.faces:
                face[token_layer] = 19
                face[mirror_layer] = 0
            original_face_id = int(first_face[face_id_layer])
            first_face[face_id_layer] = 999
            invalid_domain = copy.copy(session)
            invalid_domain.topology_resolution = snapshot_module.LazyTopologyResolution(
                topology.topology_resolution.coords64,
                topology.topology_resolution.loop_verts,
                topology.topology_resolution.loop_starts,
                topology.topology_resolution.loop_totals,
                topology.topology_resolution.hide_vertices,
                topology.topology_resolution.hide_edges,
                topology.topology_resolution.hide_faces,
                0,
                TOLERANCE,
                19,
            )
            assert not operators._restore_session_face_maps(invalid_domain, obj)
            assert invalid_domain.topology_resolution.resolve_count == 0
            first_face[face_id_layer] = original_face_id
            first_face[face_id_layer] = 0
            assert not operators._restore_session_face_maps(invalid_domain, obj)
            assert invalid_domain.topology_resolution.resolve_count == 0
            first_face[face_id_layer] = original_face_id
            for face in bm.faces:
                face_id = FaceId(int(face[face_id_layer]))
                face[mirror_layer] = int(topology.mirror_face_ids[face_id])
            extra_vertices = [
                bm.verts.new(coordinate) for coordinate in ((10.0, 0.0, 0.0), (11.0, 0.0, 0.0), (10.0, 1.0, 0.0))
            ]
            conflicting_face = bm.faces.new(extra_vertices)
            expected_mirror = int(topology.mirror_face_ids[FaceId(original_face_id)])
            conflicting_face[face_id_layer] = original_face_id
            conflicting_face[mirror_layer] = original_face_id if expected_mirror != original_face_id else 2
            conflicting_face[token_layer] = 19
            conflict = copy.copy(session)
            conflict.topology_resolution = None
            assert not operators._restore_session_face_maps(conflict, obj)
            bm.faces.layers.int.remove(face_id_layer)
            assert not operators._restore_session_face_maps(session, obj)
        finally:
            setattr(operators.bmesh, "from_edit_mesh", original_from_edit)

        unresolved = snapshot_module.LazyTopologyResolution(
            topology.topology_resolution.coords64,
            topology.topology_resolution.loop_verts,
            topology.topology_resolution.loop_starts,
            topology.topology_resolution.loop_totals,
            topology.topology_resolution.hide_vertices,
            topology.topology_resolution.hide_edges,
            topology.topology_resolution.hide_faces,
            0,
            TOLERANCE,
            19,
        )
        cloned = copy.deepcopy(unresolved)
        assert not unresolved._resolved and not cloned._resolved
        assert unresolved.resolve_count == cloned.resolve_count == 0
        assert unresolved == cloned
    finally:
        bm.free()


def _check_prepare_session_invoke_does_not_resolve():
    bm = _build_deformed_grid(2)
    try:
        prepared = snapshot_module.prepare_topology(bm, 0, TOLERANCE, 23)
        obj = SimpleNamespace(
            name="object",
            type="MESH",
            data=SimpleNamespace(name="mesh"),
            use_mesh_mirror_x=True,
            use_mesh_mirror_y=False,
            use_mesh_mirror_z=False,
        )
        context = SimpleNamespace(
            scene=SimpleNamespace(ydd_symmetric_edit=SimpleNamespace(source_side="NEGATIVE", tolerance=TOLERANCE)),
            edit_object=obj,
            window=SimpleNamespace(as_pointer=lambda: 11, modal_operators=()),
            area=SimpleNamespace(as_pointer=lambda: 12),
            region=SimpleNamespace(as_pointer=lambda: 13),
            tool_settings=SimpleNamespace(mesh_select_mode=(True, False, False)),
            preferences=SimpleNamespace(edit=SimpleNamespace(undo_steps=8)),
        )
        original_prepare = snapshot_module.prepare_topology
        original_from_edit = operators.bmesh.from_edit_mesh
        original_remove = snapshot_module.remove_temporary_layers
        original_update = operators.bmesh.update_edit_mesh
        original_cleanup = session_module.cleanup_session
        original_suspend = session_module._suspend_mesh_symmetry
        original_schedule = watcher_module._schedule_passthrough_watcher
        calls = []

        def spy_prepare(*args, **kwargs):
            calls.append(kwargs.get("mesh_object"))
            return prepared

        setattr(snapshot_module, "prepare_topology", spy_prepare)
        setattr(operators.bmesh, "from_edit_mesh", lambda _data: bm)
        setattr(snapshot_module, "remove_temporary_layers", lambda _bm: None)
        setattr(operators.bmesh, "update_edit_mesh", lambda *_args, **_kwargs: None)
        setattr(session_module, "cleanup_session", lambda *_args, **_kwargs: None)
        setattr(session_module, "_suspend_mesh_symmetry", lambda *_args, **_kwargs: None)
        setattr(watcher_module, "_schedule_passthrough_watcher", lambda *_args, **_kwargs: None)
        try:
            assert operators._prepare_session(context, lambda *_args: None)
            assert calls == [obj]
            session = operators._SESSIONS[11]
            assert session.topology_resolution is prepared.topology_resolution
            assert session.topology_resolution.resolve_count == 0
        finally:
            operators._SESSIONS.pop(11, None)
            operators.clear_history_records()
            setattr(snapshot_module, "prepare_topology", original_prepare)
            setattr(operators.bmesh, "from_edit_mesh", original_from_edit)
            setattr(snapshot_module, "remove_temporary_layers", original_remove)
            setattr(operators.bmesh, "update_edit_mesh", original_update)
            setattr(session_module, "cleanup_session", original_cleanup)
            setattr(session_module, "_suspend_mesh_symmetry", original_suspend)
            setattr(watcher_module, "_schedule_passthrough_watcher", original_schedule)
    finally:
        bm.free()


def _check_new_path_poll_count_short_circuit():
    """Unchanged element counts must skip the O(n) marker scan between polls."""

    class CountingSequence:
        def __init__(self, values):
            self.values = list(values)
            self.iterations = 0

        def __iter__(self):
            self.iterations += 1
            return iter(self.values)

        def __len__(self):
            return len(self.values)

    class FakeEdge:
        def __init__(self, marker):
            self.marker = marker

        def __getitem__(self, _layer):
            return self.marker

    edges = CountingSequence([FakeEdge(1), FakeEdge(2)])
    marker_layer = object()
    setattr(cast(Any, edges), "layers", SimpleNamespace(int=SimpleNamespace(get=lambda _name: marker_layer)))
    fake_bm = SimpleNamespace(verts=[object()] * 4, edges=edges, faces=[object()])
    fake_obj = SimpleNamespace(type="MESH", mode="EDIT", data=SimpleNamespace(name="mesh"))
    fake_bpy = SimpleNamespace(data=SimpleNamespace(objects=SimpleNamespace(get=lambda _name: fake_obj)))
    fake_bmesh = SimpleNamespace(from_edit_mesh=lambda _mesh: fake_bm)
    session = SimpleNamespace(object_name="obj", mesh_name="mesh", tool_kind="KNIFE", poll_element_counts=None)

    original_bpy = watcher_module.bpy
    original_bmesh = watcher_module.bmesh
    setattr(watcher_module, "bpy", fake_bpy)
    setattr(watcher_module, "bmesh", fake_bmesh)
    try:
        assert watcher_module._session_has_new_path(cast(Any, session)) is False
        assert edges.iterations == 1
        assert session.poll_element_counts == (4, 2, 1)
        # Same counts: the scan must not run again.
        assert watcher_module._session_has_new_path(cast(Any, session)) is False
        assert edges.iterations == 1
        # New geometry changes the counts and re-arms the scan.
        edges.values.append(FakeEdge(0))
        assert watcher_module._session_has_new_path(cast(Any, session)) is True
        assert edges.iterations == 2
        # A found path must stay found on the next tick (counts not stored).
        assert watcher_module._session_has_new_path(cast(Any, session)) is True
        assert edges.iterations == 3
    finally:
        setattr(watcher_module, "bpy", original_bpy)
        setattr(watcher_module, "bmesh", original_bmesh)


def _check_rip_invoke_does_not_resolve():
    """RIP invoke builds its snapshot without consuming full topology resolution."""

    bm = _build_deformed_grid(2)
    try:
        bm.verts.ensure_lookup_table()
        bm.verts.index_update()
        for vertex in bm.verts:
            vertex.select = False
        assert abs(float(bm.verts[0].co.x)) > TOLERANCE
        bm.verts[0].select = True
        obj = SimpleNamespace(
            name="rip-object",
            type="MESH",
            data=SimpleNamespace(name="rip-mesh"),
            use_mesh_mirror_x=True,
            use_mesh_mirror_y=False,
            use_mesh_mirror_z=False,
        )
        context = SimpleNamespace(
            scene=SimpleNamespace(ydd_symmetric_edit=SimpleNamespace(source_side="NEGATIVE", tolerance=TOLERANCE)),
            edit_object=obj,
            window=SimpleNamespace(as_pointer=lambda: 21, modal_operators=()),
            area=SimpleNamespace(as_pointer=lambda: 22),
            region=SimpleNamespace(as_pointer=lambda: 23),
            tool_settings=SimpleNamespace(
                mesh_select_mode=(True, False, False),
                use_proportional_edit=False,
                use_mesh_automerge=False,
            ),
            preferences=SimpleNamespace(edit=SimpleNamespace(undo_steps=8)),
        )
        original_prepare = snapshot_module.prepare_topology
        original_from_edit = session_module.bmesh.from_edit_mesh
        original_update = session_module.bmesh.update_edit_mesh
        original_cleanup = session_module.cleanup_session
        original_suspend = session_module._suspend_mesh_symmetry
        original_schedule = watcher_module._schedule_passthrough_watcher
        prepared = []

        def spy_prepare(*args, **kwargs):
            kwargs["mesh_object"] = None
            result = original_prepare(*args, **kwargs)
            prepared.append(result)
            return result

        setattr(snapshot_module, "prepare_topology", spy_prepare)
        setattr(session_module.bmesh, "from_edit_mesh", lambda _data: bm)
        setattr(session_module.bmesh, "update_edit_mesh", lambda *_args, **_kwargs: None)
        setattr(session_module, "cleanup_session", lambda *_args, **_kwargs: None)
        setattr(session_module, "_suspend_mesh_symmetry", lambda *_args, **_kwargs: None)
        setattr(watcher_module, "_schedule_passthrough_watcher", lambda *_args, **_kwargs: None)
        try:
            assert session_module._prepare_session(context, lambda *_args: None, tool_kind="RIP")
            assert len(prepared) == 1
            resolution = prepared[0].topology_resolution
            assert resolution.resolve_count == 0
            session = session_module.session_state._SESSIONS[21]
            assert session.rip is not None
            assert session.topology_resolution is resolution
        finally:
            session_module.session_state._SESSIONS.pop(21, None)
            operators.clear_history_records()
            setattr(snapshot_module, "prepare_topology", original_prepare)
            setattr(session_module.bmesh, "from_edit_mesh", original_from_edit)
            setattr(session_module.bmesh, "update_edit_mesh", original_update)
            setattr(session_module, "cleanup_session", original_cleanup)
            setattr(session_module, "_suspend_mesh_symmetry", original_suspend)
            setattr(watcher_module, "_schedule_passthrough_watcher", original_schedule)
    finally:
        bm.free()


def _face_resolution_handle(coords, faces, tolerance=TOLERANCE):
    coords64 = numpy.asarray(coords, dtype=numpy.float64).reshape((-1, 3))
    loop_totals = numpy.asarray([len(face) for face in faces], dtype=numpy.int64)
    loop_starts = (
        numpy.concatenate((numpy.asarray((0,), dtype=numpy.int64), numpy.cumsum(loop_totals[:-1], dtype=numpy.int64)))
        if len(loop_totals)
        else numpy.empty(0, dtype=numpy.int64)
    )
    loop_verts = numpy.asarray([vertex_id for face in faces for vertex_id in face], dtype=numpy.int64)
    empty = numpy.empty(0, dtype=numpy.int64)
    return snapshot_module.LazyTopologyResolution(
        coords64,
        loop_verts,
        loop_starts,
        loop_totals,
        numpy.zeros(len(coords64), dtype=bool),
        empty,
        numpy.zeros(len(faces), dtype=bool),
        0,
        tolerance,
        1,
    )


def _phase4_u4_grid(*, perturb=False, duplicate=False):
    coords = [(float(x), float(y), 0.0) for y in (-1, 0, 1) for x in (-2, -1, 0, 1, 2)]
    if perturb:
        coords[0] = (coords[0][0], coords[0][1] + 0.37, coords[0][2])
    faces = [
        (0, 1, 6, 5),
        (1, 2, 7, 6),
        (2, 3, 8, 7),
        (3, 4, 9, 8),
        (5, 6, 11, 10),
        (6, 7, 12, 11),
        (7, 8, 13, 12),
        (8, 9, 14, 13),
    ]
    if duplicate:
        faces.append(faces[0])
    return coords, faces


def _phase4_u4_duplicate_rows():
    coords = (
        (-2.0, -1.0, 0.0),
        (-2.0, 1.0, 0.0),
        (-1.0, 0.0, 0.0),
        (2.0, -1.0, 0.0),
        (2.0, 1.0, 0.0),
        (1.0, 0.0, 0.0),
    )
    faces = ((0, 1, 2), (3, 5, 4), (3, 5, 4))
    return coords, faces


def _full_face_results(coords, faces):
    handle = _face_resolution_handle(coords, faces)
    handle.resolve()
    return {
        FaceId(face_id): handle._mirror_face_ids.get(FaceId(face_id)) for face_id in range(1, handle.face_count + 1)
    }


def _restricted_face_results(expected, face_ids):
    return {FaceId(face_id): expected[FaceId(face_id)] for face_id in sorted(set(face_ids))}


def _check_face_registry_preserves_global_rows_and_lazy_geometry():
    coords, faces = _phase4_u4_grid(duplicate=True)
    handle = _face_resolution_handle(coords, faces)
    registry = face_mapping.FaceRegistry(
        handle.coords64,
        handle.loop_verts,
        handle.loop_starts,
        handle.loop_totals,
        handle.axis_index,
        handle.tolerance,
    )
    duplicate_key = registry.row_key(FaceId(1))
    assert registry.row_buckets[duplicate_key] == (FaceId(1), FaceId(9))
    assert set(registry.faces_for_vertex(0)) == {FaceId(1), FaceId(9)}
    assert not registry.geometry_ready
    assert registry.face_key_buckets
    assert not registry.centroid_geometry_ready
    assert registry.centroid_buckets
    assert registry.centroid_geometry_ready
    assert registry.geometry_ready

    lazy_centroid_registry = face_mapping.FaceRegistry(
        handle.coords64,
        handle.loop_verts,
        handle.loop_starts,
        handle.loop_totals,
        handle.axis_index,
        handle.tolerance,
    )
    assert not lazy_centroid_registry.geometry_ready
    assert lazy_centroid_registry.centroid_buckets
    assert lazy_centroid_registry.centroid_geometry_ready
    assert not lazy_centroid_registry.geometry_ready

    direct_registry = face_mapping.FaceRegistry(
        handle.coords64,
        handle.loop_verts,
        handle.loop_starts,
        handle.loop_totals,
        handle.axis_index,
        handle.tolerance,
    )
    direct_candidates = direct_registry.exact_geometry_candidates(
        tuple(FaceId(index) for index in range(1, len(faces) + 1))
    )
    direct_batch = direct_registry.exact_geometry_batch(tuple(FaceId(index) for index in range(1, len(faces) + 1)))
    assert direct_candidates is not None
    assert direct_batch is not None
    assert all(direct_candidates.values())
    expected_batch_targets = []
    for face_id in range(1, len(faces) + 1):
        candidates = direct_candidates[FaceId(face_id)]
        selected = candidates[0]
        if selected == FaceId(face_id) and abs(direct_registry.face_centroid(FaceId(face_id)).component(0)) > TOLERANCE:
            selected = next((candidate for candidate in candidates if candidate != FaceId(face_id)), selected)
        expected_batch_targets.append(int(selected))
    assert direct_batch.hits.all()
    assert direct_batch.targets.tolist() == expected_batch_targets
    assert direct_registry._coordinates is None
    assert direct_registry._face_key_buckets is None
    face_mapping._snapshot_face_map(
        handle.coords64,
        handle.loop_verts,
        handle.loop_starts,
        handle.loop_totals,
        handle.axis_index,
        handle.tolerance,
        vertex_pairs={},
        face_registry=direct_registry,
    )
    assert not direct_registry.centroid_geometry_ready
    assert direct_registry._coordinates is None


def _check_face_geometry_array_index_global_oracle():
    rng = random.Random(2_026_081_1)
    coords = []
    faces = []
    for _ in range(48):
        y = rng.uniform(-4.0, 4.0)
        z = rng.uniform(-4.0, 4.0)
        width = rng.uniform(0.1, 1.5)
        height = rng.uniform(0.1, 1.5)
        left = [
            (-rng.uniform(0.2, 3.0), y, z),
            (-rng.uniform(0.2, 3.0), y + height, z),
            (-rng.uniform(0.2, 3.0), y + height, z + width),
            (-rng.uniform(0.2, 3.0), y, z + width),
        ]
        right = [(-co[0] + rng.uniform(-0.2, 0.2) * TOLERANCE, co[1], co[2]) for co in left]
        left_ids = tuple(range(len(coords), len(coords) + len(left)))
        coords.extend(left)
        right_ids = tuple(range(len(coords), len(coords) + len(right)))
        coords.extend(right)
        faces.extend((left_ids, tuple(reversed(right_ids))))

    plane_start = len(coords)
    coords.extend(((-0.25, -0.5, 0.0), (0.25, -0.5, 0.0), (0.25, 0.5, 0.0), (-0.25, 0.5, 0.0)))
    faces.append(tuple(range(plane_start, plane_start + 4)))
    faces.append(faces[0])

    coords64 = numpy.asarray(coords, dtype=numpy.float64)
    loop_totals = numpy.asarray([len(face) for face in faces], dtype=numpy.int64)
    loop_starts = numpy.concatenate(
        (numpy.asarray((0,), dtype=numpy.int64), numpy.cumsum(loop_totals[:-1], dtype=numpy.int64))
    )
    loop_verts = numpy.asarray([vertex_id for face in faces for vertex_id in face], dtype=numpy.int64)
    registry = face_mapping.FaceRegistry(coords64, loop_verts, loop_starts, loop_totals, 0, TOLERANCE)
    actual = registry.exact_geometry_candidates(tuple(FaceId(index) for index in range(1, len(faces) + 1)))

    buckets = defaultdict(list)
    for face_index, face in enumerate(faces, start=1):
        key = (
            len(face),
            tuple(sorted(matching._quantized_coordinate(Vector(coords64[vertex_id]), TOLERANCE) for vertex_id in face)),
        )
        buckets[key].append(FaceId(face_index))
    expected = {}
    for face_index, face in enumerate(faces, start=1):
        mirrored = [list(coords64[vertex_id]) for vertex_id in face]
        for coordinate in mirrored:
            coordinate[0] *= -1.0
        key = (
            len(face),
            tuple(sorted(matching._quantized_coordinate(Vector(coordinate), TOLERANCE) for coordinate in mirrored)),
        )
        expected[FaceId(face_index)] = tuple(buckets[key])
    assert actual == expected

    invalid_coords = numpy.asarray(
        ((numpy.nan, 0.0, 0.0), (numpy.inf, 0.0, 0.0), (-numpy.inf, 1.0, 0.0)), dtype=numpy.float64
    )
    invalid_registry = face_mapping.FaceRegistry(
        invalid_coords,
        numpy.asarray((0, 1, 2), dtype=numpy.int64),
        numpy.asarray((0,), dtype=numpy.int64),
        numpy.asarray((3,), dtype=numpy.int64),
        0,
        TOLERANCE,
    )
    assert invalid_registry.exact_geometry_candidates((FaceId(1),)) is None

    overflow_coords = numpy.asarray(
        (
            (2**63 * TOLERANCE, 0.0, 0.0),
            (-(2**63) * TOLERANCE, 1.0, 0.0),
            (0.0, 2**63 * TOLERANCE, 1.0),
        ),
        dtype=numpy.float64,
    )
    overflow_registry = face_mapping.FaceRegistry(
        overflow_coords,
        numpy.asarray((0, 1, 2), dtype=numpy.int64),
        numpy.asarray((0,), dtype=numpy.int64),
        numpy.asarray((3,), dtype=numpy.int64),
        0,
        TOLERANCE,
    )
    assert overflow_registry.exact_geometry_candidates((FaceId(1),)) is None


def _check_face_geometry_snapshot_oracle_edges():
    boundary = float(numpy.nextafter(TOLERANCE, numpy.inf))
    rounded = float(numpy.float32(TOLERANCE))
    raw_coords = numpy.asarray(
        (
            (boundary, 1024.0, -0.0),
            (0.75, 1024.5, -0.5),
            (-0.4, 1025.0, 0.0),
            (boundary, 1024.0, -0.0),
            (0.75, 1024.5, -0.5),
            (-0.4, 1025.0, 0.0),
            (-rounded, 1024.0 + 2.0 * TOLERANCE, -0.0),
            (-0.75, 1024.5 + 2.0 * TOLERANCE, -0.5),
            (0.4, 1025.0 + 2.0 * TOLERANCE, 0.0),
        ),
        dtype=numpy.float64,
    )
    raw_loop_verts = numpy.asarray(
        (0, 1, 2, 3, 4, 5, 6, 7, 8),
        dtype=numpy.int64,
    )
    raw_loop_starts = numpy.asarray((0, 3, 6), dtype=numpy.int64)
    raw_loop_totals = numpy.asarray((3, 3, 3), dtype=numpy.int64)
    raw_actual = face_mapping._snapshot_face_map(
        raw_coords,
        raw_loop_verts,
        raw_loop_starts,
        raw_loop_totals,
        0,
        TOLERANCE,
        vertex_pairs={},
    )
    assert raw_actual == {FaceId(3): FaceId(1)}

    for axis_index in range(3):
        self_coords = numpy.asarray(((0.0, 0.0, 0.0), (0.3, 0.5, 0.7), (0.8, -0.2, 0.4)), dtype=numpy.float64)
        self_coords[:, axis_index] = 0.0
        self_registry = face_mapping.FaceRegistry(
            self_coords,
            numpy.asarray((0, 1, 2), dtype=numpy.int64),
            numpy.asarray((0,), dtype=numpy.int64),
            numpy.asarray((3,), dtype=numpy.int64),
            axis_index,
            TOLERANCE,
        )
        self_map = face_mapping._snapshot_face_map(
            self_coords,
            numpy.asarray((0, 1, 2), dtype=numpy.int64),
            numpy.asarray((0,), dtype=numpy.int64),
            numpy.asarray((3,), dtype=numpy.int64),
            axis_index,
            TOLERANCE,
            vertex_pairs={},
            face_registry=self_registry,
        )
        assert self_map == {FaceId(1): FaceId(1)}
        assert self_registry._coordinates is None

    centroid_order_x = numpy.asarray(
        (
            -0.32065773,
            74546.0078125,
            0.32065773,
            17491.48828125,
            -17491.48828125,
            -74546.0078125,
            2.5967834,
            -2.5967834,
        ),
        dtype=numpy.float64,
    )
    centroid_order_coords = numpy.column_stack(
        (centroid_order_x, numpy.zeros(len(centroid_order_x)), numpy.zeros(len(centroid_order_x)))
    )
    centroid_order_loops = numpy.tile(numpy.arange(len(centroid_order_x), dtype=numpy.int64), 2)
    centroid_order_starts = numpy.asarray((0, len(centroid_order_x)), dtype=numpy.int64)
    centroid_order_totals = numpy.asarray((len(centroid_order_x), len(centroid_order_x)), dtype=numpy.int64)
    centroid_order_registry = face_mapping.FaceRegistry(
        centroid_order_coords,
        centroid_order_loops,
        centroid_order_starts,
        centroid_order_totals,
        0,
        TOLERANCE,
    )
    centroid_order_batch = centroid_order_registry.exact_geometry_batch((FaceId(1), FaceId(2)))
    assert centroid_order_batch is not None
    assert centroid_order_batch.targets.tolist() == [2, 1]
    assert face_mapping._snapshot_face_map(
        centroid_order_coords,
        centroid_order_loops,
        centroid_order_starts,
        centroid_order_totals,
        0,
        TOLERANCE,
        vertex_pairs={},
        face_registry=centroid_order_registry,
    ) == {FaceId(1): FaceId(2), FaceId(2): FaceId(1)}

    empty = numpy.empty((0, 3), dtype=numpy.float64)
    empty_faces = numpy.empty(0, dtype=numpy.int64)
    assert face_mapping._snapshot_face_map(empty, empty_faces, empty_faces, empty_faces, 0, TOLERANCE) == {}


def _check_partial_face_resolution_duplicate_row_orderings():
    coords, faces = _phase4_u4_duplicate_rows()
    expected = _full_face_results(coords, faces)
    assert expected[FaceId(2)] is None
    assert expected[FaceId(3)] is None
    orderings = (((2,), (3,)), ((2, 3),), ((3,), (2,)))
    for batches in orderings:
        handle = _face_resolution_handle(coords, faces)
        seen = []
        for batch in batches:
            seen.extend(batch)
            assert handle.resolve_faces(batch) == _restricted_face_results(expected, batch)
        assert handle.resolve_faces(seen) == _restricted_face_results(expected, seen)


def _check_partial_face_resolution_randomized_sets():
    fixtures = (
        _phase4_u4_grid(),
        _phase4_u4_grid(perturb=True),
        _phase4_u4_grid(duplicate=True),
    )
    rng = random.Random(904_041)
    for coords, faces in fixtures:
        expected = _full_face_results(coords, faces)
        requested = rng.sample(range(1, len(faces) + 1), min(6, len(faces)))

        sequential = _face_resolution_handle(coords, faces)
        for face_id in requested:
            assert sequential.resolve_faces((face_id,)) == _restricted_face_results(expected, (face_id,))

        combined = _face_resolution_handle(coords, faces)
        assert combined.resolve_faces(requested) == _restricted_face_results(expected, requested)
        assert sequential.resolve_faces(requested) == combined.resolve_faces(requested)


def _check_partial_face_resolution_fallback_preserves_primary_values():
    coords, faces = _phase4_u4_grid(perturb=True)
    expected = _full_face_results(coords, faces)
    handle = _face_resolution_handle(coords, faces)

    primary_before = handle.resolve_faces((2,))
    assert primary_before == _restricted_face_results(expected, (2,))
    assert handle.resolve_count == 0
    assert handle.resolve_faces((1,)) == _restricted_face_results(expected, (1,))
    assert handle.resolve_count == 1
    assert handle.resolve_faces((2,)) == primary_before


def _check_face_primary_conflict_downgrades_every_claimant():
    coords = numpy.asarray(
        [(float(index), 0.0, 0.0) for index in range(9)],
        dtype=numpy.float64,
    )
    faces = ((0, 1, 2), (3, 4, 5), (6, 7, 8))
    handle = _face_resolution_handle(coords, faces)
    registry = face_mapping.FaceRegistry(
        handle.coords64,
        handle.loop_verts,
        handle.loop_starts,
        handle.loop_totals,
        handle.axis_index,
        handle.tolerance,
    )
    pairs = {
        0: 6,
        1: 7,
        2: 8,
        3: 6,
        4: 7,
        5: 8,
        6: 0,
        7: 1,
        8: 2,
    }
    closure, targets, needs_fallback = registry.resolve_primary_closure((FaceId(1),), pairs)
    assert closure == (FaceId(1), FaceId(2), FaceId(3))
    assert targets[FaceId(1)] is None
    assert targets[FaceId(2)] is None
    assert targets[FaceId(3)] == FaceId(1)
    assert needs_fallback


def _check_partial_face_resolution_counter_and_deepcopy():
    coords, faces = _phase4_u4_grid()
    expected = _full_face_results(coords, faces)
    handle = _face_resolution_handle(coords, faces)
    assert handle.resolve_faces((2,)) == _restricted_face_results(expected, (2,))
    assert handle.resolve_count == 0
    assert handle.partial_face_resolve_count == 1
    assert handle.resolve_faces((2,)) == _restricted_face_results(expected, (2,))
    assert handle.partial_face_resolve_count == 1

    cloned = copy.deepcopy(handle)
    assert cloned == handle
    assert cloned.resolve_count == handle.resolve_count == 0
    assert cloned._face_cache == handle._face_cache
    assert cloned._face_cache is not handle._face_cache
    assert cloned._face_resolved is not handle._face_resolved
    assert cloned._face_registry is None
    cloned.resolve_faces((5,))
    assert cloned._face_resolved[4]
    assert not handle._face_resolved[4]


def _check_partial_face_resolution_vertex_registry_fallback():
    coords = numpy.asarray(
        ((numpy.nan, 0.0, 0.0), (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0)),
        dtype=numpy.float64,
    )
    faces = ((0, 1, 2),)
    full = _face_resolution_handle(coords, faces)
    partial = _face_resolution_handle(coords, faces)

    def outcome(call):
        try:
            return "ok", call()
        except Exception as exc:
            return "error", type(exc)

    expected = outcome(lambda: full.resolve()._mirror_face_ids.get(FaceId(1)))
    actual = outcome(lambda: partial.resolve_faces((1,)).get(FaceId(1)))
    assert actual == expected
    assert partial.resolve_count == 1


def _check_u7_unselected_topology_bulk_capture():
    """U-7 topology arrays remain global when every selection column is empty."""

    from ydd_symmetric_edit import delete_dissolve, element_pairs

    bm = _build_deformed_grid(4)
    bulk = _clone_bmesh(bm)
    try:
        for source in (bm, bulk):
            for element in (*source.verts, *source.edges, *source.faces):
                element.select = False
        bulk_data = _BulkMeshData(bulk)
        mesh_object = _BulkMeshObject(bulk_data)
        compatibility = snapshot_module.capture_selection_snapshot(
            bm,
            domains=("EDGE", "FACE"),
            include_loops=True,
        )
        captured = snapshot_module.capture_selection_snapshot(
            bulk,
            mesh_object=mesh_object,
            domains=("EDGE", "FACE"),
            include_loops=True,
        )
        assert numpy.array_equal(captured.loop_verts, compatibility.loop_verts)
        assert numpy.array_equal(captured.loop_starts, compatibility.loop_starts)
        assert numpy.array_equal(captured.loop_totals, compatibility.loop_totals)
        assert len(captured.loop_starts) == len(bulk.faces)

        maps_compat = element_pairs.build_element_pair_maps(bm, 0, TOLERANCE)
        maps_bulk = element_pairs.build_element_pair_maps(
            bulk,
            0,
            TOLERANCE,
            mesh_object=mesh_object,
        )
        assert maps_bulk.vert_pairs == maps_compat.vert_pairs
        assert maps_bulk.edge_pair_by_index == maps_compat.edge_pair_by_index
        assert maps_bulk.face_pair_by_index == maps_compat.face_pair_by_index
        census_compat = delete_dissolve._symmetry_census(maps_compat, bm, TOLERANCE)
        census_bulk = delete_dissolve._symmetry_census(
            maps_bulk,
            bulk,
            TOLERANCE,
            mesh_object=mesh_object,
        )
        assert census_bulk == census_compat
    finally:
        bulk.free()
        bm.free()


def run():
    _check_vertex_lookup_equivalence()
    _check_batch_candidate_contract_edges()
    _check_extend_selection_matrix()
    _check_extend_selection_lazy_indices()
    _check_u6_scoped_selection_equivalence()
    _check_u6_face_duplicate_vertex_set_tie()
    _check_u6_registry_fallback()
    _check_extend_selection_wrapper_matrix()
    _check_selection_snapshot_bulk_equivalence()
    _check_topology_equivalence()
    _check_carrier_frame_lifecycle()
    _check_history_and_single_object_guard()
    _check_multi_object_native_passthrough()
    _check_rip_lookup_validation()
    _check_rip_scoped_vertex_ids()
    _check_rip_resolution_free_lookup_equivalence()
    _check_capture_resolve_contract()
    _check_rip_capture_omits_vertex_edge_hides()
    _check_bulk_edit_mesh_order_variants()
    _check_finish_scope_warning_matrix()
    _check_scoped_face_overlay_and_materialize()
    _check_finish_zero_match_decline()
    _check_hide_layer_omission_and_consumers()
    _check_bulk_guard_fallbacks()
    _check_selection_bulk_guard_fallbacks()
    _check_nonfinite_one_sided_fallback()
    _check_one_sided_candidate_arrays_contract()
    _check_phase4_u2_dependency_direction()
    _check_vertex_registry_plane_split_oracle()
    _check_vertex_registry_dense_candidates()
    _check_vertex_registry_matches_one_sided_candidate_multiset()
    _check_partial_vertex_resolution_matches_global()
    _check_partial_vertex_resolution_sequence_and_negative_cache()
    _check_partial_vertex_resolution_deepcopy_and_full_resolve()
    _check_partial_vertex_resolution_non_power_tolerance_boundary()
    _check_partial_vertex_resolution_registry_fallback_and_empty_guard()
    _check_face_registry_preserves_global_rows_and_lazy_geometry()
    _check_face_geometry_array_index_global_oracle()
    _check_face_geometry_snapshot_oracle_edges()
    _check_partial_face_resolution_duplicate_row_orderings()
    _check_partial_face_resolution_randomized_sets()
    _check_partial_face_resolution_fallback_preserves_primary_values()
    _check_face_primary_conflict_downgrades_every_claimant()
    _check_partial_face_resolution_counter_and_deepcopy()
    _check_partial_face_resolution_vertex_registry_fallback()
    _check_u7_unselected_topology_bulk_capture()
    _check_lazy_restore_state_matrix()
    _check_prepare_session_invoke_does_not_resolve()
    _check_rip_invoke_does_not_resolve()
    _check_new_path_poll_count_short_circuit()
    # Duplicate COMMITTED records and F9 ABSENT are exercised by the existing
    # GUI-only history and discriminator suites.
    print("YSE_PERF_EQUIV_OK", flush=True)


if __name__ == "__main__":
    try:
        run()
    except Exception:
        print("YSE_PERF_EQUIV_FAILED", flush=True)
        traceback.print_exc()
        sys.exit(1)
