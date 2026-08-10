# SPDX-License-Identifier: GPL-3.0-or-later

"""Headless equivalence checks for the large-mesh preparation changes."""

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
from mathutils import Vector

PACKAGE_PARENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_PARENT))

from ydd_symmetric_edit import core, operators  # noqa: E402
from ydd_symmetric_edit import rip as rip_module  # noqa: E402
from ydd_symmetric_edit._types import (  # noqa: E402
    Coordinate3D,
    FaceId,
    KnifeSession,
    MeshSelectionMode,
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
            self._bins[core._quantized_coordinate(Vector(coordinate), tolerance)].append((index, coordinate))
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
        for bin_key in core._iter_quantized_neighborhood(Vector(position), self._tolerance):
            for index, stored in self._bins.get(bin_key, ()):
                distance = core._chebyshev_distance_3d(position, stored)
                if distance <= self._tolerance:
                    candidates.append((distance, index))
        candidates.sort()
        return candidates

    def find(self, co):
        expected = core.mirror_coordinate(co, self._axis_index)
        expected = (float(expected[0]), float(expected[1]), float(expected[2]))
        best = None
        for bin_key in core._iter_quantized_neighborhood(Vector(expected), self._tolerance):
            for index, stored in self._bins.get(bin_key, ()):
                distance = core._chebyshev_distance_3d(expected, stored)
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
            (index, core.mirror_coordinate(co, self._axis_index))
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
    lookup = core.build_vertex_mirror_lookup(coords, axis_index, tolerance)
    assert not hasattr(lookup, "_bins")
    assert tuple(lookup._coords) == reference._coords
    assert core.build_vertex_pair_table(coords, axis_index, tolerance) == _pair_table(reference, coords)
    assert lookup.find_all_mirrored(coords) == tuple(reference.find_all_mirrored(coords))
    assert lookup.find_all_direct(coords) == tuple(reference.find_all_direct(coords))
    for query in coords:
        candidates = reference._candidates_for(core.mirror_coordinate(query, axis_index))
        if not candidates or len([distance for distance, _index in candidates if distance == candidates[0][0]]) > 1:
            continue
        assert lookup.find(query) == reference.find(query)


def _assert_bin_and_sphere_fixture(axis_index: int):
    source = _make_point(axis_index, -1.0, (0.25, -0.5), False)
    expected = core.mirror_coordinate(source, axis_index)
    inside = expected.copy()
    outside = expected.copy()
    # expected (=1.0) sits at the very top of its floor bin, so the straddling
    # candidates must sit above it to land in a different bin while staying
    # inside tol. Factors leave room for float32 rounding (~1.2% of tol here).
    inside[axis_index] += 0.99 * TOLERANCE
    outside[axis_index] += 1.03 * TOLERANCE
    expected_key = core._quantized_coordinate(expected, TOLERANCE)
    inside_key = core._quantized_coordinate(inside, TOLERANCE)
    outside_key = core._quantized_coordinate(outside, TOLERANCE)
    assert expected_key != inside_key and expected_key != outside_key
    assert 0.98 * TOLERANCE <= core._chebyshev_distance_3d(_vector_tuple(expected), _vector_tuple(inside)) <= TOLERANCE
    assert core._chebyshev_distance_3d(_vector_tuple(expected), _vector_tuple(outside)) > TOLERANCE
    reference = _ReferenceBinLookup((inside, outside), axis_index, TOLERANCE)
    lookup = core.build_vertex_mirror_lookup((inside, outside), axis_index, TOLERANCE)
    assert reference.find(source) == lookup.find(source) == 0
    assert reference.find_all_mirrored((source,)) == lookup.find_all_mirrored((source,)) == (0,)

    sphere_source = _make_point(axis_index, -4.0, (1.3, -1.1), False)
    sphere_expected = core.mirror_coordinate(sphere_source, axis_index)
    sphere_inside = sphere_expected.copy()
    sphere_outside = sphere_expected.copy()
    for coordinate in range(3):
        sphere_inside[coordinate] -= 0.95 * TOLERANCE
        sphere_outside[coordinate] -= 1.05 * TOLERANCE
    inside_distance = core._chebyshev_distance_3d(_vector_tuple(sphere_expected), _vector_tuple(sphere_inside))
    outside_distance = core._chebyshev_distance_3d(_vector_tuple(sphere_expected), _vector_tuple(sphere_outside))
    assert 0.9 * TOLERANCE <= inside_distance <= TOLERANCE
    assert TOLERANCE < outside_distance <= 1.1 * TOLERANCE
    reference = _ReferenceBinLookup((sphere_inside, sphere_outside), axis_index, TOLERANCE)
    lookup = core.build_vertex_mirror_lookup((sphere_inside, sphere_outside), axis_index, TOLERANCE)
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
        [index for index, right in enumerate(second) if core.coordinates_match(left, right, tolerance)]
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
    for axis_index in (core.AXIS_INDEX["X"], core.AXIS_INDEX["Y"], core.AXIS_INDEX["Z"]):
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
    empty_lookup = core.build_vertex_mirror_lookup([], 0, TOLERANCE)
    empty_query = Vector((1.0, 2.0, 3.0))
    assert empty_lookup.find(empty_query) is None
    assert empty_lookup.find_all_direct((empty_query,)) == (None,)
    assert empty_lookup.find_all_mirrored((empty_query,)) == (None,)

    plane_coords = [Vector((0.0, float(index), -float(index))) for index in range(5)]
    _assert_a_reference_case(plane_coords, 0, TOLERANCE)
    lookup = core.build_vertex_mirror_lookup(plane_coords, 0, TOLERANCE)
    assert lookup.find_all_mirrored(plane_coords) == tuple(range(5))

    tie_lookup = core.build_vertex_mirror_lookup(
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


def _assert_a_reference_case(coords, axis_index: int, tolerance: float):
    reference = _ReferenceBinLookup(coords, axis_index, tolerance)
    lookup = core.build_vertex_mirror_lookup(coords, axis_index, tolerance)
    expected_pairs = _pair_table(reference, coords)
    assert core.build_vertex_pair_table(coords, axis_index, tolerance) == expected_pairs
    old_mirrored = tuple(reference.find_all_mirrored(coords))
    new_mirrored = lookup.find_all_mirrored(coords)
    assert old_mirrored == new_mirrored
    old_direct = tuple(reference.find_all_direct(coords))
    new_direct = lookup.find_all_direct(coords)
    assert old_direct == new_direct
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
        key = core.FaceKey(
            vertex_count=len(coords),
            coordinates=tuple(
                sorted(core._quantized_coordinate(Vector(coordinate), tolerance) for coordinate in coords)
            ),
        )
        mirrored_coords = tuple(
            tuple(float(component) for component in core.mirror_coordinate(Vector(coordinate), axis_index))
            for coordinate in coords
        )
        mirrored_key = core.FaceKey(
            vertex_count=len(coords),
            coordinates=tuple(
                sorted(core._quantized_coordinate(Vector(coordinate), tolerance) for coordinate in mirrored_coords)
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
            faces_by_count_centroid[(len(record.coords), core._quantized_coordinate(centroid, tolerance))].append(
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
            mirrored_centroid = core.mirror_coordinate(Vector(record.centroid), axis_index)
            mirrored_coords = tuple(
                tuple(float(component) for component in core.mirror_coordinate(Vector(coordinate), axis_index))
                for coordinate in record.coords
            )
            candidates = []
            seen = set()
            found_self = False
            found_other = False
            for centroid_key in core._iter_quantized_neighborhood(mirrored_centroid, tolerance):
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


def _check_topology_equivalence():
    bm = _build_deformed_grid(7)
    original_face_key = core._face_key
    frame_name = (
        "_carrier_frame_from_coords" if hasattr(core, "_carrier_frame_from_coords") else "_carrier_frame_snapshot"
    )
    original_frame = getattr(core, frame_name)
    counters = {"key": 0, "frame": 0}

    def count_key(*args, **kwargs):
        counters["key"] += 1
        return original_face_key(*args, **kwargs)

    def count_frame(*args, **kwargs):
        counters["frame"] += 1
        return original_frame(*args, **kwargs)

    face_key_name = "_face_key"
    setattr(core, face_key_name, count_key)
    setattr(core, frame_name, count_frame)
    try:
        topology = core.prepare_topology(bm, 0, TOLERANCE)
    finally:
        setattr(core, face_key_name, original_face_key)
        setattr(core, frame_name, original_frame)
    assert topology.mirror_face_ids == _reference_eager_face_map(bm, 0, TOLERANCE)
    assert topology.matched_faces == topology.total_faces
    assert counters == {"key": 0, "frame": 0}, counters
    carrier_frames = cast(Any, topology.carrier_frames)
    assert not carrier_frames._cache
    bm.free()

    perturbed = _build_deformed_grid(7, asymmetric=True)
    perturbed.faces.ensure_lookup_table()
    perturbed.faces[0].hide = True
    original_face_key = core._face_key
    fallback_calls = 0

    def count_fallback_key(*args, **kwargs):
        nonlocal fallback_calls
        fallback_calls += 1
        return original_face_key(*args, **kwargs)

    try:
        face_key_name = "_face_key"
        setattr(core, face_key_name, count_fallback_key)
        topology = core.prepare_topology(perturbed, 0, TOLERANCE)
    finally:
        setattr(core, face_key_name, original_face_key)
    try:
        expected = _reference_eager_face_map(perturbed, 0, TOLERANCE)
        assert fallback_calls > 0
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
        topology = core.prepare_topology(asymmetric, 0, TOLERANCE)
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


def _reference_carrier_frame(vertices):
    if not vertices:
        zero = Coordinate3D(0.0, 0.0, 0.0)
        return core.CarrierFrameSnapshot(vertices, zero, None, None, 0.0)
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
    origin = core._coordinate_3d(origin_vector)
    if newell.length <= 1.0e-12:
        return core.CarrierFrameSnapshot(vertices, origin, None, None, 0.0)
    normal_vector = newell.normalized()
    basis_u = None
    for vertex in sorted(vertices):
        delta = Vector(vertex.as_tuple()) - origin_vector
        projected = delta - normal_vector * delta.dot(normal_vector)
        if projected.length > 1.0e-12:
            basis_u = projected.normalized()
            break
    if basis_u is None:
        return core.CarrierFrameSnapshot(vertices, origin, core._coordinate_3d(normal_vector), None, 0.0)
    deviation = max(abs((Vector(vertex.as_tuple()) - origin_vector).dot(normal_vector)) for vertex in vertices)
    return core.CarrierFrameSnapshot(
        vertices=vertices,
        origin=origin,
        normal=core._coordinate_3d(normal_vector),
        basis_u=core._coordinate_3d(basis_u),
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
    raw = cast(Any, {FaceId(index + 1): vertices for index, vertices in enumerate(raw_cases)})
    lazy_type_name = "LazyCarrierFrameMap"
    lazy_type = cast(Any, getattr(core, lazy_type_name))
    lazy = lazy_type(raw)
    assert lazy._cache == {}
    keys = tuple(lazy)
    assert len(lazy) == len(raw)
    assert keys == tuple(raw)
    assert FaceId(1) in lazy
    assert lazy._cache == {}
    same = lazy_type(dict(raw))
    different_raw = dict(raw)
    different_raw[FaceId(1)] = (Coordinate3D(99.0, 99.0, 99.0),)
    different = lazy_type(different_raw)
    assert lazy == same
    assert lazy != different
    assert lazy._cache == same._cache == different._cache == {}
    shared = copy.copy(lazy)
    assert shared._raw is lazy._raw
    assert shared._cache is lazy._cache
    cloned = copy.deepcopy(lazy)
    assert cloned._raw == lazy._raw
    assert cloned._raw is not lazy._raw
    assert cloned._cache == {}
    partial = lazy_type(dict(raw))
    partial.get(FaceId(1))
    partial_clone = copy.deepcopy(partial)
    assert tuple(partial._cache) == tuple(partial_clone._cache) == (FaceId(1),)
    assert len(partial_clone._cache) < len(raw)
    for face_id, vertices in raw.items():
        assert lazy.get(face_id) == _reference_carrier_frame(vertices)
    value = lazy.get(FaceId(2))
    assert lazy[FaceId(2)] is value
    assert lazy.get(FaceId(999)) is None
    assert len(lazy._cache) == len(raw)
    assert lazy[FaceId(6)] != lazy[FaceId(7)]
    cloned_value = cloned.get(FaceId(2))
    assert cloned_value == value
    assert cloned_value is not value


def _check_history_and_single_object_guard():
    raw = {FaceId(1): (Coordinate3D(0.0, 0.0, 0.0),)}
    lazy_type_name = "LazyCarrierFrameMap"
    lazy_type = cast(Any, getattr(core, lazy_type_name))
    carrier_frames = lazy_type(raw)
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
        history_token=operators._new_history_token(),
    )
    context = SimpleNamespace(preferences=SimpleNamespace(edit=SimpleNamespace(undo_steps=32)))
    try:
        operators._remember_history_session(session, context)
        record = operators._HISTORY_RECORDS[session.history_token]
        assert record.session.carrier_frames is carrier_frames
        deep_session = copy.deepcopy(record.session)
        assert deep_session.carrier_frames is not carrier_frames
        assert deep_session.carrier_frames == carrier_frames
        expected = carrier_frames.get(FaceId(1))
        assert deep_session.carrier_frames.get(FaceId(1)) == expected
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
        topology = core.prepare_topology(bm, 0, TOLERANCE, mark_vertex_ids=True)
        bm.verts.ensure_lookup_table()
        bm.verts[0].select = True
        coords = tuple(vertex.co.copy() for vertex in bm.verts)
        valid = topology.vertex_lookup
        variants = {
            "axis": core.build_vertex_mirror_lookup(coords, 1, TOLERANCE),
            "tolerance": core.build_vertex_mirror_lookup(coords, 0, TOLERANCE * 2.0),
            "count": core.build_vertex_mirror_lookup(coords[:-1], 0, TOLERANCE),
        }
        first_bad = core.build_vertex_mirror_lookup(coords, 0, TOLERANCE)
        first_bad._coords[0] = (first_bad._coords[0][0] + 1.0, *first_bad._coords[0][1:])
        variants["first"] = first_bad
        last_bad = core.build_vertex_mirror_lookup(coords, 0, TOLERANCE)
        last_bad._coords[-1] = (last_bad._coords[-1][0] + 1.0, *last_bad._coords[-1][1:])
        variants["last"] = last_bad

        original_match = rip_module._lookup_matches_mesh
        original_builder = core.build_vertex_mirror_lookup
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
        setattr(core, builder_name, count_rebuild)
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
            vertex_layer = bm.verts.layers.int.get(core.VERT_RIP_ID_LAYER)
            assert vertex_layer is not None
            bm.verts.layers.int.remove(vertex_layer)
            assert rip_module.build_snapshot(bm, 0, TOLERANCE, lookup=valid) is None
            assert validation_calls == 1 and rebuild_calls == 0

            topology = core.prepare_topology(bm, 0, TOLERANCE, mark_vertex_ids=True)
            for vertex in bm.verts:
                vertex.select = False
            validation_calls = 0
            rebuild_calls = 0
            assert rip_module.build_snapshot(bm, 0, TOLERANCE, lookup=topology.vertex_lookup) is None
            assert validation_calls == 1 and rebuild_calls == 0
        finally:
            setattr(rip_module, validation_name, original_match)
            setattr(core, builder_name, original_builder)
    finally:
        bm.free()


def run():
    _check_vertex_lookup_equivalence()
    _check_topology_equivalence()
    _check_carrier_frame_lifecycle()
    _check_history_and_single_object_guard()
    _check_multi_object_native_passthrough()
    _check_rip_lookup_validation()
    print("YSE_PERF_EQUIV_OK", flush=True)


if __name__ == "__main__":
    try:
        run()
    except Exception:
        print("YSE_PERF_EQUIV_FAILED", flush=True)
        traceback.print_exc()
        sys.exit(1)
