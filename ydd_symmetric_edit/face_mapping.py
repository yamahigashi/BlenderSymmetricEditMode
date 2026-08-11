from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import cast

import bmesh
import numpy  # type: ignore
from mathutils import Vector

from ._types import (
    CarrierFrameMap,
    CarrierFrameSnapshot,
    Coordinate3D,
    FaceId,
    FaceKey,
    MirrorFaceMap,
    QuantizedCoordinate,
)
from .layer_names import FACE_ID_LAYER
from .matching import (
    _coordinate_3d,
    _coords_match_chebyshev,
    _iter_quantized_neighborhood,
    _one_sided_pair_table,
    _quantized_coordinate,
    build_vertex_pair_table,
    coordinates_match,
    mirror_coordinate,
)


class _SnapshotVertex:
    def __init__(self, coordinate) -> None:
        self.co = Vector(coordinate)


class _SnapshotFace:
    def __init__(self, values) -> None:
        self.verts = tuple(_SnapshotVertex(value) for value in values)


class _FaceRowGroup:
    def __init__(
        self,
        unique_keys: numpy.ndarray,
        counts: numpy.ndarray,
        member_starts: numpy.ndarray,
        members: numpy.ndarray,
        first_face_ids: numpy.ndarray,
    ) -> None:
        self.unique_keys = unique_keys
        self.counts = counts
        self.member_starts = member_starts
        self.members = members
        self.first_face_ids = first_face_ids


class FaceRegistry:
    """Resolution-free face topology and geometry indices."""

    def __init__(
        self,
        coords64: numpy.ndarray,
        loop_verts: numpy.ndarray,
        loop_starts: numpy.ndarray,
        loop_totals: numpy.ndarray,
        axis_index: int,
        tolerance: float,
    ) -> None:
        self.coords64 = numpy.asarray(coords64, dtype=numpy.float64)
        self.loop_verts = numpy.asarray(loop_verts, dtype=numpy.int64)
        self.loop_starts = numpy.asarray(loop_starts, dtype=numpy.int64)
        self.loop_totals = numpy.asarray(loop_totals, dtype=numpy.int64)
        self.axis_index = int(axis_index)
        self.tolerance = float(tolerance)
        self.face_count = int(len(self.loop_starts))
        self._row_groups, self._row_unique_indices = self._build_row_index()
        self._row_buckets: dict[tuple[int, tuple[int, ...]], tuple[FaceId, ...]] | None = None
        self._vertex_face_ids, self._vertex_face_starts = self._build_vertex_face_index()
        self._coordinates: tuple[tuple[float, float, float], ...] | None = None
        self._face_key_buckets: dict[FaceKey, tuple[FaceId, ...]] | None = None
        self._centroid_buckets: dict[tuple[int, QuantizedCoordinate], tuple[FaceId, ...]] | None = None
        self._centroids: tuple[Coordinate3D, ...] | None = None

    def _build_row_index(self):
        groups: dict[int, _FaceRowGroup] = {}
        row_unique_indices = numpy.empty(self.face_count, dtype=numpy.int64)
        for raw_total in numpy.unique(self.loop_totals):
            total = int(raw_total)
            face_indices = numpy.flatnonzero(self.loop_totals == total)
            if total == 0:
                inverse = numpy.zeros(len(face_indices), dtype=numpy.int64)
                counts = numpy.asarray((len(face_indices),), dtype=numpy.int64)
                unique_keys = numpy.empty(0, dtype=numpy.void)
                first_rows = numpy.asarray((0,), dtype=numpy.int64)
            else:
                positions = self.loop_starts[face_indices, None] + numpy.arange(total, dtype=numpy.int64)
                rows = numpy.sort(self.loop_verts[positions], axis=1)
                row_dtype = numpy.dtype((numpy.void, rows.dtype.itemsize * total))
                row_keys = numpy.ascontiguousarray(rows).view(row_dtype).reshape(-1)
                unique_keys, first_rows, inverse, counts = numpy.unique(
                    row_keys,
                    return_index=True,
                    return_inverse=True,
                    return_counts=True,
                )
            member_order = numpy.argsort(inverse, kind="stable")
            members = face_indices[member_order] + 1
            member_starts = numpy.concatenate(
                (numpy.asarray((0,), dtype=numpy.int64), numpy.cumsum(counts, dtype=numpy.int64))
            )
            first_face_ids = face_indices[first_rows] + 1
            groups[total] = _FaceRowGroup(unique_keys, counts, member_starts, members, first_face_ids)
            row_unique_indices[face_indices] = inverse
        return groups, row_unique_indices

    def _build_vertex_face_index(self):
        vertex_count = len(self.coords64)
        if len(self.loop_verts) == 0:
            return numpy.empty(0, dtype=numpy.int64), numpy.zeros(vertex_count + 1, dtype=numpy.int64)
        loop_face_ids = numpy.repeat(
            numpy.arange(1, self.face_count + 1, dtype=numpy.int64),
            self.loop_totals,
        )
        order = numpy.lexsort((loop_face_ids, self.loop_verts))
        vertices = self.loop_verts[order]
        face_ids = loop_face_ids[order]
        keep = numpy.ones(len(vertices), dtype=bool)
        keep[1:] = (vertices[1:] != vertices[:-1]) | (face_ids[1:] != face_ids[:-1])
        vertices = vertices[keep]
        face_ids = face_ids[keep]
        if len(vertices) and (numpy.any(vertices < 0) or numpy.any(vertices >= vertex_count)):
            raise IndexError("face loop vertex index out of range")
        counts = numpy.bincount(vertices, minlength=vertex_count)
        starts = numpy.concatenate((numpy.asarray((0,), dtype=numpy.int64), numpy.cumsum(counts, dtype=numpy.int64)))
        return face_ids, starts

    def row_key(self, face_id: FaceId):
        face_index = int(face_id) - 1
        start = int(self.loop_starts[face_index])
        total = int(self.loop_totals[face_index])
        vertices = numpy.sort(self.loop_verts[start : start + total])
        return total, tuple(int(vertex_id) for vertex_id in vertices.tolist())

    def vertices(self, face_id: FaceId):
        face_index = int(face_id) - 1
        start = int(self.loop_starts[face_index])
        total = int(self.loop_totals[face_index])
        return tuple(int(vertex_id) for vertex_id in self.loop_verts[start : start + total].tolist())

    def vertices_for_faces(self, face_ids: Sequence[FaceId]) -> numpy.ndarray:
        """Return the unique vertex ids for a face sequence in sorted order."""

        indices = numpy.fromiter((int(face_id) - 1 for face_id in face_ids), dtype=numpy.int64, count=len(face_ids))
        if not len(indices):
            return numpy.empty(0, dtype=numpy.int64)
        starts = self.loop_starts[indices]
        totals = self.loop_totals[indices]
        output_starts = numpy.cumsum(totals, dtype=numpy.int64) - totals
        loop_count = int(totals.sum())
        positions = numpy.arange(loop_count, dtype=numpy.int64) + numpy.repeat(starts - output_starts, totals)
        return numpy.unique(self.loop_verts[positions])

    def _row_bucket(self, face_id: FaceId) -> tuple[FaceId, ...]:
        face_index = int(face_id) - 1
        total = int(self.loop_totals[face_index])
        group = self._row_groups[total]
        unique_index = int(self._row_unique_indices[face_index])
        start = int(group.member_starts[unique_index])
        end = int(group.member_starts[unique_index + 1])
        return tuple(FaceId(int(member)) for member in group.members[start:end].tolist())

    @property
    def row_buckets(self):
        if self._row_buckets is None:
            buckets: dict[tuple[int, tuple[int, ...]], tuple[FaceId, ...]] = {}
            for total, group in self._row_groups.items():
                if total == 0:
                    rows = ((),)
                else:
                    rows = group.unique_keys.view(numpy.int64).reshape(-1, total).tolist()
                for unique_index, row in enumerate(rows):
                    start = int(group.member_starts[unique_index])
                    end = int(group.member_starts[unique_index + 1])
                    buckets[(total, tuple(int(vertex_id) for vertex_id in row))] = tuple(
                        FaceId(int(member)) for member in group.members[start:end].tolist()
                    )
            self._row_buckets = buckets
        return self._row_buckets

    def faces_for_vertex(self, vertex_id: int):
        if vertex_id < 0 or vertex_id >= len(self.coords64):
            raise IndexError("vertex query index out of range")
        start = int(self._vertex_face_starts[vertex_id])
        end = int(self._vertex_face_starts[vertex_id + 1])
        return tuple(FaceId(int(face_id)) for face_id in self._vertex_face_ids[start:end].tolist())

    def primary_target(self, face_id: FaceId, pairs: Mapping[int, int]) -> FaceId | None:
        vertices = self.vertices(face_id)
        if not vertices:
            return None
        face_index = int(face_id) - 1
        total = int(self.loop_totals[face_index])
        group = self._row_groups[total]
        own_unique_index = int(self._row_unique_indices[face_index])
        if int(group.counts[own_unique_index]) != 1 or any(vertex_id not in pairs for vertex_id in vertices):
            return None
        mapped = numpy.fromiter((pairs[vertex_id] for vertex_id in vertices), dtype=numpy.int64, count=total)
        mapped.sort()
        mapped_key = mapped.view(group.unique_keys.dtype).reshape(-1)[0]
        destination = int(numpy.searchsorted(group.unique_keys, mapped_key))
        if (
            destination >= len(group.unique_keys)
            or group.unique_keys[destination] != mapped_key
            or int(group.counts[destination]) != 1
        ):
            return None
        return FaceId(int(group.first_face_ids[destination]))

    def _exact_claimants(
        self,
        target_id: FaceId,
        pairs: Mapping[int, int],
        pair_sources: Mapping[int, tuple[int, ...]],
        memo: dict[FaceId, FaceId | None],
    ) -> set[FaceId]:
        claimants: set[FaceId] = set()
        for target_vertex in self.vertices(target_id):
            for source_vertex in pair_sources.get(target_vertex, ()):
                for face_id in self.faces_for_vertex(source_vertex):
                    if face_id not in claimants and self._memoized_target(face_id, pairs, memo) == target_id:
                        claimants.add(face_id)
        return claimants

    def _memoized_target(
        self,
        face_id: FaceId,
        pairs: Mapping[int, int],
        memo: dict[FaceId, FaceId | None],
    ) -> FaceId | None:
        if face_id in memo:
            return memo[face_id]
        target = self.primary_target(face_id, pairs)
        memo[face_id] = target
        return target

    def resolve_primary_closure(self, face_ids, pairs: Mapping[int, int]):
        """Resolve exact row-key closure, reporting whether fallback is required."""

        closure = {FaceId(int(face_id)) for face_id in face_ids}
        if any(int(face_id) < 1 or int(face_id) > self.face_count for face_id in closure):
            raise IndexError("face query index out of range")
        pair_source_lists: dict[int, list[int]] = defaultdict(list)
        for source, target in pairs.items():
            pair_source_lists[target].append(source)
        pair_sources = {target: tuple(sources) for target, sources in pair_source_lists.items()}
        # Memo is valid only within one call: primary_target is pure in
        # (face_id, pairs) and pairs is fixed for the whole closure pass.
        memo: dict[FaceId, FaceId | None] = {}
        targets: dict[FaceId, FaceId | None] = {}
        frontier = set(closure)
        additions: set[FaceId] = set()
        while frontier:
            additions.clear()
            for face_id in frontier:
                additions.update(self._row_bucket(face_id))
                target = self._memoized_target(face_id, pairs, memo)
                if target is None:
                    closure.update(additions)
                    return tuple(sorted(closure)), targets, True
                targets[face_id] = target
                additions.update(self._row_bucket(target))
                additions.update(self._exact_claimants(target, pairs, pair_sources, memo))
            additions.difference_update(closure)
            closure.update(additions)
            frontier, additions = additions, frontier

        target_counts: dict[FaceId, int] = defaultdict(int)
        for target in targets.values():
            if target is not None:
                target_counts[target] += 1
        conflicts = {target for target, count in target_counts.items() if count > 1}
        if conflicts:
            targets = {face_id: None if target in conflicts else target for face_id, target in targets.items()}
        return tuple(sorted(closure)), targets, bool(conflicts)

    @property
    def geometry_ready(self) -> bool:
        return self._face_key_buckets is not None

    @property
    def centroid_geometry_ready(self) -> bool:
        return self._centroid_buckets is not None

    def _ensure_key_index(self) -> None:
        if self.geometry_ready:
            return
        coordinates = tuple((float(row[0]), float(row[1]), float(row[2])) for row in self.coords64.tolist())
        key_buckets: dict[FaceKey, list[FaceId]] = defaultdict(list)

        inverse_tolerance = 1.0 / max(self.tolerance, 1.0e-12)
        scaled_coordinates = self.coords64 * inverse_tolerance
        int64_info = numpy.iinfo(numpy.int64)
        use_array_keys = bool(
            numpy.isfinite(scaled_coordinates).all()
            and numpy.all(scaled_coordinates >= int64_info.min)
            and numpy.all(scaled_coordinates <= int64_info.max)
        )
        if use_array_keys:
            quantized_coordinates = numpy.floor(scaled_coordinates).astype(numpy.int64)
            for raw_total in numpy.unique(self.loop_totals):
                total = int(raw_total)
                face_indices = numpy.flatnonzero(self.loop_totals == total)
                if total == 0:
                    key_buckets[FaceKey(vertex_count=0, coordinates=())].extend(
                        FaceId(int(face_index) + 1) for face_index in face_indices.tolist()
                    )
                    continue

                positions = self.loop_starts[face_indices, None] + numpy.arange(total, dtype=numpy.int64)
                rows = quantized_coordinates[self.loop_verts[positions]]
                row_order = numpy.lexsort((rows[:, :, 2], rows[:, :, 1], rows[:, :, 0]), axis=1)
                rows = numpy.take_along_axis(rows, row_order[:, :, None], axis=1)
                row_dtype = numpy.dtype((numpy.void, rows.dtype.itemsize * total * 3))
                flat_rows = numpy.ascontiguousarray(rows).reshape(len(rows), total * 3)
                row_keys = flat_rows.view(row_dtype).reshape(-1)
                unique_keys, _first_rows, inverse, counts = numpy.unique(
                    row_keys,
                    return_index=True,
                    return_inverse=True,
                    return_counts=True,
                )
                member_order = numpy.argsort(inverse, kind="stable")
                member_starts = numpy.concatenate(
                    (numpy.asarray((0,), dtype=numpy.int64), numpy.cumsum(counts, dtype=numpy.int64))
                )
                unique_rows = unique_keys.view(numpy.int64).reshape(-1, total, 3)
                for unique_index, row in enumerate(unique_rows):
                    key = FaceKey(
                        vertex_count=total,
                        coordinates=tuple(
                            QuantizedCoordinate(int(vertex[0]), int(vertex[1]), int(vertex[2]))
                            for vertex in row.tolist()
                        ),
                    )
                    start = int(member_starts[unique_index])
                    end = int(member_starts[unique_index + 1])
                    key_buckets[key].extend(
                        FaceId(int(face_index) + 1) for face_index in face_indices[member_order[start:end]].tolist()
                    )
        else:
            for face_index in range(1, self.face_count + 1):
                face_id = FaceId(face_index)
                vertex_ids = self.vertices(face_id)
                values = tuple(coordinates[vertex_id] for vertex_id in vertex_ids)
                snapshot_face = cast(bmesh.types.BMFace, _SnapshotFace(values))
                key_buckets[_face_key(snapshot_face, self.axis_index, self.tolerance, mirrored=False)].append(face_id)
        self._coordinates = coordinates
        self._face_key_buckets = {key: tuple(face_ids) for key, face_ids in key_buckets.items()}

    def _ensure_centroid_index(self) -> None:
        if self.centroid_geometry_ready:
            return
        self._ensure_key_index()
        centroid_buckets: dict[tuple[int, QuantizedCoordinate], list[FaceId]] = defaultdict(list)
        centroids: list[Coordinate3D] = []
        for face_index in range(1, self.face_count + 1):
            face_id = FaceId(face_index)
            vertex_ids = self.vertices(face_id)
            centroid = self.face_centroid(face_id)
            centroids.append(centroid)
            centroid_buckets[
                (len(vertex_ids), _quantized_coordinate(Vector(centroid.as_tuple()), self.tolerance))
            ].append(face_id)
        self._centroid_buckets = {key: tuple(face_ids) for key, face_ids in centroid_buckets.items()}
        self._centroids = tuple(centroids)

    @property
    def face_key_buckets(self):
        self._ensure_key_index()
        return cast(dict[FaceKey, tuple[FaceId, ...]], self._face_key_buckets)

    @property
    def centroid_buckets(self):
        self._ensure_centroid_index()
        return cast(
            dict[tuple[int, QuantizedCoordinate], tuple[FaceId, ...]],
            self._centroid_buckets,
        )

    def face_coordinates(self, face_id: FaceId, *, mirrored: bool = False):
        self._ensure_key_index()
        coordinates = cast(tuple[tuple[float, float, float], ...], self._coordinates)
        values = tuple(coordinates[vertex_id] for vertex_id in self.vertices(face_id))
        if not mirrored:
            return values
        return tuple(
            tuple(-value if axis == self.axis_index else value for axis, value in enumerate(coordinate))
            for coordinate in values
        )

    def mirrored_face_key(self, face_id: FaceId):
        values = self.face_coordinates(face_id)
        snapshot_face = cast(bmesh.types.BMFace, _SnapshotFace(values))
        return _face_key(snapshot_face, self.axis_index, self.tolerance, mirrored=True)

    def face_centroid(self, face_id: FaceId):
        """Compute one centroid without materializing the fallback index."""
        if self.centroid_geometry_ready:
            return cast(tuple[Coordinate3D, ...], self._centroids)[int(face_id) - 1]
        self._ensure_key_index()
        values = self.face_coordinates(face_id)
        center = Vector((0.0, 0.0, 0.0))
        for value in values:
            center += Vector(value)
        if values:
            center /= len(values)
        return _coordinate_3d(center)

    def centroid(self, face_id: FaceId):
        self._ensure_centroid_index()
        return cast(tuple[Coordinate3D, ...], self._centroids)[int(face_id) - 1]


def _snapshot_face_map(
    coords64: numpy.ndarray,
    loop_verts: numpy.ndarray,
    loop_starts: numpy.ndarray,
    loop_totals: numpy.ndarray,
    axis_index: int,
    tolerance: float,
    vertex_pairs: dict[int, int] | None = None,
    face_registry: FaceRegistry | None = None,
):
    vertex_pairs = _one_sided_pair_table(coords64, axis_index, tolerance) if vertex_pairs is None else vertex_pairs
    if vertex_pairs is None:
        vertex_pairs = build_vertex_pair_table(
            [Vector(tuple(float(value) for value in row)) for row in coords64.tolist()], axis_index, tolerance
        )
    face_count = len(loop_starts)
    pair_targets = numpy.full(len(coords64), -1, dtype=numpy.int64)
    if vertex_pairs:
        sources = numpy.fromiter(vertex_pairs, dtype=numpy.int64, count=len(vertex_pairs))
        pair_targets[sources] = numpy.fromiter(vertex_pairs.values(), dtype=numpy.int64, count=len(vertex_pairs))
    mirror_indices = numpy.full(face_count, -1, dtype=numpy.int64)
    for raw_total in numpy.unique(loop_totals):
        total = int(raw_total)
        face_indices = numpy.flatnonzero(loop_totals == total)
        if total == 0:
            continue
        positions = loop_starts[face_indices, None] + numpy.arange(total, dtype=numpy.int64)
        original = numpy.sort(loop_verts[positions], axis=1)
        mapped = pair_targets[loop_verts[positions]]
        mapped_valid = (mapped >= 0).all(axis=1)
        mapped.sort(axis=1)
        row_dtype = numpy.dtype((numpy.void, original.dtype.itemsize * total))
        original_keys = numpy.ascontiguousarray(original).view(row_dtype).reshape(-1)
        mapped_keys = numpy.ascontiguousarray(mapped).view(row_dtype).reshape(-1)
        unique_keys, first_rows, inverse, counts = numpy.unique(
            original_keys,
            return_index=True,
            return_inverse=True,
            return_counts=True,
        )
        destinations = numpy.searchsorted(unique_keys, mapped_keys)
        in_range = destinations < len(unique_keys)
        found = numpy.zeros(len(face_indices), dtype=bool)
        found[in_range] = unique_keys[destinations[in_range]] == mapped_keys[in_range]
        exact = mapped_valid & found & (counts[inverse] == 1)
        exact_indices = numpy.flatnonzero(exact)
        if len(exact_indices):
            exact_destinations = destinations[exact_indices]
            unique_destinations = counts[exact_destinations] == 1
            exact_indices = exact_indices[unique_destinations]
            mirror_indices[face_indices[exact_indices]] = face_indices[first_rows[destinations[exact_indices]]]

    mirror_face_ids: MirrorFaceMap = {
        FaceId(face_index + 1): FaceId(int(counterpart) + 1)
        for face_index, counterpart in enumerate(mirror_indices.tolist())
        if counterpart >= 0
    }
    fallback = [FaceId(index + 1) for index in numpy.flatnonzero(mirror_indices < 0).tolist()]
    if not fallback:
        return mirror_face_ids

    if face_registry is None:
        coordinates = tuple(tuple(float(value) for value in row) for row in coords64.tolist())
        face_vertex_ids = {
            FaceId(face_index + 1): tuple(
                int(vertex_id) for vertex_id in loop_verts[int(start) : int(start) + int(total)]
            )
            for face_index, (start, total) in enumerate(zip(loop_starts, loop_totals, strict=True))
        }

        def face_coords(face_id: FaceId, mirrored: bool = False):
            values = tuple(coordinates[index] for index in face_vertex_ids[face_id])
            if mirrored:
                values = tuple(
                    tuple(-value if axis == axis_index else value for axis, value in enumerate(coordinate))
                    for coordinate in values
                )
            return values

        def face_key_for(face_id: FaceId):
            return _face_key(
                cast(bmesh.types.BMFace, _SnapshotFace(face_coords(face_id))),
                axis_index,
                tolerance,
                mirrored=False,
            )

        def mirrored_face_key_for(face_id: FaceId):
            return _face_key(
                cast(bmesh.types.BMFace, _SnapshotFace(face_coords(face_id))),
                axis_index,
                tolerance,
                mirrored=True,
            )

        def face_centroid(values) -> Coordinate3D:
            if not values:
                return Coordinate3D(0.0, 0.0, 0.0)
            center = Vector((0.0, 0.0, 0.0))
            for value in values:
                center += Vector(value)
            center /= len(values)
            return _coordinate_3d(center)

        key_to_face_ids: dict[FaceKey, Sequence[FaceId]] = defaultdict(list)
        faces_by_count_centroid: dict[
            tuple[int, QuantizedCoordinate],
            Sequence[FaceId],
        ] = defaultdict(list)
        centroids: dict[FaceId, Coordinate3D] = {}
        for face_id in face_vertex_ids:
            cast(list[FaceId], key_to_face_ids[face_key_for(face_id)]).append(face_id)

        def centroid_for(face_id: FaceId) -> Coordinate3D:
            centroid = centroids.get(face_id)
            if centroid is None:
                centroid = face_centroid(face_coords(face_id))
                centroids[face_id] = centroid
            return centroid

        fallback_index_ready = False

        def ensure_fallback_index() -> None:
            nonlocal fallback_index_ready
            if fallback_index_ready:
                return
            for candidate_id, vertex_ids in face_vertex_ids.items():
                centroid = centroid_for(candidate_id)
                key = (len(vertex_ids), _quantized_coordinate(Vector(centroid.as_tuple()), tolerance))
                centroid_index = cast(
                    defaultdict[tuple[int, QuantizedCoordinate], list[FaceId]],
                    faces_by_count_centroid,
                )
                centroid_index[key].append(candidate_id)
            fallback_index_ready = True

    else:
        key_to_face_ids = face_registry.face_key_buckets
        # Exact mirrored keys resolve without the global centroid index. Build
        # that index only when a key miss enters geometric fallback.
        faces_by_count_centroid = None
        face_coords = face_registry.face_coordinates
        mirrored_face_key_for = face_registry.mirrored_face_key
        centroid_for = face_registry.face_centroid

        def ensure_fallback_index() -> None:
            nonlocal faces_by_count_centroid
            if faces_by_count_centroid is None:
                faces_by_count_centroid = face_registry.centroid_buckets

    fallback_assignments: dict[FaceId, FaceId] = {}
    for face_id in fallback:
        values = face_coords(face_id)
        mirrored_key = mirrored_face_key_for(face_id)
        candidates = key_to_face_ids.get(mirrored_key)
        if candidates is None:
            ensure_fallback_index()
            record_centroid = centroid_for(face_id)
            mirrored_centroid = mirror_coordinate(Vector(record_centroid.as_tuple()), axis_index)
            mirrored_values = face_coords(face_id, mirrored=True)
            found: list[FaceId] = []
            seen: set[FaceId] = set()
            found_self = False
            found_other = False
            centroid_index = cast(
                Mapping[tuple[int, QuantizedCoordinate], Sequence[FaceId]],
                faces_by_count_centroid,
            )
            for centroid_key in _iter_quantized_neighborhood(mirrored_centroid, tolerance):
                for candidate_id in centroid_index.get((len(values), centroid_key), ()):
                    if candidate_id in seen:
                        continue
                    seen.add(candidate_id)
                    if not _coords_match_chebyshev(
                        mirrored_values,
                        face_coords(candidate_id),
                        tolerance,
                    ):
                        continue
                    found.append(candidate_id)
                    if candidate_id == face_id:
                        found_self = True
                    else:
                        found_other = True
                    if found_self and found_other:
                        break
                if found_self and found_other:
                    break
            candidates = found
        if candidates:
            candidate = candidates[0]
            if abs(centroid_for(face_id).component(axis_index)) > tolerance and candidate == face_id:
                candidate = next((item for item in candidates if item != face_id), candidate)
            fallback_assignments[face_id] = candidate

    primary_targets = set(mirror_face_ids.values())
    fallback_target_counts: dict[FaceId, int] = defaultdict(int)
    for counterpart in fallback_assignments.values():
        fallback_target_counts[counterpart] += 1
    for face_id, counterpart in fallback_assignments.items():
        if counterpart in primary_targets or fallback_target_counts[counterpart] > 1:
            continue
        mirror_face_ids[face_id] = counterpart

    final_target_counts: dict[FaceId, int] = defaultdict(int)
    for counterpart in mirror_face_ids.values():
        final_target_counts[counterpart] += 1
    if any(count > 1 for count in final_target_counts.values()):
        mirror_face_ids = {
            face_id: counterpart
            for face_id, counterpart in mirror_face_ids.items()
            if final_target_counts[counterpart] == 1
        }
    return mirror_face_ids


def _face_key(
    face: bmesh.types.BMFace,
    axis_index: int,
    tolerance: float,
    *,
    mirrored: bool,
) -> FaceKey:
    coordinates: list[QuantizedCoordinate] = []
    for vertex in face.verts:
        co = mirror_coordinate(vertex.co, axis_index) if mirrored else vertex.co
        coordinates.append(_quantized_coordinate(co, tolerance))
    return FaceKey(vertex_count=len(coordinates), coordinates=tuple(sorted(coordinates)))


def _region_allows_orphan_self_map(
    faces: Sequence[bmesh.types.BMFace],
    path_vertex_indices: set[int],
    axis_index: int,
    tolerance: float,
) -> bool:
    """True when non-path vertices of *faces* are geometrically self-mirrored.

    Path endpoints may be asymmetric until the X stitch completes (native cut
    on a ρ(F)=F carrier). Non-path vertices (carrier boundary / ears) must
    still pair injectively under ρ within *tolerance*. An asymmetric ear on a
    dissolved L∪R∪E union fails this guard.

    Path vertices are identified by ``BMVert.index`` (stable within one resolve
    call after ``ensure_lookup_table``). Python ``id()`` of BMesh proxies is
    **not** stable across layer ops / re-wraps and must not be used.
    """

    seen: dict[int, bmesh.types.BMVert] = {}
    for face in faces:
        if not face.is_valid:
            continue
        for vertex in face.verts:
            if vertex.is_valid:
                seen[vertex.index] = vertex
    verts = list(seen.values())
    if not verts:
        return False

    available = list(verts)
    for vertex in verts:
        if vertex.index in path_vertex_indices:
            # Path endpoints are exempt: the cut may break geometric symmetry
            # until p-stitch + mirror finish the X.
            continue
        mirrored = mirror_coordinate(vertex.co, axis_index)
        match_index = None
        for index, candidate in enumerate(available):
            if coordinates_match(candidate.co, mirrored, tolerance):
                match_index = index
                break
        if match_index is None:
            return False
        available.pop(match_index)
    return True


def resolve_live_mirror_face_map(
    bm: bmesh.types.BMesh,
    mirror_face_ids: Mapping[FaceId, FaceId | None],
    axis_index: int,
    tolerance: float,
    path_edges: Sequence[bmesh.types.BMEdge] | None = None,
) -> MirrorFaceMap:
    """Remap mirror targets orphaned by post-native dissolves onto live faces.

    Pre-native pair tables can leave a target FACE_ID with no surviving face
    after a plane edge is dissolved (two mirrored quads → one spanning face).
    Orphan self-map (source → source) is allowed only when the live region
    (all faces sharing that FACE_ID in path scope) is geometrically
    self-mirrored under ρ within *tolerance*, allowing path endpoints to be
    asymmetric until the X is completed (ρ(F)=F). Asymmetric
    dissolved unions (e.g. L∪R∪ear) fail the guard and drop the pair so the
    existing unmatched-face decline fires.

    Scope is limited to carrier faces of *path_edges* (and their pre-native
    counterparts). Unrelated faces are left untouched. When *path_edges* is
    ``None``, no orphan remapping is applied.
    """

    if path_edges is None:
        return {}
    face_layer = bm.faces.layers.int.get(FACE_ID_LAYER)
    if face_layer is None:
        return {}

    # index_update assigns stable 0..n-1 indices even on free-standing BMesh
    # (ensure_lookup_table alone leaves index==-1 there). Edit-mesh BMesh also
    # benefits after layer ops that may leave indices dirty.
    bm.verts.index_update()
    bm.faces.index_update()
    bm.verts.ensure_lookup_table()
    bm.faces.ensure_lookup_table()

    path_vertex_indices: set[int] = set()
    scope_ids: set[FaceId] = set()
    for edge in path_edges:
        if not edge.is_valid:
            continue
        for vertex in edge.verts:
            if vertex.is_valid:
                path_vertex_indices.add(vertex.index)
        for face in edge.link_faces:
            if face.is_valid:
                scope_ids.add(FaceId(int(face[face_layer])))
    for face_id in list(scope_ids):
        target = mirror_face_ids.get(face_id)
        if target is not None:
            scope_ids.add(target)

    remapped: MirrorFaceMap = {}
    for face_id in scope_ids:
        target = mirror_face_ids.get(face_id)
        if target is not None:
            remapped[face_id] = target

    faces_by_id: dict[FaceId, list[bmesh.types.BMFace]] = defaultdict(list)
    for face in bm.faces:
        if not face.is_valid:
            continue
        face_id = FaceId(int(face[face_layer]))
        if face_id in scope_ids:
            faces_by_id[face_id].append(face)
    live_ids = set(faces_by_id)

    for face_id, faces in faces_by_id.items():
        target_id = remapped.get(face_id)
        if target_id is None or target_id in live_ids:
            continue
        # Orphan target: self-map only when the live region is ρ-self-mirrored
        # (path endpoints exempt — see _region_allows_orphan_self_map).
        if _region_allows_orphan_self_map(faces, path_vertex_indices, axis_index, tolerance):
            remapped[face_id] = face_id
        else:
            # Clear so target_face_ids_for_edges treats the carrier as unmatched.
            del remapped[face_id]
    return remapped


def _canonical_carrier_frames(
    carrier_ids: set[FaceId],
    mirror_face_ids: MirrorFaceMap,
    carrier_frames: CarrierFrameMap,
    axis_index: int,
) -> tuple[list[CarrierFrameSnapshot], str]:
    by_orbit: dict[tuple[int, int], list[tuple[FaceId, CarrierFrameSnapshot]]] = defaultdict(list)
    for face_id in carrier_ids:
        mirrored = mirror_face_ids.get(face_id, face_id)
        orbit = tuple(sorted((int(face_id), int(mirrored))))
        frame = carrier_frames.get(face_id)
        if frame is None:
            return [], "a mirrored cut carrier has no pre-native canonical frame"
        by_orbit[orbit].append((face_id, frame))

    selected: list[CarrierFrameSnapshot] = []
    for entries in by_orbit.values():
        _face_id, frame = max(
            entries,
            key=lambda entry: (
                entry[1].origin.component(axis_index),
                entry[1].origin.as_tuple(),
                -int(entry[0]),
            ),
        )
        if frame.normal is None or frame.basis_u is None:
            return [], "a mirrored cut carrier has a degenerate canonical frame"
        selected.append(frame)
    selected.sort(key=lambda frame: (frame.origin.as_tuple(), frame.normal.as_tuple()))
    return selected, ""
