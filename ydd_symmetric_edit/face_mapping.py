from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
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
from .matching import (
    _coordinate_3d,
    _coords_match_chebyshev,
    _iter_quantized_neighborhood,
    _quantized_coordinate,
    build_vertex_pair_table,
    coordinates_match,
    mirror_coordinate,
)
from .snapshot import FACE_ID_LAYER, _one_sided_pair_table


def _snapshot_face_map(
    coords64: numpy.ndarray,
    loop_verts: numpy.ndarray,
    loop_starts: numpy.ndarray,
    loop_totals: numpy.ndarray,
    axis_index: int,
    tolerance: float,
    vertex_pairs: dict[int, int] | None = None,
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

    coordinates = tuple(tuple(float(value) for value in row) for row in coords64.tolist())
    face_vertex_ids = {
        FaceId(face_index + 1): tuple(int(vertex_id) for vertex_id in loop_verts[int(start) : int(start) + int(total)])
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

    class _SnapshotVertex:
        def __init__(self, coordinate):
            self.co = Vector(coordinate)

    class _SnapshotFace:
        def __init__(self, values):
            self.verts = tuple(_SnapshotVertex(value) for value in values)

    def face_key(values):
        return _face_key(cast(bmesh.types.BMFace, _SnapshotFace(values)), axis_index, tolerance, mirrored=False)

    def mirrored_face_key(values):
        return _face_key(cast(bmesh.types.BMFace, _SnapshotFace(values)), axis_index, tolerance, mirrored=True)

    def face_centroid(values) -> Coordinate3D:
        if not values:
            return Coordinate3D(0.0, 0.0, 0.0)
        center = Vector((0.0, 0.0, 0.0))
        for value in values:
            center += Vector(value)
        center /= len(values)
        return _coordinate_3d(center)

    key_to_face_ids: dict[tuple, list[FaceId]] = defaultdict(list)
    faces_by_count_centroid: dict[tuple[int, QuantizedCoordinate], list[FaceId]] = defaultdict(list)
    centroids: dict[FaceId, Coordinate3D] = {}
    for face_id in face_vertex_ids:
        values = face_coords(face_id)
        key_to_face_ids[face_key(values)].append(face_id)

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
            faces_by_count_centroid[
                (len(vertex_ids), _quantized_coordinate(Vector(centroid.as_tuple()), tolerance))
            ].append(candidate_id)
        fallback_index_ready = True

    fallback_assignments: dict[FaceId, FaceId] = {}
    for face_id in fallback:
        values = face_coords(face_id)
        mirrored_key = mirrored_face_key(values)
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
            for centroid_key in _iter_quantized_neighborhood(mirrored_centroid, tolerance):
                for candidate_id in faces_by_count_centroid.get((len(values), centroid_key), ()):
                    if candidate_id in seen:
                        continue
                    seen.add(candidate_id)
                    if not _coords_match_chebyshev(mirrored_values, face_coords(candidate_id), tolerance):
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
    mirror_face_ids: MirrorFaceMap,
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

    face_layer = bm.faces.layers.int.get(FACE_ID_LAYER)
    if face_layer is None:
        return dict(mirror_face_ids)

    remapped: MirrorFaceMap = dict(mirror_face_ids)
    if path_edges is None:
        return remapped

    # index_update assigns stable 0..n-1 indices even on free-standing BMesh
    # (ensure_lookup_table alone leaves index==-1 there). Edit-mesh BMesh also
    # benefits after layer ops that may leave indices dirty.
    bm.verts.index_update()
    bm.faces.index_update()
    bm.verts.ensure_lookup_table()
    bm.faces.ensure_lookup_table()

    live_ids = {FaceId(int(face[face_layer])) for face in bm.faces}
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
        target = remapped.get(face_id)
        if target is not None:
            scope_ids.add(target)

    faces_by_id: dict[FaceId, list[bmesh.types.BMFace]] = defaultdict(list)
    for face in bm.faces:
        if not face.is_valid:
            continue
        face_id = FaceId(int(face[face_layer]))
        if face_id in scope_ids:
            faces_by_id[face_id].append(face)

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
