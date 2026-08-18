# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import cast

import bmesh
import numpy
from mathutils import Vector

from ._types import (
    CarrierFrameMap,
    CarrierFrameSnapshot,
    Coordinate3D,
    FaceId,
    HiddenFaceMap,
    MirrorFaceMap,
    TopologyPreparation,
)
from .face_mapping import FaceRegistry, _snapshot_face_map
from .layer_names import (
    CONNECT_HISTORY_EDGE_TOKEN_LAYER,
    EDGE_APPLY_ID_LAYER,
    EDGE_HIDDEN_LAYER,
    EDGE_LIVE_HIDDEN_LAYER,
    EDGE_ORIGINAL_LAYER,
    EDGE_SELECTION_LAYER,
    FACE_APPLY_ID_LAYER,
    FACE_HIDDEN_LAYER,
    FACE_ID_LAYER,
    FACE_LIVE_HIDDEN_LAYER,
    FACE_MIRROR_ID_LAYER,
    FACE_SELECTION_LAYER,
    HISTORY_TOKEN_LAYER,
    TEMP_LAYER_NAMES,
    VERT_APPLY_ID_LAYER,
    VERT_BACKUP_ID_LAYER,
    VERT_COLLAPSE_GROUP_LAYER,
    VERT_HIDDEN_LAYER,
    VERT_LIVE_HIDDEN_LAYER,
    VERT_MERGE_GROUP_LAYER,
    VERT_SELECTION_LAYER,
    VERT_SESSION_ID_LAYER,
)
from .matching import (
    VertexMirrorLookup,
    VertexRegistry,
    _coordinate_3d,
    _one_sided_pair_table,
    build_vertex_pair_table,
)


def remove_temporary_layers(bm: bmesh.types.BMesh) -> bool:
    """Remove every layer owned by this add-on and report whether one existed."""

    removed = False
    layer_groups = (
        (bm.edges.layers.int, EDGE_ORIGINAL_LAYER),
        (bm.edges.layers.int, CONNECT_HISTORY_EDGE_TOKEN_LAYER),
        (bm.faces.layers.int, FACE_ID_LAYER),
        (bm.faces.layers.int, FACE_MIRROR_ID_LAYER),
        (bm.faces.layers.int, FACE_HIDDEN_LAYER),
        (bm.faces.layers.int, HISTORY_TOKEN_LAYER),
        (bm.verts.layers.int, VERT_SELECTION_LAYER),
        (bm.edges.layers.int, EDGE_SELECTION_LAYER),
        (bm.faces.layers.int, FACE_SELECTION_LAYER),
        (bm.verts.layers.int, VERT_HIDDEN_LAYER),
        (bm.edges.layers.int, EDGE_HIDDEN_LAYER),
        (bm.verts.layers.int, VERT_LIVE_HIDDEN_LAYER),
        (bm.edges.layers.int, EDGE_LIVE_HIDDEN_LAYER),
        (bm.faces.layers.int, FACE_LIVE_HIDDEN_LAYER),
        (bm.verts.layers.int, VERT_APPLY_ID_LAYER),
        (bm.edges.layers.int, EDGE_APPLY_ID_LAYER),
        (bm.faces.layers.int, FACE_APPLY_ID_LAYER),
        (bm.verts.layers.int, VERT_BACKUP_ID_LAYER),
        (bm.verts.layers.int, VERT_SESSION_ID_LAYER),
        (bm.verts.layers.int, VERT_MERGE_GROUP_LAYER),
        (bm.verts.layers.int, VERT_COLLAPSE_GROUP_LAYER),
    )
    for layers, name in layer_groups:
        layer = layers.get(name)
        if layer is not None:
            layers.remove(layer)
            removed = True
    return removed


def remove_temporary_mesh_attributes(mesh) -> bool:
    """Remove stale layers after Edit Mode has ended."""

    removed = False
    for name in TEMP_LAYER_NAMES:
        attribute = mesh.attributes.get(name)
        if attribute is not None:
            mesh.attributes.remove(attribute)
            removed = True
    return removed


class LazyCarrierFrameMap:
    def __init__(
        self,
        vertex_coords: tuple[tuple[float, float, float], ...],
        face_vertex_ids: dict[FaceId, tuple[int, ...]],
    ) -> None:
        self._vertex_coords: tuple[tuple[float, float, float], ...] = vertex_coords
        self._face_vertex_ids: dict[FaceId, tuple[int, ...]] = face_vertex_ids
        self._coords64: numpy.ndarray = numpy.empty((0, 3), dtype=numpy.float64)
        self._loop_verts: numpy.ndarray = numpy.empty(0, dtype=numpy.int64)
        self._loop_starts: numpy.ndarray = numpy.empty(0, dtype=numpy.int64)
        self._loop_totals: numpy.ndarray = numpy.empty(0, dtype=numpy.int64)
        self._uses_snapshot = False
        self._cache: dict[FaceId, CarrierFrameSnapshot] = {}

    @classmethod
    def from_snapshot(
        cls,
        coords64: numpy.ndarray,
        loop_verts: numpy.ndarray,
        loop_starts: numpy.ndarray,
        loop_totals: numpy.ndarray,
    ) -> LazyCarrierFrameMap:
        instance = cls.__new__(cls)
        instance._vertex_coords = ()
        instance._face_vertex_ids = {}
        instance._coords64 = coords64
        instance._loop_verts = loop_verts
        instance._loop_starts = loop_starts
        instance._loop_totals = loop_totals
        instance._uses_snapshot = True
        instance._cache = {}
        return instance

    def _snapshot_backed(self) -> bool:
        return self._uses_snapshot

    def _has_key(self, key: object) -> bool:
        if self._snapshot_backed():
            return isinstance(key, int) and 1 <= key <= len(self._loop_starts)
        return key in self._face_vertex_ids

    def _coordinates_for(self, key: FaceId) -> tuple[tuple[float, float, float], ...]:
        if self._snapshot_backed():
            index = int(key) - 1
            start = int(self._loop_starts[index])
            total = int(self._loop_totals[index])
            return cast(
                tuple[tuple[float, float, float], ...],
                tuple(
                    tuple(float(value) for value in self._coords64[int(vertex_index)])
                    for vertex_index in self._loop_verts[start : start + total]
                ),
            )
        return tuple(self._vertex_coords[index] for index in self._face_vertex_ids[key])

    def get(self, key: FaceId, default=None):
        if not self._has_key(key):
            return default
        return self[key]

    def __getitem__(self, key: FaceId) -> CarrierFrameSnapshot:
        cached = self._cache.get(key)
        if cached is None:
            vertices = tuple(Coordinate3D(*coordinate) for coordinate in self._coordinates_for(key))
            cached = _carrier_frame_from_coords(vertices)
            self._cache[key] = cached
        return cached

    def __contains__(self, key: object) -> bool:
        return self._has_key(key)

    def __len__(self) -> int:
        return len(self._loop_starts) if self._snapshot_backed() else len(self._face_vertex_ids)

    def __iter__(self) -> Iterator[FaceId]:
        if self._snapshot_backed():
            return iter(FaceId(index) for index in range(1, len(self._loop_starts) + 1))
        return iter(self._face_vertex_ids)

    def __eq__(self, other) -> bool:
        if not isinstance(other, LazyCarrierFrameMap):
            return False
        if set(self) != set(other):
            return False
        for face_id in self:
            if self._coordinates_for(face_id) != other._coordinates_for(face_id):
                return False
        return True

    def __ne__(self, other) -> bool:
        return not self == other


def _carrier_frame_from_coords(vertices: tuple[Coordinate3D, ...]) -> CarrierFrameSnapshot:
    if not vertices:
        zero = Coordinate3D(0.0, 0.0, 0.0)
        return CarrierFrameSnapshot(vertices, zero, None, None, 0.0)

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

    origin = _coordinate_3d(origin_vector)
    if newell.length <= 1.0e-12:
        return CarrierFrameSnapshot(vertices, origin, None, None, 0.0)
    normal_vector = newell.normalized()

    basis_u = None
    for vertex in sorted(vertices):
        delta = Vector(vertex.as_tuple()) - origin_vector
        projected = delta - normal_vector * delta.dot(normal_vector)
        if projected.length > 1.0e-12:
            basis_u = projected.normalized()
            break
    if basis_u is None:
        return CarrierFrameSnapshot(vertices, origin, _coordinate_3d(normal_vector), None, 0.0)

    deviation = max(abs((Vector(vertex.as_tuple()) - origin_vector).dot(normal_vector)) for vertex in vertices)
    return CarrierFrameSnapshot(
        vertices=vertices,
        origin=origin,
        normal=_coordinate_3d(normal_vector),
        basis_u=_coordinate_3d(basis_u),
        deviation=float(deviation),
    )


def _carrier_frame_snapshot(face: bmesh.types.BMFace) -> CarrierFrameSnapshot:
    return _carrier_frame_from_coords(tuple(_coordinate_3d(vertex.co) for vertex in face.verts))


class LazyTopologyResolution:
    """Pure captured topology snapshot with a memoized resolution."""

    def __init__(
        self,
        coords64: numpy.ndarray,
        loop_verts: numpy.ndarray,
        loop_starts: numpy.ndarray,
        loop_totals: numpy.ndarray,
        hide_vertices: numpy.ndarray,
        hide_edges: numpy.ndarray,
        hide_faces: numpy.ndarray,
        axis_index: int,
        tolerance: float,
        history_token: int,
        *,
        mark_vertex_ids: bool = False,
        vertex_select: numpy.ndarray | None = None,
    ) -> None:
        self.coords64 = numpy.asarray(coords64, dtype=numpy.float64)
        self.loop_verts = numpy.asarray(loop_verts, dtype=numpy.int64)
        self.loop_starts = numpy.asarray(loop_starts, dtype=numpy.int64)
        self.loop_totals = numpy.asarray(loop_totals, dtype=numpy.int64)
        self.hide_vertices = numpy.asarray(hide_vertices, dtype=bool)
        self.hide_edges = numpy.asarray(hide_edges, dtype=bool)
        self.hide_faces = numpy.asarray(hide_faces, dtype=bool)
        self.vertex_count = int(len(self.coords64))
        self.edge_count = int(len(self.hide_edges))
        self.face_count = int(len(self.loop_starts))
        self.axis_index = int(axis_index)
        self.tolerance = float(tolerance)
        self.history_token = int(history_token)
        self.mark_vertex_ids = bool(mark_vertex_ids)
        self.vertex_select = None if vertex_select is None else numpy.asarray(vertex_select, dtype=bool)
        self._resolved = False
        self._resolve_count = 0
        self._pairs: dict[int, int] = {}
        self._mirror_face_ids: MirrorFaceMap = {}
        self._carrier_frames: LazyCarrierFrameMap | None = None
        self._vertex_registry: VertexRegistry | None = None
        self._vertex_resolved = numpy.zeros(self.vertex_count, dtype=bool)
        self._vertex_cache: dict[int, int | None] = {}
        self._partial_resolve_count = 0
        self._face_registry: FaceRegistry | None = None
        self._face_resolved = numpy.zeros(self.face_count, dtype=bool)
        self._face_cache: dict[FaceId, FaceId | None] = {}
        self._partial_face_resolve_count = 0

    @property
    def resolve_count(self) -> int:
        return self._resolve_count

    @property
    def partial_resolve_count(self) -> int:
        return self._partial_resolve_count

    @property
    def partial_face_resolve_count(self) -> int:
        return self._partial_face_resolve_count

    def __eq__(self, other) -> bool:
        if not isinstance(other, LazyTopologyResolution):
            return False
        arrays_equal = all(
            numpy.array_equal(getattr(self, name), getattr(other, name), equal_nan=True)
            for name in (
                "coords64",
                "loop_verts",
                "loop_starts",
                "loop_totals",
                "hide_vertices",
                "hide_edges",
                "hide_faces",
            )
        )
        selections_equal = (self.vertex_select is None and other.vertex_select is None) or (
            self.vertex_select is not None
            and other.vertex_select is not None
            and numpy.array_equal(self.vertex_select, other.vertex_select)
        )
        return (
            arrays_equal
            and selections_equal
            and self.axis_index == other.axis_index
            and self.tolerance == other.tolerance
            and self.history_token == other.history_token
            and self.mark_vertex_ids == other.mark_vertex_ids
            and self.vertex_count == other.vertex_count
            and self.edge_count == other.edge_count
            and self.face_count == other.face_count
            and self._resolved == other._resolved
            and self._pairs == other._pairs
            and self._mirror_face_ids == other._mirror_face_ids
            and self._carrier_frames == other._carrier_frames
        )

    def __ne__(self, other) -> bool:
        return not self == other

    def resolve(self) -> LazyTopologyResolution:
        if self._resolved:
            return self
        self._resolve_count += 1
        one_sided = _one_sided_pair_table(self.coords64, self.axis_index, self.tolerance)
        if one_sided is None:
            one_sided = build_vertex_pair_table(
                [Vector(tuple(float(value) for value in row)) for row in self.coords64.tolist()],
                self.axis_index,
                self.tolerance,
            )
        self._pairs = one_sided
        mirror_face_ids = _snapshot_face_map(
            self.coords64,
            self.loop_verts,
            self.loop_starts,
            self.loop_totals,
            self.axis_index,
            self.tolerance,
            self._pairs,
            self._face_registry,
        )
        for face_id, cached in self._face_cache.items():
            if mirror_face_ids.get(face_id) != cached:
                raise RuntimeError("partial face resolution disagrees with full resolution")
        self._mirror_face_ids = mirror_face_ids
        self._carrier_frames = LazyCarrierFrameMap.from_snapshot(
            self.coords64,
            self.loop_verts,
            self.loop_starts,
            self.loop_totals,
        )
        self._vertex_resolved[:] = True
        self._face_resolved[:] = True
        self._resolved = True
        return self

    def _resolve_vertex_closure(self, vertex_ids):
        if self._vertex_registry is None:
            self._vertex_registry = VertexRegistry(self.coords64, self.axis_index, self.tolerance)
        resolution = self._vertex_registry.resolve_closure(vertex_ids)
        if resolution is None:
            return None
        closure_ids, pairs = resolution
        unresolved = closure_ids[~self._vertex_resolved[closure_ids]]
        if len(unresolved):
            self._vertex_cache.update({int(vertex_id): pairs.get(int(vertex_id)) for vertex_id in closure_ids.tolist()})
            self._vertex_resolved[closure_ids] = True
            self._partial_resolve_count += 1
        return closure_ids, pairs

    def resolve_vertices(self, vertex_ids) -> dict[int, int | None]:
        requested = numpy.unique(numpy.asarray(tuple(vertex_ids), dtype=numpy.int64).reshape(-1))
        if len(requested) and (numpy.any(requested < 0) or numpy.any(requested >= self.vertex_count)):
            raise IndexError("vertex query index out of range")
        if self._resolved:
            return {int(vertex_id): self._pairs.get(int(vertex_id)) for vertex_id in requested.tolist()}

        unresolved = requested[~self._vertex_resolved[requested]]
        if len(unresolved):
            resolution = self._resolve_vertex_closure(unresolved)
            if resolution is None:
                self.resolve()
                return {int(vertex_id): self._pairs.get(int(vertex_id)) for vertex_id in requested.tolist()}
        return {int(vertex_id): self._vertex_cache[int(vertex_id)] for vertex_id in requested.tolist()}

    def resolve_faces(self, face_ids) -> dict[FaceId, FaceId | None]:
        requested = numpy.unique(numpy.asarray(tuple(face_ids), dtype=numpy.int64).reshape(-1))
        if len(requested) and (numpy.any(requested < 1) or numpy.any(requested > self.face_count)):
            raise IndexError("face query index out of range")
        requested_ids = tuple(FaceId(int(face_id)) for face_id in requested.tolist())
        if self._resolved:
            return {face_id: self._mirror_face_ids.get(face_id) for face_id in requested_ids}

        requested_indices = requested - 1
        unresolved = requested[~self._face_resolved[requested_indices]]
        if len(unresolved):
            if self._face_registry is None:
                self._face_registry = FaceRegistry(
                    self.coords64,
                    self.loop_verts,
                    self.loop_starts,
                    self.loop_totals,
                    self.axis_index,
                    self.tolerance,
                )
            unresolved_ids = tuple(FaceId(int(face_id)) for face_id in unresolved.tolist())
            vertex_ids = self._face_registry.vertices_for_faces(unresolved_ids)
            vertex_resolution = self._resolve_vertex_closure(vertex_ids)
            if vertex_resolution is None:
                self.resolve()
                return {face_id: self._mirror_face_ids.get(face_id) for face_id in requested_ids}
            _closure_vertex_ids, pairs = vertex_resolution
            closure, targets, needs_fallback = self._face_registry.resolve_primary_closure(
                unresolved_ids,
                pairs,
            )
            if needs_fallback:
                self.resolve()
                return {face_id: self._mirror_face_ids.get(face_id) for face_id in requested_ids}
            self._face_cache.update(targets)
            closure_indices = numpy.asarray([int(face_id) - 1 for face_id in closure], dtype=numpy.int64)
            self._face_resolved[closure_indices] = True
            self._partial_face_resolve_count += 1
        return {face_id: self._face_cache[face_id] for face_id in requested_ids}

    @property
    def scoped_mirror_face_ids(self) -> Mapping[FaceId, FaceId | None]:
        """Return only the face results already available without resolving all."""

        if self._resolved:
            return self._mirror_face_ids
        return self._face_cache

    @property
    def scoped_carrier_frames(self) -> LazyCarrierFrameMap:
        """Return the snapshot-backed lazy frame map without resolving topology."""

        if self._carrier_frames is None:
            self._carrier_frames = LazyCarrierFrameMap.from_snapshot(
                self.coords64,
                self.loop_verts,
                self.loop_starts,
                self.loop_totals,
            )
        return self._carrier_frames

    def __deepcopy__(self, memo):
        clone = type(self).__new__(type(self))
        memo[id(self)] = clone
        for name in (
            "coords64",
            "loop_verts",
            "loop_starts",
            "loop_totals",
            "hide_vertices",
            "hide_edges",
            "hide_faces",
            "vertex_select",
            "_vertex_resolved",
            "_face_resolved",
        ):
            value = getattr(self, name)
            setattr(clone, name, None if value is None else value.copy())
        for name in (
            "axis_index",
            "tolerance",
            "history_token",
            "mark_vertex_ids",
            "vertex_count",
            "edge_count",
            "face_count",
            "_resolved",
            "_resolve_count",
            "_partial_resolve_count",
            "_partial_face_resolve_count",
        ):
            setattr(clone, name, getattr(self, name))
        clone._pairs = dict(self._pairs)
        clone._mirror_face_ids = dict(self._mirror_face_ids)
        clone._vertex_cache = dict(self._vertex_cache)
        clone._vertex_registry = None
        clone._face_cache = dict(self._face_cache)
        clone._face_registry = None
        if self._carrier_frames is None:
            clone._carrier_frames = None
        else:
            clone._carrier_frames = LazyCarrierFrameMap.from_snapshot(
                clone.coords64,
                clone.loop_verts,
                clone.loop_starts,
                clone.loop_totals,
            )
            clone._carrier_frames._cache = dict(self._carrier_frames._cache)
        return clone

    @property
    def pairs(self) -> dict[int, int]:
        return self.resolve()._pairs

    @property
    def mirror_face_ids(self) -> MirrorFaceMap:
        return self.resolve()._mirror_face_ids

    @property
    def carrier_frames(self) -> CarrierFrameMap:
        return cast(CarrierFrameMap, self.resolve()._carrier_frames)

    @property
    def vertex_lookup(self) -> VertexMirrorLookup:
        self.resolve()
        coordinates = cast(
            tuple[tuple[float, float, float], ...],
            tuple(tuple(float(value) for value in row) for row in self.coords64.tolist()),
        )
        return VertexMirrorLookup(
            axis_index=self.axis_index,
            tolerance=self.tolerance,
            coords=coordinates,
        )

    @property
    def vertex_lookup_unresolved(self) -> VertexMirrorLookup:
        """Build a RIP lookup from captured arrays without resolving topology.

        The registered-side batch index is materialized here so RIP can time
        registry construction separately from its selected-region resolution.
        ``coords64`` is already the exact float64 image consumed by the eager
        lookup, so no per-coordinate Python tuple conversion is required.
        """

        selected_indices = (
            None if self.vertex_select is None else numpy.flatnonzero(self.vertex_select).astype(numpy.int64)
        )
        lookup = VertexMirrorLookup(
            axis_index=self.axis_index,
            tolerance=self.tolerance,
            coords=self.coords64,
            selected_indices=selected_indices,
        )
        if len(self.coords64):
            lookup._registered_batch_index()
        return lookup

    @property
    def matched_faces(self) -> int:
        return len(self.mirror_face_ids)

    @property
    def total_faces(self) -> int:
        return self.face_count

    def materialize(self, bm: bmesh.types.BMesh) -> None:
        if not self._resolved:
            raise RuntimeError("Topology resolution must be resolved before materialize")
        face_id_layer = bm.faces.layers.int.get(FACE_ID_LAYER)
        if face_id_layer is None:
            return
        mirror_layer = bm.faces.layers.int.get(FACE_MIRROR_ID_LAYER)
        if mirror_layer is None:
            mirror_layer = bm.faces.layers.int.new(FACE_MIRROR_ID_LAYER)
            face_id_layer = bm.faces.layers.int.get(FACE_ID_LAYER)
            if face_id_layer is None:
                return
        values = self._mirror_face_ids
        for face in bm.faces:
            face_id = FaceId(int(face[face_id_layer]))
            face[mirror_layer] = int(values.get(face_id, FaceId(0)))

    def materialize_faces(self, bm: bmesh.types.BMesh, face_ids) -> None:
        """Write mirror IDs for the resolved face scope only.

        A newly-created BMesh integer layer is zero-initialized. Consequently,
        faces outside *face_ids* retain the same observable "no counterpart"
        value that full materialization writes for unmatched faces. When the
        layer already exists, out-of-scope faces keep their previous values;
        the layer-completeness heuristic in history.py therefore reports such
        sessions as incomplete and repair declines instead of restoring a
        partial map.
        """

        requested = numpy.unique(numpy.asarray(tuple(face_ids), dtype=numpy.int64).reshape(-1))
        if len(requested) == 0:
            return
        if numpy.any(requested < 1) or numpy.any(requested > self.face_count):
            raise IndexError("face materialize index out of range")
        if not self._resolved and numpy.any(~self._face_resolved[requested - 1]):
            raise RuntimeError("Topology face scope must be resolved before materialize_faces")

        face_id_layer = bm.faces.layers.int.get(FACE_ID_LAYER)
        if face_id_layer is None:
            return
        mirror_layer = bm.faces.layers.int.get(FACE_MIRROR_ID_LAYER)
        if mirror_layer is None:
            mirror_layer = bm.faces.layers.int.new(FACE_MIRROR_ID_LAYER)
            face_id_layer = bm.faces.layers.int.get(FACE_ID_LAYER)
            if face_id_layer is None:
                return

        requested_ids = {FaceId(int(face_id)) for face_id in requested.tolist()}
        values = self.scoped_mirror_face_ids
        for face in bm.faces:
            face_id = FaceId(int(face[face_id_layer]))
            if face_id not in requested_ids:
                continue
            face[mirror_layer] = int(values.get(face_id) or FaceId(0))


def _face_loop_vertex_indices_match(mesh_vertex_indices: Sequence[int], bm_face) -> bool:
    """True when Mesh loop ``vertex_index`` order equals BMesh ``face.verts`` index order (§L-3)."""

    return [int(index) for index in mesh_vertex_indices] == [int(vertex.index) for vertex in bm_face.verts]


def _first_last_face_loop_order_matches(
    first_mesh_indices: Sequence[int],
    last_mesh_indices: Sequence[int],
    bm: bmesh.types.BMesh,
) -> bool:
    """§L-3 first/last face order guard via index columns (not coordinates)."""

    return _face_loop_vertex_indices_match(first_mesh_indices, bm.faces[0]) and _face_loop_vertex_indices_match(
        last_mesh_indices, bm.faces[-1]
    )


def _read_polygon_loop_vertex_indices(polygons, loops, face_index: int) -> list[int] | None:
    """Read one polygon's loop ``vertex_index`` column via single-element access.

    Used by selection bulk capture so guards do not ``foreach_get`` every loop.
    """

    try:
        polygon = polygons[face_index]
        start = int(polygon.loop_start)
        total = int(polygon.loop_total)
        return [int(loops[offset].vertex_index) for offset in range(start, start + total)]
    except (AttributeError, IndexError, TypeError, ValueError):
        return None


def _capture_bmesh_snapshot(bm: bmesh.types.BMesh, *, skip_vertex_edge_hides: bool = False):
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    bm.verts.index_update()
    coords = numpy.asarray(
        [(float(vertex.co[0]), float(vertex.co[1]), float(vertex.co[2])) for vertex in bm.verts],
        dtype=numpy.float64,
    ).reshape((-1, 3))
    loop_verts = numpy.asarray([vertex.index for face in bm.faces for vertex in face.verts], dtype=numpy.int64)
    loop_totals = numpy.asarray([len(face.verts) for face in bm.faces], dtype=numpy.int64)
    loop_starts = (
        numpy.concatenate((numpy.asarray((0,), dtype=numpy.int64), numpy.cumsum(loop_totals[:-1], dtype=numpy.int64)))
        if len(loop_totals)
        else numpy.empty(0, dtype=numpy.int64)
    )
    return (
        coords,
        loop_verts,
        loop_starts,
        loop_totals,
        (
            numpy.zeros(len(bm.verts), dtype=bool)
            if skip_vertex_edge_hides
            else numpy.asarray([bool(vertex.hide) for vertex in bm.verts], dtype=bool)
        ),
        (
            numpy.zeros(len(bm.edges), dtype=bool)
            if skip_vertex_edge_hides
            else numpy.asarray([bool(edge.hide) for edge in bm.edges], dtype=bool)
        ),
        numpy.asarray([bool(face.hide) for face in bm.faces], dtype=bool),
    )


def _capture_mesh_snapshot(
    mesh_object,
    bm: bmesh.types.BMesh,
    *,
    skip_vertex_edge_hides: bool = False,
):
    if getattr(getattr(mesh_object, "data", None), "shape_keys", None) is not None:
        return None
    bm.verts.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    mesh_object.update_from_editmode()
    mesh = mesh_object.data
    count_vertices, count_edges, count_faces = len(mesh.vertices), len(mesh.edges), len(mesh.polygons)
    count_loops = len(mesh.loops)
    coords32 = numpy.empty(count_vertices * 3, dtype=numpy.float32)
    mesh.vertices.foreach_get("co", coords32)
    loop_verts32 = numpy.empty(count_loops, dtype=numpy.int32)
    loop_totals32 = numpy.empty(count_faces, dtype=numpy.int32)
    mesh.loops.foreach_get("vertex_index", loop_verts32)
    mesh.polygons.foreach_get("loop_total", loop_totals32)
    loop_verts = loop_verts32.astype(numpy.int64)
    loop_totals = loop_totals32.astype(numpy.int64)
    # Loops are written contiguously in polygon order by the BM->Mesh
    # conversion; the total-sum check guards that invariant.
    loop_starts = numpy.cumsum(loop_totals) - loop_totals
    if count_faces and int(loop_starts[-1] + loop_totals[-1]) != count_loops:
        return None
    # mesh.attributes is filtered while in edit mode, so non-RIP hide state
    # can only be trusted through the per-element reads. RIP does not consume
    # vertex/edge hide state and intentionally keeps these arrays all-false.
    hide_vertices = numpy.zeros(count_vertices, dtype=bool)
    hide_edges = numpy.zeros(count_edges, dtype=bool)
    hide_faces = numpy.empty(count_faces, dtype=bool)
    if not skip_vertex_edge_hides:
        mesh.vertices.foreach_get("hide", hide_vertices)
        mesh.edges.foreach_get("hide", hide_edges)
    mesh.polygons.foreach_get("hide", hide_faces)
    if len(bm.verts) != count_vertices or len(bm.edges) != count_edges or len(bm.faces) != count_faces:
        return None
    if count_vertices and (
        numpy.asarray(tuple(float(value) for value in bm.verts[0].co), dtype=numpy.float32).tobytes()
        != coords32[:3].tobytes()
        or numpy.asarray(
            tuple(float(value) for value in bm.verts[count_vertices - 1].co), dtype=numpy.float32
        ).tobytes()
        != coords32[-3:].tobytes()
    ):
        return None
    if count_faces:
        first_start, last_start = int(loop_starts[0]), int(loop_starts[-1])
        first_mesh = [int(value) for value in loop_verts[first_start : first_start + int(loop_totals[0])]]
        last_mesh = [int(value) for value in loop_verts[last_start : last_start + int(loop_totals[-1])]]
        if not _first_last_face_loop_order_matches(first_mesh, last_mesh, bm):
            return None
    return (
        coords32.reshape(count_vertices, 3).astype(numpy.float64),
        loop_verts,
        loop_starts,
        loop_totals,
        hide_vertices,
        hide_edges,
        hide_faces,
    )


def prepare_topology(
    bm: bmesh.types.BMesh,
    axis_index: int,
    tolerance: float,
    history_token: int = 0,
    *,
    mark_vertex_ids: bool = False,
    mesh_object=None,
) -> TopologyPreparation:
    """Capture topology eagerly and defer geometric resolution until consumed."""

    skip_vertex_edge_hides = mark_vertex_ids
    snapshot = (
        _capture_mesh_snapshot(mesh_object, bm, skip_vertex_edge_hides=skip_vertex_edge_hides)
        if mesh_object is not None
        else None
    )
    if snapshot is None:
        snapshot = _capture_bmesh_snapshot(bm, skip_vertex_edge_hides=skip_vertex_edge_hides)
    (
        coords64,
        loop_verts,
        loop_starts,
        loop_totals,
        hide_vertices,
        hide_edges,
        hide_faces,
    ) = snapshot

    def _reuse(layers, name, wanted):
        # CustomData add/remove reallocates the whole domain, so fully
        # overwritten layers are reused across prepares instead of recycled.
        layer = layers.get(name)
        if wanted:
            return layer if layer is not None else layers.new(name)
        if layer is not None:
            layers.remove(layer)
        return None

    for layers, name in (
        (bm.faces.layers.int, FACE_MIRROR_ID_LAYER),
        (bm.verts.layers.int, VERT_SELECTION_LAYER),
        (bm.edges.layers.int, EDGE_SELECTION_LAYER),
        (bm.faces.layers.int, FACE_SELECTION_LAYER),
        (bm.verts.layers.int, VERT_BACKUP_ID_LAYER),
        (bm.verts.layers.int, VERT_MERGE_GROUP_LAYER),
        (bm.verts.layers.int, VERT_COLLAPSE_GROUP_LAYER),
    ):
        stale = layers.get(name)
        if stale is not None:
            layers.remove(stale)

    edge_layer = _reuse(bm.edges.layers.int, EDGE_ORIGINAL_LAYER, True)
    face_id_layer = _reuse(bm.faces.layers.int, FACE_ID_LAYER, True)
    # FACE_MIRROR_ID_LAYER is created by materialize; restore treats the
    # missing layer the same as an all-zero one.
    history_layer = _reuse(bm.faces.layers.int, HISTORY_TOKEN_LAYER, True)
    edge_hidden_layer = _reuse(bm.edges.layers.int, EDGE_HIDDEN_LAYER, bool(hide_edges.any()))
    vertex_hidden_layer = _reuse(bm.verts.layers.int, VERT_HIDDEN_LAYER, bool(hide_vertices.any()))
    face_hidden_layer = _reuse(bm.faces.layers.int, FACE_HIDDEN_LAYER, bool(hide_faces.any()))
    if mark_vertex_ids:
        # Session vertex IDs are populated after prepare (RIP: selected
        # region; extrude: every vertex). Recreate the layer so a repeated
        # prepare cannot leak positive IDs from an earlier snapshot.
        stale_session_layer = bm.verts.layers.int.get(VERT_SESSION_ID_LAYER)
        if stale_session_layer is not None:
            bm.verts.layers.int.remove(stale_session_layer)
        bm.verts.layers.int.new(VERT_SESSION_ID_LAYER)
    else:
        _reuse(bm.verts.layers.int, VERT_SESSION_ID_LAYER, False)
    for edge_id, edge in enumerate(bm.edges, start=1):
        edge[edge_layer] = edge_id
        if edge_hidden_layer is not None:
            edge[edge_hidden_layer] = int(edge.hide)
    if vertex_hidden_layer is not None:
        for vertex in bm.verts:
            vertex[vertex_hidden_layer] = int(vertex.hide)
    if face_hidden_layer is None:
        hidden_by_face_id: HiddenFaceMap = dict.fromkeys(map(FaceId, range(1, len(loop_starts) + 1)), False)
        for face_id, face in enumerate(bm.faces, start=1):
            face[face_id_layer] = face_id
            face[history_layer] = history_token
    else:
        hidden_list = hide_faces.tolist()
        hidden_by_face_id = {FaceId(index): bool(value) for index, value in enumerate(hidden_list, start=1)}
        for face_id, face in enumerate(bm.faces, start=1):
            face[face_id_layer] = face_id
            face[history_layer] = history_token
            face[face_hidden_layer] = int(face.hide)
    resolution = LazyTopologyResolution(
        coords64,
        loop_verts,
        loop_starts,
        loop_totals,
        hide_vertices,
        hide_edges,
        hide_faces,
        axis_index,
        tolerance,
        history_token,
        mark_vertex_ids=mark_vertex_ids,
        vertex_select=(
            numpy.asarray([bool(vertex.select) for vertex in bm.verts], dtype=bool) if mark_vertex_ids else None
        ),
    )
    return TopologyPreparation(resolution, hidden_by_face_id, len(loop_starts))


def get_required_layers(bm: bmesh.types.BMesh):
    return (
        bm.edges.layers.int.get(EDGE_ORIGINAL_LAYER),
        bm.faces.layers.int.get(FACE_ID_LAYER),
    )


@dataclass(frozen=True, slots=True)
class SelectionCapture:
    """Bulk or BMesh-compatible capture of coordinates, topology, and selection.

    ``coords`` is always float64 shaped ``(N, 3)``. Selected-index arrays are
    empty when their domain was not requested. Face-loop arrays are populated
    only when the face domain was requested. History is populated only when
    ``include_history=True`` was passed to :func:`capture_selection_snapshot`.
    Edge history retains endpoint indices and unfiltered element type names.
    """

    coords: numpy.ndarray
    selected_verts: numpy.ndarray
    selected_edges: numpy.ndarray
    selected_faces: numpy.ndarray
    loop_verts: numpy.ndarray = field(default_factory=lambda: numpy.empty(0, dtype=numpy.int64))
    loop_starts: numpy.ndarray = field(default_factory=lambda: numpy.empty(0, dtype=numpy.int64))
    loop_totals: numpy.ndarray = field(default_factory=lambda: numpy.empty(0, dtype=numpy.int64))
    history_coords: tuple[Vector, ...] = ()
    history_indices: tuple[int, ...] = ()
    history_edge_indices: tuple[tuple[int, int], ...] = ()
    history_htypes: tuple[str, ...] = ()


def _empty_selected() -> numpy.ndarray:
    return numpy.empty(0, dtype=numpy.int64)


def _selected_indices_from_flags(flags: numpy.ndarray) -> numpy.ndarray:
    if flags.size == 0:
        return _empty_selected()
    return numpy.flatnonzero(flags).astype(numpy.int64, copy=False)


def _capture_bmesh_selection(
    bm: bmesh.types.BMesh,
    domain_set: frozenset[str],
    include_loops: bool = False,
) -> SelectionCapture:
    """Compatibility path: read coordinates and selection flags from BMesh."""

    coords = numpy.asarray(
        [(float(vertex.co[0]), float(vertex.co[1]), float(vertex.co[2])) for vertex in bm.verts],
        dtype=numpy.float64,
    ).reshape((-1, 3))
    selected_verts = (
        _selected_indices_from_flags(numpy.asarray([bool(vertex.select) for vertex in bm.verts], dtype=bool))
        if "VERT" in domain_set
        else _empty_selected()
    )
    selected_edges = (
        _selected_indices_from_flags(numpy.asarray([bool(edge.select) for edge in bm.edges], dtype=bool))
        if "EDGE" in domain_set
        else _empty_selected()
    )
    selected_faces = _empty_selected()
    loop_verts = numpy.empty(0, dtype=numpy.int64)
    loop_starts = numpy.empty(0, dtype=numpy.int64)
    loop_totals = numpy.empty(0, dtype=numpy.int64)
    if "FACE" in domain_set:
        selected_faces = _selected_indices_from_flags(
            numpy.asarray([bool(face.select) for face in bm.faces], dtype=bool)
        )
        if include_loops:
            face_loop_totals = []
            face_loop_verts = []
            for face in bm.faces:
                face_loop_totals.append(len(face.verts))
                face_loop_verts.extend(vertex.index for vertex in face.verts)
            loop_verts = numpy.asarray(face_loop_verts, dtype=numpy.int64)
            loop_totals = numpy.asarray(face_loop_totals, dtype=numpy.int64)
            loop_starts = numpy.cumsum(loop_totals, dtype=numpy.int64) - loop_totals
    return SelectionCapture(
        coords=coords,
        selected_verts=selected_verts,
        selected_edges=selected_edges,
        selected_faces=selected_faces,
        loop_verts=loop_verts,
        loop_starts=loop_starts,
        loop_totals=loop_totals,
    )


def _capture_mesh_selection(
    mesh_object,
    bm: bmesh.types.BMesh,
    domain_set: frozenset[str],
    include_loops: bool = False,
) -> SelectionCapture | None:
    """Mesh bulk path with the same integrity guards as ``_capture_mesh_snapshot``.

    Returns ``None`` when bulk is unsafe (shape keys, missing hooks, order
    mismatch) so callers fall back to the BMesh path.
    """

    data = getattr(mesh_object, "data", None)
    if data is None:
        return None
    if getattr(data, "shape_keys", None) is not None:
        return None
    update = getattr(mesh_object, "update_from_editmode", None)
    if not callable(update):
        return None
    vertices = getattr(data, "vertices", None)
    edges = getattr(data, "edges", None)
    polygons = getattr(data, "polygons", None)
    loops = getattr(data, "loops", None)
    if vertices is None or edges is None or polygons is None or loops is None:
        return None
    if not all(callable(getattr(domain, "foreach_get", None)) for domain in (vertices, edges, polygons, loops)):
        return None

    bm.verts.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    update()
    count_vertices, count_edges, count_faces = len(vertices), len(edges), len(polygons)
    if len(bm.verts) != count_vertices or len(bm.edges) != count_edges or len(bm.faces) != count_faces:
        return None

    coords32 = numpy.empty(count_vertices * 3, dtype=numpy.float32)
    vertices.foreach_get("co", coords32)
    if count_vertices and (
        numpy.asarray(tuple(float(value) for value in bm.verts[0].co), dtype=numpy.float32).tobytes()
        != coords32[:3].tobytes()
        or numpy.asarray(
            tuple(float(value) for value in bm.verts[count_vertices - 1].co), dtype=numpy.float32
        ).tobytes()
        != coords32[-3:].tobytes()
    ):
        return None

    # §L-3 first/last face order: index columns only (no full-loop foreach_get).
    if count_faces:
        first_mesh = _read_polygon_loop_vertex_indices(polygons, loops, 0)
        last_mesh = _read_polygon_loop_vertex_indices(polygons, loops, -1)
        if first_mesh is None or last_mesh is None:
            return None
        if not _first_last_face_loop_order_matches(first_mesh, last_mesh, bm):
            return None

    coords = coords32.reshape(count_vertices, 3).astype(numpy.float64)

    selected_verts = _empty_selected()
    selected_edges = _empty_selected()
    selected_faces = _empty_selected()
    loop_verts = numpy.empty(0, dtype=numpy.int64)
    loop_starts = numpy.empty(0, dtype=numpy.int64)
    loop_totals = numpy.empty(0, dtype=numpy.int64)
    # total_*_sel is maintained by update_from_editmode; zero means the flag
    # column is all false, so the foreach_get can be skipped bit-identically.
    if "VERT" in domain_set and int(getattr(data, "total_vert_sel", -1)) != 0:
        flags = numpy.empty(count_vertices, dtype=bool)
        vertices.foreach_get("select", flags)
        selected_verts = _selected_indices_from_flags(flags)
    if "EDGE" in domain_set and int(getattr(data, "total_edge_sel", -1)) != 0:
        flags = numpy.empty(count_edges, dtype=bool)
        edges.foreach_get("select", flags)
        selected_edges = _selected_indices_from_flags(flags)
    if "FACE" in domain_set and int(getattr(data, "total_face_sel", -1)) != 0:
        flags = numpy.empty(count_faces, dtype=bool)
        polygons.foreach_get("select", flags)
        selected_faces = _selected_indices_from_flags(flags)
    if "FACE" in domain_set and include_loops:
        loop_verts32 = numpy.empty(len(loops), dtype=numpy.int32)
        loop_starts32 = numpy.empty(count_faces, dtype=numpy.int32)
        loop_totals32 = numpy.empty(count_faces, dtype=numpy.int32)
        loops.foreach_get("vertex_index", loop_verts32)
        polygons.foreach_get("loop_start", loop_starts32)
        polygons.foreach_get("loop_total", loop_totals32)
        loop_verts = loop_verts32.astype(numpy.int64)
        loop_starts = loop_starts32.astype(numpy.int64)
        loop_totals = loop_totals32.astype(numpy.int64)

    return SelectionCapture(
        coords=coords,
        selected_verts=selected_verts,
        selected_edges=selected_edges,
        selected_faces=selected_faces,
        loop_verts=loop_verts,
        loop_starts=loop_starts,
        loop_totals=loop_totals,
    )


def capture_selection_snapshot(
    bm: bmesh.types.BMesh,
    *,
    mesh_object=None,
    domains: Sequence[str] = ("VERT", "EDGE", "FACE"),
    include_history: bool = False,
    include_loops: bool = False,
) -> SelectionCapture:
    """Capture vertex coordinates and selected indices for bulk consumers.

    When *mesh_object* is provided, coordinates and selection flags are read via
    Mesh ``foreach_get`` after the same integrity guards as topology bulk
    capture (shape keys, element counts, first/last co bytes, loop order).
    Guard failure or a free BMesh (``mesh_object is None``) falls back to the
    BMesh path. Select history is always read from BMesh after index refresh,
    and only when *include_history* is true (BMVert filter matches
    ``replay._vertex_snapshot``).
    """

    domain_set = frozenset(domains)
    bm.verts.ensure_lookup_table()
    bm.verts.index_update()
    if "EDGE" in domain_set:
        bm.edges.ensure_lookup_table()
        bm.edges.index_update()
    if "FACE" in domain_set:
        bm.faces.ensure_lookup_table()
        bm.faces.index_update()

    captured: SelectionCapture | None = None
    if mesh_object is not None:
        captured = _capture_mesh_selection(mesh_object, bm, domain_set, include_loops)
    if captured is None:
        captured = _capture_bmesh_selection(bm, domain_set, include_loops)

    if not include_history:
        return captured

    history = cast(
        Iterable[bmesh.types.BMVert | bmesh.types.BMEdge | bmesh.types.BMFace],
        bm.select_history,
    )
    history_elements = list(history)
    history_vertices = [element for element in history_elements if isinstance(element, bmesh.types.BMVert)]
    history_coords = tuple(element.co.copy() for element in history_vertices)
    history_indices = tuple(element.index for element in history_vertices)
    history_edge_indices = tuple(
        (int(element.verts[0].index), int(element.verts[1].index))
        for element in history_elements
        if isinstance(element, bmesh.types.BMEdge)
    )
    history_htypes = tuple(type(element).__name__ for element in history_elements)
    return SelectionCapture(
        coords=captured.coords,
        selected_verts=captured.selected_verts,
        selected_edges=captured.selected_edges,
        selected_faces=captured.selected_faces,
        loop_verts=captured.loop_verts,
        loop_starts=captured.loop_starts,
        loop_totals=captured.loop_totals,
        history_coords=history_coords,
        history_indices=history_indices,
        history_edge_indices=history_edge_indices,
        history_htypes=history_htypes,
    )
