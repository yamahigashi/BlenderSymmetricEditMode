from __future__ import annotations

from collections.abc import Iterator
from typing import cast

import bmesh
import numpy  # type: ignore
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
from .face_mapping import _snapshot_face_map
from .layer_names import (
    EDGE_HIDDEN_LAYER,
    EDGE_ORIGINAL_LAYER,
    EDGE_SELECTION_LAYER,
    FACE_HIDDEN_LAYER,
    FACE_ID_LAYER,
    FACE_MIRROR_ID_LAYER,
    FACE_SELECTION_LAYER,
    HISTORY_TOKEN_LAYER,
    TEMP_LAYER_NAMES,
    VERT_BACKUP_ID_LAYER,
    VERT_COLLAPSE_GROUP_LAYER,
    VERT_HIDDEN_LAYER,
    VERT_MERGE_GROUP_LAYER,
    VERT_RIP_ID_LAYER,
    VERT_SELECTION_LAYER,
)
from .matching import (
    VertexMirrorLookup,
    _coordinate_3d,
    _one_sided_pair_table,
    build_vertex_pair_table,
)


def remove_temporary_layers(bm: bmesh.types.BMesh) -> bool:
    """Remove every layer owned by this add-on and report whether one existed."""

    removed = False
    layer_groups = (
        (bm.edges.layers.int, EDGE_ORIGINAL_LAYER),
        (bm.faces.layers.int, FACE_ID_LAYER),
        (bm.faces.layers.int, FACE_MIRROR_ID_LAYER),
        (bm.faces.layers.int, FACE_HIDDEN_LAYER),
        (bm.faces.layers.int, HISTORY_TOKEN_LAYER),
        (bm.verts.layers.int, VERT_SELECTION_LAYER),
        (bm.edges.layers.int, EDGE_SELECTION_LAYER),
        (bm.faces.layers.int, FACE_SELECTION_LAYER),
        (bm.verts.layers.int, VERT_HIDDEN_LAYER),
        (bm.edges.layers.int, EDGE_HIDDEN_LAYER),
        (bm.verts.layers.int, VERT_BACKUP_ID_LAYER),
        (bm.verts.layers.int, VERT_RIP_ID_LAYER),
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

    @property
    def resolve_count(self) -> int:
        return self._resolve_count

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
        )
        self._mirror_face_ids = mirror_face_ids
        self._carrier_frames = LazyCarrierFrameMap.from_snapshot(
            self.coords64,
            self.loop_verts,
            self.loop_starts,
            self.loop_totals,
        )
        self._resolved = True
        return self

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
        ):
            setattr(clone, name, getattr(self, name))
        clone._pairs = dict(self._pairs)
        clone._mirror_face_ids = dict(self._mirror_face_ids)
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


def _capture_bmesh_snapshot(bm: bmesh.types.BMesh):
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
        numpy.asarray([bool(vertex.hide) for vertex in bm.verts], dtype=bool),
        numpy.asarray([bool(edge.hide) for edge in bm.edges], dtype=bool),
        numpy.asarray([bool(face.hide) for face in bm.faces], dtype=bool),
    )


def _capture_mesh_snapshot(mesh_object, bm: bmesh.types.BMesh):
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
    # mesh.attributes is filtered while in edit mode, so hide state can
    # only be trusted through the per-element reads.
    hide_vertices = numpy.empty(count_vertices, dtype=bool)
    hide_edges = numpy.empty(count_edges, dtype=bool)
    hide_faces = numpy.empty(count_faces, dtype=bool)
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
        first = [
            tuple(coords32[3 * int(value) : 3 * int(value) + 3])
            for value in loop_verts[first_start : first_start + int(loop_totals[0])]
        ]
        last = [
            tuple(coords32[3 * int(value) : 3 * int(value) + 3])
            for value in loop_verts[last_start : last_start + int(loop_totals[-1])]
        ]
        first_bm = [
            tuple(numpy.asarray(tuple(float(component) for component in vertex.co), dtype=numpy.float32))
            for vertex in bm.faces[0].verts
        ]
        last_bm = [
            tuple(numpy.asarray(tuple(float(component) for component in vertex.co), dtype=numpy.float32))
            for vertex in bm.faces[-1].verts
        ]
        if first != first_bm or last != last_bm:
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

    snapshot = _capture_mesh_snapshot(mesh_object, bm) if mesh_object is not None else None
    if snapshot is None:
        snapshot = _capture_bmesh_snapshot(bm)
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
    rip_layer = _reuse(bm.verts.layers.int, VERT_RIP_ID_LAYER, mark_vertex_ids)
    for edge_id, edge in enumerate(bm.edges, start=1):
        edge[edge_layer] = edge_id
        if edge_hidden_layer is not None:
            edge[edge_hidden_layer] = int(edge.hide)
    if vertex_hidden_layer is not None or rip_layer is not None:
        for vertex_id, vertex in enumerate(bm.verts, start=1):
            if vertex_hidden_layer is not None:
                vertex[vertex_hidden_layer] = int(vertex.hide)
            if rip_layer is not None:
                vertex[rip_layer] = vertex_id
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
