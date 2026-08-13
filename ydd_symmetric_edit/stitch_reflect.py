from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal, overload

import bmesh
import mathutils.geometry
from mathutils import Vector

from . import stitch_common
from ._types import CarrierFrameMap, FaceId, MirrorFaceMap
from .layer_names import EDGE_ORIGINAL_LAYER, FACE_ID_LAYER
from .matching import coordinates_match, mirror_coordinate


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
        distance, factor = stitch_common._point_segment_distance_and_factor(expected, edge)
        if not stitch_common._is_interior_edge_factor(factor, edge.calc_length(), tolerance):
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

    Tris and quads use their validated native tessellation.  N-gons use
    Blender's ear-clipping tessellation, which covers concave boundary-split
    faces without spanning outside the polygon.
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
        distance, factor = stitch_common._point_segment_distance_and_factor(point, edge)
        if distance <= edge_limit and stitch_common._is_interior_edge_factor(factor, edge.calc_length(), tolerance):
            return False
    for corner_a, corner_b, corner_c in _face_surface_triangles(face):
        closest = mathutils.geometry.closest_point_on_tri(point, corner_a, corner_b, corner_c)
        if (closest - point).length <= surface_limit:
            return True
    return False


def _host_ids_by_vertex(
    edge_records: list[tuple[int, int, set[FaceId]]],
) -> dict[int, FaceId]:
    """Return H(v) only when every adjacent path edge has one equal target ID."""

    edge_ids_by_vertex: dict[int, list[set[FaceId]]] = defaultdict(list)
    for source_a, source_b, target_ids in edge_records:
        edge_ids_by_vertex[source_a].append(target_ids)
        edge_ids_by_vertex[source_b].append(target_ids)

    hosts: dict[int, FaceId] = {}
    for vertex_key, target_id_sets in edge_ids_by_vertex.items():
        if not target_id_sets or any(len(target_ids) != 1 for target_ids in target_id_sets):
            continue
        singleton_ids = {next(iter(target_ids)) for target_ids in target_id_sets}
        if len(singleton_ids) == 1:
            hosts[vertex_key] = next(iter(singleton_ids))
    return hosts


def _source_face_ids_by_vertex(
    source_edges: Iterable[bmesh.types.BMEdge],
    face_layer,
) -> dict[int, set[FaceId]]:
    source_face_ids: dict[int, set[FaceId]] = defaultdict(set)
    for edge in source_edges:
        face_ids = {FaceId(int(face[face_layer])) for face in edge.link_faces}
        for vertex in edge.verts:
            source_face_ids[hash(vertex)].update(face_ids)
    return source_face_ids


def _carrier_vectors(frame):
    """Return carrier origin, normal and a stable in-plane basis."""

    if frame is None or frame.normal is None:
        return None
    origin = Vector(frame.origin.as_tuple())
    normal = Vector(frame.normal.as_tuple())
    if normal.length <= 1.0e-12:
        return None
    normal.normalize()
    if frame.basis_u is not None:
        basis_u = Vector(frame.basis_u.as_tuple())
        basis_u = basis_u - normal * basis_u.dot(normal)
    else:
        basis_u = Vector((1.0, 0.0, 0.0))
        if abs(normal.dot(basis_u)) > 0.9:
            basis_u = Vector((0.0, 1.0, 0.0))
        basis_u = basis_u - normal * basis_u.dot(normal)
    if basis_u.length <= 1.0e-12:
        return None
    basis_u.normalize()
    basis_v = normal.cross(basis_u)
    if basis_v.length <= 1.0e-12:
        return None
    basis_v.normalize()
    return origin, normal, basis_u, basis_v


def _carrier_plane(carrier_frames: CarrierFrameMap | None, face_id: FaceId):
    """Return ``(origin, normal)`` for the session carrier plane, if present."""

    if carrier_frames is None:
        return None
    frame = carrier_frames.get(face_id)
    vectors = _carrier_vectors(frame)
    if vectors is None:
        return None
    return vectors[0], vectors[1]


def _carrier_deviation(carrier_frames: CarrierFrameMap | None, face_id: FaceId) -> float | None:
    if carrier_frames is None:
        return None
    frame = carrier_frames.get(face_id)
    if frame is None or frame.normal is None:
        return None
    return float(frame.deviation)


def _carrier_polygon_2d(carrier_frames: CarrierFrameMap | None, face_id: FaceId):
    if carrier_frames is None:
        return None
    frame = carrier_frames.get(face_id)
    vectors = _carrier_vectors(frame)
    if vectors is None or not frame.vertices:
        return None
    origin, _normal, basis_u, basis_v = vectors
    return [
        (
            (Vector(vertex.as_tuple()) - origin).dot(basis_u),
            (Vector(vertex.as_tuple()) - origin).dot(basis_v),
        )
        for vertex in frame.vertices
    ]


def _orientation_2d(a, b, c) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment_2d(a, b, p) -> bool:
    return (
        abs(_orientation_2d(a, b, p)) <= 1.0e-12
        and min(a[0], b[0]) - 1.0e-12 <= p[0] <= max(a[0], b[0]) + 1.0e-12
        and min(a[1], b[1]) - 1.0e-12 <= p[1] <= max(a[1], b[1]) + 1.0e-12
    )


def _segments_intersect_2d(a, b, c, d) -> bool:
    orientations = (
        _orientation_2d(a, b, c),
        _orientation_2d(a, b, d),
        _orientation_2d(c, d, a),
        _orientation_2d(c, d, b),
    )
    if orientations[0] * orientations[1] < 0.0 and orientations[2] * orientations[3] < 0.0:
        return True
    return (
        (abs(orientations[0]) <= 1.0e-12 and _on_segment_2d(a, b, c))
        or (abs(orientations[1]) <= 1.0e-12 and _on_segment_2d(a, b, d))
        or (abs(orientations[2]) <= 1.0e-12 and _on_segment_2d(c, d, a))
        or (abs(orientations[3]) <= 1.0e-12 and _on_segment_2d(c, d, b))
    )


def _polygon_is_simple_nonzero(polygon) -> bool:
    if len(polygon) < 3:
        return False
    area = sum(
        polygon[index][0] * polygon[(index + 1) % len(polygon)][1]
        - polygon[(index + 1) % len(polygon)][0] * polygon[index][1]
        for index in range(len(polygon))
    )
    if abs(area) <= 1.0e-12:
        return False
    count = len(polygon)
    for first in range(count):
        second = (first + 1) % count
        for other in range(first + 1, count):
            other_next = (other + 1) % count
            if first == other or second == other or first == other_next or second == other_next:
                continue
            if _segments_intersect_2d(polygon[first], polygon[second], polygon[other], polygon[other_next]):
                return False
    return True


def _carrier_admissible(carrier_frames: CarrierFrameMap | None, face_id: FaceId) -> bool:
    polygon = _carrier_polygon_2d(carrier_frames, face_id)
    return polygon is not None and _polygon_is_simple_nonzero(polygon)


def _projected_point_inside_carrier(
    point: Vector,
    face: bmesh.types.BMFace,
    carrier_frames: CarrierFrameMap | None,
    face_id: FaceId,
) -> bool:
    carrier_plane = _carrier_plane(carrier_frames, face_id)
    vectors = _carrier_vectors(carrier_frames.get(face_id) if carrier_frames is not None else None)
    if (
        carrier_plane is None
        or vectors is None
        or not _carrier_admissible(carrier_frames, face_id)
        or not face.is_valid
    ):
        return False
    origin, normal = carrier_plane
    _vector_origin, _vector_normal, basis_u, basis_v = vectors
    polygon = [
        (
            (vertex.co - origin).dot(basis_u),
            (vertex.co - origin).dot(basis_v),
        )
        for vertex in face.verts
    ]
    if not _polygon_is_simple_nonzero(polygon):
        return False
    projected = point - normal * (point - origin).dot(normal)
    query = ((projected - origin).dot(basis_u), (projected - origin).dot(basis_v))
    for index, start in enumerate(polygon):
        if _on_segment_2d(start, polygon[(index + 1) % len(polygon)], query):
            return False
    inside = False
    for index, start in enumerate(polygon):
        end = polygon[(index + 1) % len(polygon)]
        if (start[1] > query[1]) != (end[1] > query[1]):
            crossing = (end[0] - start[0]) * (query[1] - start[1]) / (end[1] - start[1]) + start[0]
            if query[0] < crossing:
                inside = not inside
    return inside


def _distance_to_face_surface(point: Vector, face: bmesh.types.BMFace) -> float:
    distances = []
    for corner_a, corner_b, corner_c in _face_surface_triangles(face):
        closest = mathutils.geometry.closest_point_on_tri(point, corner_a, corner_b, corner_c)
        distances.append((closest - point).length)
    return min(distances, default=float("inf"))


def _point_is_non_near_face(point: Vector, face: bmesh.types.BMFace, tolerance: float) -> bool:
    if any(coordinates_match(vertex.co, point, tolerance) for vertex in face.verts):
        return False
    edge_limit = max(2.0 * tolerance, 1.0e-9)
    for edge in face.edges:
        if not edge.is_valid:
            continue
        distance, factor = stitch_common._point_segment_distance_and_factor(point, edge)
        if distance <= edge_limit and stitch_common._is_interior_edge_factor(factor, edge.calc_length(), tolerance):
            return False
    return True


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
    source_face_ids: tuple[FaceId, ...] = ()


@dataclass(frozen=True, slots=True)
class _InteriorNetworkSnapshot:
    """Immutable source graph captured before any target mutation (R-N1)."""

    vertices: tuple[int, ...]
    edges: tuple[tuple[int, int], ...]
    anchors: frozenset[int]
    rank: dict[int, tuple[int, int]]
    host_ids: dict[int, FaceId]
    network_vertices: frozenset[int]


@dataclass(frozen=True, slots=True)
class _InteriorNetworkPath:
    """One planner path, represented by source vertex keys."""

    vertices: tuple[int, ...]

    @property
    def edges(self) -> tuple[tuple[int, int], ...]:
        return tuple(zip(self.vertices, self.vertices[1:], strict=False))


@dataclass(frozen=True, slots=True)
class _InteriorNetworkPlan:
    """Deterministic decomposition of one or more interior network components."""

    paths: tuple[_InteriorNetworkPath, ...]
    edge_keys: frozenset[frozenset[int]]
    reason: str = ""


def _network_snapshot(
    source_vertex_by_key: dict[int, bmesh.types.BMVert],
    edge_records: list[tuple[int, int, set[FaceId]]],
    classification: dict,
    adjacency: dict[int, set[int]],
    occurrence_by_key: dict[int, int],
) -> _InteriorNetworkSnapshot:
    """Capture only source graph data used by the pure R-N1 planner."""

    interior = {key for key, value in classification.items() if value[0] == "interior"}
    network_components: list[set[int]] = []
    seen: set[int] = set()
    for start in sorted(interior, key=lambda key: (source_vertex_by_key[key].index, key)):
        if start in seen:
            continue
        component: set[int] = set()
        stack = [start]
        while stack:
            key = stack.pop()
            if key in component:
                continue
            component.add(key)
            seen.add(key)
            stack.extend(neighbour for neighbour in adjacency.get(key, ()) if neighbour in interior)
        if any(len(adjacency.get(key, ())) >= 3 for key in component):
            network_components.append(component)

    network_vertices = set().union(*network_components) if network_components else set()
    network_edges: list[tuple[int, int]] = []
    anchors: set[int] = set()
    host_ids = _host_ids_by_vertex(edge_records)
    for left, right, _target_ids in edge_records:
        if left not in network_vertices and right not in network_vertices:
            continue
        if left not in network_vertices:
            anchors.add(left)
        if right not in network_vertices:
            anchors.add(right)
        network_edges.append((left, right))
    vertices = set(anchors) | network_vertices
    # BMVert.index is assigned once by BMesh; the second component makes the
    # ordering total even for synthetic vertices with an unset/equal index.
    ordered = sorted(vertices, key=lambda key: (source_vertex_by_key[key].index, occurrence_by_key[key]))
    # Freeze occurrence once from the canonical source snapshot.  It is the
    # second rank component and is never recomputed after target mutation.
    rank = {key: (source_vertex_by_key[key].index, occurrence_by_key[key]) for key in ordered}
    network_edges.sort(key=lambda edge: (min(rank[edge[0]], rank[edge[1]]), max(rank[edge[0]], rank[edge[1]])))
    return _InteriorNetworkSnapshot(
        vertices=tuple(ordered),
        edges=tuple(network_edges),
        anchors=frozenset(anchors),
        rank=rank,
        host_ids={key: host_ids[key] for key in network_vertices if key in host_ids},
        network_vertices=frozenset(network_vertices),
    )


def _plan_interior_network(snapshot: _InteriorNetworkSnapshot) -> _InteriorNetworkPlan:
    """Pure, deterministic R-N1 path decomposition.

    The planner owns the R_V/U_E state.  It never consults or mutates BMesh;
    gate and apply therefore consume the same source snapshot semantics.
    """

    if not snapshot.edges:
        return _InteriorNetworkPlan((), frozenset(), "")
    if not snapshot.anchors or len(snapshot.anchors) < 2:
        return _InteriorNetworkPlan((), frozenset(), "interior network cannot reach two anchors")
    if set(snapshot.host_ids) != set(snapshot.network_vertices):
        return _InteriorNetworkPlan((), frozenset(), "interior network has no common host face ID (R-H1)")
    interior_adjacency: dict[int, set[int]] = defaultdict(set)
    for left, right in snapshot.edges:
        if left in snapshot.network_vertices and right in snapshot.network_vertices:
            interior_adjacency[left].add(right)
            interior_adjacency[right].add(left)
    seen: set[int] = set()
    for start in snapshot.network_vertices:
        if start in seen:
            continue
        component: set[int] = set()
        component_stack = [start]
        while component_stack:
            key = component_stack.pop()
            if key in component:
                continue
            component.add(key)
            seen.add(key)
            component_stack.extend(interior_adjacency[key])
        if len({snapshot.host_ids[key] for key in component}) != 1:
            return _InteriorNetworkPlan((), frozenset(), "interior network has no common host face ID (R-H1)")

    adjacency: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for edge_number, (left, right) in enumerate(snapshot.edges):
        adjacency[left].append((right, edge_number))
        adjacency[right].append((left, edge_number))
    for neighbours in adjacency.values():
        neighbours.sort(key=lambda item: (snapshot.rank[item[0]], item[1]))

    remaining = set(range(len(snapshot.edges)))
    realized = set(snapshot.anchors)
    paths: list[_InteriorNetworkPath] = []
    max_iterations = len(remaining)

    for _iteration in range(max_iterations):
        if not remaining:
            break
        candidates: dict[tuple[tuple[int, int], ...], _InteriorNetworkPath] = {}
        for start in sorted(realized, key=snapshot.rank.__getitem__):
            stack: list[tuple[int, tuple[int, ...], frozenset[int]]] = [(start, (start,), frozenset())]
            while stack:
                current, vertices, used = stack.pop()
                for neighbour, edge_number in adjacency.get(current, ()):
                    if edge_number not in remaining or edge_number in used:
                        continue
                    if neighbour in vertices:
                        continue
                    next_vertices = vertices + (neighbour,)
                    next_used = used | {edge_number}
                    if neighbour in realized and neighbour != start:
                        forward = tuple(snapshot.rank[key] for key in next_vertices)
                        backward = tuple(reversed(forward))
                        canonical = min(forward, backward)
                        candidates[canonical] = _InteriorNetworkPath(next_vertices)
                    elif neighbour not in realized:
                        stack.append((neighbour, next_vertices, next_used))
        if not candidates:
            return _InteriorNetworkPlan(
                tuple(paths),
                frozenset(edge for path in paths for edge in (frozenset(pair) for pair in path.edges)),
                "interior network has an unrealized edge with no admissible path",
            )
        selected = min(
            candidates.values(),
            key=lambda path: (
                len(path.edges),
                min(
                    tuple(snapshot.rank[key] for key in path.vertices),
                    tuple(reversed(tuple(snapshot.rank[key] for key in path.vertices))),
                ),
            ),
        )
        paths.append(selected)
        for left, right in selected.edges:
            edge_number = next(
                number
                for number, edge in enumerate(snapshot.edges)
                if number in remaining and set(edge) == {left, right}
            )
            remaining.remove(edge_number)
        realized.update(selected.vertices[1:-1])

    if remaining:
        return _InteriorNetworkPlan(
            tuple(paths),
            frozenset(edge for path in paths for edge in (frozenset(pair) for pair in path.edges)),
            "interior network planning exceeded its iteration bound",
        )
    return _InteriorNetworkPlan(
        tuple(paths),
        frozenset(edge for path in paths for edge in (frozenset(pair) for pair in path.edges)),
        "",
    )


def _source_vertex_occurrence(
    bm: bmesh.types.BMesh,
    source_vertex_by_key: dict[int, bmesh.types.BMVert],
) -> dict[int, int]:
    """Return one source-independent occurrence order from the BMesh snapshot."""

    source_keys = set(source_vertex_by_key)
    return {hash(vertex): position for position, vertex in enumerate(tuple(bm.verts)) if hash(vertex) in source_keys}


def _classify_reflected_vertices(
    source_vertex_by_key: dict[int, bmesh.types.BMVert],
    target_ids_by_vertex: dict[int, set[FaceId]],
    target_faces_by_id: dict[FaceId, list[bmesh.types.BMFace]],
    axis_index: int,
    tolerance: float,
    *,
    edge_records: list[tuple[int, int, set[FaceId]]] | None = None,
    source_face_ids_by_vertex: dict[int, set[FaceId]] | None = None,
    carrier_frames: CarrierFrameMap | None = None,
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
    host_ids = _host_ids_by_vertex(edge_records) if edge_records is not None else None
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
        host_id = host_ids.get(source_key) if host_ids is not None else None
        if host_id is None:
            return classification, "interior vertex has no unique singleton target face ID (R-H1)"
        if not _carrier_admissible(carrier_frames, host_id):
            return classification, "carrier face is not admissible for an interior host (R-H2)"
        source_face_ids = (source_face_ids_by_vertex or {}).get(source_key, set())
        if len(source_face_ids) != 1:
            return classification, "interior vertex has no unique source face ID"
        source_face_id = next(iter(source_face_ids))
        source_deviation = _carrier_deviation(carrier_frames, source_face_id)
        target_deviation = _carrier_deviation(carrier_frames, host_id)
        if source_deviation is None or target_deviation is None:
            return classification, "carrier frame is missing or degenerate for an interior host"
        effective_surface_limit = max(20.0 * tolerance, 2.5 * max(source_deviation, target_deviation))
        host_faces = [face for face in target_faces_by_id.get(host_id, ()) if face.is_valid]
        strict_faces = [face for face in host_faces if _point_strictly_inside_face(expected, face, tolerance)]
        if len(strict_faces) == 1:
            classification[source_key] = ("interior", None, None, 0.0, strict_faces[0], "")
            continue
        distance_faces = [
            face for face in host_faces if _distance_to_face_surface(expected, face) <= effective_surface_limit
        ]
        relaxed_faces = [
            face
            for face in distance_faces
            if _point_is_non_near_face(expected, face, tolerance)
            and _projected_point_inside_carrier(expected, face, carrier_frames, host_id)
        ]
        if len(relaxed_faces) == 1:
            classification[source_key] = ("interior", None, None, 0.0, relaxed_faces[0], "")
            continue
        if len(relaxed_faces) > 1:
            return classification, "ambiguous mirrored target faces for interior point"
        if strict_faces:
            return classification, "ambiguous mirrored target faces for interior point"
        if not distance_faces:
            return classification, "reflected interior point exceeds the carrier surface sanity bound (R-H3)"
        if not any(_point_is_non_near_face(expected, face, tolerance) for face in distance_faces):
            return classification, "reflected interior point is too close to a target boundary"
        return classification, "reflected interior point fails projected containment"
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
    *,
    edge_records: list[tuple[int, int, set[FaceId]]] | None = None,
    source_face_ids_by_vertex: dict[int, set[FaceId]] | None = None,
) -> tuple[list[_InteriorChain], str]:
    """Return chains after checking shape and common host IDs only (R-H5)."""

    interior_keys = {key for key, (kind, *_rest) in classification.items() if kind == "interior"}
    if not interior_keys:
        return [], ""
    if edge_records is None:
        return [], "interior chain host context is missing (R-H1)"

    for key in interior_keys:
        if len(adjacency.get(key, ())) != 2:
            # Degree-1 tip or branch (degree >= 3): not a simple chain.
            return [], "a reflected cut vertex is not on a target boundary edge"

    host_ids = _host_ids_by_vertex(edge_records)
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

        member_host_ids = {host_ids.get(key) for key in ordered}
        if len(member_host_ids) != 1 or None in member_host_ids:
            return [], "a reflected cut vertex is not on a target boundary edge"
        target_face_id = next(iter(member_host_ids))
        if target_face_id not in target_ids_by_vertex.get(
            end_a, set()
        ) or target_face_id not in target_ids_by_vertex.get(end_b, set()):
            return [], "a reflected cut vertex has no common target face ID"
        source_face_sets = [(source_face_ids_by_vertex or {}).get(key, set()) for key in ordered]
        source_face_ids = tuple(next(iter(face_ids)) for face_ids in source_face_sets if len(face_ids) == 1)

        chains.append(
            _InteriorChain(
                members=tuple(ordered),
                end_a=end_a,
                end_b=end_b,
                target_face_id=target_face_id,
                source_face_ids=source_face_ids,
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


def _refresh_face_split_context(
    face_id: FaceId,
    target_faces_by_id: dict[FaceId, list[bmesh.types.BMFace]],
    lineage_by_face: dict[bmesh.types.BMFace, object],
) -> None:
    """Keep the FaceId→live-face index and lineage map coherent after a split."""

    live_faces = [face for face in target_faces_by_id.get(face_id, ()) if face.is_valid]
    target_faces_by_id[face_id] = live_faces
    for face in live_faces:
        lineage_by_face.setdefault(face, object())


def _register_face_edges(
    face_layer,
    faces: Iterable[bmesh.types.BMFace],
    existing_edges,
    tolerance: float,
    registered_entries: dict[bmesh.types.BMEdge, tuple[tuple[int, int, int], tuple]] | None = None,
) -> None:
    if existing_edges is None:
        return
    for face in faces:
        if not face.is_valid:
            continue
        for edge in face.edges:
            if edge.is_valid:
                primary, coordinate_a, coordinate_b = stitch_common._canonical_edge_endpoints(
                    edge.verts[0].co,
                    edge.verts[1].co,
                    tolerance,
                )
                linked_ids = frozenset(FaceId(int(linked[face_layer])) for linked in edge.link_faces)
                entry = (None, coordinate_a, coordinate_b, linked_ids)
                if registered_entries is not None:
                    previous = registered_entries.get(edge)
                    if previous == (primary, entry):
                        continue
                    if previous is not None:
                        previous_primary, previous_entry = previous
                        bucket = existing_edges.get(previous_primary, [])
                        if previous_entry in bucket:
                            bucket.remove(previous_entry)
                        if not bucket:
                            existing_edges.pop(previous_primary, None)
                    existing_edges.setdefault(primary, []).append(entry)
                    registered_entries[edge] = (primary, entry)
                    continue
                stitch_common._register_edge_endpoint_pair(
                    existing_edges, edge.verts[0].co, edge.verts[1].co, tolerance, face_ids=linked_ids
                )


def _face_split_mutation(
    target_face: bmesh.types.BMFace,
    end_a: bmesh.types.BMVert,
    end_b: bmesh.types.BMVert,
    *,
    coords: list[tuple[float, float, float]] | None,
    face_layer,
    target_faces_by_id: dict[FaceId, list[bmesh.types.BMFace]],
    lineage_by_face: dict[bmesh.types.BMFace, object],
    existing_edges,
    realized_face_ids: set[FaceId],
    registered_entries: dict[bmesh.types.BMEdge, tuple[tuple[int, int, int], tuple]] | None,
    tolerance: float,
    selection_tracker: stitch_common._SelectionMutationTracker | None,
) -> tuple[bmesh.types.BMFace | None, str]:
    """Single mutation boundary for every direct network/chain face_split."""

    if not target_face.is_valid or end_a is end_b:
        return None, "could not split a target face"
    face_id = FaceId(int(target_face[face_layer]))
    lineage_token = lineage_by_face.get(target_face, object())
    if selection_tracker is not None:
        selection_tracker.add_vertex(end_a)
        selection_tracker.add_vertex(end_b)
        selection_tracker.add_face(target_face)
    try:
        # face_split rejects an empty/None coords argument outright and only
        # accepts plain float tuples, so normalize or omit it entirely.
        if coords:
            coordinate_tuples = [tuple(coordinate) for coordinate in coords]
            new_face, _new_loop = bmesh.utils.face_split(target_face, end_a, end_b, coords=coordinate_tuples)
        else:
            new_face, _new_loop = bmesh.utils.face_split(target_face, end_a, end_b)
    except (RuntimeError, ValueError) as exc:
        return None, f"could not split a target face: {exc}"
    if new_face is None or not new_face.is_valid:
        return None, "could not split a target face"
    lineage_by_face[target_face] = lineage_token
    lineage_by_face[new_face] = lineage_token
    target_faces_by_id.setdefault(face_id, []).append(new_face)
    _refresh_face_split_context(face_id, target_faces_by_id, lineage_by_face)
    realized_face_ids.add(face_id)
    _register_face_edges(
        face_layer,
        (target_face, new_face),
        existing_edges,
        tolerance,
        registered_entries,
    )
    if selection_tracker is not None:
        selection_tracker.add_face(target_face)
        selection_tracker.add_face(new_face)
    return new_face, ""


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
    *,
    selection_tracker: stitch_common._SelectionMutationTracker | None = None,
    target_faces_by_id: dict[FaceId, list[bmesh.types.BMFace]] | None = None,
    lineage_by_face: dict[bmesh.types.BMFace, object] | None = None,
    classification: dict | None = None,
    carrier_frames: CarrierFrameMap | None = None,
    registered_entries: dict[bmesh.types.BMEdge, tuple[tuple[int, int, int], tuple]] | None = None,
) -> tuple[int, int, str, dict | None]:
    """face_split one accepted interior chain. Returns created/already_present delta."""

    if target_faces_by_id is None or lineage_by_face is None or classification is None:
        return 0, 0, "interior chain realization context is missing (R-H4/R-P1)", existing_edges
    end_a = target_vertex_by_source_key[chain.end_a]
    end_b = target_vertex_by_source_key[chain.end_b]
    reflected_coords = [mirror_coordinate(source_vertex_by_key[member].co, axis_index) for member in chain.members]
    candidate_faces = sorted(
        (face for face in target_faces_by_id.get(chain.target_face_id, ()) if face.is_valid),
        key=lambda face: face.index,
    )
    path_keys = (*chain.members, chain.end_a, chain.end_b)
    candidate_lineages = {
        lineage_by_face.get(classification[key][4])
        for key in path_keys
        if classification is not None
        and classification.get(key, (None, None, None, None, None))[0] == "interior"
        and classification[key][4] is not None
    }
    has_interior_path_vertex = any(classification.get(key, (None,))[0] == "interior" for key in path_keys)
    if has_interior_path_vertex and (len(candidate_lineages) != 1 or None in candidate_lineages):
        return 0, 0, "interior chain classification lineages are inconsistent", existing_edges
    classification_lineage = next(iter(candidate_lineages), None)

    linked_faces = [face for face in candidate_faces if end_a in face.verts and end_b in face.verts]
    # A unique live instance with linked ends is the unconditional R-H4-1
    # route.  Geometry is consulted only for descendant ambiguity.
    if (
        len(candidate_faces) == 1
        and len(linked_faces) == 1
        and chain.target_face_id not in realized_face_ids
        and (classification is None or lineage_by_face.get(linked_faces[0]) == classification_lineage)
    ):
        target_face = linked_faces[0]
    elif not reflected_coords:
        linked_lineage_faces = [
            face
            for face in linked_faces
            if classification_lineage is None or lineage_by_face.get(face) == classification_lineage
        ]
        if len(linked_lineage_faces) != 1:
            return 0, 0, "mirrored network edge has no unique linked target face", existing_edges
        target_face = linked_lineage_faces[0]
    else:
        if carrier_frames is None:
            return 0, 0, "carrier frame is missing or degenerate for an interior host", existing_edges
        target_deviation = _carrier_deviation(carrier_frames, chain.target_face_id)
        if target_deviation is None or len(chain.source_face_ids) != len(chain.members):
            return 0, 0, "carrier frame is missing or degenerate for an interior host", existing_edges
        source_deviations = tuple(
            _carrier_deviation(carrier_frames, source_face_id) for source_face_id in chain.source_face_ids
        )
        if any(deviation is None for deviation in source_deviations):
            return 0, 0, "carrier frame is missing or degenerate for an interior host", existing_edges
        strict_faces = [
            face
            for face in candidate_faces
            if all(_point_strictly_inside_face(coordinate, face, tolerance) for coordinate in reflected_coords)
        ]
        if len(strict_faces) == 1:
            target_face = strict_faces[0]
        else:
            relaxed_faces = [
                face
                for face in candidate_faces
                if all(
                    _distance_to_face_surface(coordinate, face)
                    <= max(
                        20.0 * tolerance,
                        2.5
                        * max(
                            source_deviation,
                            target_deviation,
                        ),
                    )
                    and _point_is_non_near_face(coordinate, face, tolerance)
                    and _projected_point_inside_carrier(coordinate, face, carrier_frames, chain.target_face_id)
                    for coordinate, source_deviation in zip(reflected_coords, source_deviations, strict=True)
                )
            ]
            if len(relaxed_faces) != 1:
                return 0, 0, "could not place mirrored interior chain on a target face", existing_edges
            target_face = relaxed_faces[0]
        if target_face not in linked_faces:
            return 0, 0, "mirrored interior chain winner does not link both endpoints", existing_edges
        if classification is not None and lineage_by_face.get(target_face) != classification_lineage:
            return 0, 0, "mirrored interior chain winner has a different classification lineage", existing_edges

    if end_a is end_b:
        # Two path endpoints collapsed onto one target vertex (for example a
        # boundary split followed by an exact re-resolution within tolerance).
        return 0, 0, "mirrored chain endpoints collapse to one vertex", existing_edges
    existing_edge = bm.edges.get([end_a, end_b])
    # A pre-existing connector counts as already_present only when it bounds
    # the lineage-verified host face; an edge between the same vertex pair on
    # an unrelated same-ID instance must not satisfy this path's counting.
    if existing_edge is not None and existing_edge not in tuple(target_face.edges):
        existing_edge = None
    if not reflected_coords and existing_edge is not None:
        if existing_edges is not None:
            _register_face_edges(
                face_layer,
                existing_edge.link_faces,
                existing_edges,
                tolerance,
                registered_entries,
            )
        if selection_tracker is not None:
            selection_tracker.add_edge(existing_edge)
        existing_edge[marker_layer] = 0
        existing_edge.select = False
        return 0, 1, "", existing_edges

    new_face, split_reason = _face_split_mutation(
        target_face,
        end_a,
        end_b,
        coords=[tuple(coordinate) for coordinate in reflected_coords] or None,
        face_layer=face_layer,
        target_faces_by_id=target_faces_by_id,
        lineage_by_face=lineage_by_face,
        existing_edges=existing_edges,
        realized_face_ids=realized_face_ids,
        registered_entries=registered_entries,
        tolerance=tolerance,
        selection_tracker=selection_tracker,
    )
    if new_face is None:
        return 0, 0, split_reason or "could not split a target face for interior chain", existing_edges
    # The coords vertices lie exactly on the cut path both descendants share,
    # so they are the shared vertices minus the chain ends.  This stays local
    # to the two faces; snapshotting bm.verts/bm.edges around the split costs
    # seconds of proxy iteration on dense meshes.
    if not chain.members:
        edge = bm.edges.get([end_a, end_b])
        if edge is None:
            return 0, 0, "target face split made no edge", existing_edges
        edge[marker_layer] = 0
        edge.select = False
        if selection_tracker is not None:
            selection_tracker.add_edge(edge)
        return 1, 0, "", existing_edges

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
        if selection_tracker is not None:
            selection_tracker.add_vertex(chosen)
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
        if selection_tracker is not None:
            selection_tracker.add_edge(edge)
        edge.select = False
        for face in edge.link_faces:
            if selection_tracker is not None:
                selection_tracker.add_face(face)
            face.select = False
        created += 1
        if existing_edges is not None:
            stitch_common._register_edge_endpoint_pair(
                existing_edges,
                left.co,
                right.co,
                tolerance,
                face_ids={FaceId(int(face[face_layer])) for face in edge.link_faces},
            )
    return created, already, "", existing_edges


def _partition_wire_edges(
    source_edges: list[bmesh.types.BMEdge],
) -> tuple[list[bmesh.types.BMEdge], list[bmesh.types.BMEdge]]:
    """Split the path into wire edges (no link faces) and face edges (R-W1)."""

    wire_edges = [edge for edge in source_edges if not edge.link_faces]
    face_edges = [edge for edge in source_edges if edge.link_faces]
    return wire_edges, face_edges


def _wire_endpoint_candidates(
    bm: bmesh.types.BMesh,
    coordinate: Vector,
    tolerance: float,
) -> list[bmesh.types.BMVert]:
    return [vertex for vertex in bm.verts if vertex.is_valid and coordinates_match(vertex.co, coordinate, tolerance)]


def _wire_endpoints_resolvable(
    bm: bmesh.types.BMesh,
    wire_edges: list[bmesh.types.BMEdge],
    axis_index: int,
    tolerance: float,
) -> bool:
    for edge in wire_edges:
        for vertex in edge.verts:
            expected = mirror_coordinate(vertex.co, axis_index)
            if len(_wire_endpoint_candidates(bm, expected, tolerance)) > 1:
                return False
    return True


def _mirror_wire_edges(
    bm: bmesh.types.BMesh,
    wire_edges: list[bmesh.types.BMEdge],
    axis_index: int,
    tolerance: float,
    marker_layer,
    tracker: stitch_common._SelectionMutationTracker,
) -> tuple[int, int, str]:
    """Mirror dangling wire strokes as wires (R-W1). Runs after the face
    pipeline so endpoints it created are reusable for resolution."""

    created = 0
    already = 0
    for edge in wire_edges:
        resolved: list[bmesh.types.BMVert] = []
        pending_coords: list[Vector] = []
        for vertex in edge.verts:
            expected = mirror_coordinate(vertex.co, axis_index)
            candidates = _wire_endpoint_candidates(bm, expected, tolerance)
            if len(candidates) > 1:
                return created, already, "ambiguous mirrored wire endpoint"
            if candidates:
                resolved.append(candidates[0])
                pending_coords.append(None)
            else:
                resolved.append(None)
                pending_coords.append(expected)
        if resolved[0] is not None and resolved[0] is resolved[1]:
            return created, already, "a mirrored wire segment is degenerate"
        if resolved[0] is not None and resolved[1] is not None:
            existing = bm.edges.get([resolved[0], resolved[1]])
            if existing is not None:
                tracker.add_edge(existing)
                already += 1
                continue
        endpoints = []
        for vertex, expected in zip(resolved, pending_coords, strict=False):
            if vertex is None:
                vertex = bm.verts.new(expected)
                tracker.add_vertex(vertex)
                vertex.select = False
            endpoints.append(vertex)
        new_edge = bm.edges.new((endpoints[0], endpoints[1]))
        tracker.add_edge(new_edge)
        new_edge[marker_layer] = 0
        new_edge.select = False
        created += 1
    return created, already, ""


def reflected_path_uses_only_target_boundaries(
    bm: bmesh.types.BMesh,
    source_edges: Iterable[bmesh.types.BMEdge],
    axis_index: int,
    tolerance: float,
    mirror_face_ids: MirrorFaceMap,
    carrier_frames: CarrierFrameMap | None = None,
) -> bool:
    """Return whether the direct path builder supports every reflected vertex.

    A committed straight Knife segment and the loop-based tools terminate on
    existing face boundaries. Multi-click Knife strokes may also place an
    intentional face-interior terminal chain (degree-2 interior vertices whose
    ends resolve on one common target face). Those chains are accepted here;
    branching interior networks are planned and realized directly by R-N1.
    """

    source_edges = list(source_edges)
    face_layer = bm.faces.layers.int.get(FACE_ID_LAYER)
    if not source_edges or face_layer is None:
        return False

    wire_edges, face_edges = _partition_wire_edges(source_edges)
    if wire_edges and not _wire_endpoints_resolvable(bm, wire_edges, axis_index, tolerance):
        return False
    if not face_edges:
        return bool(wire_edges)

    (
        source_vertex_by_key,
        target_ids_by_vertex,
        edge_records,
        unmatched_face_ids,
        _status,
    ) = _collect_reflected_path_context(
        face_edges,
        face_layer,
        mirror_face_ids,
        require_all_mirrored=True,
    )
    if unmatched_face_ids or not source_vertex_by_key:
        return False

    needed_target_ids = {target_id for target_ids in target_ids_by_vertex.values() for target_id in target_ids}
    target_faces_by_id = _target_faces_by_id(bm, face_layer, needed_target_ids)
    source_face_ids_by_vertex = _source_face_ids_by_vertex(face_edges, face_layer)
    classification, reason = _classify_reflected_vertices(
        source_vertex_by_key,
        target_ids_by_vertex,
        target_faces_by_id,
        axis_index,
        tolerance,
        edge_records=edge_records,
        source_face_ids_by_vertex=source_face_ids_by_vertex,
        carrier_frames=carrier_frames,
    )
    if reason:
        return False

    adjacency = _path_adjacency(edge_records)
    occurrence_by_key = _source_vertex_occurrence(bm, source_vertex_by_key)
    network_snapshot = _network_snapshot(
        source_vertex_by_key, edge_records, classification, adjacency, occurrence_by_key
    )
    network_plan = _plan_interior_network(network_snapshot)
    network_vertices = set(network_snapshot.vertices) - set(network_snapshot.anchors)
    if network_plan.reason:
        return False
    simple_classification = {key: value for key, value in classification.items() if key not in network_vertices}
    _chains, chain_reason = _find_interior_chains(
        simple_classification,
        adjacency,
        source_vertex_by_key,
        target_ids_by_vertex,
        target_faces_by_id,
        face_layer,
        axis_index,
        tolerance,
        edge_records=edge_records,
        source_face_ids_by_vertex=source_face_ids_by_vertex,
    )
    return not chain_reason


@overload
def apply_reflected_path_topology(
    bm: bmesh.types.BMesh,
    source_edges: Iterable[bmesh.types.BMEdge],
    axis_index: int,
    tolerance: float,
    mirror_face_ids: MirrorFaceMap,
    carrier_frames: CarrierFrameMap | None = None,
    *,
    return_summary: Literal[False] = ...,
) -> tuple[int, int, str]: ...
@overload
def apply_reflected_path_topology(
    bm: bmesh.types.BMesh,
    source_edges: Iterable[bmesh.types.BMEdge],
    axis_index: int,
    tolerance: float,
    mirror_face_ids: MirrorFaceMap,
    carrier_frames: CarrierFrameMap | None = None,
    *,
    return_summary: Literal[True],
) -> tuple[int, int, str, stitch_common.SelectionMutationSummary]: ...
def apply_reflected_path_topology(
    bm: bmesh.types.BMesh,
    source_edges: Iterable[bmesh.types.BMEdge],
    axis_index: int,
    tolerance: float,
    mirror_face_ids: MirrorFaceMap,
    carrier_frames: CarrierFrameMap | None = None,
    *,
    return_summary: bool = False,
) -> tuple[int, int, str] | tuple[int, int, str, stitch_common.SelectionMutationSummary]:
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
    coordinate tolerance (same endpoint store used by the native mirror path) so a
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

    tracker = stitch_common._SelectionMutationTracker()

    def _result(created: int, already: int, reason: str):
        if return_summary:
            return created, already, reason, tracker.finish(complete=not reason)
        return created, already, reason

    source_edges = list(source_edges)
    if not source_edges:
        return _result(0, 0, "no source cut edges were supplied")

    marker_layer = bm.edges.layers.int.get(EDGE_ORIGINAL_LAYER)
    face_layer = bm.faces.layers.int.get(FACE_ID_LAYER)
    if marker_layer is None or face_layer is None:
        return _result(0, 0, "temporary topology markers are missing")

    # Wire (dangling) strokes have no carrier faces and mirror as wires after
    # the face pipeline (R-W1); the face pipeline below sees face edges only.
    wire_edges, face_edges = _partition_wire_edges(source_edges)

    # Capture every source-side relationship before modifying the target. This
    # also makes faces which touch the symmetry plane safe to process.
    (
        source_vertex_by_key,
        target_ids_by_vertex,
        edge_records,
        unmatched_face_ids,
        _status,
    ) = _collect_reflected_path_context(
        face_edges,
        face_layer,
        mirror_face_ids,
        require_all_mirrored=False,
    )

    if unmatched_face_ids:
        return _result(0, 0, f"{len(unmatched_face_ids)} source face(s) have no mirrored counterpart")
    if any(not target_ids for _a, _b, target_ids in edge_records):
        return _result(0, 0, "a source cut edge has no mirrored target face")

    needed_target_ids = {target_id for target_ids in target_ids_by_vertex.values() for target_id in target_ids}
    target_faces_by_id = _target_faces_by_id(bm, face_layer, needed_target_ids)
    source_face_ids_by_vertex = _source_face_ids_by_vertex(face_edges, face_layer)
    lineage_by_face: dict[bmesh.types.BMFace, object] = {
        face: object() for faces in target_faces_by_id.values() for face in faces if face.is_valid
    }
    classification, classify_reason = _classify_reflected_vertices(
        source_vertex_by_key,
        target_ids_by_vertex,
        target_faces_by_id,
        axis_index,
        tolerance,
        edge_records=edge_records,
        source_face_ids_by_vertex=source_face_ids_by_vertex,
        carrier_frames=carrier_frames,
    )
    if classify_reason:
        return _result(0, 0, classify_reason)

    adjacency = _path_adjacency(edge_records)
    occurrence_by_key = _source_vertex_occurrence(bm, source_vertex_by_key)
    network_snapshot = _network_snapshot(
        source_vertex_by_key, edge_records, classification, adjacency, occurrence_by_key
    )
    network_plan = _plan_interior_network(network_snapshot)
    network_vertices = set(network_snapshot.vertices) - set(network_snapshot.anchors)
    if network_plan.reason:
        return _result(0, 0, network_plan.reason)
    simple_classification = {key: value for key, value in classification.items() if key not in network_vertices}
    chains, chain_reason = _find_interior_chains(
        simple_classification,
        adjacency,
        source_vertex_by_key,
        target_ids_by_vertex,
        target_faces_by_id,
        face_layer,
        axis_index,
        tolerance,
        edge_records=edge_records,
        source_face_ids_by_vertex=source_face_ids_by_vertex,
    )
    if chain_reason:
        return _result(0, 0, chain_reason)

    chain_edge_keys = _chain_source_edge_keys(chains) | set(network_plan.edge_keys)
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
            return _result(0, 0, reason)

        assert target_edge is not None
        try:
            tracker.add_edge(target_edge)
            _new_edge, target_vertex = bmesh.utils.edge_split(
                target_edge,
                target_edge.verts[0],
                factor,
            )
        except (RuntimeError, ValueError) as exc:
            return _result(0, 0, f"could not split a mirrored target edge: {exc}")
        tracker.add_edge(_new_edge)
        tracker.add_vertex(target_vertex)
        target_vertex.co = expected
        target_vertex.select = False
        target_vertex_by_source_key[source_key] = target_vertex

    existing_edges: stitch_common._EdgeEndpointStore | None = None
    created_edges = 0
    already_present = 0

    realized_face_ids: set[FaceId] = set()
    registered_entries: dict[bmesh.types.BMEdge, tuple[tuple[int, int, int], tuple]] = {}
    if network_plan.paths:
        # Network mutation starts with a live endpoint store so every
        # face_split can update it through the common helper (R-N1).
        existing_edges = {}
        for faces in target_faces_by_id.values():
            _register_face_edges(face_layer, faces, existing_edges, tolerance, registered_entries)
    # The planner is consumed exactly once.  In particular, do not re-plan
    # after a face_split has changed BMVert/BMFace indices (R-N1).
    for planned_path in network_plan.paths:
        member_keys = planned_path.vertices[1:-1]
        target_face_id = next(
            (network_snapshot.host_ids[key] for key in planned_path.vertices if key in network_snapshot.host_ids),
            None,
        )
        if target_face_id is None:
            return _result(created_edges, already_present, "interior network has no common host face ID (R-H1)")
        network_chain = _InteriorChain(
            members=tuple(member_keys),
            end_a=planned_path.vertices[0],
            end_b=planned_path.vertices[-1],
            target_face_id=target_face_id,
            source_face_ids=tuple(
                next(iter(source_face_ids_by_vertex[key]))
                for key in member_keys
                if len(source_face_ids_by_vertex.get(key, ())) == 1
            ),
        )
        created_delta, already_delta, fail_reason, existing_edges = _realize_interior_chain(
            bm,
            network_chain,
            source_vertex_by_key,
            target_vertex_by_source_key,
            axis_index,
            tolerance,
            face_layer,
            marker_layer,
            existing_edges,
            realized_face_ids,
            selection_tracker=tracker,
            target_faces_by_id=target_faces_by_id,
            lineage_by_face=lineage_by_face,
            classification=classification,
            carrier_frames=carrier_frames,
            registered_entries=registered_entries,
        )
        if fail_reason:
            return _result(created_edges, already_present, fail_reason)
        created_edges += created_delta
        already_present += already_delta

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
            selection_tracker=tracker,
            target_faces_by_id=target_faces_by_id,
            lineage_by_face=lineage_by_face,
            classification=classification,
            carrier_frames=carrier_frames,
            registered_entries=registered_entries,
        )
        if fail_reason:
            return _result(created_edges, already_present, fail_reason)
        created_edges += created_delta
        already_present += already_delta

    # Endpoint-tol store matches the native mirror path so geometric duplicates
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
                tracker.add_edge(existing)
                existing_target_ids = {FaceId(int(face[face_layer])) for face in existing.link_faces}
                if not existing_target_ids.intersection(possible_target_ids):
                    return _result(
                        created_edges, already_present, "an existing mirrored edge is outside its target face"
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
                            stitch_common._register_edge_endpoint_pair(
                                existing_edges,
                                edge.verts[0].co,
                                edge.verts[1].co,
                                tolerance,
                                face_ids={FaceId(int(face[face_layer])) for face in edge.link_faces},
                            )

            endpoint_match = stitch_common._match_edge_endpoint_pair_for_faces(
                target_a.co,
                target_b.co,
                tolerance,
                existing_edges,
                possible_target_ids,
            )
            if endpoint_match == "ambiguous":
                return _result(
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
                tracker.add_face(candidate_faces[0])
                bmesh.utils.face_split(candidate_faces[0], target_a, target_b)
            except (RuntimeError, ValueError) as exc:
                return _result(created_edges, already_present, f"could not split a target face: {exc}")
            new_edge = bm.edges.get([target_a, target_b])
            if new_edge is None:
                return _result(created_edges, already_present, "target face split made no edge")
            tracker.add_edge(new_edge)
            new_edge[marker_layer] = 0
            new_edge.select = False
            for face in new_edge.link_faces:
                tracker.add_face(face)
                face.select = False
            assert existing_edges is not None
            stitch_common._register_edge_endpoint_pair(
                existing_edges,
                new_edge.verts[0].co,
                new_edge.verts[1].co,
                tolerance,
                face_ids={FaceId(int(face[face_layer])) for face in new_edge.link_faces},
            )
            created_edges += 1
            progress = True

        if deferred and not progress:
            return _result(created_edges, already_present, f"could not place {len(deferred)} mirrored cut segment(s)")
        pending = deferred

    if wire_edges:
        wire_created, wire_already, wire_reason = _mirror_wire_edges(
            bm,
            wire_edges,
            axis_index,
            tolerance,
            marker_layer,
            tracker,
        )
        created_edges += wire_created
        already_present += wire_already
        if wire_reason:
            return _result(created_edges, already_present, wire_reason)

    bm.normal_update()
    return _result(created_edges, already_present, "")
