from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

import bmesh
import mathutils.geometry
from mathutils import Vector

from . import stitch_common
from ._types import FaceId, MirrorFaceMap
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

    Only tris and quads are covered.  A quad diagonal is valid only when its
    two triangles wind consistently — on a concave or folded quad the other
    diagonal spans area outside the real surface and must not be offered.
    N-gons are not covered at all: their interior-chain acceptance is declined
    so those strokes keep using the projection fallback.
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


def _interior_faces_for_point(
    point: Vector,
    candidate_faces: set[bmesh.types.BMFace],
    tolerance: float,
) -> list[bmesh.types.BMFace]:
    return sorted(
        (face for face in candidate_faces if face.is_valid and _point_strictly_inside_face(point, face, tolerance)),
        key=lambda face: face.index,
    )


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


def _faces_supporting_resolution(
    kind: str,
    exact_vertex: bmesh.types.BMVert | None,
    target_edge: bmesh.types.BMEdge | None,
    interior_face: bmesh.types.BMFace | None,
    candidate_faces: set[bmesh.types.BMFace],
) -> set[bmesh.types.BMFace]:
    if kind == "exact":
        assert exact_vertex is not None
        return {face for face in exact_vertex.link_faces if face.is_valid and face in candidate_faces}
    if kind == "boundary":
        assert target_edge is not None
        return {face for face in target_edge.link_faces if face.is_valid and face in candidate_faces}
    if kind == "interior":
        assert interior_face is not None
        return {interior_face}
    return set()


def _classify_reflected_vertices(
    source_vertex_by_key: dict[int, bmesh.types.BMVert],
    target_ids_by_vertex: dict[int, set[FaceId]],
    target_faces_by_id: dict[FaceId, list[bmesh.types.BMFace]],
    axis_index: int,
    tolerance: float,
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
        interior_faces = _interior_faces_for_point(expected, candidate_faces, tolerance)
        if len(interior_faces) == 1:
            classification[source_key] = ("interior", None, None, 0.0, interior_faces[0], "")
            continue
        if len(interior_faces) > 1:
            return (
                classification,
                "ambiguous mirrored target faces for interior point",
            )
        return classification, reason
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
) -> tuple[list[_InteriorChain], str]:
    """Return accepted interior chains, or a non-empty reason on decline."""

    interior_keys = {key for key, (kind, *_rest) in classification.items() if kind == "interior"}
    if not interior_keys:
        return [], ""

    for key in interior_keys:
        if len(adjacency.get(key, ())) != 2:
            # Degree-1 tip or branch (degree >= 3): not a simple chain.
            return [], "a reflected cut vertex is not on a target boundary edge"

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

        # Common target face among chain members and both ends.
        common_faces: set[bmesh.types.BMFace] | None = None
        for key in (*ordered, end_a, end_b):
            kind, exact_vertex, target_edge, _factor, interior_face, _reason = classification[key]
            candidate_faces = {
                face
                for target_id in target_ids_by_vertex[key]
                for face in target_faces_by_id.get(target_id, ())
                if face.is_valid
            }
            supported = _faces_supporting_resolution(
                kind,
                exact_vertex,
                target_edge,
                interior_face,
                candidate_faces,
            )
            if common_faces is None:
                common_faces = set(supported)
            else:
                common_faces &= supported
        if not common_faces or len(common_faces) != 1:
            return [], "a reflected cut vertex is not on a target boundary edge"

        common_face = next(iter(common_faces))
        target_face_id = FaceId(int(common_face[face_layer]))
        # Every interior member must actually use this same face instance.
        for key in ordered:
            interior_face = classification[key][4]
            if interior_face is None or interior_face != common_face:
                return [], "a reflected cut vertex is not on a target boundary edge"

        chains.append(
            _InteriorChain(
                members=tuple(ordered),
                end_a=end_a,
                end_b=end_b,
                target_face_id=target_face_id,
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
) -> tuple[int, int, str, dict | None]:
    """face_split one accepted interior chain. Returns created/already_present delta."""

    end_a = target_vertex_by_source_key[chain.end_a]
    end_b = target_vertex_by_source_key[chain.end_b]
    reflected_coords = [mirror_coordinate(source_vertex_by_key[member].co, axis_index) for member in chain.members]
    candidate_faces = sorted(
        (
            face
            for face in set(end_a.link_faces).intersection(end_b.link_faces)
            if face.is_valid and FaceId(int(face[face_layer])) == chain.target_face_id
        ),
        key=lambda face: face.index,
    )
    # Descendants of prior splits inherit the parent face id, so several
    # candidates can share the chain's target id.  Only the face that strictly
    # contains every reflected interior coordinate is the correct host.
    #
    # The strict re-test cannot be applied to the untouched single-candidate
    # case: the boundary end splits have already turned the host quad into an
    # n-gon whose ear-clip triangulation deviates from the evaluated quad
    # surface by more than the tolerance-scale limit on curved meshes, so it
    # would falsely decline (classification accepted these coordinates on the
    # pre-split ancestor during this same apply call).  Once an earlier chain
    # has face_split this target id, a lone shared candidate may be the wrong
    # descendant, so only then is the strict containment test decisive.
    if len(candidate_faces) == 1 and chain.target_face_id not in realized_face_ids:
        target_face = candidate_faces[0]
    else:
        containing_faces = [
            face
            for face in candidate_faces
            if all(_point_strictly_inside_face(coordinate, face, tolerance) for coordinate in reflected_coords)
        ]
        if len(containing_faces) != 1:
            return 0, 0, "could not place mirrored interior chain on a target face", existing_edges
        target_face = containing_faces[0]

    try:
        if selection_tracker is not None:
            selection_tracker.add_vertex(end_a)
            selection_tracker.add_vertex(end_b)
            selection_tracker.add_face(target_face)
        new_face, _new_loop = bmesh.utils.face_split(
            target_face,
            end_a,
            end_b,
            coords=[tuple(coordinate) for coordinate in reflected_coords],
        )
    except (RuntimeError, ValueError) as exc:
        return 0, 0, f"could not split a target face for interior chain: {exc}", existing_edges
    if new_face is None or not new_face.is_valid or not target_face.is_valid:
        return 0, 0, "could not split a target face for interior chain", existing_edges
    realized_face_ids.add(chain.target_face_id)
    if selection_tracker is not None:
        selection_tracker.add_face(target_face)
        selection_tracker.add_face(new_face)

    # The coords vertices lie exactly on the cut path both descendants share,
    # so they are the shared vertices minus the chain ends.  This stays local
    # to the two faces; snapshotting bm.verts/bm.edges around the split costs
    # seconds of proxy iteration on dense meshes.
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


def reflected_path_uses_only_target_boundaries(
    bm: bmesh.types.BMesh,
    source_edges: Iterable[bmesh.types.BMEdge],
    axis_index: int,
    tolerance: float,
    mirror_face_ids: MirrorFaceMap,
) -> bool:
    """Return whether the direct path builder supports every reflected vertex.

    A committed straight Knife segment and the loop-based tools terminate on
    existing face boundaries. Multi-click Knife strokes may also place an
    intentional face-interior terminal chain (degree-2 interior vertices whose
    ends resolve on one common target face). Those chains are accepted here;
    interior networks with branches still use the Knife Project fallback.
    """

    source_edges = list(source_edges)
    face_layer = bm.faces.layers.int.get(FACE_ID_LAYER)
    if not source_edges or face_layer is None:
        return False

    (
        source_vertex_by_key,
        target_ids_by_vertex,
        edge_records,
        unmatched_face_ids,
        _status,
    ) = _collect_reflected_path_context(
        source_edges,
        face_layer,
        mirror_face_ids,
        require_all_mirrored=True,
    )
    if unmatched_face_ids or not source_vertex_by_key:
        return False

    needed_target_ids = {target_id for target_ids in target_ids_by_vertex.values() for target_id in target_ids}
    target_faces_by_id = _target_faces_by_id(bm, face_layer, needed_target_ids)
    classification, reason = _classify_reflected_vertices(
        source_vertex_by_key,
        target_ids_by_vertex,
        target_faces_by_id,
        axis_index,
        tolerance,
    )
    if reason:
        return False

    adjacency = _path_adjacency(edge_records)
    _chains, chain_reason = _find_interior_chains(
        classification,
        adjacency,
        source_vertex_by_key,
        target_ids_by_vertex,
        target_faces_by_id,
        face_layer,
        axis_index,
        tolerance,
    )
    return not chain_reason


def apply_reflected_path_topology(
    bm: bmesh.types.BMesh,
    source_edges: Iterable[bmesh.types.BMEdge],
    axis_index: int,
    tolerance: float,
    mirror_face_ids: MirrorFaceMap,
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
    coordinate tolerance (same store as :func:`build_reflected_cutter`) so a
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

    # Capture every source-side relationship before modifying the target. This
    # also makes faces which touch the symmetry plane safe to process.
    (
        source_vertex_by_key,
        target_ids_by_vertex,
        edge_records,
        unmatched_face_ids,
        _status,
    ) = _collect_reflected_path_context(
        source_edges,
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
    classification, classify_reason = _classify_reflected_vertices(
        source_vertex_by_key,
        target_ids_by_vertex,
        target_faces_by_id,
        axis_index,
        tolerance,
    )
    if classify_reason:
        return _result(0, 0, classify_reason)

    adjacency = _path_adjacency(edge_records)
    chains, chain_reason = _find_interior_chains(
        classification,
        adjacency,
        source_vertex_by_key,
        target_ids_by_vertex,
        target_faces_by_id,
        face_layer,
        axis_index,
        tolerance,
    )
    if chain_reason:
        return _result(0, 0, chain_reason)

    chain_edge_keys = _chain_source_edge_keys(chains)
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
        )
        if fail_reason:
            return _result(created_edges, already_present, fail_reason)
        created_edges += created_delta
        already_present += already_delta

    # Endpoint-tol store matches build_reflected_cutter so geometric duplicates
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
                    return _result(created_edges, already_present, "an existing mirrored edge is outside its target face")
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
                return _result(created_edges, already_present, "multiple coordinate-matching edges are ambiguous across target faces")
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

    bm.normal_update()
    return _result(created_edges, already_present, "")
