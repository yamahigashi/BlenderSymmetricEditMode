# SPDX-License-Identifier: GPL-3.0-or-later

"""Geometry helpers for ydd Symmetric Edit.

The interactive cut is deliberately left to Blender's native tools. This module
marks the pre-cut topology, identifies the edges Blender created, and mirrors
their topology directly on the opposite faces.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence
from typing import Literal, cast

import bmesh
from mathutils import Vector
from mathutils.kdtree import KDTree

from ._types import (
    Coordinate3D,
    EdgeMarkerId,
    EdgeSelectionHistory,
    FaceId,
    FaceKey,
    FaceMatchRecord,
    FaceSelectionHistory,
    HiddenFaceMap,
    MirrorFaceMap,
    MirrorOverlap,
    OverlapClassification,
    QuantizedCoordinate,
    SelectionHistory,
    SelectionSnapshot,
    TopologyPreparation,
    VertexSelectionHistory,
)

# Path length floor used by choose_source_side. Distinct from coordinate
# tolerance so short cuts remain classifiable when the user raises tolerance.
_MIN_SIDE_LENGTH = 1.0e-9

EDGE_ORIGINAL_LAYER = ".yse_original_edge"
FACE_ID_LAYER = ".yse_original_face_id"
FACE_MIRROR_ID_LAYER = ".yse_mirror_face_id"
FACE_HIDDEN_LAYER = ".yse_face_hidden"
HISTORY_TOKEN_LAYER = ".yse_history_token"
VERT_SELECTION_LAYER = ".yse_vertex_selection"
EDGE_SELECTION_LAYER = ".yse_edge_selection"
FACE_SELECTION_LAYER = ".yse_face_selection"
VERT_HIDDEN_LAYER = ".yse_vertex_hidden"
EDGE_HIDDEN_LAYER = ".yse_edge_hidden"
VERT_BACKUP_ID_LAYER = ".yse_backup_vertex_id"
VERT_RIP_ID_LAYER = ".yse_rip_vertex_id"
VERT_MERGE_GROUP_LAYER = ".yse_merge_group"

TEMP_LAYER_NAMES = (
    EDGE_ORIGINAL_LAYER,
    FACE_ID_LAYER,
    FACE_MIRROR_ID_LAYER,
    FACE_HIDDEN_LAYER,
    HISTORY_TOKEN_LAYER,
    VERT_SELECTION_LAYER,
    EDGE_SELECTION_LAYER,
    FACE_SELECTION_LAYER,
    VERT_HIDDEN_LAYER,
    EDGE_HIDDEN_LAYER,
    VERT_BACKUP_ID_LAYER,
    VERT_RIP_ID_LAYER,
    VERT_MERGE_GROUP_LAYER,
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

    Built by :func:`build_vertex_mirror_lookup`. Matching reuses the shared
    floor-bin neighborhood and Chebyshev verification (Phase 2a contract).
    """

    def __init__(
        self,
        *,
        axis_index: int,
        tolerance: float,
        bins: dict[QuantizedCoordinate, list[tuple[int, tuple[float, float, float]]]],
    ) -> None:
        self._axis_index = axis_index
        self._tolerance = tolerance
        self._bins = bins
        self._on_plane_indices: frozenset[int] | None = None

    def _on_plane_registered(self) -> frozenset[int]:
        """Registered indices whose stored coordinate lies on the plane."""

        cached = self._on_plane_indices
        if cached is None:
            axis_index = self._axis_index
            tolerance = self._tolerance
            cached = frozenset(
                index
                for entries in self._bins.values()
                for index, stored in entries
                if abs(stored[axis_index]) <= tolerance
            )
            self._on_plane_indices = cached
        return cached

    def find(self, co: Vector) -> int | None:
        """Return the index of the registered coord matching ``mirror(co)``.

        Probes the 27-bin neighborhood of the reflected point. Accepts only
        candidates whose Chebyshev distance is at most *tolerance*. When several
        qualify, returns the nearest (minimum max-axis distance).
        """

        expected = mirror_coordinate(co, self._axis_index)
        expected_coord = (float(expected[0]), float(expected[1]), float(expected[2]))
        tolerance = self._tolerance
        best: tuple[float, int] | None = None
        for bin_key in _iter_quantized_neighborhood(expected, tolerance):
            for index, stored in self._bins.get(bin_key, ()):
                # Acceptance is coordinates_match (Chebyshev <= tolerance);
                # the same distance doubles as the nearest-first ranking key.
                distance = _chebyshev_distance_3d(expected_coord, stored)
                if distance > tolerance:
                    continue
                if best is None or distance < best[0]:
                    best = (distance, index)
        return None if best is None else best[1]

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

        return tuple(self._resolve_injective(list(coords)))

    def _candidates_for(self, position) -> list[tuple[float, int]]:
        """Distance-ascending registered candidates within *tolerance*."""

        position_coord = (float(position[0]), float(position[1]), float(position[2]))
        tolerance = self._tolerance
        found = []
        for bin_key in _iter_quantized_neighborhood(position, tolerance):
            for index, stored in self._bins.get(bin_key, ()):
                distance = _chebyshev_distance_3d(position_coord, stored)
                if distance > tolerance:
                    continue
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

        if plane_side is None:
            candidate_lists = [list(self._candidates_for(position)) for position in positions]
        else:
            on_plane_targets = self._on_plane_registered()
            candidate_lists = [
                [
                    (distance, index)
                    for distance, index in self._candidates_for(position)
                    if (index in on_plane_targets) == plane_side
                ]
                for position in positions
            ]
        queries_by_target: dict[int, list[int]] = defaultdict(list)
        for query, candidates in enumerate(candidate_lists):
            for _distance, target in candidates:
                queries_by_target[target].append(query)

        results: list[int | None] = [None] * len(positions)
        visited = [False] * len(positions)
        for start in range(len(positions)):
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


# One step is one trial assignment of a candidate to a query.  Only a
# pathological mesh with many vertices packed inside one tolerance ball can
# reach this; such a component is rejected as ambiguous rather than guessed.
_INJECTIVE_STEP_LIMIT = 2_000


def _solve_injective_component(
    queries: Sequence[int],
    candidate_lists: Sequence[Sequence[tuple[float, int]]],
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

    Each coordinate is stored under its primary floor bin. ``find`` probes the
    27-bin neighborhood and verifies Chebyshev distance ≤ *tolerance* against
    ``mirror_coordinate(co, axis_index)``.
    """

    bins: dict[QuantizedCoordinate, list[tuple[int, tuple[float, float, float]]]] = defaultdict(list)
    for index, co in enumerate(coords):
        primary = _quantized_coordinate(co, tolerance)
        bins[primary].append((index, (float(co[0]), float(co[1]), float(co[2]))))
    return VertexMirrorLookup(axis_index=axis_index, tolerance=tolerance, bins=bins)


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

    lookup = build_vertex_mirror_lookup(coords, axis_index, tolerance)
    assigned = lookup.find_all_mirrored(coords)
    pairs: dict[int, int] = {}
    for source, target in enumerate(assigned):
        if target is None:
            continue
        if target == source or assigned[target] == source:
            pairs[source] = target
    return pairs


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


def prepare_topology(
    bm: bmesh.types.BMesh,
    axis_index: int,
    tolerance: float,
    history_token: int = 0,
    *,
    mark_vertex_ids: bool = False,
) -> TopologyPreparation:
    """Mark original topology and calculate mirrored face correspondences.

    Returns the captured face maps and matching totals.
    Split edges and faces inherit these integer layers in BMesh.  Edges made
    through a face by Knife start with zero and can therefore be identified
    after the native modal operator finishes.
    """

    remove_temporary_layers(bm)

    edge_layer = bm.edges.layers.int.new(EDGE_ORIGINAL_LAYER)
    edge_hidden_layer = bm.edges.layers.int.new(EDGE_HIDDEN_LAYER)
    face_layer = bm.faces.layers.int.new(FACE_ID_LAYER)
    face_mirror_layer = bm.faces.layers.int.new(FACE_MIRROR_ID_LAYER)
    face_hidden_layer = bm.faces.layers.int.new(FACE_HIDDEN_LAYER)
    history_token_layer = bm.faces.layers.int.new(HISTORY_TOKEN_LAYER)
    vertex_hidden_layer = bm.verts.layers.int.new(VERT_HIDDEN_LAYER)
    vertex_rip_id_layer = bm.verts.layers.int.new(VERT_RIP_ID_LAYER) if mark_vertex_ids else None

    # Adding a CustomData layer can invalidate previously held BMesh wrappers,
    # so all elements are intentionally acquired only after both layers exist.
    for edge_id, edge in enumerate(bm.edges, start=1):
        edge[edge_layer] = edge_id
        edge[edge_hidden_layer] = int(edge.hide)
    for vertex_id, vertex in enumerate(bm.verts, start=1):
        vertex[vertex_hidden_layer] = int(vertex.hide)
        if vertex_rip_id_layer is not None:
            vertex[vertex_rip_id_layer] = vertex_id

    # Primary face correspondence derives from the involutive vertex pair
    # table so it is symmetric (A→B implies B→A) and injective by
    # construction; per-face independent candidate picking could map two
    # sources onto one target.  Geometry matching remains the fallback for
    # faces with unpaired vertices.
    bm.verts.ensure_lookup_table()
    bm.verts.index_update()
    vertex_pairs = build_vertex_pair_table(
        tuple(vertex.co.copy() for vertex in bm.verts),
        axis_index,
        tolerance,
    )

    hidden_by_face_id: HiddenFaceMap = {}
    key_to_face_ids: dict[FaceKey, list[FaceId]] = defaultdict(list)
    face_records: dict[FaceId, FaceMatchRecord] = {}
    face_vertex_ids: dict[FaceId, tuple[int, ...]] = {}
    face_ids_by_vertex_set: dict[frozenset[int], list[FaceId]] = defaultdict(list)
    # Lazy fallback index (contract R2-3): only materialize on first exact miss.
    face_coords: dict[FaceId, tuple[tuple[float, float, float], ...]] = {}
    faces_by_count_centroid: dict[tuple[int, QuantizedCoordinate], list[FaceId]] = defaultdict(list)
    fallback_index_ready = False

    for raw_face_id, face in enumerate(bm.faces, start=1):
        face_id = FaceId(raw_face_id)
        face[face_layer] = int(face_id)
        face[face_hidden_layer] = int(face.hide)
        face[history_token_layer] = history_token
        hidden_by_face_id[face_id] = bool(face.hide)
        key = _face_key(face, axis_index, tolerance, mirrored=False)
        mirrored_key = _face_key(face, axis_index, tolerance, mirrored=True)
        centroid_vector = face.calc_center_median()
        record = FaceMatchRecord(
            key=key,
            mirrored_key=mirrored_key,
            centroid=_coordinate_3d(centroid_vector),
        )
        face_records[face_id] = record
        key_to_face_ids[record.key].append(face_id)
        vertex_ids = tuple(vertex.index for vertex in face.verts)
        face_vertex_ids[face_id] = vertex_ids
        face_ids_by_vertex_set[frozenset(vertex_ids)].append(face_id)

    def _ensure_fallback_face_index() -> None:
        nonlocal fallback_index_ready
        if fallback_index_ready:
            return
        for face in bm.faces:
            face_id = FaceId(int(face[face_layer]))
            coords = tuple((float(vertex.co[0]), float(vertex.co[1]), float(vertex.co[2])) for vertex in face.verts)
            face_coords[face_id] = coords
            # Primary centroid bin only; lookup probes the neighborhood (contract E).
            centroid_vector = Vector(face_records[face_id].centroid.as_tuple())
            faces_by_count_centroid[(len(coords), _quantized_coordinate(centroid_vector, tolerance))].append(face_id)
        fallback_index_ready = True

    def _mirror_candidates(face_id: FaceId, record: FaceMatchRecord) -> list[FaceId]:
        exact = key_to_face_ids.get(record.mirrored_key)
        if exact:
            return list(exact)

        # Bin-boundary fallback: same vertex count, centroid within tolerance
        # neighborhood, full vertex multiset within per-component tolerance.
        _ensure_fallback_face_index()
        vertex_count = record.key.vertex_count
        mirrored_centroid = mirror_coordinate(Vector(record.centroid.as_tuple()), axis_index)
        mirrored_coords = tuple(
            (
                float(mirrored[0]),
                float(mirrored[1]),
                float(mirrored[2]),
            )
            for mirrored in (mirror_coordinate(Vector(coordinate), axis_index) for coordinate in face_coords[face_id])
        )
        found: list[FaceId] = []
        seen: set[FaceId] = set()
        found_self = False
        found_other = False
        for centroid_key in _iter_quantized_neighborhood(mirrored_centroid, tolerance):
            for candidate_id in faces_by_count_centroid.get((vertex_count, centroid_key), ()):
                if candidate_id in seen:
                    continue
                seen.add(candidate_id)
                if not _coords_match_chebyshev(mirrored_coords, face_coords[candidate_id], tolerance):
                    continue
                found.append(candidate_id)
                # Consumer only needs one self and one non-self match (R2-4).
                if candidate_id == face_id:
                    found_self = True
                else:
                    found_other = True
                if found_self and found_other:
                    return found
        return found

    mirror_face_ids: MirrorFaceMap = {}
    geometric_fallback: list[tuple[FaceId, FaceMatchRecord]] = []
    for face_id, record in face_records.items():
        vertex_ids = face_vertex_ids[face_id]
        mapped = []
        for vertex_id in vertex_ids:
            partner = vertex_pairs.get(vertex_id)
            if partner is None:
                break
            mapped.append(partner)
        if len(mapped) == len(vertex_ids):
            # Both the source's own vertex set and the mapped set must be
            # unique face keys.  With duplicate coincident faces (R1, R2 over
            # one vertex set) the mapped-set lookup alone would send both to
            # the same counterpart while their own ambiguity goes unnoticed.
            own_faces = face_ids_by_vertex_set.get(frozenset(vertex_ids), ())
            counterparts = face_ids_by_vertex_set.get(frozenset(mapped), ())
            if len(own_faces) == 1 and len(counterparts) == 1:
                mirror_face_ids[face_id] = counterparts[0]
                continue
        geometric_fallback.append((face_id, record))

    fallback_assignments: dict[FaceId, FaceId] = {}
    for face_id, record in geometric_fallback:
        candidates = _mirror_candidates(face_id, record)
        if not candidates:
            continue

        # Duplicate coincident faces are unusual.  Prefer a different face for
        # an off-plane source; otherwise a center-spanning face maps to itself.
        source_is_off_plane = abs(record.centroid.component(axis_index)) > tolerance
        counterpart = candidates[0]
        if source_is_off_plane and counterpart == face_id:
            counterpart = next(
                (candidate for candidate in candidates if candidate != face_id),
                counterpart,
            )
        fallback_assignments[face_id] = counterpart

    # Injectivity check for the fallback layer: an entry whose target is
    # already taken (or contested by another fallback entry) is demoted to
    # unmatched instead of silently duplicating.
    pair_table_targets = set(mirror_face_ids.values())
    fallback_target_counts: dict[FaceId, int] = defaultdict(int)
    for counterpart in fallback_assignments.values():
        fallback_target_counts[counterpart] += 1
    for face_id, counterpart in fallback_assignments.items():
        if counterpart in pair_table_targets or fallback_target_counts[counterpart] > 1:
            continue
        mirror_face_ids[face_id] = counterpart

    # Defensive whole-map verification, origin-agnostic: if any target is
    # still referenced twice, drop every colliding entry.  Demotions show up
    # in matched_faces / total_faces.
    final_target_counts: dict[FaceId, int] = defaultdict(int)
    for counterpart in mirror_face_ids.values():
        final_target_counts[counterpart] += 1
    if any(count > 1 for count in final_target_counts.values()):
        mirror_face_ids = {
            face_id: counterpart
            for face_id, counterpart in mirror_face_ids.items()
            if final_target_counts[counterpart] == 1
        }

    for face in bm.faces:
        face_id = FaceId(int(face[face_layer]))
        mirror_face_id = mirror_face_ids.get(face_id)
        face[face_mirror_layer] = int(mirror_face_id) if mirror_face_id is not None else 0

    return TopologyPreparation(
        mirror_face_ids=mirror_face_ids,
        hidden_by_face_id=hidden_by_face_id,
        matched_faces=len(mirror_face_ids),
        total_faces=len(face_records),
    )


def get_required_layers(bm: bmesh.types.BMesh):
    return (
        bm.edges.layers.int.get(EDGE_ORIGINAL_LAYER),
        bm.faces.layers.int.get(FACE_ID_LAYER),
    )


def _edge_side(
    edge: bmesh.types.BMEdge,
    axis_index: int,
    tolerance: float,
) -> str:
    """Classify an edge relative to the mirror plane.

    Always returns one of POSITIVE / NEGATIVE / PLANE / CROSSES.
    """

    a = edge.verts[0].co[axis_index]
    b = edge.verts[1].co[axis_index]
    if a >= -tolerance and b >= -tolerance and max(a, b) > tolerance:
        return "POSITIVE"
    if a <= tolerance and b <= tolerance and min(a, b) < -tolerance:
        return "NEGATIVE"
    if abs(a) <= tolerance and abs(b) <= tolerance:
        return "PLANE"
    return "CROSSES"


def choose_source_side(
    path_edges: Iterable[bmesh.types.BMEdge],
    axis_index: int,
    tolerance: float,
    requested: str,
) -> tuple[str | None, int]:
    """Resolve AUTO and return ``(side, crossing_edge_count)``.

    Used by Loop Cut / Offset Edge Loop Cut (one-side selection). Knife no
    longer chooses a source side; see :func:`collect_knife_path_edges_by_side`.
    """

    positive_length = 0.0
    negative_length = 0.0
    crossing = 0
    for edge in path_edges:
        side = _edge_side(edge, axis_index, tolerance)
        if side == "POSITIVE":
            positive_length += edge.calc_length()
        elif side == "NEGATIVE":
            negative_length += edge.calc_length()
        elif side == "CROSSES":
            crossing += 1

    if requested in {"POSITIVE", "NEGATIVE"}:
        return requested, crossing
    if positive_length <= _MIN_SIDE_LENGTH and negative_length <= _MIN_SIDE_LENGTH:
        return None, crossing
    if positive_length >= negative_length:
        return "POSITIVE", crossing
    return "NEGATIVE", crossing


def _is_path_edge_by_markers(edge: bmesh.types.BMEdge, edge_layer, face_layer) -> bool:
    """True when *edge* is a native cut fragment (contract §2.0 novelty).

    Tag==0 is the primary signal. Existing-edge splits inherit a non-zero parent
    tag, so also accept internal edges whose link faces all share one original
    FACE_ID (FACE_ID complement). Selection is intentionally not consulted.
    """

    if edge[edge_layer] == 0:
        return True
    return (
        face_layer is not None
        and len(edge.link_faces) >= 2
        and len({FaceId(int(face[face_layer])) for face in edge.link_faces}) == 1
    )


def _discover_path_edges(
    bm: bmesh.types.BMesh,
    *,
    selected_only: bool = False,
) -> list[bmesh.types.BMEdge]:
    """Discover native path edges created by the last cut tool.

    Loop Cut and Offset Edge Loop Cut expose their complete native result as
    the current edge selection. This is more authoritative for those tools
    than CustomData inheritance on complex rings. Knife strokes can honor
    "Select Result" being disabled, so their marker-based path stays intact.

    Both branches use the same novelty test (tag==0 or FACE_ID complement;
    contract §2.0). *selected_only* only adds the selection filter for Loop
    Cut / Offset; Knife (selected_only=False) must still recover inherited-tag
    CROSSES fragments so the whole-stage decline cannot be skipped.
    """

    edge_layer = bm.edges.layers.int.get(EDGE_ORIGINAL_LAYER)
    if edge_layer is None:
        return []

    face_layer = bm.faces.layers.int.get(FACE_ID_LAYER)
    if selected_only:
        return [edge for edge in bm.edges if edge.select and _is_path_edge_by_markers(edge, edge_layer, face_layer)]
    return [edge for edge in bm.edges if _is_path_edge_by_markers(edge, edge_layer, face_layer)]


def path_ring_includes_pre_hidden_edges(bm: bmesh.types.BMesh) -> bool:
    """True when the Loop Cut / Offset ring includes a pre-hidden edge.

    Native ``loopcut`` skips hidden ring edges and yields a *partial* (open)
    path. A closed selected path means the cut ring was complete, so unrelated
    hidden geometry on another ring must not decline. An open path together
    with a pre-hidden edge in the face-neighbourhood of a path endpoint is the
    partial-ring case that must decline.
    """

    edge_hidden = bm.edges.layers.int.get(EDGE_HIDDEN_LAYER)
    edge_layer = bm.edges.layers.int.get(EDGE_ORIGINAL_LAYER)
    if edge_hidden is None or edge_layer is None:
        return False

    face_layer = bm.faces.layers.int.get(FACE_ID_LAYER)
    path_edges = [edge for edge in bm.edges if edge.select and _is_path_edge_by_markers(edge, edge_layer, face_layer)]
    if not path_edges:
        return False

    degree: dict[int, int] = defaultdict(int)
    vert_by_key: dict[int, bmesh.types.BMVert] = {}
    for edge in path_edges:
        for vertex in edge.verts:
            key = hash(vertex)
            degree[key] += 1
            vert_by_key[key] = vertex
    # Closed loop: every path vertex is incident to exactly two path edges.
    if degree and all(count == 2 for count in degree.values()):
        return False

    endpoint_verts = [vert_by_key[key] for key, count in degree.items() if count == 1]
    if not endpoint_verts:
        return False

    # BFS over faces around path endpoints: the skipped (hidden) ring edges
    # sit in the gap adjacent to the open ends.
    seen_faces: set[int] = set()
    face_queue: list[bmesh.types.BMFace] = []
    for vertex in endpoint_verts:
        for face in vertex.link_faces:
            if face.is_valid and face.index not in seen_faces:
                seen_faces.add(face.index)
                face_queue.append(face)

    # Expand one adjacency step so a one-edge gap still reaches the hidden edge.
    for face in list(face_queue):
        for edge in face.edges:
            for other in edge.link_faces:
                if other.is_valid and other.index not in seen_faces:
                    seen_faces.add(other.index)
                    face_queue.append(other)

    for face in face_queue:
        for edge in face.edges:
            if edge.is_valid and edge[edge_hidden]:
                return True
    return False


def classify_path_edges_by_side(
    path_edges: Iterable[bmesh.types.BMEdge],
    axis_index: int,
    tolerance: float,
) -> dict[str, list[bmesh.types.BMEdge]]:
    """Bucket path edges into POSITIVE / NEGATIVE / PLANE / CROSSES."""

    by_side: dict[str, list[bmesh.types.BMEdge]] = {
        "POSITIVE": [],
        "NEGATIVE": [],
        "PLANE": [],
        "CROSSES": [],
    }
    for edge in path_edges:
        by_side[_edge_side(edge, axis_index, tolerance)].append(edge)
    return by_side


def collect_knife_path_edges_by_side(
    bm: bmesh.types.BMesh,
    axis_index: int,
    tolerance: float,
) -> tuple[dict[str, list[bmesh.types.BMEdge]], int]:
    """Classify every new Knife path edge without choosing a source side.

    Phase 1/2 (axis-crossing contract §4.1): both POSITIVE and NEGATIVE
    buckets are mirrored toward each other. PLANE edges are shared.
    CROSSES edges are p-stitched (Phase 2) before the half-edges join the
    POSITIVE/NEGATIVE mirror path.

    Returns ``(by_side, total_path_edge_count)``.
    """

    all_path_edges = _discover_path_edges(bm, selected_only=False)
    by_side = classify_path_edges_by_side(all_path_edges, axis_index, tolerance)
    return by_side, len(all_path_edges)


def is_self_mirrored_edge(
    edge: bmesh.types.BMEdge,
    axis_index: int,
    tolerance: float,
) -> bool:
    """True when the edge endpoints are a mirror pair (ρ(s) = s, no X needed)."""

    a = edge.verts[0].co
    b = edge.verts[1].co
    return coordinates_match(a, mirror_coordinate(b, axis_index), tolerance) and coordinates_match(
        b,
        mirror_coordinate(a, axis_index),
        tolerance,
    )


def plane_intersection_of_edge(
    edge: bmesh.types.BMEdge,
    axis_index: int,
) -> tuple[Vector, float] | None:
    """Return ``(p, factor_from_vert0)`` where the edge meets the mirror plane.

    *p* is snapped so ``p[axis_index] == 0``. Returns ``None`` when the edge is
    parallel to the plane or the intersection is outside the segment.
    """

    a = edge.verts[0].co
    b = edge.verts[1].co
    ax = float(a[axis_index])
    bx = float(b[axis_index])
    denom = ax - bx
    if abs(denom) <= 1.0e-30:
        return None
    # a + t*(b-a) has axis component 0 ⇒ t = ax / (ax - bx)
    factor = ax / denom
    if factor < 0.0 or factor > 1.0:
        return None
    point = a.lerp(b, factor)
    point[axis_index] = 0.0
    return point, factor


def cluster_points_by_tolerance(
    points: Sequence[Vector],
    tolerance: float,
) -> list[list[int]]:
    """§1.1 clustering: lex-order scan, absorb within tol of the representative.

    Returns clusters as lists of indices into *points*. The first index of each
    cluster is the representative (lex-first member). Member-to-rep distance is
    at most *tolerance*; diameter among members can reach 2·tol.
    """

    order = sorted(
        range(len(points)),
        key=lambda index: (
            float(points[index][0]),
            float(points[index][1]),
            float(points[index][2]),
        ),
    )
    clusters: list[list[int]] = []
    for index in order:
        placed = False
        for cluster in clusters:
            representative = points[cluster[0]]
            if coordinates_match(points[index], representative, tolerance):
                cluster.append(index)
                placed = True
                break
        if not placed:
            clusters.append([index])
    return clusters


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
    asymmetric until the X is completed (contract §4.1-2 ρ(F)=F). Asymmetric
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


def apply_crosses_p_stitch(
    bm: bmesh.types.BMesh,
    crosses_edges: Iterable[bmesh.types.BMEdge],
    axis_index: int,
    tolerance: float,
) -> tuple[int, str]:
    """Split non-self-mirrored CROSSES edges at their plane intersection *p*.

    All intersections are collected before any mutation, clustered, then
    applied with priority (existing vertex → existing edge split → new via
    edge_split); mutating mid-collection would shift later intersections.
    Self-mirrored CROSSES (endpoints are a mirror pair) are left untouched.

    Returns ``(stitched_edge_count, failure_reason)``. On failure the mesh may
    already be partially mutated; callers must roll back the whole stage.
    """

    edges = [edge for edge in crosses_edges if edge.is_valid]
    if not edges:
        return 0, ""

    # (i) Collect plane intersections before any mutation.
    records: list[tuple[bmesh.types.BMEdge, Vector, float]] = []
    for edge in edges:
        if is_self_mirrored_edge(edge, axis_index, tolerance):
            continue
        intersection = plane_intersection_of_edge(edge, axis_index)
        if intersection is None:
            return 0, "a cross-plane cut segment has no plane intersection"
        point, factor = intersection
        # Degenerate: intersection already at an endpoint within tol → treat as
        # already on-plane (no split); skip so POSITIVE/NEGATIVE reclassification
        # after a previous stitch can re-bucket cleanly.
        if any(coordinates_match(vertex.co, point, tolerance) for vertex in edge.verts):
            continue
        records.append((edge, point, factor))

    if not records:
        return 0, ""

    points = [point for _edge, point, _factor in records]
    clusters = cluster_points_by_tolerance(points, tolerance)

    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()

    stitched = 0
    for cluster_indices in clusters:
        representative = points[cluster_indices[0]]
        member_edges = [records[index][0] for index in cluster_indices]

        plan_vertex, host_split, reason = _plan_plane_stitch_vertex(
            bm,
            representative,
            member_edges,
            axis_index,
            tolerance,
        )
        if reason:
            return stitched, reason

        vertex = plan_vertex
        if host_split is not None:
            host_edge, factor = host_split
            try:
                _new_edge, vertex = bmesh.utils.edge_split(host_edge, host_edge.verts[0], factor)
            except (RuntimeError, ValueError) as exc:
                return stitched, f"could not split a host edge at the knife stitch point: {exc}"
        if vertex is None:
            # Priority (3): create p by splitting the lex-first member edge.
            seed = _lex_first_edge(member_edges, axis_index)
            intersection = plane_intersection_of_edge(seed, axis_index)
            if intersection is None:
                return stitched, "a cross-plane cut segment has no plane intersection"
            _point, factor = intersection
            try:
                _new_edge, vertex = bmesh.utils.edge_split(seed, seed.verts[0], factor)
            except (RuntimeError, ValueError) as exc:
                return stitched, f"could not split a cross-plane cut at the mirror plane: {exc}"
            stitched += 1

        assert vertex is not None
        vertex.co = representative.copy()
        vertex.co[axis_index] = 0.0
        vertex.select = False

        for edge in member_edges:
            if not edge.is_valid:
                return stitched, "a cross-plane cut edge was lost during p-stitch"
            if any(endpoint == vertex for endpoint in edge.verts):
                continue
            if any(coordinates_match(endpoint.co, vertex.co, tolerance) for endpoint in edge.verts):
                for endpoint in edge.verts:
                    if coordinates_match(endpoint.co, vertex.co, tolerance) and endpoint != vertex:
                        try:
                            # pointmerge keeps verts[0] as the survivor
                            # (bmo_pointmerge_exec); put the cluster representative first.
                            bmesh.ops.pointmerge(bm, verts=[vertex, endpoint], merge_co=vertex.co)
                        except (RuntimeError, ValueError) as exc:
                            return stitched, f"could not merge plane-stitch vertices: {exc}"
                        break
                continue

            recomputed = plane_intersection_of_edge(edge, axis_index)
            if recomputed is None:
                return stitched, "a cross-plane cut segment lost its plane intersection"
            _point, factor = recomputed
            try:
                _new_edge, new_vertex = bmesh.utils.edge_split(edge, edge.verts[0], factor)
            except (RuntimeError, ValueError) as exc:
                return stitched, f"could not split a cross-plane cut at the mirror plane: {exc}"
            new_vertex.co = vertex.co.copy()
            new_vertex.select = False
            if new_vertex != vertex:
                try:
                    # Survivor first: keeps *vertex* valid across ≥3 merges in
                    # one cluster (contract §4.1-2 multi-segment X at p).
                    bmesh.ops.pointmerge(bm, verts=[vertex, new_vertex], merge_co=vertex.co)
                except (RuntimeError, ValueError) as exc:
                    return stitched, f"could not unify plane-stitch vertices: {exc}"
            stitched += 1

    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.normal_update()
    return stitched, ""


def _mirror_invariant_endpoint_key(
    co,
    axis_index: int,
) -> tuple[float, float, float]:
    """Canonical endpoint key with |axis| so ρ-related seeds share the key.

    Absolute value on the mirror axis makes (M,I) and (ρM,ρI) order-equivalent
    for seed selection. Full orbit ties are allowed under contract §1.2
    best-effort attribute inheritance (shape keys etc.).
    """

    components = [float(co[0]), float(co[1]), float(co[2])]
    components[axis_index] = abs(components[axis_index])
    return (components[0], components[1], components[2])


def _lex_first_edge(
    edges: Sequence[bmesh.types.BMEdge],
    axis_index: int = 0,
) -> bmesh.types.BMEdge:
    """Deterministic edge pick by mirror-invariant endpoint keys (§1.2 best-effort).

    Keys use |axis| on the mirror component so a seed and its mirror image
    sort equivalently. Remaining complete-orbit ties are acceptable best-effort.
    """

    return min(
        edges,
        key=lambda edge: (
            min(
                _mirror_invariant_endpoint_key(edge.verts[0].co, axis_index),
                _mirror_invariant_endpoint_key(edge.verts[1].co, axis_index),
            ),
            max(
                _mirror_invariant_endpoint_key(edge.verts[0].co, axis_index),
                _mirror_invariant_endpoint_key(edge.verts[1].co, axis_index),
            ),
        ),
    )


def _plan_plane_stitch_vertex(
    bm: bmesh.types.BMesh,
    representative: Vector,
    member_edges: Sequence[bmesh.types.BMEdge],
    axis_index: int,
    tolerance: float,
) -> tuple[
    bmesh.types.BMVert | None,
    tuple[bmesh.types.BMEdge, float] | None,
    str,
]:
    """Priority plan: (1) existing vertex (2) host edge split (3) member seed.

    Returns ``(existing_vertex, host_split_or_None, error)``. When both vertex
    and host_split are None and error is empty, the caller creates *p* by
    splitting a member CROSSES edge.
    """

    del axis_index  # reserved; host multi-candidate no longer uses mirror pairing

    # (1) Existing vertex within tol of the representative.
    exact_vertices = sorted(
        (
            ((vertex.co - representative).length, vertex.index, vertex)
            for vertex in bm.verts
            if vertex.is_valid and coordinates_match(vertex.co, representative, tolerance)
        ),
        key=lambda item: (item[0], item[1]),
    )
    if len(exact_vertices) > 1:
        return None, None, "ambiguous on-plane vertices within tolerance at a knife stitch point"
    if exact_vertices:
        return exact_vertices[0][2], None, ""

    # (2) Existing edges whose interior contains the representative.
    # Match Tolerance is the only threshold; a wider edge limit would adopt
    # unrelated edges.
    edge_limit = max(tolerance, 1.0e-9)
    host_edges: list[tuple[float, int, bmesh.types.BMEdge, float]] = []
    member_ids = {id(edge) for edge in member_edges}
    for edge in bm.edges:
        if not edge.is_valid or id(edge) in member_ids:
            continue
        distance, factor = _point_segment_distance_and_factor(representative, edge)
        if not _is_interior_edge_factor(factor, edge.calc_length(), tolerance):
            continue
        if distance > edge_limit:
            continue
        host_edges.append((distance, edge.index, edge, factor))

    if host_edges:
        host_edges.sort(key=lambda item: (item[0], item[1]))
        if len(host_edges) > 1:
            # §4.1-2: multi-candidate host edges are ambiguous.
            # - Nearest fallback is forbidden.
            # - A mirror pair both within tol is equidistant from on-plane p
            #   (degenerate); edge.index tie-break is also forbidden → decline.
            return None, None, "ambiguous host edges for a knife plane-stitch point"
        _distance, _index, host_edge, factor = host_edges[0]
        return None, (host_edge, factor), ""

    # (3) Caller creates via member edge_split.
    return None, None, ""


def collect_source_path_edges(
    bm: bmesh.types.BMesh,
    axis_index: int,
    tolerance: float,
    requested_side: str,
    *,
    selected_only: bool = False,
) -> tuple[list[bmesh.types.BMEdge], str | None, int, int]:
    """Return path edges on one source half (Loop Cut / Offset; one-side).

    Knife uses :func:`collect_knife_path_edges_by_side` instead. The two final
    integers are the total number of new path edges and the number that cross
    the mirror plane.
    """

    all_path_edges = _discover_path_edges(bm, selected_only=selected_only)
    if not all_path_edges and bm.edges.layers.int.get(EDGE_ORIGINAL_LAYER) is None:
        return [], None, 0, 0

    side, crossing = choose_source_side(all_path_edges, axis_index, tolerance, requested_side)
    if side is None:
        return [], None, len(all_path_edges), crossing

    source_edges = [edge for edge in all_path_edges if _edge_side(edge, axis_index, tolerance) == side]
    return source_edges, side, len(all_path_edges), crossing


def target_face_ids_for_edges(
    source_edges: Iterable[bmesh.types.BMEdge],
    face_layer,
    mirror_face_ids: MirrorFaceMap,
) -> tuple[set[FaceId], set[FaceId]]:
    """Return matching target IDs and source IDs without a counterpart."""

    targets: set[FaceId] = set()
    unmatched: set[FaceId] = set()
    for edge in source_edges:
        for face in edge.link_faces:
            source_id = FaceId(int(face[face_layer]))
            target_id = mirror_face_ids.get(source_id)
            if target_id is None:
                unmatched.add(source_id)
            else:
                targets.add(target_id)
    return targets, unmatched


def _point_segment_distance_and_factor(
    coordinate: Vector,
    edge: bmesh.types.BMEdge,
) -> tuple[float, float]:
    # Intentionally Euclidean: this is a geometric point-to-segment distance,
    # not a coordinate-identity test (which is coordinates_match / Chebyshev).
    a = edge.verts[0].co
    delta = edge.verts[1].co - a
    length_squared = delta.length_squared
    if length_squared <= 1.0e-30:
        return (coordinate - a).length, 0.0
    factor = max(
        0.0,
        min(1.0, (coordinate - a).dot(delta) / length_squared),
    )
    return (coordinate - (a + factor * delta)).length, factor


def _is_interior_edge_factor(factor: float, edge_length: float, tolerance: float) -> bool:
    """True when the split sits more than *tolerance* from both endpoints (contract F)."""

    if edge_length <= 0.0:
        return False
    distance_along = factor * edge_length
    return tolerance < distance_along < (edge_length - tolerance)


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

    Multiple distinct vertices within *tolerance* of *expected* are ambiguous
    (contract §1.1 / direct-topology safety): callers must decline rather than
    pick an arbitrary vertex that could invent a duplicate edge.
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
        distance, factor = _point_segment_distance_and_factor(expected, edge)
        if not _is_interior_edge_factor(factor, edge.calc_length(), tolerance):
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


def reflected_path_uses_only_target_boundaries(
    bm: bmesh.types.BMesh,
    source_edges: Iterable[bmesh.types.BMEdge],
    axis_index: int,
    tolerance: float,
    mirror_face_ids: MirrorFaceMap,
) -> bool:
    """Return whether the direct path builder supports every reflected vertex.

    A committed straight Knife segment and the loop-based tools terminate on
    existing face boundaries. Multi-click Knife strokes may also contain an
    intentional bend or intersection inside a face. Those interior networks
    still use the legacy Knife Project path for now; this preflight keeps that
    fallback available without partially editing the target first.
    """

    source_edges = list(source_edges)
    face_layer = bm.faces.layers.int.get(FACE_ID_LAYER)
    if not source_edges or face_layer is None:
        return False

    (
        source_vertex_by_key,
        target_ids_by_vertex,
        _edge_records,
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

    for source_key, source_vertex in source_vertex_by_key.items():
        expected = mirror_coordinate(source_vertex.co, axis_index)
        candidate_faces = {
            face
            for target_id in target_ids_by_vertex[source_key]
            for face in target_faces_by_id.get(target_id, ())
            if face.is_valid
        }
        kind, _vertex, _edge, _factor, _reason = _resolve_reflected_vertex_on_target(
            expected,
            candidate_faces,
            tolerance,
        )
        if kind in {"missing", "ambiguous"}:
            return False

    return True


def apply_reflected_path_topology(
    bm: bmesh.types.BMesh,
    source_edges: Iterable[bmesh.types.BMEdge],
    axis_index: int,
    tolerance: float,
    mirror_face_ids: MirrorFaceMap,
) -> tuple[int, int, str]:
    """Rebuild a native cut path exactly on its mirrored faces.

    Unlike Knife Project, this is independent of the viewport and cannot cut a
    second front/back surface when a curved loop overlaps itself on screen.
    The native source path supplies exact endpoint coordinates and inherited
    original-face IDs. Target boundary edges are split at the reflected points,
    then the corresponding target faces are split between those vertices.

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

    source_edges = list(source_edges)
    if not source_edges:
        return 0, 0, "no source cut edges were supplied"

    marker_layer = bm.edges.layers.int.get(EDGE_ORIGINAL_LAYER)
    face_layer = bm.faces.layers.int.get(FACE_ID_LAYER)
    if marker_layer is None or face_layer is None:
        return 0, 0, "temporary topology markers are missing"

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
        return (
            0,
            0,
            f"{len(unmatched_face_ids)} source face(s) have no mirrored counterpart",
        )
    if any(not target_ids for _a, _b, target_ids in edge_records):
        return 0, 0, "a source cut edge has no mirrored target face"

    needed_target_ids = {target_id for target_ids in target_ids_by_vertex.values() for target_id in target_ids}
    target_faces_by_id = _target_faces_by_id(bm, face_layer, needed_target_ids)

    target_vertex_by_source_key: dict[int, bmesh.types.BMVert] = {}
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
        if kind == "exact":
            assert exact_vertex is not None
            target_vertex_by_source_key[source_key] = exact_vertex
            continue
        if kind in {"missing", "ambiguous"}:
            return 0, 0, reason

        assert target_edge is not None
        try:
            _new_edge, target_vertex = bmesh.utils.edge_split(
                target_edge,
                target_edge.verts[0],
                factor,
            )
        except (RuntimeError, ValueError) as exc:
            return 0, 0, f"could not split a mirrored target edge: {exc}"
        target_vertex.co = expected
        target_vertex.select = False
        target_vertex_by_source_key[source_key] = target_vertex

    # Endpoint-tol store matches build_reflected_cutter so geometric duplicates
    # (different BMVert pairs within tol) count as already_present.
    existing_edges: _EdgeEndpointStore = {}
    for edge in bm.edges:
        if edge.is_valid:
            _register_edge_endpoint_pair(
                existing_edges,
                edge.verts[0].co,
                edge.verts[1].co,
                tolerance,
            )

    created_edges = 0
    already_present = 0
    pending = edge_records
    while pending:
        deferred = []
        progress = False
        for source_a, source_b, possible_target_ids in pending:
            target_a = target_vertex_by_source_key[source_a]
            target_b = target_vertex_by_source_key[source_b]
            existing = bm.edges.get([target_a, target_b])
            if existing is not None:
                existing_target_ids = {FaceId(int(face[face_layer])) for face in existing.link_faces}
                if not existing_target_ids.intersection(possible_target_ids):
                    return (
                        created_edges,
                        already_present,
                        "an existing mirrored edge is outside its target face",
                    )
                already_present += 1
                progress = True
                continue

            if _edge_coordinate_key_matches(target_a.co, target_b.co, tolerance, existing_edges):
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
                bmesh.utils.face_split(candidate_faces[0], target_a, target_b)
            except (RuntimeError, ValueError) as exc:
                return (
                    created_edges,
                    already_present,
                    f"could not split a target face: {exc}",
                )
            new_edge = bm.edges.get([target_a, target_b])
            if new_edge is None:
                return created_edges, already_present, "target face split made no edge"
            new_edge[marker_layer] = 0
            new_edge.select = False
            for face in new_edge.link_faces:
                face.select = False
            _register_edge_endpoint_pair(
                existing_edges,
                new_edge.verts[0].co,
                new_edge.verts[1].co,
                tolerance,
            )
            created_edges += 1
            progress = True

        if deferred and not progress:
            return (
                created_edges,
                already_present,
                f"could not place {len(deferred)} mirrored cut segment(s)",
            )
        pending = deferred

    bm.normal_update()
    return created_edges, already_present, ""


# Kept for internal/test compatibility with 0.4.2 and 0.4.3.
apply_reflected_loop_topology = apply_reflected_path_topology


def _edge_coordinate_key(
    a: Vector,
    b: Vector,
    tolerance: float,
) -> tuple[QuantizedCoordinate, QuantizedCoordinate]:
    qa = _quantized_coordinate(a, tolerance)
    qb = _quantized_coordinate(b, tolerance)
    return (qa, qb) if qa <= qb else (qb, qa)


def _coordinate_tuple(co: Vector) -> tuple[float, float, float]:
    return (float(co[0]), float(co[1]), float(co[2]))


def _canonical_edge_endpoints(
    a: Vector,
    b: Vector,
    tolerance: float,
) -> tuple[QuantizedCoordinate, tuple[float, float, float], tuple[float, float, float]]:
    """Return primary bin of the first endpoint plus real coords in canonical order."""

    coord_a = _coordinate_tuple(a)
    coord_b = _coordinate_tuple(b)
    qa = _quantized_coordinate(a, tolerance)
    qb = _quantized_coordinate(b, tolerance)
    if qa < qb or (qa == qb and coord_a <= coord_b):
        return qa, coord_a, coord_b
    return qb, coord_b, coord_a


# primary bin of canonical first endpoint -> (marker|None, coord_a, coord_b)
_EdgeEndpointEntry = tuple[int | None, tuple[float, float, float], tuple[float, float, float]]
_EdgeEndpointStore = dict[QuantizedCoordinate, list[_EdgeEndpointEntry]]


def _register_edge_endpoint_pair(
    store: _EdgeEndpointStore,
    a: Vector,
    b: Vector,
    tolerance: float,
    marker: int | None = None,
) -> None:
    """Store real endpoint coords under the primary bin of the canonical first endpoint."""

    primary_a, coord_a, coord_b = _canonical_edge_endpoints(a, b, tolerance)
    store.setdefault(primary_a, []).append((marker, coord_a, coord_b))


def _match_edge_endpoint_pair(
    a: Vector,
    b: Vector,
    tolerance: float,
    store: _EdgeEndpointStore,
) -> tuple[float, int | None] | None:
    """Best geometric edge match: min of max endpoint Chebyshev distances (R2-1).

    Probes the 27-bin neighborhood of each query endpoint orientation (not the
    27x27 product). Accepts only candidates with both endpoints within
    *tolerance* Chebyshev distance.
    """

    best: tuple[float, int | None] | None = None
    for query_first, query_second in ((a, b), (b, a)):
        first_coord = _coordinate_tuple(query_first)
        second_coord = _coordinate_tuple(query_second)
        for bin_key in _iter_quantized_neighborhood(query_first, tolerance):
            for marker, stored_a, stored_b in store.get(bin_key, ()):
                distance_a = _chebyshev_distance_3d(first_coord, stored_a)
                distance_b = _chebyshev_distance_3d(second_coord, stored_b)
                if distance_a > tolerance or distance_b > tolerance:
                    continue
                score = max(distance_a, distance_b)
                if best is None or score < best[0]:
                    best = (score, marker)
    return best


def _edge_coordinate_key_matches(
    a: Vector,
    b: Vector,
    tolerance: float,
    store: _EdgeEndpointStore,
) -> bool:
    """True when *(a, b)* geometrically matches a stored edge within *tolerance*."""

    return _match_edge_endpoint_pair(a, b, tolerance, store) is not None


def _edge_keys_matching_lookup(
    a: Vector,
    b: Vector,
    tolerance: float,
    store: _EdgeEndpointStore,
) -> int | None:
    """Return the geometrically nearest original marker, or None if no match."""

    match = _match_edge_endpoint_pair(a, b, tolerance, store)
    if match is None:
        return None
    return match[1]


def build_reflected_cutter(
    bm: bmesh.types.BMesh,
    source_edges: Iterable[bmesh.types.BMEdge],
    axis_index: int,
    tolerance: float,
) -> tuple[list[Vector], list[tuple[int, int]], int]:
    """Build reflected loose edges, omitting segments already in the mesh."""

    existing_edges: _EdgeEndpointStore = {}
    for edge in bm.edges:
        _register_edge_endpoint_pair(
            existing_edges,
            edge.verts[0].co,
            edge.verts[1].co,
            tolerance,
        )
    vertex_indices: dict[int, int] = {}
    vertices: list[Vector] = []
    edges: list[tuple[int, int]] = []
    already_present = 0

    bm.verts.ensure_lookup_table()
    bm.verts.index_update()

    for edge in source_edges:
        reflected = (
            mirror_coordinate(edge.verts[0].co, axis_index),
            mirror_coordinate(edge.verts[1].co, axis_index),
        )
        if _edge_coordinate_key_matches(reflected[0], reflected[1], tolerance, existing_edges):
            already_present += 1
            continue

        cutter_edge = []
        for source_vertex, coordinate in zip(edge.verts, reflected, strict=True):
            source_index = source_vertex.index
            cutter_index = vertex_indices.get(source_index)
            if cutter_index is None:
                cutter_index = len(vertices)
                vertex_indices[source_index] = cutter_index
                vertices.append(coordinate)
            cutter_edge.append(cutter_index)
        if cutter_edge[0] != cutter_edge[1]:
            edges.append((cutter_edge[0], cutter_edge[1]))

    return vertices, edges, already_present


def collapsed_offset_target_edge_markers(
    bm: bmesh.types.BMesh,
    source_edges: Iterable[bmesh.types.BMEdge],
    axis_index: int,
    tolerance: float,
) -> tuple[set[int], str]:
    """Find original target edges for an Offset Edge Slide cancelled at zero.

    Blender's Offset macro commits its topology child before Edge Slide.  Esc
    cancels only the slide, leaving two new source loops exactly coincident with
    the selected original loop.  Knife Project cannot cut a coincident edge, so
    this identifies the reflected original target loop for a matching BMesh
    ``offset_edgeloops`` operation.
    """

    marker_layer = bm.edges.layers.int.get(EDGE_ORIGINAL_LAYER)
    if marker_layer is None:
        return set(), "edge marker layer is missing"

    originals_by_endpoint: _EdgeEndpointStore = {}
    new_edges_by_endpoint: _EdgeEndpointStore = {}
    for edge in bm.edges:
        marker = int(edge[marker_layer])
        if marker <= 0:
            _register_edge_endpoint_pair(
                new_edges_by_endpoint,
                edge.verts[0].co,
                edge.verts[1].co,
                tolerance,
            )
        else:
            _register_edge_endpoint_pair(
                originals_by_endpoint,
                edge.verts[0].co,
                edge.verts[1].co,
                tolerance,
                marker=marker,
            )

    target_markers = set()
    matched_nonzero_segments = 0
    for edge in source_edges:
        reflected_a = mirror_coordinate(edge.verts[0].co, axis_index)
        reflected_b = mirror_coordinate(edge.verts[1].co, axis_index)
        if (reflected_a - reflected_b).length <= tolerance:
            # Endpoint-cap output can collapse to a point at factor zero.  The
            # target BMesh op will recreate it from the non-degenerate loop.
            # (Intentionally Euclidean: an edge-length degeneracy test, not a
            # coordinate-identity test.)
            continue
        if _edge_coordinate_key_matches(reflected_a, reflected_b, tolerance, new_edges_by_endpoint):
            return set(), "the target already contains native zero-offset topology"
        marker = _edge_keys_matching_lookup(
            reflected_a,
            reflected_b,
            tolerance,
            originals_by_endpoint,
        )
        if marker is None:
            return set(), "a reflected zero-offset segment has no original target edge"
        target_markers.add(marker)
        matched_nonzero_segments += 1

    if not target_markers or not matched_nonzero_segments:
        return set(), "no reflected original target loop was found"
    return target_markers, ""


def apply_collapsed_offset_topology(
    bm: bmesh.types.BMesh,
    target_edge_markers: set[int],
    *,
    use_cap_endpoint: bool,
) -> tuple[int, str]:
    """Create the target-side topology for a zero-factor Offset operation."""

    marker_layer = bm.edges.layers.int.get(EDGE_ORIGINAL_LAYER)
    if marker_layer is None:
        return 0, "edge marker layer is missing"
    target_edges = [edge for edge in bm.edges if int(edge[marker_layer]) in target_edge_markers]
    if len(target_edges) != len(target_edge_markers):
        return 0, "one or more target loop edges were lost"

    result = bmesh.ops.offset_edgeloops(
        bm,
        edges=target_edges,
        use_cap_endpoint=use_cap_endpoint,
    )
    output_edges = list(result.get("edges", ()))
    if not output_edges:
        return 0, "Blender did not create the target offset topology"
    for edge in output_edges:
        edge.select = False
    bm.normal_update()
    return len(output_edges), ""


def add_selection_layers(bm: bmesh.types.BMesh) -> SelectionSnapshot:
    """Snapshot native Knife's selection immediately before Knife Project."""

    for layers, name in (
        (bm.verts.layers.int, VERT_SELECTION_LAYER),
        (bm.edges.layers.int, EDGE_SELECTION_LAYER),
        (bm.faces.layers.int, FACE_SELECTION_LAYER),
    ):
        old = layers.get(name)
        if old is not None:
            layers.remove(old)

    vertex_layer = bm.verts.layers.int.new(VERT_SELECTION_LAYER)
    edge_layer = bm.edges.layers.int.new(EDGE_SELECTION_LAYER)
    face_layer = bm.faces.layers.int.new(FACE_SELECTION_LAYER)
    marker_layer = bm.edges.layers.int.get(EDGE_ORIGINAL_LAYER)
    face_id_layer = bm.faces.layers.int.get(FACE_ID_LAYER)

    # As above, acquire all elements only after creating every layer.
    for vertex in bm.verts:
        vertex[vertex_layer] = int(vertex.select)
    for edge in bm.edges:
        edge[edge_layer] = int(edge.select)
    for face in bm.faces:
        face[face_layer] = int(face.select)

    path_vertices_selected = False
    path_edges_selected = False
    if marker_layer is not None:
        for edge in bm.edges:
            if edge[marker_layer] <= 0:
                path_edges_selected |= bool(edge.select)
                path_vertices_selected |= any(vertex.select for vertex in edge.verts)
    path_faces_selected = any(face.select for face in bm.faces)

    history: SelectionHistory = []
    select_history = cast(
        Iterable[bmesh.types.BMVert | bmesh.types.BMEdge | bmesh.types.BMFace],
        bm.select_history,
    )
    for element in select_history:
        if isinstance(element, bmesh.types.BMVert):
            history.append(VertexSelectionHistory(location=_coordinate_3d(element.co)))
        elif isinstance(element, bmesh.types.BMEdge):
            midpoint = (element.verts[0].co + element.verts[1].co) * 0.5
            marker = EdgeMarkerId(int(element[marker_layer])) if marker_layer is not None else None
            history.append(
                EdgeSelectionHistory(
                    location=_coordinate_3d(midpoint),
                    marker=marker,
                )
            )
        elif isinstance(element, bmesh.types.BMFace):
            face_id = FaceId(int(element[face_id_layer])) if face_id_layer is not None else None
            history.append(
                FaceSelectionHistory(
                    location=_coordinate_3d(element.calc_center_median()),
                    face_id=face_id,
                )
            )

    return SelectionSnapshot(
        path_vertices_selected=path_vertices_selected,
        path_edges_selected=path_edges_selected,
        path_faces_selected=path_faces_selected,
        history=history,
    )


def restore_visibility_and_selection(
    bm: bmesh.types.BMesh,
    hidden_by_face_id: HiddenFaceMap,
    selection_snapshot: SelectionSnapshot,
) -> None:
    """Undo temporary hiding and Knife Project's selection replacement."""

    face_id_layer = bm.faces.layers.int.get(FACE_ID_LAYER)
    vertex_selection = bm.verts.layers.int.get(VERT_SELECTION_LAYER)
    edge_selection = bm.edges.layers.int.get(EDGE_SELECTION_LAYER)
    face_selection = bm.faces.layers.int.get(FACE_SELECTION_LAYER)
    vertex_hidden = bm.verts.layers.int.get(VERT_HIDDEN_LAYER)
    edge_hidden = bm.edges.layers.int.get(EDGE_HIDDEN_LAYER)

    # Visibility must be restored before selection: unhiding a face clears its
    # selection flag as a side effect.
    for face in bm.faces:
        if face_id_layer is not None:
            face_id = FaceId(int(face[face_id_layer]))
            face.hide = bool(hidden_by_face_id.get(face_id, False))
    for edge in bm.edges:
        edge.hide = bool(edge_hidden and edge[edge_hidden])
    for vertex in bm.verts:
        vertex.hide = bool(vertex_hidden and vertex[vertex_hidden])

    # Through-face edges and vertices created inside a temporarily unhidden face
    # start with zero hide data.  Hide them when every adjacent restored face is
    # hidden, while preserving exact flags on all pre-existing elements.
    for edge in bm.edges:
        if edge.link_faces and all(face.hide for face in edge.link_faces):
            edge.hide = True
    for vertex in bm.verts:
        if vertex.link_faces and all(face.hide for face in vertex.link_faces):
            vertex.hide = True

    # Clear broad-to-narrow, then restore narrow-to-broad.  Assigning a false
    # face flag can cascade into its boundary, while select_flush_mode() can
    # discard an explicitly restored face in face-select mode.  The snapshot
    # came from Blender's already-consistent native Knife result, so replaying
    # its exact flags needs no additional selection flush.
    for face in bm.faces:
        face.select = False
    for edge in bm.edges:
        edge.select = False
    for vertex in bm.verts:
        vertex.select = False

    for vertex in bm.verts:
        if vertex_selection is not None and vertex[vertex_selection]:
            vertex.select = True
    for edge in bm.edges:
        if edge_selection is not None and edge[edge_selection]:
            edge.select = True
    for face in bm.faces:
        if face_selection is not None and face[face_selection]:
            face.select = True

    _restore_selection_history(bm, selection_snapshot.history)


def _restore_selection_history(
    bm: bmesh.types.BMesh,
    history: SelectionHistory,
) -> None:
    marker_layer = bm.edges.layers.int.get(EDGE_ORIGINAL_LAYER)
    face_id_layer = bm.faces.layers.int.get(FACE_ID_LAYER)
    bm.select_history.clear()
    used = set()

    for entry in history:
        location = Vector(entry.location.as_tuple())
        if isinstance(entry, VertexSelectionHistory):
            candidates = [vertex for vertex in bm.verts if vertex.select]
        elif isinstance(entry, EdgeSelectionHistory):
            candidates = [edge for edge in bm.edges if edge.select]
            if marker_layer is not None and entry.marker is not None:
                matching = [edge for edge in candidates if EdgeMarkerId(int(edge[marker_layer])) == entry.marker]
                if matching:
                    candidates = matching
        else:
            candidates = [face for face in bm.faces if face.select]
            if face_id_layer is not None and entry.face_id is not None:
                matching = [face for face in candidates if FaceId(int(face[face_id_layer])) == entry.face_id]
                if matching:
                    candidates = matching

        candidates = [candidate for candidate in candidates if hash(candidate) not in used]
        if not candidates:
            continue
        chosen = min(
            candidates,
            key=lambda candidate: (_selection_element_coordinate(candidate) - location).length_squared,
        )
        used.add(hash(chosen))
        bm.select_history.add(chosen)


def _selection_element_coordinate(element) -> Vector:
    if isinstance(element, bmesh.types.BMVert):
        return element.co
    if isinstance(element, bmesh.types.BMEdge):
        return (element.verts[0].co + element.verts[1].co) * 0.5
    return element.calc_center_median()


def reserve_source_path_marker(bm: bmesh.types.BMesh) -> int:
    """Move source path edges away from zero before Knife Project runs.

    Knife Project creates its new through-face edges with the default integer
    value zero.  Marking the already-created native Knife graph as -1 makes the
    projected destination graph unambiguous, including closed-loop bridge edges.
    """

    marker_layer = bm.edges.layers.int.get(EDGE_ORIGINAL_LAYER)
    if marker_layer is None:
        return 0
    count = 0
    for edge in bm.edges:
        if edge[marker_layer] == 0:
            edge[marker_layer] = -1
            count += 1
    return count


# One step is one trial assignment of a candidate to a destination vertex.
# Assignments forced by propagation (a single remaining candidate) are free.
_PROJECTION_STEP_LIMIT = 10_000


def _assign_projection_candidates(
    candidates: list[tuple[float, int, int]],
    destination_count: int,
    destination_pairs: list[tuple[int, int]],
    expected_edge_set: set[tuple[int, int]],
) -> tuple[dict[int, int], dict[int, float], str]:
    """Adjacency-constrained matching of destination vertices to expected ones.

    Replaces the earlier distance-greedy assignment, which provably swapped
    near-coincident vertices and then failed the global adjacency check even
    though a valid solution existed.  The search combines unit propagation
    (a destination with one remaining candidate is fixed and its target is
    removed everywhere), adjacency propagation (the candidates of a fixed
    destination's unassigned neighbors shrink to the expected-graph neighbors
    of its target), and depth-first backtracking over the rest, trying
    candidates in ascending distance order.

    The returned solution is the deterministic first accepted one; total or
    maximum distance optimality is not guaranteed.  Trying candidates
    nearest-first biases the search toward short-distance solutions, which in
    practice suppresses embedding twists via expected-graph automorphisms.
    Extension point if strict optimality ever becomes necessary:
    branch-and-bound over the same propagation core.

    Returns ``(assignment, distances, failure_reason)``; a non-empty reason
    means no assignment was produced.
    """

    allowed: list[dict[int, float]] = [{} for _ in range(destination_count)]
    for distance, destination_id, expected_id in candidates:
        previous = allowed[destination_id].get(expected_id)
        if previous is None or distance < previous:
            allowed[destination_id][expected_id] = distance
    if any(not options for options in allowed):
        return {}, {}, "could not match every projected graph vertex"
    initial_allowed = [dict(options) for options in allowed]

    destination_adjacency: list[set[int]] = [set() for _ in range(destination_count)]
    for a, b in destination_pairs:
        destination_adjacency[a].add(b)
        destination_adjacency[b].add(a)
    expected_adjacency: dict[int, set[int]] = defaultdict(set)
    for a, b in expected_edge_set:
        expected_adjacency[a].add(b)
        expected_adjacency[b].add(a)

    assignment: dict[int, int] = {}
    used: set[int] = set()
    steps = 0

    def _sorted_options(destination_id: int) -> list[tuple[float, int]]:
        return sorted(
            (distance, expected_id)
            for expected_id, distance in allowed[destination_id].items()
            if expected_id not in used
        )

    def _assign_and_propagate(destination_id: int, expected_id: int, trail: list) -> bool:
        """Fix one pair, then propagate; False on contradiction."""

        queue = [(destination_id, expected_id)]
        while queue:
            current, target = queue.pop()
            if current in assignment:
                if assignment[current] != target:
                    return False
                continue
            if target in used:
                return False
            for neighbor in destination_adjacency[current]:
                fixed = assignment.get(neighbor)
                if fixed is not None and fixed not in expected_adjacency[target]:
                    return False
            assignment[current] = target
            used.add(target)
            trail.append(current)
            for neighbor in destination_adjacency[current]:
                if neighbor in assignment:
                    continue
                options = allowed[neighbor]
                restricted = {
                    expected: distance
                    for expected, distance in options.items()
                    if expected in expected_adjacency[target]
                }
                if len(restricted) != len(options):
                    trail.append((neighbor, options))
                    allowed[neighbor] = restricted
                available = [(distance, expected) for expected, distance in restricted.items() if expected not in used]
                if not available:
                    return False
                if len(available) == 1:
                    queue.append((neighbor, min(available)[1]))
        return True

    def _undo(trail: list) -> None:
        while trail:
            item = trail.pop()
            if isinstance(item, tuple):
                neighbor, previous_options = item
                allowed[neighbor] = previous_options
            else:
                used.discard(assignment.pop(item))

    # Root pass: destinations that are unique from the start.  A contradiction
    # here means no adjacency-consistent complete assignment exists at all.
    root_trail: list = []
    for destination_id in range(destination_count):
        if destination_id in assignment:
            continue
        options = _sorted_options(destination_id)
        if not options:
            return {}, {}, "could not match every projected graph vertex"
        if len(options) == 1 and not _assign_and_propagate(destination_id, options[0][1], root_trail):
            return {}, {}, "graph adjacency mismatch"

    def _choose_destination() -> tuple[int, list[tuple[float, int]]] | None:
        best: tuple[int, list[tuple[float, int]]] | None = None
        for destination_id in range(destination_count):
            if destination_id in assignment:
                continue
            options = _sorted_options(destination_id)
            if best is None or len(options) < len(best[1]):
                best = (destination_id, options)
                if len(options) <= 1:
                    break
        return best

    frames: list[list] = []
    advancing = len(assignment) < destination_count
    while len(assignment) < destination_count:
        if advancing:
            chosen = _choose_destination()
            assert chosen is not None
            frames.append([chosen[0], chosen[1], 0, []])
        if not frames:
            return {}, {}, "graph adjacency mismatch"
        frame = frames[-1]
        destination_id, options, _index, trail = frame
        _undo(trail)
        placed = False
        while frame[2] < len(options):
            _distance, expected_id = options[frame[2]]
            frame[2] += 1
            if expected_id in used:
                continue
            if len(options) > 1:
                steps += 1
                if steps > _PROJECTION_STEP_LIMIT:
                    return {}, {}, "ambiguous projection correspondence"
            if _assign_and_propagate(destination_id, expected_id, trail):
                placed = True
                break
            _undo(trail)
        if placed:
            advancing = True
        else:
            frames.pop()
            advancing = False

    distances = {
        destination_id: initial_allowed[destination_id][expected_id]
        for destination_id, expected_id in assignment.items()
    }
    return dict(assignment), distances, ""


def _nearby_projection_candidates(
    destination_vertices: list[bmesh.types.BMVert],
    destination_degree: list[int],
    expected_vertices: list[Vector],
    expected_degree: list[int],
    snap_limit: float,
    existing_limit: float,
    preexisting_vertex_keys: set[int],
) -> list[tuple[float, int, int]]:
    """Return only degree-compatible expected vertices within snapping range."""

    expected_ids_by_degree: dict[int, list[int]] = defaultdict(list)
    for expected_id, degree in enumerate(expected_degree):
        expected_ids_by_degree[degree].append(expected_id)

    trees = {}
    for degree, expected_ids in expected_ids_by_degree.items():
        tree = KDTree(len(expected_ids))
        for expected_id in expected_ids:
            tree.insert(expected_vertices[expected_id], expected_id)
        tree.balance()
        trees[degree] = tree

    candidates = []
    for destination_id, vertex in enumerate(destination_vertices):
        tree = trees.get(destination_degree[destination_id])
        if tree is None:
            continue
        limit = existing_limit if hash(vertex) in preexisting_vertex_keys else snap_limit
        # Include points on the numerical boundary of the accepted range.
        # Intentionally Euclidean: the KDTree radius only collects candidates
        # and decides nothing.  It shares the metric and the per-vertex limit
        # with the final distance validation in snap_projected_graph, so the
        # radius is a complete bound on the acceptable search space.
        search_radius = limit * (1.0 + 1.0e-12) + 1.0e-15
        for _coordinate, expected_id, distance in tree.find_range(
            vertex.co,
            search_radius,
        ):
            candidates.append((distance, destination_id, expected_id))
    return candidates


def _mapped_projection_edge_set(
    destination_pairs: list[tuple[int, int]],
    assignment: dict[int, int],
) -> set[tuple[int, int]]:
    mapped = set()
    for a, b in destination_pairs:
        ma, mb = assignment[a], assignment[b]
        mapped.add((ma, mb) if ma <= mb else (mb, ma))
    return mapped


def snap_projected_graph(
    bm: bmesh.types.BMesh,
    expected_vertices: list[Vector],
    expected_edges: list[tuple[int, int]],
    tolerance: float,
    preexisting_vertex_keys: set[int] | None = None,
) -> tuple[bool, float, str]:
    """Snap Knife Project's destination graph to exact reflected coordinates.

    Knife Project is screen-space, so even a cutter that lies exactly on the
    destination surface can return small projection errors.  The new graph is
    identified by its zero edge marker, matched by degree and proximity, checked
    for graph isomorphism, and only then snapped.
    """

    marker_layer = bm.edges.layers.int.get(EDGE_ORIGINAL_LAYER)
    if marker_layer is None:
        return False, 0.0, "edge marker layer is missing"

    destination_edges = [edge for edge in bm.edges if edge[marker_layer] == 0]
    destination_vertices = []
    destination_index = {}
    destination_pairs = []
    bm.verts.ensure_lookup_table()
    bm.verts.index_update()
    for edge in destination_edges:
        pair = []
        for vertex in edge.verts:
            key = vertex.index
            index = destination_index.get(key)
            if index is None:
                index = len(destination_vertices)
                destination_index[key] = index
                destination_vertices.append(vertex)
            pair.append(index)
        destination_pairs.append((pair[0], pair[1]))

    if len(destination_edges) != len(expected_edges):
        return (
            False,
            0.0,
            f"projected edge count {len(destination_edges)} != expected {len(expected_edges)}",
        )
    if len(destination_vertices) != len(expected_vertices):
        return (
            False,
            0.0,
            f"projected vertex count {len(destination_vertices)} != expected {len(expected_vertices)}",
        )
    if not expected_vertices:
        return True, 0.0, ""

    expected_degree = [0] * len(expected_vertices)
    for a, b in expected_edges:
        expected_degree[a] += 1
        expected_degree[b] += 1
    destination_degree = [0] * len(destination_vertices)
    for a, b in destination_pairs:
        destination_degree[a] += 1
        destination_degree[b] += 1

    expected_edge_set = {(a, b) if a <= b else (b, a) for a, b in expected_edges}
    expected_lengths = [
        (expected_vertices[a] - expected_vertices[b]).length
        for a, b in expected_edges
        if (expected_vertices[a] - expected_vertices[b]).length > tolerance
    ]
    minimum_edge_length = min(expected_lengths, default=max(tolerance, 1.0e-6))
    snap_limit = max(tolerance * 20.0, minimum_edge_length * 0.02)
    existing_limit = max(tolerance * 2.0, 1.0e-9)
    preexisting_vertex_keys = preexisting_vertex_keys or set()

    # Long Loop Cut graphs often contain thousands of same-degree vertices.
    # Searching only their local KDTree neighborhood keeps the normal path near
    # O(n log n).  The radius search is a *complete* candidate enumeration: it
    # uses the same Euclidean metric and the same per-vertex limits as the
    # final distance validation below, so an assignment using any vertex
    # outside the radius would necessarily be rejected there.  No wider
    # fallback can add an acceptable solution.  Degree compatibility is a
    # necessary condition of the final graph isomorphism check and therefore
    # never narrows the acceptable space either.
    candidates = _nearby_projection_candidates(
        destination_vertices,
        destination_degree,
        expected_vertices,
        expected_degree,
        snap_limit,
        existing_limit,
        preexisting_vertex_keys,
    )
    assignment, distances, assignment_reason = _assign_projection_candidates(
        candidates,
        len(destination_vertices),
        destination_pairs,
        expected_edge_set,
    )
    if assignment_reason:
        return False, 0.0, assignment_reason

    if len(assignment) != len(destination_vertices):
        return False, 0.0, "could not match every projected graph vertex"

    mapped_edge_set = _mapped_projection_edge_set(destination_pairs, assignment)
    if mapped_edge_set != expected_edge_set:
        return False, max(distances.values(), default=0.0), "graph adjacency mismatch"

    # Intentionally Euclidean, and it must stay that way: sharing this metric
    # and these limits with the KDTree radius search above is exactly what
    # makes that search a complete candidate enumeration.
    maximum_distance = max(distances.values(), default=0.0)
    existing_error = max(
        (
            distances[destination_id]
            for destination_id, vertex in enumerate(destination_vertices)
            if hash(vertex) in preexisting_vertex_keys
        ),
        default=0.0,
    )
    movable_error = max(
        (
            distances[destination_id]
            for destination_id, vertex in enumerate(destination_vertices)
            if hash(vertex) not in preexisting_vertex_keys
        ),
        default=0.0,
    )
    if existing_error > existing_limit:
        return (
            False,
            existing_error,
            f"existing endpoint mismatch {existing_error:.6g} exceeds {existing_limit:.6g}",
        )
    if movable_error > snap_limit:
        return (
            False,
            movable_error,
            f"projection error {movable_error:.6g} exceeds safe snap limit {snap_limit:.6g}",
        )

    for destination_id, expected_id in assignment.items():
        vertex = destination_vertices[destination_id]
        if hash(vertex) not in preexisting_vertex_keys:
            vertex.co = expected_vertices[expected_id]
    bm.normal_update()
    return True, maximum_distance, ""
