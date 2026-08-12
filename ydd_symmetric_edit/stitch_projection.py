from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

import bmesh
from mathutils import Vector
from mathutils.kdtree import KDTree

from . import stitch_common
from .layer_names import EDGE_ORIGINAL_LAYER
from .matching import mirror_coordinate


def build_reflected_cutter(
    bm: bmesh.types.BMesh,
    source_edges: Iterable[bmesh.types.BMEdge],
    axis_index: int,
    tolerance: float,
) -> tuple[list[Vector], list[tuple[int, int]], int]:
    """Build reflected loose edges, omitting segments already in the mesh."""

    existing_edges: stitch_common._EdgeEndpointStore = {}
    for edge in bm.edges:
        stitch_common._register_edge_endpoint_pair(
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
        if stitch_common._edge_coordinate_key_matches(reflected[0], reflected[1], tolerance, existing_edges):
            already_present += 1
            continue

        cutter_edge = []
        for source_vertex, coordinate in zip(edge.verts, reflected, strict=False):
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

    originals_by_endpoint: stitch_common._EdgeEndpointStore = {}
    new_edges_by_endpoint: stitch_common._EdgeEndpointStore = {}
    for edge in bm.edges:
        marker = int(edge[marker_layer])
        if marker <= 0:
            stitch_common._register_edge_endpoint_pair(
                new_edges_by_endpoint,
                edge.verts[0].co,
                edge.verts[1].co,
                tolerance,
            )
        else:
            stitch_common._register_edge_endpoint_pair(
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
        if stitch_common._edge_coordinate_key_matches(reflected_a, reflected_b, tolerance, new_edges_by_endpoint):
            return set(), "the target already contains native zero-offset topology"
        marker = stitch_common._edge_keys_matching_lookup(
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
