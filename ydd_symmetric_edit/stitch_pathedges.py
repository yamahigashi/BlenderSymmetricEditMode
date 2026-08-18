# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import bmesh

from . import stitch_common, stitch_crossings
from ._types import FaceId, MirrorFaceMap
from .layer_names import EDGE_HIDDEN_LAYER, EDGE_ORIGINAL_LAYER, FACE_ID_LAYER

_LOGGER = logging.getLogger(__name__)


_MIN_SIDE_LENGTH = 1.0e-9


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
    """True when *edge* is a native cut fragment.

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


def native_path_edge_state(bm: bmesh.types.BMesh) -> Literal["PRESENT", "ABSENT", "UNKNOWN"]:
    """Classify whether *bm* carries evidence of an unprocessed native cut.

    ABSENT is only returned when both marker layers exist and a full scan
    found no path edge, so callers can treat missing layers or read failures
    as indeterminate rather than as a clean prepared baseline.
    """

    try:
        edge_layer = bm.edges.layers.int.get(EDGE_ORIGINAL_LAYER)
        face_layer = bm.faces.layers.int.get(FACE_ID_LAYER)
        if edge_layer is None or face_layer is None:
            return "UNKNOWN"
        for edge in bm.edges:
            if _is_path_edge_by_markers(edge, edge_layer, face_layer):
                return "PRESENT"
        return "ABSENT"
    except (AttributeError, ReferenceError, RuntimeError):
        return "UNKNOWN"


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

    Both branches use the same novelty test (tag==0 or FACE_ID complement).
    *selected_only* only adds the selection filter for Loop Cut / Offset;
    Knife (selected_only=False) must still recover inherited-tag CROSSES
    fragments so the whole-stage decline cannot be skipped.
    """

    edge_layer = bm.edges.layers.int.get(EDGE_ORIGINAL_LAYER)
    if edge_layer is None:
        return []

    face_layer = bm.faces.layers.int.get(FACE_ID_LAYER)
    result: list[bmesh.types.BMEdge] = []
    for edge in bm.edges:
        if selected_only and not edge.select:
            continue
        if edge[edge_layer] == 0:
            result.append(edge)
            continue
        if face_layer is None:
            continue
        faces = edge.link_faces
        count = len(faces)
        if count < 2:
            continue
        first = faces[0][face_layer]
        if count == 2:
            if faces[1][face_layer] == first:
                result.append(edge)
            continue
        if all(face[face_layer] == first for face in faces):
            result.append(edge)
    return result


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


_KnifePathEdgeCacheEntry = tuple[
    int,
    tuple[float, float, float],
    tuple[float, float, float],
]


_KnifePathEdgeCache = tuple[_KnifePathEdgeCacheEntry, ...]


def capture_knife_path_edge_cache(
    bm: bmesh.types.BMesh,
    path_edges: Iterable[bmesh.types.BMEdge],
) -> _KnifePathEdgeCache | None:
    """Capture metadata for reclassifying an unchanged Knife path.

    No ``BMEdge`` proxy is retained: the edit BMesh may be rebuilt between
    calls.  Reuse verifies both the edge index and endpoint coordinates.
    """

    bm.edges.ensure_lookup_table()
    bm.edges.index_update()
    entries: list[_KnifePathEdgeCacheEntry] = []
    for edge in path_edges:
        index = int(edge.index)
        if index < 0:
            return None
        first = stitch_common._coordinate_tuple(edge.verts[0].co)
        second = stitch_common._coordinate_tuple(edge.verts[1].co)
        if second < first:
            first, second = second, first
        entries.append((index, first, second))
    return tuple(entries)


def reclassify_knife_path_edge_cache(
    bm: bmesh.types.BMesh,
    axis_index: int,
    tolerance: float,
    cache: _KnifePathEdgeCache | None,
) -> tuple[dict[str, list[bmesh.types.BMEdge]], int] | None:
    """Reclassify cached path edges, or return ``None`` when unverifiable."""

    if cache is None:
        return None
    bm.edges.ensure_lookup_table()
    path_edges: list[bmesh.types.BMEdge] = []
    for index, expected_first, expected_second in cache:
        if index >= len(bm.edges):
            return None
        edge = bm.edges[index]
        if not edge.is_valid:
            return None
        first = stitch_common._coordinate_tuple(edge.verts[0].co)
        second = stitch_common._coordinate_tuple(edge.verts[1].co)
        if second < first:
            first, second = second, first
        if first != expected_first or second != expected_second:
            return None
        path_edges.append(edge)
    by_side = classify_path_edges_by_side(path_edges, axis_index, tolerance)
    return by_side, len(path_edges)


def collect_knife_path_edges_by_side(
    bm: bmesh.types.BMesh,
    axis_index: int,
    tolerance: float,
) -> tuple[dict[str, list[bmesh.types.BMEdge]], int]:
    """Classify every new Knife path edge without choosing a source side.

    Both POSITIVE and NEGATIVE buckets are mirrored toward each other. PLANE
    edges are shared. CROSSES edges are p-stitched before the half-edges join
    the POSITIVE/NEGATIVE mirror path.

    Returns ``(by_side, total_path_edge_count)``.
    """

    all_path_edges = _discover_path_edges(bm, selected_only=False)
    by_side = classify_path_edges_by_side(all_path_edges, axis_index, tolerance)
    return by_side, len(all_path_edges)


def patch_knife_path_edges_by_side(
    bm: bmesh.types.BMesh,
    previous_by_side: Mapping[str, Sequence[bmesh.types.BMEdge]],
    cache: _KnifePathEdgeCache | None,
    summary: stitch_crossings.CrossingMutationSummary,
    axis_index: int,
    tolerance: float,
) -> tuple[dict[str, list[bmesh.types.BMEdge]], int] | None:
    """Patch a Knife collect result from a crossings mutation summary.

    ``None`` means the strict §I-3b conditions were not met and callers must
    perform the ordinary full collect.  The returned bucket order is formed
    by replacing each source edge in-place and appending descendants to its
    original bucket.
    """

    if cache is None or not summary.edges:
        return None
    if summary.removed_edge_count or summary.removed_face_count:
        return None
    bm.edges.ensure_lookup_table()
    previous = [
        edge for side in ("POSITIVE", "NEGATIVE", "CROSSES", "PLANE") for edge in previous_by_side.get(side, ())
    ]
    if len(previous) != len(cache):
        return None
    if len(set(cache)) != len(cache):
        return None
    cache_positions = {hash(edge): position for position, edge in enumerate(previous)}
    replacement: dict[str, list[bmesh.types.BMEdge]] = {
        side: list(previous_by_side.get(side, ())) for side in ("POSITIVE", "NEGATIVE", "CROSSES", "PLANE")
    }
    bucket_positions: dict[tuple[str, int], int] = {}
    for side, bucket in replacement.items():
        for local_position, edge in enumerate(bucket):
            cache_position = cache_positions.get(hash(edge))
            if cache_position is None or (side, cache_position) in bucket_positions:
                return None
            bucket_positions[(side, cache_position)] = local_position
    edge_layer = bm.edges.layers.int.get(EDGE_ORIGINAL_LAYER)
    face_layer = bm.faces.layers.int.get(FACE_ID_LAYER)
    if edge_layer is None:
        return None
    previous_by_id = {
        hash(edge): (edge, side, position)
        for side, bucket in replacement.items()
        for position, edge in enumerate(bucket)
    }
    expected_side_by_edge: dict[int, tuple[bmesh.types.BMEdge, str]] = {}
    for side, bucket in replacement.items():
        for edge in bucket:
            if not edge.is_valid or not _is_path_edge_by_markers(edge, edge_layer, face_layer):
                _LOGGER.warning("incremental Knife collect declined: cached path-edge membership changed")
                return None
            if _edge_side(edge, axis_index, tolerance) != side:
                _LOGGER.warning("incremental Knife collect declined: cached path-edge side changed")
                return None
            expected_side_by_edge[hash(edge)] = (edge, side)
    for mutation in summary.edges:
        if mutation.endpoint_reused or mutation.pointmerged:
            return None
        source_info = previous_by_id.get(mutation.source_edge_id)
        source = source_info[0] if source_info is not None else None
        source_position = mutation.cache_position
        if (
            source is None
            or source_position is None
            or source_position < 0
            or source_position >= len(previous)
            or previous[source_position] is not source
        ):
            return None
        side = source_info[1] if source_info is not None else None
        if side is None or not mutation.final_edges:
            return None
        final_edges = [edge for edge in mutation.final_edges if edge.is_valid]
        if len(final_edges) != len(mutation.final_edges):
            return None
        if any(not _is_path_edge_by_markers(edge, edge_layer, face_layer) for edge in final_edges):
            _LOGGER.warning("incremental Knife collect declined: split path-edge membership changed")
            return None
        if any(_edge_side(edge, axis_index, tolerance) != side for edge in final_edges):
            return None
        if source not in final_edges or len({hash(edge) for edge in final_edges}) != len(final_edges):
            return None
        for edge in final_edges:
            edge_id = hash(edge)
            prior = expected_side_by_edge.get(edge_id)
            if prior is not None and prior[1] != side:
                _LOGGER.warning("incremental Knife collect declined: mutation side assignment conflicts")
                return None
            expected_side_by_edge[edge_id] = (edge, side)
        # Remove the old record by its flattened cache position.  Descendants
        # are appended below in live BMesh order after all source replacements.
        bucket = replacement[side]
        local_position = bucket_positions.get((side, source_position))
        if local_position is None or bucket[local_position] is not source:
            return None
        bucket[local_position] = source

    if summary.unexpected_topology_change:
        return None
    tail = [bm.edges[index] for index in range(summary.pre_apply_edge_count, len(bm.edges))]
    source_ids = set(previous_by_id)
    mutation_non_sources = [
        edge
        for mutation in summary.edges
        for edge in mutation.final_edges
        if edge.is_valid and hash(edge) != mutation.source_edge_id
    ]
    if any(edge not in tail for edge in mutation_non_sources):
        _LOGGER.warning("incremental Knife collect declined: a split edge is not after pre-apply edges")
        return None
    final_non_sources = [edge for edge, _side in expected_side_by_edge.values() if hash(edge) not in source_ids]
    if len(tail) != len(final_non_sources) or any(edge not in tail for edge in final_non_sources):
        _LOGGER.warning("incremental Knife collect declined: a split edge is not after pre-apply edges")
        return None
    if len(expected_side_by_edge) != len(previous) + len(tail):
        _LOGGER.warning("incremental Knife collect declined: mutation edge set is incomplete")
        return None
    # FACE_ID complement closure is checked after all source mutations so
    # cross-source new half-edges have their expected side available.
    tracked_ids = set(expected_side_by_edge)
    closure_faces = []
    closure_face_ids: set[int] = set()

    def add_closure_face(face: bmesh.types.BMFace) -> None:
        face_id = hash(face)
        if face_id not in closure_face_ids:
            closure_face_ids.add(face_id)
            closure_faces.append(face)

    for mutation in summary.edges:
        for face in mutation.pre_faces:
            if not face.is_valid:
                _LOGGER.warning("incremental Knife collect declined: a pre-split FACE_ID face disappeared")
                return None
            add_closure_face(face)
        for edge in mutation.final_edges:
            if edge.is_valid:
                for face in edge.link_faces:
                    if face.is_valid:
                        add_closure_face(face)
    if face_layer is not None:
        for face in closure_faces:
            for closed_edge in face.edges:
                if not closed_edge.is_valid:
                    continue
                closed_id = hash(closed_edge)
                is_path = _is_path_edge_by_markers(closed_edge, edge_layer, face_layer)
                if closed_id not in tracked_ids:
                    if is_path:
                        _LOGGER.warning(
                            "incremental Knife collect declined: FACE_ID closure contains an untracked path edge"
                        )
                        return None
                    continue
                expected = expected_side_by_edge[closed_id][1]
                if not is_path or _edge_side(closed_edge, axis_index, tolerance) != expected:
                    _LOGGER.warning("incremental Knife collect declined: FACE_ID closure membership or side changed")
                    return None

    for side in ("POSITIVE", "NEGATIVE", "CROSSES", "PLANE"):
        replacement[side].extend(
            edge for edge in tail if expected_side_by_edge.get(hash(edge), (None, None))[1] == side
        )

    patched = {side: list(replacement[side]) for side in ("POSITIVE", "NEGATIVE", "CROSSES", "PLANE")}
    all_edges = [edge for side in ("POSITIVE", "NEGATIVE", "CROSSES", "PLANE") for edge in patched[side]]
    if len({hash(edge) for edge in all_edges}) != len(all_edges):
        return None
    return patched, len(all_edges)


ALREADY_SYMMETRIC: Literal["ALREADY_SYMMETRIC"] = "ALREADY_SYMMETRIC"

SourcePathStatus = Literal["OK", "ALREADY_SYMMETRIC", "UNDETERMINED"]


@dataclass(slots=True)
class SourcePathEdges:
    """Loop Cut / Offset source-path collection (contract §4).

    Unpacks as ``(source_edges, side, total_path_edges, crossing_count)``.
    ``status`` is the explicit discriminator: callers must branch on
    ``ALREADY_SYMMETRIC`` before treating an empty source list as failure.
    ``path_edges`` is the full discovered path (on-plane edges included).
    """

    source_edges: list[bmesh.types.BMEdge]
    side: str | None
    total_path_edges: int
    crossing_count: int
    status: SourcePathStatus
    path_edges: list[bmesh.types.BMEdge]

    def __iter__(self) -> Iterator[object]:
        yield self.source_edges
        yield self.side
        yield self.total_path_edges
        yield self.crossing_count


def _path_edges_already_symmetric(
    by_side: Mapping[str, Sequence[bmesh.types.BMEdge]],
) -> bool:
    """True when every path edge lies on the mirror plane (contract §4)."""

    return (
        len(by_side["PLANE"]) > 0
        and len(by_side["POSITIVE"]) == 0
        and len(by_side["NEGATIVE"]) == 0
        and len(by_side["CROSSES"]) == 0
    )


def collect_source_path_edges(
    bm: bmesh.types.BMesh,
    axis_index: int,
    tolerance: float,
    requested_side: str,
    *,
    selected_only: bool = False,
) -> SourcePathEdges:
    """Return path edges on one source half (Loop Cut / Offset; one-side).

    Knife uses :func:`collect_knife_path_edges_by_side` instead. The two final
    integers are the total number of new path edges and the number that cross
    the mirror plane.

    All-PLANE paths return ``status=ALREADY_SYMMETRIC`` before source-side
    resolution, independent of the requested AUTO / POSITIVE / NEGATIVE side.
    """

    all_path_edges = _discover_path_edges(bm, selected_only=selected_only)
    if not all_path_edges and bm.edges.layers.int.get(EDGE_ORIGINAL_LAYER) is None:
        return SourcePathEdges([], None, 0, 0, status="OK", path_edges=[])

    by_side = classify_path_edges_by_side(all_path_edges, axis_index, tolerance)
    if _path_edges_already_symmetric(by_side):
        return SourcePathEdges(
            [],
            None,
            len(all_path_edges),
            0,
            status=ALREADY_SYMMETRIC,
            path_edges=list(all_path_edges),
        )

    side, crossing = choose_source_side(all_path_edges, axis_index, tolerance, requested_side)
    if side is None:
        return SourcePathEdges(
            [],
            None,
            len(all_path_edges),
            crossing,
            status="UNDETERMINED",
            path_edges=list(all_path_edges),
        )

    return SourcePathEdges(
        list(by_side[side]),
        side,
        len(all_path_edges),
        crossing,
        status="OK",
        path_edges=list(all_path_edges),
    )


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
