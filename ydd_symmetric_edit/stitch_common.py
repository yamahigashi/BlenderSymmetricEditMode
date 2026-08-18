# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

import bmesh
from mathutils import Vector

from ._types import FaceId, QuantizedCoordinate
from .matching import (
    _chebyshev_distance_3d,
    _iter_quantized_neighborhood,
    _quantized_coordinate,
    coordinates_match,
    mirror_coordinate,
)


@dataclass
class SelectionMutationSummary:
    """Elements whose native selection may have changed during a topology apply.

    The tuples intentionally retain invalid BMesh proxies: the scoped restore
    filters them with ``is_valid`` and can therefore handle pointmerge safely.
    ``complete`` is false only when an apply reports a partial/unknown result.
    """

    vertices: tuple[bmesh.types.BMVert, ...] = ()
    edges: tuple[bmesh.types.BMEdge, ...] = ()
    faces: tuple[bmesh.types.BMFace, ...] = ()
    complete: bool = True


def combine_selection_mutation_summaries(
    summaries: Iterable[SelectionMutationSummary],
) -> SelectionMutationSummary:
    vertices: set[bmesh.types.BMVert] = set()
    edges: set[bmesh.types.BMEdge] = set()
    faces: set[bmesh.types.BMFace] = set()
    complete = True
    for summary in summaries:
        vertices.update(summary.vertices)
        edges.update(summary.edges)
        faces.update(summary.faces)
        complete &= summary.complete
    return SelectionMutationSummary(tuple(vertices), tuple(edges), tuple(faces), complete)


class _SelectionMutationTracker:
    def __init__(self) -> None:
        self.vertices: set[bmesh.types.BMVert] = set()
        self.edges: set[bmesh.types.BMEdge] = set()
        self.faces: set[bmesh.types.BMFace] = set()

    # Downward-closure invariant: every face in the summary carries its full
    # boundary edges/vertices and every edge carries both vertices. Clearing a
    # summary face/edge flushes deselection down to its boundary, so partial
    # capture would wipe untracked independently-selected neighbors that the
    # scoped restore never revisits.
    def _fold_edge(self, edge) -> None:
        self.edges.add(edge)
        if edge.is_valid:
            self.vertices.update(edge.verts)

    def _fold_face(self, face) -> None:
        self.faces.add(face)
        if face.is_valid:
            for edge in face.edges:
                if edge.is_valid:
                    self._fold_edge(edge)
            self.vertices.update(vertex for vertex in face.verts if vertex.is_valid)

    def add_vertex(self, vertex) -> None:
        self.vertices.add(vertex)
        if vertex.is_valid:
            for edge in vertex.link_edges:
                if edge.is_valid:
                    self._fold_edge(edge)
            for face in vertex.link_faces:
                if face.is_valid:
                    self._fold_face(face)

    def add_edge(self, edge) -> None:
        self._fold_edge(edge)
        if edge.is_valid:
            for face in edge.link_faces:
                if face.is_valid:
                    self._fold_face(face)

    def add_face(self, face) -> None:
        self._fold_face(face)

    def finish(self, *, complete: bool = True) -> SelectionMutationSummary:
        return SelectionMutationSummary(tuple(self.vertices), tuple(self.edges), tuple(self.faces), complete)


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


def cluster_points_by_tolerance(
    points: Sequence[Vector],
    tolerance: float,
) -> list[list[int]]:
    """Lex-order scan clustering: absorb within tol of the representative.

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


def _mirror_invariant_endpoint_key(
    co,
    axis_index: int,
) -> tuple[float, float, float]:
    """Canonical endpoint key with |axis| so ρ-related seeds share the key.

    Absolute value on the mirror axis makes (M,I) and (ρM,ρI) order-equivalent
    for seed selection. Full orbit ties are allowed as best-effort attribute
    inheritance (shape keys etc.).
    """

    components = [float(co[0]), float(co[1]), float(co[2])]
    components[axis_index] = abs(components[axis_index])
    return (components[0], components[1], components[2])


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


_EdgeEndpointEntry = tuple[
    int | None,
    tuple[float, float, float],
    tuple[float, float, float],
    frozenset[FaceId] | None,
]


_EdgeEndpointStore = dict[QuantizedCoordinate, list[_EdgeEndpointEntry]]


def _register_edge_endpoint_pair(
    store: _EdgeEndpointStore,
    a: Vector,
    b: Vector,
    tolerance: float,
    marker: int | None = None,
    face_ids: Iterable[FaceId] | None = None,
) -> None:
    """Store real endpoint coords under the primary bin of the canonical first endpoint."""

    primary_a, coord_a, coord_b = _canonical_edge_endpoints(a, b, tolerance)
    stored_face_ids = frozenset(face_ids) if face_ids is not None else None
    store.setdefault(primary_a, []).append((marker, coord_a, coord_b, stored_face_ids))


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
            for marker, stored_a, stored_b, _face_ids in store.get(bin_key, ()):
                distance_a = _chebyshev_distance_3d(first_coord, stored_a)
                distance_b = _chebyshev_distance_3d(second_coord, stored_b)
                if distance_a > tolerance or distance_b > tolerance:
                    continue
                score = max(distance_a, distance_b)
                if best is None or score < best[0]:
                    best = (score, marker)
    return best


def _match_edge_endpoint_pair_for_faces(
    a: Vector,
    b: Vector,
    tolerance: float,
    store: _EdgeEndpointStore,
    possible_target_ids: set[FaceId],
) -> Literal["no_match", "match", "ambiguous"]:
    matches: set[tuple[QuantizedCoordinate, int]] = set()
    for query_first, query_second in ((a, b), (b, a)):
        first_coord = _coordinate_tuple(query_first)
        second_coord = _coordinate_tuple(query_second)
        for bin_key in _iter_quantized_neighborhood(query_first, tolerance):
            for entry_index, (_marker, stored_a, stored_b, face_ids) in enumerate(store.get(bin_key, ())):
                if face_ids is None or not face_ids.intersection(possible_target_ids):
                    continue
                if _chebyshev_distance_3d(first_coord, stored_a) > tolerance:
                    continue
                if _chebyshev_distance_3d(second_coord, stored_b) > tolerance:
                    continue
                matches.add((bin_key, entry_index))
    if not matches:
        return "no_match"
    if len(matches) > 1:
        return "ambiguous"
    return "match"


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
