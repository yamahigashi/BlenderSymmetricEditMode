# SPDX-License-Identifier: GPL-3.0-or-later

"""Involutive vertex/edge/face pair tables and leading-domain expansion plans."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import bmesh
import numpy

from . import matching
from .matching import _one_sided_pair_table
from .snapshot import capture_selection_snapshot

if TYPE_CHECKING:
    from numpy.typing import NDArray


class _DensePartnerMap(Mapping[int, int | None]):
    """Read-only mapping view over a dense ``-1``-for-unmatched pair column."""

    __slots__ = ("_partners",)

    def __init__(self, partners: NDArray[numpy.int64]) -> None:
        self._partners = partners

    def __getitem__(self, key: int) -> int | None:
        if not isinstance(key, (int, numpy.integer)):
            raise KeyError(key)
        index = int(key)
        if index < 0 or index >= len(self._partners):
            raise KeyError(key)
        partner = int(self._partners[index])
        return None if partner < 0 else partner

    def __iter__(self):
        return iter(range(len(self._partners)))

    def __len__(self) -> int:
        return len(self._partners)

    def __eq__(self, other) -> bool:
        if isinstance(other, _DensePartnerMap):
            return numpy.array_equal(self._partners, other._partners)
        if not isinstance(other, Mapping) or len(self) != len(other):
            return False
        missing = object()
        return all(other.get(index, missing) == self[index] for index in self)

    def __repr__(self) -> str:
        unmatched = int(numpy.count_nonzero(self._partners < 0))
        return f"_DensePartnerMap(count={len(self)}, unmatched={unmatched})"


@dataclass(frozen=True)
class ElementPairMaps:
    """Involutive element correspondence used by leading-domain expansion.

    ``None`` values mean the element has no unique counterpart (unmatched
    endpoints, multi-edges, or colliding face keys).
    """

    vert_pairs: dict[int, int]
    edge_pair_by_index: Mapping[int, int | None]
    face_pair_by_index: Mapping[int, int | None]
    _vertex_partner_indices: NDArray[numpy.int64] | None = field(default=None, repr=False, compare=False)
    _edge_partner_indices: NDArray[numpy.int64] | None = field(default=None, repr=False, compare=False)
    _face_partner_indices: NDArray[numpy.int64] | None = field(default=None, repr=False, compare=False)
    _edge_vertices: NDArray[numpy.int64] | None = field(default=None, repr=False, compare=False)
    _loop_verts: NDArray[numpy.int64] | None = field(default=None, repr=False, compare=False)
    _loop_starts: NDArray[numpy.int64] | None = field(default=None, repr=False, compare=False)
    _loop_totals: NDArray[numpy.int64] | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class ExpansionPlan:
    """Indices to select for a leading-domain mirror expansion.

    Counts track unmatched / hidden counterparts; they do not alter select.
    """

    add_vert_indices: tuple[int, ...]
    add_edge_indices: tuple[int, ...]
    add_face_indices: tuple[int, ...]
    unmatched_count: int
    hidden_counterpart_count: int


def _vertex_pair_arrays(
    coords: NDArray[numpy.float64],
    axis_index: int,
    tolerance: float,
) -> tuple[dict[int, int], NDArray[numpy.int64]]:
    """Resolve all vertex pairs once and retain a dense partner column."""

    count = len(coords)
    pairs = _one_sided_pair_table(coords, axis_index, tolerance)
    if pairs is None:
        pairs = matching.build_vertex_pair_table(coords, axis_index, tolerance)

    dense = numpy.full(count, -1, dtype=numpy.int64)
    if pairs:
        sources = numpy.fromiter(pairs, dtype=numpy.int64, count=len(pairs))
        dense[sources] = numpy.fromiter(pairs.values(), dtype=numpy.int64, count=len(pairs))
    return pairs, dense


def _edge_vertex_rows(bm: bmesh.types.BMesh, mesh_object) -> NDArray[numpy.int64]:
    """Capture edge endpoints through Mesh bulk when its index order is safe."""

    count = len(bm.edges)
    data = getattr(mesh_object, "data", None) if mesh_object is not None else None
    edges = getattr(data, "edges", None) if data is not None else None
    if (
        data is not None
        and getattr(data, "shape_keys", None) is None
        and edges is not None
        and len(edges) == count
        and callable(getattr(edges, "foreach_get", None))
    ):
        endpoints32 = numpy.empty(count * 2, dtype=numpy.int32)
        try:
            edges.foreach_get("vertices", endpoints32)
            endpoints = endpoints32.astype(numpy.int64).reshape((-1, 2))
            if count:
                first = sorted(vertex.index for vertex in bm.edges[0].verts)
                last = sorted(vertex.index for vertex in bm.edges[count - 1].verts)
                if sorted(endpoints[0].tolist()) != first or sorted(endpoints[-1].tolist()) != last:
                    raise ValueError("Mesh/BMesh edge order mismatch")
            return endpoints
        except (AttributeError, RuntimeError, TypeError, ValueError):
            pass

    return numpy.fromiter(
        (vertex.index for edge in bm.edges for vertex in edge.verts),
        dtype=numpy.int64,
        count=count * 2,
    ).reshape((-1, 2))


def _unique_row_partner_indices(
    rows: NDArray[numpy.int64],
    mapped_rows: NDArray[numpy.int64],
    mapped_valid: NDArray[numpy.bool_],
) -> NDArray[numpy.int64]:
    """Match unordered fixed-width rows, rejecting source/destination collisions."""

    count, width = rows.shape
    partners = numpy.full(count, -1, dtype=numpy.int64)
    if count == 0 or width == 0:
        return partners

    canonical: numpy.ndarray = numpy.sort(rows, axis=1)
    bits_per_value = max(1, int(canonical.max(initial=0)).bit_length())
    use_packed = width * bits_per_value <= 64
    if not use_packed:
        row_dtype = numpy.dtype((numpy.void, canonical.dtype.itemsize * width))

    def _row_keys(rows_sorted: numpy.ndarray, *, clamp: bool) -> numpy.ndarray:
        if use_packed:
            keys = numpy.zeros(count, dtype=numpy.uint64)
            for column in rows_sorted.T:
                values = numpy.maximum(column, 0) if clamp else column
                keys = (keys << bits_per_value) | values.astype(numpy.uint64)
            return keys
        return numpy.ascontiguousarray(rows_sorted).view(row_dtype).reshape(-1)

    keys = _row_keys(canonical, clamp=False)
    unique_keys, first_rows, inverse, counts = numpy.unique(
        keys,
        return_index=True,
        return_inverse=True,
        return_counts=True,
    )

    mapped = numpy.sort(mapped_rows, axis=1)
    mapped_keys = _row_keys(mapped, clamp=True)
    destinations = numpy.searchsorted(unique_keys, mapped_keys)
    clipped = numpy.minimum(destinations, max(len(unique_keys) - 1, 0))
    found = (destinations < len(unique_keys)) & (unique_keys[clipped] == mapped_keys)
    usable = mapped_valid & (counts[inverse] == 1) & found
    usable &= counts[clipped] == 1
    partners[usable] = first_rows[clipped[usable]]
    return partners


def _face_partner_indices(
    loop_verts: NDArray[numpy.int64],
    loop_starts: NDArray[numpy.int64],
    loop_totals: NDArray[numpy.int64],
    vertex_partners: NDArray[numpy.int64],
) -> NDArray[numpy.int64]:
    """Build face partners in degree groups using canonical row keys."""

    partners = numpy.full(len(loop_starts), -1, dtype=numpy.int64)
    for raw_total in numpy.unique(loop_totals):
        total = int(raw_total)
        face_indices = numpy.flatnonzero(loop_totals == total)
        if total == 0:
            continue
        positions = loop_starts[face_indices, None] + numpy.arange(total, dtype=numpy.int64)
        rows = loop_verts[positions]
        mapped_rows = vertex_partners[rows]
        local_partners = _unique_row_partner_indices(
            rows,
            mapped_rows,
            numpy.all(mapped_rows >= 0, axis=1),
        )
        matched = local_partners >= 0
        partners[face_indices[matched]] = face_indices[local_partners[matched]]
    return partners


def build_element_pair_maps(
    bm: bmesh.types.BMesh,
    axis_index: int,
    tolerance: float,
    *,
    mesh_object=None,
) -> ElementPairMaps:
    """Resolve global vertex/edge/face counterparts through bulk row arrays."""

    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    bm.verts.index_update()
    bm.edges.index_update()
    bm.faces.index_update()

    capture = capture_selection_snapshot(
        bm,
        mesh_object=mesh_object,
        domains=("FACE",),
        include_history=False,
        include_loops=True,
    )
    vert_pairs, vertex_partners = _vertex_pair_arrays(capture.coords, axis_index, tolerance)
    edge_vertices = _edge_vertex_rows(bm, mesh_object)
    mapped_edge_vertices = vertex_partners[edge_vertices]
    edge_partners = _unique_row_partner_indices(
        edge_vertices,
        mapped_edge_vertices,
        numpy.all(mapped_edge_vertices >= 0, axis=1),
    )
    face_partners = _face_partner_indices(
        capture.loop_verts,
        capture.loop_starts,
        capture.loop_totals,
        vertex_partners,
    )

    return ElementPairMaps(
        vert_pairs=vert_pairs,
        edge_pair_by_index=_DensePartnerMap(edge_partners),
        face_pair_by_index=_DensePartnerMap(face_partners),
        _vertex_partner_indices=vertex_partners,
        _edge_partner_indices=edge_partners,
        _face_partner_indices=face_partners,
        _edge_vertices=edge_vertices,
        _loop_verts=capture.loop_verts,
        _loop_starts=capture.loop_starts,
        _loop_totals=capture.loop_totals,
    )


def plan_leading_domain_expansion(
    bm: bmesh.types.BMesh,
    pair_maps: ElementPairMaps,
    *,
    domains: tuple[str, ...],
) -> ExpansionPlan:
    """Plan select expansion for leading domains without mutating *bm*."""

    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()

    add_verts: list[int] = []
    add_edges: list[int] = []
    add_faces: list[int] = []
    unmatched_count = 0
    hidden_counterpart_count = 0

    if "VERT" in domains:
        for vertex in bm.verts:
            if not vertex.select or vertex.hide:
                continue
            partner = pair_maps.vert_pairs.get(vertex.index)
            if partner is None:
                unmatched_count += 1
                continue
            if partner == vertex.index:
                continue
            counterpart = bm.verts[partner]
            # Hidden first: a hidden+selected counterpart must still decline.
            if counterpart.hide:
                hidden_counterpart_count += 1
                continue
            if counterpart.select:
                continue
            add_verts.append(partner)

    if "EDGE" in domains:
        for edge in bm.edges:
            if not edge.select or edge.hide:
                continue
            partner = pair_maps.edge_pair_by_index.get(edge.index)
            if partner is None:
                unmatched_count += 1
                continue
            if partner == edge.index:
                continue
            counterpart = bm.edges[partner]
            if counterpart.hide:
                hidden_counterpart_count += 1
                continue
            if counterpart.select:
                continue
            add_edges.append(partner)

    if "FACE" in domains:
        for face in bm.faces:
            if not face.select or face.hide:
                continue
            partner = pair_maps.face_pair_by_index.get(face.index)
            if partner is None:
                unmatched_count += 1
                continue
            if partner == face.index:
                continue
            counterpart = bm.faces[partner]
            if counterpart.hide:
                hidden_counterpart_count += 1
                continue
            if counterpart.select:
                continue
            add_faces.append(partner)

    return ExpansionPlan(
        add_vert_indices=tuple(add_verts),
        add_edge_indices=tuple(add_edges),
        add_face_indices=tuple(add_faces),
        unmatched_count=unmatched_count,
        hidden_counterpart_count=hidden_counterpart_count,
    )


def apply_expansion_plan(bm: bmesh.types.BMesh, plan: ExpansionPlan) -> None:
    """Set select on planned indices only; does not call select_flush_mode."""

    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()

    for index in plan.add_vert_indices:
        bm.verts[index].select = True
    for index in plan.add_edge_indices:
        bm.edges[index].select = True
    for index in plan.add_face_indices:
        bm.faces[index].select = True
