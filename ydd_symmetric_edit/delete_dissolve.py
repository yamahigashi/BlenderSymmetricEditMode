# SPDX-License-Identifier: GPL-3.0-or-later

"""Pure helpers for symmetric delete / dissolve selection expansion."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

from . import core

if TYPE_CHECKING:
    import bmesh


@dataclass(frozen=True)
class ElementPairMaps:
    """Involutive element correspondence used by leading-domain expansion.

    ``None`` values mean the element has no unique counterpart (unmatched
    endpoints, multi-edges, or colliding face keys).
    """

    vert_pairs: dict[int, int]
    edge_pair_by_index: dict[int, int | None]
    face_pair_by_index: dict[int, int | None]


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


def build_element_pair_maps(
    bm: bmesh.types.BMesh,
    axis_index: int,
    tolerance: float,
) -> ElementPairMaps:
    """Resolve vertex / edge / face counterparts for selection expansion."""

    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    bm.verts.index_update()
    bm.edges.index_update()
    bm.faces.index_update()

    coords = [vertex.co for vertex in bm.verts]
    vert_pairs = core.build_vertex_pair_table(coords, axis_index, tolerance)

    endpoint_to_edges: dict[frozenset[int], list[int]] = defaultdict(list)
    for edge in bm.edges:
        key = frozenset((edge.verts[0].index, edge.verts[1].index))
        endpoint_to_edges[key].append(edge.index)

    multi_edge_keys = {key for key, indices in endpoint_to_edges.items() if len(indices) > 1}

    edge_pair_by_index: dict[int, int | None] = {}
    for edge in bm.edges:
        v1 = edge.verts[0].index
        v2 = edge.verts[1].index
        own_key = frozenset((v1, v2))
        if own_key in multi_edge_keys:
            edge_pair_by_index[edge.index] = None
            continue
        p1 = vert_pairs.get(v1)
        p2 = vert_pairs.get(v2)
        if p1 is None or p2 is None:
            edge_pair_by_index[edge.index] = None
            continue
        counterpart_key = frozenset((p1, p2))
        if counterpart_key in multi_edge_keys:
            edge_pair_by_index[edge.index] = None
            continue
        candidates = endpoint_to_edges.get(counterpart_key)
        if not candidates:
            edge_pair_by_index[edge.index] = None
            continue
        edge_pair_by_index[edge.index] = candidates[0]

    face_ids_by_vertex_set: dict[frozenset[int], list[int]] = defaultdict(list)
    for face in bm.faces:
        key = frozenset(vertex.index for vertex in face.verts)
        face_ids_by_vertex_set[key].append(face.index)

    conflicted_keys = {key for key, indices in face_ids_by_vertex_set.items() if len(indices) > 1}

    face_pair_by_index: dict[int, int | None] = {}
    for face in bm.faces:
        own_verts = [vertex.index for vertex in face.verts]
        own_key = frozenset(own_verts)
        if own_key in conflicted_keys:
            face_pair_by_index[face.index] = None
            continue
        mapped: list[int] = []
        missing = False
        for vertex_index in own_verts:
            partner = vert_pairs.get(vertex_index)
            if partner is None:
                missing = True
                break
            mapped.append(partner)
        if missing:
            face_pair_by_index[face.index] = None
            continue
        counterpart_key = frozenset(mapped)
        if counterpart_key in conflicted_keys:
            face_pair_by_index[face.index] = None
            continue
        counterparts = face_ids_by_vertex_set.get(counterpart_key)
        if not counterparts:
            face_pair_by_index[face.index] = None
            continue
        face_pair_by_index[face.index] = counterparts[0]

    return ElementPairMaps(
        vert_pairs=vert_pairs,
        edge_pair_by_index=edge_pair_by_index,
        face_pair_by_index=face_pair_by_index,
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
            if counterpart.select:
                continue
            if counterpart.hide:
                hidden_counterpart_count += 1
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
            if counterpart.select:
                continue
            if counterpart.hide:
                hidden_counterpart_count += 1
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
            if counterpart.select:
                continue
            if counterpart.hide:
                hidden_counterpart_count += 1
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
