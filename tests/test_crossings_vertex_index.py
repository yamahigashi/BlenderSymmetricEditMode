# SPDX-License-Identifier: GPL-3.0-or-later

"""Headless differential checks for crossings vertex-scan scoping (§I-3a).

Contract: .agents/doc/perf_epoch_finish_plan6_2026-08-12.md (v4), §I-3a.
The oracle freezes the pre-U6-3a full ``bm.verts`` listcomp scans. It does not
call or share the candidate's quantized index helpers and always runs on a
separate BMesh clone.

Marker: YSE_CROSSINGS_INDEX_TEST_OK
"""

from __future__ import annotations

import math
import sys
import traceback
from collections import defaultdict
from pathlib import Path

import bmesh
from mathutils import Vector

PACKAGE_PARENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_PARENT))

from ydd_symmetric_edit import layer_names, matching, stitch_crossings  # noqa: E402

AXIS = matching.AXIS_INDEX["X"]
TOLERANCE = 1.0e-5
MARKER = "YSE_CROSSINGS_INDEX_TEST_OK"


# ---------------------------------------------------------------------------
# Frozen pre-U6-3a oracle (independent copy; no index helpers)
# ---------------------------------------------------------------------------


def _frozen_plan_tolerance(plan):
    return plan[0].tolerance if plan else 0.0


def _frozen_apply_mirrored_path_crossings(bm, plan):
    """Pre-U6-3a apply with full-vertex listcomps frozen here (§I-3a oracle)."""

    if not plan:
        return 0, ""
    marker_layer = bm.edges.layers.int.get(layer_names.EDGE_ORIGINAL_LAYER)
    vertex_selection_layer = bm.verts.layers.int.get(layer_names.VERT_SELECTION_LAYER)
    edge_selection_layer = bm.edges.layers.int.get(layer_names.EDGE_SELECTION_LAYER)
    if marker_layer is None or vertex_selection_layer is None or edge_selection_layer is None:
        return 0, "temporary topology or selection markers are missing"

    applications = []
    for cluster in plan:
        if cluster.positive:
            applications.append((cluster.positive_coordinate, cluster.positive))
        if cluster.negative:
            applications.append((cluster.negative_coordinate, cluster.negative))

    native_edge_selection = {}
    native_vertex_selection = {}
    edge_marker = {}
    endpoint_vertex_by_occurrence = {}
    participant_endpoints = []
    for _coordinate, occurrences in applications:
        endpoints = set()
        for occurrence in occurrences:
            edge = occurrence.edge
            if not edge.is_valid:
                return 0, "a mirrored crossing source edge was lost before stitching"
            edge_key = occurrence.edge_id
            native_edge_selection.setdefault(edge_key, bool(edge.select))
            edge_marker.setdefault(edge_key, int(edge[marker_layer]))
            if occurrence.endpoint_index is not None:
                endpoint = edge.verts[occurrence.endpoint_index]
                endpoints.add(endpoint)
                endpoint_vertex_by_occurrence[id(occurrence)] = endpoint
                native_vertex_selection.setdefault(hash(endpoint), bool(endpoint.select))
        participant_endpoints.append(endpoints)

    reusable_vertex = []
    for application_index, (coordinate, _occurrences) in enumerate(applications):
        extras = [
            vertex
            for vertex in bm.verts
            if vertex.is_valid
            and vertex not in participant_endpoints[application_index]
            and matching.coordinates_match(vertex.co, coordinate, _frozen_plan_tolerance(plan))
        ]
        if len(extras) > 1:
            return 0, "multiple existing vertices are ambiguous at a mirrored cut intersection"
        reusable_vertex.append(extras[0] if extras else None)
        if extras:
            native_vertex_selection.setdefault(hash(extras[0]), bool(extras[0].select))

    split_entries_by_edge = defaultdict(list)
    edge_by_key = {}
    for application_index, (_coordinate, occurrences) in enumerate(applications):
        for occurrence in occurrences:
            if occurrence.endpoint_index is not None:
                continue
            key = occurrence.edge_id
            edge_by_key[key] = occurrence.edge
            split_entries_by_edge[key].append((occurrence.factor, application_index, occurrence))

    vertex_by_occurrence = dict(endpoint_vertex_by_occurrence)
    for edge_key, entries in split_entries_by_edge.items():
        original_edge = edge_by_key[edge_key]
        if not original_edge.is_valid:
            return 0, "a mirrored crossing source edge was lost during stitching"
        original_start = original_edge.verts[0]
        original_end = original_edge.verts[1]
        descendant = original_edge
        descendant_start = original_start
        interval_start = 0.0
        selected = native_edge_selection[edge_key]
        marker = edge_marker[edge_key]
        entries.sort(key=lambda entry: entry[0])
        for factor, _application_index, occurrence in entries:
            if factor <= interval_start or factor >= 1.0:
                return 0, "a mirrored crossing split factor is not interior to its descendant edge"
            local_factor = (factor - interval_start) / (1.0 - interval_start)
            try:
                new_edge, new_vertex = bmesh.utils.edge_split(
                    descendant,
                    descendant_start,
                    local_factor,
                )
            except (RuntimeError, ValueError) as exc:
                return 0, f"could not split a mirrored path crossing edge: {exc}"

            for half_edge in (descendant, new_edge):
                half_edge[marker_layer] = marker
                half_edge.select = selected
                half_edge[edge_selection_layer] = int(selected)
            new_vertex.select = selected
            new_vertex[vertex_selection_layer] = int(selected)
            native_vertex_selection[hash(new_vertex)] = selected
            vertex_by_occurrence[id(occurrence)] = new_vertex

            descendants = [edge for edge in (descendant, new_edge) if original_end in edge.verts]
            if len(descendants) != 1:
                return 0, "could not track a mirrored crossing descendant edge"
            descendant = descendants[0]
            descendant_start = new_vertex
            interval_start = factor

    for application_index, (coordinate, occurrences) in enumerate(applications):
        vertices = []
        edge_key_by_vertex = {}
        for occurrence in occurrences:
            vertex = vertex_by_occurrence.get(id(occurrence))
            if vertex is None or not vertex.is_valid:
                return 0, "a mirrored crossing vertex was lost before cluster unification"
            if vertex not in vertices:
                vertices.append(vertex)
            vertex_key = hash(vertex)
            current_key = edge_key_by_vertex.get(vertex_key)
            if current_key is None or occurrence.edge_key < current_key:
                edge_key_by_vertex[vertex_key] = occurrence.edge_key

        existing = reusable_vertex[application_index]
        if existing is not None:
            if not existing.is_valid:
                return 0, "an existing mirrored crossing vertex was lost before reuse"
            if existing not in vertices:
                vertices.append(existing)
            survivor = existing
        else:
            survivor = min(vertices, key=lambda vertex: edge_key_by_vertex[hash(vertex)])

        selected = any(native_edge_selection[occurrence.edge_id] for occurrence in occurrences)
        selected |= any(native_vertex_selection.get(hash(vertex), bool(vertex.select)) for vertex in vertices)
        snapshot_selected = selected or any(
            bool(vertex[vertex_selection_layer]) for vertex in vertices if vertex.is_valid
        )
        survivor.co = coordinate.copy()
        for vertex in list(vertices):
            if vertex == survivor:
                continue
            try:
                bmesh.ops.pointmerge(
                    bm,
                    verts=[survivor, vertex],
                    merge_co=coordinate,
                )
            except (RuntimeError, ValueError) as exc:
                return 0, f"could not unify mirrored crossing vertices: {exc}"
            if not survivor.is_valid:
                return 0, "the mirrored crossing survivor was lost during point merge"
        survivor.co = coordinate.copy()
        survivor.select = selected
        survivor[vertex_selection_layer] = int(snapshot_selected)

        ambiguous = [
            vertex
            for vertex in bm.verts
            if vertex.is_valid
            and vertex != survivor
            and matching.coordinates_match(vertex.co, coordinate, _frozen_plan_tolerance(plan))
        ]
        if ambiguous:
            return 0, "a separate existing vertex remains within tolerance of a mirrored cut intersection"

    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.normal_update()
    return len(applications), ""


# ---------------------------------------------------------------------------
# Mesh state / clone helpers (mirror test_scoped_edge_store patterns)
# ---------------------------------------------------------------------------


def _update_indices(bm):
    for elements in (bm.verts, bm.edges, bm.faces):
        elements.ensure_lookup_table()
        elements.index_update()


def _copy_int_layers(source_elements, clone_elements):
    for name in source_elements.layers.int.keys():
        source_layer = source_elements.layers.int.get(name)
        clone_layer = clone_elements.layers.int.new(name)
        for source, clone in zip(source_elements, clone_elements, strict=True):
            clone[clone_layer] = int(source[source_layer])


def _clone_bmesh(source):
    _update_indices(source)
    clone = bmesh.new()
    for vertex in source.verts:
        clone.verts.new(tuple(vertex.co))
    clone.verts.ensure_lookup_table()
    for edge in source.edges:
        clone.edges.new(
            (
                clone.verts[edge.verts[0].index],
                clone.verts[edge.verts[1].index],
            )
        )
    clone.edges.ensure_lookup_table()
    for face in source.faces:
        clone.faces.new([clone.verts[vertex.index] for vertex in face.verts])
    clone.faces.ensure_lookup_table()

    _copy_int_layers(source.verts, clone.verts)
    _copy_int_layers(source.edges, clone.edges)
    _copy_int_layers(source.faces, clone.faces)
    for source_vertex, clone_vertex in zip(source.verts, clone.verts, strict=True):
        clone_vertex.select = bool(source_vertex.select)
        clone_vertex.hide = bool(source_vertex.hide)
    for source_edge, clone_edge in zip(source.edges, clone.edges, strict=True):
        clone_edge.select = bool(source_edge.select)
        clone_edge.hide = bool(source_edge.hide)
    for source_face, clone_face in zip(source.faces, clone.faces, strict=True):
        clone_face.select = bool(source_face.select)
        clone_face.hide = bool(source_face.hide)
    _update_indices(clone)
    clone.normal_update()
    return clone


def _float_bits(value):
    return float(value).hex()


def _canonical_cycle(indices):
    values = tuple(indices)
    rotations = []
    for sequence in (values, tuple(reversed(values))):
        rotations.extend(sequence[offset:] + sequence[:offset] for offset in range(len(sequence)))
    return min(rotations)


def _layer_state(elements):
    return tuple(
        (
            name,
            tuple(int(element[elements.layers.int.get(name)]) for element in elements),
        )
        for name in sorted(elements.layers.int.keys())
    )


def _mesh_state(bm):
    _update_indices(bm)
    vertices = tuple(
        (
            tuple(_float_bits(component) for component in vertex.co),
            bool(vertex.select),
            bool(vertex.hide),
        )
        for vertex in bm.verts
    )
    edges = tuple(
        sorted(
            (
                tuple(sorted((edge.verts[0].index, edge.verts[1].index))),
                tuple(sorted(face.index for face in edge.link_faces if face.is_valid)),
                bool(edge.select),
                bool(edge.hide),
            )
            for edge in bm.edges
            if edge.is_valid
        )
    )
    faces = tuple(
        sorted(
            (
                _canonical_cycle(tuple(vertex.index for vertex in face.verts)),
                tuple(sorted(edge.index for edge in face.edges if edge.is_valid)),
                bool(face.select),
                bool(face.hide),
            )
            for face in bm.faces
            if face.is_valid
        )
    )
    return (
        (len(bm.verts), len(bm.edges), len(bm.faces)),
        vertices,
        edges,
        faces,
        _layer_state(bm.verts),
        _layer_state(bm.edges),
        _layer_state(bm.faces),
    )


def _ensure_layers(bm):
    if bm.edges.layers.int.get(layer_names.EDGE_ORIGINAL_LAYER) is None:
        bm.edges.layers.int.new(layer_names.EDGE_ORIGINAL_LAYER)
    if bm.edges.layers.int.get(layer_names.EDGE_SELECTION_LAYER) is None:
        bm.edges.layers.int.new(layer_names.EDGE_SELECTION_LAYER)
    if bm.verts.layers.int.get(layer_names.VERT_SELECTION_LAYER) is None:
        bm.verts.layers.int.new(layer_names.VERT_SELECTION_LAYER)
    _update_indices(bm)


def _edge_key_for_occurrence(edge, axis_index=AXIS):
    return stitch_crossings._edge_survivor_key(edge, axis_index)


def _make_occurrence(edge, factor, endpoint_index=None, axis_index=AXIS):
    return stitch_crossings._MirroredPathOccurrence(
        edge=edge,
        edge_id=hash(edge),
        factor=float(factor),
        endpoint_index=endpoint_index,
        edge_key=_edge_key_for_occurrence(edge, axis_index),
    )


def _make_cluster(positive_coord, negative_coord, positive, negative, tolerance=TOLERANCE):
    return stitch_crossings._MirroredPathCrossingCluster(
        positive_coordinate=Vector(positive_coord),
        negative_coordinate=Vector(negative_coord),
        positive=tuple(positive),
        negative=tuple(negative),
        tolerance=tolerance,
    )


def _remap_plan(plan, source_bm, target_bm):
    """Rebuild plan occurrences so edges point at *target_bm* by index."""

    _update_indices(source_bm)
    _update_indices(target_bm)
    remapped = []
    for cluster in plan:

        def remap_occurrences(occurrences):
            result = []
            for occurrence in occurrences:
                edge_index = occurrence.edge.index
                target_edge = target_bm.edges[edge_index]
                result.append(
                    stitch_crossings._MirroredPathOccurrence(
                        edge=target_edge,
                        edge_id=hash(target_edge),
                        factor=occurrence.factor,
                        endpoint_index=occurrence.endpoint_index,
                        edge_key=_edge_key_for_occurrence(target_edge),
                    )
                )
            return tuple(result)

        remapped.append(
            stitch_crossings._MirroredPathCrossingCluster(
                positive_coordinate=cluster.positive_coordinate.copy(),
                negative_coordinate=cluster.negative_coordinate.copy(),
                positive=remap_occurrences(cluster.positive),
                negative=remap_occurrences(cluster.negative),
                tolerance=cluster.tolerance,
            )
        )
    return remapped


def _run_apply(function, bm, plan):
    try:
        result = ("return", function(bm, plan))
    except Exception as exc:  # Oracle/candidate may surface build-time failures.
        result = ("exception", type(exc).__module__, type(exc).__qualname__, tuple(map(str, exc.args)))
    return result, _mesh_state(bm)


def _assert_differential(bm, plan):
    """Candidate (indexed) vs frozen full-scan oracle on a separate BMesh."""

    _update_indices(bm)
    oracle_bm = _clone_bmesh(bm)
    try:
        oracle_plan = _remap_plan(plan, bm, oracle_bm)
        candidate = _run_apply(stitch_crossings.apply_mirrored_path_crossings, bm, plan)
        oracle = _run_apply(_frozen_apply_mirrored_path_crossings, oracle_bm, oracle_plan)
        assert candidate == oracle, (candidate, oracle)
    finally:
        oracle_bm.free()
    return candidate


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _build_grid(segments: int, size: float = 2.0):
    """create_grid size is half-width: size=2.0 spans [-2, 2] (diag_u6_4 note)."""

    bm = bmesh.new()
    bmesh.ops.create_grid(bm, x_segments=segments, y_segments=segments, size=size)
    _ensure_layers(bm)
    marker = bm.edges.layers.int.get(layer_names.EDGE_ORIGINAL_LAYER)
    for index, edge in enumerate(bm.edges, start=1):
        edge[marker] = index
    _update_indices(bm)
    return bm


def _vertical_edges_at_column(bm, x_value, tol=1.0e-4):
    candidates = []
    for edge in bm.edges:
        v0, v1 = edge.verts
        if abs(float(v0.co.x) - x_value) <= tol and abs(float(v1.co.x) - x_value) <= tol:
            candidates.append((min(float(v0.co.y), float(v1.co.y)), edge))
    candidates.sort(key=lambda item: item[0])
    return [edge for _, edge in candidates]


def _build_track_a_style_plan(bm, segments: int):
    """U6-4 Track A / Track B style: off-lattice mid-edge splits on ± columns."""

    # size half-width => domain [-size, size]; default size=2.0 → [-2, 2].
    size = 2.0
    step = 2.0 * size / segments
    pos_i = int(segments * 0.65)
    neg_i = segments - pos_i

    def col_x(i: int) -> float:
        return -size + i * step

    pos_edges = _vertical_edges_at_column(bm, col_x(pos_i))
    neg_edges = _vertical_edges_at_column(bm, col_x(neg_i))
    assert pos_edges and neg_edges, (len(pos_edges), len(neg_edges), col_x(pos_i), col_x(neg_i))
    row = max(1, len(pos_edges) // 2)
    pos_edge = pos_edges[row]
    neg_edge = neg_edges[row]
    # Off-lattice factor (not 0/1) forces genuine edge_split rather than endpoint reuse.
    factor = 0.37
    pos_y0 = float(pos_edge.verts[0].co.y)
    pos_y1 = float(pos_edge.verts[1].co.y)
    neg_y0 = float(neg_edge.verts[0].co.y)
    neg_y1 = float(neg_edge.verts[1].co.y)
    # edge_split factor is from verts[0]; use that endpoint for coordinates.
    pos_coord = Vector((col_x(pos_i), pos_y0 + factor * (pos_y1 - pos_y0), 0.0))
    neg_coord = Vector((col_x(neg_i), neg_y0 + factor * (neg_y1 - neg_y0), 0.0))
    # Nudge off any lattice point that might coincide with existing verts.
    pos_coord.y += 1.3e-3
    neg_coord.y += 1.3e-3
    pos_occurrence = _make_occurrence(pos_edge, factor)
    neg_occurrence = _make_occurrence(neg_edge, factor)
    return [
        _make_cluster(
            pos_coord,
            neg_coord,
            positive=(pos_occurrence,),
            negative=(neg_occurrence,),
        )
    ]


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def check_track_a_bit_identical():
    """U6-4 Track A equivalent fixture: indexed apply == frozen full-scan bits."""

    # Modest segments keep headless runtime small while preserving off-lattice splits.
    segments = 16
    bm = _build_grid(segments)
    try:
        plan = _build_track_a_style_plan(bm, segments)
        result = _assert_differential(bm, plan)
        assert result[0] == ("return", (2, "")), result[0]
    finally:
        bm.free()


def check_track_a_multi_k_bit_identical():
    """U6-4 Track B-style multi-cluster plan on a grid: bit-identical apply."""

    segments = 24
    k = 4
    bm = _build_grid(segments)
    try:
        size = 2.0
        step = 2.0 * size / segments
        pos_i = int(segments * 0.65)
        neg_i = segments - pos_i

        def col_x(i: int) -> float:
            return -size + i * step

        pos_edges = _vertical_edges_at_column(bm, col_x(pos_i))
        neg_edges = _vertical_edges_at_column(bm, col_x(neg_i))
        margin = 2
        assert min(len(pos_edges), len(neg_edges)) >= k + 2 * margin
        plan = []
        factor = 0.41
        for j in range(k):
            row = margin + j
            pos_edge = pos_edges[row]
            neg_edge = neg_edges[row]
            pos_y0 = float(pos_edge.verts[0].co.y)
            pos_y1 = float(pos_edge.verts[1].co.y)
            neg_y0 = float(neg_edge.verts[0].co.y)
            neg_y1 = float(neg_edge.verts[1].co.y)
            pos_coord = Vector((col_x(pos_i), pos_y0 + factor * (pos_y1 - pos_y0) + 1.1e-3, 0.0))
            neg_coord = Vector((col_x(neg_i), neg_y0 + factor * (neg_y1 - neg_y0) + 1.1e-3, 0.0))
            plan.append(
                _make_cluster(
                    pos_coord,
                    neg_coord,
                    positive=(_make_occurrence(pos_edge, factor),),
                    negative=(_make_occurrence(neg_edge, factor),),
                )
            )
        result = _assert_differential(bm, plan)
        assert result[0] == ("return", (2 * k, "")), result[0]
    finally:
        bm.free()


def check_dense_ambiguous_matches_oracle():
    """Dense degeneracy: multiple verts within tol → same ambiguous extras verdict."""

    bm = bmesh.new()
    try:
        _ensure_layers(bm)
        # Two parallel edges on +X and -X; create two extra verts at the split target.
        a0 = bm.verts.new((1.0, 0.0, 0.0))
        a1 = bm.verts.new((1.0, 1.0, 0.0))
        b0 = bm.verts.new((-1.0, 0.0, 0.0))
        b1 = bm.verts.new((-1.0, 1.0, 0.0))
        pos_edge = bm.edges.new((a0, a1))
        neg_edge = bm.edges.new((b0, b1))
        target = Vector((1.0, 0.5, 0.0))
        # Two non-participant verts inside tolerance of the positive cut coordinate.
        bm.verts.new((1.0 + 0.3 * TOLERANCE, 0.5, 0.0))
        bm.verts.new((1.0 - 0.3 * TOLERANCE, 0.5 + 0.3 * TOLERANCE, 0.0))
        marker = bm.edges.layers.int.get(layer_names.EDGE_ORIGINAL_LAYER)
        pos_edge[marker] = 1
        neg_edge[marker] = 2
        _update_indices(bm)

        plan = [
            _make_cluster(
                target,
                Vector((-1.0, 0.5, 0.0)),
                positive=(_make_occurrence(pos_edge, 0.5),),
                negative=(_make_occurrence(neg_edge, 0.5),),
            )
        ]
        result = _assert_differential(bm, plan)
        assert result[0] == (
            "return",
            (0, "multiple existing vertices are ambiguous at a mirrored cut intersection"),
        ), result[0]
    finally:
        bm.free()


def check_survivor_rebin_across_applications():
    """survivor.co move rebins so a later application still sees the survivor."""

    bm = bmesh.new()
    try:
        _ensure_layers(bm)
        # Application 0: split pos edge, merge at (1.0, 0.5, 0).
        # Application 1: search near a coordinate that lands in a different primary
        # bin after the survivor was moved — index must rebin to remain complete.
        a0 = bm.verts.new((1.0, 0.0, 0.0))
        a1 = bm.verts.new((1.0, 1.0, 0.0))
        b0 = bm.verts.new((-1.0, 0.0, 0.0))
        b1 = bm.verts.new((-1.0, 1.0, 0.0))
        c0 = bm.verts.new((2.0, 0.0, 0.0))
        c1 = bm.verts.new((2.0, 1.0, 0.0))
        d0 = bm.verts.new((-2.0, 0.0, 0.0))
        d1 = bm.verts.new((-2.0, 1.0, 0.0))
        e0 = bm.edges.new((a0, a1))
        e1 = bm.edges.new((b0, b1))
        e2 = bm.edges.new((c0, c1))
        e3 = bm.edges.new((d0, d1))
        marker = bm.edges.layers.int.get(layer_names.EDGE_ORIGINAL_LAYER)
        for index, edge in enumerate((e0, e1, e2, e3), start=1):
            edge[marker] = index
        _update_indices(bm)

        # Two clusters → four applications. First pair unifies at off-lattice points;
        # second pair uses distinct edges so rebin of the first survivor cannot affect
        # completeness of the second search, but both paths must still bit-match.
        plan = [
            _make_cluster(
                Vector((1.0, 0.37, 0.0)),
                Vector((-1.0, 0.37, 0.0)),
                positive=(_make_occurrence(e0, 0.37),),
                negative=(_make_occurrence(e1, 0.37),),
            ),
            _make_cluster(
                Vector((2.0, 0.61, 0.0)),
                Vector((-2.0, 0.61, 0.0)),
                positive=(_make_occurrence(e2, 0.61),),
                negative=(_make_occurrence(e3, 0.61),),
            ),
        ]
        result = _assert_differential(bm, plan)
        assert result[0] == ("return", (4, "")), result[0]
    finally:
        bm.free()


def check_rebin_keeps_nearby_query_complete():
    """After survivor.co rebin, an ambiguous scan still matches the full scan.

    Builds a mesh where pointmerge moves the survivor across a primary bin
    boundary; a separate nearby vertex must remain discoverable (or correctly
    absent) under both implementations.
    """

    bm = bmesh.new()
    try:
        _ensure_layers(bm)
        a0 = bm.verts.new((0.0, 0.0, 0.0))
        a1 = bm.verts.new((0.0, 1.0, 0.0))
        b0 = bm.verts.new((0.5, 0.0, 0.0))
        b1 = bm.verts.new((0.5, 1.0, 0.0))
        # Nearby stray vertex just outside the pre-merge survivor bin but inside
        # tolerance of the final merge coordinate → ambiguous if not merged away.
        stray_offset = TOLERANCE * 0.5
        merge_co = Vector((TOLERANCE * 1.5, 0.4, 0.0))
        bm.verts.new((merge_co.x + stray_offset, merge_co.y, 0.0))
        e0 = bm.edges.new((a0, a1))
        e1 = bm.edges.new((b0, b1))
        marker = bm.edges.layers.int.get(layer_names.EDGE_ORIGINAL_LAYER)
        e0[marker] = 1
        e1[marker] = 2
        _update_indices(bm)

        # Single-sided cluster: only positive application so one unify pass.
        plan = [
            _make_cluster(
                merge_co,
                Vector((-merge_co.x, merge_co.y, 0.0)),
                positive=(
                    _make_occurrence(e0, 0.4),
                    _make_occurrence(e1, 0.4),
                ),
                negative=(),
            )
        ]
        result = _assert_differential(bm, plan)
        # Either both accept or both reject with the same ambiguous reason.
        assert result[0][0] == "return", result[0]
        assert result[0][1][1] in {
            "",
            "a separate existing vertex remains within tolerance of a mirrored cut intersection",
        }, result[0]
    finally:
        bm.free()


def check_nonfinite_fallback_matches_oracle():
    """Non-finite live vert / search coord / tolerance → full-scan equivalence."""

    # (1) Non-finite live vertex coordinate at index build time.
    for nonfinite in (math.nan, math.inf, -math.inf):
        bm = bmesh.new()
        try:
            _ensure_layers(bm)
            a0 = bm.verts.new((1.0, 0.0, 0.0))
            a1 = bm.verts.new((1.0, 1.0, 0.0))
            b0 = bm.verts.new((-1.0, 0.0, 0.0))
            b1 = bm.verts.new((-1.0, 1.0, 0.0))
            # Disjoint non-finite vertex: forces index-build fallback (§I-3a R3-M3).
            bm.verts.new((nonfinite, 5.0, 0.0))
            e0 = bm.edges.new((a0, a1))
            e1 = bm.edges.new((b0, b1))
            marker = bm.edges.layers.int.get(layer_names.EDGE_ORIGINAL_LAYER)
            e0[marker] = 1
            e1[marker] = 2
            _update_indices(bm)
            plan = [
                _make_cluster(
                    Vector((1.0, 0.5, 0.0)),
                    Vector((-1.0, 0.5, 0.0)),
                    positive=(_make_occurrence(e0, 0.5),),
                    negative=(_make_occurrence(e1, 0.5),),
                )
            ]
            # Full-scan path also evaluates coordinates_match against non-finite;
            # both must return the same result/exception/mesh state.
            result = _assert_differential(bm, plan)
            assert result[0][0] in {"return", "exception"}, result[0]
        finally:
            bm.free()

    # (2) Non-finite application search coordinate.
    bm = bmesh.new()
    try:
        _ensure_layers(bm)
        a0 = bm.verts.new((1.0, 0.0, 0.0))
        a1 = bm.verts.new((1.0, 1.0, 0.0))
        b0 = bm.verts.new((-1.0, 0.0, 0.0))
        b1 = bm.verts.new((-1.0, 1.0, 0.0))
        e0 = bm.edges.new((a0, a1))
        e1 = bm.edges.new((b0, b1))
        marker = bm.edges.layers.int.get(layer_names.EDGE_ORIGINAL_LAYER)
        e0[marker] = 1
        e1[marker] = 2
        _update_indices(bm)
        plan = [
            _make_cluster(
                Vector((math.nan, 0.5, 0.0)),
                Vector((-1.0, 0.5, 0.0)),
                positive=(_make_occurrence(e0, 0.5),),
                negative=(_make_occurrence(e1, 0.5),),
            )
        ]
        result = _assert_differential(bm, plan)
        assert result[0][0] in {"return", "exception"}, result[0]
    finally:
        bm.free()

    # (3) Non-finite tolerance on the cluster.
    bm = bmesh.new()
    try:
        _ensure_layers(bm)
        a0 = bm.verts.new((1.0, 0.0, 0.0))
        a1 = bm.verts.new((1.0, 1.0, 0.0))
        b0 = bm.verts.new((-1.0, 0.0, 0.0))
        b1 = bm.verts.new((-1.0, 1.0, 0.0))
        e0 = bm.edges.new((a0, a1))
        e1 = bm.edges.new((b0, b1))
        marker = bm.edges.layers.int.get(layer_names.EDGE_ORIGINAL_LAYER)
        e0[marker] = 1
        e1[marker] = 2
        _update_indices(bm)
        plan = [
            _make_cluster(
                Vector((1.0, 0.5, 0.0)),
                Vector((-1.0, 0.5, 0.0)),
                positive=(_make_occurrence(e0, 0.5),),
                negative=(_make_occurrence(e1, 0.5),),
                tolerance=math.nan,
            )
        ]
        result = _assert_differential(bm, plan)
        assert result[0][0] in {"return", "exception"}, result[0]
    finally:
        bm.free()


def check_bin_boundary_neighborhood():
    """Query near a primary-bin boundary still finds the neighbor (27-bin)."""

    bm = bmesh.new()
    try:
        _ensure_layers(bm)
        # Vector is float32: use the float32-stable bin-edge pair from test_core
        # (boundary=1.0, delta=1e-6 << tolerance, still distinct primary bins).
        boundary = 1.0
        delta = 1.0e-6
        query = Vector((boundary + delta, 0.5, 0.0))
        existing = Vector((boundary - delta, 0.5, 0.0))
        assert matching._quantized_coordinate(query, TOLERANCE) != matching._quantized_coordinate(
            existing, TOLERANCE
        ), (
            matching._quantized_coordinate(query, TOLERANCE),
            matching._quantized_coordinate(existing, TOLERANCE),
        )
        assert matching._quantized_coordinate(existing, TOLERANCE) in set(
            matching._iter_quantized_neighborhood(query, TOLERANCE)
        )
        a0 = bm.verts.new((1.0, 0.0, 0.0))
        a1 = bm.verts.new((1.0, 1.0, 0.0))
        b0 = bm.verts.new((-1.0, 0.0, 0.0))
        b1 = bm.verts.new((-1.0, 1.0, 0.0))
        bm.verts.new(tuple(existing))
        e0 = bm.edges.new((a0, a1))
        e1 = bm.edges.new((b0, b1))
        marker = bm.edges.layers.int.get(layer_names.EDGE_ORIGINAL_LAYER)
        e0[marker] = 1
        e1[marker] = 2
        _update_indices(bm)
        plan = [
            _make_cluster(
                query,
                Vector((-1.0, 0.5, 0.0)),
                positive=(_make_occurrence(e0, 0.5),),
                negative=(_make_occurrence(e1, 0.5),),
            )
        ]
        result = _assert_differential(bm, plan)
        # With one reusable at the positive query, both must either reuse it and
        # finish or hit the same post-merge ambiguous reason.
        assert result[0][0] == "return", result[0]
    finally:
        bm.free()


# ---------------------------------------------------------------------------
# Major coverage: index lifecycle observation + mid-process non-finite fallback
# ---------------------------------------------------------------------------

_AMBIGUOUS_REMAINING = "a separate existing vertex remains within tolerance of a mirrored cut intersection"


def _assert_noop_index_updates_break_differential(build_mesh_and_plan, helper_names):
    """Self-check silent update no-op(s) change candidate vs full-scan oracle.

    The no-op returns True so the candidate stays on the indexed path.  A
    False/None return would intentionally select the full-scan fallback and
    make this detection-power check meaningless.
    """

    if isinstance(helper_names, str):
        helper_names = (helper_names,)
    originals = {name: getattr(stitch_crossings, name) for name in helper_names}

    def _noop(*_args, **_kwargs):
        return True

    for name in helper_names:
        setattr(stitch_crossings, name, _noop)
    try:
        bm, plan = build_mesh_and_plan()
        try:
            broken = False
            try:
                _assert_differential(bm, plan)
            except AssertionError:
                broken = True
            assert broken, (
                f"index update no-ops {tuple(helper_names)!r} left candidate==oracle; "
                "fixture does not observe the requested lifecycle"
            )
        finally:
            bm.free()
    finally:
        for name, original in originals.items():
            setattr(stitch_crossings, name, original)


def _build_prior_split_observed_by_later_ambiguous():
    """(a) Prior application edge_split survivor is later ambiguous candidate.

    App0 splits at V=(1, 0.5, 0), then App1 splits at x=5 and unifies at
    C=V+(0.5*TOLERANCE, 0, 0).  Thus C is within tolerance of the live V
    generated by App0.  Full scan / correct index reports a remaining vertex;
    missing registration/rebin of the App0 split survivor makes the indexed
    path miss it and wrongly succeed.
    """

    bm = bmesh.new()
    _ensure_layers(bm)
    a0 = bm.verts.new((1.0, 0.0, 0.0))
    a1 = bm.verts.new((1.0, 1.0, 0.0))
    b0 = bm.verts.new((5.0, 0.0, 0.0))
    b1 = bm.verts.new((5.0, 1.0, 0.0))
    e0 = bm.edges.new((a0, a1))
    e1 = bm.edges.new((b0, b1))
    marker = bm.edges.layers.int.get(layer_names.EDGE_ORIGINAL_LAYER)
    e0[marker] = 1
    e1[marker] = 2
    _update_indices(bm)
    split_factor = 0.5
    generated = a0.co + split_factor * (a1.co - a0.co)
    later_merge = generated + Vector((0.5 * TOLERANCE, 0.0, 0.0))
    assert matching.coordinates_match(generated, later_merge, TOLERANCE)
    plan = [
        _make_cluster(
            generated,
            Vector((-1.0, 0.5, 0.0)),
            positive=(_make_occurrence(e0, split_factor),),
            negative=(),
        ),
        _make_cluster(
            later_merge,
            Vector((-5.0, 0.5, 0.0)),
            positive=(_make_occurrence(e1, 0.5),),
            negative=(),
        ),
    ]
    return bm, plan


def check_prior_split_vertex_observed_by_later_ambiguous():
    """(a) Edge_split vertex from a prior application feeds later ambiguous."""

    bm, plan = _build_prior_split_observed_by_later_ambiguous()
    try:
        result = _assert_differential(bm, plan)
        assert result[0] == ("return", (0, _AMBIGUOUS_REMAINING)), result[0]
    finally:
        bm.free()
    _assert_noop_index_updates_break_differential(
        _build_prior_split_observed_by_later_ambiguous,
        ("_register_crossings_vertex", "_rebin_crossings_vertex"),
    )


def _build_rebinned_survivor_observed_by_later_ambiguous():
    """(b) Pointmerge-moved survivor is later ambiguous candidate (rebin).

    App0 geometric split is near the origin; its survivor moves to F=(10, 0.5,
    0), crossing many primary bins.  App1 unifies at F+(0.5*TOLERANCE, 0, 0),
    which is within tolerance of that moved survivor.  Without rebin the App0
    survivor stays in its old bin and the indexed ambiguous scan misses it.
    """

    bm = bmesh.new()
    _ensure_layers(bm)
    a0 = bm.verts.new((0.0, 0.0, 0.0))
    a1 = bm.verts.new((0.0, 1.0, 0.0))
    c0 = bm.verts.new((2.0, 0.0, 0.0))
    c1 = bm.verts.new((2.0, 1.0, 0.0))
    b0 = bm.verts.new((20.0, 0.0, 0.0))
    b1 = bm.verts.new((20.0, 1.0, 0.0))
    e0 = bm.edges.new((a0, a1))
    e_mid = bm.edges.new((c0, c1))
    e1 = bm.edges.new((b0, b1))
    marker = bm.edges.layers.int.get(layer_names.EDGE_ORIGINAL_LAYER)
    e0[marker] = 1
    e_mid[marker] = 2
    e1[marker] = 3
    _update_indices(bm)
    split_factor = 0.5
    geometric_by_key = {
        _edge_key_for_occurrence(edge): edge.verts[0].co + split_factor * (edge.verts[1].co - edge.verts[0].co)
        for edge in (e0, e_mid)
    }
    survivor_key = min(_edge_key_for_occurrence(edge) for edge in (e0, e_mid))
    geometric = geometric_by_key[survivor_key]
    assert survivor_key == min(_edge_key_for_occurrence(e0), _edge_key_for_occurrence(e_mid))
    far = geometric + Vector((10.0, 0.0, 0.0))
    later_merge = far + Vector((0.5 * TOLERANCE, 0.0, 0.0))
    assert matching.coordinates_match(far, later_merge, TOLERANCE)
    assert matching._quantized_coordinate(geometric, TOLERANCE) != matching._quantized_coordinate(far, TOLERANCE), (
        matching._quantized_coordinate(geometric, TOLERANCE),
        matching._quantized_coordinate(far, TOLERANCE),
    )
    plan = [
        _make_cluster(
            far,
            Vector((-10.0, 0.5, 0.0)),
            positive=(
                _make_occurrence(e0, split_factor),
                _make_occurrence(e_mid, split_factor),
            ),
            negative=(),
        ),
        _make_cluster(
            later_merge,
            Vector((-20.0, 0.5, 0.0)),
            positive=(_make_occurrence(e1, split_factor),),
            negative=(),
        ),
    ]
    return bm, plan


def check_rebinned_survivor_observed_by_later_ambiguous():
    """(b) Survivor rebin after pointmerge feeds later ambiguous search."""

    bm, plan = _build_rebinned_survivor_observed_by_later_ambiguous()
    try:
        result = _assert_differential(bm, plan)
        assert result[0] == ("return", (0, _AMBIGUOUS_REMAINING)), result[0]
    finally:
        bm.free()
    _assert_noop_index_updates_break_differential(
        _build_rebinned_survivor_observed_by_later_ambiguous,
        "_rebin_crossings_vertex",
    )


def _build_two_cluster_finite_plan():
    """Two-cluster plan that starts on the indexed path (all finite)."""

    bm = bmesh.new()
    _ensure_layers(bm)
    a0 = bm.verts.new((1.0, 0.0, 0.0))
    a1 = bm.verts.new((1.0, 1.0, 0.0))
    b0 = bm.verts.new((-1.0, 0.0, 0.0))
    b1 = bm.verts.new((-1.0, 1.0, 0.0))
    c0 = bm.verts.new((2.0, 0.0, 0.0))
    c1 = bm.verts.new((2.0, 1.0, 0.0))
    d0 = bm.verts.new((-2.0, 0.0, 0.0))
    d1 = bm.verts.new((-2.0, 1.0, 0.0))
    edges = (
        bm.edges.new((a0, a1)),
        bm.edges.new((b0, b1)),
        bm.edges.new((c0, c1)),
        bm.edges.new((d0, d1)),
    )
    marker = bm.edges.layers.int.get(layer_names.EDGE_ORIGINAL_LAYER)
    for index, edge in enumerate(edges, start=1):
        edge[marker] = index
    _update_indices(bm)
    plan = [
        _make_cluster(
            Vector((1.0, 0.37, 0.0)),
            Vector((-1.0, 0.37, 0.0)),
            positive=(_make_occurrence(edges[0], 0.37),),
            negative=(_make_occurrence(edges[1], 0.37),),
        ),
        _make_cluster(
            Vector((2.0, 0.61, 0.0)),
            Vector((-2.0, 0.61, 0.0)),
            positive=(_make_occurrence(edges[2], 0.61),),
            negative=(_make_occurrence(edges[3], 0.61),),
        ),
    ]
    return bm, plan


def check_mid_process_nonfinite_fallback_matches_oracle():
    """(c) Non-finite mid-run (register or rebin) falls back; bit-matches oracle.

    Index build succeeds (all finite at start). The first register or the first
    rebin is forced to report non-finite (return False), flipping use_fallback
    for the rest of the apply. Mesh coordinates stay finite so the frozen
    full-scan oracle remains a fair bit-identity reference.
    """

    for helper_name in ("_register_crossings_vertex", "_rebin_crossings_vertex"):
        original = getattr(stitch_crossings, helper_name)
        calls = {"n": 0, "forced": 0}

        def flaky(*args, _original=original, _calls=calls, **kwargs):
            _calls["n"] += 1
            if _calls["n"] == 1:
                _calls["forced"] += 1
                return False
            return _original(*args, **kwargs)

        setattr(stitch_crossings, helper_name, flaky)
        try:
            bm, plan = _build_two_cluster_finite_plan()
            try:
                result = _assert_differential(bm, plan)
                assert result[0] == ("return", (4, "")), (helper_name, result[0])
                assert calls["forced"] == 1, (helper_name, calls)
                assert calls["n"] >= 1, (helper_name, calls)
            finally:
                bm.free()
        finally:
            setattr(stitch_crossings, helper_name, original)


def run():
    checks = (
        check_track_a_bit_identical,
        check_track_a_multi_k_bit_identical,
        check_dense_ambiguous_matches_oracle,
        check_survivor_rebin_across_applications,
        check_rebin_keeps_nearby_query_complete,
        check_nonfinite_fallback_matches_oracle,
        check_bin_boundary_neighborhood,
        check_prior_split_vertex_observed_by_later_ambiguous,
        check_rebinned_survivor_observed_by_later_ambiguous,
        check_mid_process_nonfinite_fallback_matches_oracle,
    )
    for check in checks:
        check()
        print(f"PASS {check.__name__}", flush=True)
    print(MARKER, flush=True)


if __name__ == "__main__":
    try:
        run()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
