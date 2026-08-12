# SPDX-License-Identifier: GPL-3.0-or-later

"""Headless differential checks for the replay edge-store scope.

Contract: .agents/doc/perf_epoch_finish_plan6_2026-08-12.md (v4), §I-2.
The oracle below freezes the pre-U6-2 eager-full edge-store construction.  It
does not call or share the candidate's scoped builder and always runs on a
separate BMesh clone.
"""

from __future__ import annotations

import math
import sys
import traceback
from contextlib import contextmanager
from pathlib import Path

import bmesh
from mathutils import Vector

PACKAGE_PARENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_PARENT))

from ydd_symmetric_edit import layer_names, matching, snapshot, stitch  # noqa: E402
from ydd_symmetric_edit._types import FaceId  # noqa: E402

AXIS = matching.AXIS_INDEX["X"]
TOLERANCE = 1.0e-5
MISSING_FACE_ID = FaceId(9001)

_ORIGINAL_COLLECT_CONTEXT = stitch._collect_reflected_path_context
_ORIGINAL_REALIZE_CHAIN = stitch._realize_interior_chain


def _frozen_eager_apply_reflected_path_topology(
    bm,
    source_edges,
    axis_index,
    tolerance,
    mirror_face_ids,
):
    """Pre-U6-2 apply implementation with the eager-full store frozen here."""

    source_edges = list(source_edges)
    if not source_edges:
        return 0, 0, "no source cut edges were supplied"

    marker_layer = bm.edges.layers.int.get(layer_names.EDGE_ORIGINAL_LAYER)
    face_layer = bm.faces.layers.int.get(layer_names.FACE_ID_LAYER)
    if marker_layer is None or face_layer is None:
        return 0, 0, "temporary topology markers are missing"

    (
        source_vertex_by_key,
        target_ids_by_vertex,
        edge_records,
        unmatched_face_ids,
        _status,
    ) = stitch._collect_reflected_path_context(
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
    target_faces_by_id = stitch._target_faces_by_id(bm, face_layer, needed_target_ids)
    classification, classify_reason = stitch._classify_reflected_vertices(
        source_vertex_by_key,
        target_ids_by_vertex,
        target_faces_by_id,
        axis_index,
        tolerance,
    )
    if classify_reason:
        return 0, 0, classify_reason

    adjacency = stitch._path_adjacency(edge_records)
    chains, chain_reason = stitch._find_interior_chains(
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
        return 0, 0, chain_reason

    chain_edge_keys = stitch._chain_source_edge_keys(chains)
    target_vertex_by_source_key = {}

    for source_key, source_vertex in source_vertex_by_key.items():
        if classification[source_key][0] == "interior":
            continue
        expected = matching.mirror_coordinate(source_vertex.co, axis_index)
        candidate_faces = {
            face
            for target_id in target_ids_by_vertex[source_key]
            for face in target_faces_by_id.get(target_id, ())
            if face.is_valid
        }
        kind, exact_vertex, target_edge, factor, reason = stitch._resolve_reflected_vertex_on_target(
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

    existing_edges = None
    created_edges = 0
    already_present = 0

    realized_face_ids = set()
    for chain in chains:
        created_delta, already_delta, fail_reason, existing_edges = stitch._realize_interior_chain(
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
        )
        if fail_reason:
            return created_edges, already_present, fail_reason
        created_edges += created_delta
        already_present += already_delta

    pending = [record for record in edge_records if frozenset((record[0], record[1])) not in chain_edge_keys]
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

            if existing_edges is None:
                existing_edges = {}
                for edge in bm.edges:
                    if edge.is_valid:
                        stitch._register_edge_endpoint_pair(
                            existing_edges,
                            edge.verts[0].co,
                            edge.verts[1].co,
                            tolerance,
                            face_ids={FaceId(int(face[face_layer])) for face in edge.link_faces},
                        )

            endpoint_match = stitch._match_edge_endpoint_pair_for_faces(
                target_a.co,
                target_b.co,
                tolerance,
                existing_edges,
                possible_target_ids,
            )
            if endpoint_match == "ambiguous":
                return (
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
            assert existing_edges is not None
            stitch._register_edge_endpoint_pair(
                existing_edges,
                new_edge.verts[0].co,
                new_edge.verts[1].co,
                tolerance,
                face_ids={FaceId(int(face[face_layer])) for face in new_edge.link_faces},
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


@contextmanager
def _replace(module, name, replacement):
    original = getattr(module, name)
    setattr(module, name, replacement)
    try:
        yield
    finally:
        setattr(module, name, original)


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
    """Full deterministic state, including partial topology after exceptions."""

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


def _run_apply(function, bm, source_edges, mirror_face_ids):
    try:
        result = ("return", function(bm, source_edges, AXIS, TOLERANCE, mirror_face_ids))
    except Exception as exc:  # The frozen oracle intentionally observes build-time failures.
        result = ("exception", type(exc).__module__, type(exc).__qualname__, tuple(map(str, exc.args)))
    return result, _mesh_state(bm)


def _assert_differential(bm, source_edges, mirror_face_ids):
    _update_indices(bm)
    source_indices = tuple(edge.index for edge in source_edges)
    eager_bm = _clone_bmesh(bm)
    eager_sources = [eager_bm.edges[index] for index in source_indices]
    candidate = _run_apply(
        stitch.apply_reflected_path_topology,
        bm,
        source_edges,
        mirror_face_ids,
    )
    eager = _run_apply(
        _frozen_eager_apply_reflected_path_topology,
        eager_bm,
        eager_sources,
        mirror_face_ids,
    )
    assert candidate == eager, (candidate, eager)
    eager_bm.free()
    return candidate


def _build_two_symmetric_quads():
    bm = bmesh.new()
    left = [
        bm.verts.new(co)
        for co in (
            (-2.0, -1.0, 0.0),
            (-1.0, -1.0, 0.0),
            (-1.0, 1.0, 0.0),
            (-2.0, 1.0, 0.0),
        )
    ]
    right = [
        bm.verts.new(co)
        for co in (
            (1.0, -1.0, 0.0),
            (2.0, -1.0, 0.0),
            (2.0, 1.0, 0.0),
            (1.0, 1.0, 0.0),
        )
    ]
    bm.faces.new(left)
    bm.faces.new(right)
    _update_indices(bm)
    assert tuple(vertex.index for vertex in left + right) == tuple(range(8))
    topology = snapshot.prepare_topology(bm, AXIS, TOLERANCE)
    mirror_face_ids = dict(topology.mirror_face_ids)
    return bm, mirror_face_ids


def _find_vertex(bm, coordinate):
    matches = [vertex for vertex in bm.verts if tuple(vertex.co) == tuple(coordinate)]
    assert len(matches) == 1, (coordinate, len(matches))
    return matches[0]


def _add_source_diagonal(bm):
    a = _find_vertex(bm, (-1.0, -1.0, 0.0))
    b = _find_vertex(bm, (-2.0, 1.0, 0.0))
    host = next(face for face in bm.faces if a in face.verts and b in face.verts)
    bmesh.utils.face_split(host, a, b)
    edge = bm.edges.get((a, b))
    assert edge is not None
    marker_layer = bm.edges.layers.int.get(layer_names.EDGE_ORIGINAL_LAYER)
    edge[marker_layer] = 0
    _update_indices(bm)
    return edge


def _add_loose_edge(bm, a, b):
    left = bm.verts.new(a)
    right = bm.verts.new(b)
    edge = bm.edges.new((left, right))
    _update_indices(bm)
    return edge


def _target_id_for_source(edge, bm, mirror_face_ids):
    face_layer = bm.faces.layers.int.get(layer_names.FACE_ID_LAYER)
    source_id = FaceId(int(edge.link_faces[0][face_layer]))
    target_id = mirror_face_ids[source_id]
    assert target_id is not None
    return FaceId(target_id)


def check_i_out_of_scope_same_bin():
    """(i) A same-coordinate U-disjoint edge cannot affect the tri-state."""

    bm, mirror_face_ids = _build_two_symmetric_quads()
    try:
        source = _add_source_diagonal(bm)
        _add_loose_edge(bm, (1.0, -1.0, 0.0), (2.0, 1.0, 0.0))
        matches = []
        original_match = stitch._match_edge_endpoint_pair_for_faces

        def capture_match(*args, **kwargs):
            result = original_match(*args, **kwargs)
            matches.append(result)
            return result

        with _replace(stitch, "_match_edge_endpoint_pair_for_faces", capture_match):
            result = _assert_differential(bm, [source], mirror_face_ids)
        assert result[0] == ("return", (1, 0, "")), result[0]
        assert matches == ["no_match", "no_match"], matches
    finally:
        bm.free()


def _context_with_missing_record_id(source_edges, face_layer, mirror_face_ids, *, include_real_record):
    context = _ORIGINAL_COLLECT_CONTEXT(
        source_edges,
        face_layer,
        mirror_face_ids,
        require_all_mirrored=False,
    )
    source_vertex_by_key, target_ids_by_vertex, records, unmatched, status = context
    rewritten = [(records[0][0], records[0][1], {MISSING_FACE_ID})]
    if include_real_record:
        rewritten.extend(records[1:])
    return source_vertex_by_key, target_ids_by_vertex, rewritten, unmatched, status


def check_ii_zero_live_faces_for_u():
    """(ii) A frozen U ID with zero live faces is skipped and both paths decline identically."""

    bm, mirror_face_ids = _build_two_symmetric_quads()
    try:
        source = _add_source_diagonal(bm)

        def injected(source_edges, face_layer, face_map, *, require_all_mirrored):
            assert not require_all_mirrored
            return _context_with_missing_record_id(
                source_edges,
                face_layer,
                face_map,
                include_real_record=False,
            )

        with _replace(stitch, "_collect_reflected_path_context", injected):
            result = _assert_differential(bm, [source], mirror_face_ids)
        assert result[0] == (
            "return",
            (0, 0, "could not place 1 mirrored cut segment(s)"),
        ), result[0]
    finally:
        bm.free()


def check_iii_deferred_retry_uses_frozen_u():
    """(iii) Retry retains U from every initial record, including a later identity hit."""

    bm, mirror_face_ids = _build_two_symmetric_quads()
    try:
        source = _add_source_diagonal(bm)
        boundary = bm.edges.get(
            (
                _find_vertex(bm, (-2.0, -1.0, 0.0)),
                _find_vertex(bm, (-1.0, -1.0, 0.0)),
            )
        )
        assert boundary is not None
        target_id = _target_id_for_source(boundary, bm, mirror_face_ids)
        registrations = []
        retries = 0
        original_register = stitch._register_edge_endpoint_pair
        original_match = stitch._match_edge_endpoint_pair_for_faces

        def injected(source_edges, face_layer, face_map, *, require_all_mirrored):
            assert not require_all_mirrored
            return _context_with_missing_record_id(
                source_edges,
                face_layer,
                face_map,
                include_real_record=True,
            )

        def capture_register(*args, **kwargs):
            registrations.append(frozenset(kwargs.get("face_ids", ())))
            return original_register(*args, **kwargs)

        def capture_retry(a, b, tolerance, store, possible_target_ids):
            nonlocal retries
            if possible_target_ids == {MISSING_FACE_ID}:
                retries += 1
            return original_match(a, b, tolerance, store, possible_target_ids)

        eager_bm = _clone_bmesh(bm)
        source_indices = (source.index, boundary.index)
        eager_sources = [eager_bm.edges[index] for index in source_indices]
        with (
            _replace(stitch, "_collect_reflected_path_context", injected),
            _replace(stitch, "_register_edge_endpoint_pair", capture_register),
            _replace(stitch, "_match_edge_endpoint_pair_for_faces", capture_retry),
        ):
            candidate = _run_apply(
                stitch.apply_reflected_path_topology,
                bm,
                [source, boundary],
                mirror_face_ids,
            )
        with _replace(stitch, "_collect_reflected_path_context", injected):
            eager = _run_apply(
                _frozen_eager_apply_reflected_path_topology,
                eager_bm,
                eager_sources,
                mirror_face_ids,
            )
        assert candidate == eager, (candidate, eager)
        assert retries == 2, retries
        assert any(target_id in face_ids for face_ids in registrations), registrations
        eager_bm.free()
    finally:
        bm.free()


def check_iv_nonfinite_u_disjoint_edges():
    """(iv) NaN/+Inf/-Inf on U-disjoint loose edges expose the accepted difference."""

    for nonfinite in (math.nan, math.inf, -math.inf):
        bm, mirror_face_ids = _build_two_symmetric_quads()
        eager_bm = None
        try:
            source = _add_source_diagonal(bm)
            _add_loose_edge(bm, (nonfinite, 10.0, 0.0), (11.0, 10.0, 0.0))
            source_index = source.index
            eager_bm = _clone_bmesh(bm)
            candidate = _run_apply(
                stitch.apply_reflected_path_topology,
                bm,
                [source],
                mirror_face_ids,
            )
            eager = _run_apply(
                _frozen_eager_apply_reflected_path_topology,
                eager_bm,
                [eager_bm.edges[source_index]],
                mirror_face_ids,
            )
            assert candidate[0] == ("return", (1, 0, "")), (nonfinite, candidate[0])
            assert eager[0][0] == "exception", (nonfinite, eager[0])
            assert eager[0][2] in {"ValueError", "OverflowError"}, (nonfinite, eager[0])
            # U-connected or record-side non-finite values are outside §I-2's
            # equivalence domain; this test intentionally makes no claim for them.
        finally:
            if eager_bm is not None:
                eager_bm.free()
            bm.free()


def check_v_fixture_determinism():
    """(v) Literal coordinates and creation order fix every fixture index."""

    first, first_map = _build_two_symmetric_quads()
    second, second_map = _build_two_symmetric_quads()
    try:
        first_source = _add_source_diagonal(first)
        second_source = _add_source_diagonal(second)
        assert first_map == second_map
        assert first_source.index == second_source.index
        assert _mesh_state(first) == _mesh_state(second)
        assert tuple(vertex.index for vertex in first.verts) == tuple(range(8))
        assert tuple(face.index for face in first.faces) == tuple(range(3))
    finally:
        first.free()
        second.free()


def _horizontal_edge(bm, y, *, negative):
    return next(
        edge
        for edge in bm.edges
        if all((vertex.co.x < 0.0) == negative for vertex in edge.verts)
        and all(float(vertex.co.y) == y for vertex in edge.verts)
    )


def _split_at_x(edge, x):
    a = float(edge.verts[0].co.x)
    b = float(edge.verts[1].co.x)
    _new_edge, vertex = bmesh.utils.edge_split(edge, edge.verts[0], (x - a) / (b - a))
    vertex.co.x = x
    return vertex


def _build_two_chain_fixture():
    bm, mirror_face_ids = _build_two_symmetric_quads()
    bottom = _horizontal_edge(bm, -1.0, negative=True)
    vb1 = _split_at_x(bottom, -1.8)
    bottom_rest = next(
        edge
        for edge in vb1.link_edges
        if all(float(vertex.co.y) == -1.0 for vertex in edge.verts)
        and max(float(vertex.co.x) for vertex in edge.verts) > -1.5
    )
    vb2 = _split_at_x(bottom_rest, -1.2)
    corner_tl = _find_vertex(bm, (-2.0, 1.0, 0.0))
    corner_tr = _find_vertex(bm, (-1.0, 1.0, 0.0))

    host_a = next(face for face in bm.faces if vb1 in face.verts and corner_tl in face.verts)
    bmesh.utils.face_split(host_a, vb1, corner_tl, coords=[(-1.85, 0.0, 0.0)])
    host_b = next(face for face in bm.faces if face.is_valid and vb2 in face.verts and corner_tr in face.verts)
    bmesh.utils.face_split(host_b, vb2, corner_tr, coords=[(-1.15, 0.0, 0.0)])
    chain_edges, side, total, crossing = stitch.collect_source_path_edges(
        bm,
        AXIS,
        TOLERANCE,
        "AUTO",
    )
    assert side == "NEGATIVE" and total == 4 and crossing == 0
    assert len(chain_edges) == 4

    # This loose record is injected into the same original FaceId. Its
    # reflected endpoints are the non-adjacent target corners (1, 1) and
    # (2, -1), so identity lookup misses deterministically.
    pending = _add_loose_edge(bm, (-1.0, 1.0, 0.0), (-2.0, -1.0, 0.0))
    return bm, mirror_face_ids, [*chain_edges, pending]


def _c1_hooks():
    state = {
        "realized": 0,
        "pre_chain_match": None,
        "live_eager_match": None,
    }

    def collect(source_edges, face_layer, mirror_face_ids, *, require_all_mirrored):
        assert not require_all_mirrored
        chain_edges = [edge for edge in source_edges if edge.link_faces]
        loose_edges = [edge for edge in source_edges if not edge.link_faces]
        assert len(chain_edges) == 4 and len(loose_edges) == 1
        source_vertex_by_key, target_ids_by_vertex, records, unmatched, status = _ORIGINAL_COLLECT_CONTEXT(
            chain_edges,
            face_layer,
            mirror_face_ids,
            require_all_mirrored=False,
        )
        target_ids = set().union(*(record[2] for record in records))
        assert len(target_ids) == 1
        target_id = next(iter(target_ids))
        loose = loose_edges[0]
        loose_keys = tuple(hash(vertex) for vertex in loose.verts)
        for key, vertex in zip(loose_keys, loose.verts, strict=True):
            source_vertex_by_key[key] = vertex
            target_ids_by_vertex[key] = {target_id}
        records.append((loose_keys[0], loose_keys[1], {target_id}))
        state["pending_keys"] = loose_keys
        state["target_id"] = target_id
        return source_vertex_by_key, target_ids_by_vertex, records, unmatched, status

    def realize(
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
        **realize_kwargs,
    ):
        if "pre_chain_store" not in state:
            store = {}
            for edge in bm.edges:
                if edge.is_valid:
                    stitch._register_edge_endpoint_pair(
                        store,
                        edge.verts[0].co,
                        edge.verts[1].co,
                        tolerance,
                        face_ids={FaceId(int(face[face_layer])) for face in edge.link_faces},
                    )
            state["pre_chain_store"] = store

        result = _ORIGINAL_REALIZE_CHAIN(
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
            **realize_kwargs,
        )
        state["realized"] += 1
        source_end_coordinates = {
            tuple(source_vertex_by_key[chain.end_a].co),
            tuple(source_vertex_by_key[chain.end_b].co),
        }
        if (-1.0, 1.0, 0.0) not in source_end_coordinates:
            return result

        assert len(chain.members) == 1
        pending_keys = state["pending_keys"]
        pending_by_coordinate = {
            tuple(source_vertex_by_key[key].co): target_vertex_by_source_key[key] for key in pending_keys
        }
        query_a = pending_by_coordinate[(-1.0, 1.0, 0.0)]
        query_b = pending_by_coordinate[(-2.0, -1.0, 0.0)]
        member = target_vertex_by_source_key[chain.members[0]]
        member.co = query_b.co + Vector((0.25 * tolerance, 0.0, 0.0))
        descendant = bm.edges.get((query_a, member))
        assert descendant is not None
        assert all(FaceId(int(face[face_layer])) == state["target_id"] for face in descendant.link_faces)

        state["pre_chain_match"] = stitch._match_edge_endpoint_pair_for_faces(
            query_a.co,
            query_b.co,
            tolerance,
            state["pre_chain_store"],
            {state["target_id"]},
        )
        live_store = {}
        for edge in bm.edges:
            if edge.is_valid:
                stitch._register_edge_endpoint_pair(
                    live_store,
                    edge.verts[0].co,
                    edge.verts[1].co,
                    tolerance,
                    face_ids={FaceId(int(face[face_layer])) for face in edge.link_faces},
                )
        state["live_eager_match"] = stitch._match_edge_endpoint_pair_for_faces(
            query_a.co,
            query_b.co,
            tolerance,
            live_store,
            {state["target_id"]},
        )
        return result

    return collect, realize, state


def check_vi_live_descendant_face_closure():
    """(vi) C1: two chains, second-chain-only edge, then non-chain pending."""

    bm = None
    eager_bm = None
    try:
        bm, mirror_face_ids, source_edges = _build_two_chain_fixture()
        eager_bm, eager_mirror_face_ids, eager_source_edges = _build_two_chain_fixture()
        _update_indices(bm)
        _update_indices(eager_bm)
        assert mirror_face_ids == eager_mirror_face_ids
        assert _mesh_state(bm) == _mesh_state(eager_bm)
        assert tuple(
            tuple(vertex.index for vertex in edge.verts) for edge in bm.edges
        ) == tuple(tuple(vertex.index for vertex in edge.verts) for edge in eager_bm.edges)
        assert tuple(
            tuple(vertex.index for vertex in face.verts) for face in bm.faces
        ) == tuple(tuple(vertex.index for vertex in face.verts) for face in eager_bm.faces)

        def source_signature(edges):
            return tuple(
                (
                    edge.index,
                    tuple(vertex.index for vertex in edge.verts),
                    tuple(
                        tuple(_float_bits(component) for component in vertex.co)
                        for vertex in edge.verts
                    ),
                )
                for edge in edges
            )

        assert source_signature(source_edges) == source_signature(eager_source_edges)
        collect, realize, state = _c1_hooks()
        with (
            _replace(stitch, "_collect_reflected_path_context", collect),
            _replace(stitch, "_realize_interior_chain", realize),
        ):
            candidate = _run_apply(
                stitch.apply_reflected_path_topology,
                bm,
                source_edges,
                mirror_face_ids,
            )

        eager_collect, eager_realize, eager_state = _c1_hooks()
        with (
            _replace(stitch, "_collect_reflected_path_context", eager_collect),
            _replace(stitch, "_realize_interior_chain", eager_realize),
        ):
            eager = _run_apply(
                _frozen_eager_apply_reflected_path_topology,
                eager_bm,
                eager_source_edges,
                eager_mirror_face_ids,
            )
        assert candidate == eager, (candidate, eager)
        assert candidate[0] == ("return", (4, 1, "")), candidate[0]
        for observed in (state, eager_state):
            assert observed["realized"] == 2
            assert observed["pre_chain_match"] == "no_match"
            assert observed["live_eager_match"] == "match"
    finally:
        if eager_bm is not None:
            eager_bm.free()
        if bm is not None:
            bm.free()


def _build_two_rows_per_side():
    bm = bmesh.new()

    def side_vertices(x_inner, x_outer):
        return {(x, y): bm.verts.new((x, y, 0.0)) for y in (-1.0, 0.0, 1.0) for x in (x_outer, x_inner)}

    left = side_vertices(-1.0, -2.0)
    right = side_vertices(1.0, 2.0)
    for vertices, inner, outer in ((left, -1.0, -2.0), (right, 1.0, 2.0)):
        bm.faces.new(
            [
                vertices[(outer, -1.0)],
                vertices[(inner, -1.0)],
                vertices[(inner, 0.0)],
                vertices[(outer, 0.0)],
            ]
        )
        bm.faces.new(
            [
                vertices[(outer, 0.0)],
                vertices[(inner, 0.0)],
                vertices[(inner, 1.0)],
                vertices[(outer, 1.0)],
            ]
        )
    _update_indices(bm)
    assert (len(bm.verts), len(bm.edges), len(bm.faces)) == (12, 14, 4)
    topology = snapshot.prepare_topology(bm, AXIS, TOLERANCE)
    mirror_face_ids = dict(topology.mirror_face_ids)
    return bm, mirror_face_ids


def check_vii_shared_edge_registered_once():
    """(vii-a) One BMEdge shared by two U faces contributes exactly one entry."""

    bm, mirror_face_ids = _build_two_rows_per_side()
    try:
        a = _find_vertex(bm, (-1.0, -1.0, 0.0))
        b = _find_vertex(bm, (-2.0, 0.0, 0.0))
        host = next(face for face in bm.faces if a in face.verts and b in face.verts)
        bmesh.utils.face_split(host, a, b)
        source = bm.edges.get((a, b))
        assert source is not None
        source[bm.edges.layers.int.get(layer_names.EDGE_ORIGINAL_LAYER)] = 0
        upper_boundary = bm.edges.get(
            (
                _find_vertex(bm, (-2.0, 1.0, 0.0)),
                _find_vertex(bm, (-1.0, 1.0, 0.0)),
            )
        )
        assert upper_boundary is not None
        shared_key = frozenset(((1.0, 0.0, 0.0), (2.0, 0.0, 0.0)))
        shared_registrations = 0
        original_register = stitch._register_edge_endpoint_pair

        def capture_register(store, edge_a, edge_b, tolerance, marker=None, face_ids=None):
            nonlocal shared_registrations
            key = frozenset((tuple(edge_a), tuple(edge_b)))
            if key == shared_key:
                shared_registrations += 1
            return original_register(
                store,
                edge_a,
                edge_b,
                tolerance,
                marker=marker,
                face_ids=face_ids,
            )

        with _replace(stitch, "_register_edge_endpoint_pair", capture_register):
            result = _assert_differential(
                bm,
                [source, upper_boundary],
                mirror_face_ids,
            )
        assert result[0] == ("return", (1, 1, "")), result[0]
        # The patch wraps candidate and eager oracle; each must register the
        # shared physical edge once despite visiting both U faces.
        assert shared_registrations == 2, shared_registrations
    finally:
        bm.free()


def check_vii_coordinate_duplicate_edges_are_ambiguous():
    """(vii-b) Distinct BMEdges at identical coordinates remain two entries."""

    bm, mirror_face_ids = _build_two_symmetric_quads()
    try:
        source = _add_source_diagonal(bm)
        face_layer = bm.faces.layers.int.get(layer_names.FACE_ID_LAYER)
        for z in (1.0, 2.0):
            vertices = [
                bm.verts.new((1.0, -1.0, 0.0)),
                bm.verts.new((2.0, 1.0, 0.0)),
                bm.verts.new((3.0, 0.0, z)),
            ]
            face = bm.faces.new(vertices)
            face[face_layer] = int(MISSING_FACE_ID)
        _update_indices(bm)

        def injected(source_edges, injected_face_layer, face_map, *, require_all_mirrored):
            assert not require_all_mirrored
            source_vertex_by_key, target_ids_by_vertex, records, unmatched, status = _ORIGINAL_COLLECT_CONTEXT(
                source_edges,
                injected_face_layer,
                face_map,
                require_all_mirrored=False,
            )
            records = [(records[0][0], records[0][1], {MISSING_FACE_ID})]
            return source_vertex_by_key, target_ids_by_vertex, records, unmatched, status

        with _replace(stitch, "_collect_reflected_path_context", injected):
            result = _assert_differential(bm, [source], mirror_face_ids)
        assert result[0] == (
            "return",
            (
                0,
                0,
                "multiple coordinate-matching edges are ambiguous across target faces",
            ),
        ), result[0]
    finally:
        bm.free()


def _chebyshev(a, b):
    return max(abs(float(a[index]) - float(b[index])) for index in range(3))


def _frozen_endpoint_match(a, b, entries, tolerance):
    best = None
    for stored_a, stored_b, marker in entries:
        for query_a, query_b in ((a, b), (b, a)):
            distance_a = _chebyshev(query_a, stored_a)
            distance_b = _chebyshev(query_b, stored_b)
            if distance_a > tolerance or distance_b > tolerance:
                continue
            score = max(distance_a, distance_b)
            if best is None or score < best[0]:
                best = (score, marker)
    return best


def _frozen_build_reflected_cutter(bm, source_edges, axis_index, tolerance):
    """Independent eager list oracle for the untouched cutter boundary."""

    existing = [(tuple(edge.verts[0].co), tuple(edge.verts[1].co), None) for edge in bm.edges]
    _update_indices(bm)
    vertex_indices = {}
    vertices = []
    edges = []
    already_present = 0
    for edge in source_edges:
        reflected = (
            matching.mirror_coordinate(edge.verts[0].co, axis_index),
            matching.mirror_coordinate(edge.verts[1].co, axis_index),
        )
        if _frozen_endpoint_match(reflected[0], reflected[1], existing, tolerance) is not None:
            already_present += 1
            continue
        cutter_edge = []
        for source_vertex, coordinate in zip(edge.verts, reflected, strict=True):
            cutter_index = vertex_indices.get(source_vertex.index)
            if cutter_index is None:
                cutter_index = len(vertices)
                vertex_indices[source_vertex.index] = cutter_index
                vertices.append(coordinate)
            cutter_edge.append(cutter_index)
        if cutter_edge[0] != cutter_edge[1]:
            edges.append((cutter_edge[0], cutter_edge[1]))
    return vertices, edges, already_present


def _frozen_collapsed_offset_markers(bm, source_edges, axis_index, tolerance):
    """Independent eager list oracle for the untouched collapsed-offset boundary."""

    marker_layer = bm.edges.layers.int.get(layer_names.EDGE_ORIGINAL_LAYER)
    if marker_layer is None:
        return set(), "edge marker layer is missing"
    originals = []
    new_edges = []
    for edge in bm.edges:
        marker = int(edge[marker_layer])
        entry = (tuple(edge.verts[0].co), tuple(edge.verts[1].co), marker)
        if marker <= 0:
            new_edges.append(entry)
        else:
            originals.append(entry)

    target_markers = set()
    matched_nonzero_segments = 0
    for edge in source_edges:
        reflected_a = matching.mirror_coordinate(edge.verts[0].co, axis_index)
        reflected_b = matching.mirror_coordinate(edge.verts[1].co, axis_index)
        if (reflected_a - reflected_b).length <= tolerance:
            continue
        if _frozen_endpoint_match(reflected_a, reflected_b, new_edges, tolerance) is not None:
            return set(), "the target already contains native zero-offset topology"
        match = _frozen_endpoint_match(reflected_a, reflected_b, originals, tolerance)
        marker = None if match is None else match[1]
        if marker is None:
            return set(), "a reflected zero-offset segment has no original target edge"
        target_markers.add(marker)
        matched_nonzero_segments += 1
    if not target_markers or not matched_nonzero_segments:
        return set(), "no reflected original target loop was found"
    return target_markers, ""


def _normalize_cutter(result):
    vertices, edges, already = result
    return (
        tuple(tuple(_float_bits(component) for component in vertex) for vertex in vertices),
        tuple(edges),
        already,
    )


def check_viii_unscoped_helper_boundaries():
    """(viii) Cutter and collapsed-offset call sites retain their eager behavior."""

    cutter = bmesh.new()
    collapsed = bmesh.new()
    cutter_oracle = None
    collapsed_oracle = None
    try:
        matched_source = _add_loose_edge(cutter, (-1.0, 0.0, 0.0), (-1.0, 1.0, 0.0))
        unmatched_source = _add_loose_edge(cutter, (-2.0, 2.0, 0.0), (-2.0, 3.0, 0.0))
        _add_loose_edge(cutter, (1.0, 0.0, 0.0), (1.0, 1.0, 0.0))
        cutter_oracle = _clone_bmesh(cutter)
        cutter_source_indices = (matched_source.index, unmatched_source.index)
        candidate_cutter = stitch.build_reflected_cutter(
            cutter,
            [matched_source, unmatched_source],
            AXIS,
            TOLERANCE,
        )
        eager_cutter = _frozen_build_reflected_cutter(
            cutter_oracle,
            [cutter_oracle.edges[index] for index in cutter_source_indices],
            AXIS,
            TOLERANCE,
        )
        assert _normalize_cutter(candidate_cutter) == _normalize_cutter(eager_cutter)

        marker_layer = collapsed.edges.layers.int.new(layer_names.EDGE_ORIGINAL_LAYER)
        source = _add_loose_edge(collapsed, (-1.0, 0.0, 0.0), (-1.0, 1.0, 0.0))
        source[marker_layer] = 0
        for shift, marker in ((1.5 * TOLERANCE, 20), (0.25 * TOLERANCE, 10), (0.8 * TOLERANCE, 30)):
            target = _add_loose_edge(
                collapsed,
                (1.0 + shift, 0.0, 0.0),
                (1.0 + shift, 1.0, 0.0),
            )
            target[marker_layer] = marker
        collapsed_oracle = _clone_bmesh(collapsed)
        candidate_markers = stitch.collapsed_offset_target_edge_markers(
            collapsed,
            [source],
            AXIS,
            TOLERANCE,
        )
        eager_markers = _frozen_collapsed_offset_markers(
            collapsed_oracle,
            [collapsed_oracle.edges[source.index]],
            AXIS,
            TOLERANCE,
        )
        assert candidate_markers == eager_markers == ({10}, "")
    finally:
        if cutter_oracle is not None:
            cutter_oracle.free()
        if collapsed_oracle is not None:
            collapsed_oracle.free()
        cutter.free()
        collapsed.free()


def run():
    checks = (
        check_i_out_of_scope_same_bin,
        check_ii_zero_live_faces_for_u,
        check_iii_deferred_retry_uses_frozen_u,
        check_iv_nonfinite_u_disjoint_edges,
        check_v_fixture_determinism,
        check_vi_live_descendant_face_closure,
        check_vii_shared_edge_registered_once,
        check_vii_coordinate_duplicate_edges_are_ambiguous,
        check_viii_unscoped_helper_boundaries,
    )
    for check in checks:
        check()
        print(f"PASS {check.__name__}", flush=True)
    print("YSE_SCOPED_STORE_TEST_OK", flush=True)


if __name__ == "__main__":
    try:
        run()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
