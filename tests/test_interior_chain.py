# SPDX-License-Identifier: GPL-3.0-or-later

"""Headless checks for face-interior paths and R-N1 networks.

Contract: .agents/doc/fix_contract_knife_interior_host_2026-08-13.md (v7.1)
Oracle items O2/O3/O4/O10 plus the retained n=1/n=2 and structural declines.
"""

from __future__ import annotations

import inspect
import sys
import traceback
from pathlib import Path

import bmesh
import mathutils.geometry
from mathutils import Vector

PACKAGE_PARENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(PACKAGE_PARENT))
from fixtures_interior_host import (  # noqa: E402
    build_endpoint_collision,
    build_endpoint_link_mismatch,
    build_interior_network_branch_relay,
    build_interior_network_decline_dead_end,
    build_interior_network_decline_loop,
    build_interior_network_decline_one_anchor,
    build_interior_network_theta,
    build_interior_network_x,
    build_interior_network_y,
    build_lineage_mismatch_chain,
    build_projection_outside_band,
    build_rev3_both_side_stroke,
    build_sanity_excess_band,
    build_weak_surface_band,
    fixture_from_builder,
)

from ydd_symmetric_edit import layer_names, matching, snapshot, stitch_pathedges, stitch_reflect  # noqa: E402

TOLERANCE = 1.0e-5
AXIS = matching.AXIS_INDEX["X"]


def _call_gate(bm, source_edges, topology, axis=AXIS, tolerance=TOLERANCE):
    """Call both the revision-3 and carrier-aware gate signatures."""

    function = stitch_reflect.reflected_path_uses_only_target_boundaries
    arguments = (bm, source_edges, axis, tolerance, topology.mirror_face_ids)
    if "carrier_frames" in inspect.signature(function).parameters:
        return bool(function(*arguments, carrier_frames=topology.carrier_frames))
    return bool(function(*arguments))


def _call_apply(bm, source_edges, topology, axis=AXIS, tolerance=TOLERANCE):
    """Call both the revision-3 and carrier-aware apply signatures."""

    function = stitch_reflect.apply_reflected_path_topology
    arguments = (bm, source_edges, axis, tolerance, topology.mirror_face_ids)
    if "carrier_frames" in inspect.signature(function).parameters:
        return function(*arguments, carrier_frames=topology.carrier_frames)
    return function(*arguments)


def _path_interior_vertices(source_edges):
    adjacency = {}
    vertices = {}
    for edge in source_edges:
        left, right = edge.verts
        left_key, right_key = hash(left), hash(right)
        vertices[left_key] = left
        vertices[right_key] = right
        adjacency.setdefault(left_key, set()).add(right_key)
        adjacency.setdefault(right_key, set()).add(left_key)
    return [vertices[key] for key, neighbours in adjacency.items() if len(neighbours) == 2]


def _target_face_for_vertex(fixture, source_vertex):
    face_layer = fixture.bm.faces.layers.int.get(layer_names.FACE_ID_LAYER)
    assert face_layer is not None
    source_face = next(face for face in source_vertex.link_faces if face.is_valid)
    source_id = int(source_face[face_layer])
    target_id = fixture.topology.mirror_face_ids[source_id]
    return next(face for face in fixture.bm.faces if int(face[face_layer]) == target_id and face.is_valid)


def _target_instances(fixture, source_vertex):
    face_layer = fixture.bm.faces.layers.int.get(layer_names.FACE_ID_LAYER)
    assert face_layer is not None
    source_face = next(face for face in source_vertex.link_faces if face.is_valid)
    target_id = fixture.topology.mirror_face_ids[int(source_face[face_layer])]
    return [face for face in fixture.bm.faces if face.is_valid and int(face[face_layer]) == target_id]


def _surface_distance(point, face):
    return min(
        float((mathutils.geometry.closest_point_on_tri(point, a, b, c) - point).length)
        for a, b, c in stitch_reflect._face_surface_triangles(face)
    )


def _s_eff(fixture, source_vertex, target_face):
    face_layer = fixture.bm.faces.layers.int.get(layer_names.FACE_ID_LAYER)
    assert face_layer is not None
    source_face = next(face for face in source_vertex.link_faces if face.is_valid)
    source_frame = fixture.topology.carrier_frames[int(source_face[face_layer])]
    target_frame = fixture.topology.carrier_frames[int(target_face[face_layer])]
    assert source_frame.normal is not None and target_frame.normal is not None
    return max(20.0 * fixture.tolerance, 2.5 * max(source_frame.deviation, target_frame.deviation))


def _project_to_carrier(point, frame):
    assert frame.normal is not None
    normal = Vector(frame.normal.as_tuple())
    origin = Vector(frame.origin.as_tuple())
    return point - normal * ((point - origin).dot(normal) / normal.length_squared)


def build_two_symmetric_quads():
    """Left face x∈[-2,-1], right face x∈[1,2], both y∈[-1,1], z=0."""

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
    bm.normal_update()
    return bm


def _bottom_edge(bm, *, negative: bool):
    sign = -1.0 if negative else 1.0
    return next(
        edge
        for edge in bm.edges
        if all(vertex.co.x * sign > 0.0 for vertex in edge.verts)
        and all(abs(vertex.co.y + 1.0) < 1.0e-8 for vertex in edge.verts)
    )


def _top_edge(bm, *, negative: bool):
    sign = -1.0 if negative else 1.0
    return next(
        edge
        for edge in bm.edges
        if all(vertex.co.x * sign > 0.0 for vertex in edge.verts)
        and all(abs(vertex.co.y - 1.0) < 1.0e-8 for vertex in edge.verts)
    )


def _split_at_x(edge, x: float):
    a = edge.verts[0].co.x
    b = edge.verts[1].co.x
    factor = (x - a) / (b - a)
    _new_edge, vertex = bmesh.utils.edge_split(edge, edge.verts[0], factor)
    return vertex


def has_exact_edge(bm, a, b, tolerance=1.0e-7):
    def close(co, expected):
        return all(abs(co[index] - expected[index]) <= tolerance for index in range(3))

    return any(
        (close(edge.verts[0].co, a) and close(edge.verts[1].co, b))
        or (close(edge.verts[0].co, b) and close(edge.verts[1].co, a))
        for edge in bm.edges
        if edge.is_valid
    )


def find_vertex(bm, expected, tolerance=1.0e-7):
    for vertex in bm.verts:
        if all(abs(vertex.co[i] - expected[i]) <= tolerance for i in range(3)):
            return vertex
    return None


def collect_source_path_edges(bm):
    source, side, total, crossing = stitch_pathedges.collect_source_path_edges(bm, AXIS, TOLERANCE, "AUTO")
    assert side == "NEGATIVE", (side, total, crossing)
    assert crossing == 0
    return source


def check_interior_chain_n1():
    """(a) Interior chain length n=1: gate accepts; apply mirrors exactly."""

    bm = build_two_symmetric_quads()
    try:
        topology = snapshot.prepare_topology(bm, AXIS, TOLERANCE)
        bottom = _split_at_x(_bottom_edge(bm, negative=True), -1.5)
        top = _split_at_x(_top_edge(bm, negative=True), -1.5)
        host = next(face for face in bm.faces if bottom in face.verts and top in face.verts)
        # Path: bottom -- interior -- top  (degree-2 interior chain n=1)
        bmesh.utils.face_split(host, bottom, top, coords=[(-1.2, 0.0, 0.0)])
        source = collect_source_path_edges(bm)
        assert len(source) == 2, len(source)

        assert _call_gate(bm, source, topology)

        created, already, reason = _call_apply(bm, source, topology)
        assert reason == "", reason
        assert created == 2, (created, already, reason)
        assert already == 0, already

        assert has_exact_edge(bm, (1.5, -1.0, 0.0), (1.2, 0.0, 0.0))
        assert has_exact_edge(bm, (1.2, 0.0, 0.0), (1.5, 1.0, 0.0))
        mirrored = find_vertex(bm, (1.2, 0.0, 0.0))
        assert mirrored is not None
        assert mirrored.select is False
        # Exact reflected coordinate (bitwise mirror of the float32 source
        # vertex, not a face_split approximation).
        source_vertex = find_vertex(bm, (-1.2, 0.0, 0.0))
        assert source_vertex is not None
        assert tuple(mirrored.co) == (
            -float(source_vertex.co.x),
            float(source_vertex.co.y),
            float(source_vertex.co.z),
        )
    finally:
        bm.free()


def check_interior_chain_n2():
    """(b) Interior chain length n=2: two consecutive degree-2 interiors."""

    bm = build_two_symmetric_quads()
    try:
        topology = snapshot.prepare_topology(bm, AXIS, TOLERANCE)
        bottom = _split_at_x(_bottom_edge(bm, negative=True), -1.5)
        top = _split_at_x(_top_edge(bm, negative=True), -1.5)
        host = next(face for face in bm.faces if bottom in face.verts and top in face.verts)
        # Path: bottom -- v1 -- v2 -- top
        bmesh.utils.face_split(
            host,
            bottom,
            top,
            coords=[(-1.4, -0.25, 0.0), (-1.2, 0.25, 0.0)],
        )
        source = collect_source_path_edges(bm)
        assert len(source) == 3, len(source)

        assert _call_gate(bm, source, topology)

        created, already, reason = _call_apply(bm, source, topology)
        assert reason == "", reason
        assert created == 3, (created, already, reason)
        assert already == 0, already

        assert has_exact_edge(bm, (1.5, -1.0, 0.0), (1.4, -0.25, 0.0))
        assert has_exact_edge(bm, (1.4, -0.25, 0.0), (1.2, 0.25, 0.0))
        assert has_exact_edge(bm, (1.2, 0.25, 0.0), (1.5, 1.0, 0.0))
        for expected in ((1.4, -0.25, 0.0), (1.2, 0.25, 0.0)):
            vertex = find_vertex(bm, expected)
            assert vertex is not None, expected
            assert vertex.select is False
            source_vertex = find_vertex(bm, (-expected[0], expected[1], expected[2]))
            assert source_vertex is not None, expected
            assert tuple(vertex.co) == (
                -float(source_vertex.co.x),
                float(source_vertex.co.y),
                float(source_vertex.co.z),
            ), expected
    finally:
        bm.free()


def check_decline_degree3_interior():
    """O15(a): degree-three interior hub is now a direct positive case."""

    fixture = fixture_from_builder("network_y", build_interior_network_y)
    try:
        assert _call_gate(fixture.bm, fixture.source_edges, fixture.topology)
        created, already, reason = _call_apply(fixture.bm, fixture.source_edges, fixture.topology)
        assert reason == "", reason
        assert created + already == len(fixture.source_edges), (created, already)
        hub = find_vertex(fixture.bm, (1.3, 0.0, 0.0))
        assert hub is not None and len(hub.link_edges) == 3
    finally:
        fixture.free()


def check_decline_no_common_face():
    """(d) Degree-2 interior chain whose members+ends lack one common target face."""

    # Two Y-stacked symmetric pairs. Build bottom--v_lower--v_upper--top where
    # each interior reflects into a different target face, so the chain has no
    # single common target face (still degree-2 throughout).
    bm = bmesh.new()
    try:
        lower_left = [
            bm.verts.new(co)
            for co in (
                (-2.0, -2.0, 0.0),
                (-1.0, -2.0, 0.0),
                (-1.0, -0.1, 0.0),
                (-2.0, -0.1, 0.0),
            )
        ]
        lower_right = [
            bm.verts.new(co)
            for co in (
                (1.0, -2.0, 0.0),
                (2.0, -2.0, 0.0),
                (2.0, -0.1, 0.0),
                (1.0, -0.1, 0.0),
            )
        ]
        upper_left = [
            bm.verts.new(co)
            for co in (
                (-2.0, 0.1, 0.0),
                (-1.0, 0.1, 0.0),
                (-1.0, 2.0, 0.0),
                (-2.0, 2.0, 0.0),
            )
        ]
        upper_right = [
            bm.verts.new(co)
            for co in (
                (1.0, 0.1, 0.0),
                (2.0, 0.1, 0.0),
                (2.0, 2.0, 0.0),
                (1.0, 2.0, 0.0),
            )
        ]
        bm.faces.new(lower_left)
        bm.faces.new(lower_right)
        bm.faces.new(upper_left)
        bm.faces.new(upper_right)
        bm.normal_update()

        topology = snapshot.prepare_topology(bm, AXIS, TOLERANCE)
        marker = bm.edges.layers.int.get(layer_names.EDGE_ORIGINAL_LAYER)
        assert marker is not None

        bottom_edge = next(
            edge
            for edge in bm.edges
            if all(vertex.co.x < 0.0 for vertex in edge.verts)
            and all(abs(vertex.co.y + 2.0) < 1.0e-8 for vertex in edge.verts)
        )
        top_edge = next(
            edge
            for edge in bm.edges
            if all(vertex.co.x < 0.0 for vertex in edge.verts)
            and all(abs(vertex.co.y - 2.0) < 1.0e-8 for vertex in edge.verts)
        )
        _e, bottom = bmesh.utils.edge_split(bottom_edge, bottom_edge.verts[0], 0.5)
        bottom.co = (-1.5, -2.0, 0.0)
        _e, top = bmesh.utils.edge_split(top_edge, top_edge.verts[0], 0.5)
        top.co = (-1.5, 2.0, 0.0)

        lower_top_edge = next(
            edge
            for edge in bm.edges
            if all(vertex.co.x < 0.0 for vertex in edge.verts)
            and all(abs(vertex.co.y + 0.1) < 1.0e-8 for vertex in edge.verts)
        )
        upper_bottom_edge = next(
            edge
            for edge in bm.edges
            if all(vertex.co.x < 0.0 for vertex in edge.verts)
            and all(abs(vertex.co.y - 0.1) < 1.0e-8 for vertex in edge.verts)
        )
        _e, lower_anchor = bmesh.utils.edge_split(lower_top_edge, lower_top_edge.verts[0], 0.5)
        lower_anchor.co = (-1.5, -0.1, 0.0)
        _e, upper_anchor = bmesh.utils.edge_split(upper_bottom_edge, upper_bottom_edge.verts[0], 0.5)
        upper_anchor.co = (-1.5, 0.1, 0.0)

        lower_host = next(face for face in bm.faces if bottom in face.verts and lower_anchor in face.verts)
        upper_host = next(face for face in bm.faces if top in face.verts and upper_anchor in face.verts)
        bmesh.utils.face_split(lower_host, bottom, lower_anchor, coords=[(-1.4, -1.0, 0.0)])
        bmesh.utils.face_split(upper_host, upper_anchor, top, coords=[(-1.4, 1.0, 0.0)])
        v_lower = find_vertex(bm, (-1.4, -1.0, 0.0))
        v_upper = find_vertex(bm, (-1.4, 1.0, 0.0))
        assert v_lower is not None and v_upper is not None

        # Drop the face-local second legs and bridge the interiors so the path
        # graph is bottom--v_lower--v_upper--top (all interiors degree 2) while
        # the two interiors resolve onto different mirrored target faces.
        drop = []
        for edge in list(bm.edges):
            verts = set(edge.verts)
            if verts == {v_lower, lower_anchor} or verts == {v_upper, upper_anchor}:
                drop.append(edge)
        assert len(drop) == 2, len(drop)
        bmesh.ops.delete(bm, geom=drop, context="EDGES")
        bridge = bm.edges.new((v_lower, v_upper))
        bridge[marker] = 0

        source = collect_source_path_edges(bm)
        assert len(source) == 3, len(source)

        # Contract v6 R-W1: the wire bridge no longer forces a decline. The
        # two face segments end on target boundaries and the bridge mirrors
        # as a wire, so the whole path is now directly acceptable.
        assert _call_gate(bm, source, topology)
        created, already, reason = _call_apply(bm, source, topology)
        assert reason == "", reason
        assert created == 3 and already == 0, (created, already)
        bridge_mirror = next(
            (
                edge
                for edge in bm.edges
                if edge.is_valid and edge.is_wire and all(vertex.co.x > 0.0 for vertex in edge.verts)
            ),
            None,
        )
        assert bridge_mirror is not None
    finally:
        bm.free()


def check_curved_quad_single_candidate():
    """(e) Curved quad (suzanne10 face 75): the boundary end split turns the
    host into a pentagon whose ear-clip triangulation deviates from the
    evaluated quad surface by more than the tolerance-scale limit, so the
    single id-matching candidate must be accepted without re-testing
    containment."""

    from ydd_symmetric_edit import stitch_reflect

    quad = (
        (-0.21765238046646118, -0.5814824104309082, -0.17536157369613647),
        (-0.2244318425655365, -0.5596591234207153, -0.18323862552642822),
        (-0.203125, -0.5625, -0.1875),
        (-0.19602273404598236, -0.5852272510528564, -0.1796875),
    )
    bm = bmesh.new()
    try:
        left = [bm.verts.new(co) for co in quad]
        right = [bm.verts.new((-x, y, z)) for (x, y, z) in reversed(quad)]
        bm.faces.new(left)
        bm.faces.new(right)
        bm.normal_update()

        topology = snapshot.prepare_topology(bm, AXIS, TOLERANCE)
        host = next(face for face in bm.faces if all(v.co.x < 0.0 for v in face.verts))
        verts = list(host.verts)
        edge = next(e for e in host.edges if verts[0] in e.verts and verts[1] in e.verts)
        _e, vb = bmesh.utils.edge_split(edge, edge.verts[0], 0.43)
        interior = verts[0].co * 0.4 + verts[1].co * 0.35 + verts[2].co * 0.25
        corners = [v for v in verts if v not in edge.verts]
        vc = max(corners, key=lambda v: (v.co - vb.co).length)

        # Regression premise: the interior point lies on the evaluated quad
        # surface but farther than 2*tol from the post-split pentagon's
        # ear-clip triangulation (measured on the source side; the target
        # pentagon is its bitwise mirror).
        import mathutils.geometry as mg

        pentagon = next(f for f in bm.faces if vb in f.verts and vc in f.verts)
        assert len(pentagon.verts) == 5, len(pentagon.verts)
        min_dist = min(
            (mg.closest_point_on_tri(interior, a, b, c) - interior).length
            for a, b, c in stitch_reflect._face_surface_triangles(pentagon)
        )
        assert min_dist > TOLERANCE * 2.0, min_dist
        # The realization re-test would run on the target-side pentagon (the
        # mirrored loop, reversed winding), whose ear-clip need not match the
        # source one; pin the old failure condition on that side as well.
        mirrored_loop = [Vector((-v.co.x, v.co.y, v.co.z)) for v in reversed(list(pentagon.verts))]
        mirrored_interior = Vector((-interior.x, interior.y, interior.z))
        target_min_dist = min(
            (
                mg.closest_point_on_tri(
                    mirrored_interior,
                    mirrored_loop[a],
                    mirrored_loop[b],
                    mirrored_loop[c],
                )
                - mirrored_interior
            ).length
            for a, b, c in mg.tessellate_polygon([mirrored_loop])
        )
        assert target_min_dist > TOLERANCE * 2.0, target_min_dist

        bmesh.utils.face_split(pentagon, vb, vc, coords=[tuple(interior)])
        source = collect_source_path_edges(bm)
        assert len(source) == 2, len(source)

        assert _call_gate(bm, source, topology)
        created, already, reason = _call_apply(bm, source, topology)
        assert reason == "", reason
        assert created == 2, (created, already)

        source_interior = find_vertex(bm, tuple(interior))
        assert source_interior is not None
        mirrored = find_vertex(
            bm,
            (-float(source_interior.co.x), float(source_interior.co.y), float(source_interior.co.z)),
        )
        assert mirrored is not None
        assert mirrored.select is False
    finally:
        bm.free()


def check_two_chains_same_face():
    """(f) Two independent interior chains sharing one ancestor face.

    The first chain may take the single-candidate fast path; the second one
    must fall back to strict containment because the target id has already
    been face_split during this apply call (a lone shared candidate could be
    the wrong descendant)."""

    bm = build_two_symmetric_quads()
    try:
        topology = snapshot.prepare_topology(bm, AXIS, TOLERANCE)
        vb1 = _split_at_x(_bottom_edge(bm, negative=True), -1.8)
        bottom_rest = next(
            edge
            for edge in vb1.link_edges
            if all(abs(vertex.co.y + 1.0) < 1.0e-8 for vertex in edge.verts)
            and max(vertex.co.x for vertex in edge.verts) > -1.5
        )
        _e, vb2 = bmesh.utils.edge_split(
            bottom_rest, vb1, (-1.2 - vb1.co.x) / (bottom_rest.other_vert(vb1).co.x - vb1.co.x)
        )
        vb2.co = (-1.2, -1.0, 0.0)
        corner_tl = find_vertex(bm, (-2.0, 1.0, 0.0))
        corner_tr = find_vertex(bm, (-1.0, 1.0, 0.0))
        assert corner_tl is not None and corner_tr is not None

        host_a = next(face for face in bm.faces if vb1 in face.verts and corner_tl in face.verts)
        bmesh.utils.face_split(host_a, vb1, corner_tl, coords=[(-1.85, 0.0, 0.0)])
        host_b = next(face for face in bm.faces if face.is_valid and vb2 in face.verts and corner_tr in face.verts)
        bmesh.utils.face_split(host_b, vb2, corner_tr, coords=[(-1.15, 0.0, 0.0)])

        source = collect_source_path_edges(bm)
        assert len(source) == 4, len(source)

        assert _call_gate(bm, source, topology)
        created, already, reason = _call_apply(bm, source, topology)
        assert reason == "", reason
        assert created == 4, (created, already)

        assert has_exact_edge(bm, (1.8, -1.0, 0.0), (1.85, 0.0, 0.0))
        assert has_exact_edge(bm, (1.85, 0.0, 0.0), (2.0, 1.0, 0.0))
        assert has_exact_edge(bm, (1.2, -1.0, 0.0), (1.15, 0.0, 0.0))
        assert has_exact_edge(bm, (1.15, 0.0, 0.0), (1.0, 1.0, 0.0))
        for expected in ((1.85, 0.0, 0.0), (1.15, 0.0, 0.0)):
            mirrored = find_vertex(bm, expected)
            assert mirrored is not None, expected
            assert mirrored.select is False
    finally:
        bm.free()


def check_o2_weak_surface_band():
    """O2: accept a weakly curved surface band with exact mirror coordinates."""

    fixture = fixture_from_builder("weak_surface_band", build_weak_surface_band)
    try:
        interiors = _path_interior_vertices(fixture.source_edges)
        assert len(interiors) == 1
        source_vertex = interiors[0]
        target_face = _target_face_for_vertex(fixture, source_vertex)
        reflected = matching.mirror_coordinate(source_vertex.co, fixture.axis)
        source_distance = _surface_distance(
            source_vertex.co, next(face for face in source_vertex.link_faces if face.is_valid)
        )
        target_distance = _surface_distance(reflected, target_face)
        s_eff = _s_eff(fixture, source_vertex, target_face)
        assert source_distance <= 1.0e-6, source_distance
        assert 2.0 * fixture.tolerance < target_distance <= s_eff, (target_distance, s_eff)
        assert _call_gate(fixture.bm, fixture.source_edges, fixture.topology)
        created, already, reason = _call_apply(fixture.bm, fixture.source_edges, fixture.topology)
        assert reason == "", reason
        assert created + already == len(fixture.source_edges)
        mirrored = next(
            vertex for vertex in fixture.bm.verts if matching.coordinates_match(vertex.co, reflected, fixture.tolerance)
        )
        assert tuple(mirrored.co) == tuple(reflected)
    finally:
        fixture.free()


def check_o3a_sanity_decline():
    """O3(a): a constructed point beyond S_eff declines for the R-H3 reason."""

    fixture = fixture_from_builder("sanity_excess_band", build_sanity_excess_band)
    try:
        source_vertex = _path_interior_vertices(fixture.source_edges)[0]
        target_face = _target_face_for_vertex(fixture, source_vertex)
        target_distance = _surface_distance(matching.mirror_coordinate(source_vertex.co, fixture.axis), target_face)
        s_eff = _s_eff(fixture, source_vertex, target_face)
        assert target_distance > s_eff, (target_distance, s_eff)
        assert not _call_gate(fixture.bm, fixture.source_edges, fixture.topology)
        created, already, reason = _call_apply(fixture.bm, fixture.source_edges, fixture.topology)
        assert created == 0 and already == 0
        assert reason and ("s_eff" in reason.lower() or "sanity" in reason.lower()), reason
    finally:
        fixture.free()


def check_o3b_projection_containment_decline():
    """O3(b): a near-surface point outside the projected carrier declines."""

    fixture = fixture_from_builder("projection_outside_band", build_projection_outside_band)
    try:
        source_vertex = _path_interior_vertices(fixture.source_edges)[0]
        target_face = _target_face_for_vertex(fixture, source_vertex)
        face_layer = fixture.bm.faces.layers.int.get(layer_names.FACE_ID_LAYER)
        assert face_layer is not None
        frame = fixture.topology.carrier_frames[int(target_face[face_layer])]
        projected = _project_to_carrier(matching.mirror_coordinate(source_vertex.co, fixture.axis), frame)
        target_distance = _surface_distance(matching.mirror_coordinate(source_vertex.co, fixture.axis), target_face)
        s_eff = _s_eff(fixture, source_vertex, target_face)
        assert 2.0 * fixture.tolerance < target_distance <= s_eff, (target_distance, s_eff)
        assert projected.x > max(float(vertex.co.x) for vertex in target_face.verts), tuple(projected)
        assert not _call_gate(fixture.bm, fixture.source_edges, fixture.topology)
        created, already, reason = _call_apply(fixture.bm, fixture.source_edges, fixture.topology)
        assert created == 0 and already == 0
        assert reason and "project" in reason.lower(), reason
    finally:
        fixture.free()


def check_o4c_endpoint_collision_apply_decline():
    """O4(c): pre-split instances + exact endpoint collapse reach R-H4."""

    fixture = fixture_from_builder("endpoint_collision", build_endpoint_collision)
    try:
        interior = _path_interior_vertices(fixture.source_edges)[0]
        assert len(_target_instances(fixture, interior)) == 2
        path_vertices = {hash(vertex): vertex for edge in fixture.source_edges for vertex in edge.verts}
        degree = {key: 0 for key in path_vertices}
        for edge in fixture.source_edges:
            degree[hash(edge.verts[0])] += 1
            degree[hash(edge.verts[1])] += 1
        endpoints = [path_vertices[key] for key, count in degree.items() if count == 1]
        assert len(endpoints) == 2
        target_vertex = next(
            vertex
            for vertex in fixture.bm.verts
            if matching.coordinates_match(
                vertex.co, matching.mirror_coordinate(endpoints[0].co, fixture.axis), fixture.tolerance
            )
        )
        assert matching.coordinates_match(
            target_vertex.co,
            matching.mirror_coordinate(endpoints[1].co, fixture.axis),
            fixture.tolerance,
        )
        assert _call_gate(fixture.bm, fixture.source_edges, fixture.topology)
        created, already, reason = _call_apply(fixture.bm, fixture.source_edges, fixture.topology)
        assert created == 0 and already == 0
        assert reason and any(token in reason.lower() for token in ("split", "endpoint", "face")), reason
    finally:
        fixture.free()


def check_o4a_lineage_apply_decline():
    """O4(a): all members share an ID but classification lineages disagree."""

    fixture = fixture_from_builder("lineage_mismatch_chain", build_lineage_mismatch_chain)
    try:
        assert len(_target_instances(fixture, _path_interior_vertices(fixture.source_edges)[0])) == 2
        assert _call_gate(fixture.bm, fixture.source_edges, fixture.topology)
        created, already, reason = _call_apply(fixture.bm, fixture.source_edges, fixture.topology)
        assert created == 0 and already == 0
        assert reason and "lineage" in reason.lower(), reason
    finally:
        fixture.free()


def check_o4b_endpoint_link_apply_decline():
    """O4(b): geometric winner does not link both chain endpoints."""

    fixture = fixture_from_builder("endpoint_link_mismatch", build_endpoint_link_mismatch)
    try:
        assert len(_target_instances(fixture, _path_interior_vertices(fixture.source_edges)[0])) == 2
        assert _call_gate(fixture.bm, fixture.source_edges, fixture.topology)
        created, already, reason = _call_apply(fixture.bm, fixture.source_edges, fixture.topology)
        assert created == 0 and already == 0
        assert reason and any(token in reason.lower() for token in ("link", "endpoint", "place")), reason
    finally:
        fixture.free()


def check_o10_both_side_stroke():
    """O10: a stroke already present on both sides remains directly accepted."""

    fixture = fixture_from_builder("rev3_both_side_stroke", build_rev3_both_side_stroke)
    try:
        assert _call_gate(fixture.bm, fixture.source_edges, fixture.topology)
        created, already, reason = _call_apply(fixture.bm, fixture.source_edges, fixture.topology)
        assert reason == "", reason
        assert created + already == len(fixture.source_edges), (created, already)
        assert already >= 1
    finally:
        fixture.free()


def check_o14_wire_mixed():
    """O14: wires mirror as wires alongside the face chain (R-W1)."""

    from fixtures_interior_host import build_wire_mixed

    fixture = fixture_from_builder("wire_mixed", build_wire_mixed)
    try:
        bm = fixture.bm
        assert _call_gate(bm, fixture.source_edges, fixture.topology)
        created, already, reason = _call_apply(bm, fixture.source_edges, fixture.topology)
        assert reason == "", reason
        assert created == len(fixture.source_edges) and already == 0, (created, already)
        for edge in fixture.source_edges:
            ra = Vector((-edge.verts[0].co.x, edge.verts[0].co.y, edge.verts[0].co.z))
            rb = Vector((-edge.verts[1].co.x, edge.verts[1].co.y, edge.verts[1].co.z))
            mirrored = next(
                (
                    other
                    for other in bm.edges
                    if other.is_valid
                    and (
                        ((other.verts[0].co - ra).length <= 1e-6 and (other.verts[1].co - rb).length <= 1e-6)
                        or ((other.verts[0].co - rb).length <= 1e-6 and (other.verts[1].co - ra).length <= 1e-6)
                    )
                ),
                None,
            )
            assert mirrored is not None, (tuple(ra), tuple(rb))
            assert mirrored.is_wire == edge.is_wire
    finally:
        fixture.free()


def check_o14_wire_ambiguous_decline():
    """O14: a wire endpoint with two tol-close reflected vertices declines."""

    from fixtures_interior_host import build_wire_ambiguous

    fixture = fixture_from_builder("wire_ambiguous", build_wire_ambiguous)
    try:
        assert not _call_gate(fixture.bm, fixture.source_edges, fixture.topology)
        created, already, reason = _call_apply(fixture.bm, fixture.source_edges, fixture.topology)
        assert reason and "wire" in reason, (created, already, reason)
    finally:
        fixture.free()


def _coordinate(vertex):
    return tuple(round(float(value), 7) for value in vertex.co)


def _normalized_incidence(bm):
    edges = tuple(sorted(tuple(sorted((_coordinate(edge.verts[0]), _coordinate(edge.verts[1])))) for edge in bm.edges))
    faces = tuple(sorted(tuple(sorted(_coordinate(vertex) for vertex in face.verts)) for face in bm.faces))
    return edges, faces


def _assert_all_edges_reflected(fixture):
    for source_edge in fixture.source_edges:
        expected_a = matching.mirror_coordinate(source_edge.verts[0].co, fixture.axis)
        expected_b = matching.mirror_coordinate(source_edge.verts[1].co, fixture.axis)
        assert any(
            (tuple(edge.verts[0].co) == tuple(expected_a) and tuple(edge.verts[1].co) == tuple(expected_b))
            or (tuple(edge.verts[0].co) == tuple(expected_b) and tuple(edge.verts[1].co) == tuple(expected_a))
            for edge in fixture.bm.edges
            if edge.is_valid
        ), (tuple(expected_a), tuple(expected_b))


def _network_plan_signature(fixture):
    """Return canonical planner paths independent of source edge iteration."""

    face_layer = fixture.bm.faces.layers.int.get(layer_names.FACE_ID_LAYER)
    assert face_layer is not None
    source_vertex_by_key, target_ids, records, unmatched, _status = stitch_reflect._collect_reflected_path_context(
        fixture.source_edges, face_layer, fixture.topology.mirror_face_ids, require_all_mirrored=True
    )
    assert not unmatched
    target_faces = stitch_reflect._target_faces_by_id(
        fixture.bm, face_layer, {target_id for ids in target_ids.values() for target_id in ids}
    )
    source_face_ids = stitch_reflect._source_face_ids_by_vertex(fixture.source_edges, face_layer)
    classification, reason = stitch_reflect._classify_reflected_vertices(
        source_vertex_by_key,
        target_ids,
        target_faces,
        fixture.axis,
        fixture.tolerance,
        edge_records=records,
        source_face_ids_by_vertex=source_face_ids,
        carrier_frames=fixture.topology.carrier_frames,
    )
    assert not reason
    adjacency = stitch_reflect._path_adjacency(records)
    occurrence = {hash(vertex): index for index, vertex in enumerate(tuple(fixture.bm.verts))}
    snapshot = stitch_reflect._network_snapshot(source_vertex_by_key, records, classification, adjacency, occurrence)
    plan = stitch_reflect._plan_interior_network(snapshot)
    assert not plan.reason, plan.reason
    return tuple(tuple(snapshot.rank[key] for key in path.vertices) for path in plan.paths)


def check_o15_networks():
    """O15(a)-(e),(g): hubs, relay ordering, theta success, and pure declines."""

    for name, builder, degree in (
        ("network_y", build_interior_network_y, 3),
        ("network_x", build_interior_network_x, 4),
        ("network_branch_relay", build_interior_network_branch_relay, None),
        ("network_theta", build_interior_network_theta, None),
    ):
        fixture = fixture_from_builder(name, builder)
        try:
            if name == "network_branch_relay":
                signature = _network_plan_signature(fixture)
                lengths = [len(path) - 1 for path in signature]
                first_single = lengths.index(1)
                assert any(length > 1 for length in lengths[first_single + 1 :]), signature
            before = (len(fixture.bm.verts), len(fixture.bm.edges), len(fixture.bm.faces))
            assert _call_gate(fixture.bm, fixture.source_edges, fixture.topology)
            assert before == (len(fixture.bm.verts), len(fixture.bm.edges), len(fixture.bm.faces))
            created, already, reason = _call_apply(fixture.bm, fixture.source_edges, fixture.topology)
            assert reason == "", (name, reason)
            assert created == len(fixture.source_edges) and already == 0, (name, created, already)
            _assert_all_edges_reflected(fixture)
            source_adjacency = {}
            source_vertices = {}
            for edge in fixture.source_edges:
                left, right = edge.verts
                source_adjacency.setdefault(hash(left), set()).add(hash(right))
                source_adjacency.setdefault(hash(right), set()).add(hash(left))
                source_vertices[hash(left)] = left
                source_vertices[hash(right)] = right
            hubs = [
                source_vertices[key]
                for key, neighbours in source_adjacency.items()
                if degree is not None and len(neighbours) == degree
            ]
            if degree is None:
                continue
            assert hubs, (name, degree)
            for source_hub in hubs:
                reflected = matching.mirror_coordinate(source_hub.co, fixture.axis)
                mirrored_hub = find_vertex(fixture.bm, reflected)
                assert mirrored_hub is not None and len(mirrored_hub.link_edges) == degree
        finally:
            fixture.free()

    first = fixture_from_builder("network_y_order_a", build_interior_network_y)
    second = fixture_from_builder("network_y_order_b", build_interior_network_y)
    try:
        for vertex in tuple(first.bm.verts) + tuple(second.bm.verts):
            vertex.index = -1
        edges = list(second.source_edges)
        second.source_edges[:] = list(reversed(edges))
        first_plan = _network_plan_signature(first)
        second_plan = _network_plan_signature(second)
        assert first_plan == second_plan, (first_plan, second_plan)
        first_before = (len(first.bm.verts), len(first.bm.edges), len(first.bm.faces))
        second_before = (len(second.bm.verts), len(second.bm.edges), len(second.bm.faces))
        assert _call_gate(first.bm, first.source_edges, first.topology)
        assert _call_gate(second.bm, second.source_edges, second.topology)
        assert first_before == (len(first.bm.verts), len(first.bm.edges), len(first.bm.faces))
        assert second_before == (len(second.bm.verts), len(second.bm.edges), len(second.bm.faces))
        assert _call_apply(first.bm, first.source_edges, first.topology)[2] == ""
        assert _call_apply(second.bm, second.source_edges, second.topology)[2] == ""
        assert _normalized_incidence(first.bm) == _normalized_incidence(second.bm)
    finally:
        first.free()
        second.free()

    theta = fixture_from_builder("network_theta_shape", build_interior_network_theta)
    try:
        adjacency = {}
        for edge in theta.source_edges:
            left, right = map(hash, edge.verts)
            adjacency.setdefault(left, set()).add(right)
            adjacency.setdefault(right, set()).add(left)
        assert sorted(len(neighbours) for neighbours in adjacency.values()).count(1) == 2
        assert len(theta.source_edges) == len(adjacency), "the two-anchor graph must contain an interior cycle"
    finally:
        theta.free()

    for name, builder in (
        ("network_decline_loop", build_interior_network_decline_loop),
        ("network_decline_one_anchor", build_interior_network_decline_one_anchor),
        ("network_decline_dead_end", build_interior_network_decline_dead_end),
    ):
        fixture = fixture_from_builder(name, builder)
        try:
            before = (len(fixture.bm.verts), len(fixture.bm.edges), len(fixture.bm.faces))
            assert not _call_gate(fixture.bm, fixture.source_edges, fixture.topology), name
            assert before == (len(fixture.bm.verts), len(fixture.bm.edges), len(fixture.bm.faces)), name
            created, already, reason = _call_apply(fixture.bm, fixture.source_edges, fixture.topology)
            assert created == 0 and already == 0 and reason, (name, created, already, reason)
        finally:
            fixture.free()


def run():
    check_interior_chain_n1()
    print("PASS check_interior_chain_n1", flush=True)
    check_interior_chain_n2()
    print("PASS check_interior_chain_n2", flush=True)
    check_decline_degree3_interior()
    print("PASS check_decline_degree3_interior", flush=True)
    check_decline_no_common_face()
    print("PASS check_decline_no_common_face", flush=True)
    check_curved_quad_single_candidate()
    print("PASS check_curved_quad_single_candidate", flush=True)
    check_two_chains_same_face()
    print("PASS check_two_chains_same_face", flush=True)
    check_o2_weak_surface_band()
    print("PASS check_o2_weak_surface_band", flush=True)
    check_o3a_sanity_decline()
    print("PASS check_o3a_sanity_decline", flush=True)
    check_o3b_projection_containment_decline()
    print("PASS check_o3b_projection_containment_decline", flush=True)
    check_o4a_lineage_apply_decline()
    print("PASS check_o4a_apply_decline", flush=True)
    check_o4b_endpoint_link_apply_decline()
    print("PASS check_o4b_apply_decline", flush=True)
    check_o4c_endpoint_collision_apply_decline()
    print("PASS check_o4c_endpoint_collision_apply_decline", flush=True)
    check_o10_both_side_stroke()
    print("PASS check_o10_both_side_stroke", flush=True)
    check_o14_wire_mixed()
    print("PASS check_o14_wire_mixed", flush=True)
    check_o14_wire_ambiguous_decline()
    print("PASS check_o14_wire_ambiguous_decline", flush=True)
    check_o15_networks()
    print("PASS check_o15_networks", flush=True)
    print("YSE_INTERIOR_CHAIN_OK", flush=True)


if __name__ == "__main__":
    try:
        run()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
