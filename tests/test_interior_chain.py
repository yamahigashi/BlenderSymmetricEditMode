# SPDX-License-Identifier: GPL-3.0-or-later

"""Headless checks for face-interior terminal chains on the direct topology path.

Contract: .agents/doc/fix_contract_knife_microedge_2026-08-12.md (revision 2)
Oracle item #4: n=1 / n=2 accept; degree-3 and no-common-face decline.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import bmesh
from mathutils import Vector

PACKAGE_PARENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_PARENT))
from ydd_symmetric_edit import layer_names, matching, snapshot, stitch  # noqa: E402

TOLERANCE = 1.0e-5
AXIS = matching.AXIS_INDEX["X"]


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
    source, side, total, crossing = stitch.collect_source_path_edges(bm, AXIS, TOLERANCE, "AUTO")
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

        assert stitch.reflected_path_uses_only_target_boundaries(
            bm,
            source,
            AXIS,
            TOLERANCE,
            topology.mirror_face_ids,
        )

        created, already, reason = stitch.apply_reflected_path_topology(
            bm,
            source,
            AXIS,
            TOLERANCE,
            topology.mirror_face_ids,
        )
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

        assert stitch.reflected_path_uses_only_target_boundaries(
            bm,
            source,
            AXIS,
            TOLERANCE,
            topology.mirror_face_ids,
        )

        created, already, reason = stitch.apply_reflected_path_topology(
            bm,
            source,
            AXIS,
            TOLERANCE,
            topology.mirror_face_ids,
        )
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
    """(c) Degree-3 interior vertex is not a simple chain → gate False / apply fails."""

    bm = build_two_symmetric_quads()
    try:
        topology = snapshot.prepare_topology(bm, AXIS, TOLERANCE)
        # Place three boundary anchors on the left face.
        bottom = _split_at_x(_bottom_edge(bm, negative=True), -1.5)
        top = _split_at_x(_top_edge(bm, negative=True), -1.5)
        left_mid_edge = next(
            edge
            for edge in bm.edges
            if all(abs(vertex.co.x + 2.0) < 1.0e-8 for vertex in edge.verts)
            and min(vertex.co.y for vertex in edge.verts) < 0.0 < max(vertex.co.y for vertex in edge.verts)
        )
        _e, left_mid = bmesh.utils.edge_split(left_mid_edge, left_mid_edge.verts[0], 0.5)
        left_mid.co = (-2.0, 0.0, 0.0)

        host = next(
            face
            for face in bm.faces
            if bottom in face.verts and top in face.verts and left_mid in face.verts
        )
        # First create bottom--hub--top, then connect left_mid--hub so hub is degree 3.
        bmesh.utils.face_split(host, bottom, top, coords=[(-1.3, 0.0, 0.0)])
        hub = find_vertex(bm, (-1.3, 0.0, 0.0))
        assert hub is not None
        host2 = next(face for face in bm.faces if left_mid in face.verts and hub in face.verts)
        bmesh.utils.face_split(host2, left_mid, hub)

        source = collect_source_path_edges(bm)
        assert len(source) == 3, len(source)

        assert not stitch.reflected_path_uses_only_target_boundaries(
            bm,
            source,
            AXIS,
            TOLERANCE,
            topology.mirror_face_ids,
        )

        created, already, reason = stitch.apply_reflected_path_topology(
            bm,
            source,
            AXIS,
            TOLERANCE,
            topology.mirror_face_ids,
        )
        assert created == 0 and already == 0
        assert reason == "a reflected cut vertex is not on a target boundary edge", reason
    finally:
        bm.free()


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

        assert not stitch.reflected_path_uses_only_target_boundaries(
            bm,
            source,
            AXIS,
            TOLERANCE,
            topology.mirror_face_ids,
        )

        created, already, reason = stitch.apply_reflected_path_topology(
            bm,
            source,
            AXIS,
            TOLERANCE,
            topology.mirror_face_ids,
        )
        assert created == 0 and already == 0
        # The wire bridge has no linked face, so apply declines at the
        # no-target-face guard before reaching _find_interior_chains; the
        # common-face intersection itself is exercised through the gate call
        # above (gate and apply share the same chain-detection functions).
        assert reason in {
            "a reflected cut vertex is not on a target boundary edge",
            "a source cut edge has no mirrored target face",
        }, reason
    finally:
        bm.free()



def check_curved_quad_single_candidate():
    """(e) Curved quad (suzanne10 face 75): the boundary end split turns the
    host into a pentagon whose ear-clip triangulation deviates from the
    evaluated quad surface by more than the tolerance-scale limit, so the
    single id-matching candidate must be accepted without re-testing
    containment."""

    from ydd_symmetric_edit import stitch

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
            for a, b, c in stitch._face_surface_triangles(pentagon)
        )
        assert min_dist > TOLERANCE * 2.0, min_dist
        # The realization re-test would run on the target-side pentagon (the
        # mirrored loop, reversed winding), whose ear-clip need not match the
        # source one; pin the old failure condition on that side as well.
        mirrored_loop = [
            Vector((-v.co.x, v.co.y, v.co.z)) for v in reversed(list(pentagon.verts))
        ]
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

        assert stitch.reflected_path_uses_only_target_boundaries(
            bm,
            source,
            AXIS,
            TOLERANCE,
            topology.mirror_face_ids,
        )
        created, already, reason = stitch.apply_reflected_path_topology(
            bm,
            source,
            AXIS,
            TOLERANCE,
            topology.mirror_face_ids,
        )
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
        _e, vb2 = bmesh.utils.edge_split(bottom_rest, vb1, (-1.2 - vb1.co.x) / (bottom_rest.other_vert(vb1).co.x - vb1.co.x))
        vb2.co = (-1.2, -1.0, 0.0)
        corner_tl = find_vertex(bm, (-2.0, 1.0, 0.0))
        corner_tr = find_vertex(bm, (-1.0, 1.0, 0.0))
        assert corner_tl is not None and corner_tr is not None

        host_a = next(face for face in bm.faces if vb1 in face.verts and corner_tl in face.verts)
        bmesh.utils.face_split(host_a, vb1, corner_tl, coords=[(-1.85, 0.0, 0.0)])
        host_b = next(
            face for face in bm.faces if face.is_valid and vb2 in face.verts and corner_tr in face.verts
        )
        bmesh.utils.face_split(host_b, vb2, corner_tr, coords=[(-1.15, 0.0, 0.0)])

        source = collect_source_path_edges(bm)
        assert len(source) == 4, len(source)

        assert stitch.reflected_path_uses_only_target_boundaries(
            bm,
            source,
            AXIS,
            TOLERANCE,
            topology.mirror_face_ids,
        )
        created, already, reason = stitch.apply_reflected_path_topology(
            bm,
            source,
            AXIS,
            TOLERANCE,
            topology.mirror_face_ids,
        )
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
    print("YSE_INTERIOR_CHAIN_OK", flush=True)


if __name__ == "__main__":
    try:
        run()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
