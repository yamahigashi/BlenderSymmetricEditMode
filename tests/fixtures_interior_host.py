# SPDX-License-Identifier: GPL-3.0-or-later

"""Deterministic BMesh fixtures for the interior-host contract.

Network builders reference R-N1/R-D2 in
``.agents/doc/fix_contract_knife_interior_host_2026-08-13.md`` v7.1.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import bmesh
from mathutils import Vector

PACKAGE_PARENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_PARENT))

from ydd_symmetric_edit import layer_names, matching, snapshot, stitch_pathedges  # noqa: E402
from ydd_symmetric_edit._types import FaceId  # noqa: E402

TOLERANCE = 1.0e-5
AXIS = matching.AXIS_INDEX["X"]

# Builders retain the preparation made before native path topology is added.
_PREPARATIONS: dict[int, object] = {}
_EXPLICIT_SOURCE_EDGES: dict[int, tuple[bmesh.types.BMEdge, ...]] = {}


@dataclass
class InteriorHostFixture:
    name: str
    bm: bmesh.types.BMesh
    source_edges: list[bmesh.types.BMEdge]
    topology: object
    axis: int = AXIS
    tolerance: float = TOLERANCE

    def free(self) -> None:
        _PREPARATIONS.pop(id(self.bm), None)
        _EXPLICIT_SOURCE_EDGES.pop(id(self.bm), None)
        self.bm.free()


def _prepare_before_native_cut(bm: bmesh.types.BMesh) -> None:
    _PREPARATIONS[id(bm)] = snapshot.prepare_topology(bm, AXIS, TOLERANCE)


def _base_pair() -> bmesh.types.BMesh:
    bm = bmesh.new()
    left = [bm.verts.new(co) for co in ((-2.0, -1.0, 0.0), (-1.0, -1.0, 0.0), (-1.0, 1.0, 0.0), (-2.0, 1.0, 0.0))]
    right = [bm.verts.new(co) for co in ((1.0, -1.0, 0.0), (2.0, -1.0, 0.0), (2.0, 1.0, 0.0), (1.0, 1.0, 0.0))]
    bm.faces.new(left)
    bm.faces.new(right)
    bm.normal_update()
    return bm


def _boundary_edge(bm: bmesh.types.BMesh, *, y: float, x: float | None = None):
    candidates = [
        edge
        for edge in bm.edges
        if all(abs(float(vertex.co.y) - y) <= 1.0e-8 for vertex in edge.verts)
        and all(float(vertex.co.x) < 0.0 for vertex in edge.verts)
    ]
    if x is None:
        return candidates[0]
    for edge in candidates:
        lo = min(float(vertex.co.x) for vertex in edge.verts)
        hi = max(float(vertex.co.x) for vertex in edge.verts)
        if lo < x < hi:
            return edge
    raise AssertionError((y, x, [(tuple(vertex.co) for vertex in edge.verts) for edge in candidates]))


def _split_boundary(bm: bmesh.types.BMesh, *, y: float, x: float):
    edge = _boundary_edge(bm, y=y, x=x)
    start = edge.verts[0]
    factor = (x - float(start.co.x)) / (float(edge.verts[1].co.x) - float(start.co.x))
    _new_edge, vertex = bmesh.utils.edge_split(edge, start, factor)
    vertex.co = (x, y, float(vertex.co.z))
    return vertex


def _split_side_boundary(bm: bmesh.types.BMesh, *, y: float, x: float, positive: bool):
    edge = next(
        edge
        for edge in bm.edges
        if all(abs(float(vertex.co.y) - y) <= 1.0e-8 for vertex in edge.verts)
        and all((float(vertex.co.x) > 0.0) == positive for vertex in edge.verts)
        and min(float(vertex.co.x) for vertex in edge.verts) < x < max(float(vertex.co.x) for vertex in edge.verts)
    )
    start = edge.verts[0]
    factor = (x - float(start.co.x)) / (float(edge.verts[1].co.x) - float(start.co.x))
    return bmesh.utils.edge_split(edge, start, factor)[1]


def _split_vertical_boundary(bm: bmesh.types.BMesh, *, x: float, y: float):
    edge = next(
        edge
        for edge in bm.edges
        if all(abs(float(vertex.co.x) - x) <= 1.0e-8 for vertex in edge.verts)
        and min(float(vertex.co.y) for vertex in edge.verts) < y < max(float(vertex.co.y) for vertex in edge.verts)
    )
    start = edge.verts[0]
    factor = (y - float(start.co.y)) / (float(edge.verts[1].co.y) - float(start.co.y))
    return bmesh.utils.edge_split(edge, start, factor)[1]


def _host(bm: bmesh.types.BMesh, first, second):
    return next(face for face in bm.faces if first in face.verts and second in face.verts)


def build_interior_chain_n1() -> bmesh.types.BMesh:
    bm = _base_pair()
    _prepare_before_native_cut(bm)
    bottom = _split_boundary(bm, y=-1.0, x=-1.5)
    top = _split_boundary(bm, y=1.0, x=-1.5)
    bmesh.utils.face_split(_host(bm, bottom, top), bottom, top, coords=[(-1.2, 0.0, 0.0)])
    return bm


def build_interior_chain_n2() -> bmesh.types.BMesh:
    bm = _base_pair()
    _prepare_before_native_cut(bm)
    bottom = _split_boundary(bm, y=-1.0, x=-1.5)
    top = _split_boundary(bm, y=1.0, x=-1.5)
    bmesh.utils.face_split(
        _host(bm, bottom, top),
        bottom,
        top,
        coords=[(-1.4, -0.25, 0.0), (-1.2, 0.25, 0.0)],
    )
    return bm


def build_interior_network_y() -> bmesh.types.BMesh:
    """Y: three boundary anchors and one degree-three interior hub (R-N1)."""

    bm = _base_pair()
    _prepare_before_native_cut(bm)
    bottom = _split_boundary(bm, y=-1.0, x=-1.5)
    top = _split_boundary(bm, y=1.0, x=-1.5)
    left = _split_vertical_boundary(bm, x=-2.0, y=0.0)
    bmesh.utils.face_split(_host(bm, bottom, top), bottom, top, coords=[(-1.3, 0.0, 0.0)])
    hub = next(
        vertex for vertex in bm.verts if abs(float(vertex.co.x) + 1.3) < 1.0e-6 and abs(float(vertex.co.y)) < 1.0e-6
    )
    bmesh.utils.face_split(_host(bm, left, hub), left, hub)
    return bm


build_network_y = build_interior_network_y


def build_interior_network_x() -> bmesh.types.BMesh:
    """X: four boundary anchors and one degree-four crossing hub (R-N1)."""

    bm = _base_pair()
    _prepare_before_native_cut(bm)
    bottom = _split_boundary(bm, y=-1.0, x=-1.5)
    top = _split_boundary(bm, y=1.0, x=-1.5)
    left = _split_vertical_boundary(bm, x=-2.0, y=-0.25)
    right = _split_vertical_boundary(bm, x=-1.0, y=0.25)
    bmesh.utils.face_split(_host(bm, bottom, top), bottom, top, coords=[(-1.3, 0.0, 0.0)])
    hub = next(
        vertex for vertex in bm.verts if abs(float(vertex.co.x) + 1.3) < 1.0e-6 and abs(float(vertex.co.y)) < 1.0e-6
    )
    bmesh.utils.face_split(_host(bm, left, hub), left, hub)
    bmesh.utils.face_split(_host(bm, right, hub), right, hub)
    return bm


build_network_x = build_interior_network_x


def build_interior_network_branch_relay() -> bmesh.types.BMesh:
    """Branch plus relay; a single-edge path precedes a later multi path (R-N1)."""

    bm = _base_pair()
    _prepare_before_native_cut(bm)
    bottom = _split_boundary(bm, y=-1.0, x=-1.7)
    top = _split_boundary(bm, y=1.0, x=-1.4)
    left = _split_vertical_boundary(bm, x=-2.0, y=-0.1)
    right = _split_vertical_boundary(bm, x=-1.0, y=0.25)
    bmesh.utils.face_split(_host(bm, bottom, top), bottom, top, coords=[(-1.3, 0.0, 0.0)])
    hub = next(
        vertex for vertex in bm.verts if abs(float(vertex.co.x) + 1.3) < 1.0e-6 and abs(float(vertex.co.y)) < 1.0e-6
    )
    bmesh.utils.face_split(_host(bm, left, hub), left, hub)
    bmesh.utils.face_split(_host(bm, right, hub), right, hub, coords=[(-1.2, 0.35, 0.0)])
    return bm


build_network_branch_relay = build_interior_network_branch_relay


def _replace_left_face_with_graph(
    bm: bmesh.types.BMesh,
    interior_coordinates: tuple[tuple[float, float, float], ...],
    face_cycles: tuple[tuple[str | int, ...], ...],
    source_pairs: tuple[tuple[str | int, str | int], ...],
) -> bmesh.types.BMesh:
    """Replace the source quad while retaining its snapshot FaceId.

    The explicit edge list isolates the contract graph from triangulation
    support edges.  All replacement faces inherit the original carrier ID.
    """

    face_layer = bm.faces.layers.int.get(layer_names.FACE_ID_LAYER)
    assert face_layer is not None
    source_face = next(face for face in bm.faces if all(float(vertex.co.x) < 0.0 for vertex in face.verts))
    face_id = int(source_face[face_layer])
    corners = {
        "a": next(vertex for vertex in source_face.verts if tuple(vertex.co) == (-2.0, -1.0, 0.0)),
        "b": next(vertex for vertex in source_face.verts if tuple(vertex.co) == (-1.0, -1.0, 0.0)),
        "c": next(vertex for vertex in source_face.verts if tuple(vertex.co) == (-1.0, 1.0, 0.0)),
        "d": next(vertex for vertex in source_face.verts if tuple(vertex.co) == (-2.0, 1.0, 0.0)),
    }
    interior = {index: bm.verts.new(coordinate) for index, coordinate in enumerate(interior_coordinates)}
    vertices: dict[str | int, bmesh.types.BMVert] = {**corners, **interior}
    bmesh.ops.delete(bm, geom=[source_face], context="FACES_ONLY")
    for cycle in face_cycles:
        face = bm.faces.new([vertices[key] for key in cycle])
        face[face_layer] = face_id
    source_edges = tuple(bm.edges.get((vertices[left], vertices[right])) for left, right in source_pairs)
    assert all(edge is not None and edge.link_faces for edge in source_edges)
    _EXPLICIT_SOURCE_EDGES[id(bm)] = source_edges
    bm.normal_update()
    return bm


def build_interior_network_theta() -> bmesh.types.BMesh:
    """R-N1/O15(e): two anchors attached to an interior four-edge cycle."""

    bm = _base_pair()
    _prepare_before_native_cut(bm)
    result = _replace_left_face_with_graph(
        bm,
        (
            (-1.7, -0.3, 0.0),
            (-1.3, -0.3, 0.0),
            (-1.3, 0.3, 0.0),
            (-1.7, 0.3, 0.0),
        ),
        (
            ("a", "b", 0),
            ("a", 0, 3),
            ("a", 3, "d"),
            ("d", 3, "c"),
            ("b", 1, 0),
            ("b", "c", 2),
            ("b", 2, 1),
            ("c", 3, 2),
            (0, 1, 2, 3),
        ),
        ((0, 1), (1, 2), (2, 3), (3, 0), ("b", 0), ("c", 3)),
    )
    return result


build_network_theta = build_interior_network_theta


def build_interior_network_decline_loop() -> bmesh.types.BMesh:
    """R-D2/O15(d): an anchor-free triangular interior loop."""

    bm = _base_pair()
    _prepare_before_native_cut(bm)
    return _replace_left_face_with_graph(
        bm,
        ((-1.65, -0.3, 0.0), (-1.25, 0.0, 0.0), (-1.65, 0.3, 0.0)),
        (
            ("a", "b", 0),
            ("b", 1, 0),
            ("b", "c", 1),
            ("c", 2, 1),
            ("c", "d", 2),
            ("d", "a", 0),
            ("d", 0, 2),
            (0, 1, 2),
        ),
        ((0, 1), (1, 2), (2, 0)),
    )


def build_interior_network_decline_one_anchor() -> bmesh.types.BMesh:
    """R-D2/O15(d): one boundary anchor attached to an interior loop."""

    bm = _base_pair()
    _prepare_before_native_cut(bm)
    return _replace_left_face_with_graph(
        bm,
        ((-1.65, -0.3, 0.0), (-1.25, 0.0, 0.0), (-1.65, 0.3, 0.0)),
        (
            ("a", "b", 0),
            ("b", 1, 0),
            ("b", "c", 1),
            ("c", 2, 1),
            ("c", "d", 2),
            ("d", "a", 0),
            ("d", 0, 2),
            (0, 1, 2),
        ),
        ((0, 1), (1, 2), (2, 0), ("a", 0)),
    )


def build_interior_network_decline_dead_end() -> bmesh.types.BMesh:
    """R-D2/O15(d): two-anchor main path plus an interior dead end."""

    bm = _base_pair()
    _prepare_before_native_cut(bm)
    return _replace_left_face_with_graph(
        bm,
        ((-1.4, 0.0, 0.0), (-1.75, 0.0, 0.0)),
        (
            ("a", "b", 1),
            ("b", 0, 1),
            ("b", "c", 0),
            ("c", "d", 1),
            ("c", 1, 0),
            ("d", "a", 1),
        ),
        (("b", 0), (0, "c"), (0, 1)),
    )


def build_plane_two_chains() -> bmesh.types.BMesh:
    """Two parallel plane chains whose source spacing is exactly 10*tol."""

    bm = _base_pair()
    _prepare_before_native_cut(bm)
    delta = 10.0 * TOLERANCE
    first_bottom = _split_boundary(bm, y=-1.0, x=-1.5)
    first_top = _split_boundary(bm, y=1.0, x=-1.5)
    bmesh.utils.face_split(_host(bm, first_bottom, first_top), first_bottom, first_top, coords=[(-1.3, 0.0, 0.0)])
    second_bottom = _split_boundary(bm, y=-1.0, x=-1.5 + delta)
    second_top = _split_boundary(bm, y=1.0, x=-1.5 + delta)
    bmesh.utils.face_split(
        _host(bm, second_bottom, second_top),
        second_bottom,
        second_top,
        coords=[(-1.3 + delta, 0.0, 0.0)],
    )
    assert abs((second_bottom.co - first_bottom.co).length - delta) <= TOLERANCE * 0.1
    return bm


build_two_plane_chains = build_plane_two_chains


def build_rev3_both_side_stroke() -> bmesh.types.BMesh:
    """A two-edge interior stroke already present on both mirrored faces."""

    bm = _base_pair()
    _prepare_before_native_cut(bm)
    for negative in (True, False):
        sign = -1.0 if negative else 1.0
        edge = next(
            edge
            for edge in bm.edges
            if all(vertex.co.x * sign > 0.0 for vertex in edge.verts)
            and all(abs(vertex.co.y + 1.0) <= 1.0e-8 for vertex in edge.verts)
        )
        _new_edge, bottom = bmesh.utils.edge_split(edge, edge.verts[0], 0.5)
        edge = next(
            edge
            for edge in bm.edges
            if all(vertex.co.x * sign > 0.0 for vertex in edge.verts)
            and all(abs(vertex.co.y - 1.0) <= 1.0e-8 for vertex in edge.verts)
        )
        _new_edge, top = bmesh.utils.edge_split(edge, edge.verts[0], 0.5)
        bmesh.utils.face_split(_host(bm, bottom, top), bottom, top, coords=[(sign * 1.3, 0.0, 0.0)])
    return bm


build_both_side_stroke = build_rev3_both_side_stroke


def build_curved_quad_single_candidate() -> bmesh.types.BMesh:
    """The accepted curved-quad case from test_interior_chain.py."""

    quad = (
        (-0.21765238046646118, -0.5814824104309082, -0.17536157369613647),
        (-0.2244318425655365, -0.5596591234207153, -0.18323862552642822),
        (-0.203125, -0.5625, -0.1875),
        (-0.19602273404598236, -0.5852272510528564, -0.1796875),
    )
    bm = bmesh.new()
    left = [bm.verts.new(co) for co in quad]
    right = [bm.verts.new((-x, y, z)) for x, y, z in reversed(quad)]
    bm.faces.new(left)
    bm.faces.new(right)
    bm.normal_update()
    _prepare_before_native_cut(bm)
    host = next(face for face in bm.faces if all(vertex.co.x < 0.0 for vertex in face.verts))
    verts = list(host.verts)
    edge = next(edge for edge in host.edges if verts[0] in edge.verts and verts[1] in edge.verts)
    _new_edge, boundary_vertex = bmesh.utils.edge_split(edge, edge.verts[0], 0.43)
    interior = verts[0].co * 0.4 + verts[1].co * 0.35 + verts[2].co * 0.25
    corners = [vertex for vertex in verts if vertex not in edge.verts]
    far_corner = max(corners, key=lambda vertex: (vertex.co - boundary_vertex.co).length)
    import mathutils.geometry as geometry

    pentagon = next(face for face in bm.faces if boundary_vertex in face.verts and far_corner in face.verts)
    assert len(pentagon.verts) == 5
    source_distance = min(
        (geometry.closest_point_on_tri(interior, a, b, c) - interior).length
        for a, b, c in _face_surface_triangles_for_fixture(pentagon)
    )
    mirrored_loop = [Vector((-vertex.co.x, vertex.co.y, vertex.co.z)) for vertex in reversed(list(pentagon.verts))]
    mirrored_interior = Vector((-interior.x, interior.y, interior.z))
    target_distance = min(
        (
            geometry.closest_point_on_tri(mirrored_interior, mirrored_loop[a], mirrored_loop[b], mirrored_loop[c])
            - mirrored_interior
        ).length
        for a, b, c in geometry.tessellate_polygon([mirrored_loop])
    )
    assert source_distance > 2.0 * TOLERANCE
    assert target_distance > 2.0 * TOLERANCE
    bmesh.utils.face_split(pentagon, boundary_vertex, far_corner, coords=[tuple(interior)])
    return bm


def _face_surface_triangles_for_fixture(face):
    import mathutils.geometry as geometry

    coordinates = [vertex.co for vertex in face.verts]
    yield from (
        (coordinates[a], coordinates[b], coordinates[c]) for a, b, c in geometry.tessellate_polygon([coordinates])
    )


def build_two_chains_same_face() -> bmesh.types.BMesh:
    """Two independent chains sharing one ancestor face, as in rev3 tests."""

    bm = _base_pair()
    _prepare_before_native_cut(bm)
    vb1 = _split_boundary(bm, y=-1.0, x=-1.8)
    bottom_rest = next(
        edge
        for edge in vb1.link_edges
        if all(abs(float(vertex.co.y) + 1.0) < 1.0e-8 for vertex in edge.verts)
        and max(float(vertex.co.x) for vertex in edge.verts) > -1.5
    )
    _new_edge, vb2 = bmesh.utils.edge_split(
        bottom_rest,
        vb1,
        (-1.2 - float(vb1.co.x)) / (float(bottom_rest.other_vert(vb1).co.x) - float(vb1.co.x)),
    )
    vb2.co = (-1.2, -1.0, 0.0)
    corner_tl = next(vertex for vertex in bm.verts if tuple(vertex.co) == (-2.0, 1.0, 0.0))
    corner_tr = next(vertex for vertex in bm.verts if tuple(vertex.co) == (-1.0, 1.0, 0.0))
    host_a = next(face for face in bm.faces if vb1 in face.verts and corner_tl in face.verts)
    bmesh.utils.face_split(host_a, vb1, corner_tl, coords=[(-1.85, 0.0, 0.0)])
    host_b = next(face for face in bm.faces if face.is_valid and vb2 in face.verts and corner_tr in face.verts)
    bmesh.utils.face_split(host_b, vb2, corner_tr, coords=[(-1.15, 0.0, 0.0)])
    return bm


def _nonplanar_pair(vertices: tuple[tuple[float, float, float], ...]) -> bmesh.types.BMesh:
    bm = bmesh.new()
    bm.faces.new([bm.verts.new(co) for co in vertices])
    bm.faces.new([bm.verts.new((-x, y, z)) for x, y, z in reversed(vertices)])
    bm.normal_update()
    return bm


def _surface_distance(point: Vector, face) -> float:
    import mathutils.geometry

    distances = []
    coordinates = [vertex.co for vertex in face.verts]
    for a, b, c in mathutils.geometry.tessellate_polygon([coordinates]):
        closest = mathutils.geometry.closest_point_on_tri(point, coordinates[a], coordinates[b], coordinates[c])
        distances.append(float((closest - point).length))
    return min(distances)


def _snapshot_face_deviation(bm, topology, face) -> float:
    face_layer = bm.faces.layers.int.get(layer_names.FACE_ID_LAYER)
    assert face_layer is not None
    frame = topology.carrier_frames.get(FaceId(int(face[face_layer])))
    assert frame is not None and frame.normal is not None
    return float(frame.deviation)


def _assert_surface_band_premise(
    source_distance: float,
    target_distance: float,
    source_deviation: float,
    target_deviation: float,
    tol: float,
) -> None:
    # BMesh coordinates are float32; on-surface placement leaves ~2^-24
    # residual, so the bound must sit above 6e-8 while staying far below
    # the 2*tol strict-containment limit the band relies on.
    assert source_distance <= 1.0e-6, source_distance
    assert 0.5 * tol <= source_deviation <= 10.0 * tol, source_deviation
    assert 0.5 * tol <= target_deviation <= 10.0 * tol, target_deviation
    effective = max(20.0 * tol, 2.5 * max(source_deviation, target_deviation))
    assert 2.0 * tol < target_distance <= effective, (target_distance, effective)
    assert target_distance <= 2.5 * max(source_deviation, target_deviation), target_distance


def build_weak_surface_band() -> bmesh.types.BMesh:
    # The mesh is EXACTLY symmetric; the band comes from the tessellation
    # asymmetry alone. The mirror face is built with reversed vertex order
    # (consistent winding), which makes ear clipping pick different ears on
    # the two sides of this concave, weakly non-planar pentagon.
    z = 5.0e-5
    vertices = (
        (-2.0, -1.0, z * -1.0),
        (-1.0, -1.0, z * 0.8),
        (-1.2, 0.1, z * 0.9),
        (-1.0, 1.0, z * -0.7),
        (-2.0, 1.0, z * 0.3),
    )
    bm = _nonplanar_pair(vertices)
    _prepare_before_native_cut(bm)
    topology = _PREPARATIONS[id(bm)]
    bottom = _split_boundary(bm, y=-1.0, x=-1.5)
    top = _split_boundary(bm, y=1.0, x=-1.5)
    source = _host(bm, bottom, top)
    target = next(face for face in bm.faces if all(float(vertex.co.x) > 0.0 for vertex in face.verts))
    source_deviation = _snapshot_face_deviation(bm, topology, source)
    target_deviation = _snapshot_face_deviation(bm, topology, target)
    selected = None
    for triangle in _face_surface_triangles_for_fixture(source):
        candidate = (triangle[0] * 0.25) + (triangle[1] * 0.35) + (triangle[2] * 0.40)
        mirrored = Vector((-candidate.x, candidate.y, candidate.z))
        source_distance = _surface_distance(candidate, source)
        target_distance = _surface_distance(mirrored, target)
        if source_distance <= 1.0e-6 and 2.0 * TOLERANCE < target_distance:
            selected = (candidate, source_distance, target_distance)
            break
    assert selected is not None, "no source render triangle produces the weak band"
    point, source_distance, target_distance = selected
    _assert_surface_band_premise(source_distance, target_distance, source_deviation, target_deviation, TOLERANCE)
    bmesh.utils.face_split(_host(bm, bottom, top), bottom, top, coords=[tuple(point)])
    return bm


build_weakly_curved_band = build_weak_surface_band
build_weak_curved_band = build_weak_surface_band


def _path_interior_vertex(bm: bmesh.types.BMesh):
    """Return the sole degree-two vertex of the native path in *bm*."""

    source_edges = _source_edges(bm)
    adjacency: dict[int, set[int]] = {}
    vertices = {}
    for edge in source_edges:
        left, right = edge.verts
        left_key, right_key = hash(left), hash(right)
        vertices[left_key] = left
        vertices[right_key] = right
        adjacency.setdefault(left_key, set()).add(right_key)
        adjacency.setdefault(right_key, set()).add(left_key)
    interiors = [vertices[key] for key, neighbours in adjacency.items() if len(neighbours) == 2]
    assert len(interiors) == 1, len(interiors)
    return interiors[0]


def build_sanity_excess_band() -> bmesh.types.BMesh:
    """Weak-band path moved off the carrier beyond the contract S_eff bound."""

    bm = build_weak_surface_band()
    interior = _path_interior_vertex(bm)
    # The carrier frame was captured before this native path vertex existed;
    # moving the live vertex therefore cannot enlarge the ancestor deviation.
    interior.co.z += 1.0e-2
    bm.normal_update()
    return bm


def build_projection_outside_band() -> bmesh.types.BMesh:
    """Weak-band path whose reflected point is outside the carrier polygon."""

    bm = build_weak_surface_band()
    interior = _path_interior_vertex(bm)
    # The target carrier's right-most boundary is x=2.  Keep the point close
    # enough for S_eff while moving its carrier-plane projection outside.
    interior.co.x = -2.0 - 8.0e-5
    bm.normal_update()
    return bm


def build_nonplanar_pentagon_pair() -> bmesh.types.BMesh:
    """Generic fallback geometry for suzanne_falied_a face 308/309."""

    # The deterministic weak-band analogue exercises the same carrier-deviation
    # contract without depending on an external .blend capture.
    return build_weak_surface_band()


def build_nonplanar_quad_pair() -> bmesh.types.BMesh:
    """Symmetric weakly non-planar quad (suzanne_falied_a 296/534 analogue).

    Rev3 accepts this: the strict interior test offers BOTH quad diagonals,
    so a point on the source render diagonal is always within tolerance of
    one target diagonal. It therefore belongs to the equivalence corpus.
    """

    # Deterministic synthetic coordinates retain the weak non-planarity and
    # mirrored winding needed by the differential corpus.
    vertices = (
        (-2.0, -1.0, 0.0),
        (-1.0, -1.0, 0.000015),
        (-1.0, 1.0, -0.000016),
        (-2.0, 1.0, 0.000006),
    )
    bm = _nonplanar_pair(vertices)
    _prepare_before_native_cut(bm)
    bottom = _split_boundary(bm, y=-1.0, x=-1.5)
    top = _split_boundary(bm, y=1.0, x=-1.5)
    source_face = _host(bm, bottom, top)
    source_triangle = next(iter(_face_surface_triangles_for_fixture(source_face)))
    point = source_triangle[0] * 0.20 + source_triangle[1] * 0.30 + source_triangle[2] * 0.50
    assert _surface_distance(point, source_face) <= 1.0e-6
    bmesh.utils.face_split(source_face, bottom, top, coords=[tuple(point)])
    return bm


def build_suzanne_falied_a() -> bmesh.types.BMesh:
    return build_nonplanar_pentagon_pair()


build_suzanne_failed_a = build_suzanne_falied_a


def build_endpoint_collision() -> bmesh.types.BMesh:
    """An interior chain whose endpoints collapse on a pre-split target.

    The target carrier is split once before apply, leaving two live instances
    with the same FACE_ID.  Both reflected chain endpoints intentionally
    resolve exactly to the shared target vertex; realization must still reach
    its endpoint/lineage checks instead of taking a unique-instance shortcut.
    """

    bm = _base_pair()
    _prepare_before_native_cut(bm)
    target_vertex = _split_side_boundary(bm, y=-1.0, x=1.5, positive=True)
    end_a = _split_boundary(bm, y=-1.0, x=-1.5)
    middle = _split_boundary(bm, y=-1.0, x=-1.5 + 0.4 * TOLERANCE)
    end_b = _split_boundary(bm, y=-1.0, x=-1.5 + 0.8 * TOLERANCE)
    # Adjacent face vertices are rejected by BMesh face_split; retain a
    # boundary vertex between the two endpoint candidates from one ancestor edge.
    assert bm.edges.get((end_a, middle)) is not None
    assert bm.edges.get((middle, end_b)) is not None
    bmesh.utils.face_split(_host(bm, end_a, end_b), end_a, end_b, coords=[(-1.3, -0.5, 0.0)])
    assert (end_b.co - end_a.co).length <= TOLERANCE
    assert abs(float(end_a.co.y) + 1.0) <= TOLERANCE
    assert abs(float(end_b.co.y) + 1.0) <= TOLERANCE
    reflected_a = Vector((-float(end_a.co.x), float(end_a.co.y), float(end_a.co.z)))
    reflected_b = Vector((-float(end_b.co.x), float(end_b.co.y), float(end_b.co.z)))
    assert matching.coordinates_match(reflected_a, target_vertex.co, TOLERANCE)
    assert matching.coordinates_match(reflected_b, target_vertex.co, TOLERANCE)
    target_top = _split_side_boundary(bm, y=1.0, x=1.5, positive=True)
    target_face = next(face for face in bm.faces if target_vertex in face.verts and target_top in face.verts)
    bmesh.utils.face_split(target_face, target_vertex, target_top, coords=[(1.3, 0.0, 0.0)])
    face_layer = bm.faces.layers.int.get(layer_names.FACE_ID_LAYER)
    assert face_layer is not None
    live_instances = [face for face in bm.faces if int(face[face_layer]) == int(target_face[face_layer])]
    assert len(live_instances) == 2, len(live_instances)
    return bm


def build_lineage_mismatch_chain() -> bmesh.types.BMesh:
    """Two-member chain whose pre-split target members resolve to lineages.

    The diagonal target split separates the two reflected interior points while
    preserving one FACE_ID.  R-H5 must let the chain reach realization; R-H4
    then declines when its classification lineage set is not singleton.
    """

    bm = _base_pair()
    _prepare_before_native_cut(bm)
    bottom = _split_boundary(bm, y=-1.0, x=-1.5)
    top = _split_boundary(bm, y=1.0, x=-1.5)
    source_face = _host(bm, bottom, top)
    bmesh.utils.face_split(
        source_face,
        bottom,
        top,
        coords=[(-1.4, -0.25, 0.0), (-1.2, 0.25, 0.0)],
    )

    target_face = next(face for face in bm.faces if all(float(vertex.co.x) > 0.0 for vertex in face.verts))
    target_bottom_left = next(vertex for vertex in target_face.verts if tuple(vertex.co) == (1.0, -1.0, 0.0))
    target_top_right = next(vertex for vertex in target_face.verts if tuple(vertex.co) == (2.0, 1.0, 0.0))
    bmesh.utils.face_split(target_face, target_bottom_left, target_top_right)
    face_layer = bm.faces.layers.int.get(layer_names.FACE_ID_LAYER)
    assert face_layer is not None
    live_instances = [face for face in bm.faces if int(face[face_layer]) == int(target_face[face_layer])]
    assert len(live_instances) == 2, len(live_instances)
    return bm


def build_endpoint_link_mismatch() -> bmesh.types.BMesh:
    """One interior chain whose geometric target winner misses one endpoint."""

    bm = _base_pair()
    _prepare_before_native_cut(bm)
    bottom = _split_boundary(bm, y=-1.0, x=-1.5)
    top = _split_boundary(bm, y=1.0, x=-1.5)
    bmesh.utils.face_split(_host(bm, bottom, top), bottom, top, coords=[(-1.2, 0.0, 0.0)])

    target_face = next(face for face in bm.faces if all(float(vertex.co.x) > 0.0 for vertex in face.verts))
    target_bottom_left = next(vertex for vertex in target_face.verts if tuple(vertex.co) == (1.0, -1.0, 0.0))
    target_top_right = next(vertex for vertex in target_face.verts if tuple(vertex.co) == (2.0, 1.0, 0.0))
    bmesh.utils.face_split(target_face, target_bottom_left, target_top_right)
    face_layer = bm.faces.layers.int.get(layer_names.FACE_ID_LAYER)
    assert face_layer is not None
    live_instances = [face for face in bm.faces if int(face[face_layer]) == int(target_face[face_layer])]
    assert len(live_instances) == 2, len(live_instances)
    return bm


def build_wire_mixed() -> bmesh.types.BMesh:
    """O14: an interior chain plus two dangling wire strokes (R-W1)."""

    bm = _base_pair()
    _prepare_before_native_cut(bm)
    bottom = _split_boundary(bm, y=-1.0, x=-1.5)
    top = _split_boundary(bm, y=1.0, x=-1.5)
    bmesh.utils.face_split(_host(bm, bottom, top), bottom, top, coords=[(-1.2, 0.0, 0.0)])
    corner = next(v for v in bm.verts if (v.co - Vector((-2.0, -1.0, 0.0))).length <= 1e-6)
    dangling = bm.verts.new((-1.7, -0.5, 0.2))
    bm.edges.new((corner, dangling))
    free_a = bm.verts.new((-1.8, 0.6, 0.1))
    free_b = bm.verts.new((-1.6, 0.8, 0.1))
    bm.edges.new((free_a, free_b))
    return bm


def build_wire_ambiguous() -> bmesh.types.BMesh:
    """O14: a wire endpoint whose reflection has two tol-close vertices."""

    bm = _base_pair()
    _prepare_before_native_cut(bm)
    # Two loose vertices straddling the reflected endpoint within tolerance.
    bm.verts.new((1.7, 0.5, 0.4 * TOLERANCE))
    bm.verts.new((1.7, 0.5, -0.4 * TOLERANCE))
    corner = next(v for v in bm.verts if (v.co - Vector((-2.0, -1.0, 0.0))).length <= 1e-6)
    dangling = bm.verts.new((-1.7, 0.5, 0.0))
    bm.edges.new((corner, dangling))
    return bm


def _source_edges(bm: bmesh.types.BMesh) -> list[bmesh.types.BMEdge]:
    source, side, total, crossing = stitch_pathedges.collect_source_path_edges(bm, AXIS, TOLERANCE, "NEGATIVE")
    assert side == "NEGATIVE", (side, total, crossing)
    assert crossing == 0
    assert source
    return source


def fixture_from_builder(name: str, builder: Callable[[], bmesh.types.BMesh]) -> InteriorHostFixture:
    bm = builder()
    topology = _PREPARATIONS.pop(id(bm), None)
    assert topology is not None, name
    source_edges = list(_EXPLICIT_SOURCE_EDGES.pop(id(bm), ())) or _source_edges(bm)
    if name == "curved_quad_single_candidate":
        assert len(source_edges) == 2
    if name == "two_chains_same_face":
        assert len(source_edges) == 4
    if name == "endpoint_collision":
        assert len(source_edges) >= 2
        assert len({hash(vertex) for edge in source_edges for vertex in edge.verts}) >= 3
    if name == "lineage_mismatch_chain":
        assert len(source_edges) == 3
    if name == "endpoint_link_mismatch":
        assert len(source_edges) == 2
    return InteriorHostFixture(name, bm, source_edges, topology)


GOLDEN_BUILDERS: tuple[tuple[str, Callable[[], bmesh.types.BMesh]], ...] = (
    ("interior_chain_n1", build_interior_chain_n1),
    ("interior_chain_n2", build_interior_chain_n2),
    ("curved_quad_single_candidate", build_curved_quad_single_candidate),
    ("two_chains_same_face", build_two_chains_same_face),
    ("plane_two_chains", build_plane_two_chains),
    ("rev3_both_side_stroke", build_rev3_both_side_stroke),
    ("nonplanar_quad_pair", build_nonplanar_quad_pair),
)


NETWORK_BUILDERS: tuple[tuple[str, Callable[[], bmesh.types.BMesh]], ...] = (
    ("network_y", build_interior_network_y),
    ("network_x", build_interior_network_x),
    ("network_branch_relay", build_interior_network_branch_relay),
    ("network_theta", build_interior_network_theta),
)

NETWORK_DECLINE_BUILDERS: tuple[tuple[str, Callable[[], bmesh.types.BMesh]], ...] = (
    ("network_decline_loop", build_interior_network_decline_loop),
    ("network_decline_one_anchor", build_interior_network_decline_one_anchor),
    ("network_decline_dead_end", build_interior_network_decline_dead_end),
)


# rev3 declines these at the gate (rev4/v5 must direct-apply them, O13) —
# except endpoint_collision, which passes the gate on both revisions and
# fails in apply (R-R1 projection-retry material, O11).
NON_EQUIVALENCE_BUILDERS: tuple[tuple[str, Callable[[], bmesh.types.BMesh]], ...] = (
    ("weak_surface_band", build_weak_surface_band),
    ("suzanne_falied_a_pentagon", build_nonplanar_pentagon_pair),
    ("endpoint_collision", build_endpoint_collision),
)


def golden_fixtures() -> list[InteriorHostFixture]:
    return [fixture_from_builder(name, builder) for name, builder in GOLDEN_BUILDERS]


def network_fixtures() -> list[InteriorHostFixture]:
    return [fixture_from_builder(name, builder) for name, builder in NETWORK_BUILDERS]


def non_equivalence_fixtures() -> list[InteriorHostFixture]:
    return [fixture_from_builder(name, builder) for name, builder in NON_EQUIVALENCE_BUILDERS]


if __name__ == "__main__":
    raise SystemExit("Run fixture builders from Blender; this module does not execute bpy/bmesh checks standalone.")
