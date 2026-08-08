# SPDX-License-Identifier: GPL-3.0-or-later

"""Phase 1: Knife both-sides mirror (axis-crossing contract §4.1).

Cases:
  (a) Stroke straddling the mirror plane via an on-plane vertex (no CROSSES)
      → full X-symmetry after finish; fixed topology counts; face incidence.
  (b) Stroke with a plane-crossing segment → native kept + WARNING, no mirror.
  (c) One-sided stroke (regression) → opposite side still mirrored.
  (d) CROSSES mixed with POSITIVE/NEGATIVE → whole-stage decline + WARNING.
  (e) Knife Project fallback (bent polyline, non-boundary) → both-sides mirror.
  (f) Partial mirror failure (monkeypatch) → full rollback + WARNING + FINISHED.
  (g) Near-self-mirrored stroke → already_present, no double cut.

Run::

    blender --factory-startup --enable-event-simulate \\
        --python test_knife_both_sides.py
"""

from __future__ import annotations

import os
import sys
import traceback
from collections import Counter
from pathlib import Path

import bmesh
import bpy
from mathutils import Quaternion

PACKAGE_PARENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_PARENT))

import ydd_symmetric_edit as addon  # noqa: E402
from ydd_symmetric_edit import core, operators  # noqa: E402

MARKER_OK = "YSE_KNIFE_BOTH_SIDES_OK"
MARKER_FAILED = "YSE_KNIFE_BOTH_SIDES_FAILED"
COORD_PRECISION = 5
CROSS_PLANE_WARNING = "cross-plane knife segments are not mirrored yet"


def fail(message: str = "") -> None:
    if message:
        print(f"YSE_KNIFE_BOTH_SIDES_ERROR={message}", flush=True)
    traceback.print_exc()
    print(MARKER_FAILED, flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(1)


def viewport_context():
    window = bpy.context.window_manager.windows[0]
    area = next(area for area in window.screen.areas if area.type == "VIEW_3D")
    region = next(region for region in area.regions if region.type == "WINDOW")
    return window, area, region


def configure_view(area) -> None:
    region_3d = area.spaces.active.region_3d
    region_3d.view_perspective = "ORTHO"
    region_3d.view_rotation = Quaternion((1.0, 0.0, 0.0, 0.0))
    region_3d.view_location = (0.0, 0.0, 0.0)
    region_3d.view_distance = 6.0
    region_3d.update()


def coordinate_key(coordinate, precision: int = COORD_PRECISION):
    return tuple(round(float(value), precision) for value in coordinate)


def mirror_key(coordinate, precision: int = COORD_PRECISION):
    x, y, z = coordinate
    return coordinate_key((-float(x), float(y), float(z)), precision)


def vertex_coord_multiset(bm, precision: int = COORD_PRECISION) -> Counter:
    return Counter(coordinate_key(vertex.co, precision) for vertex in bm.verts)


def edge_coord_multiset(bm, precision: int = COORD_PRECISION) -> Counter:
    return Counter(tuple(sorted(coordinate_key(vertex.co, precision) for vertex in edge.verts)) for edge in bm.edges)


def mirrored_vertex_multiset(bm, precision: int = COORD_PRECISION) -> Counter:
    return Counter(mirror_key(vertex.co, precision) for vertex in bm.verts)


def mirrored_edge_multiset(bm, precision: int = COORD_PRECISION) -> Counter:
    return Counter(tuple(sorted(mirror_key(vertex.co, precision) for vertex in edge.verts)) for edge in bm.edges)


def _normalize_cycle(coords: list[tuple[float, float, float]]) -> tuple[tuple[float, float, float], ...]:
    """Canonical rotation of a face cycle (lowest lexicographic start)."""

    if not coords:
        return ()
    rotations = [tuple(coords[i:] + coords[:i]) for i in range(len(coords))]
    return min(rotations)


def face_incidence_multiset(bm, precision: int = COORD_PRECISION) -> Counter:
    """Face multiset keyed by normalized vertex-cycle incidence (§1.2)."""

    keys = []
    for face in bm.faces:
        coords = [coordinate_key(vertex.co, precision) for vertex in face.verts]
        keys.append(_normalize_cycle(coords))
    return Counter(keys)


def mirrored_face_incidence_multiset(bm, precision: int = COORD_PRECISION) -> Counter:
    """Mirror each face cycle and reverse winding (mirror flips orientation)."""

    keys = []
    for face in bm.faces:
        mirrored = [mirror_key(vertex.co, precision) for vertex in face.verts]
        mirrored.reverse()
        keys.append(_normalize_cycle(mirrored))
    return Counter(keys)


def assert_no_duplicate_edges(bm, tolerance: float = 1.0e-7) -> None:
    """No two edges share the same endpoint pair within *tolerance*."""

    seen: list[tuple[tuple[float, float, float], tuple[float, float, float]]] = []
    for edge in bm.edges:
        a = coordinate_key(edge.verts[0].co, 7)
        b = coordinate_key(edge.verts[1].co, 7)
        key = (a, b) if a <= b else (b, a)
        for other in seen:
            if all(abs(key[0][i] - other[0][i]) <= tolerance for i in range(3)) and all(
                abs(key[1][i] - other[1][i]) <= tolerance for i in range(3)
            ):
                raise AssertionError(f"duplicate edge endpoints {key} ~ {other}")
        seen.append(key)


def assert_x_symmetric(bm, tolerance: float = 1.0e-4) -> None:
    """Every vertex has a mirror partner; coords, edges, and face incidence match."""

    for vertex in bm.verts:
        expected = core.mirror_coordinate(vertex.co, core.AXIS_INDEX["X"])
        assert any(core.coordinates_match(other.co, expected, tolerance) for other in bm.verts), (
            f"vertex {tuple(vertex.co)} has no X-mirror within {tolerance}"
        )
    verts = vertex_coord_multiset(bm)
    edges = edge_coord_multiset(bm)
    assert verts == mirrored_vertex_multiset(bm), f"vertex coords not X-symmetric: {verts}"
    assert edges == mirrored_edge_multiset(bm), f"edges not X-symmetric: {edges}"
    faces = face_incidence_multiset(bm)
    mirrored_faces = mirrored_face_incidence_multiset(bm)
    assert faces == mirrored_faces, f"face incidence not X-symmetric: {faces} vs {mirrored_faces}"


def has_exact_edge(bm, a, b, tolerance=1.0e-7) -> bool:
    def close(co, expected):
        return all(abs(co[index] - expected[index]) <= tolerance for index in range(3))

    return any(
        (close(edge.verts[0].co, a) and close(edge.verts[1].co, b))
        or (close(edge.verts[0].co, b) and close(edge.verts[1].co, a))
        for edge in bm.edges
    )


def clear_scene() -> None:
    for old in tuple(bpy.data.objects):
        bpy.data.objects.remove(old, do_unlink=True)


def make_straddling_grid():
    """Two quads sharing the X=0 plane edge (continuous across the axis)."""

    mesh = bpy.data.meshes.new("YSE_KnifeBothSidesMesh")
    vertices = [
        (-2.0, -1.0, 0.0),
        (0.0, -1.0, 0.0),
        (2.0, -1.0, 0.0),
        (-2.0, 1.0, 0.0),
        (0.0, 1.0, 0.0),
        (2.0, 1.0, 0.0),
    ]
    mesh.from_pydata(vertices, [], [(0, 1, 4, 3), (1, 2, 5, 4)])
    mesh.update()
    obj = bpy.data.objects.new("YSE_KnifeBothSidesObject", mesh)
    bpy.context.scene.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    obj.use_mesh_mirror_x = True
    obj.use_mesh_mirror_y = False
    obj.use_mesh_mirror_z = False
    return obj


def make_two_quads():
    """Disconnected symmetric quads (one-sided regression fixture)."""

    mesh = bpy.data.meshes.new("YSE_KnifeOneSideMesh")
    vertices = [
        (-2.0, -1.0, 0.0),
        (-1.0, -1.0, 0.0),
        (-1.0, 1.0, 0.0),
        (-2.0, 1.0, 0.0),
        (1.0, -1.0, 0.0),
        (2.0, -1.0, 0.0),
        (2.0, 1.0, 0.0),
        (1.0, 1.0, 0.0),
    ]
    mesh.from_pydata(vertices, [], [(0, 1, 2, 3), (4, 5, 6, 7)])
    mesh.update()
    obj = bpy.data.objects.new("YSE_KnifeOneSideObject", mesh)
    bpy.context.scene.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    obj.use_mesh_mirror_x = True
    return obj


def warning_messages() -> list[str]:
    return [message for kind, message in operators._FINISH_REPORTS if kind == "WARNING"]


def prepare_knife_session(context) -> None:
    prepared = operators._prepare_session(
        context,
        lambda _level, _message: None,
        tool_kind="KNIFE",
    )
    assert prepared, "failed to prepare knife session"


def simulate_straddle_via_plane(obj) -> None:
    """Asymmetric path left→plane→right (no CROSSES). Old one-side mirror fails this."""

    bm = bmesh.from_edit_mesh(obj.data)
    plane_edge = next(
        edge
        for edge in bm.edges
        if all(abs(vertex.co.x) <= 1.0e-8 for vertex in edge.verts)
        and {round(vertex.co.y, 6) for vertex in edge.verts} == {-1.0, 1.0}
    )
    _edge, plane_mid = bmesh.utils.edge_split(plane_edge, plane_edge.verts[0], 0.5)
    plane_mid.co = (0.0, 0.0, 0.0)

    left_bottom = next(
        edge
        for edge in bm.edges
        if all(vertex.co.x <= 1.0e-8 for vertex in edge.verts)
        and all(abs(vertex.co.y + 1.0) <= 1.0e-8 for vertex in edge.verts)
        and min(vertex.co.x for vertex in edge.verts) < -1.0e-8
    )
    _edge, left_point = bmesh.utils.edge_split(left_bottom, left_bottom.verts[0], 0.5)
    # Asymmetric: left at x=-1.5, right at x=+1.0 so neither side is the
    # mirror of the other; both directions must be mirrored for symmetry.
    left_point.co = (-1.5, -1.0, 0.0)

    right_top = next(
        edge
        for edge in bm.edges
        if all(vertex.co.x >= -1.0e-8 for vertex in edge.verts)
        and all(abs(vertex.co.y - 1.0) <= 1.0e-8 for vertex in edge.verts)
        and max(vertex.co.x for vertex in edge.verts) > 1.0e-8
    )
    _edge, right_point = bmesh.utils.edge_split(right_top, right_top.verts[0], 0.5)
    right_point.co = (1.0, 1.0, 0.0)

    left_face = next(
        face
        for face in bm.faces
        if left_point in face.verts and plane_mid in face.verts and face.calc_center_median().x < 0.0
    )
    bmesh.utils.face_split(left_face, left_point, plane_mid)

    right_face = next(
        face
        for face in bm.faces
        if right_point in face.verts and plane_mid in face.verts and face.calc_center_median().x > 0.0
    )
    bmesh.utils.face_split(right_face, plane_mid, right_point)
    bmesh.update_edit_mesh(obj.data, loop_triangles=True, destructive=True)


def simulate_cross_plane_segment(obj) -> tuple[int, int, int]:
    """One edge that crosses X=0 (CROSSES). Returns post-native topology counts."""

    bm = bmesh.from_edit_mesh(obj.data)
    left_bottom = next(
        edge
        for edge in bm.edges
        if all(vertex.co.x <= 1.0e-8 for vertex in edge.verts)
        and all(abs(vertex.co.y + 1.0) <= 1.0e-8 for vertex in edge.verts)
        and min(vertex.co.x for vertex in edge.verts) < -1.0e-8
    )
    right_top = next(
        edge
        for edge in bm.edges
        if all(vertex.co.x >= -1.0e-8 for vertex in edge.verts)
        and all(abs(vertex.co.y - 1.0) <= 1.0e-8 for vertex in edge.verts)
        and max(vertex.co.x for vertex in edge.verts) > 1.0e-8
    )
    _edge, left_pt = bmesh.utils.edge_split(left_bottom, left_bottom.verts[0], 0.25)
    left_pt.co = (-1.5, -1.0, 0.0)
    _edge, right_pt = bmesh.utils.edge_split(right_top, right_top.verts[0], 0.25)
    right_pt.co = (1.5, 1.0, 0.0)

    # Dissolve the shared plane edge so both quads become one face, then cut
    # a diagonal that crosses X=0.
    plane_edge = next(
        edge
        for edge in bm.edges
        if all(abs(vertex.co.x) <= 1.0e-8 for vertex in edge.verts)
        and {round(vertex.co.y, 6) for vertex in edge.verts} == {-1.0, 1.0}
    )
    bmesh.ops.dissolve_edges(bm, edges=[plane_edge], use_verts=False)
    host = next(face for face in bm.faces if left_pt in face.verts and right_pt in face.verts)
    bmesh.utils.face_split(host, left_pt, right_pt)
    counts = (len(bm.verts), len(bm.edges), len(bm.faces))
    bmesh.update_edit_mesh(obj.data, loop_triangles=True, destructive=True)
    return counts


def simulate_mixed_crosses(obj) -> tuple[int, int, int]:
    """POSITIVE + NEGATIVE + CROSSES segments in one stroke. Returns native counts."""

    bm = bmesh.from_edit_mesh(obj.data)
    left_bottom = next(
        edge
        for edge in bm.edges
        if all(vertex.co.x <= 1.0e-8 for vertex in edge.verts)
        and all(abs(vertex.co.y + 1.0) <= 1.0e-8 for vertex in edge.verts)
        and min(vertex.co.x for vertex in edge.verts) < -1.0e-8
    )
    right_top = next(
        edge
        for edge in bm.edges
        if all(vertex.co.x >= -1.0e-8 for vertex in edge.verts)
        and all(abs(vertex.co.y - 1.0) <= 1.0e-8 for vertex in edge.verts)
        and max(vertex.co.x for vertex in edge.verts) > 1.0e-8
    )
    _edge, left_pt = bmesh.utils.edge_split(left_bottom, left_bottom.verts[0], 0.25)
    left_pt.co = (-1.5, -1.0, 0.0)
    _edge, right_pt = bmesh.utils.edge_split(right_top, right_top.verts[0], 0.25)
    right_pt.co = (1.5, 1.0, 0.0)

    plane_edge = next(
        edge
        for edge in bm.edges
        if all(abs(vertex.co.x) <= 1.0e-8 for vertex in edge.verts)
        and {round(vertex.co.y, 6) for vertex in edge.verts} == {-1.0, 1.0}
    )
    bmesh.ops.dissolve_edges(bm, edges=[plane_edge], use_verts=False)
    host = next(face for face in bm.faces if left_pt in face.verts and right_pt in face.verts)
    # Polyline: NEGATIVE (-1.5,-1)→(-0.5,0), CROSSES (-0.5,0)→(0.5,0), POSITIVE (0.5,0)→(1.5,1)
    bmesh.utils.face_split(
        host,
        left_pt,
        right_pt,
        coords=[(-0.5, 0.0, 0.0), (0.5, 0.0, 0.0)],
    )
    counts = (len(bm.verts), len(bm.edges), len(bm.faces))
    bmesh.update_edit_mesh(obj.data, loop_triangles=True, destructive=True)
    return counts


def simulate_bent_both_sides(obj) -> None:
    """Asymmetric both-sides stroke with interior waypoints (Knife Project path)."""

    bm = bmesh.from_edit_mesh(obj.data)
    plane_edge = next(
        edge
        for edge in bm.edges
        if all(abs(vertex.co.x) <= 1.0e-8 for vertex in edge.verts)
        and {round(vertex.co.y, 6) for vertex in edge.verts} == {-1.0, 1.0}
    )
    _edge, plane_mid = bmesh.utils.edge_split(plane_edge, plane_edge.verts[0], 0.5)
    plane_mid.co = (0.0, 0.0, 0.0)

    left_bottom = next(
        edge
        for edge in bm.edges
        if all(vertex.co.x <= 1.0e-8 for vertex in edge.verts)
        and all(abs(vertex.co.y + 1.0) <= 1.0e-8 for vertex in edge.verts)
        and min(vertex.co.x for vertex in edge.verts) < -1.0e-8
    )
    _edge, left_point = bmesh.utils.edge_split(left_bottom, left_bottom.verts[0], 0.5)
    left_point.co = (-1.5, -1.0, 0.0)

    right_top = next(
        edge
        for edge in bm.edges
        if all(vertex.co.x >= -1.0e-8 for vertex in edge.verts)
        and all(abs(vertex.co.y - 1.0) <= 1.0e-8 for vertex in edge.verts)
        and max(vertex.co.x for vertex in edge.verts) > 1.0e-8
    )
    _edge, right_point = bmesh.utils.edge_split(right_top, right_top.verts[0], 0.5)
    right_point.co = (1.0, 1.0, 0.0)

    left_face = next(
        face
        for face in bm.faces
        if left_point in face.verts and plane_mid in face.verts and face.calc_center_median().x < 0.0
    )
    # Interior waypoint keeps the path off pure boundary endpoints.
    bmesh.utils.face_split(
        left_face,
        left_point,
        plane_mid,
        coords=[(-1.2, 0.0, 0.0)],
    )

    right_face = next(
        face
        for face in bm.faces
        if right_point in face.verts and plane_mid in face.verts and face.calc_center_median().x > 0.0
    )
    bmesh.utils.face_split(
        right_face,
        plane_mid,
        right_point,
        coords=[(0.7, 0.4, 0.0)],
    )
    bmesh.update_edit_mesh(obj.data, loop_triangles=True, destructive=True)


def simulate_one_side_cut(obj) -> None:
    bm = bmesh.from_edit_mesh(obj.data)
    bottom = next(
        edge
        for edge in bm.edges
        if all(vertex.co.x < 0.0 for vertex in edge.verts)
        and all(abs(vertex.co.y + 1.0) < 1.0e-8 for vertex in edge.verts)
    )
    _edge, bottom_vertex = bmesh.utils.edge_split(bottom, bottom.verts[0], 0.5)
    top = next(
        edge
        for edge in bm.edges
        if all(vertex.co.x < 0.0 for vertex in edge.verts)
        and all(abs(vertex.co.y - 1.0) < 1.0e-8 for vertex in edge.verts)
    )
    _edge, top_vertex = bmesh.utils.edge_split(top, top.verts[0], 0.5)
    source_face = next(face for face in bm.faces if bottom_vertex in face.verts and top_vertex in face.verts)
    bmesh.utils.face_split(source_face, bottom_vertex, top_vertex)
    bmesh.update_edit_mesh(obj.data, loop_triangles=True, destructive=True)


def simulate_self_mirrored_cut(obj) -> tuple[int, int, int]:
    """Symmetric cuts on both halves already present; mirror is already_present."""

    bm = bmesh.from_edit_mesh(obj.data)
    for x_sign in (-1.0, 1.0):
        bottom = next(
            edge
            for edge in bm.edges
            if all(vertex.co.x * x_sign > 0.0 for vertex in edge.verts)
            and all(abs(vertex.co.y + 1.0) < 1.0e-8 for vertex in edge.verts)
        )
        _edge, bottom_vertex = bmesh.utils.edge_split(bottom, bottom.verts[0], 0.5)
        bottom_vertex.co = (1.5 * x_sign, -1.0, 0.0)
        top = next(
            edge
            for edge in bm.edges
            if all(vertex.co.x * x_sign > 0.0 for vertex in edge.verts)
            and all(abs(vertex.co.y - 1.0) < 1.0e-8 for vertex in edge.verts)
        )
        _edge, top_vertex = bmesh.utils.edge_split(top, top.verts[0], 0.5)
        top_vertex.co = (1.5 * x_sign, 1.0, 0.0)
        source_face = next(
            face
            for face in bm.faces
            if bottom_vertex in face.verts and top_vertex in face.verts and face.calc_center_median().x * x_sign > 0.0
        )
        bmesh.utils.face_split(source_face, bottom_vertex, top_vertex)
    counts = (len(bm.verts), len(bm.edges), len(bm.faces))
    bmesh.update_edit_mesh(obj.data, loop_triangles=True, destructive=True)
    return counts


def case_a_straddle(window, area, region) -> None:
    print("YSE_KNIFE_BOTH_SIDES_CASE=a_straddle", flush=True)
    clear_scene()
    obj = make_straddling_grid()
    with bpy.context.temp_override(window=window, area=area, region=region):
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.context.tool_settings.mesh_select_mode = (True, False, False)
        prepare_knife_session(bpy.context)
        simulate_straddle_via_plane(obj)

        bm = bmesh.from_edit_mesh(obj.data)
        by_side, total = core.collect_knife_path_edges_by_side(
            bm,
            core.AXIS_INDEX["X"],
            1.0e-4,
        )
        assert total >= 2, (total, {key: len(value) for key, value in by_side.items()})
        assert not by_side["CROSSES"], "fixture must not introduce CROSSES"
        assert by_side["POSITIVE"] and by_side["NEGATIVE"], {key: len(value) for key, value in by_side.items()}

        finished = bpy.ops.mesh.ydd_symmetric_edit_finish("EXEC_DEFAULT")
        assert finished == {"FINISHED"}, finished

    bm = bmesh.from_edit_mesh(obj.data)
    # Native segments.
    assert has_exact_edge(bm, (-1.5, -1.0, 0.0), (0.0, 0.0, 0.0))
    assert has_exact_edge(bm, (0.0, 0.0, 0.0), (1.0, 1.0, 0.0))
    # Mirrored counterparts (the Phase 1 both-sides requirement).
    assert has_exact_edge(bm, (1.5, -1.0, 0.0), (0.0, 0.0, 0.0))
    assert has_exact_edge(bm, (0.0, 0.0, 0.0), (-1.0, 1.0, 0.0))
    # Fixed topology: post-native 9V/12E/4F + 2 mirrored boundary splits + 2 face splits.
    assert (len(bm.verts), len(bm.edges), len(bm.faces)) == (11, 16, 6), (
        len(bm.verts),
        len(bm.edges),
        len(bm.faces),
    )
    assert_no_duplicate_edges(bm)
    assert_x_symmetric(bm)
    assert bm.edges.layers.int.get(core.EDGE_ORIGINAL_LAYER) is None
    assert not operators._SESSIONS
    print("YSE_KNIFE_BOTH_SIDES_CASE_A=OK", flush=True)


def case_b_crosses(window, area, region) -> None:
    print("YSE_KNIFE_BOTH_SIDES_CASE=b_crosses", flush=True)
    clear_scene()
    obj = make_straddling_grid()
    with bpy.context.temp_override(window=window, area=area, region=region):
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.context.tool_settings.mesh_select_mode = (True, False, False)
        prepare_knife_session(bpy.context)
        native_counts = simulate_cross_plane_segment(obj)

        bm = bmesh.from_edit_mesh(obj.data)
        by_side, total = core.collect_knife_path_edges_by_side(
            bm,
            core.AXIS_INDEX["X"],
            1.0e-4,
        )
        assert total >= 1, total
        assert by_side["CROSSES"], "fixture must include CROSSES edges"

        finished = bpy.ops.mesh.ydd_symmetric_edit_finish("EXEC_DEFAULT")
        assert finished == {"FINISHED"}, finished

    bm = bmesh.from_edit_mesh(obj.data)
    assert (len(bm.verts), len(bm.edges), len(bm.faces)) == native_counts, (
        (len(bm.verts), len(bm.edges), len(bm.faces)),
        native_counts,
    )
    assert has_exact_edge(bm, (-1.5, -1.0, 0.0), (1.5, 1.0, 0.0))
    warnings = warning_messages()
    assert any(CROSS_PLANE_WARNING in message for message in warnings), (
        warnings,
        operators._FINISH_REPORTS,
    )
    # Blender's Operator.report also emits "Warning: ..." on stdout for runners.
    assert bm.edges.layers.int.get(core.EDGE_ORIGINAL_LAYER) is None
    assert not operators._SESSIONS
    print("YSE_KNIFE_BOTH_SIDES_CASE_B=OK", flush=True)


def case_c_one_side(window, area, region) -> None:
    print("YSE_KNIFE_BOTH_SIDES_CASE=c_one_side", flush=True)
    clear_scene()
    obj = make_two_quads()
    with bpy.context.temp_override(window=window, area=area, region=region):
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.context.tool_settings.mesh_select_mode = (True, False, False)
        prepare_knife_session(bpy.context)
        simulate_one_side_cut(obj)
        finished = bpy.ops.mesh.ydd_symmetric_edit_finish("EXEC_DEFAULT")
        assert finished == {"FINISHED"}, finished

    bm = bmesh.from_edit_mesh(obj.data)
    assert has_exact_edge(bm, (-1.5, -1.0, 0.0), (-1.5, 1.0, 0.0))
    assert has_exact_edge(bm, (1.5, -1.0, 0.0), (1.5, 1.0, 0.0))
    assert (len(bm.verts), len(bm.edges), len(bm.faces)) == (12, 14, 4)
    assert_x_symmetric(bm)
    assert bm.edges.layers.int.get(core.EDGE_ORIGINAL_LAYER) is None
    assert not operators._SESSIONS
    print("YSE_KNIFE_BOTH_SIDES_CASE_C=OK", flush=True)


def case_d_mixed_crosses(window, area, region) -> None:
    print("YSE_KNIFE_BOTH_SIDES_CASE=d_mixed_crosses", flush=True)
    clear_scene()
    obj = make_straddling_grid()
    with bpy.context.temp_override(window=window, area=area, region=region):
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.context.tool_settings.mesh_select_mode = (True, False, False)
        prepare_knife_session(bpy.context)
        native_counts = simulate_mixed_crosses(obj)

        bm = bmesh.from_edit_mesh(obj.data)
        by_side, total = core.collect_knife_path_edges_by_side(
            bm,
            core.AXIS_INDEX["X"],
            1.0e-4,
        )
        assert total >= 3, (total, {key: len(value) for key, value in by_side.items()})
        assert by_side["CROSSES"], "fixture must include CROSSES"
        assert by_side["POSITIVE"] or by_side["NEGATIVE"], {key: len(value) for key, value in by_side.items()}

        finished = bpy.ops.mesh.ydd_symmetric_edit_finish("EXEC_DEFAULT")
        assert finished == {"FINISHED"}, finished

    bm = bmesh.from_edit_mesh(obj.data)
    assert (len(bm.verts), len(bm.edges), len(bm.faces)) == native_counts, (
        (len(bm.verts), len(bm.edges), len(bm.faces)),
        native_counts,
    )
    warnings = warning_messages()
    assert any(CROSS_PLANE_WARNING in message for message in warnings), (
        warnings,
        operators._FINISH_REPORTS,
    )
    assert bm.edges.layers.int.get(core.EDGE_ORIGINAL_LAYER) is None
    assert not operators._SESSIONS
    print("YSE_KNIFE_BOTH_SIDES_CASE_D=OK", flush=True)


def case_e_knife_project_fallback(window, area, region) -> None:
    print("YSE_KNIFE_BOTH_SIDES_CASE=e_knife_project", flush=True)
    clear_scene()
    obj = make_straddling_grid()
    with bpy.context.temp_override(window=window, area=area, region=region):
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.context.tool_settings.mesh_select_mode = (True, False, False)
        prepare_knife_session(bpy.context)
        simulate_bent_both_sides(obj)

        bm = bmesh.from_edit_mesh(obj.data)
        by_side, total = core.collect_knife_path_edges_by_side(
            bm,
            core.AXIS_INDEX["X"],
            1.0e-4,
        )
        source_edges = by_side["POSITIVE"] + by_side["NEGATIVE"]
        assert total >= 4, total
        assert source_edges, {key: len(value) for key, value in by_side.items()}
        assert not by_side["CROSSES"]
        session = next(iter(operators._SESSIONS.values()))
        assert not core.reflected_path_uses_only_target_boundaries(
            bm,
            source_edges,
            session.axis_index,
            session.tolerance,
            session.mirror_face_ids,
        ), "fixture must force Knife Project fallback"

        finished = bpy.ops.mesh.ydd_symmetric_edit_finish("EXEC_DEFAULT")
        assert finished == {"FINISHED"}, finished

    bm = bmesh.from_edit_mesh(obj.data)
    # Native bent path.
    assert has_exact_edge(bm, (-1.5, -1.0, 0.0), (-1.2, 0.0, 0.0))
    assert has_exact_edge(bm, (-1.2, 0.0, 0.0), (0.0, 0.0, 0.0))
    assert has_exact_edge(bm, (0.0, 0.0, 0.0), (0.7, 0.4, 0.0))
    assert has_exact_edge(bm, (0.7, 0.4, 0.0), (1.0, 1.0, 0.0))
    # Mirrored counterparts.
    assert has_exact_edge(bm, (1.5, -1.0, 0.0), (1.2, 0.0, 0.0))
    assert has_exact_edge(bm, (1.2, 0.0, 0.0), (0.0, 0.0, 0.0))
    assert has_exact_edge(bm, (0.0, 0.0, 0.0), (-0.7, 0.4, 0.0))
    assert has_exact_edge(bm, (-0.7, 0.4, 0.0), (-1.0, 1.0, 0.0))
    assert_no_duplicate_edges(bm)
    assert_x_symmetric(bm)
    assert bm.edges.layers.int.get(core.EDGE_ORIGINAL_LAYER) is None
    assert not operators._SESSIONS
    print("YSE_KNIFE_BOTH_SIDES_CASE_E=OK", flush=True)


def case_f_partial_failure_rollback(window, area, region) -> None:
    print("YSE_KNIFE_BOTH_SIDES_CASE=f_partial_rollback", flush=True)
    clear_scene()
    obj = make_two_quads()
    original_apply = core.apply_reflected_path_topology
    with bpy.context.temp_override(window=window, area=area, region=region):
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.context.tool_settings.mesh_select_mode = (True, False, False)
        prepare_knife_session(bpy.context)
        simulate_one_side_cut(obj)
        bm = bmesh.from_edit_mesh(obj.data)
        native_counts = (len(bm.verts), len(bm.edges), len(bm.faces))

        def forced_apply(*args, **kwargs):
            # Apply fully then fail: exercises post-partial rollback (contract §2.2).
            created, present, _reason = original_apply(*args, **kwargs)
            return created, present, "forced partial mirror failure"

        core.apply_reflected_path_topology = forced_apply
        try:
            finished = bpy.ops.mesh.ydd_symmetric_edit_finish("EXEC_DEFAULT")
        finally:
            core.apply_reflected_path_topology = original_apply
        assert finished == {"FINISHED"}, finished

    bm = bmesh.from_edit_mesh(obj.data)
    assert (len(bm.verts), len(bm.edges), len(bm.faces)) == native_counts, (
        (len(bm.verts), len(bm.edges), len(bm.faces)),
        native_counts,
    )
    # Native left cut kept; mirrored right cut rolled back.
    assert has_exact_edge(bm, (-1.5, -1.0, 0.0), (-1.5, 1.0, 0.0))
    assert not has_exact_edge(bm, (1.5, -1.0, 0.0), (1.5, 1.0, 0.0))
    warnings = warning_messages()
    assert any("forced partial mirror failure" in message for message in warnings), (
        warnings,
        operators._FINISH_REPORTS,
    )
    assert not any(kind == "ERROR" for kind, _message in operators._FINISH_REPORTS)
    assert bm.edges.layers.int.get(core.EDGE_ORIGINAL_LAYER) is None
    assert not operators._SESSIONS
    print("YSE_KNIFE_BOTH_SIDES_CASE_F=OK", flush=True)


def case_g_self_mirrored(window, area, region) -> None:
    print("YSE_KNIFE_BOTH_SIDES_CASE=g_self_mirrored", flush=True)
    clear_scene()
    obj = make_two_quads()
    with bpy.context.temp_override(window=window, area=area, region=region):
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.context.tool_settings.mesh_select_mode = (True, False, False)
        prepare_knife_session(bpy.context)
        native_counts = simulate_self_mirrored_cut(obj)
        finished = bpy.ops.mesh.ydd_symmetric_edit_finish("EXEC_DEFAULT")
        assert finished == {"FINISHED"}, finished

    bm = bmesh.from_edit_mesh(obj.data)
    # Topology unchanged: reflected segments already present (already_present).
    assert (len(bm.verts), len(bm.edges), len(bm.faces)) == native_counts, (
        (len(bm.verts), len(bm.edges), len(bm.faces)),
        native_counts,
    )
    assert has_exact_edge(bm, (-1.5, -1.0, 0.0), (-1.5, 1.0, 0.0))
    assert has_exact_edge(bm, (1.5, -1.0, 0.0), (1.5, 1.0, 0.0))
    assert_no_duplicate_edges(bm)
    assert_x_symmetric(bm)
    assert bm.edges.layers.int.get(core.EDGE_ORIGINAL_LAYER) is None
    assert not operators._SESSIONS
    print("YSE_KNIFE_BOTH_SIDES_CASE_G=OK", flush=True)


def run_test() -> None:
    addon.register()
    window, area, region = viewport_context()
    configure_view(area)

    case_a_straddle(window, area, region)
    case_b_crosses(window, area, region)
    case_c_one_side(window, area, region)
    case_d_mixed_crosses(window, area, region)
    case_e_knife_project_fallback(window, area, region)
    case_f_partial_failure_rollback(window, area, region)
    case_g_self_mirrored(window, area, region)

    print(MARKER_OK, flush=True)
    addon.unregister()
    bpy.ops.wm.quit_blender()


def guarded_run():
    try:
        run_test()
    except BaseException:
        fail()
    return None


bpy.app.timers.register(guarded_run, first_interval=0.25)
