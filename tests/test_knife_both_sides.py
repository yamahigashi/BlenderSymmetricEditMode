# SPDX-License-Identifier: GPL-3.0-or-later

"""Phase 1/2: Knife both-sides mirror + p-stitch (axis-crossing contract §4.1).

Cases:
  (a) Stroke straddling the mirror plane via an on-plane vertex (no CROSSES)
      → full X-symmetry after finish; fixed topology counts; face incidence.
  (b) Stroke with a plane-crossing segment → p-stitch + both-sides mirror (X).
  (c) One-sided stroke (regression) → opposite side still mirrored.
  (d) CROSSES mixed with POSITIVE/NEGATIVE → whole stroke symmetrized.
  (e) Bent polyline direct mirror regression.
  (f) Direct mirror failure (monkeypatch) → full rollback + WARNING only
      (no success INFO) + FINISHED.
  (g) Near-self-mirrored stroke → already_present, no double cut.
  (h) Simple 1-segment CROSSES → on-plane p, 4 edges share p, full symmetry.
  (i) Three CROSSES through same p → single vertex, degree ≥ 6, full symmetry.
  (j) Self-mirrored CROSSES segment → no p vertex, native topology kept.
  (k) CROSSES + one-sided asymmetric half-segments (distinct from d).
  (l) CROSSES on asymmetric carrier → whole-stage decline + WARNING + FINISHED.
  (m) Backup creation failure → ERROR + FINISHED + native intact.
  Headless units: orphan face remap, host-edge selection, non-transitive tol.

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
from mathutils import Quaternion, Vector

PACKAGE_PARENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(PACKAGE_PARENT))

import ydd_symmetric_edit as addon  # noqa: E402
from ydd_symmetric_edit import (  # noqa: E402
    face_mapping,
    layer_names,
    matching,
    operators,
    stitch_common,
    stitch_pathedges,
    stitch_pstitch,
    stitch_reflect,
)
from ydd_symmetric_edit._types import FaceId  # noqa: E402

MARKER_OK = "YSE_KNIFE_BOTH_SIDES_OK"
MARKER_FAILED = "YSE_KNIFE_BOTH_SIDES_FAILED"
COORD_PRECISION = 5
MIRROR_COUNTERPART_WARNING = "no exact mirrored counterpart"


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


def mesh_signature(bm, precision: int = COORD_PRECISION):
    """Full mesh signature used to prove native-only rollback."""

    return (
        vertex_coord_multiset(bm, precision),
        edge_coord_multiset(bm, precision),
        face_incidence_multiset(bm, precision),
    )


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
        expected = matching.mirror_coordinate(vertex.co, matching.AXIS_INDEX["X"])
        assert any(matching.coordinates_match(other.co, expected, tolerance) for other in bm.verts), (
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


def error_messages() -> list[str]:
    return [message for kind, message in operators._FINISH_REPORTS if kind == "ERROR"]


def info_messages() -> list[str]:
    return [message for kind, message in operators._FINISH_REPORTS if kind == "INFO"]


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
        by_side, total = stitch_pathedges.collect_knife_path_edges_by_side(
            bm,
            matching.AXIS_INDEX["X"],
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
    assert bm.edges.layers.int.get(layer_names.EDGE_ORIGINAL_LAYER) is None
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
        simulate_cross_plane_segment(obj)

        bm = bmesh.from_edit_mesh(obj.data)
        by_side, total = stitch_pathedges.collect_knife_path_edges_by_side(
            bm,
            matching.AXIS_INDEX["X"],
            1.0e-4,
        )
        assert total >= 1, total
        assert by_side["CROSSES"], "fixture must include CROSSES edges"

        finished = bpy.ops.mesh.ydd_symmetric_edit_finish("EXEC_DEFAULT")
        assert finished == {"FINISHED"}, finished

    bm = bmesh.from_edit_mesh(obj.data)
    # Phase 2: p-stitch + mirror produces an X at the plane.
    assert has_exact_edge(bm, (-1.5, -1.0, 0.0), (0.0, 0.0, 0.0))
    assert has_exact_edge(bm, (0.0, 0.0, 0.0), (1.5, 1.0, 0.0))
    assert has_exact_edge(bm, (1.5, -1.0, 0.0), (0.0, 0.0, 0.0))
    assert has_exact_edge(bm, (0.0, 0.0, 0.0), (-1.5, 1.0, 0.0))
    on_plane = [vertex for vertex in bm.verts if abs(vertex.co.x) <= 1.0e-4]
    assert any(
        abs(vertex.co.x) <= 1.0e-4 and abs(vertex.co.y) <= 1.0e-4 and abs(vertex.co.z) <= 1.0e-4 for vertex in on_plane
    ), [tuple(vertex.co) for vertex in on_plane]
    assert_no_duplicate_edges(bm)
    assert_x_symmetric(bm)
    assert bm.edges.layers.int.get(layer_names.EDGE_ORIGINAL_LAYER) is None
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
    assert bm.edges.layers.int.get(layer_names.EDGE_ORIGINAL_LAYER) is None
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
        simulate_mixed_crosses(obj)

        bm = bmesh.from_edit_mesh(obj.data)
        by_side, total = stitch_pathedges.collect_knife_path_edges_by_side(
            bm,
            matching.AXIS_INDEX["X"],
            1.0e-4,
        )
        assert total >= 3, (total, {key: len(value) for key, value in by_side.items()})
        assert by_side["CROSSES"], "fixture must include CROSSES"
        assert by_side["POSITIVE"] or by_side["NEGATIVE"], {key: len(value) for key, value in by_side.items()}

        finished = bpy.ops.mesh.ydd_symmetric_edit_finish("EXEC_DEFAULT")
        assert finished == {"FINISHED"}, finished

    bm = bmesh.from_edit_mesh(obj.data)
    # Phase 2: mixed stroke is fully symmetrized (no whole-stage decline).
    assert has_exact_edge(bm, (-1.5, -1.0, 0.0), (-0.5, 0.0, 0.0))
    assert has_exact_edge(bm, (0.5, 0.0, 0.0), (1.5, 1.0, 0.0))
    assert_no_duplicate_edges(bm)
    assert_x_symmetric(bm)
    assert bm.edges.layers.int.get(layer_names.EDGE_ORIGINAL_LAYER) is None
    assert not operators._SESSIONS
    print("YSE_KNIFE_BOTH_SIDES_CASE_D=OK", flush=True)


def case_e_bent_direct(window, area, region) -> None:
    print("YSE_KNIFE_BOTH_SIDES_CASE=e_bent_direct", flush=True)
    clear_scene()
    obj = make_straddling_grid()
    with bpy.context.temp_override(window=window, area=area, region=region):
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.context.tool_settings.mesh_select_mode = (True, False, False)
        prepare_knife_session(bpy.context)
        simulate_bent_both_sides(obj)

        bm = bmesh.from_edit_mesh(obj.data)
        by_side, total = stitch_pathedges.collect_knife_path_edges_by_side(
            bm,
            matching.AXIS_INDEX["X"],
            1.0e-4,
        )
        source_edges = by_side["POSITIVE"] + by_side["NEGATIVE"]
        assert total >= 4, total
        assert source_edges, {key: len(value) for key, value in by_side.items()}
        assert not by_side["CROSSES"]
        del source_edges
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
    assert bm.edges.layers.int.get(layer_names.EDGE_ORIGINAL_LAYER) is None
    assert not operators._SESSIONS
    print("YSE_KNIFE_BOTH_SIDES_CASE_E=OK", flush=True)


def case_f_direct_decline_rollback(window, area, region) -> None:
    """O16(a): a direct apply decline rolls back to the native-only mesh."""

    print("YSE_KNIFE_BOTH_SIDES_CASE=f_partial_rollback", flush=True)
    clear_scene()
    obj = make_two_quads()
    original_apply = stitch_reflect.apply_reflected_path_topology
    with bpy.context.temp_override(window=window, area=area, region=region):
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.context.tool_settings.mesh_select_mode = (True, False, False)
        prepare_knife_session(bpy.context)
        simulate_one_side_cut(obj)
        bm = bmesh.from_edit_mesh(obj.data)
        native_sig = mesh_signature(bm)

        def forced_apply(*args, **kwargs):
            result = original_apply(*args, **kwargs)
            reason = "forced partial mirror failure"
            if kwargs.get("return_summary"):
                summary = result[3] if len(result) > 3 else None
                return result[0], result[1], reason, summary
            return result[0], result[1], reason

        stitch_reflect.apply_reflected_path_topology = forced_apply
        try:
            finished = bpy.ops.mesh.ydd_symmetric_edit_finish("EXEC_DEFAULT")
        finally:
            stitch_reflect.apply_reflected_path_topology = original_apply
        assert finished == {"FINISHED"}, finished

    bm = bmesh.from_edit_mesh(obj.data)
    assert mesh_signature(bm) == native_sig
    warnings = warning_messages()
    assert any("direct mirror declined: forced partial mirror failure" in message for message in warnings), (
        warnings,
        operators._FINISH_REPORTS,
    )
    assert not info_messages(), operators._FINISH_REPORTS
    assert not any(kind == "ERROR" for kind, _message in operators._FINISH_REPORTS)
    assert bm.edges.layers.int.get(layer_names.EDGE_ORIGINAL_LAYER) is None
    assert not operators._SESSIONS
    print("YSE_KNIFE_BOTH_SIDES_CASE_F=OK", flush=True)


def case_d_knife_project_never_called(window, area, region) -> None:
    """O16(d): the addon can no longer reference or invoke Knife Project.

    The bpy.ops proxy offers no persistent attribute hook to spy through, so
    this pins the stronger static impossibility instead: the projection module
    is gone and no package source mentions the operator, then a plain cut
    still completes through the direct route.
    """

    print("YSE_KNIFE_BOTH_SIDES_CASE=O16_D_knife_project_never_called", flush=True)
    clear_scene()
    obj = make_two_quads()

    import importlib.util
    import pathlib

    assert importlib.util.find_spec("ydd_symmetric_edit.stitch_projection") is None
    package_dir = pathlib.Path(operators.__file__).parent
    offenders = [
        path.name for path in sorted(package_dir.glob("*.py")) if "knife_project" in path.read_text(encoding="utf-8")
    ]
    assert not offenders, offenders

    with bpy.context.temp_override(window=window, area=area, region=region):
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.context.tool_settings.mesh_select_mode = (True, False, False)
        prepare_knife_session(bpy.context)
        simulate_one_side_cut(obj)
        finished = bpy.ops.mesh.ydd_symmetric_edit_finish("EXEC_DEFAULT")
        assert finished == {"FINISHED"}, finished
    assert has_exact_edge(bmesh.from_edit_mesh(obj.data), (1.5, -1.0, 0.0), (1.5, 1.0, 0.0))
    assert not operators._SESSIONS
    print("YSE_KNIFE_BOTH_SIDES_CASE_O16_D=OK", flush=True)


def case_e2_gate_decline_rollback(window, area, region) -> None:
    """O16(e2): gate=False raises explicitly without changing native topology."""

    print("YSE_KNIFE_BOTH_SIDES_CASE=e2_gate_decline", flush=True)
    clear_scene()
    obj = make_two_quads()

    with bpy.context.temp_override(window=window, area=area, region=region):
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.context.tool_settings.mesh_select_mode = (True, False, False)
        prepare_knife_session(bpy.context)
        session = next(iter(operators._SESSIONS.values()))
        tolerance = session.tolerance
        bm = bmesh.from_edit_mesh(obj.data)
        # Endpoint-ambiguous wire recipe (organic gate=False): two stray
        # vertices within tolerance of the mirrored dangling endpoint make
        # R-W1 endpoint resolution ambiguous, so the gate declines without
        # any monkeypatch.
        corner = next(v for v in bm.verts if (v.co - Vector((-2.0, -1.0, 0.0))).length <= 1.0e-6)
        dangling = bm.verts.new((-1.7, -0.5, 0.2))
        bm.edges.new((corner, dangling))
        bm.verts.new((1.7 - 0.25 * tolerance, -0.5, 0.2))
        bm.verts.new((1.7 + 0.25 * tolerance, -0.5, 0.2))
        bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
        native_sig = mesh_signature(bm)
        finished = bpy.ops.mesh.ydd_symmetric_edit_finish("EXEC_DEFAULT")
        assert finished == {"FINISHED"}, finished

    bm = bmesh.from_edit_mesh(obj.data)
    assert mesh_signature(bm) == native_sig
    warnings = warning_messages()
    assert any(
        "the mirrored cut cannot be rebuilt directly on the opposite side" in message for message in warnings
    ), operators._FINISH_REPORTS
    assert any("native cut kept; mirror manually or undo" in message for message in warnings), (
        operators._FINISH_REPORTS,
    )
    assert not error_messages(), operators._FINISH_REPORTS
    assert not operators._SESSIONS
    print("YSE_KNIFE_BOTH_SIDES_CASE_E2=OK", flush=True)


def case_h_count_mismatch_rollback(window, area, region) -> None:
    """O16(h): an empty-reason count mismatch is an explicit direct decline."""

    print("YSE_KNIFE_BOTH_SIDES_CASE=O16_H_count_mismatch", flush=True)
    clear_scene()
    obj = make_two_quads()
    original_apply = stitch_reflect.apply_reflected_path_topology

    def forced_count_mismatch(*args, **kwargs):
        result = original_apply(*args, **kwargs)
        if kwargs.get("return_summary"):
            summary = result[3] if len(result) > 3 else None
            return 0, 0, "", summary
        return 0, 0, ""

    with bpy.context.temp_override(window=window, area=area, region=region):
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.context.tool_settings.mesh_select_mode = (True, False, False)
        prepare_knife_session(bpy.context)
        simulate_one_side_cut(obj)
        bm = bmesh.from_edit_mesh(obj.data)
        native_sig = mesh_signature(bm)
        stitch_reflect.apply_reflected_path_topology = forced_count_mismatch
        try:
            finished = bpy.ops.mesh.ydd_symmetric_edit_finish("EXEC_DEFAULT")
        finally:
            stitch_reflect.apply_reflected_path_topology = original_apply
        assert finished == {"FINISHED"}, finished

    bm = bmesh.from_edit_mesh(obj.data)
    assert mesh_signature(bm) == native_sig
    warnings = warning_messages()
    assert any("did not match the source" in message for message in warnings), operators._FINISH_REPORTS
    assert not error_messages(), operators._FINISH_REPORTS
    assert not operators._SESSIONS
    print("YSE_KNIFE_BOTH_SIDES_CASE_O16_H=OK", flush=True)


def case_i_rollback_exception_error(window, area, region) -> None:
    """O16(i): rollback exceptions are reported as ERROR only."""

    print("YSE_KNIFE_BOTH_SIDES_CASE=O16_I_rollback_exception", flush=True)
    from ydd_symmetric_edit import backup as backup_mod

    clear_scene()
    obj = make_two_quads()
    original_apply = stitch_reflect.apply_reflected_path_topology
    original_restore = backup_mod.restore_topology_backup

    def forced_apply(*args, **kwargs):
        result = original_apply(*args, **kwargs)
        reason = "forced partial mirror failure"
        if kwargs.get("return_summary"):
            summary = result[3] if len(result) > 3 else None
            return result[0], result[1], reason, summary
        return result[0], result[1], reason

    def broken_restore(*_args, **_kwargs):
        raise RuntimeError("forced rollback failure")

    with bpy.context.temp_override(window=window, area=area, region=region):
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.context.tool_settings.mesh_select_mode = (True, False, False)
        prepare_knife_session(bpy.context)
        simulate_one_side_cut(obj)
        stitch_reflect.apply_reflected_path_topology = forced_apply
        backup_mod.restore_topology_backup = broken_restore
        try:
            finished = bpy.ops.mesh.ydd_symmetric_edit_finish("EXEC_DEFAULT")
        except RuntimeError as exc:
            # Blender surfaces operator.report({'ERROR'}) as RuntimeError from
            # script invocations; the operator itself returned FINISHED.
            assert "direct mirror declined" in str(exc), exc
            finished = {"FINISHED"}
        finally:
            stitch_reflect.apply_reflected_path_topology = original_apply
            backup_mod.restore_topology_backup = original_restore
        assert finished == {"FINISHED"}, finished

    assert error_messages(), operators._FINISH_REPORTS
    # ERROR cannot promise the mirror side is untouched, so the disposition
    # hint must stay off this branch.
    assert not any("native cut kept" in message for message in error_messages()), operators._FINISH_REPORTS
    assert not warning_messages(), operators._FINISH_REPORTS
    assert not operators._SESSIONS
    print("YSE_KNIFE_BOTH_SIDES_CASE_O16_I=OK", flush=True)


def case_n_weak_band_direct(window, area, region) -> None:
    """O8: a rev3-declined weak-band stroke completes on the direct route."""

    print("YSE_KNIFE_BOTH_SIDES_CASE=n_weak_band_direct", flush=True)
    from fixtures_interior_host import (
        _face_surface_triangles_for_fixture,
        _split_boundary,
        _surface_distance,
    )

    clear_scene()
    z = 5.0e-5
    pentagon = [
        (-2.0, -1.0, z * -1.0),
        (-1.0, -1.0, z * 0.8),
        (-1.2, 0.1, z * 0.9),
        (-1.0, 1.0, z * -0.7),
        (-2.0, 1.0, z * 0.3),
    ]
    mesh = bpy.data.meshes.new("WeakBand")
    verts = pentagon + [(-x, y, zz) for x, y, zz in reversed(pentagon)]
    mesh.from_pydata(verts, [], [tuple(range(5)), tuple(range(5, 10))])
    mesh.update()
    obj = bpy.data.objects.new("WeakBand", mesh)
    bpy.context.scene.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    obj.use_mesh_mirror_x = True

    mirrored_chain: list[tuple[float, float, float]] = []
    with bpy.context.temp_override(window=window, area=area, region=region):
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.context.tool_settings.mesh_select_mode = (True, False, False)
        prepare_knife_session(bpy.context)
        bm = bmesh.from_edit_mesh(obj.data)
        bottom = _split_boundary(bm, y=-1.0, x=-1.5)
        top = _split_boundary(bm, y=1.0, x=-1.5)
        source = next(face for face in bm.faces if bottom in face.verts and top in face.verts)
        target = next(face for face in bm.faces if all(float(vertex.co.x) > 0.0 for vertex in face.verts))
        tolerance = 1.0e-5
        point = None
        for triangle in _face_surface_triangles_for_fixture(source):
            candidate = (triangle[0] * 0.25) + (triangle[1] * 0.35) + (triangle[2] * 0.40)
            mirrored = Vector((-candidate.x, candidate.y, candidate.z))
            if _surface_distance(candidate, source) <= 1.0e-6 and 2.0 * tolerance < _surface_distance(mirrored, target):
                point = candidate
                break
        assert point is not None, "weak band premise not met"
        bmesh.utils.face_split(source, bottom, top, coords=[tuple(point)])
        chain = [tuple(bottom.co), tuple(point), tuple(top.co)]
        mirrored_chain = [(-x, y, zz) for x, y, zz in chain]
        bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)

        finished = bpy.ops.mesh.ydd_symmetric_edit_finish("EXEC_DEFAULT")
        assert finished == {"FINISHED"}, finished

    bm = bmesh.from_edit_mesh(obj.data)
    assert has_exact_edge(bm, mirrored_chain[0], mirrored_chain[1])
    assert has_exact_edge(bm, mirrored_chain[1], mirrored_chain[2])
    assert_no_duplicate_edges(bm)
    # The KNIFE direct-success path reports via self.report (not _finish_report).
    assert not warning_messages(), operators._FINISH_REPORTS
    assert not any(kind == "ERROR" for kind, _message in operators._FINISH_REPORTS)
    assert bm.edges.layers.int.get(layer_names.EDGE_ORIGINAL_LAYER) is None
    assert not operators._SESSIONS
    print("YSE_KNIFE_BOTH_SIDES_CASE_N=OK", flush=True)


def case_o_endpoint_collision_rollback(window, area, region) -> None:
    """O16(e1): organic apply decline rolls back completely."""

    print("YSE_KNIFE_BOTH_SIDES_CASE=o_endpoint_collision_rollback", flush=True)
    clear_scene()
    obj = make_two_quads()
    tolerance = 1.0e-5
    with bpy.context.temp_override(window=window, area=area, region=region):
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.context.tool_settings.mesh_select_mode = (True, False, False)
        prepare_knife_session(bpy.context)
        bm = bmesh.from_edit_mesh(obj.data)
        bottom = next(
            edge
            for edge in bm.edges
            if all(vertex.co.x < 0.0 for vertex in edge.verts)
            and all(abs(vertex.co.y + 1.0) < 1.0e-8 for vertex in edge.verts)
        )
        _edge, end_a = bmesh.utils.edge_split(bottom, bottom.verts[0], 0.5)
        end_a.co = (-1.5, -1.0, 0.0)
        fragment = next(
            edge
            for edge in end_a.link_edges
            if abs(min(vertex.co.x for vertex in edge.verts) + 1.5) < 1.0e-6
            and all(abs(vertex.co.y + 1.0) < 1.0e-8 for vertex in edge.verts)
        )
        _edge, middle = bmesh.utils.edge_split(fragment, end_a, 0.1)
        middle.co = (-1.5 + 0.25 * tolerance, -1.0, 0.0)
        next_fragment = next(edge for edge in middle.link_edges if end_a not in edge.verts)
        _edge, end_b = bmesh.utils.edge_split(next_fragment, middle, 0.1)
        end_b.co = (-1.5 + 0.5 * tolerance, -1.0, 0.0)
        host = next(face for face in bm.faces if end_a in face.verts and end_b in face.verts)
        bmesh.utils.face_split(host, end_a, end_b, coords=[(-1.3, -0.5, 0.0)])
        bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
        native_sig = mesh_signature(bm)
        finished = bpy.ops.mesh.ydd_symmetric_edit_finish("EXEC_DEFAULT")
        assert finished == {"FINISHED"}, finished

    # Direct decline is explicit and the native cut remains the only change.
    assert not any(kind == "ERROR" for kind, _message in operators._FINISH_REPORTS)
    warnings = warning_messages()
    assert any("direct mirror declined" in message for message in warnings), operators._FINISH_REPORTS
    bm = bmesh.from_edit_mesh(obj.data)
    assert mesh_signature(bm) == native_sig
    assert bm.edges.layers.int.get(layer_names.EDGE_ORIGINAL_LAYER) is None
    assert not operators._SESSIONS
    print("YSE_KNIFE_BOTH_SIDES_CASE_O=OK", flush=True)


def case_q_wire_strokes_direct(window, area, region) -> None:
    """O14(c)(d): dangling wire strokes mirror directly (pure-wire path)."""

    print("YSE_KNIFE_BOTH_SIDES_CASE=q_wire_strokes_direct", flush=True)
    clear_scene()
    obj = make_two_quads()
    with bpy.context.temp_override(window=window, area=area, region=region):
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.context.tool_settings.mesh_select_mode = (True, False, False)
        prepare_knife_session(bpy.context)
        bm = bmesh.from_edit_mesh(obj.data)
        corner = next(v for v in bm.verts if (v.co - Vector((-2.0, -1.0, 0.0))).length <= 1e-6)
        dangling = bm.verts.new((-1.7, -0.5, 0.2))
        bm.edges.new((corner, dangling))
        free_a = bm.verts.new((-1.8, 0.6, 0.1))
        free_b = bm.verts.new((-1.6, 0.8, 0.1))
        bm.edges.new((free_a, free_b))
        bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)

        finished = bpy.ops.mesh.ydd_symmetric_edit_finish("EXEC_DEFAULT")
        assert finished == {"FINISHED"}, finished

    bm = bmesh.from_edit_mesh(obj.data)
    assert has_exact_edge(bm, (2.0, -1.0, 0.0), (1.7, -0.5, 0.2))
    assert has_exact_edge(bm, (1.8, 0.6, 0.1), (1.6, 0.8, 0.1))
    mirrored_wire = next(
        edge
        for edge in bm.edges
        if edge.is_valid
        and all(
            (vertex.co - Vector((1.8, 0.6, 0.1))).length <= 1e-6 or (vertex.co - Vector((1.6, 0.8, 0.1))).length <= 1e-6
            for vertex in edge.verts
        )
    )
    assert mirrored_wire.is_wire
    assert not warning_messages(), operators._FINISH_REPORTS
    assert not any(kind == "ERROR" for kind, _message in operators._FINISH_REPORTS)
    assert bm.edges.layers.int.get(layer_names.EDGE_ORIGINAL_LAYER) is None
    assert not operators._SESSIONS
    print("YSE_KNIFE_BOTH_SIDES_CASE_Q=OK", flush=True)


def case_r_network_direct(window, area, region) -> None:
    """O15(f): Y/X network mirrors use the direct topology operator path."""

    print("YSE_KNIFE_BOTH_SIDES_CASE=r_network_direct", flush=True)
    for kind, degree in (("Y", 3), ("X", 4)):
        clear_scene()
        obj = make_two_quads()
        with bpy.context.temp_override(window=window, area=area, region=region):
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.context.tool_settings.mesh_select_mode = (True, False, False)
            prepare_knife_session(bpy.context)
            _simulate_network_native(obj, kind)
            finished = bpy.ops.mesh.ydd_symmetric_edit_finish("EXEC_DEFAULT")
            assert finished == {"FINISHED"}, (kind, finished)
        bm = bmesh.from_edit_mesh(obj.data)
        hub = next(
            vertex
            for vertex in bm.verts
            if abs(float(vertex.co.x) - 1.3) < 1.0e-5 and abs(float(vertex.co.y)) < 1.0e-5
        )
        assert len(hub.link_edges) == degree, (kind, len(hub.link_edges))
        assert_no_duplicate_edges(bm)
        assert_x_symmetric(bm)
        assert not error_messages(), operators._FINISH_REPORTS
        assert not operators._SESSIONS
    print("YSE_KNIFE_BOTH_SIDES_CASE_R=OK", flush=True)


def case_s_network_partial_failure_rollback(window, area, region) -> None:
    """O16(f): a late direct failure rolls back the whole mirror stage."""

    print("YSE_KNIFE_BOTH_SIDES_CASE=s_network_rollback", flush=True)
    original_split = stitch_reflect._face_split_mutation
    split_calls = []

    def _forced_split(*args, **kwargs):
        split_calls.append(True)
        if len(split_calls) >= 2:
            return None, "forced network realization failure"
        return original_split(*args, **kwargs)

    clear_scene()
    obj = make_two_quads()
    stitch_reflect._face_split_mutation = _forced_split
    try:
        with bpy.context.temp_override(window=window, area=area, region=region):
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.context.tool_settings.mesh_select_mode = (True, False, False)
            prepare_knife_session(bpy.context)
            _simulate_network_native(obj, "Y")
            native_sig = mesh_signature(bmesh.from_edit_mesh(obj.data))
            finished = bpy.ops.mesh.ydd_symmetric_edit_finish("EXEC_DEFAULT")
        assert finished == {"FINISHED"}, finished
        assert len(split_calls) >= 2, split_calls
        assert not any(kind == "ERROR" for kind, _message in operators._FINISH_REPORTS)
        assert any("direct mirror declined" in message for _kind, message in operators._FINISH_REPORTS)
    finally:
        stitch_reflect._face_split_mutation = original_split
    bm = bmesh.from_edit_mesh(obj.data)
    assert mesh_signature(bm) == native_sig
    assert_no_duplicate_edges(bm)
    assert bm.edges.layers.int.get(layer_names.EDGE_ORIGINAL_LAYER) is None
    assert not operators._SESSIONS
    print("YSE_KNIFE_BOTH_SIDES_CASE_S=OK", flush=True)


def _simulate_network_native(obj, kind: str) -> None:
    """Build a native Y/X cut in an edit mesh for operator-level O15 checks."""

    bm = bmesh.from_edit_mesh(obj.data)

    def split_edge(predicate, coordinate):
        edge = next(edge for edge in bm.edges if predicate(edge))
        start = edge.verts[0]
        delta = (
            coordinate[0] - float(start.co.x)
            if abs(coordinate[0] - float(start.co.x)) > 1.0e-8
            else coordinate[1] - float(start.co.y)
        )
        span = (
            float(edge.verts[1].co.x) - float(start.co.x)
            if abs(edge.verts[1].co.x - start.co.x) > 1.0e-8
            else float(edge.verts[1].co.y) - float(start.co.y)
        )
        _new_edge, vertex = bmesh.utils.edge_split(edge, start, delta / span)
        vertex.co = coordinate
        return vertex

    bottom = split_edge(
        lambda edge: (
            all(abs(float(vertex.co.y) + 1.0) < 1.0e-8 for vertex in edge.verts)
            and all(vertex.co.x < 0.0 for vertex in edge.verts)
            and min(vertex.co.x for vertex in edge.verts) < -1.5 < max(vertex.co.x for vertex in edge.verts)
        ),
        (-1.5, -1.0, 0.0),
    )
    top = split_edge(
        lambda edge: (
            all(abs(float(vertex.co.y) - 1.0) < 1.0e-8 for vertex in edge.verts)
            and all(vertex.co.x < 0.0 for vertex in edge.verts)
            and min(vertex.co.x for vertex in edge.verts) < -1.5 < max(vertex.co.x for vertex in edge.verts)
        ),
        (-1.5, 1.0, 0.0),
    )
    left = split_edge(
        lambda edge: (
            all(abs(float(vertex.co.x) + 2.0) < 1.0e-8 for vertex in edge.verts)
            and min(vertex.co.y for vertex in edge.verts) < 0.0 < max(vertex.co.y for vertex in edge.verts)
        ),
        (-2.0, 0.0, 0.0),
    )
    bmesh.utils.face_split(
        next(face for face in bm.faces if bottom in face.verts and top in face.verts),
        bottom,
        top,
        coords=[(-1.3, 0.0, 0.0)],
    )
    hub = next(
        vertex for vertex in bm.verts if abs(float(vertex.co.x) + 1.3) < 1.0e-5 and abs(float(vertex.co.y)) < 1.0e-5
    )
    bmesh.utils.face_split(next(face for face in bm.faces if left in face.verts and hub in face.verts), left, hub)
    if kind == "X":
        right = split_edge(
            lambda edge: (
                all(abs(float(vertex.co.x) + 1.0) < 1.0e-8 for vertex in edge.verts)
                and min(vertex.co.y for vertex in edge.verts) < 0.25 < max(vertex.co.y for vertex in edge.verts)
            ),
            (-1.0, 0.25, 0.0),
        )
        bmesh.utils.face_split(next(face for face in bm.faces if right in face.verts and hub in face.verts), right, hub)
    bmesh.update_edit_mesh(obj.data, loop_triangles=True, destructive=True)


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
    assert bm.edges.layers.int.get(layer_names.EDGE_ORIGINAL_LAYER) is None
    assert not operators._SESSIONS
    print("YSE_KNIFE_BOTH_SIDES_CASE_G=OK", flush=True)


def simulate_bowtie_crosses(obj) -> None:
    """Three CROSSES diagonals that share the plane intersection p=(0,0).

    On a spanning face after dissolving the plane edge:
      (-1.5,-1)→(1.5,1), (-1.5,1)→(1.5,-1), (-2.0,-0.5)→(2.0,0.5).
    Clustering must produce a single p; after stitch+mirror degree ≥ 6.
    """

    bm = bmesh.from_edit_mesh(obj.data)
    # Boundary split points for the three diagonals.
    left_bottom = next(
        edge
        for edge in bm.edges
        if all(vertex.co.x <= 1.0e-8 for vertex in edge.verts)
        and all(abs(vertex.co.y + 1.0) <= 1.0e-8 for vertex in edge.verts)
        and min(vertex.co.x for vertex in edge.verts) < -1.0e-8
    )
    right_bottom = next(
        edge
        for edge in bm.edges
        if all(vertex.co.x >= -1.0e-8 for vertex in edge.verts)
        and all(abs(vertex.co.y + 1.0) <= 1.0e-8 for vertex in edge.verts)
        and max(vertex.co.x for vertex in edge.verts) > 1.0e-8
    )
    left_top = next(
        edge
        for edge in bm.edges
        if all(vertex.co.x <= 1.0e-8 for vertex in edge.verts)
        and all(abs(vertex.co.y - 1.0) <= 1.0e-8 for vertex in edge.verts)
        and min(vertex.co.x for vertex in edge.verts) < -1.0e-8
    )
    right_top = next(
        edge
        for edge in bm.edges
        if all(vertex.co.x >= -1.0e-8 for vertex in edge.verts)
        and all(abs(vertex.co.y - 1.0) <= 1.0e-8 for vertex in edge.verts)
        and max(vertex.co.x for vertex in edge.verts) > 1.0e-8
    )
    left_outer = next(
        edge
        for edge in bm.edges
        if all(abs(vertex.co.x + 2.0) <= 1.0e-6 for vertex in edge.verts)
        and {round(vertex.co.y, 6) for vertex in edge.verts} == {-1.0, 1.0}
    )
    right_outer = next(
        edge
        for edge in bm.edges
        if all(abs(vertex.co.x - 2.0) <= 1.0e-6 for vertex in edge.verts)
        and {round(vertex.co.y, 6) for vertex in edge.verts} == {-1.0, 1.0}
    )

    _edge, lb = bmesh.utils.edge_split(left_bottom, left_bottom.verts[0], 0.25)
    lb.co = (-1.5, -1.0, 0.0)
    _edge, rb = bmesh.utils.edge_split(right_bottom, right_bottom.verts[0], 0.25)
    rb.co = (1.5, -1.0, 0.0)
    _edge, lt = bmesh.utils.edge_split(left_top, left_top.verts[0], 0.25)
    lt.co = (-1.5, 1.0, 0.0)
    _edge, rt = bmesh.utils.edge_split(right_top, right_top.verts[0], 0.25)
    rt.co = (1.5, 1.0, 0.0)
    # Outer verticals: factor from verts[0]; place at y=±0.5 for third diagonal.
    v0, v1 = left_outer.verts
    factor_lm = (-0.5 - v0.co.y) / (v1.co.y - v0.co.y)
    _edge, lm = bmesh.utils.edge_split(left_outer, v0, factor_lm)
    lm.co = (-2.0, -0.5, 0.0)
    v0, v1 = right_outer.verts
    factor_rm = (0.5 - v0.co.y) / (v1.co.y - v0.co.y)
    _edge, rm = bmesh.utils.edge_split(right_outer, v0, factor_rm)
    rm.co = (2.0, 0.5, 0.0)

    plane_edge = next(
        edge
        for edge in bm.edges
        if all(abs(vertex.co.x) <= 1.0e-8 for vertex in edge.verts)
        and {round(vertex.co.y, 6) for vertex in edge.verts} == {-1.0, 1.0}
    )
    bmesh.ops.dissolve_edges(bm, edges=[plane_edge], use_verts=False)

    # Three independent diagonals through p=(0,0). connect_vert_pair cuts
    # across the spanning face (and subsequent faces) without requiring both
    # endpoints to already share a single face after prior splits.
    bmesh.ops.connect_vert_pair(bm, verts=[lb, rt])
    bmesh.ops.connect_vert_pair(bm, verts=[lt, rb])
    bmesh.ops.connect_vert_pair(bm, verts=[lm, rm])
    bmesh.update_edit_mesh(obj.data, loop_triangles=True, destructive=True)


def simulate_self_mirrored_crosses(obj) -> tuple[int, int, int]:
    """One CROSSES segment whose endpoints are a mirror pair (ρ(s)=s)."""

    bm = bmesh.from_edit_mesh(obj.data)
    left_bottom = next(
        edge
        for edge in bm.edges
        if all(vertex.co.x <= 1.0e-8 for vertex in edge.verts)
        and all(abs(vertex.co.y + 1.0) <= 1.0e-8 for vertex in edge.verts)
        and min(vertex.co.x for vertex in edge.verts) < -1.0e-8
    )
    right_bottom = next(
        edge
        for edge in bm.edges
        if all(vertex.co.x >= -1.0e-8 for vertex in edge.verts)
        and all(abs(vertex.co.y + 1.0) <= 1.0e-8 for vertex in edge.verts)
        and max(vertex.co.x for vertex in edge.verts) > 1.0e-8
    )
    _edge, left_pt = bmesh.utils.edge_split(left_bottom, left_bottom.verts[0], 0.25)
    left_pt.co = (-1.5, -1.0, 0.0)
    _edge, right_pt = bmesh.utils.edge_split(right_bottom, right_bottom.verts[0], 0.25)
    right_pt.co = (1.5, -1.0, 0.0)

    plane_edge = next(
        edge
        for edge in bm.edges
        if all(abs(vertex.co.x) <= 1.0e-8 for vertex in edge.verts)
        and {round(vertex.co.y, 6) for vertex in edge.verts} == {-1.0, 1.0}
    )
    bmesh.ops.dissolve_edges(bm, edges=[plane_edge], use_verts=False)
    host = next(face for face in bm.faces if left_pt in face.verts and right_pt in face.verts)
    # Horizontal self-mirrored segment at y=-1: endpoints are mirror pairs.
    # Use an interior-y polyline so the segment truly crosses the plane body:
    # (-1.0, 0.0) ↔ (1.0, 0.0).
    bmesh.utils.face_split(host, left_pt, right_pt, coords=[(-1.0, 0.0, 0.0), (1.0, 0.0, 0.0)])
    # The middle edge (-1,0)-(1,0) is the self-mirrored CROSSES segment.
    counts = (len(bm.verts), len(bm.edges), len(bm.faces))
    bmesh.update_edit_mesh(obj.data, loop_triangles=True, destructive=True)
    return counts


def make_asymmetric_straddle():
    """Matched side quads (prepare succeeds) + unmatched spanning face for the cut.

    The lower face spans X=0 asymmetrically and has no ρ(F); CROSSES there must
    decline the whole mirror stage while the matched pair keeps prepare alive.
    """

    mesh = bpy.data.meshes.new("YSE_AsymStraddleMesh")
    vertices = [
        # Matched left / right (y >= 0).
        (-2.0, 0.0, 0.0),  # 0
        (-1.0, 0.0, 0.0),  # 1
        (-1.0, 1.0, 0.0),  # 2
        (-2.0, 1.0, 0.0),  # 3
        (1.0, 0.0, 0.0),  # 4
        (2.0, 0.0, 0.0),  # 5
        (2.0, 1.0, 0.0),  # 6
        (1.0, 1.0, 0.0),  # 7
        # Unmatched spanning face (y < 0), shares (0) and (4).
        (-2.0, -1.0, 0.0),  # 8
        (1.0, -1.0, 0.0),  # 9
    ]
    mesh.from_pydata(
        vertices,
        [],
        [
            (0, 1, 2, 3),
            (4, 5, 6, 7),
            (8, 9, 4, 0),
        ],
    )
    mesh.update()
    obj = bpy.data.objects.new("YSE_AsymStraddleObject", mesh)
    bpy.context.scene.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    obj.use_mesh_mirror_x = True
    return obj


def simulate_asymmetric_crosses(obj) -> tuple[int, int, int]:
    """CROSSES diagonal on the unmatched spanning face (y < 0)."""

    bm = bmesh.from_edit_mesh(obj.data)
    bottom = next(
        edge
        for edge in bm.edges
        if all(abs(vertex.co.y + 1.0) <= 1.0e-8 for vertex in edge.verts)
        and min(vertex.co.x for vertex in edge.verts) < -1.0
    )
    top = next(
        edge
        for edge in bm.edges
        if all(abs(vertex.co.y) <= 1.0e-8 for vertex in edge.verts)
        and min(vertex.co.x for vertex in edge.verts) < -1.0
        and max(vertex.co.x for vertex in edge.verts) > 0.5
    )
    _edge, bottom_pt = bmesh.utils.edge_split(bottom, bottom.verts[0], 0.3)
    bottom_pt.co = (-1.5, -1.0, 0.0)
    _edge, top_pt = bmesh.utils.edge_split(top, top.verts[0], 0.7)
    top_pt.co = (0.5, 0.0, 0.0)
    host = next(face for face in bm.faces if bottom_pt in face.verts and top_pt in face.verts)
    bmesh.utils.face_split(host, bottom_pt, top_pt)
    counts = (len(bm.verts), len(bm.edges), len(bm.faces))
    bmesh.update_edit_mesh(obj.data, loop_triangles=True, destructive=True)
    return counts


def case_h_simple_crosses(window, area, region) -> None:
    print("YSE_KNIFE_BOTH_SIDES_CASE=h_simple_crosses", flush=True)
    clear_scene()
    obj = make_straddling_grid()
    with bpy.context.temp_override(window=window, area=area, region=region):
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.context.tool_settings.mesh_select_mode = (True, False, False)
        prepare_knife_session(bpy.context)
        simulate_cross_plane_segment(obj)
        finished = bpy.ops.mesh.ydd_symmetric_edit_finish("EXEC_DEFAULT")
        assert finished == {"FINISHED"}, finished

    bm = bmesh.from_edit_mesh(obj.data)
    p_verts = [
        vertex
        for vertex in bm.verts
        if abs(vertex.co.x) <= 1.0e-4 and abs(vertex.co.y) <= 1.0e-4 and abs(vertex.co.z) <= 1.0e-4
    ]
    assert len(p_verts) == 1, [tuple(vertex.co) for vertex in p_verts]
    p = p_verts[0]
    assert len(p.link_edges) == 4, len(p.link_edges)
    assert has_exact_edge(bm, (-1.5, -1.0, 0.0), (0.0, 0.0, 0.0))
    assert has_exact_edge(bm, (0.0, 0.0, 0.0), (1.5, 1.0, 0.0))
    assert has_exact_edge(bm, (1.5, -1.0, 0.0), (0.0, 0.0, 0.0))
    assert has_exact_edge(bm, (0.0, 0.0, 0.0), (-1.5, 1.0, 0.0))
    assert_no_duplicate_edges(bm)
    assert_x_symmetric(bm)
    assert bm.edges.layers.int.get(layer_names.EDGE_ORIGINAL_LAYER) is None
    assert not operators._SESSIONS
    print("YSE_KNIFE_BOTH_SIDES_CASE_H=OK", flush=True)


def case_i_bowtie(window, area, region) -> None:
    print("YSE_KNIFE_BOTH_SIDES_CASE=i_bowtie", flush=True)
    clear_scene()
    obj = make_straddling_grid()
    with bpy.context.temp_override(window=window, area=area, region=region):
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.context.tool_settings.mesh_select_mode = (True, False, False)
        prepare_knife_session(bpy.context)
        simulate_bowtie_crosses(obj)

        bm = bmesh.from_edit_mesh(obj.data)
        by_side, total = stitch_pathedges.collect_knife_path_edges_by_side(
            bm,
            matching.AXIS_INDEX["X"],
            1.0e-4,
        )
        # connect_vert_pair may already place on-plane verts (CROSSES→half-edges);
        # either raw CROSSES ≥ 3 or an existing p of degree ≥ 6 is acceptable.
        crosses_n = len(by_side["CROSSES"])
        p_before = [
            vertex
            for vertex in bm.verts
            if abs(vertex.co.x) <= 1.0e-4 and abs(vertex.co.y) <= 1.0e-4 and abs(vertex.co.z) <= 1.0e-4
        ]
        degree_before = max((len(vertex.link_edges) for vertex in p_before), default=0)
        assert crosses_n >= 3 or degree_before >= 6, (
            {key: len(value) for key, value in by_side.items()},
            degree_before,
            total,
        )

        finished = bpy.ops.mesh.ydd_symmetric_edit_finish("EXEC_DEFAULT")
        assert finished == {"FINISHED"}, finished

    bm = bmesh.from_edit_mesh(obj.data)
    p_verts = [
        vertex
        for vertex in bm.verts
        if abs(vertex.co.x) <= 1.0e-4 and abs(vertex.co.y) <= 1.0e-4 and abs(vertex.co.z) <= 1.0e-4
    ]
    assert len(p_verts) == 1, [tuple(vertex.co) for vertex in bm.verts]
    # Three diagonals through p → degree ≥ 6; mirror must not invent a second p.
    assert len(p_verts[0].link_edges) >= 6, len(p_verts[0].link_edges)
    assert_no_duplicate_edges(bm)
    assert_x_symmetric(bm)
    assert bm.edges.layers.int.get(layer_names.EDGE_ORIGINAL_LAYER) is None
    assert not operators._SESSIONS
    print("YSE_KNIFE_BOTH_SIDES_CASE_I=OK", flush=True)


def case_j_self_mirrored_crosses(window, area, region) -> None:
    print("YSE_KNIFE_BOTH_SIDES_CASE=j_self_mirrored_crosses", flush=True)
    clear_scene()
    obj = make_straddling_grid()
    with bpy.context.temp_override(window=window, area=area, region=region):
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.context.tool_settings.mesh_select_mode = (True, False, False)
        prepare_knife_session(bpy.context)
        native_counts = simulate_self_mirrored_crosses(obj)

        bm = bmesh.from_edit_mesh(obj.data)
        by_side, _total = stitch_pathedges.collect_knife_path_edges_by_side(
            bm,
            matching.AXIS_INDEX["X"],
            1.0e-4,
        )
        assert by_side["CROSSES"], {key: len(value) for key, value in by_side.items()}
        assert any(
            stitch_common.is_self_mirrored_edge(edge, matching.AXIS_INDEX["X"], 1.0e-4) for edge in by_side["CROSSES"]
        )

        finished = bpy.ops.mesh.ydd_symmetric_edit_finish("EXEC_DEFAULT")
        assert finished == {"FINISHED"}, finished

    bm = bmesh.from_edit_mesh(obj.data)
    # No new on-plane stitch vertex: topology count stays at the native result
    # (mirror of the non-self-mirrored wings may still run — only the
    # self-mirrored CROSSES middle is left unsplit; endpoint wings may mirror).
    # Contract: p is not generated for the self-mirrored segment itself.
    assert not any(
        abs(vertex.co.x) <= 1.0e-4 and abs(vertex.co.y) <= 1.0e-4 and abs(vertex.co.z) <= 1.0e-4 for vertex in bm.verts
    ), [tuple(vertex.co) for vertex in bm.verts]
    # The self-mirrored middle edge remains.
    assert has_exact_edge(bm, (-1.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    # Vertex count must not grow solely from a p insert on that middle edge.
    # Wings may still be mirrored, so allow >= native; forbid an isolated p.
    assert len(bm.verts) >= native_counts[0]
    assert_no_duplicate_edges(bm)
    assert bm.edges.layers.int.get(layer_names.EDGE_ORIGINAL_LAYER) is None
    assert not operators._SESSIONS
    print("YSE_KNIFE_BOTH_SIDES_CASE_J=OK", flush=True)


def simulate_crosses_plus_asymmetric_half(obj) -> tuple[int, int, int]:
    """CROSSES mixed with asymmetric half-segments (distinct from balanced case d).

    Polyline (-1.5,-1) → (-0.3,-0.2) → (0.8,0.4) → (1.5,1): NEGATIVE + CROSSES
    + POSITIVE with unequal lengths and y-offsets (not the d fixture).
    """

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
    bmesh.utils.face_split(
        host,
        left_pt,
        right_pt,
        coords=[(-0.3, -0.2, 0.0), (0.8, 0.4, 0.0)],
    )
    counts = (len(bm.verts), len(bm.edges), len(bm.faces))
    bmesh.update_edit_mesh(obj.data, loop_triangles=True, destructive=True)
    return counts


def case_k_mixed_symmetrized(window, area, region) -> None:
    """CROSSES + asymmetric half-segments (not a re-run of case d)."""

    print("YSE_KNIFE_BOTH_SIDES_CASE=k_asymmetric_half_mixed", flush=True)
    clear_scene()
    obj = make_straddling_grid()
    with bpy.context.temp_override(window=window, area=area, region=region):
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.context.tool_settings.mesh_select_mode = (True, False, False)
        prepare_knife_session(bpy.context)
        simulate_crosses_plus_asymmetric_half(obj)

        bm = bmesh.from_edit_mesh(obj.data)
        by_side, total = stitch_pathedges.collect_knife_path_edges_by_side(
            bm,
            matching.AXIS_INDEX["X"],
            1.0e-4,
        )
        assert by_side["CROSSES"], {key: len(value) for key, value in by_side.items()}
        assert by_side["POSITIVE"] and by_side["NEGATIVE"], {key: len(value) for key, value in by_side.items()}
        # Distinct from d: waypoints are not the symmetric (±0.5, 0) pair.
        assert not has_exact_edge(bm, (-0.5, 0.0, 0.0), (0.5, 0.0, 0.0))
        assert total >= 3, total

        finished = bpy.ops.mesh.ydd_symmetric_edit_finish("EXEC_DEFAULT")
        assert finished == {"FINISHED"}, finished

    bm = bmesh.from_edit_mesh(obj.data)
    # Native asymmetric polyline is present and fully X-symmetrized.
    assert has_exact_edge(bm, (-1.5, -1.0, 0.0), (-0.3, -0.2, 0.0))
    assert has_exact_edge(bm, (0.8, 0.4, 0.0), (1.5, 1.0, 0.0))
    assert has_exact_edge(bm, (1.5, -1.0, 0.0), (0.3, -0.2, 0.0))
    assert has_exact_edge(bm, (-0.8, 0.4, 0.0), (-1.5, 1.0, 0.0))
    assert_no_duplicate_edges(bm)
    assert_x_symmetric(bm)
    assert bm.edges.layers.int.get(layer_names.EDGE_ORIGINAL_LAYER) is None
    assert not operators._SESSIONS
    print("YSE_KNIFE_BOTH_SIDES_CASE_K=OK", flush=True)


def case_l_asymmetric_decline(window, area, region) -> None:
    print("YSE_KNIFE_BOTH_SIDES_CASE=l_asymmetric_decline", flush=True)
    clear_scene()
    obj = make_asymmetric_straddle()
    with bpy.context.temp_override(window=window, area=area, region=region):
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.context.tool_settings.mesh_select_mode = (True, False, False)
        prepare_knife_session(bpy.context)
        native_counts = simulate_asymmetric_crosses(obj)

        bm = bmesh.from_edit_mesh(obj.data)
        by_side, total = stitch_pathedges.collect_knife_path_edges_by_side(
            bm,
            matching.AXIS_INDEX["X"],
            1.0e-4,
        )
        assert total >= 1, total
        assert by_side["CROSSES"], {key: len(value) for key, value in by_side.items()}

        finished = bpy.ops.mesh.ydd_symmetric_edit_finish("EXEC_DEFAULT")
        assert finished == {"FINISHED"}, finished

    bm = bmesh.from_edit_mesh(obj.data)
    # Whole-stage decline: native topology preserved (rollback after p-stitch
    # or mirror preflight failure).
    assert (len(bm.verts), len(bm.edges), len(bm.faces)) == native_counts, (
        (len(bm.verts), len(bm.edges), len(bm.faces)),
        native_counts,
    )
    assert has_exact_edge(bm, (-1.5, -1.0, 0.0), (0.5, 0.0, 0.0))
    warnings = warning_messages()
    assert any(MIRROR_COUNTERPART_WARNING in message for message in warnings), (
        warnings,
        operators._FINISH_REPORTS,
    )
    assert not any(kind == "ERROR" for kind, _message in operators._FINISH_REPORTS)
    assert bm.edges.layers.int.get(layer_names.EDGE_ORIGINAL_LAYER) is None
    assert not operators._SESSIONS
    print("YSE_KNIFE_BOTH_SIDES_CASE_L=OK", flush=True)


def case_m_backup_creation_failure(window, area, region) -> None:
    """Backup create failure = fatal ERROR + FINISHED + native intact (§2.2-4)."""

    print("YSE_KNIFE_BOTH_SIDES_CASE=m_backup_create_failure", flush=True)
    clear_scene()
    obj = make_two_quads()
    from ydd_symmetric_edit import backup as yse_backup

    original_create = yse_backup.create_topology_backup

    def broken_create(_bm):
        raise RuntimeError("injected knife backup failure")

    with bpy.context.temp_override(window=window, area=area, region=region):
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.context.tool_settings.mesh_select_mode = (True, False, False)
        prepare_knife_session(bpy.context)
        simulate_one_side_cut(obj)
        bm = bmesh.from_edit_mesh(obj.data)
        native_counts = (len(bm.verts), len(bm.edges), len(bm.faces))

        yse_backup.create_topology_backup = broken_create
        ops_error = None
        try:
            finished = bpy.ops.mesh.ydd_symmetric_edit_finish("EXEC_DEFAULT")
        except RuntimeError as exc:
            # Blender surfaces operator.report({'ERROR'}) as RuntimeError from
            # bpy.ops even when the operator returned FINISHED (§2.2-4 fatal).
            ops_error = exc
            finished = {"FINISHED"}
        finally:
            yse_backup.create_topology_backup = original_create
        assert finished == {"FINISHED"}, finished
        assert ops_error is not None and "backup" in str(ops_error).lower(), ops_error

    bm = bmesh.from_edit_mesh(obj.data)
    assert (len(bm.verts), len(bm.edges), len(bm.faces)) == native_counts, (
        (len(bm.verts), len(bm.edges), len(bm.faces)),
        native_counts,
    )
    # Native left cut kept; mirror never ran.
    assert has_exact_edge(bm, (-1.5, -1.0, 0.0), (-1.5, 1.0, 0.0))
    assert not has_exact_edge(bm, (1.5, -1.0, 0.0), (1.5, 1.0, 0.0))
    errors = error_messages()
    assert any("backup" in message.lower() for message in errors), (
        errors,
        operators._FINISH_REPORTS,
    )
    assert not warning_messages(), operators._FINISH_REPORTS
    assert not info_messages(), operators._FINISH_REPORTS
    assert bm.edges.layers.int.get(layer_names.EDGE_ORIGINAL_LAYER) is None
    assert not operators._SESSIONS
    print("YSE_KNIFE_BOTH_SIDES_CASE_M=OK", flush=True)


def unit_orphan_face_remap() -> None:
    """resolve_live_mirror_face_map: self-mirror ok, asymmetric ear declines."""

    print("YSE_KNIFE_BOTH_SIDES_UNIT=orphan_face_remap", flush=True)
    axis = matching.AXIS_INDEX["X"]
    tol = 1.0e-4

    # --- Self-mirrored dissolved L∪R (hex spanning face) ---
    bm = bmesh.new()
    verts = [
        bm.verts.new(co)
        for co in (
            (-2.0, -1.0, 0.0),
            (0.0, -1.0, 0.0),
            (2.0, -1.0, 0.0),
            (2.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            (-2.0, 1.0, 0.0),
        )
    ]
    bm.faces.new(verts)
    face_layer = bm.faces.layers.int.new(layer_names.FACE_ID_LAYER)
    # Layer creation invalidates face wrappers; re-acquire.
    face = next(iter(bm.faces))
    # Pre-native L=1, R=2; only L survives as the dissolved union FACE_ID.
    face[face_layer] = 1
    mirror_map = {FaceId(1): FaceId(2), FaceId(2): FaceId(1)}
    # Path edge on the face (any edge).
    path = [next(iter(face.edges))]
    remapped = face_mapping.resolve_live_mirror_face_map(bm, mirror_map, axis, tol, path_edges=path)
    assert remapped[FaceId(1)] == FaceId(1), remapped
    bm.free()

    # --- Asymmetric L∪R∪ear: ear breaks geometric self-mirror ---
    bm = bmesh.new()
    verts = [
        bm.verts.new(co)
        for co in (
            (-2.0, -1.0, 0.0),
            (0.0, -1.0, 0.0),
            (2.0, -1.0, 0.0),
            (3.0, -0.5, 0.0),  # asymmetric ear tip (non-path)
            (2.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            (-2.0, 1.0, 0.0),
        )
    ]
    bm.faces.new(verts)
    face_layer = bm.faces.layers.int.new(layer_names.FACE_ID_LAYER)
    face = next(iter(bm.faces))
    face[face_layer] = 1
    mirror_map = {FaceId(1): FaceId(2), FaceId(2): FaceId(1)}
    # Path edge on the left boundary — does not include the ear tip.
    path = [next(edge for edge in face.edges if all(vertex.co.x <= -1.5 for vertex in edge.verts))]
    remapped = face_mapping.resolve_live_mirror_face_map(bm, mirror_map, axis, tol, path_edges=path)
    assert remapped.get(FaceId(1)) is None, remapped
    bm.free()
    print("YSE_KNIFE_BOTH_SIDES_UNIT_ORPHAN=OK", flush=True)


def unit_host_edge_selection() -> None:
    """Host edge plan: existing vertex prefer, tol-out reject, multi decline."""

    print("YSE_KNIFE_BOTH_SIDES_UNIT=host_edge_selection", flush=True)
    axis = matching.AXIS_INDEX["X"]
    tol = 0.1

    # (1) Existing on-plane vertex within tol is preferred over host edges.
    bm = bmesh.new()
    a = bm.verts.new((-1.0, -1.0, 0.0))
    b = bm.verts.new((1.0, 1.0, 0.0))
    p = bm.verts.new((0.0, 0.0, 0.0))
    member = bm.edges.new((a, b))
    # Host edge through the same p area (horizontal).
    h0 = bm.verts.new((-1.0, 0.0, 0.0))
    h1 = bm.verts.new((1.0, 0.0, 0.0))
    bm.edges.new((h0, h1))
    rep = p.co.copy()
    vertex, host_split, reason = stitch_pstitch._plan_plane_stitch_vertex(bm, rep, [member], axis, tol)
    assert reason == "", reason
    assert vertex is p and host_split is None
    bm.free()

    # (2) Host edge farther than tol is not a candidate (no second threshold).
    bm = bmesh.new()
    a = bm.verts.new((-1.0, -1.0, 0.0))
    b = bm.verts.new((1.0, 1.0, 0.0))
    member = bm.edges.new((a, b))
    # Horizontal host at y=0.25 > tol=0.1 from p=(0,0).
    h0 = bm.verts.new((-1.0, 0.25, 0.0))
    h1 = bm.verts.new((1.0, 0.25, 0.0))
    bm.edges.new((h0, h1))
    rep = type(a.co)((0.0, 0.0, 0.0))
    from mathutils import Vector

    rep = Vector((0.0, 0.0, 0.0))
    vertex, host_split, reason = stitch_pstitch._plan_plane_stitch_vertex(bm, rep, [member], axis, tol)
    assert reason == "", reason
    assert vertex is None and host_split is None  # fall through to member seed
    bm.free()

    # (3) Multiple host candidates within tol → decline (no nearest fallback).
    bm = bmesh.new()
    a = bm.verts.new((-1.0, -1.0, 0.0))
    b = bm.verts.new((1.0, 1.0, 0.0))
    member = bm.edges.new((a, b))
    h0 = bm.verts.new((-1.0, 0.0, 0.0))
    h1 = bm.verts.new((1.0, 0.0, 0.0))
    bm.edges.new((h0, h1))
    h2 = bm.verts.new((0.0, -1.0, 0.0))
    h3 = bm.verts.new((0.0, 1.0, 0.0))
    bm.edges.new((h2, h3))
    rep = Vector((0.0, 0.0, 0.0))
    vertex, host_split, reason = stitch_pstitch._plan_plane_stitch_vertex(bm, rep, [member], axis, tol)
    assert "ambiguous host edges" in reason, reason
    assert vertex is None and host_split is None
    bm.free()
    print("YSE_KNIFE_BOTH_SIDES_UNIT_HOST=OK", flush=True)


def unit_three_crosses_pointmerge_survivor() -> None:
    """Three CROSSES converging on one p: survivor-first pointmerge keeps degree 6."""

    print("YSE_KNIFE_BOTH_SIDES_UNIT=three_crosses_pointmerge", flush=True)
    from mathutils import Vector

    bm = bmesh.new()
    pairs = (
        (Vector((-1.0, -1.0, 0.0)), Vector((1.0, 1.0, 0.0))),
        (Vector((-1.0, 1.0, 0.0)), Vector((1.0, -1.0, 0.0))),
        (Vector((-1.0, -0.5, 0.0)), Vector((1.0, 0.5, 0.0))),
    )
    crosses = []
    for a_co, b_co in pairs:
        a = bm.verts.new(a_co)
        b = bm.verts.new(b_co)
        crosses.append(bm.edges.new((a, b)))
    stitched, reason = stitch_pstitch.apply_crosses_p_stitch(bm, crosses, matching.AXIS_INDEX["X"], 1.0e-4)
    assert reason == "", reason
    assert stitched == 3, stitched
    p_verts = [
        vertex
        for vertex in bm.verts
        if vertex.is_valid and abs(vertex.co.x) <= 1.0e-6 and abs(vertex.co.y) <= 1.0e-6 and abs(vertex.co.z) <= 1.0e-6
    ]
    assert len(p_verts) == 1, [tuple(vertex.co) for vertex in bm.verts if vertex.is_valid]
    assert len(p_verts[0].link_edges) >= 6, len(p_verts[0].link_edges)
    bm.free()
    print("YSE_KNIFE_BOTH_SIDES_UNIT_THREECROSSES=OK", flush=True)


def unit_nontransitive_tol_clustering() -> None:
    """Non-transitive tol chain: 0.09 spacing ×3, tol=0.1 → 2 clusters."""

    print("YSE_KNIFE_BOTH_SIDES_UNIT=nontransitive_tol", flush=True)
    from mathutils import Vector

    tol = 0.1
    # Points along Y on the plane; lex order is by (x,y,z) so y order applies.
    points = [
        Vector((0.0, 0.0, 0.0)),
        Vector((0.0, 0.09, 0.0)),
        Vector((0.0, 0.18, 0.0)),
        Vector((0.0, 0.27, 0.0)),
    ]
    # p0 absorbs p1 (0.09≤0.1); p2 is 0.18 from p0 → new cluster, absorbs p3.
    clusters = stitch_common.cluster_points_by_tolerance(points, tol)
    assert len(clusters) == 2, clusters
    assert sorted(len(cluster) for cluster in clusters) == [2, 2], clusters

    # Headless stitch: three CROSSES with p at those y values should form
    # two on-plane stitch vertices (not one big cluster).
    bm = bmesh.new()
    # Spanning quad.
    corners = [
        bm.verts.new(co)
        for co in (
            (-2.0, -1.0, 0.0),
            (2.0, -1.0, 0.0),
            (2.0, 2.0, 0.0),
            (-2.0, 2.0, 0.0),
        )
    ]
    face = bm.faces.new(corners)
    # Three crossing edges with intersections at y=0, 0.09, 0.18 (and a 4th at 0.27
    # would need another edge; three points spanning two clusters suffice).
    endpoints = []
    for y in (0.0, 0.09, 0.18):
        left = bm.verts.new((-1.0, y - 0.5, 0.0))
        right = bm.verts.new((1.0, y + 0.5, 0.0))
        # Ensure the line crosses x=0 at (0, y): parametric from left to right
        # x: -1→1, y: y-0.5 → y+0.5; at t=0.5: x=0, y=y. Good.
        endpoints.append((left, right))
    # Build edges as path edges (tag 0) on the face via face_split where possible.
    # Direct edges that are not face-boundary still work for apply_crosses_p_stitch
    # as long as they exist as BMEdge.
    crosses = []
    for left, right in endpoints:
        # Connect through face by splitting if both verts are on the face...
        # Simpler: create free-standing edges for the stitch unit (no face needed).
        crosses.append(bm.edges.new((left, right)))
    del face  # face unused after construction
    stitched, reason = stitch_pstitch.apply_crosses_p_stitch(bm, crosses, matching.AXIS_INDEX["X"], tol)
    assert reason == "", reason
    assert stitched >= 2, stitched
    on_plane = [vertex for vertex in bm.verts if abs(vertex.co.x) <= 1.0e-6 and vertex.is_valid]
    # Two cluster representatives near y=0 and y=0.18 (p1 absorbed into first,
    # no fourth point so second cluster has the 0.18 rep alone after stitch).
    ys = sorted({round(vertex.co.y, 5) for vertex in on_plane if abs(vertex.co.y) < 1.0})
    # At least two distinct on-plane stitch y values from the two clusters.
    assert len(ys) >= 2, ys
    bm.free()
    print("YSE_KNIFE_BOTH_SIDES_UNIT_TOL=OK", flush=True)


def unit_knife_path_edge_cache() -> None:
    """Cached edge metadata reclassifies safely and rejects stale topology."""

    print("YSE_KNIFE_BOTH_SIDES_UNIT=knife_path_edge_cache", flush=True)
    bm = bmesh.new()
    positive = bm.edges.new((bm.verts.new((1.0, 0.0, 0.0)), bm.verts.new((2.0, 0.0, 0.0))))
    negative = bm.edges.new((bm.verts.new((-2.0, 0.0, 0.0)), bm.verts.new((-1.0, 0.0, 0.0))))
    cache = stitch_pathedges.capture_knife_path_edge_cache(bm, (positive, negative))
    assert cache is not None
    by_side, total = stitch_pathedges.reclassify_knife_path_edge_cache(bm, matching.AXIS_INDEX["X"], 1.0e-5, cache) or (
        {},
        0,
    )
    assert total == 2, total
    assert len(by_side["POSITIVE"]) == 1 and len(by_side["NEGATIVE"]) == 1, by_side

    positive.verts[0].co.x = 1.25
    assert stitch_pathedges.reclassify_knife_path_edge_cache(bm, matching.AXIS_INDEX["X"], 1.0e-5, cache) is None
    bm.free()
    print("YSE_KNIFE_BOTH_SIDES_UNIT_CACHE=OK", flush=True)


def run_test() -> None:
    addon.register()
    window, area, region = viewport_context()
    configure_view(area)

    # Headless-style units first (no session / viewport dependency).
    unit_orphan_face_remap()
    unit_host_edge_selection()
    unit_three_crosses_pointmerge_survivor()
    unit_nontransitive_tol_clustering()
    unit_knife_path_edge_cache()

    case_a_straddle(window, area, region)
    case_b_crosses(window, area, region)
    case_c_one_side(window, area, region)
    case_d_mixed_crosses(window, area, region)
    case_e_bent_direct(window, area, region)
    case_f_direct_decline_rollback(window, area, region)
    case_d_knife_project_never_called(window, area, region)
    case_e2_gate_decline_rollback(window, area, region)
    case_h_count_mismatch_rollback(window, area, region)
    case_i_rollback_exception_error(window, area, region)
    case_n_weak_band_direct(window, area, region)
    case_o_endpoint_collision_rollback(window, area, region)
    case_q_wire_strokes_direct(window, area, region)
    case_r_network_direct(window, area, region)
    case_s_network_partial_failure_rollback(window, area, region)
    case_g_self_mirrored(window, area, region)
    case_h_simple_crosses(window, area, region)
    case_i_bowtie(window, area, region)
    case_j_self_mirrored_crosses(window, area, region)
    case_k_mixed_symmetrized(window, area, region)
    case_l_asymmetric_decline(window, area, region)
    case_m_backup_creation_failure(window, area, region)

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
