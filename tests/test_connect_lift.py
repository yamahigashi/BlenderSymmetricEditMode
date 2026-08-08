# SPDX-License-Identifier: GPL-3.0-or-later

"""GUI regression for Phase 3 Connect (J) lift — union semantics (contract §4.4).

Run with Blender's real window/event loop::

    blender --factory-startup --enable-event-simulate \\
        --python test_connect_lift.py

Cases:
  (a) PARTIAL path (A→B→A′, B′ unselected) → R ∪ ρ(R) fully symmetric.
  (b) Crossing chords on a symmetric face → native p-stitch, single p, symmetric.
  (c) Missing counterpart → native only + WARNING + FINISHED, no mirror edges.
  (d) Fault injection: second native is a no-op → full rollback + WARNING + FINISHED.
  (e) Equivariance: (M, I) and (ρM, ρI) results are mirrors (coords + face incidence).
  (f) Zig-zag history (selection self-mirrored, effect not) → mirror edges added.
  (g) Excess generation injection → rollback + WARNING.
  (h) CANCELLED injection (partial effect) → rollback + WARNING.
  (i) Empty R (EDGE select mode J) → WARNING + no mirror + FINISHED.
  (j) Backup creation failure → ERROR + FINISHED + native kept.
  (k) Undo oneness: success and rollback paths restore baseline in one ed.undo.
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
sys.path.insert(0, str(PACKAGE_PARENT))

import ydd_symmetric_edit as addon  # noqa: E402
from ydd_symmetric_edit import backup, core, replay  # noqa: E402

MARKER_OK = "YSE_CONNECT_LIFT_OK"
MARKER_FAILED = "YSE_CONNECT_LIFT_FAILED"
COORD_PRECISION = 5
TOLERANCE = 1.0e-4


def fail(message: str = "") -> None:
    if message:
        print(f"YSE_CONNECT_LIFT_ERROR={message}", flush=True)
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
    region_3d.view_distance = 8.0
    region_3d.update()


def clear_scene() -> None:
    for old in tuple(bpy.data.objects):
        bpy.data.objects.remove(old, do_unlink=True)
    for mesh in tuple(bpy.data.meshes):
        bpy.data.meshes.remove(mesh)


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


def _normalize_cycle(coords):
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


def assert_x_symmetric(bm, label: str = "") -> None:
    """Coords, edges, and face incidence are X-symmetric (test_knife_both_sides style)."""

    prefix = f"{label}: " if label else ""
    verts = vertex_coord_multiset(bm)
    edges = edge_coord_multiset(bm)
    assert verts == mirrored_vertex_multiset(bm), f"{prefix}vertex coords not X-symmetric: {verts}"
    assert edges == mirrored_edge_multiset(bm), f"{prefix}edges not X-symmetric: {edges}"
    faces = face_incidence_multiset(bm)
    mirrored_faces = mirrored_face_incidence_multiset(bm)
    assert faces == mirrored_faces, f"{prefix}face incidence not X-symmetric: {faces} vs {mirrored_faces}"


def has_edge_between(bm, a, b, precision: int = COORD_PRECISION) -> bool:
    key = tuple(sorted((coordinate_key(a, precision), coordinate_key(b, precision))))
    return key in edge_coord_multiset(bm)


def find_vertex(bm, expected, precision: int = COORD_PRECISION):
    key = coordinate_key(expected, precision)
    for vertex in bm.verts:
        if coordinate_key(vertex.co, precision) == key:
            return vertex
    raise AssertionError(f"vertex not found: {expected}")


def select_path(bm, coordinates) -> None:
    for face in bm.faces:
        face.select = False
    for edge in bm.edges:
        edge.select = False
    for vertex in bm.verts:
        vertex.select = False
    bm.select_history.clear()
    for coordinate in coordinates:
        vertex = find_vertex(bm, coordinate)
        vertex.select = True
        bm.select_history.add(vertex)


def make_object(name: str, vertices, faces):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata([tuple(v) for v in vertices], [], [tuple(f) for f in faces])
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    obj.use_mesh_mirror_x = True
    obj.use_mesh_mirror_y = False
    obj.use_mesh_mirror_z = False
    return obj


def enter_edit(window, area, region, obj):
    with bpy.context.temp_override(window=window, area=area, region=region):
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.context.tool_settings.mesh_select_mode = (True, False, False)
    return bmesh.from_edit_mesh(obj.data)


def leave_edit(window, area, region) -> None:
    with bpy.context.temp_override(window=window, area=area, region=region):
        bpy.ops.object.mode_set(mode="OBJECT")


def assert_no_temp_layers(bm) -> None:
    assert bm.edges.layers.int.get(core.EDGE_ORIGINAL_LAYER) is None
    assert bm.faces.layers.int.get(core.FACE_ID_LAYER) is None
    assert bm.verts.layers.int.get(core.VERT_BACKUP_ID_LAYER) is None


def mirror_coord(coordinate):
    x, y, z = coordinate
    return (-float(x), float(y), float(z))


def hexagon_coords():
    """Symmetric hexagon on XY, used by crossing / zig-zag cases (spike S4)."""

    return (
        (-2.0, 0.0, 0.0),
        (-1.0, -1.5, 0.0),
        (1.0, -1.5, 0.0),
        (2.0, 0.0, 0.0),
        (1.0, 1.5, 0.0),
        (-1.0, 1.5, 0.0),
    )


def warning_messages() -> list[str]:
    return [message for kind, message in replay._CONNECT_REPORTS if kind == "WARNING"]


def error_messages() -> list[str]:
    return [message for kind, message in replay._CONNECT_REPORTS if kind == "ERROR"]


def info_messages() -> list[str]:
    return [message for kind, message in replay._CONNECT_REPORTS if kind == "INFO"]


def _connected_via_plane(bm, a, b) -> bool:
    va = find_vertex(bm, a)
    vb = find_vertex(bm, b)
    if any(edge.other_vert(va) is vb for edge in va.link_edges):
        return True
    for edge in va.link_edges:
        mid = edge.other_vert(va)
        if abs(float(mid.co[0])) <= TOLERANCE and any(e.other_vert(mid) is vb for e in mid.link_edges):
            return True
    return False


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------


def case_a_partial_path(window, area, region) -> None:
    """PARTIAL A→B→A′ (B′ unselected) must still produce R ∪ ρ(R)."""

    print("YSE_CONNECT_LIFT_CASE=a_partial", flush=True)
    clear_scene()
    coords = hexagon_coords()
    obj = make_object("YSE_ConnectPartial", coords, (tuple(range(6)),))
    bm = enter_edit(window, area, region, obj)
    assert_x_symmetric(bm, "a baseline")

    select_path(bm, ((1.0, -1.5, 0.0), (1.0, 1.5, 0.0), (-1.0, -1.5, 0.0)))
    bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)

    with bpy.context.temp_override(window=window, area=area, region=region):
        result = bpy.ops.mesh.ydd_symmetric_edit_connect()
    assert result == {"FINISHED"}, result

    bm = bmesh.from_edit_mesh(obj.data)
    assert has_edge_between(bm, (1.0, -1.5, 0.0), (1.0, 1.5, 0.0)) or _connected_via_plane(
        bm, (1.0, -1.5, 0.0), (1.0, 1.5, 0.0)
    ), "missing source chord A-B"
    assert has_edge_between(bm, (1.0, 1.5, 0.0), (-1.0, -1.5, 0.0)) or _connected_via_plane(
        bm, (1.0, 1.5, 0.0), (-1.0, -1.5, 0.0)
    ), "missing source chord B-A′"
    assert has_edge_between(bm, (-1.0, -1.5, 0.0), (-1.0, 1.5, 0.0)) or _connected_via_plane(
        bm, (-1.0, -1.5, 0.0), (-1.0, 1.5, 0.0)
    ), "missing mirrored chord A′-B′"
    assert has_edge_between(bm, (-1.0, 1.5, 0.0), (1.0, -1.5, 0.0)) or _connected_via_plane(
        bm, (-1.0, 1.5, 0.0), (1.0, -1.5, 0.0)
    ), "missing mirrored chord B′-A"
    assert_x_symmetric(bm, "a result")
    assert_no_temp_layers(bm)
    leave_edit(window, area, region)
    print("YSE_CONNECT_LIFT_CASE_A=OK", flush=True)


def case_b_crossing(window, area, region) -> None:
    """Crossing path + mirror path → single on-plane p, full X-symmetry."""

    print("YSE_CONNECT_LIFT_CASE=b_crossing", flush=True)
    clear_scene()
    coords = hexagon_coords()
    obj = make_object("YSE_ConnectCrossing", coords, (tuple(range(6)),))
    bm = enter_edit(window, area, region, obj)
    assert_x_symmetric(bm, "b baseline")
    baseline_verts = len(bm.verts)

    # Path [(-1,-1.5), (1,1.5)]; mirror stage does [(1,-1.5), (-1,1.5)].
    select_path(bm, ((-1.0, -1.5, 0.0), (1.0, 1.5, 0.0)))
    bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)

    with bpy.context.temp_override(window=window, area=area, region=region):
        result = bpy.ops.mesh.ydd_symmetric_edit_connect()
    assert result == {"FINISHED"}, result

    bm = bmesh.from_edit_mesh(obj.data)
    origin_like = [
        v
        for v in bm.verts
        if abs(float(v.co[0])) <= TOLERANCE and abs(float(v.co[1])) <= TOLERANCE and abs(float(v.co[2])) <= TOLERANCE
    ]
    assert len(origin_like) == 1, f"expected single p at origin, got {[(tuple(v.co),) for v in origin_like]}"
    assert len(bm.verts) == baseline_verts + 1, (len(bm.verts), baseline_verts)
    p = origin_like[0]
    assert len(p.link_edges) == 4, len(p.link_edges)
    # p.link_faces: four triangles after two chords through one n-gon (or 4).
    assert len(p.link_faces) >= 2, len(p.link_faces)
    # A–p–B realized as novel edges through p (not a single A–B edge).
    a = find_vertex(bm, (-1.0, -1.5, 0.0))
    b = find_vertex(bm, (1.0, 1.5, 0.0))
    assert any(edge.other_vert(a) is p for edge in a.link_edges), "missing A–p"
    assert any(edge.other_vert(b) is p for edge in b.link_edges), "missing p–B"
    assert not any(edge.other_vert(a) is b for edge in a.link_edges), "A–B should be split via p"
    assert_x_symmetric(bm, "b result")
    assert_no_temp_layers(bm)
    leave_edit(window, area, region)
    print("YSE_CONNECT_LIFT_CASE_B=OK", flush=True)


def case_c_missing_counterpart(window, area, region) -> None:
    """Path vertex without a mirror counterpart → native only + WARNING."""

    print("YSE_CONNECT_LIFT_CASE=c_missing", flush=True)
    clear_scene()
    vertices = (
        (0.0, -1.0, 0.0),
        (1.0, -1.0, 0.0),  # A
        (1.0, 1.0, 0.0),  # B (no B′)
        (0.0, 1.0, 0.0),
        (-1.0, -1.0, 0.0),  # A′ only
    )
    faces = (
        (0, 1, 2, 3),
        (0, 3, 4),
    )
    obj = make_object("YSE_ConnectMissing", vertices, faces)
    bm = enter_edit(window, area, region, obj)
    # Diagonal B=(1,1) → plane bottom (0,-1). (0,-1) on-plane; B has no counterpart.
    select_path(bm, ((1.0, 1.0, 0.0), (0.0, -1.0, 0.0)))
    bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)

    with bpy.context.temp_override(window=window, area=area, region=region):
        result = bpy.ops.mesh.ydd_symmetric_edit_connect()
    assert result == {"FINISHED"}, result

    warnings = warning_messages()
    assert warnings, f"expected missing-counterpart WARNING, got {replay._CONNECT_REPORTS}"
    assert any("no mirror counterpart" in message for message in warnings), warnings
    assert not info_messages(), info_messages()

    bm = bmesh.from_edit_mesh(obj.data)
    assert has_edge_between(bm, (1.0, 1.0, 0.0), (0.0, -1.0, 0.0)), "native diagonal missing"
    assert not has_edge_between(bm, (-1.0, 1.0, 0.0), (0.0, -1.0, 0.0)), "mirror must not invent B′"
    assert vertex_coord_multiset(bm) != mirrored_vertex_multiset(bm)
    assert_no_temp_layers(bm)
    leave_edit(window, area, region)
    print("YSE_CONNECT_LIFT_CASE_C=OK", flush=True)


def case_d_second_native_noop(window, area, region) -> None:
    """Second native forced to no-op → rollback to native-only + WARNING."""

    print("YSE_CONNECT_LIFT_CASE=d_fault_noop", flush=True)
    clear_scene()
    vertices = (
        (0.0, -1.0, 0.0),
        (1.0, -1.0, 0.0),
        (1.0, 1.0, 0.0),
        (0.0, 1.0, 0.0),
        (-1.0, -1.0, 0.0),
        (-1.0, 1.0, 0.0),
    )
    faces = ((0, 1, 2, 3), (0, 3, 5, 4))
    obj = make_object("YSE_ConnectFaultNoop", vertices, faces)
    bm = enter_edit(window, area, region, obj)

    select_path(bm, ((1.0, -1.0, 0.0), (0.0, 1.0, 0.0)))
    bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)

    call_count = {"n": 0}
    original = replay._native_vert_connect_path

    def patched():
        call_count["n"] += 1
        if call_count["n"] == 1:
            return original()
        # Second call: pretend success but create nothing.
        return {"FINISHED"}

    replay._native_vert_connect_path = patched
    try:
        with bpy.context.temp_override(window=window, area=area, region=region):
            result = bpy.ops.mesh.ydd_symmetric_edit_connect()
    finally:
        replay._native_vert_connect_path = original
    assert result == {"FINISHED"}, result
    assert call_count["n"] >= 2, call_count

    warnings = warning_messages()
    assert warnings, f"expected rollback WARNING, got {replay._CONNECT_REPORTS}"
    assert any("rolled back" in message or "did not match" in message for message in warnings), warnings
    assert not info_messages(), f"success INFO must be absent after rollback: {info_messages()}"

    bm = bmesh.from_edit_mesh(obj.data)
    assert has_edge_between(bm, (1.0, -1.0, 0.0), (0.0, 1.0, 0.0)), "native diagonal rolled back incorrectly"
    assert not has_edge_between(bm, (-1.0, -1.0, 0.0), (0.0, 1.0, 0.0)), "mirror edge leaked after rollback"
    assert edge_coord_multiset(bm) != mirrored_edge_multiset(bm), "edges should remain asymmetric after rollback"
    assert_no_temp_layers(bm)
    leave_edit(window, area, region)
    print("YSE_CONNECT_LIFT_CASE_D=OK", flush=True)


def case_e_equivariance(window, area, region) -> None:
    """(M, I) and (ρM, ρI) produce results that are mirrors of each other."""

    print("YSE_CONNECT_LIFT_CASE=e_equivariance", flush=True)
    clear_scene()
    vertices = (
        (0.0, -1.0, 0.0),
        (1.0, -1.0, 0.0),
        (1.0, 1.0, 0.0),
        (0.0, 1.0, 0.0),
        (-1.0, -1.0, 0.0),
        (-1.0, 1.0, 0.0),
    )
    faces = ((0, 1, 2, 3), (0, 3, 5, 4))

    obj = make_object("YSE_ConnectEq1", vertices, faces)
    bm = enter_edit(window, area, region, obj)
    select_path(bm, ((1.0, -1.0, 0.0), (0.0, 1.0, 0.0)))
    bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
    with bpy.context.temp_override(window=window, area=area, region=region):
        r1 = bpy.ops.mesh.ydd_symmetric_edit_connect()
    assert r1 == {"FINISHED"}, r1
    bm = bmesh.from_edit_mesh(obj.data)
    result_edges_1 = edge_coord_multiset(bm)
    result_verts_1 = vertex_coord_multiset(bm)
    result_faces_1 = face_incidence_multiset(bm)
    leave_edit(window, area, region)

    clear_scene()
    obj2 = make_object("YSE_ConnectEq2", vertices, faces)
    bm = enter_edit(window, area, region, obj2)
    select_path(bm, (mirror_coord((1.0, -1.0, 0.0)), mirror_coord((0.0, 1.0, 0.0))))
    bmesh.update_edit_mesh(obj2.data, loop_triangles=False, destructive=False)
    with bpy.context.temp_override(window=window, area=area, region=region):
        r2 = bpy.ops.mesh.ydd_symmetric_edit_connect()
    assert r2 == {"FINISHED"}, r2
    bm = bmesh.from_edit_mesh(obj2.data)
    result_edges_2 = edge_coord_multiset(bm)
    result_verts_2 = vertex_coord_multiset(bm)

    assert result_verts_1 == Counter(mirror_key(k) for k in result_verts_2.elements())
    assert result_edges_1 == Counter(
        tuple(sorted(mirror_key(endpoint) for endpoint in edge)) for edge in result_edges_2.elements()
    )
    # Face incidence: ρ(result2 faces with winding flip) equals result1.
    mirrored_faces_2 = mirrored_face_incidence_multiset(bm)
    assert result_faces_1 == mirrored_faces_2, (result_faces_1, mirrored_faces_2)
    assert result_verts_1 == result_verts_2  # both fully symmetric → identical
    assert_x_symmetric(bm, "e result")
    assert_no_temp_layers(bm)
    leave_edit(window, area, region)
    print("YSE_CONNECT_LIFT_CASE_E=OK", flush=True)


def case_f_zigzag(window, area, region) -> None:
    """Zig-zag history: selection is self-mirrored but effect is not → still mirror."""

    print("YSE_CONNECT_LIFT_CASE=f_zigzag", flush=True)
    clear_scene()
    coords = hexagon_coords()
    obj = make_object("YSE_ConnectZigzag", coords, (tuple(range(6)),))
    bm = enter_edit(window, area, region, obj)
    assert_x_symmetric(bm, "f baseline")

    # History 2 → 5 → 1 → 4:
    #   (1,-1.5), (-1,1.5), (-1,-1.5), (1,1.5)
    path = (
        (1.0, -1.5, 0.0),
        (-1.0, 1.5, 0.0),
        (-1.0, -1.5, 0.0),
        (1.0, 1.5, 0.0),
    )
    select_path(bm, path)
    bm.verts.ensure_lookup_table()
    bm.verts.index_update()
    coords_now = tuple(Vector(v.co) for v in bm.verts)
    selected = [v.index for v in bm.verts if v.select]
    classification = core.classify_selection_overlap(
        coords_now,
        selected,
        axis_index=0,
        tolerance=TOLERANCE,
    )
    assert classification.overlap.name == "SELF_MIRRORED", classification.overlap
    bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)

    with bpy.context.temp_override(window=window, area=area, region=region):
        result = bpy.ops.mesh.ydd_symmetric_edit_connect()
    assert result == {"FINISHED"}, result

    # R after first native must be non-self-mirrored (zig-zag counterexample).
    last_r = replay._CONNECT_LAST_R
    assert last_r, "expected non-empty R from zig-zag native connect"

    # Coordinate-only self-mirror check: ρ(5-1)=4-2 is not in R.
    def _edge_key(a, b):
        return tuple(sorted((coordinate_key(a), coordinate_key(b))))

    r_keys = {_edge_key(a, b) for a, b in last_r}
    # Mirror of (-1,1.5)-(-1,-1.5) is (1,1.5)-(1,-1.5).
    mirror_only = _edge_key((1.0, 1.5, 0.0), (1.0, -1.5, 0.0))
    assert mirror_only not in r_keys, f"R unexpectedly already self-mirrored: {r_keys}"

    bm = bmesh.from_edit_mesh(obj.data)
    # Mirror stage must add the mirror-only vertical chord on +X.
    assert has_edge_between(bm, (1.0, 1.5, 0.0), (1.0, -1.5, 0.0)) or _connected_via_plane(
        bm, (1.0, 1.5, 0.0), (1.0, -1.5, 0.0)
    ), "mirror-only edge 4-2 missing"
    assert_x_symmetric(bm, "f result")
    assert_no_temp_layers(bm)
    leave_edit(window, area, region)
    print("YSE_CONNECT_LIFT_CASE_F=OK", flush=True)


def case_g_excess_generation(window, area, region) -> None:
    """Second native injects a surplus edge → rollback + WARNING."""

    print("YSE_CONNECT_LIFT_CASE=g_excess", flush=True)
    clear_scene()
    vertices = (
        (0.0, -1.0, 0.0),
        (1.0, -1.0, 0.0),
        (1.0, 1.0, 0.0),
        (0.0, 1.0, 0.0),
        (-1.0, -1.0, 0.0),
        (-1.0, 1.0, 0.0),
    )
    faces = ((0, 1, 2, 3), (0, 3, 5, 4))
    obj = make_object("YSE_ConnectExcess", vertices, faces)
    bm = enter_edit(window, area, region, obj)
    select_path(bm, ((1.0, -1.0, 0.0), (0.0, 1.0, 0.0)))
    bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)

    call_count = {"n": 0}
    original = replay._native_vert_connect_path

    def patched():
        call_count["n"] += 1
        if call_count["n"] == 1:
            return original()
        result = original()
        # Inject a surplus edge that is not in ρ(R): diagonal on +X that
        # already has the source edge — use -X bottom to +X top via plane?
        # Simpler: add edge between (-1,-1) and (-1,1) if not present as "extra"
        # after a successful mirror of (1,-1)-(0,1) → (-1,-1)-(0,1).
        # Add an additional edge (1,1)-(0,-1) which is not expected.
        bm_live = bmesh.from_edit_mesh(obj.data)
        try:
            va = next(v for v in bm_live.verts if coordinate_key(v.co) == coordinate_key((1.0, 1.0, 0.0)))
            vb = next(v for v in bm_live.verts if coordinate_key(v.co) == coordinate_key((0.0, -1.0, 0.0)))
            if not any(e.other_vert(va) is vb for e in va.link_edges):
                # May fail if no shared face; force via bmesh.ops.connect_vert_pair if needed.
                try:
                    bm_live.edges.new((va, vb))
                except ValueError:
                    bmesh.ops.connect_vert_pair(bm_live, verts=[va, vb])
            bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=True)
        except Exception:
            traceback.print_exc()
        return result

    replay._native_vert_connect_path = patched
    try:
        with bpy.context.temp_override(window=window, area=area, region=region):
            result = bpy.ops.mesh.ydd_symmetric_edit_connect()
    finally:
        replay._native_vert_connect_path = original
    assert result == {"FINISHED"}, result
    assert call_count["n"] >= 2, call_count

    warnings = warning_messages()
    assert warnings, f"expected excess rollback WARNING, got {replay._CONNECT_REPORTS}"
    assert any("rolled back" in message or "did not match" in message for message in warnings), warnings

    bm = bmesh.from_edit_mesh(obj.data)
    assert has_edge_between(bm, (1.0, -1.0, 0.0), (0.0, 1.0, 0.0)), "native kept"
    assert not has_edge_between(bm, (-1.0, -1.0, 0.0), (0.0, 1.0, 0.0)), "mirror rolled back"
    assert not has_edge_between(bm, (1.0, 1.0, 0.0), (0.0, -1.0, 0.0)), "surplus must not remain"
    assert_no_temp_layers(bm)
    leave_edit(window, area, region)
    print("YSE_CONNECT_LIFT_CASE_G=OK", flush=True)


def case_h_cancelled_injection(window, area, region) -> None:
    """Second native creates effect then returns CANCELLED → rollback + WARNING."""

    print("YSE_CONNECT_LIFT_CASE=h_cancelled", flush=True)
    clear_scene()
    vertices = (
        (0.0, -1.0, 0.0),
        (1.0, -1.0, 0.0),
        (1.0, 1.0, 0.0),
        (0.0, 1.0, 0.0),
        (-1.0, -1.0, 0.0),
        (-1.0, 1.0, 0.0),
    )
    faces = ((0, 1, 2, 3), (0, 3, 5, 4))
    obj = make_object("YSE_ConnectCancelled", vertices, faces)
    bm = enter_edit(window, area, region, obj)
    select_path(bm, ((1.0, -1.0, 0.0), (0.0, 1.0, 0.0)))
    bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)

    call_count = {"n": 0}
    original = replay._native_vert_connect_path

    def patched():
        call_count["n"] += 1
        if call_count["n"] == 1:
            return original()
        original()  # apply real mirror effect
        return {"CANCELLED"}

    replay._native_vert_connect_path = patched
    try:
        with bpy.context.temp_override(window=window, area=area, region=region):
            result = bpy.ops.mesh.ydd_symmetric_edit_connect()
    finally:
        replay._native_vert_connect_path = original
    assert result == {"FINISHED"}, result
    assert call_count["n"] >= 2, call_count

    warnings = warning_messages()
    assert warnings, f"expected CANCELLED rollback WARNING, got {replay._CONNECT_REPORTS}"
    assert any("CANCELLED" in message or "rolled back" in message for message in warnings), warnings
    assert not info_messages(), info_messages()

    bm = bmesh.from_edit_mesh(obj.data)
    assert has_edge_between(bm, (1.0, -1.0, 0.0), (0.0, 1.0, 0.0)), "native kept"
    assert not has_edge_between(bm, (-1.0, -1.0, 0.0), (0.0, 1.0, 0.0)), "mirror rolled back on CANCELLED"
    assert_no_temp_layers(bm)
    leave_edit(window, area, region)
    print("YSE_CONNECT_LIFT_CASE_H=OK", flush=True)


def case_i_empty_r_edge_mode(window, area, region) -> None:
    """EDGE select mode J → silent native no-op → WARNING + FINISHED, no mirror."""

    print("YSE_CONNECT_LIFT_CASE=i_empty_r", flush=True)
    clear_scene()
    vertices = (
        (0.0, -1.0, 0.0),
        (1.0, -1.0, 0.0),
        (1.0, 1.0, 0.0),
        (0.0, 1.0, 0.0),
        (-1.0, -1.0, 0.0),
        (-1.0, 1.0, 0.0),
    )
    faces = ((0, 1, 2, 3), (0, 3, 5, 4))
    obj = make_object("YSE_ConnectEmptyR", vertices, faces)
    bm = enter_edit(window, area, region, obj)

    # EDGE select mode: vert_connect_path is a silent no-op (contract §4.4-6).
    with bpy.context.temp_override(window=window, area=area, region=region):
        bpy.context.tool_settings.mesh_select_mode = (False, True, False)
    for face in bm.faces:
        face.select = False
    for vertex in bm.verts:
        vertex.select = False
    for edge in bm.edges:
        edge.select = False
    # Select one edge; history empty / no vertex path.
    edge = next(e for e in bm.edges if all(abs(float(v.co[0]) - 1.0) < 0.1 for v in e.verts))
    edge.select = True
    bm.select_history.clear()
    bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
    baseline_edges = edge_coord_multiset(bm)

    with bpy.context.temp_override(window=window, area=area, region=region):
        result = bpy.ops.mesh.ydd_symmetric_edit_connect()
    assert result == {"FINISHED"}, result

    warnings = warning_messages()
    assert warnings, f"expected empty-R WARNING, got {replay._CONNECT_REPORTS}"
    assert any("native connect created no edges" in message for message in warnings), warnings
    assert not info_messages(), info_messages()

    bm = bmesh.from_edit_mesh(obj.data)
    assert edge_coord_multiset(bm) == baseline_edges, "mesh must be unchanged"
    assert_no_temp_layers(bm)
    # Restore vertex select mode for subsequent cases.
    with bpy.context.temp_override(window=window, area=area, region=region):
        bpy.context.tool_settings.mesh_select_mode = (True, False, False)
    leave_edit(window, area, region)
    print("YSE_CONNECT_LIFT_CASE_I=OK", flush=True)


def case_j_backup_failure(window, area, region) -> None:
    """Backup creation failure → ERROR + FINISHED + native kept."""

    print("YSE_CONNECT_LIFT_CASE=j_backup_fail", flush=True)
    clear_scene()
    vertices = (
        (0.0, -1.0, 0.0),
        (1.0, -1.0, 0.0),
        (1.0, 1.0, 0.0),
        (0.0, 1.0, 0.0),
        (-1.0, -1.0, 0.0),
        (-1.0, 1.0, 0.0),
    )
    faces = ((0, 1, 2, 3), (0, 3, 5, 4))
    obj = make_object("YSE_ConnectBackupFail", vertices, faces)
    bm = enter_edit(window, area, region, obj)
    select_path(bm, ((1.0, -1.0, 0.0), (0.0, 1.0, 0.0)))
    bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)

    original = backup.create_topology_backup

    def boom(_bm):
        raise RuntimeError("injected backup failure")

    backup.create_topology_backup = boom
    result = None
    raised = None
    try:
        with bpy.context.temp_override(window=window, area=area, region=region):
            try:
                result = bpy.ops.mesh.ydd_symmetric_edit_connect()
            except RuntimeError as exc:
                # Blender surfaces operator.report({ERROR}) as a Python
                # RuntimeError even when the operator returned FINISHED.
                raised = exc
                result = {"FINISHED"}
    finally:
        backup.create_topology_backup = original
    assert result == {"FINISHED"}, (result, raised)

    errors = error_messages()
    assert errors, f"expected ERROR on backup failure, got {replay._CONNECT_REPORTS}"
    assert any("backup" in message.lower() for message in errors), errors

    bm = bmesh.from_edit_mesh(obj.data)
    assert has_edge_between(bm, (1.0, -1.0, 0.0), (0.0, 1.0, 0.0)), "native must be kept"
    assert not has_edge_between(bm, (-1.0, -1.0, 0.0), (0.0, 1.0, 0.0)), "mirror must not apply without backup"
    assert_no_temp_layers(bm)
    leave_edit(window, area, region)
    print("YSE_CONNECT_LIFT_CASE_J=OK", flush=True)


def _active_mesh():
    obj = bpy.context.view_layer.objects.active
    if obj is not None and obj.type == "MESH":
        return obj
    for candidate in bpy.data.objects:
        if candidate.type == "MESH":
            return candidate
    return None


def case_k_undo_oneness(window, area, region) -> None:
    """Success and rollback paths: one ed.undo restores baseline (Cube).

    Uses ``primitive_cube_add`` + ``ed.undo_push`` (same pattern as
    test_connect_route). After ``ed.undo`` the previous Object RNA is dead —
    re-fetch via the view layer.
    """

    print("YSE_CONNECT_LIFT_CASE=k_undo", flush=True)

    def prepare_cube():
        clear_scene()
        with bpy.context.temp_override(window=window, area=area, region=region):
            bpy.ops.mesh.primitive_cube_add()
        obj = _active_mesh()
        assert obj is not None
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        obj.use_mesh_mirror_x = True
        obj.use_mesh_mirror_y = False
        obj.use_mesh_mirror_z = False
        return obj

    # --- success path ---
    obj = prepare_cube()
    with bpy.context.temp_override(window=window, area=area, region=region):
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.context.tool_settings.mesh_select_mode = (True, False, False)
        bpy.ops.ed.undo_push(message="YSE Connect Lift undo baseline success")
        bm = bmesh.from_edit_mesh(obj.data)
        baseline_counts = (len(bm.verts), len(bm.edges), len(bm.faces))
        baseline_edges = edge_coord_multiset(bm)
        select_path(bm, ((1.0, -1.0, -1.0), (1.0, 1.0, 1.0)))
        bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
        result = bpy.ops.mesh.ydd_symmetric_edit_connect()
    assert result == {"FINISHED"}, result
    obj = _active_mesh()
    assert obj is not None
    bm = bmesh.from_edit_mesh(obj.data)
    assert has_edge_between(bm, (1.0, -1.0, -1.0), (1.0, 1.0, 1.0))
    assert has_edge_between(bm, (-1.0, -1.0, -1.0), (-1.0, 1.0, 1.0))
    with bpy.context.temp_override(window=window, area=area, region=region):
        undo_result = bpy.ops.ed.undo()
    assert undo_result == {"FINISHED"}, undo_result
    obj = _active_mesh()
    assert obj is not None
    bm = enter_edit(window, area, region, obj)
    assert (len(bm.verts), len(bm.edges), len(bm.faces)) == baseline_counts
    assert edge_coord_multiset(bm) == baseline_edges
    leave_edit(window, area, region)

    # --- rollback path (second native no-op) ---
    obj = prepare_cube()
    call_count = {"n": 0}
    original = replay._native_vert_connect_path

    def patched():
        call_count["n"] += 1
        if call_count["n"] == 1:
            return original()
        return {"FINISHED"}

    with bpy.context.temp_override(window=window, area=area, region=region):
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.context.tool_settings.mesh_select_mode = (True, False, False)
        bpy.ops.ed.undo_push(message="YSE Connect Lift undo baseline rollback")
        bm = bmesh.from_edit_mesh(obj.data)
        baseline_counts = (len(bm.verts), len(bm.edges), len(bm.faces))
        baseline_edges = edge_coord_multiset(bm)
        select_path(bm, ((1.0, -1.0, -1.0), (1.0, 1.0, 1.0)))
        bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
        replay._native_vert_connect_path = patched
        try:
            result = bpy.ops.mesh.ydd_symmetric_edit_connect()
        finally:
            replay._native_vert_connect_path = original
    assert result == {"FINISHED"}, result
    assert call_count["n"] >= 2, call_count
    obj = _active_mesh()
    assert obj is not None
    bm = enter_edit(window, area, region, obj)
    assert has_edge_between(bm, (1.0, -1.0, -1.0), (1.0, 1.0, 1.0)), "native kept after rollback"
    assert not has_edge_between(bm, (-1.0, -1.0, -1.0), (-1.0, 1.0, 1.0)), "mirror rolled back"
    with bpy.context.temp_override(window=window, area=area, region=region):
        undo_result = bpy.ops.ed.undo()
    assert undo_result == {"FINISHED"}, undo_result
    obj = _active_mesh()
    assert obj is not None
    bm = enter_edit(window, area, region, obj)
    assert (len(bm.verts), len(bm.edges), len(bm.faces)) == baseline_counts
    assert edge_coord_multiset(bm) == baseline_edges
    leave_edit(window, area, region)
    print("YSE_CONNECT_LIFT_CASE_K=OK", flush=True)


def run_all():
    try:
        addon.register()
        window, area, region = viewport_context()
        configure_view(area)
        scene = bpy.context.scene
        if hasattr(scene, "ydd_symmetric_edit"):
            scene.ydd_symmetric_edit.tolerance = TOLERANCE

        case_a_partial_path(window, area, region)
        case_b_crossing(window, area, region)
        case_c_missing_counterpart(window, area, region)
        case_d_second_native_noop(window, area, region)
        case_e_equivariance(window, area, region)
        case_f_zigzag(window, area, region)
        case_g_excess_generation(window, area, region)
        case_h_cancelled_injection(window, area, region)
        case_i_empty_r_edge_mode(window, area, region)
        case_j_backup_failure(window, area, region)
        case_k_undo_oneness(window, area, region)

        print(MARKER_OK, flush=True)
        addon.unregister()
        bpy.ops.wm.quit_blender()
    except BaseException:
        fail()
    return None


bpy.context.preferences.view.show_splash = False
bpy.context.preferences.view.smooth_view = 0
bpy.context.preferences.view.use_save_prompt = False
bpy.context.preferences.filepaths.use_auto_save_temporary_files = False
bpy.app.timers.register(run_all, first_interval=0.25)
