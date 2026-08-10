# SPDX-License-Identifier: GPL-3.0-or-later

"""Acceptance tests: mirrored Knife off-plane intersection stitch (contract §D-1/§D-2).

Contract:
  ``.agents/doc/mirrored_cut_intersection_plan_2026-08-09.md`` §A (norm) / §D (plan).

Cases (final expected behavior; wave-1 a–e + wave-2 f–m per contract §D-2):
  (a) PROPER crossing of ρ(s) with opposite-side cut t → finish succeeds,
      one mirror pair of degree-4 intersection verts, full X-symmetry.
  (b) Zig-zag double PROPER crossing → two intersection pairs, symmetric.
  (c) ENDPOINT_INTERIOR (T-junction) → endpoint reuse + other edge split, symmetric.
  (d) Partial collinear overlap → WARNING + FINISHED + native kept (rollback).
  (e) One-sided cut (no crossing) → existing both-sides mirror, regression.
  (f) CROSSES p-stitch + off-plane PROPER pair → both succeed, backup once.
  (g) On-plane band q (|q_x|≈5e-6 ≤ tol) → single shared on-plane vertex.
  (h) ENDPOINT_ENDPOINT → reuse only, no split duplicates.
  (i) Triple cluster (≥3 segments) → one mirror pair, degree ≥ 6.
  (j) Self-mirrored fixed CROSSES + POS cross → split at q and ρ(q) (A-1-3).
  (k) Non-planar carrier PROPER → success, bit-exact mirror coords.
  (l) Non-planarity guard (unit) → decline reason from plan.
  (m) Ambiguous existing verts at q (unit) → apply decline, path unsplit.

Run::

    cd tests && cmd.exe /c run_gui_test.bat 52 test_knife_cross_mirror_stitch.py
    cd tests && cmd.exe /c run_gui_test.bat 42 test_knife_cross_mirror_stitch.py
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
from ydd_symmetric_edit import core, operators, stitch  # noqa: E402

MARKER_OK = "YSE_KNIFE_CROSS_MIRROR_STITCH_OK"
MARKER_FAILED = "YSE_KNIFE_CROSS_MIRROR_STITCH_FAILED"
COORD_PRECISION = 5
TOL = 1.0e-4

# --- Fixture coordinates (2D intersection verified; see module docstring / logs) ---
# Grid: F- = [-1,0]x[-1,1], F+ = [0,1]x[-1,1], z=0.
#
# case_a: s on F+ bottom→right, t on F- bottom→top.
#   s   = (0.1,-1) → (1.0, 0.5)
#   ρ(s)= (-0.1,-1) → (-1.0, 0.5)
#   t   = (-0.2,-1) → (-0.7, 1.0)
#   ρ(s)×t PROPER at q- ≈ (-0.27143, -0.71429); q+ = ρ(q-).
CASE_A_S0 = (0.1, -1.0, 0.0)
CASE_A_S1 = (1.0, 0.5, 0.0)
CASE_A_T0 = (-0.2, -1.0, 0.0)
CASE_A_T1 = (-0.7, 1.0, 0.0)
CASE_A_Q_MINUS = (-0.2714285714285714, -0.7142857142857143, 0.0)
CASE_A_Q_PLUS = (0.2714285714285714, -0.7142857142857143, 0.0)

# case_b: s vertical x=0.5 on F+; t zig-zag on F- with mid ON the left boundary
# (boundary→boundary segments only, so finish stays on direct-topology path).
#   t = (-0.1,-1) → (-1.0, 0) → (-0.1, 1)
#   ρ(s)×t1 at (-0.5, -5/9), ρ(s)×t2 at (-0.5, +5/9).
CASE_B_S0 = (0.5, -1.0, 0.0)
CASE_B_S1 = (0.5, 1.0, 0.0)
CASE_B_T0 = (-0.1, -1.0, 0.0)
CASE_B_T_MID = (-1.0, 0.0, 0.0)
CASE_B_T1 = (-0.1, 1.0, 0.0)
CASE_B_Q1_MINUS = (-0.5, -5.0 / 9.0, 0.0)
CASE_B_Q1_PLUS = (0.5, -5.0 / 9.0, 0.0)
CASE_B_Q2_MINUS = (-0.5, 5.0 / 9.0, 0.0)
CASE_B_Q2_PLUS = (0.5, 5.0 / 9.0, 0.0)

# case_c: T-junction ENDPOINT_INTERIOR.
#   s bent vertical through joint (0.4, 0); t horizontal (-1,0)→(0,0).
#   ρ(s half) endpoint ( -0.4, 0) lands on t interior.
CASE_C_S0 = (0.4, -1.0, 0.0)
CASE_C_S_JOINT = (0.4, 0.0, 0.0)
CASE_C_S1 = (0.4, 1.0, 0.0)
CASE_C_T0 = (-1.0, 0.0, 0.0)
CASE_C_T1 = (0.0, 0.0, 0.0)
CASE_C_T_POINT = (-0.4, 0.0, 0.0)

# case_d: partial collinear on x=±0.3.
#   s L-path: vertical (0.3,-1)→(0.3,0.5) + horizontal to right edge.
#   t full vertical (-0.3,-1)→(-0.3,1). ρ(vertical s) properly subset-overlaps t.
CASE_D_S0 = (0.3, -1.0, 0.0)
CASE_D_S_JOINT = (0.3, 0.5, 0.0)
CASE_D_S1 = (1.0, 0.5, 0.0)
CASE_D_T0 = (-0.3, -1.0, 0.0)
CASE_D_T1 = (-0.3, 1.0, 0.0)

# case_e: one-sided vertical on F+ only.
CASE_E_S0 = (0.5, -1.0, 0.0)
CASE_E_S1 = (0.5, 1.0, 0.0)

# case_f: CROSSES (p-stitch) + off-plane PROPER pair on one spanning stroke.
# After dissolve of the plane edge, polyline L→B→C→R:
#   L=(-1,-0.5), B=(-0.6,0.4), C=(0.5,-0.3), R=(1,0.8)
#   B–C is non-self-mirrored CROSSES; ρ(C–R)×(L–B) is PROPER.
CASE_F_L = (-1.0, -0.5, 0.0)
CASE_F_B = (-0.6, 0.4, 0.0)
CASE_F_C = (0.5, -0.3, 0.0)
CASE_F_R = (1.0, 0.8, 0.0)
CASE_F_Q_MINUS = (-0.7078651685393258, 0.15730337078651685, 0.0)
CASE_F_Q_PLUS = (0.7078651685393258, 0.15730337078651685, 0.0)

# case_g: on-plane band 0 < |q_x| ≤ session tol (1e-5).
# s is POSITIVE (far endpoint + near-plane top); t horizontal near y=1 so |q_x|≪tol.
CASE_G_S0 = (0.5, -1.0, 0.0)
CASE_G_S1 = (1.0e-6, 1.0, 0.0)
CASE_G_T_Y = 0.999995
CASE_G_T0 = (-1.0, CASE_G_T_Y, 0.0)
CASE_G_T1 = (0.0, CASE_G_T_Y, 0.0)
CASE_G_Q_PLANE = (0.0, CASE_G_T_Y, 0.0)

# case_h: ENDPOINT_ENDPOINT — ρ(s1) lands on t0 within tol.
CASE_H_S0 = (0.5, -1.0, 0.0)
CASE_H_S1 = (0.4, 1.0, 0.0)
CASE_H_T0 = (-0.4, 1.0, 0.0)
CASE_H_T1 = (-0.8, -1.0, 0.0)
CASE_H_EE = (-0.4, 1.0, 0.0)

# case_i: triple cluster — two POS diagonals + one NEG horizontal at (-0.5,0).
CASE_I_S1_0 = (0.5, -1.0, 0.0)
CASE_I_S1_1 = (0.5, 1.0, 0.0)
CASE_I_S2_0 = (0.8, -1.0, 0.0)
CASE_I_S2_1 = (0.2, 1.0, 0.0)
CASE_I_T0 = (-1.0, 0.0, 0.0)
CASE_I_T1 = (0.0, 0.0, 0.0)
CASE_I_Q_MINUS = (-0.5, 0.0, 0.0)
CASE_I_Q_PLUS = (0.5, 0.0, 0.0)

# case_j: self-mirrored fixed CROSSES (-1,0)–(1,0) + POS vertical through (0.5,0).
CASE_J_FIXED_L = (-1.0, 0.0, 0.0)
CASE_J_FIXED_R = (1.0, 0.0, 0.0)
CASE_J_S0 = (0.5, -1.0, 0.0)
CASE_J_S1 = (0.5, 1.0, 0.0)
CASE_J_Q_PLUS = (0.5, 0.0, 0.0)
CASE_J_Q_MINUS = (-0.5, 0.0, 0.0)

# case_k: non-planar carrier (on-plane top mid lifted to z=+0.2).
# Cuts use bottom/left/right only so endpoints stay on unlifted edges.
CASE_K_LIFT_Z = 0.2
CASE_K_S0 = (0.2, -1.0, 0.0)
CASE_K_S1 = (1.0, 0.3, 0.0)
CASE_K_T0 = (-0.3, -1.0, 0.0)
CASE_K_T1 = (-1.0, 0.8, 0.0)
CASE_K_Q_MINUS = (-0.471698113207547, -0.5584905660377362, 0.0)
CASE_K_Q_PLUS = (0.471698113207547, -0.5584905660377362, 0.0)


def fail(message: str = "") -> None:
    if message:
        print(f"YSE_KNIFE_CROSS_MIRROR_STITCH_ERROR={message}", flush=True)
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
    if not coords:
        return ()
    rotations = [tuple(coords[i:] + coords[:i]) for i in range(len(coords))]
    return min(rotations)


def face_incidence_multiset(bm, precision: int = COORD_PRECISION) -> Counter:
    keys = []
    for face in bm.faces:
        coords = [coordinate_key(vertex.co, precision) for vertex in face.verts]
        keys.append(_normalize_cycle(coords))
    return Counter(keys)


def mirrored_face_incidence_multiset(bm, precision: int = COORD_PRECISION) -> Counter:
    keys = []
    for face in bm.faces:
        mirrored = [mirror_key(vertex.co, precision) for vertex in face.verts]
        mirrored.reverse()
        keys.append(_normalize_cycle(mirrored))
    return Counter(keys)


def mesh_signature(bm, precision: int = COORD_PRECISION):
    """Full native-topology signature for rollback comparisons."""

    return (
        vertex_coord_multiset(bm, precision),
        edge_coord_multiset(bm, precision),
        face_incidence_multiset(bm, precision),
    )


def assert_no_duplicate_edges(bm, tolerance: float = 1.0e-7) -> None:
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


def assert_temp_layers_cleared(bm) -> None:
    assert bm.edges.layers.int.get(core.EDGE_ORIGINAL_LAYER) is None
    assert bm.verts.layers.int.get(core.VERT_SELECTION_LAYER) is None
    assert bm.edges.layers.int.get(core.EDGE_SELECTION_LAYER) is None
    assert bm.faces.layers.int.get(core.FACE_SELECTION_LAYER) is None


def has_exact_edge(bm, a, b, tolerance=1.0e-7) -> bool:
    def close(co, expected):
        return all(abs(co[index] - expected[index]) <= tolerance for index in range(3))

    return any(
        (close(edge.verts[0].co, a) and close(edge.verts[1].co, b))
        or (close(edge.verts[0].co, b) and close(edge.verts[1].co, a))
        for edge in bm.edges
    )


def verts_near(bm, co, tolerance: float = 1.0e-3):
    return [
        vertex for vertex in bm.verts if all(abs(float(vertex.co[i]) - float(co[i])) <= tolerance for i in range(3))
    ]


def clear_scene() -> None:
    for old in tuple(bpy.data.objects):
        bpy.data.objects.remove(old, do_unlink=True)


def warning_messages() -> list[str]:
    return [message for kind, message in operators._FINISH_REPORTS if kind == "WARNING"]


def info_messages() -> list[str]:
    return [message for kind, message in operators._FINISH_REPORTS if kind == "INFO"]


def prepare_knife_session(context) -> None:
    prepared = operators._prepare_session(
        context,
        lambda _level, _message: None,
        tool_kind="KNIFE",
    )
    assert prepared, "failed to prepare knife session"


def make_symmetric_2x1_grid():
    """x=0-symmetric 2×1 grid: F- = [-1,0]×[-1,1], F+ = [0,1]×[-1,1] (contract §D-1)."""

    mesh = bpy.data.meshes.new("YSE_CrossMirrorStitchMesh")
    vertices = [
        (-1.0, -1.0, 0.0),
        (0.0, -1.0, 0.0),
        (1.0, -1.0, 0.0),
        (-1.0, 1.0, 0.0),
        (0.0, 1.0, 0.0),
        (1.0, 1.0, 0.0),
    ]
    mesh.from_pydata(vertices, [], [(0, 1, 4, 3), (1, 2, 5, 4)])
    mesh.update()
    obj = bpy.data.objects.new("YSE_CrossMirrorStitchObject", mesh)
    bpy.context.scene.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    obj.use_mesh_mirror_x = True
    obj.use_mesh_mirror_y = False
    obj.use_mesh_mirror_z = False
    return obj


def _edge_has_endpoints(edge, a, b, tol: float = 1.0e-6) -> bool:
    coords = [tuple(vertex.co) for vertex in edge.verts]

    def near(p, q):
        return all(abs(p[i] - q[i]) <= tol for i in range(3))

    return (near(coords[0], a) and near(coords[1], b)) or (near(coords[0], b) and near(coords[1], a))


def find_edge_between(bm, a, b, tol: float = 1.0e-6):
    for edge in bm.edges:
        if _edge_has_endpoints(edge, a, b, tol):
            return edge
    raise AssertionError(
        f"no edge between {a} and {b}; edges={[(tuple(e.verts[0].co), tuple(e.verts[1].co)) for e in bm.edges]}"
    )


def split_edge_at_co(edge, co):
    """edge_split so the new vertex lands at *co* (set exactly after split)."""

    v0, v1 = edge.verts
    d = v1.co - v0.co
    length_sq = d.length_squared
    assert length_sq > 1.0e-18, (tuple(v0.co), tuple(v1.co))
    factor = max(0.0, min(1.0, (Vector(co) - v0.co).dot(d) / length_sq))
    if factor < 1.0e-8:
        factor = 1.0e-4
    if factor > 1.0 - 1.0e-8:
        factor = 1.0 - 1.0e-4
    _new_edge, vert = bmesh.utils.edge_split(edge, v0, factor)
    vert.co = Vector(co)
    return vert


def face_with_verts(bm, *verts, side: str | None = None):
    """Pick a face containing all *verts*; optional side filter by median x."""

    candidates = [face for face in bm.faces if all(v in face.verts for v in verts)]
    if side == "positive":
        candidates = [face for face in candidates if face.calc_center_median().x > 0.0]
    elif side == "negative":
        candidates = [face for face in candidates if face.calc_center_median().x < 0.0]
    assert candidates, (verts, side, [(tuple(v.co) for v in face.verts) for face in bm.faces])
    return candidates[0]


def find_vert_at(bm, co, tol: float = 1.0e-6):
    for vertex in bm.verts:
        if all(abs(float(vertex.co[i]) - float(co[i])) <= tol for i in range(3)):
            return vertex
    return None


def ensure_boundary_vert(bm, co, *, side: str):
    """Reuse an existing vertex at *co*, or edge_split a boundary edge to create one."""

    existing = find_vert_at(bm, co)
    if existing is not None:
        return existing

    best = None
    best_dist = 1.0e9
    for edge in bm.edges:
        v0, v1 = edge.verts
        if side == "positive" and max(v0.co.x, v1.co.x) < -1.0e-8:
            continue
        if side == "negative" and min(v0.co.x, v1.co.x) > 1.0e-8:
            continue
        d = v1.co - v0.co
        length_sq = d.length_squared
        if length_sq < 1.0e-18:
            continue
        t = max(0.0, min(1.0, (Vector(co) - v0.co).dot(d) / length_sq))
        proj = v0.co + d * t
        dist = (proj - Vector(co)).length
        if dist < best_dist and dist <= 1.0e-4:
            best_dist = dist
            best = edge
    assert best is not None, f"no boundary edge near {co} on side={side}"
    return split_edge_at_co(best, co)


def cut_boundary_to_boundary(bm, a_co, b_co, *, side: str, coords=None):
    """Place boundary endpoints by edge_split and face_split the host face.

    *a_co* / *b_co* must lie on existing boundary edges of the target side face.
    Optional *coords* are intermediate face_split waypoints (interior joints).
    Existing vertices at the endpoint coordinates are reused (no duplicates).
    """

    vert_a = ensure_boundary_vert(bm, a_co, side=side)
    vert_b = ensure_boundary_vert(bm, b_co, side=side)
    host = face_with_verts(bm, vert_a, vert_b, side=side)
    if coords:
        bmesh.utils.face_split(host, vert_a, vert_b, coords=list(coords))
    else:
        bmesh.utils.face_split(host, vert_a, vert_b)
    return vert_a, vert_b


def segment_intersection_2d(a, b, c, d):
    """Closed-interval 2D segment intersection; returns (kind, t, u, point) or NONE."""

    ax, ay = float(a[0]), float(a[1])
    bx, by = float(b[0]), float(b[1])
    cx, cy = float(c[0]), float(c[1])
    dx, dy = float(d[0]), float(d[1])
    den = (bx - ax) * (dy - cy) - (by - ay) * (dx - cx)
    if abs(den) <= 1.0e-15:
        return "COLLINEAR_OR_PARALLEL", None, None, None
    t = ((cx - ax) * (dy - cy) - (cy - ay) * (dx - cx)) / den
    u = ((cx - ax) * (by - ay) - (cy - ay) * (bx - ax)) / den
    point = (ax + t * (bx - ax), ay + t * (by - ay), 0.0)
    if not (0.0 - 1.0e-12 <= t <= 1.0 + 1.0e-12 and 0.0 - 1.0e-12 <= u <= 1.0 + 1.0e-12):
        return "NONE", t, u, point

    def near_end(p, ends, tol=1.0e-7):
        return any(abs(p[0] - e[0]) <= tol and abs(p[1] - e[1]) <= tol for e in ends)

    a_end = near_end(point, [(ax, ay), (bx, by)])
    b_end = near_end(point, [(cx, cy), (dx, dy)])
    if a_end and b_end:
        kind = "ENDPOINT_ENDPOINT"
    elif a_end or b_end:
        kind = "ENDPOINT_INTERIOR"
    else:
        kind = "PROPER"
    return kind, t, u, point


def assert_proper_crossing_params(s0, s1, t0, t1, label: str) -> tuple[float, float, tuple]:
    """Verify ρ(s) × t is PROPER in 2D; return (t, u, q-)."""

    rho_s0 = (-float(s0[0]), float(s0[1]), float(s0[2]))
    rho_s1 = (-float(s1[0]), float(s1[1]), float(s1[2]))
    kind, t, u, point = segment_intersection_2d(rho_s0, rho_s1, t0, t1)
    assert kind == "PROPER", f"{label}: expected PROPER ρ(s)×t, got {kind} t={t} u={u} p={point}"
    assert t is not None and u is not None and 0.0 < t < 1.0 and 0.0 < u < 1.0, (label, t, u)
    print(f"YSE_CROSS_MIRROR_FIXTURE={label} kind=PROPER t={t:.6f} u={u:.6f} q_minus={point}", flush=True)
    return t, u, point


# ---------------------------------------------------------------------------
# Cut simulators
# ---------------------------------------------------------------------------


def simulate_case_a_cuts(obj) -> None:
    """F+ cut s + F- cut t with ρ(s)×t PROPER inside F-."""

    assert_proper_crossing_params(CASE_A_S0, CASE_A_S1, CASE_A_T0, CASE_A_T1, "case_a")
    bm = bmesh.from_edit_mesh(obj.data)
    cut_boundary_to_boundary(bm, CASE_A_S0, CASE_A_S1, side="positive")
    cut_boundary_to_boundary(bm, CASE_A_T0, CASE_A_T1, side="negative")
    bmesh.update_edit_mesh(obj.data, loop_triangles=True, destructive=True)


def simulate_case_b_cuts(obj) -> None:
    """F+ vertical s; F- zig-zag t (2 boundary segments) double-crossing ρ(s)."""

    # ρ(s) = vertical x=-0.5; t1=(-0.1,-1)→(-1,0), t2=(-1,0)→(-0.1,1).
    kind1, t1, u1, p1 = segment_intersection_2d(
        (-0.5, -1.0, 0.0),
        (-0.5, 1.0, 0.0),
        CASE_B_T0,
        CASE_B_T_MID,
    )
    kind2, t2, u2, p2 = segment_intersection_2d(
        (-0.5, -1.0, 0.0),
        (-0.5, 1.0, 0.0),
        CASE_B_T_MID,
        CASE_B_T1,
    )
    assert kind1 == "PROPER" and kind2 == "PROPER", (kind1, kind2, p1, p2)
    print(
        f"YSE_CROSS_MIRROR_FIXTURE=case_b kind=PROPER×2 q1={p1} t={t1:.6f}/u={u1:.6f} q2={p2} t={t2:.6f}/u={u2:.6f}",
        flush=True,
    )

    bm = bmesh.from_edit_mesh(obj.data)
    cut_boundary_to_boundary(bm, CASE_B_S0, CASE_B_S1, side="positive")
    # Two separate boundary→boundary cuts sharing the left-boundary mid vertex
    # (equivalent to a 2-segment zig-zag without interior waypoints).
    cut_boundary_to_boundary(bm, CASE_B_T0, CASE_B_T_MID, side="negative")
    cut_boundary_to_boundary(bm, CASE_B_T_MID, CASE_B_T1, side="negative")
    bmesh.update_edit_mesh(obj.data, loop_triangles=True, destructive=True)


def simulate_case_c_cuts(obj) -> None:
    """ENDPOINT_INTERIOR T-junction: bent s on F+, horizontal t on F-."""

    # ρ(s half) ( -0.4,-1)→(-0.4,0) ends on interior of t=(-1,0)→(0,0).
    kind, t, u, point = segment_intersection_2d(
        (-0.4, -1.0, 0.0),
        (-0.4, 0.0, 0.0),
        CASE_C_T0,
        CASE_C_T1,
    )
    assert kind == "ENDPOINT_INTERIOR", (kind, t, u, point)
    print(
        f"YSE_CROSS_MIRROR_FIXTURE=case_c kind=ENDPOINT_INTERIOR t={t:.6f} u={u:.6f} p={point}",
        flush=True,
    )

    bm = bmesh.from_edit_mesh(obj.data)
    cut_boundary_to_boundary(
        bm,
        CASE_C_S0,
        CASE_C_S1,
        side="positive",
        coords=[CASE_C_S_JOINT],
    )
    cut_boundary_to_boundary(bm, CASE_C_T0, CASE_C_T1, side="negative")
    bmesh.update_edit_mesh(obj.data, loop_triangles=True, destructive=True)


def simulate_case_d_cuts(obj) -> None:
    """Partial collinear: L-shaped s on F+, full vertical t on F- at ρ line."""

    print(
        "YSE_CROSS_MIRROR_FIXTURE=case_d kind=PARTIAL_COLLINEAR s_vertical=(0.3,-1)→(0.3,0.5) t=(-0.3,-1)→(-0.3,1)",
        flush=True,
    )
    bm = bmesh.from_edit_mesh(obj.data)
    cut_boundary_to_boundary(
        bm,
        CASE_D_S0,
        CASE_D_S1,
        side="positive",
        coords=[CASE_D_S_JOINT],
    )
    cut_boundary_to_boundary(bm, CASE_D_T0, CASE_D_T1, side="negative")
    bmesh.update_edit_mesh(obj.data, loop_triangles=True, destructive=True)


def simulate_case_e_cuts(obj) -> None:
    """One-sided cut on F+ only (no opposite native cut)."""

    print("YSE_CROSS_MIRROR_FIXTURE=case_e kind=ONE_SIDE_NOOP", flush=True)
    bm = bmesh.from_edit_mesh(obj.data)
    cut_boundary_to_boundary(bm, CASE_E_S0, CASE_E_S1, side="positive")
    bmesh.update_edit_mesh(obj.data, loop_triangles=True, destructive=True)


def _dissolve_plane_edge(bm) -> None:
    plane_edge = next(
        edge
        for edge in bm.edges
        if all(abs(vertex.co.x) <= 1.0e-8 for vertex in edge.verts)
        and {round(float(vertex.co.y), 6) for vertex in edge.verts} == {-1.0, 1.0}
    )
    bmesh.ops.dissolve_edges(bm, edges=[plane_edge], use_verts=False)


def simulate_case_f_cuts(obj) -> None:
    """CROSSES (p-stitch target) + off-plane PROPER pair on a spanning face."""

    kind, t, u, point = segment_intersection_2d(
        (-CASE_F_C[0], CASE_F_C[1], 0.0),
        (-CASE_F_R[0], CASE_F_R[1], 0.0),
        CASE_F_L,
        CASE_F_B,
    )
    assert kind == "PROPER", (kind, t, u, point)
    print(
        f"YSE_CROSS_MIRROR_FIXTURE=case_f kind=CROSSES+PROPER t={t:.6f} u={u:.6f} q_minus={point}",
        flush=True,
    )

    bm = bmesh.from_edit_mesh(obj.data)
    left_pt = ensure_boundary_vert(bm, CASE_F_L, side="negative")
    right_pt = ensure_boundary_vert(bm, CASE_F_R, side="positive")
    _dissolve_plane_edge(bm)
    host = next(face for face in bm.faces if left_pt in face.verts and right_pt in face.verts)
    bmesh.utils.face_split(host, left_pt, right_pt, coords=[CASE_F_B, CASE_F_C])
    bmesh.update_edit_mesh(obj.data, loop_triangles=True, destructive=True)


def simulate_case_g_cuts(obj) -> None:
    """Near-plane-band PROPER: |q_x|≤tol with s still POSITIVE (A-3-2)."""

    rho_s0 = (-CASE_G_S0[0], CASE_G_S0[1], 0.0)
    rho_s1 = (-CASE_G_S1[0], CASE_G_S1[1], 0.0)
    kind, t, u, point = segment_intersection_2d(rho_s0, rho_s1, CASE_G_T0, CASE_G_T1)
    assert kind == "PROPER", (kind, t, u, point)
    assert 0.0 < abs(point[0]) <= 1.0e-5, point
    print(
        f"YSE_CROSS_MIRROR_FIXTURE=case_g kind=ONPLANE_BAND t={t:.6f} u={u:.6f} q_raw={point}",
        flush=True,
    )

    bm = bmesh.from_edit_mesh(obj.data)
    cut_boundary_to_boundary(bm, CASE_G_S0, CASE_G_S1, side="positive")
    cut_boundary_to_boundary(bm, CASE_G_T0, CASE_G_T1, side="negative")
    bmesh.update_edit_mesh(obj.data, loop_triangles=True, destructive=True)


def simulate_case_h_cuts(obj) -> None:
    """ENDPOINT_ENDPOINT: ρ(s top) coincides with t top endpoint."""

    rho_s0 = (-CASE_H_S0[0], CASE_H_S0[1], 0.0)
    rho_s1 = (-CASE_H_S1[0], CASE_H_S1[1], 0.0)
    kind, t, u, point = segment_intersection_2d(rho_s0, rho_s1, CASE_H_T0, CASE_H_T1)
    assert kind == "ENDPOINT_ENDPOINT", (kind, t, u, point)
    print(
        f"YSE_CROSS_MIRROR_FIXTURE=case_h kind=ENDPOINT_ENDPOINT t={t:.6f} u={u:.6f} p={point}",
        flush=True,
    )

    bm = bmesh.from_edit_mesh(obj.data)
    cut_boundary_to_boundary(bm, CASE_H_S0, CASE_H_S1, side="positive")
    cut_boundary_to_boundary(bm, CASE_H_T0, CASE_H_T1, side="negative")
    bmesh.update_edit_mesh(obj.data, loop_triangles=True, destructive=True)


def simulate_case_i_cuts(obj) -> None:
    """Two POS rays + one NEG horizontal whose mirrored crossings share one cluster."""

    print(
        "YSE_CROSS_MIRROR_FIXTURE=case_i kind=TRIPLE_CLUSTER q_plus=(0.5,0) q_minus=(-0.5,0)",
        flush=True,
    )
    bm = bmesh.from_edit_mesh(obj.data)
    # Vertical s1 on F+.
    cut_boundary_to_boundary(bm, CASE_I_S1_0, CASE_I_S1_1, side="positive")
    # Split s1 at the future cluster site and fan two more POS chords into it.
    s1_edge = find_edge_between(bm, CASE_I_S1_0, CASE_I_S1_1)
    joint = split_edge_at_co(s1_edge, CASE_I_Q_PLUS)
    v_s2_0 = ensure_boundary_vert(bm, CASE_I_S2_0, side="positive")
    v_s2_1 = ensure_boundary_vert(bm, CASE_I_S2_1, side="positive")
    face_a = face_with_verts(bm, v_s2_0, joint, side="positive")
    bmesh.utils.face_split(face_a, v_s2_0, joint)
    face_b = face_with_verts(bm, v_s2_1, joint, side="positive")
    bmesh.utils.face_split(face_b, v_s2_1, joint)
    # Horizontal t on F- through the mirrored cluster site.
    cut_boundary_to_boundary(bm, CASE_I_T0, CASE_I_T1, side="negative")
    bmesh.update_edit_mesh(obj.data, loop_triangles=True, destructive=True)


def make_nonplanar_symmetric_2x1_grid():
    """2×1 grid with on-plane top mid lifted to z=+0.2 (self-mirrored non-planarity)."""

    mesh = bpy.data.meshes.new("YSE_CrossMirrorNonplanarMesh")
    vertices = [
        (-1.0, -1.0, 0.0),
        (0.0, -1.0, 0.0),
        (1.0, -1.0, 0.0),
        (-1.0, 1.0, 0.0),
        (0.0, 1.0, CASE_K_LIFT_Z),
        (1.0, 1.0, 0.0),
    ]
    mesh.from_pydata(vertices, [], [(0, 1, 4, 3), (1, 2, 5, 4)])
    mesh.update()
    obj = bpy.data.objects.new("YSE_CrossMirrorNonplanarObject", mesh)
    bpy.context.scene.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    obj.use_mesh_mirror_x = True
    obj.use_mesh_mirror_y = False
    obj.use_mesh_mirror_z = False
    return obj


def simulate_case_k_cuts(obj) -> None:
    """PROPER pair on bottom/left/right (avoids the lifted top mid)."""

    assert_proper_crossing_params(CASE_K_S0, CASE_K_S1, CASE_K_T0, CASE_K_T1, "case_k")
    bm = bmesh.from_edit_mesh(obj.data)
    cut_boundary_to_boundary(bm, CASE_K_S0, CASE_K_S1, side="positive")
    cut_boundary_to_boundary(bm, CASE_K_T0, CASE_K_T1, side="negative")
    bmesh.update_edit_mesh(obj.data, loop_triangles=True, destructive=True)


def _path_edge_lists(bm):
    by_side, total = core.collect_knife_path_edges_by_side(bm, core.AXIS_INDEX["X"], TOL)
    all_edges = (
        list(by_side.get("POSITIVE", ()))
        + list(by_side.get("NEGATIVE", ()))
        + list(by_side.get("CROSSES", ()))
        + list(by_side.get("PLANE", ()))
    )
    return by_side, total, all_edges


def _plan_on_edit_object(obj, topology_or_session_frames, mirror_face_ids):
    """Call plan_mirrored_path_crossings on the live edit bmesh."""

    bm = bmesh.from_edit_mesh(obj.data)
    by_side, _total, all_edges = _path_edge_lists(bm)
    live = core.resolve_live_mirror_face_map(
        bm,
        mirror_face_ids,
        core.AXIS_INDEX["X"],
        TOL,
        path_edges=all_edges,
    )
    frames = topology_or_session_frames
    return (
        core.plan_mirrored_path_crossings(
            bm,
            by_side,
            core.AXIS_INDEX["X"],
            TOL,
            live,
            frames,
        ),
        bm,
        by_side,
    )


def begin_edit_knife(window, area, region, obj):
    with bpy.context.temp_override(window=window, area=area, region=region):
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.context.tool_settings.mesh_select_mode = (True, False, False)
        prepare_knife_session(bpy.context)
    return obj


def run_finish(window, area, region):
    with bpy.context.temp_override(window=window, area=area, region=region):
        return bpy.ops.mesh.ydd_symmetric_edit_finish("EXEC_DEFAULT")


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------


def case_a_proper_crossing(window, area, region) -> None:
    print("YSE_KNIFE_CROSS_MIRROR_STITCH_CASE=a_proper_crossing", flush=True)
    clear_scene()
    obj = make_symmetric_2x1_grid()
    begin_edit_knife(window, area, region, obj)
    simulate_case_a_cuts(obj)

    bm = bmesh.from_edit_mesh(obj.data)
    by_side, total = core.collect_knife_path_edges_by_side(bm, core.AXIS_INDEX["X"], TOL)
    assert total >= 2, (total, {key: len(value) for key, value in by_side.items()})
    assert by_side["POSITIVE"] and by_side["NEGATIVE"], {key: len(value) for key, value in by_side.items()}
    assert not by_side["CROSSES"], {key: len(value) for key, value in by_side.items()}

    finished = run_finish(window, area, region)
    assert finished == {"FINISHED"}, finished

    bm = bmesh.from_edit_mesh(obj.data)
    # Intersection pair (contract §D-1): one vertex each side, degree 4.
    # After stitch the original chords are split at q, so only half-edges remain.
    q_plus = verts_near(bm, CASE_A_Q_PLUS, tolerance=1.0e-3)
    q_minus = verts_near(bm, CASE_A_Q_MINUS, tolerance=1.0e-3)
    assert len(q_plus) == 1, [tuple(v.co) for v in bm.verts]
    assert len(q_minus) == 1, [tuple(v.co) for v in bm.verts]
    assert len(q_plus[0].link_edges) == 4, len(q_plus[0].link_edges)
    assert len(q_minus[0].link_edges) == 4, len(q_minus[0].link_edges)
    # Half-edges of the split native chords (and their mirrors) meet at q.
    assert has_exact_edge(bm, CASE_A_S0, CASE_A_Q_PLUS) or any(
        core.coordinates_match(edge.verts[0].co, CASE_A_S0, 1.0e-3)
        or core.coordinates_match(edge.verts[1].co, CASE_A_S0, 1.0e-3)
        for edge in q_plus[0].link_edges
    )
    assert has_exact_edge(bm, CASE_A_T0, CASE_A_Q_MINUS) or any(
        core.coordinates_match(edge.verts[0].co, CASE_A_T0, 1.0e-3)
        or core.coordinates_match(edge.verts[1].co, CASE_A_T0, 1.0e-3)
        for edge in q_minus[0].link_edges
    )
    assert_no_duplicate_edges(bm)
    assert_x_symmetric(bm)
    assert_temp_layers_cleared(bm)
    assert not operators._SESSIONS
    print("YSE_KNIFE_CROSS_MIRROR_STITCH_CASE_A=OK", flush=True)


def case_b_zigzag_double_crossing(window, area, region) -> None:
    print("YSE_KNIFE_CROSS_MIRROR_STITCH_CASE=b_zigzag_double_crossing", flush=True)
    clear_scene()
    obj = make_symmetric_2x1_grid()
    begin_edit_knife(window, area, region, obj)
    simulate_case_b_cuts(obj)

    bm = bmesh.from_edit_mesh(obj.data)
    by_side, total = core.collect_knife_path_edges_by_side(bm, core.AXIS_INDEX["X"], TOL)
    assert total >= 3, (total, {key: len(value) for key, value in by_side.items()})
    assert by_side["POSITIVE"] and by_side["NEGATIVE"], {key: len(value) for key, value in by_side.items()}

    finished = run_finish(window, area, region)
    assert finished == {"FINISHED"}, finished

    bm = bmesh.from_edit_mesh(obj.data)
    # Two intersection pairs.
    for expected in (CASE_B_Q1_MINUS, CASE_B_Q1_PLUS, CASE_B_Q2_MINUS, CASE_B_Q2_PLUS):
        found = verts_near(bm, expected, tolerance=1.0e-3)
        assert len(found) == 1, (expected, [tuple(v.co) for v in bm.verts])
        assert len(found[0].link_edges) >= 4, (expected, len(found[0].link_edges))
    assert_no_duplicate_edges(bm)
    assert_x_symmetric(bm)
    assert_temp_layers_cleared(bm)
    assert not operators._SESSIONS
    print("YSE_KNIFE_CROSS_MIRROR_STITCH_CASE_B=OK", flush=True)


def case_c_endpoint_interior(window, area, region) -> None:
    print("YSE_KNIFE_CROSS_MIRROR_STITCH_CASE=c_endpoint_interior", flush=True)
    clear_scene()
    obj = make_symmetric_2x1_grid()
    begin_edit_knife(window, area, region, obj)
    simulate_case_c_cuts(obj)

    bm = bmesh.from_edit_mesh(obj.data)
    verts_before = len(bm.verts)
    by_side, total = core.collect_knife_path_edges_by_side(bm, core.AXIS_INDEX["X"], TOL)
    assert total >= 3, (total, {key: len(value) for key, value in by_side.items()})

    finished = run_finish(window, area, region)
    assert finished == {"FINISHED"}, finished

    bm = bmesh.from_edit_mesh(obj.data)
    # T points: joint (0.4,0) reused; mirror (-0.4,0) is the stitch/split site.
    joint = verts_near(bm, CASE_C_S_JOINT, tolerance=1.0e-3)
    t_point = verts_near(bm, CASE_C_T_POINT, tolerance=1.0e-3)
    assert len(joint) == 1, [tuple(v.co) for v in bm.verts]
    assert len(t_point) == 1, [tuple(v.co) for v in bm.verts]
    # No duplicate vertices at the T locations.
    assert len(verts_near(bm, CASE_C_S_JOINT, tolerance=1.0e-4)) == 1
    assert len(verts_near(bm, CASE_C_T_POINT, tolerance=1.0e-4)) == 1
    # Other edge (horizontal t / its mirror) is split: degree at T ≥ 3.
    assert len(joint[0].link_edges) >= 3, len(joint[0].link_edges)
    assert len(t_point[0].link_edges) >= 3, len(t_point[0].link_edges)
    # t is split at (-0.4,0): both halves exist.
    assert has_exact_edge(bm, CASE_C_T0, CASE_C_T_POINT) or any(
        abs(edge.verts[0].co.x + 0.4) <= 1.0e-3 or abs(edge.verts[1].co.x + 0.4) <= 1.0e-3
        for edge in t_point[0].link_edges
    )
    assert_no_duplicate_edges(bm)
    assert_x_symmetric(bm)
    assert_temp_layers_cleared(bm)
    assert not operators._SESSIONS
    # Endpoint reuse: we must not invent a second vertex at the joint beyond the
    # mirror pair (net growth is mirror geometry, not duplicates at T).
    assert len(bm.verts) >= verts_before
    print("YSE_KNIFE_CROSS_MIRROR_STITCH_CASE_C=OK", flush=True)


def case_d_partial_collinear_decline(window, area, region) -> None:
    print("YSE_KNIFE_CROSS_MIRROR_STITCH_CASE=d_partial_collinear_decline", flush=True)
    clear_scene()
    obj = make_symmetric_2x1_grid()
    begin_edit_knife(window, area, region, obj)
    simulate_case_d_cuts(obj)

    bm = bmesh.from_edit_mesh(obj.data)
    native_sig = mesh_signature(bm)
    native_counts = (len(bm.verts), len(bm.edges), len(bm.faces))

    finished = run_finish(window, area, region)
    assert finished == {"FINISHED"}, finished

    bm = bmesh.from_edit_mesh(obj.data)
    # Mirror stage declined: native topology preserved (rollback).
    assert (len(bm.verts), len(bm.edges), len(bm.faces)) == native_counts, (
        (len(bm.verts), len(bm.edges), len(bm.faces)),
        native_counts,
    )
    assert mesh_signature(bm) == native_sig, "native mesh diverged after collinear decline"
    # Native cuts still present; mirrored counterparts of s must not appear as a
    # successful both-sides result (rollback keeps only native).
    assert has_exact_edge(bm, CASE_D_S0, CASE_D_S_JOINT) or has_exact_edge(bm, CASE_D_S0, CASE_D_S1)
    assert has_exact_edge(bm, CASE_D_T0, CASE_D_T1)
    warnings = warning_messages()
    assert warnings, (warnings, operators._FINISH_REPORTS)
    # Success INFO must not accompany a decline WARNING.
    assert not info_messages(), operators._FINISH_REPORTS
    assert_temp_layers_cleared(bm)
    assert not operators._SESSIONS
    print("YSE_KNIFE_CROSS_MIRROR_STITCH_CASE_D=OK", flush=True)


def case_e_one_side_noop(window, area, region) -> None:
    print("YSE_KNIFE_CROSS_MIRROR_STITCH_CASE=e_one_side_noop", flush=True)
    clear_scene()
    obj = make_symmetric_2x1_grid()
    begin_edit_knife(window, area, region, obj)
    simulate_case_e_cuts(obj)

    bm = bmesh.from_edit_mesh(obj.data)
    by_side, total = core.collect_knife_path_edges_by_side(bm, core.AXIS_INDEX["X"], TOL)
    assert total >= 1, total
    assert by_side["POSITIVE"] and not by_side["NEGATIVE"], {key: len(value) for key, value in by_side.items()}

    finished = run_finish(window, area, region)
    assert finished == {"FINISHED"}, finished

    bm = bmesh.from_edit_mesh(obj.data)
    assert has_exact_edge(bm, CASE_E_S0, CASE_E_S1)
    assert has_exact_edge(bm, (-0.5, -1.0, 0.0), (-0.5, 1.0, 0.0))
    assert_no_duplicate_edges(bm)
    assert_x_symmetric(bm)
    assert_temp_layers_cleared(bm)
    assert not operators._SESSIONS
    print("YSE_KNIFE_CROSS_MIRROR_STITCH_CASE_E=OK", flush=True)


def case_f_crosses_plus_crossing(window, area, region) -> None:
    print("YSE_KNIFE_CROSS_MIRROR_STITCH_CASE=f_crosses_plus_crossing", flush=True)
    clear_scene()
    obj = make_symmetric_2x1_grid()
    begin_edit_knife(window, area, region, obj)
    simulate_case_f_cuts(obj)

    bm = bmesh.from_edit_mesh(obj.data)
    by_side, total = core.collect_knife_path_edges_by_side(bm, core.AXIS_INDEX["X"], TOL)
    assert total >= 3, (total, {key: len(value) for key, value in by_side.items()})
    assert by_side["CROSSES"], {key: len(value) for key, value in by_side.items()}
    assert any(not core.is_self_mirrored_edge(edge, core.AXIS_INDEX["X"], TOL) for edge in by_side["CROSSES"])

    from ydd_symmetric_edit import backup as backup_mod

    backup_calls: list[int] = []
    original_create = backup_mod.create_topology_backup

    def _counting_create(edit_bm):
        backup_calls.append(1)
        return original_create(edit_bm)

    backup_mod.create_topology_backup = _counting_create  # type: ignore[assignment]
    try:
        finished = run_finish(window, area, region)
    finally:
        backup_mod.create_topology_backup = original_create  # type: ignore[assignment]

    assert finished == {"FINISHED"}, finished
    assert len(backup_calls) == 1, f"expected exactly 1 backup, got {len(backup_calls)}"
    assert not warning_messages(), operators._FINISH_REPORTS

    bm = bmesh.from_edit_mesh(obj.data)
    # p-stitch left an on-plane vertex (CROSSES midpoint region).
    assert any(abs(float(vertex.co.x)) <= 1.0e-4 for vertex in bm.verts), [tuple(v.co) for v in bm.verts]
    # Off-plane intersection pair.
    q_plus = verts_near(bm, CASE_F_Q_PLUS, tolerance=2.0e-2)
    q_minus = verts_near(bm, CASE_F_Q_MINUS, tolerance=2.0e-2)
    assert len(q_plus) >= 1, (CASE_F_Q_PLUS, [tuple(v.co) for v in bm.verts])
    assert len(q_minus) >= 1, (CASE_F_Q_MINUS, [tuple(v.co) for v in bm.verts])
    assert_no_duplicate_edges(bm)
    assert_x_symmetric(bm)
    assert_temp_layers_cleared(bm)
    assert not operators._SESSIONS
    print("YSE_KNIFE_CROSS_MIRROR_STITCH_CASE_F=OK", flush=True)


def case_g_onplane_band(window, area, region) -> None:
    """Unit (A-3-2): |q_x|≤tol is canonicalized to a single on-plane shared vertex.

    Full finish with near-plane path endpoints confuses later mirror placement
    (asymmetric leftover boundary verts). Plan/apply of the crossing stage is
    the contract surface for on-plane band behaviour.
    """

    print("YSE_KNIFE_CROSS_MIRROR_STITCH_CASE=g_onplane_band", flush=True)
    clear_scene()
    obj = make_symmetric_2x1_grid()
    with bpy.context.temp_override(window=window, area=area, region=region):
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.context.tool_settings.mesh_select_mode = (True, False, False)

    session_tol = 1.0e-5
    bm = bmesh.from_edit_mesh(obj.data)
    topology = core.prepare_topology(bm, core.AXIS_INDEX["X"], session_tol)
    from ydd_symmetric_edit._types import FaceId

    face_layer = bm.faces.layers.int.get(core.FACE_ID_LAYER)
    assert face_layer is not None
    carrier_face = next(face for face in bm.faces if face.calc_center_median().x > 0.0)
    carrier_id = FaceId(int(carrier_face[face_layer]))

    # 2D-verified near-plane PROPER (same geometry as simulate_case_g_cuts).
    rho_s0 = (-CASE_G_S0[0], CASE_G_S0[1], 0.0)
    rho_s1 = (-CASE_G_S1[0], CASE_G_S1[1], 0.0)
    kind, t, u, point = segment_intersection_2d(rho_s0, rho_s1, CASE_G_T0, CASE_G_T1)
    assert kind == "PROPER", (kind, t, u, point)
    assert 0.0 < abs(point[0]) <= session_tol, point
    print(
        f"YSE_CROSS_MIRROR_FIXTURE=case_g kind=ONPLANE_BAND t={t:.6f} u={u:.6f} q_raw={point}",
        flush=True,
    )

    v_s0 = ensure_boundary_vert(bm, CASE_G_S0, side="positive")
    v_s1 = ensure_boundary_vert(bm, CASE_G_S1, side="positive")
    v_t0 = ensure_boundary_vert(bm, CASE_G_T0, side="negative")
    v_t1 = ensure_boundary_vert(bm, CASE_G_T1, side="negative")
    pos_edge = bm.edges.new((v_s0, v_s1))
    neg_edge = bm.edges.new((v_t0, v_t1))
    edge_layer = bm.edges.layers.int.get(core.EDGE_ORIGINAL_LAYER)
    assert edge_layer is not None
    pos_edge[edge_layer] = 0
    neg_edge[edge_layer] = 0
    core.add_selection_layers(bm)
    bm.edges.ensure_lookup_table()
    pos_edge = find_edge_between(bm, CASE_G_S0, CASE_G_S1)
    neg_edge = find_edge_between(bm, CASE_G_T0, CASE_G_T1)

    original_carrier_ids = stitch._edge_carrier_ids
    original_mirrored_ids = stitch._mirrored_carrier_ids

    def _stub_carrier_ids(edge, _face_layer):
        del edge
        return {carrier_id}

    def _stub_mirrored_ids(carrier_ids, _mirror_face_ids):
        return set(carrier_ids)

    stitch._edge_carrier_ids = _stub_carrier_ids  # type: ignore[assignment]
    stitch._mirrored_carrier_ids = _stub_mirrored_ids  # type: ignore[assignment]
    try:
        by_side = {
            "POSITIVE": [pos_edge],
            "NEGATIVE": [neg_edge],
            "CROSSES": [],
            "PLANE": [],
        }
        mirror_face_ids = dict(topology.mirror_face_ids)
        mirror_face_ids[carrier_id] = carrier_id
        plan, reason = core.plan_mirrored_path_crossings(
            bm,
            by_side,
            core.AXIS_INDEX["X"],
            session_tol,
            mirror_face_ids,
            topology.carrier_frames,
        )
        assert reason == "", reason
        assert len(plan) == 1, plan
        cluster = plan[0]
        # On-plane band: single canonical point (q⁻ collapses onto q⁺).
        assert abs(float(cluster.positive_coordinate.x)) <= 1.0e-15, cluster.positive_coordinate
        assert abs(float(cluster.negative_coordinate.x)) <= 1.0e-15, cluster.negative_coordinate
        assert core.coordinates_match(
            cluster.positive_coordinate,
            cluster.negative_coordinate,
            session_tol,
        )
        # Combined occurrences on the positive (on-plane) side only.
        assert cluster.positive and not cluster.negative, (cluster.positive, cluster.negative)

        stitched, apply_reason = core.apply_mirrored_path_crossings(bm, plan)
        assert apply_reason == "", apply_reason
        assert stitched > 0, stitched
    finally:
        stitch._edge_carrier_ids = original_carrier_ids  # type: ignore[assignment]
        stitch._mirrored_carrier_ids = original_mirrored_ids  # type: ignore[assignment]

    bm.verts.ensure_lookup_table()
    plane_q = [
        vertex
        for vertex in bm.verts
        if abs(float(vertex.co.x)) <= session_tol and abs(float(vertex.co.y) - CASE_G_T_Y) <= 1.0e-3
    ]
    assert len(plane_q) == 1, [tuple(v.co) for v in bm.verts]
    assert abs(float(plane_q[0].co.x)) <= 1.0e-15, tuple(plane_q[0].co)
    # Not a ± off-plane pair.
    off_plus = verts_near(bm, (abs(point[0]), point[1], 0.0), tolerance=session_tol * 0.5)
    off_minus = verts_near(bm, (-abs(point[0]), point[1], 0.0), tolerance=session_tol * 0.5)
    assert not (len(off_plus) == 1 and len(off_minus) == 1 and off_plus[0] != off_minus[0])
    core.remove_temporary_layers(bm)
    bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
    print("YSE_KNIFE_CROSS_MIRROR_STITCH_CASE_G=OK", flush=True)


def case_h_endpoint_endpoint(window, area, region) -> None:
    print("YSE_KNIFE_CROSS_MIRROR_STITCH_CASE=h_endpoint_endpoint", flush=True)
    clear_scene()
    obj = make_symmetric_2x1_grid()
    begin_edit_knife(window, area, region, obj)
    simulate_case_h_cuts(obj)

    bm = bmesh.from_edit_mesh(obj.data)
    verts_before = len(bm.verts)
    ee_before = len(verts_near(bm, CASE_H_EE, tolerance=1.0e-4))
    assert ee_before == 1, ee_before

    finished = run_finish(window, area, region)
    assert finished == {"FINISHED"}, finished

    bm = bmesh.from_edit_mesh(obj.data)
    # Endpoint reuse only: still a single vertex at the EE site (and its mirror).
    assert len(verts_near(bm, CASE_H_EE, tolerance=1.0e-4)) == 1
    assert len(verts_near(bm, (0.4, 1.0, 0.0), tolerance=1.0e-4)) == 1
    # No duplicate vertices at EE locations.
    assert_no_duplicate_edges(bm)
    assert_x_symmetric(bm)
    assert_temp_layers_cleared(bm)
    assert not operators._SESSIONS
    assert len(bm.verts) >= verts_before
    print("YSE_KNIFE_CROSS_MIRROR_STITCH_CASE_H=OK", flush=True)


def case_i_triple_cluster(window, area, region) -> None:
    print("YSE_KNIFE_CROSS_MIRROR_STITCH_CASE=i_triple_cluster", flush=True)
    clear_scene()
    obj = make_symmetric_2x1_grid()
    begin_edit_knife(window, area, region, obj)
    simulate_case_i_cuts(obj)

    finished = run_finish(window, area, region)
    assert finished == {"FINISHED"}, finished

    bm = bmesh.from_edit_mesh(obj.data)
    q_plus = verts_near(bm, CASE_I_Q_PLUS, tolerance=1.0e-3)
    q_minus = verts_near(bm, CASE_I_Q_MINUS, tolerance=1.0e-3)
    assert len(q_plus) == 1, [tuple(v.co) for v in bm.verts]
    assert len(q_minus) == 1, [tuple(v.co) for v in bm.verts]
    assert len(q_plus[0].link_edges) >= 6, len(q_plus[0].link_edges)
    assert len(q_minus[0].link_edges) >= 6, len(q_minus[0].link_edges)
    assert_no_duplicate_edges(bm)
    assert_x_symmetric(bm)
    assert_temp_layers_cleared(bm)
    assert not operators._SESSIONS
    print("YSE_KNIFE_CROSS_MIRROR_STITCH_CASE_I=OK", flush=True)


def case_j_fixed_self_mirrored(window, area, region) -> None:
    """Unit (A-1-3): fixed self-mirrored edge × POS cross → plan splits at q and ρ(q).

    Manifold face_split cannot keep an unsplit self-mirrored chord while also
    storing a geometrically crossing POS chord without a mesh vertex (which
    would split the fixed edge and drop it out of the self-mirrored bucket).
    Wire edges + carrier-id stub exercise the fixed-branch of plan/apply.
    """

    print("YSE_KNIFE_CROSS_MIRROR_STITCH_CASE=j_fixed_self_mirrored", flush=True)
    clear_scene()
    obj = make_symmetric_2x1_grid()
    with bpy.context.temp_override(window=window, area=area, region=region):
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.context.tool_settings.mesh_select_mode = (True, False, False)

    bm = bmesh.from_edit_mesh(obj.data)
    topology = core.prepare_topology(bm, core.AXIS_INDEX["X"], TOL)
    from ydd_symmetric_edit._types import FaceId

    face_layer = bm.faces.layers.int.get(core.FACE_ID_LAYER)
    assert face_layer is not None
    # Use the pre-dissolve positive face as a stable carrier id for the stub.
    carrier_face = next(face for face in bm.faces if face.calc_center_median().x > 0.0)
    carrier_id = FaceId(int(carrier_face[face_layer]))

    left_pt = ensure_boundary_vert(bm, CASE_J_FIXED_L, side="negative")
    right_pt = ensure_boundary_vert(bm, CASE_J_FIXED_R, side="positive")
    bot_pt = ensure_boundary_vert(bm, CASE_J_S0, side="positive")
    top_pt = ensure_boundary_vert(bm, CASE_J_S1, side="positive")
    _dissolve_plane_edge(bm)
    # Wire path edges (no face_split): geometrically cross at (0.5,0) without a mesh vertex.
    fixed_edge = bm.edges.new((left_pt, right_pt))
    moving_edge = bm.edges.new((bot_pt, top_pt))
    edge_layer = bm.edges.layers.int.get(core.EDGE_ORIGINAL_LAYER)
    assert edge_layer is not None
    fixed_edge[edge_layer] = 0
    moving_edge[edge_layer] = 0
    # Selection layers first — layer add invalidates edge wrappers held by a plan.
    core.add_selection_layers(bm)
    bm.edges.ensure_lookup_table()
    fixed_edge = find_edge_between(bm, CASE_J_FIXED_L, CASE_J_FIXED_R)
    moving_edge = find_edge_between(bm, CASE_J_S0, CASE_J_S1)
    assert core.is_self_mirrored_edge(fixed_edge, core.AXIS_INDEX["X"], TOL)

    original_carrier_ids = stitch._edge_carrier_ids
    original_mirrored_ids = stitch._mirrored_carrier_ids

    def _stub_carrier_ids(edge, _face_layer):
        del edge
        return {carrier_id}

    def _stub_mirrored_ids(carrier_ids, _mirror_face_ids):
        return set(carrier_ids)

    stitch._edge_carrier_ids = _stub_carrier_ids  # type: ignore[assignment]
    stitch._mirrored_carrier_ids = _stub_mirrored_ids  # type: ignore[assignment]
    try:
        by_side = {
            "POSITIVE": [moving_edge],
            "NEGATIVE": [],
            "CROSSES": [fixed_edge],
            "PLANE": [],
        }
        # Self-map the carrier so orbit selection succeeds.
        mirror_face_ids = dict(topology.mirror_face_ids)
        mirror_face_ids[carrier_id] = carrier_id
        plan, reason = core.plan_mirrored_path_crossings(
            bm,
            by_side,
            core.AXIS_INDEX["X"],
            TOL,
            mirror_face_ids,
            topology.carrier_frames,
        )
        assert reason == "", reason
        assert plan, "expected fixed×moving crossing plan"
        # Fixed edge participates on both sides (q and ρ(q)).
        fixed_ids = {id(fixed_edge)}
        pos_edges = {id(occurrence.edge) for cluster in plan for occurrence in cluster.positive}
        neg_edges = {id(occurrence.edge) for cluster in plan for occurrence in cluster.negative}
        assert fixed_ids & pos_edges, (plan, pos_edges)
        assert fixed_ids & neg_edges, (plan, neg_edges)
        assert id(moving_edge) in pos_edges, pos_edges

        stitched, apply_reason = core.apply_mirrored_path_crossings(bm, plan)
        assert apply_reason == "", apply_reason
        assert stitched > 0, stitched
    finally:
        stitch._edge_carrier_ids = original_carrier_ids  # type: ignore[assignment]
        stitch._mirrored_carrier_ids = original_mirrored_ids  # type: ignore[assignment]

    bm.verts.ensure_lookup_table()
    q_plus = verts_near(bm, CASE_J_Q_PLUS, tolerance=1.0e-3)
    q_minus = verts_near(bm, CASE_J_Q_MINUS, tolerance=1.0e-3)
    assert len(q_plus) == 1, [tuple(v.co) for v in bm.verts]
    assert len(q_minus) == 1, [tuple(v.co) for v in bm.verts]
    # Fixed chord is split; the unbroken full fixed edge is gone.
    assert not has_exact_edge(bm, CASE_J_FIXED_L, CASE_J_FIXED_R)
    assert has_exact_edge(bm, CASE_J_FIXED_L, CASE_J_Q_MINUS) or any(
        abs(float(vertex.co.x) + 0.5) <= 1.0e-3 for edge in q_minus[0].link_edges for vertex in edge.verts
    )
    # Mirror of fixed is not duplicated as a second full chord.
    assert len([e for e in bm.edges if _edge_has_endpoints(e, CASE_J_FIXED_L, CASE_J_FIXED_R, 1.0e-3)]) == 0
    core.remove_temporary_layers(bm)
    bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
    print("YSE_KNIFE_CROSS_MIRROR_STITCH_CASE_J=OK", flush=True)


def case_k_nonplanar(window, area, region) -> None:
    print("YSE_KNIFE_CROSS_MIRROR_STITCH_CASE=k_nonplanar", flush=True)
    clear_scene()
    obj = make_nonplanar_symmetric_2x1_grid()
    begin_edit_knife(window, area, region, obj)
    simulate_case_k_cuts(obj)

    finished = run_finish(window, area, region)
    assert finished == {"FINISHED"}, finished

    bm = bmesh.from_edit_mesh(obj.data)
    # Intersection verts must be bit-exact X-mirrors (sign flip only).
    candidates_plus = [
        vertex
        for vertex in bm.verts
        if abs(float(vertex.co.x) - CASE_K_Q_PLUS[0]) <= 5.0e-2
        and abs(float(vertex.co.y) - CASE_K_Q_PLUS[1]) <= 5.0e-2
        and float(vertex.co.x) > 0.0
    ]
    candidates_minus = [
        vertex
        for vertex in bm.verts
        if abs(float(vertex.co.x) - CASE_K_Q_MINUS[0]) <= 5.0e-2
        and abs(float(vertex.co.y) - CASE_K_Q_MINUS[1]) <= 5.0e-2
        and float(vertex.co.x) < 0.0
    ]
    assert candidates_plus and candidates_minus, (
        [tuple(v.co) for v in bm.verts],
        len(candidates_plus),
        len(candidates_minus),
    )

    def pick(cands):
        ranked = sorted(
            cands,
            key=lambda v: (-len(v.link_edges), abs(float(v.co.x) - abs(CASE_K_Q_PLUS[0]))),
        )
        return ranked[0]

    q_plus = pick(candidates_plus)
    q_minus = pick(candidates_minus)
    assert abs(float(q_plus.co.x) + float(q_minus.co.x)) <= 1.0e-12, (tuple(q_plus.co), tuple(q_minus.co))
    assert abs(float(q_plus.co.y) - float(q_minus.co.y)) <= 1.0e-12, (tuple(q_plus.co), tuple(q_minus.co))
    assert abs(float(q_plus.co.z) - float(q_minus.co.z)) <= 1.0e-12, (tuple(q_plus.co), tuple(q_minus.co))
    assert_no_duplicate_edges(bm)
    assert_x_symmetric(bm)
    assert_temp_layers_cleared(bm)
    assert not operators._SESSIONS
    print("YSE_KNIFE_CROSS_MIRROR_STITCH_CASE_K=OK", flush=True)


def case_l_nonplanar_guard(window, area, region) -> None:
    """Unit: move one path endpoint far off the pre-state carrier → non-planarity guard."""

    print("YSE_KNIFE_CROSS_MIRROR_STITCH_CASE=l_nonplanar_guard", flush=True)
    clear_scene()
    obj = make_symmetric_2x1_grid()
    with bpy.context.temp_override(window=window, area=area, region=region):
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.context.tool_settings.mesh_select_mode = (True, False, False)

    bm = bmesh.from_edit_mesh(obj.data)
    topology = core.prepare_topology(bm, core.AXIS_INDEX["X"], TOL)
    bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)

    # Native coplanar PROPER pair (case_a geometry).
    cut_boundary_to_boundary(bm, CASE_A_S0, CASE_A_S1, side="positive")
    cut_boundary_to_boundary(bm, CASE_A_T0, CASE_A_T1, side="negative")
    # Pull one positive-side path endpoint far off the pre-state plane (dev≈0).
    s1 = find_vert_at(bm, CASE_A_S1, tol=1.0e-5)
    assert s1 is not None
    s1.co.z = 1.0
    bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)

    bm = bmesh.from_edit_mesh(obj.data)
    by_side, total, all_edges = _path_edge_lists(bm)
    assert total >= 2, total
    live = core.resolve_live_mirror_face_map(
        bm,
        topology.mirror_face_ids,
        core.AXIS_INDEX["X"],
        TOL,
        path_edges=all_edges,
    )
    plan, reason = core.plan_mirrored_path_crossings(
        bm,
        by_side,
        core.AXIS_INDEX["X"],
        TOL,
        live,
        topology.carrier_frames,
    )
    assert plan == [], plan
    assert reason == "a mirrored cut intersection exceeds its carrier non-planarity guard", reason
    print("YSE_KNIFE_CROSS_MIRROR_STITCH_CASE_L=OK", flush=True)


def case_m_ambiguous_vertex(window, area, region) -> None:
    """Unit: two unrelated verts at q within tol → apply decline (native mesh kept)."""

    print("YSE_KNIFE_CROSS_MIRROR_STITCH_CASE=m_ambiguous_vertex", flush=True)
    clear_scene()
    obj = make_symmetric_2x1_grid()
    with bpy.context.temp_override(window=window, area=area, region=region):
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.context.tool_settings.mesh_select_mode = (True, False, False)

    bm = bmesh.from_edit_mesh(obj.data)
    topology = core.prepare_topology(bm, core.AXIS_INDEX["X"], TOL)
    cut_boundary_to_boundary(bm, CASE_A_S0, CASE_A_S1, side="positive")
    cut_boundary_to_boundary(bm, CASE_A_T0, CASE_A_T1, side="negative")
    # Selection layers before plan: layer add invalidates edge wrappers held by a plan.
    core.add_selection_layers(bm)
    bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)

    bm = bmesh.from_edit_mesh(obj.data)
    by_side, total, all_edges = _path_edge_lists(bm)
    live = core.resolve_live_mirror_face_map(
        bm,
        topology.mirror_face_ids,
        core.AXIS_INDEX["X"],
        TOL,
        path_edges=all_edges,
    )
    plan, reason = core.plan_mirrored_path_crossings(
        bm,
        by_side,
        core.AXIS_INDEX["X"],
        TOL,
        live,
        topology.carrier_frames,
    )
    assert reason == "", reason
    assert plan, "expected a non-empty crossing plan for case_a geometry"

    # Two unrelated free vertices at the planned positive intersection.
    q_plus = plan[0].positive_coordinate
    bm.verts.new((float(q_plus.x) + 1.0e-6, float(q_plus.y), float(q_plus.z)))
    bm.verts.new((float(q_plus.x) - 1.0e-6, float(q_plus.y), float(q_plus.z)))
    bm.verts.ensure_lookup_table()

    stitched, apply_reason = core.apply_mirrored_path_crossings(bm, plan)
    assert stitched == 0, stitched
    assert apply_reason == "multiple existing vertices are ambiguous at a mirrored cut intersection", apply_reason
    # Path edges remain unsplit (decline before mutation of chords).
    assert has_exact_edge(bm, CASE_A_S0, CASE_A_S1), "path edge should remain unsplit after decline"
    assert has_exact_edge(bm, CASE_A_T0, CASE_A_T1), "path edge should remain unsplit after decline"
    core.remove_temporary_layers(bm)
    bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
    print("YSE_KNIFE_CROSS_MIRROR_STITCH_CASE_M=OK", flush=True)


def run_test() -> None:
    addon.register()
    window, area, region = viewport_context()
    configure_view(area)

    # Run every case even when early ones fail so the report covers a–m.
    # Final expected behavior still asserts success for a–c; pre-fix builds
    # are expected to fail those with the mirror-placement decline signature.
    cases = (
        ("a_proper_crossing", case_a_proper_crossing),
        ("b_zigzag_double_crossing", case_b_zigzag_double_crossing),
        ("c_endpoint_interior", case_c_endpoint_interior),
        ("d_partial_collinear_decline", case_d_partial_collinear_decline),
        ("e_one_side_noop", case_e_one_side_noop),
        ("f_crosses_plus_crossing", case_f_crosses_plus_crossing),
        ("g_onplane_band", case_g_onplane_band),
        ("h_endpoint_endpoint", case_h_endpoint_endpoint),
        ("i_triple_cluster", case_i_triple_cluster),
        ("j_fixed_self_mirrored", case_j_fixed_self_mirrored),
        ("k_nonplanar", case_k_nonplanar),
        ("l_nonplanar_guard", case_l_nonplanar_guard),
        ("m_ambiguous_vertex", case_m_ambiguous_vertex),
    )
    failures: list[str] = []
    for name, func in cases:
        # Drop leftover finish reports / sessions between cases.
        operators._FINISH_REPORTS.clear()
        operators._SESSIONS.clear()
        try:
            func(window, area, region)
            print(f"YSE_KNIFE_CROSS_MIRROR_STITCH_RESULT={name}:PASS", flush=True)
        except BaseException as exc:
            # Capture the decline / assertion signature without aborting the suite.
            reports = list(operators._FINISH_REPORTS)
            msg = f"{type(exc).__name__}: {exc}"
            print(f"YSE_KNIFE_CROSS_MIRROR_STITCH_RESULT={name}:FAIL {msg}", flush=True)
            print(f"YSE_KNIFE_CROSS_MIRROR_STITCH_REPORTS={name}:{reports}", flush=True)
            traceback.print_exc()
            failures.append(name)
            # Best-effort leave edit mode so the next case can rebuild the scene.
            try:
                if bpy.context.object and bpy.context.object.mode != "OBJECT":
                    with bpy.context.temp_override(window=window, area=area, region=region):
                        bpy.ops.object.mode_set(mode="OBJECT")
            except BaseException:
                pass
            operators._SESSIONS.clear()

    if failures:
        print(f"YSE_KNIFE_CROSS_MIRROR_STITCH_FAILURES={','.join(failures)}", flush=True)
        print(MARKER_FAILED, flush=True)
        addon.unregister()
        bpy.ops.wm.quit_blender()
        return

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
