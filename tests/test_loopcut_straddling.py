# SPDX-License-Identifier: GPL-3.0-or-later

"""Phase 5: Loop Cut / Offset straddling-ring GUI regression (§4.2).

Cases:
  (a) Straddling ring (16-seg tube, self-mirrored faces) + off-center slide
      → full mirror pairs via the §4.2-3 face-coincidence path (INFO).
  (b) Two cuts on the same straddling ring → still fully mirrored.
  (c) Two hidden longitudinal edges on the cut ring → WARNING decline,
      native (asymmetric/partial) result kept.
  (d) Offset on the planar mid loop of a 3-ring tube, value 0.3 → ±X
      symmetric flanks (§4.2-6).
  (e) Non-self-mirrored straddling faces (twist=8 pair faces), if the
      fixture is globally X-symmetric: expect WARNING decline with
      "no exact mirrored counterpart" when that path fires; otherwise
      report that the fixture does not decline (recorded in the log).

Run with Blender's real window (not --background)::

    blender --factory-startup --enable-event-simulate --no-window-focus \\
        -p 40 40 960 600 --python test_loopcut_straddling.py
"""

from __future__ import annotations

import math
import os
import sys
import traceback
from collections import Counter
from pathlib import Path

import bmesh
import bpy
from mathutils import Quaternion

bpy.context.preferences.view.show_splash = False
bpy.context.preferences.view.smooth_view = 0
bpy.context.preferences.view.use_save_prompt = False
bpy.context.preferences.filepaths.use_auto_save_temporary_files = False

PACKAGE_PARENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_PARENT))

import ydd_symmetric_edit as addon  # noqa: E402
from ydd_symmetric_edit import core, operators  # noqa: E402

MARKER_OK = "YSE_LOOPCUT_STRADDLING_OK"
MARKER_FAILED = "YSE_LOOPCUT_STRADDLING_TEST_FAILED"
SEG = 16
RADIUS = 1.0
HALF = 1.5
SLIDE = 0.3
COORD_PRECISION = 5
HIDDEN_RING_WARNING = "hidden edges"
MIRROR_COUNTERPART_WARNING = "no exact mirrored counterpart"


def fail(message: str = "") -> None:
    if message:
        print(f"YSE_LOOPCUT_STRADDLING_ERROR={message}", flush=True)
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
    region_3d.view_rotation = Quaternion((0.70710678, 0.70710678, 0.0, 0.0))
    region_3d.view_location = (0.0, 0.0, 0.0)
    region_3d.view_distance = 6.0
    region_3d.update()


def coordinate_key(coordinate, precision: int = COORD_PRECISION):
    return tuple(round(float(value), precision) for value in coordinate)


def mirror_key(coordinate, precision: int = COORD_PRECISION):
    x, y, z = coordinate
    return coordinate_key((-float(x), float(y), float(z)), precision)


def clear_scene() -> None:
    for old in tuple(bpy.data.objects):
        bpy.data.objects.remove(old, do_unlink=True)
    for mesh in tuple(bpy.data.meshes):
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)


def build_tube_coords(seg: int = SEG, radius: float = RADIUS, half: float = HALF, twist: int = 0):
    coords = []
    for i in range(seg):
        theta = 2.0 * math.pi * i / seg
        coords.append((-half, radius * math.cos(theta), radius * math.sin(theta)))
    for i in range(seg):
        theta = 2.0 * math.pi * i / seg
        coords.append((half, radius * math.cos(theta), radius * math.sin(theta)))
    faces = []
    for i in range(seg):
        a = i
        b = (i + 1) % seg
        c = ((i + 1 + twist) % seg) + seg
        d = ((i + twist) % seg) + seg
        faces.append((a, b, c, d))
    return coords, faces


def build_tube3_coords(seg: int = SEG, radius: float = RADIUS, half: float = HALF):
    coords = []
    for x in (-half, 0.0, half):
        for i in range(seg):
            theta = 2.0 * math.pi * i / seg
            coords.append((x, radius * math.cos(theta), radius * math.sin(theta)))
    faces = []
    for ring in range(2):
        base = ring * seg
        nxt = (ring + 1) * seg
        for i in range(seg):
            a = base + i
            b = base + (i + 1) % seg
            c = nxt + (i + 1) % seg
            d = nxt + i
            faces.append((a, b, c, d))
    return coords, faces


def make_object(name: str, coords, faces, *, mirror_x: bool = True):
    mesh = bpy.data.meshes.new(name + "Mesh")
    mesh.from_pydata(coords, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    obj.use_mesh_mirror_x = mirror_x
    obj.use_mesh_mirror_y = False
    obj.use_mesh_mirror_z = False
    return obj


def warning_messages() -> list[str]:
    return [message for kind, message in operators._FINISH_REPORTS if kind == "WARNING"]


def info_messages() -> list[str]:
    return [message for kind, message in operators._FINISH_REPORTS if kind == "INFO"]


def error_messages() -> list[str]:
    return [message for kind, message in operators._FINISH_REPORTS if kind == "ERROR"]


def assert_no_errors() -> None:
    assert not error_messages(), operators._FINISH_REPORTS


def assert_full_x_symmetry(bm, *, precision: int = COORD_PRECISION) -> None:
    verts = Counter(coordinate_key(vertex.co, precision) for vertex in bm.verts)
    mirrored = Counter(mirror_key(key, precision) for key in verts.elements())
    assert verts == mirrored, (verts - mirrored, mirrored - verts)


def baseline_vertex_keys(coords, *, precision: int = COORD_PRECISION) -> set[tuple]:
    return {coordinate_key(coordinate, precision) for coordinate in coords}


def new_vertex_mirror_pairs(
    bm,
    baseline_keys: set[tuple],
    *,
    precision: int = COORD_PRECISION,
) -> tuple[int, int, int]:
    """Return (matched, unmatched, n_new) among verts not in the pre-cut baseline."""

    new_keys = [
        coordinate_key(vertex.co, precision)
        for vertex in bm.verts
        if coordinate_key(vertex.co, precision) not in baseline_keys
    ]
    key_set = set(new_keys)
    matched = 0
    unmatched = 0
    for key in new_keys:
        mirror = mirror_key(key, precision)
        if key == mirror or mirror in key_set:
            matched += 1
        else:
            unmatched += 1
    return matched, unmatched, len(new_keys)


def seed_longitudinal_edge_index(bm, seg: int = SEG, twist: int = 0) -> int:
    bm.verts.ensure_lookup_table()
    v0 = bm.verts[0]
    v1 = bm.verts[seg + (twist % seg)]
    edge = bm.edges.get((v0, v1))
    assert edge is not None, "seed longitudinal edge not found"
    return edge.index


def enter_edit(obj, window, area, region) -> bmesh.types.BMesh:
    with bpy.context.temp_override(window=window, area=area, region=region):
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.context.tool_settings.mesh_select_mode = (False, True, False)
        bpy.ops.mesh.select_all(action="DESELECT")
    return bmesh.from_edit_mesh(obj.data)


def run_loopcut_addon(
    window,
    area,
    region,
    *,
    edge_index: int,
    number_cuts: int = 1,
    value: float = SLIDE,
) -> set[str]:
    operators._FINISH_REPORTS.clear()
    with bpy.context.temp_override(window=window, area=area, region=region):
        prepared = operators._prepare_session(
            bpy.context,
            lambda _level, _message: None,
            tool_kind="LOOP_CUT",
        )
        assert prepared, "failed to prepare LOOP_CUT session"
        # EXEC_DEFAULT edge-slide on a straddling ring is pinned to the plane
        # while Mesh Symmetry is on (native value has no effect). Temporarily
        # clear the flag for the native macro only so the §4.2 off-center
        # fixture is reachable; the session still owns the axis from prepare.
        obj = bpy.context.edit_object
        assert obj is not None
        obj.use_mesh_mirror_x = False
        try:
            result = bpy.ops.mesh.loopcut_slide(
                "EXEC_DEFAULT",
                MESH_OT_loopcut={
                    "number_cuts": number_cuts,
                    "object_index": 0,
                    "edge_index": edge_index,
                    "mesh_select_mode_init": (False, True, False),
                },
                TRANSFORM_OT_edge_slide={"value": value},
            )
        finally:
            obj.use_mesh_mirror_x = True
        assert "FINISHED" in result, result
        finished = bpy.ops.mesh.ydd_symmetric_edit_finish("EXEC_DEFAULT")
        assert finished == {"FINISHED"}, finished
        return set(finished)


def run_offset_addon(window, area, region, *, value: float = SLIDE) -> set[str]:
    operators._FINISH_REPORTS.clear()
    with bpy.context.temp_override(window=window, area=area, region=region):
        prepared = operators._prepare_session(
            bpy.context,
            lambda _level, _message: None,
            tool_kind="OFFSET_LOOP_CUT",
        )
        assert prepared, "failed to prepare OFFSET_LOOP_CUT session"
        # Offset already suspends Mesh Symmetry in production; clear here too
        # so EXEC_DEFAULT value is applied consistently across Blender versions.
        obj = bpy.context.edit_object
        assert obj is not None
        obj.use_mesh_mirror_x = False
        try:
            result = bpy.ops.mesh.offset_edge_loops_slide(
                "EXEC_DEFAULT",
                MESH_OT_offset_edge_loops={"use_cap_endpoint": False},
                TRANSFORM_OT_edge_slide={"value": value},
            )
        finally:
            obj.use_mesh_mirror_x = True
        assert "FINISHED" in result, result
        finished = bpy.ops.mesh.ydd_symmetric_edit_finish("EXEC_DEFAULT")
        assert finished == {"FINISHED"}, finished
        return set(finished)


def case_a_self_mirror_straddle(window, area, region) -> None:
    """(a) Self-mirrored straddling faces + off-center → full pairs, INFO."""

    clear_scene()
    coords, faces = build_tube_coords(twist=0)
    baseline = baseline_vertex_keys(coords)
    obj = make_object("StraddleA", coords, faces)
    bm = enter_edit(obj, window, area, region)
    edge_index = seed_longitudinal_edge_index(bm)
    bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)

    run_loopcut_addon(window, area, region, edge_index=edge_index, number_cuts=1, value=SLIDE)
    assert_no_errors()
    assert not any(HIDDEN_RING_WARNING in message for message in warning_messages()), warning_messages()

    bm = bmesh.from_edit_mesh(obj.data)
    matched, unmatched, n_new = new_vertex_mirror_pairs(bm, baseline)
    assert unmatched == 0, (matched, unmatched, n_new)
    assert n_new == SEG * 2, (matched, unmatched, n_new)  # +X and -X rings, SEG each
    assert matched == SEG * 2, (matched, unmatched, n_new)
    assert_full_x_symmetry(bm)

    infos = info_messages()
    assert infos, operators._FINISH_REPORTS
    assert any("already contains" in message or "Mirrored" in message for message in infos), infos
    print(
        f"YSE_LOOPCUT_STRADDLING_CASE_A matched={matched} n_new={n_new} infos={infos}",
        flush=True,
    )


def case_b_multi_cut(window, area, region) -> None:
    """(b) Two cuts on straddling ring → full pairs.

    Default evenly-spaced 2-cuts sit on both sides of the plane; a larger
    slide value pushes both native rings onto one half so the one-side
    source path (§4.2) can mirror the whole result.
    """

    clear_scene()
    coords, faces = build_tube_coords(twist=0)
    baseline = baseline_vertex_keys(coords)
    obj = make_object("StraddleB", coords, faces)
    bm = enter_edit(obj, window, area, region)
    edge_index = seed_longitudinal_edge_index(bm)
    bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)

    multi_slide = 0.55
    run_loopcut_addon(window, area, region, edge_index=edge_index, number_cuts=2, value=multi_slide)
    assert_no_errors()
    assert not any(HIDDEN_RING_WARNING in message for message in warning_messages()), warning_messages()

    bm = bmesh.from_edit_mesh(obj.data)
    matched, unmatched, n_new = new_vertex_mirror_pairs(bm, baseline)
    assert unmatched == 0, (matched, unmatched, n_new)
    # 2 native rings * SEG + 2 mirrored rings * SEG
    assert n_new == SEG * 4, (matched, unmatched, n_new)
    assert matched == SEG * 4, (matched, unmatched, n_new)
    assert_full_x_symmetry(bm)
    assert info_messages(), operators._FINISH_REPORTS
    print(
        f"YSE_LOOPCUT_STRADDLING_CASE_B matched={matched} n_new={n_new} infos={info_messages()}",
        flush=True,
    )


def case_c_hidden_ring_edges(window, area, region) -> None:
    """(c) Hidden longitudinal edges on the cut ring → WARNING decline."""

    clear_scene()
    coords, faces = build_tube_coords(twist=0)
    baseline = baseline_vertex_keys(coords)
    obj = make_object("StraddleC", coords, faces)
    bm = enter_edit(obj, window, area, region)
    bm.verts.ensure_lookup_table()

    # Hide two longitudinal edges of the straddling ring (indices 0 and 1).
    hidden_edges = []
    for i in (0, 1):
        edge = bm.edges.get((bm.verts[i], bm.verts[SEG + i]))
        assert edge is not None
        edge.hide = True
        for vertex in edge.verts:
            vertex.hide = True
        for face in edge.link_faces:
            face.hide = True
        hidden_edges.append(edge.index)

    # Seed on a visible longitudinal edge so native still cuts a partial ring.
    edge_index = seed_longitudinal_edge_index(bm, twist=0)
    # Prefer edge 2 if the seed (0) was hidden.
    if edge_index in hidden_edges:
        edge = bm.edges.get((bm.verts[2], bm.verts[SEG + 2]))
        assert edge is not None and not edge.hide
        edge_index = edge.index
    bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)

    run_loopcut_addon(window, area, region, edge_index=edge_index, number_cuts=1, value=SLIDE)
    assert_no_errors()
    warnings = warning_messages()
    assert any(HIDDEN_RING_WARNING in message for message in warnings), (
        warnings,
        operators._FINISH_REPORTS,
    )
    assert not info_messages(), operators._FINISH_REPORTS

    bm = bmesh.from_edit_mesh(obj.data)
    matched, unmatched, n_new = new_vertex_mirror_pairs(bm, baseline)
    # Partial native ring is not re-mirrored; leave asymmetric / incomplete.
    assert unmatched > 0 or matched < SEG * 2, (matched, unmatched, n_new)
    print(
        f"YSE_LOOPCUT_STRADDLING_CASE_C matched={matched} unmatched={unmatched} n_new={n_new} warnings={warnings}",
        flush=True,
    )


def case_c_unrelated_hidden(window, area, region) -> None:
    """Hidden geometry on a different ring must not decline the cut ring."""

    clear_scene()
    # 3-ring tube: cut the left longitudinal strip, hide a circumferential
    # edge on the far +X end (different edge ring).
    coords, faces = build_tube3_coords()
    baseline = baseline_vertex_keys(coords)
    obj = make_object("StraddleC2", coords, faces)
    bm = enter_edit(obj, window, area, region)
    bm.verts.ensure_lookup_table()

    # Circumferential edge on the +X ring (verts 2*SEG .. 3*SEG-1).
    v_a = bm.verts[2 * SEG]
    v_b = bm.verts[2 * SEG + 1]
    circ = bm.edges.get((v_a, v_b))
    assert circ is not None
    circ.hide = True
    for vertex in circ.verts:
        vertex.hide = True

    # Seed a longitudinal edge on the left strip (x=-HALF..0).
    edge = bm.edges.get((bm.verts[0], bm.verts[SEG]))
    assert edge is not None and not edge.hide
    edge_index = edge.index
    bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)

    run_loopcut_addon(window, area, region, edge_index=edge_index, number_cuts=1, value=SLIDE)
    assert_no_errors()
    assert not any(HIDDEN_RING_WARNING in message for message in warning_messages()), warning_messages()
    bm = bmesh.from_edit_mesh(obj.data)
    matched, unmatched, n_new = new_vertex_mirror_pairs(bm, baseline)
    assert unmatched == 0, (matched, unmatched, n_new)
    print(
        f"YSE_LOOPCUT_STRADDLING_CASE_C_UNRELATED matched={matched} n_new={n_new}",
        flush=True,
    )


def case_d_offset_planar_loop(window, area, region) -> None:
    """(d) Offset on planar mid loop → structurally ±X symmetric flanks."""

    clear_scene()
    coords, faces = build_tube3_coords()
    obj = make_object("OffsetD", coords, faces)
    bm = enter_edit(obj, window, area, region)
    bm.verts.ensure_lookup_table()

    for element in (*bm.verts, *bm.edges, *bm.faces):
        element.select = False
    mid_base = SEG
    seed = None
    for i in range(SEG):
        v0 = bm.verts[mid_base + i]
        v1 = bm.verts[mid_base + (i + 1) % SEG]
        edge = bm.edges.get((v0, v1))
        assert edge is not None
        edge.select = True
        if seed is None:
            seed = edge
    bm.select_history.clear()
    assert seed is not None
    bm.select_history.add(seed)
    bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)

    run_offset_addon(window, area, region, value=SLIDE)
    assert_no_errors()
    assert not any(HIDDEN_RING_WARNING in message for message in warning_messages()), warning_messages()

    bm = bmesh.from_edit_mesh(obj.data)
    assert_full_x_symmetry(bm)

    # New flank loops should appear at ±offset-ish X (not only on the plane).
    xs = sorted({round(vertex.co.x, 4) for vertex in bm.verts})
    assert any(x < -1.0e-4 for x in xs) and any(x > 1.0e-4 for x in xs), xs
    # Every off-plane X must have its negative counterpart.
    off_plane = [x for x in xs if abs(x) > 1.0e-4]
    for x in off_plane:
        assert any(abs(x + other) <= 1.0e-3 for other in off_plane), (x, off_plane)
    print(
        f"YSE_LOOPCUT_STRADDLING_CASE_D xs={xs} infos={info_messages()} warnings={warning_messages()}",
        flush=True,
    )


def case_e_non_self_mirror_faces(window, area, region) -> None:
    """(e) Try twist=8 pair-face straddling fixture; record decline or not."""

    clear_scene()
    coords, faces = build_tube_coords(twist=SEG // 2)
    obj = make_object("StraddleE", coords, faces)
    bm = enter_edit(obj, window, area, region)

    # Preflight: mesh must be X-symmetric and faces must not all be self-mirrored.
    topology = core.prepare_topology(bm, core.AXIS_INDEX["X"], 1.0e-5)
    self_mirrored = sum(1 for source, target in topology.mirror_face_ids.items() if source == target)
    paired = sum(1 for source, target in topology.mirror_face_ids.items() if source != target)
    total = topology.total_faces
    matched = topology.matched_faces
    core.remove_temporary_layers(bm)
    bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)

    print(
        f"YSE_LOOPCUT_STRADDLING_CASE_E_PREFLIGHT total_faces={total} matched={matched} "
        f"self_mirrored={self_mirrored} paired={paired}",
        flush=True,
    )

    if matched < total or paired == 0:
        print(
            "YSE_LOOPCUT_STRADDLING_CASE_E_SKIP reason=fixture not a complete non-self-mirror pair mesh "
            f"(matched={matched}/{total}, paired={paired})",
            flush=True,
        )
        return

    edge_index = seed_longitudinal_edge_index(bm, twist=SEG // 2)
    bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
    baseline = baseline_vertex_keys(coords)

    run_loopcut_addon(window, area, region, edge_index=edge_index, number_cuts=1, value=SLIDE)
    assert_no_errors()
    warnings = warning_messages()
    infos = info_messages()
    bm = bmesh.from_edit_mesh(obj.data)
    matched_pairs, unmatched, n_new = new_vertex_mirror_pairs(bm, baseline)

    if any(MIRROR_COUNTERPART_WARNING in message for message in warnings):
        print(
            f"YSE_LOOPCUT_STRADDLING_CASE_E_DECLINED warnings={warnings} "
            f"matched={matched_pairs} unmatched={unmatched} n_new={n_new}",
            flush=True,
        )
        return

    # Fixture built and is fully paired, but the existing path did not decline.
    # Pairing holds so the unmatched-face decline is not reached here.
    print(
        f"YSE_LOOPCUT_STRADDLING_CASE_E_NO_DECLINE infos={infos} warnings={warnings} "
        f"matched={matched_pairs} unmatched={unmatched} n_new={n_new} "
        "note=pair-face straddling fixture is X-symmetric; mirror path did not hit "
        "'no exact mirrored counterpart' (pairing holds)",
        flush=True,
    )


def run_test() -> None:
    addon.register()
    window, area, region = viewport_context()
    configure_view(area)

    case_a_self_mirror_straddle(window, area, region)
    case_b_multi_cut(window, area, region)
    case_c_hidden_ring_edges(window, area, region)
    case_c_unrelated_hidden(window, area, region)
    case_d_offset_planar_loop(window, area, region)
    case_e_non_self_mirror_faces(window, area, region)

    print(MARKER_OK, flush=True)
    addon.unregister()
    bpy.ops.wm.quit_blender()


def start() -> None:
    try:
        run_test()
    except BaseException:
        fail()


bpy.app.timers.register(start, first_interval=0.25)
