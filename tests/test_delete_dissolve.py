# SPDX-License-Identifier: GPL-3.0-or-later

"""GUI regression for symmetric dissolve operators and Ctrl+X route.

Run with Blender's real window/event loop (user keyconfig is empty in
``--background``)::

    blender --factory-startup --enable-event-simulate --no-window-focus \\
        -p 40 40 960 600 --python test_delete_dissolve.py

Cases (serial, timer-driven):
  (a) After enable, addon keyconfig has dissolve_mode replacement on factory X
      (Ctrl+X event).
  (b) dissolve_verts: one-side internal vertex → both sides dissolve, X-symmetric.
  (c) dissolve_edges: one-side internal edge → both sides dissolve, X-symmetric.
  (d) dissolve_faces: two adjacent +X faces → both sides merge, X-symmetric.
  (e) dissolve_mode dispatch: vert select mode → VERTS-equivalent result.
  (f) hidden mirror vertex → CANCELLED + WARNING, mesh unchanged.
  (g) fault-injected asymmetric native → WARNING rolled back, topology restored.
  (h) after (d), one undo restores pre-dissolve topology counts.
  (i) symmetry axes off → native passthrough (one side only).
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
from ydd_symmetric_edit import delete_dissolve as yse_delete  # noqa: E402
from ydd_symmetric_edit import keymaps as yse_keymaps  # noqa: E402
from ydd_symmetric_edit import ui as yse_ui  # noqa: E402

MARKER_OK = "YSE_DELETE_DISSOLVE_OK"
MARKER_FAILED = "YSE_DELETE_DISSOLVE_FAILED"
COORD_PRECISION = 5
# 8×4 face grid → 9×5 verts, X-symmetric about the origin.
GRID_X_SEGMENTS = 8
GRID_Y_SEGMENTS = 4
GRID_SIZE = 2.0
STATE: dict = {}


def fail(message: str = "") -> None:
    if message:
        print(f"YSE_DELETE_DISSOLVE_ERROR={message}", flush=True)
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


def assert_x_symmetric(bm, *, label: str) -> None:
    verts = vertex_coord_multiset(bm)
    edges = edge_coord_multiset(bm)
    assert verts == mirrored_vertex_multiset(bm), f"{label}: vertex coords not X-symmetric: {verts}"
    assert edges == mirrored_edge_multiset(bm), f"{label}: edges not X-symmetric: {edges}"


def topology_counts(bm) -> tuple[int, int, int]:
    return len(bm.verts), len(bm.edges), len(bm.faces)


def clear_selection(bm) -> None:
    for face in bm.faces:
        face.select = False
    for edge in bm.edges:
        edge.select = False
    for vertex in bm.verts:
        vertex.select = False
    bm.select_history.clear()


def set_cube_symmetry(obj, *, x: bool = True, y: bool = False, z: bool = False) -> None:
    obj.use_mesh_mirror_x = x
    obj.use_mesh_mirror_y = y
    obj.use_mesh_mirror_z = z


def active_mesh_object():
    obj = bpy.context.view_layer.objects.active
    if obj is not None and obj.type == "MESH":
        return obj
    for candidate in bpy.data.objects:
        if candidate.type == "MESH":
            return candidate
    return None


def ensure_edit(window, area, region, obj):
    if obj.mode != "EDIT":
        bpy.context.view_layer.objects.active = obj
        with bpy.context.temp_override(window=window, area=area, region=region):
            bpy.ops.object.mode_set(mode="EDIT")
    return bmesh.from_edit_mesh(obj.data)


def set_addon_enabled(value: bool) -> None:
    """Toggle via preferences when available; dev registration lacks the
    addons entry, so fall back to the keymap sync the update callback runs."""

    preferences = yse_ui.get_addon_preferences(bpy.context)
    if preferences is not None:
        preferences.enabled = value
    else:
        addon.sync_persistent_keymap(value)


def dissolve_warnings() -> list[str]:
    return [message for kind, message in yse_delete._DELETE_REPORTS if kind == "WARNING"]


def dissolve_mode_addon_routes() -> list[tuple[str, str, int, bool]]:
    """Return (keymap_name, event_type, ctrl, active) for dissolve_mode replacement."""

    addon_config = bpy.context.window_manager.keyconfigs.addon
    assert addon_config is not None, "addon keyconfig missing"
    found = []
    for keymap in addon_config.keymaps:
        if keymap.is_modal:
            continue
        for item in keymap.keymap_items:
            if item.idname != yse_keymaps.DISSOLVE_MODE_OPERATOR:
                continue
            found.append((keymap.name, str(item.type), int(item.ctrl), bool(item.active)))
    return found


def replace_with_symmetric_grid(window, area, region, *, push_undo: bool = True):
    """Replace the active mesh with an X-symmetric grid and enter edit mode."""

    with bpy.context.temp_override(window=window, area=area, region=region):
        if bpy.context.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")

    for old in tuple(bpy.data.objects):
        if old.type == "MESH":
            bpy.data.objects.remove(old, do_unlink=True)

    with bpy.context.temp_override(window=window, area=area, region=region):
        bpy.ops.mesh.primitive_plane_add(location=(0.0, 0.0, 0.0))
    obj = bpy.context.view_layer.objects.active
    assert obj is not None
    set_cube_symmetry(obj, x=True)

    bm = ensure_edit(window, area, region, obj)
    bmesh.ops.delete(bm, geom=list(bm.verts), context="VERTS")
    bmesh.ops.create_grid(
        bm,
        x_segments=GRID_X_SEGMENTS,
        y_segments=GRID_Y_SEGMENTS,
        size=GRID_SIZE,
    )
    bmesh.update_edit_mesh(obj.data, loop_triangles=True, destructive=True)
    bm = bmesh.from_edit_mesh(obj.data)
    assert_x_symmetric(bm, label="grid baseline")
    counts = topology_counts(bm)
    # (x_seg+1)*(y_seg+1) verts, etc.
    expected_verts = (GRID_X_SEGMENTS + 1) * (GRID_Y_SEGMENTS + 1)
    assert counts[0] == expected_verts, counts

    if push_undo:
        with bpy.context.temp_override(window=window, area=area, region=region):
            bpy.ops.ed.undo_push(message="YSE Dissolve grid baseline")

    return obj, counts


def find_internal_plus_x_vertex(bm):
    """Interior (+X) grid vertex: valence 4, x > 0, not on the symmetry plane."""

    candidates = []
    for vertex in bm.verts:
        if len(vertex.link_edges) != 4:
            continue
        x = float(vertex.co.x)
        if x <= 1e-6:
            continue
        candidates.append(vertex)
    assert candidates, "no internal +X vertex found"
    # Prefer the vertex closest to the origin in |y| for stability.
    candidates.sort(key=lambda v: (abs(float(v.co.y)), float(v.co.x)))
    return candidates[0]


def find_internal_plus_x_edge(bm):
    """Interior edge entirely on +X (both endpoints x > 0, non-boundary)."""

    candidates = []
    for edge in bm.edges:
        if edge.is_boundary:
            continue
        v0, v1 = edge.verts
        if float(v0.co.x) <= 1e-6 or float(v1.co.x) <= 1e-6:
            continue
        if len(v0.link_edges) < 3 or len(v1.link_edges) < 3:
            continue
        candidates.append(edge)
    assert candidates, "no internal +X edge found"
    candidates.sort(
        key=lambda e: (
            abs(float(e.verts[0].co.y) + float(e.verts[1].co.y)),
            float(e.verts[0].co.x) + float(e.verts[1].co.x),
        )
    )
    return candidates[0]


def find_adjacent_plus_x_faces(bm):
    """Two faces sharing an edge, both with centroid x > 0."""

    plus_faces = []
    for face in bm.faces:
        center = face.calc_center_median()
        if float(center.x) > 1e-6:
            plus_faces.append(face)
    assert len(plus_faces) >= 2, "need at least two +X faces"

    for face in plus_faces:
        for edge in face.edges:
            linked = [f for f in edge.link_faces if f in plus_faces and f is not face]
            if not linked:
                continue
            other = linked[0]
            # Prefer a pair fully off the mid-plane (more interior).
            c0 = face.calc_center_median()
            c1 = other.calc_center_median()
            if float(c0.x) > 0.2 and float(c1.x) > 0.2:
                return face, other
    # Fallback: any adjacent +X pair.
    for face in plus_faces:
        for edge in face.edges:
            linked = [f for f in edge.link_faces if f in plus_faces and f is not face]
            if linked:
                return face, linked[0]
    raise AssertionError("no adjacent +X face pair found")


def case_a_routes() -> None:
    routes = dissolve_mode_addon_routes()
    print(f"YSE_DELETE_DISSOLVE_SCAN={routes}", flush=True)
    types = {event_type for _name, event_type, _ctrl, _active in routes}
    assert routes, "no dissolve_mode replacement routes registered"
    # Factory binds dissolve_mode to Ctrl+X (and often Ctrl+DEL).
    assert "X" in types, f"missing X dissolve_mode route: {routes}"
    x_routes = [r for r in routes if r[1] == "X"]
    assert any(ctrl for _n, _t, ctrl, _a in x_routes), f"X route lacks ctrl: {x_routes}"
    assert all(active for _n, _t, _c, active in routes), f"routes not active: {routes}"
    print("YSE_DELETE_DISSOLVE_A_OK", flush=True)


def case_b_dissolve_verts(window, area, region, obj) -> None:
    bm = ensure_edit(window, area, region, obj)
    set_cube_symmetry(obj, x=True)
    before = topology_counts(bm)
    clear_selection(bm)
    target = find_internal_plus_x_vertex(bm)
    target_key = coordinate_key(target.co)
    mirror = coordinate_key((-float(target.co.x), float(target.co.y), float(target.co.z)))
    target.select = True
    bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)

    with bpy.context.temp_override(window=window, area=area, region=region):
        result = bpy.ops.mesh.ydd_symmetric_edit_dissolve(mode="VERTS")
    assert result == {"FINISHED"}, result

    bm = bmesh.from_edit_mesh(obj.data)
    after = topology_counts(bm)
    present = vertex_coord_multiset(bm)
    assert target_key not in present, present
    assert mirror not in present, present
    # Both sides dissolved → at least 2 verts gone (may cascade slightly less/more
    # depending on dissolve connectivity; require strict decrease of 2 for grid interior).
    assert after[0] == before[0] - 2, (before, after)
    assert_x_symmetric(bm, label="b after dissolve verts")
    print(f"YSE_DELETE_DISSOLVE_B_COUNTS={before}->{after}", flush=True)
    print("YSE_DELETE_DISSOLVE_B_OK", flush=True)


def case_c_dissolve_edges(window, area, region) -> None:
    obj, before = replace_with_symmetric_grid(window, area, region)
    bm = ensure_edit(window, area, region, obj)
    clear_selection(bm)
    edge = find_internal_plus_x_edge(bm)
    edge_key = tuple(sorted(coordinate_key(v.co) for v in edge.verts))
    mirror_edge_key = tuple(sorted(mirror_key(v.co) for v in edge.verts))
    edge.select = True
    bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)

    with bpy.context.temp_override(window=window, area=area, region=region):
        result = bpy.ops.mesh.ydd_symmetric_edit_dissolve(mode="EDGES", use_verts=True)
    assert result == {"FINISHED"}, result

    bm = bmesh.from_edit_mesh(obj.data)
    after = topology_counts(bm)
    edges = edge_coord_multiset(bm)
    assert edge_key not in edges, edges
    assert mirror_edge_key not in edges, edges
    assert after[1] < before[1], (before, after)
    assert_x_symmetric(bm, label="c after dissolve edges")
    print(f"YSE_DELETE_DISSOLVE_C_COUNTS={before}->{after}", flush=True)
    print("YSE_DELETE_DISSOLVE_C_OK", flush=True)


def case_d_dissolve_faces(window, area, region) -> tuple[object, tuple[int, int, int]]:
    obj, before = replace_with_symmetric_grid(window, area, region)
    bm = ensure_edit(window, area, region, obj)
    clear_selection(bm)
    face_a, face_b = find_adjacent_plus_x_faces(bm)
    face_a.select = True
    face_b.select = True
    bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)

    with bpy.context.temp_override(window=window, area=area, region=region):
        result = bpy.ops.mesh.ydd_symmetric_edit_dissolve(mode="FACES", use_verts=False)
    assert result == {"FINISHED"}, result

    bm = bmesh.from_edit_mesh(obj.data)
    after = topology_counts(bm)
    # Bilateral merge of two adjacent pairs: native dissolve_faces on a grid
    # may drop more than one face per side (shared-edge collapse of two
    # components can cascade on this topology). Require at least one face
    # gone per side and full X symmetry.
    assert after[2] <= before[2] - 2, (before, after)
    assert after[2] < before[2], (before, after)
    assert_x_symmetric(bm, label="d after dissolve faces")
    print(f"YSE_DELETE_DISSOLVE_D_COUNTS={before}->{after}", flush=True)
    print("YSE_DELETE_DISSOLVE_D_OK", flush=True)
    return obj, before


def case_h_undo(window, area, region, baseline: tuple[int, int, int]) -> None:
    with bpy.context.temp_override(window=window, area=area, region=region):
        undo_result = bpy.ops.ed.undo()
    assert undo_result == {"FINISHED"}, undo_result

    obj = active_mesh_object()
    assert obj is not None, "object missing after undo"
    bm = ensure_edit(window, area, region, obj)
    counts = topology_counts(bm)
    print(f"YSE_DELETE_DISSOLVE_H_COUNTS={counts}", flush=True)
    assert counts == baseline, f"undo did not restore baseline in 1 step: {counts} != {baseline}"
    assert_x_symmetric(bm, label="h after undo")
    print("YSE_DELETE_DISSOLVE_H_OK", flush=True)


def case_e_dissolve_mode(window, area, region) -> None:
    obj, before = replace_with_symmetric_grid(window, area, region)
    bm = ensure_edit(window, area, region, obj)
    clear_selection(bm)
    target = find_internal_plus_x_vertex(bm)
    target_key = coordinate_key(target.co)
    mirror = coordinate_key((-float(target.co.x), float(target.co.y), float(target.co.z)))
    target.select = True
    bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)

    with bpy.context.temp_override(window=window, area=area, region=region):
        bpy.context.tool_settings.mesh_select_mode = (True, False, False)
        result = bpy.ops.mesh.ydd_symmetric_edit_dissolve_mode()
    assert result == {"FINISHED"}, result

    bm = bmesh.from_edit_mesh(obj.data)
    after = topology_counts(bm)
    present = vertex_coord_multiset(bm)
    assert target_key not in present, present
    assert mirror not in present, present
    assert after[0] == before[0] - 2, (before, after)
    assert_x_symmetric(bm, label="e after dissolve_mode")
    print(f"YSE_DELETE_DISSOLVE_E_COUNTS={before}->{after}", flush=True)
    print("YSE_DELETE_DISSOLVE_E_OK", flush=True)


def case_f_hidden(window, area, region) -> None:
    obj, before = replace_with_symmetric_grid(window, area, region)
    bm = ensure_edit(window, area, region, obj)
    before_verts = vertex_coord_multiset(bm)
    clear_selection(bm)
    target = find_internal_plus_x_vertex(bm)
    mirror_co = Vector((-float(target.co.x), float(target.co.y), float(target.co.z)))
    mirror_vert = None
    for vertex in bm.verts:
        if (vertex.co - mirror_co).length < 1e-5:
            mirror_vert = vertex
            break
    assert mirror_vert is not None, "mirror vertex missing"
    mirror_vert.hide = True
    target.select = True
    bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)

    yse_delete._DELETE_REPORTS.clear()
    with bpy.context.temp_override(window=window, area=area, region=region):
        result = bpy.ops.mesh.ydd_symmetric_edit_dissolve(mode="VERTS")
    assert result == {"CANCELLED"}, result
    warnings = dissolve_warnings()
    assert any("hidden" in message for message in warnings), warnings

    bm = bmesh.from_edit_mesh(obj.data)
    assert topology_counts(bm) == before, topology_counts(bm)
    assert vertex_coord_multiset(bm) == before_verts
    for vertex in bm.verts:
        vertex.hide = False
    bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
    print("YSE_DELETE_DISSOLVE_F_OK", flush=True)


def case_g_fault_injection(window, area, region) -> None:
    obj, before = replace_with_symmetric_grid(window, area, region)
    bm = ensure_edit(window, area, region, obj)
    before_verts = vertex_coord_multiset(bm)
    clear_selection(bm)
    target = find_internal_plus_x_vertex(bm)
    target.select = True
    bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)

    original = yse_delete._native_dissolve_call

    def _faulty(mode, options):
        result = original(mode, options)
        # After a successful native dissolve, delete one extra +X-only vertex
        # so the post-dissolve census becomes asymmetric.
        edit_obj = bpy.context.edit_object
        if edit_obj is None:
            return result
        mesh = edit_obj.data
        live = bmesh.from_edit_mesh(mesh)
        extra = None
        for vertex in live.verts:
            if float(vertex.co.x) > 0.3 and len(vertex.link_edges) >= 2:
                extra = vertex
                break
        if extra is not None:
            bmesh.ops.delete(live, geom=[extra], context="VERTS")
            bmesh.update_edit_mesh(mesh, loop_triangles=True, destructive=True)
        return result

    yse_delete._native_dissolve_call = _faulty
    try:
        yse_delete._DELETE_REPORTS.clear()
        with bpy.context.temp_override(window=window, area=area, region=region):
            result = bpy.ops.mesh.ydd_symmetric_edit_dissolve(mode="VERTS")
        assert result == {"FINISHED"}, result
        warnings = dissolve_warnings()
        assert any("rolled back" in message for message in warnings), warnings

        bm = bmesh.from_edit_mesh(obj.data)
        after = topology_counts(bm)
        assert after == before, f"rollback did not restore topology: {after} != {before}"
        assert vertex_coord_multiset(bm) == before_verts
        print(f"YSE_DELETE_DISSOLVE_G_WARNINGS={warnings}", flush=True)
        print("YSE_DELETE_DISSOLVE_G_OK", flush=True)
    finally:
        yse_delete._native_dissolve_call = original


def case_i_passthrough(window, area, region) -> None:
    obj, before = replace_with_symmetric_grid(window, area, region)
    set_cube_symmetry(obj, x=False, y=False, z=False)
    bm = ensure_edit(window, area, region, obj)
    clear_selection(bm)
    target = find_internal_plus_x_vertex(bm)
    target_key = coordinate_key(target.co)
    mirror = coordinate_key((-float(target.co.x), float(target.co.y), float(target.co.z)))
    target.select = True
    bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)

    with bpy.context.temp_override(window=window, area=area, region=region):
        result = bpy.ops.mesh.ydd_symmetric_edit_dissolve(mode="VERTS")
    assert result == {"FINISHED"}, result

    bm = bmesh.from_edit_mesh(obj.data)
    after = topology_counts(bm)
    present = vertex_coord_multiset(bm)
    assert target_key not in present, present
    # Mirror side must remain (no expansion).
    assert mirror in present, present
    assert after[0] == before[0] - 1, (before, after)
    print(f"YSE_DELETE_DISSOLVE_I_COUNTS={before}->{after}", flush=True)
    print("YSE_DELETE_DISSOLVE_I_OK", flush=True)


def run_all() -> None:
    window, area, region = STATE["window"], STATE["area"], STATE["region"]

    case_a_routes()
    obj, _baseline = replace_with_symmetric_grid(window, area, region)
    case_b_dissolve_verts(window, area, region, obj)
    case_c_dissolve_edges(window, area, region)
    _obj_d, baseline_d = case_d_dissolve_faces(window, area, region)
    case_h_undo(window, area, region, baseline_d)
    case_e_dissolve_mode(window, area, region)
    case_f_hidden(window, area, region)
    case_g_fault_injection(window, area, region)
    case_i_passthrough(window, area, region)


def phase_run():
    try:
        run_all()
        print(MARKER_OK, flush=True)
        addon.unregister()
        bpy.ops.wm.quit_blender()
    except BaseException:
        fail()
    return None


def phase_setup():
    try:
        addon.register()
        preferences = yse_ui.get_addon_preferences(bpy.context)
        if preferences is not None:
            preferences.enabled = True
        else:
            addon.sync_persistent_keymap(True)

        # Force a full rebuild now that the GUI user keyconfig is populated.
        addon.sync_persistent_keymap(True)

        window, area, region = viewport_context()
        configure_view(area)

        # Factory Cube is a stable starting object for mode/edit transitions.
        obj = bpy.data.objects["Cube"]
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        set_cube_symmetry(obj, x=True)
        for name in ("Camera", "Light"):
            other = bpy.data.objects.get(name)
            if other is not None:
                other.select_set(False)

        with bpy.context.temp_override(window=window, area=area, region=region):
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.context.tool_settings.mesh_select_mode = (True, False, False)
            bpy.ops.ed.undo_push(message="YSE Dissolve setup baseline")

        STATE.update(window=window, area=area, region=region, object=obj)
        bpy.app.timers.register(phase_run, first_interval=0.2)
    except BaseException:
        fail()
    return None


bpy.context.preferences.view.show_splash = False
bpy.context.preferences.view.smooth_view = 0
bpy.context.preferences.view.use_save_prompt = False
bpy.context.preferences.filepaths.use_auto_save_temporary_files = False
bpy.app.timers.register(phase_setup, first_interval=0.5)
