# SPDX-License-Identifier: GPL-3.0-or-later

"""GUI regression for delete-menu keymap routes and symmetric delete.

Run with Blender's real window/event loop (user keyconfig is empty in
``--background``)::

    blender --factory-startup --enable-event-simulate --no-window-focus \\
        -p 40 40 960 600 --python test_delete_routes.py

Cases (serial, timer-driven):
  (a) After enable, addon keyconfig has YSE_MT_delete on factory X and DEL.
  (b) Symmetric cube, one side face selected → both faces deleted, X-symmetric.
  (c) Hidden mirror face → CANCELLED + WARNING, mesh unchanged.
  (d) Unmatched vertex in selection → FINISHED + INFO, matched pair still both go.
  (e) After (b), one undo restores pre-delete topology counts.
  (f) preferences.enabled=False → registered delete routes have active=False.
  (g) All mesh mirror axes off → native passthrough (one side only deleted).
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
from ydd_symmetric_edit import delete_dissolve as yse_delete  # noqa: E402
from ydd_symmetric_edit import keymaps as yse_keymaps  # noqa: E402
from ydd_symmetric_edit import ui as yse_ui  # noqa: E402

MARKER_OK = "YSE_DELETE_ROUTES_OK"
MARKER_FAILED = "YSE_DELETE_ROUTES_FAILED"
COORD_PRECISION = 5
STATE: dict = {}


def fail(message: str = "") -> None:
    if message:
        print(f"YSE_DELETE_ROUTES_ERROR={message}", flush=True)
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


def face_centroid(face) -> tuple[float, float, float]:
    total = face.calc_center_median()
    return float(total.x), float(total.y), float(total.z)


def find_face_by_centroid_x(bm, expected_x: float, *, precision: int = 4):
    for face in bm.faces:
        cx, _cy, _cz = face_centroid(face)
        if round(cx, precision) == round(expected_x, precision):
            return face
    raise AssertionError(f"face with centroid x={expected_x} not found")


def has_face_at_centroid_x(bm, expected_x: float, *, precision: int = 4) -> bool:
    for face in bm.faces:
        cx, _cy, _cz = face_centroid(face)
        if round(cx, precision) == round(expected_x, precision):
            return True
    return False


def find_vertex(bm, expected, precision: int = COORD_PRECISION):
    key = coordinate_key(expected, precision)
    for vertex in bm.verts:
        if coordinate_key(vertex.co, precision) == key:
            return vertex
    raise AssertionError(f"vertex not found: {expected}")


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


def delete_menu_addon_routes() -> list[tuple[str, str, bool]]:
    """Return (keymap_name, event_type, active) for YSE_MT_delete bindings."""

    addon_config = bpy.context.window_manager.keyconfigs.addon
    assert addon_config is not None, "addon keyconfig missing"
    found = []
    for keymap in addon_config.keymaps:
        if keymap.is_modal:
            continue
        for item in keymap.keymap_items:
            if item.idname != "wm.call_menu":
                continue
            if getattr(item.properties, "name", "") != yse_keymaps.DELETE_MENU:
                continue
            found.append((keymap.name, str(item.type), bool(item.active)))
    return found


def delete_warnings() -> list[str]:
    return [message for kind, message in yse_delete._DELETE_REPORTS if kind == "WARNING"]


def delete_infos() -> list[str]:
    return [message for kind, message in yse_delete._DELETE_REPORTS if kind == "INFO"]


def case_a_routes() -> None:
    routes = delete_menu_addon_routes()
    types = {event_type for _name, event_type, _active in routes}
    print(f"YSE_DELETE_ROUTES_SCAN={routes}", flush=True)
    assert yse_keymaps.has_delete_routes(), "has_delete_routes() is False after enable"
    assert "X" in types, f"missing X delete route: {routes}"
    assert "DEL" in types, f"missing DEL delete route: {routes}"
    assert all(active for _name, _type, active in routes), f"routes not active: {routes}"
    print("YSE_DELETE_ROUTES_A_OK", flush=True)


def case_b_face_delete(window, area, region, obj) -> tuple[int, int, int]:
    bm = ensure_edit(window, area, region, obj)
    set_cube_symmetry(obj, x=True)
    baseline = topology_counts(bm)
    assert baseline == (8, 12, 6), baseline
    assert_x_symmetric(bm, label="b baseline")

    clear_selection(bm)
    plus_face = find_face_by_centroid_x(bm, 1.0)
    plus_face.select = True
    bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)

    with bpy.context.temp_override(window=window, area=area, region=region):
        result = bpy.ops.mesh.ydd_symmetric_edit_delete(type="FACE")
    assert result == {"FINISHED"}, result

    bm = bmesh.from_edit_mesh(obj.data)
    counts = topology_counts(bm)
    # Both ±X faces removed.
    assert counts[2] == baseline[2] - 2, counts
    assert_x_symmetric(bm, label="b after face delete")
    assert not has_face_at_centroid_x(bm, 1.0), "+X face still present after symmetric delete"
    assert not has_face_at_centroid_x(bm, -1.0), "-X face still present after symmetric delete"
    print(f"YSE_DELETE_ROUTES_B_COUNTS={counts}", flush=True)
    print("YSE_DELETE_ROUTES_B_OK", flush=True)
    return baseline


def case_e_undo(window, area, region, obj, baseline: tuple[int, int, int]) -> None:
    with bpy.context.temp_override(window=window, area=area, region=region):
        undo_result = bpy.ops.ed.undo()
    assert undo_result == {"FINISHED"}, undo_result

    obj = active_mesh_object()
    assert obj is not None, "object missing after undo"
    bm = ensure_edit(window, area, region, obj)
    counts = topology_counts(bm)
    print(f"YSE_DELETE_ROUTES_E_COUNTS={counts}", flush=True)
    assert counts == baseline, f"undo did not restore baseline in 1 step: {counts} != {baseline}"
    assert_x_symmetric(bm, label="e after undo")
    print("YSE_DELETE_ROUTES_E_OK", flush=True)


def case_c_hidden(window, area, region, obj) -> None:
    bm = ensure_edit(window, area, region, obj)
    set_cube_symmetry(obj, x=True)
    before = topology_counts(bm)
    before_verts = vertex_coord_multiset(bm)

    clear_selection(bm)
    plus_face = find_face_by_centroid_x(bm, 1.0)
    minus_face = find_face_by_centroid_x(bm, -1.0)
    minus_face.hide = True
    plus_face.select = True
    bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)

    yse_delete._DELETE_REPORTS.clear()
    with bpy.context.temp_override(window=window, area=area, region=region):
        result = bpy.ops.mesh.ydd_symmetric_edit_delete(type="FACE")
    assert result == {"CANCELLED"}, result
    warnings = delete_warnings()
    assert any("hidden" in message for message in warnings), warnings

    bm = bmesh.from_edit_mesh(obj.data)
    assert topology_counts(bm) == before, topology_counts(bm)
    assert vertex_coord_multiset(bm) == before_verts
    # Cleanup hide for later cases.
    for face in bm.faces:
        face.hide = False
    bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
    print("YSE_DELETE_ROUTES_C_OK", flush=True)


def case_d_unmatched(window, area, region, obj) -> None:
    bm = ensure_edit(window, area, region, obj)
    set_cube_symmetry(obj, x=True)
    # Add a free +X-only vertex with no mirror counterpart.
    extra = bm.verts.new((2.5, 0.0, 0.0))
    bm.verts.ensure_lookup_table()
    bm.verts.index_update()
    clear_selection(bm)
    # Select the unmatched extra + one cube corner that has a mirror.
    extra.select = True
    corner = find_vertex(bm, (1.0, -1.0, -1.0))
    corner.select = True
    bmesh.update_edit_mesh(obj.data, loop_triangles=True, destructive=True)

    before_faces = len(bm.faces)
    yse_delete._DELETE_REPORTS.clear()
    with bpy.context.temp_override(window=window, area=area, region=region):
        result = bpy.ops.mesh.ydd_symmetric_edit_delete(type="VERT")
    assert result == {"FINISHED"}, result

    infos = delete_infos()
    assert any("no mirrored counterpart" in message for message in infos), infos

    bm = bmesh.from_edit_mesh(obj.data)
    present = vertex_coord_multiset(bm)
    # Extra and both mirrored corners should be gone.
    assert coordinate_key((2.5, 0.0, 0.0)) not in present, present
    assert coordinate_key((1.0, -1.0, -1.0)) not in present, present
    assert coordinate_key((-1.0, -1.0, -1.0)) not in present, present
    # Other cube corners remain.
    assert coordinate_key((1.0, 1.0, -1.0)) in present, present
    assert coordinate_key((-1.0, 1.0, -1.0)) in present, present
    assert len(bm.faces) < before_faces
    print(f"YSE_DELETE_ROUTES_D_INFO={infos}", flush=True)
    print("YSE_DELETE_ROUTES_D_OK", flush=True)


def set_addon_enabled(value: bool) -> None:
    """Toggle via preferences when available; dev registration lacks the
    addons entry, so fall back to the keymap sync the update callback runs."""

    preferences = yse_ui.get_addon_preferences(bpy.context)
    if preferences is not None:
        preferences.enabled = value
    else:
        addon.sync_persistent_keymap(value)


def case_f_disable() -> None:
    set_addon_enabled(False)
    routes = delete_menu_addon_routes()
    assert routes, "delete routes disappeared entirely on disable"
    assert all(not active for _name, _type, active in routes), f"routes still active: {routes}"
    print(f"YSE_DELETE_ROUTES_F={routes}", flush=True)
    print("YSE_DELETE_ROUTES_F_OK", flush=True)


def case_g_passthrough(window, area, region) -> None:
    set_addon_enabled(True)

    # Leave edit mode, replace with a fresh cube for a clean one-side delete.
    with bpy.context.temp_override(window=window, area=area, region=region):
        if bpy.context.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
    for old in tuple(bpy.data.objects):
        if old.type == "MESH":
            bpy.data.objects.remove(old, do_unlink=True)
    with bpy.context.temp_override(window=window, area=area, region=region):
        bpy.ops.mesh.primitive_cube_add(location=(0.0, 0.0, 0.0))
    obj = bpy.context.view_layer.objects.active
    assert obj is not None
    set_cube_symmetry(obj, x=False, y=False, z=False)

    bm = ensure_edit(window, area, region, obj)
    before = topology_counts(bm)
    clear_selection(bm)
    plus_face = find_face_by_centroid_x(bm, 1.0)
    plus_face.select = True
    bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)

    with bpy.context.temp_override(window=window, area=area, region=region):
        result = bpy.ops.mesh.ydd_symmetric_edit_delete(type="FACE")
    assert result == {"FINISHED"}, result

    bm = bmesh.from_edit_mesh(obj.data)
    counts = topology_counts(bm)
    assert counts[2] == before[2] - 1, counts
    # -X face must remain (native passthrough, no expansion).
    assert has_face_at_centroid_x(bm, -1.0), "-X face missing after one-side delete"
    assert not has_face_at_centroid_x(bm, 1.0), "+X face still present after one-side delete"
    print(f"YSE_DELETE_ROUTES_G_COUNTS={counts}", flush=True)
    print("YSE_DELETE_ROUTES_G_OK", flush=True)


def run_all() -> None:
    window, area, region = STATE["window"], STATE["area"], STATE["region"]
    obj = STATE["object"]

    case_a_routes()
    baseline = case_b_face_delete(window, area, region, obj)
    case_e_undo(window, area, region, obj, baseline)
    # After undo the original Cube is back.
    obj = active_mesh_object()
    assert obj is not None
    case_c_hidden(window, area, region, obj)
    case_d_unmatched(window, area, region, obj)
    case_f_disable()
    case_g_passthrough(window, area, region)


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

        # Factory Cube keeps non-modal ed.undo memfiles coherent.
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
            bpy.context.tool_settings.mesh_select_mode = (False, False, True)
            bpy.ops.ed.undo_push(message="YSE Delete Routes baseline")

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
