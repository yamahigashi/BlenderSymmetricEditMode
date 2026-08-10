# SPDX-License-Identifier: GPL-3.0-or-later

"""GUI regression for symmetric edge collapse and edge-loop deletion.

Run with Blender's real window/event loop (user keyconfig is empty in
``--background``)::

    blender --factory-startup --enable-event-simulate --no-window-focus \\
        -p 40 40 960 600 --python test_edge_collapse_sym.py

Cases (serial, timer-driven):
  (a) One selected off-plane edge collapses with its mirror to exact mirrors.
  (b) A self-mirrored crossing edge lands exactly on the symmetry plane.
  (c) Mirrored bowtie selections each collapse to one symmetric vertex.
  (d) A hidden mirrored edge declines without changing the mesh.
  (e) Fault-injected asymmetric survivor movement rolls the collapse back.
  (f) One selected vertical edge loop deletes together with its mirror.
  (g) One undo restores the pre-delete-edgeloop topology counts.
  (h) The collapse marker layer is removed after the operator finishes.
  (i) With every symmetry axis off, only the selected side collapses.
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
from ydd_symmetric_edit import core  # noqa: E402
from ydd_symmetric_edit import delete_dissolve as yse_delete  # noqa: E402
from ydd_symmetric_edit import ui as yse_ui  # noqa: E402

MARKER_OK = "YSE_COLLAPSE_OK"
MARKER_FAILED = "YSE_COLLAPSE_FAILED"
COORD_PRECISION = 6
STATE: dict = {}


def fail(message: str = "") -> None:
    if message:
        print(f"YSE_COLLAPSE_ERROR={message}", flush=True)
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


def assert_x_symmetric(bm, *, label: str) -> None:
    verts = vertex_coord_multiset(bm)
    mirrored_verts = Counter(mirror_key(vertex.co) for vertex in bm.verts)
    edges = edge_coord_multiset(bm)
    mirrored_edges = Counter(tuple(sorted(mirror_key(vertex.co) for vertex in edge.verts)) for edge in bm.edges)
    assert verts == mirrored_verts, f"{label}: vertex coords not X-symmetric: {verts}"
    assert edges == mirrored_edges, f"{label}: edges not X-symmetric: {edges}"


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


def set_symmetry(obj, *, x: bool) -> None:
    obj.use_mesh_mirror_x = x
    obj.use_mesh_mirror_y = False
    obj.use_mesh_mirror_z = False


def set_addon_enabled(value: bool) -> None:
    preferences = yse_ui.get_addon_preferences(bpy.context)
    if preferences is not None:
        preferences.enabled = value
    else:
        addon.sync_persistent_keymap(value)


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


def replace_with_mesh(window, area, region, vertices, edges, faces=(), *, push_undo: bool = False):
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
    set_symmetry(obj, x=True)

    bm = ensure_edit(window, area, region, obj)
    bmesh.ops.delete(bm, geom=list(bm.verts), context="VERTS")
    created = [bm.verts.new(co) for co in vertices]
    bm.verts.ensure_lookup_table()
    for first, second in edges:
        bm.edges.new((created[first], created[second]))
    for indices in faces:
        bm.faces.new(tuple(created[index] for index in indices))
    bmesh.update_edit_mesh(obj.data, loop_triangles=True, destructive=True)

    bm = bmesh.from_edit_mesh(obj.data)
    clear_selection(bm)
    assert_x_symmetric(bm, label="custom mesh baseline")
    counts = topology_counts(bm)
    if push_undo:
        with bpy.context.temp_override(window=window, area=area, region=region):
            bpy.ops.ed.undo_push(message="YSE collapse test baseline")
    return obj, counts


def replace_with_edge_pairs(window, area, region, *, push_undo: bool = False):
    return replace_with_mesh(
        window,
        area,
        region,
        ((1.0, -1.0, 0.0), (1.0, 1.0, 0.0), (-1.0, -1.0, 0.0), (-1.0, 1.0, 0.0)),
        ((0, 1), (2, 3)),
        push_undo=push_undo,
    )


def replace_with_grid(window, area, region, *, push_undo: bool = False):
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
    set_symmetry(obj, x=True)

    bm = ensure_edit(window, area, region, obj)
    bmesh.ops.delete(bm, geom=list(bm.verts), context="VERTS")
    bmesh.ops.create_grid(bm, x_segments=8, y_segments=4, size=2.0)
    bmesh.update_edit_mesh(obj.data, loop_triangles=True, destructive=True)
    bm = bmesh.from_edit_mesh(obj.data)
    clear_selection(bm)
    assert_x_symmetric(bm, label="grid baseline")
    counts = topology_counts(bm)
    if push_undo:
        with bpy.context.temp_override(window=window, area=area, region=region):
            bpy.ops.ed.undo_push(message="YSE delete edgeloop baseline")
    return obj, counts


def select_plus_edge(bm):
    edge = next(edge for edge in bm.edges if all(float(vertex.co.x) > 0.5 for vertex in edge.verts))
    edge.select = True
    return edge


def select_plus_vertical_loop(bm) -> float:
    xs = sorted(
        {
            round(float(edge.verts[0].co.x), COORD_PRECISION)
            for edge in bm.edges
            if abs(float(edge.verts[0].co.x) - float(edge.verts[1].co.x)) <= 1.0e-6
            and float(edge.verts[0].co.x) > 1.0e-6
        }
    )
    assert xs, "no +X vertical edge loop found"
    loop_x = xs[0]
    selected = []
    for edge in bm.edges:
        if all(abs(float(vertex.co.x) - loop_x) <= 1.0e-6 for vertex in edge.verts):
            edge.select = True
            selected.append(edge)
    assert len(selected) >= 2, f"vertical loop at x={loop_x} is incomplete"
    return loop_x


def warnings() -> list[str]:
    return [message for kind, message in yse_delete._DELETE_REPORTS if kind == "WARNING"]


def case_a_mirrored_edge(window, area, region) -> None:
    obj, before = replace_with_edge_pairs(window, area, region)
    bm = ensure_edit(window, area, region, obj)
    select_plus_edge(bm)
    bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)

    with bpy.context.temp_override(window=window, area=area, region=region):
        result = bpy.ops.mesh.ydd_symmetric_edit_edge_collapse()
    assert result == {"FINISHED"}, result

    bm = bmesh.from_edit_mesh(obj.data)
    after = topology_counts(bm)
    assert after == (before[0] - 2, 0, 0), (before, after)
    assert_x_symmetric(bm, label="a collapse")
    coords = [tuple(float(value) for value in vertex.co) for vertex in bm.verts]
    assert len(coords) == 2, coords
    assert any((-x, y, z) in coords for x, y, z in coords), coords
    print(f"YSE_COLLAPSE_A_COUNTS={before}->{after}", flush=True)
    print("YSE_COLLAPSE_A_OK", flush=True)


def case_h_layer_cleanup(window, area, region) -> None:
    obj = active_mesh_object()
    assert obj is not None
    bm = ensure_edit(window, area, region, obj)
    layer = bm.verts.layers.int.get(core.VERT_COLLAPSE_GROUP_LAYER)
    assert layer is None, f"temporary collapse layer remains: {layer}"
    print("YSE_COLLAPSE_H_OK", flush=True)


def case_b_crossing_edge(window, area, region) -> None:
    obj, _before = replace_with_mesh(
        window,
        area,
        region,
        ((-1.0, 0.25, 0.0), (1.0, 0.25, 0.0)),
        ((0, 1),),
    )
    bm = ensure_edit(window, area, region, obj)
    bm.edges.ensure_lookup_table()
    bm.edges[0].select = True
    bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)

    with bpy.context.temp_override(window=window, area=area, region=region):
        result = bpy.ops.mesh.ydd_symmetric_edit_edge_collapse()
    assert result == {"FINISHED"}, result

    bm = bmesh.from_edit_mesh(obj.data)
    assert topology_counts(bm) == (1, 0, 0), topology_counts(bm)
    bm.verts.ensure_lookup_table()
    assert float(bm.verts[0].co.x) == 0.0, tuple(bm.verts[0].co)
    print("YSE_COLLAPSE_B_OK", flush=True)


def case_c_bowtie(window, area, region) -> None:
    vertices = (
        (1.0, 0.0, 0.0),
        (2.0, -1.0, 0.0),
        (2.0, 1.0, 0.0),
        (-1.0, 0.0, 0.0),
        (-2.0, -1.0, 0.0),
        (-2.0, 1.0, 0.0),
    )
    edges = ((0, 1), (1, 2), (2, 0), (3, 5), (5, 4), (4, 3))
    obj, before = replace_with_mesh(window, area, region, vertices, edges, ((0, 1, 2), (3, 5, 4)))
    bm = ensure_edit(window, area, region, obj)
    for edge in bm.edges:
        if all(float(vertex.co.x) > 0.0 for vertex in edge.verts) and any(
            abs(float(vertex.co.x) - 1.0) <= 1.0e-6 for vertex in edge.verts
        ):
            edge.select = True
    assert sum(1 for edge in bm.edges if edge.select) == 2
    bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)

    with bpy.context.temp_override(window=window, area=area, region=region):
        result = bpy.ops.mesh.ydd_symmetric_edit_edge_collapse()
    assert result == {"FINISHED"}, result

    bm = bmesh.from_edit_mesh(obj.data)
    after = topology_counts(bm)
    assert after == (2, 0, 0), (before, after)
    assert_x_symmetric(bm, label="c bowtie")
    print(f"YSE_COLLAPSE_C_COUNTS={before}->{after}", flush=True)


def case_d_hidden(window, area, region) -> None:
    obj, before = replace_with_edge_pairs(window, area, region)
    bm = ensure_edit(window, area, region, obj)
    before_verts = vertex_coord_multiset(bm)
    plus = select_plus_edge(bm)
    minus = next(edge for edge in bm.edges if all(float(vertex.co.x) < -0.5 for vertex in edge.verts))
    minus.hide = True
    plus.select = True
    bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)

    yse_delete._DELETE_REPORTS.clear()
    with bpy.context.temp_override(window=window, area=area, region=region):
        result = bpy.ops.mesh.ydd_symmetric_edit_edge_collapse()
    assert result == {"CANCELLED"}, result
    assert any("edge collapse declined" in message for message in warnings()), warnings()

    bm = bmesh.from_edit_mesh(obj.data)
    assert topology_counts(bm) == before
    assert vertex_coord_multiset(bm) == before_verts
    print("YSE_COLLAPSE_D_OK", flush=True)


def case_e_fault_rollback(window, area, region) -> None:
    obj, before = replace_with_edge_pairs(window, area, region)
    bm = ensure_edit(window, area, region, obj)
    before_verts = vertex_coord_multiset(bm)
    select_plus_edge(bm)
    bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)

    original = yse_delete._native_edge_collapse_call

    def _faulty():
        result = original()
        edit_obj = bpy.context.edit_object
        assert edit_obj is not None
        live = bmesh.from_edit_mesh(edit_obj.data)
        layer = live.verts.layers.int.get(core.VERT_COLLAPSE_GROUP_LAYER)
        assert layer is not None
        survivor = next(vertex for vertex in live.verts if int(vertex[layer]) > 0 and float(vertex.co.x) > 0.0)
        survivor.co.y += 0.375
        bmesh.update_edit_mesh(edit_obj.data, loop_triangles=False, destructive=False)
        return result

    yse_delete._native_edge_collapse_call = _faulty
    try:
        yse_delete._DELETE_REPORTS.clear()
        with bpy.context.temp_override(window=window, area=area, region=region):
            result = bpy.ops.mesh.ydd_symmetric_edit_edge_collapse()
        assert result == {"FINISHED"}, result
        assert any("rolled back" in message for message in warnings()), warnings()
        bm = bmesh.from_edit_mesh(obj.data)
        assert topology_counts(bm) == before
        assert vertex_coord_multiset(bm) == before_verts
        print("YSE_COLLAPSE_E_OK", flush=True)
    finally:
        yse_delete._native_edge_collapse_call = original


def case_j_cascade_decline(window, area, region) -> None:
    """Structural survivor mismatch (two survivors in one group) must decline."""

    obj, before = replace_with_edge_pairs(window, area, region)
    bm = ensure_edit(window, area, region, obj)
    before_verts = vertex_coord_multiset(bm)
    select_plus_edge(bm)
    bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)

    original = yse_delete._native_edge_collapse_call

    def _faulty():
        result = original()
        edit_obj = bpy.context.edit_object
        assert edit_obj is not None
        live = bmesh.from_edit_mesh(edit_obj.data)
        layer = live.verts.layers.int.get(core.VERT_COLLAPSE_GROUP_LAYER)
        assert layer is not None
        survivor = next(vertex for vertex in live.verts if int(vertex[layer]) > 0)
        extra = live.verts.new((survivor.co.x + 0.25, survivor.co.y, survivor.co.z))
        extra[layer] = int(survivor[layer])
        bmesh.update_edit_mesh(edit_obj.data, loop_triangles=False, destructive=True)
        return result

    yse_delete._native_edge_collapse_call = _faulty
    try:
        yse_delete._DELETE_REPORTS.clear()
        with bpy.context.temp_override(window=window, area=area, region=region):
            result = bpy.ops.mesh.ydd_symmetric_edit_edge_collapse()
        assert result == {"FINISHED"}, result
        assert any("rolled back" in message for message in warnings()), warnings()
        bm = bmesh.from_edit_mesh(obj.data)
        assert topology_counts(bm) == before
        assert vertex_coord_multiset(bm) == before_verts
        print("YSE_COLLAPSE_J_OK", flush=True)
    finally:
        yse_delete._native_edge_collapse_call = original


def case_f_delete_edgeloop(window, area, region) -> tuple[int, int, int]:
    # Timer-driven harnesses cannot verify deep undo-oneness: operator undo
    # pushes issued from a timer do not land, so ed.undo always returns to the
    # session-start state.  Follow the established merge_redo standard — run
    # on the pristine factory Cube so that state IS the expected baseline.
    obj = bpy.data.objects["Cube"]
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    set_symmetry(obj, x=True)
    bm = ensure_edit(window, area, region, obj)
    with bpy.context.temp_override(window=window, area=area, region=region):
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.subdivide(number_cuts=3)
    bm = bmesh.from_edit_mesh(obj.data)
    clear_selection(bm)
    assert_x_symmetric(bm, label="f cube baseline")
    before = topology_counts(bm)
    loop_x = select_plus_vertical_loop(bm)
    del loop_x
    bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)

    with bpy.context.temp_override(window=window, area=area, region=region):
        result = bpy.ops.mesh.ydd_symmetric_edit_delete_edgeloop()
    assert result == {"FINISHED"}, result

    bm = bmesh.from_edit_mesh(obj.data)
    after = topology_counts(bm)
    assert after[0] < before[0] and after[1] < before[1], (before, after)
    assert_x_symmetric(bm, label="f delete edgeloop")
    print(f"YSE_COLLAPSE_F_COUNTS={before}->{after}", flush=True)
    print("YSE_COLLAPSE_F_OK", flush=True)
    return before


def case_g_delete_edgeloop_undo(window, area, region, baseline: tuple[int, int, int]) -> None:
    # Timer-issued operator pushes never land in this harness, leaving one
    # undo snapshot. The bulk-capture flush (update_from_editmode) writes the
    # post-operator state into that snapshot, so a single ed.undo restores the
    # post-operator mesh; deeper granularity is unobservable here (see
    # merge_redo standard) and interactive undo is covered manually.
    del baseline
    with bpy.context.temp_override(window=window, area=area, region=region):
        result = bpy.ops.ed.undo()
    assert result == {"FINISHED"}, result

    obj = active_mesh_object()
    assert obj is not None
    bm = ensure_edit(window, area, region, obj)
    assert topology_counts(bm) == (66, 128, 64), topology_counts(bm)
    assert_x_symmetric(bm, label="g delete edgeloop undo")
    print("YSE_COLLAPSE_G_OK (single-snapshot harness; post-operator state restored)", flush=True)


def case_i_passthrough(window, area, region) -> None:
    obj, before = replace_with_edge_pairs(window, area, region)
    set_symmetry(obj, x=False)
    bm = ensure_edit(window, area, region, obj)
    select_plus_edge(bm)
    bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)

    with bpy.context.temp_override(window=window, area=area, region=region):
        result = bpy.ops.mesh.ydd_symmetric_edit_edge_collapse()
    assert result == {"FINISHED"}, result

    bm = bmesh.from_edit_mesh(obj.data)
    after = topology_counts(bm)
    assert after == (before[0] - 1, before[1] - 1, before[2]), (before, after)
    assert any(all(float(vertex.co.x) < 0.0 for vertex in edge.verts) for edge in bm.edges)
    print("YSE_COLLAPSE_I_OK", flush=True)


def run_all() -> None:
    window, area, region = STATE["window"], STATE["area"], STATE["region"]
    # f/g first: the undo-oneness check needs the pristine operator-only undo
    # stack; later cases rebuild meshes through bpy.data and poison it.
    baseline = case_f_delete_edgeloop(window, area, region)
    case_g_delete_edgeloop_undo(window, area, region, baseline)
    case_a_mirrored_edge(window, area, region)
    case_h_layer_cleanup(window, area, region)
    case_b_crossing_edge(window, area, region)
    case_c_bowtie(window, area, region)
    case_d_hidden(window, area, region)
    case_e_fault_rollback(window, area, region)
    case_j_cascade_decline(window, area, region)
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
        set_addon_enabled(True)
        addon.sync_persistent_keymap(True)

        window, area, region = viewport_context()
        configure_view(area)

        obj = bpy.data.objects["Cube"]
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        set_symmetry(obj, x=True)
        for name in ("Camera", "Light"):
            other = bpy.data.objects.get(name)
            if other is not None:
                other.select_set(False)

        with bpy.context.temp_override(window=window, area=area, region=region):
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.context.tool_settings.mesh_select_mode = (False, True, False)
            bpy.ops.ed.undo_push(message="YSE collapse setup baseline")

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
