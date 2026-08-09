# SPDX-License-Identifier: GPL-3.0-or-later

"""GUI contract tests for Select Mirrored on the real Loop Cut route.

Contract: .agents/doc/select_mirrored_contract_2026-08-09.md §4-1, §4-5

1. select_mirrored OFF: Loop Cut leaves only the native-side new loop selected
   (current behaviour; no mirror add-select).
2. select_mirrored ON: after a real Ctrl+R Loop Cut (operators finish hook),
   both the native new loop and its mirror counterparts are selected.
3. Undo: after ON Loop Cut, one ``bpy.ops.ed.undo()`` rewinds topology and
   the selection extension together to the pre-cut baseline.

Run with Blender's real event loop::

    cmd.exe /c "tests\\run_gui_test.bat 42 test_select_mirrored_gui.py"
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from pathlib import Path

import bmesh
import bpy
from bpy_extras import view3d_utils
from mathutils import Quaternion, Vector

bpy.context.preferences.view.show_splash = False
bpy.context.preferences.view.smooth_view = 0
bpy.context.preferences.view.use_save_prompt = False
bpy.context.preferences.filepaths.use_auto_save_temporary_files = False

PACKAGE_PARENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_PARENT))

import ydd_symmetric_edit as addon  # noqa: E402
from ydd_symmetric_edit import core, keymaps, operators  # noqa: E402

OBJECT_NAME = "YSE_SelectMirroredGuiObject"
MESH_NAME = "YSE_SelectMirroredGuiMesh"
# Two symmetric quads; two vertical loop cuts per side (native + mirror),
# matching tests/test_loopcut_route_history.py geometry and event sequence.
BASELINE_TOPOLOGY = (8, 8, 2)
FINISHED_TOPOLOGY = (16, 20, 6)
STATE = {
    "addon_registered": False,
    "events": [],
    "deadline": 0.0,
    "phase": "startup",
}


def modal_identifiers():
    identifiers = []
    try:
        for operator in STATE["window"].modal_operators:
            identifiers.extend(
                identifier
                for identifier in (
                    getattr(getattr(operator, "bl_rna", None), "identifier", ""),
                    getattr(operator, "bl_idname", ""),
                )
                if identifier
            )
    except Exception:
        pass
    return tuple(identifiers)


def current_object():
    return bpy.data.objects.get(OBJECT_NAME)


def current_bmesh():
    obj = current_object()
    if obj is None or obj.mode != "EDIT":
        return None
    return bmesh.from_edit_mesh(obj.data)


def topology():
    bm = current_bmesh()
    if bm is None:
        return None
    return len(bm.verts), len(bm.edges), len(bm.faces)


def temporary_layer_names():
    bm = current_bmesh()
    if bm is None:
        return ()
    layer_groups = (
        bm.verts.layers.int,
        bm.edges.layers.int,
        bm.faces.layers.int,
    )
    return tuple(name for name in core.TEMP_LAYER_NAMES if any(layers.get(name) is not None for layers in layer_groups))


def fail(message=""):
    if message:
        print(f"YSE_SELECT_MIRRORED_GUI_ERROR={message}", flush=True)
    traceback.print_exc()
    print(f"YSE_SELECT_MIRRORED_GUI_PHASE={STATE.get('phase')}", flush=True)
    print(f"YSE_SELECT_MIRRORED_GUI_TOPOLOGY={topology()}", flush=True)
    obj = current_object()
    print(
        "YSE_SELECT_MIRRORED_GUI_CONTEXT="
        f"{bpy.context.mode}, object={getattr(obj, 'name', None)}, "
        f"object_mode={getattr(obj, 'mode', None)}, "
        f"edit_object={getattr(bpy.context.edit_object, 'name', None)}",
        flush=True,
    )
    print(f"YSE_SELECT_MIRRORED_GUI_MODAL_IDS={modal_identifiers()}", flush=True)
    print(f"YSE_SELECT_MIRRORED_GUI_SESSIONS={list(operators._SESSIONS)}", flush=True)
    print("YSE_SELECT_MIRRORED_GUI_TEST_FAILED", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(1)


def viewport_context():
    window = bpy.context.window_manager.windows[0]
    area = next(area for area in window.screen.areas if area.type == "VIEW_3D")
    region = next(region for region in area.regions if region.type == "WINDOW")
    return window, area, region


def configure_view(area):
    region_3d = area.spaces.active.region_3d
    region_3d.view_perspective = "ORTHO"
    region_3d.view_rotation = Quaternion((1.0, 0.0, 0.0, 0.0))
    region_3d.view_location = (0.0, 0.0, 0.0)
    region_3d.view_distance = 5.0
    region_3d.update()


def make_mesh():
    for old in tuple(bpy.data.objects):
        bpy.data.objects.remove(old, do_unlink=True)

    mesh = bpy.data.meshes.new(MESH_NAME)
    mesh.from_pydata(
        [
            (-2.0, -1.0, 0.0),
            (-1.0, -1.0, 0.0),
            (-1.0, 1.0, 0.0),
            (-2.0, 1.0, 0.0),
            (1.0, -1.0, 0.0),
            (2.0, -1.0, 0.0),
            (2.0, 1.0, 0.0),
            (1.0, 1.0, 0.0),
        ],
        [],
        [(0, 1, 2, 3), (4, 5, 6, 7)],
    )
    mesh.update()

    obj = bpy.data.objects.new(OBJECT_NAME, mesh)
    bpy.context.scene.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    obj.use_mesh_mirror_x = True
    obj.use_mesh_mirror_y = False
    obj.use_mesh_mirror_z = False
    return obj


def window_coordinate(coordinate):
    local = view3d_utils.location_3d_to_region_2d(
        STATE["region"],
        STATE["area"].spaces.active.region_3d,
        Vector(coordinate),
    )
    if local is None:
        raise RuntimeError(f"Could not project test point {coordinate}")
    return (
        int(round(STATE["region"].x + local.x)),
        int(round(STATE["region"].y + local.y)),
    )


def coordinate_key(coordinate):
    return tuple(round(float(value), 5) for value in coordinate)


def vertex_key(vertex):
    return coordinate_key(vertex.co)


def edge_key(edge):
    return frozenset(vertex_key(vertex) for vertex in edge.verts)


def selection_signature(bm):
    return (
        frozenset(vertex_key(vertex) for vertex in bm.verts if vertex.select),
        frozenset(edge_key(edge) for edge in bm.edges if edge.select),
    )


def internal_vertical_cut_coordinates(bm):
    coordinates = []
    for edge in bm.edges:
        x0, x1 = (vertex.co.x for vertex in edge.verts)
        if abs(x0 - x1) > 1.0e-6:
            continue
        x = (x0 + x1) * 0.5
        if not 1.0 + 1.0e-6 < abs(x) < 2.0 - 1.0e-6:
            continue
        ys = sorted(vertex.co.y for vertex in edge.verts)
        if abs(ys[0] + 1.0) <= 1.0e-6 and abs(ys[1] - 1.0) <= 1.0e-6:
            coordinates.append(float(x))
    return sorted(coordinates)


def production_is_busy():
    identifiers = tuple(identifier.upper() for identifier in modal_identifiers())
    native_modal = any(
        token in identifier
        for identifier in identifiers
        for token in ("LOOPCUT_SLIDE", "MESH_OT_LOOPCUT", "EDGE_SLIDE")
    )
    return bool(
        native_modal or operators._SESSIONS or operators._HISTORY_REPAIR_QUEUED or operators._HISTORY_REPAIR_BUSY
    )


def ctrl_r_route_ready():
    route_keys = {
        route.route_key
        for route in keymaps._ROUTES_BY_KEY.values()
        if route.native_operator == "mesh.loopcut_slide"
        and route.keymap_name == "Mesh"
        and route.event.type == "R"
        and route.event.value == "PRESS"
        and route.event.ctrl
        and not route.event.shift
        and not route.event.alt
    }
    if not route_keys:
        return False
    return any(
        item.active
        and item.idname == keymaps.INTERCEPT_OPERATOR
        and getattr(item.properties, "route_key", "") in route_keys
        for _keymap, item in keymaps._REGISTERED_ITEMS
    )


def set_select_mirrored(enabled: bool) -> None:
    preferences = addon.ui.get_addon_preferences(bpy.context)
    if preferences is not None:
        preferences.enabled = True
    settings = bpy.context.scene.ydd_symmetric_edit
    settings.select_mirrored = enabled
    settings.tolerance = 1.0e-5


def assert_baseline_topology():
    assert topology() == BASELINE_TOPOLOGY, topology()
    bm = current_bmesh()
    assert bm is not None
    assert internal_vertical_cut_coordinates(bm) == []
    assert temporary_layer_names() == (), temporary_layer_names()
    assert not operators._SESSIONS, operators._SESSIONS
    return selection_signature(bm)


def assert_finished_topology(*, number_cuts=2):
    assert topology() == FINISHED_TOPOLOGY, topology()
    bm = current_bmesh()
    assert bm is not None
    cuts = internal_vertical_cut_coordinates(bm)
    assert len(cuts) == number_cuts * 2, cuts
    negative = cuts[:number_cuts]
    positive = cuts[number_cuts:]
    assert all(value < 0.0 for value in negative), cuts
    assert all(value > 0.0 for value in positive), cuts
    assert all(abs(left + right) <= 1.0e-7 for left, right in zip(negative, reversed(positive), strict=True)), cuts
    assert temporary_layer_names() == (), temporary_layer_names()
    assert not operators._SESSIONS, operators._SESSIONS
    return bm, cuts, selection_signature(bm)


def assert_native_only_selection(bm, signature):
    verts, edges = signature
    assert edges, "native loop edge must be selected"
    assert all(all(coord[0] < 0.0 for coord in edge) for edge in edges), edges
    assert any(key[0] < 0.0 for key in verts), verts
    assert not any(key[0] > 0.0 for key in verts), verts
    # Mirror-side internal cut edge must exist but stay unselected.
    mirror_cut_edges = [
        edge
        for edge in bm.edges
        if abs(edge.verts[0].co.x - edge.verts[1].co.x) <= 1.0e-6
        and edge.verts[0].co.x > 0.0
        and 1.0 + 1.0e-6 < abs(edge.verts[0].co.x) < 2.0 - 1.0e-6
    ]
    assert mirror_cut_edges, "mirror cut edge missing"
    assert all(not edge.select for edge in mirror_cut_edges)


def assert_both_sides_selected(bm, signature):
    verts, edges = signature
    assert any(key[0] < 0.0 for key in verts), verts
    assert any(key[0] > 0.0 for key in verts), verts
    neg_edges = [edge for edge in edges if all(coord[0] < 0.0 for coord in edge)]
    pos_edges = [edge for edge in edges if all(coord[0] > 0.0 for coord in edge)]
    assert neg_edges, edges
    assert pos_edges, edges
    # Selected cut edges should pair across X.
    for edge in neg_edges:
        mirrored = frozenset((-x, y, z) for x, y, z in edge)
        assert mirrored in edges, (edge, edges)


def setup_scene_and_edit():
    obj = make_mesh()
    with bpy.context.temp_override(
        window=STATE["window"],
        area=STATE["area"],
        region=STATE["region"],
    ):
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.context.tool_settings.mesh_select_mode = (False, True, False)
        bpy.context.tool_settings.workspace_tool_type = "DEFAULT"
        bpy.ops.mesh.select_all(action="DESELECT")
        result = bpy.ops.wm.tool_set_by_id(name="builtin.select_box")
        assert result == {"FINISHED"}, result
        bpy.ops.ed.undo_push(message="YSE Select Mirrored GUI baseline")
    assert core.enabled_mesh_symmetry_axes(obj) == (("X", 0),)
    return assert_baseline_topology()


def wait_for_route_off():
    try:
        if ctrl_r_route_ready():
            STATE["baseline_selection_off"] = setup_scene_and_edit()
            bpy.app.timers.register(begin_loopcut_off, first_interval=0.2)
            return None
        if time.monotonic() > STATE["deadline"]:
            raise RuntimeError("The native Ctrl+R Loop Cut intercept was not registered")
        return 0.05
    except BaseException:
        fail()
    return None


def loopcut_events():
    """Same Ctrl+R two-cut off-center slide sequence as test_loopcut_route_history."""

    hover = window_coordinate((-1.5, -1.0, 0.0))
    slide = window_coordinate((-1.68, 0.0, 0.0))
    return [
        {"type": "MOUSEMOVE", "value": "NOTHING", "x": hover[0], "y": hover[1]},
        {"type": "R", "value": "PRESS", "ctrl": True, "x": hover[0], "y": hover[1]},
        {"type": "R", "value": "RELEASE", "ctrl": True, "x": hover[0], "y": hover[1]},
        {"type": "WHEELUPMOUSE", "value": "PRESS", "x": hover[0], "y": hover[1]},
        {"type": "LEFTMOUSE", "value": "PRESS", "x": hover[0], "y": hover[1]},
        {"type": "LEFTMOUSE", "value": "RELEASE", "x": hover[0], "y": hover[1]},
        {"type": "MOUSEMOVE", "value": "NOTHING", "x": slide[0], "y": slide[1]},
        {"type": "LEFTMOUSE", "value": "PRESS", "x": slide[0], "y": slide[1]},
        {"type": "LEFTMOUSE", "value": "RELEASE", "x": slide[0], "y": slide[1]},
    ]


def begin_loopcut_off():
    try:
        STATE["phase"] = "Loop Cut OFF (native-only selection)"
        STATE["events"] = loopcut_events()
        bpy.app.timers.register(send_next_event_off, first_interval=0.05)
    except BaseException:
        fail()
    return None


def send_next_event_off():
    try:
        if STATE["events"]:
            STATE["window"].event_simulate(**STATE["events"].pop(0))
            return 0.15
        STATE["phase"] = "wait for Loop Cut OFF finish"
        STATE["deadline"] = time.monotonic() + 10.0
        bpy.app.timers.register(wait_for_loopcut_off, first_interval=0.05)
    except BaseException:
        fail()
    return None


def wait_for_loopcut_off():
    try:
        if production_is_busy() or topology() != FINISHED_TOPOLOGY:
            if time.monotonic() > STATE["deadline"]:
                raise RuntimeError("Timed out waiting for Loop Cut (select_mirrored OFF)")
            return 0.05

        bm, _cuts, signature = assert_finished_topology()
        assert_native_only_selection(bm, signature)
        print("YSE_SELECT_MIRRORED_GUI_STEP=loop_cut_off", flush=True)

        # Rebuild for the ON case so topology/selection start clean.
        STATE["phase"] = "setup select_mirrored ON"
        set_select_mirrored(True)
        STATE["baseline_selection_on"] = setup_scene_and_edit()
        bpy.app.timers.register(begin_loopcut_on, first_interval=0.2)
    except BaseException:
        fail()
    return None


def begin_loopcut_on():
    try:
        STATE["phase"] = "Loop Cut ON (both-side selection)"
        STATE["events"] = loopcut_events()
        bpy.app.timers.register(send_next_event_on, first_interval=0.05)
    except BaseException:
        fail()
    return None


def send_next_event_on():
    try:
        if STATE["events"]:
            STATE["window"].event_simulate(**STATE["events"].pop(0))
            return 0.15
        STATE["phase"] = "wait for Loop Cut ON finish"
        STATE["deadline"] = time.monotonic() + 10.0
        bpy.app.timers.register(wait_for_loopcut_on, first_interval=0.05)
    except BaseException:
        fail()
    return None


def wait_for_loopcut_on():
    try:
        if production_is_busy() or topology() != FINISHED_TOPOLOGY:
            if time.monotonic() > STATE["deadline"]:
                raise RuntimeError("Timed out waiting for Loop Cut (select_mirrored ON)")
            return 0.05

        bm, _cuts, signature = assert_finished_topology()
        assert_both_sides_selected(bm, signature)
        STATE["on_selection"] = signature
        print("YSE_SELECT_MIRRORED_GUI_STEP=loop_cut_on", flush=True)

        STATE["phase"] = "ed.undo after select_mirrored ON Loop Cut"
        with bpy.context.temp_override(
            window=STATE["window"],
            area=STATE["area"],
            region=STATE["region"],
        ):
            assert bpy.ops.ed.undo() == {"FINISHED"}
        STATE["deadline"] = time.monotonic() + 10.0
        bpy.app.timers.register(wait_for_undo, first_interval=0.05)
    except BaseException:
        fail()
    return None


def wait_for_undo():
    try:
        if production_is_busy() or topology() != BASELINE_TOPOLOGY:
            if time.monotonic() > STATE["deadline"]:
                raise RuntimeError("Timed out waiting for Loop Cut Undo")
            return 0.05

        signature = assert_baseline_topology()
        # Topology and selection extension rewind together to the pre-cut baseline.
        assert signature == STATE["baseline_selection_on"], (
            signature,
            STATE["baseline_selection_on"],
        )
        print("YSE_SELECT_MIRRORED_GUI_STEP=undo", flush=True)
        print("YSE_SELECT_MIRRORED_GUI_TEST_OK", flush=True)
        addon.unregister()
        STATE["addon_registered"] = False
        bpy.ops.wm.quit_blender()
    except BaseException:
        fail()
    return None


def start_test():
    try:
        STATE["phase"] = "setup"
        addon.register()
        STATE["addon_registered"] = True
        addon.sync_persistent_keymap(True)

        window, area, region = viewport_context()
        STATE.update(window=window, area=area, region=region)
        configure_view(area)
        set_select_mirrored(False)

        STATE["deadline"] = time.monotonic() + 5.0
        bpy.app.timers.register(wait_for_route_off, first_interval=0.05)
    except BaseException:
        fail()
    return None


bpy.app.timers.register(start_test, first_interval=0.25)
