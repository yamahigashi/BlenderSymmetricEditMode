# SPDX-License-Identifier: GPL-3.0-or-later

"""GUI regression for the native Ctrl+R Loop Cut pass-through route.

Run with Blender's real event loop::

    blender --factory-startup --enable-event-simulate \
        --python test_loopcut_route_history.py

The physical Ctrl+R event must reach Blender's own ``mesh.loopcut_slide``.
The test makes two cuts, applies a visible off-center Edge Slide, and checks
that the mirrored post-process remains one native Undo/Redo history step.
"""

from __future__ import annotations

import math
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
from ydd_symmetric_edit import keymaps, layer_names, matching, operators, snapshot  # noqa: E402

OBJECT_NAME = "YSE_LoopCutRouteHistoryObject"
MESH_NAME = "YSE_LoopCutRouteHistoryMesh"
BASELINE_TOPOLOGY = (8, 8, 2)
FINISHED_TOPOLOGY = (16, 20, 6)
ADJUSTED_TOPOLOGY = (20, 26, 8)
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
    return tuple(name for name in layer_names.TEMP_LAYER_NAMES if any(layers.get(name) is not None for layers in layer_groups))


def fail(message=""):
    if message:
        print(f"YSE_LOOPCUT_ROUTE_HISTORY_ERROR={message}", flush=True)
    traceback.print_exc()
    print(f"YSE_LOOPCUT_ROUTE_HISTORY_PHASE={STATE.get('phase')}", flush=True)
    print(f"YSE_LOOPCUT_ROUTE_HISTORY_TOPOLOGY={topology()}", flush=True)
    obj = current_object()
    print(
        "YSE_LOOPCUT_ROUTE_HISTORY_CONTEXT="
        f"{bpy.context.mode}, object={getattr(obj, 'name', None)}, "
        f"object_mode={getattr(obj, 'mode', None)}, "
        f"edit_object={getattr(bpy.context.edit_object, 'name', None)}",
        flush=True,
    )
    print(
        f"YSE_LOOPCUT_ROUTE_HISTORY_TEMP_LAYERS={temporary_layer_names()}",
        flush=True,
    )
    print(
        f"YSE_LOOPCUT_ROUTE_HISTORY_MODAL_IDS={modal_identifiers()}",
        flush=True,
    )
    print(
        f"YSE_LOOPCUT_ROUTE_HISTORY_SESSIONS={list(operators._SESSIONS)}",
        flush=True,
    )
    records = [(token, record.session.tool_kind, record.status) for token, record in operators._HISTORY_RECORDS.items()]
    print(f"YSE_LOOPCUT_ROUTE_HISTORY_RECORDS={records}", flush=True)
    print("YSE_LOOPCUT_ROUTE_HISTORY_TEST_FAILED", flush=True)
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
    uv_layer = mesh.uv_layers.new(name="UVMap")
    for polygon in mesh.polygons:
        for loop_index in polygon.loop_indices:
            vertex = mesh.vertices[mesh.loops[loop_index].vertex_index]
            uv_layer.data[loop_index].uv = (
                abs(vertex.co.x) - 1.0,
                (vertex.co.y + 1.0) * 0.5,
            )

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
    return tuple(round(float(value), 6) for value in coordinate)


def vertex_key(vertex):
    return coordinate_key(vertex.co)


def edge_key(edge):
    return tuple(sorted(vertex_key(vertex) for vertex in edge.verts))


def face_key(face):
    return tuple(sorted(vertex_key(vertex) for vertex in face.verts))


def element_key(element):
    if isinstance(element, bmesh.types.BMVert):
        return "VERT", vertex_key(element)
    if isinstance(element, bmesh.types.BMEdge):
        return "EDGE", edge_key(element)
    return "FACE", face_key(element)


def selection_signature(bm):
    return (
        tuple(sorted(vertex_key(vertex) for vertex in bm.verts if vertex.select)),
        tuple(sorted(edge_key(edge) for edge in bm.edges if edge.select)),
        tuple(sorted(face_key(face) for face in bm.faces if face.select)),
        tuple(element_key(element) for element in bm.select_history),
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


def active_tool_id():
    tool = bpy.context.workspace.tools.from_space_view3d_mode(
        bpy.context.mode,
        create=False,
    )
    return tool.idname if tool is not None else None


def active_operator_id():
    operator = bpy.context.active_operator
    if operator is None:
        return None
    return getattr(operator, "bl_idname", "") or getattr(getattr(operator, "bl_rna", None), "identifier", "")


def active_loopcut_macro_children():
    operator = bpy.context.active_operator
    assert operator is not None
    assert active_operator_id() == "MESH_OT_loopcut_slide", active_operator_id()
    children = {child.bl_idname: child.properties for child in operator.macros}
    assert "MESH_OT_loopcut" in children, tuple(children)
    assert "TRANSFORM_OT_edge_slide" in children, tuple(children)
    return children


def assert_plain_undo_snapshot_is_not_treated_as_f9(committed):
    obj = current_object()
    assert obj is not None
    latest_record = committed[-1]
    snapshot_token = committed[-2].session.history_token
    assert snapshot_token != latest_record.session.history_token

    bm = bmesh.from_edit_mesh(obj.data)
    token_layer = bm.faces.layers.int.new(layer_names.HISTORY_TOKEN_LAYER)
    for face in bm.faces:
        face[token_layer] = snapshot_token
    bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)

    try:
        assert operators._object_history_tokens(obj) == {snapshot_token}
        assert not operators._prepare_adjust_last_operation_repeat()
        assert not operators._SESSIONS, operators._SESSIONS
        assert operators._object_history_tokens(obj) == {snapshot_token}
    finally:
        operators.cleanup_session(STATE["window"].as_pointer())
        bm = bmesh.from_edit_mesh(obj.data)
        snapshot.remove_temporary_layers(bm)
        bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)


def assert_uvs_finite(bm):
    uv_layer = bm.loops.layers.uv.active
    assert uv_layer is not None
    values = [component for face in bm.faces for loop in face.loops for component in loop[uv_layer].uv]
    assert values
    assert all(math.isfinite(float(value)) for value in values), values


def assert_baseline(expected_selection=None):
    obj = current_object()
    assert obj is not None
    assert topology() == BASELINE_TOPOLOGY, topology()
    assert bpy.context.mode == "EDIT_MESH", bpy.context.mode
    assert obj.mode == "EDIT"
    assert active_tool_id() == "builtin.select_box", active_tool_id()
    assert bpy.context.view_layer.objects.active is obj
    assert tuple(sorted(item.name for item in bpy.context.selected_objects)) == (OBJECT_NAME,)
    assert tuple(bpy.context.tool_settings.mesh_select_mode) == (False, True, False)
    assert (obj.use_mesh_mirror_x, obj.use_mesh_mirror_y, obj.use_mesh_mirror_z) == (
        True,
        False,
        False,
    )
    bm = bmesh.from_edit_mesh(obj.data)
    assert internal_vertical_cut_coordinates(bm) == []
    signature = selection_signature(bm)
    if expected_selection is not None:
        assert signature == expected_selection, (signature, expected_selection)
    assert_uvs_finite(bm)
    assert temporary_layer_names() == (), temporary_layer_names()
    assert not operators._SESSIONS, operators._SESSIONS
    return signature


def assert_finished(
    expected_selection=None,
    *,
    number_cuts=2,
    expected_topology=FINISHED_TOPOLOGY,
    check_active_operator=False,
):
    obj = current_object()
    assert obj is not None
    assert topology() == expected_topology, topology()
    assert bpy.context.mode == "EDIT_MESH", bpy.context.mode
    assert obj.mode == "EDIT"
    assert active_tool_id() == "builtin.select_box", active_tool_id()
    assert bpy.context.view_layer.objects.active is obj
    assert tuple(sorted(item.name for item in bpy.context.selected_objects)) == (OBJECT_NAME,)
    assert tuple(bpy.context.tool_settings.mesh_select_mode) == (False, True, False)
    assert (obj.use_mesh_mirror_x, obj.use_mesh_mirror_y, obj.use_mesh_mirror_z) == (
        True,
        False,
        False,
    )

    bm = bmesh.from_edit_mesh(obj.data)
    cuts = internal_vertical_cut_coordinates(bm)
    assert len(cuts) == number_cuts * 2, cuts
    negative = cuts[:number_cuts]
    positive = cuts[number_cuts:]
    assert all(value < 0.0 for value in negative), cuts
    assert all(value > 0.0 for value in positive), cuts
    assert all(abs(left + right) <= 1.0e-7 for left, right in zip(negative, reversed(positive), strict=True)), cuts
    source_center = sum(abs(value) for value in negative) * 0.5
    assert abs(source_center - 1.5) > 0.02, cuts

    signature = selection_signature(bm)
    assert tuple(len(elements) for elements in signature[:3]) == (
        number_cuts * 2,
        number_cuts,
        0,
    ), signature
    assert all(coordinate[0] < 0.0 for coordinate in signature[0]), signature
    assert all(all(coordinate[0] < 0.0 for coordinate in edge) for edge in signature[1]), signature
    if expected_selection is not None:
        assert signature == expected_selection, (signature, expected_selection)

    assert_uvs_finite(bm)
    assert temporary_layer_names() == (), temporary_layer_names()
    assert not operators._SESSIONS, operators._SESSIONS
    if check_active_operator:
        assert active_operator_id() == "MESH_OT_loopcut_slide", active_operator_id()
    return signature


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


def wait_for_route():
    try:
        if ctrl_r_route_ready():
            STATE["baseline_selection"] = assert_baseline()
            bpy.app.timers.register(begin_loopcut, first_interval=0.2)
            return None
        if time.monotonic() > STATE["deadline"]:
            raise RuntimeError("The native Ctrl+R Loop Cut intercept was not registered")
        return 0.05
    except BaseException:
        fail()
    return None


def begin_loopcut():
    try:
        STATE["phase"] = "native Ctrl+R two-cut off-center slide"
        hover = window_coordinate((-1.5, -1.0, 0.0))
        slide = window_coordinate((-1.68, 0.0, 0.0))
        STATE["events"] = [
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
        bpy.app.timers.register(send_next_event, first_interval=0.05)
    except BaseException:
        fail()
    return None


def send_next_event():
    try:
        if STATE["events"]:
            STATE["window"].event_simulate(**STATE["events"].pop(0))
            return 0.15
        STATE["phase"] = "wait for symmetric Loop Cut"
        STATE["deadline"] = time.monotonic() + 10.0
        bpy.app.timers.register(wait_for_loopcut, first_interval=0.05)
    except BaseException:
        fail()
    return None


def wait_for_loopcut():
    try:
        if production_is_busy() or topology() != FINISHED_TOPOLOGY:
            if time.monotonic() > STATE["deadline"]:
                raise RuntimeError("Timed out waiting for symmetric Loop Cut")
            return 0.05

        selection = assert_finished(check_active_operator=True)
        committed = [record for record in operators._HISTORY_RECORDS.values() if record.status == "COMMITTED"]
        assert all(record.session.tool_kind == "LOOP_CUT" for record in committed)

        if not STATE.get("history_verified"):
            STATE["initial_selection"] = selection
            assert len(committed) == 1, len(committed)
            STATE["phase"] = "undo initial Loop Cut"
            with bpy.context.temp_override(
                window=STATE["window"],
                area=STATE["area"],
                region=STATE["region"],
            ):
                assert bpy.ops.ed.undo() == {"FINISHED"}
            STATE["deadline"] = time.monotonic() + 10.0
            bpy.app.timers.register(wait_for_undo, first_interval=0.05)
            return None

        STATE["pre_adjust_selection"] = selection
        assert len(committed) == 2, len(committed)
        assert_plain_undo_snapshot_is_not_treated_as_f9(committed)
        STATE["phase"] = "F9 Adjust Last Operation repeat"
        children = active_loopcut_macro_children()
        assert children["MESH_OT_loopcut"].number_cuts == 2
        children["MESH_OT_loopcut"].number_cuts = 3
        children["TRANSFORM_OT_edge_slide"].value = 0.2
        with bpy.context.temp_override(
            window=STATE["window"],
            area=STATE["area"],
            region=STATE["region"],
        ):
            assert bpy.ops.ed.undo_redo() == {"FINISHED"}
        STATE["deadline"] = time.monotonic() + 10.0
        bpy.app.timers.register(wait_for_adjusted_loopcut, first_interval=0.05)
    except BaseException:
        fail()
    return None


def wait_for_adjusted_loopcut():
    try:
        if production_is_busy() or topology() != ADJUSTED_TOPOLOGY:
            if time.monotonic() > STATE["deadline"]:
                raise RuntimeError("Timed out waiting for adjusted symmetric Loop Cut")
            return 0.05

        STATE["adjusted_selection"] = assert_finished(
            number_cuts=3,
            expected_topology=ADJUSTED_TOPOLOGY,
            check_active_operator=True,
        )
        children = active_loopcut_macro_children()
        assert children["MESH_OT_loopcut"].number_cuts == 3
        assert abs(children["TRANSFORM_OT_edge_slide"].value - 0.2) <= 1.0e-6
        committed = [record for record in operators._HISTORY_RECORDS.values() if record.status == "COMMITTED"]
        assert len(committed) == 3, len(committed)
        assert all(record.session.tool_kind == "LOOP_CUT" for record in committed)
        active = bpy.context.active_operator
        assert active is not None
        assert committed[-1].session.native_operator_pointer == int(active.as_pointer())
        assert ctrl_r_route_ready()
        print("YSE_LOOPCUT_ROUTE_HISTORY_TEST_OK", flush=True)
        addon.unregister()
        STATE["addon_registered"] = False
        bpy.ops.wm.quit_blender()
    except BaseException:
        fail()
    return None


def wait_for_undo():
    try:
        if production_is_busy() or topology() != BASELINE_TOPOLOGY:
            if time.monotonic() > STATE["deadline"]:
                raise RuntimeError("Timed out waiting for Loop Cut Undo")
            return 0.05

        assert_baseline(expected_selection=STATE["baseline_selection"])
        STATE["phase"] = "redo"
        with bpy.context.temp_override(
            window=STATE["window"],
            area=STATE["area"],
            region=STATE["region"],
        ):
            assert bpy.ops.ed.redo() == {"FINISHED"}
        STATE["deadline"] = time.monotonic() + 10.0
        bpy.app.timers.register(wait_for_redo, first_interval=0.05)
    except BaseException:
        fail()
    return None


def wait_for_redo():
    try:
        if production_is_busy() or topology() != FINISHED_TOPOLOGY:
            if time.monotonic() > STATE["deadline"]:
                raise RuntimeError("Timed out waiting for Loop Cut Redo")
            return 0.05

        assert_finished(expected_selection=STATE["initial_selection"])
        STATE["phase"] = "return to baseline before F9 test"
        with bpy.context.temp_override(
            window=STATE["window"],
            area=STATE["area"],
            region=STATE["region"],
        ):
            assert bpy.ops.ed.undo() == {"FINISHED"}
        STATE["deadline"] = time.monotonic() + 10.0
        bpy.app.timers.register(wait_for_second_baseline, first_interval=0.05)
    except BaseException:
        fail()
    return None


def wait_for_second_baseline():
    try:
        if production_is_busy() or topology() != BASELINE_TOPOLOGY:
            if time.monotonic() > STATE["deadline"]:
                raise RuntimeError("Timed out returning to baseline for F9 test")
            return 0.05

        assert_baseline(expected_selection=STATE["baseline_selection"])
        STATE["history_verified"] = True
        bpy.app.timers.register(begin_loopcut, first_interval=0.2)
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
        obj = make_mesh()
        with bpy.context.temp_override(window=window, area=area, region=region):
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.context.tool_settings.mesh_select_mode = (False, True, False)
            bpy.context.tool_settings.workspace_tool_type = "DEFAULT"
            bpy.ops.mesh.select_all(action="DESELECT")
            result = bpy.ops.wm.tool_set_by_id(name="builtin.select_box")
            assert result == {"FINISHED"}, result
            bpy.ops.ed.undo_push(message="YSE Loop Cut Route History baseline")

        assert matching.enabled_mesh_symmetry_axes(obj) == (("X", 0),)
        STATE["deadline"] = time.monotonic() + 5.0
        bpy.app.timers.register(wait_for_route, first_interval=0.05)
    except BaseException:
        fail()
    return None


bpy.app.timers.register(start_test, first_interval=0.25)
