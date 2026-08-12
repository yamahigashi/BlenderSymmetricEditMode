# SPDX-License-Identifier: GPL-3.0-or-later

"""GUI regression test for the native Knife Tool route and history repair.

Run with Blender's real window/event loop::

    blender --factory-startup --enable-event-simulate \
        --python test_tool_route_history.py

The test intentionally drives the Workspace Knife Tool's LEFTMOUSE route.  It
must pass through to Blender's own ``mesh.knife_tool`` operator; calling either
the native operator or the add-on operator directly would not cover that path.
"""

from __future__ import annotations

import copy
import os
import sys
import time
import traceback
from pathlib import Path

import bmesh
import bpy
from bpy_extras import view3d_utils
from mathutils import Quaternion, Vector

# Keep startup UI and delayed view animation from consuming or moving simulated
# events during this real-window test.
bpy.context.preferences.view.show_splash = False
bpy.context.preferences.view.smooth_view = 0
bpy.context.preferences.view.use_save_prompt = False
bpy.context.preferences.filepaths.use_auto_save_temporary_files = False

PACKAGE_PARENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_PARENT))

import ydd_symmetric_edit as addon  # noqa: E402
from ydd_symmetric_edit import keymaps, layer_names, matching, operators, snapshot  # noqa: E402

OBJECT_NAME = "YSE_ToolRouteHistoryObject"
MESH_NAME = "YSE_ToolRouteHistoryMesh"
CUT_X_COORDINATES = (-1.65, -1.35)
EXPECTED_TOPOLOGY = {
    "baseline": (8, 8, 2),
    "f1": (12, 14, 4),
    "f2": (16, 20, 6),
}
EXPECTED_SELECTION_COUNTS = {
    "baseline": (0, 0, 0),
    "f1": (2, 1, 0),
    "f2": (4, 4, 1),
}
HISTORY_STEPS = (
    ("undo 1", "undo", "f1", 1),
    ("undo 2", "undo", "baseline", 0),
    ("redo 1", "redo", "f1", 1),
    ("redo 2", "redo", "f2", 2),
)
STATE = {
    "addon_registered": False,
    "cut_index": 0,
    "history_index": 0,
    "events": [],
    "deadline": 0.0,
    "snapshots": {},
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
    return identifiers


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
    return tuple(
        name for name in layer_names.TEMP_LAYER_NAMES if any(layers.get(name) is not None for layers in layer_groups)
    )


def fail(message=""):
    if message:
        print(f"YSE_TOOL_ROUTE_HISTORY_ERROR={message}", flush=True)
    traceback.print_exc()
    print(f"YSE_TOOL_ROUTE_HISTORY_PHASE={STATE.get('phase')}", flush=True)
    print(f"YSE_TOOL_ROUTE_HISTORY_TOPOLOGY={topology()}", flush=True)
    print(
        f"YSE_TOOL_ROUTE_HISTORY_TEMP_LAYERS={temporary_layer_names()}",
        flush=True,
    )
    print(f"YSE_TOOL_ROUTE_HISTORY_MODAL_IDS={modal_identifiers()}", flush=True)
    print(f"YSE_TOOL_ROUTE_HISTORY_SESSIONS={list(operators._SESSIONS)}", flush=True)
    history_records = [(token, record.status) for token, record in operators._HISTORY_RECORDS.items()]
    print(
        f"YSE_TOOL_ROUTE_HISTORY_RECORDS={history_records}",
        flush=True,
    )
    print("YSE_TOOL_ROUTE_HISTORY_TEST_FAILED", flush=True)
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


def window_coordinate(region, region_3d, coordinate):
    local = view3d_utils.location_3d_to_region_2d(
        region,
        region_3d,
        Vector(coordinate),
    )
    if local is None:
        raise RuntimeError(f"Could not project test point {coordinate}")
    return int(round(region.x + local.x)), int(round(region.y + local.y))


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


def selection_counts(signature):
    return tuple(len(elements) for elements in signature[:3])


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
            coordinates.append(x)
    return sorted(coordinates)


def assert_completed_cuts(bm, count):
    actual = internal_vertical_cut_coordinates(bm)
    expected = sorted(coordinate for source in CUT_X_COORDINATES[:count] for coordinate in (source, -source))
    assert len(actual) == len(expected), (actual, expected)
    assert all(
        abs(actual_coordinate - expected_coordinate) <= 5.0e-3
        for actual_coordinate, expected_coordinate in zip(actual, expected, strict=True)
    ), (actual, expected)
    negative = [coordinate for coordinate in actual if coordinate < 0.0]
    positive = [coordinate for coordinate in actual if coordinate > 0.0]
    assert len(negative) == len(positive) == count, actual
    assert all(abs(left + right) <= 1.0e-7 for left, right in zip(negative, reversed(positive), strict=True)), actual


def active_tool_id():
    tool = bpy.context.workspace.tools.from_space_view3d_mode(
        bpy.context.mode,
        create=False,
    )
    return tool.idname if tool is not None else None


def assert_no_temporary_data(bm):
    assert temporary_layer_names() == (), temporary_layer_names()
    assert not any(
        data.name.startswith(("YSE_TemporaryCutter", "YSE_TemporaryBackup"))
        for data in (*bpy.data.objects, *bpy.data.meshes)
    )


def assert_stage(stage_key, cut_count, expected_selection=None):
    obj = current_object()
    assert obj is not None
    assert bpy.context.mode == "EDIT_MESH", bpy.context.mode
    assert obj.mode == "EDIT"
    assert active_tool_id() == "builtin.knife", active_tool_id()
    assert bpy.context.view_layer.objects.active is obj
    assert tuple(sorted(item.name for item in bpy.context.selected_objects)) == (OBJECT_NAME,)
    assert tuple(bpy.context.tool_settings.mesh_select_mode) == (True, False, False)

    bm = bmesh.from_edit_mesh(obj.data)
    actual_topology = (len(bm.verts), len(bm.edges), len(bm.faces))
    assert actual_topology == EXPECTED_TOPOLOGY[stage_key], actual_topology
    assert_completed_cuts(bm, cut_count)
    assert_no_temporary_data(bm)
    assert not operators._SESSIONS, operators._SESSIONS

    signature = selection_signature(bm)
    assert selection_counts(signature) == EXPECTED_SELECTION_COUNTS[stage_key], (
        selection_counts(signature),
        EXPECTED_SELECTION_COUNTS[stage_key],
        signature,
    )
    if stage_key != "baseline":
        # Native Knife selects only the source-side result. The mirrored
        # post-process must not replace or broaden that selection.
        assert all(coordinate[0] < 0.0 for coordinate in signature[0]), signature
        assert all(all(coordinate[0] < 0.0 for coordinate in edge) for edge in signature[1]), signature
        assert all(all(coordinate[0] < 0.0 for coordinate in face) for face in signature[2]), signature
    if expected_selection is not None:
        assert signature == expected_selection, (signature, expected_selection)
    return signature


def native_knife_running():
    return any("KNIFE_TOOL" in identifier.upper() for identifier in modal_identifiers())


def production_is_busy():
    return bool(
        native_knife_running()
        or operators._SESSIONS
        or operators._HISTORY_REPAIR_QUEUED
        or operators._HISTORY_REPAIR_BUSY
    )


def toolbar_route_ready():
    tool_items = [
        item
        for keymap, item in keymaps._REGISTERED_ITEMS
        if keymap.name == keymaps.TOOL_KEYMAP_NAME
        and item.idname == keymaps.INTERCEPT_OPERATOR
        and item.type == "LEFTMOUSE"
        and item.value == "PRESS"
    ]
    return bool(tool_items) and all(item.active for item in tool_items)


def wait_for_toolbar_route():
    try:
        if toolbar_route_ready():
            STATE["snapshots"]["baseline"] = assert_stage("baseline", 0)
            bpy.app.timers.register(begin_cut, first_interval=0.2)
            return None
        if time.monotonic() > STATE["deadline"]:
            raise RuntimeError("The enabled Knife Tool LEFTMOUSE intercept was not registered")
        return 0.05
    except BaseException:
        fail()
    return None


def queue_cut_events(cut_x):
    region = STATE["region"]
    region_3d = STATE["area"].spaces.active.region_3d
    start = window_coordinate(region, region_3d, (cut_x, -1.0, 0.0))
    end = window_coordinate(region, region_3d, (cut_x, 1.0, 0.0))
    STATE["events"] = [
        {"type": "MOUSEMOVE", "value": "NOTHING", "x": start[0], "y": start[1]},
        {"type": "LEFTMOUSE", "value": "PRESS", "x": start[0], "y": start[1]},
        {"type": "LEFTMOUSE", "value": "RELEASE", "x": start[0], "y": start[1]},
        {"type": "MOUSEMOVE", "value": "NOTHING", "x": end[0], "y": end[1]},
        {"type": "LEFTMOUSE", "value": "PRESS", "x": end[0], "y": end[1]},
        {"type": "LEFTMOUSE", "value": "RELEASE", "x": end[0], "y": end[1]},
        {"type": "RET", "value": "PRESS", "x": end[0], "y": end[1]},
        {"type": "RET", "value": "RELEASE", "x": end[0], "y": end[1]},
    ]


def begin_cut():
    try:
        cut_number = STATE["cut_index"] + 1
        STATE["phase"] = f"toolbar cut {cut_number}"
        queue_cut_events(CUT_X_COORDINATES[STATE["cut_index"]])
        bpy.app.timers.register(send_next_event, first_interval=0.05)
    except BaseException:
        fail()
    return None


def make_history_marker_object(name, token):
    mesh = bpy.data.meshes.new(f"{name}Mesh")
    mesh.from_pydata(
        [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
        [(0, 1), (1, 2), (2, 0)],
        [(0, 1, 2)],
    )
    mesh.update()
    edge_marker = mesh.attributes.new(layer_names.EDGE_ORIGINAL_LAYER, "INT", "EDGE")
    for item in edge_marker.data:
        item.value = 1
    history_marker = mesh.attributes.new(layer_names.HISTORY_TOKEN_LAYER, "INT", "FACE")
    for item in history_marker.data:
        item.value = token
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj, mesh


def assert_running_session_markers_survive_ambiguous_repair():
    records = list(operators._HISTORY_RECORDS.values())
    assert records
    live_token = operators._new_history_token()
    unrelated_token = operators._new_history_token()
    token_owned_obj, token_owned_mesh = make_history_marker_object(
        "YSE_RunningTokenOwner",
        live_token,
    )
    mesh_owned_obj, mesh_owned_mesh = make_history_marker_object(
        "YSE_RunningMeshOwner",
        unrelated_token,
    )
    session = copy.deepcopy(records[-1].session)
    session.window_pointer = -1
    session.object_name = mesh_owned_obj.name
    session.mesh_name = mesh_owned_mesh.name
    session.history_token = live_token
    session.symmetry_suspended = False
    operators._SESSIONS[session.window_pointer] = session

    try:
        assert operators._repair_history_state() is None
        assert token_owned_mesh.attributes.get(layer_names.HISTORY_TOKEN_LAYER) is not None
        assert mesh_owned_mesh.attributes.get(layer_names.HISTORY_TOKEN_LAYER) is not None
    finally:
        operators.cleanup_session(session.window_pointer)
        snapshot.remove_temporary_mesh_attributes(token_owned_mesh)
        for obj in (token_owned_obj, mesh_owned_obj):
            bpy.data.objects.remove(obj, do_unlink=True)
        for mesh in (token_owned_mesh, mesh_owned_mesh):
            if mesh.users == 0:
                bpy.data.meshes.remove(mesh)


def send_next_event():
    try:
        if STATE["events"]:
            STATE["window"].event_simulate(**STATE["events"].pop(0))
            return 0.08
        STATE["phase"] = f"waiting for toolbar cut {STATE['cut_index'] + 1}"
        STATE["deadline"] = time.monotonic() + 10.0
        bpy.app.timers.register(wait_for_cut, first_interval=0.05)
    except BaseException:
        fail()
    return None


def wait_for_cut():
    try:
        stage_key = "f1" if STATE["cut_index"] == 0 else "f2"
        expected = EXPECTED_TOPOLOGY[stage_key]
        if production_is_busy() or topology() != expected or temporary_layer_names():
            if time.monotonic() > STATE["deadline"]:
                raise RuntimeError(f"Timed out waiting for completed toolbar cut; expected {expected}")
            return 0.05

        signature = assert_stage(stage_key, STATE["cut_index"] + 1)
        STATE["snapshots"][stage_key] = signature
        committed = [record for record in operators._HISTORY_RECORDS.values() if record.status == "COMMITTED"]
        assert len(committed) == STATE["cut_index"] + 1, len(committed)

        STATE["cut_index"] += 1
        if STATE["cut_index"] < len(CUT_X_COORDINATES):
            bpy.app.timers.register(begin_cut, first_interval=0.2)
        else:
            assert_running_session_markers_survive_ambiguous_repair()
            bpy.app.timers.register(begin_history_step, first_interval=0.3)
    except BaseException:
        fail()
    return None


def begin_history_step():
    try:
        label, operation, _stage_key, _cut_count = HISTORY_STEPS[STATE["history_index"]]
        STATE["phase"] = label
        with bpy.context.temp_override(
            window=STATE["window"],
            area=STATE["area"],
            region=STATE["region"],
        ):
            result = getattr(bpy.ops.ed, operation)()
        assert result == {"FINISHED"}, result
        STATE["deadline"] = time.monotonic() + 10.0
        bpy.app.timers.register(wait_for_history_step, first_interval=0.05)
    except BaseException:
        fail()
    return None


def wait_for_history_step():
    try:
        label, _operation, stage_key, cut_count = HISTORY_STEPS[STATE["history_index"]]
        expected = EXPECTED_TOPOLOGY[stage_key]
        if production_is_busy() or topology() != expected or temporary_layer_names():
            if time.monotonic() > STATE["deadline"]:
                raise RuntimeError(f"Timed out waiting for {label}; expected {expected}")
            return 0.05

        assert_stage(
            stage_key,
            cut_count,
            expected_selection=STATE["snapshots"][stage_key],
        )
        STATE["history_index"] += 1
        if STATE["history_index"] < len(HISTORY_STEPS):
            bpy.app.timers.register(begin_history_step, first_interval=0.2)
        else:
            bpy.app.timers.register(finish_test, first_interval=0.2)
    except BaseException:
        fail()
    return None


def finish_test():
    try:
        STATE["phase"] = "final verification"
        assert_stage(
            "f2",
            2,
            expected_selection=STATE["snapshots"]["f2"],
        )
        assert toolbar_route_ready()
        assert keymaps._ENABLED
        assert len(operators._HISTORY_RECORDS) == 2
        assert all(record.status == "COMMITTED" for record in operators._HISTORY_RECORDS.values())
        print("YSE_TOOL_ROUTE_HISTORY_TEST_OK", flush=True)
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
        obj = make_mesh()
        with bpy.context.temp_override(window=window, area=area, region=region):
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.context.tool_settings.mesh_select_mode = (True, False, False)
            bpy.context.tool_settings.workspace_tool_type = "DEFAULT"
            bpy.ops.mesh.select_all(action="DESELECT")
            tool_result = bpy.ops.wm.tool_set_by_id(name="builtin.knife")
            assert tool_result == {"FINISHED"}, tool_result
            bpy.ops.ed.undo_push(message="YSE Tool Route History baseline")

        assert matching.enabled_mesh_symmetry_axes(obj) == (("X", 0),)
        assert keymaps._ENABLED
        STATE["deadline"] = time.monotonic() + 5.0
        bpy.app.timers.register(wait_for_toolbar_route, first_interval=0.05)
    except BaseException:
        fail()
    return None


bpy.app.timers.register(start_test, first_interval=0.25)
