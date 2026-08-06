# SPDX-License-Identifier: GPL-3.0-or-later

"""GUI regression test for the native Offset Edge Loop Cut route.

Run with Blender's real window/event loop::

    blender --factory-startup --enable-event-simulate \
        --python test_offset_loopcut_route_history.py

The test drives the Workspace Tool's actual LEFTMOUSE keymap route.  It covers
both a dragged confirmation and Blender's special Escape behavior: Escape
cancels only Edge Slide and keeps the new offset topology at factor zero.
"""

from __future__ import annotations

import math
import os
import sys
import time
import traceback
from collections import Counter
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

OBJECT_NAME = "YSE_OffsetRouteHistoryObject"
MESH_NAME = "YSE_OffsetRouteHistoryMesh"
TOOL_KEYMAP_NAME = "3D View Tool: Edit Mesh, Offset Edge Loop Cut"
UV_LAYER_NAME = "YSE_OffsetRouteUV"
EXPECTED_TOPOLOGY = {
    "baseline": (24, 34, 12),
    "cap_baseline": (24, 34, 12),
    "normal": (36, 54, 20),
    "normal_f9": (36, 54, 20),
    "zero": (36, 54, 20),
    "cap_zero": (36, 58, 24),
}
STATE = {
    "addon_registered": False,
    "scenario": "normal",
    "events": [],
    "phase": "startup",
    "deadline": 0.0,
    "snapshots": {},
    "history_steps": [],
    "history_index": 0,
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
    groups = (
        bm.verts.layers.int,
        bm.edges.layers.int,
        bm.faces.layers.int,
    )
    return tuple(name for name in core.TEMP_LAYER_NAMES if any(layers.get(name) is not None for layers in groups))


def fail(message=""):
    if message:
        print(f"YSE_OFFSET_ROUTE_HISTORY_ERROR={message}", flush=True)
    traceback.print_exc()
    print(f"YSE_OFFSET_ROUTE_HISTORY_PHASE={STATE.get('phase')}", flush=True)
    print(f"YSE_OFFSET_ROUTE_HISTORY_TOPOLOGY={topology()}", flush=True)
    obj = current_object()
    print(
        "YSE_OFFSET_ROUTE_HISTORY_CONTEXT="
        f"mode={bpy.context.mode}, edit={getattr(bpy.context.edit_object, 'name', None)}, "
        f"object={None if obj is None else (obj.name, obj.mode)}",
        flush=True,
    )
    print(
        f"YSE_OFFSET_ROUTE_HISTORY_TEMP_LAYERS={temporary_layer_names()}",
        flush=True,
    )
    print(f"YSE_OFFSET_ROUTE_HISTORY_MODAL_IDS={modal_identifiers()}", flush=True)
    print(f"YSE_OFFSET_ROUTE_HISTORY_SESSIONS={list(operators._SESSIONS)}", flush=True)
    print(
        "YSE_OFFSET_ROUTE_HISTORY_RECORDS="
        f"{[(token, record.status, record.session.tool_kind) for token, record in operators._HISTORY_RECORDS.items()]}",
        flush=True,
    )
    print("YSE_OFFSET_ROUTE_HISTORY_TEST_FAILED", flush=True)
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
    region_3d.view_distance = 7.0
    region_3d.update()


def make_mesh():
    for old in tuple(bpy.data.objects):
        bpy.data.objects.remove(old, do_unlink=True)

    vertices = []
    faces = []
    for columns in ((-4.0, -3.0, -2.0, -1.0), (1.0, 2.0, 3.0, 4.0)):
        base = len(vertices)
        for y in (-1.0, 0.0, 1.0):
            for x in columns:
                vertices.append((x, y, 0.0))
        for row in range(2):
            for column in range(3):
                first = base + row * 4 + column
                faces.append((first, first + 1, first + 5, first + 4))

    mesh = bpy.data.meshes.new(MESH_NAME)
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(OBJECT_NAME, mesh)
    bpy.context.scene.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    obj.use_mesh_mirror_x = True
    obj.use_mesh_mirror_y = False
    obj.use_mesh_mirror_z = False
    return obj


def select_source_loop_and_add_uv(obj):
    bm = bmesh.from_edit_mesh(obj.data)
    for element in (*bm.verts, *bm.edges, *bm.faces):
        element.select = False
    for edge in bm.edges:
        if all(abs(vertex.co.x + 2.0) <= 1.0e-7 for vertex in edge.verts):
            edge.select = True
    bm.select_flush_mode()

    uv_layer = bm.loops.layers.uv.new(UV_LAYER_NAME)
    for face in bm.faces:
        for loop in face.loops:
            # Asymmetric islands ensure the destination cannot accidentally
            # pass by copying source-side UV coordinates.
            x, y = loop.vert.co.x, loop.vert.co.y
            loop[uv_layer].uv = (
                0.07 * x + (0.2 if x > 0.0 else 0.75),
                0.19 * y + (0.15 if x > 0.0 else 0.6),
            )
    bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)


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


def mirrored_coordinate_key(coordinate):
    x, y, z = coordinate
    return coordinate_key((-x, y, z))


def edge_key(edge):
    return tuple(sorted(coordinate_key(vertex.co) for vertex in edge.verts))


def face_key(face):
    return tuple(sorted(coordinate_key(vertex.co) for vertex in face.verts))


def mirror_edge_key(value):
    return tuple(sorted(mirrored_coordinate_key(coordinate) for coordinate in value))


def mirror_face_key(value):
    return tuple(sorted(mirrored_coordinate_key(coordinate) for coordinate in value))


def assert_exact_geometry_symmetry(bm):
    vertices = Counter(coordinate_key(vertex.co) for vertex in bm.verts)
    edges = Counter(edge_key(edge) for edge in bm.edges)
    faces = Counter(face_key(face) for face in bm.faces)
    assert vertices == Counter({mirrored_coordinate_key(key): count for key, count in vertices.items()}), vertices
    assert edges == Counter({mirror_edge_key(key): count for key, count in edges.items()}), edges
    assert faces == Counter({mirror_face_key(key): count for key, count in faces.items()}), faces


def element_key(element):
    if isinstance(element, bmesh.types.BMVert):
        return "VERT", coordinate_key(element.co)
    if isinstance(element, bmesh.types.BMEdge):
        return "EDGE", edge_key(element)
    return "FACE", face_key(element)


def selection_signature(bm):
    return (
        tuple(sorted(coordinate_key(vertex.co) for vertex in bm.verts if vertex.select)),
        tuple(sorted(edge_key(edge) for edge in bm.edges if edge.select)),
        tuple(sorted(face_key(face) for face in bm.faces if face.select)),
        tuple(element_key(element) for element in bm.select_history),
    )


def assert_source_only_selection(signature, expected_edge_count):
    assert len(signature[1]) == expected_edge_count, signature
    assert all(coordinate[0] < 0.0 for coordinate in signature[0]), signature
    assert all(all(coordinate[0] < 0.0 for coordinate in edge) for edge in signature[1]), signature
    assert all(all(coordinate[0] < 0.0 for coordinate in face) for face in signature[2]), signature


def center_vertical_x_coordinates(bm):
    result = []
    for edge in bm.edges:
        a, b = edge.verts
        if abs(a.co.x - b.co.x) > 1.0e-7:
            continue
        if abs(a.co.z - b.co.z) > 1.0e-7:
            continue
        if abs(a.co.y - b.co.y) <= 1.0e-7:
            continue
        x = (a.co.x + b.co.x) * 0.5
        if 1.0 + 1.0e-6 < abs(x) < 3.0 - 1.0e-6:
            result.append(x)
    return sorted(result)


def assert_offset_factor(stage, bm):
    values = center_vertical_x_coordinates(bm)
    assert len(values) == 12, values
    if stage == "normal":
        assert any(abs(value + 2.0) > 0.05 for value in values if value < 0.0), values
        assert any(abs(value - 2.0) > 0.05 for value in values if value > 0.0), values
    elif stage == "zero":
        assert all(abs(abs(value) - 2.0) <= 1.0e-6 for value in values), values


def assert_cap_topology(bm):
    assert sum(len(face.verts) == 3 for face in bm.faces) == 8


def assert_uv_finite(bm):
    uv_layer = bm.loops.layers.uv.get(UV_LAYER_NAME)
    assert uv_layer is not None
    values = [float(component) for face in bm.faces for loop in face.loops for component in loop[uv_layer].uv]
    assert values
    assert all(math.isfinite(value) for value in values)


def active_tool_id():
    tool = bpy.context.workspace.tools.from_space_view3d_mode(
        bpy.context.mode,
        create=False,
    )
    return tool.idname if tool is not None else None


def assert_native_f9_target():
    active = bpy.context.active_operator
    assert active is not None, active
    assert active.bl_idname == "MESH_OT_offset_edge_loops_slide", active.bl_idname
    recent = list(bpy.context.window_manager.operators)
    assert recent, recent
    assert recent[-1].bl_idname == "MESH_OT_offset_edge_loops_slide", [operator.bl_idname for operator in recent[-5:]]


def active_offset_macro_children():
    operator = bpy.context.active_operator
    assert operator is not None
    assert operator.bl_idname == "MESH_OT_offset_edge_loops_slide"
    children = {child.bl_idname: child.properties for child in operator.macros}
    assert "MESH_OT_offset_edge_loops" in children, tuple(children)
    assert "TRANSFORM_OT_edge_slide" in children, tuple(children)
    return children


def assert_no_temporary_data(bm):
    assert temporary_layer_names() == (), temporary_layer_names()
    assert not any(
        data.name.startswith(("YSE_TemporaryCutter", "YSE_TemporaryBackup"))
        for data in (*bpy.data.objects, *bpy.data.meshes)
    )


def assert_stage(stage, expected_selection=None, check_f9=False):
    obj = current_object()
    assert obj is not None
    assert bpy.context.mode == "EDIT_MESH", bpy.context.mode
    assert obj.mode == "EDIT"
    assert bpy.context.view_layer.objects.active is obj
    assert tuple(sorted(item.name for item in bpy.context.selected_objects)) == (OBJECT_NAME,)
    assert tuple(bpy.context.tool_settings.mesh_select_mode) == (False, True, False)
    assert active_tool_id() == "builtin.offset_edge_loop_cut", active_tool_id()
    assert obj.use_mesh_mirror_x
    assert not obj.use_mesh_mirror_y
    assert not obj.use_mesh_mirror_z

    bm = bmesh.from_edit_mesh(obj.data)
    actual_topology = (len(bm.verts), len(bm.edges), len(bm.faces))
    assert actual_topology == EXPECTED_TOPOLOGY[stage], actual_topology
    assert_exact_geometry_symmetry(bm)
    assert_uv_finite(bm)
    assert_no_temporary_data(bm)
    assert not operators._SESSIONS, operators._SESSIONS

    signature = selection_signature(bm)
    if stage in {"baseline", "cap_baseline"}:
        expected_edges = 2 if stage == "baseline" else 1
        assert len(signature[1]) == expected_edges, signature
    else:
        expected_edges = 6 if stage == "cap_zero" else 4
        assert_source_only_selection(signature, expected_edges)
        if stage == "cap_zero":
            assert_cap_topology(bm)
        else:
            factor_stage = "normal" if stage == "normal_f9" else stage
            assert_offset_factor(factor_stage, bm)
    if expected_selection is not None:
        assert signature == expected_selection, (signature, expected_selection)
    if check_f9:
        assert_native_f9_target()
        if stage == "cap_zero":
            assert bpy.context.active_operator.MESH_OT_offset_edge_loops.use_cap_endpoint
    return signature


def native_offset_running():
    identifiers = " ".join(modal_identifiers()).upper()
    return "OFFSET_EDGE_LOOPS_SLIDE" in identifiers


def production_is_busy():
    return bool(
        native_offset_running()
        or operators._SESSIONS
        or operators._HISTORY_REPAIR_QUEUED
        or operators._HISTORY_REPAIR_BUSY
    )


def toolbar_route_ready():
    items = [
        item
        for keymap, item in keymaps._REGISTERED_ITEMS
        if keymap.name == TOOL_KEYMAP_NAME
        and item.idname == keymaps.INTERCEPT_OPERATOR
        and item.type == "LEFTMOUSE"
        and item.value == "PRESS"
        and keymaps.route_tool_kind(item.properties.route_key) == "OFFSET_LOOP_CUT"
    ]
    return bool(items) and all(item.active for item in items)


def immediate_route_ready():
    items = [
        item
        for keymap, item in keymaps._REGISTERED_ITEMS
        if keymap.name == "Mesh"
        and item.idname == keymaps.INTERCEPT_OPERATOR
        and item.type == "R"
        and item.value == "PRESS"
        and item.shift
        and item.ctrl
        and keymaps.route_tool_kind(item.properties.route_key) == "OFFSET_LOOP_CUT"
    ]
    return bool(items) and all(item.active for item in items)


def set_native_cap_endpoint(enabled):
    window_manager = bpy.context.window_manager
    configured = []
    for keymap in window_manager.keyconfigs.user.keymaps:
        if keymap.name != TOOL_KEYMAP_NAME:
            continue
        for item in keymap.keymap_items:
            if item.idname == "mesh.offset_edge_loops_slide" and item.type == "LEFTMOUSE" and item.value == "PRESS":
                item.properties.MESH_OT_offset_edge_loops.use_cap_endpoint = enabled
                configured.append(item)
    assert configured, "Native Offset toolbar keymap item was not found"


def prepare_cap_scenario():
    try:
        STATE["phase"] = "prepare Cap Endpoint scenario"
        obj = current_object()
        bm = current_bmesh()
        for element in (*bm.verts, *bm.edges, *bm.faces):
            element.select = False
        expected_coordinates = {(-3.0, 0.0, 0.0), (-2.0, 0.0, 0.0)}
        selected = []
        for edge in bm.edges:
            coordinates = {coordinate_key(vertex.co) for vertex in edge.verts}
            if coordinates == expected_coordinates:
                edge.select = True
                selected.append(edge)
        assert len(selected) == 1, len(selected)
        bm.select_flush_mode()
        bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
        set_native_cap_endpoint(True)
        with bpy.context.temp_override(
            window=STATE["window"],
            area=STATE["area"],
            region=STATE["region"],
        ):
            bpy.ops.ed.undo_push(message="YSE Offset Cap Endpoint baseline")
        STATE["snapshots"]["cap_baseline"] = assert_stage("cap_baseline")
        STATE["scenario"] = "cap_zero"
        bpy.app.timers.register(begin_operation, first_interval=0.25)
    except BaseException:
        fail()
    return None


def prepare_f9_scenario():
    try:
        STATE["phase"] = "prepare F9 follow-up scenario"
        obj = current_object()
        bm = current_bmesh()
        for element in (*bm.verts, *bm.edges, *bm.faces):
            element.select = False
        selected = []
        for edge in bm.edges:
            if all(abs(vertex.co.x + 2.0) <= 1.0e-7 for vertex in edge.verts):
                edge.select = True
                selected.append(edge)
        assert len(selected) == 2, len(selected)
        bm.select_flush_mode()
        bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
        set_native_cap_endpoint(False)
        with bpy.context.temp_override(
            window=STATE["window"],
            area=STATE["area"],
            region=STATE["region"],
        ):
            bpy.ops.ed.undo_push(message="YSE Offset F9 baseline")
        STATE["scenario"] = "normal_f9"
        bpy.app.timers.register(begin_operation, first_interval=0.25)
    except BaseException:
        fail()
    return None


def simulate(events, callback):
    STATE["events"] = list(events)
    STATE["event_callback"] = callback
    bpy.app.timers.register(send_next_event, first_interval=0.06)


def send_next_event():
    try:
        if STATE["events"]:
            STATE["window"].event_simulate(**STATE["events"].pop(0))
            return 0.09
        callback = STATE.pop("event_callback")
        bpy.app.timers.register(callback, first_interval=0.15)
    except BaseException:
        fail()
    return None


def begin_operation():
    try:
        scenario = STATE["scenario"]
        STATE["phase"] = f"{scenario} toolbar press"
        start = window_coordinate((-2.0, 0.0, 0.0))
        simulate(
            [
                {"type": "MOUSEMOVE", "value": "NOTHING", "x": start[0], "y": start[1]},
                {"type": "LEFTMOUSE", "value": "PRESS", "x": start[0], "y": start[1]},
            ],
            inspect_live_operation,
        )
    except BaseException:
        fail()
    return None


def inspect_live_operation():
    try:
        scenario = STATE["scenario"]
        STATE["phase"] = f"{scenario} live Edge Slide"
        expected_live = (30, 46, 18) if scenario == "cap_zero" else (30, 44, 16)
        assert topology() == expected_live, topology()
        assert native_offset_running(), modal_identifiers()
        sessions = list(operators._SESSIONS.values())
        assert len(sessions) == 1, sessions
        assert sessions[0].tool_kind == "OFFSET_LOOP_CUT", sessions[0].tool_kind
        assert sessions[0].symmetry_suspended
        obj = current_object()
        assert not obj.use_mesh_mirror_x
        assert not obj.use_mesh_mirror_y
        assert not obj.use_mesh_mirror_z

        position = window_coordinate((-1.35, 0.0, 0.0))
        if scenario in {"normal", "normal_f9"}:
            events = [
                {"type": "MOUSEMOVE", "value": "NOTHING", "x": position[0], "y": position[1]},
                {"type": "ZERO", "value": "PRESS", "unicode": "0", "x": position[0], "y": position[1]},
                {"type": "ZERO", "value": "RELEASE", "x": position[0], "y": position[1]},
                {"type": "PERIOD", "value": "PRESS", "unicode": ".", "x": position[0], "y": position[1]},
                {"type": "PERIOD", "value": "RELEASE", "x": position[0], "y": position[1]},
                {"type": "THREE", "value": "PRESS", "unicode": "3", "x": position[0], "y": position[1]},
                {"type": "THREE", "value": "RELEASE", "x": position[0], "y": position[1]},
                {"type": "FIVE", "value": "PRESS", "unicode": "5", "x": position[0], "y": position[1]},
                {"type": "FIVE", "value": "RELEASE", "x": position[0], "y": position[1]},
                {"type": "RET", "value": "PRESS", "x": position[0], "y": position[1]},
                {"type": "RET", "value": "RELEASE", "x": position[0], "y": position[1]},
                {"type": "LEFTMOUSE", "value": "RELEASE", "x": position[0], "y": position[1]},
            ]
        else:
            events = [
                {"type": "ESC", "value": "PRESS", "x": position[0], "y": position[1]},
                {"type": "ESC", "value": "RELEASE", "x": position[0], "y": position[1]},
            ]
        simulate(events, begin_wait_for_operation)
    except BaseException:
        fail()
    return None


def begin_wait_for_operation():
    STATE["phase"] = f"waiting for {STATE['scenario']} symmetric result"
    STATE["deadline"] = time.monotonic() + 12.0
    bpy.app.timers.register(wait_for_operation, first_interval=0.05)
    return None


def wait_for_operation():
    try:
        scenario = STATE["scenario"]
        expected = EXPECTED_TOPOLOGY[scenario]
        if production_is_busy() or topology() != expected or temporary_layer_names():
            if time.monotonic() > STATE["deadline"]:
                raise RuntimeError(f"Timed out waiting for {scenario}; expected {expected}, got {topology()}")
            return 0.05

        signature = assert_stage(scenario, check_f9=True)
        committed = [
            record
            for record in operators._HISTORY_RECORDS.values()
            if record.status == "COMMITTED" and record.session.tool_kind == "OFFSET_LOOP_CUT"
        ]
        expected_records = {
            "normal": 1,
            "zero": 2,
            "cap_zero": 3,
            "normal_f9": 4,
        }[scenario]
        assert len(committed) == expected_records, len(committed)

        if scenario == "normal_f9":
            STATE["snapshots"]["normal_f9_initial"] = signature
            children = active_offset_macro_children()
            assert abs(children["TRANSFORM_OT_edge_slide"].value - 0.35) <= 1.0e-6
            STATE["f9_operator_pointer"] = int(bpy.context.active_operator.as_pointer())
            children["TRANSFORM_OT_edge_slide"].value = 0.6
            STATE["phase"] = "F9 Adjust Last Operation factor repeat"
            with bpy.context.temp_override(
                window=STATE["window"],
                area=STATE["area"],
                region=STATE["region"],
            ):
                assert bpy.ops.ed.undo_redo() == {"FINISHED"}
            STATE["deadline"] = time.monotonic() + 12.0
            bpy.app.timers.register(
                wait_for_adjusted_offset,
                first_interval=0.05,
            )
            return None

        STATE["snapshots"][scenario] = signature
        schedule_history_steps(scenario)
    except BaseException:
        fail()
    return None


def wait_for_adjusted_offset():
    try:
        if production_is_busy() or topology() != EXPECTED_TOPOLOGY["normal"] or temporary_layer_names():
            if time.monotonic() > STATE["deadline"]:
                raise RuntimeError("Timed out waiting for F9-adjusted symmetric Offset")
            return 0.05

        signature = assert_stage("normal_f9", check_f9=True)
        assert signature != STATE["snapshots"]["normal_f9_initial"], signature
        STATE["snapshots"]["normal_f9"] = signature
        assert int(bpy.context.active_operator.as_pointer()) == STATE["f9_operator_pointer"]
        children = active_offset_macro_children()
        assert abs(children["TRANSFORM_OT_edge_slide"].value - 0.6) <= 1.0e-6
        committed = [
            record
            for record in operators._HISTORY_RECORDS.values()
            if record.status == "COMMITTED" and record.session.tool_kind == "OFFSET_LOOP_CUT"
        ]
        assert len(committed) == 5, len(committed)
        bpy.app.timers.register(finish_test, first_interval=0.25)
    except BaseException:
        fail()
    return None


def schedule_history_steps(scenario):
    STATE["history_index"] = 0
    if scenario == "normal":
        STATE["history_steps"] = [
            ("normal undo", "undo", "baseline"),
            ("normal redo", "redo", "normal"),
            ("normal second undo", "undo", "baseline"),
        ]
    elif scenario == "zero":
        STATE["history_steps"] = [
            ("zero undo", "undo", "baseline"),
            ("zero redo", "redo", "zero"),
            ("zero second undo", "undo", "baseline"),
        ]
    else:
        STATE["history_steps"] = [
            ("cap zero undo", "undo", "cap_baseline"),
            ("cap zero redo", "redo", "cap_zero"),
            ("cap zero second undo", "undo", "cap_baseline"),
        ]
    bpy.app.timers.register(begin_history_step, first_interval=0.25)


def begin_history_step():
    try:
        label, operation, _stage = STATE["history_steps"][STATE["history_index"]]
        STATE["phase"] = label
        with bpy.context.temp_override(
            window=STATE["window"],
            area=STATE["area"],
            region=STATE["region"],
        ):
            result = getattr(bpy.ops.ed, operation)()
        assert result == {"FINISHED"}, result
        STATE["deadline"] = time.monotonic() + 12.0
        bpy.app.timers.register(wait_for_history_step, first_interval=0.05)
    except BaseException:
        fail()
    return None


def wait_for_history_step():
    try:
        _label, _operation, stage = STATE["history_steps"][STATE["history_index"]]
        expected = EXPECTED_TOPOLOGY[stage]
        if production_is_busy() or topology() != expected or temporary_layer_names():
            if time.monotonic() > STATE["deadline"]:
                raise RuntimeError(f"History step expected {expected}, got {topology()}")
            return 0.05

        expected_selection = STATE["snapshots"].get(stage)
        assert_stage(
            stage,
            expected_selection=expected_selection,
        )
        STATE["history_index"] += 1
        if STATE["history_index"] < len(STATE["history_steps"]):
            bpy.app.timers.register(begin_history_step, first_interval=0.2)
            return None

        if STATE["scenario"] == "normal":
            STATE["scenario"] = "zero"
            bpy.app.timers.register(begin_operation, first_interval=0.25)
        elif STATE["scenario"] == "zero":
            bpy.app.timers.register(prepare_cap_scenario, first_interval=0.25)
        else:
            bpy.app.timers.register(prepare_f9_scenario, first_interval=0.25)
    except BaseException:
        fail()
    return None


def wait_for_toolbar_route():
    try:
        if toolbar_route_ready() and immediate_route_ready():
            baseline = assert_stage("baseline")
            STATE["snapshots"]["baseline"] = baseline
            bpy.app.timers.register(begin_operation, first_interval=0.2)
            return None
        if time.monotonic() > STATE["deadline"]:
            raise RuntimeError("Offset LEFTMOUSE and Ctrl+Shift+R intercepts were not registered")
        return 0.05
    except BaseException:
        fail()
    return None


def finish_test():
    try:
        STATE["phase"] = "final verification"
        assert_stage(
            "normal_f9",
            expected_selection=STATE["snapshots"]["normal_f9"],
            check_f9=True,
        )
        assert toolbar_route_ready()
        assert immediate_route_ready()
        assert keymaps._ENABLED
        offset_records = [
            record for record in operators._HISTORY_RECORDS.values() if record.session.tool_kind == "OFFSET_LOOP_CUT"
        ]
        assert len(offset_records) == 5, len(offset_records)
        assert all(record.status == "COMMITTED" for record in offset_records)
        print("YSE_OFFSET_ROUTE_HISTORY_TEST_OK", flush=True)
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
            bpy.context.tool_settings.mesh_select_mode = (False, True, False)
            bpy.context.tool_settings.workspace_tool_type = "DEFAULT"
            select_source_loop_and_add_uv(obj)
            result = bpy.ops.wm.tool_set_by_id(name="builtin.offset_edge_loop_cut")
            assert result == {"FINISHED"}, result
            bpy.ops.ed.undo_push(message="YSE Offset Route History baseline")

        assert core.enabled_mesh_symmetry_axes(obj) == (("X", 0),)
        assert keymaps._ENABLED
        STATE["deadline"] = time.monotonic() + 6.0
        bpy.app.timers.register(wait_for_toolbar_route, first_interval=0.05)
    except BaseException:
        fail()
    return None


bpy.app.timers.register(start_test, first_interval=0.3)
