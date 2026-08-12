# SPDX-License-Identifier: GPL-3.0-or-later

"""Compare native and enabled toolbar Knife context after confirmation.

This GUI test uses Blender's actual ``builtin.knife`` WorkSpaceTool keymap.  It
records the confirmed native Knife result for four mesh-selection states, then
repeats the same physical toolbar events with ydd Symmetric Edit enabled.  The
mirrored topology may differ, but the source selection and tool context must be
identical to Blender's native result.
"""

from __future__ import annotations

import json
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
from ydd_symmetric_edit import keymaps, layer_names, operators  # noqa: E402

CASES = ("EDGE", "VERT", "FACE", "NONE")
PHASES = tuple((selection_case, enabled) for selection_case in CASES for enabled in (False, True))
KNIFE_PROPERTY_NAMES = (
    "use_occlude_geometry",
    "only_selected",
    "xray",
    "visible_measurements",
    "angle_snapping",
    "angle_snapping_increment",
    "wait_for_input",
)
CONTEXT_FIELDS = (
    "context_mode",
    "object_mode",
    "active_tool",
    "workspace_tool_type",
    "mesh_select_mode",
    "knife_properties",
    "cursor",
)
COMPARISON_FIELDS = (
    "selected_vertices",
    "selected_edges",
    "selected_faces",
    "selection_history",
    "active_element",
    *CONTEXT_FIELDS,
)

STATE = {
    "phase_index": 0,
    "native_results": {},
    "events": [],
    "deadline": 0.0,
}


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


def window_coordinate(region, region_3d, coordinate):
    local = view3d_utils.location_3d_to_region_2d(
        region,
        region_3d,
        Vector(coordinate),
    )
    if local is None:
        raise RuntimeError(f"Could not project test point {coordinate}")
    return int(round(region.x + local.x)), int(round(region.y + local.y))


def make_mesh():
    mesh = bpy.data.meshes.new("YSE_ToolContextMesh")
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
    obj = bpy.data.objects.new("YSE_ToolContextObject", mesh)
    bpy.context.scene.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    obj.use_mesh_mirror_x = True
    obj.use_mesh_mirror_y = False
    obj.use_mesh_mirror_z = False
    return obj


def remove_test_objects(context):
    if context.edit_object is not None:
        bpy.ops.object.mode_set(mode="OBJECT")
    for obj in tuple(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)


def configure_selection(context, obj, selection_case):
    select_modes = {
        "VERT": (True, False, False),
        "EDGE": (False, True, False),
        "FACE": (False, False, True),
        "NONE": (False, True, False),
    }
    context.tool_settings.mesh_select_mode = select_modes[selection_case]
    bpy.ops.mesh.select_all(action="DESELECT")

    bm = bmesh.from_edit_mesh(obj.data)
    bm.select_history.clear()
    if selection_case == "VERT":
        chosen = min(
            bm.verts,
            key=lambda vertex: (vertex.co - Vector((-2.0, -1.0, 0.0))).length_squared,
        )
        chosen.select = True
        bm.select_history.add(chosen)
    elif selection_case == "EDGE":
        chosen = min(
            bm.edges,
            key=lambda edge: sum((vertex.co - Vector((-1.5, -1.0, 0.0))).length_squared for vertex in edge.verts),
        )
        chosen.select = True
        for vertex in chosen.verts:
            vertex.select = True
        bm.select_history.add(chosen)
    elif selection_case == "FACE":
        chosen = min(
            bm.faces,
            key=lambda face: (face.calc_center_median() - Vector((-1.5, 0.0, 0.0))).length_squared,
        )
        chosen.select_set(True)
        bm.select_history.add(chosen)

    bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)


def configure_tool(context):
    # ``tool_set_by_id`` chooses the toolbar item, while workspace_tool_type
    # selects its primary keymap rather than Blender's fallback Select keymap.
    context.tool_settings.workspace_tool_type = "DEFAULT"
    result = bpy.ops.wm.tool_set_by_id(name="builtin.knife")
    if result != {"FINISHED"}:
        raise RuntimeError(f"Could not activate builtin.knife: {result}")

    tool = context.workspace.tools.from_space_view3d_mode(
        "EDIT_MESH",
        create=False,
    )
    if tool is None or tool.idname != "builtin.knife":
        raise RuntimeError(f"Unexpected active tool: {getattr(tool, 'idname', None)}")
    properties = tool.operator_properties("mesh.knife_tool")
    properties.use_occlude_geometry = False
    properties.only_selected = False
    properties.xray = False
    properties.visible_measurements = "DISTANCE"
    properties.angle_snapping = "NONE"
    properties.angle_snapping_increment = math.radians(17.0)


def coordinate_key(coordinate):
    return tuple(round(float(value), 6) for value in coordinate)


def element_key(element):
    if isinstance(element, bmesh.types.BMVert):
        return "VERT", coordinate_key(element.co)
    if isinstance(element, bmesh.types.BMEdge):
        return "EDGE", tuple(sorted(coordinate_key(vertex.co) for vertex in element.verts))
    if isinstance(element, bmesh.types.BMFace):
        return "FACE", tuple(sorted(coordinate_key(vertex.co) for vertex in element.verts))
    raise TypeError(type(element).__name__)


def active_tool(context):
    tool = context.workspace.tools.from_space_view3d_mode(
        "EDIT_MESH",
        create=False,
    )
    return getattr(tool, "idname", None)


def knife_properties(context):
    tool = context.workspace.tools.from_space_view3d_mode(
        "EDIT_MESH",
        create=False,
    )
    properties = tool.operator_properties("mesh.knife_tool")
    return tuple((name, getattr(properties, name)) for name in KNIFE_PROPERTY_NAMES)


def snapshot(context, obj):
    bm = bmesh.from_edit_mesh(obj.data)
    return {
        "counts": (len(bm.verts), len(bm.edges), len(bm.faces)),
        "selected_vertices": tuple(sorted(coordinate_key(vertex.co) for vertex in bm.verts if vertex.select)),
        "selected_edges": tuple(
            sorted(
                tuple(sorted(coordinate_key(vertex.co) for vertex in edge.verts)) for edge in bm.edges if edge.select
            )
        ),
        "selected_faces": tuple(
            sorted(
                tuple(sorted(coordinate_key(vertex.co) for vertex in face.verts)) for face in bm.faces if face.select
            )
        ),
        "selection_history": tuple(element_key(element) for element in bm.select_history),
        "active_element": (element_key(bm.select_history.active) if bm.select_history.active is not None else None),
        "context_mode": context.mode,
        "object_mode": obj.mode,
        "active_tool": active_tool(context),
        "workspace_tool_type": context.tool_settings.workspace_tool_type,
        "mesh_select_mode": tuple(context.tool_settings.mesh_select_mode),
        "knife_properties": knife_properties(context),
        "cursor": (
            coordinate_key(context.scene.cursor.location),
            context.scene.cursor.rotation_mode,
            coordinate_key(context.scene.cursor.rotation_euler),
        ),
    }


def modal_identifiers():
    try:
        return [operator.bl_rna.identifier for operator in STATE["window"].modal_operators]
    except Exception:
        return []


def has_active_knife_modal():
    return any("KNIFE" in identifier.upper() for identifier in modal_identifiers())


def has_temporary_layers(obj):
    bm = bmesh.from_edit_mesh(obj.data)
    return any(
        layers.get(name) is not None
        for layers, name in (
            (bm.edges.layers.int, layer_names.EDGE_ORIGINAL_LAYER),
            (bm.faces.layers.int, layer_names.FACE_ID_LAYER),
            (bm.faces.layers.int, layer_names.HISTORY_TOKEN_LAYER),
            (bm.verts.layers.int, layer_names.VERT_SELECTION_LAYER),
            (bm.edges.layers.int, layer_names.EDGE_SELECTION_LAYER),
            (bm.faces.layers.int, layer_names.FACE_SELECTION_LAYER),
        )
    )


def fail(message):
    print(f"TOOL_CONTEXT_SELECTION_ERROR={message}", flush=True)
    print(f"TOOL_CONTEXT_SELECTION_PHASE={STATE.get('phase')}", flush=True)
    print(f"TOOL_CONTEXT_SELECTION_MODALS={modal_identifiers()}", flush=True)
    print(
        f"TOOL_CONTEXT_SELECTION_SESSIONS={list(operators._SESSIONS)}",
        flush=True,
    )
    traceback.print_exc()
    print("YSE_TOOL_CONTEXT_SELECTION_TEST_FAILED", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(1)


def assert_equal(actual, expected, label):
    if actual != expected:
        raise AssertionError(f"{label} differs\nexpected={expected!r}\nactual={actual!r}")


def finish_test():
    try:
        STATE["phase"] = "final verification"
        assert not operators._SESSIONS, operators._SESSIONS
        print("YSE_TOOL_CONTEXT_SELECTION_TEST_OK", flush=True)
        addon.unregister()
        bpy.ops.wm.quit_blender()
    except BaseException as exc:
        fail(str(exc))
    return None


def verify_phase():
    try:
        selection_case, enabled = STATE["current_phase"]
        obj = STATE["object"]
        expected_counts = (12, 14, 4) if enabled else (10, 11, 3)
        bm = bmesh.from_edit_mesh(obj.data)
        counts = (len(bm.verts), len(bm.edges), len(bm.faces))
        busy = has_active_knife_modal() or bool(operators._SESSIONS)
        if busy or counts != expected_counts or has_temporary_layers(obj):
            if time.monotonic() > STATE["deadline"]:
                raise RuntimeError(
                    "Timed out waiting for toolbar Knife completion: "
                    f"counts={counts}, expected={expected_counts}, "
                    f"temporary_layers={has_temporary_layers(obj)}"
                )
            return 0.05

        with bpy.context.temp_override(
            window=STATE["window"],
            area=STATE["area"],
            region=STATE["region"],
        ):
            after = snapshot(bpy.context, obj)

        for field in CONTEXT_FIELDS:
            assert_equal(
                after[field],
                STATE["before"][field],
                f"{selection_case} {'enabled' if enabled else 'native'} {field}",
            )

        if enabled:
            native = STATE["native_results"][selection_case]
            for field in COMPARISON_FIELDS:
                assert_equal(
                    after[field],
                    native[field],
                    f"{selection_case} enabled/native {field}",
                )
        else:
            STATE["native_results"][selection_case] = after

        print(
            "TOOL_CONTEXT_SELECTION_CASE="
            + json.dumps(
                {
                    "case": selection_case,
                    "enabled": enabled,
                    "counts": after["counts"],
                    "selected": (
                        len(after["selected_vertices"]),
                        len(after["selected_edges"]),
                        len(after["selected_faces"]),
                    ),
                    "active_tool": after["active_tool"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

        STATE["phase_index"] += 1
        if STATE["phase_index"] == len(PHASES):
            bpy.app.timers.register(finish_test, first_interval=0.2)
        else:
            bpy.app.timers.register(setup_phase, first_interval=0.25)
    except BaseException as exc:
        fail(str(exc))
    return None


def send_next_event():
    try:
        if STATE["events"]:
            STATE["window"].event_simulate(**STATE["events"].pop(0))
            return 0.08
        STATE["phase"] += " / waiting for completion"
        STATE["deadline"] = time.monotonic() + 10.0
        bpy.app.timers.register(verify_phase, first_interval=0.25)
    except BaseException as exc:
        fail(str(exc))
    return None


def setup_phase():
    try:
        selection_case, enabled = PHASES[STATE["phase_index"]]
        STATE["current_phase"] = selection_case, enabled
        STATE["phase"] = f"{selection_case} / {'enabled add-on' if enabled else 'native'}"
        window, area, region = viewport_context()
        STATE.update(window=window, area=area, region=region)

        with bpy.context.temp_override(window=window, area=area, region=region):
            addon.sync_persistent_keymap(False)
            operators.clear_history_records()
            remove_test_objects(bpy.context)
            configure_view(area)
            obj = make_mesh()
            bpy.ops.object.mode_set(mode="EDIT")
            configure_selection(bpy.context, obj, selection_case)
            bpy.context.scene.cursor.location = (0.25, -0.5, 1.25)
            bpy.context.scene.cursor.rotation_mode = "XYZ"
            bpy.context.scene.cursor.rotation_euler = (0.1, 0.2, 0.3)
            configure_tool(bpy.context)

            if enabled:
                addon.sync_persistent_keymap(True)
                tool_intercepts = [
                    item
                    for keymap, item in keymaps._REGISTERED_ITEMS
                    if keymap.name == keymaps.TOOL_KEYMAP_NAME and item.active
                ]
                if not tool_intercepts:
                    raise RuntimeError("The enabled toolbar Knife intercept is missing")

            STATE["object"] = obj
            STATE["before"] = snapshot(bpy.context, obj)

        region_3d = area.spaces.active.region_3d
        start = window_coordinate(region, region_3d, (-1.65, -1.0, 0.0))
        end = window_coordinate(region, region_3d, (-1.65, 1.0, 0.0))
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
        bpy.app.timers.register(send_next_event, first_interval=0.25)
    except BaseException as exc:
        fail(str(exc))
    return None


def start_test():
    try:
        STATE["phase"] = "register add-on"
        addon.register()
        addon.sync_persistent_keymap(False)
        bpy.app.timers.register(setup_phase, first_interval=0.35)
    except BaseException as exc:
        fail(str(exc))
    return None


bpy.app.timers.register(start_test, first_interval=0.25)
