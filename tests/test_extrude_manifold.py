# SPDX-License-Identifier: GPL-3.0-or-later

"""GUI regression for Extrude Manifold (Stage 4 / EXTRUDE_MANIFOLD).

Run with Blender's real window/event loop::

    blender --factory-startup --enable-event-simulate \\
        --python test_extrude_manifold.py
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
from ydd_symmetric_edit import extrude_menu, keymaps, layer_names, operators  # noqa: E402

MARKER_OK = "YSE_EXTRUDE_MANIFOLD_TEST_OK"
MARKER_FAILED = "YSE_EXTRUDE_MANIFOLD_TEST_FAILED"
TOOL_ID = "builtin.extrude_manifold"
TOOL_KEYMAP_NAME = "3D View Tool: Edit Mesh, Extrude Manifold"
TOOL_KIND = "EXTRUDE_MANIFOLD"
NX, NY = 6, 4
PRECISION = 5
SINGLE_NATIVE = (4, 8, 4)
SINGLE_MIRROR = (8, 16, 8)
STEP_DISSOLVE_NATIVE = (4, 7, 3)
STEP_WELD_NATIVE = (2, 4, 2)
PROFILE_RIGHT = ((1.5, 0.0), (2.5, 0.0), (2.5, 1.0), (3.5, 1.0))
Y_ROWS = (0.0, 1.0)
STATE = {}

_DIGIT_TYPES = {
    "0": "ZERO",
    "1": "ONE",
    "2": "TWO",
    "3": "THREE",
    "4": "FOUR",
    "5": "FIVE",
    "6": "SIX",
    "7": "SEVEN",
    "8": "EIGHT",
    "9": "NINE",
    ".": "PERIOD",
}


def fail(message=""):
    if message:
        print(f"YSE_EXTRUDE_MANIFOLD_ERROR={message}", flush=True)
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


def click_drag_threshold_px():
    inputs = bpy.context.preferences.inputs
    if hasattr(inputs, "drag_threshold_mouse"):
        return max(1, int(inputs.drag_threshold_mouse))
    if hasattr(inputs, "drag_threshold"):
        return max(1, int(inputs.drag_threshold))
    raise RuntimeError("preferences.inputs exposes neither drag_threshold_mouse nor drag_threshold")


def configure_view(area):
    space = area.spaces.active
    space.show_gizmo = False
    space.show_gizmo_tool = False
    region_3d = space.region_3d
    region_3d.view_perspective = "ORTHO"
    region_3d.view_rotation = Quaternion((1.0, 0.0, 0.0, 0.0))
    region_3d.view_location = (0.0, 0.0, 0.0)
    region_3d.view_distance = 8.0
    region_3d.update()


def override():
    return bpy.context.temp_override(window=STATE["window"], area=STATE["area"], region=STATE["region"])


def window_coordinate(coordinate):
    region = STATE["region"]
    region_3d = STATE["area"].spaces.active.region_3d
    local = view3d_utils.location_3d_to_region_2d(region, region_3d, Vector(coordinate))
    if local is None:
        raise RuntimeError(f"could not project {coordinate}")
    return int(round(region.x + local.x)), int(round(region.y + local.y))


def grid_xy(i, j):
    return (i - NX / 2, j - NY / 2)


def _clear_scene_meshes():
    for old in tuple(bpy.data.objects):
        bpy.data.objects.remove(old, do_unlink=True)
    for old_mesh in tuple(bpy.data.meshes):
        if old_mesh.users == 0:
            bpy.data.meshes.remove(old_mesh)


def build_mesh(name):
    _clear_scene_meshes()
    mesh = bpy.data.meshes.new(f"YSE_ExtrudeManifoldMesh_{name}")
    coords, faces = [], []
    for j in range(NY + 1):
        for i in range(NX + 1):
            coords.append((i - NX / 2, j - NY / 2, 0.0))
    stride = NX + 1
    for j in range(NY):
        for i in range(NX):
            a = j * stride + i
            faces.append((a, a + 1, a + stride + 1, a + stride))
    mesh.from_pydata(coords, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(f"YSE_ExtrudeManifoldObject_{name}", mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.use_mesh_mirror_x = True
    obj.use_mesh_mirror_y = False
    obj.use_mesh_mirror_z = False
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    with override():
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.context.tool_settings.mesh_select_mode = (False, False, True)
        bpy.ops.ed.undo_push(message=f"YSE extrude manifold baseline {name}")
    STATE["object"] = obj
    return obj


def _add_step_strip(bm, profile_xz):
    rows = []
    for y in Y_ROWS:
        rows.append([bm.verts.new((x, y, z)) for (x, z) in profile_xz])
    for index in range(len(profile_xz) - 1):
        bm.faces.new((rows[0][index], rows[0][index + 1], rows[1][index + 1], rows[1][index]))


def build_step_pair(name):
    _clear_scene_meshes()
    mesh = bpy.data.meshes.new(f"YSE_ExtrudeManifoldStep_{name}")
    obj = bpy.data.objects.new(f"YSE_ExtrudeManifoldStepObject_{name}", mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.use_mesh_mirror_x = True
    obj.use_mesh_mirror_y = False
    obj.use_mesh_mirror_z = False
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bm = bmesh.new()
    _add_step_strip(bm, PROFILE_RIGHT)
    _add_step_strip(bm, tuple((-x, z) for x, z in PROFILE_RIGHT))
    bm.to_mesh(mesh)
    bm.free()
    with override():
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.context.tool_settings.mesh_select_mode = (False, False, True)
        bpy.ops.ed.undo_push(message=f"YSE extrude manifold step baseline {name}")
    STATE["object"] = obj
    return obj


def grid_face(bm, i, j):
    wanted = {
        grid_xy(i, j),
        grid_xy(i + 1, j),
        grid_xy(i + 1, j + 1),
        grid_xy(i, j + 1),
    }
    for face in bm.faces:
        have = {(round(float(vertex.co.x), 6), round(float(vertex.co.y), 6)) for vertex in face.verts}
        expect = {(round(x, 6), round(y, 6)) for x, y in wanted}
        if have == expect:
            return face
    raise AssertionError(f"grid face {i},{j} not found")


def clear_selection(bm):
    for face in bm.faces:
        face.select = False
    for edge in bm.edges:
        edge.select = False
    for vertex in bm.verts:
        vertex.select = False
    bm.select_history.clear()


def select_faces(bm, cells):
    bpy.context.tool_settings.mesh_select_mode = (False, False, True)
    clear_selection(bm)
    for cell in cells:
        grid_face(bm, *cell).select = True
    bm.select_flush_mode()


def select_right_bottom_face(bm):
    bpy.context.tool_settings.mesh_select_mode = (False, False, True)
    clear_selection(bm)
    matches = []
    for face in bm.faces:
        center = face.calc_center_median()
        if 1.5 < center.x < 2.5 and abs(center.z) < 1e-6:
            matches.append(face)
    assert len(matches) == 1, f"right bottom faces={[tuple(f.calc_center_median()) for f in matches]}"
    matches[0].select = True
    bm.select_flush_mode()


def coordinate_key(co):
    return tuple(round(float(value), PRECISION) for value in co)


def vertex_multiset(bm):
    return Counter(coordinate_key(vertex.co) for vertex in bm.verts)


def mirrored_multiset(bm):
    return Counter(coordinate_key((-vertex.co.x, vertex.co.y, vertex.co.z)) for vertex in bm.verts)


def assert_x_symmetric(bm):
    assert vertex_multiset(bm) == mirrored_multiset(bm), "vertex coordinates are not X-symmetric"


def assert_layers_removed(bm):
    domains = (
        ("verts", bm.verts.layers.int),
        ("edges", bm.edges.layers.int),
        ("faces", bm.faces.layers.int),
    )
    for name in layer_names.TEMP_LAYER_NAMES:
        for domain, layers in domains:
            assert layers.get(name) is None, f"temporary {domain} layer leaked: {name}"


def topology_counts(bm):
    return len(bm.verts), len(bm.edges), len(bm.faces)


def toolbar_route_ready():
    tool_items = [
        item
        for keymap, item in keymaps._REGISTERED_ITEMS
        if keymap.name == TOOL_KEYMAP_NAME
        and item.idname == keymaps.INTERCEPT_OPERATOR
        and item.type == "LEFTMOUSE"
        and item.value == "CLICK_DRAG"
    ]
    return bool(tool_items) and all(item.active for item in tool_items)


def activate_extrude_manifold_tool():
    with override():
        bpy.context.tool_settings.workspace_tool_type = "DEFAULT"
        result = bpy.ops.wm.tool_set_by_id(name=TOOL_ID)
        if result != {"FINISHED"}:
            raise RuntimeError(f"Could not activate {TOOL_ID}: {result}")
        tool = bpy.context.workspace.tools.from_space_view3d_mode("EDIT_MESH", create=False)
        if tool is None or tool.idname != TOOL_ID:
            raise RuntimeError(f"Unexpected active tool: {getattr(tool, 'idname', None)}")


def send_events(events, done, index=0, interval=0.09, done_delay=0.2):
    def step():
        try:
            if index < len(events):
                STATE["window"].event_simulate(**events[index])
                send_events(events, done, index + 1, interval=interval, done_delay=done_delay)
            else:
                bpy.app.timers.register(done, first_interval=done_delay)
        except BaseException:
            fail()
        return None

    bpy.app.timers.register(step, first_interval=interval)


def wait_settled(done, started=None):
    started = time.monotonic() if started is None else started

    def poll():
        try:
            busy = bool(STATE["window"].modal_operators) or bool(operators._SESSIONS)
            if busy:
                if time.monotonic() - started > 12.0:
                    raise RuntimeError(
                        f"extrude manifold flow never settled; "
                        f"modal={[op.bl_idname for op in STATE['window'].modal_operators]} "
                        f"sessions={list(operators._SESSIONS)}"
                    )
                return 0.1
            bpy.app.timers.register(done, first_interval=0.25)
        except BaseException:
            fail()
        return None

    bpy.app.timers.register(poll, first_interval=0.1)


def wait_for_toolbar_route(done, started=None):
    started = time.monotonic() if started is None else started

    def poll():
        try:
            if toolbar_route_ready():
                bpy.app.timers.register(done, first_interval=0.15)
                return None
            if time.monotonic() - started > 8.0:
                raise RuntimeError(
                    f"Extrude Manifold LEFTMOUSE CLICK_DRAG intercept was not registered; "
                    f"items={[(km.name, item.type, item.value) for km, item in keymaps._REGISTERED_ITEMS]}"
                )
            return 0.05
        except BaseException:
            fail()
        return None

    bpy.app.timers.register(poll, first_interval=0.05)


def wait_modal(done, started=None):
    started = time.monotonic() if started is None else started

    def poll():
        try:
            if tuple(STATE["window"].modal_operators):
                bpy.app.timers.register(done, first_interval=0.05)
                return None
            if time.monotonic() - started > 4.0:
                raise RuntimeError(
                    f"native dispatcher never went modal; sessions={list(operators._SESSIONS)}"
                )
            return 0.05
        except BaseException:
            fail()
        return None

    bpy.app.timers.register(poll, first_interval=0.05)


def tool_drag_events(cursor_xyz, drag=(0, 80)):
    x, y = window_coordinate(cursor_xyz)
    tx, ty = x + drag[0], y + drag[1]
    events = [
        {"type": "MOUSEMOVE", "value": "NOTHING", "x": x, "y": y},
        {"type": "LEFTMOUSE", "value": "PRESS", "x": x, "y": y},
    ]
    if drag != (0, 0):
        events.append({"type": "MOUSEMOVE", "value": "NOTHING", "x": x + drag[0] // 2, "y": y + drag[1] // 2})
        events.append({"type": "MOUSEMOVE", "value": "NOTHING", "x": tx, "y": ty})
    else:
        overshoot = click_drag_threshold_px() + 1
        events.append({"type": "MOUSEMOVE", "value": "NOTHING", "x": x, "y": y + overshoot})
        tx, ty = x, y
    events.append({"type": "LEFTMOUSE", "value": "RELEASE", "x": tx, "y": ty})
    return events


def drag_confirm_events(cursor_xyz, drag=(0, 80)):
    x, y = window_coordinate(cursor_xyz)
    tx, ty = x + drag[0], y + drag[1]
    return [
        {"type": "MOUSEMOVE", "value": "NOTHING", "x": x, "y": y},
        {"type": "MOUSEMOVE", "value": "NOTHING", "x": tx, "y": ty},
        {"type": "LEFTMOUSE", "value": "PRESS", "x": tx, "y": ty},
        {"type": "LEFTMOUSE", "value": "RELEASE", "x": tx, "y": ty},
    ]


def numeric_confirm_events(text):
    events = []
    for char in text:
        key_type = _DIGIT_TYPES[char]
        events.append({"type": key_type, "value": "PRESS", "unicode": char})
        events.append({"type": key_type, "value": "RELEASE"})
    events.append({"type": "RET", "value": "PRESS"})
    events.append({"type": "RET", "value": "RELEASE"})
    return events


def unique_record(object_name, tool_kind):
    records = [
        record
        for record in operators._HISTORY_RECORDS.values()
        if record.session.object_name == object_name and record.session.tool_kind == tool_kind
    ]
    assert records, f"no {tool_kind} history for {object_name!r}"
    sequence = max(record.sequence for record in records)
    matches = [record for record in records if record.sequence == sequence]
    assert len(matches) == 1, f"non-unique max sequence {sequence}: {matches}"
    return matches[0]


def assert_options_captured(record, *, dissolve_ortho=False):
    assert record.session.extrude_options_captured is True, record.session
    options = record.session.extrude_options
    assert options is not None, f"{record.session.tool_kind} record has no extrude_options"
    props = dict(options.transform_props)
    value = props.get("value")
    assert isinstance(value, (list, tuple)) and len(value) == 3, props
    assert all(isinstance(item, (int, float)) and math.isfinite(float(item)) for item in value), props
    if dissolve_ortho:
        assert options.use_dissolve_ortho_edges is True, options


def net_of(bm):
    return (
        len(bm.verts) - STATE["baseline"][0],
        len(bm.edges) - STATE["baseline"][1],
        len(bm.faces) - STATE["baseline"][2],
    )


def verify_net_and_symmetric(bm, expected_net, *, undo=False, dissolve_ortho=False):
    got = net_of(bm)
    assert got == expected_net, f"net {got} != {expected_net}"
    assert_x_symmetric(bm)
    assert_layers_removed(bm)
    record = unique_record(STATE["object"].name, TOOL_KIND)
    assert record.session.tool_kind == TOOL_KIND, record.session.tool_kind
    assert_options_captured(record, dissolve_ortho=dissolve_ortho)
    if undo:
        with override():
            undo_result = bpy.ops.ed.undo()
        assert undo_result == {"FINISHED"}, undo_result
        bm2 = bmesh.from_edit_mesh(STATE["object"].data)
        counts = topology_counts(bm2)
        assert counts == STATE["baseline"], f"one undo did not restore baseline: {counts} != {STATE['baseline']}"
        assert_x_symmetric(bm2)
        assert_layers_removed(bm2)


def assert_declined(bm, native_net, reason_substr, case_name):
    got = net_of(bm)
    assert got == native_net, f"declined net {got} != native-only {native_net} counts={topology_counts(bm)}"
    assert vertex_multiset(bm) != mirrored_multiset(bm), "declined extrude should keep an unmirrored native result"
    record = unique_record(STATE["object"].name, TOOL_KIND)
    assert record.session.prepare_disposition == "DECLINE", record.session
    reason = record.session.prepare_disposition_reason or ""
    assert reason, "prepare_disposition_reason must be non-empty"
    if reason_substr:
        tokens = (reason_substr,) if isinstance(reason_substr, str) else tuple(reason_substr)
        assert any(token in reason for token in tokens), reason
    freeze = getattr(record.session, "extrude_freeze", None)
    assert freeze in (None, ()), freeze
    assert_layers_removed(bm)
    print(f"YSE_EXTRUDE_MANIFOLD_DECLINE_REASON_{case_name}={reason}", flush=True)
    return record


def run_tool_case(name, prepare, cursor_xyz, verify, *, drag=(0, 80)):
    def start(next_case):
        try:
            print(f"YSE_EXTRUDE_MANIFOLD_CASE={name}", flush=True)
            obj = build_mesh(name)
            bm = bmesh.from_edit_mesh(obj.data)
            prepare(bm, obj)
            bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
            STATE["baseline"] = topology_counts(bmesh.from_edit_mesh(obj.data))
            activate_extrude_manifold_tool()

            def settled():
                try:
                    bm2 = bmesh.from_edit_mesh(STATE["object"].data)
                    verify(bm2)
                    next_case()
                except BaseException:
                    fail()

            def after_route():
                send_events(tool_drag_events(cursor_xyz, drag=drag), lambda: wait_settled(settled))

            wait_for_toolbar_route(after_route)
        except BaseException:
            fail()

    return start


def invoke_wrapper_and_drag(name, cursor_xyz, verify, *, drag=(0, 80), build=build_mesh, prepare=None):
    def start(next_case):
        try:
            print(f"YSE_EXTRUDE_MANIFOLD_CASE={name}", flush=True)
            obj = build(name)
            bm = bmesh.from_edit_mesh(obj.data)
            if prepare is not None:
                prepare(bm, obj)
            else:
                select_faces(bm, [(4, 1)])
            bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
            STATE["baseline"] = topology_counts(bmesh.from_edit_mesh(obj.data))

            x, y = window_coordinate(cursor_xyz)
            STATE["window"].event_simulate(type="MOUSEMOVE", value="NOTHING", x=x, y=y)
            with override():
                result = bpy.ops.mesh.ydd_symmetric_edit_extrude_manifold("INVOKE_DEFAULT")
            assert result == {"FINISHED"}, result
            assert extrude_menu.WRAPPER_MANIFOLD == "mesh.ydd_symmetric_edit_extrude_manifold"

            def settled():
                try:
                    bm2 = bmesh.from_edit_mesh(STATE["object"].data)
                    verify(bm2)
                    next_case()
                except BaseException:
                    fail()

            def after_modal():
                send_events(drag_confirm_events(cursor_xyz, drag=drag), lambda: wait_settled(settled))

            wait_modal(after_modal)
        except BaseException:
            fail()

    return start


def invoke_wrapper_and_numeric(name, dz, verify, cursor_xyz=(2.0, 0.5, 0.0)):
    def start(next_case):
        try:
            print(f"YSE_EXTRUDE_MANIFOLD_CASE={name}", flush=True)
            obj = build_step_pair(name)
            bm = bmesh.from_edit_mesh(obj.data)
            select_right_bottom_face(bm)
            bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
            STATE["baseline"] = topology_counts(bmesh.from_edit_mesh(obj.data))
            assert STATE["baseline"] == (16, 20, 6), STATE["baseline"]

            x, y = window_coordinate(cursor_xyz)
            STATE["window"].event_simulate(type="MOUSEMOVE", value="NOTHING", x=x, y=y)
            with override():
                result = bpy.ops.mesh.ydd_symmetric_edit_extrude_manifold("INVOKE_DEFAULT")
            assert result == {"FINISHED"}, result

            def settled():
                try:
                    bm2 = bmesh.from_edit_mesh(STATE["object"].data)
                    verify(bm2)
                    next_case()
                except BaseException:
                    fail()

            def after_modal():
                text = f"{dz:g}"
                send_events(numeric_confirm_events(text), lambda: wait_settled(settled))

            wait_modal(after_modal)
        except BaseException:
            fail()

    return start


def invoke_wrapper_zero_offset(name, cursor_xyz, verify):
    def start(next_case):
        try:
            print(f"YSE_EXTRUDE_MANIFOLD_CASE={name}", flush=True)
            settings = bpy.context.scene.ydd_symmetric_edit
            previous = float(settings.tolerance)
            settings.tolerance = 0.05
            obj = build_mesh(name)
            bm = bmesh.from_edit_mesh(obj.data)
            select_faces(bm, [(4, 1)])
            bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
            STATE["baseline"] = topology_counts(bmesh.from_edit_mesh(obj.data))

            x, y = window_coordinate(cursor_xyz)
            STATE["window"].event_simulate(type="MOUSEMOVE", value="NOTHING", x=x, y=y)
            with override():
                result = bpy.ops.mesh.ydd_symmetric_edit_extrude_manifold("INVOKE_DEFAULT")
            assert result == {"FINISHED"}, result

            def settled():
                try:
                    bm2 = bmesh.from_edit_mesh(STATE["object"].data)
                    verify(bm2)
                    settings.tolerance = previous
                    next_case()
                except BaseException:
                    settings.tolerance = previous
                    fail()

            def after_modal():
                send_events(numeric_confirm_events("0.01"), lambda: wait_settled(settled))

            wait_modal(after_modal)
        except BaseException:
            fail()

    return start


def prepare_single(bm, _obj):
    select_faces(bm, [(4, 1)])


def run_all(cases, index=0):
    if index >= len(cases):
        print(MARKER_OK, flush=True)
        sys.stdout.flush()
        addon.unregister()
        bpy.ops.wm.quit_blender()
        return
    cases[index](lambda: run_all(cases, index + 1))


def start_test():
    try:
        addon.register()
        addon.sync_persistent_keymap(True)
        window, area, region = viewport_context()
        configure_view(area)
        STATE.update(window=window, area=area, region=region)
        cursor = (1.5, -0.5, 0.0)
        cases = [
            run_tool_case(
                "tool_single_face",
                prepare_single,
                cursor,
                lambda bm: verify_net_and_symmetric(bm, SINGLE_MIRROR, undo=True, dissolve_ortho=True),
            ),
            invoke_wrapper_and_drag(
                "menu_wrapper",
                cursor,
                lambda bm: verify_net_and_symmetric(bm, SINGLE_MIRROR, dissolve_ortho=True),
            ),
            invoke_wrapper_and_numeric(
                "step_dissolve_below",
                0.5,
                lambda bm: assert_declined(bm, STEP_DISSOLVE_NATIVE, None, "step_dissolve_below"),
            ),
            invoke_wrapper_and_numeric(
                "step_dissolve_past",
                1.5,
                lambda bm: assert_declined(bm, STEP_DISSOLVE_NATIVE, None, "step_dissolve_past"),
            ),
            invoke_wrapper_and_numeric(
                "step_weld",
                1.0,
                lambda bm: assert_declined(bm, STEP_WELD_NATIVE, None, "step_weld"),
            ),
            invoke_wrapper_zero_offset(
                "zero_offset",
                cursor,
                lambda bm: assert_declined(
                    bm,
                    SINGLE_NATIVE,
                    "zero-offset",
                    "zero_offset",
                ),
            ),
        ]
        run_all(cases)
    except BaseException:
        fail()
    return None


bpy.app.timers.register(start_test, first_interval=0.3)
