# SPDX-License-Identifier: GPL-3.0-or-later

"""GUI regression for Extrude Region tool drag (EXTRUDE_CONTEXT).

Run with Blender's real window/event loop::

    blender --factory-startup --enable-event-simulate \
        --python test_extrude_tool_context.py
"""

from __future__ import annotations

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
from ydd_symmetric_edit import keymaps, layer_names, operators  # noqa: E402

MARKER_OK = "YSE_EXTRUDE_TOOL_CONTEXT_TEST_OK"
MARKER_FAILED = "YSE_EXTRUDE_TOOL_CONTEXT_TEST_FAILED"
TOOL_ID = "builtin.extrude_region"
TOOL_KEYMAP_NAME = "3D View Tool: Edit Mesh, Extrude Region"
NX, NY = 6, 4
PRECISION = 5
SINGLE_FACE_NATIVE_NET = (4, 8, 4)
STATE = {}


def fail(message=""):
    if message:
        print(f"YSE_EXTRUDE_TOOL_CONTEXT_ERROR={message}", flush=True)
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
    # Preferences store the threshold in UI dots; the window manager compares
    # against physical pixels, so display scaling widens the real threshold.
    inputs = bpy.context.preferences.inputs
    if hasattr(inputs, "drag_threshold_mouse"):
        base = int(inputs.drag_threshold_mouse)
    elif hasattr(inputs, "drag_threshold"):
        base = int(inputs.drag_threshold)
    else:
        raise RuntimeError("preferences.inputs exposes neither drag_threshold_mouse nor drag_threshold")
    scale = float(getattr(bpy.context.preferences.system, "ui_scale", 1.0))
    return max(1, int(round(base * scale)) + 1)


def configure_view(area, *, gizmos=False):
    space = area.spaces.active
    space.show_gizmo = bool(gizmos)
    space.show_gizmo_tool = bool(gizmos)
    region_3d = space.region_3d
    region_3d.view_perspective = "ORTHO"
    region_3d.view_rotation = Quaternion((1.0, 0.0, 0.0, 0.0))
    region_3d.view_location = (0.0, 0.0, 0.0)
    region_3d.view_distance = 14.0
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


def build_mesh(name):
    for old in tuple(bpy.data.objects):
        bpy.data.objects.remove(old, do_unlink=True)
    for old_mesh in tuple(bpy.data.meshes):
        if old_mesh.users == 0:
            bpy.data.meshes.remove(old_mesh)
    mesh = bpy.data.meshes.new(f"YSE_ExtrudeCtxMesh_{name}")
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
    obj = bpy.data.objects.new(f"YSE_ExtrudeCtxObject_{name}", mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.use_mesh_mirror_x = True
    obj.use_mesh_mirror_y = False
    obj.use_mesh_mirror_z = False
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    with override():
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.context.tool_settings.mesh_select_mode = (True, False, False)
        bpy.ops.ed.undo_push(message=f"YSE extrude context baseline {name}")
    STATE["object"] = obj
    return obj


def grid_vert(bm, i, j):
    x, y = grid_xy(i, j)
    for vertex in bm.verts:
        if abs(vertex.co.x - x) < 1e-4 and abs(vertex.co.y - y) < 1e-4 and abs(vertex.co.z) < 1e-4:
            return vertex
    raise AssertionError(f"grid vert {i},{j} not found")


def grid_edge(bm, a, b):
    first, second = grid_vert(bm, *a), grid_vert(bm, *b)
    for edge in first.link_edges:
        if edge.other_vert(first) is second:
            return edge
    raise AssertionError(f"grid edge {a}-{b} not found")


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


def select_edges(bm, pairs):
    bpy.context.tool_settings.mesh_select_mode = (False, True, False)
    clear_selection(bm)
    for pair in pairs:
        grid_edge(bm, *pair).select = True
    bm.select_flush_mode()


def select_verts(bm, ij_list):
    bpy.context.tool_settings.mesh_select_mode = (True, False, False)
    clear_selection(bm)
    for ij in ij_list:
        grid_vert(bm, *ij).select = True
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


def activate_extrude_region_tool():
    with override():
        bpy.context.tool_settings.workspace_tool_type = "DEFAULT"
        result = bpy.ops.wm.tool_set_by_id(name=TOOL_ID)
        if result != {"FINISHED"}:
            raise RuntimeError(f"Could not activate {TOOL_ID}: {result}")
        tool = bpy.context.workspace.tools.from_space_view3d_mode("EDIT_MESH", create=False)
        if tool is None or tool.idname != TOOL_ID:
            raise RuntimeError(f"Unexpected active tool: {getattr(tool, 'idname', None)}")
        keymap_name = getattr(tool, "keymap", None)
        if isinstance(keymap_name, (list, tuple)):
            keymap_name = keymap_name[0] if keymap_name else None
        if keymap_name not in {None, TOOL_KEYMAP_NAME}:
            print(f"YSE_EXTRUDE_TOOL_CONTEXT_KEYMAP={keymap_name!r}", flush=True)


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
                        f"extrude context flow never settled; "
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
                    f"Extrude Region LEFTMOUSE CLICK_DRAG intercept was not registered; "
                    f"items={[(km.name, item.type, item.value) for km, item in keymaps._REGISTERED_ITEMS]}"
                )
            return 0.05
        except BaseException:
            fail()
        return None

    bpy.app.timers.register(poll, first_interval=0.05)


def tool_drag_events(cursor_xyz, drag=(0, 80)):
    # event_simulate accepts only PRESS/RELEASE/NOTHING (CLICK_DRAG raises).
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
        # Exceed the tool CLICK_DRAG threshold so the KMI fires, then return to
        # the press point so the confirmed translate is a native zero.
        overshoot = click_drag_threshold_px() + 1
        events.append({"type": "MOUSEMOVE", "value": "NOTHING", "x": x, "y": y + overshoot})
        tx, ty = x, y
    events.append({"type": "LEFTMOUSE", "value": "RELEASE", "x": tx, "y": ty})
    return events


def invert_when_grace_starts(settled, started=None):
    started = time.monotonic() if started is None else started

    def poll():
        try:
            if time.monotonic() - started > 12.0:
                raise RuntimeError("extrude context grace never started for invert")
            session = next(iter(operators._SESSIONS.values()), None)
            if session is None or not session.confirmed_operator_pointer:
                return 0.01
            if tuple(STATE["window"].modal_operators):
                return 0.01
            with override():
                result = bpy.ops.mesh.select_all(action="INVERT")
            assert result == {"FINISHED"}, result
            wait_settled(settled)
        except BaseException:
            fail()
        return None

    bpy.app.timers.register(poll, first_interval=0.01)


def run_case(name, prepare, cursor_xyz, verify, *, drag=(0, 80), after_confirm=None):
    def start(next_case):
        try:
            print(f"YSE_EXTRUDE_CONTEXT_CASE={name}", flush=True)
            obj = build_mesh(name)
            bm = bmesh.from_edit_mesh(obj.data)
            prepare(bm, obj)
            bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
            live = bmesh.from_edit_mesh(obj.data)
            STATE["baseline"] = topology_counts(live)
            STATE["selected_coords"] = [coordinate_key(vertex.co) for vertex in live.verts if vertex.select]
            activate_extrude_region_tool()

            def settled():
                try:
                    bm2 = bmesh.from_edit_mesh(STATE["object"].data)
                    verify(bm2)
                    next_case()
                except BaseException:
                    fail()

            def after_route():
                events = tool_drag_events(cursor_xyz, drag=drag)
                if after_confirm == "invert":
                    send_events(events, lambda: None)
                    invert_when_grace_starts(settled)
                else:
                    send_events(events, lambda: wait_settled(settled))

            wait_for_toolbar_route(after_route)
        except BaseException:
            fail()

    return start


def verify_net_and_symmetric(bm, expected_net, *, undo=False):
    got = (
        len(bm.verts) - STATE["baseline"][0],
        len(bm.edges) - STATE["baseline"][1],
        len(bm.faces) - STATE["baseline"][2],
    )
    assert got == expected_net, f"net {got} != {expected_net}"
    assert_x_symmetric(bm)
    assert_layers_removed(bm)
    if undo:
        with override():
            undo_result = bpy.ops.ed.undo()
        assert undo_result == {"FINISHED"}, undo_result
        bm2 = bmesh.from_edit_mesh(STATE["object"].data)
        counts = topology_counts(bm2)
        assert counts == STATE["baseline"], f"one undo did not restore baseline: {counts} != {STATE['baseline']}"
        assert_x_symmetric(bm2)
        assert_layers_removed(bm2)


def verify_decline_native_kept(bm, expected_net=None):
    assert vertex_multiset(bm) != mirrored_multiset(bm), "declined extrude should keep an unmirrored native result"
    assert_layers_removed(bm)
    if expected_net is not None:
        got = (
            len(bm.verts) - STATE["baseline"][0],
            len(bm.edges) - STATE["baseline"][1],
            len(bm.faces) - STATE["baseline"][2],
        )
        assert got == expected_net, f"declined net {got} != native-only {expected_net}"


def assert_zero_offset_overlap(bm):
    live = Counter(coordinate_key(vertex.co) for vertex in bm.verts)
    selected = STATE.get("selected_coords") or []
    assert selected, "no selected vertex coordinates stored for zero-offset overlap"
    for key in selected:
        assert live[key] == 2, f"expected exactly one coincident duplicate at {key}, counts={dict(live)}"


def unique_extrude_context_record(object_name):
    records = [
        record
        for record in operators._HISTORY_RECORDS.values()
        if record.session.object_name == object_name and record.session.tool_kind == "EXTRUDE_CONTEXT"
    ]
    assert records, f"no EXTRUDE_CONTEXT history for {object_name!r}"
    sequence = max(record.sequence for record in records)
    matches = [record for record in records if record.sequence == sequence]
    assert len(matches) == 1, f"non-unique max sequence {sequence}: {matches}"
    return matches[0]


def verify_state1_native_only(bm, native_net, *, require_selection_changed=False):
    got = (
        len(bm.verts) - STATE["baseline"][0],
        len(bm.edges) - STATE["baseline"][1],
        len(bm.faces) - STATE["baseline"][2],
    )
    assert vertex_multiset(bm) != mirrored_multiset(bm), "grace-time decline must stay asymmetric"
    assert got == native_net, f"asymmetric result must be native-only {native_net}, got {got}"
    assert_layers_removed(bm)
    if require_selection_changed:
        record = unique_extrude_context_record(STATE["object"].name)
        assert record.session.prepare_disposition == "DECLINE", record.session
        assert record.session.extrude_freeze is None, record.session
        assert "selection changed" in record.session.prepare_disposition_reason, record.session


def verify_zero_offset_face(bm):
    verify_state1_native_only(bm, SINGLE_FACE_NATIVE_NET)
    assert_zero_offset_overlap(bm)


def prepare_single_face(bm, _obj):
    select_faces(bm, [(4, 1)])


def prepare_boundary_edge(bm, _obj):
    select_edges(bm, [((6, 1), (6, 2))])


def prepare_single_vertex(bm, _obj):
    select_verts(bm, [(4, 2)])


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
        cases = [
            run_case(
                "single_face",
                prepare_single_face,
                (2.5, 2.5, 0.0),
                lambda bm: verify_net_and_symmetric(bm, (8, 16, 8), undo=True),
            ),
            run_case(
                "boundary_edge",
                prepare_boundary_edge,
                (2.5, 2.5, 0.0),
                lambda bm: verify_net_and_symmetric(bm, (4, 6, 2)),
            ),
            run_case(
                "single_vertex",
                prepare_single_vertex,
                (2.5, 2.5, 0.0),
                lambda bm: verify_net_and_symmetric(bm, (2, 2, 0)),
            ),
            run_case(
                "zero_offset",
                prepare_single_face,
                (2.5, 2.5, 0.0),
                verify_zero_offset_face,
                drag=(0, 0),
            ),
            run_case(
                "invert_during_grace",
                prepare_single_face,
                (2.5, 2.5, 0.0),
                lambda bm: verify_state1_native_only(
                    bm,
                    SINGLE_FACE_NATIVE_NET,
                    require_selection_changed=True,
                ),
                after_confirm="invert",
            ),
        ]
        run_all(cases)
    except BaseException:
        fail()
    return None


bpy.app.timers.register(start_test, first_interval=0.3)
