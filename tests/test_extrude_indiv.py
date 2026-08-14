# SPDX-License-Identifier: GPL-3.0-or-later

"""GUI regression for Extrude Individual Faces (EXTRUDE_FACES_INDIV).

Run with Blender's real window/event loop::

    blender --factory-startup --enable-event-simulate \
        --python test_extrude_indiv.py
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

MARKER_OK = "YSE_EXTRUDE_INDIV_TEST_OK"
MARKER_FAILED = "YSE_EXTRUDE_INDIV_TEST_FAILED"
TOOL_ID = "builtin.extrude_individual"
TOOL_KEYMAP_NAME = "3D View Tool: Edit Mesh, Extrude Individual"
NX, NY = 6, 4
PRECISION = 5
ADJACENT_NATIVE = (8, 16, 8)
ADJACENT_MIRROR = (16, 32, 16)
SINGLE_NATIVE = (4, 8, 4)
SINGLE_MIRROR = (8, 16, 8)
STATE = {}


def fail(message=""):
    if message:
        print(f"YSE_EXTRUDE_INDIV_ERROR={message}", flush=True)
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


def configure_view(area, *, gizmos=False):
    space = area.spaces.active
    space.show_gizmo = bool(gizmos)
    space.show_gizmo_tool = bool(gizmos)
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


def build_mesh(name):
    for old in tuple(bpy.data.objects):
        bpy.data.objects.remove(old, do_unlink=True)
    for old_mesh in tuple(bpy.data.meshes):
        if old_mesh.users == 0:
            bpy.data.meshes.remove(old_mesh)
    mesh = bpy.data.meshes.new(f"YSE_ExtrudeIndivMesh_{name}")
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
    obj = bpy.data.objects.new(f"YSE_ExtrudeIndivObject_{name}", mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.use_mesh_mirror_x = True
    obj.use_mesh_mirror_y = False
    obj.use_mesh_mirror_z = False
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    with override():
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.context.tool_settings.mesh_select_mode = (True, False, False)
        bpy.ops.ed.undo_push(message=f"YSE extrude indiv baseline {name}")
    STATE["object"] = obj
    return obj


def grid_vert(bm, i, j):
    x, y = grid_xy(i, j)
    for vertex in bm.verts:
        if abs(vertex.co.x - x) < 1e-4 and abs(vertex.co.y - y) < 1e-4 and abs(vertex.co.z) < 1e-4:
            return vertex
    raise AssertionError(f"grid vert {i},{j} not found")


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


def activate_extrude_individual_tool():
    # Measured 4.2/5.2: wm.tool_set_by_id("builtin.extrude_individual") FINISHED.
    # tool.keymap is None; intercept keymap is TOOL_KEYMAP_NAME from the profile.
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
                        f"extrude indiv flow never settled; "
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
                    f"Extrude Individual LEFTMOUSE CLICK_DRAG intercept was not registered; "
                    f"items={[(km.name, item.type, item.value) for km, item in keymaps._REGISTERED_ITEMS]}"
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


def unique_indiv_record(object_name):
    records = [
        record
        for record in operators._HISTORY_RECORDS.values()
        if record.session.object_name == object_name and record.session.tool_kind == "EXTRUDE_FACES_INDIV"
    ]
    assert records, f"no EXTRUDE_FACES_INDIV history for {object_name!r}"
    sequence = max(record.sequence for record in records)
    matches = [record for record in records if record.sequence == sequence]
    assert len(matches) == 1, f"non-unique max sequence {sequence}: {matches}"
    return matches[0]


def snapshot_vid_at(snapshot, i, j):
    x, y = grid_xy(i, j)
    matches = [
        vertex_id
        for vertex_id, coord in snapshot.vertex_preop
        if abs(coord.x - x) < 1e-4 and abs(coord.y - y) < 1e-4 and abs(coord.z) < 1e-4
    ]
    assert len(matches) == 1, f"grid ({i},{j}) vids={matches}"
    return matches[0]


def assert_options_captured(record):
    assert record.session.extrude_options_captured is True, record.session
    options = record.session.extrude_options
    assert options is not None, "FACES_INDIV record has no extrude_options"
    props = dict(options.transform_props)
    assert "value" in props, props
    assert isinstance(props["value"], (int, float)), props
    assert math.isfinite(float(props["value"])), props


def assert_adjacent_d_freeze(record):
    snapshot = record.session.extrude
    freeze = record.session.extrude_freeze
    assert snapshot is not None, "missing extrude snapshot"
    assert freeze, "missing extrude freeze table"
    assert any(entry.entity_class == "d" for entry in freeze), freeze
    for i, j in ((5, 1), (5, 2)):
        vertex_id = snapshot_vid_at(snapshot, i, j)
        rows = [entry for entry in freeze if entry.vertex_id == vertex_id]
        assert rows, f"no freeze rows for shared grid ({i},{j}) vid={vertex_id}"
        assert all(entry.entity_class == "d" for entry in rows), rows
        signatures = [entry.source_face_signature for entry in rows]
        assert all(signature for signature in signatures), signatures
        assert len(set(signatures)) == len(signatures), signatures


def assert_all_class_b(record):
    freeze = record.session.extrude_freeze
    assert freeze, "missing extrude freeze table"
    assert all(entry.entity_class == "b" for entry in freeze), freeze


def zero_offset_expected_counts(cells):
    counts: Counter[tuple[float, float, float]] = Counter()
    for i, j in cells:
        for corner in ((i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)):
            x, y = grid_xy(*corner)
            counts[coordinate_key((x, y, 0.0))] += 1
    return {key: 1 + copies for key, copies in counts.items()}


def assert_zero_offset_overlap(bm, cells):
    live = vertex_multiset(bm)
    expected = zero_offset_expected_counts(cells)
    for key, count in expected.items():
        assert live[key] == count, f"expected {count} verts at {key}, live={dict(live)}"


def run_tool_case(name, prepare, cursor_xyz, verify, *, drag=(0, 80)):
    def start(next_case):
        try:
            print(f"YSE_EXTRUDE_INDIV_CASE={name}", flush=True)
            obj = build_mesh(name)
            bm = bmesh.from_edit_mesh(obj.data)
            prepare(bm, obj)
            bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
            live = bmesh.from_edit_mesh(obj.data)
            STATE["baseline"] = topology_counts(live)
            STATE["selected_coords"] = [coordinate_key(vertex.co) for vertex in live.verts if vertex.select]
            activate_extrude_individual_tool()

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


def invoke_wrapper_and_drag(cursor_xyz, verify, *, drag=(0, 80)):
    def start(next_case):
        try:
            print("YSE_EXTRUDE_INDIV_CASE=menu_wrapper", flush=True)
            obj = build_mesh("menu_wrapper")
            bm = bmesh.from_edit_mesh(obj.data)
            select_faces(bm, [(4, 1)])
            bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
            STATE["baseline"] = topology_counts(bmesh.from_edit_mesh(obj.data))

            x, y = window_coordinate(cursor_xyz)
            STATE["window"].event_simulate(type="MOUSEMOVE", value="NOTHING", x=x, y=y)
            with override():
                result = bpy.ops.mesh.ydd_symmetric_edit_extrude_individual_faces("INVOKE_DEFAULT")
            assert result == {"FINISHED"}, result
            assert extrude_menu.WRAPPER_INDIV == "mesh.ydd_symmetric_edit_extrude_individual_faces"

            def settled():
                try:
                    bm2 = bmesh.from_edit_mesh(STATE["object"].data)
                    verify(bm2)
                    next_case()
                except BaseException:
                    fail()

            def after_modal():
                send_events(drag_confirm_events(cursor_xyz, drag=drag), lambda: wait_settled(settled))

            def wait_modal(started=None):
                started = time.monotonic() if started is None else started

                def poll():
                    try:
                        if tuple(STATE["window"].modal_operators):
                            after_modal()
                            return None
                        if time.monotonic() - started > 4.0:
                            raise RuntimeError(
                                f"native dispatcher never went modal after WRAPPER_INDIV; "
                                f"sessions={list(operators._SESSIONS)}"
                            )
                        return 0.05
                    except BaseException:
                        fail()
                    return None

                bpy.app.timers.register(poll, first_interval=0.05)

            wait_modal()
        except BaseException:
            fail()

    return start


def verify_net_and_symmetric(bm, expected_net, *, undo=False, check_options=False, check_d=False, check_b=False):
    got = (
        len(bm.verts) - STATE["baseline"][0],
        len(bm.edges) - STATE["baseline"][1],
        len(bm.faces) - STATE["baseline"][2],
    )
    assert got == expected_net, f"net {got} != {expected_net}"
    assert_x_symmetric(bm)
    assert_layers_removed(bm)
    record = unique_indiv_record(STATE["object"].name)
    if check_options:
        assert_options_captured(record)
    if check_d:
        assert_adjacent_d_freeze(record)
    if check_b:
        assert_all_class_b(record)
    if undo:
        with override():
            undo_result = bpy.ops.ed.undo()
        assert undo_result == {"FINISHED"}, undo_result
        bm2 = bmesh.from_edit_mesh(STATE["object"].data)
        counts = topology_counts(bm2)
        assert counts == STATE["baseline"], f"one undo did not restore baseline: {counts} != {STATE['baseline']}"
        assert_x_symmetric(bm2)
        assert_layers_removed(bm2)


def verify_zero_offset(bm, cells, native_net):
    got = (
        len(bm.verts) - STATE["baseline"][0],
        len(bm.edges) - STATE["baseline"][1],
        len(bm.faces) - STATE["baseline"][2],
    )
    assert vertex_multiset(bm) != mirrored_multiset(bm), "declined extrude should keep an unmirrored native result"
    assert got == native_net, f"declined net {got} != native-only {native_net}"
    assert_layers_removed(bm)
    assert_zero_offset_overlap(bm, cells)


def prepare_adjacent(bm, _obj):
    select_faces(bm, [(4, 1), (5, 1)])


def prepare_distant(bm, _obj):
    select_faces(bm, [(4, 0), (5, 2)])


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
        cases = [
            run_tool_case(
                "adjacent_2",
                prepare_adjacent,
                (2.0, -0.5, 0.0),
                lambda bm: verify_net_and_symmetric(
                    bm,
                    ADJACENT_MIRROR,
                    undo=True,
                    check_options=True,
                    check_d=True,
                ),
            ),
            run_tool_case(
                "distant_2",
                prepare_distant,
                (2.0, 0.0, 0.0),
                lambda bm: verify_net_and_symmetric(bm, ADJACENT_MIRROR, check_b=True),
            ),
            run_tool_case(
                "single_face",
                prepare_single,
                (1.5, -0.5, 0.0),
                lambda bm: verify_net_and_symmetric(bm, SINGLE_MIRROR, check_options=True),
            ),
            invoke_wrapper_and_drag(
                (1.5, -0.5, 0.0),
                lambda bm: verify_net_and_symmetric(bm, SINGLE_MIRROR, check_options=True),
            ),
            run_tool_case(
                "zero_offset",
                prepare_adjacent,
                (2.0, -0.5, 0.0),
                lambda bm: verify_zero_offset(bm, [(4, 1), (5, 1)], ADJACENT_NATIVE),
                drag=(0, 0),
            ),
        ]
        run_all(cases)
    except BaseException:
        fail()
    return None


bpy.app.timers.register(start_test, first_interval=0.3)
