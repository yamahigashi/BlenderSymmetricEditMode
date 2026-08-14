# SPDX-License-Identifier: GPL-3.0-or-later

"""GUI regression for Extrude Edges / Vertices (Stage 3c menu wrappers).

Run with Blender's real window/event loop::

    blender --factory-startup --enable-event-simulate \
        --python test_extrude_edges_verts.py
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
from ydd_symmetric_edit import extrude_menu, layer_names, operators  # noqa: E402

MARKER_OK = "YSE_EXTRUDE_EDGEVERT_TEST_OK"
MARKER_FAILED = "YSE_EXTRUDE_EDGEVERT_TEST_FAILED"
NX, NY = 6, 4
PRECISION = 5
# region-formula census (F_r empty) × mirror: native nets doubled
EDGE_NATIVE = (2, 3, 1)
EDGE_MIRROR = (4, 6, 2)
EDGE_PATH_NATIVE = (3, 5, 2)
EDGE_PATH_MIRROR = (6, 10, 4)
VERT_NATIVE = (1, 1, 0)
VERT_MIRROR = (2, 2, 0)
TWO_VERT_NATIVE = (2, 2, 0)
TWO_VERT_MIRROR = (4, 4, 0)
# Measured native edges_indiv 3-spoke fan (empty F_r, all vids 1:1 class b).
FAN_NATIVE = (4, 7, 3)
FAN_MIRROR = (FAN_NATIVE[0] * 2, FAN_NATIVE[1] * 2, FAN_NATIVE[2] * 2)
# Measured native verts_indiv of two adjacent verts: rails only, no cap edge.
ADJACENT_VERT_NATIVE = (2, 2, 0)
ADJACENT_VERT_MIRROR = (4, 4, 0)
# Measured native edges_indiv of one +X face loop (F_r non-empty → census undefined).
LOOP_NATIVE = (4, 8, 4)
# Measured native edges_indiv of one +X face plus an extra edge (F_r non-empty).
MIXED_NATIVE = (6, 11, 5)
# Measured native verts_indiv of both wire endpoints.
WIRE_NATIVE = (2, 2, 0)
WIRE_MIRROR = (4, 4, 0)
CENSUS_UNDEFINED_TOKENS = ("wire", "non-manifold", "census", "undefined")
STATE = {}


def fail(message=""):
    if message:
        print(f"YSE_EXTRUDE_EDGEVERT_ERROR={message}", flush=True)
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


def build_mesh(name):
    for old in tuple(bpy.data.objects):
        bpy.data.objects.remove(old, do_unlink=True)
    for old_mesh in tuple(bpy.data.meshes):
        if old_mesh.users == 0:
            bpy.data.meshes.remove(old_mesh)
    mesh = bpy.data.meshes.new(f"YSE_ExtrudeEdgeVertMesh_{name}")
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
    obj = bpy.data.objects.new(f"YSE_ExtrudeEdgeVertObject_{name}", mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.use_mesh_mirror_x = True
    obj.use_mesh_mirror_y = False
    obj.use_mesh_mirror_z = False
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    with override():
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.context.tool_settings.mesh_select_mode = (True, False, False)
        bpy.ops.ed.undo_push(message=f"YSE extrude edgevert baseline {name}")
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
                        f"extrude edge/vert flow never settled; "
                        f"modal={[op.bl_idname for op in STATE['window'].modal_operators]} "
                        f"sessions={list(operators._SESSIONS)}"
                    )
                return 0.1
            bpy.app.timers.register(done, first_interval=0.25)
        except BaseException:
            fail()
        return None

    bpy.app.timers.register(poll, first_interval=0.1)


def drag_confirm_events(cursor_xyz, drag=(0, 80)):
    x, y = window_coordinate(cursor_xyz)
    tx, ty = x + drag[0], y + drag[1]
    return [
        {"type": "MOUSEMOVE", "value": "NOTHING", "x": x, "y": y},
        {"type": "MOUSEMOVE", "value": "NOTHING", "x": tx, "y": ty},
        {"type": "LEFTMOUSE", "value": "PRESS", "x": tx, "y": ty},
        {"type": "LEFTMOUSE", "value": "RELEASE", "x": tx, "y": ty},
    ]


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


def assert_all_class_b(record):
    freeze = record.session.extrude_freeze
    assert freeze, "missing extrude freeze table"
    assert all(entry.entity_class == "b" for entry in freeze), freeze


def assert_options_captured(record):
    assert record.session.extrude_options_captured is True, record.session
    options = record.session.extrude_options
    assert options is not None, f"{record.session.tool_kind} record has no extrude_options"
    props = dict(options.transform_props)
    value = props.get("value")
    assert isinstance(value, (list, tuple)) and len(value) == 3, props
    assert all(isinstance(v, (int, float)) and math.isfinite(float(v)) for v in value), props


def verify_net_and_symmetric(bm, expected_net, tool_kind, *, undo=False, check_b=True):
    got = (
        len(bm.verts) - STATE["baseline"][0],
        len(bm.edges) - STATE["baseline"][1],
        len(bm.faces) - STATE["baseline"][2],
    )
    assert got == expected_net, f"net {got} != {expected_net}"
    assert_x_symmetric(bm)
    assert_layers_removed(bm)
    record = unique_record(STATE["object"].name, tool_kind)
    assert_options_captured(record)
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


def assert_declined(bm, tool_kind, native_net, reason_substr):
    got = (
        len(bm.verts) - STATE["baseline"][0],
        len(bm.edges) - STATE["baseline"][1],
        len(bm.faces) - STATE["baseline"][2],
    )
    assert got == native_net, f"declined net {got} != native-only {native_net}"
    if got != (0, 0, 0):
        assert vertex_multiset(bm) != mirrored_multiset(bm), "declined extrude should keep an unmirrored native result"
    record = unique_record(STATE["object"].name, tool_kind)
    assert record.session.prepare_disposition == "DECLINE", record.session
    reason = record.session.prepare_disposition_reason or ""
    tokens = (reason_substr,) if isinstance(reason_substr, str) else tuple(reason_substr)
    assert tokens, "decline reason tokens must not be empty"
    assert any(token in reason for token in tokens), reason
    freeze = getattr(record.session, "extrude_freeze", None)
    assert freeze in (None, ()), freeze
    assert_layers_removed(bm)
    print(f"YSE_EXTRUDE_EDGEVERT_DECLINE_REASON={reason!r}", flush=True)
    return record


def verify_zero_offset(bm, native_net, tool_kind):
    assert_declined(bm, tool_kind, native_net, "zero-offset")


def invoke_wrapper_and_drag(name, op_idname, tool_kind, prepare, cursor_xyz, verify, *, drag=(0, 80)):
    def start(next_case):
        try:
            print(f"YSE_EXTRUDE_EDGEVERT_CASE={name}", flush=True)
            obj = build_mesh(name)
            bm = bmesh.from_edit_mesh(obj.data)
            prepare(bm, obj)
            bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
            STATE["baseline"] = topology_counts(bmesh.from_edit_mesh(obj.data))

            x, y = window_coordinate(cursor_xyz)
            STATE["window"].event_simulate(type="MOUSEMOVE", value="NOTHING", x=x, y=y)
            namespace, op_name = op_idname.split(".", 1)
            with override():
                result = getattr(getattr(bpy.ops, namespace), op_name)("INVOKE_DEFAULT")
            assert result == {"FINISHED"}, result

            def settled():
                try:
                    bm2 = bmesh.from_edit_mesh(STATE["object"].data)
                    verify(bm2, tool_kind)
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
                                f"native dispatcher never went modal after {op_idname}; "
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


def prepare_boundary_edge(bm, _obj):
    select_edges(bm, [((4, 1), (4, 2))])


def prepare_edge_path(bm, _obj):
    select_edges(bm, [((4, 1), (4, 2)), ((4, 2), (4, 3))])


def prepare_single_vertex(bm, _obj):
    select_verts(bm, [(4, 2)])


def prepare_two_vertices(bm, _obj):
    select_verts(bm, [(4, 1), (4, 3)])


def prepare_fan(bm, _obj):
    select_edges(bm, [((5, 0), (4, 0)), ((5, 0), (6, 0)), ((5, 0), (5, 1))])


def prepare_adjacent_vertices(bm, _obj):
    select_verts(bm, [(4, 1), (4, 2)])


def prepare_embedded_wire(bm, obj):
    attach_p = grid_vert(bm, 4, 2)
    attach_n = grid_vert(bm, 2, 2)
    hang_p = bm.verts.new((1.5, 0.5, 0.5))
    hang_n = bm.verts.new((-1.5, 0.5, 0.5))
    bm.edges.new((attach_p, hang_p))
    bm.edges.new((attach_n, hang_n))
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.verts.index_update()
    bm.edges.index_update()
    bpy.context.tool_settings.mesh_select_mode = (True, False, False)
    clear_selection(bm)
    attach_p.select = True
    hang_p.select = True
    bm.select_flush_mode()
    bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=True)


def prepare_closed_loop(bm, _obj):
    select_edges(bm, [((4, 1), (5, 1)), ((5, 1), (5, 2)), ((5, 2), (4, 2)), ((4, 2), (4, 1))])


def prepare_mixed_face_edge(bm, _obj):
    bpy.context.tool_settings.mesh_select_mode = (False, True, True)
    clear_selection(bm)
    grid_face(bm, 4, 1).select = True
    grid_edge(bm, (4, 3), (5, 3)).select = True
    bm.select_flush_mode()


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
            invoke_wrapper_and_drag(
                "edges_boundary",
                extrude_menu.WRAPPER_EDGES,
                "EXTRUDE_EDGES_INDIV",
                prepare_boundary_edge,
                (1.0, -0.5, 0.0),
                lambda bm, kind: verify_net_and_symmetric(bm, EDGE_MIRROR, kind, undo=True),
            ),
            invoke_wrapper_and_drag(
                "edges_path",
                extrude_menu.WRAPPER_EDGES,
                "EXTRUDE_EDGES_INDIV",
                prepare_edge_path,
                (1.0, 0.0, 0.0),
                lambda bm, kind: verify_net_and_symmetric(bm, EDGE_PATH_MIRROR, kind),
            ),
            invoke_wrapper_and_drag(
                "verts_single",
                extrude_menu.WRAPPER_VERTS,
                "EXTRUDE_VERTS_INDIV",
                prepare_single_vertex,
                (1.0, 0.0, 0.0),
                lambda bm, kind: verify_net_and_symmetric(bm, VERT_MIRROR, kind, undo=True),
            ),
            invoke_wrapper_and_drag(
                "verts_two",
                extrude_menu.WRAPPER_VERTS,
                "EXTRUDE_VERTS_INDIV",
                prepare_two_vertices,
                (1.0, 0.0, 0.0),
                lambda bm, kind: verify_net_and_symmetric(bm, TWO_VERT_MIRROR, kind),
            ),
            invoke_wrapper_and_drag(
                "edges_zero_offset",
                extrude_menu.WRAPPER_EDGES,
                "EXTRUDE_EDGES_INDIV",
                prepare_boundary_edge,
                (1.0, -0.5, 0.0),
                lambda bm, kind: verify_zero_offset(bm, EDGE_NATIVE, kind),
                drag=(0, 0),
            ),
            invoke_wrapper_and_drag(
                "verts_zero_offset",
                extrude_menu.WRAPPER_VERTS,
                "EXTRUDE_VERTS_INDIV",
                prepare_single_vertex,
                (1.0, 0.0, 0.0),
                lambda bm, kind: verify_zero_offset(bm, VERT_NATIVE, kind),
                drag=(0, 0),
            ),
            invoke_wrapper_and_drag(
                "edges_fan",
                extrude_menu.WRAPPER_EDGES,
                "EXTRUDE_EDGES_INDIV",
                prepare_fan,
                (2.0, -2.0, 0.0),
                lambda bm, kind: verify_net_and_symmetric(bm, FAN_MIRROR, kind),
            ),
            invoke_wrapper_and_drag(
                "verts_adjacent",
                extrude_menu.WRAPPER_VERTS,
                "EXTRUDE_VERTS_INDIV",
                prepare_adjacent_vertices,
                (1.0, -0.5, 0.0),
                lambda bm, kind: verify_net_and_symmetric(bm, ADJACENT_VERT_MIRROR, kind),
            ),
            invoke_wrapper_and_drag(
                "verts_embedded_wire",
                extrude_menu.WRAPPER_VERTS,
                "EXTRUDE_VERTS_INDIV",
                prepare_embedded_wire,
                (1.0, 0.0, 0.0),
                lambda bm, kind: verify_net_and_symmetric(bm, WIRE_MIRROR, kind),
            ),
            invoke_wrapper_and_drag(
                "edges_closed_loop",
                extrude_menu.WRAPPER_EDGES,
                "EXTRUDE_EDGES_INDIV",
                prepare_closed_loop,
                (1.5, -0.5, 0.0),
                lambda bm, kind: assert_declined(bm, kind, LOOP_NATIVE, CENSUS_UNDEFINED_TOKENS),
            ),
            invoke_wrapper_and_drag(
                "edges_mixed_face_edge",
                extrude_menu.WRAPPER_EDGES,
                "EXTRUDE_EDGES_INDIV",
                prepare_mixed_face_edge,
                (1.5, -0.5, 0.0),
                lambda bm, kind: assert_declined(bm, kind, MIXED_NATIVE, CENSUS_UNDEFINED_TOKENS),
            ),
        ]
        run_all(cases)
    except BaseException:
        fail()
    return None


bpy.app.timers.register(start_test, first_interval=0.3)
