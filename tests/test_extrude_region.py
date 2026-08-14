# SPDX-License-Identifier: GPL-3.0-or-later

"""GUI regression for symmetric Extrude (E / EXTRUDE_NORMAL).

Run with Blender's real window/event loop::

    blender --factory-startup --enable-event-simulate \
        --python test_extrude_region.py
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
from ydd_symmetric_edit import layer_names, operators  # noqa: E402

MARKER_OK = "YSE_EXTRUDE_REGION_TEST_OK"
MARKER_FAILED = "YSE_EXTRUDE_REGION_TEST_FAILED"
NX, NY = 6, 4
PRECISION = 5
STATE = {}


def fail(message=""):
    if message:
        print(f"YSE_EXTRUDE_REGION_ERROR={message}", flush=True)
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
    region_3d = area.spaces.active.region_3d
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
    mesh = bpy.data.meshes.new(f"YSE_ExtrudeMesh_{name}")
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
    obj = bpy.data.objects.new(f"YSE_ExtrudeObject_{name}", mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.use_mesh_mirror_x = True
    obj.use_mesh_mirror_y = False
    obj.use_mesh_mirror_z = False
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    with override():
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.context.tool_settings.mesh_select_mode = (True, False, False)
        bpy.ops.ed.undo_push(message=f"YSE extrude baseline {name}")
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
    assert bm.verts.layers.int.get(layer_names.VERT_SESSION_ID_LAYER) is None, "session vertex layer leaked"
    assert bm.edges.layers.int.get(layer_names.EDGE_ORIGINAL_LAYER) is None, "edge layer leaked"
    assert bm.faces.layers.int.get(layer_names.FACE_ID_LAYER) is None, "face layer leaked"


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
                        f"extrude flow never settled; modal={[op.bl_idname for op in STATE['window'].modal_operators]} "
                        f"sessions={list(operators._SESSIONS)}"
                    )
                return 0.1
            bpy.app.timers.register(done, first_interval=0.25)
        except BaseException:
            fail()
        return None

    bpy.app.timers.register(poll, first_interval=0.1)


def wait_session_gone(done, started=None):
    started = time.monotonic() if started is None else started

    def poll():
        try:
            if operators._SESSIONS:
                if time.monotonic() - started > 12.0:
                    raise RuntimeError(f"extrude session never finished; sessions={list(operators._SESSIONS)}")
                return 0.1
            bpy.app.timers.register(done, first_interval=0.05)
        except BaseException:
            fail()
        return None

    bpy.app.timers.register(poll, first_interval=0.05)


def extrude_events(cursor_xyz, drag=(0, 80), confirm="LMB"):
    x, y = window_coordinate(cursor_xyz)
    events = [
        {"type": "MOUSEMOVE", "value": "NOTHING", "x": x, "y": y},
        {"type": "E", "value": "PRESS", "x": x, "y": y},
        {"type": "E", "value": "RELEASE", "x": x, "y": y},
    ]
    tx, ty = x + drag[0], y + drag[1]
    if drag != (0, 0):
        events.append({"type": "MOUSEMOVE", "value": "NOTHING", "x": tx, "y": ty})
    if confirm == "LMB":
        events.append({"type": "LEFTMOUSE", "value": "PRESS", "x": tx, "y": ty})
        events.append({"type": "LEFTMOUSE", "value": "RELEASE", "x": tx, "y": ty})
    elif confirm == "ESC":
        events.append({"type": "ESC", "value": "PRESS", "x": tx, "y": ty})
        events.append({"type": "ESC", "value": "RELEASE", "x": tx, "y": ty})
    return events


def run_case(name, prepare, cursor_xyz, verify, *, drag=(0, 80), confirm="LMB", after_confirm=None):
    def start(next_case):
        try:
            print(f"YSE_EXTRUDE_CASE={name}", flush=True)
            obj = build_mesh(name)
            bm = bmesh.from_edit_mesh(obj.data)
            prepare(bm, obj)
            bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
            STATE["baseline"] = topology_counts(bmesh.from_edit_mesh(obj.data))

            def settled():
                try:
                    bm2 = bmesh.from_edit_mesh(STATE["object"].data)
                    verify(bm2)
                    next_case()
                except BaseException:
                    fail()

            events = extrude_events(cursor_xyz, drag=drag, confirm=confirm)
            x, y = window_coordinate(cursor_xyz)
            if after_confirm == "invert":

                def invert_when_grace_starts(started=None):
                    started = time.monotonic() if started is None else started

                    def poll():
                        try:
                            if time.monotonic() - started > 12.0:
                                raise RuntimeError("extrude grace never started for invert")
                            session = next(iter(operators._SESSIONS.values()), None)
                            if session is None or not session.confirmed_operator_pointer:
                                return 0.01
                            if tuple(STATE["window"].modal_operators):
                                return 0.01
                            with override():
                                bpy.ops.mesh.select_all(action="INVERT")
                            wait_settled(settled)
                        except BaseException:
                            fail()
                        return None

                    bpy.app.timers.register(poll, first_interval=0.01)

                send_events(events, lambda: None)
                invert_when_grace_starts()
            elif after_confirm == "grab":
                tx, ty = x + drag[0], y + drag[1]
                # Insert G immediately after LMB PRESS so it lands in watcher grace.
                trimmed = []
                inserted = False
                for event in events:
                    trimmed.append(event)
                    if not inserted and event.get("type") == "LEFTMOUSE" and event.get("value") == "PRESS":
                        trimmed.append({"type": "G", "value": "PRESS", "x": tx, "y": ty})
                        inserted = True
                send_events(
                    trimmed,
                    lambda: wait_session_gone(
                        lambda: send_events(
                            [
                                {"type": "ESC", "value": "PRESS", "x": tx, "y": ty},
                                {"type": "ESC", "value": "RELEASE", "x": tx, "y": ty},
                            ],
                            lambda: wait_settled(settled),
                            interval=0.05,
                        )
                    ),
                    interval=0.015,
                )
            else:
                send_events(events, lambda: wait_settled(settled))
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


def verify_state1_native_only(bm, native_net):
    got = (
        len(bm.verts) - STATE["baseline"][0],
        len(bm.edges) - STATE["baseline"][1],
        len(bm.faces) - STATE["baseline"][2],
    )
    assert vertex_multiset(bm) != mirrored_multiset(bm), "grace-time decline must stay asymmetric"
    assert got == native_net, f"asymmetric result must be native-only {native_net}, got {got}"
    assert_layers_removed(bm)
    reports = list(operators._FINISH_REPORTS)
    warnings = [message for kind, message in reports if kind == "WARNING"]
    infos = [message for kind, message in reports if kind == "INFO"]
    assert warnings, reports
    assert any("native kept; mirror manually or undo" in message for message in warnings), reports
    assert not any("Mirrored Extrude" in message for message in infos), reports


def _uv_tuple(face, uv_layer):
    return tuple(tuple(round(float(loop[uv_layer].uv[index]), 5) for index in (0, 1)) for loop in face.loops)


def verify_uv_material_mirror(bm):
    verify_net_and_symmetric(bm, (8, 16, 8))
    uv_layer = bm.loops.layers.uv.active
    assert uv_layer is not None, "UV layer missing after mirrored extrude"
    source_uv = frozenset({(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)})
    caps = [face for face in bm.faces if face.material_index == 1]
    assert len(caps) == 2, f"expected 2 material-1 faces, got {len(caps)}"
    for face in caps:
        assert frozenset(_uv_tuple(face, uv_layer)) == source_uv, _uv_tuple(face, uv_layer)
    positive = [face for face in caps if face.calc_center_median().x > 0]
    negative = [face for face in caps if face.calc_center_median().x < 0]
    assert len(positive) == 1 and len(negative) == 1
    source_uvs = _uv_tuple(positive[0], uv_layer)
    mirror_uvs = _uv_tuple(negative[0], uv_layer)
    assert mirror_uvs == tuple(reversed(source_uvs)), (mirror_uvs, source_uvs)

    new_faces = [face for face in bm.faces if any(abs(vertex.co.z) > 1e-4 for vertex in face.verts)]
    grouped: dict[tuple[int, frozenset], list] = {}
    for face in new_faces:
        grouped.setdefault((face.material_index, frozenset(_uv_tuple(face, uv_layer))), []).append(face)
    for key, group in grouped.items():
        positive_faces = [face for face in group if face.calc_center_median().x > 0]
        negative_faces = [face for face in group if face.calc_center_median().x < 0]
        assert len(positive_faces) == len(negative_faces), f"unpaired new faces for {key}"
        remaining = list(negative_faces)
        for source_face in positive_faces:
            expected = tuple(reversed(_uv_tuple(source_face, uv_layer)))
            match = next((face for face in remaining if _uv_tuple(face, uv_layer) == expected), None)
            assert match is not None, f"mirror loop UVs are not reversed source for {key}"
            remaining.remove(match)


def region_l_mirrored_net():
    vertex_count, edge_count, face_count = 8, 10, 3
    deleted_edges, deleted_verts = 2, 0
    created = (
        vertex_count,
        (vertex_count - deleted_verts) + edge_count,
        (edge_count - deleted_edges) + face_count,
    )
    deleted = (deleted_verts, deleted_edges, face_count)
    native = (created[0] - deleted[0], created[1] - deleted[1], created[2] - deleted[2])
    mirrored = (native[0] * 2, native[1] * 2, native[2] * 2)
    assert created == (8, 18, 11), created
    assert deleted == (0, 2, 3), deleted
    assert native == (8, 16, 8), native
    assert mirrored == (16, 32, 16), mirrored
    return mirrored


def prepare_single_face(bm, _obj):
    select_faces(bm, [(4, 1)])


def prepare_region_2x2(bm, _obj):
    select_faces(bm, [(4, 1), (5, 1), (4, 2), (5, 2)])


def prepare_boundary_edge(bm, _obj):
    select_edges(bm, [((4, 1), (4, 2))])


def prepare_single_vertex(bm, _obj):
    select_verts(bm, [(4, 2)])


def prepare_g4_on_plane(bm, _obj):
    select_faces(bm, [(3, 1)])


def prepare_g5_asymmetric(bm, _obj):
    counterpart = grid_face(bm, 1, 1)
    bmesh.ops.delete(bm, geom=[counterpart], context="FACES")
    bm.faces.ensure_lookup_table()
    select_faces(bm, [(4, 1)])


def prepare_g6_hidden(bm, _obj):
    grid_face(bm, 1, 1).hide = True
    select_faces(bm, [(4, 1)])


def prepare_edge_path(bm, _obj):
    select_edges(bm, [((4, 1), (4, 2)), ((4, 2), (4, 3))])


def prepare_region_l(bm, _obj):
    select_faces(bm, [(4, 1), (5, 1), (4, 2)])


def prepare_uv_material(bm, obj):
    mesh = obj.data
    while len(mesh.materials) < 2:
        material = bpy.data.materials.new(f"YSE_ExtrudeMat{len(mesh.materials)}")
        mesh.materials.append(material)
    uv_layer = bm.loops.layers.uv.active
    if uv_layer is None:
        uv_layer = bm.loops.layers.uv.new("UVMap")
    source = grid_face(bm, 4, 1)
    source.material_index = 1
    for loop, uv in zip(source.loops, ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)), strict=True):
        loop[uv_layer].uv = uv
    adjacent = {
        (3, 1): ((0.2, 0.2), (0.3, 0.2), (0.3, 0.3), (0.2, 0.3)),
        (5, 1): ((0.4, 0.4), (0.5, 0.4), (0.5, 0.5), (0.4, 0.5)),
        (4, 0): ((0.6, 0.1), (0.7, 0.1), (0.7, 0.2), (0.6, 0.2)),
        (4, 2): ((0.1, 0.6), (0.2, 0.6), (0.2, 0.7), (0.1, 0.7)),
    }
    for cell, pattern in adjacent.items():
        face = grid_face(bm, *cell)
        face.material_index = 0
        for loop, uv in zip(face.loops, pattern, strict=True):
            loop[uv_layer].uv = uv
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
            run_case(
                "single_face",
                prepare_single_face,
                (1.5, -0.5, 0.0),
                lambda bm: verify_net_and_symmetric(bm, (8, 16, 8), undo=True),
            ),
            run_case(
                "region_2x2",
                prepare_region_2x2,
                (2.0, 0.0, 0.0),
                lambda bm: verify_net_and_symmetric(bm, (16, 32, 16)),
            ),
            run_case(
                "boundary_edge",
                prepare_boundary_edge,
                (1.0, -0.5, 0.0),
                lambda bm: verify_net_and_symmetric(bm, (4, 6, 2)),
            ),
            run_case(
                "single_vertex",
                prepare_single_vertex,
                (1.0, 0.0, 0.0),
                lambda bm: verify_net_and_symmetric(bm, (2, 2, 0)),
            ),
            run_case(
                "zero_offset",
                prepare_single_face,
                (1.5, -0.5, 0.0),
                verify_decline_native_kept,
                drag=(0, 0),
            ),
            run_case(
                "esc",
                prepare_single_face,
                (1.5, -0.5, 0.0),
                verify_decline_native_kept,
                confirm="ESC",
            ),
            run_case(
                "g4_on_plane",
                prepare_g4_on_plane,
                (0.5, -0.5, 0.0),
                lambda bm: verify_decline_native_kept(bm, (4, 8, 4)),
            ),
            run_case(
                "g5_asymmetric",
                prepare_g5_asymmetric,
                (1.5, -0.5, 0.0),
                lambda bm: verify_decline_native_kept(bm, (4, 8, 4)),
            ),
            run_case(
                "g6_hidden",
                prepare_g6_hidden,
                (1.5, -0.5, 0.0),
                lambda bm: verify_decline_native_kept(bm, (4, 8, 4)),
            ),
            run_case(
                "invert_during_grace",
                prepare_single_face,
                (1.5, -0.5, 0.0),
                lambda bm: verify_state1_native_only(bm, (4, 8, 4)),
                after_confirm="invert",
            ),
            run_case(
                "grab_during_grace",
                prepare_single_face,
                (1.5, -0.5, 0.0),
                lambda bm: verify_state1_native_only(bm, (4, 8, 4)),
                after_confirm="grab",
            ),
            run_case(
                "uv_material",
                prepare_uv_material,
                (1.5, -0.5, 0.0),
                verify_uv_material_mirror,
            ),
            run_case(
                "edge_path",
                prepare_edge_path,
                (1.0, 0.0, 0.0),
                lambda bm: verify_net_and_symmetric(bm, (6, 10, 4)),
            ),
            run_case(
                "region_L",
                prepare_region_l,
                (2.0, 0.0, 0.0),
                lambda bm: verify_net_and_symmetric(bm, region_l_mirrored_net()),
            ),
        ]
        run_all(cases)
    except BaseException:
        fail()
    return None


bpy.app.timers.register(start_test, first_interval=0.3)
