# SPDX-License-Identifier: GPL-3.0-or-later

"""GUI regression for Stage 3a Alt+E extrude menu (opener + two wrappers).

Run with Blender's real window/event loop::

    blender --factory-startup --enable-event-simulate \\
        --python test_extrude_menu.py

Wrapper items are invoked directly (INVOKE_DEFAULT). Arrow/ENTER selection
of Alt+E menu items is not simulated — it is flaky under event_simulate.
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from collections import Counter
from math import pi
from pathlib import Path
from types import SimpleNamespace

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
from ydd_symmetric_edit import extrude_menu, keymaps, layer_names, operators, session_state  # noqa: E402
from ydd_symmetric_edit import watcher as watcher_mod  # noqa: E402

MARKER_OK = "YSE_EXTRUDE_MENU_TEST_OK"
MARKER_FAILED = "YSE_EXTRUDE_MENU_TEST_FAILED"
NX, NY = 6, 4
PRECISION = 5
STATE = {}


def fail(message=""):
    if message:
        print(f"YSE_EXTRUDE_MENU_ERROR={message}", flush=True)
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
    mesh = bpy.data.meshes.new(f"YSE_ExtrudeMenuMesh_{name}")
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
    obj = bpy.data.objects.new(f"YSE_ExtrudeMenuObject_{name}", mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.use_mesh_mirror_x = True
    obj.use_mesh_mirror_y = False
    obj.use_mesh_mirror_z = False
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    with override():
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.context.tool_settings.mesh_select_mode = (True, False, False)
        bpy.ops.ed.undo_push(message=f"YSE extrude menu baseline {name}")
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
                        f"extrude menu flow never settled; modal={[op.bl_idname for op in STATE['window'].modal_operators]} "
                        f"sessions={list(operators._SESSIONS)}"
                    )
                return 0.1
            bpy.app.timers.register(done, first_interval=0.25)
        except BaseException:
            fail()
        return None

    bpy.app.timers.register(poll, first_interval=0.1)


def wait_modal_gone(done, started=None):
    started = time.monotonic() if started is None else started

    def poll():
        try:
            if tuple(STATE["window"].modal_operators):
                if time.monotonic() - started > 12.0:
                    raise RuntimeError(
                        f"native modal never finished; modal={[op.bl_idname for op in STATE['window'].modal_operators]}"
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


class _RecordingOp:
    def __init__(self):
        self.angle = None


class _RecordingLayout:
    def __init__(self):
        self.items: list[tuple] = []
        self.operator_context = ""

    def operator(self, idname, text=""):
        op = _RecordingOp()
        self.items.append(("op", idname, text, op))
        return op

    def separator(self):
        self.items.append(("sep", "", "", None))

    def template_node_operator_asset_menu_items(self, catalog_path=""):
        self.items.append(("asset", catalog_path, "", None))


def _recorded_draw_sequence(recorder: _RecordingLayout):
    sequence = []
    for kind, idname, text, op in recorder.items:
        if kind == "op":
            sequence.append((kind, idname, text, getattr(op, "angle", None)))
        else:
            sequence.append((kind, idname, text))
    return recorder.operator_context, tuple(sequence)


def _silent_report(*_args):
    return None


def _make_prior(mesh_name, window_pointer, **kwargs):
    fields = dict(
        mesh_name=mesh_name,
        window_pointer=window_pointer,
        saw_modal=True,
        tool_kind="EXTRUDE_NORMAL",
        extrude_options_captured=False,
        confirmed_operator_idname="",
        confirmed_operator_pointer=0,
        confirmed_selection_signature=(),
        history_token=0,
    )
    fields.update(kwargs)
    return SimpleNamespace(**fields)


def _pop_sentinel(pointer):
    session_state._SESSIONS.pop(pointer, None)


class _LaunchCounter:
    def __init__(self, result=None, error=None):
        self.calls: list[str] = []
        self.result = {"FINISHED"} if result is None else result
        self.error = error

    def __call__(self, idname):
        self.calls.append(idname)
        if self.error is not None:
            raise self.error
        return set(self.result)


def opener_items_by_keymap():
    addon_config = bpy.context.window_manager.keyconfigs.addon
    assert addon_config is not None, "addon keyconfig missing"
    found: dict[tuple[str, str, str], int] = {}
    for keymap in addon_config.keymaps:
        if keymap.is_modal:
            continue
        count = sum(1 for item in keymap.keymap_items if item.idname == keymaps.EXTRUDE_MENU_OPENER)
        if count:
            found[(keymap.name, keymap.space_type, keymap.region_type)] = count
    return found


def case_live_verify() -> None:
    print("YSE_EXTRUDE_MENU_CASE=live_verify", flush=True)
    assert keymaps.has_extrude_menu_routes(), "has_extrude_menu_routes() is False after enable"
    routes, _fingerprint = keymaps._extrude_menu_routes(bpy.context.window_manager)
    assert routes, "scanner found no extrude-menu routes"
    route = routes[0]
    print(
        f"YSE_EXTRUDE_MENU_FACTORY={route.keymap_name} {route.event.type} alt={route.event.alt}",
        flush=True,
    )

    addon_config = bpy.context.window_manager.keyconfigs.addon
    user_config = bpy.context.window_manager.keyconfigs.user
    assert addon_config is not None and user_config is not None
    addon_keymap = keymaps._find_keymap(addon_config, route.keymap_identity)
    user_keymap = keymaps._find_keymap(user_config, route.keymap_identity)
    assert addon_keymap is not None, route.keymap_identity
    assert user_keymap is not None, route.keymap_identity

    user_native, _user_opener, user_other, user_supported = keymaps._extrude_menu_event_census(user_keymap, route.event)
    _addon_native, addon_opener, addon_other, addon_supported = keymaps._extrude_menu_event_census(
        addon_keymap, route.event
    )
    assert user_native == 1, (user_native, route.event)
    assert addon_opener == 1, addon_opener
    assert user_other + addon_other == 0, (user_other, addon_other)
    assert user_supported + addon_supported == 0, (user_supported, addon_supported)
    assert keymaps.extrude_menu_route_is_current(route.route_key)
    print("YSE_EXTRUDE_MENU_LIVE_VERIFY_OK", flush=True)


def case_rebuild() -> None:
    print("YSE_EXTRUDE_MENU_CASE=rebuild", flush=True)
    for _ in range(4):
        keymaps.sync(True)
        keymaps._refresh(force=True)
    counts = opener_items_by_keymap()
    assert counts, "no opener KMIs after rebuild"
    for identity, count in counts.items():
        assert count == 1, f"opener accumulated on {identity}: {count}"
    print(f"YSE_EXTRUDE_MENU_REBUILD={counts}", flush=True)
    print("YSE_EXTRUDE_MENU_REBUILD_OK", flush=True)


def case_missing_menu() -> None:
    print("YSE_EXTRUDE_MENU_CASE=missing_menu", flush=True)
    build_mesh("missing_menu")
    routes = list(keymaps._EXTRUDE_MENU_ROUTES_BY_KEY)
    assert routes, "no stored extrude-menu route keys"
    route_key = routes[0]
    bpy.utils.unregister_class(extrude_menu.YSE_MT_extrude)
    try:
        with override():
            result = bpy.ops.mesh.ydd_symmetric_edit_extrude_menu("INVOKE_DEFAULT", route_key=route_key)
        assert result == {"PASS_THROUGH"}, result
    finally:
        bpy.utils.register_class(extrude_menu.YSE_MT_extrude)
    print("YSE_EXTRUDE_MENU_MISSING_OK", flush=True)


def _draw_menu(_obj=None):
    recorder = _RecordingLayout()
    menu = SimpleNamespace(layout=recorder)
    with override():
        extrude_menu.YSE_MT_extrude.draw(menu, bpy.context)
    return _recorded_draw_sequence(recorder)


_TRAILER = (
    ("sep", "", ""),
    ("op", "mesh.extrude_repeat", "", None),
    ("op", "mesh.spin", "", pi * 2),
    ("asset", "Mesh/Extrude", ""),
)
_FACE_ITEMS = (
    ("op", extrude_menu.WRAPPER_FACES, "Extrude Faces", None),
    ("op", extrude_menu.WRAPPER_ALONG, "Extrude Faces Along Normals", None),
    ("op", extrude_menu.WRAPPER_INDIV, "Extrude Individual Faces", None),
    ("op", "view3d.edit_mesh_extrude_manifold_normal", "Extrude Manifold", None),
)


def case_mixed_draw() -> None:
    print("YSE_EXTRUDE_MENU_CASE=mixed_draw", flush=True)
    obj = build_mesh("mixed_draw")
    bm = bmesh.from_edit_mesh(obj.data)
    bpy.context.tool_settings.mesh_select_mode = (True, True, True)
    clear_selection(bm)
    grid_face(bm, 4, 1).select = True
    bm.select_flush_mode()
    bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)

    context_value, sequence = _draw_menu(obj)
    print(f"YSE_EXTRUDE_MENU_DRAW={sequence}", flush=True)
    assert context_value == "INVOKE_REGION_WIN", context_value
    expected = (
        *_FACE_ITEMS,
        ("op", "mesh.extrude_edges_move", "Extrude Edges", None),
        ("op", "mesh.extrude_vertices_move", "Extrude Vertices", None),
        *_TRAILER,
    )
    assert sequence == expected, sequence
    print("YSE_EXTRUDE_MENU_MIXED_DRAW_OK", flush=True)


def case_t3_face_only_draw() -> None:
    print("YSE_EXTRUDE_MENU_CASE=t3_face_only", flush=True)
    obj = build_mesh("t3_face_only")
    bm = bmesh.from_edit_mesh(obj.data)
    bpy.context.tool_settings.mesh_select_mode = (False, False, True)
    clear_selection(bm)
    grid_face(bm, 4, 1).select = True
    bm.select_flush_mode()
    bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
    context_value, sequence = _draw_menu(obj)
    assert context_value == "INVOKE_REGION_WIN", context_value
    assert sequence == (*_FACE_ITEMS, *_TRAILER), sequence
    print("YSE_EXTRUDE_MENU_T3_FACE_ONLY_OK", flush=True)


def case_t3_edge_selected_draw() -> None:
    print("YSE_EXTRUDE_MENU_CASE=t3_edge_selected", flush=True)
    obj = build_mesh("t3_edge_selected")
    bm = bmesh.from_edit_mesh(obj.data)
    bpy.context.tool_settings.mesh_select_mode = (False, True, False)
    clear_selection(bm)
    face = grid_face(bm, 4, 1)
    face.edges[0].select = True
    bm.select_flush_mode()
    bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
    context_value, sequence = _draw_menu(obj)
    assert context_value == "INVOKE_REGION_WIN", context_value
    expected = (("op", "mesh.extrude_edges_move", "Extrude Edges", None), *_TRAILER)
    assert sequence == expected, sequence
    print("YSE_EXTRUDE_MENU_T3_EDGE_OK", flush=True)


def case_t3_vert_selected_draw() -> None:
    print("YSE_EXTRUDE_MENU_CASE=t3_vert_selected", flush=True)
    obj = build_mesh("t3_vert_selected")
    bm = bmesh.from_edit_mesh(obj.data)
    bpy.context.tool_settings.mesh_select_mode = (True, False, False)
    clear_selection(bm)
    grid_vert(bm, 5, 1).select = True
    bm.select_flush_mode()
    bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
    context_value, sequence = _draw_menu(obj)
    assert context_value == "INVOKE_REGION_WIN", context_value
    expected = (("op", "mesh.extrude_vertices_move", "Extrude Vertices", None), *_TRAILER)
    assert sequence == expected, sequence
    print("YSE_EXTRUDE_MENU_T3_VERT_OK", flush=True)


def case_failure_cleanup() -> None:
    print("YSE_EXTRUDE_MENU_CASE=failure_cleanup", flush=True)
    obj = build_mesh("failure_cleanup")
    bm = bmesh.from_edit_mesh(obj.data)
    select_faces(bm, [(4, 1)])
    bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)

    original = extrude_menu._invoke_native_extrude_item

    def fake_cancelled(idname):
        del idname
        return {"CANCELLED"}

    extrude_menu._invoke_native_extrude_item = fake_cancelled
    try:
        with override():
            result = bpy.ops.mesh.ydd_symmetric_edit_extrude_faces("INVOKE_DEFAULT")
        assert result == {"CANCELLED"}, result
        assert not operators._SESSIONS, list(operators._SESSIONS)
    finally:
        extrude_menu._invoke_native_extrude_item = original
    print("YSE_EXTRUDE_MENU_CLEANUP_OK", flush=True)


def _invoke_wrapper_for_test():
    with override():
        return extrude_menu._invoke_extrude_wrapper(
            bpy.context,
            _silent_report,
            tool_kind="EXTRUDE_NORMAL",
            native_idname=extrude_menu.NATIVE_FACES,
        )


def case_t1_still_modal() -> None:
    print("YSE_EXTRUDE_MENU_CASE=t1_still_modal", flush=True)
    obj = build_mesh("t1_still_modal")
    prior_ptr = 0xC0FFEE
    prior = _make_prior(obj.data.name, prior_ptr, saw_modal=False)
    session_state._SESSIONS[prior_ptr] = prior
    counter = _LaunchCounter()
    original = extrude_menu._invoke_native_extrude_item
    extrude_menu._invoke_native_extrude_item = counter
    try:
        result = _invoke_wrapper_for_test()
        assert result == {"CANCELLED"}, result
        assert counter.calls == [], counter.calls
        assert session_state._SESSIONS[prior_ptr] is prior
    finally:
        extrude_menu._invoke_native_extrude_item = original
        _pop_sentinel(prior_ptr)
    print("YSE_EXTRUDE_MENU_T1_STILL_MODAL_OK", flush=True)


def case_t1_grace_capture_fail() -> None:
    print("YSE_EXTRUDE_MENU_CASE=t1_grace_capture_fail", flush=True)
    obj = build_mesh("t1_grace_fail")
    prior_ptr = 0xC0FFEE
    prior = _make_prior(obj.data.name, prior_ptr, saw_modal=True)
    session_state._SESSIONS[prior_ptr] = prior
    counter = _LaunchCounter()
    original_native = extrude_menu._invoke_native_extrude_item
    original_helper = extrude_menu._capture_confirmed_extrude_result
    original_finish = extrude_menu._sync_finish_prior
    finish_calls = []

    def fail_helper(_session):
        return False

    def track_finish(session):
        finish_calls.append(session)
        return True

    extrude_menu._invoke_native_extrude_item = counter
    extrude_menu._capture_confirmed_extrude_result = fail_helper
    extrude_menu._sync_finish_prior = track_finish
    try:
        result = _invoke_wrapper_for_test()
        assert result == {"CANCELLED"}, result
        assert counter.calls == [], counter.calls
        assert finish_calls == [], finish_calls
        assert session_state._SESSIONS[prior_ptr] is prior
    finally:
        extrude_menu._invoke_native_extrude_item = original_native
        extrude_menu._capture_confirmed_extrude_result = original_helper
        extrude_menu._sync_finish_prior = original_finish
        _pop_sentinel(prior_ptr)
    print("YSE_EXTRUDE_MENU_T1_GRACE_FAIL_OK", flush=True)


def case_t1_grace_capture_ok() -> None:
    print("YSE_EXTRUDE_MENU_CASE=t1_grace_capture_ok", flush=True)
    obj = build_mesh("t1_grace_ok")
    prior_ptr = 0xC0FFEE
    prior = _make_prior(obj.data.name, prior_ptr, saw_modal=True)
    session_state._SESSIONS[prior_ptr] = prior
    counter = _LaunchCounter()
    helper_calls = []
    original_native = extrude_menu._invoke_native_extrude_item
    original_helper = extrude_menu._capture_confirmed_extrude_result
    original_finish = extrude_menu._sync_finish_prior
    original_prepare = extrude_menu._prepare_session

    def ok_helper(session):
        helper_calls.append(session)
        return True

    def finish_and_pop(session):
        session_state._SESSIONS.pop(session.window_pointer, None)
        return True

    def fake_prepare(*_args, **_kwargs):
        return True

    extrude_menu._invoke_native_extrude_item = counter
    extrude_menu._capture_confirmed_extrude_result = ok_helper
    extrude_menu._sync_finish_prior = finish_and_pop
    extrude_menu._prepare_session = fake_prepare
    try:
        result = _invoke_wrapper_for_test()
        assert result == {"FINISHED"}, result
        assert helper_calls == [prior], helper_calls
        assert counter.calls == [extrude_menu.NATIVE_FACES], counter.calls
        assert prior_ptr not in session_state._SESSIONS
    finally:
        extrude_menu._invoke_native_extrude_item = original_native
        extrude_menu._capture_confirmed_extrude_result = original_helper
        extrude_menu._sync_finish_prior = original_finish
        extrude_menu._prepare_session = original_prepare
        _pop_sentinel(prior_ptr)
    print("YSE_EXTRUDE_MENU_T1_GRACE_OK", flush=True)


def case_t1_already_captured() -> None:
    print("YSE_EXTRUDE_MENU_CASE=t1_already_captured", flush=True)
    obj = build_mesh("t1_already_captured")
    prior_ptr = 0xC0FFEE
    prior = _make_prior(
        obj.data.name,
        prior_ptr,
        saw_modal=True,
        confirmed_operator_idname="MESH_OT_extrude_region_move",
        confirmed_operator_pointer=12345,
        extrude_options_captured=True,
    )
    session_state._SESSIONS[prior_ptr] = prior
    counter = _LaunchCounter()
    recapture_calls = []
    original_native = extrude_menu._invoke_native_extrude_item
    original_helper = extrude_menu._capture_confirmed_extrude_result
    original_finish = extrude_menu._sync_finish_prior
    original_prepare = extrude_menu._prepare_session

    def counting_helper(session):
        recapture_calls.append(session)
        return original_helper(session)

    def finish_and_pop(session):
        session_state._SESSIONS.pop(session.window_pointer, None)
        return True

    extrude_menu._invoke_native_extrude_item = counter
    extrude_menu._capture_confirmed_extrude_result = counting_helper
    extrude_menu._sync_finish_prior = finish_and_pop
    extrude_menu._prepare_session = lambda *_args, **_kwargs: True
    try:
        result = _invoke_wrapper_for_test()
        assert result == {"FINISHED"}, result
        assert recapture_calls == [prior]
        assert original_helper(prior) is True
        assert counter.calls == [extrude_menu.NATIVE_FACES], counter.calls
        assert prior_ptr not in session_state._SESSIONS
    finally:
        extrude_menu._invoke_native_extrude_item = original_native
        extrude_menu._capture_confirmed_extrude_result = original_helper
        extrude_menu._sync_finish_prior = original_finish
        extrude_menu._prepare_session = original_prepare
        _pop_sentinel(prior_ptr)
    print("YSE_EXTRUDE_MENU_T1_CAPTURED_OK", flush=True)


def case_t1_reconlict() -> None:
    print("YSE_EXTRUDE_MENU_CASE=t1_reconlict", flush=True)
    obj = build_mesh("t1_reconlict")
    prior_ptr = 0xC0FFEE
    prior = _make_prior(
        obj.data.name,
        prior_ptr,
        saw_modal=True,
        confirmed_operator_idname="MESH_OT_extrude_region_move",
        confirmed_operator_pointer=12345,
        extrude_options_captured=True,
    )
    session_state._SESSIONS[prior_ptr] = prior
    counter = _LaunchCounter()
    original_native = extrude_menu._invoke_native_extrude_item
    original_finish = extrude_menu._sync_finish_prior

    def finish_leave(session):
        del session
        return True

    extrude_menu._invoke_native_extrude_item = counter
    extrude_menu._sync_finish_prior = finish_leave
    try:
        result = _invoke_wrapper_for_test()
        assert result == {"CANCELLED"}, result
        assert counter.calls == [], counter.calls
        assert session_state._SESSIONS[prior_ptr] is prior
    finally:
        extrude_menu._invoke_native_extrude_item = original_native
        extrude_menu._sync_finish_prior = original_finish
        _pop_sentinel(prior_ptr)
    print("YSE_EXTRUDE_MENU_T1_RECONFLICT_OK", flush=True)


def case_t2_prepare_raises() -> None:
    print("YSE_EXTRUDE_MENU_CASE=t2_prepare_raises", flush=True)
    obj = build_mesh("t2_prepare_raises")
    select_faces(bmesh.from_edit_mesh(obj.data), [(4, 1)])
    bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
    sentinel_ptr = 0xDEAD
    sentinel = _make_prior("other-mesh", sentinel_ptr)
    session_state._SESSIONS[sentinel_ptr] = sentinel
    original_sched = watcher_mod._schedule_passthrough_watcher
    original_native = extrude_menu._invoke_native_extrude_item
    counter = _LaunchCounter()

    def boom(*_args, **_kwargs):
        raise RuntimeError("yse-prepare-tail")

    watcher_mod._schedule_passthrough_watcher = boom
    extrude_menu._invoke_native_extrude_item = counter
    raised = False
    try:
        try:
            _invoke_wrapper_for_test()
        except RuntimeError as exc:
            raised = True
            assert "yse-prepare-tail" in str(exc), exc
        assert raised, "prepare tail exception was swallowed"
        current = bpy.context.window.as_pointer() if bpy.context.window is not None else 0
        assert current not in session_state._SESSIONS, list(session_state._SESSIONS)
        assert session_state._SESSIONS.get(sentinel_ptr) is sentinel
        bm = bmesh.from_edit_mesh(obj.data)
        assert_layers_removed(bm)
        assert counter.calls == [], counter.calls
    finally:
        watcher_mod._schedule_passthrough_watcher = original_sched
        extrude_menu._invoke_native_extrude_item = original_native
        _pop_sentinel(sentinel_ptr)
        current = bpy.context.window.as_pointer() if bpy.context.window is not None else 0
        if current in session_state._SESSIONS:
            from ydd_symmetric_edit.session import cleanup_session

            cleanup_session(current)
    print("YSE_EXTRUDE_MENU_T2_PREPARE_RAISES_OK", flush=True)


def case_t2_native_raises() -> None:
    print("YSE_EXTRUDE_MENU_CASE=t2_native_raises", flush=True)
    obj = build_mesh("t2_native_raises")
    select_faces(bmesh.from_edit_mesh(obj.data), [(4, 1)])
    bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
    sentinel_ptr = 0xDEAD
    sentinel = _make_prior("other-mesh", sentinel_ptr)
    session_state._SESSIONS[sentinel_ptr] = sentinel
    original_native = extrude_menu._invoke_native_extrude_item
    extrude_menu._invoke_native_extrude_item = _LaunchCounter(error=RuntimeError("yse-native-boom"))
    raised = False
    try:
        try:
            _invoke_wrapper_for_test()
        except RuntimeError as exc:
            raised = True
            assert "yse-native-boom" in str(exc), exc
        assert raised, "native exception was swallowed"
        current = bpy.context.window.as_pointer() if bpy.context.window is not None else 0
        assert current not in session_state._SESSIONS, list(session_state._SESSIONS)
        assert session_state._SESSIONS.get(sentinel_ptr) is sentinel
        bm = bmesh.from_edit_mesh(obj.data)
        assert_layers_removed(bm)
    finally:
        extrude_menu._invoke_native_extrude_item = original_native
        _pop_sentinel(sentinel_ptr)
        current = bpy.context.window.as_pointer() if bpy.context.window is not None else 0
        if current in session_state._SESSIONS:
            from ydd_symmetric_edit.session import cleanup_session

            cleanup_session(current)
    print("YSE_EXTRUDE_MENU_T2_NATIVE_RAISES_OK", flush=True)


def case_t2_launch_counts() -> None:
    print("YSE_EXTRUDE_MENU_CASE=t2_launch_counts", flush=True)
    obj = build_mesh("t2_launch_counts")
    select_faces(bmesh.from_edit_mesh(obj.data), [(4, 1)])
    bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
    original_native = extrude_menu._invoke_native_extrude_item
    original_prepare = extrude_menu._prepare_session
    original_enabled = keymaps._ENABLED
    original_finish = extrude_menu._sync_finish_prior
    original_helper = extrude_menu._capture_confirmed_extrude_result

    try:
        prepared = _LaunchCounter()
        extrude_menu._invoke_native_extrude_item = prepared
        extrude_menu._prepare_session = lambda *_args, **_kwargs: True
        result = _invoke_wrapper_for_test()
        assert result == {"FINISHED"}, result
        assert prepared.calls == [extrude_menu.NATIVE_FACES], prepared.calls

        keymaps._ENABLED = False
        bypass = _LaunchCounter()
        extrude_menu._invoke_native_extrude_item = bypass
        result = _invoke_wrapper_for_test()
        assert result == {"FINISHED"}, result
        assert bypass.calls == [extrude_menu.NATIVE_FACES], bypass.calls
        keymaps._ENABLED = original_enabled

        prior_ptr = 0xC0FFEE
        prior = _make_prior(
            obj.data.name,
            prior_ptr,
            saw_modal=True,
            confirmed_operator_idname="MESH_OT_extrude_region_move",
            confirmed_operator_pointer=99,
            extrude_options_captured=True,
        )
        session_state._SESSIONS[prior_ptr] = prior

        def finish_and_pop(session):
            session_state._SESSIONS.pop(session.window_pointer, None)
            return True

        resolved = _LaunchCounter()
        extrude_menu._invoke_native_extrude_item = resolved
        extrude_menu._sync_finish_prior = finish_and_pop
        extrude_menu._capture_confirmed_extrude_result = lambda _session: True
        extrude_menu._prepare_session = lambda *_args, **_kwargs: True
        result = _invoke_wrapper_for_test()
        assert result == {"FINISHED"}, result
        assert resolved.calls == [extrude_menu.NATIVE_FACES], resolved.calls

        still = _make_prior(obj.data.name, prior_ptr, saw_modal=False)
        session_state._SESSIONS[prior_ptr] = still
        blocked = _LaunchCounter()
        extrude_menu._invoke_native_extrude_item = blocked
        result = _invoke_wrapper_for_test()
        assert result == {"CANCELLED"}, result
        assert blocked.calls == [], blocked.calls
        _pop_sentinel(prior_ptr)
    finally:
        extrude_menu._invoke_native_extrude_item = original_native
        extrude_menu._prepare_session = original_prepare
        extrude_menu._sync_finish_prior = original_finish
        extrude_menu._capture_confirmed_extrude_result = original_helper
        keymaps._ENABLED = original_enabled
        _pop_sentinel(0xC0FFEE)
        current = bpy.context.window.as_pointer() if bpy.context.window is not None else 0
        if current in session_state._SESSIONS:
            from ydd_symmetric_edit.session import cleanup_session

            cleanup_session(current)
    print("YSE_EXTRUDE_MENU_T2_LAUNCH_COUNTS_OK", flush=True)


def case_t4_opener_invoke() -> None:
    print("YSE_EXTRUDE_MENU_CASE=t4_opener_invoke", flush=True)
    build_mesh("t4_opener")
    routes = list(keymaps._EXTRUDE_MENU_ROUTES_BY_KEY)
    assert routes, "no stored extrude-menu route keys"
    route_key = routes[0]
    calls = []

    def fake_call_menu(*args, **kwargs):
        calls.append((args, kwargs))
        return {"FINISHED"}

    class _WmProxy:
        def __init__(self, real_wm):
            object.__setattr__(self, "_real", real_wm)

        def __getattr__(self, name):
            if name == "call_menu":
                return fake_call_menu
            return getattr(self._real, name)

    class _OpsProxy:
        def __init__(self, real_ops):
            object.__setattr__(self, "_real", real_ops)

        def __getattr__(self, name):
            if name == "wm":
                return _WmProxy(self._real.wm)
            return getattr(self._real, name)

    class _BpyProxy:
        def __init__(self, real):
            object.__setattr__(self, "_real", real)

        def __getattr__(self, name):
            if name == "ops":
                return _OpsProxy(self._real.ops)
            return getattr(self._real, name)

    original_bpy = extrude_menu.bpy
    extrude_menu.bpy = _BpyProxy(original_bpy)
    try:
        with override():
            result = bpy.ops.mesh.ydd_symmetric_edit_extrude_menu("INVOKE_DEFAULT", route_key=route_key)
        assert result == {"FINISHED"}, result
        assert calls, "wm.call_menu was not invoked"
        _args, kwargs = calls[0]
        assert kwargs.get("name") == extrude_menu.EXTRUDE_MENU or (
            len(_args) >= 2 and _args[1] == extrude_menu.EXTRUDE_MENU
        ), calls[0]
    finally:
        extrude_menu.bpy = original_bpy
    print("YSE_EXTRUDE_MENU_T4_OPENER_OK", flush=True)


def case_m1_own_window_not_conflict() -> None:
    print("YSE_EXTRUDE_MENU_CASE=m1_own_window_not_conflict", flush=True)
    obj = build_mesh("m1_own_window")
    current = bpy.context.window.as_pointer()
    own = _make_prior(obj.data.name, current, saw_modal=False)
    session_state._SESSIONS[current] = own
    try:
        with override():
            found = extrude_menu._find_prior_session_same_mesh(bpy.context)
            status, prior = extrude_menu._classify_wrapper_prepare(bpy.context)
        assert found is None, found
        assert status != "CONFLICT", (status, prior)
        assert prior is None
    finally:
        _pop_sentinel(current)
    print("YSE_EXTRUDE_MENU_M1_OK", flush=True)


def case_m2_draw_exception_probe() -> None:
    print("YSE_EXTRUDE_MENU_CASE=m2_draw_exception_probe", flush=True)
    build_mesh("m2_draw")
    routes = list(keymaps._EXTRUDE_MENU_ROUTES_BY_KEY)
    assert routes, "no stored extrude-menu route keys"
    route_key = routes[0]
    original_draw = extrude_menu.YSE_MT_extrude.draw
    original_print_exc = extrude_menu.traceback.print_exc
    flags = {"draw": False, "except": False}

    def raising_draw(self, context):
        del self, context
        flags["draw"] = True
        raise RuntimeError("yse-draw-probe")

    def tracking_print_exc(*args, **kwargs):
        flags["except"] = True
        return original_print_exc(*args, **kwargs)

    extrude_menu.YSE_MT_extrude.draw = raising_draw
    extrude_menu.traceback.print_exc = tracking_print_exc
    result = None
    try:
        with override():
            result = bpy.ops.mesh.ydd_symmetric_edit_extrude_menu("INVOKE_DEFAULT", route_key=route_key)
        print(
            f"YSE_EXTRUDE_MENU_M2_DRAW={flags['draw']} EXCEPT={flags['except']} RESULT={result}",
            flush=True,
        )
        STATE["m2"] = {"draw": flags["draw"], "except": flags["except"], "result": result}
    finally:
        extrude_menu.YSE_MT_extrude.draw = original_draw
        extrude_menu.traceback.print_exc = original_print_exc
        try:
            STATE["window"].event_simulate(type="ESC", value="PRESS")
            STATE["window"].event_simulate(type="ESC", value="RELEASE")
        except Exception:
            pass
    print("YSE_EXTRUDE_MENU_M2_OK", flush=True)


def invoke_wrapper_and_drag(op_idname, cursor_xyz, verify, *, drag=(0, 80)):
    def start(next_case):
        try:
            print(f"YSE_EXTRUDE_MENU_CASE={op_idname}", flush=True)
            obj = build_mesh(op_idname)
            bm = bmesh.from_edit_mesh(obj.data)
            select_faces(bm, [(4, 1)])
            bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
            STATE["baseline"] = topology_counts(bmesh.from_edit_mesh(obj.data))

            x, y = window_coordinate(cursor_xyz)
            STATE["window"].event_simulate(type="MOUSEMOVE", value="NOTHING", x=x, y=y)
            namespace, name = op_idname.split(".", 1)
            with override():
                result = getattr(getattr(bpy.ops, namespace), name)("INVOKE_DEFAULT")
            assert result == {"FINISHED"}, result

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

        case_live_verify()
        case_rebuild()
        case_missing_menu()
        case_mixed_draw()
        case_t3_face_only_draw()
        case_t3_edge_selected_draw()
        case_t3_vert_selected_draw()
        case_failure_cleanup()
        case_t1_still_modal()
        case_t1_grace_capture_fail()
        case_t1_grace_capture_ok()
        case_t1_already_captured()
        case_t1_reconlict()
        case_t2_prepare_raises()
        case_t2_native_raises()
        case_t2_launch_counts()
        case_t4_opener_invoke()
        case_m1_own_window_not_conflict()

        cursor = (1.5, -0.5, 0.0)

        def start_m2(next_case):
            try:
                case_m2_draw_exception_probe()
                next_case()
            except BaseException:
                fail()

        cases = [
            invoke_wrapper_and_drag(
                extrude_menu.WRAPPER_FACES,
                cursor,
                lambda bm: verify_net_and_symmetric(bm, (8, 16, 8), undo=True),
            ),
            invoke_wrapper_and_drag(
                extrude_menu.WRAPPER_ALONG,
                cursor,
                lambda bm: verify_net_and_symmetric(bm, (8, 16, 8)),
            ),
            invoke_wrapper_and_drag(
                extrude_menu.WRAPPER_INDIV,
                cursor,
                lambda bm: verify_net_and_symmetric(bm, (8, 16, 8)),
            ),
            start_m2,
        ]
        run_all(cases)
    except BaseException:
        fail()
    return None


bpy.app.timers.register(start_test, first_interval=0.3)
