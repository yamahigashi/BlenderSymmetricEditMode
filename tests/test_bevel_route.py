# SPDX-License-Identifier: GPL-3.0-or-later

"""GUI regression for Bevel EXPAND_PASSTHROUGH (contract v3.1 §9 / §0).

Run with Blender's real window/event loop::

    blender --factory-startup --enable-event-simulate --python test_bevel_route.py
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
from ydd_symmetric_edit import inset_bevel, keymaps, layer_names, operators, ui  # noqa: E402

MARKER_OK = "YSE_BEVEL_ROUTE_TEST_OK"
MARKER_FAILED = "YSE_BEVEL_ROUTE_TEST_FAILED"
BEVEL_TOOL_ID = "builtin.bevel"
BEVEL_TOOL_KEYMAP = "3D View Tool: Edit Mesh, Bevel"
PRECISION = 5
SYM_TOL = 1e-5
STATE: dict = {"wait_modal": False, "gen": None, "t_wait": None, "case": "startup"}


def fail(message=""):
    if message:
        print(f"YSE_BEVEL_ROUTE_ERROR={message}", flush=True)
    traceback.print_exc()
    print(f"YSE_BEVEL_ROUTE_CASE={STATE.get('case')}", flush=True)
    print(f"YSE_BEVEL_ROUTE_REPORTS={list(inset_bevel._REPORTS)}", flush=True)
    print(f"YSE_BEVEL_ROUTE_TOKEN={inset_bevel._ACTIVE_TOKEN}", flush=True)
    try:
        modal = [op.bl_idname for op in STATE["window"].modal_operators]
    except Exception:
        modal = []
    print(f"YSE_BEVEL_ROUTE_MODAL={modal}", flush=True)
    print(MARKER_FAILED, flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(1)


def viewport_context():
    window = bpy.context.window_manager.windows[0]
    area = next(area for area in window.screen.areas if area.type == "VIEW_3D")
    region = next(region for region in area.regions if region.type == "WINDOW")
    return window, area, region


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


def click_drag_threshold_px():
    inputs = bpy.context.preferences.inputs
    if hasattr(inputs, "drag_threshold_mouse"):
        base = int(inputs.drag_threshold_mouse)
    elif hasattr(inputs, "drag_threshold"):
        base = int(inputs.drag_threshold)
    else:
        raise RuntimeError("preferences.inputs exposes neither drag_threshold_mouse nor drag_threshold")
    scale = float(getattr(bpy.context.preferences.system, "ui_scale", 1.0))
    return max(1, int(round(base * scale)) + 1)


def viewport_center():
    region = STATE["region"]
    return region.x + region.width // 2, region.y + region.height // 2


def window_coordinate(coordinate):
    region = STATE["region"]
    region_3d = STATE["area"].spaces.active.region_3d
    local = view3d_utils.location_3d_to_region_2d(region, region_3d, Vector(coordinate))
    if local is None:
        raise RuntimeError(f"could not project {coordinate}")
    return int(round(region.x + local.x)), int(round(region.y + local.y))


def clear_scene():
    if bpy.context.mode != "OBJECT":
        with override():
            bpy.ops.object.mode_set(mode="OBJECT")
    for old in tuple(bpy.data.objects):
        bpy.data.objects.remove(old, do_unlink=True)
    for old_mesh in tuple(bpy.data.meshes):
        if old_mesh.users == 0:
            bpy.data.meshes.remove(old_mesh)


def build_grid(*, select_mode="EDGE"):
    clear_scene()
    with override():
        bpy.ops.mesh.primitive_grid_add(x_subdivisions=4, y_subdivisions=4, size=2.0)
        bpy.ops.wm.tool_set_by_id(name="builtin.select_box")
    obj = bpy.context.active_object
    obj.use_mesh_mirror_x = True
    obj.use_mesh_mirror_y = False
    obj.use_mesh_mirror_z = False
    with override():
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_mode(type=select_mode)
        bpy.ops.ed.undo_push(message="YSE bevel baseline")
    configure_view(STATE["area"], gizmos=False)
    STATE["object"] = obj
    return obj


def current_object():
    obj = STATE.get("object")
    if obj is not None:
        try:
            if obj.name:
                return obj
        except ReferenceError:
            pass
    return bpy.context.view_layer.objects.active


def clear_selection(bm):
    for sequence in (bm.faces, bm.edges, bm.verts):
        for element in sequence:
            element.select = False
    bm.select_history.clear()


def coordinate_key(co):
    return tuple(round(float(value), PRECISION) for value in co)


def select_one_side_edge(obj, *, on_plane=False):
    bm = bmesh.from_edit_mesh(obj.data)
    bpy.context.tool_settings.mesh_select_mode = (False, True, False)
    clear_selection(bm)
    bm.edges.ensure_lookup_table()
    target = None
    for edge in bm.edges:
        first, second = edge.verts
        xs = (first.co.x, second.co.x)
        if on_plane:
            if abs(first.co.x) > 1e-6 or abs(second.co.x) > 1e-6:
                continue
        else:
            if not (abs(first.co.x - 0.5) < 1e-6 and abs(second.co.x - 0.5) < 1e-6):
                continue
        if -0.1 < first.co.y < 0.6 and -0.1 < second.co.y < 0.6:
            target = edge
            break
    if target is None:
        raise AssertionError(f"target edge not found on_plane={on_plane} xs sample={xs}")
    target.select = True
    bm.select_flush_mode()
    bmesh.update_edit_mesh(obj.data)
    return tuple(sorted(coordinate_key(vertex.co) for vertex in target.verts))


def select_one_side_vert(obj):
    bm = bmesh.from_edit_mesh(obj.data)
    bpy.context.tool_settings.mesh_select_mode = (True, False, False)
    clear_selection(bm)
    wanted = coordinate_key((0.5, 0.5, 0.0))
    target = None
    for vertex in bm.verts:
        if coordinate_key(vertex.co) == wanted:
            target = vertex
            break
    if target is None:
        raise AssertionError("target vertex (0.5, 0.5, 0) not found")
    target.select = True
    bm.select_flush_mode()
    bmesh.update_edit_mesh(obj.data)
    return wanted


def selected_edge_keys(obj):
    bm = bmesh.from_edit_mesh(obj.data)
    return sorted(tuple(sorted(coordinate_key(vertex.co) for vertex in edge.verts)) for edge in bm.edges if edge.select)


def selected_vert_keys(obj):
    bm = bmesh.from_edit_mesh(obj.data)
    return sorted(coordinate_key(vertex.co) for vertex in bm.verts if vertex.select)


def edge_still_present(obj, edge_key):
    bm = bmesh.from_edit_mesh(obj.data)
    wanted = tuple(sorted(edge_key))
    return any(tuple(sorted(coordinate_key(vertex.co) for vertex in edge.verts)) == wanted for edge in bm.edges)


def topology_counts(obj):
    bm = bmesh.from_edit_mesh(obj.data)
    return len(bm.verts), len(bm.edges), len(bm.faces)


def vertex_multiset(bm):
    return Counter(coordinate_key(vertex.co) for vertex in bm.verts)


def mirrored_multiset(bm):
    return Counter(coordinate_key((-vertex.co.x, vertex.co.y, vertex.co.z)) for vertex in bm.verts)


def max_mirror_dev(obj):
    bm = bmesh.from_edit_mesh(obj.data)
    coords = [tuple(vertex.co) for vertex in bm.verts]
    worst = 0.0
    for co in coords:
        target = (-co[0], co[1], co[2])
        best = min(max(abs(a - b) for a, b in zip(candidate, target, strict=True)) for candidate in coords)
        worst = max(worst, best)
    return worst


def assert_x_symmetric(obj, *, label):
    bm = bmesh.from_edit_mesh(obj.data)
    assert vertex_multiset(bm) == mirrored_multiset(bm), f"{label}: vertex coordinates are not X-symmetric"
    dev = max_mirror_dev(obj)
    assert dev < SYM_TOL, f"{label}: max mirror deviation {dev}"


def assert_no_temp_state(*, label):
    assert inset_bevel._ACTIVE_TOKEN is None, f"{label}: poller token still active"
    assert inset_bevel._ARMED_GENERATION is None, f"{label}: poller generation still armed"
    assert not bpy.app.timers.is_registered(inset_bevel._poller_timer), f"{label}: poller timer leaked"
    assert not STATE["window"].modal_operators, f"{label}: modal still running {list(STATE['window'].modal_operators)}"
    assert not operators._SESSIONS, f"{label}: SESSION_REFLECT sessions leaked {list(operators._SESSIONS)}"
    obj = current_object()
    if obj is not None and obj.mode == "EDIT":
        bm = bmesh.from_edit_mesh(obj.data)
        for name in layer_names.TEMP_LAYER_NAMES:
            for domain, layers in (
                ("verts", bm.verts.layers.int),
                ("edges", bm.edges.layers.int),
                ("faces", bm.faces.layers.int),
            ):
                assert layers.get(name) is None, f"{label}: temporary {domain} layer leaked: {name}"


def route_ready(event_type, *, ctrl=False, shift=False):
    addon_config = bpy.context.window_manager.keyconfigs.addon
    if addon_config is None:
        return False
    for keymap in addon_config.keymaps:
        for item in keymap.keymap_items:
            if item.idname != keymaps.INSET_BEVEL_INTERCEPT_OPERATOR:
                continue
            if item.type != event_type:
                continue
            if bool(item.ctrl) != ctrl or bool(item.shift) != shift:
                continue
            if item.active:
                return True
    return False


def toolbar_route_ready():
    for keymap, item in keymaps._REGISTERED_ITEMS:
        if (
            keymap.name == BEVEL_TOOL_KEYMAP
            and item.idname == keymaps.INSET_BEVEL_INTERCEPT_OPERATOR
            and item.type == "LEFTMOUSE"
            and item.value == "CLICK_DRAG"
            and item.active
        ):
            return True
    return False


def send_move(x, y):
    STATE["window"].event_simulate(type="MOUSEMOVE", value="NOTHING", x=x, y=y)


def send_key(event_type, *, ctrl=False, shift=False, x=None, y=None):
    if x is None or y is None:
        x, y = viewport_center()
    kwargs = {"type": event_type, "value": "PRESS", "x": x, "y": y}
    if ctrl:
        kwargs["ctrl"] = True
    if shift:
        kwargs["shift"] = True
    STATE["window"].event_simulate(**kwargs)
    kwargs["value"] = "RELEASE"
    STATE["window"].event_simulate(**kwargs)


def send_mouse(event_type, x, y):
    STATE["window"].event_simulate(type=event_type, value="PRESS", x=x, y=y)
    STATE["window"].event_simulate(type=event_type, value="RELEASE", x=x, y=y)


def wait_until(predicate, timeout, message):
    started = time.monotonic()
    while not predicate():
        if time.monotonic() - started > timeout:
            raise RuntimeError(message)
        yield 0.1


def wait_route(event_type, *, ctrl=False, shift=False):
    yield 1.2
    yield from wait_until(
        lambda: route_ready(event_type, ctrl=ctrl, shift=shift),
        8.0,
        f"intercept route {event_type} ctrl={ctrl} shift={shift} never appeared",
    )


def wait_toolbar_route():
    yield 1.2
    yield from wait_until(toolbar_route_ready, 8.0, "Bevel LEFTMOUSE CLICK_DRAG intercept never appeared")


def wait_token_cleared():
    yield from wait_until(lambda: inset_bevel._ACTIVE_TOKEN is None, 6.0, "inset/bevel token not cleared")


def activate_bevel_tool():
    with override():
        bpy.context.tool_settings.workspace_tool_type = "DEFAULT"
        result = bpy.ops.wm.tool_set_by_id(name=BEVEL_TOOL_ID)
        if result != {"FINISHED"}:
            raise RuntimeError(f"could not activate {BEVEL_TOOL_ID}: {result}")
        tool = bpy.context.workspace.tools.from_space_view3d_mode("EDIT_MESH", create=False)
        if tool is None or tool.idname != BEVEL_TOOL_ID:
            raise RuntimeError(f"unexpected active tool: {getattr(tool, 'idname', None)}")
        return tool


def set_tool_bevel_affect(affect):
    tool = bpy.context.workspace.tools.from_space_view3d_mode("EDIT_MESH", create=False)
    if tool is None or tool.idname != BEVEL_TOOL_ID:
        raise RuntimeError(f"bevel tool not active: {getattr(tool, 'idname', None)}")
    props = tool.operator_properties("mesh.bevel")
    props.affect = affect
    got = getattr(props, "affect", None)
    if got != affect:
        raise RuntimeError(f"failed to set bevel affect={affect!r}, got {got!r}")


def bevel_key_confirm(*, shift=False, confirm="LMB"):
    cx, cy = viewport_center()
    send_move(cx, cy)
    yield 0.1
    send_key("B", ctrl=True, shift=shift, x=cx, y=cy)
    yield 0.3
    if confirm == "ESC":
        send_key("ESC", x=cx, y=cy)
        yield "modal_exit"
        yield from wait_token_cleared()
        return
    tx, ty = cx + 50, cy + 35
    send_move(tx, ty)
    yield 0.15
    if confirm == "LMB":
        send_mouse("LEFTMOUSE", tx, ty)
    else:
        raise ValueError(confirm)
    yield "modal_exit"
    yield from wait_token_cleared()


def bevel_tool_confirm(cursor_xyz):
    x, y = window_coordinate(cursor_xyz)
    overshoot = max(80, click_drag_threshold_px() + 16)
    tx, ty = x + overshoot, y
    send_move(x, y)
    yield 0.1
    STATE["window"].event_simulate(type="LEFTMOUSE", value="PRESS", x=x, y=y)
    yield 0.08
    send_move(x + overshoot // 2, y)
    yield 0.08
    send_move(tx, ty)
    yield 0.08
    STATE["window"].event_simulate(type="LEFTMOUSE", value="RELEASE", x=tx, y=ty)
    yield "modal_exit"
    yield from wait_token_cleared()


def case_ctrl_b():
    STATE["case"] = "ctrl_b"
    print("YSE_BEVEL_CASE=ctrl_b", flush=True)
    obj = build_grid(select_mode="EDGE")
    select_one_side_edge(obj)
    yield from wait_route("B", ctrl=True)
    before = topology_counts(obj)
    yield from bevel_key_confirm()
    after = topology_counts(obj)
    print(f"YSE_BEVEL_CTRL_B={before}->{after}", flush=True)
    assert after != before, "Ctrl+B produced no topology change"
    assert_x_symmetric(obj, label="ctrl_b")
    assert_no_temp_state(label="ctrl_b")


def case_ctrl_shift_b():
    STATE["case"] = "ctrl_shift_b"
    print("YSE_BEVEL_CASE=ctrl_shift_b", flush=True)
    obj = build_grid(select_mode="VERT")
    select_one_side_vert(obj)
    yield from wait_route("B", ctrl=True, shift=True)
    before = topology_counts(obj)
    yield from bevel_key_confirm(shift=True)
    after = topology_counts(obj)
    print(f"YSE_BEVEL_CTRL_SHIFT_B={before}->{after}", flush=True)
    assert after != before, "Ctrl+Shift+B produced no topology change"
    assert_x_symmetric(obj, label="ctrl_shift_b")
    assert_no_temp_state(label="ctrl_shift_b")


def case_on_plane():
    STATE["case"] = "on_plane"
    print("YSE_BEVEL_CASE=on_plane", flush=True)
    obj = build_grid(select_mode="EDGE")
    select_one_side_edge(obj, on_plane=True)
    yield from wait_route("B", ctrl=True)
    before = topology_counts(obj)
    cx, cy = viewport_center()
    send_move(cx, cy)
    yield 0.1
    send_key("B", ctrl=True, x=cx, y=cy)
    yield 0.3
    assert inset_bevel._ACTIVE_TOKEN is None, "on-plane edge armed an expansion poller"
    send_move(cx + 50, cy + 35)
    yield 0.15
    send_mouse("LEFTMOUSE", cx + 50, cy + 35)
    yield "modal_exit"
    yield from wait_token_cleared()
    after = topology_counts(obj)
    print(f"YSE_BEVEL_ONPLANE={before}->{after}", flush=True)
    assert after != before, "on-plane bevel produced no topology change"
    assert_x_symmetric(obj, label="on_plane")
    assert_no_temp_state(label="on_plane")


def case_esc_cancel():
    STATE["case"] = "esc_cancel"
    print("YSE_BEVEL_CASE=esc_cancel", flush=True)
    obj = build_grid(select_mode="EDGE")
    edge_key = select_one_side_edge(obj)
    yield from wait_route("B", ctrl=True)
    before = topology_counts(obj)
    sel_before = selected_edge_keys(obj)
    yield from bevel_key_confirm(confirm="ESC")
    bpy.context.tool_settings.mesh_select_mode = (False, True, False)
    after = topology_counts(obj)
    sel_after = selected_edge_keys(obj)
    sel_verts = selected_vert_keys(obj)
    print(f"YSE_BEVEL_ESC={after} sel_edges={sel_after} sel_verts={sel_verts} pre={sel_before}", flush=True)
    assert after == before, f"ESC changed mesh {before}->{after}"
    assert sel_after == sel_before, f"ESC did not restore the pre-expansion selection {sel_after} pre={sel_before}"
    assert edge_still_present(obj, edge_key), f"original edge missing from mesh after ESC {edge_key}"
    assert_no_temp_state(label="esc_cancel")


def case_tool_edges():
    STATE["case"] = "tool_edges"
    print("YSE_BEVEL_CASE=tool_edges", flush=True)
    obj = build_grid(select_mode="EDGE")
    select_one_side_edge(obj)
    activate_bevel_tool()
    set_tool_bevel_affect("EDGES")
    yield from wait_toolbar_route()
    before = topology_counts(obj)
    yield from bevel_tool_confirm((0.5, 0.25, 0.0))
    after = topology_counts(obj)
    print(f"YSE_BEVEL_TOOL_EDGES={before}->{after}", flush=True)
    assert after != before, "bevel tool CLICK_DRAG produced no topology change"
    assert_x_symmetric(obj, label="tool_edges")
    assert_no_temp_state(label="tool_edges")


def case_tool_verts():
    STATE["case"] = "tool_verts"
    print("YSE_BEVEL_CASE=tool_verts", flush=True)
    obj = build_grid(select_mode="VERT")
    select_one_side_vert(obj)
    activate_bevel_tool()
    set_tool_bevel_affect("VERTICES")
    yield from wait_toolbar_route()
    before = topology_counts(obj)
    yield from bevel_tool_confirm((0.5, 0.5, 0.0))
    after = topology_counts(obj)
    print(f"YSE_BEVEL_TOOL_VERTS={before}->{after}", flush=True)
    assert after != before, "bevel tool affect=VERTICES produced no topology change"
    assert_x_symmetric(obj, label="tool_verts")
    assert_no_temp_state(label="tool_verts")


def case_tool_verts_ctrl_b_edges():
    STATE["case"] = "tool_verts_ctrl_b_edges"
    print("YSE_BEVEL_CASE=tool_verts_ctrl_b_edges", flush=True)
    obj = build_grid(select_mode="EDGE")
    select_one_side_edge(obj)
    activate_bevel_tool()
    set_tool_bevel_affect("VERTICES")
    yield from wait_route("B", ctrl=True)
    before = topology_counts(obj)
    yield from bevel_key_confirm()
    after = topology_counts(obj)
    print(f"YSE_BEVEL_MIXED={before}->{after}", flush=True)
    assert after != before, "Ctrl+B with bevel tool affect=VERTICES produced no topology change"
    assert_x_symmetric(obj, label="tool_verts_ctrl_b_edges")
    assert_no_temp_state(label="tool_verts_ctrl_b_edges")


def main_gen():
    yield from case_ctrl_b()
    yield from case_ctrl_shift_b()
    yield from case_on_plane()
    yield from case_esc_cancel()
    yield from case_tool_edges()
    yield from case_tool_verts()
    yield from case_tool_verts_ctrl_b_edges()

    print(MARKER_OK, flush=True)
    sys.stdout.flush()
    addon.unregister()
    clear_scene()

    def quit_now():
        bpy.ops.wm.quit_blender()
        return None

    bpy.app.timers.register(quit_now, first_interval=0.3)


def tick():
    try:
        window = STATE["window"]
        if STATE["wait_modal"]:
            if window.modal_operators:
                if STATE["t_wait"] and time.monotonic() - STATE["t_wait"] > 15.0:
                    raise RuntimeError("modal never exited")
                return 0.1
            STATE["wait_modal"] = False
        try:
            delay = next(STATE["gen"])
        except StopIteration:
            return None
        if delay == "modal_exit":
            STATE["wait_modal"] = True
            STATE["t_wait"] = time.monotonic()
            return 0.3
        return delay
    except BaseException:
        fail()
        return None


def start_test():
    try:
        addon.register()
        preferences = ui.get_addon_preferences(bpy.context)
        if preferences is not None:
            preferences.enabled = True
        addon.sync_persistent_keymap(True)
        window, area, region = viewport_context()
        configure_view(area, gizmos=False)
        STATE.update(window=window, area=area, region=region)
        STATE["gen"] = main_gen()
        bpy.app.timers.register(tick, first_interval=0.4)
    except BaseException:
        fail()
    return None


bpy.app.timers.register(start_test, first_interval=0.3)
