# SPDX-License-Identifier: GPL-3.0-or-later

"""GUI regression for Inset Faces EXPAND_PASSTHROUGH (contract v3.1 §9 / §0).

Run with Blender's real window/event loop::

    blender --factory-startup --enable-event-simulate --python test_inset_route.py
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

MARKER_OK = "YSE_INSET_ROUTE_TEST_OK"
MARKER_FAILED = "YSE_INSET_ROUTE_TEST_FAILED"
INSET_TOOL_ID = "builtin.inset_faces"
INSET_TOOL_KEYMAP = "3D View Tool: Edit Mesh, Inset Faces"
FACE_PLUS = (0.75, 0.25)
FACE_MINUS = (-0.75, 0.25)
FACE_PLUS_NEAR = (0.25, 0.25)
FACE_MINUS_NEAR = (-0.25, 0.25)
PRECISION = 5
SYM_TOL = 1e-5
STATE: dict = {"wait_modal": False, "gen": None, "t_wait": None, "case": "startup"}


def fail(message=""):
    if message:
        print(f"YSE_INSET_ROUTE_ERROR={message}", flush=True)
    traceback.print_exc()
    print(f"YSE_INSET_ROUTE_CASE={STATE.get('case')}", flush=True)
    print(f"YSE_INSET_ROUTE_REPORTS={list(inset_bevel._REPORTS)}", flush=True)
    print(f"YSE_INSET_ROUTE_TOKEN={inset_bevel._ACTIVE_TOKEN}", flush=True)
    try:
        modal = [op.bl_idname for op in STATE["window"].modal_operators]
    except Exception:
        modal = []
    print(f"YSE_INSET_ROUTE_MODAL={modal}", flush=True)
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


def build_grid(*, select_mode="FACE"):
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
        bpy.ops.ed.undo_push(message="YSE inset baseline")
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


def face_by_center(bm, cx, cy):
    bm.faces.ensure_lookup_table()
    for face in bm.faces:
        center = face.calc_center_median()
        if abs(center.x - cx) < 1e-4 and abs(center.y - cy) < 1e-4:
            return face
    raise AssertionError(f"no face at {cx},{cy}")


def clear_selection(bm):
    for sequence in (bm.faces, bm.edges, bm.verts):
        for element in sequence:
            element.select = False
    bm.select_history.clear()


def select_faces(obj, centers):
    bm = bmesh.from_edit_mesh(obj.data)
    bpy.context.tool_settings.mesh_select_mode = (False, False, True)
    clear_selection(bm)
    for cx, cy in centers:
        face_by_center(bm, cx, cy).select = True
    bm.select_flush_mode()
    bmesh.update_edit_mesh(obj.data)


def selected_face_centers(obj):
    bm = bmesh.from_edit_mesh(obj.data)
    return sorted(
        tuple(round(float(value), 3) for value in face.calc_center_median()) for face in bm.faces if face.select
    )


def topology_counts(obj):
    bm = bmesh.from_edit_mesh(obj.data)
    return len(bm.verts), len(bm.edges), len(bm.faces)


def coordinate_key(co):
    return tuple(round(float(value), PRECISION) for value in co)


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


def assert_not_x_symmetric(obj, *, label):
    bm = bmesh.from_edit_mesh(obj.data)
    assert vertex_multiset(bm) != mirrored_multiset(bm), f"{label}: expected a one-sided (asymmetric) result"


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


def warning_messages():
    return [message for kind, message in inset_bevel._REPORTS if kind == "WARNING"]


def route_ready(event_type, *, ctrl=False, shift=False, keymap_name=None):
    addon_config = bpy.context.window_manager.keyconfigs.addon
    if addon_config is None:
        return False
    for keymap in addon_config.keymaps:
        if keymap_name is not None and keymap.name != keymap_name:
            continue
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
            keymap.name == INSET_TOOL_KEYMAP
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
    yield from wait_until(toolbar_route_ready, 8.0, "Inset Faces LEFTMOUSE CLICK_DRAG intercept never appeared")


def wait_token_cleared():
    yield from wait_until(lambda: inset_bevel._ACTIVE_TOKEN is None, 6.0, "inset/bevel token not cleared")


def activate_inset_tool():
    with override():
        bpy.context.tool_settings.workspace_tool_type = "DEFAULT"
        result = bpy.ops.wm.tool_set_by_id(name=INSET_TOOL_ID)
        if result != {"FINISHED"}:
            raise RuntimeError(f"could not activate {INSET_TOOL_ID}: {result}")
        tool = bpy.context.workspace.tools.from_space_view3d_mode("EDIT_MESH", create=False)
        if tool is None or tool.idname != INSET_TOOL_ID:
            raise RuntimeError(f"unexpected active tool: {getattr(tool, 'idname', None)}")


def inset_key_confirm(*, drag=True, confirm="LMB"):
    cx, cy = viewport_center()
    send_move(cx, cy)
    yield 0.1
    send_key("I", x=cx, y=cy)
    yield 0.3
    tx, ty = (cx + 110, cy + 75) if drag else (cx, cy)
    if drag:
        send_move(cx + 60, cy + 40)
        yield 0.15
        send_move(tx, ty)
        yield 0.15
    if confirm == "LMB":
        send_mouse("LEFTMOUSE", tx, ty)
    elif confirm == "ESC":
        send_key("ESC", x=tx, y=ty)
    elif confirm == "RMB":
        send_mouse("RIGHTMOUSE", tx, ty)
    else:
        raise ValueError(confirm)
    yield "modal_exit"
    yield from wait_token_cleared()


def inset_tool_confirm(cursor_xyz):
    x, y = window_coordinate(cursor_xyz)
    overshoot = max(80, click_drag_threshold_px() + 16)
    tx, ty = x, y + overshoot
    send_move(x, y)
    yield 0.1
    STATE["window"].event_simulate(type="LEFTMOUSE", value="PRESS", x=x, y=y)
    yield 0.08
    send_move(x, y + overshoot // 2)
    yield 0.08
    send_move(tx, ty)
    yield 0.08
    STATE["window"].event_simulate(type="LEFTMOUSE", value="RELEASE", x=tx, y=ty)
    yield "modal_exit"
    yield from wait_token_cleared()


def both_sides_selected(sel):
    return any(center[0] < 0 for center in sel) and any(center[0] > 0 for center in sel)


def case_i_confirm():
    STATE["case"] = "i_confirm"
    print("YSE_INSET_CASE=i_confirm", flush=True)
    obj = build_grid()
    select_faces(obj, [FACE_PLUS])
    yield from wait_route("I")
    before = topology_counts(obj)
    inset_bevel._REPORTS.clear()
    yield from inset_key_confirm()
    after = topology_counts(obj)
    sel = selected_face_centers(obj)
    print(f"YSE_INSET_I_CONFIRM={before}->{after} sel={sel}", flush=True)
    assert after != before, "I confirm produced no net topology change"
    assert after[0] > before[0] and after[2] > before[2], f"expected inset net increment, {before}->{after}"
    assert_x_symmetric(obj, label="i_confirm")
    assert both_sides_selected(sel), f"I confirm left a one-sided selection {sel}"
    assert_no_temp_state(label="i_confirm")


def case_undo_redo():
    STATE["case"] = "undo_redo"
    print("YSE_INSET_CASE=undo_redo", flush=True)
    obj = build_grid()
    select_faces(obj, [FACE_PLUS])
    yield from wait_route("I")
    baseline = topology_counts(obj)
    sel_before = selected_face_centers(obj)
    yield from inset_key_confirm()
    confirmed = topology_counts(obj)
    assert confirmed != baseline
    assert_x_symmetric(obj, label="undo_redo confirm")
    with override():
        undo_result = bpy.ops.ed.undo()
    assert undo_result == {"FINISHED"}, undo_result
    obj = current_object()
    undone = topology_counts(obj)
    sel_undone = selected_face_centers(obj)
    print(f"YSE_INSET_UNDO={undone} sel={sel_undone} pre={sel_before}", flush=True)
    assert undone == baseline, f"undo 1 did not restore baseline {undone} != {baseline}"
    # Keymap path: undo 1 lands on the §0 select step (pre-op mesh + expanded
    # selection). The spike-INVOKE path in §7's parenthetical "拡張前選択" is
    # a different undo grouping.
    assert both_sides_selected(sel_undone), f"undo landing was not the expanded select-step {sel_undone}"
    assert any(
        abs(center[0] - FACE_PLUS[0]) < 1e-3 and abs(center[1] - FACE_PLUS[1]) < 1e-3 for center in sel_undone
    ), f"original face missing from undo landing {sel_undone}"
    with override():
        redo_result = bpy.ops.ed.redo()
    assert redo_result == {"FINISHED"}, redo_result
    obj = current_object()
    redone = topology_counts(obj)
    assert redone == confirmed, f"redo did not restore confirmed topology {redone} != {confirmed}"
    assert_x_symmetric(obj, label="undo_redo redo")
    assert_no_temp_state(label="undo_redo")


def case_f9_undo_redo():
    STATE["case"] = "f9_undo_redo"
    print("YSE_INSET_CASE=f9_undo_redo", flush=True)
    obj = build_grid()
    select_faces(obj, [FACE_PLUS])
    yield from wait_route("I")
    yield from inset_key_confirm()
    before = topology_counts(obj)
    assert_x_symmetric(obj, label="f9 before")
    assert bpy.ops.ed.undo_redo.poll(), "ed.undo_redo.poll() is False after inset confirm"
    with override():
        result = bpy.ops.ed.undo_redo()
    assert result == {"FINISHED"}, result
    obj = current_object()
    after = topology_counts(obj)
    print(f"YSE_INSET_F9={before}->{after}", flush=True)
    assert after == before, f"undo_redo changed topology {before}->{after}"
    assert_x_symmetric(obj, label="f9 after undo_redo")
    assert_no_temp_state(label="f9_undo_redo")


def case_esc_cancel():
    STATE["case"] = "esc_cancel"
    print("YSE_INSET_CASE=esc_cancel", flush=True)
    obj = build_grid()
    select_faces(obj, [FACE_PLUS])
    yield from wait_route("I")
    before = topology_counts(obj)
    sel_before = selected_face_centers(obj)
    yield from inset_key_confirm(confirm="ESC")
    after = topology_counts(obj)
    sel_after = selected_face_centers(obj)
    print(f"YSE_INSET_ESC={after} sel={sel_after}", flush=True)
    assert after == before, f"ESC changed mesh {before}->{after}"
    assert sel_after == sel_before, f"ESC did not restore pre-expansion selection {sel_after} != {sel_before}"
    assert_no_temp_state(label="esc_cancel")


def case_rmb_cancel():
    STATE["case"] = "rmb_cancel"
    print("YSE_INSET_CASE=rmb_cancel", flush=True)
    obj = build_grid()
    select_faces(obj, [FACE_PLUS])
    yield from wait_route("I")
    before = topology_counts(obj)
    sel_before = selected_face_centers(obj)
    yield from inset_key_confirm(confirm="RMB")
    after = topology_counts(obj)
    sel_after = selected_face_centers(obj)
    print(f"YSE_INSET_RMB={after} sel={sel_after}", flush=True)
    assert after == before, f"RMB changed mesh {before}->{after}"
    assert sel_after == sel_before, f"RMB did not restore pre-expansion selection {sel_after} != {sel_before}"
    assert_no_temp_state(label="rmb_cancel")


def case_zero_drag():
    STATE["case"] = "zero_drag"
    print("YSE_INSET_CASE=zero_drag", flush=True)
    obj = build_grid()
    select_faces(obj, [FACE_PLUS])
    yield from wait_route("I")
    before = topology_counts(obj)
    sel_before = selected_face_centers(obj)
    yield from inset_key_confirm(drag=False, confirm="LMB")
    after = topology_counts(obj)
    sel_after = selected_face_centers(obj)
    print(f"YSE_INSET_ZERO={before}->{after} sel={sel_after}", flush=True)
    assert after != before, "zero-drag inset produced no topology change (RG8)"
    assert_x_symmetric(obj, label="zero_drag")
    assert sel_after != sel_before, f"zero-drag treated as CANCELLED restore {sel_after}"
    assert both_sides_selected(sel_after), f"zero-drag selection not CONFIRMED/both-sided {sel_after}"
    assert_no_temp_state(label="zero_drag")


def case_midplane():
    STATE["case"] = "midplane"
    print("YSE_INSET_CASE=midplane", flush=True)
    obj = build_grid()
    select_faces(obj, [FACE_PLUS_NEAR, FACE_MINUS_NEAR])
    yield from wait_route("I")
    before = topology_counts(obj)
    cx, cy = viewport_center()
    send_move(cx, cy)
    yield 0.1
    send_key("I", x=cx, y=cy)
    yield 0.3
    assert inset_bevel._ACTIVE_TOKEN is None, "self-mirrored region armed an expansion poller"
    send_move(cx + 60, cy + 40)
    yield 0.15
    send_move(cx + 110, cy + 75)
    yield 0.15
    send_mouse("LEFTMOUSE", cx + 110, cy + 75)
    yield "modal_exit"
    yield from wait_token_cleared()
    after = topology_counts(obj)
    print(f"YSE_INSET_MIDPLANE={before}->{after}", flush=True)
    assert after != before, "midplane inset produced no topology change"
    assert_x_symmetric(obj, label="midplane")
    assert_no_temp_state(label="midplane")


def case_tool_click_drag():
    STATE["case"] = "tool_click_drag"
    print("YSE_INSET_CASE=tool_click_drag", flush=True)
    obj = build_grid()
    select_faces(obj, [FACE_PLUS])
    activate_inset_tool()
    yield from wait_toolbar_route()
    before = topology_counts(obj)
    yield from inset_tool_confirm((FACE_PLUS[0], FACE_PLUS[1], 0.0))
    after = topology_counts(obj)
    sel = selected_face_centers(obj)
    print(f"YSE_INSET_TOOL={before}->{after} sel={sel}", flush=True)
    assert after != before, "tool CLICK_DRAG produced no topology change"
    assert_x_symmetric(obj, label="tool_click_drag")
    assert both_sides_selected(sel), f"tool inset left a one-sided selection {sel}"
    assert_no_temp_state(label="tool_click_drag")


def case_hidden():
    STATE["case"] = "hidden"
    print("YSE_INSET_CASE=hidden", flush=True)
    obj = build_grid()
    select_faces(obj, [FACE_MINUS])
    with override():
        bpy.ops.mesh.hide(unselected=False)
    select_faces(obj, [FACE_PLUS])
    yield from wait_route("I")
    before = topology_counts(obj)
    sel_before = selected_face_centers(obj)
    inset_bevel._REPORTS.clear()
    cx, cy = viewport_center()
    send_move(cx, cy)
    yield 0.1
    send_key("I", x=cx, y=cy)
    yield 0.5
    after = topology_counts(obj)
    sel_after = selected_face_centers(obj)
    warnings = warning_messages()
    print(f"YSE_INSET_HIDDEN counts={after} warnings={warnings}", flush=True)
    assert after == before, f"hidden counterpart started native inset {before}->{after}"
    assert sel_after == sel_before, f"hidden case mutated selection {sel_after}"
    assert not STATE["window"].modal_operators, "hidden case started a native modal"
    assert any("hidden counterpart" in message for message in warnings), warnings
    assert_no_temp_state(label="hidden")
    send_key("ESC", x=cx, y=cy)
    yield 0.2


def case_unmatched():
    STATE["case"] = "unmatched"
    print("YSE_INSET_CASE=unmatched", flush=True)
    obj = build_grid()
    select_faces(obj, [FACE_MINUS])
    with override():
        bpy.ops.mesh.delete(type="FACE")
    select_faces(obj, [FACE_PLUS])
    yield from wait_route("I")
    before = topology_counts(obj)
    inset_bevel._REPORTS.clear()
    yield from inset_key_confirm()
    after = topology_counts(obj)
    warnings = warning_messages()
    print(f"YSE_INSET_UNMATCHED {before}->{after} warnings={warnings}", flush=True)
    assert after != before, "unmatched case did not let native inset run"
    assert any("one side only" in message for message in warnings), warnings
    assert_not_x_symmetric(obj, label="unmatched")
    assert_no_temp_state(label="unmatched")


def main_gen():
    yield from case_i_confirm()
    yield from case_undo_redo()
    yield from case_f9_undo_redo()
    yield from case_esc_cancel()
    yield from case_rmb_cancel()
    yield from case_zero_drag()
    yield from case_midplane()
    yield from case_tool_click_drag()
    yield from case_hidden()
    yield from case_unmatched()

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
