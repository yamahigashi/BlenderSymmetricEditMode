# SPDX-License-Identifier: GPL-3.0-or-later

"""GUI regression for midplane-shared Extrude (KMI E, contract v3.1 §6 Wave 1).

Run with Blender's real window/event loop as documented in ``docs/testing.md``::

    blender --factory-startup --enable-event-simulate --no-window-focus \
        --disable-crash-handler -p 40 40 960 600 --python test_extrude_midplane.py
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

MARKER_OK = "YSE_EXTRUDE_MIDPLANE_TEST_OK"
MARKER_FAILED = "YSE_EXTRUDE_MIDPLANE_TEST_FAILED"
NX, NY = 6, 4
PRECISION = 5
# P2 one-side native nets (created − deleted). Overall mirrored net is
# one-side × 2 minus the on-plane share; values pinned from the contract
# examples and 4.2 measurement where the share is geometry-dependent.
SEAM_FACE_NATIVE = (4, 8, 4)
SEAM_FACE_INPLANE = (6, 13, 7)
SEAM_FACE_V = (8, 16, 8)
SEAM_EDGE_NATIVE = (2, 3, 1)
SEAM_EDGE_V = (4, 6, 2)
SEAM_VERT_NATIVE = (1, 1, 0)
SEAM_VERT_V = (2, 2, 0)
SEAM_1X2_NATIVE = (6, 12, 6)
# 1x2 in-plane: 3 on-plane copies are shared, partner face+edge deleted.
SEAM_1X2_INPLANE = (9, 19, 10)
# Fin: native 4/8/4 plus mirror 4/8/5 (self-face consumed, not deleted again).
FIN_FACE_NET = (8, 16, 9)
# 2x2 on-plane raised on a 0.5 stem: pinned from 4.2 APPLY (the theoretical
# 2*(8,16,8) floating-grid net does not apply once the region shares a
# manifold stem).
ONPLANE_2X2_NET = (17, 36, 20)
STATE = {}
_ONLY = os.environ.get("YSE_MIDPLANE_ONLY", "")


def fail(message=""):
    if message:
        print(f"YSE_EXTRUDE_MIDPLANE_ERROR={message}", flush=True)
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


def drag_pixels():
    return max(80, click_drag_threshold_px() + 16)


def configure_view(area):
    # Distance 10 keeps the 6×4 grid and the z≤2 fins inside the 960×600
    # safe band used by docs/testing.md (small window, --no-window-focus).
    region_3d = area.spaces.active.region_3d
    region_3d.view_perspective = "ORTHO"
    region_3d.view_rotation = Quaternion((1.0, 0.0, 0.0, 0.0))
    region_3d.view_location = (0.0, 0.0, 0.0)
    region_3d.view_distance = 10.0
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


def clear_scene():
    for old in tuple(bpy.data.objects):
        bpy.data.objects.remove(old, do_unlink=True)
    for old_mesh in tuple(bpy.data.meshes):
        if old_mesh.users == 0:
            bpy.data.meshes.remove(old_mesh)


def build_mesh(name):
    clear_scene()
    mesh = bpy.data.meshes.new(f"YSE_MidplaneMesh_{name}")
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
    obj = bpy.data.objects.new(f"YSE_MidplaneObject_{name}", mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.use_mesh_mirror_x = True
    obj.use_mesh_mirror_y = False
    obj.use_mesh_mirror_z = False
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    with override():
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.context.tool_settings.mesh_select_mode = (True, False, False)
        bpy.ops.ed.undo_push(message=f"YSE midplane baseline {name}")
    STATE["object"] = obj
    return obj


def grid_vert(bm, i, j, z=0.0):
    x, y = grid_xy(i, j)
    for vertex in bm.verts:
        if abs(vertex.co.x - x) < 1e-4 and abs(vertex.co.y - y) < 1e-4 and abs(vertex.co.z - z) < 1e-4:
            return vertex
    raise AssertionError(f"grid vert {i},{j},z={z} not found")


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
        if have == expect and all(abs(vertex.co.z) < 1e-4 for vertex in face.verts):
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


def onplane_faces(bm, *, min_z=0.25):
    return [
        face
        for face in bm.faces
        if all(abs(vertex.co.x) < 1e-4 for vertex in face.verts)
        and max(vertex.co.z for vertex in face.verts) > min_z
    ]


def native_extrude_z(delta_z):
    """Grow a fin from a floor seam with the same operator a user would fire.

    Isolated bmesh faces leave the draw extract cache stale; a later native
    extrude then null-derefs in blender::draw::extract_tris. A disconnected
    primitive plane is also wrong here: native keeps that source face
    (keep_orig) and the addon declines it. A face glued to the x=0 floor
    seam is non-manifold (incident >= 3) so the census is undefined. The
    stem-then-fin sequence below keeps the selected face manifold.
    """

    with override():
        result = bpy.ops.mesh.extrude_region_move(
            "EXEC_DEFAULT",
            TRANSFORM_OT_translate={
                "value": (0.0, 0.0, float(delta_z)),
                "orient_type": "GLOBAL",
                "orient_matrix": (
                    (1.0, 0.0, 0.0),
                    (0.0, 1.0, 0.0),
                    (0.0, 0.0, 1.0),
                ),
                "orient_matrix_type": "GLOBAL",
                "constraint_axis": (False, False, True),
            },
        )
    assert result == {"FINISHED"}, result


def select_onplane_edges_at_z(bm, z, y_span=(-1.0, 1.0), expected=2):
    bpy.context.tool_settings.mesh_select_mode = (False, True, False)
    clear_selection(bm)
    y_lo, y_hi = y_span
    matches = []
    for edge in bm.edges:
        first, second = edge.verts
        if abs(first.co.x) > 1e-4 or abs(second.co.x) > 1e-4:
            continue
        if abs(first.co.z - z) > 1e-4 or abs(second.co.z - z) > 1e-4:
            continue
        if abs(first.co.y - second.co.y) < 1e-4:
            continue
        ys = (float(first.co.y), float(second.co.y))
        if min(ys) >= y_lo - 1e-4 and max(ys) <= y_hi + 1e-4:
            matches.append(edge)
    assert len(matches) == expected, [(tuple(edge.verts[0].co), tuple(edge.verts[1].co)) for edge in matches]
    for edge in matches:
        edge.select = True
    bm.select_flush_mode()


def flush_and_extrude(obj, delta_z):
    bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
    native_extrude_z(delta_z)
    bm = bmesh.from_edit_mesh(obj.data)
    bm.normal_update()
    return bm


def coordinate_key(co):
    return tuple(round(float(value), PRECISION) for value in co)


def vertex_multiset(bm):
    return Counter(coordinate_key(vertex.co) for vertex in bm.verts)


def mirrored_multiset(bm):
    return Counter(coordinate_key((-vertex.co.x, vertex.co.y, vertex.co.z)) for vertex in bm.verts)


def assert_x_symmetric(bm):
    assert vertex_multiset(bm) == mirrored_multiset(bm), "vertex coordinates are not X-symmetric"


def assert_layers_removed(bm):
    for name in layer_names.TEMP_LAYER_NAMES:
        for sequence in (bm.verts, bm.edges, bm.faces):
            assert sequence.layers.int.get(name) is None, f"temporary layer leaked: {name}"


def topology_counts(bm):
    return len(bm.verts), len(bm.edges), len(bm.faces)


def live_net(bm):
    return (
        len(bm.verts) - STATE["baseline"][0],
        len(bm.edges) - STATE["baseline"][1],
        len(bm.faces) - STATE["baseline"][2],
    )


def latest_record():
    records = [
        record
        for record in operators._HISTORY_RECORDS.values()
        if record.session.object_name == STATE["object"].name
    ]
    return max(records, key=lambda record: record.sequence) if records else None


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


def extrude_events(cursor_xyz, drag=(0, 80), *, constrain_axis=None):
    x, y = window_coordinate(cursor_xyz)
    events = [
        {"type": "MOUSEMOVE", "value": "NOTHING", "x": x, "y": y},
        {"type": "E", "value": "PRESS", "x": x, "y": y},
        {"type": "E", "value": "RELEASE", "x": x, "y": y},
    ]
    if constrain_axis:
        events.append({"type": constrain_axis, "value": "PRESS", "x": x, "y": y})
        events.append({"type": constrain_axis, "value": "RELEASE", "x": x, "y": y})
    tx, ty = x + drag[0], y + drag[1]
    if drag != (0, 0):
        events.append({"type": "MOUSEMOVE", "value": "NOTHING", "x": x + drag[0] // 2, "y": y + drag[1] // 2})
        events.append({"type": "MOUSEMOVE", "value": "NOTHING", "x": tx, "y": ty})
    events.append({"type": "LEFTMOUSE", "value": "PRESS", "x": tx, "y": ty})
    events.append({"type": "LEFTMOUSE", "value": "RELEASE", "x": tx, "y": ty})
    return events


def run_case(name, prepare, cursor_xyz, verify, *, drag=None, constrain_axis=None):
    if drag is None:
        drag = (0, drag_pixels())

    def start(next_case):
        try:
            print(f"YSE_EXTRUDE_MIDPLANE_CASE={name}", flush=True)
            obj = build_mesh(name)
            bm = bmesh.from_edit_mesh(obj.data)
            prepare(bm, obj)
            bm = bmesh.from_edit_mesh(obj.data)
            bm.normal_update()
            bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
            with override():
                push = bpy.ops.ed.undo_push(message=f"YSE midplane prepared {name}")
            assert push == {"FINISHED"}, push
            STATE["baseline"] = topology_counts(bmesh.from_edit_mesh(obj.data))
            print(f"YSE_EXTRUDE_MIDPLANE_BASELINE {name}={STATE['baseline']}", flush=True)

            def settled():
                try:
                    bm2 = bmesh.from_edit_mesh(STATE["object"].data)
                    got = live_net(bm2)
                    print(f"YSE_EXTRUDE_MIDPLANE_NET {name}={got}", flush=True)
                    verify(bm2)
                    next_case()
                except BaseException:
                    fail()

            events = extrude_events(cursor_xyz, drag=drag, constrain_axis=constrain_axis)
            print(f"YSE_EXTRUDE_MIDPLANE_EVENTS {name} n={len(events)} origin={cursor_xyz}", flush=True)
            send_events(events, lambda: wait_settled(settled))
        except BaseException:
            fail()

    return start


def verify_apply(bm, expected_net, *, undo=True):
    got = live_net(bm)
    assert got == expected_net, f"net {got} != {expected_net}"
    assert_x_symmetric(bm)
    assert_layers_removed(bm)
    record = latest_record()
    assert record is not None, "missing history record"
    assert record.status == "COMMITTED", record
    assert record.session.prepare_disposition == "APPLY", record.session
    if undo:
        with override():
            undo_result = bpy.ops.ed.undo()
        assert undo_result == {"FINISHED"}, undo_result
        bm2 = bmesh.from_edit_mesh(STATE["object"].data)
        counts = topology_counts(bm2)
        assert counts == STATE["baseline"], f"one undo did not restore baseline: {counts} != {STATE['baseline']}"
        assert_x_symmetric(bm2)
        assert_layers_removed(bm2)


def verify_decline(bm, expected_net, reason_substr, *, require_asymmetric=True, undo=True):
    got = live_net(bm)
    assert got == expected_net, f"declined net {got} != {expected_net}"
    if require_asymmetric and got != (0, 0, 0):
        assert vertex_multiset(bm) != mirrored_multiset(bm), "declined extrude should keep an unmirrored native result"
    assert_layers_removed(bm)
    record = latest_record()
    assert record is not None, "missing history record for decline"
    assert record.session.prepare_disposition == "DECLINE", record.session
    reason = record.session.prepare_disposition_reason or ""
    assert reason_substr in reason, reason
    print(f"YSE_EXTRUDE_MIDPLANE_DECLINE_REASON={reason!r}", flush=True)
    if undo:
        with override():
            undo_result = bpy.ops.ed.undo()
        assert undo_result == {"FINISHED"}, undo_result
        bm2 = bmesh.from_edit_mesh(STATE["object"].data)
        counts = topology_counts(bm2)
        assert counts == STATE["baseline"], f"one undo did not restore baseline: {counts} != {STATE['baseline']}"
        assert_layers_removed(bm2)


def prepare_seam_face(bm, _obj):
    select_faces(bm, [(3, 1)])


def prepare_seam_edge(bm, _obj):
    select_edges(bm, [((3, 1), (3, 2))])


def prepare_seam_vert(bm, _obj):
    select_verts(bm, [(3, 2)])


def prepare_seam_1x2(bm, _obj):
    select_faces(bm, [(3, 1), (3, 2)])


def prepare_fin_face(bm, obj):
    # Stem 0.5 (unselected) then a 1.0 fin so the selected face stays
    # 2-manifold. verify_fin_box ignores on-plane faces with max z <= 0.5.
    select_edges(bm, [((3, 1), (3, 2))])
    bm = flush_and_extrude(obj, 0.5)
    select_onplane_edges_at_z(bm, 0.5, y_span=(-1.0, 0.0), expected=1)
    bm = flush_and_extrude(obj, 1.0)
    bpy.context.tool_settings.mesh_select_mode = (False, False, True)
    clear_selection(bm)
    faces = onplane_faces(bm, min_z=0.6)
    assert len(faces) == 1, faces
    faces[0].select = True
    bm.select_flush_mode()


def prepare_onplane_2x2(bm, obj):
    # Stem 0.5 then two 1.0 rows: selected 2x2 sits above the stem so the
    # bottom edges stay 2-manifold (stem + fin) instead of 3-manifold.
    select_edges(bm, [((3, 1), (3, 2)), ((3, 2), (3, 3))])
    bm = flush_and_extrude(obj, 0.5)
    select_onplane_edges_at_z(bm, 0.5, y_span=(-1.0, 1.0), expected=2)
    bm = flush_and_extrude(obj, 1.0)
    select_onplane_edges_at_z(bm, 1.5, y_span=(-1.0, 1.0), expected=2)
    bm = flush_and_extrude(obj, 1.0)
    bpy.context.tool_settings.mesh_select_mode = (False, False, True)
    clear_selection(bm)
    faces = onplane_faces(bm, min_z=0.6)
    assert len(faces) == 4, faces
    for face in faces:
        face.select = True
    bm.select_flush_mode()


def prepare_cross_selection(bm, _obj):
    select_faces(bm, [(4, 1), (1, 1)])


def verify_fin_box(bm):
    # Geometry must be read before verify_apply's undo invalidates this BMesh.
    xs = [vertex.co.x for vertex in bm.verts if vertex.co.z > 0.4]
    assert any(x > 0.05 for x in xs) and any(x < -0.05 for x in xs), xs
    on_plane_fin = [
        face
        for face in bm.faces
        if all(abs(vertex.co.x) < 1e-4 for vertex in face.verts)
        and max(vertex.co.z for vertex in face.verts) > 0.5
    ]
    assert not on_plane_fin, "self-consumed fin face should be gone"
    verify_apply(bm, FIN_FACE_NET)


def verify_onplane_2x2(bm):
    # Interior origin (0,0,1.5) of the raised 2x2 is consumed. Its
    # ID-inherited copies must survive on both sides; a live re-resolve of
    # self-delete would remove them. Read before verify_apply's undo.
    interior_copies = [
        vertex
        for vertex in bm.verts
        if abs(vertex.co.y) < 1e-3 and abs(vertex.co.z - 1.5) < 1e-3 and abs(vertex.co.x) > 1e-3
    ]
    xs = sorted(vertex.co.x for vertex in interior_copies)
    assert len(xs) >= 2 and xs[0] < 0 < xs[-1], f"interior copies lost: {xs}"
    verify_apply(bm, ONPLANE_2X2_NET)


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
        pixels = drag_pixels()
        print(f"YSE_EXTRUDE_MIDPLANE_DRAG_PX={pixels}", flush=True)
        cases = [
            run_case(
                "seam_face_inplane",
                prepare_seam_face,
                (0.5, -0.5, 0.0),
                lambda bm: verify_apply(bm, SEAM_FACE_INPLANE),
            ),
            run_case(
                "seam_face_v",
                prepare_seam_face,
                (0.5, -0.5, 0.0),
                lambda bm: verify_apply(bm, SEAM_FACE_V),
                drag=(pixels, pixels),
                constrain_axis="X",
            ),
            run_case(
                "seam_edge_inplane",
                prepare_seam_edge,
                (0.0, -0.5, 0.0),
                lambda bm: verify_apply(bm, SEAM_EDGE_NATIVE),
            ),
            run_case(
                "seam_edge_v",
                prepare_seam_edge,
                (0.0, -0.5, 0.0),
                lambda bm: verify_apply(bm, SEAM_EDGE_V),
                drag=(pixels, 0),
                constrain_axis="X",
            ),
            run_case(
                "seam_vert_inplane",
                prepare_seam_vert,
                (0.0, 0.0, 0.0),
                lambda bm: verify_apply(bm, SEAM_VERT_NATIVE),
            ),
            run_case(
                "seam_vert_v",
                prepare_seam_vert,
                (0.0, 0.0, 0.0),
                lambda bm: verify_apply(bm, SEAM_VERT_V),
                drag=(pixels, 0),
                constrain_axis="X",
            ),
            run_case(
                "seam_1x2_inplane",
                prepare_seam_1x2,
                (0.5, 0.0, 0.0),
                lambda bm: verify_apply(bm, SEAM_1X2_INPLANE),
            ),
            run_case(
                "fin_face",
                prepare_fin_face,
                (0.0, -0.5, 1.0),
                verify_fin_box,
                drag=(pixels, 0),
                constrain_axis="X",
            ),
            run_case(
                "onplane_2x2",
                prepare_onplane_2x2,
                (0.0, 0.0, 1.5),
                verify_onplane_2x2,
                drag=(pixels, 0),
                constrain_axis="X",
            ),
            run_case(
                "zero_offset",
                prepare_seam_face,
                (0.5, -0.5, 0.0),
                lambda bm: verify_decline(bm, SEAM_FACE_NATIVE, "zero-offset"),
                drag=(0, 0),
            ),
            run_case(
                "offplane_cross",
                prepare_cross_selection,
                (1.5, -0.5, 0.0),
                lambda bm: verify_decline(
                    bm,
                    (8, 16, 8),
                    "intersects its mirror image",
                    require_asymmetric=False,
                ),
            ),
        ]
        names = [
            "seam_face_inplane",
            "seam_face_v",
            "seam_edge_inplane",
            "seam_edge_v",
            "seam_vert_inplane",
            "seam_vert_v",
            "seam_1x2_inplane",
            "fin_face",
            "onplane_2x2",
            "zero_offset",
            "offplane_cross",
        ]
        if _ONLY:
            wanted = {item.strip() for item in _ONLY.replace("+", ",").split(",") if item.strip()}
            cases = [case for case, name in zip(cases, names, strict=True) if name in wanted]
        run_all(cases)
    except BaseException:
        fail()
    return None


bpy.app.timers.register(start_test, first_interval=0.3)
