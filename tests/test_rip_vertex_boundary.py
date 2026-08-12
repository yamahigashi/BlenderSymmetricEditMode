# SPDX-License-Identifier: GPL-3.0-or-later

"""GUI regression for the symmetric Rip (V) route: vertex / boundary / failure.

Run with Blender's real window/event loop::

    blender --factory-startup --enable-event-simulate \
        --python test_rip_vertex_boundary.py

Serialized cases on an X-symmetric grid plane:

1. singlevert   V on one interior vertex: fan split mirrored.
2. boundary     V on a path reaching the mesh boundary: boundary endpoint
                duplication mirrored.
3. asymmetry    a displaced mirror-side vertex makes the seam unpairable:
                preflight declines, native result stays, WARNING path.
4. rollback     an injected apply failure restores the mirror side from the
                backup while keeping the native rip.
5. bothsides    disjoint both-sides selection (no mirror pair): the session
                runs and the result stays X-symmetric.
6. crossing     selection contains a disconnected mirror pair: session may
                start (overlap passthrough removed in Phase 4) but native
                rip fails / leaves an unmirrored result.  Must stay last.
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
from ydd_symmetric_edit import layer_names, operators, rip  # noqa: E402

MARKER_OK = "YSE_RIP_VERTEX_BOUNDARY_TEST_OK"
MARKER_FAILED = "YSE_RIP_VERTEX_BOUNDARY_TEST_FAILED"
NX, NY = 6, 4
PRECISION = 5
STATE = {}


def fail(message=""):
    if message:
        print(f"YSE_RIP_VERTEX_BOUNDARY_ERROR={message}", flush=True)
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


def build_mesh(name):
    for old in tuple(bpy.data.objects):
        bpy.data.objects.remove(old, do_unlink=True)
    for old_mesh in tuple(bpy.data.meshes):
        if old_mesh.users == 0:
            bpy.data.meshes.remove(old_mesh)
    mesh = bpy.data.meshes.new(f"YSE_RipVB_{name}")
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
    obj = bpy.data.objects.new(f"YSE_RipVBObj_{name}", mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.use_mesh_mirror_x = True
    obj.use_mesh_mirror_y = False
    obj.use_mesh_mirror_z = False
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    with override():
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.context.tool_settings.mesh_select_mode = (True, False, False)
        bpy.ops.ed.undo_push(message=f"YSE rip vb baseline {name}")
    STATE["object"] = obj
    return obj


def grid_vert(bm, i, j):
    for vertex in bm.verts:
        if abs(vertex.co.x - (i - NX / 2)) < 0.4 and abs(vertex.co.y - (j - NY / 2)) < 0.4 and abs(vertex.co.z) < 0.4:
            return vertex
    raise AssertionError(f"grid vert {i},{j} not found")


def select_verts(bm, ij_list):
    for face in bm.faces:
        face.select = False
    for edge in bm.edges:
        edge.select = False
    for vertex in bm.verts:
        vertex.select = False
    bm.select_history.clear()
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
    assert bm.verts.layers.int.get(layer_names.VERT_RIP_ID_LAYER) is None, "rip vertex layer leaked"
    assert bm.edges.layers.int.get(layer_names.EDGE_ORIGINAL_LAYER) is None, "edge layer leaked"
    assert bm.faces.layers.int.get(layer_names.FACE_ID_LAYER) is None, "face layer leaked"


def topology_counts(bm):
    return len(bm.verts), len(bm.edges), len(bm.faces)


def send_events(events, done, index=0):
    def step():
        try:
            if index < len(events):
                STATE["window"].event_simulate(**events[index])
                send_events(events, done, index + 1)
            else:
                bpy.app.timers.register(done, first_interval=0.2)
        except BaseException:
            fail()
        return None

    bpy.app.timers.register(step, first_interval=0.09)


def wait_settled(done, started=None):
    started = time.monotonic() if started is None else started

    def poll():
        try:
            busy = bool(STATE["window"].modal_operators) or bool(operators._SESSIONS)
            if busy:
                if time.monotonic() - started > 12.0:
                    raise RuntimeError("rip flow never settled")
                return 0.1
            bpy.app.timers.register(done, first_interval=0.25)
        except BaseException:
            fail()
        return None

    bpy.app.timers.register(poll, first_interval=0.1)


def rip_events(cursor_ij, drag=(40, 0)):
    x, y = window_coordinate((cursor_ij[0] - NX / 2, cursor_ij[1] - NY / 2, 0.0))
    events = [
        {"type": "MOUSEMOVE", "value": "NOTHING", "x": x, "y": y},
        {"type": "V", "value": "PRESS", "x": x, "y": y},
        {"type": "V", "value": "RELEASE", "x": x, "y": y},
    ]
    tx, ty = x + drag[0], y + drag[1]
    events.append({"type": "MOUSEMOVE", "value": "NOTHING", "x": tx, "y": ty})
    events.append({"type": "LEFTMOUSE", "value": "PRESS", "x": tx, "y": ty})
    events.append({"type": "LEFTMOUSE", "value": "RELEASE", "x": tx, "y": ty})
    return events


def run_case(name, select_ij, cursor_ij, verify, *, mutate=None):
    def start(next_case):
        try:
            print(f"YSE_RIP_VB_CASE={name}", flush=True)
            obj = build_mesh(name)
            bm = bmesh.from_edit_mesh(obj.data)
            if mutate is not None:
                mutate(bm)
            STATE["baseline"] = topology_counts(bm)
            select_verts(bm, select_ij)
            bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
            operators._FINISH_REPORTS.clear()

            def settled():
                try:
                    bm2 = bmesh.from_edit_mesh(STATE["object"].data)
                    verify(bm2)
                    next_case()
                except BaseException:
                    fail()

            send_events(rip_events(cursor_ij), lambda: wait_settled(settled))
        except BaseException:
            fail()

    return start


def verify_singlevert(bm):
    dv = len(bm.verts) - STATE["baseline"][0]
    de = len(bm.edges) - STATE["baseline"][1]
    assert dv == 2, f"expected 2 new vertices, got {dv}"
    assert de == 4, f"expected 4 new edges (2 source + 2 mirror), got {de}"
    assert_x_symmetric(bm)
    assert_layers_removed(bm)


def verify_boundary(bm):
    dv = len(bm.verts) - STATE["baseline"][0]
    de = len(bm.edges) - STATE["baseline"][1]
    # Path (4,0),(4,1),(4,2): native duplicates all 3 selected verts
    # (boundary endpoint included) across 3 seam edges (R0 §5-2).
    assert dv == 6, f"expected 6 new vertices, got {dv}"
    assert de == 6, f"expected 6 new edges, got {de}"
    assert_x_symmetric(bm)
    assert_layers_removed(bm)


def displace_mirror_vertex(bm):
    grid_vert(bm, 2, 2).co.z += 0.3


def verify_asymmetry_declined(bm):
    dv = len(bm.verts) - STATE["baseline"][0]
    assert dv == 2, f"expected native-only rip to add 2 vertices, got {dv}"
    assert vertex_multiset(bm) != mirrored_multiset(bm), "asymmetric mesh must not be mirrored"
    assert_layers_removed(bm)


def install_failure_injection():
    original = rip.apply_mirrored_rip

    def broken(bm, snapshot, mirror_face_ids):
        count, reason = original(bm, snapshot, mirror_face_ids)
        del count, reason
        return 0, "injected test failure"

    STATE["original_apply"] = original
    rip.apply_mirrored_rip = broken


def verify_rollback(bm):
    try:
        dv = len(bm.verts) - STATE["baseline"][0]
        # The injected failure happens AFTER the mirror split mutated the
        # mesh; the finish must restore the backup so only the native rip
        # remains.
        assert dv == 2, f"expected only the native rip to remain, got dv={dv}"
        assert vertex_multiset(bm) != mirrored_multiset(bm), "rollback should leave the native-only result"
        assert_layers_removed(bm)
    finally:
        rip.apply_mirrored_rip = STATE["original_apply"]


def warning_messages():
    return [message for kind, message in operators._FINISH_REPORTS if kind == "WARNING"]


def verify_crossing_passthrough(bm):
    # Disconnected mirror-pair selection: after settle the session is gone.
    # Native typically fails ("Rip failed"); if it rips anything it stays
    # unmirrored (no clean seam for either path).  Must stay last in the
    # case list (a failed native V can poison the next simulated rip).
    #
    # Phase 4 removed the prepare-time overlap passthrough: a native-only
    # change must surface as a visible WARNING (not silent INFO success).
    # Asserting the WARNING keeps the old guard from being re-introduced
    # as a silent path that would still pass coordinate checks.
    assert not operators._SESSIONS, "crossing selection must leave no live session"
    dv = len(bm.verts) - STATE["baseline"][0]
    assert dv in (0, 1, 2), f"expected a native-only rip at most, got dv={dv}"
    if dv:
        assert vertex_multiset(bm) != mirrored_multiset(bm), "crossing rip must not be mirrored"
        warnings = warning_messages()
        assert warnings, (
            f"crossing with native-only change (dv={dv}) must report WARNING, got {list(operators._FINISH_REPORTS)}"
        )
        assert any(
            "not mirrored" in message or "Rip was not mirrored" in message or "partial" in message.lower()
            for message in warnings
        ), warnings
    assert_layers_removed(bm)


def verify_bothsides_mirrored(bm):
    # DISJOINT both-sides selection: the session runs.  Native rips only the
    # connected +X path under the cursor; the far -X selected vertex stays
    # untouched (same heuristic as the multi-island case), and the mirror
    # reproduces exactly what native ripped, so the result is X-symmetric.
    dv = len(bm.verts) - STATE["baseline"][0]
    assert dv == 4, f"expected 4 new vertices (2 source + 2 mirror), got {dv}"
    assert vertex_multiset(bm)[coordinate_key((-2.0, -1.0, 0.0))] == 1, "far-side vertex must stay unripped"
    assert_x_symmetric(bm)
    assert_layers_removed(bm)


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
            run_case("singlevert", [(4, 2)], (3.6, 2), verify_singlevert),
            run_case("boundary", [(4, 0), (4, 1), (4, 2)], (3.6, 1), verify_boundary),
            run_case(
                "asymmetry",
                [(4, 1), (4, 2)],
                (3.6, 2),
                verify_asymmetry_declined,
                mutate=displace_mirror_vertex,
            ),
            run_case(
                "rollback",
                [(4, 1), (4, 2)],
                (3.6, 2),
                verify_rollback,
                mutate=lambda bm: install_failure_injection(),
            ),
            run_case("bothsides", [(4, 1), (4, 2), (1, 1)], (3.6, 2), verify_bothsides_mirrored),
            # crossing must stay LAST: its native rip fails by design
            # ("Rip failed" on a disconnected mirror pair), and a failed
            # native rip leaves the simulated V route unable to start the
            # next rip in this event-simulate environment (measured, 4.2).
            run_case("crossing", [(4, 2), (2, 2)], (3.6, 2), verify_crossing_passthrough),
        ]
        run_all(cases)
    except BaseException:
        fail()
    return None


bpy.app.timers.register(start_test, first_interval=0.3)
