# SPDX-License-Identifier: GPL-3.0-or-later

"""GUI regression for rip midplane Wave 2 (contract §6-2, v4.3).

Run with Blender's real window/event loop::

    blender --factory-startup --enable-event-simulate --no-window-focus \
        --disable-crash-handler -p 40 40 960 600 --python test_rip_onplane.py

Even NX=8 → on-plane column at i=4 (x=0).  Cases rebuild the mesh each time
and push an undo baseline.  The native-failing mixed-region case is last —
its error popup can eat the next case's key press.

T-2 positive: an isolated YZ two-quad sheet (side view) so native 2-way-splits
the shared on-plane edge; a grid-attached YZ triangle at a hub 3-way-splits
and never reaches the named fin decline.

T-2 negative (C-1): ``self_face_straddle`` attaches a geometrically self-mirrored
hexagon whose off-plane corners sit outside the origin one-ring.  Sharing a
midplane grid edge would be non-manifold (already 2 faces); attaching only at
the hub makes native emit no seam edge.  Generic decline or accept are both
allowed — the fin wording must not appear.
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
from ydd_symmetric_edit import history, layer_names, operators, rip  # noqa: E402

MARKER_OK = "YSE_RIP_ONPLANE_OK"
MARKER_FAILED = "YSE_RIP_ONPLANE_TEST_FAILED"
NX, NY = 8, 4
PRECISION = 5
TEST_VID_LAYER = "yse_test_vid"
TEST_EID_LAYER = "yse_test_eid"
TEST_ROLE_LAYER = "yse_test_mirror_role"
HINT = "native kept; mirror manually or undo"
_ONLY = os.environ.get("YSE_RIP_ONPLANE_ONLY", "")
STATE = {}


def fail(message=""):
    if message:
        print(f"YSE_RIP_ONPLANE_ERROR={message}", flush=True)
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
    region_3d.view_distance = 10.0
    region_3d.update()
    area.spaces.active.shading.show_xray = True


def override():
    return bpy.context.temp_override(window=STATE["window"], area=STATE["area"], region=STATE["region"])


def window_coordinate(coordinate):
    region = STATE["region"]
    region_3d = STATE["area"].spaces.active.region_3d
    local = view3d_utils.location_3d_to_region_2d(region, region_3d, Vector(coordinate))
    if local is None:
        raise RuntimeError(f"could not project {coordinate}")
    return int(round(region.x + local.x)), int(round(region.y + local.y))


def session_tolerance():
    return float(bpy.context.scene.ydd_symmetric_edit.tolerance)


def grid_xy(i, j, nx=None, ny=None):
    nx = NX if nx is None else nx
    ny = NY if ny is None else ny
    return (i - nx / 2, j - ny / 2, 0.0)


def build_grid(name, *, nx=NX, ny=NY, with_uv=False):
    for old in tuple(bpy.data.objects):
        bpy.data.objects.remove(old, do_unlink=True)
    for old_mesh in tuple(bpy.data.meshes):
        if old_mesh.users == 0:
            bpy.data.meshes.remove(old_mesh)
    mesh = bpy.data.meshes.new(f"YSE_RipOnplane_{name}")
    coords, faces = [], []
    for j in range(ny + 1):
        for i in range(nx + 1):
            coords.append(grid_xy(i, j, nx, ny))
    stride = nx + 1
    for j in range(ny):
        for i in range(nx):
            a = j * stride + i
            faces.append((a, a + 1, a + stride + 1, a + stride))
    mesh.from_pydata(coords, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(f"YSE_RipOnplaneObj_{name}", mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.use_mesh_mirror_x = True
    obj.use_mesh_mirror_y = False
    obj.use_mesh_mirror_z = False
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    with override():
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.context.tool_settings.mesh_select_mode = (True, False, False)
    if with_uv:
        bm = bmesh.from_edit_mesh(obj.data)
        bm.loops.layers.uv.new("UVMap")
        uv = bm.loops.layers.uv.get("UVMap")
        for face in bm.faces:
            face.smooth = True
            for loop in face.loops:
                loop[uv].uv = (loop.vert.co.x * 0.1 + 0.5, loop.vert.co.y * 0.1 + 0.5)
        bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
    with override():
        bpy.ops.ed.undo_push(message=f"YSE rip onplane baseline {name}")
    STATE["object"] = obj
    STATE["nx"] = nx
    STATE["ny"] = ny
    return obj


def _build_custom_mesh(name, coords, faces):
    for old in tuple(bpy.data.objects):
        bpy.data.objects.remove(old, do_unlink=True)
    for old_mesh in tuple(bpy.data.meshes):
        if old_mesh.users == 0:
            bpy.data.meshes.remove(old_mesh)
    mesh = bpy.data.meshes.new(f"YSE_RipOnplane_{name}")
    mesh.from_pydata(coords, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(f"YSE_RipOnplaneObj_{name}", mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.use_mesh_mirror_x = True
    obj.use_mesh_mirror_y = False
    obj.use_mesh_mirror_z = False
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    with override():
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.context.tool_settings.mesh_select_mode = (True, False, False)
        bpy.ops.ed.undo_push(message=f"YSE rip onplane baseline {name}")
    STATE["object"] = obj
    STATE["nx"] = NX
    STATE["ny"] = NY
    return obj


def build_mixed_l(name):
    """L mixed-disjoint: on-plane hub, two east triangles sharing A–E, mirrored west."""

    # No on-plane column edge, so S ∩ ρ(S) = ∅ after ripping A.  Two triangles
    # share A–E / A–W so native Rip can emit a real seam edge.
    coords = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 0.6, 0.0),
        (1.0, -0.6, 0.0),
        (-1.0, 0.0, 0.0),
        (-1.0, 0.6, 0.0),
        (-1.0, -0.6, 0.0),
    ]
    faces = [(0, 1, 2), (0, 3, 1), (0, 5, 4), (0, 4, 6)]
    return _build_custom_mesh(name, coords, faces)


def build_mixed_t(name):
    """T mixed-disjoint: L plus a north stem triangle on each side."""

    coords = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 0.6, 0.0),
        (1.0, -0.6, 0.0),
        (0.25, 0.9, 0.0),
        (-1.0, 0.0, 0.0),
        (-1.0, 0.6, 0.0),
        (-1.0, -0.6, 0.0),
        (-0.25, 0.9, 0.0),
    ]
    faces = [(0, 1, 2), (0, 3, 1), (0, 2, 4), (0, 6, 5), (0, 5, 7), (0, 8, 6)]
    return _build_custom_mesh(name, coords, faces)


def build_mixed_boundary(name):
    """Boundary-end mixed-disjoint: on-plane hub on the −Y mesh end."""

    coords = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 0.6, 0.0),
        (0.25, 0.6, 0.0),
        (-1.0, 0.0, 0.0),
        (-1.0, 0.6, 0.0),
        (-0.25, 0.6, 0.0),
    ]
    faces = [(0, 1, 2), (0, 2, 3), (0, 5, 4), (0, 6, 5)]
    return _build_custom_mesh(name, coords, faces)


def grid_vert(bm, i, j):
    target = Vector(grid_xy(i, j, STATE["nx"], STATE["ny"]))
    for vertex in bm.verts:
        if (vertex.co - target).length < 1e-4:
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


def select_coords(bm, coords):
    for face in bm.faces:
        face.select = False
    for edge in bm.edges:
        edge.select = False
    for vertex in bm.verts:
        vertex.select = False
    bm.select_history.clear()
    remaining = [Vector(co) for co in coords]
    for vertex in bm.verts:
        for target in remaining:
            if (vertex.co - target).length < 1e-4:
                vertex.select = True
                remaining.remove(target)
                break
    if remaining:
        raise AssertionError(f"spoke verts not found: {remaining}")
    bm.select_flush_mode()


def coordinate_key(co, precision=PRECISION):
    return tuple(round(float(value), precision) for value in co)


def mirror_key(co, precision=PRECISION):
    return coordinate_key((-co[0], co[1], co[2]), precision)


def vertex_multiset(bm):
    return Counter(coordinate_key(vertex.co) for vertex in bm.verts)


def mirrored_vertex_multiset(bm):
    return Counter(mirror_key(vertex.co) for vertex in bm.verts)


def _normalize_cycle(coords):
    rotations = [tuple(coords[i:] + coords[:i]) for i in range(len(coords))]
    return min(rotations)


def face_incidence_multiset(bm):
    keys = []
    for face in bm.faces:
        coords = [coordinate_key(vertex.co) for vertex in face.verts]
        keys.append(_normalize_cycle(coords))
    return Counter(keys)


def mirrored_face_incidence_multiset(bm):
    keys = []
    for face in bm.faces:
        mirrored = [mirror_key(vertex.co) for vertex in face.verts]
        mirrored.reverse()
        keys.append(_normalize_cycle(mirrored))
    return Counter(keys)


def edge_incidence_multiset(bm):
    keys = []
    for edge in bm.edges:
        ends = [coordinate_key(vertex.co) for vertex in edge.verts]
        keys.append(tuple(sorted(ends)))
    return Counter(keys)


def mirrored_edge_incidence_multiset(bm):
    keys = []
    for edge in bm.edges:
        ends = [mirror_key(vertex.co) for vertex in edge.verts]
        keys.append(tuple(sorted(ends)))
    return Counter(keys)


def duplicated_test_vids(bm):
    layer = bm.verts.layers.int.get(TEST_VID_LAYER)
    assert layer is not None, "test vertex id layer missing"
    groups: dict[int, list] = {}
    for vertex in bm.verts:
        vertex_id = int(vertex[layer])
        if vertex_id > 0:
            groups.setdefault(vertex_id, []).append(vertex)
    return frozenset(vertex_id for vertex_id, group in groups.items() if len(group) > 1)


def _named_warnings(*needles):
    warnings = warning_messages()
    for needle in needles:
        if any(needle in message for message in warnings):
            return warnings
    raise AssertionError(f"expected WARNING containing {needles!r}, got {warnings}")


def install_preflight_capture(bm):
    del bm
    original = rip.preflight_reason

    def wrapped(edit_bm, snapshot, mirror_face_ids):
        STATE["preflight_counts"] = topology_counts(edit_bm)
        STATE["preflight_coords"] = vertex_multiset(edit_bm)
        STATE["preflight_selected"] = selected_multiset(edit_bm)
        return original(edit_bm, snapshot, mirror_face_ids)

    STATE["original_preflight"] = original
    rip.preflight_reason = wrapped


def restore_preflight_capture():
    original = STATE.pop("original_preflight", None)
    if original is not None:
        rip.preflight_reason = original


def assert_native_exact_kept(bm, label=""):
    prefix = f"{label}: " if label else ""
    counts = STATE.get("preflight_counts")
    coords = STATE.get("preflight_coords")
    selected = STATE.get("preflight_selected")
    assert counts is not None, f"{prefix}preflight native snapshot missing"
    assert topology_counts(bm) == counts, f"{prefix}native vertex/edge/face counts changed"
    assert vertex_multiset(bm) == coords, f"{prefix}native coordinate multiset changed"
    assert selected_multiset(bm) == selected, f"{prefix}native selection changed"


def assert_x_symmetric_coords(bm, label=""):
    prefix = f"{label}: " if label else ""
    verts = vertex_multiset(bm)
    assert verts == mirrored_vertex_multiset(bm), f"{prefix}vertex coords not X-symmetric: {verts}"


def assert_x_symmetric_full(bm, label=""):
    assert_x_symmetric_coords(bm, label)
    prefix = f"{label}: " if label else ""
    faces = face_incidence_multiset(bm)
    mirrored_faces = mirrored_face_incidence_multiset(bm)
    assert faces == mirrored_faces, f"{prefix}face incidence not X-symmetric: {faces} vs {mirrored_faces}"


def assert_layers_removed(bm):
    assert bm.verts.layers.int.get(layer_names.VERT_SESSION_ID_LAYER) is None, "rip vertex layer leaked"
    assert bm.edges.layers.int.get(layer_names.EDGE_ORIGINAL_LAYER) is None, "edge layer leaked"
    assert bm.faces.layers.int.get(layer_names.FACE_ID_LAYER) is None, "face layer leaked"


def topology_counts(bm):
    return len(bm.verts), len(bm.edges), len(bm.faces)


def selected_multiset(bm):
    return Counter(coordinate_key(vertex.co) for vertex in bm.verts if vertex.select)


def warning_messages():
    return [message for kind, message in operators._FINISH_REPORTS if kind == "WARNING"]


def info_messages():
    return [message for kind, message in operators._FINISH_REPORTS if kind == "INFO"]


def assert_disposition_hint():
    warnings = warning_messages()
    assert any(HINT in message for message in warnings), warnings


def stamp_identity_layers(bm):
    for name in (TEST_VID_LAYER, TEST_EID_LAYER, TEST_ROLE_LAYER):
        existing = bm.verts.layers.int.get(name) if name != TEST_EID_LAYER else bm.edges.layers.int.get(name)
        if existing is not None:
            if name == TEST_EID_LAYER:
                bm.edges.layers.int.remove(existing)
            else:
                bm.verts.layers.int.remove(existing)
    vid = bm.verts.layers.int.new(TEST_VID_LAYER)
    eid = bm.edges.layers.int.new(TEST_EID_LAYER)
    role = bm.verts.layers.int.new(TEST_ROLE_LAYER)
    for index, vertex in enumerate(bm.verts, start=1):
        vertex[vid] = index
        vertex[role] = index
    for index, edge in enumerate(bm.edges, start=1):
        edge[eid] = index


def duplicated_edge_groups(bm):
    layer = bm.edges.layers.int.get(TEST_EID_LAYER)
    assert layer is not None, "test edge id layer missing"
    groups: dict[int, list] = {}
    for edge in bm.edges:
        edge_id = int(edge[layer])
        if edge_id > 0:
            groups.setdefault(edge_id, []).append(edge)
    return {edge_id: group for edge_id, group in groups.items() if len(group) > 1}


def assert_dup_edge_identity(bm, expected_count, label=""):
    prefix = f"{label}: " if label else ""
    vid = bm.verts.layers.int.get(TEST_VID_LAYER)
    assert vid is not None, f"{prefix}test vertex id layer missing"
    groups = duplicated_edge_groups(bm)
    assert len(groups) == expected_count, f"{prefix}expected {expected_count} duplicated edge ids, got {sorted(groups)}"
    for edge_id, group in groups.items():
        assert len(group) == 2, f"{prefix}edge {edge_id} split into {len(group)}"
        ends = [tuple(sorted(int(vertex[vid]) for vertex in edge.verts)) for edge in group]
        assert ends[0] == ends[1] and 0 not in ends[0], f"{prefix}duplicated edge {edge_id} copies disagree: {ends}"


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
                    raise RuntimeError(
                        f"rip flow never settled; modal={[op.bl_idname for op in STATE['window'].modal_operators]} "
                        f"sessions={list(operators._SESSIONS)}"
                    )
                return 0.1
            bpy.app.timers.register(done, first_interval=0.25)
        except BaseException:
            fail()
        return None

    bpy.app.timers.register(poll, first_interval=0.1)


def rip_events(cursor_xyz, drag=(50, 0), *, alt=False, confirm="LMB"):
    x, y = window_coordinate(cursor_xyz)
    events = [{"type": "MOUSEMOVE", "value": "NOTHING", "x": x, "y": y}]
    if alt:
        events.append({"type": "LEFT_ALT", "value": "PRESS", "x": x, "y": y})
        events.append({"type": "V", "value": "PRESS", "x": x, "y": y, "alt": True})
        events.append({"type": "V", "value": "RELEASE", "x": x, "y": y, "alt": True})
        events.append({"type": "LEFT_ALT", "value": "RELEASE", "x": x, "y": y})
    else:
        events.append({"type": "V", "value": "PRESS", "x": x, "y": y})
        events.append({"type": "V", "value": "RELEASE", "x": x, "y": y})
    tx, ty = x + drag[0], y + drag[1]
    if drag != (0, 0):
        events.append({"type": "MOUSEMOVE", "value": "NOTHING", "x": tx, "y": ty})
    if confirm == "LMB":
        events.append({"type": "LEFTMOUSE", "value": "PRESS", "x": tx, "y": ty})
        events.append({"type": "LEFTMOUSE", "value": "RELEASE", "x": tx, "y": ty})
    else:
        events.append({"type": "ESC", "value": "PRESS", "x": tx, "y": ty})
        events.append({"type": "ESC", "value": "RELEASE", "x": tx, "y": ty})
    return events


def run_case(
    name,
    select_ij,
    cursor_xyz,
    verify,
    *,
    drag=(50, 0),
    alt=False,
    confirm="LMB",
    mutate=None,
    builder=None,
    nx=NX,
    ny=NY,
    with_uv=False,
    select_xyz=None,
):
    def start(next_case):
        try:
            print(f"YSE_RIP_ONPLANE_CASE={name}", flush=True)
            if builder is not None:
                obj = builder(name)
            else:
                obj = build_grid(name, nx=nx, ny=ny, with_uv=with_uv)
            bm = bmesh.from_edit_mesh(obj.data)
            if select_xyz is not None:
                select_coords(bm, select_xyz)
            else:
                select_verts(bm, select_ij)
            if mutate is not None:
                mutate(bm)
            STATE["baseline"] = topology_counts(bm)
            STATE["case_name"] = name
            bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
            operators._FINISH_REPORTS.clear()

            def settled():
                try:
                    STATE["next_case"] = next_case
                    bm2 = bmesh.from_edit_mesh(STATE["object"].data)
                    if verify(bm2) != "ASYNC":
                        next_case()
                except BaseException:
                    fail()

            send_events(
                rip_events(cursor_xyz, drag=drag, alt=alt, confirm=confirm),
                lambda: wait_settled(settled),
            )
        except BaseException:
            fail()

    return start


def _assert_source_bank_selected(bm, label=""):
    selected = [vertex for vertex in bm.verts if vertex.select]
    assert selected, f"{label}: expected the source bank to remain selected"
    selected_keys = Counter(coordinate_key(vertex.co) for vertex in selected)
    mirrored_selected = Counter(mirror_key(vertex.co) for vertex in selected)
    assert selected_keys != mirrored_selected or all(
        abs(vertex.co.x) < 1e-4 or vertex_multiset(bm)[coordinate_key(vertex.co)] >= 2 for vertex in selected
    ), f"{label}: selection should cover only the source bank"


def verify_vert_west_west(bm):
    dv = len(bm.verts) - STATE["baseline"][0]
    de = len(bm.edges) - STATE["baseline"][1]
    assert dv == 1, f"vert_ww: expected dv=1, got {dv}"
    assert de == 2, f"vert_ww: expected de=2, got {de}"
    assert_x_symmetric_coords(bm, "vert_ww")
    assert_layers_removed(bm)
    _assert_source_bank_selected(bm, "vert_ww")
    assert_dup_edge_identity(bm, 2, "vert_ww")
    assert any("Mirrored Rip" in message for message in info_messages()), info_messages()
    STATE["plus_coords"] = vertex_multiset(bm)
    STATE["plus_selected"] = selected_multiset(bm)
    STATE["plus_edges"] = edge_incidence_multiset(bm)


def verify_vert_east_east(bm):
    dv = len(bm.verts) - STATE["baseline"][0]
    assert dv == 1, f"vert_ee: expected dv=1, got {dv}"
    assert_x_symmetric_coords(bm, "vert_ee")
    assert_layers_removed(bm)
    _assert_source_bank_selected(bm, "vert_ee")
    assert_dup_edge_identity(bm, 2, "vert_ee")
    plus = STATE.get("plus_coords")
    assert plus is not None, "vert_ww must run before vert_ee"
    minus = vertex_multiset(bm)
    mirrored_minus = Counter((-key[0], key[1], key[2]) for key in minus.elements())
    assert mirrored_minus == plus, f"equivariance coords failed: ρ(east)={mirrored_minus} vs west={plus}"
    plus_selected = STATE.get("plus_selected")
    minus_selected = selected_multiset(bm)
    mirrored_selected = Counter((-key[0], key[1], key[2]) for key in minus_selected.elements())
    assert mirrored_selected == plus_selected, (
        f"equivariance selection failed: ρ(east sel)={mirrored_selected} vs west sel={plus_selected}"
    )
    plus_edges = STATE.get("plus_edges")
    assert plus_edges is not None, "vert_ww must store edge incidence"
    minus_edges = mirrored_edge_incidence_multiset(bm)
    assert minus_edges == plus_edges, f"equivariance edges failed: ρ(east)={minus_edges} vs west={plus_edges}"


def verify_vert_cross(bm):
    dv = len(bm.verts) - STATE["baseline"][0]
    assert dv == 1, f"{STATE['case_name']}: expected dv=1, got {dv}"
    assert_x_symmetric_coords(bm, STATE["case_name"])
    assert_layers_removed(bm)
    _assert_source_bank_selected(bm, STATE["case_name"])
    assert_dup_edge_identity(bm, 2, STATE["case_name"])


def verify_edge_or_path(bm):
    name = STATE["case_name"]
    expected = STATE["expected_dv_de"]
    dv = len(bm.verts) - STATE["baseline"][0]
    de = len(bm.edges) - STATE["baseline"][1]
    assert dv == expected[0], f"{name}: expected dv={expected[0]}, got {dv}"
    assert de == expected[1], f"{name}: expected de={expected[1]}, got {de}"
    assert_x_symmetric_coords(bm, name)
    assert_layers_removed(bm)
    _assert_source_bank_selected(bm, name)
    assert_dup_edge_identity(bm, expected[1], name)
    pinned = STATE.get("expect_pinned") or ()
    counts = vertex_multiset(bm)
    for key in pinned:
        assert counts[key] == 1, f"{name}: pinned endpoint {key} was split or moved ({counts[key]})"
        matches = [vertex for vertex in bm.verts if coordinate_key(vertex.co) == key]
        assert matches and not matches[0].select, f"{name}: pinned endpoint {key} should stay unselected"


def verify_zero_or_esc(bm):
    name = STATE["case_name"]
    dv = len(bm.verts) - STATE["baseline"][0]
    assert dv == 1, f"{name}: expected dv=1, got {dv}"
    seam = coordinate_key((0.0, 0.0, 0.0))
    assert vertex_multiset(bm)[seam] == 2, f"{name}: expected 2 verts at origin, got {vertex_multiset(bm)[seam]}"
    selected = selected_multiset(bm)
    assert selected[seam] == 1, f"{name}: expected exactly 1 selected copy at origin, got {selected[seam]}"
    assert_x_symmetric_coords(bm, name)
    assert_layers_removed(bm)


def shift_onplane_column(bm):
    stamp_identity_layers(bm)
    epsilon = 0.5 * session_tolerance()
    STATE["band_eps"] = epsilon
    STATE["band_pins"] = []
    STATE["band_selected_pins"] = []
    for vertex in bm.verts:
        if abs(vertex.co.x) <= 1.0e-8:
            vertex.co.x = epsilon
            key = coordinate_key(vertex.co)
            if vertex.select:
                STATE["band_selected_pins"].append(key)
            else:
                STATE["band_pins"].append(key)


def verify_tol_band(bm):
    name = STATE["case_name"]
    expected_dv = STATE.get("band_expected_dv", 1)
    expected_de = STATE.get("band_expected_de", 2)
    dv = len(bm.verts) - STATE["baseline"][0]
    de = len(bm.edges) - STATE["baseline"][1]
    assert dv == expected_dv, f"{name}: expected dv={expected_dv}, got {dv}"
    assert de == expected_de, f"{name}: expected de={expected_de}, got {de}"
    counts = vertex_multiset(bm)
    selected_keys = STATE["band_selected_pins"]
    assert selected_keys, f"{name}: no selected band vertices recorded"
    for key in selected_keys:
        assert counts[key] == 2, f"{name}: selected {key} should stay colocated, count={counts[key]}"
        reflected = coordinate_key((-key[0], key[1], key[2]))
        assert counts[reflected] == 0, f"{name}: identity must not reflect {key} to {reflected}"
    for key in STATE["band_pins"]:
        assert counts[key] == 1, f"{name}: pin {key} changed, count={counts[key]}"
        matches = [vertex for vertex in bm.verts if coordinate_key(vertex.co) == key]
        assert matches and not matches[0].select, f"{name}: pin {key} should stay unselected"
    assert_dup_edge_identity(bm, expected_de, name)
    assert_layers_removed(bm)


def install_inband_source_snap(bm):
    stamp_identity_layers(bm)
    half = 0.5 * session_tolerance()
    STATE["inband_half"] = half
    vid = bm.verts.layers.int.get(TEST_VID_LAYER)
    pre_coords = {int(vertex[vid]): coordinate_key(vertex.co) for vertex in bm.verts}
    STATE["inband_pre_coords"] = pre_coords
    mirror_of = {}
    leftover = dict(pre_coords)
    for vertex_id, key in pre_coords.items():
        if vertex_id in mirror_of:
            continue
        target = (-key[0], key[1], key[2])
        partner = next((other for other, other_key in leftover.items() if other_key == target), None)
        assert partner is not None, f"aa_inband: no pre-rip mirror for vid {vertex_id} at {key}"
        mirror_of[vertex_id] = partner
        mirror_of[partner] = vertex_id
        leftover.pop(vertex_id, None)
        leftover.pop(partner, None)
    STATE["inband_mirror_of"] = mirror_of
    original = rip.apply_mirrored_rip

    def snapped(edit_bm, snapshot, mirror_face_ids):
        for vertex in edit_bm.verts:
            if vertex.select:
                vertex.co.x = half
        return original(edit_bm, snapshot, mirror_face_ids)

    STATE["original_apply_inband"] = original
    rip.apply_mirrored_rip = snapped


def verify_aa_prime_inband(bm):
    try:
        dv = len(bm.verts) - STATE["baseline"][0]
        assert dv == 2, f"aa_inband: expected dv=2, got {dv}"
        vid = bm.verts.layers.int.get(TEST_VID_LAYER)
        assert vid is not None, "aa_inband: test vertex id layer missing"
        groups: dict[int, list] = {}
        for vertex in bm.verts:
            vertex_id = int(vertex[vid])
            if vertex_id > 0:
                groups.setdefault(vertex_id, []).append(vertex)
        ripped = {vertex_id: group for vertex_id, group in groups.items() if len(group) == 2}
        assert ripped, "aa_inband: no duplicated seam vertices"
        mirror_of = STATE["inband_mirror_of"]
        source_of = {}
        nonsource_of = {}
        for vertex_id, group in ripped.items():
            selected = [vertex for vertex in group if vertex.select]
            idle = [vertex for vertex in group if not vertex.select]
            assert len(selected) == 1 and len(idle) == 1, f"aa_inband: vid {vertex_id} banks {selected!r} / {idle!r}"
            source_of[vertex_id] = selected[0]
            nonsource_of[vertex_id] = idle[0]
        for vertex_id in source_of:
            partner = mirror_of[vertex_id]
            assert partner in source_of, f"aa_inband: source({vertex_id}) has no ripped mirror {partner}"
            expected = coordinate_key((-source_of[partner].co.x, source_of[partner].co.y, source_of[partner].co.z))
            actual = coordinate_key(nonsource_of[vertex_id].co)
            assert actual == expected, f"aa_inband: nonsource({vertex_id})={actual} != ρ(source({partner}))={expected}"
        assert_layers_removed(bm)
    finally:
        if "original_apply_inband" in STATE:
            rip.apply_mirrored_rip = STATE["original_apply_inband"]


def verify_b_form(bm):
    try:
        warnings = _named_warnings("source banks have inconsistent face-sides across the seam")
        assert_disposition_hint()
        assert_native_exact_kept(bm, "b_form")
        assert_layers_removed(bm)
        print("YSE_RIP_ONPLANE_B_FORM=constructed", flush=True)
        print(f"YSE_RIP_ONPLANE_B_FORM_WARNING={warnings}", flush=True)
        STATE["b_form"] = "constructed"
    finally:
        restore_preflight_capture()


def _source_loop_uv(vertex, uv_layer, fill_faces):
    for face in vertex.link_faces:
        if face in fill_faces:
            continue
        for loop in face.loops:
            if loop.vert is vertex:
                return tuple(round(float(value), 6) for value in loop[uv_layer].uv)
    return None


def verify_fill(bm):
    name = STATE["case_name"]
    expected_df = STATE["expected_df"]
    expected_dv, expected_de = STATE["expected_dv_de"]
    dv = len(bm.verts) - STATE["baseline"][0]
    de = len(bm.edges) - STATE["baseline"][1]
    df = len(bm.faces) - STATE["baseline"][2]
    assert dv == expected_dv, f"{name}: expected dv={expected_dv}, got {dv}"
    assert de == expected_de, f"{name}: expected de={expected_de}, got {de}"
    assert df == expected_df, f"{name}: expected df={expected_df}, got {df}"
    assert_x_symmetric_full(bm, name)
    assert_layers_removed(bm)
    uv = bm.loops.layers.uv.get("UVMap")
    assert uv is not None, f"{name}: UV layer disappeared"
    fill_faces = []
    selected = {vertex for vertex in bm.verts if vertex.select}
    selected_keys = {coordinate_key(vertex.co) for vertex in selected}
    nonsource_keys = {mirror_key(key) for key in selected_keys}
    bank_keys = selected_keys | nonsource_keys
    for face in bm.faces:
        corner_keys = [coordinate_key(vertex.co) for vertex in face.verts]
        if not all(key in bank_keys or abs(key[0]) <= 1.0e-4 for key in corner_keys):
            continue
        n_selected = sum(1 for vertex in face.verts if vertex in selected)
        n_nonsource = sum(1 for vertex in face.verts if coordinate_key(vertex.co) in nonsource_keys)
        if n_selected >= 1 and n_nonsource >= 1:
            fill_faces.append(face)
            assert face.smooth, f"{name}: fill face lost smooth flag"
    assert len(fill_faces) == expected_df, f"{name}: expected {expected_df} bank-bridge fills, got {len(fill_faces)}"
    fill_set = set(fill_faces)
    for face in fill_faces:
        for loop in face.loops:
            actual = tuple(round(float(value), 6) for value in loop[uv].uv)
            expected = _source_loop_uv(loop.vert, uv, fill_set)
            assert expected is not None, f"{name}: fill loop at {coordinate_key(loop.vert.co)} has no source UV"
            assert actual == expected, f"{name}: fill UV {actual} != source loop UV {expected}"


def verify_boundary(bm):
    dv = len(bm.verts) - STATE["baseline"][0]
    assert dv == 1, f"boundary: expected dv=1, got {dv}"
    assert_x_symmetric_coords(bm, "boundary")
    assert_layers_removed(bm)
    _assert_source_bank_selected(bm, "boundary")


def verify_mixed_path(bm):
    dv = len(bm.verts) - STATE["baseline"][0]
    de = len(bm.edges) - STATE["baseline"][1]
    assert dv == 2, f"mixed_path: expected native dv=2, got {dv}"
    assert de == 3, f"mixed_path: expected native de=3, got {de}"
    assert vertex_multiset(bm) != mirrored_vertex_multiset(bm), "mixed_path must keep the native (asymmetric) result"
    assert_disposition_hint()
    assert_layers_removed(bm)
    selected = [vertex for vertex in bm.verts if vertex.select]
    on_plane_moved = [vertex for vertex in selected if abs(vertex.co.x) <= 1.0e-4]
    assert on_plane_moved, "mixed_path: expected a selected on-plane copy"
    for vertex in on_plane_moved:
        sides = set()
        for face in vertex.link_faces:
            center_x = sum(corner.co.x for corner in face.verts) / len(face.verts)
            if center_x < -1.0e-4:
                sides.add("west")
            elif center_x > 1.0e-4:
                sides.add("east")
        assert sides == {"west", "east"}, f"mixed_path: moving on-plane copy must keep both sides, got {sides}"


def verify_decline_native_kept(bm):
    name = STATE["case_name"]
    assert vertex_multiset(bm) != mirrored_vertex_multiset(bm) or len(bm.verts) > STATE["baseline"][0], (
        f"{name}: expected a native rip to keep"
    )
    assert len(bm.verts) >= STATE["baseline"][0], f"{name}: native result disappeared"
    assert_disposition_hint()
    assert_layers_removed(bm)


def build_grid_with_fin(name):
    """Two on-plane YZ quads sharing an interior edge.

    A grid-attached YZ triangle at a hub 3-way-splits and never reaches
    ``_dup_has_self_face``.  Ripping the shared YZ edge is a 2-way split
    whose only incident faces are self-partner fins.
    """

    coords = [
        (0.0, -0.5, 0.0),
        (0.0, 0.5, 0.0),
        (0.0, 0.5, 1.0),
        (0.0, -0.5, 1.0),
        (0.0, 0.5, 2.0),
        (0.0, -0.5, 2.0),
    ]
    return _build_custom_mesh(name, coords, [(0, 1, 2, 3), (3, 2, 4, 5)])


def add_midplane_fin(bm):
    stamp_identity_layers(bm)
    install_preflight_capture(bm)
    region_3d = STATE["area"].spaces.active.region_3d
    region_3d.view_perspective = "ORTHO"
    region_3d.view_rotation = Quaternion((0.5, 0.5, 0.5, 0.5))
    region_3d.view_location = (0.0, 0.0, 1.0)
    region_3d.view_distance = 8.0
    region_3d.update()


def build_grid_with_straddle_self_face(name):
    """Grid plus an X-self-mirrored hexagon with off-plane corners outside the one-ring."""

    obj = build_grid(name)
    bm = bmesh.from_edit_mesh(obj.data)
    origin = grid_vert(bm, 4, 2)
    plus = bm.verts.new((0.0, 0.3, 0.5))
    east = bm.verts.new((2.0, 1.0, 1.0))
    north = bm.verts.new((0.0, 2.0, 1.5))
    west = bm.verts.new((-2.0, 1.0, 1.0))
    minus = bm.verts.new((0.0, -0.3, 0.5))
    bm.faces.new((origin, plus, east, north, west, minus))
    bm.verts.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    bm.normal_update()
    bmesh.update_edit_mesh(obj.data, loop_triangles=True, destructive=True)
    with override():
        bpy.ops.ed.undo_push(message="YSE rip onplane straddle self face")
    return obj


def verify_self_face(bm):
    try:
        print(
            f"YSE_RIP_ONPLANE_SELF_FACE_DELTA=now={topology_counts(bm)} "
            f"baseline={STATE['baseline']} warnings={warning_messages()} infos={info_messages()}",
            flush=True,
        )
        warnings = _named_warnings("self-mirrored face")
        assert_disposition_hint()
        assert_native_exact_kept(bm, "self_face")
        assert_layers_removed(bm)
        print("YSE_RIP_ONPLANE_SELF_FACE=named", flush=True)
        print(f"YSE_RIP_ONPLANE_SELF_FACE_WARNING={warnings}", flush=True)
    finally:
        restore_preflight_capture()
        configure_view(STATE["area"])


def verify_self_face_straddle(bm):
    try:
        warnings = warning_messages()
        assert not any("self-mirrored face" in message for message in warnings), (
            f"self_face_straddle: straddling self-partner face must not use the fin decline: {warnings}"
        )
        if warnings:
            assert_disposition_hint()
            assert_native_exact_kept(bm, "self_face_straddle")
        print(f"YSE_RIP_ONPLANE_SELF_FACE_STRADDLE={warnings or ['accepted']}", flush=True)
        assert_layers_removed(bm)
    finally:
        restore_preflight_capture()


def verify_mixed_disjoint(bm):
    name = STATE["case_name"]
    expected = STATE["expected_mixed"]
    dv = len(bm.verts) - STATE["baseline"][0]
    de = len(bm.edges) - STATE["baseline"][1]
    dups = duplicated_test_vids(bm)
    warnings = warning_messages()
    print(
        f"YSE_RIP_ONPLANE_MIXED={name} dv={dv} de={de} dups={sorted(dups)} warnings={warnings}",
        flush=True,
    )
    assert_disposition_hint()
    assert_layers_removed(bm)
    assert (dv, de) == expected["dv_de"], f"{name}: expected dv/de={expected['dv_de']}, got {(dv, de)}"
    if expected.get("reason"):
        assert any(expected["reason"] in message for message in warnings), (
            f"{name}: expected decline {expected['reason']!r}, got {warnings}"
        )
    if expected.get("dup_vids") is not None:
        assert dups == expected["dup_vids"], f"{name}: expected dups {sorted(expected['dup_vids'])}, got {sorted(dups)}"
    if not expected.get("allow_symmetric"):
        assert vertex_multiset(bm) != mirrored_vertex_multiset(bm) or dv > 0, f"{name}: expected a native rip to keep"


def verify_fill_cycle_decline(bm):
    try:
        assert STATE.get("fill_cycle_patched"), "fill_cycle: preflight input patch was not invoked"
        warnings = _named_warnings("a fill face is not closed under bank-role reflection")
        assert_disposition_hint()
        assert_native_exact_kept(bm, "fill_cycle")
        assert_layers_removed(bm)
        print(f"YSE_RIP_ONPLANE_FILL_CYCLE={warnings}", flush=True)
    finally:
        original = STATE.pop("original_fill_preflight", None)
        if original is not None:
            rip._preflight_onplane_fill = original
        restore_preflight_capture()


def verify_undo_redo(bm):
    dv = len(bm.verts) - STATE["baseline"][0]
    assert dv == 1, f"undo_redo: expected dv=1, got {dv}"
    assert_x_symmetric_coords(bm, "undo_redo")
    assert_layers_removed(bm)
    result = vertex_multiset(bm)
    selection = selected_multiset(bm)
    with override():
        undo_result = bpy.ops.ed.undo()
    assert undo_result == {"FINISHED"}, undo_result
    bm2 = bmesh.from_edit_mesh(STATE["object"].data)
    assert topology_counts(bm2) == STATE["baseline"], "undo did not restore baseline"
    with override():
        redo_result = bpy.ops.ed.redo()
    assert redo_result == {"FINISHED"}, redo_result

    def after_redo():
        try:
            bm3 = bmesh.from_edit_mesh(STATE["object"].data)
            assert vertex_multiset(bm3) == result, "redo did not restore the V-open result"
            assert selected_multiset(bm3) == selection, "redo changed source-bank selection"
            assert_layers_removed(bm3)
            STATE["next_case"]()
        except BaseException:
            fail()
        return None

    bpy.app.timers.register(after_redo, first_interval=0.6)
    return "ASYNC"


def install_native_clone(bm):
    stamp_identity_layers(bm)
    original = rip.apply_mirrored_rip

    def wrapped(edit_bm, snapshot, mirror_face_ids):
        clone = bpy.data.meshes.new("YSE_RipOnplaneNative")
        edit_bm.to_mesh(clone)
        STATE["native_clone"] = clone
        return original(edit_bm, snapshot, mirror_face_ids)

    STATE["original_apply_repair"] = original
    rip.apply_mirrored_rip = wrapped


def verify_raw_repair(bm):
    try:
        assert_x_symmetric_coords(bm, "raw_repair")
        assert_layers_removed(bm)
        result = vertex_multiset(bm)
        selection = selected_multiset(bm)
        clone = STATE.get("native_clone")
        assert clone is not None, "raw_repair: native clone missing"
        live = bmesh.from_edit_mesh(STATE["object"].data)
        if len(live.verts):
            bmesh.ops.delete(live, geom=list(live.verts), context="VERTS")
        live.from_mesh(clone)
        bmesh.update_edit_mesh(STATE["object"].data, loop_triangles=True, destructive=True)
        operators._FINISH_REPORTS.clear()
        history.repair_after_redo()
    except BaseException:
        if "original_apply_repair" in STATE:
            rip.apply_mirrored_rip = STATE["original_apply_repair"]
        raise

    def after_repair():
        try:
            bm2 = bmesh.from_edit_mesh(STATE["object"].data)
            assert vertex_multiset(bm2) == result, "raw_repair: finish replay changed the V-open result"
            assert selected_multiset(bm2) == selection, "raw_repair: selection changed"
            assert any("Mirrored Rip" in message for message in info_messages()), info_messages()
            assert_layers_removed(bm2)
        finally:
            if "original_apply_repair" in STATE:
                rip.apply_mirrored_rip = STATE["original_apply_repair"]
            clone_mesh = STATE.pop("native_clone", None)
            if clone_mesh is not None and clone_mesh.name in bpy.data.meshes and clone_mesh.users == 0:
                bpy.data.meshes.remove(clone_mesh)
        STATE["next_case"]()
        return None

    bpy.app.timers.register(after_repair, first_interval=0.6)
    return "ASYNC"


def verify_mixed_region(bm):
    dv = len(bm.verts) - STATE["baseline"][0]
    de = len(bm.edges) - STATE["baseline"][1]
    df = len(bm.faces) - STATE["baseline"][2]
    assert (dv, de, df) == (0, 0, 0), f"mixed_region: expected native no-op, got {(dv, de, df)}"
    assert_layers_removed(bm)


def raise_tolerance(bm):
    del bm
    settings = bpy.context.scene.ydd_symmetric_edit
    STATE["prev_tol"] = float(settings.tolerance)
    settings.tolerance = 0.05


def restore_tolerance():
    previous = STATE.pop("prev_tol", None)
    if previous is not None:
        bpy.context.scene.ydd_symmetric_edit.tolerance = previous


def mutate_band(bm):
    raise_tolerance(bm)
    shift_onplane_column(bm)
    STATE["band_expected_dv"] = 1
    STATE["band_expected_de"] = 2


def mutate_band_path3(bm):
    raise_tolerance(bm)
    shift_onplane_column(bm)
    STATE["band_expected_dv"] = 3
    STATE["band_expected_de"] = 4


def mutate_inband(bm):
    raise_tolerance(bm)
    install_inband_source_snap(bm)


def mutate_b_form(bm):
    install_preflight_capture(bm)


def mutate_straddle(bm):
    stamp_identity_layers(bm)
    install_preflight_capture(bm)


def mutate_fill_cycle(bm):
    STATE["expected_dv_de"] = (2, 5)
    STATE["expected_df"] = 3
    install_preflight_capture(bm)
    original = rip._preflight_onplane_fill

    def distorted(edit_bm, snapshot, derived, source_by_vid, nonsource_by_vid):
        # Role-swap is rotationally equivalent to the fill itself.  Point one
        # source slot at a different bank vertex so one corner is mis-tagged
        # and the (vid, role) cycle no longer matches its reflection.
        STATE["fill_cycle_patched"] = True
        warped_source = dict(source_by_vid)
        warped_nonsource = dict(nonsource_by_vid)
        ripped = [vertex_id for vertex_id in warped_source if vertex_id in warped_nonsource]
        if len(ripped) >= 2:
            warped_source[ripped[0]] = warped_source[ripped[1]]
        elif ripped:
            warped_source[ripped[0]] = warped_nonsource[ripped[0]]
        return original(edit_bm, snapshot, derived, warped_source, warped_nonsource)

    STATE["fill_cycle_patched"] = False
    STATE["original_fill_preflight"] = original
    rip._preflight_onplane_fill = distorted


def _mutate_mixed(expected):
    def mutate(bm):
        stamp_identity_layers(bm)
        vid = bm.verts.layers.int.get(TEST_VID_LAYER)
        origin = next(vertex for vertex in bm.verts if coordinate_key(vertex.co) == (0.0, 0.0, 0.0))
        payload = dict(expected)
        if payload.get("dup_vids") == "origin":
            payload["dup_vids"] = frozenset({int(origin[vid])})
        STATE["expected_mixed"] = payload

    return mutate


def stamp_only(bm):
    stamp_identity_layers(bm)


def run_all(cases, index=0):
    restore_tolerance()
    if index >= len(cases):
        print(f"YSE_RIP_ONPLANE_B_FORM_FINAL={STATE.get('b_form', 'unrun')}", flush=True)
        print(MARKER_OK, flush=True)
        sys.stdout.flush()
        addon.unregister()
        bpy.ops.wm.quit_blender()
        return
    cases[index](lambda: run_all(cases, index + 1))


def _filter_cases(cases):
    if not _ONLY:
        return cases
    wanted = {item.strip() for item in _ONLY.split(",") if item.strip()}
    kept = []
    for case in cases:
        name = getattr(case, "yse_name", "")
        if name in wanted:
            kept.append(case)
    if not kept:
        raise RuntimeError(f"YSE_RIP_ONPLANE_ONLY={_ONLY!r} matched no cases")
    return kept


def start_test():
    try:
        addon.register()
        addon.sync_persistent_keymap(True)
        window, area, region = viewport_context()
        configure_view(area)
        STATE.update(window=window, area=area, region=region)

        seam = (4, 2)
        cursor_w = (-0.35, 0.0, 0.0)
        cursor_e = (0.35, 0.0, 0.0)
        edge_ij = [(4, 1), (4, 2)]
        path3_ij = [(4, 1), (4, 2), (4, 3)]
        mixed_path_ij = [(4, 2), (5, 2)]
        mixed_region_ij = [(4, 2), (5, 2), (5, 3), (6, 2), (6, 3)]
        drag_w = (-50, 0)
        drag_e = (50, 0)
        drag_s = (0, -50)

        def mutate_edge(bm):
            stamp_identity_layers(bm)
            STATE["expected_dv_de"] = (2, 3)
            STATE["expect_pinned"] = (coordinate_key((0.0, -2.0, 0.0)),)

        def mutate_path3(bm):
            stamp_identity_layers(bm)
            STATE["expected_dv_de"] = (3, 4)
            STATE["expect_pinned"] = (
                coordinate_key((0.0, -2.0, 0.0)),
                coordinate_key((0.0, 2.0, 0.0)),
            )

        def mutate_fill_edge(bm):
            STATE["expected_dv_de"] = (2, 5)
            STATE["expected_df"] = 3

        def mutate_fill_vert(bm):
            STATE["expected_dv_de"] = (1, 3)
            STATE["expected_df"] = 2

        def named(case, name):
            case.yse_name = name
            return case

        cases = [
            named(
                run_case(
                    "vert_pick_west_drag_west",
                    [seam],
                    cursor_w,
                    verify_vert_west_west,
                    drag=drag_w,
                    mutate=stamp_only,
                ),
                "vert_pick_west_drag_west",
            ),
            named(
                run_case(
                    "vert_pick_east_drag_east",
                    [seam],
                    cursor_e,
                    verify_vert_east_east,
                    drag=drag_e,
                    mutate=stamp_only,
                ),
                "vert_pick_east_drag_east",
            ),
            named(
                run_case(
                    "vert_pick_west_drag_east",
                    [seam],
                    cursor_w,
                    verify_vert_cross,
                    drag=drag_e,
                    mutate=stamp_only,
                ),
                "vert_pick_west_drag_east",
            ),
            named(
                run_case(
                    "vert_pick_east_drag_west",
                    [seam],
                    cursor_e,
                    verify_vert_cross,
                    drag=drag_w,
                    mutate=stamp_only,
                ),
                "vert_pick_east_drag_west",
            ),
            named(
                run_case(
                    "edge_pick_west",
                    edge_ij,
                    (-0.35, -0.5, 0.0),
                    verify_edge_or_path,
                    drag=drag_w,
                    mutate=mutate_edge,
                ),
                "edge_pick_west",
            ),
            named(
                run_case(
                    "path3_pick_west",
                    path3_ij,
                    cursor_w,
                    verify_edge_or_path,
                    drag=drag_w,
                    mutate=mutate_path3,
                ),
                "path3_pick_west",
            ),
            named(run_case("zero_move", [seam], cursor_w, verify_zero_or_esc, drag=(0, 0)), "zero_move"),
            named(run_case("esc", [seam], cursor_w, verify_zero_or_esc, drag=(0, 0), confirm="ESC"), "esc"),
            named(
                run_case("tol_band", [seam], cursor_w, verify_tol_band, drag=(0, 0), mutate=mutate_band),
                "tol_band",
            ),
            named(
                run_case(
                    "tol_band_path3",
                    path3_ij,
                    cursor_w,
                    verify_tol_band,
                    drag=(0, 0),
                    mutate=mutate_band_path3,
                ),
                "tol_band_path3",
            ),
            named(
                run_case(
                    "aa_prime_inband",
                    [(3, 2), (4, 2)],
                    (0.0, -0.35, 0.0),
                    verify_aa_prime_inband,
                    drag=(50, 0),
                    mutate=mutate_inband,
                    nx=7,
                    ny=4,
                ),
                "aa_prime_inband",
            ),
            named(
                run_case(
                    "b_form",
                    [(3, 2), (4, 2), (5, 2)],
                    (0.0, -0.35, 0.0),
                    verify_b_form,
                    drag=drag_s,
                    mutate=mutate_b_form,
                ),
                "b_form",
            ),
            named(
                run_case(
                    "fill_edge",
                    edge_ij,
                    (-0.35, -0.5, 0.0),
                    verify_fill,
                    drag=drag_w,
                    alt=True,
                    with_uv=True,
                    mutate=mutate_fill_edge,
                ),
                "fill_edge",
            ),
            named(
                run_case(
                    "fill_edge_zero",
                    edge_ij,
                    (-0.35, -0.5, 0.0),
                    verify_fill,
                    drag=(0, 0),
                    alt=True,
                    with_uv=True,
                    mutate=mutate_fill_edge,
                ),
                "fill_edge_zero",
            ),
            named(
                run_case(
                    "fill_vert",
                    [seam],
                    cursor_w,
                    verify_fill,
                    drag=drag_w,
                    alt=True,
                    with_uv=True,
                    mutate=mutate_fill_vert,
                ),
                "fill_vert",
            ),
            named(
                run_case(
                    "fill_cycle_decline",
                    edge_ij,
                    (-0.35, -0.5, 0.0),
                    verify_fill_cycle_decline,
                    drag=drag_w,
                    alt=True,
                    with_uv=True,
                    mutate=mutate_fill_cycle,
                ),
                "fill_cycle_decline",
            ),
            named(run_case("boundary", [(4, 0)], (-0.35, -2.0, 0.0), verify_boundary, drag=drag_w), "boundary"),
            named(
                run_case(
                    "mixed_path",
                    mixed_path_ij,
                    (0.5, -0.35, 0.0),
                    verify_mixed_path,
                    drag=drag_s,
                ),
                "mixed_path",
            ),
            named(
                run_case(
                    "mixed_disjoint_l",
                    [],
                    (-0.35, 0.0, 0.0),
                    verify_mixed_disjoint,
                    drag=drag_w,
                    builder=build_mixed_l,
                    select_xyz=[(0.0, 0.0, 0.0)],
                    mutate=_mutate_mixed(
                        {
                            "dv_de": (1, 0),
                            "dup_vids": "origin",
                            "reason": "the native Rip duplicated vertices but no seam edge was found",
                        }
                    ),
                ),
                "mixed_disjoint_l",
            ),
            named(
                run_case(
                    "mixed_disjoint_t",
                    [],
                    (-0.35, 0.0, 0.0),
                    verify_mixed_disjoint,
                    drag=drag_w,
                    builder=build_mixed_t,
                    select_xyz=[(0.0, 0.0, 0.0)],
                    mutate=_mutate_mixed(
                        {
                            "dv_de": (1, 0),
                            "dup_vids": "origin",
                            "reason": "the native Rip duplicated vertices but no seam edge was found",
                        }
                    ),
                ),
                "mixed_disjoint_t",
            ),
            named(
                run_case(
                    "mixed_disjoint_boundary",
                    [],
                    (-0.35, 0.0, 0.0),
                    verify_mixed_disjoint,
                    drag=drag_w,
                    builder=build_mixed_boundary,
                    select_xyz=[(0.0, 0.0, 0.0)],
                    mutate=_mutate_mixed(
                        {
                            "dv_de": (1, 0),
                            "dup_vids": "origin",
                            "reason": "the native Rip duplicated vertices but no seam edge was found",
                        }
                    ),
                ),
                "mixed_disjoint_boundary",
            ),
            named(
                run_case(
                    "self_face",
                    [],
                    (0.0, -0.35, 1.0),
                    verify_self_face,
                    drag=drag_w,
                    builder=build_grid_with_fin,
                    mutate=add_midplane_fin,
                    select_xyz=[(0.0, 0.5, 1.0), (0.0, -0.5, 1.0)],
                ),
                "self_face",
            ),
            named(
                run_case(
                    "self_face_straddle",
                    [seam],
                    cursor_w,
                    verify_self_face_straddle,
                    drag=drag_w,
                    builder=build_grid_with_straddle_self_face,
                    mutate=mutate_straddle,
                ),
                "self_face_straddle",
            ),
            named(run_case("undo_redo", [seam], cursor_w, verify_undo_redo, drag=drag_w), "undo_redo"),
            named(
                run_case("raw_repair", [seam], cursor_w, verify_raw_repair, drag=drag_w, mutate=install_native_clone),
                "raw_repair",
            ),
            named(
                run_case(
                    "mixed_region",
                    mixed_region_ij,
                    (1.5, -0.35, 0.0),
                    verify_mixed_region,
                    drag=drag_e,
                ),
                "mixed_region",
            ),
        ]
        run_all(_filter_cases(cases))
    except BaseException:
        fail()
    return None


bpy.app.timers.register(start_test, first_interval=0.3)
