# SPDX-License-Identifier: GPL-3.0-or-later

"""GUI regression for Phase 4: axis-crossing self-mirrored Rip seams.

Run with Blender's real window/event loop::

    blender --factory-startup --enable-event-simulate --no-window-focus \
        -p 40 40 960 600 --python test_rip_self_mirrored.py

Fixture layout matches spike_s3_gui (odd NX=7 → columns at
x = …, −1.5, −0.5, 0.5, 1.5, … with no on-plane vertex column).  The
bridging edge at (i=3, i=4) is an A–A′ self-mirror seam.

Cases (contract §4.3 / S3):

(a) full_plus     complete self-mirror +X drag → V-open, X-symmetric
(b) full_minus    −X drag → equivariant pair of (a) (coords + incidence + selection)
(c) full_zero     zero move → selected copy exactly one per side
(c2) full_esc     Esc confirm → same as zero move
(d) full_converge drag that lands a bank vertex on the plane → 2 BMVerts
                  share the quantized on-plane coordinate (no weld; ±ε fails)
(e) partial       B–A–A′ partial self-overlap → decline + native kept
(f) straddle      +X one-sided seam + unrelated −X selection → external
                  mirror still works (overlap-passthrough removal regression)
(g) fill          Alt+V on complete self-mirror seam → fill face incidence
                  uses correct bank corners
(h) boundary      self-mirror seam on mesh boundary → 3-component bank split
(i) long_seam     3+ vert self-mirror path {B–A, A–A′, A′–B′} bank consistency
(j) backup_fail   backup create failure → ERROR + FINISHED + native + cleanup
(k) rollback_fail rollback itself fails → ERROR + FINISHED
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from collections import Counter, defaultdict
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
from ydd_symmetric_edit import backup, layer_names, operators, rip  # noqa: E402

MARKER_OK = "YSE_RIP_SELF_MIRRORED_OK"
MARKER_FAILED = "YSE_RIP_SELF_MIRRORED_TEST_FAILED"
NX, NY = 7, 4  # odd NX → no on-plane column; bridge at i=3/4 (x=∓0.5)
PRECISION = 5
TEST_ROLE_LAYER = "yse_test_mirror_role"
STATE = {}


def fail(message=""):
    if message:
        print(f"YSE_RIP_SELF_MIRRORED_ERROR={message}", flush=True)
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


def build_mesh(name):
    for old in tuple(bpy.data.objects):
        bpy.data.objects.remove(old, do_unlink=True)
    for old_mesh in tuple(bpy.data.meshes):
        if old_mesh.users == 0:
            bpy.data.meshes.remove(old_mesh)
    mesh = bpy.data.meshes.new(f"YSE_RipSelf_{name}")
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
    obj = bpy.data.objects.new(f"YSE_RipSelfObj_{name}", mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.use_mesh_mirror_x = True
    obj.use_mesh_mirror_y = False
    obj.use_mesh_mirror_z = False
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    with override():
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.context.tool_settings.mesh_select_mode = (True, False, False)
        bpy.ops.ed.undo_push(message=f"YSE rip self-mirror baseline {name}")
    STATE["object"] = obj
    return obj


def grid_xy(i, j):
    return (i - NX / 2, j - NY / 2, 0.0)


def grid_vert(bm, i, j):
    target = Vector(grid_xy(i, j))
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
        endpoints = tuple(sorted(coordinate_key(vertex.co) for vertex in edge.verts))
        keys.append(endpoints)
    return Counter(keys)


def mirrored_edge_incidence_multiset(bm):
    keys = []
    for edge in bm.edges:
        endpoints = tuple(sorted(mirror_key(vertex.co) for vertex in edge.verts))
        keys.append(endpoints)
    return Counter(keys)


def assert_x_symmetric_coords(bm, label=""):
    """Vertex multiset is X-symmetric (S3-verified property of the V-open rule).

    Face incidence is intentionally not required for self-mirrored horizontal
    seams: the two banks sit on opposite face-sides of the seam (different Y
    for an X-parallel bridge), so X-mirror maps a south face to a location
    where only a north-bank face exists.  Contract §1 equivariance for this
    class is checked via the +X/−X result pair, not self-symmetry of faces.
    """

    prefix = f"{label}: " if label else ""
    verts = vertex_multiset(bm)
    assert verts == mirrored_vertex_multiset(bm), f"{prefix}vertex coords not X-symmetric: {verts}"


def assert_x_symmetric_full(bm, label=""):
    """Coords + face incidence (for external-mirror seams that are self-symmetric)."""

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


def selected_coords(bm):
    return sorted(coordinate_key(vertex.co) for vertex in bm.verts if vertex.select)


def selected_multiset(bm):
    return Counter(coordinate_key(vertex.co) for vertex in bm.verts if vertex.select)


def tag_mirror_roles(bm):
    """Stamp a test-only int layer so post-rip copies keep pre-rip identity.

    Native Rip copies integer CustomData onto duplicates, so both bank copies
    of V share the same role id.  Mirror partners get distinct ids with a
    lookup in STATE['role_mirror_map'].
    """

    existing = bm.verts.layers.int.get(TEST_ROLE_LAYER)
    if existing is not None:
        bm.verts.layers.int.remove(existing)
    layer = bm.verts.layers.int.new(TEST_ROLE_LAYER)
    coord_to_role: dict[tuple[float, float, float], int] = {}
    next_id = 1
    for vertex in bm.verts:
        vertex[layer] = next_id
        coord_to_role[coordinate_key(vertex.co)] = next_id
        next_id += 1
    mirror_map: dict[int, int] = {}
    for vertex in bm.verts:
        partner = coord_to_role.get(mirror_key(vertex.co))
        if partner is not None:
            mirror_map[int(vertex[layer])] = partner
    STATE["role_mirror_map"] = mirror_map


def assert_role_bank_correspondence(bm, label=""):
    """nonsource(V) must equal ρ(source(mirror(V))) via the test role layer."""

    prefix = f"{label}: " if label else ""
    layer = bm.verts.layers.int.get(TEST_ROLE_LAYER)
    assert layer is not None, f"{prefix}test role layer missing"
    mirror_map = STATE.get("role_mirror_map") or {}
    by_role: dict[int, list] = defaultdict(list)
    for vertex in bm.verts:
        role_id = int(vertex[layer])
        if role_id > 0:
            by_role[role_id].append(vertex)

    source_by_role: dict[int, object] = {}
    nonsource_by_role: dict[int, object] = {}
    for role_id, copies in by_role.items():
        if len(copies) != 2:
            continue
        selected = [copy for copy in copies if copy.select]
        assert len(selected) == 1, f"{prefix}role {role_id} must have exactly one selected copy, got {len(selected)}"
        source = selected[0]
        nonsource = copies[1] if copies[0] is source else copies[0]
        source_by_role[role_id] = source
        nonsource_by_role[role_id] = nonsource

    assert source_by_role, f"{prefix}no duplicated role ids found for bank check"
    for role_id, nonsource in nonsource_by_role.items():
        partner_id = mirror_map.get(role_id)
        assert partner_id is not None, f"{prefix}role {role_id} has no mirror partner tag"
        source_mirror = source_by_role.get(partner_id)
        assert source_mirror is not None, f"{prefix}mirror role {partner_id} has no source bank"
        expected = coordinate_key((-source_mirror.co.x, source_mirror.co.y, source_mirror.co.z))
        actual = coordinate_key(nonsource.co)
        assert actual == expected, (
            f"{prefix}role bank mismatch for role {role_id}: nonsource={actual} expected ρ(source(mirror))={expected}"
        )


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


def rip_events(cursor_xyz, drag=(40, 0), *, alt=False, confirm="LMB"):
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


def warning_messages():
    return [message for kind, message in operators._FINISH_REPORTS if kind == "WARNING"]


def info_messages():
    return [message for kind, message in operators._FINISH_REPORTS if kind == "INFO"]


def error_messages():
    return [message for kind, message in operators._FINISH_REPORTS if kind == "ERROR"]


# ---------------------------------------------------------------------------
# case runners / verifiers
# ---------------------------------------------------------------------------

# Interior row for the A–A′ bridge; cursor south of the row selects the
# south-face bank (spike_s3_gui convention).
SEAM_ROW = 2
SEAM_IJ = [(3, SEAM_ROW), (4, SEAM_ROW)]  # A(-0.5), A'(+0.5)
CURSOR_SOUTH = (0.0, SEAM_ROW - NY / 2 - 0.35, 0.0)
# 3+ vert self-mirror path: B(-1.5)–A(-0.5)–A′(0.5)–B′(1.5)
LONG_SEAM_IJ = [(2, SEAM_ROW), (3, SEAM_ROW), (4, SEAM_ROW), (5, SEAM_ROW)]


def run_case(name, select_ij, cursor_xyz, verify, *, drag=(40, 0), alt=False, confirm="LMB", mutate=None):
    def start(next_case):
        try:
            print(f"YSE_RIP_SELF_CASE={name}", flush=True)
            obj = build_mesh(name)
            bm = bmesh.from_edit_mesh(obj.data)
            if mutate is not None:
                mutate(bm)
            STATE["baseline"] = topology_counts(bm)
            STATE["case_name"] = name
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

            send_events(
                rip_events(cursor_xyz, drag=drag, alt=alt, confirm=confirm),
                lambda: wait_settled(settled),
            )
        except BaseException:
            fail()

    return start


def _assert_source_bank_selected(bm):
    """Source bank is selected, non-source is not (native selection rule)."""

    selected = [vertex for vertex in bm.verts if vertex.select]
    assert selected, "expected the source bank to remain selected"
    # Selected vertices must not form a self-mirrored set at the same
    # multiset as the full mesh — only one bank is selected.
    selected_keys = Counter(coordinate_key(vertex.co) for vertex in selected)
    mirrored_selected = Counter(mirror_key(vertex.co) for vertex in selected)
    # After V-opening the full mesh is symmetric, so selecting only the
    # source bank means selected_keys ≠ mirrored_selected (unless zero
    # move places both banks at the pre-state coordinates — still one bank
    # selected, so the selected multiset is half of the duplicated pairs).
    assert selected_keys != mirrored_selected or all(
        abs(vertex.co.x) < 1e-4 or vertex_multiset(bm)[coordinate_key(vertex.co)] >= 2 for vertex in selected
    ), "selection should cover only the source bank"


def _assert_zero_move_selection(bm, seam_keys):
    """Zero-move / Esc: exactly one selected copy per seam endpoint side."""

    selected = [vertex for vertex in bm.verts if vertex.select]
    selected_keys = Counter(coordinate_key(vertex.co) for vertex in selected)
    for key in seam_keys:
        # Two BMVerts share the pre-state coordinate; exactly one is selected.
        assert vertex_multiset(bm)[key] == 2, f"expected 2 verts at {key}, got {vertex_multiset(bm)[key]}"
        assert selected_keys[key] == 1, f"expected exactly 1 selected copy at {key}, got {selected_keys[key]}"
    # No selected verts outside the seam pair coordinates for the simple seam.
    assert sum(selected_keys[key] for key in seam_keys) == len(selected), (
        f"selected verts outside seam coords: {dict(selected_keys)}"
    )


def verify_full_plus(bm):
    dv = len(bm.verts) - STATE["baseline"][0]
    # Native A–A′ rip: 2 duplicated verts (no external mirror split — V-open
    # is coordinate-only).  dv stays 2.
    assert dv == 2, f"full_plus: expected dv=2 (self-mirror V-open), got {dv}"
    assert_x_symmetric_coords(bm, "full_plus")
    assert_layers_removed(bm)
    _assert_source_bank_selected(bm)
    assert_role_bank_correspondence(bm, "full_plus")
    assert any("Mirrored Rip" in message for message in info_messages()), info_messages()
    STATE["plus_coords"] = vertex_multiset(bm)
    STATE["plus_faces"] = face_incidence_multiset(bm)
    STATE["plus_edges"] = edge_incidence_multiset(bm)
    STATE["plus_selected"] = selected_multiset(bm)


def verify_full_minus(bm):
    dv = len(bm.verts) - STATE["baseline"][0]
    assert dv == 2, f"full_minus: expected dv=2, got {dv}"
    assert_x_symmetric_coords(bm, "full_minus")
    assert_layers_removed(bm)
    _assert_source_bank_selected(bm)
    assert_role_bank_correspondence(bm, "full_minus")
    # Equivariance (§1-1 / §4.3-3 / §7-8): result of −X drag is the X-mirror
    # of +X drag on coords, face incidence, edge incidence, and selection.
    plus = STATE.get("plus_coords")
    assert plus is not None, "full_plus must run before full_minus"
    minus = vertex_multiset(bm)
    mirrored_minus = Counter()
    for key, count in minus.items():
        mirrored_minus[(-key[0], key[1], key[2])] += count
    assert mirrored_minus == plus, f"equivariance coords failed: ρ(minus)={mirrored_minus} vs plus={plus}"

    plus_faces = STATE.get("plus_faces")
    minus_faces_mirrored = mirrored_face_incidence_multiset(bm)
    assert minus_faces_mirrored == plus_faces, "equivariance face incidence failed: ρ(minus faces) vs plus faces differ"

    plus_edges = STATE.get("plus_edges")
    minus_edges_mirrored = mirrored_edge_incidence_multiset(bm)
    assert minus_edges_mirrored == plus_edges, "equivariance edge incidence failed: ρ(minus edges) vs plus edges differ"

    plus_selected = STATE.get("plus_selected")
    minus_selected = selected_multiset(bm)
    mirrored_minus_selected = Counter()
    for key, count in minus_selected.items():
        mirrored_minus_selected[(-key[0], key[1], key[2])] += count
    assert mirrored_minus_selected == plus_selected, (
        f"equivariance selection failed: ρ(minus sel)={mirrored_minus_selected} vs plus sel={plus_selected}"
    )


def verify_full_zero(bm):
    dv = len(bm.verts) - STATE["baseline"][0]
    assert dv == 2, f"full_zero: expected dv=2, got {dv}"
    assert_x_symmetric_coords(bm, "full_zero")
    assert_layers_removed(bm)
    a_key = coordinate_key((-0.5, SEAM_ROW - NY / 2, 0.0))
    a2_key = coordinate_key((0.5, SEAM_ROW - NY / 2, 0.0))
    _assert_zero_move_selection(bm, (a_key, a2_key))
    _assert_source_bank_selected(bm)


def verify_full_esc(bm):
    """Esc confirm equals zero-move: banks unmoved, one selected copy per side."""

    dv = len(bm.verts) - STATE["baseline"][0]
    assert dv == 2, f"full_esc: expected dv=2, got {dv}"
    assert_x_symmetric_coords(bm, "full_esc")
    assert_layers_removed(bm)
    a_key = coordinate_key((-0.5, SEAM_ROW - NY / 2, 0.0))
    a2_key = coordinate_key((0.5, SEAM_ROW - NY / 2, 0.0))
    _assert_zero_move_selection(bm, (a_key, a2_key))
    _assert_source_bank_selected(bm)
    assert any("Mirrored Rip" in message for message in info_messages()), info_messages()


def install_converge_plane_snap(bm):
    """Snap the selected (source) bank onto the plane just before V-open apply.

    Integer-pixel GUI drags cannot land exactly on x=0 (measured residual
    ~±0.01).  The contract property under test is "source on plane ⇒
    non-source shares the same coordinate, no weld", so we force the source
    bank onto the plane after native settles and before our apply runs.
    """

    del bm
    original = rip.apply_mirrored_rip

    def snapped(edit_bm, snapshot, mirror_face_ids):
        for vertex in edit_bm.verts:
            if vertex.select:
                vertex.co.x = 0.0
        return original(edit_bm, snapshot, mirror_face_ids)

    STATE["original_apply_converge"] = original
    rip.apply_mirrored_rip = snapped


def verify_full_converge(bm):
    """Source bank on the plane: two distinct BMVerts share one quantized key."""

    try:
        dv = len(bm.verts) - STATE["baseline"][0]
        assert dv == 2, f"full_converge: expected dv=2 (no weld), got {dv}"
        assert_x_symmetric_coords(bm, "full_converge")
        assert_layers_removed(bm)
        # Require a quantized co-located pair (count >= 2 at one key).  A ±ε
        # pair of distinct keys is not sufficient (contract §4.3-3 plane
        # convergence: 2 BMVerts coexist at the same coordinate, no weld).
        on_plane = [vertex for vertex in bm.verts if abs(vertex.co.x) <= 1.0e-5]
        assert len(on_plane) >= 2, (
            f"full_converge: expected ≥2 verts on the plane, got "
            f"{[(round(v.co.x, 5), round(v.co.y, 5)) for v in bm.verts if abs(v.co.x) < 0.5]}"
        )
        on_plane_keys = Counter(coordinate_key(vertex.co) for vertex in on_plane)
        has_double = any(count >= 2 for count in on_plane_keys.values())
        assert has_double, (
            f"full_converge: expected 2 distinct BMVerts at one quantized on-plane key, got {dict(on_plane_keys)}"
        )
        # Explicitly reject a residual ±ε pair being counted as success.
        near_keys = Counter(coordinate_key(vertex.co) for vertex in bm.verts if abs(vertex.co.x) <= 0.08)
        has_pm_only = (not has_double) and any(
            abs(key[0]) > 1.0e-5 and near_keys.get((-key[0], key[1], key[2]), 0) >= 1 for key in near_keys
        )
        assert not has_pm_only, f"full_converge: ±ε pair without co-location is insufficient: {dict(near_keys)}"
        _assert_source_bank_selected(bm)
    finally:
        if "original_apply_converge" in STATE:
            rip.apply_mirrored_rip = STATE["original_apply_converge"]


def verify_partial_declined(bm):
    dv = len(bm.verts) - STATE["baseline"][0]
    # Native rips the B–A–A′ path (3 verts duplicated) without mirror.
    assert dv >= 2, f"partial: expected a native rip (dv>=2), got {dv}"
    assert vertex_multiset(bm) != mirrored_vertex_multiset(bm), "partial self-overlap must not be mirrored"
    assert_layers_removed(bm)
    warnings = warning_messages()
    assert any("partial self-overlap" in message for message in warnings), warnings


def verify_straddle_mirrored(bm):
    """+X one-sided seam + unrelated −X selection: external mirror still works."""

    dv = len(bm.verts) - STATE["baseline"][0]
    # One-sided 2-vert path → native +2, mirror +2 → dv=4.
    assert dv == 4, f"straddle: expected dv=4 (source+mirror), got {dv}"
    assert_x_symmetric_full(bm, "straddle")
    assert_layers_removed(bm)
    # Far −X selected vertex must stay unripped (single copy at original).
    far_key = coordinate_key(grid_xy(1, 0))
    assert vertex_multiset(bm)[far_key] == 1, "unrelated −X vertex must stay unripped"
    assert any("Mirrored Rip" in message for message in info_messages()), info_messages()


def verify_fill_self(bm):
    dv = len(bm.verts) - STATE["baseline"][0]
    df = len(bm.faces) - STATE["baseline"][2]
    assert dv == 2, f"fill: expected dv=2, got {dv}"
    # Native fill on a 1-edge seam produces fill faces between banks; no
    # external fill mirror is created (self-mirror path is coord-only).
    assert df >= 1, f"fill: expected fill faces, got df={df}"
    assert_x_symmetric_coords(bm, "fill")
    assert_layers_removed(bm)
    _assert_source_bank_selected(bm)

    # Fill faces bridge banks: each must reference both a selected (source)
    # corner and a non-selected (non-source) corner, and every corner must be
    # one of the four bank verts of the A–A′ pair (no foreign corners).
    selected = {vertex for vertex in bm.verts if vertex.select}
    selected_keys = {coordinate_key(vertex.co) for vertex in selected}
    nonsource_keys = {mirror_key(key) for key in selected_keys}
    bank_keys = selected_keys | nonsource_keys
    fill_faces = []
    for face in bm.faces:
        corner_keys = [coordinate_key(vertex.co) for vertex in face.verts]
        if not all(key in bank_keys for key in corner_keys):
            continue
        n_selected = sum(1 for vertex in face.verts if vertex in selected)
        n_nonsource = sum(1 for vertex in face.verts if coordinate_key(vertex.co) in nonsource_keys)
        if n_selected >= 1 and n_nonsource >= 1:
            fill_faces.append(face)
    assert fill_faces, "fill: no face bridges source and non-source banks"
    for face in fill_faces:
        for vertex in face.verts:
            key = coordinate_key(vertex.co)
            assert key in bank_keys, f"fill corner {key} is not a bank vertex"
            if vertex in selected:
                assert key in selected_keys, "fill selected corner not on source bank"
            else:
                assert key in nonsource_keys, "fill non-selected corner not on non-source bank"


def verify_boundary_self(bm):
    dv = len(bm.verts) - STATE["baseline"][0]
    assert dv == 2, f"boundary: expected dv=2, got {dv}"
    assert_x_symmetric_coords(bm, "boundary")
    assert_layers_removed(bm)
    _assert_source_bank_selected(bm)

    # 3-component bank split (S3): moved side connected + stationary 2 isolates.
    selected = [vertex for vertex in bm.verts if vertex.select]
    assert len(selected) == 2, f"boundary: expected 2 selected source verts, got {len(selected)}"
    nonsource_keys = {mirror_key(vertex.co) for vertex in selected}
    nonsource = [vertex for vertex in bm.verts if (not vertex.select) and coordinate_key(vertex.co) in nonsource_keys]
    assert len(nonsource) == 2, f"boundary: expected 2 nonsource bank verts, got {len(nonsource)}"
    bank = set(selected) | set(nonsource)
    adjacency: dict[object, set] = {vertex: set() for vertex in bank}
    for edge in bm.edges:
        a, b = edge.verts
        if a in bank and b in bank:
            adjacency[a].add(b)
            adjacency[b].add(a)

    remaining = set(bank)
    components: list[set] = []
    while remaining:
        start = next(iter(remaining))
        stack = [start]
        component: set = set()
        while stack:
            node = stack.pop()
            if node in component:
                continue
            component.add(node)
            stack.extend(adjacency[node] - component)
        components.append(component)
        remaining -= component

    sizes = sorted(len(component) for component in components)
    assert sizes == [1, 1, 2], f"boundary: expected component sizes [1,1,2], got {sizes}"
    moved = next(component for component in components if len(component) == 2)
    assert set(selected) == moved, "boundary: moved/connected component must be the source bank"

    # Face-side uniqueness: every source vert attaches to faces on the same
    # side of the seam (shared sign of face-center offset from the boundary).
    face_side_signs = set()
    for vertex in selected:
        for face in vertex.link_faces:
            center_y = sum(corner.co.y for corner in face.verts) / len(face.verts)
            # Boundary seam at y = -NY/2; interior faces have center_y > boundary.
            face_side_signs.add(1 if center_y > (-NY / 2 + 0.1) else -1)
    assert len(face_side_signs) == 1, f"boundary: source bank spans multiple face-sides: {face_side_signs}"


def verify_long_seam(bm):
    """3+ vert self-mirror path: V-open + cross-vertex bank consistency."""

    dv = len(bm.verts) - STATE["baseline"][0]
    assert dv == 4, f"long_seam: expected dv=4 (4 duplicated verts), got {dv}"
    assert_x_symmetric_coords(bm, "long_seam")
    assert_layers_removed(bm)
    _assert_source_bank_selected(bm)
    assert_role_bank_correspondence(bm, "long_seam")
    selected = [vertex for vertex in bm.verts if vertex.select]
    assert len(selected) == 4, f"long_seam: expected 4 selected source verts, got {len(selected)}"
    # Source bank must itself be X-pair-complete under role map (same face-side).
    selected_keys = selected_multiset(bm)
    mirrored_selected = Counter(mirror_key(vertex.co) for vertex in selected)
    assert selected_keys != mirrored_selected, "long_seam: source bank must be one side only"
    assert any("Mirrored Rip" in message for message in info_messages()), info_messages()


def install_backup_create_failure(bm):
    del bm

    def boom(_edit_bm):
        raise RuntimeError("injected backup create failure")

    STATE["original_create_backup"] = backup.create_topology_backup
    backup.create_topology_backup = boom


def verify_backup_create_failure(bm):
    try:
        dv = len(bm.verts) - STATE["baseline"][0]
        assert dv == 2, f"backup_fail: expected native rip kept (dv=2), got {dv}"
        # Native kept, not mirrored (asymmetric after self-mirror seam without V-open
        # replacement — actually zero post-process leaves native both-banks-same-T,
        # which is NOT X-symmetric).
        assert vertex_multiset(bm) != mirrored_vertex_multiset(bm), "backup_fail: mirror must not have run"
        assert_layers_removed(bm)
        errors = error_messages()
        assert any("backup" in message.lower() for message in errors), errors
        assert not operators._SESSIONS, "backup_fail: session must be cleaned up"
    finally:
        backup.create_topology_backup = STATE["original_create_backup"]


def install_rollback_failure(bm):
    del bm
    original_apply = rip.apply_mirrored_rip
    original_restore = backup.restore_topology_backup

    def broken_apply(edit_bm, snapshot, mirror_face_ids):
        # Mutate via the real apply, then force a failure so finish rolls back.
        count, reason = original_apply(edit_bm, snapshot, mirror_face_ids)
        del count, reason
        return 0, "injected apply failure for rollback test"

    def broken_restore(mesh, topology_backup):
        del mesh, topology_backup
        raise RuntimeError("injected rollback failure")

    STATE["original_apply"] = original_apply
    STATE["original_restore"] = original_restore
    rip.apply_mirrored_rip = broken_apply
    backup.restore_topology_backup = broken_restore


def verify_rollback_failure(bm):
    try:
        assert_layers_removed(bm)
        errors = error_messages()
        assert any("rollback" in message.lower() for message in errors), errors
        assert not operators._SESSIONS, "rollback_fail: session must be cleaned up"
    finally:
        rip.apply_mirrored_rip = STATE["original_apply"]
        backup.restore_topology_backup = STATE["original_restore"]


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

        # Pixel drags: calibrated for view_distance=10 / 960×600 window.
        # +X / −X use a moderate drag; converge aims for ~0.5 mesh units so
        # A(-0.5) lands near the plane (and V-open places the partner on top).
        drag_plus = (50, 0)
        drag_minus = (-50, 0)
        # ~0.5 mesh units in X under this view calibration.
        # Measured on 4.2: drag=28 overshoots to ±0.00845; drag=27 lands on 0.
        drag_converge = (27, 0)

        # One-sided +X path (i=5,6 at x=1.5,2.5) plus far −X vertex (i=1,j=0).
        straddle_sel = [(5, SEAM_ROW), (6, SEAM_ROW), (1, 0)]
        straddle_cursor = (2.0, SEAM_ROW - NY / 2 - 0.35, 0.0)

        # B–A–A′ partial: B(1.5), A′(0.5), A(-0.5) on the seam row.
        partial_sel = [(5, SEAM_ROW), (4, SEAM_ROW), (3, SEAM_ROW)]

        boundary_sel = [(3, 0), (4, 0)]
        boundary_cursor = (0.0, 0 - NY / 2 - 0.35, 0.0)

        cases = [
            run_case(
                "full_plus",
                SEAM_IJ,
                CURSOR_SOUTH,
                verify_full_plus,
                drag=drag_plus,
                mutate=tag_mirror_roles,
            ),
            run_case(
                "full_minus",
                SEAM_IJ,
                CURSOR_SOUTH,
                verify_full_minus,
                drag=drag_minus,
                mutate=tag_mirror_roles,
            ),
            run_case("full_zero", SEAM_IJ, CURSOR_SOUTH, verify_full_zero, drag=(0, 0)),
            run_case("full_esc", SEAM_IJ, CURSOR_SOUTH, verify_full_esc, drag=(0, 0), confirm="ESC"),
            run_case(
                "full_converge",
                SEAM_IJ,
                CURSOR_SOUTH,
                verify_full_converge,
                drag=drag_converge,
                mutate=install_converge_plane_snap,
            ),
            run_case("partial", partial_sel, CURSOR_SOUTH, verify_partial_declined, drag=drag_plus),
            run_case("straddle", straddle_sel, straddle_cursor, verify_straddle_mirrored, drag=drag_plus),
            run_case("fill", SEAM_IJ, CURSOR_SOUTH, verify_fill_self, drag=drag_plus, alt=True),
            run_case("boundary", boundary_sel, boundary_cursor, verify_boundary_self, drag=drag_plus),
            run_case(
                "long_seam",
                LONG_SEAM_IJ,
                CURSOR_SOUTH,
                verify_long_seam,
                drag=drag_plus,
                mutate=tag_mirror_roles,
            ),
            run_case(
                "backup_fail",
                SEAM_IJ,
                CURSOR_SOUTH,
                verify_backup_create_failure,
                drag=drag_plus,
                mutate=install_backup_create_failure,
            ),
            run_case(
                "rollback_fail",
                SEAM_IJ,
                CURSOR_SOUTH,
                verify_rollback_failure,
                drag=drag_plus,
                mutate=install_rollback_failure,
            ),
        ]
        run_all(cases)
    except BaseException:
        fail()
    return None


bpy.app.timers.register(start_test, first_interval=0.3)
