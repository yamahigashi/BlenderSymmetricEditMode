# SPDX-License-Identifier: GPL-3.0-or-later

"""GUI acceptance: F9 discriminator state matrix (§C-1) and multi-token repair (§C-2).

Run::

    cmd.exe /c run_gui_test.bat 52 test_f9_discriminator_matrix.py
    cmd.exe /c run_gui_test.bat 42 test_f9_discriminator_matrix.py

Contract: .agents/doc/f9_discriminator_fix_plan_2026-08-09.md §A / §C-1 / §C-2.
Pattern: paint mesh state → call operators._prepare_adjust_last_operation_repeat()
(or _repair_history_state) → assert return / sessions / repair queue.
"""

from __future__ import annotations

import copy
import os
import sys
import time
import traceback
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
from ydd_symmetric_edit import core, keymaps, operators  # noqa: E402
from ydd_symmetric_edit._types import HistoryRecord  # noqa: E402

OBJECT_NAME = "YSE_F9MatrixObject"
MESH_NAME = "YSE_F9MatrixMesh"
FOREIGN_TOKEN = 999999
FINISHED_TOPOLOGY = (16, 20, 6)

STATE: dict = {
    "addon_registered": False,
    "events": [],
    "deadline": 0.0,
    "phase": "startup",
    "case_results": {},
}

MARK_OK = "YSE_F9_MATRIX_OK"
MARK_FAIL = "YSE_F9_MATRIX_FAILED"


def fail(message=""):
    if message:
        print(f"YSE_F9_MATRIX_ERROR={message}", flush=True)
    traceback.print_exc()
    print(f"YSE_F9_MATRIX_PHASE={STATE.get('phase')}", flush=True)
    for name, status in STATE.get("case_results", {}).items():
        print(f"YSE_F9_MATRIX_RESULT={name}:{status}", flush=True)
    print(MARK_FAIL, flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(1)


def record_case(name: str, passed: bool) -> None:
    status = "PASS" if passed else "FAIL"
    STATE["case_results"][name] = status
    print(f"YSE_F9_MATRIX_RESULT={name}:{status}", flush=True)


def current_object():
    return bpy.data.objects.get(OBJECT_NAME)


def current_bmesh():
    obj = current_object()
    if obj is None or obj.mode != "EDIT":
        return None
    return bmesh.from_edit_mesh(obj.data)


def topology():
    bm = current_bmesh()
    if bm is None:
        return None
    return len(bm.verts), len(bm.edges), len(bm.faces)


def modal_identifiers():
    identifiers = []
    try:
        for operator in STATE["window"].modal_operators:
            identifiers.extend(
                identifier
                for identifier in (
                    getattr(getattr(operator, "bl_rna", None), "identifier", ""),
                    getattr(operator, "bl_idname", ""),
                )
                if identifier
            )
    except Exception:
        pass
    return tuple(identifiers)


def production_is_busy():
    identifiers = tuple(identifier.upper() for identifier in modal_identifiers())
    native_modal = any(
        token in identifier
        for identifier in identifiers
        for token in ("LOOPCUT_SLIDE", "MESH_OT_LOOPCUT", "EDGE_SLIDE")
    )
    return bool(
        native_modal
        or operators._SESSIONS
        or operators._HISTORY_REPAIR_QUEUED
        or operators._HISTORY_REPAIR_BUSY
    )


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
    region_3d.view_distance = 5.0
    region_3d.update()


def make_mesh():
    for old in tuple(bpy.data.objects):
        bpy.data.objects.remove(old, do_unlink=True)

    mesh = bpy.data.meshes.new(MESH_NAME)
    mesh.from_pydata(
        [
            (-2.0, -1.0, 0.0),
            (-1.0, -1.0, 0.0),
            (-1.0, 1.0, 0.0),
            (-2.0, 1.0, 0.0),
            (1.0, -1.0, 0.0),
            (2.0, -1.0, 0.0),
            (2.0, 1.0, 0.0),
            (1.0, 1.0, 0.0),
        ],
        [],
        [(0, 1, 2, 3), (4, 5, 6, 7)],
    )
    mesh.update()
    obj = bpy.data.objects.new(OBJECT_NAME, mesh)
    bpy.context.scene.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    obj.use_mesh_mirror_x = True
    obj.use_mesh_mirror_y = False
    obj.use_mesh_mirror_z = False
    return obj


def window_coordinate(coordinate):
    local = view3d_utils.location_3d_to_region_2d(
        STATE["region"],
        STATE["area"].spaces.active.region_3d,
        Vector(coordinate),
    )
    if local is None:
        raise RuntimeError(f"Could not project test point {coordinate}")
    return (
        int(round(STATE["region"].x + local.x)),
        int(round(STATE["region"].y + local.y)),
    )


def ctrl_r_route_ready():
    route_keys = {
        route.route_key
        for route in keymaps._ROUTES_BY_KEY.values()
        if route.native_operator == "mesh.loopcut_slide"
        and route.keymap_name == "Mesh"
        and route.event.type == "R"
        and route.event.value == "PRESS"
        and route.event.ctrl
        and not route.event.shift
        and not route.event.alt
    }
    if not route_keys:
        return False
    return any(
        item.active
        and item.idname == keymaps.INTERCEPT_OPERATOR
        and getattr(item.properties, "route_key", "") in route_keys
        for _keymap, item in keymaps._REGISTERED_ITEMS
    )


def own_token() -> int:
    committed = [
        record
        for record in operators._HISTORY_RECORDS.values()
        if record.status == "COMMITTED" and record.session.tool_kind == "LOOP_CUT"
    ]
    assert committed, "expected at least one COMMITTED LOOP_CUT record"
    return committed[-1].session.history_token


def latest_committed_loopcut():
    committed = [
        record
        for record in operators._HISTORY_RECORDS.values()
        if record.status == "COMMITTED" and record.session.tool_kind == "LOOP_CUT"
    ]
    assert committed
    return committed[-1]


def cancel_repair_queue() -> None:
    if bpy.app.timers.is_registered(operators._repair_history_state):
        bpy.app.timers.unregister(operators._repair_history_state)
    operators._HISTORY_REPAIR_QUEUED = False
    operators._HISTORY_REPAIR_BUSY = False


def cleanup_after_case(*, keep_history_record: bool = False) -> None:
    """Restore mesh layers and sessions so the next matrix row starts clean."""
    cancel_repair_queue()
    window_pointer = STATE["window"].as_pointer()
    if window_pointer in operators._SESSIONS:
        operators.cleanup_session(window_pointer, keep_history_record=keep_history_record)
    # Drop any extra sessions that prepare may have parked under other keys.
    for pointer in tuple(operators._SESSIONS):
        operators.cleanup_session(pointer, keep_history_record=keep_history_record)
    obj = current_object()
    if obj is not None and obj.mode == "EDIT":
        bm = bmesh.from_edit_mesh(obj.data)
        core.remove_temporary_layers(bm)
        bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
    # prepare_session may have suspended mirror flags; restore baseline.
    if obj is not None:
        obj.use_mesh_mirror_x = True
        obj.use_mesh_mirror_y = False
        obj.use_mesh_mirror_z = False


def update_edit_mesh(bm) -> None:
    obj = current_object()
    assert obj is not None
    bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)


def ensure_layers(bm, *, edge=True, face_id=True, token=True):
    """Create requested temporary layers (without prepare_topology side effects)."""
    edge_layer = None
    face_layer = None
    token_layer = None
    if edge:
        edge_layer = bm.edges.layers.int.get(core.EDGE_ORIGINAL_LAYER)
        if edge_layer is None:
            edge_layer = bm.edges.layers.int.new(core.EDGE_ORIGINAL_LAYER)
    if face_id:
        face_layer = bm.faces.layers.int.get(core.FACE_ID_LAYER)
        if face_layer is None:
            face_layer = bm.faces.layers.int.new(core.FACE_ID_LAYER)
    if token:
        token_layer = bm.faces.layers.int.get(core.HISTORY_TOKEN_LAYER)
        if token_layer is None:
            token_layer = bm.faces.layers.int.new(core.HISTORY_TOKEN_LAYER)
    return edge_layer, face_layer, token_layer


def _paint_tokens(faces, token_layer, tokens) -> None:
    if len(tokens) == 1:
        for face in faces:
            face[token_layer] = tokens[0]
        return
    half = max(1, len(faces) // 2)
    for index, face in enumerate(faces):
        face[token_layer] = tokens[0] if index < half else tokens[1]


def paint_absent(bm, tokens) -> None:
    """ABSENT: both layers present, every edge marker ≥1, unique FACE_IDs per face."""
    edge_layer, face_layer, token_layer = ensure_layers(bm, edge=True, face_id=True, token=True)
    for index, edge in enumerate(bm.edges, start=1):
        edge[edge_layer] = index
    faces = list(bm.faces)
    for index, face in enumerate(faces, start=1):
        face[face_layer] = index
    _paint_tokens(faces, token_layer, tokens)
    assert core.native_path_edge_state(bm) == "ABSENT", core.native_path_edge_state(bm)


def paint_present_marker_zero(bm, tokens) -> None:
    """PRESENT via marker 0 path edge."""
    edge_layer, face_layer, token_layer = ensure_layers(bm, edge=True, face_id=True, token=True)
    for index, edge in enumerate(bm.edges, start=1):
        edge[edge_layer] = index
    if bm.edges:
        next(iter(bm.edges))[edge_layer] = 0
    faces = list(bm.faces)
    for index, face in enumerate(faces, start=1):
        face[face_layer] = index
    _paint_tokens(faces, token_layer, tokens)
    assert core.native_path_edge_state(bm) == "PRESENT", core.native_path_edge_state(bm)


def paint_present_face_id_complement(bm, tokens) -> None:
    """PRESENT via positive-marker inheritance + shared FACE_ID on both sides."""
    edge_layer, face_layer, token_layer = ensure_layers(bm, edge=True, face_id=True, token=True)
    for index, edge in enumerate(bm.edges, start=1):
        edge[edge_layer] = index  # all non-zero
    faces = list(bm.faces)
    assert len(faces) >= 2, "need at least two faces for complement path edge"
    shared = next((edge for edge in bm.edges if len(edge.link_faces) >= 2), None)
    assert shared is not None, "no internal edge with two link faces"
    shared_faces = list(shared.link_faces)
    for index, face in enumerate(faces, start=1):
        face[face_layer] = index
    shared_id = 9001
    for face in shared_faces:
        face[face_layer] = shared_id
    _paint_tokens(faces, token_layer, tokens)
    assert all(edge[edge_layer] != 0 for edge in bm.edges)
    assert core.native_path_edge_state(bm) == "PRESENT", core.native_path_edge_state(bm)


def paint_token_layer_only(bm, token: int) -> None:
    """UNKNOWN: history token only (no EDGE_ORIGINAL / FACE_ID)."""
    core.remove_temporary_layers(bm)
    token_layer = bm.faces.layers.int.new(core.HISTORY_TOKEN_LAYER)
    for face in bm.faces:
        face[token_layer] = token
    assert core.native_path_edge_state(bm) == "UNKNOWN", core.native_path_edge_state(bm)


def paint_edge_without_face_id(bm, token: int) -> None:
    """UNKNOWN: EDGE layer present, FACE_ID absent."""
    core.remove_temporary_layers(bm)
    edge_layer = bm.edges.layers.int.new(core.EDGE_ORIGINAL_LAYER)
    token_layer = bm.faces.layers.int.new(core.HISTORY_TOKEN_LAYER)
    for index, edge in enumerate(bm.edges, start=1):
        edge[edge_layer] = index
    for face in bm.faces:
        face[token_layer] = token
    assert core.native_path_edge_state(bm) == "UNKNOWN", core.native_path_edge_state(bm)


def repair_queue_armed() -> bool:
    return bool(
        operators._HISTORY_REPAIR_QUEUED
        or bpy.app.timers.is_registered(operators._repair_history_state)
    )


def run_case(name: str, builder, expect_true: bool, *, expect_repair_queue: bool = False) -> None:
    STATE["phase"] = f"matrix:{name}"
    cleanup_after_case()
    obj = current_object()
    assert obj is not None and obj.mode == "EDIT"
    bm = bmesh.from_edit_mesh(obj.data)
    builder(bm)
    update_edit_mesh(bm)

    sessions_before = dict(operators._SESSIONS)
    assert not sessions_before, sessions_before

    # prepare_session needs window/area/region (same as production F9 path).
    with bpy.context.temp_override(
        window=STATE["window"],
        area=STATE["area"],
        region=STATE["region"],
    ):
        result = operators._prepare_adjust_last_operation_repeat()
    sessions_after = dict(operators._SESSIONS)

    try:
        if expect_true:
            assert result is True, f"{name}: expected True, got {result}"
            assert sessions_after, f"{name}: expected a prepared session"
            if expect_repair_queue:
                assert repair_queue_armed(), f"{name}: expected repair queue"
        else:
            assert result is False, f"{name}: expected False, got {result}"
            assert not sessions_after, f"{name}: sessions leaked: {list(sessions_after)}"
            assert not repair_queue_armed(), f"{name}: decline must not arm the repair queue here"
        record_case(name, True)
    except AssertionError:
        record_case(name, False)
        raise
    finally:
        cleanup_after_case()


def reset_to_dual_quads(obj) -> None:
    """Replace edit-mesh geometry with the dual-quad baseline (same datablock)."""
    bm = bmesh.from_edit_mesh(obj.data)
    bm.clear()
    coords = [
        (-2.0, -1.0, 0.0),
        (-1.0, -1.0, 0.0),
        (-1.0, 1.0, 0.0),
        (-2.0, 1.0, 0.0),
        (1.0, -1.0, 0.0),
        (2.0, -1.0, 0.0),
        (2.0, 1.0, 0.0),
        (1.0, 1.0, 0.0),
    ]
    verts = [bm.verts.new(co) for co in coords]
    bm.verts.ensure_lookup_table()
    bm.faces.new((verts[0], verts[1], verts[2], verts[3]))
    bm.faces.new((verts[4], verts[5], verts[6], verts[7]))
    bm.normal_update()
    bmesh.update_edit_mesh(obj.data, loop_triangles=True, destructive=True)


def source_cut_left_vertical(obj) -> None:
    """Split the left-side face with a vertical path (asymmetric raw cut).

    LOOP_CUT finish discovers path edges with selected_only=True, so the new
    path edge must be selected (native Loop Cut leaves its ring selected).
    """
    bm = bmesh.from_edit_mesh(obj.data)
    edge_layer = bm.edges.layers.int.get(core.EDGE_ORIGINAL_LAYER)
    # Prefer the outer-left bottom edge when several candidates exist.
    bottom_candidates = [
        edge
        for edge in bm.edges
        if all(vertex.co.x < 0.0 for vertex in edge.verts)
        and all(abs(vertex.co.y + 1.0) < 1.0e-6 for vertex in edge.verts)
    ]
    bottom = min(bottom_candidates, key=lambda e: min(v.co.x for v in e.verts))
    _edge, a = bmesh.utils.edge_split(bottom, bottom.verts[0], 0.5)
    top_candidates = [
        edge
        for edge in bm.edges
        if all(vertex.co.x < 0.0 for vertex in edge.verts)
        and all(abs(vertex.co.y - 1.0) < 1.0e-6 for vertex in edge.verts)
    ]
    top = min(top_candidates, key=lambda e: min(v.co.x for v in e.verts))
    _edge, b = bmesh.utils.edge_split(top, top.verts[0], 0.5)
    face = next(face for face in bm.faces if a in face.verts and b in face.verts)
    bmesh.utils.face_split(face, a, b)
    path_edge = bm.edges.get((a, b))
    for edge in bm.edges:
        edge.select = False
    for vertex in bm.verts:
        vertex.select = False
    for face_item in bm.faces:
        face_item.select = False
    if path_edge is not None:
        path_edge.select = True
        for vertex in path_edge.verts:
            vertex.select = True
    elif edge_layer is not None:
        for edge in bm.edges:
            if int(edge[edge_layer]) == 0:
                edge.select = True
                for vertex in edge.verts:
                    vertex.select = True
    bm.select_flush_mode()
    bmesh.update_edit_mesh(obj.data, loop_triangles=True, destructive=True)


def mesh_is_symmetric(obj, *, decimals=5) -> bool:
    bm = bmesh.from_edit_mesh(obj.data)
    verts = sorted(tuple(round(float(c), decimals) for c in v.co) for v in bm.verts)
    mirrored = sorted((round(-x, decimals), y, z) for x, y, z in verts)
    edges = sorted(
        tuple(sorted(tuple(round(float(c), decimals) for c in v.co) for v in e.verts))
        for e in bm.edges
    )
    mirrored_edges = sorted(
        tuple(
            sorted(
                (round(-x, decimals), y, z)
                for x, y, z in (tuple(round(float(c), decimals) for c in v.co) for v in e.verts)
            )
        )
        for e in bm.edges
    )
    return verts == mirrored and edges == mirrored_edges


def temporary_layers_present(obj) -> bool:
    bm = bmesh.from_edit_mesh(obj.data)
    return any(
        layers.get(name) is not None
        for layers, name in (
            (bm.edges.layers.int, core.EDGE_ORIGINAL_LAYER),
            (bm.faces.layers.int, core.FACE_ID_LAYER),
            (bm.faces.layers.int, core.HISTORY_TOKEN_LAYER),
        )
    )


def run_repair_cases() -> None:
    """§C-2: multi-token repair candidate selection."""
    obj = current_object()
    assert obj is not None
    own = own_token()
    record = latest_committed_loopcut()

    # --- Case: own raw + foreign debris → finish + symmetrize ---
    STATE["phase"] = "repair:own_raw_plus_foreign_debris"
    cleanup_after_case()
    try:
        reset_to_dual_quads(obj)
        bm = bmesh.from_edit_mesh(obj.data)
        # prepare_topology writes face maps into CustomData; finish restores them.
        topology_prep = core.prepare_topology(bm, 0, 1.0e-5, own)
        assert topology_prep.matched_faces > 0
        bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
        source_cut_left_vertical(obj)

        bm = bmesh.from_edit_mesh(obj.data)
        token_layer = bm.faces.layers.int.get(core.HISTORY_TOKEN_LAYER)
        assert token_layer is not None
        faces = list(bm.faces)
        assert faces
        # Paint foreign token on half the faces; own remains on the rest.
        for index, face in enumerate(faces):
            face[token_layer] = FOREIGN_TOKEN if index % 2 == 0 else own
        # Ensure at least one face keeps the own token (candidate).
        faces[0][token_layer] = own
        bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)

        assert own in operators._object_history_tokens(obj)
        assert FOREIGN_TOKEN in operators._object_history_tokens(obj)
        assert FOREIGN_TOKEN not in operators._HISTORY_RECORDS
        assert operators._HISTORY_RECORDS[own].status == "COMMITTED"
        assert core.native_path_edge_state(bmesh.from_edit_mesh(obj.data)) == "PRESENT"
        assert not operators._SESSIONS

        verts_before = len(bmesh.from_edit_mesh(obj.data).verts)
        operators._repair_history_state()
        # Finish may be async-free (EXEC_DEFAULT); sessions should be cleared.
        cancel_repair_queue()
        for pointer in tuple(operators._SESSIONS):
            operators.cleanup_session(pointer, keep_history_record=True)

        bm = bmesh.from_edit_mesh(obj.data)
        layers_gone = bm.edges.layers.int.get(core.EDGE_ORIGINAL_LAYER) is None
        sym = mesh_is_symmetric(obj)
        # Finish should have run: either layers cleaned via finish path and
        # topology grew on the mirror side, or symmetry holds.
        verts_after = len(bm.verts)
        assert layers_gone, "finish should remove temporary layers"
        assert sym or verts_after > verts_before, (
            f"expected finish/symmetrize, sym={sym} verts {verts_before}->{verts_after}"
        )
        assert sym, f"repair finish did not symmetrize (verts={verts_after})"
        record_case("repair_own_raw_plus_foreign", True)
    except Exception:
        record_case("repair_own_raw_plus_foreign", False)
        raise
    finally:
        cleanup_after_case(keep_history_record=True)

    # --- Case: two COMMITTED tokens mixed → cleanup-only ---
    STATE["phase"] = "repair:two_committed_cleanup_only"
    cleanup_after_case(keep_history_record=True)
    second_token = None
    try:
        second_token = operators._new_history_token()
        second_session = copy.deepcopy(record.session)
        second_session.history_token = second_token
        operators._HISTORY_RECORDS[second_token] = HistoryRecord(
            session=second_session,
            status="COMMITTED",
            sequence=record.sequence + 1000,
        )

        reset_to_dual_quads(obj)
        bm = bmesh.from_edit_mesh(obj.data)
        topology_prep = core.prepare_topology(bm, 0, 1.0e-5, own)
        assert topology_prep.matched_faces > 0
        bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
        source_cut_left_vertical(obj)

        bm = bmesh.from_edit_mesh(obj.data)
        token_layer = bm.faces.layers.int.get(core.HISTORY_TOKEN_LAYER)
        assert token_layer is not None
        faces = list(bm.faces)
        half = max(1, len(faces) // 2)
        for index, face in enumerate(faces):
            face[token_layer] = own if index < half else second_token
        bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)

        tokens = operators._object_history_tokens(obj)
        assert own in tokens and second_token in tokens, tokens
        assert not operators._SESSIONS
        assert temporary_layers_present(obj)

        topology_before = topology()
        operators._repair_history_state()
        cancel_repair_queue()

        # cleanup-only: layers gone, topology unchanged (no finish mirror).
        assert not temporary_layers_present(obj), "cleanup-only should strip layers"
        assert topology() == topology_before, (topology(), topology_before)
        record_case("repair_two_committed_cleanup_only", True)
    except Exception:
        record_case("repair_two_committed_cleanup_only", False)
        raise
    finally:
        if second_token is not None:
            operators._HISTORY_RECORDS.pop(second_token, None)
        cleanup_after_case(keep_history_record=True)


def run_matrix() -> None:
    own = own_token()

    run_case(
        "no_tokens_clean",
        lambda bm: core.remove_temporary_layers(bm),
        expect_true=True,
        expect_repair_queue=False,
    )
    run_case(
        "own_present_marker_zero",
        lambda bm: paint_present_marker_zero(bm, (own,)),
        expect_true=False,
    )
    run_case(
        "own_present_face_id_complement",
        lambda bm: paint_present_face_id_complement(bm, (own,)),
        expect_true=False,
    )
    run_case(
        "own_absent",
        lambda bm: paint_absent(bm, (own,)),
        expect_true=True,
        expect_repair_queue=True,
    )
    run_case(
        "foreign_absent",
        lambda bm: paint_absent(bm, (FOREIGN_TOKEN,)),
        expect_true=True,
        expect_repair_queue=True,
    )
    run_case(
        "own_foreign_present",
        lambda bm: paint_present_marker_zero(bm, (own, FOREIGN_TOKEN)),
        expect_true=False,
    )
    run_case(
        "token_layer_only_unknown",
        lambda bm: paint_token_layer_only(bm, own),
        expect_true=False,
    )
    run_case(
        "edge_without_face_id_unknown",
        lambda bm: paint_edge_without_face_id(bm, own),
        expect_true=False,
    )
    run_repair_cases()


def begin_loopcut():
    try:
        STATE["phase"] = "native Ctrl+R loop cut for COMMITTED record"
        hover = window_coordinate((-1.5, -1.0, 0.0))
        slide = window_coordinate((-1.68, 0.0, 0.0))
        STATE["events"] = [
            {"type": "MOUSEMOVE", "value": "NOTHING", "x": hover[0], "y": hover[1]},
            {"type": "R", "value": "PRESS", "ctrl": True, "x": hover[0], "y": hover[1]},
            {"type": "R", "value": "RELEASE", "ctrl": True, "x": hover[0], "y": hover[1]},
            {"type": "WHEELUPMOUSE", "value": "PRESS", "x": hover[0], "y": hover[1]},
            {"type": "LEFTMOUSE", "value": "PRESS", "x": hover[0], "y": hover[1]},
            {"type": "LEFTMOUSE", "value": "RELEASE", "x": hover[0], "y": hover[1]},
            {"type": "MOUSEMOVE", "value": "NOTHING", "x": slide[0], "y": slide[1]},
            {"type": "LEFTMOUSE", "value": "PRESS", "x": slide[0], "y": slide[1]},
            {"type": "LEFTMOUSE", "value": "RELEASE", "x": slide[0], "y": slide[1]},
        ]
        bpy.app.timers.register(send_next_event, first_interval=0.05)
    except BaseException:
        fail()
    return None


def send_next_event():
    try:
        if STATE["events"]:
            STATE["window"].event_simulate(**STATE["events"].pop(0))
            return 0.15
        STATE["phase"] = "wait for loop cut finish"
        STATE["deadline"] = time.monotonic() + 12.0
        bpy.app.timers.register(wait_for_loopcut, first_interval=0.05)
    except BaseException:
        fail()
    return None


def wait_for_loopcut():
    try:
        if production_is_busy() or topology() != FINISHED_TOPOLOGY:
            if time.monotonic() > STATE["deadline"]:
                raise RuntimeError(
                    f"Timed out waiting for Loop Cut: topology={topology()} "
                    f"sessions={list(operators._SESSIONS)}"
                )
            return 0.05

        active = bpy.context.active_operator
        assert active is not None, "active_operator required for A-1 eligibility"
        identifier = getattr(active, "bl_idname", "")
        assert "LOOPCUT" in identifier.upper(), identifier
        committed = [
            record
            for record in operators._HISTORY_RECORDS.values()
            if record.status == "COMMITTED" and record.session.tool_kind == "LOOP_CUT"
        ]
        assert committed, "no COMMITTED LOOP_CUT record after cut"
        assert committed[-1].session.native_operator_pointer == int(active.as_pointer())
        assert not operators._SESSIONS
        print(
            f"YSE_F9_MATRIX_SETUP own_token={committed[-1].session.history_token} "
            f"topology={topology()} active={identifier}",
            flush=True,
        )
        run_matrix()
        failed = [name for name, status in STATE["case_results"].items() if status != "PASS"]
        if failed:
            raise RuntimeError(f"failed cases: {failed}")
        print(MARK_OK, flush=True)
        addon.unregister()
        STATE["addon_registered"] = False
        bpy.ops.wm.quit_blender()
    except BaseException:
        fail()
    return None


def wait_for_route():
    try:
        if ctrl_r_route_ready():
            bpy.app.timers.register(begin_loopcut, first_interval=0.2)
            return None
        if time.monotonic() > STATE["deadline"]:
            raise RuntimeError("Ctrl+R Loop Cut intercept was not registered")
        return 0.05
    except BaseException:
        fail()
    return None


def start_test():
    try:
        STATE["phase"] = "setup"
        addon.register()
        STATE["addon_registered"] = True
        addon.sync_persistent_keymap(True)

        window, area, region = viewport_context()
        STATE.update(window=window, area=area, region=region)
        configure_view(area)
        obj = make_mesh()
        with bpy.context.temp_override(window=window, area=area, region=region):
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.context.tool_settings.mesh_select_mode = (False, True, False)
            bpy.context.tool_settings.workspace_tool_type = "DEFAULT"
            bpy.ops.mesh.select_all(action="DESELECT")
            result = bpy.ops.wm.tool_set_by_id(name="builtin.select_box")
            assert result == {"FINISHED"}, result
            bpy.ops.ed.undo_push(message="YSE F9 matrix baseline")

        assert core.enabled_mesh_symmetry_axes(obj) == (("X", 0),)
        STATE["deadline"] = time.monotonic() + 5.0
        bpy.app.timers.register(wait_for_route, first_interval=0.05)
    except BaseException:
        fail()
    return None


bpy.app.timers.register(start_test, first_interval=0.25)
