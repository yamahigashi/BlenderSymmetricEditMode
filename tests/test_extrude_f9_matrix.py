# SPDX-License-Identifier: GPL-3.0-or-later

"""GUI unit matrix for the ownership-free Extrude F9 baseline gate.

Run::

    cmd.exe /c "tmp\\run_menu_reg.bat 42 test_extrude_f9_matrix.py"

Contract: .agents/doc/f9_extrude_plan_2026-08-15.md v3.1 §4-1.
Pattern: paint mesh state, then call the production helper/handler directly.
"""

from __future__ import annotations

import os
import statistics
import sys
import time
import traceback
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import bmesh
import bpy

bpy.context.preferences.view.show_splash = False
bpy.context.preferences.view.smooth_view = 0
bpy.context.preferences.view.use_save_prompt = False
bpy.context.preferences.filepaths.use_auto_save_temporary_files = False

PACKAGE_PARENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_PARENT))

from ydd_symmetric_edit import extrude, history, layer_names, session, session_state, snapshot  # noqa: E402
from ydd_symmetric_edit._types import (  # noqa: E402
    Coordinate3D,
    ExtrudeFreezeEntry,
    HistoryRecord,
    MeshSelectionMode,
)
from ydd_symmetric_edit.extrude import ExtrudeClassification  # noqa: E402

OBJECT_NAME = "YSE_ExtrudeF9Matrix"
MESH_NAME = "YSE_ExtrudeF9MatrixMesh"
OWN_TOKEN = 10101
FOREIGN_TOKEN = 20202
MARK_OK = "YSE_EXTRUDE_F9_MATRIX_OK"
MARK_FAIL = "YSE_EXTRUDE_F9_MATRIX_FAILED"

STATE = {"case_results": {}, "phase": "startup"}


def record_case(name: str, passed: bool) -> None:
    status = "PASS" if passed else "FAIL"
    STATE["case_results"][name] = status
    print(f"YSE_EXTRUDE_F9_MATRIX_RESULT={name}:{status}", flush=True)


def run_case(name: str, body) -> None:
    STATE["phase"] = name
    try:
        body()
    except BaseException:
        record_case(name, False)
        raise
    record_case(name, True)


def fail() -> None:
    traceback.print_exc()
    print(f"YSE_EXTRUDE_F9_MATRIX_PHASE={STATE['phase']}", flush=True)
    for name, status in STATE["case_results"].items():
        print(f"YSE_EXTRUDE_F9_MATRIX_RESULT={name}:{status}", flush=True)
    print(MARK_FAIL, flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(1)


def remove_objects() -> None:
    if bpy.context.object is not None and bpy.context.object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    for old in tuple(bpy.data.objects):
        bpy.data.objects.remove(old, do_unlink=True)


def make_baseline():
    """Two quads plus two loose edges; the latter permit edge-only rewiring."""

    remove_objects()
    mesh = bpy.data.meshes.new(MESH_NAME)
    mesh.from_pydata(
        [
            (-3.0, -1.0, 0.0),
            (-1.0, -1.0, 0.0),
            (-1.0, 1.0, 0.0),
            (-3.0, 1.0, 0.0),
            (1.0, -1.0, 0.0),
            (3.0, -1.0, 0.0),
            (3.0, 1.0, 0.0),
            (1.0, 1.0, 0.0),
            (-0.75, -2.0, 0.0),
            (-0.25, -2.0, 0.0),
            (0.25, -2.0, 0.0),
            (0.75, -2.0, 0.0),
        ],
        [(8, 9), (10, 11)],
        [(0, 1, 2, 3), (4, 5, 6, 7)],
    )
    mesh.update()
    obj = bpy.data.objects.new(OBJECT_NAME, mesh)
    bpy.context.scene.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    return obj, bmesh.from_edit_mesh(mesh)


def update_mesh(obj, *, destructive: bool = False) -> None:
    bmesh.update_edit_mesh(obj.data, loop_triangles=destructive, destructive=destructive)


def selection_mode(scope: str) -> MeshSelectionMode:
    values = {
        "vertex": (True, False, False),
        "edge": (False, True, False),
        "face": (False, False, True),
    }[scope]
    bpy.context.tool_settings.mesh_select_mode = values
    return MeshSelectionMode(vertices=values[0], edges=values[1], faces=values[2])


def paint_selection(bm, scope: str) -> None:
    for vertex in bm.verts:
        vertex.select = False
    for edge in bm.edges:
        edge.select = False
    for face in bm.faces:
        face.select = False
    if scope == "vertex":
        bm.verts.ensure_lookup_table()
        bm.verts[0].select = True
    elif scope == "edge":
        bm.edges.ensure_lookup_table()
        bm.edges[0].select = True
        for vertex in bm.edges[0].verts:
            vertex.select = True
    else:
        bm.faces.ensure_lookup_table()
        face = bm.faces[0]
        face.select = True
        for edge in face.edges:
            edge.select = True
        for vertex in face.verts:
            vertex.select = True


def add_snapshot_layers(bm) -> None:
    vertex_layer = bm.verts.layers.int.new(layer_names.VERT_SESSION_ID_LAYER)
    edge_layer = bm.edges.layers.int.new(layer_names.EDGE_ORIGINAL_LAYER)
    face_layer = bm.faces.layers.int.new(layer_names.FACE_ID_LAYER)
    for index, vertex in enumerate(bm.verts, start=1):
        vertex[vertex_layer] = index
    for index, edge in enumerate(bm.edges, start=1):
        edge[edge_layer] = index
    for index, face in enumerate(bm.faces, start=1):
        face[face_layer] = index


def capture_baseline(scope: str):
    obj, bm = make_baseline()
    mode = selection_mode(scope)
    paint_selection(bm, scope)
    add_snapshot_layers(bm)
    captured = extrude.build_snapshot(
        bm,
        0,
        1.0e-4,
        tool_kind="EXTRUDE_NORMAL",
        route_kmi_properties=(),
        mesh_select_mode=mode,
        mesh_object=obj,
    )
    assert captured is not None
    snapshot.remove_temporary_layers(bm)
    update_mesh(obj)
    return obj, bm, captured, mode


def paint_token_state(bm, state: str) -> None:
    old = bm.faces.layers.int.get(layer_names.HISTORY_TOKEN_LAYER)
    if old is not None:
        bm.faces.layers.int.remove(old)
    if state == "no_token":
        return
    layer = bm.faces.layers.int.new(layer_names.HISTORY_TOKEN_LAYER)
    faces = list(bm.faces)
    assert len(faces) >= 2
    for index, face in enumerate(faces):
        if state == "own":
            token = OWN_TOKEN
        elif state == "foreign":
            token = FOREIGN_TOKEN
        else:
            token = OWN_TOKEN if index % 2 == 0 else FOREIGN_TOKEN
        face[layer] = token


def baseline_matches(bm, captured, mode) -> bool:
    return extrude.repeat_baseline_matches(bm, captured, mesh_select_mode=mode)


def view3d_override():
    window = bpy.context.window_manager.windows[0]
    area = next(area for area in window.screen.areas if area.type == "VIEW_3D")
    region = next(region for region in area.regions if region.type == "WINDOW")
    return bpy.context.temp_override(window=window, area=area, region=region)


def _operator_history_ids() -> tuple[str, ...]:
    return tuple(getattr(operator, "bl_idname", "") for operator in bpy.context.window_manager.operators)


def last_redo_operator():
    """Return the F9 last-redo operator the production helper will see.

    ``bpy.ops`` EXEC from a timer does not become ``active_operator`` unless
    undo-register is requested (the F3 path).  ``window_manager.operators`` is
    the same stack ``context.active_operator`` reads, so a miss here is a
    real registration failure rather than a stale context pointer.
    """

    operator = bpy.context.active_operator
    if operator is not None:
        return operator
    recent = list(bpy.context.window_manager.operators)
    return recent[-1] if recent else None


def invoke_native_extrude_region():
    """Execute the native macro so it becomes the last-redo operator (F3/EXEC)."""

    with view3d_override():
        result = bpy.ops.mesh.extrude_region_move(
            "EXEC_DEFAULT",
            True,
            TRANSFORM_OT_translate={
                "value": (0.0, 0.0, 0.25),
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
    operator = last_redo_operator()
    assert operator is not None, f"native extrude did not register last-redo; wm.operators={_operator_history_ids()}"
    identifier = getattr(operator, "bl_idname", "").upper()
    pointer = int(operator.as_pointer())
    assert identifier == "MESH_OT_EXTRUDE_REGION_MOVE", identifier
    assert bpy.context.active_operator is not None, (
        f"active_operator still None after undo-registered EXEC; wm.operators={_operator_history_ids()}"
    )
    assert int(bpy.context.active_operator.as_pointer()) == pointer
    return identifier, pointer


def call_prepare_adjust():
    """Call the production helper in the same VIEW3D context F9 undo_post uses."""

    with view3d_override():
        return history._prepare_adjust_last_operation_repeat()


def call_repair_after_undo():
    with view3d_override():
        history.repair_after_undo()


def native_extrude_identity_and_restore(obj, captured, scope: str):
    """Run a real native extrude, then repaint the raw snapshot in-place."""

    bm = bmesh.from_edit_mesh(obj.data)
    selection_mode("face")
    paint_selection(bm, "face")
    update_mesh(obj)
    identifier, pointer = invoke_native_extrude_region()

    bm = bmesh.from_edit_mesh(obj.data)
    bm.clear()
    vertices = {vertex_id: bm.verts.new(coordinate.as_tuple()) for vertex_id, coordinate in captured.vertex_preop}
    # Rebuild edges before faces so bm.edges[i] matches the snapshot order
    # that paint_selection used at capture time (from_pydata loose edges first).
    for _marker, (first_id, second_id) in captured.edge_endpoints:
        first = vertices[first_id]
        second = vertices[second_id]
        if bm.edges.get((first, second)) is None:
            bm.edges.new((first, second))
    for _face_id, corners in captured.face_corners:
        bm.faces.new(tuple(vertices[vertex_id] for vertex_id in corners))
    selection_mode(scope)
    paint_selection(bm, scope)
    update_mesh(obj, destructive=True)
    assert bpy.context.active_operator is not None
    assert int(bpy.context.active_operator.as_pointer()) == pointer
    return bmesh.from_edit_mesh(obj.data), identifier, pointer


def ownership_row(scope: str, token_state: str, should_match: bool) -> None:
    obj, _bm, captured, _mode = capture_baseline(scope)
    bm, identifier, pointer = native_extrude_identity_and_restore(obj, captured, scope)
    paint_token_state(bm, token_state)
    if not should_match:
        bm.verts.ensure_lookup_table()
        bm.verts[0].co.x += captured.tolerance * 4.0
    update_mesh(obj)

    original_mapping = dict(session._WM_OPERATOR_TO_TOOL)
    original_records = session_state._HISTORY_RECORDS.copy()
    original_prepare = history._prepare_session
    original_queue = history._queue_history_repair
    prepare_calls = []
    queue_calls = []
    fake = SimpleNamespace(
        tool_kind="EXTRUDE_NORMAL",
        window_pointer=bpy.context.window_manager.windows[0].as_pointer(),
        object_name=obj.name,
        mesh_name=obj.data.name,
        native_operator_pointer=pointer,
        extrude=captured,
    )
    try:
        session._WM_OPERATOR_TO_TOOL[identifier] = "EXTRUDE_NORMAL"
        session_state._HISTORY_RECORDS.clear()
        session_state._HISTORY_RECORDS[OWN_TOKEN] = HistoryRecord(
            session=fake,
            status="COMMITTED",
            sequence=1,
        )
        history._prepare_session = lambda *_args, **kwargs: prepare_calls.append(kwargs["tool_kind"]) or True
        history._queue_history_repair = lambda *_args, **_kwargs: queue_calls.append("queue")
        if should_match:
            assert call_prepare_adjust() is True
            assert prepare_calls == ["EXTRUDE_NORMAL"]
            assert queue_calls == []
        else:
            call_repair_after_undo()
            assert prepare_calls == []
            assert queue_calls == ["queue"]
            assert not session_state._SESSIONS
    finally:
        history._prepare_session = original_prepare
        history._queue_history_repair = original_queue
        session._WM_OPERATOR_TO_TOOL.clear()
        session._WM_OPERATOR_TO_TOOL.update(original_mapping)
        session_state._HISTORY_RECORDS.clear()
        session_state._HISTORY_RECORDS.update(original_records)


def edge_only_rewire() -> None:
    obj, bm, captured, mode = capture_baseline("vertex")
    bm.verts.ensure_lookup_table()
    loose = [edge for edge in bm.edges if not edge.link_faces]
    assert len(loose) == 2
    for edge in loose:
        bm.edges.remove(edge)
    bm.edges.new((bm.verts[8], bm.verts[10]))
    bm.edges.new((bm.verts[9], bm.verts[11]))
    update_mesh(obj, destructive=True)
    assert baseline_matches(bm, captured, mode) is False


def face_only_delete() -> None:
    obj, bm, captured, mode = capture_baseline("face")
    bm.faces.ensure_lookup_table()
    bm.faces.remove(bm.faces[-1])
    update_mesh(obj, destructive=True)
    assert baseline_matches(bm, captured, mode) is False


def solver_coordinate_drift() -> None:
    obj, bm, captured, mode = capture_baseline("vertex")
    bm.verts.ensure_lookup_table()
    bm.verts[-1].co.y += captured.tolerance * 0.25
    update_mesh(obj)
    assert baseline_matches(bm, captured, mode) is True


def rip_pollution() -> None:
    obj, bm, captured, mode = capture_baseline("edge")
    bm.verts.ensure_lookup_table()
    duplicate = bm.verts.new(tuple(bm.verts[0].co))
    duplicate.select = False
    update_mesh(obj, destructive=True)
    assert baseline_matches(bm, captured, mode) is False


def heterogeneous_history(kind: str) -> None:
    obj, _bm, captured, _mode = capture_baseline("vertex")
    _bm, identifier, pointer = native_extrude_identity_and_restore(obj, captured, "vertex")
    original_mapping = dict(session._WM_OPERATOR_TO_TOOL)
    original_records = session_state._HISTORY_RECORDS.copy()
    try:
        session._WM_OPERATOR_TO_TOOL[identifier] = "EXTRUDE_NORMAL"
        session_state._HISTORY_RECORDS.clear()
        if kind != "evicted_record":
            record_kind = "LOOP_CUT" if kind == "loopcut_then_extrude" else "EXTRUDE_NORMAL"
            record_pointer = pointer if kind == "loopcut_then_extrude" else pointer + 1
            fake = SimpleNamespace(
                tool_kind=record_kind,
                window_pointer=bpy.context.window_manager.windows[0].as_pointer(),
                object_name=obj.name,
                mesh_name=obj.data.name,
                native_operator_pointer=record_pointer,
                extrude=captured,
            )
            session_state._HISTORY_RECORDS[9000] = HistoryRecord(session=fake, status="COMMITTED", sequence=1)
        assert call_prepare_adjust() is False
    finally:
        session._WM_OPERATOR_TO_TOOL.clear()
        session._WM_OPERATOR_TO_TOOL.update(original_mapping)
        session_state._HISTORY_RECORDS.clear()
        session_state._HISTORY_RECORDS.update(original_records)


def excluded_recognition(kind: str) -> None:
    profile = session.TOOL_PROFILES[kind]
    assert profile.primary_wm_operator.upper() not in session._WM_OPERATOR_TO_TOOL


def selection_only_mismatch() -> None:
    obj, _bm, captured, _mode = capture_baseline("vertex")
    bm, identifier, pointer = native_extrude_identity_and_restore(obj, captured, "vertex")
    bm.verts.ensure_lookup_table()
    bm.verts[1].select = True
    update_mesh(obj)
    original_mapping = dict(session._WM_OPERATOR_TO_TOOL)
    original_records = session_state._HISTORY_RECORDS.copy()
    original_prepare = history._prepare_session
    original_queue = history._queue_history_repair
    prepare_calls = []
    queue_calls = []
    fake = SimpleNamespace(
        tool_kind="EXTRUDE_NORMAL",
        window_pointer=bpy.context.window_manager.windows[0].as_pointer(),
        object_name=obj.name,
        mesh_name=obj.data.name,
        native_operator_pointer=pointer,
        extrude=captured,
    )
    try:
        session._WM_OPERATOR_TO_TOOL[identifier] = "EXTRUDE_NORMAL"
        session_state._HISTORY_RECORDS.clear()
        session_state._HISTORY_RECORDS[OWN_TOKEN] = HistoryRecord(
            session=fake,
            status="COMMITTED",
            sequence=1,
        )
        history._prepare_session = lambda *_args, **_kwargs: prepare_calls.append("prepare") or True
        history._queue_history_repair = lambda *_args, **_kwargs: queue_calls.append("queue")
        call_repair_after_undo()
        assert prepare_calls == []
        assert queue_calls == ["queue"]
        assert not session_state._SESSIONS
    finally:
        history._prepare_session = original_prepare
        history._queue_history_repair = original_queue
        session._WM_OPERATOR_TO_TOOL.clear()
        session._WM_OPERATOR_TO_TOOL.update(original_mapping)
        session_state._HISTORY_RECORDS.clear()
        session_state._HISTORY_RECORDS.update(original_records)


def coincident_vertex_selection_mismatch() -> None:
    """Raw-exact A/B coordinates must not hide their different neighborhoods."""

    remove_objects()
    mesh = bpy.data.meshes.new(MESH_NAME)
    mesh.from_pydata(
        [
            (2.0, 0.0, 0.0),  # A
            (2.0, 0.0, 0.0),  # B, coincident with A
            (3.0, 0.0, 0.0),  # A-only neighbor
            (2.0, 2.0, 0.0),  # B-only neighbor
        ],
        [(0, 2), (1, 3)],
        [],
    )
    obj = bpy.data.objects.new(OBJECT_NAME, mesh)
    bpy.context.scene.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bm = bmesh.from_edit_mesh(mesh)
    mode = selection_mode("vertex")
    for vertex in bm.verts:
        vertex.select = False
    bm.verts.ensure_lookup_table()
    bm.verts[0].select = True
    add_snapshot_layers(bm)
    captured = extrude.build_snapshot(
        bm,
        0,
        1.0e-4,
        tool_kind="EXTRUDE_NORMAL",
        route_kmi_properties=(),
        mesh_select_mode=mode,
        mesh_object=obj,
    )
    assert captured is not None
    snapshot.remove_temporary_layers(bm)
    bm.verts[0].select = False
    bm.verts[1].select = True
    update_mesh(obj)
    assert baseline_matches(bm, captured, mode) is False


def repair_translation_case(*, repair_busy: bool, signature_faces: int, expect_success: bool) -> None:
    bm = bmesh.new()
    try:
        # Layers first: adding CustomData after verts invalidates BMVert wrappers.
        vertex_layer = bm.verts.layers.int.new(layer_names.VERT_SESSION_ID_LAYER)
        face_layer = bm.faces.layers.int.new(layer_names.FACE_ID_LAYER)
        origin = bm.verts.new((0.0, 0.0, 0.0))
        copy = bm.verts.new((0.0, 0.0, 1.0))
        origin[vertex_layer] = 1
        copy[vertex_layer] = 1
        face_count = max(1, signature_faces)
        for index in range(face_count):
            second = bm.verts.new((1.0 + index, 0.0, 1.0))
            third = bm.verts.new((0.0, 1.0 + index, 1.0))
            second[vertex_layer] = 2 if signature_faces else 4
            third[vertex_layer] = 3 if signature_faces else 5
            face = bm.faces.new((copy, second, third))
            face[face_layer] = 999  # Deliberately unrelated to frozen FACE_ID 10.
        captured = SimpleNamespace(
            tolerance=1.0e-4,
            selected_face_ids=frozenset({10}),
            face_corner_map=lambda: {10: (1, 2, 3)},
        )
        freeze = (
            ExtrudeFreezeEntry(
                vertex_id=1,
                entity_class="d",
                origin_preop=Coordinate3D(0.0, 0.0, 0.0),
                copy_post=Coordinate3D(0.0, 0.0, 1.0),
                source_face_signature=(1, 2, 3),
            ),
        )
        original_busy = session_state._HISTORY_REPAIR_BUSY
        try:
            session_state._HISTORY_REPAIR_BUSY = repair_busy
            classified, reason = extrude.reconnect_freeze(bm, captured, freeze)
        finally:
            session_state._HISTORY_REPAIR_BUSY = original_busy
        if expect_success:
            assert reason is None, reason
            assert classified is not None
            assert classified.copy_instances[0].vertex is copy
        else:
            assert classified is None
            assert reason == "a frozen N-copy row did not match a unique copy"
    finally:
        bm.free()


def rebuilt_source_edge_declines() -> None:
    obj, bm = make_baseline()
    mode = selection_mode("vertex")
    paint_selection(bm, "vertex")
    add_snapshot_layers(bm)
    captured = extrude.build_snapshot(
        bm,
        0,
        1.0e-4,
        tool_kind="EXTRUDE_CONTEXT",
        route_kmi_properties=(),
        mesh_select_mode=mode,
        mesh_object=obj,
    )
    assert captured is not None
    vertex_layer = bm.verts.layers.int.get(layer_names.VERT_SESSION_ID_LAYER)
    edge_layer = bm.edges.layers.int.get(layer_names.EDGE_ORIGINAL_LAYER)
    assert vertex_layer is not None and edge_layer is not None
    loose = next(edge for edge in bm.edges if not edge.link_faces)
    endpoints = tuple(loose.verts)
    old_marker = int(loose[edge_layer])
    assert old_marker > 0
    bm.edges.remove(loose)
    replacement = bm.edges.new(endpoints)
    assert int(replacement[edge_layer]) == 0
    update_mesh(obj, destructive=True)
    origins = {int(vertex[vertex_layer]): vertex for vertex in bm.verts}
    classified = ExtrudeClassification(
        origins=origins,
        copies={},
        copy_instances=(),
        vanished_preop={},
        freeze=(),
    )
    description, reason = extrude.describe_source(bm, captured, classified)
    assert description is None
    assert reason == "a surviving source edge was rebuilt"


def suzanne_performance_median() -> None:
    remove_objects()
    bpy.ops.mesh.primitive_monkey_add()
    obj = bpy.context.object
    assert obj is not None
    obj.name = OBJECT_NAME
    bpy.ops.object.mode_set(mode="EDIT")
    bm = bmesh.from_edit_mesh(obj.data)
    mode = selection_mode("face")
    for vertex in bm.verts:
        vertex.select = False
    for edge in bm.edges:
        edge.select = False
    for face in bm.faces:
        face.select = False
    add_snapshot_layers(bm)
    captured = extrude.build_snapshot(
        bm,
        0,
        1.0e-4,
        tool_kind="EXTRUDE_CONTEXT",
        route_kmi_properties=(),
        mesh_select_mode=mode,
        mesh_object=obj,
    )
    assert captured is not None
    snapshot.remove_temporary_layers(bm)
    update_mesh(obj)
    timings = []
    for _index in range(7):
        started = time.perf_counter()
        assert baseline_matches(bm, captured, mode) is True
        timings.append((time.perf_counter() - started) * 1000.0)
    median_ms = statistics.median(timings)
    print(f"YSE_EXTRUDE_F9_MATRIX_SUZANNE_MEDIAN_MS={median_ms:.6f}", flush=True)


def run_matrix() -> None:
    original_profiles = dict(session.TOOL_PROFILES)
    try:
        # W-a intentionally leaves the production flags off.  Permit only the
        # tested six kinds inside this matrix and restore the immutable profiles.
        for kind in session.EXTRUDE_TOOL_KINDS - {"EXTRUDE_MANIFOLD"}:
            session.TOOL_PROFILES[kind] = replace(
                session.TOOL_PROFILES[kind],
                supports_adjust_repeat=True,
            )

        for scope in ("vertex", "edge", "face"):
            for token_state in ("no_token", "own", "foreign", "mixed"):
                for should_match in (True, False):
                    suffix = "match" if should_match else "mismatch"
                    name = f"ownership_{scope}_{token_state}_{suffix}"
                    run_case(name, lambda s=scope, t=token_state, m=should_match: ownership_row(s, t, m))

        run_case("domain_edge_only_rewire", edge_only_rewire)
        run_case("domain_face_only_delete", face_only_delete)
        run_case("domain_coordinate_drift_solver", solver_coordinate_drift)
        run_case("domain_rip_pollution", rip_pollution)
        run_case("selection_only_mismatch", selection_only_mismatch)
        run_case("selection_coincident_different_neighborhood", coincident_vertex_selection_mismatch)
        run_case("heterogeneous_loopcut_then_extrude", lambda: heterogeneous_history("loopcut_then_extrude"))
        run_case("heterogeneous_extrude_then_loopcut", lambda: heterogeneous_history("extrude_then_loopcut"))
        run_case("heterogeneous_evicted_record", lambda: heterogeneous_history("evicted_record"))
        run_case("recognition_knife_excluded", lambda: excluded_recognition("KNIFE"))
        run_case("recognition_rip_excluded", lambda: excluded_recognition("RIP"))
        run_case(
            "repair_d_translation_nonrepair_disabled",
            lambda: repair_translation_case(repair_busy=False, signature_faces=1, expect_success=False),
        )
        run_case(
            "repair_d_translation_zero_signature",
            lambda: repair_translation_case(repair_busy=True, signature_faces=0, expect_success=False),
        )
        run_case(
            "repair_d_translation_one_signature",
            lambda: repair_translation_case(repair_busy=True, signature_faces=1, expect_success=True),
        )
        run_case(
            "repair_d_translation_two_signatures",
            lambda: repair_translation_case(repair_busy=True, signature_faces=2, expect_success=False),
        )
        run_case("rewire_5_5_2_describe_source_decline", rebuilt_source_edge_declines)
        run_case("performance_suzanne_median", suzanne_performance_median)
    finally:
        session.TOOL_PROFILES.clear()
        session.TOOL_PROFILES.update(original_profiles)


def start_test():
    try:
        run_matrix()
        assert len(STATE["case_results"]) >= 36, len(STATE["case_results"])
        failed = [name for name, status in STATE["case_results"].items() if status != "PASS"]
        assert not failed, failed
        print(f"YSE_EXTRUDE_F9_MATRIX_CASES={len(STATE['case_results'])}", flush=True)
        print(MARK_OK, flush=True)
        bpy.ops.wm.quit_blender()
    except BaseException:
        fail()
    return None


bpy.app.timers.register(start_test, first_interval=0.25)
