# SPDX-License-Identifier: GPL-3.0-or-later

"""Select Mirrored contract tests (headless where possible).

Contract: .agents/doc/select_mirrored_contract_2026-08-09.md §4

Covered here (headless / operator-direct):
2. Merge ON: survivor + mirror survivor selected; select_history / active native
3. Connect ON: native path verts + mirror counterparts; both-side flags symmetric
4. Asymmetric geometry: unresolved counterparts skipped silently
6. Partial failure (§3.2): disjoint Merge cluster decline must not extend

GUI-required cases (§4-1 Loop Cut real operator path, §4-5 real ed.undo) live
in ``test_select_mirrored_gui.py``.  The headless Loop Cut helper below is a
core-level supplement only and is not a substitute for that suite.

Knife / Rip escape (contract §4): those tools need multi-step modal drag event
simulation that is unstable and high-cost under the current GUI harness
(``test_persistent_mode.py``-style).  They are therefore covered by logic-unit
checks of ``extend_selection_to_mirror`` plus structural confirmation that the
finish / replay hooks call the extend helper, not by full modal GUI round-trips.

Run::

    cmd.exe /c "tests\\run_headless_test.bat 42 test_select_mirrored.py"
"""

from __future__ import annotations

import sys
import traceback
from collections import Counter
from pathlib import Path

import bmesh
import bpy
from mathutils import Vector

PACKAGE_PARENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_PARENT))

import ydd_symmetric_edit as addon  # noqa: E402
from ydd_symmetric_edit import core  # noqa: E402

MARKER_OK = "YSE_SELECT_MIRRORED_OK"
MARKER_FAILED = "YSE_SELECT_MIRRORED_FAILED"
COORD_PRECISION = 5
TOLERANCE = 1.0e-5
AXIS = core.AXIS_INDEX["X"]


def fail(message: str = "") -> None:
    if message:
        print(f"YSE_SELECT_MIRRORED_ERROR={message}", flush=True)
    traceback.print_exc()
    print(MARKER_FAILED, flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    raise SystemExit(1)


def coordinate_key(coordinate, precision: int = COORD_PRECISION):
    return tuple(round(float(value), precision) for value in coordinate)


def find_vertex(bm, expected, precision: int = COORD_PRECISION):
    key = coordinate_key(expected, precision)
    for vertex in bm.verts:
        if coordinate_key(vertex.co, precision) == key:
            return vertex
    raise AssertionError(f"vertex not found: {expected}")


def find_edge(bm, a, b, precision: int = COORD_PRECISION):
    keys = frozenset((coordinate_key(a, precision), coordinate_key(b, precision)))
    for edge in bm.edges:
        edge_keys = frozenset(coordinate_key(v.co, precision) for v in edge.verts)
        if edge_keys == keys:
            return edge
    raise AssertionError(f"edge not found: {a} -- {b}")


def clear_selection(bm) -> None:
    for face in bm.faces:
        face.select = False
    for edge in bm.edges:
        edge.select = False
    for vertex in bm.verts:
        vertex.select = False
    bm.select_history.clear()


def selected_vert_keys(bm, precision: int = COORD_PRECISION) -> set[tuple]:
    return {coordinate_key(v.co, precision) for v in bm.verts if v.select}


def selected_edge_keys(bm, precision: int = COORD_PRECISION) -> set[frozenset]:
    return {
        frozenset(coordinate_key(v.co, precision) for v in edge.verts) for edge in bm.edges if edge.select
    }


def history_vert_keys(bm, precision: int = COORD_PRECISION) -> list[tuple]:
    keys = []
    for element in bm.select_history:
        if isinstance(element, bmesh.types.BMVert):
            keys.append(coordinate_key(element.co, precision))
    return keys


def build_two_symmetric_quads_bm():
    """Same geometry as test_core.build_two_symmetric_quads."""

    bm = bmesh.new()
    left = [
        bm.verts.new(co)
        for co in (
            (-2.0, -1.0, 0.0),
            (-1.0, -1.0, 0.0),
            (-1.0, 1.0, 0.0),
            (-2.0, 1.0, 0.0),
        )
    ]
    right = [bm.verts.new(co) for co in ((1.0, -1.0, 0.0), (2.0, -1.0, 0.0), (2.0, 1.0, 0.0), (1.0, 1.0, 0.0))]
    bm.faces.new(left)
    bm.faces.new(right)
    return bm


def split_left_mid_loop(bm):
    """Simulate a vertical loop cut through the left quad (mid x=-1.5)."""

    bottom = next(
        edge
        for edge in bm.edges
        if all(vertex.co.x < 0.0 for vertex in edge.verts)
        and all(abs(vertex.co.y + 1.0) < 1.0e-8 for vertex in edge.verts)
    )
    _new_edge, bottom_vertex = bmesh.utils.edge_split(bottom, bottom.verts[0], 0.5)
    top = next(
        edge
        for edge in bm.edges
        if all(vertex.co.x < 0.0 for vertex in edge.verts)
        and all(abs(vertex.co.y - 1.0) < 1.0e-8 for vertex in edge.verts)
    )
    _new_edge, top_vertex = bmesh.utils.edge_split(top, top.verts[0], 0.5)
    source_face = next(face for face in bm.faces if bottom_vertex in face.verts and top_vertex in face.verts)
    bmesh.utils.face_split(source_face, bottom_vertex, top_vertex)
    return bm.edges.get((bottom_vertex, top_vertex))


def check_extend_selection_verts_edges_faces_and_history() -> None:
    """Core utility: add-only, history intact, self-mirrored no-op."""

    bm = bmesh.new()
    try:
        # Symmetric strip: two verts on +X, two on -X, one on plane.
        v_neg_a = bm.verts.new((-1.0, -1.0, 0.0))
        v_neg_b = bm.verts.new((-1.0, 1.0, 0.0))
        v_pos_a = bm.verts.new((1.0, -1.0, 0.0))
        v_pos_b = bm.verts.new((1.0, 1.0, 0.0))
        v_plane = bm.verts.new((0.0, 0.0, 0.0))
        e_neg = bm.edges.new((v_neg_a, v_neg_b))
        e_pos = bm.edges.new((v_pos_a, v_pos_b))
        f_left = bm.faces.new((v_neg_a, v_neg_b, v_plane))
        # Right triangle for face pairing.
        f_right = bm.faces.new((v_pos_a, v_pos_b, v_plane))
        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        bm.faces.ensure_lookup_table()

        clear_selection(bm)
        v_pos_a.select = True
        v_pos_b.select = True
        e_pos.select = True
        f_right.select = True
        bm.select_history.clear()
        bm.select_history.add(v_pos_a)
        bm.select_history.add(v_pos_b)
        history_before = history_vert_keys(bm)

        added = core.extend_selection_to_mirror(bm, AXIS, TOLERANCE)
        assert added >= 3, f"expected verts+edge(+face) adds, got {added}"
        assert v_neg_a.select and v_neg_b.select
        assert e_neg.select
        assert f_left.select
        # Original selection preserved.
        assert v_pos_a.select and v_pos_b.select and e_pos.select and f_right.select
        # History and active (last history entry) unchanged.
        assert history_vert_keys(bm) == history_before
        # On-plane vertex is self-paired: extend must not *newly* select it as a
        # counterpart of another element.  It may become selected only as a
        # lower-element cascade of a selected face/edge (contract-allowed).
        # Selecting only the on-plane vertex itself is a no-op.
        clear_selection(bm)
        v_plane.select = True
        bm.select_history.clear()
        bm.select_history.add(v_plane)
        history_plane = history_vert_keys(bm)
        added_plane = core.extend_selection_to_mirror(bm, AXIS, TOLERANCE)
        assert added_plane == 0, added_plane
        assert history_vert_keys(bm) == history_plane
        # Re-run the full selection for idempotence of the primary case.
        clear_selection(bm)
        v_pos_a.select = True
        v_pos_b.select = True
        e_pos.select = True
        f_right.select = True
        core.extend_selection_to_mirror(bm, AXIS, TOLERANCE)
        added_again = core.extend_selection_to_mirror(bm, AXIS, TOLERANCE)
        assert added_again == 0, added_again
    finally:
        bm.free()


def check_loop_cut_on_off_selection() -> None:
    """§4.1 Loop Cut: ON selects mirror path; OFF leaves native-only selection."""

    bm_off = build_two_symmetric_quads_bm()
    bm_on = build_two_symmetric_quads_bm()
    try:
        for bm, do_extend in ((bm_off, False), (bm_on, True)):
            topology = core.prepare_topology(bm, AXIS, TOLERANCE)
            path_edge = split_left_mid_loop(bm)
            assert path_edge is not None
            created, already, reason = core.apply_reflected_path_topology(
                bm,
                [path_edge],
                AXIS,
                TOLERANCE,
                topology.mirror_face_ids,
            )
            assert reason == "" and created + already == 1
            clear_selection(bm)
            path_edge.select = True
            for vertex in path_edge.verts:
                vertex.select = True
            if do_extend:
                core.extend_selection_to_mirror(bm, AXIS, TOLERANCE)

        off_edges = selected_edge_keys(bm_off)
        on_edges = selected_edge_keys(bm_on)
        assert len(off_edges) == 1, off_edges
        assert len(on_edges) == 2, on_edges
        assert off_edges.issubset(on_edges)

        # Mirror edge endpoints at x=+1.5.
        on_vert_keys = selected_vert_keys(bm_on)
        assert any(key[0] > 0.0 for key in on_vert_keys)
        assert any(key[0] < 0.0 for key in on_vert_keys)
    finally:
        bm_off.free()
        bm_on.free()


def check_asymmetric_skip() -> None:
    """§4.4 Missing counterparts are skipped; resolvable ones still apply."""

    bm = bmesh.new()
    try:
        # Pairable pair on ±1, and a lone vertex with no mirror.
        v_pos = bm.verts.new((1.0, 0.0, 0.0))
        v_neg = bm.verts.new((-1.0, 0.0, 0.0))
        v_lone = bm.verts.new((2.0, 3.0, 0.0))
        clear_selection(bm)
        v_pos.select = True
        v_lone.select = True
        bm.select_history.add(v_lone)
        history_before = history_vert_keys(bm)

        added = core.extend_selection_to_mirror(bm, AXIS, TOLERANCE)
        assert added == 1, added
        assert v_neg.select
        assert v_pos.select and v_lone.select
        assert history_vert_keys(bm) == history_before
    finally:
        bm.free()


def ensure_addon_registered() -> None:
    try:
        addon.register()
    except Exception:
        # Already registered in this Blender process.
        pass


def clear_scene_meshes() -> None:
    if bpy.context.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    for obj in tuple(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for mesh in tuple(bpy.data.meshes):
        bpy.data.meshes.remove(mesh)


def make_symmetric_cube_like() -> bpy.types.Object:
    """Factory-like cube scaled to unit; X-symmetric about origin."""

    mesh = bpy.data.meshes.new("YSE_SelectMirroredMesh")
    mesh.from_pydata(
        [
            (-1.0, -1.0, -1.0),
            (-1.0, -1.0, 1.0),
            (-1.0, 1.0, -1.0),
            (-1.0, 1.0, 1.0),
            (1.0, -1.0, -1.0),
            (1.0, -1.0, 1.0),
            (1.0, 1.0, -1.0),
            (1.0, 1.0, 1.0),
        ],
        [],
        [
            (0, 1, 3, 2),
            (4, 6, 7, 5),
            (0, 4, 5, 1),
            (2, 3, 7, 6),
            (0, 2, 6, 4),
            (1, 5, 7, 3),
        ],
    )
    mesh.update()
    obj = bpy.data.objects.new("YSE_SelectMirroredObject", mesh)
    bpy.context.scene.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    obj.use_mesh_mirror_x = True
    obj.use_mesh_mirror_y = False
    obj.use_mesh_mirror_z = False
    return obj


def enter_edit(obj) -> bmesh.types.BMesh:
    bpy.context.view_layer.objects.active = obj
    if obj.mode != "EDIT":
        bpy.ops.object.mode_set(mode="EDIT")
    return bmesh.from_edit_mesh(obj.data)


def check_connect_on() -> None:
    """§4.3 Connect ON: path verts + mirror counterparts; selection symmetric.

    Connect restore clears edge flags and keeps the native vertex path only;
    Select Mirrored then add-selects ρ of those verts.  Both sides therefore
    show vertex selection with edge flags False (contract §4-3 note).
    """

    ensure_addon_registered()
    clear_scene_meshes()
    obj = make_symmetric_cube_like()
    preferences = addon.ui.get_addon_preferences(bpy.context)
    if preferences is not None:
        preferences.enabled = True
    settings = bpy.context.scene.ydd_symmetric_edit
    settings.select_mirrored = True
    settings.tolerance = TOLERANCE

    bm = enter_edit(obj)
    bpy.context.tool_settings.mesh_select_mode = (True, False, False)
    clear_selection(bm)
    # +X face diagonal endpoints (native connect path / final selection).
    native_path = ((1.0, -1.0, -1.0), (1.0, 1.0, 1.0))
    mirror_path = ((-1.0, -1.0, -1.0), (-1.0, 1.0, 1.0))
    a = find_vertex(bm, native_path[0])
    b = find_vertex(bm, native_path[1])
    a.select = True
    b.select = True
    bm.select_history.clear()
    bm.select_history.add(a)
    bm.select_history.add(b)
    bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)

    result = bpy.ops.mesh.ydd_symmetric_edit_connect()
    assert result == {"FINISHED"}, result

    bm = bmesh.from_edit_mesh(obj.data)
    # Topology: both diagonals exist.
    native_edge = find_edge(bm, native_path[0], native_path[1])
    mirror_edge = find_edge(bm, mirror_path[0], mirror_path[1])

    # Native final selection (connection path verts) must remain selected, and
    # their mirror counterparts must be add-selected.
    for expected in (*native_path, *mirror_path):
        assert find_vertex(bm, expected).select, expected

    # Both sides' selection state must be symmetric: path verts True, the
    # connect edges themselves False (existing Connect restore clears edges).
    assert not native_edge.select
    assert not mirror_edge.select
    for edge in bm.edges:
        assert not edge.select, (
            f"edge flags must stay cleared on both sides: "
            f"{[coordinate_key(v.co) for v in edge.verts]}"
        )

    selected = selected_vert_keys(bm)
    for expected in (*native_path, *mirror_path):
        assert coordinate_key(expected) in selected, (expected, selected)
    # No stray verts selected beyond the two-sided path.
    assert selected == {coordinate_key(p) for p in (*native_path, *mirror_path)}, selected

    # select_history stays native source path (extend must not touch it).
    hist = history_vert_keys(bm)
    assert hist == [coordinate_key(native_path[0]), coordinate_key(native_path[1])], hist


def check_merge_on_history_preserved() -> None:
    """§4.2 Merge ON: both survivors selected; history stays native survivor."""

    ensure_addon_registered()
    clear_scene_meshes()
    # Custom mesh: two pairs of mergeable verts, X-symmetric.
    mesh = bpy.data.meshes.new("YSE_MergeMesh")
    mesh.from_pydata(
        [
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (-1.0, 0.0, 0.0),
            (-1.0, 1.0, 0.0),
            (0.0, -1.0, 0.0),  # plane anchor so faces can form if needed
        ],
        [(0, 1), (2, 3), (0, 4), (1, 4), (2, 4), (3, 4)],
        [],
    )
    mesh.update()
    obj = bpy.data.objects.new("YSE_MergeObject", mesh)
    bpy.context.scene.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    obj.use_mesh_mirror_x = True
    obj.use_mesh_mirror_y = False
    obj.use_mesh_mirror_z = False

    preferences = addon.ui.get_addon_preferences(bpy.context)
    if preferences is not None:
        preferences.enabled = True
    settings = bpy.context.scene.ydd_symmetric_edit
    settings.select_mirrored = True
    settings.tolerance = TOLERANCE

    bm = enter_edit(obj)
    bpy.context.tool_settings.mesh_select_mode = (True, False, False)
    clear_selection(bm)
    v0 = find_vertex(bm, (1.0, 0.0, 0.0))
    v1 = find_vertex(bm, (1.0, 1.0, 0.0))
    v0.select = True
    v1.select = True
    bm.select_history.clear()
    bm.select_history.add(v0)
    bm.select_history.add(v1)
    bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)

    # FIRST → merge +X pair onto (1,0,0); mirror merges -X onto (-1,0,0).
    result = bpy.ops.mesh.ydd_symmetric_edit_merge(mode="FIRST")
    assert result == {"FINISHED"}, result

    bm = bmesh.from_edit_mesh(obj.data)
    survivor = find_vertex(bm, (1.0, 0.0, 0.0))
    mirror_survivor = find_vertex(bm, (-1.0, 0.0, 0.0))
    assert survivor.select, "native survivor must stay selected"
    assert mirror_survivor.select, "mirror survivor must be add-selected"
    # Gone verts.
    for missing in ((1.0, 1.0, 0.0), (-1.0, 1.0, 0.0)):
        try:
            find_vertex(bm, missing)
            raise AssertionError(f"expected merged away: {missing}")
        except AssertionError as exc:
            if "expected merged away" in str(exc):
                raise
    hist = history_vert_keys(bm)
    assert hist == [coordinate_key((1.0, 0.0, 0.0))], hist


def check_undo_same_operator_step() -> None:
    """Structural proxy: extend lives in the same UNDO operator step.

    This is an *auxiliary* check only.  Contract §4-5 requires a real
    ``bpy.ops.ed.undo()`` round-trip on the GUI path; that lives in
    ``test_select_mirrored_gui.py``.  Under ``--background`` nested native
    mesh ops do not rebuild a reliable one-step undo stack, so headless
    code here only asserts the same-step structural guarantee plus a
    single-call observation that one FINISHED op yields both topology and
    selection extension.
    """

    import inspect

    from ydd_symmetric_edit import operators as yse_ops
    from ydd_symmetric_edit import replay as yse_replay

    connect_cls = yse_replay.MESH_OT_ydd_symmetric_edit_connect
    merge_cls = yse_replay.MESH_OT_ydd_symmetric_edit_merge
    assert "UNDO" in connect_cls.bl_options
    assert "UNDO" in merge_cls.bl_options

    connect_src = inspect.getsource(connect_cls.execute)
    merge_src = inspect.getsource(merge_cls.execute)
    assert "_maybe_extend_selection_to_mirror" in connect_src
    assert "_maybe_extend_selection_to_mirror" in merge_src
    # No extra undo_push between topology and selection extension.
    assert "undo_push" not in connect_src
    assert "undo_push" not in merge_src

    extend_src = inspect.getsource(core.extend_selection_to_mirror)
    assert "undo_push" not in extend_src

    finish_src = inspect.getsource(yse_ops.MESH_OT_ydd_symmetric_edit_finish.execute)
    assert "extend_selection_to_mirror" in finish_src
    assert "undo_push" not in finish_src

    # Knife / Rip hooks: finish path hosts modal tools; replay hosts Merge/Connect.
    # Structural presence is the headless stand-in for full modal GUI tests.
    assert "select_mirrored" in finish_src

    # Single-call observation: one FINISHED operator yields both diagonals
    # and both-side selection (topology + extension in one step).
    ensure_addon_registered()
    clear_scene_meshes()
    obj = make_symmetric_cube_like()
    preferences = addon.ui.get_addon_preferences(bpy.context)
    if preferences is not None:
        preferences.enabled = True
    settings = bpy.context.scene.ydd_symmetric_edit
    settings.select_mirrored = True
    settings.tolerance = TOLERANCE

    bm = enter_edit(obj)
    clear_selection(bm)
    a = find_vertex(bm, (1.0, -1.0, -1.0))
    b = find_vertex(bm, (1.0, 1.0, 1.0))
    a.select = True
    b.select = True
    bm.select_history.clear()
    bm.select_history.add(a)
    bm.select_history.add(b)
    bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)

    result = bpy.ops.mesh.ydd_symmetric_edit_connect()
    assert result == {"FINISHED"}, result

    bm = bmesh.from_edit_mesh(obj.data)
    assert len(bm.edges) == 14, len(bm.edges)
    find_edge(bm, (1.0, -1.0, -1.0), (1.0, 1.0, 1.0))
    find_edge(bm, (-1.0, -1.0, -1.0), (-1.0, 1.0, 1.0))
    assert find_vertex(bm, (1.0, -1.0, -1.0)).select
    assert find_vertex(bm, (-1.0, -1.0, -1.0)).select


def check_property_and_ui_defaults() -> None:
    """Property defaults and presence on the Scene settings group."""

    ensure_addon_registered()
    settings = bpy.context.scene.ydd_symmetric_edit
    assert hasattr(settings, "select_mirrored")
    # Default is False (contract §1); may have been set by earlier tests — reset.
    settings.select_mirrored = False
    assert settings.select_mirrored is False
    # Description/name are RNA metadata; smoke-check the identifier only.
    rna = settings.bl_rna.properties["select_mirrored"]
    assert rna.name == "Select Mirrored"
    assert "mirror counterparts" in rna.description


def check_disjoint_merge_cluster_decline_no_extend() -> None:
    """§4.6 / §3.2: cluster-level WARNING skip must not run Select Mirrored.

    Forces a partial in-cluster native merge (3 → 2 survivors) so the disjoint
    Merge path reports the partial-merge WARNING and declines the mirror.  With
    select_mirrored ON the skipped mirror vertices must stay unselected.
    Geometry / mock pattern matches tests/test_merge_modes.py D4 (c2).
    """

    from ydd_symmetric_edit import core as yse_core
    from ydd_symmetric_edit import replay as yse_replay

    partial_warning = "native merged this cluster only partially; its mirror was skipped"

    ensure_addon_registered()
    clear_scene_meshes()
    mesh = bpy.data.meshes.new("YSE_SelectMirroredDeclineMesh")
    mesh.from_pydata(
        [
            (1.0, -1.0, 0.0),  # A
            (1.0, 0.0, 0.0),  # B
            (1.0, 1.0, 0.0),  # C
            (-1.0, -1.0, 0.0),
            (-1.0, 0.0, 0.0),
            (-1.0, 1.0, 0.0),
            (0.0, -1.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
        ],
        [],
        [
            (6, 0, 1, 7),
            (7, 1, 2, 8),
            (6, 7, 4, 3),
            (7, 8, 5, 4),
        ],
    )
    mesh.update()
    obj = bpy.data.objects.new("YSE_SelectMirroredDeclineObject", mesh)
    bpy.context.scene.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    obj.use_mesh_mirror_x = True
    obj.use_mesh_mirror_y = False
    obj.use_mesh_mirror_z = False

    preferences = addon.ui.get_addon_preferences(bpy.context)
    if preferences is not None:
        preferences.enabled = True
    settings = bpy.context.scene.ydd_symmetric_edit
    settings.select_mirrored = True
    settings.tolerance = TOLERANCE

    # Reset merge report log so the WARNING assertion is local to this case.
    yse_replay._MERGE_REPORTS.clear()

    bm = enter_edit(obj)
    bpy.context.tool_settings.mesh_select_mode = (True, False, False)
    clear_selection(bm)
    for expected in ((1.0, -1.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0)):
        vertex = find_vertex(bm, expected)
        vertex.select = True
        bm.select_history.add(vertex)
    bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)

    merge_cls = yse_replay.MESH_OT_ydd_symmetric_edit_merge
    original_native = merge_cls._native

    def partial_native(self):
        """Merge only two of the three marked members so survivors stay at 2."""

        del self
        obj_local = bpy.context.edit_object
        assert obj_local is not None
        mesh_local = obj_local.data
        bm_local = bmesh.from_edit_mesh(mesh_local)
        group_layer = bm_local.verts.layers.int.get(yse_core.VERT_MERGE_GROUP_LAYER)
        assert group_layer is not None, "group markers must exist before native"
        marked = [vertex for vertex in bm_local.verts if int(vertex[group_layer]) > 0]
        assert len(marked) >= 3, [coordinate_key(v.co) for v in marked]
        bmesh.ops.pointmerge(
            bm_local,
            verts=marked[:2],
            merge_co=marked[0].co.copy(),
        )
        bmesh.update_edit_mesh(mesh_local, loop_triangles=True, destructive=True)
        return {"FINISHED"}

    merge_cls._native = partial_native  # type: ignore[method-assign]
    try:
        result = bpy.ops.mesh.ydd_symmetric_edit_merge(mode="CENTER")
    finally:
        merge_cls._native = original_native  # type: ignore[method-assign]
    assert result == {"FINISHED"}, result

    warnings = [message for kind, message in yse_replay._MERGE_REPORTS if kind == "WARNING"]
    assert partial_warning in warnings, warnings

    bm = bmesh.from_edit_mesh(obj.data)
    # Mirror triple must remain unmerged and must NOT be add-selected.
    for expected in ((-1.0, -1.0, 0.0), (-1.0, 0.0, 0.0), (-1.0, 1.0, 0.0)):
        vertex = find_vertex(bm, expected)
        assert not vertex.select, f"mirror vert must stay unselected after decline: {expected}"


def check_off_is_noop_for_extend() -> None:
    """When the flag is off the operators must not call extend; utility alone
    is not gated — callers gate.  Regression: OFF path leaves selection alone
    after a successful connect with flag false."""

    ensure_addon_registered()
    clear_scene_meshes()
    obj = make_symmetric_cube_like()
    preferences = addon.ui.get_addon_preferences(bpy.context)
    if preferences is not None:
        preferences.enabled = True
    settings = bpy.context.scene.ydd_symmetric_edit
    settings.select_mirrored = False
    settings.tolerance = TOLERANCE

    bm = enter_edit(obj)
    clear_selection(bm)
    a = find_vertex(bm, (1.0, -1.0, -1.0))
    b = find_vertex(bm, (1.0, 1.0, 1.0))
    a.select = True
    b.select = True
    bm.select_history.clear()
    bm.select_history.add(a)
    bm.select_history.add(b)
    bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)

    result = bpy.ops.mesh.ydd_symmetric_edit_connect()
    assert result == {"FINISHED"}, result

    bm = bmesh.from_edit_mesh(obj.data)
    # Mirror endpoints must NOT be selected when the flag is off.
    assert not find_vertex(bm, (-1.0, -1.0, -1.0)).select
    assert not find_vertex(bm, (-1.0, 1.0, 1.0)).select
    assert find_vertex(bm, (1.0, -1.0, -1.0)).select
    assert find_vertex(bm, (1.0, 1.0, 1.0)).select


def run() -> None:
    print("YSE_SELECT_MIRRORED_BEGIN", flush=True)
    check_property_and_ui_defaults()
    print("YSE_SELECT_MIRRORED_STEP=property", flush=True)
    check_extend_selection_verts_edges_faces_and_history()
    print("YSE_SELECT_MIRRORED_STEP=extend_unit", flush=True)
    check_loop_cut_on_off_selection()
    print("YSE_SELECT_MIRRORED_STEP=loop_cut", flush=True)
    check_asymmetric_skip()
    print("YSE_SELECT_MIRRORED_STEP=asymmetric", flush=True)
    check_connect_on()
    print("YSE_SELECT_MIRRORED_STEP=connect_on", flush=True)
    check_off_is_noop_for_extend()
    print("YSE_SELECT_MIRRORED_STEP=connect_off", flush=True)
    check_merge_on_history_preserved()
    print("YSE_SELECT_MIRRORED_STEP=merge", flush=True)
    check_disjoint_merge_cluster_decline_no_extend()
    print("YSE_SELECT_MIRRORED_STEP=merge_decline_no_extend", flush=True)
    check_undo_same_operator_step()
    print("YSE_SELECT_MIRRORED_STEP=undo", flush=True)
    print(MARKER_OK, flush=True)


if __name__ == "__main__":
    try:
        run()
    except SystemExit:
        raise
    except BaseException:
        fail()
