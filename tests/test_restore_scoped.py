# SPDX-License-Identifier: GPL-3.0-or-later

"""Headless differential checks for C7-4 scoped selection restoration.

This file intentionally uses plain ``check_*`` functions: Blender's bundled
Python runner does not provide pytest.
"""

from __future__ import annotations

import sys
from pathlib import Path

import bmesh

PACKAGE_PARENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_PARENT))

from ydd_symmetric_edit import layer_names, selection, stitch_common  # noqa: E402
from ydd_symmetric_edit._types import (  # noqa: E402
    FaceId,
    FaceSelectionHistory,
    SelectionSnapshot,
    VertexSelectionHistory,
)
from ydd_symmetric_edit.matching import _coordinate_3d  # noqa: E402

MARKER = "YSE_RESTORE_SCOPED_TEST_OK"


def _make_fixture(*, selected: bool, hidden_mode: str, with_history: bool):
    """Create every CustomData layer before creating any geometry."""

    bm = bmesh.new()
    vertex_selection = bm.verts.layers.int.new(layer_names.VERT_SELECTION_LAYER)
    edge_selection = bm.edges.layers.int.new(layer_names.EDGE_SELECTION_LAYER)
    face_selection = bm.faces.layers.int.new(layer_names.FACE_SELECTION_LAYER)
    vertex_hidden = bm.verts.layers.int.new(layer_names.VERT_HIDDEN_LAYER)
    edge_hidden = bm.edges.layers.int.new(layer_names.EDGE_HIDDEN_LAYER)
    face_hidden = bm.faces.layers.int.new(layer_names.FACE_HIDDEN_LAYER)
    marker = bm.edges.layers.int.new(layer_names.EDGE_ORIGINAL_LAYER)
    face_id = bm.faces.layers.int.new(layer_names.FACE_ID_LAYER)

    vertices = [
        bm.verts.new((-1.0, -1.0, 0.0)),
        bm.verts.new((1.0, -1.0, 0.0)),
        bm.verts.new((1.0, 1.0, 0.0)),
        bm.verts.new((-1.0, 1.0, 0.0)),
    ]
    face = bm.faces.new(vertices)
    for edge in bm.edges:
        edge[marker] = 0
    face[face_id] = 1

    # Deselect the face first: BMFace.select assignment flushes downward and
    # would wipe any vertex/edge selection written before it.
    face.select = False
    face[face_hidden] = int(hidden_mode in {"true", "face_true"})
    for index, vertex in enumerate(vertices):
        vertex.select = selected and (not (selected and index in {1, 3}))
        vertex[vertex_hidden] = int(hidden_mode == "vert_true")
    for index, edge in enumerate(bm.edges):
        edge.select = selected and index % 2 == 0
        edge[edge_hidden] = int(hidden_mode == "edge_true")
    for vertex in vertices:
        vertex[vertex_selection] = int(vertex.select)
    for edge in bm.edges:
        edge[edge_selection] = int(edge.select)
    face[face_selection] = int(face.select)

    if with_history and selected:
        bm.select_history.add(vertices[0])

    hidden_by_face_id = {FaceId(1): hidden_mode in {"true", "map_true"}}
    history = []
    if with_history and selected:
        history = [
            VertexSelectionHistory(location=_coordinate_3d(vertices[0].co)),
            FaceSelectionHistory(location=_coordinate_3d(face.calc_center_median()), face_id=FaceId(1)),
        ]
    snapshot = SelectionSnapshot(
        path_vertices_selected=selected,
        path_edges_selected=selected,
        path_faces_selected=bool(face.select),
        history=history,
    )
    snapshot.saved_hidden_state_present = hidden_mode in {
        "true",
        "vert_true",
        "edge_true",
        "face_true",
    }
    return bm, snapshot, hidden_by_face_id, (vertex_selection, edge_selection, face_selection)


def _mutate_selection(bm, *, mutation_case="p_stitch"):
    """Stand in for direct topology's local selection writes."""

    for vertex in bm.verts:
        vertex.select = False
    for edge in bm.edges:
        edge.select = False
    for face in bm.faces:
        face.select = False
    # A direct topology operation may create or explicitly select one local
    # survivor; this also exercises broad-to-narrow restoration ordering.
    if mutation_case == "p_stitch":
        next(iter(bm.verts)).select = True
    else:
        next(iter(bm.edges)).select = True


def _summary_for_all(bm):
    return stitch_common.SelectionMutationSummary(
        vertices=tuple(bm.verts),
        edges=tuple(bm.edges),
        faces=tuple(bm.faces),
    )


def _state(bm):
    def coords(element):
        if isinstance(element, bmesh.types.BMVert):
            return tuple(round(float(value), 6) for value in element.co)
        if isinstance(element, bmesh.types.BMEdge):
            return tuple(sorted(coords(vertex) for vertex in element.verts))
        return tuple(sorted(coords(vertex) for vertex in element.verts))

    result = []
    for element_type, elements, selection_name, hidden_name in (
        ("v", bm.verts, layer_names.VERT_SELECTION_LAYER, layer_names.VERT_HIDDEN_LAYER),
        ("e", bm.edges, layer_names.EDGE_SELECTION_LAYER, layer_names.EDGE_HIDDEN_LAYER),
        ("f", bm.faces, layer_names.FACE_SELECTION_LAYER, layer_names.FACE_HIDDEN_LAYER),
    ):
        selection_layer = elements.layers.int.get(selection_name)
        hidden_layer = elements.layers.int.get(hidden_name)
        result.extend(
            (
                element_type,
                coords(element),
                bool(element.select),
                bool(element.hide),
                None if selection_layer is None else int(element[selection_layer]),
                None if hidden_layer is None else int(element[hidden_layer]),
            )
            for element in elements
        )
    history = []
    for element in bm.select_history:
        history.append((type(element).__name__, coords(element)))
    return tuple(result), tuple(history)


def check_scoped_matches_full_restore_matrix():
    for selected in (False, True):
        for hidden_mode in (
            "none",
            "false_only",
            "map_true",
            "true",
            "vert_true",
            "edge_true",
            "face_true",
        ):
            for with_history in (False, True):
                for direct, complete, mutation_case in (
                    (True, True, "p_stitch"),
                    (True, False, "p_stitch"),
                    (False, True, "direct_fallback"),
                ):
                    old_bm, old_snapshot, old_hidden, _ = _make_fixture(
                        selected=selected, hidden_mode=hidden_mode, with_history=with_history
                    )
                    new_bm, new_snapshot, new_hidden, _ = _make_fixture(
                        selected=selected, hidden_mode=hidden_mode, with_history=with_history
                    )
                    try:
                        if selected:
                            native_flags = [
                                element.select
                                for element in (*old_bm.verts, *old_bm.edges, *old_bm.faces)
                            ]
                            assert any(native_flags) and not all(native_flags)
                            assert not next(iter(old_bm.faces)).select
                        _mutate_selection(old_bm, mutation_case=mutation_case)
                        _mutate_selection(new_bm, mutation_case=mutation_case)
                        selection.restore_visibility_and_selection(old_bm, old_hidden, old_snapshot)
                        used_scoped = selection.restore_selection_for_route(
                            new_bm,
                            new_hidden,
                            new_snapshot,
                            _summary_for_all(new_bm),
                            direct_topology_success=direct,
                            summary_complete=complete,
                        )
                        expected_scoped = direct and complete and hidden_mode in {"none", "false_only"}
                        assert used_scoped is expected_scoped
                        assert _state(new_bm) == _state(old_bm), (
                            selected,
                            hidden_mode,
                            with_history,
                            direct,
                            complete,
                        )
                    finally:
                        old_bm.free()
                        new_bm.free()


def _route_restore(bm, snapshot, hidden_by_face_id, summary, *, direct, complete):
    return selection.restore_selection_for_route(
        bm,
        hidden_by_face_id,
        snapshot,
        summary,
        direct_topology_success=direct,
        summary_complete=complete,
    )


def check_fallback_conditions_increment_counter():
    original = selection.restore_visibility_and_selection
    calls = []

    def counted(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    selection.restore_visibility_and_selection = counted
    import ydd_symmetric_edit.selection as selection_module
    selection_original = selection_module.restore_visibility_and_selection
    selection_module.restore_visibility_and_selection = counted
    try:
        for hidden_mode, complete in (
            ("map_true", True),
            ("vert_true", True),
            ("edge_true", True),
            ("face_true", True),
            ("none", False),
        ):
            bm, snapshot, hidden, _ = _make_fixture(
                selected=True, hidden_mode=hidden_mode, with_history=False
            )
            try:
                _mutate_selection(bm)
                _route_restore(
                    bm,
                    snapshot,
                    hidden,
                    _summary_for_all(bm),
                    direct=True,
                    complete=complete,
                )
            finally:
                bm.free()
        assert len(calls) == 5
    finally:
        selection.restore_visibility_and_selection = original
        selection_module.restore_visibility_and_selection = selection_original


def check_summary_omission_is_detected():
    old_bm, old_snapshot, old_hidden, _ = _make_fixture(
        selected=True, hidden_mode="none", with_history=False
    )
    new_bm, new_snapshot, _new_hidden, _ = _make_fixture(
        selected=True, hidden_mode="none", with_history=False
    )
    try:
        old_face_layer = old_bm.faces.layers.int.get(layer_names.FACE_SELECTION_LAYER)
        new_face_layer = new_bm.faces.layers.int.get(layer_names.FACE_SELECTION_LAYER)
        next(iter(old_bm.faces))[old_face_layer] = 1
        next(iter(new_bm.faces))[new_face_layer] = 1
        old_snapshot.path_faces_selected = True
        new_snapshot.path_faces_selected = True
        _mutate_selection(old_bm)
        _mutate_selection(new_bm)
        selection.restore_visibility_and_selection(old_bm, old_hidden, old_snapshot)
        complete = stitch_common.SelectionMutationSummary(
            vertices=tuple(list(new_bm.verts)[1:]),
            edges=tuple(new_bm.edges),
            faces=(),
        )
        selection.restore_selection_scoped(new_bm, new_snapshot, complete)
        assert _state(new_bm) != _state(old_bm)
    finally:
        old_bm.free()
        new_bm.free()


def check_add_selection_layers_hidden_bit():
    """The real snapshot path detects each hidden domain and false-only layers."""

    for domain in ("verts", "edges", "faces"):
        for truth in (False, True):
            bm = bmesh.new()
            vertex_hidden = bm.verts.layers.int.new(layer_names.VERT_HIDDEN_LAYER)
            edge_hidden = bm.edges.layers.int.new(layer_names.EDGE_HIDDEN_LAYER)
            face_hidden = bm.faces.layers.int.new(layer_names.FACE_HIDDEN_LAYER)
            vertices = [
                bm.verts.new((0.0, 0.0, 0.0)),
                bm.verts.new((1.0, 0.0, 0.0)),
                bm.verts.new((0.0, 1.0, 0.0)),
            ]
            bm.faces.new(vertices)
            for vertex in bm.verts:
                vertex[vertex_hidden] = int(truth and domain == "verts")
            for edge in bm.edges:
                edge[edge_hidden] = int(truth and domain == "edges")
            for item in bm.faces:
                item[face_hidden] = int(truth and domain == "faces")
            try:
                snapshot = selection.add_selection_layers(bm)
                assert snapshot.saved_hidden_state_present is truth
            finally:
                bm.free()


def check_tracker_closure_preserves_untracked_selection():
    """Adversarial-review counterexample: a face pulled into the summary via
    add_edge must carry its full boundary, or clearing it flush-deselects an
    untracked independently-selected shared edge that is never restored."""

    from ydd_symmetric_edit import stitch_common

    bm = bmesh.new()
    vertex_selection = bm.verts.layers.int.new(layer_names.VERT_SELECTION_LAYER)
    edge_selection = bm.edges.layers.int.new(layer_names.EDGE_SELECTION_LAYER)
    face_selection = bm.faces.layers.int.new(layer_names.FACE_SELECTION_LAYER)
    try:
        v = [
            bm.verts.new((0.0, 0.0, 0.0)),
            bm.verts.new((1.0, 0.0, 0.0)),
            bm.verts.new((1.0, 1.0, 0.0)),
            bm.verts.new((0.0, 1.0, 0.0)),
            bm.verts.new((2.0, 0.0, 0.0)),
            bm.verts.new((2.0, 1.0, 0.0)),
        ]
        face_a = bm.faces.new((v[0], v[1], v[2], v[3]))
        bm.faces.new((v[1], v[4], v[5], v[2]))
        shared = next(e for e in bm.edges if set(e.verts) == {v[1], v[2]})

        # Independently select the shared edge (edge-select-mode style).
        shared.select = True
        for vert in shared.verts:
            vert.select = True
        for element, layer in (
            *((vert, vertex_selection) for vert in bm.verts),
            *((edge, edge_selection) for edge in bm.edges),
        ):
            element[layer] = int(element.select)
        for face in bm.faces:
            face[face_selection] = int(face.select)

        tracker = stitch_common._SelectionMutationTracker()
        west = next(e for e in bm.edges if set(e.verts) == {v[0], v[3]})
        tracker.add_edge(west)
        summary = tracker.finish()

        # Closure invariant: every summary face carries its full boundary.
        assert face_a in summary.faces
        assert shared in summary.edges, "face_a folded without its boundary"

        snapshot = SelectionSnapshot(
            path_vertices_selected=False,
            path_edges_selected=False,
            path_faces_selected=False,
            history=[],
        )
        selection.restore_selection_scoped(bm, snapshot, summary)
        assert shared.select, "untracked-selection wiped by face clear cascade"
        assert bool(shared[edge_selection]) == shared.select
        print("PASS check_tracker_closure_preserves_untracked_selection", flush=True)
    finally:
        bm.free()


def check_restore_scoped():
    check_scoped_matches_full_restore_matrix()
    check_fallback_conditions_increment_counter()
    check_summary_omission_is_detected()
    check_add_selection_layers_hidden_bit()
    check_tracker_closure_preserves_untracked_selection()
    print(MARKER, flush=True)


if __name__ == "__main__":
    check_restore_scoped()
