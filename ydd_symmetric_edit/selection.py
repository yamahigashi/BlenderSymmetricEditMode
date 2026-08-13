from __future__ import annotations

from collections.abc import Iterable
from typing import cast

import bmesh
import numpy
from mathutils import Vector

from ._types import (
    EdgeMarkerId,
    EdgeSelectionHistory,
    FaceId,
    FaceSelectionHistory,
    HiddenFaceMap,
    SelectionHistory,
    SelectionSnapshot,
    VertexSelectionHistory,
)
from .matching import VertexRegistry, _coordinate_3d, build_vertex_pair_table
from .snapshot import (
    EDGE_HIDDEN_LAYER,
    EDGE_ORIGINAL_LAYER,
    EDGE_SELECTION_LAYER,
    FACE_HIDDEN_LAYER,
    FACE_ID_LAYER,
    FACE_SELECTION_LAYER,
    VERT_HIDDEN_LAYER,
    VERT_SELECTION_LAYER,
    capture_selection_snapshot,
)


def saved_hidden_state_present(bm: bmesh.types.BMesh) -> bool:
    """Return whether any saved hidden flag is true (false-only layers do not)."""

    for sequence, name in (
        (bm.verts, VERT_HIDDEN_LAYER),
        (bm.edges, EDGE_HIDDEN_LAYER),
        (bm.faces, FACE_HIDDEN_LAYER),
    ):
        layer = sequence.layers.int.get(name)
        if layer is not None and any(bool(element[layer]) for element in sequence):
            return True
    return False


def add_selection_layers(bm: bmesh.types.BMesh) -> SelectionSnapshot:
    """Snapshot native Knife's selection immediately before the mirror stage."""

    saved_hidden_state_present = False
    for layers, name, _hidden_name in (
        (bm.verts.layers.int, VERT_SELECTION_LAYER, VERT_HIDDEN_LAYER),
        (bm.edges.layers.int, EDGE_SELECTION_LAYER, EDGE_HIDDEN_LAYER),
        (bm.faces.layers.int, FACE_SELECTION_LAYER, FACE_HIDDEN_LAYER),
    ):
        old = layers.get(name)
        if old is not None:
            layers.remove(old)

    vertex_layer = bm.verts.layers.int.new(VERT_SELECTION_LAYER)
    edge_layer = bm.edges.layers.int.new(EDGE_SELECTION_LAYER)
    face_layer = bm.faces.layers.int.new(FACE_SELECTION_LAYER)
    marker_layer = bm.edges.layers.int.get(EDGE_ORIGINAL_LAYER)
    face_id_layer = bm.faces.layers.int.get(FACE_ID_LAYER)

    # As above, acquire all elements only after creating every layer.
    for sequence, selection_layer, hidden_name in (
        (bm.verts, vertex_layer, VERT_HIDDEN_LAYER),
        (bm.edges, edge_layer, EDGE_HIDDEN_LAYER),
        (bm.faces, face_layer, FACE_HIDDEN_LAYER),
    ):
        hidden_layer = sequence.layers.int.get(hidden_name)
        for element in sequence:
            element[selection_layer] = int(element.select)
            saved_hidden_state_present |= hidden_layer is not None and bool(element[hidden_layer])

    path_vertices_selected = False
    path_edges_selected = False
    if marker_layer is not None:
        for edge in bm.edges:
            if edge[marker_layer] <= 0:
                path_edges_selected |= bool(edge.select)
                path_vertices_selected |= any(vertex.select for vertex in edge.verts)
    path_faces_selected = any(face.select for face in bm.faces)

    history: SelectionHistory = []
    select_history = cast(
        Iterable[bmesh.types.BMVert | bmesh.types.BMEdge | bmesh.types.BMFace],
        bm.select_history,
    )
    for element in select_history:
        if isinstance(element, bmesh.types.BMVert):
            history.append(VertexSelectionHistory(location=_coordinate_3d(element.co)))
        elif isinstance(element, bmesh.types.BMEdge):
            midpoint = (element.verts[0].co + element.verts[1].co) * 0.5
            marker = EdgeMarkerId(int(element[marker_layer])) if marker_layer is not None else None
            history.append(
                EdgeSelectionHistory(
                    location=_coordinate_3d(midpoint),
                    marker=marker,
                )
            )
        elif isinstance(element, bmesh.types.BMFace):
            face_id = FaceId(int(element[face_id_layer])) if face_id_layer is not None else None
            history.append(
                FaceSelectionHistory(
                    location=_coordinate_3d(element.calc_center_median()),
                    face_id=face_id,
                )
            )

    snapshot = SelectionSnapshot(
        path_vertices_selected=path_vertices_selected,
        path_edges_selected=path_edges_selected,
        path_faces_selected=path_faces_selected,
        history=history,
        saved_hidden_state_present=saved_hidden_state_present,
    )
    return snapshot


def restore_visibility_and_selection(
    bm: bmesh.types.BMesh,
    hidden_by_face_id: HiddenFaceMap,
    selection_snapshot: SelectionSnapshot,
) -> None:
    """Undo temporary hiding and the mirror stage's selection replacement."""

    face_id_layer = bm.faces.layers.int.get(FACE_ID_LAYER)
    vertex_selection = bm.verts.layers.int.get(VERT_SELECTION_LAYER)
    edge_selection = bm.edges.layers.int.get(EDGE_SELECTION_LAYER)
    face_selection = bm.faces.layers.int.get(FACE_SELECTION_LAYER)
    vertex_hidden = bm.verts.layers.int.get(VERT_HIDDEN_LAYER)
    edge_hidden = bm.edges.layers.int.get(EDGE_HIDDEN_LAYER)
    hidden_face_count = 0

    # Visibility must be restored before selection: unhiding a face clears its
    # selection flag as a side effect.
    for face in bm.faces:
        if face_id_layer is not None:
            face_id = FaceId(int(face[face_id_layer]))
            face.hide = bool(hidden_by_face_id.get(face_id, False))
        hidden_face_count += int(face.hide)
    for edge in bm.edges:
        edge.hide = bool(edge_hidden and edge[edge_hidden])
    for vertex in bm.verts:
        vertex.hide = bool(vertex_hidden and vertex[vertex_hidden])

    # Through-face edges and vertices created inside a temporarily unhidden face
    # start with zero hide data.  Hide them when every adjacent restored face is
    # hidden, while preserving exact flags on all pre-existing elements.
    # With no hidden face, every linked edge/vertex fails the all-hidden test;
    # skip those full-mesh scans while preserving the original path otherwise.
    if hidden_face_count:
        for edge in bm.edges:
            if edge.link_faces and all(face.hide for face in edge.link_faces):
                edge.hide = True
        for vertex in bm.verts:
            if vertex.link_faces and all(face.hide for face in vertex.link_faces):
                vertex.hide = True

    # Clear broad-to-narrow, then restore narrow-to-broad.  Assigning a false
    # face flag can cascade into its boundary, while select_flush_mode() can
    # discard an explicitly restored face in face-select mode.  The snapshot
    # came from Blender's already-consistent native Knife result, so replaying
    # its exact flags needs no additional selection flush.
    for face in bm.faces:
        face.select = False
    for edge in bm.edges:
        edge.select = False
    for vertex in bm.verts:
        vertex.select = False

    for vertex in bm.verts:
        if vertex_selection is not None and vertex[vertex_selection]:
            vertex.select = True
    for edge in bm.edges:
        if edge_selection is not None and edge[edge_selection]:
            edge.select = True
    for face in bm.faces:
        if face_selection is not None and face[face_selection]:
            face.select = True

    _restore_selection_history(bm, selection_snapshot.history)


def restore_selection_scoped(
    bm: bmesh.types.BMesh,
    selection_snapshot: SelectionSnapshot,
    mutation_summary,
) -> None:
    """Restore selection for the topology elements changed by direct Knife.

    Direct topology never changes visibility.  The summary is deliberately
    supplied by the topology mutators and may contain invalidated BMesh
    proxies; those are ignored.  Selection history retains the established
    full-resolution algorithm and is restored last.
    """

    vertex_layer = bm.verts.layers.int.get(VERT_SELECTION_LAYER)
    edge_layer = bm.edges.layers.int.get(EDGE_SELECTION_LAYER)
    face_layer = bm.faces.layers.int.get(FACE_SELECTION_LAYER)

    vertices = [vertex for vertex in mutation_summary.vertices if vertex.is_valid]
    edges = [edge for edge in mutation_summary.edges if edge.is_valid]
    faces = [face for face in mutation_summary.faces if face.is_valid]

    # Match restore_visibility_and_selection's broad-to-narrow ordering.  The
    # summary includes local incident faces/edges, so selection cascades stay
    # inside the mutation scope.
    for face in faces:
        face.select = False
    for edge in edges:
        edge.select = False
    for vertex in vertices:
        vertex.select = False

    if vertex_layer is not None:
        for vertex in vertices:
            if vertex[vertex_layer]:
                vertex.select = True
    if edge_layer is not None:
        for edge in edges:
            if edge[edge_layer]:
                edge.select = True
    if face_layer is not None:
        for face in faces:
            if face[face_layer]:
                face.select = True

    _restore_selection_history(bm, selection_snapshot.history)


def restore_selection_for_route(
    bm: bmesh.types.BMesh,
    hidden_by_face_id: HiddenFaceMap,
    selection_snapshot: SelectionSnapshot,
    mutation_summary,
    *,
    direct_topology_success: bool,
    summary_complete: bool,
) -> bool:
    """Apply the production restore gate and return whether it was scoped."""

    # getattr, not direct access: instances built by the pre-field class can
    # survive an addon reload and lack the attribute entirely.
    hidden_state = getattr(selection_snapshot, "saved_hidden_state_present", None)
    if hidden_state is None:
        # Compatibility for snapshots made before C7-4's additive bit.
        hidden_state = saved_hidden_state_present(bm)
    summary_has_scope = all(hasattr(mutation_summary, name) for name in ("vertices", "edges", "faces"))
    summary_is_complete = summary_complete and summary_has_scope and bool(getattr(mutation_summary, "complete", True))
    use_scoped = (
        direct_topology_success and summary_is_complete and not any(hidden_by_face_id.values()) and not hidden_state
    )
    if use_scoped:
        restore_selection_scoped(bm, selection_snapshot, mutation_summary)
        return True
    restore_visibility_and_selection(bm, hidden_by_face_id, selection_snapshot)
    return False


def _restore_selection_history(
    bm: bmesh.types.BMesh,
    history: SelectionHistory,
) -> None:
    marker_layer = bm.edges.layers.int.get(EDGE_ORIGINAL_LAYER)
    face_id_layer = bm.faces.layers.int.get(FACE_ID_LAYER)
    bm.select_history.clear()
    used = set()

    for entry in history:
        location = Vector(entry.location.as_tuple())
        if isinstance(entry, VertexSelectionHistory):
            candidates = [vertex for vertex in bm.verts if vertex.select]
        elif isinstance(entry, EdgeSelectionHistory):
            candidates = [edge for edge in bm.edges if edge.select]
            if marker_layer is not None and entry.marker is not None:
                matching = [edge for edge in candidates if EdgeMarkerId(int(edge[marker_layer])) == entry.marker]
                if matching:
                    candidates = matching
        else:
            candidates = [face for face in bm.faces if face.select]
            if face_id_layer is not None and entry.face_id is not None:
                matching = [face for face in candidates if FaceId(int(face[face_id_layer])) == entry.face_id]
                if matching:
                    candidates = matching

        candidates = [candidate for candidate in candidates if hash(candidate) not in used]
        if not candidates:
            continue
        chosen = min(
            candidates,
            key=lambda candidate: (_selection_element_coordinate(candidate) - location).length_squared,
        )
        used.add(hash(chosen))
        bm.select_history.add(chosen)


def _selection_element_coordinate(element) -> Vector:
    if isinstance(element, bmesh.types.BMVert):
        return element.co
    if isinstance(element, bmesh.types.BMEdge):
        return (element.verts[0].co + element.verts[1].co) * 0.5
    return element.calc_center_median()


def _faces_by_verts(bm: bmesh.types.BMesh, vertex_sets: Iterable[frozenset[int]]):
    vertex_indices = {vertex_index for vertex_set in vertex_sets if vertex_set for vertex_index in vertex_set}
    candidates_by_index = {}
    for vertex_index in vertex_indices:
        if vertex_index < 0 or vertex_index >= len(bm.verts):
            continue
        vertex = bm.verts[vertex_index]
        if not vertex.is_valid:
            continue
        for face in vertex.link_faces:
            if face.is_valid:
                candidates_by_index[face.index] = face

    faces_by_verts = {}
    for face in candidates_by_index.values():
        face_vertex_set = frozenset(vertex.index for vertex in face.verts)
        current = faces_by_verts.get(face_vertex_set)
        if current is None or face.index > current.index:
            faces_by_verts[face_vertex_set] = face
    return faces_by_verts


def _find_face_by_verts(bm: bmesh.types.BMesh, vertex_indices: frozenset[int]):
    return _faces_by_verts(bm, (vertex_indices,)).get(vertex_indices)


def extend_selection_to_mirror(
    bm: bmesh.types.BMesh,
    axis_index: int,
    tolerance: float,
    *,
    mesh_object=None,
) -> int:
    """Add-select mirror counterparts of currently selected mesh elements.

    Contract: Select Mirrored. Never deselects. Never mutates
    ``select_history`` or the active element. Unresolved counterparts are
    skipped silently. On-plane / self-mirrored elements are no-ops.

    Vertex pairing resolves only the selected topology's candidate closure.
    Edges and faces resolve when every constituent vertex has a pair and some
    element owns exactly that partner vertex set.

    Coordinates and selection flags are captured via
    :func:`capture_selection_snapshot` (Mesh bulk when *mesh_object* is set).

    Returns the number of elements that transitioned from unselected to
    selected. Does not call ``select_flush_mode``; callers that need a mode
    flush may do so after this returns. Selecting an edge or face may later
    cascade to lower elements if the caller flushes — that is allowed.
    """

    # capture refreshes lookup tables and indices for every requested domain;
    # pair tables are keyed by enumerate(bm.verts) order, which then holds.
    capture = capture_selection_snapshot(
        bm,
        mesh_object=mesh_object,
        domains=("VERT", "EDGE", "FACE"),
        include_history=False,
    )
    # Snapshot first: newly selected counterparts must not seed further
    # expansion in the same call (contract is one-shot add of ρ(S), not a
    # fixed point). Indices come from the capture; element wrappers are
    # resolved only for selected members.
    selected_vert_indices = numpy.asarray(capture.selected_verts, dtype=numpy.int64)
    selected_edge_indices = numpy.asarray(capture.selected_edges, dtype=numpy.int64)
    selected_faces = [bm.faces[int(index)] for index in capture.selected_faces]

    seed_parts = [selected_vert_indices]
    if len(selected_edge_indices):
        seed_parts.append(
            numpy.asarray(
                [
                    vertex.index
                    for index in selected_edge_indices.tolist()
                    if bm.edges[index].is_valid
                    for vertex in bm.edges[index].verts
                ],
                dtype=numpy.int64,
            )
        )
    if selected_faces:
        seed_parts.append(
            numpy.asarray(
                [vertex.index for face in selected_faces if face.is_valid for vertex in face.verts],
                dtype=numpy.int64,
            )
        )
    seeds = numpy.unique(numpy.concatenate(seed_parts)) if seed_parts else numpy.empty(0, dtype=numpy.int64)
    resolved = VertexRegistry(capture.coords, axis_index, tolerance).resolve_closure_arrays(seeds)
    partner_by_vertex = numpy.full(len(capture.coords), -1, dtype=numpy.int64)
    if resolved is None:
        pairs = build_vertex_pair_table(capture.coords, axis_index, tolerance)
        if not pairs:
            return 0
        for source, target in pairs.items():
            partner_by_vertex[source] = target
    else:
        _closure_ids, pair_sources, pair_targets = resolved
        if len(pair_sources) == 0:
            return 0
        partner_by_vertex[pair_sources] = pair_targets

    added = 0
    vertex_count = len(bm.verts)

    if len(selected_vert_indices):
        vert_targets = partner_by_vertex[selected_vert_indices]
        vert_mask = (vert_targets >= 0) & (vert_targets != selected_vert_indices) & (vert_targets < vertex_count)
        for partner_index in vert_targets[vert_mask].tolist():
            partner = bm.verts[partner_index]
            if partner.is_valid and not partner.select:
                partner.select = True
                added += 1

    if len(selected_edge_indices):
        bm_verts = bm.verts
        bm_edges = bm.edges
        partner_edge_list = partner_by_vertex.tolist()
        for index in selected_edge_indices.tolist():
            edge = bm_edges[index]
            if not edge.is_valid:
                continue
            edge_verts = edge.verts
            first_index = edge_verts[0].index
            second_index = edge_verts[1].index
            first_partner = partner_edge_list[first_index]
            second_partner = partner_edge_list[second_index]
            if first_partner < 0 or second_partner < 0:
                continue
            if (first_partner == first_index and second_partner == second_index) or (
                first_partner == second_index and second_partner == first_index
            ):
                continue
            # BMesh forbids duplicate edges between one vertex pair, so the
            # first link_edges match is the unique partner (no tie to break).
            first_vert = bm_verts[first_partner]
            partner_edge = None
            for candidate in first_vert.link_edges:
                if candidate.is_valid and candidate.other_vert(first_vert).index == second_partner:
                    partner_edge = candidate
                    break
            if partner_edge is not None and not partner_edge.select:
                partner_edge.select = True
                added += 1

    partner_list = partner_by_vertex.tolist() if selected_faces else ()
    for face in selected_faces:
        if not face.is_valid:
            continue
        partner_indices = []
        source_set = set()
        for vertex in face.verts:
            index = vertex.index
            source_set.add(index)
            partner = partner_list[index]
            if partner < 0:
                partner_indices = None
                break
            partner_indices.append(partner)
        if partner_indices is None:
            continue
        partner_set = set(partner_indices)
        if partner_set == source_set:
            continue
        partner_len = len(partner_set)
        anchor_index = partner_indices[0]
        if anchor_index >= vertex_count:
            continue
        anchor = bm.verts[anchor_index]
        if not anchor.is_valid:
            continue
        # Every face sharing the partner vertex set links to the anchor, so
        # max-index over these candidates equals the global tie rule.
        best = None
        for candidate in anchor.link_faces:
            if not candidate.is_valid:
                continue
            candidate_verts = candidate.verts
            if len(candidate_verts) != partner_len:
                continue
            for vertex in candidate_verts:
                if vertex.index not in partner_set:
                    break
            else:
                if best is None or candidate.index > best.index:
                    best = candidate
        if best is not None and not best.select:
            best.select = True
            added += 1

    return added
