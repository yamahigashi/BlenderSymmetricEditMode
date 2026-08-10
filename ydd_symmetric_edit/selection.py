from __future__ import annotations

from collections.abc import Iterable
from typing import cast

import bmesh
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
from .matching import _coordinate_3d, build_vertex_pair_table
from .snapshot import (
    EDGE_HIDDEN_LAYER,
    EDGE_ORIGINAL_LAYER,
    EDGE_SELECTION_LAYER,
    FACE_ID_LAYER,
    FACE_SELECTION_LAYER,
    VERT_HIDDEN_LAYER,
    VERT_SELECTION_LAYER,
    capture_selection_snapshot,
)


def add_selection_layers(bm: bmesh.types.BMesh) -> SelectionSnapshot:
    """Snapshot native Knife's selection immediately before Knife Project."""

    for layers, name in (
        (bm.verts.layers.int, VERT_SELECTION_LAYER),
        (bm.edges.layers.int, EDGE_SELECTION_LAYER),
        (bm.faces.layers.int, FACE_SELECTION_LAYER),
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
    for vertex in bm.verts:
        vertex[vertex_layer] = int(vertex.select)
    for edge in bm.edges:
        edge[edge_layer] = int(edge.select)
    for face in bm.faces:
        face[face_layer] = int(face.select)

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

    return SelectionSnapshot(
        path_vertices_selected=path_vertices_selected,
        path_edges_selected=path_edges_selected,
        path_faces_selected=path_faces_selected,
        history=history,
    )


def restore_visibility_and_selection(
    bm: bmesh.types.BMesh,
    hidden_by_face_id: HiddenFaceMap,
    selection_snapshot: SelectionSnapshot,
) -> None:
    """Undo temporary hiding and Knife Project's selection replacement."""

    face_id_layer = bm.faces.layers.int.get(FACE_ID_LAYER)
    vertex_selection = bm.verts.layers.int.get(VERT_SELECTION_LAYER)
    edge_selection = bm.edges.layers.int.get(EDGE_SELECTION_LAYER)
    face_selection = bm.faces.layers.int.get(FACE_SELECTION_LAYER)
    vertex_hidden = bm.verts.layers.int.get(VERT_HIDDEN_LAYER)
    edge_hidden = bm.edges.layers.int.get(EDGE_HIDDEN_LAYER)

    # Visibility must be restored before selection: unhiding a face clears its
    # selection flag as a side effect.
    for face in bm.faces:
        if face_id_layer is not None:
            face_id = FaceId(int(face[face_id_layer]))
            face.hide = bool(hidden_by_face_id.get(face_id, False))
    for edge in bm.edges:
        edge.hide = bool(edge_hidden and edge[edge_hidden])
    for vertex in bm.verts:
        vertex.hide = bool(vertex_hidden and vertex[vertex_hidden])

    # Through-face edges and vertices created inside a temporarily unhidden face
    # start with zero hide data.  Hide them when every adjacent restored face is
    # hidden, while preserving exact flags on all pre-existing elements.
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

    Vertex pairing reuses :func:`build_vertex_pair_table` (KDTree candidates,
    double-precision Chebyshev verification, involutive). Edges and faces
    resolve when every constituent vertex has a pair and some element owns
    exactly that partner vertex set.

    Coordinates and selection flags are captured via
    :func:`capture_selection_snapshot` (Mesh bulk when *mesh_object* is set).

    Returns the number of elements that transitioned from unselected to
    selected. Does not call ``select_flush_mode``; callers that need a mode
    flush may do so after this returns. Selecting an edge or face may later
    cascade to lower elements if the caller flushes — that is allowed.
    """

    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    # Pair tables are keyed by enumerate(bm.verts) order; .index must match.
    bm.verts.index_update()
    bm.edges.index_update()
    bm.faces.index_update()

    capture = capture_selection_snapshot(
        bm,
        mesh_object=mesh_object,
        domains=("VERT", "EDGE", "FACE"),
        include_history=False,
    )
    # numpy float64 rows are accepted by build_vertex_pair_table (float(co[i])).
    pairs = build_vertex_pair_table(capture.coords, axis_index, tolerance)
    if not pairs:
        return 0

    # Snapshot first: newly selected counterparts must not seed further
    # expansion in the same call (contract is one-shot add of ρ(S), not a
    # fixed point). Indices come from the capture; element wrappers are
    # resolved only for selected members.
    selected_verts = [bm.verts[int(index)] for index in capture.selected_verts]
    selected_edges = [bm.edges[int(index)] for index in capture.selected_edges]
    selected_faces = [bm.faces[int(index)] for index in capture.selected_faces]

    edge_by_verts = (
        {frozenset((edge.verts[0].index, edge.verts[1].index)): edge for edge in bm.edges if edge.is_valid}
        if selected_edges
        else {}
    )
    face_by_verts = (
        {frozenset(vertex.index for vertex in face.verts): face for face in bm.faces if face.is_valid}
        if selected_faces
        else {}
    )

    added = 0

    for vertex in selected_verts:
        partner_index = pairs.get(vertex.index)
        if partner_index is None or partner_index == vertex.index:
            continue
        if partner_index < 0 or partner_index >= len(bm.verts):
            continue
        partner = bm.verts[partner_index]
        if partner.is_valid and not partner.select:
            partner.select = True
            added += 1

    for edge in selected_edges:
        if not edge.is_valid:
            continue
        source_indices = (edge.verts[0].index, edge.verts[1].index)
        partner_indices = []
        for index in source_indices:
            partner = pairs.get(index)
            if partner is None:
                partner_indices = None
                break
            partner_indices.append(partner)
        if partner_indices is None:
            continue
        partner_set = frozenset(partner_indices)
        if partner_set == frozenset(source_indices):
            continue
        partner_edge = edge_by_verts.get(partner_set)
        if partner_edge is not None and partner_edge.is_valid and not partner_edge.select:
            partner_edge.select = True
            added += 1

    for face in selected_faces:
        if not face.is_valid:
            continue
        source_indices = tuple(vertex.index for vertex in face.verts)
        partner_indices = []
        for index in source_indices:
            partner = pairs.get(index)
            if partner is None:
                partner_indices = None
                break
            partner_indices.append(partner)
        if partner_indices is None:
            continue
        partner_set = frozenset(partner_indices)
        if partner_set == frozenset(source_indices):
            continue
        partner_face = face_by_verts.get(partner_set)
        if partner_face is not None and partner_face.is_valid and not partner_face.select:
            partner_face.select = True
            added += 1

    return added
