# SPDX-License-Identifier: GPL-3.0-or-later

"""Symmetric postprocess for Blender's native region extrude macros.

Stage 1+2 cover ``EXTRUDE_NORMAL``, ``EXTRUDE_CONTEXT``, and
``EXTRUDE_SHRINK_FATTEN``. Classification uses vertex-ID instance groups and a
freeze table; FACE_ID set-difference and live selection are not discriminators.
"""

from __future__ import annotations

import math
import traceback
from dataclasses import dataclass

import bmesh
import bpy
from mathutils import Vector

from . import element_pairs, layer_names, matching
from ._types import (
    Coordinate3D,
    ExtrudeFreezeEntry,
    ExtrudeNativeOptions,
    ExtrudeSnapshot,
    MeshSelectionMode,
    RipDupSignature,
    RipSignature,
)

F12_CENSUS: dict[str, tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]] = {
    "single_face": ((4, 8, 5), (0, 0, 1), (4, 8, 4)),
    "region_2x2": ((9, 20, 12), (1, 4, 4), (8, 16, 8)),
    "edge": ((2, 3, 1), (0, 0, 0), (2, 3, 1)),
    "edge_path": ((3, 5, 2), (0, 0, 0), (3, 5, 2)),
    "vertex": ((1, 1, 0), (0, 0, 0), (1, 1, 0)),
}


@dataclass
class ExtrudeClassification:
    """Live origin/copy assignment after classify or freeze-table reconnect."""

    origins: dict[int, bmesh.types.BMVert]
    copies: dict[int, bmesh.types.BMVert]
    vanished_preop: dict[int, Coordinate3D]
    freeze: tuple[ExtrudeFreezeEntry, ...]


@dataclass
class ExtrudeSourceDescription:
    """Observed source extrusion after classification."""

    new_verts: list[bmesh.types.BMVert]
    new_edges: list[bmesh.types.BMEdge]
    new_faces: list[bmesh.types.BMFace]
    face_signatures: tuple[tuple[tuple[int, str], ...], ...]
    deleted_face_ids: tuple[int, ...]
    deleted_edge_markers: tuple[int, ...]
    deleted_vertex_ids: tuple[int, ...]
    created: tuple[int, int, int]
    deleted: tuple[int, int, int]
    net: tuple[int, int, int]
    f12_shape: str | None


def stamp_all_vertex_ids(bm: bmesh.types.BMesh) -> None:
    """Stamp every vertex so inherited IDs stay unique pairing primitives."""

    layer = bm.verts.layers.int.get(layer_names.VERT_SESSION_ID_LAYER)
    if layer is None:
        raise RuntimeError("session vertex ID layer is missing")
    bm.verts.ensure_lookup_table()
    bm.verts.index_update()
    for vertex in bm.verts:
        vertex[layer] = vertex.index + 1


def build_snapshot(
    bm: bmesh.types.BMesh,
    axis_index: int,
    tolerance: float,
    *,
    tool_kind: str,
    route_kmi_properties: tuple[tuple[str, object], ...],
    mesh_select_mode: MeshSelectionMode,
    mesh_object=None,
) -> ExtrudeSnapshot | None:
    """Capture the immutable pre-op ExtrudeSnapshot used by gates and finish."""

    vertex_layer = bm.verts.layers.int.get(layer_names.VERT_SESSION_ID_LAYER)
    edge_layer = bm.edges.layers.int.get(layer_names.EDGE_ORIGINAL_LAYER)
    face_layer = bm.faces.layers.int.get(layer_names.FACE_ID_LAYER)
    if vertex_layer is None or edge_layer is None or face_layer is None:
        return None

    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    bm.verts.index_update()
    bm.edges.index_update()
    bm.faces.index_update()

    pair_maps = element_pairs.build_element_pair_maps(
        bm,
        axis_index,
        tolerance,
        mesh_object=mesh_object,
    )

    index_to_vid = {vertex.index: int(vertex[vertex_layer]) for vertex in bm.verts}
    index_to_eid = {edge.index: int(edge[edge_layer]) for edge in bm.edges}
    index_to_fid = {face.index: int(face[face_layer]) for face in bm.faces}

    selected_vertex_ids = frozenset(int(vertex[vertex_layer]) for vertex in bm.verts if vertex.select)
    selected_edge_markers = frozenset(int(edge[edge_layer]) for edge in bm.edges if edge.select)
    selected_face_ids = frozenset(int(face[face_layer]) for face in bm.faces if face.select)

    vertex_preop = tuple(
        (
            int(vertex[vertex_layer]),
            Coordinate3D(x=float(vertex.co.x), y=float(vertex.co.y), z=float(vertex.co.z)),
        )
        for vertex in bm.verts
    )

    vertex_pairs = []
    for vertex in bm.verts:
        partner_index = pair_maps.vert_pairs.get(vertex.index)
        partner_id = None if partner_index is None else index_to_vid.get(int(partner_index))
        vertex_pairs.append((int(vertex[vertex_layer]), partner_id))

    edge_pairs = []
    for edge in bm.edges:
        partner_index = pair_maps.edge_pair_by_index[edge.index]
        partner_id = None if partner_index is None else index_to_eid.get(int(partner_index))
        edge_pairs.append((int(edge[edge_layer]), partner_id))

    face_pairs = []
    for face in bm.faces:
        partner_index = pair_maps.face_pair_by_index[face.index]
        partner_id = None if partner_index is None else index_to_fid.get(int(partner_index))
        face_pairs.append((int(face[face_layer]), partner_id))

    hidden_vertex_ids = frozenset(int(vertex[vertex_layer]) for vertex in bm.verts if vertex.hide)
    hidden_edge_markers = frozenset(int(edge[edge_layer]) for edge in bm.edges if edge.hide)
    hidden_face_ids = frozenset(int(face[face_layer]) for face in bm.faces if face.hide)

    face_corners = tuple(
        (int(face[face_layer]), tuple(int(loop.vert[vertex_layer]) for loop in face.loops)) for face in bm.faces
    )
    edge_endpoints = tuple(
        (
            int(edge[edge_layer]),
            (int(edge.verts[0][vertex_layer]), int(edge.verts[1][vertex_layer])),
        )
        for edge in bm.edges
    )

    return ExtrudeSnapshot(
        axis_index=axis_index,
        tolerance=tolerance,
        tool_kind=tool_kind,
        route_kmi_properties=route_kmi_properties,
        mesh_select_mode=mesh_select_mode,
        selected_vertex_ids=selected_vertex_ids,
        selected_edge_markers=selected_edge_markers,
        selected_face_ids=selected_face_ids,
        vertex_preop=vertex_preop,
        vertex_pairs=tuple(vertex_pairs),
        edge_pairs=tuple(edge_pairs),
        face_pairs=tuple(face_pairs),
        hidden_vertex_ids=hidden_vertex_ids,
        hidden_edge_markers=hidden_edge_markers,
        hidden_face_ids=hidden_face_ids,
        face_corners=face_corners,
        edge_endpoints=edge_endpoints,
        vertex_count=len(bm.verts),
        edge_count=len(bm.edges),
        face_count=len(bm.faces),
    )


def evaluate_prepare_gates(snapshot: ExtrudeSnapshot) -> tuple[str, str]:
    """Return (APPLY|DECLINE, reason) from the snapshot only (G4–G6)."""

    preop = snapshot.vertex_preop_map()
    vertex_pairs = snapshot.vertex_pair_map()
    edge_pairs = snapshot.edge_pair_map()
    face_pairs = snapshot.face_pair_map()
    axis = snapshot.axis_index
    tol = snapshot.tolerance

    for vertex_id in snapshot.selected_vertex_ids:
        coord = preop.get(vertex_id)
        if coord is not None and abs(coord.component(axis)) <= tol:
            return "DECLINE", "the extrusion selection includes a vertex on the mirror plane"

    selected_vertices = snapshot.selected_vertex_ids
    off_plane = set()
    for vertex_id in selected_vertices:
        coord = preop.get(vertex_id)
        if coord is None or abs(coord.component(axis)) <= tol:
            continue
        off_plane.add(vertex_id)
    mirrored_off_plane = set()
    for vertex_id in off_plane:
        partner = vertex_pairs.get(vertex_id)
        if partner is not None:
            mirrored_off_plane.add(partner)
    if off_plane & mirrored_off_plane:
        return "DECLINE", "the extrusion selection intersects its mirror image"

    if _domain_lacks_injective_counterpart(snapshot.selected_vertex_ids, vertex_pairs):
        return "DECLINE", "a selected vertex has no unique mirrored counterpart"
    if _domain_lacks_injective_counterpart(snapshot.selected_edge_markers, edge_pairs):
        return "DECLINE", "a selected edge has no unique mirrored counterpart"
    if _domain_lacks_injective_counterpart(snapshot.selected_face_ids, face_pairs):
        return "DECLINE", "a selected face has no unique mirrored counterpart"

    for vertex_id in snapshot.selected_vertex_ids:
        partner = vertex_pairs.get(vertex_id)
        if partner is not None and partner in snapshot.hidden_vertex_ids:
            return "DECLINE", "a mirrored counterpart vertex is hidden"
    for marker in snapshot.selected_edge_markers:
        partner = edge_pairs.get(marker)
        if partner is not None and partner in snapshot.hidden_edge_markers:
            return "DECLINE", "a mirrored counterpart edge is hidden"
    for face_id in snapshot.selected_face_ids:
        partner = face_pairs.get(face_id)
        if partner is not None and partner in snapshot.hidden_face_ids:
            return "DECLINE", "a mirrored counterpart face is hidden"

    return "APPLY", ""


def _domain_lacks_injective_counterpart(
    selected_ids: frozenset[int],
    pair_map: dict[int, int | None],
) -> bool:
    seen_partners: set[int] = set()
    for element_id in selected_ids:
        partner = pair_map.get(element_id)
        if partner is None:
            return True
        if partner in seen_partners:
            return True
        seen_partners.add(partner)
    return False


def has_extrude_result(bm: bmesh.types.BMesh) -> bool:
    """True as soon as one duplicated vertex ID exists (including zero-offset)."""

    vertex_id_layer = bm.verts.layers.int.get(layer_names.VERT_SESSION_ID_LAYER)
    if vertex_id_layer is None:
        return False
    seen: set[int] = set()
    for vertex in bm.verts:
        vertex_id = int(vertex[vertex_id_layer])
        if vertex_id <= 0:
            continue
        if vertex_id in seen:
            return True
        seen.add(vertex_id)
    return False


def extrude_result_signature(bm: bmesh.types.BMesh) -> RipSignature | None:
    """Stable signature of duplicated vertex-ID groups (rip-style)."""

    vertex_id_layer = bm.verts.layers.int.get(layer_names.VERT_SESSION_ID_LAYER)
    if vertex_id_layer is None:
        return None
    groups: dict[int, list[Coordinate3D]] = {}
    for vertex in bm.verts:
        vertex_id = int(vertex[vertex_id_layer])
        if vertex_id <= 0:
            continue
        groups.setdefault(vertex_id, []).append(
            Coordinate3D(
                x=round(float(vertex.co[0]), 9),
                y=round(float(vertex.co[1]), 9),
                z=round(float(vertex.co[2]), 9),
            )
        )
    signature = tuple(
        RipDupSignature(vertex_id=vertex_id, coordinates=tuple(sorted(coordinates)))
        for vertex_id, coordinates in sorted(groups.items())
        if len(coordinates) > 1
    )
    return signature or None


# Both macro children must be present and readable, or the capture fails
# closed: a default-filled options object must never masquerade as captured.
_EXPECTED_MACRO_CHILDREN = {
    "EXTRUDE_NORMAL": ("MESH_OT_extrude_region", "TRANSFORM_OT_translate"),
    "EXTRUDE_CONTEXT": ("MESH_OT_extrude_context", "TRANSFORM_OT_translate"),
    "EXTRUDE_SHRINK_FATTEN": ("MESH_OT_extrude_region", "TRANSFORM_OT_shrink_fatten"),
}


def capture_native_options(operator, tool_kind: str) -> ExtrudeNativeOptions | None:
    """Read topology-child + transform-child props from a confirmed extrude macro."""

    expected = _EXPECTED_MACRO_CHILDREN.get(tool_kind)
    if operator is None or expected is None:
        return None
    topology = getattr(operator, expected[0], None)
    transform = getattr(operator, expected[1], None)
    if topology is None or transform is None:
        return None
    try:
        use_normal_flip = bool(topology.use_normal_flip)
        use_dissolve_ortho_edges = bool(topology.use_dissolve_ortho_edges)
        mirror = bool(topology.mirror)
    except Exception:
        traceback.print_exc()
        return None
    transform_props = _capture_operator_properties(transform)
    if transform_props is None:
        return None
    return ExtrudeNativeOptions(
        use_normal_flip=use_normal_flip,
        use_dissolve_ortho_edges=use_dissolve_ortho_edges,
        mirror=mirror,
        transform_props=transform_props,
    )


def _observed_copy_offset(session) -> float | None:
    """Largest pre-op → live displacement of a session-tagged vertex."""

    snapshot = getattr(session, "extrude", None)
    if snapshot is None:
        return None
    obj = bpy.data.objects.get(session.object_name)
    if obj is None or obj.type != "MESH" or obj.mode != "EDIT":
        return None
    try:
        bm = bmesh.from_edit_mesh(obj.data)
        vertex_layer = bm.verts.layers.int.get(layer_names.VERT_SESSION_ID_LAYER)
    except Exception:
        traceback.print_exc()
        return None
    if vertex_layer is None:
        return None
    preop = snapshot.vertex_preop_map()
    best = 0.0
    for vertex in bm.verts:
        vertex_id = int(vertex[vertex_layer])
        if vertex_id <= 0:
            continue
        pre = preop.get(vertex_id)
        if pre is None:
            continue
        dx = float(vertex.co.x) - pre.x
        dy = float(vertex.co.y) - pre.y
        dz = float(vertex.co.z) - pre.z
        distance = math.sqrt(dx * dx + dy * dy + dz * dz)
        if distance > best:
            best = distance
    if best <= float(snapshot.tolerance):
        return None
    return best


def ensure_observed_shrink_value(options: ExtrudeNativeOptions, session) -> ExtrudeNativeOptions:
    """Fill shrink_fatten ``value`` from observed copy motion when RNA stays 0.

    Interactive ``TRANSFORM_OT_shrink_fatten`` confirms without writing ``value``
    back onto the redo child (4.2/5.2 measured 0.0 / is_property_set False).
    """

    current = next((value for name, value in options.transform_props if name == "value"), None)
    if isinstance(current, (int, float)) and math.isfinite(float(current)) and float(current) != 0.0:
        return options
    observed = _observed_copy_offset(session)
    if observed is None:
        return options
    props = tuple((name, value) for name, value in options.transform_props if name != "value")
    return ExtrudeNativeOptions(
        use_normal_flip=options.use_normal_flip,
        use_dissolve_ortho_edges=options.use_dissolve_ortho_edges,
        mirror=options.mirror,
        transform_props=(*props, ("value", observed)),
    )


def _capture_operator_properties(operator) -> tuple[tuple[str, object], ...] | None:
    # None distinguishes a read failure from a legitimately empty capture.
    if operator is None:
        return None
    captured: list[tuple[str, object]] = []
    try:
        for prop in operator.bl_rna.properties:
            identifier = prop.identifier
            if identifier == "rna_type":
                continue
            # Transform children (shrink_fatten `value` especially) often keep
            # is_property_set False after a modal confirm. Skip only missing
            # attrs; do not require the "explicitly set" flag.
            if identifier != "value" and not operator.is_property_set(identifier):
                continue
            value = getattr(operator, identifier)
            if isinstance(value, (bool, int, float, str)):
                captured.append((identifier, value))
            else:
                captured.append((identifier, str(value)))
        if not any(name == "value" for name, _unused in captured):
            raw = operator.value
            if isinstance(raw, (bool, int, float, str)):
                captured.append(("value", raw))
            else:
                captured.append(("value", float(raw)))
    except Exception:
        traceback.print_exc()
        return None
    return tuple(captured)


def _verts_by_session_id(bm: bmesh.types.BMesh, vertex_layer) -> dict[int, list[bmesh.types.BMVert]]:
    groups: dict[int, list[bmesh.types.BMVert]] = {}
    for vertex in bm.verts:
        vertex_id = int(vertex[vertex_layer])
        if vertex_id > 0:
            groups.setdefault(vertex_id, []).append(vertex)
    return groups


def _as_coordinate(vertex: bmesh.types.BMVert) -> Coordinate3D:
    return Coordinate3D(x=float(vertex.co.x), y=float(vertex.co.y), z=float(vertex.co.z))


def classify_live(
    bm: bmesh.types.BMesh,
    snapshot: ExtrudeSnapshot,
) -> tuple[ExtrudeClassification | None, str | None]:
    """Classify vertex-ID groups from the live mesh. Selection is not used."""

    vertex_layer = bm.verts.layers.int.get(layer_names.VERT_SESSION_ID_LAYER)
    if vertex_layer is None:
        return None, "the session vertex ID layer is missing"
    groups = _verts_by_session_id(bm, vertex_layer)
    preop = snapshot.vertex_preop_map()
    selected = snapshot.selected_vertex_ids
    tol = snapshot.tolerance
    origins: dict[int, bmesh.types.BMVert] = {}
    copies: dict[int, bmesh.types.BMVert] = {}
    vanished: dict[int, Coordinate3D] = {}

    assigned: set[int] = set()
    for vertex_id, instances in groups.items():
        pre = preop.get(vertex_id)
        if pre is None:
            return None, "a live vertex ID is missing from the extrude snapshot"
        count = len(instances)
        if count == 1:
            if vertex_id not in selected:
                assigned.add(vertex_id)
                continue
            copies[vertex_id] = instances[0]
            vanished[vertex_id] = pre
            assigned.add(vertex_id)
            continue
        if count == 2:
            moved = [vertex for vertex in instances if not matching.coordinates_match(vertex.co, pre.as_tuple(), tol)]
            stayed = [vertex for vertex in instances if matching.coordinates_match(vertex.co, pre.as_tuple(), tol)]
            if len(moved) == 0:
                return None, "zero-offset extrude is not mirrored"
            if len(moved) == 1 and len(stayed) == 1:
                origins[vertex_id] = stayed[0]
                copies[vertex_id] = moved[0]
                assigned.add(vertex_id)
                continue
            return None, "a duplicated vertex could not be classified by movement"
        return None, "an unsupported N-duplicate vertex group was observed"

    missing_targets = selected - assigned
    if missing_targets:
        return None, "a selected extrusion vertex disappeared without a remaining copy"

    freeze = tuple(
        ExtrudeFreezeEntry(
            vertex_id=vertex_id,
            entity_class="c" if vertex_id in vanished else "b",
            origin_preop=preop[vertex_id],
            copy_post=_as_coordinate(copy),
        )
        for vertex_id, copy in sorted(copies.items())
    )
    if not freeze:
        return None, "the native extrude produced no classified copies"
    return ExtrudeClassification(origins, copies, vanished, freeze), None


def reconnect_freeze(
    bm: bmesh.types.BMesh,
    snapshot: ExtrudeSnapshot,
    freeze: tuple[ExtrudeFreezeEntry, ...],
) -> tuple[ExtrudeClassification | None, str | None]:
    """Reconnect live verts from the frozen origin/copy table. No reclassify."""

    vertex_layer = bm.verts.layers.int.get(layer_names.VERT_SESSION_ID_LAYER)
    if vertex_layer is None:
        return None, "the session vertex ID layer is missing"
    groups = _verts_by_session_id(bm, vertex_layer)
    tol = snapshot.tolerance
    origins: dict[int, bmesh.types.BMVert] = {}
    copies: dict[int, bmesh.types.BMVert] = {}
    vanished: dict[int, Coordinate3D] = {}

    for entry in freeze:
        instances = groups.get(entry.vertex_id, [])
        origin_hits = [
            vertex for vertex in instances if matching.coordinates_match(vertex.co, entry.origin_preop.as_tuple(), tol)
        ]
        copy_hits = [
            vertex for vertex in instances if matching.coordinates_match(vertex.co, entry.copy_post.as_tuple(), tol)
        ]
        matched = set(origin_hits)
        matched.update(copy_hits)
        extra = [vertex for vertex in instances if vertex not in matched]
        if entry.entity_class == "b":
            if len(origin_hits) != 1 or len(copy_hits) != 1 or extra:
                return None, "a frozen boundary pair did not match origin+copy"
            if origin_hits[0] is copy_hits[0]:
                return None, "the frozen origin and copy resolved to the same vertex"
            origins[entry.vertex_id] = origin_hits[0]
            copies[entry.vertex_id] = copy_hits[0]
            continue
        if entry.entity_class == "c":
            if len(copy_hits) != 1 or len(origin_hits) != 0 or extra:
                return None, "a frozen interior copy did not match a single copy"
            copies[entry.vertex_id] = copy_hits[0]
            vanished[entry.vertex_id] = entry.origin_preop
            continue
        return None, "a frozen extrude row has an unsupported class"

    return ExtrudeClassification(origins, copies, vanished, freeze), None


def _origin_entity(
    vertex_id: int,
    classified: ExtrudeClassification,
    groups: dict[int, list[bmesh.types.BMVert]],
) -> bmesh.types.BMVert | None:
    if vertex_id in classified.origins:
        return classified.origins[vertex_id]
    if vertex_id in classified.vanished_preop:
        return None
    group = groups.get(vertex_id, [])
    return group[0] if len(group) == 1 else None


def _live_faces_with_verts(origin_verts: list[bmesh.types.BMVert]) -> list[bmesh.types.BMFace]:
    if not origin_verts or any(vertex is None or not vertex.is_valid for vertex in origin_verts):
        return []
    wanted = set(origin_verts)
    matches: list[bmesh.types.BMFace] = []
    seen: set[int] = set()
    for face in origin_verts[0].link_faces:
        if not face.is_valid or face.index in seen:
            continue
        seen.add(face.index)
        if set(face.verts) == wanted and len(face.verts) == len(origin_verts):
            matches.append(face)
    return matches


def describe_source(
    bm: bmesh.types.BMesh,
    snapshot: ExtrudeSnapshot,
    classified: ExtrudeClassification,
) -> tuple[ExtrudeSourceDescription | None, str | None]:
    """Build the source description and check the derived census."""

    vertex_layer = bm.verts.layers.int.get(layer_names.VERT_SESSION_ID_LAYER)
    edge_layer = bm.edges.layers.int.get(layer_names.EDGE_ORIGINAL_LAYER)
    if vertex_layer is None or edge_layer is None:
        return None, "temporary topology markers are missing"

    groups = _verts_by_session_id(bm, vertex_layer)
    copy_set = set(classified.copies.values())
    origin_set = set(classified.origins.values())
    new_verts = list(classified.copies.values())

    new_edges: list[bmesh.types.BMEdge] = []
    seen_edges: set[int] = set()
    for copy in copy_set:
        if not copy.is_valid:
            continue
        for edge in copy.link_edges:
            if not edge.is_valid or edge.index in seen_edges:
                continue
            seen_edges.add(edge.index)
            first, second = edge.verts[:]
            if first in copy_set and second in copy_set:
                new_edges.append(edge)
                continue
            if (first in copy_set and second in origin_set) or (second in copy_set and first in origin_set):
                if int(first[vertex_layer]) == int(second[vertex_layer]):
                    new_edges.append(edge)

    deleted_face_ids: list[int] = []
    for face_id, corners in snapshot.face_corner_map().items():
        origin_verts = [_origin_entity(vertex_id, classified, groups) for vertex_id in corners]
        if any(vertex is None for vertex in origin_verts):
            deleted_face_ids.append(face_id)
            continue
        live_origins = [vertex for vertex in origin_verts if vertex is not None]
        if len(_live_faces_with_verts(live_origins)) != 1:
            deleted_face_ids.append(face_id)

    deleted_edge_markers: list[int] = []
    for marker, (first_id, second_id) in snapshot.edge_endpoint_map().items():
        first = _origin_entity(first_id, classified, groups)
        second = _origin_entity(second_id, classified, groups)
        if first is None or second is None:
            deleted_edge_markers.append(marker)
            continue
        if not any(edge.is_valid and edge.other_vert(first) is second for edge in first.link_edges):
            deleted_edge_markers.append(marker)

    deleted_vertex_ids = tuple(sorted(classified.vanished_preop))
    new_faces: list[bmesh.types.BMFace] = []
    seen_faces: set[int] = set()
    for copy in copy_set:
        if not copy.is_valid:
            continue
        for face in copy.link_faces:
            if not face.is_valid or face.index in seen_faces:
                continue
            seen_faces.add(face.index)
            if any(vertex in copy_set for vertex in face.verts):
                new_faces.append(face)

    face_signatures: list[tuple[tuple[int, str], ...]] = []
    for face in new_faces:
        signature, signature_reason = _face_corner_signature(face, vertex_layer, snapshot, classified)
        if signature is None:
            return None, signature_reason or "a new face corner is not a classified origin or copy"
        face_signatures.append(signature)

    created = (len(new_verts), len(new_edges), len(new_faces))
    deleted = (len(deleted_vertex_ids), len(deleted_edge_markers), len(deleted_face_ids))
    net = (created[0] - deleted[0], created[1] - deleted[1], created[2] - deleted[2])
    expected = _derive_expected_census(snapshot)
    if expected is None:
        return None, "the extrusion region contains a wire or non-manifold edge"
    expected_created, expected_deleted, expected_net = expected
    if created != expected_created or deleted != expected_deleted or net != expected_net:
        return None, (
            f"source census {created}/{deleted}/{net} does not match derived "
            f"{expected_created}/{expected_deleted}/{expected_net}"
        )
    shape = _detect_f12_shape(snapshot)
    if shape is not None:
        oracle_created, oracle_deleted, oracle_net = F12_CENSUS[shape]
        if expected != (oracle_created, oracle_deleted, oracle_net):
            return None, (
                f"derived census {expected} does not match oracle {shape} "
                f"{oracle_created}/{oracle_deleted}/{oracle_net}"
            )

    return (
        ExtrudeSourceDescription(
            new_verts=new_verts,
            new_edges=new_edges,
            new_faces=new_faces,
            face_signatures=tuple(face_signatures),
            deleted_face_ids=tuple(deleted_face_ids),
            deleted_edge_markers=tuple(deleted_edge_markers),
            deleted_vertex_ids=deleted_vertex_ids,
            created=created,
            deleted=deleted,
            net=net,
            f12_shape=shape,
        ),
        None,
    )


def _detect_f12_shape(snapshot: ExtrudeSnapshot) -> str | None:
    """Pin an F12 row only when the derived region census already matches it.

    Count-only labels (one selected face, two selected edges) over-pin
    triangles and disjoint edges. The F12 table is an oracle for the §5.3
    formula, not a decline filter on legal non-table shapes.
    """

    expected = _derive_expected_census(snapshot)
    if expected is None:
        return None
    for shape, oracle in F12_CENSUS.items():
        if expected != oracle:
            continue
        if shape == "region_2x2" and not _is_2x2_region(snapshot):
            continue
        return shape
    return None


def _is_2x2_region(snapshot: ExtrudeSnapshot) -> bool:
    selected = [corners for face_id, corners in snapshot.face_corners if face_id in snapshot.selected_face_ids]
    if len(selected) != 4:
        return False
    counts: dict[int, int] = {}
    for corners in selected:
        for vertex_id in corners:
            counts[vertex_id] = counts.get(vertex_id, 0) + 1
    return len(counts) == 9 and sum(1 for count in counts.values() if count == 4) == 1


def _derive_expected_census(
    snapshot: ExtrudeSnapshot,
) -> tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]] | None:
    selected_vertices = snapshot.selected_vertex_ids
    edge_endpoints = snapshot.edge_endpoint_map()
    face_corners = snapshot.face_corner_map()
    region_edges = {
        marker
        for marker, (first_id, second_id) in edge_endpoints.items()
        if first_id in selected_vertices and second_id in selected_vertices
    }
    region_faces = {
        face_id
        for face_id, corners in face_corners.items()
        if corners and all(vertex_id in selected_vertices for vertex_id in corners)
    }

    pair_to_markers: dict[frozenset[int], list[int]] = {}
    for marker, (first_id, second_id) in edge_endpoints.items():
        pair_to_markers.setdefault(frozenset((first_id, second_id)), []).append(marker)

    edge_faces: dict[int, list[int]] = {marker: [] for marker in edge_endpoints}
    for face_id, corners in face_corners.items():
        count = len(corners)
        for index in range(count):
            pair = frozenset((corners[index], corners[(index + 1) % count]))
            for marker in pair_to_markers.get(pair, ()):
                edge_faces[marker].append(face_id)

    for marker in region_edges:
        incident = len(edge_faces.get(marker, ()))
        if incident == 0 or incident >= 3:
            return None

    deleted_edges = {
        marker
        for marker in region_edges
        if len(edge_faces.get(marker, ())) == 2 and all(face_id in region_faces for face_id in edge_faces[marker])
    }

    vertex_faces: dict[int, list[int]] = {}
    for face_id, corners in face_corners.items():
        for vertex_id in corners:
            vertex_faces.setdefault(vertex_id, []).append(face_id)
    vertex_edges: dict[int, list[int]] = {}
    for marker, (first_id, second_id) in edge_endpoints.items():
        vertex_edges.setdefault(first_id, []).append(marker)
        vertex_edges.setdefault(second_id, []).append(marker)

    deleted_vertices = {
        vertex_id
        for vertex_id in selected_vertices
        if all(face_id in region_faces for face_id in vertex_faces.get(vertex_id, ()))
        and all(marker in deleted_edges for marker in vertex_edges.get(vertex_id, ()))
    }

    created = (
        len(selected_vertices),
        (len(selected_vertices) - len(deleted_vertices)) + len(region_edges),
        (len(region_edges) - len(deleted_edges)) + len(region_faces),
    )
    deleted = (len(deleted_vertices), len(deleted_edges), len(region_faces))
    net = (created[0] - deleted[0], created[1] - deleted[1], created[2] - deleted[2])
    return created, deleted, net


def _vert_role(
    vertex: bmesh.types.BMVert,
    vertex_id: int,
    snapshot: ExtrudeSnapshot,
    classified: ExtrudeClassification,
) -> str:
    copy = classified.copies.get(vertex_id)
    if copy is not None and copy.is_valid and copy is vertex:
        return "copy"
    origin = classified.origins.get(vertex_id)
    if origin is not None and origin.is_valid and origin is vertex:
        return "origin"
    preop = snapshot.vertex_preop_map().get(vertex_id)
    if vertex_id in classified.copies and preop is not None:
        if vertex_id in classified.vanished_preop:
            return "copy"
        if not matching.coordinates_match(vertex.co, preop.as_tuple(), snapshot.tolerance):
            return "copy"
    if vertex_id in classified.origins and preop is not None:
        if matching.coordinates_match(vertex.co, preop.as_tuple(), snapshot.tolerance):
            return "origin"
    return "other"


def _face_corner_signature(
    face: bmesh.types.BMFace,
    vertex_layer,
    snapshot: ExtrudeSnapshot,
    classified: ExtrudeClassification,
) -> tuple[tuple[tuple[int, str], ...] | None, str | None]:
    signature: list[tuple[int, str]] = []
    for loop in face.loops:
        vertex = loop.vert
        vertex_id = int(vertex[vertex_layer])
        role = _vert_role(vertex, vertex_id, snapshot, classified)
        if role not in {"origin", "copy"}:
            return None, "a new face corner is not a classified origin or copy"
        signature.append((vertex_id, role))
    return tuple(signature), None


def _signatures_match(
    first: tuple[tuple[int, str], ...],
    second: tuple[tuple[int, str], ...],
) -> bool:
    if len(first) != len(second):
        return False
    if not first:
        return not second
    return any(first[index:] + first[:index] == second for index in range(len(first)))


def _locate_face_by_signature(
    bm: bmesh.types.BMesh,
    signature: tuple[tuple[int, str], ...],
    vertex_layer,
    snapshot: ExtrudeSnapshot,
    classified: ExtrudeClassification,
) -> tuple[bmesh.types.BMFace | None, str | None]:
    matches: list[bmesh.types.BMFace] = []
    for face in bm.faces:
        if not face.is_valid:
            continue
        live_signature, _reason = _face_corner_signature(face, vertex_layer, snapshot, classified)
        if live_signature is None:
            continue
        if _signatures_match(live_signature, signature):
            matches.append(face)
    if len(matches) == 0:
        return None, "a source new face could not be relocated after mirroring"
    if len(matches) > 1:
        return None, "a source new face signature matched more than one face"
    return matches[0], None


def check_origin_stationarity(
    bm: bmesh.types.BMesh,
    snapshot: ExtrudeSnapshot,
    classified: ExtrudeClassification,
) -> str | None:
    """Decline when class (a) or (b) origins moved or disappeared."""

    vertex_layer = bm.verts.layers.int.get(layer_names.VERT_SESSION_ID_LAYER)
    if vertex_layer is None:
        return "the session vertex ID layer is missing"
    groups = _verts_by_session_id(bm, vertex_layer)
    preop = snapshot.vertex_preop_map()
    selected = snapshot.selected_vertex_ids
    tolerance = snapshot.tolerance

    for vertex_id, pre in preop.items():
        if vertex_id in selected:
            continue
        instances = [vertex for vertex in groups.get(vertex_id, []) if vertex.is_valid]
        if len(instances) != 1:
            return "a non-target vertex is missing or has extra instances"
        if not matching.coordinates_match(instances[0].co, pre.as_tuple(), tolerance):
            return "a non-target vertex moved during the native extrude"

    boundary_ids = [entry.vertex_id for entry in classified.freeze if entry.entity_class == "b"] or list(
        classified.origins
    )
    for vertex_id in boundary_ids:
        origin = classified.origins.get(vertex_id)
        if origin is None or not origin.is_valid:
            return "a classified origin vertex is missing"
        pre = preop.get(vertex_id)
        if pre is None or not matching.coordinates_match(origin.co, pre.as_tuple(), tolerance):
            return "a classified origin vertex moved during the native extrude"
        copy = classified.copies.get(vertex_id)
        extras = [
            vertex
            for vertex in groups.get(vertex_id, [])
            if vertex.is_valid and vertex is not origin and vertex is not copy
        ]
        if extras:
            return "a classified origin vertex has extra instances"
    return None


@dataclass
class ExtrudeApplyAudit:
    """Pre/post apply identity snapshots used by verification."""

    created: tuple[int, int, int]
    deleted: tuple[int, int, int]
    source_vert_coords: dict[int, tuple[float, float, float]]
    source_vert_select: dict[int, bool]
    source_edge_ends: dict[int, frozenset[int]]
    source_face_corners: dict[int, tuple[int, ...]]


def apply_mirror(
    bm: bmesh.types.BMesh,
    snapshot: ExtrudeSnapshot,
    classified: ExtrudeClassification,
    description: ExtrudeSourceDescription,
) -> tuple[dict[int, bmesh.types.BMVert], str | None, ExtrudeApplyAudit | None]:
    """Delete then build the mirror. Caller owns the backup checkpoint."""

    vertex_layer = bm.verts.layers.int.get(layer_names.VERT_SESSION_ID_LAYER)
    edge_layer = bm.edges.layers.int.get(layer_names.EDGE_ORIGINAL_LAYER)
    face_layer = bm.faces.layers.int.get(layer_names.FACE_ID_LAYER)
    if vertex_layer is None or edge_layer is None or face_layer is None:
        return {}, "temporary topology markers are missing", None

    vertex_pairs = snapshot.vertex_pair_map()
    edge_pairs = snapshot.edge_pair_map()
    face_pairs = snapshot.face_pair_map()
    preop = snapshot.vertex_preop_map()

    copy_coords = {
        vertex_id: (float(copy.co.x), float(copy.co.y), float(copy.co.z))
        for vertex_id, copy in classified.copies.items()
        if copy.is_valid
    }
    edge_specs: list[tuple[tuple[int, str], tuple[int, str]]] = []
    for edge in description.new_edges:
        if not edge.is_valid:
            return {}, "a source new edge was lost before mirroring", None
        ends = []
        for vertex in edge.verts:
            vertex_id = int(vertex[vertex_layer])
            role = _vert_role(vertex, vertex_id, snapshot, classified)
            if role not in {"origin", "copy"}:
                return {}, "a new edge endpoint is not a classified origin or copy", None
            ends.append((vertex_id, role))
        edge_specs.append((ends[0], ends[1]))

    if len(description.face_signatures) != len(description.new_faces):
        return {}, "new-face signatures are not aligned with the source faces", None
    face_specs: list[tuple[list[tuple[int, str]], int, bool, tuple[tuple[int, str], ...]]] = []
    for face, stored_signature in zip(description.new_faces, description.face_signatures, strict=True):
        if not face.is_valid:
            return {}, "a source new face was lost before mirroring", None
        corners: list[tuple[int, str]] = []
        for loop in face.loops:
            vertex = loop.vert
            vertex_id = int(vertex[vertex_layer])
            role = _vert_role(vertex, vertex_id, snapshot, classified)
            if role not in {"origin", "copy"}:
                return {}, "a new face corner is not a classified origin or copy", None
            corners.append((vertex_id, role))
        live_signature = tuple(corners)
        if not _signatures_match(live_signature, stored_signature):
            return {}, "a source new face no longer matches its stored signature", None
        face_specs.append(
            (
                corners,
                int(face.material_index),
                bool(face.smooth),
                stored_signature,
            )
        )

    token_layers = _stamp_apply_tokens(bm)
    before_idents = _element_token_sets(bm, token_layers)
    vertex_layer = bm.verts.layers.int.get(layer_names.VERT_SESSION_ID_LAYER)
    edge_layer = bm.edges.layers.int.get(layer_names.EDGE_ORIGINAL_LAYER)
    face_layer = bm.faces.layers.int.get(layer_names.FACE_ID_LAYER)
    if vertex_layer is None or edge_layer is None or face_layer is None:
        return {}, "temporary topology markers were lost while stamping apply tokens", None

    faces_by_id: dict[int, list[bmesh.types.BMFace]] = {}
    for face in bm.faces:
        faces_by_id.setdefault(int(face[face_layer]), []).append(face)
    edges_by_marker: dict[int, list[bmesh.types.BMEdge]] = {}
    for edge in bm.edges:
        edges_by_marker.setdefault(int(edge[edge_layer]), []).append(edge)
    verts_by_id = _verts_by_session_id(bm, vertex_layer)

    delete_faces: list[bmesh.types.BMFace] = []
    for face_id in description.deleted_face_ids:
        partner = face_pairs.get(face_id)
        if partner is None:
            continue
        matches = faces_by_id.get(partner, [])
        if len(matches) == 1:
            delete_faces.append(matches[0])
        elif len(matches) > 1:
            return {}, "a deleted face has an ambiguous mirrored counterpart", None

    delete_edges: list[bmesh.types.BMEdge] = []
    for marker in description.deleted_edge_markers:
        partner = edge_pairs.get(marker)
        if partner is None:
            continue
        matches = edges_by_marker.get(partner, [])
        if len(matches) == 1:
            delete_edges.append(matches[0])
        elif len(matches) > 1:
            return {}, "a deleted edge has an ambiguous mirrored counterpart", None

    delete_verts: list[bmesh.types.BMVert] = []
    for vertex_id in description.deleted_vertex_ids:
        partner = vertex_pairs.get(vertex_id)
        if partner is None:
            continue
        matches = verts_by_id.get(partner, [])
        if len(matches) == 1:
            delete_verts.append(matches[0])
        elif len(matches) > 1:
            return {}, "a deleted vertex has an ambiguous mirrored counterpart", None

    delete_face_set = set(delete_faces)
    delete_edge_set = set(delete_edges)
    delete_vert_set = set(delete_verts)
    vert_token, edge_token, face_token = token_layers
    source_vert_coords = {
        int(vertex[vert_token]): (float(vertex.co.x), float(vertex.co.y), float(vertex.co.z))
        for vertex in bm.verts
        if vertex not in delete_vert_set
    }
    source_vert_select = {
        int(vertex[vert_token]): bool(vertex.select) for vertex in bm.verts if vertex not in delete_vert_set
    }
    source_edge_ends = {
        int(edge[edge_token]): frozenset(int(vertex[vert_token]) for vertex in edge.verts)
        for edge in bm.edges
        if edge not in delete_edge_set
    }
    source_face_corners = {
        int(face[face_token]): tuple(int(loop.vert[vert_token]) for loop in face.loops)
        for face in bm.faces
        if face not in delete_face_set
    }

    if delete_faces:
        bmesh.ops.delete(bm, geom=delete_faces, context="FACES_ONLY")
    if delete_edges:
        still = [edge for edge in delete_edges if edge.is_valid]
        if still:
            bmesh.ops.delete(bm, geom=still, context="EDGES")
    if delete_verts:
        still = [vertex for vertex in delete_verts if vertex.is_valid]
        if still:
            bmesh.ops.delete(bm, geom=still, context="VERTS")

    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    vertex_layer = bm.verts.layers.int.get(layer_names.VERT_SESSION_ID_LAYER)
    if vertex_layer is None:
        return {}, "the session vertex ID layer was lost during delete", None

    mirror_copies: dict[int, bmesh.types.BMVert] = {}
    for vertex_id, coord in copy_coords.items():
        mirrored = matching.mirror_coordinate(Vector(coord), snapshot.axis_index)
        new_vert = bm.verts.new(mirrored)
        new_vert.select = False
        mirror_copies[vertex_id] = new_vert

    vertex_layer = bm.verts.layers.int.get(layer_names.VERT_SESSION_ID_LAYER)
    if vertex_layer is None:
        return {}, "the session vertex ID layer was lost while creating copies", None
    verts_by_id = _verts_by_session_id(bm, vertex_layer)

    def _mirror_of_spec(vertex_id: int, role: str) -> bmesh.types.BMVert | None:
        if role not in {"copy", "origin"}:
            return None
        if role == "copy":
            return mirror_copies.get(vertex_id)
        partner = vertex_pairs.get(vertex_id)
        if partner is None:
            return None
        matches = [vertex for vertex in verts_by_id.get(partner, []) if vertex.is_valid]
        if len(matches) == 1:
            return matches[0]
        if vertex_id in preop and matches:
            reflected = matching.mirror_coordinate(Vector(preop[vertex_id].as_tuple()), snapshot.axis_index)
            stayed = [
                vertex for vertex in matches if matching.coordinates_match(vertex.co, reflected, snapshot.tolerance)
            ]
            if len(stayed) == 1:
                return stayed[0]
        return None

    for first_spec, second_spec in edge_specs:
        first = _mirror_of_spec(*first_spec)
        second = _mirror_of_spec(*second_spec)
        if first is None or second is None:
            return {}, "a mirrored new edge is missing an endpoint", None
        if first == second:
            return {}, "a mirrored new edge would be degenerate", None
        existing = next((candidate for candidate in first.link_edges if candidate.other_vert(first) is second), None)
        if existing is not None:
            return {}, "a mirrored new edge already exists", None
        try:
            bm.edges.new((first, second))
        except ValueError:
            return {}, "a mirrored new edge already exists", None

    for corners, material_index, smooth, stored_signature in face_specs:
        mirror_corners = []
        for vertex_id, role in corners:
            mapped = _mirror_of_spec(vertex_id, role)
            if mapped is None:
                return {}, "a mirrored new face is missing a corner", None
            mirror_corners.append(mapped)
        if len(set(mirror_corners)) != len(mirror_corners):
            return {}, "the mirrored extrude face would be degenerate", None
        source_face, locate_reason = _locate_face_by_signature(
            bm,
            stored_signature,
            vertex_layer,
            snapshot,
            classified,
        )
        if source_face is None:
            return {}, locate_reason or "a source new face could not be relocated after mirroring", None
        source_loops = list(source_face.loops)
        try:
            mirror_face = bm.faces.new(list(reversed(mirror_corners)))
        except ValueError:
            return {}, "the mirrored extrude face already exists", None
        mirror_face.material_index = material_index
        mirror_face.smooth = smooth
        mirror_face.select = False
        for mirror_loop, source_loop in zip(mirror_face.loops, reversed(source_loops), strict=True):
            mirror_loop.copy_from(source_loop)
        mirror_face.normal_update()

    token_layers = (
        bm.verts.layers.int.get(layer_names.VERT_APPLY_ID_LAYER),
        bm.edges.layers.int.get(layer_names.EDGE_APPLY_ID_LAYER),
        bm.faces.layers.int.get(layer_names.FACE_APPLY_ID_LAYER),
    )
    if any(layer is None for layer in token_layers):
        return {}, "apply identity tokens were lost during mirroring", None
    after_idents = _element_token_sets(bm, token_layers)
    created = _count_untokened(bm, token_layers)
    deleted = (
        len(before_idents[0] - after_idents[0]),
        len(before_idents[1] - after_idents[1]),
        len(before_idents[2] - after_idents[2]),
    )
    audit = ExtrudeApplyAudit(
        created=created,
        deleted=deleted,
        source_vert_coords=source_vert_coords,
        source_vert_select=source_vert_select,
        source_edge_ends=source_edge_ends,
        source_face_corners=source_face_corners,
    )
    return mirror_copies, None, audit


def _stamp_apply_tokens(bm: bmesh.types.BMesh):
    names = (
        (bm.verts, layer_names.VERT_APPLY_ID_LAYER),
        (bm.edges, layer_names.EDGE_APPLY_ID_LAYER),
        (bm.faces, layer_names.FACE_APPLY_ID_LAYER),
    )
    layers = []
    for sequence, name in names:
        old = sequence.layers.int.get(name)
        if old is not None:
            sequence.layers.int.remove(old)
        layer = sequence.layers.int.new(name)
        for index, element in enumerate(sequence, start=1):
            element[layer] = index
        layers.append(layer)
    return tuple(layers)


def _element_token_sets(bm: bmesh.types.BMesh, token_layers) -> tuple[set[int], set[int], set[int]]:
    vert_token, edge_token, face_token = token_layers
    return (
        {int(vertex[vert_token]) for vertex in bm.verts if vertex.is_valid and int(vertex[vert_token]) > 0},
        {int(edge[edge_token]) for edge in bm.edges if edge.is_valid and int(edge[edge_token]) > 0},
        {int(face[face_token]) for face in bm.faces if face.is_valid and int(face[face_token]) > 0},
    )


def _count_untokened(bm: bmesh.types.BMesh, token_layers) -> tuple[int, int, int]:
    vert_token, edge_token, face_token = token_layers
    return (
        sum(1 for vertex in bm.verts if vertex.is_valid and int(vertex[vert_token]) == 0),
        sum(1 for edge in bm.edges if edge.is_valid and int(edge[edge_token]) == 0),
        sum(1 for face in bm.faces if face.is_valid and int(face[face_token]) == 0),
    )


def verify_mirror(
    bm: bmesh.types.BMesh,
    snapshot: ExtrudeSnapshot,
    classified: ExtrudeClassification,
    description: ExtrudeSourceDescription,
    mirror_copies: dict[int, bmesh.types.BMVert],
    source_copy_coords: dict[int, tuple[float, float, float]],
    source_origin_coords: dict[int, tuple[float, float, float]],
    audit: ExtrudeApplyAudit,
) -> str | None:
    """Verify mirror census, reflection, and source-side stationarity."""

    if audit.created != description.created or audit.deleted != description.deleted:
        return (
            f"mirror census {audit.created}/{audit.deleted} does not match source "
            f"{description.created}/{description.deleted}"
        )
    audit_net = (
        audit.created[0] - audit.deleted[0],
        audit.created[1] - audit.deleted[1],
        audit.created[2] - audit.deleted[2],
    )
    if audit_net != description.net:
        return f"mirror net {audit_net} does not match source {description.net}"

    if len(mirror_copies) != len(description.new_verts):
        return "mirrored new vertex count does not match the source"
    vertex_layer = bm.verts.layers.int.get(layer_names.VERT_SESSION_ID_LAYER)
    if vertex_layer is None:
        return "the session vertex ID layer is missing"
    live_groups = _verts_by_session_id(bm, vertex_layer)
    for vertex_id, expected in source_copy_coords.items():
        copy = _unique_vert_at(live_groups.get(vertex_id, ()), expected, snapshot.tolerance)
        if copy is None:
            return "a source copy was lost during mirroring"
    for vertex_id, expected in source_origin_coords.items():
        origin = _unique_vert_at(live_groups.get(vertex_id, ()), expected, snapshot.tolerance)
        if origin is None:
            return "a source origin was lost during mirroring"

    vert_token = bm.verts.layers.int.get(layer_names.VERT_APPLY_ID_LAYER)
    edge_token = bm.edges.layers.int.get(layer_names.EDGE_APPLY_ID_LAYER)
    face_token = bm.faces.layers.int.get(layer_names.FACE_APPLY_ID_LAYER)
    if vert_token is None or edge_token is None or face_token is None:
        return "apply identity tokens are missing"
    live_verts = {int(vertex[vert_token]): vertex for vertex in bm.verts if vertex.is_valid and int(vertex[vert_token])}
    live_edges = {int(edge[edge_token]): edge for edge in bm.edges if edge.is_valid and int(edge[edge_token])}
    live_faces = {int(face[face_token]): face for face in bm.faces if face.is_valid and int(face[face_token])}
    for ident, coords in audit.source_vert_coords.items():
        vertex = live_verts.get(ident)
        if vertex is None:
            return "a source vertex disappeared during mirroring"
        if not matching.coordinates_match(vertex.co, coords, snapshot.tolerance):
            return "a source vertex coordinate changed during mirroring"
        if ident in audit.source_vert_select and bool(vertex.select) != audit.source_vert_select[ident]:
            return "a source vertex selection changed during mirroring"
    for ident, ends in audit.source_edge_ends.items():
        edge = live_edges.get(ident)
        if edge is None:
            return "a source edge disappeared during mirroring"
        if frozenset(int(vertex[vert_token]) for vertex in edge.verts) != ends:
            return "a source edge incidence changed during mirroring"
    for ident, corners in audit.source_face_corners.items():
        face = live_faces.get(ident)
        if face is None:
            return "a source face disappeared during mirroring"
        if tuple(int(loop.vert[vert_token]) for loop in face.loops) != corners:
            return "a source face incidence changed during mirroring"

    for vertex_id, mirror in mirror_copies.items():
        if not mirror.is_valid:
            return "a mirrored copy vertex is invalid"
        if not all(math.isfinite(float(mirror.co[index])) for index in range(3)):
            return "a mirrored copy coordinate is not finite"
        expected_source = source_copy_coords.get(vertex_id)
        if expected_source is None:
            return "a mirrored copy has no recorded source coordinate"
        reflected = matching.mirror_coordinate(Vector(expected_source), snapshot.axis_index)
        if not matching.coordinates_match(mirror.co, reflected, snapshot.tolerance):
            return "a mirrored copy is not a reflection of the source copy"
    return None


def _unique_vert_at(
    instances,
    expected: tuple[float, float, float],
    tolerance: float,
) -> bmesh.types.BMVert | None:
    matches = [
        vertex for vertex in instances if vertex.is_valid and matching.coordinates_match(vertex.co, expected, tolerance)
    ]
    return matches[0] if len(matches) == 1 else None


def intervening_operator_reason(session, context) -> str | None:
    """State-1 reason when another modal or active_operator replaced the macro."""

    window_manager = getattr(context, "window_manager", None)
    window = None
    if window_manager is not None:
        window = next(
            (candidate for candidate in window_manager.windows if candidate.as_pointer() == session.window_pointer),
            None,
        )
    if window is not None and tuple(window.modal_operators):
        return "another modal operator started before the extrude could be mirrored"
    if not session.confirmed_operator_pointer or not session.confirmed_operator_idname:
        return "the confirmed extrude operator could not be captured"

    area = None
    region = None
    if window is not None:
        area = next(
            (candidate for candidate in window.screen.areas if candidate.as_pointer() == session.area_pointer),
            None,
        )
        if area is None:
            area = next((candidate for candidate in window.screen.areas if candidate.type == "VIEW_3D"), None)
    if area is not None:
        region = next(
            (candidate for candidate in area.regions if candidate.as_pointer() == session.region_pointer),
            None,
        )
        if region is None:
            region = next((candidate for candidate in area.regions if candidate.type == "WINDOW"), None)
    if window is None or area is None or region is None:
        return "the confirmed extrude operator could not be captured"
    try:
        with context.temp_override(window=window, area=area, region=region):
            active = getattr(context, "active_operator", None)
            operator_stack = tuple(getattr(getattr(context, "window_manager", None), "operators", ()) or ())
    except Exception:
        return "the confirmed extrude operator could not be read"
    if active is None:
        return "the confirmed extrude operator is no longer active"
    try:
        pointer = int(active.as_pointer())
    except (AttributeError, ReferenceError, RuntimeError):
        return "the confirmed extrude operator could not be read"
    if active.bl_idname != session.confirmed_operator_idname or pointer != session.confirmed_operator_pointer:
        return "a different operator became active before the extrude could be mirrored"
    if operator_stack:
        last = operator_stack[-1]
        try:
            last_idname = str(last.bl_idname)
            last_pointer = int(last.as_pointer())
        except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
            return "the confirmed extrude operator could not be read"
        if last_idname != session.confirmed_operator_idname or last_pointer != session.confirmed_operator_pointer:
            return "a different operator became active before the extrude could be mirrored"
    captured = getattr(session, "confirmed_selection_signature", ())
    if captured:
        current = _live_selection_signature(session)
        if current and current != captured:
            return "the selection changed before the extrude could be mirrored"
    return None


def _live_selection_signature(session) -> tuple[tuple[int, bool], ...]:
    obj = bpy.data.objects.get(session.object_name)
    if obj is None or obj.type != "MESH" or obj.mode != "EDIT":
        return ()
    try:
        bm = bmesh.from_edit_mesh(obj.data)
        vertex_layer = bm.verts.layers.int.get(layer_names.VERT_SESSION_ID_LAYER)
        if vertex_layer is None:
            return ()
        return tuple(
            sorted(
                (int(vertex[vertex_layer]), bool(vertex.select)) for vertex in bm.verts if int(vertex[vertex_layer]) > 0
            )
        )
    except (ReferenceError, RuntimeError):
        return ()
