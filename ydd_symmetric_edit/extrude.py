# SPDX-License-Identifier: GPL-3.0-or-later

"""Symmetric postprocess for Blender's native region and individual extrude macros.

Stage 1–4 cover ``EXTRUDE_NORMAL``, ``EXTRUDE_CONTEXT``,
``EXTRUDE_SHRINK_FATTEN``, ``EXTRUDE_FACES_INDIV``, ``EXTRUDE_EDGES_INDIV``,
``EXTRUDE_VERTS_INDIV``, and ``EXTRUDE_MANIFOLD``. Classification uses
vertex-ID instance groups and a freeze table; FACE_ID set-difference and live
selection are not discriminators. N-copies of one vid (FACES_INDIV only) are
keyed by ``(vertex_id, source_face_signature)``, not by vid alone. Edges/verts
indiv are 1:1 class-(b) only (shared-edge verts stay shared — F13). Manifold
is region-family: apply only when region-congruent; dissolve/weld decline via
census mismatch or pattern miss (F15).
"""

from __future__ import annotations

import math
import traceback
from collections import Counter
from dataclasses import dataclass
from itertools import permutations

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
class ExtrudeCopyInstance:
    """One classified copy, including N copies of the same vertex ID."""

    vertex_id: int
    vertex: bmesh.types.BMVert
    entity_class: str
    source_face_signature: tuple[int, ...]


@dataclass
class ExtrudeClassification:
    """Live origin/copy assignment after classify or freeze-table reconnect."""

    origins: dict[int, bmesh.types.BMVert]
    copies: dict[int, bmesh.types.BMVert]
    copy_instances: tuple[ExtrudeCopyInstance, ...]
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


def _canonical_coordinate_cycle(
    cycle: tuple[tuple[float, float, float], ...],
) -> tuple[tuple[float, float, float], ...]:
    """Canonicalize a face cycle by rotation, preserving winding."""

    if not cycle:
        return ()
    return min(cycle[index:] + cycle[:index] for index in range(len(cycle)))


def _canonical_id_cycle(cycle: tuple[int, ...]) -> tuple[int, ...]:
    if not cycle:
        return ()
    return min(cycle[index:] + cycle[:index] for index in range(len(cycle)))


def _snapshot_coordinate_domains(snapshot: ExtrudeSnapshot):
    preop = snapshot.vertex_preop_map()
    vertices = tuple(coordinate.as_tuple() for _vertex_id, coordinate in snapshot.vertex_preop)
    edges = []
    for first, second in snapshot.edge_endpoint_map().values():
        first_co = preop.get(first)
        second_co = preop.get(second)
        if first_co is None or second_co is None:
            return None
        edges.append(tuple(sorted((first_co.as_tuple(), second_co.as_tuple()))))
    faces = []
    for corners in snapshot.face_corner_map().values():
        try:
            coordinates = tuple(preop[vertex_id].as_tuple() for vertex_id in corners)
        except KeyError:
            return None
        faces.append(_canonical_coordinate_cycle(coordinates))
    return tuple(vertices), tuple(edges), tuple(faces)


_EXACT_MAPPING_STEP_LIMIT = 2_000


def _exact_coordinate_mapping(
    snapshot: ExtrudeSnapshot,
    bm: bmesh.types.BMesh,
    mesh_select_mode: MeshSelectionMode,
) -> dict[int, int] | None:
    """Find the unique topology-valid bijection among exactly equal coordinates."""

    snapshot_by_coordinate: dict[tuple[float, float, float], list[int]] = {}
    for vertex_id, coordinate in snapshot.vertex_preop:
        snapshot_by_coordinate.setdefault(coordinate.as_tuple(), []).append(vertex_id)
    live_by_coordinate: dict[tuple[float, float, float], list[int]] = {}
    for vertex in bm.verts:
        coordinate = (float(vertex.co.x), float(vertex.co.y), float(vertex.co.z))
        live_by_coordinate.setdefault(coordinate, []).append(int(vertex.index))
    if set(snapshot_by_coordinate) != set(live_by_coordinate) or any(
        len(snapshot_by_coordinate[coordinate]) != len(live_by_coordinate[coordinate])
        for coordinate in snapshot_by_coordinate
    ):
        return None

    mapping: dict[int, int] = {}
    duplicate_groups: list[tuple[list[int], list[int]]] = []
    for coordinate, snapshot_ids in snapshot_by_coordinate.items():
        live_indices = live_by_coordinate[coordinate]
        if len(snapshot_ids) == 1:
            mapping[snapshot_ids[0]] = live_indices[0]
        else:
            duplicate_groups.append((snapshot_ids, live_indices))
    if not duplicate_groups:
        return mapping

    solutions: list[dict[int, int]] = []
    leaves = 0

    def search(group_index: int) -> None:
        nonlocal leaves
        if len(solutions) > 1 or leaves >= _EXACT_MAPPING_STEP_LIMIT:
            return
        if group_index == len(duplicate_groups):
            leaves += 1
            if _mapped_domains_match(snapshot, bm, mapping, mesh_select_mode, include_selection=False):
                solutions.append(dict(mapping))
            return
        snapshot_ids, live_indices = duplicate_groups[group_index]
        for assignment in permutations(live_indices):
            mapping.update(zip(snapshot_ids, assignment, strict=True))
            search(group_index + 1)
            if len(solutions) > 1 or leaves >= _EXACT_MAPPING_STEP_LIMIT:
                return

    search(0)
    if leaves >= _EXACT_MAPPING_STEP_LIMIT or len(solutions) != 1:
        return None
    return solutions[0]


def _solver_vertex_mapping(snapshot: ExtrudeSnapshot, live_vertices) -> dict[int, int] | None:
    live_coordinates = [(float(vertex.co.x), float(vertex.co.y), float(vertex.co.z)) for vertex in live_vertices]
    lookup = matching.build_vertex_mirror_lookup(live_coordinates, snapshot.axis_index, snapshot.tolerance)
    resolved = lookup.find_all_direct(
        [Vector(coordinate.as_tuple()) for _vertex_id, coordinate in snapshot.vertex_preop]
    )
    resolved_indices = [index for index in resolved if index is not None]
    if len(resolved) != len(snapshot.vertex_preop) or len(resolved_indices) != len(resolved):
        return None
    return {
        vertex_id: int(live_index)
        for (vertex_id, _coordinate), live_index in zip(snapshot.vertex_preop, resolved_indices, strict=True)
    }


def _mapped_domains_match(
    snapshot: ExtrudeSnapshot,
    bm: bmesh.types.BMesh,
    mapping: dict[int, int],
    mesh_select_mode: MeshSelectionMode,
    *,
    include_selection: bool = True,
) -> bool:
    """Validate every topology and selection domain through one vertex map."""

    live_to_snapshot = {live_index: vertex_id for vertex_id, live_index in mapping.items()}
    if len(live_to_snapshot) != len(bm.verts):
        return False
    expected_edges = Counter(tuple(sorted(endpoints)) for endpoints in snapshot.edge_endpoint_map().values())
    live_edges = Counter()
    for edge in bm.edges:
        endpoints = tuple(sorted(live_to_snapshot.get(int(vertex.index), -1) for vertex in edge.verts))
        if -1 in endpoints:
            return False
        live_edges[endpoints] += 1
    if live_edges != expected_edges:
        return False

    expected_faces = Counter(_canonical_id_cycle(corners) for corners in snapshot.face_corner_map().values())
    live_faces = Counter()
    for face in bm.faces:
        corners = tuple(live_to_snapshot.get(int(loop.vert.index), -1) for loop in face.loops)
        if -1 in corners:
            return False
        live_faces[_canonical_id_cycle(corners)] += 1
    if live_faces != expected_faces:
        return False
    if not include_selection:
        return True
    if mesh_select_mode != snapshot.mesh_select_mode:
        return False

    selected_vertices = {live_to_snapshot[int(vertex.index)] for vertex in bm.verts if vertex.select}
    if selected_vertices != set(snapshot.selected_vertex_ids):
        return False
    edge_by_marker = snapshot.edge_endpoint_map()
    expected_selected_edges = Counter(
        tuple(sorted(edge_by_marker[marker])) for marker in snapshot.selected_edge_markers if marker in edge_by_marker
    )
    if expected_selected_edges.total() != len(snapshot.selected_edge_markers):
        return False
    live_selected_edges = Counter()
    for edge in bm.edges:
        if not edge.select:
            continue
        endpoints = tuple(sorted(live_to_snapshot.get(int(vertex.index), -1) for vertex in edge.verts))
        if -1 in endpoints:
            return False
        live_selected_edges[endpoints] += 1
    if live_selected_edges != expected_selected_edges:
        return False

    face_by_id = snapshot.face_corner_map()
    expected_selected_faces = Counter(
        _canonical_id_cycle(face_by_id[face_id]) for face_id in snapshot.selected_face_ids if face_id in face_by_id
    )
    if expected_selected_faces.total() != len(snapshot.selected_face_ids):
        return False
    live_selected_faces = Counter()
    for face in bm.faces:
        if not face.select:
            continue
        corners = tuple(live_to_snapshot.get(int(loop.vert.index), -1) for loop in face.loops)
        if -1 in corners:
            return False
        live_selected_faces[_canonical_id_cycle(corners)] += 1
    return live_selected_faces == expected_selected_faces


def repeat_baseline_matches(
    bm: bmesh.types.BMesh,
    snapshot: ExtrudeSnapshot,
    *,
    mesh_select_mode: MeshSelectionMode,
) -> bool:
    """Validate the pre-stamp mesh before an extrude F9 re-execution.

    The fast path compares exact float coordinate/topology multisets.  Only a
    mismatch enters the existing minimum-total-Chebyshev injective solver;
    the resulting single vertex mapping is then the sole basis for all
    incidence and selection checks.
    """

    if (len(bm.verts), len(bm.edges), len(bm.faces)) != (
        snapshot.vertex_count,
        snapshot.edge_count,
        snapshot.face_count,
    ):
        return False
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    bm.verts.index_update()
    bm.edges.index_update()
    bm.faces.index_update()
    live_vertices = list(bm.verts)
    live_domains = (
        tuple((float(vertex.co.x), float(vertex.co.y), float(vertex.co.z)) for vertex in live_vertices),
        tuple(
            tuple(
                sorted(
                    (
                        (float(edge.verts[0].co.x), float(edge.verts[0].co.y), float(edge.verts[0].co.z)),
                        (float(edge.verts[1].co.x), float(edge.verts[1].co.y), float(edge.verts[1].co.z)),
                    )
                )
            )
            for edge in bm.edges
        ),
        tuple(
            _canonical_coordinate_cycle(
                tuple((float(loop.vert.co.x), float(loop.vert.co.y), float(loop.vert.co.z)) for loop in face.loops)
            )
            for face in bm.faces
        ),
    )
    snapshot_domains = _snapshot_coordinate_domains(snapshot)
    if snapshot_domains is None:
        return False
    raw_equal = all(
        Counter(live_domain) == Counter(snapshot_domain)
        for live_domain, snapshot_domain in zip(live_domains, snapshot_domains, strict=True)
    )
    if raw_equal:
        mapping = _exact_coordinate_mapping(snapshot, bm, mesh_select_mode)
        return mapping is not None and _mapped_domains_match(snapshot, bm, mapping, mesh_select_mode)
    mapping = _solver_vertex_mapping(snapshot, live_vertices)
    if mapping is None or len(mapping) != len(live_vertices):
        return False
    return _mapped_domains_match(snapshot, bm, mapping, mesh_select_mode)


def evaluate_prepare_gates(snapshot: ExtrudeSnapshot) -> tuple[str, str]:
    """Return (APPLY|DECLINE, reason) from the snapshot only (G4–G6)."""

    preop = snapshot.vertex_preop_map()
    vertex_pairs = snapshot.vertex_pair_map()
    edge_pairs = snapshot.edge_pair_map()
    face_pairs = snapshot.face_pair_map()
    axis = snapshot.axis_index
    tol = snapshot.tolerance

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
    "EXTRUDE_FACES_INDIV": ("MESH_OT_extrude_faces_indiv", "TRANSFORM_OT_shrink_fatten"),
    "EXTRUDE_EDGES_INDIV": ("MESH_OT_extrude_edges_indiv", "TRANSFORM_OT_translate"),
    "EXTRUDE_VERTS_INDIV": ("MESH_OT_extrude_verts_indiv", "TRANSFORM_OT_translate"),
    "EXTRUDE_MANIFOLD": ("MESH_OT_extrude_region", "TRANSFORM_OT_translate"),
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
        # All kinds share this capture shape: topology children that lack
        # flip/dissolve/mirror RNA (faces/edges/verts indiv) fall back to False.
        use_normal_flip = bool(getattr(topology, "use_normal_flip", False))
        use_dissolve_ortho_edges = bool(getattr(topology, "use_dissolve_ortho_edges", False))
        mirror = bool(getattr(topology, "mirror", False))
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
            elif identifier == "value":
                vector = _as_numeric_tuple(value)
                captured.append((identifier, vector if vector is not None else str(value)))
            else:
                captured.append((identifier, str(value)))
        if not any(name == "value" for name, _unused in captured):
            raw = operator.value
            if isinstance(raw, (bool, int, float, str)):
                captured.append(("value", raw))
            else:
                vector = _as_numeric_tuple(raw)
                captured.append(("value", vector if vector is not None else float(raw)))
    except Exception:
        traceback.print_exc()
        return None
    return tuple(captured)


def _as_numeric_tuple(value) -> tuple[float, ...] | None:
    try:
        items = tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return None
    if not items or not all(math.isfinite(item) for item in items):
        return None
    return items


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
    pending_ncopy: list[tuple[int, list[bmesh.types.BMVert]]] = []

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
            # Class (c) is region-family only; indiv kinds allow (b) or (b)+(d).
            if snapshot.tool_kind in {
                "EXTRUDE_FACES_INDIV",
                "EXTRUDE_EDGES_INDIV",
                "EXTRUDE_VERTS_INDIV",
            }:
                return None, "interior-copy pattern is not allowed for this extrude kind"
            copies[vertex_id] = instances[0]
            vanished[vertex_id] = pre
            assigned.add(vertex_id)
            continue
        moved = [vertex for vertex in instances if not matching.coordinates_match(vertex.co, pre.as_tuple(), tol)]
        stayed = [vertex for vertex in instances if matching.coordinates_match(vertex.co, pre.as_tuple(), tol)]
        if count == 2:
            if len(moved) == 0:
                return None, "zero-offset extrude is not mirrored"
            if len(moved) == 1 and len(stayed) == 1:
                origins[vertex_id] = stayed[0]
                copies[vertex_id] = moved[0]
                assigned.add(vertex_id)
                continue
            return None, "a duplicated vertex could not be classified by movement"
        if snapshot.tool_kind != "EXTRUDE_FACES_INDIV":
            return None, "an unsupported N-duplicate vertex group was observed"
        if len(moved) == 0:
            return None, "zero-offset extrude is not mirrored"
        if len(stayed) != 1 or len(moved) < 2:
            return None, "an N-duplicate vertex group did not have a unique stationary origin"
        origins[vertex_id] = stayed[0]
        pending_ncopy.append((vertex_id, moved))
        assigned.add(vertex_id)

    missing_targets = selected - assigned
    if missing_targets:
        return None, "a selected extrusion vertex disappeared without a remaining copy"

    copy_instances, ncopy_reason = _classify_ncopy_instances(bm, snapshot, copies, vanished, pending_ncopy)
    if ncopy_reason is not None or copy_instances is None:
        return None, ncopy_reason
    freeze = tuple(
        ExtrudeFreezeEntry(
            vertex_id=inst.vertex_id,
            entity_class=inst.entity_class,
            origin_preop=preop[inst.vertex_id],
            copy_post=_as_coordinate(inst.vertex),
            source_face_signature=inst.source_face_signature,
        )
        for inst in sorted(copy_instances, key=lambda item: (item.vertex_id, item.source_face_signature))
    )
    if not freeze:
        return None, "the native extrude produced no classified copies"
    return (
        ExtrudeClassification(
            origins=origins,
            copies=copies,
            copy_instances=tuple(copy_instances),
            vanished_preop=vanished,
            freeze=freeze,
        ),
        None,
    )


def _classify_ncopy_instances(
    bm: bmesh.types.BMesh,
    snapshot: ExtrudeSnapshot,
    copies: dict[int, bmesh.types.BMVert],
    vanished: dict[int, Coordinate3D],
    pending_ncopy: list[tuple[int, list[bmesh.types.BMVert]]],
) -> tuple[list[ExtrudeCopyInstance] | None, str | None]:
    copy_instances = [
        ExtrudeCopyInstance(
            vertex_id=vertex_id,
            vertex=copy,
            entity_class="c" if vertex_id in vanished else "b",
            source_face_signature=(),
        )
        for vertex_id, copy in copies.items()
    ]
    if not pending_ncopy:
        return copy_instances, None

    face_layer = bm.faces.layers.int.get(layer_names.FACE_ID_LAYER)
    if face_layer is None:
        return None, "the face ID layer is missing"
    corner_map = snapshot.face_corner_map()
    seen_keys: set[tuple[int, tuple[int, ...]]] = set()
    for vertex_id, moved in pending_ncopy:
        for copy in moved:
            source_ids = _source_face_ids_for_copy(copy, face_layer, snapshot.selected_face_ids)
            if len(source_ids) != 1:
                return None, "an individual-face copy could not be attributed to a single source face"
            face_id = next(iter(source_ids))
            signature = corner_map.get(face_id)
            if signature is None:
                return None, "a source face signature is missing from the extrude snapshot"
            key = (vertex_id, signature)
            if key in seen_keys:
                return None, "duplicate (vertex, source-face) copy key"
            seen_keys.add(key)
            copy_instances.append(
                ExtrudeCopyInstance(
                    vertex_id=vertex_id,
                    vertex=copy,
                    entity_class="d",
                    source_face_signature=signature,
                )
            )
    return copy_instances, None


def _source_face_ids_for_copy(
    copy: bmesh.types.BMVert,
    face_layer,
    selected_face_ids: frozenset[int],
) -> set[int]:
    # S(copy) = incident inherited FACE_IDs ∩ snapshot selection.
    source_ids: set[int] = set()
    if not copy.is_valid:
        return source_ids
    for face in copy.link_faces:
        if not face.is_valid:
            continue
        face_id = int(face[face_layer])
        if face_id in selected_face_ids:
            source_ids.add(face_id)
    return source_ids


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
    copy_instances: list[ExtrudeCopyInstance] = []
    ncopy_by_vid: dict[int, list[ExtrudeFreezeEntry]] = {}

    for entry in freeze:
        if entry.entity_class == "d":
            ncopy_by_vid.setdefault(entry.vertex_id, []).append(entry)
            continue
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
            copy_instances.append(
                ExtrudeCopyInstance(
                    vertex_id=entry.vertex_id,
                    vertex=copy_hits[0],
                    entity_class="b",
                    source_face_signature=(),
                )
            )
            continue
        if entry.entity_class == "c":
            if len(copy_hits) != 1 or len(origin_hits) != 0 or extra:
                return None, "a frozen interior copy did not match a single copy"
            copies[entry.vertex_id] = copy_hits[0]
            vanished[entry.vertex_id] = entry.origin_preop
            copy_instances.append(
                ExtrudeCopyInstance(
                    vertex_id=entry.vertex_id,
                    vertex=copy_hits[0],
                    entity_class="c",
                    source_face_signature=(),
                )
            )
            continue
        return None, "a frozen extrude row has an unsupported class"

    ncopy_reason = _reconnect_ncopy_entries(
        bm,
        snapshot,
        groups,
        ncopy_by_vid,
        origins,
        copies,
        copy_instances,
        tol,
    )
    if ncopy_reason is not None:
        return None, ncopy_reason

    return (
        ExtrudeClassification(
            origins=origins,
            copies=copies,
            copy_instances=tuple(copy_instances),
            vanished_preop=vanished,
            freeze=freeze,
        ),
        None,
    )


def _reconnect_ncopy_entries(
    bm: bmesh.types.BMesh,
    snapshot: ExtrudeSnapshot,
    groups: dict[int, list[bmesh.types.BMVert]],
    ncopy_by_vid: dict[int, list[ExtrudeFreezeEntry]],
    origins: dict[int, bmesh.types.BMVert],
    copies: dict[int, bmesh.types.BMVert],
    copy_instances: list[ExtrudeCopyInstance],
    tol: float,
) -> str | None:
    if not ncopy_by_vid:
        return None
    # Frozen-table FACE_ID translation is a plain Undo/Redo repair concern.
    # During the first finish (including its backup re-classification), the
    # original FACE_ID attribution remains authoritative; only repair runs
    # against a re-numbered live table and uses complete face circulation.
    from . import session_state

    repair_translation = bool(session_state._HISTORY_REPAIR_BUSY)
    vertex_layer = bm.verts.layers.int.get(layer_names.VERT_SESSION_ID_LAYER) if repair_translation else None
    if repair_translation and vertex_layer is None:
        return "the session vertex ID layer is missing"
    face_layer = bm.faces.layers.int.get(layer_names.FACE_ID_LAYER) if not repair_translation else None
    if not repair_translation and face_layer is None:
        return "the face ID layer is missing"

    remainders: dict[int, list[bmesh.types.BMVert]] = {}
    for vertex_id, entries in ncopy_by_vid.items():
        instances = groups.get(vertex_id, [])
        origin_preop = entries[0].origin_preop
        origin_hits = [
            vertex for vertex in instances if matching.coordinates_match(vertex.co, origin_preop.as_tuple(), tol)
        ]
        if len(origin_hits) != 1:
            return "a frozen N-copy origin did not match a unique vertex"
        origins[vertex_id] = origin_hits[0]
        leftover = [vertex for vertex in instances if vertex is not origin_hits[0]]
        remainders[vertex_id] = leftover

    corner_map = snapshot.face_corner_map()
    for vertex_id, entries in ncopy_by_vid.items():
        leftover = list(remainders[vertex_id])
        for entry in entries:
            candidates = []
            for vertex in leftover:
                if not matching.coordinates_match(vertex.co, entry.copy_post.as_tuple(), tol):
                    continue
                if repair_translation:
                    incident_matches = [
                        face
                        for face in vertex.link_faces
                        if face.is_valid
                        and _signatures_match(
                            tuple(int(loop.vert[vertex_layer]) for loop in face.loops),
                            entry.source_face_signature,
                        )
                    ]
                    # (d) reconnect is keyed by the complete incident face
                    # circulation and frozen source attribution.  FACE_ID is
                    # a re-numberable repair marker and is intentionally
                    # ignored in this repair-only branch.
                    source_match = len(incident_matches) == 1
                else:
                    source_ids = _source_face_ids_for_copy(vertex, face_layer, snapshot.selected_face_ids)
                    source_match = len(source_ids) == 1 and _signatures_match(
                        corner_map.get(next(iter(source_ids)), ()),
                        entry.source_face_signature,
                    )
                if source_match:
                    candidates.append(vertex)
            if len(candidates) != 1:
                return "a frozen N-copy row did not match a unique copy"
            chosen = candidates[0]
            leftover.remove(chosen)
            copy_instances.append(
                ExtrudeCopyInstance(
                    vertex_id=vertex_id,
                    vertex=chosen,
                    entity_class="d",
                    source_face_signature=entry.source_face_signature,
                )
            )
        if leftover:
            return "a frozen N-copy vertex has leftover instances"
    return None


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
    copy_set = {inst.vertex for inst in classified.copy_instances if inst.vertex.is_valid}
    origin_set = set(classified.origins.values())
    new_verts = [inst.vertex for inst in classified.copy_instances]

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
        matches = _live_faces_with_verts(live_origins)
        if len(matches) != 1:
            deleted_face_ids.append(face_id)
            continue
        # A surviving face must keep its cyclic corner order and winding: a
        # count-preserving rebuild (same vertex set, rewired loops) must
        # decline rather than pass as "not deleted".
        live_cycle = tuple(int(loop.vert[vertex_layer]) for loop in matches[0].loops)
        if not _signatures_match(live_cycle, tuple(corners)):
            return None, "a surviving source face was rebuilt with different loops"

    deleted_edge_markers: list[int] = []
    for marker, (first_id, second_id) in snapshot.edge_endpoint_map().items():
        first = _origin_entity(first_id, classified, groups)
        second = _origin_entity(second_id, classified, groups)
        if first is None or second is None:
            deleted_edge_markers.append(marker)
            continue
        surviving = next(
            (edge for edge in first.link_edges if edge.is_valid and edge.other_vert(first) is second),
            None,
        )
        if surviving is None:
            deleted_edge_markers.append(marker)
            continue
        # Marker identity, not endpoint existence: a marker-0 replacement
        # edge on the same endpoints is a rebuild, not survival.
        if int(surviving[edge_layer]) != marker:
            return None, "a surviving source edge was rebuilt"

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
        return None, "the expected extrusion census is undefined"
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
    if snapshot.tool_kind == "EXTRUDE_FACES_INDIV":
        return _derive_faces_indiv_census(selected_vertices, face_corners)
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
    if snapshot.tool_kind in {"EXTRUDE_EDGES_INDIV", "EXTRUDE_VERTS_INDIV"} and region_faces:
        return None
    if snapshot.tool_kind == "EXTRUDE_VERTS_INDIV":
        # Vertices extrude individually: rails only, never a cap edge, even
        # when selected vertices are adjacent (measured: native net (2,2,0)
        # for two adjacent verts where the region formula expects (2,3,1)).
        count = len(selected_vertices)
        created = (count, count, 0)
        return created, (0, 0, 0), created

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


def _derive_faces_indiv_census(
    selected_vertices: frozenset[int],
    face_corners: dict[int, tuple[int, ...]],
) -> tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]] | None:
    region_faces = [
        corners
        for corners in face_corners.values()
        if corners and all(vertex_id in selected_vertices for vertex_id in corners)
    ]
    if not region_faces:
        return None
    created_v = 0
    created_e = 0
    created_f = 0
    for corners in region_faces:
        count = len(corners)
        created_v += count
        created_e += count + count
        created_f += 1 + count
    created = (created_v, created_e, created_f)
    deleted = (0, 0, len(region_faces))
    net = (created[0] - deleted[0], created[1] - deleted[1], created[2] - deleted[2])
    return created, deleted, net


def _vert_role(
    vertex: bmesh.types.BMVert,
    vertex_id: int,
    snapshot: ExtrudeSnapshot,
    classified: ExtrudeClassification,
) -> str:
    for inst in classified.copy_instances:
        if inst.vertex.is_valid and inst.vertex is vertex:
            return "copy"
    origin = classified.origins.get(vertex_id)
    if origin is not None and origin.is_valid and origin is vertex:
        return "origin"
    copy = classified.copies.get(vertex_id)
    if copy is not None and copy.is_valid and copy is vertex:
        return "copy"
    preop = snapshot.vertex_preop_map().get(vertex_id)
    has_ncopy = any(inst.vertex_id == vertex_id and inst.entity_class == "d" for inst in classified.copy_instances)
    if has_ncopy and preop is not None:
        # After bmesh.ops.delete, Python BMVert wrappers are not the same
        # objects as classify-time instances. (d) siblings share coordinates,
        # so only origin-vs-copy is recovered here (enough for face signatures).
        if matching.coordinates_match(vertex.co, preop.as_tuple(), snapshot.tolerance):
            return "origin"
        return "copy"
    if copy is not None and preop is not None:
        if vertex_id in classified.vanished_preop:
            return "copy"
        if not matching.coordinates_match(vertex.co, preop.as_tuple(), snapshot.tolerance):
            return "copy"
    if origin is not None and preop is not None:
        if matching.coordinates_match(vertex.co, preop.as_tuple(), snapshot.tolerance):
            return "origin"
    return "other"


def _copy_instance_of(classified: ExtrudeClassification, vertex: bmesh.types.BMVert) -> ExtrudeCopyInstance | None:
    for inst in classified.copy_instances:
        if inst.vertex is vertex:
            return inst
    return None


def _endpoint_spec(
    vertex: bmesh.types.BMVert,
    vertex_id: int,
    snapshot: ExtrudeSnapshot,
    classified: ExtrudeClassification,
) -> tuple[int, str, tuple[int, ...]] | None:
    role = _vert_role(vertex, vertex_id, snapshot, classified)
    if role not in {"origin", "copy"}:
        return None
    signature = ()
    if role == "copy":
        inst = _copy_instance_of(classified, vertex)
        if inst is not None:
            signature = inst.source_face_signature
    return (vertex_id, role, signature)


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


def _signatures_match(first: tuple, second: tuple) -> bool:
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
    """Decline when class (a), (b), or (d) origins moved or disappeared."""

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

    boundary_ids = [entry.vertex_id for entry in classified.freeze if entry.entity_class in {"b", "d"}] or list(
        classified.origins
    )
    copies_by_vid: dict[int, list[bmesh.types.BMVert]] = {}
    for inst in classified.copy_instances:
        copies_by_vid.setdefault(inst.vertex_id, []).append(inst.vertex)
    seen_boundary: set[int] = set()
    for vertex_id in boundary_ids:
        if vertex_id in seen_boundary:
            continue
        seen_boundary.add(vertex_id)
        origin = classified.origins.get(vertex_id)
        if origin is None or not origin.is_valid:
            return "a classified origin vertex is missing"
        pre = preop.get(vertex_id)
        if pre is None or not matching.coordinates_match(origin.co, pre.as_tuple(), tolerance):
            return "a classified origin vertex moved during the native extrude"
        known = {origin, *copies_by_vid.get(vertex_id, ())}
        extras = [vertex for vertex in groups.get(vertex_id, []) if vertex.is_valid and vertex not in known]
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
    expected_created: tuple[int, int, int]
    expected_deleted: tuple[int, int, int]
    mirror_plan: MirrorActionPlan


@dataclass(frozen=True)
class OriginRef:
    """Stable-key endpoint referring to an origin vertex."""

    vertex_id: int


@dataclass(frozen=True)
class CopyKey:
    """Stable-key endpoint referring to one native copy instance."""

    vertex_id: int
    source_face_signature: tuple[int, ...]


@dataclass
class _CopyPlanEntry:
    key: CopyKey
    coordinate: tuple[float, float, float]
    reuse: bool


@dataclass
class _ElementPlanEntry:
    """A stable-key edge/face action and its apply-local identity witness."""

    endpoints: tuple[OriginRef | CopyKey, ...]
    self_witness: bool
    witness_index: int | None = None
    face_signature: tuple[tuple[int, str], ...] = ()
    material_index: int = 0
    smooth: bool = False


@dataclass
class _DeletePlanEntry:
    domain: str
    element_id: int
    partner: int
    self_partner: bool


@dataclass
class _MirrorRuntime:
    """BMesh identity cache paired with one apply-scoped stable-key plan."""

    copy_sources: dict[CopyKey, bmesh.types.BMVert]
    copy_mirrors: dict[CopyKey, bmesh.types.BMVert]
    element_sources: dict[int, bmesh.types.BMEdge | bmesh.types.BMFace]
    witness_tokens: dict[int, int]
    created: dict[int, bmesh.types.BMEdge | bmesh.types.BMFace]
    delete_origins: dict[int, object | tuple[object, ...] | None]
    delete_live: dict[int, object]
    delete_tokens: dict[int, int]
    # These are stable rehydration records.  Wrapper caches are rebuilt after
    # APPLY-layer stamping because layer remove/new invalidates pre-stamp wrappers.
    copy_source_coords: dict[CopyKey, tuple[float, float, float]]
    delete_origin_keys: dict[int, tuple[int, ...] | None]


@dataclass
class MirrorActionPlan:
    """Apply-scoped stable-key plan; BMesh references are only local caches."""

    copies: dict[CopyKey, _CopyPlanEntry]
    edges: tuple[_ElementPlanEntry, ...]
    faces: tuple[_ElementPlanEntry, ...]
    deletes: tuple[_DeletePlanEntry, ...]

    @property
    def self_witness_edges(self) -> int:
        return sum(entry.self_witness for entry in self.edges)

    @property
    def self_witness_faces(self) -> int:
        return sum(entry.self_witness for entry in self.faces)

    @property
    def expected_created(self) -> tuple[int, int, int]:
        return (
            sum(not entry.reuse for entry in self.copies.values()),
            len(self.edges) - self.self_witness_edges,
            len(self.faces) - self.self_witness_faces,
        )

    @property
    def expected_deleted(self) -> tuple[int, int, int]:
        def count(domain: str) -> int:
            return sum(entry.domain == domain and not entry.self_partner for entry in self.deletes)

        return (count("VERT"), count("EDGE"), count("FACE"))


def _self_delete_consumed(bm: bmesh.types.BMesh, entry: _DeletePlanEntry, runtime: _MirrorRuntime) -> bool:
    """Check native consumption using origin entities, never session-ID liveness."""

    origin_value = runtime.delete_origins.get(id(entry))
    if entry.domain == "VERT":
        origin = origin_value
        return (
            origin is None or not getattr(origin, "is_valid", False) or not any(vertex is origin for vertex in bm.verts)
        )
    if entry.domain == "EDGE":
        origins = origin_value
        if not isinstance(origins, tuple) or len(origins) != 2 or any(origin is None for origin in origins):
            return True
        first, second = origins
        if not getattr(first, "is_valid", False) or not getattr(second, "is_valid", False):
            return True
        return _edge_between(first, second) is None  # ty: ignore[invalid-argument-type]  # dynamically guarded; see runtime checks above
    origins = origin_value
    if not isinstance(origins, tuple) or not origins or any(origin is None for origin in origins):
        return True
    if not all(getattr(origin, "is_valid", False) for origin in origins):
        return True
    wanted = tuple(origins)
    return not any(
        face.is_valid
        and len(face.loops) == len(wanted)
        and _cycle_identity(tuple(loop.vert for loop in face.loops), wanted)  # ty: ignore[invalid-argument-type]  # dynamically guarded; see runtime checks above
        for face in bm.faces
    )


def _edge_between(first: bmesh.types.BMVert, second: bmesh.types.BMVert) -> bmesh.types.BMEdge | None:
    return next((edge for edge in first.link_edges if edge.is_valid and edge.other_vert(first) is second), None)


def _edge_identity(edge: bmesh.types.BMEdge, expected: tuple[bmesh.types.BMVert, ...]) -> bool:
    return len(expected) == 2 and (
        (edge.verts[0] is expected[0] and edge.verts[1] is expected[1])
        or (edge.verts[0] is expected[1] and edge.verts[1] is expected[0])
    )


def _cycle_identity(actual: tuple[bmesh.types.BMVert, ...], expected: tuple[bmesh.types.BMVert, ...]) -> bool:
    """Compare a face cycle by wrapper identity, allowing rotation only."""

    if len(actual) != len(expected):
        return False
    if not expected:
        return not actual
    return any(
        all(actual[index] is expected[(index + offset) % len(expected)] for index in range(len(expected)))
        for offset in range(len(expected))
    )


def apply_mirror(
    bm: bmesh.types.BMesh,
    snapshot: ExtrudeSnapshot,
    classified: ExtrudeClassification,
    description: ExtrudeSourceDescription,
) -> tuple[dict[tuple[int, tuple[int, ...]], bmesh.types.BMVert], str | None, ExtrudeApplyAudit | None]:
    """Apply an apply-scoped stable-key plan in the contract's fixed order."""

    vertex_layer = bm.verts.layers.int.get(layer_names.VERT_SESSION_ID_LAYER)
    edge_layer = bm.edges.layers.int.get(layer_names.EDGE_ORIGINAL_LAYER)
    face_layer = bm.faces.layers.int.get(layer_names.FACE_ID_LAYER)
    if vertex_layer is None or edge_layer is None or face_layer is None:
        return {}, "temporary topology markers are missing", None

    vertex_pairs = snapshot.vertex_pair_map()
    edge_pairs = snapshot.edge_pair_map()
    face_pairs = snapshot.face_pair_map()
    preop = snapshot.vertex_preop_map()
    snapshot_face_corners = snapshot.face_corner_map()
    snapshot_edge_endpoints = snapshot.edge_endpoint_map()

    # 1. Build the stable-key plan before stamping or invalidating wrappers.
    copy_entries: dict[CopyKey, _CopyPlanEntry] = {}
    runtime = _MirrorRuntime({}, {}, {}, {}, {}, {}, {}, {}, {}, {})
    for instance in classified.copy_instances:
        if not instance.vertex.is_valid:
            return {}, "a source copy was lost before mirroring", None
        key = CopyKey(instance.vertex_id, instance.source_face_signature)
        if key in copy_entries:
            return {}, "the source copy key is ambiguous", None
        coordinate = (float(instance.vertex.co.x), float(instance.vertex.co.y), float(instance.vertex.co.z))
        copy_entries[key] = _CopyPlanEntry(
            key=key,
            coordinate=coordinate,
            reuse=abs(coordinate[snapshot.axis_index]) <= snapshot.tolerance,
        )
        runtime.copy_sources[key] = instance.vertex
        runtime.copy_source_coords[key] = coordinate

    def _stable_spec(vertex: bmesh.types.BMVert) -> OriginRef | CopyKey | None:
        vertex_id = int(vertex[vertex_layer])
        role = _vert_role(vertex, vertex_id, snapshot, classified)
        if role == "origin":
            return OriginRef(vertex_id)
        if role == "copy":
            instance = _copy_instance_of(classified, vertex)
            if instance is None:
                return None
            return CopyKey(vertex_id, instance.source_face_signature)
        return None

    def _pre_mapped(spec: OriginRef | CopyKey) -> bmesh.types.BMVert | None:
        if isinstance(spec, CopyKey):
            entry = copy_entries.get(spec)
            return runtime.copy_sources.get(spec) if entry is not None and entry.reuse else None
        partner = vertex_pairs.get(spec.vertex_id)
        if partner is None:
            return None
        return classified.origins.get(partner)

    bm.edges.index_update()
    bm.faces.index_update()
    edge_entries: list[_ElementPlanEntry] = []
    for edge in description.new_edges:
        if not edge.is_valid or len(edge.verts) != 2:
            return {}, "a source new edge was lost before mirroring", None
        specs = tuple(_stable_spec(vertex) for vertex in edge.verts)
        if any(spec is None for spec in specs):
            return {}, "a new edge endpoint is not a classified origin or copy", None
        mapped = tuple(_pre_mapped(spec) for spec in specs if spec is not None)
        witness = (
            len(mapped) == 2 and all(vertex is not None for vertex in mapped) and _edge_identity(edge, tuple(mapped))  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]  # dynamically guarded; see runtime checks above
        )
        edge_entry = _ElementPlanEntry(
            endpoints=specs,  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]  # dynamically guarded; see runtime checks above
            self_witness=witness,
            witness_index=int(edge.index) if witness else None,
        )
        edge_entries.append(edge_entry)
        runtime.element_sources[id(edge_entry)] = edge

    if len(description.face_signatures) != len(description.new_faces):
        return {}, "new-face signatures are not aligned with the source faces", None
    face_entries: list[_ElementPlanEntry] = []
    for face, stored_signature in zip(description.new_faces, description.face_signatures, strict=True):
        if not face.is_valid:
            return {}, "a source new face was lost before mirroring", None
        specs = tuple(_stable_spec(loop.vert) for loop in face.loops)
        if any(spec is None for spec in specs):
            return {}, "a new face corner is not a classified origin or copy", None
        live_signature = tuple(
            (spec.vertex_id, "origin") if isinstance(spec, OriginRef) else (spec.vertex_id, "copy")
            for spec in specs
            if spec is not None
        )
        if not _signatures_match(live_signature, stored_signature):
            return {}, "a source new face no longer matches its stored signature", None
        mapped = tuple(_pre_mapped(spec) for spec in specs if spec is not None)
        witness = (
            len(mapped) == len(face.verts)
            and all(vertex is not None for vertex in mapped)
            and _cycle_identity(
                tuple(loop.vert for loop in face.loops),
                tuple(mapped),  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]  # dynamically guarded; see runtime checks above
            )
        )
        face_entry = _ElementPlanEntry(
            endpoints=specs,  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]  # dynamically guarded; see runtime checks above
            self_witness=witness,
            witness_index=int(face.index) if witness else None,
            face_signature=stored_signature,
            material_index=int(face.material_index),
            smooth=bool(face.smooth),
        )
        face_entries.append(face_entry)
        runtime.element_sources[id(face_entry)] = face

    delete_entries: list[_DeletePlanEntry] = []
    for face_id in description.deleted_face_ids:
        partner = face_pairs.get(face_id)
        if partner is None:
            continue
        corners = snapshot.face_corner_map().get(face_id, ())
        delete_entry = _DeletePlanEntry("FACE", face_id, partner, partner == face_id)
        delete_entries.append(delete_entry)
        runtime.delete_origin_keys[id(delete_entry)] = tuple(corners)
    for marker in description.deleted_edge_markers:
        partner = edge_pairs.get(marker)
        if partner is None:
            continue
        first, second = snapshot.edge_endpoint_map().get(marker, (None, None))
        delete_entry = _DeletePlanEntry("EDGE", marker, partner, partner == marker)
        delete_entries.append(delete_entry)
        runtime.delete_origin_keys[id(delete_entry)] = (
            (first, second) if first is not None and second is not None else None
        )
    for vertex_id in description.deleted_vertex_ids:
        partner = vertex_pairs.get(vertex_id)
        if partner is None:
            continue
        delete_entry = _DeletePlanEntry("VERT", vertex_id, partner, partner == vertex_id)
        delete_entries.append(delete_entry)
        runtime.delete_origin_keys[id(delete_entry)] = (vertex_id,)
    plan = MirrorActionPlan(copy_entries, tuple(edge_entries), tuple(face_entries), tuple(delete_entries))

    # 2. Stamp; 3. resolve only non-self deletion entries against live topology.
    token_layers = _stamp_apply_tokens(bm)
    # Rebuild all lookup tables after layer creation before touching any
    # apply-local source/witness cache.  Only stable keys cross this point.
    for sequence in (bm.verts, bm.edges, bm.faces):
        sequence.ensure_lookup_table()
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

    def _origin_live(vertex_id: int) -> bmesh.types.BMVert | None:
        expected = preop.get(vertex_id)
        if expected is None:
            return None
        hits = [
            vertex
            for vertex in verts_by_id.get(vertex_id, ())
            if vertex.is_valid
            and not any(vertex is copy for copy in runtime.copy_sources.values())
            and matching.coordinates_match(vertex.co, expected.as_tuple(), snapshot.tolerance)
        ]
        return hits[0] if len(hits) == 1 else None

    def _origin_face_candidate(face: bmesh.types.BMFace, partner: int) -> bool:
        expected_corners = snapshot_face_corners.get(partner)
        if expected_corners is None:
            return False
        live_corners = tuple(int(loop.vert[vertex_layer]) for loop in face.loops)
        if not _signatures_match(live_corners, expected_corners):
            return False
        return all(_origin_live(int(loop.vert[vertex_layer])) is loop.vert for loop in face.loops)

    def _origin_edge_candidate(edge: bmesh.types.BMEdge, partner: int) -> bool:
        expected_endpoints = snapshot_edge_endpoints.get(partner)
        if expected_endpoints is None:
            return False
        live_endpoints = tuple(int(vertex[vertex_layer]) for vertex in edge.verts)
        if live_endpoints not in (expected_endpoints, expected_endpoints[::-1]):
            return False
        return all(_origin_live(int(vertex[vertex_layer])) is vertex for vertex in edge.verts)

    def _origin_vert_candidate(vertex: bmesh.types.BMVert) -> bool:
        return _origin_live(int(vertex[vertex_layer])) is vertex

    def _copy_live(key: CopyKey) -> bmesh.types.BMVert | None:
        expected = runtime.copy_source_coords.get(key)
        if expected is None:
            return None
        candidates = [
            vertex
            for vertex in verts_by_id.get(key.vertex_id, ())
            if vertex.is_valid and matching.coordinates_match(vertex.co, expected, snapshot.tolerance)
        ]
        if key.source_face_signature:
            if snapshot.route == "GIZMO_ADOPTED":
                # Gizmo caps have FACE_ID == 0 and the adopted snapshot uses
                # synthetic (negative) face IDs.  Their stable attribution is
                # the copied cap's vertex-ID cycle, not FACE_ID membership.
                candidates = [
                    vertex
                    for vertex in candidates
                    if any(
                        (live_signature := tuple(int(loop.vert[vertex_layer]) for loop in face.loops))
                        and _signatures_match(live_signature, key.source_face_signature)
                        and all(
                            CopyKey(int(loop.vert[vertex_layer]), key.source_face_signature) in plan.copies
                            for loop in face.loops
                        )
                        for face in vertex.link_faces
                        if face.is_valid
                    )
                ]
            else:
                face_layer_live = bm.faces.layers.int.get(layer_names.FACE_ID_LAYER)
                if face_layer_live is None:
                    return None
                candidates = [
                    vertex
                    for vertex in candidates
                    if any(
                        int(face[face_layer_live]) in snapshot.selected_face_ids
                        and _signatures_match(
                            tuple(int(loop.vert[vertex_layer]) for loop in face.loops),
                            key.source_face_signature,
                        )
                        for face in vertex.link_faces
                        if face.is_valid
                    )
                ]
        return candidates[0] if len(candidates) == 1 else None

    def _source_of_spec(spec: OriginRef | CopyKey) -> bmesh.types.BMVert | None:
        return _origin_live(spec.vertex_id) if isinstance(spec, OriginRef) else runtime.copy_sources.get(spec)

    def _source_face_for_entry(entry: _ElementPlanEntry) -> bmesh.types.BMFace | None:
        corners = tuple(_source_of_spec(spec) for spec in entry.endpoints)
        if any(vertex is None for vertex in corners):
            return None
        expected = tuple(corners)  # type: ignore[arg-type]
        matches = [
            face
            for face in bm.faces
            if face.is_valid and _cycle_identity(tuple(loop.vert for loop in face.loops), expected)  # ty: ignore[invalid-argument-type]  # dynamically guarded; see runtime checks above
        ]
        return matches[0] if len(matches) == 1 else None

    def _rehydrate_after_topology_change(resolve_delete_targets: bool) -> str | None:
        """Rebuild every wrapper cache after layer creation or native delete."""

        nonlocal vertex_layer, edge_layer, face_layer, verts_by_id, faces_by_id, edges_by_marker
        for sequence in (bm.verts, bm.edges, bm.faces):
            sequence.ensure_lookup_table()
        vertex_layer = bm.verts.layers.int.get(layer_names.VERT_SESSION_ID_LAYER)
        edge_layer = bm.edges.layers.int.get(layer_names.EDGE_ORIGINAL_LAYER)
        face_layer = bm.faces.layers.int.get(layer_names.FACE_ID_LAYER)
        if vertex_layer is None or edge_layer is None or face_layer is None:
            return "temporary topology markers were lost while rehydrating apply state"
        verts_by_id = _verts_by_session_id(bm, vertex_layer)
        faces_by_id = {}
        for face in bm.faces:
            faces_by_id.setdefault(int(face[face_layer]), []).append(face)
        edges_by_marker = {}
        for edge in bm.edges:
            edges_by_marker.setdefault(int(edge[edge_layer]), []).append(edge)
        runtime.copy_sources.clear()
        runtime.element_sources.clear()
        runtime.delete_origins.clear()
        runtime.delete_live.clear()
        for key in plan.copies:
            source = _copy_live(key)
            if source is None or any(source is other for other in runtime.copy_sources.values()):
                return "a source copy key was missing or ambiguous after topology change"
            runtime.copy_sources[key] = source
        for entry in plan.deletes:
            keys = runtime.delete_origin_keys.get(id(entry))
            if keys is None:
                runtime.delete_origins[id(entry)] = None
            else:
                resolved = tuple(_origin_live(vertex_id) for vertex_id in keys)
                runtime.delete_origins[id(entry)] = resolved[0] if entry.domain == "VERT" else resolved
        for entry in plan.edges:
            corners = tuple(_source_of_spec(spec) for spec in entry.endpoints)
            if any(vertex is None for vertex in corners):
                return "a source edge could not be rehydrated after topology change"
            source = _edge_between(corners[0], corners[1])  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]  # dynamically guarded; see runtime checks above
            if source is None:
                return "a source edge could not be rehydrated after topology change"
            runtime.element_sources[id(entry)] = source
        for entry in plan.faces:
            source = _source_face_for_entry(entry)
            if source is None:
                return "a source face could not be rehydrated after topology change"
            runtime.element_sources[id(entry)] = source
        if resolve_delete_targets:
            apply_layers = {
                "FACE": bm.faces.layers.int.get(layer_names.FACE_APPLY_ID_LAYER),
                "EDGE": bm.edges.layers.int.get(layer_names.EDGE_APPLY_ID_LAYER),
                "VERT": bm.verts.layers.int.get(layer_names.VERT_APPLY_ID_LAYER),
            }
            if any(layer is None for layer in apply_layers.values()):
                return "apply identity tokens are missing while resolving delete targets"
            for entries, sequence, domain in (
                (plan.edges, bm.edges, "EDGE"),
                (plan.faces, bm.faces, "FACE"),
            ):
                for entry in entries:
                    if not entry.self_witness:
                        continue
                    source = runtime.element_sources.get(id(entry))
                    witness_index = entry.witness_index
                    if (
                        source is None
                        or witness_index is None
                        or witness_index < 0
                        or witness_index >= len(sequence)
                        or source.index != witness_index
                    ):
                        return "a self-witness does not identify the source element"
                    runtime.witness_tokens[id(entry)] = int(sequence[witness_index][apply_layers[domain]])
            for entry in plan.deletes:
                if entry.self_partner:
                    continue
                table = {"FACE": faces_by_id, "EDGE": edges_by_marker, "VERT": verts_by_id}[entry.domain]
                matches = [element for element in table.get(entry.partner, ()) if element.is_valid]
                if len(matches) > 1:
                    if entry.domain == "FACE":
                        matches = [element for element in matches if _origin_face_candidate(element, entry.partner)]  # ty: ignore[invalid-argument-type]  # dynamically guarded; see runtime checks above
                    elif entry.domain == "EDGE":
                        matches = [element for element in matches if _origin_edge_candidate(element, entry.partner)]  # ty: ignore[invalid-argument-type]  # dynamically guarded; see runtime checks above
                    else:
                        matches = [element for element in matches if _origin_vert_candidate(element)]
                if len(matches) != 1:
                    return f"a deleted {entry.domain.lower()} has an ambiguous mirrored counterpart"
                runtime.delete_live[id(entry)] = matches[0]
                runtime.delete_tokens[id(entry)] = int(matches[0][apply_layers[entry.domain]])
        return None

    # Rehydrate once after stamping and again immediately after native delete;
    # bmesh.ops.delete invalidates wrappers and lookup tables a second time.
    reason = _rehydrate_after_topology_change(True)
    if reason is not None:
        return {}, reason, None

    delete_faces = [
        runtime.delete_live[id(entry)] for entry in plan.deletes if entry.domain == "FACE" and not entry.self_partner
    ]
    delete_edges = [
        runtime.delete_live[id(entry)] for entry in plan.deletes if entry.domain == "EDGE" and not entry.self_partner
    ]
    delete_verts = [
        runtime.delete_live[id(entry)] for entry in plan.deletes if entry.domain == "VERT" and not entry.self_partner
    ]
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

    # 4. Native deletion order. Self entries are deliberate no-ops.
    if delete_faces:
        bmesh.ops.delete(bm, geom=delete_faces, context="FACES_ONLY")  # ty: ignore[invalid-argument-type]  # dynamically guarded; see runtime checks above
    if delete_edges:
        still = [edge for edge in delete_edges if edge.is_valid]  # ty: ignore[unresolved-attribute]  # dynamically guarded; see runtime checks above
        if still:
            bmesh.ops.delete(bm, geom=still, context="EDGES")  # ty: ignore[invalid-argument-type]  # dynamically guarded; see runtime checks above
    if delete_verts:
        still = [vertex for vertex in delete_verts if vertex.is_valid]  # ty: ignore[unresolved-attribute]  # dynamically guarded; see runtime checks above
        if still:
            bmesh.ops.delete(bm, geom=still, context="VERTS")  # ty: ignore[invalid-argument-type]  # dynamically guarded; see runtime checks above

    reason = _rehydrate_after_topology_change(False)
    if reason is not None:
        return {}, reason, None

    # 5. Build copies. A copy that landed on the plane reuses its source object.
    mirror_copies: dict[tuple[int, tuple[int, ...]], bmesh.types.BMVert] = {}
    for key, entry in plan.copies.items():
        if entry.reuse:
            source = runtime.copy_sources[key]
            if not source.is_valid:
                return {}, "a reusable source copy was lost during deletion", None
            mirror = source
        else:
            mirror = bm.verts.new(matching.mirror_coordinate(Vector(entry.coordinate), snapshot.axis_index))
            mirror.select = False
        runtime.copy_mirrors[key] = mirror
        mirror_copies[(key.vertex_id, key.source_face_signature)] = mirror

    vertex_layer = bm.verts.layers.int.get(layer_names.VERT_SESSION_ID_LAYER)
    if vertex_layer is None:
        return {}, "the session vertex ID layer was lost while creating copies", None
    verts_by_id = _verts_by_session_id(bm, vertex_layer)

    def _mirror_of_spec(spec: OriginRef | CopyKey) -> bmesh.types.BMVert | None:
        if isinstance(spec, CopyKey):
            return runtime.copy_mirrors.get(spec)
        partner = vertex_pairs.get(spec.vertex_id)
        return None if partner is None else _origin_live(partner)

    # 6. Build elements, skipping only plan-proven self images.
    for entry in plan.edges:
        if entry.self_witness:
            continue
        endpoints = tuple(_mirror_of_spec(spec) for spec in entry.endpoints)
        if any(vertex is None for vertex in endpoints):
            return {}, "a mirrored new edge is missing an endpoint", None
        first, second = endpoints
        if first is second:
            return {}, "a mirrored new edge would be degenerate", None
        if _edge_between(first, second) is not None:  # ty: ignore[invalid-argument-type]  # dynamically guarded; see runtime checks above
            return {}, "a mirrored new edge already exists", None
        try:
            runtime.created[id(entry)] = bm.edges.new((first, second))  # ty: ignore[invalid-argument-type]  # dynamically guarded; see runtime checks above
        except ValueError:
            return {}, "a mirrored new edge already exists", None
    for entry in plan.faces:
        if entry.self_witness:
            continue
        mirror_corners = tuple(_mirror_of_spec(spec) for spec in entry.endpoints)
        if any(vertex is None for vertex in mirror_corners):
            return {}, "a mirrored new face is missing a corner", None
        if len(set(mirror_corners)) != len(mirror_corners):
            return {}, "the mirrored extrude face would be degenerate", None
        vertex_layer = bm.verts.layers.int.get(layer_names.VERT_SESSION_ID_LAYER)
        source_face, locate_reason = _locate_face_by_signature(
            bm, entry.face_signature, vertex_layer, snapshot, classified
        )
        if source_face is None:
            return {}, locate_reason or "a source new face could not be relocated after mirroring", None
        try:
            runtime.created[id(entry)] = bm.faces.new(list(reversed(mirror_corners)))  # ty: ignore[invalid-argument-type]  # dynamically guarded; see runtime checks above
        except ValueError:
            return {}, "the mirrored extrude face already exists", None
        created_face = runtime.created[id(entry)]
        created_face.material_index = entry.material_index  # ty: ignore[invalid-assignment]  # dynamically guarded; see runtime checks above
        created_face.smooth = entry.smooth
        created_face.select = False
        for mirror_loop, source_loop in zip(created_face.loops, reversed(tuple(source_face.loops)), strict=True):  # ty: ignore[unresolved-attribute]  # dynamically guarded; see runtime checks above
            mirror_loop.copy_from(source_loop)
        created_face.normal_update()

    # 7. Element postconditions, self-consumption predicates, then census.
    for entry in (*plan.edges, *plan.faces):
        if entry.self_witness:
            source = runtime.element_sources.get(id(entry))
            witness_token = runtime.witness_tokens.get(id(entry))
            sequence = bm.edges if isinstance(source, bmesh.types.BMEdge) else bm.faces
            apply_layer = sequence.layers.int.get(
                layer_names.EDGE_APPLY_ID_LAYER
                if isinstance(source, bmesh.types.BMEdge)
                else layer_names.FACE_APPLY_ID_LAYER
            )
            if (
                source is None
                or not source.is_valid
                or witness_token is None
                or apply_layer is None
                or source not in sequence
                or int(source[apply_layer]) != witness_token
                or id(entry) in runtime.created
            ):
                return {}, "a self-witness does not identify the source element", None
            mapped = tuple(_mirror_of_spec(spec) for spec in entry.endpoints)
            if any(vertex is None for vertex in mapped):
                return {}, "a self-witness has no live mirrored image", None
            if isinstance(source, bmesh.types.BMEdge):
                if not _edge_identity(source, tuple(mapped)):  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]  # dynamically guarded; see runtime checks above
                    return {}, "a self-witness does not match its mirrored image", None
            elif not (
                _cycle_identity(tuple(loop.vert for loop in source.loops), tuple(mapped))  # ty: ignore[invalid-argument-type]  # dynamically guarded; see runtime checks above
                or _cycle_identity(tuple(loop.vert for loop in source.loops), tuple(reversed(mapped)))  # ty: ignore[invalid-argument-type]  # dynamically guarded; see runtime checks above
            ):
                return {}, "a self-witness does not match its mirrored image", None
            continue
        if id(entry) in runtime.witness_tokens:
            return {}, "a non-self mirror entry unexpectedly has a self-witness", None
        created = runtime.created.get(id(entry))
        if created is None or not created.is_valid:
            return {}, "a mirrored element was not created exactly once", None
        mapped = tuple(_mirror_of_spec(spec) for spec in entry.endpoints)
        if any(vertex is None for vertex in mapped):
            return {}, "a mirrored element does not match its planned image", None
        if isinstance(created, bmesh.types.BMEdge):
            if not _edge_identity(created, tuple(mapped)):  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]  # dynamically guarded; see runtime checks above
                return {}, "a mirrored element does not match its planned image", None
            edge_apply = bm.edges.layers.int.get(layer_names.EDGE_APPLY_ID_LAYER)
            if edge_apply is None or int(created[edge_apply]) != 0:
                return {}, "a created mirrored edge retained an apply token", None
        else:
            actual = tuple(loop.vert for loop in created.loops)
            expected_cycle = tuple(reversed(tuple(mapped)))
            if not _cycle_identity(actual, expected_cycle):  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]  # dynamically guarded; see runtime checks above
                return {}, "a mirrored element does not match its planned image", None
            face_apply = bm.faces.layers.int.get(layer_names.FACE_APPLY_ID_LAYER)
            if face_apply is None or int(created[face_apply]) != 0:
                return {}, "a created mirrored face retained an apply token", None
    for entry in plan.deletes:
        if not entry.self_partner:
            token = runtime.delete_tokens.get(id(entry))
            layer_name = {
                "FACE": layer_names.FACE_APPLY_ID_LAYER,
                "EDGE": layer_names.EDGE_APPLY_ID_LAYER,
                "VERT": layer_names.VERT_APPLY_ID_LAYER,
            }.get(entry.domain)
            sequence = {"FACE": bm.faces, "EDGE": bm.edges, "VERT": bm.verts}.get(entry.domain)
            if token is None or layer_name is None or sequence is None:
                return {}, "a non-self deletion target has no recorded apply token", None
            layer = sequence.layers.int.get(layer_name)
            if layer is None:
                return {}, "apply identity tokens are missing for a deletion postcondition", None
            if any(element.is_valid and int(element[layer]) == token for element in sequence):
                return {}, "a non-self deletion target survived mirroring", None
        if entry.self_partner and not _self_delete_consumed(bm, entry, runtime):
            return {}, "a self deletion target was not consumed by the native extrude", None

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
        expected_created=plan.expected_created,
        expected_deleted=plan.expected_deleted,
        mirror_plan=plan,
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
    mirror_copies: dict[tuple[int, tuple[int, ...]], bmesh.types.BMVert],
    source_copy_coords: dict[tuple[int, tuple[int, ...]], tuple[float, float, float]],
    source_origin_coords: dict[int, tuple[float, float, float]],
    audit: ExtrudeApplyAudit,
) -> str | None:
    """Verify mirror census, reflection, and source-side stationarity."""

    if audit.created != audit.expected_created or audit.deleted != audit.expected_deleted:
        return (
            f"mirror census {audit.created}/{audit.deleted} does not match plan "
            f"{audit.expected_created}/{audit.expected_deleted}"
        )
    audit_net = (
        audit.created[0] - audit.deleted[0],
        audit.created[1] - audit.deleted[1],
        audit.created[2] - audit.deleted[2],
    )
    expected_net = tuple(audit.expected_created[index] - audit.expected_deleted[index] for index in range(3))
    if audit_net != expected_net:
        return f"mirror net {audit_net} does not match plan {expected_net}"

    expected_copy_keys = {(key.vertex_id, key.source_face_signature) for key in audit.mirror_plan.copies}
    if set(mirror_copies) != expected_copy_keys:
        return "mirrored copy keys do not match the plan"
    vertex_layer = bm.verts.layers.int.get(layer_names.VERT_SESSION_ID_LAYER)
    if vertex_layer is None:
        return "the session vertex ID layer is missing"
    live_groups = _verts_by_session_id(bm, vertex_layer)
    for copy_key, expected in source_copy_coords.items():
        vertex_id, _signature = copy_key
        # N copies of one vid can share coordinates (coplanar FACES_INDIV).
        # Require the source location still exists; do not demand uniqueness.
        hits = [
            vertex
            for vertex in live_groups.get(vertex_id, ())
            if vertex.is_valid and matching.coordinates_match(vertex.co, expected, snapshot.tolerance)
        ]
        if not hits:
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

    for copy_key, mirror in mirror_copies.items():
        if not mirror.is_valid:
            return "a mirrored copy vertex is invalid"
        if not all(math.isfinite(float(mirror.co[index])) for index in range(3)):
            return "a mirrored copy coordinate is not finite"
        expected_source = source_copy_coords.get(copy_key)
        if expected_source is None:
            return "a mirrored copy has no recorded source coordinate"
        plan_key = CopyKey(copy_key[0], copy_key[1])
        plan_entry = audit.mirror_plan.copies.get(plan_key)
        if plan_entry is None:
            return "a mirrored copy has no plan entry"
        if plan_entry.reuse:
            if abs(float(mirror.co[snapshot.axis_index])) > snapshot.tolerance:
                return "a reusable mirrored copy is not on the mirror plane"
        else:
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
    if window is not None:
        known = set(session.preexisting_modal_operators)
        newcomers = []
        for operator in window.modal_operators:
            idname = getattr(operator, "bl_idname", "") or getattr(getattr(operator, "bl_rna", None), "identifier", "")
            if (operator.as_pointer(), str(idname)) not in known:
                newcomers.append(str(idname))
        if newcomers:
            names = ", ".join(sorted(set(newcomers)))
            return f"another modal operator started before the extrude could be mirrored ({names})"
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
