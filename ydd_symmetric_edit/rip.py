# SPDX-License-Identifier: GPL-3.0-or-later

"""Symmetric postprocess for Blender's native Rip (``mesh.rip_move``).

R0 measurements (identical on Blender 4.2.23 / 5.2.0):

- ``MESH_OT_rip`` copies integer CustomData onto duplicated vertices and
  edges, so a unique per-vertex ID layer written before the native operator
  identifies every duplicate pair afterwards.  The pre-existing unique edge
  IDs in ``EDGE_ORIGINAL_LAYER`` identify duplicated (seam) edges the same
  way.
- Every observed Rip result equals ``bmesh.ops.split_edges`` over the
  duplicated edge set: vertices split purely by connectivity, interior path
  neighbours stay pinched, boundary endpoints separate.
- Rip Fill bridges the seam with quads whose corners repeat a duplicated
  vertex ID; the fill faces copy FACE_ID CustomData from neighbours, so fill
  detection uses the repeated-corner rule, never FACE_ID values.

The mirror application therefore reflects the *observed* seam onto the paired
edges and replays ``split_edges`` there, then assigns mirrored final
coordinates to the matching copies.

Axis-crossing measurements:

- Selection overlap is no longer a prepare-time passthrough; the post-native
  seam is the sole criterion.  A fully self-mirrored seam (seam edge set
  closed under endpoint mirror: S = ρ(S)) takes the V-opening path: source
  bank = native moved/selected bank, non-source bank coordinates replaced by
  ρ of the mirror-role source vertex.  Partial self-overlap (S ∩ ρ(S) nonempty
  but S ≠ ρ(S)) declines with a visible WARNING.  Bank identity also requires
  selection/movement agreement and seam-wide face-ID side consistency.
"""

from __future__ import annotations

from dataclasses import dataclass

import bmesh
from mathutils import Vector

from . import core
from ._types import (
    Coordinate3D,
    FaceId,
    MirrorFaceMap,
    RipDupSignature,
    RipSignature,
    RipSnapshot,
    RipVertexRecord,
)

_MOVE_EPSILON = 1.0e-6


# ---------------------------------------------------------------------------
# prepare-time guards and snapshot


def prepare_guard_reason(context, bm: bmesh.types.BMesh, axis_index: int, tolerance: float) -> tuple[str, str] | None:
    """(report level, reason) to skip the session and pass through to native Rip."""

    tool_settings = context.tool_settings
    if bool(getattr(tool_settings, "use_proportional_edit", False)):
        return "WARNING", "Proportional Editing is not mirrored"
    if bool(getattr(tool_settings, "use_mesh_automerge", False)):
        return "WARNING", "Auto Merge is not mirrored"
    if tuple(tool_settings.mesh_select_mode) == (False, False, True):
        return "INFO", "Face selection mode cannot rip"

    selected = [vertex for vertex in bm.verts if vertex.select and not vertex.hide]
    if not selected:
        return "INFO", "Nothing selected to rip"

    # This on-plane guard stays ahead of any seam classification: Rip does
    # not support on-plane selections at all.
    on_plane = sum(1 for vertex in selected if abs(vertex.co[axis_index]) <= tolerance)
    if on_plane:
        return "WARNING", f"{on_plane} selected vertex(es) lie on the mirror plane"

    # Selection/mirror overlap is deliberately not declined here.  The
    # session continues; post-native seam analysis decides
    # between the external split path, the self-mirrored V-opening path, or a
    # visible WARNING decline (partial self-overlap / missing counterparts).
    del axis_index, tolerance
    return None


def build_snapshot(
    bm: bmesh.types.BMesh,
    axis_index: int,
    tolerance: float,
    *,
    lookup: core.VertexMirrorLookup | None = None,
) -> RipSnapshot | None:
    """Capture the selection and its one-ring before the native Rip runs.

    Requires ``prepare_topology(..., mark_vertex_ids=True)`` to have created
    the vertex ID layer and assigned the face IDs. Vertex IDs are assigned
    here after the selected region and its mirror targets are resolved. This
    is single-use per fresh prepare; rerun prepare before reusing it with a
    different selection.
    """

    # Snapshot records index into bm.verts; the table must be valid on every
    # path, including lookup=None where no later guard rebuilds it.
    bm.verts.ensure_lookup_table()

    if lookup is not None and not _lookup_matches_mesh(lookup, bm, axis_index, tolerance):
        lookup = core.build_vertex_mirror_lookup([vertex.co for vertex in bm.verts], axis_index, tolerance)

    vertex_id_layer = bm.verts.layers.int.get(core.VERT_RIP_ID_LAYER)
    face_id_layer = bm.faces.layers.int.get(core.FACE_ID_LAYER)
    if vertex_id_layer is None or face_id_layer is None:
        return None

    if lookup is None:
        region_vertices = _snapshot_region_vertices(bm, None)
        if not region_vertices:
            return None
        lookup = core.build_vertex_mirror_lookup([vertex.co for vertex in bm.verts], axis_index, tolerance)
        mirror_indices = lookup.find_all_mirrored([vertex.co for vertex in region_vertices])
    else:
        region_vertices, mirror_indices = _resolve_snapshot_region(bm, lookup)
        if not region_vertices:
            return None

    _mark_snapshot_vertex_ids(bm, vertex_id_layer, region_vertices, mirror_indices)
    return _build_snapshot_records(
        bm,
        axis_index,
        tolerance,
        vertex_id_layer,
        face_id_layer,
        region_vertices,
        mirror_indices,
    )


def _mark_snapshot_vertex_ids(
    bm: bmesh.types.BMesh,
    vertex_id_layer,
    region_vertices: tuple[bmesh.types.BMVert, ...],
    mirror_indices: tuple[int | None, ...],
) -> None:
    """Mark only the RIP region and resolved mirror vertices."""

    marked_indices = {vertex.index for vertex in region_vertices}
    marked_indices.update(index for index in mirror_indices if index is not None)
    for index in marked_indices:
        bm.verts[int(index)][vertex_id_layer] = int(index) + 1


def _snapshot_region_vertices(
    bm: bmesh.types.BMesh,
    lookup: core.VertexMirrorLookup | None,
) -> tuple[bmesh.types.BMVert, ...]:
    """Return selected vertices plus their edge one-ring in legacy order."""

    selected_indices = None if lookup is None else lookup._selected_indices
    if selected_indices is None:
        selected = (vertex for vertex in bm.verts if vertex.select and not vertex.hide)
    else:
        candidates = (bm.verts[int(index)] for index in selected_indices.tolist())
        selected = (vertex for vertex in candidates if vertex.select and not vertex.hide)
    region = {vertex.index: vertex for vertex in selected}
    for vertex in tuple(region.values()):
        for edge in vertex.link_edges:
            other = edge.other_vert(vertex)
            region.setdefault(other.index, other)
    return tuple(region.values())


def _resolve_snapshot_region(
    bm: bmesh.types.BMesh,
    lookup: core.VertexMirrorLookup,
) -> tuple[tuple[bmesh.types.BMVert, ...], tuple[int | None, ...]]:
    """Resolve only the selected RIP region and its edge one-ring."""

    region_vertices = _snapshot_region_vertices(bm, lookup)
    mirror_indices = lookup.find_all_mirrored([vertex.co for vertex in region_vertices])
    return region_vertices, mirror_indices


def _build_snapshot_records(
    bm: bmesh.types.BMesh,
    axis_index: int,
    tolerance: float,
    vertex_id_layer,
    face_id_layer,
    region_vertices: tuple[bmesh.types.BMVert, ...],
    mirror_indices: tuple[int | None, ...],
) -> RipSnapshot:
    """Build the immutable RIP snapshot from an already-resolved region."""

    records = []
    for vertex, mirror_index in zip(region_vertices, mirror_indices, strict=True):
        records.append(
            RipVertexRecord(
                vertex_id=int(vertex[vertex_id_layer]),
                location=Coordinate3D(
                    x=float(vertex.co[0]),
                    y=float(vertex.co[1]),
                    z=float(vertex.co[2]),
                ),
                mirror_vertex_id=(None if mirror_index is None else int(bm.verts[int(mirror_index)][vertex_id_layer])),
                face_ids=tuple(sorted(int(face[face_id_layer]) for face in vertex.link_faces)),
                selected=bool(vertex.select),
            )
        )
    return RipSnapshot(
        axis_index=axis_index,
        tolerance=tolerance,
        vertices=tuple(sorted(records, key=lambda record: record.vertex_id)),
    )


def _lookup_matches_mesh(
    lookup: core.VertexMirrorLookup,
    bm: bmesh.types.BMesh,
    axis_index: int,
    tolerance: float,
) -> bool:
    if not isinstance(lookup, core.VertexMirrorLookup):
        return False
    try:
        bm.verts.ensure_lookup_table()
        if len(lookup._coords) != len(bm.verts):
            return False
        if lookup._axis_index != axis_index or lookup._tolerance != tolerance:
            return False
        if len(bm.verts) == 0:
            return True
        first = bm.verts[0].co
        last = bm.verts[len(bm.verts) - 1].co
        return (
            tuple(float(value) for value in lookup._coords[0]) == core._coordinate_3d(first).as_tuple()
            and tuple(float(value) for value in lookup._coords[-1]) == core._coordinate_3d(last).as_tuple()
        )
    except (ReferenceError, RuntimeError, IndexError):
        return False


# ---------------------------------------------------------------------------
# result detection (watcher predicates)


def has_rip_result(bm: bmesh.types.BMesh) -> bool:
    """True as soon as one duplicated vertex ID exists."""

    vertex_id_layer = bm.verts.layers.int.get(core.VERT_RIP_ID_LAYER)
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


def rip_result_signature(bm: bmesh.types.BMesh) -> RipSignature | None:
    """Stable signature of the native result: duplicated IDs and coordinates."""

    vertex_id_layer = bm.verts.layers.int.get(core.VERT_RIP_ID_LAYER)
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


# ---------------------------------------------------------------------------
# derived state shared by preflight and apply


@dataclass
class _DupVertex:
    vertex_id: int
    copies: list[bmesh.types.BMVert]
    copy_face_ids: list[frozenset[int]]


@dataclass
class _DerivedRip:
    dup_vertices: dict[int, _DupVertex]
    seam_edge_pairs: dict[int, tuple[tuple[int, ...], list[bmesh.types.BMEdge]]]
    fill_faces: list[bmesh.types.BMFace]
    verts_by_id: dict[int, list[bmesh.types.BMVert]]
    mirror_seam_edges: dict[int, bmesh.types.BMEdge]
    self_mirrored: bool = False


def _derive(
    bm: bmesh.types.BMesh, snapshot: RipSnapshot, mirror_face_ids: MirrorFaceMap
) -> tuple[_DerivedRip | None, str | None]:
    vertex_id_layer = bm.verts.layers.int.get(core.VERT_RIP_ID_LAYER)
    edge_layer = bm.edges.layers.int.get(core.EDGE_ORIGINAL_LAYER)
    face_id_layer = bm.faces.layers.int.get(core.FACE_ID_LAYER)
    if vertex_id_layer is None or edge_layer is None or face_id_layer is None:
        return None, "temporary topology markers are missing"

    records = snapshot.record_by_id()

    verts_by_id: dict[int, list[bmesh.types.BMVert]] = {}
    for vertex in bm.verts:
        vertex_id = int(vertex[vertex_id_layer])
        if vertex_id > 0:
            verts_by_id.setdefault(vertex_id, []).append(vertex)

    dup_ids = {vertex_id for vertex_id, group in verts_by_id.items() if len(group) > 1}
    if not dup_ids:
        return None, "the native Rip left no duplicated vertex"

    for vertex_id in dup_ids:
        if len(verts_by_id[vertex_id]) != 2:
            return None, f"vertex {vertex_id} split into {len(verts_by_id[vertex_id])} copies (unsupported)"
        record = records.get(vertex_id)
        if record is None or not record.selected:
            return None, "the native Rip duplicated a vertex outside the captured selection"

    # A selected vertex that moved without being duplicated means the native
    # result is not a pure rip of the captured selection.
    for record in snapshot.vertices:
        if not record.selected or record.vertex_id in dup_ids:
            continue
        group = verts_by_id.get(record.vertex_id)
        if not group:
            return None, "a selected vertex disappeared during Rip"
        old = Vector(record.location.as_tuple())
        if (group[0].co - old).length > _MOVE_EPSILON:
            return None, "a selected vertex moved without being ripped (unsupported result)"

    # Fill faces repeat a duplicated vertex ID on two corners.
    fill_faces = []
    fill_face_keys = set()
    candidate_faces = {face for vertex_id in dup_ids for vertex in verts_by_id[vertex_id] for face in vertex.link_faces}
    for face in candidate_faces:
        corner_ids = [int(loop.vert[vertex_id_layer]) for loop in face.loops]
        positive = [vertex_id for vertex_id in corner_ids if vertex_id > 0]
        if len(positive) != len(set(positive)):
            fill_faces.append(face)
            fill_face_keys.add(face.index)

    dup_vertices: dict[int, _DupVertex] = {}
    for vertex_id in sorted(dup_ids):
        copies = verts_by_id[vertex_id]
        copy_face_ids = []
        for copy in copies:
            face_ids = frozenset(
                int(face[face_id_layer]) for face in copy.link_faces if face.index not in fill_face_keys
            )
            copy_face_ids.append(face_ids)
        if copy_face_ids[0] == copy_face_ids[1]:
            return None, f"could not tell the two copies of vertex {vertex_id} apart"
        dup_vertices[vertex_id] = _DupVertex(
            vertex_id=vertex_id,
            copies=copies,
            copy_face_ids=copy_face_ids,
        )

    # Duplicated (seam) edges by shared unique edge ID.
    edges_by_id: dict[int, list[bmesh.types.BMEdge]] = {}
    candidate_edges = {edge for vertex_id in dup_ids for vertex in verts_by_id[vertex_id] for edge in vertex.link_edges}
    for edge in candidate_edges:
        edge_id = int(edge[edge_layer])
        if edge_id > 0:
            edges_by_id.setdefault(edge_id, []).append(edge)

    seam_edge_pairs: dict[int, tuple[tuple[int, ...], list[bmesh.types.BMEdge]]] = {}
    for edge_id, group in edges_by_id.items():
        if len(group) < 2:
            continue
        if len(group) > 2:
            return None, f"edge {edge_id} split into {len(group)} copies (unsupported)"
        endpoint_ids = []
        for edge in group:
            ids = tuple(sorted(int(vertex[vertex_id_layer]) for vertex in edge.verts))
            endpoint_ids.append(ids)
        if endpoint_ids[0] != endpoint_ids[1] or 0 in endpoint_ids[0]:
            return None, f"seam edge {edge_id} copies disagree about their endpoints"
        seam_edge_pairs[edge_id] = (endpoint_ids[0], group)
    if not seam_edge_pairs:
        return None, "the native Rip duplicated vertices but no seam edge was found"

    # Every duplicated vertex must sit on a seam edge; otherwise the result is
    # a rip form the measurements did not cover.
    seam_vertex_ids = {vertex_id for endpoint_ids, _group in seam_edge_pairs.values() for vertex_id in endpoint_ids}
    if not dup_ids.issubset(seam_vertex_ids):
        return None, "a duplicated vertex is not part of any seam edge (unsupported result)"

    # Classify self-mirror status via seam *edge* set mirror-closure:
    # full → V-opening path; partial → explicit decline.
    # Vertex-only "every dup's mirror is also a dup" is necessary but not
    # sufficient: a mixed seam {A–A′, A–B} with unduplicated extension
    # endpoint B has all dups self-paired yet ρ(A–B)=A′–B′ is not a seam
    # edge, so V-opening would leave asymmetric topology.
    for vertex_id in dup_ids:
        record = records.get(vertex_id)
        if record is None:
            return None, f"seam vertex {vertex_id} is outside the captured one-ring"
        if record.mirror_vertex_id is None:
            return None, "a seam vertex has no mirrored counterpart"

    seam_edge_keys: set[frozenset[int]] = set()
    for endpoint_ids, _group in seam_edge_pairs.values():
        seam_edge_keys.add(frozenset(endpoint_ids))

    mirrored_edge_keys: set[frozenset[int]] = set()
    for endpoints in seam_edge_keys:
        mirrored_endpoints: list[int] = []
        for vertex_id in endpoints:
            record = records.get(vertex_id)
            if record is None:
                return None, f"seam vertex {vertex_id} is outside the captured one-ring"
            if record.mirror_vertex_id is None:
                return None, "a seam vertex has no mirrored counterpart"
            mirrored_endpoints.append(record.mirror_vertex_id)
        mirrored_edge_keys.add(frozenset(mirrored_endpoints))

    if seam_edge_keys == mirrored_edge_keys:
        fully_self_mirrored = True
    elif seam_edge_keys & mirrored_edge_keys:
        return None, "partial self-overlap is not supported yet"
    else:
        fully_self_mirrored = False

    # Preflight the face pairing needed to tell copies / banks apart.
    for dup in dup_vertices.values():
        for face_ids in dup.copy_face_ids:
            for face_id in face_ids:
                if FaceId(face_id) not in mirror_face_ids:
                    return None, "a ripped face has no exact mirrored counterpart"

    if fully_self_mirrored:
        # No external mirror edges to split; bank identity is resolved at apply
        # time from selection + pre-native mirror_vertex_id + face-ID sets.
        return (
            _DerivedRip(
                dup_vertices=dup_vertices,
                seam_edge_pairs=seam_edge_pairs,
                fill_faces=fill_faces,
                verts_by_id=verts_by_id,
                mirror_seam_edges={},
                self_mirrored=True,
            ),
            None,
        )

    # External (non-self-mirrored) path: resolve the mirrored counterpart edge
    # for every seam edge.  Mirror endpoints must still be unripped.
    mirror_seam_edges: dict[int, bmesh.types.BMEdge] = {}
    used_mirror_edges: set[int] = set()
    for edge_id, (endpoint_ids, _group) in seam_edge_pairs.items():
        mirror_endpoint_verts = []
        for vertex_id in endpoint_ids:
            record = records.get(vertex_id)
            if record is None:
                return None, f"seam vertex {vertex_id} is outside the captured one-ring"
            mirror_vertex_id = record.mirror_vertex_id
            if mirror_vertex_id is None:
                return None, "a seam vertex has no mirrored counterpart"
            group = verts_by_id.get(mirror_vertex_id)
            if not group:
                return None, "a mirrored seam vertex disappeared during Rip"
            if len(group) > 1:
                # Should have been caught as partial self-overlap above when
                # the ripped mirror is itself a seam vertex; any residual case
                # (mirror ripped but not classified as seam) stays declined.
                return None, "a mirrored seam vertex was itself ripped (selection spans both sides?)"
            mirror_endpoint_verts.append(group[0])
        mirror_edge = bm.edges.get(mirror_endpoint_verts)
        if mirror_edge is None:
            return None, "the mirrored seam edge does not exist; topology is not symmetric"
        if mirror_edge.index in used_mirror_edges:
            return None, "two seam edges map to the same mirrored edge"
        used_mirror_edges.add(mirror_edge.index)
        mirror_seam_edges[edge_id] = mirror_edge

    return (
        _DerivedRip(
            dup_vertices=dup_vertices,
            seam_edge_pairs=seam_edge_pairs,
            fill_faces=fill_faces,
            verts_by_id=verts_by_id,
            mirror_seam_edges=mirror_seam_edges,
            self_mirrored=False,
        ),
        None,
    )


def preflight_reason(bm: bmesh.types.BMesh, snapshot: RipSnapshot, mirror_face_ids: MirrorFaceMap) -> str | None:
    """Validate the native result without touching the mesh."""

    derived, reason = _derive(bm, snapshot, mirror_face_ids)
    if reason is not None:
        return reason
    assert derived is not None
    return None


# ---------------------------------------------------------------------------
# mirror application


def apply_mirrored_rip(
    bm: bmesh.types.BMesh,
    snapshot: RipSnapshot,
    mirror_face_ids: MirrorFaceMap,
) -> tuple[int, str | None]:
    """Reproduce the observed rip on the mirrored seam.  All-or-nothing.

    Returns ``(mirrored_seam_edge_count, failure_reason)``.  The caller owns
    the topology backup and must roll back when a reason is returned.

    Fully self-mirrored seams take the V-opening path (coordinate overwrite
    only); ordinary seams still split the external mirror edge set.
    """

    derived, reason = _derive(bm, snapshot, mirror_face_ids)
    if reason is not None or derived is None:
        return 0, reason or "internal error deriving the rip result"

    if derived.self_mirrored:
        return apply_self_mirrored_rip(bm, snapshot, mirror_face_ids, derived)

    axis_index = snapshot.axis_index
    records = snapshot.record_by_id()

    # Native Rip duplicates ONLY the selected vertices; unselected seam ends
    # stay pinched even on a mesh boundary.  split_edges reproduces this
    # exactly when the split vertices are passed explicitly.
    mirror_split_verts = []
    split_plan: list[tuple[int, list[tuple[frozenset[int], Vector]]]] = []
    for vertex_id, dup in derived.dup_vertices.items():
        mirror_vertex_id = records[vertex_id].mirror_vertex_id
        assert mirror_vertex_id is not None
        mirror_group = derived.verts_by_id.get(mirror_vertex_id, [])
        if len(mirror_group) != 1:
            return 0, "a mirrored seam vertex is not unique before splitting"
        mirror_split_verts.append(mirror_group[0])
        copy_plan = []
        for copy, face_ids in zip(dup.copies, dup.copy_face_ids, strict=True):
            expected = frozenset(int(mirror_face_ids[FaceId(face_id)]) for face_id in face_ids)
            copy_plan.append((expected, core.mirror_coordinate(copy.co, axis_index)))
        split_plan.append((mirror_vertex_id, copy_plan))

    mirror_edges = [derived.mirror_seam_edges[edge_id] for edge_id in sorted(derived.mirror_seam_edges)]
    bmesh.ops.split_edges(bm, edges=mirror_edges, verts=mirror_split_verts, use_verts=True)

    # split_edges invalidates wrappers; rebuild the ID table before matching.
    vertex_id_layer = bm.verts.layers.int.get(core.VERT_RIP_ID_LAYER)
    face_id_layer = bm.faces.layers.int.get(core.FACE_ID_LAYER)
    if vertex_id_layer is None or face_id_layer is None:
        return 0, "temporary topology markers were lost while splitting"
    verts_by_id: dict[int, list[bmesh.types.BMVert]] = {}
    for vertex in bm.verts:
        vertex_id = int(vertex[vertex_id_layer])
        if vertex_id > 0:
            verts_by_id.setdefault(vertex_id, []).append(vertex)

    mirror_copy_by_key: dict[tuple[int, int], bmesh.types.BMVert] = {}
    for (vertex_id, dup), (mirror_vertex_id, copy_plan) in zip(derived.dup_vertices.items(), split_plan, strict=True):
        del dup
        mirror_group = verts_by_id.get(mirror_vertex_id, [])
        if len(mirror_group) != len(copy_plan):
            return 0, "the mirrored split produced a different number of copies"
        remaining = list(mirror_group)
        for copy_index, (expected, mirrored_co) in enumerate(copy_plan):
            match = None
            for candidate in remaining:
                candidate_face_ids = frozenset(int(face[face_id_layer]) for face in candidate.link_faces)
                if candidate_face_ids == expected:
                    match = candidate
                    break
            if match is None:
                return 0, "could not match a mirrored copy to the ripped side"
            remaining.remove(match)
            match.co = mirrored_co
            match.select = False
            mirror_copy_by_key[(vertex_id, copy_index)] = match

    if derived.fill_faces:
        created, reason = _mirror_fill_faces(
            bm,
            snapshot,
            derived,
            verts_by_id,
            mirror_copy_by_key,
            vertex_id_layer,
            face_id_layer,
        )
        if reason is not None:
            return 0, reason
        if created != len(derived.fill_faces):
            return 0, "the mirrored fill produced a different number of faces"

    return len(mirror_edges), None


def apply_self_mirrored_rip(
    bm: bmesh.types.BMesh,
    snapshot: RipSnapshot,
    mirror_face_ids: MirrorFaceMap,
    derived: _DerivedRip | None = None,
) -> tuple[int, str | None]:
    """V-open a fully self-mirrored seam: coordinate overwrite only (no topo).

    source bank = native moved/selected bank (kept).  Each non-source copy of
    vertex V receives ρ of the source copy of mirror(V) — role-based, not
    same-vid overwrite.  Fill faces ride along because their corners are the
    bank vertices.
    """

    if derived is None:
        derived, reason = _derive(bm, snapshot, mirror_face_ids)
        if reason is not None or derived is None:
            return 0, reason or "internal error deriving the rip result"
    if not derived.self_mirrored:
        return 0, "internal error: self-mirrored path invoked on a non-self-mirrored seam"

    records = snapshot.record_by_id()
    axis_index = snapshot.axis_index

    source_by_vid: dict[int, bmesh.types.BMVert] = {}
    nonsource_by_vid: dict[int, bmesh.types.BMVert] = {}
    source_face_ids_by_vid: dict[int, frozenset[int]] = {}

    for vertex_id, dup in derived.dup_vertices.items():
        source, nonsource, source_face_ids, identify_reason = _identify_source_bank(dup, records[vertex_id])
        if identify_reason is not None or source is None or nonsource is None or source_face_ids is None:
            return 0, identify_reason or f"could not identify banks of vertex {vertex_id}"
        # Face-ID sets already distinguish the two copies; require they stay
        # distinct after excluding fill (checked in _derive).  Cross-check that
        # the partner's face sets are resolvable via the face pair table so
        # bank identity is well-defined.
        for face_ids in dup.copy_face_ids:
            for face_id in face_ids:
                if FaceId(face_id) not in mirror_face_ids:
                    return 0, "a ripped face has no exact mirrored counterpart"
        source_by_vid[vertex_id] = source
        nonsource_by_vid[vertex_id] = nonsource
        source_face_ids_by_vid[vertex_id] = source_face_ids

    # Seam-wide bank consistency: source(V) and source(mirror(V)) must sit on
    # face-paired sides.  A mixed face-side assignment across vertices means
    # bank identity is not a single global choice.
    checked_pairs: set[frozenset[int]] = set()
    for vertex_id, source_face_ids in source_face_ids_by_vid.items():
        mirror_vertex_id = records[vertex_id].mirror_vertex_id
        if mirror_vertex_id is None:
            return 0, "a seam vertex has no mirrored counterpart"
        pair_key = frozenset((vertex_id, mirror_vertex_id))
        if pair_key in checked_pairs:
            continue
        checked_pairs.add(pair_key)
        partner_face_ids = source_face_ids_by_vid.get(mirror_vertex_id)
        if partner_face_ids is None:
            return 0, "a self-mirrored seam vertex has no source counterpart"
        expected = frozenset(int(mirror_face_ids[FaceId(face_id)]) for face_id in source_face_ids)
        if expected != partner_face_ids:
            return 0, "source banks have inconsistent face-sides across the seam"

    # Role-based V-opening: nonsource(V) ← ρ(source(mirror(V))).
    for vertex_id, nonsource in nonsource_by_vid.items():
        mirror_vertex_id = records[vertex_id].mirror_vertex_id
        if mirror_vertex_id is None:
            return 0, "a seam vertex has no mirrored counterpart"
        source_mirror = source_by_vid.get(mirror_vertex_id)
        if source_mirror is None:
            return 0, "a self-mirrored seam vertex has no source counterpart"
        nonsource.co = core.mirror_coordinate(source_mirror.co, axis_index)
        nonsource.select = False

    # Native already selects the source bank; re-assert so selection state is
    # definite after our non-source clear.
    for source in source_by_vid.values():
        source.select = True

    return len(derived.seam_edge_pairs), None


def _identify_source_bank(
    dup: _DupVertex, record: RipVertexRecord
) -> tuple[bmesh.types.BMVert | None, bmesh.types.BMVert | None, frozenset[int] | None, str | None]:
    """Pick (source, non-source, source face-IDs) for one duplicated seam vertex.

    Preference order:
    1. Exactly one selected copy → that is the native moved bank, provided
       movement (when measurable) does not contradict the selection signal.
    2. Exactly one copy that left the pre-rip location → moved bank.
    3. Zero move / ambiguous selection → decline (caller keeps native result).
    """

    copies = dup.copies
    if len(copies) != 2:
        return None, None, None, f"vertex {dup.vertex_id} does not have exactly two copies"

    old = Vector(record.location.as_tuple())
    moved = [copy for copy in copies if (copy.co - old).length > _MOVE_EPSILON]
    selected = [copy for copy in copies if copy.select]

    def _banks(source: bmesh.types.BMVert) -> tuple[bmesh.types.BMVert, bmesh.types.BMVert, frozenset[int]]:
        nonsource = copies[1] if copies[0] is source else copies[0]
        source_index = 0 if copies[0] is source else 1
        return source, nonsource, dup.copy_face_ids[source_index]

    if len(selected) == 1:
        source, nonsource, source_face_ids = _banks(selected[0])
        # Movement is always measurable from the pre-rip snapshot.  Accept
        # selection only when it agrees with movement (or both banks still
        # sit at the pre-state location — zero-move / Esc).
        source_moved = source in moved
        nonsource_moved = nonsource in moved
        if source_moved and nonsource_moved:
            return None, None, None, "both banks of a self-mirrored seam moved (unsupported)"
        if (not source_moved) and nonsource_moved:
            return None, None, None, "selection and movement disagree about the source bank"
        # Consistent: source moved & nonsource stationary, or zero-move.
        return source, nonsource, source_face_ids, None

    if len(moved) == 1:
        source, nonsource, source_face_ids = _banks(moved[0])
        return source, nonsource, source_face_ids, None

    if len(moved) > 1:
        return None, None, None, "both banks of a self-mirrored seam moved (unsupported)"
    return None, None, None, "could not identify the source bank of the self-mirrored seam"


def _mirror_fill_faces(
    bm: bmesh.types.BMesh,
    snapshot: RipSnapshot,
    derived: _DerivedRip,
    verts_by_id: dict[int, list[bmesh.types.BMVert]],
    mirror_copy_by_key: dict[tuple[int, int], bmesh.types.BMVert],
    vertex_id_layer,
    face_id_layer,
) -> tuple[int, str | None]:
    """Recreate the native Rip Fill bridge faces on the mirrored seam.

    Fill faces are re-found after the mirror split by the repeated-corner
    rule; corner-level copy identity comes from the stable per-copy face-ID
    sets captured before any mutation.
    """

    records = snapshot.record_by_id()

    # Resolve the source copies afresh: wrappers from before split_edges may
    # be stale.  A source copy is identified by its non-fill face-ID set.
    source_copy_by_key: dict[int, dict[int, int]] = {}
    fill_face_indices: set[int] = set()
    source_fill_faces = []
    candidate_faces = {
        face
        for vertex_id in derived.dup_vertices
        for vertex in verts_by_id.get(vertex_id, [])
        for face in vertex.link_faces
    }
    for face in candidate_faces:
        corner_ids = [int(loop.vert[vertex_id_layer]) for loop in face.loops]
        positive = [vertex_id for vertex_id in corner_ids if vertex_id > 0]
        if len(positive) != len(set(positive)):
            source_fill_faces.append(face)
            fill_face_indices.add(face.index)

    for vertex_id, dup in derived.dup_vertices.items():
        group = verts_by_id.get(vertex_id, [])
        if len(group) != len(dup.copy_face_ids):
            return 0, "a source rip copy disappeared while mirroring the fill"
        assignment: dict[int, int] = {}
        for vertex in group:
            face_ids = frozenset(
                int(face[face_id_layer]) for face in vertex.link_faces if face.index not in fill_face_indices
            )
            matched_index = None
            for copy_index, expected in enumerate(dup.copy_face_ids):
                if expected == face_ids and copy_index not in assignment:
                    matched_index = copy_index
                    break
            if matched_index is None:
                return 0, "could not identify a source rip copy for the fill"
            assignment[matched_index] = vertex.index
        source_copy_by_key[vertex_id] = assignment

    created = 0
    for face in source_fill_faces:
        mirror_corners = []
        source_loops = []
        for loop in face.loops:
            vertex = loop.vert
            vertex_id = int(vertex[vertex_id_layer])
            mirror_vertex = None
            if vertex_id in source_copy_by_key:
                copy_index = next(
                    (
                        index
                        for index, vert_index in source_copy_by_key[vertex_id].items()
                        if vert_index == vertex.index
                    ),
                    None,
                )
                if copy_index is None:
                    return 0, "a fill corner does not belong to a known rip copy"
                mirror_vertex = mirror_copy_by_key.get((vertex_id, copy_index))
            elif vertex_id > 0:
                record = records.get(vertex_id)
                if record is None or record.mirror_vertex_id is None:
                    return 0, "a fill corner has no mirrored counterpart"
                group = verts_by_id.get(record.mirror_vertex_id, [])
                if len(group) != 1:
                    return 0, "a mirrored fill corner is ambiguous"
                mirror_vertex = group[0]
            if mirror_vertex is None:
                return 0, "a fill corner could not be mirrored"
            mirror_corners.append(mirror_vertex)
            source_loops.append(loop)

        if len(set(mirror_corners)) != len(mirror_corners):
            return 0, "the mirrored fill face would be degenerate"
        try:
            mirror_face = bm.faces.new(list(reversed(mirror_corners)))
        except ValueError:
            return 0, "the mirrored fill face already exists"
        # BMFace.copy_from invalidates the freshly created wrapper (measured on
        # 4.2/5.2), so face attributes are copied field by field instead.
        mirror_face.material_index = face.material_index
        mirror_face.smooth = face.smooth
        mirror_face.select = False
        # faces.new(reversed(corners)) makes loop k sit on mirror_corners[n-1-k],
        # whose source counterpart is source_loops[n-1-k].
        for mirror_loop, source_loop in zip(mirror_face.loops, reversed(source_loops), strict=True):
            mirror_loop.copy_from(source_loop)
        mirror_face.normal_update()
        created += 1

    return created, None
