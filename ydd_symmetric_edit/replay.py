# SPDX-License-Identifier: GPL-3.0-or-later

"""One-shot symmetry replay for native Connect and Merge operations."""

from __future__ import annotations

import traceback
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import bmesh
import bpy
import numpy  # type: ignore
from bpy.props import EnumProperty, FloatProperty
from mathutils import Vector

from . import backup, core
from ._types import MirrorOverlap
from .snapshot import capture_selection_snapshot

_MERGE_MODES = frozenset({"CENTER", "COLLAPSE", "FIRST", "LAST"})

# float64 shaped (N, 3) coordinate matrix from capture_selection_snapshot / _vertex_snapshot.
VertexCoordArray = numpy.ndarray


@dataclass(frozen=True)
class MirrorSelection:
    """Coordinate-index snapshot used before a native topology operation.

    ``overlap`` / ``complete`` classify the selection against its own mirror
    image; ``pairs`` is the whole-mesh involutive vertex pair table that the
    classification was derived from (``pairs[pairs[v]] == v``).
    """

    selected: frozenset[int]
    shared: frozenset[int]
    off: frozenset[int]
    mirror_by_source: dict[int, int]
    missing: frozenset[int]
    overlap: MirrorOverlap
    complete: bool
    pairs: dict[int, int]

    @property
    def mirrors(self) -> frozenset[int]:
        return frozenset(self.mirror_by_source.values())


def classify_mirror_selection(
    coords: VertexCoordArray,
    selected_indices: Sequence[int],
    *,
    axis_index: int,
    tolerance: float,
) -> MirrorSelection:
    """Classify selected coordinates and resolve every available counterpart."""

    classification = core.classify_selection_overlap(
        coords,
        selected_indices,
        axis_index=axis_index,
        tolerance=tolerance,
    )
    selected = frozenset(selected_indices)
    shared = frozenset(index for index in selected if abs(coords[index][axis_index]) <= tolerance)
    off = selected - shared
    mirror_by_source = {index: classification.pairs[index] for index in off if index in classification.pairs}
    return MirrorSelection(
        selected=selected,
        shared=shared,
        off=off,
        mirror_by_source=mirror_by_source,
        missing=frozenset(off - mirror_by_source.keys()),
        overlap=classification.overlap,
        complete=classification.complete,
        pairs=classification.pairs,
    )


def selection_crosses_mirror(snapshot: MirrorSelection) -> bool:
    """Return whether the counterpart set already intersects the selection."""

    return bool(snapshot.mirrors & snapshot.selected)


def split_merge_clusters(
    bm: bmesh.types.BMesh,
    selected_indices: Sequence[int],
    mode: str,
) -> tuple[tuple[int, ...], ...]:
    """Partition selection as required by a deterministic mirrored merge."""

    selected = frozenset(selected_indices)
    if not selected:
        return ()
    if mode != "COLLAPSE":
        return (tuple(selected_indices),)

    bm.verts.ensure_lookup_table()
    remaining = set(selected)
    clusters = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        pending = [start]
        component = []
        while pending:
            index = pending.pop()
            component.append(index)
            vertex = bm.verts[index]
            neighbors = sorted(
                (
                    edge.other_vert(vertex).index
                    for edge in vertex.link_edges
                    if edge.other_vert(vertex).index in remaining
                ),
                reverse=True,
            )
            for neighbor in neighbors:
                remaining.remove(neighbor)
                pending.append(neighbor)
        clusters.append(tuple(sorted(component)))
    return tuple(clusters)


def calculate_merge_target(
    cluster: Sequence[int],
    coords: VertexCoordArray,
    mode: str,
    *,
    history_coords: Sequence[Vector] = (),
) -> Vector:
    """Precompute the native-side merge destination for one source cluster."""

    if not cluster:
        raise ValueError("A merge cluster cannot be empty")
    if mode == "FIRST":
        if not history_coords:
            raise ValueError("Merge At First requires selection history")
        return history_coords[0].copy()
    if mode == "LAST":
        if not history_coords:
            raise ValueError("Merge At Last requires selection history")
        return history_coords[-1].copy()
    if mode not in {"CENTER", "COLLAPSE"}:
        raise ValueError(f"Unsupported deterministic merge mode: {mode}")

    target = Vector((0.0, 0.0, 0.0))
    for index in cluster:
        row = coords[index]
        target += Vector((float(row[0]), float(row[1]), float(row[2])))
    return target / len(cluster)


def _map_history_via_pairs(
    history_indices: Sequence[int],
    pairs: dict[int, int],
) -> tuple[int, ...] | None:
    """Map a history path through the involutive pair table; ``None`` when any
    vertex is unpaired.

    Deliberately shares the classification's pair table instead of running a
    separate lookup: a coordinate-based partial batch sees no on-plane
    queries, so a near-plane vertex the classification rejected could sneak
    back in as a counterpart.
    """

    mapped = []
    for index in history_indices:
        partner = pairs.get(index)
        if partner is None:
            return None
        mapped.append(partner)
    return tuple(mapped)


def _vertex_snapshot(
    bm: bmesh.types.BMesh,
    *,
    mesh_object=None,
) -> tuple[VertexCoordArray, tuple[int, ...], tuple[Vector, ...], tuple[int, ...]]:
    """Capture vertex coords / selection / BMVert history for classify paths.

    Coordinates are float64 ``(N, 3)`` from :func:`capture_selection_snapshot`
    (Mesh bulk when *mesh_object* is set). Downstream consumers index rows and
    call ``.copy()``; numpy rows satisfy both.
    """

    capture = capture_selection_snapshot(
        bm,
        mesh_object=mesh_object,
        domains=("VERT",),
        include_history=True,
    )
    selected = tuple(int(index) for index in capture.selected_verts)
    return capture.coords, selected, capture.history_coords, capture.history_indices


def _symmetry_parameters(context) -> tuple[bpy.types.Object, int, float] | None:
    obj = context.edit_object
    if obj is None or obj.type != "MESH":
        return None
    # Multi-object edit mode: the native operator acts on every object in the
    # mode while the mirror pass below only sees the active one, so replaying
    # would double-apply to the others.  Same rule as _single_edit_mesh_poll.
    if len(context.objects_in_mode_unique_data) != 1:
        return None
    axes = core.enabled_mesh_symmetry_axes(obj)
    if len(axes) != 1:
        return None
    _axis_name, axis_index = axes[0]
    settings = context.scene.ydd_symmetric_edit
    return obj, axis_index, float(settings.tolerance)


def _clear_selection(bm: bmesh.types.BMesh) -> None:
    for face in bm.faces:
        face.select = False
    for edge in bm.edges:
        edge.select = False
    for vertex in bm.verts:
        vertex.select = False
    bm.select_history.clear()


def _set_vertex_path(
    bm: bmesh.types.BMesh,
    path_coords: Sequence[Vector],
    axis_index: int,
    tolerance: float,
    *,
    selected_coords: Sequence[Vector] | None = None,
) -> bool:
    selected_coords = path_coords if selected_coords is None else selected_coords
    bm.verts.ensure_lookup_table()
    coords = tuple(vertex.co.copy() for vertex in bm.verts)
    lookup = core.build_vertex_mirror_lookup(coords, axis_index, tolerance)

    # Deduplicate exact query positions first: the injective batch resolver
    # would otherwise see two queries competing for one vertex and reject both.
    unique_coords: list[Vector] = []
    position_by_key: dict[tuple[float, float, float], int] = {}
    for coordinate in (*selected_coords, *path_coords):
        key = (float(coordinate[0]), float(coordinate[1]), float(coordinate[2]))
        if key not in position_by_key:
            position_by_key[key] = len(unique_coords)
            unique_coords.append(coordinate)
    resolved = lookup.find_all_direct(unique_coords)

    def resolve(coordinate: Vector) -> bmesh.types.BMVert | None:
        key = (float(coordinate[0]), float(coordinate[1]), float(coordinate[2]))
        index = resolved[position_by_key[key]]
        return None if index is None else bm.verts[index]

    selected_vertices = [resolve(coordinate) for coordinate in selected_coords]
    path_vertices = [resolve(coordinate) for coordinate in path_coords]
    if any(vertex is None for vertex in (*selected_vertices, *path_vertices)):
        return False

    _clear_selection(bm)
    for vertex in selected_vertices:
        assert vertex is not None
        vertex.select = True
    for vertex in path_vertices:
        assert vertex is not None
        vertex.select = True
        bm.select_history.add(vertex)
    return True


# Test-visible Connect reports (Operator.report is not patchable; same pattern
# as operators._FINISH_REPORTS).
_CONNECT_REPORTS: list[tuple[str, str]] = []
# Test-visible Merge reports (same constraint as Operator.report above).
_MERGE_REPORTS: list[tuple[str, str]] = []
# Last extracted R edge endpoint pairs (coords only) for intermediate asserts.
_CONNECT_LAST_R: tuple[tuple[Vector, Vector], ...] = ()


def _connect_report(operator, level: set[str], message: str) -> None:
    """Report and record for Connect tests."""

    kind = "WARNING" if "WARNING" in level else "ERROR" if "ERROR" in level else "INFO"
    _CONNECT_REPORTS.append((kind, message))
    operator.report(level, message)


def _maybe_extend_selection_to_mirror(mesh, axis_index: int, tolerance: float, *, mesh_object=None) -> None:
    """When Scene ``select_mirrored`` is on, add-select ρ(S) after success.

    Best-effort.  Does not touch ``select_history`` (core contract).
    *mesh_object* enables Mesh bulk capture inside extend when provided.
    """

    try:
        settings = getattr(bpy.context.scene, "ydd_symmetric_edit", None)
        if settings is None or not bool(getattr(settings, "select_mirrored", False)):
            return
        bm = bmesh.from_edit_mesh(mesh)
        core.extend_selection_to_mirror(bm, axis_index, tolerance, mesh_object=mesh_object)
        bmesh.update_edit_mesh(mesh, loop_triangles=False, destructive=False)
    except Exception:
        traceback.print_exc()


def _merge_report(operator, level: set[str], message: str) -> None:
    """Report and record for Merge tests."""

    kind = "WARNING" if "WARNING" in level else "ERROR" if "ERROR" in level else "INFO"
    _MERGE_REPORTS.append((kind, message))
    operator.report(level, message)


def _report_missing(operator, count: int, *, partial: bool) -> None:
    if not count:
        return
    if partial:
        # Merge-side report (the partial form is only used by Merge paths).
        _merge_report(
            operator,
            {"WARNING"},
            f"{count} vertices have no mirror counterpart; mirrored partially",
        )
    else:
        _connect_report(
            operator,
            {"WARNING"},
            f"{count} vertices have no mirror counterpart; mirrored connect skipped",
        )


def _report_self_mirrored(operator, recorder=None) -> None:
    (recorder or _connect_report)(
        operator,
        {"INFO"},
        "Selection is symmetric; the native result is already symmetric",
    )


def _report_partial_overlap(operator, action: str) -> None:
    # Only Merge paths reach this helper since the Connect PARTIAL branch
    # was replaced by the lift semantics.
    _merge_report(
        operator,
        {"WARNING"},
        f"Selection partially overlaps its mirror image; ran the native {action} only",
    )


def _native_vert_connect_path() -> set[str]:
    """Invoke native Vertex Connect Path.

    Exposed as a module-level callable so fault-injection tests can replace
    the second call without touching bpy.ops registration.
    """

    return cast(set[str], bpy.ops.mesh.vert_connect_path())


def _remove_connect_markers(bm: bmesh.types.BMesh) -> None:
    """Remove only the two Connect marker layers (EDGE_ORIGINAL / FACE_ID)."""

    for layers, name in (
        (bm.edges.layers.int, core.EDGE_ORIGINAL_LAYER),
        (bm.faces.layers.int, core.FACE_ID_LAYER),
    ):
        layer = layers.get(name)
        if layer is not None:
            layers.remove(layer)


def _ensure_int_layer(layers, name: str):
    layer = layers.get(name)
    if layer is None:
        layer = layers.new(name)
    return layer


def _prepare_connect_markers(bm: bmesh.types.BMesh) -> None:
    """Stamp EDGE_ORIGINAL + FACE_ID for post-hoc R extraction.

    Only the two Connect layers are created/overwritten — other temporary
    layers (Knife session etc.) are left alone. Partial failure removes both
    Connect layers before re-raising so native fallback sees a clean mesh.
    """

    try:
        edge_layer = _ensure_int_layer(bm.edges.layers.int, core.EDGE_ORIGINAL_LAYER)
        face_layer = _ensure_int_layer(bm.faces.layers.int, core.FACE_ID_LAYER)
        for edge_id, edge in enumerate(bm.edges, start=1):
            edge[edge_layer] = edge_id
        for face_id, face in enumerate(bm.faces, start=1):
            face[face_layer] = face_id
    except Exception:
        try:
            _remove_connect_markers(bm)
        except Exception:
            traceback.print_exc()
        raise


def _remark_connect_markers(bm: bmesh.types.BMesh) -> None:
    """Re-stamp every current edge/face so the next native call's novelty is clean."""

    edge_layer = _ensure_int_layer(bm.edges.layers.int, core.EDGE_ORIGINAL_LAYER)
    face_layer = _ensure_int_layer(bm.faces.layers.int, core.FACE_ID_LAYER)
    for edge_id, edge in enumerate(bm.edges, start=1):
        edge[edge_layer] = edge_id
    for face_id, face in enumerate(bm.faces, start=1):
        face[face_layer] = face_id


def _edge_float_attr(bm: bmesh.types.BMesh, edge: bmesh.types.BMEdge, name: str) -> float:
    """Non-mutating read of crease_edge / bevel_weight_edge (default 0)."""

    layer = bm.edges.layers.float.get(name)
    if layer is None:
        return 0.0
    return float(edge[layer])


def _edge_attr_tuple(bm: bmesh.types.BMesh, edge: bmesh.types.BMEdge) -> tuple[bool, bool, float, float]:
    """Guaranteed edge attributes: seam / sharp / crease / bevel weight."""

    return (
        bool(edge.seam),
        not bool(edge.smooth),  # sharp == not smooth
        _edge_float_attr(bm, edge, "crease_edge"),
        _edge_float_attr(bm, edge, "bevel_weight_edge"),
    )


@dataclass(frozen=True)
class _ConnectEffectEdge:
    """One newly generated connect edge with incidence + attributes.

    Coordinates are stored as plain 3-tuples so type-checkers do not confuse
    them with mathutils.Vector's descriptor ``__set__`` signature.
    """

    co_a: tuple[float, float, float]
    co_b: tuple[float, float, float]
    face_ids: frozenset[int]
    attrs: tuple[bool, bool, float, float]

    def endpoint_vectors(self) -> tuple[Vector, Vector]:
        return Vector(self.co_a), Vector(self.co_b)


def _extract_connect_effect_edges(bm: bmesh.types.BMesh) -> tuple[_ConnectEffectEdge, ...]:
    """R = newly generated edges after native connect.

    Novelty is tag==0 or FACE_ID complement, same as Knife. Reused
    pre-existing edges keep a non-zero parent tag and are excluded.
    """

    path_edges = core._discover_path_edges(bm, selected_only=False)
    face_layer = bm.faces.layers.int.get(core.FACE_ID_LAYER)
    records: list[_ConnectEffectEdge] = []
    for edge in path_edges:
        if face_layer is None:
            face_ids: frozenset[int] = frozenset()
        else:
            face_ids = frozenset(int(face[face_layer]) for face in edge.link_faces)
        co_a = edge.verts[0].co
        co_b = edge.verts[1].co
        records.append(
            _ConnectEffectEdge(
                (float(co_a[0]), float(co_a[1]), float(co_a[2])),
                (float(co_b[0]), float(co_b[1]), float(co_b[2])),
                face_ids,
                _edge_attr_tuple(bm, edge),
            )
        )
    return tuple(records)


def _extract_connect_r_coords(bm: bmesh.types.BMesh) -> tuple[tuple[Vector, Vector], ...]:
    """Coordinate-only view of R (for ρ(R) multiset matching)."""

    return tuple(record.endpoint_vectors() for record in _extract_connect_effect_edges(bm))


def _edge_coords_match(
    first_a: Vector,
    first_b: Vector,
    second_a: Vector,
    second_b: Vector,
    tolerance: float,
) -> bool:
    return (
        core.coordinates_match(first_a, second_a, tolerance) and core.coordinates_match(first_b, second_b, tolerance)
    ) or (core.coordinates_match(first_a, second_b, tolerance) and core.coordinates_match(first_b, second_a, tolerance))


def _quantize_co_key(co: Vector, tolerance: float) -> tuple[float, float, float]:
    scale = max(tolerance, 1.0e-12)
    return (
        round(round(float(co[0]) / scale) * scale, 12),
        round(round(float(co[1]) / scale) * scale, 12),
        round(round(float(co[2]) / scale) * scale, 12),
    )


def _build_face_id_pairs(
    bm: bmesh.types.BMesh,
    axis_index: int,
    tolerance: float,
) -> dict[int, int] | None:
    """Pre-state FACE_ID → mirror FACE_ID (involutive). ``None`` if incomplete."""

    face_layer = bm.faces.layers.int.get(core.FACE_ID_LAYER)
    if face_layer is None:
        return {}
    by_key: dict[frozenset[tuple[float, float, float]], list[int]] = {}
    face_keys: dict[int, frozenset[tuple[float, float, float]]] = {}
    for face in bm.faces:
        face_id = int(face[face_layer])
        key = frozenset(_quantize_co_key(vertex.co, tolerance) for vertex in face.verts)
        face_keys[face_id] = key
        by_key.setdefault(key, []).append(face_id)

    pairs: dict[int, int] = {}
    used: set[int] = set()
    for face_id in sorted(face_keys):
        if face_id in used:
            continue
        key = face_keys[face_id]
        mirrored_key = frozenset(
            _quantize_co_key(core.mirror_coordinate(Vector(co), axis_index), tolerance) for co in key
        )
        candidates = by_key.get(mirrored_key, [])
        if not candidates:
            return None
        partner: int | None = None
        if key == mirrored_key and face_id in candidates:
            partner = face_id
        else:
            for candidate in sorted(candidates):
                if candidate not in used:
                    partner = candidate
                    break
        if partner is None:
            return None
        pairs[face_id] = partner
        pairs[partner] = face_id
        used.add(face_id)
        used.add(partner)
    return pairs


def _attrs_match(
    first: tuple[bool, bool, float, float],
    second: tuple[bool, bool, float, float],
    tolerance: float,
) -> bool:
    return (
        first[0] == second[0]
        and first[1] == second[1]
        and abs(first[2] - second[2]) <= tolerance
        and abs(first[3] - second[3]) <= tolerance
    )


def _connect_effect_is_self_mirrored(
    r_edges: Sequence[_ConnectEffectEdge],
    face_id_pairs: dict[int, int] | None,
    axis_index: int,
    tolerance: float,
) -> bool:
    """True when ρ(R) equals R with 1:1 cancellation.

    Each edge must have a unique mirror partner in R whose link-face FACE_IDs
    correspond under the pre-state face-pair map and whose edge attributes
    match. Coordinate-only multiset equality is insufficient when several edges
    share endpoints.
    """

    if not r_edges:
        return True
    if face_id_pairs is None:
        return False

    unmatched = set(range(len(r_edges)))
    while unmatched:
        index = min(unmatched)
        unmatched.remove(index)
        record = r_edges[index]
        co_a, co_b = record.endpoint_vectors()
        mirrored_a = core.mirror_coordinate(co_a, axis_index)
        mirrored_b = core.mirror_coordinate(co_b, axis_index)
        mapped_faces: list[int] = []
        for face_id in record.face_ids:
            partner_face = face_id_pairs.get(face_id)
            if partner_face is None:
                return False
            mapped_faces.append(partner_face)
        expected_faces = Counter(mapped_faces)

        partner_index: int | None = None
        # Self-mirrored edge may pair with itself (checked first).
        for other_index in (index, *sorted(unmatched)):
            other = r_edges[other_index]
            other_a, other_b = other.endpoint_vectors()
            if not _edge_coords_match(mirrored_a, mirrored_b, other_a, other_b, tolerance):
                continue
            if Counter(other.face_ids) != expected_faces:
                continue
            if not _attrs_match(record.attrs, other.attrs, tolerance):
                continue
            partner_index = other_index
            break
        if partner_index is None:
            return False
        if partner_index != index:
            unmatched.remove(partner_index)
    return True


def _find_vert_at(
    bm: bmesh.types.BMesh,
    coordinate: Vector,
    tolerance: float,
) -> bmesh.types.BMVert | None:
    for vertex in bm.verts:
        if core.coordinates_match(vertex.co, coordinate, tolerance):
            return vertex
    return None


def _verts_share_edge(first: bmesh.types.BMVert, second: bmesh.types.BMVert) -> bool:
    return any(edge.other_vert(first) is second for edge in first.link_edges)


def _point_on_segment_and_plane(
    point: Vector,
    endpoint_a: Vector,
    endpoint_b: Vector,
    axis_index: int,
    tolerance: float,
) -> bool:
    """True when *point* lies on segment A–B, is collinear, and on the mirror plane."""

    if abs(float(point[axis_index])) > tolerance:
        return False
    # Euclidean point-to-segment (geometric, not Chebyshev identity).
    delta = endpoint_b - endpoint_a
    length_squared = float(delta.length_squared)
    if length_squared <= 1.0e-30:
        return core.coordinates_match(point, endpoint_a, tolerance)
    factor = (point - endpoint_a).dot(delta) / length_squared
    if factor < -1.0e-9 or factor > 1.0 + 1.0e-9:
        return False
    factor = max(0.0, min(1.0, factor))
    closest = endpoint_a + factor * delta
    return (point - closest).length <= tolerance


def _vertex_coords_snapshot(bm: bmesh.types.BMesh) -> tuple[Vector, ...]:
    bm.verts.ensure_lookup_table()
    return tuple(vertex.co.copy() for vertex in bm.verts)


def _is_new_vertex(
    coordinate: Vector,
    pre_coords: Sequence[Vector],
    tolerance: float,
) -> bool:
    return all(not core.coordinates_match(coordinate, previous, tolerance) for previous in pre_coords)


def _match_edge_in_pool(
    co_a: Vector,
    co_b: Vector,
    pool: list[tuple[Vector, Vector]],
    tolerance: float,
) -> int | None:
    for index, (other_a, other_b) in enumerate(pool):
        if _edge_coords_match(co_a, co_b, other_a, other_b, tolerance):
            return index
    return None


def _match_p_stitch_in_pool(
    co_a: Vector,
    co_b: Vector,
    pool: list[tuple[Vector, Vector]],
    pre_second_coords: Sequence[Vector],
    axis_index: int,
    tolerance: float,
) -> tuple[int, int] | None:
    """Find exactly two R′ edges A–p, p–B with *p* newly created on segment A–B."""

    n = len(pool)
    for i in range(n):
        for j in range(i + 1, n):
            first = pool[i]
            second = pool[j]
            for p_candidate, other_first in (
                (first[0], first[1]),
                (first[1], first[0]),
            ):
                for p_other, other_second in (
                    (second[0], second[1]),
                    (second[1], second[0]),
                ):
                    if not core.coordinates_match(p_candidate, p_other, tolerance):
                        continue
                    p = p_candidate
                    ab_match = (
                        core.coordinates_match(other_first, co_a, tolerance)
                        and core.coordinates_match(other_second, co_b, tolerance)
                    ) or (
                        core.coordinates_match(other_first, co_b, tolerance)
                        and core.coordinates_match(other_second, co_a, tolerance)
                    )
                    if not ab_match:
                        continue
                    if not _is_new_vertex(p, pre_second_coords, tolerance):
                        continue
                    if not _point_on_segment_and_plane(p, co_a, co_b, axis_index, tolerance):
                        continue
                    return (i, j)
    return None


def _preexisting_realization(
    bm: bmesh.types.BMesh,
    co_a: Vector,
    co_b: Vector,
    pre_second_coords: Sequence[Vector],
    axis_index: int,
    tolerance: float,
) -> bool:
    """True when A–B is already realized without consuming R′.

    Allowed forms:
    - a direct edge A–B
    - A–p–B where *p* already existed before the second native (so the stitch
      came from the first native / was already in R ∩ ρ(R)), is on-plane, and
      lies on segment A–B. Arbitrary unrelated on-plane vertices are rejected.
    """

    vertex_a = _find_vert_at(bm, co_a, tolerance)
    vertex_b = _find_vert_at(bm, co_b, tolerance)
    if vertex_a is None or vertex_b is None:
        return False
    if _verts_share_edge(vertex_a, vertex_b):
        return True
    for edge in vertex_a.link_edges:
        mid = edge.other_vert(vertex_a)
        if _is_new_vertex(mid.co, pre_second_coords, tolerance):
            continue  # new p must be claimed via the R′ pool, not here
        if not _point_on_segment_and_plane(mid.co, co_a, co_b, axis_index, tolerance):
            continue
        if _verts_share_edge(mid, vertex_b):
            return True
    return False


def _verify_connect_mirror_effect(
    bm: bmesh.types.BMesh,
    r_edges: Sequence[tuple[Vector, Vector]],
    pre_second_coords: Sequence[Vector],
    axis_index: int,
    tolerance: float,
) -> bool:
    """ρ(R) ↔ R′ multiset match with pre-existing cancellation.

    R′ is re-extracted after the second native (tag==0 + FACE_ID complement).
    Each non-self-mirrored ρ(e) must be realized by exactly one of:

    1. a direct edge in R′
    2. a new-p stitch A–p–B using exactly two R′ edges (p created by the
       second native, on-plane, on segment A–B)
    3. a pre-existing realization from the first native (direct edge or
       pre-existing p-stitch) — required when R ∩ ρ(R) is non-empty (zig-zag)

    Any leftover R′ edge is excess generation and fails verification.
    """

    r_prime = list(_extract_connect_r_coords(bm))
    pool: list[tuple[Vector, Vector]] = list(r_prime)

    for co_a, co_b in r_edges:
        mirrored_a = core.mirror_coordinate(co_a, axis_index)
        mirrored_b = core.mirror_coordinate(co_b, axis_index)
        if core.coordinates_match(co_a, mirrored_b, tolerance) and core.coordinates_match(
            co_b,
            mirrored_a,
            tolerance,
        ):
            # Self-mirrored edge: already realized by the first native; must
            # still exist as a direct edge (not required to appear in R′).
            vertex_a = _find_vert_at(bm, co_a, tolerance)
            vertex_b = _find_vert_at(bm, co_b, tolerance)
            if vertex_a is None or vertex_b is None or not _verts_share_edge(vertex_a, vertex_b):
                return False
            continue

        direct = _match_edge_in_pool(mirrored_a, mirrored_b, pool, tolerance)
        if direct is not None:
            pool.pop(direct)
            continue

        stitch = _match_p_stitch_in_pool(
            mirrored_a,
            mirrored_b,
            pool,
            pre_second_coords,
            axis_index,
            tolerance,
        )
        if stitch is not None:
            for index in sorted(stitch, reverse=True):
                pool.pop(index)
            continue

        # Already present from first native (R ∩ ρ(R) or reused pre-edge).
        if _preexisting_realization(
            bm,
            mirrored_a,
            mirrored_b,
            pre_second_coords,
            axis_index,
            tolerance,
        ):
            continue

        return False  # missing mirror realization

    # Excess generation: any unmatched R′ edge fails.
    return not pool


class MESH_OT_ydd_symmetric_edit_connect(bpy.types.Operator):
    """Run Vertex Connect on the selected path and its mirrored path."""

    bl_idname = "mesh.ydd_symmetric_edit_connect"
    bl_label = "Symmetric Vertex Connect"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return context.mode == "EDIT_MESH" and context.edit_object is not None

    def execute(self, context):
        global _CONNECT_LAST_R
        _CONNECT_REPORTS.clear()
        _CONNECT_LAST_R = ()

        symmetry = _symmetry_parameters(context)
        if symmetry is None:
            return _native_vert_connect_path()
        obj, axis_index, tolerance = symmetry
        mesh = cast(bpy.types.Mesh, obj.data)

        # Knife (etc.) session in progress: do not touch temporary layers.
        try:
            from . import operators as _operators

            sessions_active = bool(_operators._SESSIONS)
        except Exception:
            sessions_active = False
        if sessions_active:
            _connect_report(
                self,
                {"INFO"},
                "A cut-tool session is active; ran the native connect only",
            )
            return _native_vert_connect_path()

        bm = bmesh.from_edit_mesh(mesh)
        coords, selected_indices, history_coords, history_indices = _vertex_snapshot(bm, mesh_object=obj)
        snapshot = classify_mirror_selection(
            coords,
            selected_indices,
            axis_index=axis_index,
            tolerance=tolerance,
        )
        mirrored_history = _map_history_via_pairs(history_indices, snapshot.pairs)

        # Stamp markers before native so R is post-hoc recoverable.
        try:
            _prepare_connect_markers(bm)
            face_id_pairs = _build_face_id_pairs(bm, axis_index, tolerance)
            bmesh.update_edit_mesh(mesh, loop_triangles=False, destructive=False)
        except Exception:
            traceback.print_exc()
            try:
                bm = bmesh.from_edit_mesh(mesh)
                _remove_connect_markers(bm)
                bmesh.update_edit_mesh(mesh, loop_triangles=False, destructive=False)
            except Exception:
                traceback.print_exc()
            return _native_vert_connect_path()

        result = _native_vert_connect_path()
        if "FINISHED" not in result:
            try:
                bm = bmesh.from_edit_mesh(mesh)
                _remove_connect_markers(bm)
                bmesh.update_edit_mesh(mesh, loop_triangles=False, destructive=False)
            except Exception:
                traceback.print_exc()
            return result

        # Native has mutated the mesh — every post-native path (including
        # exceptions) must return `result` so the undo push is kept.
        backup_mesh = None
        mirror_warning = None
        mirror_level = "WARNING"
        backup_creation_failed = False
        rollback_failed = False
        try:
            bm = bmesh.from_edit_mesh(mesh)
            r_records = _extract_connect_effect_edges(bm)
            r_edges = tuple(record.endpoint_vectors() for record in r_records)
            _CONNECT_LAST_R = r_edges

            # Empty R (all-reuse / EDGE-mode silent no-op) is WARNING, not a
            # silent self-mirror success.
            if not r_records:
                _connect_report(self, {"WARNING"}, "native connect created no edges")
                return result

            # Mirror no-op only when the *effect* R is self-mirrored (incidence
            # + edge attributes, 1:1 cancellation).
            if _connect_effect_is_self_mirrored(r_records, face_id_pairs, axis_index, tolerance):
                _report_self_mirrored(self)
                _maybe_extend_selection_to_mirror(mesh, axis_index, tolerance, mesh_object=obj)
                return result

            # Counterpart missing → legitimate decline (native kept + WARNING).
            if mirrored_history is None:
                _report_missing(self, max(1, len(snapshot.missing)), partial=False)
                return result

            mirrored_history_coords = tuple(
                Vector((float(coords[index][0]), float(coords[index][1]), float(coords[index][2])))
                for index in mirrored_history
            )
            selected_coords = tuple(
                Vector((float(coords[index][0]), float(coords[index][1]), float(coords[index][2])))
                for index in selected_indices
            )

            try:
                backup_mesh = backup.create_topology_backup(bm)
            except Exception as exc:
                traceback.print_exc()
                backup_creation_failed = True
                mirror_warning = f"Could not create topology backup for mirrored connect: {exc}"
                mirror_level = "ERROR"
                raise

            try:
                bm = bmesh.from_edit_mesh(mesh)
                if not _set_vertex_path(
                    bm,
                    mirrored_history_coords,
                    axis_index,
                    tolerance,
                ):
                    mirror_warning = "Mirrored connect path changed after native execution; mirrored connect skipped"
                else:
                    # Snapshot verts before second native (for p-newness).
                    pre_second_coords = _vertex_coords_snapshot(bm)
                    # Re-stamp so the second call's novelty set is R′ only.
                    _remark_connect_markers(bm)
                    bmesh.update_edit_mesh(mesh, loop_triangles=False, destructive=False)
                    second_result = _native_vert_connect_path()
                    bm = bmesh.from_edit_mesh(mesh)
                    # CANCELLED → always rollback regardless of side effects.
                    if "CANCELLED" in second_result and "FINISHED" not in second_result:
                        backup.restore_topology_backup(mesh, backup_mesh)
                        mirror_warning = "Mirrored connect returned CANCELLED; rolled back to the native connect only"
                    elif not _verify_connect_mirror_effect(
                        bm,
                        r_edges,
                        pre_second_coords,
                        axis_index,
                        tolerance,
                    ):
                        backup.restore_topology_backup(mesh, backup_mesh)
                        mirror_warning = (
                            "Mirrored connect effect did not match the expected mirror; "
                            "rolled back to the native connect only"
                        )
            except Exception:
                traceback.print_exc()
                try:
                    backup.restore_topology_backup(mesh, backup_mesh)
                    mirror_warning = mirror_warning or "Unexpected error; the mirrored connect was rolled back"
                except Exception:
                    rollback_failed = True
                    traceback.print_exc()
                    mirror_warning = mirror_warning or "Internal error during the mirrored connect"
                    mirror_level = "ERROR"

            # Selection restore is best effort (signature compare is post-normalize).
            try:
                bm = bmesh.from_edit_mesh(mesh)
                if _set_vertex_path(
                    bm,
                    history_coords,
                    axis_index,
                    tolerance,
                    selected_coords=selected_coords,
                ):
                    bmesh.update_edit_mesh(mesh, loop_triangles=False, destructive=False)
                else:
                    mirror_warning = (
                        mirror_warning or "Could not restore the original selection after the mirrored connect"
                    )
            except Exception:
                traceback.print_exc()
                mirror_warning = mirror_warning or "Could not restore the original selection after the mirrored connect"
        except Exception:
            if not backup_creation_failed:
                traceback.print_exc()
            mirror_warning = mirror_warning or "Internal error during the mirrored connect"
            if backup_creation_failed or rollback_failed:
                mirror_level = "ERROR"
        finally:
            try:
                bm = bmesh.from_edit_mesh(mesh)
                # Backup ID layer + Connect markers; best-effort on every path.
                core.remove_temporary_layers(bm)
                bmesh.update_edit_mesh(mesh, loop_triangles=False, destructive=False)
            except Exception:
                traceback.print_exc()
            backup.remove_backup(backup_mesh)

        try:
            if mirror_warning:
                if backup_creation_failed or rollback_failed:
                    mirror_level = "ERROR"
                _connect_report(self, {mirror_level}, mirror_warning)
            else:
                # Mirror stage completed (or was a pure no-op success path).
                # Selection was restored to the native source path above; extend
                # now so ρ(S) is add-selected without touching select_history.
                _maybe_extend_selection_to_mirror(mesh, axis_index, tolerance, mesh_object=obj)
        except Exception:
            traceback.print_exc()
        return result


class MESH_OT_ydd_symmetric_edit_merge(bpy.types.Operator):
    """Run a native merge and deterministically merge its mirrored clusters."""

    bl_idname = "mesh.ydd_symmetric_edit_merge"
    bl_label = "Symmetric Merge Vertices"
    bl_options = {"REGISTER", "UNDO"}

    if TYPE_CHECKING:
        mode: str
        threshold: float
    else:
        mode: EnumProperty(
            name="Merge Type",
            items=(
                ("CENTER", "At Center", "Merge at the selection center"),
                ("COLLAPSE", "Collapse", "Merge each connected selection island"),
                ("FIRST", "At First", "Merge at the first vertex in selection history"),
                ("LAST", "At Last", "Merge at the last vertex in selection history"),
                ("BY_DISTANCE", "By Distance", "Merge selected vertices within the distance"),
            ),
            default="CENTER",
        )
        threshold: FloatProperty(
            name="Merge Distance",
            description="Maximum distance between vertices merged by distance",
            default=0.0001,
            min=0.0,
            max=50.0,
            precision=6,
            subtype="DISTANCE",
        )

    @classmethod
    def poll(cls, context):
        return context.mode == "EDIT_MESH" and context.edit_object is not None

    def _native(self) -> set[str]:
        if self.mode == "BY_DISTANCE":
            return cast(set[str], bpy.ops.mesh.remove_doubles(threshold=self.threshold))
        if self.mode == "CENTER":
            return cast(set[str], bpy.ops.mesh.merge(type="CENTER"))
        if self.mode == "COLLAPSE":
            return cast(set[str], bpy.ops.mesh.merge(type="COLLAPSE"))
        if self.mode == "FIRST":
            return cast(set[str], bpy.ops.mesh.merge(type="FIRST"))
        return cast(set[str], bpy.ops.mesh.merge(type="LAST"))

    def execute(self, context):
        _MERGE_REPORTS.clear()
        symmetry = _symmetry_parameters(context)
        if symmetry is None:
            return self._native()
        obj, axis_index, tolerance = symmetry
        mesh = cast(bpy.types.Mesh, obj.data)
        bm = bmesh.from_edit_mesh(mesh)
        coords, selected_indices, history_coords, history_indices = _vertex_snapshot(bm, mesh_object=obj)
        snapshot = classify_mirror_selection(
            coords,
            selected_indices,
            axis_index=axis_index,
            tolerance=tolerance,
        )
        if self.mode == "BY_DISTANCE":
            _report_missing(self, len(snapshot.missing), partial=True)
            bm.verts.ensure_lookup_table()
            for index in snapshot.mirrors:
                bm.verts[index].select = True
            bmesh.update_edit_mesh(mesh, loop_triangles=False, destructive=False)
            result = self._native()
            if "FINISHED" in result:
                _maybe_extend_selection_to_mirror(mesh, axis_index, tolerance, mesh_object=obj)
            return result

        if self.mode not in _MERGE_MODES:
            return self._native()
        if self.mode in {"FIRST", "LAST"} and not history_indices:
            return self._native()

        if snapshot.overlap is MirrorOverlap.DISJOINT:
            # A both-sides selection whose mirror image does not intersect the
            # selection replays correctly, exactly like a one-side selection.
            return self._execute_disjoint_replay(
                mesh,
                bm,
                axis_index,
                tolerance,
                snapshot,
                coords,
                selected_indices,
                history_coords,
                mesh_object=obj,
            )

        if not snapshot.complete:
            # Overlapping selection with missing or ambiguous counterparts:
            # e.g. a complete pair plus one counterpart-less vertex would still
            # contribute to a CENTER target, so one native run cannot be made
            # symmetric and symmetrizing cannot help either.
            _report_partial_overlap(self, "merge")
            return self._native()

        if self.mode in {"CENTER", "COLLAPSE"}:
            if snapshot.overlap is MirrorOverlap.SELF_MIRRORED:
                # CENTER's centroid lies on the plane; COLLAPSE's islands come
                # in symmetric pairs.  One native run is already symmetric.
                _report_self_mirrored(self, _merge_report)
                result = self._native()
                if "FINISHED" in result:
                    _maybe_extend_selection_to_mirror(mesh, axis_index, tolerance, mesh_object=obj)
                return result
            # PARTIAL (complete): symmetrize the selection to reduce it to the
            # self-mirrored case, then run the native merge once.
            added = self._symmetrize_selection(bm, mesh, snapshot)
            _merge_report(
                self,
                {"INFO"},
                f"Added {added} mirrored vertex(es) to the selection to keep the merge symmetric",
            )
            result = self._native()
            if "FINISHED" in result:
                _maybe_extend_selection_to_mirror(mesh, axis_index, tolerance, mesh_object=obj)
            return result

        return self._execute_side_split_merge(
            mesh,
            bm,
            axis_index,
            tolerance,
            snapshot,
            coords,
            history_indices,
            mesh_object=obj,
        )

    def _symmetrize_selection(self, bm: bmesh.types.BMesh, mesh, snapshot: MirrorSelection) -> int:
        """Also select ρ(U) for the unpaired-in-selection part U (5-2/5-3)."""

        bm.verts.ensure_lookup_table()
        added = 0
        for index in sorted(snapshot.off):
            partner = snapshot.pairs.get(index)
            if partner is None or partner in snapshot.selected:
                continue
            vertex = bm.verts[partner]
            if not vertex.select:
                vertex.select = True
                added += 1
        bmesh.update_edit_mesh(mesh, loop_triangles=False, destructive=False)
        return added

    def _restore_selection_state(
        self,
        bm: bmesh.types.BMesh,
        mesh,
        selected: frozenset[int],
        touched: set[int],
        history_indices: Sequence[int],
    ) -> None:
        """Put selection flags and history back to the pre-native snapshot."""

        bm.verts.ensure_lookup_table()
        for index in sorted(touched):
            if index < len(bm.verts):
                bm.verts[index].select = index in selected
        bm.select_history.clear()
        for index in history_indices:
            if index < len(bm.verts):
                bm.select_history.add(bm.verts[index])
        bmesh.update_edit_mesh(mesh, loop_triangles=False, destructive=False)

    def _execute_disjoint_replay(
        self,
        mesh,
        bm: bmesh.types.BMesh,
        axis_index: int,
        tolerance: float,
        snapshot: MirrorSelection,
        coords: VertexCoordArray,
        selected_indices: tuple[int, ...],
        history_coords: tuple[Vector, ...],
        *,
        mesh_object=None,
    ) -> set[str]:
        _report_missing(self, len(snapshot.missing), partial=True)

        clusters = split_merge_clusters(bm, selected_indices, self.mode)
        targets = tuple(
            calculate_merge_target(
                cluster,
                coords,
                self.mode,
                history_coords=history_coords,
            )
            for cluster in clusters
        )
        expected_mirror_counts = tuple(
            sum(1 for index in cluster if index in snapshot.mirror_by_source) for cluster in clusters
        )
        # Pre-native cluster sizes: used after native to distinguish a full
        # no-op (survivors == size) from a partial in-cluster merge
        # (1 < survivors < size).
        cluster_sizes = {number: len(cluster) for number, cluster in enumerate(clusters, start=1)}

        # Mark members (+k) and their mirrors (-k) in a temporary layer so the
        # post-native re-identification is exact: a coordinate lookup would
        # miscount whenever an unrelated vertex sits within tolerance of an
        # old member position.  The survivor of a native
        # merge inherits its member's marker.  Layer creation invalidates
        # wrappers, hence the fresh lookup table.
        group_layer = bm.verts.layers.int.get(core.VERT_MERGE_GROUP_LAYER)
        if group_layer is not None:
            bm.verts.layers.int.remove(group_layer)
        group_layer = bm.verts.layers.int.new(core.VERT_MERGE_GROUP_LAYER)
        bm.verts.ensure_lookup_table()
        for cluster_number, cluster in enumerate(clusters, start=1):
            for index in cluster:
                bm.verts[index][group_layer] = cluster_number
            for index in cluster:
                mirror_index = snapshot.mirror_by_source.get(index)
                if mirror_index is not None:
                    bm.verts[mirror_index][group_layer] = -cluster_number
        bmesh.update_edit_mesh(mesh, loop_triangles=False, destructive=False)

        result = self._native()
        if "FINISHED" not in result:
            try:
                bm = bmesh.from_edit_mesh(mesh)
                core.remove_temporary_layers(bm)
                bmesh.update_edit_mesh(mesh, loop_triangles=False, destructive=False)
            except Exception:
                traceback.print_exc()
            return result

        # The native mesh is mutated: every path below returns `result`.
        backup_mesh = None
        mirror_warning = None
        # Contract §3.2: any cluster-level mirror decline (WARNING skip) makes
        # the whole op a partial failure → do not run Select Mirrored extend.
        cluster_declined = False
        try:
            bm = bmesh.from_edit_mesh(mesh)
            backup_mesh = backup.create_topology_backup(bm)
            try:
                bm = bmesh.from_edit_mesh(mesh)
                group_layer = bm.verts.layers.int.get(core.VERT_MERGE_GROUP_LAYER)
                if group_layer is None:
                    raise RuntimeError("the merge group markers were lost during the native merge")
                member_survivors: dict[int, list[bmesh.types.BMVert]] = {}
                mirror_verts_by_cluster: dict[int, list[bmesh.types.BMVert]] = {}
                for vertex in bm.verts:
                    value = int(vertex[group_layer])
                    if value > 0:
                        member_survivors.setdefault(value, []).append(vertex)
                    elif value < 0:
                        mirror_verts_by_cluster.setdefault(-value, []).append(vertex)

                for cluster_number, (target, expected_mirrors) in enumerate(
                    zip(targets, expected_mirror_counts, strict=True),
                    start=1,
                ):
                    survivors = member_survivors.get(cluster_number, [])
                    # Native Collapse can leave a cluster untouched (e.g. a
                    # selection island on a fully detached component); all its
                    # marked members then still exist.  Mirroring such a
                    # cluster would merge geometry the native operator never
                    # merged.  A partial in-cluster merge (some but not all
                    # members consumed) is a cluster-level decline and must
                    # be visible.
                    if len(survivors) > 1:
                        original_size = cluster_sizes.get(cluster_number, 0)
                        if original_size and len(survivors) < original_size:
                            _merge_report(
                                self,
                                {"WARNING"},
                                "native merged this cluster only partially; its mirror was skipped",
                            )
                            cluster_declined = True
                        continue
                    mirrors = [vertex for vertex in mirror_verts_by_cluster.get(cluster_number, ()) if vertex.is_valid]
                    if not mirrors:
                        if expected_mirrors:
                            _merge_report(
                                self,
                                {"WARNING"},
                                "Mirror merge skipped for one cluster: its mirrored vertices could not be re-identified",
                            )
                            cluster_declined = True
                        continue
                    if abs(target[axis_index]) <= tolerance:
                        # A merge landing on the plane welds the mirrored
                        # cluster into the same surviving vertex so the mesh
                        # stays connected.
                        survivor = survivors[0] if survivors and survivors[0].is_valid else None
                        if survivor is None:
                            _merge_report(
                                self,
                                {"WARNING"},
                                "Mirror merge skipped for one cluster: the on-plane survivor could not be identified",
                            )
                            cluster_declined = True
                            continue
                        mirrors.append(survivor)
                        merge_co = target
                    else:
                        merge_co = core.mirror_coordinate(target, axis_index)
                    unique_verts = list(dict.fromkeys(vertex for vertex in mirrors if vertex.is_valid))
                    bmesh.ops.pointmerge(bm, verts=unique_verts, merge_co=merge_co)

                bmesh.update_edit_mesh(mesh, loop_triangles=True, destructive=True)
            except Exception:
                # Roll back to the post-native state; no partially mirrored
                # cluster subset survives.
                traceback.print_exc()
                backup.restore_topology_backup(mesh, backup_mesh)
                mirror_warning = "Unexpected error; the mirrored merge was rolled back"
        except Exception:
            traceback.print_exc()
            mirror_warning = mirror_warning or "Internal error during the mirrored merge"
        finally:
            try:
                bm = bmesh.from_edit_mesh(mesh)
                core.remove_temporary_layers(bm)
                bmesh.update_edit_mesh(mesh, loop_triangles=False, destructive=False)
            except Exception:
                traceback.print_exc()
            backup.remove_backup(backup_mesh)
        try:
            if mirror_warning:
                _merge_report(self, {"WARNING"}, mirror_warning)
            elif "FINISHED" in result and not cluster_declined:
                _maybe_extend_selection_to_mirror(mesh, axis_index, tolerance, mesh_object=mesh_object)
        except Exception:
            traceback.print_exc()
        return result

    def _execute_side_split_merge(
        self,
        mesh,
        bm: bmesh.types.BMesh,
        axis_index: int,
        tolerance: float,
        snapshot: MirrorSelection,
        coords: VertexCoordArray,
        history_indices: tuple[int, ...],
        *,
        mesh_object=None,
    ) -> set[str]:
        """FIRST/LAST on a self-mirrored selection: merge each side to its own
        endpoint (5-2).  One native run would drag both sides to one point.

        On-plane vertices in the extended selection are shared by both side
        clusters but are **not** collapsed into either
        side's target: each side merges only its off-plane members, and the
        on-plane verts stay put so both survivors remain linked through them
        (sequential stand-in for "each cluster = off-plane ∪ on-plane").
        Feeding on-plane verts into the source-only native merge would absorb
        them asymmetrically and break edge X-symmetry.
        """

        # Step 1: fix the merge target from the ORIGINAL history, before any
        # selection changes; it must not move for the rest of the procedure.
        target_index = history_indices[0] if self.mode == "FIRST" else history_indices[-1]
        target_row = coords[target_index]
        target_co = Vector((float(target_row[0]), float(target_row[1]), float(target_row[2])))

        extended_selection = set(snapshot.selected)
        if snapshot.overlap is MirrorOverlap.PARTIAL:
            extended_selection |= {snapshot.pairs[index] for index in snapshot.off if index in snapshot.pairs}

        # Exception guards run BEFORE any selection mutation: a fallback to a
        # plain native run must never carry a half-applied symmetrization
        # (review counterexample: PARTIAL plus an on-plane vertex would
        # otherwise natively merge a vertex the user never selected).
        if abs(target_co[axis_index]) <= tolerance:
            # On-plane endpoint: one native run merges both sides onto the
            # plane point, which is already symmetric.  PARTIAL is symmetrized
            # first — by design — so its native result is symmetric too.
            if snapshot.overlap is MirrorOverlap.PARTIAL:
                added = self._symmetrize_selection(bm, mesh, snapshot)
                _merge_report(
                    self,
                    {"INFO"},
                    f"Added {added} mirrored vertex(es) to the selection to keep the merge symmetric",
                )
            _report_self_mirrored(self, _merge_report)
            result = self._native()
            if "FINISHED" in result:
                _maybe_extend_selection_to_mirror(mesh, axis_index, tolerance, mesh_object=mesh_object)
            return result

        # Step 2: side partition.  Off-plane verts go to the half-space of
        # their X sign; on-plane verts are shared (kept, not side-merged).
        source_sign = 1.0 if target_co[axis_index] > 0.0 else -1.0
        on_plane = {index for index in extended_selection if abs(coords[index][axis_index]) <= tolerance}
        # Off-plane only — on_plane is excluded from both merge sets so it
        # survives as the shared link between the two side survivors.
        # The exclusion must be explicit: a vertex with 0 < |x| <= tolerance is
        # on-plane yet has a nonzero sign, so a raw sign test alone would leak
        # it into a side merge.
        source_side = {
            index
            for index in extended_selection
            if index not in on_plane and coords[index][axis_index] * source_sign > 0.0
        }
        mirror_side = {
            index
            for index in extended_selection
            if index not in on_plane and coords[index][axis_index] * source_sign < 0.0
        }

        # Step 3: rebuild per-side history.  target_index is always the FIRST
        # or LAST entry of history_indices and, being the off-plane merge
        # target, always sits in source_side.  Therefore the rebuilt endpoint
        # equals target_index structurally (the old "rebuild failed → native
        # only" branch was unreachable).
        source_history = [index for index in history_indices if index in source_side]
        rebuilt_endpoint = None
        if source_history:
            rebuilt_endpoint = source_history[0] if self.mode == "FIRST" else source_history[-1]
        if rebuilt_endpoint != target_index:
            # Unreachable under Blender's own invariant (select_history is a
            # subset of the selection, so the off-plane target stays in
            # source_side).  Externally mutated histories can break that
            # invariant, so degrade gracefully instead of crashing: the
            # native merge runs on the original selection, nothing mutated.
            _merge_report(
                self,
                {"WARNING"},
                "Could not rebuild a per-side merge history; ran the native merge only",
            )
            return self._native()

        # All guards passed — now mutate.  PARTIAL symmetrization (always
        # reported), then group markers, then the source-side rebuild.
        if snapshot.overlap is MirrorOverlap.PARTIAL:
            added = self._symmetrize_selection(bm, mesh, snapshot)
            _merge_report(
                self,
                {"INFO"},
                f"Added {added} mirrored vertex(es) to the selection to keep the merge symmetric",
            )

        # Group markers (source=1, mirror=2) make the post-native
        # re-identification exact; a coordinate lookup could confuse
        # coincident vertices.  On-plane verts stay unmarked
        # (not part of either side merge).  Layer creation invalidates
        # wrappers, hence the fresh lookup table.
        group_layer = bm.verts.layers.int.get(core.VERT_MERGE_GROUP_LAYER)
        if group_layer is not None:
            bm.verts.layers.int.remove(group_layer)
        group_layer = bm.verts.layers.int.new(core.VERT_MERGE_GROUP_LAYER)
        bm.verts.ensure_lookup_table()
        for index in sorted(source_side):
            bm.verts[index][group_layer] = 1
        for index in sorted(mirror_side):
            bm.verts[index][group_layer] = 2

        # Deselecting alone does not remove a vertex from the history, so the
        # history is always rebuilt explicitly.  Mirror off-plane and on-plane
        # verts are deselected so native only sees the source off-plane set.
        for index in sorted(mirror_side | on_plane):
            bm.verts[index].select = False
        for index in sorted(source_side):
            bm.verts[index].select = True
        bm.select_history.clear()
        for index in source_history:
            bm.select_history.add(bm.verts[index])
        bmesh.update_edit_mesh(mesh, loop_triangles=False, destructive=False)

        # Step 4: the native merge now only sees the source side.
        result = self._native()
        if "FINISHED" not in result:
            self._restore_selection_state(bm, mesh, snapshot.selected, extended_selection, history_indices)
            try:
                bm = bmesh.from_edit_mesh(mesh)
                core.remove_temporary_layers(bm)
                bmesh.update_edit_mesh(mesh, loop_triangles=False, destructive=False)
            except Exception:
                traceback.print_exc()
            return result

        # Step 5-7: mirror-side deterministic merge inside the transaction.
        merge_co = core.mirror_coordinate(target_co, axis_index)
        backup_mesh = None
        mirror_warning = None
        mirror_committed = False
        try:
            bm = bmesh.from_edit_mesh(mesh)
            backup_mesh = backup.create_topology_backup(bm)
            try:
                # Wrappers were invalidated by the backup ID layer and the
                # native merge changed indices; both sides are re-identified
                # through the group markers (the source survivor inherits its
                # member's marker).
                bm = bmesh.from_edit_mesh(mesh)
                group_layer = bm.verts.layers.int.get(core.VERT_MERGE_GROUP_LAYER)
                if group_layer is None:
                    raise RuntimeError("the merge group markers were lost during the native merge")
                source_survivors = []
                mirror_verts = []
                for vertex in bm.verts:
                    value = int(vertex[group_layer])
                    if value == 1:
                        source_survivors.append(vertex)
                    elif value == 2:
                        mirror_verts.append(vertex)
                if len(source_survivors) != 1:
                    raise RuntimeError(f"expected one surviving source vertex, found {len(source_survivors)}")
                source_survivor = source_survivors[0]

                mirror_survivor = None
                if mirror_verts:
                    unique_verts = list(dict.fromkeys(mirror_verts))
                    bmesh.ops.pointmerge(bm, verts=unique_verts, merge_co=merge_co)
                    mirror_survivor = next((vertex for vertex in unique_verts if vertex.is_valid), None)

                # Step 6, post-state contract: both survivors selected, the
                # history holds the source survivor only (the native single
                # merge state, plus the mirrored side's survivor selected).
                if mirror_survivor is not None and mirror_survivor.is_valid:
                    mirror_survivor.select = True
                if source_survivor.is_valid:
                    source_survivor.select = True
                    bm.select_history.clear()
                    bm.select_history.add(source_survivor)
                bmesh.update_edit_mesh(mesh, loop_triangles=True, destructive=True)
                mirror_committed = True
            except Exception:
                # Step 7: roll back to the post-native state (source side
                # merged, mirror side untouched).
                traceback.print_exc()
                backup.restore_topology_backup(mesh, backup_mesh)
                mirror_warning = "Unexpected error; the mirrored merge was rolled back"
        except Exception:
            traceback.print_exc()
            mirror_warning = mirror_warning or "Internal error during the mirrored merge"
        finally:
            if not mirror_committed:
                # Post-state recovery on every failed path, including a failed
                # backup creation where no rollback ran: reselect the mirror
                # side, keep a source-only history.  The
                # marker layer exists on the live mesh and inside the restored
                # backup alike.
                self._reselect_side_split_groups(mesh)
            try:
                bm = bmesh.from_edit_mesh(mesh)
                core.remove_temporary_layers(bm)
                bmesh.update_edit_mesh(mesh, loop_triangles=False, destructive=False)
            except Exception:
                traceback.print_exc()
            backup.remove_backup(backup_mesh)
        try:
            if mirror_warning:
                _merge_report(self, {"WARNING"}, mirror_warning)
            else:
                side_label = "first" if self.mode == "FIRST" else "last"
                _merge_report(
                    self,
                    {"INFO"},
                    f"Merged each side to its own {side_label} vertex",
                )
                if "FINISHED" in result:
                    # Side-split already selects both survivors; this still
                    # covers any leftover selected non-survivor elements and is
                    # a no-op for already-selected mirror survivors.
                    _maybe_extend_selection_to_mirror(mesh, axis_index, tolerance, mesh_object=mesh_object)
        except Exception:
            traceback.print_exc()
        return result

    def _reselect_side_split_groups(self, mesh) -> None:
        """Best-effort post-failure reselect via the group markers; never raises."""

        try:
            bm = bmesh.from_edit_mesh(mesh)
            group_layer = bm.verts.layers.int.get(core.VERT_MERGE_GROUP_LAYER)
            if group_layer is None:
                return
            survivor = None
            for vertex in bm.verts:
                value = int(vertex[group_layer])
                if value == 2:
                    vertex.select = True
                elif value == 1:
                    vertex.select = True
                    survivor = vertex
            if survivor is not None:
                bm.select_history.clear()
                bm.select_history.add(survivor)
            bmesh.update_edit_mesh(mesh, loop_triangles=True, destructive=True)
        except Exception:
            traceback.print_exc()


class YSE_MT_merge(bpy.types.Menu):
    bl_idname = "YSE_MT_merge"
    bl_label = "Merge Vertices"

    def draw(self, context: bpy.types.Context) -> None:
        del context
        layout = self.layout
        if layout is None:
            return
        for mode, label in (
            ("CENTER", "At Center"),
            ("CURSOR", "At Cursor"),
            ("COLLAPSE", "Collapse"),
            ("FIRST", "At First"),
            ("LAST", "At Last"),
            ("BY_DISTANCE", "By Distance"),
        ):
            if mode == "CURSOR":
                operator = layout.operator("mesh.merge", text=label)
                operator.type = "CURSOR"
            else:
                operator = layout.operator(MESH_OT_ydd_symmetric_edit_merge.bl_idname, text=label)
                operator.mode = mode


CLASSES = (
    MESH_OT_ydd_symmetric_edit_connect,
    MESH_OT_ydd_symmetric_edit_merge,
    YSE_MT_merge,
)
