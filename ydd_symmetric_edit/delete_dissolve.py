# SPDX-License-Identifier: GPL-3.0-or-later

"""Symmetric delete / dissolve operators / menu, plus pure selection-expansion helpers."""

from __future__ import annotations

import math
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

import bmesh
import bpy
import numpy
from bpy.props import BoolProperty, EnumProperty, FloatProperty

from . import backup, element_pairs, layer_names, snapshot
from .element_pairs import ElementPairMaps, ExpansionPlan
from .gc_gate import gc_disabled_during_execute
from .replay import _symmetry_parameters
from .snapshot import capture_selection_snapshot

_DeleteType = Literal["VERT", "EDGE", "FACE", "EDGE_FACE", "ONLY_FACE"]
_DissolveMode = Literal["VERTS", "EDGES", "FACES"]


@dataclass(frozen=True)
class CollapseTracking:
    """Pre-native component identity retained across ``mesh.edge_collapse``."""

    group_ids: tuple[int, ...]
    original_vertices_by_group: dict[int, tuple[bmesh.types.BMVert, ...]]
    self_mirrored_groups: frozenset[int]
    mirror_group_by_group: dict[int, int | None]


# Test-visible Delete reports (Operator.report is not patchable; same pattern
# as replay._MERGE_REPORTS).
_DELETE_REPORTS: list[tuple[str, str]] = []

_TYPE_DOMAINS: dict[str, tuple[str, ...]] = {
    "VERT": ("VERT",),
    "EDGE": ("EDGE",),
    "FACE": ("FACE",),
    "ONLY_FACE": ("FACE",),
    "EDGE_FACE": ("EDGE", "FACE"),
}

_DELETE_TYPE_ITEMS = (
    ("VERT", "Vertices", "Delete selected vertices"),
    ("EDGE", "Edges", "Delete selected edges"),
    ("FACE", "Faces", "Delete selected faces"),
    ("EDGE_FACE", "Only Edges & Faces", "Delete selected edges and faces"),
    ("ONLY_FACE", "Only Faces", "Delete selected faces, keeping edges"),
)


def _delete_report(operator, level: set[str], message: str) -> None:
    """Report and record for Delete tests."""

    kind = "WARNING" if "WARNING" in level else "ERROR" if "ERROR" in level else "INFO"
    _DELETE_REPORTS.append((kind, message))
    operator.report(level, message)


def _sessions_active() -> bool:
    try:
        from . import operators as _operators

        return bool(_operators._SESSIONS)
    except Exception:
        return False


def _domains_for_type(delete_type: str) -> tuple[str, ...]:
    return _TYPE_DOMAINS.get(delete_type, ("VERT",))


def _restore_expansion_selection(bm: bmesh.types.BMesh, plan: ExpansionPlan) -> None:
    """Clear select flags added by *plan* (valid while topology is unchanged)."""

    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    for index in plan.add_vert_indices:
        if index < len(bm.verts):
            bm.verts[index].select = False
    for index in plan.add_edge_indices:
        if index < len(bm.edges):
            bm.edges[index].select = False
    for index in plan.add_face_indices:
        if index < len(bm.faces):
            bm.faces[index].select = False


_DISSOLVE_MODE_ITEMS = (
    ("VERTS", "Dissolve Vertices", "Dissolve vertices, merge edges and faces"),
    ("EDGES", "Dissolve Edges", "Dissolve edges, merging faces"),
    ("FACES", "Dissolve Faces", "Dissolve faces"),
)

_DISSOLVE_DOMAINS: dict[str, tuple[str, ...]] = {
    "VERTS": ("VERT",),
    "EDGES": ("EDGE",),
    "FACES": ("FACE",),
}


def _quantize_co_key(co, tolerance: float) -> tuple[float, float, float]:
    """Round-bin coordinate key used by the unmatched multiset census."""

    scale = max(float(tolerance), 1.0e-12)
    return (
        round(round(float(co[0]) / scale) * scale, 12),
        round(round(float(co[1]) / scale) * scale, 12),
        round(round(float(co[2]) / scale) * scale, 12),
    )


def _quantized_coordinate_keys(coords: numpy.ndarray, tolerance: float) -> numpy.ndarray:
    """Vector form of :func:`_quantize_co_key` over a float64 coordinate matrix."""

    scale = max(float(tolerance), 1.0e-12)
    scaled = numpy.asarray(coords, dtype=numpy.float64) / scale
    if not numpy.isfinite(scaled).all():
        # Preserve the scalar path's exception type/order for non-finite input.
        return numpy.asarray([_quantize_co_key(co, tolerance) for co in coords], dtype=numpy.float64)
    # numpy.round(x, 12) diverges from Python round(x, 12) beyond |x| ~ 5e3,
    # so representatives come from Python round applied to the few unique bins
    # — bit-identical to _quantize_co_key at vectorized cost.
    bins = numpy.rint(scaled)
    unique_bins, inverse = numpy.unique(bins, return_inverse=True)
    representatives = numpy.asarray(
        [round(value * scale, 12) for value in unique_bins.tolist()],
        dtype=numpy.float64,
    )
    keys = representatives[inverse].reshape(bins.shape)
    keys[keys == 0.0] = 0.0
    return keys


def _dense_partner_indices(mapping, count: int) -> numpy.ndarray:
    return numpy.fromiter(
        (-1 if (partner := mapping.get(index)) is None else int(partner) for index in range(count)),
        dtype=numpy.int64,
        count=count,
    )


def _census_topology_arrays(
    pair_maps: ElementPairMaps,
    bm: bmesh.types.BMesh,
) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    edge_vertices = pair_maps._edge_vertices
    loop_verts = pair_maps._loop_verts
    loop_starts = pair_maps._loop_starts
    loop_totals = pair_maps._loop_totals
    if edge_vertices is None or len(edge_vertices) != len(bm.edges):
        edge_vertices = element_pairs._edge_vertex_rows(bm, None)
    if loop_starts is None or len(loop_starts) != len(bm.faces):
        capture = capture_selection_snapshot(
            bm,
            domains=("FACE",),
            include_history=False,
            include_loops=True,
        )
        loop_verts = capture.loop_verts
        loop_starts = capture.loop_starts
        loop_totals = capture.loop_totals
    assert loop_verts is not None and loop_starts is not None and loop_totals is not None
    return edge_vertices, loop_verts, loop_starts, loop_totals


def _native_op_kwargs(op, options: dict) -> dict:
    """Keep only kwargs that the native operator RNA actually exposes.

    5.2-only dissolve_edges properties (angle_threshold / use_preserve_quads)
    must not be forwarded on 4.2, where the RNA lacks them.
    """

    try:
        rna = op.get_rna_type()
        allowed = {prop.identifier for prop in rna.properties if not prop.is_readonly and prop.identifier != "rna_type"}
    except Exception:
        return dict(options)
    return {key: value for key, value in options.items() if key in allowed}


def _native_dissolve_call(mode: str, options: dict) -> set[str]:
    """Invoke the matching native dissolve operator once.

    Module-level so fault-injection tests can wrap the call without touching
    bpy.ops registration (same pattern as ``replay._native_vert_connect_path``).
    """

    if mode == "VERTS":
        op = bpy.ops.mesh.dissolve_verts
        return cast(set[str], op(**_native_op_kwargs(op, options)))
    if mode == "EDGES":
        op = bpy.ops.mesh.dissolve_edges
        return cast(set[str], op(**_native_op_kwargs(op, options)))
    if mode == "FACES":
        op = bpy.ops.mesh.dissolve_faces
        return cast(set[str], op(**_native_op_kwargs(op, options)))
    raise ValueError(f"unknown dissolve mode: {mode!r}")


def _native_edge_collapse_call() -> set[str]:
    """Invoke native edge collapse once (module-level for fault injection)."""

    return cast(set[str], bpy.ops.mesh.edge_collapse())


def _native_delete_edgeloop_call(options: dict) -> set[str]:
    """Invoke native edge-loop deletion once (module-level for tests)."""

    op = bpy.ops.mesh.delete_edgeloop
    return cast(set[str], op(**_native_op_kwargs(op, options)))


def _symmetry_census(
    pair_maps: ElementPairMaps,
    bm: bmesh.types.BMesh,
    tolerance: float,
    *,
    mesh_object=None,
) -> tuple[Counter, Counter, Counter]:
    """Multiset of unmatched element signatures for post-operation verification.

    Verts: quantized coordinate keys of vertices absent from the pair table.
    Edges / faces: frozenset of endpoint (or loop) coordinate keys when the
    pair-map value is None.  Equality of the three Counters is the census
    check, so equal unmatched *counts* at different positions still fail.
    """

    capture = capture_selection_snapshot(
        bm,
        mesh_object=mesh_object,
        domains=(),
        include_history=False,
    )
    quantized = _quantized_coordinate_keys(capture.coords, tolerance)
    edge_vertices, loop_verts, loop_starts, loop_totals = _census_topology_arrays(pair_maps, bm)
    vertex_partners = pair_maps._vertex_partner_indices
    edge_partners = pair_maps._edge_partner_indices
    face_partners = pair_maps._face_partner_indices
    if vertex_partners is None or len(vertex_partners) != len(bm.verts):
        vertex_partners = _dense_partner_indices(pair_maps.vert_pairs, len(bm.verts))
    if edge_partners is None or len(edge_partners) != len(bm.edges):
        edge_partners = _dense_partner_indices(pair_maps.edge_pair_by_index, len(bm.edges))
    if face_partners is None or len(face_partners) != len(bm.faces):
        face_partners = _dense_partner_indices(pair_maps.face_pair_by_index, len(bm.faces))

    unmatched_verts: Counter = Counter(
        tuple(float(value) for value in quantized[index]) for index in numpy.flatnonzero(vertex_partners < 0)
    )
    unmatched_edges: Counter = Counter(
        frozenset(tuple(float(value) for value in quantized[vertex_id]) for vertex_id in edge_vertices[index])
        for index in numpy.flatnonzero(edge_partners < 0)
    )
    unmatched_faces: Counter = Counter()
    for index in numpy.flatnonzero(face_partners < 0):
        start = int(loop_starts[index])
        total = int(loop_totals[index])
        vertex_ids = loop_verts[start : start + total]
        key = frozenset(tuple(float(value) for value in quantized[vertex_id]) for vertex_id in vertex_ids)
        unmatched_faces[key] += 1

    return unmatched_verts, unmatched_edges, unmatched_faces


def _mark_collapse_components(
    bm: bmesh.types.BMesh,
    pair_maps: ElementPairMaps,
) -> CollapseTracking:
    """Mark visible selected-edge components and retain their pre-native identity."""

    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    selected_edge_indices = sorted(edge.index for edge in bm.edges if edge.select and not edge.hide)

    selected_edges_by_vertex: dict[int, list[int]] = defaultdict(list)
    for edge_index in selected_edge_indices:
        edge = bm.edges[edge_index]
        for vertex in edge.verts:
            selected_edges_by_vertex[vertex.index].append(edge_index)

    component_vertex_indices: list[tuple[int, ...]] = []
    visited_edges: set[int] = set()
    for start_index in selected_edge_indices:
        if start_index in visited_edges:
            continue
        pending = [start_index]
        component_vertices: set[int] = set()
        visited_edges.add(start_index)
        while pending:
            edge = bm.edges[pending.pop()]
            for vertex in edge.verts:
                component_vertices.add(vertex.index)
                for linked_index in selected_edges_by_vertex[vertex.index]:
                    if linked_index not in visited_edges:
                        visited_edges.add(linked_index)
                        pending.append(linked_index)
        component_vertex_indices.append(tuple(sorted(component_vertices)))

    group_by_vertex = {
        vertex_index: group_id
        for group_id, indices in enumerate(component_vertex_indices, start=1)
        for vertex_index in indices
    }
    self_mirrored_groups: set[int] = set()
    mirror_group_by_group: dict[int, int | None] = {}
    for group_id, indices in enumerate(component_vertex_indices, start=1):
        mirrored_groups = {
            group_by_vertex[partner]
            for vertex_index in indices
            if (partner := pair_maps.vert_pairs.get(vertex_index)) is not None and partner in group_by_vertex
        }
        is_closed = (
            len(mirrored_groups) == 1
            and group_id in mirrored_groups
            and all(
                (partner := pair_maps.vert_pairs.get(vertex_index)) is not None
                and group_by_vertex.get(partner) == group_id
                for vertex_index in indices
            )
        )
        if is_closed:
            self_mirrored_groups.add(group_id)
        mirror_group_by_group[group_id] = next(iter(mirrored_groups)) if len(mirrored_groups) == 1 else None

    old_layer = bm.verts.layers.int.get(layer_names.VERT_COLLAPSE_GROUP_LAYER)
    if old_layer is not None:
        bm.verts.layers.int.remove(old_layer)
    group_layer = bm.verts.layers.int.new(layer_names.VERT_COLLAPSE_GROUP_LAYER)
    # CustomData layer creation invalidates element wrappers. Rebuild the
    # lookup and only then retain references for post-native validity checks.
    bm.verts.ensure_lookup_table()
    original_vertices_by_group: dict[int, tuple[bmesh.types.BMVert, ...]] = {}
    for group_id, indices in enumerate(component_vertex_indices, start=1):
        marked_vertices = tuple(bm.verts[index] for index in indices)
        original_vertices_by_group[group_id] = marked_vertices
        for vertex in marked_vertices:
            vertex[group_layer] = group_id

    return CollapseTracking(
        group_ids=tuple(range(1, len(component_vertex_indices) + 1)),
        original_vertices_by_group=original_vertices_by_group,
        self_mirrored_groups=frozenset(self_mirrored_groups),
        mirror_group_by_group=mirror_group_by_group,
    )


def _collapse_survivors(
    bm: bmesh.types.BMesh,
    tracking: CollapseTracking,
) -> tuple[dict[int, bmesh.types.BMVert], bool]:
    """Recover one survivor per component, allowing marker loss on fusion."""

    group_layer = bm.verts.layers.int.get(layer_names.VERT_COLLAPSE_GROUP_LAYER)
    if group_layer is None:
        return {}, False

    survivors_by_group: dict[int, list[bmesh.types.BMVert]] = defaultdict(list)
    expected_groups = set(tracking.group_ids)
    for vertex in bm.verts:
        group_id = int(vertex[group_layer])
        if group_id > 0:
            if group_id not in expected_groups:
                return {}, False
            survivors_by_group[group_id].append(vertex)

    survivors: dict[int, bmesh.types.BMVert] = {}
    missing_groups: list[int] = []
    for group_id in tracking.group_ids:
        group_survivors = survivors_by_group.get(group_id, [])
        if len(group_survivors) > 1:
            return {}, False
        if group_survivors:
            survivors[group_id] = group_survivors[0]
            continue
        original_vertices = tracking.original_vertices_by_group[group_id]
        if any(vertex.is_valid for vertex in original_vertices):
            return {}, False
        missing_groups.append(group_id)

    # A missing marker is valid only when native fused that component into a
    # component whose marker survived. With no surviving marker there is no
    # identifiable fusion target.
    if missing_groups and not survivors:
        return {}, False
    return survivors, True


def _coordinates_are_mirrored(first, second, axis_index: int, tolerance: float) -> bool:
    for coordinate_index in range(3):
        expected = -float(first[coordinate_index]) if coordinate_index == axis_index else float(first[coordinate_index])
        if abs(expected - float(second[coordinate_index])) > tolerance:
            return False
    return True


def _validate_and_snap_collapse(
    bm: bmesh.types.BMesh,
    tracking: CollapseTracking,
    axis_index: int,
    tolerance: float,
) -> bool:
    """Validate collapse cardinality/mirroring and snap self-mirrored survivors."""

    survivors, valid = _collapse_survivors(bm, tracking)
    if not valid:
        return False

    for group_id in tracking.self_mirrored_groups:
        survivor = survivors.get(group_id)
        if survivor is not None:
            survivor.co[axis_index] = 0.0

    all_survivors = tuple(dict.fromkeys(survivors.values()))
    checked_pairs: set[frozenset[int]] = set()
    for group_id, mirror_group_id in tracking.mirror_group_by_group.items():
        if mirror_group_id is None or mirror_group_id == group_id:
            continue
        pair_key = frozenset((group_id, mirror_group_id))
        if pair_key in checked_pairs:
            continue
        checked_pairs.add(pair_key)
        survivor = survivors.get(group_id)
        mirror_survivor = survivors.get(mirror_group_id)
        if survivor is not None and mirror_survivor is not None:
            if not _coordinates_are_mirrored(survivor.co, mirror_survivor.co, axis_index, tolerance):
                return False
            continue

        # Marker inheritance is unspecified when native fuses components. If
        # one paired ID disappeared, accept any surviving fusion marker at the
        # expected mirrored coordinate; the census below remains the topology
        # safety net for complete expansion plans.
        remaining = survivor if survivor is not None else mirror_survivor
        if remaining is not None and not any(
            candidate is not remaining and _coordinates_are_mirrored(remaining.co, candidate.co, axis_index, tolerance)
            for candidate in all_survivors
        ):
            return False
    return True


def _rollback_with_report(
    operator,
    mesh,
    backup_mesh,
    *,
    warning_message: str,
    error_message: str,
) -> set[str]:
    """Restore a transaction backup and report the rollback outcome."""

    try:
        backup.restore_topology_backup(mesh, backup_mesh)
    except Exception:
        traceback.print_exc()
        _delete_report(operator, {"ERROR"}, error_message)
    else:
        _delete_report(operator, {"WARNING"}, warning_message)
    return {"FINISHED"}


def _domains_for_dissolve_mode(mode: str) -> tuple[str, ...]:
    return _DISSOLVE_DOMAINS.get(mode, ("VERT",))


def _dissolve_mode_from_select_mode(select_mode) -> str:
    """Map mesh_select_mode to VERTS/EDGES/FACES (vert > edge > face priority)."""

    # mesh_select_mode is a 3-bool sequence (vert, edge, face). Compound modes
    # pick the smallest domain first, matching native dissolve_mode.
    if select_mode[0]:
        return "VERTS"
    if select_mode[1]:
        return "EDGES"
    return "FACES"


class MESH_OT_ydd_symmetric_edit_delete(bpy.types.Operator):
    """Expand the selection to mirrored counterparts, then run mesh.delete once."""

    bl_idname = "mesh.ydd_symmetric_edit_delete"
    bl_label = "Delete"
    bl_options = {"REGISTER", "UNDO"}

    if TYPE_CHECKING:
        type: str
    else:
        type: EnumProperty(
            name="Type",
            description="Which selected elements to delete",
            items=_DELETE_TYPE_ITEMS,
            default="VERT",
        )

    @classmethod
    def poll(cls, context):
        return context.mode == "EDIT_MESH" and context.edit_object is not None

    def _native(self) -> set[str]:
        delete_type = cast(_DeleteType, self.type)
        return cast(set[str], bpy.ops.mesh.delete(type=delete_type))

    @gc_disabled_during_execute
    def execute(self, context):
        _DELETE_REPORTS.clear()

        # The enable toggle gates the keymap routes, not this operator; direct
        # invocations mirror whenever the symmetry prerequisites hold (same
        # rule as the Connect / Merge replacements).
        if _sessions_active():
            return self._native()

        symmetry = _symmetry_parameters(context)
        if symmetry is None:
            return self._native()

        obj, axis_index, tolerance = symmetry
        mesh = cast(bpy.types.Mesh, obj.data)
        bm = bmesh.from_edit_mesh(mesh)
        pair_maps = element_pairs.build_element_pair_maps(bm, axis_index, tolerance, mesh_object=obj)
        plan = element_pairs.plan_leading_domain_expansion(
            bm,
            pair_maps,
            domains=_domains_for_type(self.type),
        )

        if plan.hidden_counterpart_count > 0:
            _delete_report(
                self,
                {"WARNING"},
                "mirrored element(s) are hidden; delete declined",
            )
            return {"CANCELLED"}

        element_pairs.apply_expansion_plan(bm, plan)
        bmesh.update_edit_mesh(mesh, loop_triangles=False, destructive=False)

        result = self._native()
        if "CANCELLED" in result and "FINISHED" not in result:
            bm = bmesh.from_edit_mesh(mesh)
            _restore_expansion_selection(bm, plan)
            bmesh.update_edit_mesh(mesh, loop_triangles=False, destructive=False)
            return result

        if plan.unmatched_count > 0:
            _delete_report(
                self,
                {"INFO"},
                f"{plan.unmatched_count} element(s) had no mirrored counterpart",
            )
        return result


def _native_dissolve_mode_has(prop_name: str) -> bool:
    """Whether native mesh.dissolve_mode exposes *prop_name* on this Blender."""

    try:
        # getattr: bpy type stubs treat ops as bound methods that demand self.
        op = getattr(bpy.ops.mesh, "dissolve_mode")  # noqa: B009
        rna = op.get_rna_type()
        return prop_name in rna.properties
    except Exception:
        return False


def _operator_property_is_set(operator, name: str) -> bool:
    """Return whether *name* was explicitly set on an operator instance.

    Prefer ``Operator.properties.is_property_set`` — calling
    ``operator.is_property_set`` is not available on all Blender versions'
    Operator RNA.
    """

    props = getattr(operator, "properties", None)
    if props is not None:
        checker = getattr(props, "is_property_set", None)
        if callable(checker):
            try:
                return bool(checker(name))
            except (TypeError, AttributeError):
                pass
    checker = getattr(operator, "is_property_set", None)
    if callable(checker):
        try:
            return bool(checker(name))
        except (TypeError, AttributeError):
            pass
    return False


class MESH_OT_ydd_symmetric_edit_dissolve(bpy.types.Operator):
    """Expand the selection to mirrored counterparts, then dissolve once."""

    bl_idname = "mesh.ydd_symmetric_edit_dissolve"
    bl_label = "Dissolve"
    bl_options = {"REGISTER", "UNDO"}

    if TYPE_CHECKING:
        mode: str
        use_verts: bool
        use_face_split: bool
        use_boundary_tear: bool
        angle_threshold: float
        use_preserve_quads: bool
    else:
        mode: EnumProperty(
            name="Mode",
            description="Which dissolve domain to apply",
            items=_DISSOLVE_MODE_ITEMS,
            default="VERTS",
        )
        use_verts: BoolProperty(
            name="Dissolve Vertices",
            description="Dissolve remaining vertices which connect to only two edges",
            default=True,
        )
        use_face_split: BoolProperty(
            name="Face Split",
            description="Split off face corners to maintain surrounding geometry",
            default=False,
        )
        use_boundary_tear: BoolProperty(
            name="Tear Boundary",
            description="Split off face corners instead of merging faces",
            default=False,
        )
        # Native dissolve_edges / dissolve_mode (Blender 5.2): ANGLE, default π.
        angle_threshold: FloatProperty(
            name="Max Angle",
            description="Maximum face angle for dissolving vertices along the edge",
            subtype="ANGLE",
            default=math.pi,
            min=0.0,
            max=math.pi,
        )
        use_preserve_quads: BoolProperty(
            name="Preserve Quads",
            description="Do not dissolve vertices that form a quad face",
            default=True,
        )

    @classmethod
    def poll(cls, context):
        return context.mode == "EDIT_MESH" and context.edit_object is not None

    def draw(self, context):
        del context
        layout = self.layout
        if layout is None:
            return
        layout.prop(self, "mode")
        if self.mode == "VERTS":
            layout.prop(self, "use_face_split")
            layout.prop(self, "use_boundary_tear")
        elif self.mode == "EDGES":
            layout.prop(self, "use_verts")
            layout.prop(self, "use_face_split")
            if _native_dissolve_mode_has("angle_threshold"):
                layout.prop(self, "angle_threshold")
            if _native_dissolve_mode_has("use_preserve_quads"):
                layout.prop(self, "use_preserve_quads")
        else:
            layout.prop(self, "use_verts")

    def _options(self) -> dict:
        mode = cast(_DissolveMode, self.mode)
        if mode == "VERTS":
            return {
                "use_face_split": bool(self.use_face_split),
                "use_boundary_tear": bool(self.use_boundary_tear),
            }
        if mode == "EDGES":
            return {
                "use_verts": bool(self.use_verts),
                "use_face_split": bool(self.use_face_split),
                "angle_threshold": float(self.angle_threshold),
                "use_preserve_quads": bool(self.use_preserve_quads),
            }
        return {"use_verts": bool(self.use_verts)}

    def _native(self) -> set[str]:
        return _native_dissolve_call(cast(_DissolveMode, self.mode), self._options())

    @gc_disabled_during_execute
    def execute(self, context):
        _DELETE_REPORTS.clear()

        if _sessions_active():
            return self._native()

        symmetry = _symmetry_parameters(context)
        if symmetry is None:
            return self._native()

        obj, axis_index, tolerance = symmetry
        mesh = cast(bpy.types.Mesh, obj.data)
        bm = bmesh.from_edit_mesh(mesh)
        pair_maps = element_pairs.build_element_pair_maps(bm, axis_index, tolerance, mesh_object=obj)
        plan = element_pairs.plan_leading_domain_expansion(
            bm,
            pair_maps,
            domains=_domains_for_dissolve_mode(self.mode),
        )

        if plan.hidden_counterpart_count > 0:
            _delete_report(
                self,
                {"WARNING"},
                "mirrored element(s) are hidden; dissolve declined",
            )
            return {"CANCELLED"}

        census_before = _symmetry_census(pair_maps, bm, tolerance, mesh_object=obj)

        # Backup before expansion so census rollback restores the original
        # one-sided selection (select flags are part of the mesh backup).
        backup_mesh = None
        try:
            try:
                backup_mesh = backup.create_topology_backup(bm)
            except Exception:
                traceback.print_exc()
                _delete_report(
                    self,
                    {"ERROR"},
                    "backup creation failed; dissolve aborted",
                )
                return {"CANCELLED"}

            element_pairs.apply_expansion_plan(bm, plan)
            bmesh.update_edit_mesh(mesh, loop_triangles=False, destructive=False)

            try:
                result = _native_dissolve_call(cast(_DissolveMode, self.mode), self._options())

                if "CANCELLED" in result and "FINISHED" not in result:
                    bm = bmesh.from_edit_mesh(mesh)
                    _restore_expansion_selection(bm, plan)
                    bmesh.update_edit_mesh(mesh, loop_triangles=False, destructive=False)
                    return result

                if plan.unmatched_count == 0:
                    bm = bmesh.from_edit_mesh(mesh)
                    pair_maps_after = element_pairs.build_element_pair_maps(bm, axis_index, tolerance, mesh_object=obj)
                    census_after = _symmetry_census(pair_maps_after, bm, tolerance, mesh_object=obj)
                    if census_before != census_after:
                        try:
                            backup.restore_topology_backup(mesh, backup_mesh)
                        except Exception:
                            traceback.print_exc()
                            _delete_report(
                                self,
                                {"ERROR"},
                                "mirrored dissolve produced an asymmetric result; rollback failed",
                            )
                        else:
                            _delete_report(
                                self,
                                {"WARNING"},
                                "mirrored dissolve produced an asymmetric result; rolled back",
                            )
                        return {"FINISHED"}

                if plan.unmatched_count > 0:
                    _delete_report(
                        self,
                        {"INFO"},
                        f"{plan.unmatched_count} element(s) had no mirrored counterpart",
                    )
                return result
            except Exception:
                # F1: protect native dissolve + census path (merge/connect shape).
                traceback.print_exc()
                try:
                    backup.restore_topology_backup(mesh, backup_mesh)
                except Exception:
                    traceback.print_exc()
                    _delete_report(
                        self,
                        {"ERROR"},
                        "native dissolve failed; rollback failed",
                    )
                else:
                    _delete_report(
                        self,
                        {"WARNING"},
                        "native dissolve failed; rolled back",
                    )
                return {"FINISHED"}
        finally:
            # F10: layer cleanup must not prevent backup removal.
            try:
                if obj.mode == "EDIT":
                    bm = bmesh.from_edit_mesh(mesh)
                    if snapshot.remove_temporary_layers(bm):
                        bmesh.update_edit_mesh(mesh, loop_triangles=False, destructive=False)
            except (ReferenceError, RuntimeError):
                pass
            finally:
                backup.remove_backup(backup_mesh)


class MESH_OT_ydd_symmetric_edit_edge_collapse(bpy.types.Operator):
    """Expand selected edges, collapse once, then validate component survivors."""

    bl_idname = "mesh.ydd_symmetric_edit_edge_collapse"
    bl_label = "Edge Collapse"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return context.mode == "EDIT_MESH" and context.edit_object is not None

    def _native(self) -> set[str]:
        return _native_edge_collapse_call()

    @gc_disabled_during_execute
    def execute(self, context):
        _DELETE_REPORTS.clear()

        if _sessions_active():
            return self._native()

        symmetry = _symmetry_parameters(context)
        if symmetry is None:
            return self._native()

        obj, axis_index, tolerance = symmetry
        mesh = cast(bpy.types.Mesh, obj.data)
        bm = bmesh.from_edit_mesh(mesh)
        pair_maps = element_pairs.build_element_pair_maps(bm, axis_index, tolerance, mesh_object=obj)
        plan = element_pairs.plan_leading_domain_expansion(bm, pair_maps, domains=("EDGE",))

        if plan.hidden_counterpart_count > 0:
            _delete_report(
                self,
                {"WARNING"},
                "mirrored element(s) are hidden; edge collapse declined",
            )
            return {"CANCELLED"}

        backup_mesh = None
        try:
            try:
                backup_mesh = backup.create_topology_backup(bm)
            except Exception:
                traceback.print_exc()
                _delete_report(
                    self,
                    {"ERROR"},
                    "backup creation failed; edge collapse aborted",
                )
                return {"CANCELLED"}

            element_pairs.apply_expansion_plan(bm, plan)
            tracking = _mark_collapse_components(bm, pair_maps)
            census_before = _symmetry_census(pair_maps, bm, tolerance, mesh_object=obj)
            bmesh.update_edit_mesh(mesh, loop_triangles=False, destructive=False)

            try:
                result = _native_edge_collapse_call()

                if "CANCELLED" in result and "FINISHED" not in result:
                    bm = bmesh.from_edit_mesh(mesh)
                    _restore_expansion_selection(bm, plan)
                    bmesh.update_edit_mesh(mesh, loop_triangles=False, destructive=False)
                    return result

                bm = bmesh.from_edit_mesh(mesh)
                if not _validate_and_snap_collapse(bm, tracking, axis_index, tolerance):
                    return _rollback_with_report(
                        self,
                        mesh,
                        backup_mesh,
                        warning_message="edge collapse produced an unexpected result; rolled back",
                        error_message="edge collapse produced an unexpected result; rollback failed",
                    )
                bmesh.update_edit_mesh(mesh, loop_triangles=False, destructive=False)

                if plan.unmatched_count == 0:
                    bm = bmesh.from_edit_mesh(mesh)
                    pair_maps_after = element_pairs.build_element_pair_maps(bm, axis_index, tolerance, mesh_object=obj)
                    census_after = _symmetry_census(pair_maps_after, bm, tolerance, mesh_object=obj)
                    if census_before != census_after:
                        return _rollback_with_report(
                            self,
                            mesh,
                            backup_mesh,
                            warning_message="mirrored edge collapse produced an asymmetric result; rolled back",
                            error_message="mirrored edge collapse produced an asymmetric result; rollback failed",
                        )

                if plan.unmatched_count > 0:
                    _delete_report(
                        self,
                        {"INFO"},
                        f"{plan.unmatched_count} element(s) had no mirrored counterpart",
                    )
                return result
            except Exception:
                traceback.print_exc()
                return _rollback_with_report(
                    self,
                    mesh,
                    backup_mesh,
                    warning_message="native edge collapse failed; rolled back",
                    error_message="native edge collapse failed; rollback failed",
                )
        finally:
            try:
                if obj.mode == "EDIT":
                    bm = bmesh.from_edit_mesh(mesh)
                    if snapshot.remove_temporary_layers(bm):
                        bmesh.update_edit_mesh(mesh, loop_triangles=False, destructive=False)
            except (ReferenceError, RuntimeError):
                pass
            finally:
                backup.remove_backup(backup_mesh)


class MESH_OT_ydd_symmetric_edit_delete_edgeloop(bpy.types.Operator):
    """Expand selected edges, then run native edge-loop deletion once."""

    bl_idname = "mesh.ydd_symmetric_edit_delete_edgeloop"
    bl_label = "Edge Loops"
    bl_options = {"REGISTER", "UNDO"}

    if TYPE_CHECKING:
        use_face_split: bool
    else:
        use_face_split: BoolProperty(
            name="Face Split",
            description="Split off face corners to maintain surrounding geometry",
            default=True,
        )

    @classmethod
    def poll(cls, context):
        return context.mode == "EDIT_MESH" and context.edit_object is not None

    def _options(self) -> dict:
        return {"use_face_split": bool(self.use_face_split)}

    def _native(self) -> set[str]:
        return _native_delete_edgeloop_call(self._options())

    @gc_disabled_during_execute
    def execute(self, context):
        _DELETE_REPORTS.clear()

        if _sessions_active():
            return self._native()

        symmetry = _symmetry_parameters(context)
        if symmetry is None:
            return self._native()

        obj, axis_index, tolerance = symmetry
        mesh = cast(bpy.types.Mesh, obj.data)
        bm = bmesh.from_edit_mesh(mesh)
        pair_maps = element_pairs.build_element_pair_maps(bm, axis_index, tolerance, mesh_object=obj)
        plan = element_pairs.plan_leading_domain_expansion(bm, pair_maps, domains=("EDGE",))

        if plan.hidden_counterpart_count > 0:
            _delete_report(
                self,
                {"WARNING"},
                "mirrored element(s) are hidden; edge-loop deletion declined",
            )
            return {"CANCELLED"}

        census_before = _symmetry_census(pair_maps, bm, tolerance, mesh_object=obj)
        backup_mesh = None
        try:
            try:
                backup_mesh = backup.create_topology_backup(bm)
            except Exception:
                traceback.print_exc()
                _delete_report(
                    self,
                    {"ERROR"},
                    "backup creation failed; edge-loop deletion aborted",
                )
                return {"CANCELLED"}

            element_pairs.apply_expansion_plan(bm, plan)
            bmesh.update_edit_mesh(mesh, loop_triangles=False, destructive=False)

            try:
                result = _native_delete_edgeloop_call(self._options())

                if "CANCELLED" in result and "FINISHED" not in result:
                    bm = bmesh.from_edit_mesh(mesh)
                    _restore_expansion_selection(bm, plan)
                    bmesh.update_edit_mesh(mesh, loop_triangles=False, destructive=False)
                    return result

                if plan.unmatched_count == 0:
                    bm = bmesh.from_edit_mesh(mesh)
                    pair_maps_after = element_pairs.build_element_pair_maps(bm, axis_index, tolerance, mesh_object=obj)
                    census_after = _symmetry_census(pair_maps_after, bm, tolerance, mesh_object=obj)
                    if census_before != census_after:
                        return _rollback_with_report(
                            self,
                            mesh,
                            backup_mesh,
                            warning_message="mirrored edge-loop deletion produced an asymmetric result; rolled back",
                            error_message="mirrored edge-loop deletion produced an asymmetric result; rollback failed",
                        )

                if plan.unmatched_count > 0:
                    _delete_report(
                        self,
                        {"INFO"},
                        f"{plan.unmatched_count} element(s) had no mirrored counterpart",
                    )
                return result
            except Exception:
                traceback.print_exc()
                return _rollback_with_report(
                    self,
                    mesh,
                    backup_mesh,
                    warning_message="native edge-loop deletion failed; rolled back",
                    error_message="native edge-loop deletion failed; rollback failed",
                )
        finally:
            try:
                if obj.mode == "EDIT":
                    bm = bmesh.from_edit_mesh(mesh)
                    if snapshot.remove_temporary_layers(bm):
                        bmesh.update_edit_mesh(mesh, loop_triangles=False, destructive=False)
            except (ReferenceError, RuntimeError):
                pass
            finally:
                backup.remove_backup(backup_mesh)


class MESH_OT_ydd_symmetric_edit_dissolve_mode(bpy.types.Operator):
    """Dissolve based on the current mesh select mode (Ctrl+X replacement)."""

    bl_idname = "mesh.ydd_symmetric_edit_dissolve_mode"
    bl_label = "Dissolve Selection"
    bl_options = {"REGISTER", "UNDO"}

    if TYPE_CHECKING:
        use_verts: bool
        use_face_split: bool
        use_boundary_tear: bool
        angle_threshold: float
        use_preserve_quads: bool
    else:
        use_verts: BoolProperty(
            name="Dissolve Vertices",
            description="Dissolve remaining vertices which connect to only two edges",
            # Native mesh.dissolve_mode RNA default (Blender 4.2 / 5.2).
            default=False,
        )
        use_face_split: BoolProperty(
            name="Face Split",
            description="Split off face corners to maintain surrounding geometry",
            default=False,
        )
        use_boundary_tear: BoolProperty(
            name="Tear Boundary",
            description="Split off face corners instead of merging faces",
            default=False,
        )
        angle_threshold: FloatProperty(
            name="Max Angle",
            description="Maximum face angle for dissolving vertices along the edge",
            subtype="ANGLE",
            default=math.pi,
            min=0.0,
            max=math.pi,
        )
        use_preserve_quads: BoolProperty(
            name="Preserve Quads",
            description="Do not dissolve vertices that form a quad face",
            default=True,
        )

    @classmethod
    def poll(cls, context):
        return context.mode == "EDIT_MESH" and context.edit_object is not None

    def draw(self, context):
        del context
        layout = self.layout
        if layout is None:
            return
        # Mirror native dissolve_mode panel (all mode-relevant props).
        layout.prop(self, "use_verts")
        layout.prop(self, "use_face_split")
        layout.prop(self, "use_boundary_tear")
        if _native_dissolve_mode_has("angle_threshold"):
            layout.prop(self, "angle_threshold")
        if _native_dissolve_mode_has("use_preserve_quads"):
            layout.prop(self, "use_preserve_quads")

    @gc_disabled_during_execute
    def execute(self, context):
        mode = _dissolve_mode_from_select_mode(context.tool_settings.mesh_select_mode)
        kwargs: dict = {"mode": mode}

        if mode == "VERTS":
            kwargs["use_face_split"] = bool(self.use_face_split)
            kwargs["use_boundary_tear"] = bool(self.use_boundary_tear)
        elif mode == "EDGES":
            # Native dissolve_mode RNA defaults use_verts=False, but pure EDGE
            # dispatch calls dissolve_edges whose default is True when the
            # property is unset on the operator / KMI (probe-confirmed).
            if _operator_property_is_set(self, "use_verts"):
                kwargs["use_verts"] = bool(self.use_verts)
            else:
                kwargs["use_verts"] = True
            kwargs["use_face_split"] = bool(self.use_face_split)
            kwargs["angle_threshold"] = float(self.angle_threshold)
            kwargs["use_preserve_quads"] = bool(self.use_preserve_quads)
        else:
            # FACES: native dissolve_faces use_verts default is False; only
            # forward when the caller / KMI set the property explicitly.
            if _operator_property_is_set(self, "use_verts"):
                kwargs["use_verts"] = bool(self.use_verts)

        # Constant getattr: bpy type stubs omit dynamically registered ops (B009).
        dissolve = getattr(bpy.ops.mesh, "ydd_symmetric_edit_dissolve")  # noqa: B009
        return cast(set[str], dissolve(**kwargs))


class YSE_MT_delete(bpy.types.Menu):
    """Delete menu mirroring native VIEW3D_MT_edit_mesh_delete (Blender 5.2)."""

    bl_idname = "YSE_MT_delete"
    bl_label = "Delete"

    def draw(self, context: bpy.types.Context) -> None:
        del context
        layout = self.layout
        if layout is None:
            return

        for type_id, label, _description in _DELETE_TYPE_ITEMS:
            operator = layout.operator(
                MESH_OT_ydd_symmetric_edit_delete.bl_idname,
                text=label,
            )
            operator.type = type_id

        layout.separator()

        # Per-mode native defaults: edges use_verts=True, faces use_verts=False.
        op_v = layout.operator(
            MESH_OT_ydd_symmetric_edit_dissolve.bl_idname,
            text="Dissolve Vertices",
        )
        op_v.mode = "VERTS"

        op_e = layout.operator(
            MESH_OT_ydd_symmetric_edit_dissolve.bl_idname,
            text="Dissolve Edges",
        )
        op_e.mode = "EDGES"
        op_e.use_verts = True

        op_f = layout.operator(
            MESH_OT_ydd_symmetric_edit_dissolve.bl_idname,
            text="Dissolve Faces",
        )
        op_f.mode = "FACES"
        op_f.use_verts = False

        layout.separator()

        layout.operator("mesh.dissolve_limited")

        layout.separator()

        layout.operator(MESH_OT_ydd_symmetric_edit_edge_collapse.bl_idname)
        layout.operator(MESH_OT_ydd_symmetric_edit_delete_edgeloop.bl_idname, text="Edge Loops")

        # Native VIEW3D_MT_edit_mesh_delete ends with asset catalog items.
        if hasattr(layout, "template_node_operator_asset_menu_items"):
            layout.template_node_operator_asset_menu_items(catalog_path="Mesh/Delete")


CLASSES = (
    MESH_OT_ydd_symmetric_edit_delete,
    MESH_OT_ydd_symmetric_edit_dissolve,
    MESH_OT_ydd_symmetric_edit_edge_collapse,
    MESH_OT_ydd_symmetric_edit_delete_edgeloop,
    MESH_OT_ydd_symmetric_edit_dissolve_mode,
    YSE_MT_delete,
)
