# SPDX-License-Identifier: GPL-3.0-or-later

"""Symmetric delete / dissolve operators / menu, plus pure selection-expansion helpers."""

from __future__ import annotations

import traceback
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

import bmesh
import bpy
from bpy.props import BoolProperty, EnumProperty

from . import backup, core
from .replay import _symmetry_parameters

_DeleteType = Literal["VERT", "EDGE", "FACE", "EDGE_FACE", "ONLY_FACE"]
_DissolveMode = Literal["VERTS", "EDGES", "FACES"]


@dataclass(frozen=True)
class ElementPairMaps:
    """Involutive element correspondence used by leading-domain expansion.

    ``None`` values mean the element has no unique counterpart (unmatched
    endpoints, multi-edges, or colliding face keys).
    """

    vert_pairs: dict[int, int]
    edge_pair_by_index: dict[int, int | None]
    face_pair_by_index: dict[int, int | None]


@dataclass(frozen=True)
class ExpansionPlan:
    """Indices to select for a leading-domain mirror expansion.

    Counts track unmatched / hidden counterparts; they do not alter select.
    """

    add_vert_indices: tuple[int, ...]
    add_edge_indices: tuple[int, ...]
    add_face_indices: tuple[int, ...]
    unmatched_count: int
    hidden_counterpart_count: int


def build_element_pair_maps(
    bm: bmesh.types.BMesh,
    axis_index: int,
    tolerance: float,
) -> ElementPairMaps:
    """Resolve vertex / edge / face counterparts for selection expansion."""

    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    bm.verts.index_update()
    bm.edges.index_update()
    bm.faces.index_update()

    coords = [vertex.co for vertex in bm.verts]
    vert_pairs = core.build_vertex_pair_table(coords, axis_index, tolerance)

    endpoint_to_edges: dict[frozenset[int], list[int]] = defaultdict(list)
    for edge in bm.edges:
        key = frozenset((edge.verts[0].index, edge.verts[1].index))
        endpoint_to_edges[key].append(edge.index)

    multi_edge_keys = {key for key, indices in endpoint_to_edges.items() if len(indices) > 1}

    edge_pair_by_index: dict[int, int | None] = {}
    for edge in bm.edges:
        v1 = edge.verts[0].index
        v2 = edge.verts[1].index
        own_key = frozenset((v1, v2))
        if own_key in multi_edge_keys:
            edge_pair_by_index[edge.index] = None
            continue
        p1 = vert_pairs.get(v1)
        p2 = vert_pairs.get(v2)
        if p1 is None or p2 is None:
            edge_pair_by_index[edge.index] = None
            continue
        counterpart_key = frozenset((p1, p2))
        if counterpart_key in multi_edge_keys:
            edge_pair_by_index[edge.index] = None
            continue
        candidates = endpoint_to_edges.get(counterpart_key)
        if not candidates:
            edge_pair_by_index[edge.index] = None
            continue
        edge_pair_by_index[edge.index] = candidates[0]

    face_ids_by_vertex_set: dict[frozenset[int], list[int]] = defaultdict(list)
    for face in bm.faces:
        key = frozenset(vertex.index for vertex in face.verts)
        face_ids_by_vertex_set[key].append(face.index)

    conflicted_keys = {key for key, indices in face_ids_by_vertex_set.items() if len(indices) > 1}

    face_pair_by_index: dict[int, int | None] = {}
    for face in bm.faces:
        own_verts = [vertex.index for vertex in face.verts]
        own_key = frozenset(own_verts)
        if own_key in conflicted_keys:
            face_pair_by_index[face.index] = None
            continue
        mapped: list[int] = []
        missing = False
        for vertex_index in own_verts:
            partner = vert_pairs.get(vertex_index)
            if partner is None:
                missing = True
                break
            mapped.append(partner)
        if missing:
            face_pair_by_index[face.index] = None
            continue
        counterpart_key = frozenset(mapped)
        if counterpart_key in conflicted_keys:
            face_pair_by_index[face.index] = None
            continue
        counterparts = face_ids_by_vertex_set.get(counterpart_key)
        if not counterparts:
            face_pair_by_index[face.index] = None
            continue
        face_pair_by_index[face.index] = counterparts[0]

    return ElementPairMaps(
        vert_pairs=vert_pairs,
        edge_pair_by_index=edge_pair_by_index,
        face_pair_by_index=face_pair_by_index,
    )


def plan_leading_domain_expansion(
    bm: bmesh.types.BMesh,
    pair_maps: ElementPairMaps,
    *,
    domains: tuple[str, ...],
) -> ExpansionPlan:
    """Plan select expansion for leading domains without mutating *bm*."""

    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()

    add_verts: list[int] = []
    add_edges: list[int] = []
    add_faces: list[int] = []
    unmatched_count = 0
    hidden_counterpart_count = 0

    if "VERT" in domains:
        for vertex in bm.verts:
            if not vertex.select or vertex.hide:
                continue
            partner = pair_maps.vert_pairs.get(vertex.index)
            if partner is None:
                unmatched_count += 1
                continue
            if partner == vertex.index:
                continue
            counterpart = bm.verts[partner]
            if counterpart.select:
                continue
            if counterpart.hide:
                hidden_counterpart_count += 1
                continue
            add_verts.append(partner)

    if "EDGE" in domains:
        for edge in bm.edges:
            if not edge.select or edge.hide:
                continue
            partner = pair_maps.edge_pair_by_index.get(edge.index)
            if partner is None:
                unmatched_count += 1
                continue
            if partner == edge.index:
                continue
            counterpart = bm.edges[partner]
            if counterpart.select:
                continue
            if counterpart.hide:
                hidden_counterpart_count += 1
                continue
            add_edges.append(partner)

    if "FACE" in domains:
        for face in bm.faces:
            if not face.select or face.hide:
                continue
            partner = pair_maps.face_pair_by_index.get(face.index)
            if partner is None:
                unmatched_count += 1
                continue
            if partner == face.index:
                continue
            counterpart = bm.faces[partner]
            if counterpart.select:
                continue
            if counterpart.hide:
                hidden_counterpart_count += 1
                continue
            add_faces.append(partner)

    return ExpansionPlan(
        add_vert_indices=tuple(add_verts),
        add_edge_indices=tuple(add_edges),
        add_face_indices=tuple(add_faces),
        unmatched_count=unmatched_count,
        hidden_counterpart_count=hidden_counterpart_count,
    )


def apply_expansion_plan(bm: bmesh.types.BMesh, plan: ExpansionPlan) -> None:
    """Set select on planned indices only; does not call select_flush_mode."""

    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()

    for index in plan.add_vert_indices:
        bm.verts[index].select = True
    for index in plan.add_edge_indices:
        bm.edges[index].select = True
    for index in plan.add_face_indices:
        bm.faces[index].select = True


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


def _native_dissolve_call(mode: str, options: dict) -> set[str]:
    """Invoke the matching native dissolve operator once.

    Module-level so fault-injection tests can wrap the call without touching
    bpy.ops registration (same pattern as ``replay._native_vert_connect_path``).
    """

    if mode == "VERTS":
        return cast(
            set[str],
            bpy.ops.mesh.dissolve_verts(
                use_face_split=bool(options.get("use_face_split", False)),
                use_boundary_tear=bool(options.get("use_boundary_tear", False)),
            ),
        )
    if mode == "EDGES":
        return cast(
            set[str],
            bpy.ops.mesh.dissolve_edges(
                use_verts=bool(options.get("use_verts", True)),
                use_face_split=bool(options.get("use_face_split", False)),
            ),
        )
    if mode == "FACES":
        return cast(
            set[str],
            bpy.ops.mesh.dissolve_faces(
                use_verts=bool(options.get("use_verts", False)),
            ),
        )
    raise ValueError(f"unknown dissolve mode: {mode!r}")


def _symmetry_census(
    pair_maps: ElementPairMaps,
    bm: bmesh.types.BMesh,
) -> tuple[int, int, int]:
    """Count unmatched verts / edges / faces for post-dissolve verification.

    Verts: not present in the pair table. Edges/faces: pair map value is None.
    """

    unmatched_verts = sum(1 for vertex in bm.verts if vertex.index not in pair_maps.vert_pairs)
    unmatched_edges = sum(1 for partner in pair_maps.edge_pair_by_index.values() if partner is None)
    unmatched_faces = sum(1 for partner in pair_maps.face_pair_by_index.values() if partner is None)
    return unmatched_verts, unmatched_edges, unmatched_faces


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
        pair_maps = build_element_pair_maps(bm, axis_index, tolerance)
        plan = plan_leading_domain_expansion(
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

        apply_expansion_plan(bm, plan)
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
            }
        return {"use_verts": bool(self.use_verts)}

    def _native(self) -> set[str]:
        return _native_dissolve_call(cast(_DissolveMode, self.mode), self._options())

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
        pair_maps = build_element_pair_maps(bm, axis_index, tolerance)
        plan = plan_leading_domain_expansion(
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

        census_before = _symmetry_census(pair_maps, bm)

        apply_expansion_plan(bm, plan)
        bmesh.update_edit_mesh(mesh, loop_triangles=False, destructive=False)

        backup_mesh = None
        try:
            bm = bmesh.from_edit_mesh(mesh)
            try:
                backup_mesh = backup.create_topology_backup(bm)
            except Exception:
                traceback.print_exc()
                _delete_report(
                    self,
                    {"ERROR"},
                    "backup creation failed; dissolve aborted",
                )
                bm = bmesh.from_edit_mesh(mesh)
                _restore_expansion_selection(bm, plan)
                bmesh.update_edit_mesh(mesh, loop_triangles=False, destructive=False)
                return {"CANCELLED"}

            result = _native_dissolve_call(cast(_DissolveMode, self.mode), self._options())

            if "CANCELLED" in result and "FINISHED" not in result:
                bm = bmesh.from_edit_mesh(mesh)
                _restore_expansion_selection(bm, plan)
                bmesh.update_edit_mesh(mesh, loop_triangles=False, destructive=False)
                return result

            if plan.unmatched_count == 0:
                bm = bmesh.from_edit_mesh(mesh)
                pair_maps_after = build_element_pair_maps(bm, axis_index, tolerance)
                census_after = _symmetry_census(pair_maps_after, bm)
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
        finally:
            try:
                if obj.mode == "EDIT":
                    bm = bmesh.from_edit_mesh(mesh)
                    if core.remove_temporary_layers(bm):
                        bmesh.update_edit_mesh(mesh, loop_triangles=False, destructive=False)
            except (ReferenceError, RuntimeError):
                pass
            backup.remove_backup(backup_mesh)


class MESH_OT_ydd_symmetric_edit_dissolve_mode(bpy.types.Operator):
    """Dissolve based on the current mesh select mode (Ctrl+X replacement)."""

    bl_idname = "mesh.ydd_symmetric_edit_dissolve_mode"
    bl_label = "Dissolve Selection"
    bl_options = {"REGISTER", "UNDO"}

    if TYPE_CHECKING:
        use_verts: bool
    else:
        use_verts: BoolProperty(
            name="Dissolve Vertices",
            description="Dissolve remaining vertices which connect to only two edges",
            # Native mesh.dissolve_mode default (Blender 4.2 / 5.2).
            default=False,
        )

    @classmethod
    def poll(cls, context):
        return context.mode == "EDIT_MESH" and context.edit_object is not None

    def execute(self, context):
        mode = _dissolve_mode_from_select_mode(context.tool_settings.mesh_select_mode)
        kwargs: dict = {"mode": mode}
        if mode in ("EDGES", "FACES"):
            kwargs["use_verts"] = bool(self.use_verts)
        # VERTS uses native-like face_split / boundary_tear defaults on the
        # target operator (False / False); dissolve_mode only exposes use_verts.
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

        layout.operator("mesh.edge_collapse")
        layout.operator("mesh.delete_edgeloop", text="Edge Loops")


CLASSES = (
    MESH_OT_ydd_symmetric_edit_delete,
    MESH_OT_ydd_symmetric_edit_dissolve,
    MESH_OT_ydd_symmetric_edit_dissolve_mode,
    YSE_MT_delete,
)
