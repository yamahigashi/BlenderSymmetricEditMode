# SPDX-License-Identifier: GPL-3.0-or-later

"""One-shot symmetry replay for native Connect and Merge operations."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import bmesh
import bpy
from bpy.props import EnumProperty, FloatProperty
from mathutils import Vector

from . import core

_MERGE_MODES = frozenset({"CENTER", "COLLAPSE", "FIRST", "LAST"})


@dataclass(frozen=True)
class MirrorSelection:
    """Coordinate-index snapshot used before a native topology operation."""

    selected: frozenset[int]
    shared: frozenset[int]
    off: frozenset[int]
    mirror_by_source: dict[int, int]
    missing: frozenset[int]

    @property
    def mirrors(self) -> frozenset[int]:
        return frozenset(self.mirror_by_source.values())


def classify_mirror_selection(
    coords: Sequence[Vector],
    selected_indices: Sequence[int],
    *,
    axis_index: int,
    tolerance: float,
) -> MirrorSelection:
    """Classify selected coordinates and resolve every available counterpart."""

    lookup = core.build_vertex_mirror_lookup(coords, axis_index, tolerance)
    selected = frozenset(selected_indices)
    shared = frozenset(index for index in selected if lookup.is_on_plane(coords[index]))
    off = selected - shared
    mirror_by_source: dict[int, int] = {}
    missing = set()
    for index in off:
        mirror_index = lookup.find(coords[index])
        if mirror_index is None:
            missing.add(index)
        else:
            mirror_by_source[index] = mirror_index
    return MirrorSelection(
        selected=selected,
        shared=shared,
        off=off,
        mirror_by_source=mirror_by_source,
        missing=frozenset(missing),
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
    coords: Sequence[Vector],
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
        target += coords[index]
    return target / len(cluster)


def map_mirrored_history(
    history_coords: Sequence[Vector],
    coords: Sequence[Vector],
    *,
    axis_index: int,
    tolerance: float,
) -> tuple[int, ...] | None:
    """Map a history path atomically; plane coordinates resolve to themselves."""

    lookup = core.build_vertex_mirror_lookup(coords, axis_index, tolerance)
    mapped = []
    for coordinate in history_coords:
        index = lookup.find(coordinate)
        if index is None:
            return None
        mapped.append(index)
    return tuple(mapped)


def _vertex_snapshot(
    bm: bmesh.types.BMesh,
) -> tuple[tuple[Vector, ...], tuple[int, ...], tuple[Vector, ...]]:
    bm.verts.ensure_lookup_table()
    bm.verts.index_update()
    coords = tuple(vertex.co.copy() for vertex in bm.verts)
    selected = tuple(vertex.index for vertex in bm.verts if vertex.select)
    history = cast(
        Iterable[bmesh.types.BMVert | bmesh.types.BMEdge | bmesh.types.BMFace],
        bm.select_history,
    )
    history_coords = tuple(element.co.copy() for element in history if isinstance(element, bmesh.types.BMVert))
    return coords, selected, history_coords


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


def _find_direct_vertex(
    bm: bmesh.types.BMesh,
    coordinate: Vector,
    axis_index: int,
    tolerance: float,
) -> bmesh.types.BMVert | None:
    bm.verts.ensure_lookup_table()
    coords = tuple(vertex.co.copy() for vertex in bm.verts)
    lookup = core.build_vertex_mirror_lookup(coords, axis_index, tolerance)
    direct_index = lookup.find(core.mirror_coordinate(coordinate, axis_index))
    return None if direct_index is None else bm.verts[direct_index]


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

    def resolve(coordinate: Vector) -> bmesh.types.BMVert | None:
        index = lookup.find(core.mirror_coordinate(coordinate, axis_index))
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


def _report_missing(operator, count: int, *, partial: bool) -> None:
    if not count:
        return
    if partial:
        operator.report(
            {"WARNING"},
            f"{count} vertices have no mirror counterpart; mirrored partially",
        )
    else:
        operator.report(
            {"WARNING"},
            f"{count} vertices have no mirror counterpart; mirrored connect skipped",
        )


def _report_crossing(operator) -> None:
    operator.report(
        {"INFO"},
        "Selection already includes mirrored vertices; ran native operation only",
    )


class MESH_OT_ydd_symmetric_edit_connect(bpy.types.Operator):
    """Run Vertex Connect on the selected path and its mirrored path."""

    bl_idname = "mesh.ydd_symmetric_edit_connect"
    bl_label = "Symmetric Vertex Connect"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return context.mode == "EDIT_MESH" and context.edit_object is not None

    def execute(self, context):
        symmetry = _symmetry_parameters(context)
        if symmetry is None:
            return bpy.ops.mesh.vert_connect_path()
        obj, axis_index, tolerance = symmetry
        mesh = cast(bpy.types.Mesh, obj.data)
        bm = bmesh.from_edit_mesh(mesh)
        coords, selected_indices, history_coords = _vertex_snapshot(bm)
        snapshot = classify_mirror_selection(
            coords,
            selected_indices,
            axis_index=axis_index,
            tolerance=tolerance,
        )
        mirrored_history = map_mirrored_history(
            history_coords,
            coords,
            axis_index=axis_index,
            tolerance=tolerance,
        )

        result = cast(set[str], bpy.ops.mesh.vert_connect_path())
        if "FINISHED" not in result:
            return result
        if selection_crosses_mirror(snapshot):
            _report_crossing(self)
            return result
        if mirrored_history is None:
            _report_missing(self, max(1, len(snapshot.missing)), partial=False)
            return result

        mirrored_history_coords = tuple(coords[index].copy() for index in mirrored_history)
        selected_coords = tuple(coords[index].copy() for index in selected_indices)
        bm = bmesh.from_edit_mesh(mesh)
        if not _set_vertex_path(
            bm,
            mirrored_history_coords,
            axis_index,
            tolerance,
        ):
            self.report({"WARNING"}, "Mirrored connect path changed after native execution; mirrored connect skipped")
            return result
        bmesh.update_edit_mesh(mesh, loop_triangles=False, destructive=False)
        mirrored_result = cast(set[str], bpy.ops.mesh.vert_connect_path())
        if "FINISHED" not in mirrored_result:
            self.report({"WARNING"}, "Mirrored connect did not finish; applied the native connect only")

        # The source-side mesh is already mutated, so from here on the return
        # value must stay FINISHED: returning CANCELLED (or raising) would skip
        # the undo push and fold that mutation into the previous undo step.
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
            self.report({"WARNING"}, "Could not restore the original selection after the mirrored connect")
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
        symmetry = _symmetry_parameters(context)
        if symmetry is None:
            return self._native()
        obj, axis_index, tolerance = symmetry
        mesh = cast(bpy.types.Mesh, obj.data)
        bm = bmesh.from_edit_mesh(mesh)
        coords, selected_indices, history_coords = _vertex_snapshot(bm)
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
            return self._native()

        if selection_crosses_mirror(snapshot):
            _report_crossing(self)
            return self._native()
        if self.mode not in _MERGE_MODES:
            return self._native()
        if self.mode in {"FIRST", "LAST"} and not history_coords:
            return self._native()
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
        bm.verts.ensure_lookup_table()
        mirrored_clusters = tuple(
            tuple(bm.verts[snapshot.mirror_by_source[index]] for index in cluster if index in snapshot.mirror_by_source)
            for cluster in clusters
        )
        source_clusters = tuple(tuple(bm.verts[index] for index in cluster) for cluster in clusters)

        result = self._native()
        if "FINISHED" not in result:
            return result

        bm = bmesh.from_edit_mesh(mesh)
        for target, mirrored_cluster, source_cluster in zip(targets, mirrored_clusters, source_clusters, strict=True):
            # Native Collapse can leave a cluster untouched (e.g. a selection
            # island on a fully detached component).  Mirroring such a cluster
            # would merge geometry the native operator never merged, so apply
            # the mirror only when the native side actually collapsed.
            if sum(1 for vertex in source_cluster if vertex.is_valid) > 1:
                continue
            mirrors = list(dict.fromkeys(vertex for vertex in mirrored_cluster if vertex.is_valid))
            if not mirrors:
                continue
            lookup = core.build_vertex_mirror_lookup(
                tuple(vertex.co.copy() for vertex in bm.verts),
                axis_index,
                tolerance,
            )
            if lookup.is_on_plane(target):
                survivor = _find_direct_vertex(bm, target, axis_index, tolerance)
                if survivor is None:
                    # The native mesh is already mutated; raising here would
                    # skip the undo push and fold that change into the previous
                    # undo step.  Leave this cluster unmirrored instead.
                    self.report(
                        {"WARNING"},
                        "Mirror merge skipped for one cluster: the on-plane survivor could not be identified",
                    )
                    continue
                mirrors.append(survivor)
                merge_co = target
            else:
                merge_co = core.mirror_coordinate(target, axis_index)
            unique_verts = list(dict.fromkeys(vertex for vertex in mirrors if vertex.is_valid))
            bmesh.ops.pointmerge(bm, verts=unique_verts, merge_co=merge_co)

        bmesh.update_edit_mesh(mesh, loop_triangles=True, destructive=True)
        return result


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
