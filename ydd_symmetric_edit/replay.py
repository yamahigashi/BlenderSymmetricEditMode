# SPDX-License-Identifier: GPL-3.0-or-later

"""One-shot symmetry replay for native Connect and Merge operations."""

from __future__ import annotations

import traceback
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import bmesh
import bpy
from bpy.props import EnumProperty, FloatProperty
from mathutils import Vector

from . import backup, core
from ._types import MirrorOverlap

_MERGE_MODES = frozenset({"CENTER", "COLLAPSE", "FIRST", "LAST"})


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
    coords: Sequence[Vector],
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
    mapped = lookup.find_all_mirrored(history_coords)
    if any(index is None for index in mapped):
        return None
    return cast(tuple[int, ...], mapped)


def _map_history_via_pairs(
    history_indices: Sequence[int],
    pairs: dict[int, int],
) -> tuple[int, ...] | None:
    """Map a history path through the involutive pair table; ``None`` when any
    vertex is unpaired.

    Deliberately shares the classification's pair table instead of running a
    separate lookup: a coordinate-based partial batch sees no on-plane
    queries, so a near-plane vertex the classification rejected could sneak
    back in as a counterpart (review finding).
    """

    mapped = []
    for index in history_indices:
        partner = pairs.get(index)
        if partner is None:
            return None
        mapped.append(partner)
    return tuple(mapped)


def _history_is_mirror_invariant(
    history_indices: Sequence[int],
    pairs: dict[int, int],
) -> bool:
    """True when ρ(H) equals H itself or reversed(H), elementwise.

    A self-mirrored vertex *set* does not make the ordered connect path
    symmetric (a zig-zag alternating between sides is the counterexample);
    only these two sequence shapes keep the native edge set mirror-invariant.
    On-plane vertices map to themselves in the pair table.
    """

    if not history_indices:
        return False
    mapped = []
    for index in history_indices:
        partner = pairs.get(index)
        if partner is None:
            return False
        mapped.append(partner)
    sequence = list(history_indices)
    return mapped == sequence or mapped == sequence[::-1]


def _vertex_snapshot(
    bm: bmesh.types.BMesh,
) -> tuple[tuple[Vector, ...], tuple[int, ...], tuple[Vector, ...], tuple[int, ...]]:
    bm.verts.ensure_lookup_table()
    bm.verts.index_update()
    coords = tuple(vertex.co.copy() for vertex in bm.verts)
    selected = tuple(vertex.index for vertex in bm.verts if vertex.select)
    history = cast(
        Iterable[bmesh.types.BMVert | bmesh.types.BMEdge | bmesh.types.BMFace],
        bm.select_history,
    )
    history_vertices = [element for element in history if isinstance(element, bmesh.types.BMVert)]
    history_coords = tuple(element.co.copy() for element in history_vertices)
    history_indices = tuple(element.index for element in history_vertices)
    return coords, selected, history_coords, history_indices


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


def _report_self_mirrored(operator) -> None:
    operator.report(
        {"INFO"},
        "Selection is symmetric; the native result is already symmetric",
    )


def _report_partial_overlap(operator, action: str) -> None:
    operator.report(
        {"WARNING"},
        f"Selection partially overlaps its mirror image; ran the native {action} only",
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
        coords, selected_indices, history_coords, history_indices = _vertex_snapshot(bm)
        snapshot = classify_mirror_selection(
            coords,
            selected_indices,
            axis_index=axis_index,
            tolerance=tolerance,
        )
        mirrored_history = _map_history_via_pairs(history_indices, snapshot.pairs)

        result = cast(set[str], bpy.ops.mesh.vert_connect_path())
        if "FINISHED" not in result:
            return result
        if snapshot.overlap is MirrorOverlap.SELF_MIRRORED:
            if _history_is_mirror_invariant(history_indices, snapshot.pairs):
                # The native cut path is mirror-invariant, so one native run
                # already produced the symmetric result.
                _report_self_mirrored(self)
            else:
                _report_partial_overlap(self, "connect")
            return result
        if snapshot.overlap is MirrorOverlap.PARTIAL:
            _report_partial_overlap(self, "connect")
            return result
        if mirrored_history is None:
            _report_missing(self, max(1, len(snapshot.missing)), partial=False)
            return result

        mirrored_history_coords = tuple(coords[index].copy() for index in mirrored_history)
        selected_coords = tuple(coords[index].copy() for index in selected_indices)
        # The source-side mesh is already mutated, so from here on every path,
        # including exceptions, must return `result`: returning CANCELLED (or
        # raising) would skip the undo push and fold that mutation into the
        # previous undo step.
        backup_mesh = None
        mirror_warning = None
        try:
            bm = bmesh.from_edit_mesh(mesh)
            backup_mesh = backup.create_topology_backup(bm)
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
                    bmesh.update_edit_mesh(mesh, loop_triangles=False, destructive=False)
                    mirrored_result = cast(set[str], bpy.ops.mesh.vert_connect_path())
                    if "FINISHED" not in mirrored_result:
                        # A clean native refusal keeps the source result;
                        # rollback is reserved for exceptions.
                        mirror_warning = "Mirrored connect did not finish; applied the native connect only"
            except Exception:
                traceback.print_exc()
                backup.restore_topology_backup(mesh, backup_mesh)
                mirror_warning = "Unexpected error; the mirrored connect was rolled back"
        except Exception:
            # Backup creation or the rollback itself failed.  There is no way
            # to restore, so returning `result` (and keeping the undo push)
            # takes priority over a perfect mesh state.
            traceback.print_exc()
            mirror_warning = mirror_warning or "Internal error during the mirrored connect"
        finally:
            try:
                # Selection restore is best effort on every path.
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
            try:
                # The backup ID layer survives both the success path and the
                # rollback path (the rollback writes it back from the backup
                # mesh), so it is removed unconditionally.
                bm = bmesh.from_edit_mesh(mesh)
                core.remove_temporary_layers(bm)
                bmesh.update_edit_mesh(mesh, loop_triangles=False, destructive=False)
            except Exception:
                traceback.print_exc()
            backup.remove_backup(backup_mesh)
        try:
            if mirror_warning:
                self.report({"WARNING"}, mirror_warning)
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
        symmetry = _symmetry_parameters(context)
        if symmetry is None:
            return self._native()
        obj, axis_index, tolerance = symmetry
        mesh = cast(bpy.types.Mesh, obj.data)
        bm = bmesh.from_edit_mesh(mesh)
        coords, selected_indices, history_coords, history_indices = _vertex_snapshot(bm)
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
                _report_self_mirrored(self)
                return self._native()
            # PARTIAL (complete): symmetrize the selection to reduce it to the
            # self-mirrored case, then run the native merge once.
            added = self._symmetrize_selection(bm, mesh, snapshot)
            self.report(
                {"INFO"},
                f"Added {added} mirrored vertex(es) to the selection to keep the merge symmetric",
            )
            return self._native()

        return self._execute_side_split_merge(
            mesh,
            bm,
            axis_index,
            tolerance,
            snapshot,
            coords,
            history_indices,
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
        coords: tuple[Vector, ...],
        selected_indices: tuple[int, ...],
        history_coords: tuple[Vector, ...],
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

        # Mark members (+k) and their mirrors (-k) in a temporary layer so the
        # post-native re-identification is exact: a coordinate lookup would
        # miscount whenever an unrelated vertex sits within tolerance of an
        # old member position (review finding).  The survivor of a native
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
                    # merged.
                    if len(survivors) > 1:
                        continue
                    mirrors = [vertex for vertex in mirror_verts_by_cluster.get(cluster_number, ()) if vertex.is_valid]
                    if not mirrors:
                        if expected_mirrors:
                            self.report(
                                {"WARNING"},
                                "Mirror merge skipped for one cluster: its mirrored vertices could not be re-identified",
                            )
                        continue
                    if abs(target[axis_index]) <= tolerance:
                        # A merge landing on the plane welds the mirrored
                        # cluster into the same surviving vertex so the mesh
                        # stays connected.
                        survivor = survivors[0] if survivors and survivors[0].is_valid else None
                        if survivor is None:
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
                self.report({"WARNING"}, mirror_warning)
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
        coords: tuple[Vector, ...],
        history_indices: tuple[int, ...],
    ) -> set[str]:
        """FIRST/LAST on a self-mirrored selection: merge each side to its own
        endpoint (5-2).  One native run would drag both sides to one point."""

        # Step 1: fix the merge target from the ORIGINAL history, before any
        # selection changes; it must not move for the rest of the procedure.
        target_index = history_indices[0] if self.mode == "FIRST" else history_indices[-1]
        target_co = coords[target_index].copy()

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
                self.report(
                    {"INFO"},
                    f"Added {added} mirrored vertex(es) to the selection to keep the merge symmetric",
                )
            _report_self_mirrored(self)
            return self._native()
        if any(abs(coords[index][axis_index]) <= tolerance for index in extended_selection):
            # An on-plane vertex belongs to both sides at once; the merge runs
            # natively on the ORIGINAL selection only.
            self.report(
                {"WARNING"},
                "Selection includes on-plane vertices; ran the native merge only",
            )
            return self._native()

        # Step 2: the side containing the target is the source side.
        source_sign = 1.0 if target_co[axis_index] > 0.0 else -1.0
        source_side = {index for index in extended_selection if coords[index][axis_index] * source_sign > 0.0}
        mirror_side = extended_selection - source_side

        # Step 3 verification, still before any mutation: the rebuilt history
        # must keep the original endpoint.
        source_history = [index for index in history_indices if index in source_side]
        rebuilt_endpoint = None
        if source_history:
            rebuilt_endpoint = source_history[0] if self.mode == "FIRST" else source_history[-1]
        if rebuilt_endpoint != target_index:
            self.report(
                {"WARNING"},
                "Could not rebuild a per-side merge history; ran the native merge only",
            )
            return self._native()

        # All guards passed — now mutate.  PARTIAL symmetrization (always
        # reported), then group markers, then the source-side rebuild.
        if snapshot.overlap is MirrorOverlap.PARTIAL:
            added = self._symmetrize_selection(bm, mesh, snapshot)
            self.report(
                {"INFO"},
                f"Added {added} mirrored vertex(es) to the selection to keep the merge symmetric",
            )

        # Group markers (source=1, mirror=2) make the post-native
        # re-identification exact; a coordinate lookup could confuse
        # coincident vertices (review finding).  Layer creation invalidates
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
        # history is always rebuilt explicitly.
        for index in sorted(mirror_side):
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
                # backup creation where no rollback ran (review finding):
                # reselect the mirror side, keep a source-only history.  The
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
                self.report({"WARNING"}, mirror_warning)
            else:
                side_label = "first" if self.mode == "FIRST" else "last"
                self.report({"INFO"}, f"Merged each side to its own {side_label} vertex")
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
