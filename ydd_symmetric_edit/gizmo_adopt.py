# SPDX-License-Identifier: GPL-3.0-or-later

"""Adopt already-running native extrudes started by toolbar gizmos."""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass

import bmesh
import bpy

from . import extrude, layer_names, matching, selection, session_state, snapshot
from ._types import (
    Coordinate3D,
    ExtrudeFreezeEntry,
    ExtrudeSnapshot,
    FaceId,
    GizmoExclusionTicket,
    KnifeSession,
    MeshSelectionMode,
    SymmetryAxes,
)

GIZMO_POLL_INTERVAL = 0.05
TICKET_GRACE = 0.5
MENU_TICKET_GRACE = 10.0
GIZMO_ROUTE = "GIZMO_ADOPTED"

_TOOL_IDS = {
    "builtin.extrude_region": "EXTRUDE_CONTEXT",
    "builtin.extrude_along_normals": "EXTRUDE_SHRINK_FATTEN",
    "builtin.extrude_individual": "EXTRUDE_FACES_INDIV",
    "builtin.extrude_manifold": "EXTRUDE_MANIFOLD",
}


@dataclass(slots=True)
class _ReadPlan:
    bm: bmesh.types.BMesh
    survivors: set[bmesh.types.BMVert]
    copies: set[bmesh.types.BMVert]
    vertex_partner: dict[bmesh.types.BMVert, bmesh.types.BMVert]
    copy_origin: dict[bmesh.types.BMVert, bmesh.types.BMVert]
    survivor_edges: set[bmesh.types.BMEdge]
    new_edges: set[bmesh.types.BMEdge]
    edge_partner: dict[bmesh.types.BMEdge, bmesh.types.BMEdge]
    deleted_edge_targets: tuple[bmesh.types.BMEdge, ...]
    survivor_faces: set[bmesh.types.BMFace]
    new_faces: set[bmesh.types.BMFace]
    face_partner: dict[bmesh.types.BMFace, bmesh.types.BMFace]
    deleted_face_targets: tuple[bmesh.types.BMFace, ...]
    selected_copy_edges: tuple[bmesh.types.BMEdge, ...]
    copy_source_cycles: dict[bmesh.types.BMVert, tuple[bmesh.types.BMVert, ...]]
    mesh_select_mode: MeshSelectionMode


def issue_exclusion_ticket(context, grace: float = TICKET_GRACE) -> None:
    """Exclude one window/mesh from adoption before an addon-owned native invoke."""

    window = getattr(context, "window", None)
    obj = getattr(context, "edit_object", None)
    if window is None or obj is None or getattr(obj, "type", None) != "MESH":
        return
    key = (int(window.as_pointer()), str(obj.data.name))
    session_state._GIZMO_TICKETS[key] = GizmoExclusionTicket(
        window_pointer=key[0],
        mesh_name=key[1],
        created_at=time.monotonic(),
        grace=grace,
    )


def clear_runtime_state() -> None:
    session_state._GIZMO_TICKETS.clear()
    session_state._GIZMO_TOMBSTONES.clear()
    session_state._GIZMO_MODAL_POINTERS_BY_WINDOW.clear()
    session_state._GIZMO_POLL_ARMED = False


def _view_contexts():
    manager = getattr(bpy.context, "window_manager", None)
    if manager is None:
        return
    for window in tuple(manager.windows):
        area = next((candidate for candidate in window.screen.areas if candidate.type == "VIEW_3D"), None)
        if area is None:
            continue
        region = next((candidate for candidate in area.regions if candidate.type == "WINDOW"), None)
        if region is not None:
            yield window, area, region


def _active_gizmo_kind(context) -> str | None:
    try:
        tool = context.workspace.tools.from_space_view3d_mode("EDIT_MESH", create=False)
    except (AttributeError, ReferenceError, RuntimeError):
        return None
    return _TOOL_IDS.get(getattr(tool, "idname", "")) if tool is not None else None


def _arm_context(context) -> tuple[str, int] | None:
    from . import session

    kind = _active_gizmo_kind(context)
    if kind is None or not session._single_edit_mesh_poll(context):
        return None
    obj = context.edit_object
    axes = matching.enabled_mesh_symmetry_axes(obj)
    if len(axes) != 1:
        return None
    return kind, axes[0][1]


def arm_required() -> bool:
    # Any escape would unregister the calling timer permanently, taking the
    # 1s keymap watcher down with it, so every failure reads as "not armed".
    for window, area, region in _view_contexts() or ():
        try:
            with bpy.context.temp_override(window=window, area=area, region=region):
                if _arm_context(bpy.context) is not None:
                    return True
        except Exception:  # noqa: BLE001
            continue
    return False


def prime_onset_state() -> None:
    session_state._GIZMO_MODAL_POINTERS_BY_WINDOW.clear()
    for window, _area, _region in _view_contexts() or ():
        pointer = int(window.as_pointer())
        session_state._GIZMO_MODAL_POINTERS_BY_WINDOW[pointer] = frozenset(
            _operator_pointer(operator) for _kind, operator in _extrude_modals(window)
        )


def _operator_pointer(operator) -> int:
    try:
        return int(operator.as_pointer())
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        return 0


def _operator_identifier(operator) -> str:
    values = (
        getattr(getattr(operator, "bl_rna", None), "identifier", ""),
        getattr(operator, "bl_idname", ""),
    )
    return " ".join(str(value).upper() for value in values if value)


def _extrude_modals(window):
    from . import session

    for operator in tuple(window.modal_operators):
        identifier = _operator_identifier(operator)
        for kind in _TOOL_IDS.values():
            profile = session.TOOL_PROFILES[kind]
            if profile.primary_wm_operator.upper() in identifier:
                yield kind, operator
                break


def _retire_tombstones(live_pointers: set[int]) -> None:
    # A dead wrapper's as_pointer() still returns the old address, so liveness
    # is tracked against the observed modal stacks: once a tombstoned pointer
    # leaves the stacks it is retired, and a later onset at the same address
    # is a recycled allocation, not the same operator.
    for pointer, live in tuple(session_state._GIZMO_TOMBSTONES.items()):
        if live and pointer not in live_pointers:
            session_state._GIZMO_TOMBSTONES[pointer] = False


def _collect_tickets(modals_by_window: dict[int, tuple[tuple[str, object], ...]]) -> None:
    now = time.monotonic()
    for key, ticket in tuple(session_state._GIZMO_TICKETS.items()):
        modals = modals_by_window.get(ticket.window_pointer, ())
        live_pointers = {_operator_pointer(operator) for _kind, operator in modals}
        if ticket.operator_pointer:
            if ticket.operator_pointer not in live_pointers:
                session_state._GIZMO_TICKETS.pop(key, None)
            continue
        if modals:
            operator = modals[0][1]
            pointer = _operator_pointer(operator)
            if pointer:
                ticket.operator_pointer = pointer
                session_state._GIZMO_TOMBSTONES[pointer] = True
            continue
        if now - ticket.created_at > ticket.grace:
            session_state._GIZMO_TICKETS.pop(key, None)


def poll_global() -> None:
    contexts = tuple(_view_contexts() or ())
    modals_by_window = {int(window.as_pointer()): tuple(_extrude_modals(window)) for window, _area, _region in contexts}
    all_live_pointers = {
        _operator_pointer(operator) for modals in modals_by_window.values() for _kind, operator in modals
    }
    _retire_tombstones(all_live_pointers)
    _collect_tickets(modals_by_window)

    live_windows = set(modals_by_window)
    for pointer in tuple(session_state._GIZMO_MODAL_POINTERS_BY_WINDOW):
        if pointer not in live_windows:
            session_state._GIZMO_MODAL_POINTERS_BY_WINDOW.pop(pointer, None)

    for window, area, region in contexts:
        window_pointer = int(window.as_pointer())
        modals = modals_by_window[window_pointer]
        current = frozenset(_operator_pointer(operator) for _kind, operator in modals)
        prior = session_state._GIZMO_MODAL_POINTERS_BY_WINDOW.get(window_pointer, frozenset())
        session_state._GIZMO_MODAL_POINTERS_BY_WINDOW[window_pointer] = current
        onset = current - prior
        for pointer in onset:
            if session_state._GIZMO_TOMBSTONES.get(pointer) is False:
                session_state._GIZMO_TOMBSTONES.pop(pointer, None)
        for kind, operator in modals:
            if _operator_pointer(operator) not in onset:
                continue
            try:
                with bpy.context.temp_override(window=window, area=area, region=region):
                    _adopt_onset(bpy.context, window, kind, operator)
            except Exception:
                traceback.print_exc()


def _ticket_exists(window_pointer: int, mesh_name: str) -> bool:
    return (window_pointer, mesh_name) in session_state._GIZMO_TICKETS


def _p_gizmo(context, window, kind: str, operator) -> tuple[object, int] | None:
    armed = _arm_context(context)
    if armed is None or armed[0] != kind:
        return None
    obj = context.edit_object
    if obj is None or obj.type != "MESH":
        return None
    pointer = _operator_pointer(operator)
    if not pointer or pointer in session_state._GIZMO_TOMBSTONES:
        return None
    if any(active.mesh_name == obj.data.name for active in session_state._SESSIONS.values()):
        return None
    window_pointer = int(window.as_pointer())
    if _ticket_exists(window_pointer, obj.data.name):
        return None
    if not any(_operator_pointer(candidate) == pointer for _candidate_kind, candidate in _extrude_modals(window)):
        return None
    return obj, armed[1]


def _proportional_edit_enabled(operator, kind: str) -> bool | None:
    # Gizmo macros leave operator.use_proportional_edit False even when the
    # user toggle is on (measured: TRANSFORM_OT_translate=False while
    # tool_settings.use_proportional_edit=True). Decline if either is true.
    try:
        if bool(getattr(bpy.context.tool_settings, "use_proportional_edit", False)):
            return True
    except (AttributeError, ReferenceError, RuntimeError):
        return None
    candidates = [operator]
    child_name = (
        "TRANSFORM_OT_shrink_fatten"
        if kind in {"EXTRUDE_SHRINK_FATTEN", "EXTRUDE_FACES_INDIV"}
        else "TRANSFORM_OT_translate"
    )
    child = getattr(operator, child_name, None)
    if child is not None:
        candidates.append(child)
    seen = False
    for candidate in candidates:
        try:
            if hasattr(candidate, "use_proportional_edit"):
                seen = True
                if bool(candidate.use_proportional_edit):
                    return True
        except (AttributeError, ReferenceError, RuntimeError):
            return None
    return False if seen else None


def _remember_decline(context, obj, axis_index: int, kind: str, reason: str, pointer: int) -> None:
    from . import history, session

    token = session_state._new_history_token()
    settings = context.scene.ydd_symmetric_edit
    declined = KnifeSession(
        window_pointer=session._window_key(context),
        area_pointer=context.area.as_pointer(),
        region_pointer=context.region.as_pointer(),
        object_name=obj.name,
        mesh_name=obj.data.name,
        axis_index=axis_index,
        source_side=settings.source_side,
        tolerance=settings.tolerance,
        mirror_face_ids={},
        hidden_by_face_id={},
        carrier_frames={},
        mesh_select_mode=_mesh_select_mode(context),
        started_at=time.monotonic(),
        tool_kind=kind,
        history_token=token,
        prepare_disposition="DECLINE",
        prepare_disposition_reason=reason,
        route=GIZMO_ROUTE,
        gizmo_operator_pointer=pointer,
    )
    history._remember_history_session(declined, context)
    session_state._HISTORY_RECORDS[token].status = "COMMITTED"


def _adopt_onset(context, window, kind: str, operator) -> None:
    preflight = _p_gizmo(context, window, kind, operator)
    if preflight is None:
        return
    _obj, axis_index = preflight
    obj = context.edit_object
    if obj is None or obj.type != "MESH":
        return
    pointer = _operator_pointer(operator)
    session_state._GIZMO_TOMBSTONES[pointer] = True

    if kind == "EXTRUDE_MANIFOLD":
        _remember_decline(context, obj, axis_index, kind, "gizmo Manifold extrude is not mirrored", pointer)
        return
    proportional = _proportional_edit_enabled(operator, kind)
    if proportional is None:
        return
    if proportional:
        _remember_decline(context, obj, axis_index, kind, "proportional editing is enabled", pointer)
        return

    bm = bmesh.from_edit_mesh(obj.data)
    settings = context.scene.ydd_symmetric_edit
    plan, reason = _build_read_plan(
        bm,
        axis_index,
        float(settings.tolerance),
        kind,
        _mesh_select_mode(context),
    )
    if plan is None:
        _remember_decline(context, obj, axis_index, kind, reason or "gizmo extrude could not be adopted", pointer)
        return
    _commit_adoption(context, obj, operator, kind, axis_index, plan)


def _mesh_select_mode(context) -> MeshSelectionMode:
    mode = context.tool_settings.mesh_select_mode
    return MeshSelectionMode(vertices=bool(mode[0]), edges=bool(mode[1]), faces=bool(mode[2]))


def _cycle_matches(first: tuple, second: tuple) -> bool:
    return extrude._signatures_match(first, second)


def _classify_vertices(bm, axis_index: int, tolerance: float):
    # Selected vertices are the copy candidates and, at detection time, may
    # still coincide with their origins; they must stay out of the survivor
    # matching entirely (rails and census validate them later).  Mid-modal the
    # vertex flags may lag the edge/face flags, so all three domains count.
    vertices = tuple(bm.verts)
    new: set[bmesh.types.BMVert] = {vertex for vertex in vertices if vertex.select}
    for edge in bm.edges:
        if edge.select:
            new.update(edge.verts)
    for face in bm.faces:
        if face.select:
            new.update(face.verts)
    unselected = [vertex for vertex in vertices if vertex not in new]
    lookup = matching.build_vertex_mirror_lookup([vertex.co for vertex in unselected], axis_index, tolerance)
    assigned = lookup.find_all_mirrored([vertex.co for vertex in unselected])
    survivors: set[bmesh.types.BMVert] = set()
    partners: dict[bmesh.types.BMVert, bmesh.types.BMVert] = {}
    for vertex, index in zip(unselected, assigned, strict=True):
        if index is None:
            print(
                f"YSE_GIZMO_DEBUG unpaired co={tuple(vertex.co)} select={vertex.select} nsel={len(new)}"
                f" census={len(bm.verts)}/{len(bm.edges)}/{len(bm.faces)}"
            )
            return None, None, None, "an unselected vertex has no unique mirrored counterpart"
        survivors.add(vertex)
        partners[vertex] = unselected[index]
    return survivors, new, partners, None


def _edge_between(first, second):
    return next((edge for edge in first.link_edges if edge.is_valid and edge.other_vert(first) is second), None)


def _face_with_cycle(start, cycle):
    wanted = set(cycle)
    matches = []
    for face in start.link_faces:
        if not face.is_valid or len(face.verts) != len(cycle) or set(face.verts) != wanted:
            continue
        live = tuple(loop.vert for loop in face.loops)
        if _cycle_matches(live, cycle):
            matches.append(face)
    return matches


def _build_read_plan(bm, axis_index: int, tolerance: float, kind: str, mesh_select_mode: MeshSelectionMode):
    survivors, copies, vertex_partner, reason = _classify_vertices(bm, axis_index, tolerance)
    if survivors is None or copies is None or vertex_partner is None:
        return None, reason
    if not copies:
        return None, "the native gizmo made no extrusion copies"

    copy_origin: dict[bmesh.types.BMVert, bmesh.types.BMVert] = {}
    for copy in copies:
        origins = {
            edge.other_vert(copy)
            for edge in copy.link_edges
            if edge.is_valid and edge.other_vert(copy) in survivors and not edge.other_vert(copy).select
        }
        if len(origins) == 0:
            return None, "a copy has no unique surviving rail origin"
        if len(origins) > 1:
            return None, "a copy has multiple surviving rail origins"
        origin = next(iter(origins))
        if abs(float(origin.co[axis_index])) <= tolerance:
            return None, "an extrusion origin lies on the mirror plane"
        partner = vertex_partner.get(origin)
        if partner is None or partner.hide:
            return None, "an extrusion origin has a missing or hidden mirror vertex"
        copy_origin[copy] = origin

    survivor_edges = {edge for edge in bm.edges if all(vertex in survivors for vertex in edge.verts)}
    new_edges = set(bm.edges) - survivor_edges
    edge_partner: dict[bmesh.types.BMEdge, bmesh.types.BMEdge] = {}
    one_sided_edges: list[bmesh.types.BMEdge] = []
    for edge in survivor_edges:
        first = vertex_partner[edge.verts[0]]
        second = vertex_partner[edge.verts[1]]
        partner = _edge_between(first, second)
        if partner is None:
            one_sided_edges.append(edge)
        else:
            if partner.hide:
                return None, "a mirrored counterpart edge is hidden"
            edge_partner[edge] = partner
    if len(set(edge_partner.values())) != len(edge_partner):
        return None, "the surviving edge correspondence is not injective"

    survivor_faces = {face for face in bm.faces if all(vertex in survivors for vertex in face.verts)}
    new_faces = set(bm.faces) - survivor_faces
    face_partner: dict[bmesh.types.BMFace, bmesh.types.BMFace] = {}
    one_sided_faces: list[bmesh.types.BMFace] = []
    for face in survivor_faces:
        reflected = tuple(vertex_partner[loop.vert] for loop in reversed(face.loops))
        matches = _face_with_cycle(reflected[0], reflected)
        if len(matches) > 1:
            return None, "a reflected face has multiple live counterparts"
        if not matches:
            one_sided_faces.append(face)
        else:
            partner = matches[0]
            if partner.hide:
                return None, "a mirrored counterpart face is hidden"
            face_partner[face] = partner
    if len(set(face_partner.values())) != len(face_partner):
        return None, "the surviving face correspondence is not injective"

    mirror_origins = {vertex_partner[origin] for origin in copy_origin.values()}
    deleted_edge_targets = tuple(edge for edge in one_sided_edges if set(edge.verts).issubset(mirror_origins))
    deleted_face_targets = tuple(face for face in one_sided_faces if set(face.verts).issubset(mirror_origins))
    if len(deleted_edge_targets) != len(one_sided_edges) or len(deleted_face_targets) != len(one_sided_faces):
        return None, "the live mesh is asymmetric outside the mirrored extrusion region"
    if any(edge.hide for edge in deleted_edge_targets) or any(face.hide for face in deleted_face_targets):
        return None, "a mirrored region counterpart is hidden"

    selected_copy_edges = tuple(
        edge for edge in new_edges if edge.select and all(vertex in copies for vertex in edge.verts)
    )
    copy_source_cycles: dict[bmesh.types.BMVert, tuple[bmesh.types.BMVert, ...]] = {}
    if mesh_select_mode.faces:
        caps = [face for face in new_faces if all(vertex in copies for vertex in face.verts)]
        source_cycles = [tuple(copy_origin[loop.vert] for loop in face.loops) for face in caps]
        deleted_cycles = [
            tuple(vertex_partner[loop.vert] for loop in reversed(face.loops)) for face in deleted_face_targets
        ]
        unmatched = list(deleted_cycles)
        for cap, cycle in zip(caps, source_cycles, strict=True):
            matches = [candidate for candidate in unmatched if _cycle_matches(cycle, candidate)]
            if len(matches) != 1:
                return None, "cap faces do not bijectively match the reconstructed deleted faces"
            matched = matches[0]
            unmatched.remove(matched)
            if kind == "EXTRUDE_FACES_INDIV":
                for copy in cap.verts:
                    if copy in copy_source_cycles:
                        return None, "an individual-face copy belongs to multiple cap faces"
                    copy_source_cycles[copy] = matched
        if unmatched or len(caps) != len(deleted_face_targets):
            return None, "the reconstructed deleted face set is incomplete"
    elif mesh_select_mode.edges:
        if deleted_edge_targets or deleted_face_targets:
            return None, "edge-mode gizmo extrusion unexpectedly deleted topology"
        if not selected_copy_edges:
            return None, "edge-mode gizmo extrusion has no selected copy edges"
        for edge in selected_copy_edges:
            origins = tuple(copy_origin[vertex] for vertex in edge.verts)
            if _edge_between(*origins) is None:
                return None, "the copy edge graph is not isomorphic to the origin graph"
            rail = set(edge.verts) | set(origins)
            quads = [face for face in new_faces if len(face.verts) == 4 and set(face.verts) == rail]
            if len(quads) != 1:
                return None, "a selected copy edge has no unique side quad"
    elif mesh_select_mode.vertices:
        if deleted_edge_targets or deleted_face_targets:
            return None, "vertex-mode gizmo extrusion unexpectedly deleted topology"
    else:
        return None, "the mesh selection mode is unsupported"

    if kind == "EXTRUDE_FACES_INDIV" and set(copy_source_cycles) != copies:
        return None, "individual-face copy attribution is incomplete"
    # Structural bijections bound every element, and this pins the totals to
    # the per-kind generation formulas so a shape outside them cannot adopt.
    live_created = (len(copies), len(new_edges), len(new_faces))
    live_deleted = (0, len(deleted_edge_targets), len(deleted_face_targets))
    if mesh_select_mode.faces and kind == "EXTRUDE_FACES_INDIV":
        corner_total = sum(len(face.verts) for face in deleted_face_targets)
        expected_created = (corner_total, 2 * corner_total, corner_total + len(deleted_face_targets))
        expected_deleted = (0, 0, len(deleted_face_targets))
    elif mesh_select_mode.faces:
        region_vertices = set(copy_origin.values())
        region_edges = {edge for face in deleted_face_targets for edge in face.edges}
        expected_created = (
            len(region_vertices),
            len(region_vertices) + len(region_edges),
            (len(region_edges) - len(deleted_edge_targets)) + len(deleted_face_targets),
        )
        expected_deleted = (0, len(deleted_edge_targets), len(deleted_face_targets))
    elif mesh_select_mode.edges:
        expected_created = (len(copies), len(copies) + len(selected_copy_edges), len(selected_copy_edges))
        expected_deleted = (0, 0, 0)
    else:
        expected_created = (len(copies), len(copies), 0)
        expected_deleted = (0, 0, 0)
    if live_created != expected_created or live_deleted != expected_deleted:
        return None, "the extrusion does not match the pinned census formula"
    return (
        _ReadPlan(
            bm=bm,
            survivors=survivors,
            copies=copies,
            vertex_partner=vertex_partner,
            copy_origin=copy_origin,
            survivor_edges=survivor_edges,
            new_edges=new_edges,
            edge_partner=edge_partner,
            deleted_edge_targets=deleted_edge_targets,
            survivor_faces=survivor_faces,
            new_faces=new_faces,
            face_partner=face_partner,
            deleted_face_targets=deleted_face_targets,
            selected_copy_edges=selected_copy_edges,
            copy_source_cycles=copy_source_cycles,
            mesh_select_mode=mesh_select_mode,
        ),
        None,
    )


def _new_int_layer(sequence, name: str):
    old = sequence.layers.int.get(name)
    if old is not None:
        sequence.layers.int.remove(old)
    return sequence.layers.int.new(name)


def _plan_to_indices(plan: _ReadPlan) -> dict:
    return {
        "survivors": [vertex.index for vertex in plan.survivors],
        "copies": [vertex.index for vertex in plan.copies],
        "vertex_partner": [(key.index, value.index) for key, value in plan.vertex_partner.items()],
        "copy_origin": [(key.index, value.index) for key, value in plan.copy_origin.items()],
        "survivor_edges": [edge.index for edge in plan.survivor_edges],
        "new_edges": [edge.index for edge in plan.new_edges],
        "edge_partner": [(key.index, value.index) for key, value in plan.edge_partner.items()],
        "deleted_edge_targets": [edge.index for edge in plan.deleted_edge_targets],
        "survivor_faces": [face.index for face in plan.survivor_faces],
        "new_faces": [face.index for face in plan.new_faces],
        "face_partner": [(key.index, value.index) for key, value in plan.face_partner.items()],
        "deleted_face_targets": [face.index for face in plan.deleted_face_targets],
        "selected_copy_edges": [edge.index for edge in plan.selected_copy_edges],
        "copy_source_cycles": [
            (key.index, [vertex.index for vertex in cycle]) for key, cycle in plan.copy_source_cycles.items()
        ],
    }


def _plan_from_indices(bm, data: dict, mesh_select_mode: MeshSelectionMode) -> _ReadPlan:
    verts, edges, faces = bm.verts, bm.edges, bm.faces
    return _ReadPlan(
        bm=bm,
        survivors={verts[index] for index in data["survivors"]},
        copies={verts[index] for index in data["copies"]},
        vertex_partner={verts[key]: verts[value] for key, value in data["vertex_partner"]},
        copy_origin={verts[key]: verts[value] for key, value in data["copy_origin"]},
        survivor_edges={edges[index] for index in data["survivor_edges"]},
        new_edges={edges[index] for index in data["new_edges"]},
        edge_partner={edges[key]: edges[value] for key, value in data["edge_partner"]},
        deleted_edge_targets=tuple(edges[index] for index in data["deleted_edge_targets"]),
        survivor_faces={faces[index] for index in data["survivor_faces"]},
        new_faces={faces[index] for index in data["new_faces"]},
        face_partner={faces[key]: faces[value] for key, value in data["face_partner"]},
        deleted_face_targets=tuple(faces[index] for index in data["deleted_face_targets"]),
        selected_copy_edges=tuple(edges[index] for index in data["selected_copy_edges"]),
        copy_source_cycles={
            verts[key]: tuple(verts[index] for index in cycle) for key, cycle in data["copy_source_cycles"]
        },
        mesh_select_mode=mesh_select_mode,
    )


def _stamp_and_snapshot(plan: _ReadPlan, axis_index: int, tolerance: float, kind: str) -> ExtrudeSnapshot:
    # Creating custom-data layers invalidates every held element reference,
    # so the plan round-trips through indices across all layer creation.
    bm = plan.bm
    for sequence in (bm.verts, bm.edges, bm.faces):
        sequence.index_update()
    plan_indices = _plan_to_indices(plan)
    vertex_layer = _new_int_layer(bm.verts, layer_names.VERT_SESSION_ID_LAYER)
    edge_layer = _new_int_layer(bm.edges, layer_names.EDGE_ORIGINAL_LAYER)
    face_layer = _new_int_layer(bm.faces, layer_names.FACE_ID_LAYER)
    selection.snapshot_live_hidden(bm)
    for sequence in (bm.verts, bm.edges, bm.faces):
        sequence.ensure_lookup_table()
    plan = _plan_from_indices(bm, plan_indices, plan.mesh_select_mode)

    next_id = 1
    vertex_ids: dict[bmesh.types.BMVert, int] = {}
    for vertex in bm.verts:
        if vertex not in plan.survivors:
            continue
        vertex_ids[vertex] = next_id
        vertex[vertex_layer] = next_id
        next_id += 1
    for copy, origin in plan.copy_origin.items():
        copy[vertex_layer] = vertex_ids[origin]

    edge_ids: dict[bmesh.types.BMEdge, int] = {}
    next_id = 1
    for edge in bm.edges:
        marker = next_id if edge in plan.survivor_edges else 0
        edge[edge_layer] = marker
        if marker:
            edge_ids[edge] = marker
            next_id += 1
    face_ids: dict[bmesh.types.BMFace, int] = {}
    next_id = 1
    for face in bm.faces:
        face_id = next_id if face in plan.survivor_faces else 0
        face[face_layer] = face_id
        if face_id:
            face_ids[face] = face_id
            next_id += 1

    synthetic_edges = {edge: -(index + 1) for index, edge in enumerate(plan.deleted_edge_targets)}
    synthetic_faces = {face: -(index + 1) for index, face in enumerate(plan.deleted_face_targets)}
    vertex_preop = tuple(
        (vertex_ids[vertex], Coordinate3D(float(vertex.co.x), float(vertex.co.y), float(vertex.co.z)))
        for vertex in plan.survivors
    )
    vertex_pairs = tuple((vertex_ids[vertex], vertex_ids[plan.vertex_partner[vertex]]) for vertex in plan.survivors)

    edge_pairs = [(edge_ids[edge], edge_ids[partner]) for edge, partner in plan.edge_partner.items()]
    edge_endpoints = [
        (edge_ids[edge], (vertex_ids[edge.verts[0]], vertex_ids[edge.verts[1]])) for edge in plan.survivor_edges
    ]
    for target, synthetic in synthetic_edges.items():
        target_id = edge_ids[target]
        edge_pairs.extend(((synthetic, target_id), (target_id, synthetic)))
        source = tuple(vertex_ids[plan.vertex_partner[vertex]] for vertex in target.verts)
        edge_endpoints.append((synthetic, (source[0], source[1])))

    face_pairs = [(face_ids[face], face_ids[partner]) for face, partner in plan.face_partner.items()]
    face_corners = [
        (face_ids[face], tuple(vertex_ids[loop.vert] for loop in face.loops)) for face in plan.survivor_faces
    ]
    for target, synthetic in synthetic_faces.items():
        target_id = face_ids[target]
        face_pairs.extend(((synthetic, target_id), (target_id, synthetic)))
        source = tuple(vertex_ids[plan.vertex_partner[loop.vert]] for loop in reversed(target.loops))
        face_corners.append((synthetic, source))

    origin_ids = frozenset(vertex_ids[origin] for origin in plan.copy_origin.values())
    all_preop_edges = dict(edge_endpoints)
    if plan.mesh_select_mode.edges:
        region_edge_markers = frozenset(
            edge_ids[_edge_between(plan.copy_origin[edge.verts[0]], plan.copy_origin[edge.verts[1]])]
            for edge in plan.selected_copy_edges
        )
    elif plan.mesh_select_mode.faces:
        region_edge_markers = frozenset(
            marker
            for marker, endpoints in all_preop_edges.items()
            if endpoints[0] in origin_ids and endpoints[1] in origin_ids
        )
    else:
        region_edge_markers = frozenset()
    region_face_ids = frozenset(synthetic_faces.values()) if plan.mesh_select_mode.faces else frozenset()

    created = (len(plan.copies), len(plan.new_edges), len(plan.new_faces))
    deleted = (0, len(synthetic_edges), len(synthetic_faces))
    net = (
        created[0] - deleted[0],
        created[1] - deleted[1],
        created[2] - deleted[2],
    )
    expected_preop = (
        len(plan.survivors),
        len(plan.survivor_edges) + len(synthetic_edges),
        len(plan.survivor_faces) + len(synthetic_faces),
    )

    copy_keys = []
    for copy, origin in plan.copy_origin.items():
        signature = tuple(vertex_ids[vertex] for vertex in plan.copy_source_cycles.get(copy, ()))
        copy_keys.append((vertex_ids[origin], signature))
    hidden_vertex_ids = frozenset(vertex_ids[vertex] for vertex in plan.survivors if vertex.hide)
    hidden_edge_markers = frozenset(edge_ids[edge] for edge in plan.survivor_edges if edge.hide)
    hidden_face_ids = frozenset(face_ids[face] for face in plan.survivor_faces if face.hide)
    return ExtrudeSnapshot(
        axis_index=axis_index,
        tolerance=tolerance,
        tool_kind=kind,
        route_kmi_properties=(),
        mesh_select_mode=plan.mesh_select_mode,
        selected_vertex_ids=origin_ids,
        selected_edge_markers=region_edge_markers,
        selected_face_ids=region_face_ids,
        vertex_preop=vertex_preop,
        vertex_pairs=tuple(vertex_pairs),
        edge_pairs=tuple(edge_pairs),
        face_pairs=tuple(face_pairs),
        hidden_vertex_ids=hidden_vertex_ids,
        hidden_edge_markers=hidden_edge_markers,
        hidden_face_ids=hidden_face_ids,
        face_corners=tuple(face_corners),
        edge_endpoints=tuple(edge_endpoints),
        vertex_count=expected_preop[0],
        edge_count=expected_preop[1],
        face_count=expected_preop[2],
        route=GIZMO_ROUTE,
        region_vertex_ids=origin_ids,
        region_edge_markers=region_edge_markers,
        region_face_ids=region_face_ids,
        expected_created=created,
        expected_deleted=deleted,
        expected_net=net,
        gizmo_copy_keys=tuple(sorted(copy_keys)),
    )


def _commit_adoption(context, obj, operator, kind: str, axis_index: int, plan: _ReadPlan) -> None:
    from . import history, session, watcher

    settings = context.scene.ydd_symmetric_edit
    history_token = session_state._new_history_token()
    registered = False
    remembered = False
    completed = False
    try:
        extrude_snapshot = _stamp_and_snapshot(plan, axis_index, float(settings.tolerance), kind)
        bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
        face_layer = plan.bm.faces.layers.int.get(layer_names.FACE_ID_LAYER)
        hidden = {
            FaceId(int(face[face_layer])): bool(face.hide)
            for face in plan.bm.faces
            if face_layer is not None and int(face[face_layer]) > 0
        }
        adopted = KnifeSession(
            window_pointer=session._window_key(context),
            area_pointer=context.area.as_pointer(),
            region_pointer=context.region.as_pointer(),
            object_name=obj.name,
            mesh_name=obj.data.name,
            axis_index=axis_index,
            source_side=settings.source_side,
            tolerance=settings.tolerance,
            mirror_face_ids={},
            hidden_by_face_id=hidden,
            carrier_frames={},
            mesh_select_mode=plan.mesh_select_mode,
            started_at=time.monotonic(),
            tool_kind=kind,
            history_token=history_token,
            saw_modal=True,
            symmetry_flags=SymmetryAxes(
                x=bool(obj.use_mesh_mirror_x),
                y=bool(obj.use_mesh_mirror_y),
                z=bool(obj.use_mesh_mirror_z),
            ),
            extrude=extrude_snapshot,
            route=GIZMO_ROUTE,
            gizmo_operator_pointer=_operator_pointer(operator),
        )
        session_state._SESSIONS[adopted.window_pointer] = adopted
        registered = True
        history._remember_history_session(adopted, context)
        remembered = True
        watcher._schedule_passthrough_watcher(adopted.window_pointer, history_token)
        completed = True
    finally:
        if not completed:
            if registered:
                session_state._SESSIONS.pop(session._window_key(context), None)
            if remembered:
                session_state._HISTORY_RECORDS.pop(history_token, None)
            try:
                snapshot.remove_temporary_layers(plan.bm)
            except Exception:
                traceback.print_exc()
                print("ydd Symmetric Edit: gizmo adoption rollback could not remove temporary layers")
            finally:
                try:
                    bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
                except Exception:
                    traceback.print_exc()


def classify_live(
    bm: bmesh.types.BMesh,
    extrude_snapshot: ExtrudeSnapshot,
) -> tuple[extrude.ExtrudeClassification | None, str | None]:
    vertex_layer = bm.verts.layers.int.get(layer_names.VERT_SESSION_ID_LAYER)
    if vertex_layer is None:
        return None, "the gizmo session vertex ID layer is missing"
    groups = extrude._verts_by_session_id(bm, vertex_layer)
    expected_by_vid: dict[int, list[tuple[int, ...]]] = {}
    for vertex_id, signature in extrude_snapshot.gizmo_copy_keys:
        expected_by_vid.setdefault(vertex_id, []).append(signature)

    origins = {}
    copies = {}
    instances = []
    freeze = []
    preop = extrude_snapshot.vertex_preop_map()
    for vertex_id, signatures in expected_by_vid.items():
        live = groups.get(vertex_id, [])
        origin_preop = preop.get(vertex_id)
        if origin_preop is None:
            return None, "a gizmo origin is missing from the snapshot"
        origin_hits = [
            vertex
            for vertex in live
            if matching.coordinates_match(vertex.co, origin_preop.as_tuple(), extrude_snapshot.tolerance)
        ]
        if len(origin_hits) != 1:
            return None, "a gizmo origin did not reconnect uniquely"
        origin = origin_hits[0]
        remaining = [vertex for vertex in live if vertex is not origin]
        if len(remaining) != len(signatures):
            return None, "a gizmo copy group has missing or excess instances"
        origins[vertex_id] = origin
        if signatures == [()]:
            copies[vertex_id] = remaining[0]
            assignments = [(remaining[0], ())]
        else:
            assignments = _resolve_ncopy_assignments(remaining, signatures, vertex_layer, extrude_snapshot)
            if assignments is None:
                return None, "a gizmo individual-face copy did not resolve uniquely"
        for copy, signature in assignments:
            instance = extrude.ExtrudeCopyInstance(
                vertex_id=vertex_id,
                vertex=copy,
                entity_class="d" if signature else "b",
                source_face_signature=signature,
            )
            instances.append(instance)
            freeze.append(
                ExtrudeFreezeEntry(
                    vertex_id=vertex_id,
                    entity_class=instance.entity_class,
                    origin_preop=origin_preop,
                    copy_post=Coordinate3D(float(copy.co.x), float(copy.co.y), float(copy.co.z)),
                    source_face_signature=signature,
                )
            )
    if not instances:
        return None, "the gizmo snapshot contains no copy attribution"
    frozen = tuple(sorted(freeze, key=lambda entry: (entry.vertex_id, entry.source_face_signature)))
    return (
        extrude.ExtrudeClassification(
            origins=origins,
            copies=copies,
            copy_instances=tuple(instances),
            vanished_preop={},
            freeze=frozen,
        ),
        None,
    )


def _resolve_ncopy_assignments(remaining, signatures, vertex_layer, extrude_snapshot):
    assignments = []
    unused = list(remaining)
    for signature in signatures:
        candidates = []
        for vertex in unused:
            for face in vertex.link_faces:
                if not face.is_valid:
                    continue
                live = tuple(int(loop.vert[vertex_layer]) for loop in face.loops)
                if _cycle_matches(live, signature):
                    candidates.append(vertex)
                    break
        if len(candidates) != 1:
            return None
        chosen = candidates[0]
        unused.remove(chosen)
        assignments.append((chosen, signature))
    return assignments if not unused else None


def reconnect_freeze(
    bm: bmesh.types.BMesh,
    extrude_snapshot: ExtrudeSnapshot,
    freeze: tuple[ExtrudeFreezeEntry, ...],
) -> tuple[extrude.ExtrudeClassification | None, str | None]:
    classified, reason = classify_live(bm, extrude_snapshot)
    if classified is None:
        return None, reason
    frozen_keys = tuple(sorted((entry.vertex_id, entry.source_face_signature) for entry in freeze))
    live_keys = tuple(
        sorted((instance.vertex_id, instance.source_face_signature) for instance in classified.copy_instances)
    )
    if live_keys != frozen_keys:
        return None, "the gizmo freeze attribution table no longer matches"
    tolerance = extrude_snapshot.tolerance
    freeze_by_key = {(entry.vertex_id, entry.source_face_signature): entry for entry in freeze}
    for instance in classified.copy_instances:
        entry = freeze_by_key[(instance.vertex_id, instance.source_face_signature)]
        if not matching.coordinates_match(instance.vertex.co, entry.copy_post.as_tuple(), tolerance):
            return None, "a frozen gizmo copy moved after classification"
    classified.freeze = freeze
    return classified, None


def describe_source(
    bm: bmesh.types.BMesh,
    extrude_snapshot: ExtrudeSnapshot,
    classified: extrude.ExtrudeClassification,
) -> tuple[extrude.ExtrudeSourceDescription | None, str | None]:
    vertex_layer = bm.verts.layers.int.get(layer_names.VERT_SESSION_ID_LAYER)
    edge_layer = bm.edges.layers.int.get(layer_names.EDGE_ORIGINAL_LAYER)
    face_layer = bm.faces.layers.int.get(layer_names.FACE_ID_LAYER)
    if vertex_layer is None or edge_layer is None or face_layer is None:
        return None, "temporary gizmo topology markers are missing"
    copy_set = {instance.vertex for instance in classified.copy_instances if instance.vertex.is_valid}
    new_edges = [edge for edge in bm.edges if int(edge[edge_layer]) == 0 and any(v in copy_set for v in edge.verts)]
    new_faces = [face for face in bm.faces if int(face[face_layer]) == 0 and any(v in copy_set for v in face.verts)]
    signatures = []
    for face in new_faces:
        signature, reason = extrude._face_corner_signature(face, vertex_layer, extrude_snapshot, classified)
        if signature is None:
            return None, reason or "a gizmo new face could not be described"
        signatures.append(signature)
    deleted_faces = tuple(sorted(face_id for face_id in extrude_snapshot.selected_face_ids if face_id < 0))
    deleted_edges = tuple(sorted(marker for marker in extrude_snapshot.edge_endpoint_map() if marker < 0))
    created = (len(copy_set), len(new_edges), len(new_faces))
    deleted = (0, len(deleted_edges), len(deleted_faces))
    net = (
        created[0] - deleted[0],
        created[1] - deleted[1],
        created[2] - deleted[2],
    )
    expected = (
        extrude_snapshot.expected_created,
        extrude_snapshot.expected_deleted,
        extrude_snapshot.expected_net,
    )
    if expected != (created, deleted, net):
        return None, f"gizmo source census {created}/{deleted}/{net} does not match {expected}"
    return (
        extrude.ExtrudeSourceDescription(
            new_verts=list(copy_set),
            new_edges=new_edges,
            new_faces=new_faces,
            face_signatures=tuple(signatures),
            deleted_face_ids=deleted_faces,
            deleted_edge_markers=deleted_edges,
            deleted_vertex_ids=(),
            created=created,
            deleted=deleted,
            net=net,
            f12_shape=None,
        ),
        None,
    )
