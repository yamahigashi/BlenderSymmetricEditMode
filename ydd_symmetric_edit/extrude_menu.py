# SPDX-License-Identifier: GPL-3.0-or-later

"""Alt+E extrude menu clone: opener, YSE_MT_extrude, and Stage 3a–3c wrappers."""

from __future__ import annotations

import traceback
from typing import TYPE_CHECKING, Literal, cast

import bmesh
import bpy
from bpy.props import StringProperty

from . import matching, session_state
from .session import (
    EXTRUDE_TOOL_KINDS,
    _find_saved_view,
    _find_window,
    _prepare_session,
    _single_edit_mesh_poll,
    _window_key,
    cleanup_session,
)
from .watcher import _capture_confirmed_extrude_result, _native_tool_is_active

EXTRUDE_MENU = "YSE_MT_extrude"
OPENER_IDNAME = "mesh.ydd_symmetric_edit_extrude_menu"
WRAPPER_FACES = "mesh.ydd_symmetric_edit_extrude_faces"
WRAPPER_ALONG = "mesh.ydd_symmetric_edit_extrude_along_normals"
WRAPPER_INDIV = "mesh.ydd_symmetric_edit_extrude_individual_faces"
WRAPPER_EDGES = "mesh.ydd_symmetric_edit_extrude_edges"
WRAPPER_VERTS = "mesh.ydd_symmetric_edit_extrude_vertices"
NATIVE_FACES = "view3d.edit_mesh_extrude_move_normal"
NATIVE_ALONG = "view3d.edit_mesh_extrude_move_shrink_fatten"
NATIVE_INDIV = "mesh.extrude_faces_move"
NATIVE_EDGES = "mesh.extrude_edges_move"
NATIVE_VERTS = "mesh.extrude_vertices_move"

_PrepareStatus = Literal["PREPARED", "CONFLICT", "BYPASS"]


def _addon_routes_enabled() -> bool:
    from . import keymaps

    return bool(keymaps._RUNNING and keymaps._ENABLED)


def _invoke_native_extrude_item(idname: str) -> set[str]:
    """Launch a native menu-item dispatcher. Tests may wrap this."""

    namespace, name = idname.split(".", 1)
    operator = getattr(getattr(bpy.ops, namespace), name)
    return cast(set[str], operator("INVOKE_DEFAULT"))


def _wrapper_gates_bypass(context) -> bool:
    """True when the wrapper should launch native without preparing a session."""

    if not _addon_routes_enabled():
        return True
    if not _single_edit_mesh_poll(context):
        return True
    obj = context.edit_object
    if obj is None or obj.type != "MESH":
        return True
    if len(matching.enabled_mesh_symmetry_axes(obj)) != 1:
        return True
    bm = bmesh.from_edit_mesh(obj.data)
    return len(bm.faces) == 0


def _find_prior_session_same_mesh(context):
    obj = context.edit_object
    if obj is None or obj.type != "MESH":
        return None
    mesh_name = obj.data.name
    current_window = context.window.as_pointer() if context.window is not None else 0
    return next(
        (
            session
            for session in session_state._SESSIONS.values()
            if session.mesh_name == mesh_name and session.window_pointer != current_window
        ),
        None,
    )


def _classify_wrapper_prepare(context) -> tuple[_PrepareStatus, object | None]:
    prior = _find_prior_session_same_mesh(context)
    if prior is not None:
        return "CONFLICT", prior
    if _wrapper_gates_bypass(context):
        return "BYPASS", None
    return "PREPARED", None


def _prior_still_modal(session) -> bool:
    # Start-grace / not-yet-saw_modal is still modal: do not finish mid-drag.
    if not session.saw_modal:
        return True
    window = _find_window(session.window_pointer)
    if window is None:
        return False
    return _native_tool_is_active(window, session.tool_kind)


def _sync_finish_prior(session) -> bool:
    from .operators import _invoke_finish_operator

    window, area, region = _find_saved_view(session)
    if window is None or area is None or region is None:
        return False
    try:
        with bpy.context.temp_override(window=window, area=area, region=region):
            result = _invoke_finish_operator()
    except Exception:
        traceback.print_exc()
        return False
    return "FINISHED" in result


def _resolve_conflict(prior) -> bool:
    """Finish a post-confirm prior session. False means do not launch later."""

    if _prior_still_modal(prior):
        return False
    if prior.tool_kind in EXTRUDE_TOOL_KINDS:
        if not _capture_confirmed_extrude_result(prior):
            return False
    return _sync_finish_prior(prior)


def _invoke_extrude_wrapper(context, report, *, tool_kind: str, native_idname: str) -> set[str]:
    prior = _find_prior_session_same_mesh(context)
    if prior is not None:
        if not _resolve_conflict(prior):
            return {"CANCELLED"}
        if _find_prior_session_same_mesh(context) is not None:
            return {"CANCELLED"}

    if _wrapper_gates_bypass(context):
        return _invoke_native_extrude_item(native_idname)

    created_key: int | None = None
    try:
        prepared = _prepare_session(context, report, tool_kind=tool_kind)
        if not prepared:
            if _find_prior_session_same_mesh(context) is not None:
                return {"CANCELLED"}
            if _wrapper_gates_bypass(context):
                return _invoke_native_extrude_item(native_idname)
            return {"CANCELLED"}
        created_key = _window_key(context)
        result = _invoke_native_extrude_item(native_idname)
    except Exception:
        if created_key is not None:
            cleanup_session(created_key)
        raise

    if "CANCELLED" in result and "FINISHED" not in result:
        if created_key is not None:
            cleanup_session(created_key)
        return result

    # Dispatcher FINISHED is not completion; watcher finishes the session.
    return {"FINISHED"}


class MESH_OT_ydd_symmetric_edit_extrude_menu(bpy.types.Operator):
    """Verify the Alt+E route, then open YSE_MT_extrude."""

    bl_idname = OPENER_IDNAME
    bl_label = "Extrude Menu"
    bl_options = {"INTERNAL"}

    if TYPE_CHECKING:
        route_key: str
    else:
        route_key: StringProperty(options={"HIDDEN", "SKIP_SAVE"})

    @classmethod
    def poll(cls, context):
        return _single_edit_mesh_poll(context)

    def invoke(self, context, event):
        del event
        from . import keymaps

        if not keymaps.extrude_menu_route_is_current(self.route_key):
            return {"PASS_THROUGH"}
        if getattr(bpy.types, EXTRUDE_MENU, None) is None:
            return {"PASS_THROUGH"}
        try:
            result = cast(set[str], bpy.ops.wm.call_menu("INVOKE_DEFAULT", name=EXTRUDE_MENU))
        except Exception:
            # call_menu itself (missing type, etc.). Menu.draw exceptions are
            # swallowed by Blender RNA and still return INTERFACE.
            traceback.print_exc()
            return {"PASS_THROUGH"}
        if "CANCELLED" in result and "FINISHED" not in result:
            return {"PASS_THROUGH"}
        return result


class MESH_OT_ydd_symmetric_edit_extrude_faces(bpy.types.Operator):
    """Prepare EXTRUDE_NORMAL, then invoke the native Faces dispatcher."""

    bl_idname = WRAPPER_FACES
    bl_label = "Extrude Faces"
    bl_options = {"INTERNAL"}

    @classmethod
    def poll(cls, context):
        return context.mode == "EDIT_MESH" and context.edit_object is not None

    def invoke(self, context, event):
        del event
        return _invoke_extrude_wrapper(
            context,
            self.report,
            tool_kind="EXTRUDE_NORMAL",
            native_idname=NATIVE_FACES,
        )


class MESH_OT_ydd_symmetric_edit_extrude_along_normals(bpy.types.Operator):
    """Prepare EXTRUDE_SHRINK_FATTEN, then invoke the native Along Normals dispatcher."""

    bl_idname = WRAPPER_ALONG
    bl_label = "Extrude Faces Along Normals"
    bl_options = {"INTERNAL"}

    @classmethod
    def poll(cls, context):
        return context.mode == "EDIT_MESH" and context.edit_object is not None

    def invoke(self, context, event):
        del event
        return _invoke_extrude_wrapper(
            context,
            self.report,
            tool_kind="EXTRUDE_SHRINK_FATTEN",
            native_idname=NATIVE_ALONG,
        )


class MESH_OT_ydd_symmetric_edit_extrude_individual_faces(bpy.types.Operator):
    """Prepare EXTRUDE_FACES_INDIV, then invoke the native Individual Faces item."""

    bl_idname = WRAPPER_INDIV
    bl_label = "Extrude Individual Faces"
    bl_options = {"INTERNAL"}

    @classmethod
    def poll(cls, context):
        return context.mode == "EDIT_MESH" and context.edit_object is not None

    def invoke(self, context, event):
        del event
        return _invoke_extrude_wrapper(
            context,
            self.report,
            tool_kind="EXTRUDE_FACES_INDIV",
            native_idname=NATIVE_INDIV,
        )


class MESH_OT_ydd_symmetric_edit_extrude_edges(bpy.types.Operator):
    """Prepare EXTRUDE_EDGES_INDIV, then invoke the native Extrude Edges item."""

    bl_idname = WRAPPER_EDGES
    bl_label = "Extrude Edges"
    bl_options = {"INTERNAL"}

    @classmethod
    def poll(cls, context):
        return context.mode == "EDIT_MESH" and context.edit_object is not None

    def invoke(self, context, event):
        del event
        return _invoke_extrude_wrapper(
            context,
            self.report,
            tool_kind="EXTRUDE_EDGES_INDIV",
            native_idname=NATIVE_EDGES,
        )


class MESH_OT_ydd_symmetric_edit_extrude_vertices(bpy.types.Operator):
    """Prepare EXTRUDE_VERTS_INDIV, then invoke the native Extrude Vertices item."""

    bl_idname = WRAPPER_VERTS
    bl_label = "Extrude Vertices"
    bl_options = {"INTERNAL"}

    @classmethod
    def poll(cls, context):
        return context.mode == "EDIT_MESH" and context.edit_object is not None

    def invoke(self, context, event):
        del event
        return _invoke_extrude_wrapper(
            context,
            self.report,
            tool_kind="EXTRUDE_VERTS_INDIV",
            native_idname=NATIVE_VERTS,
        )


class YSE_MT_extrude(bpy.types.Menu):
    """Source-faithful clone of VIEW3D_MT_edit_mesh_extrude (Blender 4.2 / 5.2)."""

    bl_idname = EXTRUDE_MENU
    bl_label = "Extrude"

    def draw(self, context):
        from math import pi

        layout = self.layout
        if layout is None:
            return
        layout.operator_context = "INVOKE_REGION_WIN"
        tool_settings = context.tool_settings
        select_mode = tool_settings.mesh_select_mode
        ob = context.object
        mesh = ob.data
        if mesh.total_face_sel:
            layout.operator(WRAPPER_FACES, text="Extrude Faces")
            layout.operator(WRAPPER_ALONG, text="Extrude Faces Along Normals")
            layout.operator(WRAPPER_INDIV, text="Extrude Individual Faces")
            layout.operator("view3d.edit_mesh_extrude_manifold_normal", text="Extrude Manifold")
        if mesh.total_edge_sel and (select_mode[0] or select_mode[1]):
            layout.operator(WRAPPER_EDGES, text="Extrude Edges")
        if mesh.total_vert_sel and select_mode[0]:
            layout.operator(WRAPPER_VERTS, text="Extrude Vertices")
        layout.separator()
        layout.operator("mesh.extrude_repeat")
        layout.operator("mesh.spin").angle = pi * 2
        layout.template_node_operator_asset_menu_items(catalog_path="Mesh/Extrude")


CLASSES = (
    MESH_OT_ydd_symmetric_edit_extrude_menu,
    MESH_OT_ydd_symmetric_edit_extrude_faces,
    MESH_OT_ydd_symmetric_edit_extrude_along_normals,
    MESH_OT_ydd_symmetric_edit_extrude_individual_faces,
    MESH_OT_ydd_symmetric_edit_extrude_edges,
    MESH_OT_ydd_symmetric_edit_extrude_vertices,
    YSE_MT_extrude,
)
