# SPDX-License-Identifier: GPL-3.0-or-later

import sys
from typing import TYPE_CHECKING

import bpy
from bpy.props import BoolProperty, EnumProperty, FloatProperty

from . import core, keymaps


def _update_persistent_mode(self, _context):
    package = sys.modules.get(__package__)
    sync = getattr(package, "sync_persistent_keymap", None)
    if sync is not None:
        sync(bool(self.enabled))


class YSE_AddonPreferences(bpy.types.AddonPreferences):
    bl_idname = __package__

    if TYPE_CHECKING:
        enabled: bool
    else:
        enabled: BoolProperty(
            name="Enable Symmetric Edit Mode",
            description=(
                "Mirror Blender's native Knife, Loop Cut, Offset Edge Loop Cut, Rip, Vertex Connect, and Merge Vertices"
            ),
            default=False,
            update=_update_persistent_mode,
        )

    def draw(self, _context):
        layout = self.layout
        layout.prop(self, "enabled", toggle=True)
        layout.label(text="Knife, Loop Cut, Offset Edge Loop Cut, Rip (V), Connect (J), Merge (M)")
        layout.label(text="Native events and tools are not replaced.")


def get_addon_preferences(context):
    addon = context.preferences.addons.get(__package__)
    if addon is not None and isinstance(addon.preferences, YSE_AddonPreferences):
        return addon.preferences

    # Direct source registration used during development may use a different
    # package key.  The installed extension normally takes the fast path above.
    for candidate in context.preferences.addons:
        preferences = getattr(candidate, "preferences", None)
        if isinstance(preferences, YSE_AddonPreferences):
            return preferences
    return None


class YSE_PG_settings(bpy.types.PropertyGroup):
    if TYPE_CHECKING:
        source_side: str
        tolerance: float
    else:
        source_side: EnumProperty(
            name="Source Side",
            description=(
                "Side on which the native cut tools create new topology; Connect and Merge follow the selection"
            ),
            items=(
                ("AUTO", "Auto", "Use the side containing most of the new cut path"),
                ("NEGATIVE", "Negative", "Mirror cuts drawn on the negative half"),
                ("POSITIVE", "Positive", "Mirror cuts drawn on the positive half"),
            ),
            default="AUTO",
        )
        tolerance: FloatProperty(
            name="Match Tolerance",
            description="Tolerance for matching exact mirrored topology in local space",
            default=1.0e-5,
            min=1.0e-8,
            soft_min=1.0e-6,
            soft_max=1.0e-3,
            precision=6,
        )


class VIEW3D_PT_ydd_symmetric_edit(bpy.types.Panel):
    bl_label = "ydd Symmetric Edit"
    bl_idname = "VIEW3D_PT_ydd_symmetric_edit"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Edit"

    @classmethod
    def poll(cls, context):
        return context.mode == "EDIT_MESH"

    def draw(self, context):
        layout = self.layout
        if layout is None:
            return
        settings = context.scene.ydd_symmetric_edit
        preferences = get_addon_preferences(context)
        obj = context.edit_object
        axes = tuple(axis for axis, _index in core.enabled_mesh_symmetry_axes(obj))

        if preferences is not None:
            layout.prop(
                preferences,
                "enabled",
                text="Enable Symmetric Edit Mode",
                icon="MOD_MIRROR",
                toggle=True,
            )
            if preferences.enabled and not keymaps.has_delete_routes():
                layout.label(text="Delete key route not found", icon="ERROR")
        else:
            layout.label(text="Persistent setting unavailable", icon="ERROR")

        status = layout.box()
        if len(axes) == 1:
            status.label(text=f"Mesh Symmetry Axis: {axes[0]}", icon="CHECKMARK")
        elif not axes:
            status.label(text="Enable a Mesh Symmetry axis", icon="ERROR")
        else:
            status.label(
                text=f"Enable one axis only (currently {', '.join(axes)})",
                icon="ERROR",
            )
        status.label(text="Uses Blender's X / Y / Z symmetry setting")
        layout.prop(settings, "source_side")
        layout.prop(settings, "tolerance")
        box = layout.box()
        if preferences is not None and preferences.enabled:
            box.label(text="Cut, Rip (V), Connect (J), and Merge (M) are mirrored.")
        else:
            box.label(text="Symmetric editing is disabled.")
        box.label(text="Use the native tool on one side, then confirm.")
        box.label(text="GG Edge Slide uses Blender's native mirror.")


CLASSES = (
    YSE_AddonPreferences,
    YSE_PG_settings,
    VIEW3D_PT_ydd_symmetric_edit,
)
