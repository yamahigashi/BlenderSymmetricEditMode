# SPDX-License-Identifier: GPL-3.0-or-later

bl_info = {
    "name": "ydd Symmetric Edit",
    "author": "yamahigashi dot dev",
    "version": (0, 5, 0),
    "blender": (4, 2, 0),
    "location": "3D Viewport > Edit Mode > Edit tab",
    "description": "Mirror native Knife and Loop Cut topology without a modifier",
    "category": "Mesh",
    "license": "GPL-3.0-or-later",
}

import bpy  # noqa: E402
from bpy.props import PointerProperty  # noqa: E402

from . import keymaps, operators, replay, ui  # noqa: E402


def sync_persistent_keymap(enabled: bool) -> None:
    """Synchronize supported native cut routes with the saved toggle."""

    keymaps.sync(enabled)


def register():
    for cls in (*ui.CLASSES, *operators.CLASSES, *replay.CLASSES):
        bpy.utils.register_class(cls)

    setattr(
        bpy.types.Scene,
        "ydd_symmetric_edit",
        PointerProperty(type=ui.YSE_PG_settings),
    )

    if operators.cleanup_stale_attributes not in bpy.app.handlers.save_pre:
        bpy.app.handlers.save_pre.append(operators.cleanup_stale_attributes)
    if operators.cleanup_after_load not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(operators.cleanup_after_load)
    operators.register_history_handlers()

    preferences = ui.get_addon_preferences(bpy.context)
    keymaps.register(enabled=preferences.enabled if preferences is not None else False)


def unregister():
    keymaps.unregister()
    operators.unregister_history_handlers()
    operators.cleanup_all_sessions()
    if operators.cleanup_stale_attributes in bpy.app.handlers.save_pre:
        bpy.app.handlers.save_pre.remove(operators.cleanup_stale_attributes)
    if operators.cleanup_after_load in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(operators.cleanup_after_load)

    delattr(bpy.types.Scene, "ydd_symmetric_edit")
    for cls in reversed((*ui.CLASSES, *operators.CLASSES, *replay.CLASSES)):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
