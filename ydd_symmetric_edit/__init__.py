# SPDX-License-Identifier: GPL-3.0-or-later

bl_info = {
    "name": "ydd Symmetric Edit",
    "author": "yamahigashi dot dev",
    "version": (0, 9, 0),
    "blender": (4, 2, 0),
    "location": "3D Viewport > Edit Mode > Edit tab",
    "description": "Mirror native cut, connect, merge, delete, and extrude edits without a modifier",
    "category": "Mesh",
    "license": "GPL-3.0-or-later",
}

import traceback  # noqa: E402
from datetime import date  # noqa: E402

import bpy  # noqa: E402
from bpy.props import PointerProperty  # noqa: E402

from . import delete_dissolve, extrude_menu, inset_bevel, keymaps, operators, replay, trial, ui  # noqa: E402

_TRIAL_START_MAX_ATTEMPTS = 20


def sync_persistent_keymap(enabled: bool) -> None:
    """Synchronize supported native cut routes with the saved toggle."""

    keymaps.sync(enabled)


def _make_trial_start_timer():
    # Retries a bounded number of times until addon preferences become available.
    attempts = 0

    def _stamp_trial_start():
        nonlocal attempts
        preferences = ui.get_addon_preferences(bpy.context)
        if preferences is None:
            attempts += 1
            if attempts >= _TRIAL_START_MAX_ATTEMPTS:
                return None
            return 1.0
        if not preferences.trial_started:
            preferences.trial_started = date.today().isoformat()
            # Property writes from timers do not tag preferences dirty, so the
            # stamp would be lost on exit without an explicit save.
            user_preferences = bpy.context.preferences
            if user_preferences is not None and user_preferences.use_preferences_save:
                bpy.ops.wm.save_userpref()
        return None

    return _stamp_trial_start


_trial_start_timer = None


def _remove_scene_settings():
    if hasattr(bpy.types.Scene, "ydd_symmetric_edit"):
        delattr(bpy.types.Scene, "ydd_symmetric_edit")


def _remove_save_load_handlers():
    if operators.cleanup_stale_attributes in bpy.app.handlers.save_pre:
        bpy.app.handlers.save_pre.remove(operators.cleanup_stale_attributes)
    if operators.cleanup_after_load in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(operators.cleanup_after_load)


def _remove_trial_start_timer():
    global _trial_start_timer
    if _trial_start_timer is not None:
        try:
            bpy.app.timers.unregister(_trial_start_timer)
        except ValueError:
            pass
        _trial_start_timer = None


def register():
    # A failure partway through must not leave classes, handlers, keymaps, or
    # the Scene pointer behind; unwind everything registered so far.
    cleanups = []
    try:
        for cls in (
            *ui.CLASSES,
            *operators.CLASSES,
            *replay.CLASSES,
            *delete_dissolve.CLASSES,
            *extrude_menu.CLASSES,
            *inset_bevel.CLASSES,
        ):
            bpy.utils.register_class(cls)
            cleanups.append(lambda cls=cls: bpy.utils.unregister_class(cls))

        setattr(
            bpy.types.Scene,
            "ydd_symmetric_edit",
            PointerProperty(type=ui.YSE_PG_settings),
        )
        cleanups.append(_remove_scene_settings)

        if operators.cleanup_stale_attributes not in bpy.app.handlers.save_pre:
            bpy.app.handlers.save_pre.append(operators.cleanup_stale_attributes)
        if operators.cleanup_after_load not in bpy.app.handlers.load_post:
            bpy.app.handlers.load_post.append(operators.cleanup_after_load)
        cleanups.append(_remove_save_load_handlers)

        operators.register_history_handlers()
        cleanups.append(operators.unregister_history_handlers)

        preferences = ui.get_addon_preferences(bpy.context)
        keymaps.register(enabled=preferences.enabled if preferences is not None else False)
        cleanups.append(keymaps.unregister)

        global _trial_start_timer
        if trial.TRIAL_BUILD:
            _trial_start_timer = _make_trial_start_timer()
            bpy.app.timers.register(_trial_start_timer, first_interval=0)
            cleanups.append(_remove_trial_start_timer)
    except BaseException:
        for cleanup in reversed(cleanups):
            try:
                cleanup()
            except Exception:
                traceback.print_exc()
        raise


def unregister():
    _remove_trial_start_timer()

    keymaps.unregister()
    operators.unregister_history_handlers()
    operators.cleanup_all_sessions()
    _remove_save_load_handlers()

    _remove_scene_settings()
    for cls in reversed(
        (
            *ui.CLASSES,
            *operators.CLASSES,
            *replay.CLASSES,
            *delete_dissolve.CLASSES,
            *extrude_menu.CLASSES,
            *inset_bevel.CLASSES,
        )
    ):
        # register() may already have unwound some classes after a partial
        # failure; unregistering twice raises RuntimeError.  getattr because
        # the fake-bpy stubs do not model is_registered on every class.
        if getattr(cls, "is_registered", False):
            bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
