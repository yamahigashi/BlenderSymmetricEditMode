# SPDX-License-Identifier: GPL-3.0-or-later

"""Two-stage test for the installed extension's persistent mode preference."""

from __future__ import annotations

import importlib
import os
import sys
import traceback

import bpy


def extension_entry():
    return next(
        addon
        for addon in bpy.context.preferences.addons
        if addon.module.endswith(".ydd_symmetric_edit") or addon.module == "ydd_symmetric_edit"
    )


def run():
    stage = os.environ["YSE_PREF_TEST_STAGE"]
    entry = extension_entry()
    addon = importlib.import_module(entry.module)
    preferences = entry.preferences

    if stage == "1":
        assert preferences.enabled is False
        assert addon.keymaps._ENABLED is False
        preferences.enabled = True
        assert addon.keymaps._ENABLED is True
        assert addon.keymaps._REGISTERED_ITEMS
        assert all(item.active for _keymap, item in addon.keymaps._REGISTERED_ITEMS)
        bpy.ops.wm.save_userpref()
        print("YSE_PREFERENCES_STAGE1_OK", flush=True)
    elif stage == "2":
        assert preferences.enabled is True
        assert addon.keymaps._ENABLED is True
        assert addon.keymaps._REGISTERED_ITEMS
        preferences.enabled = False
        assert addon.keymaps._ENABLED is False
        assert all(not item.active for _keymap, item in addon.keymaps._REGISTERED_ITEMS)
        bpy.ops.wm.save_userpref()
        print("YSE_PREFERENCES_STAGE2_OK", flush=True)
    else:
        raise AssertionError(f"Unexpected test stage: {stage}")


def guarded():
    try:
        run()
    except BaseException:
        traceback.print_exc()
        print("YSE_PREFERENCES_TEST_FAILED", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    bpy.ops.wm.quit_blender()
    return None


bpy.app.timers.register(guarded, first_interval=0.25)
