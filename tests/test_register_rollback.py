# SPDX-License-Identifier: GPL-3.0-or-later

"""Headless checks: register() unwinds fully when a step fails partway."""

from __future__ import annotations

import sys
from pathlib import Path

import bpy

PACKAGE_PARENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_PARENT))

import ydd_symmetric_edit as addon  # noqa: E402
from ydd_symmetric_edit import (  # noqa: E402
    delete_dissolve,
    extrude_menu,
    keymaps,
    operators,
    replay,
    ui,
)


class InjectedError(RuntimeError):
    pass


def all_classes():
    return (
        *ui.CLASSES,
        *operators.CLASSES,
        *replay.CLASSES,
        *delete_dissolve.CLASSES,
        *extrude_menu.CLASSES,
    )


def assert_unregistered_state(stage):
    for cls in all_classes():
        if cls.is_registered:
            raise AssertionError(f"[{stage}] class left registered: {cls.__name__}")
    if hasattr(bpy.types.Scene, "ydd_symmetric_edit"):
        raise AssertionError(f"[{stage}] Scene pointer left behind")
    if operators.cleanup_stale_attributes in bpy.app.handlers.save_pre:
        raise AssertionError(f"[{stage}] save_pre handler left behind")
    if operators.cleanup_after_load in bpy.app.handlers.load_post:
        raise AssertionError(f"[{stage}] load_post handler left behind")
    if keymaps._REGISTERED_ITEMS:
        raise AssertionError(f"[{stage}] keymap items left behind: {len(keymaps._REGISTERED_ITEMS)}")
    if addon._trial_start_timer is not None:
        raise AssertionError(f"[{stage}] trial start timer left behind")


def check_rollback(module, attribute_name):
    original = getattr(module, attribute_name)

    def _boom(*_args, **_kwargs):
        raise InjectedError(attribute_name)

    setattr(module, attribute_name, _boom)
    try:
        try:
            addon.register()
        except InjectedError:
            pass
        else:
            raise AssertionError(f"register() did not propagate failure in {attribute_name}")
    finally:
        setattr(module, attribute_name, original)

    assert_unregistered_state(f"after {attribute_name} failure")


def check_register_unregister_cycle():
    addon.register()
    for cls in all_classes():
        if not cls.is_registered:
            raise AssertionError(f"class not registered: {cls.__name__}")
    if not hasattr(bpy.types.Scene, "ydd_symmetric_edit"):
        raise AssertionError("Scene pointer missing after register()")
    addon.unregister()
    assert_unregistered_state("after normal unregister")
    addon.unregister()
    assert_unregistered_state("after repeated unregister")


def run():
    assert_unregistered_state("initial")
    # Late-stage failure: everything before the keymaps must unwind.
    check_rollback(keymaps, "register")
    # Mid-stage failure: classes, Scene pointer, and save/load handlers unwind.
    check_rollback(operators, "register_history_handlers")
    # The rollbacks must not poison a subsequent normal registration.
    check_register_unregister_cycle()
    print("YSE_REGISTER_ROLLBACK_OK", flush=True)


if __name__ == "__main__":
    run()
