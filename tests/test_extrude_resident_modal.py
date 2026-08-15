"""GUI regression: a resident pass-through modal must not block extrude mirroring.

Flow: grid fixture -> activate Extrude Region tool -> gizmo/tool drag ->
wait well past the watcher grace -> second drag -> report mirror results and
dump window.modal_operators after each confirm.
Marker: YSE_EXTRUDE_RESIDENT_TEST_OK
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import bmesh
import bpy

bpy.context.preferences.view.show_splash = False
bpy.context.preferences.view.use_save_prompt = False

PACKAGE_PARENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_PARENT))

import ydd_symmetric_edit as addon  # noqa: E402

STATE: dict[str, object] = {}
NX, NY = 6, 4


def fail(message=""):
    print(f"YSE_EXTRUDE_RESIDENT_ERROR={message}", flush=True)
    traceback.print_exc()
    print("YSE_EXTRUDE_RESIDENT_TEST_FAILED", flush=True)
    import os

    os._exit(1)


def viewport():
    window = bpy.context.window_manager.windows[0]
    area = next(area for area in window.screen.areas if area.type == "VIEW_3D")
    region = next(region for region in area.regions if region.type == "WINDOW")
    return window, area, region


def dump_modal(tag):
    window = STATE["window"]
    names = []
    for op in window.modal_operators:
        names.append(getattr(op, "bl_idname", getattr(op, "name", repr(op))))
    print(f"YSE_EXTRUDE_RESIDENT_MODAL[{tag}]={names}", flush=True)


def build_fixture():
    bpy.ops.object.select_all(action="DESELECT")
    mesh = bpy.data.meshes.new("YSE_DiagMesh")
    obj = bpy.data.objects.new("YSE_DiagObject", mesh)
    bpy.context.collection.objects.link(obj)
    verts = []
    faces = []
    for j in range(NY + 1):
        for i in range(NX + 1):
            verts.append((i - NX / 2, j - NY / 2, 0.0))
    for j in range(NY):
        for i in range(NX):
            a = j * (NX + 1) + i
            faces.append((a, a + 1, a + NX + 2, a + NX + 1))
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    obj.use_mesh_mirror_x = True
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.context.tool_settings.mesh_select_mode = (False, False, True)
    bm = bmesh.from_edit_mesh(obj.data)
    for face in bm.faces:
        face.select = False
    for face in bm.faces:
        xs = sorted(round(v.co.x, 2) for v in face.verts)
        ys = sorted(round(v.co.y, 2) for v in face.verts)
        if xs[0] == 0.0 and xs[-1] == 1.0 and ys[0] == -1.0 and ys[-1] == 0.0:
            face.select = True
    bm.select_flush(True)
    bmesh.update_edit_mesh(obj.data)
    bpy.ops.ed.undo_push(message="YSE diag fixture")
    STATE["object"] = obj
    return obj


def census():
    obj = bpy.data.objects["YSE_DiagObject"]
    bm = bmesh.from_edit_mesh(obj.data)
    return (len(bm.verts), len(bm.edges), len(bm.faces))


def is_x_symmetric():
    from collections import Counter

    obj = bpy.data.objects["YSE_DiagObject"]
    bm = bmesh.from_edit_mesh(obj.data)
    live = Counter(tuple(round(float(c), 4) for c in v.co) for v in bm.verts)
    mirrored = Counter((-x, y, z) for x, y, z in live.elements())
    return live == mirrored


def drag(cx, cy, dx, dy, then):
    window = STATE["window"]
    steps = [
        ("MOUSEMOVE", "NOTHING", cx, cy),
        ("LEFTMOUSE", "PRESS", cx, cy),
        ("MOUSEMOVE", "NOTHING", cx + dx // 2, cy + dy // 2),
        ("MOUSEMOVE", "NOTHING", cx + dx, cy + dy),
        ("LEFTMOUSE", "RELEASE", cx + dx, cy + dy),
    ]

    def send(index=0):
        if index >= len(steps):
            bpy.app.timers.register(then, first_interval=1.5)
            return None
        kind, value, x, y = steps[index]
        window.event_simulate(type=kind, value=value, x=x, y=y)
        bpy.app.timers.register(lambda: send(index + 1), first_interval=0.09)
        return None

    send()


class YSE_OT_diag_resident_modal(bpy.types.Operator):
    """Persistent pass-through modal, mimicking screencast-keys style addons."""

    bl_idname = "wm.yse_diag_resident_modal"
    bl_label = "YSE Diag Resident Modal"

    def invoke(self, context, event):
        context.window_manager.modal_handler_add(self)
        self._timer = context.window_manager.event_timer_add(0.5, window=context.window)
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        return {"PASS_THROUGH"}


def main():
    try:
        addon.register()
        addon.sync_persistent_keymap(True)
        bpy.utils.register_class(YSE_OT_diag_resident_modal)
        window, area, region = viewport()
        STATE.update(window=window, area=area, region=region)
        space = area.spaces.active
        space.show_gizmo = True
        space.show_gizmo_tool = True
        r3d = space.region_3d
        r3d.view_perspective = "ORTHO"
        from mathutils import Quaternion

        r3d.view_rotation = Quaternion((1.0, 0.0, 0.0, 0.0))
        r3d.view_location = (0.0, 0.0, 0.0)
        r3d.view_distance = 8.0
        r3d.update()
        build_fixture()
        with bpy.context.temp_override(window=window, area=area, region=region):
            bpy.ops.wm.tool_set_by_id(name="builtin.extrude_region")
        STATE["baseline"] = census()
        print(f"YSE_EXTRUDE_RESIDENT_BASELINE={STATE['baseline']}", flush=True)
        cx = region.x + region.width // 2 + 16
        cy = region.y + region.height // 2 - 16

        def after_fourth():
            dump_modal("after_fourth_finish")
            print(f"YSE_EXTRUDE_RESIDENT_CENSUS4={census()} sym={is_x_symmetric()}", flush=True)
            if is_x_symmetric():
                fail("grab during grace must decline the mirror")
            window.event_simulate(type="ESC", value="PRESS", x=cx, y=cy)
            print("YSE_EXTRUDE_RESIDENT_TEST_OK", flush=True)
            bpy.app.timers.register(lambda: (bpy.ops.wm.quit_blender(), None)[-1], first_interval=0.5)
            return None

        def fourth():
            # True intervening: start G (grab) right after the confirm and keep
            # it modal through the finish window -> must decline, naming it.
            steps = [
                ("MOUSEMOVE", "NOTHING", cx, cy),
                ("LEFTMOUSE", "PRESS", cx, cy),
                ("MOUSEMOVE", "NOTHING", cx, cy + 35),
                ("MOUSEMOVE", "NOTHING", cx, cy + 70),
                ("LEFTMOUSE", "RELEASE", cx, cy + 70),
                ("G", "PRESS", cx, cy + 70),
                ("MOUSEMOVE", "NOTHING", cx + 25, cy + 70),
            ]

            def send(index=0):
                if index >= len(steps):
                    dump_modal("grab_running")
                    bpy.app.timers.register(after_fourth, first_interval=2.0)
                    return None
                kind, value, x, y = steps[index]
                window.event_simulate(type=kind, value=value, x=x, y=y)
                bpy.app.timers.register(lambda: send(index + 1), first_interval=0.05)
                return None

            send()
            return None

        def after_third():
            dump_modal("after_third_finish")
            print(f"YSE_EXTRUDE_RESIDENT_CENSUS3={census()} sym={is_x_symmetric()}", flush=True)
            if not is_x_symmetric():
                fail("extrude 3 was not mirrored with a resident modal running")
            print("YSE_EXTRUDE_RESIDENT_TEST_OK", flush=True)
            bpy.app.timers.register(lambda: (bpy.ops.wm.quit_blender(), None)[-1], first_interval=0.5)
            return None

        def third():
            # Variant: orbit (modal view3d.rotate) right after the confirm of
            # the third drag, before the watcher can finish.
            def orbit_after_confirm():
                window = STATE["window"]
                window.event_simulate(type="MIDDLEMOUSE", value="PRESS", x=cx, y=cy)
                window.event_simulate(type="MOUSEMOVE", value="NOTHING", x=cx + 30, y=cy + 20)
                window.event_simulate(type="MIDDLEMOUSE", value="RELEASE", x=cx + 30, y=cy + 20)
                dump_modal("during_orbit_window")
                bpy.app.timers.register(after_third, first_interval=2.0)
                return None

            steps = [
                ("MOUSEMOVE", "NOTHING", cx, cy),
                ("LEFTMOUSE", "PRESS", cx, cy),
                ("MOUSEMOVE", "NOTHING", cx, cy + 35),
                ("MOUSEMOVE", "NOTHING", cx, cy + 70),
                ("LEFTMOUSE", "RELEASE", cx, cy + 70),
            ]
            window = STATE["window"]

            def send(index=0):
                if index >= len(steps):
                    bpy.app.timers.register(orbit_after_confirm, first_interval=0.02)
                    return None
                kind, value, x, y = steps[index]
                window.event_simulate(type=kind, value=value, x=x, y=y)
                bpy.app.timers.register(lambda: send(index + 1), first_interval=0.09)
                return None

            send()
            return None

        def after_second():
            dump_modal("after_second_finish")
            print(f"YSE_EXTRUDE_RESIDENT_CENSUS2={census()} sym={is_x_symmetric()}", flush=True)
            if not is_x_symmetric():
                fail("extrude 2 was not mirrored with a resident modal running")
            third()
            return None

        def second():
            dump_modal("before_second")
            print(f"YSE_EXTRUDE_RESIDENT_CENSUS1={census()} sym={is_x_symmetric()}", flush=True)
            if not is_x_symmetric():
                fail("extrude 1 was not mirrored with a resident modal running")
            drag(cx, cy, 0, 70, after_second)
            return None

        def first():
            drag(cx, cy, 0, 70, second)
            return None

        def start_resident_then_first():
            with bpy.context.temp_override(window=window, area=area, region=region):
                bpy.ops.wm.yse_diag_resident_modal("INVOKE_DEFAULT")
            dump_modal("resident_started")
            bpy.app.timers.register(first, first_interval=0.3)
            return None

        def wait_route(started=None):
            import time as _time

            started = _time.monotonic() if started is None else started

            def poll():
                from ydd_symmetric_edit import keymaps

                hooked = any(
                    item.idname == keymaps.INTERCEPT_OPERATOR and item.type == "LEFTMOUSE"
                    for _km, item in keymaps._REGISTERED_ITEMS
                )
                if hooked:
                    print("YSE_EXTRUDE_RESIDENT_ROUTE_HOOKED", flush=True)
                    bpy.app.timers.register(start_resident_then_first, first_interval=0.1)
                    return None
                if _time.monotonic() - started > 8.0:
                    fail("intercept route never hooked")
                return 0.05

            bpy.app.timers.register(poll, first_interval=0.05)
            return None

        bpy.app.timers.register(wait_route, first_interval=0.5)
    except Exception as exc:
        fail(repr(exc))


main()
