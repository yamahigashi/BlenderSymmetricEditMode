# SPDX-License-Identifier: GPL-3.0-or-later

"""GUI regression for adopting native extrudes started by toolbar gizmos.

Run with Blender's real window/event loop as documented in ``docs/testing.md``.
The orchestrator runs this file on both supported Blender versions.
"""

from __future__ import annotations

import math
import os
import sys
import time
import traceback
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import bmesh
import bpy
from bpy_extras import view3d_utils
from mathutils import Quaternion, Vector

bpy.context.preferences.view.show_splash = False
bpy.context.preferences.view.smooth_view = 0
bpy.context.preferences.view.use_save_prompt = False
bpy.context.preferences.filepaths.use_auto_save_temporary_files = False

PACKAGE_PARENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_PARENT))

import ydd_symmetric_edit as addon  # noqa: E402
from ydd_symmetric_edit import gizmo_adopt, keymaps, layer_names, operators  # noqa: E402

MARKER_OK = "YSE_EXTRUDE_GIZMO_TEST_OK"
MARKER_FAILED = "YSE_EXTRUDE_GIZMO_TEST_FAILED"
NX, NY = 6, 4
STATE: dict[str, object] = {}

TOOLS = {
    "EXTRUDE_CONTEXT": ("builtin.extrude_region", "3D View Tool: Edit Mesh, Extrude Region"),
    "EXTRUDE_SHRINK_FATTEN": (
        "builtin.extrude_along_normals",
        "3D View Tool: Edit Mesh, Extrude Along Normals",
    ),
    "EXTRUDE_FACES_INDIV": (
        "builtin.extrude_individual",
        "3D View Tool: Edit Mesh, Extrude Individual",
    ),
    "EXTRUDE_MANIFOLD": ("builtin.extrude_manifold", "3D View Tool: Edit Mesh, Extrude Manifold"),
}


@dataclass(frozen=True)
class Case:
    name: str
    kind: str
    selection: str
    symmetric: bool | None
    disposition: str | None = "APPLY"
    intercept: bool = False
    second_axis: bool = False
    proportional: bool = False
    hidden_partner: bool = False
    asymmetric: str | None = None
    drag: tuple[int, int] = (0, 80)
    event_interval: float = 0.09
    undo_redo: bool = False
    angled_view: bool = False


CASES = (
    Case("region_single_face", "EXTRUDE_CONTEXT", "face", True, undo_redo=True),
    Case("along_single_face", "EXTRUDE_SHRINK_FATTEN", "face", True),
    Case("individual_adjacent_faces", "EXTRUDE_FACES_INDIV", "indiv_faces", True),
    Case("region_two_faces", "EXTRUDE_CONTEXT", "two_faces", True),
    Case("region_internal_vertex", "EXTRUDE_CONTEXT", "four_faces", False, "DECLINE"),
    Case("edge_open", "EXTRUDE_CONTEXT", "edge_open", True, angled_view=True),
    Case("edge_closed", "EXTRUDE_CONTEXT", "edge_closed", True, angled_view=True),
    Case("vertex", "EXTRUDE_CONTEXT", "vertex", True, angled_view=True),
    Case("manifold_decline", "EXTRUDE_MANIFOLD", "face", False, "DECLINE"),
    Case("proportional_decline", "EXTRUDE_CONTEXT", "face", False, "DECLINE", proportional=True),
    # Midplane contract v3.1: on-plane origins are adopted; the in-plane drag
    # exercises the copy-reuse path (mirror shares the seam copies).
    Case("plane_origin_adopted", "EXTRUDE_CONTEXT", "plane_face", True, "APPLY"),
    Case("hidden_partner_abort", "EXTRUDE_CONTEXT", "face", False, "DECLINE", hidden_partner=True),
    Case("asymmetric_far_adopted", "EXTRUDE_CONTEXT", "face", False, "APPLY", asymmetric="far"),
    Case("asymmetric_partner_decline", "EXTRUDE_CONTEXT", "face", False, "DECLINE", asymmetric="partner"),
    Case("two_axis_not_adopted", "EXTRUDE_CONTEXT", "face", False, None, second_axis=True),
    Case("click_only", "EXTRUDE_CONTEXT", "face", None, None, drag=(0, 0)),
    Case(
        "fast_flick_no_fallback",
        "EXTRUDE_CONTEXT",
        "face",
        False,
        None,
        event_interval=0.005,
    ),
    Case("kmi_exclusion", "EXTRUDE_CONTEXT", "face", True, intercept=True),
    Case("gizmo_then_kmi_grace", "EXTRUDE_CONTEXT", "face", True),
    Case("double_extrude_undo", "EXTRUDE_CONTEXT", "face", None, None),
    # Midplane contract v3.1: a YZ fin on x=0 is adopted through the
    # self-deleted region path (F_r includes the consumed face; delete
    # targets do not). Drag along +X, the face normal.
    Case(
        "fin_face",
        "EXTRUDE_CONTEXT",
        "fin_face",
        True,
        "APPLY",
        drag=(80, 0),
        angled_view=True,
        undo_redo=True,
    ),
    # The wire fixture stays last: rebuilding a faced grid after the faceless
    # native extrude crashes Blender in the next case's undo sync.
    Case("faceless_bypass_not_adopted", "EXTRUDE_CONTEXT", "wire_edge", False, None, intercept=True),
)

_only = os.environ.get("YSE_GIZMO_ONLY")
if _only:
    CASES = tuple(case for case in CASES if case.name in _only.replace("+", ",").split(","))


def fail(message=""):
    if message:
        print(f"YSE_EXTRUDE_GIZMO_ERROR={message}", flush=True)
    try:
        from ydd_symmetric_edit import session_state

        records = [
            (
                r.session.route,
                r.session.tool_kind,
                getattr(r, "status", None),
                r.session.prepare_disposition,
                r.session.prepare_disposition_reason,
            )
            for r in session_state._HISTORY_RECORDS.values()
        ]
        print(
            f"YSE_EXTRUDE_GIZMO_STATE records={records} tombstones={session_state._GIZMO_TOMBSTONES}"
            f" tickets={list(session_state._GIZMO_TICKETS)}",
            flush=True,
        )
    except Exception:
        traceback.print_exc()
    traceback.print_exc()
    print(MARKER_FAILED, flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(1)


def viewport():
    window = bpy.context.window_manager.windows[0]
    area = next(area for area in window.screen.areas if area.type == "VIEW_3D")
    region = next(region for region in area.regions if region.type == "WINDOW")
    return window, area, region


def test_object():
    return bpy.data.objects[STATE["object_name"]]


def override():
    return bpy.context.temp_override(window=STATE["window"], area=STATE["area"], region=STATE["region"])


def configure_view(angled=False):
    # The top view maps screen drags onto the grid plane; +Z extrudes (interior
    # edges, vertices) need the tilted view so the offset has a screen component.
    area = STATE["area"]
    space = area.spaces.active
    space.show_gizmo = True
    space.show_gizmo_tool = True
    region_3d = space.region_3d
    region_3d.view_perspective = "ORTHO"
    if angled:
        region_3d.view_rotation = Quaternion((1.0, 0.0, 0.0), math.radians(60.0)).normalized()
    else:
        region_3d.view_rotation = Quaternion((1.0, 0.0, 0.0, 0.0))
    region_3d.view_location = (0.0, 0.0, 0.0)
    region_3d.view_distance = 8.0
    region_3d.update()


def coordinate_key(co):
    return tuple(round(float(value), 5) for value in co)


def vertex_multiset(bm):
    return Counter(coordinate_key(vertex.co) for vertex in bm.verts)


def is_x_symmetric(bm):
    live = vertex_multiset(bm)
    mirrored = Counter((-x, y, z) for x, y, z in live.elements())
    return live == mirrored


def assert_layers_removed(bm):
    for name in layer_names.TEMP_LAYER_NAMES:
        for sequence in (bm.verts, bm.edges, bm.faces):
            assert sequence.layers.int.get(name) is None, f"temporary layer leaked: {name}"


def grid_xy(i, j):
    return i - NX / 2, j - NY / 2


def grid_vert(bm, i, j):
    x, y = grid_xy(i, j)
    matches = [
        vertex
        for vertex in bm.verts
        if abs(float(vertex.co.x) - x) < 1.0e-5
        and abs(float(vertex.co.y) - y) < 1.0e-5
        and abs(float(vertex.co.z)) < 1.0e-5
    ]
    assert len(matches) == 1, (i, j, matches)
    return matches[0]


def grid_face(bm, i, j):
    wanted = {grid_vert(bm, i, j), grid_vert(bm, i + 1, j), grid_vert(bm, i + 1, j + 1), grid_vert(bm, i, j + 1)}
    matches = [face for face in bm.faces if set(face.verts) == wanted]
    assert len(matches) == 1, (i, j, matches)
    return matches[0]


def grid_edge(bm, first, second):
    a = grid_vert(bm, *first)
    b = grid_vert(bm, *second)
    matches = [edge for edge in a.link_edges if edge.other_vert(a) is b]
    assert len(matches) == 1, (first, second, matches)
    return matches[0]


def clear_selection(bm):
    for sequence in (bm.faces, bm.edges, bm.verts):
        for element in sequence:
            element.select = False
    bm.select_history.clear()


def _native_extrude_z(delta_z):
    with override():
        result = bpy.ops.mesh.extrude_region_move(
            "EXEC_DEFAULT",
            TRANSFORM_OT_translate={
                "value": (0.0, 0.0, float(delta_z)),
                "orient_type": "GLOBAL",
                "orient_matrix": (
                    (1.0, 0.0, 0.0),
                    (0.0, 1.0, 0.0),
                    (0.0, 0.0, 1.0),
                ),
                "orient_matrix_type": "GLOBAL",
                "constraint_axis": (False, False, True),
            },
        )
    if result != {"FINISHED"}:
        fail(f"native fin extrude failed: {result}")


def add_native_fin_face(obj):
    """One YZ quad on x=0, raised above a short native stem.

    Isolated bmesh faces plus a later native extrude crash Blender in
    blender::draw::extract_tris (EXCEPTION_ACCESS_VIOLATION). A disconnected
    primitive plane keeps the source face (addon declines). A face glued to
    the x=0 floor seam is 3-manifold (census undefined). Stem + fin keeps
    the selected face 2-manifold so the self-consumed path can apply.
    """

    bm = bmesh.from_edit_mesh(obj.data)
    bpy.context.tool_settings.mesh_select_mode = (False, True, False)
    clear_selection(bm)
    grid_edge(bm, (3, 1), (3, 2)).select = True
    bm.select_flush_mode()
    bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
    _native_extrude_z(0.5)
    bm = bmesh.from_edit_mesh(obj.data)
    clear_selection(bm)
    matches = []
    for edge in bm.edges:
        first, second = edge.verts
        if abs(first.co.x) > 1e-4 or abs(second.co.x) > 1e-4:
            continue
        if abs(first.co.z - 0.5) > 1e-4 or abs(second.co.z - 0.5) > 1e-4:
            continue
        if abs(first.co.y - second.co.y) < 1e-4:
            continue
        ys = (float(first.co.y), float(second.co.y))
        if min(ys) >= -1.0 - 1e-4 and max(ys) <= 1e-4:
            matches.append(edge)
    assert len(matches) == 1, matches
    matches[0].select = True
    bm.select_flush_mode()
    bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
    _native_extrude_z(1.0)
    bm = bmesh.from_edit_mesh(obj.data)
    for sequence in (bm.verts, bm.edges, bm.faces):
        sequence.ensure_lookup_table()
        sequence.index_update()
    bm.normal_update()
    return bm


def prepare_selection(bm, name):
    clear_selection(bm)
    if name in {"face", "plane_face", "two_faces", "four_faces", "indiv_faces", "fin_face"}:
        bpy.context.tool_settings.mesh_select_mode = (False, False, True)
        if name == "fin_face":
            matches = [
                face
                for face in bm.faces
                if all(abs(float(vertex.co.x)) < 1.0e-5 for vertex in face.verts)
                and max(float(vertex.co.z) for vertex in face.verts) > 0.5
            ]
            assert len(matches) == 1, matches
            matches[0].select = True
        else:
            cells = {
                "face": ((4, 1),),
                "plane_face": ((3, 1),),
                "two_faces": ((4, 1), (5, 1)),
                "four_faces": ((4, 1), (5, 1), (4, 2), (5, 2)),
                "indiv_faces": ((4, 1), (4, 2)),
            }[name]
            for cell in cells:
                grid_face(bm, *cell).select = True
        bm.select_flush_mode()
    elif name in {"edge_open", "edge_closed"}:
        bpy.context.tool_settings.mesh_select_mode = (False, True, False)
        pairs = (
            (((6, 1), (6, 2)),)
            if name == "edge_open"
            else (
                ((4, 1), (5, 1)),
                ((5, 1), (5, 2)),
                ((5, 2), (4, 2)),
                ((4, 2), (4, 1)),
            )
        )
        for pair in pairs:
            edge = grid_edge(bm, *pair)
            edge.select = True
            for vertex in edge.verts:
                vertex.select = True
        # Do not select_flush_mode(): a fully selected face boundary would
        # select that face and native region-extrude would delete it.
    elif name == "wire_edge":
        bpy.context.tool_settings.mesh_select_mode = (False, True, False)
        edge = grid_edge(bm, (6, 1), (6, 2))
        bmesh.ops.delete(bm, geom=[face for face in bm.faces], context="FACES_ONLY")
        edge.select = True
        for vertex in edge.verts:
            vertex.select = True
    else:
        bpy.context.tool_settings.mesh_select_mode = (True, False, False)
        grid_vert(bm, 4, 2).select = True
        bm.select_flush_mode()


def grid_coordinates_and_faces():
    coordinates = [(i - NX / 2, j - NY / 2, 0.0) for j in range(NY + 1) for i in range(NX + 1)]
    stride = NX + 1
    faces = [
        (j * stride + i, j * stride + i + 1, (j + 1) * stride + i + 1, (j + 1) * stride + i)
        for j in range(NY)
        for i in range(NX)
    ]
    return coordinates, faces


def live_census(obj):
    bm = bmesh.from_edit_mesh(obj.data)
    return (len(bm.verts), len(bm.edges), len(bm.faces))


def revert_previous_native_step():
    """Drop the previous case's native extrude from the undo stack.

    Gizmo modals reload Blender's last undo copy when they start. A leftover
    native step (especially a sessionless decline) is that copy, so the next
    case must undo it before rebuilding. The following operator then discards
    the redo buffer that still holds the extruded mesh.
    """
    baseline = STATE.get("baseline")
    if baseline is None:
        return
    obj = bpy.data.objects.get("YSE_GizmoObject")
    if obj is None:
        return
    for _ in range(4):
        if obj.mode != "EDIT":
            break
        if live_census(obj) == baseline:
            return
        with override():
            result = bpy.ops.ed.undo()
            if result != {"FINISHED"}:
                fail(f"revert undo failed: {result}")
        obj = bpy.data.objects.get("YSE_GizmoObject")
        if obj is None:
            fail("object lost after revert undo")
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        if obj.mode != "EDIT":
            with override():
                result = bpy.ops.object.mode_set(mode="EDIT")
                if result != {"FINISHED"}:
                    fail(f"re-enter edit after undo failed: {result}")
    if obj.mode != "EDIT" or live_census(obj) != baseline:
        fail(f"could not revert to baseline {baseline}")


def build_mesh(case: Case):
    revert_previous_native_step()
    with override():
        obj = bpy.data.objects.get("YSE_GizmoObject")
        if obj is None:
            mesh = bpy.data.meshes.new("YSE_GizmoObject")
            coordinates, faces = grid_coordinates_and_faces()
            mesh.from_pydata(coordinates, [], faces)
            mesh.update()
            obj = bpy.data.objects.new("YSE_GizmoObject", mesh)
            bpy.context.scene.collection.objects.link(obj)
            bpy.context.view_layer.objects.active = obj
            obj.select_set(True)
            for other in tuple(bpy.data.objects):
                if other is not obj:
                    bpy.data.objects.remove(other, do_unlink=True)
            bpy.ops.object.mode_set(mode="EDIT")
        else:
            bpy.context.view_layer.objects.active = obj
            obj.select_set(True)
            if bpy.context.mode != "EDIT_MESH":
                bpy.ops.object.mode_set(mode="EDIT")
        obj.use_mesh_mirror_x = True
        obj.use_mesh_mirror_y = case.second_axis
        obj.use_mesh_mirror_z = False
    bm = bmesh.from_edit_mesh(obj.data)
    bm.select_history.clear()
    if bm.verts:
        bmesh.ops.delete(bm, geom=list(bm.verts), context="VERTS")
    coordinates, faces = grid_coordinates_and_faces()
    grid = [bm.verts.new(co) for co in coordinates]
    for indices in faces:
        bm.faces.new(tuple(grid[index] for index in indices))
    for sequence in (bm.verts, bm.edges, bm.faces):
        sequence.ensure_lookup_table()
        sequence.index_update()
    if case.selection == "fin_face":
        bm = add_native_fin_face(obj)
    prepare_selection(bm, case.selection)
    if case.hidden_partner:
        for vertex in bm.verts:
            if float(vertex.co.x) < -0.5:
                vertex.hide = True
    if case.asymmetric == "far":
        grid_vert(bm, 1, 0).co.y += 0.125
    elif case.asymmetric == "partner":
        grid_vert(bm, 2, 1).co.y += 0.125
    bm.normal_update()
    bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
    with override():
        result = bpy.ops.ed.undo_push(message=f"YSE gizmo baseline {case.name}")
        if result != {"FINISHED"}:
            fail(f"undo_push failed for {case.name}: {result}")
    STATE["object_name"] = obj.name
    STATE["baseline"] = (len(bm.verts), len(bm.edges), len(bm.faces))
    return obj


def route_items(kind):
    keymap_name = TOOLS[kind][1]
    return [
        item
        for keymap, item in keymaps._REGISTERED_ITEMS
        if keymap.name == keymap_name and item.idname == keymaps.INTERCEPT_OPERATOR
    ]


def set_intercept(kind, active):
    for item in route_items(kind):
        item.active = active


def activate_tool(case: Case):
    with override():
        bpy.context.tool_settings.workspace_tool_type = "DEFAULT"
        result = bpy.ops.wm.tool_set_by_id(name=TOOLS[case.kind][0])
        assert result == {"FINISHED"}, result
        bpy.context.tool_settings.use_proportional_edit = case.proportional
    set_intercept(case.kind, case.intercept)


def window_coordinate(coordinate):
    region = STATE["region"]
    area = STATE["area"]
    local = view3d_utils.location_3d_to_region_2d(region, area.spaces.active.region_3d, Vector(coordinate))
    assert local is not None, coordinate
    return int(round(region.x + local.x)), int(round(region.y + local.y))


def drag_anchor(bm, case: Case):
    # A closed edge loop's vertex centroid sits on the enclosed face, so the
    # region gizmo would face-extrude and delete that face.  Drag from an
    # edge midpoint so native stays in edge extrude.
    if case.selection in {"edge_open", "edge_closed", "wire_edge"}:
        selected_edges = [edge for edge in bm.edges if edge.select]
        assert selected_edges, case.selection
        edge = selected_edges[0]
        return (edge.verts[0].co + edge.verts[1].co) * 0.5
    selected = [vertex for vertex in bm.verts if vertex.select]
    return sum((vertex.co for vertex in selected), Vector()) / len(selected)


def drag_events(case: Case):
    bm = bmesh.from_edit_mesh(test_object().data)
    x, y = window_coordinate(drag_anchor(bm, case))
    if case.intercept:
        # The gizmo plus sits on the selection.  Offset so LEFTMOUSE hits the
        # tool CLICK_DRAG KMI instead of the gizmo handle.
        x += 48
        y += 48
    dx, dy = case.drag
    events = [
        {"type": "MOUSEMOVE", "value": "NOTHING", "x": x, "y": y},
        {"type": "LEFTMOUSE", "value": "PRESS", "x": x, "y": y},
    ]
    if case.drag != (0, 0):
        events.extend(
            (
                {"type": "MOUSEMOVE", "value": "NOTHING", "x": x + dx // 2, "y": y + dy // 2},
                {"type": "MOUSEMOVE", "value": "NOTHING", "x": x + dx, "y": y + dy},
            )
        )
    events.append({"type": "LEFTMOUSE", "value": "RELEASE", "x": x + dx, "y": y + dy})
    return events


def send_events(events, done, interval, index=0):
    def step():
        try:
            if index < len(events):
                STATE["window"].event_simulate(**events[index])
                send_events(events, done, interval, index + 1)
            else:
                bpy.app.timers.register(done, first_interval=0.2)
        except BaseException:
            fail()
        return None

    bpy.app.timers.register(step, first_interval=interval)


def send_undetected_flick(case, done):
    """Finish the native modal before the 50ms poll can observe onset.

    The contract forbids post-modal fallback adoption.  The 50ms poll would
    otherwise randomly catch a 15ms flick, which is detection, not fallback.
    """
    if bpy.app.timers.is_registered(keymaps._poll_gizmo_global):
        bpy.app.timers.unregister(keymaps._poll_gizmo_global)
    for event in drag_events(case):
        STATE["window"].event_simulate(**event)
    started = time.monotonic()

    def wait_dead():
        try:
            if bpy.app.timers.is_registered(keymaps._poll_gizmo_global):
                bpy.app.timers.unregister(keymaps._poll_gizmo_global)
            if STATE["window"].modal_operators or operators._SESSIONS:
                if time.monotonic() - started > 15.0:
                    raise RuntimeError(f"fast flick did not settle: {case.name}")
                return 0.02
            keymaps._sync_gizmo_poll()
            bpy.app.timers.register(lambda: wait_settled(case, done), first_interval=0.15)
        except BaseException:
            fail()
        return None

    bpy.app.timers.register(wait_dead, first_interval=0.05)


def wait_quiet(done, started=None):
    started = time.monotonic() if started is None else started

    def poll():
        try:
            if STATE["window"].modal_operators or operators._SESSIONS:
                if time.monotonic() - started > 15.0:
                    raise RuntimeError("double extrude leg did not settle")
                return 0.05
            bpy.app.timers.register(done, first_interval=0.25)
        except BaseException:
            fail()
        return None

    bpy.app.timers.register(poll, first_interval=0.05)


def send_double_extrude_undo(case: Case, done):
    # Two adopted extrudes stack; a single undo must drop only the second one
    # on BOTH sides (the repair remirrors the first from its baked token).
    def second_leg():
        # Reselect a flat face away from the first cap so the second press has
        # a deterministic anchor on both Blender versions.  Even if the modal
        # start reverts the unpushed selection, the reverted selection is the
        # first cap, which extrudes too, so the undo assertions still hold.
        obj = test_object()
        bm = bmesh.from_edit_mesh(obj.data)
        clear_selection(bm)
        target = grid_face(bm, 5, 0)
        target.select = True
        bm.select_flush_mode()
        bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)

        def press():
            # A programmatic INVOKE is gizmo-equivalent under the armed poll
            # and sidesteps version-specific event routing after the first
            # adopted extrude.
            bm2 = bmesh.from_edit_mesh(test_object().data)
            selected = [vertex for vertex in bm2.verts if vertex.select]
            center = sum((vertex.co for vertex in selected), Vector()) / len(selected)
            x, y = window_coordinate(center)
            with override():
                result = bpy.ops.mesh.extrude_context_move("INVOKE_DEFAULT")
            assert "RUNNING_MODAL" in result or "FINISHED" in result, result
            dx, dy = case.drag
            events = [
                {"type": "MOUSEMOVE", "value": "NOTHING", "x": x + dx // 2, "y": y + dy // 2},
                {"type": "MOUSEMOVE", "value": "NOTHING", "x": x + dx, "y": y + dy},
                {"type": "LEFTMOUSE", "value": "PRESS", "x": x + dx, "y": y + dy},
                {"type": "LEFTMOUSE", "value": "RELEASE", "x": x + dx, "y": y + dy},
            ]
            send_events(events, lambda: wait_quiet(run_undo), case.event_interval)
            return None

        bpy.app.timers.register(press, first_interval=0.4)
        return None

    def run_undo():
        try:
            with override():
                assert bpy.ops.ed.undo() == {"FINISHED"}
            bpy.app.timers.register(verify, first_interval=0.6)
        except BaseException:
            fail()
        return None

    def verify():
        try:
            bm = bmesh.from_edit_mesh(test_object().data)
            base = STATE["baseline"]
            assert is_x_symmetric(bm), "undo after two gizmo extrudes lost the first mirror"
            assert (len(bm.verts), len(bm.edges), len(bm.faces)) == (
                base[0] + 8,
                base[1] + 16,
                base[2] + 8,
            ), (len(bm.verts), len(bm.edges), len(bm.faces))

            assert_layers_removed(bm)
            done()
        except BaseException:
            fail()
        return None

    send_events(drag_events(case), lambda: wait_quiet(second_leg), case.event_interval)


def send_gizmo_then_kmi(case: Case, done):
    # First drag adopts through the gizmo route; the second fires the KMI
    # right after confirm so a still-live GIZMO_ADOPTED session must be
    # sync-finished by _prepare_session before the new session is prepared.
    import dataclasses

    def after_first():
        try:
            set_intercept(case.kind, True)
            second = drag_events(dataclasses.replace(case, intercept=True))
            send_events(second, lambda: wait_settled(case, done), case.event_interval)
        except BaseException:
            fail()
        return None

    send_events(drag_events(case), after_first, case.event_interval)


def history_watermark():
    records = operators._HISTORY_RECORDS.values()
    return max((record.sequence for record in records), default=-1)


def latest_record(case: Case):
    records = [
        record
        for record in operators._HISTORY_RECORDS.values()
        if record.session.object_name == STATE["object_name"]
        and record.session.tool_kind == case.kind
        and record.sequence > STATE.get("record_watermark", -1)
    ]
    return max(records, key=lambda record: record.sequence) if records else None


def wait_settled(case: Case, done, started=None):
    started = time.monotonic() if started is None else started

    def poll():
        try:
            if STATE["window"].modal_operators or operators._SESSIONS:
                if time.monotonic() - started > 15.0:
                    raise RuntimeError(f"case did not settle: {case.name}")
                return 0.05
            bpy.app.timers.register(lambda: verify_case(case, done), first_interval=0.25)
        except BaseException:
            fail()
        return None

    bpy.app.timers.register(poll, first_interval=0.05)


def verify_case(case: Case, done):
    try:
        set_intercept(case.kind, True)
        obj = test_object()
        bm = bmesh.from_edit_mesh(obj.data)
        if case.name == "gizmo_then_kmi_grace":
            records = [
                record
                for record in operators._HISTORY_RECORDS.values()
                if record.session.object_name == STATE["object_name"]
                and record.sequence > STATE.get("record_watermark", -1)
            ]
            routes = sorted(record.session.route for record in records)
            assert routes == ["GIZMO_ADOPTED", "KMI"], routes
            for record in records:
                assert record.status == "COMMITTED", record
                assert record.session.prepare_disposition == "APPLY", record.session
            assert is_x_symmetric(bm), case.name
            assert_layers_removed(bm)
            done()
            return None
        record = latest_record(case)
        if case.disposition is None:
            assert record is None, f"unexpected adopted record: {record}"
        else:
            assert record is not None, f"missing history record for {case.name}"
            assert record.status == "COMMITTED", record
            expected_route = "KMI" if case.intercept else gizmo_adopt.GIZMO_ROUTE
            assert record.session.route == expected_route, record.session
            if case.disposition == "DECLINE":
                assert record.session.prepare_disposition == "DECLINE", record.session
        if case.symmetric is not None:
            assert is_x_symmetric(bm) is case.symmetric, case.name
        assert_layers_removed(bm)
        if case.undo_redo:
            with override():
                assert bpy.ops.ed.undo() == {"FINISHED"}
            bm = bmesh.from_edit_mesh(obj.data)
            assert (len(bm.verts), len(bm.edges), len(bm.faces)) == STATE["baseline"]
            assert_layers_removed(bm)
            with override():
                assert bpy.ops.ed.redo() == {"FINISHED"}

            def after_repair():
                try:
                    repaired = bmesh.from_edit_mesh(obj.data)
                    assert is_x_symmetric(repaired), "gizmo redo was not remirrored by repair"
                    assert_layers_removed(repaired)
                    done()
                except BaseException:
                    fail()

            bpy.app.timers.register(after_repair, first_interval=0.4)
            return None
        done()
    except BaseException:
        fail()
    return None


def run_case(index):
    if index >= len(CASES):
        run_idle_checks()
        return
    case = CASES[index]
    try:
        print(f"YSE_EXTRUDE_GIZMO_CASE={case.name}", flush=True)
        STATE["record_watermark"] = history_watermark()
        configure_view(case.angled_view)
        build_mesh(case)
        activate_tool(case)

        def start_events():
            try:

                def proceed():
                    run_case(index + 1)

                if case.name == "fast_flick_no_fallback":
                    send_undetected_flick(case, proceed)
                elif case.name == "gizmo_then_kmi_grace":
                    send_gizmo_then_kmi(case, proceed)
                elif case.name == "double_extrude_undo":
                    send_double_extrude_undo(case, proceed)
                else:
                    send_events(
                        drag_events(case),
                        lambda: wait_settled(case, proceed),
                        case.event_interval,
                    )
            except BaseException:
                fail()
            return None

        bpy.app.timers.register(start_events, first_interval=1.2)
    except BaseException:
        fail()


def run_idle_checks():
    try:
        with override():
            bpy.ops.wm.tool_set_by_id(name="builtin.select_box")

        def verify_tool_idle():
            try:
                assert not bpy.app.timers.is_registered(keymaps._poll_gizmo_global)
                with override():
                    bpy.ops.object.mode_set(mode="OBJECT")
                bpy.app.timers.register(verify_object_idle, first_interval=1.2)
            except BaseException:
                fail()
            return None

        def verify_object_idle():
            try:
                assert not bpy.app.timers.is_registered(keymaps._poll_gizmo_global)
                obj = test_object()
                duplicate = bpy.data.objects.new("YSE_GizmoMultiObject", obj.data.copy())
                bpy.context.scene.collection.objects.link(duplicate)
                obj.select_set(True)
                duplicate.select_set(True)
                bpy.context.view_layer.objects.active = obj
                with override():
                    bpy.ops.object.mode_set(mode="EDIT")
                    bpy.ops.wm.tool_set_by_id(name=TOOLS["EXTRUDE_CONTEXT"][0])

                def verify_multi_object():
                    try:
                        assert len(bpy.context.objects_in_mode_unique_data) == 2
                        assert not bpy.app.timers.is_registered(keymaps._poll_gizmo_global)
                        assert not operators._SESSIONS
                        print(MARKER_OK, flush=True)
                        addon.unregister()
                        bpy.ops.wm.quit_blender()
                    except BaseException:
                        fail()
                    return None

                bpy.app.timers.register(verify_multi_object, first_interval=1.2)
            except BaseException:
                fail()
            return None

        bpy.app.timers.register(verify_tool_idle, first_interval=1.2)
    except BaseException:
        fail()


def start():
    try:
        addon.register()
        addon.sync_persistent_keymap(True)
        window, area, region = viewport()
        STATE.update(window=window, area=area, region=region)
        configure_view()
        run_case(0)
    except BaseException:
        fail()
    return None


bpy.app.timers.register(start, first_interval=0.3)
