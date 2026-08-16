"""Background operator regressions for Connect history forms.

Run with Blender's background loop::

    blender --factory-startup --background --python test_connect_history_forms.py
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

import bmesh
import bpy

PACKAGE_PARENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_PARENT))

import ydd_symmetric_edit as addon  # noqa: E402
from ydd_symmetric_edit import backup, layer_names, replay  # noqa: E402

MARKER_OK = "YSE_CONNECT_FORMS_OK"
MARKER_FAILED = "YSE_CONNECT_FORMS_FAILED"
TOLERANCE = 1.0e-4


def fail() -> None:
    traceback.print_exc()
    print(MARKER_FAILED, flush=True)
    os._exit(1)


def clear_scene() -> None:
    if bpy.context.object is not None and bpy.context.object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    for obj in tuple(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for mesh in tuple(bpy.data.meshes):
        bpy.data.meshes.remove(mesh)


def make_object(name, vertices, faces=(), edges=()):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(
        [tuple(vertex) for vertex in vertices], [tuple(edge) for edge in edges], [tuple(face) for face in faces]
    )
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    obj.use_mesh_mirror_x = True
    obj.use_mesh_mirror_y = False
    obj.use_mesh_mirror_z = False
    return obj


def make_monkey_object(name):
    mesh = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    source_bm = bmesh.new()
    try:
        bmesh.ops.create_monkey(source_bm)
        source_bm.to_mesh(mesh)
        mesh.update()
    finally:
        source_bm.free()
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    obj.use_mesh_mirror_x = True
    obj.use_mesh_mirror_y = False
    obj.use_mesh_mirror_z = False
    return obj


def enter_edit(obj):
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.context.tool_settings.mesh_select_mode = (True, False, False)
    return bmesh.from_edit_mesh(obj.data)


def leave_edit() -> None:
    if bpy.context.object is not None and bpy.context.object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")


def coordinate_key(coordinate):
    return tuple(round(float(value), 5) for value in coordinate)


def find_vertex(bm, coordinate):
    key = coordinate_key(coordinate)
    for vertex in bm.verts:
        if coordinate_key(vertex.co) == key:
            return vertex
    raise AssertionError(coordinate)


def find_edge(bm, first, second):
    expected = {coordinate_key(first), coordinate_key(second)}
    for edge in bm.edges:
        if {coordinate_key(edge.verts[0].co), coordinate_key(edge.verts[1].co)} == expected:
            return edge
    raise AssertionError((first, second))


def find_edge_tolerant(bm, first, second, tolerance=TOLERANCE):
    def close(actual, expected):
        return all(abs(float(left) - float(right)) <= tolerance for left, right in zip(actual, expected, strict=True))

    for edge in bm.edges:
        if (close(edge.verts[0].co, first) and close(edge.verts[1].co, second)) or (
            close(edge.verts[0].co, second) and close(edge.verts[1].co, first)
        ):
            return edge
    raise AssertionError((first, second))


def select_vertices(bm, coordinates, *, history=True):
    bpy.context.tool_settings.mesh_select_mode = (True, False, False)
    for face in bm.faces:
        face.select = False
    for edge in bm.edges:
        edge.select = False
    for vertex in bm.verts:
        vertex.select = False
    bm.select_history.clear()
    for coordinate in coordinates:
        vertex = find_vertex(bm, coordinate)
        vertex.select = True
        if history:
            bm.select_history.add(vertex)
    # GUI vertex clicks flush upward (edges between selected verts get selected);
    # the close-the-loop native branch depends on that state.
    bm.select_flush(True)


def select_edges(bm, edge_indices):
    bpy.context.tool_settings.mesh_select_mode = (False, True, False)
    for face in bm.faces:
        face.select = False
    for edge in bm.edges:
        edge.select = False
    for vertex in bm.verts:
        vertex.select = False
    bm.select_history.clear()
    for index in edge_indices:
        edge = bm.edges[index]
        edge.select = True
        for vertex in edge.verts:
            vertex.select = True
        bm.select_history.add(edge)


def select_edges_by_coordinates(bm, endpoints):
    bpy.context.tool_settings.mesh_select_mode = (False, True, False)
    for face in bm.faces:
        face.select = False
    for edge in bm.edges:
        edge.select = False
    for vertex in bm.verts:
        vertex.select = False
    bm.select_history.clear()
    for first, second in endpoints:
        edge = find_edge(bm, first, second)
        edge.select = True
        for vertex in edge.verts:
            vertex.select = True
        bm.select_history.add(edge)


def select_edges_by_tolerant_coordinates(bm, endpoints):
    bpy.context.tool_settings.mesh_select_mode = (False, True, False)
    for face in bm.faces:
        face.select = False
    for edge in bm.edges:
        edge.select = False
    for vertex in bm.verts:
        vertex.select = False
    bm.select_history.clear()
    for first, second in endpoints:
        edge = find_edge_tolerant(bm, first, second)
        edge.select = True
        for vertex in edge.verts:
            vertex.select = True
        bm.select_history.add(edge)


def edge_signature(edge):
    return frozenset((coordinate_key(edge.verts[0].co), coordinate_key(edge.verts[1].co)))


def edge_signatures(bm):
    return {edge_signature(edge) for edge in bm.edges}


def topology_signature(bm):
    vertex_coords = tuple(sorted(coordinate_key(vertex.co) for vertex in bm.verts))
    faces = tuple(sorted(tuple(coordinate_key(vertex.co) for vertex in face.verts) for face in bm.faces))
    return vertex_coords, edge_signatures(bm), faces


def connected_via_plane(bm, first, second):
    first_vertex = find_vertex(bm, first)
    second_vertex = find_vertex(bm, second)
    if any(edge.other_vert(first_vertex) is second_vertex for edge in first_vertex.link_edges):
        return True
    for edge in first_vertex.link_edges:
        midpoint = edge.other_vert(first_vertex)
        if abs(float(midpoint.co.x)) <= TOLERANCE and any(
            item.other_vert(midpoint) is second_vertex for item in midpoint.link_edges
        ):
            return True
    return False


def history_edge_signatures(bm):
    return tuple(edge_signature(element) for element in bm.select_history if isinstance(element, bmesh.types.BMEdge))


def mirror_signature(signature):
    return frozenset((-point[0], point[1], point[2]) for point in signature)


def run_connect(obj):
    bpy.context.view_layer.objects.active = obj
    result = bpy.ops.mesh.ydd_symmetric_edit_connect()
    assert result == {"FINISHED"}, result


def warnings():
    return [message for kind, message in replay._CONNECT_REPORTS if kind == "WARNING"]


def errors():
    return [message for kind, message in replay._CONNECT_REPORTS if kind == "ERROR"]


def assert_clean(bm):
    assert bm.edges.layers.int.get(layer_names.EDGE_ORIGINAL_LAYER) is None
    assert bm.edges.layers.int.get(layer_names.CONNECT_HISTORY_EDGE_TOKEN_LAYER) is None
    assert bm.faces.layers.int.get(layer_names.FACE_ID_LAYER) is None
    assert bm.verts.layers.int.get(layer_names.VERT_BACKUP_ID_LAYER) is None


def case_1_close_loop() -> None:
    """§5-1: on-plane pair plus one off-plane vertex; the close edge is
    off-plane so R is not self-mirrored and the mirror-side close-the-loop
    runs (this is the suzanne_falied_c shape; it requires the vertex flush)."""

    clear_scene()
    obj = make_object(
        "YSE_FormCloseLoop",
        ((0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 0), (-1, 0, 0), (-1, 1, 0)),
        faces=((0, 2, 3, 1), (0, 1, 5, 4)),
    )
    bm = enter_edit(obj)
    select_vertices(bm, ((0, 1, 0), (0, 0, 0), (1, 0, 0)))
    bmesh.update_edit_mesh(obj.data)
    original_native = replay._native_vert_connect_path
    calls = {"count": 0}

    def observed_native():
        calls["count"] += 1
        return original_native()

    replay._native_vert_connect_path = observed_native
    try:
        run_connect(obj)
    finally:
        replay._native_vert_connect_path = original_native
    assert calls["count"] == 2
    result_bm = bmesh.from_edit_mesh(obj.data)
    assert edge_signature(find_edge(result_bm, (1, 0, 0), (0, 1, 0)))
    assert edge_signature(find_edge(result_bm, (-1, 0, 0), (0, 1, 0)))
    assert not warnings(), warnings()
    assert_clean(result_bm)
    leave_edit()


def case_2_pair() -> None:
    """§5-2: history-free two-vertex pair."""

    clear_scene()
    obj = make_object("YSE_FormPair", ((-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)), faces=((0, 1, 2, 3),))
    bm = enter_edit(obj)
    before = edge_signatures(bm)
    select_vertices(bm, ((-1, -1, 0), (1, 1, 0)), history=False)
    bmesh.update_edit_mesh(obj.data)
    run_connect(obj)
    after = edge_signatures(bmesh.from_edit_mesh(obj.data))
    assert connected_via_plane(bmesh.from_edit_mesh(obj.data), (-1, -1, 0), (1, 1, 0))
    mirror_bm = bmesh.from_edit_mesh(obj.data)
    assert connected_via_plane(mirror_bm, (1, -1, 0), (-1, 1, 0))
    assert any(signature not in before for signature in after)
    assert not warnings(), warnings()
    assert_clean(bmesh.from_edit_mesh(obj.data))
    leave_edit()


def case_3_edge() -> None:
    """§5-3: adjacent edge history closes a path on both sides."""

    clear_scene()
    obj = make_object(
        "YSE_FormEdge",
        ((-2, 0, 0), (-1, -1, 0), (1, -1, 0), (2, 0, 0), (1, 1, 0), (-1, 1, 0)),
        faces=(tuple(range(6)),),
    )
    bm = enter_edit(obj)
    bm.edges.ensure_lookup_table()
    select_edges(bm, (2, 3))
    before = edge_signatures(bm)
    source_edges = tuple(edge_signature(bm.edges[index]) for index in (2, 3))
    bmesh.update_edit_mesh(obj.data)
    run_connect(obj)
    restored = bmesh.from_edit_mesh(obj.data)
    effects = edge_signatures(restored) - before
    assert effects
    assert effects & {mirror_signature(signature) for signature in effects}
    assert all(isinstance(element, bmesh.types.BMEdge) for element in restored.select_history)
    assert history_edge_signatures(restored) == source_edges
    assert {edge_signature(edge) for edge in restored.edges if edge.select} == set(source_edges)
    assert not warnings(), warnings()
    assert_clean(bmesh.from_edit_mesh(obj.data))
    leave_edit()


def case_4_edge_missing() -> None:
    """§5-4: missing mirrored history edge keeps native and reports the old warning."""

    clear_scene()
    obj = make_object("YSE_FormEdgeMissing", ((-1, -1, 0), (1, -1, 0), (1, 1, 0), (0, 1, 0)), faces=((0, 1, 2, 3),))
    bm = enter_edit(obj)
    bm.edges.ensure_lookup_table()
    select_edges(bm, (2, 3))
    before = edge_signatures(bm)
    bmesh.update_edit_mesh(obj.data)
    run_connect(obj)
    after = edge_signatures(bmesh.from_edit_mesh(obj.data))
    source_effect = after - before
    assert source_effect
    assert not source_effect & {mirror_signature(signature) for signature in source_effect}
    assert not any("rolled back" in message for message in warnings())
    assert any("no mirror counterpart" in message for message in warnings()), warnings()
    assert_clean(bmesh.from_edit_mesh(obj.data))
    leave_edit()


def case_5_other_wire_partial_history() -> None:
    """§5-5: wire/isolated partial vertex history is declined after native success."""

    clear_scene()
    obj = make_object(
        "YSE_FormOtherWire",
        ((-3, 0, 0), (-2, 0, 0), (-1, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0), (0, 2, 0)),
        edges=((0, 1), (4, 5)),
    )
    bm = enter_edit(obj)
    before = edge_signatures(bm)
    select_vertices(bm, ((1, 0, 0), (2, 0, 0), (0, 2, 0)), history=False)
    bm.select_history.add(find_vertex(bm, (1, 0, 0)))
    bm.select_history.add(find_vertex(bm, (2, 0, 0)))
    bmesh.update_edit_mesh(obj.data)
    original_native = replay._native_vert_connect_path
    calls = {"count": 0}

    def observed_native():
        calls["count"] += 1
        return original_native()

    replay._native_vert_connect_path = observed_native
    try:
        run_connect(obj)
    finally:
        replay._native_vert_connect_path = original_native
    assert calls["count"] == 1
    after = edge_signatures(bmesh.from_edit_mesh(obj.data))
    assert frozenset(((1, 0, 0), (2, 0, 0))) in after - before
    assert frozenset(((-1, 0, 0), (-2, 0, 0))) not in after
    assert warnings() == ["Mirrored connect skipped: unsupported selection history"]
    assert_clean(bmesh.from_edit_mesh(obj.data))
    leave_edit()


def case_6_first_native_error() -> None:
    """§5-6: history-free multi-vertex native error is sanitized."""

    clear_scene()
    obj = make_object("YSE_FormNativeError", ((-1, 0, 0), (0, 1, 0), (1, 0, 0)), faces=((0, 1, 2),))
    bm = enter_edit(obj)
    before = topology_signature(bm)
    select_vertices(bm, ((-1, 0, 0), (0, 1, 0), (1, 0, 0)), history=False)
    bmesh.update_edit_mesh(obj.data)
    try:
        result = bpy.ops.mesh.ydd_symmetric_edit_connect()
    except RuntimeError:
        result = {"CANCELLED"}
    assert result == {"CANCELLED"}, result
    assert errors() and all(not message.startswith("Error: ") for message in errors())
    assert before == topology_signature(bmesh.from_edit_mesh(obj.data))
    assert_clean(bmesh.from_edit_mesh(obj.data))
    leave_edit()


def case_7_edge_validation_rollback() -> None:
    """§5-7: natural Suzanne edge-path mismatch rolls back the mirror."""

    clear_scene()
    obj = make_monkey_object("YSE_FormEdgeRollback")
    bm = enter_edit(obj)
    select_edges_by_tolerant_coordinates(
        bm,
        (
            ((0.1719, -0.7812, 0.2188), (0.1953, -0.75, 0.2266)),
            ((0.2188, -0.4297, -0.2812), (0.2344, -0.4062, -0.3516)),
        ),
    )
    before = edge_signatures(bm)
    source_edges = history_edge_signatures(bm)
    bmesh.update_edit_mesh(obj.data)
    original = replay._native_vert_connect_path
    calls = {"count": 0}
    post_first = {"edges": None}

    def observed_native():
        calls["count"] += 1
        result = original()
        if calls["count"] == 1:
            post_first["edges"] = edge_signatures(bmesh.from_edit_mesh(obj.data))
        return result

    replay._native_vert_connect_path = observed_native
    try:
        run_connect(obj)
    finally:
        replay._native_vert_connect_path = original
    assert calls["count"] >= 2
    after = edge_signatures(bmesh.from_edit_mesh(obj.data))
    source_effect = after - before
    assert source_effect
    # Rollback must restore exactly the post-first-native topology.
    assert after == post_first["edges"]
    restored = bmesh.from_edit_mesh(obj.data)
    assert history_edge_signatures(restored) == source_edges
    assert {edge_signature(edge) for edge in restored.edges if edge.select} == set(source_edges)
    assert any("rolled back" in message or "did not match" in message for message in warnings()), warnings()
    assert_clean(restored)
    leave_edit()


def case_9_edge_pre_resolution_failure() -> None:
    """§5-9: source cut splitting a mirrored target declines before backup."""

    clear_scene()
    obj = make_object(
        "YSE_FormEdgePreResolve",
        (
            (-4, 1, 0),
            (-4, 3, 0),
            (-2, 0, 0),
            (-2, 2, 0),
            (2, 0, 0),
            (2, 2, 0),
            (4, 1, 0),
            (4, 3, 0),
        ),
        faces=((0, 1, 3, 2), (2, 3, 5, 4), (4, 5, 7, 6)),
    )
    bm = enter_edit(obj)
    select_edges_by_coordinates(bm, (((2, 0, 0), (2, 2, 0)), ((-4, 1, 0), (-4, 3, 0))))
    before = edge_signatures(bm)
    bmesh.update_edit_mesh(obj.data)
    original_native = replay._native_vert_connect_path
    original_backup = backup.create_topology_backup
    calls = {"native": 0, "backup": 0}

    def observed_native():
        calls["native"] += 1
        return original_native()

    def observed_backup(_bm):
        calls["backup"] += 1
        return original_backup(_bm)

    replay._native_vert_connect_path = observed_native
    backup.create_topology_backup = observed_backup
    try:
        run_connect(obj)
    finally:
        replay._native_vert_connect_path = original_native
        backup.create_topology_backup = original_backup
    after = edge_signatures(bmesh.from_edit_mesh(obj.data))
    assert calls == {"native": 1, "backup": 0}
    assert after - before
    # The source cut split the mirrored history edge, so it must be gone.
    assert frozenset((coordinate_key((-2, 0, 0)), coordinate_key((-2, 2, 0)))) not in after
    assert any("no mirror counterpart" in message for message in warnings()), warnings()
    assert not any("rolled back" in message for message in warnings())
    assert_clean(bmesh.from_edit_mesh(obj.data))
    leave_edit()


def case_10_pair_self_mirrored_effect() -> None:
    """§5-10: self-mirrored pair with forced non-self effect never calls native twice."""

    clear_scene()
    obj = make_object("YSE_FormPairSelf", ((-1, -1, 0), (0, -1, 0), (1, 1, 0), (0, 1, 0)), faces=((0, 1, 2, 3),))
    bm = enter_edit(obj)
    before = edge_signatures(bm)
    select_vertices(bm, ((0, -1, 0), (0, 1, 0)), history=False)
    bmesh.update_edit_mesh(obj.data)
    original_native = replay._native_vert_connect_path
    original_self_check = replay._connect_effect_is_self_mirrored
    calls = {"count": 0}

    def observed_native():
        calls["count"] += 1
        return original_native()

    replay._native_vert_connect_path = observed_native
    replay._connect_effect_is_self_mirrored = lambda *args: False
    try:
        run_connect(obj)
    finally:
        replay._native_vert_connect_path = original_native
        replay._connect_effect_is_self_mirrored = original_self_check
    assert calls["count"] == 1
    after = edge_signatures(bmesh.from_edit_mesh(obj.data))
    assert after - before
    assert any("its own mirror" in message for message in warnings()), warnings()
    assert_clean(bmesh.from_edit_mesh(obj.data))
    leave_edit()


def run() -> None:
    try:
        addon.register()
        if hasattr(bpy.context.scene, "ydd_symmetric_edit"):
            bpy.context.scene.ydd_symmetric_edit.tolerance = TOLERANCE
        case_1_close_loop()
        case_2_pair()
        case_3_edge()
        case_4_edge_missing()
        case_5_other_wire_partial_history()
        case_6_first_native_error()
        case_7_edge_validation_rollback()
        case_9_edge_pre_resolution_failure()
        case_10_pair_self_mirrored_effect()
        print(MARKER_OK, flush=True)
        addon.unregister()
    except BaseException:
        fail()


# --background never runs bpy.app.timers; call synchronously (docs/testing.md).
run()
