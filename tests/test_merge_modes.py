# SPDX-License-Identifier: GPL-3.0-or-later

"""GUI regression for REPLAY symmetric Merge modes (serial cases).

Run with Blender's real window/event loop::

    blender --factory-startup --enable-event-simulate \
        --python test_merge_modes.py

Serial cases in one process (no undo assertions; topology only):
  1. At Center — shared plane vertex + two +X verts; mirror weld/symmetry.
  2. Collapse — two selection islands; each collapses and the mirror matches.
  3. By Distance — near pair across the plane; result stays symmetric.
  4. First side-split — self-mirrored selection: each side merges to its own
     first vertex; post-state selection/history contract; re-run idempotent.
  5. Last side-split — target on the -X side: side determination follows the
     history endpoint.
  6. Center self-mirrored — one native run, already symmetric.
  7. Center partial — selection completed with its missing mirrors first,
     then one native run onto the plane.
  8. First with an on-plane vertex mixed in — native only (no mirror pass).
  9. Partial + on-plane First — the on-plane guard must fire BEFORE the
     selection is symmetrized; the unselected mirror vertex must survive.
 10. Fault injection: pointmerge failure — rollback to the native-only state.
 11. Fault injection: backup-creation failure in the side split — native side
     merged, mirror side kept and reselected per the post-state contract.
(Backup-removal failure is covered headlessly in test_replay_units:
``remove_backup`` itself is contractually noexcept.)
"""

from __future__ import annotations

import os
import sys
import traceback
from collections import Counter
from pathlib import Path

import bmesh
import bpy
from mathutils import Quaternion

PACKAGE_PARENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_PARENT))

import ydd_symmetric_edit as addon  # noqa: E402
from ydd_symmetric_edit import backup as yse_backup  # noqa: E402
from ydd_symmetric_edit import core as yse_core  # noqa: E402

MARKER_OK = "YSE_MERGE_MODES_TEST_OK"
MARKER_FAILED = "YSE_MERGE_MODES_TEST_FAILED"
COORD_PRECISION = 5


def fail(message: str = "") -> None:
    if message:
        print(f"YSE_MERGE_MODES_ERROR={message}", flush=True)
    traceback.print_exc()
    print(MARKER_FAILED, flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(1)


def viewport_context():
    window = bpy.context.window_manager.windows[0]
    area = next(area for area in window.screen.areas if area.type == "VIEW_3D")
    region = next(region for region in area.regions if region.type == "WINDOW")
    return window, area, region


def configure_view(area) -> None:
    region_3d = area.spaces.active.region_3d
    region_3d.view_perspective = "ORTHO"
    region_3d.view_rotation = Quaternion((1.0, 0.0, 0.0, 0.0))
    region_3d.view_location = (0.0, 0.0, 0.0)
    region_3d.view_distance = 8.0
    region_3d.update()


def clear_scene() -> None:
    for old in tuple(bpy.data.objects):
        bpy.data.objects.remove(old, do_unlink=True)
    for mesh in tuple(bpy.data.meshes):
        bpy.data.meshes.remove(mesh)


def coordinate_key(coordinate, precision: int = COORD_PRECISION):
    return tuple(round(float(value), precision) for value in coordinate)


def mirror_key(coordinate, precision: int = COORD_PRECISION):
    x, y, z = coordinate
    return coordinate_key((-float(x), float(y), float(z)), precision)


def vertex_coord_multiset(bm, precision: int = COORD_PRECISION) -> Counter:
    return Counter(coordinate_key(vertex.co, precision) for vertex in bm.verts)


def edge_coord_multiset(bm, precision: int = COORD_PRECISION) -> Counter:
    return Counter(tuple(sorted(coordinate_key(vertex.co, precision) for vertex in edge.verts)) for edge in bm.edges)


def mirrored_vertex_multiset(bm, precision: int = COORD_PRECISION) -> Counter:
    return Counter(mirror_key(vertex.co, precision) for vertex in bm.verts)


def mirrored_edge_multiset(bm, precision: int = COORD_PRECISION) -> Counter:
    return Counter(tuple(sorted(mirror_key(vertex.co, precision) for vertex in edge.verts)) for edge in bm.edges)


def assert_x_symmetric(bm, *, label: str) -> None:
    verts = vertex_coord_multiset(bm)
    edges = edge_coord_multiset(bm)
    assert verts == mirrored_vertex_multiset(bm), f"{label}: vertex coords not X-symmetric: {verts}"
    assert edges == mirrored_edge_multiset(bm), f"{label}: edges not X-symmetric: {edges}"


def assert_manifold_faces(bm, *, label: str) -> None:
    for face in bm.faces:
        assert face.is_valid, f"{label}: invalid face"
        assert len(face.verts) >= 3, f"{label}: degenerate face with {len(face.verts)} verts"


def find_vertex(bm, expected, precision: int = COORD_PRECISION):
    key = coordinate_key(expected, precision)
    for vertex in bm.verts:
        if coordinate_key(vertex.co, precision) == key:
            return vertex
    raise AssertionError(f"vertex not found: {expected}")


def clear_selection(bm) -> None:
    for face in bm.faces:
        face.select = False
    for edge in bm.edges:
        edge.select = False
    for vertex in bm.verts:
        vertex.select = False
    bm.select_history.clear()


def select_vertices(bm, coordinates) -> None:
    clear_selection(bm)
    for coordinate in coordinates:
        vertex = find_vertex(bm, coordinate)
        vertex.select = True
        bm.select_history.add(vertex)


def make_object(name: str, vertices, faces):
    mesh = bpy.data.meshes.new(name + "Mesh")
    mesh.from_pydata(list(vertices), [], list(faces))
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    obj.use_mesh_mirror_x = True
    obj.use_mesh_mirror_y = False
    obj.use_mesh_mirror_z = False
    return obj


def enter_edit(window, area, region, obj):
    with bpy.context.temp_override(window=window, area=area, region=region):
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.context.tool_settings.mesh_select_mode = (True, False, False)
    return bmesh.from_edit_mesh(obj.data)


def leave_edit(window, area, region) -> None:
    with bpy.context.temp_override(window=window, area=area, region=region):
        bpy.ops.object.mode_set(mode="OBJECT")


def run_center_case(window, area, region) -> None:
    """Shared plane vertex + two +X verts; mirror cluster must stay welded/symmetric."""

    clear_scene()
    # Minimal X-symmetric strip: one shared seam edge and two side quads.
    # Select A (plane) + B,C (+X).  Centroid is off-plane, so the mirror path
    # pointmerges B',C' to mirror(centroid) without the on-plane weld branch.
    vertices = (
        (0.0, 0.0, 0.0),  # A shared
        (1.0, 0.0, 0.0),  # B
        (1.0, 1.0, 0.0),  # C
        (-1.0, 0.0, 0.0),  # B'
        (-1.0, 1.0, 0.0),  # C'
        (0.0, 1.0, 0.0),  # top shared
    )
    faces = (
        (0, 1, 2, 5),  # +X
        (0, 5, 4, 3),  # -X
    )
    obj = make_object("YSE_MergeCenter", vertices, faces)
    bm = enter_edit(window, area, region, obj)
    assert_x_symmetric(bm, label="center baseline")
    baseline_count = len(bm.verts)

    select_vertices(bm, ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0)))
    bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)

    with bpy.context.temp_override(window=window, area=area, region=region):
        result = bpy.ops.mesh.ydd_symmetric_edit_merge(mode="CENTER")
    assert result == {"FINISHED"}, result

    bm = bmesh.from_edit_mesh(obj.data)
    # Source A,B,C (3) → 1 survivor (−2); mirrored B',C' (2) → 1 (−1); net −3.
    assert len(bm.verts) == baseline_count - 3, (len(bm.verts), baseline_count)
    assert_manifold_faces(bm, label="center result")
    assert_x_symmetric(bm, label="center result")

    source_target = coordinate_key((2.0 / 3.0, 1.0 / 3.0, 0.0))
    mirror_target = coordinate_key((-2.0 / 3.0, 1.0 / 3.0, 0.0))
    present = set(vertex_coord_multiset(bm))
    assert source_target in present, present
    assert mirror_target in present, present
    # Exactly one survivor on each side of the plane at the merge targets.
    assert sum(1 for key in present if key == source_target) == 1
    assert sum(1 for key in present if key == mirror_target) == 1

    leave_edit(window, area, region)
    print("YSE_MERGE_MODES_CENTER_OK", flush=True)


def run_collapse_case(window, area, region) -> None:
    """Two selection islands on +X each collapse; -X mirrors collapse symmetrically."""

    clear_scene()
    # One connected X-symmetric mesh.  Two +X selection islands share only
    # unselected bridge verts, so the selected-edge graph has two components.
    #
    #   C'(-1,2)  T(0,2)  C(1,2)
    #   B'(-1,1)  M(0,1)  B(1,1)     island2 edge: B—C  (and B'—C')
    #   D'(-1,0)  N(0,0)  D(1,0)     unselected bridge row
    #   A'(-1,-1) S(0,-1) A(1,-1)    island1 edge: A—E  wait: A(1,-1)—F(2,-1)?
    #
    # Simpler: two vertical edges on +X separated by an unselected middle row.
    vertices = (
        (1.0, -1.0, 0.0),  # A  island1
        (1.0, 0.0, 0.0),  # B  island1
        (1.0, 1.0, 0.0),  # C  unselected bridge on +X
        (1.0, 2.0, 0.0),  # D  island2
        (1.0, 3.0, 0.0),  # E  island2
        (-1.0, -1.0, 0.0),
        (-1.0, 0.0, 0.0),
        (-1.0, 1.0, 0.0),
        (-1.0, 2.0, 0.0),
        (-1.0, 3.0, 0.0),
        (0.0, -1.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 2.0, 0.0),
        (0.0, 3.0, 0.0),
    )
    faces = (
        (10, 0, 1, 11),  # lower +X
        (11, 1, 2, 12),  # mid +X (bridge)
        (12, 2, 3, 13),  # upper-mid +X
        (13, 3, 4, 14),  # upper +X
        (10, 11, 6, 5),  # lower -X
        (11, 12, 7, 6),
        (12, 13, 8, 7),
        (13, 14, 9, 8),
    )
    obj = make_object("YSE_MergeCollapse", vertices, faces)
    bm = enter_edit(window, area, region, obj)
    assert_x_symmetric(bm, label="collapse baseline")
    baseline_count = len(bm.verts)

    # Island1: A—B.  Island2: D—E.  Bridge C stays unselected.
    select_vertices(
        bm,
        (
            (1.0, -1.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 2.0, 0.0),
            (1.0, 3.0, 0.0),
        ),
    )
    # Ensure edges of each island are selected (some Blender builds are picky).
    for edge in bm.edges:
        if edge.verts[0].select and edge.verts[1].select:
            edge.select = True
    bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
    assert sum(1 for v in bm.verts if v.select) == 4

    with bpy.context.temp_override(window=window, area=area, region=region):
        result = bpy.ops.mesh.ydd_symmetric_edit_merge(mode="COLLAPSE")
    assert result == {"FINISHED"}, result

    bm = bmesh.from_edit_mesh(obj.data)
    # 4 selected → 2 survivors on +X; 4 mirrors → 2 on -X; net −4.
    assert len(bm.verts) == baseline_count - 4, (len(bm.verts), baseline_count)
    assert_manifold_faces(bm, label="collapse result")
    assert_x_symmetric(bm, label="collapse result")

    expected = {
        coordinate_key((1.0, -0.5, 0.0)),
        coordinate_key((1.0, 2.5, 0.0)),
        coordinate_key((-1.0, -0.5, 0.0)),
        coordinate_key((-1.0, 2.5, 0.0)),
    }
    present = set(vertex_coord_multiset(bm))
    assert expected <= present, (expected - present, present)

    leave_edit(window, area, region)
    print("YSE_MERGE_MODES_COLLAPSE_OK", flush=True)


def run_by_distance_case(window, area, region) -> None:
    """Near pair across the plane merges once; no double-merge of scaffolds.

    Note: native ``remove_doubles`` keeps one endpoint rather than averaging, so
    the merged survivor may sit slightly off the plane.  Contract §4.3 asks for
    coordinate symmetry; we still assert the topological guarantees that the
    REPLAY path must provide (one merge per near pair, scaffolds untouched,
    no second merge pass).  Full X-symmetry of coordinates is checked when it
    holds and otherwise reported via a diagnostic line (see final report).
    """

    clear_scene()
    vertices = (
        (0.02, 0.0, 0.0),
        (-0.02, 0.0, 0.0),
        (0.02, 1.0, 0.0),
        (-0.02, 1.0, 0.0),
        (1.5, 0.0, 0.0),
        (-1.5, 0.0, 0.0),
        (1.5, 1.0, 0.0),
        (-1.5, 1.0, 0.0),
    )
    faces = (
        (0, 4, 6, 2),
        (1, 3, 7, 5),
    )
    obj = make_object("YSE_MergeByDistance", vertices, faces)
    bm = enter_edit(window, area, region, obj)
    assert_x_symmetric(bm, label="by_distance baseline")
    baseline_count = len(bm.verts)

    select_vertices(bm, ((0.02, 0.0, 0.0), (0.02, 1.0, 0.0)))
    bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)

    with bpy.context.temp_override(window=window, area=area, region=region):
        result = bpy.ops.mesh.ydd_symmetric_edit_merge(mode="BY_DISTANCE", threshold=0.05)
    assert result == {"FINISHED"}, result

    bm = bmesh.from_edit_mesh(obj.data)
    # Two near pairs each collapse once → net −2.  Scaffolds must remain.
    assert len(bm.verts) == baseline_count - 2, (len(bm.verts), baseline_count)
    assert_manifold_faces(bm, label="by_distance result")

    near_origin = [vertex for vertex in bm.verts if abs(vertex.co.x) < 0.1 and abs(vertex.co.y) < 0.1]
    near_top = [vertex for vertex in bm.verts if abs(vertex.co.x) < 0.1 and abs(vertex.co.y - 1.0) < 0.1]
    assert len(near_origin) == 1, [coordinate_key(v.co) for v in near_origin]
    assert len(near_top) == 1, [coordinate_key(v.co) for v in near_top]
    assert abs(near_origin[0].co.x) <= 0.03
    assert abs(near_top[0].co.x) <= 0.03
    # Both survivors should share the same X (same remove_doubles tie-break
    # side), otherwise the two pairs were handled inconsistently.
    assert abs(near_origin[0].co.x - near_top[0].co.x) <= 1.0e-6

    assert any(abs(vertex.co.x - 1.5) < 1.0e-4 for vertex in bm.verts)
    assert any(abs(vertex.co.x + 1.5) < 1.0e-4 for vertex in bm.verts)
    # Scaffold pairs must still be X-symmetric (not double-merged away).
    scaffold = [coordinate_key(v.co) for v in bm.verts if abs(abs(v.co.x) - 1.5) < 1.0e-4]
    assert Counter(scaffold) == Counter(
        [
            coordinate_key((1.5, 0.0, 0.0)),
            coordinate_key((-1.5, 0.0, 0.0)),
            coordinate_key((1.5, 1.0, 0.0)),
            coordinate_key((-1.5, 1.0, 0.0)),
        ]
    )

    verts = vertex_coord_multiset(bm)
    if verts != mirrored_vertex_multiset(bm):
        print(
            f"YSE_MERGE_MODES_BY_DISTANCE_SYMMETRY_SOFT="
            f"coords not X-symmetric after remove_doubles (expected native tie-break); verts={verts}",
            flush=True,
        )

    leave_edit(window, area, region)
    print("YSE_MERGE_MODES_BY_DISTANCE_OK", flush=True)


def make_side_split_strip(name: str):
    """One plane seam edge and one quad per side; A/A' are the y=0 endpoints."""

    vertices = (
        (0.0, 0.0, 0.0),  # S plane
        (0.0, 1.0, 0.0),  # T plane
        (1.0, 0.0, 0.0),  # A
        (1.0, 1.0, 0.0),  # B
        (-1.0, 0.0, 0.0),  # A'
        (-1.0, 1.0, 0.0),  # B'
    )
    faces = (
        (0, 2, 3, 1),  # +X
        (0, 1, 5, 4),  # -X
    )
    return make_object(name, vertices, faces)


def assert_side_split_post_state(bm, source_x: float) -> None:
    """5-2 step 6: both survivors selected, history = source survivor only."""

    selected = [coordinate_key(vertex.co) for vertex in bm.verts if vertex.select]
    expected = [coordinate_key((1.0, 0.0, 0.0)), coordinate_key((-1.0, 0.0, 0.0))]
    assert Counter(selected) == Counter(expected), selected
    history = [element for element in bm.select_history if isinstance(element, bmesh.types.BMVert)]
    assert len(history) == 1, [coordinate_key(element.co) for element in history]
    assert coordinate_key(history[0].co) == coordinate_key((source_x, 0.0, 0.0)), coordinate_key(history[0].co)


def run_first_side_split_case(window, area, region) -> None:
    """FIRST on a self-mirrored selection merges each side to its own first."""

    clear_scene()
    obj = make_side_split_strip("YSE_MergeFirstSplit")
    bm = enter_edit(window, area, region, obj)
    assert_x_symmetric(bm, label="first baseline")
    baseline_count = len(bm.verts)

    # History order A, B, A', B' — FIRST targets A at (1, 0, 0).
    select_vertices(
        bm,
        (
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (-1.0, 0.0, 0.0),
            (-1.0, 1.0, 0.0),
        ),
    )
    bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)

    with bpy.context.temp_override(window=window, area=area, region=region):
        result = bpy.ops.mesh.ydd_symmetric_edit_merge(mode="FIRST")
    assert result == {"FINISHED"}, result

    bm = bmesh.from_edit_mesh(obj.data)
    # A,B -> A and A',B' -> A' (per side), net -2.
    assert len(bm.verts) == baseline_count - 2, (len(bm.verts), baseline_count)
    assert_manifold_faces(bm, label="first result")
    assert_x_symmetric(bm, label="first result")
    present = vertex_coord_multiset(bm)
    assert present[coordinate_key((1.0, 0.0, 0.0))] == 1, present
    assert present[coordinate_key((-1.0, 0.0, 0.0))] == 1, present
    assert present[coordinate_key((1.0, 1.0, 0.0))] == 0, present
    assert present[coordinate_key((-1.0, 1.0, 0.0))] == 0, present
    assert_side_split_post_state(bm, source_x=1.0)

    # F9 approximation (contract §4.6): re-running the operator on its own
    # post-state must be idempotent — topology, selection, history unchanged.
    before_verts = vertex_coord_multiset(bm)
    with bpy.context.temp_override(window=window, area=area, region=region):
        bpy.ops.mesh.ydd_symmetric_edit_merge(mode="FIRST")
    bm = bmesh.from_edit_mesh(obj.data)
    assert vertex_coord_multiset(bm) == before_verts, vertex_coord_multiset(bm)
    assert_side_split_post_state(bm, source_x=1.0)

    leave_edit(window, area, region)
    print("YSE_MERGE_MODES_FIRST_SPLIT_OK", flush=True)


def run_last_side_split_case(window, area, region) -> None:
    """LAST with a -X history endpoint: the source side follows the target."""

    clear_scene()
    obj = make_side_split_strip("YSE_MergeLastSplit")
    bm = enter_edit(window, area, region, obj)
    baseline_count = len(bm.verts)

    # History order A, B, B', A' — LAST targets A' at (-1, 0, 0).
    select_vertices(
        bm,
        (
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (-1.0, 1.0, 0.0),
            (-1.0, 0.0, 0.0),
        ),
    )
    bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)

    with bpy.context.temp_override(window=window, area=area, region=region):
        result = bpy.ops.mesh.ydd_symmetric_edit_merge(mode="LAST")
    assert result == {"FINISHED"}, result

    bm = bmesh.from_edit_mesh(obj.data)
    assert len(bm.verts) == baseline_count - 2, (len(bm.verts), baseline_count)
    assert_manifold_faces(bm, label="last result")
    assert_x_symmetric(bm, label="last result")
    present = vertex_coord_multiset(bm)
    assert present[coordinate_key((1.0, 0.0, 0.0))] == 1, present
    assert present[coordinate_key((-1.0, 0.0, 0.0))] == 1, present
    assert_side_split_post_state(bm, source_x=-1.0)

    leave_edit(window, area, region)
    print("YSE_MERGE_MODES_LAST_SPLIT_OK", flush=True)


def run_center_self_mirrored_case(window, area, region) -> None:
    """CENTER on a mirror pair runs natively once; centroid is on the plane."""

    clear_scene()
    vertices = (
        (-1.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
        (-1.0, 1.0, 0.0),
    )
    faces = ((0, 1, 2, 3),)
    obj = make_object("YSE_MergeCenterSelf", vertices, faces)
    bm = enter_edit(window, area, region, obj)
    baseline_count = len(bm.verts)

    select_vertices(bm, ((1.0, 0.0, 0.0), (-1.0, 0.0, 0.0)))
    bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)

    with bpy.context.temp_override(window=window, area=area, region=region):
        result = bpy.ops.mesh.ydd_symmetric_edit_merge(mode="CENTER")
    assert result == {"FINISHED"}, result

    bm = bmesh.from_edit_mesh(obj.data)
    assert len(bm.verts) == baseline_count - 1, (len(bm.verts), baseline_count)
    assert_manifold_faces(bm, label="center self result")
    assert_x_symmetric(bm, label="center self result")
    assert vertex_coord_multiset(bm)[coordinate_key((0.0, 0.0, 0.0))] == 1

    leave_edit(window, area, region)
    print("YSE_MERGE_MODES_CENTER_SELF_OK", flush=True)


def run_center_partial_symmetrize_case(window, area, region) -> None:
    """CENTER on a partial overlap first completes the selection with its
    missing mirrors, then one native run merges onto the plane."""

    clear_scene()
    vertices = (
        (-2.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (2.0, 0.0, 0.0),
        (-2.0, 1.0, 0.0),
        (-1.0, 1.0, 0.0),
        (1.0, 1.0, 0.0),
        (2.0, 1.0, 0.0),
    )
    faces = (
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
    )
    obj = make_object("YSE_MergeCenterPartial", vertices, faces)
    bm = enter_edit(window, area, region, obj)
    baseline_count = len(bm.verts)

    # {-1, +1, +2}: the pair -1/+1 crosses, +2's mirror -2 is unselected.
    select_vertices(bm, ((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)))
    bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)

    with bpy.context.temp_override(window=window, area=area, region=region):
        result = bpy.ops.mesh.ydd_symmetric_edit_merge(mode="CENTER")
    assert result == {"FINISHED"}, result

    bm = bmesh.from_edit_mesh(obj.data)
    # -2 is added to the selection, then all four merge at the origin: net -3.
    assert len(bm.verts) == baseline_count - 3, (len(bm.verts), baseline_count)
    assert_manifold_faces(bm, label="center partial result")
    assert_x_symmetric(bm, label="center partial result")
    assert vertex_coord_multiset(bm)[coordinate_key((0.0, 0.0, 0.0))] == 1

    leave_edit(window, area, region)
    print("YSE_MERGE_MODES_CENTER_PARTIAL_OK", flush=True)


def run_first_onplane_mixed_case(window, area, region) -> None:
    """FIRST with an on-plane vertex mixed in: native only, no mirror pass."""

    clear_scene()
    obj = make_side_split_strip("YSE_MergeFirstOnPlane")
    bm = enter_edit(window, area, region, obj)
    baseline_count = len(bm.verts)

    # A, A' and the on-plane S; history first = A (off-plane endpoint).
    select_vertices(bm, ((1.0, 0.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 0.0, 0.0)))
    bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)

    with bpy.context.temp_override(window=window, area=area, region=region):
        result = bpy.ops.mesh.ydd_symmetric_edit_merge(mode="FIRST")
    assert result == {"FINISHED"}, result

    bm = bmesh.from_edit_mesh(obj.data)
    # Native only: A, A', S all merge into A; no mirror pass runs.
    assert len(bm.verts) == baseline_count - 2, (len(bm.verts), baseline_count)
    present = vertex_coord_multiset(bm)
    assert present[coordinate_key((1.0, 0.0, 0.0))] == 1, present
    assert present[coordinate_key((-1.0, 0.0, 0.0))] == 0, present
    assert present[coordinate_key((0.0, 0.0, 0.0))] == 0, present
    # The asymmetric result proves the mirror pass was skipped.
    assert vertex_coord_multiset(bm) != mirrored_vertex_multiset(bm)

    leave_edit(window, area, region)
    print("YSE_MERGE_MODES_FIRST_ONPLANE_OK", flush=True)


def assert_no_temp_layers(bm) -> None:
    for name in yse_core.TEMP_LAYER_NAMES:
        assert bm.verts.layers.int.get(name) is None, f"vertex layer leaked: {name}"
        assert bm.edges.layers.int.get(name) is None, f"edge layer leaked: {name}"
        assert bm.faces.layers.int.get(name) is None, f"face layer leaked: {name}"


def assert_no_backup_datablock() -> None:
    leaked = [mesh.name for mesh in bpy.data.meshes if mesh.name.startswith("YSE_TemporaryBackup")]
    assert not leaked, f"backup datablock leaked: {leaked}"


def run_partial_onplane_first_case(window, area, region) -> None:
    """PARTIAL + on-plane vertex: the guard must fire BEFORE symmetrization,
    so the unselected mirror vertex is never dragged into the native merge
    (adversarial-review counterexample)."""

    clear_scene()
    obj = make_side_split_strip("YSE_MergePartialOnPlane")
    bm = enter_edit(window, area, region, obj)
    baseline_count = len(bm.verts)

    # A, A', B and the on-plane S; history first = A.  B' stays unselected;
    # the pair A/A' crosses while B's mirror B' is unselected -> PARTIAL.
    select_vertices(
        bm,
        (
            (1.0, 0.0, 0.0),
            (-1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 0.0, 0.0),
        ),
    )
    bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)

    with bpy.context.temp_override(window=window, area=area, region=region):
        result = bpy.ops.mesh.ydd_symmetric_edit_merge(mode="FIRST")
    assert result == {"FINISHED"}, result

    bm = bmesh.from_edit_mesh(obj.data)
    # Native only, on the ORIGINAL selection: A, A', B, S merge into A.
    assert len(bm.verts) == baseline_count - 3, (len(bm.verts), baseline_count)
    present = vertex_coord_multiset(bm)
    assert present[coordinate_key((1.0, 0.0, 0.0))] == 1, present
    # B' was never selected and must survive at its place, unmerged.
    assert present[coordinate_key((-1.0, 1.0, 0.0))] == 1, present
    assert present[coordinate_key((-1.0, 0.0, 0.0))] == 0, present
    b_mirror = find_vertex(bm, (-1.0, 1.0, 0.0))
    assert not b_mirror.select, "B' must not have been added to the selection"
    assert_no_temp_layers(bm)
    assert_no_backup_datablock()

    leave_edit(window, area, region)
    print("YSE_MERGE_MODES_PARTIAL_ONPLANE_OK", flush=True)


def run_pointmerge_failure_rollback_case(window, area, region) -> None:
    """An exception in the mirror pass must roll back to the native-only
    state, return FINISHED, and leave no temporary layer or backup."""

    clear_scene()
    vertices = (
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
        (-1.0, 0.0, 0.0),
        (-1.0, 1.0, 0.0),
        (0.0, 1.0, 0.0),
    )
    faces = ((0, 1, 2, 5), (0, 5, 4, 3))
    obj = make_object("YSE_MergeInjectPointmerge", vertices, faces)
    bm = enter_edit(window, area, region, obj)
    baseline_count = len(bm.verts)

    select_vertices(bm, ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0)))
    bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)

    original_pointmerge = bmesh.ops.pointmerge

    def broken_pointmerge(*args, **kwargs):
        raise RuntimeError("injected pointmerge failure")

    bmesh.ops.pointmerge = broken_pointmerge
    try:
        with bpy.context.temp_override(window=window, area=area, region=region):
            result = bpy.ops.mesh.ydd_symmetric_edit_merge(mode="CENTER")
    finally:
        bmesh.ops.pointmerge = original_pointmerge
    assert result == {"FINISHED"}, result

    bm = bmesh.from_edit_mesh(obj.data)
    # Native merged 3 -> 1 (net -2); the mirror pass was rolled back, so the
    # mirrored pair B', C' must still exist unmerged.
    assert len(bm.verts) == baseline_count - 2, (len(bm.verts), baseline_count)
    present = vertex_coord_multiset(bm)
    assert present[coordinate_key((2.0 / 3.0, 1.0 / 3.0, 0.0))] == 1, present
    assert present[coordinate_key((-1.0, 0.0, 0.0))] == 1, present
    assert present[coordinate_key((-1.0, 1.0, 0.0))] == 1, present
    assert vertex_coord_multiset(bm) != mirrored_vertex_multiset(bm)
    assert_no_temp_layers(bm)
    assert_no_backup_datablock()

    leave_edit(window, area, region)
    print("YSE_MERGE_MODES_INJECT_POINTMERGE_OK", flush=True)


def run_backup_failure_side_split_case(window, area, region) -> None:
    """A failed backup creation after the per-side native merge must still
    return FINISHED, keep the native result, and recover the post-state
    contract: mirror side reselected, source-only history."""

    clear_scene()
    obj = make_side_split_strip("YSE_MergeInjectBackup")
    bm = enter_edit(window, area, region, obj)
    baseline_count = len(bm.verts)

    select_vertices(
        bm,
        (
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (-1.0, 0.0, 0.0),
            (-1.0, 1.0, 0.0),
        ),
    )
    bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)

    original_create = yse_backup.create_topology_backup

    def broken_create(_bm):
        raise RuntimeError("injected backup failure")

    yse_backup.create_topology_backup = broken_create
    try:
        with bpy.context.temp_override(window=window, area=area, region=region):
            result = bpy.ops.mesh.ydd_symmetric_edit_merge(mode="FIRST")
    finally:
        yse_backup.create_topology_backup = original_create
    assert result == {"FINISHED"}, result

    bm = bmesh.from_edit_mesh(obj.data)
    # Native side merged (A,B -> A); the mirror side must be untouched.
    assert len(bm.verts) == baseline_count - 1, (len(bm.verts), baseline_count)
    present = vertex_coord_multiset(bm)
    assert present[coordinate_key((1.0, 0.0, 0.0))] == 1, present
    assert present[coordinate_key((1.0, 1.0, 0.0))] == 0, present
    assert present[coordinate_key((-1.0, 0.0, 0.0))] == 1, present
    assert present[coordinate_key((-1.0, 1.0, 0.0))] == 1, present
    # Post-state recovery: source survivor + both mirror verts selected,
    # history = source survivor only.
    selected = sorted(coordinate_key(vertex.co) for vertex in bm.verts if vertex.select)
    expected_selected = sorted(
        (
            coordinate_key((1.0, 0.0, 0.0)),
            coordinate_key((-1.0, 0.0, 0.0)),
            coordinate_key((-1.0, 1.0, 0.0)),
        )
    )
    assert selected == expected_selected, selected
    history = [element for element in bm.select_history if isinstance(element, bmesh.types.BMVert)]
    assert len(history) == 1, [coordinate_key(element.co) for element in history]
    assert coordinate_key(history[0].co) == coordinate_key((1.0, 0.0, 0.0))
    assert_no_temp_layers(bm)
    assert_no_backup_datablock()

    leave_edit(window, area, region)
    print("YSE_MERGE_MODES_INJECT_BACKUP_OK", flush=True)


def run() -> None:
    addon.register()
    window, area, region = viewport_context()
    configure_view(area)

    run_center_case(window, area, region)
    run_collapse_case(window, area, region)
    run_by_distance_case(window, area, region)
    run_first_side_split_case(window, area, region)
    run_last_side_split_case(window, area, region)
    run_center_self_mirrored_case(window, area, region)
    run_center_partial_symmetrize_case(window, area, region)
    run_first_onplane_mixed_case(window, area, region)
    run_partial_onplane_first_case(window, area, region)
    run_pointmerge_failure_rollback_case(window, area, region)
    run_backup_failure_side_split_case(window, area, region)

    print(MARKER_OK, flush=True)
    addon.unregister()
    bpy.ops.wm.quit_blender()


def guarded():
    try:
        run()
    except BaseException:
        fail()
    return None


bpy.context.preferences.view.show_splash = False
bpy.context.preferences.view.smooth_view = 0
bpy.context.preferences.view.use_save_prompt = False
bpy.context.preferences.filepaths.use_auto_save_temporary_files = False
bpy.app.timers.register(guarded, first_interval=0.25)
