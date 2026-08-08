# SPDX-License-Identifier: GPL-3.0-or-later

"""Headless unit checks for the REPLAY pipeline's deterministic helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import bmesh
from mathutils import Vector

PACKAGE_PARENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_PARENT))

from ydd_symmetric_edit import backup as yse_backup  # noqa: E402
from ydd_symmetric_edit import replay  # noqa: E402
from ydd_symmetric_edit._types import TopologyBackup  # noqa: E402


def check_shared_off_classification_and_partial_mirror_set() -> None:
    coords = tuple(
        Vector(co)
        for co in (
            (-1.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (-2.0, 2.0, 0.0),
        )
    )
    snapshot = replay.classify_mirror_selection(
        coords,
        (0, 2, 3),
        axis_index=0,
        tolerance=1.0e-5,
    )

    assert snapshot.selected == frozenset({0, 2, 3})
    assert snapshot.shared == frozenset({2})
    assert snapshot.off == frozenset({0, 3})
    assert snapshot.mirror_by_source == {0: 1}
    assert snapshot.mirrors == frozenset({1})
    assert snapshot.missing == frozenset({3})
    assert not replay.selection_crosses_mirror(snapshot)

    crossing = replay.classify_mirror_selection(
        coords,
        (0, 1, 2),
        axis_index=0,
        tolerance=1.0e-5,
    )
    assert replay.selection_crosses_mirror(crossing)


def check_merge_cluster_partitioning() -> None:
    bm = bmesh.new()
    try:
        vertices = [bm.verts.new((float(index), 0.0, 0.0)) for index in range(5)]
        for first, second in zip(vertices, vertices[1:], strict=False):
            bm.edges.new((first, second))
        bm.verts.ensure_lookup_table()
        bm.verts.index_update()

        selected = (0, 1, 3, 4)
        assert replay.split_merge_clusters(bm, selected, "COLLAPSE") == ((0, 1), (3, 4))
        for mode in ("CENTER", "FIRST", "LAST"):
            assert replay.split_merge_clusters(bm, selected, mode) == (selected,)
    finally:
        bm.free()


def check_merge_target_precomputation() -> None:
    coords = (
        Vector((-2.0, 0.0, 0.0)),
        Vector((-1.0, 2.0, 0.0)),
        Vector((0.0, 1.0, 0.0)),
    )
    cluster = (0, 1, 2)
    expected_centroid = Vector((-1.0, 1.0, 0.0))

    assert replay.calculate_merge_target(cluster, coords, "CENTER") == expected_centroid
    assert replay.calculate_merge_target(cluster, coords, "COLLAPSE") == expected_centroid
    assert (
        replay.calculate_merge_target(
            cluster,
            coords,
            "FIRST",
            history_coords=(coords[1], coords[0]),
        )
        == coords[1]
    )
    assert (
        replay.calculate_merge_target(
            cluster,
            coords,
            "LAST",
            history_coords=(coords[1], coords[0]),
        )
        == coords[0]
    )


def check_connect_history_mapping_is_all_or_nothing() -> None:
    coords = tuple(
        Vector(co)
        for co in (
            (-1.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (-2.0, 2.0, 0.0),
            (2.0, 2.0, 0.0),
        )
    )
    history = (coords[0].copy(), coords[2].copy(), coords[3].copy())
    assert replay.map_mirrored_history(
        history,
        coords,
        axis_index=0,
        tolerance=1.0e-5,
    ) == (1, 2, 4)

    incomplete_coords = coords[:-1]
    assert (
        replay.map_mirrored_history(
            history,
            incomplete_coords,
            axis_index=0,
            tolerance=1.0e-5,
        )
        is None
    )


def check_remove_backup_is_noexcept() -> None:
    """remove_backup runs from `finally` blocks whose contract is "always
    return the native result"; it must swallow even an invalidated-RNA-style
    failure on attribute access (adversarial-review finding)."""

    class _InvalidatedMesh:
        @property
        def name(self):
            raise RuntimeError("injected invalidated datablock")

    broken = TopologyBackup(mesh=_InvalidatedMesh(), shape_values={})  # type: ignore[arg-type]
    yse_backup.remove_backup(broken)  # must not raise
    yse_backup.remove_backup(None)  # must not raise either


def run() -> None:
    check_shared_off_classification_and_partial_mirror_set()
    check_merge_cluster_partitioning()
    check_merge_target_precomputation()
    check_connect_history_mapping_is_all_or_nothing()
    check_remove_backup_is_noexcept()
    print("YSE_REPLAY_UNITS_OK", flush=True)


if __name__ == "__main__":
    run()
