# SPDX-License-Identifier: GPL-3.0-or-later

"""Measure rev3 interior-chain gate plus apply on the 58k-class grid."""

from __future__ import annotations

import json
import statistics
import sys
import time
import traceback
from pathlib import Path

import bmesh

PACKAGE_PARENT = Path(__file__).resolve().parents[1]
TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS_DIR))
sys.path.insert(0, str(PACKAGE_PARENT))

import perf_fixtures  # noqa: E402
from generate_interior_host_golden import call_reflected_apply, call_reflected_gate  # noqa: E402

from ydd_symmetric_edit import matching, snapshot, stitch_pathedges  # noqa: E402

TOLERANCE = perf_fixtures.TOLERANCE
AXIS = matching.AXIS_INDEX["X"]
REPEATS = 5
GRID_SEGMENTS = 240
EXPECTED_VERTS = (GRID_SEGMENTS + 1) ** 2
OUTPUT = Path(__file__).resolve().parent / "benchmarks" / "interior_host_rev3.json"


def _split(edge, factor: float):
    return bmesh.utils.edge_split(edge, edge.verts[0], factor)[1]


def build_large_interior_fixture():
    bm = bmesh.new()
    perf_fixtures.build_grid(bm, segments=GRID_SEGMENTS)
    assert len(bm.verts) == EXPECTED_VERTS == 58081
    topology = snapshot.prepare_topology(bm, AXIS, TOLERANCE)
    host = next(
        face
        for face in bm.faces
        if float(face.calc_center_median().x) < -0.25 and abs(float(face.calc_center_median().y)) < 0.05
    )
    horizontal = [
        edge
        for edge in host.edges
        if abs(float(edge.verts[0].co.y) - float(edge.verts[1].co.y)) <= 1.0e-8
        and all(float(vertex.co.x) < -TOLERANCE for vertex in edge.verts)
    ]
    assert len(horizontal) == 2, len(horizontal)
    bottom = _split(min(horizontal, key=lambda edge: min(float(vertex.co.y) for vertex in edge.verts)), 0.5)
    top = _split(max(horizontal, key=lambda edge: max(float(vertex.co.y) for vertex in edge.verts)), 0.5)
    host = next(face for face in bm.faces if bottom in face.verts and top in face.verts)
    point = host.calc_center_median()
    bmesh.utils.face_split(host, bottom, top, coords=[tuple(point)])
    source, side, total, crossing = stitch_pathedges.collect_source_path_edges(bm, AXIS, TOLERANCE, "NEGATIVE")
    assert side == "NEGATIVE", (side, total, crossing)
    assert crossing == 0
    assert source
    return bm, source, topology


def measure_once() -> tuple[float, bool, int, int, str, tuple[int, int, int], int]:
    bm, source, topology = build_large_interior_fixture()
    counts = (len(bm.verts), len(bm.edges), len(bm.faces))
    try:
        started = time.perf_counter_ns()
        gate = call_reflected_gate(bm, source, AXIS, TOLERANCE, topology)
        created, already, reason = call_reflected_apply(bm, source, AXIS, TOLERANCE, topology)
        elapsed_ms = (time.perf_counter_ns() - started) / 1.0e6
        assert gate is True
        assert reason == "", reason
        assert int(created) + int(already) == len(source), (created, already, len(source))
        return elapsed_ms, bool(gate), int(created), int(already), reason, counts, len(source)
    finally:
        bm.free()


def main(output: Path = OUTPUT) -> None:
    samples = [measure_once() for _ in range(REPEATS)]
    assert len({sample[5] for sample in samples}) == 1
    assert len({sample[6] for sample in samples}) == 1
    corpus = {
        "schema": 1,
        "revision": "rev3",
        "fixture": "perf_fixtures.build_grid_with_interior_chain_segments_240",
        "mesh_counts": list(samples[0][5]),
        "source_count": samples[0][6],
        "repeats": REPEATS,
        "samples_ms": [sample[0] for sample in samples],
        "median_ms": statistics.median(sample[0] for sample in samples),
        "max_ms": max(sample[0] for sample in samples),
        "gate": [sample[1] for sample in samples],
        "created": [sample[2] for sample in samples],
        "already": [sample[3] for sample in samples],
        "reasons": [sample[4] for sample in samples],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(corpus, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(corpus, sort_keys=True), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise SystemExit(1) from None
