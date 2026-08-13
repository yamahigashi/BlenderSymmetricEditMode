# SPDX-License-Identifier: GPL-3.0-or-later

"""Generate the revision-3 interior-host differential golden corpus.

Run from Blender in background mode.  The WSL test environment intentionally
does not execute this module because it imports Blender's BMesh runtime.
"""

from __future__ import annotations

import inspect
import json
import re
import subprocess
import sys
import traceback
from pathlib import Path

PACKAGE_PARENT = Path(__file__).resolve().parents[1]
TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS_DIR))
sys.path.insert(0, str(PACKAGE_PARENT))

from fixtures_interior_host import GOLDEN_BUILDERS, fixture_from_builder  # noqa: E402

from ydd_symmetric_edit import stitch_reflect  # noqa: E402

OUTPUT = Path(__file__).resolve().parent / "golden" / "interior_host_rev3.json"


def _hex_coordinate(vertex) -> tuple[str, str, str]:
    return tuple(float(component).hex() for component in vertex.co)


def _canonical_cycle(indices: list[int]) -> list[int]:
    if not indices:
        return []
    rotations = [indices[offset:] + indices[:offset] for offset in range(len(indices))]
    reversed_indices = list(reversed(indices))
    rotations.extend(reversed_indices[offset:] + reversed_indices[:offset] for offset in range(len(indices)))
    return list(min(rotations))


def canonicalize_bmesh(bm) -> dict:
    """Return topology, incidence, coordinates, and selection without BMesh indices."""

    vertices = sorted(
        (vertex for vertex in bm.verts if vertex.is_valid),
        key=lambda vertex: (_hex_coordinate(vertex), int(vertex.index)),
    )
    vertex_ids = {hash(vertex): index for index, vertex in enumerate(vertices)}
    coordinate_keys = [_hex_coordinate(vertex) for vertex in vertices]
    assert len(coordinate_keys) == len(set(coordinate_keys)), "canonical vertex coordinates are not unique"
    vertex_records = [{"co": list(_hex_coordinate(vertex)), "selected": bool(vertex.select)} for vertex in vertices]

    face_layer = bm.faces.layers.int.get(".yse_original_face_id")
    faces = [face for face in bm.faces if face.is_valid]
    raw_faces = []
    for face in faces:
        cycle = [vertex_ids[hash(vertex)] for vertex in face.verts]
        face_id = int(face[face_layer]) if face_layer is not None else None
        raw_faces.append(
            (
                tuple(_canonical_cycle(cycle)),
                int(face.index),
                face,
                {"id": face_id, "verts": _canonical_cycle(cycle), "selected": bool(face.select)},
            )
        )
    raw_faces.sort(key=lambda item: (item[0], item[3]["id"] is None, item[3]["id"] or -1, item[1]))
    face_ids = {hash(item[2]): index for index, item in enumerate(raw_faces)}
    face_records = [item[3] for item in raw_faces]

    edges = []
    for edge in (edge for edge in bm.edges if edge.is_valid):
        endpoints = tuple(sorted((vertex_ids[hash(edge.verts[0])], vertex_ids[hash(edge.verts[1])])))
        linked_faces = sorted(
            face_ids[hash(face)] for face in edge.link_faces if face.is_valid and hash(face) in face_ids
        )
        edges.append(
            (
                endpoints,
                tuple(linked_faces),
                int(edge.index),
                {"verts": list(endpoints), "faces": linked_faces, "selected": bool(edge.select)},
            )
        )
    edges.sort(key=lambda item: (item[0], item[1], item[2]))
    edge_records = [item[3] for item in edges]
    return {"vertices": vertex_records, "edges": edge_records, "faces": face_records}


def call_reflected_gate(bm, source_edges, axis, tolerance, topology) -> bool:
    function = stitch_reflect.reflected_path_uses_only_target_boundaries
    arguments = (bm, source_edges, axis, tolerance, topology.mirror_face_ids)
    if "carrier_frames" in inspect.signature(function).parameters:
        return bool(function(*arguments, carrier_frames=topology.carrier_frames))
    return bool(function(*arguments))


def call_reflected_apply(bm, source_edges, axis, tolerance, topology):
    function = stitch_reflect.apply_reflected_path_topology
    arguments = (bm, source_edges, axis, tolerance, topology.mirror_face_ids)
    if "carrier_frames" in inspect.signature(function).parameters:
        return function(*arguments, carrier_frames=topology.carrier_frames)
    return function(*arguments)


def run_fixture(name: str, builder) -> dict:
    fixture = fixture_from_builder(name, builder)
    try:
        gate = call_reflected_gate(fixture.bm, fixture.source_edges, fixture.axis, fixture.tolerance, fixture.topology)
        created, already, reason = call_reflected_apply(
            fixture.bm, fixture.source_edges, fixture.axis, fixture.tolerance, fixture.topology
        )
        if not gate or reason:
            raise AssertionError((name, gate, created, already, reason))
        return {
            "name": name,
            "gate": gate,
            "gate_reason": "",
            "created": int(created),
            "already": int(already),
            "apply_reason": reason,
            "topology": canonicalize_bmesh(fixture.bm),
        }
    finally:
        fixture.free()


def git_head() -> str:
    value = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(PACKAGE_PARENT),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert re.fullmatch(r"[0-9a-fA-F]{40,64}", value), value
    return value


def generate(path: Path = OUTPUT) -> dict:
    corpus = {
        "schema": 1,
        "revision": "rev3",
        "git_head": git_head(),
        "fixtures": [run_fixture(name, builder) for name, builder in GOLDEN_BUILDERS],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(corpus, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return corpus


def main() -> None:
    corpus = generate()
    print(f"wrote {OUTPUT} fixtures={len(corpus['fixtures'])}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise SystemExit(1) from None
