# SPDX-License-Identifier: GPL-3.0-or-later

"""Reproducible performance fixtures for large-mesh gates (plan4 §T-0).

Deterministic (no RNG). Shared by diagnostics, T-gates, and equivalence tests.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import bmesh

# Prefer production constant; fall back to the layer name if the package is
# unavailable (standalone fixture self-check under a minimal path).
try:
    from ydd_symmetric_edit.layer_names import EDGE_ORIGINAL_LAYER
except Exception:  # pragma: no cover - Blender headless always has the package
    EDGE_ORIGINAL_LAYER = ".yse_original_edge"

TOLERANCE = 1.0e-5

# Grid contract (§T-0): bmesh.ops.create_grid segments = cell count.
GRID_SEGMENTS = 224
GRID_SIZE = 2.0
EXPECTED_VERTS = 50625  # 225^2
EXPECTED_EDGES = 100800
EXPECTED_FACES = 50176  # 224^2

# Dense one-sided component (§T-0 (e) / spike_one_sided dense mode).
# ±(1.0 + 0.3·tol·i, 0, 0), i = 0..3 → 8 vertices.
DENSE_CANDIDATE_COORDS: tuple[tuple[float, float, float], ...] = tuple(
    (sign * (1.0 + 0.3 * TOLERANCE * i), 0.0, 0.0) for i in range(4) for sign in (1.0, -1.0)
)

Domain = Literal["vert", "edge", "face"]


def build_grid(bm: bmesh.types.BMesh, *, segments: int = GRID_SEGMENTS, size: float = GRID_SIZE) -> None:
    """Fill *bm* with the contract grid and self-verify V/E/F plus lattice columns."""

    bmesh.ops.create_grid(bm, x_segments=segments, y_segments=segments, size=size)
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    bm.verts.index_update()
    bm.edges.index_update()
    bm.faces.index_update()

    if segments == GRID_SEGMENTS and abs(size - GRID_SIZE) < 1.0e-12:
        assert len(bm.verts) == EXPECTED_VERTS, f"verts={len(bm.verts)} expected {EXPECTED_VERTS}"
        assert len(bm.edges) == EXPECTED_EDGES, f"edges={len(bm.edges)} expected {EXPECTED_EDGES}"
        assert len(bm.faces) == EXPECTED_FACES, f"faces={len(bm.faces)} expected {EXPECTED_FACES}"
        assert _column_exists(bm, 0.0), "X=0 vertex column missing"
        assert _column_exists(bm, 0.5), "X=+0.5 vertex column missing"


def _column_exists(bm: bmesh.types.BMesh, x_value: float, tol: float = TOLERANCE) -> bool:
    return any(abs(float(v.co.x) - x_value) <= tol for v in bm.verts)


def _ensure_edge_original_layer(bm: bmesh.types.BMesh):
    layer = bm.edges.layers.int.get(EDGE_ORIGINAL_LAYER)
    if layer is None:
        layer = bm.edges.layers.int.new(EDGE_ORIGINAL_LAYER)
    return layer


def apply_cut_fixture(bm: bmesh.types.BMesh, *, count: int = 200, tol: float = TOLERANCE) -> list[int]:
    """Mark the first *count* X=+0.5 column edges as native-cut (EDGE_ORIGINAL=0).

    Ordering: ascending min(endpoint Y), ties by edge index ascending.
    Returns the involved edge index list (do not re-discover by coordinates).
    Asserts affected face count ≤ 400.
    """

    bm.edges.ensure_lookup_table()
    bm.edges.index_update()
    layer = _ensure_edge_original_layer(bm)

    # Stamp every edge with a non-zero original id, then zero the cut set.
    for edge_id, edge in enumerate(bm.edges, start=1):
        edge[layer] = edge_id

    candidates: list[tuple[float, int, bmesh.types.BMEdge]] = []
    for edge in bm.edges:
        v0, v1 = edge.verts
        if abs(float(v0.co.x) - 0.5) <= tol and abs(float(v1.co.x) - 0.5) <= tol:
            y_key = min(float(v0.co.y), float(v1.co.y))
            candidates.append((y_key, edge.index, edge))

    candidates.sort(key=lambda item: (item[0], item[1]))
    assert len(candidates) >= count, f"X=+0.5 column edges={len(candidates)} < {count}"

    selected = candidates[:count]
    edge_indices: list[int] = []
    affected_faces: set[int] = set()
    for _, edge_index, edge in selected:
        edge[layer] = 0
        edge_indices.append(edge_index)
        for face in edge.link_faces:
            if face.is_valid:
                affected_faces.add(face.index)

    assert len(affected_faces) <= 400, f"affected faces={len(affected_faces)} > 400"
    return edge_indices


def _domain_sequence(bm: bmesh.types.BMesh, domain: Domain) -> Sequence:
    if domain == "vert":
        bm.verts.ensure_lookup_table()
        return bm.verts
    if domain == "edge":
        bm.edges.ensure_lookup_table()
        return bm.edges
    if domain == "face":
        bm.faces.ensure_lookup_table()
        return bm.faces
    raise ValueError(f"unknown domain: {domain!r}")


def _representative_x(element, domain: Domain) -> float:
    if domain == "vert":
        return float(element.co.x)
    if domain == "edge":
        return 0.5 * (float(element.verts[0].co.x) + float(element.verts[1].co.x))
    # face centroid X
    total = 0.0
    for vertex in element.verts:
        total += float(vertex.co.x)
    return total / max(len(element.verts), 1)


def selection_indices(
    bm: bmesh.types.BMesh,
    domain: Domain,
    *,
    positive_only: bool = False,
    count: int = 1000,
    tol: float = TOLERANCE,
) -> list[int]:
    """Return deterministic selection indices for *domain* (§T-0 (b)).

    k = n // 1000; take indices where index % k == 0, ascending, first *count*.
    positive_only filters to representative X > tol before the same rule.
    """

    elements = _domain_sequence(bm, domain)
    if positive_only:
        pool = [el for el in elements if _representative_x(el, domain) > tol]
    else:
        pool = list(elements)

    n = len(pool)
    assert n >= count, f"{domain} pool size {n} < {count}"
    k = n // 1000
    assert k >= 1, f"{domain} n={n} too small for k = n//1000"

    # Sort by mesh index, then apply index % k == 0 over the *pool position*
    # (0..n-1 in that order). For an unfiltered domain this coincides with
    # mesh index; after +X filtering, mesh-index modulo undershoots 1000.
    pool.sort(key=lambda el: el.index)
    chosen: list[int] = []
    for position, element in enumerate(pool):
        if position % k == 0:
            chosen.append(element.index)
            if len(chosen) >= count:
                break
    assert len(chosen) == count, f"{domain} selected {len(chosen)} != {count} (k={k}, n={n})"
    return chosen


def apply_selection_fixture(
    bm: bmesh.types.BMesh,
    domain: Domain,
    *,
    positive_only: bool = False,
    count: int = 1000,
    tol: float = TOLERANCE,
) -> list[int]:
    """Clear all selection and select the §T-0 (b) index set. Returns indices."""

    for v in bm.verts:
        v.select = False
    for e in bm.edges:
        e.select = False
    for f in bm.faces:
        f.select = False

    indices = selection_indices(bm, domain, positive_only=positive_only, count=count, tol=tol)
    if domain == "vert":
        bm.verts.ensure_lookup_table()
        for index in indices:
            bm.verts[index].select = True
    elif domain == "edge":
        bm.edges.ensure_lookup_table()
        for index in indices:
            bm.edges[index].select = True
    else:
        bm.faces.ensure_lookup_table()
        for index in indices:
            bm.faces[index].select = True
    return indices


def apply_on_plane_mixed_selection(
    bm: bmesh.types.BMesh,
    *,
    count: int = 1000,
    tol: float = TOLERANCE,
) -> dict[str, list[int]]:
    """§T-0 (c): vertex selection (b) plus every X=0 column vertex.

    Returns ``{"base": [...], "on_plane": [...], "all": [...]}`` index lists.
    """

    base = apply_selection_fixture(bm, "vert", positive_only=False, count=count, tol=tol)
    on_plane: list[int] = []
    bm.verts.ensure_lookup_table()
    for vertex in bm.verts:
        if abs(float(vertex.co.x)) <= tol:
            vertex.select = True
            on_plane.append(vertex.index)
    on_plane.sort()
    all_indices = sorted(set(base) | set(on_plane))
    return {"base": base, "on_plane": on_plane, "all": all_indices}


def apply_asymmetric_point(bm: bmesh.types.BMesh, *, delta: float = 0.37) -> int:
    """§T-0 (d): bm.verts[0].co.x += delta. Returns the perturbed vertex index."""

    bm.verts.ensure_lookup_table()
    bm.verts[0].co.x += delta
    return 0


def apply_dense_candidates(bm: bmesh.types.BMesh, *, tol: float = TOLERANCE) -> list[int]:
    """§T-0 (e): add 8 dense-component verts at ±(1.0 + 0.3·tol·i, 0, 0), i=0..3.

    Returns the new vertex indices (in creation order).
    """

    del tol  # coordinates already baked with TOLERANCE in DENSE_CANDIDATE_COORDS
    new_indices: list[int] = []
    for x, y, z in DENSE_CANDIDATE_COORDS:
        vertex = bm.verts.new((x, y, z))
        new_indices.append(vertex.index)
    bm.verts.ensure_lookup_table()
    bm.verts.index_update()
    # Re-read indices after index_update (stable 0..n-1).
    # Newly created verts are the last len(DENSE_CANDIDATE_COORDS) entries.
    n = len(bm.verts)
    count = len(DENSE_CANDIDATE_COORDS)
    return list(range(n - count, n))


def apply_full_mesh_selection(bm: bmesh.types.BMesh) -> dict[str, int]:
    """§T-0 (f): select every vertex; return domain counts for full-face queries."""

    for vertex in bm.verts:
        vertex.select = True
    for edge in bm.edges:
        edge.select = True
    for face in bm.faces:
        face.select = True
    return {
        "verts": len(bm.verts),
        "edges": len(bm.edges),
        "faces": len(bm.faces),
    }


def clear_selection(bm: bmesh.types.BMesh) -> None:
    for vertex in bm.verts:
        vertex.select = False
    for edge in bm.edges:
        edge.select = False
    for face in bm.faces:
        face.select = False


def self_check() -> None:
    """Build every fixture variant and assert contract invariants."""

    bm = bmesh.new()
    try:
        build_grid(bm)
        assert len(bm.verts) == EXPECTED_VERTS
        assert len(bm.edges) == EXPECTED_EDGES
        assert len(bm.faces) == EXPECTED_FACES

        cut_edges = apply_cut_fixture(bm)
        assert len(cut_edges) == 200
        assert len(set(cut_edges)) == 200
        # Order is min(endpoint Y) ascending, then edge index (not pure index sort).
        layer = bm.edges.layers.int.get(EDGE_ORIGINAL_LAYER)
        assert layer is not None
        prev_key = (-float("inf"), -1)
        for index in cut_edges:
            edge = bm.edges[index]
            assert int(edge[layer]) == 0
            y_key = min(float(edge.verts[0].co.y), float(edge.verts[1].co.y))
            key = (y_key, index)
            assert key >= prev_key
            prev_key = key

        for domain in ("vert", "edge", "face"):
            indices = selection_indices(bm, domain)  # type: ignore[arg-type]
            assert len(indices) == 1000
            assert indices == sorted(indices)
            pos = selection_indices(bm, domain, positive_only=True)  # type: ignore[arg-type]
            assert len(pos) == 1000
            for index in pos:
                el = _domain_sequence(bm, domain)[index]  # type: ignore[arg-type]
                assert _representative_x(el, domain) > TOLERANCE  # type: ignore[arg-type]

        mixed = apply_on_plane_mixed_selection(bm)
        assert len(mixed["base"]) == 1000
        assert len(mixed["on_plane"]) > 0
        assert set(mixed["base"]) | set(mixed["on_plane"]) == set(mixed["all"])
        for index in mixed["on_plane"]:
            assert abs(float(bm.verts[index].co.x)) <= TOLERANCE

        # Fresh mesh for destructive / additive fixtures.
    finally:
        bm.free()

    bm = bmesh.new()
    try:
        build_grid(bm)
        x_before = float(bm.verts[0].co.x)
        apply_asymmetric_point(bm)
        # BMesh stores co as float32; 0.37 is not binary-exact.
        assert abs(float(bm.verts[0].co.x) - (x_before + 0.37)) < 1.0e-5
    finally:
        bm.free()

    bm = bmesh.new()
    try:
        build_grid(bm)
        before = len(bm.verts)
        dense = apply_dense_candidates(bm)
        assert len(dense) == 8
        assert len(bm.verts) == before + 8
        # BMesh co is float32; match each expected coord within tol.
        for i, (ex, ey, ez) in enumerate(DENSE_CANDIDATE_COORDS):
            v = bm.verts[dense[i]]
            assert abs(float(v.co.x) - ex) <= TOLERANCE * 2
            assert abs(float(v.co.y) - ey) <= TOLERANCE
            assert abs(float(v.co.z) - ez) <= TOLERANCE
    finally:
        bm.free()

    bm = bmesh.new()
    try:
        build_grid(bm)
        counts = apply_full_mesh_selection(bm)
        assert counts["verts"] == EXPECTED_VERTS
        assert counts["faces"] == EXPECTED_FACES
        assert all(v.select for v in bm.verts)
        assert all(f.select for f in bm.faces)
    finally:
        bm.free()

    print("PERF_FIXTURES_SELF_CHECK_OK")
    print(f"  grid V/E/F={EXPECTED_VERTS}/{EXPECTED_EDGES}/{EXPECTED_FACES}")
    print(f"  cut edges=200 dense_coords={len(DENSE_CANDIDATE_COORDS)}")
    print(f"  EDGE_ORIGINAL_LAYER={EDGE_ORIGINAL_LAYER!r}")


def main() -> None:
    self_check()


if __name__ == "__main__":
    main()
