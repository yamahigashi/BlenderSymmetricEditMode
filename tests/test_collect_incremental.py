# SPDX-License-Identifier: GPL-3.0-or-later

"""Standalone differential checks for §I-3b incremental Knife collect.

This file is executed by Blender's ``run_headless_test.bat`` and intentionally
has no external test-runner dependency.  The frozen collector below is the independent
record-order oracle; the incremental candidate is accepted only on exact
bucket/record equality.
"""

from __future__ import annotations

import logging
import sys
import traceback
from pathlib import Path

import bmesh
from mathutils import Vector

PACKAGE_PARENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_PARENT))

from ydd_symmetric_edit import core, stitch  # noqa: E402

AXIS = core.AXIS_INDEX["X"]
TOLERANCE = 1.0e-5
MARKER = "YSE_COLLECT_INCREMENTAL_TEST_OK"
_SIDES = ("POSITIVE", "NEGATIVE", "CROSSES", "PLANE")


def _edge_side_for_test(edge):
    first, second = edge.verts
    a = float(first.co.x)
    b = float(second.co.x)
    if a >= -TOLERANCE and b >= -TOLERANCE and max(a, b) > TOLERANCE:
        return "POSITIVE"
    if a <= TOLERANCE and b <= TOLERANCE and min(a, b) < -TOLERANCE:
        return "NEGATIVE"
    if abs(a) <= TOLERANCE and abs(b) <= TOLERANCE:
        return "PLANE"
    return "CROSSES"


def _update_indices(bm):
    for elements in (bm.verts, bm.edges, bm.faces):
        elements.ensure_lookup_table()
        elements.index_update()


def _build_grid(segments=2):
    bm = bmesh.new()
    bmesh.ops.create_grid(bm, x_segments=segments, y_segments=segments, size=2.0)
    marker = bm.edges.layers.int.new(core.EDGE_ORIGINAL_LAYER)
    bm.edges.layers.int.new(core.EDGE_SELECTION_LAYER)
    bm.verts.layers.int.new(core.VERT_SELECTION_LAYER)
    face_ids = bm.faces.layers.int.new(core.FACE_ID_LAYER)
    for edge in bm.edges:
        edge[marker] = 0
        edge[bm.edges.layers.int.get(core.EDGE_SELECTION_LAYER)] = 0
    for face in bm.faces:
        face[face_ids] = 1
    _update_indices(bm)
    return bm


def _record_bits(by_side):
    return tuple(tuple(hash(edge) for edge in by_side[side]) for side in ("POSITIVE", "NEGATIVE", "CROSSES", "PLANE"))


def _frozen_collect(bm, axis_index, tolerance):
    """Independent eager oracle; do not call candidate discovery helpers."""

    marker = bm.edges.layers.int.get(core.EDGE_ORIGINAL_LAYER)
    face_ids = bm.faces.layers.int.get(core.FACE_ID_LAYER)
    by_side = {"POSITIVE": [], "NEGATIVE": [], "CROSSES": [], "PLANE": []}
    for edge in bm.edges:
        if marker is None:
            continue
        complement = (
            face_ids is not None
            and len(edge.link_faces) >= 2
            and len({int(face[face_ids]) for face in edge.link_faces}) == 1
        )
        if int(edge[marker]) != 0 and not complement:
            continue
        a = float(edge.verts[0].co[axis_index])
        b = float(edge.verts[1].co[axis_index])
        if a >= -tolerance and b >= -tolerance and max(a, b) > tolerance:
            side = "POSITIVE"
        elif a <= tolerance and b <= tolerance and min(a, b) < -tolerance:
            side = "NEGATIVE"
        elif abs(a) <= tolerance and abs(b) <= tolerance:
            side = "PLANE"
        else:
            side = "CROSSES"
        by_side[side].append(edge)
    return by_side, sum(len(edges) for edges in by_side.values())


def _summary(
    source,
    final_edges,
    cache_position,
    *,
    endpoint=False,
    pointmerge=False,
    pre_count=0,
    pre_faces=(),
    unexpected=False,
    removed_edges=0,
    removed_faces=0,
):
    return stitch.CrossingMutationSummary(
        (
            stitch.CrossingEdgeMutation(
                source_edge_id=hash(source),
                final_edges=tuple(final_edges),
                cache_position=cache_position,
                endpoint_reused=endpoint,
                pointmerged=pointmerge,
                pre_faces=tuple(pre_faces),
            ),
        ),
        removed_edge_count=removed_edges,
        removed_face_count=removed_faces,
        pre_apply_edge_count=pre_count,
        unexpected_topology_change=unexpected,
    )


def _fallback_collect(bm, previous, cache, summary):
    patched = stitch.patch_knife_path_edges_by_side(bm, previous, cache, summary, AXIS, TOLERANCE)
    if patched is not None:
        raise RuntimeError("incremental patch unexpectedly accepted a fallback case")
    return stitch.collect_knife_path_edges_by_side(bm, AXIS, TOLERANCE)


def check_track_a_record_bits_match_full_collect():
    bm = _build_grid(8)
    try:
        expected, expected_count = _frozen_collect(bm, AXIS, TOLERANCE)
        actual, actual_count = stitch.collect_knife_path_edges_by_side(bm, AXIS, TOLERANCE)
        if (actual_count, _record_bits(actual)) != (expected_count, _record_bits(expected)):
            raise RuntimeError("full collect differs from independent record oracle")
        candidates = []
        for edge in bm.edges:
            first, second = edge.verts
            if (
                abs(float(first.co.x) - float(second.co.x)) <= TOLERANCE
                and abs(abs(float(first.co.x)) - 1.0) <= TOLERANCE
            ):
                candidates.append(edge)
        positive = next(edge for edge in candidates if edge.verts[0].co.x > 0 and edge.verts[1].co.x > 0)
        negative = next(edge for edge in candidates if edge.verts[0].co.x < 0 and edge.verts[1].co.x < 0)
        previous = [edge for side in ("POSITIVE", "NEGATIVE", "CROSSES", "PLANE") for edge in actual[side]]
        cache = stitch.capture_knife_path_edge_cache(bm, previous)
        positions = {hash(edge): index for index, edge in enumerate(previous)}
        factor = 0.37
        positive_co = positive.verts[0].co.lerp(positive.verts[1].co, factor)
        negative_co = negative.verts[0].co.lerp(negative.verts[1].co, factor)
        cluster = stitch._MirroredPathCrossingCluster(
            positive_coordinate=positive_co,
            negative_coordinate=negative_co,
            positive=(_make_crossing_occurrence(positive, factor),),
            negative=(_make_crossing_occurrence(negative, factor),),
            tolerance=TOLERANCE,
        )
        result = stitch.apply_mirrored_path_crossings(bm, [cluster], cache_positions=positions, return_summary=True)
        if result[1]:
            raise RuntimeError(f"Track A producer apply failed: {result}")
        patched = stitch.patch_knife_path_edges_by_side(bm, actual, cache, result[2], AXIS, TOLERANCE)
        full, _ = stitch.collect_knife_path_edges_by_side(bm, AXIS, TOLERANCE)
        if (
            patched is None
            or patched[1] != sum(len(bucket) for bucket in full.values())
            or _record_bits(patched[0]) != _record_bits(full)
        ):
            raise RuntimeError("Track A incremental result differs from full collect")
    finally:
        bm.free()


def _make_crossing_occurrence(edge, factor):
    return stitch._MirroredPathOccurrence(
        edge=edge,
        edge_id=hash(edge),
        factor=factor,
        endpoint_index=None,
        edge_key=stitch._edge_survivor_key(edge, AXIS),
    )


def _build_crossing_fixture(multiple=False):
    bm = bmesh.new()
    # Layers first: layers.int.new reallocates the whole CustomData domain
    # and would invalidate BMEdge references captured before it.
    edge_marker = bm.edges.layers.int.new(core.EDGE_ORIGINAL_LAYER)
    edge_selection = bm.edges.layers.int.new(core.EDGE_SELECTION_LAYER)
    vertex_selection = bm.verts.layers.int.new(core.VERT_SELECTION_LAYER)
    bm.faces.layers.int.new(core.FACE_ID_LAYER)
    positive = bm.edges.new((bm.verts.new((1.0, -1.0, 0.0)), bm.verts.new((1.0, 1.0, 0.0))))
    negative = bm.edges.new((bm.verts.new((-1.0, -1.0, 0.0)), bm.verts.new((-1.0, 1.0, 0.0))))
    duplicate_positive = (
        bm.edges.new((bm.verts.new((1.0, -1.0, 0.0)), bm.verts.new((1.0, 1.0, 0.0)))) if multiple else None
    )
    duplicate_negative = (
        bm.edges.new((bm.verts.new((-1.0, -1.0, 0.0)), bm.verts.new((-1.0, 1.0, 0.0)))) if multiple else None
    )
    for edge in (positive, negative, duplicate_positive, duplicate_negative):
        if edge is None:
            continue
        edge[edge_marker] = 0
        edge[edge_selection] = 0
    for vertex in bm.verts:
        vertex[vertex_selection] = 0
    bm.edges.ensure_lookup_table()
    return bm, positive, negative, duplicate_positive, duplicate_negative


def check_producer_summary_and_patch_m2():
    bm, positive, negative, _duplicate_positive, _duplicate_negative = _build_crossing_fixture()
    try:
        before, _ = stitch.collect_knife_path_edges_by_side(bm, AXIS, TOLERANCE)
        previous = [edge for side in ("POSITIVE", "NEGATIVE", "CROSSES", "PLANE") for edge in before[side]]
        cache = stitch.capture_knife_path_edge_cache(bm, previous)
        cache_positions = {hash(edge): index for index, edge in enumerate(previous)}
        positive_factors = (0.25, 0.75)
        negative_factors = (0.25, 0.75)
        clusters = []
        for positive_factor, negative_factor in zip(positive_factors, negative_factors, strict=True):
            clusters.append(
                stitch._MirroredPathCrossingCluster(
                    positive_coordinate=Vector((1.0, -1.0 + 2.0 * positive_factor, 0.0)),
                    negative_coordinate=Vector((-1.0, -1.0 + 2.0 * negative_factor, 0.0)),
                    positive=(_make_crossing_occurrence(positive, positive_factor),),
                    negative=(_make_crossing_occurrence(negative, negative_factor),),
                    tolerance=TOLERANCE,
                )
            )
        result = stitch.apply_mirrored_path_crossings(
            bm,
            clusters,
            cache_positions=cache_positions,
            return_summary=True,
        )
        if result[1] or len(result) != 3:
            raise RuntimeError(f"producer apply failed: {result}")
        summary = result[2]
        by_source = {mutation.source_edge_id: mutation for mutation in summary.edges}
        tail = tuple(bm.edges[index] for index in range(summary.pre_apply_edge_count, len(bm.edges)))
        tail_cursor = 0
        for source in (positive, negative):
            mutation = by_source[hash(source)]
            if len(mutation.final_edges) != 3 or mutation.cache_position != cache_positions[hash(source)]:
                raise RuntimeError("producer did not report source m+1/cache position")
            if mutation.pre_faces:
                raise RuntimeError("wire fixture unexpectedly has pre faces")
            if mutation.final_edges[0] is not source:
                raise RuntimeError("producer final edge column did not retain source first")
            expected_tail = tail[tail_cursor : tail_cursor + 2]
            if tuple(mutation.final_edges[1:]) != expected_tail:
                raise RuntimeError("producer source final edges differ from relative BMesh tail order")
            if any(edge not in tail for edge in mutation.final_edges[1:]):
                raise RuntimeError("producer created edge was not in the post-apply tail")
            tail_cursor += 2
        if tail_cursor != len(tail):
            raise RuntimeError("producer tail ownership was incomplete")
        patched = stitch.patch_knife_path_edges_by_side(bm, before, cache, summary, AXIS, TOLERANCE)
        full, _ = stitch.collect_knife_path_edges_by_side(bm, AXIS, TOLERANCE)
        if patched is None or _record_bits(patched[0]) != _record_bits(full):
            raise RuntimeError("producer incremental patch differs from full collect")
    finally:
        bm.free()


def _build_same_face_positive_fixture():
    bm = bmesh.new()
    # Layers first: layers.int.new reallocates the whole CustomData domain
    # and would invalidate element references captured before it.
    marker = bm.edges.layers.int.new(core.EDGE_ORIGINAL_LAYER)
    selection = bm.edges.layers.int.new(core.EDGE_SELECTION_LAYER)
    vertex_selection = bm.verts.layers.int.new(core.VERT_SELECTION_LAYER)
    face_ids = bm.faces.layers.int.new(core.FACE_ID_LAYER)
    vertices = [
        bm.verts.new((1.0, -1.0, 0.0)),
        bm.verts.new((3.0, -1.0, 0.0)),
        bm.verts.new((3.0, 1.0, 0.0)),
        bm.verts.new((1.0, 1.0, 0.0)),
    ]
    face = bm.faces.new(vertices)
    face[face_ids] = 1
    for edge in bm.edges:
        edge[marker] = 0
        edge[selection] = 0
    for vertex in bm.verts:
        vertex[vertex_selection] = 0
    bm.edges.ensure_lookup_table()
    return bm, face


def check_same_face_cross_source_pure_split():
    bm, face = _build_same_face_positive_fixture()
    try:
        before, _ = stitch.collect_knife_path_edges_by_side(bm, AXIS, TOLERANCE)
        sources = [edge for edge in face.edges if _edge_side_for_test(edge) == "POSITIVE"]
        if len(sources) != 4:
            raise RuntimeError("same-face fixture did not create four positive boundary sources")
        first, second = sources[0], sources[2]
        previous = [edge for side in _SIDES for edge in before[side]]
        cache = stitch.capture_knife_path_edge_cache(bm, previous)
        positions = {hash(edge): index for index, edge in enumerate(previous)}
        clusters = []
        for edge, factor in ((first, 0.25), (second, 0.75)):
            coordinate = edge.verts[0].co.lerp(edge.verts[1].co, factor)
            clusters.append(
                stitch._MirroredPathCrossingCluster(
                    positive_coordinate=coordinate,
                    negative_coordinate=coordinate.copy(),
                    positive=(_make_crossing_occurrence(edge, factor),),
                    negative=(),
                    tolerance=TOLERANCE,
                )
            )
        result = stitch.apply_mirrored_path_crossings(bm, clusters, cache_positions=positions, return_summary=True)
        if result[1]:
            raise RuntimeError(f"same-face producer apply failed: {result}")
        summary = result[2]
        by_source = {mutation.source_edge_id: mutation for mutation in summary.edges}
        if any(len(by_source[hash(edge)].final_edges) != 2 for edge in (first, second)):
            raise RuntimeError("same-face source summary did not report m+1 edges")
        patched = stitch.patch_knife_path_edges_by_side(bm, before, cache, summary, AXIS, TOLERANCE)
        full, _ = stitch.collect_knife_path_edges_by_side(bm, AXIS, TOLERANCE)
        if patched is None or _record_bits(patched[0]) != _record_bits(full):
            raise RuntimeError("same-face cross-source patch differs from full collect")
    finally:
        bm.free()


def check_producer_endpoint_and_pointmerge_flags():
    bm, positive, negative, _duplicate_positive, _duplicate_negative = _build_crossing_fixture()
    try:
        previous, _ = stitch.collect_knife_path_edges_by_side(bm, AXIS, TOLERANCE)
        previous_edges = [edge for side in ("POSITIVE", "NEGATIVE", "CROSSES", "PLANE") for edge in previous[side]]
        cache = stitch.capture_knife_path_edge_cache(bm, previous_edges)
        cache_positions = {hash(edge): index for index, edge in enumerate(previous_edges)}
        endpoint_plan = [
            stitch._MirroredPathCrossingCluster(
                positive_coordinate=positive.verts[0].co.copy(),
                negative_coordinate=negative.verts[0].co.copy(),
                positive=(_make_crossing_occurrence(positive, 0.0),),
                negative=(_make_crossing_occurrence(negative, 0.0),),
                tolerance=TOLERANCE,
            )
        ]
        endpoint_plan[0] = stitch._MirroredPathCrossingCluster(
            positive_coordinate=endpoint_plan[0].positive_coordinate,
            negative_coordinate=endpoint_plan[0].negative_coordinate,
            positive=(
                stitch._MirroredPathOccurrence(
                    edge=positive,
                    edge_id=hash(positive),
                    factor=0.0,
                    endpoint_index=0,
                    edge_key=stitch._edge_survivor_key(positive, AXIS),
                ),
            ),
            negative=(
                stitch._MirroredPathOccurrence(
                    edge=negative,
                    edge_id=hash(negative),
                    factor=0.0,
                    endpoint_index=0,
                    edge_key=stitch._edge_survivor_key(negative, AXIS),
                ),
            ),
            tolerance=TOLERANCE,
        )
        endpoint_result = stitch.apply_mirrored_path_crossings(
            bm, endpoint_plan, cache_positions=cache_positions, return_summary=True
        )
        if endpoint_result[1]:
            raise RuntimeError(f"producer endpoint apply failed: {endpoint_result}")
        endpoint_summary = {item.source_edge_id: item for item in endpoint_result[2].edges}
        if not endpoint_summary[hash(positive)].endpoint_reused:
            raise RuntimeError("producer endpoint reuse flag was not set")
        if stitch.patch_knife_path_edges_by_side(bm, previous, cache, endpoint_result[2], AXIS, TOLERANCE) is not None:
            raise RuntimeError("producer endpoint reuse was incorrectly accepted by patch")
    finally:
        bm.free()

    bm, positive, negative, duplicate_positive, duplicate_negative = _build_crossing_fixture(multiple=True)
    try:
        previous, _ = stitch.collect_knife_path_edges_by_side(bm, AXIS, TOLERANCE)
        previous_edges = [edge for side in ("POSITIVE", "NEGATIVE", "CROSSES", "PLANE") for edge in previous[side]]
        cache = stitch.capture_knife_path_edge_cache(bm, previous_edges)
        cache_positions = {hash(edge): index for index, edge in enumerate(previous_edges)}
        source_ids = tuple(hash(edge) for edge in (positive, negative, duplicate_positive, duplicate_negative))
        cluster = stitch._MirroredPathCrossingCluster(
            positive_coordinate=Vector((1.0, 0.0, 0.0)),
            negative_coordinate=Vector((-1.0, 0.0, 0.0)),
            positive=(_make_crossing_occurrence(positive, 0.5), _make_crossing_occurrence(duplicate_positive, 0.5)),
            negative=(_make_crossing_occurrence(negative, 0.5), _make_crossing_occurrence(duplicate_negative, 0.5)),
            tolerance=TOLERANCE,
        )
        result = stitch.apply_mirrored_path_crossings(
            bm, [cluster], cache_positions=cache_positions, return_summary=True
        )
        if result[1]:
            raise RuntimeError(f"producer pointmerge apply failed: {result}")
        summary = {item.source_edge_id: item for item in result[2].edges}
        if not all(summary[source_id].pointmerged for source_id in source_ids):
            raise RuntimeError("producer pointmerge flag was not set")
        if stitch.patch_knife_path_edges_by_side(bm, previous, cache, result[2], AXIS, TOLERANCE) is not None:
            raise RuntimeError("producer pointmerge was incorrectly accepted by patch")
    finally:
        bm.free()


def check_multi_split_m2_record_order():
    bm = bmesh.new()
    try:
        marker = bm.edges.layers.int.new(core.EDGE_ORIGINAL_LAYER)
        bm.faces.layers.int.new(core.FACE_ID_LAYER)
        left = bm.verts.new((1.0, 0.0, 0.0))
        right = bm.verts.new((4.0, 0.0, 0.0))
        source = bm.edges.new((left, right))
        source[marker] = 0
        _update_indices(bm)
        before, _ = stitch.collect_knife_path_edges_by_side(bm, AXIS, TOLERANCE)
        cache = stitch.capture_knife_path_edge_cache(bm, [source])
        cache_position = 0
        pre_count = len(bm.edges)
        first, _ = bmesh.utils.edge_split(source, left, 0.25)
        second, _ = bmesh.utils.edge_split(first, first.verts[0], 2.0 / 3.0)
        for edge in (source, first, second):
            edge[marker] = 0
        _update_indices(bm)
        expected, expected_count = _frozen_collect(bm, AXIS, TOLERANCE)
        summary = _summary(source, (source, first, second), cache_position, pre_count=pre_count)
        patched = stitch.patch_knife_path_edges_by_side(bm, before, cache, summary, AXIS, TOLERANCE)
        if patched is None or patched[1] != expected_count or _record_bits(patched[0]) != _record_bits(expected):
            raise RuntimeError("m>=2 split record order differs from full collect")
    finally:
        bm.free()


def check_side_change_falls_back():
    bm = bmesh.new()
    try:
        marker = bm.edges.layers.int.new(core.EDGE_ORIGINAL_LAYER)
        bm.faces.layers.int.new(core.FACE_ID_LAYER)
        left = bm.verts.new((-2.0, 0.0, 0.0))
        right = bm.verts.new((2.0, 0.0, 0.0))
        source = bm.edges.new((left, right))
        source[marker] = 0
        _update_indices(bm)
        previous = {"POSITIVE": [], "NEGATIVE": [], "CROSSES": [source], "PLANE": []}
        cache = stitch.capture_knife_path_edge_cache(bm, [source])
        position = 0
        first, _ = bmesh.utils.edge_split(source, left, 0.5)
        second, _ = bmesh.utils.edge_split(first, first.verts[0], 0.5)
        for edge in (source, first, second):
            edge[marker] = 0
        _update_indices(bm)
        full, _ = _fallback_collect(
            bm,
            previous,
            cache,
            _summary(source, (source, first, second), position),
        )
        if _record_bits(full) != _record_bits(_frozen_collect(bm, AXIS, TOLERANCE)[0]):
            raise RuntimeError("side-change fallback did not use full result")
    finally:
        bm.free()


def check_producer_self_mirrored_side_fallback():
    bm = bmesh.new()
    try:
        marker = bm.edges.layers.int.new(core.EDGE_ORIGINAL_LAYER)
        bm.edges.layers.int.new(core.EDGE_SELECTION_LAYER)
        bm.verts.layers.int.new(core.VERT_SELECTION_LAYER)
        bm.faces.layers.int.new(core.FACE_ID_LAYER)
        left = bm.verts.new((-2.0, 0.0, 0.0))
        right = bm.verts.new((2.0, 0.0, 0.0))
        source = bm.edges.new((left, right))
        source[marker] = 0
        bm.edges.ensure_lookup_table()
        before, _ = stitch.collect_knife_path_edges_by_side(bm, AXIS, TOLERANCE)
        previous = [edge for side in _SIDES for edge in before[side]]
        cache = stitch.capture_knife_path_edge_cache(bm, previous)
        positions = {hash(edge): index for index, edge in enumerate(previous)}
        cluster = stitch._MirroredPathCrossingCluster(
            positive_coordinate=Vector((1.0, 0.0, 0.0)),
            negative_coordinate=Vector((-1.0, 0.0, 0.0)),
            positive=(_make_crossing_occurrence(source, 0.75),),
            negative=(_make_crossing_occurrence(source, 0.25),),
            tolerance=TOLERANCE,
        )
        result = stitch.apply_mirrored_path_crossings(bm, [cluster], cache_positions=positions, return_summary=True)
        if result[1]:
            raise RuntimeError(f"self-mirrored producer apply failed: {result}")
        mutation = result[2].edges[0]
        if len(mutation.final_edges) != 3 or not any(
            _edge_side_for_test(edge) != "CROSSES" for edge in mutation.final_edges
        ):
            raise RuntimeError("self-mirrored producer did not create side-changing final edges")
        if stitch.patch_knife_path_edges_by_side(bm, before, cache, result[2], AXIS, TOLERANCE) is not None:
            raise RuntimeError("self-mirrored side change was incorrectly accepted")
        full, _ = stitch.collect_knife_path_edges_by_side(bm, AXIS, TOLERANCE)
        if _record_bits(full) != _record_bits(_frozen_collect(bm, AXIS, TOLERANCE)[0]):
            raise RuntimeError("self-mirrored fallback differs from frozen full collect")
    finally:
        bm.free()


def check_pointmerge_endpoint_and_order_fallbacks():
    bm = _build_grid(2)
    try:
        previous, _ = stitch.collect_knife_path_edges_by_side(bm, AXIS, TOLERANCE)
        all_edges = [edge for bucket in previous.values() for edge in bucket]
        cache = stitch.capture_knife_path_edge_cache(bm, all_edges)
        source = all_edges[0]
        position = next(index for index, edge in enumerate(all_edges) if edge is source)
        full_bits = _record_bits(stitch.collect_knife_path_edges_by_side(bm, AXIS, TOLERANCE)[0])
        for endpoint, pointmerge, final_edges in (
            (True, False, (source,)),
            (False, True, (source,)),
        ):
            summary = _summary(
                source,
                final_edges,
                position,
                endpoint=endpoint,
                pointmerge=pointmerge,
                pre_count=len(bm.edges),
            )
            result = _fallback_collect(bm, previous, cache, summary)
            if _record_bits(result[0]) != full_bits:
                raise RuntimeError("fallback result differs from full collect")
    finally:
        bm.free()


def check_order_fallback_warning():
    bm = _build_grid(2)
    records = []
    handler = logging.Handler()
    handler.emit = lambda record: records.append(record)
    logger = logging.getLogger("ydd_symmetric_edit.stitch")
    logger.addHandler(handler)
    try:
        previous, _ = stitch.collect_knife_path_edges_by_side(bm, AXIS, TOLERANCE)
        all_edges = [edge for bucket in previous.values() for edge in bucket]
        cache = stitch.capture_knife_path_edge_cache(bm, all_edges)
        source = all_edges[0]
        source_side = next(side for side in previous if source in previous[side])
        peer = next(edge for edge in previous[source_side] if edge is not source)
        position = next(index for index, edge in enumerate(all_edges) if edge is source)
        summary = _summary(source, (source, peer), position, pre_count=len(bm.edges))
        if stitch.patch_knife_path_edges_by_side(bm, previous, cache, summary, AXIS, TOLERANCE) is not None:
            raise RuntimeError("order fallback was accepted")
        if not any("not after pre-apply edges" in record.getMessage() for record in records):
            raise RuntimeError("order fallback warning was not observed")
    finally:
        logger.removeHandler(handler)
        bm.free()


def check_face_id_closure_logs_and_falls_back():
    bm = _build_grid(2)
    records = []
    handler = logging.Handler()
    handler.emit = lambda record: records.append(record)
    logger = logging.getLogger("ydd_symmetric_edit.stitch")
    logger.addHandler(handler)
    try:
        marker = bm.edges.layers.int.get(core.EDGE_ORIGINAL_LAYER)
        face_ids = bm.faces.layers.int.get(core.FACE_ID_LAYER)
        for index, face in enumerate(bm.faces, start=1):
            face[face_ids] = index
        for edge in bm.edges:
            edge[marker] = 1
        source = next(edge for edge in bm.edges if edge.link_faces)
        source[marker] = 0
        previous, _ = stitch.collect_knife_path_edges_by_side(bm, AXIS, TOLERANCE)
        # An untracked path edge in a carrier face is a closure contradiction.
        extra = next(edge for face in source.link_faces for edge in face.edges if edge is not source)
        extra[marker] = 0
        for bucket in previous.values():
            if extra in bucket:
                bucket.remove(extra)
        all_edges = [edge for bucket in previous.values() for edge in bucket]
        cache = stitch.capture_knife_path_edge_cache(bm, all_edges)
        position = next(index for index, edge in enumerate(all_edges) if edge is source)
        summary = _summary(source, (source,), position, pre_count=len(bm.edges), pre_faces=source.link_faces)
        if stitch.patch_knife_path_edges_by_side(bm, previous, cache, summary, AXIS, TOLERANCE) is not None:
            raise RuntimeError("FACE_ID closure contradiction was accepted")
        if not any("FACE_ID closure" in record.getMessage() for record in records):
            raise RuntimeError("FACE_ID closure fallback emitted no warning")
    finally:
        logger.removeHandler(handler)
        bm.free()


def check_unexpected_topology_fallback():
    bm = _build_grid(2)
    try:
        previous, _ = stitch.collect_knife_path_edges_by_side(bm, AXIS, TOLERANCE)
        all_edges = [edge for bucket in previous.values() for edge in bucket]
        cache = stitch.capture_knife_path_edge_cache(bm, all_edges)
        source = all_edges[0]
        position = next(index for index, edge in enumerate(all_edges) if edge is source)
        for removed_edges, removed_faces in ((1, 0), (0, 1)):
            summary = _summary(
                source,
                (source,),
                position,
                pre_count=len(bm.edges),
                unexpected=True,
                removed_edges=removed_edges,
                removed_faces=removed_faces,
            )
            if stitch.patch_knife_path_edges_by_side(bm, previous, cache, summary, AXIS, TOLERANCE) is not None:
                raise RuntimeError("unexpected topology fallback was accepted")
            full, _ = stitch.collect_knife_path_edges_by_side(bm, AXIS, TOLERANCE)
            if not any(full.values()):
                raise RuntimeError("full fallback result was empty")
    finally:
        bm.free()


def run():
    checks = (
        check_track_a_record_bits_match_full_collect,
        check_producer_summary_and_patch_m2,
        check_same_face_cross_source_pure_split,
        check_producer_endpoint_and_pointmerge_flags,
        check_multi_split_m2_record_order,
        check_side_change_falls_back,
        check_producer_self_mirrored_side_fallback,
        check_pointmerge_endpoint_and_order_fallbacks,
        check_order_fallback_warning,
        check_face_id_closure_logs_and_falls_back,
        check_unexpected_topology_fallback,
    )
    for check in checks:
        check()
        print(f"PASS {check.__name__}", flush=True)
    print(MARKER, flush=True)


if __name__ == "__main__":
    try:
        run()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
