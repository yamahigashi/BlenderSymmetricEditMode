from __future__ import annotations

from collections.abc import Iterable

import bmesh

from . import stitch_common
from .layer_names import EDGE_ORIGINAL_LAYER
from .matching import mirror_coordinate


def collapsed_offset_target_edge_markers(
    bm: bmesh.types.BMesh,
    source_edges: Iterable[bmesh.types.BMEdge],
    axis_index: int,
    tolerance: float,
) -> tuple[set[int], str]:
    """Find original target edges for an Offset Edge Slide cancelled at zero.

    Blender's Offset macro commits its topology child before Edge Slide.  Esc
    cancels only the slide, leaving two new source loops exactly coincident with
    the selected original loop.  A coincident edge cannot be cut again, so this
    identifies the reflected original target loop for a matching BMesh
    ``offset_edgeloops`` operation.
    """

    marker_layer = bm.edges.layers.int.get(EDGE_ORIGINAL_LAYER)
    if marker_layer is None:
        return set(), "edge marker layer is missing"

    originals_by_endpoint: stitch_common._EdgeEndpointStore = {}
    new_edges_by_endpoint: stitch_common._EdgeEndpointStore = {}
    for edge in bm.edges:
        marker = int(edge[marker_layer])
        if marker <= 0:
            stitch_common._register_edge_endpoint_pair(
                new_edges_by_endpoint,
                edge.verts[0].co,
                edge.verts[1].co,
                tolerance,
            )
        else:
            stitch_common._register_edge_endpoint_pair(
                originals_by_endpoint,
                edge.verts[0].co,
                edge.verts[1].co,
                tolerance,
                marker=marker,
            )

    target_markers = set()
    matched_nonzero_segments = 0
    for edge in source_edges:
        reflected_a = mirror_coordinate(edge.verts[0].co, axis_index)
        reflected_b = mirror_coordinate(edge.verts[1].co, axis_index)
        if (reflected_a - reflected_b).length <= tolerance:
            # Endpoint-cap output can collapse to a point at factor zero.  The
            # target BMesh op will recreate it from the non-degenerate loop.
            # (Intentionally Euclidean: an edge-length degeneracy test, not a
            # coordinate-identity test.)
            continue
        if stitch_common._edge_coordinate_key_matches(reflected_a, reflected_b, tolerance, new_edges_by_endpoint):
            return set(), "the target already contains native zero-offset topology"
        marker = stitch_common._edge_keys_matching_lookup(
            reflected_a,
            reflected_b,
            tolerance,
            originals_by_endpoint,
        )
        if marker is None:
            return set(), "a reflected zero-offset segment has no original target edge"
        target_markers.add(marker)
        matched_nonzero_segments += 1

    if not target_markers or not matched_nonzero_segments:
        return set(), "no reflected original target loop was found"
    return target_markers, ""


def apply_collapsed_offset_topology(
    bm: bmesh.types.BMesh,
    target_edge_markers: set[int],
    *,
    use_cap_endpoint: bool,
) -> tuple[int, str]:
    """Create the target-side topology for a zero-factor Offset operation."""

    marker_layer = bm.edges.layers.int.get(EDGE_ORIGINAL_LAYER)
    if marker_layer is None:
        return 0, "edge marker layer is missing"
    target_edges = [edge for edge in bm.edges if int(edge[marker_layer]) in target_edge_markers]
    if len(target_edges) != len(target_edge_markers):
        return 0, "one or more target loop edges were lost"

    result = bmesh.ops.offset_edgeloops(
        bm,
        edges=target_edges,
        use_cap_endpoint=use_cap_endpoint,
    )
    output_edges = list(result.get("edges", ()))
    if not output_edges:
        return 0, "Blender did not create the target offset topology"
    for edge in output_edges:
        edge.select = False
    bm.normal_update()
    return len(output_edges), ""
