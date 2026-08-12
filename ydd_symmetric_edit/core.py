# ruff: noqa: F401
# SPDX-License-Identifier: GPL-3.0-or-later

"""Geometry helpers for ydd Symmetric Edit.

The interactive cut is deliberately left to Blender's native tools. This module
marks the pre-cut topology, identifies the edges Blender created, and mirrors
their topology directly on the opposite faces.
"""

from __future__ import annotations

from mathutils.kdtree import KDTree  # noqa: F401

from ._types import (
    CarrierFrameSnapshot,
    FaceKey,
    SelectionSnapshot,
)  # noqa: F401
from .face_mapping import (
    FaceRegistry,
    _snapshot_face_map,
    resolve_live_mirror_face_map,
)  # noqa: F401
from .matching import (
    AXIS_INDEX,
    MESH_SYMMETRY_PROPERTIES,
    VertexMirrorLookup,
    VertexRegistry,
    _chebyshev_distance_3d,
    _coordinate_3d,
    _coords_match_chebyshev,
    _iter_quantized_neighborhood,
    _one_sided_candidate_arrays,
    _one_sided_pair_table,
    _quantized_coordinate,
    _solve_injective_component,
    build_vertex_mirror_lookup,
    build_vertex_pair_table,
    classify_selection_overlap,
    coordinates_match,
    enabled_mesh_symmetry_axes,
    mirror_coordinate,
)  # noqa: F401
from .selection import (
    add_selection_layers,
    extend_selection_to_mirror,
    restore_selection_for_route,
    restore_selection_scoped,
    restore_visibility_and_selection,
)  # noqa: F401
from .snapshot import (
    EDGE_HIDDEN_LAYER,
    EDGE_ORIGINAL_LAYER,
    EDGE_SELECTION_LAYER,
    FACE_HIDDEN_LAYER,
    FACE_ID_LAYER,
    FACE_MIRROR_ID_LAYER,
    FACE_SELECTION_LAYER,
    HISTORY_TOKEN_LAYER,
    TEMP_LAYER_NAMES,
    VERT_BACKUP_ID_LAYER,
    VERT_COLLAPSE_GROUP_LAYER,
    VERT_HIDDEN_LAYER,
    VERT_MERGE_GROUP_LAYER,
    VERT_RIP_ID_LAYER,
    VERT_SELECTION_LAYER,
    LazyCarrierFrameMap,
    LazyTopologyResolution,
    _capture_bmesh_snapshot,
    _capture_mesh_snapshot,
    capture_selection_snapshot,
    get_required_layers,
    prepare_topology,
    remove_temporary_layers,
    remove_temporary_mesh_attributes,
)  # noqa: F401
from .stitch import (
    _MIN_SIDE_LENGTH,
    SelectionMutationSummary,
    _assign_projection_candidates,
    _discover_path_edges,
    _edge_coordinate_key_matches,
    _edge_side,
    _is_interior_edge_factor,
    _plan_plane_stitch_vertex,
    _register_edge_endpoint_pair,
    _resolve_reflected_vertex_on_target,
    apply_collapsed_offset_topology,
    apply_crosses_p_stitch,
    apply_mirrored_path_crossings,
    apply_reflected_path_topology,
    build_reflected_cutter,
    capture_knife_path_edge_cache,
    choose_source_side,
    cluster_points_by_tolerance,
    collapsed_offset_target_edge_markers,
    collect_knife_path_edges_by_side,
    collect_source_path_edges,
    combine_selection_mutation_summaries,
    is_self_mirrored_edge,
    native_path_edge_state,
    path_ring_includes_pre_hidden_edges,
    plan_mirrored_path_crossings,
    reclassify_knife_path_edge_cache,
    reflected_path_uses_only_target_boundaries,
    reserve_source_path_marker,
    snap_projected_graph,
    target_face_ids_for_edges,
)  # noqa: F401
