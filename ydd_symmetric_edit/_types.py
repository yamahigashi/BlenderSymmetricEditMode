# SPDX-License-Identifier: GPL-3.0-or-later

"""Shared, Blender-independent structural types for the add-on.

Blender modules stay behind ``TYPE_CHECKING`` so importing this module does not
add runtime dependencies or create import cycles.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, NewType, Protocol, TypeAlias

if TYPE_CHECKING:
    import bpy
    from bpy.stub_internal.rna_enums import OperatorReturnItems
    from mathutils import Quaternion, Vector

    from .matching import VertexMirrorLookup
    from .snapshot import LazyCarrierFrameMap, LazyTopologyResolution


FaceId = NewType("FaceId", int)
EdgeMarkerId = NewType("EdgeMarkerId", int)
MirrorFaceMap: TypeAlias = dict[FaceId, FaceId]
HiddenFaceMap: TypeAlias = dict[FaceId, bool]
if TYPE_CHECKING:
    OperatorResult: TypeAlias = set[OperatorReturnItems]
else:
    OperatorResult: TypeAlias = set[str]


@dataclass(frozen=True, slots=True, order=True)
class QuantizedCoordinate:
    """A tolerance-normalized mesh coordinate used for topology matching."""

    x: int
    y: int
    z: int


@dataclass(frozen=True, slots=True, order=True)
class Coordinate3D:
    """A Blender-independent three-dimensional point."""

    x: float
    y: float
    z: float

    def as_tuple(self) -> tuple[float, float, float]:
        return self.x, self.y, self.z

    def component(self, axis_index: int) -> float:
        if axis_index == 0:
            return self.x
        if axis_index == 1:
            return self.y
        if axis_index == 2:
            return self.z
        raise IndexError(axis_index)


@dataclass(frozen=True, slots=True)
class CarrierFrameSnapshot:
    """An ordered pre-native face walk and its canonical planar frame."""

    vertices: tuple[Coordinate3D, ...]
    origin: Coordinate3D
    normal: Coordinate3D | None
    basis_u: Coordinate3D | None
    deviation: float


CarrierFrameMap: TypeAlias = "Mapping[FaceId, CarrierFrameSnapshot] | LazyCarrierFrameMap"


@dataclass(frozen=True, slots=True, order=True)
class PathEdgeSignature:
    """One canonical, rounded edge in a newly created native cut path."""

    first: Coordinate3D
    second: Coordinate3D


PathSignature: TypeAlias = tuple[PathEdgeSignature, ...]


@dataclass(frozen=True, slots=True, order=True)
class RipDupSignature:
    """One duplicated-vertex-ID group in a native Rip result."""

    vertex_id: int
    coordinates: tuple[Coordinate3D, ...]


RipSignature: TypeAlias = tuple[RipDupSignature, ...]


@dataclass(frozen=True, slots=True)
class RipVertexRecord:
    """Pre-rip facts about one vertex in the selection or its one-ring."""

    vertex_id: int
    location: Coordinate3D
    mirror_vertex_id: int | None
    face_ids: tuple[int, ...]
    selected: bool


@dataclass(frozen=True, slots=True)
class RipSnapshot:
    """Immutable pre-rip capture for the REFLECT Rip postprocess.

    Covers the selected vertices and their one-ring neighborhood; every edge
    the native Rip can duplicate has both endpoints inside this region.
    """

    axis_index: int
    tolerance: float
    vertices: tuple[RipVertexRecord, ...]

    def record_by_id(self) -> dict[int, RipVertexRecord]:
        return {record.vertex_id: record for record in self.vertices}


class MirrorOverlap(Enum):
    """How a selection S relates to its own mirror image ρ(S).

    S⁰ is the on-plane part of S; the classification deliberately ignores it
    because ρ is the identity there (ρ(S) ∩ S ⊇ S⁰ always holds).
    """

    DISJOINT = "disjoint"  # ρ(S∖S⁰) ∩ S = ∅
    SELF_MIRRORED = "self_mirrored"  # ρ(S∖S⁰) = S∖S⁰ (complete, bidirectional)
    PARTIAL = "partial"  # any other intersection


@dataclass(frozen=True)
class OverlapClassification:
    """Selection/mirror-image overlap plus the pair table it was derived from.

    ``complete`` is True when every off-plane selected vertex resolved to an
    involutive pair (no missing counterparts, no ambiguous ties).  ``pairs``
    covers the whole mesh and satisfies ``pairs[pairs[v]] == v``; on-plane
    vertices pair with themselves.
    """

    overlap: MirrorOverlap
    complete: bool
    pairs: dict[int, int]


@dataclass(frozen=True, slots=True)
class MeshSelectionMode:
    """Blender's vertex / edge / face selection-mode flags."""

    vertices: bool
    edges: bool
    faces: bool

    def as_tuple(self) -> tuple[bool, bool, bool]:
        return self.vertices, self.edges, self.faces


@dataclass(frozen=True, slots=True)
class SymmetryAxes:
    """Blender's X / Y / Z edit-mesh symmetry flags."""

    x: bool
    y: bool
    z: bool

    def as_tuple(self) -> tuple[bool, bool, bool]:
        return self.x, self.y, self.z


@dataclass(frozen=True, slots=True)
class FaceKey:
    """The geometry signature used to find a face's mirrored counterpart."""

    vertex_count: int
    coordinates: tuple[QuantizedCoordinate, ...]


@dataclass(frozen=True, slots=True)
class FaceMatchRecord:
    """Pre-cut geometry needed to locate one face's mirrored partner."""

    key: FaceKey
    mirrored_key: FaceKey
    centroid: Coordinate3D


@dataclass(slots=True)
class TopologyPreparation:
    """Topology metadata captured before the native cutting operator runs."""

    topology_resolution: LazyTopologyResolution
    hidden_by_face_id: HiddenFaceMap
    total_faces: int

    @property
    def mirror_face_ids(self) -> MirrorFaceMap:
        return self.topology_resolution.mirror_face_ids

    @property
    def carrier_frames(self) -> CarrierFrameMap:
        return self.topology_resolution.carrier_frames

    @property
    def vertex_lookup(self) -> VertexMirrorLookup:
        return self.topology_resolution.vertex_lookup

    @property
    def matched_faces(self) -> int:
        return self.topology_resolution.matched_faces


class PointerLike(Protocol):
    """Minimal interface shared by Blender RNA objects with a stable pointer."""

    def as_pointer(self) -> int: ...


class WindowContext(Protocol):
    """The portion of a Blender context needed to create a session key."""

    @property
    def window(self) -> PointerLike | None: ...


class KeymapEventLike(Protocol):
    """Event properties read from a Blender keymap item."""

    active: bool
    idname: str
    type: str
    value: str
    any: bool
    shift: int
    ctrl: int
    alt: int
    oskey: int
    key_modifier: str
    direction: str
    repeat: bool


@dataclass(frozen=True, slots=True)
class KeymapEvent:
    """The complete physical event cloned from a native Blender keymap item."""

    type: str
    value: str
    any: bool
    shift: int
    ctrl: int
    alt: int
    oskey: int
    hyper: int
    key_modifier: str
    direction: str
    repeat: bool


@dataclass(frozen=True, slots=True)
class KeymapIdentity:
    """The fields that uniquely locate a non-modal Blender keymap."""

    name: str
    space_type: str
    region_type: str


@dataclass
class ViewState:
    view_rotation: Quaternion
    view_location: Vector
    view_distance: float
    view_perspective: str
    view_camera_offset: Vector
    view_camera_zoom: float


@dataclass
class KnifeSession:
    window_pointer: int
    area_pointer: int
    region_pointer: int
    object_name: str
    mesh_name: str
    axis_index: int
    source_side: str
    tolerance: float
    mirror_face_ids: MirrorFaceMap
    hidden_by_face_id: HiddenFaceMap
    carrier_frames: CarrierFrameMap
    mesh_select_mode: MeshSelectionMode
    started_at: float
    tool_kind: str = "KNIFE"
    history_token: int = 0
    saw_modal: bool = False
    projection_view: ViewState | None = None
    symmetry_flags: SymmetryAxes = SymmetryAxes(False, False, False)
    symmetry_suspended: bool = False
    modal_absent_since: float | None = None
    path_signature: PathSignature | RipSignature | None = None
    stable_path_ticks: int = 0
    offset_use_cap_endpoint: bool = False
    native_operator_pointer: int = 0
    rip: RipSnapshot | None = None
    topology_resolution: LazyTopologyResolution | None = None


@dataclass
class HistoryRecord:
    session: KnifeSession
    status: str = "RUNNING"
    sequence: int = 0


@dataclass
class TopologyBackup:
    mesh: bpy.types.Mesh
    shape_values: dict[str, list[Vector]]


@dataclass(frozen=True, slots=True)
class VertexSelectionHistory:
    """A selected vertex's position in Blender's ordered selection history."""

    location: Coordinate3D


@dataclass(frozen=True, slots=True)
class EdgeSelectionHistory:
    """A selected edge's midpoint and optional stable pre-cut marker."""

    location: Coordinate3D
    marker: EdgeMarkerId | None


@dataclass(frozen=True, slots=True)
class FaceSelectionHistory:
    """A selected face's centroid and optional stable pre-cut identifier."""

    location: Coordinate3D
    face_id: FaceId | None


SelectionHistoryEntry: TypeAlias = VertexSelectionHistory | EdgeSelectionHistory | FaceSelectionHistory
SelectionHistory: TypeAlias = list[SelectionHistoryEntry]


@dataclass
class SelectionSnapshot:
    path_vertices_selected: bool
    path_edges_selected: bool
    path_faces_selected: bool
    history: SelectionHistory
    saved_hidden_state_present: bool | None = None


@dataclass(frozen=True, slots=True)
class NativeRoute:
    keymap_name: str
    space_type: str
    region_type: str
    is_tool: bool
    native_operator: str
    tool_kind: str
    event: KeymapEvent
    route_key: str

    @property
    def keymap_identity(self) -> KeymapIdentity:
        return KeymapIdentity(
            name=self.keymap_name,
            space_type=self.space_type,
            region_type=self.region_type,
        )


@dataclass(frozen=True, slots=True)
class KeymapFingerprint:
    """Comparable snapshot used to detect native keymap changes."""

    active_config_name: str
    routes: tuple[NativeRoute, ...]
