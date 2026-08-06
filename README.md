# BlenderSymmetricEditMode

**ydd Symmetric Edit** makes Blender's own editing tools symmetry-aware, without a
Mirror modifier. It covers two families of native operations on an already
symmetrical mesh:

- **Cut tools** — **Knife**, **Loop Cut**, and **Offset Edge Loop Cut**. The
  native modal interaction stays untouched; once the operation is confirmed,
  the new topology is reproduced on the opposite side.
- **Topology edits** — **Vertex Connect (J)** and **Merge Vertices (M)**. The
  operation runs on the selected side and is applied to the mirrored vertices
  in the same step.

In both cases the native tool remains the one you are using: shortcuts, modal
controls, operator options, and the Adjust Last Operation (F9) panel all stay
native.

## Install

In Blender 4.2 or later, open **Edit > Preferences > Add-ons**, choose
**Install from Disk**, select the release ZIP, then enable
**ydd Symmetric Edit**.

## Use

1. The mesh must already have matching topology around its object-local origin.
2. In Edit Mode, enable exactly one of Blender's own **Mesh Symmetry X / Y / Z**
   settings in the 3D Viewport header or Tool settings.
3. Open **N panel > Edit > ydd Symmetric Edit** and enable **Enable for Native
   Cut Tools**.
4. Work on one side of the mesh with the supported tools and confirm normally.
5. The toggle persists across sessions.

## Cut tools (Knife / Loop Cut / Offset Edge Loop Cut)

The add-on does not watch fixed physical keys. It finds the active key
configuration's `mesh.knife_tool`, `mesh.loopcut_slide`, and
`mesh.offset_edge_loops_slide` routes — including custom and Industry
Compatible keymaps — and places a pass-through preparation hook ahead of them.
The same physical event still invokes Blender's operator, so modal controls,
toolbar first-click behavior, the active Workspace Tool, selection, and the
Adjust Last Operation target all behave as usual. The mirrored topology is
created right after the native confirmation.

Loop Cut and Offset results are rebuilt directly on the paired target edges and
faces with BMesh, so curved or self-occluding surfaces do not depend on the
viewport's screen-space projection. Knife paths whose vertices all resolve to
paired boundary edges use the same direct BMesh rebuild; Knife graphs with
intentional interior waypoints or intersections use a Knife Project fallback.
During Offset Edge Loop Cut only, the Mesh Symmetry flags are suspended for the
duration of the native modal operation and restored on every exit path, so its
internal Edge Slide cannot move the opposite half prematurely.

Undo and Redo treat each confirmed native-plus-mirrored result as one
operation. Loop Cut and Offset changes made through **Adjust Last Operation
(F9)** receive the same mirrored post-process; the redo panel itself remains
the native operator.

## Vertex Connect and Merge (J / M)

With the add-on enabled and one symmetry axis active:

- **J** connects the selected vertex path and the mirrored path in one step.
- **M** opens a Merge menu whose **At Center**, **Collapse**, **At First**,
  **At Last**, and **By Distance** entries also merge the mirrored vertex
  clusters. **At Cursor** runs the native merge unchanged.

Both are single undo steps and support the redo panel (for example, adjusting
the By Distance threshold re-applies symmetrically). Vertices on the symmetry
plane are treated as shared by both sides: when a merge lands on the plane, the
mirrored cluster is welded into the same surviving vertex, keeping the mesh
connected.

The mirror step is skipped — with a report, never silently — when it cannot
apply cleanly:

- A selection that already spans both sides runs the native operation once
  (the selection is symmetrized first for By Distance).
- Vertices without a mirrored counterpart are merged natively; Connect requires
  the full path to have counterparts and otherwise connects the source side
  only.
- Multi-object Edit Mode, no symmetry axis, or multiple axes: the native
  operator runs unchanged.

Exact coordinate symmetry of **By Distance** results follows the native
`Merge by Distance` behavior of the running Blender version; pairs straddling
the plane merge once, and mismatches stay within the merge distance.

## Edge Slide (GG)

Standalone **Edge Slide (GG)** is not intercepted: it moves existing topology,
which Blender's own **Mirror Editing** option already handles. This add-on
covers the operations that create or remove topology, which has no native
mirrored equivalent.

## Direct operator calls

Menu, F3, and scripted calls to the native cut operators bypass the keymap
hook, so they are not mirrored; use a shortcut or toolbar route instead.
Connect and Merge are mirrored through their J / M bindings and the M menu.

## Scope and data

- One mesh object in Edit Mode at a time.
- Exactly one Blender Mesh Symmetry axis enabled per operation.
- Both halves must already exist as editable geometry. A missing half supplied
  only by a Mirror modifier cannot be matched.
- Symmetry is evaluated in object-local coordinates around X=0, Y=0, or Z=0.
- Cut targets on the opposite side must have an exact mirrored counterpart
  within **Match Tolerance** before the native operation starts; Connect and
  Merge match individual vertices with the same tolerance.
- Knife strokes should remain on one side. Cross-plane segments are skipped;
  endpoints may lie on the plane.
- UVs and other CustomData are interpolated on the target side and kept finite,
  but exact parity with every native **Correct UVs** case is not guaranteed.

If post-processing reports an error, the confirmed source result may remain;
use Undo before correcting the axis, tolerance, or source topology.

## Development

The development baseline is Python 3.11, matching Blender 4.2 LTS. The add-on
package lives in `ydd_symmetric_edit/`; Blender-driven tests live in `tests/`.
Ruff handles linting, import sorting, and formatting; ty checks the package
using Blender 4.2 API stubs. From the repository root:

```powershell
uv sync --group dev
uv run ruff check
uv run ruff format --check
uv run ty check
```

Shared semantic types and structural protocols live in
`ydd_symmetric_edit/_types.py`; the test procedure, pass criteria, and release
packaging steps are documented in `docs/testing.md`.

## License

ydd Symmetric Edit is licensed under GPL-3.0-or-later. See `LICENSE` for the
full license text.
