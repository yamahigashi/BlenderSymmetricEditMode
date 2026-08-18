# ydd Symmetric Edit — Known Limitations / 既知の制限

Applies to version 0.9.0. Read this before purchase.
バージョン 0.9.0 対象。購入前にお読みください。

## Prerequisites / 動作前提

- One mesh object in Edit Mode at a time; multi-object Edit Mode runs the native operators unchanged.
- Exactly one Blender Mesh Symmetry axis (X/Y/Z) enabled per operation. No axis or more than one axis runs the native operators unchanged.
- The topology must already be symmetric, and both halves must exist as real editable geometry. A half supplied only by a Mirror modifier cannot be matched.
- Symmetry is evaluated in object-local coordinates around X=0, Y=0, or Z=0, within the Match Tolerance.

- 同時に扱えるのは Edit Mode の単一メッシュのみ。Multi-Object Edit Mode ではネイティブ動作になります。
- Blender の Mesh Symmetry 軸は X/Y/Z のうち正確に 1 軸のみ有効にしてください。軸なし・複数軸ではネイティブ動作になります。
- トポロジーは事前に対称であり、両側が実在の編集可能ジオメトリである必要があります。Mirror modifier だけで補われた片側は照合できません。
- 対称性はオブジェクトローカル座標の X=0 / Y=0 / Z=0 を基準に、Match Tolerance の範囲で判定されます。

## Behavior on failure / 失敗時の挙動

- When a selected element has no visible mirrored counterpart, some operations keep the native one-sided result with a warning; use Undo, then correct the axis, tolerance, or topology.
- If the mirror-side repeat of an Inset fails, the mesh keeps the one-sided native result and a warning is reported.
- If post-processing reports an error, the confirmed native result may remain; Undo before correcting.

- 選択要素に可視のミラー相手が存在しない場合、一部の操作は警告つきでネイティブ側の片側結果を残します。Undo 後に軸・許容差・トポロジーを修正してください。
- Inset のミラー側再実行が失敗した場合、片側のネイティブ結果が残り、警告が報告されます。
- 後処理でエラーが報告された場合、確定済みのネイティブ結果が残ることがあります。修正の前に Undo してください。

## Per-tool limits / ツール別の制限

- **Inset undo is two steps**: the first undo shows the one-sided native result, the second returns to the pre-operation state. Bevel undo is a single step.
- **Rip (V / Alt+V)**: the F9 Adjust Last Operation panel is not supported. Rip also passes through to the native tool when the selection touches the symmetry plane, or when Proportional Editing or Auto Merge is enabled.
- **Extrude**: zero-offset extrudes (click or Esc without moving) keep the native result unmirrored, with a warning.
- **Extrude Manifold** is mirrored only while its result is congruent with a plain region extrude; otherwise the native result is kept with a warning. Its F9 panel adjusts only the native one-sided result.
- **Merge (M)**: the **At Cursor** entry runs the native merge unchanged; the other merge modes are mirrored.
- **Delete / Dissolve**: **Limited Dissolve** stays native (out of scope).
- **Inset / Bevel F9** stays symmetric; if the mesh can no longer be mirrored at that moment, the adjustment is neutralized with a console warning instead of producing a one-sided result.
- **Redo after an undo of a Bevel** restores the native result with both sides selected; the selection is normalized again after your next operation.
- For Inset, a selection that mixes faces crossing the symmetry plane with one-sided faces (or that already includes parts of both sides) passes through to the native operator with a warning.
- Tools that only move existing topology (e.g. Edge Slide) are left to Blender's own Mirror Editing.

- **Inset の Undo は 2 段階**です。1 回目で片側のネイティブ結果、2 回目で操作前の状態に戻ります。Bevel の Undo は 1 段階です。
- **Rip(V / Alt+V)**: F9(最後の操作を調整)パネルは非対応です。また、選択が対称面に接している場合、Proportional Editing または Auto Merge が有効な場合はネイティブ動作になります。
- **Extrude**: 移動なしの押し出し(クリックのみ・Esc)は警告つきでネイティブ結果が片側のまま残ります。
- **Extrude Manifold** は結果が通常の Region 押し出しと合同である場合のみミラーされます。それ以外は警告つきでネイティブ結果が残ります。F9 パネルは片側のネイティブ結果のみを調整します。
- **Merge(M)**: **At Cursor** はネイティブのマージのまま動作します。他のマージモードはミラーされます。
- **Delete / Dissolve**: **Limited Dissolve** はネイティブ動作です(対象外)。
- **Inset / Bevel の F9** は対称を維持します。その時点でミラー不能になった場合は片側結果を出さず、コンソール警告つきで調整を無効化します。
- **Bevel の Undo 後の Redo** は両側が選択された状態でネイティブ結果を復元します。選択は次の操作後に正規化されます。
- Inset で、対称面をまたぐ面と片側の面が混在する選択(すでに両側を部分的に含む選択も同様)は、警告つきでネイティブ動作になります。
- 既存トポロジーを移動するだけのツール(Edge Slide 等)は Blender 標準の Mirror Editing に委ねています。

## Intentionally native routes / 意図的にネイティブ動作となる経路

- Dragging the **Inset Faces / Bevel toolbar gizmo handle** does not go through the keymap and is not mirrored. (The Extrude toolbar tools' gizmo handles ARE mirrored.)
- F3 search, menus, and scripted calls to the native operators (e.g. `mesh.inset`, `mesh.bevel`, native cut/delete/dissolve operators) stay native. Use the shortcut or toolbar routes listed in the README.

- **Inset Faces / Bevel のツールバー gizmo ハンドル**のドラッグは keymap を経由しないためミラーされません(Extrude 系ツールバーツールの gizmo ハンドルはミラーされます)。
- F3 検索・メニュー・スクリプトからのネイティブ operator 直接呼出し(`mesh.inset`、`mesh.bevel`、ネイティブの cut/delete/dissolve 系など)はネイティブ動作のままです。README 記載のショートカットまたはツールバー経路を使用してください。

## Data / データ

- UVs and other CustomData are interpolated on the mirrored side and kept finite, but exact parity with every native **Correct UVs** case is not guaranteed.

- ミラー側の UV その他の CustomData は補間され有限値が維持されますが、ネイティブの **Correct UVs** の全ケースとの厳密一致は保証されません。
