# ydd Symmetric Edit エンドユーザー使用許諾契約兼購入・サポート条件
## GPL補足規約／End User License Agreement and Purchase & Support Terms — GPL Supplement

**版:** Draft 1.0  
**作成日:** 2026年8月18日  
**対象製品:** ydd Symmetric Edit 0.9.0以降の、販売ページで本規約が適用される版  
**販売者:** `[販売者の正式氏名・商号 / Seller Legal Name]`  
**所在地:** `[所在地 / Address]`  
**連絡先:** `[サポート用メールまたはURL / Support Email or URL]`  
**発効日:** `[発効日 / Effective Date]`

> **導入前の重要事項:** 角括弧の項目を確定情報で置換してください。本規約は購入前に商品ページ、チェックアウト画面、または販売サイトのEULA提示機能で閲覧可能にし、購入者の同意を取得してください。購入後にファイルへ同梱するだけでは、すべての法域で契約成立を保証できません。

---

# 日本語版

## 第1条（定義）

1. 「本製品」とは、販売ページで `ydd Symmetric Edit` として提供されるBlender拡張、付属文書、更新物および販売者が明示した関連ファイルをいいます。
2. 「本プログラム」とは、本製品のうちGNU General Public License version 3またはそれ以降（以下「GPL-3.0-or-later」）の表示があるソースコードおよび実行可能形式をいいます。
3. 「購入者」とは、BOOTH、Gumroad、Superhiveその他販売者が承認した販売経路を通じて本製品を取得した個人または法人をいいます。
4. 「販売サイト」とは、本製品の注文、決済、配布または返金手続を仲介・再販売するオンラインサービスをいいます。
5. 「商品文書」とは、購入時の商品ページ、インストール手順、対応環境、既知制限、変更履歴、FAQおよびサポート条件をいいます。

## 第2条（適用・同意・販売サイト規約）

1. 本規約は、購入前に購入者へ提示され、購入者が購入、チェックボックス、ダウンロードその他適用法上有効な方法で同意した場合に、販売者と購入者との間に適用されます。
2. 販売サイトの利用規約、返金ポリシー、決済条件および強行法規は、それぞれの適用範囲で本規約に優先します。
3. 本規約は販売サイトと購入者との契約を変更せず、販売サイトを本規約の当事者にしません。ただし、販売サイトがmerchant of recordまたは再販売者として負う役割は、その販売サイトの規約に従います。

## 第3条（GPLライセンスの優先）

1. 本プログラムはGPL-3.0-or-laterに基づき許諾されます。完全なGPL本文は本プログラムに同梱されます。
2. 購入者は、GPLに従う限り、本プログラムを目的を問わず実行、研究、改変、複製、再配布および商用利用できます。
3. 本規約は、GPLが認める権利へ追加的制限を課しません。本規約とGPLが抵触する場合、本プログラムについてはGPLが優先します。
4. 本規約の「購入」「公式配布」「サポート」「更新」「返金」等の条件は、GPLに基づく本プログラムの権利と区別されます。
5. 購入者が本プログラムを再配布しても、販売者は再配布先に対し、サポート、更新、返金、個別対応その他の有償サービスを提供する義務を負いません。

## 第4条（購入により提供されるもの）

1. 購入者は、販売サイトおよび商品文書に記載された範囲で、公式配布物へのアクセス、購入時点の本製品、ならびに販売者が明示的に約束したサポートまたは更新を受けます。
2. 購入は、本プログラムについて独占権、再配布禁止権、席数制限権または第三者によるGPL利用を阻止する権利を購入者に与えません。
3. 公式サポートおよび販売サイト上の更新アクセスは、購入に使用したアカウントまたは正当な購入証明に紐づけることができます。この制限は、本プログラム自体の複製・再配布を制限しません。

## 第5条（著作権・商標・非コード資料）

1. 本プログラムの著作権は、各ファイルの表示および同梱ライセンスに従い、販売者または各権利者に帰属します。
2. 本製品に同梱された文書は、当該ファイルに別のライセンス表示がある場合はその表示に従い、表示がない場合は本プログラムとともにGPL-3.0-or-laterで提供されるものとして扱います。
3. 商品ページの画像、動画、ロゴ、販売用コピー、デモ映像その他本プログラムに同梱されない販促資料は、別途明示されない限り販売者または権利者に留保されます。
4. GPLは販売者の商号、製品名、ロゴまたは商標の使用権を当然には付与しません。購入者は、互換性・由来を正確に説明するための合理的な表示を除き、販売者による承認・提携を誤認させる使用をしてはなりません。

## 第6条（対応環境・機能・既知制限）

1. 対応Blender版、OS、インストール方法およびサポート対象は、購入時の商品文書に明示された内容に限ります。最小要件はBlender 4.2ですが、すべての将来版との互換性を保証するものではありません。
2. 本製品は、原則として単一メッシュのEdit Mode、X/Y/Zのうち正確に1つのMesh Symmetry軸、および既に対称な編集対象トポロジーを前提とします。
3. 購入者は、次の重要な制限を理解します。
   - 相手側トポロジーが欠ける場合、非対応経路または後処理失敗時に、ネイティブ側の片側結果が残ることがあること。
   - RipのAdjust Last Operation（F9）は非対応であること。
   - InsetのUndoは2段階となること。
   - Extrude Manifoldその他一部のF9、直接operator呼出し、toolbar gizmo経路は片側またはネイティブ動作となる場合があること。
   - 軸なし、複数軸、Multi-Object Edit Modeではネイティブ動作へパススルーすること。
   - UVその他CustomDataについて、すべてのネイティブCorrect UVs結果との厳密一致を保証しないこと。
4. 商品文書に記載した制限と実際の動作が重要な点で異なる場合、第12条および第13条が適用されます。

## 第7条（購入者の責任）

1. 購入者は、本製品を重要なデータへ適用する前に、`.blend` ファイルおよび関連資産のバックアップを作成し、コピー上でテストします。
2. 購入者は、対象メッシュの対称性、軸、Match Tolerance、選択、hidden要素、Auto Merge、Proportional Editingその他の状態を確認します。
3. 警告、片側結果または予期しない結果が生じた場合、購入者は作業を継続する前にUndoし、データを検証します。
4. 改変版、第三者再配布版、非対応Blender版、他アドオンとの競合または販売者が再現できない環境について、販売者はサポートを断ることができます。ただしGPL上の権利は制限されません。

## 第8条（本製品のローカル動作・プライバシー）

1. バージョン0.9.0の確認時点で、本製品はテレメトリを送信せず、ライセンスサーバまたは販売者のサーバへ接続しません。
2. 試用版は、試用開始日をBlenderのアドオン設定へローカル保存します。Blenderの設定自動保存が有効な場合、本製品はその設定保存処理を呼び出すことがあります。
3. 本製品は機能提供のため、Blender内でkeymap、handler、timer、Scene設定および一時的なメッシュ属性を登録・使用する場合があります。
4. 販売サイトによるアカウント、決済、税務、ダウンロード履歴等の処理は、各販売サイトのプライバシーポリシーに従います。
5. 将来版で販売者へのデータ送信を追加する場合、販売者は更新前に明確なプライバシー通知を提供し、適用法上必要な同意を取得します。

## 第9条（試用版）

1. 試用版が提供される場合、標準の試用表示期間は、商品文書に別段の記載がない限り、ローカルに記録された開始日から14日間です。
2. バージョン0.9.0の試用版は、期間終了後に購入案内を表示しますが、技術的な機能停止を保証するものではありません。
3. 試用版の本プログラムもGPL-3.0-or-laterであり、試用期間はGPL上の実行、研究、改変、複製または再配布権を制限しません。
4. 試用版から製品版への更新方法は商品文書に従います。設定保持または自動置換は、明示的に保証された場合を除き保証されません。

## 第10条（更新・サポート）

1. 無償更新の期間、対象バージョン、サポート方法および回答目標は、購入時の商品文書に記載された内容に限ります。記載がない限り、永続的な更新・サポートは約束されません。
2. 推奨する初回回答目標は3営業日以内、または販売サイトが定めるより短い期間です。これは解決期限の保証ではありません。
3. 販売者は、不具合報告に再現手順、Blender版、OS、サンプルファイル、コンソール出力その他合理的な情報を求めることができます。
4. GPLに従い再配布されたコピーまたは改変版に対するサポートは、別途合意がない限り対象外です。
5. Blender、OS、販売サイトその他第三者サービスの変更により保守が著しく困難になった場合、販売者は将来のサポート対象を変更できます。ただし、既に明示した有償期間および強行法規上の義務を不当に害しません。

## 第11条（第三者コンポーネント）

本製品が第三者コンポーネントを含む場合、そのコンポーネントには各権利者のライセンスおよび通知が適用されます。本規約は第三者ライセンスが付与する権利を縮減しません。

## 第12条（限定保証）

1. 販売者は、購入時点において、(a) 本製品を表示されたライセンスで提供するために必要な権利を有すること、(b) 未改変の公式配布物が商品文書の重要な説明に実質的に適合することを保証します。
2. 前項および強行法規上の保証を除き、本製品は現状有姿で提供され、販売者は、無停止、無誤謬、すべてのメッシュ・Blender版・OS・アドオンとの互換性、特定目的適合性または結果の完全対称性を保証しません。
3. 本条は、販売者の故意または重過失、生命・身体への損害、詐欺、不実表示、契約不適合その他法令上排除できない責任を免除しません。

## 第13条（不具合・返金・救済）

1. 購入者は、重大な不具合または商品説明との重要な相違を発見した場合、販売サイトの指定手続およびサポート窓口を通じ、合理的な再現情報を添えて通知します。
2. 適用法および販売サイトのポリシーに従い、販売者は合理的な期間内に、修正版、回避策、再ダウンロード、代替物、サポートまたは返金承認のいずれか適切な救済を提供します。
3. 購入者都合、購入前に明示された非対応環境、購入前に明示された既知制限、または購入者の改変・誤使用のみを理由とする裁量的返金は、販売サイトのポリシーまたは販売者の明示的な返金方針が認める場合を除き保証されません。
4. 「返金不可」の表示がある場合でも、商品説明との重大な相違、契約不適合、販売者の故意・重過失、または消費者保護法その他強行法規に基づく権利は制限されません。
5. 決済の取消し、チャージバックまたは返金により、公式サポート・将来の更新アクセスを終了できる場合があります。ただし、購入者が既に適法に受領したGPLプログラムの権利は、GPLに従います。

## 第14条（責任制限）

1. 法令で認められる最大限の範囲で、販売者は、本製品の使用または使用不能から生じる間接損害、特別損害、付随損害、結果損害、逸失利益、データ消失または業務中断について責任を負いません。
2. 販売者の通常の過失に基づく損害賠償責任の総額は、当該請求の対象となる本製品について購入者が実際に支払った額を上限とします。
3. 前二項は、販売者の故意または重過失、生命・身体への損害、詐欺、不実表示、消費者契約法その他法令上制限できない責任には適用されません。
4. 一部の法域が前各項の制限を認めない場合、その法域で許される最小限の制限へ読み替えます。

## 第15条（GPL上の権利とサポート資格の終了）

1. 本プログラムに関するライセンスの終了・回復はGPLの定めに従い、本規約のみを理由としてGPL権利を追加的に終了させません。
2. 販売者は、詐欺的購入、サポート窓口への濫用、販売サイト規約違反、未払、返金またはチャージバックがある場合、法令と販売サイト規約の範囲で、公式サポート、非公開更新チャネル、購入者専用サービスへのアクセスを停止できます。
3. 前項は、既に受領した本プログラムのGPL上の権利を遡及的に取り消しません。

## 第16条（販売サイト別条件）

1. **BOOTH:** 注文・配布・キャンセル・返金の可否および手続はBOOTHの当時有効な規約・ヘルプに従います。BOOTHがダウンロード商品の返金処理を提供しない場合でも、第12条から第14条および強行法規上必要な救済は、修正、代替配布その他利用可能な適法手段により処理されます。
2. **Gumroad:** Gumroadがmerchant of recordまたは再販売者として取り扱う取引では、決済、税、一次返金、チャージバック等はGumroad規約に従います。本製品のライセンスは、Gumroadを通じて販売者から購入者へ付与されるものとして扱います。
3. **Superhive:** Superhiveでの購入には、当時有効なSuperhiveのGPL、商品文書、サポートおよび返金ポリシーが適用され、本規約より購入者に有利な強制的プラットフォーム条件は優先します。
4. その他の販売サイトでは、同様に、その販売サイトの強制的な決済・返金・消費者保護条件が優先します。

## 第17条（消費者の権利）

1. 購入者が消費者である場合、本規約は居住地その他適用される消費者保護法上の強行的権利を制限しません。
2. 通信販売にクーリング・オフ制度が適用されない場合でも、販売者は購入前に返品・返金の可否、期間、条件、費用負担およびソフトウェアの動作環境を明確に表示します。
3. 本規約の条項が消費者契約法その他適用法により無効となる場合、その条項は無効となる範囲に限り適用されず、残りは存続します。

## 第18条（準拠法・紛争解決）

1. 強行法規に反しない範囲で、本規約は日本法に準拠します。
2. 購入者が事業者である場合、本規約に関する第一審の専属的合意管轄裁判所を東京地方裁判所とします。
3. 購入者が消費者である場合、購入者の居住地法、法定管轄、少額訴訟、行政・裁判外紛争解決その他放棄できない権利を妨げません。
4. 当事者は、訴訟前に、サポート窓口および販売サイトの紛争手続を通じて誠実に解決を試みます。

## 第19条（言語）

日本語版と英語版は同一の意味を意図します。矛盾がある場合は日本語版を優先します。ただし、販売サイトの強制条件、購入者に適用される強行法規または消費者保護法が別の結果を要求する場合はこの限りではありません。

## 第20条（完全合意・変更・分離可能性・連絡先）

1. 本規約、GPL、購入時の商品文書および適用される販売サイト規約は、それぞれの対象事項について当事者間の合意を構成します。
2. 販売者は将来の販売または更新について本規約を変更できます。変更は、購入者へ事前に明確に通知し、適用法上必要な同意を得た範囲で適用されます。変更により既に付与されたGPL権利を縮減しません。
3. いずれかの条項が無効または執行不能でも、残りの条項は存続します。
4. 通知およびサポート連絡先は、本書冒頭または購入時の商品文書に記載された窓口とします。

## 別紙A（製品固有情報）

- 製品名: `ydd Symmetric Edit`
- 対象版: `0.9.0`（販売ページで更新された場合はその版）
- 製品ID: `ydd_symmetric_edit`
- コードライセンス: GNU GPL v3.0 or later
- 最小Blender: 4.2
- 標準試用表示期間: 14日（試用版を提供する場合）
- サポート期間: `[例: 購入日から12か月／当該メジャー版のサポート終了まで]`
- サポート対象Blender: `[販売ページに列挙]`
- 初回回答目標: `[例: 3営業日以内。Superhiveでは72時間以内]`
- 返金方針: `[各販売サイトの設定と一致させる]`

---

# English Version

## 1. Definitions

1. “Product” means the Blender extension offered as `ydd Symmetric Edit`, together with documentation, updates, and related files expressly identified by the Seller on the product page.
2. “Program” means the source code and executable forms within the Product that are marked as licensed under the GNU General Public License version 3 or any later version (“GPL-3.0-or-later”).
3. “Purchaser” means an individual or entity that obtains the Product through BOOTH, Gumroad, Superhive, or another sales channel approved by the Seller.
4. “Platform” means an online service that intermediates or resells the ordering, payment, delivery, or refund of the Product.
5. “Product Documentation” means the product page, installation instructions, compatibility information, known limitations, change log, FAQ, and support terms available at the time of purchase.

## 2. Scope, Acceptance, and Platform Terms

1. These Terms apply between the Seller and Purchaser only when they were presented before purchase and accepted through purchase, a checkbox, download, or another method valid under applicable law.
2. The applicable Platform terms, refund policy, payment terms, and mandatory law prevail within their respective scope.
3. These Terms do not amend the contract between a Platform and Purchaser and do not make the Platform a party to these Terms. A Platform’s role as merchant of record or reseller remains governed by that Platform’s terms.

## 3. GPL License Controls

1. The Program is licensed under GPL-3.0-or-later. The complete GPL text accompanies the Program.
2. Subject to the GPL, Purchaser may run, study, modify, copy, redistribute, and commercially use the Program for any purpose.
3. These Terms impose no additional restriction on rights granted by the GPL. If these Terms conflict with the GPL, the GPL controls with respect to the Program.
4. Provisions concerning purchase, official delivery, support, updates, and refunds are separate from the GPL rights in the Program.
5. Redistribution of the Program does not require the Seller to provide support, updates, refunds, or other paid services to recipients of redistributed copies.

## 4. What the Purchase Provides

1. Subject to the Platform and Product Documentation, Purchaser receives access to the official delivery, the Product as offered at purchase, and any support or updates expressly promised by the Seller.
2. Purchase does not grant exclusivity, a right to prohibit redistribution, a seat-restriction right, or a right to prevent third parties from exercising GPL rights.
3. Official support and update access through a Platform may be tied to the purchasing account or valid proof of purchase. This does not restrict copying or redistribution of the Program itself.

## 5. Copyright, Trademarks, and Non-Code Materials

1. Copyright in the Program remains with the Seller or other rights holders identified in the files and accompanying notices.
2. Documentation included in the Product is governed by any license stated in that file. If no separate notice is stated, it will be treated as distributed with the Program under GPL-3.0-or-later.
3. Store images, videos, logos, sales copy, demonstrations, and promotional materials not included in the Program are reserved to the Seller or their rights holders unless expressly licensed otherwise.
4. The GPL does not automatically grant rights in the Seller’s name, Product name, logo, or trademarks. Purchaser may make accurate compatibility or provenance references, but may not imply endorsement or affiliation without permission.

## 6. Compatibility, Functionality, and Known Limitations

1. Supported Blender versions, operating systems, installation methods, and support scope are limited to those expressly stated in the Product Documentation at purchase. The minimum requirement is Blender 4.2, but compatibility with every future version is not warranted.
2. The Product generally requires one mesh in Edit Mode, exactly one X/Y/Z Mesh Symmetry axis, and already symmetric editable topology.
3. Purchaser acknowledges the following material limitations:
   - Missing counterpart topology, unsupported routes, or post-processing failure may leave the native one-sided result.
   - Adjust Last Operation (F9) is not supported for Rip.
   - Inset undo uses two steps.
   - Extrude Manifold and certain other F9, direct-operator, and toolbar-gizmo routes may remain one-sided or native.
   - No axis, multiple axes, or Multi-Object Edit Mode may pass through to native behavior.
   - Exact parity with every native Correct UVs result is not warranted for UV and other CustomData.
4. If the actual behavior materially differs from the Product Documentation, Sections 12 and 13 apply.

## 7. Purchaser Responsibilities

1. Purchaser will back up `.blend` files and related assets and test the Product on copies before applying it to important data.
2. Purchaser will check mesh symmetry, axis, Match Tolerance, selection, hidden elements, Auto Merge, Proportional Editing, and other relevant state.
3. If a warning, one-sided result, or unexpected result occurs, Purchaser will Undo and inspect the data before continuing.
4. Seller may decline support for modified builds, third-party redistributed builds, unsupported Blender versions, add-on conflicts, or environments the Seller cannot reproduce. GPL rights remain unaffected.

## 8. Local Operation and Privacy

1. As reviewed in version 0.9.0, the Product sends no telemetry and does not contact a license server or Seller server.
2. The trial build stores the trial start date locally in Blender add-on preferences. If Blender’s automatic preference saving is enabled, the Product may invoke the preference-save operation.
3. To provide its functions, the Product may register or use Blender keymaps, handlers, timers, Scene settings, and temporary mesh attributes.
4. Platform processing of accounts, payments, tax, and download history is governed by each Platform’s privacy policy.
5. If a future version introduces data transmission to the Seller, the Seller will provide clear notice before the update and obtain consent where required by law.

## 9. Trial Build

1. Where a trial build is offered, the standard display period is fourteen days from the locally recorded start date unless the Product Documentation states otherwise.
2. In version 0.9.0, the trial build displays a purchase notice after the period ends but does not warrant technical feature lockout.
3. The Program in the trial build is also GPL-3.0-or-later. The trial period does not restrict GPL rights to run, study, modify, copy, or redistribute it.
4. Trial-to-full upgrade instructions are governed by the Product Documentation. Preference retention or automatic replacement is not warranted unless expressly stated.

## 10. Updates and Support

1. Free update duration, covered versions, support methods, and response targets are limited to what the Product Documentation states at purchase. Perpetual updates or support are not promised unless expressly stated.
2. The recommended initial response target is within three business days, or any shorter period required by the Platform. This is not a guaranteed resolution time.
3. Seller may request reasonable information such as reproduction steps, Blender version, operating system, sample file, and console output.
4. Support for redistributed or modified copies is excluded unless separately agreed, without affecting GPL rights.
5. Seller may change future support scope where changes to Blender, operating systems, Platforms, or other third-party services make maintenance materially difficult, but will not improperly impair an expressly paid support period or mandatory legal obligations.

## 11. Third-Party Components

If the Product contains third-party components, their own licenses and notices apply. These Terms do not reduce rights granted by a third-party license.

## 12. Limited Warranty

1. At purchase, Seller warrants that (a) Seller has the rights necessary to provide the Product under the stated licenses, and (b) the unmodified official delivery materially conforms to the material descriptions in the Product Documentation.
2. Except for the preceding warranty and rights that cannot be excluded by law, the Product is provided “as is.” Seller does not warrant uninterrupted or error-free operation, compatibility with every mesh, Blender version, operating system, or add-on, fitness for a particular purpose, or perfectly symmetric results in every case.
3. Nothing in this Section excludes liability for Seller’s intentional misconduct or gross negligence, death or personal injury, fraud, misrepresentation, non-conformity, or other liability that cannot lawfully be excluded.

## 13. Defects, Refunds, and Remedies

1. Purchaser will report a material defect or material mismatch with the description through the Platform’s designated process and the support contact, with reasonable reproduction information.
2. Subject to applicable law and Platform policy, Seller will provide an appropriate remedy within a reasonable period, which may be a fix, workaround, re-download, replacement, support, or authorization of a refund.
3. Discretionary refunds for change of mind, a pre-disclosed unsupported environment, a pre-disclosed known limitation, or Purchaser modification or misuse are not promised unless allowed by Platform policy or Seller’s express refund policy.
4. Any “no refund” statement does not limit rights arising from a material mismatch, non-conformity, Seller’s intentional misconduct or gross negligence, or mandatory consumer law.
5. A payment reversal, chargeback, or refund may end official support or future update access. Rights in a lawfully received GPL Program remain governed by the GPL.

## 14. Limitation of Liability

1. To the maximum extent permitted by law, Seller is not liable for indirect, special, incidental, or consequential damages, lost profit, lost data, or business interruption arising from use or inability to use the Product.
2. Seller’s aggregate liability for ordinary negligence is limited to the amount Purchaser actually paid for the Product giving rise to the claim.
3. The preceding limitations do not apply to Seller’s intentional misconduct or gross negligence, death or personal injury, fraud, misrepresentation, or liability that cannot be limited under consumer or other mandatory law.
4. Where a jurisdiction does not allow a limitation above, it is reduced to the minimum limitation permitted in that jurisdiction.

## 15. GPL Rights and Ending Support Eligibility

1. Termination and reinstatement of the Program license are governed by the GPL. These Terms do not create an additional basis to terminate GPL rights.
2. Subject to law and Platform terms, Seller may suspend official support, private update channels, or purchaser-only services for fraudulent acquisition, abusive support conduct, Platform violations, nonpayment, refund, or chargeback.
3. Such suspension does not retroactively revoke GPL rights in a copy already received.

## 16. Platform-Specific Terms

1. **BOOTH:** Ordering, delivery, cancellation, and refund availability follow BOOTH’s then-current terms and help documentation. If BOOTH does not provide refund processing for downloadable products, remedies required under Sections 12–14 or mandatory law may be handled through a fix, replacement delivery, or another lawful method that is available.
2. **Gumroad:** Where Gumroad acts as merchant of record or reseller, payment, tax, first-tier refunds, and chargebacks follow Gumroad’s terms. The Product license is treated as granted by Seller to Purchaser through Gumroad.
3. **Superhive:** Purchases through Superhive are subject to its then-current GPL, documentation, support, and refund policies. A mandatory Platform term more favorable to Purchaser prevails over these Terms.
4. The same principle applies to another Platform’s mandatory payment, refund, and consumer-protection terms.

## 17. Consumer Rights

1. If Purchaser is a consumer, these Terms do not limit mandatory consumer rights under the law of Purchaser’s residence or another applicable law.
2. Even where statutory cooling-off does not apply to distance sales, Seller will clearly disclose before purchase the availability, period, conditions, and cost allocation for returns or refunds and the software operating environment.
3. If a provision is invalid under the Consumer Contract Act or another applicable law, it is inapplicable only to the invalid extent; the remainder survives.

## 18. Governing Law and Disputes

1. To the extent not prohibited by mandatory law, these Terms are governed by the laws of Japan.
2. If Purchaser is a business, the Tokyo District Court has exclusive jurisdiction as the court of first instance for disputes concerning these Terms.
3. If Purchaser is a consumer, this clause does not impair mandatory home-state law, statutory venue, small-claims rights, administrative remedies, or alternative dispute-resolution rights.
4. Before litigation, the parties will attempt in good faith to resolve the matter through the support contact and the Platform’s dispute process.

## 19. Language

The Japanese and English versions are intended to have the same meaning. If they conflict, the Japanese version controls, except where mandatory Platform terms, mandatory law, or consumer protection requires another result.

## 20. Entire Agreement, Changes, Severability, and Contact

1. These Terms, the GPL, the Product Documentation at purchase, and applicable Platform terms form the agreement for their respective subject matter.
2. Seller may change these Terms for future sales or updates. A change applies only after clear advance notice and any consent required by law. No change reduces GPL rights already granted.
3. If a provision is invalid or unenforceable, the remaining provisions survive.
4. Notices and support communications will be sent to the contact stated at the beginning of these Terms or in the Product Documentation at purchase.

## Schedule A — Product-Specific Information

- Product: `ydd Symmetric Edit`
- Covered version: `0.9.0` (or an updated version identified on the product page)
- Product ID: `ydd_symmetric_edit`
- Code license: GNU GPL v3.0 or later
- Minimum Blender version: 4.2
- Standard trial display period: 14 days, if a trial is offered
- Support period: `[e.g., 12 months from purchase / until support ends for the applicable major version]`
- Supported Blender versions: `[list on product page]`
- Initial response target: `[e.g., within 3 business days; within 72 hours on Superhive]`
- Refund policy: `[must match each Platform setting and product page]`

---

# 導入メモ（契約本文ではありません）／Implementation Notes — Not Part of the Agreement

1. `[販売者情報]`、サポート期間、対応Blender、返金方針を埋めてください。
2. 商品ページで購入前にEULAを表示し、GPLであることと重要な既知制限を要約してください。
3. 本製品の全Pythonファイルへ著作権表示とGPL通知を入れ、完全な`LICENSE`を同梱してください。
4. BOOTHでは商品説明に特定商取引法上の必要表示、動作環境、返品特約を明示してください。
5. GumroadではEULAおよびProduct DocumentationをGumroadが購入前に提示できる形で登録し、custom refund policyと一致させてください。
6. SuperhiveではGPLを選択し、英語のInstallation、Requirements、Usage、Known Limits、FAQ、Supportを商品ページ内へ直接記載してください。
7. 販売開始直前に各サイトの規約・返金ポリシーを再確認してください。オンライン規約は変更されます。
8. 日本法・販売対象国の消費者法・税務・事業者表示について専門家の確認を受けてください。
