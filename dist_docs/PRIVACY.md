# ydd Symmetric Edit — Privacy Notes / プライバシーについて

Applies to version 0.9.0.
バージョン 0.9.0 対象。

## What this add-on does NOT do / 行わないこと

- No telemetry, analytics, or usage tracking.
- No account, license-server, or online activation communication.
- No network access of any kind at runtime.
- No bundled third-party Python packages; only the Blender API, the Python standard library, and Blender's bundled NumPy are used.

- テレメトリ、アナリティクス、利用状況の収集は行いません。
- アカウント認証、ライセンスサーバ、オンラインアクティベーションの通信は行いません。
- 実行時にいかなるネットワークアクセスも行いません。
- 外部 Python パッケージは同梱せず、Blender API・標準ライブラリ・Blender 同梱の NumPy のみを使用します。

## What is stored locally / ローカルに保存されるもの

- The trial build stores its start date in the add-on preferences (`trial_started`) inside your Blender configuration. If Blender's "Auto-Save Preferences" is enabled, Blender may write this to your user preferences file. The standard build does not use this field.
- The add-on registers keymap entries, application handlers, and timers inside the running Blender session as part of normal operation.

- 試用版ビルドは試用開始日をアドオン設定(`trial_started`)として Blender の設定内に保存します。Blender の「プリファレンスを自動保存」が有効な場合、ユーザー設定ファイルへ書き込まれることがあります。標準版はこのフィールドを使用しません。
- 本アドオンは通常動作の一部として、実行中の Blender セッションに keymap、アプリケーションハンドラ、タイマーを登録します。

## Future changes / 将来の変更

If any future version adds data collection of any kind, it will be announced before the update and will require separate notice and consent.

将来のバージョンで何らかのデータ収集を追加する場合は、更新前に告知し、別途の通知と同意を行います。
