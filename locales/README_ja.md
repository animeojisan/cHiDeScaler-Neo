# locales

cHiDeScaler-Neo のGUI翻訳ファイルを格納するフォルダです。

標準の言語JSONに加えて、ユーザーが独自の `*.json` を追加できます。
翻訳が存在しないキーは英語へフォールバックします。

## 標準収録言語

- `ja-JP.json` — 日本語
- `en-US.json` — English
- `zh-CN.json` — 简体中文
- `zh-TW.json` — 繁體中文（台灣）
- `ko-KR.json` — 한국어
- `pt-BR.json` — Português (Brasil)
- `es.json` — Español
- `fr-FR.json` — Français
- `de-DE.json` — Deutsch

> `es.json` は現行Neoで汎用スペイン語タグ `es` として組み込まれています。
> 単純に `es-ES.json` へ改名しないでください。

## 翻訳JSONを追加する

1. `en-US.json` をコピーします。
2. ファイル名をBCP 47タグへ変更します（例: `it-IT.json`）。
3. 先頭の3項目を変更します。

```json
{
  "_language_name": "Italiano",
  "_language_code": "IT",
  "_language_tag": "it-IT"
}
```

値だけを翻訳し、キー名や `{name}` などのプレースホルダーは変更しません。

追加後はNeoを再起動してください。

## 関連資料

- [`LANGUAGE_CODES_ja.md`](LANGUAGE_CODES_ja.md) — 言語コード一覧
- [`TRANSLATION_KEYS.md`](TRANSLATION_KEYS.md) — 現在の翻訳キーと英語基準文
