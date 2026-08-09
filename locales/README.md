# locales

This folder contains the GUI localization files used by cHiDeScaler-Neo.

Built-in language JSON files are loaded together with optional user-created
`*.json` files. Missing translation keys fall back to English.

## Built-in locale files

- `ja-JP.json` — 日本語
- `en-US.json` — English
- `zh-CN.json` — 简体中文
- `zh-TW.json` — 繁體中文（台灣）
- `ko-KR.json` — 한국어
- `pt-BR.json` — Português (Brasil)
- `es.json` — Español
- `fr-FR.json` — Français
- `de-DE.json` — Deutsch

> `es.json` intentionally uses the generic built-in locale tag `es`.
> Do not rename it to `es-ES.json` without changing the application source.

## Adding a translation

1. Copy `en-US.json`.
2. Rename the file to a BCP 47 locale tag such as `it-IT.json`.
3. Edit the three metadata fields at the top:

```json
{
  "_language_name": "Italiano",
  "_language_code": "IT",
  "_language_tag": "it-IT"
}
```

Translate values only. Do not rename translation keys or placeholders such as
`{name}`.

Restart Neo after adding a locale file.

## Reference

- [`LANGUAGE_CODES.md`](LANGUAGE_CODES.md) — language-code reference table
- [`TRANSLATION_KEYS.md`](TRANSLATION_KEYS.md) — current translation keys and English base text
