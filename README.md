<details>
<summary>English</summary>

# cHiDeScaler-Neo

**AI Upscaler & Frame Interpolator for Windows**

cHiDeScaler-Neo captures a Windows application window and applies real-time GLSL / ONNX video processing.

## Features

* Windows Graphics Capture (WGC)
* GLSL / ONNX filter chains and presets
* DirectML ONNX backend
* Optional TensorRT backend for GeForce RTX 20 / 30 / 40 / 50 Series
* RIFE / DRBA frame interpolation (x2–x5)
* FPS limit / duplicate-frame reduction
* Draw Stabilization / VSync / capture-resolution controls
* Mini / Basic / Full GUI
* Multi-language GUI and portable configuration

The bundled `presets.json`, GLSL shaders, and ONNX models are kept together so the included presets can resolve their required files.

**GPU Load** is a reference value and does not directly represent a percentage of the GPU's maximum theoretical performance.

### Currently frozen

HDR → SDR / HDR highlight protection and some experimental implementations remain disabled for later review.

## Build

On Windows:

```bat
BUILD_RELEASE.bat
```

The standard DirectML runtime files are included under `backends/`. TensorRT is an optional separate backend pack; see [`backends/tensorrt/README.md`](backends/tensorrt/README.md).

## Acknowledgements / Reference Projects

* [mpv_PlayKit (hooke007)](https://github.com/hooke007/mpv_PlayKit)
* [vs_temporalfix (pifroggi)](https://github.com/pifroggi/vs_temporalfix)
* [Magpie (Blinue)](https://github.com/Blinue/Magpie)
* [Anime4K (bloc97)](https://github.com/bloc97/Anime4K)
* [OpenModelDB](https://openmodeldb.info/)

Source / license information for some bundled third-party models and shaders is still being organized and will be added as it is confirmed. See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## Disclaimer

This software is an independently developed personal project. Please use it at your own discretion and responsibility.

The author cannot be held responsible for any malfunction, damage, trouble, or other issue arising from the installation, configuration, or use of this software.

</details>

# cHiDeScaler-Neo

**Windows用 AI アップスケーラー & フレーム補間**

cHiDeScaler-Neo は、PC上の任意のウィンドウをリアルタイムにキャプチャし、GLSL / ONNXによる映像処理を適用して拡大表示するWindows用ポータブルアプリケーションです。

## 主な機能

* Windows Graphics Capture（WGC）
* GLSL / ONNXフィルターチェーンとプリセット
* DirectML ONNXバックエンド
* GeForce RTX 20 / 30 / 40 / 50シリーズ向けオプションTensorRTバックエンド
* RIFE / DRBA フレーム補間（x2～x5）
* FPS上限 / 重複フレーム削減
* 描画安定化 / VSync / キャプチャ解像度設定
* Mini / Basic / Full GUI
* 多言語GUI / ポータブル設定

同梱 `presets.json` から必要なファイルを参照できるよう、現在使用しているGLSLシェーダー / ONNXモデルは同じ構成で維持しています。

GUIの **GPU負荷率は参考値** です。GPUの最大演算性能をそのまま何％消費しているかを示す値ではありません。

## 現在凍結中

HDR → SDR / HDR白飛び防止処理など、一部の実装は将来の再検討用としてコードを保持したまま現在は無効化しています。

## ビルド

Windowsで、

```bat
BUILD_RELEASE.bat
```

を実行してください。

標準のDirectML動作に必要なランタイムは `backends/` に同梱しています。TensorRTは別途追加するオプションバックエンドです。詳細は [`backends/tensorrt/README.md`](backends/tensorrt/README.md) を参照してください。

## 参考プロジェクト / 謝辞

* [mpv_PlayKit (hooke007)](https://github.com/hooke007/mpv_PlayKit)
* [vs_temporalfix (pifroggi)](https://github.com/pifroggi/vs_temporalfix)
* [Magpie (Blinue)](https://github.com/Blinue/Magpie)
* [Anime4K (bloc97)](https://github.com/bloc97/Anime4K)
* [OpenModelDB](https://openmodeldb.info/)

一部の外部モデル / シェーダーは、出所・ライセンス情報を現在整理中です。確認できたものから順次記載を追加します。詳細は [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) を参照してください。

## 免責事項

本ソフトウェアは個人制作によるものです。ご利用は各自の判断と責任でお願いいたします。  
本ソフトウェアの導入、設定、使用により生じたいかなる不具合、損害、トラブル等についても作者は責任を負いかねます。

