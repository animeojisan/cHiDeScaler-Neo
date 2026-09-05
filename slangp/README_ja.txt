cHiDeScaler-Neo 外部SLANGPプリセット導入ガイド

NeoのSLANGP機能は、ユーザーが用意したLibretro / RetroArch Slangシェーダーを
外部ファイルとして読み込みます。
Neo本体にはRetroCrisis、Guest Advanced、Libretroのシェーダーコード、
プリセット、LUT画像などは同梱していません。
外部シェーダーパックは、それぞれの配布元のライセンスに従って利用してください。

============================================================
1. 必要ファイルの取得・配置方法
============================================================

【GitHubからLibretro公式 slang-shadersを取得する場合】

1. Libretro公式 slang-shadersを開きます。
   https://github.com/libretro/slang-shaders

2. 「Code」→「Download ZIP」を選択します。

3. ZIPを展開し、展開された
   slang-shaders-master
   フォルダを
   shaders_slang
   へ名前変更します。

4. shaders_slangフォルダをNeoのslangpフォルダ内へ配置します。

配置先:
  cHiDeScaler-Neo/slangp/shaders_slang/


【RetroArchから取得する場合】

RetroArchを導入済みの場合は、以下の方法でも取得できます。

1. RetroArchで
   Main Menu → Online Updater → Update Slang Shaders
   を実行します。

2. RetroArch内の
   shaders/shaders_slang
   フォルダをコピーします。

3. コピーしたshaders_slangフォルダをNeoの
   slangp/shaders_slang
   へ配置します。

配置先:
  cHiDeScaler-Neo/slangp/shaders_slang/

この更新方法はLibretro公式ドキュメントでも案内されています。
  https://docs.libretro.com/guides/shaders/


============================================================
2. RetroCrisisプリセットの入手先
============================================================

Retro Crisis GDV-NTSCプリセットはこちらから入手できます。

RetroCrisis / Retro-Crisis-GDV-NTSC
https://github.com/RetroCrisis/Retro-Crisis-GDV-NTSC

GitHubの「Code」→「Download ZIP」からダウンロードし、展開したプリセットフォルダを
Neoのslangpフォルダ内へ配置してください。

例:
  cHiDeScaler-Neo/
  └─ slangp/
     ├─ shaders_slang/
     │  ├─ crt/
     │  ├─ ntsc/
     │  └─ ...
     │
     └─ retro crisis/
        ├─ 1080p Flat/
        ├─ 1080p Curved/
        ├─ 1440p Flat/
        ├─ 4K Flat/
        └─ ...

RetroCrisisの一部プリセットは、別のプリセットやshaders_slang内の
.slangファイル、LUT画像などを相対パスで参照します。
そのため、特定の.slangpファイルだけを単独でコピーするのではなく、
配布パックのフォルダ構造をできるだけ維持したまま配置してください。

============================================================
3. Neoでの使用方法
============================================================

1. 必要なファイルをslangpフォルダへ配置します。
2. Neoを起動します。
3. フィルター追加画面を開きます。
4. 使用したい.slangpプリセットを選択します。

Neoのフィルター一覧には.slangpプリセットのみ表示されます。
内部で使用される.slangファイルやLUT画像は依存ファイルとして扱われ、
フィルター一覧には表示されません。

フィルターチェーンではSLANGPとして1つのフィルター項目にまとめて表示されます。

============================================================
4. Portable配置について
============================================================

Neoは、プリセットから参照されるshaders_slangパスを
slangp/shaders_slangへ再基準化して解決できます。
これにより、RetroArch本体とは別のポータブル環境でも外部SLANGPを利用できます。

推奨構成:
  cHiDeScaler-Neo/
  └─ slangp/
     ├─ shaders_slang/
     └─ <任意のSLANGPプリセットパック>/

外部シェーダーパックはNeo本体とは別の配布物です。
各ファイルのライセンス・利用条件は、それぞれの配布元をご確認ください。
