# Third-party notices

cHiDeScaler-Neo contains original project code together with third-party source,
runtime binaries, GLSL shaders, and ONNX models. The cHiDeScaler-Neo project
license does not replace or override the license of any third-party component.

## Standard DirectML runtime

The following runtime files are included because they are required by the
standard DirectML configuration used by cHiDeScaler-Neo:

- `backends/onnxruntime.dll`
- `backends/onnxruntime_providers_shared.dll`
- `backends/DirectML.dll`

ONNX Runtime is a Microsoft open-source project distributed under the MIT
License. DirectML is provided by Microsoft and its standalone redistributable
is subject to Microsoft's applicable redistribution terms.

The exact runtime versions and redistribution notices should continue to be
checked whenever these DLLs are updated.

## Vendored source

### windows-capture

Location: `vendor/windows-capture/`

The upstream MIT license is preserved as:

`vendor/windows-capture/LICENCE`

## Bundled GLSL shaders and ONNX models

The current `shaders/` and `models/` trees are intentionally retained because
the bundled `presets.json` references these files and removing them would make
included presets incomplete.

Some of these resources have clearly identifiable upstream projects; for other
files, source / authorship / license metadata is still being organized.

**Status:** provenance and license documentation is being reviewed and will be
updated as information is confirmed.

Inclusion in this repository does not transfer ownership of a third-party
resource and does not relicense it under the cHiDeScaler-Neo source license.

Before independently redistributing, modifying, or commercially using an
individual third-party model or shader, check the terms applicable to that
specific file.

## TensorRT / CUDA

TensorRT and CUDA runtime binaries are not included in this repository. They
are installed separately as the optional Neo TensorRT Backend Pack and remain
subject to NVIDIA's applicable licenses and redistribution terms.

## Reference projects

- mpv_PlayKit: https://github.com/hooke007/mpv_PlayKit
- vs_temporalfix: https://github.com/pifroggi/vs_temporalfix
- Magpie: https://github.com/Blinue/Magpie
- Anime4K: https://github.com/bloc97/Anime4K
- OpenModelDB: https://openmodeldb.info/
