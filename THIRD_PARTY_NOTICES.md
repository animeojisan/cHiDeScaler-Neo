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

### Blur Busters CRT Beam Simulator

Neo includes an adapted refresh-cycle shader at:

`shaders/CRT/CRT Beam Simulator-Neo.glsl`

It is derived from the Blur Busters `crt-beam-simulator` project and is kept
under the upstream MIT License terms. The Neo filename carries the `-Neo`
suffix because the shader has been adapted to cHiDeScaler-Neo's shader
interface and Display-Hz execution path.

The upstream license text is preserved as:

`shaders/CRT/CRT Beam Simulator LICENSE.txt`

Upstream project: https://github.com/blurbusters/crt-beam-simulator

## TensorRT / CUDA

TensorRT and CUDA runtime binaries are not included in this repository. They
are installed separately as the optional Neo TensorRT Backend Pack and remain
subject to NVIDIA's applicable licenses and redistribution terms.

## DLSS Neural Rendering / NVIDIA NGX

DLSS Neural Rendering and NVIDIA NGX runtime/SDK binaries are not included in
this repository. Optional DLSSNR support is isolated behind the external
`backends/dlssnr/` Backend Pack ABI. Any NVIDIA or community-modified runtime
placed there remains subject to its own applicable license, authorization, and
redistribution terms.

Public implementation research used to validate the initial same-resolution,
zero-guidance architecture includes the MIT-licensed experimental project:

- Resolve DLSS5 Experimental: https://github.com/SAOG0721/DaVinci-Resolve-DLSS5

## Reference projects

- mpv_PlayKit: https://github.com/hooke007/mpv_PlayKit
- vs_temporalfix: https://github.com/pifroggi/vs_temporalfix
- Magpie: https://github.com/Blinue/Magpie
- Anime4K: https://github.com/bloc97/Anime4K
- OpenModelDB: https://openmodeldb.info/

## External SLANGP shader packs

cHiDeScaler-Neo includes only its generic `.slangp/.slang` loader.
No RetroCrisis, Guest Advanced, libretro Slang preset, shader source, or LUT
asset is bundled in this source package. Users may place separately obtained
shader packs below the portable `slangp/` directory. Those external files remain
subject to their own licenses; loading them at runtime does not relicense them
under the cHiDeScaler-Neo project license.

## DLSS Neural Rendering Backend Pack runtime flexibility
The optional DLSSNR backend boundary can be configured to accept a user-supplied
runtime below `backends/dlssnr/runtime/`. Such a runtime is not part of
cHiDeScaler-Neo, is not authenticated or endorsed by Neo when hash pinning is
explicitly disabled for that runtime file, and remains subject to its own
license/redistribution terms. The Neo bridge DLL itself remains hash-pinned.

### Resolve DLSS5 Experimental implementation reference

The `backends/dlssnr/feature18_backend/` implementation uses an independent
Neo ABI wrapper and adapts Feature 18 D3D12 initialization/compatibility
techniques from the MIT-licensed `SAOG0721/DaVinci-Resolve-DLSS5` project.
The retained MIT terms are in:

`backends/dlssnr/feature18_backend/LICENSE.resolve-dlss5.txt`

No NVIDIA SDK library, header, or DLSSNR runtime binary is bundled in this source package.
