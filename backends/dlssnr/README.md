# Neo DLSS Neural Rendering Backend Pack

DLSS Neural Rendering is optional. A normal Neo installation must continue to
work without this directory containing a usable pack.

## Stability boundary

Neo does not link NVIDIA NGX/DLSS libraries into the main executable. A pack
provides a small `neo_dlssnr_backend.dll` bridge with the C ABI in
`include/neo_dlssnr_backend.h`. The bridge privately owns D3D12, NGX/DLSSNR,
its runtime DLLs, history, compatibility workarounds, and future motion/depth
providers.

Normal Neo startup only reads `backend.json` and checks file existence. It does
not load the bridge and does not hash a large runtime. Full preflight and DLL
loading are reserved for an explicit DLSSNR enable/probe request.

Every bridge failure is fail-open by contract: the caller keeps the original
frame. The bridge must never return black/stale output as an error substitute,
terminate Neo, modify system/game DLLs, or inject into the captured process.

## Runtime flexibility

The host deliberately does not hard-code one DLSSNR runtime build. A pack can
use `runtime_candidates` to describe locations its bridge understands.

Strict packs keep `allow_user_runtime_replacement` false and pin every required
file with SHA-256. A mod-friendly pack may set it true and list selected files in
`replaceable_runtime_files`. Those entries:

- must remain below `runtime/`;
- are checked for existence but are not hash-pinned by Neo;
- can never include `neo_dlssnr_backend.dll`;
- are loaded/validated only by the backend bridge, never directly by Neo.

This is intended to let advanced users try future official/community/modded
DLSSNR runtimes without requiring a Neo rebuild. It does not make an unknown
runtime trusted; only use runtime binaries from sources you accept and under
terms that permit your use.

The public package includes the active `backend.json`. Rebuilding the bridge with
`BUILD_DLSSNR_BACKEND.bat` regenerates that manifest with the matching bridge
SHA-256.

## Initial ABI

ABI v1 intentionally uses CPU RGBA8 input/output. This is the conservative
bring-up path. A later optional path can add shared D3D12 resources after the
DLSSNR path is proven stable; changing that transport must not change normal
DirectML, TensorRT, OpenGL, Vulkan, WGC, pacing, or input behavior.

The first real backend should use same-resolution DLSS Neural Rendering with
zero motion/depth. Motion vectors should be added later as an optional backend
capability, preferably NVIDIA Optical Flow. Estimated depth remains optional.

ABI-v1 also defines optional capability/description exports. Old v1 bridges do
not need them and remain loadable.

## Pack layout

```text
backends/dlssnr/
  backend.json
  neo_dlssnr_backend.dll
  runtime/
    nvngx_dlssnr.dll
```

Runtime and SDK files are not included in the Neo source tree. Pack distributors
and users must comply with the applicable licenses and redistribution terms.

## Feature 18 implementation

`feature18_backend/` contains the real bridge implementation. It creates a
private NVIDIA D3D12/NGX session, evaluates same-resolution RGBA8 frames, and
returns RGBA8 through the conservative ABI-v1 boundary. Zero motion/depth is the
baseline guidance mode. The main Neo process does not load NVIDIA runtime DLLs.

### Build

1. Obtain the NVIDIA DLSS/NGX SDK and place it in `external/DLSS/` or
   `external/DLSS-main/` (or set `DLSS_SDK_ROOT`).
2. Run `BUILD_DLSSNR_BACKEND.bat`.
3. The script builds/installs the bridge, writes `backend.json` as BOM-free
   UTF-8, then automatically builds the Neo release executable.
4. Place an authorized `nvngx_dlssnr.dll` in
   `backends/dlssnr/runtime/nvngx_dlssnr.dll` (or a declared community/mod
   candidate directory).

`TEST_DLSSNR.bat` is an optional standalone diagnostic. It is not required for a
normal build or normal Neo use.

### Controls and alignment

Bridges advertising the advanced-options capability can expose Render Preset,
Style, Intensity, Local Tone, Local Structure, Skin Structure, Automatic Mask,
and UI Correction. Applying create-latched settings restarts only the isolated
DLSSNR worker/session.

For runtimes requiring four-pixel aligned dimensions, Neo preserves the logical
capture size and edge-pads only the private worker geometry. Example:
`1440x810 -> 1440x812 -> 1440x810`. Aligned modes are unchanged.

The NVIDIA SDK/runtime is not bundled. This integration remains an experimental
Feature 18 path; compatibility with arbitrary runtime versions or GPUs is not
guaranteed. Backend failures remain fail-open to the original frame.
