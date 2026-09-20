# NeoAMD external backend ABI 1

## Ownership boundary

Neo owns pack discovery, integrity checks, GUI/backend selection, model-chain
lifetime, statistics labels and safe DirectML fallback. The Backend Pack owns:

- HIP/RDNA4 device mapping and architecture reporting;
- ONNX graph parsing and **structural** model-family recognition;
- weight extraction/packing and model-specific execution plans;
- gfx1200/gfx1201 HIP/WMMA kernels;
- dynamic-resolution workspace allocation/reuse;
- inference correctness and runtime diagnostics.

The host must never need a filename/hash allow-list for NeoAMD. A reviewed hash can
be used by the pack as additional evidence, but graph structure/tensor/operator
semantics are authoritative and unknown graphs must fail closed.

## Resolution contract

`neoamd_create_session()` receives model path + adapter LUID only. It must not
compile or cache an HxW-specific engine. `width` and `height` first appear in
`NeoAmdRunDesc`. Workspaces may grow/reuse lazily, but changing resolution must
not rebuild the model execution engine or block on a shape compilation step.

## RDNA4 contract

ABI 1 accepts `gfx1200` and `gfx1201`; do not whitelist one RX 9060 XT PCI ID.
`neoamd_query_adapter()` must map the DXGI LUID to the corresponding HIP device
and report the HIP architecture. Neo requires these capability bits before the
option is exposed: dynamic resolution, native FP16, gfx12 WMMA and structural
routing.

## Failure contract

Any non-zero bridge status is treated as unsupported/error. At model creation or
runtime, Neo retains/pre-creates a DirectML session and falls back safely. Backend
errors must be human-readable via `neoamd_last_error()` and must never terminate
the host process.

## ABI evolution

Keep ABI 1 binary-compatible: honor `struct_size`, leave reserved fields zero and
append future fields only behind a new `struct_size` check. Breaking changes use
Bridge ABI 2. Pack releases may otherwise update independently from Neo as long as
they preserve this ABI and manifest contract.

## GPU-resident extension

`NEOAMD_CAP_D3D12_SHARED_BUFFER` is reserved for the GPU-resident handoff. Host
v745 does not call that extension yet; production performance work should add new
optional exports without changing the existing CPU-visible safety path. Do not
repurpose existing ABI fields.
