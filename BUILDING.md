# Building

## Requirements

- Windows 10 / 11 (64-bit)
- Rust toolchain with `cargo`
- MSVC-compatible Windows build tools

Run:

```bat
BUILD_RELEASE.bat
```

or:

```bat
cargo build --release --bin chidescaler-neo
```

The standard DirectML runtime DLLs, bundled models, shaders, and presets are
kept in this repository so a normal checkout preserves the application's
expected runtime layout.

TensorRT is optional and is installed separately under `backends/tensorrt/`.
