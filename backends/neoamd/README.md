# NeoAMD Backend Pack

NeoAMD is an optional backend for supported AMD RDNA4 GPUs. cHiDeScaler-Neo works normally with DirectML when this pack is not installed.

## Installation

1. Download `NeoAMD-Backend-Pack.zip` separately from the Neo application release.
2. Extract the ZIP.
3. Copy the extracted files directly into this directory:

```text
backends/neoamd/
```

The final layout must look like:

```text
backends/neoamd/backend.json
backends/neoamd/neo_amd_backend.dll
backends/neoamd/models/...
```

Do not place the files inside an additional nested `NeoAMD-Backend-Pack` folder.

Neo shows the **NeoAMD** option only after a compatible pack and supported AMD adapter are detected. Unsupported models or backend failures fall back to the established DirectML path.

The public Backend Pack does not include AMD's HIP runtime DLL. Use a current compatible AMD GPU driver, which provides the HIP runtime used by the backend.
