# Neo TensorRT Backend

TensorRT is an optional backend. The normal DirectML path works without this pack.

## Target GPUs

The current backend design targets NVIDIA GeForce RTX:

- RTX 20 Series
- RTX 30 Series
- RTX 40 Series
- RTX 50 Series

The backend is based on TensorRT 10.14.1 / CUDA 12.x generation components.
Use a current compatible NVIDIA driver.

TensorRT engines are generated on the user's PC for the selected GPU, model,
resolution, and profile. Do not assume an engine generated on one GPU can be
copied to a different GPU generation.

Models that cannot use the TensorRT path may fall back to DirectML.

## Installation

Place the separately distributed backend pack files in:

```text
backends/tensorrt/
```

A valid pack contains `backend.json` and the required TensorRT / CUDA provider
files. `backend.json.example` is included here as a reference only.

NVIDIA components remain subject to NVIDIA's licenses and redistribution terms.
