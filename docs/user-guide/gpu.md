# GPU Support

This page describes how `nrxrdct` uses NVIDIA GPU acceleration for
tomographic reconstruction, which algorithm is selected in each mode,
and how to verify GPU availability.

---

## 1. Automatic device detection

`nrxrdct` detects NVIDIA GPUs automatically at import time via
`astra.get_gpu_info()`.  When a GPU is available, `reconstruct_slice` and
`reconstruct_astra_gpu_3d` switch to CUDA-accelerated algorithms; when no
GPU is found the code falls back to CPU algorithms transparently — no code
changes are required.

| Algorithm | Mode |
|---|---|
| `SART_CUDA` | GPU |
| `SIRT3D_CUDA` | GPU |
| `CGLS3D_CUDA` | GPU |
| `FBP` | CPU fallback |

---

## 2. Requirements

- NVIDIA GPU with CUDA support
- [ASTRA Toolbox](https://astra-toolbox.com/) compiled with CUDA support

---

## 3. Checking GPU availability

```python
import astra
print(astra.get_gpu_info())
```
