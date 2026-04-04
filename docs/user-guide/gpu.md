# GPU Support

`nrxrdct` detects NVIDIA GPUs automatically at import time via `astra.get_gpu_info()`.

When a GPU is available, `reconstruct_slice` and `reconstruct_astra_gpu_3d` use CUDA-accelerated algorithms:

| Algorithm | Mode |
|---|---|
| `SART_CUDA` | GPU |
| `SIRT3D_CUDA` | GPU |
| `CGLS3D_CUDA` | GPU |
| `FBP` | CPU fallback |

When no GPU is found, the code falls back to CPU algorithms transparently — no code changes required.

## Requirements

- NVIDIA GPU with CUDA support
- [ASTRA Toolbox](https://astra-toolbox.com/) compiled with CUDA support

## Checking GPU availability

```python
import astra
print(astra.get_gpu_info())
```
