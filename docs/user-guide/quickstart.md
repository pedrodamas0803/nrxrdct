# Quickstart

This page walks through the minimal steps to get from raw detector images to reconstructed XRD-CT maps.

## 1. Describe your scan

```python
from pathlib import Path
from nrxrdct.parameters import Scan

scan = Scan(
    acquisition_file=Path("data/sample.h5"),
    sample_name="my_sample",
    beam_energy=44,       # keV
    beam_size=100e-6,     # m
)
```

## 2. Integrate frames

```python
import numpy as np
from nrxrdct.integration import integrate_powder_parallel

angles = np.linspace(0, np.pi, 901)

integrate_powder_parallel(
    master_file=Path("data/sample.h5"),
    output_file=Path("integrated.h5"),
    poni_file="detector.poni",
    mask_file="mask.edf",
    rot=angles,
    n_points=1000,
    remove_spots=True,
)
```

## 3. Assemble sinogram and reconstruct

```python
from nrxrdct.reconstruction import assemble_sinogram, reconstruct_slice

sino = assemble_sinogram(Path("integrated.h5"), n_rot=901, n_tth_angles=1000)
# sino shape: (n_tth, n_lines, n_rot)

tth = np.load("tth.npy")
volume = np.stack([
    reconstruct_slice(sino[i], angles_rad=angles)
    for i in range(sino.shape[0])
])
```

## 4. Per-voxel Rietveld refinement

```python
from nrxrdct.reconstruction import ReconstructedVolume

rv = ReconstructedVolume(volume, tth, sample_name="my_sample", phases=["Al"])
rv.write_xy_files_parallel()
rv.refine_models_parallel(my_refinement_function)  # user-supplied
```

## 5. Extract parameter maps

```python
rwp_map = rv.get_Rwp_map()
a_map, b_map, c_map = rv.get_cell_map()
```

---

See [Typical Workflow](workflow.md) for the full pipeline including instrument calibration.
