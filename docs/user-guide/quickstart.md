# Quickstart

This page walks through the minimal steps to get from raw detector images to
reconstructed XRD-CT maps using `nrxrdct`.  It covers the five mandatory
pipeline stages: scan description, frame integration, sinogram assembly and
reconstruction, per-voxel Rietveld refinement, and parameter map extraction.

> **Prerequisites**: an installed `nrxrdct` environment — see
> [Installation](../installation.md).  For a full pipeline including instrument
> calibration see [Typical Workflow](workflow.md).

---

## 1. Describe your scan

Create a `Scan` object that holds the acquisition file path and beam metadata.
All downstream pipeline functions accept a `Scan` instance (or its fields
directly) so that calibration parameters are not repeated.

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

---

## 2. Integrate frames

Azimuthally integrate every detector frame in parallel using the pyFAI geometry
stored in the `.poni` calibration file.  The optional `remove_spots` flag
applies a sigma-clipping filter to remove Bragg spots before integration
(recommended for single-crystal or coarse-grained samples).

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

The result is an HDF5 file containing one 1-D powder pattern per scan position
and rotation angle.

---

## 3. Assemble sinogram and reconstruct

Stack the integrated patterns into a sinogram and reconstruct each
two-theta channel with filtered back-projection (or an iterative algorithm
when a GPU is available).

```python
from nrxrdct.reconstruction import assemble_sinogram, reconstruct_slice

sino, dty = assemble_sinogram(Path("integrated.h5"), n_rot=901, n_tth_angles=1000)
# sino shape: (n_tth, n_lines, n_rot); dty is aligned with sino's line axis

tth = np.load("tth.npy")
volume = np.stack([
    reconstruct_slice(sino[i], angles_rad=angles)
    for i in range(sino.shape[0])
])
```

See [GPU Support](gpu.md) for algorithm selection when a CUDA device is present.

---

## 4. Per-voxel Rietveld refinement

Wrap the reconstructed volume in a `ReconstructedVolume` object, export
per-voxel `.xy` patterns, then run a user-supplied GSAS-II refinement
function in parallel across all spatial voxels.

```python
from nrxrdct.reconstruction import ReconstructedVolume

rv = ReconstructedVolume(volume, tth, sample_name="my_sample", phases=["Al"])
rv.write_xy_files_parallel()
rv.refine_models_parallel(my_refinement_function)  # user-supplied
```

---

## 5. Extract parameter maps

Query the fitted results to obtain 2-D maps of any refined parameter.

```python
rwp_map = rv.get_Rwp_map()
a_map, b_map, c_map = rv.get_cell_map()
```

---

See [Typical Workflow](workflow.md) for the full pipeline including instrument
calibration and an annotated package-structure overview.
