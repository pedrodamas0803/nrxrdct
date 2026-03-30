# nrxrdct

**Near-field X-ray diffraction computed tomography** — a Python toolkit for the full XRD-CT data-reduction pipeline, from raw detector images to spatially-resolved maps of crystallographic parameters.

---

## Overview

`nrxrdct` covers every step of a synchrotron XRD-CT experiment:

1. **Preprocessing** — hot-pixel (zinger) removal from detector images.
2. **Integration** — parallelised 1-D azimuthal integration and 2-D CAKE regrouping with pyFAI, including sigma-clipping and percentile-filtering strategies for single-crystal spot rejection.
3. **Sinogram assembly** — stacking integrated patterns from HDF5 master files into sinogram arrays with background subtraction and monitor normalisation.
4. **Tomographic reconstruction** — GPU-accelerated (SIRT3D, CGLS3D, SART, SIRT) and CPU (FBP) reconstruction via the ASTRA Toolbox, with automatic GPU detection.
5. **Rietveld refinement** — GSAS-II scripting wrappers (`BaseRefinement`, `InstrumentCalibration`) for sequential step-by-step refinement (background, scale, zero shift, peak broadening, cell parameters, preferred orientation, crystallite size, microstrain, extinction).
6. **Volume analysis** — the `ReconstructedVolume` class manages per-voxel `.xy` file I/O and parallelised GSAS-II refinement, with map extractors for Rwp, unit-cell lengths, and crystallite sizes.
7. **Fluorescence** — loading of XRF ROI and full-spectrum sinograms from HDF5 files, with emission line look-up via xraylib.
8. **NMF decomposition** — non-negative matrix factorisation of hyperspectral volumes for phase mapping (`HyperspectralNMF`).
9. **Utilities** — baseline fitting, powder pattern simulation, circular masking, peak listing from CIF files.
10. **Visualisation** — napari-based interactive 3-D volume and slice viewers with live Z-profile extraction on pixel click.

---

## Installation

### Requirements

- Python **3.11** or later
- A working [GSAS-II](https://gsas-ii.readthedocs.io/) installation (scripting interface `GSASIIscriptable` must be importable)
- [ASTRA Toolbox](https://astra-toolbox.com/) (GPU reconstruction requires an NVIDIA GPU and CUDA)

### With `uv` (recommended)

```bash
git clone <repo-url>
cd nrxrdct
uv sync
```

### With `pip`

```bash
git clone <repo-url>
cd nrxrdct
pip install -e .
```

---

## Dependencies

### Declared (`pyproject.toml`)

| Package | Purpose |
|---|---|
| `numpy >= 2.4` | Array operations throughout |
| `scipy >= 1.17` | Median filter, Gaussian filter |
| `h5py >= 3.16` | HDF5 file I/O |
| `matplotlib >= 3.10` | Plotting |
| `pyFAI >= 2026.2` | Azimuthal integration |
| `scikit-image >= 0.26` | Block reduce for XRF binning |
| `scikit-learn >= 1.8` | NMF decomposition |
| `pandas >= 3.0` | Peak table output |
| `xraylib >= 4.2` | XRF emission line energies |
| `xrayutilities >= 1.7` | Powder pattern simulation and peak listing |

### Additional runtime dependencies

These are used by specific modules but are not yet declared in `pyproject.toml` and must be installed separately:

| Package | Module | Purpose |
|---|---|---|
| `astra-toolbox` | `reconstruction` | Tomographic reconstruction (GPU + CPU) |
| `GSAS-II` | `reconstruction`, `refinement` | Rietveld refinement scripting |
| `fabio` | `integration` | Reading mask files |
| `hdf5plugin` | `reconstruction` | Compressed HDF5 dataset support |
| `napari` | `visualization` | Interactive 3-D volume viewer |
| `pybaselines` | `utils` | XRD baseline fitting |
| `tqdm` | `integration`, `fluorescence`, `nmf` | Progress bars |

---

## Package structure

```text
src/nrxrdct/
├── __init__.py          # Package entry point
├── parameters.py        # Scan metadata container (Scan class)
├── preprocessing.py     # Zinger removal
├── integration.py       # pyFAI azimuthal integration pipeline
├── reconstruction.py    # ASTRA reconstruction + ReconstructedVolume
├── refinement.py        # GSAS-II refinement wrappers
├── refine_dict.py       # Pre-built GSAS-II refinement dictionary templates
├── fluorescence.py      # XRF sinogram loading
├── nmf.py               # NMF decomposition for hyperspectral volumes
├── utils.py             # Baseline fitting, simulation, masking, padding
├── plotting.py          # CAKE integration plotting
├── visualization.py     # napari-based interactive viewers
└── io.py                # HDF5 and .xy file I/O, GSAS-II instprm export
```

---

## Typical workflow

```python
from pathlib import Path
from nrxrdct.parameters import Scan
from nrxrdct.integration import integrate_powder_parallel
from nrxrdct.reconstruction import assemble_sinogram, reconstruct_slice, ReconstructedVolume
import numpy as np

# 1. Describe the scan
scan = Scan(
    acquisition_file=Path("data/sample.h5"),
    sample_name="my_sample",
    beam_energy=44,       # keV
    beam_size=100e-6,     # m
)

# 2. Integrate all frames in parallel (outputs to integrated.h5)
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

# 3. Assemble sinogram and reconstruct
sino = assemble_sinogram(Path("integrated.h5"), n_rot=901, n_tth_angles=1000)
# sino shape: (n_tth, n_lines, n_rot)

tth = np.load("tth.npy")
volume = np.stack([
    reconstruct_slice(sino[i], angles_rad=angles)
    for i in range(sino.shape[0])
])

# 4. Per-voxel Rietveld refinement
rv = ReconstructedVolume(volume, tth, sample_name="my_sample", phases=["Al"])
rv.write_xy_files_parallel()
rv.refine_models_parallel(my_refinement_function)  # user-supplied

# 5. Extract maps
rwp_map = rv.get_Rwp_map()
a_map, b_map, c_map = rv.get_cell_map()
```

### Instrument calibration

```python
from nrxrdct.refinement import InstrumentCalibration

cal = InstrumentCalibration(
    acquisition_file=Path("data/calib.h5"),
    sample_name="LaB6",
    beam_energy=44,
    xy_file=Path("integrated_data.xy"),
    tth_lims=(3.0, 25.0),
)
cal.create_model(gpx_file=Path("calibration/LaB6.gpx"))
cal.add_phase(cif_file=Path("LaB6.cif"), phase_name="LaB6", block_cell=True)
cal.refine_background()
cal.refine_scale()
cal.refine_zero_shift()
cal.refine_gaussian_broadening(["W"])
cal.refine_lorentzian_broadening(["X", "Y"])
cal.free_and_refine_cell()
cal.write_calibrated_instrument_pars()
cal.plot_calibration_results()
```

---

## HPC / SLURM integration

For large datasets where integrating thousands of frames on a single machine would take too long, `nrxrdct` provides a SLURM pipeline that splits the work across cluster nodes.  All functionality is exposed through a Python API and the `nrxrdct-slurm` CLI.

### Submitting jobs

Python API:

```python
from pathlib import Path
from nrxrdct.slurm_integration import launch

slurm_ids = launch(
    master_file  = Path("/data/raw/sample_master.h5"),
    output_file  = Path("/data/processed/integrated.h5"),
    poni_file    = Path("/data/calib/detector.poni"),
    mask_file    = Path("/data/calib/mask.edf"),
    n_jobs       = 8,          # number of SLURM jobs
    n_points     = 1000,       # radial bins
    n_workers    = 16,         # integration threads per job
    batch_size   = 32,         # frames streamed per batch (RAM control)
    method       = "filter",   # "standard" | "filter" | "sigma_clip"
    percentile   = (10, 90),   # used with method="filter"
    partition    = "nice",
    time         = "04:00:00",
    mem          = "64G",
    cpus         = 16,
    conda_env    = "nrxrdct",  # or env_activate=Path("/path/to/activate")
)

print("Submitted job IDs:", slurm_ids)
```

CLI:

```bash
nrxrdct-slurm launch \
    --master-file  /data/raw/sample_master.h5 \
    --output-file  /data/processed/integrated.h5 \
    --poni-file    /data/calib/detector.poni \
    --mask-file    /data/calib/mask.edf \
    --n-jobs       8 \
    --n-points     1000 \
    --n-workers    16 \
    --method       filter \
    --percentile   10,90 \
    --partition    nice \
    --time         04:00:00 \
    --mem          64G \
    --cpus         16 \
    --conda-env    nrxrdct
```

Sbatch scripts and logs are written to `<output_file_dir>/slurm_logs/`.

---

### Monitoring progress

Python API:

```python
from nrxrdct.slurm_integration import monitor

# Single snapshot (non-blocking)
result = monitor(
    slurm_ids   = slurm_ids,
    output_file = Path("/data/processed/integrated.h5"),
)

# Block until all jobs finish, polling every 60 s
result = monitor(
    slurm_ids   = slurm_ids,
    output_file = Path("/data/processed/integrated.h5"),
    watch       = True,
    interval    = 60,
)
```

CLI:

```bash
# Single snapshot
nrxrdct-slurm monitor \
    --slurm-ids   12345,12346,12347 \
    --output-file /data/processed/integrated.h5

# Blocking watch
nrxrdct-slurm monitor \
    --slurm-ids   12345,12346,12347 \
    --output-file /data/processed/integrated.h5 \
    --watch --interval 60
```

The monitor prints a status table like:

```text
────────────────────────────────────────────────────────
  nrxrdct SLURM monitor   elapsed: 0:42:17
────────────────────────────────────────────────────────
  Jobs total   : 8
  ⏳ Pending   : 0
  ▶  Running   : 6
  ✓  Completed : 2
  ✗  Failed    : 0
────────────────────────────────────────────────────────
  Scans  [████████████░░░░░░░░░░░░░░░░░░]  421/901  (46.7%)
  Rate   601.4 scans/hr
  ETA    0:47:53
────────────────────────────────────────────────────────
```

---

### Checking and repairing output

After all jobs finish, verify that every scan was written correctly.

Python API:

```python
from nrxrdct.slurm_integration import check, repair

# Check only — report missing and corrupted scans
result = check(output_file=Path("/data/processed/integrated.h5"))
# result["missing"]   → list of scan indices not written
# result["corrupted"] → list of scan indices with truncated/corrupt datasets

# Check and print manual resubmit hints
check(
    output_file = Path("/data/processed/integrated.h5"),
    resubmit    = True,
)

# Automatic repair — delete corrupted entries and resubmit SLURM jobs
repair_ids = repair(
    output_file = Path("/data/processed/integrated.h5"),
    master_file = Path("/data/raw/sample_master.h5"),
    poni_file   = Path("/data/calib/detector.poni"),
    mask_file   = Path("/data/calib/mask.edf"),
    n_jobs      = 2,
    watch       = True,   # block until repair jobs finish
    partition   = "nice",
    conda_env   = "nrxrdct",
)
```

CLI:

```bash
# Check only
nrxrdct-slurm check --output-file /data/processed/integrated.h5

# Check with manual resubmit hints
nrxrdct-slurm check \
    --output-file /data/processed/integrated.h5 \
    --resubmit

# Automatic repair
nrxrdct-slurm check \
    --output-file  /data/processed/integrated.h5 \
    --repair \
    --master-file  /data/raw/sample_master.h5 \
    --poni-file    /data/calib/detector.poni \
    --mask-file    /data/calib/mask.edf \
    --n-jobs       2 \
    --partition    nice \
    --conda-env    nrxrdct \
    --watch
```

If the output HDF5 is deeply corrupted (damaged B-tree), use `rebuild` to salvage all readable scans into a fresh file before repairing:

```python
from nrxrdct.slurm_integration import rebuild

rebuild(
    output_file = Path("/data/processed/integrated.h5"),
    master_file = Path("/data/raw/sample_master.h5"),
    poni_file   = Path("/data/calib/detector.poni"),
    mask_file   = Path("/data/calib/mask.edf"),
    n_jobs      = 4,
    watch       = True,
    conda_env   = "nrxrdct",
)
# The original file is renamed to integrated.bak.h5
# The rebuilt file takes its place at the original path
```

---

## GPU support

`nrxrdct` detects NVIDIA GPUs automatically at import time via `astra.get_gpu_info()`. When a GPU is available, `reconstruct_slice` and `reconstruct_astra_gpu_3d` use CUDA-accelerated algorithms (`SART_CUDA`, `SIRT3D_CUDA`, `CGLS3D_CUDA`); otherwise the code falls back to CPU algorithms transparently.

---

## Acknowledgements

The NMF module (`nmf.py`) was partially developed by **Beatriz G. Foschiani** (CEA Grenoble).
