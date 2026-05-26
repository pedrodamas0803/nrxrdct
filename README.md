# nrxrdct

**Far-field X-ray diffraction computed tomography** — a Python toolkit for the full XRD-CT data-reduction pipeline, from raw detector images to spatially-resolved maps of crystallographic parameters.

---

## Table of contents

- [Overview](#overview)
- [Installation](#installation)
- [Package structure](#package-structure)
- [Typical workflow](#typical-workflow)
- [Volume analysis](#volume-analysis)
- [HPC / SLURM integration](#hpc--slurm-integration)
- [GPU support](#gpu-support)
- [Laue diffraction](#laue-diffraction)
- [Dependencies](#dependencies)
- [Acknowledgements](#acknowledgements)

---

## Overview

`nrxrdct` is organised into focused subpackages that cover every step of a synchrotron XRD-CT experiment:

| Subpackage | What it does |
|---|---|
| `nrxrdct.azimuthal` | Parallelised 1-D and CAKE azimuthal integration via pyFAI; SLURM pipeline for large datasets |
| `nrxrdct.xrdct` | Sinogram assembly, ASTRA tomographic reconstruction, `ReconstructedVolume` for per-voxel refinement, napari visualisation |
| `nrxrdct.rietveld` | GSAS-II scripting wrappers — `BaseRefinement`, `InstrumentCalibration`, pre-built refinement dictionaries |
| `nrxrdct.fitting` | 1-D peak fitting and NMF decomposition for hyperspectral phase mapping |
| `nrxrdct.fluo` | XRF sinogram loading, emission line look-up via xraylib, spectral fitting |
| `nrxrdct.laue` | Self-contained polychromatic (Laue) diffraction pipeline: simulation, segmentation, orientation/strain fitting, grain maps |
| `nrxrdct.utils` | Baseline fitting, powder pattern simulation, circular masking, peak listing from CIF |

---

## Installation

### Requirements

- Python **3.11** or later
- A working [GSAS-II](https://gsas-ii.readthedocs.io/) installation (`GSASIIscriptable` must be importable)
- [ASTRA Toolbox](https://astra-toolbox.com/) (GPU reconstruction requires an NVIDIA GPU and CUDA)

### With `uv` (recommended)

[`uv`](https://docs.astral.sh/uv/) is a fast Python package and project manager.

**Linux / macOS**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell)**

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Via pip (any platform)**

```bash
pip install uv
```

After installing, restart your terminal so `uv` is on your `PATH`, then clone and install:

```bash
git clone <repo-url>
cd nrxrdct
uv sync
```

`uv sync` creates `.venv/`, installs all declared dependencies, and installs `nrxrdct` in editable mode.

To activate the environment:

**Linux / macOS**

```bash
source .venv/bin/activate
```

**Windows (PowerShell)**

```powershell
.venv\Scripts\Activate.ps1
```

Or skip activation entirely with `uv run`:

```bash
uv run python my_script.py
uv run jupyter lab
```

**Registering a Jupyter kernel (e.g. ESRF Jupyter-SLURM)**

```bash
uv add ipykernel
uv run python -m ipykernel install --user --name nrxrdct --display-name "nrxrdct"
```

### With `pip`

```bash
git clone <repo-url>
cd nrxrdct
pip install -e .
```

### Building the documentation locally

```bash
uv sync --extra docs
uv run mkdocs serve
```

---

## Package structure

```text
src/nrxrdct/
├── __init__.py          # Top-level public API re-exports
├── utils.py             # Baseline fitting, simulation, masking, padding
├── azimuthal/           # pyFAI azimuthal integration
│   ├── integration.py   # 1-D and CAKE integration, parallel pipeline
│   └── slurm_integration/
│       ├── cli.py               # nrxrdct-slurm entry point
│       ├── launch_jobs.py       # SLURM job submission
│       ├── monitor.py           # Job progress monitoring
│       ├── check_output.py      # Output validation and repair
│       ├── merge.py             # Merge tmp files into output HDF5
│       └── integrate_worker.py  # Per-job integration worker
├── fitting/             # Peak fitting and NMF
│   ├── peakfit.py       # 1-D peak fitting utilities
│   └── nmf.py           # NMF decomposition for hyperspectral volumes
├── fluo/                # X-ray fluorescence
│   ├── constants.py     # Default emission line sets
│   └── fluorescence.py  # XRF sinogram loading and fitting
├── rietveld/            # GSAS-II Rietveld refinement
│   ├── refinement.py    # BaseRefinement, InstrumentCalibration wrappers
│   └── refine_dict.py   # Pre-built refinement dictionary templates
├── xrdct/               # XRD-CT pipeline core
│   ├── io.py            # HDF5 and .xy file I/O, GSAS-II instprm export
│   ├── parameters.py    # Scan metadata container (Scan class)
│   ├── preprocessing.py # Zinger removal
│   ├── sinogram.py      # Sinogram assembly and XRF ROI extraction
│   ├── reconstruction.py# ASTRA tomographic reconstruction
│   ├── volume.py        # ReconstructedVolume — per-voxel refinement and maps
│   ├── visualization.py # napari-based interactive viewers
│   └── slurm_reconstruction/
│       └── cli.py       # nrxrdct-slurm-recon entry point
└── laue/                # Polychromatic (Laue) diffraction — self-contained
    ├── camera.py                # Detector geometry (Camera, fit_calibration)
    ├── crystal.py               # Crystal structure builders (CIF, BCC, B2)
    ├── simulation.py            # Forward models (simulate_laue, _stack, _darwin)
    ├── layers.py                # Layered structures, orientation relationships
    ├── segmentation.py          # Spot segmentation and peak fitting pipeline
    ├── fitting.py               # Orientation & strain fitting
    ├── map.py                   # GrainMap — raster micro-Laue map management
    ├── interactive.py           # IPyWidgets orientation / calibration widgets
    ├── laue_plotting.py         # Laue-specific plotting helpers
    ├── slurm_seg_worker.py      # SLURM worker: batch spot segmentation
    ├── slurm_orient_worker.py   # SLURM worker: batch orientation fitting
    └── slurm_strain_worker.py   # SLURM worker: batch strain fitting
```

---

## Typical workflow

### 1. Describe the scan

```python
from pathlib import Path
from nrxrdct.xrdct import Scan

scan = Scan(
    acquisition_file=Path("data/sample.h5"),
    sample_name="my_sample",
    beam_energy=44,     # keV
    beam_size=100e-6,   # m
)
```

### 2. Integrate frames

```python
import numpy as np
from nrxrdct.azimuthal import integrate_powder_parallel

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

### 3. Assemble sinogram and reconstruct

```python
from nrxrdct.xrdct import assemble_sinogram, reconstruct_slice

sino = assemble_sinogram(Path("integrated.h5"), n_rot=901, n_tth_angles=1000)
# sino shape: (n_tth, n_lines, n_rot)

volume = np.stack([
    reconstruct_slice(sino[i], angles_rad=angles)
    for i in range(sino.shape[0])
])
```

### 4. Per-voxel Rietveld refinement

```python
from nrxrdct.xrdct import ReconstructedVolume

tth = np.load("tth.npy")
rv = ReconstructedVolume(
    volume=volume,
    tth_deg=tth,
    sample_name="my_sample",
    phases=["Al"],
    processing_folder=Path("output"),
)
rv.write_xy_files_parallel()
rv.refine_models_parallel(my_refinement_function)  # user-supplied
```

### 5. Instrument calibration

```python
from nrxrdct.rietveld import InstrumentCalibration

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

## Volume analysis

`ReconstructedVolume` provides a full set of map extractors and analysis tools after refinement:

```python
# Goodness-of-fit maps
rwp_map   = rv.get_Rwp_map()
chi2_map  = rv.get_chi2_map()

# Unit-cell maps
a_map, b_map, c_map = rv.get_cell_map()

# Microstructure maps
size_map     = rv.get_crystallite_size_map(phase=0)
strain_map   = rv.get_microstrain_map(phase=0)
scale_map    = rv.get_scale_map(phase=0)

# All maps at once
maps = rv.get_all_maps(phase=0)
# maps keys: 'Rwp', 'chi2', 'a', 'b', 'c', 'alpha', 'beta', 'gamma',
#            'size', 'microstrain', 'scale'

# Per-voxel peak fitting (no GSAS-II required)
rv.fit_peak_map(tth_range=(10.0, 12.0))

# Quick visual overview
rv.plot_maps()

# Interactive Jupyter widget (pick voxel → show diffractogram + live refine)
rv.pick_and_refine_jupyter()
```

A boolean mask can be set to skip empty or background voxels:

```python
rv.mask = volume.max(axis=0) > threshold  # only refine signal voxels
```

---

## HPC / SLURM integration

For large datasets, `nrxrdct` provides a SLURM pipeline that distributes integration across cluster nodes. All functionality is available through a Python API and the `nrxrdct-slurm` CLI.

### Submitting jobs

```python
from pathlib import Path
from nrxrdct.azimuthal.slurm_integration import launch

result = launch(
    master_file = Path("/data/raw/sample_master.h5"),
    output_file = Path("/data/processed/integrated.h5"),
    poni_file   = Path("/data/calib/detector.poni"),
    mask_file   = Path("/data/calib/mask.edf"),
    n_jobs      = 8,          # number of SLURM jobs
    n_points    = 1000,       # radial bins
    n_workers   = 16,         # integration threads per job
    batch_size  = 32,         # frames per batch (RAM control)
    method      = "filter",   # "standard" | "filter" | "sigma_clip"
    percentile  = (10, 90),   # used with method="filter"
    partition   = "nice",
    time        = "04:00:00",
    mem         = "64G",
    cpus        = 16,
    conda_env   = "nrxrdct",  # or env_activate=Path("/path/to/activate")
)

slurm_ids = result["slurm_ids"]
tmp_dir   = result["tmp_dir"]
```

CLI equivalent:

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

### Monitoring progress

```python
from nrxrdct.azimuthal.slurm_integration import monitor

# Single snapshot (non-blocking)
status = monitor(slurm_ids, tmp_dir)

# Block until all jobs finish, polling every 60 s
status = monitor(slurm_ids, tmp_dir, watch=True, interval=60)
```

CLI equivalent:

```bash
nrxrdct-slurm monitor \
    --slurm-ids 12345,12346,12347 \
    --tmp-dir   /data/processed/integrated_tmp

# Blocking watch
nrxrdct-slurm monitor \
    --slurm-ids 12345,12346,12347 \
    --tmp-dir   /data/processed/integrated_tmp \
    --watch --interval 60
```

The monitor prints a live status table:

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

### Checking and repairing output

After all jobs finish, verify that every scan was written correctly:

```python
from nrxrdct.azimuthal.slurm_integration import check, repair

# Report missing scans
result = check(tmp_dir=tmp_dir, output_file=Path("/data/processed/integrated.h5"))

# Resubmit missing scans automatically
repair(
    tmp_dir     = tmp_dir,
    n_jobs      = 2,
    watch       = True,
    partition   = "nice",
    conda_env   = "nrxrdct",
)
```

CLI equivalent:

```bash
nrxrdct-slurm check --output-file /data/processed/integrated.h5

nrxrdct-slurm check \
    --output-file /data/processed/integrated.h5 \
    --repair \
    --master-file /data/raw/sample_master.h5 \
    --poni-file   /data/calib/detector.poni \
    --mask-file   /data/calib/mask.edf \
    --n-jobs 2 --partition nice --conda-env nrxrdct --watch
```

---

## GPU support

`nrxrdct` detects NVIDIA GPUs automatically at import time via `astra.get_gpu_info()`. When a GPU is available, `reconstruct_slice` and `reconstruct_astra_gpu_3d` use CUDA-accelerated algorithms (`SART_CUDA`, `SIRT3D_CUDA`, `CGLS3D_CUDA`). The fallback to CPU (FBP) is transparent — no code changes required.

---

## Laue diffraction

`nrxrdct.laue` is a self-contained subpackage for polychromatic (Laue) diffraction analysis, targeting synchrotron micro-Laue experiments (e.g. BM32 at the ESRF).

### Capabilities

| Area | What is provided |
|---|---|
| **Detector geometry** | `Camera` — pixel ↔ reciprocal-space projection, multi-stage `fit_calibration` |
| **Crystal structures** | `crystal_from_cif`, BCC / B2 builders, vectorised B-matrix for HKL enumeration |
| **Forward simulation** | `simulate_laue` — single grain; `simulate_laue_stack` — thin-film satellites; `simulate_laue_darwin` — dynamical (Darwin) model |
| **Synchrotron spectra** | Bending-magnet and undulator models, KB-mirror reflectivity, Lorentz–polarisation factors |
| **Spot segmentation** | LoG, white-top-hat, and hybrid strategies; 2-D rotated-Gaussian peak fitting; HDF5 spotsfile I/O |
| **Orientation indexing** | `index_orientation` — exhaustive angular search; `interactive_orientation` — IPyWidgets manual indexing |
| **Orientation fitting** | `fit_orientation`, `fit_orientation_stack`, `fit_orientation_mixed` — multi-stage Nelder-Mead |
| **Strain fitting** | `fit_strain_orientation` — simultaneous deviatoric strain tensor + orientation refinement |
| **Layered structures** | `Layer` / `LayeredCrystal` — epitaxial stacks with orientation relationships (K–S, N–W, Pitsch, …) |
| **Grain maps** | `GrainMap` — manages raster micro-Laue maps and SLURM jobs for each analysis step |

### Quick example — single-grain simulation and fitting

```python
import numpy as np
from nrxrdct.laue import (
    Camera, crystal_from_cif,
    simulate_laue, precompute_allowed_hkl,
    fit_orientation,
)

cam = Camera(dd=70.0, xcen=1081.0, ycen=1034.0,
             xbet=0.0, xgam=0.0, pixelsize=0.075,
             n_pix_h=2162, n_pix_v=2068)

crystal = crystal_from_cif("Fe.cif")
allowed = precompute_allowed_hkl(crystal, E_max_eV=30000)

U = np.eye(3)
spots  = simulate_laue(crystal, cam, U, allowed_hkl=allowed)
obs_xy = np.array([[s["pix"][0], s["pix"][1]] for s in spots])

result = fit_orientation(crystal, cam, obs_xy,
                         max_match_px=[10.0, 3.0],
                         allowed_hkl=allowed)
print(f"RMS: {result.rms_px[-1]:.2f} px  matched: {result.n_matched[-1]}/{len(obs_xy)}")
```

### Quick example — layered structure with thin-film satellites

```python
from nrxrdct.laue import (
    Layer, LayeredCrystal, crystal_from_cif,
    or_nishiyama_wassermann, simulate_laue_stack,
)

fe  = crystal_from_cif("Fe.cif")
fen = crystal_from_cif("FeN.cif")

substrate = Layer(fe,  thickness_nm=None, label="Fe substrate")
film      = Layer(fen, thickness_nm=50.0,
                  orientation_relation=or_nishiyama_wassermann(fe, fen),
                  label="FeN film")

stack = LayeredCrystal([substrate, film])
spots = simulate_laue_stack(stack, cam, np.eye(3), allowed_hkl=allowed)
```

### SLURM micro-Laue map pipeline

```python
from pathlib import Path
from nrxrdct.laue import GrainMap

gmap = GrainMap(
    scan_dir   = Path("/data/scan"),
    output_dir = Path("/data/results"),
    crystal    = crystal,
    camera     = cam,
    n_grains   = 2,
)

seg_ids = gmap.submit_segmentation(partition="nice", time="02:00:00",
                                   n_jobs=8, conda_env="nrxrdct")
gmap.monitor(seg_ids)

ub_ids = gmap.submit_orientation(partition="nice", time="04:00:00",
                                 n_jobs=8, conda_env="nrxrdct")
gmap.monitor(ub_ids)

str_ids = gmap.submit_strain(partition="nice", time="06:00:00",
                             n_jobs=8, fit_strain=["e_xx", "e_yy", "e_zz"],
                             conda_env="nrxrdct")
gmap.monitor(str_ids)

grain_map = gmap.collect_strain_map(grain_index=0)
```

---

## Dependencies

### Declared in `pyproject.toml`

| Package | Purpose |
|---|---|
| `numpy` | Array operations throughout |
| `scipy` | Filters, optimisation |
| `h5py` | HDF5 file I/O |
| `matplotlib` | Plotting |
| `pyFAI` | Azimuthal integration |
| `scikit-image` | Segmentation and image processing |
| `scikit-learn` | NMF decomposition |
| `pandas` | Peak table output |
| `xraylib` | XRF emission line energies |
| `xrayutilities` | Powder pattern simulation, peak listing, structure factors |
| `tqdm` | Progress bars |
| `dill` | Serialisation of crystal objects for SLURM workers |
| `pybaselines` | XRD baseline fitting |
| `ipympl` | Interactive Matplotlib in Jupyter |

### Additional runtime dependencies (install separately)

| Package | Used by | Purpose |
|---|---|---|
| `astra-toolbox` | `xrdct.reconstruction` | Tomographic reconstruction (GPU + CPU) |
| `GSAS-II` | `xrdct.volume`, `rietveld` | Rietveld refinement scripting |
| `fabio` | `azimuthal.integration` | Reading mask files (`.edf`, `.cbf`) |
| `hdf5plugin` | `xrdct` | Compressed HDF5 dataset support |
| `napari` | `xrdct.visualization` | Interactive 3-D volume viewer |
| `ipywidgets` | `laue.interactive` | Interactive orientation / calibration widgets |

---

## Acknowledgements

The NMF module (`nrxrdct.fitting.nmf`) was partially developed by **Beatriz G. Foschiani** (CEA Grenoble).
