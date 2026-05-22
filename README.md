# nrxrdct

**Far-field X-ray diffraction computed tomography** — a Python toolkit for the full XRD-CT data-reduction pipeline, from raw detector images to spatially-resolved maps of crystallographic parameters.

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
11. **Laue diffraction** — a self-contained sub-package (`nrxrdct.laue`) covering the full micro-Laue analysis pipeline: spot segmentation, orientation fitting, strain mapping, and SLURM-based map processing (see [Laue diffraction](#laue-diffraction) below).

---

## Installation

### Requirements

- Python **3.11** or later
- A working [GSAS-II](https://gsas-ii.readthedocs.io/) installation (scripting interface `GSASIIscriptable` must be importable)
- [ASTRA Toolbox](https://astra-toolbox.com/) (GPU reconstruction requires an NVIDIA GPU and CUDA)

### With `uv` (recommended)

[`uv`](https://docs.astral.sh/uv/) is a fast Python package and project manager. If you do not have it installed yet:

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

After installing, restart your terminal so the `uv` command is on your `PATH`, then clone and install the project:

```bash
git clone <repo-url>
cd nrxrdct
uv sync
```

`uv sync` creates an isolated virtual environment in `.venv/`, installs all declared dependencies, and installs `nrxrdct` itself in editable mode — no separate `pip install -e .` step is needed.

To activate the environment manually:

```bash
# Linux / macOS
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

Alternatively, prefix any command with `uv run` to execute it inside the managed environment without activating it:

```bash
uv run python my_script.py
uv run jupyter lab
```

**Registering a Jupyter kernel (ESRF Jupyter-SLURM)**

On shared Jupyter servers such as the ESRF Jupyter-SLURM portal, you need to register the environment as a named kernel so it appears in the launcher:

```bash
# Install ipykernel into the uv-managed environment
uv add ipykernel

# Register the kernel — the --name must be unique on the server
uv run python -m ipykernel install --user --name nrxrdct --display-name "nrxrdct"
```

After refreshing the JupyterLab page the kernel **nrxrdct** will appear in the kernel selector. To remove it later:

```bash
jupyter kernelspec remove nrxrdct
```

### With `pip`

```bash
git clone <repo-url>
cd nrxrdct
pip install -e .
```

**Registering a Jupyter kernel (ESRF Jupyter-SLURM)**

```bash
pip install ipykernel
python -m ipykernel install --user --name nrxrdct --display-name "nrxrdct"
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
| `dill` | `laue` | Crystal object serialisation for SLURM workers |
| `ipywidgets` | `laue` | Interactive orientation / calibration widgets |

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
├── io.py                # HDF5 and .xy file I/O, GSAS-II instprm export
└── laue/
    ├── __init__.py              # Laue sub-package public API
    ├── camera.py                # Detector geometry (Camera, CalibrationResult, fit_calibration)
    ├── crystal.py               # Crystal structure builders (from CIF, BCC, B2)
    ├── simulation.py            # Laue forward models (simulate_laue, _stack, _darwin)
    ├── layers.py                # Layered structures, orientation relationships
    ├── segmentation.py          # Spot segmentation and peak fitting pipeline
    ├── fitting.py               # Orientation & strain fitting (fit_orientation, fit_strain_orientation)
    ├── map.py                   # GrainMap — raster micro-Laue map management
    ├── interactive.py           # Interactive orientation / calibration widgets
    ├── laue_plotting.py         # Laue-specific plotting helpers
    ├── slurm_seg_worker.py      # SLURM worker: batch spot segmentation
    ├── slurm_orient_worker.py   # SLURM worker: batch orientation fitting
    └── slurm_strain_worker.py   # SLURM worker: batch strain fitting
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

## Laue diffraction

`nrxrdct.laue` is a self-contained sub-package for polychromatic (Laue) diffraction analysis, targeting synchrotron micro-Laue experiments (e.g. BM32 at the ESRF).

### Key capabilities

| Area | What is provided |
|---|---|
| **Detector geometry** | `Camera` — pixel ↔ reciprocal-space projection, multi-stage `fit_calibration` with optional staged `max_match_px` refinement |
| **Crystal structures** | Load from CIF (`crystal_from_cif`), build BCC / B2 analytically, vectorised B-matrix for fast HKL enumeration |
| **Forward simulation** | `simulate_laue` — single grain, full spectrum; `simulate_laue_stack` — epitaxial layer stacks with thin-film satellites; `simulate_laue_darwin` — dynamical (Darwin) model for thick layers |
| **Synchrotron spectra** | Bending-magnet and undulator spectral models, KB-mirror reflectivity, Lorentz–polarisation factors |
| **Spot segmentation** | Laplacian-of-Gaussian (`LoG_segmentation`), white-top-hat (`WTH_segmentation`), and hybrid strategies; 2-D rotated-Gaussian peak fitting; HDF5 spotsfile I/O |
| **Orientation indexing** | `index_orientation` — exhaustive angular search; `interactive_orientation` — IPyWidgets-based manual indexing |
| **Orientation fitting** | `fit_orientation`, `fit_orientation_stack`, `fit_orientation_mixed` — multi-stage Nelder-Mead minimisation with vectorised residuals |
| **Strain fitting** | `fit_strain_orientation` — simultaneous deviatoric strain tensor + orientation refinement |
| **Layered structures** | `Layer` / `LayeredCrystal` — epitaxial stacks with built-in orientation relationships (K–S, N–W, Pitsch, Baker–Nutting, …) and pseudomorphic d-spacing calculations |
| **Grain maps** | `GrainMap` — manages raster micro-Laue maps; submits and monitors SLURM jobs for segmentation, orientation, and strain steps |
| **Batch conversion** | `convert_spotsfiles_to_dat` — parallel (multi-process) conversion of HDF5 spotsfiles to `.dat` peaklist files |

### Quick example — single-grain simulation and fitting

```python
import numpy as np
from nrxrdct.laue import (
    Camera, crystal_from_cif,
    simulate_laue, precompute_allowed_hkl,
    fit_orientation,
)

# 1. Detector geometry
cam = Camera(dd=70.0, xcen=1081.0, ycen=1034.0,
             xbet=0.0, xgam=0.0, pixelsize=0.075,
             n_pix_h=2162, n_pix_v=2068)

# 2. Crystal from CIF
crystal = crystal_from_cif("Fe.cif")
allowed = precompute_allowed_hkl(crystal, E_max_eV=30000)

# 3. Forward simulation (geometry-only, vectorised)
U = np.eye(3)   # orientation matrix
spots = simulate_laue(crystal, cam, U, allowed_hkl=allowed)
obs_xy = np.array([[s["pix"][0], s["pix"][1]] for s in spots])

# 4. Orientation fit
result = fit_orientation(crystal, cam, obs_xy,
                         max_match_px=[10.0, 3.0],
                         allowed_hkl=allowed)
print(f"RMS residual: {result.rms_px[-1]:.2f} px  "
      f"matched: {result.n_matched[-1]}/{len(obs_xy)}")
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

stack  = LayeredCrystal([substrate, film])
U_sub  = np.eye(3)
spots  = simulate_laue_stack(stack, cam, U_sub, allowed_hkl=allowed)
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

# 1. Segment all frames
seg_ids = gmap.submit_segmentation(partition="nice", time="02:00:00",
                                   n_jobs=8, conda_env="nrxrdct")
gmap.monitor(seg_ids)

# 2. Fit orientations
ub_ids = gmap.submit_orientation(partition="nice", time="04:00:00",
                                  n_jobs=8, conda_env="nrxrdct")
gmap.monitor(ub_ids)

# 3. Fit strain
str_ids = gmap.submit_strain(partition="nice", time="06:00:00",
                              n_jobs=8, fit_strain=["e_xx","e_yy","e_zz"],
                              conda_env="nrxrdct")
gmap.monitor(str_ids)

# 4. Collect results
grain_map_array = gmap.collect_strain_map(grain_index=0)
```

### Additional runtime dependencies for `nrxrdct.laue`

| Package | Purpose |
|---|---|
| `xrayutilities` | Crystal structure and structure-factor calculations |
| `dill` | Serialisation of crystal objects for SLURM workers |
| `ipywidgets` | Interactive orientation / calibration widgets |
| `scipy` | Optimisation (`minimize`), image filters |
| `scikit-image` | Segmentation morphology helpers |

---

## Acknowledgements

The NMF module (`nmf.py`) was partially developed by **Beatriz G. Foschiani** (CEA Grenoble).
