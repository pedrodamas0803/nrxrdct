# Typical Workflow

This page describes the end-to-end XRD-CT data-reduction workflow as
implemented in `nrxrdct`, from initial instrument calibration through to
final parameter maps.  It covers the calibration procedure, the five-stage
processing pipeline, and the repository package structure.

> **Prerequisite**: see [Quickstart](quickstart.md) for the minimal five-step
> pipeline without calibration.

---

## 1. Instrument calibration

Before processing experimental data, calibrate the detector geometry using a
known powder standard (e.g. LaB₆).  The `InstrumentCalibration` class wraps a
GSAS-II project and exposes each refinement step individually so that the
sequence can be adapted to the standard and beam conditions.

```python
from pathlib import Path
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

The calibrated instrument parameters are written to an `instprm` file that is
passed to all subsequent per-voxel refinements.

---

## 2. Full pipeline

The five stages run sequentially on a single machine or can be distributed
across an HPC cluster — see [SLURM / HPC Integration](slurm.md).

```
Raw HDF5 frames
    │
    ▼  preprocessing.py  — zinger removal
    │
    ▼  integration.py    — azimuthal integration (pyFAI)
    │
    ▼  reconstruction.py — sinogram assembly + ASTRA reconstruction
    │
    ▼  refinement.py     — per-voxel Rietveld (GSAS-II)
    │
    ▼  Parameter maps    — Rwp, unit-cell, crystallite size, …
```

Each stage writes its output to an HDF5 file that is consumed by the next
stage, so any step can be re-run independently without repeating earlier
computation.

---

## 3. Package structure

```text
src/nrxrdct/
├── __init__.py
├── utils.py                         # Baseline fitting, simulation, masking, padding
├── azimuthal/
│   ├── integration.py               # pyFAI azimuthal integration pipeline
│   └── slurm_integration/           # SLURM-distributed integration
│       ├── cli.py
│       ├── launch_jobs.py
│       ├── integrate_worker.py
│       ├── check_output.py
│       ├── merge.py
│       └── monitor.py
├── fitting/
│   ├── peakfit.py                   # 1-D peak fitting
│   └── nmf.py                       # NMF decomposition for hyperspectral volumes
├── fluo/
│   └── fluorescence.py              # XRF sinogram loading
├── powder/
│   ├── simulation.py                # Powder diffraction simulation
│   └── structures.py                # Crystal structure utilities
├── rietveld/
│   ├── refinement.py                # GSAS-II refinement wrappers
│   └── refine_dict.py               # Pre-built GSAS-II refinement dictionary templates
├── xrdct/
│   ├── parameters.py                # Scan metadata container (Scan class)
│   ├── preprocessing.py             # Zinger removal
│   ├── sinogram.py                  # Sinogram assembly
│   ├── reconstruction.py            # ASTRA reconstruction + ReconstructedVolume
│   ├── volume.py                    # ReconstructedVolume I/O and map extraction
│   ├── visualization.py             # napari-based interactive viewers
│   ├── io.py                        # HDF5 and .xy file I/O, GSAS-II instprm export
│   └── slurm_reconstruction/        # SLURM-distributed reconstruction
│       ├── cli.py
│       ├── launch_recon.py
│       └── reconstruct_worker.py
└── laue/                            # Polychromatic (Laue) diffraction subpackage
    ├── camera.py
    ├── crystal.py
    ├── simulation.py
    ├── layers.py
    ├── segmentation.py
    ├── fitting.py
    ├── fit_io.py
    ├── map.py
    ├── beamline.py
    ├── interactive.py
    └── laue_plotting.py
```
