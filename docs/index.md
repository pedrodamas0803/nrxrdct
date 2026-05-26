# nrxrdct

**Far-field X-ray diffraction computed tomography** — a Python toolkit for the full XRD-CT data-reduction pipeline, from raw detector images to spatially-resolved maps of crystallographic parameters.

---

## Overview

`nrxrdct` covers every step of a synchrotron XRD-CT experiment and is organised into focused subpackages:

### `nrxrdct.azimuthal` — Integration

Parallelised 1-D azimuthal integration and 2-D CAKE regrouping via pyFAI. Supports standard, sigma-clipping, and percentile-filtering strategies for single-crystal spot rejection. Includes a full SLURM pipeline (`slurm_integration`) for distributing integration across cluster nodes, with job submission, progress monitoring, output validation, and automatic repair.

### `nrxrdct.xrdct` — XRD-CT Pipeline Core

End-to-end XRD-CT data reduction:

- **Preprocessing** — hot-pixel (zinger) removal from detector images.
- **Sinogram assembly** — stacking integrated patterns from HDF5 master files with background subtraction and monitor normalisation.
- **Reconstruction** — GPU-accelerated (SIRT3D, CGLS3D, SART, SIRT) and CPU (FBP) reconstruction via the ASTRA Toolbox, with automatic GPU detection.
- **Volume analysis** — `ReconstructedVolume` manages per-voxel `.xy` file I/O and parallelised GSAS-II refinement, with map extractors for Rwp, unit-cell lengths, and crystallite sizes.
- **Visualisation** — napari-based interactive 3-D volume and slice viewers with live Z-profile extraction on pixel click.
- **I/O** — HDF5 and `.xy` file reading/writing, GSAS-II instrument parameter export.

### `nrxrdct.rietveld` — Rietveld Refinement

GSAS-II scripting wrappers (`BaseRefinement`, `InstrumentCalibration`) for sequential step-by-step refinement: background, scale, zero shift, peak broadening, cell parameters, preferred orientation, crystallite size, microstrain, and extinction. Pre-built refinement dictionary templates in `refine_dict`.

### `nrxrdct.fitting` — Peak Fitting and NMF

1-D peak fitting utilities (`peakfit`) and non-negative matrix factorisation of hyperspectral volumes for phase mapping (`HyperspectralNMF`).

### `nrxrdct.fluo` — X-ray Fluorescence

Loading of XRF ROI and full-spectrum sinograms from HDF5 files, emission line look-up via xraylib, and spectral fitting.

### `nrxrdct.laue` — Polychromatic (Laue) Diffraction

A self-contained subpackage for micro-Laue diffraction analysis targeting synchrotron experiments (e.g. BM32 at the ESRF):

- **Forward simulation** — `simulate_laue`, `simulate_laue_stack` (thin-film satellites), `simulate_laue_darwin` (dynamical model for thick layers).
- **Crystal structures** — load from CIF, build BCC/B2 analytically, vectorised B-matrix for fast HKL enumeration.
- **Detector geometry** — `Camera` with multi-stage calibration and gnomonic projection.
- **Spot segmentation** — LoG, white-top-hat, and hybrid strategies with 2-D Gaussian peak fitting.
- **Orientation indexing and fitting** — exhaustive angular search, interactive IPyWidgets indexing, multi-stage Nelder-Mead minimisation.
- **Strain fitting** — simultaneous deviatoric strain tensor and orientation refinement.
- **Layered structures** — `Layer` / `LayeredCrystal` with built-in orientation relationships (K–S, N–W, Pitsch, …).
- **Grain maps** — `GrainMap` for raster micro-Laue maps with SLURM-based segmentation, orientation, and strain steps.

### `nrxrdct.utils` — Utilities

Baseline fitting, powder pattern simulation, circular masking, and peak listing from CIF files.

---

## Quick links

- [Installation](installation.md)
- [Quickstart](user-guide/quickstart.md)
- [Typical Workflow](user-guide/workflow.md)
- [SLURM / HPC Integration](user-guide/slurm.md)
- [API Reference](api/utils.md)

---

## Acknowledgements

The NMF module (`nrxrdct.fitting.nmf`) was partially developed by **Beatriz G. Foschiani** (CEA Grenoble).
