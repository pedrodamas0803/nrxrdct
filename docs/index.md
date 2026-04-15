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
11. **Laue simulation** — white-beam Laue diffraction simulator (`Camera`, `simulate_laue`) with BCC/B2 crystal builders, synchrotron spectrum models, and plotting utilities for 2θ/χ and gnomonic projections.

---

## Quick links

- [Installation](installation.md)
- [Quickstart](user-guide/quickstart.md)
- [Typical Workflow](user-guide/workflow.md)
- [SLURM / HPC Integration](user-guide/slurm.md)
- [API Reference](api/parameters.md)

---

## Acknowledgements

The NMF module (`nmf.py`) was partially developed by **Beatriz G. Foschiani** (CEA Grenoble).
