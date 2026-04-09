# Typical Workflow

## Instrument calibration

Before processing experimental data, calibrate detector geometry using a known standard (e.g. LaB₆):

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

## Full pipeline

```
Raw HDF5 frames
    │
    ▼ preprocessing.py  — zinger removal
    │
    ▼ integration.py    — azimuthal integration (pyFAI)
    │
    ▼ reconstruction.py — sinogram assembly + ASTRA reconstruction
    │
    ▼ refinement.py     — per-voxel Rietveld (GSAS-II)
    │
    ▼ Parameter maps    — Rwp, unit-cell, crystallite size, …
```

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
