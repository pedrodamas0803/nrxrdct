# Rietveld Refinement

This page describes how to use `nrxrdct.rietveld` to drive GSAS-II
Rietveld refinements from Python: instrument calibration, sequential
parameter refinement, multi-phase models, and per-voxel refinement as
part of the XRD-CT pipeline.

> **Prerequisite**: GSAS-II must be installed and importable as `GSASIIscriptable`.
> See [Installation](../installation.md) for setup instructions.

---

## 1. Two classes

| Class | Purpose |
|---|---|
| `BaseRefinement` | General Rietveld refinement for any powder pattern |
| `InstrumentCalibration` | Calibrant-specific subclass with dedicated calibration plotting |

Both inherit from `Scan` and accept the same base parameters.

---

## 2. Instrument calibration

Calibrate the detector geometry and instrument profile using a known
standard (e.g. LaB₆, CeO₂):

```python
from pathlib import Path
from nrxrdct.rietveld.refinement import InstrumentCalibration

cal = InstrumentCalibration(
    acquisition_file=Path("data/calib.h5"),
    sample_name="LaB6",
    beam_energy=44,                          # keV
    xy_file=Path("integrated_data.xy"),
    param_file=Path("calibrated_instrument.instprm"),
    tth_lims=(3.0, 25.0),
)

# Create new GSAS-II project and add phase
cal.create_model(gpx_file=Path("calibration/LaB6.gpx"))
cal.add_phase(cif_file=Path("LaB6.cif"), phase_name="LaB6", block_cell=True)

# Sequential refinement
cal.refine_background()
cal.refine_histogram_scale()
cal.refine_zero_shift()
cal.refine_gaussian_broadening(["W"])
cal.refine_lorentzian_broadening(["X", "Y"])
cal.free_and_refine_cell()

# Write calibrated instrument parameters
cal.write_calibrated_instrument_pars()
cal.plot_calibration_results()
```

The calibrated `.instprm` file is consumed by all subsequent per-voxel
refinements.

---

## 3. Sample refinement

### 3.1 Creating a model

```python
from nrxrdct.rietveld.refinement import BaseRefinement

ref = BaseRefinement(
    acquisition_file=Path("data/scan.h5"),
    sample_name="steel",
    beam_energy=44,
    xy_file=Path("integrated_data.xy"),
    param_file=Path("calibrated_instrument.instprm"),
    tth_lims=(3.0, 25.0),
)

ref.create_model(gpx_file=Path("models/steel.gpx"))
ref.add_phase(cif_file=Path("Fe_bcc.cif"), phase_name="ferrite")
ref.add_phase(cif_file=Path("Fe3C.cif"),   phase_name="cementite")
```

### 3.2 Typical sequential refinement sequence

```python
ref.refine_background()
ref.refine_histogram_scale()
ref.refine_zero_shift()
ref.refine_gaussian_broadening(["U", "V", "W"])
ref.refine_lorentzian_broadening(["X", "Y"])
ref.free_and_refine_cell()
ref.refine_phase_content()           # scale factors → weight fractions
ref.refine_crystallite_size()
ref.refine_mustrain()
ref.save()
```

### 3.3 Inspecting results

```python
print(f"Rwp  = {ref.get_Rwp():.2f} %")
print(f"χ²   = {ref.get_chi2():.3f}")
ref.print_refinement_results()
ref.plot_results()
```

---

## 4. Key methods reference

### Background

```python
ref.refine_background(
    background_type="chebyschev-1",  # polynomial type
    n_terms=6,                       # number of terms
    freeze=False,                    # freeze after refining
)
```

### Peak profile

```python
# Gaussian (U, V, W) — instrument and size broadening
ref.refine_gaussian_broadening(["U", "V", "W"])

# Lorentzian (X, Y) — strain broadening
ref.refine_lorentzian_broadening(["X", "Y"])

# Or refine the full profile in one call
ref.refine_peak_profile(gaussian=["W"], lorentzian=["X", "Y"])
```

### Unit cell

```python
ref.free_and_refine_cell(phase="ferrite")   # refine cell for one phase
ref.free_and_refine_cell()                   # all phases
ref.freeze_cell(phase="ferrite")             # lock cell parameters
```

### Microstructure

```python
# Crystallite size (Scherrer/Voigt)
ref.refine_crystallite_size(
    phase="ferrite",
    model="isotropic",       # "isotropic" | "uniaxial" | "generalized"
)

# Microstrain (Voigt peak broadening)
ref.refine_mustrain(
    phase="ferrite",
    model="isotropic",
)
```

### Preferred orientation

```python
ref.refine_preferential_orientation(
    phase="ferrite",
    model="March-Dollase",   # or "SH" for spherical harmonics
)
```

### Absorption

```python
ref.set_absorption(
    phase="ferrite",
    mu_r=0.45,              # µr for Debye–Scherrer geometry
)
```

---

## 5. Backup and restore

```python
# Save a timestamped backup before a risky step
ref.backup_model(label="before_cell")

# List backups
print(ref.list_backups())

# Restore by index or label
ref.restore_backup(0)
ref.restore_backup("before_cell")
```

---

## 6. Multi-phase workflow

For multiphase samples, pass the `phase` argument to target a specific
phase; omit it to apply to all phases simultaneously:

```python
ref.add_phase("austenite.cif", phase_name="austenite")

# Independent cell refinement per phase
ref.free_and_refine_cell(phase="ferrite")
ref.free_and_refine_cell(phase="austenite")

# Phase content (weight fractions from scale factors)
ref.refine_phase_content()
ref.print_HAP_parameters()
```

---

## 7. Per-voxel refinement in XRD-CT

In the XRD-CT pipeline, `BaseRefinement` is used indirectly through
`ReconstructedVolume.refine_models`, which calls a user-supplied
`refining_function` for each voxel `.xy` file:

```python
from nrxrdct.xrdct.volume import ReconstructedVolume
from pathlib import Path

def my_refinement(xy_file):
    ref = BaseRefinement(
        acquisition_file=Path("scan.h5"),
        sample_name="sample",
        xy_file=xy_file,
        param_file=Path("calibrated_instrument.instprm"),
        tth_lims=(3.0, 25.0),
    )
    ref.load_model(Path("models/template.gpx"))
    ref.refine_background()
    ref.free_and_refine_cell()
    ref.save()

vol.refine_models(my_refinement)
```

See [Typical Workflow](workflow.md) for the full pipeline context.

---

## 8. Refinement dictionary templates

`nrxrdct.rietveld.refine_dict` provides pre-built GSAS-II parameter
dictionaries for common cases:

```python
from nrxrdct.rietveld.refine_dict import (
    SIZE_ISO_DICT,
    MUSTRAIN_UNI_DICT,
    SH_DICT,
)
```

These can be passed directly to `gpx.set_refinement` for lower-level
control outside the `BaseRefinement` API.
