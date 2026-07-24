# Azimuthal Integration

This page describes how `nrxrdct` converts raw 2-D detector frames into
1-D powder patterns using pyFAI, which three integration strategies are
available, when to use each, and how to run bulk integration in parallel.

> **Prerequisite**: a PONI calibration file produced by pyFAI-calib2 or an
> equivalent tool.  See [Quickstart](quickstart.md) for the full pipeline
> context.

---

## 1. Overview

Azimuthal integration collapses the 2-D intensity ring pattern on the
detector into a 1-D pattern as a function of scattering angle 2θ (or
momentum transfer *q*).  The three functions differ in how they handle
diffraction spots from single-crystal grains or hot pixels embedded in
the powder signal:

```
Raw 2-D frame
    │
    ▼  (optional) dark/flat correction
    │
    ▼  integration method
    │
    ▼  1-D pattern  (radial, intensity, sigma)
```

All functions share the same output signature:

```python
radial, intensity, sigma = integrate_*(image, poni_file, ...)
```

The `AzimuthalIntegrator` is cached per PONI file, so calling any
function in a loop over thousands of frames does not reload the detector
geometry.

---

## 2. Integration methods

### 2.1 Basic integration — `azimuthal_integration_1d`

```python
from nrxrdct.azimuthal.integration import azimuthal_integration_1d

radial, I, sigma = azimuthal_integration_1d(
    image,
    poni_file="detector.poni",
    npt=1000,
    unit="2th_deg",
    mask=mask,
    dark=dark,
    flat=flat,
    error_model="poisson",
)
```

Internally calls pyFAI's `separate()` to isolate the powder background
from single-crystal spots, then integrates the background component.
Use this when the frame contains a strong powder ring and the
crystalline spots are clearly separable.

| Parameter | Default | Description |
|---|---|---|
| `npt` | `1000` | Number of radial bins |
| `unit` | `"2th_deg"` | `"2th_deg"`, `"q_A^-1"`, `"q_nm^-1"`, `"2th_rad"`, `"r_mm"` |
| `mask` | `None` | 2-D array, `1` = masked, `0` = valid |
| `dark` | `None` | Dark-current image |
| `flat` | `None` | Flat-field image |
| `error_model` | `None` | `"poisson"` for photon-counting detectors |
| `radial_range` | `None` | `(min, max)` in chosen unit |
| `azimuth_range` | `None` | `(min, max)` in degrees |

---

### 2.2 Sigma-clipping — `azimuthal_integration_1d_sigma_clip`

```python
from nrxrdct.azimuthal.integration import azimuthal_integration_1d_sigma_clip

radial, I, sigma = azimuthal_integration_1d_sigma_clip(
    image,
    poni_file="detector.poni",
    npt=1000,
    thres=3.0,
    max_iter=5,
    error_model="hybrid",
)
```

Iteratively removes pixels whose intensity deviates more than `thres`
standard deviations from the bin mean, then integrates the surviving
pixels.  Best for frames where Bragg spots from secondary phases or
single crystals contaminate the powder rings.

| Parameter | Default | Description |
|---|---|---|
| `thres` | `3.0` | Sigma-clipping threshold (standard deviations) |
| `max_iter` | `5` | Maximum clipping iterations |
| `error_model` | `"hybrid"` | `"hybrid"` (Poisson + readout) or `"poisson"` |

**When to use**: polycrystalline samples with occasional strong
single-crystal reflections; frames where `separate()` clips too
aggressively and removes real powder signal.

---

### 2.3 Percentile filtering — `azimuthal_integration_1d_filter`

```python
from nrxrdct.azimuthal.integration import azimuthal_integration_1d_filter

radial, I, sigma = azimuthal_integration_1d_filter(
    image,
    poni_file="detector.poni",
    npt=1000,
    percentile=(0, 99),
)
```

Within each radial bin, discards pixels outside the `(low, high)`
percentile range before averaging.  Simpler than sigma-clipping — no
iterative loop — and robust against outliers when the spot fraction
per bin is below `(100 - high) %`.

| Parameter | Default | Description |
|---|---|---|
| `percentile` | `(0, 99)` | `(low, high)` percentile cutoffs per bin |

**When to use**: fast batch processing where a fixed percentile cutoff
is acceptable; frames with well-isolated hot pixels.

---

### 2.4 Choosing a method

| Situation | Recommended |
|---|---|
| Pure powder, no single-crystal spots | `azimuthal_integration_1d` |
| Mixture with occasional strong Bragg spots | `azimuthal_integration_1d_sigma_clip` |
| Fast scan, simple outlier rejection | `azimuthal_integration_1d_filter` |

---

## 3. 2-D CAKE integration

`cake_integration` produces a 2-D (azimuth × radial) map that preserves
texture information:

```python
from nrxrdct.azimuthal.integration import cake_integration

cake, radial, azimuthal = cake_integration(
    image,
    poni_file="detector.poni",
    npt_rad=1000,
    npt_azim=360,
    unit="2th_deg",
    mask=mask,
)
```

The returned `cake` array has shape `(npt_azim, npt_rad)`.  Plot it
with `plot_integrated_cake` from `nrxrdct.visualization`:

```python
from nrxrdct.visualization import plot_integrated_cake
plot_integrated_cake(cake, radial, azimuthal)
```

---

## 4. Parallel batch integration — `integrate_powder_parallel`

For large scans (thousands of frames) run all integrations in parallel
using a `ProcessPoolExecutor`:

```python
from nrxrdct.azimuthal.integration import integrate_powder_parallel

integrate_powder_parallel(
    master_file="data/scan.h5",
    output_file="integrated.h5",
    poni_file="detector.poni",
    mask_file="mask.npy",
    rot=90,               # detector rotation in degrees
    n_points=1000,
    n_workers=16,
)
```

| Parameter | Default | Description |
|---|---|---|
| `n_points` | `1000` | Radial bins per pattern |
| `n_workers` | `16` | Parallel worker processes |
| `rot` | — | Detector rotation correction (degrees) |

Results are written to `output_file` as an HDF5 dataset consumable by
`assemble_sinogram`.  For HPC / SLURM clusters see [SLURM Integration](slurm.md).

---

## 5. Output units

| `unit` string | Axis label |
|---|---|
| `"2th_deg"` | 2θ (degrees) |
| `"2th_rad"` | 2θ (radians) |
| `"q_A^-1"` | *q* (Å⁻¹) |
| `"q_nm^-1"` | *q* (nm⁻¹) |
| `"r_mm"` | Radius on detector (mm) |

Convert between 2θ and *q* using $q = \frac{4\pi\sin\theta}{\lambda}$.
