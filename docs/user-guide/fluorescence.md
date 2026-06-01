# X-Ray Fluorescence

This page describes the `nrxrdct.fluo` submodule: looking up XRF emission
lines, building element spectral templates, and fitting fluorescence spectra
or full hyperspectral volumes to extract element intensity maps.

---

## 1. Emission line lookup

`get_fluo_lines` queries the xraylib database and returns the emission line
energies for one or more elements within a given energy window:

```python
from nrxrdct.fluo.fluorescence import get_fluo_lines

lines = get_fluo_lines(
    elements=["Fe", "Cu", "Zn"],
    energy_range=(6.0, 10.0),   # keV
)
# lines["Fe"] is a pd.DataFrame with columns ["line", "energy_keV"]
print(lines["Fe"])
```

| Parameter | Description |
|---|---|
| `elements` | Chemical symbol string or list of strings |
| `energy_range` | `(emin, emax)` in keV; lines outside this range are excluded |
| `names` | Optional list of custom keys for the returned dict |
| `lines` | Emission line names to query; defaults to all standard K, L, M Siegbahn lines |

The function prints a formatted table to stdout and returns a `dict` mapping
each element name to a `pd.DataFrame` with columns `line` and `energy_keV`.

---

## 2. Building element spectral components

`build_element_component` constructs the expected spectral shape for a single
element, weighting each emission line by its fluorescence cross-section:

```python
import numpy as np
from nrxrdct.fluo.fluorescence import build_element_component

energy_axis = np.linspace(4.0, 12.0, 512)  # MCA energy axis in keV

fe_template = build_element_component(
    element="Fe",
    energy_axis=energy_axis,
    excitation_energy=17.5,   # incident beam energy (keV)
    fwhm_keV=0.15,            # detector energy resolution (keV)
)
# fe_template is a normalised 1-D array (max = 1)
```

| Parameter | Description |
|---|---|
| `element` | Chemical symbol, e.g. `"Fe"` |
| `energy_axis` | 1-D MCA energy axis (keV) |
| `excitation_energy` | Incident beam energy in keV — determines which lines are excited |
| `fwhm_keV` | Detector energy resolution (FWHM); typically 0.13–0.25 keV for SDD |

To build templates for multiple elements at once use `build_fit_matrix`:

```python
from nrxrdct.fluo.fluorescence import build_fit_matrix

A = build_fit_matrix(
    elements=["Fe", "Cu", "Zn"],
    energy_axis=energy_axis,
    excitation_energy=17.5,
    fwhm_keV=0.15,
)
# A has shape (len(energy_axis), n_elements)
```

---

## 3. Fitting a single spectrum

`fit_fluo_spectrum` fits a measured MCA spectrum to the element templates
using non-negative least squares (NNLS), returning the intensity coefficient
for each element:

```python
from nrxrdct.fluo.fluorescence import fit_fluo_spectrum

spectrum = np.load("spectrum.npy")   # 1-D MCA spectrum

result = fit_fluo_spectrum(
    spectrum=spectrum,
    energy_axis=energy_axis,
    elements=["Fe", "Cu", "Zn"],
    excitation_energy=17.5,
    fwhm_keV=0.15,
)
# result is a dict: {"Fe": 123.4, "Cu": 56.7, "Zn": 12.3}
```

The NNLS solver guarantees non-negative coefficients, which is physically
correct for fluorescence intensities.

---

## 4. Fitting a hyperspectral volume

`fit_fluo_volume` applies the same fitting to every spatial pixel of a 3-D
data cube `(nx, ny, n_channels)`:

```python
from nrxrdct.fluo.fluorescence import fit_fluo_volume

volume = np.load("fluo_volume.npy")   # shape: (nx, ny, n_channels)

maps = fit_fluo_volume(
    volume=volume,
    energy_axis=energy_axis,
    elements=["Fe", "Cu", "Zn"],
    excitation_energy=17.5,
    fwhm_keV=0.15,
)
# maps["Fe"] is a (nx, ny) array of Fe intensity
```

The fit runs in parallel across spatial pixels using a `ThreadPoolExecutor`.

---

## 5. Typical workflow

```python
import numpy as np
import matplotlib.pyplot as plt
from nrxrdct.fluo.fluorescence import get_fluo_lines, fit_fluo_volume

# 1. Inspect which lines fall in the detector window
energy_axis = np.linspace(4.0, 12.0, 512)
get_fluo_lines(["Fe", "Cu", "Ni"], energy_range=(4.0, 12.0))

# 2. Fit all pixels
volume = np.load("fluo_volume.npy")
maps = fit_fluo_volume(
    volume, energy_axis,
    elements=["Fe", "Cu", "Ni"],
    excitation_energy=17.5,
    fwhm_keV=0.15,
)

# 3. Plot element maps
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, elem in zip(axes, ["Fe", "Cu", "Ni"]):
    im = ax.imshow(maps[elem], origin="upper")
    ax.set_title(elem)
    fig.colorbar(im, ax=ax)
plt.tight_layout()
```

---

## 6. Choosing `fwhm_keV`

The detector resolution directly controls how well overlapping lines from
different elements are separated.  Typical values:

| Detector type | FWHM at 6 keV |
|---|---|
| Si drift detector (SDD) | 0.13–0.16 keV |
| Ge detector | 0.18–0.25 keV |
| CdTe / CZT pixel | 0.25–0.40 keV |

Measure the FWHM from a known emission line in a standard spectrum and pass
that value as `fwhm_keV`.
