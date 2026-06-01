# Peak Fitting and NMF Decomposition

This page describes the two analysis tools in `nrxrdct.fitting`: single-peak
profile fitting for 1-D powder patterns and non-negative matrix factorisation
(NMF) for 3-D hyperspectral volumes.

---

## 1. Single-peak fitting

### 1.1 Quick start

```python
import numpy as np
from nrxrdct.fitting.peakfit import fit_peak

tth, I = np.loadtxt("pattern.xy", unpack=True)

result = fit_peak(
    tth, I,
    center=3.56,   # nominal peak position (°)
    window=0.4,    # total window width (°)
    model="pseudo_voigt",
    bg_method="snip",
)

print(f"centre = {result['center']:.4f} °")
print(f"FWHM   = {result['fwhm']:.4f} °")
print(f"area   = {result['area']:.1f}")
print(f"R²     = {result['r2']:.4f}")
```

### 1.2 How it works

```
1-D pattern
    │
    ▼  extract_window()   — clip to ±window/2 around center
    │
    ▼  calculate_xrd_baseline()  — estimate and subtract background
    │
    ▼  curve_fit()        — fit chosen peak profile
    │
    ▼  result dict        — center, FWHM, area, R², …
```

Background is estimated and subtracted before fitting, so no prior
background correction is required.

### 1.3 Peak profile models

| `model` | Parameters | Notes |
|---|---|---|
| `"gaussian"` | `amplitude`, `center`, `sigma` | Fastest; good for well-resolved peaks |
| `"lorentzian"` | `amplitude`, `center`, `gamma` | Heavier tails; good for strain-broadened peaks |
| `"voigt"` | `amplitude`, `center`, `sigma`, `gamma` | Convolution of Gaussian + Lorentzian |
| `"pseudo_voigt"` | `amplitude`, `center`, `fwhm`, `eta` | Linear mix: η·Lorentzian + (1−η)·Gaussian (default) |

`pseudo_voigt` is the default and recommended starting point — it is
numerically cheaper than the true Voigt and fits most synchrotron peaks well.

### 1.4 Result dictionary keys

| Key | Present for | Description |
|---|---|---|
| `"center"` | all | Refined peak position (°) |
| `"amplitude"` | all | Peak height above background |
| `"fwhm"` | all | Full-width at half-maximum (°) |
| `"area"` | all | Integrated area (intensity · °) |
| `"residual"` | all | RMS residual |
| `"r2"` | all | Coefficient of determination R² |
| `"success"` | all | `True` if optimiser converged |
| `"sigma"` | Gaussian, Voigt | Gaussian standard deviation (°) |
| `"gamma"` | Lorentzian, Voigt | Lorentzian half-width (°) |
| `"eta"` | pseudo-Voigt | Lorentzian mixing fraction ∈ [0, 1] |

On convergence failure all numeric values are `nan` and `"success"` is
`False`.

### 1.5 Background methods

`bg_method` is forwarded to `calculate_xrd_baseline` from
`nrxrdct.powder.simulation`:

| `bg_method` | Speed | Best for |
|---|---|---|
| `"snip"` | Very fast | Quick scans, sparse peaks (default) |
| `"iasls"` | Moderate | Dense peaks, asymmetric background |
| `"aspls"` | Moderate | Strongly asymmetric patterns |
| `"arpls"` | Moderate | Similar to iasls, alternative tuning |
| `"mor"` | Fast | Morphological baseline, very broad features |

Pass extra tuning arguments via `bg_kwargs`:

```python
result = fit_peak(
    tth, I, center=5.1, window=0.5,
    bg_method="iasls",
    bg_kwargs={"lam": 1e7, "p": 5e-3},
)
```

### 1.6 Loading from file

```python
from nrxrdct.fitting.peakfit import fit_peak_from_file

result = fit_peak_from_file(
    "pattern.xy",
    center=3.56,
    window=0.4,
    model="pseudo_voigt",
)
```

### 1.7 Windowing utility

```python
from nrxrdct.fitting.peakfit import extract_window

tth_w, I_w = extract_window(tth, I, center=3.56, window=0.4)
```

Returns only the points within `[center − window/2, center + window/2]`.
Useful for inspecting a region before fitting.

---

## 2. NMF decomposition for hyperspectral volumes

`HyperspectralNMF` decomposes a 3-D data cube
`(nx, ny, n_channels)` into *K* spatial component maps and *K* spectral
signatures using scikit-learn's non-negative matrix factorisation.

### 2.1 Quick start

```python
import numpy as np
from nrxrdct.fitting.nmf import HyperspectralNMF

# volume shape: (nx, ny, n_channels)
volume = np.load("xrdct_volume.npy")
energy = np.linspace(5.0, 25.0, volume.shape[-1])   # keV axis

nmf = HyperspectralNMF(
    volume,
    n_components=4,
    spectral_axis=energy,
    unit_name="energy (keV)",
)
nmf.fit_data()
nmf.plot()
```

### 2.2 Key attributes after `fit_data()`

| Attribute | Shape | Description |
|---|---|---|
| `W_maps` | `(nx, ny, K)` | Spatial intensity map per component |
| `H` | `(K, n_channels)` | Spectral signature per component |
| `X_rec` | `(n_pixels, n_channels)` | Reconstructed (denoised) data |
| `E_map` | `(nx, ny)` | Per-pixel reconstruction RMSE |
| `model` | — | Fitted `sklearn.decomposition.NMF` object |

### 2.3 Constructor parameters

| Parameter | Default | Description |
|---|---|---|
| `n_components` | — | Number of NMF components K |
| `spectral_axis` | — | 1-D energy/2θ axis for the spectral dimension |
| `unit_name` | `"energy (keV)"` | Axis label for plots |
| `loss_function` | `"frobenius"` | `"frobenius"` or `"kullback-leibler"` |
| `init` | `"nndsvdar"` | Initialisation — see scikit-learn NMF docs |
| `max_iter` | `1000` | Maximum NMF iterations |
| `alpha_W` | `0.0` | L1/L2 regularisation on spatial maps |
| `alpha_H` | `0.0` | L1/L2 regularisation on spectra |
| `clip_negative` | `True` | Zero-clip volume before factorisation |

### 2.4 Choosing the number of components

There is no automatic method; a practical approach is to run the
decomposition for *K* = 2, 3, 4, … and inspect `nmf.E_map` — the
reconstruction RMSE drops sharply when *K* is sufficient to describe the
data, then levels off.

```python
import matplotlib.pyplot as plt

rmse = []
for k in range(2, 8):
    m = HyperspectralNMF(volume, n_components=k, spectral_axis=energy)
    m.fit_data()
    rmse.append(m.E_map.mean())

plt.plot(range(2, 8), rmse, 'o-')
plt.xlabel("Number of components K")
plt.ylabel("Mean RMSE")
```
