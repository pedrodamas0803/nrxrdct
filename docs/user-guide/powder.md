# Powder Diffraction Simulation

This page describes `nrxrdct.powder`: simulating powder XRD patterns from
CIF files, extracting peak tables, estimating backgrounds, and building
multi-element alloy crystal structures via Vegard's law.

---

## 1. Background estimation — `calculate_xrd_baseline`

Before fitting or simulating peaks it is often necessary to estimate the
diffuse background in a measured pattern.  Five algorithms are available,
all backed by pybaselines:

```python
import numpy as np
from nrxrdct.powder.simulation import calculate_xrd_baseline

tth, I = np.loadtxt("pattern.xy", unpack=True)
baseline, info = calculate_xrd_baseline(I, tth, method="snip")
I_net = I - baseline
```

| `method` | Speed | Description |
|---|---|---|
| `"snip"` | Very fast | Statistics-sensitive Non-linear Iterative Peak-clipping |
| `"iasls"` | Moderate | Iterative asymmetric least squares — robust default |
| `"aspls"` | Moderate | Adaptive smoothness penalized least squares |
| `"arpls"` | Moderate | Asymmetrically reweighted penalized least squares |
| `"mor"` | Fast | Morphological baseline — best for broad smooth backgrounds |

Default tuning arguments are set automatically for XRD data; override
them via `**kwargs`:

```python
baseline, _ = calculate_xrd_baseline(
    I, tth,
    method="iasls",
    lam=1e7,    # smoothness penalty (larger = smoother baseline)
    p=5e-3,     # asymmetry parameter (smaller = hugs the baseline more)
)
```

---

## 2. Powder pattern simulation — `simulate_powder_xrd_monophase`

Simulate a powder XRD pattern from one or more CIF files using
xrayutilities:

```python
import numpy as np
from nrxrdct.powder.simulation import simulate_powder_xrd_monophase

tth = np.linspace(1.0, 30.0, 3000)   # 2θ axis in degrees

intensity = simulate_powder_xrd_monophase(
    tth=tth,
    cif_files=["Fe_bcc.cif", "Fe3C.cif"],
    energy_keV=44.0,
    crystallite_size=200e-9,   # 200 nm — controls Gaussian peak width
    do_plot=True,
    do_save=True,
)
```

| Parameter | Default | Description |
|---|---|---|
| `cif_files` | — | CIF path or list of paths; each phase simulated independently |
| `energy_keV` | `100.0` | X-ray energy in keV |
| `crystallite_size` | `100e-9` | Gaussian crystallite size in metres (Scherrer broadening) |
| `do_plot` | `True` | Produce a matplotlib figure per phase |
| `do_save` | `True` | Save each pattern to `<phase_name>_simulated.xy` |

When multiple CIF files are passed, the function loops over them and
returns the pattern for the **last** phase.  Each pattern is saved and
plotted independently.

---

## 3. Peak position lookup — `get_powder_xrd_peaks`

Extract all allowed reflection positions and Miller indices for one or
more phases:

```python
from nrxrdct.powder.simulation import get_powder_xrd_peaks

peaks = get_powder_xrd_peaks(
    cif_files=["Fe_bcc.cif", "Fe3C.cif"],
    names=["ferrite", "cementite"],      # optional custom labels
    energy_keV=44.0,
    tth_min=1.0,
    tth_max=30.0,
)

# peaks["ferrite"] is a pd.DataFrame with columns: tth, d, hkl, F2, ...
print(peaks["ferrite"].head())
```

This is useful for overlaying expected peak positions on a measured
pattern, or for identifying unindexed reflections.

---

## 4. Multi-element alloy structures — `nrxrdct.powder.structures`

`make_alloy_crystal` builds an xrayutilities `Crystal` for a random
solid-solution alloy with a lattice parameter estimated from Vegard's law
(weighted average of CN-12 metallic radii):

```python
from nrxrdct.powder.structures import make_alloy_crystal

# Equiatomic FeCrMnNi HEA
crystal = make_alloy_crystal(
    composition={"Fe": 0.25, "Cr": 0.25, "Mn": 0.25, "Ni": 0.25},
    structure="FCC",
)
```

Supported structures:

| `structure` | Space group | Elements on site |
|---|---|---|
| `"FCC"` | Fm−3m | All elements on one 4a Wyckoff position |
| `"BCC"` | Im−3m | All elements on one 2a Wyckoff position |
| `"HCP"` | P6₃/mmc | All elements on 2c Wyckoff position |
| `"SC"` | Pm−3m | Simple cubic, one atom per cell |

Override the Vegard-law lattice parameter with an explicit value:

```python
crystal = make_alloy_crystal(
    composition={"Fe": 0.7, "Ni": 0.3},
    structure="FCC",
    latparam=3.595,   # Å — bypasses Vegard's law
)
```

The returned `Crystal` object is compatible with all xrayutilities simulation
functions and with `nrxrdct.laue.crystal` tools.

### 4.1 Listing supported structures

```python
from nrxrdct.powder.structures import list_structures
print(list_structures())   # ['BCC', 'FCC', 'HCP', 'SC']
```

### 4.2 Looking up metallic radii

```python
from nrxrdct.powder.structures import elem_radius
print(elem_radius("Fe"))   # CN-12 metallic radius in Å
```

---

## 5. Combining simulation with fitting

A common workflow is to simulate a pattern for phase identification and
then overlay it on a measured pattern with the background removed:

```python
import numpy as np
import matplotlib.pyplot as plt
from nrxrdct.powder.simulation import (
    calculate_xrd_baseline,
    simulate_powder_xrd_monophase,
    get_powder_xrd_peaks,
)

tth_meas, I_meas = np.loadtxt("measured.xy", unpack=True)
tth_sim  = tth_meas.copy()

# Remove background
baseline, _ = calculate_xrd_baseline(I_meas, tth_meas, method="snip")
I_net = I_meas - baseline

# Simulate
I_sim = simulate_powder_xrd_monophase(
    tth_sim, "Fe_bcc.cif", energy_keV=44, do_plot=False, do_save=False
)
I_sim *= I_net.max() / I_sim.max()   # scale to data

# Peak positions
peaks = get_powder_xrd_peaks("Fe_bcc.cif", energy_keV=44,
                              tth_min=tth_meas.min(), tth_max=tth_meas.max())

fig, ax = plt.subplots()
ax.plot(tth_meas, I_net, label="measured")
ax.plot(tth_sim,  I_sim, label="simulated", alpha=0.7)
for _, row in peaks["Fe_bcc"].iterrows():
    ax.axvline(row["tth"], color="gray", lw=0.5, alpha=0.5)
ax.legend()
ax.set_xlabel("2θ (°)")
```
