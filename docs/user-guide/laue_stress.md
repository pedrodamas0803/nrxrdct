# Elastic stress from Laue strain maps

This page covers the conversion of fitted Laue strain maps to elastic stress
maps using Hooke's law, the required elastic-constant input, the Voigt
notation conventions used internally, and the available reference-frame
transformations.

> **Prerequisite**: stress calculation requires per-pixel strain data.  Run
> `fit_strain_orientation` and collect results into a `GrainMap` before
> proceeding.  See [Strain Analysis](laue_strain.md) for that workflow.

---

## 1. Hooke's law

In the linear-elastic regime the Cauchy stress tensor
$\boldsymbol{\sigma}$ and the strain tensor $\boldsymbol{\varepsilon}$ are
related by the **generalised Hooke's law**:

$$
\sigma_{ij} = C_{ijkl}\,\varepsilon_{kl}
$$

where $C_{ijkl}$ is the **fourth-rank elastic stiffness tensor** (24 independent
components for the most general triclinic crystal; far fewer for higher
symmetries).

In **Voigt notation** the rank-4 tensor is reduced to a symmetric $6\times6$
matrix $\mathbf{C}$ and the law becomes a simple matrix–vector product:

$$
\boldsymbol{\sigma}_V = \mathbf{C}\,\boldsymbol{\varepsilon}_V^{(\text{eng})}
$$

where $\boldsymbol{\varepsilon}_V^{(\text{eng})}$ uses the **engineering** shear
convention (factor-of-two on off-diagonal components):

$$
\boldsymbol{\varepsilon}_V^{(\text{eng})}
= [\,\varepsilon_{xx},\; \varepsilon_{yy},\; \varepsilon_{zz},\;
    \underbrace{2\varepsilon_{yz}}_{\gamma_{yz}},\;
    \underbrace{2\varepsilon_{xz}}_{\gamma_{xz}},\;
    \underbrace{2\varepsilon_{xy}}_{\gamma_{xy}}\,]
$$

The factor of 2 on the shear components is required for $\mathbf{C}$ to be the
same matrix as in the rank-4 contraction.  The strain values stored by `nrxrdct`
are *tensor* components (no factor of 2), so the code multiplies the off-diagonal
elements by 2 internally before applying $\mathbf{C}$.

---

## 2. Voigt ordering conventions

Two Voigt orderings are in common use:

| Index | Standard crystallographic | `nrxrdct` internal |
|---|---|---|
| 0 | $xx$ | $xx$ |
| 1 | $yy$ | $yy$ |
| 2 | $zz$ | $zz$ |
| 3 | $yz$ | $xy$ |
| 4 | $xz$ | $xz$ |
| 5 | $xy$ | $yz$ |

The standard crystallographic ordering (used by xrayutilities `crystal.cij`)
is $[xx, yy, zz, yz, xz, xy]$.  `nrxrdct` stores strain and stress internally
in the ordering $[xx, yy, zz, xy, xz, yz]$.

The permutation between the two is $[0, 1, 2, 5, 4, 3]$, which is its own
inverse — the same index array maps in both directions.  The code applies this
reordering automatically; users supply and receive arrays in the
`nrxrdct`-internal ordering unless working with `cij` directly.

---

## 3. Elastic constants

### From an xrayutilities `Crystal` object

```python
import xrayutilities as xu
fe = xu.materials.Fe    # body-centred cubic iron

# The stiffness matrix is in crystal.cij  (6×6, Pa)
# nrxrdct converts to GPa automatically
print(fe.cij * 1e-9)    # GPa
```

Pass the crystal object directly to `stress_voigt` or `plot_stress_component`
and the stiffness matrix is extracted automatically:

```python
sig = gmap.stress_voigt(fe, grain=gi_merged)
```

Resolution order inside `_extract_cij`:

1. Explicit `cij` keyword (6×6 array, GPa, standard Voigt ordering).
2. `crystal.cij` — xrayutilities `Crystal` attribute (converted Pa → GPa when
   `max(cij) > 1e6`).
3. `crystal.cijkl` — full rank-4 tensor (contracted to Voigt + converted).

### Supplying elastic constants manually

If the crystal object does not carry `cij`, or you want to use literature
values, pass a `(6, 6)` array directly:

```python
# Cubic iron (GPa), standard Voigt ordering [xx,yy,zz,yz,xz,xy]
C_Fe = np.array([
    [230, 135, 135,   0,   0,   0],
    [135, 230, 135,   0,   0,   0],
    [135, 135, 230,   0,   0,   0],
    [  0,   0,   0, 117,   0,   0],
    [  0,   0,   0,   0, 117,   0],
    [  0,   0,   0,   0,   0, 117],
], dtype=float)

sig = gmap.stress_voigt(None, grain=0, cij=C_Fe)
```

---

## 4. `stress_voigt` — full stress map

```python
sig = gmap.stress_voigt(
    crystal,           # xrayutilities Crystal, or None if cij is given
    grain=gi_merged,   # grain slot index
    cij=None,          # optional explicit (6,6) stiffness matrix in GPa
    frame="crystal",   # "crystal" | "lab" | "sample"
    sample_tilt_deg=-40.0,
    sample_tilt_axis="y",
)
# sig shape: (ny, nx, 6)  —  GPa, code ordering [s_xx,s_yy,s_zz,s_xy,s_xz,s_yz]
```

`NaN` is returned at pixels where no strain data exist.

### Reference frames

| `frame` | Meaning |
|---|---|
| `"crystal"` | Stress components in the crystal coordinate system (same frame as the fitted strain). |
| `"lab"` | Rotated to the lab frame via $\boldsymbol{\sigma}_\text{lab} = \mathbf{U}\,\boldsymbol{\sigma}_\text{crystal}\,\mathbf{U}^T$. |
| `"sample"` | Lab frame further rotated by `sample_tilt_deg` about `sample_tilt_axis` (default −40° about $Y$). |

The transformation between frames follows the standard second-rank tensor
rotation law.  See [Strain Analysis — Reference frames](laue_strain.md#reference-frames)
for the equivalent derivation for the strain tensor.

---

## 5. `plot_stress_component` — single-component map

```python
fig, ax = gmap.plot_stress_component(
    component="s_xx",       # one of s_xx, s_yy, s_zz, s_xy, s_xz, s_yz
    grain=gi_merged,
    crystal=fe,
    frame="crystal",
    scale=1e3,              # GPa → MPa (default)
    cmap="RdBu_r",          # diverging map centred at zero
    vmin=-200, vmax=200,    # MPa
    motor_x="pz", motor_y="py",
    motor_units={"pz": "mm", "py": "mm"},
)
```

Available component strings:

| String | Symbol | Meaning |
|---|---|---|
| `"s_xx"` | $\sigma_{xx}$ | Normal stress along crystal $x$ |
| `"s_yy"` | $\sigma_{yy}$ | Normal stress along crystal $y$ |
| `"s_zz"` | $\sigma_{zz}$ | Normal stress along crystal $z$ |
| `"s_xy"` | $\sigma_{xy}$ | In-plane shear $xy$ |
| `"s_xz"` | $\sigma_{xz}$ | Shear $xz$ |
| `"s_yz"` | $\sigma_{yz}$ | Shear $yz$ |

The default colour scale is `"RdBu_r"` with units of MPa (`scale=1e3`).
Use `scale=1.0` for GPa.

---

## 6. Extracting components manually

```python
sig = gmap.stress_voigt(fe, grain=0, frame="crystal")  # (ny, nx, 6)

# Code ordering: [s_xx=0, s_yy=1, s_zz=2, s_xy=3, s_xz=4, s_yz=5]
s_xx = sig[..., 0] * 1e3   # MPa
s_zz = sig[..., 2] * 1e3
s_xy = sig[..., 3] * 1e3
```

---

## 7. Important caveats

### Hydrostatic stress is unconstrained

White-beam Laue diffraction cannot measure the **hydrostatic** part of the
strain tensor (see [Strain Analysis — Hydrostatic blind spot](laue_strain.md#limitation-of-polychromatic-laue-the-hydrostatic-blind-spot)).
Because only the five deviatoric strain components are reliably determined,
the stress tensor derived from them also reflects only the **deviatoric** stress:

$$
\boldsymbol{\sigma} = \mathbf{C} : \boldsymbol{\varepsilon}_\text{deviatoric}
$$

The **mean stress** (pressure) $p = -\tfrac{1}{3}\operatorname{tr}(\boldsymbol{\sigma})$
is not well-constrained.  Individual normal stresses ($\sigma_{xx}$, $\sigma_{yy}$,
$\sigma_{zz}$) contain contributions from both the deviatoric and hydrostatic
parts, so they carry larger absolute uncertainties than the shear components or
the differences between normal components.

In practice:

* **Shear stresses** $\sigma_{xy}$, $\sigma_{xz}$, $\sigma_{yz}$ are the most
  reliable outputs — they are directly proportional to the corresponding shear
  strain components, which *are* measurable.
* **Differences** of normal stresses (e.g. $\sigma_{xx} - \sigma_{zz}$) are
  reliable — they probe the deviatoric part.
* **Individual** normal stresses and the mean stress should be interpreted with
  caution unless the hydrostatic component is constrained by an independent
  measurement.

### Elastic constants must match the crystal frame

The stiffness matrix $\mathbf{C}$ must be expressed in the same coordinate
system as the strain tensor — i.e. the **crystal frame** as defined by the
xrayutilities `Crystal` object.  If you supply `cij` manually, make sure the
axes of your stiffness matrix match the crystal axes used in the simulation.

### Single-crystal assumption

Hooke's law as applied here assumes the crystal is **elastically homogeneous**
within each fitted pixel.  For polycrystalline pixels (overlapping grains,
sub-grain boundaries) the result is an effective average that may not correspond
to the stress in any single crystallite.

---

## 8. Complete example

```python
import xrayutilities as xu
from nrxrdct.laue import GrainMap

# Load a pre-fitted map
gmap = GrainMap.load("grainmap.h5")

# Crystal object with elastic constants
fe = xu.materials.Fe

# Merge grains
best_grain, metrics = gmap.merge(metric="match_rate", min_match_rate=0.3)
gi = gmap.apply_merge(best_grain, metrics)

# Plot all normal-stress components in the crystal frame (MPa)
for comp in ("s_xx", "s_yy", "s_zz"):
    gmap.plot_stress_component(
        comp, grain=gi, crystal=fe,
        frame="crystal", scale=1e3,
        vmin=-500, vmax=500,
    )

# Plot shear stress in the sample frame
gmap.plot_stress_component(
    "s_xz", grain=gi, crystal=fe,
    frame="sample", sample_tilt_deg=-40.0,
    scale=1e3, cmap="RdBu_r",
)

# Access the full stress array for custom analysis
sig_MPa = gmap.stress_voigt(fe, grain=gi, frame="crystal") * 1e3
# Biaxial stress proxy: (σ_xx + σ_yy) / 2
biaxial = (sig_MPa[..., 0] + sig_MPa[..., 1]) / 2
```
