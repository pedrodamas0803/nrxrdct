# Elastic stress from Laue strain maps

This page covers the conversion of fitted Laue strain maps to elastic stress
maps using Hooke's law, the required elastic-constant input, the Voigt
notation conventions used internally, and the available reference-frame
transformations.

> **Prerequisite**: stress calculation requires per-pixel strain data.  Run
> `fit_strain_orientation` and collect results into a `GrainMap` before
> proceeding.  See [Strain Analysis](laue_strain.md) for that workflow.

---

## 1. Hooke's law and the Laue limitation

In the linear-elastic regime the Cauchy stress tensor
$\boldsymbol{\sigma}$ and the strain tensor $\boldsymbol{\varepsilon}$ are
related by the **generalised Hooke's law**:

$$
\sigma_{ij} = C_{ijkl}\,\varepsilon_{kl}
$$

where $C_{ijkl}$ is the **fourth-rank elastic stiffness tensor**.

### What Laue diffraction can and cannot access

White-beam (polychromatic) Laue diffraction measures the **directions** of
diffracted beams, not their wavelengths.  Spot positions are determined by the
crystal orientation and the *shape* of the unit cell — how the lattice vectors
tilt and shear relative to one another.  A uniform scaling of all d-spacings
(i.e. a purely hydrostatic strain) leaves every spot angle unchanged and is
therefore **invisible** to Laue.

The full strain tensor decomposes as:

$$
\boldsymbol{\varepsilon} = \boldsymbol{\varepsilon}_\text{dev}
  + \underbrace{\tfrac{1}{3}\operatorname{tr}(\boldsymbol{\varepsilon})\,\mathbf{I}}_{\text{hydrostatic — unobservable}}
$$

Only the **deviatoric** part $\boldsymbol{\varepsilon}_\text{dev}$ (traceless,
five independent components) is reliably determined.  Consequently, the stress
computed here is the **deviatoric stress**:

$$
\boxed{\boldsymbol{\sigma}_\text{dev} = \mathbf{C} : \boldsymbol{\varepsilon}_\text{dev}}
$$

The code uses `strain_tensor_deviatoric` (whose trace is exactly zero by
construction) as input, so no hydrostatic contribution can enter through
numerical drift in the fitted diagonal strains.

### Reliability by component

| Quantity | Reliable? | Reason |
|---|---|---|
| Shear stresses $\sigma_{xy}$, $\sigma_{xz}$, $\sigma_{yz}$ | **Yes** | Shear has no hydrostatic part |
| Normal-stress *differences* e.g. $\sigma_{xx}-\sigma_{zz}$ | **Yes** | Differences cancel the unknown pressure |
| Von Mises stress $\sigma_\text{VM}$ | **Yes** | Built from deviatoric stress only |
| Individual normal stresses $\sigma_{xx}$, $\sigma_{yy}$, $\sigma_{zz}$ | Relative only | Each carries the same unknown offset $P$ |
| Mean stress $\tfrac{1}{3}\operatorname{tr}(\boldsymbol{\sigma})$ | **No** | This *is* the unknown hydrostatic pressure |

In **Voigt notation** the law becomes a matrix–vector product:

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

## 4. `stress_voigt` — deviatoric stress map

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

The returned tensor is the **deviatoric stress** $\mathbf{C}:\boldsymbol{\varepsilon}_\text{dev}$.
Its trace is zero by construction; individual normal components
($\sigma_{xx}$, $\sigma_{yy}$, $\sigma_{zz}$) are reliable only as
*relative* values — they share an unknown hydrostatic offset equal to the
true (inaccessible) pressure.

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

## 5. Scalar invariants — von Mises stress and equivalent strain

Scalar invariants are the most reliable and interpretable quantities from Laue
data.  They do not depend on the choice of reference frame (they are
second-invariant scalars), and they are completely insensitive to the unknown
hydrostatic component.

### 5.1 Von Mises stress

#### Formulation

The **von Mises stress** (equivalent tensile stress) is built from the second
invariant $J_2$ of the deviatoric stress tensor
$\mathbf{s} = \boldsymbol{\sigma}_\text{dev}$:

$$
J_2 = \tfrac{1}{2}\,s_{ij}\,s_{ij}
\qquad\Longrightarrow\qquad
\sigma_\text{VM} = \sqrt{3\,J_2} = \sqrt{\tfrac{3}{2}\,s_{ij}\,s_{ij}}
$$

Expanded in Cartesian components:

$$
\sigma_\text{VM}
  = \sqrt{\tfrac{3}{2}(s_{xx}^2 + s_{yy}^2 + s_{zz}^2)
          + 3\,(s_{xy}^2 + s_{xz}^2 + s_{yz}^2)}
$$

An equivalent, often more intuitive form expresses $\sigma_\text{VM}$ through
*differences* of principal stresses $\sigma_1,\sigma_2,\sigma_3$:

$$
\sigma_\text{VM}
  = \sqrt{\tfrac{1}{2}\bigl[(\sigma_1-\sigma_2)^2
                           +(\sigma_2-\sigma_3)^2
                           +(\sigma_3-\sigma_1)^2\bigr]}
$$

The hydrostatic pressure $P$ shifts all three principal stresses by the same
amount and therefore cancels in every difference — confirming that
$\sigma_\text{VM}$ is immune to the missing hydrostatic component.

#### Why it is unambiguous for Laue data

`stress_voigt` returns the purely deviatoric stress
($\operatorname{tr}(\boldsymbol{\sigma})=0$ by construction), so:

$$
\sigma_\text{VM}(\boldsymbol{\sigma}_\text{dev} + P\,\mathbf{I})
  = \sigma_\text{VM}(\boldsymbol{\sigma}_\text{dev})
$$

for *any* value of the unknown $P$.

#### Physical meaning

$\sigma_\text{VM}$ governs the onset of plastic yielding in the von Mises (J₂)
criterion: yielding occurs when $\sigma_\text{VM} \geq \sigma_Y$ (uniaxial
yield stress).  It provides a single-number measure of "how stressed" a grain
is, independent of the reference frame.

#### API

```python
# (ny, nx) array, GPa
sigma_vm = gmap.von_mises_stress(crystal, grain=gi_merged, frame="crystal")
print(f"max σ_VM = {np.nanmax(sigma_vm) * 1e3:.1f} MPa")

fig, ax = gmap.plot_von_mises_stress(
    grain=gi_merged,
    crystal=fe,
    frame="crystal",
    scale=1e3,           # GPa → MPa
    cmap="viridis",      # sequential — σ_VM ≥ 0
    vmin=0, vmax=300,
    motor_x="pz", motor_y="py",
    motor_units={"pz": "mm", "py": "mm"},
)
```

---

### 5.2 Von Mises equivalent strain

#### Formulation

The **equivalent strain** $\varepsilon_\text{eq}$ (also called the effective
strain) is the strain-space counterpart of $\sigma_\text{VM}$, built from the
second invariant of the deviatoric strain tensor
$\mathbf{e} = \boldsymbol{\varepsilon}_\text{dev}$:

$$
\varepsilon_\text{eq}
  = \sqrt{\tfrac{2}{3}\,e_{ij}\,e_{ij}}
  = \sqrt{\tfrac{2}{3}(e_{xx}^2 + e_{yy}^2 + e_{zz}^2)
          + \tfrac{4}{3}(e_{xy}^2 + e_{xz}^2 + e_{yz}^2)}
$$

The prefactor $\tfrac{2}{3}$ (versus $\tfrac{3}{2}$ for the stress) is the
conventional normalisation that makes the two quantities **work-conjugate**:

$$
\dot{w}_\text{plastic} = \boldsymbol{\sigma}_\text{dev} : \dot{\mathbf{e}}
  = \sigma_\text{VM}\,\dot{\varepsilon}_\text{eq}
$$

and ensures that in uniaxial tension
$\varepsilon_\text{eq} = \varepsilon_{xx}$ (the applied axial strain).

An alternative form in terms of principal-strain differences
$\varepsilon_1,\varepsilon_2,\varepsilon_3$:

$$
\varepsilon_\text{eq}
  = \sqrt{\tfrac{2}{9}\bigl[(\varepsilon_1-\varepsilon_2)^2
                            +(\varepsilon_2-\varepsilon_3)^2
                            +(\varepsilon_3-\varepsilon_1)^2\bigr]
          + \tfrac{2}{3}(\varepsilon_{12}^2+\varepsilon_{13}^2+\varepsilon_{23}^2)}
$$

Again, the hydrostatic component (mean strain $\varepsilon_m$) cancels in every
principal-strain difference.

#### Why it is unambiguous for Laue data

$e_{ij}\,e_{ij}$ is a scalar double contraction — it is **frame-independent**
and depends only on the deviatoric strain.  Because
`strain_tensor_deviatoric` has exactly zero trace, the hydrostatic contribution
is absent by construction.

#### Comparison with σ_VM

| | $\sigma_\text{VM}$ | $\varepsilon_\text{eq}$ |
|---|---|---|
| Input | `strain_tensor_deviatoric` + stiffness matrix | `strain_tensor_deviatoric` only |
| Units | GPa (or MPa) | dimensionless (or millistrain) |
| Requires elastic constants | Yes | No |
| Frame-independent | Yes | Yes |
| Hydrostatic-immune | Yes | Yes |
| Yields criterion | $\sigma_\text{VM} \geq \sigma_Y$ | Accumulated plastic strain |

$\varepsilon_\text{eq}$ is particularly useful when no stiffness matrix is
available, or as a geometry-only map to identify highly distorted grains before
computing stress.

#### API

```python
# (ny, nx) array, dimensionless
eps_eq = gmap.equivalent_strain(grain=gi_merged)
print(f"max ε_eq = {np.nanmax(eps_eq) * 1e3:.2f} ×10⁻³")

fig, ax = gmap.plot_equivalent_strain(
    grain=gi_merged,
    scale=1e3,           # dimensionless → millistrain
    cmap="viridis",
    vmin=0, vmax=2,      # millistrain
    motor_x="pz", motor_y="py",
    motor_units={"pz": "mm", "py": "mm"},
)
```

Note that `plot_equivalent_strain` takes no `frame` argument — the scalar
invariant is the same in every frame.

---

## 6. `plot_stress_component` — single-component map

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

| String | Symbol | Reliable? |
|---|---|---|
| `"s_xx"` | $\sigma_{xx}$ | Relative only |
| `"s_yy"` | $\sigma_{yy}$ | Relative only |
| `"s_zz"` | $\sigma_{zz}$ | Relative only |
| `"s_xy"` | $\sigma_{xy}$ | **Yes** |
| `"s_xz"` | $\sigma_{xz}$ | **Yes** |
| `"s_yz"` | $\sigma_{yz}$ | **Yes** |

The default colour scale is `"RdBu_r"` with units of MPa (`scale=1e3`).
Use `scale=1.0` for GPa.

---

## 7. Extracting components manually

```python
sig = gmap.stress_voigt(fe, grain=0, frame="crystal")  # (ny, nx, 6)

# Code ordering: [s_xx=0, s_yy=1, s_zz=2, s_xy=3, s_xz=4, s_yz=5]
s_xx = sig[..., 0] * 1e3   # MPa  (deviatoric normal stress — relative)
s_zz = sig[..., 2] * 1e3
s_xy = sig[..., 3] * 1e3   # MPa  (shear — absolute)

# Reliable normal-stress *difference*
biaxial = (sig[..., 0] - sig[..., 2]) * 1e3   # (σ_xx − σ_zz) in MPa
```

---

## 8. Important caveats

### Hydrostatic stress is unconstrained

White-beam Laue diffraction cannot measure the **hydrostatic** part of the
strain tensor (see [Strain Analysis — Hydrostatic blind spot](laue_strain.md#limitation-of-polychromatic-laue-the-hydrostatic-blind-spot)).
`stress_voigt` explicitly uses `strain_tensor_deviatoric` (trace exactly zero)
as input, so the returned stress tensor is purely deviatoric: its trace is zero
by construction and carries no information about the true hydrostatic pressure.

Concretely:

* **Shear stresses** $\sigma_{xy}$, $\sigma_{xz}$, $\sigma_{yz}$ are the most
  reliable outputs — they are directly proportional to the corresponding shear
  strain components, which *are* measurable.
* **Von Mises stress** is the recommended scalar summary — it is a tensor
  invariant, physically meaningful, and immune to the missing pressure.
* **Differences** of normal stresses (e.g. $\sigma_{xx} - \sigma_{zz}$) are
  reliable — they probe the deviatoric part.
* **Individual** normal stresses should be read as values relative to an
  unknown offset $P$ (the true hydrostatic pressure), not as absolute
  normal stresses.  Do not interpret them as the full Cauchy normal stress
  unless the hydrostatic component is constrained by an independent measurement
  (e.g. monochromatic XRD with a calibrated reference).

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

## 9. Complete example

```python
import numpy as np
import xrayutilities as xu
from nrxrdct.laue import GrainMap

# Load a pre-fitted map
gmap = GrainMap.load("grainmap.h5")

# Crystal object with elastic constants
fe = xu.materials.Fe

# Merge grains
best_grain, metrics = gmap.merge(metric="match_rate", min_match_rate=0.3)
gi = gmap.apply_merge(best_grain, metrics)

# ── Equivalent strain (no elastic constants needed) ────────────────────
fig, ax = gmap.plot_equivalent_strain(
    grain=gi,
    scale=1e3,              # dimensionless → millistrain
    cmap="viridis", vmin=0, vmax=2,
    motor_x="pz", motor_y="py",
    motor_units={"pz": "mm", "py": "mm"},
)

# ── Von Mises stress ────────────────────────────────────────────────────
fig, ax = gmap.plot_von_mises_stress(
    grain=gi, crystal=fe,
    frame="crystal", scale=1e3,
    cmap="viridis", vmin=0, vmax=300,   # MPa
    motor_x="pz", motor_y="py",
    motor_units={"pz": "mm", "py": "mm"},
)

# ── Individual stress components ───────────────────────────────────────
# Shear components are the most reliable
for comp in ("s_xy", "s_xz", "s_yz"):
    gmap.plot_stress_component(
        comp, grain=gi, crystal=fe,
        frame="crystal", scale=1e3,
        vmin=-200, vmax=200,
    )

# Normal components — interpret as relative (deviatoric) values
for comp in ("s_xx", "s_yy", "s_zz"):
    gmap.plot_stress_component(
        comp, grain=gi, crystal=fe,
        frame="crystal", scale=1e3,
        vmin=-500, vmax=500,
    )

# ── Manual access for custom analysis ─────────────────────────────────
sig = gmap.stress_voigt(fe, grain=gi, frame="crystal")  # (ny, nx, 6), GPa

# Reliable: normal-stress difference
biaxial_MPa = (sig[..., 0] - sig[..., 2]) * 1e3   # σ_xx − σ_zz

# Reliable: von Mises from the array directly
s = sig  # already deviatoric (trace = 0)
vm_MPa = np.sqrt(
    1.5 * (s[...,0]**2 + s[...,1]**2 + s[...,2]**2)
    + 3.0 * (s[...,3]**2 + s[...,4]**2 + s[...,5]**2)
) * 1e3
```
