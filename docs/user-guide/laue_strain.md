# Strain analysis in white-beam Laue diffraction

This page covers the theoretical background of the strain calculations in
`nrxrdct`, the fitting model, the available output quantities, the reference-frame
transformations, and the inherent limitations of polychromatic Laue for strain
measurement.

---

## The strain tensor

A small, homogeneous deformation of a crystal is described by the **infinitesimal
strain tensor** $\boldsymbol{\varepsilon}$, a symmetric $3 \times 3$ second-rank
tensor:

$$
\boldsymbol{\varepsilon} =
\begin{pmatrix}
\varepsilon_{xx} & \varepsilon_{xy} & \varepsilon_{xz} \\
\varepsilon_{xy} & \varepsilon_{yy} & \varepsilon_{yz} \\
\varepsilon_{xz} & \varepsilon_{yz} & \varepsilon_{zz}
\end{pmatrix}
$$

The diagonal components $\varepsilon_{xx}$, $\varepsilon_{yy}$, $\varepsilon_{zz}$
are **normal strains** — relative changes in length along each axis.  The
off-diagonal components $\varepsilon_{xy}$, $\varepsilon_{xz}$, $\varepsilon_{yz}$
are **shear strains**.

Because $\boldsymbol{\varepsilon}$ is symmetric, only six independent components
exist.  In **Voigt notation** these are written as a vector:

$$
\boldsymbol{\varepsilon}_{\text{Voigt}}
= [\,\varepsilon_{xx},\; \varepsilon_{yy},\; \varepsilon_{zz},\;
         \varepsilon_{xy},\; \varepsilon_{xz},\; \varepsilon_{yz}\,]
$$

### Volumetric and deviatoric decomposition

Any strain tensor can be split into a **hydrostatic** (isotropic) part and a
**deviatoric** part:

$$
\boldsymbol{\varepsilon}
= \underbrace{\frac{1}{3}\operatorname{tr}(\boldsymbol{\varepsilon})\,\mathbf{I}}_{\text{hydrostatic}}
+ \underbrace{\boldsymbol{\varepsilon}
  - \frac{1}{3}\operatorname{tr}(\boldsymbol{\varepsilon})\,\mathbf{I}}_{\text{deviatoric}}
$$

The trace $\operatorname{tr}(\boldsymbol{\varepsilon}) = \varepsilon_{xx} +
\varepsilon_{yy} + \varepsilon_{zz}$ is the **volumetric strain** — the relative
change in unit-cell volume $\Delta V / V$.

---

## Limitation of polychromatic Laue: the hydrostatic blind spot

In a **monochromatic** experiment the diffraction angle $2\theta$ of a reflection
$(hkl)$ depends directly on the $d$-spacing via Bragg's law, so absolute lattice
parameters — and therefore the full strain tensor including its hydrostatic part
— are measurable.

In a **polychromatic (white-beam)** Laue experiment, the detector only records
spot *positions*, not the wavelength used for each reflection.  A pure hydrostatic
strain $\boldsymbol{\varepsilon} = \alpha\,\mathbf{I}$ uniformly scales every
$d$-spacing by the same factor, which is exactly equivalent to illuminating the
unstrained crystal at a slightly different wavelength.  The spot positions on the
detector are therefore **unchanged**, making the hydrostatic component
**inaccessible**.

!!! warning "Hydrostatic strain is not measurable by white-beam Laue alone"
    Polychromatic Laue is sensitive only to the **deviatoric** part of the strain
    tensor (shape changes of the unit cell).  The absolute lattice parameters, and
    therefore the volumetric strain, cannot be determined without additional
    information such as a monochromatic measurement or knowledge of one lattice
    parameter from an independent source (e.g. a reference layer with known strain
    state).

In practice this means:

* The **five** independent deviatoric components are well-determined.
* The **sixth** component (the hydrostatic contribution to any one normal strain) has
  a large, correlated uncertainty that is entirely absorbed by the fitted rotation
  $\delta\boldsymbol{\omega}$.
* Fitting all six components (`fit_strain` default) is valid but the user should
  interpret the *individual* normal strains $\varepsilon_{xx}$, $\varepsilon_{yy}$,
  $\varepsilon_{zz}$ with this in mind — only their **differences** and the shear
  components are truly independent.

---

## The fitting model

`fit_strain_orientation` fits a rotation increment **and** a strain tensor
simultaneously by minimising the pixel-distance residuals between observed and
simulated Laue spots.

### Effective orientation matrix

The strained crystal is modelled through an **effective orientation matrix**:

$$
\mathbf{U}_\text{eff}
= \mathbf{R}(\delta\boldsymbol{\omega})\;\mathbf{U}_0\;(\mathbf{I} + \boldsymbol{\varepsilon})
$$

where:

| Symbol | Meaning |
|---|---|
| $\mathbf{U}_0$ | Starting orientation matrix (from `fit_orientation` or supplied by the user) |
| $\mathbf{R}(\delta\boldsymbol{\omega})$ | Small rotation increment parametrised as a rotation vector $\delta\boldsymbol{\omega} \in \mathbb{R}^3$ |
| $\boldsymbol{\varepsilon}$ | Symmetric strain tensor in the **crystal** frame |
| $\mathbf{U}_\text{eff}$ | Effective orientation passed to the Laue simulator |

The strain modifies the reciprocal-lattice vectors, shifting each spot to its
strained position on the detector.  The rotation increment $\delta\boldsymbol{\omega}$
absorbs any residual misalignment from the previous orientation fit.

### Free parameters

The optimisation vector is

$$
\mathbf{x} = [\,\delta\omega_x,\;\delta\omega_y,\;\delta\omega_z,\;
               s\,\varepsilon_{xx},\; s\,\varepsilon_{yy},\; s\,\varepsilon_{zz},\;
               s\,\varepsilon_{xy},\; s\,\varepsilon_{xz},\; s\,\varepsilon_{yz}\,]
$$

where $s$ is `strain_scale` ($10^{-4}$ by default), used to bring the strain
parameters to the same order of magnitude as the rotation angles (radians) so that
the Levenberg–Marquardt optimiser performs well.

The `fit_strain` parameter selects a **subset** of the six strain components to
refine; the remainder are held at zero.  For example:

```python
# Fit only the two in-plane normal strains and all shear components
fit_strain_orientation(crystal, camera, obs_xy, U0,
                       fit_strain=("e_xx", "e_yy", "e_xy", "e_xz", "e_yz"))
```

### Residuals and matching

At each function evaluation the simulator generates the predicted spot positions
with the current $\mathbf{U}_\text{eff}$.  Observed and simulated spots are
matched one-to-one using the Hungarian algorithm within a matching radius
`max_match_px`, and the residual vector is the stack of pixel-distance components
of all matched pairs.  `scipy.optimize.least_squares` (Levenberg–Marquardt by
default) minimises the sum of squared residuals.

Staged refinement is supported via a list of decreasing `max_match_px` values; each
stage warm-starts from the solution of the previous one.

---

## Reference frames

### Crystal frame (storage frame)

The strain tensor as stored in `StrainFitResult.strain_tensor` (and in
`GrainMap.strain_tensor`) is expressed in the **crystal coordinate system** — the
axes are those of the unit cell as defined by the xrayutilities `Crystal` object.
This is the frame in which the fitting is performed.

### Lab frame

The **lab frame** is the instrument coordinate system.  The orientation matrix
$\mathbf{U}$ maps crystal vectors to lab vectors:

$$
\mathbf{r}_\text{lab} = \mathbf{U}\,\mathbf{r}_\text{crystal}
$$

The strain tensor in the lab frame is obtained by the standard second-rank tensor
rotation:

$$
\boldsymbol{\varepsilon}_\text{lab}
= \mathbf{U}\;\boldsymbol{\varepsilon}_\text{crystal}\;\mathbf{U}^\top
$$

### Sample frame

The **sample frame** accounts for the physical orientation of the sample on the
diffractometer.  In the default configuration it is related to the lab frame by a
rotation $\mathbf{R}_s$ of $-40°$ about the lab $Y$ axis (the sample tilt):

$$
\mathbf{r}_\text{sample} = \mathbf{R}_s\,\mathbf{r}_\text{lab}
$$

$$
\boldsymbol{\varepsilon}_\text{sample}
= \mathbf{R}_s\;\boldsymbol{\varepsilon}_\text{lab}\;\mathbf{R}_s^\top
= \mathbf{R}_s\,\mathbf{U}\;\boldsymbol{\varepsilon}_\text{crystal}\;
  \mathbf{U}^\top\mathbf{R}_s^\top
$$

The tilt angle and axis are configurable in all analysis functions via
`sample_tilt_deg` and `sample_tilt_axis`.

---

## Available output quantities

### From `fit_strain_orientation`

| Attribute | Shape | Description |
|---|---|---|
| `result.strain_tensor` | `(3, 3)` | Symmetric strain tensor in the crystal frame |
| `result.strain_voigt` | `(6,)` | Voigt vector $[\varepsilon_{xx}, \varepsilon_{yy}, \varepsilon_{zz}, \varepsilon_{xy}, \varepsilon_{xz}, \varepsilon_{yz}]$ in the crystal frame |
| `result.U` | `(3, 3)` | Refined orientation matrix (rotation only; no strain) |
| `result.U_eff` | `(3, 3)` | Effective orientation $= \mathbf{R}\,\mathbf{U}_0\,(\mathbf{I}+\boldsymbol{\varepsilon})$ used by the simulator |
| `result.rms_px` | scalar | RMS pixel residual of matched spots |
| `result.mean_px` | scalar | Mean pixel residual of matched spots |
| `result.n_matched` | int | Number of matched spots |
| `result.match_rate` | scalar | Fraction of observed spots matched |

### From `GrainMap` (after `collect_strain`)

| Array | Shape | Description |
|---|---|---|
| `gmap.strain_tensor` | `(n_grains, ny, nx, 3, 3)` | Full strain tensor — crystal frame |
| `gmap.strain_voigt` | `(n_grains, ny, nx, 6)` | Voigt vector — crystal frame |
| `gmap.U` | `(n_grains, ny, nx, 3, 3)` | Orientation matrices |
| `gmap.rms_px` | `(n_grains, ny, nx)` | RMS pixel residual |
| `gmap.mean_px` | `(n_grains, ny, nx)` | Mean pixel residual |
| `gmap.n_matched` | `(n_grains, ny, nx)` | Matched spot count |
| `gmap.match_rate` | `(n_grains, ny, nx)` | Match rate |

Unfitted pixels have `NaN` (float arrays) or `-1` (`n_matched`).

---

## Plotting and frame selection

### Single-component map

```python
# Crystal frame (default)
gmap.plot_strain_component("e_zz", grain=0, frame="crystal")

# Lab frame
gmap.plot_strain_component("e_zz", grain=0, frame="lab")

# Sample frame (−40° about Y by default)
gmap.plot_strain_component("e_zz", grain=0, frame="sample")

# Custom tilt
gmap.plot_strain_component("e_zz", grain=0, frame="sample",
                           sample_tilt_deg=-35.0, sample_tilt_axis="y",
                           motor_x="pz", motor_y="py",
                           motor_units={"pz": "mm", "py": "mm"})
```

Component names: `"e_xx"`, `"e_yy"`, `"e_zz"`, `"e_xy"`, `"e_xz"`, `"e_yz"`.

### Manual frame rotation

To rotate to an arbitrary frame, use `GrainMap._strain_component_map` directly or
apply the rotation manually:

```python
eps = gmap.strain_tensor[0]           # (ny, nx, 3, 3) — crystal frame
U   = gmap.U[0]                       # (ny, nx, 3, 3)

# Lab frame (vectorised)
eps_lab = np.einsum("...ik,...kl,...jl->...ij", U, eps, U)

# Custom rotation R applied on top of the lab frame
R = scipy.spatial.transform.Rotation.from_euler("y", -40, degrees=True).as_matrix()
eps_custom = np.einsum("ik,...kl,jl->...ij", R, eps_lab, R)

# Extract a component
e_zz_custom = eps_custom[..., 2, 2]   # (ny, nx)
```

---

## Practical notes

* **Units**: strain is dimensionless (relative deformation).  Typical elastic
  strains in thin films and strained crystals are in the range $10^{-4}$ to
  $10^{-2}$.

* **Sign convention**: tensile strain along an axis is **positive**; compressive
  strain is **negative**.

* **Shear components**: $\varepsilon_{xy}$, $\varepsilon_{xz}$, $\varepsilon_{yz}$
  are the *engineering* shear strains divided by two (i.e. the tensor components,
  not the full engineering shear $\gamma = 2\varepsilon$).

* **Fitting stability**: because the hydrostatic component is poorly constrained,
  the individual normal strains may have large absolute uncertainties even when the
  deviatoric components are well-determined.  Comparing `e_zz` maps across grains
  is most meaningful when the grains share the same reference state.

* **Staged refinement**: always run orientation fitting first (`fit_orientation` or
  `submit_orientation`), then strain fitting starting from the converged orientation.
  A typical `max_match_px` sequence for strain is `[10, 3]` — looser initial
  matching to survive small orientation errors, tightening to sharpen the strain
  estimate.

---

## Layered crystal (stack) — `fit_strain_orientation_stack`

For epitaxial stacks modelled as a
[`LayeredCrystal`][nrxrdct.laue.layers.LayeredCrystal], a single global
rotation is shared by all layers (preserving inter-layer relationships) while
each layer receives its own independent strain tensor:

$$
\mathbf{U}_{\text{eff},i}
= \mathbf{R}(\delta\boldsymbol{\omega})\;\mathbf{U}_{0,i}\;(\mathbf{I} + \boldsymbol{\varepsilon}_i)
$$

The parameter vector is

$$
\mathbf{x} = [\,\delta\omega_x,\;\delta\omega_y,\;\delta\omega_z,\;
               s\,\varepsilon_{i=0,xx},\;\ldots,\;s\,\varepsilon_{i=0,yz},\;
               s\,\varepsilon_{i=1,xx},\;\ldots\,]
$$

with total length $3 + N_\text{layers} \times n_\text{strain}$.

### Usage

```python
import nrxrdct.laue as laue

# stack already built and orientation fitted via fit_orientation_stack
result = laue.fit_strain_orientation_stack(
    stack, camera, peaks[:, :2],
    fit_strain    = ("e_xx", "e_yy", "e_zz"),  # diagonal / biaxial only
    max_match_px  = [5, 2, 0.5],
    verbose       = True,
)
# StackStrainFitResult [OK]  rms=0.41 px  mean=0.38 px  matched=47/52 (90%)  |δω|=0.003°

# per-layer strain
for i, (eps, voigt) in enumerate(zip(result.strain_tensors, result.strain_voigts)):
    print(f"Layer {i}: ε_zz = {eps[2,2]:.2e}")

# effective matrices for simulation
for layer, U_eff in zip(stack.all_layers, result.U_eff_layers):
    layer.U = U_eff
spots = laue.simulate_laue_stack(stack, camera)
```

### Result fields

| Attribute | Shape | Description |
|---|---|---|
| `result.R_global` | `(3, 3)` | Global rotation applied to all layers |
| `result.rotvec` | `(3,)` | Rotation vector (radians) |
| `result.U_layers` | `list of (3, 3)` | Pure rotation part per layer |
| `result.U_eff_layers` | `list of (3, 3)` | Effective matrix `R @ U0_i @ (I + ε_i)` — pass as `U` to `simulate_laue` |
| `result.strain_tensors` | `list of (3, 3)` | Symmetric strain tensor per layer, crystal frame |
| `result.strain_voigts` | `list of (6,)` | Voigt vector `[ε_xx, ε_yy, ε_zz, ε_xy, ε_xz, ε_yz]` per layer |
| `result.rms_px` | scalar | RMS pixel residual |
| `result.n_matched` | int | Matched spots |

!!! tip "Recommended workflow for stacks"
    1. `fit_orientation_stack` — converge the global rotation with a loose match window.
    2. `fit_strain_orientation_stack` — add per-layer strain starting from the refined orientations.
    3. `refine_strain_image_stack` — image-space polishing pass (see [Image refinement](laue_image_refinement.md#layered-crystal-stack-variants)).

---

## Comparison with LaueTools strain refinement

`nrxrdct` adopts the LaueTools lab-frame convention and can read LaueTools
calibration files and `matstarlab` orientation matrices (see
[`decompose_matstarlab`][nrxrdct.laue.simulation.decompose_matstarlab]).
However, the two implementations differ in parameterisation, decomposition
strategy, and residual construction.

### Parameterisation

**LaueTools** refines the full $3 \times 3$ UB matrix (nine elements) in a
single nonlinear optimisation.  After convergence the matrix is decomposed into
a pure rotation and a stretch via the **right polar decomposition**:

$$
\mathbf{F} = \mathbf{U}\,\mathbf{P}
$$

where $\mathbf{U}$ is orthogonal ($\mathbf{U}^\top\mathbf{U} = \mathbf{I}$,
$\det\mathbf{U} = +1$) and $\mathbf{P}$ is symmetric positive-definite (the
right-stretch tensor).  The strain tensor is then extracted as:

$$
\boldsymbol{\varepsilon}_\text{LT} = \mathbf{P} - \mathbf{I}
$$

This decomposition is **exact** — no small-strain approximation is made.

**`nrxrdct`** parameterises the problem explicitly from the outset.  The
optimisation vector is

$$
\mathbf{x} = [\,\delta\omega_x,\;\delta\omega_y,\;\delta\omega_z,\;
               s\,\varepsilon_{xx},\;\ldots,\;s\,\varepsilon_{yz}\,]
$$

and the effective orientation is assembled as:

$$
\mathbf{U}_\text{eff}
= \mathbf{R}(\delta\boldsymbol{\omega})\;\mathbf{U}_0\;(\mathbf{I} + \boldsymbol{\varepsilon})
$$

This uses the **small-strain approximation** — the deformation gradient is
linearised as $\mathbf{F} \approx \mathbf{I} + \boldsymbol{\varepsilon}$ and
the full effective matrix is $\mathbf{U}_0(\mathbf{I}+\boldsymbol{\varepsilon})$.
For elastic strains typical of metals ($|\boldsymbol{\varepsilon}| \lesssim 10^{-3}$)
the error relative to the exact polar-decomposition approach is of order
$|\boldsymbol{\varepsilon}|^2 \sim 10^{-6}$, which is negligible.

The key practical difference is that `nrxrdct` exposes individual strain
components directly as optimisation parameters, making it straightforward to
**fix symmetry constraints** (e.g. biaxial strain: set
`fit_strain=("e_xx","e_yy","e_xy","e_xz","e_yz")`) without post-processing.

### Rotation–strain coupling and the bake-in step

LaueTools refines a single UB matrix and separates rotation from strain only
after convergence, so the two are decoupled implicitly through the polar
decomposition.

`nrxrdct` uses **staged refinement** with an explicit rotation bake-in between
stages.  After stage $i$ (pixel tolerance $r_i$), the accumulated rotation is
absorbed into the reference orientation:

$$
\mathbf{U}_0^{(i+1)}
= \mathbf{R}\!\left(\delta\boldsymbol{\omega}^{(i)}\right)\,\mathbf{U}_0^{(i)}
$$

and the strain parameters are **reset to zero** for stage $i+1$ (tighter
tolerance $r_{i+1} < r_i$).  This means that at the final stage the strain
$\boldsymbol{\varepsilon}$ is always measured relative to the cumulatively
rotated but **unstrained** reference structure, and cannot accumulate
compounding linearisation errors across stages.

### Residual construction

Both methods minimise **pixel-space** residuals between simulated and observed
spot positions.  The differences are in how unmatched spots are handled:

| | LaueTools | `nrxrdct` |
|---|---|---|
| Matching criterion | Angular tolerance (degrees) | Pixel-radius threshold `max_match_px` |
| Unmatched spot penalty | Spot excluded from residuals | Residual set to exactly $\pm r_\text{max}$ (Hungarian penalty) |
| Hydrostatic sensitivity | Not enforced | Not enforced |

Both approaches are **insensitive to hydrostatic strain** for the physical
reason described above: a uniform dilation leaves all spot positions invariant.
Neither software enforces the deviatoric constraint $\operatorname{tr}(\boldsymbol{\varepsilon}) = 0$;
instead, the hydrostatic component floats freely and is absorbed partly into
the fitted rotation increment and partly into the unconstrained trace of the
strain tensor.

### Reading LaueTools results into `nrxrdct`

Because `nrxrdct` uses the LaueTools lab frame and $2\pi$-scaled reciprocal
vectors internally, the interoperability functions handle the two frame and
scaling differences automatically:

```python
from nrxrdct.laue.simulation import decompose_matstarlab

# matstarlab: LaueTools grain matrix (LT2/OR frame, no 2π)
U, F, eps, eps_voigt = decompose_matstarlab(matstarlab, crystal)

# U  — pure rotation (use in simulate_laue for rotation-only simulation)
# F  — full deformation gradient (use in simulate_laue to include strain)
# eps — small-strain tensor in the crystal frame,  ε = P − I  (exact)
# eps_voigt — [ε_xx, ε_yy, ε_zz, ε_yz, ε_xz, ε_xy]
```

`decompose_matstarlab` uses the exact polar decomposition (same as LaueTools),
so `eps` here is directly comparable to the strain output of LaueTools'
`fitorient` module.

### Comparing LaueTools and `nrxrdct` strain maps on a `GrainMap`

#### Voigt ordering mismatch — always compare via the 3×3 tensor

The Voigt component order differs between the two outputs:

| Source | Voigt order |
|---|---|
| `StrainFitResult.strain_voigt` / `gmap.strain_voigt` | $[\varepsilon_{xx}, \varepsilon_{yy}, \varepsilon_{zz}, \varepsilon_{xy}, \varepsilon_{xz}, \varepsilon_{yz}]$ |
| `decompose_matstarlab` `eps_voigt` | $[\varepsilon_{xx}, \varepsilon_{yy}, \varepsilon_{zz}, \varepsilon_{yz}, \varepsilon_{xz}, \varepsilon_{xy}]$ |

Never compare the Voigt vectors element-by-element. Always work through the 3×3
`strain_tensor`.

#### The deviatoric field

Because white-beam Laue is insensitive to hydrostatic strain, the only quantity
that is robustly comparable between the two methods is the **deviatoric strain**:

$$
\boldsymbol{\varepsilon}_\text{dev}
= \boldsymbol{\varepsilon} - \frac{\operatorname{tr}(\boldsymbol{\varepsilon})}{3}\,\mathbf{I}
$$

`GrainMap` stores this automatically in `strain_tensor_deviatoric`
(shape `(n_grains, ny, nx, 3, 3)`, crystal frame), derived from `strain_tensor`
whenever strain results are stored or loaded.

#### Practical comparison workflow

```python
import numpy as np
from nrxrdct.laue.simulation import decompose_matstarlab
from nrxrdct.laue import fit_strain_orientation

# 1. Convert a LaueTools matstarlab to strain
U_lt, F_lt, eps_lt, _ = decompose_matstarlab(matstarlab, crystal)
eps_dev_lt = eps_lt - np.trace(eps_lt) / 3.0 * np.eye(3)

# 2. Run nrxrdct fitter from the same starting orientation
result = fit_strain_orientation(crystal, camera, obs_xy, U0=U_lt)

# 3. Compare deviatoric tensors (should agree within fit uncertainty)
diff = result.strain_tensor - np.trace(result.strain_tensor) / 3.0 * np.eye(3) - eps_dev_lt
print("Max |Δε_dev|:", np.abs(diff).max())

# 4. On a full GrainMap, access the stored deviatoric field
#    gmap.strain_tensor_deviatoric has shape (n_grains, ny, nx, 3, 3)
eps_dev_map = gmap.strain_tensor_deviatoric[grain_idx]   # (ny, nx, 3, 3)
e_dev_zz    = eps_dev_map[..., 2, 2]                     # (ny, nx)  zz component
```

#### What to expect

| Quantity | Should agree? | Reason |
|---|---|---|
| `strain_tensor_deviatoric` | Yes, within fit uncertainty | Both methods are sensitive to the same 5 deviatoric DOFs |
| Shear components $\varepsilon_{xy}$, $\varepsilon_{xz}$, $\varepsilon_{yz}$ | Yes | Purely deviatoric |
| Individual normal strains $\varepsilon_{xx}$, $\varepsilon_{yy}$, $\varepsilon_{zz}$ | Only their *differences* | Each mixes deviatoric + hydrostatic |
| $\operatorname{tr}(\boldsymbol{\varepsilon})$ | No | Unobservable from Laue; absorbs noise differently in each method |

Residual disagreement in the deviatoric part after accounting for the above
typically points to one of: different matched spot sets, different convergence
basins due to differing tolerance strategies, or a misaligned starting
orientation.

### Summary

| Aspect | LaueTools | `nrxrdct` |
|---|---|---|
| Optimisation variables | Full $3\times3$ UB (9 DOF) | $\delta\boldsymbol{\omega}$ (3) + selected $\varepsilon$ components (up to 6) |
| Strain extraction | Post-hoc polar decomposition — exact | Small-strain model $(\mathbf{I}+\boldsymbol{\varepsilon})$ — valid for $\|\boldsymbol{\varepsilon}\| \lesssim 10^{-3}$ |
| Symmetry constraints | Via lattice-parameter constraints | Via `fit_strain` subset selection |
| Matching tolerance | Angular (degrees) | Pixel radius (`max_match_px`) |
| Unmatched spots | Excluded | Penalised at $r_\text{max}$ |
| Staged refinement | Angular threshold coarsening | Pixel threshold coarsening + rotation bake-in |
| Hydrostatic constraint | Not enforced | Not enforced |
