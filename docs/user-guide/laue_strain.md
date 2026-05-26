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
