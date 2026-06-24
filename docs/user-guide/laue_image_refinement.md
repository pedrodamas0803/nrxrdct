# Image-based orientation and strain refinement

This page describes the image-based post-refinement functions
[`refine_orientation_image`][nrxrdct.laue.fitting.refine_orientation_image] and
[`refine_strain_image`][nrxrdct.laue.fitting.refine_strain_image], which refine
orientation and strain by maximising the agreement between simulated Laue spot
positions and the raw detector image — without requiring a segmented peak list.

---

## Motivation

The standard refinement pipeline (`fit_orientation` → `fit_strain_orientation`)
relies on a list of segmented peak positions.  Segmentation can miss spots that
are:

- too **dim** relative to the background or to nearby stronger spots,
- too **close** to a dominant reflection to be resolved by the peak finder,
- from a **secondary grain** with a low match rate that falls below the
  acceptance threshold.

In all these cases the raw detector image still contains the information needed
to refine the orientation.  The image-based functions bypass segmentation
entirely and work directly on the pixel data.

---

## Objective function

For a candidate rotation vector $\delta\boldsymbol{\omega}$ the orientation is

$$
\mathbf{U} = R(\delta\boldsymbol{\omega})\,\mathbf{U}_0
$$

The **score** is computed with a single FFT-based convolution per optimizer
iteration:

1. **Simulate** Laue spots for the candidate $\mathbf{U}$ using
   `geometry_only=True` (structure factors are skipped for speed; pass a
   pre-computed `allowed_hkl` to retain systematic-absence filtering at no
   per-call cost).
2. **Build a delta map** — an image of zeros with each simulated pixel position
   incremented by the predicted spot intensity $I_\text{sim}(hkl)$.  Intensity
   weighting means that strong reflections dominate the objective, reducing
   sensitivity to weak background features.
3. **Convolve** the delta map with a Gaussian kernel of width $\sigma$ (the
   `kernel_sigma` parameter) using an FFT.  Each point becomes a smooth blob
   of radius $\approx 3\sigma$.
4. **Score** — element-wise product of the convolved map with the
   background-subtracted detector image, summed over all pixels:

$$
S(\delta\boldsymbol{\omega})
= \sum_{x,y} \bigl[D(x,y) - B(x,y)\bigr]
  \cdot \bigl[K_\sigma * \Delta(x,y;\,\delta\boldsymbol{\omega})\bigr]
$$

where $D$ is the raw image, $B$ is the slowly-varying background estimated by a
large-$\sigma$ Gaussian (see below), $\Delta$ is the intensity-weighted delta
map, and $K_\sigma$ is the Gaussian kernel.

Maximising $S$ pulls each simulated spot position toward the nearest bright pixel
region, refining the orientation.

### Background subtraction

Before the optimizer runs, a large-$\sigma$ FFT Gaussian (default
$\sigma_\text{bg} = 251$ px, same routine as
[`gaussian_background`][nrxrdct.laue.segmentation.gaussian_background]) is
subtracted from the image.  This removes the slowly-varying detector pedestal
(beam-centre falloff, inter-module offsets, diffuse scattering) while leaving
Bragg peaks intact.  The subtraction is computed **once**, outside the optimizer
loop, so it adds no per-call overhead.  Set `bg_sigma=0` to skip it.

---

## Choosing `kernel_sigma`

`kernel_sigma` ($\sigma$ in pixels) controls the spatial selectivity of the
objective.  The Gaussian weight at distance $d$ from a predicted spot is
$\exp(-d^2 / 2\sigma^2)$:

| $\sigma$ (px) | Weight at $d = 2$ px | Use case |
|---|---|---|
| 5.0 | 0.92 | Isolated spots, smooth landscape, easiest convergence |
| 1.0 | 0.14 | Spots a few pixels apart |
| 0.3 (default) | < 0.001 | Spots 1–2 px from a dominant neighbour |

!!! warning "Spot confusion with large kernels"
    When a secondary grain's spots are dim and within a few pixels of much
    brighter primary-grain spots, a large $\sigma$ causes the optimizer to
    climb the gradient of the primary spot rather than the secondary one.
    Reduce `kernel_sigma` to $\lesssim 1$ px (or the default 0.3 px) to
    decouple them spatially.

---

## The two functions

### `refine_orientation_image` — orientation only (3 parameters)

Refines only the rotation correction $\delta\boldsymbol{\omega}$:

$$
\mathbf{U} = R(\delta\boldsymbol{\omega})\,\mathbf{U}_0
$$

**When to use:** as a first pass when segmentation-based orientation is
unavailable or unreliable, or when you want a fast, robust refinement before
attempting strain.

```python
import nrxrdct.laue as laue

hkl = laue.precompute_allowed_hkl(crystal, E_max_eV=27000)

result = laue.refine_orientation_image(
    crystal, U0, camera, raw_frame,
    allowed_hkl   = hkl,
    kernel_sigma  = 0.3,     # tight — decouples spots 2 px apart
    max_angle_deg = 0.2,     # keep small when starting from a good U
    verbose       = True,
)

print(result)
# ImageRefinementResult [OK]  |δω|=0.021°  score=8741.2  Δscore=+612.4  n_sim=138
U_refined = result.U
```

### `refine_strain_image` — orientation + strain (9 parameters)

Extends the parameter space to include the six independent strain components,
using the same deformation model as [`fit_strain_orientation`][nrxrdct.laue.fitting.fit_strain_orientation]:

$$
\mathbf{U}_\text{eff} = R(\delta\boldsymbol{\omega})\,\mathbf{U}_0\,({\bf I} + \boldsymbol{\varepsilon})
$$

The parameter vector is

$$
\mathbf{p} = [\omega_x,\; \omega_y,\; \omega_z,\; \varepsilon_{xx},\; \varepsilon_{yy},\; \varepsilon_{zz},\; \varepsilon_{xy},\; \varepsilon_{xz},\; \varepsilon_{yz}]
$$

with strain components internally divided by `strain_scale` (default $10^{-4}$)
so that all parameters have comparable magnitudes inside the optimizer.

**When to use:** as a polishing step after conventional strain fitting, passing
the prior strain tensor as `strain0` and the image-refined U as `U0`.

```python
r_str = laue.refine_strain_image(
    crystal, r_ori.U, camera, raw_frame,
    strain0       = prior_result.strain_tensor,
    allowed_hkl   = hkl,
    kernel_sigma  = 0.3,
    max_angle_deg = 0.2,
    verbose       = True,
)

print(r_str)
# StrainImageRefinementResult [OK]  |δω|=0.004°  score=9102.1  Δscore=+361.9
#   n_sim=138  ε_diag=[1.2e-04, -8.7e-05, -3.6e-05]

eps_dev = r_str.strain_tensor_deviatoric   # deviatoric tensor in crystal frame
eps_lab = r_str.strain_tensor_lab          # rotated to lab frame
```

---

## Layered crystal (stack) variants

For epitaxial stacks modelled as a
[`LayeredCrystal`][nrxrdct.laue.layers.LayeredCrystal], three stack-aware
counterparts mirror the single-crystal functions.  All three apply a **single
global rotation** to every layer, so inter-layer orientation relationships are
preserved.

### `refine_orientation_image_stack` — orientation only (3 parameters)

$$
\mathbf{U}_i = R(\delta\boldsymbol{\omega})\,\mathbf{U}_{0,i}
$$

```python
import nrxrdct.laue as laue

# precompute one allowed-hkl dict per unique crystal in the enumeration pool
hkl = {
    id(layer.crystal): laue.precompute_allowed_hkl(layer.crystal, E_max_eV=27000)
    for layer in stack.buffer_layers + stack.layers[:1]
}

result = laue.refine_orientation_image_stack(
    stack, camera, raw_frame,
    allowed_hkl   = hkl,
    kernel_sigma  = 0.3,
    max_angle_deg = 0.2,
    verbose       = True,
)
# StackImageRefinementResult [OK]  |δω|=0.018°  score=12340.5  Δscore=+890.2  n_sim=184

# apply refined orientations
for layer, U in zip(stack.all_layers, result.U_layers):
    layer.U = U
```

### `refine_strain_image_stack` — orientation + per-layer strain

$$
\mathbf{U}_{\text{eff},i} = R(\delta\boldsymbol{\omega})\,\mathbf{U}_{0,i}\,(\mathbf{I} + \boldsymbol{\varepsilon}_i)
$$

Parameter vector length: $3 + N_\text{layers} \times n_\text{strain}$.

```python
# warm-start strain from a prior peak-based fit
result = laue.refine_strain_image_stack(
    stack, camera, raw_frame,
    strain0_list  = prior.strain_tensors,   # from fit_strain_orientation_stack
    fit_strain    = ("e_xx", "e_yy", "e_zz"),
    allowed_hkl   = hkl,
    kernel_sigma  = 0.3,
    max_angle_deg = 0.2,
    verbose       = True,
)
# StackStrainImageRefinementResult [OK]  |δω|=0.004°  score=13102.1  Δscore=+762.4  n_sim=184

for i, voigt in enumerate(result.strain_voigts):
    print(f"Layer {i}: ε_zz = {voigt[2]:.2e}")
```

`strain0_list` accepts a list of `(3, 3)` tensors in `stack.all_layers` order.
Pass `result_peaks.strain_tensors` directly from
[`fit_strain_orientation_stack`][nrxrdct.laue.fitting.fit_strain_orientation_stack]
to warm-start the image refinement.

### Stack result fields

| Attribute | `refine_orientation_image_stack` | `refine_strain_image_stack` |
|---|---|---|
| `R_global` | Global rotation `(3, 3)` | same |
| `rotvec` | Rotation vector `(3,)` | same |
| `U_layers` | Refined `U` per layer | same |
| `U0_layers` | Starting `U` per layer | same |
| `U_eff_layers` | — | `R @ U0_i @ (I + ε_i)` per layer |
| `strain_tensors` | — | Per-layer `(3, 3)` strain tensors |
| `strain_voigts` | — | Per-layer Voigt `(6,)` |
| `score` / `score0` | Gaussian-weighted pixel score | same |
| `n_sim` | Total simulated spots | same |

---

## Recommended workflow

The image-based functions are designed as **post-processing steps** in the
standard pipeline, not replacements for it:

```
submit_segmentation  →  submit_orientation  →  submit_strain
                                                      ↓
                                          submit_image_refine          (orientation polish)
                                                      ↓
                                      submit_strain_image_refine       (strain polish)
```

Running the conventional pipeline first gives a good starting $\mathbf{U}_0$
and $\boldsymbol{\varepsilon}_0$.  The image steps then recover signal from
reflections that were too weak to segment.

!!! tip "Always pre-compute `allowed_hkl`"
    Without a pre-computed HKL list, `simulate_laue` recomputes the full
    reflection set on every optimizer call (hundreds of times per frame).
    Always pass `allowed_hkl = laue.precompute_allowed_hkl(crystal, E_max_eV=...)`.

---

## SLURM batch processing

Both functions have corresponding SLURM submission methods on
[`GrainMap`][nrxrdct.laue.map.GrainMap]:

| Step | Submit | Collect |
|---|---|---|
| Orientation polish | `gmap.submit_image_refine(...)` | `gmap.collect_image_refine(base_dir)` |
| Strain polish | `gmap.submit_strain_image_refine(...)` | `gmap.collect_strain_image_refine(base_dir)` |

Results land in `base_dir/img_refine/` and `base_dir/strain_img_refine/`
respectively.  Each method reads the starting $\mathbf{U}$ (and optionally
$\boldsymbol{\varepsilon}$) from a prior `ubs/` or `strain/` result specified
via `u_source`.

```python
# Submit orientation polish (reads U from strain/)
job_ids = gmap.submit_image_refine(
    base_dir,
    crystal,
    camera,
    h5_dataset    = "1.1/measurement/eiger4m",
    u_source      = "strain",
    kernel_sigma  = 0.3,
    max_angle_deg = 0.2,
    n_jobs        = 20,
    time          = "01:00:00",
    mem           = "8G",
)

# After jobs finish:
gmap.collect_image_refine(base_dir)   # updates self.U

# Submit strain polish
job_ids = gmap.submit_strain_image_refine(
    base_dir,
    crystal,
    camera,
    h5_dataset    = "1.1/measurement/eiger4m",
    u_source      = "strain",          # seeds strain0 from strain/ npz
    kernel_sigma  = 0.3,
    max_angle_deg = 0.2,
    n_jobs        = 20,
    time          = "02:00:00",
    mem           = "8G",
)

# After jobs finish:
gmap.collect_strain_image_refine(base_dir)   # updates self.U, strain_tensor, strain_voigt
```

---

## Search-based refinement

The local-polish functions above require a good starting $\mathbf{U}_0$ —
they converge to the wrong orientation if started more than `max_angle_deg`
away from the true one.  When only a rough estimate of the orientation is
available (e.g. a secondary grain whose orientation was not indexed by the
standard pipeline), use the **search** variants instead.

### `search_orientation_image` — grid search + local polish (3 parameters)

A two-phase approach:

1. **Grid search** — sample `n_search` orientations uniformly within a
   misorientation ball of radius `search_misor_deg` around `U_ref` using
   $r^{1/3}$ radial sampling for uniform 3-D coverage.  Each candidate is
   scored by direct intensity lookup at the simulated spot positions (no FFT —
   fast, O(n_spots × n_search)).
2. **Local refinement** — Powell from the best grid candidate with the full
   Gaussian-convolution FFT objective, constrained to ±`max_angle_deg` around
   that point.

```python
result = laue.search_orientation_image(
    crystal, U_ref, camera, raw_frame,
    allowed_hkl      = hkl,
    search_misor_deg = 5.0,    # grid-search radius — must cover the uncertainty in U_ref
    n_search         = 500,    # random candidates; increase for noisier images
    kernel_sigma     = 0.3,
    max_angle_deg    = 0.2,    # local-polish radius after grid search
)

U_found = result.U
print(f"|δω| from U_ref: {np.degrees(np.linalg.norm(result.rotvec)):.3f}°")
```

The result has the same type as `refine_orientation_image` (`ImageRefinementResult`),
so it is a drop-in replacement wherever that result is used.

### `search_strain_image` — grid search + local polish (9 parameters)

Same two-phase grid search (orientation only during the grid step — strain
shifts spots by $\lesssim 10^{-3}$ detector widths and is negligible at the
misorientation scale being searched), followed by a 9-parameter local Powell
that jointly refines orientation and all six strain components.

```python
result = laue.search_strain_image(
    crystal, U_ref, camera, raw_frame,
    allowed_hkl      = hkl,
    search_misor_deg = 5.0,
    n_search         = 500,
    kernel_sigma     = 0.3,
    max_angle_deg    = 0.2,
    strain_scale     = 1e-4,
)

U_found    = result.U
eps_tensor = result.strain_tensor         # (3, 3)
eps_voigt  = result.strain_voigt          # (6,)
```

---

## SLURM batch processing — search variants

For raster maps where the orientation of one grain slot is only approximately
known, use the search SLURM methods.  The starting orientation is supplied as
`U_refs` — a dict mapping grain index to a reference matrix — rather than
reading per-pixel fits from disk.

| Step | Submit | Collect | Output dir |
|---|---|---|---|
| Orientation search | `gmap.submit_search_image_refine(...)` | `gmap.collect_search_image_refine(base_dir)` | `search_img_refine/` |
| Orientation + strain search | `gmap.submit_search_strain_image_refine(...)` | `gmap.collect_search_strain_image_refine(base_dir)` | `search_strain_img_refine/` |

Unlike the local-polish methods, these methods do **not** read prior per-pixel
fits — the same `U_refs[gi]` is used as the grid-search centre for every map
pixel.  Fields populated by the standard pipeline (`match_rate`, `cost`, …)
are **not** written; only `U` (and strain arrays for the strain variant) are
updated.

```python
# Rough reference orientation for grain 1
U_ref_1 = np.array([[...], [...], [...]])   # (3, 3)

# Submit orientation search
job_ids = gmap.submit_search_image_refine(
    base_dir,
    crystal,
    camera,
    h5_dataset       = "1.1/measurement/eiger4m",
    U_refs           = {1: U_ref_1},
    n_jobs           = 80,
    search_misor_deg = 5.0,
    n_search         = 500,
    kernel_sigma     = 0.3,
    max_angle_deg    = 0.2,
    partition        = "nice",
    time             = "04:00:00",
    mem              = "32G",
    cpus_per_task    = 20,
    python_bin       = "/path/to/python",
)

# After jobs finish:
gmap.collect_search_image_refine(base_dir)   # updates self.U[1]

# Submit orientation + strain search
job_ids = gmap.submit_search_strain_image_refine(
    base_dir,
    crystal,
    camera,
    h5_dataset       = "1.1/measurement/eiger4m",
    U_refs           = {1: U_ref_1},
    n_jobs           = 80,
    search_misor_deg = 5.0,
    n_search         = 500,
    kernel_sigma     = 0.3,
    max_angle_deg    = 0.2,
    partition        = "nice",
    time             = "04:00:00",
    mem              = "32G",
    cpus_per_task    = 20,
    python_bin       = "/path/to/python",
)

# After jobs finish:
gmap.collect_search_strain_image_refine(base_dir)
# updates self.U[1], self.strain_tensor[1], self.strain_voigt[1],
# self.strain_tensor_deviatoric[1]
```

!!! note "HDF5 read-only guarantee"
    All search workers open the raw data file with `"r"` mode.  The master
    HDF5 file is never modified.

!!! note "Inspecting search-refined grains"
    Because the search methods do not fill `match_rate` / `cost`, the default
    left-panel of `inspect_frame` will be blank for these grain slots.  Use
    `map_quantity='misorientation'` instead, which works for any grain that
    has `U` filled:

    ```python
    gmap.inspect_frame(
        crystal, camera, base_dir,
        h5_dataset   = "1.1/measurement/eiger4m",
        grains       = [1],
        map_grain    = 1,
        map_quantity = 'misorientation',
    )
    ```

---

## Parameter reference

| Parameter | Default | Description |
|---|---|---|
| `kernel_sigma` | 0.3 px | Gaussian kernel width at simulated spot positions.  Decrease to decouple spots that are a few pixels apart. |
| `bg_sigma` | 251 px | Background Gaussian width.  Large values remove only the slow pedestal. |
| `max_angle_deg` | 0.2° | Half-width of the local-polish rotation search space per axis. |
| `search_misor_deg` | 5.0° | Radius of the grid-search misorientation ball (search variants only). |
| `n_search` | 500 | Number of random grid-search candidates per frame (search variants only). |
| `allowed_hkl` | `None` | Pre-computed HKL set.  Always provide this for speed and correctness. |
| `strain_scale` | `1e-4` | Internal divisor for strain parameters (strain refinement only). |
| `method` | `'Powell'` | `scipy.optimize.minimize` method. |
| `verbose` | `False` | Print starting score and convergence summary. |
