# Orientation mapping and visualisation

This page describes the orientation-related plots available in `GrainMap`,
the crystallographic concepts behind them, the reference frames involved, and
how to interpret the output.

---

## Background: the orientation matrix U

After fitting, each map pixel stores a **3×3 orientation matrix** $\mathbf{U}$
that describes how the crystal is rotated relative to the laboratory frame:

$$
\mathbf{r}_\text{lab} = \mathbf{U}\,\mathbf{r}_\text{crystal}
$$

- Columns of $\mathbf{U}$ are the crystal axes **a**, **b**, **c** expressed in
  lab coordinates.
- Rows of $\mathbf{U}$ are the lab axes **X**, **Y**, **Z** expressed in
  crystal coordinates.

A pixel where the orientation fit failed stores `NaN` in all U entries and is
shown as a neutral colour (white or grey) in every orientation map.

---

## Scalar quality maps

`GrainMap.plot_map` plots per-pixel scalar quantities.  The quantities most
relevant to orientation quality are:

| `quantity` argument | Description | Good values |
|---|---|---|
| `"match_rate"` | Fraction of observed spots matched to a simulated reflection | ≥ 0.3–0.5 |
| `"n_matched"` | Absolute number of matched spots | ≥ 5–10 |
| `"rms_px"` | RMS Euclidean distance (pixels) between observed and simulated spot positions | ≤ 2–5 px |
| `"mean_px"` | Mean Euclidean distance (pixels) — less sensitive to outlier spots than RMS | ≤ 2–5 px |
| `"cost"` | Final value of the least-squares cost function | lower = better |
| `"misorientation"` | Rotation angle (°) from a reference orientation | ≈ 0° for a uniform grain |
| `"euler_phi1"` / `"euler_Phi"` / `"euler_phi2"` | ZXZ Euler angles (°) | — |

### Pixel deviation maps

`plot_map("rms_px")` and `plot_map("mean_px")` (or the convenience wrapper
`plot_mean_px`) visualise the *goodness of fit* spatially.

- A **uniform, low-residual** map means the Laue pattern can be explained
  consistently across the scan — the grain is well-identified.
- **Localised hot-spots** (high residuals at isolated pixels) often indicate
  double-grain positions, sub-grain boundaries, or strong local strain
  gradients that violate the single-crystal assumption.
- A **gradually increasing** residual towards the sample edges can signal
  detector calibration drift or a systematic error in the sample geometry.

`mean_px` is preferred over `rms_px` when the spot list contains a few
unusually bright or anomalous spots: the mean is less sensitive to those
outliers.

```python
# All grains in one figure, shared colour scale
fig, axes = gmap.plot_mean_px(
    motor_x="pz", motor_y="py",
    motor_units={"pz": "mm", "py": "mm"},
)

# Single grain, manual colour range
fig, ax = gmap.plot_mean_px(
    grains=[0], vmin=0, vmax=5,
    motor_x="pz", motor_y="py",
    motor_units={"pz": "mm", "py": "mm"},
)

# Via the general interface
fig, ax = gmap.plot_map("rms_px", grain=0,
                        motor_x="pz", motor_y="py",
                        motor_units={"pz": "mm", "py": "mm"})
```

### Misorientation map

The misorientation map shows, for each pixel, the smallest rotation angle
(in degrees) required to bring the local orientation into coincidence with a
reference orientation:

$$
\omega_\text{misor} = \left|\text{Rotation}(\mathbf{U}\,\mathbf{U}_\text{ref}^{\top})\right|
$$

The reference defaults to `U_ref[grain]` (the starting matrix used for
indexing); it can be overridden via `GrainMap.misorientation_map(grain,
reference=my_U)`.

**Interpretation:**

- Values close to **0°** indicate that the crystal orientation at that pixel
  matches the reference — the grain is nearly strain-free and defect-free.
- **Low-angle variations** (< 1°) across a single grain arise from lattice
  curvature, long-range elastic strain gradients, or gentle sub-grain
  misorientation.
- **Sharp jumps** at grain boundaries are expected and confirm that the map
  correctly separates distinct orientations.
- Typical elastic misorientation in lightly strained thin films is a few
  tenths of a degree; heavily deformed metals can reach several degrees.

```python
fig, ax = gmap.plot_map(
    "misorientation", grain=0,
    motor_x="pz", motor_y="py",
    motor_units={"pz": "mm", "py": "mm"},
    vmax=2.0,           # clip colour scale at 2°
)
```

### Euler angle maps

The ZXZ Euler angles $(\varphi_1, \Phi, \varphi_2)$ parametrise the rotation
from the crystal frame to the lab frame.  They are a compact representation of
orientation but are **not unique** near $\Phi = 0$ (gimbal lock) and are
**non-linearly related** to physical rotation axes, so spatial gradients should
be interpreted with care.

Prefer the **misorientation map** or the **IPF map** for quantitative analysis;
Euler maps are most useful for getting a quick overview of orientation spread.

```python
fig, ax = gmap.plot_map("euler_phi1", grain=0, cmap="hsv")
fig, ax = gmap.plot_map("euler_Phi",  grain=0, cmap="viridis")
fig, ax = gmap.plot_map("euler_phi2", grain=0, cmap="hsv")
```

---

## Inverse Pole Figure (IPF) map

`GrainMap.plot_ipf_map` colours each map pixel by the crystal direction that
is aligned with a chosen reference direction in the lab or sample frame.

### Concept

For a given reference direction $\hat{\mathbf{n}}$ (expressed in the lab
frame), the **crystal direction** parallel to it at each pixel is:

$$
\hat{\mathbf{c}} = \mathbf{U}^{\top}\,\hat{\mathbf{n}}
$$

This vector is then **mapped to an RGB colour** using the standard cubic IPF
colour scheme:

| Colour | Crystal direction |
|---|---|
| Blue | ⟨001⟩ family |
| Green | ⟨011⟩ family |
| Red | ⟨111⟩ family |
| Intermediate | Directions between the above poles |

White pixels indicate positions where no orientation was fitted.

### Reference directions and frames

The `axis` parameter selects the reference direction, and `frame` selects
the coordinate system in which that direction is defined:

| `frame` | Meaning |
|---|---|
| `"lab"` | Axis is in the lab frame (X, Y, Z of the diffractometer) |
| `"sample"` | Axis is in the sample frame; converted to lab via the inverse of the sample tilt |

The sample frame is related to the lab frame by a rotation of `sample_tilt_deg`
degrees about `sample_tilt_axis` (default −40° about Y).  Specifying
`frame="sample"` and `axis="z"` therefore asks *"which crystal direction is
parallel to the sample surface normal?"*

```python
# Which crystal direction is parallel to the lab Z axis?
fig, ax = gmap.plot_ipf_map(
    axis="z", grain=0, frame="lab",
    motor_x="pz", motor_y="py",
    motor_units={"pz": "mm", "py": "mm"},
    show_colorkey=True,
)

# Which crystal direction is parallel to the sample surface normal?
fig, ax = gmap.plot_ipf_map(
    axis="z", grain=0, frame="sample",
    sample_tilt_deg=-40.0, sample_tilt_axis="y",
    motor_x="pz", motor_y="py",
    motor_units={"pz": "mm", "py": "mm"},
)

# Custom reference direction (arbitrary lab vector)
fig, ax = gmap.plot_ipf_map(
    axis=[0, 1, 1], grain=0, frame="lab",
)
```

### The colour key

Pass `show_colorkey=True` (the default) to overlay the standard [001]–[011]–[111]
triangle in the lower-right corner of the plot.  The triangle is rendered with
the same colour function used for the map itself, so the colour of any pixel
corresponds directly to a position within the triangle.

### Interpretation

- A **uniform colour** across the map means the crystal orientation is nearly
  the same everywhere — a single, defect-free grain.
- **Gradual colour drift** indicates a systematic lattice rotation (e.g. grain
  curvature, tilt from mounting).
- **Sharp colour boundaries** are grain boundaries or twin boundaries.
- **Speckle** (random pixel-to-pixel colour variation) arises from poor fit
  quality: check the `match_rate` and `rms_px` maps to confirm.

---

## IPF scatter plot

`GrainMap.plot_ipf_scatter` creates a three-panel scatter pole figure showing
where the **a**, **b**, and **c** crystal axes point in the chosen frame, for
every fitted map pixel.

### What is plotted

Each panel projects the tip of the unit crystal-axis vector onto the XY plane
of the chosen frame.  For `frame="lab"` the horizontal axis is the lab X
direction and the vertical axis is lab Y.  Points are coloured with the same
cubic IPF scheme (colour = crystal direction → RGB), so the colour encodes
which crystal family is pointing in that direction.

The black circle marks the unit sphere equator; the cross-hairs mark the X and
Y axes.

### When to use scatter vs map

| | IPF map | IPF scatter |
|---|---|---|
| **Spatial distribution** | Yes — colour encodes orientation at each pixel | No — spatial information is lost |
| **Full orientation spread** | No — only one axis shown at a time | Yes — all three crystal axes in one figure |
| **Fibre texture detection** | Weak | Strong — a fibre shows as an arc in one panel |
| **Point texture** | Visible as uniform colour in map | Visible as tight cluster in scatter |

```python
# Scatter in the lab frame
fig, axes = gmap.plot_ipf_scatter(
    grain=0, frame="lab",
    s=20, alpha=0.7,
)

# Scatter in the sample frame
fig, axes = gmap.plot_ipf_scatter(
    grain=0, frame="sample",
    sample_tilt_deg=-40.0, sample_tilt_axis="y",
)
```

---

## Reference frames summary

| Frame | Definition | How to select |
|---|---|---|
| **Lab** | Instrument coordinate system; Z along beam or detector normal, X/Y in the detector plane | `frame="lab"` |
| **Sample** | Lab frame rotated by `sample_tilt_deg` about `sample_tilt_axis` to align with the physical sample surface | `frame="sample"` |
| **Crystal** | Unit cell axes as defined by the xrayutilities `Crystal` object | Implicit in `plot_map` (orientation matrix U relates crystal ↔ lab) |

The sample tilt defaults to −40° about the lab Y axis, which is the standard
incidence geometry at many Laue micro-diffraction beamlines.  Override with
`sample_tilt_deg` and `sample_tilt_axis` on any plot method.

---

## Practical workflow

A typical orientation-analysis session after collecting results:

```python
# 1. Overview: how many spots were matched, and how well?
gmap.plot_map("match_rate", grain=0,
              motor_x="pz", motor_y="py",
              motor_units={"pz": "mm", "py": "mm"})

gmap.plot_mean_px(motor_x="pz", motor_y="py",
                  motor_units={"pz": "mm", "py": "mm"})

# 2. Spatial orientation map — which crystal direction faces the beam?
gmap.plot_ipf_map(axis="z", grain=0, frame="lab",
                  motor_x="pz", motor_y="py",
                  motor_units={"pz": "mm", "py": "mm"})

# 3. Misorientation — how much does the grain rotate across the scan?
gmap.plot_map("misorientation", grain=0, vmax=1.5,
              motor_x="pz", motor_y="py",
              motor_units={"pz": "mm", "py": "mm"})

# 4. Full orientation texture — scatter of all three crystal axes
gmap.plot_ipf_scatter(grain=0, frame="sample")
```

!!! tip "Combining maps"
    For a quick side-by-side comparison use the `ax` argument to place multiple
    plots on a shared figure:

    ```python
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    gmap.plot_map("match_rate",    grain=0, ax=axes[0])
    gmap.plot_map("misorientation", grain=0, ax=axes[1], vmax=2)
    gmap.plot_ipf_map("z",          grain=0, ax=axes[2])
    fig.tight_layout()
    ```
