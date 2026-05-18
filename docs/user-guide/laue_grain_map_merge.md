# Multi-grain merging and symmetry reduction in micro-Laue maps

This page describes two closely related post-processing steps for
`GrainMap` objects:

1. **Grain merging** — combining independent per-grain fits into a single
   spatially consistent orientation map.
2. **Symmetry reduction** — resolving isolated pixels whose orientations are
   physically correct but use a different symmetry-equivalent representation
   from their neighbours.

Both steps are usually needed when mapping a sample that contains more than
one crystallographic orientation (subgrains, twin variants, oriented thin
films, …).

---

## 1. Why multiple grains need to be merged

A `GrainMap` stores the result of fitting every map pixel against a fixed
set of *reference* orientations `U_ref[0], U_ref[1], …`.  Each reference
is defined once (pre-indexed manually or via `index_orientation`) and the
fitting routine then refines it pixel by pixel independently.

This means that at every map position there are **N independent fits**, one
per reference grain.  In a real sample a given pixel belongs to at most one
grain, so the map contains a lot of redundant (and often poor) fits.  The
merging step selects the best fit at each position, collapsing N maps into one.

---

## 2. The merging workflow

### 2.1 `merge` — select the best grain per pixel

```python
best_grain, metrics = gmap.merge(
    metric="match_rate",       # primary ranking criterion
    min_match_rate=0.25,       # discard fits below this threshold
    min_n_matched=3,           # require at least 3 matched spots
    max_rms_px=2.0,            # discard fits with large residuals
)
```

`best_grain` is an `(ny, nx)` integer array.  `best_grain[iy, ix]` is the
index (0-based) of the grain slot that won the comparison at position
`(iy, ix)`.  Positions where **all** grains failed the quality filters
receive `best_grain = -1`.

`metrics` is a dictionary of `(ny, nx)` arrays containing the quality
values *of the winning grain*:

| Key | Meaning |
|---|---|
| `"match_rate"` | Fraction of observed spots matched by the winner |
| `"rms_px"` | RMS pixel residual of the matched pairs |
| `"mean_px"` | Mean pixel residual |
| `"n_matched"` | Number of matched spots (integer) |
| `"cost"` | Optimizer cost at convergence |
| `"U"` | Orientation matrix of the winner, shape `(ny, nx, 3, 3)` |
| `"source"` | The `source` argument (scalar string) |

`merge` is **read-only**: it does not modify the `GrainMap` object.

#### Quality metrics

The default ranking criterion is `"match_rate"` (= `n_matched / n_observed`).
It is defined on [0, 1] regardless of how many peaks are in the pattern, so
it is directly comparable across all map positions without any normalisation.

`"rms_px"` is useful as a secondary filter (via `max_rms_px`) once
`min_match_rate` already ensures a minimum coverage: a low RMS among very
few matches can be misleading.

#### Quality filters

Three optional filters are applied before ranking:

| Filter | Default | Effect |
|---|---|---|
| `min_match_rate` | `0.0` | Exclude any grain fit below this match rate |
| `min_n_matched` | `1` | Exclude fits with fewer matched spots |
| `max_rms_px` | `∞` | Exclude fits with large pixel residuals |

A position where *every* grain fails at least one filter receives
`best_grain = -1` and `NaN` metric values.  It will appear as a white
(masked) pixel in all maps.

Typical starting values: `min_match_rate=0.2`, `min_n_matched=3`.

### 2.2 `apply_merge` — register the result as a new grain slot

```python
gi_merged = gmap.apply_merge(best_grain, metrics)
print(f"Merged grain stored in slot {gi_merged}")
```

`apply_merge` appends a new grain slot (grain index `n_grains - 1` after
the call) that is filled from `best_grain` and `metrics`.  All existing
analysis and plotting methods work on this slot as on any other:

```python
gmap.plot_map("match_rate", grain=gi_merged)
gmap.plot_ipf_map(gi_merged, direction=[0, 0, 1])
```

`apply_merge` is **idempotent**: calling it a second time with updated
parameters overwrites the same grain slot in-place rather than creating a
new one.  The slot index `gi_merged` is stored in `gmap._merged_grain`.

`inspect_frame` uses the merged grain by default when it exists:

```python
gmap.inspect_frame(frame_idx)   # automatically uses gi_merged
```

### 2.3 `write_merge_links` — on-disk persistence

After merging you can write symlinks so that the merged selection is visible
on disk in the same format as the per-grain result files:

```python
gmap.write_merge_links(
    base_dir="./processing",
    best_grain=best_grain,
    metrics=metrics,           # inherits the source directory automatically
    overwrite=True,
)
```

This creates a `merged/` subfolder inside `base_dir` and populates it with
`frame_NNNNN_g{gi_merged:02d}.npz` symlinks, each pointing to the
corresponding `ubs/` or `strain/` result file of the winning grain.

To re-load the merged map from those links in a new session:

```python
gi_loaded = gmap.collect_merged(base_dir="./processing")
```

`collect_merged` auto-detects whether the files contain orientation-only or
strain results by checking for the `"strain_voigt"` key.

---

## 3. Symmetry-equivalent orientations

### 3.1 The problem

In polychromatic (white-beam) Laue diffraction, the positions of spots on
the detector depend on the orientation matrix **U** and the reciprocal lattice
**B**, but are **independent of the X-ray wavelength**.  A direct consequence is
that every proper rotation **S** in the crystal's point group leaves the spot
pattern invariant:

$$
\text{spots}(\mathbf{U}) \equiv \text{spots}(\mathbf{U}\,\mathbf{S})
\quad \forall\, \mathbf{S} \in G
$$

This means the residual landscape has N\ :sub:`sym` equivalent global minima
(N\ :sub:`sym` = 24 for cubic, 12 for hexagonal, 8 for tetragonal, 4 for
orthorhombic).  A least-squares optimizer starting from slightly different
initial conditions can converge to any of them.

During a raster scan, pixels are refined independently.  Pixels near grain
boundaries or in strained regions are more sensitive to the initial conditions
than those at the core of a well-crystallised grain.  As a result, the
optimized map may contain **isolated pixels** whose orientation *looks*
different from their neighbours but is in fact physically equivalent — it just
belongs to a different member of the symmetry family.

The effect is most visible in:

- **IPF maps**: an isolated pixel has a very different colour from its
  surroundings even though the physical misorientation is zero.
- **Euler angle maps**: a discontinuous jump of, e.g., 90° between neighbours
  that should be perfectly continuous.
- **KAM (Kernel Average Misorientation)**: artificially elevated values at
  isolated pixels.

### 3.2 The solution: `reduce_to_fundamental_zone`

`reduce_to_fundamental_zone` relabels every pixel to the symmetry-equivalent
that is closest (minimum misorientation angle) to a common reference **R**:

$$
s^*(\text{iy, ix}) = \underset{s}{\operatorname{argmax}}\;
                       \operatorname{tr}\!\bigl(R^T\,\mathbf{U}^{(s)}\bigr),
\quad
\mathbf{U}^{(s)} = \mathbf{U} \cdot \mathbf{S}_s
$$

Maximising the matrix trace is equivalent to minimising the geodesic
misorientation angle
$\omega = \arccos\!\left(\tfrac{\operatorname{tr}(R^T U^{(s)})-1}{2}\right)$.

```python
changed = gmap.reduce_to_fundamental_zone(
    grain=0,
    symmetry="cubic",     # point group
    reference=None,       # None = quaternion mean of all valid pixels
)
print(f"{changed.sum()} pixels relabelled")
```

The method modifies `gmap.U[grain]` **in place**.  It returns a boolean mask
of the pixels that were actually changed.

#### Reference orientation

When `reference=None` (default) the target **R** is computed as the
**quaternion mean** of all valid map pixels:

1. Each U matrix is converted to a unit quaternion.
2. Quaternions that point to the opposite hemisphere from the first are
   flipped (`q ← −q`), avoiding cancellation artefacts in the average.
3. The flipped quaternions are averaged and re-normalised.
4. The mean quaternion is converted back to a rotation matrix.

This works well when one grain dominates the map.  If the grain of interest
occupies only a minority of pixels (e.g. in a merged map with two distinct
grains), supply the centre of that grain explicitly:

```python
# Use the median orientation of grain 1 as reference
U_grain1 = gmap.U[1][grain1_mask]          # shape (M, 3, 3)
from scipy.spatial.transform import Rotation
q = Rotation.from_matrix(U_grain1).as_quat()
q = np.where((q @ q[0]) < 0, -q, q)
R_ref = Rotation.from_quat(q.mean(axis=0) / np.linalg.norm(q.mean(axis=0))).as_matrix()

changed = gmap.reduce_to_fundamental_zone(grain=1, symmetry="cubic", reference=R_ref)
```

#### Strain tensor rotation

When strain data are present the strain tensor is rotated consistently with
the symmetry operation:

$$
\boldsymbol{\varepsilon}' = \mathbf{S}^T\,\boldsymbol{\varepsilon}\,\mathbf{S}
$$

This keeps the physical meaning of the strain components aligned with the
(now corrected) crystal frame.  The Voigt representation is automatically
updated in `gmap.strain_voigt[grain]`.

> **Note**: this rotation makes the *representation* consistent but does not
> re-optimise the strain values.  If the corrected pixels are numerous, a
> re-refinement starting from the corrected **U** will give more accurate
> strain values.

---

## 4. Recommended workflow

```
1.  Pre-index reference orientations
        index_orientation / interactive_orientation

2.  Run per-grain fits across the full map
        fit_orientation / fit_strain_orientation

3.  For each grain: resolve symmetry equivalents
        gmap.reduce_to_fundamental_zone(grain=gi, symmetry="cubic")

4.  Inspect individual grain maps
        gmap.plot_ipf_map(gi, direction=[0, 0, 1])
        gmap.plot_map("match_rate", grain=gi)

5.  Merge
        best_grain, metrics = gmap.merge(
            metric="match_rate",
            min_match_rate=0.25,
            min_n_matched=3,
        )

6.  Register merged slot
        gi_merged = gmap.apply_merge(best_grain, metrics)

7.  Optionally: re-run symmetry reduction on the merged slot
        gmap.reduce_to_fundamental_zone(grain=gi_merged, symmetry="cubic")

8.  Persist to disk
        gmap.write_merge_links("./processing", best_grain, metrics)
        gmap.save("grainmap.h5")

9.  Analyse
        gmap.plot_overview(grain=gi_merged)
        gmap.plot_ipf_map(gi_merged, direction=[0, 0, 1], stretch="local",
                          best_grain=best_grain)
        gmap.plot_map("match_rate", grain=gi_merged)
```

Step 3 (symmetry reduction) should be performed *before* merging.  The merge
step compares quality metrics and selects the best grain; running symmetry
reduction first ensures that the merged orientations are in a consistent
representation.

---

## 5. Plotting merged results

### IPF maps

```python
# Global colour stretch — spans entire map range
gmap.plot_ipf_map(gi_merged, direction=[0, 0, 1], stretch="global")

# Per-grain colour stretch — each grain region stretched independently
# (requires best_grain to define the regions)
gmap.plot_ipf_map(gi_merged, direction=[0, 0, 1],
                  stretch="local", best_grain=best_grain)
```

`stretch="local"` is most useful when the map contains two (or more)
distinct grains: with global stretching the inter-grain colour contrast
dominates the gamut and the intra-grain spread is compressed.  Local
stretching gives each grain region its own colour scale, revealing the
internal orientation spread within each grain.

### Overview map

```python
fig, axes = gmap.plot_overview(
    grain=gi_merged,
    ipf_axes=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    quality_metrics=["match_rate", "rms_px", "kam"],
    show_strain=True,
    strain_components=["exx", "eyy", "ezz"],
    strain_frame="crystal",
)
```

### Frame inspection

```python
gmap.inspect_frame(frame_idx)   # defaults to gi_merged when it exists
gmap.inspect_frame(frame_idx, map_grain=0)   # force a specific grain
```
