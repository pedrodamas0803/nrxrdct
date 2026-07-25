# Texture tomography

This page describes `nrxrdct.texture`, which extracts pole-figure data from
the same diffraction images used by the XRD-CT pipeline and inverts it into
an orientation distribution function (ODF), instead of the scalar
density/phase maps `nrxrdct.xrdct` produces.

> **Prerequisite**: see [Texture Tomography Theory](texture_theory.md) for
> the pole-figure/ODF background, the diffraction-geometry derivation, and —
> importantly — the coverage and uniqueness caveats referenced throughout
> this page. See [Azimuthal Integration](azimuthal_integration.md) and
> [Quickstart](quickstart.md) for the powder side of the pipeline this
> module reuses.

---

## 1. Why this exists

Azimuthal integration collapses each detector frame into a 1-D powder
pattern; scanning-3DXRD ([s3DXRD](s3dxrd.md)) segments and indexes Bragg
spots grain-by-grain. Texture tomography sits between the two: it reads the
same raw frames but, for one or more chosen `hkl` rings, keeps the
**azimuthal (χ) intensity distribution** around the ring instead of either
averaging it away or indexing individual spots — the raw material a pole
figure (and, from several pole figures, an ODF) is built from.

```
Raw 2-D frames (same master_file as azimuthal integration / s3dxrd)
    │
    ▼  assemble_pole_figure_data() / nrxrdct.texture.slurm_pole_figures  — per-frame ring extraction
    │                                                                       -> pole_figures/<hkl>/scan_XXXX
    ▼  assemble_pole_figure_sinogram()  — stack into (n_chi, n_lines, n_rot) sinogram
    │
    ▼  load_pole_figures()              — convert to (alpha, beta, intensity) pole-figure points
    │
    ▼  compute_odf()                    — WIMV inversion -> discretised ODF
    │
    ▼  plot_pole_figure() / plot_pole_figure_comparison()  — visualise / QA the fit
```

Each stage writes/returns data independently, matching the rest of the
XRD-CT pipeline (see [Typical Workflow](workflow.md)).

### Design notes

- **Bulk / per-line only.** Every ODF `compute_odf` produces aggregates (or,
  with `line_index` set, uses one line of) the *entire* beam path through
  the sample — it is not a per-voxel reconstruction. See
  [Theory §6](texture_theory.md#6-beyond-bulk-per-voxel-reconstruction-experimental-not-shipped)
  for why per-voxel reconstruction is a substantially harder, unsolved
  problem in this package.
- **Incomplete pole-figure coverage by construction.** The single vertical
  rotation axis (no tilt) only ever sweeps a *curve* on the pole sphere per
  `hkl`, not a full hemisphere — combine several `hkl` rings to build up
  coverage. See [Theory §5](texture_theory.md#5-coverage-why-wimv-not-the-harmonic-method).
- **χ = 0 direction needs calibration.** `pole_figure_coordinates` assumes χ
  is measured from the vertical detector axis; verify against your own setup
  via `chi_offset_deg`/`chi_sign` — pyFAI's own default was empirically
  found to differ from this assumption. See
  [Theory §3](texture_theory.md#calibration-where-is-χ--0).
- **Hemisphere folding defaults off.** `compute_odf`'s `fold_hemisphere`
  parameter defaults to `False` because folding was found, during
  development, to occasionally converge to a ~180°-misoriented "ghost"
  solution instead of the true orientation. See
  [Theory §4](texture_theory.md#4-friedels-law-and-the-ghost-ambiguity).

---

## 2. Quick example

```python
from pathlib import Path
import numpy as np

from nrxrdct.texture import (
    assemble_pole_figure_data,
    load_pole_figures,
    compute_odf,
    plot_pole_figure_comparison,
    recalculate_pole_figure,
)

master_file = Path("data/sample_master.h5")
poni_file   = Path("data/detector.poni")
mask_file   = Path("data/mask.edf")
output_file = Path("data/pole_figures.h5")

# 1. Extract pole-figure data for each hkl ring you want (repeat per hkl;
#    or submit via nrxrdct.texture.slurm_pole_figures for a full dataset)
rings = {"111": (4.0, 6.0), "200": (7.0, 9.0), "220": (11.0, 13.0)}
for hkl, tth_range in rings.items():
    assemble_pole_figure_data(
        master_file=master_file, output_file=output_file,
        poni_file=poni_file, mask_file=mask_file,
        tth_range=tth_range, hkl_label=hkl,
    )

# 2. Convert the merged file into compute_odf's input format
pole_figures = load_pole_figures(
    pole_figure_file=output_file,
    hkl_labels=list(rings.keys()),
    n_rot=901,          # rotation steps per scan line
    line_index=None,    # None = aggregate all lines into one bulk pole figure
)

# 3. Crystal directions (cubic example: normalised [h,k,l])
crystal_directions = {
    "111": np.array([1, 1, 1]) / np.sqrt(3),
    "200": np.array([1, 0, 0]),
    "220": np.array([1, 1, 0]) / np.sqrt(2),
}

# 4. Invert to an ODF
result = compute_odf(pole_figures, crystal_directions, step_deg=10.0, smoothing_deg=7.5, n_iter=10)

# 5. QA: compare measured vs. recalculated pole figure for one hkl
alpha, beta, measured = pole_figures["111"]
recalculated = recalculate_pole_figure(result, crystal_directions["111"], alpha, beta)
plot_pole_figure_comparison(alpha, beta, measured, recalculated, title="111 pole figure")
```

---

## 3. Stage-by-stage

### 3.1 Pole-figure extraction — `nrxrdct.texture.odf`

```python
from nrxrdct.texture import extract_ring_intensity, assemble_pole_figure_data

assemble_pole_figure_data(
    master_file=master_file, output_file=output_file,
    poni_file=poni_file, mask_file=mask_file,
    tth_range=(4.0, 6.0), hkl_label="111",
    background_ranges=((3.0, 3.8), (6.2, 7.0)),  # optional flanking-window background subtraction
)
```

Mirrors `nrxrdct.azimuthal.integration.integrate_powder_parallel` (same
master-file layout, entry validation) but cake-integrates each frame and
reduces it to an azimuthal (χ) intensity profile via
`extract_ring_intensity`, instead of a 1-D powder pattern. Results are
appended to `output_file` under `pole_figures/<hkl_label>/scan_XXXX`; a
partially-completed run resumes automatically (scans already present are
skipped).

| Parameter | Default | Description |
|---|---|---|
| `tth_range` | required | `(low, high)` 2θ window bracketing the `hkl` ring |
| `hkl_label` | required | Ring identifier, used as an HDF5 group name |
| `background_ranges` | `None` | Flanking 2θ windows for background subtraction |
| `npt_azim` | `360` | Number of azimuthal (χ) bins |

### 3.2 HPC distribution — `nrxrdct.texture.slurm_pole_figures`

For a full dataset, `assemble_pole_figure_data` is too slow to run on one
machine — distribute it across SLURM the same way as the powder pipeline
(see [SLURM / HPC Integration](slurm.md)). Unlike the single-hkl
`assemble_pole_figure_data` above, `launch()` accepts **any number of hkls
in one call** — each worker cake-integrates every frame once and extracts
every requested ring from that single cake, so extracting several hkls costs
one pass over the raw frames, not one pass per hkl:

```python
from nrxrdct.texture.slurm_pole_figures import launch, monitor, merge, check

result = launch(
    master_file=master_file, output_file=output_file,
    poni_file=poni_file, mask_file=mask_file,
    hkls={"111": (4.0, 6.0), "200": (7.0, 9.0), "220": (11.0, 13.0)},
    n_jobs=8, partition="nice", conda_env="nrxrdct",
)
monitor(result["slurm_ids"], result["tmp_dir"], watch=True)
merge(tmp_dir=result["tmp_dir"], output_file=output_file)          # writes every hkl
check(tmp_dir=result["tmp_dir"], output_file=output_file)          # checks every hkl by default
```

Workers stream frames in batches and write per-scan tmp files (never the
final HDF5 directly, avoiding concurrent-write corruption); `merge` is the
sole, single-process writer of the final file. `repair()` resubmits any
scans found missing by `check()`, re-reading all settings from the
`launch_meta.json` sidecar so nothing needs repeating.

Also available as a CLI: `nrxrdct-slurm-texture launch|monitor|merge|check`.

### 3.3 ODF inversion — `nrxrdct.texture.odf_inversion`

```python
from nrxrdct.texture import load_pole_figures, compute_odf

pole_figures = load_pole_figures(output_file, hkl_labels=["111", "200", "220"], n_rot=901)
result = compute_odf(pole_figures, crystal_directions, step_deg=10.0, smoothing_deg=7.5, n_iter=10)
```

| Parameter | Default | Description |
|---|---|---|
| `step_deg` | `10.0` | Euler-angle orientation-grid spacing; halving it multiplies grid size (and run time) by ~8 |
| `smoothing_deg` | `7.5` | Gaussian kernel width — the effective angular resolution of the recovered ODF |
| `n_iter` | `10` | WIMV iterations |
| `fold_hemisphere` | `False` | See the ghost-ambiguity caveat above — leave off unless you've verified it's safe for your case |

`result["f"]` is the fitted ODF value per orientation-grid cell
(`result["euler_deg"]`), normalised so `sum(f * cell_weight) / sum(cell_weight) == 1`.
`result["rp_history"]` tracks the per-iteration relative pole-figure
residual for convergence checking — but see
[Theory §6](texture_theory.md#6-beyond-bulk-per-voxel-reconstruction-experimental-not-shipped)
before assuming a low residual alone proves the global optimum was found.

### 3.4 Plotting / QA — `nrxrdct.texture.texture_plotting`

```python
from nrxrdct.texture import plot_pole_figure, plot_pole_figure_comparison

plot_pole_figure(alpha_deg, beta_deg, intensity, title="111")
plot_pole_figure_comparison(alpha_deg, beta_deg, measured, recalculated)
```

Both render a standard equal-area (Schmidt) upper-hemisphere projection,
folding any `alpha > 90°` points onto their antipodal equivalent for
display (Friedel's law) — display-time folding only; this is unrelated to,
and does not re-introduce, the fit-time ghost-ambiguity risk in §3.3.

---

## 4. See also

- [Texture Tomography Theory](texture_theory.md) — pole figures, ODFs, the
  diffraction-geometry derivation, WIMV, and the ghost/coverage caveats.
- [Azimuthal Integration](azimuthal_integration.md) — the powder side of the
  pipeline, reading the same raw frames.
- [Scanning 3DXRD (s3DXRD)](s3dxrd.md) — grain-resolved orientation mapping
  from the same raw frames, a different technique for a related question.
- API reference: [odf](../api/texture/odf.md),
  [odf_inversion](../api/texture/odf_inversion.md),
  [texture_plotting](../api/texture/texture_plotting.md),
  [slurm_pole_figures](../api/texture/slurm_pole_figures.md).