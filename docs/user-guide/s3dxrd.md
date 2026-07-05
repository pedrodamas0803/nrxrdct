# Scanning 3DXRD (s3DXRD)

This page describes `nrxrdct.xrdct.s3dxrd`, which recovers a per-voxel
single-crystal orientation map from the same spotty diffraction images used
for XRD-CT powder integration, using the scanning-3DXRD (s3DXRD)
point-by-point method.

> **Prerequisite**: install the optional `xrdct` extra to get
> [ImageD11](https://github.com/FABLE-3DXRD/ImageD11):
>
> ```bash
> uv sync --extra xrdct        # with uv
> pip install "nrxrdct[xrdct]" # with pip
> ```
>
> You'll also need an ImageD11 `.par` parameter file (detector geometry +
> unit cell) — see [Calibration](#2-calibration) below if you only have a
> pyFAI `.poni` file. See [Azimuthal Integration](azimuthal_integration.md)
> and [Quickstart](quickstart.md) for the powder side of the pipeline this
> module reuses.

---

## 1. Why this exists

Azimuthal integration collapses each detector frame into a 1-D powder
pattern, treating Bragg spots from single-crystal grains as noise to be
averaged or filtered away (`integrate_powder_parallel(..., remove_spots=True)`).
The scanning-3DXRD method reads the *same* raw frames but does the opposite:
it segments the Bragg spots and indexes them point-by-point across the
scanned rotation/translation grid, recovering a single-crystal orientation
(a UBI matrix) at every pixel of the sample cross-section — the same
cross-section that `nrxrdct.xrdct.reconstruction.reconstruct_slice`
reconstructs from the powder signal.

```
Raw 2-D frames (same master_file as azimuthal integration)
    │
    ▼  segment_slice()        — per-frame Bragg-spot centroids
    │
    ▼  build_columnfile()     — assemble peaks + diffraction geometry
    │
    ▼  index_slice()          — point-by-point indexing (per-pixel UBI)
    │
    ▼  combine_with_powder()  — powder slice + orientation map, one file
```

Each stage writes its output to disk, so any step can be re-run
independently — matching the design of the rest of the XRD-CT pipeline
(see [Typical Workflow](workflow.md)).

### Design notes

- **No `ImageD11.sinograms.dataset.DataSet`.** ImageD11's own dataset class
  assumes an ESRF Bliss `{dataroot}/{sample}/{sample}_{dset}/...` folder
  layout. This module instead reads the same single `master_file` +
  `poni`/`mask` convention used by `nrxrdct.azimuthal.integration`, and
  drives ImageD11's lower-level segmentation, columnfile, and
  point-by-point primitives directly.
- **No spline/distortion correction.** Peak positions are used as raw pixel
  coordinates, appropriate for pixel-array detectors such as Eiger.
- **Resumable, fault-tolerant segmentation.** `segment_slice` writes each
  scan to disk as soon as it's segmented (skip-if-already-done) and
  isolates per-scan failures instead of aborting the whole slice —
  matching `integrate_powder_parallel`.
- **`dty` alignment.** Every segmented peak carries its own scan's `dty`
  value rather than relying on positional order, so a scan skipped anywhere
  in the middle of the sequence can't silently misalign the translation
  axis (`assemble_sinogram` was fixed to follow the same convention — see
  [Typical Workflow](workflow.md)).

---

## 2. Calibration

Indexing needs an ImageD11 `.par` file, which is independent of the `.poni`
file used for azimuthal integration — the two packages describe detector
geometry with incompatible conventions (order of rotations vs. distance
correction, sign conventions, axis ordering).

If you don't already have an independently-calibrated `.par` file,
`poni_to_par` converts one from your existing `.poni`, following the
closed-form conversion documented at
[pyfai.readthedocs.io/en/stable/geometry_conversion.html](https://pyfai.readthedocs.io/en/stable/geometry_conversion.html):

```python
from nrxrdct.xrdct import poni_to_par

poni_to_par(
    "detector.poni",
    "detector.par",
    cell_params={
        "cell__a": 5.41143, "cell__b": 5.41143, "cell__c": 5.41143,
        "cell_alpha": 90.0, "cell_beta": 90.0, "cell_gamma": 90.0,
        "cell_lattice_[P,A,B,C,I,F,R]": 225,  # 225 = F, e.g. FCC
    },
)
```

| Parameter | Default | Description |
|---|---|---|
| `cell_params` | `None` | Unit cell dict (`cell__a/b/c`, `cell_alpha/beta/gamma`, `cell_lattice_[...]`); omit to write geometry only |
| `o11`, `o12`, `o21`, `o22` | `1, 0, 0, -1` | Detector flip matrix — pyFAI's documented default for "no additional flip" |

**Two things this conversion does *not* derive from the `.poni` file:**

1. **The detector flip matrix.** The default is correct *if* the frames
   used to calibrate the `.poni` were read with the same row/column
   orientation as the frames `segment_scan` reads — true by construction
   here, since both read `{entry}/measurement/{camera_name}` from the same
   master file. If your `.poni` came from frames read or transposed
   differently, override `o11`/`o12`/`o21`/`o22`.
2. **Sample/rotation-axis parameters with no PONI equivalent** (`chi`,
   `wedge`, `t_x`, `t_y`, `t_z`) — written as ImageD11 defaults (`0.0`).

Always verify the converted `.par` against a few known reflections before
trusting it for real indexing — a wrong flip or sign produces
plausible-looking but incorrect orientations.

---

## 3. Quick example

```python
from nrxrdct.xrdct import S3DXRDSlice, SegmentationOptions

sl = S3DXRDSlice(
    master_file="scan.h5",
    mask_file="mask.edf",
    par_file="detector.par",
    sample_name="sample1",
    symmetry="cubic",
)

sl.segment("segmented.h5", options=SegmentationOptions(cut=50, pixels_in_spot=3))
sl.build_columnfile()
sl.index("grains.txt", n_procs=8)
sl.save("s3dxrd_slice.h5")

print(sl)
# S3DXRDSlice(sample='sample1', IndexingResult(phase=None, symmetry='cubic', indexed 812/1024 pixels))
```

`sl.indexing.best_ubi` is the per-pixel UBI matrix, shape `(NI, NJ, 3, 3)`
(NaN where no grain was indexed); `sl.indexing.best_nuniq` /
`best_npks` give the match quality at each pixel.

---

## 4. Stage-by-stage

### 4.1 Segmentation — `segment_slice`

```python
from nrxrdct.xrdct import segment_slice, SegmentationOptions

results = segment_slice(
    master_file="scan.h5",
    output_file="segmented.h5",
    mask_file="mask.edf",
    options=SegmentationOptions(cut=50, howmany=100_000, pixels_in_spot=3),
    camera_name="eiger",
    translation_motor="dty",
    rotation_motor="rot",
)
```

Reads the same raw frames as `integrate_powder_parallel`, but segments
Bragg-spot peaks (via ImageD11's connected-pixel labelling and moment
centroids) instead of azimuthally averaging them away. Each scan is
written to `output_file` under `segmented/scan_XXXX` as soon as it's
segmented; a scan already present is skipped, and a scan that fails to
read or segment is skipped with a warning rather than aborting the run.

| Parameter | Default | Description |
|---|---|---|
| `cut` | `1.0` | Minimum pixel intensity kept on first pass |
| `howmany` | `100000` | Maximum pixels kept per frame |
| `pixels_in_spot` | `3` | Minimum connected-pixel count for a Bragg spot |

### 4.2 Columnfile assembly — `build_columnfile`

```python
from nrxrdct.xrdct import build_columnfile

colf = build_columnfile("segmented.h5", par_file="detector.par")
# colf.tth, colf.eta, colf.ds, colf.gx/gy/gz now available
```

Assembles all segmented peaks across the slice into a single ImageD11
columnfile and computes full diffraction geometry (2θ, η, d-spacing,
g-vectors) via the supplied `.par` file.

### 4.3 Point-by-point indexing — `index_slice`

```python
from nrxrdct.xrdct import index_slice

result = index_slice(
    colf, "detector.par", "grains.txt",
    symmetry="cubic", hkl_tol=0.05, gridstep=1, n_procs=8,
)
```

For every `(i, j)` pixel of the sample cross-section, finds the
single-crystal orientation(s) consistent with the Bragg peaks whose
sinusoidal trajectory in `(omega, dty)` space passes through it — the
scanning-3DXRD method (Henningsson & Hall).

| Parameter | Default | Description |
|---|---|---|
| `symmetry` | `"cubic"` | Crystal symmetry |
| `hkl_tol` | `0.05` | HKL matching tolerance |
| `ds_tol` | `0.005` | d-spacing tolerance for ring assignment |
| `gmax` | `5` | Max candidate grains kept per pixel |
| `gridstep` | `1` | Sampling step of the `(i, j)` grid |
| `n_procs` | all cores | Worker processes for indexing |

### 4.4 Superposing with the powder reconstruction — `combine_with_powder`

```python
from nrxrdct.xrdct import combine_with_powder

combine_with_powder(
    "s3dxrd_slice.h5",
    reconstruction_file="reconstruction.h5",
    tth_index=44,
    pixel_size_mm=0.01,
)
```

Copies the powder (ASTRA) reconstruction of one 2θ channel into the same
HDF5 file as the s3DXRD orientation map, under `powder/` and `s3dxrd/`
respectively — so the powder background and single-crystal grain map for
the same slice live side by side in one file.

---

## 5. See also

- [Azimuthal Integration](azimuthal_integration.md) — the powder side of
  the pipeline, reading the same raw frames.
- [Typical Workflow](workflow.md) — sinogram assembly and tomographic
  reconstruction of the powder signal.
- [API reference](../api/s3dxrd.md) for the full function/class docstrings.
