# Spot Segmentation

This page describes how `nrxrdct` converts raw Laue detector frames into
detected spot lists, the three available segmentation algorithms, their
key parameters, and how to choose among them.

> **Prerequisite**: a calibrated `Camera` object and a stack of raw detector
> frames.  See [Diffraction and Laue Theory](laue_theory.md) for the detector
> geometry setup.

---

## 1. Overview

Each Laue detector frame is a 2-D image containing up to hundreds of
diffraction spots on top of fluorescence background, parasitic scattering, and
detector noise.  The segmentation step converts this image into a list of
sub-pixel peak positions (plus shape parameters) that can be passed to the
orientation indexer.

The pipeline for a single frame is:

```
Raw frame
    │
    ▼  gaussian_background()   — FFT-based smooth background estimate
    │
    ▼  frame − background      — spot-only residual
    │
    ▼  segmentation method     — binary mask of candidate spot pixels
    │
    ▼  clean_segmentation()    — size filter, gap exclusion, border removal
    │
    ▼  label_segmented_image() — connected-component labelling
    │
    ▼  write_h5_spotsfile()    — Gaussian fitting + HDF5 output
```

At the `GrainMap` level this pipeline is distributed across SLURM nodes
(via `GrainMap.submit_segmentation`) or run locally on multiple CPU cores
(via `run_segmentation_local`).

---

## 2. Segmentation methods

All three methods share the same signature:

```python
mask = method(image, detector_mask, **kwargs)  # → (Nv, Nh) bool array
```

They all begin with a `log1p` transform and intensity normalisation so their
responses are approximately gain-independent.

### 2.1 LoG — Laplacian-of-Gaussian

`LoG_segmentation(image, mask, sigmas, threshold_percentile)`

The **Laplacian-of-Gaussian** acts as a band-pass blob detector.  It responds
strongly to bright compact objects whose spatial scale matches the filter width:

$$
\text{response} = -\nabla^2 G_\sigma * I
$$

A local maximum of the response marks the centre of a spot with radius
$r \approx \sqrt{2}\,\sigma$.

| Parameter | Type | Description |
|---|---|---|
| `sigmas` | float or list | Gaussian σ in pixels. Match to the expected spot radius: σ ≈ FWHM / 2.35. Pass a list for multi-scale detection. |
| `threshold_percentile` | float | Percentile of the LoG response (within the valid mask) used as the binary threshold. Default `99.9`. |

**When to use**: round or weakly elongated spots of approximately known size.
Multi-scale LoG (list of sigmas) handles frames where spot size varies.

```python
from nrxrdct.laue.segmentation import LoG_segmentation

mask = LoG_segmentation(
    image,
    detector_mask,
    sigmas=[1.5, 3.0, 5.0],      # three scales in pixels
    threshold_percentile=99.8,
)
```

---

### 2.2 WTH — White Top-Hat

`WTH_segmentation(image, mask, disk_radius, threshold_percentile)`

The **white top-hat** transform removes slowly varying background while
keeping compact bright features:

$$
\text{WTH}(I) = I - \text{opening}(I, \text{disk}_r)
$$

Any structure wider than `disk_radius` pixels is suppressed; structures
smaller than the disk appear unmodified.

| Parameter | Type | Description |
|---|---|---|
| `disk_radius` | int or list | Radius of the disk structuring element in pixels. Each radius should be slightly larger than the largest expected spot. Pass a list for multi-scale detection. Default `7`. |
| `threshold_percentile` | float | Percentile of the WTH response used as the binary threshold. Default `99.9`. |

**When to use**: spots sitting on a strong, uneven background (e.g.
fluorescence halos, detector module roll-off).  Also works well for elongated
or asymmetric spots that LoG tends to under-detect.

```python
from nrxrdct.laue.segmentation import WTH_segmentation

mask = WTH_segmentation(
    image,
    detector_mask,
    disk_radius=[5, 9],           # two scales
    threshold_percentile=99.9,
)
```

---

### 2.3 HYBRID — LoG + WTH union

`hybrid_segmentation(image, mask, log_sigmas, wth_disk_radius, threshold_percentile)`

Runs LoG and WTH concurrently (all scales in one `ThreadPoolExecutor`),
thresholds each family independently at `threshold_percentile`, then
combines the two binary masks with a logical OR.

| Parameter | Type | Description |
|---|---|---|
| `log_sigmas` | float or list | Sigma(s) for the LoG family. Default `[2, 4, 8]`. |
| `wth_disk_radius` | int or list | Disk radius/radii for the WTH family. Default `[5, 7]`. |
| `threshold_percentile` | float | Applied independently to each family. Default `99.9`. |

**When to use**: frames that contain a mix of round and elongated/sharp
spots, or when either method alone leaves significant false negatives.

```python
from nrxrdct.laue.segmentation import hybrid_segmentation

mask = hybrid_segmentation(
    image,
    detector_mask,
    log_sigmas=[2, 4],
    wth_disk_radius=[5, 7],
    threshold_percentile=99.85,
)
```

---

## 3. Cleaning parameters

After segmentation, `clean_segmentation` post-processes the binary mask:

| Parameter | Default | Description |
|---|---|---|
| `min_size` | `3` | Minimum spot area in pixels. Objects smaller than this are discarded as noise. |
| `max_size` | `500` | Maximum spot area in pixels. Objects larger than this (artefacts, beam-stop shadow) are discarded. |
| `gap_exclude` | `3` | Dilation radius (pixels) around detector gaps. Spots that overlap this zone are removed to avoid bad centroid positions. |
| `gap_closing` | `3` | Closing radius applied to the valid-pixel mask before the gap dilation. Fills isolated dead pixels so spots near single bad pixels are not incorrectly discarded. Set to `0` to disable. |

---

## 4. Background subtraction

`gaussian_background(image, valid_mask, sigma=251)`

Estimates and removes the slowly varying background (fluorescence, parasitic
scattering) using an FFT-based large-sigma Gaussian:

$$
\text{background}(x,y) = \frac{(G_\sigma * W \cdot I)(x,y)}{(G_\sigma * W)(x,y)}
$$

where $W$ is the valid-pixel mask.  The normalisation by the smoothed mask
ensures correct handling of detector gaps without boundary artefacts.

| Parameter | Default | Description |
|---|---|---|
| `sigma` | `251` | Gaussian width in pixels. Should be large enough that the background varies slowly on this scale but the kernel does not wash out spot signals — typically 5–20 × the typical spot size. |

The FFT-based implementation (`scipy.fft`) makes large sigma values
computationally cheap (runtime scales as $O(N\log N)$, independent of sigma).

---

## 5. Spot fitting and HDF5 output

After cleaning, each connected component is fitted with a mixture of rotated
2-D Gaussians:

$$
I(x,y) = \sum_{k=1}^{n} A_k \exp\!\left(-\frac{1}{2} (\mathbf{r} - \mathbf{r}_{0,k})^T \boldsymbol{\Sigma}_k^{-1} (\mathbf{r} - \mathbf{r}_{0,k})\right) + C
$$

`write_h5_spotsfile` controls the fitting via two parameters:

| Parameter | Default | Description |
|---|---|---|
| `max_components` | `1` | Maximum number of Gaussian components to try. Fitting is attempted with 1, 2, … components; the first model that reaches `r_squared_min` is accepted. |
| `r_squared_min` | `0.9` | Minimum R² for a fit to be considered successful. Spots that do not reach this threshold are either stored as unfitted (`include_unfitted=True`) or skipped (`False`). |
| `d` | `10` | Half-size of the ROI cropped around each spot centroid (pixels). |
| `fit_spots` | `True` | Set to `False` to skip Gaussian fitting entirely and store only the weighted centroid from `regionprops`. Much faster; sufficient for orientation indexing when shape is not needed. |
| `include_unfitted` | `True` | Whether to write spots whose best fit did not reach `r_squared_min`. Their positions come from the heuristic initialisation, not a converged fit. |

---

## 6. Using `GrainMap.submit_segmentation`

The recommended way to run segmentation at scale is through the `GrainMap`
SLURM pipeline:

```python
from nrxrdct.laue import GrainMap

gmap = GrainMap(camera=cam, map_shape=(ny, nx))

gmap.submit_segmentation(
    h5_path="data/scan.h5",
    h5_dataset="entry/eiger/data",
    seg_dir="seg/",
    # ── detector ──────────────────────────────────────
    mask_path="mask.npy",          # bool array, True = valid pixel
    # ── background ────────────────────────────────────
    bg_sigma=251,
    # ── segmentation method ───────────────────────────
    method="LoG",                  # "LoG" | "WTH" | "HYBRID"
    method_kwargs={"sigmas": [2, 4, 8], "threshold_percentile": 99.9},
    # ── cleaning ──────────────────────────────────────
    min_size=3,
    max_size=500,
    gap_exclude=3,
    gap_closing=3,
    # ── spot fitting ──────────────────────────────────
    fit_spots=True,                # False = centroid only (faster)
    max_components=1,
    d=10,
    # ── SLURM ─────────────────────────────────────────
    n_jobs=20,
    frames_per_job=50,
    slurm_kwargs={"mem": "8G", "time": "01:00:00"},
)
```

Each SLURM job processes a subset of frames and writes one
`seg/frame_{idx:05d}.h5` file per frame.

---

## 7. Reading results

Convert the per-frame HDF5 files to peak lists for orientation fitting:

```python
from nrxrdct.laue.segmentation import convert_spotsfile2peaklist

# Single frame — returns (N, 9) array sorted by peak intensity
pl = convert_spotsfile2peaklist(
    "seg/frame_00100.h5",
    r_squared_min=0.9,       # only well-fitted spots
    include_unfitted=False,
)

# Columns: peak_X, peak_Y, peak_I, fwaxmaj, fwaxmin,
#          inclination, Xdev, Ydev, peak_bkg
xy = pl[:, :2]   # (N, 2) pixel coordinates for orientation fitting
```

`GrainMap` reads these files automatically during
`submit_orientation` / `submit_orientation_mixed`.

---

## 8. Choosing parameters

### Method selection

1. **Start with LoG** using a scalar `sigmas` roughly equal to the typical
   spot half-width (pixels).  Inspect a representative frame.
2. If many spots are missed due to uneven background, switch to **WTH**
   with `disk_radius` set slightly larger than the largest spot.
3. Use **HYBRID** when spots vary widely in morphology or when either method
   alone misses a significant fraction.

### Threshold percentile

`threshold_percentile=99.9` keeps the top 0.1% of response pixels — a good
starting point for typical Laue maps.  If too many false positives appear
(noise blobs passing `min_size`), raise towards `99.95–99.99`.  If faint
spots are missed, lower to `99.5–99.8`.

### Background sigma

Set `bg_sigma` so that the Gaussian kernel is larger than the largest spot
but small enough to follow the background envelope.  For a 2048×2048 detector
with spots of 5–20 px, `bg_sigma=251` (≈ 250 pixels ≈ 12 % of the frame) is
a reasonable default.  For smaller detectors or finer beam, scale accordingly.

### Fitting vs centroid-only

| Use case | Recommended |
|---|---|
| Orientation indexing only | `fit_spots=False` — 5–10× faster |
| Strain analysis | `fit_spots=True`, `max_components=1` |
| Overlapping spots / multi-grain pixels | `fit_spots=True`, `max_components=2` |
