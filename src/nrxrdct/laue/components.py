"""
nrxrdct.laue.components
------------------------
Discover candidate grain orientations (UB matrices) from a stack of Laue
diffraction images via non-negative matrix factorisation (NMF).

Rationale
---------
A raw Laue frame is, to good approximation, a non-negative sum of each
illuminated grain's spot pattern plus a slowly varying background.  Across a
stack of frames NMF recovers a small number of non-negative "basis" images
-- each one closer to a single-grain diffraction pattern than any raw frame,
because the factorisation separates out the frame-to-frame mixing of
overlapping grains.  Each basis ("component") image is then segmented and
autoindexed independently to propose a candidate UB matrix.

Candidate matrices use the same convention and on-disk format as the rest of
the package (`G_lab = U @ G_crystal`, persisted as `UB<n>.npy`; see
:class:`~nrxrdct.laue.fitting.IndexResult` and
:meth:`~nrxrdct.laue.map.GrainMap._load_ub_matrices`), so the output of
:func:`save_ub_candidates` is immediately usable by
:meth:`~nrxrdct.laue.map.GrainMap.reload_ub_matrices`.

Workflow
--------
1. :func:`load_laue_stack` -- load a stack of frames from a TIFF directory or
   an HDF5 dataset, optionally spatially binned for speed.
2. :func:`find_ub_candidates_nmf` -- NMF-decompose the stack, segment and
   autoindex each component image, and return one :class:`ComponentCandidate`
   per component.
3. :func:`plot_component_candidates` -- static overview: each component's
   image with its segmented spots and, where indexing succeeded, the
   simulated pattern for the extracted UB.
4. For components where automatic indexing failed or is unreliable, use
   :func:`refine_component_candidate_manual` -- a thin wrapper around
   :func:`~nrxrdct.laue.interactive.interactive_orientation` seeded with the
   component's own image and spots -- to find the orientation by hand.
5. :func:`save_ub_candidates` -- write the accepted candidates out as
   `UB<n>.npy` files, ready for `GrainMap.reload_ub_matrices()`.
"""

import glob
import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np


@dataclass
class ComponentCandidate:
    """
    One NMF basis image and the candidate orientation extracted from it.

    Attributes:
        index (int): Component index (0-based) within the NMF decomposition.
        image ((ny_b, nx_b) ndarray): Non-negative component image, at the
            (possibly binned) resolution the stack was decomposed at.
        weights ((n_frames,) ndarray): NMF mixing weight of this component in
            each input frame -- large values indicate frames dominated by
            this component's pattern.
        peaklist ((N, 9) ndarray): Segmented spots in the same column layout
            as :func:`~nrxrdct.laue.segmentation.convert_spotsfile2peaklist`
            (`peak_X, peak_Y, peak_I, ...`), already rescaled to full
            detector-resolution pixel coordinates.
        obs_xy ((N, 2) ndarray): `peaklist[:, :2]`, sorted by descending
            intensity -- ready for :func:`~nrxrdct.laue.fitting.index_orientation`.
        bin_factor (int): Spatial binning factor the stack was decomposed at.
        index_result (IndexResult): Result of autoindexing `obs_xy`.
    """

    index: int
    image: np.ndarray
    weights: np.ndarray
    peaklist: np.ndarray
    obs_xy: np.ndarray
    bin_factor: int
    index_result: object

    @property
    def U(self) -> np.ndarray:
        """Best available orientation matrix (from `index_result`)."""
        return self.index_result.U

    @property
    def success(self) -> bool:
        return bool(self.index_result.success)


def bin_image_stack(stack: np.ndarray, bin_factor: int) -> np.ndarray:
    """
    Mean-pool a `(N, ny, nx)` image stack by `bin_factor` in both spatial dims.

    Trailing rows/columns that don't fit an integer number of bins are
    dropped (cropped from the bottom/right).

    Args:
        stack ((N, ny, nx) ndarray): Image stack.
        bin_factor (int): Binning factor (e.g. `2` averages 2x2 pixel blocks).
            `<= 1` returns `stack` unchanged.

    Returns:
        binned ((N, ny // bin_factor, nx // bin_factor) ndarray):
    """
    bin_factor = int(bin_factor)
    if bin_factor <= 1:
        return stack

    n, ny, nx = stack.shape
    ny_b = ny // bin_factor
    nx_b = nx // bin_factor
    cropped = stack[:, : ny_b * bin_factor, : nx_b * bin_factor]
    return cropped.reshape(n, ny_b, bin_factor, nx_b, bin_factor).mean(axis=(2, 4))


def load_laue_stack(
    path: str,
    dataset: "str | None" = None,
    *,
    bin_factor: int = 1,
    frame_indices=None,
) -> np.ndarray:
    """
    Load a stack of Laue frames from a TIFF directory or an HDF5 dataset.

    Args:
        path (str): Either a directory of `.tif` files (when `dataset` is
            `None`) or the path to an HDF5 file (when `dataset` is given).
        dataset (str or None): HDF5 dataset path inside `path`, e.g.
            `'1.1/measurement/det'`.  First axis must be the frame index.
            `None` (default) loads `.tif` files from the `path` directory
            via :func:`~nrxrdct.laue.segmentation.load_images`.
        bin_factor (int): Spatial binning factor applied after loading (see
            :func:`bin_image_stack`).  Default `1` (no binning).
        frame_indices (sequence of int or None): Subset of frames to load.
            `None` (default) loads every frame.

    Returns:
        stack ((N, ny, nx) float32 ndarray):
    """
    if dataset is not None:
        import h5py

        with h5py.File(path, "r") as f:
            ds = f[dataset]
            if frame_indices is None:
                stack = ds[()]
            else:
                stack = ds[sorted(frame_indices)]
        stack = np.asarray(stack, dtype=np.float32)
    else:
        from .segmentation import load_images

        stack = load_images(path)
        if frame_indices is not None:
            stack = stack[sorted(frame_indices)]
        stack = stack.astype(np.float32)

    return bin_image_stack(stack, bin_factor)


def _find_peaks(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    method: str = "LoG",
    method_kwargs: "dict | None" = None,
    min_size: int = 3,
    max_size: int = 500,
) -> np.ndarray:
    """Segment `image` and return a peaklist sorted by descending intensity.

    Column layout matches
    :func:`~nrxrdct.laue.segmentation.convert_spotsfile2peaklist`
    (`peak_X, peak_Y, peak_I, peak_fwaxmaj, peak_fwaxmin, peak_inclination,
    Xdev, Ydev, peak_bkg`); shape-fit columns are `0` since only weighted
    centroids are computed here (no Gaussian fit), analogous to
    `write_h5_spotsfile(..., fit_spots=False)`.
    """
    from .segmentation import (
        LoG_segmentation,
        WTH_segmentation,
        hybrid_segmentation,
        clean_segmentation,
    )

    seg_fn = {
        "LOG": LoG_segmentation,
        "WTH": WTH_segmentation,
        "HYBRID": hybrid_segmentation,
    }[method.upper()]

    raw_mask = seg_fn(image, mask, **(method_kwargs or {}))
    _final_mask, regionprops = clean_segmentation(
        raw_mask, mask, image, min_size=min_size, max_size=max_size
    )

    if not regionprops:
        return np.empty((0, 9), dtype=np.float64)

    rows = []
    for region in regionprops:
        ycen, xcen = region.centroid_weighted
        imax = float(region.image_intensity.max())
        rows.append([xcen, ycen, imax, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    peaklist = np.array(rows, dtype=np.float64)
    order = np.argsort(peaklist[:, 2])[::-1]
    return peaklist[order]


def find_ub_candidates_nmf(
    crystal,
    camera,
    stack: np.ndarray,
    n_components: int,
    *,
    bin_factor: int = 1,
    mask: "np.ndarray | None" = None,
    bg_sigma: float = 251.0,
    nmf_kwargs: "dict | None" = None,
    segmentation_method: str = "LoG",
    segmentation_kwargs: "dict | None" = None,
    min_size: int = 3,
    max_size: int = 500,
    n_obs_use: int = 20,
    index_kwargs: "dict | None" = None,
    verbose: bool = True,
) -> list:
    """
    NMF-decompose a Laue image stack and autoindex each component.

    Each frame is background-subtracted and clipped to non-negative values
    (NMF requires non-negative input), then flattened and stacked into an
    `(n_frames, n_pixels)` matrix.  `sklearn.decomposition.NMF` extracts
    `n_components` non-negative basis images; each is segmented for spots
    and passed to :func:`~nrxrdct.laue.fitting.index_orientation`.

    Args:
        crystal (Crystal): xrayutilities crystal structure used for indexing.
        camera (Camera): Detector geometry, at full (unbinned) resolution.
        stack ((N, ny, nx) ndarray): Image stack, e.g. from
            :func:`load_laue_stack`.  May already be spatially binned.
        n_components (int): Number of NMF components (candidate orientations)
            to extract.
        bin_factor (int): Binning factor `stack` was already downsampled by
            relative to `camera`'s native resolution (e.g. the `bin_factor`
            passed to :func:`load_laue_stack`).  Used to rescale segmented
            spot positions back to full-resolution pixel coordinates before
            indexing/simulation.  Default `1` (stack is at full resolution).
        mask ((ny, nx) bool ndarray or None): Valid-pixel mask at `stack`'s
            resolution.  `None` (default) treats every pixel as valid.
        bg_sigma (float): Gaussian sigma (pixels, at `stack`'s resolution) for
            per-frame background subtraction.  Scale down roughly by
            `bin_factor` from the full-resolution default (`251`) if binning.
        nmf_kwargs (dict or None): Extra keyword arguments forwarded to
            `sklearn.decomposition.NMF`.
        segmentation_method (`'LoG'`, `'WTH'`, or `'HYBRID'`): Spot-detection method applied to each
            component image; see :mod:`~nrxrdct.laue.segmentation`.
        segmentation_kwargs (dict or None): Extra keyword arguments forwarded to the segmentation
            function.
        min_size, max_size (int): Connected-component size bounds (pixels) passed to
            :func:`~nrxrdct.laue.segmentation.clean_segmentation`.
        n_obs_use (int): Number of brightest spots per component passed to
            `index_orientation`.  Default `20`.
        index_kwargs (dict or None): Extra keyword arguments forwarded to
            :func:`~nrxrdct.laue.fitting.index_orientation`.
        verbose (bool): Print a one-line summary per component.

    Returns:
        list of ComponentCandidate: One entry per NMF component, in
        component order.

    Example:
        >>> stack = load_laue_stack(tiff_dir, bin_factor=2)
        >>> candidates = find_ub_candidates_nmf(crystal, camera, stack, 6, bin_factor=2)
        >>> plot_component_candidates(candidates, crystal, camera)
        >>> failed = [c for c in candidates if not c.success]
        >>> state = refine_component_candidate_manual(failed[0], crystal, camera)
        >>> save_ub_candidates(candidates, gmap.processing_dir,
        ...                    U_override={failed[0].index: state.U} if state.accepted else {})
    """
    from sklearn.decomposition import NMF

    from .fitting import IndexResult, index_orientation
    from .segmentation import gaussian_background

    stack = np.asarray(stack, dtype=np.float32)
    n_frames, ny, nx = stack.shape

    if mask is None:
        mask = np.ones((ny, nx), dtype=bool)

    def _bg_subtract(frame):
        bg = gaussian_background(frame, mask, sigma=bg_sigma)
        return frame - bg

    with ThreadPoolExecutor() as pool:
        bg_sub = np.stack(list(pool.map(_bg_subtract, stack)), axis=0)

    bg_sub[:, ~mask] = 0.0
    np.clip(bg_sub, 0.0, None, out=bg_sub)

    X = bg_sub.reshape(n_frames, ny * nx).astype(np.float64)

    _nmf_kwargs = dict(nmf_kwargs or {})
    _nmf_kwargs.setdefault("init", "nndsvda")
    _nmf_kwargs.setdefault("max_iter", 500)
    _nmf_kwargs.setdefault("random_state", 0)

    if verbose:
        print(f"find_ub_candidates_nmf: fitting NMF ({n_frames} frames, "
              f"{ny}x{nx} px, K={n_components}) ...")

    model = NMF(n_components=n_components, **_nmf_kwargs)
    W = model.fit_transform(X)   # (n_frames, K)
    H = model.components_        # (K, ny*nx)

    _index_kwargs = dict(index_kwargs or {})
    _index_kwargs.setdefault("n_obs_use", n_obs_use)

    offset = (bin_factor - 1) / 2.0 if bin_factor and bin_factor > 1 else 0.0

    candidates = []
    for k in range(n_components):
        comp_image = H[k].reshape(ny, nx)

        peaklist = _find_peaks(
            comp_image, mask,
            method=segmentation_method, method_kwargs=segmentation_kwargs,
            min_size=min_size, max_size=max_size,
        )
        if bin_factor and bin_factor > 1:
            peaklist[:, 0] = peaklist[:, 0] * bin_factor + offset
            peaklist[:, 1] = peaklist[:, 1] * bin_factor + offset

        obs_xy = peaklist[:, :2]

        if len(obs_xy) >= 2:
            idx_result = index_orientation(crystal, camera, obs_xy, **_index_kwargs)
        else:
            idx_result = IndexResult(
                U=np.eye(3), n_matched=0, n_obs=len(obs_xy), match_rate=0.0,
                hkl_pair=((0, 0, 0), (0, 0, 0)), angle_deg=0.0,
                n_candidates=0, success=False,
            )

        candidates.append(ComponentCandidate(
            index=k, image=comp_image, weights=W[:, k],
            peaklist=peaklist, obs_xy=obs_xy, bin_factor=bin_factor,
            index_result=idx_result,
        ))

        if verbose:
            print(f"  component {k}: {len(obs_xy)} spots  ->  {idx_result}")

    return candidates


def plot_component_candidates(
    candidates: list,
    crystal,
    camera,
    *,
    E_min_eV: float = 5000.0,
    E_max_eV: float = 27000.0,
    n_cols: "int | None" = None,
    figsize_per: tuple = (4.5, 4.5),
):
    """
    Plot each NMF component image with its segmented spots and (if indexing
    succeeded) the simulated pattern for the extracted UB.

    The component image is displayed stretched to the camera's full pixel
    grid (`extent=[0, camera.Nh, camera.Nv, 0]`) regardless of
    `candidate.bin_factor`, so it lines up with `candidate.obs_xy` and with
    simulated spots, which are always in full-resolution pixel coordinates.

    Args:
        candidates (list of ComponentCandidate): Output of
            :func:`find_ub_candidates_nmf`.
        crystal, camera: As elsewhere; used to simulate the pattern for each
            successfully indexed candidate.
        E_min_eV, E_max_eV (float): Energy range forwarded to
            :func:`~nrxrdct.laue.simulation.simulate_laue`.
        n_cols (int or None): Number of subplot columns.  `None` (default)
            uses `min(4, len(candidates))`.
        figsize_per (tuple): Figure size *per subplot*, in inches.

    Returns:
        fig (Figure):
        axes (ndarray of Axes):
    """
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    from .simulation import simulate_laue

    n = len(candidates)
    n_cols = n_cols or min(4, n)
    n_rows = int(np.ceil(n / n_cols))

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_per[0] * n_cols, figsize_per[1] * n_rows),
        squeeze=False,
    )
    axes_flat = axes.ravel()

    for cand, ax in zip(candidates, axes_flat):
        img = cand.image
        ny_b, nx_b = img.shape
        extent = [0, nx_b * cand.bin_factor, ny_b * cand.bin_factor, 0]

        pos = img[img > 0]
        vmax = float(np.percentile(pos, 99)) if pos.size else 1.0
        vmin = max(vmax * 1e-3, 1e-12)
        ax.imshow(
            img, origin="upper", extent=extent, cmap="inferno",
            norm=mcolors.LogNorm(vmin=vmin, vmax=max(vmax, vmin * 10)),
            aspect="equal",
        )

        if len(cand.obs_xy):
            ax.scatter(
                cand.obs_xy[:, 0], cand.obs_xy[:, 1],
                s=40, facecolors="none", edgecolors="white", linewidths=1.0,
                label=f"obs ({len(cand.obs_xy)})", zorder=5,
            )

        if cand.index_result.success:
            spots = simulate_laue(crystal, cand.U, camera, E_min=E_min_eV, E_max=E_max_eV)
            sim_xy = np.array([s["pix"] for s in spots if s.get("pix") is not None])
            if len(sim_xy):
                ax.scatter(
                    sim_xy[:, 0], sim_xy[:, 1],
                    s=30, marker="D", facecolors="none", edgecolors="C1",
                    linewidths=1.0, label=f"sim ({len(sim_xy)})", zorder=4,
                )

        ax.set_xlim(0, camera.Nh)
        ax.set_ylim(camera.Nv, 0)
        ax.set_title(f"component {cand.index}  |  {cand.index_result}", fontsize=8)
        ax.legend(fontsize=6, loc="upper right")
        ax.tick_params(labelsize=6)

    for ax in axes_flat[n:]:
        ax.axis("off")

    fig.tight_layout()
    return fig, axes


def refine_component_candidate_manual(
    candidate: ComponentCandidate,
    crystal,
    camera,
    *,
    U0: "np.ndarray | None" = None,
    **kwargs,
):
    """
    Manually find/refine the orientation for one NMF component.

    For components where :func:`find_ub_candidates_nmf` failed to index (or
    found a poor match), this opens the same interactive widget used
    throughout the package -- :func:`~nrxrdct.laue.interactive.interactive_orientation`
    -- seeded with the component's own image and segmented spots.

    The component image may be spatially binned (`candidate.bin_factor > 1`);
    it is displayed stretched to the camera's full pixel grid by
    `interactive_orientation` itself (`extent=[0, camera.Nh, camera.Nv, 0]`),
    so it lines up with `candidate.obs_xy`, which is already in
    full-resolution pixel coordinates.

    Args:
        candidate (ComponentCandidate): One entry from `find_ub_candidates_nmf`.
        crystal, camera: As elsewhere.
        U0 ((3, 3) ndarray or None): Starting orientation for the widget.
            Defaults to `candidate.U` -- even a failed autoindex result is
            usually a reasonable starting guess -- falling back to the
            identity matrix if unavailable.
        **kwargs: Forwarded to `interactive_orientation` (e.g. `rot_range_deg`,
            `space`, `max_match_px`).

    Returns:
        OrientationState: `state.U` (final orientation), `state.accepted`
        (`True` if "Accept" was clicked).  Feed `state.U` into
        `save_ub_candidates` via its `U_override` argument to persist it.
    """
    from .interactive import interactive_orientation

    if U0 is None:
        U0 = candidate.U if candidate.U is not None else np.eye(3)

    return interactive_orientation(
        crystal, camera, candidate.obs_xy, U0=U0, image=candidate.image, **kwargs
    )


def save_ub_candidates(
    candidates: list,
    processing_dir: str,
    *,
    min_match_rate: "float | None" = None,
    U_override: "dict | None" = None,
) -> list:
    """
    Write accepted candidates to auto-numbered `UB<n>.npy` files.

    Uses the same `UB<n>.npy` naming/numbering convention as the rest of the
    package (existing files in `processing_dir` are respected; new files
    continue the numbering), so the result is immediately picked up by
    :meth:`~nrxrdct.laue.map.GrainMap.reload_ub_matrices`.

    Args:
        candidates (list of ComponentCandidate): Typically the output of
            `find_ub_candidates_nmf`, after review.
        processing_dir (str): Directory to write into -- normally the same
            one the target `GrainMap` was constructed with.
        min_match_rate (float or None): Save any candidate whose
            `index_result.match_rate >= min_match_rate`, instead of relying
            on `index_result.success`.  `None` (default) saves candidates
            with `index_result.success` `True`.  Ignored for candidates
            listed in `U_override`.
        U_override (dict or None): Map from `candidate.index` to a
            replacement `U` matrix -- e.g. the result of
            `refine_component_candidate_manual` for components where
            automatic indexing failed.  A candidate listed here is always
            saved, using this matrix.

    Returns:
        list of str: Paths of the files written, in the order saved.
    """
    os.makedirs(processing_dir, exist_ok=True)
    U_override = U_override or {}

    existing = glob.glob(os.path.join(processing_dir, "UB[0-9]*.npy"))
    max_n = -1
    for fpath in existing:
        m = re.search(r"UB(\d+)\.npy$", os.path.basename(fpath))
        if m:
            max_n = max(max_n, int(m.group(1)))

    saved = []
    for cand in candidates:
        if cand.index in U_override:
            U = U_override[cand.index]
        elif min_match_rate is not None:
            if cand.index_result.match_rate < min_match_rate:
                continue
            U = cand.U
        elif cand.index_result.success:
            U = cand.U
        else:
            continue

        max_n += 1
        fname = f"UB{max_n:02d}.npy"
        out_path = os.path.join(processing_dir, fname)
        np.save(out_path, np.asarray(U, dtype=float))
        saved.append(out_path)
        print(f"saved {fname}  <-  component {cand.index}  ({cand.index_result})")

    return saved
