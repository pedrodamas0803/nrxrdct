"""
LaueTools .fit file reader and comparison plotter.

The .fit file written by IndexingSpotsSet.py contains indexed spot data and
the grain orientation as a UB matrix.  These functions parse that file and
plot the result against an nrxrdct simulation.
"""

import os
import re

import h5py
import numpy as np


def read_fit_file(path):
    """
    Parse a LaueTools IndexingSpotsSet ``.fit`` file.

    Expected column header (line starting ``##``)::

        spot_index  Intensity  h  k  l  pixDev  energy(keV)  Xexp  Yexp
        2theta_exp  chi_exp  Xtheo  Ytheo  ...

    Column positions are detected from the ``##`` header line; absent columns
    fall back to the order above (Xexp=7, Yexp=8, Xtheo=11, Ytheo=12).

    The UB matrix is read from the ``#UB matrix`` block in the footer.

    Args:
        path (str): Path to the ``.fit`` file.

    Returns:
        obs_xy ((N, 2) ndarray): Segmented spot positions ``[Xexp, Yexp]``.
        theo_xy ((N, 2) ndarray): Refined theoretical positions ``[Xtheo, Ytheo]``.
        UB ((3, 3) ndarray or None): Orientation matrix from the ``#UB matrix``
            block, ready for :func:`~nrxrdct.laue.simulate_laue`.
        meta (dict): Keys when found: ``'element'``, ``'grain_index'`` (str),
            ``'n_indexed'`` (int), ``'mean_dev_px'`` (float),
            ``'euler_deg'`` (3-element array).
    """
    lines = open(path).read().splitlines()

    UB = None
    meta = {}
    col_x = col_y = col_xt = col_yt = None
    obs_rows = []
    theo_rows = []

    next_key = None
    mat_buf = []
    in_UB = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        is_comment = stripped.startswith('#')
        content = stripped.lstrip('#').strip()

        # ── "key on previous line, value here" metadata ───────────────────
        if next_key is not None:
            if next_key == 'element':
                meta['element'] = content
            elif next_key == 'grain_index':
                meta['grain_index'] = content
            elif next_key == 'euler':
                nums = re.findall(r'[-+]?\d+\.?\d*(?:[eE][+-]?\d+)?', content)
                if len(nums) >= 3:
                    meta['euler_deg'] = np.array(nums[:3], dtype=float)
            next_key = None
            continue

        if is_comment:
            # ── UB matrix block ───────────────────────────────────────────
            if re.search(r'\bUB matrix\b', content) and 'UBB0' not in content:
                in_UB = True
                mat_buf = []
                continue
            if re.search(r'\bB0 matrix\b|\bUBB0 matrix\b', content):
                in_UB = False
                mat_buf = []
                continue

            if in_UB:
                nums = re.findall(r'[-+]?\d+\.?\d*(?:[eE][+-]?\d+)?', content)
                if nums:
                    mat_buf.extend(nums)
                    if len(mat_buf) >= 9:
                        try:
                            UB = np.array(mat_buf[:9], dtype=float).reshape(3, 3)
                        except Exception:
                            pass
                        in_UB = False
                elif mat_buf:
                    in_UB = False
                continue

            # ── one-line "key: value" metadata ────────────────────────────
            if ':' in content:
                key, _, val = content.partition(':')
                key_l = key.strip().lower()
                val = val.strip()
                if 'number of indexed' in key_l or 'nb indexed' in key_l:
                    m = re.search(r'\d+', val)
                    if m:
                        meta['n_indexed'] = int(m.group())
                elif 'mean deviation' in key_l:
                    m = re.search(r'[\d.]+', val)
                    if m:
                        meta['mean_dev_px'] = float(m.group())
                continue

            # ── "key on this line, value on next" metadata ────────────────
            cl = content.lower()
            if cl == 'element':
                next_key = 'element'
                continue
            if cl == 'grainindex':
                next_key = 'grain_index'
                continue
            if cl.startswith('euler angles'):
                next_key = 'euler'
                continue

            # ── column header (##spot_index ... Xexp Yexp ... Xtheo Ytheo)
            cl = content.lower()
            if 'xexp' in cl or 'xcam' in cl:
                parts = content.split()
                pl = [p.lower() for p in parts]
                for xname, yname in (('xexp', 'yexp'), ('xcam', 'ycam')):
                    if xname in pl:
                        col_x = pl.index(xname)
                        col_y = pl.index(yname)
                        break
                for xtname, ytname in (('xtheo', 'ytheo'),):
                    if xtname in pl:
                        col_xt = pl.index(xtname)
                        col_yt = pl.index(ytname)
                        break
            continue

        # ── data row ─────────────────────────────────────────────────────
        parts = stripped.split()
        try:
            xe = float(parts[col_x])  if col_x  is not None else float(parts[7])
            ye = float(parts[col_y])  if col_y  is not None else float(parts[8])
            xt = float(parts[col_xt]) if col_xt is not None else float(parts[11])
            yt = float(parts[col_yt]) if col_yt is not None else float(parts[12])
            obs_rows.append([xe, ye])
            theo_rows.append([xt, yt])
        except (IndexError, ValueError):
            continue

    obs_xy  = np.array(obs_rows,  dtype=float) if obs_rows  else np.empty((0, 2))
    theo_xy = np.array(theo_rows, dtype=float) if theo_rows else np.empty((0, 2))
    return obs_xy, theo_xy, UB, meta


def read_raw(h5_path: str, dataset: str, index: int) -> np.ndarray:
    """
    Read a single detector frame from an HDF5 file.

    A thin convenience wrapper for loading a raw image to pass as the
    ``image`` argument to :func:`plot_fit_frame`.

    Args:
        h5_path (str): Path to the HDF5 file.
        dataset (str): Dataset path inside the file, e.g. ``'1.1/measurement/det'``.
        index (int): Frame index along the first axis.

    Returns:
        image ((Nv, Nh) ndarray): Frame as float64.
    """
    with h5py.File(h5_path, 'r') as f:
        return f[dataset][index].astype(float)


def prepare_image(image: np.ndarray, bg_sigma: float = 251.0) -> np.ndarray:
    """
    Fill detector gaps, subtract background, and clip to non-negative values.

    Pipeline:

    1. Build a valid-pixel mask (``image >= 0``; gap pixels on Eiger-type
       detectors are typically stored as ``-1`` or ``-2``).
    2. Fill gap pixels by nearest-neighbor propagation from valid pixels.
    3. Estimate a smooth background via FFT Gaussian filtering of the
       filled image (see :func:`~nrxrdct.laue.gaussian_background`).
    4. Subtract the background.
    5. Shift by ``−min`` so the lowest value is 0.

    The result is ready to pass to ``ax.imshow`` without further scaling.

    Args:
        image ((Nv, Nh) ndarray): Raw detector frame (e.g. from
            :func:`read_raw`).  May contain negative gap-pixel values.
        bg_sigma (float): Gaussian sigma (pixels) for background estimation.
            Larger values produce a smoother, more slowly varying background.
            Default ``251``.

    Returns:
        out ((Nv, Nh) ndarray float32): Background-subtracted, gap-filled image
            with minimum value 0.
    """
    from .segmentation import fill_gaps_nearest, gaussian_background

    valid = image >= 0
    filled = fill_gaps_nearest(image, valid)
    bg = gaussian_background(filled, valid, sigma=bg_sigma)
    out = (filled - bg).astype(np.float32)
    out -= out.min()
    return out


def F_from_UBB0(UBB0, crystal):
    """
    Convert a UBB0 matrix (LT frame, no 2π) to a deformation gradient F.

    UBB0 stores the reciprocal-lattice basis vectors as columns in the
    LaueTools LT frame (x ∥ beam, no 2π factor) — see the ``#UBB0 matrix``
    block in a ``.fit`` file.  This function rescales by 2π and removes the
    reference lattice so the result is a pure rotation (for an unstrained
    crystal) or F = U @ P (rotation times stretch) when strain is present.

    The returned matrix can be passed directly to
    :func:`~nrxrdct.laue.simulate_laue` as the ``U`` argument.

    Args:
        UBB0 ((3, 3) ndarray): From :func:`read_fit_file`.
        crystal: xrayutilities ``Crystal`` matching the phase in the ``.fit``.

    Returns:
        F ((3, 3) ndarray): Deformation gradient in LT frame.
    """
    B0 = np.column_stack([crystal.Q(1, 0, 0), crystal.Q(0, 1, 0), crystal.Q(0, 0, 1)])
    return (np.asarray(UBB0) * 2.0 * np.pi) @ np.linalg.inv(B0)


def plot_fit_frame(fit_path, *, image=None, bg_sigma=251.0,
                   vmin=0.0, vmax=None, figsize=(10, 8)):
    """
    Plot a LaueTools ``.fit`` frame: segmented vs refined spot positions.

    Reads ``Xexp``/``Yexp`` (segmented) and ``Xtheo``/``Ytheo`` (refined) from
    the ``.fit`` file and draws both sets on the detector image, connected
    spot-by-spot (pairs are already matched in the file).

    * **White circles** — segmented positions (Xexp, Yexp).
    * **Red diamonds** — LaueTools refined positions (Xtheo, Ytheo).
    * **Red lines** — displacement vectors between each pair.

    When *image* is supplied it is preprocessed with :func:`prepare_image`
    (gap fill → background subtraction → shift to zero) before display.

    Args:
        fit_path (str): Path to the ``.fit`` file.
        image ((Nv, Nh) ndarray or None): Raw detector frame (e.g. from
            :func:`read_raw`).  Preprocessed automatically before display.
        bg_sigma (float): Gaussian sigma for background estimation.  Default
            ``251``.
        vmin, vmax (float or None): Colour scale limits.  ``vmin`` defaults to
            ``0``; ``vmax`` defaults to the 99.5th percentile of the
            preprocessed image.
        figsize (tuple): Figure size.

    Returns:
        fig (Figure):
        ax (Axes):
    """
    import matplotlib.pyplot as plt

    obs_xy, theo_xy, _, meta = read_fit_file(fit_path)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor('k')
    ax.set_aspect('equal')
    ax.set_xlabel('x (px)')
    ax.set_ylabel('y (px)')

    parts = [os.path.basename(fit_path)]
    if 'element' in meta:
        parts.append(meta['element'])
    if 'grain_index' in meta:
        parts.append(f"grain {meta['grain_index']}")
    n = meta.get('n_indexed', len(obs_xy))
    if n:
        parts.append(f"{n} spots")
    if 'mean_dev_px' in meta:
        parts.append(f"mean dev {meta['mean_dev_px']:.3f} px")
    ax.set_title('  |  '.join(parts), fontsize=9)

    # Background image
    if image is not None:
        disp = prepare_image(image, bg_sigma=bg_sigma)
        nv_im, nh_im = disp.shape
        _vmax = vmax if vmax is not None else float(np.percentile(disp, 99.5))
        ax.imshow(
            disp,
            origin='upper', extent=[0, nh_im, nv_im, 0],
            cmap='gray', vmin=vmin, vmax=_vmax, aspect='auto',
        )
        ax.set_xlim(0, nh_im)
        ax.set_ylim(nv_im, 0)

    # Segmented positions
    if len(obs_xy):
        ax.scatter(
            obs_xy[:, 0], obs_xy[:, 1],
            s=80, facecolors='none', edgecolors='white', linewidths=1.2,
            label=f'Xexp/Yexp ({len(obs_xy)})', zorder=5,
        )

    # Refined positions + displacement lines
    if len(theo_xy):
        ax.scatter(
            theo_xy[:, 0], theo_xy[:, 1],
            s=50, marker='D', facecolors='none', edgecolors='C3', linewidths=1.0,
            label=f'Xtheo/Ytheo ({len(theo_xy)})', zorder=4,
        )
        for (xe, ye), (xt, yt) in zip(obs_xy, theo_xy):
            ax.plot([xe, xt], [ye, yt], '-', color='C3', alpha=0.6, lw=0.8, zorder=3)

    ax.legend(fontsize=8, loc='upper right')
    fig.tight_layout()
    return fig, ax
