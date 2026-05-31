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

        spot_index  Intensity  h  k  l  pixDev  energy(keV)  Xexp  Yexp  ...

    Column positions are detected from the ``##`` header line; if absent the
    function falls back to columns 7 and 8 (Xexp and Yexp).

    The UB matrix is read from the ``#UB matrix`` block in the footer and
    returned as-is — it is the rotation matrix ready for
    :func:`~nrxrdct.laue.simulate_laue`.

    Args:
        path (str): Path to the ``.fit`` file.

    Returns:
        obs_xy ((N, 2) ndarray): Observed spot pixel positions ``[Xexp, Yexp]``.
        UB ((3, 3) ndarray or None): Orientation matrix, passed directly to
            :func:`~nrxrdct.laue.simulate_laue` as the ``U`` argument.
        meta (dict): Keys when found: ``'element'``, ``'grain_index'`` (str),
            ``'n_indexed'`` (int), ``'mean_dev_px'`` (float),
            ``'euler_deg'`` (3-element array).
    """
    lines = open(path).read().splitlines()

    UB = None
    meta = {}
    col_x = None
    col_y = None
    rows = []

    next_key = None   # for "key on one line, value on next" metadata
    mat_buf = []      # accumulates numbers for the current matrix block
    in_UB = False     # True while inside the #UB matrix block

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
            # ── matrix blocks ─────────────────────────────────────────────
            # Match "UB matrix" but not "UBB0 matrix" or "B0 matrix"
            if re.search(r'\bUB matrix\b', content) and 'UBB0' not in content:
                in_UB = True
                mat_buf = []
                continue
            # B0 and UBB0 blocks — skip
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

            # ── column header (##spot_index ... Xexp Yexp ...) ────────────
            cl = content.lower()
            if 'xexp' in cl or 'xcam' in cl:
                parts = content.split()
                pl = [p.lower() for p in parts]
                for xname, yname in (('xexp', 'yexp'), ('xcam', 'ycam')):
                    if xname in pl:
                        col_x = pl.index(xname)
                        col_y = pl.index(yname)
                        break
            continue

        # ── data row (non-comment line) ───────────────────────────────────
        parts = stripped.split()
        try:
            if col_x is not None:
                xcam = float(parts[col_x])
                ycam = float(parts[col_y])
            else:
                # fallback: spot_index Intensity h k l pixDev energy Xexp Yexp ...
                xcam = float(parts[7])
                ycam = float(parts[8])
            rows.append([xcam, ycam])
        except (IndexError, ValueError):
            continue

    obs_xy = np.array(rows, dtype=float) if rows else np.empty((0, 2))
    return obs_xy, UB, meta


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


def plot_fit_frame(crystal, camera, fit_path, *, image=None, bg_sigma=251.0,
                   vmin=0.0, vmax=None,
                   E_min_eV=5000.0, E_max_eV=27000.0, max_match_px=10.0,
                   top_n_sim=None, figsize=(10, 8)):
    """
    Plot a LaueTools ``.fit`` frame for side-by-side comparison with nrxrdct.

    Reads the ``.fit`` file with :func:`read_fit_file`, builds the
    deformation gradient F with :func:`F_from_UBB0`, simulates spots with
    :func:`~nrxrdct.laue.simulate_laue`, and draws both sets on the detector.

    When *image* is supplied it is preprocessed with :func:`prepare_image`
    (gap fill → background subtraction → shift to zero) before display.

    * **White circles** — indexed spots from the ``.fit`` file (LaueTools).
    * **Blue diamonds** — nrxrdct simulated spots from the UBB0 orientation.
    * **Blue lines** — matched pairs within *max_match_px*.

    Args:
        crystal: xrayutilities ``Crystal`` matching the phase in the ``.fit``.
        camera (Camera): Detector geometry.
        fit_path (str): Path to the ``.fit`` file.
        image ((Nv, Nh) ndarray or None): Raw detector frame (e.g. from
            :func:`read_raw`).  Preprocessed automatically before display.
        bg_sigma (float): Gaussian sigma passed to :func:`prepare_image` for
            background estimation.  Default ``251``.
        vmin, vmax (float or None): Colour scale limits for the image.
            ``vmin`` defaults to ``0``; ``vmax`` defaults to the 99.5th
            percentile of the preprocessed image.
        E_min_eV, E_max_eV (float): Energy range for the simulation.
        max_match_px (float): Match radius (px) for drawing connection lines.
        top_n_sim (int or None): Cap on the number of simulated spots shown.
        figsize (tuple): Figure size.

    Returns:
        fig (Figure):
        ax (Axes):
    """
    import matplotlib.pyplot as plt
    from .simulation import simulate_laue
    from .fitting import _match_spots

    obs_xy, UB, meta = read_fit_file(fit_path)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor('k')
    ax.set_xlim(0, camera.Nh)
    ax.set_ylim(camera.Nv, 0)
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
        parts.append(f"{n} indexed spots")
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

    # LaueTools observed/indexed spots
    if len(obs_xy):
        ax.scatter(
            obs_xy[:, 0], obs_xy[:, 1],
            s=80, facecolors='none', edgecolors='white', linewidths=1.2,
            label=f'LaueTools ({len(obs_xy)})', zorder=5,
        )

    # nrxrdct simulation from UB
    if UB is not None:
        spots = simulate_laue(crystal, UB, camera, E_min=E_min_eV, E_max=E_max_eV)
        on_det = [s for s in spots if s.get('pix') is not None]
        if top_n_sim is not None:
            on_det = on_det[:top_n_sim]
        if on_det:
            sim_xy = np.array([s['pix'] for s in on_det])
            ax.scatter(
                sim_xy[:, 0], sim_xy[:, 1],
                s=50, marker='D', facecolors='none', edgecolors='C0',
                linewidths=1.0, label=f'nrxrdct sim ({len(sim_xy)})', zorder=4,
            )
            if len(obs_xy):
                row_ind, col_ind, dist_px = _match_spots(obs_xy, sim_xy, max_match_px)
                ok = dist_px < max_match_px
                for r, c in zip(row_ind[ok], col_ind[ok]):
                    ax.plot(
                        [obs_xy[r, 0], sim_xy[c, 0]],
                        [obs_xy[r, 1], sim_xy[c, 1]],
                        '-', color='C0', alpha=0.5, lw=0.8, zorder=3,
                    )
    else:
        ax.text(
            0.5, 0.5, 'no UB matrix found in file',
            transform=ax.transAxes, ha='center', va='center',
            color='white', fontsize=9,
        )

    ax.legend(fontsize=8, loc='upper right')
    fig.tight_layout()
    return fig, ax
