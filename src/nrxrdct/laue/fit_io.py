"""
LaueTools .fit file reader and comparison plotter.

A LaueTools indexspotset .fit file contains matched spots (Xcam, Ycam pixel
coordinates) and the grain orientation matrix (matstarlab) in the header.
These two functions let you load that data and compare it with an nrxrdct
simulation on a single detector image.
"""

import os
import re

import numpy as np


def read_fit_file(path):
    """
    Parse a LaueTools indexspotset ``.fit`` file.

    Extracts the observed spot pixel positions and the orientation matrix
    (matstarlab) from the header.

    The expected column order in the data block is::

        spot_ind  H  K  L  Energy  2theta  chi  Xcam  Ycam  Intens  pixDev  [grain_ind]

    Column positions are detected automatically from any ``# ... Xcam ...``
    header line; if none is found the function falls back to the order above
    (columns 7 and 8).

    Args:
        path (str): Path to the ``.fit`` file.

    Returns:
        obs_xy ((N, 2) ndarray): Spot pixel positions ``[Xcam, Ycam]``.
            Empty ``(0, 2)`` array if the file has no data rows.
        matstarlab ((3, 3) ndarray or None): Orientation matrix in the
            LaueTools LT2/OR frame (no 2π factor).  ``None`` if no matrix
            block is found in the header.
        meta (dict): Metadata extracted from the header.  Keys present when
            found: ``'grain_index'`` (int), ``'n_indexed'`` (int),
            ``'mean_dev_px'`` (float), ``'euler_deg'`` (array of 3 floats).
    """
    lines = open(path).read().splitlines()

    matstarlab = None
    meta = {}
    col_x = None  # column index of Xcam
    col_y = None  # column index of Ycam
    rows = []

    mat_buf = []   # accumulate raw text of the 3×3 matrix
    in_mat = False  # True while inside a matstarlab block

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        is_comment = stripped.startswith('#')
        content = stripped[1:].strip() if is_comment else stripped

        if is_comment:
            # ── matstarlab / UBmatrix block ──────────────────────────────
            if re.search(r'matstarlab|UBmatrix', content, re.IGNORECASE):
                in_mat = True
                mat_buf = []
                continue

            if in_mat:
                nums_here = re.findall(r'[-+]?\d+\.?\d*(?:[eE][+-]?\d+)?', content)
                if nums_here:
                    mat_buf.extend(nums_here)
                    if len(mat_buf) >= 9:
                        try:
                            matstarlab = np.array(mat_buf[:9], dtype=float).reshape(3, 3)
                        except Exception:
                            pass
                        in_mat = False
                elif mat_buf:
                    in_mat = False
                continue

            # ── other metadata ───────────────────────────────────────────
            m = re.search(r'nb.*?indexed.*?(\d+)', content, re.IGNORECASE)
            if m:
                meta['n_indexed'] = int(m.group(1))

            m = re.search(r'mean.*?([\d.]+)', content, re.IGNORECASE)
            if m and 'mean_dev_px' not in meta:
                meta['mean_dev_px'] = float(m.group(1))

            m = re.search(r'grain.*?(\d+)', content, re.IGNORECASE)
            if m and 'grain_index' not in meta:
                meta['grain_index'] = int(m.group(1))

            m = re.search(
                r'euler.*?([\d.+-]+)\s+([\d.+-]+)\s+([\d.+-]+)', content, re.IGNORECASE
            )
            if m:
                meta['euler_deg'] = np.array(
                    [float(m.group(1)), float(m.group(2)), float(m.group(3))]
                )

            # column-name line (e.g. "# spot_ind H K L ... Xcam Ycam ...")
            if re.search(r'[Xx]cam', content):
                parts = content.split()
                pl = [p.lower() for p in parts]
                if 'xcam' in pl:
                    col_x = pl.index('xcam')
                    col_y = pl.index('ycam')
            continue

        # ── non-comment column-name line ─────────────────────────────────
        if col_x is None and re.search(r'[Xx]cam', stripped):
            parts = stripped.split()
            pl = [p.lower() for p in parts]
            if 'xcam' in pl:
                col_x = pl.index('xcam')
                col_y = pl.index('ycam')
            continue

        # ── data row ─────────────────────────────────────────────────────
        parts = stripped.split()
        try:
            if col_x is not None:
                xcam = float(parts[col_x])
                ycam = float(parts[col_y])
            else:
                # default: spot_ind H K L Energy 2theta chi Xcam Ycam ...
                xcam = float(parts[7])
                ycam = float(parts[8])
            rows.append([xcam, ycam])
        except (IndexError, ValueError):
            continue

    obs_xy = np.array(rows, dtype=float) if rows else np.empty((0, 2))
    return obs_xy, matstarlab, meta


def plot_fit_frame(crystal, camera, fit_path, *, image=None,
                   E_min_eV=5000.0, E_max_eV=27000.0, max_match_px=10.0,
                   top_n_sim=None, figsize=(10, 8)):
    """
    Plot a LaueTools ``.fit`` frame for comparison with an nrxrdct simulation.

    Reads the ``.fit`` file with :func:`read_fit_file`, converts the
    ``matstarlab`` orientation to an nrxrdct U matrix via
    :func:`~nrxrdct.laue.U_from_matstarlab`, simulates spots with
    :func:`~nrxrdct.laue.simulate_laue`, and draws both sets on the detector
    image.

    * **White circles** — spots from the ``.fit`` file (LaueTools observed /
      indexed).
    * **Blue diamonds** — nrxrdct simulated spots from the converted U matrix.
    * **Blue lines** — matched pairs within *max_match_px*.

    Args:
        crystal: xrayutilities ``Crystal`` object matching the phase indexed
            in the ``.fit`` file.
        camera (Camera): Detector geometry.
        fit_path (str): Path to the ``.fit`` file.
        image ((Nv, Nh) ndarray or None): Raw detector image.  When given it
            is shown in log-scale grey in the background.
        E_min_eV, E_max_eV (float): Energy range for the nrxrdct simulation.
        max_match_px (float): Radius (px) for drawing matched-pair lines.
        top_n_sim (int or None): Cap on the number of simulated spots shown.
        figsize (tuple): Figure size passed to ``plt.subplots``.

    Returns:
        fig (Figure):
        ax (Axes):
    """
    import matplotlib.pyplot as plt
    from .simulation import simulate_laue, U_from_matstarlab
    from .fitting import _match_spots

    obs_xy, matstarlab, meta = read_fit_file(fit_path)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor('k')
    ax.set_xlim(0, camera.Nh)
    ax.set_ylim(camera.Nv, 0)
    ax.set_aspect('equal')
    ax.set_xlabel('x (px)')
    ax.set_ylabel('y (px)')

    title_parts = [os.path.basename(fit_path)]
    if 'grain_index' in meta:
        title_parts.append(f"grain {meta['grain_index']}")
    if meta.get('n_indexed') or len(obs_xy):
        n = meta.get('n_indexed', len(obs_xy))
        title_parts.append(f"{n} indexed spots")
    if 'mean_dev_px' in meta:
        title_parts.append(f"mean dev {meta['mean_dev_px']:.2f} px")
    ax.set_title('  |  '.join(title_parts), fontsize=9)

    # Background image
    if image is not None:
        pos = image[image > 0]
        vmax = float(np.percentile(pos, 99)) if pos.size else 1.0
        nv_im, nh_im = image.shape
        ax.imshow(
            np.log1p(image / vmax * 1000),
            origin='upper', extent=[0, nh_im, nv_im, 0],
            cmap='gray', vmin=0, vmax=7, aspect='auto',
        )

    # LaueTools spots
    if len(obs_xy):
        ax.scatter(
            obs_xy[:, 0], obs_xy[:, 1],
            s=80, facecolors='none', edgecolors='white', linewidths=1.2,
            label=f'LaueTools .fit ({len(obs_xy)})', zorder=5,
        )

    # nrxrdct simulation from matstarlab
    if matstarlab is not None:
        U = U_from_matstarlab(matstarlab, crystal)
        spots = simulate_laue(crystal, U, camera, E_min=E_min_eV, E_max=E_max_eV)
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
    elif matstarlab is None:
        ax.text(
            0.5, 0.5, 'no matstarlab found in header',
            transform=ax.transAxes, ha='center', va='center',
            color='white', fontsize=9,
        )

    ax.legend(fontsize=8, loc='upper right')
    fig.tight_layout()
    return fig, ax
