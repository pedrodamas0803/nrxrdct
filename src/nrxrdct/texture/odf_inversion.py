"""
ODF inversion from pole-figure data (WIMV method).

Inverts one or more pole figures (from :mod:`nrxrdct.texture.odf`) into a
discretised orientation distribution function (ODF) on a grid of Bunge Euler
angles, using the WIMV algorithm (Matthies & Vinel, 1982): an iterative,
multiplicative correction scheme that is the standard choice in texture
analysis whenever pole-figure coverage is incomplete, because it degrades
gracefully over uncovered regions and — unlike a literal generalised
spherical-harmonic fit — guarantees a non-negative ODF by construction.

This is the natural next step after single-rotation-axis pole-figure
extraction: a fixed vertical rotation axis with no tilt only ever sweeps a
*curve* on the pole sphere per hkl (see the coverage caveat in
:mod:`nrxrdct.texture.odf`), so pole-figure coverage here is inherently
incomplete — the reason WIMV rather than the harmonic method is used.

Orientation convention
-----------------------
Orientations are represented as rotation matrices mapping a crystal-frame
vector to the sample frame, ``v_sample = R @ v_crystal``, parametrised by
Bunge-style Euler angles (phi1, Phi, phi2) via

    R(phi1, Phi, phi2) = Rz(phi1) @ Rx(Phi) @ Rz(phi2)

This is one of several conventions in use across texture software (Bunge,
Roe, Kocks differ, and even "Bunge" implementations disagree on axis order
and active/passive sense) — verify against your reference before comparing
Euler angles quantitatively with another package.

Crystal directions passed to :func:`compute_odf` must already be unit
vectors in an orthonormal crystal Cartesian frame (this module does not
parse lattice parameters); for a cubic crystal, ``[h, k, l]`` normalised is
sufficient, otherwise apply your crystal's B-matrix first.
"""
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import h5py
import numpy as np
from scipy.spatial import cKDTree
from scipy import sparse

from .odf import assemble_pole_figure_sinogram, sinogram_to_pole_figure


# ─────────────────────────────────────────────────────────────────────────────
# Orientation grid
# ─────────────────────────────────────────────────────────────────────────────

def orientation_grid(step_deg: float = 10.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a regular grid of orientations in Bunge Euler space.

    Args:
        step_deg (float, optional): Grid spacing in degrees along each Euler
            angle (default 10.0). Cell count scales as ``(360/step)^2 * (180/step)``,
            so halving *step_deg* multiplies the grid size (and run time) by ~8.

    Returns:
        euler_deg (np.ndarray): Grid orientations as ``(phi1, Phi, phi2)`` in
            degrees, shape ``(N, 3)``.
        R (np.ndarray): Rotation matrices for each grid orientation, mapping
            crystal-frame to sample-frame vectors, shape ``(N, 3, 3)``.
        cell_weight (np.ndarray): Relative SO(3) (Haar-measure) volume of each
            cell, proportional to ``sin(Phi)``, shape ``(N,)``. Use these as
            weights when averaging or normalising the ODF over the grid.
    """
    phi1 = np.arange(0.0, 360.0, step_deg) + step_deg / 2
    Phi = np.arange(0.0, 180.0, step_deg) + step_deg / 2
    phi2 = np.arange(0.0, 360.0, step_deg) + step_deg / 2

    P1, PH, P2 = np.meshgrid(phi1, Phi, phi2, indexing="ij")
    euler_deg = np.stack([P1.ravel(), PH.ravel(), P2.ravel()], axis=1)
    cell_weight = np.sin(np.deg2rad(euler_deg[:, 1]))

    R = _euler_to_matrix(euler_deg)
    return euler_deg, R, cell_weight


def _euler_to_matrix(euler_deg: np.ndarray) -> np.ndarray:
    """Vectorised Bunge Euler angles (N,3) [deg] -> rotation matrices (N,3,3)."""
    phi1, Phi, phi2 = (np.deg2rad(euler_deg[:, i]) for i in range(3))
    N = euler_deg.shape[0]

    def _rz(t):
        c, s = np.cos(t), np.sin(t)
        Rz = np.zeros((N, 3, 3))
        Rz[:, 0, 0] = c
        Rz[:, 0, 1] = -s
        Rz[:, 1, 0] = s
        Rz[:, 1, 1] = c
        Rz[:, 2, 2] = 1.0
        return Rz

    def _rx(t):
        c, s = np.cos(t), np.sin(t)
        Rx = np.zeros((N, 3, 3))
        Rx[:, 0, 0] = 1.0
        Rx[:, 1, 1] = c
        Rx[:, 1, 2] = -s
        Rx[:, 2, 1] = s
        Rx[:, 2, 2] = c
        return Rx

    return _rz(phi1) @ _rx(Phi) @ _rz(phi2)


def misorientation_angle_deg(R1: np.ndarray, R2: np.ndarray) -> float:
    """
    Angle (degrees) of the relative rotation between two orientation matrices.

    Args:
        R1 (np.ndarray): Rotation matrix, shape ``(3, 3)``.
        R2 (np.ndarray): Rotation matrix, shape ``(3, 3)``.

    Returns:
        float: Misorientation angle in degrees, in ``[0, 180]``.
    """
    dR = R1.T @ R2
    cos_angle = np.clip((np.trace(dR) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.rad2deg(np.arccos(cos_angle)))


# ─────────────────────────────────────────────────────────────────────────────
# Pole-direction geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def _alpha_beta_to_xyz(alpha_deg: np.ndarray, beta_deg: np.ndarray) -> np.ndarray:
    a = np.deg2rad(alpha_deg)
    b = np.deg2rad(beta_deg)
    x = np.sin(a) * np.cos(b)
    y = np.sin(a) * np.sin(b)
    z = np.cos(a)
    return np.stack([x, y, z], axis=-1)


def _fold_upper_hemisphere(alpha_deg: np.ndarray, beta_deg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Map (alpha, beta) with alpha>90 to their antipodal upper-hemisphere equivalent."""
    lower = alpha_deg > 90.0
    alpha_f = np.where(lower, 180.0 - alpha_deg, alpha_deg)
    beta_f = np.where(lower, (beta_deg + 180.0) % 360.0, beta_deg)
    return alpha_f, beta_f


# ─────────────────────────────────────────────────────────────────────────────
# Kernel (correspondence) matrix
# ─────────────────────────────────────────────────────────────────────────────

def _build_kernel_matrix(
    data_xyz: np.ndarray,
    grid_xyz_variants: np.ndarray,
    smoothing_deg: float,
    cutoff_sigma: float = 2.0,
    workers: int = -1,
) -> sparse.csr_matrix:
    """
    Build the (n_data, n_grid) row-normalised Gaussian correspondence matrix.

    grid_xyz_variants has shape (M, N, 3) — M symmetry-equivalent crystal
    directions projected through the same N grid orientations. All M variants
    are queried against a single KDTree (built over the M*N points at once)
    and folded back to column index ``n`` via ``flat_index % N``; duplicate
    (row, col) pairs from different variants landing near the same grid
    orientation are summed automatically by the sparse constructor, so the
    returned matrix has shape (n_data, N) exactly as if M were queried
    separately and added together.

    workers is forwarded to ``cKDTree.query_ball_point`` (-1 uses all cores);
    pass a smaller value when several hkls are being processed concurrently
    to avoid oversubscribing CPU cores.
    """
    n_data = data_xyz.shape[0]
    M, N, _ = grid_xyz_variants.shape
    sigma = np.deg2rad(smoothing_deg)
    cutoff_chord = 2 * np.sin(np.deg2rad(cutoff_sigma * smoothing_deg) / 2)

    grid_flat = grid_xyz_variants.reshape(M * N, 3)
    tree = cKDTree(grid_flat)
    neighbor_lists = tree.query_ball_point(data_xyz, r=cutoff_chord, workers=workers)

    lengths = np.fromiter((len(nb) for nb in neighbor_lists), dtype=np.int64, count=n_data)
    if lengths.sum() == 0:
        acc = sparse.csr_matrix((n_data, N))
    else:
        rows = np.repeat(np.arange(n_data), lengths)
        flat_idx = np.concatenate([np.asarray(nb, dtype=np.int64) for nb in neighbor_lists if len(nb)])
        cols = flat_idx % N

        dots = np.clip(np.einsum("ij,ij->i", grid_flat[flat_idx], data_xyz[rows]), -1.0, 1.0)
        weights = np.exp(-(np.arccos(dots) ** 2) / (2 * sigma ** 2))

        acc = sparse.csr_matrix((weights, (rows, cols)), shape=(n_data, N))

    row_sums = np.asarray(acc.sum(axis=1)).ravel()
    empty_rows = np.where(row_sums == 0)[0]
    if len(empty_rows) > 0:
        print(
            f"  ⚠  {len(empty_rows)}/{n_data} measured points had no grid cell within "
            f"{cutoff_sigma}*smoothing_deg — increase smoothing_deg or step_deg."
        )
        row_sums[empty_rows] = 1.0  # avoid divide-by-zero; those rows stay all-zero

    inv_row_sums = sparse.diags(1.0 / row_sums)
    return inv_row_sums @ acc


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def compute_odf(
    pole_figures: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    crystal_directions: Dict[str, np.ndarray],
    step_deg: float = 10.0,
    smoothing_deg: float = 7.5,
    n_iter: int = 10,
    fold_hemisphere: bool = False,
) -> dict:
    """
    Invert pole-figure data into a discretised ODF via WIMV.

    Args:
        pole_figures (dict): Maps hkl label to ``(alpha_deg, beta_deg, intensity)``,
            each a 1-D array of equal length — measured pole-figure points, e.g.
            built from :func:`nrxrdct.texture.odf.pole_figure_coordinates` applied
            to a merged pole-figure sinogram. Intensities should be non-negative
            (background-subtracted, as from :func:`extract_ring_intensity`, may
            need clipping at zero first).
        crystal_directions (dict): Maps the same hkl labels to unit vector(s) in
            an orthonormal crystal Cartesian frame — shape ``(3,)`` for a single
            direction, or ``(M, 3)`` to include symmetry-equivalent directions of
            the same ``{hkl}`` family (the pole figure is then treated as
            covering all M equivalents, standard practice for a ``{hkl}``
            pole figure).
        step_deg (float, optional): Euler-angle grid spacing passed to
            :func:`orientation_grid` (default 10.0).
        smoothing_deg (float, optional): Gaussian kernel width (degrees) relating
            measured pole directions to grid orientations — effectively the
            angular resolution of the recovered ODF (default 7.5).
        n_iter (int, optional): Number of WIMV iterations (default 10).
        fold_hemisphere (bool, optional): If ``True``, map every pole direction
            with ``alpha > 90`` to its antipodal upper-hemisphere equivalent
            before fitting (Friedel's law), matching how
            :func:`nrxrdct.texture.texture_plotting.plot_pole_figure` displays
            data. **Default is False** — folding independently per hkl is
            mathematically equivalent to conflating orientation ``R`` with its
            point-inversion ``-R`` across every hkl simultaneously, and WIMV's
            positivity constraint only sometimes breaks that tie: confirmed by
            testing to occasionally converge to the ~180°-misoriented "ghost"
            solution instead of the true orientation, for both single- and
            multi-hkl fits. Only enable this if you specifically need folded
            display-style output and have separately verified it doesn't
            corrupt your fit.

    Returns:
        dict: With keys ``'euler_deg'`` (grid, shape ``(N,3)``), ``'R'`` (grid
            rotation matrices, shape ``(N,3,3)``), ``'cell_weight'`` (shape
            ``(N,)``), ``'f'`` (fitted ODF value per grid cell, shape ``(N,)``,
            normalised so ``sum(f * cell_weight) / sum(cell_weight) == 1``),
            and ``'rp_history'`` (list of per-iteration relative pole-figure
            error, for convergence checking).
    """
    if set(pole_figures) != set(crystal_directions):
        raise ValueError(
            f"pole_figures and crystal_directions must cover the same hkl labels; "
            f"got {sorted(pole_figures)} vs {sorted(crystal_directions)}"
        )

    euler_deg, R, cell_weight = orientation_grid(step_deg)
    N = euler_deg.shape[0]

    data_xyz_by_hkl = {}
    grid_variants_by_hkl = {}
    measured = {}
    for hkl, (alpha_deg, beta_deg, intensity) in pole_figures.items():
        alpha_f, beta_f = np.asarray(alpha_deg), np.asarray(beta_deg)
        if fold_hemisphere:
            alpha_f, beta_f = _fold_upper_hemisphere(alpha_f, beta_f)
        data_xyz = _alpha_beta_to_xyz(alpha_f, beta_f)

        h = np.atleast_2d(crystal_directions[hkl]).astype(np.float64)
        h = h / np.linalg.norm(h, axis=1, keepdims=True)
        grid_xyz_variants = np.einsum("nij,mj->mni", R, h)  # (M, N, 3)
        if fold_hemisphere:
            alpha_g = np.rad2deg(np.arccos(np.clip(grid_xyz_variants[..., 2], -1.0, 1.0)))
            beta_g = np.rad2deg(np.arctan2(grid_xyz_variants[..., 1], grid_xyz_variants[..., 0])) % 360.0
            alpha_gf, beta_gf = _fold_upper_hemisphere(alpha_g, beta_g)
            grid_xyz_variants = _alpha_beta_to_xyz(alpha_gf, beta_gf)

        data_xyz_by_hkl[hkl] = data_xyz
        grid_variants_by_hkl[hkl] = grid_xyz_variants
        measured[hkl] = np.asarray(intensity, dtype=np.float64)

    # Kernel building (KDTree query per hkl) is the expensive, independent-per-hkl
    # step — run it across processes when there's more than one hkl and more than
    # one CPU available, splitting cores between concurrent hkls so each hkl's
    # internal cKDTree parallelism (`workers=`) doesn't oversubscribe the machine.
    hkl_list = list(pole_figures)
    cpu_count = os.cpu_count() or 1
    n_jobs = max(1, min(len(hkl_list), cpu_count))

    if n_jobs == 1:
        kernels = {
            hkl: _build_kernel_matrix(data_xyz_by_hkl[hkl], grid_variants_by_hkl[hkl], smoothing_deg)
            for hkl in hkl_list
        }
    else:
        inner_workers = max(1, cpu_count // n_jobs)
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            futures = {
                hkl: pool.submit(
                    _build_kernel_matrix,
                    data_xyz_by_hkl[hkl], grid_variants_by_hkl[hkl], smoothing_deg,
                    2.0, inner_workers,
                )
                for hkl in hkl_list
            }
            kernels = {hkl: fut.result() for hkl, fut in futures.items()}

    f = np.ones(N, dtype=np.float64)
    rp_history = []

    for it in range(n_iter):
        log_ratio_num = np.zeros(N)
        weight_den = np.zeros(N)
        sse, ssm = 0.0, 0.0

        for hkl, K in kernels.items():
            predicted = K @ f
            predicted_safe = np.where(predicted > 0, predicted, np.nan)
            ratio = measured[hkl] / predicted_safe
            valid = np.isfinite(ratio) & (ratio > 0)

            log_ratio_num += K[valid].T @ np.log(ratio[valid])
            weight_den += np.asarray(K[valid].sum(axis=0)).ravel()

            sse += float(np.nansum((predicted - measured[hkl]) ** 2))
            ssm += float(np.sum(measured[hkl] ** 2))

        correction = np.ones(N)
        has_support = weight_den > 0
        correction[has_support] = np.exp(log_ratio_num[has_support] / weight_den[has_support])
        f = f * correction

        rp = float(np.sqrt(sse / ssm)) if ssm > 0 else float("nan")
        rp_history.append(rp)
        print(f"  WIMV iter {it+1}/{n_iter}: Rp = {rp:.4f}")

    norm = np.sum(f * cell_weight) / np.sum(cell_weight)
    if norm > 0:
        f = f / norm

    return {
        "euler_deg": euler_deg,
        "R": R,
        "cell_weight": cell_weight,
        "f": f,
        "rp_history": rp_history,
    }


def recalculate_pole_figure(
    odf_result: dict,
    crystal_direction: np.ndarray,
    alpha_deg: np.ndarray,
    beta_deg: np.ndarray,
    smoothing_deg: float = 7.5,
    fold_hemisphere: bool = False,
) -> np.ndarray:
    """
    Forward-project the fitted ODF back onto pole-figure coordinates, for
    comparison against the original measured pole figure (quality check).

    Args:
        odf_result (dict): Output of :func:`compute_odf`.
        crystal_direction (np.ndarray): Unit vector(s) in the crystal frame,
            shape ``(3,)`` or ``(M, 3)`` (same convention as in :func:`compute_odf`).
        alpha_deg (np.ndarray): Polar angles (degrees) at which to evaluate.
        beta_deg (np.ndarray): Azimuthal angles (degrees) at which to evaluate.
        smoothing_deg (float, optional): Kernel width, should match the value
            used in :func:`compute_odf` (default 7.5).
        fold_hemisphere (bool, optional): Must match the value passed to
            :func:`compute_odf` when *odf_result* was produced — see the
            warning in that function's docstring (default ``False``).

    Returns:
        np.ndarray: Recalculated pole-figure intensity at each query point,
            same shape as *alpha_deg*.
    """
    R = odf_result["R"]
    f = odf_result["f"]

    h = np.atleast_2d(crystal_direction).astype(np.float64)
    h = h / np.linalg.norm(h, axis=1, keepdims=True)
    grid_xyz_variants = np.einsum("nij,mj->mni", R, h)
    alpha_f, beta_f = np.asarray(alpha_deg), np.asarray(beta_deg)
    if fold_hemisphere:
        alpha_g = np.rad2deg(np.arccos(np.clip(grid_xyz_variants[..., 2], -1.0, 1.0)))
        beta_g = np.rad2deg(np.arctan2(grid_xyz_variants[..., 1], grid_xyz_variants[..., 0])) % 360.0
        alpha_gf, beta_gf = _fold_upper_hemisphere(alpha_g, beta_g)
        grid_xyz_variants = _alpha_beta_to_xyz(alpha_gf, beta_gf)
        alpha_f, beta_f = _fold_upper_hemisphere(alpha_f, beta_f)

    query_xyz = _alpha_beta_to_xyz(alpha_f, beta_f)

    K = _build_kernel_matrix(query_xyz, grid_xyz_variants, smoothing_deg)
    return K @ f


def load_pole_figures(
    pole_figure_file: Path,
    hkl_labels: Sequence[str],
    n_rot: int,
    line_index: Optional[int] = None,
    translation_motor: str = "dty",
    rotation_motor: str = "rot",
    chi_offset_deg: float = 0.0,
    chi_sign: float = 1.0,
    clip_negative: bool = True,
) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Build the ``pole_figures`` dict :func:`compute_odf` expects, directly from
    a merged pole-figure HDF5 file.

    For each hkl, assembles the pole-figure sinogram
    (:func:`~nrxrdct.texture.odf.assemble_pole_figure_sinogram`), reads its
    stored 2theta window, and converts it to flat pole-figure coordinates
    (:func:`~nrxrdct.texture.odf.sinogram_to_pole_figure`) — the glue between
    a file written by :func:`~nrxrdct.texture.odf.assemble_pole_figure_data` /
    :mod:`nrxrdct.texture.slurm_pole_figures` and :func:`compute_odf`.

    Args:
        pole_figure_file (Path): HDF5 file written by
            :func:`~nrxrdct.texture.odf.assemble_pole_figure_data` or
            :mod:`nrxrdct.texture.slurm_pole_figures`'s ``merge``.
        hkl_labels (sequence of str): Ring identifiers to load, e.g. ``["111", "200"]``.
        n_rot (int): Number of rotation steps, forwarded to
            :func:`~nrxrdct.texture.odf.assemble_pole_figure_sinogram`.
        line_index (int, optional): Translation line to use; ``None`` (default)
            averages over all lines into one bulk pole figure for the whole
            scanned cross-section. Pass an explicit index for a per-line bulk
            pole figure instead.
        translation_motor (str, optional): Passed through to
            :func:`~nrxrdct.texture.odf.assemble_pole_figure_sinogram` (default ``"dty"``).
        rotation_motor (str, optional): Passed through to
            :func:`~nrxrdct.texture.odf.assemble_pole_figure_sinogram` (default ``"rot"``).
        chi_offset_deg (float, optional): Forwarded to
            :func:`~nrxrdct.texture.odf.pole_figure_coordinates` — verify against
            your beamline before trusting results (default 0.0).
        chi_sign (float, optional): Forwarded likewise (default 1.0).
        clip_negative (bool, optional): Clip negative intensities to zero
            (default ``True``). Background-subtracted ring intensities can go
            slightly negative from noise; :func:`compute_odf` assumes
            non-negative measurements.

    Returns:
        dict: ``{hkl_label: (alpha_deg, beta_deg, intensity)}``, ready to pass
            directly as :func:`compute_odf`'s ``pole_figures`` argument.
    """
    pole_figures = {}
    for hkl in hkl_labels:
        sino, azimuthal, _, rot_deg = assemble_pole_figure_sinogram(
            pole_figure_file, hkl, n_rot, translation_motor, rotation_motor
        )
        with h5py.File(pole_figure_file, "r") as hin:
            tth_deg = float(np.mean(hin[f"pole_figures/{hkl}/tth_range"][:]))

        alpha, beta, intensity = sinogram_to_pole_figure(
            sino, azimuthal, rot_deg, tth_deg,
            line_index=line_index, chi_offset_deg=chi_offset_deg, chi_sign=chi_sign,
        )
        if clip_negative:
            intensity = np.clip(intensity, 0.0, None)
        pole_figures[hkl] = (alpha, beta, intensity)

    return pole_figures