"""
Kernel-density ODF fitting (MTEX-style), as an alternative to the
hard-binning WIMV solver in :mod:`nrxrdct.texture.odf_inversion`.

MTEX (Hielscher & Schaeben, 2008) represents the ODF as a sum of smooth,
unimodal kernel functions on SO(3) rather than a piecewise-constant density
over grid cells, and exploits a specific property of the de la Vallee
Poussin kernel family: projecting an SO(3) kernel down to a pole figure (the
fiber-average integral in ``nrxrdct.texture.odf_inversion``'s module
docstring) yields *another* kernel of the same family on the sphere, in
closed form. That closed form turns pole-figure fitting into a genuinely
linear inverse problem in the kernel coefficients, solvable by ordinary
non-negative least squares (NNLS) instead of WIMV's iterative multiplicative
correction on hard-binned cells.

Derivation (not taken from the literature verbatim, since getting a
half-remembered spherical-harmonic-adjacent formula wrong is an easy way to
silently corrupt every downstream fit — this was derived from scratch and
verified numerically to machine precision against a brute-force fiber
integral before being trusted here; see the derivation notes in
docs/user-guide/texture_theory.md).

**SO(3) kernel** (de la Vallee Poussin), a function of the rotation angle
``omega`` between an orientation ``g`` and the kernel's center ``g0``
(``omega = angle(g0^-1 g)``):

    psi_kappa(omega) = c_kappa * cos(omega/2)^(2*kappa)
    c_kappa = sqrt(pi) * Gamma(kappa + 2) / Gamma(kappa + 0.5)

``c_kappa`` is fixed by requiring ``integral_SO(3) psi_kappa(g) dg == 1``
(the "m.r.d." convention already used throughout this package: kappa=0 gives
psi==1 everywhere, i.e. a uniform/untextured ODF).

**Pole-figure projection**, for a kernel centered at orientation ``g0`` and
crystal direction ``h`` (so the kernel's own pole is ``y0 = g0 @ h``),
evaluated at a measured pole direction ``y``:

    P_h(y) = (kappa + 1) * ((1 + y . y0) / 2) ** kappa

Same kappa, same kernel family, exact (verified to ~1e-14 relative error
against numerical fiber-integration over a wide range of kappa and angle).
This is linear in the kernel *coefficients* once ``g0`` ranges over a fixed
grid, which is what makes the NNLS formulation below possible.
"""
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.optimize import nnls
from scipy.special import gammaln

from .odf_inversion import _alpha_beta_to_xyz, _fold_upper_hemisphere, orientation_grid


# ─────────────────────────────────────────────────────────────────────────────
# de la Vallee Poussin kernel
# ─────────────────────────────────────────────────────────────────────────────

def dlvp_norm_const(kappa: float) -> float:
    """
    SO(3) normalisation constant for the de la Vallee Poussin kernel of
    concentration *kappa*, derived from requiring
    ``integral_SO(3) psi_kappa(g) dg == 1`` under the Haar measure.

    Uses ``gammaln`` (log-gamma) rather than ``gamma`` directly since
    ``Gamma(kappa + 2)`` overflows in double precision for realistic *kappa*
    (a bandwidth of a few hundred is not unusual for these kernels).

    Args:
        kappa (float): Kernel concentration parameter (larger = narrower).

    Returns:
        float: The normalisation constant ``c_kappa``.
    """
    log_c = 0.5 * np.log(np.pi) + gammaln(kappa + 2) - gammaln(kappa + 0.5)
    return float(np.exp(log_c))


def dlvp_kernel_so3(omega_rad, kappa: float):
    """
    de la Vallee Poussin SO(3) kernel value at rotation angle *omega_rad*.

    Args:
        omega_rad (float or np.ndarray): Rotation angle(s) between an
            orientation and the kernel center, radians, in ``[0, pi]``.
        kappa (float): Kernel concentration parameter.

    Returns:
        float or np.ndarray: Kernel value(s), same shape as *omega_rad*.
    """
    c = dlvp_norm_const(kappa)
    return c * np.cos(np.asarray(omega_rad) / 2.0) ** (2.0 * kappa)


def dlvp_pole_figure_kernel(cos_angle, kappa: float):
    """
    Closed-form pole-figure projection of a de la Vallee Poussin SO(3) kernel
    — see this module's docstring for the derivation and numerical
    verification. ``cos_angle`` is the dot product between the measured pole
    direction and the kernel-center orientation's implied pole (``g0 @ h``).

    Args:
        cos_angle (float or np.ndarray): ``y . (g0 @ h)``, in ``[-1, 1]``.
        kappa (float): Same concentration parameter as the SO(3) kernel.

    Returns:
        float or np.ndarray: Pole-figure kernel value(s), same shape as
            *cos_angle*.
    """
    return (kappa + 1.0) * np.clip((1.0 + np.asarray(cos_angle)) / 2.0, 0.0, None) ** kappa


def kappa_from_halfwidth_deg(halfwidth_deg: float) -> float:
    """
    Convert an SO(3) kernel half-width (the rotation angle, in degrees, at
    which the kernel drops to half its peak value) to the concentration
    parameter *kappa*.

    Solved in closed form from ``cos(halfwidth/2)^(2*kappa) == 0.5``:
    ``kappa = -ln(2) / (2 * ln(cos(halfwidth_rad / 2)))``.

    Args:
        halfwidth_deg (float): Desired SO(3) kernel half-width, degrees.

    Returns:
        float: The corresponding *kappa*.
    """
    half_rad = np.deg2rad(halfwidth_deg)
    return float(-np.log(2.0) / (2.0 * np.log(np.cos(half_rad / 2.0))))


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def compute_odf_kernel(
    pole_figures: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    crystal_directions: Dict[str, np.ndarray],
    step_deg: float = 10.0,
    kappa: Optional[float] = None,
    halfwidth_deg: float = 15.0,
    fold_hemisphere: bool = False,
) -> dict:
    """
    Fit a smooth, kernel-density ODF to pole-figure data via non-negative
    least squares (MTEX-style), as an alternative to
    :func:`nrxrdct.texture.odf_inversion.compute_odf`'s WIMV solver.

    The ODF is represented as ``f(g) = sum_k c_k * psi_kappa(g, g_k)`` over a
    fixed orientation grid (:func:`~nrxrdct.texture.odf_inversion.orientation_grid`);
    because the pole-figure projection of each kernel has the closed form
    :func:`dlvp_pole_figure_kernel`, the forward model is linear in the
    coefficients ``c_k`` and is solved directly by NNLS — no iteration, no
    convergence trace, and (like WIMV) non-negative by construction. Same
    ghost-ambiguity caveat as WIMV applies: see
    ``docs/user-guide/texture_theory.md`` §4 before enabling *fold_hemisphere*.

    Args:
        pole_figures (dict): Same format as
            :func:`~nrxrdct.texture.odf_inversion.compute_odf` — maps hkl
            label to ``(alpha_deg, beta_deg, intensity)``.
        crystal_directions (dict): Same format as ``compute_odf`` — maps hkl
            label to unit vector(s), shape ``(3,)`` or ``(M, 3)`` for
            symmetry-equivalent directions. Unlike ``compute_odf``'s hard
            nearest-cell binning (which picks a single closest symmetry
            variant), every variant contributes additively here — matching
            how a real ``{hkl}`` pole figure aggregates intensity from all
            symmetry-equivalent reflections simultaneously.
        step_deg (float, optional): Orientation-grid spacing, forwarded to
            ``orientation_grid`` (default 10.0). Now purely the *density* of
            kernel centers — angular resolution is set independently by
            *kappa*/*halfwidth_deg*, unlike WIMV where grid spacing was the
            only resolution knob.
        kappa (float, optional): SO(3) kernel concentration. If ``None``
            (default), derived from *halfwidth_deg*.
        halfwidth_deg (float, optional): SO(3) kernel half-width in degrees,
            used to derive *kappa* when not given directly (default 15.0).
            Should not be much smaller than *step_deg*, or the grid will
            under-sample the kernels.
        fold_hemisphere (bool, optional): See
            :func:`~nrxrdct.texture.odf_inversion.compute_odf`'s docstring —
            same risk, same default (default ``False``).

    Returns:
        dict: With keys ``'euler_deg'``, ``'R'``, ``'cell_weight'`` (as
            ``orientation_grid``), ``'f'`` (fitted kernel coefficients per
            grid cell, non-negative, normalised so
            ``sum(f * cell_weight) / sum(cell_weight) == 1``), ``'kappa'``
            (the concentration actually used), and ``'residual'`` (the NNLS
            residual norm).
    """
    if set(pole_figures) != set(crystal_directions):
        raise ValueError(
            f"pole_figures and crystal_directions must cover the same hkl labels; "
            f"got {sorted(pole_figures)} vs {sorted(crystal_directions)}"
        )

    if kappa is None:
        kappa = kappa_from_halfwidth_deg(halfwidth_deg)

    euler_deg, R, cell_weight = orientation_grid(step_deg)
    N = euler_deg.shape[0]

    A_blocks = []
    b_blocks = []
    for hkl, (alpha_deg, beta_deg, intensity) in pole_figures.items():
        alpha_f, beta_f = np.asarray(alpha_deg), np.asarray(beta_deg)
        if fold_hemisphere:
            alpha_f, beta_f = _fold_upper_hemisphere(alpha_f, beta_f)
        data_xyz = _alpha_beta_to_xyz(alpha_f, beta_f)  # (n_data, 3)

        h = np.atleast_2d(crystal_directions[hkl]).astype(np.float64)
        h = h / np.linalg.norm(h, axis=1, keepdims=True)
        grid_xyz_variants = np.einsum("nij,mj->mni", R, h)  # (M, N, 3)
        if fold_hemisphere:
            alpha_g = np.rad2deg(np.arccos(np.clip(grid_xyz_variants[..., 2], -1.0, 1.0)))
            beta_g = np.rad2deg(np.arctan2(grid_xyz_variants[..., 1], grid_xyz_variants[..., 0])) % 360.0
            alpha_gf, beta_gf = _fold_upper_hemisphere(alpha_g, beta_g)
            grid_xyz_variants = _alpha_beta_to_xyz(alpha_gf, beta_gf)

        # cos_angle[j, m, k] = data_xyz[j] . grid_xyz_variants[m, k]; sum over
        # symmetry variants m (see docstring: additive, not nearest-match).
        cos_angle = np.einsum("jc,mkc->jmk", data_xyz, grid_xyz_variants)
        A_hkl = dlvp_pole_figure_kernel(cos_angle, kappa).sum(axis=1)  # (n_data, N)

        A_blocks.append(A_hkl)
        b_blocks.append(np.asarray(intensity, dtype=np.float64))

    A = np.concatenate(A_blocks, axis=0)
    b = np.concatenate(b_blocks, axis=0)

    f, residual = nnls(A, b)

    norm = np.sum(f * cell_weight) / np.sum(cell_weight)
    if norm > 0:
        f = f / norm

    return {
        "euler_deg": euler_deg,
        "R": R,
        "cell_weight": cell_weight,
        "f": f,
        "kappa": kappa,
        "residual": float(residual),
    }


def recalculate_pole_figure_kernel(
    odf_result: dict,
    crystal_direction: np.ndarray,
    alpha_deg: np.ndarray,
    beta_deg: np.ndarray,
    fold_hemisphere: bool = False,
) -> np.ndarray:
    """
    Forward-project a :func:`compute_odf_kernel` fit back onto pole-figure
    coordinates, for comparison against the original measured pole figure.

    Args:
        odf_result (dict): Output of :func:`compute_odf_kernel`.
        crystal_direction (np.ndarray): Unit vector(s), shape ``(3,)`` or
            ``(M, 3)`` (same convention as :func:`compute_odf_kernel`).
        alpha_deg (np.ndarray): Polar angles (degrees) at which to evaluate.
        beta_deg (np.ndarray): Azimuthal angles (degrees) at which to evaluate.
        fold_hemisphere (bool, optional): Must match the value used to
            produce *odf_result* (default ``False``).

    Returns:
        np.ndarray: Recalculated pole-figure intensity at each query point,
            same shape as *alpha_deg*.
    """
    R = odf_result["R"]
    f = odf_result["f"]
    kappa = odf_result["kappa"]

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
    cos_angle = np.einsum("jc,mkc->jmk", query_xyz, grid_xyz_variants)
    A = dlvp_pole_figure_kernel(cos_angle, kappa).sum(axis=1)
    return A @ f