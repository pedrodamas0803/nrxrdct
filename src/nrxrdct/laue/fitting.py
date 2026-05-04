"""
Laue orientation-matrix fitting
================================
Refines orientation matrices by minimising pixel-space residuals between
simulated and observed Laue spot positions.

Three simulation back-ends are supported:

    fit_orientation        — single crystal  (:func:`simulate_laue`)
    fit_orientation_stack  — layered crystal (:func:`simulate_laue_stack`)
    fit_orientation_mixed  — multi-phase     (:func:`simulate_mixed_phases`)

Parametrisation
---------------
The free parameters are always rotation vectors δω (radians).  At each
optimizer iteration the current orientation of phase / layer *i* is:

    U_i = Rotation.from_rotvec(δω_i) @ U0_i

where U0_i is the starting estimate.  Keeping δω small (linearised update)
avoids gimbal lock and gives a well-conditioned Jacobian.

Stack and mixed-phase fitting
------------------------------
Both functions fit a *single shared* rotation by default (all
layers/phases rotate together, preserving their relative orientations).
Set ``shared=False`` in the mixed-phase case to optimise one independent
rotation vector per phase (3 × N_phases free parameters).

Spot matching
-------------
At each function evaluation the assignment between simulated and observed
spots is unknown.  It is solved with the Hungarian algorithm
(``scipy.optimize.linear_sum_assignment``) on the pixel-distance cost
matrix, capped at ``max_match_px``.  Unmatched observed spots contribute
``(max_match_px, max_match_px)`` to the residual vector — a soft wall
that steers the optimizer away from orientations that leave spots orphaned.

Residual vector length
----------------------
All residual functions return a vector of length ``2 * N_obs_use``,
regardless of how many simulated spots exist.  The fixed length makes
them directly compatible with ``scipy.optimize.least_squares``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial

import numpy as np
from scipy.optimize import least_squares, linear_sum_assignment
from scipy.spatial.transform import Rotation

from .simulation import (
    BM32_KB,
    E_MAX_eV,
    E_MIN_eV,
    F2_THRESHOLD,
    HMAX,
    precompute_allowed_hkl,
    simulate_laue,
    simulate_laue_stack,
    simulate_mixed_phases,
)

# ─────────────────────────────────────────────────────────────────────────────
# Result containers
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class OrientationFitResult:
    """
    Result of a single-crystal orientation refinement (:func:`fit_orientation`).

    Attributes
    ----------
    U          : (3, 3) ndarray  Refined orientation matrix (LT frame).
    U0         : (3, 3) ndarray  Starting orientation passed to the fitter.
    rotvec     : (3,) ndarray    Rotation vector δω (radians) such that
                                 ``U = Rotation.from_rotvec(rotvec) @ U0``.
                                 Its magnitude is the total rotation angle.
    cost       : float           ½ Σ residuals² as returned by ``least_squares``.
    rms_px     : float           RMS pixel distance of matched observed-simulated
                                 pairs (``nan`` if no matches).
    n_matched  : int             Number of observed spots matched within
                                 ``max_match_px`` at the solution.
    n_obs      : int             Number of observed spots used in the fit.
    n_sim      : int             Number of simulated spots on the detector at
                                 the solution (before any ``top_n_sim`` cut).
    match_rate : float           ``n_matched / n_obs``.
    success    : bool            ``True`` if the optimizer converged.
    message    : str             Human-readable optimizer termination message.
    optimizer  : OptimizeResult  Raw ``scipy.optimize.OptimizeResult`` (not shown
                                 in repr); inspect for Jacobian, gradient, etc.
    """

    U          : np.ndarray
    U0         : np.ndarray
    rotvec     : np.ndarray
    cost       : float
    rms_px     : float
    n_matched  : int
    n_obs      : int
    n_sim      : int
    match_rate : float
    success    : bool
    message    : str
    optimizer  : object = field(repr=False)

    def __str__(self) -> str:
        status = "OK" if self.success else "FAILED"
        dw = float(np.degrees(np.linalg.norm(self.rotvec)))
        return (
            f"OrientationFitResult [{status}]  "
            f"rms={self.rms_px:.2f} px  "
            f"matched={self.n_matched}/{self.n_obs} ({self.match_rate:.0%})  "
            f"|δω|={dw:.4f}°"
        )


@dataclass
class StackFitResult:
    """
    Result of a layered-crystal orientation refinement (:func:`fit_orientation_stack`).

    A single global rotation is applied to all layers, preserving their
    relative orientation relationships.

    Attributes
    ----------
    R_global  : (3, 3) ndarray        Global rotation matrix applied to every layer.
    rotvec    : (3,) ndarray          Rotation vector (radians) for ``R_global``.
    U_layers  : list of (3, 3) arrays Refined U matrix for each layer, in
                                      ``stack.all_layers`` order.
    U0_layers : list of (3, 3) arrays Starting U matrix for each layer.
    cost      : float                 ½ Σ residuals² at convergence.
    rms_px    : float                 RMS pixel distance of matched spot pairs.
    n_matched : int                   Matched spots within ``max_match_px``.
    n_obs     : int                   Observed spots used.
    n_sim     : int                   Simulated spots on detector at solution.
    match_rate: float                 ``n_matched / n_obs``.
    success   : bool                  Optimizer convergence flag.
    message   : str                   Optimizer termination message.
    optimizer : OptimizeResult        Raw ``scipy.optimize.OptimizeResult``.
    """

    R_global   : np.ndarray
    rotvec     : np.ndarray
    U_layers   : list[np.ndarray]
    U0_layers  : list[np.ndarray]
    cost       : float
    rms_px     : float
    n_matched  : int
    n_obs      : int
    n_sim      : int
    match_rate : float
    success    : bool
    message    : str
    optimizer  : object = field(repr=False)

    def __str__(self) -> str:
        status = "OK" if self.success else "FAILED"
        dw = float(np.degrees(np.linalg.norm(self.rotvec)))
        return (
            f"StackFitResult [{status}]  "
            f"rms={self.rms_px:.2f} px  "
            f"matched={self.n_matched}/{self.n_obs} ({self.match_rate:.0%})  "
            f"|δω|={dw:.4f}°"
        )


@dataclass
class MixedFitResult:
    """
    Result of a multi-phase orientation refinement (:func:`fit_orientation_mixed`).

    Attributes
    ----------
    U_phases  : list of (3, 3) arrays Refined orientation matrix per phase,
                                      in input order.
    U0_phases : list of (3, 3) arrays Starting orientation matrix per phase.
    rotvecs   : list of (3,) arrays   Rotation vector per phase.  In shared
                                      mode every entry is identical; in per-phase
                                      mode each entry is independent.
    cost      : float                 ½ Σ residuals² at convergence.
    rms_px    : float                 RMS pixel distance of matched spot pairs.
    n_matched : int                   Matched spots within ``max_match_px``.
    n_obs     : int                   Observed spots used.
    n_sim     : int                   Total simulated spots on detector at solution
                                      (all phases combined).
    match_rate: float                 ``n_matched / n_obs``.
    success   : bool                  Optimizer convergence flag.
    message   : str                   Optimizer termination message.
    optimizer : OptimizeResult        Raw ``scipy.optimize.OptimizeResult``.
    """

    U_phases   : list[np.ndarray]
    U0_phases  : list[np.ndarray]
    rotvecs    : list[np.ndarray]
    cost       : float
    rms_px     : float
    n_matched  : int
    n_obs      : int
    n_sim      : int
    match_rate : float
    success    : bool
    message    : str
    optimizer  : object = field(repr=False)

    def __str__(self) -> str:
        status = "OK" if self.success else "FAILED"
        return (
            f"MixedFitResult [{status}]  "
            f"rms={self.rms_px:.2f} px  "
            f"matched={self.n_matched}/{self.n_obs} ({self.match_rate:.0%})"
        )


@dataclass
class StrainFitResult:
    """
    Result of a simultaneous orientation + strain refinement
    (:func:`fit_strain_orientation`).

    Attributes
    ----------
    U             : (3, 3) ndarray  Pure rotation part of the refined matrix.
                                    ``U = Rotation.from_rotvec(rotvec) @ U0``.
    U0            : (3, 3) ndarray  Starting orientation passed to the fitter.
    U_eff         : (3, 3) ndarray  Full deformation matrix used by the
                                    simulator: ``U @ (I + strain_tensor)``.
                                    Pass this as ``U`` to
                                    :func:`~nrxrdct.laue.simulate_laue` to
                                    reproduce the fitted spot pattern.
    rotvec        : (3,) ndarray    Rotation increment δω (radians).
    strain_tensor : (3, 3) ndarray  Symmetric strain tensor in the crystal
                                    frame.  Diagonal entries are axial strains
                                    (Δa/a, Δb/b, Δc/c); off-diagonal entries
                                    are the engineering shear strains / 2.
    strain_voigt  : (6,) ndarray    Voigt representation
                                    ``[ε_xx, ε_yy, ε_zz, ε_xy, ε_xz, ε_yz]``.
                                    Components not listed in ``fit_strain``
                                    are zero.
    fit_strain    : tuple[str, …]   Strain components that were free parameters.
    cost          : float           ½ Σ residuals² at convergence.
    rms_px        : float           RMS pixel distance of matched pairs.
    n_matched     : int             Matched spots within ``max_match_px``.
    n_obs         : int             Observed spots used.
    n_sim         : int             Simulated spots on detector at solution.
    match_rate    : float           ``n_matched / n_obs``.
    success       : bool            Optimizer convergence flag.
    message       : str             Optimizer termination message.
    optimizer     : OptimizeResult  Raw ``scipy.optimize.OptimizeResult``
                                    (not shown in repr).
    """

    U             : np.ndarray
    U0            : np.ndarray
    U_eff         : np.ndarray
    rotvec        : np.ndarray
    strain_tensor : np.ndarray
    strain_voigt  : np.ndarray
    fit_strain    : tuple
    cost          : float
    rms_px        : float
    n_matched     : int
    n_obs         : int
    n_sim         : int
    match_rate    : float
    success       : bool
    message       : str
    optimizer     : object = field(repr=False)

    def __str__(self) -> str:
        status = "OK" if self.success else "FAILED"
        dw = float(np.degrees(np.linalg.norm(self.rotvec)))
        e  = self.strain_voigt
        return (
            f"StrainFitResult [{status}]  "
            f"rms={self.rms_px:.2f} px  "
            f"matched={self.n_matched}/{self.n_obs} ({self.match_rate:.0%})  "
            f"|δω|={dw:.4f}°  "
            f"ε_diag=[{e[0]:.2e}, {e[1]:.2e}, {e[2]:.2e}]"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────


def _extract_sim_xy(spots: list) -> np.ndarray:
    """
    Extract pixel positions from a simulate_laue spot list.

    Parameters
    ----------
    spots : list of dicts
        Output of any ``simulate_laue*`` function.  Each dict is expected
        to contain a ``'pix'`` key with a ``[xcam, ycam]`` value; spots
        that reach the detector have a non-None ``'pix'``.

    Returns
    -------
    xy : (N_sim, 2) ndarray
        Pixel positions ``[xcam, ycam]`` for all on-detector spots.
        Returns ``(0, 2)`` if the spot list is empty or no spot is on the
        detector.
    """
    xy = [s["pix"] for s in spots if s.get("pix") is not None]
    return np.array(xy, dtype=float) if xy else np.empty((0, 2), dtype=float)


def _match_spots(
    obs_xy: np.ndarray,
    sim_xy: np.ndarray,
    max_match_px: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Optimal one-to-one assignment of observed to simulated spots.

    Uses the Hungarian algorithm (``scipy.optimize.linear_sum_assignment``)
    on a pixel-distance cost matrix capped at ``max_match_px``.  Capping
    means the optimizer treats any pair farther than the cap as equally bad,
    preventing a single large outlier from dominating the assignment.

    Parameters
    ----------
    obs_xy      : (N_obs, 2) ndarray   Observed pixel positions [xcam, ycam].
    sim_xy      : (N_sim, 2) ndarray   Simulated pixel positions [xcam, ycam].
    max_match_px : float               Distance cap applied before solving the
                                       assignment problem.

    Returns
    -------
    row_ind : (K,) int array   Indices into ``obs_xy`` for accepted pairs.
    col_ind : (K,) int array   Corresponding indices into ``sim_xy``.
    dist_px : (K,) float array Euclidean pixel distance for each pair.
                               Note: pairs where the true distance exceeds
                               ``max_match_px`` are included — callers must
                               filter on ``dist_px`` themselves.
    """
    if len(obs_xy) == 0 or len(sim_xy) == 0:
        empty = np.array([], dtype=int)
        return empty, empty, np.array([], dtype=float)

    diff     = obs_xy[:, None, :] - sim_xy[None, :, :]       # (N_obs, N_sim, 2)
    dist_px  = np.sqrt((diff ** 2).sum(axis=-1))              # (N_obs, N_sim)
    row_ind, col_ind = linear_sum_assignment(np.minimum(dist_px, max_match_px))
    return row_ind, col_ind, dist_px[row_ind, col_ind]


def _build_residuals(
    obs_use: np.ndarray,
    sim_xy: np.ndarray,
    max_match_px: float,
) -> np.ndarray:
    """
    Build the fixed-length residual vector used by ``least_squares``.

    For each observed spot the nearest simulated spot is found via
    :func:`_match_spots`.  If the assigned pair is within ``max_match_px``
    the residual components are the signed pixel differences (Δx, Δy).
    Otherwise both components are set to ``max_match_px``, acting as a
    soft penalty wall that steers the optimizer toward orientations where
    all spots are explained.

    Parameters
    ----------
    obs_use     : (N_obs, 2)  Observed pixel positions [xcam, ycam].
    sim_xy      : (N_sim, 2)  Simulated pixel positions, or empty array.
    max_match_px : float      Penalty value assigned to unmatched spots.

    Returns
    -------
    residuals : (2 * N_obs,) ndarray
        Interleaved [Δx₀, Δy₀, Δx₁, Δy₁, ...].  Length is always
        ``2 * N_obs`` regardless of how many simulated spots exist,
        making the vector directly usable with ``least_squares``.
    """
    N_obs = len(obs_use)
    residuals = np.full(2 * N_obs, max_match_px, dtype=float)

    sim_use = sim_xy if len(sim_xy) > 0 else None
    if sim_use is None:
        return residuals

    row_ind, col_ind, dist_px = _match_spots(obs_use, sim_use, max_match_px)
    for r, c, d in zip(row_ind, col_ind, dist_px):
        if d < max_match_px:
            residuals[2 * r]     = obs_use[r, 0] - sim_use[c, 0]  # Δx
            residuals[2 * r + 1] = obs_use[r, 1] - sim_use[c, 1]  # Δy

    return residuals


def _compute_match_stats(
    residuals: np.ndarray, max_match_px: float, N_obs: int
) -> tuple[int, float]:
    """
    Derive match count and RMS error from a residual vector.

    Unmatched spots were filled with exactly ``max_match_px`` in both Δx
    and Δy by :func:`_build_residuals`.  They are detected by checking
    ``|Δx| ≥ max_match_px − ε  AND  |Δy| ≥ max_match_px − ε``.

    Parameters
    ----------
    residuals    : (2 * N_obs,) ndarray  Residual vector from :func:`_build_residuals`.
    max_match_px : float                 Penalty threshold used when building residuals.
    N_obs        : int                   Number of observed spots (= len(residuals) // 2).

    Returns
    -------
    n_matched : int    Number of spots with a matched simulated counterpart.
    rms_px    : float  RMS Euclidean pixel distance of matched pairs only.
                       Returns ``nan`` when no spots are matched.
    """
    r = residuals.reshape(N_obs, 2)
    unmatched = np.all(np.abs(r) >= max_match_px - 1e-9, axis=1)
    matched   = ~unmatched
    n_matched = int(matched.sum())
    rms_px    = float(np.sqrt((r[matched] ** 2).mean())) if n_matched > 0 else float("nan")
    return n_matched, rms_px


def _normalise_phases(phases: list) -> list[dict]:
    """
    Convert phases to a uniform list-of-dicts representation.

    :func:`simulate_mixed_phases` accepts phases as either dicts or
    ``(crystal, U, volume_fraction[, label])`` tuples.  This helper
    normalises both forms into dicts so the fitting code can always
    mutate ``p["U"]`` without special-casing.

    Parameters
    ----------
    phases : list of dict or tuple
        Input phase list in either accepted format.

    Returns
    -------
    out : list of dict
        New list of independent dicts (shallow-copies of input dicts;
        tuples converted).  The originals are not mutated.
    """
    out = []
    for p in phases:
        if isinstance(p, dict):
            out.append(dict(p))
        else:
            d = {"crystal": p[0], "U": p[1], "volume_fraction": p[2]}
            if len(p) > 3:
                d["label"] = p[3]
            out.append(d)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Strain helpers
# ─────────────────────────────────────────────────────────────────────────────

_STRAIN_IDX: dict[str, tuple[int, int]] = {
    "e_xx": (0, 0), "e_yy": (1, 1), "e_zz": (2, 2),
    "e_xy": (0, 1), "e_xz": (0, 2), "e_yz": (1, 2),
}
_STRAIN_ALL: tuple[str, ...] = ("e_xx", "e_yy", "e_zz", "e_xy", "e_xz", "e_yz")


def _strain_matrix(strain_vals, fit_strain) -> np.ndarray:
    """Return the symmetric (3,3) strain tensor for the given Voigt components."""
    eps = np.zeros((3, 3))
    for val, name in zip(strain_vals, fit_strain):
        i, j = _STRAIN_IDX[name]
        eps[i, j] = float(val)
        eps[j, i] = float(val)
    return eps


def _strain_to_voigt(strain_vals, fit_strain) -> np.ndarray:
    """Pack strain components into the full 6-element Voigt vector."""
    voigt = np.zeros(6)
    for val, name in zip(strain_vals, fit_strain):
        voigt[_STRAIN_ALL.index(name)] = float(val)
    return voigt


# ─────────────────────────────────────────────────────────────────────────────
# Residual functions  (compatible with scipy.optimize.least_squares)
# ─────────────────────────────────────────────────────────────────────────────


def laue_residuals(
    rotvec: np.ndarray,
    crystal,
    camera,
    obs_xy: np.ndarray,
    U0: np.ndarray,
    E_min_eV: float = E_MIN_eV,
    E_max_eV: float = E_MAX_eV,
    source: str = "bending_magnet",
    source_kwargs: dict | None = None,
    hmax: int = HMAX,
    f2_thresh: float = F2_THRESHOLD,
    kb_params=BM32_KB,
    max_match_px: float = 30.0,
    top_n_obs: int | None = None,
    top_n_sim: int | None = None,
    geometry_only: bool = False,
    allowed_hkl=None,
) -> np.ndarray:
    """
    Pixel-space residual vector for single-crystal orientation refinement.

    Intended to be passed directly to ``scipy.optimize.least_squares``
    via :func:`functools.partial` (all arguments except ``rotvec`` frozen).

    Parameters
    ----------
    rotvec       : (3,) ndarray   Rotation-vector increment δω (radians).
                                  The orientation evaluated at each call is
                                  ``Rotation.from_rotvec(δω) @ U0``.
                                  Initialise with ``np.zeros(3)``.
    crystal      : Crystal        xrayutilities crystal structure.
    camera       : Camera         Detector geometry (pixel size, distance, …).
    obs_xy       : (N_obs, 2)     Observed pixel positions ``[xcam, ycam]``,
                                  sorted by descending intensity.  Pass
                                  ``peaklist[:, :2]`` directly from
                                  :func:`~nrxrdct.laue.segmentation.convert_spotsfile2peaklist`.
    U0           : (3, 3)         Starting orientation matrix (LT frame,
                                  x-axis // beam direction).
    E_min_eV     : float          Low-energy cut-off of the white beam (eV).
    E_max_eV     : float          High-energy cut-off (eV).
    source       : str            Spectral model — ``'bending_magnet'`` or
                                  ``'undulator'``.
    source_kwargs: dict or None   Extra keyword arguments forwarded to the
                                  spectral function (e.g. ``{'B': 0.4}`` for
                                  a bending magnet field).
    hmax         : int            Maximum Miller index searched.
    f2_thresh    : float          Minimum squared structure factor |F|² to
                                  include a reflection.
    kb_params    : KB params      KB mirror reflectivity parameters
                                  (see :data:`BM32_KB`).
    max_match_px : float          Pixel radius inside which a simulated spot
                                  is considered a match for an observed spot.
                                  Unmatched observations contribute
                                  ``(max_match_px, max_match_px)`` to the
                                  residual vector.
    top_n_obs    : int or None    Use only the N brightest observed spots.
                                  ``None`` uses all.
    top_n_sim    : int or None    Consider only the N brightest simulated spots
                                  (after intensity-sorting by the simulator).

    Returns
    -------
    residuals : (2 * N_obs_use,) ndarray
        Interleaved ``[Δx₀, Δy₀, Δx₁, Δy₁, …]``.  Length is fixed at
        ``2 * min(N_obs, top_n_obs)`` so it is compatible with
        ``least_squares``.
    """
    delta_R = Rotation.from_rotvec(np.asarray(rotvec, dtype=float)).as_matrix()
    U = delta_R @ np.asarray(U0, dtype=float)

    spots = simulate_laue(
        crystal, U, camera,
        E_min=E_min_eV, E_max=E_max_eV,
        source=source, source_kwargs=source_kwargs,
        hmax=hmax, f2_thresh=f2_thresh, kb_params=kb_params,
        geometry_only=geometry_only,
        allowed_hkl=allowed_hkl,
    )

    obs_use = np.asarray(obs_xy, dtype=float)
    if top_n_obs is not None:
        obs_use = obs_use[:top_n_obs]

    sim_xy = _extract_sim_xy(spots)
    if top_n_sim is not None:
        sim_xy = sim_xy[:top_n_sim]

    return _build_residuals(obs_use, sim_xy, max_match_px)


def laue_stack_residuals(
    rotvec: np.ndarray,
    stack,
    camera,
    obs_xy: np.ndarray,
    U0_layers: list[np.ndarray],
    E_min_eV: float = E_MIN_eV,
    E_max_eV: float = E_MAX_eV,
    source: str = "bending_magnet",
    source_kwargs: dict | None = None,
    hmax: int = HMAX,
    f2_thresh: float = F2_THRESHOLD,
    kb_params=BM32_KB,
    structure_model: str = "average",
    max_match_px: float = 30.0,
    top_n_obs: int | None = None,
    top_n_sim: int | None = None,
    geometry_only: bool = False,
    allowed_hkl=None,
) -> np.ndarray:
    """
    Pixel-space residual vector for a layered crystal — single global rotation.

    A single rotation ``R = Rotation.from_rotvec(δω)`` is applied to every
    layer: ``stack.all_layers[i].U = R @ U0_layers[i]``.  All inter-layer
    orientation relationships are therefore preserved throughout the fit.

    The stack is mutated in-place on every call.  When used through
    :func:`fit_orientation_stack` the original U matrices are automatically
    restored after the optimizer returns.  If called directly, the caller is
    responsible for saving and restoring ``layer.U``.

    Parameters
    ----------
    rotvec       : (3,) ndarray     Global rotation-vector increment δω (rad).
                                    Initialise with ``np.zeros(3)``.
    stack        : LayeredCrystal   Layered structure; mutated in-place.
    camera       : Camera           Detector geometry.
    obs_xy       : (N_obs, 2)       Observed pixel positions ``[xcam, ycam]``.
    U0_layers    : list of (3, 3)   Base orientation for each layer, in
                                    ``stack.all_layers`` order.  Typically
                                    captured as
                                    ``[l.U.copy() for l in stack.all_layers]``
                                    before the first call.
    E_min_eV     : float            Low-energy cut-off (eV).
    E_max_eV     : float            High-energy cut-off (eV).
    source       : str              Spectral model (``'bending_magnet'`` or
                                    ``'undulator'``).
    source_kwargs: dict or None     Extra kwargs for the spectral function.
    hmax         : int              Maximum Miller index.
    f2_thresh    : float            Minimum |F|² threshold.
    kb_params    :                  KB mirror reflectivity parameters.
    structure_model : str           How to combine layer contributions —
                                    ``'average'`` (default) or ``'incoherent'``.
    max_match_px : float            Match radius in pixels.
    top_n_obs    : int or None      Brightest N observed spots to use.
    top_n_sim    : int or None      Brightest N simulated spots to consider.

    Returns
    -------
    residuals : (2 * N_obs_use,) ndarray
        Fixed-length interleaved ``[Δx, Δy]`` residual vector.
    """
    delta_R = Rotation.from_rotvec(np.asarray(rotvec, dtype=float)).as_matrix()
    for layer, U0 in zip(stack.all_layers, U0_layers):
        layer.U = delta_R @ U0

    spots = simulate_laue_stack(
        stack, camera,
        E_min_eV=E_min_eV, E_max_eV=E_max_eV,
        source=source, source_kwargs=source_kwargs,
        hmax=hmax, f2_thresh=f2_thresh, kb_params=kb_params,
        structure_model=structure_model,
        verbose=False,
        geometry_only=geometry_only,
        allowed_hkl=allowed_hkl,
    )

    obs_use = np.asarray(obs_xy, dtype=float)
    if top_n_obs is not None:
        obs_use = obs_use[:top_n_obs]

    sim_xy = _extract_sim_xy(spots)
    if top_n_sim is not None:
        sim_xy = sim_xy[:top_n_sim]

    return _build_residuals(obs_use, sim_xy, max_match_px)


def laue_mixed_residuals(
    params: np.ndarray,
    phases: list[dict],
    camera,
    obs_xy: np.ndarray,
    U0_list: list[np.ndarray],
    shared: bool = True,
    E_min_eV: float = E_MIN_eV,
    E_max_eV: float = E_MAX_eV,
    source: str = "bending_magnet",
    source_kwargs: dict | None = None,
    hmax: int = HMAX,
    f2_thresh: float | None = None,
    kb_params=BM32_KB,
    structure_model: str = "average",
    max_match_px: float = 30.0,
    top_n_obs: int | None = None,
    top_n_sim: int | None = None,
    geometry_only: bool = False,
    allowed_hkl=None,
) -> np.ndarray:
    """
    Pixel-space residual vector for a multi-phase Laue pattern.

    Parameters
    ----------
    params       : (3,) or (3 * N_phases,) ndarray
                   Rotation-vector increment(s).  In shared mode (``shared=True``)
                   a single 3-element vector rotates every phase together.  In
                   per-phase mode (``shared=False``) the vector has length
                   ``3 * N_phases``: elements ``[3i : 3i+3]`` rotate phase *i*.
                   Initialise with ``np.zeros(3)`` or ``np.zeros(3 * N_phases)``.
    phases       : list of dicts   Phase descriptors; ``p["U"]`` is updated
                                   in-place on every call.  Each dict must
                                   contain ``'crystal'``, ``'U'``, and
                                   ``'volume_fraction'``.
    camera       : Camera          Detector geometry.
    obs_xy       : (N_obs, 2)      Observed pixel positions ``[xcam, ycam]``.
    U0_list      : list of (3, 3)  Base orientation per phase, in input order.
    shared       : bool            ``True`` → one global rotation for all phases.
                                   ``False`` → independent rotation per phase.
    E_min_eV     : float           Low-energy cut-off (eV).
    E_max_eV     : float           High-energy cut-off (eV).
    source       : str             Spectral model (``'bending_magnet'`` or
                                   ``'undulator'``).
    source_kwargs: dict or None    Extra kwargs for the spectral function.
    hmax         : int             Maximum Miller index.
    f2_thresh    : float or None   Minimum |F|² threshold.
    kb_params    :                 KB mirror reflectivity parameters.
    structure_model : str          Layer combination model (``'average'`` or
                                   ``'incoherent'``).
    max_match_px : float           Match radius in pixels.
    top_n_obs    : int or None     Brightest N observed spots to use.
    top_n_sim    : int or None     Brightest N simulated spots to consider.

    Returns
    -------
    residuals : (2 * N_obs_use,) ndarray
        Fixed-length interleaved ``[Δx, Δy]`` residual vector.
    """
    params = np.asarray(params, dtype=float)

    if shared:
        delta_R = Rotation.from_rotvec(params).as_matrix()
        for p, U0 in zip(phases, U0_list):
            p["U"] = delta_R @ U0
    else:
        for i, (p, U0) in enumerate(zip(phases, U0_list)):
            rv = params[3 * i : 3 * i + 3]
            p["U"] = Rotation.from_rotvec(rv).as_matrix() @ U0

    spots = simulate_mixed_phases(
        phases, camera,
        E_min_eV=E_min_eV, E_max_eV=E_max_eV,
        source=source, source_kwargs=source_kwargs,
        hmax=hmax, f2_thresh=f2_thresh, kb_params=kb_params,
        structure_model=structure_model,
        verbose=False,
        geometry_only=geometry_only,
        allowed_hkl=allowed_hkl,
    )

    obs_use = np.asarray(obs_xy, dtype=float)
    if top_n_obs is not None:
        obs_use = obs_use[:top_n_obs]

    sim_xy = _extract_sim_xy(spots)
    if top_n_sim is not None:
        sim_xy = sim_xy[:top_n_sim]

    return _build_residuals(obs_use, sim_xy, max_match_px)


def laue_strain_residuals(
    params: np.ndarray,
    crystal,
    camera,
    obs_xy: np.ndarray,
    U0: np.ndarray,
    fit_strain: tuple[str, ...] = _STRAIN_ALL,
    strain_scale: float = 1e-4,
    E_min_eV: float = E_MIN_eV,
    E_max_eV: float = E_MAX_eV,
    source: str = "bending_magnet",
    source_kwargs: dict | None = None,
    hmax: int = HMAX,
    f2_thresh: float = F2_THRESHOLD,
    kb_params=BM32_KB,
    max_match_px: float = 30.0,
    top_n_obs: int | None = None,
    top_n_sim: int | None = None,
    geometry_only: bool = False,
    allowed_hkl=None,
) -> np.ndarray:
    """
    Pixel-space residual vector for simultaneous orientation + strain refinement.

    The effective deformation matrix applied to the crystal is::

        U_eff = R(δω) @ U0 @ (I + ε)

    where R(δω) is a small rotation and ε is the symmetric strain tensor.
    Because ``simulate_laue`` does not enforce SO(3), the non-orthogonal
    ``U_eff`` correctly shifts every d-spacing by the corresponding strain
    component.

    Parameters
    ----------
    params       : (3 + n_strain,) ndarray
                   First 3 elements: rotation-vector increment δω (radians).
                   Remaining ``n_strain`` elements: strain components scaled by
                   ``strain_scale`` (i.e. divide by ``strain_scale`` to get
                   physical strain).  Initialise with ``np.zeros(3 + n_strain)``.
    crystal      : Crystal        xrayutilities crystal structure.
    camera       : Camera         Detector geometry.
    obs_xy       : (N_obs, 2)     Observed pixel positions ``[xcam, ycam]``.
    U0           : (3, 3)         Starting orientation matrix.
    fit_strain   : tuple of str   Which strain components are free.  Any
                                  subset of ``('e_xx','e_yy','e_zz','e_xy',
                                  'e_xz','e_yz')``.  Default: all six.
    strain_scale : float          Internal scale factor for strain parameters.
                                  Optimizer parameters = physical_strain /
                                  strain_scale.  Default 1e-4 keeps parameters
                                  near order-1 for typical strains of 10⁻⁴–10⁻³.
    E_min_eV, E_max_eV, source, source_kwargs, hmax, f2_thresh, kb_params,
    max_match_px, top_n_obs, top_n_sim, geometry_only, allowed_hkl
                   Forwarded to :func:`simulate_laue`; see :func:`laue_residuals`.

    Returns
    -------
    residuals : (2 * N_obs_use,) ndarray
        Fixed-length interleaved ``[Δx, Δy]`` residual vector.
    """
    params = np.asarray(params, dtype=float)
    rotvec = params[:3]
    strain_vals = params[3:] * strain_scale

    R = Rotation.from_rotvec(rotvec).as_matrix()
    eps = _strain_matrix(strain_vals, fit_strain)
    U_eff = R @ np.asarray(U0, dtype=float) @ (np.eye(3) + eps)

    spots = simulate_laue(
        crystal, U_eff, camera,
        E_min=E_min_eV, E_max=E_max_eV,
        source=source, source_kwargs=source_kwargs,
        hmax=hmax, f2_thresh=f2_thresh, kb_params=kb_params,
        geometry_only=geometry_only,
        allowed_hkl=allowed_hkl,
    )

    obs_use = np.asarray(obs_xy, dtype=float)
    if top_n_obs is not None:
        obs_use = obs_use[:top_n_obs]

    sim_xy = _extract_sim_xy(spots)
    if top_n_sim is not None:
        sim_xy = sim_xy[:top_n_sim]

    return _build_residuals(obs_use, sim_xy, max_match_px)


# ─────────────────────────────────────────────────────────────────────────────
# Fitting wrappers
# ─────────────────────────────────────────────────────────────────────────────

def fit_orientation(
    crystal,
    camera,
    obs_xy: np.ndarray,
    U0: np.ndarray,
    E_min_eV: float = E_MIN_eV,
    E_max_eV: float = E_MAX_eV,
    source: str = "bending_magnet",
    source_kwargs: dict | None = None,
    hmax: int = HMAX,
    f2_thresh: float = F2_THRESHOLD,
    kb_params=BM32_KB,
    max_match_px: float = 30.0,
    top_n_obs: int | None = None,
    top_n_sim: int | None = None,
    method: str = "lm",
    ftol: float = 1e-6,
    xtol: float = 1e-6,
    gtol: float = 1e-8,
    max_nfev: int = 500,
    geometry_only: bool = True,
    z_scan_step_deg: float | None = None,
    z_axis: np.ndarray | None = None,
    verbose: bool = False,
) -> OrientationFitResult:
    """
    Refine the orientation matrix of a single crystal to match observed spots.

    Wraps :func:`laue_residuals` + ``scipy.optimize.least_squares``.

    Parameters
    ----------
    crystal      : Crystal        xrayutilities crystal structure.
    camera       : Camera         Detector geometry.
    obs_xy       : (N_obs, 2)     Observed pixel positions ``[xcam, ycam]``,
                                  sorted by descending intensity.  Pass
                                  ``peaklist[:, :2]`` directly from
                                  :func:`~nrxrdct.laue.segmentation.convert_spotsfile2peaklist`.
    U0           : (3, 3)         Starting orientation matrix (LT frame,
                                  x-axis // beam direction).
    E_min_eV     : float          Low-energy cut-off of the white beam (eV).
    E_max_eV     : float          High-energy cut-off (eV).
    source       : str            Spectral model — ``'bending_magnet'`` or
                                  ``'undulator'``.
    source_kwargs: dict or None   Extra kwargs forwarded to the spectral function.
    hmax         : int            Maximum Miller index searched.
    f2_thresh    : float          Minimum squared structure factor |F|² to
                                  include a reflection.
    kb_params    :                KB mirror reflectivity parameters.
    max_match_px : float          Pixel tolerance for spot matching.  Start
                                  large (~30–50 px) for a rough alignment and
                                  tighten for final refinement.  Observed spots
                                  farther than this from any simulated spot are
                                  penalised but not excluded.
    top_n_obs    : int or None    Use only the brightest N observed spots.
                                  Reduces cost per iteration; useful when the
                                  spot list contains many weak peaks.
    top_n_sim    : int or None    Consider only the brightest N simulated spots.
    method       : str            ``least_squares`` algorithm: ``'lm'`` (fast,
                                  unconstrained Levenberg–Marquardt) or ``'trf'``
                                  (trust-region reflective, more robust for large
                                  initial misalignments).
    ftol, xtol, gtol : float      Convergence tolerances forwarded to
                                  ``least_squares``.
    max_nfev     : int            Maximum number of residual evaluations.
    z_scan_step_deg : float or None
                                  When not ``None``, perform a coarse grid
                                  search over in-plane rotations before the
                                  local refinement.  The starting orientation
                                  ``U0`` is rotated around ``z_axis`` in steps
                                  of ``z_scan_step_deg`` degrees from 0° to
                                  360°.  The candidate with the lowest residual
                                  cost is used as the starting point for
                                  ``least_squares``.  Useful for non-cubic
                                  crystals where Euler-angle initialisation may
                                  land in the wrong basin.  Typical values:
                                  10–30° for a fast scan, 2–5° for a fine one.
    z_axis       : (3,) array or None
                                  Unit vector (in the LaueTools lab frame) to
                                  rotate around during the scan.  Defaults to
                                  the lab Z axis ``[0, 0, 1]`` (vertical).
                                  Pass the crystal c-axis direction (in the lab
                                  frame) for a structure-aware scan.
    verbose      : bool           Print a one-line summary after convergence.

    Returns
    -------
    OrientationFitResult
        Call ``str(result)`` for a one-line summary.  Apply the refined
        orientation with::

            spots = simulate_laue(crystal, result.U, camera, ...)
    """
    U0_input = np.asarray(U0, dtype=float)   # preserve original for result.U0
    U0 = U0_input.copy()
    obs_use = np.asarray(obs_xy, dtype=float)
    if top_n_obs is not None:
        obs_use = obs_use[:top_n_obs]
    N_obs = len(obs_use)

    if verbose:
        print(f"fit_orientation: {N_obs} observed spots")

    # Precompute which (hkl) are structurally allowed once — avoids calling
    # crystal.StructureFactor on every optimizer iteration.
    _allowed = (
        precompute_allowed_hkl(crystal, hmax, f2_thresh=f2_thresh)
        if geometry_only else None
    )

    # ── coarse Z-rotation scan (optional) ────────────────────────────────────
    if z_scan_step_deg is not None:
        _ax = np.asarray(z_axis if z_axis is not None else [0.0, 0.0, 1.0],
                         dtype=float)
        _ax = _ax / np.linalg.norm(_ax)
        angles_deg = np.arange(0.0, 360.0, float(z_scan_step_deg))

        _scan_kwargs = dict(
            crystal=crystal, camera=camera, obs_xy=obs_use,
            E_min_eV=E_min_eV, E_max_eV=E_max_eV,
            source=source, source_kwargs=source_kwargs,
            hmax=hmax, f2_thresh=f2_thresh, kb_params=kb_params,
            max_match_px=max_match_px, top_n_obs=None, top_n_sim=top_n_sim,
            geometry_only=False, allowed_hkl=_allowed,
        )

        best_cost = np.inf
        best_U0 = U0.copy()
        best_angle = 0.0

        for alpha in angles_deg:
            R_z = Rotation.from_rotvec(np.radians(alpha) * _ax).as_matrix()
            U_trial = R_z @ U0
            res = laue_residuals(np.zeros(3), U0=U_trial, **_scan_kwargs)
            cost = float(np.dot(res, res))
            if cost < best_cost:
                best_cost = cost
                best_U0 = U_trial.copy()
                best_angle = alpha

        if verbose:
            print(
                f"  Z-scan ({len(angles_deg)} steps, Δ={z_scan_step_deg}°): "
                f"best angle = {best_angle:.1f}°, "
                f"cost = {best_cost:.2f}"
            )
        U0 = best_U0

    fun = partial(
        laue_residuals,
        crystal=crystal, camera=camera, obs_xy=obs_use, U0=U0,
        E_min_eV=E_min_eV, E_max_eV=E_max_eV,
        source=source, source_kwargs=source_kwargs,
        hmax=hmax, f2_thresh=f2_thresh, kb_params=kb_params,
        max_match_px=max_match_px, top_n_obs=None, top_n_sim=top_n_sim,
        geometry_only=False, allowed_hkl=_allowed,
    )

    opt = least_squares(
        fun, x0=np.zeros(3),
        method=method, ftol=ftol, xtol=xtol, gtol=gtol, max_nfev=max_nfev,
    )

    U_final = Rotation.from_rotvec(opt.x).as_matrix() @ U0
    n_matched, rms_px = _compute_match_stats(opt.fun, max_match_px, N_obs)

    # One extra simulation to report n_sim at solution.
    final_spots = simulate_laue(
        crystal, U_final, camera,
        E_min=E_min_eV, E_max=E_max_eV,
        source=source, source_kwargs=source_kwargs,
        hmax=hmax, f2_thresh=f2_thresh, kb_params=kb_params,
        allowed_hkl=_allowed,
    )
    n_sim = len(_extract_sim_xy(final_spots))

    result = OrientationFitResult(
        U=U_final, U0=U0_input, rotvec=opt.x.copy(),
        cost=float(opt.cost), rms_px=rms_px,
        n_matched=n_matched, n_obs=N_obs, n_sim=n_sim,
        match_rate=n_matched / max(N_obs, 1),
        success=opt.success, message=opt.message, optimizer=opt,
    )

    if verbose:
        print(f"  {result}")

    return result


def fit_orientation_stack(
    stack,
    camera,
    obs_xy: np.ndarray,
    E_min_eV: float = E_MIN_eV,
    E_max_eV: float = E_MAX_eV,
    source: str = "bending_magnet",
    source_kwargs: dict | None = None,
    hmax: int = HMAX,
    f2_thresh: float = F2_THRESHOLD,
    kb_params=BM32_KB,
    structure_model: str = "average",
    max_match_px: float = 30.0,
    top_n_obs: int | None = None,
    top_n_sim: int | None = None,
    method: str = "lm",
    ftol: float = 1e-6,
    xtol: float = 1e-6,
    gtol: float = 1e-8,
    max_nfev: int = 500,
    update_stack: bool = True,
    geometry_only: bool = True,
    verbose: bool = False,
) -> StackFitResult:
    """
    Refine the orientation of a :class:`~nrxrdct.laue.layers.LayeredCrystal`.

    A single global rotation is applied to all layers simultaneously so
    all inter-layer orientation relationships are preserved throughout the
    fit.  The starting U matrix of each layer is snapshotted at call time;
    the stack is restored to a clean state regardless of optimizer success
    or failure.

    Parameters
    ----------
    stack          : LayeredCrystal   Layered structure to fit.  Layer U
                                      matrices are used as starting orientations
                                      and optionally updated after convergence.
    camera         : Camera           Detector geometry.
    obs_xy         : (N_obs, 2)       Observed pixel positions ``[xcam, ycam]``.
    E_min_eV       : float            Low-energy cut-off (eV).
    E_max_eV       : float            High-energy cut-off (eV).
    source         : str              Spectral model (``'bending_magnet'`` or
                                      ``'undulator'``).
    source_kwargs  : dict or None     Extra kwargs for the spectral function.
    hmax           : int              Maximum Miller index.
    f2_thresh      : float            Minimum |F|² threshold.
    kb_params      :                  KB mirror reflectivity parameters.
    structure_model: str              Layer combination model — ``'average'``
                                      (default) or ``'incoherent'``.
    max_match_px   : float            Pixel match radius.
    top_n_obs      : int or None      Brightest N observed spots to use.
    top_n_sim      : int or None      Brightest N simulated spots to consider.
    method         : str              ``least_squares`` algorithm (``'lm'`` or
                                      ``'trf'``).
    ftol, xtol, gtol : float          Convergence tolerances.
    max_nfev       : int              Maximum residual evaluations.
    update_stack   : bool             If ``True`` (default), write the refined
                                      U matrices back into ``stack.all_layers``
                                      after convergence.  Set ``False`` to leave
                                      the stack unchanged and inspect the result
                                      before committing.
    verbose        : bool             Print a one-line summary after convergence.

    Returns
    -------
    StackFitResult
        ``result.R_global`` is the 3×3 rotation applied to all layers.
        ``result.U_layers`` lists the refined U for each layer in
        ``stack.all_layers`` order.
    """
    obs_use = np.asarray(obs_xy, dtype=float)
    if top_n_obs is not None:
        obs_use = obs_use[:top_n_obs]
    N_obs = len(obs_use)

    # Capture the current U matrices as the starting orientations.
    U0_layers = [layer.U.copy() for layer in stack.all_layers]

    if verbose:
        print(
            f"fit_orientation_stack: {N_obs} observed spots, "
            f"{len(stack.all_layers)} layers"
        )

    # Precompute allowed hkl for each unique crystal in the enumeration pool
    # (buffer layers + first MQW layer for "average" model) so that _try_append
    # inside simulate_laue_stack never calls the stack structure factor during
    # fitting.  Keyed by id(crystal) for per-layer lookup.
    if geometry_only:
        _enum_pool = (
            stack.buffer_layers + stack.layers[:1]
            if structure_model == "average"
            else stack.all_layers
        )
        _allowed = {
            id(layer.crystal): precompute_allowed_hkl(
                layer.crystal, hmax, f2_thresh=f2_thresh
            )
            for layer in _enum_pool
        }
    else:
        _allowed = None

    fun = partial(
        laue_stack_residuals,
        stack=stack, camera=camera, obs_xy=obs_use,
        U0_layers=U0_layers,
        E_min_eV=E_min_eV, E_max_eV=E_max_eV,
        source=source, source_kwargs=source_kwargs,
        hmax=hmax, f2_thresh=f2_thresh, kb_params=kb_params,
        structure_model=structure_model,
        max_match_px=max_match_px, top_n_obs=None, top_n_sim=top_n_sim,
        geometry_only=False, allowed_hkl=_allowed,
    )

    try:
        opt = least_squares(
            fun, x0=np.zeros(3),
            method=method, ftol=ftol, xtol=xtol, gtol=gtol, max_nfev=max_nfev,
        )
    finally:
        # Always restore original U matrices so the stack is in a known state.
        for layer, U0 in zip(stack.all_layers, U0_layers):
            layer.U = U0.copy()

    R_global = Rotation.from_rotvec(opt.x).as_matrix()
    U_layers_final = [R_global @ U0 for U0 in U0_layers]

    if update_stack:
        for layer, U_new in zip(stack.all_layers, U_layers_final):
            layer.U = U_new.copy()

    n_matched, rms_px = _compute_match_stats(opt.fun, max_match_px, N_obs)

    # Final simulation for n_sim.
    for layer, U_new in zip(stack.all_layers, U_layers_final):
        layer.U = U_new.copy()
    final_spots = simulate_laue_stack(
        stack, camera, E_min_eV=E_min_eV, E_max_eV=E_max_eV,
        source=source, source_kwargs=source_kwargs,
        hmax=hmax, f2_thresh=f2_thresh, kb_params=kb_params,
        structure_model=structure_model, verbose=False,
        allowed_hkl=_allowed,
    )
    n_sim = len(_extract_sim_xy(final_spots))

    # Restore to the refined state (or original if update_stack is False).
    if not update_stack:
        for layer, U0 in zip(stack.all_layers, U0_layers):
            layer.U = U0.copy()

    result = StackFitResult(
        R_global=R_global, rotvec=opt.x.copy(),
        U_layers=U_layers_final, U0_layers=U0_layers,
        cost=float(opt.cost), rms_px=rms_px,
        n_matched=n_matched, n_obs=N_obs, n_sim=n_sim,
        match_rate=n_matched / max(N_obs, 1),
        success=opt.success, message=opt.message, optimizer=opt,
    )

    if verbose:
        print(f"  {result}")

    return result


def fit_orientation_mixed(
    phases: list,
    camera,
    obs_xy: np.ndarray,
    shared: bool = True,
    E_min_eV: float = E_MIN_eV,
    E_max_eV: float = E_MAX_eV,
    source: str = "bending_magnet",
    source_kwargs: dict | None = None,
    hmax: int = HMAX,
    f2_thresh: float | None = None,
    kb_params=BM32_KB,
    structure_model: str = "average",
    max_match_px: float = 30.0,
    top_n_obs: int | None = None,
    top_n_sim: int | None = None,
    method: str = "lm",
    ftol: float = 1e-6,
    xtol: float = 1e-6,
    gtol: float = 1e-8,
    max_nfev: int = 500,
    update_phases: bool = True,
    geometry_only: bool = True,
    verbose: bool = False,
) -> MixedFitResult:
    """
    Refine orientations of a multi-phase Laue pattern.

    Supports two coupling modes controlled by ``shared``:

    - **Shared** (default): one global rotation for all phases (3 free
      parameters).  Use this when the phases are co-oriented (e.g. an
      epitaxial stack measured as a mixed signal) and you want to preserve
      their relative orientations.
    - **Per-phase**: independent rotation per phase (3 × N_phases free
      parameters).  Use this when phases may have drifted independently or
      when the initial guess for each phase was set separately.

    Parameters
    ----------
    phases         : list of dicts or tuples
                     Phase descriptors in the same format as
                     :func:`~nrxrdct.laue.simulation.simulate_mixed_phases`.
                     Each entry must provide ``'crystal'``, ``'U'``, and
                     ``'volume_fraction'``.
    camera         : Camera           Detector geometry.
    obs_xy         : (N_obs, 2)       Observed pixel positions ``[xcam, ycam]``.
    shared         : bool             ``True`` → single global rotation (3 DOF).
                                      ``False`` → independent rotation per phase
                                      (3 × N_phases DOF).
    E_min_eV       : float            Low-energy cut-off (eV).
    E_max_eV       : float            High-energy cut-off (eV).
    source         : str              Spectral model (``'bending_magnet'`` or
                                      ``'undulator'``).
    source_kwargs  : dict or None     Extra kwargs for the spectral function.
    hmax           : int              Maximum Miller index.
    f2_thresh      : float or None    Minimum |F|² threshold.
    kb_params      :                  KB mirror reflectivity parameters.
    structure_model: str              Layer combination model (``'average'`` or
                                      ``'incoherent'``).
    max_match_px   : float            Pixel match radius.
    top_n_obs      : int or None      Brightest N observed spots to use.
    top_n_sim      : int or None      Brightest N simulated spots to consider.
    method         : str              ``least_squares`` algorithm (``'lm'`` or
                                      ``'trf'``).
    ftol, xtol, gtol : float          Convergence tolerances.
    max_nfev       : int              Maximum residual evaluations.
    update_phases  : bool             If ``True`` (default), write the refined
                                      U matrices back into the original phase
                                      dicts (dict input only; tuple input is
                                      not mutated).
    verbose        : bool             Print a one-line summary after convergence.

    Returns
    -------
    MixedFitResult
        ``result.U_phases[i]`` is the refined orientation of phase *i*.
        ``result.rotvecs[i]``  is the corresponding rotation vector (all
        identical in shared mode).
    """
    phases_work = _normalise_phases(phases)
    N_phases    = len(phases_work)
    U0_list     = [np.asarray(p["U"], dtype=float) for p in phases_work]

    obs_use = np.asarray(obs_xy, dtype=float)
    if top_n_obs is not None:
        obs_use = obs_use[:top_n_obs]
    N_obs = len(obs_use)

    n_params = 3 if shared else 3 * N_phases

    if verbose:
        mode = "shared" if shared else f"per-phase ({N_phases} phases)"
        print(
            f"fit_orientation_mixed: {N_obs} observed spots, "
            f"{N_phases} phases, {mode}"
        )

    # Precompute per-crystal allowed hkl sets once; keyed by id(crystal) so
    # simulate_mixed_phases can look up the right set for each phase.
    if geometry_only:
        _f2 = f2_thresh if f2_thresh is not None else F2_THRESHOLD
        _allowed: dict | None = {
            id(p["crystal"]): precompute_allowed_hkl(p["crystal"], hmax, f2_thresh=_f2)
            for p in phases_work
        }
    else:
        _allowed = None

    fun = partial(
        laue_mixed_residuals,
        phases=phases_work, camera=camera, obs_xy=obs_use,
        U0_list=U0_list, shared=shared,
        E_min_eV=E_min_eV, E_max_eV=E_max_eV,
        source=source, source_kwargs=source_kwargs,
        hmax=hmax, f2_thresh=f2_thresh, kb_params=kb_params,
        structure_model=structure_model,
        max_match_px=max_match_px, top_n_obs=None, top_n_sim=top_n_sim,
        geometry_only=False, allowed_hkl=_allowed,
    )

    try:
        opt = least_squares(
            fun, x0=np.zeros(n_params),
            method=method, ftol=ftol, xtol=xtol, gtol=gtol, max_nfev=max_nfev,
        )
    finally:
        # Restore original U matrices in the working copy.
        for p, U0 in zip(phases_work, U0_list):
            p["U"] = U0.copy()

    # Unpack refined orientations.
    params = opt.x
    if shared:
        R = Rotation.from_rotvec(params).as_matrix()
        U_phases_final = [R @ U0 for U0 in U0_list]
        rotvecs = [params.copy()] * N_phases
    else:
        U_phases_final = []
        rotvecs        = []
        for i, U0 in enumerate(U0_list):
            rv = params[3 * i : 3 * i + 3]
            U_phases_final.append(Rotation.from_rotvec(rv).as_matrix() @ U0)
            rotvecs.append(rv.copy())

    if update_phases:
        for p_orig, p_work, U_new in zip(
            phases if isinstance(phases[0], dict) else [None] * N_phases,
            phases_work,
            U_phases_final,
        ):
            p_work["U"] = U_new.copy()
            if isinstance(p_orig, dict):
                p_orig["U"] = U_new.copy()

    n_matched, rms_px = _compute_match_stats(opt.fun, max_match_px, N_obs)

    # Final simulation for n_sim.
    for p, U_new in zip(phases_work, U_phases_final):
        p["U"] = U_new.copy()
    final_spots = simulate_mixed_phases(
        phases_work, camera,
        E_min_eV=E_min_eV, E_max_eV=E_max_eV,
        source=source, source_kwargs=source_kwargs,
        hmax=hmax, f2_thresh=f2_thresh, kb_params=kb_params,
        structure_model=structure_model, verbose=False,
        allowed_hkl=_allowed,
    )
    n_sim = len(_extract_sim_xy(final_spots))

    result = MixedFitResult(
        U_phases=U_phases_final, U0_phases=U0_list,
        rotvecs=rotvecs,
        cost=float(opt.cost), rms_px=rms_px,
        n_matched=n_matched, n_obs=N_obs, n_sim=n_sim,
        match_rate=n_matched / max(N_obs, 1),
        success=opt.success, message=opt.message, optimizer=opt,
    )

    if verbose:
        print(f"  {result}")

    return result


def fit_strain_orientation(
    crystal,
    camera,
    obs_xy: np.ndarray,
    U0: np.ndarray,
    fit_strain: tuple[str, ...] = _STRAIN_ALL,
    strain_scale: float = 1e-4,
    E_min_eV: float = E_MIN_eV,
    E_max_eV: float = E_MAX_eV,
    source: str = "bending_magnet",
    source_kwargs: dict | None = None,
    hmax: int = HMAX,
    f2_thresh: float = F2_THRESHOLD,
    kb_params=BM32_KB,
    max_match_px: float = 30.0,
    top_n_obs: int | None = None,
    top_n_sim: int | None = None,
    method: str = "lm",
    ftol: float = 1e-8,
    xtol: float = 1e-8,
    gtol: float = 1e-8,
    max_nfev: int = 2000,
    geometry_only: bool = True,
    verbose: bool = False,
) -> StrainFitResult:
    """
    Simultaneously refine orientation and lattice strain for a single crystal.

    Wraps :func:`laue_strain_residuals` + ``scipy.optimize.least_squares``.

    The effective orientation passed to the simulator is::

        U_eff = R(δω) @ U0 @ (I + ε)

    where δω is a small rotation increment and ε is the symmetric strain
    tensor.  Because ``simulate_laue`` accepts any 3×3 matrix, the strained
    d-spacings are naturally encoded in ``U_eff``.

    Parameters
    ----------
    crystal      : Crystal        xrayutilities crystal structure.
    camera       : Camera         Detector geometry.
    obs_xy       : (N_obs, 2)     Observed pixel positions ``[xcam, ycam]``,
                                  sorted by descending intensity.
    U0           : (3, 3)         Starting orientation matrix (LT frame).
    fit_strain   : tuple of str   Strain components to refine.  Any subset of
                                  ``('e_xx','e_yy','e_zz','e_xy','e_xz','e_yz')``.
                                  Default: all six.  Pass a subset to fix
                                  symmetry constraints, e.g.
                                  ``('e_xx', 'e_yy', 'e_zz')`` for diagonal
                                  (biaxial) strain only.
    strain_scale : float          Internal scale for strain parameters (see
                                  :func:`laue_strain_residuals`).  Default 1e-4.
    E_min_eV, E_max_eV, source, source_kwargs, hmax, f2_thresh, kb_params,
    max_match_px, top_n_obs, top_n_sim, method, ftol, xtol, gtol, max_nfev,
    geometry_only
                   Forwarded to ``least_squares`` / :func:`laue_strain_residuals`.
    verbose      : bool           Print a one-line summary after convergence.

    Returns
    -------
    StrainFitResult
        Call ``str(result)`` for a compact summary.  The refined effective
        matrix is ``result.U_eff``; the pure rotation part is ``result.U``.

    Notes
    -----
    *Scaling*: strain values for metals are typically 10⁻⁴–10⁻³.
    ``strain_scale=1e-4`` keeps all optimizer parameters near order-1,
    which is important for Levenberg–Marquardt whose finite-difference step
    is proportional to parameter magnitude.

    *Starting point*: it is usually best to first obtain a good orientation
    with :func:`fit_orientation` and then pass the result as ``U0`` here.
    """
    U0_arr = np.asarray(U0, dtype=float)
    obs_use = np.asarray(obs_xy, dtype=float)
    if top_n_obs is not None:
        obs_use = obs_use[:top_n_obs]
    N_obs = len(obs_use)
    n_strain = len(fit_strain)

    if verbose:
        print(
            f"fit_strain_orientation: {N_obs} observed spots, "
            f"strain components: {list(fit_strain)}"
        )

    _allowed = (
        precompute_allowed_hkl(crystal, hmax, f2_thresh=f2_thresh)
        if geometry_only else None
    )

    fun = partial(
        laue_strain_residuals,
        crystal=crystal, camera=camera, obs_xy=obs_use, U0=U0_arr,
        fit_strain=fit_strain, strain_scale=strain_scale,
        E_min_eV=E_min_eV, E_max_eV=E_max_eV,
        source=source, source_kwargs=source_kwargs,
        hmax=hmax, f2_thresh=f2_thresh, kb_params=kb_params,
        max_match_px=max_match_px, top_n_obs=None, top_n_sim=top_n_sim,
        geometry_only=False, allowed_hkl=_allowed,
    )

    x0 = np.zeros(3 + n_strain)
    opt = least_squares(
        fun, x0=x0,
        method=method, ftol=ftol, xtol=xtol, gtol=gtol, max_nfev=max_nfev,
    )

    # Unpack solution.
    rotvec = opt.x[:3]
    strain_vals = opt.x[3:] * strain_scale
    R = Rotation.from_rotvec(rotvec).as_matrix()
    eps = _strain_matrix(strain_vals, fit_strain)
    U_final = R @ U0_arr
    U_eff = R @ U0_arr @ (np.eye(3) + eps)
    voigt = _strain_to_voigt(strain_vals, fit_strain)

    n_matched, rms_px = _compute_match_stats(opt.fun, max_match_px, N_obs)

    final_spots = simulate_laue(
        crystal, U_eff, camera,
        E_min=E_min_eV, E_max=E_max_eV,
        source=source, source_kwargs=source_kwargs,
        hmax=hmax, f2_thresh=f2_thresh, kb_params=kb_params,
        allowed_hkl=_allowed,
    )
    n_sim = len(_extract_sim_xy(final_spots))

    result = StrainFitResult(
        U=U_final, U0=U0_arr, U_eff=U_eff,
        rotvec=rotvec, strain_tensor=eps, strain_voigt=voigt,
        fit_strain=fit_strain,
        cost=float(opt.cost), rms_px=rms_px,
        n_matched=n_matched, n_obs=N_obs, n_sim=n_sim,
        match_rate=n_matched / max(N_obs, 1),
        success=opt.success, message=opt.message, optimizer=opt,
    )

    if verbose:
        print(f"  {result}")

    return result
