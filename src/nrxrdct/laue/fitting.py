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
    simulate_laue,
    simulate_laue_stack,
    simulate_mixed_phases,
)

# ─────────────────────────────────────────────────────────────────────────────
# Result containers
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class OrientationFitResult:
    """Result of a single-crystal orientation refinement."""

    U          : np.ndarray             # refined (3, 3) orientation matrix
    U0         : np.ndarray             # starting (3, 3)
    rotvec     : np.ndarray             # rotation vector from U0 (rad)
    cost       : float                  # ½ Σ residuals²  (from least_squares)
    rms_px     : float                  # RMS pixel error of matched spots
    n_matched  : int                    # spots within max_match_px
    n_obs      : int                    # observed spots used
    n_sim      : int                    # simulated spots on detector at solution
    match_rate : float                  # n_matched / n_obs
    success    : bool
    message    : str
    optimizer  : object = field(repr=False)   # raw scipy OptimizeResult

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
    """Result of a layered-crystal orientation refinement (single global rotation)."""

    R_global   : np.ndarray             # (3, 3) global rotation applied to all layers
    rotvec     : np.ndarray             # rotation vector (rad)
    U_layers   : list[np.ndarray]       # refined U per layer (stack.all_layers order)
    U0_layers  : list[np.ndarray]       # starting U per layer
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
    """Result of a multi-phase orientation refinement."""

    U_phases   : list[np.ndarray]       # refined U per phase (input order)
    U0_phases  : list[np.ndarray]       # starting U per phase
    rotvecs    : list[np.ndarray]       # rotation vector per phase (each (3,))
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


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────


def _extract_sim_xy(spots: list) -> np.ndarray:
    """Return (N_sim, 2) array of [xcam, ycam] from a spot list."""
    xy = [s["pix"] for s in spots if s.get("pix") is not None]
    return np.array(xy, dtype=float) if xy else np.empty((0, 2), dtype=float)


def _match_spots(
    obs_xy: np.ndarray,
    sim_xy: np.ndarray,
    max_match_px: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Optimal bipartite matching via the Hungarian algorithm.

    Returns (row_ind, col_ind, dist_px) — indices into obs_xy / sim_xy
    and the pixel distance of each accepted pair.
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
    Fixed-length residual vector from matched positions.

    Length = 2 * N_obs.  Unmatched observed spots contribute
    ``(max_match_px, max_match_px)``.
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
    Count matched spots and compute RMS pixel error from a residual vector.

    Unmatched spots were filled with exactly ``max_match_px`` in both
    components; they are identified by ``|Δx| >= max_match_px - ε AND
    |Δy| >= max_match_px - ε``.
    """
    r = residuals.reshape(N_obs, 2)
    unmatched = np.all(np.abs(r) >= max_match_px - 1e-9, axis=1)
    matched   = ~unmatched
    n_matched = int(matched.sum())
    rms_px    = float(np.sqrt((r[matched] ** 2).mean())) if n_matched > 0 else float("nan")
    return n_matched, rms_px


def _normalise_phases(phases: list) -> list[dict]:
    """Convert tuple-format phases to dict format (in-place copy)."""
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
) -> np.ndarray:
    """
    Pixel-space residual vector for single-crystal orientation refinement.

    Parameters
    ----------
    rotvec  : (3,)      rotation-vector increment δω (rad); current
                        orientation is ``Rotation.from_rotvec(δω) @ U0``.
    crystal : Crystal   xrayutilities crystal structure.
    camera  : Camera    detector geometry.
    obs_xy  : (N, 2)    observed pixel positions [xcam, ycam], sorted by
                        descending intensity (from
                        :func:`~nrxrdct.laue.segmentation.convert_spotsfile2peaklist`).
    U0      : (3, 3)    starting orientation matrix (LT frame, x // beam).
    max_match_px : float  pixel radius within which a pair is considered matched.
    top_n_obs : int     use only the N brightest observed spots (``None`` = all).
    top_n_sim : int     consider only the N brightest simulated spots.

    Returns
    -------
    residuals : (2 * N_obs_use,)  interleaved [Δx, Δy] per observed spot.
    """
    delta_R = Rotation.from_rotvec(np.asarray(rotvec, dtype=float)).as_matrix()
    U = delta_R @ np.asarray(U0, dtype=float)

    spots = simulate_laue(
        crystal, U, camera,
        E_min=E_min_eV, E_max=E_max_eV,
        source=source, source_kwargs=source_kwargs,
        hmax=hmax, f2_thresh=f2_thresh, kb_params=kb_params,
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
) -> np.ndarray:
    """
    Residual vector for a ``LayeredCrystal`` stack — single global rotation.

    A single rotation R = from_rotvec(δω) is applied to every layer:
    ``stack.all_layers[i].U = R @ U0_layers[i]``.  The stack is mutated
    in place during each optimizer call (single-threaded safe; the caller
    must restore the originals if needed — :func:`fit_orientation_stack`
    does this automatically).

    Parameters
    ----------
    rotvec    : (3,)            global rotation vector increment (rad).
    stack     : LayeredCrystal  will be mutated in place.
    U0_layers : list of (3,3)   base orientations in ``stack.all_layers`` order.
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
) -> np.ndarray:
    """
    Residual vector for a multi-phase Laue pattern.

    Parameters
    ----------
    params  : (3,) if ``shared=True`` else (3 * N_phases,)
              Rotation-vector increment(s).  In shared mode a single
              rotation is applied to every phase; in per-phase mode the
              first 3 elements rotate phase 0, the next 3 rotate phase 1,
              etc.
    phases  : list of dicts (will be mutated — U updated on each call).
              Each dict must have ``'crystal'``, ``'U'``, ``'volume_fraction'``.
    U0_list : list of (3, 3) base orientations, one per phase.
    shared  : bool  True → single global rotation; False → per-phase.
    """
    N = len(phases)
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

_LSQ_DEFAULTS = dict(method="lm", ftol=1e-6, xtol=1e-6, gtol=1e-8, max_nfev=500)


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
    verbose: bool = False,
) -> OrientationFitResult:
    """
    Refine the orientation matrix of a single crystal to match observed spots.

    Wraps :func:`laue_residuals` + ``scipy.optimize.least_squares``.

    Parameters
    ----------
    crystal  : xrayutilities Crystal
    camera   : Camera
    obs_xy   : (N_obs, 2)  measured pixel positions [xcam, ycam], sorted by
               descending intensity.  Pass ``peaklist[:, :2]`` directly from
               :func:`~nrxrdct.laue.segmentation.convert_spotsfile2peaklist`.
    U0       : (3, 3)  starting orientation (LT frame, x // beam).
    max_match_px : float  pixel tolerance for matching (start large, ~30–50,
               and tighten for final refinement).
    top_n_obs : int  use only the brightest N observed spots (default: all).
    top_n_sim : int  consider only the brightest N simulated spots.
    method   : str  least_squares method — ``'lm'`` (fast, no bounds) or
               ``'trf'`` (bounded, more robust for large δω).
    verbose  : bool  print progress.

    Returns
    -------
    OrientationFitResult
        Call ``str(result)`` for a one-line summary.
        The refined orientation is ``result.U``; apply it with::

            spots = simulate_laue(crystal, result.U, camera, ...)
    """
    U0 = np.asarray(U0, dtype=float)
    obs_use = np.asarray(obs_xy, dtype=float)
    if top_n_obs is not None:
        obs_use = obs_use[:top_n_obs]
    N_obs = len(obs_use)

    if verbose:
        print(f"fit_orientation: {N_obs} observed spots")

    fun = partial(
        laue_residuals,
        crystal=crystal, camera=camera, obs_xy=obs_use, U0=U0,
        E_min_eV=E_min_eV, E_max_eV=E_max_eV,
        source=source, source_kwargs=source_kwargs,
        hmax=hmax, f2_thresh=f2_thresh, kb_params=kb_params,
        max_match_px=max_match_px, top_n_obs=None, top_n_sim=top_n_sim,
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
    )
    n_sim = len(_extract_sim_xy(final_spots))

    result = OrientationFitResult(
        U=U_final, U0=U0.copy(), rotvec=opt.x.copy(),
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
    verbose: bool = False,
) -> StackFitResult:
    """
    Refine the orientation of a :class:`~nrxrdct.laue.layers.LayeredCrystal`.

    A single global rotation is applied to all layers simultaneously,
    preserving all inter-layer orientation relationships.

    Parameters
    ----------
    stack        : LayeredCrystal
    camera       : Camera
    obs_xy       : (N_obs, 2) observed pixel positions [xcam, ycam].
    update_stack : bool
        If ``True`` (default), write the refined U matrices back into
        ``stack.all_layers[i].U`` after convergence.  Set ``False`` to
        leave the stack unchanged and inspect the result first.

    Returns
    -------
    StackFitResult
        ``result.R_global`` is the 3×3 rotation applied to all layers.
        ``result.U_layers`` contains the refined U matrix for each layer
        in ``stack.all_layers`` order.
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

    fun = partial(
        laue_stack_residuals,
        stack=stack, camera=camera, obs_xy=obs_use,
        U0_layers=U0_layers,
        E_min_eV=E_min_eV, E_max_eV=E_max_eV,
        source=source, source_kwargs=source_kwargs,
        hmax=hmax, f2_thresh=f2_thresh, kb_params=kb_params,
        structure_model=structure_model,
        max_match_px=max_match_px, top_n_obs=None, top_n_sim=top_n_sim,
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
    verbose: bool = False,
) -> MixedFitResult:
    """
    Refine orientations of a multi-phase Laue pattern.

    Parameters
    ----------
    phases  : list of dicts or tuples — same format as :func:`simulate_mixed_phases`.
    camera  : Camera
    obs_xy  : (N_obs, 2) observed pixel positions [xcam, ycam].
    shared  : bool
        ``True``  (default) — one global rotation for all phases (3 free params).
        ``False`` — independent rotation per phase (3 × N_phases free params).
    update_phases : bool
        If ``True``, write refined U matrices back into the phase dicts.

    Returns
    -------
    MixedFitResult
        ``result.U_phases[i]`` is the refined orientation of phase *i*.
        ``result.rotvecs[i]``  is the corresponding rotation vector.
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

    fun = partial(
        laue_mixed_residuals,
        phases=phases_work, camera=camera, obs_xy=obs_use,
        U0_list=U0_list, shared=shared,
        E_min_eV=E_min_eV, E_max_eV=E_max_eV,
        source=source, source_kwargs=source_kwargs,
        hmax=hmax, f2_thresh=f2_thresh, kb_params=kb_params,
        structure_model=structure_model,
        max_match_px=max_match_px, top_n_obs=None, top_n_sim=top_n_sim,
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
