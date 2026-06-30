"""
Laue orientation-matrix fitting
================================
Refines orientation matrices by minimising pixel-space residuals between
simulated and observed Laue spot positions.

Three simulation back-ends are supported:

    fit_orientation        — single crystal  (:func:`simulate_laue`)
    fit_orientation_stack  — layered crystal (:func:`simulate_laue_stack`)
    fit_orientation_mixed  — multi-phase     (:func:`simulate_mixed_phases`)

**Parametrisation**
The free parameters are always rotation vectors δω (radians).  At each
optimizer iteration the current orientation of phase / layer *i* is:

    U_i = Rotation.from_rotvec(δω_i) @ U0_i

where U0_i is the starting estimate.  Keeping δω small (linearised update)
avoids gimbal lock and gives a well-conditioned Jacobian.

**Stack and mixed-phase fitting**
Both functions fit a *single shared* rotation by default (all
layers/phases rotate together, preserving their relative orientations).
Set `shared=False` in the mixed-phase case to optimise one independent
rotation vector per phase (3 × N_phases free parameters).

**Spot matching**
At each function evaluation the assignment between simulated and observed
spots is unknown.  It is solved with the Hungarian algorithm
(`scipy.optimize.linear_sum_assignment`) on the pixel-distance cost
matrix, capped at `max_match_px`.  Unmatched observed spots contribute
`(max_match_px, max_match_px)` to the residual vector — a soft wall
that steers the optimizer away from orientations that leave spots orphaned.

**Residual vector length**
All residual functions return a vector of length `2 * N_obs_use`,
regardless of how many simulated spots exist.  The fixed length makes
them directly compatible with `scipy.optimize.least_squares`.

**Staged refinement**
All three fitting functions accept ``max_match_px`` as either a single float
or a decreasing sequence (e.g. ``[50, 15, 3]``).  When a sequence is given,
each entry defines one stage.  At the end of stage *k* the optimizer
solution is composed into the warm-start orientation(s) before stage *k+1*
begins with a tighter match window:

- *Single crystal*: ``U0 ← R(δω_k) @ U0``
- *Stack*: ``U0_i ← R(δω_k) @ U0_i`` for every layer *i* (same global rotation)
- *Mixed, shared*: same global rotation applied to every phase
- *Mixed, per-phase*: each phase gets its own correction ``U0_i ← R(δω_k,i) @ U0_i``

Because δω is re-zeroed at the start of each stage, Levenberg–Marquardt
always begins from a well-conditioned, near-identity Jacobian.  A coarse
first stage (large ``max_match_px``) captures the correct basin even when
the initial misalignment is large; subsequent stages tighten the matching
tolerance to sharpen convergence.  The ``rotvec``/``R_global`` fields in
every result type represent the *total* rotation accumulated across all
stages, not just the last correction.
"""

from __future__ import annotations

import os
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from functools import partial

import dill as _dill
import numpy as np
import scipy.fft as _sp_fft
import scipy.ndimage as _ndi
from scipy.optimize import least_squares, linear_sum_assignment, minimize
from scipy.spatial.transform import Rotation

from .simulation import (
    BM32_KB,
    E_MAX_eV,
    E_MIN_eV,
    F2_THRESHOLD,
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

    Attributes:
        U ((3, 3) ndarray): Refined orientation matrix (LT frame).
        U0 ((3, 3) ndarray): Starting orientation passed to the fitter.
        rotvec ((3,) ndarray): Rotation vector δω (radians) such that
            `U = Rotation.from_rotvec(rotvec) @ U0`.
            Its magnitude is the total rotation angle.
        cost (float): ½ Σ residuals² as returned by `least_squares`.
        rms_px (float): RMS pixel distance of matched observed-simulated
            pairs (`nan` if no matches).
        mean_px (float): Mean Euclidean pixel distance of matched pairs
            (less sensitive to outliers than RMS;
            `nan` if no matches).
        n_matched (int): Number of observed spots matched within
            `max_match_px` at the solution.
        n_obs (int): Number of observed spots used in the fit.
        n_sim (int): Number of simulated spots on the detector at
            the solution (before any `top_n_sim` cut).
        match_rate (float): `n_matched / n_obs`.
        success (bool): `True` if the optimizer converged.
        message (str): Human-readable optimizer termination message.
        optimizer (OptimizeResult): Raw `scipy.optimize.OptimizeResult` (not shown
            in repr); inspect for Jacobian, gradient, etc.
"""

    U          : np.ndarray
    U0         : np.ndarray
    rotvec     : np.ndarray
    cost       : float
    rms_px     : float
    mean_px    : float
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
            f"rms={self.rms_px:.2f} px  mean={self.mean_px:.2f} px  "
            f"matched={self.n_matched}/{self.n_obs} ({self.match_rate:.0%})  "
            f"|δω|={dw:.4f}°"
        )


@dataclass
class StackFitResult:
    """
    Result of a layered-crystal orientation refinement (:func:`fit_orientation_stack`).

    A single global rotation is applied to all layers, preserving their
    relative orientation relationships.

    Attributes:
        R_global ((3, 3) ndarray): Global rotation matrix applied to every layer.
        rotvec ((3,) ndarray): Rotation vector (radians) for `R_global`.
        U_layers (list of (3, 3) arrays Refined U matrix for each layer, in): `stack.all_layers` order.
        U0_layers (list of (3, 3) arrays Starting U matrix for each layer.):
        cost (float): ½ Σ residuals² at convergence.
        rms_px (float): RMS pixel distance of matched spot pairs.
        n_matched (int): Matched spots within `max_match_px`.
        n_obs (int): Observed spots used.
        n_sim (int): Simulated spots on detector at solution.
        match_rate: float                 `n_matched / n_obs`.
        success (bool): Optimizer convergence flag.
        message (str): Optimizer termination message.
        optimizer (OptimizeResult): Raw `scipy.optimize.OptimizeResult`.
"""

    R_global   : np.ndarray
    rotvec     : np.ndarray
    U_layers   : list[np.ndarray]
    U0_layers  : list[np.ndarray]
    cost       : float
    rms_px     : float
    mean_px    : float
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
            f"rms={self.rms_px:.2f} px  mean={self.mean_px:.2f} px  "
            f"matched={self.n_matched}/{self.n_obs} ({self.match_rate:.0%})  "
            f"|δω|={dw:.4f}°"
        )


@dataclass
class StackStrainFitResult:
    """
    Result of a simultaneous orientation + per-layer strain refinement
    (:func:`fit_strain_orientation_stack`).

    Attributes:
        R_global ((3, 3) ndarray): Global rotation matrix applied to every layer.
        rotvec ((3,) ndarray): Rotation vector (radians) for `R_global`.
        U_layers (list of (3, 3)): Pure rotation part per layer: `R_global @ U0_i`.
        U0_layers (list of (3, 3)): Starting orientations.
        U_eff_layers (list of (3, 3)): Effective deformation matrices per layer:
            `R_global @ U0_i @ (I + ε_i)`.  Pass these as `U` to
            :func:`~nrxrdct.laue.simulation.simulate_laue` to reproduce the
            fitted spot pattern for each layer individually.
        strain_tensors (list of (3, 3)): Symmetric strain tensor per layer in
            the crystal frame.
        strain_voigts (list of (6,)): Voigt vector per layer
            `[ε_xx, ε_yy, ε_zz, ε_xy, ε_xz, ε_yz]`.  Components not in
            `fit_strain` are zero.
        fit_strain (tuple[str, …]): Strain components that were free parameters.
        cost (float): ½ Σ residuals² at convergence.
        rms_px (float): RMS pixel distance of matched spot pairs.
        mean_px (float): Mean pixel distance of matched spot pairs.
        n_matched (int): Matched spots within `max_match_px`.
        n_obs (int): Observed spots used.
        n_sim (int): Simulated spots on detector at solution.
        match_rate (float): `n_matched / n_obs`.
        success (bool): Optimizer convergence flag.
        message (str): Optimizer termination message.
        optimizer (OptimizeResult): Raw `scipy.optimize.OptimizeResult`.
"""

    R_global       : np.ndarray
    rotvec         : np.ndarray
    U_layers       : list[np.ndarray]
    U0_layers      : list[np.ndarray]
    U_eff_layers   : list[np.ndarray]
    strain_tensors : list[np.ndarray]
    strain_voigts  : list[np.ndarray]
    fit_strain     : tuple
    cost           : float
    rms_px         : float
    mean_px        : float
    n_matched      : int
    n_obs          : int
    n_sim          : int
    match_rate     : float
    success        : bool
    message        : str
    optimizer      : object = field(repr=False)

    def __str__(self) -> str:
        status = "OK" if self.success else "FAILED"
        dw = float(np.degrees(np.linalg.norm(self.rotvec)))
        return (
            f"StackStrainFitResult [{status}]  "
            f"rms={self.rms_px:.2f} px  mean={self.mean_px:.2f} px  "
            f"matched={self.n_matched}/{self.n_obs} ({self.match_rate:.0%})  "
            f"|δω|={dw:.4f}°"
        )


@dataclass
class MixedFitResult:
    """
    Result of a multi-phase orientation refinement (:func:`fit_orientation_mixed`).

    Attributes:
        U_phases (list of (3, 3) arrays Refined orientation matrix per phase,): in input order.
        U0_phases (list of (3, 3) arrays Starting orientation matrix per phase.):
        rotvecs (list of (3,) arrays): Rotation vector per phase.  In shared
            mode every entry is identical; in per-phase
            mode each entry is independent.
        cost (float): ½ Σ residuals² at convergence.
        rms_px (float): RMS pixel distance of matched spot pairs.
        n_matched (int): Matched spots within `max_match_px`.
        n_obs (int): Observed spots used.
        n_sim (int): Total simulated spots on detector at solution
            (all phases combined).
        match_rate: float                 `n_matched / n_obs`.
        success (bool): Optimizer convergence flag.
        message (str): Optimizer termination message.
        optimizer (OptimizeResult): Raw `scipy.optimize.OptimizeResult`.
"""

    U_phases   : list[np.ndarray]
    U0_phases  : list[np.ndarray]
    rotvecs    : list[np.ndarray]
    cost       : float
    rms_px     : float
    mean_px    : float
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

    Attributes:
        U ((3, 3) ndarray): Pure rotation part of the refined matrix.
            `U = Rotation.from_rotvec(rotvec) @ U0`.
        U0 ((3, 3) ndarray): Starting orientation passed to the fitter.
        U_eff ((3, 3) ndarray): Full deformation matrix used by the
            simulator: `U @ (I + strain_tensor)`.
            Pass this as `U` to
            :func:`~nrxrdct.laue.simulate_laue` to
            reproduce the fitted spot pattern.
        rotvec ((3,) ndarray): Rotation increment δω (radians).
        strain_tensor ((3, 3) ndarray): Symmetric strain tensor in the crystal
            frame.  Diagonal entries are axial strains
            (Δa/a, Δb/b, Δc/c); off-diagonal entries
            are the engineering shear strains / 2.
        strain_voigt ((6,) ndarray): Voigt representation
            `[ε_xx, ε_yy, ε_zz, ε_xy, ε_xz, ε_yz]`
            in the crystal frame.  Components not
            listed in `fit_strain` are zero.
        strain_tensor_deviatoric ((3, 3) ndarray): Deviatoric part of
            `strain_tensor` in the crystal frame:
            ``ε − (Tr(ε)/3) I`` (computed property).
        strain_tensor_lab ((3,3) ndarray): `strain_tensor` rotated to the
            lab frame via `U @ ε @ Uᵀ`
            (computed property).
        strain_voigt_lab ((6,) ndarray): Voigt form of `strain_tensor_lab`
            (computed property).
        fit_strain (tuple[str, …]): Strain components that were free parameters.
        cost (float): ½ Σ residuals² at convergence.
        rms_px (float): RMS pixel distance of matched pairs.
        mean_px (float): Mean Euclidean pixel distance of matched pairs.
        n_matched (int): Matched spots within `max_match_px`.
        n_obs (int): Observed spots used.
        n_sim (int): Simulated spots on detector at solution.
        match_rate (float): `n_matched / n_obs`.
        success (bool): Optimizer convergence flag.
        message (str): Optimizer termination message.
        optimizer (OptimizeResult): Raw `scipy.optimize.OptimizeResult`
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
    mean_px       : float
    n_matched     : int
    n_obs         : int
    n_sim         : int
    match_rate    : float
    success       : bool
    message       : str
    optimizer     : object = field(repr=False)

    @property
    def strain_tensor_deviatoric(self) -> np.ndarray:
        """
        Deviatoric part of the strain tensor in the crystal frame:
        ``ε_dev = ε − (Tr(ε)/3) I``.

        Returns:
            (3, 3) ndarray
"""
        eps = self.strain_tensor
        return eps - np.trace(eps) / 3.0 * np.eye(3)

    @property
    def strain_tensor_lab(self) -> np.ndarray:
        """
        Strain tensor rotated into the laboratory frame.

        The stored `strain_tensor` is expressed in the crystal Cartesian
        frame (right-hand side of U0 in `U_eff = R @ U0 @ (I + ε)`).
        This property applies the similarity transform

            ε_lab = U @ ε_crystal @ Uᵀ

        where `U = R @ U0` is the pure rotation part, yielding the same
        physical deformation expressed in the lab Cartesian axes
        (x ∥ beam, z vertical).

        Returns:
            (3, 3) ndarray
"""
        return self.U @ self.strain_tensor @ self.U.T

    @property
    def strain_voigt_lab(self) -> np.ndarray:
        """
        Voigt representation of `strain_tensor_lab`:
        `[ε_xx, ε_yy, ε_zz, ε_xy, ε_xz, ε_yz]` in the lab frame.

        Returns:
            (6,) ndarray
"""
        e = self.strain_tensor_lab
        return np.array([e[0, 0], e[1, 1], e[2, 2], e[0, 1], e[0, 2], e[1, 2]])

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


@dataclass
class IndexResult:
    """
    Result of Laue autoindexing (:func:`index_orientation`).

    Attributes:
        U ((3,3) ndarray): Best candidate orientation matrix.
        n_matched (int): Observed spots matching within
            `angle_tol_deg` at the returned orientation.
        n_obs (int): Number of observed spots used.
        match_rate (float): `n_matched / n_obs`.
        hkl_pair (tuple): `((h₁,k₁,l₁), (h₂,k₂,l₂))` seed
            reflection pair that produced the best
            candidate.
        angle_deg (float): Inter-spot angle of the seed pair (degrees).
        n_candidates (int): Total candidate matrices evaluated.
        success (bool): `match_rate >= min_match_rate`.
"""

    U            : np.ndarray
    n_matched    : int
    n_obs        : int
    match_rate   : float
    hkl_pair     : tuple
    angle_deg    : float
    n_candidates : int
    success      : bool

    def save_U(self, path: str) -> None:
        """
        Save the orientation matrix to a `.npy` file.

        Args:
            path (str): Output path.  If the filename has no extension, `.npy` is
                appended automatically by :func:`numpy.save`.

        Example:
        >>> idx = index_orientation(crystal, camera, obs_xy)
        >>> idx.save_U("UB0.npy")   # ready for submit_orientation
"""
        import os
        np.save(path, self.U)
        print(f"IndexResult.save_U → {os.path.abspath(path)}")

    def __str__(self) -> str:
        status = "OK" if self.success else "FAILED"
        h1, h2 = self.hkl_pair
        return (
            f"IndexResult [{status}]  "
            f"matched={self.n_matched}/{self.n_obs} ({self.match_rate:.0%})  "
            f"seed={h1}/{h2}  angle={self.angle_deg:.2f}°"
        )


@dataclass
class StrainImageRefinementResult:
    """
    Result of image-based strain + orientation post-refinement
    (:func:`refine_strain_image`).

    Attributes:
        U ((3,3) ndarray): Refined rotation matrix (pure rotation part).
        U0 ((3,3) ndarray): Starting orientation.
        U_eff ((3,3) ndarray): Full deformation matrix used in simulation:
            ``U_eff = R(rotvec) @ U0 @ (I + ε)``.
        rotvec ((3,) ndarray): Rotation correction applied to U0 (radians).
        strain_tensor ((3,3) ndarray): Symmetric strain tensor ε in the
            crystal frame (only free components are non-zero).
        strain_voigt ((6,) ndarray): ``[ε_xx, ε_yy, ε_zz, ε_xy, ε_xz, ε_yz]``.
        fit_strain (tuple of str): Strain components that were free.
        score (float): Total Gaussian-weighted intensity at the refined
            spot positions (higher is better).
        score0 (float): Score at the starting orientation/strain.
        n_sim (int): Number of simulated spots at the refined solution.
        success (bool): Optimizer convergence flag.
        message (str): Optimizer termination message.
        optimizer (OptimizeResult): Raw result (not shown in repr).
"""

    U             : np.ndarray
    U0            : np.ndarray
    U_eff         : np.ndarray
    rotvec        : np.ndarray
    strain_tensor : np.ndarray
    strain_voigt  : np.ndarray
    fit_strain    : tuple
    score         : float
    score0        : float
    n_sim         : int
    success       : bool
    message       : str
    optimizer     : object = field(repr=False)

    @property
    def strain_tensor_deviatoric(self) -> np.ndarray:
        """Deviatoric part: ``ε_dev = ε − (Tr(ε)/3) I``."""
        eps = self.strain_tensor
        return eps - np.trace(eps) / 3.0 * np.eye(3)

    @property
    def strain_tensor_lab(self) -> np.ndarray:
        """Strain tensor rotated to the lab frame: ``ε_lab = U @ ε @ Uᵀ``."""
        return self.U @ self.strain_tensor @ self.U.T

    @property
    def strain_voigt_lab(self) -> np.ndarray:
        """Voigt representation of ``strain_tensor_lab``."""
        e = self.strain_tensor_lab
        return np.array([e[0, 0], e[1, 1], e[2, 2], e[0, 1], e[0, 2], e[1, 2]])

    def __str__(self) -> str:
        status = "OK" if self.success else "FAILED"
        dw   = float(np.degrees(np.linalg.norm(self.rotvec)))
        e    = self.strain_voigt
        gain = self.score - self.score0
        return (
            f"StrainImageRefinementResult [{status}]  "
            f"|δω|={dw:.4f}°  "
            f"score={self.score:.1f}  Δscore={gain:+.1f}  "
            f"n_sim={self.n_sim}  "
            f"ε_diag=[{e[0]:.2e}, {e[1]:.2e}, {e[2]:.2e}]"
        )


@dataclass
class StackStrainImageRefinementResult:
    """
    Result of image-based strain + orientation refinement for a layered crystal
    (:func:`refine_strain_image_stack`).

    Attributes:
        R_global ((3,3) ndarray): Global rotation matrix applied to all layers.
        rotvec ((3,) ndarray): Rotation vector for `R_global` (radians).
        U_layers (list of (3,3)): Pure rotation part per layer:
            `R_global @ U0_i`.
        U0_layers (list of (3,3)): Starting orientations.
        U_eff_layers (list of (3,3)): Effective deformation matrices per layer:
            `R_global @ U0_i @ (I + ε_i)`.  Pass each as `U` to
            :func:`~nrxrdct.laue.simulation.simulate_laue` to reproduce the
            individual layer's spot pattern.
        strain_tensors (list of (3,3)): Per-layer symmetric strain tensor.
        strain_voigts (list of (6,)): Per-layer Voigt vector
            `[ε_xx, ε_yy, ε_zz, ε_xy, ε_xz, ε_yz]`.
        fit_strain (tuple of str): Strain components that were free parameters.
        score (float): Gaussian-weighted pixel score at the refined solution
            (higher is better).
        score0 (float): Score at the starting orientation/strain.
        n_sim (int): Total simulated spots on the detector at the solution.
        success (bool): Optimizer convergence flag.
        message (str): Optimizer termination message.
        optimizer: Raw ``scipy.optimize.OptimizeResult`` (not shown in repr).
"""

    R_global       : np.ndarray
    rotvec         : np.ndarray
    U_layers       : list[np.ndarray]
    U0_layers      : list[np.ndarray]
    U_eff_layers   : list[np.ndarray]
    strain_tensors : list[np.ndarray]
    strain_voigts  : list[np.ndarray]
    fit_strain     : tuple
    score          : float
    score0         : float
    n_sim          : int
    success        : bool
    message        : str
    optimizer      : object = field(repr=False)

    def __str__(self) -> str:
        status = "OK" if self.success else "FAILED"
        dw   = float(np.degrees(np.linalg.norm(self.rotvec)))
        gain = self.score - self.score0
        return (
            f"StackStrainImageRefinementResult [{status}]  "
            f"|δω|={dw:.4f}°  "
            f"score={self.score:.1f}  Δscore={gain:+.1f}  "
            f"n_sim={self.n_sim}"
        )


@dataclass
class ImageRefinementResult:
    """
    Result of image-based orientation post-refinement
    (:func:`refine_orientation_image`).

    Attributes:
        U ((3,3) ndarray): Refined orientation matrix.
        U0 ((3,3) ndarray): Starting orientation.
        rotvec ((3,) ndarray): Rotation vector δω (radians) applied to U0,
            i.e. ``U = Rotation.from_rotvec(rotvec) @ U0``.
        score (float): Total Gaussian-weighted intensity at the refined
            spot positions (higher is better).
        score0 (float): Score at the starting orientation U0, for
            comparison.
        n_sim (int): Number of simulated spots on the detector at
            the refined orientation.
        success (bool): Optimizer convergence flag.
        message (str): Optimizer termination message.
        optimizer (OptimizeResult): Raw ``scipy.optimize.OptimizeResult``
            (not shown in repr).
"""

    U         : np.ndarray
    U0        : np.ndarray
    rotvec    : np.ndarray
    score     : float
    score0    : float
    n_sim     : int
    success   : bool
    message   : str
    optimizer : object = field(repr=False)

    def __str__(self) -> str:
        status = "OK" if self.success else "FAILED"
        dw = float(np.degrees(np.linalg.norm(self.rotvec)))
        gain = self.score - self.score0
        return (
            f"ImageRefinementResult [{status}]  "
            f"|δω|={dw:.4f}°  "
            f"score={self.score:.1f}  Δscore={gain:+.1f}  "
            f"n_sim={self.n_sim}"
        )


@dataclass
class StackImageRefinementResult:
    """
    Result of image-based orientation refinement for a layered crystal
    (:func:`refine_orientation_image_stack`).

    Attributes:
        R_global ((3,3) ndarray): Global rotation matrix applied to all layers.
        rotvec ((3,) ndarray): Rotation vector for `R_global` (radians).
        U_layers (list of (3,3)): Refined orientation per layer:
            `R_global @ U0_i`.
        U0_layers (list of (3,3)): Starting orientations.
        score (float): Gaussian-weighted pixel score at the refined solution.
        score0 (float): Score at the starting orientations.
        n_sim (int): Total simulated spots on the detector at the solution.
        success (bool): Optimizer convergence flag.
        message (str): Optimizer termination message.
        optimizer: Raw ``scipy.optimize.OptimizeResult`` (not shown in repr).
"""

    R_global  : np.ndarray
    rotvec    : np.ndarray
    U_layers  : list[np.ndarray]
    U0_layers : list[np.ndarray]
    score     : float
    score0    : float
    n_sim     : int
    success   : bool
    message   : str
    optimizer : object = field(repr=False)

    def __str__(self) -> str:
        status = "OK" if self.success else "FAILED"
        dw   = float(np.degrees(np.linalg.norm(self.rotvec)))
        gain = self.score - self.score0
        return (
            f"StackImageRefinementResult [{status}]  "
            f"|δω|={dw:.4f}°  "
            f"score={self.score:.1f}  Δscore={gain:+.1f}  "
            f"n_sim={self.n_sim}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────


def _extract_sim_xy(spots: list) -> np.ndarray:
    """
    Extract pixel positions from a simulate_laue spot list.

    Args:
        spots (list of dicts): Output of any `simulate_laue*` function.  Each dict is expected
            to contain a `'pix'` key with a `[xcam, ycam]` value; spots
            that reach the detector have a non-None `'pix'`.

    Returns:
        xy ((N_sim, 2) ndarray): Pixel positions `[xcam, ycam]` for all on-detector spots.
            Returns `(0, 2)` if the spot list is empty or no spot is on the
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

    Uses the Hungarian algorithm (`scipy.optimize.linear_sum_assignment`)
    on a pixel-distance cost matrix capped at `max_match_px`.  Capping
    means the optimizer treats any pair farther than the cap as equally bad,
    preventing a single large outlier from dominating the assignment.

    Args:
        obs_xy ((N_obs, 2) ndarray): Observed pixel positions [xcam, ycam].
        sim_xy ((N_sim, 2) ndarray): Simulated pixel positions [xcam, ycam].
        max_match_px (float): Distance cap applied before solving the
            assignment problem.

    Returns:
        row_ind ((K,) int array): Indices into `obs_xy` for accepted pairs.
        col_ind ((K,) int array): Corresponding indices into `sim_xy`.
        dist_px ((K,) float array Euclidean pixel distance for each pair.): Note: pairs where the true distance exceeds
            `max_match_px` are included — callers must
            filter on `dist_px` themselves.
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
    Build the fixed-length residual vector used by `least_squares`.

    For each observed spot the nearest simulated spot is found via
    :func:`_match_spots`.  If the assigned pair is within `max_match_px`
    the residual components are the signed pixel differences (Δx, Δy).
    Otherwise both components are set to `max_match_px`, acting as a
    soft penalty wall that steers the optimizer toward orientations where
    all spots are explained.

    Args:
        obs_use ((N_obs, 2)): Observed pixel positions [xcam, ycam].
        sim_xy ((N_sim, 2)): Simulated pixel positions, or empty array.
        max_match_px (float): Penalty value assigned to unmatched spots.

    Returns:
        residuals ((2 * N_obs,) ndarray): Interleaved [Δx₀, Δy₀, Δx₁, Δy₁, ...].  Length is always
            `2 * N_obs` regardless of how many simulated spots exist,
            making the vector directly usable with `least_squares`.
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

    Unmatched spots were filled with exactly `max_match_px` in both Δx
    and Δy by :func:`_build_residuals`.  They are detected by checking
    `|Δx| ≥ max_match_px − ε  AND  |Δy| ≥ max_match_px − ε`.

    Args:
        residuals ((2 * N_obs,) ndarray): Residual vector from :func:`_build_residuals`.
        max_match_px (float): Penalty threshold used when building residuals.
        N_obs (int): Number of observed spots (= len(residuals) // 2).

    Returns:
        n_matched (int): Number of spots with a matched simulated counterpart.
        rms_px (float): RMS Euclidean pixel distance of matched pairs only.
            Returns `nan` when no spots are matched.
        mean_px (float): Mean Euclidean pixel distance of matched pairs only.
            Less sensitive to outliers than RMS.
            Returns `nan` when no spots are matched.
"""
    r = residuals.reshape(N_obs, 2)
    unmatched = np.all(np.abs(r) >= max_match_px - 1e-9, axis=1)
    matched   = ~unmatched
    n_matched = int(matched.sum())
    if n_matched > 0:
        dists   = np.linalg.norm(r[matched], axis=1)
        rms_px  = float(np.sqrt((r[matched] ** 2).mean()))
        mean_px = float(dists.mean())
    else:
        rms_px  = float("nan")
        mean_px = float("nan")
    return n_matched, rms_px, mean_px


def _normalise_phases(phases: list) -> list[dict]:
    """
    Convert phases to a uniform list-of-dicts representation.

    :func:`simulate_mixed_phases` accepts phases as either dicts or
    `(crystal, U, volume_fraction[, label])` tuples.  This helper
    normalises both forms into dicts so the fitting code can always
    mutate `p["U"]` without special-casing.

    Args:
        phases (list of dict or tuple): Input phase list in either accepted format.

    Returns:
        out (list of dict): New list of independent dicts (shallow-copies of input dicts;
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
# Spot attribution
# ─────────────────────────────────────────────────────────────────────────────


def remove_grain_spots(
    obs: np.ndarray,
    U: np.ndarray,
    crystal,
    camera,
    max_match_px: float = 5.0,
    f2_thresh: float = 1e-6,
    E_min_eV: float = E_MIN_eV,
    E_max_eV: float = E_MAX_eV,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Remove from *obs* the spots that are one-to-one matched to a grain.

    Uses the same Hungarian algorithm as the fitter so that the attribution
    is identical to what `fit_orientation` does internally.  Only spots
    that are uniquely assigned to a simulated reflection **and** within
    *max_match_px* are removed; ambiguous or distant observed spots are kept.

    Typical use — iterative multi-grain peeling::

        remaining = peaks.copy()  # any (N, K) array; first two cols are x, y

        fit1 = laue.fit_orientation(crystal, cam, remaining[:, :2], U0_grain1)
        remaining, claimed1 = laue.remove_grain_spots(remaining, fit1.U,
                                                      crystal, cam)

        # now fit grain 2 starting from your own U guess
        fit2 = laue.fit_orientation(crystal, cam, remaining[:, :2], U0_grain2)

    Args:
        obs ((N, K) array-like): Observed spots. The first two columns must
            be pixel positions `[xcam, ycam]`; any
            additional columns (intensities, widths, …)
            are preserved unchanged in the output.
        U ((3, 3) array-like): Orientation matrix of the grain to remove.
        crystal (Crystal): xrayutilities crystal structure.
        camera (Camera): Detector geometry.
        max_match_px (float): Maximum pixel distance for a match.
            Should match the tolerance used in
            `fit_orientation`.  Default `5.0`.
        f2_thresh (float): Structure-factor threshold for the
            removal simulation.  Use a very small value
            (default `1e-6`) to generate essentially
            all allowed reflections and avoid leaving
            grain spots behind.
        E_min_eV, E_max_eV (float): Energy range forwarded to
            :func:`~nrxrdct.laue.simulate_laue`.

    Returns:
        remaining ((M, K) ndarray): Observed spots **not** claimed by this grain  (M ≤ N), with all
            original columns intact.
        claimed ((N,) bool ndarray): Boolean mask over *obs*: `True` where a spot was removed.
"""
    from .simulation import simulate_laue as _sim

    obs    = np.asarray(obs, dtype=float)
    obs_xy = obs[:, :2]

    spots  = _sim(
        crystal, U, camera,
        E_min=E_min_eV, E_max=E_max_eV,
        f2_thresh=f2_thresh,
        geometry_only=True,
    )
    sim_xy = _extract_sim_xy(spots)

    claimed = np.zeros(len(obs), dtype=bool)

    if len(sim_xy) > 0 and len(obs) > 0:
        diff    = obs_xy[:, None, :] - sim_xy[None, :, :]      # (N_obs, N_sim, 2)
        dist    = np.sqrt((diff ** 2).sum(axis=-1))             # (N_obs, N_sim)
        row_ind, col_ind = linear_sum_assignment(
            np.minimum(dist, max_match_px)
        )
        hit = row_ind[dist[row_ind, col_ind] < max_match_px]
        claimed[hit] = True

    return obs[~claimed], claimed


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
# Autoindexing helpers
# ─────────────────────────────────────────────────────────────────────────────


def _obs_q_vecs(camera, obs_xy: np.ndarray) -> np.ndarray:
    """Pixel positions → unit scattering vectors (N, 3)."""
    kf = camera.pixel_to_kf(obs_xy[:, 0], obs_xy[:, 1])
    ki = np.array([1.0, 0.0, 0.0])
    q = kf - ki[None, :]
    norms = np.linalg.norm(q, axis=1, keepdims=True)
    return q / norms


def _build_g_table(crystal, hkl_list: list):
    """
    Build a pairwise angle lookup table for the given list of (h,k,l).

    Returns:
        cos_sorted ((P,)): pairwise cosines sorted ascending
        ii_sorted ((P,)): first index into hkl_list for each pair
        jj_sorted ((P,)): second index
        G_hats ((M, 3)): unit reciprocal-lattice vectors, one per hkl
        hkl_list (list): input list (unchanged, for index→hkl lookup)
"""
    G_vecs = np.array([crystal.Q(h, k, l) for h, k, l in hkl_list])
    norms = np.linalg.norm(G_vecs, axis=1)
    G_hats = G_vecs / norms[:, None]

    ii, jj = np.triu_indices(len(G_hats), k=1)
    cos_vals = np.einsum("ij,ij->i", G_hats[ii], G_hats[jj])
    np.clip(cos_vals, -1.0, 1.0, out=cos_vals)

    order = np.argsort(cos_vals)
    return (
        cos_vals[order],
        ii[order].astype(np.int32),
        jj[order].astype(np.int32),
        G_hats,
        hkl_list,
    )


def _rotation_from_two_vecs(
    q1: np.ndarray, q2: np.ndarray,
    G1: np.ndarray, G2: np.ndarray,
) -> np.ndarray | None:
    """
    Rotation R such that R @ G1 = q1 and R approximately maps G2 → q2.

    Builds orthonormal frames from each pair via Gram-Schmidt, then
    R = Fq @ FG^T.  Returns None if the two vectors in either pair are
    nearly parallel (underdetermined).
"""
    def _frame(a, b):
        a = a / np.linalg.norm(a)
        b_perp = b - np.dot(a, b) * a
        n = np.linalg.norm(b_perp)
        if n < 1e-6:
            return None
        b_perp /= n
        return np.column_stack([a, b_perp, np.cross(a, b_perp)])

    Fq = _frame(q1, q2)
    FG = _frame(G1, G2)
    if Fq is None or FG is None:
        return None
    return Fq @ FG.T


def _score_index(
    U: np.ndarray,
    q_hats: np.ndarray,
    G_hats: np.ndarray,
    cos_tol: float,
) -> int:
    """
    Count observed q_hats that match any crystal G_hat under rotation U.

    Equivalent to checking angle(U^T @ q, G) < tol for each obs spot.
"""
    crystal_dirs = q_hats @ U           # (N, 3) — U^T @ q for each q
    cos_mat = crystal_dirs @ G_hats.T   # (N, M)
    return int((cos_mat.max(axis=1) >= cos_tol).sum())


# ─────────────────────────────────────────────────────────────────────────────
# Autoindexing
# ─────────────────────────────────────────────────────────────────────────────


def index_orientation(
    crystal,
    camera,
    obs_xy: np.ndarray,
    *,
    f2_thresh: float = F2_THRESHOLD,
    n_hkl_max: int = 200,
    E_ref_eV: float | None = None,
    angle_tol_deg: float = 0.5,
    min_match_rate: float = 0.25,
    n_obs_use: int = 20,
    max_pairs: int = 200,
    n_candidates_per_pair: int = 20,
    min_pair_angle_deg: float = 5.0,
    max_pair_angle_deg: float = 175.0,
    verbose: bool = False,
) -> "IndexResult":
    """
    Autoindex a Laue pattern: find an orientation matrix U from spot positions.

    Computes pairwise inter-spot angles from the observed scattering vectors
    and compares them against a lookup table of allowed inter-planar angles
    for the given crystal.  For each matching table entry a candidate U is
    built by Gram-Schmidt frame alignment; the candidate that matches the most
    observed spots is returned.

    The result is typically a rough orientation (±1° accuracy) suitable as the
    starting point for :func:`fit_orientation`.

    Args:
        crystal (Crystal): xrayutilities crystal structure.
        camera (Camera): Detector geometry.
        obs_xy ((N, 2)): Observed spot pixel positions `[xcam, ycam]`, sorted by descending
            intensity.  The `n_obs_use` brightest are used.
        f2_thresh (float): Minimum |F|² threshold for allowed reflections.
        n_hkl_max (int): Keep only the `n_hkl_max` strongest reflections (by |F|²) for the
            lookup table.  Limits table size and avoids O(M²) memory blow-up.
            Default 200.
        E_ref_eV (float or None): Reference photon energy (eV) used for |F|² ranking.  Defaults to the
            midpoint of the default energy window.
        angle_tol_deg (float): Angular tolerance (degrees) for both table look-up and final scoring.
            Tight values (0.3–0.5°) give fewer false matches; looser values
            (1–2°) help when the geometry calibration is coarse.
        min_match_rate (float): Minimum fraction of matched spots required for `result.success`.
        n_obs_use (int): Number of brightest observed spots to use.  Pairwise complexity is
            O(n_obs_use²), so keep this ≤ 30 for fast execution.
        max_pairs (int): Maximum number of observed pairs to evaluate.  Pairs are sorted so
            that those with angles nearest 90° (most discriminating) are tried
            first.
        n_candidates_per_pair (int): Maximum number of table entries (hkl₁, hkl₂) to try per observed
            pair.  When there are many table hits the candidates are sub-sampled
            uniformly.
        min_pair_angle_deg (float): Skip observed pairs whose inter-spot angle is below this threshold
            (nearly-parallel beams constrain U poorly).
        max_pair_angle_deg (float): Skip observed pairs above this threshold (nearly anti-parallel).
        verbose (bool): Print progress to stdout.

    Returns:
        IndexResult
            `result.U` is the best candidate orientation matrix.
            Pass it directly to :func:`fit_orientation` for pixel-space
            refinement::

            idx = index_orientation(crystal, camera, obs_xy, verbose=True)
            if idx.success:
                fit = fit_orientation(crystal, camera, obs_xy, idx.U)

    Note:
    The algorithm exploits the fact that in Laue diffraction the unit
    scattering vector satisfies `q̂ = U @ Ĝ(hkl)` independently of
    wavelength.  The inter-spot angle `arccos(q̂ᵢ · q̂ⱼ)` therefore
    equals the inter-planar angle `arccos(Ĝᵢ · Ĝⱼ)`, which is a
    fixed property of the crystal metric — no energy knowledge is
    required.
"""
    from .simulation import E_MIN_eV, E_MAX_eV

    obs_xy = np.asarray(obs_xy, dtype=float)
    if n_obs_use is not None:
        obs_xy = obs_xy[:n_obs_use]
    n_obs = len(obs_xy)

    if n_obs < 2:
        raise ValueError("Need at least 2 observed spots for indexing.")

    if E_ref_eV is None:
        E_ref_eV = 0.5 * (E_MIN_eV + E_MAX_eV)

    if verbose:
        print(f"index_orientation: {n_obs} observed spots")

    # ── step 1: observed unit q-vectors ──────────────────────────────────────
    q_hats = _obs_q_vecs(camera, obs_xy)

    # ── step 2: build HKL angle table ────────────────────────────────────────
    allowed = precompute_allowed_hkl(crystal, f2_thresh=f2_thresh)

    # Rank by |F|² and keep the n_hkl_max strongest reflections.
    hkl_all = list(allowed)
    if n_hkl_max is not None and len(hkl_all) > n_hkl_max:
        f2_vals = np.array([
            abs(crystal.StructureFactor(crystal.Q(h, k, l), en=E_ref_eV)) ** 2
            for h, k, l in hkl_all
        ])
        top_idx = np.argsort(f2_vals)[::-1][:n_hkl_max]
        hkl_all = [hkl_all[i] for i in top_idx]

    cos_sorted, ii_sorted, jj_sorted, G_hats, hkl_list = _build_g_table(
        crystal, hkl_all
    )

    if verbose:
        print(
            f"  HKL table: {len(hkl_list)} reflections, "
            f"{len(cos_sorted)} pairs"
        )

    # ── step 3: iterate over observed pairs ──────────────────────────────────
    cos_tol = float(np.cos(np.radians(angle_tol_deg)))
    # Conservative search window: |Δcos| ≤ Δα_rad  (|d cos/dα| = |sin α| ≤ 1)
    cos_search_win = float(np.radians(angle_tol_deg))

    cos_min_pair = float(np.cos(np.radians(max_pair_angle_deg)))
    cos_max_pair = float(np.cos(np.radians(min_pair_angle_deg)))

    # All observed pairs + their cosines
    obs_i_all, obs_j_all = zip(
        *[(i, j) for i in range(n_obs) for j in range(i + 1, n_obs)]
    )
    obs_i_all = np.array(obs_i_all)
    obs_j_all = np.array(obs_j_all)
    cos_obs_all = np.einsum(
        "ij,ij->i", q_hats[obs_i_all], q_hats[obs_j_all]
    )

    # Filter to useful angle range
    in_range = (cos_obs_all >= cos_min_pair) & (cos_obs_all <= cos_max_pair)
    obs_i_use = obs_i_all[in_range]
    obs_j_use = obs_j_all[in_range]
    cos_obs_use = cos_obs_all[in_range]

    # Sort by proximity to cos = 0 (90° angle = most discriminating)
    order = np.argsort(np.abs(cos_obs_use))
    obs_i_use = obs_i_use[order][:max_pairs]
    obs_j_use = obs_j_use[order][:max_pairs]
    cos_obs_use = cos_obs_use[order][:max_pairs]

    best_score = -1
    best_U: np.ndarray | None = None
    best_hkl_pair: tuple = ((0, 0, 0), (0, 0, 0))
    best_angle_deg = 0.0
    n_candidates_total = 0

    for oi, oj, cos_obs in zip(obs_i_use, obs_j_use, cos_obs_use):
        q1, q2 = q_hats[oi], q_hats[oj]

        lo = int(np.searchsorted(cos_sorted, cos_obs - cos_search_win))
        hi = int(np.searchsorted(cos_sorted, cos_obs + cos_search_win))
        n_hits = hi - lo
        if n_hits == 0:
            continue

        # Sub-sample uniformly when there are too many hits
        step = max(1, n_hits // n_candidates_per_pair)
        for k in range(lo, hi, step):
            G1 = G_hats[ii_sorted[k]]
            G2 = G_hats[jj_sorted[k]]
            U_cand = _rotation_from_two_vecs(q1, q2, G1, G2)
            if U_cand is None:
                continue
            n_candidates_total += 1
            score = _score_index(U_cand, q_hats, G_hats, cos_tol)
            if score > best_score:
                best_score = score
                best_U = U_cand
                best_hkl_pair = (
                    tuple(int(x) for x in hkl_list[ii_sorted[k]]),
                    tuple(int(x) for x in hkl_list[jj_sorted[k]]),
                )
                best_angle_deg = float(
                    np.degrees(np.arccos(np.clip(cos_obs, -1.0, 1.0)))
                )

    if best_U is None:
        best_U = np.eye(3)
        best_score = 0

    match_rate = best_score / max(n_obs, 1)
    result = IndexResult(
        U=best_U,
        n_matched=best_score,
        n_obs=n_obs,
        match_rate=match_rate,
        hkl_pair=best_hkl_pair,
        angle_deg=best_angle_deg,
        n_candidates=n_candidates_total,
        success=match_rate >= min_match_rate,
    )

    if verbose:
        print(f"  {result}")
        print(f"  evaluated {n_candidates_total} candidates")

    return result


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

    Intended to be passed directly to `scipy.optimize.least_squares`
    via :func:`functools.partial` (all arguments except `rotvec` frozen).

    Args:
        rotvec ((3,) ndarray): Rotation-vector increment δω (radians).
            The orientation evaluated at each call is
            `Rotation.from_rotvec(δω) @ U0`.
            Initialise with `np.zeros(3)`.
        crystal (Crystal): xrayutilities crystal structure.
        camera (Camera): Detector geometry (pixel size, distance, …).
        obs_xy ((N_obs, 2)): Observed pixel positions `[xcam, ycam]`,
            sorted by descending intensity.  Pass
            `peaklist[:, :2]` directly from
            :func:`~nrxrdct.laue.segmentation.convert_spotsfile2peaklist`.
        U0 ((3, 3)): Starting orientation matrix (LT frame,
            x-axis // beam direction).
        E_min_eV (float): Low-energy cut-off of the white beam (eV).
        E_max_eV (float): High-energy cut-off (eV).
        source (str            Spectral model): `'bending_magnet'` or
            `'undulator'`.
        source_kwargs: dict or None   Extra keyword arguments forwarded to the
            spectral function (e.g. `{'B': 0.4}` for
            a bending magnet field).
        f2_thresh (float): Minimum squared structure factor |F|² to
            include a reflection.
        kb_params (KB params): KB mirror reflectivity parameters
            (see :data:`BM32_KB`).
        max_match_px (float): Pixel radius inside which a simulated spot
            is considered a match for an observed spot.
            Unmatched observations contribute
            `(max_match_px, max_match_px)` to the
            residual vector.
        top_n_obs (int or None): Use only the N brightest observed spots.
            `None` uses all.
        top_n_sim (int or None): Consider only the N brightest simulated spots
            (after intensity-sorting by the simulator).

    Returns:
        residuals ((2 * N_obs_use,) ndarray): Interleaved `[Δx₀, Δy₀, Δx₁, Δy₁, …]`.  Length is fixed at
            `2 * min(N_obs, top_n_obs)` so it is compatible with
            `least_squares`.
"""
    delta_R = Rotation.from_rotvec(np.asarray(rotvec, dtype=float)).as_matrix()
    U = delta_R @ np.asarray(U0, dtype=float)

    obs_use = np.asarray(obs_xy, dtype=float)
    if top_n_obs is not None:
        obs_use = obs_use[:top_n_obs]

    sim_xy = simulate_laue(
        crystal, U, camera,
        E_min=E_min_eV, E_max=E_max_eV,
        source=source, source_kwargs=source_kwargs,
        f2_thresh=f2_thresh, kb_params=kb_params,
        geometry_only=geometry_only,
        allowed_hkl=allowed_hkl,
        _pixels_only=(allowed_hkl is not None),
    )
    if allowed_hkl is None:
        sim_xy = _extract_sim_xy(sim_xy)
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
    f2_thresh: float = F2_THRESHOLD,
    kb_params=BM32_KB,
    structure_model: str = "average",
    max_match_px: float = 30.0,
    top_n_obs: int | None = None,
    top_n_sim: int | None = None,
    geometry_only: bool = False,
    allowed_hkl=None,
    correct_depth: bool = False,
) -> np.ndarray:
    """
    Pixel-space residual vector for a layered crystal — single global rotation.

    A single rotation `R = Rotation.from_rotvec(δω)` is applied to every
    layer: `stack.all_layers[i].U = R @ U0_layers[i]`.  All inter-layer
    orientation relationships are therefore preserved throughout the fit.

    The stack is mutated in-place on every call.  When used through
    :func:`fit_orientation_stack` the original U matrices are automatically
    restored after the optimizer returns.  If called directly, the caller is
    responsible for saving and restoring `layer.U`.

    Args:
        rotvec ((3,) ndarray): Global rotation-vector increment δω (rad).
            Initialise with `np.zeros(3)`.
        stack (LayeredCrystal): Layered structure; mutated in-place.
        camera (Camera): Detector geometry.
        obs_xy ((N_obs, 2)): Observed pixel positions `[xcam, ycam]`.
        U0_layers (list of (3, 3)): Base orientation for each layer, in
            `stack.all_layers` order.  Typically
            captured as
            `[l.U.copy() for l in stack.all_layers]`
            before the first call.
        E_min_eV (float): Low-energy cut-off (eV).
        E_max_eV (float): High-energy cut-off (eV).
        source (str): Spectral model (`'bending_magnet'` or
            `'undulator'`).
        source_kwargs: dict or None     Extra kwargs for the spectral function.
        f2_thresh (float): Minimum |F|² threshold.
        kb_params (KB mirror reflectivity parameters.):
        structure_model (str): How to combine layer contributions —
            `'average'` (default) or `'incoherent'`.
        max_match_px (float): Match radius in pixels.
        top_n_obs (int or None): Brightest N observed spots to use.
        top_n_sim (int or None): Brightest N simulated spots to consider.

    Returns:
        residuals ((2 * N_obs_use,) ndarray): Fixed-length interleaved `[Δx, Δy]` residual vector.
"""
    delta_R = Rotation.from_rotvec(np.asarray(rotvec, dtype=float)).as_matrix()
    for layer, U0 in zip(stack.all_layers, U0_layers):
        layer.U = delta_R @ U0

    spots = simulate_laue_stack(
        stack, camera,
        E_min_eV=E_min_eV, E_max_eV=E_max_eV,
        source=source, source_kwargs=source_kwargs,
        f2_thresh=f2_thresh, kb_params=kb_params,
        structure_model=structure_model,
        verbose=False,
        geometry_only=geometry_only,
        allowed_hkl=allowed_hkl,
        correct_depth=correct_depth,
    )

    obs_use = np.asarray(obs_xy, dtype=float)
    if top_n_obs is not None:
        obs_use = obs_use[:top_n_obs]

    sim_xy = _extract_sim_xy(spots)
    if top_n_sim is not None:
        sim_xy = sim_xy[:top_n_sim]

    return _build_residuals(obs_use, sim_xy, max_match_px)


def laue_strain_stack_residuals(
    params: np.ndarray,
    stack,
    camera,
    obs_xy: np.ndarray,
    U0_layers: list[np.ndarray],
    fit_strain: tuple[str, ...] = _STRAIN_ALL,
    strain_scale: float = 1e-4,
    E_min_eV: float = E_MIN_eV,
    E_max_eV: float = E_MAX_eV,
    source: str = "bending_magnet",
    source_kwargs: dict | None = None,
    f2_thresh: float = F2_THRESHOLD,
    kb_params=BM32_KB,
    structure_model: str = "average",
    max_match_px: float = 30.0,
    top_n_obs: int | None = None,
    top_n_sim: int | None = None,
    geometry_only: bool = False,
    allowed_hkl=None,
    correct_depth: bool = False,
) -> np.ndarray:
    """
    Pixel-space residual vector for simultaneous orientation + per-layer strain
    refinement of a layered crystal.

    A single global rotation is applied to all layers; each layer additionally
    gets its own strain tensor.  The effective matrix for layer *i* is::

        U_eff_i = R(δω) @ U0_i @ (I + ε_i)

    Parameter layout::

        params = [δω_x, δω_y, δω_z,
                  ε_layer0_0, …, ε_layer0_{n_strain-1},
                  ε_layer1_0, …, ε_layer1_{n_strain-1}, …]

    Total length: ``3 + N_layers * len(fit_strain)``.

    Args:
        params ((3 + N_layers * n_strain,) ndarray): Rotation-vector δω (first 3)
            followed by per-layer strain components, each divided by
            `strain_scale`.  Initialise with ``np.zeros(3 + N * n_strain)``.
        stack (LayeredCrystal): Mutated in-place on every call.
        camera (Camera): Detector geometry.
        obs_xy ((N_obs, 2)): Observed pixel positions `[xcam, ycam]`.
        U0_layers (list of (3, 3)): Base orientation per layer in
            `stack.all_layers` order.
        fit_strain (tuple of str): Active strain components.
        strain_scale (float): Internal scale factor for strain parameters.

    Returns:
        residuals ((2 * N_obs_use,) ndarray): Fixed-length interleaved `[Δx, Δy]`.
"""
    params = np.asarray(params, dtype=float)
    n_strain = len(fit_strain)
    R = Rotation.from_rotvec(params[:3]).as_matrix()

    for ii, (layer, U0) in enumerate(zip(stack.all_layers, U0_layers)):
        strain_vals = params[3 + ii * n_strain : 3 + (ii + 1) * n_strain] * strain_scale
        eps = _strain_matrix(strain_vals, fit_strain)
        layer.U = R @ U0 @ (np.eye(3) + eps)

    spots = simulate_laue_stack(
        stack, camera,
        E_min_eV=E_min_eV, E_max_eV=E_max_eV,
        source=source, source_kwargs=source_kwargs,
        f2_thresh=f2_thresh, kb_params=kb_params,
        structure_model=structure_model,
        verbose=False,
        geometry_only=geometry_only,
        allowed_hkl=allowed_hkl,
        correct_depth=correct_depth,
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

    Args:
        params ((3,) or (3 * N_phases,) ndarray): Rotation-vector increment(s).  In shared mode (`shared=True`)
            a single 3-element vector rotates every phase together.  In
            per-phase mode (`shared=False`) the vector has length
            `3 * N_phases`: elements `[3i : 3i+3]` rotate phase *i*.
            Initialise with `np.zeros(3)` or `np.zeros(3 * N_phases)`.
        phases (list of dicts): Phase descriptors; `p["U"]` is updated
            in-place on every call.  Each dict must
            contain `'crystal'`, `'U'`, and
            `'volume_fraction'`.
        camera (Camera): Detector geometry.
        obs_xy ((N_obs, 2)): Observed pixel positions `[xcam, ycam]`.
        U0_list (list of (3, 3)): Base orientation per phase, in input order.
        shared (bool): `True` → one global rotation for all phases.
            `False` → independent rotation per phase.
        E_min_eV (float): Low-energy cut-off (eV).
        E_max_eV (float): High-energy cut-off (eV).
        source (str): Spectral model (`'bending_magnet'` or
            `'undulator'`).
        source_kwargs: dict or None    Extra kwargs for the spectral function.
        f2_thresh (float or None): Minimum |F|² threshold.
        kb_params (KB mirror reflectivity parameters.):
        structure_model (str): Layer combination model (`'average'` or
            `'incoherent'`).
        max_match_px (float): Match radius in pixels.
        top_n_obs (int or None): Brightest N observed spots to use.
        top_n_sim (int or None): Brightest N simulated spots to consider.

    Returns:
        residuals ((2 * N_obs_use,) ndarray): Fixed-length interleaved `[Δx, Δy]` residual vector.
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
        f2_thresh=f2_thresh, kb_params=kb_params,
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
    Because `simulate_laue` does not enforce SO(3), the non-orthogonal
    `U_eff` correctly shifts every d-spacing by the corresponding strain
    component.

    Args:
        params ((3 + n_strain,) ndarray): First 3 elements: rotation-vector increment δω (radians).
            Remaining `n_strain` elements: strain components scaled by
            `strain_scale` (i.e. divide by `strain_scale` to get
            physical strain).  Initialise with `np.zeros(3 + n_strain)`.
        crystal (Crystal): xrayutilities crystal structure.
        camera (Camera): Detector geometry.
        obs_xy ((N_obs, 2)): Observed pixel positions `[xcam, ycam]`.
        U0 ((3, 3)): Starting orientation matrix.
        fit_strain (tuple of str): Which strain components are free.  Any
            subset of `('e_xx','e_yy','e_zz','e_xy',
            'e_xz','e_yz')`.  Default: all six.
        strain_scale (float): Internal scale factor for strain parameters.
            Optimizer parameters = physical_strain /
            strain_scale.  Default 1e-4 keeps parameters
            near order-1 for typical strains of 10⁻⁴–10⁻³.
        E_min_eV, E_max_eV, source, source_kwargs, f2_thresh, kb_params,
        max_match_px, top_n_obs, top_n_sim, geometry_only, allowed_hkl
            Forwarded to :func:`simulate_laue`; see :func:`laue_residuals`.

    Returns:
        residuals ((2 * N_obs_use,) ndarray): Fixed-length interleaved `[Δx, Δy]` residual vector.
"""
    params = np.asarray(params, dtype=float)
    rotvec = params[:3]
    strain_vals = params[3:] * strain_scale

    R = Rotation.from_rotvec(rotvec).as_matrix()
    eps = _strain_matrix(strain_vals, fit_strain)
    U_eff = R @ np.asarray(U0, dtype=float) @ (np.eye(3) + eps)

    obs_use = np.asarray(obs_xy, dtype=float)
    if top_n_obs is not None:
        obs_use = obs_use[:top_n_obs]

    sim_xy = simulate_laue(
        crystal, U_eff, camera,
        E_min=E_min_eV, E_max=E_max_eV,
        source=source, source_kwargs=source_kwargs,
        f2_thresh=f2_thresh, kb_params=kb_params,
        geometry_only=geometry_only,
        allowed_hkl=allowed_hkl,
        _pixels_only=(allowed_hkl is not None),
    )
    if allowed_hkl is None:
        sim_xy = _extract_sim_xy(sim_xy)
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
    f2_thresh: float = F2_THRESHOLD,
    kb_params=BM32_KB,
    max_match_px: float | list[float] = (15.0, 3.0),
    top_n_obs: int | None = 300,
    top_n_sim: int | None = 300,
    method: str = "lm",
    ftol: float = 1e-6,
    xtol: float = 1e-6,
    gtol: float = 1e-8,
    max_nfev: int = 500,
    geometry_only: bool = True,
    allowed_hkl=None,
    z_scan_step_deg: float | None = None,
    z_axis: np.ndarray | None = None,
    verbose: bool = False,
) -> OrientationFitResult:
    """
    Refine the orientation matrix of a single crystal to match observed spots.

    Wraps :func:`laue_residuals` + `scipy.optimize.least_squares`.

    Args:
        crystal (Crystal): xrayutilities crystal structure.
        camera (Camera): Detector geometry.
        obs_xy ((N_obs, 2)): Observed pixel positions `[xcam, ycam]`,
            sorted by descending intensity.  Pass
            `peaklist[:, :2]` directly from
            :func:`~nrxrdct.laue.segmentation.convert_spotsfile2peaklist`.
        U0 ((3, 3)): Starting orientation matrix (LT frame,
            x-axis // beam direction).
        E_min_eV (float): Low-energy cut-off of the white beam (eV).
        E_max_eV (float): High-energy cut-off (eV).
        source (str            Spectral model): `'bending_magnet'` or
            `'undulator'`.
        source_kwargs: dict or None   Extra kwargs forwarded to the spectral function.
        f2_thresh (float): Minimum squared structure factor |F|² to
            include a reflection.
        kb_params (KB mirror reflectivity parameters.):
        max_match_px (float or list of float): Pixel tolerance(s) for spot matching.  A
            single float runs one fit.  A decreasing
            list (e.g. `[50, 20, 5]`) runs staged
            refinement: each stage warm-starts from the
            previous solution, progressively tightening
            the matching window to sharpen convergence.
        top_n_obs (int or None): Use only the brightest N observed spots.
            Reduces cost per iteration; useful when the
            spot list contains many weak peaks.
        top_n_sim (int or None): Consider only the brightest N simulated spots.
        method (str): `least_squares` algorithm: `'lm'` (fast,
            unconstrained Levenberg–Marquardt) or `'trf'`
            (trust-region reflective, more robust for large
            initial misalignments).
        ftol, xtol, gtol (float): Convergence tolerances forwarded to
            `least_squares`.
        max_nfev (int): Maximum number of residual evaluations.
        z_scan_step_deg (float or None): When not `None`, perform a coarse grid
            search over in-plane rotations before the
            local refinement.  The starting orientation
            `U0` is rotated around `z_axis` in steps
            of `z_scan_step_deg` degrees from 0° to
            360°.  The candidate with the lowest residual
            cost is used as the starting point for
            `least_squares`.  Useful for non-cubic
            crystals where Euler-angle initialisation may
            land in the wrong basin.  Typical values:
            10–30° for a fast scan, 2–5° for a fine one.
        z_axis ((3,) array or None): Unit vector (in the LaueTools lab frame) to
            rotate around during the scan.  Defaults to
            the lab Z axis `[0, 0, 1]` (vertical).
            Pass the crystal c-axis direction (in the lab
            frame) for a structure-aware scan.
        verbose (bool): Print a one-line summary after convergence.

    Returns:
        OrientationFitResult
            Call `str(result)` for a one-line summary.  Apply the refined
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

    _stages = (
        [float(max_match_px)] if np.isscalar(max_match_px)
        else [float(v) for v in max_match_px]
    )

    _allowed = (
        allowed_hkl if allowed_hkl is not None
        else precompute_allowed_hkl(crystal, E_max_eV=E_max_eV, f2_thresh=f2_thresh)
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
            f2_thresh=f2_thresh, kb_params=kb_params,
            max_match_px=_stages[0], top_n_obs=None, top_n_sim=top_n_sim,
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

    # ── staged refinement loop ────────────────────────────────────────────────
    U0_stage = U0.copy()
    opt = None
    for _si, _px in enumerate(_stages):
        _fun = partial(
            laue_residuals,
            crystal=crystal, camera=camera, obs_xy=obs_use, U0=U0_stage,
            E_min_eV=E_min_eV, E_max_eV=E_max_eV,
            source=source, source_kwargs=source_kwargs,
            f2_thresh=f2_thresh, kb_params=kb_params,
            max_match_px=_px, top_n_obs=None, top_n_sim=top_n_sim,
            geometry_only=False, allowed_hkl=_allowed,
        )
        opt = least_squares(
            _fun, x0=np.zeros(3),
            method=method, ftol=ftol, xtol=xtol, gtol=gtol, max_nfev=max_nfev,
        )
        if verbose and len(_stages) > 1:
            _nm, _rms, _ = _compute_match_stats(opt.fun, _px, N_obs)
            print(
                f"  stage {_si + 1}/{len(_stages)}  px={_px:.1f}:"
                f"  matched={_nm}  rms={_rms:.2f} px"
            )
        if _si < len(_stages) - 1:
            U0_stage = Rotation.from_rotvec(opt.x).as_matrix() @ U0_stage

    U_final = Rotation.from_rotvec(opt.x).as_matrix() @ U0_stage
    n_matched, rms_px, mean_px = _compute_match_stats(opt.fun, _stages[-1], N_obs)

    # One extra simulation to report n_sim at solution.
    final_spots = simulate_laue(
        crystal, U_final, camera,
        E_min=E_min_eV, E_max=E_max_eV,
        source=source, source_kwargs=source_kwargs,
        f2_thresh=f2_thresh, kb_params=kb_params,
        allowed_hkl=_allowed,
    )
    n_sim = len(_extract_sim_xy(final_spots))

    rotvec_total = Rotation.from_matrix(U_final @ U0_input.T).as_rotvec()
    result = OrientationFitResult(
        U=U_final, U0=U0_input, rotvec=rotvec_total,
        cost=float(opt.cost), rms_px=rms_px, mean_px=mean_px,
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
    f2_thresh: float = F2_THRESHOLD,
    kb_params=BM32_KB,
    structure_model: str = "average",
    max_match_px: float | list[float] = 30.0,
    top_n_obs: int | None = 300,
    top_n_sim: int | None = 300,
    method: str = "lm",
    ftol: float = 1e-6,
    xtol: float = 1e-6,
    gtol: float = 1e-8,
    max_nfev: int = 500,
    update_stack: bool = True,
    geometry_only: bool = True,
    correct_depth: bool = False,
    allowed_hkl=None,
    verbose: bool = False,
) -> StackFitResult:
    """
    Refine the orientation of a :class:`~nrxrdct.laue.layers.LayeredCrystal`.

    A single global rotation is applied to all layers simultaneously so
    all inter-layer orientation relationships are preserved throughout the
    fit.  The starting U matrix of each layer is snapshotted at call time;
    the stack is restored to a clean state regardless of optimizer success
    or failure.

    **Staged refinement**
    When ``max_match_px`` is a decreasing list (e.g. ``[50, 15, 3]``) the
    fit runs in multiple stages.  After each non-final stage the global
    rotation found so far is composed into every layer's warm-start
    orientation:

    .. code-block:: text

        U0_i ← R(δω_stage) @ U0_i   for all layers i

    The next stage then solves from δω = 0 with a tighter matching window,
    so Levenberg–Marquardt always operates near the identity.  The
    ``R_global`` and ``rotvec`` in the returned :class:`StackFitResult`
    represent the total rotation accumulated across all stages, computed as
    ``R_global = U_final @ U0_original.T`` for the first layer.

    Args:
        stack (LayeredCrystal): Layered structure to fit.  Layer U
            matrices are used as starting orientations
            and optionally updated after convergence.
        camera (Camera): Detector geometry.
        obs_xy ((N_obs, 2)): Observed pixel positions `[xcam, ycam]`.
        E_min_eV (float): Low-energy cut-off (eV).
        E_max_eV (float): High-energy cut-off (eV).
        source (str): Spectral model (`'bending_magnet'` or
            `'undulator'`).
        source_kwargs (dict or None): Extra kwargs for the spectral function.
        f2_thresh (float): Minimum |F|² threshold.
        kb_params (KB mirror reflectivity parameters.):
        structure_model: str              Layer combination model — `'average'`
            (default) or `'incoherent'`.
        max_match_px (float or list of float): Pixel match radius.  A single float
            runs one fit.  A decreasing list (e.g.
            ``[50, 15, 3]``) runs staged refinement: each
            stage warm-starts from the previous solution,
            progressively tightening the matching window.
        top_n_obs (int or None): Brightest N observed spots to use.
        top_n_sim (int or None): Brightest N simulated spots to consider.
        method (str): `least_squares` algorithm (`'lm'` or
            `'trf'`).
        ftol, xtol, gtol (float): Convergence tolerances.
        max_nfev (int): Maximum residual evaluations.
        update_stack (bool): If `True` (default), write the refined
            U matrices back into `stack.all_layers`
            after convergence.  Set `False` to leave
            the stack unchanged and inspect the result
            before committing.
        verbose (bool): Print a one-line summary after convergence.

    Returns:
        StackFitResult
            `result.R_global` is the total 3×3 rotation applied to all layers.
            `result.U_layers` lists the refined U for each layer in
            `stack.all_layers` order.
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
    if allowed_hkl is not None:
        _allowed = allowed_hkl
    elif geometry_only:
        _enum_pool = (
            stack.buffer_layers + stack.layers[:1]
            if structure_model == "average"
            else stack.all_layers
        )
        _allowed = {
            id(layer.crystal): precompute_allowed_hkl(
                layer.crystal, E_max_eV=E_max_eV, f2_thresh=f2_thresh
            )
            for layer in _enum_pool
        }
    else:
        _allowed = None

    _stages = (
        [float(max_match_px)] if np.isscalar(max_match_px)
        else [float(v) for v in max_match_px]
    )

    U0_stage = [U0.copy() for U0 in U0_layers]
    opt = None

    try:
        for _si, _px in enumerate(_stages):
            fun = partial(
                laue_stack_residuals,
                stack=stack, camera=camera, obs_xy=obs_use,
                U0_layers=U0_stage,
                E_min_eV=E_min_eV, E_max_eV=E_max_eV,
                source=source, source_kwargs=source_kwargs,
                f2_thresh=f2_thresh, kb_params=kb_params,
                structure_model=structure_model,
                max_match_px=_px, top_n_obs=None, top_n_sim=top_n_sim,
                geometry_only=False, allowed_hkl=_allowed,
                correct_depth=correct_depth,
            )
            opt = least_squares(
                fun, x0=np.zeros(3),
                method=method, ftol=ftol, xtol=xtol, gtol=gtol, max_nfev=max_nfev,
            )
            if verbose and len(_stages) > 1:
                _nm, _rms, _ = _compute_match_stats(opt.fun, _px, N_obs)
                print(
                    f"  stage {_si + 1}/{len(_stages)}  px={_px:.1f}:"
                    f"  matched={_nm}  rms={_rms:.2f} px"
                )
            if _si < len(_stages) - 1:
                R_step = Rotation.from_rotvec(opt.x).as_matrix()
                U0_stage = [R_step @ U0 for U0 in U0_stage]
    finally:
        # Always restore original U matrices so the stack is in a known state.
        for layer, U0 in zip(stack.all_layers, U0_layers):
            layer.U = U0.copy()

    R_last = Rotation.from_rotvec(opt.x).as_matrix()
    U_layers_final = [R_last @ U0 for U0 in U0_stage]
    R_global = Rotation.from_matrix(U_layers_final[0] @ U0_layers[0].T).as_matrix()

    if update_stack:
        for layer, U_new in zip(stack.all_layers, U_layers_final):
            layer.U = U_new.copy()

    n_matched, rms_px, mean_px = _compute_match_stats(opt.fun, _stages[-1], N_obs)

    # Final simulation for n_sim.
    for layer, U_new in zip(stack.all_layers, U_layers_final):
        layer.U = U_new.copy()
    final_spots = simulate_laue_stack(
        stack, camera, E_min_eV=E_min_eV, E_max_eV=E_max_eV,
        source=source, source_kwargs=source_kwargs,
        f2_thresh=f2_thresh, kb_params=kb_params,
        structure_model=structure_model, verbose=False,
        allowed_hkl=_allowed, correct_depth=correct_depth,
    )
    n_sim = len(_extract_sim_xy(final_spots))

    # Restore to the refined state (or original if update_stack is False).
    if not update_stack:
        for layer, U0 in zip(stack.all_layers, U0_layers):
            layer.U = U0.copy()

    rotvec_total = Rotation.from_matrix(R_global).as_rotvec()
    result = StackFitResult(
        R_global=R_global, rotvec=rotvec_total,
        U_layers=U_layers_final, U0_layers=U0_layers,
        cost=float(opt.cost), rms_px=rms_px, mean_px=mean_px,
        n_matched=n_matched, n_obs=N_obs, n_sim=n_sim,
        match_rate=n_matched / max(N_obs, 1),
        success=opt.success, message=opt.message, optimizer=opt,
    )

    if verbose:
        print(f"  {result}")

    return result


def fit_strain_orientation_stack(
    stack,
    camera,
    obs_xy: np.ndarray,
    fit_strain: tuple[str, ...] = _STRAIN_ALL,
    strain_scale: float = 1e-4,
    E_min_eV: float = E_MIN_eV,
    E_max_eV: float = E_MAX_eV,
    source: str = "bending_magnet",
    source_kwargs: dict | None = None,
    f2_thresh: float = F2_THRESHOLD,
    kb_params=BM32_KB,
    structure_model: str = "average",
    max_match_px: float | list[float] = (15.0, 3.0),
    top_n_obs: int | None = 300,
    top_n_sim: int | None = 300,
    method: str = "lm",
    ftol: float = 1e-8,
    xtol: float = 1e-8,
    gtol: float = 1e-8,
    max_nfev: int = 2000,
    update_stack: bool = True,
    geometry_only: bool = True,
    correct_depth: bool = False,
    allowed_hkl=None,
    verbose: bool = False,
) -> StackStrainFitResult:
    """
    Simultaneously refine orientation and per-layer lattice strain for a
    :class:`~nrxrdct.laue.layers.LayeredCrystal`.

    A single global rotation is shared by all layers (preserving inter-layer
    relationships) while each layer is given its own independent strain tensor.
    The effective deformation matrix for layer *i* is::

        U_eff_i = R(δω) @ U0_i @ (I + ε_i)

    **Staged refinement** works identically to :func:`fit_orientation_stack`:
    between stages the accumulated rotation is baked into ``U0_stage`` and the
    strain parameters are carried forward so the final tight stage does not
    collapse into a degenerate minimum.

    Args:
        stack (LayeredCrystal): Layered structure to fit.  Layer U matrices
            are used as starting orientations and optionally updated after
            convergence.
        camera (Camera): Detector geometry.
        obs_xy ((N_obs, 2) ndarray): Observed pixel positions `[xcam, ycam]`.
        fit_strain (tuple of str): Strain components to refine.  Any subset of
            `('e_xx','e_yy','e_zz','e_xy','e_xz','e_yz')`.
            Default: all six per layer.
        strain_scale (float): Internal scale for strain parameters; see
            :func:`laue_strain_residuals`.  Default ``1e-4``.
        E_min_eV, E_max_eV, source, source_kwargs, f2_thresh, kb_params,
        structure_model, max_match_px, top_n_obs, top_n_sim, method,
        ftol, xtol, gtol, max_nfev, geometry_only
            Forwarded to :func:`laue_strain_stack_residuals` / ``least_squares``.
        update_stack (bool): If ``True`` (default), write the refined ``U_eff``
            matrices back into ``stack.all_layers`` after convergence.
        verbose (bool): Print a one-line summary after each stage.

    Returns:
        StackStrainFitResult
            `result.U_eff_layers[i]` is the effective matrix for layer *i*;
            pass it as `U` to :func:`~nrxrdct.laue.simulation.simulate_laue`
            to reproduce the fitted spot pattern for that layer alone.
"""
    obs_use = np.asarray(obs_xy, dtype=float)
    if top_n_obs is not None:
        obs_use = obs_use[:top_n_obs]
    N_obs = len(obs_use)

    n_layers = len(stack.all_layers)
    n_strain = len(fit_strain)
    n_params  = 3 + n_layers * n_strain

    U0_layers = [layer.U.copy() for layer in stack.all_layers]

    if verbose:
        print(
            f"fit_strain_orientation_stack: {N_obs} observed spots, "
            f"{n_layers} layers, strain components: {list(fit_strain)}"
        )

    if allowed_hkl is not None:
        _allowed = allowed_hkl
    elif geometry_only:
        _enum_pool = (
            stack.buffer_layers + stack.layers[:1]
            if structure_model == "average"
            else stack.all_layers
        )
        _allowed = {
            id(layer.crystal): precompute_allowed_hkl(
                layer.crystal, E_max_eV=E_max_eV, f2_thresh=f2_thresh
            )
            for layer in _enum_pool
        }
    else:
        _allowed = None

    _stages = (
        [float(max_match_px)] if np.isscalar(max_match_px)
        else [float(v) for v in max_match_px]
    )

    U0_stage = [U0.copy() for U0 in U0_layers]
    opt = None
    _x0 = np.zeros(n_params)

    try:
        for _si, _px in enumerate(_stages):
            fun = partial(
                laue_strain_stack_residuals,
                stack=stack, camera=camera, obs_xy=obs_use,
                U0_layers=U0_stage,
                fit_strain=fit_strain, strain_scale=strain_scale,
                E_min_eV=E_min_eV, E_max_eV=E_max_eV,
                source=source, source_kwargs=source_kwargs,
                f2_thresh=f2_thresh, kb_params=kb_params,
                structure_model=structure_model,
                max_match_px=_px, top_n_obs=None, top_n_sim=top_n_sim,
                geometry_only=False, allowed_hkl=_allowed,
                correct_depth=correct_depth,
            )
            opt = least_squares(
                fun, x0=_x0,
                method=method, ftol=ftol, xtol=xtol, gtol=gtol, max_nfev=max_nfev,
            )
            if verbose and len(_stages) > 1:
                _nm, _rms, _ = _compute_match_stats(opt.fun, _px, N_obs)
                print(
                    f"  stage {_si + 1}/{len(_stages)}  px={_px:.1f}:"
                    f"  matched={_nm}  rms={_rms:.2f} px"
                )
            if _si < len(_stages) - 1:
                R_step = Rotation.from_rotvec(opt.x[:3]).as_matrix()
                U0_stage = [R_step @ U0 for U0 in U0_stage]
                # Bake rotation into U0_stage; carry strain forward.
                _x0 = np.zeros(n_params)
                _x0[3:] = opt.x[3:]
    finally:
        for layer, U0 in zip(stack.all_layers, U0_layers):
            layer.U = U0.copy()

    # Unpack solution.
    R_last = Rotation.from_rotvec(opt.x[:3]).as_matrix()
    U_layers_final = []
    U_eff_layers   = []
    strain_tensors = []
    strain_voigts  = []

    for ii, U0 in enumerate(U0_stage):
        strain_vals = opt.x[3 + ii * n_strain : 3 + (ii + 1) * n_strain] * strain_scale
        eps    = _strain_matrix(strain_vals, fit_strain)
        U_pure = R_last @ U0
        U_eff  = U_pure @ (np.eye(3) + eps)
        U_layers_final.append(U_pure)
        U_eff_layers.append(U_eff)
        strain_tensors.append(eps)
        strain_voigts.append(_strain_to_voigt(strain_vals, fit_strain))

    R_global = Rotation.from_matrix(U_layers_final[0] @ U0_layers[0].T).as_matrix()

    if update_stack:
        for layer, U_eff in zip(stack.all_layers, U_eff_layers):
            layer.U = U_eff.copy()

    n_matched, rms_px, mean_px = _compute_match_stats(opt.fun, _stages[-1], N_obs)

    # Final simulation for n_sim.
    for layer, U_eff in zip(stack.all_layers, U_eff_layers):
        layer.U = U_eff.copy()
    final_spots = simulate_laue_stack(
        stack, camera, E_min_eV=E_min_eV, E_max_eV=E_max_eV,
        source=source, source_kwargs=source_kwargs,
        f2_thresh=f2_thresh, kb_params=kb_params,
        structure_model=structure_model, verbose=False,
        allowed_hkl=_allowed, correct_depth=correct_depth,
    )
    n_sim = len(_extract_sim_xy(final_spots))

    if not update_stack:
        for layer, U0 in zip(stack.all_layers, U0_layers):
            layer.U = U0.copy()

    rotvec_total = Rotation.from_matrix(R_global).as_rotvec()
    result = StackStrainFitResult(
        R_global=R_global, rotvec=rotvec_total,
        U_layers=U_layers_final, U0_layers=U0_layers,
        U_eff_layers=U_eff_layers,
        strain_tensors=strain_tensors, strain_voigts=strain_voigts,
        fit_strain=tuple(fit_strain),
        cost=float(opt.cost), rms_px=rms_px, mean_px=mean_px,
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
    f2_thresh: float | None = None,
    kb_params=BM32_KB,
    structure_model: str = "average",
    max_match_px: float | list[float] = 30.0,
    top_n_obs: int | None = 300,
    top_n_sim: int | None = 300,
    method: str = "lm",
    ftol: float = 1e-6,
    xtol: float = 1e-6,
    gtol: float = 1e-8,
    max_nfev: int = 500,
    update_phases: bool = True,
    geometry_only: bool = True,
    allowed_hkl: dict | None = None,
    verbose: bool = False,
) -> MixedFitResult:
    """
    Refine orientations of a multi-phase Laue pattern.

    Supports two coupling modes controlled by `shared`:

    - **Shared** (default): one global rotation for all phases (3 free
      parameters).  Use this when the phases are co-oriented (e.g. an
      epitaxial stack measured as a mixed signal) and you want to preserve
      their relative orientations.
    - **Per-phase**: independent rotation per phase (3 × N_phases free
      parameters).  Use this when phases may have drifted independently or
      when the initial guess for each phase was set separately.

    **Staged refinement**
    When ``max_match_px`` is a decreasing list (e.g. ``[50, 15, 3]``) the
    fit runs in multiple stages.  Between stages the current solution is
    composed into the warm-start orientations so the next stage always
    starts from δω = 0:

    - *Shared mode*: one global rotation is applied to all phases::

          U0_i ← R(δω_stage) @ U0_i   for all phases i

    - *Per-phase mode*: each phase receives its own independent correction::

          U0_i ← R(δω_stage,i) @ U0_i   for phase i

    The ``rotvecs`` list in the returned :class:`MixedFitResult` always
    contains the total rotation vectors accumulated across all stages,
    computed as ``Rotation.from_matrix(U_final_i @ U0_original_i.T).as_rotvec()``
    for each phase.

    Args:
        phases (list of dicts or tuples): Phase descriptors in the same format as
            :func:`~nrxrdct.laue.simulation.simulate_mixed_phases`.
            Each entry must provide `'crystal'`, `'U'`, and
            `'volume_fraction'`.
        camera (Camera): Detector geometry.
        obs_xy ((N_obs, 2)): Observed pixel positions `[xcam, ycam]`.
        shared (bool): `True` → single global rotation (3 DOF).
            `False` → independent rotation per phase
            (3 × N_phases DOF).
        E_min_eV (float): Low-energy cut-off (eV).
        E_max_eV (float): High-energy cut-off (eV).
        source (str): Spectral model (`'bending_magnet'` or
            `'undulator'`).
        source_kwargs (dict or None): Extra kwargs for the spectral function.
        f2_thresh (float or None): Minimum |F|² threshold.
        kb_params (KB mirror reflectivity parameters.):
        structure_model (str): Layer combination model (`'average'` or
            `'incoherent'`).
        max_match_px (float or list of float): Pixel match radius.  A single float
            runs one fit.  A decreasing list (e.g.
            ``[50, 15, 3]``) runs staged refinement: each
            stage warm-starts from the previous solution,
            progressively tightening the matching window.
        top_n_obs (int or None): Brightest N observed spots to use.
        top_n_sim (int or None): Brightest N simulated spots to consider.
        method (str): `least_squares` algorithm (`'lm'` or
            `'trf'`).
        ftol, xtol, gtol (float): Convergence tolerances.
        max_nfev (int): Maximum residual evaluations.
        update_phases (bool): If `True` (default), write the refined
            U matrices back into the original phase
            dicts (dict input only; tuple input is
            not mutated).
        allowed_hkl (dict or None): Pre-computed allowed-reflection sets keyed by
            ``id(crystal)``, as returned by
            :func:`~nrxrdct.laue.simulation.precompute_allowed_hkl`.
            When supplied, ``geometry_only`` is ignored and no
            recomputation is performed.  Pass this from the
            SLURM worker to avoid redundant per-frame recomputation.
        verbose (bool): Print a one-line summary after convergence.

    Returns:
        MixedFitResult
            `result.U_phases[i]` is the refined orientation of phase *i*.
            `result.rotvecs[i]`  is the corresponding rotation vector (all
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
    if allowed_hkl is not None:
        _allowed: dict | None = allowed_hkl
    elif geometry_only:
        _f2 = f2_thresh if f2_thresh is not None else F2_THRESHOLD
        _allowed = {
            id(p["crystal"]): precompute_allowed_hkl(p["crystal"], E_max_eV=E_max_eV, f2_thresh=_f2)
            for p in phases_work
        }
    else:
        _allowed = None

    _stages = (
        [float(max_match_px)] if np.isscalar(max_match_px)
        else [float(v) for v in max_match_px]
    )

    U0_stage = [U0.copy() for U0 in U0_list]
    opt = None

    try:
        for _si, _px in enumerate(_stages):
            fun = partial(
                laue_mixed_residuals,
                phases=phases_work, camera=camera, obs_xy=obs_use,
                U0_list=U0_stage, shared=shared,
                E_min_eV=E_min_eV, E_max_eV=E_max_eV,
                source=source, source_kwargs=source_kwargs,
                f2_thresh=f2_thresh, kb_params=kb_params,
                structure_model=structure_model,
                max_match_px=_px, top_n_obs=None, top_n_sim=top_n_sim,
                geometry_only=False, allowed_hkl=_allowed,
            )
            opt = least_squares(
                fun, x0=np.zeros(n_params),
                method=method, ftol=ftol, xtol=xtol, gtol=gtol, max_nfev=max_nfev,
            )
            if verbose and len(_stages) > 1:
                _nm, _rms, _ = _compute_match_stats(opt.fun, _px, N_obs)
                print(
                    f"  stage {_si + 1}/{len(_stages)}  px={_px:.1f}:"
                    f"  matched={_nm}  rms={_rms:.2f} px"
                )
            if _si < len(_stages) - 1:
                if shared:
                    R_step = Rotation.from_rotvec(opt.x).as_matrix()
                    U0_stage = [R_step @ U0 for U0 in U0_stage]
                else:
                    U0_stage = [
                        Rotation.from_rotvec(opt.x[3 * i : 3 * i + 3]).as_matrix() @ U0
                        for i, U0 in enumerate(U0_stage)
                    ]
    finally:
        # Restore original U matrices in the working copy.
        for p, U0 in zip(phases_work, U0_list):
            p["U"] = U0.copy()

    # Unpack refined orientations.
    params = opt.x
    if shared:
        R_last = Rotation.from_rotvec(params).as_matrix()
        U_phases_final = [R_last @ U0 for U0 in U0_stage]
        rotvecs = [
            Rotation.from_matrix(U_new @ U0_orig.T).as_rotvec()
            for U_new, U0_orig in zip(U_phases_final, U0_list)
        ]
    else:
        U_phases_final = []
        rotvecs = []
        for i, (U0_s, U0_orig) in enumerate(zip(U0_stage, U0_list)):
            rv = params[3 * i : 3 * i + 3]
            U_new = Rotation.from_rotvec(rv).as_matrix() @ U0_s
            U_phases_final.append(U_new)
            rotvecs.append(Rotation.from_matrix(U_new @ U0_orig.T).as_rotvec())

    if update_phases:
        for p_orig, p_work, U_new in zip(
            phases if isinstance(phases[0], dict) else [None] * N_phases,
            phases_work,
            U_phases_final,
        ):
            p_work["U"] = U_new.copy()
            if isinstance(p_orig, dict):
                p_orig["U"] = U_new.copy()

    n_matched, rms_px, mean_px = _compute_match_stats(opt.fun, _stages[-1], N_obs)

    # Final simulation for n_sim.
    for p, U_new in zip(phases_work, U_phases_final):
        p["U"] = U_new.copy()
    final_spots = simulate_mixed_phases(
        phases_work, camera,
        E_min_eV=E_min_eV, E_max_eV=E_max_eV,
        source=source, source_kwargs=source_kwargs,
        f2_thresh=f2_thresh, kb_params=kb_params,
        structure_model=structure_model, verbose=False,
        allowed_hkl=_allowed,
    )
    n_sim = len(_extract_sim_xy(final_spots))

    result = MixedFitResult(
        U_phases=U_phases_final, U0_phases=U0_list,
        rotvecs=rotvecs,
        cost=float(opt.cost), rms_px=rms_px, mean_px=mean_px,
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
    f2_thresh: float = F2_THRESHOLD,
    kb_params=BM32_KB,
    max_match_px: float | list[float] = (15.0, 3.0),
    top_n_obs: int | None = 300,
    top_n_sim: int | None = 300,
    method: str = "lm",
    ftol: float = 1e-8,
    xtol: float = 1e-8,
    gtol: float = 1e-8,
    max_nfev: int = 2000,
    geometry_only: bool = True,
    allowed_hkl=None,
    verbose: bool = False,
) -> StrainFitResult:
    """
    Simultaneously refine orientation and lattice strain for a single crystal.

    Wraps :func:`laue_strain_residuals` + `scipy.optimize.least_squares`.

    The effective orientation passed to the simulator is::

        U_eff = R(δω) @ U0 @ (I + ε)

    where δω is a small rotation increment and ε is the symmetric strain
    tensor.  Because `simulate_laue` accepts any 3×3 matrix, the strained
    d-spacings are naturally encoded in `U_eff`.

    Args:
        crystal (Crystal): xrayutilities crystal structure.
        camera (Camera): Detector geometry.
        obs_xy ((N_obs, 2)): Observed pixel positions `[xcam, ycam]`,
            sorted by descending intensity.
        U0 ((3, 3)): Starting orientation matrix (LT frame).
        fit_strain (tuple of str): Strain components to refine.  Any subset of
            `('e_xx','e_yy','e_zz','e_xy','e_xz','e_yz')`.
            Default: all six.  Pass a subset to fix
            symmetry constraints, e.g.
            `('e_xx', 'e_yy', 'e_zz')` for diagonal
            (biaxial) strain only.
        strain_scale (float): Internal scale for strain parameters (see
            :func:`laue_strain_residuals`).  Default 1e-4.
        E_min_eV, E_max_eV, source, source_kwargs, f2_thresh, kb_params,
        max_match_px, top_n_obs, top_n_sim, method, ftol, xtol, gtol, max_nfev,
        geometry_only
            Forwarded to `least_squares` / :func:`laue_strain_residuals`.
        verbose (bool): Print a one-line summary after convergence.

    Returns:
        StrainFitResult
            Call `str(result)` for a compact summary.  The refined effective
            matrix is `result.U_eff`; the pure rotation part is `result.U`.

    Note:
    *Scaling*: strain values for metals are typically 10⁻⁴–10⁻³.
    `strain_scale=1e-4` keeps all optimizer parameters near order-1,
    which is important for Levenberg–Marquardt whose finite-difference step
    is proportional to parameter magnitude.

    *Starting point*: it is usually best to first obtain a good orientation
    with :func:`fit_orientation` and then pass the result as `U0` here.
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
        allowed_hkl if allowed_hkl is not None
        else precompute_allowed_hkl(crystal, E_max_eV=E_max_eV, f2_thresh=f2_thresh)
        if geometry_only else None
    )

    _stages = (
        [float(max_match_px)] if np.isscalar(max_match_px)
        else [float(v) for v in max_match_px]
    )

    # ── staged refinement loop ────────────────────────────────────────────────
    # Between stages the rotation is baked into U0_stage and the strain from
    # the previous stage is carried forward as the starting guess for the next.
    # Only the rotation component of x0 is reset (it is now encoded in U0_stage).
    # Carrying strain prevents the tight final stage from collapsing into a
    # degenerate minimum when the correct strain already shifts spots by more
    # than the final match-radius.
    U0_stage = U0_arr.copy()
    opt = None
    _x0 = np.zeros(3 + n_strain)
    for _si, _px in enumerate(_stages):
        _fun = partial(
            laue_strain_residuals,
            crystal=crystal, camera=camera, obs_xy=obs_use, U0=U0_stage,
            fit_strain=fit_strain, strain_scale=strain_scale,
            E_min_eV=E_min_eV, E_max_eV=E_max_eV,
            source=source, source_kwargs=source_kwargs,
            f2_thresh=f2_thresh, kb_params=kb_params,
            max_match_px=_px, top_n_obs=None, top_n_sim=top_n_sim,
            geometry_only=False, allowed_hkl=_allowed,
        )
        opt = least_squares(
            _fun, x0=_x0,
            method=method, ftol=ftol, xtol=xtol, gtol=gtol, max_nfev=max_nfev,
        )
        if verbose and len(_stages) > 1:
            _nm, _rms, _ = _compute_match_stats(opt.fun, _px, N_obs)
            print(
                f"  stage {_si + 1}/{len(_stages)}  px={_px:.1f}:"
                f"  matched={_nm}  rms={_rms:.2f} px"
            )
        if _si < len(_stages) - 1:
            U0_stage = Rotation.from_rotvec(opt.x[:3]).as_matrix() @ U0_stage
            # Reset rotation (now baked into U0_stage); carry strain forward.
            _x0 = np.zeros(3 + n_strain)
            _x0[3:] = opt.x[3:]

    # Unpack solution.
    rotvec = opt.x[:3]
    strain_vals = opt.x[3:] * strain_scale
    R = Rotation.from_rotvec(rotvec).as_matrix()
    eps = _strain_matrix(strain_vals, fit_strain)
    U_final = R @ U0_stage
    U_eff = R @ U0_stage @ (np.eye(3) + eps)
    voigt = _strain_to_voigt(strain_vals, fit_strain)

    n_matched, rms_px, mean_px = _compute_match_stats(opt.fun, _stages[-1], N_obs)

    final_spots = simulate_laue(
        crystal, U_eff, camera,
        E_min=E_min_eV, E_max=E_max_eV,
        source=source, source_kwargs=source_kwargs,
        f2_thresh=f2_thresh, kb_params=kb_params,
        allowed_hkl=_allowed,
    )
    n_sim = len(_extract_sim_xy(final_spots))

    result = StrainFitResult(
        U=U_final, U0=U0_arr, U_eff=U_eff,
        rotvec=rotvec, strain_tensor=eps, strain_voigt=voigt,
        fit_strain=fit_strain,
        cost=float(opt.cost), rms_px=rms_px, mean_px=mean_px,
        n_matched=n_matched, n_obs=N_obs, n_sim=n_sim,
        match_rate=n_matched / max(N_obs, 1),
        success=opt.success, message=opt.message, optimizer=opt,
    )

    if verbose:
        print(f"  {result}")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Local (in-process) parallel fitting  –  mirrors SLURM worker pipelines
# ─────────────────────────────────────────────────────────────────────────────

# ── per-process globals set by pool initializers ──────────────────────────────
_g_crystal  = None
_g_camera   = None
_g_allowed  = None


def _local_pool_init(crystal_pkl_path: str, camera, allowed_hkl) -> None:
    global _g_crystal, _g_camera, _g_allowed
    with open(crystal_pkl_path, "rb") as fh:
        _g_crystal = _dill.load(fh)
    _g_camera  = camera
    _g_allowed = allowed_hkl


# ── orientation ───────────────────────────────────────────────────────────────

def _orient_process_frame(
    frame_idx: int,
    obs_xy: "np.ndarray | None",
    *,
    ubs_dir: str,
    ub_arrays: list,
    max_match_px,
    min_matched: int,
    min_match_rate: float,
    max_rms_px: "float | None",
    fit_kwargs: dict,
    overwrite: bool,
) -> tuple:
    """Process one frame for orientation fitting.  Returns (frame_idx, n_saved)."""
    if obs_xy is None or len(obs_xy) < min_matched:
        return frame_idx, 0

    crystal = _g_crystal
    camera  = _g_camera
    n_saved = 0

    for gi, U_ref in enumerate(ub_arrays):
        out_path = os.path.join(ubs_dir, f"frame_{frame_idx:05d}_g{gi:02d}.npz")
        if os.path.exists(out_path) and not overwrite:
            n_saved += 1
            continue

        try:
            result = fit_orientation(
                crystal, camera, obs_xy, U_ref,
                max_match_px=list(max_match_px),
                allowed_hkl=_g_allowed,
                **fit_kwargs,
            )
        except Exception as exc:
            print(f"  ✗  frame {frame_idx} g{gi}: fit: {exc}", flush=True)
            continue

        if result.n_matched < min_matched:
            continue
        if result.match_rate < min_match_rate:
            continue
        if max_rms_px is not None and result.rms_px > max_rms_px:
            continue

        tmp = out_path + ".tmp.npz"
        np.savez(
            tmp,
            U          = result.U,
            rotvec     = result.rotvec,
            rms_px     = np.array(result.rms_px),
            mean_px    = np.array(result.mean_px),
            n_matched  = np.array(result.n_matched),
            match_rate = np.array(result.match_rate),
            cost       = np.array(result.cost),
        )
        os.replace(tmp, out_path)
        n_saved += 1

    return frame_idx, n_saved


def run_orientation_local(
    crystal,
    camera,
    ub_arrays: list,
    seg_dir: str,
    ubs_dir: str,
    *,
    r_squared_min: float = 0.9,
    include_unfitted: bool = False,
    max_match_px=(30, 10, 3),
    min_matched: int = 5,
    min_match_rate: float = 0.2,
    max_rms_px: "float | None" = None,
    f2_thresh: float = 1e-4,
    geometry_only: bool = True,
    n_workers: "int | None" = None,
    overwrite: bool = False,
    frame_indices: "list | None" = None,
    **fit_kwargs,
) -> int:
    """Orientation fitting on local cores, mirroring the SLURM orient worker.

    Args:
        crystal (Crystal object.):
        camera (Camera object.):
        ub_arrays (List of (3,3) U matrices): one per grain reference.
        seg_dir (Directory with `frame_{idx:05d}.h5` peaklist files.):
        ubs_dir (Output directory; receives `frame_{idx:05d}_g{gi:02d}.npz`.):
        frame_indices (Subset of frames to process; defaults to all found in seg_dir.):

    Returns:
        n_saved (Total number of orientation files written.):
"""
    from .segmentation import convert_spotsfile2peaklist
    import glob

    os.makedirs(ubs_dir, exist_ok=True)

    if frame_indices is None:
        h5s = sorted(glob.glob(os.path.join(seg_dir, "frame_?????.h5")))
        frame_indices = [int(os.path.basename(p)[6:11]) for p in h5s]

    # Precompute allowed HKL once.
    t_hkl = time.time()
    allowed_hkl = None
    if geometry_only:
        _E_max = fit_kwargs.get("E_max_eV", E_MAX_eV)
        allowed_hkl = precompute_allowed_hkl(crystal, E_max_eV=_E_max, f2_thresh=f2_thresh)
        print(
            f"  allowed_hkl: {len(allowed_hkl)} reflections "
            f"({time.time() - t_hkl:.1f}s)",
            flush=True,
        )

    # Serialize crystal to a temp file so the pool initializer can load it.
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tf:
        crystal_pkl = tf.name
    try:
        with open(crystal_pkl, "wb") as fh:
            _dill.dump(crystal, fh)

        # Load peaklists.
        t_io = time.time()
        peaklists: dict[int, np.ndarray] = {}
        for fi in frame_indices:
            seg_path = os.path.join(seg_dir, f"frame_{fi:05d}.h5")
            if not os.path.exists(seg_path):
                continue
            try:
                pl = convert_spotsfile2peaklist(
                    seg_path,
                    r_squared_min=r_squared_min,
                    include_unfitted=include_unfitted,
                )
                if len(pl) >= min_matched:
                    peaklists[fi] = pl[:, :2]
            except Exception as exc:
                print(f"  ✗  frame {fi}: load peaklist: {exc}", flush=True)

        n_total = len(frame_indices)
        print(
            f"Orient local — {n_total} frames | {len(ub_arrays)} UB(s) | "
            f"{len(peaklists)} with enough spots | "
            f"I/O: {time.time() - t_io:.1f}s",
            flush=True,
        )

        _fk = dict(fit_kwargs)
        if geometry_only:
            _fk["geometry_only"] = True

        common = dict(
            ubs_dir        = ubs_dir,
            ub_arrays      = ub_arrays,
            max_match_px   = max_match_px,
            min_matched    = min_matched,
            min_match_rate = min_match_rate,
            max_rms_px     = max_rms_px,
            fit_kwargs     = _fk,
            overwrite      = overwrite,
        )
        n_workers = min(len(peaklists) or 1, n_workers or os.cpu_count() or 1)
        tick = max(1, n_total // 20)

        t0 = time.time()
        n_saved = 0
        done = 0
        with ProcessPoolExecutor(
            max_workers = n_workers,
            initializer = _local_pool_init,
            initargs    = (crystal_pkl, camera, allowed_hkl),
        ) as pool:
            futs = {
                pool.submit(_orient_process_frame, fi, peaklists.get(fi), **common): fi
                for fi in frame_indices
            }
            for fut in as_completed(futs):
                done += 1
                try:
                    _, n = fut.result()
                    n_saved += n
                except Exception as exc:
                    print(f"  ✗  frame {futs[fut]}: {exc}", flush=True)

                if done % tick == 0 or done == n_total:
                    elapsed = time.time() - t0
                    rate = done / elapsed if elapsed > 0 else float("inf")
                    eta  = (n_total - done) / rate if rate > 0 else float("inf")
                    print(
                        f"  {done}/{n_total}  {n_saved} fits saved  "
                        f"{elapsed:.0f}s elapsed  ETA {eta:.0f}s",
                        flush=True,
                    )

    finally:
        os.unlink(crystal_pkl)

    print(
        f"\nOrient local done — {n_saved} fits saved  "
        f"({time.time() - t0:.1f}s total)",
        flush=True,
    )
    return n_saved


# ── strain ────────────────────────────────────────────────────────────────────

def _strain_process_frame(
    frame_idx: int,
    obs_xy: "np.ndarray | None",
    *,
    ubs_dir: str,
    strain_dir: str,
    n_grains: int,
    max_match_px,
    fit_strain: tuple,
    fit_kwargs: dict,
    overwrite: bool,
) -> tuple:
    """Process one frame for strain fitting.  Returns (frame_idx, n_saved)."""
    if obs_xy is None or len(obs_xy) < 3:
        return frame_idx, 0

    crystal = _g_crystal
    camera  = _g_camera
    n_saved = 0

    for gi in range(n_grains):
        orient_path = os.path.join(ubs_dir, f"frame_{frame_idx:05d}_g{gi:02d}.npz")
        if not os.path.exists(orient_path):
            continue

        out_path = os.path.join(strain_dir, f"frame_{frame_idx:05d}_g{gi:02d}.npz")
        if os.path.exists(out_path) and not overwrite:
            n_saved += 1
            continue

        try:
            U0 = np.load(orient_path)["U"]
            result = fit_strain_orientation(
                crystal, camera, obs_xy, U0,
                max_match_px = list(max_match_px),
                fit_strain   = fit_strain,
                allowed_hkl  = _g_allowed,
                **fit_kwargs,
            )

            tmp = out_path + ".tmp.npz"
            np.savez(
                tmp,
                U             = result.U,
                U_eff         = result.U_eff,
                strain_tensor = result.strain_tensor,
                strain_voigt  = result.strain_voigt,
                rotvec        = result.rotvec,
                rms_px        = np.array(result.rms_px),
                mean_px       = np.array(result.mean_px),
                n_matched     = np.array(result.n_matched),
                match_rate    = np.array(result.match_rate),
                cost          = np.array(result.cost),
            )
            os.replace(tmp, out_path)
            n_saved += 1

        except Exception as exc:
            print(f"  ✗  frame {frame_idx} g{gi}: strain fit: {exc}", flush=True)

    return frame_idx, n_saved


def run_strain_local(
    crystal,
    camera,
    seg_dir: str,
    ubs_dir: str,
    strain_dir: str,
    n_grains: int,
    *,
    fit_strain=("e_xx", "e_yy", "e_zz", "e_xy", "e_xz", "e_yz"),
    r_squared_min: float = 0.9,
    include_unfitted: bool = False,
    max_match_px=(10, 3),
    f2_thresh: float = 1e-4,
    geometry_only: bool = True,
    n_workers: "int | None" = None,
    overwrite: bool = False,
    frame_indices: "list | None" = None,
    **fit_kwargs,
) -> int:
    """Strain fitting on local cores, mirroring the SLURM strain worker.

    Args:
        crystal (Crystal object.):
        camera (Camera object.):
        seg_dir (Directory with `frame_{idx:05d}.h5` peaklist files.):
        ubs_dir (Directory with orientation `frame_{idx:05d}_g{gi:02d}.npz` files.):
        strain_dir (Output directory; receives `frame_{idx:05d}_g{gi:02d}.npz`.):
        n_grains (Number of grains (grain indices 0 … n_grains-1).):
        frame_indices (Subset of frames to process; defaults to all found in seg_dir.):

    Returns:
        n_saved (Total number of strain files written.):
"""
    from .segmentation import convert_spotsfile2peaklist
    import glob

    os.makedirs(strain_dir, exist_ok=True)

    if frame_indices is None:
        h5s = sorted(glob.glob(os.path.join(seg_dir, "frame_?????.h5")))
        frame_indices = [int(os.path.basename(p)[6:11]) for p in h5s]

    # Precompute allowed HKL once.
    t_hkl = time.time()
    allowed_hkl = None
    if geometry_only:
        _E_max = fit_kwargs.get("E_max_eV", E_MAX_eV)
        allowed_hkl = precompute_allowed_hkl(crystal, E_max_eV=_E_max, f2_thresh=f2_thresh)
        print(
            f"  allowed_hkl: {len(allowed_hkl)} reflections "
            f"({time.time() - t_hkl:.1f}s)",
            flush=True,
        )

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tf:
        crystal_pkl = tf.name
    try:
        with open(crystal_pkl, "wb") as fh:
            _dill.dump(crystal, fh)

        # Load peaklists.
        t_io = time.time()
        peaklists: dict[int, np.ndarray] = {}
        for fi in frame_indices:
            seg_path = os.path.join(seg_dir, f"frame_{fi:05d}.h5")
            if not os.path.exists(seg_path):
                continue
            try:
                pl = convert_spotsfile2peaklist(
                    seg_path,
                    r_squared_min=r_squared_min,
                    include_unfitted=include_unfitted,
                )
                if len(pl) >= 3:
                    peaklists[fi] = pl[:, :2]
            except Exception as exc:
                print(f"  ✗  frame {fi}: load peaklist: {exc}", flush=True)

        n_total = len(frame_indices)
        print(
            f"Strain local — {n_total} frames | {n_grains} grain(s) | "
            f"{len(peaklists)} with spots | "
            f"fit_strain={list(fit_strain)} | "
            f"I/O: {time.time() - t_io:.1f}s",
            flush=True,
        )

        _fk = dict(fit_kwargs)
        if geometry_only:
            _fk["geometry_only"] = True

        common = dict(
            ubs_dir      = ubs_dir,
            strain_dir   = strain_dir,
            n_grains     = n_grains,
            max_match_px = max_match_px,
            fit_strain   = tuple(fit_strain),
            fit_kwargs   = _fk,
            overwrite    = overwrite,
        )
        n_workers = min(len(peaklists) or 1, n_workers or os.cpu_count() or 1)
        tick = max(1, n_total // 20)

        t0 = time.time()
        n_saved = 0
        done = 0
        with ProcessPoolExecutor(
            max_workers = n_workers,
            initializer = _local_pool_init,
            initargs    = (crystal_pkl, camera, allowed_hkl),
        ) as pool:
            futs = {
                pool.submit(_strain_process_frame, fi, peaklists.get(fi), **common): fi
                for fi in frame_indices
            }
            for fut in as_completed(futs):
                done += 1
                try:
                    _, n = fut.result()
                    n_saved += n
                except Exception as exc:
                    print(f"  ✗  frame {futs[fut]}: {exc}", flush=True)

                if done % tick == 0 or done == n_total:
                    elapsed = time.time() - t0
                    rate = done / elapsed if elapsed > 0 else float("inf")
                    eta  = (n_total - done) / rate if rate > 0 else float("inf")
                    print(
                        f"  {done}/{n_total}  {n_saved} fits saved  "
                        f"{elapsed:.0f}s elapsed  ETA {eta:.0f}s",
                        flush=True,
                    )

    finally:
        os.unlink(crystal_pkl)

    print(
        f"\nStrain local done — {n_saved} fits saved  "
        f"({time.time() - t0:.1f}s total)",
        flush=True,
    )
    return n_saved


# ── mixed-phase orientation ───────────────────────────────────────────────────

_g_crystals     = None
_g_allowed_list = None   # list of allowed_hkl arrays, one per crystal


def _mixed_pool_init(crystals_pkl: str, camera, allowed_hkl_list) -> None:
    global _g_crystals, _g_camera, _g_allowed_list
    with open(crystals_pkl, "rb") as fh:
        _g_crystals = _dill.load(fh)
    _g_camera        = camera
    _g_allowed_list  = allowed_hkl_list


def _mixed_process_frame(
    frame_idx: int,
    obs_xy: "np.ndarray | None",
    *,
    mixed_dir: str,
    ub_arrays: list,
    shared: bool,
    max_match_px,
    min_matched: int,
    min_match_rate: float,
    max_rms_px: "float | None",
    fit_kwargs: dict,
    overwrite: bool,
) -> tuple:
    """Process one frame for mixed-phase orientation fitting.  Returns (frame_idx, ok)."""
    if obs_xy is None or len(obs_xy) < min_matched:
        return frame_idx, False

    out_path = os.path.join(mixed_dir, f"frame_{frame_idx:05d}.npz")
    if os.path.exists(out_path) and not overwrite:
        return frame_idx, True

    crystals = _g_crystals
    camera   = _g_camera

    # Rebuild allowed_hkl dict keyed by process-local id(crystal).
    allowed = None
    if _g_allowed_list is not None:
        allowed = {id(c): a for c, a in zip(crystals, _g_allowed_list)}

    stages = max_match_px if isinstance(max_match_px, (list, tuple)) else [max_match_px]

    phases = [
        {"crystal": c, "U": np.asarray(U, dtype=float).copy(),
         "volume_fraction": 1.0 / len(crystals)}
        for c, U in zip(crystals, ub_arrays)
    ]

    try:
        result = None
        for px in stages:
            result = fit_orientation_mixed(
                phases, camera, obs_xy,
                shared=shared,
                max_match_px=float(px),
                geometry_only=False,
                allowed_hkl=allowed,
                update_phases=True,   # phases[i]["U"] updated in-place for next stage
                **fit_kwargs,
            )
    except Exception as exc:
        print(f"  ✗  frame {frame_idx}: mixed fit: {exc}", flush=True)
        return frame_idx, False

    if result.n_matched < min_matched:
        return frame_idx, False
    if result.match_rate < min_match_rate:
        return frame_idx, False
    if max_rms_px is not None and result.rms_px > max_rms_px:
        return frame_idx, False

    save_dict = {
        "rms_px":     np.array(result.rms_px),
        "mean_px":    np.array(result.mean_px),
        "n_matched":  np.array(result.n_matched),
        "match_rate": np.array(result.match_rate),
        "cost":       np.array(result.cost),
    }
    for i, (U, rv) in enumerate(zip(result.U_phases, result.rotvecs)):
        save_dict[f"U_{i}"]      = U
        save_dict[f"rotvec_{i}"] = rv

    tmp = out_path + ".tmp.npz"
    np.savez(tmp, **save_dict)
    os.replace(tmp, out_path)
    return frame_idx, True


def run_orientation_mixed_local(
    crystals: list,
    camera,
    ub_arrays: list,
    seg_dir: str,
    mixed_dir: str,
    *,
    shared: bool = False,
    r_squared_min: float = 0.9,
    include_unfitted: bool = False,
    max_match_px=(30, 10, 3),
    min_matched: int = 5,
    min_match_rate: float = 0.2,
    max_rms_px: "float | None" = None,
    f2_thresh: float = 1e-4,
    geometry_only: bool = True,
    n_workers: "int | None" = None,
    overwrite: bool = False,
    frame_indices: "list | None" = None,
    **fit_kwargs,
) -> int:
    """Mixed-phase orientation fitting on local cores.

    Fits all phases simultaneously for each frame using
    :func:`fit_orientation_mixed`, sharing the detector image and observed
    spot list across all phases.

    Args:
        crystals (list of Crystal): One crystal per phase, in grain-index order.
        camera (Camera):
        ub_arrays (list of (3,3)): Reference U matrix per phase (same order as crystals).
        seg_dir (str): Directory with ``frame_{idx:05d}.h5`` peaklist files.
        mixed_dir (str): Output directory; one ``frame_{idx:05d}.npz`` per frame
            containing ``U_0``, ``U_1``, … and shared quality metrics.
        shared (bool): If True, one shared rotation for all phases.
            If False (default), independent rotation per phase.
        frame_indices (list or None): Subset of frames; defaults to all found in seg_dir.

    Returns:
        n_saved (int): Number of frames successfully fitted and written.
"""
    from .segmentation import convert_spotsfile2peaklist
    import glob

    if len(crystals) != len(ub_arrays):
        raise ValueError("crystals and ub_arrays must have the same length")

    os.makedirs(mixed_dir, exist_ok=True)

    if frame_indices is None:
        h5s = sorted(glob.glob(os.path.join(seg_dir, "frame_?????.h5")))
        frame_indices = [int(os.path.basename(p)[6:11]) for p in h5s]

    # Precompute allowed HKL once per crystal.
    t_hkl = time.time()
    allowed_hkl_list = None
    if geometry_only:
        allowed_hkl_list = [
            precompute_allowed_hkl(c, f2_thresh=f2_thresh)
            for c in crystals
        ]
        n_refs = sum(len(a) for a in allowed_hkl_list)
        print(
            f"  allowed_hkl: {n_refs} reflections total across {len(crystals)} phases "
            f"({time.time() - t_hkl:.1f}s)",
            flush=True,
        )

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tf:
        crystals_pkl = tf.name
    try:
        with open(crystals_pkl, "wb") as fh:
            _dill.dump(crystals, fh)

        t_io = time.time()
        peaklists: dict[int, np.ndarray] = {}
        for fi in frame_indices:
            seg_path = os.path.join(seg_dir, f"frame_{fi:05d}.h5")
            if not os.path.exists(seg_path):
                continue
            try:
                pl = convert_spotsfile2peaklist(
                    seg_path,
                    r_squared_min=r_squared_min,
                    include_unfitted=include_unfitted,
                )
                if len(pl) >= min_matched:
                    peaklists[fi] = pl[:, :2]
            except Exception as exc:
                print(f"  ✗  frame {fi}: load peaklist: {exc}", flush=True)

        n_total = len(frame_indices)
        print(
            f"Mixed local — {n_total} frames | {len(crystals)} phase(s) | "
            f"{len(peaklists)} with enough spots | "
            f"shared={shared} | I/O: {time.time() - t_io:.1f}s",
            flush=True,
        )

        _fk = dict(fit_kwargs)

        common = dict(
            mixed_dir      = mixed_dir,
            ub_arrays      = ub_arrays,
            shared         = shared,
            max_match_px   = max_match_px,
            min_matched    = min_matched,
            min_match_rate = min_match_rate,
            max_rms_px     = max_rms_px,
            fit_kwargs     = _fk,
            overwrite      = overwrite,
        )
        n_workers = min(len(peaklists) or 1, n_workers or os.cpu_count() or 1)
        tick = max(1, n_total // 20)

        t0 = time.time()
        n_saved = 0
        done = 0
        with ProcessPoolExecutor(
            max_workers = n_workers,
            initializer = _mixed_pool_init,
            initargs    = (crystals_pkl, camera, allowed_hkl_list),
        ) as pool:
            futs = {
                pool.submit(_mixed_process_frame, fi, peaklists.get(fi), **common): fi
                for fi in frame_indices
            }
            for fut in as_completed(futs):
                done += 1
                try:
                    _, ok = fut.result()
                    n_saved += ok
                except Exception as exc:
                    print(f"  ✗  frame {futs[fut]}: {exc}", flush=True)

                if done % tick == 0 or done == n_total:
                    elapsed = time.time() - t0
                    rate = done / elapsed if elapsed > 0 else float("inf")
                    eta  = (n_total - done) / rate if rate > 0 else float("inf")
                    print(
                        f"  {done}/{n_total}  {n_saved} fits saved  "
                        f"{elapsed:.0f}s elapsed  ETA {eta:.0f}s",
                        flush=True,
                    )

    finally:
        os.unlink(crystals_pkl)

    print(
        f"\nMixed local done — {n_saved}/{n_total} frames fitted  "
        f"({time.time() - t0:.1f}s total)",
        flush=True,
    )
    return n_saved


# ── mixed-phase strain ────────────────────────────────────────────────────────

def _mixed_strain_process_frame(
    frame_idx: int,
    obs_xy: "np.ndarray | None",
    *,
    mixed_dir: str,
    strain_dir: str,
    n_grains: int,
    max_match_px,
    fit_strain: tuple,
    fit_kwargs: dict,
    overwrite: bool,
) -> tuple:
    """Per-phase strain fitting starting from mixed orientation results.  Returns (frame_idx, n_saved)."""
    if obs_xy is None or len(obs_xy) < 3:
        return frame_idx, 0

    mixed_path = os.path.join(mixed_dir, f"frame_{frame_idx:05d}.npz")
    if not os.path.exists(mixed_path):
        return frame_idx, 0

    d_mixed  = np.load(mixed_path)
    crystals = _g_crystals
    camera   = _g_camera
    n_saved  = 0

    for gi in range(n_grains):
        u_key = f"U_{gi}"
        if u_key not in d_mixed:
            continue

        out_path = os.path.join(strain_dir, f"frame_{frame_idx:05d}_g{gi:02d}.npz")
        if os.path.exists(out_path) and not overwrite:
            n_saved += 1
            continue

        crystal = crystals[gi]
        allowed = _g_allowed_list[gi] if _g_allowed_list is not None else None

        try:
            U0     = d_mixed[u_key]
            result = fit_strain_orientation(
                crystal, camera, obs_xy, U0,
                max_match_px = list(max_match_px),
                fit_strain   = fit_strain,
                allowed_hkl  = allowed,
                **fit_kwargs,
            )

            tmp = out_path + ".tmp.npz"
            np.savez(
                tmp,
                U             = result.U,
                U_eff         = result.U_eff,
                strain_tensor = result.strain_tensor,
                strain_voigt  = result.strain_voigt,
                rotvec        = result.rotvec,
                rms_px        = np.array(result.rms_px),
                mean_px       = np.array(result.mean_px),
                n_matched     = np.array(result.n_matched),
                match_rate    = np.array(result.match_rate),
                cost          = np.array(result.cost),
            )
            os.replace(tmp, out_path)
            n_saved += 1

        except Exception as exc:
            print(f"  ✗  frame {frame_idx} g{gi}: mixed strain fit: {exc}", flush=True)

    return frame_idx, n_saved


def run_strain_mixed_local(
    crystals: list,
    camera,
    seg_dir: str,
    mixed_dir: str,
    strain_dir: str,
    n_grains: int,
    *,
    fit_strain=("e_xx", "e_yy", "e_zz", "e_xy", "e_xz", "e_yz"),
    r_squared_min: float = 0.9,
    include_unfitted: bool = False,
    max_match_px=(10, 3),
    f2_thresh: float = 1e-4,
    geometry_only: bool = True,
    n_workers: "int | None" = None,
    overwrite: bool = False,
    frame_indices: "list | None" = None,
    **fit_kwargs,
) -> int:
    """Per-phase strain fitting on local cores, starting from mixed orientation results.

    Reads ``mixed_dir/frame_{idx:05d}.npz`` (written by
    :func:`run_orientation_mixed_local`) for the starting U of each phase,
    then calls :func:`fit_strain_orientation` independently per phase.
    Output is written to ``strain_dir/frame_{idx:05d}_g{gi:02d}.npz`` — the
    same format as :func:`run_strain_local` — so :meth:`GrainMap.collect_strain`
    works unchanged.

    Args:
        crystals (list of Crystal): One crystal per phase, in grain-index order.
        camera (Camera):
        seg_dir (str): Directory with ``frame_{idx:05d}.h5`` peaklist files.
        mixed_dir (str): Directory with mixed orientation ``frame_{idx:05d}.npz`` files.
        strain_dir (str): Output directory for strain results.
        n_grains (int): Number of phases (grain indices 0 … n_grains-1).
        frame_indices (list or None): Subset of frames; defaults to all found in mixed_dir.

    Returns:
        n_saved (int): Total number of per-phase strain files written.
"""
    from .segmentation import convert_spotsfile2peaklist
    import glob

    if len(crystals) != n_grains:
        raise ValueError(f"len(crystals)={len(crystals)} must equal n_grains={n_grains}")

    os.makedirs(strain_dir, exist_ok=True)

    if frame_indices is None:
        npzs = sorted(glob.glob(os.path.join(mixed_dir, "frame_?????.npz")))
        frame_indices = [int(os.path.basename(p)[6:11]) for p in npzs]

    # Precompute allowed HKL once per crystal.
    t_hkl = time.time()
    allowed_hkl_list = None
    if geometry_only:
        allowed_hkl_list = [
            precompute_allowed_hkl(c, f2_thresh=f2_thresh)
            for c in crystals
        ]
        n_refs = sum(len(a) for a in allowed_hkl_list)
        print(
            f"  allowed_hkl: {n_refs} reflections across {n_grains} phases "
            f"({time.time() - t_hkl:.1f}s)",
            flush=True,
        )

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tf:
        crystals_pkl = tf.name
    try:
        with open(crystals_pkl, "wb") as fh:
            _dill.dump(crystals, fh)

        t_io = time.time()
        peaklists: dict[int, np.ndarray] = {}
        for fi in frame_indices:
            seg_path = os.path.join(seg_dir, f"frame_{fi:05d}.h5")
            if not os.path.exists(seg_path):
                continue
            try:
                pl = convert_spotsfile2peaklist(
                    seg_path,
                    r_squared_min=r_squared_min,
                    include_unfitted=include_unfitted,
                )
                if len(pl) >= 3:
                    peaklists[fi] = pl[:, :2]
            except Exception as exc:
                print(f"  ✗  frame {fi}: load peaklist: {exc}", flush=True)

        n_total = len(frame_indices)
        print(
            f"Mixed strain local — {n_total} frames | {n_grains} phase(s) | "
            f"{len(peaklists)} with spots | "
            f"fit_strain={list(fit_strain)} | "
            f"I/O: {time.time() - t_io:.1f}s",
            flush=True,
        )

        _fk = dict(fit_kwargs)
        if geometry_only:
            _fk["geometry_only"] = True

        common = dict(
            mixed_dir    = mixed_dir,
            strain_dir   = strain_dir,
            n_grains     = n_grains,
            max_match_px = max_match_px,
            fit_strain   = tuple(fit_strain),
            fit_kwargs   = _fk,
            overwrite    = overwrite,
        )
        n_workers = min(len(peaklists) or 1, n_workers or os.cpu_count() or 1)
        tick = max(1, n_total // 20)

        t0 = time.time()
        n_saved = 0
        done = 0
        with ProcessPoolExecutor(
            max_workers = n_workers,
            initializer = _mixed_pool_init,
            initargs    = (crystals_pkl, camera, allowed_hkl_list),
        ) as pool:
            futs = {
                pool.submit(_mixed_strain_process_frame, fi, peaklists.get(fi), **common): fi
                for fi in frame_indices
            }
            for fut in as_completed(futs):
                done += 1
                try:
                    _, n = fut.result()
                    n_saved += n
                except Exception as exc:
                    print(f"  ✗  frame {futs[fut]}: {exc}", flush=True)

                if done % tick == 0 or done == n_total:
                    elapsed = time.time() - t0
                    rate = done / elapsed if elapsed > 0 else float("inf")
                    eta  = (n_total - done) / rate if rate > 0 else float("inf")
                    print(
                        f"  {done}/{n_total}  {n_saved} fits saved  "
                        f"{elapsed:.0f}s elapsed  ETA {eta:.0f}s",
                        flush=True,
                    )

    finally:
        os.unlink(crystals_pkl)

    print(
        f"\nMixed strain local done — {n_saved} fits saved  "
        f"({time.time() - t0:.1f}s total)",
        flush=True,
    )
    return n_saved


# ─────────────────────────────────────────────────────────────────────────────
# Image-based orientation refinement
# ─────────────────────────────────────────────────────────────────────────────


def _fft_gauss_convolve(arr: np.ndarray, sigma: float) -> np.ndarray:
    """FFT-based Gaussian convolution (O(N log N), same kernel as gaussian_background)."""
    f = _sp_fft.fft2(arr, workers=-1)
    _ndi.fourier_gaussian(f, sigma=sigma, output=f)
    return _sp_fft.ifft2(f, workers=-1).real


def refine_orientation_image(
    crystal,
    U0: np.ndarray,
    camera,
    image: np.ndarray,
    *,
    kernel_sigma: float = 0.3,
    bg_sigma: float = 251.0,
    E_min: float = E_MIN_eV,
    E_max: float = E_MAX_eV,
    allowed_hkl=None,
    max_angle_deg: float = 0.2,
    method: str = "Powell",
    options: dict | None = None,
    verbose: bool = False,
) -> "ImageRefinementResult":
    """
    Post-refine an orientation matrix by maximising the total Gaussian-weighted
    pixel intensity at simulated Laue spot positions.

    Unlike the spot-matching refinement in :func:`fit_orientation`, this
    function works directly on the raw detector image and does not require a
    segmented peak list.  It is therefore useful for secondary grains or
    weak reflections that are not reliably captured by the segmentation step.

    **Objective function**

    For a candidate rotation vector δω the orientation is::

        U = Rotation.from_rotvec(δω) @ U0

    The score is computed with a single FFT per optimizer iteration,
    regardless of the number of simulated spots:

    1. Simulate Laue spots with ``geometry_only=True`` (fast — structure
       factors skipped; pass a pre-computed *allowed_hkl* to apply the F²
       threshold without the per-call overhead).
    2. Build a *delta map* — an image of zeros with each simulated pixel
       position incremented by the predicted spot intensity (gap pixels
       skipped).  Intensity weighting ensures that strong reflections
       dominate the objective, reducing sensitivity to weak background
       features.
    3. Convolve the delta map with a Gaussian kernel (σ = *kernel_sigma*)
       via FFT — each point becomes a smooth blob of radius ≈ 3σ.
    4. Compute the element-wise product with the background-subtracted
       detector image and sum.

    Maximising this sum pulls the simulated positions toward bright detector
    regions, refining the orientation.

    **Background subtraction**

    A large-σ FFT Gaussian (same routine as
    :func:`~.segmentation.gaussian_background`) is subtracted from the
    image **once before the optimisation loop**, so it adds no per-call
    overhead.  It removes the slowly-varying detector pedestal (beam-centre
    falloff, inter-module offsets) while leaving Bragg peaks intact.
    Set *bg_sigma=0* to skip.

    **Kernel size and spot confusion**

    *kernel_sigma* controls how spatially selective the objective is:

    * Large σ (e.g. 5 px) — smooth landscape, easier convergence, but
      spots within ~3σ of each other are not resolved.  The optimizer may
      drift toward brighter nearby spots.
    * Small σ (e.g. 0.3 px, default) — tight, position-sensitive objective.
      At σ = 0.3 the Gaussian weight at 2 px distance is < 0.001, so spots
      2 px apart are effectively decoupled.  Recommended when refining
      secondary grains whose spots are close to and dimmer than primary
      grain spots.

    **always pass** *allowed_hkl* (output of :func:`precompute_allowed_hkl`)
    to avoid recomputing the HKL list on every optimizer call and to ensure
    that systematically absent reflections are excluded from the delta map.

    Args:
        crystal: Crystal object passed to :func:`simulate_laue`.
        U0 ((3,3) ndarray): Starting orientation matrix (LT frame).  A
            good starting point (e.g. from LaueTools indexing) is important
            because the search space is intentionally narrow.
        camera (Camera): Detector geometry.
        image ((ny, nx) ndarray): Raw detector frame.  Gap / invalid pixels
            must be flagged as negative (Eiger convention: −1).
        kernel_sigma (float): σ in pixels of the Gaussian kernel placed at
            each simulated spot position.  Smaller values give a more
            position-sensitive objective and reduce cross-talk between spots
            separated by only a few pixels.  Default ``0.3``.
        bg_sigma (float): σ in pixels of the large-scale background
            Gaussian subtracted before optimisation.  Set to ``0`` to skip.
            Default ``251``.
        E_min, E_max (float): Photon energy range in eV forwarded to
            :func:`simulate_laue`.
        allowed_hkl: Pre-computed allowed HKL frozenset from
            :func:`precompute_allowed_hkl`.  Strongly recommended — without
            it :func:`simulate_laue` recomputes the full HKL list on every
            optimizer call, which is slow and includes systematically absent
            reflections that inflate the delta map.
        max_angle_deg (float): Symmetric bound (degrees) applied to each
            component of the rotation vector δω.  Restricts the search
            space to a ±*max_angle_deg* cube in rotation-vector space.
            Keep small (0.1–0.5°) when the starting U is already close to
            the true orientation.  Default ``0.2``.
        method (str): ``scipy.optimize.minimize`` method.  ``'Powell'``
            (default) is derivative-free and handles the smooth Gaussian
            landscape well.  ``'L-BFGS-B'`` can converge faster when
            many iterations are needed.
        options (dict or None): Forwarded to ``scipy.optimize.minimize`` as
            the ``options`` keyword, merged over sensible defaults
            (``maxiter=2000``, ``xtol/ftol=1e-6`` for Powell).
        verbose (bool): If ``True``, print the starting score and a
            one-line result summary on completion.  Default ``False``.

    Returns:
        ImageRefinementResult
    """
    ny, nx = image.shape
    valid = image >= 0
    img = image.astype(np.float64)
    img[~valid] = 0.0

    # Background subtraction — computed once, outside the optimisation loop.
    if bg_sigma > 0:
        smooth = _fft_gauss_convolve(img, bg_sigma)
        norm   = _fft_gauss_convolve(valid.astype(np.float64), bg_sigma)
        norm[norm < 1e-6] = 1.0
        img = np.clip(img - smooth / norm, 0.0, None)
        img[~valid] = 0.0

    lim = float(np.radians(max_angle_deg))
    bounds = [(-lim, lim)] * 3

    def _score(rotvec: np.ndarray) -> float:
        U = Rotation.from_rotvec(rotvec).as_matrix() @ U0
        spots = simulate_laue(
            crystal, U, camera,
            E_min=E_min, E_max=E_max,
            allowed_hkl=allowed_hkl,
            geometry_only=True,
        )
        if not spots:
            return 0.0

        # Delta map: 1.0 at each valid simulated pixel position.
        delta = np.zeros((ny, nx), dtype=np.float64)
        for s in spots:
            xc, yc = s["pix"]          # pix = (col, row) in detector coords
            col = int(round(xc))
            row = int(round(yc))
            if 0 <= row < ny and 0 <= col < nx and valid[row, col]:
                delta[row, col] += float(s["intensity"])

        # Convolve delta map with Gaussian → each spot becomes a smooth blob.
        kernel_map = _fft_gauss_convolve(delta, kernel_sigma)

        return float(np.sum(kernel_map * img))

    def _objective(rotvec: np.ndarray) -> float:
        return -_score(rotvec)  # minimise → maximise score

    score0 = _score(np.zeros(3))

    if verbose:
        print(f"refine_orientation_image: score0={score0:.1f}  method={method}  max_angle={max_angle_deg}°")

    opts: dict = {"maxiter": 2000}
    if method == "Powell":
        opts.update({"xtol": 1e-6, "ftol": 1e-6})
    else:
        opts.update({"gtol": 1e-7})
    if options:
        opts.update(options)

    result = minimize(_objective, np.zeros(3), method=method,
                      bounds=bounds, options=opts)

    rotvec_opt = result.x
    U_refined  = Rotation.from_rotvec(rotvec_opt).as_matrix() @ U0

    spots_final = simulate_laue(
        crystal, U_refined, camera,
        E_min=E_min, E_max=E_max,
        allowed_hkl=allowed_hkl,
        geometry_only=True,
    )

    out = ImageRefinementResult(
        U        = U_refined,
        U0       = U0,
        rotvec   = rotvec_opt,
        score    = -float(result.fun),
        score0   = score0,
        n_sim    = len(spots_final),
        success  = result.success,
        message  = result.message,
        optimizer= result,
    )

    if verbose:
        print(f"  {out}")

    return out


def refine_orientation_image_stack(
    stack,
    camera,
    image: np.ndarray,
    *,
    kernel_sigma: float = 0.3,
    bg_sigma: float = 251.0,
    E_min: float = E_MIN_eV,
    E_max: float = E_MAX_eV,
    allowed_hkl=None,
    max_angle_deg: float = 0.2,
    structure_model: str = "average",
    method: str = "Powell",
    options: "dict | None" = None,
    update_stack: bool = False,
    correct_depth: bool = False,
    verbose: bool = False,
) -> StackImageRefinementResult:
    """
    Post-refine the orientation of a
    :class:`~nrxrdct.laue.layers.LayeredCrystal` by maximising the total
    Gaussian-weighted pixel intensity at simulated Laue spot positions.

    Extends :func:`refine_orientation_image` to the layered-crystal case.
    A single global rotation is applied to all layers, preserving inter-layer
    orientation relationships::

        U_i = R(δω) @ U0_i   for all layers i

    The image-based objective requires no segmented peak list, making it
    useful as a polishing step after :func:`fit_orientation_stack`.

    Args:
        stack (LayeredCrystal): Layered structure.  Layer U matrices are used
            as starting orientations and always restored if an exception
            occurs.
        camera (Camera): Detector geometry.
        image ((ny, nx) ndarray): Raw detector frame.  Invalid / gap pixels
            must be negative (Eiger convention: ``−1``).
        kernel_sigma (float): σ (pixels) of the Gaussian placed at each
            simulated spot.  Default ``0.3``.
        bg_sigma (float): σ (pixels) of the background subtracted before
            optimisation.  ``0`` to skip.  Default ``251``.
        E_min, E_max (float): Photon energy range (eV).
        allowed_hkl: Pre-computed allowed HKL dict keyed by ``id(crystal)``,
            from :func:`precompute_allowed_hkl`.  Strongly recommended.
        max_angle_deg (float): Symmetric bound on each rotation-vector
            component (degrees).  Default ``0.2``.
        structure_model (str): ``'average'`` or ``'incoherent'``.
        method (str): ``scipy.optimize.minimize`` method.  Default ``'Powell'``.
        options (dict or None): Forwarded to ``minimize`` (merged over
            defaults ``maxiter=2000``, ``xtol/ftol=1e-6`` for Powell).
        verbose (bool): Print starting score and result summary.

    Returns:
        StackImageRefinementResult
"""
    ny, nx = image.shape
    valid  = image >= 0
    img    = image.astype(np.float64)
    img[~valid] = 0.0

    if bg_sigma > 0:
        smooth = _fft_gauss_convolve(img, bg_sigma)
        norm   = _fft_gauss_convolve(valid.astype(np.float64), bg_sigma)
        norm[norm < 1e-6] = 1.0
        img    = np.clip(img - smooth / norm, 0.0, None)
        img[~valid] = 0.0

    U0_layers = [layer.U.copy() for layer in stack.all_layers]

    lim    = float(np.radians(max_angle_deg))
    bounds = [(-lim, lim)] * 3

    def _score(rotvec: np.ndarray) -> float:
        R = Rotation.from_rotvec(rotvec).as_matrix()
        for layer, U0 in zip(stack.all_layers, U0_layers):
            layer.U = R @ U0

        spots = simulate_laue_stack(
            stack, camera,
            E_min_eV=E_min, E_max_eV=E_max,
            structure_model=structure_model,
            verbose=False,
            geometry_only=True,
            allowed_hkl=allowed_hkl,
            correct_depth=correct_depth,
        )
        if not spots:
            return 0.0

        delta = np.zeros((ny, nx), dtype=np.float64)
        for s in spots:
            xc, yc = s["pix"]
            col = int(round(xc))
            row = int(round(yc))
            if 0 <= row < ny and 0 <= col < nx and valid[row, col]:
                delta[row, col] += float(s["intensity"])

        return float(np.sum(_fft_gauss_convolve(delta, kernel_sigma) * img))

    score0 = _score(np.zeros(3))

    if verbose:
        print(
            f"refine_orientation_image_stack: score0={score0:.1f}  "
            f"method={method}  max_angle={max_angle_deg}°  "
            f"{len(U0_layers)} layers"
        )

    opts: dict = {"maxiter": 2000}
    if method == "Powell":
        opts.update({"xtol": 1e-6, "ftol": 1e-6})
    else:
        opts.update({"gtol": 1e-7})
    if options:
        opts.update(options)

    try:
        result = minimize(
            lambda rv: -_score(rv), np.zeros(3),
            method=method, bounds=bounds, options=opts,
        )
    finally:
        for layer, U0 in zip(stack.all_layers, U0_layers):
            layer.U = U0.copy()

    R_opt        = Rotation.from_rotvec(result.x).as_matrix()
    U_layers_final = [R_opt @ U0 for U0 in U0_layers]
    R_global     = Rotation.from_matrix(U_layers_final[0] @ U0_layers[0].T).as_matrix()
    rotvec_total = Rotation.from_matrix(R_global).as_rotvec()

    for layer, U in zip(stack.all_layers, U_layers_final):
        layer.U = U.copy()
    try:
        final_spots = simulate_laue_stack(
            stack, camera,
            E_min_eV=E_min, E_max_eV=E_max,
            structure_model=structure_model,
            verbose=False,
            geometry_only=True,
            allowed_hkl=allowed_hkl,
            correct_depth=correct_depth,
        )
        n_sim = len(final_spots)
    finally:
        for layer, U0 in zip(stack.all_layers, U0_layers):
            layer.U = U0.copy()

    out = StackImageRefinementResult(
        R_global  = R_global,
        rotvec    = rotvec_total,
        U_layers  = U_layers_final,
        U0_layers = U0_layers,
        score     = -float(result.fun),
        score0    = score0,
        n_sim     = n_sim,
        success   = result.success,
        message   = result.message,
        optimizer = result,
    )

    if update_stack:
        for layer, U in zip(stack.all_layers, U_layers_final):
            layer.U = U.copy()

    if verbose:
        print(f"  {out}")

    return out


def search_orientation_image(
    crystal,
    U_ref: np.ndarray,
    camera,
    image: np.ndarray,
    *,
    kernel_sigma: float = 0.3,
    bg_sigma: float = 251.0,
    E_min: float = E_MIN_eV,
    E_max: float = E_MAX_eV,
    allowed_hkl=None,
    search_misor_deg: float = 5.0,
    n_search: int = 500,
    max_angle_deg: float = 0.2,
    seed: "int | None" = None,
    method: str = "Powell",
    options: "dict | None" = None,
    verbose: bool = False,
) -> "ImageRefinementResult":
    """
    Search for the best orientation within a misorientation ball around a
    reference orientation, then locally refine from the best candidate.

    Designed for **epitaxial contexts** where a film or secondary phase is
    expected to be within a few degrees of the reference (substrate) orientation
    but the exact rotation is unknown.  Unlike :func:`refine_orientation_image`,
    which only performs a local gradient-based search from a single starting
    point, this function performs a coarse grid search over the full
    misorientation ball before the local polish.

    **Algorithm**

    1. **Background subtraction** — same single large-σ FFT Gaussian as the
       other image-based functions; performed once before the optimisation.

    2. **Grid search** (fast, O(n_spots × n_search)): sample *n_search* random
       orientations uniformly within a ball of misorientation radius
       *search_misor_deg* around *U_ref*.  For each candidate the score is
       computed by a **direct intensity lookup** — the bg-subtracted image is
       sampled at each simulated spot position and weighted by the predicted
       spot intensity.  No FFT is needed for this step.  The candidate with the
       highest score is selected as the starting point for the local search.

    3. **Local refinement** (FFT objective): a Powell optimisation starting
       from the best grid candidate, with search space constrained to
       ±*max_angle_deg* around that point.  Uses the same Gaussian-convolution
       objective as :func:`refine_orientation_image` for precise sub-pixel
       accuracy.

    The total rotation from *U_ref* to the final orientation is reported as
    *rotvec* in the result.

    Args:
        crystal: Crystal object passed to :func:`simulate_laue`.
        U_ref ((3,3) ndarray): Reference orientation matrix (e.g. substrate
            grain).  The search is centred on this orientation.
        camera (Camera): Detector geometry.
        image ((ny, nx) ndarray): Raw detector frame.  Gap / invalid pixels
            must be flagged as negative (Eiger convention: −1).
        kernel_sigma (float): σ in pixels for the local-refinement Gaussian
            kernel.  Default ``0.3``.
        bg_sigma (float): σ in pixels for the background Gaussian.  Set to
            ``0`` to skip.  Default ``251``.
        E_min, E_max (float): Photon energy range in eV.
        allowed_hkl: Pre-computed allowed HKL frozenset.  Strongly recommended.
        search_misor_deg (float): Misorientation radius of the search ball in
            degrees.  All orientations within this angular distance from
            *U_ref* are candidates.  Default ``5.0``.
        n_search (int): Number of random orientations sampled in the grid
            search.  The reference orientation itself is always included.
            Default ``500``.
        max_angle_deg (float): Half-width of the local-refinement search space
            around the best grid candidate (degrees per rotation axis).
            Default ``0.2``.
        seed (int or None): Random seed for reproducible grid sampling.
        method (str): ``scipy.optimize.minimize`` method for local refinement.
            Default ``'Powell'``.
        options (dict or None): Options forwarded to ``scipy.optimize.minimize``.
        verbose (bool): Print grid-search summary and local-refinement result.

    Returns:
        ImageRefinementResult: ``U0`` is *U_ref*; ``rotvec`` is the total
            rotation from *U_ref* to the refined orientation; ``score0`` is
            the direct-lookup score at *U_ref*; ``score`` is the FFT-objective
            score at the refined orientation.

    Example::

        hkl = laue.precompute_allowed_hkl(crystal_film, E_max_eV=27000)

        result = laue.search_orientation_image(
            crystal_film, U_substrate, camera, raw_frame,
            allowed_hkl      = hkl,
            search_misor_deg = 5.0,   # search within 5° of substrate
            n_search         = 500,
            max_angle_deg    = 0.2,   # local polish radius
            verbose          = True,
        )
        print(result)
        # ImageRefinementResult [OK]  |δω|=2.341°  score=7823.4  Δscore=+1204.6  n_sim=92
        U_film = result.U
    """
    U_ref = np.asarray(U_ref, dtype=float)
    ny, nx = image.shape
    valid  = image >= 0
    img    = image.astype(np.float64)
    img[~valid] = 0.0

    # ── background subtraction (once) ────────────────────────────────────────
    if bg_sigma > 0:
        smooth = _fft_gauss_convolve(img, bg_sigma)
        norm   = _fft_gauss_convolve(valid.astype(np.float64), bg_sigma)
        norm[norm < 1e-6] = 1.0
        img = np.clip(img - smooth / norm, 0.0, None)
        img[~valid] = 0.0

    # ── direct-lookup score (no FFT — used for grid search) ──────────────────
    def _fast_score(U_cand: np.ndarray) -> float:
        spots = simulate_laue(
            crystal, U_cand, camera,
            E_min=E_min, E_max=E_max,
            allowed_hkl=allowed_hkl,
            geometry_only=True,
        )
        s = 0.0
        for sp in spots:
            xc, yc = sp["pix"]
            col = int(round(xc))
            row = int(round(yc))
            if 0 <= row < ny and 0 <= col < nx and valid[row, col]:
                s += float(sp["intensity"]) * img[row, col]
        return s

    # ── FFT-convolution score (used for local refinement) ────────────────────
    def _fft_score(rotvec: np.ndarray, U_base: np.ndarray) -> float:
        U = Rotation.from_rotvec(rotvec).as_matrix() @ U_base
        spots = simulate_laue(
            crystal, U, camera,
            E_min=E_min, E_max=E_max,
            allowed_hkl=allowed_hkl,
            geometry_only=True,
        )
        if not spots:
            return 0.0
        delta = np.zeros((ny, nx), dtype=np.float64)
        for sp in spots:
            xc, yc = sp["pix"]
            col = int(round(xc))
            row = int(round(yc))
            if 0 <= row < ny and 0 <= col < nx and valid[row, col]:
                delta[row, col] += float(sp["intensity"])
        return float(np.sum(_fft_gauss_convolve(delta, kernel_sigma) * img))

    # ── grid search ──────────────────────────────────────────────────────────
    rng = np.random.default_rng(seed)
    max_rad = float(np.radians(search_misor_deg))

    # Sample n_search random rotvecs uniform within a ball of radius max_rad.
    # r^(1/3) sampling gives uniform density in 3-D.
    dirs   = rng.standard_normal((n_search, 3))
    dirs  /= np.linalg.norm(dirs, axis=1, keepdims=True)
    radii  = max_rad * rng.random(n_search) ** (1.0 / 3.0)
    rotvecs_grid = np.vstack([np.zeros(3), dirs * radii[:, None]])   # include δω=0

    grid_scores = np.empty(len(rotvecs_grid))
    for i, dw in enumerate(rotvecs_grid):
        U_cand = Rotation.from_rotvec(dw).as_matrix() @ U_ref
        grid_scores[i] = _fast_score(U_cand)

    best_idx   = int(np.argmax(grid_scores))
    score0     = float(grid_scores[0])          # direct-lookup score at U_ref
    best_dw    = rotvecs_grid[best_idx]
    U_best     = Rotation.from_rotvec(best_dw).as_matrix() @ U_ref

    if verbose:
        mis_best = float(np.degrees(np.linalg.norm(best_dw)))
        print(
            f"search_orientation_image: grid score0={score0:.1f}  "
            f"best={grid_scores[best_idx]:.1f} at |δω|={mis_best:.3f}°  "
            f"n_search={n_search}  search_misor={search_misor_deg}°"
        )

    # ── local FFT refinement from best grid candidate ─────────────────────────
    lim    = float(np.radians(max_angle_deg))
    bounds = [(-lim, lim)] * 3

    opts: dict = {"maxiter": 2000}
    if method == "Powell":
        opts.update({"xtol": 1e-6, "ftol": 1e-6})
    else:
        opts.update({"gtol": 1e-7})
    if options:
        opts.update(options)

    result = minimize(
        lambda dw: -_fft_score(dw, U_best),
        np.zeros(3),
        method=method,
        bounds=bounds,
        options=opts,
    )

    rotvec_local = result.x
    U_final      = Rotation.from_rotvec(rotvec_local).as_matrix() @ U_best

    # Total rotvec from U_ref to U_final
    rotvec_total = Rotation.from_matrix(U_final @ U_ref.T).as_rotvec()

    spots_final = simulate_laue(
        crystal, U_final, camera,
        E_min=E_min, E_max=E_max,
        allowed_hkl=allowed_hkl,
        geometry_only=True,
    )

    out = ImageRefinementResult(
        U        = U_final,
        U0       = U_ref,
        rotvec   = rotvec_total,
        score    = -float(result.fun),
        score0   = score0,
        n_sim    = len(spots_final),
        success  = result.success,
        message  = result.message,
        optimizer= result,
    )

    if verbose:
        print(f"  {out}")

    return out


def search_strain_image(
    crystal,
    U_ref: np.ndarray,
    camera,
    image: np.ndarray,
    *,
    strain0: "np.ndarray | None" = None,
    fit_strain: "tuple[str, ...] | None" = None,
    kernel_sigma: float = 0.3,
    bg_sigma: float = 251.0,
    E_min: float = E_MIN_eV,
    E_max: float = E_MAX_eV,
    allowed_hkl=None,
    search_misor_deg: float = 5.0,
    n_search: int = 500,
    max_angle_deg: float = 0.2,
    strain_scale: float = 1e-4,
    seed: "int | None" = None,
    method: str = "Powell",
    options: "dict | None" = None,
    verbose: bool = False,
) -> "StrainImageRefinementResult":
    """
    Search for the best orientation within a misorientation ball around a
    reference orientation, then jointly refine orientation and strain from the
    best candidate.

    Combines the wide-area search of :func:`search_orientation_image` with
    the 9-parameter (3 rotation + 6 strain) objective of
    :func:`refine_strain_image`.  Designed for epitaxial contexts where the
    film is expected within *search_misor_deg* of a known reference (substrate)
    orientation but the exact rotation and lattice distortion are both unknown.

    **Algorithm**

    1. **Background subtraction** — single large-σ FFT Gaussian, computed once.

    2. **Orientation grid search** (fast, O(n_spots × n_search)): sample
       *n_search* random orientations uniformly within a ball of misorientation
       radius *search_misor_deg* around *U_ref*.  Strain is **not** searched at
       this stage — the spot-position shift from typical film strains (∼10⁻³)
       is negligible compared to the orientation offsets being searched.  Each
       candidate is scored by **direct intensity lookup** (no FFT).

    3. **Joint orientation + strain local refinement** (9-parameter FFT
       objective): starting from the best grid candidate and *strain0*, Powell
       optimises all free parameters simultaneously within ±*max_angle_deg* for
       rotations and ±5 % for strain components.  Uses the same Gaussian-
       convolution objective as :func:`refine_strain_image`.

    The result's *rotvec* is the **total** rotation from *U_ref* to the refined
    pure-rotation part *U*, not just the local correction from the grid candidate.

    Args:
        crystal: Crystal object passed to :func:`simulate_laue`.
        U_ref ((3,3) ndarray): Reference orientation matrix (e.g. substrate
            grain).  The orientation search is centred on this matrix.
        camera (Camera): Detector geometry.
        image ((ny, nx) ndarray): Raw detector frame.  Gap / invalid pixels
            must be flagged as negative (Eiger convention: −1).
        strain0 ((3,3) ndarray or None): Starting symmetric strain tensor ε
            for the local refinement.  ``None`` starts from zero strain.
        fit_strain (tuple of str or None): Strain components to refine.  Any
            subset of ``('e_xx', 'e_yy', 'e_zz', 'e_xy', 'e_xz', 'e_yz')``.
            ``None`` refines all six.  Default ``None``.
        kernel_sigma (float): σ in pixels for the local-refinement Gaussian
            kernel.  Default ``0.3``.
        bg_sigma (float): σ in pixels for the background Gaussian.  ``0``
            skips it.  Default ``251``.
        E_min, E_max (float): Photon energy range in eV.
        allowed_hkl: Pre-computed allowed HKL frozenset.  Strongly recommended.
        search_misor_deg (float): Misorientation radius of the orientation
            search ball in degrees.  Default ``5.0``.
        n_search (int): Number of random orientations sampled in the grid
            search.  *U_ref* itself is always included.  Default ``500``.
        max_angle_deg (float): Half-width of the local-refinement rotation
            search space around the best grid candidate (degrees per axis).
            Default ``0.2``.
        strain_scale (float): Internal divisor for strain parameters so their
            magnitudes match the rotation angles inside the optimizer.
            Default ``1e-4``.
        seed (int or None): Random seed for reproducible grid sampling.
        method (str): ``scipy.optimize.minimize`` method.  Default ``'Powell'``.
        options (dict or None): Options forwarded to ``scipy.optimize.minimize``.
        verbose (bool): Print grid-search summary and local-refinement result.

    Returns:
        StrainImageRefinementResult: ``U0`` is *U_ref*; ``rotvec`` is the total
            rotation from *U_ref* to the refined orientation; ``score0`` is the
            direct-lookup score at *U_ref*; ``score`` is the FFT-objective score
            at the refined orientation + strain.

    Example::

        hkl = laue.precompute_allowed_hkl(crystal_film, E_max_eV=27000)

        result = laue.search_strain_image(
            crystal_film, U_substrate, camera, raw_frame,
            strain0          = prior_strain_tensor,   # or None
            allowed_hkl      = hkl,
            search_misor_deg = 5.0,
            n_search         = 500,
            max_angle_deg    = 0.2,
            verbose          = True,
        )
        print(result)
        U_film   = result.U
        eps_film = result.strain_tensor_deviatoric
    """
    _fit_strain: tuple[str, ...] = tuple(fit_strain) if fit_strain is not None else _STRAIN_ALL

    U_ref = np.asarray(U_ref, dtype=float)
    ny, nx = image.shape
    valid  = image >= 0
    img    = image.astype(np.float64)
    img[~valid] = 0.0

    # ── background subtraction (once) ────────────────────────────────────────
    if bg_sigma > 0:
        smooth = _fft_gauss_convolve(img, bg_sigma)
        norm   = _fft_gauss_convolve(valid.astype(np.float64), bg_sigma)
        norm[norm < 1e-6] = 1.0
        img = np.clip(img - smooth / norm, 0.0, None)
        img[~valid] = 0.0

    # ── direct-lookup score — orientation only, no strain, no FFT ────────────
    def _fast_score(U_cand: np.ndarray) -> float:
        spots = simulate_laue(
            crystal, U_cand, camera,
            E_min=E_min, E_max=E_max,
            allowed_hkl=allowed_hkl,
            geometry_only=True,
        )
        s = 0.0
        for sp in spots:
            xc, yc = sp["pix"]
            col = int(round(xc))
            row = int(round(yc))
            if 0 <= row < ny and 0 <= col < nx and valid[row, col]:
                s += float(sp["intensity"]) * img[row, col]
        return s

    # ── FFT-convolution score for local 9-parameter refinement ───────────────
    def _fft_score(params: np.ndarray, U_base: np.ndarray) -> float:
        rotvec     = params[:3]
        strain_vals = params[3:] * strain_scale
        R          = Rotation.from_rotvec(rotvec).as_matrix()
        eps        = _strain_matrix(strain_vals, _fit_strain)
        U_eff      = R @ U_base @ (np.eye(3) + eps)

        spots = simulate_laue(
            crystal, U_eff, camera,
            E_min=E_min, E_max=E_max,
            allowed_hkl=allowed_hkl,
            geometry_only=True,
        )
        if not spots:
            return 0.0
        delta = np.zeros((ny, nx), dtype=np.float64)
        for sp in spots:
            xc, yc = sp["pix"]
            col = int(round(xc))
            row = int(round(yc))
            if 0 <= row < ny and 0 <= col < nx and valid[row, col]:
                delta[row, col] += float(sp["intensity"])
        return float(np.sum(_fft_gauss_convolve(delta, kernel_sigma) * img))

    # ── orientation grid search ───────────────────────────────────────────────
    rng     = np.random.default_rng(seed)
    max_rad = float(np.radians(search_misor_deg))

    dirs   = rng.standard_normal((n_search, 3))
    dirs  /= np.linalg.norm(dirs, axis=1, keepdims=True)
    radii  = max_rad * rng.random(n_search) ** (1.0 / 3.0)
    rotvecs_grid = np.vstack([np.zeros(3), dirs * radii[:, None]])

    grid_scores = np.empty(len(rotvecs_grid))
    for i, dw in enumerate(rotvecs_grid):
        U_cand = Rotation.from_rotvec(dw).as_matrix() @ U_ref
        grid_scores[i] = _fast_score(U_cand)

    best_idx = int(np.argmax(grid_scores))
    score0   = float(grid_scores[0])           # direct-lookup score at U_ref
    best_dw  = rotvecs_grid[best_idx]
    U_best   = Rotation.from_rotvec(best_dw).as_matrix() @ U_ref

    if verbose:
        mis_best = float(np.degrees(np.linalg.norm(best_dw)))
        print(
            f"search_strain_image: grid score0={score0:.1f}  "
            f"best={grid_scores[best_idx]:.1f} at |δω|={mis_best:.3f}°  "
            f"n_search={n_search}  search_misor={search_misor_deg}°"
        )

    # ── local 9-parameter FFT refinement from best grid candidate ────────────
    eps0 = np.asarray(strain0, dtype=float) if strain0 is not None else np.zeros((3, 3))
    strain0_vals = np.array([eps0[_STRAIN_IDX[n]] for n in _fit_strain], dtype=float)
    x0 = np.concatenate([np.zeros(3), strain0_vals / strain_scale])

    rot_lim    = float(np.radians(max_angle_deg))
    strain_lim = 0.05 / strain_scale
    bounds = (
        [(-rot_lim, rot_lim)] * 3
        + [(-strain_lim, strain_lim)] * len(_fit_strain)
    )

    opts: dict = {"maxiter": 5000}
    if method == "Powell":
        opts.update({"xtol": 1e-7, "ftol": 1e-7})
    else:
        opts.update({"gtol": 1e-8})
    if options:
        opts.update(options)

    if verbose:
        print(
            f"  local refine: fit_strain={_fit_strain}  "
            f"method={method}  max_angle={max_angle_deg}°"
        )

    result = minimize(
        lambda p: -_fft_score(p, U_best),
        x0,
        method=method,
        bounds=bounds,
        options=opts,
    )

    rotvec_opt  = result.x[:3]
    strain_vals = result.x[3:] * strain_scale
    R_opt       = Rotation.from_rotvec(rotvec_opt).as_matrix()
    eps_opt     = _strain_matrix(strain_vals, _fit_strain)
    U_refined   = R_opt @ U_best
    U_eff_final = U_refined @ (np.eye(3) + eps_opt)

    # Total rotvec from U_ref to U_refined
    rotvec_total = Rotation.from_matrix(U_refined @ U_ref.T).as_rotvec()

    spots_final = simulate_laue(
        crystal, U_eff_final, camera,
        E_min=E_min, E_max=E_max,
        allowed_hkl=allowed_hkl,
        geometry_only=True,
    )

    out = StrainImageRefinementResult(
        U             = U_refined,
        U0            = U_ref,
        U_eff         = U_eff_final,
        rotvec        = rotvec_total,
        strain_tensor = eps_opt,
        strain_voigt  = _strain_to_voigt(strain_vals, _fit_strain),
        fit_strain    = _fit_strain,
        score         = -float(result.fun),
        score0        = score0,
        n_sim         = len(spots_final),
        success       = result.success,
        message       = result.message,
        optimizer     = result,
    )

    if verbose:
        print(f"  {out}")

    return out


def refine_strain_image(
    crystal,
    U0: np.ndarray,
    camera,
    image: np.ndarray,
    *,
    strain0: "np.ndarray | None" = None,
    fit_strain: "tuple[str, ...] | None" = None,
    kernel_sigma: float = 0.3,
    bg_sigma: float = 251.0,
    E_min: float = E_MIN_eV,
    E_max: float = E_MAX_eV,
    allowed_hkl=None,
    max_angle_deg: float = 0.2,
    strain_scale: float = 1e-4,
    method: str = "Powell",
    options: "dict | None" = None,
    verbose: bool = False,
) -> "StrainImageRefinementResult":
    """
    Post-refine orientation **and** strain tensor simultaneously by maximising
    the total Gaussian-weighted pixel intensity at simulated Laue spot
    positions.

    Uses the same image-based objective as :func:`refine_orientation_image`
    but extends the parameter space to include the six independent strain
    components, using the same deformation model as
    :func:`fit_strain_orientation`::

        U_eff = R(δω) @ U0 @ (I + ε)

    This makes it useful as a polishing pass after conventional strain fitting
    when some reflections are too weak or too few to be reliably segmented.

    **Parameter vector**

    ``params = [ωx, ωy, ωz, *strain_free]``

    * The first 3 entries are the Rodriguez rotation vector δω (radians).
    * The remaining entries are the free strain components, internally
      divided by *strain_scale* so their magnitudes are comparable to the
      rotation angles and Powell steps uniformly across all parameters.

    **Recommended workflow**

    Run :func:`refine_orientation_image` first (3-parameter, robust), then
    pass its refined ``U`` as *U0* here together with the prior
    ``strain_tensor`` as *strain0*::

        r_ori = refine_orientation_image(crystal, U0, camera, frame,
                                         allowed_hkl=hkl, verbose=True)
        r_str = refine_strain_image(crystal, r_ori.U, camera, frame,
                                     strain0=prior_strain_tensor,
                                     allowed_hkl=hkl, verbose=True)

    **Spot confusion and kernel size**

    The same *kernel_sigma* / *max_angle_deg* trade-offs as
    :func:`refine_orientation_image` apply here.  Because the 9-parameter
    landscape is less constrained than the 3-parameter one, keeping both
    values small is especially important when secondary spots are close to
    and dimmer than primary grain spots.  The ±5 % hard wall on each strain
    component prevents the optimizer from exploring unphysical deformations.

    Args:
        crystal: Crystal object passed to :func:`simulate_laue`.
        U0 ((3,3) ndarray): Starting orientation matrix (pure rotation
            part).  Ideally the output of :func:`refine_orientation_image`.
        camera (Camera): Detector geometry.
        image ((ny, nx) ndarray): Raw detector frame.  Gap / invalid pixels
            must be negative (Eiger convention: −1).
        strain0 ((3,3) ndarray or None): Starting symmetric strain tensor ε.
            When *u_source* is ``'strain'`` in the SLURM pipeline this is
            loaded automatically from the prior strain-fit ``.npz``.
            ``None`` starts from zero strain.  Default ``None``.
        fit_strain (tuple of str or None): Strain components to refine.
            Any subset of
            ``('e_xx', 'e_yy', 'e_zz', 'e_xy', 'e_xz', 'e_yz')``.
            Components not listed are fixed at their *strain0* value.
            ``None`` refines all six.  Default ``None``.
        kernel_sigma (float): σ in pixels of the Gaussian kernel placed at
            each simulated spot.  See :func:`refine_orientation_image` for
            guidance on choosing this value.  Default ``0.3``.
        bg_sigma (float): σ in pixels of the large-scale background
            Gaussian subtracted before optimisation.  Set to ``0`` to skip.
            Default ``251``.
        E_min, E_max (float): Photon energy range in eV forwarded to
            :func:`simulate_laue`.
        allowed_hkl: Pre-computed allowed HKL frozenset from
            :func:`precompute_allowed_hkl`.  Strongly recommended for the
            same reasons as in :func:`refine_orientation_image`.
        max_angle_deg (float): Symmetric bound on each rotation-vector
            component (degrees).  Default ``0.2``.
        strain_scale (float): Internal divisor applied to strain parameters
            inside the optimizer.  Strain components (∼10⁻³) are divided by
            *strain_scale* so their internal magnitudes match the rotation
            angles (∼10⁻²).  Default ``1e-4``.
        method (str): ``scipy.optimize.minimize`` method.  ``'Powell'``
            (default) works well for the smooth Gaussian landscape.
        options (dict or None): Forwarded to ``scipy.optimize.minimize``
            as the ``options`` keyword, merged over sensible defaults
            (``maxiter=5000``, ``xtol/ftol=1e-7`` for Powell).
        verbose (bool): If ``True``, print the starting score, free strain
            components, and a one-line result summary.  Default ``False``.

    Returns:
        StrainImageRefinementResult
    """
    _fit_strain: tuple[str, ...] = tuple(fit_strain) if fit_strain is not None else _STRAIN_ALL

    ny, nx = image.shape
    valid  = image >= 0
    img    = image.astype(np.float64)
    img[~valid] = 0.0

    if bg_sigma > 0:
        smooth = _fft_gauss_convolve(img, bg_sigma)
        norm   = _fft_gauss_convolve(valid.astype(np.float64), bg_sigma)
        norm[norm < 1e-6] = 1.0
        img = np.clip(img - smooth / norm, 0.0, None)
        img[~valid] = 0.0

    # Starting parameter vector: [rotvec(3), strain_free(N)] scaled.
    eps0 = np.asarray(strain0, dtype=float) if strain0 is not None else np.zeros((3, 3))
    strain0_vals = np.array(
        [eps0[_STRAIN_IDX[n]] for n in _fit_strain], dtype=float
    )
    x0 = np.concatenate([np.zeros(3), strain0_vals / strain_scale])

    rot_lim    = float(np.radians(max_angle_deg))
    strain_lim = 0.05 / strain_scale          # ±5 % hard wall — far outside physical range
    bounds = (
        [(-rot_lim, rot_lim)] * 3
        + [(-strain_lim, strain_lim)] * len(_fit_strain)
    )

    def _score(params: np.ndarray) -> float:
        rotvec     = params[:3]
        strain_vals = params[3:] * strain_scale
        R          = Rotation.from_rotvec(rotvec).as_matrix()
        eps        = _strain_matrix(strain_vals, _fit_strain)
        U_eff      = R @ np.asarray(U0, dtype=float) @ (np.eye(3) + eps)

        spots = simulate_laue(
            crystal, U_eff, camera,
            E_min=E_min, E_max=E_max,
            allowed_hkl=allowed_hkl,
            geometry_only=True,
        )
        if not spots:
            return 0.0

        delta = np.zeros((ny, nx), dtype=np.float64)
        for s in spots:
            xc, yc = s["pix"]
            col = int(round(xc))
            row = int(round(yc))
            if 0 <= row < ny and 0 <= col < nx and valid[row, col]:
                delta[row, col] += float(s["intensity"])

        kernel_map = _fft_gauss_convolve(delta, kernel_sigma)
        return float(np.sum(kernel_map * img))

    def _objective(params: np.ndarray) -> float:
        return -_score(params)

    score0 = _score(x0)

    if verbose:
        print(
            f"refine_strain_image: score0={score0:.1f}  "
            f"fit_strain={_fit_strain}  method={method}  max_angle={max_angle_deg}°"
        )

    opts: dict = {"maxiter": 5000}
    if method == "Powell":
        opts.update({"xtol": 1e-7, "ftol": 1e-7})
    else:
        opts.update({"gtol": 1e-8})
    if options:
        opts.update(options)

    result = minimize(_objective, x0, method=method, bounds=bounds, options=opts)

    rotvec_opt   = result.x[:3]
    strain_vals  = result.x[3:] * strain_scale
    R_opt        = Rotation.from_rotvec(rotvec_opt).as_matrix()
    eps_opt      = _strain_matrix(strain_vals, _fit_strain)
    U_refined    = R_opt @ np.asarray(U0, dtype=float)
    U_eff_final  = U_refined @ (np.eye(3) + eps_opt)

    spots_final = simulate_laue(
        crystal, U_eff_final, camera,
        E_min=E_min, E_max=E_max,
        allowed_hkl=allowed_hkl,
        geometry_only=True,
    )

    out = StrainImageRefinementResult(
        U             = U_refined,
        U0            = np.asarray(U0, dtype=float),
        U_eff         = U_eff_final,
        rotvec        = rotvec_opt,
        strain_tensor = eps_opt,
        strain_voigt  = _strain_to_voigt(strain_vals, _fit_strain),
        fit_strain    = _fit_strain,
        score         = -float(result.fun),
        score0        = score0,
        n_sim         = len(spots_final),
        success       = result.success,
        message       = result.message,
        optimizer     = result,
    )

    if verbose:
        print(f"  {out}")

    return out


def refine_strain_image_stack(
    stack,
    camera,
    image: np.ndarray,
    *,
    strain0_list: "list[np.ndarray] | None" = None,
    fit_strain: "tuple[str, ...] | None" = None,
    kernel_sigma: float = 0.3,
    bg_sigma: float = 251.0,
    E_min: float = E_MIN_eV,
    E_max: float = E_MAX_eV,
    allowed_hkl=None,
    max_angle_deg: float = 0.2,
    max_shift_px: "float | list[float] | None" = None,
    strain_scale: float = 1e-4,
    structure_model: str = "average",
    method: str = "Powell",
    options: "dict | None" = None,
    update_stack: bool = False,
    correct_depth: bool = False,
    verbose: bool = False,
) -> StackStrainImageRefinementResult:
    """
    Post-refine orientation **and** per-layer strain for a
    :class:`~nrxrdct.laue.layers.LayeredCrystal` by maximising the total
    Gaussian-weighted pixel intensity at simulated Laue spot positions.

    Extends :func:`refine_strain_image` to the layered-crystal case.  A single
    global rotation is shared by all layers while each layer receives its own
    independent strain tensor::

        U_eff_i = R(δω) @ U0_i @ (I + ε_i)

    The image-based objective does not require segmented peaks, making it
    useful when spots are too weak or too crowded to segment reliably.

    **Recommended workflow**

    Run :func:`fit_orientation_stack` (or :func:`fit_strain_orientation_stack`)
    first, then pass the refined stack here for image-space polishing::

        result_peaks = laue.fit_strain_orientation_stack(stack, cam, peaks, ...)
        result_image = laue.refine_strain_image_stack(stack, cam, frame,
                           strain0_list=result_peaks.strain_tensors,
                           allowed_hkl=hkl, verbose=True)

    Args:
        stack (LayeredCrystal): Layered structure.  Layer U matrices are used
            as starting orientations.  Always restored to their original state
            if an exception occurs.
        camera (Camera): Detector geometry.
        image ((ny, nx) ndarray): Raw detector frame.  Invalid / gap pixels
            must be negative (Eiger convention: ``−1``).
        strain0_list (list of (3,3) or None): Starting strain tensor per layer
            in ``stack.all_layers`` order.  ``None`` starts every layer from
            zero strain.
        fit_strain (tuple of str or None): Strain components to refine.  Any
            subset of ``('e_xx','e_yy','e_zz','e_xy','e_xz','e_yz')``.
            ``None`` refines all six.
        kernel_sigma (float): σ (pixels) of the Gaussian placed at each
            simulated spot.  Default ``0.3``.
        bg_sigma (float): σ (pixels) of the background estimate subtracted
            before optimisation.  ``0`` to skip.  Default ``251``.
        E_min, E_max (float): Photon energy range (eV).
        allowed_hkl: Pre-computed allowed HKL dict keyed by ``id(crystal)``,
            as returned by :func:`precompute_allowed_hkl`.  Strongly
            recommended to avoid recomputing structure factors on every call.
        max_angle_deg (float): Symmetric bound on each rotation-vector
            component (degrees).  Default ``0.2``.
        max_shift_px (float or list of float, optional): Maximum allowed
            detector-pixel displacement caused by strain for each layer.
            Converted to a strain limit using the camera geometry
            (``max_strain ≈ max_shift_px / (D / pixel_size)``).  A single
            float applies the same bound to every layer; a list sets a
            per-layer bound so that weaker or thinner layers can be
            constrained more tightly::

                # tighten the second (weaker) layer to ±3 px,
                # leave the first layer at the default ±5 %
                result = refine_strain_image_stack(
                    stack, camera, frame,
                    max_shift_px=[None, 3.0],   # None → default ±5 %
                )

            If ``None`` (default), falls back to ``±5 %`` absolute strain
            for all layers.
        strain_scale (float): Internal divisor for strain parameters.
            Default ``1e-4``.
        structure_model (str): ``'average'`` or ``'incoherent'``.
        method (str): ``scipy.optimize.minimize`` method.  Default ``'Powell'``.
        options (dict or None): Forwarded to ``minimize`` (merged over
            defaults ``maxiter=5000``, ``xtol/ftol=1e-7`` for Powell).
        verbose (bool): Print a one-line summary after optimisation.

    Returns:
        StackStrainImageRefinementResult
"""
    _fit_strain: tuple[str, ...] = (
        tuple(fit_strain) if fit_strain is not None else _STRAIN_ALL
    )

    ny, nx = image.shape
    valid  = image >= 0
    img    = image.astype(np.float64)
    img[~valid] = 0.0

    if bg_sigma > 0:
        smooth = _fft_gauss_convolve(img, bg_sigma)
        norm   = _fft_gauss_convolve(valid.astype(np.float64), bg_sigma)
        norm[norm < 1e-6] = 1.0
        img    = np.clip(img - smooth / norm, 0.0, None)
        img[~valid] = 0.0

    n_layers = len(stack.all_layers)
    n_strain = len(_fit_strain)
    n_params  = 3 + n_layers * n_strain

    U0_layers = [layer.U.copy() for layer in stack.all_layers]

    # Build starting parameter vector from optional prior strain tensors.
    x0 = np.zeros(n_params)
    if strain0_list is not None:
        for ii, eps0 in enumerate(strain0_list):
            eps0_arr = np.asarray(eps0, dtype=float)
            for jj, name in enumerate(_fit_strain):
                x0[3 + ii * n_strain + jj] = eps0_arr[_STRAIN_IDX[name]] / strain_scale

    rot_lim = float(np.radians(max_angle_deg))

    # Per-layer strain limits: derived from max_shift_px if given, else ±5%.
    # Pixel sensitivity: a unit strain shifts a spot by roughly D/pixel_size pixels,
    # so max_strain = max_shift_px / (D / pixel_size).
    _default_lim = 0.05 / strain_scale
    if max_shift_px is not None:
        _px_sens = camera.dd / camera.pixel_mm
        if isinstance(max_shift_px, (int, float)):
            _per_layer_lim = [float(max_shift_px) / _px_sens / strain_scale] * n_layers
        else:
            _per_layer_lim = [
                _default_lim if v is None else float(v) / _px_sens / strain_scale
                for v in max_shift_px
            ]
    else:
        _per_layer_lim = [_default_lim] * n_layers

    bounds = (
        [(-rot_lim, rot_lim)] * 3
        + [b for lim in _per_layer_lim for b in [(-lim, lim)] * n_strain]
    )

    def _score(params: np.ndarray) -> float:
        R = Rotation.from_rotvec(params[:3]).as_matrix()
        for ii, (layer, U0) in enumerate(zip(stack.all_layers, U0_layers)):
            sv  = params[3 + ii * n_strain : 3 + (ii + 1) * n_strain] * strain_scale
            eps = _strain_matrix(sv, _fit_strain)
            layer.U = R @ U0 @ (np.eye(3) + eps)

        spots = simulate_laue_stack(
            stack, camera,
            E_min_eV=E_min, E_max_eV=E_max,
            structure_model=structure_model,
            verbose=False,
            geometry_only=True,
            allowed_hkl=allowed_hkl,
            correct_depth=correct_depth,
        )
        if not spots:
            return 0.0

        delta = np.zeros((ny, nx), dtype=np.float64)
        for s in spots:
            xc, yc = s["pix"]
            col = int(round(xc))
            row = int(round(yc))
            if 0 <= row < ny and 0 <= col < nx and valid[row, col]:
                delta[row, col] += float(s["intensity"])

        kernel_map = _fft_gauss_convolve(delta, kernel_sigma)
        return float(np.sum(kernel_map * img))

    score0 = _score(x0)

    if verbose:
        print(
            f"refine_strain_image_stack: score0={score0:.1f}  "
            f"fit_strain={_fit_strain}  {n_layers} layers  "
            f"method={method}  max_angle={max_angle_deg}°"
        )

    opts: dict = {"maxiter": 5000}
    if method == "Powell":
        opts.update({"xtol": 1e-7, "ftol": 1e-7})
    else:
        opts.update({"gtol": 1e-8})
    if options:
        opts.update(options)

    try:
        result = minimize(
            lambda p: -_score(p), x0,
            method=method, bounds=bounds, options=opts,
        )
    finally:
        for layer, U0 in zip(stack.all_layers, U0_layers):
            layer.U = U0.copy()

    # Unpack solution.
    R_opt = Rotation.from_rotvec(result.x[:3]).as_matrix()
    U_layers_final = []
    U_eff_layers   = []
    strain_tensors = []
    strain_voigts  = []

    for ii, U0 in enumerate(U0_layers):
        sv  = result.x[3 + ii * n_strain : 3 + (ii + 1) * n_strain] * strain_scale
        eps = _strain_matrix(sv, _fit_strain)
        U_pure = R_opt @ U0
        U_eff  = U_pure @ (np.eye(3) + eps)
        U_layers_final.append(U_pure)
        U_eff_layers.append(U_eff)
        strain_tensors.append(eps)
        strain_voigts.append(_strain_to_voigt(sv, _fit_strain))

    R_global     = Rotation.from_matrix(U_layers_final[0] @ U0_layers[0].T).as_matrix()
    rotvec_total = Rotation.from_matrix(R_global).as_rotvec()

    # Count final simulated spots.
    for layer, U_eff in zip(stack.all_layers, U_eff_layers):
        layer.U = U_eff.copy()
    try:
        final_spots = simulate_laue_stack(
            stack, camera,
            E_min_eV=E_min, E_max_eV=E_max,
            structure_model=structure_model,
            verbose=False,
            geometry_only=True,
            allowed_hkl=allowed_hkl,
            correct_depth=correct_depth,
        )
        n_sim = len(final_spots)
    finally:
        for layer, U0 in zip(stack.all_layers, U0_layers):
            layer.U = U0.copy()

    out = StackStrainImageRefinementResult(
        R_global       = R_global,
        rotvec         = rotvec_total,
        U_layers       = U_layers_final,
        U0_layers      = U0_layers,
        U_eff_layers   = U_eff_layers,
        strain_tensors = strain_tensors,
        strain_voigts  = strain_voigts,
        fit_strain     = _fit_strain,
        score          = -float(result.fun),
        score0         = score0,
        n_sim          = n_sim,
        success        = result.success,
        message        = result.message,
        optimizer      = result,
    )

    if update_stack:
        for layer, U_eff in zip(stack.all_layers, U_eff_layers):
            layer.U = U_eff.copy()

    if verbose:
        print(f"  {out}")

    return out
