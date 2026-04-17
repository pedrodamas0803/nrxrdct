"""
White-Beam Synchrotron Laue Diffraction – Reflection Geometry
==============================================================
Simulates single-crystal Laue diffraction with a synchrotron white beam
in reflection geometry, with a full pixelated camera model.

System  : equiatomic AlCoCrFeNi HEA  –  BCC (Im-3m) + B2 (Pm-3m)

Physics
-------
Laue condition (Ewald construction):
    lambda_hkl = -4*pi * (k_i_hat . G_hkl) / |G_hkl|^2

Spot intensity:
    I(hkl) = |F(G,E)|^2 * LP(2theta) * S(E)

  F(G,E)    – structure factor via xrayutilities (Cromer-Mann f0 +
               Henke f'(E), f''(E) anomalous corrections)
  LP(2theta) – Lorentz-polarisation (unpolarised beam):
               LP = (1 + cos^2(2theta)) / (2*sin^2(theta)*cos(theta))
  S(E)      – synchrotron spectrum (bending magnet, wiggler, or undulator)
               NO bremsstrahlung

Synchrotron spectra
-------------------
  Bending magnet / wiggler (on-axis, Kim 1989):
      S(E) ∝ (E/Ec)^2 * K_{2/3}^2(E / 2*Ec)
      Peak at E ≈ 0.83*Ec.  Wiggler: flux x 2*N_poles.

  Undulator (planar, odd harmonics):
      S(E) = sum_n (1/n) * exp[-0.5*((E - n*E1)/sigma_n)^2]

Camera model
------------
  The detector is a flat pixelated area detector (e.g. Eiger, Pilatus,
  MAR, Perkin-Elmer, ...) described by:

    PIXEL_SIZE_MM      – pixel pitch (mm)
    N_PIX_H, N_PIX_V   – number of pixels (horizontal, vertical)
    DET_DIST_MM        – sample to detector-centre distance (mm)
    TTH_CENTER_DEG     – 2theta angle at the detector centre (deg)
                         Can be ANY angle, not restricted to 90°.
    NU_DEG             – out-of-plane (elevation) angle of detector centre
    CHI_DEG            – in-plane rotation of detector about its own normal

  For each diffracted beam direction kf_hat the code:
    1. Intersects the ray with the detector plane (exact geometry).
    2. Converts the hit position to (col, row) pixel coordinates.
    3. Renders a synthetic image with Gaussian spot profiles whose
       width is set by SPOT_SIGMA_PIX.

  The direct-beam footprint on the detector is also computed (if it
  would hit) so you can check the geometry is sensible.

Orientation
-----------
  Full orientation via Bunge ZXZ Euler angles (phi1, Phi, phi2).
  A Bragg-energy reference table is printed at runtime.
"""

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.special import kv

# ─────────────────────────────────────────────────────────────────────────────
# USER PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

# ── Synchrotron source ────────────────────────────────────────────────────────
SOURCE_TYPE = "bending_magnet"  # 'bending_magnet' | 'wiggler' | 'undulator'

E_CRIT_eV = 20_000  # Critical energy Ec (eV) for BM/wiggler
#   ESRF 6 GeV  : ~20 keV
#   APS  7 GeV  : ~19 keV
#   SOLEIL 2.75 : ~8.6 keV
N_WIGGLER_POLES = 40  # Number of wiggler poles (flux x 2N); BM: 1

E_FUNDAMENTAL_eV = 17_000  # Undulator fundamental energy E1 (eV)
N_HARMONICS = 7  # Number of odd harmonics
HARMONIC_WIDTH = 0.015  # Relative width per harmonic (angular acceptance)

# ── Energy window ─────────────────────────────────────────────────────────────
E_MIN_eV = 5_000  # Lower cut-off (eV)  – mirror/detector sensitivity
E_MAX_eV = 27_000  # Upper cut-off (eV)  – mirror/detector efficiency

# ── Crystal orientation – Bunge ZXZ Euler angles (degrees) ──────────────────
# Bunge ZXZ: R = Rz(phi1) · Rx(Phi) · Rz(phi2),   G_lab = U @ G_crystal
#
# Physical setup (BM32 / ID01-style top-camera Laue):
#   Beam along +x (LaueTools LT frame),  z vertical (up, ~ detector normal)
#   Sample surface tilted 40 deg w.r.t. beam:
#     -> surface normal at 50 deg from beam
#     -> detector at z (xbet~0) has 2theta_centre = 90 - xbet ~ 90 deg
#
# Phi=90 puts crystal [001] along beam (+x): 4-fold symmetric Laue pattern.
# Vary phi1 (in-plane) or Phi to bring other zone axes / families into condition.
PHI1_DEG = 0.0
PHI_DEG = 90.0
PHI2_DEG = 0.0

# ── Lattice ───────────────────────────────────────────────────────────────────
A_LATTICE = 2.881  # Angstrom (same for BCC and B2)

# ── Camera / detector model  (LaueTools calibration format) ─────────────────
# Parameters match exactly the LaueTools CCD calibration dictionary:
#   CCDCalibParameters: [dd, xcen, ycen, xbet, xgam]
#
# LaueTools LT2 lab frame (used here):
#   y // ki  (beam along +y)
#   z vertical up
#   x horizontal (towards wall)
#
# dd   : distance from sample to detector reference point O (mm)
#         = norm of vector IO
# xcen : pixel X coordinate of point O (normal incidence / beam footprint)
# ycen : pixel Y coordinate of point O
# xbet : angle (deg) between IO and the z axis
#         xbet ~ 0  => camera on top  (Z>0 geometry, 2theta_centre ~ 90 deg)
#         xbet ~ 90 => transmission forward camera
# xgam : in-plane rotation (deg) of the CCD array axes around the IO direction
#
# kf_direction: LaueTools geometry label
#   'Z>0'  top/side reflection  (default, xbet small)
#   'X>0'  transmission forward
#   'X<0'  back-reflection
#
# These values come directly from your LaueTools calibration file.
# Example from the sCMOS calibration shown:
DD = 85.475  # dd    (mm)
XCEN = 1040.26  # xcen  (pixels)
YCEN = 1126.63  # ycen  (pixels)
XBET = 0.447  # xbet  (degrees)
XGAM = 0.333  # xgam  (degrees)
PIXEL_SIZE_MM = 0.0734  # xpixelsize = ypixelsize (mm)
N_PIX_H = 2018  # framedim[0]
N_PIX_V = 2016  # framedim[1]
KF_DIRECTION = "Z>0"  # kf_direction from calibration file

# Spot rendering
SPOT_SIGMA_PIX = 5.0  # Gaussian sigma of each spot (pixels)
# Increase for mosaicity / divergence broadening

# ── Beam direction ────────────────────────────────────────────────────────────
# LaueTools frame (LT):
#   x // ki  (beam along +x)             <- canonical LaueTools convention
#   z  perpendicular to CCD, close to detector normal (vertical, pointing up)
#   y  = z ^ x  (horizontal, towards the door)
#
# The internal LT2 frame used in LaueGeometry.py has y//ki, but all
# PUBLIC LaueTools quantities (2theta, chi, UB matrix) are in the LT frame.
# We work in LT throughout.
KI_HAT = np.array([1.0, 0.0, 0.0])  # LT frame: beam along +x  (do not change)

# ── Simulation ────────────────────────────────────────────────────────────────
HMAX = 14
F2_THRESHOLD = 0.5

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS & UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

HC = 12_398.42  # h*c in eV*Angstrom


def en2lam(E_eV):
    return HC / E_eV


def lam2en(l_ang):
    return HC / l_ang


# ─────────────────────────────────────────────────────────────────────────────
# ORIENTATION
# ─────────────────────────────────────────────────────────────────────────────


def euler_to_U(phi1, Phi, phi2, sample_tilt_deg=0.0):
    """
    Bunge ZXZ Euler angles (deg) → 3×3 orientation matrix in the LT lab frame.

    Bunge Euler angles describe the crystal orientation relative to the
    **sample surface frame** (z = surface normal, x/y in the surface plane).
    ``simulate_laue`` expects U in the **lab frame** (x // beam).  When the
    sample is tilted on the stage the two frames differ by a rotation around
    the horizontal axis (y in LT).

    Parameters
    ----------
    phi1, Phi, phi2 : float
        Bunge ZXZ Euler angles in degrees.
    sample_tilt_deg : float, optional
        Tilt of the sample surface relative to the horizontal plane (deg).
        Positive = front edge of sample tilted downward so the surface faces
        the incoming beam (standard reflection geometry).

        - BM32 / ID01 Z>0 geometry, 40° grazing incidence → ``sample_tilt_deg=40``
        - LaueTools refined UB matrix (already in lab frame) → ``sample_tilt_deg=0``

    Returns
    -------
    U : ndarray, shape (3, 3)
        Orientation matrix such that ``G_lab = U @ G_crystal``.

    Notes
    -----
    The sample tilt is the **sample→lab** rotation about **+y** (horizontal
    axis perpendicular to the beam) by ``sample_tilt_deg``:

        R_tilt = Ry(+sample_tilt_deg)

    This maps the sample surface normal from +z (horizontal surface) to
    (sin θ, 0, cos θ) in the lab frame, which for θ = 40° gives a grazing
    angle of 40° with the beam and a specular 2θ of 80°, consistent with the
    BM32 Z>0 top-camera geometry.

    This is the **inverse** of the LaueTools ``matstarlab_to_matstarsample3x3``
    convention, which applies ``Rx_LT2(+omega)`` (rotation around x in the LT2
    frame) as the lab→sample direction.  Because ``x_LT2 = −y_LT``, that
    operation equals ``Ry_LT(−omega)`` (lab→sample), so the sample→lab
    direction used here is ``Ry_LT(+omega)``.

    When Euler angles come from a LaueTools indexing result (grain_matrix /
    deviatoric matrix) they are already expressed in the lab frame; pass
    ``sample_tilt_deg=0`` (the default) in that case.
    """
    U_sample = Rotation.from_euler("ZXZ", [phi1, Phi, phi2], degrees=True).as_matrix()
    if sample_tilt_deg == 0.0:
        return U_sample
    # Rx_LT2(+omega) is the LaueTools lab→sample rotation (matstarlab_to_matstarsample3x3).
    # x_LT2 = −y_LT, so Rx_LT2(omega) = Ry_LT(−omega).
    # We need the inverse (sample→lab): Ry_LT(+omega).
    R_tilt = Rotation.from_euler("Y", +sample_tilt_deg, degrees=True).as_matrix()
    return R_tilt @ U_sample


def beam_in_crystal(U):
    """Crystal-frame direction of the incident beam (x in LT lab frame).
    U must already be in the lab frame (use sample_tilt_deg in euler_to_U)."""
    return U.T @ np.array([1.0, 0.0, 0.0])


# LT2→LT passive rotation (coordinate-frame change, not a physical rotation)
# LaueTools stores matstarlab in LT2 (y//beam, OR/XMAS frame).
# simulate_laue works in LT (x//beam, LaueTools public frame).
#   x_LT = y_LT2,  y_LT = -x_LT2,  z_LT = z_LT2
_R_LT2_TO_LT = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


def _build_B0(crystal):
    """Return the 3×3 reference reciprocal-lattice matrix B0 (with 2π, crystal frame)."""
    return np.column_stack(
        [
            crystal.Q(1, 0, 0),
            crystal.Q(0, 1, 0),
            crystal.Q(0, 0, 1),
        ]
    )


def _matstarlab_to_F(matstarlab, crystal):
    """
    Internal: convert matstarlab (LT2, no 2π) → deformation gradient F (LT frame, with 2π).

    F = U @ P  where U is pure rotation and P is the right-stretch tensor.
    F maps crystal-frame reciprocal vectors to LT lab-frame vectors:
        G_LT = F @ G_crystal   (with G_crystal = B0 @ [h,k,l])
    """
    B0 = _build_B0(crystal)
    matstarlab_LT = _R_LT2_TO_LT @ (np.asarray(matstarlab, dtype=float) * 2.0 * np.pi)
    return matstarlab_LT @ np.linalg.inv(B0)


def U_from_matstarlab(matstarlab, crystal):
    """
    Convert a LaueTools ``matstarlab`` (LT2/OR frame, no 2π) into an effective
    orientation matrix for ``simulate_laue`` (LT frame, with 2π from xrayutilities).

    This function returns the **full deformation gradient** F = U @ P, which
    combines the pure crystal rotation U with the right-stretch tensor P
    (lattice distortion due to strain).  Passing F to ``simulate_laue`` gives
    spot positions that account for both the orientation **and** any elastic
    strain in the grain.

    To separate rotation from strain use :func:`decompose_matstarlab`.

    LaueTools defines (LT2 frame, no 2π)::

        G_LT2 = matstarlab @ [h, k, l]

    This function applies two corrections:

    1. **Frame change LT2→LT**: ``x_LT = y_LT2``, ``y_LT = −x_LT2``, ``z_LT = z_LT2``
    2. **2π rescaling**: LaueTools uses |G| = 1/d; xrayutilities uses |G| = 2π/d.

    Parameters
    ----------
    matstarlab : array-like, shape (3, 3)
        LaueTools grain matrix in LT2/OR frame (columns = a*, b*, c* in lab,
        in Å⁻¹ **without** the 2π factor).
    crystal : xu.materials.Crystal
        Reference (unstrained) phase — same object passed to ``simulate_laue``.

    Returns
    -------
    F : ndarray, shape (3, 3)
        Deformation gradient in LT frame.  Pass directly to ``simulate_laue``
        as the ``U`` argument to include strain in the spot geometry.
        For a strain-free grain F is a pure rotation matrix.
    """
    return _matstarlab_to_F(matstarlab, crystal)


def decompose_matstarlab(matstarlab, crystal):
    """
    Decompose a LaueTools ``matstarlab`` into pure rotation and elastic strain.

    Uses the **right polar decomposition** of the deformation gradient F:

    .. math::

        F = U \\cdot P

    where:

    * **F** — full deformation gradient (LT frame) = what ``U_from_matstarlab`` returns
    * **U** — pure rotation (orthogonal, det = +1): the rigid crystal orientation
    * **P** — right-stretch tensor (symmetric positive-definite): the lattice distortion

    The small-strain tensor is extracted from P as ``ε = P − I``.

    Parameters
    ----------
    matstarlab : array-like, shape (3, 3)
        LaueTools grain matrix in LT2/OR frame (no 2π).
    crystal : xu.materials.Crystal
        Reference (unstrained) crystal — same object passed to ``simulate_laue``.

    Returns
    -------
    U : ndarray, shape (3, 3)
        Pure rotation in LT frame (orthogonal, det ≈ +1).  Use this in
        ``simulate_laue`` when you want rotation-only simulation (strain
        effects on peak positions are ignored).
    F : ndarray, shape (3, 3)
        Full deformation gradient (= ``U_from_matstarlab`` output).  Use
        this in ``simulate_laue`` to include strain in the spot geometry.
    eps : ndarray, shape (3, 3)
        Small-strain tensor in the crystal frame: ``ε = P − I``.
        Diagonal entries are normal strains (ε₁₁, ε₂₂, ε₃₃);
        off-diagonal entries are shear strains (engineering convention ×½).
    eps_voigt : ndarray, shape (6,)
        Voigt representation ``[ε₁₁, ε₂₂, ε₃₃, ε₂₃, ε₁₃, ε₁₂]``.

    Notes
    -----
    * The decomposition is exact (no small-strain approximation).
    * For strains ≲ 10⁻³ (typical elastic), P ≈ I and F ≈ U.
    * The strain ε is expressed in the **crystal frame** (principal axes of P).
      To express it in the lab frame: ``ε_lab = U @ ε @ U.T``.
    * To check: ``np.allclose(U @ (eps + np.eye(3)) @ B0, F @ B0)`` should hold.

    Example
    -------
    >>> U, F, eps, eps_v = decompose_matstarlab(matstarlab, crystal)
    >>> spots_rot_only = simulate_laue(crystal, U, camera)   # rotation only
    >>> spots_with_strain = simulate_laue(crystal, F, camera) # rotation + strain
    >>> print("normal strains:", np.diag(eps))
    >>> print("shear  strains:", eps[0,1], eps[0,2], eps[1,2])
    """
    from scipy.linalg import polar

    F = _matstarlab_to_F(matstarlab, crystal)

    # Right polar decomposition: F = U @ P
    # U: orthogonal (rotation),  P: symmetric positive-definite (stretch)
    U, P = polar(F, side="right")

    # Small-strain tensor: ε = P − I  (exact for symmetric P)
    eps = P - np.eye(3)

    # Voigt: [e11, e22, e33, e23, e13, e12]
    eps_voigt = np.array(
        [
            eps[0, 0],
            eps[1, 1],
            eps[2, 2],
            eps[1, 2],
            eps[0, 2],
            eps[0, 1],
        ]
    )

    return U, F, eps, eps_voigt


# ─────────────────────────────────────────────────────────────────────────────
# SYNCHROTRON SPECTRA  (no bremsstrahlung)
# ─────────────────────────────────────────────────────────────────────────────


def spectrum_bm(E_eV, Ec_eV=E_CRIT_eV, N=1):
    """Bending magnet / wiggler on-axis spectral flux (Kim 1989)."""
    y = E_eV / Ec_eV
    if y <= 1e-7:
        return 0.0
    return float(2 * N * y**2 * kv(2 / 3, y / 2) ** 2)


def spectrum_undulator(
    E_eV, E1_eV=E_FUNDAMENTAL_eV, n_harm=N_HARMONICS, sig_rel=HARMONIC_WIDTH
):
    """Planar undulator: sum of odd Gaussian harmonics."""
    s = 0.0
    for n in range(1, 2 * n_harm, 2):
        En = n * E1_eV
        s += (1.0 / n) * np.exp(-0.5 * ((E_eV - En) / (En * sig_rel)) ** 2)
    return float(s)


def synchrotron_spectrum(E_eV):
    if SOURCE_TYPE == "bending_magnet":
        return spectrum_bm(E_eV, E_CRIT_eV, 1)
    elif SOURCE_TYPE == "wiggler":
        return spectrum_bm(E_eV, E_CRIT_eV, N_WIGGLER_POLES)
    elif SOURCE_TYPE == "undulator":
        return spectrum_undulator(E_eV)
    raise ValueError(f"Unknown SOURCE_TYPE: {SOURCE_TYPE!r}")


# ─────────────────────────────────────────────────────────────────────────────
# LORENTZ-POLARISATION
# ─────────────────────────────────────────────────────────────────────────────


def lorentz_pol(tth_deg):
    r = np.radians(tth_deg)
    s, c = np.sin(r / 2), np.cos(r / 2)
    if abs(s) < 1e-8 or abs(c) < 1e-8:
        return 0.0
    return abs((1 + np.cos(r) ** 2) / (2 * s**2 * c))


def is_superlattice(h, k, l):
    return (abs(h) + abs(k) + abs(l)) % 2 == 1


# ─────────────────────────────────────────────────────────────────────────────
# STRAIN BROADENING
# ─────────────────────────────────────────────────────────────────────────────

# Voigt index → symmetric (i,j) tensor index
# Order: [ε₁₁, ε₂₂, ε₃₃, ε₂₃, ε₁₃, ε₁₂]
_VOIGT_IJ = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]


def strain_spot_jacobian(spots, crystal, U, camera, eps_step=1e-5):
    """
    Compute the 2×6 Jacobian ∂(xcam, ycam)/∂ε_voigt for each Laue spot.

    For each reflection (hkl) in ``spots``, a small strain increment is applied
    to each of the six independent tensor strain components (Voigt order:
    ε₁₁, ε₂₂, ε₃₃, ε₂₃, ε₁₃, ε₁₂) and the resulting shift in pixel
    coordinates is measured via finite differences.

    **Physics note** — in white-beam Laue the incident wavelength adjusts
    freely to satisfy the Laue condition.  A pure hydrostatic strain rescales
    |G| but not its direction, so it does **not** shift the spot.  Only the
    *deviatoric* part of the strain (which rotates G) moves spots.  The
    Jacobian captures this automatically.

    Parameters
    ----------
    spots : list of dict
        Output of :func:`simulate_laue` (must contain ``'hkl'`` and ``'pix'``).
    crystal : xu.materials.Crystal
        Same crystal used to produce ``spots``.
    U : ndarray, shape (3, 3)
        Orientation matrix used to produce ``spots`` (LT frame).
        Pass the **rotation-only** U from :func:`decompose_matstarlab`,
        not the full deformation gradient F, so that strain perturbations
        are applied on top of a clean orientation.
    camera : Camera
        Same camera used to produce ``spots``.
    eps_step : float, optional
        Finite-difference step size for each strain component (dimensionless).
        Default 1e-5 is safe for typical elastic strains ~10⁻³.

    Returns
    -------
    jacobians : dict  {(h,k,l): ndarray shape (2, 6)}
        Maps each hkl tuple to its 2×6 Jacobian matrix J where::

            [δxcam, δycam] ≈ J @ δε_voigt

        Spots for which the perturbed beam misses the detector for one or
        more components will have those columns set to zero.
    """
    ki_hat = KI_HAT / np.linalg.norm(KI_HAT)
    U = np.asarray(U, dtype=float)
    jacobians = {}

    for s in spots:
        if s.get("pix") is None:
            continue
        h, k, l = s["hkl"]
        G_cry = crystal.Q(h, k, l)
        pix0 = np.array(s["pix"], dtype=float)
        J = np.zeros((2, 6))

        for vi, (i, j) in enumerate(_VOIGT_IJ):
            # Symmetric strain perturbation tensor
            deps = np.zeros((3, 3))
            deps[i, j] += eps_step
            if i != j:
                deps[j, i] += eps_step  # symmetrise off-diagonal

            # Perturbed G in lab frame: G' = U @ (I + δε) @ G_cry
            G_lab_p = U @ (np.eye(3) + deps) @ G_cry

            # Re-apply Laue condition for perturbed G
            Gm2 = float(G_lab_p @ G_lab_p)
            kdG = float(ki_hat @ G_lab_p)
            if kdG >= 0 or Gm2 < 1e-30:
                continue
            lam_p = -4.0 * np.pi * kdG / Gm2
            km_p = 2.0 * np.pi / lam_p
            kf_p = ki_hat * km_p + G_lab_p
            kf_hat_p = kf_p / np.linalg.norm(kf_p)

            pix_p = camera.project(kf_hat_p)
            if pix_p is None:
                continue

            J[:, vi] = (np.array(pix_p) - pix0) / eps_step

        jacobians[(h, k, l)] = J

    return jacobians


def strain_broadening(
    spots, crystal, U, camera, eps_voigt_std=1e-3, eps_cov=None, eps_step=1e-5
):
    """
    Estimate the pixel-space broadening of each Laue spot due to strain.

    Given a strain distribution characterised by a covariance matrix Σ_ε
    (6×6 in Voigt space), the pixel-space covariance of a spot is:

    .. math::

        \\Sigma_{\\text{pix}} = J \\, \\Sigma_{\\varepsilon} \\, J^{\\top}

    where J (2×6) is the strain Jacobian from :func:`strain_spot_jacobian`.
    The broadening is reported as the RMS pixel spread (square root of the
    largest eigenvalue of Σ_pix) and the full 2×2 pixel covariance, from
    which the ellipse axes and orientation can be extracted.

    Parameters
    ----------
    spots : list of dict
        Output of :func:`simulate_laue`.
    crystal : xu.materials.Crystal
        Same crystal used to produce ``spots``.
    U : ndarray, shape (3, 3)
        Rotation-only orientation matrix (from :func:`decompose_matstarlab`).
    camera : Camera
        Same camera used to produce ``spots``.
    eps_voigt_std : float or array-like, shape (6,), optional
        Standard deviation of each Voigt strain component
        [σ₁₁, σ₂₂, σ₃₃, σ₂₃, σ₁₃, σ₁₂].
        If scalar, the same std is applied to all six components.
        Ignored when ``eps_cov`` is provided.
        Default: 1e-3 (typical elastic strain).
    eps_cov : array-like, shape (6, 6), optional
        Full 6×6 covariance matrix of the strain distribution (Voigt basis).
        When provided, overrides ``eps_voigt_std``.
    eps_step : float, optional
        Finite-difference step for the Jacobian computation.

    Returns
    -------
    list of dict
        Copy of ``spots`` with three new keys added to each entry:

        ``'sigma_strain_pix'`` : float
            RMS broadening (pixels) = √(largest eigenvalue of Σ_pix).
            This is the semi-major axis of the broadening ellipse.
        ``'sigma_strain_minor'`` : float
            Semi-minor axis of the broadening ellipse (pixels).
        ``'cov_pix'`` : ndarray, shape (2, 2)
            Full pixel-space covariance matrix.  Its eigenvectors give the
            orientations of the broadening ellipse on the detector.

    Notes
    -----
    * The broadening is relative to the spot centre; it does **not** include
      the intrinsic diffraction spot width (set by ``sigma_pix`` in
      :meth:`Camera.render`).
    * To render spots with strain broadening included, pass
      ``sigma_pix=sigma_strain_pix`` to :meth:`Camera.render`, or add it in
      quadrature: ``sigma_total = sqrt(sigma_instrument² + sigma_strain²)``.
    * The Jacobian approach is linear (first-order); it is accurate for
      strain spreads ≪ 1 and fails if strain is so large that spots migrate
      by more than ~10 px.
    """
    if eps_cov is not None:
        Sigma_eps = np.asarray(eps_cov, dtype=float)
    else:
        std = np.broadcast_to(np.asarray(eps_voigt_std, dtype=float), (6,))
        Sigma_eps = np.diag(std**2)

    jacobians = strain_spot_jacobian(spots, crystal, U, camera, eps_step=eps_step)

    result = []
    for s in spots:
        s = dict(s)
        J = jacobians.get(s["hkl"])
        if J is not None and np.any(J != 0):
            cov_pix = J @ Sigma_eps @ J.T  # (2,2)
            eigvals = np.linalg.eigvalsh(cov_pix)
            eigvals = np.maximum(eigvals, 0.0)  # numerical guard
            s["sigma_strain_pix"] = float(np.sqrt(eigvals.max()))
            s["sigma_strain_minor"] = float(np.sqrt(eigvals.min()))
            s["cov_pix"] = cov_pix
        else:
            s["sigma_strain_pix"] = 0.0
            s["sigma_strain_minor"] = 0.0
            s["cov_pix"] = np.zeros((2, 2))
        result.append(s)

    return result


def fit_strain_distribution(
    jacobians,
    sigma_meas_pix,
    sigma_instrument,
    mode="isotropic",
    min_sensitivity=0.1,
):
    """
    Estimate the strain distribution from measured Laue spot widths (inverse problem).

    Solves for Σ_ε given the measured semi-major broadening of each spot:

    .. math::

        \\sigma_{\\text{meas},k}^2 = \\sigma_{\\text{inst}}^2
                                   + \\lambda_{\\max}(J_k \\, \\Sigma_\\varepsilon \\, J_k^\\top)

    Two modes are supported:

    **isotropic** — single scalar σ_ε (Σ_ε = σ²I)
        The equation becomes linear in σ²:

        .. math::

            y_k = \\sigma^2 \\cdot \\lambda_{\\max}(J_k J_k^\\top)

        Solved by weighted least squares over all spots.

    **diagonal** — six independent variances σᵢ² (Σ_ε = diag(σ₁²,…,σ₆²))
        λ_max is non-linear in σᵢ², so the **trace** is used as a linear proxy:

        .. math::

            y_k \\approx \\sum_i \\sigma_i^2 \\, \\|J_{k,i}\\|^2

        (exact when the two eigenvalues of J Σ Jᵀ are equal; conservative
        otherwise, since trace ≥ λ_max).
        Solved by non-negative least squares (:func:`scipy.optimize.nnls`).

    Parameters
    ----------
    jacobians : dict  {(h,k,l): ndarray (2, 6)}
        Output of :func:`strain_spot_jacobian`.
    sigma_meas_pix : dict  {(h,k,l): float}
        Measured semi-major spot width (pixels, 1σ) for each indexed
        reflection, obtained by fitting a 2-D Gaussian to the experimental
        detector image.  Only hkl keys present in both ``jacobians`` and
        this dict are used.
    sigma_instrument : float or dict or result dict from :func:`estimate_instrument_broadening`
        Instrument broadening (pixels, 1σ), subtracted in quadrature per spot.
        Three forms are accepted:

        * **float** — the same value is applied to all spots.
        * **dict {(h,k,l): float}** — per-spot instrumental width (e.g. the
          ``'sigma_per_spot'`` entry from :func:`estimate_instrument_broadening`).
        * **result dict** — the full dict returned by
          :func:`estimate_instrument_broadening`; the ``'sigma_per_spot'`` field
          is extracted automatically.  For spots not covered by the calibrant,
          the scalar ``'sigma_instrument'`` fallback is used.
    mode : {'isotropic', 'diagonal'}
        Fitting model.  ``'isotropic'`` fits a single σ_ε;
        ``'diagonal'`` fits all six Voigt variances independently.
    min_sensitivity : float, optional
        Minimum RMS Jacobian magnitude (px per unit strain) required for a
        spot to be included.  Spots with ``sqrt(mean(J**2)) < min_sensitivity``
        are insensitive to strain and are excluded.  Default: 0.1.

    Returns
    -------
    result : dict with keys:

        ``'sigma_eps'`` : float
            Isotropic RMS strain (scalar).  For ``mode='isotropic'`` this is
            the direct fit result; for ``mode='diagonal'`` it is
            ``sqrt(mean(eps_voigt_std**2))``.
        ``'eps_voigt_std'`` : ndarray, shape (6,)
            Per-component standard deviation
            [σ₁₁, σ₂₂, σ₃₃, σ₂₃, σ₁₃, σ₁₂].
            For ``mode='isotropic'`` all six entries are equal.
        ``'Sigma_eps'`` : ndarray, shape (6, 6)
            Fitted covariance matrix (diagonal for both modes).
            Pass directly to :func:`strain_broadening` as ``eps_cov``.
        ``'residuals_pix'`` : ndarray
            Per-spot residual: measured − predicted broadening (pixels).
        ``'hkl_used'`` : list of tuples
            hkl indices of the spots that entered the fit.
        ``'n_spots'`` : int
            Number of spots used.
        ``'mode'`` : str
            The mode that was used.

    Raises
    ------
    ValueError
        If fewer than 2 spots survive the sensitivity cut.

    Notes
    -----
    * Feed the returned ``'Sigma_eps'`` to :func:`strain_broadening` to verify
      the fit: the predicted ``sigma_strain_pix`` values should match
      ``sigma_meas_pix - sigma_instrument`` (in quadrature).
    * For ``mode='diagonal'``, the system is underdetermined if fewer than
      6 spots are available.  In that case, prefer ``mode='isotropic'``.
    * Negative excess variance (``sigma_meas < sigma_instrument``) is clamped
      to zero rather than treated as an error.

    Example
    -------
    >>> # Calibrate instrument broadening from a strain-free reference
    >>> res_cal = estimate_instrument_broadening(spots_cal, sigma_meas_cal,
    ...                                          mode='linear_tth')
    >>>
    >>> # Measure spot widths from experimental image (e.g. with a 2-D Gaussian fit)
    >>> sigma_meas = {(1,1,0): 4.2, (2,0,0): 3.8, (1,1,2): 5.1, ...}
    >>> J = strain_spot_jacobian(spots, crystal, U, camera)
    >>>
    >>> # Pass the calibration result directly — per-spot σ_inst is applied
    >>> res = fit_strain_distribution(J, sigma_meas, sigma_instrument=res_cal)
    >>> print(f"σ_ε = {res['sigma_eps']:.2e}")
    >>> spots_check = strain_broadening(spots, crystal, U, camera,
    ...                                 eps_cov=res['Sigma_eps'])
    """
    from scipy.optimize import nnls

    # ── Resolve sigma_instrument → per-spot lookup + scalar fallback ─────────
    if isinstance(sigma_instrument, dict):
        if "sigma_per_spot" in sigma_instrument:
            # Full result dict from estimate_instrument_broadening
            _per_spot = sigma_instrument["sigma_per_spot"]
            _fallback = float(sigma_instrument.get("sigma_instrument", 0.0))
        else:
            # Plain {hkl: float} dict
            _per_spot = sigma_instrument
            _fallback = (
                float(np.median(list(sigma_instrument.values())))
                if sigma_instrument
                else 0.0
            )
    else:
        _per_spot = {}
        _fallback = float(sigma_instrument)

    def _sig_inst(hkl):
        return _per_spot.get(tuple(hkl), _fallback)

    # ── Collect common hkl keys, apply sensitivity filter ────────────────────
    common = set(jacobians.keys()) & set(sigma_meas_pix.keys())

    rows = []  # (hkl, J, sigma_meas, sigma_inst)
    for hkl in common:
        J = jacobians[hkl]
        if J is None or not np.any(J != 0):
            continue
        rms_J = np.sqrt(np.mean(J**2))
        if rms_J < min_sensitivity:
            continue
        rows.append((hkl, J, float(sigma_meas_pix[hkl]), _sig_inst(hkl)))

    if len(rows) < 2:
        raise ValueError(
            f"Only {len(rows)} spot(s) survived the sensitivity cut "
            f"(min_sensitivity={min_sensitivity} px/unit strain). "
            "Lower min_sensitivity or provide more measured spots."
        )

    hkl_used = [r[0] for r in rows]

    # ── Per-spot excess variance: y_k = max(0, σ_meas_k² − σ_inst_k²) ───────
    sig_inst2_arr = np.array([r[3] ** 2 for r in rows])
    y = np.array([max(0.0, r[2] ** 2 - r[3] ** 2) for r in rows])

    Js = [r[1] for r in rows]

    # ── Mode: isotropic ───────────────────────────────────────────────────────
    if mode == "isotropic":
        # y_k = σ² · λ_max(J_k J_kᵀ)
        x = np.array([np.linalg.eigvalsh(J @ J.T).max() for J in Js])
        # Weighted least squares: σ² = (xᵀy) / (xᵀx)
        denom = float(x @ x)
        if denom < 1e-30:
            raise ValueError("All Jacobian sensitivities are zero.")
        sigma_eps2 = max(0.0, float(x @ y) / denom)
        sigma_eps = float(np.sqrt(sigma_eps2))
        eps_voigt_std = np.full(6, sigma_eps)
        Sigma_eps = np.eye(6) * sigma_eps2

        # Residuals
        predicted = np.sqrt(sig_inst2_arr + x * sigma_eps2)
        residuals = np.array([r[2] for r in rows]) - predicted

    # ── Mode: diagonal ────────────────────────────────────────────────────────
    elif mode == "diagonal":
        # y_k ≈ Σᵢ σᵢ² · ‖J_k[:,i]‖²   (trace proxy)
        # A[k, i] = J_k[0,i]² + J_k[1,i]²
        A = np.array([J[0] ** 2 + J[1] ** 2 for J in Js])  # (n_spots, 6)
        v, _ = nnls(A, y)  # v[i] = σᵢ²
        eps_voigt_std = np.sqrt(v)
        Sigma_eps = np.diag(v)
        sigma_eps = float(np.sqrt(np.mean(v)))

        # Residuals (using trace proxy for consistency)
        predicted = np.sqrt(sig_inst2_arr + A @ v)
        residuals = np.array([r[2] for r in rows]) - predicted

    else:
        raise ValueError(f"mode must be 'isotropic' or 'diagonal', got {mode!r}")

    return {
        "sigma_eps": sigma_eps,
        "eps_voigt_std": eps_voigt_std,
        "Sigma_eps": Sigma_eps,
        "residuals_pix": residuals,
        "hkl_used": hkl_used,
        "n_spots": len(rows),
        "mode": mode,
    }


def estimate_instrument_broadening(
    spots,
    sigma_meas_pix,
    mode="constant",
    tth_range=None,
    chi_range=None,
    min_spots=3,
):
    """
    Estimate the instrumental spot broadening from a calibrant measurement.

    For a strain-free calibrant (e.g. Si, CeO₂, LaB₆), all measured spot
    widths arise from instrumental contributions only (beam divergence, detector
    point-spread, geometric aberrations).  This function fits a model
    σ_instrument(2θ, χ) to those widths.

    The returned scalar ``'sigma_instrument'`` (or the callable model) can be
    passed directly to :func:`fit_strain_distribution` to subtract the
    instrumental baseline before fitting strain.

    Parameters
    ----------
    spots : list of dicts
        Simulated spots for the **calibrant** crystal, from :func:`simulate_laue`.
        Must contain ``'hkl'``, ``'two_theta'``, and ``'chi'`` keys.
    sigma_meas_pix : dict  {(h, k, l): float}
        Measured semi-major spot width (pixels, 1σ) for each indexed reflection
        of the calibrant, obtained by fitting a 2-D Gaussian to the detector image.
    mode : {'constant', 'linear_tth', 'quadratic_tth'}
        Model for the angular dependence of instrumental broadening:

        ``'constant'``
            Single value for all spots: median of the measured widths.
            Robust to outliers; use when the detector is well-focused.

        ``'linear_tth'``
            σ_inst(2θ) = a + b · 2θ (degrees).
            Captures defocus that increases with scattering angle.

        ``'quadratic_tth'``
            σ_inst(2θ) = a + b · 2θ + c · 2θ².
            Fits a second-order trend; requires ≥ 5 spots.

    tth_range : (float, float), optional
        Only include spots with 2θ in this range (degrees).
        Useful to exclude regions where the model fit is unreliable.
    chi_range : (float, float), optional
        Only include spots with χ in this range (degrees).
    min_spots : int, optional
        Minimum number of matching spots required. Default: 3.

    Returns
    -------
    result : dict with keys:

        ``'sigma_instrument'`` : float
            Scalar estimate of σ_instrument (pixels).
            For ``'constant'`` mode: the median.
            For parametric modes: the median of the fitted values at each spot.
        ``'model'`` : callable  f(tth_deg) → float
            Model function for σ_instrument as a function of 2θ (degrees).
            Always returned; for ``'constant'`` mode it returns the same scalar
            for any input.
        ``'params'`` : ndarray
            Fitted polynomial coefficients [a] or [a, b] or [a, b, c].
        ``'sigma_per_spot'`` : dict  {(h,k,l): float}
            Model-predicted σ_instrument for each spot used in the fit.
        ``'residuals_pix'`` : ndarray
            Measured − fitted σ_instrument per spot (pixels).
        ``'rmse_pix'`` : float
            Root-mean-square of residuals (pixels).
        ``'hkl_used'`` : list of tuples
            hkl of the spots that entered the fit.
        ``'n_spots'`` : int
            Number of spots used.
        ``'mode'`` : str
            The mode that was used.

    Raises
    ------
    ValueError
        If fewer than ``min_spots`` spots match (wrong hkl, out of range filters,
        or too few calibrant reflections on the detector).

    Notes
    -----
    * Robust to a few outlier spots in ``'constant'`` mode (median used).
    * For parametric modes the fit is an ordinary least-squares polynomial;
      strongly deviant spots can be manually excluded via ``tth_range``.
    * The model is evaluated at 2θ only.  If broadening has a strong χ
      dependence (e.g. astigmatism), measure and apply it separately.
    * A good calibrant should have sharp, well-separated spots with low
      absorption and no texture — Si (powder or single crystal cut along
      low-index direction), CeO₂, LaB₆, or Al₂O₃ are common choices.

    Example
    -------
    >>> # Simulate calibrant spots
    >>> si = crystal_from_cif('silicon.cif')
    >>> U_si = np.eye(3)  # known orientation
    >>> spots_cal = simulate_laue(si, U_si, camera)
    >>>
    >>> # Measure widths from experiment
    >>> sigma_meas_cal = {(1,1,1): 2.3, (2,2,0): 2.8, (3,1,1): 3.1,
    ...                   (4,0,0): 3.5, (3,3,1): 3.4}
    >>>
    >>> res = estimate_instrument_broadening(
    ...     spots_cal, sigma_meas_cal, mode='linear_tth')
    >>> print(f"σ_inst(2θ=30°) = {res['model'](30):.2f} px")
    >>>
    >>> # Use in strain fitting
    >>> sigma_inst_fn = res['model']   # or res['sigma_instrument'] (scalar)
    >>> J = strain_spot_jacobian(spots_sample, crystal, U, camera)
    >>> # Build per-spot sigma_instrument dict for fit_strain_distribution:
    >>> sigma_inst_per_spot = {
    ...     hkl: sigma_inst_fn(spots_sample_dict[hkl]['two_theta'])
    ...     for hkl in J}
    """
    # ── Build a lookup: hkl → (two_theta, chi) from simulated spots ──────────
    spot_info = {}
    for s in spots:
        hkl = tuple(s["hkl"])
        spot_info[hkl] = (float(s["two_theta"]), float(s["chi"]))

    # ── Intersect with measured keys ──────────────────────────────────────────
    common = set(spot_info.keys()) & set(sigma_meas_pix.keys())

    rows = []  # (hkl, tth, chi, sigma_meas)
    for hkl in common:
        tth, chi = spot_info[hkl]
        if tth_range is not None and not (tth_range[0] <= tth <= tth_range[1]):
            continue
        if chi_range is not None and not (chi_range[0] <= chi <= chi_range[1]):
            continue
        rows.append((hkl, tth, chi, float(sigma_meas_pix[hkl])))

    if len(rows) < min_spots:
        raise ValueError(
            f"Only {len(rows)} calibrant spot(s) matched (need ≥ {min_spots}). "
            "Check that sigma_meas_pix keys match the hkl in spots, "
            "or relax tth_range / chi_range."
        )

    hkl_used = [r[0] for r in rows]
    tth_arr = np.array([r[1] for r in rows])
    sigma_arr = np.array([r[3] for r in rows])

    # ── Fit model ─────────────────────────────────────────────────────────────
    if mode == "constant":
        params = np.array([float(np.median(sigma_arr))])

        def model(_tth_deg):
            return float(params[0])

    elif mode in ("linear_tth", "quadratic_tth"):
        deg = 1 if mode == "linear_tth" else 2
        if len(rows) < deg + 2:
            raise ValueError(
                f"mode='{mode}' requires ≥ {deg + 2} spots, got {len(rows)}."
            )
        params = np.polyfit(tth_arr, sigma_arr, deg)

        def model(tth_deg, _p=params):
            return float(np.polyval(_p, tth_deg))

    else:
        raise ValueError(
            f"mode must be 'constant', 'linear_tth', or 'quadratic_tth', got {mode!r}"
        )

    # ── Evaluate model at each spot ───────────────────────────────────────────
    fitted = np.array([model(t) for t in tth_arr])
    residuals = sigma_arr - fitted
    rmse = float(np.sqrt(np.mean(residuals**2)))

    sigma_per_spot = {hkl: float(model(tth)) for hkl, tth, *_ in rows}

    # Scalar summary: median of fitted values (constant mode → same as params[0])
    sigma_instrument = float(np.median(fitted))

    return {
        "sigma_instrument": sigma_instrument,
        "model": model,
        "params": params,
        "sigma_per_spot": sigma_per_spot,
        "residuals_pix": residuals,
        "rmse_pix": rmse,
        "hkl_used": hkl_used,
        "n_spots": len(rows),
        "mode": mode,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CORE LAUE SIMULATION
# ─────────────────────────────────────────────────────────────────────────────


def simulate_laue(
    crystal,
    U,
    camera,
    E_min=E_MIN_eV,
    E_max=E_MAX_eV,
    hmax=HMAX,
    f2_thresh=F2_THRESHOLD,
):
    """
    Simulate single-crystal white-beam Laue diffraction in reflection geometry.

    For every reciprocal-lattice vector G_hkl within the Miller-index shell
    ``[-hmax, hmax]^3`` the function:

    1. Rotates G from the crystal frame into the lab frame via the orientation
       matrix ``U``:  ``G_lab = U @ G_cry``.
    2. Applies the Laue condition to find the wavelength (and photon energy) at
       which this reflection is excited::

           lambda_hkl = -4*pi * (k_i_hat . G_lab) / |G_lab|^2

       Reflections whose wavelength falls outside ``[E_min, E_max]`` are
       skipped.
    3. Computes the scattered-beam direction ``kf_hat`` and projects it onto
       the detector plane via ``camera.project()``.  Reflections that miss the
       active area are discarded.
    4. Evaluates the spot intensity::

           I_raw(hkl) = |F(G, E)|^2  *  LP(2theta)  *  S(E)

       where:

       - ``|F(G, E)|^2``  – kinematical structure factor squared (Cromer-Mann
         ``f0`` plus Henke anomalous corrections ``f'``, ``f''`` via
         *xrayutilities*).  Reflections below ``f2_thresh`` are dropped.
       - ``LP(2theta)``   – Lorentz-polarisation factor for an unpolarised
         beam::

               LP = (1 + cos^2(2θ)) / (2 * sin^2(θ) * cos(θ))

       - ``S(E)``         – synchrotron spectral weight at energy ``E``
         (bending-magnet, wiggler, or undulator model set by the module-level
         ``SOURCE_TYPE``).

    5. Normalises all surviving ``I_raw`` values by the brightest spot so that
       ``intensity`` lies in ``(0, 1]``.

    Parameters
    ----------
    crystal : Crystal-like
        An *xrayutilities*-compatible crystal object that exposes:

        - ``crystal.Q(h, k, l)``  → reciprocal-lattice vector in crystal frame
          (Å⁻¹, 2π convention).
        - ``crystal.StructureFactor(G_cry, en=E)``  → complex structure factor
          at energy ``E`` (eV).

    U : array-like, shape (3, 3)
        Orientation matrix mapping crystal-frame vectors to the LaueTools lab
        frame (beam along ``+x``).  Typically obtained from
        ``euler_to_U(phi1, Phi, phi2)`` or an indexing result.

    camera : Camera
        Detector geometry object (see ``camera.py``).  Must implement
        ``camera.project(kf_hat)`` which returns ``(col, row)`` pixel
        coordinates or ``None`` if the ray misses the detector.

    E_min : float, optional
        Low-energy cut-off of the white beam in eV.
        Default: ``E_MIN_eV`` (5 000 eV).

    E_max : float, optional
        High-energy cut-off of the white beam in eV.
        Default: ``E_MAX_eV`` (27 000 eV).

    hmax : int, optional
        Maximum absolute Miller index to enumerate.  The search space is a
        cube ``[-hmax, hmax]^3`` (excluding 000).
        Default: ``HMAX`` (14).

    f2_thresh : float, optional
        Minimum ``|F|^2`` threshold (arbitrary units, same scale as
        *xrayutilities* output).  Reflections below this value are treated as
        systematically absent or too weak and discarded before the LP / spectrum
        weighting step.
        Default: ``F2_THRESHOLD`` (0.5).

    Returns
    -------
    list of dict
        One dictionary per spot that satisfies all selection criteria, sorted
        by **descending** ``intensity``.  Each dictionary contains:

        ==================  ====================================================
        Key                 Description
        ==================  ====================================================
        ``hkl``             ``(h, k, l)`` tuple of Miller indices.
        ``E``               Photon energy at which the reflection is excited (eV).
        ``lambda``          Corresponding wavelength (Å).
        ``tth``             Bragg angle ``2θ`` (degrees), measured from the
                            forward-beam direction ``+x``.
        ``chi``             LaueTools χ angle (degrees):
                            ``arctan2(kf_y, kf_z)``.
        ``az``              Azimuthal angle (degrees):
                            ``arctan2(kf_z, kf_y)``.
        ``pix``             ``(col, row)`` pixel coordinate on the detector
                            (LaueTools convention: ``xcam, ycam``).
        ``F2``              ``|F(G, E)|^2``, structure factor squared.
        ``LP``              Lorentz-polarisation factor.
        ``sw``              Synchrotron spectral weight ``S(E)``.
        ``I_raw``           Un-normalised intensity: ``F2 * LP * sw``.
        ``intensity``       ``I_raw`` normalised to ``[0, 1]`` by the
                            brightest spot in this simulation.
        ``is_superlattice`` ``True`` when ``|h|+|k|+|l|`` is odd (B2-type
                            superlattice reflection).
        ==================  ====================================================

        Returns an **empty list** if no reflection satisfies all criteria.

    Notes
    -----
    * The incident beam is fixed along ``+x`` in the LaueTools lab frame
      (``KI_HAT = [1, 0, 0]``).  Do not modify this without updating the
      camera geometry accordingly.
    * ``intensity`` is a *relative* quantity within a single call.  When
      comparing patterns from different phases or orientations use ``I_raw``
      and apply an external weighting (see ``simulate_mixed_phases``).
    * The ``is_superlattice`` flag uses the BCC extinction rule
      (``h+k+l`` odd → forbidden for BCC, but *allowed* for B2).  It is
      provided as a convenience tag; the structure factor already accounts for
      the actual systematic absences via ``crystal.StructureFactor``.
    """
    lam_lo = en2lam(E_max)
    lam_hi = en2lam(E_min)
    ki_hat = KI_HAT / np.linalg.norm(KI_HAT)

    spots = []
    for h in range(-hmax, hmax + 1):
        for k in range(-hmax, hmax + 1):
            for l in range(-hmax, hmax + 1):
                if h == 0 and k == 0 and l == 0:
                    continue

                G_cry = crystal.Q(h, k, l)
                G_lab = U @ G_cry
                Gm2 = float(np.dot(G_lab, G_lab))
                kdG = float(np.dot(ki_hat, G_lab))

                if kdG >= 0:
                    continue

                # Laue wavelength: lambda = -4*pi*(k_hat.G) / |G|^2
                lam = -4.0 * np.pi * kdG / Gm2  # Angstrom
                if not (lam_lo <= lam <= lam_hi):
                    continue

                E = lam2en(lam)

                # Scattered beam direction
                km = 2.0 * np.pi / lam
                kf_vec = ki_hat * km + G_lab
                kf_hat = kf_vec / np.linalg.norm(kf_vec)

                # Project onto camera
                pix = camera.project(kf_hat)
                if pix is None:
                    continue

                # 2theta, chi, azimuth  --  LaueTools LT frame (x // ki)
                # 2theta = arccos(kf_x)   [kf_x = component along beam]
                cos2th = np.clip(kf_hat[0], -1.0, 1.0)
                tth = np.degrees(np.arccos(cos2th))
                # chi: LaueTools convention  chi = arctan2(kf_y, kf_z)
                # y = z^x (horizontal), z up (close to detector normal)
                chi = np.degrees(np.arctan2(kf_hat[1], kf_hat[2] + 1e-17))
                az = np.degrees(np.arctan2(kf_hat[2], kf_hat[1]))

                # Structure factor (energy-dependent)
                F = crystal.StructureFactor(G_cry, en=E)
                F2 = abs(F) ** 2
                if F2 < f2_thresh:
                    continue

                LP = lorentz_pol(tth)
                if LP == 0.0:
                    continue

                sw = synchrotron_spectrum(E)
                if sw <= 0.0:
                    continue

                spots.append(
                    {
                        "hkl": (h, k, l),
                        "E": E,
                        "lambda": lam,
                        "tth": tth,
                        "chi": chi,
                        "az": az,
                        "pix": pix,
                        "F2": F2,
                        "LP": LP,
                        "sw": sw,
                        "I_raw": F2 * LP * sw,
                        "is_superlattice": is_superlattice(h, k, l),
                    }
                )

    if spots:
        imax = max(s["I_raw"] for s in spots)
        for s in spots:
            s["intensity"] = s["I_raw"] / imax

    return sorted(spots, key=lambda s: s["intensity"], reverse=True)


def simulate_laue_stack(
    stack,
    camera,
    E_min_eV=5_000,
    E_max_eV=27_000,
    source="bending_magnet",
    source_kwargs=None,
    hmax=12,
    f2_thresh=1.0,
    ki_hat=None,
    verbose=True,
):
    """
    Compute Laue spots for a ``LayeredCrystal`` stack projected onto ``camera``.

    For each phase in the stack the reciprocal lattice is enumerated up to
    ``hmax``.  Every G_lab that satisfies the Laue condition within the energy
    window is kept, and the structure factor is evaluated on the **full stack**
    at that Q.  This ensures coherent superposition of all layers and natural
    emergence of superlattice satellites.

    Parameters
    ----------
    stack : LayeredCrystal
        The layered structure (from layers.py).
    camera : Camera
        Detector geometry (from camera.py).
    E_min_eV, E_max_eV : float
        Energy window  (eV).
    source : str
        Synchrotron source model: ``'bending_magnet'``, ``'wiggler'``,
        ``'undulator'``, or ``'flat'`` (uniform spectrum).
    source_kwargs : dict, optional
        Extra keyword arguments forwarded to the spectrum function:
          bending_magnet / wiggler : ``Ec_eV``, ``N_poles``
          undulator                : ``E1_eV``, ``n_harm``, ``sig_rel``
    hmax : int
        Maximum Miller index to enumerate for each phase.
    f2_thresh : float
        Minimum |F_stack|²  to retain a spot (absolute, in e.u.²).
        Because the stack coherently sums many unit cells the absolute
        value scales roughly as (N_cells × N_rep)² at Bragg peaks.
        A value of ``None`` uses an auto-scaled threshold:
            f2_thresh = (max single-cell |F|)² × 0.001
        which keeps spots down to 0.1 % of the strongest unit-cell peak.
        For a manual value, typical starting point: ~10–1000.
    ki_hat : array-like (3,), optional
        Incident beam direction in the LaueTools LT frame (x // beam).
        Default: [1, 0, 0].
    verbose : bool

    Returns
    -------
    spots : list of dicts
        Same format as ``simulate_laue()`` in laue_white_synchrotron.py,
        plus extra keys:
          ``'phase_label'``  – which phase's hkl triggered this candidate
          ``'F2_stack'``     – |F_stack|² (full coherent stack)

    Notes
    -----
    For a stack with many layers / large N_cells the absolute values of
    ``F2_stack`` can be large (~N² × single-cell value at Bragg peaks).
    The returned ``intensity`` key is always normalised 0–1 within the
    returned list.

    Performance
    -----------
    Each spot requires one ``stack.structure_factor()`` call, which itself
    calls ``crystal.StructureFactor()`` once per layer.  For a 2-layer stack
    with ~4000 candidate spots, total time is ~2 s on a modern CPU.
    """
    stack._update_offsets()
    source_kwargs = source_kwargs or {}
    ki = np.asarray(ki_hat if ki_hat is not None else KI_HAT, dtype=float)
    ki /= np.linalg.norm(ki)

    lam_lo = en2lam(E_max_eV)
    lam_hi = en2lam(E_min_eV)

    def spectrum(E):
        if source == "bending_magnet":
            return spectrum_bm(E, **source_kwargs)
        elif source == "wiggler":
            kw = dict(N_poles=source_kwargs.get("N_poles", 40))
            kw["Ec_eV"] = source_kwargs.get("Ec_eV", 20_000)
            return spectrum_bm(E, **kw)
        elif source == "undulator":
            return spectrum_undulator(E, **source_kwargs)
        elif source == "flat":
            return 1.0
        raise ValueError(f"Unknown source: {source!r}")

    # Auto-scale f2_thresh if not provided
    if f2_thresh is None:
        f2_thresh = 0.0  # will be set after first structure factor call

    spots = []

    # Deduplicate: if two layers share the exact same crystal AND orientation,
    # we only enumerate once (the stack F already includes both contributions).
    seen_combos = []  # list of (crystal.name, U_rounded_tuple)

    for layer in stack.layers:
        crystal = layer.crystal
        U = layer.U
        label = layer.label

        # Skip enumeration if this (crystal, orientation) was already done
        u_key = (crystal.name, tuple(np.round(U, 4).ravel()))
        if u_key in seen_combos:
            if verbose:
                print(
                    f"  Skipping {label} (same crystal+orientation already enumerated)"
                )
            continue
        seen_combos.append(u_key)

        if verbose:
            print(f"  Enumerating {label} (hmax={hmax}) ...", end="", flush=True)

        n_added = 0
        for h in range(-hmax, hmax + 1):
            for k in range(-hmax, hmax + 1):
                for l in range(-hmax, hmax + 1):
                    if h == 0 and k == 0 and l == 0:
                        continue

                    # G in lab frame from this phase
                    G_cry = crystal.Q(h, k, l)
                    G_lab = U @ G_cry
                    Gm2 = float(np.dot(G_lab, G_lab))
                    kdG = float(np.dot(ki, G_lab))

                    if kdG >= 0:
                        continue

                    # Laue wavelength
                    lam = -4.0 * np.pi * kdG / Gm2
                    if not (lam_lo <= lam <= lam_hi):
                        continue

                    E = lam2en(lam)

                    # Scattered beam direction (in LT2 frame for camera)
                    km = 2.0 * np.pi / lam
                    kf_vec = ki * km + G_lab  # in LT frame
                    kf_hat = kf_vec / np.linalg.norm(kf_vec)

                    # Camera projection (camera.project expects LT frame)
                    pix = camera.project(kf_hat)
                    if pix is None:
                        continue

                    # 2θ and χ  (LaueTools LT convention)
                    tth = np.degrees(np.arccos(np.clip(kf_hat[0], -1.0, 1.0)))
                    chi = np.degrees(np.arctan2(kf_hat[1], kf_hat[2] + 1e-17))
                    az = np.degrees(np.arctan2(kf_hat[2], kf_hat[1]))

                    # ── FULL STACK structure factor at this Q ─────────────
                    F_stack = stack.structure_factor(G_lab, energy_eV=E)
                    F2 = abs(F_stack) ** 2
                    if f2_thresh == 0.0:
                        # Auto-scale: first valid spot sets the threshold
                        # Use 0.1% of the first computed |F|² as floor
                        f2_thresh = max(1.0, F2 * 1e-3)
                    if F2 < f2_thresh:
                        continue

                    # LP and spectrum
                    LP = lorentz_pol(tth)
                    if LP == 0.0:
                        continue
                    sw = spectrum(E)
                    if sw <= 0.0:
                        continue

                    spots.append(
                        {
                            "phase_label": label,
                            "hkl": (h, k, l),
                            "G_lab": G_lab.copy(),
                            "E": E,
                            "lambda": lam,
                            "tth": tth,
                            "chi": chi,
                            "az": az,
                            "pix": pix,
                            "F2": F2,
                            "F2_stack": F2,
                            "LP": LP,
                            "sw": sw,
                            "I_raw": F2 * LP * sw,
                            # is_superlattice: True when h+k+l is odd
                            # (meaningful for BCC-based phases; kept for compat.)
                            "is_superlattice": (abs(h) + abs(k) + abs(l)) % 2 == 1,
                        }
                    )
                    n_added += 1

        if verbose:
            print(f" {n_added} spots")

    # Normalise
    if spots:
        imax = max(s["I_raw"] for s in spots)
        for s in spots:
            s["intensity"] = s["I_raw"] / imax

    spots.sort(key=lambda s: s["intensity"], reverse=True)

    if verbose:
        print(f"  Total spots on detector: {len(spots)}")
        print(
            f"  Stack: {stack.name}  "
            f"Λ={stack.bilayer_thickness:.2f} Å  "
            f"N_rep={stack.n_rep}"
        )
        print(
            f"  Superlattice satellite 2π/Λ = "
            f"{2*np.pi/stack.bilayer_thickness:.5f} Å⁻¹"
        )

    return spots


def simulate_mixed_phases(
    phases,
    camera,
    E_min_eV=5_000,
    E_max_eV=27_000,
    source="bending_magnet",
    source_kwargs=None,
    hmax=12,
    f2_thresh=None,
    normalise="volume",
    verbose=True,
):
    """
    Simulate a Laue pattern from a multi-phase sample with known volume
    fractions.

    Each phase scatters **independently** (incoherent between phases —
    different grains, different orientations).  The contribution of each
    phase is weighted by its volume fraction and unit-cell number density
    before the spot lists are merged into one.

    Intensity weighting
    -------------------
    The number of unit cells of phase p contributing to diffraction scales as:

        N_uc_p  ∝  f_p / V_uc_p

    where  f_p  is the volume fraction and  V_uc_p  is the unit-cell volume
    (Å³).  This is the standard Rietveld weight used in powder diffraction
    and is correct for any single-crystal Laue measurement of a multi-phase
    polycrystal.

    For a ``LayeredCrystal`` phase, V_uc_p is taken as the thickness-weighted
    harmonic mean of the individual layer unit-cell volumes — i.e. the
    effective number of unit cells per unit volume of the stack.

    The ``normalise`` argument controls how the final intensities are scaled:
      ``'volume'``   (default) — weight by f_p / V_uc_p  (physics-correct)
      ``'fraction'`` — weight by f_p only (ignore V_uc differences)
      ``'equal'``    — all phases equally weighted regardless of fraction
      ``'none'``     — no rescaling; I_raw values are kept as-is from each
                       phase's simulation

    Parameters
    ----------
    phases : list of dicts or list of tuples
        Each entry describes one phase.  Accepted formats:

        **dict** (recommended)::

            {
              'crystal'         : xu.materials.Crystal  or  LayeredCrystal,
              'U'               : np.ndarray (3×3) orientation matrix,
              'volume_fraction' : float,          # must sum to 1 (normalised)
              'label'           : str,            # optional, default crystal.name
              'hmax'            : int,            # optional, overrides global hmax
              'f2_thresh'       : float | None,   # optional, overrides global
            }

        **tuple** (short form)::

            (crystal_or_stack, U, volume_fraction)
            (crystal_or_stack, U, volume_fraction, label)

    camera : Camera
        Detector geometry (from laue_white_synchrotron.py).

    E_min_eV, E_max_eV : float
        Energy window (eV), applied to all phases.

    source : str
        Synchrotron source: ``'bending_magnet'``, ``'wiggler'``,
        ``'undulator'``, or ``'flat'``.

    source_kwargs : dict, optional
        Forwarded to the spectrum function (e.g. ``{'Ec_eV': 20000}``).

    hmax : int
        Maximum Miller index (global default, overridable per phase).

    f2_thresh : float | None
        Minimum |F|² threshold (global default, overridable per phase).
        ``None`` = auto-scale per phase.

    normalise : str
        Weighting mode: ``'volume'``, ``'fraction'``, ``'equal'``, ``'none'``.

    verbose : bool

    Returns
    -------
    spots : list of dicts
        Merged, weighted, and renormalised spot list.  Each dict has all the
        standard keys plus:

          ``'phase_label'``      – which phase this spot belongs to
          ``'volume_fraction'``  – f_p of that phase
          ``'phase_weight'``     – the weight applied (f_p / V_uc_p or variant)
          ``'intensity'``        – normalised 0–1 over the full mixed pattern
          ``'intensity_phase'``  – normalised 0–1 within that phase alone

    Raises
    ------
    ValueError
        If volume fractions do not sum to approximately 1.0 (within ±0.01).

    Examples
    --------
    >>> import xrayutilities as xu
    >>> from simulate_laue_layered import simulate_mixed_phases
    >>> from layered_structure_factor import orientation_along_z, or_kurdjumov_sachs
    >>>
    >>> Fe = xu.materials.Fe
    >>> Cu = xu.materials.Cu
    >>> U_Fe = orientation_along_z([0,0,1], Fe)
    >>> U_Cu = U_Fe @ or_kurdjumov_sachs(Fe, Cu).T
    >>>
    >>> phases = [
    ...     {'crystal': Fe, 'U': U_Fe, 'volume_fraction': 0.6, 'label': 'austenite'},
    ...     {'crystal': Cu, 'U': U_Cu, 'volume_fraction': 0.4, 'label': 'Cu KS'},
    ... ]
    >>> spots = simulate_mixed_phases(phases, camera)
    >>> plot_detector_image(spots, camera, colour_by='phase')

    Notes
    -----
    Orientation relationship between phases does NOT produce interference
    fringes here — use ``LayeredCrystal`` + ``simulate_laue_stack`` for that.
    This function is for incoherent multi-grain mixtures (e.g. a polycrystal
    with two phases, or a transformed microstructure).
    """
    import os
    import sys

    import numpy as np
    import xrayutilities as xu

    from .layers import LayeredCrystal

    source_kwargs = source_kwargs or {}

    # ── Normalise phase list ──────────────────────────────────────────────────
    parsed = []
    for entry in phases:
        if isinstance(entry, dict):
            p = entry.copy()
        elif isinstance(entry, (list, tuple)):
            p = {}
            p["crystal"] = entry[0]
            p["U"] = entry[1]
            p["volume_fraction"] = float(entry[2])
            if len(entry) >= 4:
                p["label"] = str(entry[3])
        else:
            raise TypeError(f"Each phase must be a dict or tuple, got {type(entry)}")
        parsed.append(p)

    # Default label
    for p in parsed:
        if "label" not in p:
            c = p["crystal"]
            p["label"] = c.name if hasattr(c, "name") else str(c)

    # Check fractions
    total_f = sum(float(p["volume_fraction"]) for p in parsed)
    if abs(total_f - 1.0) > 0.01:
        raise ValueError(
            f"Volume fractions sum to {total_f:.4f}, expected 1.0. "
            "Please normalise them."
        )

    # ── Compute effective unit-cell volume per phase ───────────────────────────
    def eff_vuc(crystal_or_stack):
        """Effective unit-cell volume (Å³) for weighting."""
        if isinstance(crystal_or_stack, LayeredCrystal):
            stk = crystal_or_stack
            stk._update_offsets()
            total_t = stk.total_thickness
            if total_t < 1e-10:
                return 1.0
            # Thickness-weighted harmonic mean of layer V_uc values
            # = effective V_uc per unit volume of the stack
            w_sum = 0.0
            for layer in stk.layers:
                frac = layer.thickness / total_t
                w_sum += frac / layer.crystal.lattice.UnitCellVolume()
            return 1.0 / w_sum if w_sum > 1e-30 else 1.0
        else:
            return crystal_or_stack.lattice.UnitCellVolume()

    # # ── Load simulation helpers ────────────────────────────────────────────────
    # mod_src = open(os.path.join(_here, "laue_white_synchrotron.py")).read()
    # ns = {}
    # exec(compile(mod_src.split("\ndef main():")[0], "laue_sim", "exec"), ns)
    # _simulate_laue_single = ns["simulate_laue"]

    # ── Simulate each phase ────────────────────────────────────────────────────
    if verbose:
        print(f"\n  Mixed-phase Laue simulation  ({len(parsed)} phases)")
        print(f"  {'─'*52}")
        print(f"  {'Phase':22s} {'f_vol':>6}  {'V_uc(Å³)':>10}  {'weight':>10}")

    phase_results = []
    for p in parsed:
        crystal = p["crystal"]
        U = np.asarray(p["U"], dtype=float)
        f = float(p["volume_fraction"])
        label = p["label"]
        ph_hmax = int(p.get("hmax", hmax))
        ph_f2 = p.get("f2_thresh", f2_thresh)

        vuc = eff_vuc(crystal)

        # Compute weight
        if normalise == "volume":
            weight = f / vuc
        elif normalise == "fraction":
            weight = f
        elif normalise == "equal":
            weight = 1.0 / len(parsed)
        elif normalise == "none":
            weight = 1.0
        else:
            raise ValueError(f"normalise must be 'volume','fraction','equal','none'")

        if verbose:
            print(f"  {label:22s} {f:6.3f}  {vuc:10.3f}  {weight:10.6f}")

        # Simulate
        if isinstance(crystal, LayeredCrystal):
            spots_p = simulate_laue_stack(
                crystal,
                camera,
                E_min_eV=E_min_eV,
                E_max_eV=E_max_eV,
                source=source,
                source_kwargs=source_kwargs,
                hmax=ph_hmax,
                f2_thresh=ph_f2,
                verbose=False,
            )
        else:
            spots_p = simulate_laue(
                crystal,
                U,
                camera,
                E_min=E_min_eV,
                E_max=E_max_eV,
                hmax=ph_hmax,
                f2_thresh=(ph_f2 if ph_f2 is not None else 0.5),
            )

        if verbose:
            print(f"    → {len(spots_p)} spots on camera")

        # Tag each spot and apply weight to I_raw
        for s in spots_p:
            s["phase_label"] = label
            s["volume_fraction"] = f
            s["phase_weight"] = weight
            s["I_raw_weighted"] = s["I_raw"] * weight

        # Store per-phase normalised intensity (within this phase)
        if spots_p:
            imax_p = max(s["I_raw"] for s in spots_p)
            for s in spots_p:
                s["intensity_phase"] = s["I_raw"] / imax_p if imax_p > 0 else 0.0
        else:
            for s in spots_p:
                s["intensity_phase"] = 0.0

        phase_results.append((label, f, vuc, weight, spots_p))

    # ── Merge and renormalise ──────────────────────────────────────────────────
    all_spots = []
    for _, _, _, _, spots_p in phase_results:
        all_spots.extend(spots_p)

    if all_spots:
        imax = max(s["I_raw_weighted"] for s in all_spots)
        for s in all_spots:
            s["intensity"] = s["I_raw_weighted"] / imax if imax > 0 else 0.0

    all_spots.sort(key=lambda s: s["intensity"], reverse=True)

    if verbose:
        print(f"  {'─'*52}")
        print(f"  Total spots on camera : {len(all_spots)}")
        for label, f, vuc, weight, spots_p in phase_results:
            pct = len(spots_p) / max(len(all_spots), 1) * 100
            print(
                f"    {label:22s}: {len(spots_p):4d} spots  " f"({pct:.0f}% of total)"
            )
        print(f"  normalise = '{normalise}'")

    return all_spots


# ─────────────────────────────────────────────────────────────────────────────
# PER-LAYER INTENSITY DECOMPOSITION
# ─────────────────────────────────────────────────────────────────────────────


def layer_contributions_spots(spots, stack):
    """
    Decompose the intensity of each Laue spot into per-layer contributions.

    For each spot the total structure factor is:
        F_total(Q) = S_rep(Q) · Σ_l  F_layer_l(Q)

    where S_rep is the geometric factor for bilayer repetitions.
    Each layer contributes a complex amplitude F_l = S_rep · F_layer_l.

    The intensity is decomposed as:
        I_total = |F_total|²
        I_l     = Re(F_l · F_total*) / |F_total|²  × I_total

    This is the **coherent intensity contribution** of layer l.  It sums
    exactly to I_total over all layers (including interference terms):
        Σ_l  Re(F_l · F_total*) = |F_total|²

    Parameters
    ----------
    spots : list of dicts
        Output of ``simulate_laue_stack()``.  Must contain ``'G_lab'`` and
        ``'E'`` keys (present by default).
    stack : LayeredCrystal

    Returns
    -------
    spots : the same list, each dict extended with:
        ``'layer_F'``      : dict  { label : complex amplitude F_l }
        ``'layer_I'``      : dict  { label : float  absolute intensity I_l }
        ``'layer_I_frac'`` : dict  { label : float  fraction 0-1 (sums to 1) }

    Notes
    -----
    Negative fractions are physically meaningful — they indicate a layer
    that **destructively interferes** with the rest of the stack at that Q.
    The sum over all layers is still exactly 1.

    Example
    -------
    >>> spots = simulate_laue_stack(stack, camera)
    >>> spots = layer_contributions_spots(spots, stack)
    >>> for s in spots[:5]:
    ...     print(s['hkl'], s['layer_I_frac'])
    """
    stack._update_offsets()
    Lambda = stack._bilayer_thickness
    labels = [layer.label for layer in stack.layers]

    for s in spots:
        Q = np.asarray(s["G_lab"], dtype=float)
        E = float(s["E"])
        Qz = Q[2]

        # Geometric repetition factor S_rep
        phi_rep = Qz * Lambda
        phi_mod = phi_rep % (2.0 * np.pi)
        if abs(phi_mod) < 1e-10 or abs(phi_mod - 2 * np.pi) < 1e-10:
            S_rep = float(stack.n_rep) + 0j
        else:
            S_rep = (1.0 - np.exp(1j * stack.n_rep * phi_rep)) / (
                1.0 - np.exp(1j * phi_rep)
            )

        # Per-layer amplitude (including S_rep)
        layer_F = {}
        for layer, z0 in zip(stack.layers, stack._z_offsets):
            F_l = layer.structure_factor(Q, E, z0=z0) * S_rep
            lbl = layer.label
            layer_F[lbl] = layer_F.get(lbl, 0j) + F_l

        # Total F (should match s['F2'] ** 0.5 up to LP/spectrum)
        F_total = sum(layer_F.values())
        F2_total = abs(F_total) ** 2

        # Coherent intensity fractions
        if F2_total > 1e-30:
            layer_I_frac = {
                lbl: float((F * F_total.conjugate()).real / F2_total)
                for lbl, F in layer_F.items()
            }
        else:
            layer_I_frac = {lbl: 0.0 for lbl in layer_F}

        # Absolute per-layer intensities (×LP×S for comparison with spots)
        I_total = s["I_raw"]
        layer_I = {lbl: frac * I_total for lbl, frac in layer_I_frac.items()}

        s["layer_F"] = layer_F
        s["layer_I"] = layer_I
        s["layer_I_frac"] = layer_I_frac

    return spots


# ─────────────────────────────────────────────────────────────────────────────
# REPORTING
# ─────────────────────────────────────────────────────────────────────────────
def print_layer_contributions(spots, n=15):
    """
    Pretty-print per-layer intensity contributions for the top N spots.
    Requires ``layer_contributions_spots()`` to have been called first.
    """
    if not spots or "layer_I_frac" not in spots[0]:
        raise ValueError("Call layer_contributions_spots(spots, stack) first.")

    labels = list(spots[0]["layer_I_frac"].keys())
    col_w = max(12, max(len(l) for l in labels))

    header = (
        f"  {'phase':12s} {'hkl':^10} {'E(keV)':>7} "
        f"{'2th':>6} {'I/Imax':>7}  "
        + "  ".join(f"{l[:col_w]:>{col_w}}" for l in labels)
    )
    print(f"\n  Per-layer intensity fractions  (top {n} spots)")
    print("  " + "─" * len(header))
    print(header)
    print("  " + "─" * len(header))

    for s in spots[:n]:
        h, k, l = s["hkl"]
        fracs = "  ".join(
            f"{s['layer_I_frac'].get(lbl, 0.)*100:>{col_w}.1f}%" for lbl in labels
        )
        print(
            f"  {s['phase_label']:12s} ({h:+d}{k:+d}{l:+d})  "
            f"{s['E']/1e3:7.3f} {s['tth']:6.1f} "
            f"{s['intensity']:7.4f}  {fracs}"
        )


def print_mixed_summary(spots, top_n=20):
    """
    Print a summary table of the strongest spots in a mixed-phase pattern,
    grouped by phase.

    Parameters
    ----------
    spots : list of dicts from ``simulate_mixed_phases()``
    top_n : int  number of strongest spots to list per phase
    """
    from collections import defaultdict

    import numpy as np

    # Group by phase
    by_phase = defaultdict(list)
    for s in spots:
        by_phase[s["phase_label"]].append(s)

    print(f"\n  Mixed-phase Laue spot summary")
    print(f"  Total spots: {len(spots)}")

    for label, phase_spots in by_phase.items():
        f = phase_spots[0]["volume_fraction"]
        w = phase_spots[0]["phase_weight"]
        print(
            f"\n  ── {label}  (f={f:.3f}  weight={w:.6f})"
            f"  –  {len(phase_spots)} spots ──"
        )
        print(
            f"  {'hkl':^10} {'E(keV)':>7} {'2th':>7} {'chi':>7} "
            f"{'I/Imax':>8} {'I_phase':>8}  type"
        )
        print("  " + "─" * 68)
        top = sorted(phase_spots, key=lambda s: s["intensity"], reverse=True)
        for s in top[:top_n]:
            h, k, l = s["hkl"]
            tag = "superl." if s.get("is_superlattice") else "fund."
            print(
                f"  ({h:+d}{k:+d}{l:+d})  "
                f"{s['E']/1e3:7.3f} {s['tth']:7.2f} {s['chi']:7.2f} "
                f"{s['intensity']:8.4f} {s.get('intensity_phase',0):8.4f}  {tag}"
            )


def print_spot_table(title, spots, n=15):
    print(f"\n  ── {title} ──  ({len(spots)} total spots on camera)")
    print(
        f"  {'hkl':^10} {'E(keV)':>7} {'lambda(A)':>9} {'2th(deg)':>9} "
        f"{'az(deg)':>8} {'col':>6} {'row':>6} "
        f"{'|F|^2':>8} {'LP':>7} {'S(E)':>7} {'I/Imax':>7}  type"
    )
    print("  " + "-" * 110)
    for s in spots[:n]:
        h, k, l = s["hkl"]
        c, r = s["pix"]
        tag = "superlat." if s["is_superlattice"] else "fund."
        print(
            f"  ({h:+2d}{k:+2d}{l:+2d})  "
            f"{s['E']/1e3:7.3f}  {s['lambda']:9.5f}  "
            f"{s['tth']:9.3f}  {s['az']:8.2f}  "
            f"{c:6.0f}  {r:6.0f}  "
            f"{s['F2']:8.2f}  {s['LP']:7.4f}  "
            f"{s['sw']:7.4f}  {s['intensity']:7.4f}  {tag}"
        )


def print_bragg_table(a):
    print("\n  ── Bragg-energy reference: E for 2theta=90° (side detector) ──")
    print(f"  {'(hkl)':^7} {'d (A)':>8} {'E at 90deg (keV)':>18}")
    print("  " + "-" * 38)
    for hkl in [
        (1, 1, 0),
        (2, 0, 0),
        (2, 1, 1),
        (2, 2, 0),
        (2, 2, 2),
        (3, 1, 0),
        (3, 2, 1),
        (4, 0, 0),
        (3, 3, 0),
        (4, 0, 4),
        (4, 4, 0),
        (5, 1, 0),
    ]:
        h, k, l = hkl
        if (h + k + l) % 2 != 0:
            continue
        d = a / np.sqrt(h * h + k * k + l * l)
        E90 = lam2en(d * np.sqrt(2))
        print(f"  ({h}{k}{l})    {d:8.4f}  {E90/1e3:18.3f}")
