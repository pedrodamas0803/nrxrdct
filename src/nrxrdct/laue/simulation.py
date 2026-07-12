"""
White-Beam Synchrotron Laue Diffraction – Reflection Geometry
==============================================================
Simulates single-crystal Laue diffraction with a synchrotron white beam
in reflection geometry, with a full pixelated camera model.

System  : equiatomic AlCoCrFeNi HEA  –  BCC (Im-3m) + B2 (Pm-3m)

**Physics**

Laue condition (Ewald construction):

$$
\\lambda_{hkl} = -\\frac{4\\pi\\,(\\hat{k}_i \\cdot \\mathbf{G}_{hkl})}{|\\mathbf{G}_{hkl}|^2}
$$

Spot intensity:

$$
I(hkl) = |F(\\mathbf{G}, E)|^2 \\cdot LP(2\\theta) \\cdot S(E)
$$

where

- $F(\\mathbf{G}, E)$ — structure factor via xrayutilities (Cromer–Mann $f^0$ + Henke $f'(E)$, $f''(E)$ anomalous corrections)
- $LP(2\\theta)$ — Lorentz–polarisation factor (unpolarised beam):

$$
LP = \\frac{1 + \\cos^2 2\\theta}{2\\sin^2\\theta\\,\\cos\\theta}
$$

- $S(E)$ — synchrotron spectrum (bending magnet, wiggler, or undulator); no bremsstrahlung

**Synchrotron spectra**

Bending magnet / wiggler (on-axis, Kim 1989):

$$
S(E) \\propto \\left(\\frac{E}{E_c}\\right)^2 K_{2/3}^2\\!\\left(\\frac{E}{2E_c}\\right)
$$

Peak at $E \\approx 0.83\\,E_c$.  Wiggler: flux $\\times 2N_\\text{poles}$.

Undulator (planar, odd harmonics):

$$
S(E) = \\sum_n \\frac{1}{n}\\exp\\!\\left[-\\tfrac{1}{2}
\\left(\\frac{E - nE_1}{\\sigma_n}\\right)^2\\right]
$$

**Camera model**

The detector is a flat pixelated area detector (e.g. Eiger, Pilatus,
MAR, Perkin-Elmer, …) described by:

- `PIXEL_SIZE_MM` — pixel pitch (mm)
- `N_PIX_H`, `N_PIX_V` — number of pixels (horizontal, vertical)
- `DET_DIST_MM` — sample-to-detector-centre distance (mm)
- `TTH_CENTER_DEG` — $2\\theta$ angle at the detector centre (deg); can be any angle, not restricted to 90°
- `NU_DEG` — out-of-plane (elevation) angle of detector centre
- `CHI_DEG` — in-plane rotation of detector about its own normal

For each diffracted beam direction $\\hat{k}_f$ the code:

1. Intersects the ray with the detector plane (exact geometry).
2. Converts the hit position to (col, row) pixel coordinates.
3. Renders a synthetic image with Gaussian spot profiles whose width is set by `SPOT_SIGMA_PIX`.

The direct-beam footprint on the detector is also computed (if it
would hit) so you can check the geometry is sensible.

**Orientation**

Full orientation via Bunge ZXZ Euler angles $(\phi_1, \\Phi, \\phi_2)$.
A Bragg-energy reference table is printed at runtime.
"""

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.special import kv

# Module-level cache: kb_params tuple key → (E_grid, R_grid) arrays.
# Populated lazily on the first simulate_laue_stack call with a given KB setup
# so the 300-point xrayutilities loop never runs more than once per configuration.
_KB_CACHE: dict = {}

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
HMAX = 12
F2_THRESHOLD = 1e-6

# Module-level cache: (crystal_name, E_max_eV, E_ref_eV, f2_thresh) → frozenset
_allowed_hkl_cache: dict = {}

# Module-level cache: id(allowed_hkl) → (hkl_arr, G_cry_arr, crystal_name)
# Keyed on the Python object id of the frozenset returned by precompute_allowed_hkl,
# which is stable for the lifetime of the cached object.  Avoids re-running
# np.array(list(frozenset)) and the B @ hkl_arr.T matmul on every residual call.
_hkl_arrays_cache: dict = {}


def clear_allowed_hkl_cache() -> None:
    """Clear the module-level cache used by :func:`precompute_allowed_hkl`."""
    _allowed_hkl_cache.clear()
    _hkl_arrays_cache.clear()

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
    `simulate_laue` expects U in the **lab frame** (x // beam).  When the
    sample is tilted on the stage the two frames differ by a rotation around
    the horizontal axis (y in LT).

    Args:
    phi1, Phi, phi2 : float
        Bunge ZXZ Euler angles in degrees.
    sample_tilt_deg : float, optional
        Tilt of the sample surface relative to the horizontal plane (deg).
        Positive = front edge of sample tilted downward so the surface faces
        the incoming beam (standard reflection geometry).

    Returns:
    U : ndarray, shape (3, 3)
        Orientation matrix such that `G_lab = U @ G_crystal`.

    Note:
    The sample tilt is the **sample→lab** rotation about **+y** (horizontal
    axis perpendicular to the beam) by `sample_tilt_deg`:

        R_tilt = Ry(+sample_tilt_deg)

    This maps the sample surface normal from +z (horizontal surface) to
    (sin θ, 0, cos θ) in the lab frame, which for θ = 40° gives a grazing
    angle of 40° with the beam and a specular 2θ of 80°, consistent with the
    BM32 Z>0 top-camera geometry.

    This is the **inverse** of the LaueTools `matstarlab_to_matstarsample3x3`
    convention, which applies `Rx_LT2(+omega)` (rotation around x in the LT2
    frame) as the lab→sample direction.  Because `x_LT2 = −y_LT`, that
    operation equals `Ry_LT(−omega)` (lab→sample), so the sample→lab
    direction used here is `Ry_LT(+omega)`.

    When Euler angles come from a LaueTools indexing result (grain_matrix /
    deviatoric matrix) they are already expressed in the lab frame; pass
    `sample_tilt_deg=0` (the default) in that case.
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


def rotate_U_about_axis(U, angle_deg, axis: str = "z"):
    """
    Rotate an orientation matrix by *angle_deg* about a lab-frame axis.

    Args:
    U : array-like, shape (3, 3)
        Orientation matrix in the lab frame (`G_lab = U @ G_crystal`).
    angle_deg : float
        Rotation angle in degrees.  Positive = right-hand rule about the axis.
    axis : `'x'` | `'y'` | `'z'`
        Lab-frame axis to rotate about.  `'z'` is the surface normal in the
        standard BM32 / LT geometry; `'x'` is along the beam; `'y'` is
        horizontal and perpendicular to the beam.

    Returns:
    U_rot : ndarray, shape (3, 3)
        Rotated orientation matrix: `U_rot = R(axis, angle_deg) @ U`.

    Example:
    Rotate a GaN (001) orientation by 30° around the surface normal (z)::

        U0  = orientation_along_z(GaN, [0, 0, 1], [1, 0, 0])
        U30 = rotate_U_about_axis(U0, 30.0, axis='z')

    Tilt by 2° around the beam direction (x) to simulate a small miscut::

        U_tilt = rotate_U_about_axis(U0, 2.0, axis='x')
"""
    axis = axis.lower().strip()
    if axis not in ("x", "y", "z"):
        raise ValueError(f"axis must be 'x', 'y', or 'z', got {axis!r}")
    R = Rotation.from_euler(axis.upper(), angle_deg, degrees=True).as_matrix()
    return R @ np.asarray(U, dtype=float)


def rotate_U_about_crystal_axis(
    U: np.ndarray,
    angle_deg: float,
    crystal_axis: np.ndarray,
) -> np.ndarray:
    """
    Rotate an orientation matrix by *angle_deg* about a crystal-frame axis.

    The rotation axis is specified in the **crystal frame** and is first
    mapped into the lab frame via `U` before the rotation is applied.
    This is the natural way to express rotations such as "60° about the
    c-axis [0001]" or "180° about an in-plane direction [1, 0, 0]".

    Args:
    U : array-like, shape (3, 3)
        Orientation matrix in the lab frame (`G_lab = U @ G_crystal`).
    angle_deg : float
        Rotation angle in degrees.  Positive = right-hand rule about the
        crystal axis as expressed in the lab frame.
    crystal_axis : array-like, shape (3,)
        Rotation axis in the **crystal frame** (does not need to be a unit
        vector; it is normalised internally).  Examples:

        * `[0, 0, 1]` — $c$-axis (for hexagonal / tetragonal crystals)
        * `[1, 0, 0]` — $a$-axis
        * `[1, 1, 0]` — diagonal in-plane direction

    Returns:
    U_rot : ndarray, shape (3, 3)
        Rotated orientation matrix.

    Example:
    60° rotation about the GaN $c$-axis (rotational domain variant)::

        U_domain = rotate_U_about_crystal_axis(U, 60.0, [0, 0, 1])

    180° flip about the in-plane $a$-axis::

        U_flip = rotate_U_about_crystal_axis(U, 180.0, [1, 0, 0])
"""
    U = np.asarray(U, dtype=float)
    axis_cry = np.asarray(crystal_axis, dtype=float)
    axis_cry = axis_cry / np.linalg.norm(axis_cry)
    axis_lab = U @ axis_cry
    axis_lab = axis_lab / np.linalg.norm(axis_lab)
    R = Rotation.from_rotvec(np.radians(angle_deg) * axis_lab).as_matrix()
    return R @ U


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
    Convert a LaueTools `matstarlab` (LT2/OR frame, no 2π) into an effective
    orientation matrix for `simulate_laue` (LT frame, with 2π from xrayutilities).

    This function returns the **full deformation gradient** F = U @ P, which
    combines the pure crystal rotation U with the right-stretch tensor P
    (lattice distortion due to strain).  Passing F to `simulate_laue` gives
    spot positions that account for both the orientation **and** any elastic
    strain in the grain.

    To separate rotation from strain use :func:`decompose_matstarlab`.

    LaueTools defines (LT2 frame, no 2π)::

        G_LT2 = matstarlab @ [h, k, l]

    This function applies two corrections:

    1. **Frame change LT2→LT**: `x_LT = y_LT2`, `y_LT = −x_LT2`, `z_LT = z_LT2`
    2. **2π rescaling**: LaueTools uses |G| = 1/d; xrayutilities uses |G| = 2π/d.

    Args:
    matstarlab : array-like, shape (3, 3)
        LaueTools grain matrix in LT2/OR frame (columns = a*, b*, c* in lab,
        in Å⁻¹ **without** the 2π factor).
    crystal : xu.materials.Crystal
        Reference (unstrained) phase — same object passed to `simulate_laue`.

    Returns:
    F : ndarray, shape (3, 3)
        Deformation gradient in LT frame.  Pass directly to `simulate_laue`
        as the `U` argument to include strain in the spot geometry.
        For a strain-free grain F is a pure rotation matrix.
"""
    return _matstarlab_to_F(matstarlab, crystal)


def decompose_matstarlab(matstarlab, crystal):
    """
    Decompose a LaueTools `matstarlab` into pure rotation and elastic strain.

    Uses the **right polar decomposition** of the deformation gradient F:

    $$
    F = U \\cdot P
    $$
    where:

    * **F** — full deformation gradient (LT frame) = what `U_from_matstarlab` returns
    * **U** — pure rotation (orthogonal, det = +1): the rigid crystal orientation
    * **P** — right-stretch tensor (symmetric positive-definite): the lattice distortion

    The small-strain tensor is extracted from P as `ε = P − I`.

    Args:
    matstarlab : array-like, shape (3, 3)
        LaueTools grain matrix in LT2/OR frame (no 2π).
    crystal : xu.materials.Crystal
        Reference (unstrained) crystal — same object passed to `simulate_laue`.

    Returns:
    U : ndarray, shape (3, 3)
        Pure rotation in LT frame (orthogonal, det ≈ +1).  Use this in
        `simulate_laue` when you want rotation-only simulation (strain
        effects on peak positions are ignored).
    F : ndarray, shape (3, 3)
        Full deformation gradient (= `U_from_matstarlab` output).  Use
        this in `simulate_laue` to include strain in the spot geometry.
    eps : ndarray, shape (3, 3)
        Small-strain tensor in the crystal frame: `ε = P − I`.
        Diagonal entries are normal strains (ε₁₁, ε₂₂, ε₃₃);
        off-diagonal entries are shear strains (engineering convention ×½).
    eps_voigt : ndarray, shape (6,)
        Voigt representation `[ε₁₁, ε₂₂, ε₃₃, ε₂₃, ε₁₃, ε₁₂]`.

    Note:
    * The decomposition is exact (no small-strain approximation).
    * For strains ≲ 10⁻³ (typical elastic), P ≈ I and F ≈ U.
    * The strain ε is expressed in the **crystal frame** (principal axes of P).
      To express it in the lab frame: `ε_lab = U @ ε @ U.T`.
    * To check: `np.allclose(U @ (eps + np.eye(3)) @ B0, F @ B0)` should hold.

    Example:
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
# SYNCHROTRON SPECTRA
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
# KB MIRROR REFLECTIVITY
# ─────────────────────────────────────────────────────────────────────────────

#: Default KB parameters for BM32/ESRF (Rh-coated, 2 mirrors, ~2.5 mrad grazing)
BM32_KB = dict(
    material="Ir",
    grazing_angle_mrad=2.8,
    n_mirrors=2,
    roughness_ang=3.0,
)


def kb_reflectivity(
    energy_eV,
    material: str = "Ir",
    grazing_angle_mrad: float = 2.8,
    n_mirrors: int = 2,
    roughness_ang: float = 3.0,
) -> float:
    """
    Fresnel reflectivity of a KB mirror system (s-polarisation, kinematical limit).

    Uses the optical constants δ, β of the coating material (via *xrayutilities*)
    to compute the critical-angle total-external-reflection profile, with
    Névot–Croce roughness damping.  The result is raised to the power
    `n_mirrors` to model paired mirrors.

    Args:
    energy_eV : float
        Photon energy (eV).
    material : str
        Coating material name recognised by *xrayutilities*
        (e.g. `'Rh'`, `'Pt'`, `'Si'`).
    grazing_angle_mrad : float
        Nominal grazing incidence angle of each mirror (mrad).
    n_mirrors : int
        Number of mirrors in the KB system (typically 2).
    roughness_ang : float
        RMS surface roughness (Å).  Used in the Névot–Croce factor
        `exp(-(2 k sinθ σ)²)`.

    Returns:
    float
        Total reflectivity in [0, 1].

    Note:
    The Fresnel reflectivity for s-polarisation is:

    $$
    r_s = \\frac{\\sin\\theta - \\sqrt{n^2 - \\cos^2\\theta}}
               {\\sin\\theta + \\sqrt{n^2 - \\cos^2\\theta}}
    $$
    where $n = 1 - \\delta + i\\beta$ and $\\theta$ is the
    grazing angle.  The Névot–Croce roughness correction is applied as:

    $$
    R_{\\text{rough}} = R_{\\text{smooth}} \\cdot
        \\exp\\!\\left[-(2 k \\sin\\theta\\,\\sigma)^2\\right]
    $$
    The two KB mirrors are assumed identical, giving
    $R_{\\text{total}} = R_{\\text{single}}^{n_{\\text{mirrors}}}$.
"""
    import xrayutilities as xu

    theta = grazing_angle_mrad * 1e-3  # rad
    lam_ang = en2lam(energy_eV)  # Å
    k = 2.0 * np.pi / lam_ang  # Å⁻¹

    # Optical constants via xrayutilities
    # xu.materials.Rh is an Element object, which has no delta_beta method.
    # Wrap it in Amorphous (which accepts element name + density) to get δ, β.
    try:
        mat = getattr(xu.materials, material, None)
        if mat is None or not hasattr(mat, "delta_beta"):
            # Element objects don't have delta_beta — create an Amorphous proxy
            elem = getattr(xu.materials.elements, material, None)
            if elem is None:
                return 1.0  # unknown material: assume perfect (no correction)
            density = getattr(elem, "density", None)
            if not density:
                return 1.0
            mat = xu.materials.Amorphous(material, density)
        delta, beta = mat.delta_beta(energy_eV)
    except Exception:
        # If xrayutilities fails for any reason, skip the correction rather
        # than zeroing out all spectral weights (which produces 0 spots).
        return 1.0

    n_sq = (1.0 - delta + 1j * beta) ** 2  # n²

    sin_th = np.sin(theta)
    cos_th = np.cos(theta)

    # sqrt(n² - cos²θ)  — complex
    sq = np.sqrt(n_sq - cos_th**2 + 0j)

    denom = sin_th + sq
    if abs(denom) < 1e-15:
        return 0.0

    r_s = (sin_th - sq) / denom
    R_smooth = float(abs(r_s) ** 2)

    # Névot–Croce roughness factor
    nc = np.exp(-((2.0 * k * sin_th * roughness_ang) ** 2))
    R_single = R_smooth * float(nc)
    R_single = max(0.0, min(1.0, R_single))

    return R_single**n_mirrors


# ─────────────────────────────────────────────────────────────────────────────
# LORENTZ-POLARISATION
# ─────────────────────────────────────────────────────────────────────────────


def lorentz_pol(tth_deg):
    r = np.radians(tth_deg)
    s, c = np.sin(r / 2), np.cos(r / 2)
    if abs(s) < 1e-8 or abs(c) < 1e-8:
        return 0.0
    return abs((1 + np.cos(r) ** 2) / (2 * s**2 * c))


def _lorentz_pol_vec(tth_deg):
    """Vectorized Lorentz–polarisation factor for an array of 2θ values (degrees)."""
    r = np.radians(np.asarray(tth_deg, dtype=float))
    s = np.sin(r / 2)
    c = np.cos(r / 2)
    valid = (np.abs(s) >= 1e-8) & (np.abs(c) >= 1e-8)
    out = np.zeros(r.shape)
    out[valid] = np.abs((1.0 + np.cos(r[valid]) ** 2) / (2.0 * s[valid] ** 2 * c[valid]))
    return out



# ─────────────────────────────────────────────────────────────────────────────
# STRAIN BROADENING
# ─────────────────────────────────────────────────────────────────────────────

# Voigt index → symmetric (i,j) tensor index
# Order: [ε₁₁, ε₂₂, ε₃₃, ε₂₃, ε₁₃, ε₁₂]
_VOIGT_IJ = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]


def strain_spot_jacobian(spots, crystal, U, camera, eps_step=1e-5):
    """
    Compute the 2×6 Jacobian ∂(xcam, ycam)/∂ε_voigt for each Laue spot.

    For each reflection (hkl) in `spots`, a small strain increment is applied
    to each of the six independent tensor strain components (Voigt order:
    ε₁₁, ε₂₂, ε₃₃, ε₂₃, ε₁₃, ε₁₂) and the resulting shift in pixel
    coordinates is measured via finite differences.

    **Physics note** — in white-beam Laue the incident wavelength adjusts
    freely to satisfy the Laue condition.  A pure hydrostatic strain rescales
    |G| but not its direction, so it does **not** shift the spot.  Only the
    *deviatoric* part of the strain (which rotates G) moves spots.  The
    Jacobian captures this automatically.

    Args:
    spots : list of dict
        Output of :func:`simulate_laue` (must contain `'hkl'` and `'pix'`).
    crystal : xu.materials.Crystal
        Same crystal used to produce `spots`.
    U : ndarray, shape (3, 3)
        Orientation matrix used to produce `spots` (LT frame).
        Pass the **rotation-only** U from :func:`decompose_matstarlab`,
        not the full deformation gradient F, so that strain perturbations
        are applied on top of a clean orientation.
    camera : Camera
        Same camera used to produce `spots`.
    eps_step : float, optional
        Finite-difference step size for each strain component (dimensionless).
        Default 1e-5 is safe for typical elastic strains ~10⁻³.

    Returns:
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

    $$
    \\Sigma_{\\text{pix}} = J \\, \\Sigma_{\\varepsilon} \\, J^{\\top}
    $$
    where J (2×6) is the strain Jacobian from :func:`strain_spot_jacobian`.
    The broadening is reported as the RMS pixel spread (square root of the
    largest eigenvalue of Σ_pix) and the full 2×2 pixel covariance, from
    which the ellipse axes and orientation can be extracted.

    Args:
    spots : list of dict
        Output of :func:`simulate_laue`.
    crystal : xu.materials.Crystal
        Same crystal used to produce `spots`.
    U : ndarray, shape (3, 3)
        Rotation-only orientation matrix (from :func:`decompose_matstarlab`).
    camera : Camera
        Same camera used to produce `spots`.
    eps_voigt_std : float or array-like, shape (6,), optional
        Standard deviation of each Voigt strain component
        [σ₁₁, σ₂₂, σ₃₃, σ₂₃, σ₁₃, σ₁₂].
        If scalar, the same std is applied to all six components.
        Ignored when `eps_cov` is provided.
        Default: 1e-3 (typical elastic strain).
    eps_cov : array-like, shape (6, 6), optional
        Full 6×6 covariance matrix of the strain distribution (Voigt basis).
        When provided, overrides `eps_voigt_std`.
    eps_step : float, optional
        Finite-difference step for the Jacobian computation.

    Returns:
    list of dict
        Copy of `spots` with three new keys added to each entry:

        `'sigma_strain_pix'` : float
            RMS broadening (pixels) = √(largest eigenvalue of Σ_pix).
            This is the semi-major axis of the broadening ellipse.
        `'sigma_strain_minor'` : float
            Semi-minor axis of the broadening ellipse (pixels).
        `'cov_pix'` : ndarray, shape (2, 2)
            Full pixel-space covariance matrix.  Its eigenvectors give the
            orientations of the broadening ellipse on the detector.

    Note:
    * The broadening is relative to the spot centre; it does **not** include
      the intrinsic diffraction spot width (set by `sigma_pix` in
      :meth:`Camera.render`).
    * To render spots with strain broadening included, pass
      `sigma_pix=sigma_strain_pix` to :meth:`Camera.render`, or add it in
      quadrature: `sigma_total = sqrt(sigma_instrument² + sigma_strain²)`.
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

    $$
    \\sigma_{\\text{meas},k}^2 = \\sigma_{\\text{inst}}^2
                               + \\lambda_{\\max}(J_k \\, \\Sigma_\\varepsilon \\, J_k^\\top)
    $$
    Two modes are supported:

    **isotropic** — single scalar σ_ε (Σ_ε = σ²I)
        The equation becomes linear in σ²:

        $$
        y_k = \\sigma^2 \\cdot \\lambda_{\\max}(J_k J_k^\\top)
        $$
        Solved by weighted least squares over all spots.

    **diagonal** — six independent variances σᵢ² (Σ_ε = diag(σ₁²,…,σ₆²))
        λ_max is non-linear in σᵢ², so the **trace** is used as a linear proxy:

        $$
        y_k \\approx \\sum_i \\sigma_i^2 \\, \\|J_{k,i}\\|^2
        $$
        (exact when the two eigenvalues of J Σ Jᵀ are equal; conservative
        otherwise, since trace ≥ λ_max).
        Solved by non-negative least squares (:func:`scipy.optimize.nnls`).

    Args:
    jacobians : dict  {(h,k,l): ndarray (2, 6)}
        Output of :func:`strain_spot_jacobian`.
    sigma_meas_pix : dict  {(h,k,l): float}
        Measured semi-major spot width (pixels, 1σ) for each indexed
        reflection, obtained by fitting a 2-D Gaussian to the experimental
        detector image.  Only hkl keys present in both `jacobians` and
        this dict are used.
    sigma_instrument : float or dict or result dict from :func:`estimate_instrument_broadening`
        Instrument broadening (pixels, 1σ), subtracted in quadrature per spot.
        Three forms are accepted:

        * **float** — the same value is applied to all spots.
        * **dict {(h,k,l): float}** — per-spot instrumental width (e.g. the
          `'sigma_per_spot'` entry from :func:`estimate_instrument_broadening`).
        * **result dict** — the full dict returned by
          :func:`estimate_instrument_broadening`; the `'sigma_per_spot'` field
          is extracted automatically.  For spots not covered by the calibrant,
          the scalar `'sigma_instrument'` fallback is used.
    mode : {'isotropic', 'diagonal'}
        Fitting model.  `'isotropic'` fits a single σ_ε;
        `'diagonal'` fits all six Voigt variances independently.
    min_sensitivity : float, optional
        Minimum RMS Jacobian magnitude (px per unit strain) required for a
        spot to be included.  Spots with `sqrt(mean(J**2)) < min_sensitivity`
        are insensitive to strain and are excluded.  Default: 0.1.

    Returns:
    result : dict with keys:

        `'sigma_eps'` : float
            Isotropic RMS strain (scalar).  For `mode='isotropic'` this is
            the direct fit result; for `mode='diagonal'` it is
            `sqrt(mean(eps_voigt_std**2))`.
        `'eps_voigt_std'` : ndarray, shape (6,)
            Per-component standard deviation
            [σ₁₁, σ₂₂, σ₃₃, σ₂₃, σ₁₃, σ₁₂].
            For `mode='isotropic'` all six entries are equal.
        `'Sigma_eps'` : ndarray, shape (6, 6)
            Fitted covariance matrix (diagonal for both modes).
            Pass directly to :func:`strain_broadening` as `eps_cov`.
        `'residuals_pix'` : ndarray
            Per-spot residual: measured − predicted broadening (pixels).
        `'hkl_used'` : list of tuples
            hkl indices of the spots that entered the fit.
        `'n_spots'` : int
            Number of spots used.
        `'mode'` : str
            The mode that was used.

    Raises:
    ValueError
        If fewer than 2 spots survive the sensitivity cut.

    Note:
    * Feed the returned `'Sigma_eps'` to :func:`strain_broadening` to verify
      the fit: the predicted `sigma_strain_pix` values should match
      `sigma_meas_pix - sigma_instrument` (in quadrature).
    * For `mode='diagonal'`, the system is underdetermined if fewer than
      6 spots are available.  In that case, prefer `mode='isotropic'`.
    * Negative excess variance (`sigma_meas < sigma_instrument`) is clamped
      to zero rather than treated as an error.

    Example:
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


def measure_spot_widths(
    spots,
    meas,
    window: int = 9,
):
    """
    Fit a 2-D Gaussian to each simulated spot position in a detector image
    and return the measured 1σ widths as a dict compatible with
    :func:`estimate_instrument_broadening`.

    Args:
    spots : list of dicts
        Spot list from any `simulate_laue*` function.
        Required keys: `'hkl'`, `'pix'` (pixel position as (xcam, ycam)).
    meas : ndarray, shape (Nv, Nh)
        Detector image (raw counts).
    window : int
        Half-width of the fitting sub-window in pixels.  A `2*window × 2*window`
        patch centred on each spot is extracted for the Gaussian fit.
        Default: 9.

    Returns:
    sigma_meas_pix : dict  {(h, k, l): float}
        Measured 1σ spot width in pixels (mean of σ_x and σ_y from the
        2-D Gaussian fit) for each indexed reflection.
        Spots whose fit fails or whose window falls outside the detector are
        silently skipped.
        Pass directly to :func:`estimate_instrument_broadening` as
        `sigma_meas_pix`.
"""
    from .segmentation import auto_init_gaussian_mixture_global, fit_gaussian_mixture_2d

    meas = np.asarray(meas)
    nv, nh = meas.shape
    sigma_meas_pix = {}

    for spot in spots:
        hkl = spot.get("hkl")
        pix = spot.get("pix")
        if hkl is None or pix is None:
            continue

        key = tuple(int(x) for x in hkl)
        xcen, ycen = round(pix[0]), round(pix[1])

        # Skip spots whose window would reach outside the detector
        if (ycen - window < 0 or ycen + window >= nv
                or xcen - window < 0 or xcen + window >= nh):
            continue

        pk = meas[ycen - window: ycen + window,
                  xcen - window: xcen + window]

        try:
            init = auto_init_gaussian_mixture_global(pk, n_components=1)
            popt = fit_gaussian_mixture_2d(pk, 1, init)[0]
            sigma_x, sigma_y = popt[3], popt[4]
            sigma_meas_pix[key] = float(np.mean([sigma_x, sigma_y]))
        except Exception:
            continue

    return sigma_meas_pix


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

    The returned scalar `'sigma_instrument'` (or the callable model) can be
    passed directly to :func:`fit_strain_distribution` to subtract the
    instrumental baseline before fitting strain.

    Args:
    spots : list of dicts
        Simulated spots for the **calibrant** crystal, from :func:`simulate_laue`.
        Must contain `hkl`, `tth`, and `chi` keys.
    sigma_meas_pix : dict  {(h, k, l): float}
        Measured semi-major spot width (pixels, 1σ) for each indexed reflection
        of the calibrant, obtained by fitting a 2-D Gaussian to the detector image.
    mode : {'constant', 'linear_tth', 'quadratic_tth'}
        Model for the angular dependence of instrumental broadening:

        `'constant'`
            Single value for all spots: median of the measured widths.
            Robust to outliers; use when the detector is well-focused.

        `'linear_tth'`
            σ_inst(2θ) = a + b · 2θ (degrees).
            Captures defocus that increases with scattering angle.

        `'quadratic_tth'`
            σ_inst(2θ) = a + b · 2θ + c · 2θ².
            Fits a second-order trend; requires ≥ 5 spots.

    tth_range : (float, float), optional
        Only include spots with 2θ in this range (degrees).
        Useful to exclude regions where the model fit is unreliable.
    chi_range : (float, float), optional
        Only include spots with χ in this range (degrees).
    min_spots : int, optional
        Minimum number of matching spots required. Default: 3.

    Returns:
    result : dict with keys:

        `'sigma_instrument'` : float
            Scalar estimate of σ_instrument (pixels).
            For `'constant'` mode: the median.
            For parametric modes: the median of the fitted values at each spot.
        `'model'` : callable  f(tth_deg) → float
            Model function for σ_instrument as a function of 2θ (degrees).
            Always returned; for `'constant'` mode it returns the same scalar
            for any input.
        `'params'` : ndarray
            Fitted polynomial coefficients [a] or [a, b] or [a, b, c].
        `'sigma_per_spot'` : dict  {(h,k,l): float}
            Model-predicted σ_instrument for each spot used in the fit.
        `'residuals_pix'` : ndarray
            Measured − fitted σ_instrument per spot (pixels).
        `'rmse_pix'` : float
            Root-mean-square of residuals (pixels).
        `'hkl_used'` : list of tuples
            hkl of the spots that entered the fit.
        `'n_spots'` : int
            Number of spots used.
        `'mode'` : str
            The mode that was used.

    Raises:
    ValueError
        If fewer than `min_spots` spots match (wrong hkl, out of range filters,
        or too few calibrant reflections on the detector).

    Note:
    * Robust to a few outlier spots in `'constant'` mode (median used).
    * For parametric modes the fit is an ordinary least-squares polynomial;
      strongly deviant spots can be manually excluded via `tth_range`.
    * The model is evaluated at 2θ only.  If broadening has a strong χ
      dependence (e.g. astigmatism), measure and apply it separately.
    * A good calibrant should have sharp, well-separated spots with low
      absorption and no texture — Si (powder or single crystal cut along
      low-index direction), CeO₂, LaB₆, or Al₂O₃ are common choices.

    Example:
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
        spot_info[hkl] = (float(s["tth"]), float(s["chi"]))

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
# SPECTRUM HELPER
# ─────────────────────────────────────────────────────────────────────────────


def _make_spectrum_fn(source, source_kwargs=None, kb_params=None):
    """
    Build and return a scalar spectrum-weight function  `sw(E_eV) -> float`.

    This factory is called once per simulation run.  For `'shadow4'` and
    `'tabulated'` sources it pre-computes a linear interpolator so that
    per-spot evaluation is O(1) regardless of the source complexity.

    Args:
    source : str
        One of:

        * `'bending_magnet'` — Kim 1989 analytical BM spectrum (fast).
        * `'wiggler'` — same formula scaled by number of poles (fast).
        * `'undulator'` — Gaussian harmonic model (fast).
        * `'flat'` — uniform weight of 1.0 (fast, for testing).
        * `'shadow4'` — full optical-chain Monte Carlo via Shadow4 + xraylib.
          Traces SBM32 → M1 (Ir) → M2 (Ir) → KB (Rh) and returns absolute
          flux [ph/s/eV] at the sample.  **KB reflectivity is included** in
          the output; pass `kb_params=None` (or it will be silently ignored).
          Slow — runs once per simulation call.  Accepted `source_kwargs`:

            `nrays`         — number of Monte Carlo rays (default 1 000 000).
            `n_energy_bins` — histogram bins for the output (default 250).

        * `'tabulated'` — arbitrary pre-computed spectrum supplied as arrays.
          **KB reflectivity must already be included** in the flux values.
          Required `source_kwargs` keys:

            `energy_eV` — 1-D array of photon energies (eV).
            `flux`      — 1-D array of spectral weights (any units; only the
                           relative shape matters).

          Typical use: pass the `flux_at_sample` array returned by
          :func:`~nrxrdct.laue.beamline.simulate_bm32_pink_beam_spectrum`::

              result = simulate_bm32_pink_beam_spectrum(nrays=500_000, plot=False)
              spots  = simulate_laue_stack(
                  stack, cam,
                  source='tabulated',
                  source_kwargs={
                      'energy_eV': result['energy_eV'],
                      'flux':      result['flux_at_sample'],
                  },
                  kb_params=None,   # already included
              )

    source_kwargs : dict or None
        Extra arguments for the chosen source (see above).
    kb_params : dict or None
        Forwarded to :func:`kb_reflectivity` for **analytical sources only**.
        Silently ignored for `'shadow4'` and `'tabulated'` because mirror
        reflectivity is already embedded in those spectra.

    Returns:
    fn : callable `(E_eV: float) -> float`
        Spectrum weight at photon energy *E_eV*.  Values are normalised so that
        `max(fn) ≈ 1`; only relative weights matter for spot intensities.
"""
    source_kwargs = source_kwargs or {}

    # ── Interpolated sources (pre-compute once) ───────────────────────────────
    if source == "shadow4":
        try:
            from .beamline import simulate_bm32_pink_beam_spectrum
        except ImportError as exc:
            raise ImportError(
                "source='shadow4' requires the shadow4, xraylib, srxraylib, and syned "
                "packages.\nInstall with:  pip install shadow4 xraylib srxraylib"
            ) from exc
        nrays = int(source_kwargs.get("nrays", 1_000_000))
        n_bins = int(source_kwargs.get("n_energy_bins", 250))
        result = simulate_bm32_pink_beam_spectrum(
            nrays=nrays, n_energy_bins=n_bins, plot=False, save_fig=None
        )
        e_arr = result["energy_eV"]
        f_arr = result["flux_at_sample"]
        # KB reflectivity already included — ignore kb_params

    elif source == "tabulated":
        if "energy_eV" not in source_kwargs or "flux" not in source_kwargs:
            raise ValueError(
                "source='tabulated' requires source_kwargs to contain "
                "'energy_eV' and 'flux' arrays."
            )
        e_arr = np.asarray(source_kwargs["energy_eV"], dtype=float)
        f_arr = np.asarray(source_kwargs["flux"], dtype=float)
        # KB reflectivity assumed already included — ignore kb_params

    # ── KB reflectivity lookup table (computed once per unique parameter set) ─
    if kb_params is not None:
        _kb_key = tuple(sorted(kb_params.items()))
        if _kb_key not in _KB_CACHE:
            _E_kb = np.linspace(1_000.0, 120_000.0, 400)
            _R_kb = np.array([kb_reflectivity(e, **kb_params) for e in _E_kb])
            _KB_CACHE[_kb_key] = (_E_kb, _R_kb)
        _E_kb, _R_kb = _KB_CACHE[_kb_key]

        def _kb_scale(E_arr):
            return np.interp(E_arr, _E_kb, _R_kb)
    else:
        _kb_scale = None

    if source in ("shadow4", "tabulated"):
        from scipy.interpolate import interp1d
        f_max = f_arr.max()
        if f_max <= 0:
            raise ValueError("Spectrum flux array is all-zero or negative.")
        _interp = interp1d(
            e_arr, f_arr / f_max,
            kind="linear", bounds_error=False, fill_value=0.0,
        )

        def _sw(E):
            E_arr = np.atleast_1d(np.asarray(E, dtype=float))
            out = np.asarray(_interp(E_arr), dtype=float)
            if _kb_scale is not None:
                out = out * _kb_scale(E_arr)
            return float(out[0]) if np.ndim(E) == 0 else out

        return _sw

    # ── Analytical sources — array-compatible ────────────────────────────────
    _Ec    = source_kwargs.get("Ec_eV", E_CRIT_eV)
    _N_bm  = source_kwargs.get("N_poles", 40) if source == "wiggler" else 1
    _E1    = source_kwargs.get("E1_eV",   E_FUNDAMENTAL_eV)
    _nh    = source_kwargs.get("n_harm",   N_HARMONICS)
    _sig   = source_kwargs.get("sig_rel",  HARMONIC_WIDTH)

    def _sw(E):
        E_arr = np.atleast_1d(np.asarray(E, dtype=float))
        if source in ("bending_magnet", "wiggler"):
            y = E_arr / _Ec
            out = np.zeros_like(y)
            m = y > 1e-7
            if m.any():
                out[m] = 2.0 * _N_bm * y[m] ** 2 * kv(2 / 3, y[m] / 2) ** 2
        elif source == "undulator":
            out = np.zeros_like(E_arr)
            for _n in range(1, 2 * _nh, 2):
                En = _n * _E1
                out += (1.0 / _n) * np.exp(-0.5 * ((E_arr - En) / (En * _sig)) ** 2)
        elif source == "flat":
            out = np.ones_like(E_arr)
        else:
            raise ValueError(
                f"Unknown source {source!r}.  "
                "Choose from: 'bending_magnet', 'wiggler', 'undulator', "
                "'flat', 'shadow4', 'tabulated'."
            )
        if _kb_scale is not None:
            out = out * _kb_scale(E_arr)
        return float(out[0]) if np.ndim(E) == 0 else out

    return _sw


# ─────────────────────────────────────────────────────────────────────────────
# CORE LAUE SIMULATION
# ─────────────────────────────────────────────────────────────────────────────


def precompute_allowed_hkl(
    crystal,
    E_max_eV: float = E_MAX_eV,
    E_ref_eV: float | None = None,
    f2_thresh: float = F2_THRESHOLD,
) -> frozenset:
    """
    Return the frozenset of (h, k, l) tuples whose structure factor exceeds
    `f2_thresh` at a single reference energy.

    Enumerates candidate (h, k, l) reflections using a physically correct
    sphere criterion: only reflections with |G_hkl| ≤ G_max are considered,
    where G_max = 4π · E_max_eV / 12398.4 Å⁻¹ (back-reflection limit).
    This correctly handles anisotropic unit cells (e.g. hexagonal with large
    c/a) by computing per-axis index limits from the reciprocal lattice
    parameters.

    The result can then be passed as `allowed_hkl` to :func:`simulate_laue`
    or :func:`simulate_laue_stack` to skip all per-spot structure factor calls
    during fitting while still respecting systematic absences.

    For a :class:`~nrxrdct.laue.layers.LayeredCrystal`, the union of allowed
    sets across all unique constituent crystals is returned.

    Args:
    crystal   : Crystal or LayeredCrystal
    E_max_eV  : float   Maximum photon energy (eV) that controls the
                        enumeration cutoff via G_max = 4π·E_max_eV/12398.4.
                        Default: `E_MAX_eV`.
    E_ref_eV  : float   Reference photon energy (eV) for structure factor
                        evaluation.  Defaults to the midpoint of the default
                        energy window `(E_MIN_eV + E_MAX_eV) / 2`.
                        Must be a valid energy within the Henke table range.
    f2_thresh : float   Same threshold used in the simulation.

    Returns:
    frozenset of (int, int, int)
        Immutable, hashable set; safe to share across threads.
"""
    from .layers import LayeredCrystal

    # Resolve default before building the cache key so that None and the
    # explicit midpoint value always hit the same cache entry.
    if E_ref_eV is None:
        E_ref_eV = (E_MIN_eV + E_MAX_eV) / 2.0

    if isinstance(crystal, LayeredCrystal):
        _lc_key = (
            tuple(sorted({l.crystal.name for l in crystal.all_layers})),
            E_max_eV, E_ref_eV, f2_thresh,
        )
        if _lc_key in _allowed_hkl_cache:
            return _allowed_hkl_cache[_lc_key]
        allowed: set = set()
        seen_names: set = set()
        for layer in crystal.all_layers:
            if layer.crystal.name not in seen_names:
                seen_names.add(layer.crystal.name)
                allowed |= precompute_allowed_hkl(
                    layer.crystal, E_max_eV=E_max_eV, E_ref_eV=E_ref_eV,
                    f2_thresh=f2_thresh
                )
        result = frozenset(allowed)
        _allowed_hkl_cache[_lc_key] = result
        return result

    _key = (crystal.name, E_max_eV, E_ref_eV, f2_thresh)
    if _key in _allowed_hkl_cache:
        return _allowed_hkl_cache[_key]

    HC_eV_ANG = 12398.4
    G_max = 4.0 * np.pi * E_max_eV / HC_eV_ANG
    G_max_sq = G_max ** 2

    # Build the reciprocal-lattice matrix B (columns = a*, b*, c* in Å⁻¹).
    # Q(h,k,l) = B @ [h,k,l] by linearity, so we can replace (2·h_lim+1)³
    # individual crystal.Q calls with three calls + one numpy matmul.
    B = np.column_stack([crystal.Q(1, 0, 0), crystal.Q(0, 1, 0), crystal.Q(0, 0, 1)])
    a_star = float(np.linalg.norm(B[:, 0]))
    b_star = float(np.linalg.norm(B[:, 1]))
    c_star = float(np.linalg.norm(B[:, 2]))
    h_lim = int(G_max / a_star) + 1
    k_lim = int(G_max / b_star) + 1
    l_lim = int(G_max / c_star) + 1

    # Generate the full bounding-box grid and filter to the sphere in one pass.
    h_vals = np.arange(-h_lim, h_lim + 1, dtype=np.int32)
    k_vals = np.arange(-k_lim, k_lim + 1, dtype=np.int32)
    l_vals = np.arange(-l_lim, l_lim + 1, dtype=np.int32)
    H, K, L = np.meshgrid(h_vals, k_vals, l_vals, indexing="ij")
    hkl_all = np.column_stack([H.ravel(), K.ravel(), L.ravel()])

    # Remove (0, 0, 0)
    hkl_all = hkl_all[np.any(hkl_all != 0, axis=1)]

    # Single matmul replaces N individual crystal.Q calls.
    G_all = (B @ hkl_all.T).T          # (N, 3)
    G_sq  = np.einsum("ij,ij->i", G_all, G_all)  # (N,) squared norms

    # Sphere filter: only keep |G| ≤ G_max.
    in_sphere = G_sq <= G_max_sq
    hkl_sphere = hkl_all[in_sphere]
    G_sphere   = G_all[in_sphere]

    F2_arr = np.abs(crystal.StructureFactorForQ(G_sphere, en0=E_ref_eV)) ** 2
    mask   = F2_arr >= f2_thresh
    result = frozenset(map(tuple, hkl_sphere[mask].tolist()))
    _allowed_hkl_cache[_key] = result
    return result


def simulate_laue(
    crystal,
    U,
    camera,
    E_min=E_MIN_eV,
    E_max=E_MAX_eV,
    source="bending_magnet",
    source_kwargs=None,
    f2_thresh=F2_THRESHOLD,
    kb_params=BM32_KB,
    sigma_h_mrad=0.0,
    sigma_v_mrad=0.0,
    sigma_beam_h_nm=0.0,
    sigma_beam_v_nm=0.0,
    n_hat_sample=None,
    geometry_only=False,
    allowed_hkl=None,
    depth_mm=0.0,
    _pixels_only=False,
):
    """
    Simulate single-crystal white-beam Laue diffraction in reflection geometry.

    For every reciprocal-lattice vector G_hkl satisfying the sphere criterion
    |G_hkl| ≤ G_max = 4π·E_max/12398.4 Å⁻¹ the function:

    1. Rotates G from the crystal frame into the lab frame via the orientation
       matrix `U`:  `G_lab = U @ G_cry`.
    2. Applies the Laue condition to find the wavelength (and photon energy) at
       which this reflection is excited::

           lambda_hkl = -4*pi * (k_i_hat . G_lab) / |G_lab|^2

       Reflections whose wavelength falls outside `[E_min, E_max]` are
       skipped.
    3. Computes the scattered-beam direction `kf_hat` and projects it onto
       the detector plane via `camera.project()`.  Reflections that miss the
       active area are discarded.
    4. Evaluates the spot intensity::

           I_raw(hkl) = |F(G, E)|^2  *  LP(2theta)  *  S(E)

       where:

       - `|F(G, E)|^2`  – kinematical structure factor squared (Cromer-Mann
         `f0` plus Henke anomalous corrections `f'`, `f''` via
         *xrayutilities*).  Reflections below `f2_thresh` are dropped.
       - `LP(2theta)`   – Lorentz-polarisation factor for an unpolarised
         beam::

               LP = (1 + cos^2(2θ)) / (2 * sin^2(θ) * cos(θ))

       - `S(E)`         – synchrotron spectral weight at energy `E`
         (bending-magnet, wiggler, or undulator model set by the module-level
         `SOURCE_TYPE`).

    5. Normalises all surviving `I_raw` values by the brightest spot so that
       `intensity` lies in `(0, 1]`.

    Args:
    crystal : Crystal-like
        An *xrayutilities*-compatible crystal object that exposes:

        - `crystal.Q(h, k, l)`  → reciprocal-lattice vector in crystal frame
          (Å⁻¹, 2π convention).
        - `crystal.StructureFactor(G_cry, en=E)`  → complex structure factor
          at energy `E` (eV).

    U : array-like, shape (3, 3)
        Orientation matrix mapping crystal-frame vectors to the LaueTools lab
        frame (beam along `+x`).  Typically obtained from
        `euler_to_U(phi1, Phi, phi2)` or an indexing result.

    camera : Camera
        Detector geometry object (see `camera.py`).  Must implement
        `camera.project(kf_hat)` which returns `(col, row)` pixel
        coordinates or `None` if the ray misses the detector.

    E_min : float, optional
        Low-energy cut-off of the white beam in eV.
        Default: `E_MIN_eV` (5 000 eV).

    E_max : float, optional
        High-energy cut-off of the white beam in eV.
        Default: `E_MAX_eV` (27 000 eV).

    source : str, optional
        Synchrotron source model.  See :func:`_make_spectrum_fn` for the full
        list (`'bending_magnet'`, `'wiggler'`, `'undulator'`, `'flat'`,
        `'shadow4'`, `'tabulated'`).  Default: `'bending_magnet'`.

    source_kwargs : dict or None, optional
        Extra arguments for the chosen source (see :func:`_make_spectrum_fn`).

    kb_params : dict or None, optional
        KB mirror reflectivity correction (see :func:`kb_reflectivity`).
        Ignored for `'shadow4'` and `'tabulated'` sources.

    f2_thresh : float, optional
        Minimum `|F|^2` threshold (arbitrary units, same scale as
        *xrayutilities* output).  Reflections below this value are treated as
        systematically absent or too weak and discarded before the LP / spectrum
        weighting step.
        Default: `F2_THRESHOLD` (1e-6).

    Returns:
    list of dict
        One dictionary per spot that satisfies all selection criteria, sorted
        by **descending** `intensity`.  Each dictionary contains:

        ==================  ====================================================
        Key                 Description
        ==================  ====================================================
        `hkl`             `(h, k, l)` tuple of Miller indices.
        `E`               Photon energy at which the reflection is excited (eV).
        `lambda`          Corresponding wavelength (Å).
        `tth`             Bragg angle `2θ` (degrees), measured from the
                            forward-beam direction `+x`.
        `chi`             LaueTools χ angle (degrees):
                            `arctan2(kf_y, kf_z)`.
        `az`              Azimuthal angle (degrees):
                            `arctan2(kf_z, kf_y)`.
        `pix`             `(col, row)` pixel coordinate on the detector
                            (LaueTools convention: `xcam, ycam`).
        `F2`              `|F(G, E)|^2`, structure factor squared.
        `LP`              Lorentz-polarisation factor.
        `sw`              Synchrotron spectral weight `S(E)`.
        `I_raw`           Un-normalised intensity: `F2 * LP * sw`.
        `intensity`       `I_raw` normalised to `[0, 1]` by the
                            brightest spot in this simulation.
        ==================  ====================================================

        Returns an **empty list** if no reflection satisfies all criteria.

    Note:
    * The incident beam is fixed along `+x` in the LaueTools lab frame
      (`KI_HAT = [1, 0, 0]`).  Do not modify this without updating the
      camera geometry accordingly.
    * `intensity` is a *relative* quantity within a single call.  When
      comparing patterns from different phases or orientations use `I_raw`
      and apply an external weighting (see `simulate_mixed_phases`).
"""
    lam_lo = en2lam(E_max)
    lam_hi = en2lam(E_min)
    ki_hat = KI_HAT / np.linalg.norm(KI_HAT)
    source_kwargs = source_kwargs or {}

    _spectrum = _make_spectrum_fn(source, source_kwargs, kb_params)

    # Sphere enumeration: only reflections with |G| ≤ G_max can satisfy
    # the Laue condition within the energy window (back-reflection limit).
    HC_eV_ANG = 12398.4
    G_max = 4.0 * np.pi * E_max / HC_eV_ANG
    G_max_sq = G_max ** 2
    a_star = float(np.linalg.norm(crystal.Q(1, 0, 0)))
    b_star = float(np.linalg.norm(crystal.Q(0, 1, 0)))
    c_star = float(np.linalg.norm(crystal.Q(0, 0, 1)))
    h_lim = int(G_max / a_star) + 1
    k_lim = int(G_max / b_star) + 1
    l_lim = int(G_max / c_star) + 1

    # Build B matrix: Q(h,k,l) = B @ [h,k,l] avoids repeated crystal.Q calls.
    # Deferred to avoid the cost when the hkl_arrays cache is warm.
    def _get_B():
        return np.column_stack([crystal.Q(1, 0, 0), crystal.Q(0, 1, 0), crystal.Q(0, 0, 1)])

    spots = []

    if allowed_hkl is not None:
        # ── VECTORISED PATH (fitting / geometry-only) ─────────────────────────
        # Spectrum and LP are not needed — only pixel positions matter.
        _aid = id(allowed_hkl)
        if _aid in _hkl_arrays_cache:
            _hkl_arr, _G_cry_arr, _cached_crystal = _hkl_arrays_cache[_aid]
            if _cached_crystal is not crystal:
                # Different crystal with same frozenset object — recompute G_cry.
                _G_cry_arr = (_get_B() @ _hkl_arr.T).T
                _hkl_arrays_cache[_aid] = (_hkl_arr, _G_cry_arr, crystal)
        else:
            _B = _get_B()
            _hkl_arr   = np.array(list(allowed_hkl), dtype=np.int32).reshape(-1, 3)  # (M, 3)
            _G_cry_arr = (_B @ _hkl_arr.T).T                          # (M, 3)
            _hkl_arrays_cache[_aid] = (_hkl_arr, _G_cry_arr, crystal)

        G_lab = _G_cry_arr @ U.T                                   # (M, 3)
        kdG   = G_lab @ ki_hat                                     # (M,)
        v1    = kdG < 0
        if np.any(v1):
            G_lab = G_lab[v1];  _hkl_arr = _hkl_arr[v1];  kdG = kdG[v1]
            Gm2   = np.einsum("ij,ij->i", G_lab, G_lab)
            lam   = -4.0 * np.pi * kdG / Gm2
            v2    = (lam >= lam_lo) & (lam <= lam_hi)
            if np.any(v2):
                G_lab = G_lab[v2];  _hkl_arr = _hkl_arr[v2];  lam = lam[v2]
                E_arr = HC / lam
                km    = 2.0 * np.pi / lam
                kf_v  = ki_hat[None, :] * km[:, None] + G_lab     # (M, 3)
                kf_v /= np.linalg.norm(kf_v, axis=1, keepdims=True)
                pix_arr, on_det = camera.project_batch(kf_v, source_depth_mm=depth_mm)
                if np.any(on_det):
                    kf_v    = kf_v[on_det];  _hkl_arr = _hkl_arr[on_det]
                    lam     = lam[on_det];   E_arr    = E_arr[on_det]
                    pix_arr = pix_arr[on_det]
                    if _pixels_only:
                        return pix_arr          # (N, 2) — skip dict construction
                    cos2th  = np.clip(kf_v[:, 0], -1.0, 1.0)
                    tth_arr = np.degrees(np.arccos(cos2th))
                    chi_arr = np.degrees(np.arctan2(kf_v[:, 1], kf_v[:, 2] + 1e-17))
                    az_arr  = np.degrees(np.arctan2(kf_v[:, 2], kf_v[:, 1]))
                    for _i in range(len(lam)):
                        spots.append({
                            "hkl":             (int(_hkl_arr[_i, 0]),
                                                int(_hkl_arr[_i, 1]),
                                                int(_hkl_arr[_i, 2])),
                            "satellite_order": 0,
                            "is_superlattice": False,
                            "E":               float(E_arr[_i]),
                            "lambda":          float(lam[_i]),
                            "tth":             float(tth_arr[_i]),
                            "chi":             float(chi_arr[_i]),
                            "az":              float(az_arr[_i]),
                            "pix":             (float(pix_arr[_i, 0]),
                                                float(pix_arr[_i, 1])),
                            "F2":              1.0,
                            "LP":              1.0,
                            "sw":              1.0,
                            "I_raw":           1.0,
                        })
    else:
        # ── VECTORISED GEOMETRY + PER-SPOT STRUCTURE FACTOR ──────────────────
        # Geometry filters (sphere, kdG, lambda, detector) are applied in bulk
        # with numpy; StructureFactor / LP / spectrum are evaluated only for the
        # small number of on-detector survivors.
        _B = _get_B()
        _H2, _K2, _L2 = np.meshgrid(
            np.arange(-h_lim, h_lim + 1, dtype=np.int32),
            np.arange(-k_lim, k_lim + 1, dtype=np.int32),
            np.arange(-l_lim, l_lim + 1, dtype=np.int32),
            indexing="ij",
        )
        _hkl = np.column_stack([_H2.ravel(), _K2.ravel(), _L2.ravel()])
        _hkl = _hkl[np.any(_hkl != 0, axis=1)]
        G_cry_all = (_B @ _hkl.T).T                                    # (N, 3)

        # Sphere filter (U orthogonal → |G_lab| = |G_cry|)
        G_sq = np.einsum("ij,ij->i", G_cry_all, G_cry_all)
        v0 = G_sq <= G_max_sq
        G_cry_all = G_cry_all[v0];  _hkl = _hkl[v0]

        G_lab_all = G_cry_all @ U.T                                     # (M, 3)

        kdG = G_lab_all @ ki_hat                                        # (M,)
        v1 = kdG < 0
        G_cry_all = G_cry_all[v1];  G_lab_all = G_lab_all[v1]
        _hkl = _hkl[v1];           kdG       = kdG[v1]

        Gm2 = np.einsum("ij,ij->i", G_lab_all, G_lab_all)
        lam = -4.0 * np.pi * kdG / Gm2
        v2 = (lam >= lam_lo) & (lam <= lam_hi)
        G_cry_all = G_cry_all[v2];  G_lab_all = G_lab_all[v2]
        _hkl      = _hkl[v2];      lam       = lam[v2]

        if len(lam) > 0:
            E_arr = HC / lam
            km    = 2.0 * np.pi / lam
            kf_v  = ki_hat[None, :] * km[:, None] + G_lab_all          # (K, 3)
            kf_v /= np.linalg.norm(kf_v, axis=1, keepdims=True)
            pix_arr, on_det = camera.project_batch(kf_v, source_depth_mm=depth_mm)
            if np.any(on_det):
                G_cry_all = G_cry_all[on_det];  _hkl    = _hkl[on_det]
                lam       = lam[on_det];        E_arr   = E_arr[on_det]
                kf_v      = kf_v[on_det];       pix_arr = pix_arr[on_det]
                cos2th  = np.clip(kf_v[:, 0], -1.0, 1.0)
                tth_arr = np.degrees(np.arccos(cos2th))
                chi_arr = np.degrees(np.arctan2(kf_v[:, 1], kf_v[:, 2] + 1e-17))
                az_arr  = np.degrees(np.arctan2(kf_v[:, 2], kf_v[:, 1]))
                for _i in range(len(lam)):
                    LP = lorentz_pol(float(tth_arr[_i]))
                    if LP == 0.0:
                        continue
                    E  = float(E_arr[_i])
                    sw = _spectrum(E)
                    if sw <= 0.0:
                        continue
                    if geometry_only:
                        F2 = 1.0
                    else:
                        F  = crystal.StructureFactor(G_cry_all[_i], en=E)
                        F2 = abs(F) ** 2
                        if F2 < f2_thresh:
                            continue
                    h, k, l = int(_hkl[_i, 0]), int(_hkl[_i, 1]), int(_hkl[_i, 2])
                    spots.append({
                        "hkl":             (h, k, l),
                        "satellite_order": 0,
                        "is_superlattice": False,
                        "E":               E,
                        "lambda":          float(lam[_i]),
                        "tth":             float(tth_arr[_i]),
                        "chi":             float(chi_arr[_i]),
                        "az":              float(az_arr[_i]),
                        "pix":             (float(pix_arr[_i, 0]), float(pix_arr[_i, 1])),
                        "F2":              F2,
                        "LP":              LP,
                        "sw":              sw,
                        "I_raw":           F2 * LP * sw,
                    })

    if spots:
        imax = max(s["I_raw"] for s in spots)
        for s in spots:
            s["intensity"] = s["I_raw"] / imax

    spots = sorted(spots, key=lambda s: s["intensity"], reverse=True)
    beam_divergence_ellipses(
        spots, camera, sigma_h_mrad, sigma_v_mrad,
        sigma_beam_h_nm=sigma_beam_h_nm,
        sigma_beam_v_nm=sigma_beam_v_nm,
        n_hat_sample=n_hat_sample,
    )
    if _pixels_only:
        return np.empty((0, 2), dtype=float)
    return spots


def _flatten_if_multiblock(stack):
    """
    Transparently flatten a multi-block :class:`~nrxrdct.laue.layers.LayeredCrystal`
    (see :meth:`LayeredCrystal.add_layer`) into an equivalent single-block
    stack, so the rest of the (single-block-only) Laue-simulation pipeline
    can consume it directly.

    No-op for a plain ``Crystal``, a stack with zero or one blocks, or
    anything without a ``.blocks`` attribute.
"""
    blocks = getattr(stack, "blocks", None)
    if blocks is not None and len(blocks) > 1:
        from .layers import combine_stacks
        return combine_stacks([stack])
    return stack


def _layer_depths_mm(stack) -> dict:
    """
    Compute the centre depth of every layer in ``stack.all_layers`` (mm).

    Layers are stored deepest-first (index 0 = substrate).  Thickness is in
    Ångströms (1 Å = 1e-7 mm).  The centre depth of layer *i* is the sum of
    thicknesses of all shallower layers (indices > i) plus half the layer's
    own thickness.

    Returns:
        dict mapping ``id(layer) → depth_mm`` for each layer.
    """
    all_layers = stack.all_layers
    n = len(all_layers)
    thicknesses = [float(getattr(l, "thickness", 0.0)) for l in all_layers]
    result = {}
    for i, layer in enumerate(all_layers):
        # Layers above i (shallower): indices i+1 … n-1
        depth_top_ang    = sum(thicknesses[j] for j in range(i + 1, n))
        depth_center_ang = depth_top_ang + thicknesses[i] / 2.0
        result[id(layer)] = depth_center_ang * 1e-7  # Å → mm
    return result


def simulate_laue_stack(
    stack,
    camera,
    E_min_eV=5_000,
    E_max_eV=27_000,
    source="bending_magnet",
    source_kwargs=None,
    f2_thresh=None,
    ki_hat=None,
    max_satellites=5,
    kb_params=BM32_KB,
    structure_model="average",
    sigma_h_mrad=0.0,
    sigma_v_mrad=0.0,
    sigma_beam_h_nm=0.0,
    sigma_beam_v_nm=0.0,
    n_hat_sample=None,
    verbose=True,
    geometry_only=False,
    allowed_hkl=None,
    correct_depth=False,
):
    """
    Compute Laue spots for a `LayeredCrystal` stack projected onto `camera`.

    For each phase in the stack the reciprocal lattice is enumerated using the
    sphere criterion |G_hkl| ≤ G_max = 4π·E_max_eV/12398.4 Å⁻¹.  Every
    G_lab that satisfies the Laue condition within the energy window is kept,
    and the structure factor is evaluated on the **full stack** at that Q.
    This ensures coherent superposition of all layers and natural emergence of
    superlattice satellites.

    Args:
    stack : LayeredCrystal
        The layered structure (from layers.py).
    camera : Camera
        Detector geometry (from camera.py).
    E_min_eV, E_max_eV : float
        Energy window  (eV).
    source : str
        Synchrotron source model.  See :func:`_make_spectrum_fn` for full
        details.  Summary:

        * `'bending_magnet'` *(default)* — Kim 1989 analytical formula, fast.
        * `'wiggler'` — same formula × N poles.
        * `'undulator'` — Gaussian harmonic model.
        * `'flat'` — uniform weight, useful for testing.
        * `'shadow4'` — full Monte Carlo via Shadow4 + xraylib tracing
          SBM32 → M1 (Ir) → M2 (Ir) → KB (Rh).  Runs once per call; slow
          (~1–2 min for 1 M rays).  Pass `kb_params=None`.
        * `'tabulated'` — arbitrary pre-computed spectrum supplied via
          `source_kwargs`.  Pass `kb_params=None`.

    source_kwargs : dict, optional
        Extra arguments for the chosen source:

        * `'bending_magnet'` / `'wiggler'` : `Ec_eV`, `N_poles`
        * `'undulator'`                       : `E1_eV`, `n_harm`, `sig_rel`
        * `'shadow4'`                         : `nrays` (default 1 000 000),
          `n_energy_bins` (default 250)
        * `'tabulated'`                       : `energy_eV` (array), `flux` (array)

    kb_params : dict or None, optional
        KB mirror reflectivity correction applied by multiplying the source
        spectrum by :func:`kb_reflectivity` at each photon energy.  The dict
        is forwarded as keyword arguments.  Defaults to :data:`BM32_KB`
        (Rh-coated, 2.5 mrad, 2 mirrors, 3 Å roughness — BM32/ESRF).
        Pass `None` to disable, and **must** be `None` for `'shadow4'`
        and `'tabulated'` sources (reflectivity already included)::

            spots = simulate_laue_stack(stack, cam)               # BM32_KB default
            spots = simulate_laue_stack(stack, cam, kb_params=None)  # no correction

        Keys: `material`, `grazing_angle_mrad`, `n_mirrors`,
        `roughness_ang`.
    f2_thresh : float
        Minimum |F_stack|²  to retain a spot (absolute, in e.u.²).
        Because the stack coherently sums many unit cells the absolute
        value scales roughly as (N_cells × N_rep)² at Bragg peaks.
        A value of `None` uses an auto-scaled threshold:
            f2_thresh = (max single-cell |F|)² × 0.001
        which keeps spots down to 0.1 % of the strongest unit-cell peak.
        For a manual value, typical starting point: ~10–1000.
    ki_hat : array-like (3,), optional
        Incident beam direction in the LaueTools LT frame (x // beam).
        Default: [1, 0, 0].
    structure_model : {'coherent', 'average'}, optional
        Controls how the MQW repeating unit structure factor is evaluated
        and which crystals are used to enumerate candidate G vectors.

        * `'coherent'` — full layer-by-layer coherent sum: each layer
          contributes `F_uc_i × N_eff_i × exp(i Qₙ z_rel_i)` within the
          period, multiplied by `S_rep` over `N_rep` periods.  G vectors
          are enumerated from **every** unique crystal in the stack, so
          slightly displaced GaN and InGaN Bragg peaks appear separately.
          Reproduces superlattice satellites and thickness fringes at their
          physically correct relative intensities.
        * `'average'` *(default)* — treats the MQW period as a single
          effective material.  G vectors are enumerated **only from the
          buffer layers** (or the first MQW layer if no buffers exist), so
          only one set of Bragg positions is produced — matching what is seen
          in a monochromatic scan.  The structure factor per period is the
          composition-weighted sum `Σ F_uc_i × N_eff_i` (no intra-period
          `exp(i Qₙ z_rel)` phases), then multiplied by `S_rep`.
          Satellite positions and `N_rep`-dependent peak widths are
          identical to the coherent model; intensities reflect the average
          composition rather than the layer-ordering interference.

        Note: strained d-spacings (from :meth:`add_pseudomorphic_layer`)
        enter through `N_eff = thickness / d_strained` in both modes.
        The unit-cell structure factor amplitude `F_uc` uses the bulk
        crystal in both cases.
    sigma_h_mrad : float, optional
        Horizontal beam divergence 1σ (mrad).  When non-zero,
        :func:`beam_divergence_ellipses` is called automatically and every
        spot receives broadening keys (`cov_px`, `sigma_major_px`, …).
        Typical BM32/ESRF value: 2–3 mrad.  Default: 0 (no broadening).
    sigma_v_mrad : float, optional
        Vertical beam divergence 1σ (mrad).
        Typical BM32/ESRF value: 0.2–0.5 mrad.  Default: 0.
    verbose : bool

    Returns:
    spots : list of dicts
        Same format as `simulate_laue()` in laue_white_synchrotron.py,
        plus extra keys:
          `'phase_label'`  – which phase's hkl triggered this candidate
          `'F2_stack'`     – |F_stack|² (full coherent stack)

    Note:
    For a stack with many layers / large N_cells the absolute values of
    `F2_stack` can be large (~N² × single-cell value at Bragg peaks).
    The returned `intensity` key is always normalised 0–1 within the
    returned list.

    **Performance**
    Each spot requires one `stack.structure_factor()` call, which itself
    calls `crystal.StructureFactor()` once per layer.  For a 2-layer stack
    with ~4000 candidate spots, total time is ~2 s on a modern CPU.

    Note:
    If `stack` has several independently-repeated blocks (see
    :meth:`~nrxrdct.laue.layers.LayeredCrystal.add_layer`), it is
    transparently flattened via
    :func:`~nrxrdct.laue.layers.combine_stacks` before simulation --
    you can pass a multi-block stack directly, no manual flattening needed.
"""
    stack = _flatten_if_multiblock(stack)
    stack._update_offsets()
    source_kwargs = source_kwargs or {}
    ki = np.asarray(ki_hat if ki_hat is not None else KI_HAT, dtype=float)
    ki /= np.linalg.norm(ki)

    lam_lo = en2lam(E_max_eV)
    lam_hi = en2lam(E_min_eV)

    spectrum = _make_spectrum_fn(source, source_kwargs, kb_params)

    # Auto-scale f2_thresh if not provided
    if f2_thresh is None:
        f2_thresh = 0.0  # will be set after first structure factor call

    # ── Fringe / satellite wavevectors ───────────────────────────────────────
    # Thickness fringes and superlattice satellites both sit at
    #     G_hkl + m * q_fringe * n̂    (m = ±1, ±2, ...)
    # where q_fringe = 2π / t  and  t  is a relevant thickness.
    #
    # Two contributions:
    #   1. Each individual finite layer produces fringes at  q = 2π / t_layer.
    #      (This is the dominant effect for n_rep = 1, e.g. a single InGaN QW.)
    #   2. The full bilayer period Λ produces superlattice satellites at
    #      q = 2π / Λ  when n_rep > 1.
    #
    # Layers thicker than MAX_FRINGE_THICK_ANG are skipped: their fringe
    # spacing 2π/t is so small that the corresponding λ falls far outside any
    # realistic white-beam window and they never pass the Laue-condition check.
    MAX_FRINGE_THICK_ANG = 20000.0  # 2 µm — tune if needed

    fringe_q_vecs = []  # list of (q_vector_3d, description)
    sat_orders = [m for m in range(-max_satellites, max_satellites + 1) if m != 0]

    if max_satellites > 0:
        # Per-layer thickness fringes
        seen_t = set()
        for lyr in stack.layers:
            t = lyr.thickness
            if t < 1e-6 or t > MAX_FRINGE_THICK_ANG:
                continue
            t_key = round(t, 2)
            if t_key in seen_t:
                continue
            seen_t.add(t_key)
            q_vec = (2.0 * np.pi / t) * stack.n_hat
            fringe_q_vecs.append((q_vec, f"layer '{lyr.label}' (t={t/10:.1f} nm)"))

        # Superlattice period (only if n_rep > 1 and Λ ≠ any single-layer t)
        Lambda = stack.bilayer_thickness
        if stack.n_rep > 1 and Lambda > 1e-6:
            t_key = round(Lambda, 2)
            if t_key not in seen_t:
                q_vec = (2.0 * np.pi / Lambda) * stack.n_hat
                fringe_q_vecs.append((q_vec, f"bilayer Λ={Lambda/10:.1f} nm"))

    if verbose and fringe_q_vecs:
        print("  Fringe / satellite periods to probe:")
        for qv, desc in fringe_q_vecs:
            t_nm = 2.0 * np.pi / np.linalg.norm(qv) / 10.0
            print(
                f"    {desc}  →  2π/t = {np.linalg.norm(qv):.4f} Å⁻¹  (t = {t_nm:.2f} nm)"
            )

    # Deduplicated pixel set: avoid appending two spots within 0.5 px of each other.
    seen_pix = set()  # set of (round(xcam), round(ycam))

    # Per-layer allowed hkl set — reassigned in the outer layer loop so that
    # _try_append always sees the set that matches the crystal being enumerated.
    _layer_allowed_hkl = None

    # Depth correction: per-layer centre depth (mm), updated as the outer loop
    # advances through the enumeration pool.
    _depth_mm_state = [0.0]
    if correct_depth:
        _depths_mm = _layer_depths_mm(stack)
        # Convert perpendicular depth to depth along the incident beam.
        # depth_along_beam = depth_perp / |cos θ|  where θ is the angle
        # between ki and the surface normal (stack.n_hat).
        _n_hat = np.asarray(stack.n_hat, dtype=float)
        _n_hat = _n_hat / np.linalg.norm(_n_hat)
        _cos_inc = abs(float(np.dot(ki, _n_hat)))
        if _cos_inc > 1e-6:
            _depths_mm = {k: v / _cos_inc for k, v in _depths_mm.items()}

    def _try_append(G_vec, hkl, sat_order, phase_label):
        """Evaluate the Laue condition + camera + F² for G_vec and append if valid."""
        nonlocal f2_thresh
        Gm2 = float(np.dot(G_vec, G_vec))
        if Gm2 < 1e-20:
            return 0
        kdG = float(np.dot(ki, G_vec))
        if kdG >= 0:
            return 0
        lam = -4.0 * np.pi * kdG / Gm2
        if not (lam_lo <= lam <= lam_hi):
            return 0
        E = lam2en(lam)
        km = 2.0 * np.pi / lam
        kf_vec = ki * km + G_vec
        kf_hat = kf_vec / np.linalg.norm(kf_vec)
        pix = camera.project(kf_hat, source_depth_mm=_depth_mm_state[0])
        if pix is None:
            return 0
        # Pixel-level deduplication
        pix_key = (round(pix[0]), round(pix[1]))
        if pix_key in seen_pix:
            return 0
        tth = np.degrees(np.arccos(np.clip(kf_hat[0], -1.0, 1.0)))
        chi = np.degrees(np.arctan2(kf_hat[1], kf_hat[2] + 1e-17))
        az = np.degrees(np.arctan2(kf_hat[2], kf_hat[1]))
        LP = lorentz_pol(tth)
        if LP == 0.0:
            return 0
        sw = spectrum(E)
        if sw <= 0.0:
            return 0
        if _layer_allowed_hkl is not None:
            if hkl not in _layer_allowed_hkl:
                return 0
            F2 = 1.0
        elif geometry_only:
            F2 = 1.0
        else:
            if structure_model == "average":
                F_stack = stack.average_structure_factor(G_vec, energy_eV=E, kf_hat=kf_hat)
            else:
                F_stack = stack.structure_factor(G_vec, energy_eV=E, kf_hat=kf_hat)
            F2 = abs(F_stack) ** 2
            if f2_thresh == 0.0:
                f2_thresh = max(1.0, F2 * 1e-3)
            effective_thresh = f2_thresh * 1e-4 if sat_order != 0 else f2_thresh
            if F2 < effective_thresh:
                return 0
        seen_pix.add(pix_key)
        spots.append(
            {
                "phase_label": phase_label,
                "hkl": hkl,
                "satellite_order": sat_order,
                "is_superlattice": sat_order != 0,
                "G_lab": G_vec.copy(),
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
            }
        )
        return 1

    spots = []

    # In average mode enumerate G vectors from buffer layers plus the first MQW
    # layer.  Adding the first MQW layer ensures that film reflections are always
    # found even when the only buffer layer is a dissimilar substrate (e.g.
    # sapphire): sapphire G vectors are at completely different positions from
    # GaN/InGaN G vectors, so enumerating from the substrate alone would yield
    # zero film spots.  The crystal+orientation deduplication below then
    # removes any duplicate if the GaN buffer and first MQW layer share the
    # same species and U matrix.  InGaN (second MQW layer) is intentionally
    # excluded so only one set of Bragg positions is produced — matching what
    # is seen in a monochromatic scan.
    if structure_model == "average":
        _enum_pool = stack.buffer_layers + stack.layers[:1]
    else:
        _enum_pool = stack.all_layers

    # Deduplicate: if two layers share the exact same crystal AND orientation,
    # enumerate once — the stack F already includes both contributions.
    seen_combos = []  # list of (crystal.name, U_rounded_tuple)

    # Buffer layers (substrate, thick intermediate layers) are non-repeating and
    # have no finite-thickness interference.  Satellites computed as
    # G_substrate + m·q_fringe_film are physically meaningless and are skipped.
    _buffer_set = set(id(l) for l in stack.buffer_layers)

    for layer in _enum_pool:
        crystal = layer.crystal
        U = layer.U
        label = layer.label
        # Update the closure variables so _try_append uses this layer's
        # allowed HKL set and (when requested) its centre depth.
        _layer_allowed_hkl = allowed_hkl.get(id(crystal)) if isinstance(allowed_hkl, dict) else allowed_hkl
        if correct_depth:
            _depth_mm_state[0] = _depths_mm.get(id(layer), 0.0)

        u_key = (crystal.name, tuple(np.round(U, 4).ravel()))
        if u_key in seen_combos:
            if verbose:
                print(
                    f"  Skipping {label} (same crystal+orientation already enumerated)"
                )
            continue
        seen_combos.append(u_key)

        # Satellites are only meaningful around film (repeating-unit) G vectors.
        layer_has_satellites = id(layer) not in _buffer_set and bool(fringe_q_vecs)

        # Sphere enumeration limits for this layer's crystal
        HC_eV_ANG = 12398.4
        G_max = 4.0 * np.pi * E_max_eV / HC_eV_ANG
        G_max_sq = G_max ** 2
        _B = np.column_stack([crystal.Q(1, 0, 0), crystal.Q(0, 1, 0), crystal.Q(0, 0, 1)])
        a_star = float(np.linalg.norm(_B[:, 0]))
        b_star = float(np.linalg.norm(_B[:, 1]))
        c_star = float(np.linalg.norm(_B[:, 2]))
        h_lim = int(G_max / a_star) + 1
        k_lim = int(G_max / b_star) + 1
        l_lim = int(G_max / c_star) + 1

        if verbose:
            n_sat_orders = len(sat_orders) if layer_has_satellites else 0
            sat_info = f", ±{max_satellites} satellite orders" if n_sat_orders else ""
            print(
                f"  Enumerating {label} (G_max={G_max:.2f} Å⁻¹{sat_info}) ...", end="", flush=True
            )

        # Build full bounding-box grid, filter to sphere, rotate to lab — all vectorised.
        _HS, _KS, _LS = np.meshgrid(
            np.arange(-h_lim, h_lim + 1, dtype=np.int32),
            np.arange(-k_lim, k_lim + 1, dtype=np.int32),
            np.arange(-l_lim, l_lim + 1, dtype=np.int32),
            indexing="ij",
        )
        _hkl_s = np.column_stack([_HS.ravel(), _KS.ravel(), _LS.ravel()])
        _hkl_s = _hkl_s[np.any(_hkl_s != 0, axis=1)]
        _G_cry_s = (_B @ _hkl_s.T).T
        _mask_s = np.einsum("ij,ij->i", _G_cry_s, _G_cry_s) <= G_max_sq
        _hkl_s = _hkl_s[_mask_s]
        _G_lab_s = (U @ (_B @ _hkl_s.T)).T  # (M, 3)

        n_added = 0

        # ── Vectorised geometry filter for main Bragg peaks ───────────────────
        _G_lab_b = _G_lab_s.copy()
        _hkl_b   = _hkl_s.copy()

        kdG_b = _G_lab_b @ ki
        v1b   = kdG_b < 0
        _G_lab_b = _G_lab_b[v1b];  _hkl_b = _hkl_b[v1b];  kdG_b = kdG_b[v1b]

        if len(kdG_b) > 0:
            Gm2_b = np.einsum("ij,ij->i", _G_lab_b, _G_lab_b)
            lam_b = -4.0 * np.pi * kdG_b / Gm2_b
            v2b   = (lam_b >= lam_lo) & (lam_b <= lam_hi)
            _G_lab_b = _G_lab_b[v2b];  _hkl_b = _hkl_b[v2b];  lam_b = lam_b[v2b]
        else:
            lam_b = np.empty(0)

        if len(lam_b) > 0:
            E_b  = HC_eV_ANG / lam_b
            km_b = 2.0 * np.pi / lam_b
            kf_b = ki[None, :] * km_b[:, None] + _G_lab_b
            kf_b /= np.linalg.norm(kf_b, axis=1, keepdims=True)
            pix_b, on_det_b = camera.project_batch(kf_b, source_depth_mm=_depth_mm_state[0])

            if np.any(on_det_b):
                _G_lab_b = _G_lab_b[on_det_b];  _hkl_b = _hkl_b[on_det_b]
                lam_b    = lam_b[on_det_b];     E_b    = E_b[on_det_b]
                kf_b     = kf_b[on_det_b];      pix_b  = pix_b[on_det_b]

                tth_b = np.degrees(np.arccos(np.clip(kf_b[:, 0], -1.0, 1.0)))
                chi_b = np.degrees(np.arctan2(kf_b[:, 1], kf_b[:, 2] + 1e-17))
                az_b  = np.degrees(np.arctan2(kf_b[:, 2], kf_b[:, 1]))

                # ── Batch LP + spectrum pre-filter ────────────────────────────
                LP_b = _lorentz_pol_vec(tth_b)
                sw_b = spectrum(E_b)
                pre_ok = (LP_b > 0) & (sw_b > 0)

                # Batch allowed-HKL mask (skips structure-factor path entirely)
                if _layer_allowed_hkl is not None and pre_ok.any():
                    ok_idx = np.where(pre_ok)[0]
                    hkl_ok = np.fromiter(
                        (tuple(int(x) for x in _hkl_b[i]) in _layer_allowed_hkl
                         for i in ok_idx),
                        dtype=bool, count=len(ok_idx),
                    )
                    new_ok = np.zeros(len(pre_ok), dtype=bool)
                    new_ok[ok_idx[hkl_ok]] = True
                    pre_ok = new_ok

                pix_round = np.round(pix_b).astype(np.int64)

                for _si in np.where(pre_ok)[0]:
                    pix_key = (int(pix_round[_si, 0]), int(pix_round[_si, 1]))
                    if pix_key in seen_pix:
                        continue

                    if _layer_allowed_hkl is not None or geometry_only:
                        F2 = 1.0
                    else:
                        G_lab  = _G_lab_b[_si]
                        E      = float(E_b[_si])
                        kf_hat = kf_b[_si]
                        if structure_model == "average":
                            F_stack = stack.average_structure_factor(G_lab, energy_eV=E, kf_hat=kf_hat)
                        else:
                            F_stack = stack.structure_factor(G_lab, energy_eV=E, kf_hat=kf_hat)
                        F2 = abs(F_stack) ** 2
                        if f2_thresh == 0.0:
                            f2_thresh = max(1.0, F2 * 1e-3)
                        if F2 < f2_thresh:
                            continue

                    seen_pix.add(pix_key)
                    G_lab = _G_lab_b[_si]
                    h = int(_hkl_b[_si, 0]); k = int(_hkl_b[_si, 1]); l = int(_hkl_b[_si, 2])
                    lp = float(LP_b[_si]);    sw = float(sw_b[_si])
                    E  = float(E_b[_si])
                    spots.append({
                        "phase_label":     label,
                        "hkl":             (h, k, l),
                        "satellite_order": 0,
                        "is_superlattice": False,
                        "G_lab":           G_lab.copy(),
                        "E":               E,
                        "lambda":          float(lam_b[_si]),
                        "tth":             float(tth_b[_si]),
                        "chi":             float(chi_b[_si]),
                        "az":              float(az_b[_si]),
                        "pix":             (float(pix_b[_si, 0]), float(pix_b[_si, 1])),
                        "F2":              F2,
                        "F2_stack":        F2,
                        "LP":              lp,
                        "sw":              sw,
                        "I_raw":           F2 * lp * sw,
                    })
                    n_added += 1

                    # ── Thickness fringes / satellites for this Bragg peak ─────
                    if layer_has_satellites:
                        for q_fringe_vec, _ in fringe_q_vecs:
                            for m in sat_orders:
                                frac  = m + 0.5 if m > 0 else m - 0.5
                                G_sat = G_lab + frac * q_fringe_vec
                                n_added += _try_append(G_sat, (h, k, l), m, label)

        if verbose:
            print(f" {n_added} spots")

    # Normalise
    if spots:
        imax = max(s["I_raw"] for s in spots)
        for s in spots:
            s["intensity"] = s["I_raw"] / imax

    spots.sort(key=lambda s: s["intensity"], reverse=True)
    beam_divergence_ellipses(
        spots, camera, sigma_h_mrad, sigma_v_mrad, ki_hat=ki,
        sigma_beam_h_nm=sigma_beam_h_nm,
        sigma_beam_v_nm=sigma_beam_v_nm,
        n_hat_sample=n_hat_sample,
    )

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


# ─────────────────────────────────────────────────────────────────────────────
# 3D Q-SPACE RECONSTRUCTION AROUND A SPOT
# ─────────────────────────────────────────────────────────────────────────────


def qspace_around_spot(
    stack,
    hkl,
    layer=None,
    *,
    n_along=301,
    n_lateral=7,
    extent_along=None,
    extent_lateral=None,
    max_satellites=6,
    camera=None,
    E_min_eV=E_MIN_eV,
    E_max_eV=E_MAX_eV,
    source="bending_magnet",
    source_kwargs=None,
    kb_params=BM32_KB,
    ki_hat=None,
    structure_model="coherent",
    correct_depth=False,
    energy_ref_eV=None,
    verbose=True,
):
    """
    Reconstruct the 3-D reciprocal-space intensity distribution around one
    Bragg spot of a `LayeredCrystal` stack, and (optionally) the detector
    pixel/energy each voxel maps to.

    This generalises the discrete thickness-fringe/satellite construction
    used internally by :func:`simulate_laue_stack` (which only evaluates a
    handful of points at `G0 + m * q_fringe * n_hat`) into a continuous 3-D
    grid, so the full rod shape and any off-axis behaviour can be inspected
    directly.

    **Grid geometry**

    The grid is built in a local orthonormal frame centred on
    `G0 = layer.U @ layer.crystal.Q(*hkl)`:

    - one axis along `layer.n_hat` (the stack growth direction) — sampled
      finely, since finite-thickness fringes and superlattice satellites are
      *only* a function of `Q · n_hat` in this kinematical model;
    - two axes transverse to `n_hat` (arbitrary orthonormal in-plane
      directions) — sampled coarsely, since the structure factor of this
      model has no lateral interference structure and only varies smoothly
      there.

    **Physical intensity per voxel**

    For every voxel `Q = G0 + Δ`, the elastic (Laue) condition is solved
    for the unique photon energy that would excite it:

    $$
    E(Q) = hc \\,\\big/\\, \\lambda(Q), \\qquad
    \\lambda(Q) = -\\frac{4\\pi\\,(\\hat k_i \\cdot Q)}{|Q|^2}
    $$

    Voxels with `\\hat k_i \\cdot Q \\geq 0` or `E(Q)` outside
    `[E_min_eV, E_max_eV]` are marked unreachable (`I2 = 0`, `E = NaN`).
    For reachable voxels the diffracted direction `kf_hat` follows directly,
    and the stack structure factor is evaluated there
    (`stack.structure_factor` or `stack.average_structure_factor`,
    per `structure_model`), giving the same `F2 · LP · S(E)` intensity
    convention used throughout this module.

    **Detector intersection**

    If `camera` is given, every voxel's `kf_hat` is projected onto the
    detector (`camera.project_batch`), so the returned `pix` / `on_detector`
    arrays show exactly which pixel (if any) each point of the reconstructed
    volume lands on — this is how a satellite/fringe feature in Q-space
    connects to a supplementary spot seen on the detector image.

    Args:
    stack : LayeredCrystal
        The layered structure.
    hkl : (int, int, int)
        Miller indices of the reference reflection, expressed in
        `layer.crystal`'s lattice.
    layer : Layer, str, or None, optional
        Which layer's `(crystal, U, n_hat)` defines `G0` and the rod axis.
        A `str` is looked up by `.label` against `stack.all_layers`.
        Default `None` uses `stack.layers[0]` (the first film/repeating
        layer) — pass explicitly for buffer/substrate reflections or
        multi-layer stacks with more than one distinct film.
    n_along, n_lateral : int, optional
        Grid points along the rod axis / each transverse axis.
        `n_lateral` is used for both transverse axes; pass `1` to collapse
        the grid to a pure 1-D rod.
    extent_along, extent_lateral : float or None, optional
        Half-width of the grid (Å⁻¹) along the rod axis / transverse axes.
        `None` auto-scales from the fringe period (see Note).
    max_satellites : int, optional
        Only used to size the default `extent_along`: spans
        `±(max_satellites + 0.5) * 2π/period` so the same satellite orders
        `simulate_laue_stack` would report are covered.
    camera : Camera or None, optional
        Detector geometry.  When `None` (default), only the Q-space volume
        is computed — no pixel/energy intersection.
    E_min_eV, E_max_eV : float, optional
        White-beam energy window (eV).
    source, source_kwargs, kb_params :
        Synchrotron spectrum model, forwarded to :func:`_make_spectrum_fn`
        (same meaning as in :func:`simulate_laue_stack`).
    ki_hat : array-like (3,) or None, optional
        Incident beam direction (LT frame).  Default `KI_HAT` = `[1,0,0]`.
    structure_model : {'coherent', 'average'}, optional
        `'coherent'` (default) evaluates the full layer-by-layer coherent
        stack structure factor (`stack.structure_factor`) — physically
        accurate fringe/satellite intensities.  `'average'` uses
        `stack.average_structure_factor` — faster, satellite *positions*
        are identical but intensities are the composition-weighted
        average (matches `simulate_laue_stack(..., structure_model="average")`).
    correct_depth : bool, optional
        Apply the parallax correction for `layer`'s centre depth below the
        sample surface (see `Camera.project`'s `source_depth_mm`).
        Only affects `pix` when `camera` is given.  Default `False`.
    energy_ref_eV : float or None, optional
        Reference photon energy (eV) used to evaluate the unit-cell
        structure factor for **every** voxel at once (via
        `Layer.structure_factor_batch` → `StructureFactorForQ`), instead of
        each voxel's own energy.  `None` (default) uses the median energy
        of the reachable voxels.  This is the one approximation in an
        otherwise-exact computation: it only affects the smooth
        `f'(E)`/`f''(E)` anomalous-dispersion terms, which barely change
        across a typical local energy window — pass an explicit value if
        your window straddles an absorption edge and you want to pin it to
        one side.
    verbose : bool, optional
        Print a short summary (reachable/on-detector voxel counts, energy
        range) after the grid is evaluated.

    Returns:
    dict with keys:

        `'hkl'`, `'layer'` : the resolved inputs (`layer` as its label).
        `'G0'` : ndarray (3,) — nominal reflection Q vector (lab frame, Å⁻¹).
        `'axes'` : dict `{'n_hat', 't1', 't2'}` — orthonormal grid basis
            (lab frame unit vectors).
        `'along'`, `'lateral1'`, `'lateral2'` : ndarray, 1-D coordinate
            arrays (Å⁻¹ offset from `G0` along each axis).
        `'Q'` : ndarray, shape `(n_along, n_lateral, n_lateral, 3)` —
            lab-frame Q at every voxel.
        `'F2'` : ndarray, shape `(n_along, n_lateral, n_lateral)` —
            `|F_stack(Q)|²`, zero where unreachable.
        `'I'` : ndarray, same shape — `F2 · LP(2θ) · S(E)`, the intensity
            convention used elsewhere in this module (un-normalised).
        `'E'` : ndarray, same shape — photon energy (eV), `NaN` where
            unreachable.
        `'reachable'` : ndarray of bool, same shape — satisfies the elastic
            condition within `[E_min_eV, E_max_eV]`.
        `'pix'` : ndarray, shape `(n_along, n_lateral, n_lateral, 2)`, or
            `None` if `camera` was not given.  `(xcam, ycam)`, `NaN` where
            off-detector or unreachable.
        `'on_detector'` : ndarray of bool, same shape as `I`, or `None`.

    Note:
    * The default `extent_along` uses `stack.bilayer_thickness` when
      `stack.n_rep > 1`, else `layer.thickness` — the same period
      `simulate_laue_stack` uses for its satellite spacing.  If that
      period is degenerate (≤ 0), it falls back to `0.01 * |G0|`.
    * The default `extent_lateral` is `0.05 * extent_along`: this model has
      no transverse interference structure, so the transverse extent only
      needs to be large enough to make the array genuinely 3-D (e.g. for
      later isosurface/slice visualisation) — it is not a physical
      resolution limit.
    * The structure factor is evaluated for the whole grid in one vectorised
      batch per layer (`Layer.structure_factor_batch`), not per voxel — see
      `energy_ref_eV` above for the one approximation this introduces.
      Runtime therefore scales with `len(stack.all_layers)` (typically a
      handful of xrayutilities calls total), not with the number of voxels.

    Example:
    >>> vol = qspace_around_spot(stack, (0, 0, 2), camera=cam)
    >>> lit_up = vol['on_detector'] & (vol['I'] > 0)
    >>> vol['pix'][lit_up]              # every detector pixel this rod reaches
"""
    from .layers import LayeredCrystal

    if not isinstance(stack, LayeredCrystal):
        raise TypeError(f"stack must be a LayeredCrystal, got {type(stack).__name__}")
    if structure_model not in ("coherent", "average"):
        raise ValueError(f"structure_model must be 'coherent' or 'average', got {structure_model!r}")

    stack = _flatten_if_multiblock(stack)
    stack._update_offsets()

    # ── resolve layer ────────────────────────────────────────────────────────
    if layer is None:
        if not stack.layers:
            raise ValueError("stack has no repeating/film layers; pass `layer` explicitly")
        layer = stack.layers[0]
    elif isinstance(layer, str):
        match = next((l for l in stack.all_layers if l.label == layer), None)
        if match is None:
            available = [l.label for l in stack.all_layers]
            raise ValueError(f"no layer labeled {layer!r}; available labels: {available}")
        layer = match

    h, k, l = (int(x) for x in hkl)
    G0 = layer.U @ layer.crystal.Q(h, k, l)
    n_hat = np.asarray(layer.n_hat, dtype=float)
    n_hat = n_hat / np.linalg.norm(n_hat)

    # ── transverse basis (arbitrary orthonormal pair ⟂ n_hat) ────────────────
    tmp = np.array([1.0, 0.0, 0.0]) if abs(n_hat[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e_t1 = tmp - n_hat * np.dot(tmp, n_hat)
    e_t1 /= np.linalg.norm(e_t1)
    e_t2 = np.cross(n_hat, e_t1)

    # ── default extents ───────────────────────────────────────────────────────
    if extent_along is None:
        t_period = stack.bilayer_thickness if stack.n_rep > 1 else layer.thickness
        q_fringe = (2.0 * np.pi / t_period) if t_period > 1e-6 else 0.01 * float(np.linalg.norm(G0))
        extent_along = (max_satellites + 0.5) * q_fringe
    if extent_lateral is None:
        extent_lateral = 0.05 * extent_along

    along_vals = np.linspace(-extent_along, extent_along, n_along)
    t1_vals = np.linspace(-extent_lateral, extent_lateral, n_lateral) if n_lateral > 1 else np.array([0.0])
    t2_vals = np.linspace(-extent_lateral, extent_lateral, n_lateral) if n_lateral > 1 else np.array([0.0])

    AA, T1, T2 = np.meshgrid(along_vals, t1_vals, t2_vals, indexing="ij")
    shape = AA.shape
    Q_grid = (
        G0[None, None, None, :]
        + AA[..., None] * n_hat
        + T1[..., None] * e_t1
        + T2[..., None] * e_t2
    )
    Q_flat = Q_grid.reshape(-1, 3)
    n_vox = len(Q_flat)

    # ── elastic (Laue) condition per voxel, vectorised ────────────────────────
    ki = np.asarray(ki_hat if ki_hat is not None else KI_HAT, dtype=float)
    ki = ki / np.linalg.norm(ki)
    lam_lo, lam_hi = en2lam(E_max_eV), en2lam(E_min_eV)

    kdG = Q_flat @ ki
    Gm2 = np.einsum("ij,ij->i", Q_flat, Q_flat)
    backward = kdG < 0

    lam_flat = np.full(n_vox, np.nan)
    lam_flat[backward] = -4.0 * np.pi * kdG[backward] / Gm2[backward]
    reachable_flat = backward & (lam_flat >= lam_lo) & (lam_flat <= lam_hi)
    idx = np.where(reachable_flat)[0]

    E_flat = np.full(n_vox, np.nan)
    kf_flat = np.full((n_vox, 3), np.nan)
    if len(idx):
        E_flat[idx] = HC / lam_flat[idx]
        km = 2.0 * np.pi / lam_flat[idx]
        kf = ki[None, :] * km[:, None] + Q_flat[idx]
        kf /= np.linalg.norm(kf, axis=1, keepdims=True)
        kf_flat[idx] = kf

    # ── structure factor + LP + spectrum, vectorised over all reachable voxels ─
    # The elastic-condition geometry (tth, kf_hat) stays exact and per-voxel;
    # only the unit-cell structure factor's F_uc(Q, E) is evaluated at one
    # shared reference energy (see `energy_ref_eV`), which is what collapses
    # this from N scalar xrayutilities calls to len(stack.all_layers) batched
    # ones.
    spectrum = _make_spectrum_fn(source, source_kwargs or {}, kb_params)
    F2_flat = np.zeros(n_vox)
    I_flat = np.zeros(n_vox)

    if len(idx):
        tth_arr = np.degrees(np.arccos(np.clip(kf_flat[idx, 0], -1.0, 1.0)))
        LP_arr = _lorentz_pol_vec(tth_arr)
        sw_arr = spectrum(E_flat[idx])

        E_ref = (
            float(energy_ref_eV) if energy_ref_eV is not None
            else float(np.median(E_flat[idx]))
        )
        batch_fn = (
            stack.average_structure_factor_batch
            if structure_model == "average"
            else stack.structure_factor_batch
        )
        F_arr = batch_fn(Q_flat[idx], E_ref, kf_hat_arr=kf_flat[idx])
        F2_arr = np.abs(F_arr) ** 2

        keep = (LP_arr > 0) & (sw_arr > 0)
        keep_idx = idx[keep]
        F2_flat[keep_idx] = F2_arr[keep]
        I_flat[keep_idx] = F2_arr[keep] * LP_arr[keep] * sw_arr[keep]

    # ── detector intersection ─────────────────────────────────────────────────
    pix_flat = None
    on_det_flat = None
    if camera is not None:
        depth_mm = _layer_depths_mm(stack).get(id(layer), 0.0) if correct_depth else 0.0
        pix_flat = np.full((n_vox, 2), np.nan)
        on_det_flat = np.zeros(n_vox, dtype=bool)
        if len(idx):
            pix_idx, on_det_idx = camera.project_batch(kf_flat[idx], source_depth_mm=depth_mm)
            pix_flat[idx] = pix_idx
            on_det_flat[idx] = on_det_idx

    if verbose:
        n_on_det = int(on_det_flat.sum()) if on_det_flat is not None else None
        e_reach = E_flat[idx]
        e_range = f"{e_reach.min():.0f}–{e_reach.max():.0f} eV" if len(idx) else "—"
        print(
            f"  qspace_around_spot: hkl={hkl}  layer='{layer.label}'  "
            f"grid={shape}  reachable={len(idx)}/{n_vox}  E∈[{e_range}]"
            + (f"  on_detector={n_on_det}" if n_on_det is not None else "")
        )

    return {
        "hkl": (h, k, l),
        "layer": layer.label,
        "G0": G0,
        "axes": {"n_hat": n_hat, "t1": e_t1, "t2": e_t2},
        "along": along_vals,
        "lateral1": t1_vals,
        "lateral2": t2_vals,
        "Q": Q_grid,
        "F2": F2_flat.reshape(shape),
        "I": I_flat.reshape(shape),
        "E": E_flat.reshape(shape),
        "reachable": reachable_flat.reshape(shape),
        "pix": pix_flat.reshape(*shape, 2) if pix_flat is not None else None,
        "on_detector": on_det_flat.reshape(shape) if on_det_flat is not None else None,
    }


def qspace_per_layer(
    stack,
    hkl,
    layers=None,
    *,
    n_along=301,
    n_lateral=7,
    extent_along=None,
    extent_lateral=None,
    max_satellites=6,
    camera=None,
    E_min_eV=E_MIN_eV,
    E_max_eV=E_MAX_eV,
    source="bending_magnet",
    source_kwargs=None,
    kb_params=BM32_KB,
    ki_hat=None,
    structure_model="coherent",
    correct_depth=False,
    energy_ref_eV=None,
    verbose=True,
):
    """
    Run :func:`qspace_around_spot` once per layer in the stack, using each
    layer's own G0 and n_hat as the grid centre, but always evaluating the
    **full coherent** structure factor of the entire stack at every Q point.

    Because different layers may have slightly different lattice parameters,
    their individual Bragg peaks sit at slightly different absolute
    ``Q · n̂`` positions.  By running the simulation for each layer
    separately you can see all peaks on a common absolute-Qn axis and
    directly compare them against the measured image — revealing whether
    any apparent shift between simulation and measurement is caused by
    inter-layer interference, lattice mismatch, or a calibration offset.

    Args:
        stack: :class:`~nrxrdct.laue.layers.LayeredCrystal` stack.
        hkl: Miller indices ``(h, k, l)`` of the reflection.
        layers: List of :class:`~nrxrdct.laue.layers.Layer` objects (or layer
            labels as strings) to simulate.  ``None`` (default) uses
            ``stack.all_layers`` — every buffer and repeating layer.
        All remaining keyword arguments are forwarded unchanged to
        :func:`qspace_around_spot`.

    Returns:
        ``list[dict]`` — one vol dict per layer, in the same order as
        ``layers`` (or ``stack.all_layers``), each with the same structure as
        the return value of :func:`qspace_around_spot`.

    Example:
    >>> vols = qspace_per_layer(stack, (0, 0, 2), camera=cam)
    >>> plot_qspace_summary(vol_main, camera=cam, image=img,
    ...                     per_layer_vols=vols)
    """
    from .layers import LayeredCrystal

    if not isinstance(stack, LayeredCrystal):
        raise TypeError(f"stack must be a LayeredCrystal, got {type(stack).__name__}")

    if layers is None:
        # Deduplicate by label: stack.layers concatenates every block's layers,
        # so a multi-block stack (e.g. lower_cladding / MQW / upper_cladding)
        # can contain the same material type in multiple blocks.  Keeping the
        # first occurrence per label avoids identical duplicate simulations.
        seen_labels: set[str] = set()
        layers = []
        for lyr in stack.all_layers:
            if lyr.label not in seen_labels:
                seen_labels.add(lyr.label)
                layers.append(lyr)
    else:
        resolved = []
        for lyr in layers:
            if isinstance(lyr, str):
                match = next((l for l in stack.all_layers if l.label == lyr), None)
                if match is None:
                    raise ValueError(f"no layer labeled {lyr!r}")
                resolved.append(match)
            else:
                resolved.append(lyr)
        layers = resolved

    vols = []
    for layer in layers:
        v = qspace_around_spot(
            stack, hkl, layer=layer,
            n_along=n_along, n_lateral=n_lateral,
            extent_along=extent_along, extent_lateral=extent_lateral,
            max_satellites=max_satellites,
            camera=camera, E_min_eV=E_min_eV, E_max_eV=E_max_eV,
            source=source, source_kwargs=source_kwargs,
            kb_params=kb_params, ki_hat=ki_hat,
            structure_model=structure_model,
            correct_depth=correct_depth,
            energy_ref_eV=energy_ref_eV,
            verbose=verbose,
        )
        vols.append(v)
    return vols


# ─────────────────────────────────────────────────────────────────────────────
# DARWIN (DYNAMICAL) LAUE SIMULATION
# ─────────────────────────────────────────────────────────────────────────────

# Classical electron radius (Å)
_R_E_ANG: float = 2.8179403227e-5


def _darwin_n_eff(
    F_abs: float,
    lam: float,
    tth_deg: float,
    V_uc: float,
    n_cells: int,
    d: float,
) -> float:
    """
    Darwin primary-extinction corrected effective cell count.

    Replaces the kinematical factor N with

        N_eff = N_ext × tanh(N / N_ext)

    where the extinction length in unit cells is

        N_ext = ξ / d ,   ξ = V_uc sin θ / (rₑ λ |F_H|)

    * N ≪ N_ext  →  N_eff ≈ N          (kinematical, thin crystal)
    * N ≫ N_ext  →  N_eff ≈ N_ext      (dynamical saturation, thick crystal)
"""
    if F_abs < 1e-10 or d < 1e-6:
        return float(n_cells)
    sin_th = max(abs(np.sin(np.radians(tth_deg / 2.0))), 1e-6)
    xi = V_uc * sin_th / (_R_E_ANG * lam * F_abs)   # extinction depth (Å)
    N_ext = xi / d
    if N_ext < 1e-6:
        return 1.0
    return float(N_ext * np.tanh(float(n_cells) / N_ext))


def simulate_laue_darwin(
    stack,
    camera,
    E_min_eV: float = E_MIN_eV,
    E_max_eV: float = E_MAX_eV,
    source: str = "bending_magnet",
    source_kwargs: dict | None = None,
    f2_thresh: float = F2_THRESHOLD,
    ki_hat=None,
    kb_params=BM32_KB,
    max_satellites: int = 5,
    structure_model: str = "average",
    sigma_h_mrad: float = 0.0,
    sigma_v_mrad: float = 0.0,
    sigma_beam_h_nm: float = 0.0,
    sigma_beam_v_nm: float = 0.0,
    n_hat_sample=None,
    correct_depth: bool = False,
    allowed_hkl: "frozenset | dict | None" = None,
    verbose: bool = True,
):
    """
    White-beam Laue simulation with Darwin (dynamical diffraction) intensities.

    Spot **positions** are identical to :func:`simulate_laue_stack` — they
    come from the same Laue condition and camera geometry.  The difference is
    in how the **intensity** of each spot is computed.

    **Kinematical vs Darwin**
    The kinematical model gives

        I ∝ |F_uc|² × N²

    which diverges for thick perfect-crystal regions (substrate).  The Darwin
    model corrects this with the *primary extinction* factor:

        N_eff = N_ext × tanh(N / N_ext)

        N_ext = V_uc sin θ / (rₑ λ |F_uc|) / d    [extinction length, unit cells]

    so that thin layers remain kinematical (N_eff ≈ N) while thick regions
    saturate at N_ext.

    For a multilayer stack the amplitudes from all layers are summed
    **coherently** (same phase relationship as the kinematical model), so
    superlattice satellites and fringe patterns are preserved:

        F_total(G) = F_buffer(G) + exp(i Qₙ z_buf) · F_unit(G) · S_rep(Qₙ Λ)

        F_layer_i(G) = F_uc_i × N_eff_i × exp(i Qₙ z_i)

    Absorption limiting (Beer-Lambert) is applied on top of extinction:
    the effective cell count is `min(N_eff_darwin, N_eff_absorption)`.

    **Alloys**
    Any `xu.materials.Crystal`-compatible object can be used as the layer
    crystal, including alloys created with xrayutilities (e.g.
    `xu.materials.HexagonalAlloy`, VCA compositions, etc.).  The
    composition enters through the structure factor `F_uc` and the unit-cell
    volume `V_uc`.

    Args:
    stack : LayeredCrystal
        Same input as :func:`simulate_laue_stack`.
    camera : Camera
    E_min_eV, E_max_eV : float
    source : str
        See :func:`simulate_laue_stack` and :func:`_make_spectrum_fn` for the
        full list (`'bending_magnet'`, `'wiggler'`, `'undulator'`,
        `'flat'`, `'shadow4'`, `'tabulated'`).
    source_kwargs : dict, optional
    f2_thresh : float
        Minimum Darwin-corrected |F_total|² to keep a spot.
    ki_hat : array-like (3,) or None
    kb_params : dict or None
    max_satellites : int, optional
        Number of satellite / fringe orders to probe on each side of every
        Bragg peak (default 5).  Probes orders ±1 … ±max_satellites.
        Set to 0 to skip satellite calculation entirely.
    structure_model : {'coherent', 'average'}, optional
        See :func:`simulate_laue_stack` for full description.  In
        `'average'` mode G vectors are enumerated from buffer layers only,
        intra-period `exp(i Qₙ z_rel)` phases are dropped, and `S_rep`
        is kept — producing a single average Bragg peak with satellites, as
        seen in a monochromatic scan.  Darwin-corrected `N_eff` values are
        still computed and reported in the returned `'N_eff'` key.
    allowed_hkl : frozenset or dict or None, optional
        Pre-computed set of allowed reflections from
        :func:`precompute_allowed_hkl`.  When supplied, systematically absent
        reflections are filtered **before** any geometry computation, avoiding
        :func:`_darwin_amp` calls for forbidden hkl.  Accepts:

        * ``frozenset`` — shared across all phases (useful when all layers
          have the same crystal type).
        * ``dict`` mapping ``id(crystal)`` → ``frozenset`` — per-crystal
          sets, as returned by the internal cache.

        Default ``None``: all (h, k, l) in the sphere pass to the Laue
        condition check, relying on the structure factor threshold to reject
        forbidden reflections.
    correct_depth : bool, optional
        When ``True``, each layer's spots are projected from the beam-path
        depth of that layer's centre rather than from the sample surface
        (depth = 0).  The depth is taken from :func:`_layer_depths_mm` and
        divided by ``cos(angle between ki and surface normal)`` to convert
        from normal-direction depth to along-beam displacement, which is
        then passed as ``source_depth_mm`` to ``camera.project()``.

        This corrects the **depth-parallax** shift: in reflection geometry
        a spot originating at depth *z* exits from a laterally displaced
        surface point, causing substrate spots to appear at a different
        detector position than surface or film spots.  Default ``False``
        (all spots projected from the surface, matching the behaviour of
        :func:`simulate_laue_stack`).
    sigma_h_mrad, sigma_v_mrad : float, optional
        Horizontal / vertical beam divergence 1σ (mrad).  See
        :func:`beam_divergence_ellipses`.  Typical BM32/ESRF: 2–3 / 0.2–0.5.
    sigma_beam_h_nm, sigma_beam_v_nm : float, optional
        Beam footprint size 1σ at the sample (nm).  Requires
        ``n_hat_sample`` to activate footprint broadening.
    n_hat_sample : array-like (3,), optional
        Sample surface normal in the LT frame (for footprint broadening).
    verbose : bool

    Returns:
    spots : list[dict]
        Same keys as :func:`simulate_laue_stack` plus:

        ``'N_eff'``          list of Darwin N_eff per layer
        ``'N_ext'``          list of extinction lengths (unit cells) per layer
        ``'F2_darwin'``      ``|F_total_darwin|²``
        ``'source_depth_mm'``  beam-path depth used for projection (mm);
                             0.0 when ``correct_depth=False``

    Note:
    As with :func:`simulate_laue_stack`, a multi-block `stack` is
    transparently flattened via
    :func:`~nrxrdct.laue.layers.combine_stacks` before simulation.
"""
    stack = _flatten_if_multiblock(stack)
    stack._update_offsets()
    source_kwargs = source_kwargs or {}
    ki = np.asarray(ki_hat if ki_hat is not None else KI_HAT, dtype=float)
    ki /= np.linalg.norm(ki)

    # ── Depth correction ──────────────────────────────────────────────────────
    _depth_mm_state = [0.0]
    if correct_depth:
        _depths_mm_raw = _layer_depths_mm(stack)
        _n_hat_norm = np.asarray(stack.n_hat, dtype=float)
        _n_hat_norm = _n_hat_norm / np.linalg.norm(_n_hat_norm)
        _cos_inc = abs(float(np.dot(ki, _n_hat_norm)))
        if _cos_inc > 1e-6:
            _depths_mm = {k: v / _cos_inc for k, v in _depths_mm_raw.items()}
        else:
            _depths_mm = _depths_mm_raw
    else:
        _depths_mm = {}

    lam_lo = en2lam(E_max_eV)
    lam_hi = en2lam(E_min_eV)

    _spectrum = _make_spectrum_fn(source, source_kwargs, kb_params)

    # ── Fringe / satellite wavevectors ────────────────────────────────────────
    # Same logic as simulate_laue_stack: probe G_hkl + frac * q_fringe
    # where frac = m ± 0.5  (bright-fringe maxima between dark zeros).
    MAX_FRINGE_THICK_ANG = 20_000.0
    fringe_q_vecs: list = []
    sat_orders = [m for m in range(-max_satellites, max_satellites + 1) if m != 0]

    if max_satellites > 0 and stack.layers:
        seen_t: set = set()
        for lyr in stack.layers:
            t = lyr.thickness
            if t < 1e-6 or t > MAX_FRINGE_THICK_ANG:
                continue
            t_key = round(t, 2)
            if t_key in seen_t:
                continue
            seen_t.add(t_key)
            q_vec = (2.0 * np.pi / t) * np.asarray(stack.n_hat)
            fringe_q_vecs.append((q_vec, f"layer '{lyr.label}' (t={t/10:.1f} nm)"))

        Lambda = stack.bilayer_thickness
        if stack.n_rep > 1 and Lambda > 1e-6:
            t_key = round(Lambda, 2)
            if t_key not in seen_t:
                q_vec = (2.0 * np.pi / Lambda) * np.asarray(stack.n_hat)
                fringe_q_vecs.append((q_vec, f"bilayer Λ={Lambda/10:.1f} nm"))

    if verbose and fringe_q_vecs:
        print("  Fringe / satellite periods to probe:")
        for qv, desc in fringe_q_vecs:
            t_nm = 2.0 * np.pi / np.linalg.norm(qv) / 10.0
            print(f"    {desc}  →  2π/t = {np.linalg.norm(qv):.4f} Å⁻¹  (t = {t_nm:.2f} nm)")

    # ── Darwin amplitude helper ───────────────────────────────────────────────
    def _darwin_amp(G_vec, E_ev, lam_ang, tth_deg, kf_hat=None):
        """
        Compute the Darwin-corrected coherent amplitude F_total for G_vec.

        kf_hat : unit vector of the diffracted beam (lab frame).  When given,
        the two-beam absorption correction and overlying-layer attenuation are
        applied to each buffer layer's contribution.

        Returns (F_total, n_eff_list, n_ext_list).
"""
        Qn = float(np.dot(G_vec, stack.n_hat))

        # ── Overlying-layer transmission helper ───────────────────────────────
        def _T_slab(lyr, thickness):
            if kf_hat is None:
                return 1.0
            mu = lyr._linear_mu(E_ev)
            if mu <= 0:
                return 1.0
            kf = np.asarray(kf_hat, dtype=float)
            cos_in  = max(abs(float(np.dot(ki, stack.n_hat))), 1e-3)
            cos_out = max(abs(float(np.dot(stack.n_hat, kf))), 1e-3)
            return float(np.exp(-mu * thickness * (1.0 / cos_in + 1.0 / cos_out)))

        # Attenuation from the full MQW block (sits above all buffer layers)
        T_mqw = 1.0
        for lyr in stack.layers:
            T_mqw *= _T_slab(lyr, lyr.thickness * stack.n_rep)

        F_buf = 0.0 + 0j
        n_eff_b, n_ext_b = [], []
        for i, (lyr, z0) in enumerate(zip(stack.buffer_layers, stack._buffer_z_offsets)):
            # Attenuation from MQW + all buffer layers shallower than i
            T_above = T_mqw
            for j in range(i + 1, len(stack.buffer_layers)):
                T_above *= _T_slab(stack.buffer_layers[j],
                                   stack.buffer_layers[j].thickness)

            Q_cry_l = lyr.U.T @ G_vec
            F_uc = lyr.crystal.StructureFactor(Q_cry_l, en=E_ev)
            if not (np.isfinite(F_uc.real) and np.isfinite(F_uc.imag)):
                n_eff_b.append(0.0); n_ext_b.append(np.inf)
                continue
            F_abs = abs(F_uc)
            V_uc  = lyr.crystal.lattice.UnitCellVolume()
            N_d   = _darwin_n_eff(F_abs, lam_ang, tth_deg, V_uc, lyr.n_cells, lyr.d)
            N_a   = float(lyr._effective_n_cells(E_ev, kf_hat=kf_hat))
            N_eff = min(N_d, N_a)
            sin_th = max(abs(np.sin(np.radians(tth_deg / 2.0))), 1e-6)
            N_ext_l = (V_uc * sin_th / (_R_E_ANG * lam_ang * max(F_abs, 1e-30)) / lyr.d)
            n_eff_b.append(N_eff); n_ext_b.append(N_ext_l)
            # Buffer layers always keep their depth phase — they are not part
            # of the periodic unit and their z0 offsets are fixed.
            F_buf += T_above * F_uc * N_eff * np.exp(1j * Qn * z0)

        F_unit = 0.0 + 0j
        n_eff_u, n_ext_u = [], []
        for lyr, z0_rel in zip(stack.layers, stack._z_offsets):
            Q_cry_l = lyr.U.T @ G_vec
            F_uc = lyr.crystal.StructureFactor(Q_cry_l, en=E_ev)
            if not (np.isfinite(F_uc.real) and np.isfinite(F_uc.imag)):
                n_eff_u.append(0.0); n_ext_u.append(np.inf)
                continue
            F_abs = abs(F_uc)
            V_uc  = lyr.crystal.lattice.UnitCellVolume()
            N_d   = _darwin_n_eff(F_abs, lam_ang, tth_deg, V_uc, lyr.n_cells, lyr.d)
            sin_th = max(abs(np.sin(np.radians(tth_deg / 2.0))), 1e-6)
            N_ext_l = (V_uc * sin_th / (_R_E_ANG * lam_ang * max(F_abs, 1e-30)) / lyr.d)
            n_eff_u.append(N_d); n_ext_u.append(N_ext_l)
            # Average mode: drop intra-period z_rel phases to produce the
            # structural envelope of one bilayer period.  S_rep is still
            # applied below so satellite peaks appear at the correct positions.
            phase = 1.0 if structure_model == "average" else np.exp(1j * Qn * z0_rel)
            F_unit += F_uc * N_d * phase

        if stack.layers:
            z_buf   = stack._buffer_thickness
            Lambda  = stack._bilayer_thickness
            phi_rep = Qn * Lambda
            phi_mod = phi_rep % (2.0 * np.pi)
            if abs(phi_mod) < 1e-10 or abs(phi_mod - 2.0 * np.pi) < 1e-10:
                S_rep = float(stack.n_rep) + 0j
            else:
                S_rep = ((1.0 - np.exp(1j * stack.n_rep * phi_rep))
                         / (1.0 - np.exp(1j * phi_rep)))
            F_mqw = np.exp(1j * Qn * z_buf) * F_unit * S_rep
        else:
            F_mqw = 0.0 + 0j

        return F_buf + F_mqw, n_eff_b + n_eff_u, n_ext_b + n_ext_u

    seen_pix: set = set()
    spots: list = []

    # ── Deduplicate orientations for enumeration ──────────────────────────────
    # In average mode enumerate from buffer layers plus the first MQW layer.
    # Same reasoning as simulate_laue_stack: a dissimilar substrate (sapphire,
    # SrTiO3 …) has G vectors at completely different positions from the film,
    # so substrate-only enumeration would miss all film reflections.  Adding
    # the first MQW layer guarantees that film G vectors are always probed.
    # Crystal+orientation deduplication below removes any true duplicate.
    if structure_model == "average":
        _enum_pool = stack.buffer_layers + stack.layers[:1]
    else:
        _enum_pool = stack.all_layers

    seen_combos: list = []
    enum_layers: list = []
    for layer in _enum_pool:
        u_key = (layer.crystal.name, tuple(np.round(layer.U, 4).ravel()))
        if u_key not in seen_combos:
            seen_combos.append(u_key)
            enum_layers.append(layer)

    # Buffer layers provide Bragg peaks only — satellites around substrate
    # G vectors combined with film fringe q vectors are physically meaningless.
    _buffer_set = set(id(l) for l in stack.buffer_layers)

    for enum_layer in enum_layers:
        U = enum_layer.U
        crystal = enum_layer.crystal
        label = enum_layer.label

        if correct_depth:
            _depth_mm_state[0] = _depths_mm.get(id(enum_layer), 0.0)

        layer_has_satellites = id(enum_layer) not in _buffer_set and bool(fringe_q_vecs)
        sat_info = f", ±{max_satellites} satellite orders" if layer_has_satellites else ""

        # Sphere enumeration limits for this layer's crystal
        HC_eV_ANG = 12398.4
        G_max = 4.0 * np.pi * E_max_eV / HC_eV_ANG
        G_max_sq = G_max ** 2
        a_star = float(np.linalg.norm(crystal.Q(1, 0, 0)))
        b_star = float(np.linalg.norm(crystal.Q(0, 1, 0)))
        c_star = float(np.linalg.norm(crystal.Q(0, 0, 1)))
        h_lim = int(G_max / a_star) + 1
        k_lim = int(G_max / b_star) + 1
        l_lim = int(G_max / c_star) + 1

        if verbose:
            print(f"  Enumerating {label} (G_max={G_max:.2f} Å⁻¹{sat_info}) ...", end="", flush=True)

        n_added = 0

        # ── Vectorised geometry — mirrors simulate_laue_stack ─────────────────
        # Build B matrix once; Q(h,k,l) = B @ [h,k,l] by linearity.
        _B = np.column_stack([crystal.Q(1, 0, 0), crystal.Q(0, 1, 0),
                              crystal.Q(0, 0, 1)])
        _HS, _KS, _LS = np.meshgrid(
            np.arange(-h_lim, h_lim + 1, dtype=np.int32),
            np.arange(-k_lim, k_lim + 1, dtype=np.int32),
            np.arange(-l_lim, l_lim + 1, dtype=np.int32),
            indexing="ij",
        )
        _hkl_all = np.column_stack([_HS.ravel(), _KS.ravel(), _LS.ravel()])
        _hkl_all = _hkl_all[np.any(_hkl_all != 0, axis=1)]
        _G_cry_all = (_B @ _hkl_all.T).T                    # (N, 3)
        _in_sphere = np.einsum("ij,ij->i", _G_cry_all, _G_cry_all) <= G_max_sq
        _hkl_all = _hkl_all[_in_sphere]

        # allowed_hkl pre-filter: skip systematically absent reflections before
        # the geometry loop so _darwin_amp is never called for forbidden hkl.
        _layer_allowed = (
            allowed_hkl.get(id(crystal)) if isinstance(allowed_hkl, dict)
            else allowed_hkl
        )
        if _layer_allowed is not None and len(_hkl_all):
            _hkl_ok = np.fromiter(
                (tuple(int(x) for x in row) in _layer_allowed
                 for row in _hkl_all),
                dtype=bool, count=len(_hkl_all),
            )
            _hkl_all = _hkl_all[_hkl_ok]

        # G vectors in lab frame
        _G_lab_all = (_B @ _hkl_all.T).T @ U.T              # (M, 3)

        def _batch_darwin(G_lab_batch, hkl_batch, sat_order):
            """Geometry-filter a batch, call _darwin_amp on survivors, append."""
            if len(G_lab_batch) == 0:
                return 0
            kdG = G_lab_batch @ ki
            v1  = kdG < 0
            if not np.any(v1):
                return 0
            G_b   = G_lab_batch[v1];  hkl_b = hkl_batch[v1];  kdG_b = kdG[v1]
            Gm2_b = np.einsum("ij,ij->i", G_b, G_b)
            lam_b = -4.0 * np.pi * kdG_b / Gm2_b
            v2    = (lam_b >= lam_lo) & (lam_b <= lam_hi)
            if not np.any(v2):
                return 0
            G_b   = G_b[v2];   hkl_b = hkl_b[v2];   lam_b = lam_b[v2]
            E_b   = HC / lam_b
            km_b  = 2.0 * np.pi / lam_b
            kf_b  = ki[None, :] * km_b[:, None] + G_b
            kf_b /= np.linalg.norm(kf_b, axis=1, keepdims=True)
            pix_b, on_det_b = camera.project_batch(
                kf_b, source_depth_mm=_depth_mm_state[0])
            if not np.any(on_det_b):
                return 0
            G_b   = G_b[on_det_b];   hkl_b  = hkl_b[on_det_b]
            lam_b = lam_b[on_det_b]; E_b    = E_b[on_det_b]
            kf_b  = kf_b[on_det_b];  pix_b  = pix_b[on_det_b]
            tth_b = np.degrees(np.arccos(np.clip(kf_b[:, 0], -1.0, 1.0)))
            chi_b = np.degrees(np.arctan2(kf_b[:, 1], kf_b[:, 2] + 1e-17))
            az_b  = np.degrees(np.arctan2(kf_b[:, 2], kf_b[:, 1]))
            LP_b  = _lorentz_pol_vec(tth_b)
            sw_b  = _spectrum(E_b)
            pre_ok = (LP_b > 0) & (sw_b > 0)
            pix_round = np.round(pix_b).astype(np.int64)
            eff_thresh = f2_thresh * 1e-4 if sat_order != 0 else f2_thresh
            n_new = 0
            for _si in np.where(pre_ok)[0]:
                pix_key = (int(pix_round[_si, 0]), int(pix_round[_si, 1]))
                if pix_key in seen_pix:
                    continue
                F_total, n_effs, n_exts = _darwin_amp(
                    G_b[_si], float(E_b[_si]), float(lam_b[_si]),
                    float(tth_b[_si]), kf_hat=kf_b[_si],
                )
                F2 = abs(F_total) ** 2
                if F2 < eff_thresh:
                    continue
                seen_pix.add(pix_key)
                spots.append({
                    "phase_label":     label,
                    "hkl":             (int(hkl_b[_si, 0]),
                                        int(hkl_b[_si, 1]),
                                        int(hkl_b[_si, 2])),
                    "satellite_order": sat_order,
                    "is_superlattice": sat_order != 0,
                    "G_lab":           G_b[_si].copy(),
                    "E":               float(E_b[_si]),
                    "lambda":          float(lam_b[_si]),
                    "tth":             float(tth_b[_si]),
                    "chi":             float(chi_b[_si]),
                    "az":              float(az_b[_si]),
                    "pix":             (float(pix_b[_si, 0]),
                                        float(pix_b[_si, 1])),
                    "source_depth_mm": _depth_mm_state[0],
                    "F2":              F2,
                    "F2_darwin":       F2,
                    "LP":              float(LP_b[_si]),
                    "sw":              float(sw_b[_si]),
                    "I_raw":           F2 * float(LP_b[_si]) * float(sw_b[_si]),
                    "N_eff":           n_effs,
                    "N_ext":           n_exts,
                })
                n_new += 1
            return n_new

        # Main Bragg peaks
        n_added += _batch_darwin(_G_lab_all, _hkl_all, 0)

        # Thickness fringes / superlattice satellites
        if layer_has_satellites:
            for q_fringe_vec, _ in fringe_q_vecs:
                for m in sat_orders:
                    frac = m + 0.5 if m > 0 else m - 0.5
                    _G_sat = _G_lab_all + frac * q_fringe_vec
                    n_added += _batch_darwin(_G_sat, _hkl_all, m)

        if verbose:
            print(f" {n_added} spots")

    if spots:
        imax = max(s["I_raw"] for s in spots)
        for s in spots:
            s["intensity"] = s["I_raw"] / imax

    spots.sort(key=lambda s: s["intensity"], reverse=True)
    beam_divergence_ellipses(
        spots, camera, sigma_h_mrad, sigma_v_mrad, ki_hat=ki,
        sigma_beam_h_nm=sigma_beam_h_nm,
        sigma_beam_v_nm=sigma_beam_v_nm,
        n_hat_sample=n_hat_sample,
    )

    if verbose:
        print(f"  Total spots (Darwin): {len(spots)}")
        if stack.layers and stack.n_rep > 1:
            print(
                f"  Superlattice satellite 2π/Λ = "
                f"{2*np.pi/stack.bilayer_thickness:.5f} Å⁻¹"
            )

    return spots


def simulate_laue_multibeam(
    crystal,
    U,
    camera,
    E_min_eV: float = E_MIN_eV,
    E_max_eV: float = E_MAX_eV,
    source: str = "bending_magnet",
    source_kwargs: dict | None = None,
    f2_thresh: float = F2_THRESHOLD,
    ki_hat=None,
    kb_params=BM32_KB,
    delta_E_rel: float = 0.005,
    umweg_f2_thresh: float | None = None,
    verbose: bool = True,
):
    """
    White-beam Laue simulation with N-beam (multi-beam) dynamical diffraction.

    Extends :func:`simulate_laue` with **Umweganregung** (detour excitation):
    systematically-forbidden reflections that are activated via two-step
    scattering paths through the crystal.

    **Physics**

    In two-beam kinematical theory, a reflection $\\mathbf{G}_t$ with
    $|F(\\mathbf{G}_t)| = 0$ (systematic absence) produces no spot.  When a
    secondary reflection $\\mathbf{G}_s$ is simultaneously in Bragg condition
    at the same energy $E_t$, the beam can take the detour

    .. math::

        \\mathbf{k}_i \\xrightarrow{\\mathbf{G}_s} \\mathbf{k}_{int}
        \\xrightarrow{\\mathbf{G}_r} \\mathbf{k}_f

    with $\\mathbf{G}_r = \\mathbf{G}_t - \\mathbf{G}_s$, producing a spot at
    the position $\\mathbf{G}_t$ would occupy even though
    $F(\\mathbf{G}_t) = 0$.

    The Umweganregung amplitude at $\\mathbf{G}_t$ is

    .. math::

        A_t = \\sum_{\\mathbf{G}_s}
              \\frac{F(\\mathbf{G}_s)\\,F(\\mathbf{G}_r)}{\\xi_s + i\\,\\delta_s}

    where

    * $\\xi_s$ — excitation error of $\\mathbf{G}_s$ at energy $E_t$
      (Å⁻¹, positive when $\\mathbf{G}_s$ is not exactly in Bragg condition)
    * $\\delta_s = r_e \\lambda |F(\\mathbf{G}_s)| / (\\pi V_{uc} \\sin\\theta_s)$ —
      Darwin half-width of $\\mathbf{G}_s$ (regularises the resonance when
      $\\xi_s \\to 0$)

    Resonance ($\\xi_s \\approx 0$, i.e.\\ $E_s \\approx E_t$) gives the
    strongest contribution; off-resonance paths decay as $1/\\xi_s$.

    All spots (kinematical + Umweganregung) are tagged with
    ``'is_umweganregung'`` and extra diagnostic keys.

    Args:
    crystal : xrayutilities Crystal
        Single-crystal object (not a LayeredCrystal stack).
    U : ndarray, shape (3, 3)
        Orientation matrix in the LaueTools lab frame (beam along +x).
    camera : Camera
        Detector geometry.
    E_min_eV, E_max_eV : float
        Energy window (eV).
    source : str
        Synchrotron source model — same options as :func:`simulate_laue`.
    source_kwargs : dict or None
    f2_thresh : float
        Minimum $|F|^2$ threshold for **kinematical** reflections.
    ki_hat : array-like (3,) or None
        Incident beam direction (defaults to :data:`KI_HAT`).
    kb_params : dict or None
        KB mirror reflectivity correction.
    delta_E_rel : float
        Relative energy tolerance for simultaneous excitation of a secondary
        beam $\\mathbf{G}_s$ at the primary energy $E_t$:

        .. math:: |E_s - E_t| / E_t < \\text{delta\\_E\\_rel}

        Default 0.005 (0.5 %).  Larger values find more (weaker) paths but
        increase runtime and may produce spurious spots.
    umweg_f2_thresh : float or None
        Minimum $|A_t|^2$ to retain an Umweganregung spot.
        Defaults to ``1e-4 * f2_thresh``.
    verbose : bool

    Returns:
    spots : list of dict
        All standard keys from :func:`simulate_laue` plus:

        ``'is_umweganregung'`` : bool
            True for spots produced by Umweganregung (forbidden kinematically).
        ``'umweg_paths'`` : list of dict
            For Umweganregung spots: one entry per contributing two-step path,
            containing ``'hkl_s'``, ``'hkl_r'``, ``'F_s'``, ``'F_r'``,
            ``'xi_s'``, ``'delta_s'``.  Empty list for kinematical spots.
        ``'n_umweg_paths'`` : int
            Number of contributing paths (0 for kinematical spots).
        ``'F2_umweg'`` : float
            $|A_t|^2$ for Umweganregung spots; 0 for kinematical spots.
"""
    ki = np.asarray(ki_hat if ki_hat is not None else KI_HAT, dtype=float)
    ki /= np.linalg.norm(ki)
    U = np.asarray(U, dtype=float)
    source_kwargs = source_kwargs or {}
    if umweg_f2_thresh is None:
        umweg_f2_thresh = 1e-4 * f2_thresh

    _spectrum = _make_spectrum_fn(source, source_kwargs, kb_params)

    lam_lo = en2lam(E_max_eV)
    lam_hi = en2lam(E_min_eV)

    # ── Build reciprocal lattice matrix ───────────────────────────────────────
    B = np.column_stack([crystal.Q(1, 0, 0), crystal.Q(0, 1, 0), crystal.Q(0, 0, 1)])
    V_uc = float(crystal.lattice.UnitCellVolume())

    HC_eV_ANG = 12398.4
    G_max = 4.0 * np.pi * E_max_eV / HC_eV_ANG
    G_max_sq = G_max ** 2

    a_star = float(np.linalg.norm(B[:, 0]))
    b_star = float(np.linalg.norm(B[:, 1]))
    c_star = float(np.linalg.norm(B[:, 2]))
    h_lim = int(G_max / a_star) + 1
    k_lim = int(G_max / b_star) + 1
    l_lim = int(G_max / c_star) + 1

    # ── Enumerate all G in sphere ─────────────────────────────────────────────
    hv = np.arange(-h_lim, h_lim + 1, dtype=np.int32)
    kv = np.arange(-k_lim, k_lim + 1, dtype=np.int32)
    lv = np.arange(-l_lim, l_lim + 1, dtype=np.int32)
    H, K, L = np.meshgrid(hv, kv, lv, indexing="ij")
    hkl_all = np.column_stack([H.ravel(), K.ravel(), L.ravel()])
    hkl_all = hkl_all[np.any(hkl_all != 0, axis=1)]

    G_cry_all = (B @ hkl_all.T).T                                   # (N, 3)
    G_sq_all  = np.einsum("ij,ij->i", G_cry_all, G_cry_all)
    in_sphere = G_sq_all <= G_max_sq
    hkl_all   = hkl_all[in_sphere]
    G_cry_all = G_cry_all[in_sphere]

    G_lab_all = G_cry_all @ U.T                                      # (N, 3)
    kdG_all   = G_lab_all @ ki                                       # (N,)

    # Laue energies for all G (NaN where kdG >= 0, i.e. not in Bragg condition)
    G_sq_lab  = np.einsum("ij,ij->i", G_lab_all, G_lab_all)
    with np.errstate(invalid="ignore", divide="ignore"):
        lam_all = np.where(
            kdG_all < 0,
            -4.0 * np.pi * kdG_all / G_sq_lab,
            np.nan,
        )
    E_laue_all = np.where(np.isfinite(lam_all), HC_eV_ANG / lam_all, np.nan)

    # Index map: (h, k, l) → position in hkl_all (for fast G_r lookup)
    _hkl_index: dict[tuple, int] = {
        (int(hkl_all[i, 0]), int(hkl_all[i, 1]), int(hkl_all[i, 2])): i
        for i in range(len(hkl_all))
    }

    # Pre-compute structure factors for all G in sphere at a reference energy.
    # Per-spot energies can vary ±~5% from midpoint; using one reference is a
    # good approximation for the secondary-beam search (full F(E) is used for
    # the final amplitude calculation below).
    E_ref = 0.5 * (E_min_eV + E_max_eV)
    if verbose:
        print(f"  Computing structure factors for {len(hkl_all)} G vectors ...",
              end="", flush=True)
    F_ref_all = crystal.StructureFactorForQ(G_cry_all, en0=E_ref)   # (N,) complex
    F2_ref_all = np.abs(F_ref_all) ** 2
    if verbose:
        print(" done")

    # ── Step 1: kinematical spots (primary reflections on detector) ───────────
    in_energy = (lam_all >= lam_lo) & (lam_all <= lam_hi)
    kin_mask  = in_energy & (F2_ref_all >= f2_thresh)
    kin_idx   = np.where(kin_mask)[0]

    if verbose:
        print(f"  Kinematical candidates: {kin_mask.sum()}  (energy window + F² cut)")

    kin_spots: list[dict] = []
    for idx in kin_idx:
        lam   = float(lam_all[idx])
        E     = HC_eV_ANG / lam
        G_lab = G_lab_all[idx]
        km    = 2.0 * np.pi / lam
        kf    = ki * km + G_lab
        kf_hat = kf / np.linalg.norm(kf)
        pix = camera.project(kf_hat)
        if pix is None:
            continue
        tth = float(np.degrees(np.arccos(np.clip(kf_hat[0], -1.0, 1.0))))
        chi = float(np.degrees(np.arctan2(kf_hat[1], kf_hat[2] + 1e-17)))
        az  = float(np.degrees(np.arctan2(kf_hat[2], kf_hat[1])))
        LP  = lorentz_pol(tth)
        if LP == 0.0:
            continue
        sw = _spectrum(E)
        if sw <= 0.0:
            continue
        F_val = crystal.StructureFactor(G_cry_all[idx], en=E)
        F2    = abs(F_val) ** 2
        if F2 < f2_thresh:
            continue
        hkl = (int(hkl_all[idx, 0]), int(hkl_all[idx, 1]), int(hkl_all[idx, 2]))
        kin_spots.append({
            "hkl":              hkl,
            "is_umweganregung": False,
            "umweg_paths":      [],
            "n_umweg_paths":    0,
            "F2_umweg":         0.0,
            "G_lab":            G_lab.copy(),
            "E":                E,
            "lambda":           lam,
            "tth":              tth,
            "chi":              chi,
            "az":               az,
            "pix":              pix,
            "F2":               F2,
            "LP":               LP,
            "sw":               sw,
            "I_raw":            F2 * LP * sw,
        })

    if verbose:
        print(f"  Kinematical spots on detector: {len(kin_spots)}")

    # Set of pixel positions already claimed by kinematical spots (coarse grid)
    kin_pix_set: set[tuple] = {
        (round(s["pix"][0]), round(s["pix"][1])) for s in kin_spots
    }

    # ── Step 2: Umweganregung — forbidden G_t in energy range ─────────────────
    # Forbidden: |F_t|² < f2_thresh but G_t is in energy window and hits detector
    forbidden_mask = in_energy & (F2_ref_all < f2_thresh)
    forbidden_idx  = np.where(forbidden_mask)[0]

    if verbose:
        print(f"  Forbidden G vectors in energy window: {forbidden_mask.sum()}")
        print(f"  Searching for Umweganregung paths (delta_E_rel={delta_E_rel:.3f}) ...")

    umweg_spots: list[dict] = []
    n_forbidden_on_det = 0

    # Precompute per-G normalised Laue energies for fast tolerance check
    E_laue_finite = np.where(np.isfinite(E_laue_all), E_laue_all, -1.0)

    for idx_t in forbidden_idx:
        lam_t = float(lam_all[idx_t])
        E_t   = HC_eV_ANG / lam_t
        G_lab_t = G_lab_all[idx_t]
        km_t    = 2.0 * np.pi / lam_t
        kf_t    = ki * km_t + G_lab_t
        kf_hat_t = kf_t / np.linalg.norm(kf_t)

        pix_t = camera.project(kf_hat_t)
        if pix_t is None:
            continue
        pix_key_t = (round(pix_t[0]), round(pix_t[1]))
        if pix_key_t in kin_pix_set:
            # Position already occupied by a kinematical spot — skip
            continue

        n_forbidden_on_det += 1

        tth_t = float(np.degrees(np.arccos(np.clip(kf_hat_t[0], -1.0, 1.0))))
        LP_t  = lorentz_pol(tth_t)
        if LP_t == 0.0:
            continue
        sw_t = _spectrum(E_t)
        if sw_t <= 0.0:
            continue

        # ── Find secondary beams G_s with E_s ≈ E_t ──────────────────────────
        # Simultaneous excitation condition: |E_s - E_t| / E_t < delta_E_rel
        near_mask = (
            np.isfinite(E_laue_all)
            & (np.abs(E_laue_finite - E_t) / E_t < delta_E_rel)
        )
        near_idx = np.where(near_mask)[0]

        hkl_t = (int(hkl_all[idx_t, 0]),
                 int(hkl_all[idx_t, 1]),
                 int(hkl_all[idx_t, 2]))

        A_total  = 0.0 + 0j
        paths    = []
        # Normalization constant for this G_t: r_e λ / (π V_uc) makes
        # |A_total|² comparable in units to kinematical |F_uc|²
        norm = _R_E_ANG * lam_t / (np.pi * V_uc)

        for idx_s in near_idx:
            hs, ks, ls = int(hkl_all[idx_s, 0]), int(hkl_all[idx_s, 1]), int(hkl_all[idx_s, 2])

            # G_r must be in our sphere enumeration
            hr, kr, lr = hkl_t[0] - hs, hkl_t[1] - ks, hkl_t[2] - ls
            if (hr, kr, lr) == (0, 0, 0):
                continue
            idx_r = _hkl_index.get((hr, kr, lr))
            if idx_r is None:
                continue

            F_s_ref = F_ref_all[idx_s]
            F_r_ref = F_ref_all[idx_r]
            if abs(F_s_ref) < 1e-6 or abs(F_r_ref) < 1e-6:
                continue

            # Evaluate structure factors at exact energy E_t
            F_s = crystal.StructureFactor(G_cry_all[idx_s], en=E_t)
            F_r = crystal.StructureFactor(G_cry_all[idx_r], en=E_t)
            if abs(F_s) < 1e-8 or abs(F_r) < 1e-8:
                continue

            # Excitation error of G_s at energy E_t (Å⁻¹)
            # ξ_s = k_m(E_t) · (k_hat · G_s_lab) + G_s²/2
            xi_s = float(km_t * float(ki @ G_lab_all[idx_s])
                         + 0.5 * float(G_sq_lab[idx_s]))

            # Darwin half-width δ_s = r_e λ |F_s| / (π V_uc sin θ_s)
            # θ_s is the Bragg angle G_s WOULD have at E_t
            cos_tth_s = float(np.clip(
                (ki * km_t + G_lab_all[idx_s]) @ (ki * km_t + G_lab_all[idx_s]),
                0.0, None,
            ))
            # simpler: use tth_t as proxy for the angular factor (same order)
            sin_th_s  = max(abs(np.sin(np.radians(tth_t / 2.0))), 1e-6)
            delta_s   = (_R_E_ANG * lam_t * abs(F_s)
                         / (np.pi * V_uc * sin_th_s))

            denom = xi_s + 1j * delta_s
            A_path = norm * F_s * F_r / denom
            A_total += A_path

            paths.append({
                "hkl_s":   (hs, ks, ls),
                "hkl_r":   (hr, kr, lr),
                "F_s":     complex(F_s),
                "F_r":     complex(F_r),
                "xi_s":    xi_s,
                "delta_s": delta_s,
            })

        if not paths:
            continue

        F2_umweg = abs(A_total) ** 2
        if F2_umweg < umweg_f2_thresh:
            continue

        chi_t = float(np.degrees(np.arctan2(kf_hat_t[1], kf_hat_t[2] + 1e-17)))
        az_t  = float(np.degrees(np.arctan2(kf_hat_t[2], kf_hat_t[1])))

        umweg_spots.append({
            "hkl":              hkl_t,
            "is_umweganregung": True,
            "umweg_paths":      paths,
            "n_umweg_paths":    len(paths),
            "F2_umweg":         F2_umweg,
            "G_lab":            G_lab_t.copy(),
            "E":                E_t,
            "lambda":           lam_t,
            "tth":              tth_t,
            "chi":              chi_t,
            "az":               az_t,
            "pix":              pix_t,
            "F2":               F2_umweg,
            "LP":               LP_t,
            "sw":               sw_t,
            "I_raw":            F2_umweg * LP_t * sw_t,
        })

    if verbose:
        print(f"  Forbidden G on detector (not obscured): {n_forbidden_on_det}")
        print(f"  Umweganregung spots found: {len(umweg_spots)}")

    # ── Merge and normalise ───────────────────────────────────────────────────
    all_spots = kin_spots + umweg_spots
    if all_spots:
        imax = max(s["I_raw"] for s in all_spots)
        for s in all_spots:
            s["intensity"] = s["I_raw"] / imax

    all_spots.sort(key=lambda s: s["I_raw"], reverse=True)

    if verbose:
        print(f"  Total spots (multi-beam): {len(all_spots)}  "
              f"({len(kin_spots)} kinematical + {len(umweg_spots)} Umweganregung)")

    return all_spots


def simulate_mixed_phases(
    phases,
    camera,
    E_min_eV=5_000,
    E_max_eV=27_000,
    source="bending_magnet",
    source_kwargs=None,
    f2_thresh=None,
    normalise="volume",
    kb_params=BM32_KB,
    structure_model="average",
    sigma_h_mrad=0.0,
    sigma_v_mrad=0.0,
    sigma_beam_h_nm=0.0,
    sigma_beam_v_nm=0.0,
    n_hat_sample=None,
    verbose=True,
    geometry_only=False,
    allowed_hkl=None,
):
    """
    Simulate a Laue pattern from a multi-phase sample with known volume
    fractions.

    Each phase scatters **independently** (incoherent between phases —
    different grains, different orientations).  The contribution of each
    phase is weighted by its volume fraction and unit-cell number density
    before the spot lists are merged into one.

    **Intensity weighting**
    The number of unit cells of phase p contributing to diffraction scales as:

        N_uc_p  ∝  f_p / V_uc_p

    where  f_p  is the volume fraction and  V_uc_p  is the unit-cell volume
    (Å³).  This is the standard Rietveld weight used in powder diffraction
    and is correct for any single-crystal Laue measurement of a multi-phase
    polycrystal.

    For a `LayeredCrystal` phase, V_uc_p is taken as the thickness-weighted
    harmonic mean of the individual layer unit-cell volumes — i.e. the
    effective number of unit cells per unit volume of the stack.

    The `normalise` argument controls how the final intensities are scaled:
      `'volume'`   (default) — weight by f_p / V_uc_p  (physics-correct)
      `'fraction'` — weight by f_p only (ignore V_uc differences)
      `'equal'`    — all phases equally weighted regardless of fraction
      `'none'`     — no rescaling; I_raw values are kept as-is from each
                       phase's simulation

    Args:
    phases : list of dicts or list of tuples
        Each entry describes one phase.  Accepted formats:

        **dict** (recommended)::

            {
              'crystal'         : xu.materials.Crystal  or  LayeredCrystal,
              'U'               : np.ndarray (3×3) orientation matrix,
              'volume_fraction' : float,          # must sum to 1 (normalised)
              'label'           : str,            # optional, default crystal.name
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
        Synchrotron source model, forwarded to each per-phase simulation.
        Accepts all options supported by :func:`simulate_laue_stack`:
        `'bending_magnet'`, `'wiggler'`, `'undulator'`, `'flat'`,
        `'shadow4'`, or `'tabulated'`.

        For `'shadow4'`, the Monte Carlo simulation is run **once** here
        and the resulting spectrum is passed to every phase as
        `'tabulated'`, so the cost is paid only once regardless of how
        many phases are in the mix.

    source_kwargs : dict, optional
        Forwarded to the spectrum function.  For `'shadow4'`, accepts
        `nrays` and `n_energy_bins` (see :func:`_make_spectrum_fn`).
        For `'tabulated'`, must contain `'energy_eV'` and `'flux'`.

    kb_params : dict or None, optional
        KB mirror reflectivity correction, forwarded unchanged to each
        per-phase simulation call.  Defaults to :data:`BM32_KB`.
        Pass `None` to disable, and **must** be `None` for
        `'shadow4'` and `'tabulated'` sources.

    f2_thresh : float | None
        Minimum |F|² threshold (global default, overridable per phase).
        `None` = auto-scale per phase.

    normalise : str
        Weighting mode: `'volume'`, `'fraction'`, `'equal'`, `'none'`.
    structure_model : {'coherent', 'average'}, optional
        Forwarded to :func:`simulate_laue_stack` for any `LayeredCrystal`
        phase.  `'average'` *(default)* enumerates G vectors from buffer
        layers only and uses the composition-weighted average MQW structure
        factor, matching the single-peak-plus-satellites appearance of a
        monochromatic scan.  `'coherent'` enumerates every crystal
        separately and preserves full inter-layer interference.  Ignored for
        plain `xu.materials.Crystal` phases (handled by
        :func:`simulate_laue`).
    verbose : bool

    Returns:
    spots : list of dicts
        Merged, weighted, and renormalised spot list.  Each dict has all the
        standard keys plus:

          `'phase_label'`      – which phase this spot belongs to
          `'volume_fraction'`  – f_p of that phase
          `'phase_weight'`     – the weight applied (f_p / V_uc_p or variant)
          `'intensity'`        – normalised 0–1 over the full mixed pattern
          `'intensity_phase'`  – normalised 0–1 within that phase alone

    Raises:
    ValueError
        If volume fractions do not sum to approximately 1.0 (within ±0.01).

    Example:
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

    Note:
    Orientation relationship between phases does NOT produce interference
    fringes here — use `LayeredCrystal` + `simulate_laue_stack` for that.
    This function is for incoherent multi-grain mixtures (e.g. a polycrystal
    with two phases, or a transformed microstructure).
"""
    import os
    import sys

    import numpy as np
    import xrayutilities as xu

    from .layers import LayeredCrystal

    source_kwargs = source_kwargs or {}

    # ── Pre-compute spectrum for interpolated sources (pay the cost once) ─────
    # For 'shadow4' the Monte Carlo run takes ~1-2 min; running it once per
    # phase would multiply that cost by the number of phases.  Instead, run it
    # here, then switch source to 'tabulated' so every sub-call gets a free
    # interpolation.  The same logic applies when the caller already passes
    # 'tabulated' — we just pass those arrays through unchanged.
    if source in ("shadow4", "tabulated"):
        _fn = _make_spectrum_fn(source, source_kwargs, kb_params)
        # Evaluate on a dense grid covering the energy window so the per-phase
        # simulate_laue / simulate_laue_stack calls get a fine interpolator.
        _e_grid = np.linspace(E_min_eV, E_max_eV, 500)
        _f_grid = np.array([_fn(e) for e in _e_grid])
        source = "tabulated"
        source_kwargs = {"energy_eV": _e_grid, "flux": _f_grid}
        kb_params = None   # already embedded in _f_grid

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

        # Resolve per-phase allowed_hkl: dict keyed by id(crystal) takes priority,
        # then a bare frozenset is used for all phases, then None falls through.
        _phase_allowed = (
            allowed_hkl.get(id(crystal)) if isinstance(allowed_hkl, dict)
            else allowed_hkl
        )

        # Simulate
        if isinstance(crystal, LayeredCrystal):
            spots_p = simulate_laue_stack(
                crystal,
                camera,
                E_min_eV=E_min_eV,
                E_max_eV=E_max_eV,
                source=source,
                source_kwargs=source_kwargs,
                f2_thresh=ph_f2,
                kb_params=kb_params,
                structure_model=structure_model,
                verbose=False,
                geometry_only=geometry_only,
                allowed_hkl=_phase_allowed,
            )
        else:
            spots_p = simulate_laue(
                crystal,
                U,
                camera,
                E_min=E_min_eV,
                E_max=E_max_eV,
                source=source,
                source_kwargs=source_kwargs,
                f2_thresh=(ph_f2 if ph_f2 is not None else F2_THRESHOLD),
                kb_params=kb_params,
                geometry_only=geometry_only,
                allowed_hkl=_phase_allowed,
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
    beam_divergence_ellipses(
        all_spots, camera, sigma_h_mrad, sigma_v_mrad,
        sigma_beam_h_nm=sigma_beam_h_nm,
        sigma_beam_v_nm=sigma_beam_v_nm,
        n_hat_sample=n_hat_sample,
    )

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
# BEAM DIVERGENCE BROADENING
# ─────────────────────────────────────────────────────────────────────────────


def beam_divergence_ellipses(
    spots: list,
    camera,
    sigma_h_mrad: float = 0.0,
    sigma_v_mrad: float = 0.0,
    ki_hat=None,
    sigma_beam_h_nm: float = 0.0,
    sigma_beam_v_nm: float = 0.0,
    n_hat_sample=None,
) -> list:
    """
    Add per-spot detector broadening from beam angular divergence and geometric
    footprint elongation due to a tilted sample surface.

    Two independent broadening mechanisms are modelled and combined:

    1. **Angular divergence** — the incident beam has an angular spread
       (σ_h, σ_v).  Each k̂_i direction satisfying the Laue condition maps
       the G-vector to a slightly different pixel.  The pixel-space Jacobian
       J_div = ∂pix/∂(δ_h, δ_v) is estimated by central differences on the
       perturbed beam direction.

    2. **Geometric footprint** — the beam illuminates a finite area on the
       sample surface.  When the surface is tilted relative to the beam (as
       in reflection geometry), the footprint is elongated by 1/sin(α_inc).
       Each illuminated point scatters from a displaced origin, shifting the
       pixel hit via geometric parallax.  The sample-plane Jacobian is
       computed analytically:

           δr_sample = δr_beam − k̂_i (δr_beam · n̂_sample) / (k̂_i · n̂_sample)

       and the resulting pixel shift is evaluated by central differences on
       the camera ray-intersection formula for a displaced source.

    The two pixel covariances add:

        C_total = C_divergence + C_footprint

    A second Jacobian  J_pa = ∂(2θ, χ)/∂pix  maps the combined result into
    angle space.  Both representations are stored in each spot dict.

    **New keys written to every spot**
    `cov_px`              (2, 2) ndarray  total pixel covariance (px²)
    `sigma_major_px`      float           semi-major axis, 1σ (px)
    `sigma_minor_px`      float           semi-minor axis, 1σ (px)
    `ellipse_angle_px_deg`float           major-axis angle, CCW from +x (°)
    `cov_ang`             (2, 2) ndarray  angle covariance in (2θ, χ) (deg²)
    `sigma_major_ang_deg` float           semi-major axis, 1σ (°)
    `sigma_minor_ang_deg` float           semi-minor axis, 1σ (°)
    `ellipse_angle_ang_deg`float          major-axis angle in (2θ, χ) space (°)
    `sigma_tth_deg`       float           marginal 1σ along 2θ (°)
    `sigma_chi_deg`       float           marginal 1σ along χ  (°)

    Args:
    spots         : list of dicts from :func:`simulate_laue_stack` etc.
    camera        : Camera
    sigma_h_mrad  : float  horizontal beam divergence 1σ (mrad).
                    Typical BM32/ESRF: 2–3 mrad.
    sigma_v_mrad  : float  vertical beam divergence 1σ (mrad).
                    Typical BM32/ESRF: 0.2–0.5 mrad.
    ki_hat        : array-like (3,), optional
                    Incident beam direction (LT frame, x // beam).
                    Default: `[1, 0, 0]`.
    sigma_beam_h_nm : float  horizontal (in-plane) beam size 1σ at the sample
                    (nm).  Footprint broadening requires `n_hat_sample`.
    sigma_beam_v_nm : float  vertical beam size 1σ at the sample (nm).
    n_hat_sample  : array-like (3,), optional
                    Unit normal to the sample surface in the LT frame.
                    Required to activate footprint broadening.  For a sample
                    with its surface normal pointing toward the detector (
                    typical reflection geometry with ~45° sample tilt) use
                    e.g. `[0, 0, 1]` for a horizontal flat sample or the
                    crystal normal obtained from the orientation matrix.

    Returns:
    spots  (modified in-place and returned for chaining)

    Note:
    The divergence Jacobian is evaluated with a 0.5 mrad central-difference
    step; the footprint Jacobian with a 10 µm step.  Spots whose G vector
    does not satisfy the Laue condition after perturbation are assigned zero
    broadening for that direction.
"""
    _zero2 = np.zeros((2, 2))
    _zero_keys = {
        "cov_px": _zero2, "sigma_major_px": 0.0, "sigma_minor_px": 0.0,
        "ellipse_angle_px_deg": 0.0,
        "cov_ang": _zero2, "sigma_major_ang_deg": 0.0, "sigma_minor_ang_deg": 0.0,
        "ellipse_angle_ang_deg": 0.0, "sigma_tth_deg": 0.0, "sigma_chi_deg": 0.0,
    }

    _no_divergence = sigma_h_mrad <= 0.0 and sigma_v_mrad <= 0.0
    _no_footprint = (
        (sigma_beam_h_nm <= 0.0 and sigma_beam_v_nm <= 0.0)
        or n_hat_sample is None
    )
    if _no_divergence and _no_footprint:
        for s in spots:
            s.update(_zero_keys)
        return spots

    ki = np.asarray(ki_hat if ki_hat is not None else KI_HAT, dtype=float)
    ki /= np.linalg.norm(ki)

    # Horizontal and vertical directions perpendicular to ki.
    # For a beam along x:  ê_h ≈ ŷ (horizontal),  ê_v ≈ ẑ (vertical).
    z_hat = np.array([0.0, 0.0, 1.0])
    cross_zk = np.cross(z_hat, ki)
    if np.linalg.norm(cross_zk) > 1e-6:
        e_h = cross_zk / np.linalg.norm(cross_zk)
    else:
        e_h = np.array([0.0, 1.0, 0.0])
    e_v = np.cross(ki, e_h)
    e_v /= np.linalg.norm(e_v)

    sigma_h = sigma_h_mrad * 1e-3  # rad
    sigma_v = sigma_v_mrad * 1e-3  # rad
    _H = 5e-4  # 0.5 mrad central-difference step (rad)

    # ── Footprint: directions in the sample plane per mm of beam offset ────────
    # A point at δr_beam = δh·ê_h + δv·ê_v in the beam cross-section hits the
    # sample at δr_sample = δr_beam − k̂_i·(δr_beam·n̂)/(k̂_i·n̂), giving a
    # 3-vector footprint direction per mm of beam displacement.
    do_footprint = not _no_footprint
    if do_footprint:
        n_s = np.asarray(n_hat_sample, dtype=float)
        n_s = n_s / np.linalg.norm(n_s)
        ki_dot_n = float(np.dot(ki, n_s))
        if abs(ki_dot_n) < 1e-3:  # beam nearly parallel to surface
            do_footprint = False
        else:
            drs_dh = e_h - ki * float(np.dot(e_h, n_s)) / ki_dot_n  # (mm/mm)
            drs_dv = e_v - ki * float(np.dot(e_v, n_s)) / ki_dot_n
            _S = 0.01  # 10 µm central-difference step (mm)

    def _perturbed_pix(G, e, delta):
        ki_p = ki + delta * e
        ki_p /= np.linalg.norm(ki_p)
        Gm2 = float(np.dot(G, G))
        kdG = float(np.dot(ki_p, G))
        if kdG >= 0:
            return None
        lam = -4.0 * np.pi * kdG / Gm2
        kf = (2.0 * np.pi / lam) * ki_p + G
        return camera.project(kf / np.linalg.norm(kf))

    def _project_from_source(kf_hat_lt, src_lt):
        """Project kf_hat onto the detector for a source displaced from origin."""
        # LT → LT2: x_LT2 = -y_LT,  y_LT2 = x_LT,  z_LT2 = z_LT
        kf_lt2 = np.array([-kf_hat_lt[1], kf_hat_lt[0], kf_hat_lt[2]])
        src_lt2 = np.array([-src_lt[1], src_lt[0], src_lt[2]])
        scal = float(np.dot(kf_lt2, camera.normal))
        if scal < 1e-8:
            return None
        dd_eff = camera.dd - float(np.dot(src_lt2, camera.normal))
        IM = src_lt2 + kf_lt2 * (dd_eff / scal)
        OM = IM - camera.IOlab
        xca0 = float(OM[0])
        yca0 = (
            float(OM[1]) / camera._sinbeta
            if abs(camera._sinbeta) > 1e-8
            else -float(OM[2]) / camera._cosbeta
        )
        xcam1 = camera._cosgam * xca0 + camera._singam * yca0
        ycam1 = -camera._singam * xca0 + camera._cosgam * yca0
        return np.array([
            camera.xcen + xcam1 / camera.pixel_mm,
            camera.ycen + ycam1 / camera.pixel_mm,
        ])

    def _pix_to_angles(x, y):
        ufs = camera.pixel_to_kf(np.array([x]), np.array([y]))
        tth = float(np.degrees(np.arccos(np.clip(ufs[0, 0], -1.0, 1.0))))
        chi = float(np.degrees(np.arctan2(float(ufs[0, 1]), float(ufs[0, 2]) + 1e-17)))
        return tth, chi

    def _ellipse_params(cov):
        """Return (sigma_major, sigma_minor, angle_deg) from a 2×2 covariance."""
        try:
            eigvals, eigvecs = np.linalg.eigh(cov)
            eigvals = np.maximum(eigvals, 0.0)
            idx = int(np.argmax(eigvals))
            sig_maj = float(np.sqrt(eigvals[idx]))
            sig_min = float(np.sqrt(eigvals[1 - idx]))
            v = eigvecs[:, idx]
            ang = float(np.degrees(np.arctan2(v[1], v[0])))
            return sig_maj, sig_min, ang
        except Exception:
            return 0.0, 0.0, 0.0

    for s in spots:
        G = np.asarray(s["G_lab"], dtype=float)
        pix0 = np.asarray(s["pix"], dtype=float)

        # ── Beam-divergence pixel Jacobian ────────────────────────────────
        J_div = np.zeros((2, 2))
        if not _no_divergence:
            for col, e in enumerate([e_h, e_v]):
                pp = _perturbed_pix(G, e, +_H)
                pm = _perturbed_pix(G, e, -_H)
                if pp is not None and pm is not None:
                    J_div[:, col] = (np.asarray(pp) - np.asarray(pm)) / (2.0 * _H)

        D_beam = np.diag([sigma_h ** 2, sigma_v ** 2])
        cov_div = J_div @ D_beam @ J_div.T

        # ── Footprint pixel Jacobian ──────────────────────────────────────
        # k̂_f is the same for all source positions (same G, same λ).
        cov_fp = _zero2.copy()
        if do_footprint:
            lam_s = float(s["lam"])
            kf_vec = (2.0 * np.pi / lam_s) * ki + G
            kf_hat_s = kf_vec / np.linalg.norm(kf_vec)
            J_fp = np.zeros((2, 2))
            for col, d_src in enumerate([drs_dh, drs_dv]):
                pp = _project_from_source(kf_hat_s, +_S * d_src)
                pm = _project_from_source(kf_hat_s, -_S * d_src)
                if pp is not None and pm is not None:
                    J_fp[:, col] = (pp - pm) / (2.0 * _S)
            D_fp = np.diag([(sigma_beam_h_nm * 1e-6) ** 2, (sigma_beam_v_nm * 1e-6) ** 2])
            cov_fp = J_fp @ D_fp @ J_fp.T

        # ── Combined pixel covariance ─────────────────────────────────────
        cov_px = cov_div + cov_fp
        s["cov_px"] = cov_px
        s["sigma_major_px"], s["sigma_minor_px"], s["ellipse_angle_px_deg"] = (
            _ellipse_params(cov_px)
        )

        # ── Pixel→angle Jacobian ──────────────────────────────────────────
        try:
            _dp = 1.0  # 1-pixel step
            J_pa = np.zeros((2, 2))
            tp, cp = _pix_to_angles(pix0[0] + _dp, pix0[1])
            tm, cm = _pix_to_angles(pix0[0] - _dp, pix0[1])
            J_pa[0, 0] = (tp - tm) / (2 * _dp)   # ∂tth/∂x
            J_pa[1, 0] = (cp - cm) / (2 * _dp)   # ∂chi/∂x
            tp, cp = _pix_to_angles(pix0[0], pix0[1] + _dp)
            tm, cm = _pix_to_angles(pix0[0], pix0[1] - _dp)
            J_pa[0, 1] = (tp - tm) / (2 * _dp)   # ∂tth/∂y
            J_pa[1, 1] = (cp - cm) / (2 * _dp)   # ∂chi/∂y
            cov_ang = J_pa @ cov_px @ J_pa.T
        except Exception:
            cov_ang = _zero2.copy()

        s["cov_ang"] = cov_ang
        s["sigma_major_ang_deg"], s["sigma_minor_ang_deg"], s["ellipse_angle_ang_deg"] = (
            _ellipse_params(cov_ang)
        )
        s["sigma_tth_deg"] = float(np.sqrt(max(cov_ang[0, 0], 0.0)))
        s["sigma_chi_deg"] = float(np.sqrt(max(cov_ang[1, 1], 0.0)))

    return spots


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

    Args:
    spots : list of dicts
        Output of `simulate_laue_stack()`.  Must contain `'G_lab'` and
        `'E'` keys (present by default).
    stack : LayeredCrystal

    Returns:
    spots : the same list, each dict extended with:
        `'layer_F'`      : dict  { label : complex amplitude F_l }
        `'layer_I'`      : dict  { label : float  absolute intensity I_l }
        `'layer_I_frac'` : dict  { label : float  fraction 0-1 (sums to 1) }

    Note:
    Negative fractions are physically meaningful — they indicate a layer
    that **destructively interferes** with the rest of the stack at that Q.
    The sum over all layers is still exactly 1.

    Example:
    >>> spots = simulate_laue_stack(stack, camera)
    >>> spots = layer_contributions_spots(spots, stack)
    >>> for s in spots[:5]:
    ...     print(s['hkl'], s['layer_I_frac'])
"""
    stack = _flatten_if_multiblock(stack)
    stack._update_offsets()
    Lambda = stack._bilayer_thickness
    n_rep = stack.n_rep
    # Hoisted out of the spot loop: `.layers` is a property that rebuilds its
    # list on every access, and a flattened (unrolled) multi-block stack often
    # repeats the same (crystal, U, thickness) many times.
    layers_z0 = list(zip(stack.layers, stack._z_offsets))
    labels = [layer.label for layer in stack.layers]

    for s in spots:
        Q = np.asarray(s["G_lab"], dtype=float)
        E = float(s["E"])
        Qz = Q[2]

        # Geometric repetition factor S_rep
        phi_rep = Qz * Lambda
        phi_mod = phi_rep % (2.0 * np.pi)
        if abs(phi_mod) < 1e-10 or abs(phi_mod - 2 * np.pi) < 1e-10:
            S_rep = float(n_rep) + 0j
        else:
            S_rep = (1.0 - np.exp(1j * n_rep * phi_rep)) / (
                1.0 - np.exp(1j * phi_rep)
            )

        # Per-layer amplitude (including S_rep).  structure_factor(Q, E, z0)
        # factors exactly as F0(Q, E) * exp(i * Qn * z0), so layer instances
        # sharing the same (crystal, U, thickness) -- e.g. repeated bilayers
        # from an unrolled multi-block stack -- can reuse one F0/Qn evaluation
        # instead of recomputing the (relatively expensive) crystal structure
        # factor for every repetition.
        f0_cache = {}
        layer_F = {}
        for layer, z0 in layers_z0:
            key = (id(layer.crystal), layer.U.tobytes(), layer.thickness)
            cached = f0_cache.get(key)
            if cached is None:
                F0 = layer.structure_factor(Q, E, z0=0.0)
                Qn = float(np.dot(Q, layer.n_hat))
                cached = (F0, Qn)
                f0_cache[key] = cached
            F0, Qn = cached
            F_l = F0 * np.exp(1j * Qn * z0) * S_rep
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
    Requires `layer_contributions_spots()` to have been called first.
"""
    if not spots or "layer_I_frac" not in spots[0]:
        raise ValueError("Call layer_contributions_spots(spots, stack) first.")

    labels = list(spots[0]["layer_I_frac"].keys())
    col_w = max(12, max(len(l) for l in labels))

    header = (
        f"  {'phase':12s} {'hkl':^10} {'order':>6} {'satellite':>10} "
        f"{'E(keV)':>7} {'2th':>6} {'I_raw':>11} {'I/Imax':>7}  "
        + "  ".join(f"{l[:col_w]:>{col_w}}" for l in labels)
    )
    print(f"\n  Per-layer intensity fractions  (top {n} spots)")
    print("  " + "─" * len(header))
    print(header)
    print("  " + "─" * len(header))

    for s in spots[:n]:
        h, k, l = s["hkl"]
        order = s.get("satellite_order", 0)
        sat_tag = f"m={order:+d}" if order != 0 else "—"
        fracs = "  ".join(
            f"{s['layer_I_frac'].get(lbl, 0.)*100:>{col_w}.1f}%" for lbl in labels
        )
        print(
            f"  {s['phase_label']:12s} ({h:+d}{k:+d}{l:+d})  "
            f"{order:>6d} {sat_tag:>10s} "
            f"{s['E']/1e3:7.3f} {s['tth']:6.1f} "
            f"{s['I_raw']:11.3e} {s['intensity']:7.4f}  {fracs}"
        )


def print_mixed_summary(spots, top_n=20):
    """
    Print a summary table of the strongest spots in a mixed-phase pattern,
    grouped by phase.

    Args:
    spots : list of dicts from `simulate_mixed_phases()`
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
            f"{'I_raw':>11} {'I/Imax':>8} {'I_phase':>8}  type"
        )
        print("  " + "─" * 82)
        top = sorted(phase_spots, key=lambda s: s["intensity"], reverse=True)
        for s in top[:top_n]:
            h, k, l = s["hkl"]
            print(
                f"  ({h:+d}{k:+d}{l:+d})  "
                f"{s['E']/1e3:7.3f} {s['tth']:7.2f} {s['chi']:7.2f} "
                f"{s['I_raw']:11.3e} {s['intensity']:8.4f} {s.get('intensity_phase',0):8.4f}"
            )


def print_spot_table(title, spots, n=15):
    print(f"\n  ── {title} ──  ({len(spots)} total spots on camera)")
    print(
        f"  {'hkl':^10} {'E(keV)':>7} {'lambda(A)':>9} {'2th(deg)':>9} "
        f"{'az(deg)':>8} {'col':>9} {'row':>9} "
        f"{'|F|^2':>8} {'LP':>7} {'S(E)':>7} {'I_raw':>11} {'I/Imax':>7}  satellite"
    )
    print("  " + "-" * 130)
    for s in spots[:n]:
        h, k, l = s["hkl"]
        c, r = s["pix"]
        order = s.get("satellite_order", 0)
        sat_col = f"m={order:+d}" if order != 0 else "—"
        print(
            f"  ({h:+2d}{k:+2d}{l:+2d})  "
            f"{s['E']/1e3:7.3f}  {s['lambda']:9.5f}  "
            f"{s['tth']:9.3f}  {s['az']:8.2f}  "
            f"{c:9.3f}  {r:9.3f}  "
            f"{s['F2']:8.2f}  {s['LP']:7.4f}  "
            f"{s['sw']:7.4f}  {s['I_raw']:11.3e}  {s['intensity']:7.4f}  {sat_col:>9s}"
        )


def print_hkl_family(spots: list, h: int, k: int, l: int, n: int = 5) -> None:
    """
    Print all spots whose Miller indices are integer multiples of (h, k, l).

    Searches for spots with hkl = m*(h, k, l) for m = 1, 2, …, n, covering
    both positive and negative orders (m and -m).

    Args:
    spots : list[dict]
        Spot list from any `simulate_laue*` function.
    h, k, l : int
        Base Miller indices of the family.
    n : int
        Highest multiple to include (default 5).  Checks ±1·hkl … ±n·hkl.
"""
    h0, k0, l0 = int(h), int(k), int(l)
    targets = set()
    for m in range(1, n + 1):
        targets.add(( m * h0,  m * k0,  m * l0))
        targets.add((-m * h0, -m * k0, -m * l0))

    matches = [s for s in spots if tuple(int(x) for x in s["hkl"]) in targets]
    matches.sort(key=lambda s: (
        abs(round(s["hkl"][0] / h0)) if h0 else
        abs(round(s["hkl"][1] / k0)) if k0 else
        abs(round(s["hkl"][2] / l0)),
        s["tth"],
    ))

    title = f"({h0:+d}{k0:+d}{l0:+d}) family  [multiples 1 … {n}]"
    print(f"\n  ── {title} ──  ({len(matches)} spots found)")
    if not matches:
        return

    print(
        f"  {'hkl':^10} {'E(keV)':>7} {'lambda(A)':>9} {'2th(deg)':>9} "
        f"{'az(deg)':>8} {'col':>6} {'row':>6} "
        f"{'|F|^2':>8} {'LP':>7} {'S(E)':>7} {'I_raw':>11} {'I/Imax':>7}  satellite"
    )
    print("  " + "-" * 124)
    for s in matches:
        hh, kk, ll = s["hkl"]
        c, r = s["pix"]
        order = s.get("satellite_order", 0)
        sat_col = f"m={order:+d}" if order != 0 else "—"
        print(
            f"  ({hh:+2d}{kk:+2d}{ll:+2d})  "
            f"{s['E']/1e3:7.3f}  {s['lambda']:9.5f}  "
            f"{s['tth']:9.3f}  {s['az']:8.2f}  "
            f"{c:6.0f}  {r:6.0f}  "
            f"{s['F2']:8.2f}  {s['LP']:7.4f}  "
            f"{s['sw']:7.4f}  {s['I_raw']:11.3e}  {s['intensity']:7.4f}  {sat_col:>9s}"
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


# ── Symmetry / fundamental-zone utilities ─────────────────────────────────────


_SYMMETRY_OPS_CACHE: "dict[str, np.ndarray]" = {}


def _symmetry_ops_np(symmetry: str) -> np.ndarray:
    """
    Return proper rotations of the crystal point group as an (N, 3, 3) array.

    Pure-numpy implementation; no orix dependency.
    Supported symmetries and operator counts:
    ``'cubic'`` (24), ``'hexagonal'`` (12), ``'tetragonal'`` (8),
    ``'orthorhombic'`` (4).
    """
    if not isinstance(symmetry, str):
        raise TypeError(
            f"symmetry must be a string "
            f"('cubic', 'hexagonal', 'tetragonal', 'orthorhombic'), "
            f"got {type(symmetry).__name__!r}"
        )
    if symmetry in _SYMMETRY_OPS_CACHE:
        return _SYMMETRY_OPS_CACHE[symmetry]

    from itertools import permutations, product as _iproduct

    if symmetry == "cubic":
        ops = []
        for perm in permutations(range(3)):
            for signs in _iproduct((-1, 1), repeat=3):
                R = np.zeros((3, 3))
                for j in range(3):
                    R[perm[j], j] = signs[j]
                if round(np.linalg.det(R)) == 1:
                    ops.append(R)
        result = np.array(ops)   # (24, 3, 3)

    elif symmetry == "hexagonal":
        ops = []
        for n in range(6):
            a = n * np.pi / 3
            c, s = np.cos(a), np.sin(a)
            ops.append(np.array([[ c, -s, 0.], [ s,  c, 0.], [0., 0.,  1.]]))
        for n in range(6):
            a = n * np.pi / 6
            c, s = np.cos(2 * a), np.sin(2 * a)
            ops.append(np.array([[ c,  s, 0.], [ s, -c, 0.], [0., 0., -1.]]))
        result = np.array(ops)   # (12, 3, 3)

    elif symmetry == "tetragonal":
        ops = []
        for n in range(4):
            a = n * np.pi / 2
            c, s = np.cos(a), np.sin(a)
            ops.append(np.array([[ c, -s, 0.], [ s,  c, 0.], [0., 0.,  1.]]))
        for n in range(4):
            a = n * np.pi / 2
            c, s = np.cos(a), np.sin(a)
            ops.append(np.array([[ c,  s, 0.], [ s, -c, 0.], [0., 0., -1.]]))
        result = np.array(ops)   # (8, 3, 3)

    elif symmetry == "orthorhombic":
        result = np.array([
            np.eye(3),
            np.diag([1., -1., -1.]),
            np.diag([-1., 1., -1.]),
            np.diag([-1., -1., 1.]),
        ])   # (4, 3, 3)

    else:
        raise ValueError(
            f"symmetry must be one of "
            f"['cubic', 'hexagonal', 'tetragonal', 'orthorhombic'], "
            f"got {symmetry!r}"
        )

    _SYMMETRY_OPS_CACHE[symmetry] = result
    return result


def map_to_fundamental_zone(
    U: "np.ndarray",
    symmetry: str = "cubic",
) -> "np.ndarray":
    """
    Map orientation matrix/matrices to the fundamental zone.

    For each input rotation the symmetry-equivalent member with the smallest
    rotation angle from the identity is selected.  Only **proper rotations**
    (det = +1) from the crystal point group are considered, so every output
    matrix is a valid rotation matrix.

    The equivalent is chosen by right-multiplication:
    ``U_fz = U @ ops[s*]``  where  ``s* = argmax trace(U @ ops[s])``.
    Maximising the trace is equivalent to minimising the geodesic angle to
    the identity, consistent with the convention used in
    :func:`disorientation`.

    Args:
        U ((3, 3) or (..., 3, 3) ndarray): Rotation matrix/matrices.
            Any number of leading batch dimensions is supported.
        symmetry (str): Crystal point-group symmetry.  One of ``'cubic'``
            (24 proper rotations), ``'hexagonal'`` (12), ``'tetragonal'``
            (8), ``'orthorhombic'`` (4).

    Returns:
        U_fz (same shape as *U*, float64): Orientation matrices reduced to
            the fundamental zone.

    Example::

        U_fz = map_to_fundamental_zone(U_raw, symmetry='cubic')
    """
    ops = _symmetry_ops_np(symmetry)   # (N_sym, 3, 3)  proper rotations only

    U = np.asarray(U, dtype=float)
    shape = U.shape
    U_flat = U.reshape(-1, 3, 3)      # (M, 3, 3)

    # Project to nearest SO(3) element via SVD so that non-orthogonal inputs
    # (e.g. U_eff = U @ (I + ε) from strain fits) are handled correctly.
    U_flat = Rotation.from_matrix(U_flat).as_matrix()

    # All equivalents: U_equiv[s, m] = U_flat[m] @ ops[s]
    # U_flat[None]: (1, M, 3, 3)  ops[:, None]: (N_sym, 1, 3, 3)
    U_equiv = U_flat[None] @ ops[:, None]               # (N_sym, M, 3, 3)

    # Trace of each equivalent — maximise ↔ minimise angle to identity
    traces = U_equiv[:, :, 0, 0] + U_equiv[:, :, 1, 1] + U_equiv[:, :, 2, 2]  # (N_sym, M)
    best   = np.argmax(traces, axis=0)                  # (M,)

    # Gather: U_fz[m] = U_flat[m] @ ops[best[m]]
    ops_sel = ops[best]                                 # (M, 3, 3)
    U_fz    = np.einsum("...ij,...jk->...ik", U_flat, ops_sel)  # (M, 3, 3)
    return U_fz.reshape(shape)


def disorientation(
    U1: "np.ndarray",
    U2: "np.ndarray",
    symmetry: str = "cubic",
) -> "tuple[float, np.ndarray]":
    """
    Disorientation between two orientation matrices with crystal symmetry.

    The disorientation is the symmetry-equivalent misorientation
    $\\mathbf{M} = \\mathbf{U}_2 \\mathbf{U}_1^T$ with the smallest rotation
    angle, minimised over all pairs of point-group operators:

    $$
    \\omega_\\text{dis} =
        \\min_{S_i,\\,S_j \\in G}\\;
        \\bigl|\\!\\operatorname{angle}(S_i\\,\\mathbf{M}\\,S_j^T)\\bigr|
    $$

    Vectorised over all pairs of point-group operators using pure numpy —
    no dependency on orix internals.

    Args:
        U1 ((3, 3) ndarray): First orientation matrix.
        U2 ((3, 3) ndarray): Second orientation matrix.
        symmetry (str): Crystal point-group symmetry — same options as
            :func:`map_to_fundamental_zone`.

    Returns:
        angle_deg (float): Disorientation angle in degrees.
        R_dis ((3, 3) ndarray): The disorientation rotation matrix
            (minimum-angle symmetry-equivalent misorientation).

    Example::

        angle, R = disorientation(U1, U2, symmetry='cubic')
        print(f"disorientation = {angle:.3f}°")
    """
    ops = _symmetry_ops_np(symmetry)                          # (N, 3, 3)

    # Project both inputs to SO(3) so non-orthogonal matrices such as
    # U_eff = U @ (I + ε) from strain fits give correct results.
    R1 = Rotation.from_matrix(np.asarray(U1, dtype=float)).as_matrix()
    R2 = Rotation.from_matrix(np.asarray(U2, dtype=float)).as_matrix()
    R_mis = R2 @ R1.T

    # Candidates: S_i @ R_mis @ S_j^T  for all (i, j) pairs → (N*N, 3, 3)
    ops_R = ops @ R_mis                     # (N, 3, 3)  S_i @ R_mis
    ops_T = ops.transpose(0, 2, 1)         # (N, 3, 3)  S_j^T
    candidates = (ops_R[:, None] @ ops_T[None]).reshape(-1, 3, 3)

    traces = candidates[:, 0, 0] + candidates[:, 1, 1] + candidates[:, 2, 2]
    angles = np.arccos(np.clip((traces - 1.0) / 2.0, -1.0, 1.0))

    best = int(np.argmin(angles))
    return float(np.degrees(angles[best])), candidates[best]


# ─────────────────────────────────────────────────────────────────────────────
# DEPTH-SCAN RECONSTRUCTION
# ─────────────────────────────────────────────────────────────────────────────

def depth_scan_reconstruction(
    spots: list[dict],
    peaklist: "np.ndarray",
    camera,
    stack,
    ki_hat=None,
    *,
    n_steps: int = 100,
    z_min_mm: float = 0.0,
    z_max_mm: "float | None" = None,
    tolerance_px: float = 5.0,
    min_intensity: float = 0.01,
    score_weighted: bool = True,
) -> dict:
    """
    Poor man's depth-resolved Laue reconstruction by parallax scanning.

    For each candidate depth *z* below the surface, every simulated spot is
    projected to its expected detector position using the depth-parallax
    model.  A **hit-or-miss** score counts how many projections land within
    *tolerance_px* of a measured peak.  The score profile over *z* reveals
    which depths are actively diffracting.

    Additionally, for each measured peak the depth of origin is estimated by
    linearly inverting the parallax model against the best-matching simulated
    spot — giving a per-peak depth assignment without scanning.

    The projection is **exactly linear** in depth (``camera.project`` uses a
    linear formula), so only two calls per spot are needed to precompute the
    exact slope; subsequent depth evaluations are pure array operations.

    Args:
        spots: Simulated spot list from :func:`simulate_laue_darwin` or
            :func:`simulate_laue_stack`.
        peaklist: ``(N, ≥2)`` array ``[col, row, ...]`` as returned by
            :func:`convert_spotsfile2peaklist`.
        camera: :class:`Camera` used in the simulation.
        stack: :class:`LayeredCrystal` used in the simulation.  Provides the
            surface normal and total thickness.
        ki_hat: Incident beam direction (3-vector, LT frame).  Defaults to
            ``[1, 0, 0]``.
        n_steps: Number of depth values to scan.
        z_min_mm: Minimum physical depth (mm, along surface normal).
        z_max_mm: Maximum physical depth (mm).  Defaults to total stack
            thickness.
        tolerance_px: Matching radius in pixels for hit-or-miss counting and
            per-peak depth inversion.
        min_intensity: Skip spots below this normalised intensity.
        score_weighted: If ``True`` each hit is weighted by the spot's
            normalised intensity; if ``False`` every hit counts as 1.

    Returns:
        dict with keys:

        ``'z_mm'``
            ndarray (n_steps,) — physical depth values scanned (mm).
        ``'score'``
            ndarray (n_steps,) — total hit score at each depth.
        ``'score_per_phase'``
            dict ``{phase_label: ndarray(n_steps,)}`` — score broken out by
            phase.
        ``'z_est_per_peak'``
            ndarray (N_peaks,) — estimated depth of origin for each measured
            peak (NaN if no simulated spot matched within *tolerance_px*).
        ``'phase_per_peak'``
            list (N_peaks,) — phase label of the best-matching simulated spot
            for each peak (``None`` if unmatched).
        ``'spot_idx_per_peak'``
            list (N_peaks,) — index into the filtered spot list for each peak.
        ``'cos_in'``
            float — ``|k̂_i · n̂|``, used to convert beam-path depth to
            physical depth.
    """
    from scipy.spatial import cKDTree

    stack = _flatten_if_multiblock(stack)
    ki = np.asarray(ki_hat if ki_hat is not None else KI_HAT, dtype=float)
    ki = ki / np.linalg.norm(ki)
    n_hat = np.asarray(stack.n_hat, dtype=float)
    n_hat = n_hat / np.linalg.norm(n_hat)
    cos_in = float(abs(np.dot(ki, n_hat)))
    if cos_in < 1e-6:
        raise ValueError("ki is nearly parallel to the sample surface — cos_in ≈ 0")

    # Total stack thickness in mm (Å → mm)
    total_mm = sum(
        lyr.thickness for lyr in (stack.layers * stack.n_rep + stack.buffer_layers)
    ) * 1e-7
    if z_max_mm is None:
        z_max_mm = total_mm

    z_mm = np.linspace(z_min_mm, z_max_mm, n_steps)

    # ── Filter spots ──────────────────────────────────────────────────────────
    filtered = [
        s for s in spots
        if s.get("pix") is not None and float(s.get("intensity", 0)) >= min_intensity
    ]
    if not filtered:
        raise ValueError("No spots pass the min_intensity filter.")

    phases = list(dict.fromkeys(s.get("phase_label", "unknown") for s in filtered))

    # ── Precompute surface positions and parallax slopes ──────────────────────
    # Projection is exact-linear in source_depth_mm, so two calls suffice.
    _EPS = 0.01  # mm — finite-difference step (along surface normal)
    p0_list: list[np.ndarray] = []
    slope_list: list[np.ndarray] = []
    valid_mask: list[bool] = []

    for s in filtered:
        tth = float(np.radians(s["tth"]))
        chi = float(np.radians(s["chi"]))
        kf = np.array([np.cos(tth),
                        np.sin(tth) * np.sin(chi),
                        np.sin(tth) * np.cos(chi)])
        p0 = camera.project(kf, source_depth_mm=0.0)
        p1 = camera.project(kf, source_depth_mm=_EPS / cos_in)
        if p0 is None or p1 is None:
            valid_mask.append(False)
            p0_list.append(np.array([np.nan, np.nan]))
            slope_list.append(np.array([0.0, 0.0]))
        else:
            valid_mask.append(True)
            p0_arr = np.array(p0, dtype=float)
            slope = (np.array(p1, dtype=float) - p0_arr) / _EPS
            p0_list.append(p0_arr)
            slope_list.append(slope)

    valid = np.array(valid_mask)
    p0_arr    = np.array(p0_list)     # (n_spots, 2)
    slope_arr = np.array(slope_list)  # (n_spots, 2)  [px / mm of physical depth]
    w_arr = np.array([float(s.get("intensity", 1.0)) for s in filtered])

    phase_arr = np.array([s.get("phase_label", "unknown") for s in filtered])

    # ── Build peaklist KDTree ─────────────────────────────────────────────────
    pl = np.asarray(peaklist, dtype=float)
    meas_xy = pl[:, :2]
    tree = cKDTree(meas_xy)

    # ── Depth scan — hit-or-miss scoring ──────────────────────────────────────
    score       = np.zeros(n_steps)
    score_phase = {ph: np.zeros(n_steps) for ph in phases}

    for i_z, z in enumerate(z_mm):
        p_z = p0_arr + z * slope_arr           # (n_spots, 2)
        on_det = (
            valid
            & (p_z[:, 0] >= 0) & (p_z[:, 0] < camera.Nh)
            & (p_z[:, 1] >= 0) & (p_z[:, 1] < camera.Nv)
        )
        if not on_det.any():
            continue
        dists, _ = tree.query(p_z[on_det])
        hits = dists <= tolerance_px
        w_hit = w_arr[on_det][hits] if score_weighted else hits.astype(float)
        score[i_z] = float(w_hit.sum())
        for ph in phases:
            mask_ph = (phase_arr[on_det][hits] == ph)
            score_phase[ph][i_z] = float(
                (w_arr[on_det][hits][mask_ph] if score_weighted
                 else mask_ph.astype(float)).sum()
            )

    # ── Per-peak depth inversion ──────────────────────────────────────────────
    # For each measured peak p_m, find the spot i that minimises the residual
    # |p0_i + z_opt_i * slope_i - p_m| with z_opt_i clipped to [z_min, z_max].
    z_est_per_peak  = np.full(len(meas_xy), np.nan)
    phase_per_peak  = [None] * len(meas_xy)
    sidx_per_peak   = [None] * len(meas_xy)

    valid_p0    = p0_arr[valid]
    valid_slope = slope_arr[valid]
    valid_w     = w_arr[valid]
    valid_phase = phase_arr[valid]
    valid_indices = np.where(valid)[0]

    denom = np.einsum("ij,ij->i", valid_slope, valid_slope)
    denom_safe = np.where(denom > 1e-12, denom, 1.0)

    for j, p_m in enumerate(meas_xy):
        dp = p_m[None, :] - valid_p0                         # (n_valid, 2)
        z_num = np.einsum("ij,ij->i", dp, valid_slope)
        z_opt = np.clip(z_num / denom_safe, z_min_mm, z_max_mm)
        p_opt = valid_p0 + z_opt[:, None] * valid_slope
        residuals = np.linalg.norm(p_opt - p_m[None], axis=1)
        best = int(np.argmin(residuals))
        if residuals[best] <= tolerance_px:
            z_est_per_peak[j] = float(z_opt[best])
            phase_per_peak[j] = str(valid_phase[best])
            sidx_per_peak[j]  = int(valid_indices[best])

    return {
        "z_mm":             z_mm,
        "score":            score,
        "score_per_phase":  score_phase,
        "z_est_per_peak":   z_est_per_peak,
        "phase_per_peak":   phase_per_peak,
        "spot_idx_per_peak": sidx_per_peak,
        "cos_in":           cos_in,
    }


def depth_scan_image(
    spots: list[dict],
    image: "np.ndarray",
    camera,
    stack,
    ki_hat=None,
    *,
    n_steps: int = 100,
    z_min_mm: float = 0.0,
    z_max_mm: "float | None" = None,
    min_intensity: float = 0.01,
    score_weighted: bool = True,
    interp_order: int = 1,
) -> dict:
    """
    Image-based depth-resolved Laue reconstruction by parallax sampling.

    For every simulated spot and every candidate depth *z*, the expected
    detector position is computed via the exact linear parallax formula and
    the raw pixel intensity is sampled from the detector image using
    bilinear interpolation.  No peak extraction is required.

    The full depth × spot intensity matrix is returned in a single batched
    :func:`scipy.ndimage.map_coordinates` call, making it fast even for
    large spot lists and fine depth grids.

    Two outputs are provided:

    * **Score profile** ``score[z]`` — sum (optionally intensity-weighted)
      of all sampled image values at depth *z*; its peak indicates the
      depth of dominant diffracting volume.
    * **Score matrix** ``score_matrix[z, spot]`` — per-spot depth profiles;
      each column is the raw image signal along the parallax trail of one
      spot.  Bright columns mark spots that are genuinely localised at a
      specific depth; broad columns indicate depth-spread (thick diffracting
      layers).

    Args:
        spots: Simulated spot list from :func:`simulate_laue_darwin` or
            :func:`simulate_laue_stack`.
        image: Raw detector image ``(Nv, Nh)`` float array.
        camera: :class:`Camera` used in the simulation.
        stack: :class:`LayeredCrystal` used in the simulation.
        ki_hat: Incident beam direction (3-vector, LT frame). Default
            ``[1, 0, 0]``.
        n_steps: Number of depth values to sample.
        z_min_mm: Minimum physical depth (mm, along surface normal).
        z_max_mm: Maximum physical depth (mm). Defaults to total stack
            thickness.
        min_intensity: Skip simulated spots below this normalised intensity.
        score_weighted: If ``True`` each sampled value is multiplied by the
            spot's normalised intensity before summing into ``score``.
        interp_order: Interpolation order for
            :func:`~scipy.ndimage.map_coordinates` (1 = bilinear,
            3 = bicubic). Default 1.

    Returns:
        dict with keys:

        ``'z_mm'``
            ndarray (n_steps,) — physical depth values (mm).
        ``'score'``
            ndarray (n_steps,) — aggregated image signal at each depth.
        ``'score_per_phase'``
            dict ``{phase: ndarray(n_steps,)}`` — per-phase score.
        ``'score_matrix'``
            ndarray (n_steps, n_valid_spots) — raw sampled intensities;
            columns are spots, rows are depths.
        ``'spot_phases'``
            list (n_valid_spots,) — phase label for each column of
            ``score_matrix``.
        ``'spot_hkls'``
            list (n_valid_spots,) — ``(h, k, l)`` for each column.
        ``'cos_in'``
            float — ``|k̂_i · n̂|``.
    """
    from scipy.ndimage import map_coordinates

    stack = _flatten_if_multiblock(stack)
    ki = np.asarray(ki_hat if ki_hat is not None else KI_HAT, dtype=float)
    ki = ki / np.linalg.norm(ki)
    n_hat = np.asarray(stack.n_hat, dtype=float)
    n_hat = n_hat / np.linalg.norm(n_hat)
    cos_in = float(abs(np.dot(ki, n_hat)))
    if cos_in < 1e-6:
        raise ValueError("ki is nearly parallel to the sample surface — cos_in ≈ 0")

    img = np.asarray(image, dtype=float)
    Nv, Nh = img.shape

    total_mm = sum(
        lyr.thickness for lyr in (stack.layers * stack.n_rep + stack.buffer_layers)
    ) * 1e-7
    if z_max_mm is None:
        z_max_mm = total_mm

    z_mm = np.linspace(z_min_mm, z_max_mm, n_steps)

    # ── Filter spots and precompute parallax geometry ─────────────────────────
    filtered = [
        s for s in spots
        if s.get("pix") is not None and float(s.get("intensity", 0)) >= min_intensity
    ]
    if not filtered:
        raise ValueError("No spots pass the min_intensity filter.")

    _EPS = 0.01  # mm
    p0_list, slope_list, valid_mask = [], [], []

    for s in filtered:
        tth = float(np.radians(s["tth"]))
        chi = float(np.radians(s["chi"]))
        kf = np.array([np.cos(tth),
                        np.sin(tth) * np.sin(chi),
                        np.sin(tth) * np.cos(chi)])
        p0 = camera.project(kf, source_depth_mm=0.0)
        p1 = camera.project(kf, source_depth_mm=_EPS / cos_in)
        if p0 is None or p1 is None:
            valid_mask.append(False)
            p0_list.append([np.nan, np.nan])
            slope_list.append([0.0, 0.0])
        else:
            valid_mask.append(True)
            p0_list.append(list(p0))
            slope_list.append([(p1[0] - p0[0]) / _EPS, (p1[1] - p0[1]) / _EPS])

    valid = np.array(valid_mask)
    p0_arr    = np.array(p0_list)[valid]     # (n_valid, 2)  [col, row]
    slope_arr = np.array(slope_list)[valid]  # (n_valid, 2)  [px / mm]
    w_arr     = np.array([float(s.get("intensity", 1.0)) for s in filtered])[valid]
    phase_arr = [s.get("phase_label", "unknown") for i, s in enumerate(filtered) if valid[i]]
    hkl_arr   = [s.get("hkl", (0, 0, 0))        for i, s in enumerate(filtered) if valid[i]]
    phases    = list(dict.fromkeys(phase_arr))

    n_valid = int(valid.sum())

    # ── Batch all (z, spot) positions into one map_coordinates call ───────────
    # p_all shape: (n_steps, n_valid, 2)
    p_all = p0_arr[None, :, :] + z_mm[:, None, None] * slope_arr[None, :, :]  # (n_steps, n_valid, 2)

    # map_coordinates expects (row, col) = (y, x)
    rows = p_all[:, :, 1].ravel()   # (n_steps * n_valid,)
    cols = p_all[:, :, 0].ravel()

    # Out-of-bounds positions → 0 via cval
    sampled = map_coordinates(
        img, [rows, cols],
        order=interp_order, mode="constant", cval=0.0,
    )
    score_matrix = sampled.reshape(n_steps, n_valid)  # (n_steps, n_valid)

    # Zero out off-detector entries
    on_det = (
        (p_all[:, :, 0] >= 0) & (p_all[:, :, 0] < Nh)
        & (p_all[:, :, 1] >= 0) & (p_all[:, :, 1] < Nv)
    )
    score_matrix = np.where(on_det, score_matrix, 0.0)

    # ── Aggregate scores ──────────────────────────────────────────────────────
    if score_weighted:
        score = (score_matrix * w_arr[None, :]).sum(axis=1)
    else:
        score = score_matrix.sum(axis=1)

    phase_arr_np = np.array(phase_arr)
    score_per_phase = {}
    for ph in phases:
        mask_ph = (phase_arr_np == ph)
        if score_weighted:
            score_per_phase[ph] = (score_matrix[:, mask_ph] * w_arr[None, mask_ph]).sum(axis=1)
        else:
            score_per_phase[ph] = score_matrix[:, mask_ph].sum(axis=1)

    return {
        "z_mm":            z_mm,
        "score":           score,
        "score_per_phase": score_per_phase,
        "score_matrix":    score_matrix,
        "spot_phases":     phase_arr,
        "spot_hkls":       hkl_arr,
        "cos_in":          cos_in,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SPOT / PEAK INTERSECTION
# ─────────────────────────────────────────────────────────────────────────────


def filter_spots_by_peaks(spots, peaks, tol_pix=5.0):
    """
    Keep only simulated spots that have a measured peak within *tol_pix* pixels.

    Args:
    spots : list of dict
        Output of any ``simulate_laue*`` function.  Each dict must contain a
        ``'pix'`` key with a ``(xcam, ycam)`` tuple (column, row) in detector
        pixel coordinates.  Spots whose ``'pix'`` is ``None`` are always dropped.
    peaks : array-like, shape (N, ≥2), or DataFrame with columns ``peak_X``, ``peak_Y``
        Measured peak positions from the segmentation pipeline.
        The first two columns (or ``peak_X`` / ``peak_Y`` columns) are interpreted
        as *(x_col, y_row)* pixel coordinates, matching the ``'pix'`` convention
        used by the spot dicts.
        Accepted inputs:

        * ``(N, ≥2)`` ndarray — columns 0 and 1 are ``peak_X`` and ``peak_Y``.
          This is the format returned by :func:`~nrxrdct.laue.segmentation.convert_spotsfile2peaklist`.
        * pandas DataFrame with ``'peak_X'`` and ``'peak_Y'`` columns.
        * Any sequence of *(x, y)* pairs.

    tol_pix : float, optional
        Maximum Euclidean distance (pixels) between a simulated spot and a
        measured peak for the spot to be retained.  Default: 5.0.

    Returns:
    list of dict
        Subset of *spots* — same dict objects, same order — containing only
        spots for which at least one peak lies within *tol_pix* pixels.

    Example:
    >>> peaklist = convert_spotsfile2peaklist('frame_00001.h5')
    >>> spots = simulate_laue(crystal, U, camera)
    >>> matched = filter_spots_by_peaks(spots, peaklist, tol_pix=8.0)
    >>> print(f"{len(matched)}/{len(spots)} simulated spots matched a peak")
    """
    from scipy.spatial import KDTree

    # ── Build (N, 2) array of measured peak positions ─────────────────────────
    try:
        import pandas as pd
        if isinstance(peaks, pd.DataFrame):
            xy_peaks = peaks[["peak_X", "peak_Y"]].to_numpy(dtype=float)
        else:
            xy_peaks = np.asarray(peaks, dtype=float)[:, :2]
    except ImportError:
        xy_peaks = np.asarray(peaks, dtype=float)[:, :2]

    if len(xy_peaks) == 0:
        return []

    tree = KDTree(xy_peaks)

    matched = []
    for s in spots:
        pix = s.get("pix")
        if pix is None:
            continue
        dist, _ = tree.query(pix, workers=1)
        if dist <= tol_pix:
            matched.append(s)

    return matched
