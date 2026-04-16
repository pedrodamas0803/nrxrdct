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
    The sample tilt is modelled as a rotation about **−y** (the horizontal axis
    perpendicular to the beam) by ``sample_tilt_deg``:

        R_tilt = Ry(−sample_tilt_deg)

    This maps the sample surface normal from +z (horizontal surface) to
    (−sin θ, 0, cos θ) in the lab frame, which for θ = 40° gives a grazing
    angle of 40° with the beam and a specular 2θ of 80°, consistent with the
    BM32 Z>0 top-camera geometry.

    When Euler angles come from a LaueTools indexing result (grain_matrix /
    deviatoric matrix) they are already expressed in the lab frame; pass
    ``sample_tilt_deg=0`` (the default) in that case.
    """
    U_sample = Rotation.from_euler("ZXZ", [phi1, Phi, phi2], degrees=True).as_matrix()
    if sample_tilt_deg == 0.0:
        return U_sample
    R_tilt = Rotation.from_euler("Y", -sample_tilt_deg, degrees=True).as_matrix()
    return R_tilt @ U_sample


def beam_in_crystal(U):
    """Crystal-frame direction of the incident beam (x in LT lab frame).
    U must already be in the lab frame (use sample_tilt_deg in euler_to_U)."""
    return U.T @ np.array([1.0, 0.0, 0.0])


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
    E_max_eV=80_000,
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
        The layered structure (from layered_structure_factor.py).
    camera : Camera
        Detector geometry (from laue_white_synchrotron.py).
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
    ki = np.asarray(ki_hat if ki_hat is not None else KI_HAT_DEFAULT, dtype=float)
    ki /= np.linalg.norm(ki)

    lam_lo = en2lam(E_max_eV)
    lam_hi = en2lam(E_min_eV)

    def spectrum(E):
        if source == "bending_magnet":
            return spectrum_bending_magnet(E, **source_kwargs)
        elif source == "wiggler":
            kw = dict(N_poles=source_kwargs.get("N_poles", 40))
            kw["Ec_eV"] = source_kwargs.get("Ec_eV", 20_000)
            return spectrum_bending_magnet(E, **kw)
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
    E_max_eV=80_000,
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
