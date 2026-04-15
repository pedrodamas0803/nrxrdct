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

import matplotlib.colors as mcolors
import matplotlib.gridspec as mgridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import xrayutilities as xu
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
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
# CRYSTAL STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────


def build_bcc(a=A_LATTICE):
    lat = xu.materials.SGLattice(
        229, a, atoms=["Al", "Co", "Cr", "Fe", "Ni"], pos=["2a"] * 5, occ=[0.2] * 5
    )
    return xu.materials.Crystal("BCC  Im-3m", lat)


def build_b2(a=A_LATTICE):
    lat = xu.materials.SGLattice(
        221,
        a,
        atoms=["Al", "Ni", "Co", "Cr", "Fe"],
        pos=["1a", "1a", "1b", "1b", "1b"],
        occ=[0.5, 0.5, 1 / 3, 1 / 3, 1 / 3],
    )
    return xu.materials.Crystal("B2   Pm-3m", lat)


# ─────────────────────────────────────────────────────────────────────────────
# ORIENTATION
# ─────────────────────────────────────────────────────────────────────────────


def euler_to_U(phi1, Phi, phi2):
    """Bunge ZXZ Euler angles (deg) -> 3x3 orientation matrix."""
    return Rotation.from_euler("ZXZ", [phi1, Phi, phi2], degrees=True).as_matrix()


def beam_in_crystal(U):
    """Crystal-frame direction of the incident beam (x in LT lab frame)."""
    return U.T @ np.array([1.0, 0.0, 0.0])


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
# CAMERA MODEL
# ─────────────────────────────────────────────────────────────────────────────


class Camera:
    """
    Pixelated area detector fully compatible with LaueTools calibration files.

    LaueTools LT2 lab frame
    -----------------------
    y  : along incident beam  (ki direction)
    z  : vertical up
    x  : horizontal (towards the wall, perpendicular to beam)

    Calibration parameters  (CCDCalibParameters = [dd, xcen, ycen, xbet, xgam])
    ---------------------------------------------------------------------------
    dd   : distance sample → detector reference point O  (mm)
    xcen : pixel X of point O  (normal-incidence / beam-footprint pixel)
    ycen : pixel Y of point O
    xbet : angle (°) between the vector IO and the vertical z axis
             xbet ≈ 0  →  camera directly above sample  (Z>0 geometry, 2θ ~ 90°)
             xbet ≈ 90 →  transmission forward camera
    xgam : in-plane rotation (°) of the CCD pixel axes around the IO direction

    kf_direction : geometry label used by LaueTools
        'Z>0'  top/side reflection  (most common, xbet small)
        'X>0'  transmission (forward)
        'X<0'  back-reflection

    Pixel convention  (identical to LaueTools)
    ------------------------------------------
    (xcam=0, ycam=0) : top-left corner of the array
    xcam increases to the right (columns)
    ycam increases downward    (rows)
    (xcen, ycen) : sub-pixel reference point where the detector normal
                   intersects the pixel array.

    The IO vector and detector normal
    ----------------------------------
    For Z>0 geometry:
        beta  = pi/2 - xbet * pi/180
        IO    = dd * [0,  cos(beta),  sin(beta)]
              = dd * [0,  sin(xbet),  cos(xbet)]
        normal = IO / |IO|

    The two key functions mirror LaueTools exactly:
        kf_to_pixel  : uflab (N×3)  →  (xcam, ycam)   [LaueTools: calc_xycam]
        pixel_to_kf  : (xcam, ycam) →  uflab (N×3)    [LaueTools: calc_uflab]
    """

    def __init__(
        self,
        dd=DD,
        xcen=XCEN,
        ycen=YCEN,
        xbet=XBET,
        xgam=XGAM,
        pixelsize=PIXEL_SIZE_MM,
        n_pix_h=N_PIX_H,
        n_pix_v=N_PIX_V,
        kf_direction=KF_DIRECTION,
    ):

        self.dd = float(dd)
        self.xcen = float(xcen)
        self.ycen = float(ycen)
        self.xbet = float(xbet)
        self.xgam = float(xgam)
        self.pixel_mm = float(pixelsize)
        self.Nh = int(n_pix_h)
        self.Nv = int(n_pix_v)
        self.kf_direction = kf_direction

        self._build_geometry()

    # ── internal geometry ──────────────────────────────────────────────────────

    def _build_geometry(self):
        DEG = np.pi / 180.0
        xbet = self.xbet
        xgam = self.xgam
        dd = self.dd

        if self.kf_direction in ("Z>0", "Y>0", "Y<0"):
            # Top / side reflection geometry (default)
            # beta = pi/2 - xbet*DEG  (angle between IO and y axis)
            self._cosbeta = np.cos(np.pi / 2 - xbet * DEG)  # = sin(xbet)
            self._sinbeta = np.sin(np.pi / 2 - xbet * DEG)  # = cos(xbet)
            # IO vector: points from sample I to detector reference point O
            self.IOlab = dd * np.array([0.0, self._cosbeta, self._sinbeta])

        elif self.kf_direction == "X>0":
            # Transmission geometry
            self._cosbeta = np.cos(-xbet * DEG)
            self._sinbeta = np.sin(-xbet * DEG)
            self.IOlab = dd * np.array([0.0, self._cosbeta, self._sinbeta])

        elif self.kf_direction == "X<0":
            # Back-reflection geometry
            self._cosbeta = np.cos(-xbet * DEG)
            self._sinbeta = np.sin(-xbet * DEG)
            self.IOlab = dd * np.array([0.0, -self._cosbeta, self._sinbeta])

        else:
            raise ValueError(f"Unknown kf_direction: {self.kf_direction!r}")

        # Detector unit normal
        self.normal = self.IOlab / np.linalg.norm(self.IOlab)
        # Precompute gam rotation coefficients (used in both directions)
        self._cosgam = np.cos(-xgam * DEG)
        self._singam = np.sin(-xgam * DEG)
        # Physical size
        self.size_h_mm = self.Nh * self.pixel_mm
        self.size_v_mm = self.Nv * self.pixel_mm

    # ── forward projection: kf unit vector → pixel ────────────────────────────

    def kf_to_pixel(self, uflab_arr):
        """
        Map scattered unit vectors to pixel coordinates.
        Implements LaueTools calc_xycam() for Z>0 geometry.

        Parameters
        ----------
        uflab_arr : (N, 3) array of unit scattered vectors in LT2 frame

        Returns
        -------
        xcam, ycam : (N,) arrays of pixel coordinates (float, sub-pixel precision)
                     Returns NaN for beams that miss the detector or go backward.
        """
        # Input is in LT frame (x // beam); camera internals use LT2 (y // beam)
        # Convert LT -> LT2:  x_LT2 = -y_LT,  y_LT2 = x_LT,  z_LT2 = z_LT
        uf_lt = np.atleast_2d(np.array(uflab_arr, dtype=float))
        norms = np.linalg.norm(uf_lt, axis=1, keepdims=True)
        uf_lt = uf_lt / norms
        uf = np.column_stack([-uf_lt[:, 1], uf_lt[:, 0], uf_lt[:, 2]])

        scal = uf @ self.normal  # cos(angle between kf and detector normal)
        valid = scal > 1e-8
        normeIM = np.where(valid, self.dd / scal, np.nan)

        IMlab = uf * normeIM[:, None]
        OMlab = IMlab - self.IOlab

        xca0 = OMlab[:, 0]
        if abs(self._sinbeta) > 1e-8:
            yca0 = OMlab[:, 1] / self._sinbeta
        else:
            yca0 = -OMlab[:, 2] / self._cosbeta

        xcam1 = self._cosgam * xca0 + self._singam * yca0
        ycam1 = -self._singam * xca0 + self._cosgam * yca0

        xcam = self.xcen + xcam1 / self.pixel_mm
        ycam = self.ycen + ycam1 / self.pixel_mm

        xcam[~valid] = np.nan
        ycam[~valid] = np.nan
        return xcam, ycam

    # ── inverse projection: pixel → kf unit vector ────────────────────────────

    def pixel_to_kf(self, xcam_arr, ycam_arr):
        """
        Map pixel coordinates to scattered unit vectors.
        Implements LaueTools calc_uflab() for Z>0 geometry.

        Parameters
        ----------
        xcam_arr, ycam_arr : array-like of pixel coordinates

        Returns
        -------
        uflab : (N, 3) unit scattered vectors in LT2 frame  (y // ki)
        """
        xcam1 = (np.asarray(xcam_arr, float) - self.xcen) * self.pixel_mm
        ycam1 = (np.asarray(ycam_arr, float) - self.ycen) * self.pixel_mm

        xca0 = self._cosgam * xcam1 - self._singam * ycam1
        yca0 = self._singam * xcam1 + self._cosgam * ycam1

        xO, yO, zO = self.IOlab
        xM = xO + xca0
        yM = yO + yca0 * self._sinbeta
        zM = zO - yca0 * self._cosbeta

        nIM = np.sqrt(xM**2 + yM**2 + zM**2)
        uflab = np.array([xM, yM, zM]).T / nIM[:, None]
        return uflab

    # ── single-spot projection for simulation ────────────────────────────────

    def project(self, kf_hat):
        """
        Project one scattered beam direction (in LT frame, x // beam) onto
        the detector.  Returns (xcam, ycam) in pixels, or None if beam misses.
        """
        # Convert LT -> LT2 for camera geometry
        kf_lt = np.array(kf_hat, dtype=float)
        kf_lt = kf_lt / np.linalg.norm(kf_lt)
        kf = np.array([-kf_lt[1], kf_lt[0], kf_lt[2]])  # LT2 frame
        scal = float(np.dot(kf, self.normal))
        if scal < 1e-8:
            return None
        normeIM = self.dd / scal
        IM = kf * normeIM
        OM = IM - self.IOlab
        xca0 = OM[0]
        yca0 = (
            OM[1] / self._sinbeta
            if abs(self._sinbeta) > 1e-8
            else -OM[2] / self._cosbeta
        )
        xcam1 = self._cosgam * xca0 + self._singam * yca0
        ycam1 = -self._singam * xca0 + self._cosgam * yca0
        xcam = self.xcen + xcam1 / self.pixel_mm
        ycam = self.ycen + ycam1 / self.pixel_mm
        if 0 <= xcam < self.Nh and 0 <= ycam < self.Nv:
            return float(xcam), float(ycam)
        return None

    # ── 2theta / chi from pixel ───────────────────────────────────────────────

    def pixel_to_2theta_chi(self, xcam, ycam):
        """
        Compute 2theta and chi (degrees) from pixel position.

        The camera geometry is computed in the LT2 frame (y // beam) which is
        what LaueGeometry.py uses internally.  We then convert to the canonical
        LaueTools LT frame (x // beam) before computing 2theta and chi:

            LT2 -> LT :   x_LT = y_LT2,   y_LT = -x_LT2,   z_LT = z_LT2

        In LT frame:
            2theta = arccos(uf_x)
            chi    = arctan2(uf_y, uf_z)
        """
        uf_lt2 = self.pixel_to_kf([xcam], [ycam])[0]
        # Convert LT2 -> LT
        uf_lt = np.array([uf_lt2[1], -uf_lt2[0], uf_lt2[2]])
        tth = np.degrees(np.arccos(np.clip(uf_lt[0], -1, 1)))
        chi = np.degrees(np.arctan2(uf_lt[1], uf_lt[2] + 1e-17))
        return tth, chi

    # ── 2theta grid on detector ───────────────────────────────────────────────

    def tth_grid(self, step=None):
        """
        Return a 2theta map over the whole detector (shape Nv x Nh).
        Useful for contour overlays on detector images.
        """
        if step is None:
            step = max(1, self.Nh // 40)
        cs = np.arange(0, self.Nh, step)
        rs = np.arange(0, self.Nv, step)
        CC, RR = np.meshgrid(cs, rs)
        uf = self.pixel_to_kf(CC.ravel(), RR.ravel())
        TTH = np.degrees(np.arccos(np.clip(uf[:, 1], -1, 1))).reshape(CC.shape)
        return CC, RR, TTH

    # ── describe ──────────────────────────────────────────────────────────────

    def describe(self):
        tth_cen, chi_cen = self.pixel_to_2theta_chi(self.xcen, self.ycen)
        corners = [
            (0, 0),
            (self.Nh - 1, 0),
            (0, self.Nv - 1),
            (self.Nh - 1, self.Nv - 1),
        ]
        tths = [self.pixel_to_2theta_chi(c, r)[0] for c, r in corners]
        print(
            f"  Camera ({self.kf_direction}) : {self.Nh} x {self.Nv} px  "
            f"pixel={self.pixel_mm*1e3:.1f} um"
        )
        print(f"  Physical size : {self.size_h_mm:.1f} x {self.size_v_mm:.1f} mm²")
        print(f"  LaueTools calibration:")
        print(f"    dd={self.dd:.3f} mm   xcen={self.xcen:.2f}   ycen={self.ycen:.2f}")
        print(f"    xbet={self.xbet:.3f} deg   xgam={self.xgam:.3f} deg")
        print(
            f"  2theta at (xcen,ycen) : {tth_cen:.4f} deg  "
            f"(= 90 - xbet = {90-self.xbet:.4f} deg)"
        )
        print(f"  chi   at (xcen,ycen) : {chi_cen:.4f} deg")
        print(
            f"  Angular coverage (corners): "
            f"2theta = {min(tths):.1f} - {max(tths):.1f} deg"
        )
        # direct beam position
        ki_hat = np.array([0.0, 1.0, 0.0])
        db = self.project(ki_hat)
        if db:
            print(
                f"  Direct beam footprint: xcam={db[0]:.1f}  ycam={db[1]:.1f}  "
                f"(pixel; should match xcen,ycen for xbet~0)"
            )
        else:
            print("  Direct beam does not hit this detector")

    # ── load from LaueTools calibration dict or list ──────────────────────────

    @classmethod
    def from_lauetools(cls, calib, pixelsize=None, framedim=None, kf_direction="Z>0"):
        """
        Build a Camera from a LaueTools calibration.

        Parameters
        ----------
        calib : list or array  [dd, xcen, ycen, xbet, xgam]
                (= CCDCalibParameters in LaueTools)
        pixelsize : float, mm  (= xpixelsize in LaueTools dict)
        framedim  : (Nh, Nv)   (= framedim in LaueTools dict)
        kf_direction : str     (= kf_direction in LaueTools dict)
        """
        dd, xcen, ycen, xbet, xgam = calib[:5]
        px = pixelsize if pixelsize is not None else PIXEL_SIZE_MM
        Nh, Nv = framedim if framedim is not None else (N_PIX_H, N_PIX_V)
        return cls(
            dd=dd,
            xcen=xcen,
            ycen=ycen,
            xbet=xbet,
            xgam=xgam,
            pixelsize=px,
            n_pix_h=int(Nh),
            n_pix_v=int(Nv),
            kf_direction=kf_direction,
        )

    # ── synthetic image ────────────────────────────────────────────────────────

    def render(self, spots, sigma_pix=SPOT_SIGMA_PIX, log_scale=True):
        """
        Render a synthetic detector image (float32, shape Nv x Nh).
        Each spot is a 2D Gaussian of width sigma_pix.
        spot's 'pix' entry must be (xcam, ycam) in LaueTools convention.
        """
        img = np.zeros((self.Nv, self.Nh), dtype=np.float32)
        margin = int(5 * sigma_pix) + 1
        for s in spots:
            if s.get("pix") is None:
                continue
            c, r = s["pix"]  # xcam, ycam
            ci, ri = int(round(c)), int(round(r))
            c0, c1 = max(0, ci - margin), min(self.Nh, ci + margin + 1)
            r0, r1 = max(0, ri - margin), min(self.Nv, ri + margin + 1)
            if c0 >= c1 or r0 >= r1:
                continue
            yy, xx = np.mgrid[r0:r1, c0:c1]
            gauss = np.exp(-((xx - c) ** 2 + (yy - r) ** 2) / (2 * sigma_pix**2))
            img[r0:r1, c0:c1] += s["intensity"] * gauss
        if log_scale and img.max() > 0:
            img = np.log1p(img / img.max() * 1000)
        return img


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
    Enumerate Laue spots, project onto the camera, compute intensities.

    Returns list of spot dicts sorted by descending normalised intensity.
    Each dict: hkl, E, lambda, tth, az, pix=(col,row),
               F2, LP, sw, intensity, is_superlattice
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


# ─────────────────────────────────────────────────────────────────────────────
# REPORTING
# ─────────────────────────────────────────────────────────────────────────────


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


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

BG = "#080c14"
FG = "#ccccee"
COL_BCC = "#4fc3f7"
COL_SUP = "#ff6633"
COL_DB = "#ffffaa"


def _ax_style(ax, title):
    ax.set_facecolor(BG)
    ax.set_title(title, color=FG, fontsize=9, pad=5)
    ax.tick_params(colors="#7788aa", labelsize=6)
    for sp in ax.spines.values():
        sp.set_edgecolor("#1a1f2e")


# ─────────────────────────────────────────────────────────────────────────────
# 2θ / χ  AND  GNOMONIC PROJECTION PLOTS
# ─────────────────────────────────────────────────────────────────────────────


def _uf_from_tth_chi(tth_deg, chi_deg):
    """Scattered unit vector from 2theta, chi  (LaueTools LT2 frame)."""
    tth = np.radians(tth_deg)
    chi = np.radians(chi_deg)
    return np.array(
        [-np.sin(tth) * np.sin(chi), np.cos(tth), np.sin(tth) * np.cos(chi)]
    )


def _gnomonic(tth_deg, chi_deg):
    """
    Gnomonic projection of a scattered beam onto the plane perpendicular
    to the forward beam direction (tangent plane at the north pole of the
    unit sphere).

        gX = -sin(2θ) sin χ  /  (1 + cos 2θ)
        gY =  sin(2θ) cos χ  /  (1 + cos 2θ)

    For 2θ < 90°  the point lies inside the unit circle (|g| < 1).
    For 2θ = 90°  |g| = 1.
    For 2θ > 90°  |g| > 1 (back-hemisphere).
    Straight lines in gnomonic space = crystallographic zones.
    """
    tth = np.asarray(tth_deg, float)
    chi = np.asarray(chi_deg, float)
    denom = 1.0 + np.cos(np.radians(tth))
    # guard against 2theta = 180 (denom = 0)
    safe = np.where(np.abs(denom) > 1e-10, denom, np.nan)
    gX = -np.sin(np.radians(tth)) * np.sin(np.radians(chi)) / safe
    gY = np.sin(np.radians(tth)) * np.cos(np.radians(chi)) / safe
    return gX, gY


def _style_angular_ax(ax, title):
    ax.set_facecolor("#080c14")
    ax.set_title(title, color="#ccccee", fontsize=9, pad=5)
    ax.tick_params(colors="#7788aa", labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor("#1a1f2e")
    ax.grid(True, ls=":", lw=0.35, color="#181e2e")


def plot_2theta_chi(spots_bcc, spots_b2, out_path="laue_2theta_chi.png"):
    """
    Plot Laue patterns in angular space: two representations side-by-side
    for each phase (BCC and B2):

    Left column  – 2θ vs χ scatter plot  (LaueTools .cor file convention)
    Right column – Gnomonic projection   (gX, gY)
                   Straight lines = crystallographic zone axes
                   Unit circle = 2θ = 90°

    Spots are coloured by photon energy and sized by normalised intensity.
    Fundamental reflections: circles (○).
    B2 superlattice reflections: stars (★) in orange.

    Parameters
    ----------
    spots_bcc, spots_b2 : lists of spot dicts from simulate_laue()
    out_path            : output PNG path
    """
    import matplotlib.colors as mcolors
    import matplotlib.gridspec as mgridspec
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    BG = "#080c14"
    FG = "#ccccee"
    COL_FUND = "#4fc3f7"
    COL_SUPER = "#ff6633"

    all_E = [s["E"] for s in spots_bcc + spots_b2]
    E_norm = mcolors.Normalize(vmin=E_MIN_eV / 1e3, vmax=E_MAX_eV / 1e3)
    cmap = "plasma"

    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor(BG)

    gs = mgridspec.GridSpec(
        2,
        3,
        width_ratios=[1, 1, 0.06],
        height_ratios=[1, 1],
        hspace=0.38,
        wspace=0.28,
        left=0.07,
        right=0.93,
        top=0.92,
        bottom=0.07,
    )

    ax_bcc_ang = fig.add_subplot(gs[0, 0])
    ax_bcc_gno = fig.add_subplot(gs[0, 1])
    ax_b2_ang = fig.add_subplot(gs[1, 0])
    ax_b2_gno = fig.add_subplot(gs[1, 1])
    ax_cb = fig.add_subplot(gs[:, 2])

    # ── helper: 2theta vs chi scatter ────────────────────────────────────────
    def draw_tth_chi(ax, spots, title):
        _style_angular_ax(ax, title)
        ax.set_xlabel("χ  (degrees)", color="#7788aa", fontsize=8)
        ax.set_ylabel("2θ  (degrees)", color="#7788aa", fontsize=8)
        ax.axvline(0, color="#252b40", lw=0.8)

        fund = [s for s in spots if not s["is_superlattice"]]
        super_ = [s for s in spots if s["is_superlattice"]]

        for subset, mk, ec in [(fund, "o", COL_FUND), (super_, "*", COL_SUPER)]:
            if not subset:
                continue
            chis = [s["chi"] for s in subset]
            tths = [s["tth"] for s in subset]
            Es = [s["E"] / 1e3 for s in subset]
            sz = [max(4, 80 * s["intensity"] ** 0.4) for s in subset]
            ax.scatter(
                chis,
                tths,
                s=sz,
                c=Es,
                cmap=cmap,
                norm=E_norm,
                alpha=0.80,
                edgecolors=ec,
                linewidths=0.35,
                marker=mk,
                zorder=3,
            )

        # Label 8 strongest fundamental spots
        for s in sorted(fund, key=lambda x: -x["intensity"])[:8]:
            h, k, l = s["hkl"]
            ax.annotate(
                f"({h}{k}{l})",
                xy=(s["chi"], s["tth"]),
                xytext=(3, 2),
                textcoords="offset points",
                fontsize=5.5,
                color="#aaccff",
                alpha=0.9,
            )

        # Label 4 strongest superlattice spots
        for s in sorted(super_, key=lambda x: -x["intensity"])[:4]:
            h, k, l = s["hkl"]
            ax.annotate(
                f"({h}{k}{l})",
                xy=(s["chi"], s["tth"]),
                xytext=(3, 2),
                textcoords="offset points",
                fontsize=5.5,
                color=COL_SUPER,
                alpha=0.9,
            )

        # Draw 2theta reference lines
        all_tths = [s["tth"] for s in spots]
        tth_min = max(0, min(all_tths) - 5) if all_tths else 60
        tth_max = min(180, max(all_tths) + 5) if all_tths else 130
        all_chis = [s["chi"] for s in spots]
        chi_min = min(all_chis) - 5 if all_chis else -50
        chi_max = max(all_chis) + 5 if all_chis else 50

        for tth_ref in np.arange(round(tth_min / 10) * 10, tth_max + 1, 10):
            ax.axhline(tth_ref, color="#1a2a3a", lw=0.6, ls="--", alpha=0.7, zorder=1)
            ax.text(
                chi_max + 0.5,
                tth_ref,
                f"{tth_ref:.0f}°",
                color="#445566",
                fontsize=6,
                va="center",
            )

        ax.set_xlim(chi_min, chi_max)
        ax.set_ylim(tth_min, tth_max)

    # ── helper: gnomonic projection ───────────────────────────────────────────
    def draw_gnomonic(ax, spots, title):
        _style_angular_ax(ax, title)
        ax.set_xlabel("gX  =  −sin2θ·sinχ / (1+cos2θ)", color="#7788aa", fontsize=7)
        ax.set_ylabel("gY  =   sin2θ·cosχ / (1+cos2θ)", color="#7788aa", fontsize=7)
        ax.set_aspect("equal")

        fund = [s for s in spots if not s["is_superlattice"]]
        super_ = [s for s in spots if s["is_superlattice"]]

        for subset, mk, ec in [(fund, "o", COL_FUND), (super_, "*", COL_SUPER)]:
            if not subset:
                continue
            gXs = [_gnomonic(s["tth"], s["chi"])[0] for s in subset]
            gYs = [_gnomonic(s["tth"], s["chi"])[1] for s in subset]
            Es = [s["E"] / 1e3 for s in subset]
            sz = [max(4, 80 * s["intensity"] ** 0.4) for s in subset]
            ax.scatter(
                gXs,
                gYs,
                s=sz,
                c=Es,
                cmap=cmap,
                norm=E_norm,
                alpha=0.80,
                edgecolors=ec,
                linewidths=0.35,
                marker=mk,
                zorder=3,
            )

        # Label strongest fundamental spots
        for s in sorted(fund, key=lambda x: -x["intensity"])[:8]:
            h, k, l = s["hkl"]
            gx, gy = _gnomonic(s["tth"], s["chi"])
            ax.annotate(
                f"({h}{k}{l})",
                xy=(gx, gy),
                xytext=(3, 2),
                textcoords="offset points",
                fontsize=5.5,
                color="#aaccff",
                alpha=0.9,
            )

        for s in sorted(super_, key=lambda x: -x["intensity"])[:4]:
            h, k, l = s["hkl"]
            gx, gy = _gnomonic(s["tth"], s["chi"])
            ax.annotate(
                f"({h}{k}{l})",
                xy=(gx, gy),
                xytext=(3, 2),
                textcoords="offset points",
                fontsize=5.5,
                color=COL_SUPER,
                alpha=0.9,
            )

        # Reference circles: 2theta = 60, 70, 80, 90, 100, 110, 120 deg
        theta_circ = np.linspace(0, 2 * np.pi, 360)
        for tth_ref in range(60, 131, 10):
            gXc, gYc = _gnomonic(np.full(360, tth_ref), np.degrees(theta_circ))
            # Only draw arc where spots exist
            valid = np.isfinite(gXc) & np.isfinite(gYc)
            if valid.any():
                col = "#ffffaa" if tth_ref == 90 else "#1a2a3a"
                lw = 0.9 if tth_ref == 90 else 0.5
                ax.plot(
                    gXc[valid],
                    gYc[valid],
                    color=col,
                    lw=lw,
                    ls="--",
                    alpha=0.7,
                    zorder=1,
                )
                # Label
                ax.text(
                    0,
                    _gnomonic(tth_ref, 0)[1] + 0.02,
                    f"{tth_ref}°",
                    color="#445566" if tth_ref != 90 else "#ffffaa",
                    fontsize=5.5,
                    ha="center",
                    va="bottom",
                )

        # Chi reference lines (radial lines at chi = 0, ±30, ±60, ±90 deg)
        for chi_ref in [0, 30, -30, 60, -60, 90, -90]:
            # Draw radial line from origin
            r_max = 3.0
            gx_r = r_max * (-np.sin(np.radians(chi_ref)))  # at 2theta=90 gX = -sin(chi)
            gy_r = r_max * np.cos(np.radians(chi_ref))
            ax.plot(
                [0, gx_r],
                [0, gy_r],
                color="#1a2a3a",
                lw=0.5,
                ls=":",
                alpha=0.6,
                zorder=1,
            )
            if abs(chi_ref) <= 90:
                ax.text(
                    gx_r * 0.9,
                    gy_r * 0.9,
                    f"χ={chi_ref}°",
                    color="#334455",
                    fontsize=5.5,
                    ha="center",
                    va="center",
                )

        # Origin crosshair (forward beam)
        ax.plot(0, 0, "+", color="#ffffaa", ms=8, mew=1.2, zorder=6)

        # Auto-scale with margin
        all_gx = [_gnomonic(s["tth"], s["chi"])[0] for s in spots]
        all_gy = [_gnomonic(s["tth"], s["chi"])[1] for s in spots]
        all_gx = [v for v in all_gx if np.isfinite(v)]
        all_gy = [v for v in all_gy if np.isfinite(v)]
        if all_gx and all_gy:
            margin = 0.3
            xmin, xmax = min(all_gx) - margin, max(all_gx) + margin
            ymin, ymax = min(all_gy) - margin, max(all_gy) + margin
            # Keep square
            r = max(abs(xmin), abs(xmax), abs(ymin), abs(ymax)) + margin
            ax.set_xlim(-r, r)
            ax.set_ylim(-r, r)

    # ── Draw all four panels ──────────────────────────────────────────────────
    n_super = sum(1 for s in spots_b2 if s["is_superlattice"])

    draw_tth_chi(
        ax_bcc_ang, spots_bcc, f"BCC  Im-3m   –   2θ vs χ   ({len(spots_bcc)} spots)"
    )
    draw_gnomonic(ax_bcc_gno, spots_bcc, f"BCC  Im-3m   –   Gnomonic projection")
    draw_b2_title = (
        f"B2   Pm-3m   –   2θ vs χ   "
        f"({len(spots_b2)} spots, {n_super} superlattice)"
    )
    draw_tth_chi(ax_b2_ang, spots_b2, draw_b2_title)
    draw_gnomonic(ax_b2_gno, spots_b2, f"B2   Pm-3m   –   Gnomonic projection")

    # ── Shared legend ─────────────────────────────────────────────────────────
    leg = [
        Line2D(
            [0],
            [0],
            marker="o",
            lw=0,
            mfc=COL_FUND,
            mec=COL_FUND,
            ms=7,
            label="Fundamental (BCC & B2)",
        ),
        Line2D(
            [0],
            [0],
            marker="*",
            lw=0,
            mfc=COL_SUPER,
            mec=COL_SUPER,
            ms=9,
            label="B2 superlattice  (h+k+l odd)",
        ),
        Line2D(
            [0],
            [0],
            color="#ffffaa",
            lw=1,
            ls="--",
            label="2θ = 90°  (gnomonic unit circle)",
        ),
    ]
    fig.legend(
        handles=leg,
        loc="upper center",
        ncol=3,
        fontsize=8,
        framealpha=0.25,
        facecolor=BG,
        labelcolor="white",
        bbox_to_anchor=(0.5, 0.97),
    )

    # ── Colourbar ─────────────────────────────────────────────────────────────
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=E_norm)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=ax_cb)
    cb.set_label("E  (keV)", color="#8899aa", fontsize=8)
    cb.ax.yaxis.set_tick_params(color="#8899aa", labelsize=7)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="#8899aa")

    # ── Title ─────────────────────────────────────────────────────────────────
    fig.text(
        0.5,
        0.995,
        "Laue pattern  –  AlCoCrFeNi  –  angular coordinates  "
        "(LaueTools convention: beam ∥ y,  χ = arctan(−uf_x / uf_z))",
        ha="center",
        va="top",
        fontsize=11,
        color="white",
        fontweight="bold",
    )

    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  Angular plot saved -> {out_path}")
    return out_path


def plot_all(spots_bcc, spots_b2, crystal_bcc, camera, U):

    # Central 2theta of detector (LaueTools: 90 - xbet)
    tc, _ = camera.pixel_to_2theta_chi(camera.xcen, camera.ycen)
    n_super = sum(1 for s in spots_b2 if s["is_superlattice"])
    all_E = [s["E"] for s in spots_bcc + spots_b2]
    E_norm = mcolors.Normalize(vmin=E_MIN_eV / 1e3, vmax=E_MAX_eV / 1e3)
    cmap = "plasma"
    cmap_obj = plt.get_cmap(cmap)

    fig = plt.figure(figsize=(22, 13))
    fig.patch.set_facecolor(BG)

    gs = mgridspec.GridSpec(
        2,
        5,
        width_ratios=[1.4, 1.4, 1, 1, 0.28],
        height_ratios=[1, 1],
        hspace=0.38,
        wspace=0.28,
        left=0.03,
        right=0.97,
        top=0.93,
        bottom=0.06,
    )

    ax_img_bcc = fig.add_subplot(gs[0, 0])
    ax_img_b2 = fig.add_subplot(gs[0, 1])
    ax_spec = fig.add_subplot(gs[0, 2])
    ax_sf = fig.add_subplot(gs[0, 3])
    ax_scatter = fig.add_subplot(gs[1, 0])
    ax_tth = fig.add_subplot(gs[1, 1])
    ax_geo = fig.add_subplot(gs[1, 2])
    ax_int = fig.add_subplot(gs[1, 3])
    ax_info = fig.add_subplot(gs[:, 4])

    Nh, Nv = camera.Nh, camera.Nv

    # ── helper: draw one detector image ───────────────────────────────────────
    def draw_det_image(ax, spots, title, sigma=SPOT_SIGMA_PIX):
        img = camera.render(spots, sigma_pix=sigma, log_scale=True)
        ax.imshow(
            img,
            origin="upper",
            cmap="hot",
            extent=[0, Nh, Nv, 0],
            aspect="auto",
            interpolation="nearest",
        )

        # Axis labels in mm and pixels
        ax.set_xlabel(
            f"col  (pixel,  pitch={camera.pixel_mm*1e3:.0f} µm)",
            color="#7788aa",
            fontsize=7,
        )
        ax.set_ylabel("row  (pixel)", color="#7788aa", fontsize=7)

        # 2theta contours overlaid on detector image
        # Sample a grid of pixels, compute 2theta, contour
        CC, RR, TTH = camera.tth_grid(step=max(1, Nh // 20))

        # Contour levels around the centre 2theta
        tc, _ = camera.pixel_to_2theta_chi(camera.xcen, camera.ycen)
        levels = [tc - 20, tc - 10, tc, tc + 10, tc + 20]
        levels = [l for l in levels if TTH.min() < l < TTH.max()]
        if levels:
            ct = ax.contour(
                CC, RR, TTH, levels=levels, colors="#2244aa", linewidths=0.5, alpha=0.6
            )
            ax.clabel(ct, fmt="%.0f°", fontsize=5, colors="#4466cc")

        # Direct beam marker (if on detector)
        ki_hat = KI_HAT / np.linalg.norm(KI_HAT)
        db = camera.project(ki_hat)
        if db:
            ax.plot(*db, "x", color=COL_DB, ms=8, mew=1.3, zorder=8)

        # Centre cross
        ax.plot(Nh / 2, Nv / 2, "+", color="#aaaaff", ms=6, mew=0.8, zorder=7)

        _ax_style(ax, title)
        ax.set_xlim(0, Nh)
        ax.set_ylim(Nv, 0)

        # Tick labels in mm
        def mm_fmt_h(x, pos):
            return f"{(x-Nh/2)*camera.pixel_mm:.0f}"

        def mm_fmt_v(x, pos):
            return f"{(x-Nv/2)*camera.pixel_mm:.0f}"

        ax2h = ax.secondary_xaxis(
            "top",
            functions=(
                lambda c: (c - Nh / 2) * camera.pixel_mm,
                lambda m: m / camera.pixel_mm + Nh / 2,
            ),
        )
        ax2v = ax.secondary_yaxis(
            "right",
            functions=(
                lambda r: (r - Nv / 2) * camera.pixel_mm,
                lambda m: m / camera.pixel_mm + Nv / 2,
            ),
        )
        ax2h.set_xlabel("mm from centre", color="#7788aa", fontsize=6)
        ax2v.set_ylabel("mm from centre", color="#7788aa", fontsize=6)
        ax2h.tick_params(colors="#7788aa", labelsize=5)
        ax2v.tick_params(colors="#7788aa", labelsize=5)
        for sp in ax2h.spines.values():
            sp.set_edgecolor("#1a1f2e")
        for sp in ax2v.spines.values():
            sp.set_edgecolor("#1a1f2e")

    draw_det_image(
        ax_img_bcc, spots_bcc, f"BCC detector image  ({len(spots_bcc)} spots)"
    )
    draw_det_image(
        ax_img_b2,
        spots_b2,
        f"B2 detector image  ({len(spots_b2)} spots,  " f"{n_super} superlattice)",
    )

    # ── Spectrum panel ─────────────────────────────────────────────────────────
    _ax_style(ax_spec, f'Synchrotron spectrum  ({SOURCE_TYPE.replace("_"," ")})')
    E_plot = np.linspace(max(500, E_MIN_eV * 0.4), E_MAX_eV * 1.1, 800)

    if SOURCE_TYPE in ("bending_magnet", "wiggler"):
        S = np.array([synchrotron_spectrum(E) for E in E_plot])
        S /= S.max()
        ax_spec.fill_between(E_plot / 1e3, S, alpha=0.18, color="#88aaff")
        ax_spec.plot(
            E_plot / 1e3,
            S,
            color="#88aaff",
            lw=1.4,
            label=f"Ec = {E_CRIT_eV/1e3:.1f} keV",
        )
        ax_spec.axvline(
            0.83 * E_CRIT_eV / 1e3,
            color="#88aaff",
            ls="--",
            lw=0.7,
            alpha=0.5,
            label=f"Peak ~0.83 Ec",
        )
    else:
        S_tot = np.zeros(len(E_plot))
        for n in range(1, 2 * N_HARMONICS, 2):
            En = n * E_FUNDAMENTAL_eV
            sig = En * HARMONIC_WIDTH
            Sh = (1 / n) * np.exp(-0.5 * ((E_plot - En) / sig) ** 2)
            S_tot += Sh
            if n <= 9:
                ax_spec.fill_between(
                    E_plot / 1e3, Sh / Sh.max() * 0.7, alpha=0.12, color="#88aaff"
                )
        if S_tot.max() > 0:
            S_tot /= S_tot.max()
        ax_spec.plot(E_plot / 1e3, S_tot, color="#88aaff", lw=1.4)

    # energy window
    ax_spec.axvspan(E_MIN_eV / 1e3, E_MAX_eV / 1e3, alpha=0.07, color="white")
    ax_spec.axvline(E_MIN_eV / 1e3, color="#888888", lw=0.7, ls="--")
    ax_spec.axvline(E_MAX_eV / 1e3, color="#888888", lw=0.7, ls="--")

    # spot energies as stems
    if spots_bcc:
        sw_arr = np.array([s["sw"] for s in spots_bcc])
        sw_norm = sw_arr / sw_arr.max() if sw_arr.max() > 0 else sw_arr
        ax_spec.vlines(
            [s["E"] / 1e3 for s in spots_bcc],
            0,
            sw_norm,
            color=COL_BCC,
            lw=0.4,
            alpha=0.35,
        )

    ax_spec.set_xlim(max(0.5, E_MIN_eV * 0.4 / 1e3), E_MAX_eV * 1.1 / 1e3)
    ax_spec.set_ylim(0, 1.3)
    ax_spec.set_xlabel("E  (keV)", color="#7788aa", fontsize=7)
    ax_spec.set_ylabel("S(E)  (norm.)", color="#7788aa", fontsize=7)
    ax_spec.legend(fontsize=6, framealpha=0.2, facecolor=BG, labelcolor="white")

    # ── |F(E)| panel ──────────────────────────────────────────────────────────
    _ax_style(ax_sf, "|F(G, E)|  vs energy  (BCC, top 4 spots)")
    E_arr = np.linspace(E_MIN_eV, E_MAX_eV, 500)
    plotted_E = []
    for s in sorted(spots_bcc, key=lambda x: -x["intensity"])[:6]:
        if any(abs(s["E"] - pe) < 1000 for pe in plotted_E):
            continue
        G = crystal_bcc.Q(*s["hkl"])
        FE = crystal_bcc.StructureFactorForEnergy(G, E_arr)
        col = cmap_obj(E_norm(s["E"] / 1e3))
        h, k, l = s["hkl"]
        ax_sf.plot(E_arr / 1e3, np.abs(FE), color=col, lw=1.1, label=f"({h}{k}{l})")
        ax_sf.axvline(s["E"] / 1e3, color=col, lw=0.6, ls="--", alpha=0.4)
        plotted_E.append(s["E"])
    ax_sf.set_xlabel("E  (keV)", color="#7788aa", fontsize=7)
    ax_sf.set_ylabel("|F|  (e.u.)", color="#7788aa", fontsize=7)
    ax_sf.legend(fontsize=6, framealpha=0.2, facecolor=BG, labelcolor="white")

    # ── Scatter plot col/row coloured by energy ────────────────────────────────
    _ax_style(ax_scatter, "Spot map  (pixel coordinates, coloured by E)")
    ax_scatter.set_facecolor("#04060e")
    ax_scatter.set_xlim(0, Nh)
    ax_scatter.set_ylim(Nv, 0)
    ax_scatter.set_aspect("equal")
    ax_scatter.set_xlabel("col  (pixel)", color="#7788aa", fontsize=7)
    ax_scatter.set_ylabel("row  (pixel)", color="#7788aa", fontsize=7)

    # Draw detector outline
    ax_scatter.add_patch(
        Rectangle((0, 0), Nh, Nv, fill=False, edgecolor="#334466", lw=0.8)
    )

    for spots_s, mk, ec in [
        (spots_bcc, "o", COL_BCC),
        ([s for s in spots_b2 if s["is_superlattice"]], "*", COL_SUP),
    ]:
        if not spots_s:
            continue
        cs = [s["pix"][0] for s in spots_s]
        rs = [s["pix"][1] for s in spots_s]
        Es = [s["E"] / 1e3 for s in spots_s]
        sz = [max(3, 60 * s["intensity"] ** 0.4) for s in spots_s]
        ax_scatter.scatter(
            cs,
            rs,
            s=sz,
            c=Es,
            cmap=cmap,
            norm=E_norm,
            alpha=0.75,
            edgecolors=ec,
            linewidths=0.3,
            marker=mk,
            zorder=3,
        )

    # Centre and direct beam
    ax_scatter.plot(Nh / 2, Nv / 2, "+", color="#aaaaff", ms=8, mew=1, zorder=6)
    ki_hat = KI_HAT / np.linalg.norm(KI_HAT)
    db = camera.project(ki_hat)
    if db:
        ax_scatter.plot(*db, "x", color=COL_DB, ms=10, mew=1.5, zorder=7)

    # 2theta grid lines
    CC, RR, TTH_g = camera.tth_grid(step=max(1, Nh // 20))
    tc, _ = camera.pixel_to_2theta_chi(camera.xcen, camera.ycen)
    lvls = sorted({tc - 20, tc - 10, tc, tc + 10, tc + 20})
    lvls = [l for l in lvls if TTH_g.min() < l < TTH_g.max()]
    if lvls:
        ct = ax_scatter.contour(
            CC, RR, TTH_g, levels=lvls, colors="#1a2a44", linewidths=0.6, alpha=0.8
        )
        ax_scatter.clabel(ct, fmt="%.0f°", fontsize=5, colors="#3355aa")

    leg_sc = [
        Line2D(
            [0],
            [0],
            marker="o",
            lw=0,
            mfc=COL_BCC,
            mec=COL_BCC,
            ms=5,
            label="BCC fundamental",
        ),
        Line2D(
            [0],
            [0],
            marker="*",
            lw=0,
            mfc=COL_SUP,
            mec=COL_SUP,
            ms=7,
            label="B2 superlattice",
        ),
        Line2D(
            [0],
            [0],
            marker="+",
            lw=0,
            color="#aaaaff",
            ms=6,
            mew=1,
            label="Det. centre",
        ),
        Line2D(
            [0], [0], marker="x", lw=0, color=COL_DB, ms=6, mew=1.3, label="Direct beam"
        ),
    ]
    ax_scatter.legend(
        handles=leg_sc,
        fontsize=5.5,
        framealpha=0.2,
        facecolor=BG,
        labelcolor="white",
        loc="upper right",
    )

    # ── 2theta histogram ──────────────────────────────────────────────────────
    _ax_style(ax_tth, "2theta distribution (intensity-weighted)")
    bins = np.linspace(0, 180, 72)
    if spots_bcc:
        ax_tth.hist(
            [s["tth"] for s in spots_bcc],
            bins=bins,
            weights=[s["intensity"] for s in spots_bcc],
            color=COL_BCC,
            alpha=0.55,
            label="BCC fund.",
        )
    sup = [s for s in spots_b2 if s["is_superlattice"]]
    if sup:
        ax_tth.hist(
            [s["tth"] for s in sup],
            bins=bins,
            weights=[s["intensity"] for s in sup],
            color=COL_SUP,
            alpha=0.70,
            label="B2 superlat.",
        )
    ax_tth.axvline(tc, color="#ffffaa", lw=1, ls="--", label=f"Det. centre = {tc:.0f}°")
    ax_tth.set_xlabel("2theta  (deg)", color="#7788aa", fontsize=7)
    ax_tth.set_ylabel("Sum intensity", color="#7788aa", fontsize=7)
    ax_tth.legend(fontsize=6, framealpha=0.2, facecolor=BG, labelcolor="white")

    # ── Geometry schematic ────────────────────────────────────────────────────
    _ax_style(ax_geo, "Geometry  (top view: x-y plane)")
    ax_geo.set_facecolor("#04060e")
    ax_geo.set_xlim(-1.7, 2.5)
    ax_geo.set_ylim(-1.8, 1.8)
    ax_geo.set_aspect("equal")
    ax_geo.axis("off")

    # beam – draw from negative KI_HAT direction
    ki_2d = np.array([KI_HAT[0], KI_HAT[1]])
    if np.linalg.norm(ki_2d) > 1e-6:
        ki_2d /= np.linalg.norm(ki_2d)
    else:
        ki_2d = np.array([0.0, 1.0])
    ax_geo.annotate(
        "",
        xy=(0, 0),
        xytext=tuple(-1.6 * ki_2d),
        arrowprops=dict(arrowstyle="->", color=COL_DB, lw=2.2),
    )
    ki_str = "".join([f"{v:+.2g}" if v != 0 else "" for v in KI_HAT])
    ax_geo.text(
        -0.8, 0.14, f"white beam ({ki_str})", color=COL_DB, fontsize=7.5, ha="center"
    )

    # sample
    ax_geo.add_patch(
        plt.Polygon(
            [(-0.12, -0.22), (0.12, -0.22), (0.12, 0.22), (-0.12, 0.22)],
            color="#445599",
            zorder=3,
        )
    )
    ax_geo.text(0, -0.38, "crystal", color="#aabbdd", fontsize=7, ha="center")

    # detector: draw as a rotated rectangle representing its orientation
    tth_c = np.radians(tc)
    # In 2D schematic (x-y plane): rotate KI by tth_c to get detector direction
    ki_2d_n = ki_2d  # already defined above (normalised 2D projection of KI_HAT)
    c_tth, s_tth = np.cos(tth_c), np.sin(tth_c)
    det_dir = np.array(
        [
            c_tth * ki_2d_n[0] - s_tth * ki_2d_n[1],
            s_tth * ki_2d_n[0] + c_tth * ki_2d_n[1],
        ]
    )
    det_perp = np.array([det_dir[1], -det_dir[0]])

    L = 1.4  # diagram scale
    half_det = 0.45
    dc_diag = L * det_dir
    p1 = dc_diag + half_det * det_perp
    p2 = dc_diag - half_det * det_perp
    ax_geo.plot(
        [p1[0], p2[0]],
        [p1[1], p2[1]],
        color="#888899",
        lw=6,
        solid_capstyle="round",
        alpha=0.75,
    )
    ax_geo.text(
        dc_diag[0] + det_dir[0] * 0.22,
        dc_diag[1] + det_dir[1] * 0.22,
        "detector",
        color="#888899",
        fontsize=6.5,
        ha="center",
        va="center",
    )

    # scattered beams at a few angles
    for tth_s, col_s, lbl in [
        (tc, "#ffffaa", f"2th={tc:.0f}deg (centre)"),
        (tc - 15, "#88ddaa", f"2th={tc-15:.0f}deg"),
        (tc + 15, "#ffaa66", f"2th={tc+15:.0f}deg"),
    ]:
        if 5 < tth_s < 175:
            tr = np.radians(tth_s)
            c_s, s_s = np.cos(tr), np.sin(tr)
            # Rotate KI_HAT by tth_s
            kf_2d = np.array(
                [
                    c_s * ki_2d_n[0] - s_s * ki_2d_n[1],
                    s_s * ki_2d_n[0] + c_s * ki_2d_n[1],
                ]
            )
            ax_geo.annotate(
                "",
                xy=(kf_2d[0] * L * 0.85, kf_2d[1] * L * 0.85),
                xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color=col_s, lw=1.2),
            )
            ax_geo.text(
                kf_2d[0] * L * 0.9 + 0.05,
                kf_2d[1] * L * 0.9,
                lbl,
                color=col_s,
                fontsize=5.5,
                ha="left",
                va="center",
            )

    # 2theta arc (around KI direction)
    ki_angle = np.arctan2(ki_2d_n[1], ki_2d_n[0])  # angle of KI in 2D
    arc_angles = np.linspace(
        ki_angle + np.radians(max(5, tc - 25)),
        ki_angle + np.radians(min(175, tc + 25)),
        80,
    )
    ax_geo.plot(
        0.65 * np.cos(arc_angles),
        0.65 * np.sin(arc_angles),
        color="#334455",
        lw=1,
        ls="--",
    )
    mid_arc = ki_angle + np.radians(tc)
    ax_geo.text(
        0.75 * np.cos(mid_arc),
        0.75 * np.sin(mid_arc),
        "2θ",
        color="#556677",
        fontsize=9,
    )

    # beam direction label
    bd = beam_in_crystal(U)
    ax_geo.text(
        0,
        -1.7,
        f"beam \u2225 [{bd[0]:.2g},{bd[1]:.2g},{bd[2]:.2g}]  "
        f"\u03c6\u2081={PHI1_DEG:.0f}\u00b0 \u03a6={PHI_DEG:.0f}\u00b0 "
        f"\u03c6\u2082={PHI2_DEG:.0f}\u00b0",
        color="#aaaacc",
        fontsize=6.5,
        ha="center",
    )

    # ── Intensity vs pixel column (horizontal cross-section) ──────────────────
    _ax_style(ax_int, "Intensity vs. 2theta (all spots)")
    if spots_bcc:
        tths_b = [s["tth"] for s in spots_bcc]
        intn_b = [s["intensity"] for s in spots_bcc]
        Es_b = [s["E"] / 1e3 for s in spots_bcc]
        ax_int.scatter(
            tths_b, intn_b, s=6, c=Es_b, cmap=cmap, norm=E_norm, alpha=0.6, zorder=3
        )
    if sup:
        ax_int.scatter(
            [s["tth"] for s in sup],
            [s["intensity"] for s in sup],
            s=15,
            color=COL_SUP,
            marker="*",
            alpha=0.85,
            zorder=4,
            label="B2 superlat.",
        )
    ax_int.axvline(tc, color="#ffffaa", lw=0.8, ls="--", label=f"Centre {tc:.0f}°")
    ax_int.set_xlabel("2theta  (deg)", color="#7788aa", fontsize=7)
    ax_int.set_ylabel("I / I_max", color="#7788aa", fontsize=7)
    ax_int.legend(fontsize=6, framealpha=0.2, facecolor=BG, labelcolor="white")

    # ── Colour bar ────────────────────────────────────────────────────────────
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=E_norm)
    sm.set_array([])
    cb_ax = fig.add_axes([0.805, 0.56, 0.008, 0.34])
    cb = fig.colorbar(sm, cax=cb_ax)
    cb.set_label("E  (keV)", color="#8899aa", fontsize=7)
    cb.ax.yaxis.set_tick_params(color="#8899aa", labelsize=6)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="#8899aa")

    # ── Info panel ────────────────────────────────────────────────────────────
    ax_info.set_facecolor("#0b0f1c")
    ax_info.axis("off")
    bd = beam_in_crystal(U)

    src_detail = (
        f"Ec={E_CRIT_eV/1e3:.1f} keV"
        if SOURCE_TYPE in ("bending_magnet", "wiggler")
        else f"E1={E_FUNDAMENTAL_eV/1e3:.1f} keV"
    )

    lines = [
        ("AlCoCrFeNi  HEA", 12, "white", True),
        ("White-Beam Laue Reflection", 9, "#aaaaff", True),
        ("", 0, "", False),
        ("Source ──────────────────", 7, "#334466", False),
        (SOURCE_TYPE.replace("_", " "), 7, "#88aaff", False),
        (src_detail, 7, "#88aaff", False),
        (f"{E_MIN_eV/1e3:.0f}-{E_MAX_eV/1e3:.0f} keV window", 7, "#88aaff", False),
        ("", 0, "", False),
        ("Crystal ─────────────────", 7, "#334466", False),
        (
            f"phi1={PHI1_DEG:.1f} Phi={PHI_DEG:.1f} phi2={PHI2_DEG:.1f} deg",
            7,
            "#88aaff",
            False,
        ),
        (f"beam||[{bd[0]:.2g},{bd[1]:.2g},{bd[2]:.2g}]", 7, "#88aaff", False),
        (
            f"ki=[{KI_HAT[0]:.2g},{KI_HAT[1]:.2g},{KI_HAT[2]:.2g}] lab",
            7,
            "#88aaff",
            False,
        ),
        (f"a = {A_LATTICE} Ang", 7, "#88aaff", False),
        ("", 0, "", False),
        ("Camera ──────────────────", 7, "#334466", False),
        (f"{camera.Nh} x {camera.Nv} pixels", 7, "#88aaff", False),
        (f"pixel = {camera.pixel_mm*1e3:.0f} um", 7, "#88aaff", False),
        (
            f"size {camera.size_h_mm:.0f} x {camera.size_v_mm:.0f} mm^2",
            7,
            "#88aaff",
            False,
        ),
        (f"dist = {camera.dd:.1f} mm", 7, "#88aaff", False),
        (f"2th at (xcen,ycen) = {90-camera.xbet:.2f} deg", 7, "#88aaff", False),
        (f"xbet = {camera.xbet:.3f} deg", 7, "#88aaff", False),
        (f"xgam = {camera.xgam:.3f} deg", 7, "#88aaff", False),
        ("", 0, "", False),
        ("Results ─────────────────", 7, "#334466", False),
        (f"BCC : {len(spots_bcc)} spots", 8, "#4fc3f7", False),
        (f"B2  : {len(spots_b2)} spots", 8, "#ffb74d", False),
        (f"  fund. : {len(spots_b2)-n_super}", 7, "#88aaff", False),
        (f"  superl: {n_super}", 7, "#ff6633", False),
        ("", 0, "", False),
        ("Intensity ────────────────", 7, "#334466", False),
        ("I=|F(Q,E)|^2*LP*S(E)", 7, "#88aaff", False),
        ("Cromer-Mann+Henke f',f\"", 6, "#556677", False),
        ("LP=(1+cos^2 2T)/(2s^2 c)", 6, "#556677", False),
        ("Gaussian spot profile", 6, "#556677", False),
    ]

    y = 0.98
    for txt, fs, col, bold in lines:
        if txt == "":
            y -= 0.010
            continue
        ax_info.text(
            0.04,
            y,
            txt,
            transform=ax_info.transAxes,
            fontsize=fs,
            color=col,
            fontweight="bold" if bold else "normal",
            va="top",
            fontfamily="monospace",
        )
        y -= 0.030 if fs >= 9 else 0.024 if fs >= 7 else 0.020

    # ── Title ─────────────────────────────────────────────────────────────────
    bd_str = f"[{bd[0]:.2g},{bd[1]:.2g},{bd[2]:.2g}]"
    src_str = SOURCE_TYPE.replace("_", " ")
    fig.text(
        0.5,
        0.965,
        f"White-Beam Laue  |  AlCoCrFeNi  |  {src_str}  "
        f"{E_MIN_eV/1e3:.0f}-{E_MAX_eV/1e3:.0f} keV  |  "
        f"beam || {bd_str}  |  "
        f"2theta_centre = {tc:.0f}deg  |  "
        f"{camera.Nh}x{camera.Nv} px  {camera.pixel_mm*1e3:.0f}um",
        ha="center",
        fontsize=11,
        color="white",
        fontweight="bold",
    )

    IMAGE_OUTPUT = "laue_white_synchrotron.png"
    plt.savefig(
        IMAGE_OUTPUT, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor()
    )
    print(f"\n  Figure saved -> {IMAGE_OUTPUT}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────


def main():
    print("Building crystal structures ...")
    bcc = build_bcc()
    b2 = build_b2()
    for c in (bcc, b2):
        lat = c.lattice
        print(
            f"  {c.name}  SG {lat.space_group}  "
            f"a={lat.a:.4f} A  V={lat.UnitCellVolume():.4f} A^3"
        )

    print(f'\nSource : {SOURCE_TYPE.replace("_"," ")}')
    if SOURCE_TYPE in ("bending_magnet", "wiggler"):
        print(
            f"  Ec = {E_CRIT_eV/1e3:.2f} keV  "
            f"(peak at ~{0.83*E_CRIT_eV/1e3:.2f} keV)"
        )
    else:
        print(
            f"  E1 = {E_FUNDAMENTAL_eV/1e3:.2f} keV  " f"({N_HARMONICS} odd harmonics)"
        )

    print(f"Energy window: {E_MIN_eV/1e3:.1f} - {E_MAX_eV/1e3:.1f} keV")

    print(f"\nOrientation: phi1={PHI1_DEG}  Phi={PHI_DEG}  phi2={PHI2_DEG} deg")
    U = euler_to_U(PHI1_DEG, PHI_DEG, PHI2_DEG)
    bd = beam_in_crystal(U)
    print(f"  Beam in crystal: [{bd[0]:.3f},{bd[1]:.3f},{bd[2]:.3f}]")

    print("\nCamera:")
    # Build camera from LaueTools calibration parameters
    # Alternatively use: cam = Camera.from_lauetools([dd,xcen,ycen,xbet,xgam], pixelsize, framedim)
    cam = Camera()
    cam.describe()

    print("\nSimulating ...")
    spots_bcc = simulate_laue(bcc, U, cam)
    spots_b2 = simulate_laue(b2, U, cam)

    n_super = sum(1 for s in spots_b2 if s["is_superlattice"])
    print(f"  BCC : {len(spots_bcc)} spots on camera")
    print(
        f"  B2  : {len(spots_b2)} spots  "
        f"({len(spots_b2)-n_super} fundamental + {n_super} superlattice)"
    )

    print(f'\n{"-"*112}')
    print_spot_table("BCC  Im-3m (SG 229)", spots_bcc, n=12)
    print_spot_table("B2   Pm-3m (SG 221)", spots_b2, n=12)

    print(f'\n{"-"*112}')
    q100 = bcc.Q(1, 0, 0)
    E_ref = 17000
    print(
        f"  |F(100)| BCC = {abs(bcc.StructureFactor(q100,en=E_ref)):.6f}"
        f"  (zero by Im-3m symmetry)"
    )
    print(
        f"  |F(100)| B2  = {abs(b2.StructureFactor(q100,en=E_ref)):.4f}"
        f"  (non-zero -> B2 superlattice)"
    )

    print(f'\n{"-"*112}')
    print_bragg_table(A_LATTICE)

    print(f'\n{"-"*112}')
    print("Rendering detector image figure ...")
    plot_all(spots_bcc, spots_b2, bcc, cam, U)

    print("Rendering 2theta/chi angular plot ...")
    plot_2theta_chi(spots_bcc, spots_b2)


if __name__ == "__main__":
    main()
