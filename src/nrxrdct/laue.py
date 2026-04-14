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

# ── Image output file ────────────────────────────────────────────────────────

IMAGE_OUTPUT = "laue_simulation.png"


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
# Phi=90 puts [001] along the beam (default, 4-fold symmetry in pattern).
# Vary phi1 (in-plane) or Phi (tilt) to bring other families to Bragg condition.

# 77.781  34.372 106.736
PHI1_DEG = 77.781
PHI_DEG = 34.372
PHI2_DEG = 106.736

# ── Lattice ───────────────────────────────────────────────────────────────────
A_LATTICE = 2.881  # Angstrom (same for BCC and B2)

# ── Camera / detector model ───────────────────────────────────────────────────
# Pixel detector (e.g. Dectris Eiger2 CdTe 9M: 3110x3269 @ 75 um)
PIXEL_SIZE_MM = 0.0734  # Pixel pitch (mm) – same in H and V
N_PIX_H = 2018  # Number of pixels, horizontal
N_PIX_V = 2016  # Number of pixels, vertical

# Detector placement
DET_DIST_MM = 86.127  # Sample → detector centre (mm)
TTH_CENTER_DEG = 82.0  # 2theta of the CENTRE of the detector (deg)
# Can be any angle: 60, 70, 80, 90, 100, 110 ...
NU_DEG = 3.528  # Out-of-plane tilt (elevation) of detector (deg)
CHI_DEG = 0.236  # In-plane rotation of detector about its normal

# Spot rendering
SPOT_SIGMA_PIX = 2.0  # Gaussian sigma of each spot (pixels)
# Increase for mosaicity / divergence broadening

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
    """Crystal-frame direction of the incident beam (+y lab)."""
    return U.T @ np.array([0.0, 1.0, 0.0])


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
    Flat pixelated area detector.

    Coordinate system
    -----------------
    Lab frame : beam along +y, x horizontal (to the right), z vertical (up).

    The detector centre is placed at angle TTH_CENTER_DEG (in the x-y plane)
    and NU_DEG elevation (out of plane), at distance DET_DIST_MM from sample.
    CHI_DEG rotates the detector about its own normal (0 = H axis in scattering plane).

    Pixel coordinates
    -----------------
    (col, row) with (0,0) at top-left corner.
    Centre pixel: (N_PIX_H/2, N_PIX_V/2).
    col increases to the right (along det_h axis).
    row increases downward (opposite to det_v axis).
    """

    def __init__(
        self,
        pixel_mm=PIXEL_SIZE_MM,
        n_pix_h=N_PIX_H,
        n_pix_v=N_PIX_V,
        dist_mm=DET_DIST_MM,
        tth_center=TTH_CENTER_DEG,
        nu_deg=NU_DEG,
        chi_deg=CHI_DEG,
    ):

        self.pixel_mm = float(pixel_mm)
        self.Nh = int(n_pix_h)
        self.Nv = int(n_pix_v)
        self.dist_mm = float(dist_mm)
        self.tth_center = float(tth_center)
        self.nu_deg = float(nu_deg)
        self.chi_deg = float(chi_deg)

        self._build_geometry()

    # ── geometry ──────────────────────────────────────────────────────────────

    def _build_geometry(self):
        tth = np.radians(self.tth_center)
        nu = np.radians(self.nu_deg)
        chi = np.radians(self.chi_deg)

        # Detector normal (unit vector from sample to detector centre)
        self.normal = np.array(
            [np.sin(tth) * np.cos(nu), np.cos(tth) * np.cos(nu), np.sin(nu)]
        )

        # Detector centre in lab coordinates
        self.centre = self.dist_mm * self.normal

        # Primary detector axes (before chi rotation)
        z_lab = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(self.normal, z_lab)) < 0.999:
            dh = np.cross(self.normal, z_lab)
            dh /= np.linalg.norm(dh)
            dv = np.cross(self.normal, dh)
            dv /= np.linalg.norm(dv)
        else:
            dh = np.array([1.0, 0.0, 0.0])
            dv = np.cross(self.normal, dh)
            dv /= np.linalg.norm(dv)

        # Apply chi rotation about detector normal
        if abs(chi) > 1e-10:
            R = Rotation.from_rotvec(chi * self.normal).as_matrix()
            dh = R @ dh
            dv = R @ dv

        self.det_h = dh  # horizontal axis (col increases along +det_h)
        self.det_v = dv  # vertical axis   (row increases along -det_v)

        # Physical size
        self.size_h_mm = self.Nh * self.pixel_mm
        self.size_v_mm = self.Nv * self.pixel_mm

    # ── projection ────────────────────────────────────────────────────────────

    def project(self, kf_hat):
        """
        Intersect scattered beam kf_hat with detector plane.

        Returns (col, row) in pixels, or None if beam misses detector.
        col, row are float (sub-pixel precision).
        """
        denom = float(np.dot(kf_hat, self.normal))
        if denom < 1e-8:  # parallel or going away
            return None
        t = float(np.dot(self.centre, self.normal)) / denom
        if t <= 0:
            return None
        hit = t * kf_hat
        delta = hit - self.centre
        dx_mm = float(np.dot(delta, self.det_h))
        dy_mm = float(np.dot(delta, self.det_v))
        col = self.Nh / 2.0 + dx_mm / self.pixel_mm
        row = self.Nv / 2.0 - dy_mm / self.pixel_mm  # row down = -det_v
        if 0.0 <= col < self.Nh and 0.0 <= row < self.Nv:
            return col, row
        return None

    def pixel_to_2theta(self, col, row):
        """
        Convert pixel position to 2theta (deg) and azimuth (deg) in lab frame.
        Useful for axis labelling on detector images.
        """
        ki_hat = np.array([0.0, 1.0, 0.0])
        dx_mm = (col - self.Nh / 2.0) * self.pixel_mm
        dy_mm = -(row - self.Nv / 2.0) * self.pixel_mm
        hit = self.centre + dx_mm * self.det_h + dy_mm * self.det_v
        hit_hat = hit / np.linalg.norm(hit)
        tth = np.degrees(np.arccos(np.clip(np.dot(ki_hat, hit_hat), -1, 1)))
        # azimuth: angle from +x in the plane perpendicular to ki
        az = np.degrees(np.arctan2(hit_hat[2], hit_hat[0]))
        return tth, az

    def describe(self):
        """Print a summary of the camera configuration."""
        print(
            f"  Camera : {self.Nh} x {self.Nv} pixels  "
            f"({self.pixel_mm*1e3:.0f} µm pitch)"
        )
        print(f"  Physical size : {self.size_h_mm:.1f} x {self.size_v_mm:.1f} mm²")
        print(f"  Sample-to-detector : {self.dist_mm:.1f} mm")
        print(
            f"  2theta centre : {self.tth_center:.2f}°   "
            f"nu = {self.nu_deg:.2f}°   chi = {self.chi_deg:.2f}°"
        )
        # Angular range at corners
        corners = [
            (0, 0),
            (self.Nh - 1, 0),
            (0, self.Nv - 1),
            (self.Nh - 1, self.Nv - 1),
        ]
        tths = [self.pixel_to_2theta(c, r)[0] for c, r in corners]
        print(f"  Angular coverage : 2theta = {min(tths):.1f}° – {max(tths):.1f}°")
        # Direct beam position
        ki_hat = np.array([0.0, 1.0, 0.0])
        db_pix = self.project(ki_hat)
        if db_pix:
            print(f"  Direct beam would hit : col={db_pix[0]:.0f} row={db_pix[1]:.0f}")
        else:
            print("  Direct beam does not hit this detector")

    # ── synthetic image ────────────────────────────────────────────────────────

    def render(self, spots, sigma_pix=SPOT_SIGMA_PIX, log_scale=True):
        """
        Render a synthetic detector image (float32 array, shape Nv x Nh).
        Each spot is a 2D Gaussian with width sigma_pix.
        """
        img = np.zeros((self.Nv, self.Nh), dtype=np.float32)
        margin = int(5 * sigma_pix) + 1
        for s in spots:
            if s.get("pix") is None:
                continue
            c, r = s["pix"]
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
    ki_hat = np.array([0.0, 1.0, 0.0])

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

                # 2theta and azimuth
                cos2th = np.clip(np.dot(ki_hat, kf_hat), -1.0, 1.0)
                tth = np.degrees(np.arccos(cos2th))
                az = np.degrees(np.arctan2(kf_hat[2], kf_hat[0]))

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


def plot_all(spots_bcc, spots_b2, crystal_bcc, camera, U):

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
        step = max(1, Nh // 20)
        cs = np.arange(0, Nh, step)
        rs = np.arange(0, Nv, step)
        CC, RR = np.meshgrid(cs, rs)
        TTH = np.zeros_like(CC, dtype=float)
        for i in range(RR.shape[0]):
            for j in range(CC.shape[1]):
                TTH[i, j] = camera.pixel_to_2theta(CC[i, j], RR[i, j])[0]

        # Contour levels around the centre 2theta
        tc = camera.tth_center
        levels = [tc - 20, tc - 10, tc, tc + 10, tc + 20]
        levels = [l for l in levels if TTH.min() < l < TTH.max()]
        if levels:
            ct = ax.contour(
                CC, RR, TTH, levels=levels, colors="#2244aa", linewidths=0.5, alpha=0.6
            )
            ax.clabel(ct, fmt="%.0f°", fontsize=5, colors="#4466cc")

        # Direct beam marker (if on detector)
        ki_hat = np.array([0.0, 1.0, 0.0])
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
    ki_hat = np.array([0.0, 1.0, 0.0])
    db = camera.project(ki_hat)
    if db:
        ax_scatter.plot(*db, "x", color=COL_DB, ms=10, mew=1.5, zorder=7)

    # 2theta grid lines
    step = max(1, Nh // 20)
    cs_g = np.arange(0, Nh, step)
    rs_g = np.arange(0, Nv, step)
    CC, RR = np.meshgrid(cs_g, rs_g)
    TTH_g = np.zeros_like(CC, dtype=float)
    for i in range(RR.shape[0]):
        for j in range(CC.shape[1]):
            TTH_g[i, j] = camera.pixel_to_2theta(CC[i, j], RR[i, j])[0]
    tc = camera.tth_center
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
    ax_tth.axvline(
        camera.tth_center,
        color="#ffffaa",
        lw=1,
        ls="--",
        label=f"Det. centre = {camera.tth_center:.0f}°",
    )
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

    # beam
    ax_geo.annotate(
        "",
        xy=(0, 0),
        xytext=(-1.6, 0),
        arrowprops=dict(arrowstyle="->", color=COL_DB, lw=2.2),
    )
    ax_geo.text(-0.8, 0.14, "white beam", color=COL_DB, fontsize=7.5, ha="center")

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
    tth_c = np.radians(camera.tth_center)
    det_dir = np.array([np.sin(tth_c), np.cos(tth_c)])  # centre direction (x,y)
    det_perp = np.array([det_dir[1], -det_dir[0]])  # perpendicular

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
        (camera.tth_center, "#ffffaa", f"2th={camera.tth_center:.0f}° (centre)"),
        (camera.tth_center - 15, "#88ddaa", f"2th={camera.tth_center-15:.0f}°"),
        (camera.tth_center + 15, "#ffaa66", f"2th={camera.tth_center+15:.0f}°"),
    ]:
        if 5 < tth_s < 175:
            tr = np.radians(tth_s)
            dx, dy = np.sin(tr), np.cos(tr)
            ax_geo.annotate(
                "",
                xy=(dx * L * 0.85, dy * L * 0.85),
                xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color=col_s, lw=1.2),
            )
            ax_geo.text(
                dx * L * 0.9 + 0.05,
                dy * L * 0.9,
                lbl,
                color=col_s,
                fontsize=5.5,
                ha="left",
                va="center",
            )

    # 2theta arc
    arc = np.linspace(
        np.radians(max(5, camera.tth_center - 25)),
        np.radians(min(175, camera.tth_center + 25)),
        80,
    )
    ax_geo.plot(0.65 * np.sin(arc), 0.65 * np.cos(arc), color="#334455", lw=1, ls="--")
    ax_geo.text(
        0.7 * np.sin(tth_c),
        0.7 * np.cos(tth_c) + 0.12,
        "2\u03b8",
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
    ax_int.axvline(
        camera.tth_center,
        color="#ffffaa",
        lw=0.8,
        ls="--",
        label=f"Centre {camera.tth_center:.0f}°",
    )
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
        (f"dist = {camera.dist_mm:.1f} mm", 7, "#88aaff", False),
        (f"2th centre = {camera.tth_center:.1f} deg", 7, "#88aaff", False),
        (f"nu = {camera.nu_deg:.1f} deg", 7, "#88aaff", False),
        (f"chi = {camera.chi_deg:.1f} deg", 7, "#88aaff", False),
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
        f"2theta_centre = {camera.tth_center:.0f}deg  |  "
        f"{camera.Nh}x{camera.Nv} px  {camera.pixel_mm*1e3:.0f}um",
        ha="center",
        fontsize=11,
        color="white",
        fontweight="bold",
    )

    plt.savefig(
        IMAGE_OUTPUT, dpi=600, bbox_inches="tight", facecolor=fig.get_facecolor()
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
    print("Rendering figure ...")
    plot_all(spots_bcc, spots_b2, bcc, cam, U)


if __name__ == "__main__":
    main()
