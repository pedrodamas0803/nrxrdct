# -*- coding: utf-8 -*-
"""
BM32 beamline — element-by-element API
=======================================

Design principles
-----------------
* Every optical element is a function  element_*(beam) -> (beam_out, extra)
  that reads geometry from module-level variables and is stateless otherwise.
* All positions, angles, and apertures are module-level variables so they
  can be overridden from a notebook before any call.
* _geo() recomputes all derived distances from the current variables, so
  changes take effect immediately without re-importing.
* beam_at_distance(beam, d) propagates a beam d metres forward for inspection
  at any intermediate plane.
* Slits are generic: element_slit(beam, H, V, p) clips the beam at a plane
  located p metres from the current beam origin.

Notebook usage
--------------
    import beamline2 as bm          # works regardless of filename

    # Override any parameter
    bm.G_M1   = 3.1e-3
    bm.SL2_H  = 0.050e-3
    bm.SL3_H  = 1.0e-3

    # Run element by element
    beam, norm = bm.source_bm32(nrays=500_000)
    beam_m1, _ = bm.element_m1(beam)
    beam_m2, _ = bm.element_m2(beam_m1)
    beam_sl2   = bm.element_slit(beam_m2, bm.SL2_H, bm.SL2_V, bm._geo()['L_M2_SL2'])
    beam_sl3   = bm.element_slit(beam_sl2, bm.SL3_H, bm.SL3_V, bm._geo()['L_SL2_SL3'])
    beam_kb    = bm.set_kb_source_from_beam(beam_sl3, nrays=500_000)
    beam_kb1, fp1 = bm.element_kb1(beam_kb)
    beam_kb2, fp2 = bm.element_kb2(beam_kb1)
    bm.plot_beam(beam_kb2, "At sample")
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

from shadow4.beamline.optical_elements.mirrors.s4_ellipsoid_mirror import (
    S4EllipsoidMirror, S4EllipsoidMirrorElement,
)
from shadow4.beamline.optical_elements.mirrors.s4_plane_mirror import (
    S4PlaneMirror, S4PlaneMirrorElement,
)
from shadow4.beamline.s4_optical_element_decorators import Direction, SurfaceCalculation
from shadow4.physical_models.prerefl.prerefl import PreRefl
from shadow4.sources.bending_magnet.s4_bending_magnet import S4BendingMagnet
from shadow4.sources.bending_magnet.s4_bending_magnet_light_source import S4BendingMagnetLightSource
from shadow4.sources.s4_electron_beam import S4ElectronBeam
from shadow4.sources.source_geometrical.source_geometrical import SourceGeometrical
from syned.beamline.element_coordinates import ElementCoordinates
from syned.beamline.shape import Convexity, Rectangle

# ── notebook / script compatibility ──────────────────────────────────────────
try:
    from IPython import get_ipython as _get_ipython
    INLINE_PLOTS = _get_ipython() is not None
except ImportError:
    INLINE_PLOTS = False

def _maybe_show(fig=None):
    if not INLINE_PLOTS:
        plt.show()

def _self():
    """Return this module regardless of filename."""
    import sys
    return sys.modules[__name__]

# =============================================================================
# GEOMETRY — edit these to match motor readbacks
# =============================================================================

# ── Longitudinal positions [m from source] ───────────────────────────────────
D_SL1  = 26.368   # m  Slits 1
D_M1   = 28.309   # m  Mirror 1
D_M2   = 31.732   # m  Mirror 2
D_SL2  = 35.370   # m  Slits 2 (mu-slits)
D_SL3  = 44.361   # m  Slits 3 (before KB)
D_KB1  = 44.645   # m  KB1
D_KB2  = 44.880   # m  KB2
D_SA   = 45.000   # m  Sample

# ── Grazing angles [rad] — from motor readbacks ──────────────────────────────
G_M1  = 3.062181698459972e-3   # rad  (ma1 motor)
G_M2  = 2.5627372258861216e-3  # rad  (ma2 motor)
G_KB1 = 2.2e-3                 # rad  (Ry1 motor)
G_KB2 = 13.97e-3               # rad  (Rz2 motor)

# ── Mirror curvature flag ─────────────────────────────────────────────────────
MIRROR_CURVED = False   # True = bent cylinder, False = flat (no bending applied)

# ── Mirror bending radii [m] — from bending motor encoder ────────────────────
R_M1 = 2327.636
R_M2 = 1766.232

# ── Vertical offset M1->M2 in lab frame [m] ──────────────────────────────────
# M1 deflects beam UP; M2 is higher and deflects back DOWN.
# Replace with: DZ_M1_M2 = mh2 - mh1  (height motor M2 - height motor M1)
DZ_M1_M2 = 0.02108279594058422   # m

# ── Mirror physical dimensions [m] ───────────────────────────────────────────
M1_LENGTH  = 1.100;  M1_WIDTH  = 0.050
M2_LENGTH  = 1.100;  M2_WIDTH  = 0.050
KB1_LENGTH = 0.300;  KB1_WIDTH = 0.020
KB2_LENGTH = 0.150;  KB2_WIDTH = 0.020

# ── Slit gaps [m] — FULL gap (total opening), replace with motor-derived values
# Convention: FULL gap in metres (half-opening used internally = value / 2).
# Example: SL2_H = 0.200e-3  means the slit is 0.200 mm wide total (+-0.1 mm).
# Helper: bm.mm(x) sets a gap of x mm total.
#   bm.SL2_H = bm.mm(0.2)   # 0.2 mm total gap
SL1_H = 10.000e-3;  SL1_V = 4.000e-3   # Slits 1  (10 mm H, 4 mm V total)
SL2_H =  0.200e-3;  SL2_V = 0.200e-3   # Slits 2  (0.2 mm H, 0.2 mm V total)
SL3_H =  2.000e-3;  SL3_V = 2.000e-3   # Slits 3  (2 mm H, 2 mm V total)

def mm(gap_mm):
    """Set a slit gap from a value in mm (total gap).

    Usage:
        bm.SL2_H = bm.mm(0.2)    # 0.2 mm total gap
        bm.SL1_V = bm.mm(4.0)    # 4 mm total gap
    """
    return gap_mm * 1e-3


def mrad(value_mrad):
    """Convert milliradians to radians.

    Usage:
        bm.G_M1  = bm.mrad(3.062)   # 3.062 mrad
        bm.G_KB1 = bm.mrad(2.2)     # 2.2 mrad
    """
    return value_mrad * 1e-3

# ── Energy range [eV] ────────────────────────────────────────────────────────
E_MIN = 5_000.0
E_MAX = 35_000.0

# ── ESRF EBS electron beam (SBM32) ───────────────────────────────────────────
E_GEV         = 6.04
CURRENT_A     = 0.2
ENERGY_SPREAD = 9.3e-4
SIGMA_X       = 30.1e-6
SIGMA_XP      = 4.2e-6
SIGMA_Y       = 3.6e-6
SIGMA_YP      = 1.4e-6
BM_RADIUS     = 23.588
BM_FIELD      = 0.857
BM_LENGTH     = 0.1

# ── Spectrum cache (set by set_spectrum_from_beam or compute_spectrum_at_sl2) ─
SPECTRUM_ENERGY_EV = None
SPECTRUM_FLUX      = None


# =============================================================================
# HELPERS
# =============================================================================

def _geo():
    """
    Return a dict of all derived distances from current module globals.
    Called at the start of every element function so geometry overrides
    take effect immediately.
    """
    m = _self()
    return dict(
        # Source to each element
        L_SRC_SL1 = m.D_SL1,
        L_SRC_M1  = m.D_M1,
        # Inter-element drifts (for p_coord in ElementCoordinates)
        L_SL1_M1  = m.D_M1  - m.D_SL1,
        L_M1_M2   = m.D_M2  - m.D_M1,
        L_M2_SL2  = m.D_SL2 - m.D_M2,
        L_SL2_SL3 = m.D_SL3 - m.D_SL2,
        L_SL3_KB1 = m.D_KB1 - m.D_SL3,
        L_SL2_KB1 = m.D_KB1 - m.D_SL2,
        L_SL2_KB2 = m.D_KB2 - m.D_SL2,
        L_SL3_KB2 = m.D_KB2 - m.D_SL3,
        L_KB1_KB2 = m.D_KB2 - m.D_KB1,
        L_KB1_SA  = m.D_SA  - m.D_KB1,
        L_KB2_SA  = m.D_SA  - m.D_KB2,
        # SL1 angular half-acceptance from BM source
        A_SL1_H   = (m.SL1_H / 2) / m.D_SL1,
        A_SL1_V   = (m.SL1_V / 2) / m.D_SL1,
        # SL2 angular half-acceptance from BM source
        A_SL2_H   = (m.SL2_H / 2) / m.D_SL2,
        A_SL2_V   = (m.SL2_V / 2) / m.D_SL2,
        # SL3 angular half-acceptance from BM source
        A_SL3_H   = (m.SL3_H / 2) / m.D_SL3,
        A_SL3_V   = (m.SL3_V / 2) / m.D_SL3,
        # Vertical offset tracking
        DZ_M1_M2  = m.DZ_M1_M2,
        DZ_AT_SL2 = m.DZ_M1_M2 - 2.0 * m.G_M2 * (m.D_SL2 - m.D_M2),
    )

def _ar(grazing_rad):
    return np.pi / 2 - grazing_rad

def _aperture(width, length):
    return Rectangle(-width/2, width/2, -length/2, length/2)

def _good(beam):
    return beam.rays[beam.rays[:, 9] > 0]

def _print_geometry():
    m = _self(); g = _geo()
    print("=" * 65)
    print("BM32 GEOMETRY  (current values)")
    print("=" * 65)
    g = _geo()
    print(f"  SL1 : D={m.D_SL1:.3f} m  H={m.SL1_H*1e3:.3f} mm  V={m.SL1_V*1e3:.3f} mm  -> L_SL1_M1={g['L_SL1_M1']:.3f} m to M1")
    print(f"  M1  : D={m.D_M1:.3f} m  G={m.G_M1*1e3:.4f} mrad  "
          f"({'curved' if m.MIRROR_CURVED else 'flat'})  "
          f"p={g['L_SRC_M1']:.3f} m  q={m.D_SL2-m.D_M1:.3f} m")
    print(f"  M2  : D={m.D_M2:.3f} m  G={m.G_M2*1e3:.4f} mrad  "
          f"({'curved' if m.MIRROR_CURVED else 'flat'})  "
          f"p={m.D_M2:.3f} m  q={g['L_M2_SL2']:.3f} m")
    print(f"  DZ(M1->M2) = {g['DZ_M1_M2']*1e3:.2f} mm  |  "
          f"DZ(net at SL2) = {g['DZ_AT_SL2']*1e3:.3f} mm")
    print(f"  SL2 : D={m.D_SL2:.3f} m  H={m.SL2_H*1e3:.3f} mm  V={m.SL2_V*1e3:.3f} mm")
    print(f"  SL3 : D={m.D_SL3:.3f} m  H={m.SL3_H*1e3:.3f} mm  V={m.SL3_V*1e3:.3f} mm")
    print(f"  KB1 : D={m.D_KB1:.3f} m  G={m.G_KB1*1e3:.3f} mrad  "
          f"p={g['L_SL2_KB1']:.3f} m  q={g['L_KB1_SA']:.3f} m  "
          f"demag={g['L_KB1_SA']/g['L_SL2_KB1']:.5f}")
    print(f"  KB2 : D={m.D_KB2:.3f} m  G={m.G_KB2*1e3:.3f} mrad  "
          f"p={g['L_SL2_KB2']:.3f} m  q={g['L_KB2_SA']:.3f} m  "
          f"demag={g['L_KB2_SA']/g['L_SL2_KB2']:.5f}")
    print(f"  SA  : D={m.D_SA:.3f} m")
    focus_v = m.SIGMA_Y * 2.355 * g['L_KB1_SA'] / g['L_SL2_KB1']
    focus_h = m.SIGMA_X * 2.355 * g['L_KB2_SA'] / g['L_SL2_KB2']
    print(f"  Geometric focus (BM source): "
          f"H={focus_h*1e9:.1f} nm  V={focus_v*1e9:.1f} nm FWHM")
    print("=" * 65)


# =============================================================================
# PLOTTING UTILITIES
# =============================================================================

def plot_beam(beam, label="beam", position_m=None, n_bins=100, figsize=(12, 5)):
    """Two-panel: transverse cross-section (H vs V) + energy spectrum."""
    g = _good(beam)
    if len(g) == 0:
        print(f"  [plot_beam] {label}: no surviving rays"); return
    from shadow4.beam.s4_beam import A2EV
    e_keV = g[:, 10] / A2EV / 1e3
    x_mm  = g[:, 0] * 1e3
    z_mm  = g[:, 2] * 1e3
    pos   = f"  z={position_m:.3f} m" if position_m is not None else ""
    fig, (ax_s, ax_e) = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(f"{label}{pos}  ({len(g)} rays)", fontsize=11)
    sc = ax_s.scatter(x_mm, z_mm, s=0.5, alpha=0.4, c=e_keV, cmap="plasma")
    fig.colorbar(sc, ax=ax_s, label="E (keV)")
    ax_s.set_xlabel("col1  H (mm)")
    ax_s.set_ylabel("col3  V (mm)")
    ax_s.set_title(f"H FWHM={x_mm.std()*2.355:.3f} mm  "
                   f"V FWHM={z_mm.std()*2.355:.3f} mm")
    ax_s.set_aspect("equal", adjustable="datalim")
    ax_s.grid(True, alpha=0.3)
    ax_e.hist(e_keV, bins=n_bins, range=(E_MIN/1e3, E_MAX/1e3),
              color="crimson", alpha=0.7)
    ax_e.set_xlabel("Photon energy (keV)")
    ax_e.set_ylabel("Ray count")
    ax_e.set_title("Energy distribution")
    ax_e.set_xlim(E_MIN/1e3, E_MAX/1e3)
    ax_e.grid(True, alpha=0.3)
    plt.tight_layout(); _maybe_show(fig); return fig


def plot_spectrum(beam, label="spectrum", norm_factor=1.0,
                  n_bins=200, figsize=(8, 5)):
    """Spectral flux plot. norm_factor = ph/s/ray from source_bm32."""
    g = _good(beam)
    if len(g) == 0:
        print(f"  [plot_spectrum] {label}: no rays"); return
    e_eV = beam.get_column(26, nolost=1)
    i    = beam.get_column(23, nolost=1)
    counts, edges = np.histogram(e_eV, bins=n_bins,
                                 range=(E_MIN, E_MAX), weights=i)
    en   = 0.5 * (edges[:-1] + edges[1:])
    dE   = (E_MAX - E_MIN) / n_bins
    flux = counts * norm_factor / dE
    unit = "ph/s/eV" if norm_factor != 1.0 else "a.u./eV"
    peak = en[np.argmax(flux)] / 1e3
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(en/1e3, flux, color="crimson", lw=2, label=label)
    ax.axvline(peak, color="darkred", lw=0.8, ls="--",
               label=f"peak {peak:.1f} keV")
    ax.set_xlabel("Photon energy (keV)")
    ax.set_ylabel(f"Spectral flux ({unit})")
    ax.set_title(f"Spectrum  -  {label}")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    ax.set_xlim(E_MIN/1e3, E_MAX/1e3); ax.set_ylim(0)
    plt.tight_layout(); _maybe_show(fig); return fig


def plot_footprint(footprint_beam, label="footprint", figsize=(7, 6)):
    """Mirror footprint in local (sagittal x tangential) frame."""
    g = _good(footprint_beam)
    if len(g) == 0:
        print(f"  [plot_footprint] {label}: no rays"); return
    from shadow4.beam.s4_beam import A2EV
    e_keV = g[:, 10] / A2EV / 1e3
    fig, ax = plt.subplots(figsize=figsize)
    sc = ax.scatter(g[:, 0]*1e3, g[:, 1]*1e3,
                    s=0.4, alpha=0.3, c=e_keV, cmap="plasma")
    plt.colorbar(sc, ax=ax, label="E (keV)")
    ax.set_xlabel("x sagittal (mm)")
    ax.set_ylabel("y tangential (mm)")
    ax.set_title(f"{label}  ({len(g)} rays)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout(); _maybe_show(fig); return fig


# =============================================================================
# BEAM PROPAGATION
# =============================================================================

def beam_at_distance(beam, distance_m):
    """
    Propagate beam forward by distance_m along its current direction.
    Returns a new beam at the new plane — the input beam is not modified.

    Parameters
    ----------
    beam       : S4Beam
    distance_m : float  [m]  positive = forward, negative = backward

    Example
    -------
    # Beam at KB1 entrance (from SL2)
    beam_at_kb1 = bm.beam_at_distance(beam_sl2, bm._geo()['L_SL2_KB1'])
    bm.plot_beam(beam_at_kb1, "At KB1 entrance")
    """
    from shadow4.beam.s4_beam import S4Beam
    new_rays = beam.rays.copy()
    good = new_rays[:, 9] > 0
    py   = new_rays[good, 4]
    pys  = np.where(np.abs(py) > 1e-10, py, 1e-10)
    new_rays[good, 0] += new_rays[good, 3] / pys * distance_m
    new_rays[good, 1] += distance_m
    new_rays[good, 2] += new_rays[good, 5] / pys * distance_m
    b = S4Beam(); b.rays = new_rays; return b


# =============================================================================
# GENERIC SLIT ELEMENT
# =============================================================================

def element_slit(beam, H, V, p,
                 label=None, plot=False):
    """
    Generic slit: propagate beam p metres then clip to +-H x +-V.

    This is the correct way to model any set of slits at any position:
    the beam is first propagated to the slit plane (p metres from the
    current beam origin), then rays outside the aperture are flagged lost.

    Parameters
    ----------
    beam  : S4Beam  incoming beam (any upstream position)
    H     : float   slit horizontal half-opening [m]
    V     : float   slit vertical   half-opening [m]
    p     : float   distance to propagate before clipping [m]
            Set p=0 if the beam is already at the slit plane.
    label : str     name for printout and plots (auto-generated if None)
    plot  : bool    show cross-section before/after

    Returns
    -------
    S4Beam  with rays outside aperture flagged lost

    Examples
    --------
    # SL2 from M2 exit:
    beam_sl2 = bm.element_slit(beam_m2, bm.SL2_H, bm.SL2_V,
                                p=bm._geo()['L_M2_SL2'], label="SL2")

    # SL3 from SL2 exit (beam already at SL2 plane after element_slit):
    beam_sl3 = bm.element_slit(beam_sl2, bm.SL3_H, bm.SL3_V,
                                p=bm._geo()['L_SL2_SL3'], label="SL3")

    # Change opening and re-clip the same upstream beam:
    bm.SL2_H = 0.050e-3
    beam_sl2_tight = bm.element_slit(beam_m2, bm.SL2_H, bm.SL2_V,
                                      p=bm._geo()['L_M2_SL2'], label="SL2 tight")
    """
    # H, V are FULL gaps; half-openings used for clipping
    H_half = H / 2
    V_half = V / 2
    lbl = label or f"Slit {H*1e3:.3f} x {V*1e3:.3f} mm"

    # 1. Propagate to slit plane
    if abs(p) > 1e-9:
        beam_at = beam_at_distance(beam, p)
    else:
        from shadow4.beam.s4_beam import S4Beam
        beam_at = S4Beam(); beam_at.rays = beam.rays.copy()

    # 2. Clip to +-H/2 x +-V/2
    rays  = beam_at.rays
    good  = rays[:, 9] > 0
    n_in  = good.sum()
    inside = (np.abs(rays[:, 0]) <= H_half) & (np.abs(rays[:, 2]) <= V_half)
    rays[good & ~inside, 9] = -1
    n_out = (rays[:, 9] > 0).sum()

    print(f"[Slit] {lbl}  (H=+-{H_half*1e3:.3f} mm  V=+-{V_half*1e3:.3f} mm)")
    print(f"       p={p:.3f} m  ->  "
          f"{n_out} / {n_in} rays survive  ({100*n_out/max(n_in,1):.3f}%)")

    from shadow4.beam.s4_beam import S4Beam
    beam_out = S4Beam(); beam_out.rays = rays

    if plot:
        from shadow4.beam.s4_beam import A2EV
        from matplotlib.patches import Rectangle as MplRect
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f"{lbl}", fontsize=10)
        for ax, b, ttl in [(axes[0], beam_at, "Before"),
                           (axes[1], beam_out, "After")]:
            g2 = _good(b)
            if len(g2):
                sc = ax.scatter(g2[:, 0]*1e3, g2[:, 2]*1e3, s=0.4,
                                alpha=0.4, c=g2[:, 10]/A2EV/1e3, cmap="plasma")
                fig.colorbar(sc, ax=ax, label="E (keV)")
            rect = MplRect((-H_half*1e3, -V_half*1e3), H*1e3, V*1e3,
                           lw=1.5, edgecolor="red", facecolor="none")
            ax.add_patch(rect)
            ax.set_xlabel("H (mm)"); ax.set_ylabel("V (mm)")
            ax.set_title(f"{ttl}  ({len(_good(b))} rays)")
            ax.grid(True, alpha=0.3)
        plt.tight_layout(); _maybe_show(fig)

    return beam_out


# =============================================================================
# BEAMLINE ELEMENTS
# =============================================================================

def source_bm32(nrays=500_000, seed=5676561):
    """
    Sample rays from the SBM32 bending-magnet source.

    Returns (beam, norm_factor) where norm_factor is ph/s per surviving ray.
    """
    m = _self(); _print_geometry()
    print(f"\n[Source] SBM32  ({nrays} rays) ...")
    ebeam = S4ElectronBeam(energy_in_GeV=m.E_GEV,
                           energy_spread=m.ENERGY_SPREAD, current=m.CURRENT_A)
    ebeam.set_sigmas_all(sigma_x=m.SIGMA_X, sigma_xp=m.SIGMA_XP,
                         sigma_y=m.SIGMA_Y,  sigma_yp=m.SIGMA_YP)
    bm_src = S4BendingMagnet(
        radius=m.BM_RADIUS, magnetic_field=m.BM_FIELD, length=m.BM_LENGTH,
        emin=m.E_MIN, emax=m.E_MAX, ng_e=200, flag_emittance=1)
    light_source = S4BendingMagnetLightSource(
        name="SBM32", electron_beam=ebeam,
        magnetic_structure=bm_src, nrays=nrays, seed=seed)
    beam = light_source.get_beam()
    from scipy.integrate import trapezoid as trapz
    flux  = trapz(light_source.tot,
                  light_source.angle_array_mrad * 1e-3, axis=0)
    total = trapz(flux, light_source.photon_energy_array)
    norm  = total / nrays
    print(f"       Total flux : {total:.4e} ph/s  |  "
          f"norm factor : {norm:.4e} ph/s/ray")
    print(f"       Rays       : {_good(beam).shape[0]}")
    return beam, norm


def _bm32_worker(args):
    """Module-level worker for source_bm32_parallel — must be picklable."""
    n, seed = args
    m = _self()
    from shadow4.sources.bending_magnet.s4_bending_magnet import S4BendingMagnet
    from shadow4.sources.bending_magnet.s4_bending_magnet_light_source import (
        S4BendingMagnetLightSource)
    from shadow4.sources.s4_electron_beam import S4ElectronBeam
    from scipy.integrate import trapezoid as trapz
    ebeam = S4ElectronBeam(energy_in_GeV=m.E_GEV,
                           energy_spread=m.ENERGY_SPREAD,
                           current=m.CURRENT_A)
    ebeam.set_sigmas_all(sigma_x=m.SIGMA_X, sigma_xp=m.SIGMA_XP,
                         sigma_y=m.SIGMA_Y,  sigma_yp=m.SIGMA_YP)
    bm_s = S4BendingMagnet(
        radius=m.BM_RADIUS, magnetic_field=m.BM_FIELD,
        length=m.BM_LENGTH, emin=m.E_MIN, emax=m.E_MAX,
        ng_e=200, flag_emittance=1)
    ls = S4BendingMagnetLightSource(
        name="SBM32", electron_beam=ebeam,
        magnetic_structure=bm_s, nrays=n, seed=seed)
    b = ls.get_beam()
    flux  = trapz(ls.tot, ls.angle_array_mrad * 1e-3, axis=0)
    total = trapz(flux, ls.photon_energy_array)
    return b.rays, total



def source_bm32_parallel(nrays=2_000_000, ncores=None):
    """
    Generate BM source rays in parallel across multiple CPU cores.

    Splits nrays across ncores independent workers, each generating
    nrays/ncores rays with a different random seed, then merges the
    resulting beams. The norm_factor is computed from the full set.

    The speedup is nearly linear with ncores for the BM source step,
    which is the main bottleneck for large ray counts.

    Parameters
    ----------
    nrays  : int   total number of rays (split evenly across cores)
    ncores : int   number of parallel workers (default = all CPUs)

    Returns
    -------
    beam        : S4Beam  merged beam from all workers
    norm_factor : float   ph/s per surviving ray

    Example
    -------
    beam, norm = bm.source_bm32_parallel(nrays=2_000_000, ncores=8)
    beam_sl1 = bm.element_slit(beam, bm.SL1_H, bm.SL1_V,
                                p=bm.D_SL1, label='SL1')
    """
    import multiprocessing as mp
    from shadow4.beam.s4_beam import S4Beam

    m = _self()
    if ncores is None:
        ncores = mp.cpu_count()
    ncores = max(1, ncores)

    rays_per_core = nrays // ncores
    remainder     = nrays - rays_per_core * ncores
    counts        = [rays_per_core + (1 if i < remainder else 0)
                     for i in range(ncores)]
    seeds         = [5676561 + i * 1000 for i in range(ncores)]

    print(f"\n[Source parallel] SBM32  {nrays} rays total  "
          f"across {ncores} cores ({rays_per_core} rays/core) ...")

    import time
    t0 = time.time()
    if ncores == 1:
        results = [_bm32_worker((counts[0], seeds[0]))]
    else:
        ctx = mp.get_context('fork')   # 'fork' avoids re-importing overhead
        with ctx.Pool(processes=ncores) as pool:
            results = pool.map(_bm32_worker, zip(counts, seeds))

    # Merge rays from all workers
    all_rays = np.vstack([r for r, _ in results])
    total_flux = np.mean([t for _, t in results])  # same BM flux regardless of nrays
    norm_factor = total_flux / nrays               # ph/s per ray (total)

    # Re-number rays sequentially
    all_rays[:, 11] = np.arange(1, len(all_rays) + 1, dtype=float)

    beam_out = S4Beam()
    beam_out.rays = all_rays

    dt = time.time() - t0
    n_good = (all_rays[:, 9] > 0).sum()
    print(f"       {n_good} rays in {dt:.1f}s  "
          f"({nrays/dt/1e3:.0f}k rays/s)  "
          f"norm={norm_factor:.4e} ph/s/ray")
    return beam_out, norm_factor


def _make_mirror_ir(name, boundary, p_foc, q_foc, G, curved):
    """Internal: build Ir mirror (flat or bent cylinder)."""
    if curved:
        return S4EllipsoidMirror(
            name=name, boundary_shape=boundary,
            surface_calculation=SurfaceCalculation.INTERNAL,
            p_focus=p_foc, q_focus=q_foc, grazing_angle=G,
            is_cylinder=True, cylinder_direction=Direction.TANGENTIAL,
            convexity=Convexity.UPWARD,
            f_reflec=1, f_refl=5, file_refl="",
            coating_material="Ir", coating_density=22.56, coating_roughness=3.0)
    else:
        return S4PlaneMirror(
            name=name, boundary_shape=boundary,
            f_reflec=1, f_refl=5, file_refl="",
            coating_material="Ir", coating_density=22.56, coating_roughness=3.0)


def element_m1(beam, p_from=None):
    """
    Mirror 1 - Ir, deflects beam UP (azimuthal=0).
    Flat or bent cylinder depending on MIRROR_CURVED.
    """
    m = _self(); g = _geo()
    curved = m.MIRROR_CURVED
    # p_from: distance from the last element to M1.
    # Default = D_M1 (from source). If SL1 was applied first, pass L_SL1_M1.
    p_coord = p_from if p_from is not None else g['L_SRC_M1']
    print(f"\n[M1] D={m.D_M1:.3f} m  G={m.G_M1*1e3:.4f} mrad  "
          f"L={m.M1_LENGTH*1e3:.0f} mm  ({'bent' if curved else 'flat'})  "
          f"p_coord={p_coord:.3f} m ...")
    mirror = _make_mirror_ir("M1", _aperture(m.M1_WIDTH, m.M1_LENGTH),
                              g['L_SRC_M1'], m.D_SL2 - m.D_M1, m.G_M1, curved)
    coords = ElementCoordinates(p=p_coord, q=g['L_M1_M2'],
                                angle_radial=_ar(m.G_M1), angle_azimuthal=0.0)
    if curved:
        beam_out, fp = S4EllipsoidMirrorElement(
            optical_element=mirror, coordinates=coords,
            input_beam=beam).trace_beam()
    else:
        beam_out, fp = S4PlaneMirrorElement(
            optical_element=mirror, coordinates=coords,
            input_beam=beam).trace_beam()
    n = _good(beam_out).shape[0]; n0 = _good(beam).shape[0]
    print(f"       {n} / {n0} rays survive  ({100*n/max(n0,1):.1f}%)  |  "
          f"beam rises {2*m.G_M1*g['L_M1_M2']*1e3:.2f} mm to M2")
    return beam_out, fp


def element_m2(beam):
    """
    Mirror 2 - Ir, deflects beam DOWN (azimuthal=pi), restores direction.
    Flat or bent cylinder depending on MIRROR_CURVED.
    """
    m = _self(); g = _geo()
    curved = m.MIRROR_CURVED
    print(f"\n[M2] D={m.D_M2:.3f} m  G={m.G_M2*1e3:.4f} mrad  "
          f"L={m.M2_LENGTH*1e3:.0f} mm  ({'bent' if curved else 'flat'}) ...")
    mirror = _make_mirror_ir("M2", _aperture(m.M2_WIDTH, m.M2_LENGTH),
                              m.D_M2, g['L_M2_SL2'], m.G_M2, curved)
    coords = ElementCoordinates(p=g['L_M1_M2'], q=g['L_M2_SL2'],
                                angle_radial=_ar(m.G_M2), angle_azimuthal=np.pi)
    if curved:
        beam_out, fp = S4EllipsoidMirrorElement(
            optical_element=mirror, coordinates=coords,
            input_beam=beam).trace_beam()
    else:
        beam_out, fp = S4PlaneMirrorElement(
            optical_element=mirror, coordinates=coords,
            input_beam=beam).trace_beam()
    n = _good(beam_out).shape[0]; n0 = _good(beam).shape[0]
    print(f"       {n} / {n0} rays survive  ({100*n/max(n0,1):.1f}%)  |  "
          f"net DZ at SL2 = {g['DZ_AT_SL2']*1e3:.3f} mm")
    return beam_out, fp


def element_kb1(beam):
    """
    KB1 - Ir cylinder, focuses VERTICALLY (azimuthal=0).
    p_focus = L_SL2_KB1 (SL2 is effective virtual source).
    """
    m = _self(); g = _geo()
    accept_v_mirror = (m.KB1_LENGTH/2) * np.sin(m.G_KB1) / g['L_SL2_KB1']
    accept_v_sl2    = (m.SL2_V / 2) / m.D_SL2
    accept_v_sl3    = (m.SL3_V / 2) / m.D_SL3 if m.D_SL3 < m.D_KB1 else np.inf
    accept_v_bm     = 3.0 * m.SIGMA_YP
    candidates_v    = [('mirror', accept_v_mirror), ('SL2', accept_v_sl2),
                       ('SL3', accept_v_sl3), ('BM div', accept_v_bm)]
    limiting_v      = min(candidates_v, key=lambda x: x[1])[0]
    print(f"\n[KB1] D={m.D_KB1:.3f} m  G={m.G_KB1*1e3:.3f} mrad  "
          f"L={m.KB1_LENGTH*1e3:.0f} mm  (V-focus) ...")
    print(f"       p={g['L_SL2_KB1']:.3f} m  q={g['L_KB1_SA']:.3f} m  "
          f"demag={g['L_KB1_SA']/g['L_SL2_KB1']:.5f}")
    print(f"       accept_V: mirror=+-{accept_v_mirror*1e6:.1f}  "
          f"SL2=+-{accept_v_sl2*1e6:.1f}  SL3=+-{accept_v_sl3*1e6:.1f}  "
          f"BM=+-{accept_v_bm*1e6:.2f} urad  -> {limiting_v} limits")
    mirror = S4EllipsoidMirror(
        name="KB1", boundary_shape=_aperture(m.KB1_WIDTH, m.KB1_LENGTH),
        surface_calculation=SurfaceCalculation.INTERNAL,
        p_focus=g['L_SL2_KB1'], q_focus=g['L_KB1_SA'],
        grazing_angle=m.G_KB1, is_cylinder=True,
        cylinder_direction=Direction.TANGENTIAL, convexity=Convexity.UPWARD,
        f_reflec=1, f_refl=5, file_refl="",
        coating_material="Ir", coating_density=22.56, coating_roughness=3.0)
    coords = ElementCoordinates(
        p=g['L_SL2_KB1'], q=g['L_KB1_KB2'],
        angle_radial=_ar(m.G_KB1), angle_azimuthal=0.0)
    beam_out, fp = S4EllipsoidMirrorElement(
        optical_element=mirror, coordinates=coords,
        input_beam=beam).trace_beam()
    n = _good(beam_out).shape[0]; n0 = _good(beam).shape[0]
    print(f"       {n} / {n0} rays survive  ({100*n/max(n0,1):.1f}%)")
    return beam_out, fp


def element_kb2(beam):
    """
    KB2 - Ir cylinder, focuses HORIZONTALLY (azimuthal=pi/2).
    p_focus = L_SL2_KB2 (SL2 is effective virtual source).
    """
    m = _self(); g = _geo()
    accept_h_mirror = (m.KB2_LENGTH/2) * np.sin(m.G_KB2) / g['L_SL2_KB2']
    accept_h_sl2    = (m.SL2_H / 2) / m.D_SL2
    accept_h_sl3    = (m.SL3_H / 2) / m.D_SL3 if m.D_SL3 < m.D_KB2 else np.inf
    accept_h_bm     = 3.0 * m.SIGMA_XP
    candidates_h    = [('mirror', accept_h_mirror), ('SL2', accept_h_sl2),
                       ('SL3', accept_h_sl3), ('BM div', accept_h_bm)]
    limiting_h      = min(candidates_h, key=lambda x: x[1])[0]
    print(f"\n[KB2] D={m.D_KB2:.3f} m  G={m.G_KB2*1e3:.3f} mrad  "
          f"L={m.KB2_LENGTH*1e3:.0f} mm  (H-focus) ...")
    print(f"       p={g['L_SL2_KB2']:.3f} m  q={g['L_KB2_SA']:.3f} m  "
          f"demag={g['L_KB2_SA']/g['L_SL2_KB2']:.5f}")
    print(f"       accept_H: mirror=+-{accept_h_mirror*1e6:.1f}  "
          f"SL2=+-{accept_h_sl2*1e6:.1f}  SL3=+-{accept_h_sl3*1e6:.1f}  "
          f"BM=+-{accept_h_bm*1e6:.2f} urad  -> {limiting_h} limits")
    fwhm_v = m.SIGMA_Y * 2.355 * g['L_KB1_SA'] / g['L_SL2_KB1']
    fwhm_h = m.SIGMA_X * 2.355 * g['L_KB2_SA'] / g['L_SL2_KB2']
    print(f"\n  -- Focus at sample (BM source imaged by KB) --")
    print(f"       Geometric FWHM  H={fwhm_h*1e9:.2f} nm  V={fwhm_v*1e9:.2f} nm")
    mirror = S4EllipsoidMirror(
        name="KB2", boundary_shape=_aperture(m.KB2_WIDTH, m.KB2_LENGTH),
        surface_calculation=SurfaceCalculation.INTERNAL,
        p_focus=g['L_SL2_KB2'], q_focus=g['L_KB2_SA'],
        grazing_angle=m.G_KB2, is_cylinder=True,
        cylinder_direction=Direction.TANGENTIAL, convexity=Convexity.UPWARD,
        f_reflec=1, f_refl=5, file_refl="",
        coating_material="Ir", coating_density=22.56, coating_roughness=3.0)
    coords = ElementCoordinates(
        p=g['L_KB1_KB2'], q=g['L_KB2_SA'],
        angle_radial=_ar(m.G_KB2), angle_azimuthal=np.pi/2)
    beam_out, fp = S4EllipsoidMirrorElement(
        optical_element=mirror, coordinates=coords,
        input_beam=beam).trace_beam()
    n = _good(beam_out).shape[0]; n0 = _good(beam).shape[0]
    print(f"       {n} / {n0} rays survive  ({100*n/max(n0,1):.1f}%)")
    return beam_out, fp


# =============================================================================
# SPECTRUM TOOLS
# =============================================================================

def set_spectrum_from_beam(beam, n_bins=300, plot=True,
                           label="beam", save_fig=""):
    """
    Store the energy distribution of beam in SPECTRUM_ENERGY_EV / SPECTRUM_FLUX
    so element_kb_source() uses it automatically.

    Parameters
    ----------
    beam    : S4Beam  -- any beam (e.g. after M2 or after SL2)
    n_bins  : int
    plot    : bool
    label   : str
    save_fig: str

    Returns
    -------
    energy_eV, flux
    """
    m = _self()
    from shadow4.beam.s4_beam import A2EV
    e_eV = beam.get_column(26, nolost=1)
    i    = beam.get_column(23, nolost=1)
    if len(e_eV) == 0:
        raise ValueError("set_spectrum_from_beam: no surviving rays.")
    counts, edges = np.histogram(e_eV, bins=n_bins,
                                 range=(m.E_MIN, m.E_MAX), weights=i)
    en    = 0.5 * (edges[:-1] + edges[1:])
    total = counts.sum()
    flux  = counts / total if total > 0 else np.ones_like(counts) / n_bins
    peak  = en[np.argmax(flux)] / 1e3
    print(f"[set_spectrum_from_beam] '{label}'  "
          f"{len(e_eV)} rays  peak={peak:.1f} keV")
    print(f"       SPECTRUM_ENERGY_EV / SPECTRUM_FLUX updated.")
    m.SPECTRUM_ENERGY_EV = en
    m.SPECTRUM_FLUX      = flux
    if plot:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(en/1e3, flux, color="royalblue", lw=2, label=label)
        ax.axvline(peak, color="navy", lw=0.8, ls="--",
                   label=f"peak {peak:.1f} keV")
        ax.set_xlabel("Photon energy (keV)")
        ax.set_ylabel("Normalised flux")
        ax.set_title(f"Spectrum from beam  -  {label}")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
        ax.set_xlim(m.E_MIN/1e3, m.E_MAX/1e3); ax.set_ylim(0)
        plt.tight_layout()
        if save_fig:
            plt.savefig(save_fig, dpi=150)
        _maybe_show(fig)
    return en, flux


def compute_spectrum_at_sl2(nrays=500_000, n_bins=300,
                             plot=True, save_fig=""):
    """
    Trace BM -> M1 -> M2 and store the resulting spectrum.
    Respects MIRROR_CURVED flag. Stores result in SPECTRUM_*.
    """
    m = _self(); g = _geo()
    print(f"\n[Spectrum@SL2] BM -> M1 -> M2  ({nrays} rays)  "
          f"({'curved' if m.MIRROR_CURVED else 'flat'} mirrors) ...")
    ebeam = S4ElectronBeam(energy_in_GeV=m.E_GEV,
                           energy_spread=m.ENERGY_SPREAD, current=m.CURRENT_A)
    ebeam.set_sigmas_all(sigma_x=m.SIGMA_X, sigma_xp=m.SIGMA_XP,
                         sigma_y=m.SIGMA_Y, sigma_yp=m.SIGMA_YP)
    bm_s = S4BendingMagnet(radius=m.BM_RADIUS, magnetic_field=m.BM_FIELD,
                            length=m.BM_LENGTH, emin=m.E_MIN, emax=m.E_MAX,
                            ng_e=200, flag_emittance=1)
    ls = S4BendingMagnetLightSource(name="SBM32", electron_beam=ebeam,
                                     magnetic_structure=bm_s,
                                     nrays=nrays, seed=5676561)
    beam = ls.get_beam()
    for name, p_foc, q_foc, G, p_coord, q_coord, az in [
        ("M1", g['L_SRC_M1'], m.D_SL2-m.D_M1, m.G_M1,
               g['L_SRC_M1'], g['L_M1_M2'], 0.0),
        ("M2", m.D_M2, g['L_M2_SL2'], m.G_M2,
               g['L_M1_M2'], g['L_M2_SL2'], np.pi),
    ]:
        width = m.M1_WIDTH if name == "M1" else m.M2_WIDTH
        length= m.M1_LENGTH if name == "M1" else m.M2_LENGTH
        mir   = _make_mirror_ir(name, _aperture(width, length),
                                p_foc, q_foc, G, m.MIRROR_CURVED)
        coords= ElementCoordinates(p=p_coord, q=q_coord,
                                   angle_radial=_ar(G), angle_azimuthal=az)
        if m.MIRROR_CURVED:
            beam, _ = S4EllipsoidMirrorElement(
                optical_element=mir, coordinates=coords,
                input_beam=beam).trace_beam()
        else:
            beam, _ = S4PlaneMirrorElement(
                optical_element=mir, coordinates=coords,
                input_beam=beam).trace_beam()
        print(f"       After {name}: {_good(beam).shape[0]} rays  "
              f"({100*_good(beam).shape[0]/nrays:.1f}%)")
    return set_spectrum_from_beam(beam, n_bins=n_bins,
                                  plot=plot, label="After M2",
                                  save_fig=save_fig)


# =============================================================================
# KB SOURCE
# =============================================================================

def element_kb_source(nrays=500_000, seed=1234):
    """
    Synthetic KB source at SL2: BM spatial size, SL2/SL3-limited acceptance,
    M1+M2 filtered spectrum (if SPECTRUM_* is set).

    For a physically accurate footprint, use set_kb_source_from_beam() with
    an actual simulated beam instead.
    """
    m = _self(); g = _geo()
    accept_v = min((m.KB1_LENGTH/2)*np.sin(m.G_KB1)/g['L_SL2_KB1'],
                   (m.SL2_V/2)/m.D_SL2,
                   (m.SL3_V/2)/m.D_SL3 if m.D_SL3 < m.D_KB1 else np.inf,
                   3.0 * m.SIGMA_YP)
    accept_h = min((m.KB2_LENGTH/2)*np.sin(m.G_KB2)/g['L_SL2_KB2'],
                   (m.SL2_H/2)/m.D_SL2,
                   (m.SL3_H/2)/m.D_SL3 if m.D_SL3 < m.D_KB2 else np.inf,
                   3.0 * m.SIGMA_XP)
    fwhm_v = m.SIGMA_Y * 2.355 * g['L_KB1_SA'] / g['L_SL2_KB1']
    fwhm_h = m.SIGMA_X * 2.355 * g['L_KB2_SA'] / g['L_SL2_KB2']
    print(f"\n[KB source] Virtual source at SL2  ({nrays} rays)")
    print(f"       BM sigma: H={m.SIGMA_X*1e6:.1f} um  V={m.SIGMA_Y*1e6:.1f} um")
    print(f"       accept_V=+-{accept_v*1e6:.2f} urad  "
          f"accept_H=+-{accept_h*1e6:.2f} urad")
    print(f"       Geometric focus FWHM: V={fwhm_v*1e9:.2f} nm  "
          f"H={fwhm_h*1e9:.2f} nm")
    src = SourceGeometrical(name="kb_source", nrays=nrays, seed=seed)
    src.set_spatial_type_gaussian(sigma_h=m.SIGMA_X, sigma_v=m.SIGMA_Y)
    src.set_angular_distribution_flat(hdiv1=-accept_h, hdiv2=accept_h,
                                       vdiv1=-accept_v,  vdiv2=accept_v)
    if m.SPECTRUM_ENERGY_EV is not None and m.SPECTRUM_FLUX is not None:
        src.set_energy_distribution_userdefined(
            spectrum_abscissas=m.SPECTRUM_ENERGY_EV,
            spectrum_ordinates=m.SPECTRUM_FLUX, unit="eV")
        peak = m.SPECTRUM_ENERGY_EV[np.argmax(m.SPECTRUM_FLUX)] / 1e3
        print(f"       Energy: M1+M2 spectrum  (peak {peak:.1f} keV)")
    else:
        src.set_energy_distribution_uniform(
            value_min=m.E_MIN, value_max=m.E_MAX, unit="eV")
        print(f"       Energy: UNIFORM  "
              f"(run compute_spectrum_at_sl2() for real spectrum)")
    src.set_depth_distribution_off()
    beam = src.get_beam()
    print(f"       {_good(beam).shape[0]} rays generated")
    return beam


def set_kb_source_from_beam(beam, nrays=500_000, seed=1234,
                              apply_sl2_clip=True,
                              plot=False, label="beam", save_fig=""):
    """
    Build the KB source by resampling phase space from an existing beam.

    Preserves the angular distribution (and thus footprint shape) from
    the real simulated beam. Also updates SPECTRUM_ENERGY_EV/FLUX.

    Parameters
    ----------
    beam          : S4Beam  -- beam after M2, SL2, SL3 or any upstream element
    nrays         : int     -- number of resampled rays
    seed          : int
    apply_sl2_clip: bool    -- clip to SL2 angular acceptance before resampling
                               (set False if beam already passed through SL2)
    plot          : bool
    label         : str
    save_fig      : str

    Returns
    -------
    S4Beam  ready for element_kb1()
    """
    m = _self(); g = _geo()
    from shadow4.beam.s4_beam import A2EV, S4Beam

    rays_in = beam.rays.copy()
    good    = rays_in[:, 9] > 0

    if apply_sl2_clip:
        py   = rays_in[:, 4]
        pys  = np.where(np.abs(py) > 1e-10, py, 1e-10)
        ah   = np.abs(rays_in[:, 3] / pys)
        av   = np.abs(rays_in[:, 5] / pys)
        clip = (ah <= g['A_SL2_H']) & (av <= g['A_SL2_V'])
        good = good & clip
        n_cl = good.sum()
        print(f"\n[set_kb_source_from_beam] '{label}'")
        print(f"       Input: {(rays_in[:,9]>0).sum()} rays  "
              f"-> after SL2 clip: {n_cl} rays")
        if n_cl < 20:
            print(f"  WARNING: only {n_cl} rays after SL2 clip - "
                  f"using full beam for resampling.")
            good = rays_in[:, 9] > 0
    else:
        print(f"\n[set_kb_source_from_beam] '{label}'  (no SL2 clip)")
        print(f"       Input: {good.sum()} rays")

    src_rays = rays_in[good]
    n_src    = len(src_rays)
    if n_src == 0:
        raise ValueError("set_kb_source_from_beam: no rays available.")

    # Store spectrum
    e_eV   = src_rays[:, 10] / A2EV
    counts, edges = np.histogram(e_eV, bins=300, range=(m.E_MIN, m.E_MAX))
    en     = 0.5 * (edges[:-1] + edges[1:])
    flux   = counts / counts.sum() if counts.sum() > 0 else np.ones(300)/300
    peak   = en[np.argmax(flux)] / 1e3
    m.SPECTRUM_ENERGY_EV = en
    m.SPECTRUM_FLUX      = flux
    print(f"       Spectrum stored: peak={peak:.1f} keV")

    # Resample
    rng      = np.random.default_rng(seed)
    idx      = rng.choice(n_src, size=nrays, replace=True)
    new_rays = src_rays[idx].copy()
    new_rays[:, 9]  = 1
    new_rays[:, 1]  = 0.0
    new_rays[:, 11] = np.arange(1, nrays + 1, dtype=float)
    print(f"       Resampled {nrays} rays from {n_src} unique rays")

    fwhm_v = m.SIGMA_Y * 2.355 * g['L_KB1_SA'] / g['L_SL2_KB1']
    fwhm_h = m.SIGMA_X * 2.355 * g['L_KB2_SA'] / g['L_SL2_KB2']
    print(f"       Geometric focus FWHM: V={fwhm_v*1e9:.2f} nm  "
          f"H={fwhm_h*1e9:.2f} nm")

    beam_out = S4Beam(); beam_out.rays = new_rays

    if plot:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle(f"KB source phase space from '{label}'", fontsize=11)
        ax = axes[0]
        px_urad = src_rays[:, 3] * 1e6
        pz_urad = src_rays[:, 5] * 1e6
        ax.scatter(px_urad, pz_urad, s=0.3, alpha=0.3, color="steelblue")
        for lim, col in [(g['A_SL2_H']*1e6, "red"),
                         (g['A_SL3_H']*1e6 if m.D_SL3 < m.D_KB1 else None, "orange")]:
            if lim:
                ax.axvline( lim, color=col, lw=0.8, ls="--")
                ax.axvline(-lim, color=col, lw=0.8, ls="--")
                ax.axhline( lim, color=col, lw=0.8, ls="--")
                ax.axhline(-lim, color=col, lw=0.8, ls="--")
        ax.set_xlabel("px/py (urad)"); ax.set_ylabel("pz/py (urad)")
        ax.set_title("Angular distribution"); ax.grid(True, alpha=0.3)
        axes[1].scatter(src_rays[:,0]*1e3, src_rays[:,2]*1e3,
                        s=0.3, alpha=0.3, color="royalblue")
        axes[1].set_xlabel("x H (mm)"); axes[1].set_ylabel("z V (mm)")
        axes[1].set_title("Spatial distribution"); axes[1].grid(True, alpha=0.3)
        axes[2].plot(en/1e3, flux, color="crimson", lw=2)
        axes[2].axvline(peak, color="darkred", lw=0.8, ls="--",
                        label=f"peak {peak:.1f} keV")
        axes[2].set_xlabel("Energy (keV)"); axes[2].set_ylabel("Norm. flux")
        axes[2].set_title("Spectrum"); axes[2].legend(fontsize=8)
        axes[2].set_xlim(m.E_MIN/1e3, m.E_MAX/1e3); axes[2].grid(True, alpha=0.3)
        plt.tight_layout()
        if save_fig: plt.savefig(save_fig, dpi=150)
        _maybe_show(fig)

    return beam_out


# =============================================================================
# CONVENIENCE: full chain
# =============================================================================

def run_full_kb_chain(nrays_bm=2_000_000, nrays_kb=500_000,
                      plot_each=False, plot_final=True):
    """
    Full two-stage chain with all slits as explicit spatial apertures.

    Stage 1  (BM statistics):
        BM -> M1 -> M2
        -> element_slit(SL2) at D_SL2
        -> element_slit(SL3) at D_SL3   [if D_SL3 < D_KB1]

    Stage 2  (KB statistics):
        set_kb_source_from_beam() resamples from Stage 1 survivors
        -> KB1 -> KB2 -> sample

    The slit openings SL2_H/V and SL3_H/V act as real spatial boundaries.
    Change them and re-run to evaluate their effect on flux and footprints.

    Parameters
    ----------
    nrays_bm  : int   BM source rays for Stage 1 (recommend >= 1M)
    nrays_kb  : int   KB source rays for Stage 2 (recommend >= 200k)
    plot_each : bool  plot beam after every element
    plot_final: bool  plot spectrum + footprints at sample

    Returns
    -------
    dict: beam_m1, beam_m2, beam_sl2, beam_sl3, beam_kb_source,
          beam_kb1, footprint_kb1, beam_kb2, footprint_kb2, norm_factor
    """
    m = _self(); g = _geo()
    print("=" * 60)
    print("Stage 1  BM -> M1 -> M2 -> slits")
    print("=" * 60)
    beam, norm = source_bm32(nrays=nrays_bm)

    # SL1 (optional — uses module SL1_H/V)
    beam_sl1 = element_slit(beam, m.SL1_H, m.SL1_V,
                             p=g['L_SRC_SL1'], label="SL1",
                             plot=plot_each)
    if plot_each: plot_beam(beam_sl1, "After SL1", position_m=m.D_SL1)

    # M1: beam is now at SL1, so p_from = D_M1 - D_SL1
    beam_m1, _ = element_m1(beam_sl1, p_from=g['L_SL1_M1'])
    if plot_each: plot_beam(beam_m1, "After M1", position_m=m.D_M1)

    beam_m2, _ = element_m2(beam_m1)
    if plot_each: plot_beam(beam_m2, "After M2", position_m=m.D_M2)

    # SL2 as real spatial aperture
    beam_sl2 = element_slit(beam_m2, m.SL2_H, m.SL2_V,
                             p=g['L_M2_SL2'], label="SL2",
                             plot=plot_each)
    if plot_each: plot_beam(beam_sl2, "After SL2", position_m=m.D_SL2)

    # SL3 (if it is between M2 and KB1)
    beam_sl3 = beam_sl2
    if m.D_SL3 < m.D_KB1:
        beam_sl3 = element_slit(beam_sl2, m.SL3_H, m.SL3_V,
                                 p=g['L_SL2_SL3'], label="SL3",
                                 plot=plot_each)
        if plot_each: plot_beam(beam_sl3, "After SL3", position_m=m.D_SL3)

    print()
    print("=" * 60)
    print("Stage 2  KB source -> KB1 -> KB2")
    print("=" * 60)
    beam_kb = set_kb_source_from_beam(
        beam_sl3, nrays=nrays_kb, apply_sl2_clip=False,
        plot=plot_each, label="After last slit")

    beam_kb1, fp1 = element_kb1(beam_kb)
    if plot_each:
        plot_beam(beam_kb1, "After KB1", position_m=m.D_KB1)
        plot_footprint(fp1, "KB1 footprint")

    beam_kb2, fp2 = element_kb2(beam_kb1)
    if plot_each:
        plot_beam(beam_kb2, "At sample", position_m=m.D_SA)
        plot_footprint(fp2, "KB2 footprint")

    if plot_final:
        plot_spectrum(beam_kb2, "At sample", norm_factor=norm)
        plot_footprint(fp1, "KB1 footprint")
        plot_footprint(fp2, "KB2 footprint")

    return dict(
        beam_sl1=beam_sl1,
        beam_m1=beam_m1, beam_m2=beam_m2,
        beam_sl2=beam_sl2, beam_sl3=beam_sl3,
        beam_kb_source=beam_kb,
        beam_kb1=beam_kb1, footprint_kb1=fp1,
        beam_kb2=beam_kb2, footprint_kb2=fp2,
        norm_factor=norm,
    )


# =============================================================================
def plot_beam_path(results=None, n_samples=80, figsize=(18, 7), save_fig=""):
    """
    Render the BM32 beamline beam path from source to sample.

    Draws two panels:
      Top    — Side view (Y longitudinal vs Z vertical, lab frame).
               Shows the vertical deflections at M1 and M2, and the
               vertical focusing by KB1.
      Bottom — Top view  (Y longitudinal vs X horizontal, lab frame).
               Shows the horizontal focusing by KB2.

    The beam envelope (±1 sigma) is shown at each position, computed
    analytically from the BM source emittance and the optical geometry.

    Parameters
    ----------
    results : dict (optional)
        Output of run_full_kb_chain().  If provided, the actual beam
        sizes from the simulation are overlaid on the analytic envelope.
    n_samples : int
        Number of longitudinal positions for the envelope.
    figsize : tuple
    save_fig : str   filename to save ('' = don't save)

    Returns
    -------
    matplotlib Figure
    """
    m = _self(); g = _geo()

    # ── Beamline element positions and labels ─────────────────────────────────
    elements = [
        (0.0,      'Source',  'S'),
        (m.D_SL1,  'SL1',     'slit'),
        (m.D_M1,   'M1',      'mirror_v'),
        (m.D_M2,   'M2',      'mirror_v'),
        (m.D_SL2,  'SL2',     'slit'),
        (m.D_SL3,  'SL3',     'slit'),
        (m.D_KB1,  'KB1',     'mirror_v'),
        (m.D_KB2,  'KB2',     'mirror_h'),
        (m.D_SA,   'Sample',  'S'),
    ]

    # ── Lab-frame beam centroid path ──────────────────────────────────────────
    # Track z_lab (vertical) and x_lab (horizontal) of beam centroid.
    # Deflections happen at mirrors; beam propagates as straight lines between.
    #
    # Vertical (side view):
    #   M1 deflects UP   by 2*G_M1 → beam rises until M2
    #   M2 deflects DOWN by 2*G_M2 → beam (nearly) restored to axis
    #   KB1 deflects the beam for focusing (azimuthal=0, vertical plane)
    #
    # Horizontal (top view):
    #   M1, M2 sagittal → no horizontal deflection
    #   KB2 deflects horizontally (azimuthal=pi/2)

    # Build piecewise-linear centroid path
    # Segments: (y_start, y_end, dz_start, slope_z, dx_start, slope_x)
    def _segments():
        segs = []
        # Source → M1: on axis, horizontal
        segs.append(dict(y0=0, y1=m.D_M1,
                         z0=0,         sz=0,
                         x0=0,         sx=0))
        # M1 → M2: beam rises by 2*G_M1 rad, starts at z=0 at M1
        segs.append(dict(y0=m.D_M1, y1=m.D_M2,
                         z0=0,         sz=2*m.G_M1,
                         x0=0,         sx=0))
        # M2 → SL2: M2 is at z_lab = DZ_M1_M2; M2 deflects back down by 2*G_M2
        z_at_m2 = 2*m.G_M1 * (m.D_M2 - m.D_M1)
        slope_after_m2 = 2*m.G_M1 - 2*m.G_M2   # net slope (should be ~0)
        segs.append(dict(y0=m.D_M2, y1=m.D_SL2,
                         z0=z_at_m2,   sz=slope_after_m2,
                         x0=0,         sx=0))
        # SL2 → KB1: beam continues at same slope
        z_at_sl2 = z_at_m2 + slope_after_m2 * (m.D_SL2 - m.D_M2)
        segs.append(dict(y0=m.D_SL2, y1=m.D_KB1,
                         z0=z_at_sl2,  sz=slope_after_m2,
                         x0=0,         sx=0))
        # KB1 → KB2: KB1 deflects vertically by 2*G_KB1 downward (azimuthal=0)
        z_at_kb1 = z_at_sl2 + slope_after_m2 * (m.D_KB1 - m.D_SL2)
        slope_after_kb1 = slope_after_m2 - 2*m.G_KB1
        segs.append(dict(y0=m.D_KB1, y1=m.D_KB2,
                         z0=z_at_kb1,  sz=slope_after_kb1,
                         x0=0,         sx=0))
        # KB2 → Sample: KB2 deflects horizontally by 2*G_KB2 (azimuthal=pi/2)
        z_at_kb2 = z_at_kb1 + slope_after_kb1 * (m.D_KB2 - m.D_KB1)
        segs.append(dict(y0=m.D_KB2, y1=m.D_SA,
                         z0=z_at_kb2,  sz=slope_after_kb1,
                         x0=0,         sx=-2*m.G_KB2))
        return segs

    segs = _segments()

    def centroid_at(y):
        """Lab-frame (z, x) centroid at longitudinal position y."""
        for s in segs:
            if s['y0'] <= y <= s['y1'] + 1e-9:
                dy = y - s['y0']
                return s['z0'] + s['sz']*dy, s['x0'] + s['sx']*dy
        s = segs[-1]
        dy = y - s['y0']
        return s['z0'] + s['sz']*dy, s['x0'] + s['sx']*dy

    # ── Analytic beam envelope (±1 sigma) ─────────────────────────────────────
    # Before KB: free propagation from BM source with emittance (sigma, sigma')
    # After KB1: beam converges vertically to sample
    # After KB2: beam converges horizontally to sample
    def envelope_v(y):
        """Vertical ±1 sigma [m] at position y (lab frame)."""
        if y <= m.D_KB1:
            # Free propagation from source
            return np.sqrt(m.SIGMA_Y**2 + (m.SIGMA_YP * y)**2)
        else:
            # KB1 focuses: beam converges from KB1 to sample
            # Linear interpolation between size at KB1 entrance and ~0 at sample
            s_kb1 = np.sqrt(m.SIGMA_Y**2 + (m.SIGMA_YP * m.D_KB1)**2)
            frac  = (y - m.D_KB1) / g['L_KB1_SA']
            # At focus: sigma = SIGMA_Y * q_KB1/p_KB1
            s_foc = m.SIGMA_Y * g['L_KB1_SA'] / g['L_SL2_KB1']
            return s_kb1 * (1 - frac) + s_foc * frac

    def envelope_h(y):
        """Horizontal ±1 sigma [m] at position y (lab frame)."""
        if y <= m.D_KB2:
            return np.sqrt(m.SIGMA_X**2 + (m.SIGMA_XP * y)**2)
        else:
            s_kb2 = np.sqrt(m.SIGMA_X**2 + (m.SIGMA_XP * m.D_KB2)**2)
            frac  = (y - m.D_KB2) / g['L_KB2_SA']
            s_foc = m.SIGMA_X * g['L_KB2_SA'] / g['L_SL2_KB2']
            return s_kb2 * (1 - frac) + s_foc * frac

    # Sample positions
    y_pts  = np.linspace(0, m.D_SA, n_samples)
    cz     = np.array([centroid_at(y)[0] for y in y_pts]) * 1e3   # mm
    cx     = np.array([centroid_at(y)[1] for y in y_pts]) * 1e3
    env_v  = np.array([envelope_v(y)     for y in y_pts]) * 1e3
    env_h  = np.array([envelope_h(y)     for y in y_pts]) * 1e3

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, (ax_side, ax_top) = plt.subplots(2, 1, figsize=figsize,
                                           sharex=True)
    fig.suptitle("BM32 beam path  —  source to sample", fontsize=12)

    BEAM_COL   = "#e07b39"
    MIRROR_COL = "#4a90d9"
    SLIT_COL   = "#555555"

    for ax, cen, env, ylabel, view in [
        (ax_side, cz, env_v, "Z vertical (mm)", "side"),
        (ax_top,  cx, env_h, "X horizontal (mm)", "top"),
    ]:
        # Beam envelope
        ax.fill_between(y_pts, cen - env, cen + env,
                        color=BEAM_COL, alpha=0.35, label="Beam ±1σ")
        ax.plot(y_pts, cen + env, color=BEAM_COL, lw=1.0, alpha=0.7)
        ax.plot(y_pts, cen - env, color=BEAM_COL, lw=1.0, alpha=0.7)
        ax.plot(y_pts, cen,       color=BEAM_COL, lw=1.5, label="Centroid")

        # Optical axis (dashed)
        ax.axhline(0, color="gray", lw=0.7, ls="--", alpha=0.5)

        # Draw elements
        for (pos, name, etype) in elements:
            if etype == 'slit':
                # Vertical line + jaw markers
                ax.axvline(pos, color=SLIT_COL, lw=1.0, ls=":", alpha=0.7)
                z0, x0 = centroid_at(pos)
                c = (z0 if view == "side" else x0) * 1e3
                gap = (m.SL1_V if name == "SL1" else
                       m.SL2_V if name == "SL2" else m.SL3_V) if view == "side" else \
                      (m.SL1_H if name == "SL1" else
                       m.SL2_H if name == "SL2" else m.SL3_H)
                h = gap/2 * 1e3
                ax.annotate("", xy=(pos, c+h+0.3), xytext=(pos, c+h+1.2),
                             arrowprops=dict(arrowstyle="-|>", color=SLIT_COL, lw=1.2))
                ax.annotate("", xy=(pos, c-h-0.3), xytext=(pos, c-h-1.2),
                             arrowprops=dict(arrowstyle="-|>", color=SLIT_COL, lw=1.2))
                ax.text(pos, ax.get_ylim()[1] if ax.get_ylim()[1] != 1.0 else 2,
                        name, ha='center', va='bottom', fontsize=7,
                        color=SLIT_COL)

            elif etype in ('mirror_v', 'mirror_h'):
                active = (etype == 'mirror_v' and view == 'side') or \
                         (etype == 'mirror_h' and view == 'top')
                col   = MIRROR_COL if active else "lightgray"
                lw    = 2.0 if active else 0.8
                z0, x0 = centroid_at(pos)
                c   = (z0 if view == "side" else x0) * 1e3
                if name in ('M1', 'M2'):
                    half = m.M1_LENGTH/2 * np.sin(m.G_M1 if name=='M1' else m.G_M2) * 1e3
                elif name == 'KB1':
                    half = m.KB1_LENGTH/2 * np.sin(m.G_KB1) * 1e3
                else:
                    half = m.KB2_LENGTH/2 * np.sin(m.G_KB2) * 1e3
                ax.plot([pos, pos], [c - half, c + half],
                        color=col, lw=lw, solid_capstyle='round')
                ax.text(pos + 0.05, c + half + 0.2, name,
                        ha='left', va='bottom', fontsize=7, color=col)

            elif etype == 'S':
                z0, x0 = centroid_at(pos)
                c = (z0 if view == "side" else x0) * 1e3
                ax.plot(pos, c, 'o', color="crimson" if name == "Sample" else "black",
                        ms=5, zorder=5)
                ax.text(pos + 0.1, c + 0.3, name, fontsize=7,
                        color="crimson" if name == "Sample" else "black")

        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(True, alpha=0.2)
        ax.set_xlim(-0.5, m.D_SA + 0.3)
        label = "Side view  (vertical plane)" if view == "side" \
                else "Top view  (horizontal plane)"
        ax.set_title(label, fontsize=9, loc="left")

    ax_top.set_xlabel("Distance from source  (m)", fontsize=9)

    # Shared legend
    from matplotlib.lines import Line2D
    legend_els = [
        Line2D([0],[0], color=BEAM_COL, lw=2, label="Beam envelope ±1σ"),
        Line2D([0],[0], color=MIRROR_COL, lw=2, label="Active mirror"),
        Line2D([0],[0], color="lightgray", lw=2, label="Inactive plane"),
        Line2D([0],[0], color=SLIT_COL, lw=1, ls=":", label="Slits"),
    ]
    ax_side.legend(handles=legend_els, fontsize=7, loc="upper right", ncol=4)

    plt.tight_layout()
    if save_fig:
        plt.savefig(save_fig, dpi=150)
        print(f"[plot_beam_path] Saved -> {save_fig}")
    _maybe_show(fig)
    return fig



    # Notebook-style step-by-step usage:
    #
    #   import beamline2 as bm
    #
    #   bm.G_KB1 = 2.5e-3              # override any parameter
    #   bm.SL2_H = 0.050e-3
    #   bm.SL3_H = 0.500e-3
    #   bm.D_SL3 = 44.200              # SL3 position
    #
    #   beam, norm = bm.source_bm32(nrays=500_000)
    #   beam_m1, _ = bm.element_m1(beam)
    #   beam_m2, _ = bm.element_m2(beam_m1)
    #   beam_sl2   = bm.element_slit(beam_m2, bm.SL2_H, bm.SL2_V,
    #                                 p=bm._geo()['L_M2_SL2'], label="SL2")
    #   beam_sl3   = bm.element_slit(beam_sl2, bm.SL3_H, bm.SL3_V,
    #                                 p=bm._geo()['L_SL2_SL3'], label="SL3")
    #   # Inspect beam just before KB1:
    #   bm.plot_beam(bm.beam_at_distance(beam_sl3, bm._geo()['L_SL3_KB1']),
    #                "At KB1 entrance")
    #   beam_kb = bm.set_kb_source_from_beam(beam_sl3, nrays=500_000)
    #   beam_kb1, fp1 = bm.element_kb1(beam_kb)
    #   beam_kb2, fp2 = bm.element_kb2(beam_kb1)
    #   bm.plot_footprint(fp1, "KB1"); bm.plot_footprint(fp2, "KB2")
    #   bm.plot_beam(beam_kb2, "At sample")

    r = run_full_kb_chain(nrays_bm=200_000, nrays_kb=100_000,
                          plot_each=False, plot_final=True)