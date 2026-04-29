"""
BM32 beamline - element-by-element API
=======================================

Each optical element is a standalone function:

    beam = source_bm32(nrays=500_000)
    beam = element_m1(beam)
    beam = element_m2(beam)
    beam = element_slits2(beam)        # (clips beam, no reflection)
    beam = element_kb1(beam)
    beam = element_kb2(beam)

Every function also returns the beam so you can chain or inspect at any point.
All geometry variables are module-level constants - edit them at the top of
this file (or override from a notebook) to match motor readbacks.

Notebook usage
--------------
    bm = _self()

    # Override any parameter before running
    bm.G_M1 = 3.1e-3         # update M1 angle from motor
    bm.SL2_H = 0.050e-3      # update slit opening

    # Run element by element
    beam0 = bm.source_bm32(nrays=200_000)
    beam1 = bm.element_m1(beam0)
    beam2 = bm.element_m2(beam1)
    beam3 = bm.element_slits2(beam2)
    beam4 = bm.element_kb1(beam3)
    beam5 = bm.element_kb2(beam4)

    # Inspect any beam
    bm.plot_beam(beam3, label='At SL2')
    bm.plot_beam(beam5, label='At sample')
    bm.plot_spectrum(beam5, label='Sample spectrum')
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

# -- notebook / script compatibility -----------------------------------------
try:
    from IPython import get_ipython as _get_ipython
    INLINE_PLOTS = _get_ipython() is not None
except ImportError:
    INLINE_PLOTS = False

def _maybe_show(fig=None):
    if not INLINE_PLOTS:
        plt.show()


def _self():
    """Return this module object - works regardless of filename or package path."""
    import sys
    return sys.modules[__name__]

# ==============================================================================
# GEOMETRY - edit these to match motor readbacks
# ==============================================================================

# -- Longitudinal positions [m from source] ------------------------------------
D_SL1  = 26.368   # m  Slits 1 (upstream aperture)
D_M1   = 28.309   # m  Mirror 1 centre
D_M2   = 31.732   # m  Mirror 2 centre
D_SL2  = 35.370   # m  Slits 2 / mu-slits / secondary source
D_KB1  = 44.645   # m  KB1 centre (vertical focus)
D_KB2  = 44.880   # m  KB2 centre (horizontal focus)
D_SA   = 45.000   # m  Sample (LaueMAX)

# -- Derived propagation distances --------------------------------------------
# These are computed from the D_* globals at call time via _geo().
# Do NOT use these module-level names directly inside element functions -
# always call _geo() so that mid-session overrides (e.g. bm.D_KB1 = 44.700)
# are picked up immediately.
def _geo():
    """Return a dict of all derived distances from current module globals."""
    _bm = _self()
    return dict(
        L_SRC_M1  = _bm.D_M1,
        L_M1_M2   = _bm.D_M2  - _bm.D_M1,
        L_M2_SL2  = _bm.D_SL2 - _bm.D_M2,
        L_SL2_KB1 = _bm.D_KB1 - _bm.D_SL2,
        L_KB1_KB2 = _bm.D_KB2 - _bm.D_KB1,
        L_KB2_SA  = _bm.D_SA  - _bm.D_KB2,
        L_SL2_KB2 = _bm.D_KB2 - _bm.D_SL2,
        L_KB1_SA  = _bm.D_SA  - _bm.D_KB1,
        # p_focus for KB mirrors = distance from BM source (flat M1+M2)
        P_KB1     = _bm.D_KB1,              # 44.645 m
        P_KB2     = _bm.D_KB2,              # 44.880 m
        # SL2 angular half-acceptance as seen from BM source
        A_SL2_H   = _bm.SL2_H / _bm.D_SL2,  # rad
        A_SL2_V   = _bm.SL2_V / _bm.D_SL2,  # rad
        DZ_M1_M2  = _bm.DZ_M1_M2,
        DZ_AT_SL2 = _bm.DZ_M1_M2 - 2.0 * _bm.G_M2 * (_bm.D_SL2 - _bm.D_M2),
    )

# Module-level aliases for backward compatibility (not used inside element fns)
L_SRC_M1  = D_M1
L_M1_M2   = D_M2  - D_M1
L_M2_SL2  = D_SL2 - D_M2
L_SL2_KB1 = D_KB1 - D_SL2
L_KB1_KB2 = D_KB2 - D_KB1
L_KB2_SA  = D_SA  - D_KB2
L_SL2_KB2 = D_KB2 - D_SL2
L_KB1_SA  = D_SA  - D_KB1

# -- Grazing angles [rad] - replace with motor-derived values -----------------
G_M1  = 3.062181698459972e-3   # rad  M1 (from M1 pitch motor)
G_M2  = 2.5627372258861216e-3  # rad  M2 (from M2 pitch motor)
G_KB1 = 2.2e-3                 # rad  KB1 (estimated; confirm with BM32 team)
G_KB2 = 13.97e-3               # rad  KB2 (estimated; confirm with BM32 team)

# -- Mirror bending radii [m] - from H5 / motor encoder ----------------------
R_M1 = 2327.636   # m
R_M2 = 1766.232   # m

# -- Vertical offset M1->M2 in lab frame [m] -----------------------------------
# M1 deflects beam UPWARD; M2 is positioned higher and deflects back DOWN.
DZ_M1_M2  = 0.02108279594058422  # m  (from height motor: M2_z - M1_z)
DZ_AT_SL2 = DZ_M1_M2 - 2.0 * G_M2 * L_M2_SL2   # net offset at SL2 (~0)

# -- Mirror physical dimensions [m] -------------------------------------------
M1_LENGTH  = 1.100   # m  useful tangential length
M1_WIDTH   = 0.050   # m  sagittal width (assumed 50 mm)
M2_LENGTH  = 1.100   # m
M2_WIDTH   = 0.050   # m
KB1_LENGTH = 0.300   # m
KB1_WIDTH  = 0.020   # m  (assumed 20 mm)
KB2_LENGTH = 0.150   # m
KB2_WIDTH  = 0.020   # m

# -- Slit apertures [m half-opening] - replace with motor-derived values -------
SL1_H = 5.000e-3    # m  Slits 1 horizontal half-opening
SL1_V = 2.000e-3    # m  Slits 1 vertical   half-opening
SL2_H = 0.100e-3    # m  Slits 2 horizontal half-opening
SL2_V = 0.100e-3    # m  Slits 2 vertical   half-opening

# -- Mirror curvature flag ----------------------------------------------------
# Set to True if the bending motors are active and mirrors are curved.
# Set to False (default) when mirrors are flat (no bending applied).
MIRROR_CURVED = False

# -- Energy range [eV] --------------------------------------------------------
E_MIN = 5_000.0
E_MAX = 35_000.0

# -- ESRF EBS electron beam (SBM32) -------------------------------------------
E_GEV         = 6.04       # GeV
CURRENT_A     = 0.2        # A
ENERGY_SPREAD = 9.3e-4
SIGMA_X       = 30.1e-6    # m  horizontal beam size
SIGMA_XP      = 4.2e-6     # rad
SIGMA_Y       = 3.6e-6     # m  vertical beam size
SIGMA_YP      = 1.4e-6     # rad
BM_RADIUS     = 23.588     # m
BM_FIELD      = 0.857      # T
BM_LENGTH     = 0.1        # m  effective magnet length

# -- Spectrum at SL2 -----------------------------------------------------------
# Set by running compute_spectrum_at_sl2(), or assign manually:
#   bm.SPECTRUM_ENERGY_EV = np.array([...])   # energy bin centres [eV]
#   bm.SPECTRUM_FLUX      = np.array([...])   # relative flux weights (any units)
# element_kb_source() uses these automatically; falls back to uniform if None.
SPECTRUM_ENERGY_EV = None
SPECTRUM_FLUX      = None


# ==============================================================================
# HELPERS
# ==============================================================================

def _ar(grazing_rad):
    """angle_radial = pi/2 - grazing  (Shadow4 convention)."""
    return np.pi / 2 - grazing_rad


def _ir_reflectivity(energies_eV, grazing_mrad, roughness_A=3.0):
    rs, rp = PreRefl.reflectivity_amplitudes_fresnel_external_xraylib(
        photon_energy_ev=energies_eV, coating_material="Ir",
        coating_density=22.56, grazing_angle_mrad=grazing_mrad,
        roughness_rms_A=roughness_A)
    return 0.5 * (np.abs(rs)**2 + np.abs(rp)**2)


def _aperture(width, length):
    return Rectangle(-width/2, width/2, -length/2, length/2)


def _good(beam):
    """Return surviving rays from beam."""
    return beam.rays[beam.rays[:, 9] > 0]


def _print_geometry():
    """Print current geometry - always reflects latest module variable values."""
    g = _geo()
    _bm = _self()
    print("=" * 60)
    print("BM32 WHITE-BEAM GEOMETRY  (current values)")
    print("=" * 60)
    print(f"  SL1 : D={_bm.D_SL1:.3f} m  H=±{_bm.SL1_H*1e3:.2f} mm  V=±{_bm.SL1_V*1e3:.2f} mm")
    print(f"  M1  : D={_bm.D_M1:.3f} m  G={_bm.G_M1*1e3:.4f} mrad  R={_bm.R_M1:.3f} m  "
          f"p={g['L_SRC_M1']:.3f} m  q={_bm.D_SL2-_bm.D_M1:.3f} m")
    print(f"  M2  : D={_bm.D_M2:.3f} m  G={_bm.G_M2*1e3:.4f} mrad  R={_bm.R_M2:.3f} m  "
          f"p={_bm.D_M2:.3f} m  q={g['L_M2_SL2']:.3f} m")
    print(f"  DZ(M1->M2) = {g['DZ_M1_M2']*1e3:.2f} mm  |  "
          f"DZ(net at SL2) = {g['DZ_AT_SL2']*1e3:.3f} mm")
    print(f"  SL2 : D={_bm.D_SL2:.3f} m  H=±{_bm.SL2_H*1e3:.3f} mm  V=±{_bm.SL2_V*1e3:.3f} mm")
    print(f"  KB1 : D={_bm.D_KB1:.3f} m  G={_bm.G_KB1*1e3:.3f} mrad  "
          f"p={g['L_SL2_KB1']:.3f} m  q={g['L_KB1_SA']:.3f} m  "
          f"demag={g['L_KB1_SA']/g['L_SL2_KB1']:.4f}")
    print(f"  KB2 : D={_bm.D_KB2:.3f} m  G={_bm.G_KB2*1e3:.3f} mrad  "
          f"p={g['L_SL2_KB2']:.3f} m  q={g['L_KB2_SA']:.3f} m  "
          f"demag={g['L_KB2_SA']/g['L_SL2_KB2']:.4f}")
    print(f"  SA  : D={_bm.D_SA:.3f} m")
    print("=" * 60)


# ==============================================================================
# PLOTTING UTILITIES
# ==============================================================================

def plot_beam(beam, label="beam", position_m=None, n_bins=100, figsize=(12, 5)):
    """
    Two-panel diagnostic for any beam.

    Left  - transverse cross-section: col1(H) vs col3(V), colour = energy
    Right - energy spectrum of surviving rays
    """
    g = _good(beam)
    if len(g) == 0:
        print(f"  [plot_beam] {label}: no surviving rays")
        return

    e_keV = g[:, 10] * 197.3e-9 / 1e3
    x_mm  = g[:, 0] * 1e3
    z_mm  = g[:, 2] * 1e3

    title_pos = f"  at z={position_m:.3f} m" if position_m is not None else ""
    fig, (ax_spot, ax_spec) = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(f"{label}{title_pos}  -  {len(g)} rays", fontsize=11)

    # Cross-section
    sc = ax_spot.scatter(x_mm, z_mm, s=0.5, alpha=0.4, c=e_keV, cmap="plasma")
    fig.colorbar(sc, ax=ax_spot, label="E (keV)")
    ax_spot.set_xlabel("col1  ->  lab horizontal (mm)")
    ax_spot.set_ylabel("col3  ->  lab vertical (mm)")
    ax_spot.set_title(
        f"Beam cross-section\n"
        f"H FWHM={x_mm.std()*2.355:.3f} mm  |  V FWHM={z_mm.std()*2.355:.3f} mm"
    )
    ax_spot.set_aspect("equal", adjustable="datalim")
    ax_spot.grid(True, alpha=0.3)

    # Spectrum
    ax_spec.hist(e_keV, bins=n_bins,
                 range=(E_MIN/1e3, E_MAX/1e3),
                 weights=g[:, 10] * 197.3e-9 / 1e3,   # weight by energy for visual
                 color="crimson", alpha=0.7)
    ax_spec.set_xlabel("Photon energy (keV)")
    ax_spec.set_ylabel("Ray count (weighted)")
    ax_spec.set_title("Energy spectrum")
    ax_spec.set_xlim(E_MIN/1e3, E_MAX/1e3)
    ax_spec.grid(True, alpha=0.3)

    plt.tight_layout()
    _maybe_show(fig)
    return fig


def plot_spectrum(beam, label="spectrum", norm_factor=1.0,
                  n_bins=200, figsize=(8, 5)):
    """
    Absolute spectral flux plot for a beam.

    norm_factor : ph/s per ray (from source_bm32 return value).
                  If 1.0, y-axis is in arbitrary units.
    """
    g = _good(beam)
    if len(g) == 0:
        print(f"  [plot_spectrum] {label}: no surviving rays")
        return

    e_eV = beam.get_column(26, nolost=1)
    i    = beam.get_column(23, nolost=1)
    counts, edges = np.histogram(e_eV, bins=n_bins,
                                 range=(E_MIN, E_MAX), weights=i)
    en     = 0.5*(edges[:-1]+edges[1:])
    dE     = (E_MAX - E_MIN) / n_bins
    flux   = counts * norm_factor / dE   # ph/s/eV  (or a.u./eV if norm=1)
    unit   = "ph/s/eV" if norm_factor != 1.0 else "a.u./eV"
    peak   = en[np.argmax(flux)] / 1e3

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(en/1e3, flux, color="crimson", lw=2, label=label)
    ax.axvline(peak, color="darkred", lw=0.8, ls="--",
               label=f"peak {peak:.1f} keV")
    ax.set_xlabel("Photon energy (keV)")
    ax.set_ylabel(f"Spectral flux ({unit})")
    ax.set_title(f"Spectrum  -  {label}")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(E_MIN/1e3, E_MAX/1e3)
    ax.set_ylim(0)
    plt.tight_layout()
    _maybe_show(fig)
    return fig


def plot_footprint(footprint_beam, label="footprint", figsize=(7, 6)):
    """Mirror footprint in local (sagittal x tangential) frame."""
    g = _good(footprint_beam)
    if len(g) == 0:
        print(f"  [plot_footprint] {label}: no rays")
        return
    e_keV = g[:, 10] * 197.3e-9 / 1e3


def beam_at_distance(beam, distance_m):
    """
    Propagate a beam forward by distance_m along its current direction.

    Returns a new beam with ray positions updated to the new plane.
    Use this to get the beam cross-section at any position between elements.

    Parameters
    ----------
    beam       : S4Beam  — beam to propagate
    distance_m : float   — distance to propagate [m]

    Returns
    -------
    S4Beam with updated positions

    Example
    -------
    # Get beam cross-section 5 m after SL2 (i.e. at D_SL2 + 5 m)
    beam_m2, _ = bm.element_m2(beam_m1)
    beam_5m    = bm.beam_at_distance(beam_m2, 5.0)
    bm.plot_beam(beam_5m, label=f"5 m after SL2")

    # Get beam just before KB1 (propagate from SL2 to KB1)
    beam_at_kb1 = bm.beam_at_distance(beam_m2, bm.L_SL2_KB1)
    bm.plot_beam(beam_at_kb1, label="At KB1 entrance")
    """
    from shadow4.beam.s4_beam import S4Beam
    import copy
    new_rays = beam.rays.copy()
    good = new_rays[:, 9] > 0
    # Propagate: new_pos = old_pos + (direction / |direction|) * distance
    # In shadow4 frame: propagation along Y, so dy = distance * py/|p|
    # |p| = 1 by convention, so new_y = old_y + py * distance
    # But px, pz also shift the transverse position:
    py = new_rays[good, 4]
    py_safe = np.where(np.abs(py) > 1e-10, py, 1e-10)
    new_rays[good, 0] += new_rays[good, 3] / py_safe * distance_m   # x
    new_rays[good, 1] += distance_m                                   # y
    new_rays[good, 2] += new_rays[good, 5] / py_safe * distance_m   # z
    beam_new = S4Beam()
    beam_new.rays = new_rays
    return beam_new


def plot_footprint(footprint_beam, label="footprint", figsize=(7, 6)):
    """Mirror footprint in local (sagittal x tangential) frame."""
    g = _good(footprint_beam)
    if len(g) == 0:
        print(f"  [plot_footprint] {label}: no rays")
        return
    e_keV = g[:, 10] * 197.3e-9 / 1e3
    fig, ax = plt.subplots(figsize=figsize)
    sc = ax.scatter(g[:, 0]*1e3, g[:, 1]*1e3,
                    s=0.4, alpha=0.3, c=e_keV, cmap="plasma")
    plt.colorbar(sc, ax=ax, label="E (keV)")
    ax.set_xlabel("x sagittal (mm)")
    ax.set_ylabel("y tangential (mm)")
    ax.set_title(f"{label}  ({len(g)} rays)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _maybe_show(fig)
    return fig


# ==============================================================================
# BEAMLINE ELEMENTS - each takes a beam in, returns a beam out
# ==============================================================================

def source_bm32(nrays=500_000, seed=5676561):
    """
    Sample rays from the SBM32 bending-magnet source.

    Returns
    -------
    beam       : S4Beam
    norm_factor: float  ph/s per surviving ray  (use for absolute flux)
    """
    _print_geometry()
    print(f"\n[Source] SBM32  ({nrays} rays) ...")
    ebeam = S4ElectronBeam(energy_in_GeV=E_GEV,
                           energy_spread=ENERGY_SPREAD, current=CURRENT_A)
    ebeam.set_sigmas_all(sigma_x=SIGMA_X, sigma_xp=SIGMA_XP,
                         sigma_y=SIGMA_Y,  sigma_yp=SIGMA_YP)
    bm = S4BendingMagnet(radius=BM_RADIUS, magnetic_field=BM_FIELD,
                         length=BM_LENGTH,
                         emin=E_MIN, emax=E_MAX, ng_e=200, flag_emittance=1)
    source = S4BendingMagnetLightSource(
        name="SBM32", electron_beam=ebeam,
        magnetic_structure=bm, nrays=nrays, seed=seed)
    beam = source.get_beam()

    from scipy.integrate import trapezoid as trapz
    flux_per_E  = trapz(source.tot, source.angle_array_mrad*1e-3, axis=0)
    total_flux  = trapz(flux_per_E, source.photon_energy_array)
    norm_factor = total_flux / nrays
    print(f"       Total BM flux in cone : {total_flux:.4e} ph/s")
    print(f"       Norm factor           : {norm_factor:.4e} ph/s/ray")
    print(f"       Rays sampled          : {(_good(beam).shape[0])}")
    return beam, norm_factor


_bm_ = _self()
g = _geo()
def element_slits1(beam):
    """
    Apply Slits 1 aperture (upstream white-beam slits).

    Clips rays outside ±_bm_.SL1_H x ±_bm_.SL1_V in the lab frame at D_SL1.
    No reflection - rays outside are flagged as lost.
    """
    print(f"\n[SL1] Slits 1  D={_bm_.D_SL1:.3f} m  "
          f"H=±{_bm_.SL1_H*1e3:.2f} mm  V=±{_bm_.SL1_V*1e3:.2f} mm ...")
    from shadow4.beamline.optical_elements.absorbers.s4_screen import (
        S4Screen, S4ScreenElement)
    screen = S4Screen(name="SL1",
                      boundary_shape=_aperture(2*_bm_.SL1_H, 2*_bm_.SL1_V))
    coords = ElementCoordinates(p=_bm_.D_SL1, q=0.)
    beam_out, _ = S4ScreenElement(optical_element=screen,
                                  coordinates=coords,
                                  input_beam=beam).trace_beam()
    n_in  = (_good(beam)).shape[0]
    n_out = (_good(beam_out)).shape[0]
    print(f"       {n_out} / {n_in} rays survive  ({100*n_out/max(n_in,1):.1f}%)")
    return beam_out


_bm_ = _self()
g = _geo()
def element_m1(beam):
    """
    Mirror 1 - Ir bent cylinder, deflects beam UPWARD.

    azimuthal = 0  ->  incidence plane is vertical (Y-Z).
    p_coord   = g['L_SRC_M1']   (from source to M1)
    q_coord   = g['L_M1_M2']    (from M1 to M2, for coordinate frame propagation)
    p_focus   = g['L_SRC_M1']   (ellipse shape: focuses source to SL2)
    q_focus   = _bm_.D_SL2-_bm_.D_M1 (ellipse shape: image at SL2)
    """
    curved = _bm_.MIRROR_CURVED
    print(f"\n[M1] Mirror 1  D={_bm_.D_M1:.3f} m  G={_bm_.G_M1*1e3:.4f} mrad  "
          f"L={_bm_.M1_LENGTH*1e3:.0f} mm  "
          f"({'bent cylinder' if curved else 'FLAT - no bending'}) ...")
    if curved:
        mirror = S4EllipsoidMirror(
            name="M1", boundary_shape=_aperture(_bm_.M1_WIDTH, _bm_.M1_LENGTH),
            surface_calculation=SurfaceCalculation.INTERNAL,
            p_focus=g['L_SRC_M1'], q_focus=_bm_.D_SL2-_bm_.D_M1,
            grazing_angle=_bm_.G_M1, is_cylinder=True,
            cylinder_direction=Direction.TANGENTIAL, convexity=Convexity.UPWARD,
            f_reflec=1, f_refl=5, file_refl="",
            coating_material="Ir", coating_density=22.56, coating_roughness=3.0)
        coords = ElementCoordinates(
            p=g['L_SRC_M1'], q=g['L_M1_M2'],
            angle_radial=_ar(_bm_.G_M1), angle_azimuthal=0.0)
        beam_out, footprint = S4EllipsoidMirrorElement(
            optical_element=mirror, coordinates=coords,
            input_beam=beam).trace_beam()
    else:
        mirror = S4PlaneMirror(
            name="M1", boundary_shape=_aperture(_bm_.M1_WIDTH, _bm_.M1_LENGTH),
            f_reflec=1, f_refl=5, file_refl="",
            coating_material="Ir", coating_density=22.56, coating_roughness=3.0)
        coords = ElementCoordinates(
            p=g['L_SRC_M1'], q=g['L_M1_M2'],
            angle_radial=_ar(_bm_.G_M1), angle_azimuthal=0.0)
        beam_out, footprint = S4PlaneMirrorElement(
            optical_element=mirror, coordinates=coords,
            input_beam=beam).trace_beam()
    n_out = _good(beam_out).shape[0]
    n_in  = _good(beam).shape[0]
    print(f"       {n_out} / {n_in} rays survive  ({100*n_out/max(n_in,1):.1f}%)")
    print(f"       Beam rises by {2*_bm_.G_M1*g['L_M1_M2']*1e3:.2f} mm over {g['L_M1_M2']:.3f} m to M2")
    return beam_out, footprint


_bm_ = _self()
g = _geo()
def element_m2(beam):
    """
    Mirror 2 - Ir bent cylinder, deflects beam DOWNWARD (restores direction).

    azimuthal = pi  ->  incidence plane is vertical but mirror faces up,
                        so deflection is downward (compensates M1).
    p_coord   = g['L_M1_M2']    (drift from M1)
    q_coord   = g['L_M2_SL2']   (drift to SL2)
    p_focus   = _bm_.D_M2        (from source to M2)
    q_focus   = g['L_M2_SL2']   (from M2 to SL2)
    """
    curved = _bm_.MIRROR_CURVED
    print(f"\n[M2] Mirror 2  D={_bm_.D_M2:.3f} m  G={_bm_.G_M2*1e3:.4f} mrad  "
          f"L={_bm_.M2_LENGTH*1e3:.0f} mm  "
          f"({'bent cylinder' if curved else 'FLAT - no bending'}) ...")
    if curved:
        mirror = S4EllipsoidMirror(
            name="M2", boundary_shape=_aperture(_bm_.M2_WIDTH, _bm_.M2_LENGTH),
            surface_calculation=SurfaceCalculation.INTERNAL,
            p_focus=_bm_.D_M2, q_focus=g['L_M2_SL2'],
            grazing_angle=_bm_.G_M2, is_cylinder=True,
            cylinder_direction=Direction.TANGENTIAL, convexity=Convexity.UPWARD,
            f_reflec=1, f_refl=5, file_refl="",
            coating_material="Ir", coating_density=22.56, coating_roughness=3.0)
        coords = ElementCoordinates(
            p=g['L_M1_M2'], q=g['L_M2_SL2'],
            angle_radial=_ar(_bm_.G_M2), angle_azimuthal=np.pi)
        beam_out, footprint = S4EllipsoidMirrorElement(
            optical_element=mirror, coordinates=coords,
            input_beam=beam).trace_beam()
    else:
        mirror = S4PlaneMirror(
            name="M2", boundary_shape=_aperture(_bm_.M2_WIDTH, _bm_.M2_LENGTH),
            f_reflec=1, f_refl=5, file_refl="",
            coating_material="Ir", coating_density=22.56, coating_roughness=3.0)
        coords = ElementCoordinates(
            p=g['L_M1_M2'], q=g['L_M2_SL2'],
            angle_radial=_ar(_bm_.G_M2), angle_azimuthal=np.pi)
        beam_out, footprint = S4PlaneMirrorElement(
            optical_element=mirror, coordinates=coords,
            input_beam=beam).trace_beam()
    n_out = _good(beam_out).shape[0]
    n_in  = _good(beam).shape[0]
    print(f"       {n_out} / {n_in} rays survive  ({100*n_out/max(n_in,1):.1f}%)")
    print(f"       Net vertical offset at SL2 ~ {DZ_AT_SL2*1e3:.3f} mm")
    return beam_out, footprint


_bm_ = _self()
g = _geo()
def element_slits2(beam):
    """
    Slits 2 / mu-slits - secondary source aperture at D_SL2.

    Clips to ±_bm_.SL2_H x ±SL2_V.  No reflection.
    """
    print(f"\n[SL2] Slits 2 (mu-slits)  D={_bm_.D_SL2:.3f} m  "
          f"H=±{_bm_.SL2_H*1e3:.3f} mm  V=±{_bm_.SL2_V*1e3:.3f} mm ...")
    # ⚠️  STATISTICS WARNING
    # The SL2 aperture (100 µm x 100 µm) is tiny compared to the beam after M2
    # (~30 mm wide x 4 mm tall). Only ~0.003% of rays pass through.
    # With 30k source rays only ~1 ray survives - you need ~1M+ rays for useful
    # statistics through the full BM -> M1 -> M2 -> SL2 -> KB chain.
    # For KB focus studies, use the standalone KB source instead (element_kb_source).
    from shadow4.beamline.optical_elements.absorbers.s4_screen import (
        S4Screen, S4ScreenElement)
    screen = S4Screen(name="SL2",
                      boundary_shape=_aperture(2*_bm_.SL2_H, 2*_bm_.SL2_V))
    coords = ElementCoordinates(p=g['L_M2_SL2'], q=0.)
    beam_out, _ = S4ScreenElement(optical_element=screen,
                                  coordinates=coords,
                                  input_beam=beam).trace_beam()
    n_in  = _good(beam).shape[0]
    n_out = _good(beam_out).shape[0]
    survival_pct = 100*n_out/max(n_in,1)
    print(f"       {n_out} / {n_in} rays survive  ({survival_pct:.3f}%)")
    if n_out < 10:
        needed = max(int(n_in / max(survival_pct/100, 1e-8)), 1_000_000)
        print(f"  ⚠️  Too few survivors for reliable statistics.")
        print(f"       Run source_bm32(nrays>={needed:,}) for useful KB statistics,")
        print(f"       or use element_kb_source() for a dedicated KB secondary source.")
    return beam_out



_bm_ = _self()
g = _geo()
def compute_spectrum_at_sl2(nrays=500_000, n_bins=300,
                             plot=True, save_fig=""):
    """
    Compute the spectral flux at SL2 (after M1 + M2 Ir reflectivity).

    Traces the BM source through M1 and M2, histograms the surviving ray
    energies weighted by intensity, and stores the result in the module-level
    variables SPECTRUM_ENERGY_EV and SPECTRUM_FLUX.

    element_kb_source() then uses those arrays via
    set_energy_distribution_userdefined() so each KB ray is assigned an energy
    drawn from the physically correct M1+M2-filtered spectrum rather than a
    flat uniform distribution.

    Call this once at the start of your notebook session, or whenever you
    change G_M1, G_M2 or mirror geometry.

    Parameters
    ----------
    nrays   : BM source rays to trace (more = smoother spectrum)
    n_bins  : histogram bins
    plot    : show spectrum figure inline
    save_fig: filename to save figure ('' = don't save)

    Returns
    -------
    energy_eV : np.ndarray  bin centres [eV]
    flux      : np.ndarray  normalised flux weights

    Example
    -------
    import beamline_elements as bm
    bm.compute_spectrum_at_sl2(nrays=200_000)
    beam = bm.element_kb_source(nrays=500_000)   # uses real spectrum
    """
    _bm_ = _self()
    g    = _geo()

    print(f"\n[Spectrum@SL2] Tracing BM -> M1 -> M2  ({nrays} rays)  "
          f"({'curved' if _bm_.MIRROR_CURVED else 'flat'} mirrors) ...")

    # BM source
    ebeam = S4ElectronBeam(energy_in_GeV=_bm_.E_GEV,
                           energy_spread=_bm_.ENERGY_SPREAD,
                           current=_bm_.CURRENT_A)
    ebeam.set_sigmas_all(sigma_x=_bm_.SIGMA_X, sigma_xp=_bm_.SIGMA_XP,
                         sigma_y=_bm_.SIGMA_Y,  sigma_yp=_bm_.SIGMA_YP)
    bm_src = S4BendingMagnet(
        radius=_bm_.BM_RADIUS, magnetic_field=_bm_.BM_FIELD,
        length=_bm_.BM_LENGTH,
        emin=_bm_.E_MIN, emax=_bm_.E_MAX, ng_e=200, flag_emittance=1)
    light_source = S4BendingMagnetLightSource(
        name="SBM32", electron_beam=ebeam,
        magnetic_structure=bm_src, nrays=nrays, seed=5676561)
    beam = light_source.get_beam()

    # M1 - flat or curved depending on MIRROR_CURVED

    # M1 - flat or curved depending on MIRROR_CURVED
    if _bm_.MIRROR_CURVED:
        m1 = S4EllipsoidMirror(
            name="M1", boundary_shape=_aperture(_bm_.M1_WIDTH, _bm_.M1_LENGTH),
            surface_calculation=SurfaceCalculation.INTERNAL,
            p_focus=g['L_SRC_M1'], q_focus=_bm_.D_SL2 - _bm_.D_M1,
            grazing_angle=_bm_.G_M1, is_cylinder=True,
            cylinder_direction=Direction.TANGENTIAL, convexity=Convexity.UPWARD,
            f_reflec=1, f_refl=5, file_refl="",
            coating_material="Ir", coating_density=22.56, coating_roughness=3.0)
        c1 = ElementCoordinates(p=g['L_SRC_M1'], q=g['L_M1_M2'],
                                 angle_radial=_ar(_bm_.G_M1), angle_azimuthal=0.0)
        beam, _ = S4EllipsoidMirrorElement(
            optical_element=m1, coordinates=c1, input_beam=beam).trace_beam()
    else:
        m1 = S4PlaneMirror(
            name="M1", boundary_shape=_aperture(_bm_.M1_WIDTH, _bm_.M1_LENGTH),
            f_reflec=1, f_refl=5, file_refl="",
            coating_material="Ir", coating_density=22.56, coating_roughness=3.0)
        c1 = ElementCoordinates(p=g['L_SRC_M1'], q=g['L_M1_M2'],
                                 angle_radial=_ar(_bm_.G_M1), angle_azimuthal=0.0)
        beam, _ = S4PlaneMirrorElement(
            optical_element=m1, coordinates=c1, input_beam=beam).trace_beam()
    print(f"       After M1: {_good(beam).shape[0]} rays  "
          f"({100*_good(beam).shape[0]/nrays:.1f}%)  "
          f"({'curved' if _bm_.MIRROR_CURVED else 'flat'})")

    # M2 - flat or curved
    if _bm_.MIRROR_CURVED:
        m2 = S4EllipsoidMirror(
            name="M2", boundary_shape=_aperture(_bm_.M2_WIDTH, _bm_.M2_LENGTH),
            surface_calculation=SurfaceCalculation.INTERNAL,
            p_focus=_bm_.D_M2, q_focus=g['L_M2_SL2'],
            grazing_angle=_bm_.G_M2, is_cylinder=True,
            cylinder_direction=Direction.TANGENTIAL, convexity=Convexity.UPWARD,
            f_reflec=1, f_refl=5, file_refl="",
            coating_material="Ir", coating_density=22.56, coating_roughness=3.0)
        c2 = ElementCoordinates(p=g['L_M1_M2'], q=g['L_M2_SL2'],
                                 angle_radial=_ar(_bm_.G_M2), angle_azimuthal=np.pi)
        beam, _ = S4EllipsoidMirrorElement(
            optical_element=m2, coordinates=c2, input_beam=beam).trace_beam()
    else:
        m2 = S4PlaneMirror(
            name="M2", boundary_shape=_aperture(_bm_.M2_WIDTH, _bm_.M2_LENGTH),
            f_reflec=1, f_refl=5, file_refl="",
            coating_material="Ir", coating_density=22.56, coating_roughness=3.0)
        c2 = ElementCoordinates(p=g['L_M1_M2'], q=g['L_M2_SL2'],
                                 angle_radial=_ar(_bm_.G_M2), angle_azimuthal=np.pi)
        beam, _ = S4PlaneMirrorElement(
            optical_element=m2, coordinates=c2, input_beam=beam).trace_beam()
    n_m2 = _good(beam).shape[0]
    print(f"       After M2: {n_m2} rays  ({100*n_m2/nrays:.1f}%)  "
          f"({'curved' if _bm_.MIRROR_CURVED else 'flat'})")

    # Apply SL2 angular clip: keep only rays within ±A_SL2_V and ±A_SL2_H
    # as seen from the BM source (position ~ 0, direction cosines px/py, pz/py)
    rays = beam.rays
    good = rays[:, 9] > 0
    px = rays[:, 3]; py = rays[:, 4]; pz = rays[:, 5]
    # Angular deviation from axis (in paraxial approximation px/py, pz/py)
    angle_h = np.abs(px / np.where(py > 0, py, 1e-10))
    angle_v = np.abs(pz / np.where(py > 0, py, 1e-10))
    sl2_pass = (angle_h <= g['A_SL2_H']) & (angle_v <= g['A_SL2_V'])
    # Flag rays that don't pass SL2 as lost
    rays[good & ~sl2_pass, 9] = -1
    n_sl2 = (rays[:, 9] > 0).sum()
    print(f"       After SL2 angular clip "
          f"(H=±{g['A_SL2_H']*1e6:.1f} µrad  V=±{g['A_SL2_V']*1e6:.1f} µrad): "
          f"{n_sl2} rays  ({100*n_sl2/nrays:.3f}%)")

    # Histogram energy, weighted by ray intensity - only rays passing SL2
    e_eV = beam.get_column(26, nolost=1)
    i    = beam.get_column(23, nolost=1)
    if len(e_eV) < 20:
        print(f"  ⚠️  Only {len(e_eV)} rays pass SL2 - increase nrays for reliable spectrum.")
        print(f"       Falling back to M2-exit spectrum (no SL2 clip).")
        # use all M2 rays for spectrum shape
        rays[:, 9] = np.where(rays[:, 9] < 0, -1, 1)  # restore
        e_eV = beam.get_column(26, nolost=1)
        i    = beam.get_column(23, nolost=1)
    counts, edges = np.histogram(
        e_eV, bins=n_bins, range=(_bm_.E_MIN, _bm_.E_MAX), weights=i)
    en   = 0.5 * (edges[:-1] + edges[1:])
    # Normalise to sum = 1 (set_energy_distribution_userdefined needs weights,
    # not absolute flux - the shape is what matters)
    total = counts.sum()
    flux  = counts / total if total > 0 else np.ones_like(counts) / n_bins

    peak_keV = en[np.argmax(flux)] / 1e3
    print(f"       Peak at {peak_keV:.1f} keV")

    # Store in module so element_kb_source picks it up automatically
    _bm_.SPECTRUM_ENERGY_EV = en
    _bm_.SPECTRUM_FLUX      = flux
    print(f"       SPECTRUM_ENERGY_EV / SPECTRUM_FLUX stored in module.")

    if plot:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(en/1e3, flux, color="royalblue", lw=2,
                label="At SL2  (after M1 + M2 Ir reflectivity)")
        ax.axvline(peak_keV, color="navy", lw=0.8, ls="--",
                   label=f"peak  {peak_keV:.1f} keV")
        ax.set_xlabel("Photon energy (keV)")
        ax.set_ylabel("Normalised spectral flux (a.u.)")
        ax.set_title("Spectrum at SL2  -  used as KB source energy distribution")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
        ax.set_xlim(_bm_.E_MIN/1e3, _bm_.E_MAX/1e3); ax.set_ylim(0)
        plt.tight_layout()
        if save_fig:
            plt.savefig(save_fig, dpi=150)
            print(f"       Figure saved -> {save_fig}")
        _maybe_show(fig)

    return en, flux


def set_spectrum_from_beam(beam, n_bins=300, plot=True, label="beam", save_fig=""):
    """
    Set the KB source energy distribution from any existing beam object.

    Use this when you already have a simulated beam (e.g. from a previous
    run of element_m2, element_slits2, or any other element) and want to
    use its energy distribution for the KB source - without re-running the
    full BM->M1->M2 chain.

    Stores the result in SPECTRUM_ENERGY_EV and SPECTRUM_FLUX so that
    the next call to element_kb_source() uses this distribution.

    Parameters
    ----------
    beam   : S4Beam  - any beam object with surviving rays
    n_bins : int     - number of histogram bins
    plot   : bool    - show the spectrum inline
    label  : str     - label for the plot title
    save_fig : str   - filename to save figure ('' = don't save)

    Returns
    -------
    energy_eV : np.ndarray   bin centres [eV]
    flux      : np.ndarray   normalised flux weights

    Example
    -------
    import beamline2 as bm

    # Suppose you ran M1+M2 in a previous cell:
    beam_src, norm = bm.source_bm32(nrays=500_000)
    beam_m1, _    = bm.element_m1(beam_src)
    beam_m2, _    = bm.element_m2(beam_m1)

    # Now store that spectrum for the KB:
    bm.set_spectrum_from_beam(beam_m2, label="After M2")

    # Next KB source call uses this spectrum automatically:
    beam_kb = bm.element_kb_source(nrays=500_000)
    """
    _bm_ = _self()

    e_eV = beam.get_column(26, nolost=1)
    i    = beam.get_column(23, nolost=1)

    if len(e_eV) == 0:
        raise ValueError("set_spectrum_from_beam: no surviving rays in beam.")

    counts, edges = np.histogram(
        e_eV, bins=n_bins,
        range=(_bm_.E_MIN, _bm_.E_MAX),
        weights=i)
    en    = 0.5 * (edges[:-1] + edges[1:])
    total = counts.sum()
    flux  = counts / total if total > 0 else np.ones_like(counts) / n_bins

    peak_keV = en[np.argmax(flux)] / 1e3
    n_rays   = len(e_eV)
    print(f"[set_spectrum_from_beam] '{label}'")
    print(f"       {n_rays} surviving rays   peak = {peak_keV:.1f} keV")
    print(f"       SPECTRUM_ENERGY_EV / SPECTRUM_FLUX updated.")

    _bm_.SPECTRUM_ENERGY_EV = en
    _bm_.SPECTRUM_FLUX      = flux

    if plot:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(en / 1e3, flux, color="royalblue", lw=2, label=label)
        ax.axvline(peak_keV, color="navy", lw=0.8, ls="--",
                   label=f"peak  {peak_keV:.1f} keV")
        ax.set_xlabel("Photon energy (keV)")
        ax.set_ylabel("Normalised spectral flux (a.u.)")
        ax.set_title(f"Spectrum set from beam  -  {label}")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(_bm_.E_MIN / 1e3, _bm_.E_MAX / 1e3)
        ax.set_ylim(0)
        plt.tight_layout()
        if save_fig:
            plt.savefig(save_fig, dpi=150)
            print(f"       Figure saved -> {save_fig}")
        _maybe_show(fig)

    return en, flux


def set_kb_source_from_beam(beam, nrays=500_000, seed=1234,
                             apply_sl2_clip=True,
                             plot=True, label="beam", save_fig=""):
    """
    Build the KB source by resampling phase space from an existing beam.

    Unlike element_kb_source() which uses a synthetic Gaussian + flat angular
    distribution, this function extracts the actual angular and energy
    distribution from a real simulated beam (e.g. after M2 or SL2) and
    resamples it to produce the desired number of rays.

    This preserves the correct footprint shape on the KB mirrors, which
    depends on how the BM emission profile is shaped by M1+M2 apertures
    and clipped by SL2.

    Physical basis
    --------------
    The footprint on each KB mirror is determined by:
    - The angular acceptance of SL2 (±A_SL2 µrad from BM)  - clips the cone
    - SL2 only illuminates a fraction of each mirror:
        KB1: SL2 / KB1-acceptance ~ 38%  (central 115 mm of 300 mm)
        KB2: SL2 / KB2-acceptance ~ 12%  (central  18 mm of 150 mm)
    - The angular distribution within the SL2 cone comes from the BM profile
      as shaped by M1+M2 mirror apertures - this is what this function captures.

    The function also stores the beam's energy distribution in
    SPECTRUM_ENERGY_EV / SPECTRUM_FLUX so that element_kb_source() will use
    the same spectrum if called subsequently.

    Parameters
    ----------
    beam          : S4Beam  - beam after M2 (or SL2, or any upstream element)
    nrays         : int     - number of resampled rays for the KB source
    seed          : int     - random seed for resampling
    apply_sl2_clip: bool    - if True, clip to SL2 angular acceptance before
                              resampling (default True)
    plot          : bool    - show diagnostic plots
    label         : str     - label for plots
    save_fig      : str     - filename to save figure ('' = don't save)

    Returns
    -------
    beam_out : S4Beam  - resampled KB source beam, ready for element_kb1()

    Example
    -------
    import beamline2 as bm

    # Run M1+M2 (needs many rays because SL2 is tiny)
    beam_src, norm = bm.source_bm32(nrays=2_000_000)
    beam_m1, _    = bm.element_m1(beam_src)
    beam_m2, _    = bm.element_m2(beam_m1)

    # Build KB source from M2 beam, preserving footprint
    beam_kb = bm.set_kb_source_from_beam(beam_m2, nrays=500_000)
    beam_kb1, fp1 = bm.element_kb1(beam_kb)
    beam_kb2, fp2 = bm.element_kb2(beam_kb1)
    bm.plot_footprint(fp1, 'KB1 footprint')
    """
    _bm_ = _self()
    g    = _geo()

    # -- 1. Get surviving rays from input beam ---------------------------------
    rays_in = beam.rays.copy()
    good    = rays_in[:, 9] > 0

    if apply_sl2_clip:
        py = rays_in[:, 4]
        py_safe = np.where(np.abs(py) > 1e-10, py, 1e-10)
        angle_h = np.abs(rays_in[:, 3] / py_safe)
        angle_v = np.abs(rays_in[:, 5] / py_safe)
        sl2_pass = (angle_h <= g['A_SL2_H']) & (angle_v <= g['A_SL2_V'])
        good = good & sl2_pass
        n_clip = good.sum()
        print(f"\n[set_kb_source_from_beam] '{label}'")
        print(f"       Input beam: {(rays_in[:, 9] > 0).sum()} rays")
        print(f"       After SL2 angular clip "
              f"(H=±{g['A_SL2_H']*1e6:.1f} µrad  V=±{g['A_SL2_V']*1e6:.1f} µrad): "
              f"{n_clip} rays")
        if n_clip < 20:
            print(f"  ⚠️  Too few rays after SL2 clip ({n_clip}).")
            print(f"       Using full beam without SL2 clip for resampling.")
            print(f"       Run source_bm32 with >={int(1e6 / max(n_clip, 1) * 20):,} "
                  f"rays for reliable SL2 statistics.")
            good = rays_in[:, 9] > 0
    else:
        good = good
        print(f"\n[set_kb_source_from_beam] '{label}'  (no SL2 clip)")
        print(f"       Input beam: {good.sum()} rays")

    source_rays = rays_in[good]
    n_src = len(source_rays)
    if n_src == 0:
        raise ValueError("set_kb_source_from_beam: no rays available for resampling.")

    # -- 2. Store energy spectrum -----------------------------------------------
    # Use the same conversion as beam.get_column(26):
    #   E [eV] = rays[:, 10] / A2EV   where A2EV = 50677.307 Å^-1 / eV
    from shadow4.beam.s4_beam import A2EV
    e_eV_clip = source_rays[:, 10] / A2EV   # eV  (identical to get_column(26))
    # Build spectrum histogram
    counts, edges = np.histogram(
        e_eV_clip, bins=300,
        range=(_bm_.E_MIN, _bm_.E_MAX))
    en    = 0.5 * (edges[:-1] + edges[1:])
    total = counts.sum()
    flux  = counts / total if total > 0 else np.ones_like(counts) / 300
    _bm_.SPECTRUM_ENERGY_EV = en
    _bm_.SPECTRUM_FLUX      = flux
    peak_keV = en[flux.argmax()] / 1e3
    print(f"       Spectrum stored: peak = {peak_keV:.1f} keV  "
          f"(from {n_src} rays)")

    # -- 3. Resample to requested nrays ----------------------------------------
    rng = np.random.default_rng(seed)
    idx = rng.choice(n_src, size=nrays, replace=True)
    new_rays = source_rays[idx].copy()

    # Reset col 9 (flag) to 1 (good) and col 1 (y position) to 0
    # so the rays start at the source plane
    new_rays[:, 9]  = 1
    new_rays[:, 1]  = 0.0   # y position = 0 (at source plane)
    # Assign sequential ray numbers
    new_rays[:, 11] = np.arange(1, nrays + 1, dtype=float)

    print(f"       Resampled {nrays} rays from {n_src} unique rays  "
          f"(bootstrap)")

    # -- 4. Wrap in S4Beam -----------------------------------------------------
    from shadow4.beam.s4_beam import S4Beam
    beam_out = S4Beam()
    beam_out.rays = new_rays

    # -- 5. Print summary ------------------------------------------------------
    g2 = _geo()
    fwhm_v = _bm_.SIGMA_Y * 2.355 * g2['L_KB1_SA'] / g2['P_KB1']
    fwhm_h = _bm_.SIGMA_X * 2.355 * g2['L_KB2_SA'] / g2['P_KB2']
    print(f"       Geometric focus FWHM: V={fwhm_v*1e9:.2f} nm  H={fwhm_h*1e9:.2f} nm")
    print(f"       Angular spread: "
          f"px/py std={source_rays[:,3].std()*1e6:.2f} µrad  "
          f"pz/py std={source_rays[:,5].std()*1e6:.2f} µrad")

    # -- 6. Plot ---------------------------------------------------------------
    if plot:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle(f"KB source phase space from '{label}'", fontsize=11)

        # Angular distribution
        px_py = source_rays[:, 3] * 1e6   # µrad
        pz_py = source_rays[:, 5] * 1e6
        ax = axes[0]
        ax.scatter(px_py, pz_py, s=0.3, alpha=0.3, color="steelblue")
        ax.axvline( g['A_SL2_H']*1e6, color="red", lw=0.8, ls="--",
                    label=f"SL2 ±{g['A_SL2_H']*1e6:.1f} µrad")
        ax.axvline(-g['A_SL2_H']*1e6, color="red", lw=0.8, ls="--")
        ax.axhline( g['A_SL2_V']*1e6, color="orange", lw=0.8, ls="--")
        ax.axhline(-g['A_SL2_V']*1e6, color="orange", lw=0.8, ls="--")
        ax.set_xlabel("px/py  (µrad)  ->  lab horizontal")
        ax.set_ylabel("pz/py  (µrad)  ->  lab vertical")
        ax.set_title("Angular distribution\n(SL2 limits shown)")
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

        # Spatial distribution
        ax2 = axes[1]
        ax2.scatter(source_rays[:, 0]*1e3, source_rays[:, 2]*1e3,
                    s=0.3, alpha=0.3, color="royalblue")
        ax2.set_xlabel("x  (mm)  ->  lab horizontal")
        ax2.set_ylabel("z  (mm)  ->  lab vertical")
        ax2.set_title("Spatial distribution\n(in exit frame)")
        ax2.grid(True, alpha=0.3)

        # Spectrum
        ax3 = axes[2]
        ax3.plot(en/1e3, flux, color="crimson", lw=2)
        ax3.axvline(peak_keV, color="darkred", lw=0.8, ls="--",
                    label=f"peak {peak_keV:.1f} keV")
        ax3.set_xlabel("Photon energy (keV)")
        ax3.set_ylabel("Normalised flux")
        ax3.set_title("Energy spectrum")
        ax3.legend(fontsize=8); ax3.grid(True, alpha=0.3)
        ax3.set_xlim(_bm_.E_MIN/1e3, _bm_.E_MAX/1e3); ax3.set_ylim(0)

        plt.tight_layout()
        if save_fig:
            plt.savefig(save_fig, dpi=150)
            print(f"       Figure saved -> {save_fig}")
        _maybe_show(fig)

    return beam_out


_bm_ = _self()
g = _geo()
def element_kb_source(nrays=500_000, seed=1234):
    """
    Polychromatic source for KB ray tracing  (BM as true source).

    Physical model (flat M1+M2, BM as true source)
    ------------------------------------------------
    The BM is the source. M1 and M2 deflect the beam but do not focus it.
    SL2 clips the angular cone entering the KB mirrors.

        FWHM_H = 2 x SL2_H x 2.355 x q_KB2/p_KB2

    Parameters
    ----------
    nrays : int   number of rays

    Returns
    -------
    beam : S4Beam - source at BM position, ready for element_kb1()
    """
    _bm_ = _self(); g = _geo()

    # p_focus = BM to KB (flat mirrors don't change the source position)
    p_KB1 = g['P_KB1']   # 44.645 m
    p_KB2 = g['P_KB2']   # 44.880 m

    # Angular acceptance: SL2 aperture limits the cone entering the KB mirrors
    accept_v_mirror = (_bm_.KB1_LENGTH/2) * np.sin(_bm_.G_KB1) / p_KB1
    accept_h_mirror = (_bm_.KB2_LENGTH/2) * np.sin(_bm_.G_KB2) / p_KB2
    accept_v = min(accept_v_mirror, g['A_SL2_V'])
    accept_h = min(accept_h_mirror, g['A_SL2_H'])

    # Geometric focus from BM source sigma
    fwhm_v = _bm_.SIGMA_Y * 2.355 * g['L_KB1_SA'] / p_KB1
    fwhm_h = _bm_.SIGMA_X * 2.355 * g['L_KB2_SA'] / p_KB2

    print(f"\n[KB source] Virtual source at SL2  ({nrays} rays)")
    print(f"       BM σ: H={_bm_.SIGMA_X*1e6:.1f} µm  V={_bm_.SIGMA_Y*1e6:.1f} µm")
    print(f"       accept_V = ±{accept_v*1e6:.2f} µrad  "
          f"({'SL2' if g['A_SL2_V'] < accept_v_mirror else 'KB1 mirror'} limits)")
    print(f"       accept_H = ±{accept_h*1e6:.2f} µrad  "
          f"({'SL2' if g['A_SL2_H'] < accept_h_mirror else 'KB2 mirror'} limits)")
    print(f"       Geometric focus FWHM: V={fwhm_v*1e9:.2f} nm  H={fwhm_h*1e9:.2f} nm")

    src = SourceGeometrical(name="kb_source", nrays=nrays, seed=seed)
    src.set_spatial_type_gaussian(
        sigma_h=_bm_.SIGMA_X,
        sigma_v=_bm_.SIGMA_Y)
    src.set_angular_distribution_flat(
        hdiv1=-accept_h, hdiv2=+accept_h,
        vdiv1=-accept_v,  vdiv2=+accept_v)

    if _bm_.SPECTRUM_ENERGY_EV is not None and _bm_.SPECTRUM_FLUX is not None:
        src.set_energy_distribution_userdefined(
            spectrum_abscissas=_bm_.SPECTRUM_ENERGY_EV,
            spectrum_ordinates=_bm_.SPECTRUM_FLUX,
            unit="eV")
        peak = _bm_.SPECTRUM_ENERGY_EV[np.argmax(_bm_.SPECTRUM_FLUX)] / 1e3
        print(f"       Energy: M1+M2 filtered spectrum  (peak {peak:.1f} keV)")
    else:
        src.set_energy_distribution_uniform(
            value_min=_bm_.E_MIN, value_max=_bm_.E_MAX, unit="eV")
        print(f"       Energy: UNIFORM - run compute_spectrum_at_sl2() for "              "physically correct spectrum")

    src.set_depth_distribution_off()
    beam = src.get_beam()
    print(f"       {_good(beam).shape[0]} rays generated")
    return beam


_bm_ = _self()
g = _geo()
def element_kb1(beam):
    """
    KB1 - Ir fixed-curvature ellipsoidal cylinder, focuses VERTICALLY.

    azimuthal = 0   ->  deflects in vertical (Y-Z) plane.
    p_coord   = g['L_SL2_KB1']   (from SL2 to KB1)
    q_coord   = g['L_KB1_KB2']   (from KB1 to KB2, coordinate frame drift)
    p_focus   = g['L_SL2_KB1']   (ellipse p)
    q_focus   = g['L_KB1_SA']    (ellipse q - focuses to sample)

    Demagnification (vertical): q_KB1/p_KB1  (computed at call time)
    """
    # p_focus = BM source to KB1 (flat M1+M2 don't focus - BM is the source)
    # p_coord = drift from wherever the beam currently is (SL2 or BM equiv.)
    # Angular acceptance of KB1 from the BM source:
    accept_v_mirror = (_bm_.KB1_LENGTH/2) * np.sin(_bm_.G_KB1) / g['P_KB1']
    # SL2 may further restrict angular acceptance:
    accept_v = min(accept_v_mirror, g['A_SL2_V'])
    demag_v  = g['L_KB1_SA'] / g['P_KB1']
    print(f"\n[KB1] KB1 (V-focus)  D={_bm_.D_KB1:.3f} m  G={_bm_.G_KB1*1e3:.3f} mrad  "
          f"L={_bm_.KB1_LENGTH*1e3:.0f} mm ...")
    print(f"       p={g['P_KB1']:.3f} m (from BM)  q={g['L_KB1_SA']:.3f} m  "
          f"demag={demag_v:.5f}")
    print(f"       Mirror accept=±{accept_v_mirror*1e6:.1f} µrad  "
          f"SL2 accept=±{g['A_SL2_V']*1e6:.1f} µrad  "
          f"effective=±{accept_v*1e6:.1f} µrad  "
          f"({'SL2 limits' if g['A_SL2_V'] < accept_v_mirror else 'mirror limits'})")
    mirror = S4EllipsoidMirror(
        name="KB1", boundary_shape=_aperture(_bm_.KB1_WIDTH, _bm_.KB1_LENGTH),
        surface_calculation=SurfaceCalculation.INTERNAL,
        p_focus=g['L_SL2_KB1'], q_focus=g['L_KB1_SA'],
        grazing_angle=_bm_.G_KB1, is_cylinder=True,
        cylinder_direction=Direction.TANGENTIAL, convexity=Convexity.UPWARD,
        f_reflec=1, f_refl=5, file_refl="",
        coating_material="Ir", coating_density=22.56, coating_roughness=3.0)
    coords = ElementCoordinates(
        p=g['L_SL2_KB1'], q=g['L_KB1_KB2'],
        angle_radial=_ar(_bm_.G_KB1), angle_azimuthal=0.0)
    beam_out, footprint = S4EllipsoidMirrorElement(
        optical_element=mirror, coordinates=coords,
        input_beam=beam).trace_beam()
    n_out = _good(beam_out).shape[0]
    n_in  = _good(beam).shape[0]
    print(f"       {n_out} / {n_in} rays survive  ({100*n_out/max(n_in,1):.1f}%)")
    return beam_out, footprint


_bm_ = _self()
g = _geo()
def element_kb2(beam):
    """
    KB2 - Ir fixed-curvature ellipsoidal cylinder, focuses HORIZONTALLY.

    azimuthal = pi/2  ->  deflects in horizontal (X-Y) plane.
    p_coord   = g['L_KB1_KB2']   (drift from KB1)
    q_coord   = g['L_KB2_SA']    (from KB2 to sample)
    p_focus   = g['L_SL2_KB2']   (ellipse p - from SL2 to KB2)
    q_focus   = g['L_KB2_SA']    (ellipse q - focuses to sample)

    Demagnification (horizontal): q_KB2/p_KB2  (computed at call time)
    """
    accept_h_mirror = (_bm_.KB2_LENGTH/2) * np.sin(_bm_.G_KB2) / g['P_KB2']
    accept_h = min(accept_h_mirror, g['A_SL2_H'])
    demag_h  = g['L_KB2_SA'] / g['P_KB2']
    print(f"\n[KB2] KB2 (H-focus)  D={_bm_.D_KB2:.3f} m  G={_bm_.G_KB2*1e3:.3f} mrad  "
          f"L={_bm_.KB2_LENGTH*1e3:.0f} mm ...")
    print(f"       p={g['P_KB2']:.3f} m (from BM)  q={g['L_KB2_SA']:.3f} m  "
          f"demag={demag_h:.5f}")
    print(f"       Mirror accept=±{accept_h_mirror*1e6:.1f} µrad  "
          f"SL2 accept=±{g['A_SL2_H']*1e6:.1f} µrad  "
          f"effective=±{accept_h*1e6:.1f} µrad  "
          f"({'SL2 limits' if g['A_SL2_H'] < accept_h_mirror else 'mirror limits'})")
    mirror = S4EllipsoidMirror(
        name="KB2", boundary_shape=_aperture(_bm_.KB2_WIDTH, _bm_.KB2_LENGTH),
        surface_calculation=SurfaceCalculation.INTERNAL,
        p_focus=g['L_SL2_KB2'], q_focus=g['L_KB2_SA'],
        grazing_angle=_bm_.G_KB2, is_cylinder=True,
        cylinder_direction=Direction.TANGENTIAL, convexity=Convexity.UPWARD,
        f_reflec=1, f_refl=5, file_refl="",
        coating_material="Ir", coating_density=22.56, coating_roughness=3.0)
    coords = ElementCoordinates(
        p=g['L_KB1_KB2'], q=g['L_KB2_SA'],
        angle_radial=_ar(_bm_.G_KB2), angle_azimuthal=np.pi/2)
    beam_out, footprint = S4EllipsoidMirrorElement(
        optical_element=mirror, coordinates=coords,
        input_beam=beam).trace_beam()
    n_out = _good(beam_out).shape[0]
    n_in  = _good(beam).shape[0]
    print(f"       {n_out} / {n_in} rays survive  ({100*n_out/max(n_in,1):.1f}%)")
    print(f"\n  -- Focus at sample --------------------------------------")
    print(f"       Geometric FWHM  H = {10*2.355*g['L_KB2_SA']/g['L_SL2_KB2']:.4f} µm  "
          f"V = {10*2.355*g['L_KB1_SA']/g['L_SL2_KB1']:.4f} µm  (for 10 µm source σ)")
    return beam_out, footprint


# ==============================================================================
# CONVENIENCE: run full chain at once
# ==============================================================================

def run_full_chain(nrays=500_000, plot_each=True):
    """
    Run all elements in sequence and return a dict of all intermediate beams.

    Parameters
    ----------
    nrays      : number of source rays
    plot_each  : call plot_beam() after each element

    Returns
    -------
    dict with keys:
        beam_source, norm_factor,
        beam_sl1   (after Slits 1),
        beam_m1, footprint_m1,
        beam_m2, footprint_m2,
        beam_sl2   (after Slits 2),
        beam_kb1, footprint_kb1,
        beam_kb2, footprint_kb2   (= beam at sample)
    """
    # Note: SL2 passes only ~0.003% of rays. With nrays < 1M most KB rays
    # will be zero. For KB focus use element_kb_source() in a separate call.
    beam, norm = source_bm32(nrays=nrays)
    if plot_each: plot_beam(beam, "Source",  position_m=0.)

    beam_sl1 = element_slits1(beam)
    if plot_each: plot_beam(beam_sl1, "After SL1", position_m=D_SL1)

    beam_m1, fp_m1 = element_m1(beam_sl1)
    if plot_each: plot_beam(beam_m1, "After M1", position_m=D_M1)

    beam_m2, fp_m2 = element_m2(beam_m1)
    if plot_each: plot_beam(beam_m2, "After M2", position_m=D_M2)

    beam_sl2 = element_slits2(beam_m2)
    if plot_each: plot_beam(beam_sl2, "After SL2", position_m=D_SL2)

    # Switch to dedicated KB source if no rays survive SL2
    if _good(beam_sl2).shape[0] < 10:
        print("  -> Switching to element_kb_source() for KB tracing")
        beam_sl2 = element_kb_source(nrays=nrays)

    beam_kb1, fp_kb1 = element_kb1(beam_sl2)
    if plot_each:
        plot_beam(beam_kb1, "After KB1", position_m=D_KB1)
        plot_footprint(fp_kb1, "KB1 footprint")

    beam_kb2, fp_kb2 = element_kb2(beam_kb1)
    if plot_each:
        plot_beam(beam_kb2, "At sample (after KB2)", position_m=D_SA)
        plot_footprint(fp_kb2, "KB2 footprint")
        plot_spectrum(beam_kb2, "Spectrum at sample", norm_factor=norm)

    return dict(
        beam_source   =beam,
        norm_factor   =norm,
        beam_sl1      =beam_sl1,
        beam_m1       =beam_m1,   footprint_m1 =fp_m1,
        beam_m2       =beam_m2,   footprint_m2 =fp_m2,
        beam_sl2      =beam_sl2,
        beam_kb1      =beam_kb1,  footprint_kb1=fp_kb1,
        beam_kb2      =beam_kb2,  footprint_kb2=fp_kb2,
    )


# ==============================================================================
if __name__ == "__main__":
    # -- Example: run step by step --------------------------------------------
    # beam, norm   = source_bm32(nrays=200_000)
    # beam_m1, _   = element_m1(beam)
    # beam_m2, _   = element_m2(beam_m1)
    # beam_sl2     = element_slits2(beam_m2)
    # beam_kb1, _  = element_kb1(beam_sl2)
    # beam_kb2, _  = element_kb2(beam_kb1)
    # plot_beam(beam_kb2, 'At sample', position_m=D_SA)
    # plot_spectrum(beam_kb2, 'Sample spectrum', norm_factor=norm)

    # -- Or run the full chain at once ----------------------------------------
    results = run_full_chain(nrays=200_000, plot_each=True)