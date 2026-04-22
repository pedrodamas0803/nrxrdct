"""
BM32 pink-beam spectrum simulation — Shadow4 + xraylib
=======================================================

Full optical chain (polychromatic / pink-beam mode):

  SBM32 (bending magnet, ESRF-EBS 6.04 GeV)
    -> M1  (Ir bent cylinder, 26 m, 3.062 mrad, vertical focus)
    -> M2  (Ir bent cylinder, 30 m, 2.563 mrad, vertical focus)
    -> KB_V + KB_H reflectivity applied analytically
       Rh coating (post-2023), grazing angle 2.2 mrad (estimated)
    -> final spectrum at sample  (observable range: 5-33 keV)

All distances/angles from the official ESRF BM32 optics page:
https://www.esrf.fr/home/UsersAndScience/Experiments/CRG/BM32/Beamline/optics.html

Note on KB grazing angle
------------------------
The original Ir KB mirrors (pre-2023) operated at 2.8 mrad.
After the 2023 replacement with Rh-coated mirrors, the angle was
reduced. 2.8 mrad with Rh cuts the spectrum off at ~23 keV, which
is inconsistent with the observed 5-33 keV range at BM32.
A grazing angle of ~2.2 mrad with Rh gives a 0.1%-flux cutoff at
~34 keV, matching observations. Use G_KB = 2.2e-3 rad as best
estimate until the exact value is confirmed with the BM32 local team.

Design note — why KB is handled analytically
---------------------------------------------
The 20x20 um2 secondary source slit passes ~1 ray per 2M BM source rays.
Tracing through KB geometry would require O(10^8) rays. Applying R_KB(E)^2
to the M2 spectrum is physically exact for spectrum computation.

Shadow4 column conventions (1-based API, 0-based in rays[:,i]):
  col 10  (rays[:,9])  = ray flag (1=alive, -1=lost)
  col 26  via beam.get_column(26, nolost=1) = photon energy [eV]
  col 23  via beam.get_column(23, nolost=1) = total intensity
  angle_radial in ElementCoordinates = pi/2 - grazing_angle_rad

Install:
    pip install shadow4 xraylib srxraylib matplotlib numpy scipy
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from shadow4.beamline.optical_elements.mirrors.s4_ellipsoid_mirror import (
    S4EllipsoidMirror,
    S4EllipsoidMirrorElement,
)
from shadow4.beamline.s4_optical_element_decorators import Direction, SurfaceCalculation
from shadow4.physical_models.prerefl.prerefl import PreRefl
from shadow4.sources.bending_magnet.s4_bending_magnet import S4BendingMagnet
from shadow4.sources.bending_magnet.s4_bending_magnet_light_source import (
    S4BendingMagnetLightSource,
)
from shadow4.sources.s4_electron_beam import S4ElectronBeam
from syned.beamline.element_coordinates import ElementCoordinates
from syned.beamline.shape import Rectangle

# ── helpers ─────────────────────────────────────────────────────────────────


def _ar(grazing_rad: float) -> float:
    """angle_radial = pi/2 - grazing (Shadow4 / syned convention)."""
    return np.pi / 2 - grazing_rad


def _reflectivity(energies_eV, material, density, grazing_mrad, roughness_A=3.0):
    """Average (s+p)/2 Fresnel reflectivity for any coating via xraylib."""
    rs, rp = PreRefl.reflectivity_amplitudes_fresnel_external_xraylib(
        photon_energy_ev=energies_eV,
        coating_material=material,
        coating_density=density,
        grazing_angle_mrad=grazing_mrad,
        roughness_rms_A=roughness_A,
    )
    return 0.5 * (np.abs(rs) ** 2 + np.abs(rp) ** 2)


# Convenience wrappers
def _ir_reflectivity(energies_eV, grazing_mrad, roughness_A=3.0):
    """Ir coating (density 22.56 g/cm3)."""
    return _reflectivity(energies_eV, "Ir", 22.56, grazing_mrad, roughness_A)


def _rh_reflectivity(energies_eV, grazing_mrad, roughness_A=3.0):
    """Rh coating (density 12.41 g/cm3)."""
    return _reflectivity(energies_eV, "Rh", 12.41, grazing_mrad, roughness_A)


# ── source ───────────────────────────────────────────────────────────────────


def make_bm32_source(nrays=1_000_000):
    """ESRF SBM32 short bending magnet — ESRF-EBS parameters."""
    ebeam = S4ElectronBeam(energy_in_GeV=6.04, energy_spread=9.3e-4, current=0.2)
    ebeam.set_sigmas_all(
        sigma_x=30.1e-6, sigma_xp=4.2e-6, sigma_y=3.6e-6, sigma_yp=1.4e-6
    )
    bm = S4BendingMagnet(
        radius=23.588,
        magnetic_field=0.857,
        length=0.063,
        emin=5_000.0,
        emax=35_000.0,
        ng_e=200,
        flag_emittance=1,
    )
    return S4BendingMagnetLightSource(
        name="SBM32",
        electron_beam=ebeam,
        magnetic_structure=bm,
        nrays=nrays,
        seed=5676561,
    )


# ── mirror ───────────────────────────────────────────────────────────────────


def make_bent_cylinder_mirror(name, p_focus, q_focus, grazing_rad, aperture):
    """
    Ir-coated BENT CYLINDRICAL mirror (M1 and M2 at BM32).

    is_cylinder=True  + Direction.TANGENTIAL means the mirror is curved only
    in the tangential (beam-propagation) plane, and flat in the sagittal plane
    — the standard model for a mechanically bent mirror.
    Coating: Ir, 22.56 g/cm3.
    """
    return S4EllipsoidMirror(
        name=name,
        boundary_shape=aperture,
        surface_calculation=SurfaceCalculation.INTERNAL,
        p_focus=p_focus,
        q_focus=q_focus,
        grazing_angle=grazing_rad,
        is_cylinder=True,
        cylinder_direction=Direction.TANGENTIAL,
        f_reflec=1,
        f_refl=5,
        file_refl="",
        coating_material="Ir",
        coating_density=22.56,
        coating_roughness=3.0,
    )


def make_rh_kb_mirror(name, p_focus, q_focus, grazing_rad, aperture):
    """
    Rh-coated fixed-curvature ELLIPSOIDAL KB mirror (KB_V and KB_H at BM32).

    The BM32 KB mirrors (installed 2012) are fixed-curvature elliptical mirrors,
    not bent, so is_cylinder=False (full ellipsoid).
    Coating: Rh, 12.41 g/cm3.
    """
    return S4EllipsoidMirror(
        name=name,
        boundary_shape=aperture,
        surface_calculation=SurfaceCalculation.INTERNAL,
        p_focus=p_focus,
        q_focus=q_focus,
        grazing_angle=grazing_rad,
        is_cylinder=False,
        f_reflec=1,
        f_refl=5,
        file_refl="",
        coating_material="Rh",
        coating_density=12.41,
        coating_roughness=3.0,
    )


def trace_mirror(mirror, beam, p_coord, q_coord, grazing_rad, azimuthal):
    """
    Wrap mirror in element, inject beam, trace, return output beam.

    p_coord / q_coord are the COORDINATE-FRAME distances (drift lengths
    to/from the previous/next element), NOT p_focus/q_focus.

    azimuthal:  0   -> deflects vertically downward  (M1, KB_V)
                pi  -> deflects vertically upward     (M2)
                pi/2 -> deflects horizontally         (KB_H)
    """
    coords = ElementCoordinates(
        p=p_coord,
        q=q_coord,
        angle_radial=_ar(grazing_rad),
        angle_azimuthal=azimuthal,
    )
    out, _ = S4EllipsoidMirrorElement(
        optical_element=mirror,
        coordinates=coords,
        input_beam=beam,
    ).trace_beam()
    return out


# ── main simulation ──────────────────────────────────────────────────────────


def simulate_bm32_pink_beam_spectrum(
    nrays=1_000_000,
    n_energy_bins=250,
    plot=True,
    save_fig="bm32_pink_beam_spectrum.png",
):
    """
    Simulate the BM32 pink-beam energy spectrum with ABSOLUTE flux units [ph/s/eV].

    The flux normalization factor is derived by integrating the 2D BM spectral/angular
    distribution stored in source.tot (photons/s/eV) over the sampled vertical angle
    range, then dividing by nrays. Each surviving ray is thus weighted by
    total_flux_in_cone / nrays [ph/s/ray], converting histogram counts to ph/s.

    Returns dict with keys:
        energy_eV, flux_source, flux_after_m1, flux_after_m2,
        flux_at_sample  [all in ph/s per energy bin],
        norm_factor     [ph/s per ray],
        R_m1, R_m2, R_kb,
        beam_source, beam_after_m1, beam_after_m2
    """

    # ── BM32 layout — from beamline physical location table ────────────
    # Source: Physical Location (fixed components) document
    D_SL1 = 26.368  # m  Slits 1 (upstream aperture, defines horiz. acceptance)
    D_M1 = 28.309  # m  Mirror 1 (Ir bent cylinder)
    D_M2 = 31.732  # m  Mirror 2 (Ir bent cylinder)
    D_SL2 = 35.370  # m  Slits 2  (secondary source / mu-slits)
    D_KB1 = 44.645  # m  KB1 — vertical focusing  (L = 0.3 m)
    D_KB2 = 44.880  # m  KB2 — horizontal focusing (L = 0.15 m)
    D_SA = 45.000  # m  Sample position (LaueMAX)

    G_M1 = 3.062181698459972e-3  # rad  M1 grazing angle
    G_M2 = 2.5627372258861216e-3  # rad  M2 grazing angle
    G_KB = 2.2e-3  # rad  KB grazing angle (Rh, post-2023 estimate)
    # Original Ir KB used 2.8 mrad; Rh at 2.8 mrad
    # cuts off at ~23 keV, inconsistent with observed
    # 5-33 keV range. 2.2 mrad gives ~34 keV cutoff.

    # Mirror apertures in local frame: Rectangle(x_sag_left, x_sag_right,
    #                                             y_tan_bottom, y_tan_top)  [m]
    # M1 and M2: Ir bent cylinders, 1.2 m total / 1.1 m useful length
    #   Width not specified in document — assuming 50 mm (typical ESRF BM mirror)
    APT_M1 = Rectangle(-0.025, 0.025, -0.550, 0.550)  # 50 mm wide x 1100 mm useful
    APT_M2 = Rectangle(-0.025, 0.025, -0.550, 0.550)  # same for M2
    # KB1: L = 0.3 m,  KB2: L = 0.15 m
    #   Width not specified — assuming 20 mm (typical for KB)
    APT_KB1 = Rectangle(-0.010, 0.010, -0.150, 0.150)  # 20 mm x 300 mm
    APT_KB2 = Rectangle(-0.010, 0.010, -0.075, 0.075)  # 20 mm x 150 mm

    E_RANGE = (5_000, 35_000)  # eV — covers full Rh KB spectrum (5-33 keV)

    # ── 1. Source ───────────────────────────────────────────────────────
    print("[1/4] Sampling SBM32 source rays ...")
    source = make_bm32_source(nrays=nrays)
    beam = source.get_beam()
    beam_source = beam.duplicate()

    # ── Absolute flux normalization ─────────────────────────────────────
    # source.tot  : 2D array (n_psi, n_energy) in ph/s/eV, integrated over
    #               the full BM horizontal cone (HDIV1+HDIV2).
    # Integrating over psi [rad] and energy [eV] gives total ph/s in cone.
    # Each of the nrays Monte-Carlo rays represents (total_flux / nrays) ph/s.
    from scipy.integrate import trapezoid as trapz

    _flux_per_E = trapz(source.tot, source.angle_array_mrad * 1e-3, axis=0)
    _total_flux = trapz(_flux_per_E, source.photon_energy_array)  # ph/s
    norm_factor = _total_flux / nrays  # ph/s per ray
    print(f"      Total BM flux in cone   : {_total_flux:.4e} ph/s")
    print(f"      Normalization factor    : {norm_factor:.4e} ph/s / ray")

    e_src = beam.get_column(26, nolost=1)
    i_src = beam.get_column(23, nolost=1)
    print(
        f"      {len(e_src)} rays  |  E = [{e_src.min()/1e3:.1f}, {e_src.max()/1e3:.1f}] keV"
    )

    # ── 2. M1 ──────────────────────────────────────────────────────────
    print("[2/4] Tracing M1 (Ir bent cylinder, 28.309 m, 3.062 mrad, 1100 mm) ...")
    m1 = make_bent_cylinder_mirror(
        "M1",
        p_focus=D_M1,
        q_focus=D_SL2 - D_M1,  # 7.061 m to Slits 2
        grazing_rad=G_M1,
        aperture=APT_M1,
    )
    beam = trace_mirror(
        m1,
        beam,
        p_coord=D_M1,
        q_coord=D_M2 - D_M1,  # 3.423 m drift to M2
        grazing_rad=G_M1,
        azimuthal=0.0,
    )
    beam_after_m1 = beam.duplicate()
    e_m1 = beam.get_column(26, nolost=1)
    i_m1 = beam.get_column(23, nolost=1)
    print(f"      {len(e_m1)} rays survive M1  ({100*len(e_m1)/nrays:.2f}%)")

    # ── 3. M2 ──────────────────────────────────────────────────────────
    print("[3/4] Tracing M2 (Ir bent cylinder, 31.732 m, 2.563 mrad, 1100 mm) ...")
    m2 = make_bent_cylinder_mirror(
        "M2",
        p_focus=D_M2,
        q_focus=D_SL2 - D_M2,  # 3.638 m to Slits 2
        grazing_rad=G_M2,
        aperture=APT_M2,
    )
    # azimuthal=pi: M2 deflects upward, restoring overall beam direction
    beam = trace_mirror(
        m2,
        beam,
        p_coord=D_M2 - D_M1,  # 3.423 m drift from M1
        q_coord=D_SL2 - D_M2,  # 3.638 m to Slits 2
        grazing_rad=G_M2,
        azimuthal=np.pi,
    )
    beam_after_m2 = beam.duplicate()
    e_m2 = beam.get_column(26, nolost=1)
    i_m2 = beam.get_column(23, nolost=1)
    print(f"      {len(e_m2)} rays survive M2  ({100*len(e_m2)/nrays:.3f}%)")

    # ── 4. KB reflectivity (analytic, Rh coating) ──────────────────────
    # KB1 (vertical):   L=0.3 m,  at 44.645 m, p=9.275 m from SL2, q=0.355 m to sample
    # KB2 (horizontal): L=0.15 m, at 44.880 m, p=9.510 m from SL2, q=0.120 m to sample
    # Both Rh coated (post-2023), grazing angle 2.2 mrad (estimated)
    print("[4/4] Applying KB1 + KB2 Rh reflectivity analytically ...")
    e_centers = np.linspace(E_RANGE[0], E_RANGE[1], n_energy_bins)
    R_m1 = _ir_reflectivity(e_centers, G_M1 * 1e3)  # Ir, M1 (3.062 mrad)
    R_m2 = _ir_reflectivity(e_centers, G_M2 * 1e3)  # Ir, M2 (2.563 mrad)
    R_kb = _rh_reflectivity(e_centers, G_KB * 1e3)  # Rh, KB1 or KB2
    R_kb2 = R_kb**2  # KB1 × KB2

    # ── Histograms — counts × norm_factor = ph/s per bin ───────────────
    def _hist(e, i):
        """Return (bin_centres_eV, flux_ph_s_per_bin)."""
        counts, edges = np.histogram(e, bins=n_energy_bins, range=E_RANGE, weights=i)
        return 0.5 * (edges[:-1] + edges[1:]), counts * norm_factor

    en, sp_src = _hist(e_src, i_src)
    en, sp_m1 = _hist(e_m1, i_m1)
    en, sp_m2 = _hist(e_m2, i_m2)

    # Apply KB reflectivity (Rh × 2) to the M2 spectrum
    sp_sample = sp_m2 * interp1d(e_centers, R_kb2, bounds_error=False, fill_value=0.0)(
        en
    )

    # Bin width in eV (for ph/s/eV conversion)
    dE_eV = (E_RANGE[1] - E_RANGE[0]) / n_energy_bins

    peak_keV = en[np.argmax(sp_sample)] / 1e3 if sp_sample.max() > 0 else float("nan")
    total_sample_ph = sp_sample.sum()  # ph/s integrated over full band
    print(f"\n  Peak energy at sample        : {peak_keV:.1f} keV")
    print(f"  Integrated flux at sample   : {total_sample_ph:.3e} ph/s  (5–35 keV)")

    # ── plot ────────────────────────────────────────────────────────────
    if plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Left: absolute spectral flux at each stage
        for sp, lbl, col, lw in [
            # (sp_src / dE_eV, "BM source", "gray", 1.2),
            (
                sp_m1 / dE_eV,
                "After M1 (Ir bent cyl. 3.062 mrad, 28.3 m)",
                "steelblue",
                1.5,
            ),
            (
                sp_m2 / dE_eV,
                "After M2 (Ir bent cyl. 2.563 mrad, 31.7 m)",
                "royalblue",
                1.5,
            ),
            (sp_sample / dE_eV, "At sample (×KB1×KB2 Rh 2.2 mrad)", "crimson", 2.0),
        ]:
            ax1.plot(en / 1e3, sp, label=lbl, color=col, lw=lw)

        # Annotate integrated sample flux
        ax1.text(
            0.97,
            0.97,
            f"Sample flux\n{total_sample_ph:.2e} ph/s\n(5–35 keV)",
            transform=ax1.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray", alpha=0.8),
        )

        ax1.set_xlabel("Photon energy (keV)")
        ax1.set_ylabel("Spectral flux  (ph/s/eV)")
        ax1.set_title(
            "BM32 pink-beam spectrum\nM1/M2: Ir bent cylinder  |  KB: Rh (2.2 mrad, post-2023)"
        )
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(5, 35)
        ax1.set_ylim(0)

        ax2.plot(
            e_centers / 1e3,
            R_m1,
            label="M1  (Ir, 3.062 mrad)",
            color="steelblue",
            lw=1.5,
        )
        ax2.plot(
            e_centers / 1e3,
            R_m2,
            label="M2  (Ir, 2.563 mrad)",
            color="cornflowerblue",
            lw=1.5,
        )
        ax2.plot(
            e_centers / 1e3,
            R_kb,
            label="KB1 or KB2 (Rh, 2.2 mrad)",
            color="orange",
            lw=1.5,
        )
        ax2.plot(
            e_centers / 1e3,
            R_m1 * R_m2,
            label="M1×M2",
            color="royalblue",
            lw=1.5,
            ls="--",
        )
        ax2.plot(
            e_centers / 1e3, R_kb2, label="KB1×KB2", color="darkorange", lw=1.5, ls="--"
        )
        ax2.plot(
            e_centers / 1e3,
            R_m1 * R_m2 * R_kb2,
            label="Total (M1×M2×KB1×KB2)",
            color="crimson",
            lw=2,
        )
        ax2.set_xlabel("Photon energy (keV)")
        ax2.set_ylabel("Reflectivity")
        ax2.set_title(
            "Mirror reflectivities\n(Ir: M1/M2  |  Rh 2.2 mrad: KB1/KB2, post-2023)"
        )
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(5, 35)
        ax2.set_ylim(0, 1)

        plt.tight_layout()
        if save_fig:
            plt.savefig(save_fig, dpi=150)
            print(f"Figure saved -> {save_fig}")
        plt.show()

    return dict(
        energy_eV=en,
        flux_source=sp_src,  # ph/s per energy bin
        flux_after_m1=sp_m1,  # ph/s per energy bin
        flux_after_m2=sp_m2,  # ph/s per energy bin
        flux_at_sample=sp_sample,  # ph/s per energy bin
        dE_eV=dE_eV,  # bin width [eV]  -> divide by this for ph/s/eV
        norm_factor=norm_factor,  # ph/s per surviving ray
        total_flux_source=_total_flux,
        R_m1=R_m1,
        R_m2=R_m2,
        R_kb=R_kb,
        beam_source=beam_source,
        beam_after_m1=beam_after_m1,
        beam_after_m2=beam_after_m2,
    )


if __name__ == "__main__":
    results = simulate_bm32_pink_beam_spectrum(
        nrays=1_000_000,
        n_energy_bins=250,
        plot=True,
        save_fig="bm32_pink_beam_spectrum.png",
    )
    en = results["energy_eV"]
    sp = results["flux_at_sample"]
    dE = results["dE_eV"]
    print(f"\nPeak energy at sample        : {en[np.argmax(sp)]/1e3:.2f} keV")
    print(f"Peak spectral flux           : {sp.max()/dE:.3e} ph/s/eV")
    print(f"Integrated flux (5–35 keV)   : {sp.sum():.3e} ph/s")
    print(f"Normalization factor         : {results['norm_factor']:.4e} ph/s / ray")
    print(f"Total BM source flux (cone)  : {results['total_flux_source']:.4e} ph/s")
