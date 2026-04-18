"""
nrxrdct.laue — Laue diffraction simulation utilities.
"""

from .camera import Camera
from .crystal import build_b2, build_bcc, crystal_from_cif, crystals_from_cifs
from .laue_plotting import plot_2theta_chi, plot_all, plot_compare_spots, plot_interactive_tth_chi, plot_layer_scheme, plot_laue_stack_spots, plot_strain_broadening, plot_tth_chi_overlay, warp_image_to_tth_chi
from .layers import (
    Layer,
    LayeredCrystal,
    or_baker_nutting,
    or_from_directions,
    or_kurdjumov_sachs,
    or_nishiyama_wassermann,
    or_pitsch,
    orientation_along_z,
)
from .simulation import (
    BM32_KB,
    beam_in_crystal,
    decompose_matstarlab,
    en2lam,
    estimate_instrument_broadening,
    measure_spot_widths,
    euler_to_U,
    fit_strain_distribution,
    kb_reflectivity,
    lam2en,
    lorentz_pol,
    print_bragg_table,
    print_spot_table,
    simulate_laue,
    simulate_laue_stack,
    spectrum_bm,
    spectrum_undulator,
    strain_broadening,
    strain_spot_jacobian,
    synchrotron_spectrum,
    U_from_matstarlab,
)

__all__ = [
    # Energy / wavelength conversion
    "en2lam",
    "lam2en",
    # Synchrotron spectra
    "spectrum_bm",
    "spectrum_undulator",
    "synchrotron_spectrum",
    # KB mirror optics
    "kb_reflectivity",
    "BM32_KB",
    # Crystal builders
    "build_bcc",
    "build_b2",
    "crystal_from_cif",
    "crystals_from_cifs",
    # Orientation
    "euler_to_U",
    "U_from_matstarlab",
    "decompose_matstarlab",
    "beam_in_crystal",
    # Strain analysis
    "measure_spot_widths",
    "estimate_instrument_broadening",
    "fit_strain_distribution",
    "strain_spot_jacobian",
    "strain_broadening",
    # Physics helpers
    "lorentz_pol",
    # Camera / detector
    "Camera",
    # Simulation
    "simulate_laue",
    "simulate_laue_stack",
    "print_spot_table",
    "print_bragg_table",
    # Layered structures
    "Layer",
    "LayeredCrystal",
    "orientation_along_z",
    "or_from_directions",
    "or_kurdjumov_sachs",
    "or_nishiyama_wassermann",
    "or_baker_nutting",
    "or_pitsch",
    # Plotting
    "plot_2theta_chi",
    "plot_all",
    "plot_compare_spots",
    "plot_interactive_tth_chi",
    "plot_layer_scheme",
    "plot_laue_stack_spots",
    "plot_strain_broadening",
    "plot_tth_chi_overlay",
    "warp_image_to_tth_chi",
]
