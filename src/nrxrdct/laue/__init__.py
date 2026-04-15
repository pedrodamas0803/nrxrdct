"""
nrxrdct.laue — Laue diffraction simulation utilities.
"""

from .camera import Camera
from .crystal import build_b2, build_bcc, crystal_from_cif, crystals_from_cifs
from .laue_plotting import plot_2theta_chi, plot_all
from .simulation import (
    beam_in_crystal,
    en2lam,
    euler_to_U,
    is_superlattice,
    lam2en,
    lorentz_pol,
    print_bragg_table,
    print_spot_table,
    simulate_laue,
    spectrum_bm,
    spectrum_undulator,
    synchrotron_spectrum,
)

__all__ = [
    "en2lam",
    "lam2en",
    "spectrum_bm",
    "spectrum_undulator",
    "synchrotron_spectrum",
    "build_bcc",
    "build_b2",
    "euler_to_U",
    "beam_in_crystal",
    "lorentz_pol",
    "is_superlattice",
    "Camera",
    "simulate_laue",
    "print_spot_table",
    "print_bragg_table",
    "plot_2theta_chi",
    "plot_all",
    "crystal_from_cif",
    "crystals_from_cifs",
]
