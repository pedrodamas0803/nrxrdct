"""
Peak fitting and NMF decomposition utilities for XRD-CT data.
"""

from .nmf import HyperspectralNMF
from .peakfit import extract_window, fit_peak, fit_peak_from_file

__all__ = [
    "HyperspectralNMF",
    "extract_window",
    "fit_peak",
    "fit_peak_from_file",
]
