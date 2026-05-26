"""
nrxrdct.powder — Powder XRD simulation and analysis utilities.

Functions
---------
simulate_powder_xrd_monophase
    Simulate a powder XRD pattern for one or more CIF-described phases.
get_powder_xrd_peaks
    Return peak positions and hkl families as DataFrames for one or more phases.
calculate_xrd_baseline
    Estimate the background baseline of an XRD intensity curve.
"""

from .simulation import (
    calculate_xrd_baseline,
    simulate_powder_xrd_monophase,
    get_powder_xrd_peaks,
)

__all__ = [
    "calculate_xrd_baseline",
    "simulate_powder_xrd_monophase",
    "get_powder_xrd_peaks",
]
