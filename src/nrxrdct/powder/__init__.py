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
make_alloy_crystal
    Build an xrayutilities Crystal for a multi-element solid-solution alloy
    with Vegard's-law lattice parameters.
elem_radius
    Return the CN-12 metallic radius (Å) for an element symbol.
list_structures
    List supported crystal structures for :func:`make_alloy_crystal`.
"""

from .simulation import (
    calculate_xrd_baseline,
    simulate_powder_xrd_monophase,
    get_powder_xrd_peaks,
)
from .structures import (
    make_alloy_crystal,
    elem_radius,
    list_structures,
)

__all__ = [
    "calculate_xrd_baseline",
    "simulate_powder_xrd_monophase",
    "get_powder_xrd_peaks",
    "make_alloy_crystal",
    "elem_radius",
    "list_structures",
]
