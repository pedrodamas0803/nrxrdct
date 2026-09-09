"""
nrxrdct.powder.structures
--------------------------
Build xrayutilities Crystal objects for multi-element solid-solution alloys.

Lattice parameters are estimated via Vegard's law: the molar-fraction-weighted
average of per-element CN-12 equivalent metallic radii is converted to the
lattice parameter of the target structure.

Metallic radii are derived from crystallographic data when available, using
a uniform volume-to-radius formula:

    r_CN12 = (3 · V_atom · φ_FCC / (4π))^(1/3)

where V_atom is the volume per atom in the element's own crystal and
φ_FCC = π/(3√2) ≈ 0.7405 is the FCC packing fraction.  This converts
any structure's atomic volume to the CN-12 equivalent radius used in
Vegard's-law calculations.

Data sources (in priority order):
    1. xrayutilities — crystallographic lattice parameters (exact)
    2. xraylib       — elemental density + atomic weight (< 1 % error)

Also provides :func:`symmetrize_crystal_from_p1`, which turns a P1 structure
(e.g. a DFT-relaxed cell with no assumed symmetry) into an xrayutilities
Crystal built from its true space group, using spglib to detect the
symmetry and idealize the atomic positions.
"""
from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np
import spglib
import xraylib
import xrayutilities as xu

# ── Physical constants ────────────────────────────────────────────────────────
_NA       = 6.02214076e23          # Avogadro (mol⁻¹)
_FCC_PACK = np.pi / (3 * np.sqrt(2))  # FCC packing fraction ≈ 0.7405

# ── Space-group number → structure tag ───────────────────────────────────────
_SG_STRUCT: dict[int, str] = {
    225: "FCC",     # Fm-3m  (Al, Cu, Au, Ag, Pt, Pd, …)
    229: "BCC",     # Im-3m  (Fe, Cr, W, Mo, V, Nb, Ta, …)
    194: "HCP",     # P6₃/mmc (Ti, Mg, Co, Zn, Zr, …)
    221: "SC",      # Pm-3m
    227: "DIAMOND", # Fd-3m  (Si, Ge, C, …)
}

# ── Supported target structures ───────────────────────────────────────────────
#   sgnum  : ITA space-group number
#   wyck   : Wyckoff label used for the mixed-occupancy site
#   hex    : True if the lattice needs both a and c
#   ideal_ca : ideal c/a ratio (HCP only)
_TARGET: dict[str, dict] = {
    "BCC": {"sgnum": 229, "wyck": "2a", "hex": False},
    "FCC": {"sgnum": 225, "wyck": "4a", "hex": False},
    "HCP": {"sgnum": 194, "wyck": "2c", "hex": True,
            "ideal_ca": np.sqrt(8 / 3)},
    "SC":  {"sgnum": 221, "wyck": "1a", "hex": False},
}


# ── Radius helpers ────────────────────────────────────────────────────────────

def _v_to_r(V_atom: float) -> float:
    """Convert atomic volume (Å³) to CN-12 equivalent metallic radius (Å)."""
    return (3.0 * V_atom * _FCC_PACK / (4.0 * np.pi)) ** (1.0 / 3.0)


def _xu_radius(symbol: str) -> float | None:
    """
    Derive the CN-12 metallic radius from xrayutilities' built-in element crystal.

    Returns None if the element or its structure is not recognised.
    """
    cr = getattr(xu.materials, symbol, None)
    if not isinstance(cr, xu.materials.Crystal):
        return None
    sg = cr.lattice.space_group
    a  = cr.a
    struct = _SG_STRUCT.get(sg)
    if struct == "BCC":
        V_atom = a ** 3 / 2.0
    elif struct == "FCC":
        V_atom = a ** 3 / 4.0
    elif struct == "HCP":
        c = cr.lattice.c
        V_atom = a ** 2 * c * np.sqrt(3.0) / 4.0
    else:
        return None
    return _v_to_r(V_atom)


def _xrl_radius(symbol: str) -> float | None:
    """
    Derive the CN-12 metallic radius from xraylib density and atomic weight.

    Accurate to < 1 % for most metallic elements.  Returns None if xraylib
    has no density data (ElementDensity returns 0).
    """
    try:
        Z   = xraylib.SymbolToAtomicNumber(symbol)
        rho = xraylib.ElementDensity(Z)   # g cm⁻³
        M   = xraylib.AtomicWeight(Z)     # g mol⁻¹
    except Exception:
        return None
    if rho == 0.0:
        return None
    V_atom = M / (_NA * rho) * 1e24    # Å³ per atom
    return _v_to_r(V_atom)


def elem_radius(symbol: str) -> float:
    """
    Return the CN-12 equivalent metallic radius (Å) for an element symbol.

    Tries xrayutilities' crystallographic data first, falls back to
    xraylib density data.

    Args:
        symbol (str): Element symbol, e.g. ``"Fe"``.

    Returns:
        float: CN-12 metallic radius in Å.

    Raises:
        ValueError: If no radius data is available for the element.
    """
    r = _xu_radius(symbol)
    if r is not None:
        return r
    r = _xrl_radius(symbol)
    if r is not None:
        return r
    raise ValueError(
        f"No radius data found for '{symbol}'. "
        "Supply the lattice parameter `a` (and `c` for HCP) directly."
    )


# ── Lattice-parameter ↔ radius conversions ────────────────────────────────────

def _r_to_a(r: float, structure: str) -> float:
    """Convert CN-12 metallic radius to lattice parameter *a* for *structure*."""
    if structure == "FCC":
        return 2.0 * np.sqrt(2.0) * r
    if structure == "BCC":
        return 4.0 * r / np.sqrt(3.0)
    if structure in ("HCP", "SC"):
        return 2.0 * r
    raise ValueError(f"Unknown structure '{structure}'.")


# ── Public API ────────────────────────────────────────────────────────────────

def make_alloy_crystal(
    elements: Sequence[str],
    fractions: Sequence[float] | None = None,
    *,
    structure: str = "BCC",
    name: str | None = None,
    a: float | None = None,
    c: float | None = None,
    b_factor: float = 0.0,
) -> xu.materials.Crystal:
    """
    Build an xrayutilities Crystal for a multi-element solid-solution alloy.

    Lattice parameters are estimated via Vegard's law unless overridden.
    Each element is placed on the same Wyckoff site with occupancy equal to
    its molar fraction.  When *fractions* is not supplied every element gets
    equal occupancy (1/N).

    Args:
        elements (list of str): Element symbols, e.g. ``["Fe", "Ni", "Cr"]``.
        fractions (list of float, optional): Molar fractions for each element,
            same order as *elements*.  Must be positive and are normalised
            automatically.  Defaults to ``1/N`` (equal occupancies).
        structure (str): Target crystal structure — one of ``"BCC"``,
            ``"FCC"``, ``"HCP"``, ``"SC"`` (default ``"BCC"``).
        name (str, optional): Name of the Crystal object.  Defaults to the
            concatenated element symbols + ``_<structure>``,
            e.g. ``"FeNiCr_BCC"``.
        a (float, optional): Lattice parameter *a* in Å.  Overrides the
            Vegard's-law estimate.
        c (float, optional): Lattice parameter *c* in Å (HCP only).  When
            omitted the ideal *c/a* ratio (√(8/3) ≈ 1.633) is used.
        b_factor (float): Isotropic Debye–Waller *B*-factor applied to all
            sites (default 0.0).

    Returns:
        xu.materials.Crystal: Crystal object ready for use with
            :func:`~nrxrdct.powder.simulate_powder_xrd_monophase` or
            xrayutilities' ``PowderModel``.

    Raises:
        ValueError: If *structure* is not recognised, *elements* is empty,
            *fractions* length differs from *elements*, or a required metallic
            radius is unavailable and *a* was not supplied.

    Example:
        >>> # Equiatomic CrMnFeCoNi Cantor alloy (FCC)
        >>> crystal = make_alloy_crystal(
        ...     ["Cr", "Mn", "Fe", "Co", "Ni"], structure="FCC"
        ... )
        >>> print(f"a = {crystal.a:.4f} A")

        >>> # Fe60Cr20Ni20 (wt% approximate) BCC steel, explicit a
        >>> crystal = make_alloy_crystal(
        ...     ["Fe", "Cr", "Ni"], fractions=[0.60, 0.20, 0.20],
        ...     structure="BCC", a=2.874
        ... )

        >>> # Ti-6Al-4V HCP phase (approximate)
        >>> crystal = make_alloy_crystal(
        ...     ["Ti", "Al", "V"], fractions=[0.90, 0.06, 0.04],
        ...     structure="HCP",
        ... )
    """
    structure = structure.upper()
    if structure not in _TARGET:
        raise ValueError(
            f"Unknown structure '{structure}'. "
            f"Choose from: {sorted(_TARGET)}."
        )

    elements = list(elements)
    if not elements:
        raise ValueError("`elements` must not be empty.")
    N = len(elements)

    # ── Normalise fractions ────────────────────────────────────────────────────
    if fractions is None:
        fracs = np.ones(N) / N
    else:
        fracs = np.asarray(fractions, dtype=float)
        if len(fracs) != N:
            raise ValueError(
                f"`fractions` length ({len(fracs)}) must match "
                f"`elements` length ({N})."
            )
        if np.any(fracs < 0):
            raise ValueError("`fractions` must be non-negative.")
        total = fracs.sum()
        if total <= 0:
            raise ValueError("`fractions` must sum to a positive value.")
        if not np.isclose(total, 1.0, atol=1e-6):
            warnings.warn(
                f"fractions sum to {total:.6g}; normalising to 1.",
                stacklevel=2,
            )
            fracs = fracs / total

    # ── Vegard's-law lattice parameters ───────────────────────────────────────
    sp = _TARGET[structure]

    if a is None:
        radii = np.array([elem_radius(sym) for sym in elements])
        r_avg = float(fracs @ radii)
        a = _r_to_a(r_avg, structure)

    if sp["hex"] and c is None:
        c = sp["ideal_ca"] * a

    # ── Auto-name ─────────────────────────────────────────────────────────────
    if name is None:
        name = "".join(elements) + f"_{structure}"

    # ── Build SGLattice ───────────────────────────────────────────────────────
    wyck  = sp["wyck"]
    sgnum = sp["sgnum"]

    lat_kwargs = dict(
        atoms=elements,
        pos=[wyck] * N,
        occ=fracs.tolist(),
        b=[b_factor] * N,
    )

    if sp["hex"]:
        lat = xu.materials.SGLattice(sgnum, a, c, **lat_kwargs)
    else:
        lat = xu.materials.SGLattice(sgnum, a, **lat_kwargs)

    return xu.materials.Crystal(name, lat)


def list_structures() -> list[str]:
    """Return the supported structure types for :func:`make_alloy_crystal`."""
    return sorted(_TARGET)


# ── P1 → symmetrized Crystal ──────────────────────────────────────────────────

def _lattice_params(lattice: np.ndarray) -> tuple[float, float, float, float, float, float]:
    """Convert a 3x3 lattice-vector matrix (rows) to (a, b, c, alpha, beta, gamma)."""
    va, vb, vc = lattice
    a, b, c = np.linalg.norm(va), np.linalg.norm(vb), np.linalg.norm(vc)
    alpha = np.degrees(np.arccos(np.dot(vb, vc) / (b * c)))
    beta = np.degrees(np.arccos(np.dot(va, vc) / (a * c)))
    gamma = np.degrees(np.arccos(np.dot(va, vb) / (a * b)))
    return a, b, c, alpha, beta, gamma


def _sgrp_lattice_args(
    sgnum: int, a: float, b: float, c: float, alpha: float, beta: float, gamma: float
) -> tuple[float, ...]:
    """Select the (a[, b, c, angles...]) subset SGLattice expects for *sgnum*."""
    if sgnum >= 195:  # cubic
        return (a,)
    if sgnum >= 75:  # tetragonal, trigonal, hexagonal
        return (a, c)
    if sgnum >= 16:  # orthorhombic
        return (a, b, c)
    if sgnum >= 3:  # monoclinic
        return (a, b, c, beta)
    return (a, b, c, alpha, beta, gamma)  # triclinic


def symmetrize_crystal_from_p1(
    lattice: np.ndarray,
    frac_coords: np.ndarray,
    species: Sequence[str],
    *,
    symprec: float = 1e-2,
    name: str | None = None,
    b_factor: float = 0.0,
) -> xu.materials.Crystal:
    """
    Symmetrize a P1 structure (e.g. DFT-relaxed) into an xrayutilities Crystal.

    Uses spglib to detect the true space group and produce an idealized,
    standardized cell. Each symmetry-independent atom is then handed to
    :class:`xu.materials.SGLattice`, which identifies its Wyckoff site from
    the space group automatically.

    Args:
        lattice (np.ndarray): 3x3 array of lattice vectors (rows), in Å.
        frac_coords (np.ndarray): (N, 3) fractional atomic coordinates, one
            row per atom. No symmetry is assumed (P1).
        species (Sequence[str]): Length-N element symbols, same order as
            *frac_coords*.
        symprec (float): spglib distance tolerance (Å) used to detect
            symmetry. DFT-relaxed structures are rarely exactly symmetric —
            scan a few values (e.g. ``1e-3`` to ``1e-1``) and check where the
            detected space group number stabilizes.
        name (str, optional): Crystal name. Defaults to the spglib
            international symbol (e.g. ``"Fm-3m"``).
        b_factor (float): Isotropic Debye–Waller *B*-factor applied to every
            site (default 0.0).

    Returns:
        xu.materials.Crystal: Crystal built from the symmetrized,
            standardized cell.

    Raises:
        RuntimeError: If spglib detects no symmetry at *symprec* (the input
            noise exceeds the tolerance — try loosening *symprec*).
    """
    numbers = [xraylib.SymbolToAtomicNumber(s) for s in species]
    cell = (np.asarray(lattice), np.asarray(frac_coords), numbers)

    ds = spglib.get_symmetry_dataset(cell, symprec=symprec)
    if ds is None:
        raise RuntimeError(
            f"spglib found no symmetry at symprec={symprec}; "
            "try loosening symprec for a DFT-relaxed structure."
        )

    # Re-run on the standardized/idealized cell to find symmetry-equivalent
    # atoms within it, so only one representative per Wyckoff orbit is kept
    # — SGLattice expands the rest itself.
    std = spglib.get_symmetry_dataset(
        (ds.std_lattice, ds.std_positions, ds.std_types), symprec=symprec
    )
    reps = sorted(set(std.equivalent_atoms))

    atoms = [xraylib.AtomicNumberToSymbol(int(std.std_types[i])) for i in reps]
    pos = [tuple(std.std_positions[i]) for i in reps]
    occ = [1.0] * len(reps)
    bvals = [b_factor] * len(reps)

    a, b, c, alpha, beta, gamma = _lattice_params(ds.std_lattice)
    lat_args = _sgrp_lattice_args(ds.number, a, b, c, alpha, beta, gamma)

    lat = xu.materials.SGLattice(
        ds.number, *lat_args, atoms=atoms, pos=pos, occ=occ, b=bvals
    )
    return xu.materials.Crystal(name or ds.international.replace("/", "_"), lat)
