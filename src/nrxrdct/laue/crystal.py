"""
crystal_from_cif.py
===================
Utility to build an xrayutilities Crystal object from a CIF file,
ready for use in Laue / powder diffraction simulations.

Supports:
  - CIF files on disk  (pass a file path)
  - CIF content as a Python string  (useful for embedded or downloaded CIFs)
  - Multiple datasets within a single CIF file
  - Partial site occupancies
  - Isotropic displacement parameters (Biso / Uiso -> b-factor)
  - All crystal systems handled by xrayutilities.materials.CIFFile

**Quick usage**
    from crystal_from_cif import crystal_from_cif

    # From a file on disk
    crystal = crystal_from_cif('my_phase.cif')

    # From a CIF string (e.g. downloaded from COD / ICSD / CCDC)
    crystal = crystal_from_cif(cif_string, name='My Phase')

    # Use in Laue simulation  (see laue_white_synchrotron.py)
    spots = simulate_laue(crystal, U, camera)

    # Use in powder simulation
    import xrayutilities as xu
    import numpy as np
    pm = xu.simpack.PowderModel(xu.simpack.Powder(crystal, 1.0), I0=1e6)
    tt = np.linspace(10, 100, 2000)
    pattern = pm.simulate(tt)
    pm.close()
"""

import os

import numpy as np
import xrayutilities as xu

from .layers import LayeredCrystal, nitride_elastic_constants, orientation_along_z

# ─────────────────────────────────────────────────────────────────────────────
# MAIN FUNCTION
# ─────────────────────────────────────────────────────────────────────────────


def crystal_from_cif(cif_source, name=None, dataset=None, use_p1=False, verbose=True):
    """
    Build an `xu.materials.Crystal` from a CIF file or CIF string.

    Args:
        cif_source (str): Either a path to a `.cif` file OR the raw CIF text as a string.
            xrayutilities automatically detects which one is provided.

        name (str, optional): Name to give the Crystal object.
            If None the function tries (in order):
            1. The dataset name embedded in the CIF  (`data_<name>` block)
            2. The base filename without extension    (when a file path is given)
            3. `'crystal'`                         (fallback)

        dataset (str, optional): Name of the data block to use when the CIF contains multiple datasets.
            If None the first dataset that contains atomic positions is used.

        use_p1 (bool, optional): Force P1 symmetry (space group 1), expanding all atoms explicitly.
            Useful when the CIF's symmetry operations are incomplete or non-standard.
            Default: False.

        verbose (bool, optional): Print a summary of the parsed structure. Default: True.

    Returns:
        crystal (xu.materials.Crystal): Ready-to-use Crystal object with the correct SGLattice (space group,
            lattice parameters, Wyckoff positions, occupancies, B-factors).

    Raises:
        FileNotFoundError: If `cif_source` looks like a file path but the file does not exist.
        ValueError: If the CIF contains no datasets with atomic positions, or if a
            requested `dataset` name is not found in the file.
        RuntimeError: If xrayutilities cannot identify the space group from the CIF data.

    Note:
    xrayutilities uses the Cromer-Mann parameterisation for atomic scattering
    factors f0(Q) and the Henke tables for anomalous corrections f'(E), f''(E).
    These are automatically assigned based on the element symbols in the CIF.

    The `_atom_site_U_iso_or_equiv` tag (if present) is converted to the
    B-factor used by xrayutilities via  B = 8*pi^2 * U.

    **Common CIF sources**
    - Crystallography Open Database (COD):  https://www.crystallography.net/cod/
    - ICSD (subscription):                 https://icsd.fiz-karlsruhe.de/
    - Materials Project:                   https://next-gen.materialsproject.org/
    - CCDC (organics):                     https://www.ccdc.cam.ac.uk/

    Example:
    >>> # From a file
    >>> fe = crystal_from_cif('bcc_iron.cif')

    >>> # From a string
    >>> cif_text = open('al2o3.cif').read()
    >>> al2o3 = crystal_from_cif(cif_text, name='Corundum')

    >>> # Use in Laue simulation
    >>> import numpy as np
    >>> from scipy.spatial.transform import Rotation
    >>> U = Rotation.from_euler('ZXZ', [0, 90, 0], degrees=True).as_matrix()
    >>> G = al2o3.Q(1, 0, 4)
    >>> F = al2o3.StructureFactor(G, en=17000)
    >>> print(f'|F(104)| = {abs(F):.3f} e.u.')

    >>> # Use in powder simulation
    >>> import xrayutilities as xu
    >>> pm = xu.simpack.PowderModel(xu.simpack.Powder(fe, 1.0), I0=1e6)
    >>> tt = np.linspace(20, 100, 2000)
    >>> pattern = pm.simulate(tt)
    >>> pm.close()
"""

    # ── Input validation ──────────────────────────────────────────────────────
    if not isinstance(cif_source, str):
        raise TypeError(
            f"cif_source must be a str (file path or CIF text), "
            f"got {type(cif_source).__name__}"
        )

    is_filepath = os.path.isfile(cif_source)
    if not is_filepath and "\n" not in cif_source:
        # Looks like it was meant to be a path but does not exist
        raise FileNotFoundError(
            f"CIF file not found: '{cif_source}'\n"
            "Pass a valid file path or the raw CIF text as a string."
        )

    # ── Parse CIF ─────────────────────────────────────────────────────────────
    try:
        cif = xu.materials.CIFFile(cif_source)
    except Exception as exc:
        raise RuntimeError(f"xrayutilities failed to parse CIF: {exc}") from exc

    if not cif.data:
        raise ValueError("No datasets found in CIF source.")

    # ── Select dataset ────────────────────────────────────────────────────────
    if dataset is not None:
        if dataset not in cif.data:
            available = list(cif.data.keys())
            raise ValueError(
                f"Dataset '{dataset}' not found in CIF. " f"Available: {available}"
            )
        ds_key = dataset
    else:
        ds_key = cif.default_dataset
        if ds_key is None:
            # No dataset with atoms found; try the first one anyway
            ds_key = next(iter(cif.data))

    ds = cif.data[ds_key]

    if not ds.has_atoms:
        raise ValueError(
            f"Dataset '{ds_key}' contains no atomic positions. "
            "Check the CIF _atom_site_* fields."
        )

    # ── Build SGLattice ───────────────────────────────────────────────────────
    try:
        lattice = cif.SGLattice(dataset=ds_key, use_p1=use_p1)
    except Exception as exc:
        raise RuntimeError(
            f"xrayutilities could not build SGLattice from dataset '{ds_key}': {exc}\n"
            "Try use_p1=True to bypass symmetry identification."
        ) from exc

    # ── Determine crystal name ────────────────────────────────────────────────
    if name is None:
        # Try dataset name (from 'data_<name>' block, may be set to chemical formula)
        ds_name = getattr(ds, "name", ds_key) or ds_key
        if ds_name and ds_name != ds_key:
            name = ds_name
        elif is_filepath:
            name = os.path.splitext(os.path.basename(cif_source))[0]
        else:
            name = ds_key or "crystal"

    # Sanitise name (xrayutilities dislikes some special characters)
    name = str(name).strip().replace("\n", " ")

    # ── Build Crystal ─────────────────────────────────────────────────────────
    crystal = xu.materials.Crystal(name, lattice)

    # ── Report ────────────────────────────────────────────────────────────────
    if verbose:
        lat = crystal.lattice
        n_atoms_unique = len(ds.atoms)
        n_atoms_uc = (
            sum(len(pos) for pos in ds.unique_positions)
            if hasattr(ds, "unique_positions")
            else "?"
        )

        # Count unique elements
        elements = sorted({str(a[0]) for a in ds.atoms})

        print(f"\n  Crystal loaded from CIF")
        print(f"  {'─'*48}")
        print(f"  Name          : {crystal.name}")
        if is_filepath:
            print(f"  File          : {cif_source}")
        print(f"  Dataset       : {ds_key}")
        print(f"  Space group   : {lat.space_group}  " f"({getattr(lat, 'name', '')})")
        print(f"  Crystal system: {lat.crystal_system}")
        print(f"  a = {lat.a:.5f} Å    " f"b = {lat.b:.5f} Å    " f"c = {lat.c:.5f} Å")
        print(
            f"  α = {lat.alpha:.3f}°   "
            f"β = {lat.beta:.3f}°   "
            f"γ = {lat.gamma:.3f}°"
        )
        print(f"  V = {lat.UnitCellVolume():.4f} Å³")
        print(f"  Elements      : {', '.join(elements)}")
        print(f"  Unique sites  : {n_atoms_unique}")
        if getattr(ds, "occ", None):
            any_partial = any(abs(o - 1.0) > 0.01 for o in ds.occ)
            if any_partial:
                print(f"  Occupancies   : partial occupancy detected")
        if getattr(ds, "biso", None) and any(b != 0 for b in ds.biso):
            print(f"  Biso range    : " f"{min(ds.biso):.3f} – {max(ds.biso):.3f} Å²")
        print(f"  {'─'*48}")

    return crystal


# ─────────────────────────────────────────────────────────────────────────────
# CONVENIENCE: load multiple phases from a list of CIF sources
# ─────────────────────────────────────────────────────────────────────────────


def crystals_from_cifs(cif_sources, names=None, verbose=True):
    """
    Load multiple Crystal objects from a list of CIF files or strings.

    Args:
        cif_sources (list of str): List of file paths or CIF text strings.
        names (list of str, optional): Names for each crystal. If None, names are inferred from the CIFs.
        verbose (bool, optional): Print summaries. Default True.

    Returns:
        crystals (list of xu.materials.Crystal):

    Example:
    >>> bcc, b2 = crystals_from_cifs(['bcc.cif', 'b2.cif'])
"""
    if names is None:
        names = [None] * len(cif_sources)
    if len(names) != len(cif_sources):
        raise ValueError("len(names) must equal len(cif_sources)")

    return [
        crystal_from_cif(src, name=n, verbose=verbose)
        for src, n in zip(cif_sources, names)
    ]


def build_bcc(a=2.881):
    lat = xu.materials.SGLattice(
        229, a, atoms=["Al", "Co", "Cr", "Fe", "Ni"], pos=["2a"] * 5, occ=[0.2] * 5
    )
    return xu.materials.Crystal("BCC  Im-3m", lat)


def build_b2(a=2.881):
    lat = xu.materials.SGLattice(
        221,
        a,
        atoms=["Al", "Ni", "Co", "Cr", "Fe"],
        pos=["1a", "1a", "1b", "1b", "1b"],
        occ=[0.5, 0.5, 1 / 3, 1 / 3, 1 / 3],
    )
    return xu.materials.Crystal("B2   Pm-3m", lat)


# ─────────────────────────────────────────────────────────────────────────────
# III-NITRIDE WURTZITE BINARIES AND ALLOYS
# ─────────────────────────────────────────────────────────────────────────────


def _build_wurtzite_nitride(name, element, a, c):
    """Build a wurtzite (SG 186) binary nitride the same way xrayutilities builds GaN."""
    ce = nitride_elastic_constants(name)
    lat = xu.materials.SGLattice(
        186, a, c, atoms=[element, "N"], pos=[("2b", 0), ("2b", 3 / 8.0)]
    )
    cij = xu.materials.HexagonalElasticTensor(
        ce["C11"] * 1e9, ce["C12"] * 1e9, ce["C13"] * 1e9, ce["C33"] * 1e9, ce["C44"] * 1e9
    )
    return xu.materials.Crystal(name, lat, cij)


def build_aln(a=3.112, c=4.982):
    """AlN, wurtzite (Vurgaftman & Meyer 2003 lattice constants)."""
    return _build_wurtzite_nitride("AlN", "Al", a, c)


def build_inn(a=3.545, c=5.703):
    """InN, wurtzite (Vurgaftman & Meyer 2003 lattice constants)."""
    return _build_wurtzite_nitride("InN", "In", a, c)

def micron2angstrom(num):
    return num * 1e4

def nano2angstrom(num):
    return num * 10

def build_MLed(
    UB_GaN=np.eye(3),
    UB_subst = np.eye(3),
    x_in_defect=0.03,
    x_in_active=0.15,
    x_al_clad=0.10,
    x_in_clad=0.02,
    x_al_ebl=0.20,
):
    """
    Build a single `LayeredCrystal` for an EBL/QW nitride LED-laser stack,
    with four sections stacked bottom (substrate) to top (surface):

        Al2O3 / GaN undoped / GaN doped                    buffer (non-repeating)
        80 nm defect-filtering  : InGaN / GaN MQW           5x
        40 nm active region     : InGaN / GaN MQW           4x
        80 nm optical cladding  : InGaAlN / GaAlN MQW       8x
        160 nm AlGaN electron-blocking layer (EBL)          1x  (cap, top)

    Each MQW section is its own independently-repeated block (one
    `add_pseudomorphic_layer` pair followed by `set_repetitions`) within the
    *same* `LayeredCrystal` -- see :meth:`LayeredCrystal.add_layer` for how
    block boundaries work. The EBL cap is added with `add_layer` (not
    `add_buffer_layer`): buffer layers always sit at the very bottom of the
    stack, so a genuine top-side cap layer must be its own trailing block
    instead. All MQW/cap layers are grown pseudomorphically on a common GaN
    template (`a_substrate=GaN.lattice.a`), [001] (c-axis) growth.
    Compositions are illustrative -- adjust to your real epitaxy.
    Wells/barriers split each group's total thickness evenly.

    Returns:
        LayeredCrystal: the full stack, with `.blocks` = [defect (×5), active (×4),
        clad (×8), ebl (×1)] on top of `.buffer_layers` = [substrate, undoped, doped].

    Example:
        >>> stack = build_MLed()
        >>> stack.describe()
        >>> from nrxrdct.laue.simulation import simulate_laue_stack
        >>> spots = simulate_laue_stack(combine_stacks([stack]), camera, allowed_hkl=allowed)
    """
    GaN = xu.materials.GaN
    Al2O3 = xu.materials.Al2O3
    AlN = build_aln()
    InN = build_inn()

    InGaN_defect = xu.materials.Alloy(GaN, InN, x_in_defect)
    InGaN_active = xu.materials.Alloy(GaN, InN, x_in_active)
    GaAlN_clad = xu.materials.Alloy(GaN, AlN, x_al_clad)
    InGaAlN_clad = xu.materials.Alloy(GaAlN_clad, InN, x_in_clad)
    AlGaN_ebl = xu.materials.Alloy(GaN, AlN, x_al_ebl)

    a_sub = GaN.lattice.a
    c_GaN = nitride_elastic_constants("GaN")
    # Elastic constants for each alloy, interpolated at its own composition
    # (Vegard's law), matching how each Alloy crystal above was built.
    c_InGaN_defect = nitride_elastic_constants("InN", x_in_defect, "GaN")
    c_InGaN_active = nitride_elastic_constants("InN", x_in_active, "GaN")
    c_GaAlN_clad = nitride_elastic_constants("AlN", x_al_clad, "GaN")
    c_AlGaN_ebl = nitride_elastic_constants("AlN", x_al_ebl, "GaN")
    # InGaAlN_clad is a quaternary (GaAlN base + InN): interpolate InN into
    # the already-interpolated GaAlN_clad constants for the same x_in_clad.
    c_InGaAlN_clad = {
        k: x_in_clad * nitride_elastic_constants("InN")[k] + (1 - x_in_clad) * c_GaAlN_clad[k]
        for k in ("C11", "C12", "C13", "C33", "C44")
    }

    stacking_dir = UB_GaN @ np.array([0, 0, 2])

    stacking_dir /= np.linalg.norm(stacking_dir)

    
    stack = LayeredCrystal(name = "LED", stacking_direction=stacking_dir)

    subst= stack.add_buffer_layer(Al2O3, UB_subst, micron2angstrom(600), label = 'Al2O3 Substrate')

    buf = stack.add_buffer_layer(GaN, UB_GaN, micron2angstrom(1.8), label = 'GaN undoped layer '  )

    dop = stack.add_buffer_layer(GaN, UB_GaN, micron2angstrom(3.2), label = 'GaN doped layer '  )

    # defect-filtering layer
    defect = stack.add_pseudomorphic_layer(GaN, UB_GaN, nano2angstrom(8), a_sub, c_GaN["C13"], c_GaN["C33"], label = 'Defect filtering')
    defect = stack.add_pseudomorphic_layer(InGaN_defect, UB_GaN, nano2angstrom(8), a_sub, c_InGaN_defect["C13"], c_InGaN_defect["C33"], label = 'Defect filtering')
    defect.set_repetitions(5)

    # active region

    active  = stack.add_pseudomorphic_layer(GaN, UB_GaN, nano2angstrom(5), a_sub, c_GaN["C13"], c_GaN["C33"], label = 'Active zone')
    active = stack.add_pseudomorphic_layer(InGaN_active, UB_GaN, nano2angstrom(5), a_sub, c_InGaN_active["C13"], c_InGaN_active["C33"], label = 'Active zone')
    active.set_repetitions(4)

    # optical cladding zone
    clad  = stack.add_pseudomorphic_layer(GaAlN_clad, UB_GaN, nano2angstrom(5), a_sub, c_GaAlN_clad["C13"], c_GaAlN_clad["C33"], label = 'Optical cladding')
    clad = stack.add_pseudomorphic_layer(InGaAlN_clad, UB_GaN, nano2angstrom(5), a_sub, c_InGaAlN_clad["C13"], c_InGaAlN_clad["C33"], label = 'Optical cladding')
    clad.set_repetitions(8)

    # electron blocking layer -- a non-repeating CAP on top of the clad MQW.
    # Must use add_layer (its own trailing block, n_rep=1), not
    # add_buffer_layer: buffer layers are always placed at the very bottom
    # of the stack, so add_buffer_layer here would put the EBL underneath
    # the defect/active/clad blocks instead of on top of them.
    ebl = stack.add_layer(AlGaN_ebl, UB_GaN, nano2angstrom(160), label='Electron blocking layer')

    return stack

