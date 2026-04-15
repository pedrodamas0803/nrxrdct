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

Quick usage
-----------
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

# ─────────────────────────────────────────────────────────────────────────────
# MAIN FUNCTION
# ─────────────────────────────────────────────────────────────────────────────


def crystal_from_cif(cif_source, name=None, dataset=None, use_p1=False, verbose=True):
    """
    Build an ``xu.materials.Crystal`` from a CIF file or CIF string.

    Parameters
    ----------
    cif_source : str
        Either a path to a ``.cif`` file OR the raw CIF text as a string.
        xrayutilities automatically detects which one is provided.

    name : str, optional
        Name to give the Crystal object.
        If None the function tries (in order):
          1. The dataset name embedded in the CIF  (``data_<name>`` block)
          2. The base filename without extension    (when a file path is given)
          3. ``'crystal'``                         (fallback)

    dataset : str, optional
        Name of the data block to use when the CIF contains multiple datasets.
        If None the first dataset that contains atomic positions is used.

    use_p1 : bool, optional
        Force P1 symmetry (space group 1), expanding all atoms explicitly.
        Useful when the CIF's symmetry operations are incomplete or non-standard.
        Default: False.

    verbose : bool, optional
        Print a summary of the parsed structure. Default: True.

    Returns
    -------
    crystal : xu.materials.Crystal
        Ready-to-use Crystal object with the correct SGLattice (space group,
        lattice parameters, Wyckoff positions, occupancies, B-factors).

    Raises
    ------
    FileNotFoundError
        If ``cif_source`` looks like a file path but the file does not exist.
    ValueError
        If the CIF contains no datasets with atomic positions, or if a
        requested ``dataset`` name is not found in the file.
    RuntimeError
        If xrayutilities cannot identify the space group from the CIF data.

    Notes
    -----
    xrayutilities uses the Cromer-Mann parameterisation for atomic scattering
    factors f0(Q) and the Henke tables for anomalous corrections f'(E), f''(E).
    These are automatically assigned based on the element symbols in the CIF.

    The ``_atom_site_U_iso_or_equiv`` tag (if present) is converted to the
    B-factor used by xrayutilities via  B = 8*pi^2 * U.

    Common CIF sources
    ------------------
    - Crystallography Open Database (COD):  https://www.crystallography.net/cod/
    - ICSD (subscription):                 https://icsd.fiz-karlsruhe.de/
    - Materials Project:                   https://next-gen.materialsproject.org/
    - CCDC (organics):                     https://www.ccdc.cam.ac.uk/

    Examples
    --------
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

    Parameters
    ----------
    cif_sources : list of str
        List of file paths or CIF text strings.
    names : list of str, optional
        Names for each crystal. If None, names are inferred from the CIFs.
    verbose : bool, optional
        Print summaries. Default True.

    Returns
    -------
    crystals : list of xu.materials.Crystal

    Example
    -------
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
