"""Tests for nrxrdct.powder.structures.symmetrize_crystal_from_p1."""
from __future__ import annotations

import numpy as np
import pytest

from nrxrdct.powder.structures import symmetrize_crystal_from_p1


def test_fcc_p1_recovers_fm3m():
    """A P1 listing of the 4 FCC basis atoms should resolve to space group 225."""
    a = 4.05
    lattice = np.eye(3) * a
    frac_coords = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
    ])
    species = ["Al"] * 4

    crystal = symmetrize_crystal_from_p1(lattice, frac_coords, species)

    assert crystal.lattice.space_group_nr == 225
    assert crystal.name == "Fm-3m"
    assert crystal.a == pytest.approx(a)
    assert crystal.b == pytest.approx(a)
    assert crystal.c == pytest.approx(a)
    # all 4 atoms collapse to a single Wyckoff orbit (4a)
    assert crystal.lattice.nsites == 1


def test_bcc_p1_recovers_im3m():
    """A P1 listing of the 2 BCC basis atoms should resolve to space group 229."""
    a = 2.87
    lattice = np.eye(3) * a
    frac_coords = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.5],
    ])
    species = ["Fe"] * 2

    crystal = symmetrize_crystal_from_p1(lattice, frac_coords, species)

    assert crystal.lattice.space_group_nr == 229
    assert crystal.a == pytest.approx(a)
    assert crystal.lattice.nsites == 1


def test_custom_name_and_b_factor():
    a = 4.05
    lattice = np.eye(3) * a
    frac_coords = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
    ])
    species = ["Al"] * 4

    crystal = symmetrize_crystal_from_p1(
        lattice, frac_coords, species, name="MyAl", b_factor=0.3
    )

    assert crystal.name == "MyAl"
    _, _, _, b = next(iter(crystal.lattice.base()))
    assert b == pytest.approx(0.3)


def test_hexagonal_p1_recovers_hcp_spacegroup():
    """A P1 listing of both HCP basis atoms (Wyckoff 2c) resolves to sg 194."""
    a, c = 2.95, 4.68
    lattice = np.array([[a, 0, 0], [-a / 2, a * np.sqrt(3) / 2, 0], [0, 0, c]])
    frac_coords = np.array([[1 / 3, 2 / 3, 1 / 4], [2 / 3, 1 / 3, 3 / 4]])

    crystal = symmetrize_crystal_from_p1(lattice, frac_coords, ["Ti", "Ti"])

    assert crystal.lattice.space_group_nr == 194
    assert crystal.a == pytest.approx(a)
    assert crystal.c == pytest.approx(c)
    assert crystal.lattice.gamma == pytest.approx(120.0)


def test_orthorhombic_p1_single_atom():
    """A single atom in a general orthorhombic P cell has full site symmetry
    (mmm), so spglib resolves it to Pmmm (sg 47) — exercises the a,b,c branch.
    """
    lattice = np.diag([4.0, 5.0, 6.0])
    frac_coords = np.array([[0.0, 0.0, 0.0]])

    crystal = symmetrize_crystal_from_p1(lattice, frac_coords, ["Fe"])

    assert crystal.lattice.space_group_nr == 47
    assert (crystal.a, crystal.b, crystal.c) == pytest.approx((4.0, 5.0, 6.0))


def test_tetragonal_p1_single_atom():
    """a == b != c, single atom -> exercises the (a, c) branch."""
    lattice = np.diag([4.0, 4.0, 6.0])
    frac_coords = np.array([[0.0, 0.0, 0.0]])

    crystal = symmetrize_crystal_from_p1(lattice, frac_coords, ["Fe"])

    assert crystal.lattice.crystal_system.startswith("tetragonal")
    assert crystal.a == pytest.approx(4.0)
    assert crystal.c == pytest.approx(6.0)


def test_monoclinic_p1_single_atom():
    """a, b, c distinct with beta != 90 -> exercises the (a, b, c, beta) branch."""
    beta = np.radians(100.0)
    lattice = np.array([
        [4.0, 0.0, 0.0],
        [0.0, 5.0, 0.0],
        [6.0 * np.cos(beta), 0.0, 6.0 * np.sin(beta)],
    ])
    frac_coords = np.array([[0.0, 0.0, 0.0]])

    crystal = symmetrize_crystal_from_p1(lattice, frac_coords, ["Fe"])

    assert crystal.lattice.crystal_system.startswith("monoclinic")
    assert (crystal.a, crystal.b, crystal.c) == pytest.approx((4.0, 5.0, 6.0))
    assert crystal.lattice.beta == pytest.approx(100.0)


def test_triclinic_p1_single_atom():
    """Fully general cell, single atom -> exercises the 6-parameter branch."""
    lattice = np.array([[4.0, 0.0, 0.0], [0.5, 5.0, 0.0], [0.3, 0.7, 6.0]])
    frac_coords = np.array([[0.0, 0.0, 0.0]])

    crystal = symmetrize_crystal_from_p1(lattice, frac_coords, ["Fe"])

    assert crystal.lattice.space_group_nr in (1, 2)
    assert crystal.lattice.nsites == 1


def test_no_symmetry_detected_raises():
    """A degenerate (zero-volume) cell makes spglib return no dataset, which
    should surface as a RuntimeError rather than an opaque AttributeError.
    """
    lattice = np.array([[5.0, 0.0, 0.0], [5.0, 0.0, 0.0], [0.0, 0.0, 5.0]])
    frac_coords = np.array([[0.1, 0.2, 0.3]])
    species = ["Si"]

    with pytest.raises(RuntimeError):
        symmetrize_crystal_from_p1(lattice, frac_coords, species)
