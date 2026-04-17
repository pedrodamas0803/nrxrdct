"""
layered_structure_factor.py
============================
Model a stack of crystalline layers with known orientation relationships (OR)
and compute the total kinematical structure factor  F(Q)  and intensity  |F(Q)|².

Physics
-------
In the kinematical (Born) approximation the total scattered amplitude is the
coherent sum of contributions from every unit cell in the stack:

    A(Q) = Σ_{layers l}  F_l(Q_cry_l) · Σ_{n=0}^{N_l-1} exp(i Q·R_{l,n})

where
    Q          – scattering vector in the lab frame  (Å⁻¹)
    Q_cry_l    – Q expressed in the crystal frame of layer l  =  U_l^T · Q
    F_l(Q_cry) – unit-cell structure factor of layer l
                 computed by xrayutilities with energy-dependent
                 Cromer-Mann + Henke anomalous scattering factors
    R_{l,n}    – position of the nth unit cell of layer l
                 = z0_l · ẑ  +  n · c_l · ẑ  (1D stacking along ẑ)
    z0_l       – cumulative thickness of all layers below layer l

The geometric sum  Σ exp(i n φ)  is evaluated analytically:
    (1 − exp(iNφ)) / (1 − exp(iφ))  if  φ mod 2π ≠ 0,  else  N

For a superlattice with N_rep bilayer repetitions an additional geometric
factor is applied over the full bilayer period Λ.

Orientation relationship
------------------------
The orientation U of each layer maps crystal-frame vectors to the lab frame:
    G_lab = U @ G_crystal

The OR between two phases A and B is specified by two direction pairs
(primary and secondary), defining a rotation R_OR such that:
    U_B = R_OR @ U_A

Common ORs implemented as helpers:
    or_nishiyama_wassermann  (NW)  – FCC/BCC
    or_kurdjumov_sachs       (KS)  – FCC/BCC
    or_baker_nutting         (BN)  – BCC/rock-salt (e.g. Fe/MgO)
    or_from_directions            – generic two-direction specification

Usage
-----
    from layered_structure_factor import LayeredCrystal, or_kurdjumov_sachs
    import xrayutilities as xu
    import numpy as np

    Fe = xu.materials.Fe
    Cu = xu.materials.Cu

    # Orient Fe with [001] along the stacking direction (lab z)
    U_Fe = orientation_along_z([0, 0, 1], Fe)

    # Derive Cu orientation via Kurdjumov-Sachs OR
    U_Cu = or_kurdjumov_sachs(Fe, Cu) @ U_Fe

    # Build the stack: 20 Fe unit cells / 20 Cu unit cells, 5 repetitions
    stack = LayeredCrystal()
    stack.add_layer(Fe, U_Fe, n_cells=20, label='Fe')
    stack.add_layer(Cu, U_Cu, n_cells=20, label='Cu')
    stack.set_repetitions(5)

    # Compute |F(Q)|² along a scan
    qz = np.linspace(1.0, 6.0, 4000)
    Q_scan = np.column_stack([np.zeros_like(qz),
                              np.zeros_like(qz), qz])   # along lab z
    intensity = stack.intensity(Q_scan, energy_eV=17000)

    # Or single Q-point
    Q = np.array([0., 0., 3.09])
    F  = stack.structure_factor(Q, energy_eV=17000)
    print(f'|F(Q)| = {abs(F):.4f} e.u.')
"""

import numpy as np
from scipy.spatial.transform import Rotation
import xrayutilities as xu

# ─────────────────────────────────────────────────────────────────────────────
# ORIENTATION UTILITIES
# ─────────────────────────────────────────────────────────────────────────────


def crystal_to_cartesian(uvw, crystal):
    """
    Convert Miller direction [uvw] to a Cartesian vector using the crystal's
    reciprocal lattice (for directions, use the direct metric).

    For a general crystal with lattice parameters a, b, c, α, β, γ the direct
    metric tensor G is used:  x_cart = A · [uvw]  where A is the lattice matrix.

    Parameters
    ----------
    uvw     : array-like, shape (3,)   Miller direction indices
    crystal : xu.materials.Crystal

    Returns
    -------
    cart : ndarray, shape (3,)   Cartesian vector (Å)
    """
    lat = crystal.lattice
    # Build direct lattice matrix A (columns = a1, a2, a3 vectors)
    A = lat._ai  # shape (3,3), rows are a1, a2, a3
    return A.T @ np.array(uvw, dtype=float)


def orientation_along_z(zone_axis_crystal, crystal, up_crystal=None):
    """
    Build a 3×3 orientation matrix U that places the crystal direction
    ``zone_axis_crystal`` along the lab +z axis (stacking direction).

    Optionally align ``up_crystal`` as close as possible to lab +x.

    Parameters
    ----------
    zone_axis_crystal : array-like  Miller direction [uvw] in the crystal frame
    crystal           : xu.materials.Crystal
    up_crystal        : array-like, optional  secondary alignment direction

    Returns
    -------
    U : ndarray, shape (3,3)   orientation matrix  (G_lab = U @ G_crystal)
    """
    b = crystal_to_cartesian(zone_axis_crystal, crystal)
    b /= np.linalg.norm(b)
    z = np.array([0.0, 0.0, 1.0])

    ax = np.cross(b, z)
    if np.linalg.norm(ax) < 1e-10:
        U = np.eye(3) if np.dot(b, z) > 0 else -np.eye(3)
    else:
        ax /= np.linalg.norm(ax)
        ang = np.arccos(np.clip(np.dot(b, z), -1.0, 1.0))
        R1 = Rotation.from_rotvec(ang * ax).as_matrix()
        if up_crystal is not None:
            u = crystal_to_cartesian(up_crystal, crystal)
            ur = R1 @ u
            up_perp = ur - np.dot(ur, z) * z
            if np.linalg.norm(up_perp) > 1e-10:
                up_perp /= np.linalg.norm(up_perp)
                x = np.array([1.0, 0.0, 0.0])
                twist = np.arctan2(np.dot(np.cross(up_perp, x), z), np.dot(up_perp, x))
                R2 = Rotation.from_rotvec(twist * z).as_matrix()
                U = R2 @ R1
            else:
                U = R1
        else:
            U = R1
    return U


def _or_from_two_pairs(v1_A, v2_A, v1_B, v2_B):
    """
    Rotation R such that R maps v1_A -> v1_B (primary)
    and R @ v2_A is as close as possible to v2_B (secondary).
    All vectors are in Cartesian (lab or crystal) coordinates.
    """

    def make_frame(e1, e2):
        e1 = e1 / np.linalg.norm(e1)
        e3 = np.cross(e1, e2)
        e3 /= np.linalg.norm(e3)
        e2 = np.cross(e3, e1)
        return np.column_stack([e1, e2, e3])

    FA = make_frame(v1_A, v2_A)
    FB = make_frame(v1_B, v2_B)
    return FB @ FA.T


def or_from_directions(crystal_A, dir1_A, dir2_A, crystal_B, dir1_B, dir2_B):
    """
    Compute the orientation relationship rotation  R_OR  such that
    crystal direction ``dir1_A`` in phase A is parallel to ``dir1_B`` in B,
    and ``dir2_A`` is as close as possible to ``dir2_B``.

    Usage:
        R_OR = or_from_directions(Fe, [1,1,1], [1,-1,0],
                                  Cu, [1,1,0], [1,-1,-2])
        U_Cu = R_OR @ U_Fe

    Parameters
    ----------
    crystal_A, crystal_B : xu.materials.Crystal
    dir1_A, dir2_A       : Miller direction in phase A  [u,v,w]
    dir1_B, dir2_B       : Miller direction in phase B  [u,v,w]

    Returns
    -------
    R_OR : ndarray, shape (3,3)   rotation matrix
    """
    v1A = crystal_to_cartesian(dir1_A, crystal_A)
    v2A = crystal_to_cartesian(dir2_A, crystal_A)
    v1B = crystal_to_cartesian(dir1_B, crystal_B)
    v2B = crystal_to_cartesian(dir2_B, crystal_B)
    return _or_from_two_pairs(v1A, v2A, v1B, v2B)


# ── Named orientation relationships ──────────────────────────────────────────


def or_kurdjumov_sachs(crystal_bcc, crystal_fcc):
    """
    Kurdjumov-Sachs (KS) OR between BCC and FCC phases:
        {110}_BCC ∥ {111}_FCC
        <111>_BCC ∥ <110>_FCC

    Returns R_OR  (rotation from BCC crystal frame to FCC crystal frame).
    To derive U_FCC from U_BCC use:   U_FCC = U_BCC @ R_OR.T
    """
    return or_from_directions(
        crystal_bcc, [1, 1, 1], [1, -1, 0], crystal_fcc, [1, 1, 0], [1, -1, -2]
    )


def or_nishiyama_wassermann(crystal_bcc, crystal_fcc):
    """
    Nishiyama-Wassermann (NW) OR between BCC and FCC phases:
        {110}_BCC ∥ {111}_FCC
        <100>_BCC ∥ <011>_FCC

    Returns R_OR  (rotation from BCC crystal frame to FCC crystal frame).
    To derive U_FCC from U_BCC use:   U_FCC = U_BCC @ R_OR.T
    """
    return or_from_directions(
        crystal_bcc, [1, 1, 0], [0, 0, 1], crystal_fcc, [1, 1, 1], [0, 1, -1]
    )


def or_baker_nutting(crystal_bcc, crystal_rocksalt):
    """
    Baker-Nutting (BN) OR between BCC metal and rock-salt oxide
    (e.g. Fe / MgO, Fe / FeO):
        {100}_BCC ∥ {100}_RS
        <110>_BCC ∥ <010>_RS

    Returns R_OR (BCC->rock-salt). Use: U_RS = U_BCC @ R_OR.T
    """
    return or_from_directions(
        crystal_bcc, [1, 0, 0], [1, 1, 0], crystal_rocksalt, [1, 0, 0], [0, 1, 0]
    )  # [010]_RS perpendicular to primary


def or_pitsch(crystal_bcc, crystal_fcc):
    """
    Pitsch OR between BCC and FCC:
        {100}_BCC ∥ {110}_FCC
        <011>_BCC ∥ <111>_FCC

    Returns R_OR  (rotation from BCC crystal frame to FCC crystal frame).
    To derive U_FCC from U_BCC use:   U_FCC = U_BCC @ R_OR.T
    """
    return or_from_directions(
        crystal_bcc, [0, 1, 1], [1, 0, 0], crystal_fcc, [1, 1, 1], [1, 0, -1]
    )


# ─────────────────────────────────────────────────────────────────────────────
# LAYER
# ─────────────────────────────────────────────────────────────────────────────


class Layer:
    """
    One crystalline layer in the stack.

    Parameters
    ----------
    crystal     : xu.materials.Crystal
    U           : (3,3) orientation matrix   G_lab = U @ G_crystal
    n_cells     : int   number of unit cells along the stacking direction
    d_spacing   : float, optional
        Repeat distance along the stacking direction (Å).
        If None, the component of the c lattice vector along lab-z is used,
        or the smallest d-spacing relevant to the zone axis.
    label       : str, optional   name for this layer
    """

    def __init__(self, crystal, U, n_cells, d_spacing=None, label=None):
        self.crystal = crystal
        self.U = np.asarray(U, dtype=float)
        self.n_cells = int(n_cells)
        self.label = label or crystal.name

        if d_spacing is not None:
            self.d = float(d_spacing)
        else:
            # Find the primitive real-space repeat along lab Z.
            #
            # The geometric sum in structure_factor() uses phase  φ = Q_z · d,
            # so d must be the smallest positive z-component of any lattice
            # vector when rotated to the lab frame.  Using only the c-vector
            # z-projection (old code) is wrong for orientations where c is not
            # along lab Z (e.g. [110] stacking via KS OR: c ⊥ Z → c_lab[2]≈0
            # → fallback to lat.c, giving φ = 2π√2 at the [110] Bragg peak
            # instead of the correct 2π).
            lat = crystal.lattice
            z_comps = []
            for vec in lat._ai:          # rows: a1, a2, a3
                v_lab = self.U @ np.asarray(vec, dtype=float)
                z = abs(v_lab[2])
                if z > 1e-6:
                    z_comps.append(z)
            if z_comps:
                self.d = min(z_comps)
            else:
                # All three lattice vectors lie in the XY plane — degenerate;
                # fall back to |c| as a safe non-zero value.
                self.d = lat.c

    @property
    def thickness(self):
        """Total layer thickness in Å."""
        return self.n_cells * self.d

    def structure_factor(self, Q_lab, energy_eV, z0=0.0):
        """
        Kinematical structure factor of this layer at scattering vector Q_lab.

        F_layer(Q) = F_uc(Q_crystal) · Σ_{n=0}^{N-1} exp(i Q_z (z0 + n·d))
                   = F_uc(Q_crystal) · exp(i Q_z z0) · geo_sum(Q_z·d, N)

        Parameters
        ----------
        Q_lab    : array-like (3,)   scattering vector in lab frame  (Å⁻¹)
        energy_eV: float             photon energy  (eV)
        z0       : float             cumulative z-offset of this layer  (Å)

        Returns
        -------
        F : complex   kinematical structure factor (electron units)
        """
        Q = np.asarray(Q_lab, dtype=float)

        # Q in crystal frame
        Q_cry = self.U.T @ Q

        # Unit-cell structure factor (Cromer-Mann + anomalous)
        F_uc = self.crystal.StructureFactor(Q_cry, en=energy_eV)
        if not (np.isfinite(F_uc.real) and np.isfinite(F_uc.imag)):
            return 0.0 + 0j  # Q outside Cromer-Mann range

        # Geometric sum along z
        Qz = Q[2]
        phi = Qz * self.d

        phi_mod = phi % (2.0 * np.pi)
        if abs(phi_mod) < 1e-10 or abs(phi_mod - 2 * np.pi) < 1e-10:
            geo_sum = self.n_cells + 0j
        else:
            geo_sum = (1.0 - np.exp(1j * self.n_cells * phi)) / (1.0 - np.exp(1j * phi))

        phase_z0 = np.exp(1j * Qz * z0)
        return F_uc * phase_z0 * geo_sum

    def __repr__(self):
        return (
            f"Layer('{self.label}', {self.n_cells} cells × "
            f"{self.d:.4f} Å = {self.thickness:.2f} Å)"
        )


# ─────────────────────────────────────────────────────────────────────────────
# LAYERED CRYSTAL STACK
# ─────────────────────────────────────────────────────────────────────────────


class LayeredCrystal:
    """
    A stack of crystalline layers with specified orientations,
    optionally repeated as a superlattice.

    The stacking direction is always the lab +z axis.

    Example
    -------
    >>> stack = LayeredCrystal(name='Fe/Cu KS superlattice')
    >>> stack.add_layer(Fe, U_Fe, n_cells=20, label='Fe')
    >>> stack.add_layer(Cu, U_Cu, n_cells=20, label='Cu')
    >>> stack.set_repetitions(10)          # 10 bilayer repetitions
    >>>
    >>> # Structure factor at a single Q
    >>> F = stack.structure_factor([0, 0, 3.09], energy_eV=17000)
    >>>
    >>> # Intensity along a qz scan
    >>> qz   = np.linspace(1.0, 6.0, 4000)
    >>> Q_arr = np.column_stack([np.zeros((len(qz), 2)), qz])
    >>> I     = stack.intensity(Q_arr, energy_eV=17000)
    """

    def __init__(self, name="layered_crystal"):
        self.name = name
        self.layers = []  # list of Layer objects (one bilayer unit)
        self.n_rep = 1  # number of bilayer repetitions
        self._z_offsets = []  # cumulative z of each layer in one bilayer

    # ── Building the stack ────────────────────────────────────────────────────

    def add_layer(self, crystal, U, n_cells, d_spacing=None, label=None):
        """
        Append a layer to the repeating unit cell (bilayer).

        Parameters
        ----------
        crystal   : xu.materials.Crystal
        U         : (3,3) orientation matrix
        n_cells   : int   number of unit cells along stacking direction
        d_spacing : float, optional   stacking repeat distance (Å)
        label     : str, optional
        """
        layer = Layer(crystal, U, n_cells, d_spacing=d_spacing, label=label)
        self.layers.append(layer)
        self._update_offsets()
        return self

    def set_repetitions(self, n):
        """Set the number of times the bilayer unit is repeated."""
        self.n_rep = int(n)
        return self

    def _update_offsets(self):
        """Recompute cumulative z-offsets of each layer within one bilayer."""
        self._z_offsets = []
        z = 0.0
        for layer in self.layers:
            self._z_offsets.append(z)
            z += layer.thickness
        self._bilayer_thickness = z

    @property
    def bilayer_thickness(self):
        """Thickness of one repeating unit (Å)."""
        self._update_offsets()
        return self._bilayer_thickness

    @property
    def total_thickness(self):
        """Total stack thickness (Å)."""
        return self.n_rep * self.bilayer_thickness

    # ── Structure factor ──────────────────────────────────────────────────────

    def structure_factor(self, Q_lab, energy_eV):
        """
        Total kinematical structure factor of the stack at Q_lab.

        For a superlattice with N_rep repetitions of the bilayer unit:

            F_total(Q) = [ Σ_layers  F_layer(Q, z0_l) ] · S_rep(Q)

        where S_rep is the geometric factor for N_rep repetitions of the
        bilayer with period Λ:

            S_rep(Q) = Σ_{m=0}^{N_rep-1} exp(i m Q_z Λ)

        Parameters
        ----------
        Q_lab    : array-like (3,)   scattering vector in lab frame  (Å⁻¹)
        energy_eV: float             photon energy  (eV)

        Returns
        -------
        F : complex   total structure factor (electron units)
        """
        self._update_offsets()
        Q = np.asarray(Q_lab, dtype=float)
        Qz = Q[2]

        # Sum over layers within one bilayer
        F_bilayer = 0.0 + 0j
        for layer, z0 in zip(self.layers, self._z_offsets):
            F_bilayer += layer.structure_factor(Q, energy_eV, z0=z0)

        # Geometric sum over N_rep bilayer repetitions
        Lambda = self._bilayer_thickness
        phi_rep = Qz * Lambda
        phi_mod = phi_rep % (2.0 * np.pi)
        if abs(phi_mod) < 1e-10 or abs(phi_mod - 2 * np.pi) < 1e-10:
            S_rep = self.n_rep + 0j
        else:
            S_rep = (1.0 - np.exp(1j * self.n_rep * phi_rep)) / (
                1.0 - np.exp(1j * phi_rep)
            )

        return F_bilayer * S_rep

    def intensity(self, Q_arr, energy_eV):
        """
        Compute |F(Q)|² for an array of Q-points.

        Parameters
        ----------
        Q_arr    : array-like, shape (N, 3)  scattering vectors in lab frame
        energy_eV: float                      photon energy (eV)

        Returns
        -------
        I : ndarray, shape (N,)   |F(Q)|² in (electron units)²
        """
        Q_arr = np.asarray(Q_arr, dtype=float)
        return np.array([abs(self.structure_factor(Q, energy_eV)) ** 2 for Q in Q_arr])

    # ── Per-layer and per-Q analysis ─────────────────────────────────────────

    def layer_contributions(self, Q_lab, energy_eV):
        """
        Return the individual structure factor contribution of each layer type.

        Parameters
        ----------
        Q_lab    : array-like (3,)
        energy_eV: float

        Returns
        -------
        dict  { label : complex F_layer }
        """
        self._update_offsets()
        Q = np.asarray(Q_lab, dtype=float)
        result = {}
        for layer, z0 in zip(self.layers, self._z_offsets):
            F = layer.structure_factor(Q, energy_eV, z0=z0)
            result[layer.label] = result.get(layer.label, 0j) + F
        return result

    # ── Description ──────────────────────────────────────────────────────────

    def describe(self):
        """Print a summary of the stack."""
        self._update_offsets()
        Lambda = self._bilayer_thickness
        print(f"\n  LayeredCrystal: '{self.name}'")
        print(f"  {'─'*52}")
        print(f"  Layers in bilayer unit:")
        for i, (layer, z0) in enumerate(zip(self.layers, self._z_offsets)):
            lat = layer.crystal.lattice
            print(
                f"    [{i}] {layer.label:20s}  "
                f"{layer.n_cells:4d} cells × {layer.d:.4f} Å = "
                f"{layer.thickness:8.3f} Å   (z0 = {z0:.3f} Å)"
            )
            print(
                f"         SG {lat.space_group}  "
                f"a={lat.a:.4f} b={lat.b:.4f} c={lat.c:.4f} Å"
            )
        print(f"  Bilayer period Λ = {Lambda:.4f} Å")
        print(f"  Repetitions      = {self.n_rep}")
        print(
            f"  Total thickness  = {self.total_thickness:.2f} Å  "
            f"= {self.total_thickness/10:.2f} nm"
        )
        print(f"  Satellite spacing 2π/Λ = {2*np.pi/Lambda:.5f} Å⁻¹")
        print(f"  {'─'*52}")


# ─────────────────────────────────────────────────────────────────────────────
# DEMO / SELF-TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("=" * 60)
    print("  layered_structure_factor.py  –  demonstration")
    print("=" * 60)

    # ── Materials ─────────────────────────────────────────────────────────────
    Fe = xu.materials.Fe  # BCC, a = 2.8665 Å, SG 229

    Cr_lat = xu.materials.SGLattice(229, 2.884, atoms=["Cr"], pos=["2a"])
    Cr = xu.materials.Crystal("Cr", Cr_lat)

    Cu = xu.materials.Cu  # FCC, a = 3.6150 Å, SG 225

    # ─────────────────────────────────────────────────────────────────────────
    # Example 1: Fe/Cr superlattice (BCC/BCC, [001] || z for both)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[1] Fe/Cr BCC superlattice  –  [001] stacking, no OR needed")

    # Both BCC with similar lattice parameters -> trivial OR (same orientation)
    U_Fe_001 = orientation_along_z([0, 0, 1], Fe)
    U_Cr_001 = orientation_along_z([0, 0, 1], Cr)

    stack_FeCr = LayeredCrystal(name="Fe20/Cr20 × 10  [001]")
    stack_FeCr.add_layer(Fe, U_Fe_001, n_cells=20, label="Fe")
    stack_FeCr.add_layer(Cr, U_Cr_001, n_cells=20, label="Cr")
    stack_FeCr.set_repetitions(10)
    stack_FeCr.describe()

    # Scan along qz (BCC [001] rod)
    qz = np.linspace(0.5, 5.5, 5000)
    Q_scan = np.column_stack([np.zeros((len(qz), 2)), qz])
    I_FeCr = stack_FeCr.intensity(Q_scan, energy_eV=17000)

    # ─────────────────────────────────────────────────────────────────────────
    # Example 2: Fe/Cu with Kurdjumov-Sachs OR
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[2] Fe(BCC) / Cu(FCC) – Kurdjumov-Sachs orientation relationship")
    print("    (111)_Fe ∥ (110)_Cu   AND   [1-10]_Fe ∥ [11-2]_Cu")

    # Stack along [110]_Fe (i.e. the plane that is parallel in KS)
    # crystal_to_cartesian is imported from this module
    U_Fe_110 = orientation_along_z([1, 1, 0], Fe, up_crystal=[0, 0, 1])
    R_KS = or_kurdjumov_sachs(Fe, Cu)
    # U_Cu = U_Fe @ R_OR.T  because:
    #   G_lab = U @ G_crystal, so U_Cu @ v_Cu = U_Fe @ v_Fe (same lab direction)
    #   R_OR maps Fe-crystal-frame -> Cu-crystal-frame: v_Cu = R_OR @ v_Fe
    #   => U_Cu @ R_OR @ v_Fe = U_Fe @ v_Fe  =>  U_Cu = U_Fe @ R_OR.T
    U_Cu_KS = U_Fe_110 @ R_KS.T

    # Verify OR directly on R_OR (not on U-rotated lab vectors,
    # which are additionally constrained by the stacking direction):
    #   R_OR maps Fe crystal frame -> Cu crystal frame
    #   Primary:   R_OR @ v1_Fe  should be parallel to v1_Cu
    #   Secondary: R_OR @ v2_Fe  should be parallel to v2_Cu
    def _check_or(R, crystal_A, dir_A, crystal_B, dir_B, label):
        vA = crystal_to_cartesian(dir_A, crystal_A)
        vA /= np.linalg.norm(vA)
        vB = crystal_to_cartesian(dir_B, crystal_B)
        vB /= np.linalg.norm(vB)
        mapped = R @ vA
        ang = np.degrees(np.arccos(np.clip(np.dot(mapped, vB), -1, 1)))
        print(f"  OR check {label}: {ang:.4f}° (should be 0°)")

    _check_or(R_KS, Fe, [1, 1, 1], Cu, [1, 1, 0], "[111]_Fe || [110]_Cu  (primary)")
    _check_or(
        R_KS, Fe, [1, -1, 0], Cu, [1, -1, -2], "[1-10]_Fe || [1-1-2]_Cu (secondary)"
    )

    # d-spacing along stacking direction
    d_Fe = Fe.lattice.a / np.sqrt(2)  # d(110)
    d_Cu = Cu.lattice.a / np.sqrt(2)  # d(110)

    stack_FeCu = LayeredCrystal(name="Fe15/Cu15 × 8  KS-OR  [110] stacking")
    stack_FeCu.add_layer(Fe, U_Fe_110, n_cells=15, d_spacing=d_Fe, label="Fe  [110]")
    stack_FeCu.add_layer(Cu, U_Cu_KS, n_cells=15, d_spacing=d_Cu, label="Cu  [110]_KS")
    stack_FeCu.set_repetitions(8)
    stack_FeCu.describe()

    I_FeCu = stack_FeCu.intensity(Q_scan, energy_eV=17000)

    # ─────────────────────────────────────────────────────────────────────────
    # Example 3: Layer contribution analysis at a specific Q
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[3] Layer contribution analysis at qz = 3.09 Å⁻¹")
    Q_pt = np.array([0.0, 0.0, 3.09])
    contribs = stack_FeCr.layer_contributions(Q_pt, energy_eV=17000)
    for lbl, F in contribs.items():
        print(
            f"    {lbl:25s}  |F| = {abs(F):10.3f}  "
            f"phase = {np.angle(F)*180/np.pi:+7.2f}°"
        )
    F_total = stack_FeCr.structure_factor(Q_pt, energy_eV=17000)
    print(
        f"    {'TOTAL':25s}  |F| = {abs(F_total):10.3f}  "
        f"|F|² = {abs(F_total)**2:.1f}"
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Plot
    # ─────────────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(12, 9))
    fig.patch.set_facecolor("#0d1117")

    for ax, I, stack, title, col in [
        (
            axes[0],
            I_FeCr,
            stack_FeCr,
            "Fe/Cr BCC superlattice  [001]  –  20/20 cells × 10 rep",
            "#4fc3f7",
        ),
        (
            axes[1],
            I_FeCu,
            stack_FeCu,
            "Fe(BCC)/Cu(FCC)  KS-OR  [110] stacking  –  15/15 cells × 8 rep",
            "#ff9f43",
        ),
    ]:
        ax.set_facecolor("#080c14")
        ax.semilogy(qz, I, color=col, lw=0.7)

        # Mark bilayer satellite positions
        Lambda = stack.bilayer_thickness
        Qz_main = 2 * np.pi / stack.layers[0].d  # approx main Bragg peak
        for m in range(-5, 6):
            Qz_sat = Qz_main + m * 2 * np.pi / Lambda
            if qz[0] < Qz_sat < qz[-1]:
                ax.axvline(Qz_sat, color="white", lw=0.5, alpha=0.3, ls="--")
                if m == 0:
                    ax.text(
                        Qz_sat,
                        I.max() * 0.5,
                        " main",
                        color="white",
                        fontsize=6,
                        va="center",
                    )
                else:
                    ax.text(
                        Qz_sat,
                        I.max() * 0.1,
                        f" {m:+d}",
                        color="#aaaaaa",
                        fontsize=5.5,
                        va="center",
                    )

        ax.set_xlabel("q_z  (Å⁻¹)", color="#7788aa", fontsize=9)
        ax.set_ylabel("|F(Q)|²  (arb.)", color="#7788aa", fontsize=9)
        ax.set_title(title, color="#ccccee", fontsize=9, pad=5)
        ax.set_xlim(qz[0], qz[-1])
        ax.tick_params(colors="#7788aa", labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor("#1a1f2e")
        ax.grid(True, which="both", ls=":", lw=0.3, color="#1a1f2e")

        info = (
            f"Λ = {Lambda:.2f} Å  |  "
            f"2π/Λ = {2*np.pi/Lambda:.4f} Å⁻¹  |  "
            f"E = 17 keV"
        )
        ax.text(
            0.99,
            0.97,
            info,
            transform=ax.transAxes,
            ha="right",
            va="top",
            color="#888899",
            fontsize=7,
        )

    plt.tight_layout()
    out = "/mnt/user-data/outputs/layered_structure_factor.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"\n  Figure saved -> {out}")
    print("\n  Done. Import with:")
    print(
        "    from layered_structure_factor import "
        "LayeredCrystal, or_kurdjumov_sachs, "
        "or_nishiyama_wassermann, or_baker_nutting, "
        "or_from_directions, orientation_along_z"
    )
