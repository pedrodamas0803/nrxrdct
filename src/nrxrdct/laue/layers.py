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
    stack.add_layer(Fe, U_Fe, thickness=57.4, label='Fe')   # ~20 cells × 2.87 Å
    stack.add_layer(Cu, U_Cu, thickness=72.6, label='Cu')   # ~20 cells × 3.63 Å
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
import xrayutilities as xu
from scipy.spatial.transform import Rotation

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
# EPITAXIAL STRAIN
# ─────────────────────────────────────────────────────────────────────────────

# Elastic stiffness constants (GPa) for common wurtzite III-nitrides.
# Sources: Wright (1997) PRB 55, 6250  /  Vurgaftman & Meyer (2003) JAP 94, 3675.
# All values for the hexagonal (wurtzite) phase; C44 included for completeness
# but not needed for the biaxial strain formula.
_NITRIDE_ELASTIC = {
    #          C11    C12    C13    C33    C44
    "GaN": (390.0, 145.0, 106.0, 398.0, 105.0),
    "InN": (223.0, 115.0, 92.0, 224.0, 48.0),
    "AlN": (396.0, 137.0, 108.0, 373.0, 116.0),
}


def nitride_elastic_constants(material: str, x: float = 0.0, end_material: str = "GaN"):
    """
    Elastic stiffness constants for a binary or ternary III-nitride (GPa).

    For ternary alloys the constants are linearly interpolated between the
    two binary end-members (Vegard's law approximation).

    Parameters
    ----------
    material : ``'GaN'`` | ``'InN'`` | ``'AlN'``
        First end-member (or the only material when ``x=0``).
    x : float, optional
        Alloy fraction of ``material`` in the ternary (default 0).
        Example: for In₀.₂Ga₀.₈N pass
        ``material='InN', x=0.2, end_material='GaN'``.
    end_material : ``'GaN'`` | ``'InN'`` | ``'AlN'``, optional
        Second end-member (default ``'GaN'``).

    Returns
    -------
    dict with keys ``'C11'``, ``'C12'``, ``'C13'``, ``'C33'``, ``'C44'``
    (all in GPa).

    Examples
    --------
    >>> c = nitride_elastic_constants('GaN')
    >>> c = nitride_elastic_constants('InN', x=0.20, end_material='GaN')
    >>> d, eps_par, eps_perp = pseudomorphic_d_spacing(
    ...     InGaN, GaN.lattice.a, C13=c['C13'], C33=c['C33'])
    """
    for m in (material, end_material):
        if m not in _NITRIDE_ELASTIC:
            raise ValueError(
                f"Unknown nitride {m!r}.  Available: {list(_NITRIDE_ELASTIC)}"
            )
    keys = ("C11", "C12", "C13", "C33", "C44")
    v1 = _NITRIDE_ELASTIC[material]
    v2 = _NITRIDE_ELASTIC[end_material]
    return {k: x * a + (1.0 - x) * b for k, a, b in zip(keys, v1, v2)}


def d_spacing_hkl(crystal, h, k, l):
    """
    Interplanar spacing of the ``(hkl)`` family for *crystal*.

    Uses the reciprocal lattice directly, so it is valid for any crystal
    system (cubic, hexagonal, orthorhombic, triclinic, …):

    .. math::

        d_{hkl} = \\frac{2\\pi}{|\\mathbf{G}_{hkl}|}
        \\qquad
        \\mathbf{G}_{hkl} = h\\,\\mathbf{b}_1 + k\\,\\mathbf{b}_2 + l\\,\\mathbf{b}_3

    where :math:`\\mathbf{b}_i` are the reciprocal-lattice basis vectors in
    the xrayutilities convention (:math:`\\mathbf{b}_i \\cdot \\mathbf{a}_j =
    2\\pi\\,\\delta_{ij}`).

    Parameters
    ----------
    crystal : xu.materials.Crystal
    h, k, l : int or float
        Miller indices.

    Returns
    -------
    float
        d-spacing in Å.

    Examples
    --------
    >>> import xrayutilities as xu
    >>> GaN = xu.materials.GaN
    >>> d_spacing_hkl(GaN, 0, 0, 2)   # GaN 002 reflection
    2.593...
    """
    b1, b2, b3 = crystal.lattice._bi   # reciprocal vectors, Å⁻¹ (2π convention)
    G = h * np.asarray(b1) + k * np.asarray(b2) + l * np.asarray(b3)
    return float(2 * np.pi / np.linalg.norm(G))


def pseudomorphic_d_spacing(
    crystal_film,
    a_substrate,
    C13: float,
    C33: float,
    growth_dir=(0, 0, 1),
) -> float:
    """
    Out-of-plane repeat distance for a pseudomorphic (coherently strained) film.

    Assuming biaxial in-plane stress with a free surface perpendicular to the
    growth direction, the out-of-plane strain is:

    .. math::

        \\varepsilon_\\perp = -\\frac{2C_{13}}{C_{33}}\\,\\varepsilon_\\parallel
        \\qquad
        \\varepsilon_\\parallel = \\frac{a_{\\text{sub}} - a_{\\text{film}}}{a_{\\text{film}}}

    The strained repeat along ``growth_dir`` is then
    :math:`d = d_{\\text{bulk}}(1 + \\varepsilon_\\perp)`.

    This formula is exact for hexagonal **c-axis** growth and for cubic
    **[001]** growth (substitute :math:`C_{12}/C_{11}` for
    :math:`C_{13}/C_{33}`).

    .. warning::

        **Not valid for non-c-axis hexagonal or off-axis cubic growth.**
        For semipolar / non-polar orientations (e.g. GaN grown along
        [2,-2,0], [1,1,-2,3], etc.) the in-plane strain is anisotropic and
        the correct out-of-plane response requires rotating the full
        elastic stiffness tensor into the growth frame.  Calling this
        function with a non-c-axis ``growth_dir`` for a hexagonal crystal
        will raise ``ValueError``.

    Parameters
    ----------
    crystal_film : xu.materials.Crystal
        Bulk (relaxed) film crystal.
    a_substrate : float  or  xu.materials.Crystal
        In-plane lattice parameter of the template / substrate (Å).
        If a Crystal is passed its ``.lattice.a`` is used.
    C13, C33 : float
        Elastic stiffness constants in any consistent units (GPa or Pa).

        * Hexagonal c-axis growth  → :math:`C_{13}`, :math:`C_{33}`
        * Cubic [001] growth       → :math:`C_{12}`, :math:`C_{11}`
    growth_dir : array-like (3,), optional
        Miller direction of the stacking / growth axis in the **film's crystal
        frame** (default: ``(0, 0, 1)``).
        For hexagonal crystals only ``(0, 0, 1)`` (c-axis) is supported.

    Returns
    -------
    d_strained : float
        Strained repeat along ``growth_dir`` (Å).  Pass directly as
        ``d_spacing`` to :meth:`LayeredCrystal.add_layer`.
    eps_par : float
        In-plane strain :math:`\\varepsilon_\\parallel` (positive = tensile).
    eps_perp : float
        Out-of-plane strain :math:`\\varepsilon_\\perp`.

    Examples
    --------
    GaN/InGaN LED — pseudomorphic InGaN well on GaN buffer::

        d_InGaN, eps_par, eps_perp = pseudomorphic_d_spacing(
            InGaN, GaN.lattice.a, C13=92.0, C33=224.0)
        stack.add_layer(InGaN, U_InGaN, thickness=n_InGaN * d_InGaN,
                        d_spacing=d_InGaN, label="InGaN (strained)")
    """
    # Resolve substrate in-plane parameter
    if hasattr(a_substrate, "lattice"):
        a_sub = float(a_substrate.lattice.a)
    else:
        a_sub = float(a_substrate)

    lat = crystal_film.lattice
    A = lat._ai  # rows are direct lattice vectors in Cartesian (Å)

    growth_dir = np.asarray(growth_dir, dtype=float)

    # Guard: for hexagonal lattices (a == b, γ == 120°) only c-axis growth is
    # supported.  Non-c-axis hexagonal growth requires a rotated compliance
    # tensor calculation that this scalar formula cannot provide.
    _is_hexagonal = (
        abs(lat.a - lat.b) < 1e-4
        and abs(lat.alpha - 90.0) < 0.5
        and abs(lat.beta - 90.0) < 0.5
        and abs(lat.gamma - 120.0) < 0.5
    )
    if _is_hexagonal:
        g_norm = growth_dir / np.linalg.norm(growth_dir)
        c_axis = np.array([0.0, 0.0, 1.0])
        if abs(abs(float(np.dot(g_norm, c_axis))) - 1.0) > 1e-3:
            raise ValueError(
                f"pseudomorphic_d_spacing: growth_dir={growth_dir.tolist()} is not "
                f"the c-axis for a hexagonal crystal.  For non-c-axis hexagonal "
                f"orientations (semipolar / non-polar) the scalar biaxial formula "
                f"ε_⊥ = -2(C₁₃/C₃₃)·ε_∥ is not valid.  Provide d_spacing manually "
                f"from a full elastic-tensor calculation."
            )

    g_cart = A.T @ growth_dir
    g_cart /= np.linalg.norm(g_cart)

    # Bulk repeat along growth direction
    projs_along_g = []
    projs_inplane = []
    for vec in A:
        along = abs(float(np.dot(vec, g_cart)))
        if along > 1e-6:
            projs_along_g.append(along)
        inplane_vec = vec - np.dot(vec, g_cart) * g_cart
        inplane_len = float(np.linalg.norm(inplane_vec))
        if inplane_len > 1e-6:
            projs_inplane.append(inplane_len)

    d_bulk = min(projs_along_g) if projs_along_g else float(lat.c)
    a_film = min(projs_inplane) if projs_inplane else float(lat.a)

    eps_par = (a_sub - a_film) / a_film
    eps_perp = -2.0 * (C13 / C33) * eps_par
    d_strained = d_bulk * (1.0 + eps_perp)

    return float(d_strained), float(eps_par), float(eps_perp)


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
    thickness   : float   physical thickness of the layer in Å
    n_hat       : array-like (3,), optional
        Unit vector in the **lab frame** that defines the stacking / growth
        direction (= sample-surface normal).  Defaults to ``[0, 0, 1]``
        (lab Z), which is correct when ``U`` was obtained from
        ``orientation_along_z``.  When ``U`` comes from a Laue indexation
        result, pass the growth-direction vector explicitly:
        ``n_hat = U @ growth_dir_crystal / np.linalg.norm(...)``.

        The growth direction always coincides with the sample-surface normal
        (the plane given as ``growth_crystal`` is parallel to the sample
        surface).  The angle between the incident beam (LT-frame x-axis,
        ``[1,0,0]``) and this vector is therefore fixed by the sample
        mounting and is used to correct the Beer-Lambert absorption depth
        when ``absorption_limit=True``.
    d_spacing   : float, optional
        Repeat distance along the stacking direction (Å).
        If ``None``, computed as the primitive lattice repeat along ``n_hat``.
    label       : str, optional   name for this layer
    """

    def __init__(
        self,
        crystal,
        U,
        thickness,
        n_hat=None,
        d_spacing=None,
        label=None,
        absorption_limit=False,
    ):
        self.crystal = crystal
        self.U = np.asarray(U, dtype=float)
        self.label = label or crystal.name
        # When True, structure_factor uses an energy-dependent effective thickness
        # min(real thickness, 1/μ) to model Beer-Lambert absorption depth.
        # Set automatically for buffer layers by LayeredCrystal.add_buffer_layer.
        self.absorption_limit = bool(absorption_limit)

        if n_hat is None:
            self.n_hat = np.array([0.0, 0.0, 1.0])
        else:
            nh = np.asarray(n_hat, dtype=float)
            self.n_hat = nh / np.linalg.norm(nh)

        if d_spacing is not None:
            self.d = float(d_spacing)
        else:
            # Find the primitive real-space repeat along the stacking direction
            # n_hat.  The geometric sum uses phase φ = (Q · n_hat) · d, so d
            # must equal the smallest positive projection of any lattice vector
            # onto n_hat.
            #
            # This is correct for any stacking orientation:
            #   [001] stacking (n_hat = Z):   d = c_param
            #   [110] stacking (n_hat = U@[110]):  d = a/√2  for cubic
            #   U from Laue indexation + growth dir [001]:
            #       n_hat = U @ [0,0,1],  d = projection of c onto n_hat
            lat = crystal.lattice
            proj = []
            for vec in lat._ai:  # rows: a1, a2, a3
                v_lab = self.U @ np.asarray(vec, dtype=float)
                p = abs(float(np.dot(v_lab, self.n_hat)))
                if p > 1e-6:
                    proj.append(p)
            self.d = min(proj) if proj else lat.c

        self.n_cells = max(1, round(float(thickness) / self.d))

    @property
    def thickness(self):
        """Total layer thickness in Å (always the real physical thickness)."""
        return self.n_cells * self.d

    def _linear_mu(self, energy_eV: float) -> float:
        """
        Linear absorption coefficient μ (Å⁻¹) for this material at *energy_eV*.

        Returns ``0.0`` if material data are unavailable or absorption is zero.
        """
        _HC_ANG = 12398.419843
        try:
            if hasattr(self.crystal, "delta_beta"):
                _, beta = self.crystal.delta_beta(energy_eV)
            else:
                elem = getattr(xu.materials.elements, self.crystal.name, None)
                if elem is None or not getattr(elem, "density", 0):
                    return 0.0
                mat = xu.materials.Amorphous(self.crystal.name, elem.density)
                _, beta = mat.delta_beta(energy_eV)
            if not (beta > 0):
                return 0.0
            lam_ang = _HC_ANG / energy_eV
            mu = 4.0 * np.pi * beta / lam_ang
            return float(mu) if mu > 0 else 0.0
        except Exception:
            return 0.0

    def _effective_n_cells(self, energy_eV: float, kf_hat=None) -> int:
        """
        Effective number of unit cells after Beer-Lambert absorption limiting.

        With only the incident beam considered (``kf_hat=None``):

            n_eff = cos_in / (μ · d)

            cos_in = |n̂ · x̂| = |n_hat[0]|

        When the diffracted-beam direction ``kf_hat`` is supplied, the
        **two-beam** formula accounts for both the incident and exit paths
        through the layer:

            n_eff = cos_in · cos_out / (μ · d · (cos_in + cos_out))

            cos_out = |n̂ · k̂_f|

        This is equivalent to using an effective linear attenuation

            μ_eff = μ · (1/cos_in + 1/cos_out)

        which is the standard symmetric absorption correction used in
        surface-diffraction and thin-film rocking-curve analysis.

        Returns ``self.n_cells`` unchanged if the material lookup fails or if
        the absorption depth exceeds the real layer thickness.
        """
        mu = self._linear_mu(energy_eV)
        if mu <= 0:
            return self.n_cells

        cos_in = max(abs(float(self.n_hat[0])), 1e-3)

        if kf_hat is not None:
            kf = np.asarray(kf_hat, dtype=float)
            cos_out = max(abs(float(np.dot(self.n_hat, kf))), 1e-3)
            n_eff = int(min(self.n_cells,
                           cos_in * cos_out / (mu * self.d * (cos_in + cos_out))))
        else:
            n_eff = int(min(self.n_cells, cos_in / (mu * self.d)))

        return max(n_eff, 1)

    def structure_factor(self, Q_lab, energy_eV, z0=0.0, kf_hat=None):
        """
        Kinematical structure factor of this layer at scattering vector Q_lab.

        F_layer(Q) = F_uc(Q_crystal) · Σ_{n=0}^{N-1} exp(i (Q·n̂)(z0 + n·d))

        The phase uses the projection of Q onto the stacking direction ``n_hat``
        (not Q_z), so the result is correct for any sample orientation.

        Parameters
        ----------
        Q_lab    : array-like (3,)   scattering vector in lab frame  (Å⁻¹)
        energy_eV: float             photon energy  (eV)
        z0       : float             cumulative offset along n_hat (Å)

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

        # Projection of Q onto the stacking direction
        Qn = float(np.dot(Q, self.n_hat))
        phi = Qn * self.d

        # Effective cell count: limited by absorption depth for buffer layers.
        # kf_hat enables the two-beam (incident + exit) correction.
        n_eff = (
            self._effective_n_cells(energy_eV, kf_hat=kf_hat)
            if self.absorption_limit
            else self.n_cells
        )

        phi_mod = phi % (2.0 * np.pi)
        if abs(phi_mod) < 1e-10 or abs(phi_mod - 2 * np.pi) < 1e-10:
            geo_sum = n_eff + 0j
        else:
            geo_sum = (1.0 - np.exp(1j * n_eff * phi)) / (1.0 - np.exp(1j * phi))

        phase_z0 = np.exp(1j * Qn * z0)
        return F_uc * phase_z0 * geo_sum

    def __repr__(self):
        return (
            f"Layer('{self.label}', {self.thickness:.2f} Å"
            f" ({self.n_cells} cells × {self.d:.4f} Å))"
        )


# ─────────────────────────────────────────────────────────────────────────────
# LAYERED CRYSTAL STACK
# ─────────────────────────────────────────────────────────────────────────────


class LayeredCrystal:
    """
    A stack of crystalline layers with specified orientations,
    optionally repeated as a superlattice.

    The stacking direction is a lab-frame unit vector ``n_hat`` (default
    ``[0, 0, 1]``, i.e. lab Z).  All phase calculations use the projection
    ``Q · n_hat`` rather than ``Q_z``, so the structure factor is correct
    regardless of how the sample sits on the diffractometer.

    Parameters
    ----------
    name : str, optional
    stacking_direction : array-like (3,), optional
        Unit vector in the **lab frame** defining the growth / stacking
        direction (sample-surface normal).

        - When all ``U`` matrices come from ``orientation_along_z``, the
          stacking direction is lab Z and the default ``[0, 0, 1]`` is correct.
        - When ``U`` comes from a Laue indexation result, pass the actual
          growth direction:  ``stacking_direction = U @ growth_dir_crystal``
          (e.g. ``U @ [0, 0, 1]`` for GaN grown along its c-axis).

    Example — using orientation_along_z (default n_hat = Z)
    --------------------------------------------------------
    >>> stack = LayeredCrystal(name='Fe/Cu KS superlattice')
    >>> stack.add_layer(Fe, U_Fe, thickness=57.4, label='Fe')
    >>> stack.add_layer(Cu, U_Cu, thickness=72.6, label='Cu')
    >>> stack.set_repetitions(10)

    Example — using a U matrix from Laue indexation (GaN grown along c)
    --------------------------------------------------------------------
    >>> n_hat = U_GaN @ np.array([0., 0., 1.])   # growth dir in lab frame
    >>> stack = LayeredCrystal(name='GaN/InGaN', stacking_direction=n_hat)
    >>> stack.add_layer(GaN,   U_GaN,   thickness=51800.0, label='GaN')    # ~1000 cells
    >>> stack.add_layer(InGaN, U_InGaN, thickness=259.0,   label='InGaN')  # ~50 cells

    """

    def __init__(self, name="layered_crystal", stacking_direction=None):
        self.name = name
        self.buffer_layers = []   # non-repeating layers (substrate, buffer) — bottom of stack
        self.layers = []          # repeating unit (MQW bilayer)
        self.n_rep = 1            # number of bilayer repetitions
        self._buffer_z_offsets = []
        self._z_offsets = []

        if stacking_direction is None:
            self.n_hat = np.array([0.0, 0.0, 1.0])
        else:
            nh = np.asarray(stacking_direction, dtype=float)
            self.n_hat = nh / np.linalg.norm(nh)

    # ── Building the stack ────────────────────────────────────────────────────

    def add_buffer_layer(self, crystal, U, thickness, d_spacing=None, label=None):
        """
        Append a **non-repeating** layer at the bottom of the stack (substrate
        side), below the repeating MQW / bilayer unit.

        Buffer layers are always added in order from deepest to shallowest:
        the first call places the layer closest to the substrate, subsequent
        calls place layers closer to the surface.

        Parameters
        ----------
        crystal   : xu.materials.Crystal
        U         : (3,3) orientation matrix   G_lab = U @ G_crystal
        thickness : float   physical thickness of the layer in Å
        d_spacing : float, optional   stacking repeat distance (Å)
        label     : str, optional
        """
        layer = Layer(
            crystal,
            U,
            thickness,
            n_hat=self.n_hat,
            d_spacing=d_spacing,
            label=label,
            absorption_limit=True,
        )
        self.buffer_layers.append(layer)
        self._update_offsets()
        return self

    def add_layer(self, crystal, U, thickness, d_spacing=None, label=None):
        """
        Append a layer to the **repeating** unit (MQW / bilayer).

        Layers are stacked in the order they are added; the first call
        places the layer at the bottom of the unit, the last at the top.
        The full unit is then repeated ``n_rep`` times above the buffer layers.

        Parameters
        ----------
        crystal   : xu.materials.Crystal
        U         : (3,3) orientation matrix
        thickness : float   physical thickness of the layer in Å
        d_spacing : float, optional   stacking repeat distance (Å)
        label     : str, optional
        """
        layer = Layer(
            crystal, U, thickness, n_hat=self.n_hat, d_spacing=d_spacing, label=label
        )
        self.layers.append(layer)
        self._update_offsets()
        return self

    def add_pseudomorphic_layer(
        self,
        crystal,
        U,
        thickness,
        a_substrate,
        C13: float,
        C33: float,
        growth_dir=(0, 0, 1),
        label=None,
    ):
        """
        Append a pseudomorphically strained layer to the repeating unit.

        The out-of-plane repeat distance is computed from the biaxial strain
        state imposed by the in-plane lattice constraint:

        .. math::

            d = d_{\\text{bulk}}\\left(1 - \\frac{2C_{13}}{C_{33}}\\,
            \\varepsilon_\\parallel\\right)
            \\qquad
            \\varepsilon_\\parallel =
            \\frac{a_{\\text{sub}} - a_{\\text{film}}}{a_{\\text{film}}}

        Equivalent to calling :func:`pseudomorphic_d_spacing` then
        :meth:`add_layer` with the computed ``d_spacing``.

        Parameters
        ----------
        crystal : xu.materials.Crystal
            Bulk (relaxed) film crystal.
        U : (3,3) ndarray
            Orientation matrix for this layer.
        thickness : float
            Physical thickness of the layer in Å.
        a_substrate : float  or  xu.materials.Crystal
            In-plane lattice parameter of the template (Å), or a Crystal
            whose ``.lattice.a`` is used.
        C13, C33 : float
            Elastic stiffness constants (GPa).

            * Hexagonal c-axis growth → :math:`C_{13}`, :math:`C_{33}`
            * Cubic [001] growth      → :math:`C_{12}`, :math:`C_{11}`
        growth_dir : array-like (3,), optional
            Growth direction in the **film's crystal frame**.
            Default: ``(0, 0, 1)`` (c-axis).
        label : str, optional

        Returns
        -------
        self  (for method chaining)

        Notes
        -----
        The strain values are printed on addition so you can verify the
        mismatch.
        """
        d_strained, eps_par, eps_perp = pseudomorphic_d_spacing(
            crystal, a_substrate, C13, C33, growth_dir
        )
        lbl = label or crystal.name
        print(
            f"  {lbl}: ε_∥ = {eps_par:+.4f}  ε_⊥ = {eps_perp:+.4f}"
            f"  d_bulk → {d_strained / (1 + eps_perp):.4f} Å"
            f"  d_strained = {d_strained:.4f} Å"
        )
        return self.add_layer(crystal, U, thickness, d_spacing=d_strained, label=lbl)

    def set_repetitions(self, n):
        """Set the number of times the repeating unit (MQW bilayer) is stacked."""
        self.n_rep = int(n)
        return self

    def set_U(self, U) -> "LayeredCrystal":
        """
        Replace the orientation matrix of layers in the stack.

        Two calling forms:

        **Global** — apply one ``U`` to every layer::

            stack.set_U(U)

        **Per-material** — supply a dict mapping crystal name to ``U``; only
        layers whose ``crystal.name`` matches a key are updated::

            stack.set_U({'GaN': U_GaN, 'InGaN': U_InGaN})

        Layers whose material is not listed in the dict are left unchanged.
        Useful for modelling domain variants or orientation relationships
        between different materials in the same stack without rebuilding it.

        Parameters
        ----------
        U : array-like (3, 3) or dict[str, array-like (3, 3)]
            A single orientation matrix applied to all layers, or a mapping
            ``{crystal_name: U_matrix}``.

        Returns
        -------
        self  (for method chaining)
        """
        if isinstance(U, dict):
            U_map = {name: np.asarray(mat, dtype=float) for name, mat in U.items()}
            for layer in self.buffer_layers + self.layers:
                if layer.crystal.name in U_map:
                    layer.U = U_map[layer.crystal.name].copy()
        else:
            U_mat = np.asarray(U, dtype=float)
            for layer in self.buffer_layers + self.layers:
                layer.U = U_mat.copy()
        return self

    def _update_offsets(self):
        """Recompute cumulative z-offsets for buffer layers and the repeating unit."""
        # Buffer layers: z = 0 at deepest point, increasing toward surface
        self._buffer_z_offsets = []
        z = 0.0
        for layer in self.buffer_layers:
            self._buffer_z_offsets.append(z)
            z += layer.thickness
        self._buffer_thickness = z

        # Repeating unit offsets (relative to the start of the MQW section)
        self._z_offsets = []
        z = 0.0
        for layer in self.layers:
            self._z_offsets.append(z)
            z += layer.thickness
        self._bilayer_thickness = z

    @property
    def buffer_thickness(self):
        """Total thickness of all non-repeating buffer layers (Å)."""
        self._update_offsets()
        return self._buffer_thickness

    @property
    def bilayer_thickness(self):
        """Thickness of one repeating MQW unit (Å)."""
        self._update_offsets()
        return self._bilayer_thickness

    @property
    def total_thickness(self):
        """Total stack thickness: buffer + n_rep × bilayer (Å)."""
        return self.buffer_thickness + self.n_rep * self.bilayer_thickness

    @property
    def all_layers(self):
        """All layers in stack order: buffer layers then repeating unit layers."""
        return self.buffer_layers + self.layers

    # ── Structure factor ──────────────────────────────────────────────────────

    def structure_factor(self, Q_lab, energy_eV, kf_hat=None):
        """
        Total kinematical structure factor of the stack at Q_lab.

        The stack is divided into two sections:

        1. **Buffer layers** (non-repeating, at the bottom):

               F_buf(Q) = Σ_j  T_above_j · F_layer_j(Q, z0_j)

           where ``T_above_j`` is the amplitude attenuation from all layers
           above layer *j* (both the MQW block and the shallower buffer layers).

        2. **Repeating unit** (MQW / bilayer, sitting on top of the buffer):

               F_MQW(Q) = exp(i·Qₙ·z_buf) · F_unit(Q) · S_rep(Q)

           where ``z_buf`` is the total buffer thickness,
           ``F_unit`` is the single-bilayer structure factor, and

               S_rep(Q) = Σ_{m=0}^{N_rep-1} exp(i·m·Qₙ·Λ)

           is the superlattice geometric factor.

        The total structure factor is ``F_buf + F_MQW``.

        Parameters
        ----------
        Q_lab    : array-like (3,)   scattering vector in lab frame  (Å⁻¹)
        energy_eV: float             photon energy  (eV)
        kf_hat   : array-like (3,) or None
            Unit vector of the diffracted beam in the lab frame.  When
            provided, the **exit-path** absorption through each overlying layer
            is included via the two-beam transmission factor

                T = exp(−μ · t · (1/cos_in + 1/cos_out))

            where ``cos_in = |n̂ · x̂|`` and ``cos_out = |n̂ · k̂_f|``.
            If ``None``, only the incident-path (one-beam) correction already
            embedded in each buffer layer's ``_effective_n_cells`` is applied.

        Returns
        -------
        F : complex   total structure factor (electron units)
        """
        self._update_offsets()
        Q = np.asarray(Q_lab, dtype=float)
        Qn = float(np.dot(Q, self.n_hat))

        # ── Overlying-layer transmission helper ───────────────────────────────
        def _T_slab(lyr, thickness):
            """Amplitude transmission exp(-μ·t·(1/cos_in + 1/cos_out))."""
            if kf_hat is None:
                return 1.0
            mu = lyr._linear_mu(energy_eV)
            if mu <= 0:
                return 1.0
            kf = np.asarray(kf_hat, dtype=float)
            cos_in  = max(abs(float(self.n_hat[0])), 1e-3)
            cos_out = max(abs(float(np.dot(self.n_hat, kf))), 1e-3)
            return float(np.exp(-mu * thickness * (1.0 / cos_in + 1.0 / cos_out)))

        # Attenuation from the full MQW block (sits above all buffer layers)
        T_mqw = 1.0
        for lyr in self.layers:
            T_mqw *= _T_slab(lyr, lyr.thickness * self.n_rep)

        # ── Buffer layers (non-repeating) ─────────────────────────────────────
        F_total = 0.0 + 0j
        for i, (layer, z0) in enumerate(zip(self.buffer_layers, self._buffer_z_offsets)):
            # Attenuation from: MQW above + all buffer layers shallower than i
            T_above = T_mqw
            for j in range(i + 1, len(self.buffer_layers)):
                T_above *= _T_slab(self.buffer_layers[j],
                                   self.buffer_layers[j].thickness)
            F_total += T_above * layer.structure_factor(Q, energy_eV, z0=z0,
                                                        kf_hat=kf_hat)

        # ── Repeating unit (MQW) ──────────────────────────────────────────────
        if self.layers:
            F_unit = 0.0 + 0j
            for layer, z0 in zip(self.layers, self._z_offsets):
                F_unit += layer.structure_factor(Q, energy_eV, z0=z0,
                                                 kf_hat=kf_hat)

            # Phase to shift MQW to its z position above the buffer
            phase_buf = np.exp(1j * Qn * self._buffer_thickness)

            # Geometric sum over N_rep repetitions of the bilayer
            Lambda = self._bilayer_thickness
            phi_rep = Qn * Lambda
            phi_mod = phi_rep % (2.0 * np.pi)
            if abs(phi_mod) < 1e-10 or abs(phi_mod - 2 * np.pi) < 1e-10:
                S_rep = self.n_rep + 0j
            else:
                S_rep = (1.0 - np.exp(1j * self.n_rep * phi_rep)) / (
                    1.0 - np.exp(1j * phi_rep)
                )

            F_total += phase_buf * F_unit * S_rep

        return F_total

    def average_structure_factor(self, Q_lab, energy_eV, kf_hat=None):
        """
        Structure factor of the thickness-weighted average unit cell at Q_lab.

        Instead of summing layer amplitudes with their relative depth phases
        (the coherent model), this method sums ``F_uc_i × N_eff_i`` over all
        layers **without** the inter-layer phase factors
        ``exp(i Q_n z_{0,i})``.  The result is the *structural envelope* of
        the diffraction pattern: the intensity it predicts at any Q is the
        maximum each satellite could reach if all unit cells happened to
        scatter perfectly in phase.

        This approximation is useful to:

        * predict which satellite positions carry significant scattering power
          based on the average composition, before running the slower coherent
          simulation;
        * compare the coherent fringe pattern against its envelope to isolate
          which features arise from constructive interference vs. from the unit
          cell structure factor.

        All absorption corrections (Beer-Lambert two-beam depth limit,
        overlying-layer attenuation) are applied identically to
        :meth:`structure_factor`.

        Parameters
        ----------
        Q_lab : array-like (3,)
            Scattering vector in the lab frame (Å⁻¹).
        energy_eV : float
            Photon energy (eV).
        kf_hat : array-like (3,) or None
            Diffracted beam unit vector for the two-beam absorption correction.

        Returns
        -------
        F : complex  (electron units)
        """
        self._update_offsets()
        Q = np.asarray(Q_lab, dtype=float)
        Qn = float(np.dot(Q, self.n_hat))

        # ── Overlying-layer transmission (identical to structure_factor) ──────
        def _T_slab(lyr, thickness):
            if kf_hat is None:
                return 1.0
            mu = lyr._linear_mu(energy_eV)
            if mu <= 0:
                return 1.0
            kf = np.asarray(kf_hat, dtype=float)
            cos_in  = max(abs(float(self.n_hat[0])), 1e-3)
            cos_out = max(abs(float(np.dot(self.n_hat, kf))), 1e-3)
            return float(np.exp(-mu * thickness * (1.0 / cos_in + 1.0 / cos_out)))

        T_mqw = 1.0
        for lyr in self.layers:
            T_mqw *= _T_slab(lyr, lyr.thickness * self.n_rep)

        # ── Buffer layers — coherent sum with depth phase offsets ─────────────
        # Buffer layers are not part of the periodic unit; their positions are
        # fixed and their phase offsets must be kept even in average mode.
        F_total = 0.0 + 0j
        for i, (layer, z0) in enumerate(
                zip(self.buffer_layers, self._buffer_z_offsets)):
            T_above = T_mqw
            for j in range(i + 1, len(self.buffer_layers)):
                T_above *= _T_slab(self.buffer_layers[j],
                                   self.buffer_layers[j].thickness)
            Q_cry = layer.U.T @ Q
            F_uc = layer.crystal.StructureFactor(Q_cry, en=energy_eV)
            if not (np.isfinite(F_uc.real) and np.isfinite(F_uc.imag)):
                continue
            n_eff = (layer._effective_n_cells(energy_eV, kf_hat=kf_hat)
                     if layer.absorption_limit else layer.n_cells)
            F_total += T_above * F_uc * n_eff * np.exp(1j * Qn * z0)

        # ── Repeating MQW unit — average over the period, keep S_rep ─────────
        # Sum F_uc_i × N_cells_i over one bilayer period WITHOUT intra-period
        # phase factors (z_rel).  The inter-period interference is still
        # captured by the geometric series S_rep, so satellite peaks appear at
        # the correct positions with N_rep²-enhanced intensities.
        if self.layers:
            F_unit = 0.0 + 0j
            for layer in self.layers:
                Q_cry = layer.U.T @ Q
                F_uc = layer.crystal.StructureFactor(Q_cry, en=energy_eV)
                if not (np.isfinite(F_uc.real) and np.isfinite(F_uc.imag)):
                    continue
                F_unit += F_uc * layer.n_cells

            z_buf   = self._buffer_thickness
            Lambda  = self._bilayer_thickness
            phi_rep = Qn * Lambda
            phi_mod = phi_rep % (2.0 * np.pi)
            if abs(phi_mod) < 1e-10 or abs(phi_mod - 2.0 * np.pi) < 1e-10:
                S_rep = float(self.n_rep) + 0j
            else:
                S_rep = ((1.0 - np.exp(1j * self.n_rep * phi_rep))
                         / (1.0 - np.exp(1j * phi_rep)))
            F_total += np.exp(1j * Qn * z_buf) * F_unit * S_rep

        return F_total

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
        Return the individual structure factor contribution of each layer.

        Buffer layers are included at their absolute z positions.  Repeating
        unit layers are included for the *first* repetition only (z0 relative
        to the start of the MQW section), without the superlattice factor.

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
        for layer, z0 in zip(self.buffer_layers, self._buffer_z_offsets):
            F = layer.structure_factor(Q, energy_eV, z0=z0)
            result[layer.label] = result.get(layer.label, 0j) + F
        for layer, z0 in zip(self.layers, self._z_offsets):
            F = layer.structure_factor(Q, energy_eV, z0=self._buffer_thickness + z0)
            result[layer.label] = result.get(layer.label, 0j) + F
        return result

    # ── Description ──────────────────────────────────────────────────────────

    def describe(self):
        """Print a summary of the stack."""
        self._update_offsets()
        nh = self.n_hat
        W = 56

        def _layer_row(i, layer, z0_abs):
            lat = layer.crystal.lattice
            print(
                f"    [{i}] {layer.label:22s} "
                f"{layer.n_cells:5d} cells × {layer.d:.4f} Å"
                f" = {layer.thickness:9.3f} Å   z = {z0_abs:.1f} Å"
            )
            print(
                f"         SG {lat.space_group}  "
                f"a={lat.a:.4f}  b={lat.b:.4f}  c={lat.c:.4f} Å"
            )

        beam_angle_deg = np.degrees(np.arccos(np.clip(abs(float(nh[0])), 0.0, 1.0)))
        print(f"\n  LayeredCrystal: '{self.name}'")
        print(f"  {'─'*W}")
        print(f"  Stacking / surface normal (lab): [{nh[0]:+.4f}, {nh[1]:+.4f}, {nh[2]:+.4f}]"
              f"   beam angle = {beam_angle_deg:.1f}°")

        # Buffer layers
        if self.buffer_layers:
            print(
                f"\n  Buffer layers  (non-repeating, {len(self.buffer_layers)} layer"
                f"{'s' if len(self.buffer_layers) != 1 else ''}):"
            )
            for i, (layer, z0) in enumerate(
                zip(self.buffer_layers, self._buffer_z_offsets)
            ):
                _layer_row(i, layer, z0)
        else:
            print(f"\n  Buffer layers: none")

        # Repeating unit
        if self.layers:
            print(
                f"\n  Repeating unit  (× {self.n_rep},"
                f" starts at z = {self._buffer_thickness:.1f} Å):"
            )
            for i, (layer, z0) in enumerate(zip(self.layers, self._z_offsets)):
                _layer_row(i, layer, self._buffer_thickness + z0)
            Lambda = self._bilayer_thickness
            print(f"    Bilayer period  Λ = {Lambda:.4f} Å")
            if Lambda > 1e-6:
                print(f"    Satellite spacing 2π/Λ = {2*np.pi/Lambda:.5f} Å⁻¹")
        else:
            print(f"\n  Repeating unit: none")

        print(
            f"\n  Total thickness = {self.total_thickness:.2f} Å"
            f"  = {self.total_thickness/10:.2f} nm"
        )
        print(f"  {'─'*W}")

    def plot_lattice_parameter(
        self,
        param: str = "c",
        unit: str = "A",
        ax=None,
        figsize=(8, 4),
    ):
        """
        Plot a lattice parameter profile through the stack depth.

        Draws a step function showing how *param* (``'a'``, ``'b'``, or
        ``'c'``) varies with depth, with buffer layers at the bottom and the
        surface at the top.  The repeating MQW unit is unrolled across all
        ``n_rep`` repetitions.

        Parameters
        ----------
        param : ``'a'`` | ``'b'`` | ``'c'``
            Which lattice parameter to plot.
        unit : ``'A'`` | ``'nm'``
            Display unit for the depth axis (Å or nm).
        ax : matplotlib.axes.Axes, optional
            Draw into an existing axes; a new figure is created if ``None``.
        figsize : (float, float)

        Returns
        -------
        fig, ax
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        if param not in ("a", "b", "c"):
            raise ValueError(f"param must be 'a', 'b', or 'c', got {param!r}")
        if unit not in ("A", "nm"):
            raise ValueError(f"unit must be 'A' or 'nm', got {unit!r}")

        self._update_offsets()
        scale = 0.1 if unit == "nm" else 1.0
        unit_label = "nm" if unit == "nm" else "Å"

        # Build full layer sequence with absolute z offsets (bottom = 0)
        # Each entry: (z_start, z_end, layer)
        segments = []
        z = 0.0
        for layer in self.buffer_layers:
            segments.append((z, z + layer.thickness, layer))
            z += layer.thickness
        for _ in range(self.n_rep):
            for layer in self.layers:
                segments.append((z, z + layer.thickness, layer))
                z += layer.thickness

        # Collect unique phase labels for the legend
        seen_labels = {}
        colors_cycle = plt.get_cmap("tab10")
        color_idx = 0

        standalone = ax is None
        if standalone:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        for z0, z1, layer in segments:
            val = getattr(layer.crystal.lattice, param)
            y0 = z0 * scale
            y1 = z1 * scale
            lbl = layer.label

            if lbl not in seen_labels:
                seen_labels[lbl] = colors_cycle(color_idx / 9)
                color_idx = (color_idx + 1) % 10
            col = seen_labels[lbl]

            # Horizontal bar: lattice parameter value across the layer depth range
            ax.hlines(val, y0, y1, colors=col, linewidths=2.5)
            # Vertical connectors between adjacent layers
            ax.vlines(y0, val, val, colors=col, linewidths=1.0, linestyles="dotted")

        # Draw vertical connectors between adjacent steps
        for i in range(1, len(segments)):
            _, _, layer_prev = segments[i - 1]
            z_curr, _, layer_curr  = segments[i]
            val_prev = getattr(layer_prev.crystal.lattice, param)
            val_curr = getattr(layer_curr.crystal.lattice, param)
            z_joint = z_curr * scale
            ax.vlines(z_joint, min(val_prev, val_curr), max(val_prev, val_curr),
                      colors="#555555", linewidths=0.8, linestyles="--")

        # Legend
        handles = [
            mpatches.Patch(color=col, label=lbl)
            for lbl, col in seen_labels.items()
        ]
        ax.legend(handles=handles, fontsize=8, loc="best",
                  framealpha=0.4, facecolor="#1a1f2e", edgecolor="#3a3f4e",
                  labelcolor="#ccccee")

        ax.set_xlabel(f"depth  ({unit_label})", fontsize=9)
        ax.set_ylabel(f"lattice parameter {param}  (Å)", fontsize=9)
        ax.set_title(
            f"{self.name}  —  lattice parameter {param} profile",
            fontsize=10,
        )
        ax.set_xlim(0, self.total_thickness * scale)

        if standalone:
            fig.tight_layout()

        return fig, ax

    def plot_strain_profile(
        self,
        param: str = "c",
        reference=None,
        unit: str = "A",
        ax=None,
        figsize=(8, 4),
    ):
        """
        Plot the strain profile of a lattice parameter through the stack depth.

        Strain is defined as

        .. math::

            \\varepsilon = \\frac{p_{\\text{layer}} - p_{\\text{ref}}}{p_{\\text{ref}}}

        where *p* is the chosen lattice parameter (``'a'``, ``'b'``, or
        ``'c'``).

        Parameters
        ----------
        param : ``'a'`` | ``'b'`` | ``'c'``
            Lattice parameter to use.
        reference : float, xu.materials.Crystal, or None
            Reference value for zero strain.

            * **float** — use this value directly (Å).
            * **Crystal** — use ``crystal.lattice.<param>``.
            * **None** *(default)* — use the lattice parameter of the first
              buffer layer; if there are no buffer layers, use the first
              repeating layer.
        unit : ``'A'`` | ``'nm'``
            Display unit for the depth axis.
        ax : matplotlib.axes.Axes, optional
        figsize : (float, float)

        Returns
        -------
        fig, ax
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        if param not in ("a", "b", "c"):
            raise ValueError(f"param must be 'a', 'b', or 'c', got {param!r}")
        if unit not in ("A", "nm"):
            raise ValueError(f"unit must be 'A' or 'nm', got {unit!r}")

        self._update_offsets()
        scale = 0.1 if unit == "nm" else 1.0
        unit_label = "nm" if unit == "nm" else "Å"

        # Resolve reference value
        if reference is None:
            ref_layer = (self.buffer_layers or self.layers)[0]
            p_ref = getattr(ref_layer.crystal.lattice, param)
        elif isinstance(reference, (int, float)):
            p_ref = float(reference)
        else:
            # Assume xu.materials.Crystal-like object
            p_ref = float(getattr(reference.lattice, param))

        # Build full layer sequence (same logic as plot_lattice_parameter)
        segments = []
        z = 0.0
        for layer in self.buffer_layers:
            segments.append((z, z + layer.thickness, layer))
            z += layer.thickness
        for _ in range(self.n_rep):
            for layer in self.layers:
                segments.append((z, z + layer.thickness, layer))
                z += layer.thickness

        seen_labels = {}
        colors_cycle = plt.get_cmap("tab10")
        color_idx = 0

        standalone = ax is None
        if standalone:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        for z0, z1, layer in segments:
            p_val = getattr(layer.crystal.lattice, param)
            strain = (p_val - p_ref) / p_ref
            y0 = z0 * scale
            y1 = z1 * scale
            lbl = layer.label

            if lbl not in seen_labels:
                seen_labels[lbl] = colors_cycle(color_idx / 9)
                color_idx = (color_idx + 1) % 10
            col = seen_labels[lbl]

            ax.hlines(strain, y0, y1, colors=col, linewidths=2.5)

        # Vertical connectors between adjacent steps
        for i in range(1, len(segments)):
            _, _, layer_prev = segments[i - 1]
            z_curr, _, layer_curr = segments[i]
            s_prev = (getattr(layer_prev.crystal.lattice, param) - p_ref) / p_ref
            s_curr = (getattr(layer_curr.crystal.lattice, param) - p_ref) / p_ref
            ax.vlines(z_curr * scale, min(s_prev, s_curr), max(s_prev, s_curr),
                      colors="#555555", linewidths=0.8, linestyles="--")

        # Zero-strain reference line
        ax.axhline(0.0, color="#888888", linewidth=0.8, linestyle=":")

        handles = [
            mpatches.Patch(color=col, label=lbl)
            for lbl, col in seen_labels.items()
        ]
        ax.legend(handles=handles, fontsize=8, loc="best",
                  framealpha=0.4, facecolor="#1a1f2e", edgecolor="#3a3f4e",
                  labelcolor="#ccccee")

        ref_str = f"{p_ref:.4f} Å"
        ax.set_xlabel(f"depth  ({unit_label})", fontsize=9)
        ax.set_ylabel(f"ε_{param}  =  (p − p_ref) / p_ref", fontsize=9)
        ax.set_title(
            f"{self.name}  —  strain profile  [{param},  ref = {ref_str}]",
            fontsize=10,
        )
        ax.set_xlim(0, self.total_thickness * scale)

        if standalone:
            fig.tight_layout()

        return fig, ax
