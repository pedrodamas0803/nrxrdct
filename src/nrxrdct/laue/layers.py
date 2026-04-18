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
# EPITAXIAL STRAIN
# ─────────────────────────────────────────────────────────────────────────────


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
        stack.add_layer(InGaN, U_InGaN, n_cells=n_InGaN,
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
        and abs(lat.beta  - 90.0) < 0.5
        and abs(lat.gamma - 120.0) < 0.5
    )
    if _is_hexagonal:
        g_norm = growth_dir / np.linalg.norm(growth_dir)
        c_axis = np.array([0., 0., 1.])
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

    eps_par  = (a_sub - a_film) / a_film
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
    n_cells     : int   number of unit cells along the stacking direction
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

    def __init__(self, crystal, U, n_cells, n_hat=None,
                 d_spacing=None, label=None, absorption_limit=False):
        self.crystal = crystal
        self.U = np.asarray(U, dtype=float)
        self.n_cells = int(n_cells)
        self.label = label or crystal.name
        # When True, structure_factor uses an energy-dependent effective thickness
        # min(real thickness, 1/μ) to model Beer-Lambert absorption depth.
        # Set automatically for buffer layers by LayeredCrystal.add_buffer_layer.
        self.absorption_limit = bool(absorption_limit)

        if n_hat is None:
            self.n_hat = np.array([0., 0., 1.])
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
            for vec in lat._ai:          # rows: a1, a2, a3
                v_lab = self.U @ np.asarray(vec, dtype=float)
                p = abs(float(np.dot(v_lab, self.n_hat)))
                if p > 1e-6:
                    proj.append(p)
            self.d = min(proj) if proj else lat.c

    @property
    def thickness(self):
        """Total layer thickness in Å (always the real physical thickness)."""
        return self.n_cells * self.d

    def _effective_n_cells(self, energy_eV: float) -> int:
        """
        Effective number of unit cells after Beer-Lambert absorption limiting.

        The incident beam travels along the LT-frame x-axis ``[1,0,0]``.
        The sample-surface normal coincides with the stacking direction
        ``n_hat`` (the growth-crystal plane is always parallel to the wafer
        surface).  The angle α between beam and surface normal is therefore
        fixed by the sample mounting:

            cos α = |n̂ · x̂| = |n_hat[0]|

        The 1/e absorption depth *along the surface normal* is ``1/μ``.
        Projecting onto the stacking direction (same as the surface normal
        here) and accounting for the oblique path gives the effective number
        of unit cells:

            n_eff = |n_hat[0]| / (μ · d)

        For a beam at near-normal incidence (|n_hat[0]| → 1) this reduces to
        the usual ``1/(μ·d)``.  For a highly tilted surface (|n_hat[0]| → 0,
        grazing incidence) the beam spends a long path per unit depth and
        n_eff → 0 (the layer is fully opaque).

        Returns ``self.n_cells`` unchanged if the material lookup fails or if
        the absorption depth exceeds the real layer thickness.
        """
        _HC_ANG = 12398.419843   # hc in eV·Å

        try:
            # Prefer the crystal's own delta_beta if available
            if hasattr(self.crystal, "delta_beta"):
                _, beta = self.crystal.delta_beta(energy_eV)
            else:
                # Fall back to element density → Amorphous proxy
                elem = getattr(xu.materials.elements, self.crystal.name, None)
                if elem is None or not getattr(elem, "density", 0):
                    return self.n_cells
                mat = xu.materials.Amorphous(self.crystal.name, elem.density)
                _, beta = mat.delta_beta(energy_eV)

            if not (beta > 0):
                return self.n_cells

            lam_ang = _HC_ANG / energy_eV
            mu = 4.0 * np.pi * beta / lam_ang   # Å⁻¹
            if mu <= 0:
                return self.n_cells

            # cos of angle between beam (LT x-axis) and surface normal (n_hat)
            # Clamped to avoid zero (grazing incidence → n_eff = 1 minimum)
            cos_alpha = abs(float(self.n_hat[0]))
            cos_alpha = max(cos_alpha, 1e-3)

            n_eff = int(min(self.n_cells, cos_alpha / (mu * self.d)))
            return max(n_eff, 1)

        except Exception:
            return self.n_cells

    def structure_factor(self, Q_lab, energy_eV, z0=0.0):
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

        # Effective cell count: limited by absorption depth for buffer layers
        n_eff = self._effective_n_cells(energy_eV) if self.absorption_limit else self.n_cells

        phi_mod = phi % (2.0 * np.pi)
        if abs(phi_mod) < 1e-10 or abs(phi_mod - 2 * np.pi) < 1e-10:
            geo_sum = n_eff + 0j
        else:
            geo_sum = (1.0 - np.exp(1j * n_eff * phi)) / (1.0 - np.exp(1j * phi))

        phase_z0 = np.exp(1j * Qn * z0)
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
    >>> stack.add_layer(Fe, U_Fe, n_cells=20, label='Fe')
    >>> stack.add_layer(Cu, U_Cu, n_cells=20, label='Cu')
    >>> stack.set_repetitions(10)

    Example — using a U matrix from Laue indexation (GaN grown along c)
    --------------------------------------------------------------------
    >>> n_hat = U_GaN @ np.array([0., 0., 1.])   # growth dir in lab frame
    >>> stack = LayeredCrystal(name='GaN/InGaN', stacking_direction=n_hat)
    >>> stack.add_layer(GaN,   U_GaN,   n_cells=1000, label='GaN')
    >>> stack.add_layer(InGaN, U_InGaN, n_cells=50,   label='InGaN')

    """

    def __init__(self, name="layered_crystal", stacking_direction=None):
        self.name = name
        self.buffer_layers = []   # non-repeating layers (substrate, buffer) — bottom of stack
        self.layers = []          # repeating unit (MQW bilayer)
        self.n_rep = 1            # number of bilayer repetitions
        self._buffer_z_offsets = []
        self._z_offsets = []

        if stacking_direction is None:
            self.n_hat = np.array([0., 0., 1.])
        else:
            nh = np.asarray(stacking_direction, dtype=float)
            self.n_hat = nh / np.linalg.norm(nh)

    # ── Building the stack ────────────────────────────────────────────────────

    def add_buffer_layer(self, crystal, U, n_cells, d_spacing=None, label=None):
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
        n_cells   : int   number of unit cells along the stacking direction
        d_spacing : float, optional   stacking repeat distance (Å)
        label     : str, optional
        """
        layer = Layer(crystal, U, n_cells,
                      n_hat=self.n_hat, d_spacing=d_spacing, label=label,
                      absorption_limit=True)
        self.buffer_layers.append(layer)
        self._update_offsets()
        return self

    def add_layer(self, crystal, U, n_cells, d_spacing=None, label=None):
        """
        Append a layer to the **repeating** unit (MQW / bilayer).

        Layers are stacked in the order they are added; the first call
        places the layer at the bottom of the unit, the last at the top.
        The full unit is then repeated ``n_rep`` times above the buffer layers.

        Parameters
        ----------
        crystal   : xu.materials.Crystal
        U         : (3,3) orientation matrix
        n_cells   : int   number of unit cells along stacking direction
        d_spacing : float, optional   stacking repeat distance (Å)
        label     : str, optional
        """
        layer = Layer(crystal, U, n_cells,
                      n_hat=self.n_hat, d_spacing=d_spacing, label=label)
        self.layers.append(layer)
        self._update_offsets()
        return self

    def add_pseudomorphic_layer(
        self,
        crystal,
        U,
        n_cells,
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
        n_cells : int
            Number of unit cells along the stacking direction.
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
            crystal, a_substrate, C13, C33, growth_dir)
        lbl = label or crystal.name
        print(
            f"  {lbl}: ε_∥ = {eps_par:+.4f}  ε_⊥ = {eps_perp:+.4f}"
            f"  d_bulk → {d_strained / (1 + eps_perp):.4f} Å"
            f"  d_strained = {d_strained:.4f} Å"
        )
        return self.add_layer(crystal, U, n_cells, d_spacing=d_strained, label=lbl)

    def set_repetitions(self, n):
        """Set the number of times the repeating unit (MQW bilayer) is stacked."""
        self.n_rep = int(n)
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

    def structure_factor(self, Q_lab, energy_eV):
        """
        Total kinematical structure factor of the stack at Q_lab.

        The stack is divided into two sections:

        1. **Buffer layers** (non-repeating, at the bottom):

               F_buf(Q) = Σ_j  F_layer_j(Q, z0_j)

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

        Returns
        -------
        F : complex   total structure factor (electron units)
        """
        self._update_offsets()
        Q = np.asarray(Q_lab, dtype=float)
        Qn = float(np.dot(Q, self.n_hat))

        # ── Buffer layers (non-repeating) ─────────────────────────────────────
        F_total = 0.0 + 0j
        for layer, z0 in zip(self.buffer_layers, self._buffer_z_offsets):
            F_total += layer.structure_factor(Q, energy_eV, z0=z0)

        # ── Repeating unit (MQW) ──────────────────────────────────────────────
        if self.layers:
            F_unit = 0.0 + 0j
            for layer, z0 in zip(self.layers, self._z_offsets):
                F_unit += layer.structure_factor(Q, energy_eV, z0=z0)

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
            print(f"\n  Buffer layers  (non-repeating, {len(self.buffer_layers)} layer"
                  f"{'s' if len(self.buffer_layers) != 1 else ''}):")
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
