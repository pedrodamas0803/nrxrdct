# Texture tomography theory

This page covers the theoretical foundations behind `nrxrdct.texture`: what
crystallographic texture and pole figures are, how a pole figure relates to
the orientation distribution function (ODF), the diffraction geometry used
to convert single-rotation-axis XRD-CT scan data into pole-figure
coordinates, the WIMV inversion method, and — importantly — the coverage and
uniqueness limitations that come with this specific instrument geometry.

---

## 1. Texture and the orientation distribution function

A polycrystalline sample is a collection of grains, each with its own
crystal orientation relative to the sample (lab) frame. **Texture** is any
statistical tendency for those orientations to cluster around preferred
values rather than being uniformly random.

An orientation is a proper rotation $g \in SO(3)$ mapping a crystal-frame
vector to the sample frame, $\mathbf{v}_\text{sample} = g\,\mathbf{v}_\text{crystal}$.
The **orientation distribution function** (ODF) $f(g)$ is the volume
fraction of material with orientation in $[g, g+dg]$, normalised over the
Haar (rotation-invariant) measure of $SO(3)$:

$$
\int_{SO(3)} f(g)\,dg = 1
$$

`nrxrdct.texture.odf_inversion` parametrises $g$ by Bunge-style Euler angles
$(\varphi_1, \Phi, \varphi_2)$:

$$
g(\varphi_1, \Phi, \varphi_2) = R_z(\varphi_1)\,R_x(\Phi)\,R_z(\varphi_2)
$$

!!! warning "Convention risk"
    This is *one* of several conventions in use across texture software
    (Bunge, Roe, Kocks differ; even "Bunge" implementations disagree on axis
    order and active/passive sense). Don't compare Euler angles quantitatively
    against another package without checking.

The volume element on $SO(3)$ in these coordinates carries a $\sin\Phi$
weight — `orientation_grid` returns this as `cell_weight`, used both to
normalise the fitted ODF and (if you resample or average `f`) to correctly
weight each grid cell.

---

## 2. Pole figures

A **pole figure** for a given lattice plane family $\{hkl\}$ plots, for
every direction $\hat{y}$ on the unit sphere (the "pole figure" itself), the
volume fraction of grains whose $hkl$-plane normal points along $\hat{y}$ in
the sample frame. It is a 2-D projection of the 3-D ODF: many different
orientations can share the same $hkl$-pole direction (they differ only by a
rotation about $\hat{y}$), so the pole figure is a **fiber-averaged**
(marginalised) view of $f(g)$:

$$
P_h(\hat{y}) = \frac{1}{2\pi}\oint f(g)\,d\varphi
\qquad\text{where } g \text{ ranges over all rotations mapping } \hat{h}\to\hat{y}
$$

A single pole figure under-determines $f(g)$ — this is why
[`compute_odf`](../api/texture/odf_inversion.md) needs pole figures from
**several independent `hkl` families** (`crystal_directions` in the API),
not just one.

---

## 3. Diffraction geometry: from (2θ, χ, ω) to pole-figure coordinates

`nrxrdct.texture.odf` extracts, for one `hkl` ring, the Debye-Scherrer ring
intensity as a function of the detector azimuthal angle χ
(`extract_ring_intensity`, via `cake_integration`) at every sample rotation
angle ω (the XRD-CT `rot` motor). `pole_figure_coordinates` converts each
$(2\theta, \chi, \omega)$ triple into a pole direction in the sample frame.

**Lab frame.** Incident beam along $+x$, rotation axis (vertical) along
$+z$. For a ring point at azimuth $\chi$ (measured from the vertical
detector axis in this derivation — see the calibration note below) and
Bragg angle $\theta = \tfrac12(2\theta)$, the diffracted beam is

$$
\hat{k}_f = (\cos 2\theta,\ \sin 2\theta \sin\chi,\ \sin 2\theta \cos\chi)
$$

with $\hat{k}_i = (1,0,0)$. The scattering vector $\mathbf{Q} = \hat{k}_f - \hat{k}_i$
bisects $-\hat{k}_i$ and $\hat{k}_f$; working out its normalised direction
gives, in the **lab frame**:

$$
\hat{Q}_\text{lab} = (-\sin\theta,\ \cos\theta\sin\chi,\ \cos\theta\cos\chi)
$$

Note the $x$-component depends only on $\theta$ (i.e. only on which `hkl`
ring), **not** on $\chi$ — every point on a given Debye-Scherrer ring shares
the same projection onto the beam axis. This is the reason a single
rotation axis gives *incomplete* pole-figure coverage (§5 below): a
sample-frame pole is only ever swept onto the ring at the specific rotation
angles $\omega$ where this $x$-projection constraint can be satisfied at
all, not at every $\omega$.

**Sample frame.** Rotating by $-\omega$ about $z$ (undoing the sample
rotation) gives the pole direction in the sample-fixed frame:

$$
\begin{aligned}
\hat{Q}_{x,\text{sample}} &= -\sin\theta\cos\omega + \cos\theta\sin\chi\sin\omega \\
\hat{Q}_{y,\text{sample}} &= \ \ \sin\theta\sin\omega + \cos\theta\sin\chi\cos\omega \\
\hat{Q}_{z,\text{sample}} &= \cos\theta\cos\chi
\end{aligned}
$$

converted to polar/azimuthal pole-figure coordinates:

$$
\alpha = \arccos\!\left(\hat{Q}_{z,\text{sample}}\right), \qquad
\beta = \operatorname{atan2}\!\left(\hat{Q}_{y,\text{sample}},\ \hat{Q}_{x,\text{sample}}\right)
$$

### Calibration: where is χ = 0?

This derivation assumes χ = 0 lies along the **vertical** (rotation-axis)
detector direction. Beamlines and pyFAI both differ in where χ = 0 actually
points and which way it increases — `pole_figure_coordinates` exposes
`chi_offset_deg` and `chi_sign` to align the assumed convention with reality.

As a concrete, empirically-verified data point: pyFAI's own azimuthal-angle
convention (`AzimuthalIntegrator.chiArray` / the `azimuthal` axis returned
by `cake_integration`) was checked directly and found to be
$\chi_\text{pyFAI} = \operatorname{atan2}(\text{row offset}, \text{col offset})$
relative to the beam-center pixel — i.e. **χ = 0 along the horizontal
(column) detector axis**, not vertical as this derivation assumes by
default. A `chi_offset_deg` around 90° (sign depending on your detector's
row/column handedness) is a more likely starting point than 0 for data
straight out of `cake_integration` — but detector rotation/flip settings
vary by beamline, so verify rather than assume.

---

## 4. Friedel's law and the "ghost" ambiguity

X-ray diffraction obeys **Friedel's law**: in the kinematic approximation
(negligible anomalous dispersion), the intensity at $\mathbf{Q}$ equals the
intensity at $-\mathbf{Q}$. For pole figures this means a real, physically
present companion reflection can appear at the antipodal point of any
measured pole.

It is tempting to exploit this by folding every measured pole direction
with $\alpha > 90°$ onto its antipode ($\alpha \to 180° - \alpha$,
$\beta \to \beta + 180°$) before fitting — useful for *displaying* a pole
figure on a single upper-hemisphere disk
(`nrxrdct.texture.texture_plotting.plot_pole_figure` does exactly this).

**This is not safe to do before fitting an ODF.** Folding every `hkl`'s pole
directions independently is mathematically equivalent to conflating
orientation $R$ with its point-inversion $-R$ (an improper rotation, i.e.
$R$ combined with spatial inversion) **simultaneously across every `hkl`** —
because $-R\hat{h} = -(R\hat{h})$ for *every* crystal direction $\hat{h}$ at
once, folding maps the true solution and this "ghost" solution onto
identical data for all `hkl` families together, not just one.

This is a specific instance of the well-known **ghost phenomenon** in
pole-figure-based texture analysis (Friedel's law makes certain harmonic
coefficients of $f(g)$ unrecoverable from X-ray pole figures alone). WIMV's
non-negativity constraint is known to suppress ghosts better than a literal
harmonic-series fit, but testing during development confirmed it does not
always break the tie — occasionally converging to the ~180°-misoriented
ghost solution instead of the true orientation, for both single- and
multi-`hkl` fits.

**Consequence:** `compute_odf` and `recalculate_pole_figure` default to
`fold_hemisphere=False`. Only enable folding if you specifically need
folded, display-style output and have independently verified it doesn't
corrupt your fit.

---

## 5. Coverage: why WIMV, not the harmonic method

A single vertical rotation axis with no tilt only ever sweeps a **curve** on
the pole sphere per `hkl` (§3: the beam-axis projection constraint pins
down $\theta$ but leaves only a 1-parameter family in $\chi,\omega$
jointly satisfying it) — not the near-complete hemisphere coverage that
classical texture-goniometry inversion methods (including the harmonic
series expansion) assume. Combining several `hkl` rings (different $\theta$,
hence different curves) can partially fill the sphere, and crystal symmetry
(passing multiple symmetry-equivalent directions per `hkl` family to
`compute_odf`'s `crystal_directions`) helps further, but achievable ODF
resolution is fundamentally limited by how much coverage your geometry
allows — there is no tilt stage in this pipeline to fill it in.

**WIMV** (Matthies & Vinel, 1982) is the standard choice under incomplete
coverage: an iterative, multiplicative correction scheme,

$$
f^{(n+1)}(g) = f^{(n)}(g) \cdot \exp\!\left[
\frac{\sum_{h,\hat{y}} K_h(\hat{y}, g)\,\ln\!\left(P_h^\text{meas}(\hat{y}) \,/\, P_h^{(n)}(\hat{y})\right)}
     {\sum_{h,\hat{y}} K_h(\hat{y}, g)}
\right]
$$

where $K_h(\hat{y}, g)$ is a hard nearest-cell indicator — 1 if $g\hat{h}$
(considering every symmetry-equivalent $\hat{h}$ at once) is the closest grid
orientation's implied pole to the measured direction $\hat{y}$, else 0 — the
original Matthies & Vinel binning, rather than a soft/Gaussian correspondence.
Angular resolution is therefore set purely by the orientation-grid spacing
(`step_deg`); there is no separate kernel-width parameter, and because every
data point contributes to exactly one grid cell, there's no wide-radius
neighbour search whose match count can blow up with grid density or symmetry
family size — this is what makes the implementation tractable at MTEX-like
speed without the accuracy cost a truncated Gaussian kernel would carry.
$P_h^{(n)}(\hat{y}) = \sum_g K_h(\hat{y}, g)\,f^{(n)}(g)$ is the pole figure
the current ODF estimate predicts (in practice just $f^{(n)}$ evaluated at
$\hat{y}$'s assigned cell). Grid cells with no nearby data (`weight_den == 0`
in the implementation) simply keep their previous value — coverage gaps
degrade gracefully rather than diverging, and because the update is purely
multiplicative from a non-negative start, $f \geq 0$ is guaranteed
throughout, unlike an unconstrained harmonic fit.

`recalculate_pole_figure` forward-projects a fitted ODF back through the
same kernel — always inspect measured vs. recalculated pole figures
(`plot_pole_figure_comparison`) before trusting a fit; nothing about the
`rp_history` convergence trace guarantees the *global* optimum was found
(§6 discusses a concrete failure mode).

---

## 6. Beyond bulk: per-voxel reconstruction (experimental, not shipped)

Everything above describes a **bulk** or **per-line** ODF: one fit
aggregating (or per translation line) the entire beam path through the
sample, exactly analogous to how a normal XRD-CT sinogram is a line
integral before tomographic reconstruction. Recovering a genuinely
**per-voxel** ODF — the texture-tomography analogue of reconstructing a 2-D
density map — turns out to be a substantially harder problem, explored but
**not implemented** in this package. The forward model couples a spatial
Radon operator with the WIMV kernel:

$$
I(h, \omega, \delta, \chi) = \sum_{v \in \text{ray}(\omega,\delta)} \text{pathlength}(v) \sum_g K_h(\hat{y}, g)\,f_v(g)
$$

(unknowns $f_v(g)$ per voxel $v$ *and* orientation cell $g$ — the two
operators aren't separable, because the kernel depends on $\omega$, exactly
the axis tomography integrates over). Prototyping during development found:

- A **discrete orientation-grid density per voxel** (the direct analogue of
  bulk WIMV, extended per-voxel) is underdetermined: the aggregate data fit
  improves monotonically over iterations, but per-voxel accuracy does not —
  voxels can drift to wrong solutions even as the global residual keeps
  improving, because many spatially-incoherent per-voxel assignments explain
  the same projected data equally well.
- A **single dominant orientation per voxel** (far fewer unknowns, fit via
  Gauss–Seidel coordinate descent) reliably recovers interior, single-grain
  voxels, but is structurally unreliable for voxels sitting exactly on a
  grain boundary — a genuine partial-volume effect (a ray through a boundary
  voxel also passes through neighbouring, differently-oriented voxels), not
  a fixable bug, confirmed independently by two different algorithms failing
  at the *same* physical locations.

Closing this out properly would need an explicit per-voxel orientation-
*mixture* model at boundaries — a substantially larger undertaking than
either prototype. Treat any future per-voxel texture-tomography feature in
this package as building on this open problem, not a solved one.

---

## 7. References

- H.-J. Bunge, *Texture Analysis in Materials Science*, Butterworths, 1982.
- S. Matthies, G. W. Vinel, "On the reproduction of the orientation
  distribution function of texturized samples from reduced pole figures
  using the conception of a conditional ghost correction", *Phys. Status
  Solidi B* **112**, K111 (1982) — the WIMV method.
- U. F. Kocks, C. N. Tomé, H.-R. Wenk, *Texture and Anisotropy*, Cambridge
  University Press, 1998 — general reference for pole figures, ODFs, and the
  ghost phenomenon.

## 8. See also

- [Texture Tomography Workflow](texture_tomography.md) — practical usage,
  pipeline stages, and API examples.
- [API reference](../api/texture/odf.md) for the full function docstrings.