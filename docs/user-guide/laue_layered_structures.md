# Layered crystalline structures — model and theory

This page describes the physical model behind `LayeredCrystal` and its
associated utilities.  It covers the stacking geometry, the kinematical
structure factor for a multilayer stack, pseudomorphic epitaxial strain
mechanics (including the Poisson response, critical thickness, and how the
strained $d$-spacing is used in the simulation), and orientation relationships
between phases.

---

## 1. The stacking model

A `LayeredCrystal` represents an epitaxial thin-film stack divided into two
sections:

```
┌───────────────────────────────────────┐  ← surface (top)
│  repeating unit  (layer A + layer B)  │
│  ×  n_rep                             │
├───────────────────────────────────────┤
│  repeating unit  (layer A + layer B)  │
│  …                                    │
├───────────────────────────────────────┤
│  buffer layer  (e.g. template layer)  │
├───────────────────────────────────────┤
│  buffer layer  (e.g. substrate)       │
└───────────────────────────────────────┘  ← bottom (deepest)
```

| Section | Description |
|---|---|
| **Buffer layers** | Non-repeating layers at the bottom of the stack (substrate, thick template layers). Added deepest-first via `add_buffer_layer`. Absorption limiting is applied automatically. |
| **Repeating unit** | One bilayer period (e.g. QW + barrier). Added in bottom-to-top order via `add_layer` or `add_pseudomorphic_layer`. Repeated `n_rep` times above the buffer. |

The **stacking direction** $\hat{n}$ is a unit vector in the lab frame pointing
from the substrate toward the surface.  All structure-factor phase calculations
use the projection $Q_n = \mathbf{Q}\cdot\hat{n}$, so the model is correct for
any sample mounting angle.

---

## 2. Kinematical structure factor

### 2.1 Single-layer amplitude

In the kinematical (Born) approximation, the scattered amplitude from a single
crystalline layer is the coherent sum over all unit cells:

$$
F_\text{layer}(\mathbf{Q}) = F_\text{uc}(\mathbf{Q}_\text{cry})\;
\sum_{n=0}^{N-1} e^{\,i\mathbf{Q}\cdot\mathbf{R}_n}
$$

where:

- $F_\text{uc}(\mathbf{Q}_\text{cry})$ — the **unit-cell structure factor**, evaluated at the
  scattering vector expressed in the crystal frame
  $\mathbf{Q}_\text{cry} = U^T \mathbf{Q}$ (see [Section 6](#6-coordinate-frames-and-orientation-matrix));
- $\mathbf{R}_n = (z_0 + n\,d)\,\hat{n}$ — the position of the $n$-th unit cell
  along the stacking direction, with $z_0$ the layer's depth offset and $d$ the
  stacking repeat distance;
- $N$ — the number of unit cells in the layer; $Nd = t$ is the physical thickness.

Because the positions are collinear along $\hat{n}$, the sum depends only on
$Q_n = \mathbf{Q}\cdot\hat{n}$:

$$
\sum_{n=0}^{N-1} e^{\,i n Q_n d}
= \begin{cases}
  N & Q_n d \equiv 0 \pmod{2\pi} \\[4pt]
  \dfrac{1 - e^{\,i N Q_n d}}{1 - e^{\,i Q_n d}} & \text{otherwise}
\end{cases}
$$

The squared modulus of this sum is the **Laue interference function** (see the
[Thin-Film Satellites](laue_thin_film_satellites.md) page for its fringe structure).

### 2.2 Full stack structure factor

For a stack of buffer layers plus $N_\text{rep}$ repetitions of a bilayer unit,
the total amplitude is

$$
\boxed{
F_\text{total}(\mathbf{Q})
= \underbrace{
    \sum_{j \in \text{buf}} F_j(\mathbf{Q},\, z_{0,j})
  }_{F_\text{buf}}
+ \underbrace{
    e^{\,i Q_n z_\text{buf}}\;
    F_\text{unit}(\mathbf{Q})\;
    S_\text{rep}(Q_n \Lambda)
  }_{F_\text{MQW}}
}
$$

where:

- $z_\text{buf}$ — total thickness of the buffer section (phase shift placing
  the MQW above the buffer);
- $F_\text{unit}(\mathbf{Q}) = \sum_{j \in \text{unit}} F_j(\mathbf{Q},\, z_{0,j}^\text{rel})$ —
  the structure factor of one bilayer period (phases relative to the period
  bottom);
- $\Lambda = \sum_j t_j$ — the bilayer period thickness;
- $S_\text{rep}$ — the **superlattice geometric factor**:

$$
S_\text{rep}(Q_n \Lambda) = \sum_{m=0}^{N_\text{rep}-1} e^{\,i m Q_n \Lambda}
= \begin{cases}
  N_\text{rep} & Q_n \Lambda \equiv 0 \pmod{2\pi} \\[4pt]
  \dfrac{1 - e^{\,i N_\text{rep} Q_n \Lambda}}{1 - e^{\,i Q_n \Lambda}}
  & \text{otherwise}
\end{cases}
$$

The modulus of $S_\text{rep}$ peaks sharply at the **superlattice Bragg
conditions** $Q_n \Lambda = 2\pi m$ ($m$ integer), where
$|S_\text{rep}|^2 = N_\text{rep}^2$.  Between these peaks it creates a pattern
of satellite fringes with spacing $\Delta Q_n = 2\pi/\Lambda$.

### 2.3 Intensity

The observable intensity at a Laue spot is

$$
I = |F_\text{total}(\mathbf{G})|^2 \times LP(2\theta) \times S(E)
$$

where $LP$ is the Lorentz–polarisation factor and $S(E)$ is the synchrotron
source spectrum at the Bragg energy $E_{hkl}$ (see the
[Theory](laue_theory.md) page for both).

---

## 3. Structure model

The `structure_model` parameter accepted by `simulate_laue_stack`,
`simulate_laue_darwin`, and `simulate_mixed_phases` controls **two things at
once**: which crystals are enumerated to generate candidate Bragg reflections,
and how the structure factor amplitude is computed at each of those
reflections.  The default is `'average'`.

---

### 3.1 Coherent model

`structure_model='coherent'` reproduces the physical kinematical sum exactly.

**G-vector enumeration.** Every unique crystal in the stack (buffer layers and
all MQW layers) contributes its own set of reciprocal-lattice vectors.  For an
InGaN/GaN MQW this means both the GaN and InGaN sub-lattices are enumerated,
producing *two slightly displaced sets of Bragg peaks* — one for GaN
($d_\text{GaN}$) and one for the strained InGaN ($d_\text{InGaN, strained}$).

**Structure factor.** The full coherent sum over all layers is evaluated at
each candidate $\mathbf{G}$:

$$
F_\text{total}(\mathbf{Q})
= F_\text{buf}(\mathbf{Q})
+ e^{\,i Q_n z_\text{buf}}\;F_\text{unit}(\mathbf{Q})\;S_\text{rep}(Q_n\Lambda)
$$

where every layer contributes with its exact depth phase $e^{\,i Q_n z_j}$
(see [Section 2.2](#22-full-stack-structure-factor)).  This preserves the full
inter-layer interference pattern: superlattice satellites and thickness fringes
appear with their correct relative intensities.

**When to use.** Coherent mode is appropriate when you need physically accurate
fringe intensities — for example to fit satellite sidelobes and infer
individual layer thicknesses — or when the layer contrast is large enough that
separate GaN / InGaN Bragg peaks are meaningful.

---

### 3.2 Average model (default)

`structure_model='average'` treats the MQW as a single effective material,
matching the appearance of the pattern as seen in a **monochromatic
rocking-curve** scan.

**G-vector enumeration.** Only the buffer layers are enumerated (or the first
MQW layer if no buffer layers exist).  This produces a *single set of Bragg
positions* — the average-lattice positions — with no separate InGaN peak.

**Structure factor.** The repeating unit is replaced by a
*composition-weighted average over one bilayer period*:

$$
\boxed{
F_\text{unit}^\text{avg}(\mathbf{Q})
= \sum_{j \in \text{unit}} F_{\text{uc},j}(\mathbf{Q})\; N_{\text{eff},j}
}
$$

where the intra-period depth phases $e^{\,i Q_n z_j^\text{rel}}$ are omitted.
The inter-period geometric series $S_\text{rep}$ is **retained**:

$$
F_\text{total}^\text{avg}(\mathbf{Q})
= F_\text{buf}(\mathbf{Q})
+ e^{\,i Q_n z_\text{buf}}\;
  F_\text{unit}^\text{avg}(\mathbf{Q})\;
  S_\text{rep}(Q_n\Lambda)
$$

Buffer layer phase offsets $e^{\,i Q_n z_{0,j}}$ are always preserved; only
the *intra-period* phases are averaged out.

**Physical interpretation.** $S_\text{rep}$ still peaks sharply at
$Q_n\Lambda = 2\pi m$, so satellite spots appear at exactly the same detector
positions as in the coherent model.  What changes is the amplitude at each
satellite: instead of depending on the layer *ordering* within the period
(which creates strong intensity asymmetry between $+m$ and $-m$ satellites),
every satellite order carries an amplitude proportional to
$F_\text{unit}^\text{avg}$ evaluated at that $\mathbf{Q}$.  The result is the
*structural envelope* — the maximum intensity each satellite order could carry
if all unit cells in the period scattered in phase.

**Effect of strain.** The strained $d$-spacing from
`add_pseudomorphic_layer` enters through
$N_\text{eff} = t / d_\text{strained}$ in $F_\text{unit}^\text{avg}$, so the
composition weighting correctly accounts for how many unit cells each strained
layer contributes.  The unit-cell structure factor amplitude $F_{\text{uc},j}$
is still evaluated at the bulk crystal positions in both models (the
tetragonal distortion changes the inter-plane spacing, not the in-plane atom
arrangement, as explained in [Section 4.3](#44-how-the-simulation-represents-a-pseudomorphic-layer)).

**When to use.** Average mode is the right default for most Laue measurements.
It gives a pattern that looks like a monochromatic scan: one average Bragg
peak per reflection family with satellites symmetrically distributed around
it, without artefacts from the doubled enumeration of strained and unstrained
sub-lattices.

---

### 3.3 Comparison

| | `'coherent'` | `'average'` (default) |
|---|---|---|
| G-vector sources | all layers | buffer layers only |
| Separate InGaN / GaN peaks | yes | no |
| Satellite positions | correct | identical |
| Satellite intensities | full interference, layer-ordering effects | structural envelope, composition-weighted |
| Intra-period depth phases | yes | no |
| $S_\text{rep}$ inter-period sum | yes | yes |
| Buffer layer depth phases | yes | yes |
| Absorption corrections | same | same |

---

## 4. Pseudomorphic layers — physics and simulation model

### 4.1 What "pseudomorphic" means

A layer is **pseudomorphic** (also called *coherently strained* or *fully strained*)
when every atomic plane in the film is registry-matched to the substrate: the
in-plane atom spacings are identical to those of the template below, regardless
of the film's natural (bulk, relaxed) lattice parameter.

```
Substrate (GaN):   |  a_sub  |  a_sub  |  a_sub  |  a_sub  |
                   ──────────────────────────────────────────
Film (InGaN bulk): | a_film  | a_film  | a_film  |            ← relaxed
                   ──────────────────────────────────────────
Film (strained):   |  a_sub  |  a_sub  |  a_sub  |  a_sub  |  ← pseudomorphic
```

The lateral constraint compresses or stretches the film's in-plane bond lengths.
To conserve volume (to first order), the film responds by distorting its
out-of-plane lattice parameter in the **opposite sense** — the so-called
*Poisson response*.  For InGaN on GaN ($a_\text{film} > a_\text{sub}$), the
film is compressed in-plane and the $c$-axis **expands** beyond the bulk value.

This tetragonal distortion persists as long as the layer is thinner than the
**critical thickness** $h_c$ (see [Section 3.6](#46-critical-thickness)).
Above $h_c$, misfit dislocations nucleate and partially relax the strain.

### 4.2 Biaxial strain state

The in-plane mismatch strain is

$$
\varepsilon_\parallel
= \frac{a_\text{sub} - a_\text{film}}{a_\text{film}}
$$

| Sign | Meaning | Example |
|---|---|---|
| $\varepsilon_\parallel < 0$ | film compressed in-plane | InGaN on GaN ($a_\text{InGaN} > a_\text{GaN}$) |
| $\varepsilon_\parallel > 0$ | film stretched in-plane | AlGaN on GaN ($a_\text{AlGaN} < a_\text{GaN}$) |

Because the strain is **biaxial** (equal in both in-plane directions for a
hexagonal or cubic layer on a (001)/(0001) substrate), the stress tensor has
the form $\sigma_{xx} = \sigma_{yy} = \sigma$, $\sigma_{zz} = 0$ (free surface).
Inverting Hooke's law under this constraint yields the out-of-plane strain.

### 4.3 Out-of-plane Poisson response

For a **hexagonal (wurtzite) crystal grown along its $c$-axis** $[0001]$ with a
free surface, the biaxial stress–strain relation in the Voigt notation reduces to

$$
\boxed{
\varepsilon_\perp = -\frac{2C_{13}}{C_{33}}\,\varepsilon_\parallel
}
$$

where $C_{13}$ and $C_{33}$ are elastic stiffness constants (GPa).

The negative sign means an in-plane compression ($\varepsilon_\parallel < 0$)
gives an out-of-plane expansion ($\varepsilon_\perp > 0$).

For a **cubic crystal grown along [001]**, the same formula holds with
$C_{12}$ in place of $C_{13}$ and $C_{11}$ in place of $C_{33}$.

### 4.4 How the simulation represents a pseudomorphic layer

`add_pseudomorphic_layer` internally calls `pseudomorphic_d_spacing`, which:

1. Reads the **bulk** out-of-plane repeat $d_\text{bulk}$ by projecting the
   direct-lattice basis vectors of the relaxed film crystal onto the growth
   direction.
2. Computes $\varepsilon_\parallel$ from the lattice mismatch.
3. Applies the Poisson formula to get $\varepsilon_\perp$.
4. Returns the **strained** repeat distance:

$$
d_\text{strained} = d_\text{bulk}\,(1 + \varepsilon_\perp)
$$

This value is passed as the `d_spacing` argument to `add_layer`.  The unit-cell
structure factor $F_\text{uc}$ is still evaluated using the **original
(relaxed) crystal** object and its atomic positions — only the **inter-plane
spacing** is changed.  This is the correct kinematical treatment: the
tetragonal distortion shifts the atom positions along $\hat{n}$ by a uniform
scale factor, but does not change the scattering power of any atom.

```
                   ↑ n̂ (stacking direction)
                   │
          ─── d_strained ───     ←  phase φ = Q_n · d_strained
          ─── d_strained ───
          ─── d_strained ───     N = round(thickness / d_strained) planes
          ─── d_strained ───
```

The number of planes is $N = \text{round}(t / d_\text{strained})$ where $t$ is
the requested physical thickness.

### 4.5 Effect on the diffraction pattern

The strained $d$-spacing shifts the Bragg condition for the film peak relative
to the substrate.  For a reflection along $\hat{n}$ (e.g. GaN / InGaN $0002$),
the difference in the reciprocal-lattice vector magnitude is

$$
\Delta Q_n = \frac{2\pi}{d_\text{strained}} - \frac{2\pi}{d_\text{sub}}
= \frac{2\pi}{d_\text{sub}}\,\frac{-\varepsilon_\perp}{1 + \varepsilon_\perp}
\approx -\frac{2\pi\,\varepsilon_\perp}{d_\text{sub}}
$$

For InGaN on GaN with $\varepsilon_\perp \approx +0.01$ (1% $c$-axis expansion),
the InGaN $0002$ peak is shifted to **smaller** $Q_n$ (larger $d$, longer
wavelength in Laue) relative to the GaN substrate peak.

In the **white-beam Laue geometry** this peak shift manifests as a slightly
different photon energy selected for the InGaN $0002$ reflection compared to
GaN $0002$.  Both reflections appear at **the same pixel** on the detector
(same $2\theta$, $\chi$), but the colour (energy) differs.  The Laue pattern
therefore does not directly resolve the peak splitting — for that, a
monochromatic rocking curve is needed.

However, in a **coherent superlattice** the interference between the strained
QW layers and the unstrained barriers produces **satellite peaks** at positions
offset from the substrate Bragg peak by $\Delta Q_n = 2\pi m / \Lambda$
(the superlattice periodicity).  The satellite positions encode $\Lambda$
directly, and the satellite intensities carry information about the QW
thickness, composition, and strain state.  These satellites appear at different
pixels on the Laue detector because their $Q_n$ differs from the substrate
Bragg condition — they satisfy the Laue condition at shifted photon energies.

### 4.6 Critical thickness

A pseudomorphic layer can only exist below the **critical thickness** $h_c$,
beyond which it becomes energetically favourable to nucleate misfit dislocations
and partially relax the strain.
The Matthews–Blakeslee equilibrium critical thickness for a single layer is
approximately

$$
h_c \approx \frac{b(1 - \nu\cos^2\alpha)}{8\pi\,|\varepsilon_\parallel|(1+\nu)\cos\lambda}
\left[\ln\!\left(\frac{h_c}{b}\right) + 1\right]
$$

where $b$ is the Burgers vector length, $\nu$ is the Poisson ratio, and
$\alpha$, $\lambda$ are angles between the dislocation line, Burgers vector,
and slip plane.  For practical purposes with III-nitrides:

| System | $\lvert\varepsilon_\parallel\rvert$ (%) | Typical $h_c$ |
|---|---|---|
| In$_{0.10}$GaN / GaN | 0.57 | ~10–15 nm |
| In$_{0.20}$GaN / GaN | 1.15 | ~3–5 nm |
| Al$_{0.20}$GaN / GaN | 0.51 | ~15–25 nm |

The simulation assumes perfect pseudomorphic growth.  If a layer is thicker
than $h_c$, part of the strain is relaxed and the effective $\varepsilon_\parallel$
is reduced.  In that case `pseudomorphic_d_spacing` should be called with a
corrected $a_\text{sub}$ reflecting the partially relaxed in-plane parameter.

### 4.7 Elastic constants for III-nitrides

The stiffness constants $C_{ij}$ (GPa) used by `nitride_elastic_constants`:

| Material | $C_{11}$ | $C_{12}$ | $C_{13}$ | $C_{33}$ | $C_{44}$ | $2C_{13}/C_{33}$ |
|---|---|---|---|---|---|---|
| GaN  | 390 | 145 | 106 | 398 | 105 | 0.533 |
| InN  | 223 | 115 |  92 | 224 |  48 | 0.821 |
| AlN  | 396 | 137 | 108 | 373 | 116 | 0.579 |

Sources: Wright (1997) *Phys. Rev. B* **55**, 6250 and
Vurgaftman & Meyer (2003) *J. Appl. Phys.* **94**, 3675.

The ratio $2C_{13}/C_{33}$ is the **biaxial Poisson ratio**: it quantifies how
strongly the $c$-axis responds to in-plane strain.  InN has the largest ratio
(~0.82), meaning InGaN alloys at high indium content develop a larger $c$-axis
expansion per unit mismatch strain than GaN or AlN.

For ternary alloys (Vegard's law):

$$
C_{ij}^\text{alloy}(x) = x\,C_{ij}^\text{InN} + (1-x)\,C_{ij}^\text{GaN}
$$

### 4.8 Validity limits

The scalar biaxial formula is valid for:

- Hexagonal wurtzite grown along $[0001]$ ($c$-axis).
- Cubic crystals grown along $[001]$ (use $C_{12}/C_{11}$).

It is **not valid** for semipolar hexagonal orientations (e.g. $[10\bar{1}3]$,
$[11\bar{2}2]$) or off-axis cubic growth, where the in-plane strain is
anisotropic and requires rotating the full stiffness tensor into the growth
frame.  `pseudomorphic_d_spacing` raises `ValueError` for these cases.

### 4.9 Worked example — In$_{0.20}$Ga$_{0.80}$N on GaN

```python
import xrayutilities as xu
import nrxrdct.laue as laue

GaN   = xu.materials.GaN          # a = 3.189 Å, c = 5.186 Å
InGaN = xu.materials.InGaN(0.20)  # a ≈ 3.260 Å, c ≈ 5.342 Å (Vegard)

C = laue.nitride_elastic_constants('InN', x=0.20, end_material='GaN')
# C13 ≈ 95.2 GPa, C33 ≈ 253.6 GPa

d_strained, eps_par, eps_perp = laue.pseudomorphic_d_spacing(
    InGaN,
    a_substrate = GaN.lattice.a,     # 3.189 Å — in-plane constraint
    C13 = C['C13'],
    C33 = C['C33'],
)
# eps_par  ≈ −0.0220   (2.2 % compressive in-plane)
# eps_perp ≈ +0.0164   (1.64 % tensile out-of-plane, c expands)
# d_strained ≈ 2.714 Å  vs. d_bulk = 2.671 Å
```

The strained repeat $d_\text{strained} \approx 2.714$ Å is used in the
superlattice phase factor instead of the bulk value $d_\text{bulk} \approx
2.671$ Å.  The GaN barrier uses the unmodified $d_\text{GaN} = 2.593$ Å.
The coherent interference between these two values of $d$ is what generates
the superlattice satellite peaks in the Laue pattern.

---

## 5. Stacking repeat distance

The stacking repeat distance $d$ is the periodicity of the lattice along $\hat{n}$.
It is used for:

- Computing $N = t/d$ (the number of unit cells from the physical thickness);
- The phase increment $\phi = Q_n d$ in the geometric sum.

`d_spacing_hkl(crystal, h, k, l)` computes the interplanar spacing of the
$(hkl)$ family using the reciprocal lattice directly:

$$
d_{hkl} = \frac{2\pi}{|\mathbf{G}_{hkl}|}
= \frac{2\pi}{|h\,\mathbf{b}_1 + k\,\mathbf{b}_2 + l\,\mathbf{b}_3|}
$$

This is valid for any crystal system.  For $c$-axis growth the relevant
repeat is the $d_{0002}$ spacing ($= c/2$ for a 2-atom hexagonal primitive
cell, or $c$ if the structure factor of 0001 is non-zero).

If `d_spacing` is not supplied to `add_layer`, the `Layer` class finds $d$
automatically by projecting each direct-lattice basis vector onto $\hat{n}$
and taking the shortest non-zero projection.

---

## 6. Coordinate frames and orientation matrix

### 6.1 Lab frame

The `nrxrdct.laue` lab frame (LaueTools LT frame) has:

| Axis | Direction |
|---|---|
| $x$ | along the incident beam |
| $z$ | vertical up |
| $y$ | $y = z \times x$ (horizontal) |

### 6.2 Crystal frame and $U$ matrix

The orientation matrix $U$ is a $3\times3$ rotation that maps crystal-frame
vectors to the lab frame:

$$
\mathbf{G}_\text{lab} = U\,\mathbf{G}_\text{crystal}
\qquad
\mathbf{Q}_\text{crystal} = U^T\,\mathbf{Q}_\text{lab}
$$

$U$ can be obtained in two ways:

1. **`euler_to_U(phi1, Phi, phi2, sample_tilt_deg)`** — from Bunge ZXZ Euler
   angles describing the crystal orientation relative to the sample surface,
   plus the sample tilt on the diffractometer.
2. **`U_from_matstarlab(matstarlab)`** — from a LaueTools refined `matstarlab`
   9-element array (already in the lab frame).

For a stack where all layers share the same crystallographic orientation (e.g.
epitaxial GaN / InGaN both grown along $c$), all layers share the **same** $U$.
The strained $d$-spacing (not the $U$ matrix) encodes the tetragonal distortion.

### 6.3 Stacking direction

The stacking direction in the lab frame is

$$
\hat{n}_\text{lab} = U\,\hat{n}_\text{crystal} / |U\,\hat{n}_\text{crystal}|
$$

where $\hat{n}_\text{crystal}$ is the growth direction in the crystal frame
(e.g. $[001]$ for $c$-axis wurtzite).  Pass this vector as
`stacking_direction` when constructing a `LayeredCrystal` from a Laue-indexed
$U$.

---

## 7. Absorption corrections — two-beam path and overlying layers

Photoelectric absorption attenuates the X-ray amplitude along **both** the
incident and exit paths through the sample.  The simulation applies two related
but distinct corrections.

### 7.1 Two-beam effective depth (within a layer)

The classical one-beam estimate limits the effective depth of a thick buffer
layer using the incident path only:

$$
N_\text{abs}^\text{1-beam} = \frac{\cos\alpha_\text{in}}{\mu\,d},
\qquad \cos\alpha_\text{in} = |\hat{n}\cdot\hat{x}|
$$

This underestimates the total absorption because the diffracted photon also
travels obliquely back through the same layer on its way to the detector.  The
**two-beam** correction accounts for both legs of the path:

$$
\boxed{
N_\text{abs} = \frac{\cos\alpha_\text{in}\,\cos\alpha_\text{out}}
                     {\mu\,d\;(\cos\alpha_\text{in} + \cos\alpha_\text{out})}
}
$$

where

$$
\cos\alpha_\text{out} = |\hat{n}\cdot\hat{k}_f|
$$

Here $\cos\alpha_\text{out}$ is the cosine of the angle between the **diffracted** beam direction
$\hat{k}_f$ and the surface normal.  This is the standard *symmetric
absorption correction* used in surface-diffraction analysis.

The one-beam and two-beam limits:

| Geometry | $\alpha_\text{out}$ | Effect |
|---|---|---|
| Near-normal exit ($\cos\alpha_\text{out} \approx 1$) | small | $\approx$ one-beam result |
| Grazing exit ($\cos\alpha_\text{out} \to 0$) | large | strongly reduced $N_\text{abs}$, layer appears very thin |
| Symmetric ($\alpha_\text{in} = \alpha_\text{out}$) | equal | $N_\text{abs} = \cos\alpha / (2\mu d)$ — exactly half the one-beam value |

Because $\hat{k}_f$ is spot-specific (it depends on the particular $(hkl)$
reflection being computed), the two-beam correction is applied **per spot**
inside `simulate_laue_stack` and `simulate_laue_darwin`.  It activates
automatically whenever these functions are called; there is no user-visible
parameter to set.

### 7.2 Overlying-layer attenuation

A photon must also pass through every layer **above** the layer of interest
twice: once on the way in and once on the way out.  For a layer at depth
$z_i$ below the surface, all shallower layers (buffer layers $j > i$ and the
entire MQW block) contribute a multiplicative transmission factor

$$
T_\text{above}^{(i)} = \prod_{j > i} T_j \;\times\; T_\text{MQW}
$$

$$
T_j = \exp\!\left[
  -\mu_j\, t_j \left(\frac{1}{\cos\alpha_\text{in}} + \frac{1}{\cos\alpha_\text{out}}\right)
\right]
$$

This amplitude factor is applied coherently — the structure-factor sum becomes

$$
F_\text{buf}(\mathbf{Q}) = \sum_i T_\text{above}^{(i)}\; F_i(\mathbf{Q},\, z_{0,i})
$$

Deeper buffer layers (substrate) are therefore dimmer than shallower ones (template
layers, thin interlayers) independently of their own extinction.

The MQW block sits at the top of the stack and has no overlying buffer
material, so no $T_\text{above}$ factor is applied to the repeating-unit
amplitude.

### 7.3 Practical impact

For typical III-nitride heterostructures at BM32 energies (10–22 keV) the
corrections have the following effect:

| Layer | Dominant correction |
|---|---|
| GaN/sapphire substrate (0.5 mm) | Self-absorption (one-beam → two-beam halves $N_\text{abs}$) |
| GaN buffer (2 μm) | Overlying-layer $T_\text{above}$ from MQW (~30 nm) — negligible at 15 keV |
| InGaN QW (3 nm) | Essentially transparent; corrections $< 0.1\%$ |

For a grazing-exit geometry (e.g. high-angle Bragg reflection with small $\alpha_\text{out}$),
the two-beam correction can reduce the substrate amplitude by an order of
magnitude relative to the one-beam estimate and is the dominant effect.

`add_buffer_layer` sets `absorption_limit=True` on the layer; `add_layer` does
not.  For repeating MQW layers the individual thicknesses are always much
smaller than $t_\text{abs}$, so no depth-limiting cap is needed (the
$T_\text{above}$ overlying-layer factor still applies).

For the Darwin model (`simulate_laue_darwin`), an additional **primary
extinction** correction is applied on top of absorption limiting — see the
[Darwin Model](laue_darwin.md) page.

---

## 8. Orientation relationships between phases

When two crystalline phases are epitaxially related, their orientation matrices
satisfy

$$
U_B = R_\text{OR}\, U_A
$$

where $R_\text{OR}$ is the **orientation relationship rotation** between the
two crystal frames.  $R_\text{OR}$ is fully specified by two direction pairs:

$$
R_\text{OR}\,\mathbf{v}_{1,A} \parallel \mathbf{v}_{1,B}
\quad\text{(primary — exact)},
\qquad
R_\text{OR}\,\mathbf{v}_{2,A} \text{ as close as possible to } \mathbf{v}_{2,B}
\quad\text{(secondary — minimised angle)}
$$

`or_from_directions(crystal_A, dir1_A, dir2_A, crystal_B, dir1_B, dir2_B)`
implements this for any pair of Miller directions.

### 8.1 Standard ORs implemented

| Function | Name | Primary constraint | Secondary constraint |
|---|---|---|---|
| `or_kurdjumov_sachs` | Kurdjumov–Sachs (KS) | $\{110\}_\text{BCC} \parallel \{111\}_\text{FCC}$ | $\langle 111\rangle_\text{BCC} \parallel \langle 110\rangle_\text{FCC}$ |
| `or_nishiyama_wassermann` | Nishiyama–Wassermann (NW) | $\{110\}_\text{BCC} \parallel \{111\}_\text{FCC}$ | $\langle 100\rangle_\text{BCC} \parallel \langle 011\rangle_\text{FCC}$ |
| `or_baker_nutting` | Baker–Nutting (BN) | $\{100\}_\text{BCC} \parallel \{100\}_\text{RS}$ | $\langle 110\rangle_\text{BCC} \parallel \langle 010\rangle_\text{RS}$ |
| `or_pitsch` | Pitsch | $\{100\}_\text{BCC} \parallel \{110\}_\text{FCC}$ | $\langle 011\rangle_\text{BCC} \parallel \langle 111\rangle_\text{FCC}$ |

The KS and NW relationships both satisfy $\{110\}_\text{BCC} \parallel \{111\}_\text{FCC}$
but differ in the secondary direction.  They are related by a small $5.26°$
rotation about the common plane normal and frequently coexist in the same
martensitic microstructure.

### 8.2 Computation

Internally, `_or_from_two_pairs` builds orthonormal frames from each pair of
directions and computes the rotation $R$ that maps one frame to the other:

$$
R_\text{OR} = F_B\,F_A^T
$$

where $F_A$, $F_B$ are $3\times3$ matrices whose columns are the Gram-Schmidt
orthonormalised direction pairs.  This is exact for the primary direction and
least-squares optimal for the secondary.

---

## 9. Alloy crystals — Vegard's law

For solid-solution alloys (e.g. In$_x$Ga$_{1-x}$N), lattice parameters and
elastic constants vary linearly with composition:

$$
a_\text{alloy}(x) = x\,a_\text{InN} + (1-x)\,a_\text{GaN}
$$

$$
C_{ij}^\text{alloy}(x) = x\,C_{ij}^\text{InN} + (1-x)\,C_{ij}^\text{GaN}
$$

This is the **Vegard's law** approximation, which is accurate to within a few
percent for most III-nitride alloys.  Larger deviations (bowing) occur near
the miscibility gap; for high In-content alloys a bowing parameter
$b = C_{ij}^\text{A} + C_{ij}^\text{B} - 2C_{ij}^\text{AB}$ can be added
manually if needed.

The alloy crystal for the simulation is constructed using xrayutilities.  Its
`StructureFactor` method uses the virtual-crystal approximation (VCA), weighting
each sublattice site by the occupancy fractions.  The composition enters the
structure factor through the atomic form factors, so even weak composition
modulations affect the diffracted intensities via anomalous scattering near
absorption edges.

---

## 10. Quick reference — stack construction

```python
import xrayutilities as xu
import nrxrdct.laue as laue

GaN   = xu.materials.GaN
InGaN = xu.materials.InGaN(0.20)   # In₀.₂₀Ga₀.₈₀N

U = laue.euler_to_U(0, 0, 0, sample_tilt_deg=40)

# Elastic constants (Vegard interpolation)
C = laue.nitride_elastic_constants('InN', x=0.20, end_material='GaN')

# d-spacings
d_GaN   = laue.d_spacing_hkl(GaN,   0, 0, 2)   # = c_GaN / 2 ≈ 2.593 Å
d_InGaN = laue.d_spacing_hkl(InGaN, 0, 0, 2)   # relaxed (bulk)

# Build stack: GaN buffer + 10× (InGaN QW + GaN barrier)
stack = laue.LayeredCrystal(n_hat=(0, 0, 1), n_rep=10)

stack.add_buffer_layer(GaN, U, thickness=5000.0,     # 500 nm
                       d_spacing=d_GaN, label='GaN buffer')

stack.add_pseudomorphic_layer(                        # computes d_strained
    InGaN, U, thickness=30.0,                         # 3 nm QW
    a_substrate=GaN.lattice.a,
    C13=C['C13'], C33=C['C33'], label='InGaN QW')

stack.add_layer(GaN, U, thickness=100.0,              # 10 nm barrier
                d_spacing=d_GaN, label='GaN barrier')

stack.describe()

# Visualise
stack.plot_lattice_parameter('c', unit='nm')
stack.plot_strain_profile('c', reference=GaN)
```

---

## References

1. **Matthews, J. W. & Blakeslee, A. E.** Defects in epitaxial multilayers. *J. Cryst. Growth* **27**, 118–125 (1974). *(Critical thickness for pseudomorphic growth.)*
2. **Wright, A. F.** Elastic properties of zinc-blende and wurtzite AlN, GaN, and InN. *Phys. Rev. B* **55**, 6250–6258 (1997).
3. **Vurgaftman, I. & Meyer, J. R.** Band parameters for nitrogen-containing semiconductors. *J. Appl. Phys.* **94**, 3675–3696 (2003). *(Elastic constants and Vegard's law for III-nitrides.)*
4. **Kurdjumov, G. & Sachs, G.** Über den Mechanismus der Stahlhärtung. *Z. Phys.* **64**, 325–343 (1930). *(Original KS orientation relationship.)*
5. **Nishiyama, Z.** *Martensitic Transformation*. Academic Press, New York, 1978. *(NW relationship and transformation crystallography.)*
6. **Authier, A.** *Dynamical Theory of X-Ray Diffraction*. Oxford University Press, 2001. *(Chapter 2 — kinematical limit and the structure factor of a finite crystal.)*
