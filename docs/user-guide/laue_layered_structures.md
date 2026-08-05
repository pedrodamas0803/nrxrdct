# Layered crystalline structures — model and theory

This page describes the physical model behind `LayeredCrystal` and its
associated utilities.  It covers the stacking geometry, the kinematical
structure factor for a multilayer stack, pseudomorphic epitaxial strain
mechanics (including the Poisson response, critical thickness, and how the
strained $d$-spacing is used in the simulation), and orientation relationships
between phases.

---

## 1. The stacking model

A `LayeredCrystal` represents an epitaxial thin-film stack divided into a
non-repeating buffer section plus **one or more independently-repeated
blocks** stacked above it:

```
┌───────────────────────────────────────┐  ← surface (top)
│  block 2  (layer C)                   │
│  ×  n_rep = 1        (a non-repeating │
│                        "cap" layer is │
│                        just a block   │
│                        with n_rep=1)  │
├───────────────────────────────────────┤
│  block 1  (layer A + layer B)         │
│  ×  n_rep = 4                         │
│  …                                    │
├───────────────────────────────────────┤
│  block 0  (layer A + layer B)         │
│  ×  n_rep = 5                         │
│  …                                    │
├───────────────────────────────────────┤
│  buffer layer  (e.g. template layer)  │
├───────────────────────────────────────┤
│  buffer layer  (e.g. substrate)       │
└───────────────────────────────────────┘  ← bottom (deepest)
```

| Section | Description |
|---|---|
| **Buffer layers** | Non-repeating layers at the very bottom of the stack (substrate, thick template layers). Added deepest-first via `add_buffer_layer`. Absorption limiting is applied automatically. Because they are always the bottommost section, a non-repeating layer that must sit *above* a repeating block (e.g. a cap layer) cannot be added this way — see below. |
| **Repeating blocks** | An ordered sequence of independently-repeated units (e.g. one bilayer period such as QW + barrier), stacked bottom to top above the buffer. Each block has its own layer list and its own repeat count. |

### 1.1 Building blocks

A block is simply "the layers added since the last block was closed".
`add_layer` (and `add_pseudomorphic_layer`, which calls it) appends to the
*currently open* block; `set_repetitions(n)` sets that block's repeat count
and **closes** it, so the next `add_layer` call starts a brand-new block:

```python
stack = laue.LayeredCrystal(stacking_direction=n_hat)

stack.add_layer(A, U_A, tA, label='A')
stack.add_layer(B, U_B, tB, label='B')
stack.set_repetitions(5)          # block 0 = (A, B) × 5

stack.add_layer(C, U_C, tC, label='C')
stack.set_repetitions(8)          # block 1 = (C,) × 8

stack.add_layer(D, U_D, tD, label='D')   # block 2 = (D,) × 1 -- a cap layer,
                                          # since no set_repetitions follows
```

If `set_repetitions` is never called at all, every `add_layer` call keeps
extending a single block — this is exactly the original single-repeating-unit
behaviour, so simple stacks are unaffected by the existence of multiple
blocks. `stack.blocks` returns the list of blocks (each with `.layers` and
`.n_rep`) for inspection; `stack.layers` / `stack.n_rep` remain valid
shortcuts *only* when the stack has a single block, and raise a clear error
otherwise (use `.blocks` instead).

A practical consequence: a non-repeating "cap" layer that must sit **above**
a repeating block (for example an electron-blocking layer on top of an MQW)
has to be added with `add_layer` — as its own trailing block with the
default `n_rep=1` — **not** with `add_buffer_layer`, since buffer layers are
always placed at the very bottom regardless of when they were added.

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

For a stack of buffer layers plus one repeating block of $N_\text{rep}$
repetitions, the total amplitude is

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

#### Several independently-repeated blocks

When the stack has several blocks (Section 1.1) — each with its own period
$\Lambda_b$, repeat count $N_{\text{rep},b}$, and start position $z_b$ — the
$F_\text{MQW}$ term above is replaced by a sum over blocks, each attenuated by
everything sitting **above** it:

$$
F_\text{MQW}(\mathbf{Q})
= \sum_{b} T_{\text{above},b}\;
    e^{\,i Q_n z_b}\;
    F_{\text{unit},b}(\mathbf{Q})\;
    S_{\text{rep},b}(Q_n \Lambda_b)
$$

where $T_{\text{above},b}$ is the transmission through every block that sits
above block $b$ (Section 7.2 generalises the same way).  Setting $b=1$
recovers the single-block formula exactly. This is what
`LayeredCrystal.structure_factor` / `average_structure_factor` compute
internally; you never need to sum over blocks by hand.

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
frame.  `pseudomorphic_d_spacing` raises `ValueError` for these cases —
**except** when `growth_dir` is left at its default `(0,0,1)`: that trivially
passes the $c$-axis check regardless of what direction the layer is actually
stacked along, so it silently returns a $c$-axis strain answer unrelated to
the real growth direction rather than raising. Always pass `growth_dir`
explicitly for anything other than default $c$-axis growth.

### 4.9 Non-polar (m-plane/a-plane) pseudomorphic strain

Non-polar hexagonal growth — m-plane `{1,0,-1,0}` or a-plane `{1,1,-2,0}` —
needs a genuinely different formula, not a rotated version of the $c$-axis
one, and `pseudomorphic_d_spacing`/`add_pseudomorphic_layer` cannot be reused
for it (see the warning above). `pseudomorphic_d_spacing_nonpolar` /
`add_pseudomorphic_layer_nonpolar` implement the correct version.

**Why the scalar formula fails here.** $\varepsilon_\perp = -2(C_{13}/C_{33})\varepsilon_\parallel$
relies on the two in-plane directions of a $c$-axis film being elastically
*identical* — both are $a$-type directions, related by the hexagonal basal
plane's transverse isotropy, so a single mismatch strain and a single ratio
suffice. A non-polar growth surface breaks that: its two in-plane directions
are $c$ itself (elastic constants $C_{13}$, $C_{33}$) and the *other*
remaining in-plane $a$-type direction (elastic constants $C_{11}$, $C_{12}$,
$C_{13}$) — not equivalent, so each generally has a different mismatch
strain and pulls on the free surface with a different coefficient.

**The correct formula.** Because hexagonal elastic stiffness is transversely
isotropic about $c$, the growth normal itself (an $a$-type direction) and the
in-plane $a$-type direction share the same $(C_{11}, C_{12})$ pair, with
$C_{13}$ coupling each to $c$ — so, unlike a truly semipolar direction, no
stiffness-tensor rotation is needed here, just a different combination of
the same five hexagonal constants. Solving $\sigma_\perp = 0$ in this local
frame:

$$
\boxed{
\varepsilon_\perp = -\frac{C_{12}\,\varepsilon_a + C_{13}\,\varepsilon_c}{C_{11}}
}
\qquad
\varepsilon_a = \frac{a_\text{sub} - a_\text{film}}{a_\text{film}},
\qquad
\varepsilon_c = \frac{c_\text{sub} - c_\text{film}}{c_\text{film}}
$$

Both substrate lattice parameters are needed — the growth surface contains
*both* the $a$- and $c$-type in-plane directions, so both are independently
lattice-matched to the template. This is why `pseudomorphic_d_spacing_nonpolar`
takes `c_substrate` in addition to `a_substrate`, and `C11`, `C12`, `C13`
instead of just `C13`, `C33`.

**Bulk repeat.** `d_bulk` along the growth normal comes from
`shortest_lattice_vector`, not `d_spacing_hkl` — see
[Section 5.1](#51-non-c-axis-stacking-m-plane-and-other-growth-planes) for
why these differ (a factor of 2 for GaN's m-plane).

```python
GaN   = xu.materials.GaN
InGaN = xu.materials.Alloy(GaN, InN, 0.10)   # In0.10Ga0.90N

c_GaN   = laue.nitride_elastic_constants('GaN')
c_InGaN = laue.nitride_elastic_constants('InN', x=0.10, end_material='GaN')

U_GaN = laue.orientation_along_plane([0, 1, 0], GaN, up_crystal=[0, 0, 1])
n_hat = U_GaN @ np.array([0.0, 1.0, 0.0])

stack = laue.LayeredCrystal(name='m-plane MQW', stacking_direction=n_hat)
stack.add_buffer_layer(GaN, U_GaN, thickness=20000.0, label='GaN core')

stack.add_pseudomorphic_layer_nonpolar(
    InGaN, U_GaN, thickness=60.0,
    a_substrate=GaN.lattice.a, c_substrate=GaN.lattice.c,
    C11=c_InGaN['C11'], C12=c_InGaN['C12'], C13=c_InGaN['C13'],
    growth_hkl=(0, 1, 0), label='InGaN QW')
stack.add_pseudomorphic_layer_nonpolar(
    GaN, U_GaN, thickness=80.0,
    a_substrate=GaN.lattice.a, c_substrate=GaN.lattice.c,
    C11=c_GaN['C11'], C12=c_GaN['C12'], C13=c_GaN['C13'],
    growth_hkl=(0, 1, 0), label='GaN barrier')
stack.set_repetitions(15)
# eps_a = -0.0110  eps_c = -0.0099  eps_perp = +0.0070  d_strained = 5.6241 A
```

`growth_hkl` must match whichever specific `{1,0,-1,0}`-family member is
physically correct for *this* `U` — there are 6 symmetry-equivalent choices
around the hexagonal cross-section (see
[Section 6.4](#64-building-u-for-a-general-growth-plane)); nothing here
infers which one your crystal/orientation actually uses.

Note that this computes the **idealized fully-coherent** (zero-relaxation)
strain — the same assumption `add_pseudomorphic_layer` already makes for
$c$-axis growth (Section 4.6). Real core–shell nanowires typically relax
substantially *more* than an equivalent planar film of the same nominal
thickness, since the free lateral wire surfaces allow elastic relaxation a
flat substrate does not; measured strain can therefore come out well below
this idealized estimate.

### 4.10 Worked example — In$_{0.20}$Ga$_{0.80}$N on GaN

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

Every unit cell in a `Layer` is assumed to contribute the **same** structure
factor $F_\text{uc}(\mathbf{Q}_\text{cry})$, evaluated once. That is only
correct if stepping by $d$ along $\hat{n}$ is a genuine **lattice
translation** — i.e. it reproduces an identical atomic arrangement each time.
$d$ is therefore *not*, in general, the same thing as an interplanar spacing
$d_{hkl}$; see [Section 5.1](#51-non-c-axis-stacking-m-plane-and-other-growth-planes)
for why they can differ by an integer factor.

`d_spacing_hkl(crystal, h, k, l)` computes the bare interplanar spacing of the
$(hkl)$ family using the reciprocal lattice directly:

$$
d_{hkl} = \frac{2\pi}{|\mathbf{G}_{hkl}|}
= \frac{2\pi}{|h\,\mathbf{b}_1 + k\,\mathbf{b}_2 + l\,\mathbf{b}_3|}
$$

This is valid for any crystal system, and is the correct stacking $d$ for
$c$-axis wurtzite growth ($d_{0002} = c/2$) and any orthogonal-lattice growth
plane where the plane normal happens to be parallel to a single primitive
lattice vector.

If `d_spacing` is not supplied to `add_layer`, the `Layer` class finds $d$
automatically via `shortest_lattice_vector(U.T @ n_hat, crystal)`: an
exhaustive search over small integer combinations
$T = p\,\mathbf{a}_1 + q\,\mathbf{a}_2 + r\,\mathbf{a}_3$ of the direct-lattice
basis for the **shortest one parallel to** $\hat{n}$, using $d=|T|$. This
generalises the older "project each primitive vector individually" heuristic,
which only found $T$ when it happened to be a single $\mathbf{a}_i$ (true for
cubic axes and hexagonal $c$-axis growth, but not in general — see below). It
raises `ValueError` if no such $T$ exists within its search range, i.e. if
$\hat{n}$ is not a lattice-commensurate direction for this crystal (some
semipolar planes) — in that case supply `d_spacing` explicitly.

### 5.1 Non-`c`-axis stacking: m-plane and other growth planes

For a **hexagonal m-plane** film (surface normal along the prism-plane family
$\{1,0,\bar{1},0\}$ — the 3-index equivalent used throughout this module is
$(1,0,0)$), the shortest lattice vector parallel to the growth direction is
**not** a primitive vector but the combination

$$
\mathbf{T} = 2\mathbf{a}_1 + \mathbf{a}_2, \qquad |\mathbf{T}| = a\sqrt{3}
$$

— exactly **twice** the bare interplanar spacing $d_{(1,0,0)} = a\sqrt{3}/2$
returned by `d_spacing_hkl`. This is not a numerical curiosity: wurtzite has
two formula units per conventional cell, so translating by one bare
$d_{(1,0,0)}$ step does **not** reproduce an identical atomic plane — only
every *second* step does. Using $d_{(1,0,0)}$ directly as the stacking
repeat would impose a spurious periodicity at half the true repeat,
corrupting satellite spacing and systematic absences for that reflection.
`shortest_lattice_vector` finds $\mathbf{T}$ (not just $d_{hkl}$) precisely to
avoid this.

This is a special case of a more general geometric fact: in a **non-orthogonal**
lattice (hexagonal, monoclinic, triclinic), a plane normal $(hkl)$ is in
general **not parallel** to the real-space direction $[hkl]$ of the same
indices — they coincide only in orthogonal (cubic, tetragonal, orthorhombic)
systems, or for the hexagonal $c$-axis (where $[0001] \parallel (0001)^*$
regardless of $a/c$). For the m-plane, the real-space $[1,0,0]$ direction is
actually parallel to the **a-plane** normal $\{1,1,\bar{2},0\}$ instead — a
30° rotation away from the m-plane normal in the basal plane:

```python
>>> import numpy as np
>>> np.dot(orientation_along_z([1,0,0], GaN)[:, 0],
...        orientation_along_plane([1,0,0], GaN)[:, 0])
0.8660...   # cos(30°)
```

Concretely, for m-plane (or any non-`c`-axis hexagonal, or generally
non-cubic) growth:

1. **Build $U$ with `orientation_along_plane`, not `orientation_along_z`.**
   `orientation_along_z([1,0,0], GaN)` places the *a-axis real-space
   direction* along the stacking axis — the wrong surface for an m-plane
   film. `orientation_along_plane((1,0,0), GaN)` places the *reciprocal*
   $\mathbf{G}_{(1,0,0)}$ — the actual m-plane normal — along $\hat{n}$
   instead, by aligning `plane_normal_cartesian(hkl, crystal)` rather than
   `crystal_to_cartesian(uvw, crystal)`.
2. **Let `Layer` find $d$ automatically**, or pass it explicitly via
   `shortest_lattice_vector` if you need to inspect it first — do **not**
   pass `d_spacing_hkl(...)` directly for a non-orthogonal-lattice growth
   plane, since (as above) it can silently be a submultiple of the true
   repeat.

```python
import numpy as np
import xrayutilities as xu
import nrxrdct.laue as laue

GaN = xu.materials.GaN

# m-plane surface normal along lab z; in-plane c-axis fixed along lab x
U_GaN = laue.orientation_along_plane([1, 0, 0], GaN, up_crystal=[0, 0, 1])
n_hat = U_GaN @ np.array([1.0, 0.0, 0.0])   # growth dir, crystal frame -> lab z

stack = laue.LayeredCrystal(name='m-plane GaN', stacking_direction=n_hat)
stack.add_buffer_layer(GaN, U_GaN, thickness=5000.0, label='GaN m-plane substrate')
# d auto-resolves to a*sqrt(3) ~= 5.524 A here, not d_spacing_hkl(GaN,1,0,0) ~= 2.762 A
```

If `shortest_lattice_vector` cannot find a lattice-parallel translation within
its default search range (most likely for a semipolar plane, where the
real/reciprocal correspondence depends on $a/c$ and need not close on a short
integer vector), `Layer` raises a `ValueError` naming the failing `n_hat`
rather than silently returning an incorrect $d$ — pass `d_spacing` by hand in
that case.

Once $U$ and $\hat{n}$ are set up this way, every other part of the model —
`structure_factor`, superlattice satellites, absorption depth, and rod
tangency (see the [Rod Tangency](laue_rod_tangency.md) page) — is unchanged:
they all consume `n_hat` and the resolved period generically and do not need
to know which plane it came from.

One part is **not** unchanged: pseudomorphically strained layers on a
non-`c`-axis surface (m-plane, a-plane) need
`add_pseudomorphic_layer_nonpolar`, not `add_pseudomorphic_layer` — the two
in-plane directions of such a surface are elastically inequivalent, which
needs a different formula entirely, not just a different `growth_dir`. See
[Section 4.9](#49-non-polar-m-planea-plane-pseudomorphic-strain).

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

### 6.4 Building $U$ for a general growth plane

`orientation_along_z(zone_axis_crystal, crystal)` and
`orientation_along_plane(hkl, crystal)` both return a $U$ placing something
along lab $z$, but a **different** something:

| Function | Aligns to lab $z$ | Correct when |
|---|---|---|
| `orientation_along_z` | real-space direction $[uvw]$ | cubic (any $[uvw]$); hexagonal $c$-axis $[0001]$ |
| `orientation_along_plane` | reciprocal plane normal $\mathbf{G}_{hkl}$ | any crystal system, any growth plane |

They agree only when $[uvw]$ (same indices as `hkl`) happens to be parallel
to $\mathbf{G}_{hkl}$ — true in orthogonal lattices, and true for hexagonal
$(0001)$, but **not** true for hexagonal m-plane / a-plane growth (see
[Section 5.1](#51-non-c-axis-stacking-m-plane-and-other-growth-planes)). Use
`orientation_along_plane` whenever the growth surface is specified by Miller
plane indices rather than a zone axis — which is the physically natural way
to specify an epitaxial growth surface in the first place, since the surface
*is* parallel to the $(hkl)$ planes, not necessarily to the $[hkl]$ direction.

### 6.5 Adjusting orientation after the fact — `rotate`

`LayeredCrystal.rotate(angle_deg, axis, frame='lab', layers=None)` nudges one
or more layers' `U` by a small rotation without rebuilding the stack —
useful for exploring whether a residual misorientation between measured and
simulated patterns is explained by a rotation about a specific axis. It
supports two, differently-composed, conventions:

| `frame` | `axis` is expressed in | Applied as | Physical meaning |
|---|---|---|---|
| `'lab'` (default) | the lab frame | $U_\text{new} = R(\hat{n})\,U$ | rotating the *sample* about a fixed external axis (e.g. vertical, or the beam) |
| `'crystal'` | each selected layer's own crystal frame | $U_\text{new} = U\,R(\hat{n})$ | twisting the *lattice* about its own crystallographic axis, independent of the layer's absolute orientation |

Because `frame='crystal'` composes on the right, applying it to several
layers that already have *different* `U` (e.g. an independently fitted core
and shell) twists each about *its own* `c`, not about one shared lab
direction — exactly what you want when probing a possible relative twist
between them:

```python
# Nudge only the shell by 0.05° about its own c-axis and re-check the fit
stack.rotate(0.05, [0, 0, 1], frame='crystal', layers='InGaN QW')
```

`layers` accepts `None` (every layer, the default), a single `Layer`/label,
or a list of either — same resolution rule as `print_reflections`. For a
permanent reassignment rather than an incremental nudge, use
`LayeredCrystal.set_U` instead; `rotate` composes onto whatever `U` a layer
already has.

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
$z_i$ below the surface, all shallower layers (buffer layers $j > i$ and
every repeating block above it) contribute a multiplicative transmission
factor

$$
T_\text{above}^{(i)} = \prod_{j > i} T_j \;\times\; \prod_b T_{\text{block},b}
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

With a single repeating block, that block sits at the top of the stack and
has no overlying material, so no $T_\text{above}$ factor is applied to its
own amplitude. With **several** blocks (Section 1.1), a deeper block *is*
attenuated by every block stacked above it — using the same two-beam $T_j$
formula, evaluated over that whole block's thickness ($t_j \to \Lambda_b
\times N_{\text{rep},b}$) — while the topmost block still has
$T_\text{above} = 1$.

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

### 10.1 Single repeating block

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
stack = laue.LayeredCrystal(stacking_direction=(0, 0, 1))

stack.add_buffer_layer(GaN, U, thickness=5000.0,     # 500 nm
                       d_spacing=d_GaN, label='GaN buffer')

stack.add_pseudomorphic_layer(                        # computes d_strained
    InGaN, U, thickness=30.0,                         # 3 nm QW
    a_substrate=GaN.lattice.a,
    C13=C['C13'], C33=C['C33'], label='InGaN QW')

stack.add_layer(GaN, U, thickness=100.0,              # 10 nm barrier
                d_spacing=d_GaN, label='GaN barrier')

stack.set_repetitions(10)   # QW + barrier, × 10

stack.describe()

# Visualise
stack.plot_lattice_parameter('c', unit='nm')
stack.plot_strain_profile('c', reference=GaN)
```

### 10.2 Several independently-repeated blocks

Real device stacks often have several MQW sections with *different* repeat
counts stacked on top of each other (defect-filtering superlattice → active
region → optical cladding → a non-repeating cap), which is exactly what
`nrxrdct.laue.crystal.build_MLed` builds. Each section is one call to
`add_layer` (or `add_pseudomorphic_layer`) per layer, followed by
`set_repetitions` — see Section 1.1 for how block boundaries work:

```python
stack = laue.LayeredCrystal(stacking_direction=(0, 0, 1))

stack.add_buffer_layer(GaN, U, thickness=600e4,   # 600 µm, in Å
                       label='substrate')

# block 0: defect-filtering superlattice, × 5
stack.add_pseudomorphic_layer(GaN,         U, t_GaN,    a_sub, C13, C33, label='defect')
stack.add_pseudomorphic_layer(InGaN_low_x, U, t_InGaN,  a_sub, C13, C33, label='defect')
stack.set_repetitions(5)

# block 1: active region, × 4
stack.add_pseudomorphic_layer(GaN,          U, t_GaN,   a_sub, C13, C33, label='active')
stack.add_pseudomorphic_layer(InGaN_high_x, U, t_InGaN, a_sub, C13, C33, label='active')
stack.set_repetitions(4)

# block 2: cap layer, × 1 (no set_repetitions call needed -- it's the last block)
stack.add_layer(AlGaN_ebl, U, thickness=1600.0, label='EBL')

stack.describe()               # prints buffer layers, then each block in turn
stack.plot_layer_scheme()      # draws every block at its own real scale

# `.n_rep` / `.layers` only work for a single-block stack -- inspect `.blocks`
# for a multi-block one:
for blk in stack.blocks:
    print([lyr.label for lyr in blk.layers], '×', blk.n_rep)

# simulate_laue_stack (and the rest of the Laue-spot simulation pipeline)
# only understands a single repeating block -- flatten first:
flat_stack = laue.combine_stacks([stack])
```

### 10.3 Saving and reloading a stack

`LayeredCrystal.save`/`.load` persist a whole stack (materials, orientations,
thicknesses, resolved `d`-spacings, block/repetition structure, buffer
layers) to a single file via `dill` — the same mechanism `LayeredMap`
already uses internally to ship a stack to local worker processes (see
[Section 7](#7-absorption-corrections-two-beam-path-and-overlying-layers)'s
neighbour, `layered_map._serialize_stack`), just to a permanent path instead
of a temp file:

```python
stack.save('gan_mplane_stack.pkl')

# ...resume later, or in a different session/notebook...
stack = laue.LayeredCrystal.load('gan_mplane_stack.pkl')
```

This is the natural hand-off into a per-pixel map: build (and validate) one
stack by hand for a representative pixel — orientation, non-polar
pseudomorphic strain, everything from Sections 4–6 — save it once, then reuse
it as the starting-point orientation for every pixel in a
[`LayeredMap`](../api/laue/layered_map.md):

```python
stack = laue.LayeredCrystal.load('gan_mplane_stack.pkl')
lmap = laue.LayeredMap(ny=21, nx=21, stack=stack, h5_path='scan.h5')
lmap.run_orientation_local(camera, seg_dir='seg/', out_dir='ubs/')
```

`LayeredMap.save`/`.load` (its own, separate persistence) only stores
per-pixel numeric arrays, not the stack itself — `LayeredMap.load(path, stack=...)`
still expects a `stack` argument, which is exactly what
`LayeredCrystal.load` is for.

---

## References

1. **Matthews, J. W. & Blakeslee, A. E.** Defects in epitaxial multilayers. *J. Cryst. Growth* **27**, 118–125 (1974). *(Critical thickness for pseudomorphic growth.)*
2. **Wright, A. F.** Elastic properties of zinc-blende and wurtzite AlN, GaN, and InN. *Phys. Rev. B* **55**, 6250–6258 (1997).
3. **Vurgaftman, I. & Meyer, J. R.** Band parameters for nitrogen-containing semiconductors. *J. Appl. Phys.* **94**, 3675–3696 (2003). *(Elastic constants and Vegard's law for III-nitrides.)*
4. **Kurdjumov, G. & Sachs, G.** Über den Mechanismus der Stahlhärtung. *Z. Phys.* **64**, 325–343 (1930). *(Original KS orientation relationship.)*
5. **Nishiyama, Z.** *Martensitic Transformation*. Academic Press, New York, 1978. *(NW relationship and transformation crystallography.)*
6. **Authier, A.** *Dynamical Theory of X-Ray Diffraction*. Oxford University Press, 2001. *(Chapter 2 — kinematical limit and the structure factor of a finite crystal.)*
