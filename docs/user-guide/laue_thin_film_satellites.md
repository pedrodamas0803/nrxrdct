# Thin-film satellites and thickness fringes in Laue diffraction

White-beam Laue diffraction can reveal **thickness fringes** and
**superlattice satellites** from thin epitaxial films and multilayer stacks.
This page derives the positions and intensities of those features and explains
the conventions used by `simulate_laue_stack`.

---

## 1. Single-layer interference — the Laue function

A crystalline slab of $N$ unit cells, each of thickness $d$ along the stacking
direction $\hat{n}$, contributes a scattering amplitude

$$
F_\text{slab}(\mathbf{Q}) = f_\text{cell}(\mathbf{Q})\,
\sum_{n=0}^{N-1} e^{\,i n \varphi}, \qquad
\varphi = \mathbf{Q}\cdot d\,\hat{n}
$$

where $f_\text{cell}$ is the unit-cell structure factor and $t = Nd$ is the
total layer thickness.  The geometric sum evaluates to

$$
F_\text{slab} = f_\text{cell}\,
\frac{\sin(N\varphi/2)}{\sin(\varphi/2)}\,
e^{\,i(N-1)\varphi/2}
$$

and its squared modulus — the **Laue interference function** — is

$$
\left|F_\text{slab}\right|^2 = \left|f_\text{cell}\right|^2
\frac{\sin^2(N\varphi/2)}{\sin^2(\varphi/2)}.
$$

### Bragg peaks

At reciprocal-lattice vectors $\mathbf{G}_{hkl}$, $\varphi = 2\pi\ell$
(integer), and $|F_\text{slab}|^2 = N^2\,|f_\text{cell}|^2$.

### Zeros between Bragg peaks

$|F_\text{slab}|^2 = 0$ whenever $\varphi = 2\pi\ell + 2\pi m/N$, i.e.

$$
\Delta q_n \equiv (\mathbf{Q} - \mathbf{G}_{hkl})\cdot\hat{n}
= \frac{2\pi m}{t}, \qquad m = \pm 1,\pm 2,\ldots
$$

> **Important:** these integer-$m$ positions are *dark* fringes (zeros), **not**
> the observable bright fringes.

### Side maxima (observable thickness fringes)

The subsidiary maxima of $\sin^2(N\varphi/2)/\sin^2(\varphi/2)$ occur
between consecutive zeros.  For large $N$ they converge to the
*half-integer* positions

$$
\boxed{
\Delta q_n \approx \left(|m| + \tfrac{1}{2}\right)\frac{2\pi}{t},
\qquad m = \pm 1, \pm 2, \ldots
}
$$

The first side maximum ($|m|=1$) lies at $\approx 1.43\,(2\pi/t)$
(converging toward $1.5\,(2\pi/t)$ for large $N$).

The intensity of the $m$-th side maximum relative to the Bragg peak is

$$
\frac{|F_\text{sat}|^2}{|F_\text{Bragg}|^2}
\approx \frac{4}{\pi^2(2|m|+1)^2} \approx
\begin{cases}
4.5\,\% & |m|=1 \\
0.8\,\% & |m|=2 \\
0.3\,\% & |m|=3
\end{cases}
$$

---

## 2. Satellite positions in the lab frame

In the LaueTools lab frame ($x \parallel$ beam, $z$ vertical), the stacking
direction is

$$
\hat{n}_\text{lab} = U\,\hat{n}_\text{crystal}
$$

where $U$ is the $3\times3$ orientation matrix from Laue indexation
(columns are crystal basis vectors expressed in lab coordinates) and
$\hat{n}_\text{crystal}$ is the growth direction in the crystal frame
(e.g.\ $[001]$ for $c$-axis GaN).

The satellite wavevectors are

$$
\mathbf{G}_\text{sat}^{(m)} = \mathbf{G}_{hkl}
+ \left(|m| + \tfrac{1}{2}\right)\operatorname{sgn}(m)\,
\frac{2\pi}{t}\,\hat{n}_\text{lab},
\qquad m = \pm 1, \pm 2, \ldots
$$

Each satellite satisfies the Laue condition at its own photon energy

$$
E_\text{sat}^{(m)} = -\frac{\hbar c\,|\mathbf{G}_\text{sat}|^2}
{2\,G_{\text{sat},x}}
$$

which is slightly different from the Bragg energy $E_0$ of the parent
reflection.  Whether a given satellite falls within the white-beam energy
window $[E_\text{min}, E_\text{max}]$ depends on the geometry; typically
only one of $m=+1$ or $m=-1$ is accessible for a given reflection.

---

## 3. Layered / superlattice structures

For a bilayer stack with $N_\text{rep}$ repetitions, the period
$\Lambda = t_A + t_B$ gives additional **superlattice satellites** at

$$
\mathbf{G}_\text{SL}^{(m)} = \mathbf{G}_{hkl}
+ m\,\frac{2\pi}{\Lambda}\,\hat{n}_\text{lab}, \qquad m = \pm 1,\pm 2,\ldots
$$

These are true satellites (not zeros) because the superlattice period $\Lambda$
is the repeat unit, not the individual-layer thickness.  For $N_\text{rep}=1$
only the single-layer thickness fringes at $\pm(2\pi/t)$ exist.

The total stack structure factor coherently sums all layer contributions
weighted by their phase offsets $z_j$ along $\hat{n}$:

$$
F_\text{stack}(\mathbf{Q}) =
\sum_j F_j(\mathbf{Q})\,e^{\,i\mathbf{Q}\cdot z_j\hat{n}}
$$

---

## 4. Detector displacement direction

This section works out where a satellite lands on the detector relative to
its parent Bragg spot, in five steps:

1. the spot direction depends only on $\hat{G}$;
2. only the part of $\delta\mathbf{G}$ perpendicular to $\mathbf{G}$ moves the spot;
3. that tilt is split into radial ($2\theta$) and azimuthal parts;
4. the change of direction is projected onto the flat detector;
5. the energy shift is computed.

Throughout, the satellite offset is

$$
\delta\mathbf{G} \equiv \mathbf{G}_\text{sat} - \mathbf{G}_{hkl}
= s\,q\,\hat{n}_\text{lab},
\qquad s = \left(|m|+\tfrac{1}{2}\right)\operatorname{sgn}(m),
\quad q = \frac{2\pi}{t},
$$

and $|\delta\mathbf{G}| \ll |\mathbf{G}_{hkl}|$ (for $t = 50$ nm,
$q \approx 0.013$ Å$^{-1}$, compared with $|\mathbf{G}| \sim 2$–$10$ Å$^{-1}$).

### Step 1 — In Laue geometry the spot direction depends only on $\hat{G}$

The incident beam is $\mathbf{k}_i = k\,\hat{x}$ with $k = 2\pi/\lambda = E/\hbar c$.
Elastic scattering requires $|\mathbf{k}_i + \mathbf{G}| = |\mathbf{k}_i|$:

$$
|\mathbf{k}_i + \mathbf{G}|^2 = k^2
\;\Longrightarrow\;
2k\,G_x + |\mathbf{G}|^2 = 0
\;\Longrightarrow\;
k = -\frac{|\mathbf{G}|^2}{2\,G_x},
$$

which is the energy formula from §2 (a reflection is accessible only if
$G_x < 0$).  Substituting this $k$ back into $\mathbf{k}_f = k\hat{x} + \mathbf{G}$:

$$
\hat{k}_f = \frac{\mathbf{k}_f}{k}
= \hat{x} + \frac{\mathbf{G}}{k}
= \hat{x} - \frac{2\,G_x}{|\mathbf{G}|^2}\,\mathbf{G}
= \hat{x} - 2\,(\hat{x}\cdot\hat{G})\,\hat{G}.
$$

This is the mirror reflection of the beam direction in the lattice plane with
normal $\hat{G}$.  $|\mathbf{G}|$ has cancelled out, so it sets only the
wavelength and not the spot position.  The white beam simply supplies whichever
wavelength is needed.

### Step 2 — Only the component of $\delta\mathbf{G}$ perpendicular to $\mathbf{G}$ moves the spot

Split the offset into components parallel and perpendicular to $\mathbf{G}$:

$$
\delta\mathbf{G} = \delta G_\parallel\,\hat{G} + \delta\mathbf{G}_\perp,
\qquad
\delta G_\parallel = \delta\mathbf{G}\cdot\hat{G},
\qquad
\delta\mathbf{G}_\perp = \delta\mathbf{G} - (\delta\mathbf{G}\cdot\hat{G})\,\hat{G}.
$$

To first order, the unit vector changes by

$$
\delta\hat{G} = \frac{\delta\mathbf{G}_\perp}{|\mathbf{G}|}
= \frac{s\,q}{|\mathbf{G}|}\,
\bigl[\hat{n}_\text{lab} - (\hat{n}_\text{lab}\cdot\hat{G})\,\hat{G}\bigr].
$$

$\delta G_\parallel$ only rescales $|\mathbf{G}|$. By Step 1 that changes the
energy but not the pixel.  **The quantity that sets the displacement is the
projection of $\hat{n}_\text{lab}$ perpendicular to $\mathbf{G}$.**  It is
*not* the projection onto the detector plane, and not the part perpendicular
to $\hat{k}_f$.

> **Special case — symmetric reflections.**  If $\mathbf{G}_{hkl} \parallel \hat{n}$
> (e.g. $00\ell$ for $c$-axis growth), $\delta\mathbf{G}_\perp = 0$.  All
> fringes then fall **on the same pixel** as the Bragg spot at slightly
> different energies.  A white-beam detector that does not resolve energy
> cannot separate them.  Fringes are visible as separate spots only on
> asymmetric reflections.

### Step 3 — Radial and azimuthal parts of the tilt

Differentiate the mirror formula from Step 1:

$$
\delta\hat{k}_f = -2\left[(\hat{x}\cdot\delta\hat{G})\,\hat{G}
+ (\hat{x}\cdot\hat{G})\,\delta\hat{G}\right].
$$

To read this off, define an orthonormal frame attached to the reflection.
$\theta$ is the Bragg angle, so $\hat{x}\cdot\hat{G} = -\sin\theta$.

* $\hat{G}$: the scattering vector direction;
* $\hat{e}_\parallel$: the unit vector perpendicular to $\hat{G}$ **in** the
  scattering plane (the plane containing $\hat{x}$ and $\hat{G}$), chosen so
  that $\hat{x} = -\sin\theta\,\hat{G} + \cos\theta\,\hat{e}_\parallel$;
* $\hat{e}_\perp = \hat{G}\times\hat{e}_\parallel$: perpendicular to the
  scattering plane.

In this frame $\hat{k}_f = \sin\theta\,\hat{G} + \cos\theta\,\hat{e}_\parallel$.
Write the tilt as $\delta\hat{G} = \alpha\,\hat{e}_\parallel + \beta\,\hat{e}_\perp$,
with

$$
\alpha = \frac{\delta\mathbf{G}\cdot\hat{e}_\parallel}{|\mathbf{G}|},
\qquad
\beta = \frac{\delta\mathbf{G}\cdot\hat{e}_\perp}{|\mathbf{G}|}.
$$

Then $\hat{x}\cdot\delta\hat{G} = \alpha\cos\theta$. Substituting:

$$
\delta\hat{k}_f
= -2\alpha\,\underbrace{\left(\cos\theta\,\hat{G} - \sin\theta\,\hat{e}_\parallel\right)}_{\text{unit vector in scattering plane},\ \perp\,\hat{k}_f}
\;+\; 2\beta\sin\theta\,\hat{e}_\perp .
$$

So:

| Tilt of $\hat{G}$ | Effect on the scattered beam | On the detector |
|---|---|---|
| $\alpha$ (in scattering plane) | rotates by $2\alpha$, i.e. $\Delta(2\theta) = -2\alpha$ | **radial** shift (along the $2\theta$ direction) |
| $\beta$ (out of scattering plane) | rotates by $2\beta\sin\theta$ | **azimuthal** shift (along the $\chi$ / Debye-ring direction) |

The in-plane tilt is doubled, as for any mirror.  The out-of-plane tilt is
reduced by $\sin\theta$, so for low-angle reflections most of the visible
displacement is radial.

### Step 4 — Projection onto the flat detector

Put the sample at the origin. Let the detector plane have unit normal
$\hat{n}_d$ and lie a distance $D$ from the sample along $\hat{n}_d$.  The
scattered ray reaches the detector at

$$
\mathbf{P} = \frac{D}{\hat{k}_f\cdot\hat{n}_d}\,\hat{k}_f .
$$

Varying $\hat{k}_f$ (both the numerator and the denominator change) gives the
in-plane displacement:

$$
\delta\mathbf{P} = \frac{D}{\hat{k}_f\cdot\hat{n}_d}
\left[\delta\hat{k}_f
- \frac{\delta\hat{k}_f\cdot\hat{n}_d}{\hat{k}_f\cdot\hat{n}_d}\,\hat{k}_f\right],
\qquad \delta\mathbf{P}\cdot\hat{n}_d = 0 .
$$

The pixel displacement is $\delta\mathbf{P}$ projected onto the detector's
two pixel axes and divided by the pixel size.  For a spot near the detector
centre ($\hat{k}_f \approx \hat{n}_d$) this reduces to
$\delta\mathbf{P} \approx D\,\delta\hat{k}_f$.

In practice `simulate_laue_stack` does not use this linearisation.  It
projects $\mathbf{G}_\text{sat}$ exactly with the `Camera` geometry, just like
any Bragg reflection.  Steps 1–4 are for understanding and estimating the
displacement.

### Step 5 — Energy shift

From Step 1, $E = \hbar c\,|\mathbf{G}|^2 / (2|G_x|) = \hbar c\,|\mathbf{G}|/(2\sin\theta)$.
Take the logarithmic derivative and use $\delta\sin\theta = -\hat{x}\cdot\delta\hat{G} = -\alpha\cos\theta$:

$$
\frac{\delta E}{E} = \frac{\delta G_\parallel}{|\mathbf{G}|} + \alpha\cot\theta .
$$

The $m = +1$ and $m = -1$ satellites therefore sit at energies on opposite
sides of $E_0$.  Near the edge of the spectrum, only one of the two may fall
inside $[E_\text{min}, E_\text{max}]$.

### Worked example

Take a $t = 50$ nm film and a reflection with $d_{hkl} = 1$ Å
($|\mathbf{G}| = 2\pi$ Å$^{-1}$), and suppose $\hat{n}$ lies entirely in the
scattering plane, perpendicular to $\mathbf{G}$.  For the first fringe:

$$
|\delta\mathbf{G}| = 1.5 \times \frac{2\pi}{500\ \text{Å}} \approx 0.019\ \text{Å}^{-1},
\qquad
\alpha = \frac{0.019}{6.28} \approx 3.0\times10^{-3}\ \text{rad},
$$

$$
|\Delta(2\theta)| = 2\alpha \approx 6.0\ \text{mrad} \approx 0.34^\circ,
\qquad
|\delta\mathbf{P}| \approx D\cdot 2\alpha \approx 0.48\ \text{mm at } D = 80\ \text{mm},
$$

which is about 6 pixels on an 80 µm-pixel detector.  The displacement scales
as $1/(t\,|\mathbf{G}|)$, so thinner films and lower-order reflections give
satellites that are easier to resolve.

### Why flipping $\hat{n}$ alone does not flip the satellite side

Both $m=+1$ ($\delta\mathbf{G} = +1.5\,q\,\hat{n}$) and $m=-1$
($\delta\mathbf{G} = -1.5\,q\,\hat{n}$) are always enumerated.  Under
$\hat{n}\to-\hat{n}$ the pair of offset vectors $\{+\delta\mathbf{G},
-\delta\mathbf{G}\}$ is mapped onto itself, with the $m$-labels swapped.  The
set of $\mathbf{G}_\text{sat}$ vectors is unchanged, so Steps 1–5 give the
same pixels and energies.  Flipping $\hat{n}$ does **not** move any
spot.

What the sign of $\hat{n}$ *does* affect is the layer phases
$e^{i\mathbf{Q}\cdot z_j\hat{n}}$ in $F_\text{stack}$ (§3).  Reversing
$\hat{n}$ reverses the stacking order, which changes how the layers interfere.
In a strained or multi-layer stack this can make one side's fringes brighter
than the other's.  To get the right *intensities*, make sure the stacking
direction $\hat{n}_\text{crystal}$ points **from substrate toward surface**
(the growth direction).  For $c$-axis GaN use $[001]$, not $[00\bar 1]$.

---

## 5. Signal-to-noise considerations

Satellite spots are intrinsically weaker than Bragg peaks:

| Feature | $\lvert F\rvert^2 / \lvert F_\text{Bragg}\rvert^2$ |
|---|---|
| Bragg peak | $1$ |
| 1st thickness fringe | $\approx 0.045$ |
| 2nd thickness fringe | $\approx 0.008$ |
| Superlattice satellite ($N_\text{rep} \gg 1$) | $\approx 4/(\pi^2 m^2)$ |

In `simulate_laue_stack` the structure-factor threshold `f2_thresh` is
auto-calibrated from the strongest Bragg peak.  Satellite spots use an
effective threshold of `f2_thresh × 1e-4` so that thin-layer fringes are
not suppressed.

---

## 6. Implementation in `simulate_laue_stack`

The key steps in the simulation are:

1. **Collect fringe periods** — for each layer thinner than 2 µm compute
   $\mathbf{q}_\text{fringe} = (2\pi/t)\,\hat{n}_\text{lab}$.
2. **Select enumeration crystals** — determined by `structure_model` (see
   [Structure model](laue_layered_structures.md#3-structure-model)):
   all layers in `'coherent'` mode, buffer layers only in `'average'` mode.
3. **Probe satellite positions** — for each Bragg reflection $\mathbf{G}_{hkl}$
   and each fringe period, evaluate  
   $\mathbf{G}_\text{sat} = \mathbf{G}_{hkl} + (|m|+\tfrac{1}{2})\operatorname{sgn}(m)\,\mathbf{q}_\text{fringe}$  
   for $m = \pm 1, \ldots, \pm m_\text{max}$.
4. **Laue condition** — compute the required wavelength and check it lies in
   $[\lambda_\text{lo}, \lambda_\text{hi}]$.
5. **Project onto detector** — use the `Camera` geometry to find the pixel;
   discard spots that miss the active area.
6. **Structure factor** — evaluate $|F_\text{stack}(\mathbf{G}_\text{sat})|^2$
   using either the full coherent sum or the average-period model depending on
   `structure_model`; apply relaxed threshold for $m \neq 0$.
7. **Intensity** — $I \propto |F|^2 \times LP(2\theta) \times S(E)$, where
   $LP$ is the Lorentz–polarisation factor and $S(E)$ is the synchrotron
   spectrum.
