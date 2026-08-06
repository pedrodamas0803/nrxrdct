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

**Why these are exact zeros, not just small.** $F_\text{slab}$ is a sum of $N$
unit-length phasors, one per unit cell, with phasor $n$ pointing in direction
$n\varphi$ (its position along $\hat n$ is $nd$, so its phase is $\mathbf
Q\cdot nd\,\hat n = n\varphi$). At the Bragg condition $\varphi = 2\pi\ell$
every phasor points the same way, so they add tip-to-tail in a straight line
and reach the maximum possible length $N$ — hence $|F_\text{slab}|^2=N^2$.
Move $\varphi$ away from that condition and successive phasors are rotated
a little further from their neighbour, so the chain starts to curl into an
arc instead of a straight line, and the resultant (the chord closing the
arc) shortens. It closes all the way to a point — the phasors return exactly
to where they started, cancelling completely — whenever the chain completes
a whole number of turns over its $N$ steps without that number being a
multiple of $N$ itself. That is exactly the condition $N\varphi = 2\pi(N\ell + m)$
with $m\not\equiv 0\pmod N$, i.e. $\varphi = 2\pi\ell + 2\pi m/N$: the $N$
phasors sit at the $N$ evenly-spaced vertices of a closed regular polygon
(the $N$-th roots of unity, rotated by $2\pi\ell$), and any set of $N$
equally-spaced unit vectors around a full circle sums to zero by symmetry —
each vector is exactly cancelled by the one diametrically opposite it (or,
for odd $N$, by the vector sum of its neighbours). This is the same
root-of-unity cancellation behind the sharp dark fringes of an $N$-slit
optical diffraction grating, since $F_\text{slab}$ has the identical
finite-geometric-series form $\sum_n e^{in\varphi}$.

Between two such zeros the phasor chain traces only a fraction of a full
turn, so it doesn't close — there is always some nonzero net chord — which
is why intensity revives into the side maxima described next. It never
revives anywhere near $N^2$, though: even at the best-case angle between two
zeros the chain has curled through the better part of a full turn, so the
chord stays short compared to the fully-aligned $N$-phasor line at a true
Bragg peak. That is the geometric reason side-maximum intensities are
suppressed by roughly $1/m^2$ relative to the Bragg peak (worked out below),
not just "smaller by some amount."

> **Important:** the integer-$m$ positions above are *dark* fringes (zeros),
> **not** the observable bright fringes — don't mistake $\Delta q_n = 2\pi/t$
> for the first visible satellite. The actual bright fringes sit at the
> half-integer-ish positions derived next, offset from these zeros by about
> half a fringe spacing.

### Side maxima (observable thickness fringes)

Between two zeros the phasor chain no longer closes into a symmetric,
cancelling polygon — it sweeps through only part of a full turn, so there is
always some nonzero leftover chord. That chord vanishes at both zeros
bounding the lobe and necessarily peaks somewhere in between: that peak is
the observable fringe.

**Where, exactly.** Differentiating $\sin^2(N\varphi/2)/\sin^2(\varphi/2)$
and setting the result to zero gives the exact extremum condition
$N\tan(\varphi/2) = \tan(N\varphi/2)$ — the classical diffraction-grating
equation for subsidiary maxima. The fringes of interest sit at small
$\varphi \propto 1/N$ for $N\gg1$, so $\sin(\varphi/2)\approx\varphi/2$ and,
writing $u \equiv N\varphi/2$, the interference function collapses onto the
**sinc² function**

$$
\left|F_\text{slab}\right|^2 \approx \left|f_\text{cell}\right|^2 N^2
\left(\frac{\sin u}{u}\right)^{\!2}
$$

— exactly the Fraunhofer diffraction pattern of a single slit of length $t$,
since $N$ discrete unit cells blur into a continuous slab in this limit. Its
subsidiary maxima solve $\tan u = u$, a transcendental equation with a fixed,
$N$-independent set of roots; the first nontrivial one is
$u_1 \approx 4.4934\,\text{rad} = 1.4303\,\pi$, giving

$$
\boxed{
\Delta q_n \approx \left(|m| + \tfrac{1}{2}\right)\frac{2\pi}{t},
\qquad m = \pm 1, \pm 2, \ldots
}
$$

as a **rough** guide (exact only in the limit $|m|\to\infty$), with the first
side maximum ($|m|=1$) actually at $u_1/\pi \times (2\pi/t) \approx
1.4303\,(2\pi/t)$ rather than $1.5\,(2\pi/t)$.

This 5% undershoot is **not** a finite-$N$ artefact that disappears for
thicker layers: $u_1$ is already the $N\to\infty$ limit, so the first fringe
sits at $1.4303\,(2\pi/t)$ for essentially any $N$ of practical interest —
numerically $1.451$ at $N=5$, $1.435$ at $N=10$, $1.4303$ by $N\approx100$,
unchanged beyond that. It happens because the half-integer rule assumes
$\sin^2(\varphi/2)$ is flat across one lobe and maximises the numerator
alone; since that envelope is smaller on the side of the lobe nearer the
Bragg peak, the true maximum is pulled toward that side. What *does* make the
half-integer rule accurate is going to **higher fringe order** $m$ at fixed
$N$, not larger $N$ at fixed order — the roots of $\tan u = u$ approach
$(m+\tfrac12)\pi$ as $m$ grows ($m=2$: $2.459$; $m=10$: $10.49$; $m=50$:
$50.498$, in units of $2\pi/t$). Use the boxed formula to locate roughly
where a higher-order fringe ($|m|\gtrsim 3$) should be; use $1.4303\,(2\pi/t)$
for the first fringe specifically if the exact position matters.

The intensity of the $m$-th side maximum relative to the Bragg peak follows
from the same sinc² approximation: at the extremum $u_m$, $\sin^2u_m =
u_m^2/(1+u_m^2) \to 1$ for the $u_m\gg1$ fringes, so
$|F_\text{slab}(u_m)|^2 \approx |f_\text{cell}|^2 N^2/u_m^2$, while the Bragg
peak itself is $|F_\text{slab}(0)|^2=N^2|f_\text{cell}|^2$. Approximating
$u_m\approx(|m|+\tfrac12)\pi$ gives

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

The displacement of a satellite on the detector relative to its parent Bragg
spot is set by how $\delta\mathbf{G} = \mathbf{G}_\text{sat} - \mathbf{G}_{hkl}$
rotates the scattered wavevector $\mathbf{k}_f = \mathbf{k}_i + \mathbf{G}$.
For small displacements

$$
\delta\mathbf{k}_f \approx \delta\mathbf{G}
- \left(\delta\mathbf{G}\cdot\hat{k}_f\right)\hat{k}_f
$$

(the component along $\hat{k}_f$ changes only the energy, not the direction).
The **in-plane** part of $\delta\mathbf{G}$ — i.e.\ the projection of
$\hat{n}_\text{lab}$ onto the detector plane — determines the pixel
displacement direction.

### Why flipping $\hat{n}$ alone does not flip the satellite side

Both $m=+1$ (at $+\Delta q_n$) and $m=-1$ (at $-\Delta q_n$) are always
enumerated.  The satellite energies $E^{(m)}$ depend on the actual
$\mathbf{G}_\text{sat}$ vectors, which are unchanged by relabelling $m$.
Thus flipping $\hat{n} \to -\hat{n}$ merely swaps the $m$-labels; it does
not move any spot to a new detector position.

The correct way to control which side the fringe appears on is to ensure the
stacking direction $\hat{n}_\text{crystal}$ points **from substrate toward
surface** (the growth direction).  For $c$-axis GaN use $[001]$ not $[00\bar 1]$.

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
7. **Intensity** — $I \propto |F|^2 \times LP(2\theta) \times S(E)$, using the
   Laue LP factor and synchrotron spectrum derived in
   [Theory §4](laue_theory.md#4-spot-intensity).

---

## References

**Textbooks**

- Warren, B. E. *X-Ray Diffraction.* Dover Publications, New York, 1990. (Unabridged reprint of the 1969 Addison-Wesley edition.) Chapter 3 derives the Laue interference function and the positions and intensities of subsidiary maxima between Bragg peaks — the canonical reference for §1.
- Als-Nielsen, J. & McMorrow, D. *Elements of Modern X-ray Physics*, 2nd ed. Wiley, Chichester, 2011. ISBN 978-0-470-97395-0. Chapter 3 derives the single-slab structure factor and the thin-film fringe spacing.
- Authier, A. *Dynamical Theory of X-Ray Diffraction.* Oxford University Press, Oxford, 2001. ISBN 978-0-19-855960-2. Chapter 1 covers the kinematical (Born) limit used throughout `simulate_laue_stack`.
- Guinier, A. *X-Ray Diffraction in Crystals, Imperfect Crystals, and Amorphous Bodies.* Dover Publications, New York, 1994. (Reprint of the 1963 W. H. Freeman edition.) A comprehensive account of the geometric-series structure factor and its relation to crystal size and shape.

**Superlattice satellites and multilayer X-ray diffraction**

- Segmüller, A. & Blakeslee, A. E. X-ray diffraction from one-dimensional superlattices in GaAs$_{1-x}$P$_x$ crystals. *J. Appl. Crystallogr.* **6**, 19–24 (1973). DOI: [10.1107/S0021889873008228](https://doi.org/10.1107/S0021889873008228). Seminal paper demonstrating superlattice satellites in compound semiconductors and relating satellite spacing to the bilayer period $\Lambda$.
- Bartels, W. J., Hornstra, J. & Lobeek, D. J. W. X-ray diffraction of multilayers and superlattices. *Acta Cryst.* **A42**, 539–545 (1986). DOI: [10.1107/S0108767386098768](https://doi.org/10.1107/S0108767386098768). Derives the kinematical structure factor for a periodic bilayer stack and the selection rules for superlattice satellites.
- Fullerton, E. E., Schuller, I. K., Vanderstraeten, H. & Bruynseraede, Y. Structural refinement of superlattices from x-ray diffraction. *Phys. Rev. B* **45**, 9292–9310 (1992). DOI: [10.1103/PhysRevB.45.9292](https://doi.org/10.1103/PhysRevB.45.9292). A complete kinematical model for fitting satellite intensities and extracting individual layer thicknesses and interface roughness.

**White-beam Laue diffraction and LaueTools**

- Robach, O., Micha, J.-S., Ulrich, O. & Gergaud, P. Full local elastic strain tensor from Laue microdiffraction. *J. Appl. Cryst.* **44**, 688–696 (2011). DOI: [10.1107/S0021889811003099](https://doi.org/10.1107/S0021889811003099). Describes the LaueTools framework (frame conventions, calibration parameters, indexation) on which `nrxrdct.laue` is built.
- Chung, J.-S. & Ice, G. E. Automated indexing for texture and strain measurement with broad-bandpass x-ray microbeams. *J. Appl. Phys.* **86**, 5249–5255 (1999). DOI: [10.1063/1.371507](https://doi.org/10.1063/1.371507). Introduces the orientation-matrix conventions (LT frame, `matstarlab`) used by LaueTools and this package.

**GaN / III-nitride thin films**

- Metzger, T. H. *et al.* X-ray diffraction study of InGaN/GaN superlattices on GaN/(0001) sapphire. *Phil. Mag. A* **77**, 1013–1025 (1998). DOI: [10.1080/01418619808221234](https://doi.org/10.1080/01418619808221234). Reports thickness fringes and superlattice satellites from InGaN/GaN multilayers grown along $[0001]$ — directly analogous to the structures modelled by `simulate_laue_stack`.
- Vickers, M. E. *et al.* Determination of InGaN layer thicknesses in InGaN/GaN quantum well structures by x-ray reflectometry and scattering. *J. Appl. Phys.* **94**, 1559–1566 (2003). DOI: [10.1063/1.1586996](https://doi.org/10.1063/1.1586996). Demonstrates extraction of quantum-well thicknesses from fringe spacing, providing experimental validation of the $2\pi/t$ fringe-period formula.
