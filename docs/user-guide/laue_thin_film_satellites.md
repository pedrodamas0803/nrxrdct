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
7. **Intensity** — $I \propto |F|^2 \times LP(2\theta) \times S(E)$, where
   $LP$ is the Lorentz–polarisation factor and $S(E)$ is the synchrotron
   spectrum.
