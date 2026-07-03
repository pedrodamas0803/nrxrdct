# Darwin model for thick-layer Laue diffraction

This page describes the **Darwin primary extinction** correction implemented in
`simulate_laue_darwin` and explains when and why it is necessary compared to the
kinematical model used by `simulate_laue_stack`.

---

## 1. The problem with the kinematical model

In the kinematical (Born) approximation, the diffracted amplitude from a perfect
crystal slab of $N$ unit cells is simply proportional to $N$, so the intensity
scales as

$$
I_\text{kin} \propto \lvert F_\text{uc}\rvert^2 \, N^2.
$$

This is accurate for **thin** layers (epitaxial films, quantum wells) where each
photon interacts with the crystal at most once before leaving.  For **thick**
perfect-crystal regions — a substrate, a thick buffer layer, or any slab with
$N \gtrsim N_\text{ext}$ — the assumption breaks down:

- The diffracted beam begins to re-diffract back into the forward beam
  (*primary extinction*).
- The leading unit cells act as a partial mirror, shadowing the deeper ones.
- The intensity saturates rather than continuing to grow as $N^2$.

Ignoring this effect produces substrate peaks that are unrealistically strong
relative to the thin-film signals of interest.

---

## 2. The Darwin extinction length

Darwin (1914, 1922) showed that for a perfect plane-wave incident on a perfect
crystal, the *extinction length* $N_\text{ext}$ is the number of reflecting planes
at which the diffracted amplitude equals the incident amplitude.  For X-ray
diffraction it is

$$
\boxed{
N_\text{ext} = \frac{V_\text{uc}\,\sin\theta}{r_e\,\lambda\,\lvert F_\text{uc}\rvert\,d_{hkl}}
}
$$

where

| Symbol | Meaning | Typical value |
|---|---|---|
| $V_\text{uc}$ | unit-cell volume (Å³) | ~45 Å³ for GaN |
| $\sin\theta$ | sine of the Bragg angle | 0.1–0.9 |
| $r_e$ | classical electron radius | $2.818 \times 10^{-5}$ Å |
| $\lambda$ | X-ray wavelength (Å) | 0.5–2.5 Å |
| $\lvert F_\text{uc}\rvert$ | unit-cell structure factor amplitude | depends on reflection |
| $d_{hkl}$ | interplanar spacing (Å) | e.g. 2.59 Å for GaN 0002 |

For a strong reflection in a high-$Z$ material $N_\text{ext}$ can be as small as
a few hundred unit cells ($\sim 100$ nm); for a weak reflection or a light-element
material it can reach tens of thousands.

---

## 3. Primary extinction correction — Darwin factor

The effective number of coherently scattering unit cells under primary extinction is

$$
\boxed{
N_\text{eff} = N_\text{ext}\,\tanh\!\left(\frac{N}{N_\text{ext}}\right)
}
$$

This formula has two natural limits:

| Regime | Condition | Behaviour |
|---|---|---|
| Thin (kinematical) | $N \ll N_\text{ext}$ | $\tanh(N/N_\text{ext}) \approx N/N_\text{ext}$, so $N_\text{eff} \approx N$ |
| Thick (saturation) | $N \gg N_\text{ext}$ | $\tanh \to 1$, so $N_\text{eff} \to N_\text{ext}$ |

The intensity correction is therefore

$$
I_\text{Darwin} \propto \lvert F_\text{uc}\rvert^2\,N_\text{eff}^2
$$

For a thick substrate with $N = 10^6$ cells and $N_\text{ext} = 500$, the kinematical
model overestimates the intensity by a factor of $(10^6/500)^2 = 4\times 10^6$.
The Darwin correction reduces it to the physically correct saturation level.

!!! note "Primary vs. secondary extinction"
    *Primary extinction* (corrected here) arises within a single perfect-crystal
    domain — the diffracted beam re-enters the forward direction and interferes
    destructively.  *Secondary extinction* is a mosaic / grain-statistics effect
    that reduces intensity when many grains are simultaneously in the Bragg
    condition.  Only primary extinction is implemented in `simulate_laue_darwin`.

---

## 4. Absorption corrections — two-beam path and overlying layers

In addition to primary extinction, **photoelectric absorption** limits the
effective depth and attenuates the amplitude from each layer.  Three effects are
modelled.

### 4.1 Linear absorption coefficient

The Beer-Lambert attenuation length for a material with imaginary refractive
index $\beta$ at energy $E$ is

$$
t_\text{abs}(E) = \frac{1}{\mu(E)}, \qquad \mu = \frac{4\pi\beta}{\lambda}
$$

The corresponding maximum number of contributing unit cells is
$N_\text{abs} = t_\text{abs} / d$.

### 4.2 Two-beam effective depth

The incident beam and the diffracted beam both travel obliquely through each
layer.  The one-beam estimate uses only the incident angle:

$$
N_\text{abs}^\text{1-beam} = \frac{\cos\alpha_\text{in}}{\mu\,d}
$$

The **two-beam** correction used by the simulation accounts for both legs:

$$
\boxed{
N_\text{abs} = \frac{\cos\alpha_\text{in}\,\cos\alpha_\text{out}}
                     {\mu\,d\;(\cos\alpha_\text{in} + \cos\alpha_\text{out})}
}
$$

$$
\cos\alpha_\text{in}  = |\hat{n}\cdot\hat{x}|, \qquad
\cos\alpha_\text{out} = |\hat{n}\cdot\hat{k}_f|
$$

Because $\hat{k}_f$ is spot-specific, this correction is evaluated per spot.
For a symmetric geometry ($\alpha_\text{in} = \alpha_\text{out}$) the two-beam
result is exactly half the one-beam value.  For near-grazing exit the reduction
is much larger.

The combined Darwin + absorption limit becomes

$$
N_\text{eff}^\text{final} = \min\!\left(N_\text{eff}^\text{Darwin},\; N_\text{abs}\right)
$$

### 4.3 Overlying-layer attenuation

A photon scattered from a buffer layer must also pass through all **shallower
layers** (other buffer layers and the MQW block) on the exit path.  Each
overlying layer of thickness $t_j$ and attenuation coefficient $\mu_j$
contributes a transmission factor

$$
T_j = \exp\!\left[
  -\mu_j\, t_j \left(\frac{1}{\cos\alpha_\text{in}} + \frac{1}{\cos\alpha_\text{out}}\right)
\right]
$$

The amplitude from buffer layer $i$ is therefore scaled by

$$
T_\text{above}^{(i)} = T_\text{MQW} \times \prod_{j > i} T_j
$$

The product runs over all buffer layers shallower than $i$ and
$T_\text{MQW}$ is the transmission through the full MQW block.  The total
buffer structure factor becomes

$$
F_\text{buf}(\mathbf{G}) = \sum_i T_\text{above}^{(i)}\; F_i^{\text{uc}}\, N_\text{eff,i}\,
e^{\,i Q_n z_i}
$$

The MQW block sits at the top of the stack and receives no $T_\text{above}$
correction.

!!! note
    Repeating MQW layers have individual thicknesses always much
    smaller than $t_\text{abs}$, so no absorption depth cap is applied to them.
    The $T_\text{above}$ overlying-layer factor still modulates the amplitude
    from deeper buffer layers as described above.

---

## 5. Coherent multilayer sum

Like `simulate_laue_stack`, `simulate_laue_darwin` sums all layer amplitudes
**coherently**.  The coherence preserves superlattice satellites and thickness
fringes.  For a stack of buffer layers plus $N_\text{rep}$ repetitions of a
bilayer unit, the total diffracted amplitude is

$$
F_\text{total}(\mathbf{G}) = F_\text{buffer}(\mathbf{G})
+ e^{\,i Q_n z_\text{buf}}\, F_\text{unit}(\mathbf{G})\, S_\text{rep}(Q_n \Lambda)
$$

where

$$
F_\text{buffer}(\mathbf{G}) = \sum_j F_j^{\text{uc}} N_\text{eff,j}\, e^{\,i Q_n z_j}
$$

$$
F_\text{unit}(\mathbf{G}) = \sum_j F_j^{\text{uc}} N_\text{eff,j}\, e^{\,i Q_n z_j^\text{rel}}
$$

$$
S_\text{rep}(Q_n \Lambda) =
\frac{1 - e^{\,i N_\text{rep} Q_n \Lambda}}{1 - e^{\,i Q_n \Lambda}}
$$

Here $Q_n = \mathbf{G}\cdot\hat{n}$ is the component of the reciprocal-lattice vector
along the stacking direction $\hat{n}$, $z_j$ is the absolute depth of each layer,
$z_j^\text{rel}$ is the relative depth within one bilayer period, and $\Lambda$ is the
bilayer period.

The structure of the sum is identical to the kinematical case in
`simulate_laue_stack` — the **only difference** is that each layer's $N$ is replaced
by its Darwin-corrected $N_\text{eff}$.

---

## 6. Superlattice satellites in the Darwin model

Satellite peak positions are determined purely by geometry (the Laue condition),
not by the diffraction model.  `simulate_laue_darwin` therefore probes the same
satellite wavevectors as `simulate_laue_stack`:

$$
\mathbf{G}_\text{sat}^{(m)} = \mathbf{G}_{hkl}
+ \left(|m| + \tfrac{1}{2}\right)\operatorname{sgn}(m)\,
\frac{2\pi}{t}\,\hat{n}
$$

for both per-layer thickness fringes ($t = $ layer thickness) and superlattice
satellites ($t = \Lambda$, only when $N_\text{rep} > 1$).

The key difference from `simulate_laue_stack` is in the **amplitude** at each
satellite $\mathbf{G}_\text{sat}$: the Darwin-corrected $N_\text{eff}$ is
recomputed at the shifted wavevector (because $Q_n$ and $\sin\theta$ both change),
so the relative satellite / Bragg-peak intensity is automatically modified by the
extinction correction.

---

## 7. When to use `simulate_laue_darwin` vs. `simulate_laue_stack`

| Feature | `simulate_laue_stack` | `simulate_laue_darwin` |
|---|---|---|
| Thin epitaxial layers ($N \ll N_\text{ext}$) | ✓ accurate | ✓ reduces to kinematical |
| Thick substrate / buffer ($N \gg N_\text{ext}$) | ✗ over-predicts | ✓ saturates correctly |
| Superlattice satellites | ✓ | ✓ |
| Mosaic / strained crystals | ✓ (effective N broadening) | use kinematical |
| Computation speed | faster | slightly slower |

For a typical InGaN/GaN MQW on a GaN substrate, the QW and barrier layers are
always in the kinematical regime ($N \sim 10$–$200 \ll N_\text{ext} \sim 10^3$),
while the GaN substrate can be many micrometres thick.  In this situation
`simulate_laue_darwin` gives the most physically meaningful relative
intensities between the substrate Bragg peak and the MQW satellites.

Both functions accept the `structure_model` parameter (`'average'` by default,
`'coherent'` for the full inter-layer interference pattern).  See the
[Structure model](laue_layered_structures.md#3-structure-model)
section of the Layered Structures page for a full description of what each
mode computes and when to use each.

---

## 8. Example — InGaN/GaN MQW

```python
import xrayutilities as xu
import nrxrdct.laue as laue

GaN   = xu.materials.GaN
InGaN = xu.materials.InGaN(0.20)   # In₀.₂₀Ga₀.₈₀N  (Vegard's law)

U = laue.euler_to_U(0, 0, 0, sample_tilt_deg=40)
C = laue.nitride_elastic_constants('InN', x=0.20, end_material='GaN')

stack = laue.LayeredCrystal(n_hat=(0, 0, 1), n_rep=10)
stack.add_buffer_layer(GaN, U, thickness=5000.0,           # 500 nm buffer
                       d_spacing=laue.d_spacing_hkl(GaN, 0, 0, 2),
                       label='GaN buffer')
stack.add_pseudomorphic_layer(InGaN, U, thickness=30.0,    # 3 nm QW
                              a_substrate=GaN.lattice.a,
                              C13=C['C13'], C33=C['C33'],
                              label='InGaN QW')
stack.add_layer(GaN, U, thickness=100.0,                   # 10 nm barrier
                d_spacing=laue.d_spacing_hkl(GaN, 0, 0, 2),
                label='GaN barrier')

camera = laue.Camera()

spots = laue.simulate_laue_darwin(
    stack, camera,
    E_min_eV=5_000, E_max_eV=22_000,
    max_satellites=5,
    verbose=True,
)

laue.print_hkl_family(spots, 0, 0, 2, n=5)  # GaN 000ℓ family + MQW satellites
```

---

## 9. Depth-parallax correction and spot elongation

### 9.1 Physical origin

In **reflection geometry**, photons that diffract from depth $z$ below the
surface exit the crystal from a point that is **laterally displaced** relative
to the surface intercept of the incident beam.  The displacement along the
beam direction is

$$
\delta_\text{beam} = z \cdot \cos\alpha_\text{in},
\qquad \cos\alpha_\text{in} = |\hat{n}\cdot\hat{k}_i|
$$

For a 775 µm Si crystal at 40° incidence ($\cos\alpha_\text{in} \approx 0.65$),
substrate spots are displaced by ~500 µm along the beam direction relative to
surface spots — a shift of several detector pixels that is clearly visible as
**spot elongation** when integrating over a thick diffracting volume.

### 9.2 Enabling depth-corrected projection

Pass ``correct_depth=True`` to ``simulate_laue_darwin`` (or
``simulate_laue_stack``) to project each layer's spots from its
centre depth rather than the surface:

```python
spots = laue.simulate_laue_darwin(
    stack, camera,
    sigma_h_mrad=2.5,      # BM32 horizontal divergence
    sigma_v_mrad=0.3,      # BM32 vertical divergence
    correct_depth=True,    # substrate spots projected from ~387 µm depth
)
```

Each spot dict gains a ``'source_depth_mm'`` key recording the beam-path
depth used.  Without ``correct_depth``, all spots are projected from
``source_depth_mm = 0`` (the surface).

### 9.3 Visualising the elongation trail

:func:`~nrxrdct.laue.plot_depth_elongation` sweeps each spot's diffracting
depth from the top to the bottom of its contributing layer and projects the
position at each depth, producing an absorption-weighted *trail* that shows
exactly how much the spot moves across the detector:

```python
fig, ax = laue.plot_depth_elongation(
    spots, stack, camera,
    top_n=20,
    min_intensity=1e-4,
    n_steps_per_layer=12,
    space="detector",
    show_divergence=True,    # combined depth-parallax + beam-divergence ellipse
    divergence_nsigma=2.0,
    image=detector_image,    # optional: show real frame behind the trails
)
```

The figure contains three elements per spot:

| Element | What it shows |
|---|---|
| **Coloured trail** | Range of projected positions from layer top (opaque) to bottom (faint) |
| **Tick markers** (``\|``) | Exact trail endpoints |
| **Circle** | Where the simulation placed the spot (`spot['pix']`, at layer-centre depth when `correct_depth=True`) |
| **Ellipse** | Combined 2σ broadening from depth-parallax covariance + beam-divergence covariance (`cov_px`) |

The ellipse major axis automatically aligns with the dominant elongation
direction: depth-parallax (trail direction) for thick layers, beam divergence
for thin layers.

### 9.4 Quantifying simulation accuracy

:func:`~nrxrdct.laue.plot_pix_deviation` matches each simulated spot to the
nearest measured peak and plots the displacement norm $|\Delta\text{pix}|$
as a function of spot properties (energy, 2θ, χ, intensity):

```python
fig, axes, matched = laue.plot_pix_deviation(
    spots, peaklist,
    max_dist_px=15.0,
    properties=["E", "tth", "chi", "intensity"],
)
```

Systematic trends in the residuals diagnose calibration errors:

| Trend | Likely cause |
|---|---|
| $|\Delta\text{pix}|$ grows with $2\theta$ | Detector distance or tilt error |
| Offset correlated with energy | Incorrect ``correct_depth`` setting |
| Phase-dependent offset | Layer depth or orientation error |
| Random scatter $\lesssim 2$ px | Acceptable residual for BM32 geometry |

### 9.5 Image-based depth reconstruction

`depth_scan_image` works directly on the raw detector pixel array — **no peak
extraction required**.  For every simulated spot and every candidate depth *z*
the expected detector position is

$$p(z) = p_0 + z \cdot \frac{dp}{dz}$$

where the slope $dp/dz$ is obtained from two `camera.project` calls at
$z = 0$ and $z = \epsilon$.  The pixel intensity is then sampled at every
$(z, \text{spot})$ combination via a single batched bilinear-interpolation
call, yielding a **score matrix** of shape `(n_steps, n_valid_spots)`.

```python
res_img = depth_scan_image(
    spots, detector_image, camera, stack,
    n_steps=200, z_max_mm=0.8,
    min_intensity=0.02,
    score_weighted=True,   # weight by simulated spot intensity
)

fig, axes = plot_depth_scan_image(res_img, top_n_spots=30)
```

The two output quantities complement each other:

| Output | Shape | Interpretation |
|--------|-------|----------------|
| `score` | `(n_steps,)` | Sum of sampled intensities vs depth; peak = best-matching depth |
| `score_matrix` | `(n_steps, n_valid)` | Per-spot depth profile; narrow column = localised spot, broad column = depth-spread diffracting layer |

**Advantages over the peak-list method (`depth_scan_reconstruction`):**

* Works on elongated or merged spots that do not produce clean extracted peaks.
* No threshold on peak detection — faint features in the tails contribute.
* Naturally handles background gradients (they appear as a broad, featureless
  baseline in the score profile).

**When to prefer the peak-list method:**

* When the detector image contains strong parasitic signal (fluorescence, air
  scatter) that inflates the image score away from the true Bragg positions.
* When you need per-peak depth estimates rather than a global profile.

The two methods can be run together and their score profiles compared as a
cross-check.

---

## References

- **Darwin, C. G.** The theory of X-ray reflexion. *Philos. Mag.* **27**, 315–333 and 675–690 (1914).
- **Darwin, C. G.** The reflexion of X-rays from imperfect crystals. *Philos. Mag.* **43**, 800–829 (1922).
- **Zachariasen, W. H.** *Theory of X-Ray Diffraction in Crystals.* Dover, New York, 1945. (Chapter 3 — extinction in perfect crystals.)
- **Als-Nielsen, J. & McMorrow, D.** *Elements of Modern X-ray Physics*, 2nd ed.  Wiley, 2011. (Section 6.4 — Darwin width and primary extinction.)
- **Authier, A.** *Dynamical Theory of X-Ray Diffraction.* Oxford University Press, 2001. (Chapters 2–4 — full dynamical treatment of which the Darwin model is the plane-wave limiting case.)
