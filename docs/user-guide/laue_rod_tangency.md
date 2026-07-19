# Rod tangency and per-pixel forward simulation

This page answers a specific question: **why do some reflections show
several distinct, resolvable superlattice satellite spots, while others show
only one blurred spot or an elongated streak?** The short answer is
*tangency* — how a reflection's superlattice rod happens to sit relative to
the Ewald sphere at your particular detector geometry — and this page
derives that, then documents the toolchain built around it:
`rod_tangency`, `simulate_spot_image`, `simulate_full_detector_image`, and a
few smaller bookkeeping additions (`LayeredCrystal.print_reflections`,
`harmonic_orders`).

This builds directly on
[Thin-Film Satellites](laue_thin_film_satellites.md), which derives the
satellite positions themselves (§1–3) and touches on detector displacement
direction (§4) — read that first if the rod/satellite construction itself
is unfamiliar. This page picks up from there and asks a question that page
doesn't: *given the rod exists, why does it look the way it does on my
detector?*

---

## 1. Refresher — the rod is a line in Q-space

A repeating block with period $\Lambda$ and $N_\text{rep}$ repetitions
produces satellites at

$$
\mathbf{Q}(t) = \mathbf{G}_{hkl} + t\,\hat{n}, \qquad t = m\,\frac{2\pi}{\Lambda}
\ \ (m = 0, \pm1, \pm2, \ldots)
$$

i.e. a discrete comb of points along a **line** through $\mathbf{G}_{hkl}$
in the growth direction $\hat{n}$ — "the rod." Each point on this line
satisfies the elastic (Laue) condition at its own photon energy:

$$
\lambda(t) = -\frac{4\pi\,\hat{k}_i \cdot \mathbf{Q}(t)}{|\mathbf{Q}(t)|^2},
\qquad E(t) = \frac{hc}{\lambda(t)}
$$

Energy and pixel position are **both** functions of the single parameter
$t$ — they co-vary together as you move along the rod. That coupling is the
whole story of this page.

---

## 2. The tangency Jacobian

Define $\mathbf{k}_f(t) = \hat{k}_i\,k(t) + \mathbf{Q}(t)$ (normalised),
projected onto the detector via the camera geometry to get a pixel
$\mathbf{p}(t) = (x_\text{cam}(t), y_\text{cam}(t))$. The quantity that
matters is the **Jacobian**

$$
\left.\frac{d\mathbf{p}}{dt}\right|_{t=0}
$$

— how far the pixel moves on the detector per unit reciprocal-space step
along the rod. This is exactly what `rod_tangency` estimates, by a
central finite difference: solve the elastic condition at
$\mathbf{Q}(\pm\delta)$, project both to pixels, and divide by $2\delta$.

> **A wrong way to compute this (and why it's wrong).** An earlier version
> of this tool computed $\partial Q_n/\partial(\text{pixel})$ at a *fixed*
> energy instead — i.e. "if my beam couldn't change energy at all, which
> pixel direction keeps $\mathbf{Q}\cdot\hat n$ unchanged?" That's a
> real, well-defined direction, but it isn't the rod: the rod's own
> parametrisation lets energy move too. The two directions can differ by
> tens of degrees. This was caught by cross-checking the predicted angle
> against `qspace_around_spot`'s own computed pixel positions (fitting a
> line through them) for two different reflections — a good reminder to
> verify a *geometric* quantity against an independently-computed one
> before trusting it, the same way `harmonic_hkls`/`I_satellites` were
> checked against reconstruction identities elsewhere on this page.

### Why "tangency"

If the rod happens to run nearly parallel to the local Ewald-sphere surface
at $\mathbf{G}_{hkl}$ (a near-tangency condition), a *tiny* change in photon
energy sweeps through a large arc of $\mathbf{k}_f$ directions — so
$|d\mathbf{p}/dt|$ is **large**: the rod maps to a long streak on the
detector. If the rod instead cuts across the Ewald sphere steeply, a
$t$-step needs comparatively little help from energy to stay on-shell, and
$|d\mathbf{p}/dt|$ is **small**: the whole comb of satellite orders stays
compressed into a tight cluster of pixels.

This is purely a function of $\hat n$, $\mathbf{G}_{hkl}$, $\hat k_i$, and
the camera geometry — nothing about intensity enters. Two reflections from
the *same* superlattice period can have wildly different $|d\mathbf p/dt|$
simply because they sit at different points on the Ewald sphere.

---

## 3. Why this controls how many satellite spots you actually *see*

The satellite comb is spaced by $\Delta t = 2\pi/\Lambda$ in $t$. Via the
Jacobian, consecutive orders land on the detector separated by
approximately

$$
\Delta p_\text{orders} \;\approx\; \left|\frac{d\mathbf p}{dt}\right|
\cdot \frac{2\pi}{\Lambda}
$$

Whether two adjacent orders show up as **separate, resolvable spots** or
**blur into one feature** depends on how this spacing compares to whatever
*smears* an individual order's own footprint on the detector:

* The intrinsic coherence width of the $S_\text{rep}$ peak itself
  ($\approx \Delta t / N_\text{rep}$ — narrower than the spacing by a
  factor of $N_\text{rep}$, so essentially never the limiting factor for a
  superlattice with more than a few repeats);
* Real instrumental broadening — beam divergence, detector point-spread,
  pixel binning — which is roughly a *fixed* number of pixels, largely
  independent of $|d\mathbf p/dt|$.

Because the instrumental smearing doesn't scale with the Jacobian while
$\Delta p_\text{orders}$ does, **a large $|d\mathbf p/dt|$ (near-tangency)
pushes the satellite orders apart faster than it blurs any individual
one** — so a reflection sitting near tangency is the one where you're most
likely to resolve several distinct satellite spots (a "beads on a string"
pattern along the streak direction). A reflection far from tangency
compresses the *same* physical set of orders into a pixel cluster smaller
than the instrumental resolution, so they overlap into what looks like a
single spot — even though, in reciprocal space, just as many orders are
present and contributing.

This is also the direct explanation for the streak-vs-blur confusion worked
through interactively while building this: with `dE_eV` wide enough to
integrate several periods (needed for correct destructive-interference
contrast — see §5), a near-tangent reflection's per-pixel image looked like
a single uniform stripe. Splitting the same data by satellite order
(`I_satellites`, §5) revealed the "beads" directly — each order individually
compact, just displaced along the tangency direction, with a real,
substantial intensity envelope decay between them (a factor of ~170× was
measured between adjacent orders in one worked example).

---

## 4. Diagnosing tangency: `rod_tangency` / `plot_rod_tangency`

```python
from nrxrdct.laue import rod_tangency, plot_rod_tangency

info = rod_tangency(stack, (1, 0, 5), layer='GaN buffer', camera=cam, max_satellites=3)
# {'hkl': (1, 0, 5), 'layer': 'GaN buffer', 'pix0': (643.8, 354.6),
#  'E0': 5252.2, 'streak_dir_px': array([0.71, 0.70]),
#  'perp_dir_px': array([-0.70, 0.71]), 'dpix_dalong': 259.8,
#  'on_detector': True, 'period': 54000.0,
#  'satellites': {-3: {...}, ..., 0: {...}, ..., 3: {...}}}

# layer='all' compares the *same* hkl across every layer in the stack —
# useful since a strained layer's G0 (hence pix0/E0/streak direction) can
# differ measurably from an unstrained one even for "the same" hkl label
fig, ax, infos = plot_rod_tangency(
    stack, (1, 0, 5), layer='all', camera=cam,
    max_satellites=3,        # mark each layer's own satellite-order positions
    image=measured_frame,    # optional — overlay on a real/simulated frame
    arrow_length_px=60,
)
# infos is always a list (one dict per plotted layer, `rod_tangency`'s own
# return format) — even for a single layer, so `infos[0]['dpix_dalong']`
# rather than `info['dpix_dalong']`. Layers whose hkl doesn't land
# on-detector are skipped with a printed note rather than raising.
```

* **`streak_dir_px`** — unit vector, the direction the rod actually traces
  on the detector (drawn as a coloured line, one colour per layer when
  comparing several). This is the physically meaningful output.
* **`perp_dir_px`** — a plain 90°-rotated reference axis (drawn dashed
  only in single-layer mode) for visual scale; the streak's *width* comes
  from off-rod structure-factor decay and instrumental broadening, not
  from anything this function derives.
* **`dpix_dalong`** (px / Å⁻¹) — the number to actually compare across
  candidate `hkl` *or* across layers. Large ⇒ elongated streak, satellite
  orders likely resolvable as separate spots (given fine enough
  instrumental resolution); small ⇒ compact spot, orders likely blended
  together.
* **`satellites`** (when `max_satellites > 0`) — each layer's own discrete
  comb of order positions along its streak, at that layer's own period
  (its own repeating block's period if it belongs to one, else its own
  thickness) — drawn as small dots along the streak line. Comparing this
  across layers shows directly whether two layers' satellite combs would
  overlap, interleave, or sit well apart on the detector.

This is pure geometry (no structure factor, no spectrum) — a cheap way to
screen candidate reflections, or compare layers, before committing to the
much more expensive tools below.

---

## 5. Per-pixel forward simulation: `simulate_spot_image`

Where `simulate_laue_stack` (and friends) work by *enumerating candidate
hkl* and checking each against the elastic condition and an intensity
threshold, `simulate_spot_image` inverts the direction: it starts from real
detector pixels around one already-indexed reflection and asks what `Q`
a photon of a given energy would need to have, for every pixel in a small
window:

$$
\mathbf{Q}(E) = k(E)\,(\hat k_f(\text{pixel}) - \hat k_i)
$$

```python
img = simulate_spot_image(
    stack, spots, (1, 0, 5), camera=cam, layer='GaN buffer',
    structure_model='average', darwin=True,
    isolate_layer=['Sapphire'],
    sigma_h_mrad=2.0, sigma_v_mrad=0.3,
    max_satellite_order=1,
)
plot_spot_image(img, exp_image=measured_frame)
```

A handful of subtleties turned out to matter a great deal in practice:

### 5.1 Anchoring to the exact reflection (`G_lab`)

Reconstructing $\mathbf{Q}$ from a *stored* pixel position via
`Camera.pixel_to_kf` carries a tiny (~$10^{-5}$ Å⁻¹) round-trip precision
error — utterly negligible geometrically, but every buried layer's
coherent phase is $e^{i Q_n z_0}$ with $z_0$ up to hundreds of thousands of
Å for a deep substrate, so $\Delta Q_n \cdot z_0$ can be many radians. The
fix: anchor to the **exact** $\mathbf{Q}$ (`spots[i]['G_lab']`, the value
`simulate_laue_stack` itself used — not recomputed from
`layer.crystal.Q(h,k,l)`, which silently ignores any pseudomorphic strain
correction) and apply the same small correction across the sampled window.

### 5.2 Deep buffer layers: `isolate_layer`

Even with exact anchoring, a coherent sum across a stack that includes a
deep substrate is often **numerically unresolvable**: the cross-term phase
between two widely-separated layers oscillates faster than any realistic
pixel/energy sampling can track, so the image looks flat/featureless not
because there's no signal, but because the phase is effectively
randomised at every sampled point except the exact anchor. `isolate_layer`
builds a reduced stand-in stack (reusing the same `Layer` objects, so
`absorption_limit`/`d_spacing`/`n_cells` are preserved exactly) that drops
the offending cross-terms:

* `isolate_layer=True` — keep only the reflecting layer (or its own
  repeating block).
* `isolate_layer=['Sapphire', ...]` — exclude specific buffer layers by
  label, keeping everything else (including cross-terms between layers
  that are physically close together and may genuinely interfere
  constructively — often what actually produces a real, sharp measured
  peak).

Works with both `darwin=False` and `darwin=True`.

### 5.3 `dE_eV`/`n_energy` and destructive interference

A repeating block's $S_\text{rep}$ term oscillates in energy at a fixed
pixel — real destructive interference, not noise. A `dE_eV` window
narrower than a few oscillation periods *underestimates* the true
peak-to-background contrast, because it only samples part of one
rise/fall rather than full periods including their minima (measured:
roughly 2× more dynamic range going from a 150 eV to a 500 eV half-width
before plateauing, in one worked case). `pin_satellites=True` (default)
separately guarantees the exact fringe-peak energies are inserted into the
sampling grid regardless of `n_energy` — fixing "missing a peak entirely,"
which is a different failure mode from "window too narrow to judge
contrast." Both matter; neither substitutes for the other.

### 5.4 `I_satellites` / `max_satellite_order`

Summing every satellite order that falls inside a (necessarily wide)
`dE_eV` blends them into one flat-looking total — most pixels in the
window end up integrating over the *same* few dominant orders rather than
each showing "its own." `I_satellites` (populated when there's exactly one
relevant repeating block) reports each order's own contribution
separately; `max_satellite_order` caps how many actually get summed into
`I`/`I_harmonics` in the first place (e.g. `0` keeps only the main peak —
measured ~12× better dynamic range than an unlimited sum in one case).

### 5.5 Beam divergence (`sigma_h_mrad`, `sigma_v_mrad`, `n_div`)

With zero divergence, a Bragg reflection is a true delta function in
angle — there is nothing broader than that to resolve, and an isolated
single layer's own image can look flat even after fixing everything in
§5.1–§5.4, simply because a delta-function beam implies a delta-function
response. A real measured spot's finite size mostly comes from convolving
that intrinsic response with the source's actual angular divergence
(same convention as `beam_divergence_ellipses`: typical BM32/ESRF values
are 2–3 mrad horizontal, 0.2–0.5 mrad vertical).

---

## 6. Whole-frame version: `simulate_full_detector_image`

The pixel-space sibling of `simulate_spot_image` with **no** `hkl`
dependence at all — every (binned) pixel sweeps the full white-beam
window and integrates directly, with no candidate enumeration, no
`f2_thresh`, no satellite bookkeeping:

```python
img = simulate_full_detector_image(
    stack, cam, bin_px=8, exclude_layers=['Sapphire'],
    structure_model='average',
)
plt.imshow(np.log1p(img['I']),
           extent=[img['x0'][0], img['x0'][-1], img['y0'][-1], img['y0'][0]])
```

There is no anchoring correction here (unlike `simulate_spot_image`,
nothing external is being matched — every pixel's $\mathbf{Q}(E)$ is
simply computed fresh) and no `pin_satellites` equivalent (there is no
known reflection to derive an exact peak energy from), so a narrow Bragg
peak can be aliased away entirely between energy samples. Treat this as an
overview/screening tool — a way to see the broad intensity landscape and
spot candidate regions — not a substitute for `simulate_spot_image` once
you know where to look.

---

## 7. Bookkeeping: `print_reflections` and `harmonic_orders`

**`LayeredCrystal.print_reflections(spots, layer=None, *, satellite_order=None, ...)`**
prints what's actually in a `spots` list for a layer — hkl, satellite
order, energy, intensity, pixel — sorted by intensity. `simulate_spot_image`
raises a `ValueError` naming the exact hkl/layer/`satellite_order`
combination it couldn't find (and, when something with that hkl/layer
exists at a *different* order, says so directly) — this is the fast way to
check what's available before hitting that error.

**`harmonic_orders`** (parallel to `harmonic_hkls` on every merged spot)
labels each coincident-pixel entry with its integer order relative to the
group's own GCD-reduced primitive direction, or `None` if it's a genuine
accidental (non-integer-ratio) overlap rather than a true harmonic.
Non-integer ratios among `harmonic_hkls` are *expected* — the merge in
`_merge_or_append_spot` is keyed on pixel coincidence alone, by design,
since an undiscriminating detector pixel sums whatever lands on it
regardless of *why* (real harmonic or accidental overlap are both real
physics). The order is derived from the whole group's primitive, not
whichever reflection happened to be enumerated first — comparing pairwise
against a fixed first-seen reference mislabels groups where the
first-enumerated hkl is itself a higher harmonic (e.g. `(-10,4,2)`
enumerated before `(-5,2,1)`, its true 1× fundamental).

---

## 8. Practical recipes

**Screen candidate reflections for tangency before simulating anything:**

```python
for hkl in candidate_hkls:
    info = rod_tangency(stack, hkl, layer=layer, camera=cam)
    print(hkl, info['dpix_dalong'])
# pick the smallest dpix_dalong for a compact-spot comparison,
# the largest for a streak/satellite-rich comparison
```

**Compare where the same hkl lands for every layer in the stack (e.g. to
see whether a strained layer's satellite comb overlaps an unstrained
one's):**

```python
fig, ax, infos = plot_rod_tangency(
    stack, hkl, layer='all', camera=cam, max_satellites=3,
)
for info in infos:
    print(info['layer'], info['pix0'], info['dpix_dalong'])
```

**Check what's actually indexed before calling `simulate_spot_image`:**

```python
stack.print_reflections(spots, layer='GaN buffer', satellite_order=0)
```

**Image one known reflection, robust to a deep substrate:**

```python
img = simulate_spot_image(
    stack, spots, hkl, camera=cam, layer=layer,
    isolate_layer=['Sapphire'], structure_model='average',
    dE_eV=300, pin_satellites=True, max_satellite_order=1,
)
```

**See which satellite order dominates where:**

```python
for m, arr in img['I_satellites'][1].items():
    print(m, arr.max())
```

---

## 9. Summary of gotchas

| Symptom | Likely cause | Fix |
|---|---|---|
| Image is flat/featureless everywhere | Deep-buffer coherent phase numerically unresolvable | `isolate_layer` |
| Image is flat even after isolating the layer | Zero beam divergence ⇒ true delta-function response | `sigma_h_mrad`/`sigma_v_mrad` |
| Peak intensity looks lower than expected | `dE_eV` too narrow — misses destructive-interference minima | widen `dE_eV` (span several periods) |
| Peak position is right but intensity is ~0 | Narrow `S_rep` resonance missed by uniform energy sampling | `pin_satellites=True` (default) |
| One uniform streak instead of distinct spots | Summing many satellite orders together | `I_satellites`, `max_satellite_order` |
| "Reflection doesn't exist" but you can see it | Wrong `satellite_order` requested | `stack.print_reflections(...)`, then pass the right `satellite_order` |
| Non-integer `n` in a harmonic group | Expected — pixel-coincidence merge includes accidental overlaps | check `harmonic_orders`, `None` means accidental |
| Predicted streak angle doesn't match a real streak | Confirm you're on the current `rod_tangency` (uses `d(pixel)/d(along)`, not a fixed-energy gradient) | cross-check against `qspace_around_spot`'s own pixel positions if unsure |
