# X-ray diffraction and white-beam Laue theory

This page provides the theoretical foundations for X-ray diffraction and
white-beam Laue diffraction as implemented in `nrxrdct.laue`.  It covers
kinematical scattering, the Laue condition, spot intensities, coordinate
conventions, and pattern indexing.

---

## 1. Kinematical X-ray scattering

### 1.1 Scattering from a single atom

An X-ray photon with wavevector $\mathbf{k}_i = (2\pi/\lambda)\hat{k}_i$
scatters elastically into direction $\hat{k}_f$.  The momentum transfer is

$$
\mathbf{Q} = \mathbf{k}_f - \mathbf{k}_i,
\qquad
\lvert\mathbf{Q}\rvert = \frac{4\pi\sin\theta}{\lambda}
$$

where $\theta$ is the half-scattering angle (Bragg angle).
The amplitude scattered by atom $j$ at position $\mathbf{r}_j$ is

$$
A_j(\mathbf{Q}) = f_j(\mathbf{Q})\,e^{\,i\mathbf{Q}\cdot\mathbf{r}_j}
$$

where $f_j$ is the **atomic scattering factor** (form factor).  For X-rays
near an absorption edge the form factor has anomalous (dispersion) corrections:

$$
f_j(Q, E) = f_j^0(Q) + f_j'(E) + i\,f_j''(E)
$$

$f^0$ is the Thomson scattering term (Fourier transform of the electron
density); $f'$ and $f''$ are the real and imaginary anomalous corrections
tabulated by Henke *et al.*

### 1.2 Structure factor of a unit cell

The total amplitude from one unit cell containing atoms at fractional
coordinates $(x_j, y_j, z_j)$ is

$$
F_{hkl} = \sum_j f_j\,
e^{\,2\pi i(h x_j + k y_j + \ell z_j)}
$$

$F_{hkl}$ is complex; the observable intensity is proportional to
$\lvert F_{hkl}\rvert^2$.  Systematic absences arise when translational
symmetry forces $F_{hkl} = 0$ for certain $(h,k,\ell)$ combinations
(e.g.\ BCC: $h+k+\ell$ odd; FCC: mixed parity).

### 1.3 Bragg's law

Constructive interference from parallel lattice planes with spacing $d_{hkl}$
requires

$$
\boxed{2\,d_{hkl}\sin\theta = n\lambda}
$$

In vector form: $\mathbf{Q} = \mathbf{G}_{hkl}$, where $\mathbf{G}_{hkl}$
is a reciprocal-lattice vector of magnitude $2\pi/d_{hkl}$.

---

## 2. Reciprocal space and the Ewald sphere

### 2.1 Reciprocal lattice

Given direct-lattice basis vectors $\mathbf{a}, \mathbf{b}, \mathbf{c}$, the
reciprocal-lattice basis vectors are

$$
\mathbf{a}^* = \frac{2\pi\,\mathbf{b}\times\mathbf{c}}{V}, \quad
\mathbf{b}^* = \frac{2\pi\,\mathbf{c}\times\mathbf{a}}{V}, \quad
\mathbf{c}^* = \frac{2\pi\,\mathbf{a}\times\mathbf{b}}{V}
$$

where $V = \mathbf{a}\cdot(\mathbf{b}\times\mathbf{c})$ is the unit-cell
volume.  Every reciprocal-lattice vector is

$$
\mathbf{G}_{hkl} = h\,\mathbf{a}^* + k\,\mathbf{b}^* + \ell\,\mathbf{c}^*
$$

with $d_{hkl} = 2\pi/\lvert\mathbf{G}_{hkl}\rvert$.

### 2.2 Ewald sphere construction

The Ewald sphere has radius $\lvert\mathbf{k}_i\rvert = 2\pi/\lambda$ and is
centred at $-\hat{k}_i \cdot (2\pi/\lambda)$ (the tail of $\mathbf{k}_i$).
A reciprocal-lattice point $\mathbf{G}_{hkl}$ diffracts **if and only if**
it lies on the Ewald sphere surface:

$$
\lvert\mathbf{k}_i + \mathbf{G}_{hkl}\rvert = \lvert\mathbf{k}_i\rvert
\;\Longrightarrow\;
\lvert\mathbf{G}_{hkl}\rvert^2 + 2\,\mathbf{k}_i\cdot\mathbf{G}_{hkl} = 0
$$

This is the **Laue condition** in vector form.

---

## 3. White-beam Laue diffraction

### 3.1 Monochromatic vs. white-beam

| Mode | Fixed | Varied | Bragg condition met by |
|---|---|---|---|
| Monochromatic | $\lambda$ | crystal $\theta$ (rotation) | rotating sample to bring $\mathbf{G}$ onto Ewald sphere |
| White-beam Laue | crystal orientation | $\lambda \in [\lambda_\text{lo}, \lambda_\text{hi}]$ | wavelength self-selects for each $\mathbf{G}$ |

In white-beam geometry the Ewald sphere sweeps a **shell** in reciprocal
space: all $\mathbf{G}_{hkl}$ whose magnitude falls in the range

$$
\frac{4\pi\sin\theta_\text{min}}{\lambda_\text{hi}}
\;\leq\; \lvert\mathbf{G}_{hkl}\rvert \;\leq\;
\frac{4\pi\sin\theta_\text{max}}{\lambda_\text{lo}}
$$

simultaneously diffract.  A single exposure can record dozens to hundreds of
spots from an unstrained single crystal without any sample rotation.

### 3.2 The Laue condition and wavelength selection

From the vector Laue condition, the wavelength (and photon energy) at which
reflection $hkl$ is excited is

$$
\lambda_{hkl} = \frac{-4\pi\,(\hat{k}_i \cdot \mathbf{G}_{hkl})}
                     {\lvert\mathbf{G}_{hkl}\rvert^2},
\qquad
E_{hkl} = \frac{hc}{\lambda_{hkl}}
$$

The dot product $\hat{k}_i \cdot \mathbf{G}_{hkl}$ must be **negative**
(backscattering / reflection geometry) for $\lambda > 0$.  Reflections with
$\hat{k}_i \cdot \mathbf{G} \geq 0$ cannot diffract in reflection geometry.

### 3.3 Scattered beam direction

The scattered wavevector is

$$
\mathbf{k}_f = \mathbf{k}_i + \mathbf{G}_{hkl},
\qquad
\hat{k}_f = \frac{\mathbf{k}_f}{\lvert\mathbf{k}_f\rvert}
$$

The scattering angle $2\theta$ is $\arccos(\hat{k}_i \cdot \hat{k}_f)$.

---

## 4. Spot intensity

The measured intensity of a Laue spot is

$$
I_{hkl} = \lvert F_{hkl}\rvert^2 \cdot LP(2\theta) \cdot S(E_{hkl})
$$

### 4.1 Structure factor

$\lvert F_{hkl}\rvert^2$ is evaluated at the reflection energy $E_{hkl}$
using the energy-dependent anomalous form factors $f'(E)$, $f''(E)$ from the
Henke tables.  *xrayutilities* handles this automatically via
`crystal.StructureFactor(G, en=E)`.

### 4.2 Lorentz–polarisation factor

For an **unpolarised** beam the Lorentz–polarisation (LP) factor is

$$
LP(2\theta) = \frac{1 + \cos^2 2\theta}{2\sin^2\theta\cos\theta}
$$

This accounts for:

- **Lorentz factor** $1/\sin^2\theta\cos\theta$ — the time a reciprocal-lattice
  point spends intersecting the Ewald sphere (here it is absorbed into the
  angular width of the white-beam window);
- **Polarisation factor** $(1+\cos^2 2\theta)/2$ — for an unpolarised incident
  beam the scattered intensity is reduced by the projection of the polarisation
  onto the scattering plane.

### 4.3 Synchrotron source spectrum

The effective flux weight $S(E)$ applied to each Laue spot is the product of
the source spectrum and the KB mirror transmission:

$$
S_\text{eff}(E) = S_\text{source}(E) \times R_\text{KB}(E)
$$

Three source types are implemented in `spectrum_bm` / `spectrum_undulator`.

---

#### 4.3.1 Bending magnet

**Physical origin.**  An electron moving on a curved path (magnetic field $B$,
beam energy $E_e$) radiates synchrotron light over a broad spectrum.
The characteristic frequency is set by the **critical energy**

$$
\boxed{
E_c = 0.665\;B\,[\text{T}]\;\left(E_e\,[\text{GeV}]\right)^2\;\text{keV}
}
$$

Half the total radiated power lies below $E_c$ and half above.

| Facility | $E_e$ (GeV) | $B$ (T) | $E_c$ (keV) |
|---|---|---|---|
| ESRF (6 GeV)    | 6.04 | 0.86 | 20.0 |
| APS (7 GeV)     | 7.00 | 0.60 | 19.5 |
| SOLEIL (2.75 GeV)| 2.75 | 1.72 | 8.6 |
| Diamond (3 GeV) | 3.00 | 1.40 | 8.4 |

**On-axis spectral flux** (Kim 1989):

$$
\boxed{
S_\text{BM}(E) = 2N\left(\frac{E}{E_c}\right)^{\!2}
K_{2/3}^2\!\left(\frac{E}{2E_c}\right)
}
$$

where $K_{2/3}$ is the **modified Bessel function of the second kind** of order
$\frac{2}{3}$ (`scipy.special.kv(2/3, x)`), and $N = 1$ for a single bending
magnet pole.

**Asymptotic behaviour.**  The Bessel function has known limits:

- $E \ll E_c$: $K_{2/3}(x) \to \Gamma(2/3)\,(x/2)^{-2/3}/\sqrt{\pi}$ as $x \to 0$,
  so $S_\text{BM}(E) \propto E^{2/3}$ — a gentle power-law rise.
- $E \gg E_c$: $K_{2/3}(x) \to \sqrt{\pi/2x}\,e^{-x}$ as $x \to \infty$,
  so $S_\text{BM}(E) \propto E^{3/2}\,e^{-E/E_c}$ — an exponential cutoff.
- **Peak** at $E \approx 0.83\,E_c$.

```python
from scipy.special import kv
import numpy as np

def spectrum_bm(E_eV, Ec_eV=20_000, N=1):
    y = E_eV / Ec_eV
    return 2 * N * y**2 * kv(2/3, y/2)**2
```

---

#### 4.3.2 Wiggler

A wiggler is an insertion device with $2N_\text{poles}$ alternating magnetic
poles.  Each pole produces a bending-magnet-like arc, and the contributions add
**incoherently** (the arcs are too short and widely spaced for interference).
The spectrum is therefore the same shape as a bending magnet but with a flux
multiplied by the number of poles:

$$
S_\text{wiggler}(E) = 2N_\text{poles}\left(\frac{E}{E_c}\right)^{\!2}
K_{2/3}^2\!\left(\frac{E}{2E_c}\right)
$$

The critical energy uses the **peak** magnetic field of the wiggler, which is
typically stronger than the ring bending magnets, shifting the spectrum to
higher energies.  `spectrum_bm(E, Ec_eV, N=N_poles)` covers this by passing
`N = N_poles` (the factor of 2 is absorbed into the formula) and an
appropriately large `Ec_eV`.

---

#### 4.3.3 Undulator

**Physical origin.**  A planar undulator has $N_u$ identical alternating-pole
periods of length $\lambda_u$.  The electron oscillates transversely, and the
radiation from each period interferes **constructively** at specific resonant
energies.  The result is a spectrum of narrow **harmonic peaks** rather than a
broad continuum.

**Deflection parameter:**

$$
K = \frac{eB_0\lambda_u}{2\pi m_e c} \approx 0.934\;B_0\,[\text{T}]\;\lambda_u\,[\text{cm}]
$$

**Fundamental energy** (on-axis, in the electron rest frame boosted to the lab):

$$
E_1 = \frac{2\gamma^2 h c / \lambda_u}{1 + K^2/2}
$$

where $\gamma = E_e / m_e c^2$ is the Lorentz factor.  For typical undulators
$E_1$ is tunable between $\sim 5$ and $20$ keV by adjusting $B_0$ (the magnet
gap).

**Harmonic structure.**  On-axis, only **odd** harmonics $n = 1, 3, 5, \ldots$
are emitted.  The $n$-th harmonic appears at energy $E_n = n E_1$.  The
on-axis intensity of the $n$-th harmonic scales as $1/n$ (for a planar
undulator in the far-field limit).

**Implemented model** — Gaussian harmonic approximation:

$$
\boxed{
S_\text{und}(E) = \sum_{\substack{n=1,3,5,\ldots \\ n \leq n_\text{max}}}
\frac{1}{n}\exp\!\left[
-\frac{(E - nE_1)^2}{2\,(nE_1\,\sigma_\text{rel})^2}
\right]
}
$$

where $\sigma_\text{rel}$ is the **relative harmonic width** (dimensionless),
which is set by the angular acceptance of the downstream optics and typically
lies in the range 0.01–0.05.  The Gaussian shape is a valid approximation when
the angular divergence of the beam dominates the natural line width $\sim 1/N_u$.

| Parameter | Default | Meaning |
|---|---|---|
| `E1_eV` | 17 000 eV | Fundamental energy |
| `n_harm` | 7 | Number of odd harmonics computed (up to $n = 13$) |
| `sig_rel` | 0.015 | Relative harmonic width $\sigma_\text{rel}$ |

```python
def spectrum_undulator(E_eV, E1_eV=17_000, n_harm=7, sig_rel=0.015):
    s = 0.0
    for n in range(1, 2*n_harm, 2):   # n = 1, 3, 5, ..., 2*n_harm-1
        En = n * E1_eV
        s += (1.0/n) * np.exp(-0.5 * ((E_eV - En) / (En * sig_rel))**2)
    return s
```

---

### 4.4 KB mirror reflectivity

Kirkpatrick–Baez (KB) mirrors focus the beam by grazing-incidence total
external reflection.  They also act as a high-energy bandpass filter: photons
above a **hard-edge energy** $E_\text{cut}$ are not reflected.

#### 4.4.1 Complex refractive index

For X-rays, the refractive index of a material is

$$
n = 1 - \delta + i\beta
\qquad
\delta, \beta \ll 1
$$

where $\delta$ causes refraction and $\beta$ causes absorption.  Both depend on
energy through the atomic scattering factors:

$$
\delta = \frac{r_e \lambda^2}{2\pi} \sum_j \rho_j\,f_j^0(0)
\qquad
\beta = \frac{r_e \lambda^2}{2\pi} \sum_j \rho_j\,f_j''(E)
$$

where $\rho_j$ is the number density of atom $j$ and $r_e$ is the classical
electron radius.  At hard X-ray energies $\delta \propto E^{-2}$ and
$\beta \propto E^{-3}$ to $E^{-4}$ (between edges).

The **critical angle for total external reflection** is

$$
\theta_c(E) = \sqrt{2\delta} \approx \sqrt{\frac{r_e \lambda^2 \rho Z}{\pi}}
\;\propto\; E^{-1}
$$

Photons incident at $\theta < \theta_c$ are totally reflected
($R \approx 1$); above $\theta_c$ the reflectivity drops sharply.  Because
$\theta_c \propto E^{-1}$, at a fixed grazing angle $\theta_m$ the mirror
transmits only energies $E < E_\text{cut}$, where

$$
E_\text{cut} \approx \frac{hc}{\lambda_\text{cut}},
\qquad
\lambda_\text{cut} = \frac{\pi}{\rho Z r_e} \cdot \frac{2\pi}{\theta_m^2}
$$

#### 4.4.2 Fresnel reflectivity (s-polarisation)

The exact reflectivity at a flat interface for s-polarised X-rays is

$$
r_s = \frac{\sin\theta - \sqrt{n^2 - \cos^2\theta}}
           {\sin\theta + \sqrt{n^2 - \cos^2\theta}},
\qquad
R_\text{smooth} = |r_s|^2
$$

The square root is complex (the transmitted field decays exponentially below
$\theta_c$).  In the limit $\delta, \beta \to 0$, $R_\text{smooth} \to 0$ for
$\theta > \theta_c$ (Snell's law).

#### 4.4.3 Névot–Croce roughness correction

Real mirror surfaces have an RMS roughness $\sigma$ (typically 2–5 Å for
polished optics).  Roughness damps the specular reflectivity via the
Névot–Croce factor:

$$
R_\text{rough} = R_\text{smooth} \cdot \exp\!\left[-(2k\sin\theta\,\sigma)^2\right]
$$

where $k = 2\pi/\lambda$ is the X-ray wavenumber.  At high energies (small
$\lambda$) $k$ is large, so the roughness suppression is stronger — an
additional reason why flux drops above $E_\text{cut}$.

#### 4.4.4 Two-mirror KB system

A KB system uses two independent mirrors (horizontal + vertical focusing).
Assuming both mirrors have identical coatings and grazing angles, the total
reflectivity is

$$
R_\text{KB}(E) = R_\text{single}(E)^{n_\text{mirrors}}
$$

For $n_\text{mirrors} = 2$ (the standard BM32 setup) this squares the single-mirror
curve, sharpening the cutoff.

```python
# BM32 / ESRF defaults
BM32_KB = dict(
    material        = 'Rh',       # Rhodium coating
    grazing_angle_mrad = 2.5,     # 2.5 mrad ≈ 0.143°
    n_mirrors       = 2,          # HFM + VFM
    roughness_ang   = 3.0,        # 3 Å RMS roughness
)
```

The Rh $L$-edge is at $\sim 3$ keV (outside the useful range) and the $K$-edge
at $\sim 23$ keV introduces a small jump in reflectivity near the cutoff.

#### 4.4.5 Effective spectral weight

Combining source and mirrors:

$$
\boxed{
S_\text{eff}(E) = S_\text{source}(E) \times R_\text{KB}(E)
}
$$

For BM32 with bending-magnet source ($E_c = 20$ keV), Rh mirrors at 2.5 mrad,
and a 3 Å roughness, the effective bandpass is approximately **5–22 keV**:

- Below 5 keV: detector sensitivity drops and the mirror reflectivity is high
  but the source flux is low.
- Above ~22 keV: the Rh mirror cutoff ($\theta_c \approx \theta_m$ at ~22 keV
  for 2.5 mrad) and the exponential tail of the BM spectrum jointly suppress
  the flux.

| Contribution | Shape | Controls |
|---|---|---|
| $S_\text{BM}(E)$ | broad, peaks at $0.83\,E_c$ | $E_c$ (ring energy + $B$) |
| $R_\text{KB}(E)$ | flat below $E_\text{cut}$, hard edge above | mirror material, $\theta_m$ |
| $R_\text{rough}$ | gentle high-energy damping | surface roughness $\sigma$ |
| $S_\text{eff}(E)$ | effective bandpass | all of the above |

---

## 5. Coordinate frames and orientation matrix

### 5.1 LaueTools lab frame (LT frame)

`nrxrdct.laue` uses the **LaueTools LT frame** throughout:

| Axis | Direction |
|---|---|
| $x$ | along the incident beam ($+x = $ beam direction) |
| $z$ | vertical, pointing up |
| $y$ | $y = z \times x$ (horizontal, pointing away from the ring) |

The sample surface is typically tilted $\sim 40°$ from the beam so that the
surface normal lies $\sim 50°$ from $+x$.  The area detector is placed near
$z$ (i.e.\ $2\theta_\text{centre} \approx 90°$) in top-reflection geometry.

### 5.2 Crystal frame and orientation matrix $U$

A reciprocal-lattice vector expressed in the crystal frame is

$$
\mathbf{G}_\text{crystal} = h\,\mathbf{a}^* + k\,\mathbf{b}^* + \ell\,\mathbf{c}^*
$$

The orientation matrix $U$ (a $3\times3$ rotation) maps it to the lab frame:

$$
\mathbf{G}_\text{lab} = U\,\mathbf{G}_\text{crystal}
$$

$U$ is obtained either from Laue pattern indexing or by specifying Bunge ZXZ
Euler angles $(\varphi_1, \Phi, \varphi_2)$:

$$
U = R_z(\varphi_1)\,R_x(\Phi)\,R_z(\varphi_2)
$$

The columns of $U$ are the crystal basis vectors expressed in lab coordinates.

### 5.3 `matstarlab` convention

LaueTools stores orientations as the **$3\times3$ matrix of reciprocal basis
vectors in lab coordinates**, flattened to a 9-element array in the order
$[a^*_x, a^*_y, a^*_z, b^*_x, \ldots, c^*_z]$.  The conversion utilities
`euler_to_U`, `U_from_matstarlab`, and `decompose_matstarlab` handle
interconversion.

---

## 6. Detector geometry

### 6.1 Projection onto a flat detector

The detector is described by:

| Parameter | Symbol | Description |
|---|---|---|
| `DET_DIST_MM` | $D$ | sample-to-detector-centre distance (mm) |
| `TTH_CENTER_DEG` | $2\theta_0$ | $2\theta$ angle of detector centre |
| `NU_DEG` | $\nu$ | out-of-plane (elevation) tilt of detector centre |
| `CHI_DEG` | $\chi_D$ | in-plane rotation of detector about its own normal |
| `PIXEL_SIZE_MM` | $p$ | pixel pitch (mm) |

For each scattered beam $\hat{k}_f$, the ray is intersected with the detector
plane (defined by its centre and normal) to give a pixel coordinate
$(x_\text{cam}, y_\text{cam})$ in the LaueTools convention.

### 6.2 Angular coordinates

Two angular coordinates describe the position of a spot:

$$
2\theta = \arccos\!\left(\hat{k}_{f,x}\right),
\qquad
\chi = \arctan2\!\left(\hat{k}_{f,y},\, \hat{k}_{f,z}\right)
$$

$\chi = 0$ for spots in the vertical plane; $\chi$ increases counter-clockwise
when viewed from the detector side.

---

## 7. Laue pattern indexing

### 7.1 The gnomonic projection

Spots in a Laue pattern lie on **zones** — great circles on the unit sphere
of $\hat{k}_f$ directions.  All reflections $hkl$ with the same zone axis
$[uvw]$ (i.e.\ $hu + kv + \ell w = 0$) map to the same great circle.

The **gnomonic projection** maps $\hat{k}_f$ to a plane tangent to the unit
sphere at $\hat{k}_i$.  In this projection, great circles become **straight
lines**, making zone axes easy to identify visually.

### 7.2 Indexing strategy

Automated indexing (as in LaueTools) proceeds in three stages:

1. **Peak detection** — locate spot centres in the detector image to
   sub-pixel accuracy.

2. **Inter-spot angles** — compute the angle between every pair of spots:

    $$
    \cos\alpha_{ij} = \hat{k}_{f,i}\cdot\hat{k}_{f,j}
    $$

    This angle is independent of the orientation matrix and depends only on
    the crystal metric tensor.

3. **Look-up and match** — compare measured angles against a pre-computed
   table of all $\cos\alpha_{hkl,h'k'l'}$ for the candidate lattice.
   Consistent triples $(hkl)$, $(h'k'l')$, $(h''k''l'')$ uniquely determine
   $U$ up to a small refinement.

### 7.3 Refinement

Once an initial $U$ is found, it is refined by minimising the reprojection
error between simulated and measured spot positions:

$$
\min_{U,\,\mathbf{p}} \sum_i
\left\lVert \mathbf{x}_i^\text{meas}
- \mathbf{x}_i^\text{sim}(U, \mathbf{p})\right\rVert^2
$$

where $\mathbf{p}$ are the detector calibration parameters
$(D, 2\theta_0, \nu, \chi_D, \ldots)$.

---

## 8. Strain determination

A strained crystal has a distorted reciprocal lattice.  The observed
reciprocal-lattice vector $\mathbf{G}_{hkl}^\text{obs}$ is related to the
unstrained reference $\mathbf{G}_{hkl}^0$ by the strain tensor $\varepsilon$:

$$
\mathbf{G}_{hkl}^\text{obs} = (\mathbf{I} + \varepsilon)\,\mathbf{G}_{hkl}^0
$$

### 8.1 Deviatoric vs. hydrostatic strain

White-beam Laue is sensitive **only to deviatoric (shape-changing) strain**,
not to hydrostatic (volume-changing) strain.  This is because the Laue
condition fixes the *direction* of $\mathbf{G}$ but not its magnitude (the
wavelength self-adjusts).  Hydrostatic expansion/contraction changes
$\lvert\mathbf{G}\rvert$ uniformly and shifts $E_{hkl}$ without moving spots
on the detector.

The **deviatoric strain tensor** $\varepsilon'$ is defined as

$$
\varepsilon' = \varepsilon - \tfrac{1}{3}\operatorname{tr}(\varepsilon)\,\mathbf{I}
$$

It has 5 independent components (a traceless symmetric $3\times3$ tensor).

### 8.2 Peak-shift Jacobian

The shift in the $(2\theta, \chi)$ position of spot $hkl$ due to a small
deviatoric strain $\delta\varepsilon'$ is

$$
\begin{pmatrix}\delta(2\theta)\\\delta\chi\end{pmatrix}
= J_{hkl}\,\delta\varepsilon'_\text{Voigt}
$$

where $J_{hkl}$ is a $2\times5$ Jacobian matrix that depends on $\mathbf{G}_{hkl}$
and the detector geometry.  With enough spots ($\geq 3$ independent reflections)
the 5-component deviatoric strain tensor can be extracted by least-squares.
The function `strain_spot_jacobian` in `nrxrdct.laue` computes $J_{hkl}$
analytically.

---

## 9. Quick-reference: key equations

| Quantity | Formula |
|---|---|
| Wavelength at Bragg condition | $\lambda_{hkl} = -4\pi\,(\hat{k}_i\cdot\mathbf{G}) / \lvert\mathbf{G}\rvert^2$ |
| Photon energy | $E = hc/\lambda \approx 12398\,\text{eV}/\lambda[\text{Å}]$ |
| Bragg angle | $\sin\theta = \lvert\mathbf{G}\rvert\lambda/(4\pi)$ |
| Scattering angle | $2\theta = \arccos(\hat{k}_i\cdot\hat{k}_f)$ |
| LP factor (unpolarised) | $LP = (1+\cos^2 2\theta)/(2\sin^2\theta\cos\theta)$ |
| BM critical energy (ESRF) | $E_c \approx 0.665\,B[\text{T}]\,(E_e[\text{GeV}])^2\,\text{keV}$ |
