# Rietveld Refinement

This page describes the theory behind Rietveld refinement and how to use
`nrxrdct.rietveld` to drive GSAS-II refinements from Python: instrument
calibration, sequential parameter refinement, multi-phase models, and
per-voxel refinement as part of the XRD-CT pipeline.

The first part (sections 1–9) explains the physical model that GSAS-II fits
and what each refinable parameter means. The second part (sections 10–18)
shows the corresponding `nrxrdct` API.

> **Prerequisite**: GSAS-II must be installed and importable as
> `GSASII.GSASIIscriptable`. See [Installation](../installation.md) for setup
> instructions.

---

# Part I — Theory

## 1. The Rietveld method

A powder diffraction pattern is a list of intensities $y_i^{\text{obs}}$
measured at discrete steps $2\theta_i$. In contrast to single-peak
fitting, the Rietveld method (Rietveld, 1969) does not fit each reflection
independently. Instead it calculates the **whole pattern** from a physical
model, made up of the crystal structure, the microstructure, the sample and
the instrument, and adjusts the parameters of that model by least squares
until the calculated pattern matches the observed one point by point.

The calculated intensity at step $i$ is

$$
y_i^{\text{calc}} = y_{b,i}
+ S_h \sum_{\phi} S_{\phi}
\sum_{hkl} L_{hkl}\, \lvert F_{hkl}\rvert^2\, m_{hkl}\, P_{hkl}\, A\, E_{hkl}\;
\Phi\!\left(2\theta_i - 2\theta_{hkl}\right)
$$

| Symbol | Meaning | Section |
|---|---|---|
| $y_{b,i}$ | Background at step $i$ | [7](#7-background) |
| $S_h$ | Histogram scale factor | [6.6](#66-scale-factors-and-quantitative-phase-analysis) |
| $S_\phi$ | Phase scale factor (HAP *Scale*) | [6.6](#66-scale-factors-and-quantitative-phase-analysis) |
| $L_{hkl}$ | Lorentz–polarization factor | [6.2](#62-lorentzpolarization-factor) |
| $F_{hkl}$ | Structure factor | [6.1](#61-structure-factor) |
| $m_{hkl}$ | Reflection multiplicity | — |
| $P_{hkl}$ | Preferred-orientation correction | [6.3](#63-preferred-orientation) |
| $A$ | Absorption correction | [6.4](#64-absorption) |
| $E_{hkl}$ | Extinction correction | [6.5](#65-extinction) |
| $\Phi$ | Normalised peak-profile function | [5](#5-peak-profile) |
| $2\theta_{hkl}$ | Calculated peak position | [4](#4-peak-positions) |

The outer sum runs over all phases $\phi$ in the histogram, the inner sum
over every reflection whose profile reaches step $i$. Overlapping
reflections are therefore handled naturally: the model distributes the
observed intensity between them according to their calculated structure
factors.

In GSAS-II terms the parameters fall into three groups, which matters for
the API below:

* **Histogram parameters**: instrument (`Zero`, `Lam`, `U`, `V`, `W`, `X`,
  `Y`, `Z`, `SH/L`), sample (`Scale`, `Absorption`, `Shift`, `DisplaceX/Y`)
  and background. They belong to one pattern.
* **Phase parameters**: unit cell, atomic coordinates, occupancies,
  displacement parameters. They belong to one crystal structure.
* **HAP (histogram-and-phase) parameters**: phase scale, crystallite size,
  microstrain, hydrostatic strain, preferred orientation, extinction, Babinet.
  They describe *this phase as seen in this pattern*.

---

## 2. Least-squares minimisation

The refinement minimises the weighted sum of squared residuals

$$
M = \sum_i w_i \left(y_i^{\text{obs}} - y_i^{\text{calc}}\right)^2,
\qquad w_i = \frac{1}{\sigma_i^2}
$$

where $\sigma_i$ is the standard uncertainty of point $i$. Because
$y^{\text{calc}}$ is non-linear in most parameters (cell, positions, peak
widths), $M$ is minimised iteratively. At each cycle the model is
linearised around the current parameter vector $\mathbf{p}$:

$$
\left(\mathbf{J}^\mathsf{T}\mathbf{W}\mathbf{J}\right)\Delta\mathbf{p}
= \mathbf{J}^\mathsf{T}\mathbf{W}\,\Delta\mathbf{y},
\qquad J_{ij} = \frac{\partial y_i^{\text{calc}}}{\partial p_j}
$$

GSAS-II solves this with a Levenberg–Marquardt-damped Gauss–Newton
algorithm. Two consequences follow directly from the equation:

* **Starting values matter.** The linearisation is only valid near the
  minimum. A cell or zero shift that puts calculated peaks far from the
  observed ones gives derivatives that point nowhere useful, and the
  refinement diverges or settles in a false minimum. This is why parameters
  are freed progressively (section [8](#8-refinement-strategy-and-correlations)).
* **Correlated parameters are ill-conditioned.** If two parameters change
  $y^{\text{calc}}$ in nearly the same way, their columns of $\mathbf{J}$
  are nearly parallel and the normal matrix is nearly singular.

### 2.1 Uncertainties and correlations

The inverse of the normal matrix gives the covariance matrix of the refined
parameters,

$$
\mathbf{C} = \left(\mathbf{J}^\mathsf{T}\mathbf{W}\mathbf{J}\right)^{-1}\cdot \chi^2_\nu,
$$

scaled by the reduced χ² (section [3](#3-agreement-indices)) so that the
estimated standard deviations (esds) are not over-optimistic when the fit is
imperfect. The esd of parameter $j$ is $\sqrt{C_{jj}}$ and the correlation
between $j$ and $k$ is

$$
\rho_{jk} = \frac{C_{jk}}{\sqrt{C_{jj}\,C_{kk}}} \in [-1, 1].
$$

$\lvert\rho\rvert > 0.9$ means the data cannot really distinguish the two
parameters. `print_covariance_matrix`, `plot_covariance_matrix` and
`print_variable_diagnostics` (section [15](#15-diagnostics-and-reporting))
display these quantities.

> **Note.** A covariance matrix only exists for parameters that were free
> **in the same cycle**. The usual refine-then-freeze pattern refines
> parameters in separate cycles, so their mutual correlation is never
> computed. Run one final cycle with everything free
> (`refine_ever_refined_variables`) to get a complete covariance matrix and
> realistic esds.

### 2.2 Where the weights come from

If the `.xy` file has a third column (for example pyFAI's propagated
uncertainty, written by `save_xy_file`), it is used as $\sigma_i$. For a
two-column file GSAS-II assumes Poisson statistics, $\sigma_i^2 = y_i$.

For patterns obtained by azimuthal integration of 2-D detector images, and
especially for voxel patterns reconstructed by XRD-CT, neither assumption
is strictly correct. The intensities are averages or linear combinations of
many pixels and do not follow counting statistics. In that case the
*absolute* value of χ² is not meaningful, but its *change* between
models still is.

---

## 3. Agreement indices

GSAS-II reports several figures of merit. With sums over all included
points $i$ ($N$ points, $P$ refined parameters):

**Profile R-factor**

$$
R_p = \frac{\sum_i \lvert y_i^{\text{obs}} - y_i^{\text{calc}}\rvert}{\sum_i y_i^{\text{obs}}}
$$

**Weighted-profile R-factor**, the quantity actually minimised:

$$
R_{wp} = \sqrt{\frac{\sum_i w_i \left(y_i^{\text{obs}} - y_i^{\text{calc}}\right)^2}
{\sum_i w_i \left(y_i^{\text{obs}}\right)^2}}
$$

**Expected R-factor**, the best $R_{wp}$ achievable given the noise:

$$
R_{\text{exp}} = \sqrt{\frac{N - P}{\sum_i w_i \left(y_i^{\text{obs}}\right)^2}}
$$

**Goodness of fit**

$$
\text{GOF} = \sqrt{\chi^2_\nu} = \frac{R_{wp}}{R_{\text{exp}}}
= \sqrt{\frac{\sum_i w_i \left(y_i^{\text{obs}} - y_i^{\text{calc}}\right)^2}{N - P}}
$$

**Bragg R-factor**, computed on integrated intensities of the reflections of
one phase rather than on profile points:

$$
R_{B} = \frac{\sum_{hkl} \lvert I_{hkl}^{\text{obs}} - I_{hkl}^{\text{calc}}\rvert}
{\sum_{hkl} I_{hkl}^{\text{obs}}}
$$

How to read them:

* $R_{wp}$ depends strongly on the background: a high background lowers
  $R_{wp}$ without the structural model being better. Always compare it
  to $R_{\text{exp}}$, or look at the GOF.
* GOF ≈ 1 means the residuals are consistent with the noise. GOF ≫ 1
  means systematic misfit, or under-estimated $\sigma_i$. GOF < 1 usually
  means the $\sigma_i$ are over-estimated, or the model has too many
  parameters.
* $R_B$ is the best indicator of the quality of the **structural** model,
  because it is insensitive to background and profile errors.
* **No number replaces the difference plot.** A flat residual with no
  features at peak positions is the real criterion. Residuals shaped like a
  derivative indicate a position error (cell, zero, displacement), symmetric
  "W"/"M" shapes indicate a width or shape error, and residuals that grow or
  shrink with 2θ indicate an intensity error (displacement parameters,
  absorption, preferred orientation).

`get_Rwp()` returns $R_{wp}$ in percent and `get_chi2()` returns the GOF
(square it for reduced χ²).

---

## 4. Peak positions

### 4.1 Bragg's law and the metric tensor

A reflection $hkl$ appears at

$$
2\theta_{hkl} = 2\arcsin\!\left(\frac{\lambda}{2\,d_{hkl}}\right),
\qquad
\frac{1}{d_{hkl}^2} = \mathbf{h}^\mathsf{T}\,\mathbf{G}^*\,\mathbf{h},
$$

where $\mathbf{h} = (h, k, l)$ and $\mathbf{G}^*$ is the reciprocal metric
tensor, a function of the six cell parameters
$(a, b, c, \alpha, \beta, \gamma)$. GSAS-II refines the six independent
components $A_1 \dots A_6$ of $\mathbf{G}^*$:

$$
\frac{1}{d_{hkl}^2} = A_1 h^2 + A_2 k^2 + A_3 l^2 + A_4 hk + A_5 hl + A_6 kl
$$

subject to the constraints of the space group (cubic: one free parameter;
hexagonal/tetragonal: two; orthorhombic: three; monoclinic: four;
triclinic: six).

### 4.2 Instrument and sample shifts

The observed position also contains shifts that are not structural:

$$
2\theta_{\text{obs}} = 2\theta_{hkl} + \text{Zero} + \Delta_{\text{disp}}(2\theta)
$$

| Parameter | Geometry | Angular dependence |
|---|---|---|
| `Zero` | any | constant |
| `Shift` | Bragg–Brentano | $\propto \cos\theta$ |
| `DisplaceX` | Debye–Scherrer, ⟂ beam | $\propto \cos 2\theta$ |
| `DisplaceY` | Debye–Scherrer, ∥ beam | $\propto \sin 2\theta$ |

### 4.3 Hydrostatic / deviatoric strain

A uniform elastic strain in the phase changes all d-spacings in an
hkl-dependent way. GSAS-II models this as a perturbation $\Delta A_j$ of the
metric tensor (the *HStrain* $D_{ij}$ terms). Unlike microstrain, it
**shifts** peaks without broadening them. In a single-histogram fit it is
indistinguishable from the cell parameters. It becomes useful when the same
phase is refined in several histograms that share one cell, or when the
cell is fixed to a reference value.

### 4.4 The wavelength–cell–zero correlation

Differentiating Bragg's law gives

$$
\frac{\Delta d}{d} = \frac{\Delta\lambda}{\lambda} - \cot\theta\,\Delta\theta.
$$

A relative error in $\lambda$ produces exactly the same pattern as the same
relative error in all cell lengths. **Wavelength and cell parameters can
never be refined together from one pattern.** Zero and displacement are
separable from them only because they have a different $\theta$ dependence,
which requires a wide 2θ range. This is why a calibrant with a certified
cell is used to fix $\lambda$ (or the sample-to-detector distance during
integration) and `Zero` before any sample cell is refined.

---

## 5. Peak profile

### 5.1 The pseudo-Voigt function

The profile function $\Phi$ for constant-wavelength data is, in GSAS-II, a
pseudo-Voigt approximation of a Voigt function (the convolution of a
Gaussian and a Lorentzian):

$$
\Phi(\Delta) = \eta\, L(\Delta, \Gamma) + (1 - \eta)\, G(\Delta, \Gamma)
$$

with

$$
G(\Delta,\Gamma) = \frac{2}{\Gamma}\sqrt{\frac{\ln 2}{\pi}}\,
\exp\!\left(-\frac{4\ln 2\,\Delta^2}{\Gamma^2}\right),
\qquad
L(\Delta,\Gamma) = \frac{2}{\pi\Gamma}\,
\frac{1}{1 + 4\Delta^2/\Gamma^2}.
$$

In the Thompson–Cox–Hastings (TCH) formulation, the Gaussian width
$\Gamma_G$ and Lorentzian width $\Gamma_L$ are parametrised separately and
combined into the total FWHM $\Gamma$ and the mixing parameter $\eta$:

$$
\Gamma^5 = \Gamma_G^5 + 2.69269\,\Gamma_G^4\Gamma_L + 2.42843\,\Gamma_G^3\Gamma_L^2
+ 4.47163\,\Gamma_G^2\Gamma_L^3 + 0.07842\,\Gamma_G\Gamma_L^4 + \Gamma_L^5
$$

$$
\eta = 1.36603\,\frac{\Gamma_L}{\Gamma} - 0.47719\left(\frac{\Gamma_L}{\Gamma}\right)^2
+ 0.11116\left(\frac{\Gamma_L}{\Gamma}\right)^3
$$

The advantage is that each width has a clear physical origin and angular
dependence.

### 5.2 Instrumental broadening

**Gaussian (Caglioti) term.** GSAS-II's `U`, `V`, `W` define the Gaussian
*variance* in centidegrees²:

$$
\sigma^2 = U \tan^2\theta + V \tan\theta + W,
\qquad \Gamma_G = \sqrt{8\ln 2\;\sigma^2}
$$

`W` is the angle-independent term from beam size, detector pixel size and
point-spread function. For synchrotron data with a 2-D detector it is often
the only significant one. `U` and `V` come from the divergence and
wavelength spread of the beam and matter mostly for laboratory sources.

**Lorentzian term**, in centidegrees:

$$
\Gamma_L = \frac{X}{\cos\theta} + Y \tan\theta + Z
$$

**Axial divergence (`SH/L`).** The finite height of the beam and detector
makes the Debye–Scherrer cones intersect the detector along curved lines,
giving an asymmetric low-angle tail at small 2θ (and a high-angle tail
above 90°). GSAS-II convolves the pseudo-Voigt with the
Finger–Cox–Jephcoat (FCJ) asymmetry function, parametrised by
$\text{SH/L} = (S + H)/L$. With a 2-D detector and a small beam, `SH/L` is
usually fixed at a very small value. GSAS-II's CW profile calculation uses
`max(SH/L, 0.002)`, and each refinement cycle writes `SH/L` back clamped to
at least 0.0005, so anything below 0.002 has no effect on the fit. The
starting `.instprm` therefore uses 0.002.

**Pink-beam profiles.** `ExpFCJVoigt` and `EpsVoigt` additionally convolve
the profile with back-to-back exponentials,
$\alpha = \alpha_0 + \alpha_1 \sin\theta$ (rise) and
$\beta = \beta_0 + \beta_1 \sin\theta$ (decay). These model the asymmetric
tails produced by a polychromatic (pink) beam bandpass.

### 5.3 Sample broadening

The sample adds its own widths on top of the instrumental ones. In GSAS-II
the Lorentzian widths are **added** to $\Gamma_L$ and the Gaussian variances
are added to $\sigma^2$, with a mixing coefficient (default fully
Lorentzian) that decides which component receives the sample contribution.

**Crystallite size (Scherrer).** Coherently diffracting domains of finite
size $D$ broaden the reflection by

$$
\Gamma_{\text{size}} = \frac{K\lambda}{D\cos\theta}
\quad\Longrightarrow\quad
\Gamma_{\text{size}}\,[\text{centideg}] = \frac{18000}{\pi}\,\frac{K\lambda}{D}\,\frac{1}{\cos\theta}
$$

GSAS-II uses $K = 1$ and reports $D$ in µm. The $1/\cos\theta$ dependence
is the same as `X`.

**Microstrain.** A distribution of d-spacings with relative width
$\varepsilon = \Delta d/d$ broadens the reflection by
$\Gamma_{\text{strain}} = 2\,\varepsilon\tan\theta$ (in radians). GSAS-II
parametrises it as

$$
\Gamma_{\text{strain}}\,[\text{centideg}] = \frac{18000}{\pi}\,10^{-6}\,\mu\varepsilon\;\tan\theta,
$$

so its microstrain $\mu\varepsilon$ satisfies
$10^{-6}\,\mu\varepsilon = 2\,\Delta d/d$, where $\Delta d/d$ is the FWHM of
the strain distribution. Keep this factor in mind when comparing with values
from other programs or from a Williamson–Hall analysis. The $\tan\theta$
dependence is the same as `Y`.

**Williamson–Hall.** Combining both in reciprocal units gives

$$
\Gamma\cos\theta = \frac{K\lambda}{D} + 2\varepsilon\sin\theta,
$$

so a plot of $\Gamma\cos\theta$ against $\sin\theta$ has an intercept
related to size and a slope related to strain. A wide 2θ range is needed to
separate the two, and they always remain correlated in a refinement.

**Anisotropic broadening.** When the broadening depends on $hkl$, the
uniaxial and ellipsoidal size models and the uniaxial and generalized
(Stephens) microstrain models apply. They are described with their exact
GSAS-II formulas in section [9](#9-histogram-and-phase-hap-models), together
with how the sample widths are split between the Lorentzian and Gaussian
components.

### 5.4 Why instrument calibration is necessary

Because sample and instrument contributions have the **same angular
dependence** (size ↔ `X`, strain ↔ `Y`), a single pattern cannot separate
them. The instrument resolution must be determined beforehand from a
calibrant with negligible size and strain broadening (LaB₆ NIST SRM 660,
CeO₂, Si). Its refined `Zero`, `W`, `X`, `Y`, … are then **fixed** in all
sample refinements, so that any additional broadening is attributed to the
sample.

`InstrumentCalibration.add_phase` sets the calibrant's Size to 10 µm and
Mustrain to 0 and freezes them. If GSAS-II's defaults (1 µm, 1000 µε) were
left in place, they would contribute about $5.7\tan\theta$ centideg of
spurious Lorentzian width, biasing the calibrated `Y` low (often negative),
and every sample microstrain refined later would come out too high.

---

## 6. Peak intensities

### 6.1 Structure factor

$$
F_{hkl} = \sum_j o_j\, f_j(Q)\,
\exp\!\left[2\pi i\,(h x_j + k y_j + l z_j)\right]\,
\exp\!\left(-8\pi^2 U_{\text{iso},j}\,\frac{\sin^2\theta}{\lambda^2}\right)
$$

The sum runs over all atoms in the unit cell, with occupancy $o_j$,
scattering factor $f_j$ (including anomalous terms $f' + if''$) and
fractional coordinates $(x_j, y_j, z_j)$. The last factor is the
**Debye–Waller** factor. $U_{\text{iso}}$ is the mean-square atomic
displacement (thermal vibration plus static disorder), and
$B_{\text{iso}} = 8\pi^2 U_{\text{iso}}$ is the older notation. It damps
intensities increasingly at high angle, so it controls the overall
intensity fall-off with $2\theta$. Typical room-temperature values are
0.003–0.02 Å².

### 6.2 Lorentz–polarization factor

The product of the Lorentz factor (the time each reflection spends in
diffraction condition, and the fraction of the Debye–Scherrer cone
intercepted) and the polarization factor is

$$
L_p = \frac{p(2\theta)}{\sin^2\theta\,\cos\theta},
$$

where the polarization term $p(2\theta)$ depends on the source. For an
unpolarised laboratory beam $p = (1 + \cos^2 2\theta)/2$. For a synchrotron
beam polarised in the horizontal plane with fraction $P$ (`Polariz.` in the
instrument file, typically 0.95–0.99), $p$ also depends on the detector
azimuth $\psi$, and GSAS-II uses that azimuth-dependent form. For an
azimuthally integrated pattern, pyFAI may already have applied the
polarization correction during integration. In that case set `Polariz.`
consistently in the instrument file, otherwise the correction is applied
twice.

### 6.3 Preferred orientation

A random powder has every crystallite orientation equally likely. Texture
changes the fraction of crystallites in diffraction condition for each
$hkl$.

**March–Dollase** (one parameter $r$ about a direction $\mathbf{n}$):

$$
P_{hkl} = \frac{1}{m}\sum_{\{hkl\}}
\left(r^2\cos^2\alpha + \frac{\sin^2\alpha}{r}\right)^{-3/2}
$$

where $\alpha$ is the angle between the scattering vector of $hkl$ and
$\mathbf{n}$, and the sum is over symmetry-equivalent reflections. $r = 1$
means no texture. $r < 1$ and $r > 1$ correspond to plate-like and
needle-like habits (or the reverse, depending on the geometry).

**Spherical harmonics** (Bunge / Von Dreele 1997):

$$
P_{hkl}(\Phi, \beta) = 1 + \sum_{l=2}^{L}\frac{4\pi}{2l+1}
\sum_{m}\sum_{n} C_l^{mn}\, k_l^m(h)\, k_l^n(y)
$$

This expands the orientation distribution function in symmetrised harmonics
up to even order $L$, constrained by the crystal symmetry and an assumed
*sample* symmetry (cylindrical, orthorhombic, …). It is more flexible but
needs many more parameters.

> In transmission with a 2-D detector, integrating the full ring averages
> over many sample directions and largely washes out texture. Residual
> texture effects in azimuthally integrated data usually indicate *large
> grains* (spotty rings, poor particle statistics) rather than true
> preferred orientation, and no texture model can fix that. In that case
> the problem has to be addressed at the integration stage (masking,
> rotating the sample, larger beam).

### 6.4 Absorption

The incident and diffracted beams are attenuated in the sample. For a
cylinder of radius $r$ and linear attenuation coefficient $\mu$, the
correction depends on $\mu r$ and $\theta$ and mostly suppresses
**low-angle** intensities relative to high-angle ones. It is therefore
correlated with $U_{\text{iso}}$, which suppresses **high-angle** intensities.
At high synchrotron energies $\mu r$ is usually small (< 0.1), and the
correction can be calculated from composition and density and kept fixed
(`set_absorption`, `nrxrdct.utils.calculate_absorption_coefficient`).

### 6.5 Extinction

For large, nearly perfect crystallites, strong reflections deplete the
incident beam inside a grain (primary extinction), so their intensity falls
below the kinematical $\lvert F\rvert^2$. It is rarely significant for
powders. Refine it only if the strongest low-angle reflections are
systematically over-calculated after the structure has converged. The
GSAS-II extinction model is given in section [9.6](#96-extinction).

### 6.6 Scale factors and quantitative phase analysis

Each phase's integrated intensities are proportional to the amount of that
phase in the beam. Following Hill and Howard (1987), the weight fraction of
phase $p$ in a mixture of $n$ crystalline phases is

$$
W_p = \frac{S_p\,(Z M V)_p}{\sum_{k=1}^{n} S_k\,(Z M V)_k}
$$

with $S$ the phase scale factor, $Z$ the number of formula units per cell,
$M$ the formula mass and $V$ the cell volume. The scale factors themselves
do **not** sum to 1; only their mass-weighted ratios are meaningful.
GSAS-II puts the cell-volume normalisation into its structure factors, so
its HAP "phase fraction" enters as $W_p = S_p\,\mathrm{Mass}_p / \sum_k
S_k\,\mathrm{Mass}_k$, with $\mathrm{Mass}$ the unit-cell mass. Refine the
phase scales against a fixed histogram scale (`refine_phase_content`),
then call `weight_fractions()` to tabulate $W_p$ with esds propagated
from the scale-factor covariance.

Phases in Le Bail mode are excluded: their extracted intensities are free
parameters, so the scale is degenerate with them and says nothing about
phase abundance.

The fractions are **relative to the crystalline phases in the model**.
Amorphous content or unmodelled phases are not accounted for unless an
internal standard of known weight fraction is added. Large differences in
absorption between phases (microabsorption) also bias the result; the
Brindley correction addresses this.

### 6.7 Le Bail extraction

In Le Bail mode (`set_LeBail`), the structure factors are not calculated
from atoms. Instead, at each cycle, the observed intensity under each peak
is partitioned between overlapping reflections in proportion to their
current calculated intensities, and these partitioned values become the new
$\lvert F_{hkl}\rvert^2$. Only cell, profile, background and zero are
refined. Le Bail fits are useful to:

* calibrate the instrument without depending on a structural model;
* check the cell and space group before Rietveld refinement;
* fit an unknown or poorly described phase alongside phases refined by
  Rietveld.

Because the intensities are free, a Le Bail fit always gives a lower
$R_{wp}$ than the corresponding Rietveld fit. The difference between the two
measures how much the structural model is limiting the fit.

---

## 7. Background

The background contains scattering from air, the sample environment,
amorphous content, Compton and thermal diffuse scattering and fluorescence.
It is usually modelled as a smooth function, by default a Chebyshev
polynomial in a reduced 2θ variable:

$$
y_{b}(2\theta) = \sum_{j=0}^{N-1} B_j\, T_j(x),
\qquad x = \frac{2(2\theta) - (2\theta_{\max} + 2\theta_{\min})}{2\theta_{\max} - 2\theta_{\min}}
$$

Too few terms leave broad residual humps. Too many terms let the
background absorb the tails of the peaks, which correlates with peak
widths, $U_{\text{iso}}$ and scale. 6–12 terms are typical. Diffuse
humps from amorphous material can be modelled physically with Debye
terms,

$$
y_{\text{Debye}}(Q) = A\,\frac{\sin(QR)}{QR}\,\exp\!\left(-U Q^2\right),
$$

each representing a characteristic interatomic distance $R$. Alternatively,
a pre-computed background curve can be supplied (`function="user"`), for
example from a SNIP or asymmetric least-squares estimate.

---

## 8. Refinement strategy and correlations

Parameters are freed progressively: first those that affect the pattern
most strongly and are least correlated, then those that refine subtler
features. A typical order is:

1. **Background** and **scale**, which set the overall intensity level.
2. **Zero** (or displacement), then **cell**, to bring every calculated
   peak onto its observed position. Nothing else converges otherwise.
3. **Peak profile**: instrument parameters for a calibrant, size and
   microstrain for a sample (with the instrument fixed).
4. **Intensity corrections**: $U_{\text{iso}}$, then atomic coordinates,
   then occupancies, preferred orientation, absorption.
5. **A final cycle with everything free together** to get the full
   covariance matrix and correct esds.

At each step, check the difference plot and $R_{wp}$. If $R_{wp}$
increases or a parameter takes an unphysical value, go back
(`restore_backup`) and change the order or fix the problematic parameter.

Common correlations:

| Parameters | Reason | Mitigation |
|---|---|---|
| `Lam` ↔ cell | identical effect on all d-spacings | never refine together; fix λ from calibration |
| `Zero` ↔ displacement ↔ cell | position shifts with similar θ dependence | wide 2θ range; fix `Zero` from calibration |
| Size ↔ `X`, Mustrain ↔ `Y` | same angular dependence | fix instrument parameters from a calibrant |
| Size ↔ Mustrain | both broaden; separated only by θ dependence | wide 2θ range; refine one first |
| Scale ↔ $U_{\text{iso}}$ ↔ occupancy | all scale intensities | refine scale first; fix occupancies unless needed |
| Absorption ↔ $U_{\text{iso}}$ | opposite θ trends that partly compensate | calculate μr and fix it |
| Background ↔ peak tails, $U_{\text{iso}}$ | high-order polynomial absorbs tails | use the fewest terms that work |
| HStrain ↔ cell | both change d-spacings | only use HStrain with a fixed/shared cell |

---

## 9. Histogram-and-phase (HAP) models

HAP parameters describe how one phase appears in one histogram. They are
stored per (phase, histogram) pair rather than per phase, because the same
crystal structure can have a different amount, microstructure or texture in
different measurements. In XRD-CT, for example, every voxel is its own
histogram, so the ferrite cell and atoms can be shared while its microstrain
and phase fraction vary from voxel to voxel.

| HAP entry | Affects | Models | `nrxrdct` method |
|---|---|---|---|
| `Scale` | intensity | — | `refine_phase_scale`, `refine_phase_content` |
| `Size` | peak width (∝ 1/cosθ) | isotropic, uniaxial, ellipsoidal | `refine_crystallite_size` |
| `Mustrain` | peak width (∝ tanθ) | isotropic, uniaxial, generalized | `refine_mustrain` |
| `HStrain` | peak position | symmetry-constrained $D_{ij}$ | `refine_hstrain` |
| `Pref.Ori.` | intensity | March–Dollase, spherical harmonics | `refine_preferential_orientation` |
| `Extinction` | intensity | Sabine | `refine_extinction` |
| `Babinet` | intensity (low angle) | Babinet solvent | `refine_babinet` |
| `LeBail` | intensity | Le Bail extraction on/off | `set_LeBail` |

All values can be inspected with `print_HAP_parameters`, set with
`set_HAP_parameter`, and fixed with `freeze_HAP_parameter`.

The formulas below are those GSAS-II evaluates for constant-wavelength
X-ray data. Angles are in degrees, widths in centidegrees (0.01°), $\lambda$
in Å, and $\theta$ is the Bragg angle of the reflection.

### 9.1 How sample broadening enters the profile

Size and microstrain each give a total sample width, $\Gamma_S$ and
$\Gamma_M$. A **mixing coefficient** `LGmix` ($\eta_S$ for size, $\eta_M$ for
strain, between 0 and 1) splits each width between the Lorentzian and
Gaussian parts of the TCH profile (section [5.1](#51-the-pseudo-voigt-function)):

$$
\Gamma_L^{\text{sample}} = \eta_S\,\Gamma_S + \eta_M\,\Gamma_M,
\qquad
\sigma^2_{\text{sample}} = \frac{\left[(1-\eta_S)\,\Gamma_S\right]^2 + \left[(1-\eta_M)\,\Gamma_M\right]^2}{8\ln 2}
$$

These are added to the instrumental $\Gamma_L$ and $\sigma^2$. The default is
$\eta = 1$, so all sample broadening is Lorentzian. This is a good
approximation for size broadening, where the column-length distributions
of real powders give nearly Lorentzian tails. Strain broadening from a
roughly normal distribution of d-spacings is closer to Gaussian, and
refining $\eta_M$ can help when a high-quality pattern shows that the peak
shape (not only its width) is wrong. In most cases $\eta$ should stay
fixed: it is strongly correlated with the widths themselves. Set it through
the Size/Mustrain `refine_dict` as `"LGmix": 0.8` or
`"LGmix": {"value": 0.8, "refine": False}` (section [13](#13-key-methods-reference)).

### 9.2 Crystallite size

Size broadening comes from the finite number of lattice planes in a
coherently diffracting domain. GSAS-II reports an **apparent** size $D$ in
µm, using the Scherrer equation with $K = 1$. Converting it to a physical
particle dimension requires a shape-dependent Scherrer constant and a
choice of averaging (volume- or area-weighted), so $D$ is best compared
between related samples rather than read as a particle diameter.

**Isotropic.** A single size for all reflections:

$$
\Gamma_S = \frac{1.8\,\lambda}{\pi\,D\cos\theta}
$$

The factor $1.8/\pi$ converts radians to centidegrees ($18000/\pi$) and µm
to Å ($10^{-4}$). In practice size broadening is only measurable for $D
\lesssim 1$ µm at synchrotron resolution; above that the refined value is
poorly determined and should be fixed at a large value.

**Uniaxial.** Two sizes, $D_{\text{eq}}$ perpendicular to and
$D_{\text{ax}}$ along a unique direction $\mathbf{n}$. For a reflection whose
scattering vector makes an angle $\varphi$ with $\mathbf{n}$:

$$
\Gamma_S(\varphi) = \frac{1.8\,\lambda}{\pi\cos\theta}
\sqrt{\frac{\sin^2\varphi}{D_{\text{eq}}^2} + \frac{\cos^2\varphi}{D_{\text{ax}}^2}}
$$

so $D(\varphi = 0) = D_{\text{ax}}$ and $D(\varphi = 90°) = D_{\text{eq}}$.
This describes needle-shaped ($D_{\text{ax}} > D_{\text{eq}}$) or plate-shaped
($D_{\text{ax}} < D_{\text{eq}}$) crystallites. The direction is given as
Miller indices $hkl$ and GSAS-II uses it as a reciprocal-lattice vector,
i.e. the normal to the $(hkl)$ planes. For orthogonal axes (cubic,
tetragonal, orthorhombic, and $[001]$ in hexagonal) this is the same as the
direct-lattice axis. Choose $\mathbf{n}$ from the crystal habit, e.g.
$[001]$ for hexagonal platelets.

**Ellipsoidal** (`refine_type="ellipsoidal"`; `"generalized"` is accepted as
an alias). The size along the scattering direction $\hat{\mathbf{h}}$ (a
unit vector in Cartesian crystal coordinates) is that of an ellipsoid
described by a symmetric tensor $\mathbf{S}$ (in µm⁻²) with six components
$S_{11} \dots S_{23}$:

$$
D(\hat{\mathbf{h}}) = \left(\hat{\mathbf{h}}^{\mathsf{T}}\,\mathbf{S}\,\hat{\mathbf{h}}\right)^{-1/2},
\qquad
\Gamma_S = \frac{1.8\,\lambda}{\pi\,D(\hat{\mathbf{h}})\cos\theta}
$$

A sphere of diameter $D$ is $\mathbf{S} = \mathbf{I}/D^2$, which is how
`nrxrdct` initialises the tensor from the current isotropic size.
GSAS-II does **not** constrain $\mathbf{S}$ by the crystal symmetry: the six
terms are independent, and terms the data cannot determine make the
refinement singular. `refine_crystallite_size` therefore refines only the
diagonal terms by default; pass `"terms": [...]` in `refine_dict` to choose
others. For cubic phases the isotropic model is the symmetry-consistent
choice, and for hexagonal and tetragonal phases the uniaxial model. Use the
ellipsoidal model for lower symmetries when the crystallite shape has no
single unique axis, and only with enough well-resolved reflections in all
directions.

### 9.3 Microstrain

Microstrain broadening comes from a distribution of d-spacings *within* the
illuminated volume: dislocations, stacking faults, composition gradients,
intergranular stresses. GSAS-II reports it in µε. With the convention of
section [5.3](#53-sample-broadening), $10^{-6}\,\mu\varepsilon$ is twice the
FWHM of the relative d-spacing distribution $\Delta d/d$.

**Isotropic.**

$$
\Gamma_M = \frac{0.018}{\pi}\,\mu\varepsilon\,\tan\theta
$$

(the same $18000/\pi$ factor with $10^{-6}$). Typical values are 0–500 µε
for annealed materials and 1000–5000 µε for heavily deformed metals or
nanocrystalline oxides.

**Uniaxial.** Two strains, $\mu\varepsilon_{\text{eq}}$ and
$\mu\varepsilon_{\text{ax}}$, about a unique direction $\mathbf{n}$ (same
convention as uniaxial size):

$$
\Gamma_M(\varphi) = \frac{0.018}{\pi}\,\tan\theta\;
\frac{\mu\varepsilon_{\text{eq}}\;\mu\varepsilon_{\text{ax}}}
{\sqrt{\mu\varepsilon_{\text{eq}}^2\cos^2\varphi + \mu\varepsilon_{\text{ax}}^2\sin^2\varphi}}
$$

which gives $\mu\varepsilon_{\text{ax}}$ along $\mathbf{n}$ and
$\mu\varepsilon_{\text{eq}}$ perpendicular to it. It suits materials with an
obvious single axis, such as layered structures or uniaxially loaded
samples.

**Generalized** (Stephens, 1999). Each crystallite is assumed to have a
slightly different metric, so $M_{hkl} = 1/d_{hkl}^2$ has a variance. For a
quadratic form in $h, k, l$ the variance is a **fourth-order** polynomial:

$$
\sigma^2\!\left(M_{hkl}\right) = \sum_{H+K+L=4} S_{HKL}\; h^H k^K l^L
$$

and the width follows from $\Delta d/d = \tfrac12\,d^2\,\sigma(M_{hkl})$:

$$
\Gamma_M = \frac{0.018}{\pi}\,\tan\theta\; d_{hkl}^2\,
\sqrt{\sum_{H+K+L=4} S_{HKL}\,\Gamma_{HKL}(h,k,l)}
$$

where $\Gamma_{HKL}$ are the symmetry-adapted monomials. The Laue symmetry
fixes which $S_{HKL}$ are independent:

| Laue class | Terms | Parameters |
|---|---|---|
| $m\bar{3}$, $m\bar{3}m$ | 2 | $S_{400}$, $S_{220}$ |
| $6/m$, $6/mmm$, $\bar{3}m1$, $\bar{3}$ | 3 | $S_{400}$, $S_{004}$, $S_{202}$ |
| $\bar{3}1m$ | 4 | + $S_{301}$ |
| $\bar{3}$ (rhombohedral axes) | 4 | $S_{400}$, $S_{220}$, $S_{310}$, $S_{211}$ |
| $4/m$, $4/mmm$ | 4 | $S_{400}$, $S_{004}$, $S_{220}$, $S_{022}$ |
| $mmm$ | 6 | $S_{400}$, $S_{040}$, $S_{004}$, $S_{220}$, $S_{202}$, $S_{022}$ |
| $2/m$ | 9 | 6 above + 3 depending on the unique axis |
| $\bar{1}$ | 15 | all fourth-order terms |

In cubic metals, the $S_{220}/S_{400}$ ratio carries the same information as
the dislocation contrast factors of the modified Williamson–Hall method
(Ungár): it reflects how strongly the dislocation strain fields broaden
$h00$ compared to $hhh$ reflections. The model is phenomenological: it
describes the hkl dependence without assuming a defect type. The fitted
$S_{HKL}$ must give a positive variance in every direction. GSAS-II's GUI
can plot the resulting strain surface to check this.

> **Choosing a model.** Start isotropic. Move to uniaxial or generalized only
> if the difference plot shows that some reflections are systematically too
> broad and others too narrow, and if the change lowers $R_{wp}$
> significantly (a few relative %) with sensible values. A Williamson–Hall
> plot of individual peak widths, coloured by hkl family, shows quickly
> whether the anisotropy follows the crystal axes.

### 9.4 Hydrostatic / deviatoric strain (HStrain)

HStrain adds a small perturbation $D_{ij}$ to the reciprocal metric tensor
terms of section [4.1](#41-braggs-law-and-the-metric-tensor), shifting peak
positions without broadening them:

$$
\frac{1}{d_{hkl}^2} = \sum_j A_j\,m_j(h,k,l) + \Delta_{hkl},
\qquad
\frac{\Delta d}{d} \approx -\tfrac12\,d_{hkl}^2\,\Delta_{hkl}
$$

with $\Delta_{hkl}$ built from the symmetry-allowed terms:

| Laue class | $\Delta_{hkl}$ |
|---|---|
| cubic | $D_{11}(h^2+k^2+l^2) + e_A\,\dfrac{h^2k^2+h^2l^2+k^2l^2}{(h^2+k^2+l^2)^2}$ |
| hexagonal / trigonal | $D_{11}(h^2+k^2+hk) + D_{33}\,l^2$ |
| rhombohedral axes | $D_{11}(h^2+k^2+l^2) + D_{12}(hk+hl+kl)$ |
| tetragonal | $D_{11}(h^2+k^2) + D_{33}\,l^2$ |
| orthorhombic | $D_{11}h^2 + D_{22}k^2 + D_{33}l^2$ |
| monoclinic | orthorhombic + one of $D_{12}hk$, $D_{13}hl$, $D_{23}kl$ |
| triclinic | all six $D_{ij}$ |

All terms except the cubic $e_A$ have exactly the form of a cell change, so
in a single histogram they are fully correlated with the cell. HStrain is
useful when one cell is **shared** between histograms and each histogram
gets its own $D_{ij}$, for example azimuthal sectors of the same ring
(strain scanning), patterns at increasing load, or XRD-CT voxels refined
against a fixed stress-free reference cell. The cubic $e_A$ term is
different: it shifts $h00$ and $hhh$ reflections by different relative
amounts. That is the signature of an elastically anisotropic grain
interaction, or of stacking faults in fcc metals.

### 9.5 Preferred orientation

**March–Dollase.** Exactly as implemented in GSAS-II, for a reflection with
symmetry-equivalent set $\{\mathbf{h}\}$ of size $m$:

$$
P_{hkl} = \frac{1}{m}\sum_{\{\mathbf{h}\}}
\left(r^2\cos^2\alpha_{\mathbf{h}} + \frac{\sin^2\alpha_{\mathbf{h}}}{r}\right)^{-3/2}
$$

where $\alpha_{\mathbf{h}}$ is the angle between $\mathbf{h}$ and the
March–Dollase axis. The function is normalised: integrated over all
orientations it conserves the total intensity, so $r$ redistributes
intensity between reflections and does not change the phase scale.
Physically, $r$ is the ratio by which the sample has been compressed
($r < 1$) or elongated ($r > 1$) along the axis, assuming platy or
needle-like crystallites that align during preparation. Its effect is
symmetric about one direction, so it can only describe a single fibre-like
texture component. Values between 0.7 and 1.3 are common for packed
powders; values far outside this range usually mean the model is wrong or
particle statistics are poor.

**Spherical harmonics.** The general texture correction expands the
orientation distribution up to even order $L$ (`SHord`):

$$
P_{hkl} = 1 + \sum_{l=2,4,\dots}^{L} \frac{4\pi}{2l+1}
\sum_{m}\sum_{n} C_l^{mn}\,\ddot{k}_l^{m}(\mathbf{h})\,\dot{k}_l^{n}(\mathbf{y})
$$

where $\ddot{k}$ are harmonics adapted to the crystal symmetry, evaluated at
the reflection direction $\mathbf{h}$, and $\dot{k}$ are harmonics adapted
to the sample symmetry, evaluated at the sample direction $\mathbf{y}$ of the
scattering vector. The HAP correction in GSAS-II always assumes
**cylindrical** sample symmetry, which reduces the expansion to
coefficients $C_l^{n}$ (named `C(L,N)`) that depend only on the crystal
direction. Their number depends on $L$ and the Laue class; $L = 4$–$8$ is
usually enough. A full ODF with lower sample symmetry is a phase-level
texture analysis in GSAS-II and is not part of the HAP model. GSAS-II
reports the **texture index**

$$
J = 1 + \sum_l \frac{1}{2l+1} \sum_{n} \left\lvert C_l^{n}\right\rvert^2,
$$

which is 1 for a random powder and increases with texture strength. It is a
compact way to compare texture between voxels or samples.

With a single integrated pattern, only the projection of the texture onto
the scattering vectors probed is accessible, so the SH coefficients are not
uniquely determined. Real texture analysis needs several sample
orientations or azimuthal sectors (see [Texture Tomography](texture_theory.md)).
In a single-pattern Rietveld fit, treat the SH correction as an empirical
intensity correction.

### 9.6 Extinction

GSAS-II uses the Sabine (1988) model for powders. For each reflection it
computes

$$
x = E_x\,\lvert F_{hkl}\rvert^2 \left(\frac{\lambda}{V}\right)^2 K_p
$$

where $E_x$ is the refined extinction parameter, $V$ the cell volume and
$K_p$ a polarization factor. The correction combines Bragg-like
($2\theta = 180°$) and Laue-like ($2\theta = 0°$) limits:

$$
E_{hkl} = E_B\,\sin^2\theta + E_L\,\cos^2\theta,
\qquad
E_B = \frac{1}{\sqrt{1+x}}
$$

$$
E_L =
\begin{cases}
1 - \dfrac{x}{2} + \dfrac{x^2}{4} - \dfrac{5x^3}{48} + \dfrac{7x^4}{192} - \dots & x \le 1 \\[2ex]
\sqrt{\dfrac{2}{\pi x}}\left(1 - \dfrac{1}{8x}\right) & x > 1
\end{cases}
$$

Extinction reduces the strongest reflections most, and low-angle ones more
than high-angle ones. Its signature (strong low-angle peaks
over-calculated) can also come from preferred orientation, large-grain
statistics or an incorrect $U_{\text{iso}}$, so check those first.

### 9.7 Babinet solvent correction

In porous or partially disordered materials (zeolites, MOFs, proteins, any
phase with disordered guests filling its voids), the voids are not empty.
They contain a disordered, roughly uniform electron density that cancels
part of the low-angle scattering. Babinet's principle models this by
subtracting a smooth, low-Q-only term from every atomic scattering factor:

$$
f_j \;\rightarrow\; f_j - A_{\text{Bab}}\,
\exp\!\left(-8\pi^2\,U_{\text{Bab}}\,\frac{\sin^2\theta}{\lambda^2}\right)
$$

$A_{\text{Bab}}$ (`BabA`) sets the strength of the solvent contrast and
$U_{\text{Bab}}$ (`BabU`, Å²) sets how fast it fades with angle. It is
typically a few Å², much larger than an atomic $U_{\text{iso}}$, so only
the first few reflections are affected. The correction only makes sense
when the lowest-angle reflections are clearly over-calculated after
background and scale have converged. Refine `BabA` first, then `BabU`.

### 9.8 Phase scale

The HAP `Scale` multiplies the whole contribution of a phase. With a
single phase it is completely redundant with the histogram scale $S_h$, so
only one of them should be refined. With several phases, fix $S_h$ and
refine the phase scales (`refine_phase_content`). Their ratios give the
weight fractions of section [6.6](#66-scale-factors-and-quantitative-phase-analysis).
Phases in Le Bail mode have no meaningful scale (section
[6.7](#67-le-bail-extraction)).

### 9.9 Which HAP term for which misfit

| What the difference plot shows | Likely HAP term | Check first |
|---|---|---|
| All peaks too broad/narrow, error grows with 2θ | isotropic Mustrain | instrument `Y` fixed from calibration |
| All peaks too broad/narrow, error roughly constant in $\cos\theta$ | isotropic Size | instrument `X` fixed from calibration |
| Some hkl families too broad, others too narrow | uniaxial/generalized Mustrain, uniaxial/ellipsoidal Size | Williamson–Hall plot by hkl |
| Derivative-shaped residual on some hkl only | HStrain ($e_A$ for cubic), wrong space group | cell and zero converged |
| Intensity of reflections along one direction off | Pref.Ori. | particle statistics (spotty rings) |
| Strongest low-angle peaks over-calculated | Extinction | Pref.Ori., $U_{\text{iso}}$, large grains |
| First few reflections over-calculated in a porous phase | Babinet | background |
| Peak shape (not width) wrong in tails | `LGmix` | profile model, `SH/L` |

---

# Part II — Using `nrxrdct.rietveld`

## 10. Two classes

| Class | Purpose |
|---|---|
| `BaseRefinement` | General Rietveld refinement for any powder pattern |
| `InstrumentCalibration` | Calibrant-specific subclass: neutralises the calibrant's sample broadening, writes the calibrated `.instprm`, dedicated diagnostic plots |

Both inherit from `Scan` and accept the same base parameters. Each
`refine_*` method frees the relevant parameters and runs **one** GSAS-II
refinement cycle. By default the parameters stay free afterwards. Pass
`freeze=True` (or `freeze_after=True` for the cell) to fix them again after
the cycle.

---

## 11. Instrument calibration

Calibrate the instrument profile and zero using a known standard (LaB₆,
CeO₂, Si). The cell is kept fixed at the certified value (`block_cell=True`)
for the reason given in section [4.4](#44-the-wavelengthcellzero-correlation).

```python
from pathlib import Path
from nrxrdct.rietveld.refinement import InstrumentCalibration

cal = InstrumentCalibration(
    acquisition_file=Path("data/calib.h5"),
    sample_name="LaB6",
    beam_energy=44,                          # keV
    xy_file=Path("integrated_calibrant.xy"),
    param_file=Path("calibrated_instrument.instprm"),
    tth_lims=(3.0, 25.0),
)

# Create new GSAS-II project and add the calibrant.
# Size (10 µm) and Mustrain (0) are set to negligible values and frozen,
# so all observed broadening goes into the instrument parameters.
cal.create_model(gpx_file=Path("calibration/LaB6.gpx"))
cal.add_phase(cif_file=Path("LaB6.cif"), phase_name="LaB6", block_cell=True)

# Step-by-step sequence
cal.refine_background(number_coeff=12)
cal.refine_histogram_scale()
cal.refine_zero_shift()
cal.refine_gaussian_broadening(["W"])
cal.refine_lorentzian_broadening(["X", "Y"])

# Write calibrated instrument parameters to calibration/<param_file>
cal.write_calibrated_instrument_pars()
cal.plot_calibration_results()
```

The same sequence (background → scale → zero → profile → export) is
available as a single call:

```python
cal.refine_instrument_parameters(profile_params=["W", "X", "Y"], use_lebail=True)
```

With `use_lebail=True` the calibrant intensities are extracted Le Bail-style
(section [6.7](#67-le-bail-extraction)), so errors in the calibrant structure
model (e.g. $U_{\text{iso}}$, absorption) cannot leak into the profile
parameters.

For a laboratory source, also refine `U` and `V`, and `SH/L` if the low-angle
peaks are visibly asymmetric. Refine the wavelength (`refine_wavelength`)
only when it is genuinely uncertain, and only with the calibrant cell fixed.

The calibrated `.instprm` file is consumed by all subsequent sample and
per-voxel refinements, with `Zero`, `W`, `X`, `Y` kept fixed.

### Starting values inferred from the data

By default (`infer_instrument_pars=True`) the starting instrument parameters
are estimated from the calibrant pattern rather than taken from generic
defaults, and written to `calibration/instrument_init.instprm`, which
`create_model` builds the project from:

- **Peak widths** (at construction): isolated peaks within `tth_lims` are
  fitted one by one with a pseudo-Voigt. Each FWHM/η is split into Gaussian and
  Lorentzian widths (Thompson-Cox-Hastings, as in GSAS-II), and
  $\sigma^2 = U\tan^2\theta + V\tan\theta + W$ and
  $\gamma = X/\cos\theta + Y\tan\theta + Z$ are fitted to them. Only the terms
  in `inferred_profile_params` (default `["W", "X", "Y"]`) are estimated; the
  other terms of each width law start at 0, so match this list to the
  parameters you will refine.
- **Zero** (in `add_phase`): the fitted peak centres are matched to the
  calibrant's calculated reflections (from its cell, space group and the
  wavelength), searching within ±0.3°.

The per-peak fits are printed and kept in `cal.inferred_peaks`. Both steps
can be rerun by hand with `cal.infer_profile_parameters(...)` and
`cal.infer_zero(max_shift=...)`. Pass `infer_instrument_pars=False` to start
from the generic defaults (`W=1`, all other width terms 0, `Zero=0`) instead.
Width terms that are neither inferred nor refined stay at these defaults, so
they add no broadening.

---

## 12. Sample refinement

### 12.1 Creating a model

```python
from nrxrdct.rietveld.refinement import BaseRefinement

ref = BaseRefinement(
    acquisition_file=Path("data/scan.h5"),
    sample_name="steel",
    beam_energy=44,
    xy_file=Path("integrated_data.xy"),
    param_file=Path("calibration/calibrated_instrument.instprm"),
    tth_lims=(3.0, 25.0),
)

ref.create_model(gpx_file=Path("models/steel.gpx"))
ref.add_phase(cif_file=Path("Fe_bcc.cif"), phase_name="ferrite", block_cell=False)
ref.add_phase(cif_file=Path("Fe3C.cif"),   phase_name="cementite", block_cell=False)
```

### 12.2 Typical sequential refinement

Following the strategy in section [8](#8-refinement-strategy-and-correlations),
with the instrument profile fixed from calibration:

```python
ref.refine_background(number_coeff=8)
ref.refine_histogram_scale()
ref.free_and_refine_cell()                       # all phases
ref.backup_model(label="after_cell")

ref.refine_phase_content()                       # phase scales → weight fractions
ref.refine_crystallite_size(refine_type="isotropic")
ref.refine_mustrain(refine_type="isotropic")
ref.refine_Uiso(freeze=False)
ref.backup_model(label="after_microstructure")

# Final joint cycle: free everything refined so far together
ref.refine_ever_refined_variables()
ref.save()
```

### 12.3 Inspecting results

```python
print(f"Rwp  = {ref.get_Rwp():.2f} %")
print(f"GOF  = {ref.get_chi2():.3f}")       # = sqrt(reduced χ²)
ref.print_refinement_results()
ref.print_HAP_parameters()
ref.plot_results(image_path="steel_fit.png")

# Same plot plus the refinement history
ref.plot_results(image_path="steel_fit.png", history=True)
ref.plot_results(image_path="steel_fit.png", history=True,
                 history_params=["Size", "Mustrain", "Scale"])
```

With `history=True` a panel is added below the fit. It shows $R_{wp}$ and
GOF after every refinement cycle, labelled with the step that was run (e.g.
`W`, `Cell [ferrite]`, `restore after_cell`). Below that is one small plot
per parameter showing how its value changed from cycle to cycle. Filled
markers with error bars are cycles in which the parameter was refined; open
markers are cycles in which it was fixed. This makes it easy to spot a step
that made the fit worse, a parameter that drifts instead of converging, or
two correlated parameters trading values.

By default every refined parameter is shown except background
coefficients, up to `max_history_params` (12). `history_params` selects
parameters by substring of their GSAS-II name. The history covers every
cycle run since the object was created or since the last `load_model`. It
survives `restore_backup`, which appears as its own entry. The raw numbers
are available from `ref.get_parameter_history()`.

---

## 13. Key methods reference

### Range and exclusions

```python
ref.set_limits(3.0, 22.0)                 # active 2θ range
ref.add_excluded_region(8.10, 8.35)       # e.g. a sample-holder peak

# Lab sources with incomplete Kβ filtering
windows = ref.find_kbeta_exclusions(kbeta_wavelength=1.39222, apply=False)
ref.plot_kbeta_exclusions(kbeta_wavelength=1.39222)
```

### Background

```python
ref.refine_background(
    number_coeff=8,
    function="chebyschev",    # "cosine", "Q^2 power series", "lin interpolate", "user", ...
    freeze=False,
)

# Amorphous hump modelled with a Debye term (section 7)
ref.refine_background(
    number_coeff=6,
    debye_terms=[{"A": 1000.0, "R": 4.5, "U": 0.01, "refine_U": False}],
)

# Pre-computed background curve on the same 2θ grid as the data
ref.refine_background(function="user", user_background=bkg_curve)
```

### Peak position

```python
ref.refine_zero_shift(freeze=True)
ref.refine_sample_displacement("DisplaceX")   # "Shift" for Bragg–Brentano
ref.refine_wavelength()                       # calibrant only, cell fixed
```

### Peak profile (instrument)

```python
# Gaussian (U, V, W) — Caglioti variance terms, refined one per cycle
ref.refine_gaussian_broadening(["U", "V", "W"])

# Lorentzian (X, Y) — 1/cosθ and tanθ terms
ref.refine_lorentzian_broadening(["X", "Y"])

# Choose the profile model and refine a list of its parameters
ref.refine_peak_profile(profile="FCJVoigt", parameters=["W", "X", "Y", "SH/L"])
ref.refine_peak_profile(profile="ExpFCJVoigt")   # pink beam

# Inspect / set values directly
ref.print_instrument_parameters()
ref.set_instrument_parameter("SH/L", 0.002, freeze=True)
```

### Unit cell and strain

```python
ref.free_and_refine_cell(phase="ferrite")     # one phase
ref.free_and_refine_cell()                    # all phases
ref.freeze_cell(phase="ferrite")
ref.refine_hstrain(phase="ferrite")           # D_ij terms (sections 4.3, 9.4)
ref.set_HAP_parameter("D11", 0.0, phase="ferrite", freeze=True)
```

### Microstructure (HAP models, section 9)

```python
# Crystallite size — isotropic (section 9.2)
ref.refine_crystallite_size(refine_type="isotropic", phase="ferrite")

# Microstrain — isotropic or generalized Stephens model (section 9.3)
ref.refine_mustrain(refine_type="isotropic", phase="ferrite")
ref.refine_mustrain(refine_type="generalized", phase="ferrite")
```

Calling a `refine_*` method again keeps the current values and continues
from the refined state. Switching model initialises the new parameters from
the current isotropic value (uniaxial: both values; ellipsoidal: a sphere;
generalized: GSAS-II's isotropic-equivalent $S_{HKL}$). For explicit
starting values, axes or mixing coefficients, pass a `refine_dict` with
these keys:

| Key | Meaning |
|---|---|
| `type` | model name |
| `refine` | refine the model parameters |
| `value` | isotropic starting value (µm or µε), also used to initialise the other models |
| `equatorial`, `axial` | uniaxial starting values |
| `axis` (or `direction`) | unique axis `[h, k, l]` |
| `LGmix` | Lorentzian fraction: a float, or `{"value": ..., "refine": ...}` |
| `terms` | ellipsoidal size only: which of `S11` … `S23` to refine (default the diagonal) |

```python
# Uniaxial size about c*: plate-like crystallites, 0.5 µm wide, 0.1 µm thick
ref.refine_crystallite_size(
    refine_dict={"Size": {"type": "uniaxial", "refine": True,
                          "equatorial": 0.5, "axial": 0.1, "axis": [0, 0, 1]}},
    phase="cementite",
)

# Ellipsoidal size, refining the diagonal and the S13 term (monoclinic phase)
ref.refine_crystallite_size(
    refine_dict={"Size": {"type": "ellipsoidal", "refine": True,
                          "terms": ["S11", "S22", "S33", "S13"]}},
    phase="monoclinic_phase",
)

# Uniaxial microstrain about [001]; Mustrain dicts have no top-level key
ref.refine_mustrain(
    refine_dict={"type": "uniaxial", "refine": True, "axis": [0, 0, 1]},
    phase="cementite",
)

# Isotropic microstrain with a fixed 50 % Lorentzian / 50 % Gaussian mix
ref.refine_mustrain(
    refine_dict={"type": "isotropic", "refine": True,
                 "LGmix": {"value": 0.5, "refine": False}},
    phase="ferrite",
)

# Set and fix HAP values (isotropic models only)
ref.set_HAP_parameter("Size", 0.5, phase="cementite", freeze=True)
ref.freeze_HAP_parameter(["Size", "Mustrain"], phase="ferrite")
ref.print_HAP_parameters()
```

### Structure

```python
ref.refine_Uiso(phase="ferrite", freeze=False)
ref.refine_atomic_positions(flags=["U", "XU"], phase="cementite")
ref.refine_occupancy(phase="ferrite", atoms=["Fe1"])
ref.print_atoms()
```

### Intensity corrections

```python
# Preferred orientation (sections 6.3, 9.5)
ref.refine_preferential_orientation(model="MD", phase="ferrite")   # March–Dollase
ref.refine_preferential_orientation(model="SH", phase="ferrite")   # spherical harmonics

# Absorption: μr (cylinder) or μt (flat plate), calculated and fixed
ref.set_absorption(0.05, refine=False)

ref.refine_extinction(phase="ferrite")              # Sabine model (section 9.6)
ref.refine_babinet("BabA", phase="zeolite")         # porous phases (section 9.7)
ref.refine_babinet(["BabA", "BabU"], phase="zeolite")
```

### Le Bail

```python
ref.set_LeBail(phase="unknown_phase", enable=True)
ref.set_LeBail(enable=False)          # back to Rietveld for all phases
```

---

## 14. Backup and restore

```python
# Save a timestamped backup before a risky step
ref.backup_model(label="before_cell")

# List backups
ref.list_backups()

# Restore by index or folder name
ref.restore_backup(-1)                           # most recent
ref.restore_backup("20240115_143022_before_cell")
```

Backups are stored in `bkp_model/<timestamp>[_<label>]/` next to the `.gpx`
file.

---

## 15. Diagnostics and reporting

```python
# Variables free in the last cycle, and everything refined in this session
ref.print_free_variables()
ref.print_ever_refined_variables()

# Correlation matrix of the last cycle (or an earlier one via cycle=...)
ref.print_covariance_matrix()
ref.plot_covariance_matrix()
ref.print_covariance_history()

# Significance (value/esd) and correlation flags per variable
ref.print_variable_diagnostics(significance_threshold=3.0, high_corr_threshold=0.90)

# Exports
ref.export_pattern("steel_fit.csv")     # 2θ, yobs, ycalc, diff, background
ref.export_cif(phase="ferrite")
ref.generate_report(Path("steel_report.pdf"))
```

A parameter whose value is smaller than about 3 esd is not significantly
determined by the data and is usually better fixed. Pairs with
$\lvert\rho\rvert > 0.9$ should not be refined together (section
[2.1](#21-uncertainties-and-correlations)).

---

## 16. Multi-phase workflow

For multiphase samples, pass the `phase` argument to target a specific
phase; omit it to apply to all phases simultaneously:

```python
ref.add_phase(Path("austenite.cif"), phase_name="austenite", block_cell=False)

# Independent cell refinement per phase
ref.free_and_refine_cell(phase="ferrite")
ref.free_and_refine_cell(phase="austenite")

# Phase content (weight fractions from scale factors, section 6.6)
ref.refine_phase_content()
ref.print_HAP_parameters()
wf = ref.weight_fractions()   # {"ferrite": (w, esd), "austenite": (w, esd)}
```

Minor phases (< a few wt %) have only a few weak reflections. Keep their
microstructure parameters fixed, or tied to typical values with
`set_HAP_parameter`, otherwise their scale and width parameters become
strongly correlated and unstable.

---

## 17. Per-voxel refinement in XRD-CT

In XRD-CT every voxel has its own reconstructed powder pattern. The pipeline
calls a user-supplied refinement function for each voxel through
`ReconstructedVolume.refine_models` (sequential) or
`refine_models_parallel` (process pool). The function receives the voxel's
`.xy` file and the `.gpx` file to write:

```python
from pathlib import Path
from nrxrdct.rietveld.refinement import BaseRefinement

def my_refinement(xy_file: Path, gpx_file: Path) -> None:
    ref = BaseRefinement(
        acquisition_file=Path("scan.h5"),
        sample_name="sample",
        xy_file=xy_file,
        param_file=Path("calibration/calibrated_instrument.instprm"),
        tth_lims=(3.0, 25.0),
    )
    ref.create_model(gpx_file=gpx_file)
    ref.add_phase(Path("Fe_bcc.cif"), phase_name="ferrite", block_cell=False)
    ref.refine_background(number_coeff=6)
    ref.refine_histogram_scale()
    ref.free_and_refine_cell()
    ref.refine_mustrain()
    ref.save()

vol.refine_models(my_refinement)
# or, with a module-level (picklable) function:
vol.refine_models_parallel(my_refinement)
```

Considerations specific to tomographic data:

* **No sample-position errors.** The tomographic reconstruction places each
  voxel at its true position, so parallax and displacement shifts that affect
  bulk measurements of thick samples are removed. Keep `Zero` and
  displacement fixed from the calibration.
* **Weights are approximate.** Reconstructed intensities are not counts, so
  the absolute GOF is not meaningful (section [2.2](#22-where-the-weights-come-from)).
  Compare $R_{wp}$ and parameter maps between voxels, not against a GOF of 1.
* **Scale factors are relative.** Phase *fractions* are meaningful per
  voxel. Absolute scale maps reflect reconstructed density and
  absorption.
* **Keep the model minimal.** Thousands of independent refinements have to
  converge without supervision. Free only the parameters that the voxel
  data can support (typically background, scale, cell, one broadening term),
  use a converged bulk model as the starting point, and inspect maps of
  $R_{wp}$ to find voxels where the fit failed.

`get_step_refinements` / `save_step_refinements` record the sequence of
steps run interactively on one representative pattern, and
`apply_step_refinements` replays it on another project. This keeps the
per-voxel recipe identical to the one tested by hand.

See [Typical Workflow](workflow.md) for the full pipeline context.

---

## 18. Refinement dictionary templates

`nrxrdct.rietveld.refine_dict` provides pre-built model dictionaries for
common cases:

```python
from nrxrdct.rietveld.refine_dict import (
    MD_DICT, SH_DICT,                                    # preferred orientation
    SIZE_ISO_DICT, SIZE_UNI_DICT, SIZE_ELL_DICT,         # crystallite size
    MUSTRAIN_ISO_DICT, MUSTRAIN_UNI_DICT, MUSTRAIN_GEN_DICT,
)
```

They are the defaults of `refine_crystallite_size`, `refine_mustrain` and
`refine_preferential_orientation`. They carry no starting values, so
repeated calls continue from the current state. Pass a modified copy through
`refine_dict=` / `parsMD=` / `parsSH=` to set starting values, axes or the
SH order (keys in section [13](#13-key-methods-reference)). `SIZE_GEN_DICT` is
kept as an alias of `SIZE_ELL_DICT`.

Do not pass these dictionaries to GSAS-II's `phase.set_HAP_refinements`
directly. It reads only part of them: for `Pref.Ori.` it sets only the
refine flag (model, ratio, axis and order are ignored), and for
Size/Mustrain it ignores `equatorial`/`axial`. The `BaseRefinement` methods
write the full model definition into the project and record it in the step
recipe (`"hap_model"`), which `apply_step_refinements` replays.

GSAS-II parameter names are documented at
<https://gsas-ii.readthedocs.io/en/latest/objvarorg.html#parameter-names-in-gsas-ii>.

---

## References

* H. M. Rietveld, "A profile refinement method for nuclear and magnetic
  structures", *J. Appl. Cryst.* **2**, 65–71 (1969).
* B. H. Toby, R. B. Von Dreele, "GSAS-II: the genesis of a modern
  open-source all purpose crystallography software package", *J. Appl. Cryst.*
  **46**, 544–549 (2013).
* L. B. McCusker, R. B. Von Dreele, D. E. Cox, D. Louër, P. Scardi,
  "Rietveld refinement guidelines", *J. Appl. Cryst.* **32**, 36–50 (1999).
* B. H. Toby, "R factors in Rietveld analysis: How good is good enough?",
  *Powder Diffraction* **21**, 67–70 (2006).
* P. Thompson, D. E. Cox, J. B. Hastings, "Rietveld refinement of
  Debye–Scherrer synchrotron X-ray data from Al₂O₃", *J. Appl. Cryst.* **20**,
  79–83 (1987).
* L. W. Finger, D. E. Cox, A. P. Jephcoat, "A correction for powder
  diffraction peak asymmetry due to axial divergence", *J. Appl. Cryst.* **27**,
  892–900 (1994).
* P. W. Stephens, "Phenomenological model of anisotropic peak broadening in
  powder diffraction", *J. Appl. Cryst.* **32**, 281–289 (1999).
* W. A. Dollase, "Correction of intensities for preferred orientation in
  powder diffractometry", *J. Appl. Cryst.* **19**, 267–272 (1986).
* R. B. Von Dreele, "Quantitative texture analysis by Rietveld refinement",
  *J. Appl. Cryst.* **30**, 517–525 (1997).
* R. J. Hill, C. J. Howard, "Quantitative phase analysis from neutron powder
  diffraction data using the Rietveld method", *J. Appl. Cryst.* **20**,
  467–474 (1987).
* T. M. Sabine, R. B. Von Dreele, J.-E. Jørgensen, "Extinction in
  time-of-flight neutron powder diffractometry", *Acta Cryst.* **A44**,
  374–379 (1988).
* T. Ungár, A. Borbély, "The effect of dislocation contrast on X-ray line
  broadening: a new approach to line profile analysis", *Appl. Phys. Lett.*
  **69**, 3173–3175 (1996).
* G. K. Williamson, W. H. Hall, "X-ray line broadening from filed aluminium
  and wolfram", *Acta Metall.* **1**, 22–31 (1953).
