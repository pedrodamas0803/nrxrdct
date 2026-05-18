# References — diffraction and Laue theory

Draft references for `laue_theory.md`.
**Please verify DOIs and page numbers before merging.**

---

## Textbooks — general X-ray diffraction

1. **Warren, B. E.**
   *X-Ray Diffraction.*
   Dover Publications, New York, 1990.  (Unabridged reprint of the 1969 Addison-Wesley edition.)
   *Chapters 1–4 cover kinematical scattering, the structure factor, systematic
   absences, and the Lorentz–polarisation factor — the canonical reference for
   Sections 1–4.*

2. **Als-Nielsen, J. & McMorrow, D.**
   *Elements of Modern X-ray Physics*, 2nd ed.
   Wiley, Chichester, 2011.  ISBN 978-0-470-97395-0.
   *Chapters 1–3 give a clear modern derivation of scattering amplitudes,
   anomalous dispersion, and the Ewald sphere construction.*

3. **Authier, A.**
   *Dynamical Theory of X-Ray Diffraction.*
   Oxford University Press, Oxford, 2001.  ISBN 978-0-19-855960-2.
   *Chapter 1 reviews the kinematical (Born) limit used throughout `nrxrdct.laue`.*

4. **Cullity, B. D. & Stock, S. R.**
   *Elements of X-Ray Diffraction*, 3rd ed.
   Pearson/Prentice Hall, Upper Saddle River NJ, 2001.  ISBN 978-0-201-61091-0.
   *A pedagogical introduction to Bragg's law, the reciprocal lattice, and
   systematic absences, suitable as background reading for Section 1–2.*

5. **Hammond, C.**
   *The Basics of Crystallography and Diffraction*, 4th ed.
   Oxford University Press, Oxford, 2015.  ISBN 978-0-19-873868-8.
   *Covers direct and reciprocal lattices, Laue and powder methods, and the
   gnomonic projection (Section 7.1).*

---

## Synchrotron radiation and source spectra

6. **Kim, K.-J.**
   Characteristics of synchrotron radiation.
   In *AIP Conference Proceedings* **184**, 565–632 (1989).
   DOI: [10.1063/1.38046](https://doi.org/10.1063/1.38046)
   ⚠️ *Verify volume and page range.*
   *Derives the on-axis bending-magnet / wiggler spectral flux formula
   (the `spectrum_bm` function) used in Section 4.3.*

7. **Attwood, D.**
   *Soft X-Rays and Extreme Ultraviolet Radiation.*
   Cambridge University Press, Cambridge, 1999.  ISBN 978-0-521-02997-1.
   *Chapter 5 covers synchrotron radiation characteristics, critical energy,
   and undulator harmonics.*

---

## KB mirrors and X-ray optics

8. **Kirkpatrick, P. & Baez, A. V.**
   Formation of optical images by X-rays.
   *J. Opt. Soc. Am.* **38**, 766–774 (1948).
   DOI: [10.1364/JOSA.38.000766](https://doi.org/10.1364/JOSA.38.000766)
   *The original paper introducing the KB mirror geometry (Section 4.4).*

9. **Névot, L. & Croce, P.**
   Caractérisation des surfaces par réflexion rasante de rayons X.
   Application à l'étude du polissage de quelques verres silicates.
   *Rev. Phys. Appl.* **15**, 761–779 (1980).
   DOI: [10.1051/rphysap:01980001503076100](https://doi.org/10.1051/rphysap:01980001503076100)
   ⚠️ *Verify DOI.*
   *Introduces the roughness damping factor for mirror reflectivity used in
   `kb_reflectivity`.*

10. **Parratt, L. G.**
    Surface studies of solids by total reflection of X-rays.
    *Phys. Rev.* **95**, 359–369 (1954).
    DOI: [10.1103/PhysRev.95.359](https://doi.org/10.1103/PhysRev.95.359)
    *Fresnel reflectivity near the critical angle for total external reflection —
    the basis of the single-mirror reflectivity calculation.*

---

## White-beam Laue diffraction and LaueTools

11. **Robach, O., Micha, J.-S., Ulrich, O. & Gergaud, P.**
    Full local elastic strain tensor from Laue microdiffraction: simultaneous
    Laue pattern and scanning-electron-microscopy measurements.
    *J. Appl. Cryst.* **44**, 688–696 (2011).
    DOI: [10.1107/S0021889811003099](https://doi.org/10.1107/S0021889811003099)
    ⚠️ *Verify DOI.*
    *Describes the LaueTools framework: frame conventions, detector calibration,
    and the `matstarlab` orientation representation used throughout `nrxrdct.laue`.*

12. **Chung, J.-S. & Ice, G. E.**
    Automated indexing for texture and strain measurement with broad-bandpass
    x-ray microbeams.
    *J. Appl. Phys.* **86**, 5249–5255 (1999).
    DOI: [10.1063/1.371507](https://doi.org/10.1063/1.371507)
    ⚠️ *Verify DOI.*
    *Introduces the inter-spot angle matching strategy for automated Laue
    indexing described in Section 7.2.*

13. **Ice, G. E., Budai, J. D. & Pang, J. W. L.**
    The race to X-ray microbeam and nanobeam science.
    *Science* **334**, 1234–1239 (2011).
    DOI: [10.1126/science.1202366](https://doi.org/10.1126/science.1202366)
    *Reviews white-beam Laue microdiffraction for strain and orientation mapping,
    providing motivation for the `simulate_laue_stack` approach.*

---

## Strain analysis from Laue patterns

14. **Tamura, N. *et al.***
    Scanning X-ray microdiffraction with submicrometer white beam for strain/stress
    and orientation mapping in thin films.
    *J. Synchrotron Rad.* **10**, 137–143 (2003).
    DOI: [10.1107/S0909049502021362](https://doi.org/10.1107/S0909049502021362)
    ⚠️ *Verify authors and page numbers.*
    *Describes the deviatoric strain determination from Laue peak-shift Jacobians
    (Section 8.2) and the insensitivity to hydrostatic strain.*

15. **Barabash, R. I. & Ice, G. E.** (eds.)
    *Strain and Dislocation Gradients from Diffraction.*
    Imperial College Press, London, 2014.  ISBN 978-1-908979-62-9.
    ⚠️ *Verify ISBN.*
    *Comprehensive treatment of deviatoric vs. hydrostatic strain separation and
    the Laue diffraction strain tensor formalism.*

---

## Anomalous scattering (form factors)

16. **Henke, B. L., Gullikson, E. M. & Davis, J. C.**
    X-ray interactions: photoabsorption, scattering, transmission, and reflection
    at E = 50–30,000 eV, Z = 1–92.
    *At. Data Nucl. Data Tables* **54**, 181–342 (1993).
    DOI: [10.1006/adnd.1993.1013](https://doi.org/10.1006/adnd.1993.1013)
    *The tabulated $f'(E)$ and $f''(E)$ anomalous form factors used by
    xrayutilities and referenced in Section 1.1 and 4.1.*
