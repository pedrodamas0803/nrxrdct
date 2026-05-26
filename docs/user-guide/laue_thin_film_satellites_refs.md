# References — thin-film satellites in Laue diffraction

Draft references for `laue_thin_film_satellites.md`.
**Please verify DOIs and page numbers before merging.**

---

## Textbooks

1. **Warren, B. E.**  
   *X-Ray Diffraction.*  
   Dover Publications, New York, 1990.  (Unabridged reprint of the 1969 Addison-Wesley edition.)  
   *Chapter 3 derives the Laue interference function and the positions and
   intensities of subsidiary maxima between Bragg peaks — the canonical
   reference for the formulas in Section 1.*

2. **Als-Nielsen, J. & McMorrow, D.**  
   *Elements of Modern X-ray Physics*, 2nd ed.  
   Wiley, Chichester, 2011.  ISBN 978-0-470-97395-0.  
   *Chapter 3 (kinematical scattering) gives a clear modern derivation of
   the single-slab structure factor and the thin-film fringe spacing.*

3. **Authier, A.**  
   *Dynamical Theory of X-Ray Diffraction.*  
   Oxford University Press, Oxford, 2001.  ISBN 978-0-19-855960-2.  
   *Chapter 1 covers the kinematical (Born) limit, which is the approximation
   used throughout `simulate_laue_stack`.*

4. **Guinier, A.**  
   *X-Ray Diffraction in Crystals, Imperfect Crystals, and Amorphous Bodies.*  
   Dover Publications, New York, 1994.  (Reprint of the 1963 W. H. Freeman edition.)  
   *A comprehensive account of the geometric series structure factor and its
   relation to crystal size and shape.*

---

## Superlattice satellites and multilayer X-ray diffraction

5. **Segmüller, A. & Blakeslee, A. E.**  
   X-ray diffraction from one-dimensional superlattices in GaAs$_{1-x}$P$_x$ crystals.  
   *J. Appl. Crystallogr.* **6**, 19–24 (1973).  
   DOI: [10.1107/S0021889873008228](https://doi.org/10.1107/S0021889873008228)  
   ⚠️ *Verify DOI — journal was not online in 1973; this may be a retroactive identifier.*  
   *Seminal paper demonstrating superlattice satellites in compound
   semiconductors and relating the satellite spacing to the bilayer period $\Lambda$.*

6. **Bartels, W. J., Hornstra, J. & Lobeek, D. J. W.**  
   X-ray diffraction of multilayers and superlattices.  
   *Acta Cryst.* **A42**, 539–545 (1986).  
   DOI: [10.1107/S0108767386098768](https://doi.org/10.1107/S0108767386098768)  
   ⚠️ *Verify DOI.*  
   *Derives the kinematical structure factor for a periodic bilayer stack and
   the selection rules for superlattice satellites.*

7. **Fullerton, E. E., Schuller, I. K., Vanderstraeten, H. & Bruynseraede, Y.**  
   Structural refinement of superlattices from x-ray diffraction.  
   *Phys. Rev. B* **45**, 9292–9310 (1992).  
   DOI: [10.1103/PhysRevB.45.9292](https://doi.org/10.1103/PhysRevB.45.9292)  
   ⚠️ *Verify page numbers.*  
   *Provides a complete kinematical model for fitting satellite intensities and
   extracting individual layer thicknesses and interface roughness.*

---

## White-beam Laue diffraction and LaueTools

8. **Robach, O., Micha, J.-S., Ulrich, O. & Gergaud, P.**  
   Full local elastic strain tensor from Laue microdiffraction: simultaneous
   Laue pattern and scanning-electron-microscopy measurements.  
   *J. Appl. Cryst.* **44**, 688–696 (2011).  
   DOI: [10.1107/S0021889811003099](https://doi.org/10.1107/S0021889811003099)  
   ⚠️ *Verify DOI — may be S0021889811003099 or similar.*  
   *Describes the LaueTools framework (frame conventions, calibration
   parameters, indexation) on which `nrxrdct.laue` is built.*

9. **Chung, J.-S. & Ice, G. E.**  
   Automated indexing for texture and strain measurement with broad-bandpass
   x-ray microbeams.  
   *J. Appl. Phys.* **86**, 5249–5255 (1999).  
   DOI: [10.1063/1.371507](https://doi.org/10.1063/1.371507)  
   ⚠️ *Verify DOI.*  
   *Introduces the orientation-matrix conventions (LT frame, `matstarlab`)
   used by LaueTools and this package.*

---

## GaN / III-nitride thin films

10. **Metzger, T. H. *et al.***  
    X-ray diffraction study of InGaN/GaN superlattices on GaN/(0001) sapphire.  
    *Phil. Mag. A* **77**, 1013–1025 (1998).  
    DOI: [10.1080/01418619808221234](https://doi.org/10.1080/01418619808221234)  
    ⚠️ *Verify authors, volume, and pages.*  
    *Reports thickness fringes and superlattice satellites from InGaN/GaN
    multilayers grown along $[0001]$ — directly analogous to the structures
    modelled by `simulate_laue_stack`.*

11. **Vickers, M. E. *et al.***  
    Determination of InGaN layer thicknesses in InGaN/GaN quantum well
    structures by x-ray reflectometry and scattering.  
    *J. Appl. Phys.* **94**, 1559–1566 (2003).  
    DOI: [10.1063/1.1586996](https://doi.org/10.1063/1.1586996)  
    ⚠️ *Verify authors and page numbers.*  
    *Demonstrates extraction of quantum-well thicknesses from fringe spacing,
    providing experimental validation of the $2\pi/t$ fringe-period formula.*
