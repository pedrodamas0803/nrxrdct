# laue

White-beam synchrotron Laue diffraction simulation — single crystals, mixed phases,
and coherent layered superlattices.

The module covers the full simulation pipeline:

1. **Crystal construction** — from built-in helpers or CIF files.
2. **Orientation** — Bunge Euler angles, LaueTools `matstarlab`, or explicit rotation matrices.
3. **Camera / detector** — LaueTools-compatible pixelated area detector.
4. **Simulation** — single crystal (`simulate_laue`) or coherent layer stack (`simulate_laue_stack`).
5. **Layered structures** — `LayeredCrystal` for superlattice / multilayer stacks.
6. **Strain analysis** — spot Jacobians, strain broadening, instrument broadening fitting.
7. **Plotting** — 2θ/χ maps, detector images, strain-broadening overlays.

---

## Camera

::: nrxrdct.laue.Camera

---

## Crystal utilities

::: nrxrdct.laue.crystal_from_cif

::: nrxrdct.laue.crystals_from_cifs

::: nrxrdct.laue.build_bcc

::: nrxrdct.laue.build_b2

---

## Simulation

::: nrxrdct.laue.simulate_laue

::: nrxrdct.laue.simulate_laue_stack

::: nrxrdct.laue.print_spot_table

::: nrxrdct.laue.print_bragg_table

---

## Layered structures

::: nrxrdct.laue.LayeredCrystal

::: nrxrdct.laue.Layer

---

## Orientation utilities

::: nrxrdct.laue.euler_to_U

::: nrxrdct.laue.U_from_matstarlab

::: nrxrdct.laue.decompose_matstarlab

::: nrxrdct.laue.beam_in_crystal

::: nrxrdct.laue.orientation_along_z

::: nrxrdct.laue.or_from_directions

::: nrxrdct.laue.or_kurdjumov_sachs

::: nrxrdct.laue.or_nishiyama_wassermann

::: nrxrdct.laue.or_baker_nutting

::: nrxrdct.laue.or_pitsch

---

## Synchrotron spectra

::: nrxrdct.laue.spectrum_bm

::: nrxrdct.laue.spectrum_undulator

::: nrxrdct.laue.synchrotron_spectrum

---

## Strain analysis

::: nrxrdct.laue.strain_spot_jacobian

::: nrxrdct.laue.strain_broadening

::: nrxrdct.laue.estimate_instrument_broadening

::: nrxrdct.laue.fit_strain_distribution

---

## Physics helpers

::: nrxrdct.laue.lorentz_pol

::: nrxrdct.laue.en2lam

::: nrxrdct.laue.lam2en

::: nrxrdct.laue.is_superlattice

---

## Plotting

::: nrxrdct.laue.plot_2theta_chi

::: nrxrdct.laue.plot_all

::: nrxrdct.laue.plot_strain_broadening
