# laue

White-beam synchrotron Laue diffraction simulation — single crystals, mixed phases,
coherent layered superlattices, and dynamical (Darwin) intensity correction.

The module covers the full simulation pipeline:

1. **Crystal construction** — from built-in helpers or CIF files.
2. **Orientation** — Bunge Euler angles, LaueTools `matstarlab`, or explicit rotation matrices.
3. **Camera / detector** — LaueTools-compatible pixelated area detector.
4. **Simulation** — single crystal (`simulate_laue`), coherent layer stack (`simulate_laue_stack`),
   or Darwin-corrected multilayer (`simulate_laue_darwin`).
5. **Layered structures** — `LayeredCrystal` for superlattice / multilayer stacks with
   pseudomorphic strain, elastic constants, and lattice-parameter / strain profiling.
6. **Strain analysis** — spot Jacobians, strain broadening, instrument broadening fitting.
7. **Plotting** — 2θ/χ maps, detector images, comparison panels, strain-broadening overlays.

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

::: nrxrdct.laue.simulate_laue_darwin

::: nrxrdct.laue.print_spot_table

::: nrxrdct.laue.print_bragg_table

::: nrxrdct.laue.print_hkl_family

---

## Layered structures

::: nrxrdct.laue.LayeredCrystal

::: nrxrdct.laue.Layer

::: nrxrdct.laue.d_spacing_hkl

::: nrxrdct.laue.pseudomorphic_d_spacing

::: nrxrdct.laue.nitride_elastic_constants

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

::: nrxrdct.laue.measure_spot_widths

::: nrxrdct.laue.estimate_instrument_broadening

::: nrxrdct.laue.fit_strain_distribution

---

## Physics helpers

::: nrxrdct.laue.lorentz_pol

::: nrxrdct.laue.en2lam

::: nrxrdct.laue.lam2en

::: nrxrdct.laue.kb_reflectivity

::: nrxrdct.laue.BM32_KB

---

## Plotting

::: nrxrdct.laue.plot_2theta_chi

::: nrxrdct.laue.plot_all

::: nrxrdct.laue.plot_compare_spots

::: nrxrdct.laue.plot_laue_comparison

::: nrxrdct.laue.plot_interactive_tth_chi

::: nrxrdct.laue.plot_tth_chi_overlay

::: nrxrdct.laue.plot_layer_scheme

::: nrxrdct.laue.plot_laue_stack_spots

::: nrxrdct.laue.plot_strain_broadening

::: nrxrdct.laue.warp_image_to_tth_chi
