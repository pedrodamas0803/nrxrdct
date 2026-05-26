"""
Miscellaneous utility functions.

Covers zinger (hot-pixel) removal, percentile estimation for spot removal,
circular mask generation, absorption coefficient estimation, and array padding
helpers used by the reconstruction pipeline.

Powder XRD simulation and baseline fitting have moved to
:mod:`nrxrdct.powder.simulation`.
"""
import concurrent.futures
import os
import time
from typing import Optional, Tuple

import numpy as np
import scipy.ndimage as ndi
import xraylib

from nrxrdct.azimuthal.integration import _get_integrator, azimuthal_integration_1d

NTHREAD = os.cpu_count() - 1


def zinger_remove(dimg, medsize=3, nsigma=5) -> np.ndarray:
    """
    Remove zingers (hot pixels) from a single 2-D detector image.

    Pixels whose deviation from the local median exceeds ``nsigma`` standard
    deviations (computed over the whole residual image) are replaced by the
    median-filtered value.  The replacement region is dilated by one pixel to
    catch ringing artefacts around bright zingers.

    Args:
        dimg (np.ndarray): 2-D detector image.
        medsize (int, optional): Side length of the square median-filter kernel (default 3).
        nsigma (int or float, optional): Sigma threshold above which a pixel is considered
            a zinger (default 5).

    Returns:
        np.ndarray: Cleaned image with the same shape and dtype as *dimg*.
    """
    med = ndi.median_filter(dimg, medsize)
    err = dimg - med
    ds0 = err.std()
    msk = err > ds0 * nsigma
    gromsk = ndi.binary_dilation(msk)
    return np.where(gromsk, med, dimg)


def dezinger(image, medsize: int = 3, nsigma: int = 5) -> np.ndarray:
    """
    Apply zinger removal to a stack of detector images in parallel.

    Each frame is processed independently via :func:`zinger_remove` using a
    thread pool sized to ``os.cpu_count() - 1``.

    Args:
        image (np.ndarray): 3-D array of shape ``(N, rows, cols)`` containing *N* detector frames.
        medsize (int, optional): Median-filter kernel size passed to :func:`zinger_remove`
            (default 3).
        nsigma (int, optional): Sigma threshold passed to :func:`zinger_remove` (default 5).

    Returns:
        np.ndarray: Dezingered stack with the same shape and dtype as *image*.
    """
    t0 = time.time()
    N = image.shape[0]

    def dezing(im):
        return zinger_remove(im, medsize, nsigma)

    print(f"Will dezinger {N} images. Might take few seconds.")
    out_image = np.zeros_like(image)
    with concurrent.futures.ThreadPoolExecutor(NTHREAD) as pool:
        for i, result in enumerate(pool.map(dezing, image)):
            out_image[i] = result
    t1 = time.time()

    print(f"It took {(t1-t0):.2f}s to dezinger {N} images.")
    return out_image



def estimate_percentile_from_separate(
    image: np.ndarray,
    poni_file: str,
    npt: int = 1000,
    unit: str = "2th_deg",
    mask: Optional[np.ndarray] = None,
    dark: Optional[np.ndarray] = None,
    flat: Optional[np.ndarray] = None,
    radial_range: Optional[Tuple[float, float]] = None,
    azimuth_range: Optional[Tuple[float, float]] = None,
    npt_azim: int = 360,
    threshold: float = 0.95,
    plot: bool = False,
) -> Tuple[float, float]:
    """
    Estimate the optimal (low, high) percentile bounds for single-crystal
    spot removal by analysing the amorphous/polycrystalline background
    separated by pyFAI's `AzimuthalIntegrator.separate`.

    The strategy is:
        1. Run `ai.separate` to decompose the image into a crystalline
           (Bragg spots) component and a smooth amorphous/powder background.
        2. For each radial bin, compute the azimuthal intensity distribution
           of the *original* image and find the fraction of pixels that lie
           within the background signal range.
        3. The low percentile is set so that `threshold` of the background
           pixels are retained; the high percentile clips the Bragg-spot tail.

    Args:
        image (np.ndarray): 2D detector image.
        poni_file (str): Path to the PONI calibration file.
        npt (int): Number of radial bins (default: 1000).
        unit (str): Radial unit (default: "2th_deg").
        mask (np.ndarray, optional): Detector mask (1 = masked, 0 = valid).
        dark (np.ndarray, optional): Dark-current image.
        flat (np.ndarray, optional): Flat-field image.
        radial_range (tuple, optional): Radial range to consider.
        azimuth_range (tuple, optional): Azimuthal range in degrees.
        npt_azim (int): Number of azimuthal bins used internally by `separate`
            (default: 360).
        threshold (float): Fraction of background pixels to retain when setting
            the high percentile (default: 0.95). Lower values clip more aggressively.
        plot (bool): If True, plot the original vs background 1D patterns and mark
            the estimated percentile cut-offs.

    Returns:
        low_percentile (float): Recommended lower percentile bound (always 0.0).
        high_percentile (float): Recommended upper percentile bound in [0, 100] that
            clips Bragg spots while retaining `threshold` of the background signal.

    Note:
        `ai.separate` is relatively slow (it runs a full 2D integration internally).
        Call this function on a representative image to obtain the percentile estimate,
        then reuse the result across all frames via `azimuthal_integration_1d_filter`
        or `integrate_powder_parallel`.

    Example:
        >>> low, high = estimate_percentile_from_separate(image, "det.poni")
        >>> print(f"Recommended percentile: ({low:.1f}, {high:.1f})")
        >>> q, I, _ = azimuthal_integration_1d_filter(
        ...     image, "det.poni", percentile=(low, high)
        ... )
    """
    if image.ndim != 2:
        raise ValueError(f"image must be 2D, got shape {image.shape}")

    ai = _get_integrator(poni_file)

    # ── Step 1: separate crystalline and amorphous components ─────────────────
    print("Running ai.separate (this may take a moment)...")
    crystalline, amorphous = ai.separate(
        image,
        npt,
        unit=unit,
        mask=mask,
    )

    # result_separate returns (crystalline_image, amorphous_image)
    # both are 2D arrays in detector space
    # crystalline = result_separate.crystalite  # Bragg-spot component
    # amorphous = result_separate.amorphous  # smooth background component

    # ── Step 2: 1D integrate both components for diagnostics ─────────────────
    _, I_full, _ = azimuthal_integration_1d(
        image, poni_file, npt=npt, unit=unit, mask=mask
    )
    _, I_bg, _ = azimuthal_integration_1d(
        amorphous, poni_file, npt=npt, unit=unit, mask=mask
    )
    _, I_cryst, _ = azimuthal_integration_1d(
        crystalline, poni_file, npt=npt, unit=unit, mask=mask
    )

    # ── Step 3: compute the ratio of crystalline to total signal per bin ──────
    with np.errstate(divide="ignore", invalid="ignore"):
        bragg_fraction = np.where(I_full > 0, I_cryst / I_full, 0.0)

    # ── Step 4: derive the high percentile ───────────────────────────────────
    # In bins dominated by background (low bragg_fraction), the amorphous
    # image and the original image are nearly identical. We look at the
    # distribution of (amorphous / original) pixel ratios across the whole
    # image (valid pixels only) to find where `threshold` fraction of
    # background pixels sit.

    valid = (mask == 0) if mask is not None else np.ones(image.shape, dtype=bool)

    with np.errstate(divide="ignore", invalid="ignore"):
        pixel_ratio = np.where(image > 0, amorphous / image, 0.0)

    # Only consider pixels where the background model is meaningful (>0)
    meaningful = valid & (amorphous > 0) & (image > 0)
    ratios = pixel_ratio[meaningful]

    # The high percentile is the point where `threshold` of the background
    # pixels have ratio >= cut (i.e. are well-described by the bg model)
    high_percentile = float(np.percentile(ratios, threshold * 100))

    # Convert the ratio threshold to a percentile in [0, 100]
    # A ratio close to 1.0 means the pixel IS background; we want to clip
    # pixels where the original >> amorphous (i.e. ratio << 1 → Bragg spots)
    # Map: high_percentile ratio → percentile on intensity scale
    high_pct = float(np.clip(high_percentile * 100, 50.0, 99.0))
    low_pct = 0.0

    # Summary statistics
    mean_bragg_fraction = float(np.mean(bragg_fraction))
    print(f"\nSeparation summary:")
    print(f"  Mean Bragg fraction across radial bins : {mean_bragg_fraction:.3f}")
    print(
        f"  Background pixel ratio @ {threshold*100:.0f}th pct : {high_percentile:.4f}"
    )
    print(f"  → Recommended percentile range         : ({low_pct:.1f}, {high_pct:.1f})")

    return low_pct, high_pct




def generate_circular_mask(shape, center, diameter) -> np.ndarray:
    """
    Generate a 2-D boolean circular mask for a given image or volume shape.

    Args:
        shape (tuple): Shape of the array.  Both 2-D ``(ny, nx)`` and 3-D
            ``(n_tth, ny, nx)`` (or any shape with at least 2 dimensions) are
            accepted; the mask is always built over the **last two** dimensions.
        center (tuple): ``(cx, cy)`` centre of the circle in pixel coordinates,
            where *cx* indexes along the last axis (columns) and *cy* along the
            second-to-last axis (rows).
        diameter (int or float): Diameter of the circular region in pixels.

    Returns:
        np.ndarray: 2-D boolean array of shape ``(ny, nx)`` where ``True`` marks
            pixels inside the circle.
    """
    ny, nx = shape[-2], shape[-1]
    x, y = np.arange(nx), np.arange(ny)
    X, Y = np.meshgrid(x, y)
    z = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
    return z < diameter / 2


def calculate_absorption_coefficient(
    compound: str,
    density: float,
    energy_keV: float,
    sample_diameter_mm: float,
    geometry: str = "debye-scherrer",
) -> dict[str, float]:
    """
    Estimate the GSAS-II absorption parameter for a powder sample using xraylib.

    The function computes the linear attenuation coefficient μ (cm⁻¹) from the
    mass attenuation coefficient and density, then converts it to the
    dimensionless parameter expected by GSAS-II for the chosen geometry.

    Args:
        compound (str): Chemical formula understood by xraylib, e.g. ``"Fe2O3"``,
            ``"LaB6"``, ``"Al"``. Elements and simple compounds are supported; use
            standard Hill notation (C first, H second, then alphabetical).
        density (float): Sample density in g/cm³. For a packed powder bed this should
            be the effective (packing) density, not the crystallographic density.
        energy_keV (float): Incident X-ray energy in keV.
        sample_diameter_mm (float): Sample diameter (or thickness for flat-plate) in mm.
            For a capillary this is the outer diameter; for a flat-plate it is the
            thickness.
        geometry (str, optional): Sample geometry (default ``"debye-scherrer"``).
            ``"debye-scherrer"`` — cylindrical capillary in transmission; GSAS-II
            expects the dimensionless product μr (μ in cm⁻¹ × radius in cm).
            ``"flat-plate"`` — flat-plate geometry; GSAS-II expects μt
            (μ in cm⁻¹ × thickness in cm).

    Returns:
        dict: A dictionary with keys ``"mu_cm"`` (float, μ in cm⁻¹),
            ``"mu_mm"`` (float, μ in mm⁻¹), ``"mass_atten_cm2_g"`` (float, μ/ρ
            in cm²/g from xraylib), and ``"gsasii_absorption"`` (float,
            dimensionless GSAS-II Absorption parameter μr or μt).

    Note:
        xraylib's ``CS_Total_CP`` returns the *total* mass attenuation coefficient
        (photoelectric + Compton + Rayleigh) in cm²/g. For multi-phase samples the
        effective μ can be approximated as the volume-weighted average of the individual
        phase μ values.

    Example:
        >>> result = calculate_absorption_coefficient(
        ...     compound="Al2O3", density=3.99, energy_keV=44.0,
        ...     sample_diameter_mm=0.3, geometry="debye-scherrer"
        ... )
        >>> print(f"μ = {result['mu_cm']:.2f} cm⁻¹")
        >>> print(f"GSAS-II Absorption (μr) = {result['gsasii_absorption']:.4f}")
    """
    valid_geometries = {"debye-scherrer", "flat-plate"}
    if geometry not in valid_geometries:
        raise ValueError(
            f"Unknown geometry '{geometry}'. Valid options: {sorted(valid_geometries)}"
        )

    mass_atten = xraylib.CS_Total_CP(compound, energy_keV)  # cm²/g
    mu_cm = mass_atten * density                              # cm⁻¹
    mu_mm = mu_cm / 10.0                                      # mm⁻¹

    # Convert sample_diameter_mm to cm for GSAS-II parameter
    if geometry == "debye-scherrer":
        radius_cm = (sample_diameter_mm / 2.0) / 10.0
        gsasii_absorption = mu_cm * radius_cm                 # μr, dimensionless
    else:  # flat-plate
        thickness_cm = sample_diameter_mm / 10.0
        gsasii_absorption = mu_cm * thickness_cm              # μt, dimensionless

    print(f"Compound           : {compound}")
    print(f"Density            : {density:.4f} g/cm³")
    print(f"Energy             : {energy_keV:.3f} keV")
    print(f"μ/ρ                : {mass_atten:.4f} cm²/g")
    print(f"μ                  : {mu_cm:.4f} cm⁻¹  ({mu_mm:.4f} mm⁻¹)")
    print(f"Geometry           : {geometry}")
    print(f"GSAS-II Absorption : {gsasii_absorption:.6f}  (μr)" if geometry == "debye-scherrer"
          else f"GSAS-II Absorption : {gsasii_absorption:.6f}  (μt)")

    return {
        "mu_cm": mu_cm,
        "mu_mm": mu_mm,
        "mass_atten_cm2_g": mass_atten,
        "gsasii_absorption": gsasii_absorption,
    }


def calculate_padding_widths_2D(input_shape: tuple, desired_shape: tuple) -> tuple[tuple[int, int], tuple[int, int]]:
    """
    Compute symmetric ``np.pad`` widths to centre a 2-D array in a larger shape.

    Args:
        input_shape (tuple): ``(height, width)`` of the array to be padded.
        desired_shape (tuple): ``(height, width)`` of the target shape; must be
            >= *input_shape* in both dimensions.

    Returns:
        tuple: Padding widths ``((top, bottom), (left, right))`` suitable for
            passing directly to :func:`numpy.pad`.
    """
    y_in, x_in = input_shape
    y_des, x_des = desired_shape

    y_beg = (y_des - y_in) // 2
    y_end = (y_des - y_in) // 2 + (y_des - y_in) % 2

    x_beg = (x_des - x_in) // 2
    x_end = (x_des - x_in) // 2 + (x_des - x_in) % 2

    return ((y_beg, y_end), (x_beg, x_end))
