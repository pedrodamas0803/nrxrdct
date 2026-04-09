"""
Miscellaneous utility functions for XRD-CT data processing.

Covers percentile estimation for spot removal, XRD baseline fitting,
circular mask generation, powder pattern simulation and peak listing via
xrayutilities, and array padding helpers used by the reconstruction pipeline.
"""
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xraylib
import xrayutilities as xu
from pybaselines import Baseline
from pyFAI.integrator.azimuthal import AzimuthalIntegrator

from nrxrdct.integration import _get_integrator, azimuthal_integration_1d

from .io import save_xy_file


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


def calculate_xrd_baseline(
    y: np.ndarray, x: np.ndarray | None = None, method: str = "iasls", **kwargs
) -> tuple[np.ndarray, dict]:
    """
    Calculate the baseline of an X-ray diffraction (XRD) intensity curve.

    Args:
        y (np.ndarray): 1D array of intensity values from the XRD measurement.
        x (np.ndarray, optional): 1D array of 2θ (or q) values corresponding to `y`.
            If None, integer indices [0, 1, ..., len(y)-1] are used.
        method (str): Baseline algorithm to use. Recommended options for XRD:
            "iasls" – Iterative asymmetric least squares (default), good general-purpose
            choice; "aspls" – Adaptive smoothness penalized least squares, better when
            peaks are dense or asymmetric; "snip" – Statistics-sensitive Non-linear
            Iterative Peak-clipping, fast and parameter-light; "arpls" – Asymmetrically
            reweighted penalized least squares; "mor" – Morphological baseline (very fast).
        **kwargs: Extra keyword arguments forwarded to the chosen pybaselines method.
            Useful tuning parameters: iasls → lam=1e6, lam_1=1e-4, p=1e-2;
            aspls → lam=1e5, diff_order=2; snip → max_half_window=40,
            decreasing=True, smooth_half_window=3; arpls → lam=1e5, threshold=0.001.

    Returns:
        baseline (np.ndarray): Estimated baseline array, same shape as `y`.
        params (dict): Dictionary returned by pybaselines containing diagnostic info
            (e.g. weights, residuals, number of iterations).

    Example:
        >>> import numpy as np
        >>> two_theta = np.linspace(10, 80, 1000)
        >>> intensity  = np.random.default_rng(0).normal(500, 10, 1000)
        >>> baseline, info = calculate_xrd_baseline(intensity, two_theta)
        >>> corrected = intensity - baseline
    """
    y = np.asarray(y, dtype=float)

    if x is None:
        x = np.arange(len(y), dtype=float)
    else:
        x = np.asarray(x, dtype=float)

    if y.ndim != 1 or x.ndim != 1:
        raise ValueError("Both `y` and `x` must be 1-dimensional arrays.")
    if len(y) != len(x):
        raise ValueError("`y` and `x` must have the same length.")

    fitter = Baseline(x_data=x)

    # Default parameters tuned for typical XRD data
    defaults = {
        "iasls": {"lam": 1e6, "lam_1": 1e-4, "p": 1e-2},
        "aspls": {"lam": 1e5, "diff_order": 2},
        "snip": {"max_half_window": 40, "decreasing": True, "smooth_half_window": 3},
        "arpls": {"lam": 1e5},
        "mor": {},
    }

    if method not in defaults:
        raise ValueError(
            f"Unknown method '{method}'. Choose from: {list(defaults.keys())}"
        )

    # Merge defaults with any user-supplied kwargs (user wins)
    params = {**defaults[method], **kwargs}

    algo = getattr(fitter, method)
    baseline, info = algo(y, **params)

    return baseline, info


def generate_circular_mask(shape, center, diameter):
    """
    Generate a 2-D boolean circular mask for a given volume shape.

    Args:
        shape (tuple): Shape of the volume ``(n_tth, ny, nx)``; the mask is built
            over the last two dimensions.
        center (tuple): ``(cx, cy)`` centre of the circle in pixel coordinates.
        diameter (int or float): Diameter of the circular region in pixels.

    Returns:
        np.ndarray: 2-D boolean array of shape ``(ny, nx)`` where ``True`` marks
            pixels inside the circle.
    """
    x, y = np.arange(0, shape[1]), np.arange(0, shape[2])
    X, Y = np.meshgrid(x, y)
    z = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
    mask = z < diameter // 2
    return mask


def simulate_powder_xrd_monophase(
    tth,
    cif_files,
    do_plot=True,
    en_eV=100000,
    crystallite_size: float = 100e-9,
    do_save: bool = True,
):
    """
    Simulate a powder XRD pattern for one or more phases using xrayutilities.

    Args:
        tth (np.ndarray): 2θ array in degrees at which to evaluate the pattern.
        cif_files (str or list): Path(s) to CIF file(s); each phase is simulated
            independently.
        do_plot (bool, optional): If ``True``, produce a matplotlib figure for each
            phase (default ``True``).
        en_eV (float, optional): X-ray energy in eV (default 100 000 eV = 100 keV).
        crystallite_size (float, optional): Gaussian crystallite size in metres used
            for peak broadening (default 100 nm).
        do_save (bool, optional): If ``True``, save each pattern to
            ``<phase_name>_simulated.xy`` (default ``True``).

    Returns:
        np.ndarray: Simulated intensity array for the *last* phase in *cif_files*,
            evaluated at *tth*.
    """
    if not isinstance(cif_files, list):
        cif_files = [cif_files]

    for cif in cif_files:
        mat = xu.materials.Crystal.fromCIF(cif)
        pwdr = xu.simpack.Powder(mat, 1, crystallite_size_gauss=crystallite_size)
        model = xu.simpack.PowderModel(pwdr, I0=100, en=en_eV)
        intensity = model.simulate(tth)
        model.close()

        phase_name = mat.name.replace(" ", "_")
        if do_save:
            save_xy_file(tth, intensity, None, Path(f"{phase_name}_simulated.xy"))
        if do_plot:
            plt.figure()
            plt.plot(tth, intensity)
            plt.xlabel(r"2$\theta$ [degree]")
            plt.ylabel(r"Intensity")
            plt.title(f"{phase_name} simulated powder")

    return intensity


def get_powder_xrd_peaks(
    cif_files,
    en_eV: float = 100000,
    tth_min: float = None,
    tth_max: float = None,
) -> dict[str, pd.DataFrame]:
    """
    Return peak positions and hkl families for one or more CIF files.

    Args:
        cif_files (path or list): Path(s) to CIF file(s).
        en_eV (float): X-ray energy in eV (default 100 keV).
        tth_min (float, optional): Minimum 2theta in degrees (auto if None).
        tth_max (float, optional): Maximum 2theta in degrees (auto if None).

    Returns:
        dict: Mapping of phase_name to DataFrame with columns
            h, k, l, hkl, tth, d_hkl, r (structure factor |F|²).
    """
    if not isinstance(cif_files, list):
        cif_files = [cif_files]

    wavelength = xu.en2lam(en_eV)  # Å

    if tth_min is None:
        tth_min = 2 * np.degrees(np.arcsin(wavelength / (2 * 10.0)))
    if tth_max is None:
        tth_max = 2 * np.degrees(np.arcsin(wavelength / (2 * 0.5)))

    results = {}

    for cif in cif_files:
        mat = xu.materials.Crystal.fromCIF(cif)
        phase_name = mat.name.replace(" ", "_")

        pd_obj = xu.simpack.PowderDiffraction(mat, en=en_eV)

        rows = []
        for hkl, data in pd_obj.data.items():
            # only keep active (non-extinct) reflections
            if not data["active"]:
                continue

            tth_peak = data["ang"] * 2  # ang is in radians
            # d from Bragg's law: d = lambda / (2 * sin(theta))
            d = wavelength / (2 * np.sin(data["ang"]))  # ang is theta in radians

            if tth_min <= tth_peak <= tth_max:
                h, k, l = hkl
                rows.append(
                    {
                        "h": h,
                        "k": k,
                        "l": l,
                        "hkl": f"({h} {k} {l})",
                        "tth": round(float(tth_peak), 4),
                        "d_hkl": round(float(d), 4),
                        "r": float(data["r"]),  # |F|² structure factor
                    }
                )

        df = pd.DataFrame(rows).sort_values("r", ascending=False).reset_index(drop=True)
        results[phase_name] = df

    _print_peaks_table(results)
    return results


def _print_peaks_table(results: dict[str, pd.DataFrame]) -> None:
    """Print a well-aligned summary of XRD peak tables to stdout."""
    col_fmts = {
        "h":    ("{:>4}",  "{:>4}"),
        "k":    ("{:>4}",  "{:>4}"),
        "l":    ("{:>4}",  "{:>4}"),
        "hkl":  ("{:>12}", "{:>12}"),
        "tth":  ("{:>10}", "{:>10.4f}"),
        "d_hkl":("{:>10}", "{:>10.4f}"),
        "r":    ("{:>14}", "{:>14.2f}"),
    }
    header = "".join(hfmt.format(col) for col, (hfmt, _) in col_fmts.items())
    sep = "-" * len(header)

    for phase, df in results.items():
        print(f"\n{phase}  ({len(df)} reflections)")
        print(sep)
        print(header)
        print(sep)
        for _, row in df.iterrows():
            line = "".join(
                rfmt.format(row[col]) for col, (_, rfmt) in col_fmts.items()
            )
            print(line)
        print(sep)


def calculate_absorption_coefficient(
    compound: str,
    density: float,
    energy_keV: float,
    sample_diameter_mm: float,
    geometry: str = "debye-scherrer",
) -> dict:
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


def calculate_padding_widths_2D(input_shape: tuple, desired_shape: tuple):
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
