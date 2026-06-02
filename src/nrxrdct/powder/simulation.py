from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import xrayutilities as xu
from pybaselines import Baseline

from nrxrdct.xrdct.io import save_xy_file


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

def simulate_powder_xrd_monophase(
    tth,
    cif_files,
    do_plot=True,
    energy_keV: float = 100.0,
    crystallite_size: float = 100e-9,
    do_save: bool = True,
) -> np.ndarray:
    """
    Simulate a powder XRD pattern for one or more phases using xrayutilities.

    Args:
        tth (np.ndarray): 2θ array in degrees at which to evaluate the pattern.
        cif_files (str or list): Path(s) to CIF file(s); each phase is simulated
            independently.
        do_plot (bool, optional): If ``True``, produce a matplotlib figure for each
            phase (default ``True``).
        energy_keV (float, optional): X-ray energy in keV (default 100 keV).
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
        model = xu.simpack.PowderModel(pwdr, I0=100, en=energy_keV * 1000)
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
    names: list[str] = None,
    energy_keV: float = 100.0,
    tth_min: float = None,
    tth_max: float = None,
) -> dict[str, pd.DataFrame]:
    """
    Return peak positions and hkl families for one or more CIF files.

    Args:
        cif_files (path or list): Path(s) to CIF file(s).
        names (list of str, optional): Phase names to use as dict keys, one per
            CIF file.  If None or shorter than cif_files, the name embedded in
            the CIF is used for any entry without a supplied name.
        energy_keV (float): X-ray energy in keV (default 100 keV).
        tth_min (float, optional): Minimum 2theta in degrees (auto if None).
        tth_max (float, optional): Maximum 2theta in degrees (auto if None).

    Returns:
        dict: Mapping of phase_name to DataFrame with columns
            h, k, l, hkl, tth, d_hkl, r (structure factor |F|²).
    """
    if not isinstance(cif_files, list):
        cif_files = [cif_files]

    if names is None:
        names = []

    wavelength = xu.en2lam(energy_keV * 1000)  # Å

    if tth_min is None:
        tth_min = 2 * np.degrees(np.arcsin(wavelength / (2 * 10.0)))
    if tth_max is None:
        tth_max = 2 * np.degrees(np.arcsin(wavelength / (2 * 0.5)))

    results = {}

    for i, cif in enumerate(cif_files):
        mat = xu.materials.Crystal.fromCIF(cif)
        phase_name = names[i] if i < len(names) else mat.name.replace(" ", "_")

        pd_obj = xu.simpack.PowderDiffraction(mat, en=energy_keV * 1000)

        rows = []
        for hkl, data in pd_obj.data.items():
            # only keep active (non-extinct) reflections
            if not data["active"]:
                continue

            tth_peak = data["ang"] * 2  # ang is theta in degrees; tth in degrees
            # d from Bragg's law: d = lambda / (2 * sin(theta))
            d = wavelength / (2 * np.sin(np.radians(data["ang"])))

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

