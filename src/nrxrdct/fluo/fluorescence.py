"""
X-ray fluorescence (XRF) data loading and sinogram assembly for XRD-CT.

Provides helpers to look up XRF emission line energies via xraylib, read
per-scan ROI or full-spectrum fluorescence data from HDF5 master files,
and assemble the resulting data into sinogram arrays ready for reconstruction.
"""

import numpy as np
import pandas as pd
import xraylib
from scipy.optimize import nnls
from tqdm.auto import tqdm # type: ignore

from .constants import DEFAULT_LINES, _LINE_MAP


def get_fluo_lines(
    elements,
    energy_range,
    names: list[str] = None,
    lines: list[str] = DEFAULT_LINES,
) -> dict[str, pd.DataFrame]:
    """
    Return XRF emission line energies for one or more elements within a given
    energy range.

    Args:
        elements (str or list): Chemical symbol(s) (e.g. ``"Fe"`` or
            ``["Fe", "Cu", "Zn"]``).
        energy_range (tuple): ``(emin, emax)`` energy window in keV; lines
            outside this range are excluded.
        names (list of str, optional): Keys to use in the returned dict, one
            per element.  If ``None`` or shorter than *elements*, the chemical
            symbol is used for any entry without a supplied name.
        lines (list, optional): Emission line names to query.  Must be keys of
            ``_LINE_MAP``; unknown names are silently skipped.  Defaults to
            ``DEFAULT_LINES`` — all standard Siegbahn-named K, L, and M lines.

    Returns:
        dict: Mapping of element name (or supplied name) to a
        ``pd.DataFrame`` with columns ``line`` and ``energy_keV``, sorted by
        energy.  Elements for which no lines fall within *energy_range* are
        included with an empty DataFrame.
    """
    if not isinstance(elements, list):
        elements = [elements]

    if names is None:
        names = []

    emin, emax = energy_range

    results = {}
    for i, element in enumerate(elements):
        key = names[i] if i < len(names) else element
        Z = xraylib.SymbolToAtomicNumber(element)

        rows = []
        for line_name in lines:
            line_const = _LINE_MAP.get(line_name)
            if line_const is None:
                continue
            try:
                en = xraylib.LineEnergy(Z, line_const)
            except Exception:
                continue
            if emin <= en <= emax:
                rows.append({"line": line_name, "energy_keV": round(float(en), 4)})

        df = pd.DataFrame(rows, columns=["line", "energy_keV"]).sort_values(
            "energy_keV"
        ).reset_index(drop=True)
        results[key] = df

    _print_fluo_table(results)
    return results


def _print_fluo_table(results: dict[str, pd.DataFrame]) -> None:
    """Print a well-aligned summary of XRF line tables to stdout."""
    col_fmts = {
        "line":       ("{:>8}",  "{:>8}"),
        "energy_keV": ("{:>14}", "{:>14.4f}"),
    }
    header = "".join(hfmt.format(col) for col, (hfmt, _) in col_fmts.items())
    sep = "-" * len(header)

    for element, df in results.items():
        print(f"\n{element}  ({len(df)} lines)")
        print(sep)
        print(header)
        print(sep)
        for _, row in df.iterrows():
            print("".join(rfmt.format(row[col]) for col, (_, rfmt) in col_fmts.items()))
        print(sep)




# ---------------------------------------------------------------------------
# Spectrum fitting
# ---------------------------------------------------------------------------


def _gaussian(x, center, sigma) -> np.ndarray:
    return np.exp(-0.5 * ((x - center) / sigma) ** 2)


def build_element_component(element, energy_axis, excitation_energy, fwhm_keV) -> np.ndarray:
    """
    Build the expected spectral shape for one element on a given energy axis.

    Each emission line is modelled as a Gaussian weighted by its fluorescence
    cross-section (``CS_FluorLine_Kissel``), so relative amplitudes between
    lines of the same element are physically motivated.

    Args:
        element (str): Chemical symbol, e.g. ``"Fe"``.
        energy_axis (np.ndarray): 1-D array of energy values in keV (from MCA
            calibration).
        excitation_energy (float): Incident beam energy in keV.
        fwhm_keV (float): Detector energy resolution (FWHM) in keV — typically
            0.13–0.25 keV for a Si drift detector.

    Returns:
        component (np.ndarray): Normalised spectral template for the element
            (max = 1), shape ``(len(energy_axis),)``. Returns a zero array if no
            lines fall within the energy axis range.
    """
    sigma = fwhm_keV / (2 * np.sqrt(2 * np.log(2)))
    Z = xraylib.SymbolToAtomicNumber(element)
    emin, emax = energy_axis[0], energy_axis[-1]

    component = np.zeros_like(energy_axis, dtype=np.float64)
    for line_name in DEFAULT_LINES:
        lconst = _LINE_MAP.get(line_name)
        if lconst is None:
            continue
        try:
            en = xraylib.LineEnergy(Z, lconst)
            if not (emin <= en <= emax):
                continue
            cs = xraylib.CS_FluorLine_Kissel(Z, lconst, excitation_energy)
            component += cs * _gaussian(energy_axis, en, sigma)
        except Exception:
            continue

    norm = component.max()
    return component / norm if norm > 0 else component


def fit_fluo_spectrum(
    spectrum,
    energy_axis,
    elements,
    excitation_energy,
    fwhm_keV=0.18,
    background_order=2,
) -> tuple[dict[str, float], np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """
    Fit a fluorescence spectrum as a linear combination of element templates.

    A polynomial background is included as additional free components.  The fit
    is solved with non-negative least squares (NNLS), which prevents unphysical
    negative elemental intensities.

    Args:
        spectrum (np.ndarray): Measured (normalised) MCA spectrum, shape
            ``(n_channels,)``.
        energy_axis (np.ndarray): Energy in keV for each MCA channel, shape
            ``(n_channels,)``.
        elements (list): Elements to include, e.g. ``["Fe", "Cu", "Zn"]``.
        excitation_energy (float): Incident beam energy in keV.
        fwhm_keV (float, optional): Detector FWHM in keV (default 0.18).
        background_order (int, optional): Polynomial degree for the background
            model (default 2).

    Returns:
        coefficients (dict): Mapping ``element -> fitted amplitude`` (arbitrary
            units proportional to element concentration × thickness).
        residual (np.ndarray): Measured minus fitted spectrum.
        fitted (np.ndarray): Reconstructed spectrum (element contributions +
            background).
        components (dict): Spectral template used for each element (before scaling).
    """
    E_norm = (energy_axis - energy_axis.mean()) / energy_axis.ptp()

    columns = {}
    for el in elements:
        comp = build_element_component(el, energy_axis, excitation_energy, fwhm_keV)
        if comp.max() > 0:
            columns[el] = comp

    for deg in range(background_order + 1):
        columns[f"_bg{deg}"] = E_norm**deg

    labels = list(columns.keys())
    A = np.column_stack([columns[k] for k in labels])

    coeffs, _ = nnls(A, spectrum)

    fitted = A @ coeffs
    residual = spectrum - fitted

    coefficients = {k: v for k, v in zip(labels, coeffs) if not k.startswith("_bg")}
    components = {k: columns[k] for k in elements if k in columns}

    return coefficients, residual, fitted, components


def build_fit_matrix(
    energy_axis, elements, excitation_energy, fwhm_keV=0.18, background_order=2
) -> tuple[np.ndarray, list[str]]:
    """
    Build the design matrix for XRF spectrum fitting.

    Separating matrix construction from fitting allows the (identical) matrix
    to be built once and reused across all pixels.

    Args:
        energy_axis (np.ndarray): Energy in keV for each MCA channel, shape
            ``(n_channels,)``.
        elements (list): Elements to model, e.g. ``["Fe", "Cu", "Zn"]``.
        excitation_energy (float): Incident beam energy in keV.
        fwhm_keV (float, optional): Detector FWHM in keV (default 0.18).
        background_order (int, optional): Polynomial degree for the background
            model (default 2).

    Returns:
        A (np.ndarray): Design matrix of shape ``(n_channels, n_components)``.
            Columns are element templates followed by background polynomial terms.
        labels (list): Column labels. Element names come first; background terms
            are named ``"_bg0"``, ``"_bg1"``, etc.
    """
    E_norm = (energy_axis - energy_axis.mean()) / energy_axis.ptp()

    columns, labels = [], []
    for el in elements:
        comp = build_element_component(el, energy_axis, excitation_energy, fwhm_keV)
        if comp.max() > 0:
            columns.append(comp)
            labels.append(el)

    for deg in range(background_order + 1):
        columns.append(E_norm**deg)
        labels.append(f"_bg{deg}")

    return np.column_stack(columns), labels


def fit_fluo_volume(
    data,
    energy_axis,
    elements,
    excitation_energy,
    fwhm_keV=0.18,
    background_order=2,
    method="lstsq",
    n_jobs=-1,
) -> dict[str, np.ndarray]:
    """
    Fit fluorescence spectra for every pixel in a 3-D dataset.

    Args:
        data (np.ndarray): Full-spectrum dataset of shape ``(n_energy, n_y, n_x)``;
            axis 0 is the energy dimension.
        energy_axis (np.ndarray): Energy in keV for each channel, shape
            ``(n_energy,)``.
        elements (list): Elements to fit, e.g. ``["Fe", "Cu", "Zn"]``.
        excitation_energy (float): Incident beam energy in keV.
        fwhm_keV (float, optional): Detector FWHM in keV (default 0.18).
        background_order (int, optional): Polynomial degree for the background
            model (default 2).
        method (str, optional): Fitting strategy. ``"lstsq"`` solves all pixels
            in a single batched ``np.linalg.lstsq`` call — very fast but
            non-negativity is only enforced by clipping. ``"nnls"`` calls
            ``scipy.optimize.nnls`` per pixel (parallelised with ``joblib``) —
            strictly non-negative but slower. Default ``"lstsq"``.
        n_jobs (int, optional): Number of parallel workers for ``method="nnls"``
            (``-1`` uses all CPUs). Ignored for ``method="lstsq"`` (default ``-1``).

    Returns:
        coeff_maps (dict): Mapping ``element -> np.ndarray`` of shape ``(n_y, n_x)``
            with the fitted amplitude for each spatial pixel.
    """
    n_energy, n_y, n_x = data.shape
    n_pixels = n_y * n_x

    A, labels = build_fit_matrix(
        energy_axis, elements, excitation_energy, fwhm_keV, background_order
    )
    element_labels = [l for l in labels if not l.startswith("_bg")]

    # Shape: (n_energy, n_pixels) — keeps memory contiguous for LAPACK
    spectra = data.reshape(n_energy, n_pixels)

    if method == "lstsq":
        # Single batched solve: A (n_energy, n_comp) \ spectra (n_energy, n_pixels)
        coeffs, _, _, _ = np.linalg.lstsq(A, spectra, rcond=None)
        # coeffs: (n_comp, n_pixels) -> (n_pixels, n_comp)
        coeffs = np.clip(coeffs.T, 0, None)

    elif method == "nnls":
        # spectra.T: (n_pixels, n_energy) for row-wise iteration
        spectra_t = np.ascontiguousarray(spectra.T)
        try:
            from joblib import Parallel, delayed

            results = Parallel(n_jobs=n_jobs)(
                delayed(nnls)(A, spectra_t[i]) for i in range(n_pixels)
            )
            coeffs = np.array([r[0] for r in results])
        except ImportError:
            coeffs = np.array(
                [nnls(A, spectra_t[i])[0] for i in tqdm(range(n_pixels))]
            )

    else:
        raise ValueError(f"Unknown method '{method}'. Choose 'lstsq' or 'nnls'.")

    coeff_maps = {
        label: coeffs[:, j].reshape(n_y, n_x)
        for j, label in enumerate(labels)
        if label in element_labels
    }
    return coeff_maps
