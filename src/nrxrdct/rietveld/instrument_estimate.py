"""
Data-driven starting values for GSAS-II CW instrument parameters.

Used by :class:`~nrxrdct.rietveld.refinement.InstrumentCalibration` to replace
the generic defaults of :func:`~nrxrdct.xrdct.io.write_starting_instrument_pars`
with values inferred from the calibrant pattern itself:

* **U, V, W, X, Y, Z** — isolated peaks are fitted one by one with a
  pseudo-Voigt, each total FWHM/η is split into Gaussian and Lorentzian widths
  by inverting the Thompson-Cox-Hastings relation GSAS-II uses, and the
  Caglioti / Lorentzian width laws are then fitted to those widths.
* **Zero** — observed peak centres are matched to the calibrant's calculated
  reflection positions.

Pure numpy/scipy; nothing here depends on GSAS-II.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.optimize import brentq, curve_fit, lsq_linear
from scipy.signal import find_peaks, peak_widths

_LN2 = np.log(2.0)

# Width-law columns in GSAS-II units (θ in radians):
#   Gaussian   σ² [centideg²] = U·tan²θ + V·tanθ + W
#   Lorentzian γ  [centideg]  = X/cosθ + Y·tanθ + Z
_GAUSS_TERMS = {
    "U": lambda t, c: t**2,
    "V": lambda t, c: t,
    "W": lambda t, c: np.ones_like(t),
}
_LORENTZ_TERMS = {
    "X": lambda t, c: 1.0 / c,
    "Y": lambda t, c: t,
    "Z": lambda t, c: np.ones_like(t),
}


def _pseudo_voigt(x, area, x0, fwhm, eta, b0, b1):
    """Area-normalised pseudo-Voigt on a linear background."""
    dx = x - x0
    hw = fwhm / 2.0
    gauss = np.sqrt(4 * _LN2 / np.pi) / fwhm * np.exp(-4 * _LN2 * dx**2 / fwhm**2)
    lorentz = (hw / np.pi) / (dx**2 + hw**2)
    return area * (eta * lorentz + (1.0 - eta) * gauss) + b0 + b1 * dx


def _tch_fwhm(fG: float, fL: float) -> float:
    """Total FWHM from Gaussian and Lorentzian FWHMs (Thompson-Cox-Hastings)."""
    return (
        fG**5
        + 2.69269 * fG**4 * fL
        + 2.42843 * fG**3 * fL**2
        + 4.47163 * fG**2 * fL**3
        + 0.07842 * fG * fL**4
        + fL**5
    ) ** 0.2


def split_tch(fwhm: float, eta: float) -> tuple[float, float]:
    """
    Invert the Thompson-Cox-Hastings pseudo-Voigt approximation.

    Args:
        fwhm (float): Total pseudo-Voigt FWHM.
        eta (float): Pseudo-Voigt Lorentzian fraction, clipped to ``[0, 1]``.

    Returns:
        tuple of (float, float): ``(fwhm_G, fwhm_L)`` in the units of ``fwhm``.
    """
    eta = float(np.clip(eta, 0.0, 1.0))
    # η = 1.36603·q − 0.47719·q² + 0.11116·q³, q = fL/fwhm; monotonic on [0, 1]
    # with η(0) = 0 and η(1) = 1.
    if eta <= 0.0:
        return fwhm, 0.0
    if eta >= 1.0:
        return 0.0, fwhm
    q = brentq(lambda q: 1.36603 * q - 0.47719 * q**2 + 0.11116 * q**3 - eta, 0.0, 1.0)
    fL = q * fwhm
    fG = brentq(lambda g: _tch_fwhm(g, fL) - fwhm, 0.0, fwhm)
    return fG, fL


def _noise_sigma(y: np.ndarray) -> float:
    """Robust point-to-point noise estimate (MAD of first differences)."""
    d = np.diff(y)
    return 1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2.0)


def fit_isolated_peaks(
    tth: np.ndarray,
    intensity: np.ndarray,
    min_snr: float = 10.0,
    isolation: float = 3.0,
    window: float = 4.0,
    max_rel_err: float = 0.2,
    min_points_per_fwhm: float = 2.0,
) -> list[dict]:
    """
    Find isolated peaks and fit each with a pseudo-Voigt on a linear background.

    Args:
        tth (array): 2θ in degrees, ascending.
        intensity (array): Intensities at ``tth``.
        min_snr (float, optional): Minimum peak prominence in units of the
            point-to-point noise (default 10).
        isolation (float, optional): A peak is kept only if no other detected
            peak lies within ``isolation`` × its rough FWHM (default 3).
        window (float, optional): Half-width of the fit window in rough FWHMs
            (default 4), clipped at the midpoint to the neighbouring peaks.
        max_rel_err (float, optional): Reject fits whose FWHM relative
            uncertainty exceeds this (default 0.2).
        min_points_per_fwhm (float, optional): Reject peaks sampled by fewer
            points per FWHM than this (default 2) — their width is not resolved.

    Returns:
        list of dict: One dict per accepted peak with keys ``tth``, ``fwhm``,
        ``eta``, ``fwhm_G``, ``fwhm_L`` (all widths in degrees) and
        ``fwhm_err``.
    """
    tth = np.asarray(tth, dtype=float)
    y = np.asarray(intensity, dtype=float)
    step = np.gradient(tth)

    sigma = _noise_sigma(y)
    prominence = min_snr * sigma if sigma > 0 else 0.01 * np.ptp(y)
    idx, _ = find_peaks(y, prominence=prominence)
    if len(idx) == 0:
        return []
    rough = peak_widths(y, idx, rel_height=0.5)[0] * step[idx]
    centres = tth[idx]

    peaks = []
    for i, (p, c, w) in enumerate(zip(idx, centres, rough)):
        others = np.delete(centres, i)
        nearest = np.min(np.abs(others - c)) if len(others) else np.inf
        if nearest < isolation * w:
            continue
        lo = c - window * w
        hi = c + window * w
        if i > 0:
            lo = max(lo, 0.5 * (c + centres[i - 1]))
        if i < len(centres) - 1:
            hi = min(hi, 0.5 * (c + centres[i + 1]))
        sel = (tth >= lo) & (tth <= hi)
        if sel.sum() < 8:
            continue
        x, yy = tth[sel], y[sel]

        bg = min(yy[0], yy[-1])
        p0 = [max((y[p] - bg) * w, 1e-12), c, w, 0.5, bg, 0.0]
        lower = [0.0, c - w, 0.2 * w, 0.0, -np.inf, -np.inf]
        upper = [np.inf, c + w, 5.0 * w, 1.0, np.inf, np.inf]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                popt, pcov = curve_fit(
                    _pseudo_voigt, x, yy, p0=p0, bounds=(lower, upper), maxfev=5000
                )
        except (RuntimeError, ValueError):
            continue

        _, x0, fwhm, eta, _, _ = popt
        fwhm_err = np.sqrt(pcov[2, 2]) if np.isfinite(pcov[2, 2]) else np.inf
        if fwhm_err > max_rel_err * fwhm:
            continue
        if fwhm < min_points_per_fwhm * step[p]:
            continue
        fG, fL = split_tch(fwhm, eta)
        peaks.append(
            {
                "tth": x0,
                "fwhm": fwhm,
                "fwhm_err": fwhm_err,
                "eta": eta,
                "fwhm_G": fG,
                "fwhm_L": fL,
            }
        )
    return peaks


def _fit_width_law(
    terms: dict, names: list[str], tan_th, cos_th, target, lower: dict
) -> dict[str, float]:
    """Bounded linear least squares of ``target`` on the selected width-law terms."""
    A = np.column_stack([terms[n](tan_th, cos_th) for n in names])
    lb = [lower[n] for n in names]
    res = lsq_linear(A, target, bounds=(lb, [np.inf] * len(names)))
    return dict(zip(names, res.x))


def estimate_profile_parameters(
    peaks: list[dict], params: list[str] = ["W", "X", "Y"]
) -> dict[str, float]:
    """
    Fit GSAS-II's CW width laws to per-peak Gaussian/Lorentzian widths.

    Only the parameters in ``params`` are fitted; the other terms of a width
    law that has at least one fitted term are returned as ``0.0``, so the
    starting profile is exactly the fitted one (e.g. the default ``Y = 5``
    must not add on top of a fitted ``X``). A width law with no term in
    ``params`` is left out of the result entirely.

    When there are fewer usable peaks than terms to fit in a width law, only
    a single term is fitted for it (``W`` for the Gaussian if requested,
    otherwise the first requested term) and a warning is issued.

    Args:
        peaks (list of dict): Output of :func:`fit_isolated_peaks`.
        params (list of str, optional): Terms to estimate, among
            ``U, V, W, X, Y, Z`` (default ``["W", "X", "Y"]``).

    Returns:
        dict: Parameter name → value in GSAS-II units (U/V/W in centideg²,
        X/Y/Z in centideg).
    """
    unknown = set(params) - set(_GAUSS_TERMS) - set(_LORENTZ_TERMS)
    if unknown:
        raise ValueError(f"Cannot estimate {sorted(unknown)}; choose among U, V, W, X, Y, Z.")
    if not peaks:
        warnings.warn("No usable peaks: profile parameters not estimated.")
        return {}

    tth = np.array([p["tth"] for p in peaks])
    tan_th = np.tan(np.radians(tth / 2.0))
    cos_th = np.cos(np.radians(tth / 2.0))
    # GSAS-II units: centidegrees.
    sig2 = (100.0 * np.array([p["fwhm_G"] for p in peaks])) ** 2 / (8.0 * _LN2)
    gam = 100.0 * np.array([p["fwhm_L"] for p in peaks])

    # V is negative for typical lab optics; all other terms are non-negative.
    lower = {"U": 0.0, "V": -np.inf, "W": 0.0, "X": 0.0, "Y": 0.0, "Z": 0.0}

    result: dict[str, float] = {}
    for terms, target, preferred in (
        (_GAUSS_TERMS, sig2, "W"),
        (_LORENTZ_TERMS, gam, None),
    ):
        names = [n for n in terms if n in params]
        if not names:
            continue
        if len(peaks) < len(names):
            single = preferred if preferred in names else names[0]
            warnings.warn(
                f"Only {len(peaks)} usable peak(s) for {names}: estimating {single} alone."
            )
            names = [single]
        result.update(dict.fromkeys(terms, 0.0))
        result.update(_fit_width_law(terms, names, tan_th, cos_th, target, lower))
    return result


def estimate_zero(
    tth_obs: np.ndarray,
    tth_calc: np.ndarray,
    tolerance: float,
    max_shift: float = 0.3,
) -> float | None:
    """
    Estimate the GSAS-II zero shift from observed and calculated peak positions.

    GSAS-II places a reflection at ``2θ_calc + Zero``. A grid search over
    ``Zero ∈ [-max_shift, max_shift]`` minimises the robust cost
    ``Σ min(|2θ_obs − (2θ_calc + Zero)|, tolerance)²`` (nearest calculated
    reflection per observed peak), which tolerates unmatched or spurious
    peaks. The result is then refined as the median offset of the matched
    pairs.

    Args:
        tth_obs (array): Observed peak centres in degrees.
        tth_calc (array): Calculated reflection positions in degrees (zero shift excluded).
        tolerance (float): Maximum |offset| in degrees for a peak to count as
            matched — about one peak FWHM.
        max_shift (float, optional): Search range in degrees (default 0.3).
            Keep it below half the spacing between neighbouring reflections.

    Returns:
        float or None: Zero shift in degrees, or ``None`` when fewer than two
        observed peaks can be matched.
    """
    obs = np.asarray(tth_obs, dtype=float)
    calc = np.sort(np.asarray(tth_calc, dtype=float))
    if len(obs) == 0 or len(calc) == 0:
        return None

    def offsets(z):
        shifted = calc + z
        j = np.clip(np.searchsorted(shifted, obs), 1, len(shifted) - 1)
        left, right = shifted[j - 1], shifted[j]
        nearest = np.where(np.abs(obs - left) < np.abs(obs - right), left, right)
        return obs - nearest

    step = tolerance / 20.0
    grid = np.arange(-max_shift, max_shift + step, step)
    costs = [np.sum(np.minimum(np.abs(offsets(z)), tolerance) ** 2) for z in grid]
    z = grid[int(np.argmin(costs))]

    d = offsets(z)
    matched = np.abs(d) < tolerance
    if matched.sum() < 2:
        return None
    return float(z + np.median(d[matched]))
