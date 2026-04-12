"""
Single-peak fitting utilities for 1-D XRD diffraction patterns.

The background is estimated first with :func:`~nrxrdct.utils.calculate_xrd_baseline`
(powered by `pybaselines`), then a Gaussian, Lorentzian, Voigt, or pseudo-Voigt
profile is fitted to the background-subtracted data using
:func:`scipy.optimize.curve_fit`.

Functions
---------
extract_window(tth, intensity, center, window)
    Clip pattern arrays to the fitting window.
fit_peak(tth, intensity, center, window, model, bg_method, bg_kwargs)
    Estimate background, subtract it, fit one peak, return a parameter dict.
fit_peak_from_file(xy_file, center, window, model, bg_method, bg_kwargs)
    Load a ``.xy`` file and call :func:`fit_peak`.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Literal

import numpy as np
from scipy.optimize import curve_fit
from scipy.special import voigt_profile as _scipy_voigt

from .io import read_xy_file
from .utils import calculate_xrd_baseline


# ---------------------------------------------------------------------------
# Internal peak profile models  (background-subtracted signal only)
# All use centre-shifted coordinates  x_c = x - center.
# ---------------------------------------------------------------------------

def _gaussian(x: np.ndarray, A: float, x0: float, sigma: float) -> np.ndarray:
    return A * np.exp(-0.5 * ((x - x0) / sigma) ** 2)


def _lorentzian(x: np.ndarray, A: float, x0: float, gamma: float) -> np.ndarray:
    return A * gamma ** 2 / ((x - x0) ** 2 + gamma ** 2)


def _voigt(x: np.ndarray, A: float, x0: float, sigma: float, gamma: float) -> np.ndarray:
    """Voigt profile scaled so that the peak value equals A."""
    norm = _scipy_voigt(0.0, sigma, gamma)
    return A * _scipy_voigt(x - x0, sigma, gamma) / norm


def _pseudo_voigt(
    x: np.ndarray, A: float, x0: float, fwhm: float, eta: float
) -> np.ndarray:
    """Pseudo-Voigt: eta × Lorentzian + (1−eta) × Gaussian, same FWHM."""
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    gamma = fwhm / 2.0
    G = np.exp(-0.5 * ((x - x0) / sigma) ** 2)
    L = gamma ** 2 / ((x - x0) ** 2 + gamma ** 2)
    return A * (eta * L + (1.0 - eta) * G)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def extract_window(
    tth: np.ndarray,
    intensity: np.ndarray,
    center: float,
    window: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Clip (tth, intensity) to a ±window/2 neighbourhood of *center*.

    Args:
        tth (np.ndarray): 2θ axis in degrees.
        intensity (np.ndarray): Intensity values, same length as *tth*.
        center (float): Nominal peak position in degrees.
        window (float): Total window width in degrees.

    Returns:
        tuple: ``(tth_win, int_win)``

    Raises:
        ValueError: Fewer than 5 data points fall inside the window.
    """
    half = window / 2.0
    mask = (tth >= center - half) & (tth <= center + half)
    n = int(mask.sum())
    if n < 5:
        raise ValueError(
            f"Only {n} points found in window "
            f"[{center - half:.4g}, {center + half:.4g}] °. "
            "Widen the window or adjust the center position."
        )
    return tth[mask], intensity[mask]


# ---------------------------------------------------------------------------
# Core fitting function
# ---------------------------------------------------------------------------

def fit_peak(
    tth: np.ndarray,
    intensity: np.ndarray,
    center: float,
    window: float,
    model: Literal["gaussian", "lorentzian", "voigt", "pseudo_voigt"] = "pseudo_voigt",
    bg_method: str = "snip",
    bg_kwargs: dict[str, Any] | None = None,
) -> dict:
    """
    Fit a single diffraction peak within a 2θ window.

    The background is estimated with :func:`~nrxrdct.utils.calculate_xrd_baseline`
    on the windowed data before fitting, so no prior background subtraction is
    required.

    Args:
        tth (np.ndarray): 2θ axis in degrees (full pattern or pre-cropped).
        intensity (np.ndarray): Intensity values, same length as *tth*.
        center (float): Nominal peak centre (°); defines the window midpoint.
        window (float): Total window width in degrees around *center*.
        model (str): Peak profile – ``"gaussian"``, ``"lorentzian"``,
            ``"voigt"``, or ``"pseudo_voigt"`` (default).
        bg_method (str): Background algorithm forwarded to
            :func:`~nrxrdct.utils.calculate_xrd_baseline`.  Supported values:
            ``"snip"`` (default, fast), ``"iasls"``, ``"aspls"``,
            ``"arpls"``, ``"mor"``.
        bg_kwargs (dict or None): Extra keyword arguments for the background
            estimator (forwarded verbatim).

    Returns:
        dict: Fitted parameters.  Keys present for **all** models:

        * ``"center"``    – refined peak position (°)
        * ``"amplitude"`` – peak height above background
        * ``"fwhm"``      – full-width at half-maximum (°)
        * ``"area"``      – integrated peak area (intensity × °)
        * ``"residual"``  – RMS residual between background-subtracted data
          and fit (intensity units)
        * ``"success"``   – ``True`` if the optimiser converged

        Model-specific keys:

        * ``"sigma"`` – Gaussian standard deviation (°) — Gaussian and Voigt
        * ``"gamma"`` – Lorentzian half-width (°) — Lorentzian and Voigt
        * ``"eta"``   – Lorentzian mixing fraction ∈ [0, 1] — pseudo-Voigt

        On failure all numeric values are ``nan`` and ``success`` is ``False``.

    Raises:
        ValueError: *model* not recognised or window too narrow.

    Example::

        tth, I = np.loadtxt("pattern.xy", unpack=True)
        result = fit_peak(tth, I, center=3.56, window=0.4, model="pseudo_voigt")
        print(f"FWHM = {result['fwhm']:.4f} °   centre = {result['center']:.4f} °")
    """
    model = model.lower().replace("-", "_")
    if model not in ("gaussian", "lorentzian", "voigt", "pseudo_voigt"):
        raise ValueError(
            f"model={model!r} not recognised. "
            "Choose 'gaussian', 'lorentzian', 'voigt', or 'pseudo_voigt'."
        )

    x, y = extract_window(tth, intensity, center, window)
    x_c = x - center  # centre-shift for numerical stability

    # ── Background estimation & subtraction ───────────────────────────────
    bg, _ = calculate_xrd_baseline(y, x, method=bg_method, **(bg_kwargs or {}))
    y_net = y - bg

    # ── Initial parameter guesses ─────────────────────────────────────────
    A0 = max(float(y_net.max()), 1e-3)
    x0_0 = float(x_c[np.argmax(y_net)])
    dx = float(np.diff(x).mean())
    above_half = y_net > (A0 / 2.0)
    if above_half.sum() >= 2:
        fwhm0 = max(float(np.ptp(x[above_half])) * 1.2, dx * 2.0)
    else:
        fwhm0 = max(window * 0.3, dx * 2.0)
    sigma0 = fwhm0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    gamma0 = fwhm0 / 2.0
    half_w = window / 2.0

    _nan: dict = dict(
        center=np.nan, amplitude=np.nan, fwhm=np.nan, area=np.nan,
        residual=np.nan, success=False,
    )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # ── Gaussian ──────────────────────────────────────────────────
            if model == "gaussian":
                p0 = [A0, x0_0, sigma0]
                lo = [0.0, -half_w, 1e-9]
                hi = [np.inf, half_w, window]
                popt, _ = curve_fit(
                    _gaussian, x_c, y_net, p0=p0, bounds=(lo, hi), maxfev=10_000
                )
                A, x0r, sigma = popt
                sigma = abs(sigma)
                fwhm = 2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma
                area = A * sigma * np.sqrt(2.0 * np.pi)
                extra = {"sigma": sigma}
                y_fit = _gaussian(x_c, *popt)

            # ── Lorentzian ────────────────────────────────────────────────
            elif model == "lorentzian":
                p0 = [A0, x0_0, gamma0]
                lo = [0.0, -half_w, 1e-9]
                hi = [np.inf, half_w, window]
                popt, _ = curve_fit(
                    _lorentzian, x_c, y_net, p0=p0, bounds=(lo, hi), maxfev=10_000
                )
                A, x0r, gamma = popt
                gamma = abs(gamma)
                fwhm = 2.0 * gamma
                area = A * gamma * np.pi
                extra = {"gamma": gamma}
                y_fit = _lorentzian(x_c, *popt)

            # ── Voigt ─────────────────────────────────────────────────────
            elif model == "voigt":
                p0 = [A0, x0_0, sigma0, gamma0]
                lo = [0.0, -half_w, 1e-9, 1e-9]
                hi = [np.inf, half_w, window, window]
                popt, _ = curve_fit(
                    _voigt, x_c, y_net, p0=p0, bounds=(lo, hi), maxfev=10_000
                )
                A, x0r, sigma, gamma = popt
                sigma, gamma = abs(sigma), abs(gamma)
                # Thompson–Cox–Hastings (1987) FWHM approximation
                fG = 2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma
                fL = 2.0 * gamma
                fwhm = (
                    fG**5 + 2.69269 * fG**4 * fL + 2.42843 * fG**3 * fL**2
                    + 4.47163 * fG**2 * fL**3 + 0.07842 * fG * fL**4 + fL**5
                ) ** 0.2
                area = A / _scipy_voigt(0.0, sigma, gamma)
                extra = {"sigma": sigma, "gamma": gamma}
                y_fit = _voigt(x_c, *popt)

            # ── Pseudo-Voigt ──────────────────────────────────────────────
            else:
                p0 = [A0, x0_0, fwhm0, 0.5]
                lo = [0.0, -half_w, 1e-9, 0.0]
                hi = [np.inf, half_w, window, 1.0]
                popt, _ = curve_fit(
                    _pseudo_voigt, x_c, y_net, p0=p0, bounds=(lo, hi), maxfev=10_000
                )
                A, x0r, fwhm, eta = popt
                sg = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
                gm = fwhm / 2.0
                area = A * (
                    (1.0 - eta) * sg * np.sqrt(2.0 * np.pi) + eta * gm * np.pi
                )
                extra = {"eta": float(eta)}
                y_fit = _pseudo_voigt(x_c, *popt)

    except Exception:
        return _nan

    return dict(
        center=float(center + x0r),
        amplitude=float(A),
        fwhm=float(fwhm),
        area=float(area),
        residual=float(np.sqrt(np.mean((y_net - y_fit) ** 2))),
        success=True,
        **{k: float(v) for k, v in extra.items()},
    )


# ---------------------------------------------------------------------------
# Convenience file-level wrapper
# ---------------------------------------------------------------------------

def fit_peak_from_file(
    xy_file: Path | str,
    center: float,
    window: float,
    model: Literal["gaussian", "lorentzian", "voigt", "pseudo_voigt"] = "pseudo_voigt",
    bg_method: str = "snip",
    bg_kwargs: dict[str, Any] | None = None,
) -> dict:
    """
    Load a ``.xy`` diffraction file and fit a single peak.

    Convenience wrapper around :func:`fit_peak`.

    Args:
        xy_file (Path or str): Path to the ``.xy`` file.
        center (float): Nominal peak centre in degrees.
        window (float): Total window width in degrees.
        model (str): Peak profile (default ``"pseudo_voigt"``).
        bg_method (str): Background algorithm (default ``"snip"``).
        bg_kwargs (dict or None): Extra kwargs for the background estimator.

    Returns:
        dict: Same as :func:`fit_peak`.

    Example::

        result = fit_peak_from_file("pattern.xy", center=3.56, window=0.4)
        print(result["fwhm"], result["center"])
    """
    arrays = read_xy_file(xy_file)
    tth, intensity = arrays[0], arrays[1]
    return fit_peak(tth, intensity, center, window, model, bg_method, bg_kwargs)
