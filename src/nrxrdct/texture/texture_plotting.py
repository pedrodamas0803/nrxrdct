"""
Pole-figure plotting.

Renders pole-figure data (measured, from :mod:`nrxrdct.texture.odf`, or
recalculated from a fitted ODF via
:func:`nrxrdct.texture.odf_inversion.recalculate_pole_figure`) as a standard
equal-area (Schmidt) upper-hemisphere projection.
"""
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def _fold_upper_hemisphere(alpha_deg: np.ndarray, beta_deg: np.ndarray):
    """Map (alpha, beta) with alpha>90 to their antipodal upper-hemisphere equivalent (Friedel's law)."""
    lower = alpha_deg > 90.0
    alpha_f = np.where(lower, 180.0 - alpha_deg, alpha_deg)
    beta_f = np.where(lower, (beta_deg + 180.0) % 360.0, beta_deg)
    return alpha_f, beta_f


def plot_pole_figure(
    alpha_deg: np.ndarray,
    beta_deg: np.ndarray,
    intensity: np.ndarray,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    cmap: str = "viridis",
    s: float = 20.0,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    colorbar: bool = True,
) -> plt.Axes:
    """
    Plot pole-figure data as an equal-area (Schmidt) upper-hemisphere projection.

    Points below the equator (alpha > 90 deg) are folded onto their antipodal
    upper-hemisphere equivalent per Friedel's law, so every point is shown.

    Args:
        alpha_deg (np.ndarray): Polar angle from the rotation/reference axis, in
            degrees, as returned by
            :func:`nrxrdct.texture.odf.pole_figure_coordinates`.
        beta_deg (np.ndarray): Azimuthal angle, in degrees, same shape as *alpha_deg*.
        intensity (np.ndarray): Pole-figure intensity, same shape as *alpha_deg*;
            used as the point colour.
        ax (matplotlib.axes.Axes, optional): Existing polar axes to draw into
            (must have been created with ``subplot_kw={"projection": "polar"}``).
            A new figure and axes are created when ``None`` (default).
        title (str, optional): Axes title (default ``None``).
        cmap (str, optional): Matplotlib colormap name (default ``"viridis"``).
        s (float, optional): Marker size passed to ``scatter`` (default 20.0).
        vmin (float, optional): Colour scale lower bound (default: data min).
        vmax (float, optional): Colour scale upper bound (default: data max).
        colorbar (bool, optional): Draw a colorbar next to the axes (default ``True``).

    Returns:
        matplotlib.axes.Axes: The polar axes the pole figure was drawn on.
    """
    if ax is None:
        _, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(5, 5))

    alpha_f, beta_f = _fold_upper_hemisphere(np.asarray(alpha_deg), np.asarray(beta_deg))
    r = np.sqrt(1 - np.cos(np.deg2rad(alpha_f)))
    theta = np.deg2rad(beta_f)

    sc = ax.scatter(theta, r, c=intensity, cmap=cmap, s=s, vmin=vmin, vmax=vmax)

    ax.set_rmax(1.0)
    ax.set_rticks([np.sqrt(1 - np.cos(np.deg2rad(a))) for a in (30, 60, 90)])
    ax.set_yticklabels(["30°", "60°", "90°"])
    ax.set_theta_zero_location("E")
    if title:
        ax.set_title(title)
    if colorbar:
        ax.figure.colorbar(sc, ax=ax, label="Intensity (a.u.)", shrink=0.7)

    return ax


def plot_pole_figure_comparison(
    alpha_deg: np.ndarray,
    beta_deg: np.ndarray,
    measured: np.ndarray,
    recalculated: np.ndarray,
    title: Optional[str] = None,
    cmap: str = "viridis",
    s: float = 20.0,
) -> np.ndarray:
    """
    Plot measured vs. ODF-recalculated pole-figure intensity side by side, on a
    shared colour scale — a standard fit-quality check after
    :func:`nrxrdct.texture.odf_inversion.compute_odf`.

    Args:
        alpha_deg (np.ndarray): Polar angles (degrees), shared by both pole figures.
        beta_deg (np.ndarray): Azimuthal angles (degrees), same shape as *alpha_deg*.
        measured (np.ndarray): Measured pole-figure intensity, same shape as *alpha_deg*.
        recalculated (np.ndarray): ODF-recalculated intensity at the same points,
            e.g. from :func:`nrxrdct.texture.odf_inversion.recalculate_pole_figure`.
        title (str, optional): Figure suptitle (default ``None``).
        cmap (str, optional): Matplotlib colormap name (default ``"viridis"``).
        s (float, optional): Marker size passed to ``scatter`` (default 20.0).

    Returns:
        np.ndarray: The two polar axes, ``[ax_measured, ax_recalculated]``.
    """
    vmin = float(min(np.min(measured), np.min(recalculated)))
    vmax = float(max(np.max(measured), np.max(recalculated)))

    fig, axes = plt.subplots(1, 2, subplot_kw={"projection": "polar"}, figsize=(10, 5))
    plot_pole_figure(alpha_deg, beta_deg, measured, ax=axes[0], title="Measured",
                      cmap=cmap, s=s, vmin=vmin, vmax=vmax, colorbar=False)
    plot_pole_figure(alpha_deg, beta_deg, recalculated, ax=axes[1], title="Recalculated (ODF)",
                      cmap=cmap, s=s, vmin=vmin, vmax=vmax, colorbar=False)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    fig.colorbar(sm, ax=axes, label="Intensity (a.u.)", shrink=0.7)
    if title:
        fig.suptitle(title)

    return axes