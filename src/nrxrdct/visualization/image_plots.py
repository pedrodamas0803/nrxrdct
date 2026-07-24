"""Simple static (non-interactive) 2-D array display helpers."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


def plot_integrated_cake(cake, radial, azimuthal, unit: str = "2th_deg", log_scale: bool = False) -> None:
    """
    Display a 2-D CAKE integration result as a false-colour image.

    Args:
        cake (np.ndarray): 2-D CAKE array of shape ``(npt_azim, npt_rad)`` as
            returned by :func:`~nrxrdct.integration.cake_integration`.
        radial (np.ndarray): Radial axis values (length ``npt_rad``).
        azimuthal (np.ndarray): Azimuthal axis values in degrees (length ``npt_azim``).
        unit (str, optional): Label for the radial axis (default ``"2th_deg"``).
        log_scale (bool, optional): If ``True``, display ``log(1 + I)`` instead of
            raw intensity (default ``False``).
    """
    display_data = np.log1p(np.clip(cake, 0, None)) if log_scale else cake

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(
        display_data,
        origin="lower",
        aspect="auto",
        extent=[radial.min(), radial.max(), azimuthal.min(), azimuthal.max()],
        cmap="turbo",
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("log(1 + I)" if log_scale else "Intensity")
    ax.set_xlabel(unit)
    ax.set_ylabel("Azimuthal angle χ (°)")
    ax.set_title("CAKE integration")
    plt.tight_layout()


def plot_labeled_image(label_img_rgb, regionprops, cmap="turbo"):
    """
    Show labeled image and plot bounding boxes of regionprops.
    """
    f, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(label_img_rgb, cmap=cmap)

    for region in regionprops:
        minr, minc, maxr, maxc = region.bbox

        rect = Rectangle(
            (minc - 1, minr - 1),
            (maxc - minc),
            (maxr - minr),
            fill=False,
            edgecolor="red",
            linewidth=0.5,
        )
        ax.add_patch(rect)

    f.tight_layout()
    return