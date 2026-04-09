"""
Convenience plotting functions for XRD-CT integration results.
"""
import numpy as np
import matplotlib.pyplot as plt


def plot_integrated_cake(cake, radial, azimuthal, log_scale = False):
    """
    Display a 2-D CAKE integration result as a false-colour image.

    Args:
        cake (np.ndarray): 2-D CAKE array of shape ``(npt_azim, npt_rad)`` as
            returned by :func:`~nrxrdct.integration.cake_integration`.
        radial (np.ndarray): Radial axis values (length ``npt_rad``).
        azimuthal (np.ndarray): Azimuthal axis values in degrees (length ``npt_azim``).
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

