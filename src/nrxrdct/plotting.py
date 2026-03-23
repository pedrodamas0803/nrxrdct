import numpy as np
import matplotlib.pyplot as plt


def plot_integrated_cake(cake, radial, azimuthal, log_scale = False):
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

