from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xrayutilities as xu

from .io import save_xy_file


def generate_circular_mask(shape, center, diameter):

    x, y = np.arange(0, shape[1]), np.arange(0, shape[2])
    X, Y = np.meshgrid(x, y)
    z = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
    mask = z < diameter //2
    return mask


def simulate_powder_xrd_monophase(
    tth,
    cif_files,
    do_plot=True,
    en_eV=100000,
    crystallite_size: float = 100e-9,
    do_save: bool = True,
):
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

    Parameters
    ----------
    cif_files : path or list of paths to CIF files
    en_eV     : X-ray energy in eV (default 100 keV)
    tth_min   : minimum 2theta in degrees (optional, auto if None)
    tth_max   : maximum 2theta in degrees (optional, auto if None)

    Returns
    -------
    dict mapping phase_name -> DataFrame with columns:
        h, k, l, hkl, tth, d_hkl, r (structure factor |F|²)
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

    return results


def calculate_padding_widths_2D(input_shape: tuple, desired_shape: tuple):

    y_in, x_in = input_shape
    y_des, x_des = desired_shape

    y_beg = (y_des - y_in) // 2
    y_end = (y_des - y_in) // 2 + (y_des - y_in) % 2

    x_beg = (x_des - x_in) // 2
    x_end = (x_des - x_in) // 2 + (x_des - x_in) % 2

    return ((y_beg, y_end), (x_beg, x_end))
