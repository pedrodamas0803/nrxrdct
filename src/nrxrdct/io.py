import os
from pathlib import Path

import h5py
import numpy as np


def save_sinogram(sinogram: np.ndarray, output_file: Path):

    with h5py.File(str(output_file), "a") as hout:
        hout["sinogram"] = sinogram


def save_volume(volume: np.ndarray, output_file: Path):

    with h5py.File(str(output_file), "a") as hout:
        hout["volume"] = volume


def add_array_to_output(array: np.ndarray, array_name: str, output_file: Path):
    with h5py.File(str(output_file), "a") as hout:
        hout[array_name] = array


def get_array_from_file(
    filename: Path,
    array_name: str,
):
    with h5py.File(str(filename), "a") as hin:
        out = hin[array_name][:]
    return out


def read_sinogram_from_file(input_file: Path, slicing: tuple | None = None):

    with h5py.File(input_file, "r") as hin:

        if isinstance(slicing, tuple):
            tthmin, tthmax, xmin, xmax, ymin, ymax = slicing
            sinogram = hin["sinogram"][tthmin:tthmax, xmin, xmax, ymin, ymax].astype(
                np.float32
            )
        else:
            sinogram = hin["sinogram"][:].astype(np.float32)

    return sinogram


def read_volume_from_file(input_file: Path, slicing: tuple | None = None):
    with h5py.File(input_file, "r") as hin:
        if isinstance(slicing, tuple):
            tthmin, tthmax, xmin, xmax, ymin, ymax = slicing
            volume = hin["volume"][tthmin:tthmax, xmin, xmax, ymin, ymax].astype(
                np.float32
            )
        else:
            volume = hin["volume"][:].astype(np.float32)

    return volume


def save_xy_file(
    x: np.array,
    y: np.array,
    err: np.array = None,
    output_file: Path = Path("integrated_data.xy"),
    unit: str = "2th_deg",
    verbose: bool = True,
):

    if not isinstance(err, np.ndarray):
        err = np.zeros_like(y)
    header = (
        f"# pyFAI multi-geometry azimuthal integration\n"
        f"# Unit: {unit}\n"
        f"# Columns: {unit}  Intensity  Sigma\n"
    )
    np.savetxt(str(output_file), np.column_stack([x, y]), fmt="%.6f")
    if verbose:
        print(f"Integrated pattern saved to:\n  {str(output_file)}")


def read_xy_file(input_file: Path = "integrated_data.xy"):

    return np.loadtxt(str(input_file), unpack=True)


def write_starting_instrument_pars(
    output_file: Path = Path("instrument_init.instprm"),
    polarization: float = 0.99,
    wavelength: float = 1.5418,
):
    """
    Wavelength must be in angstrom.
    """
    lines = [
        "#GSAS-II instrument parameter file\n",
        "Type:PXC\n",
        "Bank:1\n",
        f"Lam:{wavelength}\n",
        "Zero:0.0\n",
        f"Polariz.:{polarization}\n",
        "Azimuth:0.0\n",
        "U:0.0\n",
        "V:0.0\n",
        "W:1.0\n",
        "X:0.0\n",
        "Y:5.0\n",
        "Z:0.0\n",
        "SH/L:0.0001\n",
    ]
    if os.path.exists(str(output_file)):
        return output_file
    with open(str(output_file), "w") as f:
        f.writelines(lines)
    print(f"Starting instprm written to:\n  {str(output_file)}")

    return output_file


def write_calibrated_intrument_pars(
    hist,
    wavelength: float = 1.5418,
    output_file: Path = Path("calibrated_instrument.instprm"),
):
    """
    Wavelength must be in angstrom.
    """

    print("\n" + "=" * 60)
    print("Exporting calibrated instprm")
    print("=" * 60)

    ip = hist["Instrument Parameters"][0]
    lines = ["#GSAS-II instrument parameter file\n"]

    # Dual-wavelength keys written manually — values are physical constants,
    # not refined, so we hardcode them rather than reading from ip
    key_order_single = [
        "Type",
        "Bank",
        "Zero",
        "Polariz.",
        "Azimuth",
        "U",
        "V",
        "W",
        "X",
        "Y",
        "Z",
        "SH/L",
    ]

    lines.append(f"Lam:{wavelength}\n")

    for p in key_order_single:
        if p in ip:
            val = ip[p][1] if isinstance(ip[p], list) else ip[p]
            lines.append(f"{p}:{val}\n")

    with open(output_file, "w") as f:
        f.writelines(lines)

    print(f"Calibrated instprm saved to:\n  {output_file}")
    print("\nFinal calibrated parameters:")
    for line in lines[1:]:  # skip the header comment
        print(f"  {line.rstrip()}")

    print(f"Calibrated instprm saved to:\n  {output_file}")
    print("\nIn your sample refinements:")
    print("  - Use this file as INST_PARAMS")
    print("  - Fix Zero, W, X, Y (carry from calibration)")
    print("  - U, V, SH/L remain fixed at 0 / 0.0001")
