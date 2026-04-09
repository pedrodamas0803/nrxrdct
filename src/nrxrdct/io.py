"""
I/O helpers for XRD-CT data and instrument parameter files.

Functions cover reading and writing sinograms, reconstructed volumes, and
integrated patterns (HDF5 / plain-text .xy), as well as generating GSAS-II
instrument parameter files (``.instprm``).
"""
import os
from pathlib import Path

import h5py
import numpy as np


def save_sinogram(sinogram: np.ndarray, output_file: Path):
    """
    Save a sinogram array to an HDF5 file under the key ``"sinogram"``.

    Args:
        sinogram (np.ndarray): Sinogram data to store.
        output_file (Path): Destination HDF5 file path (opened in append mode).
    """
    with h5py.File(str(output_file), "a") as hout:
        hout["sinogram"] = sinogram


def save_volume(volume: np.ndarray, output_file: Path):
    """
    Save a reconstructed volume array to an HDF5 file under the key ``"volume"``.

    Args:
        volume (np.ndarray): Volume data to store.
        output_file (Path): Destination HDF5 file path (opened in append mode).
    """
    with h5py.File(str(output_file), "a") as hout:
        hout["volume"] = volume


def add_array_to_output(array: np.ndarray, array_name: str, output_file: Path):
    """
    Append an array to an HDF5 file under an arbitrary key.

    Args:
        array (np.ndarray): Data to store.
        array_name (str): HDF5 dataset key.
        output_file (Path): Destination HDF5 file path (opened in append mode).
    """
    with h5py.File(str(output_file), "a") as hout:
        hout[array_name] = array


def get_array_from_file(
    filename: Path,
    array_name: str,
):
    """
    Read an array from an HDF5 file by key name.

    Args:
        filename (Path): HDF5 file to read from.
        array_name (str): HDF5 dataset key.

    Returns:
        np.ndarray: The dataset loaded into memory.
    """
    with h5py.File(str(filename), "a") as hin:
        out = hin[array_name][:]
    return out


def read_sinogram_from_file(input_file: Path, slicing: tuple | None = None):
    """
    Read a sinogram from an HDF5 file, optionally with sub-region slicing.

    Args:
        input_file (Path): HDF5 file containing a ``"sinogram"`` dataset.
        slicing (tuple or None, optional): If provided, a 6-element tuple
            ``(tthmin, tthmax, xmin, xmax, ymin, ymax)`` used to extract a
            sub-region. If ``None``, the full sinogram is loaded.

    Returns:
        np.ndarray: Sinogram data as ``float32``.
    """
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
    """
    Read a reconstructed volume from an HDF5 file, optionally with sub-region slicing.

    Args:
        input_file (Path): HDF5 file containing a ``"volume"`` dataset.
        slicing (tuple or None, optional): If provided, a 6-element tuple
            ``(tthmin, tthmax, xmin, xmax, ymin, ymax)`` used to extract a
            sub-region. If ``None``, the full volume is loaded.

    Returns:
        np.ndarray: Volume data as ``float32``.
    """
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
    """
    Save an integrated 1-D diffraction pattern to a plain-text .xy file.

    Args:
        x (np.ndarray): Scattering axis values (e.g. 2-theta in degrees).
        y (np.ndarray): Intensity values.
        err (np.ndarray or None, optional): Per-point uncertainties. Defaults to
            an array of zeros when not provided.
        output_file (Path, optional): Destination file path
            (default ``"integrated_data.xy"``).
        unit (str, optional): Label for the scattering-angle axis written into
            the header (default ``"2th_deg"``).
        verbose (bool, optional): If ``True``, print the output path after saving
            (default ``True``).
    """
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
    """
    Load a plain-text .xy diffraction pattern file.

    Args:
        input_file (Path, optional): Source file path (default ``"integrated_data.xy"``).

    Returns:
        tuple: Columns unpacked as separate arrays (e.g. ``(x, y)`` or ``(x, y, err)``).
    """
    return np.loadtxt(str(input_file), unpack=True)


def write_starting_instrument_pars(
    output_file: Path = Path("instrument_init.instprm"),
    polarization: float = 0.99,
    wavelength: float = 1.5418,
):
    """
    Write a minimal GSAS-II instrument parameter file with starting (unfitted) values.

    The file is skipped without error if it already exists, so this function is
    safe to call at the start of a workflow.

    Args:
        output_file (Path, optional): Destination ``.instprm`` file
            (default ``"instrument_init.instprm"``).
        polarization (float, optional): Beam polarization fraction (default 0.99).
        wavelength (float, optional): Incident wavelength in angstrom
            (default 1.5418 Å, Cu Kα).

    Returns:
        Path: Path to the (existing or newly created) instrument parameter file.
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
    Export a calibrated GSAS-II instrument parameter file from a refined histogram.

    Reads the refined instrument parameters from ``hist["Instrument Parameters"][0]``
    and writes them to a ``.instprm`` file alongside the specified wavelength.

    Args:
        hist (dict): GSAS-II histogram object (e.g. ``gpx.histogram(0)``).
        wavelength (float, optional): Incident wavelength in angstrom to embed in
            the file (default 1.5418 Å).
        output_file (Path, optional): Destination ``.instprm`` file
            (default ``"calibrated_instrument.instprm"``).
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
