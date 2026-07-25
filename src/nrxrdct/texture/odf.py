"""
Pole-figure extraction for texture tomography.

This is the texture-tomography analogue of :mod:`nrxrdct.xrdct.sinogram`:
instead of assembling a powder (2theta) sinogram, it extracts azimuthal
(chi) intensity profiles around a chosen hkl Debye-Scherrer ring from
cake-integrated diffraction images acquired by the same XRD-CT scan
(:class:`nrxrdct.xrdct.parameters.Scan`, :mod:`nrxrdct.azimuthal.integration`),
and converts scan geometry (2theta, chi, sample rotation) into pole-figure
coordinates on the sample's orientation sphere.

This module stops at pole-figure data: it does not invert pole figures into
an orientation distribution function (ODF). The output is the per-scan-
position raw material an ODF inversion (harmonic series, WIMV, ...) would
consume, analogous to how :func:`nrxrdct.xrdct.sinogram.assemble_sinogram`
produces the sinogram that :func:`nrxrdct.xrdct.reconstruction.reconstruct_slice`
later consumes.

Diffraction geometry convention
--------------------------------
:func:`pole_figure_coordinates` assumes a single vertical rotation axis
(the XRD-CT ``rot`` motor) and a horizontal incident beam, with the
azimuthal (chi) angle of :func:`~nrxrdct.azimuthal.integration.cake_integration`
measured from the vertical (rotation-axis) direction on the detector,
increasing towards the horizontal direction. Beamlines differ in where
chi=0 points and which way it increases; use *chi_offset_deg* and
*chi_sign* to align the convention with your setup before trusting the
resulting pole-figure angles.

As a concrete data point (not a substitute for calibrating your own setup):
pyFAI's own azimuthal-angle convention (``AzimuthalIntegrator.chiArray`` /
the *azimuthal* axis returned by :func:`~nrxrdct.azimuthal.integration.cake_integration`)
was verified empirically to be ``chi = atan2(row_offset, col_offset)`` relative
to the detector's beam-center pixel — i.e. chi=0 along the **horizontal**
(column) detector axis, not vertical as this module's derivation assumes by
default. A ``chi_offset_deg`` of roughly 90 (sign depending on your detector's
row/column handedness) is therefore a more likely starting point than 0 for
data straight out of ``cake_integration`` — but detector rotation/flip flags
can shift this per beamline, so verify rather than assume.
"""
from pathlib import Path
from typing import Optional, Tuple

import fabio
import h5py
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm  # type: ignore

from ..azimuthal.integration import cake_integration
from ..utils import calculate_padding_widths_2D


def extract_ring_intensity(
    cake: np.ndarray,
    radial: np.ndarray,
    azimuthal: np.ndarray,
    tth_range: Tuple[float, float],
    background_ranges: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract the azimuthal (chi) intensity profile of one hkl ring from a
    cake-integrated image.

    Args:
        cake (np.ndarray): 2-D CAKE image of shape ``(npt_azim, npt_rad)``, as
            returned by :func:`nrxrdct.azimuthal.integration.cake_integration`.
        radial (np.ndarray): Radial (2theta) axis of *cake*, shape ``(npt_rad,)``.
        azimuthal (np.ndarray): Azimuthal (chi) axis of *cake* in degrees, shape
            ``(npt_azim,)``.
        tth_range (tuple): ``(low, high)`` 2theta window bracketing the hkl ring.
        background_ranges (tuple, optional): ``((low1, high1), (low2, high2))`` pair
            of flanking 2theta windows used to estimate a per-chi-bin background,
            which is subtracted from the ring intensity. Pass ``None`` to skip
            background subtraction (default ``None``).

    Returns:
        azimuthal (np.ndarray): The input *azimuthal* axis, unchanged.
        intensity (np.ndarray): Ring intensity per azimuthal bin, shape ``(npt_azim,)``.

    Raises:
        ValueError: If *tth_range* (or *background_ranges*) does not overlap *radial*.
    """
    lo, hi = tth_range
    peak_mask = (radial >= lo) & (radial <= hi)
    if not np.any(peak_mask):
        raise ValueError(
            f"tth_range {tth_range} does not overlap radial axis "
            f"[{radial.min():.4g}, {radial.max():.4g}]"
        )
    intensity = np.nanmean(cake[:, peak_mask], axis=1)

    if background_ranges is not None:
        (blo1, bhi1), (blo2, bhi2) = background_ranges
        bg_mask = ((radial >= blo1) & (radial <= bhi1)) | ((radial >= blo2) & (radial <= bhi2))
        if not np.any(bg_mask):
            raise ValueError(
                f"background_ranges {background_ranges} do not overlap radial axis "
                f"[{radial.min():.4g}, {radial.max():.4g}]"
            )
        background = np.nanmean(cake[:, bg_mask], axis=1)
        intensity = intensity - background

    return azimuthal, intensity


def pole_figure_coordinates(
    tth_deg,
    chi_deg,
    omega_deg,
    chi_offset_deg: float = 0.0,
    chi_sign: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert diffraction-geometry angles into pole-figure coordinates.

    Assumes a vertical rotation axis, a horizontal incident beam, and *chi*
    measured from the vertical (rotation-axis) direction as described in this
    module's docstring. *tth_deg*, *chi_deg*, and *omega_deg* may be scalars or
    arrays that broadcast together (e.g. a 1-D *chi_deg* array combined with a
    scalar *omega_deg* for one frame, or a 2-D grid built from
    ``np.meshgrid(chi_deg, omega_deg)`` for a full scan).

    Args:
        tth_deg (float or np.ndarray): Full scattering angle 2theta of the hkl
            ring, in degrees.
        chi_deg (float or np.ndarray): Azimuthal angle on the detector, in degrees,
            as returned by :func:`nrxrdct.azimuthal.integration.cake_integration`.
        omega_deg (float or np.ndarray): Sample rotation angle (the XRD-CT ``rot``
            motor) at which the frame was acquired, in degrees.
        chi_offset_deg (float, optional): Additive correction (degrees) aligning
            this beamline's chi=0 direction with the convention assumed here
            (default 0.0).
        chi_sign (float, optional): ``+1`` or ``-1``, flips the sense in which chi
            increases if it runs opposite to the assumed convention (default 1.0).

    Returns:
        alpha_deg (np.ndarray): Polar angle from the rotation axis, in ``[0, 180]``
            degrees, same broadcast shape as the inputs.
        beta_deg (np.ndarray): Azimuthal angle about the rotation axis, in
            ``[0, 360)`` degrees, same broadcast shape as the inputs.
    """
    theta = np.deg2rad(np.asarray(tth_deg, dtype=np.float64) / 2.0)
    chi = np.deg2rad(chi_sign * np.asarray(chi_deg, dtype=np.float64) + chi_offset_deg)
    omega = np.deg2rad(np.asarray(omega_deg, dtype=np.float64))

    # Scattering-vector direction in the lab frame (beam along +x, rotation
    # axis +z); see the geometry derivation in this module's docstring.
    qx = -np.sin(theta)
    qy = np.cos(theta) * np.sin(chi)
    qz = np.cos(theta) * np.cos(chi)

    # Rotate by -omega about the rotation axis (z) to express the pole in the
    # sample-fixed frame.
    qx_s = qx * np.cos(omega) + qy * np.sin(omega)
    qy_s = -qx * np.sin(omega) + qy * np.cos(omega)
    qz_s = qz

    alpha_deg = np.rad2deg(np.arccos(np.clip(qz_s, -1.0, 1.0)))
    beta_deg = np.rad2deg(np.arctan2(qy_s, qx_s)) % 360.0

    return alpha_deg, beta_deg


def assemble_pole_figure_data(
    master_file: Path,
    output_file: Path,
    poni_file: Path,
    mask_file: Path,
    tth_range: Tuple[float, float],
    hkl_label: str,
    background_ranges: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
    npt_rad: int = 1000,
    npt_azim: int = 360,
    n_workers: int = 16,
    camera_name: str = "eiger",
    monitor_name: str = "fpico6",
    translation_motor: str = "dty",
    rotation_motor: str = "rot",
) -> None:
    """
    Extract per-frame pole-figure (chi-intensity) profiles for one hkl ring
    from a raw XRD-CT master file, in parallel.

    Mirrors :func:`nrxrdct.azimuthal.integration.integrate_powder_parallel`
    (same master-file layout, entry validation, and monitor normalisation) but
    cake-integrates each frame and reduces it to an azimuthal intensity
    profile via :func:`extract_ring_intensity`, instead of a 1-D powder
    pattern. Results are appended to *output_file* under
    ``pole_figures/<hkl_label>/scan_XXXX``; scans already present are skipped,
    so a failed run can be resumed by calling this function again.

    Args:
        master_file (Path): Path to the master HDF5 file containing all scan entries.
        output_file (Path): Path to the output HDF5 file.
        poni_file (Path): Path to the PONI calibration file.
        mask_file (Path): Path to the mask file (fabio-readable).
        tth_range (tuple): ``(low, high)`` 2theta window bracketing the hkl ring,
            passed to :func:`extract_ring_intensity`.
        hkl_label (str): Identifier for the ring being extracted (e.g. ``"111"``),
            used as an HDF5 group name under ``pole_figures/``.
        background_ranges (tuple, optional): Flanking 2theta windows passed to
            :func:`extract_ring_intensity` for background subtraction
            (default ``None``).
        npt_rad (int, optional): Number of radial bins used internally by
            :func:`~nrxrdct.azimuthal.integration.cake_integration` (default 1000).
        npt_azim (int, optional): Number of azimuthal (chi) bins (default 360).
        n_workers (int, optional): Number of parallel cake-integration threads
            (default 16).
        camera_name (str, optional): Detector dataset name under ``measurement/``
            (default ``"eiger"``). Change this to match your beamline.
        monitor_name (str, optional): Monitor/ion-chamber dataset name under
            ``measurement/`` used for normalisation (default ``"fpico6"``).
        translation_motor (str, optional): Translation-motor name read from
            ``instrument/positioners/`` in each entry (default ``"dty"``).
        rotation_motor (str, optional): Rotation-motor dataset name under
            ``measurement/`` (default ``"rot"``).
    """
    mask = fabio.open(mask_file).data

    print("Reading entries from master file...")
    valid_entries, bad_entries, dty_values = [], [], []

    with h5py.File(master_file, "r") as hin:
        all_entries = list(hin.keys())
        for entry in tqdm(all_entries, desc="Validating entries"):
            try:
                _ = hin[f"{entry}/measurement/{camera_name}"].shape
                _ = hin[f"{entry}/measurement/{monitor_name}"].shape
                dty = float(hin[f"{entry}/instrument/positioners/{translation_motor}"][()])
                valid_entries.append(entry)
                dty_values.append(dty)
            except KeyError as e:
                print(f"  ⚠  Entry {entry} missing expected dataset ({e}) — skipping")
                bad_entries.append(entry)

    print(f"\n✓  {len(valid_entries)}/{len(all_entries)} entries OK")
    if bad_entries:
        print(f"⚠  Skipping {len(bad_entries)} incomplete entries: {bad_entries}\n")

    group = f"pole_figures/{hkl_label}"

    with h5py.File(output_file, "a") as hout:
        if f"{group}/azimuthal" not in hout:
            azimuthal_axis = -180.0 + (360.0 / npt_azim) * (np.arange(npt_azim) + 0.5)
            hout[f"{group}/azimuthal"] = azimuthal_axis
            hout[f"{group}/tth_range"] = np.asarray(tth_range, dtype=np.float64)
        if f"motors/{translation_motor}" not in hout:
            hout[f"motors/{translation_motor}"] = dty_values

    def process_frame(image: np.ndarray) -> np.ndarray:
        cake, radial, azimuthal = cake_integration(
            image, str(poni_file), npt_rad=npt_rad, npt_azim=npt_azim, mask=mask
        )
        _, intensity = extract_ring_intensity(
            cake, radial, azimuthal, tth_range, background_ranges
        )
        return intensity

    for ii, entry in enumerate(valid_entries):
        scan_name = f"scan_{ii:04d}"
        group_path = f"{group}/{scan_name}"

        with h5py.File(output_file, "r") as hout:
            if group_path in hout:
                print(f"Skipping {scan_name} (already done)")
                continue

        print(
            f"\n{'='*60}\nProcessing {scan_name} — entry {entry}  [{ii+1}/{len(valid_entries)}]\n{'='*60}"
        )

        try:
            with h5py.File(master_file, "r") as hin:
                images = hin[f"{entry}/measurement/{camera_name}"][:].astype(np.float32)
                monitor = hin[f"{entry}/measurement/{monitor_name}"][:].astype(np.float64)
                rot = hin[f"{entry}/measurement/{rotation_motor}"][:]
        except OSError as e:
            print(f"  ✗ Failed to read entry {entry}: {e} — skipping")
            continue

        if rot[-1] < rot[0]:
            images = images[::-1]
            monitor = monitor[::-1]
            rot = rot[::-1]

        n_frames = len(images)
        profiles = np.empty((n_frames, npt_azim), dtype=np.float32)

        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(process_frame, images[jj]): jj for jj in range(n_frames)}
            for future in tqdm(as_completed(futures), total=n_frames, desc=scan_name):
                jj = futures[future]
                try:
                    intensity = future.result()
                    profiles[jj] = intensity / monitor[jj] if monitor[jj] > 0 else intensity
                except Exception as e:
                    print(f"  ✗ Frame {jj} failed: {e}")
                    profiles[jj] = np.nan

        with h5py.File(output_file, "a") as hout:
            ds = hout.create_dataset(
                group_path,
                data=profiles,
                compression="gzip",
                compression_opts=4,
                chunks=(1, npt_azim),
            )
            ds.attrs["entry"] = entry
            ds.attrs[translation_motor] = dty_values[ii]
            ds.attrs["source"] = str(master_file)
            ds.attrs[rotation_motor] = rot


def assemble_pole_figure_sinogram(
    pole_figure_file: Path,
    hkl_label: str,
    n_rot: int,
    translation_motor: str = "dty",
    rotation_motor: str = "rot",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Stack per-scan pole-figure profiles written by :func:`assemble_pole_figure_data`
    into a texture-tomography sinogram.

    Mirrors :func:`nrxrdct.xrdct.sinogram.assemble_sinogram`: scans are sorted by
    the recorded *translation_motor* value (not file order, for the same reason
    documented there) and zero-padded along the rotation axis to *n_rot*.

    Args:
        pole_figure_file (Path): HDF5 file written by :func:`assemble_pole_figure_data`.
        hkl_label (str): Ring identifier passed to :func:`assemble_pole_figure_data`.
        n_rot (int): Number of rotation steps (sinogram angular dimension).
        translation_motor (str, optional): Name of the translation-motor attribute
            stored on each scan group (default ``"dty"``).
        rotation_motor (str, optional): Name of the rotation-motor attribute stored
            on each scan group, used to recover the physical rotation axis for
            *rot_deg* (default ``"rot"``).

    Returns:
        sino (np.ndarray): Pole-figure sinogram of shape ``(n_chi, n_lines, n_rot)``
            as ``float32``, where ``n_chi`` is the number of azimuthal bins.
        azimuthal (np.ndarray): Chi axis of *sino*'s first dimension, in degrees.
        dty (np.ndarray): Translation-motor value for each line of *sino*, sorted
            to match its translation axis.
        rot_deg (np.ndarray): Rotation angle for each step of *sino*'s rotation
            axis, in degrees, shape ``(n_rot,)``. Taken from the first scan whose
            stored rotation array already has length *n_rot* (i.e. needed no
            padding); if every scan was padded, falls back to
            ``np.linspace(rot_min, rot_max, n_rot)`` over the observed range and
            prints a warning, since the true per-step spacing can't be recovered
            in that case.
    """
    group = f"pole_figures/{hkl_label}"

    with h5py.File(pole_figure_file, "r") as hin:
        azimuthal = hin[f"{group}/azimuthal"][:]
        n_chi = azimuthal.shape[0]

        keys = [key for key in hin[group].keys() if key.startswith("scan_")]
        dty_values = np.array(
            [hin[f"{group}/{key}"].attrs[translation_motor] for key in keys]
        )
        order = np.argsort(dty_values)
        valid_keys = [keys[i] for i in order]
        dty_values = dty_values[order]

        sino = np.zeros((len(valid_keys), n_rot, n_chi), dtype=np.float32)
        rot_deg = None
        rot_min, rot_max = np.inf, -np.inf
        for ii, scan in enumerate(valid_keys):
            ds = hin[f"{group}/{scan}"]
            profiles = ds[:]
            scan_rot = np.asarray(ds.attrs[rotation_motor], dtype=np.float64)
            rot_min = min(rot_min, float(scan_rot.min()))
            rot_max = max(rot_max, float(scan_rot.max()))
            if rot_deg is None and len(scan_rot) == n_rot:
                rot_deg = scan_rot

            padding_width = calculate_padding_widths_2D(profiles.shape, (n_rot, n_chi))
            sino[ii] = np.pad(profiles, padding_width)

        if rot_deg is None:
            print(
                "  ⚠  No scan's rotation array has length n_rot (every scan was "
                "padded) — falling back to a uniform linspace over the observed "
                "rotation range, which may not match the true per-step spacing."
            )
            rot_deg = np.linspace(rot_min, rot_max, n_rot)

    sino = np.rollaxis(sino, 2, 0)
    return np.rollaxis(sino, 1, 2), azimuthal, dty_values, rot_deg


def sinogram_to_pole_figure(
    sino: np.ndarray,
    azimuthal: np.ndarray,
    rot_deg: np.ndarray,
    tth_deg: float,
    line_index: Optional[int] = None,
    chi_offset_deg: float = 0.0,
    chi_sign: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert a pole-figure sinogram into flat pole-figure coordinates, ready for
    :func:`nrxrdct.texture.odf_inversion.compute_odf`.

    Applies :func:`pole_figure_coordinates` to every ``(chi, rot)`` grid point
    of *sino* and flattens the result — the direct bridge between
    :func:`assemble_pole_figure_sinogram`'s output and the bulk ODF fit.

    Args:
        sino (np.ndarray): Pole-figure sinogram of shape ``(n_chi, n_lines, n_rot)``,
            as returned by :func:`assemble_pole_figure_sinogram`.
        azimuthal (np.ndarray): Chi axis, degrees, shape ``(n_chi,)``.
        rot_deg (np.ndarray): Rotation-angle axis, degrees, shape ``(n_rot,)``.
        tth_deg (float): 2theta of the hkl ring (e.g. the mean of the
            ``tth_range`` stored alongside the sinogram).
        line_index (int, optional): Which translation line (index into *sino*'s
            second axis) to use. ``None`` (default) averages over all lines,
            producing one bulk pole figure for the whole scanned cross-section
            rather than one per line — pass an explicit index for a per-line
            bulk pole figure instead.
        chi_offset_deg (float, optional): Forwarded to :func:`pole_figure_coordinates`
            (default 0.0).
        chi_sign (float, optional): Forwarded to :func:`pole_figure_coordinates`
            (default 1.0).

    Returns:
        alpha_deg (np.ndarray): Polar angles, flat, shape ``(n_chi * n_rot,)``.
        beta_deg (np.ndarray): Azimuthal angles, flat, same shape as *alpha_deg*.
        intensity (np.ndarray): Sinogram intensity at each point, same shape.
    """
    if line_index is None:
        intensity_grid = sino.mean(axis=1)  # (n_chi, n_rot)
    else:
        intensity_grid = sino[:, line_index, :]  # (n_chi, n_rot)

    chi_grid, omega_grid = np.meshgrid(azimuthal, rot_deg, indexing="ij")
    alpha_deg, beta_deg = pole_figure_coordinates(
        tth_deg=tth_deg,
        chi_deg=chi_grid,
        omega_deg=omega_grid,
        chi_offset_deg=chi_offset_deg,
        chi_sign=chi_sign,
    )
    return alpha_deg.ravel(), beta_deg.ravel(), intensity_grid.ravel()