import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple

import fabio
import h5py
import matplotlib.pyplot as plt
import numpy as np
from pyFAI.integrator.azimuthal import AzimuthalIntegrator
from tqdm import tqdm


def cake_integration(
    image: np.ndarray,
    poni_file: str,
    npt_rad: int = 1000,
    npt_azim: int = 360,
    unit: str = "2th_deg",
    mask: Optional[np.ndarray] = None,
    dark: Optional[np.ndarray] = None,
    flat: Optional[np.ndarray] = None,
    radial_range: Optional[Tuple[float, float]] = None,
    azimuth_range: Optional[Tuple[float, float]] = (-180, 180),
    plot: bool = False,
    log_scale: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform 2D azimuthal regrouping (CAKE) of a detector image using pyFAI.

    Parameters
    ----------
    image : np.ndarray
        2D detector image.
    poni_file : str
        Path to the PONI calibration file.
    npt_rad : int
        Number of radial bins (x-axis resolution), default 1000.
    npt_azim : int
        Number of azimuthal bins (y-axis resolution), default 360 (1°/bin).
    unit : str
        Radial unit: "q_A^-1", "q_nm^-1", "2th_deg", "2th_rad", "r_mm".
    mask : np.ndarray, optional
        Mask array (1 = ignore, 0 = valid).
    dark : np.ndarray, optional
        Dark current image to subtract.
    flat : np.ndarray, optional
        Flat field image for correction.
    radial_range : tuple, optional
        (min, max) radial range in the chosen unit.
    azimuth_range : tuple, optional
        (min, max) azimuthal range in degrees, default (-180, 180).
    plot : bool
        If True, display the CAKE image with matplotlib.
    log_scale : bool
        If True, plot on a log intensity scale (recommended).

    Returns
    -------
    cake : np.ndarray
        2D CAKE image array, shape (npt_azim, npt_rad).
    radial : np.ndarray
        Radial axis values (length npt_rad).
    azimuthal : np.ndarray
        Azimuthal axis values in degrees (length npt_azim).
    """
    if image.ndim != 2:
        raise ValueError(f"image must be 2D, got shape {image.shape}")

    ai = AzimuthalIntegrator()
    ai.load(poni_file)

    result = ai.integrate2d(
        image,
        npt_rad=npt_rad,
        npt_azim=npt_azim,
        unit=unit,
        mask=mask,
        dark=dark,
        flat=flat,
        radial_range=radial_range,
        azimuth_range=azimuth_range,
        method=("no", "histogram", "cython"),
    )

    cake = result.intensity  # shape (npt_azim, npt_rad)
    radial = result.radial  # shape (npt_rad,)
    azimuthal = result.azimuthal  # shape (npt_azim,)



    return cake, radial, azimuthal



@lru_cache(maxsize=8)
def _get_integrator(poni_file: str) -> AzimuthalIntegrator:
    """
    Load and cache an AzimuthalIntegrator for a given PONI file.
    Avoids reloading the geometry on every integration call.
    """
    ai = AzimuthalIntegrator()
    ai.load(poni_file)
    return ai


def azimuthal_integration_1d_sigma_clip(
    image: np.ndarray,
    poni_file: str,
    npt: int = 1000,
    unit: str = "2th_deg",
    mask: Optional[np.ndarray] = None,
    dark: Optional[np.ndarray] = None,
    flat: Optional[np.ndarray] = None,
    error_model: Optional[str] = "hybrid",
    radial_range: Optional[Tuple[float, float]] = None,
    azimuth_range: Optional[Tuple[float, float]] = None,
    thres: float = 3.0,
    max_iter: int = 5,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Perform 1D azimuthal integration with sigma-clipping using pyFAI.

    Sigma-clipping iteratively removes pixels whose intensity deviates
    more than `thres` standard deviations from the bin mean, which is
    useful for rejecting hot pixels, zingers, and diffraction spots
    from a polycrystalline or single-crystal background.

    The AzimuthalIntegrator is cached per PONI file so repeated calls
    (e.g. looping over thousands of frames) do not reload the geometry.

    Parameters
    ----------
    image : np.ndarray
        2D detector image as a numpy array.
    poni_file : str
        Path to the PONI (Point Of Normal Incidence) calibration file.
    npt : int, optional
        Number of radial bins in the output pattern (default: 1000).
    unit : str, optional
        Output radial unit:
            "2th_deg" – two-theta in degrees (default)
            "q_A^-1"  – scattering vector q in Å⁻¹
            "q_nm^-1" – scattering vector q in nm⁻¹
            "2th_rad" – two-theta in radians
            "r_mm"    – radius on detector in mm
    mask : np.ndarray, optional
        2D mask array (1 = masked/ignored, 0 = valid).
    dark : np.ndarray, optional
        Dark-current image to subtract before integration.
    flat : np.ndarray, optional
        Flat-field image for pixel-efficiency correction.
    error_model : str, optional
        Error model for uncertainty propagation.
        "hybrid"  – combines Poisson and readout noise (default).
        "poisson" – pure photon-counting noise.
    radial_range : tuple of (float, float), optional
        (min, max) radial range in the chosen unit.
    azimuth_range : tuple of (float, float), optional
        (min, max) azimuthal range in degrees, e.g. (-180, 180).
    thres : float, optional
        Sigma-clipping threshold in units of standard deviations (default: 3.0).
    max_iter : int, optional
        Maximum number of sigma-clipping iterations (default: 5).

    Returns
    -------
    radial : np.ndarray
        Radial axis values in the requested unit, shape (npt,).
    intensity : np.ndarray
        Integrated intensity at each radial position, shape (npt,).
    sigma : np.ndarray or None
        Per-bin uncertainty (only when error_model is set).

    Raises
    ------
    ValueError
        If the image is not 2-dimensional.
    """
    if image.ndim != 2:
        raise ValueError(f"image must be 2D, got shape {image.shape}")

    ai = _get_integrator(poni_file)

    result = ai.sigma_clip(
        image,
        npt=npt,
        unit=unit,
        mask=mask,
        dark=dark,
        flat=flat,
        error_model=error_model,
        radial_range=radial_range,
        azimuth_range=azimuth_range,
        thres=thres,
        max_iter=max_iter,
        method=("no", "csr", "cython"),
    )

    return result.radial, result.intensity, result.sigma




@lru_cache(maxsize=8)
def _get_integrator(poni_file: str) -> AzimuthalIntegrator:
    """
    Load and cache an AzimuthalIntegrator for a given PONI file.
    Avoids reloading the geometry on every integration call.
    """
    ai = AzimuthalIntegrator()
    ai.load(poni_file)
    return ai


def azimuthal_integration_1d(
    image: np.ndarray,
    poni_file: str,
    npt: int = 1000,
    unit: str = "2th_deg",
    mask: Optional[np.ndarray] = None,
    dark: Optional[np.ndarray] = None,
    flat: Optional[np.ndarray] = None,
    error_model: Optional[str] = None,
    radial_range: Optional[Tuple[float, float]] = None,
    azimuth_range: Optional[Tuple[float, float]] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Perform 1D azimuthal integration of a detector image using pyFAI.

    Parameters
    ----------
    image : np.ndarray
        2D detector image as a numpy array.
    poni_file : str
        Path to the PONI (Point Of Normal Incidence) calibration file.
    npt : int, optional
        Number of radial bins in the output pattern (default: 1000).
    unit : str, optional
        Output radial unit:
            "2th_deg" – two-theta in degrees (default)
            "q_A^-1"  – scattering vector q in Å⁻¹
            "q_nm^-1" – scattering vector q in nm⁻¹
            "2th_rad" – two-theta in radians
            "r_mm"    – radius on detector in mm
    mask : np.ndarray, optional
        2D mask array (1 = masked/ignored, 0 = valid).
    dark : np.ndarray, optional
        Dark-current image to subtract before integration.
    flat : np.ndarray, optional
        Flat-field image for pixel-efficiency correction.
    error_model : str, optional
        Error model for uncertainty propagation.
        "poisson" for photon-counting detectors.
    radial_range : tuple of (float, float), optional
        (min, max) radial range in the chosen unit.
    azimuth_range : tuple of (float, float), optional
        (min, max) azimuthal range in degrees, e.g. (-180, 180).

    Returns
    -------
    radial : np.ndarray
        Radial axis values in the requested unit, shape (npt,).
    intensity : np.ndarray
        Integrated intensity at each radial position, shape (npt,).
    sigma : np.ndarray or None
        Per-bin uncertainty (only when error_model is set).

    Raises
    ------
    ValueError
        If the image is not 2-dimensional.

    Examples
    --------
    >>> image = np.random.poisson(1000, (2048, 2048)).astype(np.float32)
    >>> q, I, sigma = azimuthal_integration_1d(image, "detector.poni", npt=500)
    >>> print(q.shape, I.shape)  # (500,) (500,)
    """
    if image.ndim != 2:
        raise ValueError(f"image must be 2D, got shape {image.shape}")

    ai = _get_integrator(poni_file)

    result = ai.integrate1d(
        image,
        npt=npt,
        unit=unit,
        mask=mask,
        dark=dark,
        flat=flat,
        error_model=error_model,
        radial_range=radial_range,
        azimuth_range=azimuth_range,
        method=("no", "histogram", "cython"),
    )

    return result.radial, result.intensity, result.sigma


def azimuthal_integration_1d_filter(
    image: np.ndarray,
    poni_file: str,
    npt: int = 1000,
    unit: str = "2th_deg",
    mask: Optional[np.ndarray] = None,
    dark: Optional[np.ndarray] = None,
    flat: Optional[np.ndarray] = None,
    error_model: Optional[str] = None,
    radial_range: Optional[Tuple[float, float]] = None,
    azimuth_range: Optional[Tuple[float, float]] = None,
    percentile: Tuple[float, float] = (10, 90),
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Perform 1D azimuthal integration with percentile filtering using pyFAI.

    Pixels whose azimuthal intensity falls outside the given percentile
    range within each radial bin are rejected before averaging, making
    this robust against hot pixels and zingers without iterative clipping.

    Parameters
    ----------
    image : np.ndarray
        2D detector image as a numpy array.
    poni_file : str
        Path to the PONI (Point Of Normal Incidence) calibration file.
    npt : int, optional
        Number of radial bins in the output pattern (default: 1000).
    unit : str, optional
        Output radial unit:
            "2th_deg" – two-theta in degrees (default)
            "q_A^-1"  – scattering vector q in Å⁻¹
            "q_nm^-1" – scattering vector q in nm⁻¹
            "2th_rad" – two-theta in radians
            "r_mm"    – radius on detector in mm
    mask : np.ndarray, optional
        2D mask array (1 = masked/ignored, 0 = valid).
    dark : np.ndarray, optional
        Dark-current image to subtract before integration.
    flat : np.ndarray, optional
        Flat-field image for pixel-efficiency correction.
    error_model : str, optional
        Error model for uncertainty propagation.
        "poisson" for photon-counting detectors.
    radial_range : tuple of (float, float), optional
        (min, max) radial range in the chosen unit.
    azimuth_range : tuple of (float, float), optional
        (min, max) azimuthal range in degrees, e.g. (-180, 180).
    percentile : tuple of (float, float), optional
        (low, high) percentile bounds for pixel rejection within each
        radial bin (default: (10, 90)).

    Returns
    -------
    radial : np.ndarray
        Radial axis values in the requested unit, shape (npt,).
    intensity : np.ndarray
        Filtered integrated intensity at each radial position, shape (npt,).
    sigma : np.ndarray or None
        Per-bin uncertainty (only when error_model is set).

    Raises
    ------
    ValueError
        If the image is not 2-dimensional.

    Examples
    --------
    >>> image = np.random.poisson(1000, (2048, 2048)).astype(np.float32)
    >>> q, I, sigma = azimuthal_integration_1d_filter(image, "detector.poni", npt=500)
    >>> print(q.shape, I.shape)  # (500,) (500,)
    """
    if image.ndim != 2:
        raise ValueError(f"image must be 2D, got shape {image.shape}")

    ai = _get_integrator(poni_file)

    result = ai.medfilt1d_ng(
        image,
        npt=npt,
        unit=unit,
        mask=mask,
        dark=dark,
        flat=flat,
        error_model=error_model,
        radial_range=radial_range,
        azimuth_range=azimuth_range,
        percentile=percentile,
        method=("full", "csr", "cython"),
    )

    return result.radial, result.intensity, result.sigma


def azimuthal_integration_1d_sigma_clip(
    image: np.ndarray,
    poni_file: str,
    npt: int = 1000,
    unit: str = "2th_deg",
    mask: Optional[np.ndarray] = None,
    dark: Optional[np.ndarray] = None,
    flat: Optional[np.ndarray] = None,
    error_model: Optional[str] = "hybrid",
    radial_range: Optional[Tuple[float, float]] = None,
    azimuth_range: Optional[Tuple[float, float]] = None,
    thres: float = 3.0,
    max_iter: int = 5,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Perform 1D azimuthal integration with sigma-clipping using pyFAI.

    Iteratively removes pixels whose intensity deviates more than `thres`
    standard deviations from the bin mean. Useful for rejecting hot pixels,
    zingers, and single-crystal spots from a polycrystalline background.

    Parameters
    ----------
    image : np.ndarray
        2D detector image as a numpy array.
    poni_file : str
        Path to the PONI (Point Of Normal Incidence) calibration file.
    npt : int, optional
        Number of radial bins in the output pattern (default: 1000).
    unit : str, optional
        Output radial unit:
            "2th_deg" – two-theta in degrees (default)
            "q_A^-1"  – scattering vector q in Å⁻¹
            "q_nm^-1" – scattering vector q in nm⁻¹
            "2th_rad" – two-theta in radians
            "r_mm"    – radius on detector in mm
    mask : np.ndarray, optional
        2D mask array (1 = masked/ignored, 0 = valid).
    dark : np.ndarray, optional
        Dark-current image to subtract before integration.
    flat : np.ndarray, optional
        Flat-field image for pixel-efficiency correction.
    error_model : str, optional
        Error model for uncertainty propagation.
        "hybrid"  – combines Poisson and readout noise (default).
        "poisson" – pure photon-counting noise.
    radial_range : tuple of (float, float), optional
        (min, max) radial range in the chosen unit.
    azimuth_range : tuple of (float, float), optional
        (min, max) azimuthal range in degrees, e.g. (-180, 180).
    thres : float, optional
        Clipping threshold in standard deviations (default: 3.0).
    max_iter : int, optional
        Maximum number of sigma-clipping iterations (default: 5).

    Returns
    -------
    radial : np.ndarray
        Radial axis values in the requested unit, shape (npt,).
    intensity : np.ndarray
        Clipped integrated intensity at each radial position, shape (npt,).
    sigma : np.ndarray or None
        Per-bin uncertainty (only when error_model is set).

    Raises
    ------
    ValueError
        If the image is not 2-dimensional.

    Examples
    --------
    >>> image = np.random.poisson(1000, (2048, 2048)).astype(np.float32)
    >>> q, I, sigma = azimuthal_integration_1d_sigma_clip(image, "detector.poni", npt=500)
    >>> print(q.shape, I.shape)  # (500,) (500,)
    """
    if image.ndim != 2:
        raise ValueError(f"image must be 2D, got shape {image.shape}")

    ai = _get_integrator(poni_file)

    result = ai.sigma_clip(
        image,
        npt=npt,
        unit=unit,
        mask=mask,
        dark=dark,
        flat=flat,
        error_model=error_model,
        radial_range=radial_range,
        azimuth_range=azimuth_range,
        thres=thres,
        max_iter=max_iter,
        method=("no", "csr", "cython"),
    )

    return result.radial, result.intensity, result.sigma

def integrate_powder_parallel(
    master_file: Path,
    output_file: Path,
    poni_file: Path,
    mask_file: Path,
    rot: np.ndarray,
    n_points: int = 1000,
    n_workers: int = 16,
    unit: str = "2th_deg",
    remove_spots:bool = False, 
    percentile:tuple = (10, 90)
):
    """
    Perform parallel 1D azimuthal integration reading image stacks directly
    from each entry in the HDF5 master file, with normalization by fpico6.

    Parameters
    ----------
    master_file : Path
        Path to the master HDF5 file containing all scan entries.
    output_file : Path
        Path to the output HDF5 file.
    poni_file : Path
        Path to the PONI calibration file.
    mask_file : Path
        Path to the mask file (fabio-readable).
    rot : np.ndarray
        Array of rotation angles.
    n_points : int
        Number of radial bins in the integrated pattern.
    n_workers : int
        Number of parallel integration threads.
    unit : str
        Radial unit for integration (e.g. "2th_deg", "q_A^-1").
    """
    t0 = time.time()
    mask = fabio.open(mask_file).data

    # ── Read and validate entries from master file ────────────────────────────
    print("Reading entries from master file...")
    valid_entries, bad_entries, dty_values = [], [], []

    with h5py.File(master_file, "r") as hin:
        all_entries = list(hin.keys())
        for entry in tqdm(all_entries, desc="Validating entries"):
            try:
                _ = hin[f"{entry}/measurement/eiger"].shape
                _ = hin[f"{entry}/measurement/fpico6"].shape
                dty = float(hin[f"{entry}/instrument/positioners/dty"][()])
                valid_entries.append(entry)
                dty_values.append(dty)
            except KeyError as e:
                print(f"  ⚠  Entry {entry} missing expected dataset ({e}) — skipping")
                bad_entries.append(entry)

    print(f"\n✓  {len(valid_entries)}/{len(all_entries)} entries OK")
    if bad_entries:
        print(f"⚠  Skipping {len(bad_entries)} incomplete entries: {bad_entries}\n")

    # ── Initialise output file ────────────────────────────────────────────────
    with h5py.File(master_file, "r") as hin, h5py.File(output_file, "a") as hout:

        if "integrated/radial" not in hout:
            first_image = hin[f"{valid_entries[0]}/measurement/eiger"][0].astype(np.float32)
            tt, _, _ = azimuthal_integration_1d(
                image=first_image, poni_file=poni_file, npt=n_points, mask=mask, unit=unit
            )
            mascake = cake_integration(np.ones_like(first_image)*10, poni_file, npt_rad=n_points, mask=mask)[0]>0
            hout['integrated/cake_mask'] = mascake
            hout["integrated/radial"] = tt
            hout["integrated/radial"].attrs["unit"] = unit

        if "motors/dty" not in hout:
            hout["motors/dty"] = dty_values
        if "motors/rot" not in hout:
            hout["motors/rot"] = rot
        if bad_entries and "bad_entries" not in hout:
            hout["bad_entries"] = bad_entries

    # ── Helper: integrate and normalise one frame ─────────────────────────────
    def integrate_frame(args: tuple) -> tuple:
        jj, image, monitor = args
        if remove_spots:

            _, itt, _ = azimuthal_integration_1d_filter(image=image, poni_file=poni_file, npt=n_points, mask=mask, unit=unit, percentile=percentile)
                        
        else:
            _, itt, _ = azimuthal_integration_1d(
                image=image, poni_file=poni_file, npt=n_points, mask=mask, unit=unit
            )
        if monitor <= 0:
            print(f"  ⚠  Frame {jj}: fpico6 monitor value is {monitor:.4g}, skipping normalisation")
            return jj, itt
        return jj, itt / monitor

    # ── Main loop over master file entries ────────────────────────────────────
    for ii, entry in enumerate(valid_entries):

        scan_name  = f"scan_{ii:04d}"
        group_path = f"integrated/{scan_name}"

        with h5py.File(output_file, "r") as hout:
            if group_path in hout:
                print(f"Skipping {scan_name} (already done)")
                continue

        print(f"\n{'='*60}\nProcessing {scan_name} — entry {entry}  [{ii+1}/{len(valid_entries)}]\n{'='*60}")

        try:
            with h5py.File(master_file, "r") as hin:
                images  = hin[f"{entry}/measurement/eiger"][:].astype(np.float32)
                fpico6  = hin[f"{entry}/measurement/fpico6"][:].astype(np.float64)
                rot     = hin[f"{entry}/measurement/rot"][:]
        except OSError as e:
            print(f"  ✗ Failed to read entry {entry}: {e} — skipping")
            continue

        if rot[-1] < rot[0]:
            images = images[::-1]
            fpico6 = fpico6[::-1]
            rot = rot[::-1]

        # Sanity check lengths match
        if len(fpico6) != len(images):
            print(f"  ✗ Entry {entry}: fpico6 length {len(fpico6)} != images length {len(images)} — skipping")
            continue

        n_frames = len(images)
        sinogram  = np.empty((n_frames, n_points), dtype=np.float32)

        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(integrate_frame, (jj, images[jj], fpico6[jj])): jj
                for jj in range(n_frames)
            }
            for future in tqdm(as_completed(futures), total=n_frames, desc=scan_name):
                try:
                    jj, itt = future.result()
                    sinogram[jj] = itt
                except Exception as e:
                    print(f"  ✗ Frame {futures[future]} failed: {e}")
                    sinogram[futures[future]] = np.nan

        with h5py.File(output_file, "a") as hout:
            ds = hout.create_dataset(
                group_path,
                data=sinogram,
                compression="gzip",
                compression_opts=4,
                chunks=(1, n_points),
            )
            ds.attrs["entry"]          = entry
            ds.attrs["dty"]            = dty_values[ii]
            ds.attrs["source"]         = str(master_file)
            ds.attrs["fpico6_mean"]    = float(np.mean(fpico6))
            ds.attrs["fpico6_min"]     = float(np.min(fpico6))
            ds.attrs["fpico6_max"]     = float(np.max(fpico6))
            ds.attrs["normalised_by"]  = "fpico6"
            ds.attrs["valid"]          = True

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f} s  ({elapsed/len(valid_entries):.1f} s/scan).")

