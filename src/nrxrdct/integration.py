import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Tuple

import fabio
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pyFAI
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator
from tqdm import tqdm


def cake_integration(
    image: np.ndarray,
    poni_file: str,
    npt_rad: int = 1000,
    npt_azim: int = 360,
    unit: str = "q_A^-1",
    mask: Optional[np.ndarray] = None,
    dark: Optional[np.ndarray] = None,
    flat: Optional[np.ndarray] = None,
    radial_range: Optional[Tuple[float, float]] = None,
    azimuth_range: Optional[Tuple[float, float]] = (-180, 180),
    plot: bool = True,
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

    if plot:
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

    return cake, radial, azimuthal


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
        Number of points in the output 1D pattern (default: 1000).
    unit : str, optional
        Output radial unit. Options:
            "q_A^-1"  – scattering vector q in Å⁻¹  (default)
            "q_nm^-1" – scattering vector q in nm⁻¹
            "2th_deg" – two-theta in degrees
            "2th_rad" – two-theta in radians
            "r_mm"    – radius on detector in mm
    mask : np.ndarray, optional
        2D boolean/integer mask array (1 = masked/ignored, 0 = valid).
    dark : np.ndarray, optional
        Dark-current image to subtract before integration.
    flat : np.ndarray, optional
        Flat-field image for pixel-efficiency correction.
    error_model : str, optional
        Error model for uncertainty propagation.
        Use "poisson" for photon-counting detectors.
    radial_range : tuple of (float, float), optional
        (min, max) radial range to integrate over, in the chosen unit.
    azimuth_range : tuple of (float, float), optional
        (min, max) azimuthal range in degrees, e.g. (-180, 180).

    Returns
    -------
    q : np.ndarray
        Radial axis values (in the requested unit).
    intensity : np.ndarray
        Integrated intensity at each radial position.
    sigma : np.ndarray or None
        Uncertainty on the intensity (only when error_model is set).

    Raises
    ------
    FileNotFoundError
        If the PONI file does not exist.
    ValueError
        If the image is not 2-dimensional.

    Examples
    --------
    >>> import numpy as np
    >>> image = np.random.poisson(1000, (2048, 2048)).astype(np.float32)
    >>> q, I, sigma = azimuthal_integration_1d(image, "detector.poni", npt=500)
    >>> print(q.shape, I.shape)   # (500,) (500,)
    """
    if image.ndim != 2:
        raise ValueError(f"image must be 2D, got shape {image.shape}")

    # Load the calibration geometry from the PONI file
    ai = pyFAI.azimuthalIntegrator.AzimuthalIntegrator()
    ai.load(poni_file)

    # Run the integration
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
        # Return a named-tuple result object
        method=("no", "histogram", "cython"),  # fast CPU method
    )

    q = result.radial
    intensity = result.intensity
    sigma = result.sigma  # None unless error_model was specified

    return q, intensity, sigma


def integrate_multigeo(
    images, poni_files, n_bins=2000, unit="2th_deg", polarization=0.5, radial_range=None
):

    print("=" * 60)
    print("STEP 1: pyFAI multi-geometry integration")
    print("=" * 60)

    print("Loading images...")
    imgs = []
    masks = []
    for img_path, poni_path in zip(images, poni_files):
        img = fabio.open(img_path).data.astype(np.float32)
        imgs.append(img)
        print(f"  Loaded: {os.path.basename(img_path)}  shape={img.shape}")

        # Use detector gap mask if available
        ai = pyFAI.load(poni_path)
        det_mask = ai.detector.mask
        if det_mask is not None:
            masks.append(det_mask.astype(bool))
            print(f"    Detector mask applied: {det_mask.sum()} pixels masked")
        else:
            masks.append(np.zeros(img.shape, dtype=bool))
            print(f"    No detector mask found")

    print("\nBuilding MultiGeometry...")
    mg = pyFAI.multi_geometry.MultiGeometry(
        ais=poni_files, unit=unit, radial_range=radial_range, empty=0.0
    )

    print("Integrating...")
    result = mg.integrate1d(
        lst_data=imgs,
        npt=n_bins,
        lst_mask=masks,
        polarization_factor=polarization,
        error_model="poisson",
    )

    tth_integrated = result.radial
    intensity_integrated = result.intensity
    sigma_integrated = (
        result.sigma
        if result.sigma is not None
        else np.zeros_like(intensity_integrated)
    )

    return tth_integrated, intensity_integrated, sigma_integrated


def integrate_powder_parallel(
    h5files: Path,
    master_file: Path,
    output_file: Path,
    poni_file: Path,
    mask_file: Path,
    rot: np.array,
    n_points: int = 1000,
    n_workers: int = 16,
    unit: str = "2th_deg",
):
    t0 = time.time()
    mask = fabio.open(mask_file).data

    # ── Validate HDF5 files upfront ───────────────────────────────────────────────
    def is_valid_h5(path: str) -> bool:
        """Return True only if the file is a readable, non-empty HDF5 file."""
        try:
            with h5py.File(path, "r") as f:
                return "entry_0000/measurement/data" in f
        except (OSError, KeyError):
            return False

    print("Validating HDF5 files...")
    valid_files, bad_files = [], []
    for f in tqdm(h5files):
        (valid_files if is_valid_h5(f) else bad_files).append(f)

    if bad_files:
        print(f"\n⚠  Skipping {len(bad_files)} corrupt/incomplete file(s):")
        for bf in bad_files:
            print(f"   {os.path.basename(bf)}")

    print(f"\n✓  {len(valid_files)}/{len(h5files)} files OK\n")

    # ── Read motors from master file ──────────────────────────────────────────────
    DTY = []
    with h5py.File(master_file, "r") as hin:
        for entry in sorted(hin.keys()):
            try:
                dty = float(hin[f"{entry}/instrument/positioners/dty"][()])
            except KeyError:
                dty = float("nan")
            DTY.append(dty)

    # ── Initialise output file ────────────────────────────────────────────────────
    with h5py.File(output_file, "a") as hout:
        if "integrated/radial" not in hout:
            with h5py.File(valid_files[0], "r") as htmp:
                first_image = htmp["entry_0000/measurement/data"][0].astype(np.float32)
            tt, _, _ = azimuthal_integration_1d(
                image=first_image,
                poni_file=poni_file,
                npt=n_points,
                mask=mask,
                unit=unit,
            )
            hout["integrated/radial"] = tt
            hout["integrated/radial"].attrs["unit"] = f"{unit}"

        if "motors/dty" not in hout:
            hout["motors/dty"] = DTY
        if "motors/rot" not in hout:
            hout["motors/rot"] = rot

        # Log bad files so you can investigate later
        if bad_files and "bad_files" not in hout:
            hout["bad_files"] = [os.path.basename(b) for b in bad_files]

    # ── Helper: integrate one frame ───────────────────────────────────────────────
    def integrate_frame(args):
        jj, image = args
        _, itt, _ = azimuthal_integration_1d(
            image=image, poni_file=poni_file, npt=n_points, mask=mask, unit=unit
        )
        return jj, itt

    # ── Main loop (valid files only) ──────────────────────────────────────────────
    for ii, f in enumerate(valid_files):

        scan_name = f"scan_{ii:04d}"
        group_path = f"integrated/{scan_name}"

        with h5py.File(output_file, "r") as hin:
            if group_path in hin:
                print(f"Skipping {scan_name} (already done)")
                continue

        print(
            f"\n{'='*60}\nProcessing {scan_name}  [{ii+1}/{len(valid_files)}]\n{'='*60}"
        )

        try:
            with h5py.File(f, "r") as hin:
                images = hin["entry_0000/measurement/data"][:].astype(np.float32)
        except OSError as e:
            print(f"  ✗ Failed to read {f}: {e} — skipping")
            continue

        n_frames = len(images)
        sinogram = np.empty((n_frames, n_points), dtype=np.float32)

        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(integrate_frame, (jj, images[jj])): jj
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
            ds.attrs["dty"] = DTY[ii] if ii < len(DTY) else float("nan")
            ds.attrs["source"] = os.path.basename(f)
            ds.attrs["valid"] = True

    print(f"\nDone in {(t0-time.time()):.2f} s.")
