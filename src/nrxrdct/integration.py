import os
import numpy as np
import fabio
import pyFAI
from typing import Optional, Tuple


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

def integrate_multigeo(images, poni_files, n_bins=2000, unit = "2th_deg", polarization = 0.5, radial_range = None):
   
    print("=" * 60)
    print("STEP 1: pyFAI multi-geometry integration")
    print("=" * 60)

    print("Loading images...")
    imgs = []
    masks  = []
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
        ais=poni_files,
        unit=unit,
        radial_range=radial_range,
        empty=0.0
    )

    print("Integrating...")
    result = mg.integrate1d(
        lst_data=imgs,
        npt=n_bins,
        lst_mask=masks,
        polarization_factor=polarization,
        error_model="poisson",
    )

    tth_integrated       = result.radial
    intensity_integrated = result.intensity
    sigma_integrated     = result.sigma if result.sigma is not None \
                        else np.zeros_like(intensity_integrated)

    return tth_integrated, intensity_integrated, sigma_integrated