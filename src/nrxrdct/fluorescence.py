"""
X-ray fluorescence (XRF) data loading and sinogram assembly for XRD-CT.

Provides helpers to look up XRF emission line energies via xraylib, read
per-scan ROI or full-spectrum fluorescence data from HDF5 master files,
and assemble the resulting data into sinogram arrays ready for reconstruction.
"""

import h5py
import numpy as np
import xraylib
from scipy.ndimage import median_filter
from scipy.optimize import nnls
from skimage.measure import block_reduce
from tqdm.auto import tqdm

DEFAULT_LINES = ["Ka1", "Ka2", "Kb1", "Kb2", "La1", "Lb1", "Lg1"]


def get_fluo_lines(
    element, energy_range, lines: list[str] = DEFAULT_LINES, verbose=False
):
    """
    Return the XRF emission line energies for an element within a given energy range.

    Args:
        element (str): Chemical symbol (e.g. ``"Fe"``, ``"Cu"``).
        energy_range (tuple): ``(emin, emax)`` energy window in keV; lines outside
            this range are excluded.
        lines (list, optional): Emission line names to query (default: Ka1, Ka2,
            Kb1, Kb2, La1, Lb1, Lg1).
        verbose (bool, optional): If ``True``, print each line name and energy to
            stdout (default ``False``).

    Returns:
        dict: Mapping of line name (str) to energy in keV (float) for lines that
            fall within *energy_range*. Lines not available for the element are
            silently skipped.
    """

    emin, emax = energy_range
    Z = xraylib.SymbolToAtomicNumber(element)

    line_name = lines
    line_poss = [
        xraylib.KA1_LINE,
        xraylib.KA2_LINE,
        xraylib.KB1_LINE,
        xraylib.KB2_LINE,
        xraylib.LA1_LINE,
        xraylib.LB1_LINE,
        xraylib.LG1_LINE,
    ]
    lines = {}
    for name, line in zip(line_name, line_poss):
        try:
            en = xraylib.LineEnergy(Z, line)
            if en > emax:
                continue
            elif en < emin:
                continue
            else:
                lines[name] = en
        except Exception:
            print(f"Line {name} not available for {element}.")
            continue

    if verbose:
        for name, line in lines.items():
            print(f"{element} {name}: {line:.4f} keV")
    return lines


def get_fluo_roi(
    fn, n_angles=901, data_entry="mca_det0_all", monitor_entry="fpico6", filter_size=3
):
    """
    Load per-scan ROI fluorescence data from an HDF5 master file.

    For each scan entry in the file the rotation axis and the chosen MCA
    dataset are read, optionally median-filtered, zero-padded to *n_angles*,
    and sorted in ascending rotation order.

    Args:
        fn (str or Path): Path to the HDF5 master file.
        n_angles (int, optional): Target number of rotation steps; shorter scans
            are symmetrically zero-padded to this length (default 901).
        data_entry (str, optional): HDF5 dataset name under ``<entry>/measurement/``
            to read (default ``"mca_det0_all"``).
        filter_size (int, optional): Size of the median filter applied along the
            rotation axis; set to ``0`` to skip filtering (default 3).

    Returns:
        meas (np.ndarray): 2-D array of shape ``(n_scans, n_angles)`` with ROI
            intensities.
        ypos (np.ndarray): 1-D array of ``dty`` translation positions, one per scan.
        rotz (np.ndarray): 2-D array of shape ``(n_scans, n_angles)`` with rotation
            angles.
    """
    ypos = []
    rotz = []
    meas = []
    with h5py.File(fn, "r") as hin:
        for key in hin.keys():
            ypos.append((hin[f"{key}/instrument/positioners/dty"][()]))
            rot = hin[f"{key}/measurement/rot"][:]
            dat = hin[f"{key}/measurement/{data_entry}"][:].astype(np.float32)
            mon = hin[f"{key}/measurement/{monitor_entry}"][:].astype(np.float32)

            dat /= mon
            if filter_size != 0:
                dat = median_filter(dat, size=filter_size)

            rstart, rend = rot[0], rot[-1]
            # print(rt.shape, r1.shape, len(ypos))
            if len(rot) < n_angles:
                pad_width_start = (n_angles - len(rot) - 1) // 2
                pad_width_end = ((n_angles - len(rot) - 1) // 2) + (
                    (n_angles - len(rot) - 1) % 2
                )

                while len(rot) + pad_width_end + pad_width_start < n_angles:
                    pad_width_end += 1

                rot = np.pad(
                    rot, (pad_width_start, pad_width_end), constant_values=1e-6
                )
                dat = np.pad(
                    dat, (pad_width_start, pad_width_end), constant_values=1e-6
                )

            if rstart > rend:
                # print('reversed array')
                rotz.append(rot[::-1])
                meas.append(dat[::-1])
            else:
                # print('did not reverse array')
                rotz.append(rot)
                meas.append(dat)

    return np.array(meas), np.array(ypos), np.array(rotz)


def get_fluo_full_spectra(
    h5_file,
    n_angles=901,
    binning_factor=1,
    filter_size=7,
    dat_entry="mca_det0",
    binning_func=np.mean,
    monitor_entry="fpico6",
):
    """
    Assemble a full-spectrum XRF sinogram from an HDF5 master file.

    Each scan entry is monitor-normalised, median-filtered, optionally
    energy-binned, zero-padded to *n_angles*, and stacked into a 3-D
    sinogram array.

    Args:
        h5_file (str or Path): Path to the HDF5 master file.
        n_angles (int, optional): Target number of rotation steps; shorter scans
            are symmetrically zero-padded (default 901).
        binning_factor (int, optional): Number of adjacent energy channels to
            combine; ``1`` means no binning (default 1).
        filter_size (int, optional): Size of the median filter applied along the
            energy axis after normalisation (default 7).
        dat_entry (str, optional): HDF5 dataset name under ``<entry>/measurement/``
            containing the MCA spectra (default ``"mca_det0"``).
        binning_func (callable, optional): Aggregation function used when
            *binning_factor* > 1 (default ``np.mean``).
        monitor_entry (str, optional): HDF5 dataset name for the beam-intensity
            monitor used for normalisation (default ``"fpico6"``).

    Returns:
        sino (np.ndarray): 3-D sinogram of shape ``(n_energy_bins, n_scans,
            n_angles)`` as ``float32``.
        rot (np.ndarray): 1-D rotation angle array from the last scan entry
            processed.
    """
    with h5py.File(h5_file, "r") as hin:
        N = len(hin.keys())
        n_bins = hin[f"1.1/measurement/{dat_entry}"].shape[1]

    sino = np.zeros(
        (n_bins // binning_factor, N, n_angles), dtype=np.float32, order="F"
    )

    print(f"The signal is going to be binned by a factor {binning_factor}")

    with h5py.File(h5_file, "r") as hin:
        N = len(hin.keys())
        for ii, key in tqdm(enumerate(hin.keys()), total=N):

            rot = hin[f"{key}/measurement/rot"][:].astype(np.float32)
            dat = hin[f"{key}/measurement/{dat_entry}"][:].astype(np.float32)

            p = hin[f"{key}/measurement/{monitor_entry}"][:]
            pico = np.repeat(p, dat.shape[1]).reshape(dat.shape)
            dat = dat / (pico + 1e-8)
            dat = median_filter(dat, (1, filter_size))

            if binning_factor > 1:
                # print(f'The signal is going to be binned by a factor {binning_factor}')
                dat = block_reduce(
                    dat, block_size=(1, binning_factor), func=binning_func
                )

            rstart, rend = rot[0], rot[-1]

            if len(rot) < n_angles:
                pad_width_start = (n_angles - len(rot) - 1) // 2
                pad_width_end = ((n_angles - len(rot) - 1) // 2) + (
                    (n_angles - len(rot) - 1) % 2
                )

                while len(rot) + pad_width_end + pad_width_start < n_angles:
                    pad_width_end += 1

                rot = np.pad(
                    rot, (pad_width_start, pad_width_end), constant_values=1e-6
                )
                dat = np.pad(
                    dat,
                    ((pad_width_start, pad_width_end), (0, 0)),
                    constant_values=1e-6,
                )

            if rstart > rend:
                sino[:, ii, :] = dat[::-1, :].T
            else:
                sino[:, ii, :] = dat.T

    return sino, rot


# ---------------------------------------------------------------------------
# Spectrum fitting
# ---------------------------------------------------------------------------

_LINE_CONSTS = [
    xraylib.KA1_LINE,
    xraylib.KA2_LINE,
    xraylib.KB1_LINE,
    xraylib.KB2_LINE,
    xraylib.LA1_LINE,
    xraylib.LB1_LINE,
    xraylib.LG1_LINE,
]


def _gaussian(x, center, sigma):
    return np.exp(-0.5 * ((x - center) / sigma) ** 2)


def build_element_component(element, energy_axis, excitation_energy, fwhm_keV):
    """
    Build the expected spectral shape for one element on a given energy axis.

    Each emission line is modelled as a Gaussian weighted by its fluorescence
    cross-section (``CS_FluorLine_Kissel``), so relative amplitudes between
    lines of the same element are physically motivated.

    Args:
        element (str): Chemical symbol, e.g. ``"Fe"``.
        energy_axis (np.ndarray): 1-D array of energy values in keV (from MCA
            calibration).
        excitation_energy (float): Incident beam energy in keV.
        fwhm_keV (float): Detector energy resolution (FWHM) in keV — typically
            0.13–0.25 keV for a Si drift detector.

    Returns:
        component (np.ndarray): Normalised spectral template for the element
            (max = 1), shape ``(len(energy_axis),)``. Returns a zero array if no
            lines fall within the energy axis range.
    """
    sigma = fwhm_keV / (2 * np.sqrt(2 * np.log(2)))
    Z = xraylib.SymbolToAtomicNumber(element)
    emin, emax = energy_axis[0], energy_axis[-1]

    component = np.zeros_like(energy_axis, dtype=np.float64)
    for _, lconst in zip(DEFAULT_LINES, _LINE_CONSTS):
        try:
            en = xraylib.LineEnergy(Z, lconst)
            if not (emin <= en <= emax):
                continue
            cs = xraylib.CS_FluorLine_Kissel(Z, lconst, excitation_energy)
            component += cs * _gaussian(energy_axis, en, sigma)
        except Exception:
            continue

    norm = component.max()
    return component / norm if norm > 0 else component


def fit_fluo_spectrum(
    spectrum,
    energy_axis,
    elements,
    excitation_energy,
    fwhm_keV=0.18,
    background_order=2,
):
    """
    Fit a fluorescence spectrum as a linear combination of element templates.

    A polynomial background is included as additional free components.  The fit
    is solved with non-negative least squares (NNLS), which prevents unphysical
    negative elemental intensities.

    Args:
        spectrum (np.ndarray): Measured (normalised) MCA spectrum, shape
            ``(n_channels,)``.
        energy_axis (np.ndarray): Energy in keV for each MCA channel, shape
            ``(n_channels,)``.
        elements (list): Elements to include, e.g. ``["Fe", "Cu", "Zn"]``.
        excitation_energy (float): Incident beam energy in keV.
        fwhm_keV (float, optional): Detector FWHM in keV (default 0.18).
        background_order (int, optional): Polynomial degree for the background
            model (default 2).

    Returns:
        coefficients (dict): Mapping ``element -> fitted amplitude`` (arbitrary
            units proportional to element concentration × thickness).
        residual (np.ndarray): Measured minus fitted spectrum.
        fitted (np.ndarray): Reconstructed spectrum (element contributions +
            background).
        components (dict): Spectral template used for each element (before scaling).
    """
    E_norm = (energy_axis - energy_axis.mean()) / energy_axis.ptp()

    columns = {}
    for el in elements:
        comp = build_element_component(el, energy_axis, excitation_energy, fwhm_keV)
        if comp.max() > 0:
            columns[el] = comp

    for deg in range(background_order + 1):
        columns[f"_bg{deg}"] = E_norm**deg

    labels = list(columns.keys())
    A = np.column_stack([columns[k] for k in labels])

    coeffs, _ = nnls(A, spectrum)

    fitted = A @ coeffs
    residual = spectrum - fitted

    coefficients = {k: v for k, v in zip(labels, coeffs) if not k.startswith("_bg")}
    components = {k: columns[k] for k in elements if k in columns}

    return coefficients, residual, fitted, components


def build_fit_matrix(
    energy_axis, elements, excitation_energy, fwhm_keV=0.18, background_order=2
):
    """
    Build the design matrix for XRF spectrum fitting.

    Separating matrix construction from fitting allows the (identical) matrix
    to be built once and reused across all pixels.

    Args:
        energy_axis (np.ndarray): Energy in keV for each MCA channel, shape
            ``(n_channels,)``.
        elements (list): Elements to model, e.g. ``["Fe", "Cu", "Zn"]``.
        excitation_energy (float): Incident beam energy in keV.
        fwhm_keV (float, optional): Detector FWHM in keV (default 0.18).
        background_order (int, optional): Polynomial degree for the background
            model (default 2).

    Returns:
        A (np.ndarray): Design matrix of shape ``(n_channels, n_components)``.
            Columns are element templates followed by background polynomial terms.
        labels (list): Column labels. Element names come first; background terms
            are named ``"_bg0"``, ``"_bg1"``, etc.
    """
    E_norm = (energy_axis - energy_axis.mean()) / energy_axis.ptp()

    columns, labels = [], []
    for el in elements:
        comp = build_element_component(el, energy_axis, excitation_energy, fwhm_keV)
        if comp.max() > 0:
            columns.append(comp)
            labels.append(el)

    for deg in range(background_order + 1):
        columns.append(E_norm**deg)
        labels.append(f"_bg{deg}")

    return np.column_stack(columns), labels


def fit_fluo_volume(
    data,
    energy_axis,
    elements,
    excitation_energy,
    fwhm_keV=0.18,
    background_order=2,
    method="lstsq",
    n_jobs=-1,
):
    """
    Fit fluorescence spectra for every pixel in a 3-D dataset.

    Args:
        data (np.ndarray): Full-spectrum dataset of shape ``(n_energy, n_y, n_x)``;
            axis 0 is the energy dimension.
        energy_axis (np.ndarray): Energy in keV for each channel, shape
            ``(n_energy,)``.
        elements (list): Elements to fit, e.g. ``["Fe", "Cu", "Zn"]``.
        excitation_energy (float): Incident beam energy in keV.
        fwhm_keV (float, optional): Detector FWHM in keV (default 0.18).
        background_order (int, optional): Polynomial degree for the background
            model (default 2).
        method (str, optional): Fitting strategy. ``"lstsq"`` solves all pixels
            in a single batched ``np.linalg.lstsq`` call — very fast but
            non-negativity is only enforced by clipping. ``"nnls"`` calls
            ``scipy.optimize.nnls`` per pixel (parallelised with ``joblib``) —
            strictly non-negative but slower. Default ``"lstsq"``.
        n_jobs (int, optional): Number of parallel workers for ``method="nnls"``
            (``-1`` uses all CPUs). Ignored for ``method="lstsq"`` (default ``-1``).

    Returns:
        coeff_maps (dict): Mapping ``element -> np.ndarray`` of shape ``(n_y, n_x)``
            with the fitted amplitude for each spatial pixel.
    """
    n_energy, n_y, n_x = data.shape
    n_pixels = n_y * n_x

    A, labels = build_fit_matrix(
        energy_axis, elements, excitation_energy, fwhm_keV, background_order
    )
    element_labels = [l for l in labels if not l.startswith("_bg")]

    # Shape: (n_energy, n_pixels) — keeps memory contiguous for LAPACK
    spectra = data.reshape(n_energy, n_pixels)

    if method == "lstsq":
        # Single batched solve: A (n_energy, n_comp) \ spectra (n_energy, n_pixels)
        coeffs, _, _, _ = np.linalg.lstsq(A, spectra, rcond=None)
        # coeffs: (n_comp, n_pixels) -> (n_pixels, n_comp)
        coeffs = np.clip(coeffs.T, 0, None)

    elif method == "nnls":
        # spectra.T: (n_pixels, n_energy) for row-wise iteration
        spectra_t = np.ascontiguousarray(spectra.T)
        try:
            from joblib import Parallel, delayed

            results = Parallel(n_jobs=n_jobs)(
                delayed(nnls)(A, spectra_t[i]) for i in range(n_pixels)
            )
            coeffs = np.array([r[0] for r in results])
        except ImportError:
            coeffs = np.array(
                [nnls(A, spectra_t[i])[0] for i in tqdm(range(n_pixels))]
            )

    else:
        raise ValueError(f"Unknown method '{method}'. Choose 'lstsq' or 'nnls'.")

    coeff_maps = {
        label: coeffs[:, j].reshape(n_y, n_x)
        for j, label in enumerate(labels)
        if label in element_labels
    }
    return coeff_maps
