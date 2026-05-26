import numpy as np
import h5py
from pathlib import Path
from scipy.ndimage import median_filter
from skimage.measure import block_reduce
from ..utils import calculate_padding_widths_2D
from tqdm.auto import tqdm # type: ignore


def assemble_sinogram(
    integrated_file: Path, n_rot: int, n_tth_angles: int, n_lines: int = 10
) -> np.ndarray:
    """
    Build a 3-D sinogram from an HDF5 file of integrated patterns.

    Scans stored under ``integrated/scan*`` keys are background-subtracted
    (using the mean of the first and last scans), zero-padded to
    ``(n_rot, n_tth_angles)``, and stacked.  The resulting array is rolled so
    that the 2θ axis comes first: shape ``(n_tth_angles, n_lines, n_rot)``.

    Args:
        integrated_file (Path): HDF5 file containing integrated patterns under
            the ``"integrated"`` group.
        n_rot (int): Number of rotation steps (sinogram angular dimension).
        n_tth_angles (int): Number of 2θ bins (spectral dimension).
        n_lines (int, optional): Expected number of translation lines; currently
            unused (default 10).

    Returns:
        np.ndarray: Sinogram array of shape ``(n_tth_angles, n_lines, n_rot)``
            as ``float32``.
    """
    with h5py.File(integrated_file, "r") as hin:
        keys = list(hin["integrated"].keys())
        valid_keys = [key for key in keys if "scan" in key]
        bkg1 = np.mean((hin[f"integrated/{valid_keys[0]}"][0:n_lines]), axis=0)
        bkg2 = np.mean((hin[f"integrated/{valid_keys[-1]}"][0:n_lines]), axis=0)
        bkg = (bkg1 + bkg2) / 2
        bkg /= bkg.max()
        sino = np.zeros((len(valid_keys), n_rot, n_tth_angles), dtype=np.float32)
        for ii, scan in enumerate(valid_keys):
            im = hin[f"integrated/{scan}"][:]
            for jj, line in enumerate(im):
                im[jj] /= line.max()
                im[jj] = line - bkg
            # bkg = gaussian_filter(im, (10, 100))
            # im -= bkg

            padding_width = calculate_padding_widths_2D(im.shape, (n_rot, n_tth_angles))
            im = np.pad(im, padding_width)
            sino[ii] = im
        sino = np.rollaxis(sino, 2, 0)

    return np.rollaxis(sino, 1, 2)

def get_fluo_roi(
    fn, n_angles=901, data_entry="mca_det0_all", monitor_entry="fpico6", filter_size=3
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        monitor_entry (str or None, optional): HDF5 dataset name for the beam-intensity
            monitor used for normalisation. Pass ``None`` to skip normalisation
            (default ``"fpico6"``).

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

            if monitor_entry is not None:
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
) -> tuple[np.ndarray, np.ndarray]:
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
        monitor_entry (str or None, optional): HDF5 dataset name for the beam-intensity
            monitor used for normalisation. Pass ``None`` to skip normalisation
            (default ``"fpico6"``).

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

            if monitor_entry is not None:
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
