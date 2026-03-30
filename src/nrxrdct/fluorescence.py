import h5py
import numpy as np
import xraylib
from scipy.ndimage import median_filter
from skimage.measure import block_reduce
from tqdm.auto import tqdm

DEFAULT_LINES = ["Ka1", "Ka2", "Kb1", "Kb2", "La1", "Lb1", "Lg1"]


def get_fluo_lines(
    element, energy_range, lines: list[str] = DEFAULT_LINES, verbose=False
):
    """ """

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


def get_fluo_roi(fn, n_angles=901, data_entry="mca_det0_all", filter_size=3):
    ypos = []
    rotz = []
    meas = []
    with h5py.File(fn, "r") as hin:
        for key in hin.keys():
            ypos.append((hin[f"{key}/instrument/positioners/dty"][()]))
            rot = hin[f"{key}/measurement/rot"][:]
            dat = hin[f"{key}/measurement/{data_entry}"][:].astype(np.float32)

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
