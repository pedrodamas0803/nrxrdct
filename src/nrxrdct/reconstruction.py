import concurrent.futures
import os
import time
from pathlib import Path

import astra
import GSASIIscriptable as G2sc
import h5py
import hdf5plugin
import numpy as np
from tqdm.auto import tqdm

from .io import save_sinogram, save_xy_file
from .refinement import BaseRefinement
from .utils import calculate_padding_widths_2D

HAS_GPU = True if "nvidia" in astra.get_gpu_info().lower() else False
NTHREADS = os.cpu_count() - 2

def reconstruct_astra_gpu_3d(
    data: np.ndarray,
    dty_step: float = 1.0,
    angles_rad: np.ndarray = np.empty((1,)),
    algo: str = "SIRT3D_CUDA",
    num_iter: int = 40,
) -> np.ndarray:
    """
    3D GPU-accelerated reconstruction using the ASTRA Toolbox.

    Parameters
    ----------
    data : np.ndarray
        Sinogram stack of shape (num_detectors_x, num_angles, num_detectors_y).
        - num_detectors_x: horizontal detector size (columns)
        - num_angles:       number of projection angles
        - num_detectors_y:  vertical detector size / number of slices (rows)
    dty_step : float
        Detector pixel spacing.
    angles_rad : np.ndarray
        1D array of projection angles in radians.
    algo : str
        ASTRA 3D CUDA algorithm. One of:
        "SIRT3D_CUDA", "CGLS3D_CUDA".
    num_iter : int
        Number of iterations.

    Returns
    -------
    np.ndarray
        Reconstructed volume of shape (num_detectors_y, N, N).
    """
    # data is expected as (num_detectors_x, num_angles, num_detectors_y)
    # ASTRA 3D expects projections as (num_detectors_y, num_angles, num_detectors_x)
    if data.ndim != 3:
        raise ValueError(f"Expected 3D input array, got shape {data.shape}")

    # data = np.transpose(data, (2, 1, 0))  # -> (num_detectors_y, num_angles, num_detectors_x)
    num_slices, num_angles, num_det_x = data.shape

    if num_angles != len(angles_rad):
        raise ValueError(
            f"Angle axis mismatch: data has {num_angles} angles "
            f"but angles_rad has {len(angles_rad)} entries."
        )

    valid_algos = {"SIRT3D_CUDA", "CGLS3D_CUDA"}
    if algo not in valid_algos:
        raise ValueError(f"Unsupported algorithm '{algo}'. Choose from {valid_algos}.")

    proj_geom = astra.create_proj_geom("parallel3d", dty_step, dty_step, num_slices, num_det_x, angles_rad)
    vol_geom = astra.create_vol_geom(num_det_x, num_det_x, num_slices)

    proj_id = astra.data3d.create("-proj3d", proj_geom, data)
    recon_id = astra.data3d.create("-vol", vol_geom)

    cfg = astra.astra_dict(algo)
    cfg["ProjectionDataId"] = proj_id
    cfg["ReconstructionDataId"] = recon_id
    if algo in ["SIRT3D_CUDA", "CGLS3D_CUDA"]:
        cfg["option"] = {"MinConstraint": 0.0}

    algorithm_id = astra.algorithm.create(cfg)

    try:
        astra.algorithm.run(algorithm_id, num_iter)
        reconstruction = astra.data3d.get(recon_id)
    finally:
        astra.algorithm.delete(algorithm_id)
        astra.data3d.delete([proj_id, recon_id])

    return reconstruction


def reconstruct_astra_gpu(
    data: np.ndarray,
    dty_step: float = 1.0,
    angles_rad: np.array = np.empty((1,)),
    algo: str = "SART_CUDA",
    num_iter: int = 200,
):

    N = data.shape[0]
    data = data.T
    # Ensure correct sinogram shape:
    # ASTRA expects (num_angles, num_detectors)
    if data.shape[0] != len(angles_rad):
        raise ValueError("Sinogram must have shape (num_angles, num_detectors)")

    proj_geom = astra.create_proj_geom("parallel", dty_step, N, angles_rad)

    vol_geom = astra.create_vol_geom(N, N)

    sinogram_id = astra.data2d.create("-sino", proj_geom, data)
    recon_id = astra.data2d.create("-vol", vol_geom)

    cfg = astra.astra_dict(algo)
    cfg["ProjectionDataId"] = sinogram_id
    cfg["ReconstructionDataId"] = recon_id

    if algo in ["SIRT_CUDA", "SART_CUDA"]:
        cfg["option"] = {"MinConstraint": 0.0}

    algorithm_id = astra.algorithm.create(cfg)
    astra.algorithm.run(algorithm_id, num_iter)

    reconstruction = astra.data2d.get(recon_id)

    astra.algorithm.delete(algorithm_id)
    astra.data2d.delete([sinogram_id, recon_id])

    return reconstruction


def reconstruct_astra_cpu(
    data: np.ndarray,
    dty_step: float = 1.0,
    angles_rad: np.array = np.empty((1,)),
    algo: str = "FBP",
    num_iter: int = 200,
):

    N = data.shape[0]
    data = data.T
    # Ensure correct sinogram shape:
    # ASTRA expects (num_angles, num_detectors)
    if data.shape[0] != len(angles_rad):
        raise ValueError("Sinogram must have shape (num_angles, num_detectors)")

    num_detectors = data.shape[1]

    # Create geometries
    proj_geom = astra.create_proj_geom("parallel", dty_step, num_detectors, angles_rad)

    vol_geom = astra.create_vol_geom(num_detectors, num_detectors)

    # CPU projector (important!)
    projector_id = astra.create_projector(
        "linear", proj_geom, vol_geom  # CPU projector
    )

    # Create data objects
    sinogram_id = astra.data2d.create("-sino", proj_geom, data)

    recon_id = astra.data2d.create("-vol", vol_geom, data=0.0)

    # Configure algorithm
    cfg = astra.astra_dict(algo)
    cfg["ProjectorId"] = projector_id
    cfg["ProjectionDataId"] = sinogram_id
    cfg["ReconstructionDataId"] = recon_id

    # Optional positivity constraint
    if algo in ["SIRT", "SART", "FBP"]:
        cfg["option"] = {"MinConstraint": 0.0}

    algorithm_id = astra.algorithm.create(cfg)

    # Run reconstruction
    astra.algorithm.run(algorithm_id, num_iter)

    reconstruction = astra.data2d.get(recon_id)

    # Cleanup
    astra.algorithm.delete(algorithm_id)
    astra.data2d.delete([sinogram_id, recon_id])
    astra.projector.delete(projector_id)

    return reconstruction


def forward_project_gpu(
    volume, angles_rad, det_spacing: float = 1.0, algo: str = "FP_CUDA"
):
    # Create geometries
    N = volume.shape[1]
    proj_geom = astra.create_proj_geom("parallel", det_spacing, N, angles_rad)
    vol_geom = astra.create_vol_geom(N, N)

    # Generate phantom image
    phantom_id = astra.data2d.create("-vol", vol_geom, volume)

    # Calculate forward projection
    projection_id = astra.data2d.create("-sino", proj_geom)
    cfg = astra.astra_dict("FP_CUDA")
    cfg["ProjectionDataId"] = projection_id
    cfg["VolumeDataId"] = phantom_id
    algorithm_id = astra.algorithm.create(cfg)

    astra.algorithm.run(algorithm_id)

    projection = astra.data2d.get(projection_id)

    # Clean up
    astra.data2d.delete([projection_id, phantom_id])
    astra.algorithm.delete(algorithm_id)

    return projection


def reconstruct_slice(
    data: np.ndarray,
    dty_step: float = 1.0,
    angles_rad: np.array = np.empty((1,)),
    algo: str = "SART_CUDA",
    num_iter: int = 200,
):
    N = data.shape[0]
    if angles_rad.shape[0] < 10:
        angles_rad = np.linspace(0, np.pi, N)
    if HAS_GPU:
        # print("Reconstructing data using GPU.")
        slc = reconstruct_astra_gpu(data, dty_step, angles_rad, algo, num_iter)
    else:
        # print("Reconstructing data using CPU.")
        slc = reconstruct_astra_cpu(data, dty_step, angles_rad, algo, num_iter)

    return slc


def assemble_sinogram(integrated_file: Path, n_rot: int, n_tth_angles: int):

    with h5py.File(integrated_file, "r") as hin:
        keys = list(hin["integrated"].keys())
        valid_keys = [key for key in keys if "scan" in key]
        sino = np.zeros((len(valid_keys), n_rot, n_tth_angles), dtype=np.float32)
        for ii, scan in enumerate(valid_keys):
            im = hin[f"integrated/{scan}"][:]
            padding_width = calculate_padding_widths_2D(im.shape, (n_rot, n_tth_angles))
            im = np.pad(im, padding_width)
            sino[ii] = im
        sino = np.rollaxis(sino, 2, 0)

    return np.rollaxis(sino, 1, 2)


# def _write_xy_file_shared(args):
#     shm_name, shape, dtype, tth, ii, jj, filename = args
#     shm = shared_memory.SharedMemory(name=shm_name)
#     volume = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
#     save_xy_file(tth, volume[:, ii, jj], None, str(filename), verbose=False)
#     shm.close()
#     return f"Did {filename}."


class ReconstructedVolume:
    def __init__(
        self,
        volume: np.ndarray,
        tth_deg: np.array,
        sample_name: str,
        phases: list,
        processing_folder: Path = Path("volume"),
    ):
        self.volume = volume
        self.tth = tth_deg
        self.phases = phases
        self.name = sample_name
        self.shape = volume.shape
        self.folder = processing_folder
        self.folder_xy = self.folder / "xy_files"
        self.folder_models = self.folder / "gpx_files"
        os.makedirs(str(self.folder_xy), exist_ok=True)
        os.makedirs(str(self.folder_models), exist_ok=True)

    def write_xy_files(self):
        t0 = time.time()
        assert self.volume.shape[0] == self.tth.shape[0], "Wrong shapes"

        for ii in tqdm(range(self.volume.shape[1]), total=self.volume.shape[1]):
            for jj in range(self.volume.shape[2]):
                filename = self.folder / "xy_files" / f"{self.name}_{ii:04}_{jj:04}.xy"
                save_xy_file(
                    self.tth, self.volume[:, ii, jj], None, str(filename), verbose=False
                )
        t1 = time.time()
        print(60 * "=")
        print(f"Finished writing xy files to {self.folder} in {t1-t0:.2f} s.")
        print(60 * "=")

    def write_xy_files_parallel(self):
        t0 = time.time()

        def write_ii_jj(index):
            ii, jj = index
            filename = self.folder / "xy_files" / f"{self.name}_{ii:04}_{jj:04}.xy"
            save_xy_file(
                self.tth, self.volume[:, ii, jj], None, str(filename), verbose=False
            )
            return f"Did {filename}."

        indexes = (
            (ii, jj)
            for ii in range(self.volume.shape[1])
            for jj in range(self.volume.shape[2])
        )

        with concurrent.futures.ThreadPoolExecutor(NTHREADS) as pool:
            results = list(pool.map(write_ii_jj, indexes, chunksize=64))

        print("\n".join(results))
        print(f"Finished in {time.time() - t0:.2f} s")

    def refine_models(self, refining_function):

        t0 = time.time()

        for ii in range(self.volume.shape[1]):
            for jj in range(self.volume.shape[2]):
                _refine_ii_jj((ii, jj, refining_function, self.folder, self.name))

        print(f"Refined models in {time.time()-t0:.2f} s.")

    def refine_models_parallel(self, refining_function):

        t0 = time.time()

        indexes = (
            (ii, jj, refining_function, self.folder, self.name)
            for ii in range(self.volume.shape[1])
            for jj in range(self.volume.shape[2])
        )

        with concurrent.futures.ThreadPoolExecutor(NTHREADS) as pool:
            _ = list(pool.map(_refine_ii_jj, indexes, chunksize=64))

        print(f"Finished in {time.time() - t0:.2f} s")

    def get_Rwp_map(self):

        result = np.zeros_like(self.volume.sum(axis=0))
        t0 = time.time()

        for ii in range(self.volume.shape[1]):
            for jj in range(self.volume.shape[2]):
                rwp = _get_Rwp_ii_jj((ii, jj, self.folder, self.name))
                result[ii, jj] = rwp
        print(f"Fetched Rwp map in {time.time()-t0:.2f} s.")
        return result

    def get_cell_map(self):

        t0 = time.time()
        a_map = np.zeros_like(self.volume.sum(axis=0))
        b_map = np.zeros_like(self.volume.sum(axis=0))
        c_map = np.zeros_like(self.volume.sum(axis=0))

        for ii in range(self.volume.shape[1]):
            for jj in range(self.volume.shape[2]):
                a, b, c = _get_cell_params_ii_jj((ii, jj, self.folder, self.name))
                a_map[ii, jj] = a
                b_map[ii, jj] = b
                c_map[ii, jj] = c

        print(f"Fetched cell parameters map in {time.time()-t0:.2f} s.")
        return a_map, b_map, c_map

    def get_crystallite_size_map(self):

        t0 = time.time()
        size_map = np.zeros_like(self.volume.sum(axis=0))

        for ii in range(self.volume.shape[1]):
            for jj in range(self.volume.shape[2]):
                size = _get_crystallite_sizes((ii, jj, self.folder, self.name))
                size_map[ii, jj] = size

        print(f"Fetched crystallite size map in {time.time()-t0:.2f} s.")
        return size_map


def _get_Rwp_ii_jj(args):

    try:
        ii, jj, folder, name = args
        gpx_filename = folder / "gpx_files" / f"{name}_{ii:04}_{jj:04}.gpx"
        g, hists, phases = _load_data_from_gpx(gpx_filename)
        wR = hists[0].get_wR()
        return wR
    except:
        return np.nan


def _get_crystallite_sizes(args):
    try:
        ii, jj, folder, name = args
        gpx_filename = folder / "gpx_files" / f"{name}_{ii:04}_{jj:04}.gpx"
        g, hists, phases = _load_data_from_gpx(gpx_filename)
        hap = phases[0].data["Histograms"][hists[0].name]
        sz = hap["Size"]["Size"]
        print(sz)
        return sz
    except:
        return np.nan


def _get_cell_params_ii_jj(args):

    try:
        ii, jj, folder, name = args
        gpx_filename = folder / "gpx_files" / f"{name}_{ii:04}_{jj:04}.gpx"
        g, hists, phases = _load_data_from_gpx(gpx_filename)
        cell = phases[0].get_cell()
        return cell["length_a"], cell["length_b"], cell["length_c"]
    except:
        return np.nan, np.nan, np.nan


def _load_data_from_gpx(filename: str):
    g = G2sc.G2Project(filename)
    hists = g.histograms()
    phases = g.phases()
    return g, hists, phases


def _refine_ii_jj(args):
    try:
        ii, jj, func, folder, name = args
        xy_filename = folder / "xy_files" / f"{name}_{ii:04}_{jj:04}.xy"
        gpx_filename = folder / "gpx_files" / f"{name}_{ii:04}_{jj:04}.gpx"
        func(xy_filename, gpx_filename)
    except:
        pass
