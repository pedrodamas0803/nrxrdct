"""
Tomographic reconstruction and volume analysis for XRD-CT data.

Provides GPU- and CPU-backed ASTRA Toolbox reconstruction routines, a helper to
assemble sinograms from integrated HDF5 files, and the :class:`ReconstructedVolume`
container that manages per-voxel .xy file I/O and parallelised GSAS-II refinement.
"""

import concurrent.futures
import os
import time
from pathlib import Path
from typing import Any, Callable, Tuple

import astra
import h5py
import hdf5plugin
import numpy as np
from GSASII import GSASIIscriptable as G2sc
from scipy.ndimage import gaussian_filter
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

    Args:
        data (np.ndarray): Sinogram stack of shape (num_detectors_x, num_angles,
            num_detectors_y), where num_detectors_x is the horizontal detector size,
            num_angles is the number of projection angles, and num_detectors_y is the
            vertical detector size / number of slices.
        dty_step (float): Detector pixel spacing.
        angles_rad (np.ndarray): 1D array of projection angles in radians.
        algo (str): ASTRA 3D CUDA algorithm: "SIRT3D_CUDA" or "CGLS3D_CUDA".
        num_iter (int): Number of iterations.

    Returns:
        np.ndarray: Reconstructed volume of shape (num_detectors_y, N, N).
    """
    # data is expected as (num_detectors_x, num_angles, num_detectors_y)
    # ASTRA 3D expects projections as (num_detectors_y, num_angles, num_detectors_x)
    if data.ndim != 3:
        raise ValueError(f"Expected 3D input array, got shape {data.shape}")

    # data = np.transpose(data, (2, 1, 0))  # -> (num_detectors_y, num_angles, num_detectors_x)
    data = np.rollaxis(data, 2, 1)
    num_slices, num_angles, num_det_x = data.shape

    if num_angles != len(angles_rad):
        raise ValueError(
            f"Angle axis mismatch: data has {num_angles} angles "
            f"but angles_rad has {len(angles_rad)} entries."
        )

    valid_algos = {"SIRT3D_CUDA", "CGLS3D_CUDA"}
    if algo not in valid_algos:
        raise ValueError(f"Unsupported algorithm '{algo}'. Choose from {valid_algos}.")

    proj_geom = astra.create_proj_geom(
        "parallel3d", dty_step, dty_step, num_slices, num_det_x, angles_rad
    )
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
    angles_rad: np.ndarray = np.empty((1,)),
    algo: str = "SART_CUDA",
    num_iter: int = 200,
) -> np.ndarray:
    """
    Reconstruct a single 2-D slice using a GPU-accelerated ASTRA algorithm.

    Args:
        data (np.ndarray): 2-D sinogram of shape ``(num_detectors, num_angles)``.
        dty_step (float, optional): Detector pixel spacing (default 1.0).
        angles_rad (np.ndarray, optional): 1-D array of projection angles in radians.
        algo (str, optional): ASTRA 2D CUDA algorithm, e.g. ``"SART_CUDA"`` or
            ``"SIRT_CUDA"`` (default ``"SART_CUDA"``).
        num_iter (int, optional): Number of iterations (default 200).

    Returns:
        np.ndarray: Reconstructed 2-D slice of shape ``(N, N)`` where ``N`` equals
            the number of detectors.
    """
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
    angles_rad: np.ndarray = np.empty((1,)),
    algo: str = "FBP",
    num_iter: int = 200,
) -> np.ndarray:
    """
    Reconstruct a single 2-D slice using a CPU ASTRA algorithm.

    Args:
        data (np.ndarray): 2-D sinogram of shape ``(num_detectors, num_angles)``.
        dty_step (float, optional): Detector pixel spacing (default 1.0).
        angles_rad (np.ndarray, optional): 1-D array of projection angles in radians.
        algo (str, optional): ASTRA 2D CPU algorithm, e.g. ``"FBP"``, ``"SIRT"``, or
            ``"SART"`` (default ``"FBP"``).
        num_iter (int, optional): Number of iterations; for FBP only one pass is
            performed regardless (default 200).

    Returns:
        np.ndarray: Reconstructed 2-D slice of shape ``(num_detectors, num_detectors)``.
    """
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
    volume: np.ndarray,
    angles_rad: np.ndarray,
    det_spacing: float = 1.0,
    algo: str = "FP_CUDA",
) -> np.ndarray:
    """
    Compute the GPU forward projection (sinogram) of a 2-D volume.

    Args:
        volume (np.ndarray): 2-D image/slice to project, shape ``(N, N)``.
        angles_rad (np.ndarray): 1-D array of projection angles in radians.
        det_spacing (float, optional): Detector pixel spacing (default 1.0).
        algo (str, optional): ASTRA forward-projection algorithm (default ``"FP_CUDA"``).

    Returns:
        np.ndarray: Sinogram of shape ``(num_angles, N)``.
    """
    # Create geometries
    N = volume.shape[1]
    proj_geom = astra.create_proj_geom("parallel", det_spacing, N, angles_rad)
    vol_geom = astra.create_vol_geom(N, N)

    # Generate phantom image
    phantom_id = astra.data2d.create("-vol", vol_geom, volume)

    # Calculate forward projection
    projection_id = astra.data2d.create("-sino", proj_geom)
    cfg = astra.astra_dict(algo)
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
    angles_rad: np.ndarray = np.empty((1,)),
    algo: str = "SART_CUDA",
    num_iter: int = 200,
) -> np.ndarray:
    """
    Reconstruct a single 2-D slice, dispatching to GPU or CPU automatically.

    If an NVIDIA GPU is detected at import time (``HAS_GPU`` is ``True``),
    :func:`reconstruct_astra_gpu` is called; otherwise falls back to
    :func:`reconstruct_astra_cpu`.  If fewer than 10 angles are provided a
    full 180° linspace is generated automatically.

    Args:
        data (np.ndarray): 2-D sinogram of shape ``(num_detectors, num_angles)``.
        dty_step (float, optional): Detector pixel spacing (default 1.0).
        angles_rad (np.ndarray, optional): 1-D array of projection angles in radians.
        algo (str, optional): ASTRA algorithm name passed to the backend
            (default ``"SART_CUDA"``).
        num_iter (int, optional): Number of iterations (default 200).

    Returns:
        np.ndarray: Reconstructed 2-D slice.
    """
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
        bkg1 = np.mean((hin[f"integrated/{valid_keys[0]}"][0:10]), axis=0)
        bkg2 = np.mean((hin[f"integrated/{valid_keys[-1]}"][0:10]), axis=0)
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


# def _write_xy_file_shared(args):
#     shm_name, shape, dtype, tth, ii, jj, filename = args
#     shm = shared_memory.SharedMemory(name=shm_name)
#     volume = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
#     save_xy_file(tth, volume[:, ii, jj], None, str(filename), verbose=False)
#     shm.close()
#     return f"Did {filename}."


class ReconstructedVolume:
    """
    Container for a reconstructed XRD-CT volume with per-voxel analysis helpers.

    Manages the reconstructed 4-D array (``tth × x × y``), the associated 2θ
    axis, phase information, and the output folder layout.  Provides sequential
    and parallelised methods to write per-voxel .xy files and run GSAS-II
    refinements, as well as map extractors for Rwp, unit-cell parameters, and
    crystallite sizes.
    """

    def __init__(
        self,
        volume: np.ndarray,
        tth_deg: np.ndarray,
        sample_name: str,
        phases: list,
        processing_folder: Path = Path("volume"),
        mask: np.ndarray | None = None,
    ):
        """
        Args:
            volume (np.ndarray): Reconstructed volume of shape ``(n_tth, nx, ny)``.
            tth_deg (np.ndarray): 1-D array of 2θ values in degrees, length ``n_tth``.
            sample_name (str): Base name used for output file naming.
            phases (list): List of phase objects or identifiers (passed through; not
                used internally).
            processing_folder (Path, optional): Root output directory; ``xy_files/``
                and ``gpx_files/`` sub-folders are created automatically
                (default ``"volume"``).
            mask (np.ndarray or None, optional): Boolean or integer array of shape
                ``(nx, ny)``.  Truthy pixels are processed; falsy pixels are skipped
                (no .xy file written, no refinement run) and appear as **zero** in
                all output maps.  ``None`` processes every voxel (default).
        """
        self.volume = volume
        self.tth = tth_deg
        self.phases = phases
        self.name = sample_name
        self.shape = volume.shape
        self.folder = processing_folder
        self.mask = mask  # shape (nx, ny) or None; truthy = process
        self.folder_xy = self.folder / "xy_files"
        self.folder_models = self.folder / "gpx_files"
        os.makedirs(str(self.folder_xy), exist_ok=True)
        os.makedirs(str(self.folder_models), exist_ok=True)

    @property
    def mask(self) -> np.ndarray | None:
        """Boolean/integer mask of shape ``(nx, ny)``; truthy = process, falsy = skip."""
        return self._mask

    @mask.setter
    def mask(self, value: np.ndarray | None) -> None:
        if value is not None:
            value = np.asarray(value)
            if value.shape != self.volume.shape[1:]:
                raise ValueError(
                    f"mask shape {value.shape} does not match volume spatial shape {self.volume.shape[1:]}"
                )
        self._mask = value

    @property
    def _active_indices(self) -> list:
        """List of ``(ii, jj)`` pairs that should be processed (mask truthy or no mask)."""
        nx, ny = self.volume.shape[1], self.volume.shape[2]
        if self.mask is None:
            return [(ii, jj) for ii in range(nx) for jj in range(ny)]
        return [(ii, jj) for ii in range(nx) for jj in range(ny) if self.mask[ii, jj]]

    def write_xy_files(self) -> None:
        """Write one .xy file per active (unmasked) voxel sequentially, with a tqdm progress bar."""
        t0 = time.time()
        assert self.volume.shape[0] == self.tth.shape[0], "Wrong shapes"

        active = self._active_indices
        for ii, jj in tqdm(active, total=len(active)):
            filename = self.folder / "xy_files" / f"{self.name}_{ii:04}_{jj:04}.xy"
            save_xy_file(
                self.tth, self.volume[:, ii, jj], None, str(filename), verbose=False
            )
        t1 = time.time()
        print(60 * "=")
        print(
            f"Finished writing {len(active)} xy files to {self.folder} in {t1-t0:.2f} s."
        )
        print(60 * "=")

    def write_xy_files_parallel(self) -> None:
        """Write one .xy file per active (unmasked) voxel using a thread pool for faster I/O."""
        t0 = time.time()

        def write_ii_jj(index: Tuple[int, int]) -> None:
            ii, jj = index
            filename = self.folder / "xy_files" / f"{self.name}_{ii:04}_{jj:04}.xy"
            save_xy_file(
                self.tth, self.volume[:, ii, jj], None, str(filename), verbose=False
            )

        indexes = self._active_indices

        with concurrent.futures.ThreadPoolExecutor(NTHREADS) as pool:
            for _ in tqdm(
                pool.map(write_ii_jj, indexes),
                total=len(indexes),
                desc="Writing xy",
            ):
                pass

        print(f"Finished writing {len(indexes)} xy files in {time.time() - t0:.2f} s")

    def refine_models(self, refining_function: Callable[[Path, Path], Any]) -> None:
        """
        Run a GSAS-II refinement function on every active voxel sequentially.

        Voxels whose .xy file does not exist (masked out or never written) are
        silently skipped.

        Args:
            refining_function (callable): Function with signature
                ``f(xy_file, gpx_file)`` that performs the refinement and saves
                results to *gpx_file*.
        """
        t0 = time.time()
        active = self._active_indices

        for ii, jj in active:
            _refine_ii_jj((ii, jj, refining_function, self.folder, self.name))

        print(f"Refined models in {time.time()-t0:.2f} s.")

    def refine_models_parallel(
        self, refining_function: Callable[[Path, Path], Any]
    ) -> None:
        """
        Run a GSAS-II refinement function on every voxel using a process pool.

        GSAS-II refinements are CPU-bound; ``ProcessPoolExecutor`` is used so that
        all available cores are utilised without GIL contention.

        Args:
            refining_function (callable): Module-level function with signature
                ``f(xy_file, gpx_file)`` that performs the refinement and saves
                results to *gpx_file*.  Must be picklable (i.e. defined at
                module level, not a lambda or closure).
        """
        t0 = time.time()

        args = [
            (ii, jj, refining_function, self.folder, self.name)
            for ii, jj in self._active_indices
        ]

        with concurrent.futures.ProcessPoolExecutor(NTHREADS) as pool:
            for _ in tqdm(
                pool.map(_refine_ii_jj, args, chunksize=16),
                total=len(args),
                desc="Refining",
            ):
                pass

        print(f"Finished in {time.time() - t0:.2f} s")

    def get_Rwp_map(self) -> np.ndarray:
        """
        Extract the weighted R-factor (Rwp) from each voxel's .gpx file.

        Returns:
            np.ndarray: 2-D map of shape ``(nx, ny)`` with Rwp values; voxels
                whose .gpx file is missing or failed return ``nan``.
        """
        t0 = time.time()
        nx, ny = self.volume.shape[1], self.volume.shape[2]
        active = self._active_indices
        args = [(ii, jj, self.folder, self.name) for ii, jj in active]

        with concurrent.futures.ProcessPoolExecutor(NTHREADS) as pool:
            values = list(
                tqdm(
                    pool.map(_get_Rwp_ii_jj, args, chunksize=64),
                    total=len(args),
                    desc="Rwp map",
                )
            )

        result = np.zeros((nx, ny), dtype=np.float32)
        for (ii, jj), val in zip(active, values):
            result[ii, jj] = val
        print(f"Fetched Rwp map in {time.time()-t0:.2f} s.")
        return result

    def get_cell_map(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract unit-cell lengths a, b, c from each voxel's .gpx file.

        Returns:
            tuple: Three 2-D maps ``(a_map, b_map, c_map)`` of shape ``(nx, ny)``;
                voxels whose .gpx file is missing or failed return ``nan``.
        """
        t0 = time.time()
        nx, ny = self.volume.shape[1], self.volume.shape[2]
        active = self._active_indices
        args = [(ii, jj, self.folder, self.name) for ii, jj in active]

        with concurrent.futures.ProcessPoolExecutor(NTHREADS) as pool:
            values = list(
                tqdm(
                    pool.map(_get_cell_params_ii_jj, args, chunksize=64),
                    total=len(args),
                    desc="Cell map",
                )
            )

        a_map = np.zeros((nx, ny), dtype=np.float32)
        b_map = np.zeros((nx, ny), dtype=np.float32)
        c_map = np.zeros((nx, ny), dtype=np.float32)
        for (ii, jj), (a, b, c) in zip(active, values):
            a_map[ii, jj] = a
            b_map[ii, jj] = b
            c_map[ii, jj] = c
        print(f"Fetched cell parameters map in {time.time()-t0:.2f} s.")
        return a_map, b_map, c_map

    def get_crystallite_size_map(self) -> np.ndarray:
        """
        Extract the refined isotropic crystallite size from each voxel's .gpx file.

        Returns:
            np.ndarray: 2-D map of shape ``(nx, ny)`` with crystallite sizes; voxels
                whose .gpx file is missing or failed return ``nan``.
        """
        t0 = time.time()
        nx, ny = self.volume.shape[1], self.volume.shape[2]
        active = self._active_indices
        args = [(ii, jj, self.folder, self.name) for ii, jj in active]

        with concurrent.futures.ProcessPoolExecutor(NTHREADS) as pool:
            values = list(
                tqdm(
                    pool.map(_get_crystallite_sizes, args, chunksize=64),
                    total=len(args),
                    desc="Size map",
                )
            )

        size_map = np.zeros((nx, ny), dtype=np.float32)
        for (ii, jj), val in zip(active, values):
            size_map[ii, jj] = val
        print(f"Fetched crystallite size map in {time.time()-t0:.2f} s.")
        return size_map

    def get_all_maps(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract Rwp, unit-cell lengths (a, b, c), and crystallite size from every
        voxel's .gpx file in a **single** parallel pass — 3× faster than calling
        the individual methods separately.

        Returns:
            tuple: ``(rwp_map, a_map, b_map, c_map, size_map)`` — five 2-D arrays
                of shape ``(nx, ny)``; failed voxels return ``nan``.
        """
        t0 = time.time()
        nx, ny = self.volume.shape[1], self.volume.shape[2]
        active = self._active_indices
        args = [(ii, jj, self.folder, self.name) for ii, jj in active]

        with concurrent.futures.ProcessPoolExecutor(NTHREADS) as pool:
            values = list(
                tqdm(
                    pool.map(_get_all_maps_ii_jj, args, chunksize=64),
                    total=len(args),
                    desc="All maps",
                )
            )

        rwp_map = np.zeros((nx, ny), dtype=np.float32)
        a_map = np.zeros((nx, ny), dtype=np.float32)
        b_map = np.zeros((nx, ny), dtype=np.float32)
        c_map = np.zeros((nx, ny), dtype=np.float32)
        size_map = np.zeros((nx, ny), dtype=np.float32)
        for (ii, jj), (rwp, a, b, c, sz) in zip(active, values):
            rwp_map[ii, jj] = rwp
            a_map[ii, jj] = a
            b_map[ii, jj] = b
            c_map[ii, jj] = c
            size_map[ii, jj] = sz
        print(f"Fetched all maps in {time.time()-t0:.2f} s.")
        return rwp_map, a_map, b_map, c_map, size_map

    def write_slurm_scripts(
        self,
        volume_hdf5: Path,
        refining_module: str,
        refining_function: str,
        refining_module_dir: str = "",
        n_array_jobs: int = 500,
        conda_env: str = "nrxrdct",
        conda_base: str = "",
        python_executable: str = "",
        mem: str = "4G",
        time_limit: str = "02:00:00",
        partition: str = "all",
    ) -> Tuple[Path, Path]:
        """
        Generate a SLURM array-job script and a matching Python worker for
        cluster-side refinement of all voxels.

        The volume and 2θ axis must already be saved to *volume_hdf5* under
        keys ``"volume"`` and ``"tth"`` respectively::

            import h5py
            with h5py.File("volume.h5", "a") as f:
                f["volume"] = vol.volume
                f["tth"] = vol.tth

        Each SLURM array element writes the xy files for its chunk (if not yet
        present) and then runs the refinements, so ``write_xy_files`` need not be
        called in advance.  The mask (if set) is read from the HDF5 file; store it
        before calling this method::

            with h5py.File("volume.h5", "a") as f:
                f["volume"] = vol.volume
                f["tth"]    = vol.tth
                if vol.mask is not None:
                    f["mask"] = vol.mask

        Args:
            volume_hdf5 (Path): HDF5 file with keys ``"volume"`` and ``"tth"``
                (and optionally ``"mask"``).
            refining_module (str): Importable Python module containing
                *refining_function* (e.g. ``"refine_volume"``).
            refining_function (str): Name of a module-level function with
                signature ``f(xy_file: Path, gpx_file: Path)``.
            refining_module_dir (str, optional): Absolute path to the directory
                containing *refining_module*, e.g. ``"/home/user/scripts"``.
                Inserted into ``sys.path`` at the top of the worker so the
                module can be found.  Leave empty if the module is already
                installed or on the default path (default ``""``).
            n_array_jobs (int, optional): Number of SLURM array elements
                (default 500; one element covers ~500 voxels for 500×500).
            conda_env (str, optional): Conda environment to activate
                (default ``"nrxrdct"``).
            conda_base (str, optional): Path to the conda installation root,
                e.g. ``"/cvmfs/hpc.esrf.fr/software/packages/linux/x86_64/jupyter-slurm/2025.04.5"``.
                When provided, ``<conda_base>/etc/profile.d/conda.sh`` is sourced
                before ``conda activate`` so the environment works in non-interactive
                SLURM batch shells.  Leave empty to skip (default ``""``).
            python_executable (str, optional): Absolute path to the Python binary
                inside your conda environment, e.g.
                ``"/path/to/envs/nrxrdct/bin/python"``.  When set, the conda
                activation block is skipped entirely and this binary is used
                directly — the most robust option on clusters where
                ``conda activate`` is unreliable.  Find it with ``which python``
                while your environment is active.  Leave empty to use conda
                activation instead (default ``""``).
            mem (str, optional): Memory per job (default ``"4G"``).
            time_limit (str, optional): Wall-time limit (default ``"02:00:00"``).
            partition (str, optional): SLURM partition (default ``"all"``).

        Returns:
            tuple: ``(worker_path, submit_path)`` — absolute paths to the
                generated Python worker and the ``sbatch`` submission script.
                Submit with ``sbatch <submit_path>``.
        """
        nx, ny = self.volume.shape[1], self.volume.shape[2]
        worker_path = self.folder / "worker_refine.py"
        submit_path = self.folder / "submit_refine.sh"
        folder_abs = str(self.folder.resolve())
        volume_abs = str(Path(volume_hdf5).resolve())

        # Build the worker script via template substitution so that inner
        # Python f-strings (run at worker time) are not evaluated here.
        worker_template = (
            "#!/usr/bin/env python\n"
            '"""SLURM array worker — refine a chunk of voxels.\n\n'
            "Usage::\n\n"
            "    python worker_refine.py --job-id JOB_ID --n-jobs N_JOBS\n\n"
            "*job_id* is SLURM_ARRAY_TASK_ID (0-indexed).\n"
            '"""\n\n'
            "import argparse\n"
            "import importlib\n"
            "import sys\n"
            "from pathlib import Path\n\n"
            "import h5py\n\n"
            + (f'sys.path.insert(0, {repr(refining_module_dir)})\n\n' if refining_module_dir else "")
            + "from nrxrdct.io import save_xy_file\n\n\n"
            "def main() -> None:\n"
            "    parser = argparse.ArgumentParser()\n"
            '    parser.add_argument("--job-id", type=int, required=True)\n'
            '    parser.add_argument("--n-jobs", type=int, required=True)\n'
            "    args = parser.parse_args()\n\n"
            "    volume_hdf5 = Path(VOLUME_HDF5)\n"
            '    with h5py.File(volume_hdf5, "r") as h:\n'
            '        volume = h["volume"][:]\n'
            '        tth = h["tth"][:]\n'
            '        mask = h["mask"][:] if "mask" in h else None\n\n'
            "    folder = Path(FOLDER_ABS)\n"
            "    name = SAMPLE_NAME\n"
            "    nx, ny = NX, NY\n"
            "    all_indexes = [\n"
            "        (ii, jj)\n"
            "        for ii in range(nx)\n"
            "        for jj in range(ny)\n"
            "        if mask is None or mask[ii, jj]\n"
            "    ]\n"
            "    chunk_size = -(-len(all_indexes) // args.n_jobs)\n"
            "    start = args.job_id * chunk_size\n"
            "    chunk = all_indexes[start : start + chunk_size]\n\n"
            "    mod = importlib.import_module(REFINING_MODULE)\n"
            "    refine_fn = getattr(mod, REFINING_FUNCTION)\n\n"
            "    for ii, jj in chunk:\n"
            '        xy_file = folder / "xy_files" / f"{name}_{ii:04}_{jj:04}.xy"\n'
            '        gpx_file = folder / "gpx_files" / f"{name}_{ii:04}_{jj:04}.gpx"\n'
            "        if not xy_file.exists():\n"
            "            save_xy_file(tth, volume[:, ii, jj], None, str(xy_file), verbose=False)\n"
            "        try:\n"
            "            refine_fn(xy_file, gpx_file)\n"
            "        except Exception as exc:\n"
            '            print(f"Voxel ({ii}, {jj}) failed: {exc}")\n\n\n'
            'if __name__ == "__main__":\n'
            "    main()\n"
        )
        worker_code = (
            worker_template.replace("VOLUME_HDF5", repr(volume_abs))
            .replace("FOLDER_ABS", repr(folder_abs))
            .replace("SAMPLE_NAME", repr(self.name))
            .replace("NX, NY", f"{nx}, {ny}")
            .replace("REFINING_MODULE", repr(refining_module))
            .replace("REFINING_FUNCTION", repr(refining_function))
        )

        submit_script = (
            "#!/bin/bash\n"
            f"#SBATCH --job-name={self.name}_refine\n"
            f"#SBATCH --array=0-{n_array_jobs - 1}\n"
            f"#SBATCH --mem={mem}\n"
            f"#SBATCH --time={time_limit}\n"
            f"#SBATCH --partition={partition}\n"
            f"#SBATCH --output={folder_abs}/slurm_logs/slurm_%A_%a.out\n\n"
            + (
                f"{python_executable} {str(worker_path.resolve())} \\\n"
                if python_executable
                else (
                    "# Conda environment activation\n"
                    + (
                        f"source {conda_base}/etc/profile.d/conda.sh\n"
                        if conda_base
                        else ""
                    )
                    + f"conda activate {conda_env}\n\n"
                    f"python {str(worker_path.resolve())} \\\n"
                )
            )
            + "    --job-id $SLURM_ARRAY_TASK_ID \\\n"
            + f"    --n-jobs {n_array_jobs}\n"
        )

        worker_path.write_text(worker_code)
        submit_path.write_text(submit_script)
        print(f"Worker script : {worker_path}")
        print(f"Submit script : {submit_path}")
        print(f"Submit with   : sbatch {submit_path}")
        return worker_path, submit_path


def _get_Rwp_ii_jj(args: tuple) -> float:
    """Return Rwp for voxel ``(ii, jj)``.

    Returns ``0.0`` if the .xy file does not exist (masked-out pixel),
    ``nan`` if the .xy file exists but the parameter could not be extracted.
    """
    ii, jj, folder, name = args
    xy_filename = folder / "xy_files" / f"{name}_{ii:04}_{jj:04}.xy"
    if not xy_filename.exists():
        return 0.0
    try:
        gpx_filename = folder / "gpx_files" / f"{name}_{ii:04}_{jj:04}.gpx"
        g, hists, phases = _load_data_from_gpx(gpx_filename)
        return hists[0].get_wR()
    except Exception:
        return np.nan


def _get_crystallite_sizes(args: tuple) -> float:
    """Return crystallite size for voxel ``(ii, jj)``.

    Returns ``0.0`` if the .xy file does not exist (masked-out pixel),
    ``nan`` if the .xy file exists but the parameter could not be extracted.
    """
    ii, jj, folder, name = args
    xy_filename = folder / "xy_files" / f"{name}_{ii:04}_{jj:04}.xy"
    if not xy_filename.exists():
        return 0.0
    try:
        gpx_filename = folder / "gpx_files" / f"{name}_{ii:04}_{jj:04}.gpx"
        g, hists, phases = _load_data_from_gpx(gpx_filename)
        hap = phases[0].data["Histograms"][hists[0].name]
        return hap["Size"]["Size"]
    except Exception:
        return np.nan


def _get_all_maps_ii_jj(args: tuple) -> Tuple[float, float, float, float, float]:
    """Return ``(rwp, a, b, c, size)`` for voxel ``(ii, jj)`` in one .gpx open.

    Returns all-zeros if the .xy file does not exist (masked-out pixel),
    all-nan if the .xy file exists but parameters could not be extracted.
    """
    ii, jj, folder, name = args
    xy_filename = folder / "xy_files" / f"{name}_{ii:04}_{jj:04}.xy"
    if not xy_filename.exists():
        return 0.0, 0.0, 0.0, 0.0, 0.0
    try:
        gpx_filename = folder / "gpx_files" / f"{name}_{ii:04}_{jj:04}.gpx"
        g, hists, phases = _load_data_from_gpx(gpx_filename)
        wR = hists[0].get_wR()
        cell = phases[0].get_cell()
        hap = phases[0].data["Histograms"][hists[0].name]
        sz = hap["Size"]["Size"]
        return wR, cell["length_a"], cell["length_b"], cell["length_c"], sz
    except Exception:
        return np.nan, np.nan, np.nan, np.nan, np.nan


def _get_cell_params_ii_jj(args: tuple) -> Tuple[float, float, float]:
    """Return ``(a, b, c)`` unit-cell lengths for voxel ``(ii, jj)``.

    Returns ``(0., 0., 0.)`` if the .xy file does not exist (masked-out pixel),
    ``(nan, nan, nan)`` if the .xy file exists but parameters could not be extracted.
    """
    ii, jj, folder, name = args
    xy_filename = folder / "xy_files" / f"{name}_{ii:04}_{jj:04}.xy"
    if not xy_filename.exists():
        return 0.0, 0.0, 0.0
    try:
        gpx_filename = folder / "gpx_files" / f"{name}_{ii:04}_{jj:04}.gpx"
        g, hists, phases = _load_data_from_gpx(gpx_filename)
        cell = phases[0].get_cell()
        return cell["length_a"], cell["length_b"], cell["length_c"]
    except Exception:
        return np.nan, np.nan, np.nan


def _load_data_from_gpx(filename: Path) -> Tuple[Any, list, list]:
    """Open a GSAS-II project file and return ``(project, histograms, phases)``."""
    g = G2sc.G2Project(filename)
    hists = g.histograms()
    phases = g.phases()
    return g, hists, phases


def _refine_ii_jj(args: tuple) -> None:
    """Run the refinement for one voxel; silently skips if the .xy file is absent."""
    try:
        ii, jj, func, folder, name = args
        xy_filename = folder / "xy_files" / f"{name}_{ii:04}_{jj:04}.xy"
        if not xy_filename.exists():
            return
        gpx_filename = folder / "gpx_files" / f"{name}_{ii:04}_{jj:04}.gpx"
        func(xy_filename, gpx_filename)
    except Exception:
        pass
