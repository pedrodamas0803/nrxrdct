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

    def get_chi2_map(self) -> np.ndarray:
        """
        Extract the reduced chi-squared (χ²) from each voxel's .gpx file.

        χ² is a histogram-level metric and does not depend on the phase index.

        Returns:
            np.ndarray: 2-D map of shape ``(nx, ny)``; masked pixels return ``0``,
                failed extractions return ``nan``.
        """
        t0 = time.time()
        nx, ny = self.volume.shape[1], self.volume.shape[2]
        active = self._active_indices
        args = [(ii, jj, self.folder, self.name) for ii, jj in active]

        with concurrent.futures.ProcessPoolExecutor(NTHREADS) as pool:
            values = list(
                tqdm(pool.map(_get_chi2_ii_jj, args, chunksize=64), total=len(args), desc="Chi2 map")
            )

        result = np.zeros((nx, ny), dtype=np.float32)
        for (ii, jj), val in zip(active, values):
            result[ii, jj] = val
        print(f"Fetched chi2 map in {time.time()-t0:.2f} s.")
        return result

    def get_cell_map(
        self, phase: int = 0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract unit-cell lengths a, b, c from each voxel's .gpx file.

        Args:
            phase (int, optional): Zero-based index of the phase to extract
                (default ``0``).

        Returns:
            tuple: ``(a_map, b_map, c_map)`` — 2-D arrays of shape ``(nx, ny)``;
                masked pixels return ``0``, failed extractions return ``nan``.
        """
        t0 = time.time()
        nx, ny = self.volume.shape[1], self.volume.shape[2]
        active = self._active_indices
        args = [(ii, jj, self.folder, self.name, phase) for ii, jj in active]

        with concurrent.futures.ProcessPoolExecutor(NTHREADS) as pool:
            values = list(
                tqdm(
                    pool.map(_get_cell_params_ii_jj, args, chunksize=64),
                    total=len(args),
                    desc=f"Cell map (phase {phase})",
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

    def get_crystallite_size_map(self, phase: int = 0) -> np.ndarray:
        """
        Extract the refined isotropic crystallite size from each voxel's .gpx file.

        Args:
            phase (int, optional): Zero-based index of the phase to extract
                (default ``0``).

        Returns:
            np.ndarray: 2-D map of shape ``(nx, ny)``; masked pixels return ``0``,
                failed extractions return ``nan``.
        """
        t0 = time.time()
        nx, ny = self.volume.shape[1], self.volume.shape[2]
        active = self._active_indices
        args = [(ii, jj, self.folder, self.name, phase) for ii, jj in active]

        with concurrent.futures.ProcessPoolExecutor(NTHREADS) as pool:
            values = list(
                tqdm(
                    pool.map(_get_crystallite_sizes, args, chunksize=64),
                    total=len(args),
                    desc=f"Size map (phase {phase})",
                )
            )

        size_map = np.zeros((nx, ny), dtype=np.float32)
        for (ii, jj), val in zip(active, values):
            size_map[ii, jj] = val
        print(f"Fetched crystallite size map in {time.time()-t0:.2f} s.")
        return size_map

    def get_microstrain_map(self, phase: int = 0) -> np.ndarray:
        """
        Extract the refined isotropic microstrain from each voxel's .gpx file.

        Args:
            phase (int, optional): Zero-based index of the phase to extract
                (default ``0``).

        Returns:
            np.ndarray: 2-D map of shape ``(nx, ny)``; masked pixels return ``0``,
                failed extractions return ``nan``.
        """
        t0 = time.time()
        nx, ny = self.volume.shape[1], self.volume.shape[2]
        active = self._active_indices
        args = [(ii, jj, self.folder, self.name, phase) for ii, jj in active]

        with concurrent.futures.ProcessPoolExecutor(NTHREADS) as pool:
            values = list(
                tqdm(
                    pool.map(_get_microstrain_ii_jj, args, chunksize=64),
                    total=len(args),
                    desc=f"Microstrain map (phase {phase})",
                )
            )

        mustrain_map = np.zeros((nx, ny), dtype=np.float32)
        for (ii, jj), val in zip(active, values):
            mustrain_map[ii, jj] = val
        print(f"Fetched microstrain map in {time.time()-t0:.2f} s.")
        return mustrain_map

    def get_scale_map(self, phase: int = 0) -> np.ndarray:
        """
        Extract the refined HAP scale factor from each voxel's .gpx file.

        The HAP scale factor is proportional to the phase fraction and can be
        used as a proxy for phase abundance.

        Args:
            phase (int, optional): Zero-based index of the phase to extract
                (default ``0``).

        Returns:
            np.ndarray: 2-D map of shape ``(nx, ny)``; masked pixels return ``0``,
                failed extractions return ``nan``.
        """
        t0 = time.time()
        nx, ny = self.volume.shape[1], self.volume.shape[2]
        active = self._active_indices
        args = [(ii, jj, self.folder, self.name, phase) for ii, jj in active]

        with concurrent.futures.ProcessPoolExecutor(NTHREADS) as pool:
            values = list(
                tqdm(
                    pool.map(_get_scale_ii_jj, args, chunksize=64),
                    total=len(args),
                    desc=f"Scale map (phase {phase})",
                )
            )

        scale_map = np.zeros((nx, ny), dtype=np.float32)
        for (ii, jj), val in zip(active, values):
            scale_map[ii, jj] = val
        print(f"Fetched scale map in {time.time()-t0:.2f} s.")
        return scale_map

    def get_all_maps(self, phase: int = 0) -> dict:
        """
        Extract all refined parameters from every voxel's .gpx file in a
        **single** parallel pass.

        Args:
            phase (int, optional): Zero-based index of the phase to extract
                (default ``0``).

        Returns:
            dict: 2-D maps of shape ``(nx, ny)`` keyed by parameter name:

            * ``"rwp"``      — weighted R-factor (histogram level, phase-independent)
            * ``"chi2"``     — reduced chi-squared (histogram level, phase-independent)
            * ``"a"``, ``"b"``, ``"c"`` — unit-cell lengths (Å)
            * ``"size"``     — isotropic crystallite size
            * ``"mustrain"`` — isotropic microstrain
            * ``"scale"``    — HAP scale factor

            Masked pixels are ``0``; active pixels that failed extraction are ``nan``.
        """
        t0 = time.time()
        nx, ny = self.volume.shape[1], self.volume.shape[2]
        active = self._active_indices
        args = [(ii, jj, self.folder, self.name, phase) for ii, jj in active]

        with concurrent.futures.ProcessPoolExecutor(NTHREADS) as pool:
            values = list(
                tqdm(
                    pool.map(_get_all_maps_ii_jj, args, chunksize=64),
                    total=len(args),
                    desc=f"All maps (phase {phase})",
                )
            )

        keys = ["rwp", "chi2", "a", "b", "c", "size", "mustrain", "scale"]
        maps = {k: np.zeros((nx, ny), dtype=np.float32) for k in keys}
        for (ii, jj), vals in zip(active, values):
            for k, v in zip(keys, vals):
                maps[k][ii, jj] = v
        print(f"Fetched all maps in {time.time()-t0:.2f} s.")
        return maps

    def list_phases(
        self,
        ii: int | None = None,
        jj: int | None = None,
        verbose: bool = True,
    ) -> list:
        """
        Open one voxel's .gpx file and return the index and name of each phase.

        Useful for discovering phase indices before calling map methods with
        ``phase=N``.  Defaults to the centre pixel of the volume.

        Args:
            ii (int or None, optional): Row index of the voxel to inspect.
                Defaults to the centre row.
            jj (int or None, optional): Column index of the voxel to inspect.
                Defaults to the centre column.
            verbose (bool, optional): Print the phase list (default ``True``).

        Returns:
            list of ``(index, name)`` tuples, one per phase found in the project.

        Raises:
            FileNotFoundError: If the .gpx file for the requested voxel does not
                exist (voxel masked out or refinement not yet run).
        """
        if ii is None:
            ii = self.volume.shape[1] // 2
        if jj is None:
            jj = self.volume.shape[2] // 2

        gpx_filename = self.folder / "gpx_files" / f"{self.name}_{ii:04}_{jj:04}.gpx"
        if not gpx_filename.exists():
            raise FileNotFoundError(
                f"No .gpx file found for voxel ({ii}, {jj}): {gpx_filename}\n"
                "Make sure refinements have been run for this voxel."
            )

        _, _, phases = _load_data_from_gpx(gpx_filename)
        result = [(idx, ph.name) for idx, ph in enumerate(phases)]
        if verbose:
            for idx, name in result:
                print(f"  phase {idx}: {name}")
        return result

    def plot_maps(
        self,
        phase: int = 0,
        cmap: str = "viridis",
        figsize: tuple | None = None,
        return_fig: bool = False,
    ):
        """
        Plot a mosaic of all refined parameter maps for a given phase.

        Calls :meth:`get_all_maps` internally.  Masked-out pixels are shown
        as white.  The colour range of each panel is computed from the
        unmasked, finite values only.

        Args:
            phase (int, optional): Phase index to display (default ``0``).
                Use :meth:`list_phases` to discover available indices.
            cmap (str, optional): Matplotlib colormap name
                (default ``"viridis"``).
            figsize (tuple or None, optional): ``(width, height)`` in inches.
                Defaults to ``(4 * ncols, 4 * nrows)``.
            return_fig (bool, optional): If ``True``, return the
                ``matplotlib.figure.Figure`` object.  Set to ``False``
                (default) to avoid duplicate display in Jupyter notebooks.

        Returns:
            matplotlib.figure.Figure or None
        """
        import matplotlib.pyplot as plt

        maps = self.get_all_maps(phase=phase)

        labels = {
            "rwp":      "Rwp",
            "chi2":     "χ²",
            "a":        "a (Å)",
            "b":        "b (Å)",
            "c":        "c (Å)",
            "size":     "Crystallite size",
            "mustrain": "Microstrain",
            "scale":    "Scale",
        }

        keys = list(labels.keys())
        n = len(keys)
        ncols = 4
        nrows = -(-n // ncols)  # ceiling division

        if figsize is None:
            figsize = (ncols * 4, nrows * 4)

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        axes_flat = np.array(axes).flatten()

        for idx, key in enumerate(keys):
            ax = axes_flat[idx]
            data = maps[key].astype(float)
            if self.mask is not None:
                data[self.mask == 0] = np.nan

            finite = data[np.isfinite(data)]
            vmin, vmax = (float(finite.min()), float(finite.max())) if finite.size else (0, 1)

            im = ax.imshow(data, cmap=cmap, origin="lower", vmin=vmin, vmax=vmax)
            ax.set_title(labels[key])
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        for ax in axes_flat[n:]:
            ax.set_visible(False)

        try:
            phase_name = self.list_phases(verbose=False)[phase][1]
            title = f"{self.name}  —  phase {phase}: {phase_name}"
        except Exception:
            title = f"{self.name}  —  phase {phase}"

        fig.suptitle(title, fontsize=13)
        fig.tight_layout()
        return fig if return_fig else None

    def fit_peak_map(
        self,
        center: float,
        window: float,
        model: str = "pseudo_voigt",
        bg_method: str = "snip",
        bg_kwargs: dict | None = None,
        fit_mask: np.ndarray | None = None,
        output_h5: Path | str | None = None,
        n_workers: int | None = None,
        plot: bool = False,
        cmap: str = "viridis",
        figsize: tuple | None = None,
        return_fig: bool = False,
    ) -> dict:
        """
        Fit a single diffraction peak for every active voxel and return
        2-D parameter maps.

        For each unmasked voxel the spectrum ``volume[:, ii, jj]`` is passed
        to :func:`~nrxrdct.peakfit.fit_peak`.  The background is estimated
        with ``pybaselines`` before fitting, so no prior background subtraction
        is needed.  Failed fits return ``nan`` at that pixel.

        Args:
            center (float): Nominal peak centre in degrees 2θ.
            window (float): Total fitting window width in degrees around
                *center*.
            model (str): Peak profile – ``"gaussian"``, ``"lorentzian"``,
                ``"voigt"``, or ``"pseudo_voigt"`` (default).
            bg_method (str): Background algorithm forwarded to
                :func:`~nrxrdct.utils.calculate_xrd_baseline`.
                Options: ``"snip"`` (default), ``"iasls"``, ``"aspls"``,
                ``"arpls"``, ``"mor"``.
            bg_kwargs (dict or None): Extra keyword arguments for the
                background estimator.
            fit_mask (np.ndarray or None): Boolean array of shape ``(nx, ny)``.
                When supplied, only pixels that are truthy in *both*
                ``self.mask`` (if set) and *fit_mask* are fitted.  Pixels
                outside *fit_mask* are left as ``nan`` in the output maps.
                ``None`` fits all active pixels (default).
            output_h5 (Path, str, or None): If provided, all parameter maps
                are saved to this HDF5 file under a group named after the
                peak centre (e.g. ``"peak_3.5600"``).  Fit metadata
                (``center``, ``window``, ``model``, ``bg_method``) are stored
                as group attributes.  The file is created if it does not exist
                and the group is overwritten if it already does (default
                ``None``).
            n_workers (int or None): Number of worker threads.  ``None``
                uses :data:`NTHREADS` (cpu_count − 2).
            plot (bool): If ``True``, display a mosaic of all parameter maps
                after fitting (default ``False``).
            cmap (str): Matplotlib colormap used in the mosaic
                (default ``"viridis"``).
            figsize (tuple or None): ``(width, height)`` in inches for the
                mosaic figure.  Defaults to ``(4 * ncols, 4 * nrows)``.
            return_fig (bool): If ``True``, return the
                ``matplotlib.figure.Figure`` alongside the maps dict.
                Ignored when *plot* is ``False`` (default ``False``).

        Returns:
            dict[str, np.ndarray]: 2-D maps of shape ``(nx, ny)`` keyed by
            parameter name.  Always present: ``"center"``, ``"amplitude"``,
            ``"fwhm"``, ``"area"``, ``"residual"``, ``"success"``.
            Model-specific: ``"sigma"`` (Gaussian / Voigt),
            ``"gamma"`` (Lorentzian / Voigt), ``"eta"`` (pseudo-Voigt).
            Pixels outside the effective mask and failed fits are ``nan``;
            ``"success"`` uses ``1.0`` / ``0.0``.
            When *plot* is ``True`` and *return_fig* is ``True``, returns a
            ``(maps, fig)`` tuple instead.

        Example::

            maps = vol.fit_peak_map(center=3.56, window=0.4, plot=True)

            # Fit only inside a phase region identified by a separate mask
            maps = vol.fit_peak_map(center=3.56, window=0.4,
                                    fit_mask=austenite_mask, plot=True)
        """
        from .peakfit import fit_peak as _fit_peak

        nx, ny = self.volume.shape[1], self.volume.shape[2]

        if fit_mask is not None:
            fit_mask = np.asarray(fit_mask)
            if fit_mask.shape != (nx, ny):
                raise ValueError(
                    f"fit_mask shape {fit_mask.shape} does not match "
                    f"volume spatial shape {(nx, ny)}."
                )
            # Intersect with self.mask when both are present
            if self.mask is not None:
                effective = [(ii, jj) for ii, jj in self._active_indices
                             if fit_mask[ii, jj]]
            else:
                effective = [(ii, jj) for ii in range(nx) for jj in range(ny)
                             if fit_mask[ii, jj]]
        else:
            effective = self._active_indices

        workers = n_workers if n_workers is not None else NTHREADS

        def _fit_one(idx: Tuple[int, int]) -> Tuple[int, int, dict]:
            ii, jj = idx
            return ii, jj, _fit_peak(
                self.tth, self.volume[:, ii, jj],
                center, window, model, bg_method, bg_kwargs,
            )

        collected: dict[str, np.ndarray] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            for ii, jj, params in tqdm(
                pool.map(_fit_one, effective),
                total=len(effective),
                desc=f"Fitting peak @ {center} °  [{model}]",
            ):
                for key, val in params.items():
                    if key not in collected:
                        collected[key] = np.full((nx, ny), np.nan, dtype=np.float32)
                    collected[key][ii, jj] = float(val) if val is not None else np.nan

        # ── Save to HDF5 ─────────────────────────────────────────────────
        if output_h5 is not None:
            group_name = f"peak_{center:.4f}"
            with h5py.File(str(output_h5), "a") as f:
                if group_name in f:
                    del f[group_name]
                grp = f.create_group(group_name)
                grp.attrs["center"]    = center
                grp.attrs["window"]    = window
                grp.attrs["model"]     = model
                grp.attrs["bg_method"] = bg_method
                grp.attrs["sample"]    = self.name
                for key, arr in collected.items():
                    grp.create_dataset(key, data=arr, compression="gzip")
            print(f"Peak-fit maps saved → {output_h5}  (group '{group_name}')")

        if not plot:
            return collected

        # ── Mosaic plot ───────────────────────────────────────────────────
        import matplotlib.pyplot as plt

        # Human-readable labels and preferred display order
        label_map = {
            "center":    f"Centre (°)",
            "amplitude": "Amplitude",
            "fwhm":      "FWHM (°)",
            "area":      "Area",
            "sigma":     "σ (°)",
            "gamma":     "γ (°)",
            "eta":       "η  (Lorentzian fraction)",
            "residual":  "RMS residual",
            "r2":        "R²",
            "success":   "Success",
        }
        # Show keys in preferred order; put model-specific ones after the common set
        order = ["center", "amplitude", "fwhm", "area",
                 "sigma", "gamma", "eta", "residual", "r2", "success"]
        keys = [k for k in order if k in collected] + \
               [k for k in collected if k not in order]

        n = len(keys)
        ncols = 3
        nrows = -(-n // ncols)  # ceiling division

        if figsize is None:
            figsize = (ncols * 4, nrows * 4)

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        axes_flat = np.array(axes).flatten()

        # Colormap copy with a distinct colour for masked / failed pixels
        import matplotlib.cm as _cm
        cmap_obj = _cm.get_cmap(cmap).copy()
        cmap_obj.set_bad(color="#aaaaaa")  # light grey for NaN / masked

        for idx, key in enumerate(keys):
            ax = axes_flat[idx]
            data = collected[key].astype(float)
            if self.mask is not None:
                data[self.mask == 0] = np.nan
            if fit_mask is not None:
                data[fit_mask == 0] = np.nan

            finite = data[np.isfinite(data)]
            vmin, vmax = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)

            im = ax.imshow(data, cmap=cmap_obj, origin="lower", vmin=vmin, vmax=vmax)
            ax.set_title(label_map.get(key, key))
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        for ax in axes_flat[n:]:
            ax.set_visible(False)

        fig.suptitle(
            f"{self.name}  —  peak fit @ {center} °  [{model}]", fontsize=13
        )
        fig.tight_layout()

        return (collected, fig) if return_fig else collected

    def pick_and_refine_jupyter(
        self,
        refining_function: Callable[[Path, Path], Any] | None = None,
        projection: str | int = "max",
        colormap: str = "viridis",
        contrast_limits: tuple | None = None,
        figsize: tuple = (12, 5),
        phases=None,
        return_fig: bool = False,
    ):
        """
        Interactive Jupyter picker: click a pixel on the projected map to
        extract its diffraction spectrum, save it as a ``.xy`` file, and
        optionally run a Rietveld refinement.

        Layout
        ------
        * **Left panel** – 2-D projection (or single tth-slice) of the volume.
          Click any pixel to select it; a crosshair marks the selection.
        * **Right panel** – diffraction pattern at the selected pixel plotted
          against the 2θ axis. When *phases* is supplied a tick row is drawn
          below the profile.
        * **Buttons** – "Save .xy" writes the spectrum to disk; "Save .xy &
          Refine" also runs *refining_function* on that file.

        Args:
            refining_function (callable or None): Function with signature
                ``f(xy_file: Path, gpx_file: Path)`` that performs the GSAS-II
                refinement and saves results to *gpx_file*.  When ``None`` the
                "Save .xy & Refine" button is disabled (default ``None``).
            projection (str or int): How to collapse the tth axis into a 2-D
                map.  ``"max"`` (default), ``"mean"``, or ``"sum"`` apply the
                corresponding reduction; an integer is used as a tth-axis index
                (single diffraction slice).
            colormap (str): Matplotlib colormap for the map panel
                (default ``"viridis"``).
            contrast_limits (tuple or None): ``(vmin, vmax)`` for the map
                image. Defaults to data min / max.
            figsize (tuple): Figure size in inches (default ``(12, 5)``).
            phases (dict or None): Phase peak positions forwarded to the
                spectrum panel. Same format as
                :func:`~nrxrdct.visualization.visualize_slices_with_profile_jupyter`:
                a ``dict`` mapping phase name → list of 2θ positions (float)
                or a ``pandas.DataFrame`` with a ``"tth"`` column.
            return_fig (bool): If ``True``, return the
                ``matplotlib.figure.Figure`` object (default ``False``).

        Returns:
            matplotlib.figure.Figure or None

        Note:
            The cell must use ``%matplotlib widget`` (ipympl backend) and have
            ``ipywidgets`` installed::

                %matplotlib widget
                vol.pick_and_refine_jupyter(refining_function=my_refine)
        """
        try:
            import ipywidgets as widgets
            import matplotlib.gridspec as gridspec
            import matplotlib.pyplot as plt
            from IPython.display import display
        except ImportError as exc:
            raise ImportError(
                "pick_and_refine_jupyter requires ipywidgets and ipympl.\n"
                "Install them with:  pip install ipywidgets ipympl"
            ) from exc

        from .visualization import _draw_phase_ticks

        n_tth, nx, ny = self.volume.shape

        # ── Output folders ────────────────────────────────────────────────
        out_xy  = self.folder / "refinement_pixel" / "xy_files"
        out_gpx = self.folder / "refinement_pixel" / "gpx_files"
        out_xy.mkdir(parents=True, exist_ok=True)
        out_gpx.mkdir(parents=True, exist_ok=True)

        # ── 2-D projection map ────────────────────────────────────────────
        if isinstance(projection, int):
            if not (0 <= projection < n_tth):
                raise ValueError(
                    f"projection index {projection} out of range [0, {n_tth - 1}]."
                )
            map_2d = self.volume[projection]
            proj_label = f"slice {projection}"
        elif projection == "max":
            map_2d = self.volume.max(axis=0)
            proj_label = "max projection"
        elif projection == "mean":
            map_2d = self.volume.mean(axis=0)
            proj_label = "mean projection"
        elif projection == "sum":
            map_2d = self.volume.sum(axis=0)
            proj_label = "sum projection"
        else:
            raise ValueError(
                f"projection must be 'max', 'mean', 'sum', or an int; got {projection!r}."
            )

        vmin_map, vmax_map = (
            contrast_limits
            if contrast_limits
            else (float(map_2d.min()), float(map_2d.max()))
        )

        # ── Shared state ──────────────────────────────────────────────────
        state: dict = {"ii": None, "jj": None}

        # ── Figure & axes ─────────────────────────────────────────────────
        with plt.ioff():
            fig = plt.figure(figsize=figsize, facecolor="#0e1117")
        fig.suptitle(
            f"{self.name}  —  pick a pixel to extract spectrum",
            color="#e6edf3",
            fontsize=12,
            fontweight="bold",
            y=0.98,
        )

        gs = gridspec.GridSpec(
            1, 2,
            figure=fig,
            left=0.06, right=0.97, bottom=0.12, top=0.91, wspace=0.35,
        )

        ax_map = fig.add_subplot(gs[0], facecolor="#161b22")

        if phases:
            n_phases = len(phases)
            gs_right = gridspec.GridSpecFromSubplotSpec(
                2, 1,
                subplot_spec=gs[1],
                height_ratios=[5, max(1, n_phases)],
                hspace=0.06,
            )
            ax_prof = fig.add_subplot(gs_right[0], facecolor="#161b22")
            ax_ticks = fig.add_subplot(gs_right[1], facecolor="#161b22", sharex=ax_prof)
            _draw_phase_ticks(ax_ticks, phases, self.tth)
            ax_ticks.set_xlabel("2θ (°)", color="#8b949e", fontsize=8)
            ax_prof.tick_params(colors="#8b949e", labelsize=7, axis="x", labelbottom=False)
            ax_prof.tick_params(colors="#8b949e", labelsize=7, axis="y")
        else:
            ax_prof = fig.add_subplot(gs[1], facecolor="#161b22")
            ax_prof.tick_params(colors="#8b949e", labelsize=7)
            ax_prof.set_xlabel("2θ (°)", color="#8b949e", fontsize=8)

        # Map panel styling
        ax_map.tick_params(colors="#8b949e", labelsize=7)
        for sp in ax_map.spines.values():
            sp.set_edgecolor("#30363d")
        ax_map.set_xlabel("j  (column)", color="#8b949e", fontsize=8)
        ax_map.set_ylabel("i  (row)", color="#8b949e", fontsize=8)
        ax_map.set_title(
            f"Map ({proj_label})  —  click to pick pixel",
            color="#8b949e", fontsize=9, pad=6,
        )

        ax_map.imshow(
            map_2d,
            cmap=colormap,
            vmin=vmin_map,
            vmax=vmax_map,
            origin="upper",
            interpolation="nearest",
            aspect="auto",
        )
        (dot,) = ax_map.plot(
            [], [], "o",
            color="#ff4444", markersize=7,
            markeredgecolor="white", markeredgewidth=0.9,
        )
        (hline,) = ax_map.plot([], [], color="#f0883e", linewidth=0.8, alpha=0.7)
        (vline_img,) = ax_map.plot([], [], color="#f0883e", linewidth=0.8, alpha=0.7)

        # Profile panel styling
        for sp in ax_prof.spines.values():
            sp.set_edgecolor("#30363d")
        ax_prof.set_ylabel("Intensity", color="#8b949e", fontsize=8)
        ax_prof.set_xlim(float(self.tth[0]), float(self.tth[-1]))
        prof_title = ax_prof.set_title(
            "Spectrum  —  select a pixel", color="#8b949e", fontsize=9, pad=6,
        )
        (profile_line,) = ax_prof.plot(
            [], [], color="#58a6ff", linewidth=1.4, solid_capstyle="round",
        )

        # ── Widgets ───────────────────────────────────────────────────────
        status_label = widgets.HTML(
            value="<span style='color:#8b949e;font-size:12px'>No pixel selected.</span>",
            layout=widgets.Layout(width="auto"),
        )
        btn_save = widgets.Button(
            description="Save .xy",
            button_style="info",
            disabled=True,
            layout=widgets.Layout(width="130px"),
            icon="download",
        )
        btn_refine = widgets.Button(
            description="Save .xy & Refine",
            button_style="success",
            disabled=True,
            layout=widgets.Layout(width="180px"),
            icon="cogs",
            tooltip=(
                "Provide a refining_function to enable."
                if refining_function is None
                else "Save spectrum and run Rietveld refinement."
            ),
        )

        def _set_status(msg: str, color: str = "#8b949e") -> None:
            status_label.value = (
                f"<span style='color:{color};font-size:12px'>{msg}</span>"
            )

        def _redraw_profile(ii: int, jj: int) -> None:
            profile = self.volume[:, ii, jj]
            profile_line.set_xdata(self.tth)
            profile_line.set_ydata(profile)
            p_min, p_max = float(profile.min()), float(profile.max())
            margin = max((p_max - p_min) * 0.05, 1e-9)
            ax_prof.set_ylim(p_min - margin, p_max + margin)
            prof_title.set_text(f"Spectrum  |  pixel  i={ii},  j={jj}")
            # Crosshair on map
            hline.set_xdata([0, ny - 1])
            hline.set_ydata([ii, ii])
            vline_img.set_xdata([jj, jj])
            vline_img.set_ydata([0, nx - 1])
            dot.set_xdata([jj])
            dot.set_ydata([ii])
            fig.canvas.draw_idle()

        def on_click(event) -> None:
            if event.inaxes is not ax_map or event.button != 1:
                return
            jj = int(np.clip(round(event.xdata), 0, ny - 1))
            ii = int(np.clip(round(event.ydata), 0, nx - 1))
            state["ii"] = ii
            state["jj"] = jj
            _redraw_profile(ii, jj)
            btn_save.disabled = False
            if refining_function is not None:
                btn_refine.disabled = False
            xy_path = out_xy / f"{self.name}_{ii:04}_{jj:04}.xy"
            _set_status(
                f"Selected pixel  i={ii}, j={jj}  →  {xy_path.name}", "#79c0ff"
            )

        fig.canvas.mpl_connect("button_press_event", on_click)

        def on_save(_) -> None:
            ii, jj = state["ii"], state["jj"]
            if ii is None:
                return
            xy_path = out_xy / f"{self.name}_{ii:04}_{jj:04}.xy"
            save_xy_file(
                self.tth, self.volume[:, ii, jj], None, str(xy_path), verbose=False
            )
            _set_status(f"Saved  {xy_path}", "#7ee787")

        def on_save_and_refine(_) -> None:
            import io as _io
            import sys
            import traceback

            ii, jj = state["ii"], state["jj"]
            if ii is None or refining_function is None:
                return
            btn_save.disabled = True
            btn_refine.disabled = True
            xy_path = out_xy / f"{self.name}_{ii:04}_{jj:04}.xy"
            gpx_path = out_gpx / f"{self.name}_{ii:04}_{jj:04}.gpx"
            res_path = out_gpx / f"{self.name}_{ii:04}_{jj:04}.res"
            _set_status(f"Saving  {xy_path.name} …", "#f0883e")
            save_xy_file(
                self.tth, self.volume[:, ii, jj], None, str(xy_path), verbose=False
            )
            _set_status(f"Refining  {xy_path.name} …", "#f0883e")
            buf = _io.StringIO()
            old_stdout, old_stderr = sys.stdout, sys.stderr
            sys.stdout = sys.stderr = buf
            status = "ok"
            try:
                refining_function(xy_path, gpx_path)
            except Exception as exc:
                traceback.print_exc()
                status = str(exc)
            finally:
                sys.stdout, sys.stderr = old_stdout, old_stderr
            output = buf.getvalue()
            res_path.write_text(output, encoding="utf-8")
            if status == "ok":
                _set_status(f"Done  —  {gpx_path.name}  (log → {res_path.name})", "#7ee787")
            else:
                _set_status(f"Refinement failed: {status}  (log → {res_path.name})", "#ff7b72")
            btn_save.disabled = False
            if refining_function is not None:
                btn_refine.disabled = False

        btn_save.on_click(on_save)
        btn_refine.on_click(on_save_and_refine)

        controls = widgets.VBox(
            [
                widgets.HBox([btn_save, btn_refine]),
                status_label,
            ],
            layout=widgets.Layout(margin="6px 0 0 0"),
        )
        display(widgets.VBox([fig.canvas, controls]))

        if return_fig:
            return fig

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

    def write_slurm_scripts_peak_fit(
        self,
        volume_hdf5: Path,
        center: float,
        window: float,
        model: str = "pseudo_voigt",
        bg_method: str = "snip",
        fit_mask: np.ndarray | None = None,
        n_array_jobs: int = 500,
        conda_env: str = "nrxrdct",
        conda_base: str = "",
        python_executable: str = "",
        mem: str = "2G",
        time_limit: str = "01:00:00",
        partition: str = "all",
    ) -> Tuple[Path, Path]:
        """
        Generate a SLURM array-job script and a matching Python worker for
        cluster-side single-peak fitting of all voxels.

        Each array element processes a contiguous chunk of active (unmasked)
        voxels and writes its results to a ``peak_fit_results/chunk_NNNN.npz``
        file.  Once all jobs have finished, call :meth:`load_peak_fit_maps` to
        assemble the per-parameter 2-D maps.

        The volume and 2θ axis must be saved to *volume_hdf5* in advance::

            import h5py
            with h5py.File("volume.h5", "a") as f:
                f["volume"] = vol.volume
                f["tth"]    = vol.tth
                if vol.mask is not None:
                    f["mask"] = vol.mask

        When *fit_mask* is provided it is written to the same HDF5 file under
        the key ``"fit_mask"`` automatically — no manual step required.

        Args:
            volume_hdf5 (Path): HDF5 file containing ``"volume"``, ``"tth"``,
                and optionally ``"mask"``.
            center (float): Nominal peak centre in degrees 2θ.
            window (float): Total fitting window width in degrees.
            model (str): Peak profile – ``"gaussian"``, ``"lorentzian"``,
                ``"voigt"``, or ``"pseudo_voigt"`` (default).
            bg_method (str): Background algorithm passed to
                :func:`~nrxrdct.utils.calculate_xrd_baseline`
                (default ``"snip"``).
            fit_mask (np.ndarray or None): Boolean array of shape ``(nx, ny)``.
                Only pixels that are truthy in *both* ``self.mask`` (if set)
                and *fit_mask* are fitted.  Written to *volume_hdf5* under the
                key ``"fit_mask"`` so the cluster workers can read it.
                ``None`` fits all active pixels (default).
            n_array_jobs (int): Number of SLURM array elements
                (default ``500``).
            conda_env (str): Conda environment name (default ``"nrxrdct"``).
            conda_base (str): Path to the conda installation root.  When set,
                ``<conda_base>/etc/profile.d/conda.sh`` is sourced before
                ``conda activate``.  Leave empty to skip.
            python_executable (str): Absolute path to the Python binary.
                When set, the conda activation block is skipped entirely.
                Leave empty to use conda activation instead.
            mem (str): Memory per job (default ``"2G"``).
            time_limit (str): Wall-time limit (default ``"01:00:00"``).
            partition (str): SLURM partition (default ``"all"``).

        Returns:
            tuple: ``(worker_path, submit_path)`` — absolute paths to the
                generated Python worker and the ``sbatch`` submission script.
                Submit with ``sbatch <submit_path>``.
        """
        nx, ny = self.volume.shape[1], self.volume.shape[2]
        folder_abs = str(self.folder.resolve())
        volume_abs = str(Path(volume_hdf5).resolve())
        out_dir_abs = str((self.folder / "peak_fit_results").resolve())
        worker_path = self.folder / "worker_peak_fit.py"
        submit_path = self.folder / "submit_peak_fit.sh"

        # ── Persist fit_mask to the HDF5 so workers can load it ──────────
        if fit_mask is not None:
            fit_mask = np.asarray(fit_mask)
            if fit_mask.shape != (nx, ny):
                raise ValueError(
                    f"fit_mask shape {fit_mask.shape} does not match "
                    f"volume spatial shape {(nx, ny)}."
                )
            with h5py.File(str(volume_hdf5), "a") as f:
                if "fit_mask" in f:
                    del f["fit_mask"]
                f.create_dataset("fit_mask", data=fit_mask.astype(np.uint8))
            print(f"fit_mask written → {volume_hdf5}  (key 'fit_mask')")

        worker_template = (
            "#!/usr/bin/env python\n"
            '"""SLURM array worker — fit a single peak for a chunk of voxels.\n\n'
            "Usage::\n\n"
            "    python worker_peak_fit.py --job-id JOB_ID --n-jobs N_JOBS\n\n"
            "*job_id* is SLURM_ARRAY_TASK_ID (0-indexed).\n"
            '"""\n\n'
            "import argparse\n"
            "from pathlib import Path\n\n"
            "import h5py\n"
            "import numpy as np\n\n"
            "from nrxrdct.peakfit import fit_peak\n\n\n"
            "def main() -> None:\n"
            "    parser = argparse.ArgumentParser()\n"
            '    parser.add_argument("--job-id", type=int, required=True)\n'
            '    parser.add_argument("--n-jobs", type=int, required=True)\n'
            "    args = parser.parse_args()\n\n"
            "    with h5py.File(Path(VOLUME_HDF5), 'r') as h:\n"
            '        volume   = h["volume"][:]\n'
            '        tth      = h["tth"][:]\n'
            '        mask     = h["mask"][:]     if "mask"     in h else None\n'
            '        fit_mask = h["fit_mask"][:] if "fit_mask" in h else None\n\n'
            "    nx, ny = NX, NY\n"
            "    all_indexes = [\n"
            "        (ii, jj)\n"
            "        for ii in range(nx)\n"
            "        for jj in range(ny)\n"
            "        if (mask     is None or mask[ii, jj])\n"
            "        and (fit_mask is None or fit_mask[ii, jj])\n"
            "    ]\n"
            "    chunk_size = -(-len(all_indexes) // args.n_jobs)\n"
            "    start      = args.job_id * chunk_size\n"
            "    chunk      = all_indexes[start : start + chunk_size]\n"
            "    if not chunk:\n"
            "        return\n\n"
            "    out_dir = Path(OUT_DIR)\n"
            "    out_dir.mkdir(parents=True, exist_ok=True)\n\n"
            "    rows = []\n"
            "    for ii, jj in chunk:\n"
            "        params = fit_peak(\n"
            "            tth, volume[:, ii, jj],\n"
            "            center=CENTER, window=WINDOW,\n"
            "            model=MODEL, bg_method=BG_METHOD,\n"
            "        )\n"
            '        rows.append({"ii": float(ii), "jj": float(jj), **params})\n\n'
            "    keys   = list(rows[0].keys())\n"
            "    arrays = {k: np.array([r[k] for r in rows], dtype=np.float32)\n"
            "              for k in keys}\n"
            "    out_file = out_dir / f'chunk_{args.job_id:04d}.npz'\n"
            "    np.savez(out_file, **arrays)\n"
            '    print(f"Saved {len(rows)} results → {out_file}")\n\n\n'
            'if __name__ == "__main__":\n'
            "    main()\n"
        )

        worker_code = (
            worker_template
            .replace("VOLUME_HDF5", repr(volume_abs))
            .replace("NX, NY",      f"{nx}, {ny}")
            .replace("OUT_DIR",     repr(out_dir_abs))
            .replace("CENTER",      repr(float(center)))
            .replace("WINDOW",      repr(float(window)))
            .replace("MODEL",       repr(model))
            .replace("BG_METHOD",   repr(bg_method))
        )

        submit_script = (
            "#!/bin/bash\n"
            f"#SBATCH --job-name={self.name}_peak_fit\n"
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
        print(f"Worker script  : {worker_path}")
        print(f"Submit script  : {submit_path}")
        print(f"Results folder : {out_dir_abs}")
        print(f"Submit with    : sbatch {submit_path}")
        print(f"Load results   : vol.load_peak_fit_maps()")
        return worker_path, submit_path

    def load_peak_fit_maps(
        self,
        plot: bool = False,
        cmap: str = "viridis",
        figsize: tuple | None = None,
        return_fig: bool = False,
    ) -> dict:
        """
        Assemble per-voxel peak-fit results from SLURM chunk files into 2-D
        parameter maps.

        Reads every ``peak_fit_results/chunk_*.npz`` file written by the
        worker generated by :meth:`write_slurm_scripts_peak_fit` and
        accumulates the results into arrays of shape ``(nx, ny)``.  Pixels
        that were not processed (masked or missing) remain ``nan``.

        Args:
            plot (bool): If ``True``, display a mosaic of all maps after
                loading (default ``False``).
            cmap (str): Matplotlib colormap for the mosaic
                (default ``"viridis"``).
            figsize (tuple or None): Figure size in inches.  Defaults to
                ``(4 * ncols, 4 * nrows)``.
            return_fig (bool): If ``True`` and *plot* is ``True``, return
                ``(maps, fig)`` instead of just *maps*.

        Returns:
            dict[str, np.ndarray]: Same format as :meth:`fit_peak_map`.

        Raises:
            FileNotFoundError: No chunk files found — jobs have not finished
                or the output folder is missing.
        """
        out_dir = self.folder / "peak_fit_results"
        chunk_files = sorted(out_dir.glob("chunk_*.npz"))
        if not chunk_files:
            raise FileNotFoundError(
                f"No chunk files found in {out_dir}. "
                "Run the SLURM jobs first (sbatch submit_peak_fit.sh)."
            )

        nx, ny = self.volume.shape[1], self.volume.shape[2]
        maps: dict[str, np.ndarray] = {}

        for chunk_file in tqdm(chunk_files, desc="Loading peak-fit chunks"):
            data = np.load(chunk_file)
            ii_arr = data["ii"].astype(int)
            jj_arr = data["jj"].astype(int)
            for key in data.files:
                if key in ("ii", "jj"):
                    continue
                if key not in maps:
                    maps[key] = np.full((nx, ny), np.nan, dtype=np.float32)
                maps[key][ii_arr, jj_arr] = data[key]

        if not plot:
            return maps

        # Re-use the same mosaic logic as fit_peak_map
        import matplotlib.cm as _cm
        import matplotlib.pyplot as plt

        label_map = {
            "center":    "Centre (°)",
            "amplitude": "Amplitude",
            "fwhm":      "FWHM (°)",
            "area":      "Area",
            "sigma":     "σ (°)",
            "gamma":     "γ (°)",
            "eta":       "η  (Lorentzian fraction)",
            "residual":  "RMS residual",
            "success":   "Success",
        }
        order = ["center", "amplitude", "fwhm", "area",
                 "sigma", "gamma", "eta", "residual", "success"]
        keys = [k for k in order if k in maps] + \
               [k for k in maps if k not in order]

        n = len(keys)
        ncols = 3
        nrows = -(-n // ncols)

        if figsize is None:
            figsize = (ncols * 4, nrows * 4)

        cmap_obj = _cm.get_cmap(cmap).copy()
        cmap_obj.set_bad(color="#aaaaaa")

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        axes_flat = np.array(axes).flatten()

        for idx, key in enumerate(keys):
            ax = axes_flat[idx]
            data = maps[key].astype(float)
            if self.mask is not None:
                data[self.mask == 0] = np.nan

            finite = data[np.isfinite(data)]
            vmin, vmax = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)

            im = ax.imshow(data, cmap=cmap_obj, origin="lower", vmin=vmin, vmax=vmax)
            ax.set_title(label_map.get(key, key))
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        for ax in axes_flat[n:]:
            ax.set_visible(False)

        fig.suptitle(f"{self.name}  —  peak fit (SLURM results)", fontsize=13)
        fig.tight_layout()

        return (maps, fig) if return_fig else maps


def _get_chi2_ii_jj(args: tuple) -> float:
    """Return reduced chi-squared for voxel ``(ii, jj)``.

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
        return float(hists[0].data["Residuals"]["chisq"])
    except Exception:
        return np.nan


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
    """Return crystallite size for voxel ``(ii, jj)``, phase ``phase_idx``.

    Returns ``0.0`` if the .xy file does not exist (masked-out pixel),
    ``nan`` if the .xy file exists but the parameter could not be extracted.
    """
    ii, jj, folder, name, phase_idx = args
    xy_filename = folder / "xy_files" / f"{name}_{ii:04}_{jj:04}.xy"
    if not xy_filename.exists():
        return 0.0
    try:
        gpx_filename = folder / "gpx_files" / f"{name}_{ii:04}_{jj:04}.gpx"
        g, hists, phases = _load_data_from_gpx(gpx_filename)
        hap = phases[phase_idx].data["Histograms"][hists[0].name]
        sz = hap["Size"]["Size"]
        return float(sz) if np.isscalar(sz) else float(sz[0])
    except Exception:
        return np.nan


def _get_microstrain_ii_jj(args: tuple) -> float:
    """Return isotropic microstrain for voxel ``(ii, jj)``, phase ``phase_idx``.

    Returns ``0.0`` if the .xy file does not exist (masked-out pixel),
    ``nan`` if the .xy file exists but the parameter could not be extracted.
    """
    ii, jj, folder, name, phase_idx = args
    xy_filename = folder / "xy_files" / f"{name}_{ii:04}_{jj:04}.xy"
    if not xy_filename.exists():
        return 0.0
    try:
        gpx_filename = folder / "gpx_files" / f"{name}_{ii:04}_{jj:04}.gpx"
        g, hists, phases = _load_data_from_gpx(gpx_filename)
        hap = phases[phase_idx].data["Histograms"][hists[0].name]
        mustrain = hap["Mustrain"]["Mustrain"]
        return float(mustrain) if np.isscalar(mustrain) else float(mustrain[0])
    except Exception:
        return np.nan


def _get_scale_ii_jj(args: tuple) -> float:
    """Return HAP scale factor for voxel ``(ii, jj)``, phase ``phase_idx``.

    Returns ``0.0`` if the .xy file does not exist (masked-out pixel),
    ``nan`` if the .xy file exists but the parameter could not be extracted.
    """
    ii, jj, folder, name, phase_idx = args
    xy_filename = folder / "xy_files" / f"{name}_{ii:04}_{jj:04}.xy"
    if not xy_filename.exists():
        return 0.0
    try:
        gpx_filename = folder / "gpx_files" / f"{name}_{ii:04}_{jj:04}.gpx"
        g, hists, phases = _load_data_from_gpx(gpx_filename)
        hap = phases[phase_idx].data["Histograms"][hists[0].name]
        return float(hap["Scale"][0])
    except Exception:
        return np.nan


def _get_all_maps_ii_jj(args: tuple) -> Tuple[float, ...]:
    """Return ``(rwp, a, b, c, size, mustrain, scale)`` for voxel ``(ii, jj)``.

    Returns all-zeros if the .xy file does not exist (masked-out pixel).
    Each parameter is extracted independently so a missing/unrefined parameter
    returns ``nan`` for that slot without affecting the others.
    """
    ii, jj, folder, name, phase_idx = args
    xy_filename = folder / "xy_files" / f"{name}_{ii:04}_{jj:04}.xy"
    if not xy_filename.exists():
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    try:
        gpx_filename = folder / "gpx_files" / f"{name}_{ii:04}_{jj:04}.gpx"
        g, hists, phases = _load_data_from_gpx(gpx_filename)
    except Exception:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    try:
        wR = hists[0].get_wR()
    except Exception:
        wR = np.nan

    try:
        chi2 = float(hists[0].data["Residuals"]["chisq"])
    except Exception:
        chi2 = np.nan

    try:
        cell = phases[phase_idx].get_cell()
        a, b, c = cell["length_a"], cell["length_b"], cell["length_c"]
    except Exception:
        a = b = c = np.nan

    try:
        hap = phases[phase_idx].data["Histograms"][hists[0].name]
    except Exception:
        return wR, chi2, a, b, c, np.nan, np.nan, np.nan

    try:
        sz = hap["Size"]["Size"]
        sz = float(sz) if np.isscalar(sz) else float(sz[0])
    except Exception:
        sz = np.nan

    try:
        mustrain = hap["Mustrain"]["Mustrain"]
        mustrain = float(mustrain) if np.isscalar(mustrain) else float(mustrain[0])
    except Exception:
        mustrain = np.nan

    try:
        scale = float(hap["Scale"][0])
    except Exception:
        scale = np.nan

    return wR, chi2, a, b, c, sz, mustrain, scale


def _get_cell_params_ii_jj(args: tuple) -> Tuple[float, float, float]:
    """Return ``(a, b, c)`` unit-cell lengths for voxel ``(ii, jj)``, phase ``phase_idx``.

    Returns ``(0., 0., 0.)`` if the .xy file does not exist (masked-out pixel),
    ``(nan, nan, nan)`` if the .xy file exists but parameters could not be extracted.
    """
    ii, jj, folder, name, phase_idx = args
    xy_filename = folder / "xy_files" / f"{name}_{ii:04}_{jj:04}.xy"
    if not xy_filename.exists():
        return 0.0, 0.0, 0.0
    try:
        gpx_filename = folder / "gpx_files" / f"{name}_{ii:04}_{jj:04}.gpx"
        g, hists, phases = _load_data_from_gpx(gpx_filename)
        cell = phases[phase_idx].get_cell()
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
