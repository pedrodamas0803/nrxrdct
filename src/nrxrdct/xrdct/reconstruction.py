"""
Tomographic reconstruction routines for XRD-CT data.

Provides GPU- and CPU-backed ASTRA Toolbox reconstruction routines and a
helper to assemble sinograms from integrated HDF5 files.

:class:`ReconstructedVolume` has moved to :mod:`nrxrdct.volume` and is
re-exported here for backwards compatibility.
"""

import os

import numpy as np

from .io import save_sinogram  # noqa: F401 — re-exported for backwards compat

try:
    import astra  # type: ignore
    HAS_GPU = "nvidia" in astra.get_gpu_info().lower()
    _ASTRA_AVAILABLE = True
except ImportError:
    astra = None  # type: ignore[assignment]
    HAS_GPU = False
    _ASTRA_AVAILABLE = False

NTHREADS = os.cpu_count() - 2


def _require_astra() -> None:
    if not _ASTRA_AVAILABLE:
        raise ImportError(
            "astra-toolbox is required for tomographic reconstruction. "
            "Install it separately — see https://www.astra-toolbox.com/."
        )


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
    _require_astra()
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
    _require_astra()
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
    _require_astra()
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
    _require_astra()
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
    _require_astra()
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



