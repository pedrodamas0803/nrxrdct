"""
nrxrdct.slurm_reconstruction.reconstruct_worker
------------------------------------------------
Worker executed inside each SLURM reconstruction job.

For each assigned 2θ index the worker:
1. Reads ``sinogram[tth_idx]`` — shape ``(n_rot, n_lines)`` — from the
   sinogram HDF5 file (one chunk at a time, keeping RAM bounded).
2. Transposes to ``(n_lines, n_rot)`` as expected by :func:`reconstruct_slice`.
3. Reconstructs the 2-D slice with :func:`reconstruct_slice` (dispatches to
   GPU if available, CPU otherwise).
4. Writes the result under ``reconstruction/slice_<tth_idx:04d>`` with an
   exclusive POSIX lock to serialise concurrent writes from sibling jobs.

Invoked by launch_recon.py via:
    python -m nrxrdct.slurm_reconstruction.reconstruct_worker <args>
"""

from __future__ import annotations

import argparse
import fcntl
import sys
import time
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

from nrxrdct.reconstruction import reconstruct_slice

# Exposed for launch-side validation
RECONSTRUCTION_ALGOS_GPU = ("SART_CUDA", "SIRT_CUDA", "FBP_CUDA", "CGLS_CUDA")
RECONSTRUCTION_ALGOS_CPU = ("FBP", "SIRT", "SART", "CGLS")
RECONSTRUCTION_ALGOS = RECONSTRUCTION_ALGOS_GPU + RECONSTRUCTION_ALGOS_CPU


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────


def _read_sinogram_slice(
    sinogram_file: Path,
    tth_idx: int,
) -> np.ndarray:
    """
    Read one 2θ slice from the sinogram HDF5 file.

    Args:
        sinogram_file (Path): HDF5 file containing the ``sinogram`` dataset of shape
            ``(n_tth, n_rot, n_lines)``.
        tth_idx (int): Index along the 2θ axis to read.

    Returns:
        np.ndarray: Array of shape ``(n_lines, n_rot)`` ready for :func:`reconstruct_slice`.
    """
    with h5py.File(sinogram_file, "r") as hin:
        # sino shape: (n_tth, n_rot, n_lines)
        # We read one slice and transpose to (n_lines, n_rot)
        sino_slice = hin["sinogram"][tth_idx, :, :].astype(np.float32)
    return sino_slice.T  # (n_lines, n_rot)


def _write_slice(
    *,
    output_file: Path,
    tth_idx: int,
    reconstruction: np.ndarray,
    algo: str,
    num_iter: int,
    dty_step: float,
) -> None:
    """
    Write a reconstructed 2-D slice under an exclusive POSIX lock.

    The dataset is created at ``reconstruction/slice_<tth_idx:04d>`` with
    provenance attributes.  If the dataset already exists the write is
    silently skipped (idempotent).

    Args:
        output_file (Path): Destination HDF5 file.
        tth_idx (int): 2θ bin index used to name the dataset.
        reconstruction (np.ndarray): 2-D reconstructed image of shape ``(N, N)``.
        algo (str): ASTRA algorithm name stored as an attribute.
        num_iter (int): Number of iterations stored as an attribute.
        dty_step (float): Detector pixel spacing stored as an attribute.
    """
    dataset_path = f"reconstruction/slice_{tth_idx:04d}"
    lock_path    = str(output_file) + ".lock"

    with open(lock_path, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            with h5py.File(output_file, "a") as hout:
                try:
                    already_exists = dataset_path in hout
                except (RuntimeError, OSError):
                    already_exists = False
                if already_exists:
                    return
                ds = hout.create_dataset(
                    dataset_path,
                    data=reconstruction.astype(np.float32),
                    compression="gzip",
                    compression_opts=4,
                )
                ds.attrs["tth_idx"]  = tth_idx
                ds.attrs["algo"]     = algo
                ds.attrs["num_iter"] = num_iter
                ds.attrs["dty_step"] = dty_step
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)


# ─────────────────────────────────────────────────────────────────────────────
# Slice-level processing
# ─────────────────────────────────────────────────────────────────────────────


def _process_slice(
    tth_idx: int,
    *,
    sinogram_file: Path,
    output_file: Path,
    rot_rad: np.ndarray,
    algo: str,
    num_iter: int,
    dty_step: float,
) -> bool:
    """
    Read, reconstruct, and write one 2θ slice.

    Skips silently if ``reconstruction/slice_<tth_idx:04d>`` is already
    present in *output_file*.

    Args:
        tth_idx (int): 2θ bin index to reconstruct.
        sinogram_file (Path): HDF5 file containing the full sinogram.
        output_file (Path): Destination HDF5 file for reconstructed slices.
        rot_rad (np.ndarray): Rotation angles in radians, length ``n_rot``.
        algo (str): ASTRA reconstruction algorithm.
        num_iter (int): Number of iterations.
        dty_step (float): Detector pixel spacing.

    Returns:
        bool: ``True`` if the slice was processed (or already done), ``False`` on
            unrecoverable failure.
    """
    dataset_path = f"reconstruction/slice_{tth_idx:04d}"

    with h5py.File(output_file, "r") as hout:
        try:
            already_done = dataset_path in hout
        except (RuntimeError, OSError):
            already_done = False
    if already_done:
        return True

    try:
        sino = _read_sinogram_slice(sinogram_file, tth_idx)
    except Exception as e:
        print(f"  ✗ tth_idx={tth_idx}: failed to read sinogram slice: {e}")
        return False

    try:
        recon = reconstruct_slice(sino, dty_step, rot_rad, algo, num_iter)
    except Exception as e:
        print(f"  ✗ tth_idx={tth_idx}: reconstruction failed: {e}")
        return False

    _write_slice(
        output_file   = output_file,
        tth_idx       = tth_idx,
        reconstruction = recon,
        algo          = algo,
        num_iter      = num_iter,
        dty_step      = dty_step,
    )
    return True


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def _parse_args():
    p = argparse.ArgumentParser(
        description="nrxrdct reconstruction worker (one SLURM job)"
    )
    p.add_argument("--sinogram-file", required=True, type=Path,
                   help="HDF5 sinogram file (shape: n_tth, n_rot, n_lines)")
    p.add_argument("--output-file",   required=True, type=Path,
                   help="Destination HDF5 file for reconstructed slices")
    p.add_argument("--tth-indices",   required=True,
                   help="Comma-separated 2θ bin indices assigned to this job")
    p.add_argument("--algo",          default="SART_CUDA",
                   choices=RECONSTRUCTION_ALGOS,
                   help="ASTRA reconstruction algorithm (default: SART_CUDA)")
    p.add_argument("--num-iter",      type=int,   default=200,
                   help="Number of reconstruction iterations (default: 200)")
    p.add_argument("--dty-step",      type=float, default=1.0,
                   help="Detector pixel spacing (default: 1.0)")
    return p.parse_args()


def main():
    """
    Entry point for the reconstruction worker launched inside each SLURM job.

    Reads the assigned 2θ indices from ``--tth-indices``, loads the rotation
    axis from *output_file*, and calls :func:`_process_slice` for each index.
    Exits with code 1 if any slice failed, 0 otherwise.
    """
    args = _parse_args()
    tth_indices = [int(x) for x in args.tth_indices.split(",")]

    print(
        f"Worker started — {len(tth_indices)} 2\u03b8 slices | "
        f"algo={args.algo} | "
        f"num_iter={args.num_iter} | "
        f"dty_step={args.dty_step}"
    )

    with h5py.File(args.output_file, "r") as hout:
        rot_deg = hout["motors/rot"][:]

    rot_rad = np.deg2rad(rot_deg)

    t0     = time.time()
    n_ok   = 0
    n_fail = 0

    for tth_idx in tqdm(tth_indices, desc="2\u03b8 slices"):
        ok = _process_slice(
            tth_idx,
            sinogram_file = args.sinogram_file,
            output_file   = args.output_file,
            rot_rad       = rot_rad,
            algo          = args.algo,
            num_iter      = args.num_iter,
            dty_step      = args.dty_step,
        )
        n_ok   += ok
        n_fail += not ok

    elapsed = time.time() - t0
    print(f"\nWorker done in {elapsed:.1f}s — {n_ok} OK, {n_fail} failed")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
