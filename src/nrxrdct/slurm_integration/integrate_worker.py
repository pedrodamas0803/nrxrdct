"""
nrxrdct.slurm_integration.integrate_worker
-------------------------------------------
Memory-efficient worker executed inside each SLURM job.

Memory strategies applied
--------------------------
1. **Streaming** — frames are never all loaded at once. Only lightweight
   metadata (fpico6, rot, shape) is read upfront; raw images are streamed
   from the HDF5 dataset in batches.

2. **Batching** — frames are processed in chunks of `--batch-size` B.
   The sinogram is built row-by-row. Peak RAM ≈ B × frame_bytes × 2.

3. **Auto thread scaling** — if `--n-workers` is omitted, the worker
   estimates a safe thread count from available system memory and frame size,
   capped at the number of logical CPUs allocated by SLURM.

Invoked by launch_jobs.py via:
    python -m nrxrdct.slurm_integration.integrate_worker <args>
"""

from __future__ import annotations

import argparse
import fcntl
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import fabio
import h5py
import numpy as np
from tqdm import tqdm

from nrxrdct.integration import (
    azimuthal_integration_1d,
    azimuthal_integration_1d_filter,
    azimuthal_integration_1d_sigma_clip,
)
from nrxrdct.utils import calculate_xrd_baseline

# Valid method names — checked at startup so failures are obvious
INTEGRATION_METHODS = ("standard", "filter", "sigma_clip")


# ─────────────────────────────────────────────────────────────────────────────
# Memory helpers
# ─────────────────────────────────────────────────────────────────────────────


def _available_ram_bytes() -> int:
    """Read MemAvailable from /proc/meminfo (Linux). Falls back to 32 GB."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024  # kB → bytes
    except Exception:
        pass
    return 32 * 1024**3


def _safe_n_workers(
    frame_shape: tuple[int, int],
    batch_size: int,
    mem_fraction: float = 0.6,
    requested: int | None = None,
) -> int:
    """
    Return a thread count that keeps peak batch RAM under `mem_fraction`
    of available memory.

    Peak RAM estimate = batch_size × n_workers × frame_bytes × 2
    (factor 2: float32 input + integration output buffer)

    If `requested` is given, trust it and skip the estimation.
    """
    if requested is not None:
        return requested

    frame_bytes = int(np.prod(frame_shape)) * 4  # float32
    budget_bytes = _available_ram_bytes() * mem_fraction
    # Frames that fit in budget simultaneously (each worker holds ~1 frame)
    max_frames = max(1, int(budget_bytes / (frame_bytes * 2)))
    n_cpus = os.cpu_count() or 16
    n = max(1, min(max_frames, batch_size, n_cpus))
    print(
        f"  Auto thread count: {n}  "
        f"(frame={frame_bytes/1e6:.1f} MB, "
        f"budget={budget_bytes/1e9:.1f} GB, "
        f"batch={batch_size}, cpus={n_cpus})"
    )
    return n


# ─────────────────────────────────────────────────────────────────────────────
# Frame-level integration
# ─────────────────────────────────────────────────────────────────────────────


def _integrate_frame(
    jj: int,
    image: np.ndarray,
    monitor: float,
    *,
    poni_file: Path,
    n_points: int,
    mask: np.ndarray,
    unit: str,
    method: str,
    percentile: tuple,  # used by "filter"
    thres: float,  # used by "sigma_clip"
    max_iter: int,  # used by "sigma_clip"
) -> tuple[int, np.ndarray]:
    """Dispatch to the requested integration method and normalise by monitor."""
    if method == "filter":
        tt, itt, _ = azimuthal_integration_1d_filter(
            image=image,
            poni_file=str(poni_file),
            npt=n_points,
            mask=mask,
            unit=unit,
            percentile=percentile,
        )
    elif method == "sigma_clip":
        tt, itt, _ = azimuthal_integration_1d_sigma_clip(
            image=image,
            poni_file=str(poni_file),
            npt=n_points,
            mask=mask,
            unit=unit,
            thres=thres,
            max_iter=max_iter,
        )
    else:  # "standard"
        tt, itt, _ = azimuthal_integration_1d(
            image=image,
            poni_file=str(poni_file),
            npt=n_points,
            mask=mask,
            unit=unit,
        )

    if monitor <= 0:
        print(f"  ⚠  Frame {jj}: fpico6={monitor:.4g}, skipping normalisation")
        return jj, itt

    # bkg, _ = calculate_xrd_baseline(itt, tt)
    # itt -= bkg
    return jj, itt / monitor


# ─────────────────────────────────────────────────────────────────────────────
# HDF5 write with POSIX advisory lock
# ─────────────────────────────────────────────────────────────────────────────


def _write_scan(
    *,
    output_file: Path,
    group_path: str,
    sinogram: np.ndarray,
    entry: str,
    dty_value: float,
    master_file: Path,
    fpico6: np.ndarray,
    method: str,
) -> None:
    """
    Write a completed sinogram to the output HDF5 file under an exclusive POSIX lock.

    Uses ``fcntl.flock`` to serialise concurrent writes from multiple threads.
    Silently skips if the dataset already exists (idempotent).  Stores
    provenance attributes (source file, dty position, monitor statistics,
    integration method) alongside the sinogram data.

    Parameters
    ----------
    output_file : Path
        Destination HDF5 file.
    group_path : str
        HDF5 dataset path, e.g. ``"integrated/scan_0042"``.
    sinogram : np.ndarray
        2-D array of shape ``(n_frames, n_points)`` to store.
    entry : str
        Original master-file entry key stored as an attribute.
    dty_value : float
        Translation motor position stored as an attribute.
    master_file : Path
        Path to the source master file stored as an attribute.
    fpico6 : np.ndarray
        Monitor values; mean/min/max stored as attributes.
    method : str
        Integration method name stored as an attribute.
    """
    lock_path = str(output_file) + ".lock"
    with open(lock_path, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            with h5py.File(output_file, "a") as hout:
                # Use try/except — the plain `in` operator crashes on
                # corrupted B-tree/cache entries ("bad symbol table node").
                try:
                    already_exists = group_path in hout
                except (RuntimeError, OSError):
                    already_exists = False  # corrupted entry: overwrite it
                if already_exists:
                    return
                ds = hout.create_dataset(
                    group_path,
                    data=sinogram,
                    compression="gzip",
                    compression_opts=4,
                    chunks=(1, sinogram.shape[1]),
                )
                ds.attrs["entry"] = entry
                ds.attrs["dty"] = dty_value
                ds.attrs["source"] = str(master_file)
                ds.attrs["fpico6_mean"] = float(np.mean(fpico6))
                ds.attrs["fpico6_min"] = float(np.min(fpico6))
                ds.attrs["fpico6_max"] = float(np.max(fpico6))
                ds.attrs["normalised_by"] = "fpico6"
                ds.attrs["integration_method"] = method
                ds.attrs["valid"] = True
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)


# ─────────────────────────────────────────────────────────────────────────────
# Scan-level processing  (streaming + batched)
# ─────────────────────────────────────────────────────────────────────────────


def _process_scan(
    ii: int,
    entry: str,
    dty_value: float,
    *,
    master_file: Path,
    output_file: Path,
    poni_file: Path,
    mask: np.ndarray,
    n_points: int,
    n_workers: int | None,
    batch_size: int,
    unit: str,
    method: str,
    percentile: tuple,
    thres: float,
    max_iter: int,
) -> bool:
    """
    Integrate and write a single scan entry, streaming frames in batches.

    Skips the scan silently if it is already present in the output file.
    Reads only lightweight metadata (monitor values, rotation, frame count)
    upfront; raw detector images are streamed from HDF5 in chunks of
    *batch_size* to keep peak RAM bounded.  Frames are integrated in parallel
    within each batch using a :class:`~concurrent.futures.ThreadPoolExecutor`.

    Parameters
    ----------
    ii : int
        Global scan index (used to name the output dataset ``scan_<ii:04d>``).
    entry : str
        HDF5 entry key in the master file (e.g. ``"1.1"``).
    dty_value : float
        Translation motor position for this scan.
    master_file : Path
        Source HDF5 master file.
    output_file : Path
        Destination HDF5 output file.
    poni_file : Path
        pyFAI ``.poni`` calibration file.
    mask : np.ndarray
        Detector mask array (1 = masked).
    n_points : int
        Number of radial bins in the integrated pattern.
    n_workers : int or None
        Integration threads; ``None`` triggers auto-scaling from available RAM.
    batch_size : int
        Number of frames loaded from HDF5 per iteration.
    unit : str
        Radial unit for integration (e.g. ``"2th_deg"``).
    method : str
        Integration method: ``"standard"``, ``"filter"``, or ``"sigma_clip"``.
    percentile : tuple of (float, float)
        Percentile bounds used when *method* is ``"filter"``.
    thres : float
        Sigma threshold used when *method* is ``"sigma_clip"``.
    max_iter : int
        Maximum sigma-clipping iterations.

    Returns
    -------
    bool
        ``True`` if the scan was processed (or already done), ``False`` on
        unrecoverable read failure.
    """
    scan_name = f"scan_{ii:04d}"
    group_path = f"integrated/{scan_name}"

    with h5py.File(output_file, "r") as hout:
        try:
            already_done = group_path in hout
        except (RuntimeError, OSError):
            already_done = False  # corrupted entry: reprocess it
        if already_done:
            print(f"  → Skipping {scan_name} (already done)")
            return True

    print(f"\n{'='*60}\n{scan_name} — entry {entry}  [global idx {ii}]\n{'='*60}")

    # ── 1. Read only lightweight metadata — NOT the images ────────────────────
    try:
        with h5py.File(master_file, "r") as hin:
            fpico6 = hin[f"{entry}/measurement/fpico6"][:].astype(np.float64)
            rot = hin[f"{entry}/measurement/rot"][:]
            n_frames = hin[f"{entry}/measurement/eiger"].shape[0]
            frame_shape = hin[f"{entry}/measurement/eiger"].shape[1:]  # (H, W)
    except OSError as e:
        print(f"  ✗ Failed to read metadata for {entry}: {e} — skipping")
        return False

    if len(fpico6) != n_frames:
        print(
            f"  ✗ Length mismatch: fpico6={len(fpico6)}, frames={n_frames} — skipping"
        )
        return False

    # Sort order (ascending rotation)
    descending = rot[-1] < rot[0]
    frame_order = (
        list(range(n_frames - 1, -1, -1)) if descending else list(range(n_frames))
    )
    fpico6 = fpico6[frame_order]

    # ── 2. Decide thread count using frame size ───────────────────────────────
    workers = _safe_n_workers(
        frame_shape=frame_shape,
        batch_size=batch_size,
        requested=n_workers,
    )

    sinogram = np.empty((n_frames, n_points), dtype=np.float32)
    n_batches = max(1, (n_frames + batch_size - 1) // batch_size)

    # ── 3. Process frame-by-frame in streaming batches ────────────────────────
    with tqdm(total=n_frames, desc=scan_name) as pbar:
        for b in range(n_batches):
            batch_start = b * batch_size
            batch_end = min(batch_start + batch_size, n_frames)
            # Indices into the original (unsorted) HDF5 dataset
            batch_h5_idx = frame_order[batch_start:batch_end]

            # Read the tightest contiguous slice to minimise HDF5 overhead,
            # then select only the frames we actually want from that slice.
            try:
                with h5py.File(master_file, "r") as hin:
                    ds = hin[f"{entry}/measurement/eiger"]
                    lo = min(batch_h5_idx)
                    hi = max(batch_h5_idx) + 1
                    raw = ds[lo:hi].astype(np.float32)  # (≤B, H, W)
                    batch_images = raw[[j - lo for j in batch_h5_idx]]
            except OSError as e:
                print(
                    f"  ✗ Batch {b}: failed to read frames {batch_start}–{batch_end}: {e}"
                )
                sinogram[batch_start:batch_end] = np.nan
                pbar.update(batch_end - batch_start)
                continue

            batch_fpico6 = fpico6[batch_start:batch_end]

            # Integrate this batch in parallel
            def _task(args):
                local_jj, image, monitor = args
                return _integrate_frame(
                    local_jj,
                    image,
                    monitor,
                    poni_file=poni_file,
                    n_points=n_points,
                    mask=mask,
                    unit=unit,
                    method=method,
                    percentile=percentile,
                    thres=thres,
                    max_iter=max_iter,
                )

            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(
                        _task,
                        (batch_start + k, batch_images[k], batch_fpico6[k]),
                    ): k
                    for k in range(len(batch_images))
                }
                for future in as_completed(futures):
                    try:
                        jj, itt = future.result()
                        sinogram[jj] = itt / itt.max()
                    except Exception as e:
                        k = futures[future]
                        jj = batch_start + k
                        print(f"  ✗ Frame {jj} failed: {e}")
                        sinogram[jj] = np.nan
                    pbar.update(1)

            # ── Explicitly release this batch before the next HDF5 read ───────
            del batch_images, raw

    _write_scan(
        output_file=output_file,
        group_path=group_path,
        sinogram=sinogram,
        entry=entry,
        dty_value=dty_value,
        master_file=master_file,
        fpico6=fpico6,
        method=method,
    )
    return True


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def _parse_args():
    """Parse command-line arguments for the integrate worker."""
    p = argparse.ArgumentParser(
        description="nrxrdct powder integration worker (one SLURM job)"
    )
    p.add_argument("--master-file", required=True, type=Path)
    p.add_argument("--output-file", required=True, type=Path)
    p.add_argument("--poni-file", required=True, type=Path)
    p.add_argument("--mask-file", required=True, type=Path)
    p.add_argument(
        "--entry-indices",
        required=True,
        help="Comma-separated global scan indices for this job",
    )
    p.add_argument("--n-points", type=int, default=1000)
    p.add_argument(
        "--n-workers",
        type=int,
        default=None,
        help="Integration threads. Omit to auto-scale from available RAM.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Frames streamed from HDF5 per batch (default: 32)",
    )
    p.add_argument("--unit", default="2th_deg")
    # ── Integration method ────────────────────────────────────────────────────
    p.add_argument(
        "--method",
        default="standard",
        choices=INTEGRATION_METHODS,
        help=(
            "Integration method: "
            "'standard' – plain azimuthal average; "
            "'filter'   – percentile-based pixel rejection; "
            "'sigma_clip' – iterative sigma-clipping. "
            "(default: standard)"
        ),
    )
    p.add_argument(
        "--percentile",
        default="10,90",
        help="Low,high percentile for 'filter' method (default: 10,90)",
    )
    p.add_argument(
        "--thres",
        type=float,
        default=3.0,
        help="Sigma threshold for 'sigma_clip' method (default: 3.0)",
    )
    p.add_argument(
        "--max-iter",
        type=int,
        default=5,
        help="Max iterations for 'sigma_clip' method (default: 5)",
    )
    return p.parse_args()


def main():
    """
    Entry point for the integrate worker process launched inside each SLURM job.

    Reads the assigned scan indices from ``--entry-indices``, loads the mask,
    resolves the valid entry list from the output HDF5 file, and calls
    :func:`_process_scan` for each index.  Exits with code 1 if any scan
    failed, 0 otherwise.
    """
    args = _parse_args()
    entry_indices = [int(x) for x in args.entry_indices.split(",")]
    percentile = tuple(int(x) for x in args.percentile.split(","))

    print(
        f"Worker started — {len(entry_indices)} scans | "
        f"method={args.method} | "
        f"batch={args.batch_size} | "
        f"threads={'auto' if args.n_workers is None else args.n_workers}"
    )

    mask = fabio.open(args.mask_file).data

    with h5py.File(args.output_file, "r") as hout:
        valid_entries = [
            e.decode() if isinstance(e, bytes) else e
            for e in hout["meta/valid_entries"][:]
        ]
        dty_values = hout["motors/dty"][:]

    t0 = time.time()
    n_ok = n_fail = 0

    for ii in entry_indices:
        ok = _process_scan(
            ii,
            valid_entries[ii],
            dty_values[ii],
            master_file=args.master_file,
            output_file=args.output_file,
            poni_file=args.poni_file,
            mask=mask,
            n_points=args.n_points,
            n_workers=args.n_workers,
            batch_size=args.batch_size,
            unit=args.unit,
            method=args.method,
            percentile=percentile,
            thres=args.thres,
            max_iter=args.max_iter,
        )
        n_ok += ok
        n_fail += not ok

    elapsed = time.time() - t0
    print(f"\nWorker done in {elapsed:.1f}s — {n_ok} OK, {n_fail} failed")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
