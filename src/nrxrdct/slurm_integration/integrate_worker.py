"""
nrxrdct.slurm_integration.integrate_worker
-------------------------------------------
Worker executed inside each SLURM job.  Processes the scan indices assigned
to this job and writes sinograms into the shared output HDF5 file.

Invoked by launch_jobs.py via:
    python -m nrxrdct.slurm_integration.integrate_worker <args>

Not meant to be called directly by users — use `launch` instead.
"""

from __future__ import annotations

import argparse
import fcntl
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np
import fabio
from tqdm import tqdm

from nrxrdct.integration import (
    azimuthal_integration_1d,
    azimuthal_integration_1d_filter,
)


# ─────────────────────────────────────────────────────────────────────────────
# Frame-level integration
# ─────────────────────────────────────────────────────────────────────────────

def _integrate_frame(
    args: tuple,
    *,
    poni_file: Path,
    n_points: int,
    mask: np.ndarray,
    unit: str,
    remove_spots: bool,
    percentile: tuple,
) -> tuple[int, np.ndarray]:
    jj, image, monitor = args
    if remove_spots:
        _, itt, _ = azimuthal_integration_1d_filter(
            image=image, poni_file=poni_file, npt=n_points,
            mask=mask, unit=unit, percentile=percentile,
        )
    else:
        _, itt, _ = azimuthal_integration_1d(
            image=image, poni_file=poni_file, npt=n_points,
            mask=mask, unit=unit,
        )
    if monitor <= 0:
        print(f"  ⚠  Frame {jj}: fpico6={monitor:.4g}, skipping normalisation")
        return jj, itt
    return jj, itt / monitor


# ─────────────────────────────────────────────────────────────────────────────
# HDF5 write with lock-file guard
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
) -> None:
    """Write one sinogram; uses a POSIX advisory lock so concurrent jobs
    do not corrupt the shared HDF5 file."""
    lock_path = str(output_file) + ".lock"
    with open(lock_path, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            with h5py.File(output_file, "a") as hout:
                if group_path in hout:   # already written by another job
                    return
                ds = hout.create_dataset(
                    group_path,
                    data=sinogram,
                    compression="gzip",
                    compression_opts=4,
                    chunks=(1, sinogram.shape[1]),
                )
                ds.attrs["entry"]         = entry
                ds.attrs["dty"]           = dty_value
                ds.attrs["source"]        = str(master_file)
                ds.attrs["fpico6_mean"]   = float(np.mean(fpico6))
                ds.attrs["fpico6_min"]    = float(np.min(fpico6))
                ds.attrs["fpico6_max"]    = float(np.max(fpico6))
                ds.attrs["normalised_by"] = "fpico6"
                ds.attrs["valid"]         = True
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)


# ─────────────────────────────────────────────────────────────────────────────
# Scan-level processing
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
    n_workers: int,
    unit: str,
    remove_spots: bool,
    percentile: tuple,
) -> bool:
    scan_name  = f"scan_{ii:04d}"
    group_path = f"integrated/{scan_name}"

    # Skip if already present (resume after crash)
    with h5py.File(output_file, "r") as hout:
        if group_path in hout:
            print(f"  → Skipping {scan_name} (already in output)")
            return True

    print(f"\n{'='*60}\n{scan_name} — entry {entry}  [global idx {ii}]\n{'='*60}")

    try:
        with h5py.File(master_file, "r") as hin:
            images = hin[f"{entry}/measurement/eiger"][:].astype(np.float32)
            fpico6 = hin[f"{entry}/measurement/fpico6"][:].astype(np.float64)
            rot    = hin[f"{entry}/measurement/rot"][:]
    except OSError as e:
        print(f"  ✗ Failed to read {entry}: {e} — skipping")
        return False

    if rot[-1] < rot[0]:
        images = images[::-1]
        fpico6 = fpico6[::-1]
        rot    = rot[::-1]

    if len(fpico6) != len(images):
        print(f"  ✗ {entry}: length mismatch fpico6={len(fpico6)} images={len(images)} — skipping")
        return False

    n_frames = len(images)
    sinogram  = np.empty((n_frames, n_points), dtype=np.float32)

    def _task(args):
        return _integrate_frame(
            args,
            poni_file=poni_file, n_points=n_points, mask=mask,
            unit=unit, remove_spots=remove_spots, percentile=percentile,
        )

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(_task, (jj, images[jj], fpico6[jj])): jj
            for jj in range(n_frames)
        }
        for future in tqdm(as_completed(futures), total=n_frames, desc=scan_name):
            try:
                jj, itt = future.result()
                sinogram[jj] = itt / itt.max()
            except Exception as e:
                jj = futures[future]
                print(f"  ✗ Frame {jj} failed: {e}")
                sinogram[jj] = np.nan

    _write_scan(
        output_file = output_file,
        group_path  = group_path,
        sinogram    = sinogram,
        entry       = entry,
        dty_value   = dty_value,
        master_file = master_file,
        fpico6      = fpico6,
    )
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="nrxrdct powder integration worker (one SLURM job)")
    p.add_argument("--master-file",    required=True, type=Path)
    p.add_argument("--output-file",    required=True, type=Path)
    p.add_argument("--poni-file",      required=True, type=Path)
    p.add_argument("--mask-file",      required=True, type=Path)
    p.add_argument("--entry-indices",  required=True)
    p.add_argument("--n-points",       type=int, default=1000)
    p.add_argument("--n-workers",      type=int, default=16)
    p.add_argument("--unit",           default="2th_deg")
    p.add_argument("--remove-spots",   action="store_true")
    p.add_argument("--percentile",     default="10,90")
    return p.parse_args()


def main():
    args = parse_args() if False else _parse_args()   # keep one code path
    entry_indices = [int(x) for x in args.entry_indices.split(",")]
    percentile    = tuple(int(x) for x in args.percentile.split(","))

    print(f"Worker started — {len(entry_indices)} scans assigned: {entry_indices}")

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
            ii, valid_entries[ii], dty_values[ii],
            master_file  = args.master_file,
            output_file  = args.output_file,
            poni_file    = args.poni_file,
            mask         = mask,
            n_points     = args.n_points,
            n_workers    = args.n_workers,
            unit         = args.unit,
            remove_spots = args.remove_spots,
            percentile   = percentile,
        )
        n_ok += ok
        n_fail += not ok

    elapsed = time.time() - t0
    print(f"\nWorker done in {elapsed:.1f}s — {n_ok} OK, {n_fail} failed")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()