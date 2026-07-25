"""
nrxrdct.texture.slurm_pole_figures.pole_figure_worker
--------------------------------------------------------
Worker executed inside each SLURM job.

Each scan is cake-integrated frame-by-frame (streaming + batching to control
RAM) and reduced to an azimuthal (chi) intensity profile via
:func:`nrxrdct.texture.odf.extract_ring_intensity`. The result is written as
two files in a shared tmp directory:

    <tmp_dir>/scan_XXXX.npy        — profiles, shape (n_frames, npt_azim)
    <tmp_dir>/scan_XXXX.meta.json  — scan attributes (dty, monitor stats, …)

No HDF5 file is touched during extraction, eliminating all concurrent-write
corruption issues. The final HDF5 is assembled by merge.py after all jobs
finish.

Invoked by launch_jobs.py via:
    python -m nrxrdct.texture.slurm_pole_figures.pole_figure_worker <args>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Tuple

import fabio
import h5py
import numpy as np
from tqdm import tqdm

from nrxrdct.azimuthal.integration import cake_integration
from nrxrdct.texture.odf import extract_ring_intensity


# ─────────────────────────────────────────────────────────────────────────────
# Memory helpers
# ─────────────────────────────────────────────────────────────────────────────

def _available_ram_bytes() -> int:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except Exception:
        pass
    return 32 * 1024 ** 3


def _safe_n_workers(
    frame_shape: tuple[int, int],
    batch_size: int,
    mem_fraction: float = 0.6,
    requested: Optional[int] = None,
) -> int:
    if requested is not None:
        return requested
    frame_bytes = int(np.prod(frame_shape)) * 4
    budget_bytes = _available_ram_bytes() * mem_fraction
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
# Frame-level extraction
# ─────────────────────────────────────────────────────────────────────────────

def _extract_frame(
    jj: int,
    image: np.ndarray,
    monitor: float,
    *,
    poni_file: Path,
    npt_rad: int,
    npt_azim: int,
    mask: np.ndarray,
    tth_range: Tuple[float, float],
    background_ranges,
) -> tuple[int, np.ndarray]:
    cake, radial, azimuthal = cake_integration(
        image, str(poni_file), npt_rad=npt_rad, npt_azim=npt_azim, mask=mask
    )
    _, intensity = extract_ring_intensity(cake, radial, azimuthal, tth_range, background_ranges)
    if monitor <= 0:
        print(f"  ⚠  Frame {jj}: monitor={monitor:.4g}, skipping normalisation")
        return jj, intensity
    return jj, intensity / monitor


# ─────────────────────────────────────────────────────────────────────────────
# Scan-level processing
# ─────────────────────────────────────────────────────────────────────────────

def _process_scan(
    ii: int,
    entry: str,
    dty_value: float,
    *,
    master_file: Path,
    tmp_dir: Path,
    poni_file: Path,
    mask: np.ndarray,
    tth_range: Tuple[float, float],
    background_ranges,
    npt_rad: int,
    npt_azim: int,
    n_workers: Optional[int],
    batch_size: int,
    camera_name: str,
    monitor_name: str,
    rotation_motor: str,
) -> bool:
    scan_name = f"scan_{ii:04d}"
    npy_path = tmp_dir / f"{scan_name}.npy"
    meta_path = tmp_dir / f"{scan_name}.meta.json"

    if npy_path.exists() and meta_path.exists():
        print(f"  → Skipping {scan_name} (already in tmp)")
        return True

    print(f"\n{'='*60}\n{scan_name} — entry {entry}  [global idx {ii}]\n{'='*60}")

    try:
        with h5py.File(master_file, "r") as hin:
            monitor = hin[f"{entry}/measurement/{monitor_name}"][:].astype(np.float64)
            rot = hin[f"{entry}/measurement/{rotation_motor}"][:]
            n_frames = hin[f"{entry}/measurement/{camera_name}"].shape[0]
            frame_shape = hin[f"{entry}/measurement/{camera_name}"].shape[1:]
    except (OSError, KeyError) as e:
        print(f"  ✗ Failed to read metadata for {entry}: {e} — skipping")
        return False

    if len(monitor) != n_frames:
        print(f"  ✗ Length mismatch: monitor={len(monitor)}, frames={n_frames} — skipping")
        return False

    descending = rot[-1] < rot[0]
    frame_order = list(range(n_frames - 1, -1, -1)) if descending else list(range(n_frames))
    monitor = monitor[frame_order]
    rot_sorted = np.asarray(rot)[frame_order]

    workers = _safe_n_workers(frame_shape=frame_shape, batch_size=batch_size, requested=n_workers)
    profiles = np.empty((n_frames, npt_azim), dtype=np.float32)
    n_batches = max(1, (n_frames + batch_size - 1) // batch_size)

    with tqdm(total=n_frames, desc=scan_name) as pbar:
        for b in range(n_batches):
            batch_start = b * batch_size
            batch_end = min(batch_start + batch_size, n_frames)
            batch_h5_idx = frame_order[batch_start:batch_end]

            try:
                with h5py.File(master_file, "r") as hin:
                    ds = hin[f"{entry}/measurement/{camera_name}"]
                    lo = min(batch_h5_idx)
                    hi = max(batch_h5_idx) + 1
                    raw = ds[lo:hi].astype(np.float32)
                    batch_images = raw[[j - lo for j in batch_h5_idx]]
            except (OSError, KeyError) as e:
                print(f"  ✗ Batch {b}: {e}")
                profiles[batch_start:batch_end] = np.nan
                pbar.update(batch_end - batch_start)
                continue

            batch_monitor = monitor[batch_start:batch_end]

            def _task(args):
                local_jj, image, mon = args
                return _extract_frame(
                    local_jj, image, mon,
                    poni_file=poni_file, npt_rad=npt_rad, npt_azim=npt_azim,
                    mask=mask, tth_range=tth_range, background_ranges=background_ranges,
                )

            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(_task, (batch_start + k, batch_images[k], batch_monitor[k])): k
                    for k in range(len(batch_images))
                }
                for future in as_completed(futures):
                    try:
                        jj, intensity = future.result()
                        profiles[jj] = intensity
                    except Exception as e:
                        jj = batch_start + futures[future]
                        print(f"  ✗ Frame {jj} failed: {e}")
                        profiles[jj] = np.nan
                    pbar.update(1)

            del batch_images, raw

    # ── Write tmp files atomically ────────────────────────────────────────────
    npy_tmp_stem = tmp_dir / f"{scan_name}.tmp"   # np.save → scan_XXXX.tmp.npy
    meta_tmp = tmp_dir / f"{scan_name}.meta.json.tmp"

    np.save(npy_tmp_stem, profiles)
    npy_tmp = tmp_dir / f"{scan_name}.tmp.npy"

    meta = {
        "scan_index": ii,
        "scan_name": scan_name,
        "entry": entry,
        "dty": dty_value,
        "rot": rot_sorted.tolist(),
        "monitor_mean": float(np.nanmean(monitor)),
        "monitor_min": float(np.nanmin(monitor)),
        "monitor_max": float(np.nanmax(monitor)),
        "normalised_by": monitor_name,
        "valid": True,
    }
    meta_tmp.write_text(json.dumps(meta, indent=2))

    npy_tmp.rename(npy_path)
    meta_tmp.rename(meta_path)

    print(f"  ✓ {scan_name} → {npy_path.name}")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_tth_range(s: str) -> Tuple[float, float]:
    lo, hi = (float(x) for x in s.split(","))
    return (lo, hi)


def _parse_background_ranges(s: str):
    if not s:
        return None
    lo1, hi1, lo2, hi2 = (float(x) for x in s.split(","))
    return ((lo1, hi1), (lo2, hi2))


def _parse_args():
    p = argparse.ArgumentParser(
        description="nrxrdct pole-figure extraction worker (one SLURM job)"
    )
    p.add_argument("--master-file", required=True, type=Path)
    p.add_argument("--tmp-dir", required=True, type=Path,
                   help="Shared tmp directory for .npy and .meta.json outputs")
    p.add_argument("--poni-file", required=True, type=Path)
    p.add_argument("--mask-file", required=True, type=Path)
    p.add_argument("--entry-indices", required=True)
    p.add_argument("--tth-range", required=True, help="low,high 2theta window")
    p.add_argument("--hkl-label", required=True)
    p.add_argument("--background-ranges", default="")
    p.add_argument("--npt-rad", type=int, default=1000)
    p.add_argument("--npt-azim", type=int, default=360)
    p.add_argument("--n-workers", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--camera-name", default="eiger")
    p.add_argument("--monitor-name", default="fpico6")
    p.add_argument("--rotation-motor", default="rot")
    return p.parse_args()


def main():
    args = _parse_args()
    entry_indices = [int(x) for x in args.entry_indices.split(",")]
    tth_range = _parse_tth_range(args.tth_range)
    background_ranges = _parse_background_ranges(args.background_ranges)

    print(
        f"Worker started — {len(entry_indices)} scans | hkl={args.hkl_label} | "
        f"tth_range={tth_range} | batch={args.batch_size} | "
        f"threads={'auto' if args.n_workers is None else args.n_workers}"
    )

    meta_sidecar = args.tmp_dir / "launch_meta.json"
    with open(meta_sidecar) as f:
        launch_meta = json.load(f)
    valid_entries = launch_meta["valid_entries"]
    dty_values = launch_meta["dty_values"]

    mask = fabio.open(args.mask_file).data
    args.tmp_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    n_ok = n_fail = 0

    for ii in entry_indices:
        ok = _process_scan(
            ii, valid_entries[ii], dty_values[ii],
            master_file=args.master_file,
            tmp_dir=args.tmp_dir,
            poni_file=args.poni_file,
            mask=mask,
            tth_range=tth_range,
            background_ranges=background_ranges,
            npt_rad=args.npt_rad,
            npt_azim=args.npt_azim,
            n_workers=args.n_workers,
            batch_size=args.batch_size,
            camera_name=args.camera_name,
            monitor_name=args.monitor_name,
            rotation_motor=args.rotation_motor,
        )
        n_ok += ok
        n_fail += not ok

    elapsed = time.time() - t0
    print(f"\nWorker done in {elapsed:.1f}s — {n_ok} OK, {n_fail} failed")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()