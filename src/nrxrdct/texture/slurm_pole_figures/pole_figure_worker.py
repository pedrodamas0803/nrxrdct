"""
nrxrdct.texture.slurm_pole_figures.pole_figure_worker
--------------------------------------------------------
Worker executed inside each SLURM job.

Each scan is cake-integrated frame-by-frame (streaming + batching to control
RAM) **once per frame**, and every requested hkl ring's azimuthal (chi)
intensity profile is extracted from that same cake via
:func:`nrxrdct.texture.odf.extract_ring_intensity` — extracting several hkls
costs one pass over the raw frames, not one pass per hkl. Results are written
per scan in a shared tmp directory:

    <tmp_dir>/scan_XXXX_<hkl>.npy   — profiles for one hkl, shape (n_frames, npt_azim)
    <tmp_dir>/scan_XXXX.meta.json  — scan attributes shared by all hkls (dty, monitor stats, …)

A scan is only considered done once its meta file and every requested hkl's
.npy file are all present; the whole scan is reprocessed together on retry
(cheap, since cake integration was going to happen once per frame anyway).

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
from typing import Dict, Optional, Tuple

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
    hkls: Dict[str, Tuple[float, float]],
    background_ranges: Dict[str, Optional[Tuple[Tuple[float, float], Tuple[float, float]]]],
) -> tuple[int, Dict[str, np.ndarray]]:
    cake, radial, azimuthal = cake_integration(
        image, str(poni_file), npt_rad=npt_rad, npt_azim=npt_azim, mask=mask
    )
    intensities = {}
    for hkl, tth_range in hkls.items():
        _, intensity = extract_ring_intensity(
            cake, radial, azimuthal, tth_range, background_ranges.get(hkl)
        )
        intensities[hkl] = intensity
    if monitor <= 0:
        print(f"  ⚠  Frame {jj}: monitor={monitor:.4g}, skipping normalisation")
        return jj, intensities
    return jj, {hkl: i / monitor for hkl, i in intensities.items()}


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
    hkls: Dict[str, Tuple[float, float]],
    background_ranges: Dict[str, Optional[Tuple[Tuple[float, float], Tuple[float, float]]]],
    npt_rad: int,
    npt_azim: int,
    n_workers: Optional[int],
    batch_size: int,
    camera_name: str,
    monitor_name: str,
    rotation_motor: str,
) -> bool:
    scan_name = f"scan_{ii:04d}"
    meta_path = tmp_dir / f"{scan_name}.meta.json"
    npy_paths = {hkl: tmp_dir / f"{scan_name}_{hkl}.npy" for hkl in hkls}

    if meta_path.exists() and all(p.exists() for p in npy_paths.values()):
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
    profiles = {hkl: np.empty((n_frames, npt_azim), dtype=np.float32) for hkl in hkls}
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
                for hkl in hkls:
                    profiles[hkl][batch_start:batch_end] = np.nan
                pbar.update(batch_end - batch_start)
                continue

            batch_monitor = monitor[batch_start:batch_end]

            def _task(args):
                local_jj, image, mon = args
                return _extract_frame(
                    local_jj, image, mon,
                    poni_file=poni_file, npt_rad=npt_rad, npt_azim=npt_azim,
                    mask=mask, hkls=hkls, background_ranges=background_ranges,
                )

            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(_task, (batch_start + k, batch_images[k], batch_monitor[k])): k
                    for k in range(len(batch_images))
                }
                for future in as_completed(futures):
                    try:
                        jj, intensities = future.result()
                        for hkl, intensity in intensities.items():
                            profiles[hkl][jj] = intensity
                    except Exception as e:
                        jj = batch_start + futures[future]
                        print(f"  ✗ Frame {jj} failed: {e}")
                        for hkl in hkls:
                            profiles[hkl][jj] = np.nan
                    pbar.update(1)

            del batch_images, raw

    # ── Write tmp files atomically ────────────────────────────────────────────
    npy_tmps = {}
    for hkl in hkls:
        npy_tmp_stem = tmp_dir / f"{scan_name}_{hkl}.tmp"   # np.save → ..._{hkl}.tmp.npy
        np.save(npy_tmp_stem, profiles[hkl])
        npy_tmps[hkl] = tmp_dir / f"{scan_name}_{hkl}.tmp.npy"

    meta_tmp = tmp_dir / f"{scan_name}.meta.json.tmp"
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
        "hkls": list(hkls.keys()),
        "valid": True,
    }
    meta_tmp.write_text(json.dumps(meta, indent=2))

    for hkl in hkls:
        npy_tmps[hkl].rename(npy_paths[hkl])
    meta_tmp.rename(meta_path)

    print(f"  ✓ {scan_name} → {', '.join(p.name for p in npy_paths.values())}")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="nrxrdct pole-figure extraction worker (one SLURM job)"
    )
    p.add_argument("--master-file", required=True, type=Path)
    p.add_argument("--tmp-dir", required=True, type=Path,
                   help="Shared tmp directory for .npy and .meta.json outputs; "
                        "also where launch_meta.json (hkls, integration settings) is read from")
    p.add_argument("--poni-file", required=True, type=Path)
    p.add_argument("--mask-file", required=True, type=Path)
    p.add_argument("--entry-indices", required=True)
    return p.parse_args()


def main():
    args = _parse_args()
    entry_indices = [int(x) for x in args.entry_indices.split(",")]

    meta_sidecar = args.tmp_dir / "launch_meta.json"
    with open(meta_sidecar) as f:
        launch_meta = json.load(f)
    valid_entries = launch_meta["valid_entries"]
    dty_values = launch_meta["dty_values"]
    hkls = {hkl: tuple(tth_range) for hkl, tth_range in launch_meta["hkls"].items()}
    background_ranges = {
        hkl: (tuple(bg[0]), tuple(bg[1]))
        for hkl, bg in launch_meta.get("background_ranges", {}).items()
    }
    npt_rad = launch_meta.get("npt_rad", 1000)
    npt_azim = launch_meta.get("npt_azim", 360)
    n_workers = launch_meta.get("n_workers")
    batch_size = launch_meta.get("batch_size", 32)
    camera_name = launch_meta.get("camera_name", "eiger")
    monitor_name = launch_meta.get("monitor_name", "fpico6")
    rotation_motor = launch_meta.get("rotation_motor", "rot")

    print(
        f"Worker started — {len(entry_indices)} scans | hkls={list(hkls.keys())} | "
        f"batch={batch_size} | threads={'auto' if n_workers is None else n_workers}"
    )

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
            hkls=hkls,
            background_ranges=background_ranges,
            npt_rad=npt_rad,
            npt_azim=npt_azim,
            n_workers=n_workers,
            batch_size=batch_size,
            camera_name=camera_name,
            monitor_name=monitor_name,
            rotation_motor=rotation_motor,
        )
        n_ok += ok
        n_fail += not ok

    elapsed = time.time() - t0
    print(f"\nWorker done in {elapsed:.1f}s — {n_ok} OK, {n_fail} failed")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()