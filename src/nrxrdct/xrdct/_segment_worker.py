"""
nrxrdct.xrdct._segment_worker
------------------------------
Worker executed inside each SLURM job for scanning-3DXRD segmentation.

Each assigned scan is segmented via segment_scan() and the peak arrays
are written atomically as two files in a shared tmp directory:

    <tmp_dir>/scan_XXXX.npz        — arrays: sc, fc, omega, sum_intensity, n_pixels
    <tmp_dir>/scan_XXXX.meta.json  — scan attributes (entry, dty, n_peaks)

No HDF5 file is touched during segmentation — the final segmented.h5 is
assembled by slurm_s3dxrd.merge() after all jobs finish.

Invoked by slurm_s3dxrd.launch() via:
    python -m nrxrdct.xrdct._segment_worker <args>
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import fabio
import numpy as np

from nrxrdct.xrdct.s3dxrd import SegmentationOptions, segment_scan


def _parse_args():
    p = argparse.ArgumentParser(
        description="nrxrdct scanning-3DXRD segmentation worker (one SLURM job)"
    )
    p.add_argument("--master-file",       required=True, type=Path)
    p.add_argument("--tmp-dir",           required=True, type=Path)
    p.add_argument("--mask-file",         required=True, type=Path)
    p.add_argument(
        "--entry-indices", required=True,
        help="Comma-separated global scan indices assigned to this job",
    )
    p.add_argument("--camera-name",       default="eiger")
    p.add_argument("--translation-motor", default="dty")
    p.add_argument("--rotation-motor",    default="rot")
    p.add_argument("--cut",               type=float, default=1.0)
    p.add_argument("--howmany",           type=int,   default=100_000)
    p.add_argument("--pixels-in-spot",    type=int,   default=3)
    return p.parse_args()


def _process_scan(
    ii: int,
    entry: str,
    *,
    master_file: Path,
    tmp_dir: Path,
    mask: "np.ndarray",
    options: SegmentationOptions,
    camera_name: str,
    translation_motor: str,
    rotation_motor: str,
) -> bool:
    scan_name = f"scan_{ii:04d}"
    npz_path  = tmp_dir / f"{scan_name}.npz"
    meta_path = tmp_dir / f"{scan_name}.meta.json"

    if npz_path.exists() and meta_path.exists():
        print(f"  → Skipping {scan_name} (already in tmp)")
        return True

    print(f"\n{'='*60}\n{scan_name} — entry {entry}  [global idx {ii}]\n{'='*60}")

    try:
        result = segment_scan(
            master_file, entry, mask, options,
            camera_name=camera_name,
            translation_motor=translation_motor,
            rotation_motor=rotation_motor,
        )
    except (OSError, KeyError) as e:
        print(f"  ✗ Failed to segment entry {entry}: {e} — skipping")
        return False

    # Atomic write: write to .tmp names, then rename both together.
    # np.savez_compressed adds .npz automatically, so use a stem without it.
    npz_tmp_stem = tmp_dir / f"{scan_name}.tmp"   # → scan_XXXX.tmp.npz on disk
    meta_tmp     = tmp_dir / f"{scan_name}.meta.json.tmp"

    np.savez_compressed(
        npz_tmp_stem,
        sc            = result.sc,
        fc            = result.fc,
        omega         = result.omega,
        sum_intensity = result.sum_intensity,
        n_pixels      = result.n_pixels,
    )
    npz_tmp = tmp_dir / f"{scan_name}.tmp.npz"   # actual file written by savez

    meta = {
        "scan_index": ii,
        "entry":      entry,
        "dty":        result.dty,
        "n_peaks":    result.n_peaks,
    }
    meta_tmp.write_text(json.dumps(meta, indent=2))

    npz_tmp.rename(npz_path)
    meta_tmp.rename(meta_path)

    print(f"  ✓ {scan_name} — {result.n_peaks:,} peaks  →  {npz_path.name}")
    return True


def main():
    args          = _parse_args()
    entry_indices = [int(x) for x in args.entry_indices.split(",")]

    print(
        f"Segmentation worker started — {len(entry_indices)} scans | "
        f"cut={args.cut} | howmany={args.howmany} | "
        f"pixels_in_spot={args.pixels_in_spot}"
    )

    sidecar = args.tmp_dir / "launch_meta.json"
    with open(sidecar) as f:
        launch_meta = json.load(f)
    valid_entries = launch_meta["valid_entries"]

    mask    = fabio.open(args.mask_file).data
    options = SegmentationOptions(
        cut=args.cut,
        howmany=args.howmany,
        pixels_in_spot=args.pixels_in_spot,
    )
    args.tmp_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    n_ok = n_fail = 0

    for ii in entry_indices:
        ok = _process_scan(
            ii, valid_entries[ii],
            master_file=args.master_file,
            tmp_dir=args.tmp_dir,
            mask=mask,
            options=options,
            camera_name=args.camera_name,
            translation_motor=args.translation_motor,
            rotation_motor=args.rotation_motor,
        )
        n_ok   += ok
        n_fail += not ok

    elapsed = time.time() - t0
    print(f"\nWorker done in {elapsed:.1f}s — {n_ok} OK, {n_fail} failed")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
