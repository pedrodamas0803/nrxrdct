"""
nrxrdct.xrdct._segment_worker
------------------------------
Worker executed inside each SLURM job for scanning-3DXRD segmentation.

Each assigned scan is segmented via segment_scan() and written atomically
as a single HDF5 file in a shared scan directory:

    <scan_dir>/scan_XXXX.h5   — datasets: sc, fc, omega, sum_intensity, n_pixels
                               — attrs:    entry, <translation_motor>, n_peaks

No shared HDF5 file is touched during segmentation — the final segmented.h5
is assembled by slurm_s3dxrd.merge() after all jobs finish, using HDF5
external links that point back into this directory.

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
import h5py

from nrxrdct.xrdct.s3dxrd import SegmentationOptions, _write_scan_group, segment_scan


def _parse_args():
    p = argparse.ArgumentParser(
        description="nrxrdct scanning-3DXRD segmentation worker (one SLURM job)"
    )
    p.add_argument("--master-file",       required=True, type=Path)
    p.add_argument("--scan-dir",          required=True, type=Path)
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
    scan_dir: Path,
    mask,
    options: SegmentationOptions,
    camera_name: str,
    translation_motor: str,
    rotation_motor: str,
) -> bool:
    scan_name = f"scan_{ii:04d}"
    h5_path   = scan_dir / f"{scan_name}.h5"

    if h5_path.exists():
        print(f"  → Skipping {scan_name} (already done)")
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

    # Atomic write: write to .tmp.h5, then rename.
    h5_tmp = scan_dir / f"{scan_name}.tmp.h5"
    with h5py.File(h5_tmp, "w") as hout:
        _write_scan_group(hout, "scan", result, translation_motor)
    h5_tmp.rename(h5_path)

    print(f"  ✓ {scan_name} — {result.n_peaks:,} peaks  →  {h5_path.name}")
    return True


def main():
    args          = _parse_args()
    entry_indices = [int(x) for x in args.entry_indices.split(",")]

    print(
        f"Segmentation worker started — {len(entry_indices)} scans | "
        f"cut={args.cut} | howmany={args.howmany} | "
        f"pixels_in_spot={args.pixels_in_spot}"
    )

    sidecar = args.scan_dir / "launch_meta.json"
    with open(sidecar) as f:
        launch_meta = json.load(f)
    valid_entries = launch_meta["valid_entries"]

    mask    = fabio.open(args.mask_file).data
    options = SegmentationOptions(
        cut=args.cut,
        howmany=args.howmany,
        pixels_in_spot=args.pixels_in_spot,
    )
    args.scan_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    n_ok = n_fail = 0

    for ii in entry_indices:
        ok = _process_scan(
            ii, valid_entries[ii],
            master_file=args.master_file,
            scan_dir=args.scan_dir,
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
