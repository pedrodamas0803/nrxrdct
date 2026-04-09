"""
nrxrdct.slurm_integration.merge
---------------------------------
Assemble the final output HDF5 from the per-scan .npy / .meta.json files
produced by integrate_worker.py.

This is the only step that writes to the output HDF5 file, and it is
strictly single-threaded / single-process — no concurrent access, no
corruption possible.

Python API
----------
    from nrxrdct.slurm_integration import merge

    merge(
        tmp_dir     = Path("integration_tmp"),
        output_file = Path("output.h5"),
    )

CLI
---
    nrxrdct-slurm merge --tmp-dir integration_tmp --output-file output.h5
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def merge(
    tmp_dir: Path,
    output_file: Path,
    *,
    overwrite: bool = False,
) -> dict:
    """
    Assemble the output HDF5 from per-scan tmp files.

    Parameters
    ----------
    tmp_dir     : Path
        Directory containing scan_XXXX.npy and scan_XXXX.meta.json files,
        plus the launch_meta.json sidecar written by launch().
    output_file : Path
        Path to the output HDF5 file to create (or append to).
    overwrite   : bool
        If True, overwrite any existing scan datasets in the output file.
        If False (default), skip scans already present (safe to re-run).

    Returns
    -------
    dict with keys 'n_merged', 'n_skipped', 'n_missing'.
    """
    tmp_dir     = Path(tmp_dir)
    output_file = Path(output_file)

    # ── Read launch metadata ──────────────────────────────────────────────────
    meta_sidecar = tmp_dir / "launch_meta.json"
    if not meta_sidecar.exists():
        raise FileNotFoundError(
            f"launch_meta.json not found in {tmp_dir}. "
            "Was launch() called with this tmp_dir?"
        )
    with open(meta_sidecar) as f:
        launch_meta   = json.load(f)

    valid_entries = launch_meta["valid_entries"]
    dty_values    = launch_meta["dty_values"]
    rot           = np.array(launch_meta["rot"])
    bad_entries   = launch_meta.get("bad_entries", [])
    unit          = launch_meta["unit"]
    n_total       = len(valid_entries)

    # ── Collect available scan files ──────────────────────────────────────────
    # Support both correctly named files (scan_XXXX.npy + scan_XXXX.meta.json)
    # and the previously misnamed tmp files (scan_XXXX.npy.tmp.npy +
    # scan_XXXX.meta.meta.json.tmp) so existing runs can be recovered.
    available: dict[int, tuple[Path, Path]] = {}

    for npy_path in sorted(tmp_dir.glob("scan_????.npy")):
        ii        = int(npy_path.stem.split("_")[1])
        meta_path = tmp_dir / f"scan_{ii:04d}.meta.json"
        if meta_path.exists():
            available[ii] = (npy_path, meta_path)

    # Recover misnamed files from previous bug (rename them in-place)
    for bad_npy in sorted(tmp_dir.glob("scan_????.npy.tmp.npy")):
        ii       = int(bad_npy.name.split("_")[1].split(".")[0])
        good_npy = tmp_dir / f"scan_{ii:04d}.npy"
        # Find matching misnamed meta file
        bad_metas = list(tmp_dir.glob(f"scan_{ii:04d}.meta.meta.json.tmp"))
        bad_metas += list(tmp_dir.glob(f"scan_{ii:04d}.meta.json.tmp"))
        if ii not in available and bad_metas:
            good_meta = tmp_dir / f"scan_{ii:04d}.meta.json"
            print(f"  ↻  Recovering scan_{ii:04d}: renaming misnamed tmp files")
            bad_npy.rename(good_npy)
            bad_metas[0].rename(good_meta)
            available[ii] = (good_npy, good_meta)

    print(f"\n{'='*60}")
    print(f"Merging {len(available)}/{n_total} scans → {output_file.name}")
    print(f"{'='*60}\n")

    n_merged = n_skipped = n_missing = 0

    with h5py.File(output_file, "a") as hout:

        # ── Write global metadata (idempotent) ────────────────────────────────
        if "motors/dty" not in hout:
            hout["motors/dty"] = dty_values
        if "motors/rot" not in hout:
            hout["motors/rot"] = rot
        if "meta/valid_entries" not in hout:
            hout["meta/valid_entries"] = np.array(
                valid_entries, dtype=h5py.string_dtype()
            )
        if bad_entries and "bad_entries" not in hout:
            hout["bad_entries"] = np.array(
                bad_entries, dtype=h5py.string_dtype()
            )

        # ── Write radial axis from first available scan ────────────────────────
        if "integrated/radial" not in hout:
            # Read the radial axis from launch_meta (stored by launch())
            if "radial" in launch_meta:
                hout["integrated/radial"]            = np.array(launch_meta["radial"])
                hout["integrated/radial"].attrs["unit"] = unit
            else:
                print("  ⚠  No radial axis in launch_meta.json — "
                      "it will be written from the first scan npy sidecar.")

        if "integrated/cake_mask" not in hout and "cake_mask" in launch_meta:
            hout["integrated/cake_mask"] = np.array(launch_meta["cake_mask"])

        # ── Merge scan datasets ───────────────────────────────────────────────
        for ii in tqdm(range(n_total), desc="Merging scans"):
            scan_name  = f"scan_{ii:04d}"
            group_path = f"integrated/{scan_name}"

            # Skip if already in output and not overwriting
            if group_path in hout and not overwrite:
                n_skipped += 1
                continue

            if ii not in available:
                n_missing += 1
                continue

            npy_path, meta_path = available[ii]

            try:
                sinogram = np.load(npy_path)
                with open(meta_path) as f:
                    meta = json.load(f)
            except Exception as e:
                print(f"  ✗ {scan_name}: failed to load tmp files — {e}")
                n_missing += 1
                continue

            # Delete existing dataset if overwriting
            if group_path in hout and overwrite:
                del hout[group_path]

            ds = hout.create_dataset(
                group_path,
                data=sinogram,
                compression="gzip",
                compression_opts=4,
                chunks=(1, sinogram.shape[1]),
            )
            for k, v in meta.items():
                ds.attrs[k] = v

            n_merged += 1

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Merged  : {n_merged}")
    print(f"  Skipped : {n_skipped}  (already in output)")
    print(f"  Missing : {n_missing}  (no tmp file — rerun integration)")
    print(f"{'='*60}\n")

    if n_missing:
        missing_idx = [
            ii for ii in range(n_total)
            if ii not in available and
               f"integrated/scan_{ii:04d}" not in
               (h5py.File(output_file, "r") if output_file.exists() else {})
        ]
        print(f"  Missing indices: {sorted(set(range(n_total)) - set(available))}")
        print(f"  Re-run repair() or launch() for those indices.\n")

    return {
        "n_merged":  n_merged,
        "n_skipped": n_skipped,
        "n_missing": n_missing,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser(sub=None):
    import argparse
    desc = "Assemble output HDF5 from per-scan tmp files"
    p = (
        sub.add_parser("merge", help=desc, description=desc)
        if sub else
        argparse.ArgumentParser(description=desc)
    )
    p.add_argument("--tmp-dir",     required=True, type=Path)
    p.add_argument("--output-file", required=True, type=Path)
    p.add_argument("--overwrite",   action="store_true",
                   help="Overwrite existing scan datasets in the output file")
    return p


def _cli_merge(args):
    merge(
        tmp_dir     = args.tmp_dir,
        output_file = args.output_file,
        overwrite   = args.overwrite,
    )


if __name__ == "__main__":
    p = _build_parser()
    _cli_merge(p.parse_args())