"""
nrxrdct.texture.slurm_pole_figures.merge
-------------------------------------------
Assemble the final pole-figure HDF5 from the per-scan .npy / .meta.json files
produced by pole_figure_worker.py.

This is the only step that writes to the output HDF5 file, and it is
strictly single-threaded / single-process — no concurrent access, no
corruption possible.

Python API
----------
    from nrxrdct.texture.slurm_pole_figures import merge

    merge(
        tmp_dir     = Path("pole_figures_tmp"),
        output_file = Path("pole_figures.h5"),
    )

CLI
---
    nrxrdct-slurm-texture merge --tmp-dir pole_figures_tmp --output-file pole_figures.h5
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

    Args:
        tmp_dir (Path): Directory containing scan_XXXX.npy and scan_XXXX.meta.json
            files, plus the launch_meta.json sidecar written by launch().
        output_file (Path): Path to the output HDF5 file to create (or append to).
        overwrite (bool): If True, overwrite any existing scan datasets in the output
            file. If False (default), skip scans already present (safe to re-run).

    Returns:
        dict: With keys ``'n_merged'``, ``'n_skipped'``, ``'n_missing'``.
    """
    tmp_dir = Path(tmp_dir)
    output_file = Path(output_file)

    meta_sidecar = tmp_dir / "launch_meta.json"
    if not meta_sidecar.exists():
        raise FileNotFoundError(
            f"launch_meta.json not found in {tmp_dir}. "
            "Was launch() called with this tmp_dir?"
        )
    with open(meta_sidecar) as f:
        launch_meta = json.load(f)

    valid_entries = launch_meta["valid_entries"]
    dty_values = launch_meta["dty_values"]
    bad_entries = launch_meta.get("bad_entries", [])
    hkl_label = launch_meta["hkl_label"]
    tth_range = launch_meta["tth_range"]
    npt_azim = launch_meta["npt_azim"]
    translation_motor = launch_meta["translation_motor"]
    n_total = len(valid_entries)

    group = f"pole_figures/{hkl_label}"

    # ── Collect available scan files ──────────────────────────────────────────
    available: dict[int, tuple[Path, Path]] = {}
    for npy_path in sorted(tmp_dir.glob("scan_????.npy")):
        ii = int(npy_path.stem.split("_")[1])
        meta_path = tmp_dir / f"scan_{ii:04d}.meta.json"
        if meta_path.exists():
            available[ii] = (npy_path, meta_path)

    print(f"\n{'='*60}")
    print(f"Merging {len(available)}/{n_total} scans → {output_file.name} [{group}]")
    print(f"{'='*60}\n")

    n_merged = n_skipped = n_missing = 0

    with h5py.File(output_file, "a") as hout:

        if f"motors/{translation_motor}" not in hout:
            hout[f"motors/{translation_motor}"] = dty_values
        if bad_entries and "bad_entries" not in hout:
            hout["bad_entries"] = np.array(bad_entries, dtype=h5py.string_dtype())

        if f"{group}/azimuthal" not in hout:
            azimuthal_axis = -180.0 + (360.0 / npt_azim) * (np.arange(npt_azim) + 0.5)
            hout[f"{group}/azimuthal"] = azimuthal_axis
            hout[f"{group}/tth_range"] = np.asarray(tth_range, dtype=np.float64)

        for ii in tqdm(range(n_total), desc="Merging scans"):
            scan_name = f"scan_{ii:04d}"
            group_path = f"{group}/{scan_name}"

            if group_path in hout and not overwrite:
                n_skipped += 1
                continue

            if ii not in available:
                n_missing += 1
                continue

            npy_path, meta_path = available[ii]

            try:
                profiles = np.load(npy_path)
                with open(meta_path) as f:
                    meta = json.load(f)
            except Exception as e:
                print(f"  ✗ {scan_name}: failed to load tmp files — {e}")
                n_missing += 1
                continue

            if group_path in hout and overwrite:
                del hout[group_path]

            ds = hout.create_dataset(
                group_path,
                data=profiles,
                compression="gzip",
                compression_opts=4,
                chunks=(1, profiles.shape[1]),
            )
            for k, v in meta.items():
                ds.attrs[k] = v

            n_merged += 1

    print(f"\n{'='*60}")
    print(f"  Merged  : {n_merged}")
    print(f"  Skipped : {n_skipped}  (already in output)")
    print(f"  Missing : {n_missing}  (no tmp file — rerun extraction)")
    print(f"{'='*60}\n")

    if n_missing:
        print(f"  Missing indices: {sorted(set(range(n_total)) - set(available))}")
        print(f"  Re-run repair() for those indices.\n")

    return {
        "n_merged": n_merged,
        "n_skipped": n_skipped,
        "n_missing": n_missing,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser(sub=None):
    import argparse
    desc = "Assemble pole-figure output HDF5 from per-scan tmp files"
    p = (
        sub.add_parser("merge", help=desc, description=desc)
        if sub else
        argparse.ArgumentParser(description=desc)
    )
    p.add_argument("--tmp-dir", required=True, type=Path)
    p.add_argument("--output-file", required=True, type=Path)
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite existing scan datasets in the output file")
    return p


def _cli_merge(args):
    merge(
        tmp_dir=args.tmp_dir,
        output_file=args.output_file,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    p = _build_parser()
    _cli_merge(p.parse_args())