"""
nrxrdct.slurm_integration.check_output
----------------------------------------
Verify progress and completeness of the integration pipeline.

Two stages can be checked independently:

1. **Integration progress** — counts .npy files in the tmp directory
   (before merge).
2. **Merge completeness** — counts scan datasets in the output HDF5
   (after merge).

Python API
----------
    from nrxrdct.slurm_integration import check

    # Check integration progress (tmp files)
    check(tmp_dir=Path("output_tmp"))

    # Check merge completeness (HDF5 datasets)
    check(output_file=Path("output.h5"))

    # Check both
    check(tmp_dir=Path("output_tmp"), output_file=Path("output.h5"))

CLI
---
    nrxrdct-slurm check --tmp-dir output_tmp [--output-file output.h5]
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def check(
    tmp_dir: Path | None = None,
    output_file: Path | None = None,
) -> dict:
    """
    Verify integration progress and/or merge completeness.

    Parameters
    ----------
    tmp_dir     : Path, optional
        Tmp directory written by workers. If provided, counts completed
        .npy files and reports missing indices.
    output_file : Path, optional
        Output HDF5 file. If provided, counts merged scan datasets.

    Returns
    -------
    dict with keys:
        'n_total'         — total expected scans
        'n_integrated'    — .npy files present in tmp_dir
        'n_merged'        — datasets present in output_file
        'missing_tmp'     — indices not yet integrated
        'missing_h5'      — indices not yet merged
    """
    if tmp_dir is None and output_file is None:
        raise ValueError("Provide at least one of tmp_dir or output_file.")

    result = {
        "n_total": 0,
        "n_integrated": 0,
        "n_merged": 0,
        "missing_tmp": [],
        "missing_h5": [],
    }

    # ── Read total expected scans from launch_meta.json ───────────────────────
    if tmp_dir is not None:
        tmp_dir = Path(tmp_dir)
        meta_sidecar = tmp_dir / "launch_meta.json"
        if not meta_sidecar.exists():
            raise FileNotFoundError(
                f"launch_meta.json not found in {tmp_dir}. " "Was launch() called?"
            )
        with open(meta_sidecar) as f:
            launch_meta = json.load(f)

        valid_entries = launch_meta["valid_entries"]
        n_total = len(valid_entries)
        result["n_total"] = n_total

        # Count .npy files (only count if matching .meta.json also exists)
        integrated = {
            int(p.stem.split("_")[1])
            for p in tmp_dir.glob("scan_????.npy")
            if (tmp_dir / f"{p.stem}.meta.json").exists()
        }
        missing_tmp = sorted(set(range(n_total)) - integrated)

        result["n_integrated"] = len(integrated)
        result["missing_tmp"] = missing_tmp

        print(f"\n{'='*60}")
        print(f"  Integration progress  ({tmp_dir.name})")
        print(f"{'='*60}")
        print(f"  Expected    : {n_total}")
        print(f"  Integrated  : {len(integrated)}")
        print(f"  Remaining   : {len(missing_tmp)}")
        if missing_tmp:
            print(f"  Missing idx : {missing_tmp}")
        else:
            print(f"  ✓  All scans integrated — ready to merge.")

    # ── Check HDF5 merge completeness ─────────────────────────────────────────
    if output_file is not None:
        output_file = Path(output_file)
        if not output_file.exists():
            print(f"\n  ⚠  Output file not found: {output_file}")
            print(f"     Run merge() first.")
            return result

        try:
            with h5py.File(output_file, "r") as hout:
                if "meta/valid_entries" not in hout:
                    print(
                        f"\n  ⚠  'meta/valid_entries' missing from {output_file.name}"
                    )
                    return result

                n_total = len(hout["meta/valid_entries"])
                result["n_total"] = max(result["n_total"], n_total)

                merged = {
                    ii for ii in range(n_total) if f"integrated/scan_{ii:04d}" in hout
                }
                missing_h5 = sorted(set(range(n_total)) - merged)

                result["n_merged"] = len(merged)
                result["missing_h5"] = missing_h5

                has_radial = "integrated/radial" in hout
                radial_info = (
                    (
                        hout["integrated/radial"].shape[0],
                        hout["integrated/radial"].attrs.get("unit", "?"),
                    )
                    if has_radial
                    else None
                )

        except OSError as e:
            print(f"\n  ✗  Cannot open {output_file.name}: {e}")
            print(f"     The file may be corrupted. Run rebuild().")
            return result

        print(f"\n{'='*60}")
        print(f"  Merge completeness    ({output_file.name})")
        print(f"{'='*60}")
        print(f"  Expected  : {n_total}")
        print(f"  Merged    : {len(merged)}")
        print(f"  Missing   : {len(missing_h5)}")
        if missing_h5:
            print(f"  Missing idx : {missing_h5}")
        else:
            print(f"  ✓  All scans merged.")
        if radial_info:
            print(f"  Radial    : {radial_info[0]} pts, unit={radial_info[1]}")

    print()
    return result


# ─────────────────────────────────────────────────────────────────────────────
# repair() — resubmit missing scans
# ─────────────────────────────────────────────────────────────────────────────


def repair(
    tmp_dir: Path,
    master_file: Path | None = None,
    poni_file: Path | None = None,
    mask_file: Path | None = None,
    *,
    output_file: Path | None = None,
    n_jobs: int = 1,
    watch: bool = False,
    interval: int = 30,
    **kwargs,
) -> dict:
    """
    Resubmit SLURM jobs for any scans missing from the tmp directory.

    All integration and SLURM settings are read from launch_meta.json so
    you don't need to repeat them. Pass **kwargs to override any individual
    setting (e.g. partition, mem, n_workers).

    Parameters
    ----------
    tmp_dir     : Path  — tmp directory from the original launch()
    master_file : Path  — override master HDF5 (default: from launch_meta)
    poni_file   : Path  — override calibration file (default: from launch_meta)
    mask_file   : Path  — override mask file (default: from launch_meta)
    output_file : Path  — only used to check merge status if provided
    n_jobs      : int   — number of repair jobs (default 1)
    watch       : bool  — block until repair jobs finish
    interval    : int   — polling interval in seconds when watch=True
    **kwargs    : override any setting from launch_meta (partition, mem, etc.)
    """
    from .launch_jobs import _split_indices, _submit_job

    tmp_dir = Path(tmp_dir)
    result = check(tmp_dir=tmp_dir, output_file=output_file)

    missing = result["missing_tmp"]
    if not missing:
        print("✓  Nothing to repair — all scans present in tmp dir.")
        return result

    print(f"\n🔧  Repairing {len(missing)} missing scans across {n_jobs} job(s)...")

    with open(tmp_dir / "launch_meta.json") as f:
        lm = json.load(f)

    # Resolve paths — kwargs override launch_meta, launch_meta overrides defaults
    _master_file = Path(kwargs.pop("master_file", master_file or lm["master_file"]))
    _poni_file = Path(kwargs.pop("poni_file", poni_file or lm["poni_file"]))
    _mask_file = Path(kwargs.pop("mask_file", mask_file or lm["mask_file"]))
    _env_activate = kwargs.pop("env_activate", lm.get("env_activate"))
    _env_activate = Path(_env_activate) if _env_activate else None

    # All settings fall back to what was used at launch time
    settings = dict(
        n_points=lm.get("n_points", 1000),
        n_workers=lm.get("cpus", 16),
        batch_size=lm.get("batch_size", 32),
        unit=lm.get("unit", "2th_deg"),
        method=lm.get("method", "standard"),
        percentile=lm.get("percentile", "10,90"),
        thres=lm.get("thres", 3.0),
        max_iter=lm.get("max_iter", 5),
        partition=lm.get("partition", "cpu"),
        time=lm.get("time", "04:00:00"),
        mem=lm.get("mem", "32G"),
        cpus=lm.get("cpus", 16),
        gpu=lm.get("gpu", False),
        conda_env=lm.get("conda_env", None),
    )
    # Apply user overrides
    settings.update(kwargs)

    log_dir = Path(lm["output_file"]).parent / "slurm_logs"
    log_dir.mkdir(exist_ok=True)
    base_id = len(sorted(log_dir.glob("job_*.sh")))

    chunks = _split_indices(len(missing), min(n_jobs, len(missing)))
    chunks = [[missing[i] for i in chunk] for chunk in chunks]

    slurm_ids = []
    for offset, chunk in enumerate(chunks):
        sid = _submit_job(
            base_id + offset,
            chunk,
            master_file=_master_file,
            tmp_dir=tmp_dir,
            poni_file=_poni_file,
            mask_file=_mask_file,
            n_points=settings["n_points"],
            n_workers=settings["n_workers"],
            batch_size=settings["batch_size"],
            unit=settings["unit"],
            method=settings["method"],
            percentile=settings["percentile"],
            thres=settings["thres"],
            max_iter=settings["max_iter"],
            partition=settings["partition"],
            time=settings["time"],
            mem=settings["mem"],
            cpus=settings["cpus"],
            gpu=settings["gpu"],
            env_activate=_env_activate,
            conda_env=settings["conda_env"],
            log_dir=log_dir,
        )
        slurm_ids.append(sid)

    print(
        f"\n✓  {len(slurm_ids)} repair job(s) submitted — IDs: {', '.join(slurm_ids)}"
    )

    if watch:
        from .monitor import monitor as _monitor

        _monitor(
            slurm_ids=slurm_ids,
            tmp_dir=tmp_dir,
            watch=True,
            interval=interval,
        )

    result["repair_job_ids"] = slurm_ids
    return result


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def _build_parser(sub=None):
    import argparse

    desc = "Check integration progress and merge completeness"
    p = (
        sub.add_parser("check", help=desc, description=desc)
        if sub
        else argparse.ArgumentParser(description=desc)
    )
    p.add_argument(
        "--tmp-dir",
        type=Path,
        default=None,
        help="Tmp directory from launch() — checks .npy progress",
    )
    p.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="Output HDF5 — checks merge completeness",
    )
    return p


def _cli_check(args):
    check(tmp_dir=args.tmp_dir, output_file=args.output_file)


if __name__ == "__main__":
    p = _build_parser()
    _cli_check(p.parse_args())

# """
# nrxrdct.slurm_integration.check_output
# ----------------------------------------
# Verify completeness of the output HDF5 file after all SLURM jobs finish.

# Python API
# ----------
#     from nrxrdct.slurm_integration import check

#     # Just report
#     result = check(output_file=Path("output.h5"))

#     # Report + print resubmit hints
#     result = check(output_file=Path("output.h5"), resubmit=True)

#     # Report + automatically delete corrupted/missing datasets and resubmit
#     result = check(
#         output_file  = Path("output.h5"),
#         repair       = True,
#         master_file  = Path("master.h5"),
#         poni_file    = Path("calib.poni"),
#         mask_file    = Path("mask.edf"),
#         partition    = "cpu",
#         conda_env    = "nrxrdct",
#     )

# CLI
# ---
#     nrxrdct-slurm check --output-file output.h5 [--resubmit]
#     nrxrdct-slurm check --output-file output.h5 --repair \\
#         --master-file master.h5 --poni-file calib.poni --mask-file mask.edf
# """

# from __future__ import annotations

# import subprocess
# import textwrap
# from pathlib import Path

# import h5py
# import numpy as np

# # ─────────────────────────────────────────────────────────────────────────────
# # Internal helpers
# # ─────────────────────────────────────────────────────────────────────────────


# def _delete_datasets(output_file: Path, indices: list[int]) -> None:
#     """
#     Delete HDF5 datasets for the given scan indices.

#     Uses ``h5py.h5g.unlink`` (low-level C API, available in all h5py versions)
#     instead of ``del h[path]`` or ``path in h``, both of which crash when the
#     HDF5 link/B-tree table is corrupted ("bad symbol table node signature",
#     "incorrect cache entry type", etc.).
#     """
#     with h5py.File(output_file, "a") as h:
#         for i in indices:
#             scan_name = f"scan_{i:04d}"
#             path = f"integrated/{scan_name}"

#             # ── Try via parent group (most reliable) ──────────────────────────
#             try:
#                 parent_id = h["integrated"].id
#                 h5py.h5g.unlink(parent_id, scan_name.encode())
#                 print(f"  🗑  Deleted {path}")
#                 continue
#             except KeyError:
#                 pass  # link didn't exist — nothing to do
#             except Exception as e1:
#                 pass  # parent group unreadable — fall through to root attempt

#             # ── Fallback: unlink from root using full path ────────────────────
#             try:
#                 h5py.h5g.unlink(h["/"].id, path.encode())
#                 print(f"  🗑  Deleted {path} (root fallback)")
#             except KeyError:
#                 pass  # truly doesn't exist
#             except Exception as e2:
#                 print(
#                     f"  ⚠  Could not delete {path}: {e2} — will be treated as missing"
#                 )


# def _resubmit(
#     output_file: Path,
#     needs_rerun: list[int],
#     *,
#     master_file: Path,
#     poni_file: Path,
#     mask_file: Path,
#     n_jobs: int,
#     n_points: int,
#     n_workers: int | None,
#     batch_size: int,
#     unit: str,
#     method: str,
#     percentile: str,
#     thres: float,
#     max_iter: int,
#     partition: str,
#     time: str,
#     mem: str,
#     cpus: int,
#     gpu: bool,
#     env_activate: Path | None,
#     conda_env: str | None,
# ) -> list[str]:
#     """
#     Submit one or more sbatch repair jobs covering all needs_rerun indices.

#     If n_jobs > 1, the indices are split into chunks (same logic as launch())
#     and one job is submitted per chunk. Returns a list of SLURM job IDs.
#     """
#     import math

#     from .launch_jobs import _split_indices, _submit_job

#     log_dir = output_file.parent / "slurm_logs"
#     log_dir.mkdir(exist_ok=True)

#     effective_n_workers = n_workers if n_workers is not None else cpus

#     # Split into at most n_jobs chunks (may be fewer if n_jobs > len(needs_rerun))
#     actual_n_jobs = min(n_jobs, len(needs_rerun))
#     chunks = _split_indices(len(needs_rerun), actual_n_jobs)
#     # _split_indices works on range(n), remap to actual indices
#     chunks = [[needs_rerun[i] for i in chunk] for chunk in chunks]

#     # Start job_id counter after existing scripts to avoid collisions
#     existing = sorted(log_dir.glob("job_*.sh"))
#     base_id = len(existing)

#     slurm_ids = []
#     for offset, chunk in enumerate(chunks):
#         slurm_id = _submit_job(
#             base_id + offset,
#             chunk,
#             master_file=master_file,
#             output_file=output_file,
#             poni_file=poni_file,
#             mask_file=mask_file,
#             n_points=n_points,
#             n_workers=effective_n_workers,
#             batch_size=batch_size,
#             unit=unit,
#             method=method,
#             percentile=percentile,
#             thres=thres,
#             max_iter=max_iter,
#             partition=partition,
#             time=time,
#             mem=mem,
#             cpus=cpus,
#             gpu=gpu,
#             env_activate=env_activate,
#             conda_env=conda_env,
#             log_dir=log_dir,
#         )
#         slurm_ids.append(slurm_id)

#     return slurm_ids


# # ─────────────────────────────────────────────────────────────────────────────
# # Public API
# # ─────────────────────────────────────────────────────────────────────────────


# def check(
#     output_file: Path,
#     *,
#     resubmit: bool = False,
#     repair: bool = False,
#     master_file: Path | None = None,
#     poni_file: Path | None = None,
#     mask_file: Path | None = None,
#     n_jobs: int = 1,
#     n_points: int = 1000,
#     n_workers: int | None = None,
#     batch_size: int = 32,
#     unit: str = "2th_deg",
#     method: str = "standard",
#     percentile: tuple = (10, 90),
#     thres: float = 3.0,
#     max_iter: int = 5,
#     partition: str = "cpu",
#     time: str = "04:00:00",
#     mem: str = "32G",
#     cpus: int = 16,
#     gpu: bool = False,
#     env_activate: Path | None = None,
#     conda_env: str | None = None,
#     watch: bool = False,
#     interval: int = 30,
# ) -> dict:
#     """
#     Verify completeness of the output HDF5 file.

#     Parameters
#     ----------
#     output_file : Path
#     resubmit    : bool
#         Print the ``--entry-indices`` hint needed to rerun missing/corrupted
#         scans manually.
#     repair      : bool
#         Automatically delete corrupted datasets and submit SLURM jobs to
#         reintegrate all missing and corrupted scans.
#         Requires master_file, poni_file, and mask_file.
#     n_jobs      : int
#         Number of SLURM jobs to split the repair work across (default: 1).
#         Useful when many scans need reintegrating after a rebuild.
#     watch       : bool
#         If True, block after submitting repair jobs and poll until they all
#         finish (passed to ``monitor()``). Default: False.
#     interval    : int
#         Polling interval in seconds when watch=True (default: 30).

#     Returns
#     -------
#     dict with keys 'n_expected', 'present', 'missing', 'corrupted',
#     'nan_scans', and (if repair=True) 'repair_job_ids'.
#     """
#     output_file = Path(output_file)

#     if repair and not all([master_file, poni_file, mask_file]):
#         raise ValueError("repair=True requires master_file, poni_file, and mask_file.")

#     # ── 1. Scan the output file ───────────────────────────────────────────────
#     # A truncated file (eof < stored_eof) crashes on open — catch it early
#     # and tell the user to call rebuild() instead of check().
#     try:
#         hout_handle = h5py.File(output_file, "r")
#     except OSError as e:
#         msg = str(e)
#         if "truncated" in msg.lower() or "stored_eof" in msg.lower():
#             raise OSError(
#                 f"\n\nThe output file is truncated (a job was killed mid-write):\n"
#                 f"  {output_file}\n\n"
#                 f"The file cannot be opened for reading. Run rebuild() to recover:\n\n"
#                 f"  from nrxrdct.slurm_integration import rebuild\n"
#                 f"  rebuild(\n"
#                 f"      output_file = Path('{output_file}'),\n"
#                 f"      master_file = Path('<master.h5>'),\n"
#                 f"      poni_file   = Path('<calib.poni>'),\n"
#                 f"      mask_file   = Path('<mask.edf>'),\n"
#                 f"      n_jobs      = 4,\n"
#                 f"      watch       = True,\n"
#                 f"  )\n"
#             ) from e
#         raise  # re-raise unrelated OSErrors unchanged

#     with hout_handle as hout:
#         if "meta/valid_entries" not in hout:
#             raise RuntimeError(
#                 f"'meta/valid_entries' not found in {output_file}. "
#                 "Was the output file initialised by launch_jobs?"
#             )

#         valid_entries = [
#             e.decode() if isinstance(e, bytes) else e
#             for e in hout["meta/valid_entries"][:]
#         ]
#         n_expected = len(valid_entries)

#         present, missing, nan_scans, corrupted = [], [], [], []

#         for ii, entry in enumerate(valid_entries):
#             scan_name = f"scan_{ii:04d}"
#             group_path = f"integrated/{scan_name}"

#             # Membership check — RuntimeError fires when the HDF5 link/cache
#             # table itself is damaged ("incorrect cache entry type").
#             try:
#                 exists = group_path in hout
#             except (RuntimeError, OSError, KeyError) as e:
#                 print(f"  ✗ {scan_name} (index {ii}): corrupted link — {e}")
#                 corrupted.append(ii)
#                 continue

#             if not exists:
#                 missing.append(ii)
#                 continue

#             # Data read — OSError fires when gzip chunks are truncated
#             # (OOM-killed mid-write).
#             try:
#                 data = hout[group_path][:]
#                 nan_rows = int(np.all(np.isnan(data), axis=1).sum())
#                 present.append((ii, scan_name, nan_rows, data.shape))
#                 if nan_rows:
#                     nan_scans.append((ii, scan_name, nan_rows, data.shape))
#             except (RuntimeError, OSError, KeyError) as e:
#                 print(f"  ✗ {scan_name} (index {ii}): corrupted dataset — {e}")
#                 corrupted.append(ii)

#         has_radial = "integrated/radial" in hout
#         radial_info = (
#             (
#                 hout["integrated/radial"].shape[0],
#                 hout["integrated/radial"].attrs.get("unit", "unknown"),
#             )
#             if has_radial
#             else None
#         )

#     # ── 2. Report ─────────────────────────────────────────────────────────────
#     needs_rerun = sorted(missing + corrupted)

#     print(f"\n{'='*60}")
#     print(f"Output file : {output_file}")
#     print(f"Expected    : {n_expected} scans")
#     print(f"Present     : {len(present)}")
#     print(f"Missing     : {len(missing)}")
#     print(f"Corrupted   : {len(corrupted)}")
#     print(f"{'='*60}\n")

#     if not needs_rerun:
#         print("✓  All scans present and readable!")
#     else:
#         if missing:
#             print(f"⚠  Missing scan indices  : {missing}")
#         if corrupted:
#             print(f"⚠  Corrupted scan indices: {corrupted}")
#             print("   (likely OOM-killed mid-write)")

#     if nan_scans:
#         print(f"\n⚠  {len(nan_scans)} scans have fully-NaN frames:")
#         for ii, name, nan, shape in nan_scans:
#             print(f"   {name} (index {ii}): {nan}/{shape[0]} NaN frames")

#     if radial_info:
#         print(f"\n✓  Radial axis: {radial_info[0]} points, unit={radial_info[1]}")
#     else:
#         print("\n✗  'integrated/radial' missing!")

#     # ── 3a. Resubmit hint (manual) ────────────────────────────────────────────
#     if resubmit and needs_rerun and not repair:
#         idx_str = ",".join(str(i) for i in needs_rerun)
#         print(f"\nResubmit hint:")
#         if corrupted:
#             print(
#                 textwrap.dedent(
#                     f"""
#   # Step 1 — delete corrupted datasets
#   python - <<'EOF'
#   import h5py
#   with h5py.File("{output_file}", "a") as h:
#       for i in {corrupted}:
#           path = f"integrated/scan_{{i:04d}}"
#           if path in h:
#               del h[path]; print(f"Deleted {{path}}")
#   EOF"""
#                 )
#             )
#         print(
#             f"\n  # Step 2 — rerun worker\n"
#             f"  python -m nrxrdct.slurm_integration.integrate_worker \\\n"
#             f'      --output-file  "{output_file}" \\\n'
#             f'      --entry-indices "{idx_str}" \\\n'
#             f"      --master-file <...> --poni-file <...> --mask-file <...>"
#         )

#     # ── 3b. Auto-repair ───────────────────────────────────────────────────────
#     repair_job_ids = []
#     if repair and needs_rerun:
#         print(
#             f"\n🔧  Repair mode: fixing {len(needs_rerun)} scans "
#             f"(missing={len(missing)}, corrupted={len(corrupted)})"
#         )

#         # Delete corrupted datasets so the worker can rewrite them
#         if corrupted:
#             print("  Deleting corrupted datasets...")
#             _delete_datasets(output_file, corrupted)

#         # Submit repair jobs — split across n_jobs if requested
#         print(f"  Submitting {n_jobs} repair job(s) for {len(needs_rerun)} scans...")
#         repair_job_ids = _resubmit(
#             output_file=output_file,
#             needs_rerun=needs_rerun,
#             master_file=Path(master_file),
#             poni_file=Path(poni_file),
#             mask_file=Path(mask_file),
#             n_jobs=n_jobs,
#             n_points=n_points,
#             n_workers=n_workers,
#             batch_size=batch_size,
#             unit=unit,
#             method=method,
#             percentile=f"{percentile[0]},{percentile[1]}",
#             thres=thres,
#             max_iter=max_iter,
#             partition=partition,
#             time=time,
#             mem=mem,
#             cpus=cpus,
#             gpu=gpu,
#             env_activate=env_activate,
#             conda_env=conda_env,
#         )
#         print(
#             f"\n✓  {len(repair_job_ids)} repair job(s) submitted — "
#             f"SLURM IDs: {', '.join(repair_job_ids)}"
#         )
#         print(f"   Re-run check() after they complete to verify.")

#         if watch:
#             from .monitor import monitor as _monitor

#             _monitor(
#                 slurm_ids=repair_job_ids,
#                 output_file=output_file,
#                 watch=True,
#                 interval=interval,
#             )

#     elif repair and not needs_rerun:
#         print("\n✓  Nothing to repair.")

#     result = {
#         "n_expected": n_expected,
#         "present": present,
#         "missing": missing,
#         "corrupted": corrupted,
#         "nan_scans": nan_scans,
#         "needs_rerun": needs_rerun,
#     }
#     if repair:
#         result["repair_job_ids"] = repair_job_ids if needs_rerun else []
#     return result


# # ─────────────────────────────────────────────────────────────────────────────
# # Convenience wrapper
# # ─────────────────────────────────────────────────────────────────────────────


# def repair(
#     output_file: Path,
#     master_file: Path,
#     poni_file: Path,
#     mask_file: Path,
#     *,
#     n_jobs: int = 1,
#     watch: bool = False,
#     interval: int = 30,
#     **kwargs,
# ) -> dict:
#     """
#     Shorthand for ``check(..., repair=True)``.

#     Detects missing and corrupted scans, deletes the broken HDF5 datasets,
#     and submits SLURM repair jobs — all in one call.

#     Parameters
#     ----------
#     n_jobs   : int
#         Number of SLURM jobs to split the repair work across (default: 1).
#     watch    : bool
#         If True, block until all repair jobs finish (default: False).
#     interval : int
#         Polling interval in seconds when watch=True (default: 30).

#     Example
#     -------
#         from nrxrdct.slurm_integration import repair

#         repair(
#             output_file = Path("output.h5"),
#             master_file = Path("master.h5"),
#             poni_file   = Path("calib.poni"),
#             mask_file   = Path("mask.edf"),
#             n_jobs      = 4,
#             watch       = True,
#             partition   = "cpu",
#             conda_env   = "nrxrdct",
#         )
#     """
#     return check(
#         output_file=output_file,
#         repair=True,
#         master_file=master_file,
#         poni_file=poni_file,
#         mask_file=mask_file,
#         n_jobs=n_jobs,
#         watch=watch,
#         interval=interval,
#         **kwargs,
#     )


# def rebuild(
#     output_file: Path,
#     master_file: Path,
#     poni_file: Path,
#     mask_file: Path,
#     *,
#     rebuilt_file: Path | None = None,
#     n_jobs: int = 1,
#     watch: bool = False,
#     interval: int = 30,
#     **kwargs,
# ) -> dict:
#     """
#     Rebuild a deeply corrupted output HDF5 file.

#     When the B-tree/symbol table of the output file is damaged, even writing
#     new datasets fails.  This function:

#     1. Copies all *readable* scan datasets into a fresh HDF5 file.
#     2. Calls ``repair()`` on the new file to submit a SLURM job for the
#        remaining missing/corrupted scans.

#     The original file is renamed to ``<name>.bak`` and the rebuilt file
#     takes its place, so all downstream code can keep using the original path.

#     Parameters
#     ----------
#     output_file  : Path  — the corrupted output file
#     master_file  : Path  — original master HDF5
#     poni_file    : Path  — calibration file
#     mask_file    : Path  — mask file
#     rebuilt_file : Path  — destination for the rebuilt file (default:
#                            ``<output_file>.rebuilt`` while building, then
#                            swapped in place of ``output_file``)
#     **kwargs     : forwarded to ``repair()`` (partition, conda_env, etc.)

#     Returns
#     -------
#     dict — same as ``repair()`` called on the rebuilt file
#     """
#     output_file = Path(output_file)
#     tmp_file = (
#         Path(rebuilt_file) if rebuilt_file else output_file.with_suffix(".rebuilt.h5")
#     )

#     print(f"\n{'='*60}")
#     print(f"Rebuilding {output_file.name} → {tmp_file.name}")
#     print(f"{'='*60}\n")

#     # ── 1. Open the corrupted source — handle truncated files ─────────────────
#     # A truncated file crashes on open. In that case we can't recover any scan
#     # data, so we create a fresh output file and reintegrate everything from
#     # the master. The metadata (valid_entries, dty, rot, radial) is re-read
#     # from the master file via launch_jobs._init_output_file.
#     try:
#         src_handle = h5py.File(output_file, "r")
#         src_opened = True
#     except OSError as e:
#         if "truncated" in str(e).lower() or "stored_eof" in str(e).lower():
#             print(f"  ⚠  Source file is truncated and cannot be opened.")
#             print(
#                 f"     No scan data can be recovered — all scans will be reintegrated."
#             )
#             src_opened = False
#         else:
#             raise

#     if src_opened:
#         with src_handle as src:
#             valid_entries = [
#                 e.decode() if isinstance(e, bytes) else e
#                 for e in src["meta/valid_entries"][:]
#             ]
#             n_total = len(valid_entries)

#             # ── 2a. Create fresh destination file from readable source ─────────
#             with h5py.File(tmp_file, "w") as dst:

#                 # Copy all non-scan groups verbatim
#                 for key in src.keys():
#                     if key != "integrated":
#                         try:
#                             src.copy(key, dst)
#                         except Exception as e:
#                             print(f"  ⚠  Could not copy group '{key}': {e}")

#                 dst.require_group("integrated")

#                 # Copy integrated/radial and integrated/cake_mask explicitly
#                 for special in ("integrated/radial", "integrated/cake_mask"):
#                     try:
#                         exists = special in src
#                     except (RuntimeError, OSError, KeyError):
#                         exists = False
#                     if exists:
#                         try:
#                             parent_dst = dst.require_group("integrated")
#                             name = special.split("/")[-1]
#                             src.copy(special, parent_dst, name=name)
#                         except Exception as e:
#                             print(f"  ⚠  Could not copy '{special}': {e}")

#                 # Copy readable scan datasets
#                 n_copied = n_skipped = 0
#                 for ii in range(n_total):
#                     scan_name = f"scan_{ii:04d}"
#                     group_path = f"integrated/{scan_name}"

#                     try:
#                         exists = group_path in src
#                     except (RuntimeError, OSError, KeyError):
#                         exists = False

#                     if not exists:
#                         n_skipped += 1
#                         continue

#                     try:
#                         data = src[group_path][:]
#                         ds = dst.create_dataset(
#                             group_path,
#                             data=data,
#                             compression="gzip",
#                             compression_opts=4,
#                             chunks=(1, data.shape[1]),
#                         )
#                         for k, v in src[group_path].attrs.items():
#                             ds.attrs[k] = v
#                         n_copied += 1
#                     except (RuntimeError, OSError, ValueError, KeyError) as e:
#                         print(f"  ✗  {scan_name}: unreadable — {e}")
#                         n_skipped += 1

#         print(f"\n✓  Copied {n_copied}/{n_total} scans into {tmp_file.name}")
#         print(f"   Skipped {n_skipped} corrupted/missing scans (will be reintegrated)")

#     else:
#         # ── 2b. Truncated — initialise a blank output file from master ─────────
#         import fabio
#         import numpy as np

#         from .launch_jobs import _init_output_file, _validate_entries

#         print("  Validating master file to rebuild metadata...")
#         valid_entries, bad_entries, dty_values = _validate_entries(Path(master_file))
#         n_total = len(valid_entries)

#         with h5py.File(master_file, "r") as hin:
#             rot = hin[f"{valid_entries[0]}/measurement/rot"][:]

#         _init_output_file(
#             master_file=Path(master_file),
#             output_file=tmp_file,
#             poni_file=Path(poni_file),
#             mask_file=Path(mask_file),
#             valid_entries=valid_entries,
#             bad_entries=bad_entries,
#             dty_values=dty_values,
#             rot=rot,
#             n_points=kwargs.get("n_points", 1000),
#             unit=kwargs.get("unit", "2th_deg"),
#         )
#         n_copied = 0
#         n_skipped = n_total
#         print(
#             f"\n✓  Fresh output file initialised — all {n_total} scans will be reintegrated"
#         )

#     # ── 3. Swap files: original → .bak, rebuilt → original path ──────────────
#     bak_file = output_file.with_suffix(".bak.h5")
#     output_file.rename(bak_file)
#     tmp_file.rename(output_file)
#     print(f"\n  Original backed up → {bak_file.name}")
#     print(f"  Rebuilt file       → {output_file.name}")

#     # ── 4. Run repair on the clean rebuilt file ───────────────────────────────
#     print(f"\nRunning repair on rebuilt file...")
#     return repair(
#         output_file=output_file,
#         master_file=master_file,
#         poni_file=poni_file,
#         mask_file=mask_file,
#         n_jobs=n_jobs,
#         watch=watch,
#         interval=interval,
#         **kwargs,
#     )


# # ─────────────────────────────────────────────────────────────────────────────
# # CLI
# # ─────────────────────────────────────────────────────────────────────────────


# def _build_parser(sub=None):
#     import argparse

#     desc = "Verify (and optionally repair) the output HDF5 file"
#     p = (
#         sub.add_parser("check", help=desc, description=desc)
#         if sub
#         else argparse.ArgumentParser(description=desc)
#     )

#     p.add_argument("--output-file", required=True, type=Path)
#     p.add_argument(
#         "--resubmit",
#         action="store_true",
#         help="Print manual resubmit hints for missing/corrupted scans",
#     )

#     repair = p.add_argument_group("repair mode (auto-fix)")
#     repair.add_argument(
#         "--repair",
#         action="store_true",
#         help="Delete corrupted datasets and submit repair SLURM job(s)",
#     )
#     repair.add_argument(
#         "--n-jobs",
#         type=int,
#         default=1,
#         help="Number of SLURM jobs to split repair work across (default: 1)",
#     )
#     repair.add_argument("--master-file", type=Path, default=None)
#     repair.add_argument("--poni-file", type=Path, default=None)
#     repair.add_argument("--mask-file", type=Path, default=None)
#     repair.add_argument("--n-points", type=int, default=1000)
#     repair.add_argument("--n-workers", type=int, default=None)
#     repair.add_argument("--batch-size", type=int, default=32)
#     repair.add_argument("--unit", default="2th_deg")
#     repair.add_argument(
#         "--method",
#         default="standard",
#         choices=("standard", "filter", "sigma_clip"),
#         help="Integration method (default: standard)",
#     )
#     repair.add_argument(
#         "--percentile", default="10,90", help="Low,high percentile for 'filter' method"
#     )
#     repair.add_argument(
#         "--thres",
#         type=float,
#         default=3.0,
#         help="Sigma threshold for 'sigma_clip' method",
#     )
#     repair.add_argument(
#         "--max-iter", type=int, default=5, help="Max iterations for 'sigma_clip' method"
#     )
#     repair.add_argument("--partition", default="cpu")
#     repair.add_argument("--time", default="04:00:00")
#     repair.add_argument("--mem", default="32G")
#     repair.add_argument("--cpus", type=int, default=16)
#     repair.add_argument("--gpu", action="store_true")
#     repair.add_argument("--env-activate", type=Path, default=None)
#     repair.add_argument("--conda-env", default=None)
#     repair.add_argument(
#         "--watch",
#         action="store_true",
#         help="Block until repair jobs finish (calls monitor internally)",
#     )
#     repair.add_argument(
#         "--interval",
#         type=int,
#         default=30,
#         help="Polling interval in seconds when --watch is set (default: 30)",
#     )
#     return p


# def _cli_check(args):
#     pct = tuple(int(x) for x in args.percentile.split(","))
#     check(
#         output_file=args.output_file,
#         resubmit=args.resubmit,
#         repair=args.repair,
#         master_file=args.master_file,
#         poni_file=args.poni_file,
#         mask_file=args.mask_file,
#         n_jobs=args.n_jobs,
#         n_points=args.n_points,
#         n_workers=args.n_workers,
#         batch_size=args.batch_size,
#         unit=args.unit,
#         method=args.method,
#         percentile=pct,
#         thres=args.thres,
#         max_iter=args.max_iter,
#         partition=args.partition,
#         time=args.time,
#         mem=args.mem,
#         cpus=args.cpus,
#         gpu=args.gpu,
#         env_activate=args.env_activate,
#         conda_env=args.conda_env,
#         watch=args.watch,
#         interval=args.interval,
#     )


# if __name__ == "__main__":
#     p = _build_parser()
#     a = p.parse_args()
#     _cli_check(a)
