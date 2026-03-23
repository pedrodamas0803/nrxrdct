"""
nrxrdct.slurm_integration.check_output
----------------------------------------
Verify completeness of the output HDF5 file after all SLURM jobs finish.

Python API
----------
    from nrxrdct.slurm_integration import check

    # Just report
    result = check(output_file=Path("output.h5"))

    # Report + print resubmit hints
    result = check(output_file=Path("output.h5"), resubmit=True)

    # Report + automatically delete corrupted/missing datasets and resubmit
    result = check(
        output_file  = Path("output.h5"),
        repair       = True,
        master_file  = Path("master.h5"),
        poni_file    = Path("calib.poni"),
        mask_file    = Path("mask.edf"),
        partition    = "cpu",
        conda_env    = "nrxrdct",
    )

CLI
---
    nrxrdct-slurm check --output-file output.h5 [--resubmit]
    nrxrdct-slurm check --output-file output.h5 --repair \\
        --master-file master.h5 --poni-file calib.poni --mask-file mask.edf
"""

from __future__ import annotations

import subprocess
import textwrap
from pathlib import Path

import h5py
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _delete_datasets(output_file: Path, indices: list[int]) -> None:
    """Delete the HDF5 datasets for the given scan indices (in-place)."""
    with h5py.File(output_file, "a") as h:
        for i in indices:
            path = f"integrated/scan_{i:04d}"
            if path in h:
                del h[path]
                print(f"  🗑  Deleted {path}")


def _resubmit(
    output_file: Path,
    needs_rerun: list[int],
    *,
    master_file: Path,
    poni_file: Path,
    mask_file: Path,
    n_points: int,
    n_workers: int | None,
    batch_size: int,
    unit: str,
    method: str,
    percentile: str,
    thres: float,
    max_iter: int,
    partition: str,
    time: str,
    mem: str,
    cpus: int,
    gpu: bool,
    env_activate: Path | None,
    conda_env: str | None,
) -> str:
    """Submit a single sbatch repair job covering all needs_rerun indices."""
    from .launch_jobs import _submit_job

    log_dir = output_file.parent / "slurm_logs"
    log_dir.mkdir(exist_ok=True)

    existing            = sorted(log_dir.glob("job_*.sh"))
    job_id              = len(existing)
    effective_n_workers = n_workers if n_workers is not None else cpus

    slurm_id = _submit_job(
        job_id,
        needs_rerun,
        master_file  = master_file,
        output_file  = output_file,
        poni_file    = poni_file,
        mask_file    = mask_file,
        n_points     = n_points,
        n_workers    = effective_n_workers,
        batch_size   = batch_size,
        unit         = unit,
        method       = method,
        percentile   = percentile,
        thres        = thres,
        max_iter     = max_iter,
        partition    = partition,
        time         = time,
        mem          = mem,
        cpus         = cpus,
        gpu          = gpu,
        env_activate = env_activate,
        conda_env    = conda_env,
        log_dir      = log_dir,
    )
    return slurm_id


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def check(
    output_file: Path,
    *,
    resubmit: bool = False,
    repair: bool = False,
    master_file: Path | None = None,
    poni_file: Path | None = None,
    mask_file: Path | None = None,
    n_points: int = 1000,
    n_workers: int | None = None,
    batch_size: int = 32,
    unit: str = "2th_deg",
    method: str = "standard",
    percentile: tuple = (10, 90),
    thres: float = 3.0,
    max_iter: int = 5,
    partition: str = "cpu",
    time: str = "04:00:00",
    mem: str = "32G",
    cpus: int = 16,
    gpu: bool = False,
    env_activate: Path | None = None,
    conda_env: str | None = None,
) -> dict:
    """
    Verify completeness of the output HDF5 file.

    Parameters
    ----------
    output_file : Path
    resubmit    : bool
        Print the ``--entry-indices`` hint needed to rerun missing/corrupted
        scans manually.
    repair      : bool
        Automatically delete corrupted datasets and submit a new SLURM job
        to reintegrate all missing and corrupted scans.
        Requires master_file, poni_file, and mask_file.

    Returns
    -------
    dict with keys 'n_expected', 'present', 'missing', 'corrupted',
    'nan_scans', and (if repair=True) 'repair_job_id'.
    """
    output_file = Path(output_file)

    if repair and not all([master_file, poni_file, mask_file]):
        raise ValueError(
            "repair=True requires master_file, poni_file, and mask_file."
        )

    # ── 1. Scan the output file ───────────────────────────────────────────────
    with h5py.File(output_file, "r") as hout:
        if "meta/valid_entries" not in hout:
            raise RuntimeError(
                f"'meta/valid_entries' not found in {output_file}. "
                "Was the output file initialised by launch_jobs?"
            )

        valid_entries = [
            e.decode() if isinstance(e, bytes) else e
            for e in hout["meta/valid_entries"][:]
        ]
        n_expected = len(valid_entries)

        present, missing, nan_scans, corrupted = [], [], [], []

        for ii, entry in enumerate(valid_entries):
            scan_name  = f"scan_{ii:04d}"
            group_path = f"integrated/{scan_name}"

            if group_path not in hout:
                missing.append(ii)
                continue

            try:
                data     = hout[group_path][:]
                nan_rows = int(np.all(np.isnan(data), axis=1).sum())
                present.append((ii, scan_name, nan_rows, data.shape))
                if nan_rows:
                    nan_scans.append((ii, scan_name, nan_rows, data.shape))
            except OSError as e:
                print(f"  ✗ {scan_name} (index {ii}): corrupted — {e}")
                corrupted.append(ii)

        has_radial = "integrated/radial" in hout
        radial_info = (
            (hout["integrated/radial"].shape[0],
             hout["integrated/radial"].attrs.get("unit", "unknown"))
            if has_radial else None
        )

    # ── 2. Report ─────────────────────────────────────────────────────────────
    needs_rerun = sorted(missing + corrupted)

    print(f"\n{'='*60}")
    print(f"Output file : {output_file}")
    print(f"Expected    : {n_expected} scans")
    print(f"Present     : {len(present)}")
    print(f"Missing     : {len(missing)}")
    print(f"Corrupted   : {len(corrupted)}")
    print(f"{'='*60}\n")

    if not needs_rerun:
        print("✓  All scans present and readable!")
    else:
        if missing:
            print(f"⚠  Missing scan indices  : {missing}")
        if corrupted:
            print(f"⚠  Corrupted scan indices: {corrupted}")
            print( "   (likely OOM-killed mid-write)")

    if nan_scans:
        print(f"\n⚠  {len(nan_scans)} scans have fully-NaN frames:")
        for ii, name, nan, shape in nan_scans:
            print(f"   {name} (index {ii}): {nan}/{shape[0]} NaN frames")

    if radial_info:
        print(f"\n✓  Radial axis: {radial_info[0]} points, unit={radial_info[1]}")
    else:
        print("\n✗  'integrated/radial' missing!")

    # ── 3a. Resubmit hint (manual) ────────────────────────────────────────────
    if resubmit and needs_rerun and not repair:
        idx_str = ",".join(str(i) for i in needs_rerun)
        print(f"\nResubmit hint:")
        if corrupted:
            print(textwrap.dedent(f"""
  # Step 1 — delete corrupted datasets
  python - <<'EOF'
  import h5py
  with h5py.File("{output_file}", "a") as h:
      for i in {corrupted}:
          path = f"integrated/scan_{{i:04d}}"
          if path in h:
              del h[path]; print(f"Deleted {{path}}")
  EOF"""))
        print(
            f"\n  # Step 2 — rerun worker\n"
            f"  python -m nrxrdct.slurm_integration.integrate_worker \\\n"
            f'      --output-file  "{output_file}" \\\n'
            f'      --entry-indices "{idx_str}" \\\n'
            f"      --master-file <...> --poni-file <...> --mask-file <...>"
        )

    # ── 3b. Auto-repair ───────────────────────────────────────────────────────
    repair_job_id = None
    if repair and needs_rerun:
        print(f"\n🔧  Repair mode: fixing {len(needs_rerun)} scans "
              f"(missing={len(missing)}, corrupted={len(corrupted)})")

        # Delete corrupted datasets so the worker can rewrite them
        if corrupted:
            print("  Deleting corrupted datasets...")
            _delete_datasets(output_file, corrupted)

        # Submit one repair job covering everything that needs rerunning
        print(f"  Submitting repair job for indices: {needs_rerun}")
        repair_job_id = _resubmit(
            output_file  = output_file,
            needs_rerun  = needs_rerun,
            master_file  = Path(master_file),
            poni_file    = Path(poni_file),
            mask_file    = Path(mask_file),
            n_points     = n_points,
            n_workers    = n_workers,
            batch_size   = batch_size,
            unit         = unit,
            method       = method,
            percentile   = f"{percentile[0]},{percentile[1]}",
            thres        = thres,
            max_iter     = max_iter,
            partition    = partition,
            time         = time,
            mem          = mem,
            cpus         = cpus,
            gpu          = gpu,
            env_activate = env_activate,
            conda_env    = conda_env,
        )
        print(f"\n✓  Repair job submitted — SLURM ID: {repair_job_id}")
        print(f"   Re-run check() after it completes to verify.")

    elif repair and not needs_rerun:
        print("\n✓  Nothing to repair.")

    result = {
        "n_expected":    n_expected,
        "present":       present,
        "missing":       missing,
        "corrupted":     corrupted,
        "nan_scans":     nan_scans,
        "needs_rerun":   needs_rerun,
    }
    if repair:
        result["repair_job_id"] = repair_job_id
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Convenience wrapper
# ─────────────────────────────────────────────────────────────────────────────

def repair(
    output_file: Path,
    master_file: Path,
    poni_file: Path,
    mask_file: Path,
    **kwargs,
) -> dict:
    """
    Shorthand for ``check(..., repair=True)``.

    Detects missing and corrupted scans, deletes the broken HDF5 datasets,
    and submits a single SLURM repair job — all in one call.

    Example
    -------
        from nrxrdct.slurm_integration import repair

        repair(
            output_file = Path("output.h5"),
            master_file = Path("master.h5"),
            poni_file   = Path("calib.poni"),
            mask_file   = Path("mask.edf"),
            partition   = "cpu",
            conda_env   = "nrxrdct",
        )
    """
    return check(
        output_file = output_file,
        repair      = True,
        master_file = master_file,
        poni_file   = poni_file,
        mask_file   = mask_file,
        **kwargs,
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser(sub=None):
    import argparse
    desc = "Verify (and optionally repair) the output HDF5 file"
    p = sub.add_parser("check", help=desc, description=desc) if sub else argparse.ArgumentParser(description=desc)

    p.add_argument("--output-file",  required=True, type=Path)
    p.add_argument("--resubmit",     action="store_true",
                   help="Print manual resubmit hints for missing/corrupted scans")

    repair = p.add_argument_group("repair mode (auto-fix)")
    repair.add_argument("--repair",       action="store_true",
                        help="Delete corrupted datasets and submit a repair SLURM job")
    repair.add_argument("--master-file",  type=Path, default=None)
    repair.add_argument("--poni-file",    type=Path, default=None)
    repair.add_argument("--mask-file",    type=Path, default=None)
    repair.add_argument("--n-points",     type=int,  default=1000)
    repair.add_argument("--n-workers",    type=int,  default=None)
    repair.add_argument("--batch-size",   type=int,  default=32)
    repair.add_argument("--unit",         default="2th_deg")
    repair.add_argument("--method",       default="standard",
                        choices=("standard", "filter", "sigma_clip"),
                        help="Integration method (default: standard)")
    repair.add_argument("--percentile",   default="10,90",
                        help="Low,high percentile for 'filter' method")
    repair.add_argument("--thres",        type=float, default=3.0,
                        help="Sigma threshold for 'sigma_clip' method")
    repair.add_argument("--max-iter",     type=int,   default=5,
                        help="Max iterations for 'sigma_clip' method")
    repair.add_argument("--partition",    default="cpu")
    repair.add_argument("--time",         default="04:00:00")
    repair.add_argument("--mem",          default="32G")
    repair.add_argument("--cpus",         type=int, default=16)
    repair.add_argument("--gpu",          action="store_true")
    repair.add_argument("--env-activate", type=Path, default=None)
    repair.add_argument("--conda-env",    default=None)
    return p


def _cli_check(args):
    pct = tuple(int(x) for x in args.percentile.split(","))
    check(
        output_file  = args.output_file,
        resubmit     = args.resubmit,
        repair       = args.repair,
        master_file  = args.master_file,
        poni_file    = args.poni_file,
        mask_file    = args.mask_file,
        n_points     = args.n_points,
        n_workers    = args.n_workers,
        batch_size   = args.batch_size,
        unit         = args.unit,
        method       = args.method,
        percentile   = pct,
        thres        = args.thres,
        max_iter     = args.max_iter,
        partition    = args.partition,
        time         = args.time,
        mem          = args.mem,
        cpus         = args.cpus,
        gpu          = args.gpu,
        env_activate = args.env_activate,
        conda_env    = args.conda_env,
    )


if __name__ == "__main__":
    p = _build_parser()
    a = p.parse_args()
    _cli_check(a)