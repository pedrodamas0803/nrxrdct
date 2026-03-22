"""
nrxrdct.slurm_integration.launch_jobs
--------------------------------------
Validates master HDF5 entries, initialises the output file, splits scans into
N chunks, and submits one sbatch job per chunk.

Python API
----------
    from nrxrdct.slurm_integration import launch

    launch(
        master_file  = Path("master.h5"),
        output_file  = Path("output.h5"),
        poni_file    = Path("calib.poni"),
        mask_file    = Path("mask.edf"),
        n_jobs       = 8,
        n_workers    = 16,
        partition    = "cpu",
        time         = "04:00:00",
        mem          = "64G",
        cpus         = 16,
        conda_env    = "nrxrdct",   # or env_activate=Path(...)
    )

CLI (registered as 'nrxrdct-slurm launch')
-------------------------------------------
    nrxrdct-slurm launch \\
        --master-file master.h5 --output-file output.h5 \\
        --poni-file calib.poni  --mask-file mask.edf   \\
        --n-jobs 8 --partition cpu --conda-env nrxrdct
"""

from __future__ import annotations

import math
import subprocess
import sys
import textwrap
from pathlib import Path

import h5py
import numpy as np
import fabio
from tqdm import tqdm

from nrxrdct.integration import azimuthal_integration_1d, cake_integration


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _validate_entries(master_file: Path) -> tuple[list, list, list]:
    valid_entries, bad_entries, dty_values = [], [], []
    with h5py.File(master_file, "r") as hin:
        all_entries = list(hin.keys())
        for entry in tqdm(all_entries, desc="Validating entries"):
            try:
                _ = hin[f"{entry}/measurement/eiger"].shape
                _ = hin[f"{entry}/measurement/fpico6"].shape
                dty = float(hin[f"{entry}/instrument/positioners/dty"][()])
                valid_entries.append(entry)
                dty_values.append(dty)
            except KeyError as e:
                print(f"  ⚠  Entry {entry} missing dataset ({e}) — skipping")
                bad_entries.append(entry)

    print(f"\n✓  {len(valid_entries)}/{len(all_entries)} entries OK")
    if bad_entries:
        print(f"⚠  Skipping {len(bad_entries)} entries: {bad_entries}\n")
    return valid_entries, bad_entries, dty_values


def _init_output_file(
    master_file: Path,
    output_file: Path,
    poni_file: Path,
    mask_file: Path,
    valid_entries: list,
    bad_entries: list,
    dty_values: list,
    rot: np.ndarray,
    n_points: int,
    unit: str,
) -> None:
    mask = fabio.open(mask_file).data
    with h5py.File(master_file, "r") as hin, h5py.File(output_file, "a") as hout:
        if "integrated/radial" not in hout:
            first_image = hin[f"{valid_entries[0]}/measurement/eiger"][0].astype(np.float32)
            tt, _, _ = azimuthal_integration_1d(
                image=first_image, poni_file=poni_file,
                npt=n_points, mask=mask, unit=unit,
            )
            mascake = (
                cake_integration(
                    np.ones_like(first_image) * 10,
                    poni_file, npt_rad=n_points, mask=mask,
                )[0] > 0
            )
            hout["integrated/cake_mask"] = mascake
            hout["integrated/radial"]    = tt
            hout["integrated/radial"].attrs["unit"] = unit

        if "motors/dty" not in hout:
            hout["motors/dty"] = dty_values
        if "motors/rot" not in hout:
            hout["motors/rot"] = rot
        if "meta/valid_entries" not in hout:
            hout["meta/valid_entries"] = np.array(valid_entries, dtype=h5py.string_dtype())
        if bad_entries and "bad_entries" not in hout:
            hout["bad_entries"] = np.array(bad_entries, dtype=h5py.string_dtype())

    print(f"✓  Output file initialised: {output_file}")


def _split_indices(n_scans: int, n_jobs: int) -> list[list[int]]:
    all_idx    = list(range(n_scans))
    chunk_size = math.ceil(n_scans / n_jobs)
    chunks = [all_idx[i : i + chunk_size] for i in range(0, n_scans, chunk_size)]
    print(f"✓  {n_scans} scans → {len(chunks)} jobs (~{chunk_size} scans each)")
    return chunks


def _submit_job(
    job_id: int,
    indices: list[int],
    *,
    master_file: Path,
    output_file: Path,
    poni_file: Path,
    mask_file: Path,
    n_points: int,
    n_workers: int,
    unit: str,
    remove_spots: bool,
    percentile: str,
    partition: str,
    time: str,
    mem: str,
    cpus: int,
    gpu: bool,
    env_activate: Path | None,
    conda_env: str | None,
    log_dir: Path,
) -> str:
    indices_str = ",".join(str(i) for i in indices)
    script_path = log_dir / f"job_{job_id:04d}.sh"
    log_out     = log_dir / f"job_{job_id:04d}_%j.out"
    log_err     = log_dir / f"job_{job_id:04d}_%j.err"

    if env_activate:
        activate_cmd = f"source {env_activate}"
    elif conda_env:
        activate_cmd = f"conda activate {conda_env}"
    else:
        activate_cmd = "# no environment activation"

    gpu_line          = "#SBATCH --gres=gpu:1" if gpu else ""
    remove_spots_flag = "--remove-spots" if remove_spots else ""

    # The worker is invoked as a module so it works regardless of install location
    script = textwrap.dedent(f"""\
        #!/bin/bash
        #SBATCH --job-name=nrxrdct_{job_id:04d}
        #SBATCH --output={log_out}
        #SBATCH --error={log_err}
        #SBATCH --partition={partition}
        #SBATCH --time={time}
        #SBATCH --mem={mem}
        #SBATCH --cpus-per-task={cpus}
        {gpu_line}
        conda init bash
        {activate_cmd}

        echo "Job {job_id} started on $(hostname) at $(date)"
        echo "Indices: {indices_str}"

        python -m nrxrdct.slurm_integration.integrate_worker \\
            --master-file   "{master_file}"   \\
            --output-file   "{output_file}"   \\
            --poni-file     "{poni_file}"     \\
            --mask-file     "{mask_file}"     \\
            --entry-indices "{indices_str}"   \\
            --n-points      {n_points}        \\
            --n-workers     {n_workers}       \\
            --unit          "{unit}"          \\
            --percentile    "{percentile}"    \\
            {remove_spots_flag}

        echo "Job {job_id} finished at $(date)"
    """)

    script_path.write_text(script)
    script_path.chmod(0o755)

    result = subprocess.run(
        ["sbatch", str(script_path)],
        capture_output=True, text=True, check=True,
    )
    slurm_id = result.stdout.strip().split()[-1]
    print(f"  Submitted job {job_id:04d} "
          f"(indices {indices[0]}–{indices[-1]}) → SLURM {slurm_id}")
    return slurm_id


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def launch(
    master_file: Path,
    output_file: Path,
    poni_file: Path,
    mask_file: Path,
    rot: np.ndarray | None = None,
    n_jobs: int = 20,
    n_points: int = 1000,
    n_workers: int = 20,
    unit: str = "2th_deg",
    remove_spots: bool = False,
    percentile: tuple = (10, 90),
    # SLURM
    partition: str = "nice",
    time: str = "04:00:00",
    mem: str = "100G",
    cpus: int = 40,
    gpu: bool = False,
    # Environment
    env_activate: Path | None = None,
    conda_env: str | None = "xrdct",
) -> list[str]:
    """
    Validate, initialise output, and submit N SLURM jobs for powder integration.

    Returns
    -------
    list[str]
        SLURM job IDs of the submitted jobs.
    """
    master_file = Path(master_file)
    output_file = Path(output_file)
    poni_file   = Path(poni_file)
    mask_file   = Path(mask_file)
    percentile_str = f"{percentile[0]},{percentile[1]}"

    # ── 1. Validate ───────────────────────────────────────────────────────────
    print("=" * 60)
    print("Step 1 — Validating master file entries")
    print("=" * 60)
    valid_entries, bad_entries, dty_values = _validate_entries(master_file)

    if not valid_entries:
        raise RuntimeError("No valid entries found in master file.")

    # ── 2. Rotation angles ────────────────────────────────────────────────────
    if rot is None:
        with h5py.File(master_file, "r") as hin:
            rot = hin[f"{valid_entries[0]}/measurement/rot"][:]

    # ── 3. Initialise output file ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 2 — Initialising output HDF5 file")
    print("=" * 60)
    _init_output_file(
        master_file   = master_file,
        output_file   = output_file,
        poni_file     = poni_file,
        mask_file     = mask_file,
        valid_entries = valid_entries,
        bad_entries   = bad_entries,
        dty_values    = dty_values,
        rot           = rot,
        n_points      = n_points,
        unit          = unit,
    )

    # ── 4. Split & submit ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 3 — Splitting work and submitting jobs")
    print("=" * 60)
    chunks  = _split_indices(len(valid_entries), n_jobs)
    log_dir = output_file.parent / "slurm_logs"
    log_dir.mkdir(exist_ok=True)

    slurm_ids = []
    for job_id, indices in enumerate(chunks):
        sid = _submit_job(
            job_id,
            indices,
            master_file   = master_file,
            output_file   = output_file,
            poni_file     = poni_file,
            mask_file     = mask_file,
            n_points      = n_points,
            n_workers     = n_workers,
            unit          = unit,
            remove_spots  = remove_spots,
            percentile    = percentile_str,
            partition     = partition,
            time          = time,
            mem           = mem,
            cpus          = cpus,
            gpu           = gpu,
            env_activate  = env_activate,
            conda_env     = conda_env,
            log_dir       = log_dir,
        )
        slurm_ids.append(sid)

    print(f"\n✓  {len(slurm_ids)} jobs submitted — IDs: {', '.join(slurm_ids)}")
    print(f"   Logs in: {log_dir}/")
    print(f"\n   Monitor : squeue -u $USER")
    print(f"   Verify  : python -m nrxrdct.slurm_integration.check_output "
          f"--output-file {output_file}")
    return slurm_ids


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point (called by 'nrxrdct-slurm launch ...')
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser(sub=None):
    import argparse
    desc = "Submit powder integration across N SLURM jobs"
    p = sub.add_parser("launch", help=desc, description=desc) if sub else argparse.ArgumentParser(description=desc)

    p.add_argument("--master-file",   required=True, type=Path)
    p.add_argument("--output-file",   required=True, type=Path)
    p.add_argument("--poni-file",     required=True, type=Path)
    p.add_argument("--mask-file",     required=True, type=Path)
    p.add_argument("--n-jobs",        type=int, default=8)
    p.add_argument("--n-points",      type=int, default=1000)
    p.add_argument("--n-workers",     type=int, default=16)
    p.add_argument("--unit",          default="2th_deg")
    p.add_argument("--remove-spots",  action="store_true")
    p.add_argument("--percentile",    default="10,90")
    p.add_argument("--partition",     default="cpu")
    p.add_argument("--time",          default="04:00:00")
    p.add_argument("--mem",           default="32G")
    p.add_argument("--cpus",          type=int, default=16)
    p.add_argument("--gpu",           action="store_true")
    p.add_argument("--env-activate",  type=Path, default=None)
    p.add_argument("--conda-env",     default=None)
    return p


def _cli_launch(args):
    pct = tuple(int(x) for x in args.percentile.split(","))
    launch(
        master_file  = args.master_file,
        output_file  = args.output_file,
        poni_file    = args.poni_file,
        mask_file    = args.mask_file,
        n_jobs       = args.n_jobs,
        n_points     = args.n_points,
        n_workers    = args.n_workers,
        unit         = args.unit,
        remove_spots = args.remove_spots,
        percentile   = pct,
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
    _cli_launch(p.parse_args())