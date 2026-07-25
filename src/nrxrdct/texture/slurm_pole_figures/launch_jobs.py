"""
nrxrdct.texture.slurm_pole_figures.launch_jobs
------------------------------------------------
Validates master HDF5 entries, writes a launch_meta.json sidecar into the
tmp directory, and submits N sbatch jobs that extract pole-figure (chi
intensity) profiles for one or more hkl rings.

Each job cake-integrates every frame **once** and extracts all requested
rings from that single cake — extracting several hkls costs one pass over
the raw frames, not one pass per hkl.

All per-job settings (which hkls, backgrounds, integration parameters, ...)
live in launch_meta.json rather than being passed as sbatch/worker CLI
arguments, since the worker already reads that file for valid_entries and
dty_values.

The output HDF5 file is NOT created here — it is assembled by merge() after
all jobs finish.

Python API
----------
    from nrxrdct.texture.slurm_pole_figures import launch

    result = launch(
        master_file = Path("master.h5"),
        output_file = Path("pole_figures.h5"),
        poni_file   = Path("calib.poni"),
        mask_file   = Path("mask.edf"),
        hkls        = {"111": (4.0, 6.0), "200": (7.0, 9.0), "220": (11.0, 13.0)},
        n_jobs      = 8,
        partition   = "nice",
        conda_env   = "nrxrdct",
    )

CLI
---
    nrxrdct-slurm-texture launch --master-file master.h5 --output-file pole_figures.h5 \\
        --poni-file calib.poni --mask-file mask.edf \\
        --hkls "111:4.0,6.0;200:7.0,9.0;220:11.0,13.0" \\
        --n-jobs 8 --partition nice --conda-env nrxrdct
"""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path
from typing import Dict, Optional, Tuple

import h5py
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────


def _validate_entries(
    master_file: Path,
    camera_name: str,
    monitor_name: str,
    translation_motor: str,
) -> tuple[list, list, list]:
    valid_entries, bad_entries, dty_values = [], [], []
    with h5py.File(master_file, "r") as hin:
        all_entries = list(hin.keys())
        for entry in tqdm(all_entries, desc="Validating entries"):
            try:
                _ = hin[f"{entry}/measurement/{camera_name}"].shape
                _ = hin[f"{entry}/measurement/{monitor_name}"].shape
                dty = float(hin[f"{entry}/instrument/positioners/{translation_motor}"][()])
                valid_entries.append(entry)
                dty_values.append(dty)
            except KeyError as e:
                print(f"  ⚠  Entry {entry} missing dataset ({e}) — skipping")
                bad_entries.append(entry)
    print(f"\n✓  {len(valid_entries)}/{len(all_entries)} entries OK")
    if bad_entries:
        print(f"⚠  Skipping {len(bad_entries)} entries: {bad_entries}\n")
    return valid_entries, bad_entries, dty_values


def _split_indices(n_scans: int, n_jobs: int) -> list[list[int]]:
    all_idx = list(range(n_scans))
    chunk_size = math.ceil(n_scans / n_jobs)
    chunks = [all_idx[i : i + chunk_size] for i in range(0, n_scans, chunk_size)]
    print(f"✓  {n_scans} scans → {len(chunks)} jobs (~{chunk_size} scans each)")
    return chunks


def _submit_job(
    job_id: int,
    indices: list[int],
    *,
    master_file: Path,
    tmp_dir: Path,
    poni_file: Path,
    mask_file: Path,
    partition: str,
    time: str,
    mem: str,
    cpus: int,
    gpu: bool,
    env_activate: Optional[Path],
    conda_env: Optional[str],
    log_dir: Path,
) -> str:
    indices_str = ",".join(str(i) for i in indices)
    script_path = log_dir / f"job_{job_id:04d}.sh"
    log_out = log_dir / f"job_{job_id:04d}_%j.out"
    log_err = log_dir / f"job_{job_id:04d}_%j.err"

    worker_args = (
        f'    --master-file        "{master_file}"        \\\n'
        f'    --tmp-dir            "{tmp_dir}"             \\\n'
        f'    --poni-file          "{poni_file}"           \\\n'
        f'    --mask-file          "{mask_file}"           \\\n'
        f'    --entry-indices      "{indices_str}"'
    )

    if env_activate:
        env_block = f"source {env_activate}"
        python_line = (
            f"python -m nrxrdct.texture.slurm_pole_figures.pole_figure_worker \\\n{worker_args}"
        )
    elif conda_env:
        env_block = "# conda run used below"
        python_line = (
            f"conda run -n {conda_env} --no-capture-output "
            f"python -m nrxrdct.texture.slurm_pole_figures.pole_figure_worker \\\n{worker_args}"
        )
    else:
        env_block = "# no environment activation"
        python_line = (
            f"python -m nrxrdct.texture.slurm_pole_figures.pole_figure_worker \\\n{worker_args}"
        )

    script = (
        f"#!/bin/bash\n"
        f"#SBATCH --job-name=nrxrdct_pf_{job_id:04d}\n"
        f"#SBATCH --output={log_out}\n"
        f"#SBATCH --error={log_err}\n"
        f"#SBATCH --partition={partition}\n"
        f"#SBATCH --time={time}\n"
        f"#SBATCH --mem={mem}\n"
        f"#SBATCH --cpus-per-task={cpus}\n"
        + (f"#SBATCH --gres=gpu:1\n" if gpu else "")
        + f"\n"
        f"{env_block}\n"
        f"\n"
        f'echo "Job {job_id} started on $(hostname) at $(date)"\n'
        f'echo "Indices: {indices_str}"\n'
        f"\n"
        f"{python_line}\n"
        f"\n"
        f'echo "Job {job_id} finished at $(date)"\n'
    )

    script_path.write_text(script)
    script_path.chmod(0o755)

    result = subprocess.run(
        ["sbatch", str(script_path)],
        capture_output=True,
        text=True,
        check=True,
    )
    slurm_id = result.stdout.strip().split()[-1]
    print(
        f"  Submitted job {job_id:04d} "
        f"(indices {indices[0]}–{indices[-1]}) → SLURM {slurm_id}"
    )
    return slurm_id


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def launch(
    master_file: Path,
    output_file: Path,
    poni_file: Path,
    mask_file: Path,
    hkls: Dict[str, Tuple[float, float]],
    background_ranges: Optional[Dict[str, Tuple[Tuple[float, float], Tuple[float, float]]]] = None,
    n_jobs: int = 8,
    npt_rad: int = 1000,
    npt_azim: int = 360,
    n_workers: Optional[int] = None,
    batch_size: int = 32,
    camera_name: str = "eiger",
    monitor_name: str = "fpico6",
    translation_motor: str = "dty",
    rotation_motor: str = "rot",
    # SLURM
    partition: str = "nice",
    time: str = "04:00:00",
    mem: str = "32G",
    cpus: int = 16,
    gpu: bool = False,
    # Environment
    env_activate: Optional[Path] = None,
    conda_env: Optional[str] = None,
) -> dict:
    """
    Validate master file, write launch_meta.json, and submit N SLURM jobs
    that extract pole-figure profiles for one or more hkl rings.

    Each worker cake-integrates every frame once and extracts all requested
    rings from that single cake, so adding more hkls does not multiply the
    (expensive) integration cost the way calling :func:`launch` once per hkl
    would.

    Workers write results to ``<output_file.parent>/<output_file.stem>_tmp/``.
    Call :func:`nrxrdct.texture.slurm_pole_figures.merge` after all jobs finish
    to assemble the output HDF5.

    Args:
        master_file (Path): HDF5 master file containing all scan entries.
        output_file (Path): Destination HDF5 file for pole-figure data; used to
            derive the tmp directory name.
        poni_file (Path): pyFAI ``.poni`` calibration file.
        mask_file (Path): Detector mask file (fabio-readable).
        hkls (dict): Maps hkl label to its ``(low, high)`` 2theta window, e.g.
            ``{"111": (4.0, 6.0), "200": (7.0, 9.0)}``. Each label becomes an
            HDF5 group name under ``pole_figures/``.
        background_ranges (dict, optional): Maps hkl label to
            ``((low1, high1), (low2, high2))`` flanking 2theta windows for
            background subtraction. Labels not present get no background
            subtraction; pass ``None`` (default) to skip it for every hkl.
        n_jobs (int, optional): Number of SLURM jobs to submit (default 8).
        npt_rad (int, optional): Radial bins used internally by CAKE integration
            (default 1000).
        npt_azim (int, optional): Number of azimuthal (chi) bins (default 360).
        n_workers (int, optional): Integration threads per scan; auto-scaled from
            available RAM when ``None`` (default ``None``).
        batch_size (int, optional): Frames streamed from HDF5 per batch in the
            worker (default 32).
        camera_name (str, optional): Detector dataset name under ``measurement/``
            (default ``"eiger"``).
        monitor_name (str, optional): Monitor dataset name under ``measurement/``
            used for normalisation (default ``"fpico6"``).
        translation_motor (str, optional): Translation-motor name read from
            ``instrument/positioners/`` (default ``"dty"``).
        rotation_motor (str, optional): Rotation-motor dataset name under
            ``measurement/`` (default ``"rot"``).
        partition (str, optional): SLURM partition (default ``"nice"``).
        time (str, optional): SLURM wall-time limit (default ``"04:00:00"``).
        mem (str, optional): SLURM memory request (default ``"32G"``).
        cpus (int, optional): CPUs per task (default 16).
        gpu (bool, optional): Request a GPU node (default ``False``).
        env_activate (Path, optional): Shell activate script sourced before the
            worker command.
        conda_env (str, optional): Conda environment used via ``conda run``
            (alternative to *env_activate*).

    Returns:
        dict: With keys ``'slurm_ids'``, ``'tmp_dir'``, ``'n_scans'``, ``'hkls'``.
    """
    if not hkls:
        raise ValueError("hkls must contain at least one hkl label.")

    master_file = Path(master_file)
    output_file = Path(output_file)
    poni_file = Path(poni_file)
    mask_file = Path(mask_file)

    tmp_dir = output_file.parent / (output_file.stem + "_tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_file.parent / "slurm_logs_texture"
    log_dir.mkdir(exist_ok=True)

    # ── 1. Validate ───────────────────────────────────────────────────────────
    print("=" * 60)
    print("Step 1 — Validating master file entries")
    print("=" * 60)
    valid_entries, bad_entries, dty_values = _validate_entries(
        master_file, camera_name, monitor_name, translation_motor
    )
    if not valid_entries:
        raise RuntimeError("No valid entries found in master file.")

    # ── 2. Write launch_meta.json sidecar ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 2 — Writing launch metadata")
    print("=" * 60)
    print(f"  hkls: {list(hkls.keys())}")
    launch_meta = {
        "valid_entries": valid_entries,
        "bad_entries": bad_entries,
        "dty_values": dty_values,
        "hkls": {hkl: list(tth_range) for hkl, tth_range in hkls.items()},
        "background_ranges": (
            {hkl: [list(bg[0]), list(bg[1])] for hkl, bg in background_ranges.items()}
            if background_ranges is not None
            else {}
        ),
        "npt_rad": npt_rad,
        "npt_azim": npt_azim,
        "master_file": str(master_file),
        "output_file": str(output_file),
        "poni_file": str(poni_file),
        "mask_file": str(mask_file),
        "camera_name": camera_name,
        "monitor_name": monitor_name,
        "translation_motor": translation_motor,
        "rotation_motor": rotation_motor,
        # SLURM settings — reused by repair()
        "partition": partition,
        "time": time,
        "mem": mem,
        "cpus": cpus,
        "gpu": gpu,
        "env_activate": str(env_activate) if env_activate else None,
        "conda_env": conda_env,
        # Worker settings — reused by repair()
        "n_workers": n_workers,
        "batch_size": batch_size,
    }
    sidecar_path = tmp_dir / "launch_meta.json"
    sidecar_path.write_text(json.dumps(launch_meta, indent=2))
    print(f"✓  launch_meta.json → {sidecar_path}")

    # ── 3. Split & submit ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 3 — Splitting work and submitting jobs")
    print("=" * 60)
    chunks = _split_indices(len(valid_entries), n_jobs)
    slurm_ids = []
    for job_id, indices in enumerate(chunks):
        sid = _submit_job(
            job_id,
            indices,
            master_file=master_file,
            tmp_dir=tmp_dir,
            poni_file=poni_file,
            mask_file=mask_file,
            partition=partition,
            time=time,
            mem=mem,
            cpus=cpus,
            gpu=gpu,
            env_activate=env_activate,
            conda_env=conda_env,
            log_dir=log_dir,
        )
        slurm_ids.append(sid)

    print(f"\n✓  {len(slurm_ids)} jobs submitted — IDs: {', '.join(slurm_ids)}")
    print(f"   Tmp dir : {tmp_dir}/")
    print(f"   Logs    : {log_dir}/")
    print(
        f"\n   Monitor : nrxrdct-slurm-texture monitor --slurm-ids {','.join(slurm_ids)} "
        f"--tmp-dir {tmp_dir} --watch"
    )
    print(
        f"   Merge   : nrxrdct-slurm-texture merge --tmp-dir {tmp_dir} "
        f"--output-file {output_file}"
    )

    return {
        "slurm_ids": slurm_ids,
        "tmp_dir": tmp_dir,
        "n_scans": len(valid_entries),
        "hkls": list(hkls.keys()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def _build_parser(sub=None):
    import argparse

    desc = "Submit pole-figure extraction (one or more hkls) across N SLURM jobs"
    p = (
        sub.add_parser("launch", help=desc, description=desc)
        if sub
        else argparse.ArgumentParser(description=desc)
    )

    p.add_argument("--master-file", required=True, type=Path)
    p.add_argument("--output-file", required=True, type=Path)
    p.add_argument("--poni-file", required=True, type=Path)
    p.add_argument("--mask-file", required=True, type=Path)
    p.add_argument(
        "--hkls", required=True,
        help='semicolon-separated "label:low,high" entries, '
             'e.g. "111:4.0,6.0;200:7.0,9.0;220:11.0,13.0"',
    )
    p.add_argument(
        "--background-ranges", default="",
        help='semicolon-separated "label:low1,high1,low2,high2" entries '
             '(omit a label to skip background subtraction for it)',
    )
    p.add_argument("--n-jobs", type=int, default=8)
    p.add_argument("--npt-rad", type=int, default=1000)
    p.add_argument("--npt-azim", type=int, default=360)
    p.add_argument("--n-workers", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--camera-name", default="eiger")
    p.add_argument("--monitor-name", default="fpico6")
    p.add_argument("--translation-motor", default="dty")
    p.add_argument("--rotation-motor", default="rot")
    p.add_argument("--partition", default="nice")
    p.add_argument("--time", default="04:00:00")
    p.add_argument("--mem", default="32G")
    p.add_argument("--cpus", type=int, default=16)
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--env-activate", type=Path, default=None)
    p.add_argument("--conda-env", default=None)
    return p


def _parse_hkls(s: str) -> Dict[str, Tuple[float, float]]:
    hkls = {}
    for entry in s.split(";"):
        entry = entry.strip()
        if not entry:
            continue
        label, ranges = entry.split(":")
        lo, hi = (float(x) for x in ranges.split(","))
        hkls[label] = (lo, hi)
    return hkls


def _parse_background_ranges_multi(s: str) -> Dict[str, Tuple[Tuple[float, float], Tuple[float, float]]]:
    result = {}
    for entry in s.split(";"):
        entry = entry.strip()
        if not entry:
            continue
        label, ranges = entry.split(":")
        lo1, hi1, lo2, hi2 = (float(x) for x in ranges.split(","))
        result[label] = ((lo1, hi1), (lo2, hi2))
    return result


def _cli_launch(args):
    launch(
        master_file=args.master_file,
        output_file=args.output_file,
        poni_file=args.poni_file,
        mask_file=args.mask_file,
        hkls=_parse_hkls(args.hkls),
        background_ranges=_parse_background_ranges_multi(args.background_ranges) or None,
        n_jobs=args.n_jobs,
        npt_rad=args.npt_rad,
        npt_azim=args.npt_azim,
        n_workers=args.n_workers,
        batch_size=args.batch_size,
        camera_name=args.camera_name,
        monitor_name=args.monitor_name,
        translation_motor=args.translation_motor,
        rotation_motor=args.rotation_motor,
        partition=args.partition,
        time=args.time,
        mem=args.mem,
        cpus=args.cpus,
        gpu=args.gpu,
        env_activate=args.env_activate,
        conda_env=args.conda_env,
    )


if __name__ == "__main__":
    p = _build_parser()
    _cli_launch(p.parse_args())