"""
nrxrdct.slurm_reconstruction.launch_recon
------------------------------------------
Assemble (or accept pre-built) sinogram, then split the 2θ axis across N
SLURM jobs and submit one sbatch script per chunk.

Python API
----------
    from nrxrdct.slurm_reconstruction import build_sinogram, launch_recon

    # Build sinogram from integrated HDF5 (skipped if file already exists)
    build_sinogram(
        integrated_file = Path("output.h5"),
        sinogram_file   = Path("sinogram.h5"),
        n_rot           = 360,
        n_tth_angles    = 1000,
        n_lines         = 10,
    )

    # Submit reconstruction jobs (accepts pre-built sinogram_file)
    launch_recon(
        sinogram_file = Path("sinogram.h5"),
        output_file   = Path("reconstruction.h5"),
        n_jobs        = 8,
        algo          = "SART_CUDA",
        num_iter      = 200,
        dty_step      = 1.0,
        partition     = "nice",
        time          = "08:00:00",
        mem           = "64G",
        cpus          = 8,
        gpu           = True,
        conda_env     = "nrxrdct",
    )

CLI (registered as 'nrxrdct-slurm-recon')
------------------------------------------
    nrxrdct-slurm-recon build  \\
        --integrated-file output.h5 --sinogram-file sinogram.h5 \\
        --n-rot 360 --n-tth-angles 1000 --n-lines 10

    nrxrdct-slurm-recon launch \\
        --sinogram-file sinogram.h5 --output-file reconstruction.h5 \\
        --n-jobs 8 --algo SART_CUDA --num-iter 200 --gpu \\
        --partition nice --conda-env nrxrdct
"""

from __future__ import annotations

import math
import subprocess
from pathlib import Path

import h5py
import numpy as np

from nrxrdct.slurm_reconstruction.reconstruct_worker import RECONSTRUCTION_ALGOS


# ─────────────────────────────────────────────────────────────────────────────
# Sinogram assembly
# ─────────────────────────────────────────────────────────────────────────────


def build_sinogram(
    integrated_file: Path,
    sinogram_file: Path,
    n_rot: int,
    n_tth_angles: int,
    n_lines: int = 10,
    overwrite: bool = False,
) -> Path:
    """
    Assemble a sinogram from an integrated HDF5 file and save it to disk.

    Calls :func:`nrxrdct.reconstruction.assemble_sinogram` and writes the
    result to *sinogram_file* as an HDF5 dataset.  If *sinogram_file* already
    exists and *overwrite* is ``False``, the function returns immediately
    without rebuilding.

    The sinogram is stored with axes labelled so that the worker can read
    individual 2θ slices efficiently:

    - ``sinogram``    — shape ``(n_tth, n_rot, n_lines)``, dtype float32
    - ``radial``      — 2θ axis copied from the integrated file, with ``unit``
      attribute preserved
    - ``motors/rot``  — rotation angles in degrees
    - ``motors/dty``  — translation motor positions

    Parameters
    ----------
    integrated_file : Path
        HDF5 file produced by the integration pipeline (contains
        ``integrated/scan_*``, ``integrated/radial``, ``motors/rot``,
        ``motors/dty``).
    sinogram_file : Path
        Output HDF5 file to write (created or overwritten).
    n_rot : int
        Number of rotation frames per scan (sinogram angular dimension).
    n_tth_angles : int
        Number of 2θ bins (spectral dimension).
    n_lines : int, optional
        Number of translation (dty) lines; passed to
        :func:`~nrxrdct.reconstruction.assemble_sinogram` (default 10).
    overwrite : bool, optional
        Re-build and overwrite even if *sinogram_file* already exists
        (default ``False``).

    Returns
    -------
    Path
        Path to the written sinogram file.
    """
    from nrxrdct.reconstruction import assemble_sinogram

    sinogram_file = Path(sinogram_file)
    integrated_file = Path(integrated_file)

    if sinogram_file.exists() and not overwrite:
        print(f"✓  Sinogram file already exists, skipping build: {sinogram_file}")
        return sinogram_file

    print("=" * 60)
    print("Building sinogram from integrated file …")
    print("=" * 60)

    sino = assemble_sinogram(integrated_file, n_rot, n_tth_angles, n_lines)
    # sino shape: (n_tth, n_rot, n_lines)

    with h5py.File(integrated_file, "r") as hin:
        radial = hin["integrated/radial"][:]
        radial_unit = hin["integrated/radial"].attrs.get("unit", "2th_deg")
        rot = hin["motors/rot"][:]
        dty = hin["motors/dty"][:]

    sinogram_file.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(sinogram_file, "w") as hout:
        ds = hout.create_dataset(
            "sinogram",
            data=sino,
            compression="gzip",
            compression_opts=4,
            chunks=(1, sino.shape[1], sino.shape[2]),
        )
        ds.attrs["axes"] = "tth, rot, dty"

        ds_rad = hout.create_dataset("radial", data=radial)
        ds_rad.attrs["unit"] = radial_unit

        hout["motors/rot"] = rot
        hout["motors/dty"] = dty
        hout["meta/source"] = str(integrated_file)
        hout["meta/n_tth"] = len(radial)
        hout["meta/n_rot"] = n_rot
        hout["meta/n_lines"] = n_lines

    print(f"✓  Sinogram saved: {sinogram_file}  "
          f"shape={sino.shape}  dtype={sino.dtype}")
    return sinogram_file


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────


def _read_sinogram_metadata(sinogram_file: Path) -> tuple[int, int, int]:
    """Return ``(n_tth, n_rot, n_lines)`` from a sinogram HDF5 file."""
    with h5py.File(sinogram_file, "r") as hin:
        n_tth   = int(hin["meta/n_tth"][()])
        n_rot   = int(hin["meta/n_rot"][()])
        n_lines = int(hin["meta/n_lines"][()])
    return n_tth, n_rot, n_lines


def _init_output_file(
    output_file: Path,
    sinogram_file: Path,
) -> None:
    """
    Initialise the reconstruction output HDF5 file with axis metadata (idempotent).

    Copies the radial axis, motor arrays, and provenance metadata from
    *sinogram_file*.  Datasets that already exist are not overwritten.

    Parameters
    ----------
    output_file : Path
        Destination HDF5 file (opened in append mode).
    sinogram_file : Path
        Source sinogram file to copy metadata from.
    """
    with h5py.File(sinogram_file, "r") as hin, h5py.File(output_file, "a") as hout:
        if "radial" not in hout:
            ds = hout.create_dataset("radial", data=hin["radial"][:])
            ds.attrs["unit"] = hin["radial"].attrs.get("unit", "2th_deg")
        if "motors/rot" not in hout:
            hout["motors/rot"] = hin["motors/rot"][:]
        if "motors/dty" not in hout:
            hout["motors/dty"] = hin["motors/dty"][:]
        if "meta/sinogram_source" not in hout:
            hout["meta/sinogram_source"] = str(sinogram_file)

    print(f"✓  Output file initialised: {output_file}")


def _split_indices(n_total: int, n_jobs: int) -> list[list[int]]:
    """Split ``range(n_total)`` into *n_jobs* roughly equal chunks."""
    chunk_size = math.ceil(n_total / n_jobs)
    all_idx = list(range(n_total))
    chunks = [all_idx[i : i + chunk_size] for i in range(0, n_total, chunk_size)]
    print(f"✓  {n_total} 2θ slices → {len(chunks)} jobs (~{chunk_size} slices each)")
    return chunks


def _submit_job(
    job_id: int,
    tth_indices: list[int],
    *,
    sinogram_file: Path,
    output_file: Path,
    algo: str,
    num_iter: int,
    dty_step: float,
    partition: str,
    time: str,
    mem: str,
    cpus: int,
    gpu: bool,
    env_activate: Path | None,
    conda_env: str | None,
    log_dir: Path,
) -> str:
    """
    Write an sbatch script for *tth_indices* and submit it, returning the SLURM job ID.

    The script is written to ``<log_dir>/recon_job_<job_id:04d>.sh`` and
    invokes :mod:`nrxrdct.slurm_reconstruction.reconstruct_worker` as a
    Python module.

    Parameters
    ----------
    job_id : int
        Sequential job identifier used for script and log file naming.
    tth_indices : list of int
        Global 2θ bin indices assigned to this job.
    sinogram_file, output_file : Path
        Files forwarded verbatim to the worker CLI.
    algo : str
        ASTRA reconstruction algorithm forwarded to the worker.
    num_iter : int
        Number of reconstruction iterations forwarded to the worker.
    dty_step : float
        Detector pixel spacing forwarded to the worker.
    partition, time, mem : str
        SLURM resource directives.
    cpus : int
        ``--cpus-per-task`` value.
    gpu : bool
        If ``True``, adds ``#SBATCH --gres=gpu:1``.
    env_activate : Path or None
        Shell script to ``source`` before the worker command.
    conda_env : str or None
        Conda environment for ``conda run``; used when *env_activate* is ``None``.
    log_dir : Path
        Directory where the script and log files are written.

    Returns
    -------
    str
        SLURM job ID string returned by ``sbatch``.
    """
    indices_str = ",".join(str(i) for i in tth_indices)
    script_path = log_dir / f"recon_job_{job_id:04d}.sh"
    log_out     = log_dir / f"recon_job_{job_id:04d}_%j.out"
    log_err     = log_dir / f"recon_job_{job_id:04d}_%j.err"

    worker_args = (
        f'    --sinogram-file  "{sinogram_file}"  \\\n'
        f'    --output-file    "{output_file}"    \\\n'
        f'    --tth-indices    "{indices_str}"    \\\n'
        f'    --algo           "{algo}"           \\\n'
        f'    --num-iter       {num_iter}         \\\n'
        f'    --dty-step       {dty_step}'
    )

    if env_activate:
        env_block   = f"source {env_activate}"
        python_line = (
            f"python -m nrxrdct.slurm_reconstruction.reconstruct_worker \\\n"
            f"{worker_args}"
        )
    elif conda_env:
        env_block   = "# conda run used below — no separate activate needed"
        python_line = (
            f"conda run -n {conda_env} --no-capture-output "
            f"python -m nrxrdct.slurm_reconstruction.reconstruct_worker \\\n"
            f"{worker_args}"
        )
    else:
        env_block   = "# no environment activation"
        python_line = (
            f"python -m nrxrdct.slurm_reconstruction.reconstruct_worker \\\n"
            f"{worker_args}"
        )

    script = (
        f"#!/bin/bash\n"
        f"#SBATCH --job-name=nrxrdct_recon_{job_id:04d}\n"
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
        f'echo "Recon job {job_id} started on $(hostname) at $(date)"\n'
        f'echo "2\u03b8 indices: {indices_str}"\n'
        f"\n"
        f"{python_line}\n"
        f"\n"
        f'echo "Recon job {job_id} finished at $(date)"\n'
    )

    script_path.write_text(script)
    script_path.chmod(0o755)

    result = subprocess.run(
        ["sbatch", str(script_path)],
        capture_output=True, text=True, check=True,
    )
    slurm_id = result.stdout.strip().split()[-1]
    print(
        f"  Submitted recon job {job_id:04d} "
        f"(2\u03b8 {tth_indices[0]}\u2013{tth_indices[-1]}) \u2192 SLURM {slurm_id}"
    )
    return slurm_id


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def launch_recon(
    sinogram_file: Path,
    output_file: Path,
    n_jobs: int = 8,
    algo: str = "SART_CUDA",
    num_iter: int = 200,
    dty_step: float = 1.0,
    # SLURM
    partition: str = "nice",
    time: str = "08:00:00",
    mem: str = "64G",
    cpus: int = 8,
    gpu: bool = True,
    # Environment
    env_activate: Path | None = None,
    conda_env: str | None = None,
) -> list[str]:
    """
    Split 2θ reconstruction across N SLURM jobs and submit.

    Reads sinogram metadata from *sinogram_file*, initialises *output_file*,
    divides the 2θ axis into *n_jobs* chunks, and submits one sbatch job per
    chunk.  Each job runs :mod:`nrxrdct.slurm_reconstruction.reconstruct_worker`
    with a GPU node (when *gpu* is ``True``).

    Use :func:`build_sinogram` first if you need to assemble the sinogram from
    an integrated HDF5 file.

    Parameters
    ----------
    sinogram_file : Path
        HDF5 sinogram file produced by :func:`build_sinogram`.
    output_file : Path
        Destination HDF5 file for reconstructed slices.
    n_jobs : int, optional
        Number of SLURM jobs to submit (default 8).
    algo : str, optional
        ASTRA reconstruction algorithm.  GPU algorithms:
        ``"SART_CUDA"``, ``"SIRT_CUDA"``, ``"FBP_CUDA"``, ``"CGLS_CUDA"``.
        CPU algorithms: ``"FBP"``, ``"SIRT"``, ``"SART"``, ``"CGLS"``.
        (default ``"SART_CUDA"``).
    num_iter : int, optional
        Number of reconstruction iterations (default 200).
    dty_step : float, optional
        Detector pixel spacing passed to ASTRA (default 1.0).
    partition : str, optional
        SLURM partition (default ``"nice"``).
    time : str, optional
        SLURM wall-time limit (default ``"08:00:00"``).
    mem : str, optional
        SLURM memory request (default ``"64G"``).
    cpus : int, optional
        CPUs per task (default 8).
    gpu : bool, optional
        Request a GPU node via ``--gres=gpu:1`` (default ``True``).
    env_activate : Path or None, optional
        Shell activate script sourced before the worker command.
    conda_env : str or None, optional
        Conda environment used via ``conda run`` (alternative to *env_activate*).

    Returns
    -------
    list of str
        SLURM job IDs of the submitted jobs.
    """
    if algo not in RECONSTRUCTION_ALGOS:
        raise ValueError(
            f"algo must be one of {RECONSTRUCTION_ALGOS}, got '{algo}'"
        )

    sinogram_file = Path(sinogram_file)
    output_file   = Path(output_file)

    if not sinogram_file.exists():
        raise FileNotFoundError(
            f"Sinogram file not found: {sinogram_file}\n"
            "Run build_sinogram() first, or use 'nrxrdct-slurm-recon build'."
        )

    # ── 1. Read sinogram metadata ─────────────────────────────────────────────
    print("=" * 60)
    print("Step 1 — Reading sinogram metadata")
    print("=" * 60)
    n_tth, n_rot, n_lines = _read_sinogram_metadata(sinogram_file)
    print(f"✓  n_tth={n_tth}, n_rot={n_rot}, n_lines={n_lines}")

    # ── 2. Initialise output file ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 2 — Initialising output HDF5 file")
    print("=" * 60)
    _init_output_file(output_file, sinogram_file)

    # ── 3. Split & submit ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 3 — Splitting 2\u03b8 axis and submitting jobs")
    print("=" * 60)
    chunks  = _split_indices(n_tth, n_jobs)
    log_dir = output_file.parent / "slurm_logs_recon"
    log_dir.mkdir(exist_ok=True)

    slurm_ids = []
    for job_id, tth_indices in enumerate(chunks):
        sid = _submit_job(
            job_id,
            tth_indices,
            sinogram_file = sinogram_file,
            output_file   = output_file,
            algo          = algo,
            num_iter      = num_iter,
            dty_step      = dty_step,
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

    print(f"\n\u2713  {len(slurm_ids)} jobs submitted \u2014 IDs: {', '.join(slurm_ids)}")
    print(f"   Logs in: {log_dir}/")
    print(f"\n   Monitor : squeue -u $USER")
    return slurm_ids


# ─────────────────────────────────────────────────────────────────────────────
# CLI helpers (called from cli.py)
# ─────────────────────────────────────────────────────────────────────────────


def _build_parser_build(sub=None):
    """Build the ``build`` sub-command parser."""
    import argparse
    desc = "Assemble sinogram from an integrated HDF5 file and save to disk"
    p = (
        sub.add_parser("build", help=desc, description=desc)
        if sub
        else argparse.ArgumentParser(description=desc)
    )
    p.add_argument("--integrated-file", required=True, type=Path,
                   help="HDF5 file produced by the integration pipeline")
    p.add_argument("--sinogram-file",   required=True, type=Path,
                   help="Output HDF5 file to write")
    p.add_argument("--n-rot",           required=True, type=int,
                   help="Number of rotation frames per scan")
    p.add_argument("--n-tth-angles",    required=True, type=int,
                   help="Number of 2θ bins")
    p.add_argument("--n-lines",         type=int, default=10,
                   help="Number of translation (dty) lines (default: 10)")
    p.add_argument("--overwrite",       action="store_true",
                   help="Overwrite existing sinogram file")
    return p


def _build_parser_launch(sub=None):
    """Build the ``launch`` sub-command parser."""
    import argparse
    desc = "Submit GPU reconstruction jobs across the 2θ axis"
    p = (
        sub.add_parser("launch", help=desc, description=desc)
        if sub
        else argparse.ArgumentParser(description=desc)
    )
    p.add_argument("--sinogram-file",  required=True, type=Path,
                   help="HDF5 sinogram file (from 'build' step)")
    p.add_argument("--output-file",    required=True, type=Path,
                   help="Destination HDF5 file for reconstructed slices")
    p.add_argument("--n-jobs",         type=int,   default=8,
                   help="Number of SLURM jobs to submit (default: 8)")
    p.add_argument("--algo",           default="SART_CUDA",
                   choices=RECONSTRUCTION_ALGOS,
                   help="ASTRA reconstruction algorithm (default: SART_CUDA)")
    p.add_argument("--num-iter",       type=int,   default=200,
                   help="Number of reconstruction iterations (default: 200)")
    p.add_argument("--dty-step",       type=float, default=1.0,
                   help="Detector pixel spacing (default: 1.0)")
    p.add_argument("--partition",      default="nice",
                   help="SLURM partition (default: nice)")
    p.add_argument("--time",           default="08:00:00",
                   help="SLURM wall-time (default: 08:00:00)")
    p.add_argument("--mem",            default="64G",
                   help="SLURM memory request (default: 64G)")
    p.add_argument("--cpus",           type=int, default=8,
                   help="CPUs per task (default: 8)")
    p.add_argument("--gpu",            action="store_true",
                   help="Request a GPU node (--gres=gpu:1)")
    p.add_argument("--env-activate",   type=Path, default=None,
                   help="Shell activate script to source before the worker")
    p.add_argument("--conda-env",      default=None,
                   help="Conda environment for conda run")
    return p


def _cli_build(args):
    build_sinogram(
        integrated_file = args.integrated_file,
        sinogram_file   = args.sinogram_file,
        n_rot           = args.n_rot,
        n_tth_angles    = args.n_tth_angles,
        n_lines         = args.n_lines,
        overwrite       = args.overwrite,
    )


def _cli_launch(args):
    launch_recon(
        sinogram_file = args.sinogram_file,
        output_file   = args.output_file,
        n_jobs        = args.n_jobs,
        algo          = args.algo,
        num_iter      = args.num_iter,
        dty_step      = args.dty_step,
        partition     = args.partition,
        time          = args.time,
        mem           = args.mem,
        cpus          = args.cpus,
        gpu           = args.gpu,
        env_activate  = args.env_activate,
        conda_env     = args.conda_env,
    )
