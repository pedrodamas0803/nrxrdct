"""
nrxrdct.xrdct.slurm_s3dxrd
---------------------------
SLURM-parallel scanning-3DXRD segmentation, following the same pattern as
nrxrdct.azimuthal.slurm_integration.

Workflow
--------
1. launch()  — validate entries, write launch_meta.json, submit N sbatch jobs
2. monitor() — poll SLURM job states and count completed scans
3. merge()   — assemble per-scan .npz tmp files into segmented.h5
4. check()   — verify completeness against launch_meta.json

After merge(), the returned segmented.h5 is consumed by
build_columnfile() / load_segmentation() exactly as if segment_slice()
had been called locally.

Example
-------
    from nrxrdct.xrdct.slurm_s3dxrd import launch, monitor, merge, check

    result = launch(
        master_file = MASTER_FILE,
        output_file = SEGMENTATION_FILE,
        mask_file   = MASK_FILE,
        n_jobs      = 60,
        partition   = "nice",
        mem         = "32G",
        conda_env   = "xrdct",
    )
    monitor(result["slurm_ids"], result["tmp_dir"], watch=True, interval=30)
    merge(result["tmp_dir"], SEGMENTATION_FILE)
    check(result["tmp_dir"], SEGMENTATION_FILE)
"""
from __future__ import annotations

import json
import math
import subprocess
import time
from datetime import timedelta
from pathlib import Path
from typing import List, Optional, Union

import h5py
import numpy as np
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _validate_entries(
    master_file: Path,
    camera_name: str,
    translation_motor: str,
) -> tuple[list, list, list]:
    import fabio  # only needed at launch time
    valid_entries, bad_entries, dty_values = [], [], []
    with h5py.File(master_file, "r") as hin:
        all_entries = list(hin.keys())
        for entry in tqdm(all_entries, desc="Validating entries"):
            try:
                _ = hin[f"{entry}/measurement/{camera_name}"].shape
                dty = float(
                    hin[f"{entry}/instrument/positioners/{translation_motor}"][()]
                )
                valid_entries.append(entry)
                dty_values.append(dty)
            except KeyError as e:
                print(f"  ⚠  Entry {entry} missing dataset ({e}) — skipping")
                bad_entries.append(entry)
    print(f"\n✓  {len(valid_entries)}/{len(all_entries)} entries OK")
    if bad_entries:
        print(f"⚠  Skipping {len(bad_entries)} entries: {bad_entries}")
    return valid_entries, bad_entries, dty_values


def _split_indices(n_scans: int, n_jobs: int) -> list[list[int]]:
    chunk_size = math.ceil(n_scans / n_jobs)
    all_idx    = list(range(n_scans))
    chunks     = [all_idx[i: i + chunk_size] for i in range(0, n_scans, chunk_size)]
    print(f"✓  {n_scans} scans → {len(chunks)} jobs (~{chunk_size} scans each)")
    return chunks


def _submit_job(
    job_id: int,
    indices: list[int],
    *,
    master_file: Path,
    scan_dir: Path,
    mask_file: Path,
    camera_name: str,
    translation_motor: str,
    rotation_motor: str,
    cut: float,
    howmany: int,
    pixels_in_spot: int,
    partition: str,
    time_limit: str,
    mem: str,
    cpus: int,
    python_bin: str,
    log_dir: Path,
) -> str:
    indices_str = ",".join(str(i) for i in indices)
    log_out = log_dir / f"job_{job_id:04d}_%j.out"
    log_err = log_dir / f"job_{job_id:04d}_%j.err"

    wrap_cmd = (
        f'{python_bin} -m nrxrdct.xrdct._segment_worker'
        f' --master-file "{master_file}"'
        f' --scan-dir "{scan_dir}"'
        f' --mask-file "{mask_file}"'
        f' --entry-indices "{indices_str}"'
        f' --camera-name "{camera_name}"'
        f' --translation-motor "{translation_motor}"'
        f' --rotation-motor "{rotation_motor}"'
        f' --cut {cut}'
        f' --howmany {howmany}'
        f' --pixels-in-spot {pixels_in_spot}'
    )

    result = subprocess.run(
        [
            "sbatch",
            f"--job-name=s3dxrd_{job_id:04d}",
            f"--partition={partition}",
            f"--time={time_limit}",
            f"--mem={mem}",
            f"--cpus-per-task={cpus}",
            f"--output={log_out}",
            f"--error={log_err}",
            "--wrap", wrap_cmd,
        ],
        capture_output=True, text=True, check=True,
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
    master_file: Union[str, Path],
    output_file: Union[str, Path],
    mask_file: Union[str, Path],
    n_jobs: int = 8,
    # Segmentation options
    cut: float = 1.0,
    howmany: int = 100_000,
    pixels_in_spot: int = 3,
    # Motor / detector names
    camera_name: str = "eiger",
    translation_motor: str = "dty",
    rotation_motor: str = "rot",
    # SLURM
    partition: str = "nice",
    time: str = "04:00:00",
    mem: str = "32G",
    cpus: int = 4,
    python_bin: str = "python",
) -> dict:
    """
    Validate master file entries, write launch_meta.json, and submit N SLURM
    jobs for scanning-3DXRD Bragg-spot segmentation.

    Each job runs :mod:`nrxrdct.xrdct._segment_worker` for a disjoint subset
    of scans, writing per-scan ``.npz`` and ``.meta.json`` files to a shared
    tmp directory (``<output_file.stem>_tmp/`` next to *output_file*).

    Call :func:`merge` after all jobs finish to assemble ``segmented.h5``,
    then feed it directly to :func:`~nrxrdct.xrdct.s3dxrd.build_columnfile`.

    Args:
        master_file: HDF5 master file containing all scan entries.
        output_file: Desired path for the final ``segmented.h5`` (used to
            derive the tmp directory name; the file is not created here).
        mask_file: Detector mask file (fabio-readable, ``1`` = masked).
        n_jobs: Number of SLURM jobs to submit.
        cut: Segmentation intensity threshold (passed to
            :class:`~nrxrdct.xrdct.s3dxrd.SegmentationOptions`).
        howmany: Max pixels kept per frame.
        pixels_in_spot: Min connected pixels for a spot to be kept.
        camera_name: Detector dataset name under ``measurement/``.
        translation_motor: Translation motor name under ``instrument/positioners/``.
        rotation_motor: Rotation motor dataset name under ``measurement/``.
        partition: SLURM partition name.
        time: SLURM wall-time limit (``HH:MM:SS``).
        mem: SLURM memory request (e.g. ``"32G"``).
        cpus: ``--cpus-per-task`` value (segmentation is single-threaded per
            scan, so 1–4 is typical).
        python_bin: Full path to the Python interpreter on the compute nodes
            (e.g. ``"/path/to/env/bin/python"``). Use the absolute path so
            the job does not depend on the node knowing about any conda or
            virtual environment activation.

    Returns:
        dict with keys ``'slurm_ids'``, ``'tmp_dir'``, ``'n_scans'``.
    """
    master_file = Path(master_file)
    output_file = Path(output_file)
    mask_file   = Path(mask_file)

    scan_dir = output_file.parent / (output_file.stem + "_scans")
    scan_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_file.parent / "slurm_logs_s3dxrd"
    log_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("Step 1 — Validating master file entries")
    print("=" * 60)
    valid_entries, bad_entries, dty_values = _validate_entries(
        master_file, camera_name, translation_motor
    )
    if not valid_entries:
        raise RuntimeError("No valid entries found in master file.")

    print("\n" + "=" * 60)
    print("Step 2 — Writing launch metadata")
    print("=" * 60)
    launch_meta = {
        "valid_entries":     valid_entries,
        "bad_entries":       bad_entries,
        "dty_values":        dty_values,
        "master_file":       str(master_file),
        "output_file":       str(output_file),
        "mask_file":         str(mask_file),
        "camera_name":       camera_name,
        "translation_motor": translation_motor,
        "rotation_motor":    rotation_motor,
        "cut":               cut,
        "howmany":           howmany,
        "pixels_in_spot":    pixels_in_spot,
        "partition":         partition,
        "time":              time,
        "mem":               mem,
        "cpus":              cpus,
        "python_bin":        python_bin,
    }
    sidecar = scan_dir / "launch_meta.json"
    sidecar.write_text(json.dumps(launch_meta, indent=2))
    print(f"✓  launch_meta.json → {sidecar}")

    print("\n" + "=" * 60)
    print("Step 3 — Splitting work and submitting jobs")
    print("=" * 60)
    chunks    = _split_indices(len(valid_entries), n_jobs)
    slurm_ids = []
    for job_id, indices in enumerate(chunks):
        sid = _submit_job(
            job_id, indices,
            master_file=master_file,
            scan_dir=scan_dir,
            mask_file=mask_file,
            camera_name=camera_name,
            translation_motor=translation_motor,
            rotation_motor=rotation_motor,
            cut=cut,
            howmany=howmany,
            pixels_in_spot=pixels_in_spot,
            partition=partition,
            time_limit=time,
            mem=mem,
            cpus=cpus,
            python_bin=python_bin,
            log_dir=log_dir,
        )
        slurm_ids.append(sid)

    print(f"\n✓  {len(slurm_ids)} jobs submitted — IDs: {', '.join(slurm_ids)}")
    print(f"   Scan dir : {scan_dir}/")
    print(f"   Logs     : {log_dir}/")
    return {"slurm_ids": slurm_ids, "scan_dir": scan_dir, "n_scans": len(valid_entries)}


def monitor(
    slurm_ids: list,
    scan_dir: Union[str, Path],
    watch: bool = False,
    interval: int = 30,
) -> None:
    """
    Poll SLURM job states and count completed ``.h5`` files in *scan_dir*.

    Args:
        slurm_ids: List of SLURM job ID strings returned by :func:`launch`.
        scan_dir: Scan directory written by the workers (same as
            ``result["scan_dir"]`` from :func:`launch`).
        watch: If ``True``, poll repeatedly until all jobs finish.
        interval: Seconds between polls when *watch* is ``True``.
    """
    scan_dir  = Path(scan_dir)
    slurm_ids = [str(s) for s in slurm_ids]

    _RUNNING = {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "REQUEUED"}
    _DONE    = {"COMPLETED"}
    _FAILED  = {"FAILED", "OUT_OF_MEMORY", "TIMEOUT", "CANCELLED",
                "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE"}

    def _query_slurm():
        try:
            r = subprocess.run(
                ["sacct", "--jobs", ",".join(slurm_ids),
                 "--format", "JobID,State", "--noheader", "--parsable2"],
                capture_output=True, text=True, check=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            return {jid: "UNKNOWN" for jid in slurm_ids}
        states: dict[str, str] = {}
        for line in r.stdout.strip().splitlines():
            parts = line.split("|")
            if len(parts) < 2:
                continue
            jid, state = parts[0].strip(), parts[1].strip().split()[0]
            if "." not in jid and jid in slurm_ids:
                states[jid] = state
        for jid in slurm_ids:
            states.setdefault(jid, "PENDING")
        return states

    def _query_progress():
        try:
            sidecar = scan_dir / "launch_meta.json"
            with open(sidecar) as f:
                meta = json.load(f)
            n_total = len(meta["valid_entries"])
            n_done  = sum(
                1 for ii in range(n_total)
                if (scan_dir / f"scan_{ii:04d}.h5").exists()
            )
            return n_done, n_total
        except Exception:
            return 0, 0

    def _bar(done, total, width=30):
        if not total:
            return f"[{'?' * width}]"
        filled = int(width * done / total)
        return f"[{'█' * filled}{'░' * (width - filled)}]"

    def _build_snapshot(states, n_done, n_total, elapsed) -> str:
        pct   = 100 * n_done / n_total if n_total else 0
        lines = [
            f"{'─'*56}",
            f"  s3dxrd SLURM monitor   elapsed: {timedelta(seconds=int(elapsed))}",
            f"{'─'*56}",
            f"  Jobs   pending={sum(s == 'PENDING' for s in states.values())}  "
            f"running={sum(s == 'RUNNING' for s in states.values())}  "
            f"done={sum(s in _DONE for s in states.values())}  "
            f"failed={sum(s in _FAILED for s in states.values())}",
            f"  Scans  {_bar(n_done, n_total)}  {n_done}/{n_total}  ({pct:.1f}%)",
        ]
        if n_done > 0 and elapsed > 0:
            rate = n_done / elapsed
            eta  = (n_total - n_done) / rate if rate else float("inf")
            lines.append(
                f"  Rate   {rate * 3600:.1f} scans/hr  |  ETA {timedelta(seconds=int(eta))}"
            )
        lines.append(f"{'─'*56}")
        return "\n".join(lines)

    def _erase(n_lines: int) -> None:
        try:
            from IPython.display import clear_output
            clear_output(wait=True)
        except ImportError:
            import sys
            sys.stdout.write(f"\033[{n_lines}A\033[J")
            sys.stdout.flush()

    t0        = time.time()
    prev_lines = 0
    while True:
        states          = _query_slurm()
        n_done, n_total = _query_progress()
        snapshot        = _build_snapshot(states, n_done, n_total, time.time() - t0)

        if prev_lines:
            _erase(prev_lines)
        print(snapshot)
        prev_lines = snapshot.count("\n") + 1

        still_running = any(s in _RUNNING for s in states.values())
        if not watch or not still_running:
            break
        time.sleep(interval)


def merge(
    scan_dir: Union[str, Path],
    output_file: Union[str, Path],
    *,
    overwrite: bool = False,
) -> dict:
    """
    Build ``segmented.h5`` as an index of HDF5 external links into *scan_dir*.

    Each per-scan ``scan_XXXX.h5`` written by the workers is referenced via an
    ``h5py.ExternalLink`` rather than copied.  This makes merge essentially
    instant (no data read or written) and leaves the scan files as the
    authoritative data store.

    .. note::
       ``segmented.h5`` uses **relative** paths for its external links, so
       moving it without moving *scan_dir* alongside will break the links.
       Move both together, or keep ``segmented.h5`` in the parent of *scan_dir*.

    Args:
        scan_dir: Directory containing per-scan ``scan_XXXX.h5`` files and
            ``launch_meta.json``, as created by :func:`launch`.
        output_file: Destination index file (e.g. ``segmented.h5``).  Created
            if absent; existing links are skipped unless *overwrite* is ``True``.
        overwrite: If ``True``, re-link scans already present in *output_file*.

    Returns:
        dict with keys ``'n_linked'``, ``'n_skipped'``, ``'n_missing'``.
        To load the peak data call
        :func:`~nrxrdct.xrdct.s3dxrd.load_segmentation` on *output_file*.
    """
    scan_dir    = Path(scan_dir)
    output_file = Path(output_file)

    sidecar = scan_dir / "launch_meta.json"
    if not sidecar.exists():
        raise FileNotFoundError(
            f"launch_meta.json not found in {scan_dir}. Was launch() called?"
        )
    with open(sidecar) as f:
        launch_meta = json.load(f)

    valid_entries     = launch_meta["valid_entries"]
    translation_motor = launch_meta.get("translation_motor", "dty")
    n_total           = len(valid_entries)

    n_linked = n_skipped = n_missing = 0

    with h5py.File(output_file, "a") as hout:
        for ii in tqdm(range(n_total), desc="Linking scans"):
            gp      = f"segmented/scan_{ii:04d}"
            h5_path = scan_dir / f"scan_{ii:04d}.h5"

            if gp in hout:
                if not overwrite:
                    n_skipped += 1
                    continue
                del hout[gp]

            if not h5_path.exists():
                print(f"  ⚠  scan_{ii:04d}.h5 missing — skipping")
                n_missing += 1
                continue

            # Prefer relative path so the pair (output_file, scan_dir) can be
            # moved together without breaking links.
            try:
                rel = h5_path.relative_to(output_file.parent)
            except ValueError:
                rel = h5_path  # different drive / mount: fall back to absolute
            hout[gp] = h5py.ExternalLink(str(rel), "/scan")
            n_linked += 1

    print(
        f"\n✓  {n_linked} scans linked, {n_skipped} already present, "
        f"{n_missing} missing"
    )
    return {"n_linked": n_linked, "n_skipped": n_skipped, "n_missing": n_missing}


def check(
    scan_dir: Union[str, Path],
    output_file: Union[str, Path],
) -> dict:
    """
    Report completeness: how many scans have ``.h5`` files and how many are linked.

    Args:
        scan_dir: Scan directory written by the workers.
        output_file: Index file produced by :func:`merge` (``segmented.h5``).

    Returns:
        dict with keys ``'n_total'``, ``'n_scans'``, ``'n_linked'``,
        ``'missing_indices'``.
    """
    scan_dir    = Path(scan_dir)
    output_file = Path(output_file)

    with open(scan_dir / "launch_meta.json") as f:
        launch_meta = json.load(f)
    n_total = len(launch_meta["valid_entries"])

    n_scans  = sum(1 for ii in range(n_total)
                   if (scan_dir / f"scan_{ii:04d}.h5").exists())
    n_linked = 0
    if output_file.exists():
        with h5py.File(output_file, "r") as hout:
            if "segmented" in hout:
                n_linked = len(hout["segmented"])

    missing = [ii for ii in range(n_total)
               if not (scan_dir / f"scan_{ii:04d}.h5").exists()]

    print(f"Total scans : {n_total}")
    print(f"Scan files  : {n_scans} / {n_total}  ({100 * n_scans / n_total:.1f} %)")
    print(f"Linked      : {n_linked} / {n_total}  ({100 * n_linked / n_total:.1f} %)")
    if missing:
        preview = missing[:20]
        suffix  = "…" if len(missing) > 20 else ""
        print(f"Missing indices: {preview}{suffix}")
    else:
        print("All scans present ✓")

    return {
        "n_total":         n_total,
        "n_scans":         n_scans,
        "n_linked":        n_linked,
        "missing_indices": missing,
    }
