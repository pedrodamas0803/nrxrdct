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
    tmp_dir: Path,
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
        f' --tmp-dir "{tmp_dir}"'
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

    tmp_dir = output_file.parent / (output_file.stem + "_tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)
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
    sidecar = tmp_dir / "launch_meta.json"
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
            tmp_dir=tmp_dir,
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
    print(f"   Tmp dir : {tmp_dir}/")
    print(f"   Logs    : {log_dir}/")
    return {"slurm_ids": slurm_ids, "tmp_dir": tmp_dir, "n_scans": len(valid_entries)}


def monitor(
    slurm_ids: list,
    tmp_dir: Union[str, Path],
    watch: bool = False,
    interval: int = 30,
) -> None:
    """
    Poll SLURM job states and count completed ``.npz`` files in *tmp_dir*.

    Args:
        slurm_ids: List of SLURM job ID strings returned by :func:`launch`.
        tmp_dir: Tmp directory written by the workers (same as
            ``result["tmp_dir"]`` from :func:`launch`).
        watch: If ``True``, poll repeatedly until all jobs finish.
        interval: Seconds between polls when *watch* is ``True``.
    """
    tmp_dir   = Path(tmp_dir)
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
            sidecar = tmp_dir / "launch_meta.json"
            with open(sidecar) as f:
                meta = json.load(f)
            n_total = len(meta["valid_entries"])
            n_done  = sum(
                1 for ii in range(n_total)
                if (tmp_dir / f"scan_{ii:04d}.npz").exists()
                and (tmp_dir / f"scan_{ii:04d}.meta.json").exists()
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
    tmp_dir: Union[str, Path],
    output_file: Union[str, Path],
    *,
    overwrite: bool = False,
    n_threads: int = 8,
) -> List:
    """
    Assemble per-scan ``.npz`` tmp files into ``segmented.h5``.

    Reads ``launch_meta.json`` to determine the expected scan list. Already-
    merged scans are skipped unless *overwrite* is ``True``. The returned list
    has the same format as :func:`~nrxrdct.xrdct.s3dxrd.segment_slice`, so it
    can be fed directly to :func:`~nrxrdct.xrdct.s3dxrd.build_columnfile`.

    Reading the ``.npz`` files is done in parallel with a thread pool;
    writing to HDF5 is serial (h5py does not support concurrent writes).

    Args:
        tmp_dir: Tmp directory written by the workers.
        output_file: Destination ``segmented.h5``.
        overwrite: If ``True``, re-merge scans already present in *output_file*.
        n_threads: Number of threads for parallel ``.npz`` reading. Helps most
            on network/parallel filesystems (Lustre, GPFS).

    Returns:
        list[:class:`~nrxrdct.xrdct.s3dxrd.SegmentationResult`] in scan order.
    """
    from concurrent.futures import ThreadPoolExecutor
    from nrxrdct.xrdct.s3dxrd import SegmentationResult, _read_scan_group, _write_scan_group

    tmp_dir     = Path(tmp_dir)
    output_file = Path(output_file)

    sidecar = tmp_dir / "launch_meta.json"
    if not sidecar.exists():
        raise FileNotFoundError(
            f"launch_meta.json not found in {tmp_dir}. Was launch() called?"
        )
    with open(sidecar) as f:
        launch_meta = json.load(f)

    valid_entries     = launch_meta["valid_entries"]
    dty_values        = launch_meta["dty_values"]
    translation_motor = launch_meta.get("translation_motor", "dty")
    n_total           = len(valid_entries)

    # Determine which scans are already in the output (read-only pass).
    already_merged: set = set()
    if output_file.exists() and not overwrite:
        with h5py.File(output_file, "r") as hr:
            seg = hr.get("segmented", {})
            already_merged = {ii for ii in range(n_total) if f"scan_{ii:04d}" in seg}

    to_load = set(range(n_total)) - already_merged

    # Read .npz + .meta.json files in parallel (pure I/O, no HDF5 involved).
    def _load(ii: int):
        npz  = tmp_dir / f"scan_{ii:04d}.npz"
        meta = tmp_dir / f"scan_{ii:04d}.meta.json"
        if not npz.exists():
            return ii, None
        try:
            data = np.load(npz)
            if meta.exists():
                with open(meta) as f:
                    m = json.load(f)
                entry, dty = m["entry"], m["dty"]
            else:
                entry, dty = valid_entries[ii], dty_values[ii]
            return ii, SegmentationResult(
                entry         = entry,
                dty           = dty,
                sc            = data["sc"],
                fc            = data["fc"],
                omega         = data["omega"],
                sum_intensity = data["sum_intensity"],
                n_pixels      = data["n_pixels"],
            )
        except Exception as e:
            print(f"  ✗  scan_{ii:04d}: load error — {e}")
            return ii, None

    print(f"Loading {len(to_load)} scans with {n_threads} threads …")
    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        futures = {ii: pool.submit(_load, ii) for ii in to_load}
    # pool.__exit__ waits for all reads to finish before writing begins.

    # Write serially in scan order (HDF5 constraint).
    results: list = []
    n_merged = n_skipped = n_missing = 0

    with h5py.File(output_file, "a") as hout:
        for ii in tqdm(range(n_total), desc="Merging scans"):
            group_path = f"segmented/scan_{ii:04d}"

            if ii in already_merged:
                results.append(_read_scan_group(hout[group_path], translation_motor))
                n_skipped += 1
                continue

            _, result = futures[ii].result()
            if result is None:
                print(f"  ⚠  scan_{ii:04d}.npz missing — skipping")
                n_missing += 1
                continue

            if overwrite and group_path in hout:
                del hout[group_path]
            _write_scan_group(hout, group_path, result, translation_motor)
            results.append(result)
            n_merged += 1

    print(
        f"\n✓  Merge complete — {n_merged} merged, "
        f"{n_skipped} already done, {n_missing} missing"
    )
    return results


def check(
    tmp_dir: Union[str, Path],
    output_file: Union[str, Path],
) -> dict:
    """
    Report completeness: how many scans have tmp files and how many are merged.

    Args:
        tmp_dir: Tmp directory written by the workers.
        output_file: Destination ``segmented.h5``.

    Returns:
        dict with keys ``'n_total'``, ``'n_tmp'``, ``'n_merged'``,
        ``'missing_indices'``.
    """
    tmp_dir     = Path(tmp_dir)
    output_file = Path(output_file)

    with open(tmp_dir / "launch_meta.json") as f:
        launch_meta = json.load(f)
    n_total = len(launch_meta["valid_entries"])

    n_tmp    = sum(1 for ii in range(n_total)
                   if (tmp_dir / f"scan_{ii:04d}.npz").exists())
    n_merged = 0
    if output_file.exists():
        with h5py.File(output_file, "r") as hout:
            if "segmented" in hout:
                n_merged = len(hout["segmented"])

    missing = [ii for ii in range(n_total)
               if not (tmp_dir / f"scan_{ii:04d}.npz").exists()]

    print(f"Total scans : {n_total}")
    print(f"Tmp files   : {n_tmp} / {n_total}  ({100 * n_tmp / n_total:.1f} %)")
    print(f"Merged      : {n_merged} / {n_total}  ({100 * n_merged / n_total:.1f} %)")
    if missing:
        preview = missing[:20]
        suffix  = "…" if len(missing) > 20 else ""
        print(f"Missing indices: {preview}{suffix}")
    else:
        print("All scans present ✓")

    return {
        "n_total":         n_total,
        "n_tmp":           n_tmp,
        "n_merged":        n_merged,
        "missing_indices": missing,
    }
