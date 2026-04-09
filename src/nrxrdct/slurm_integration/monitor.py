"""
nrxrdct.slurm_integration.monitor
-----------------------------------
Monitor SLURM job progress by polling both SLURM state and the tmp directory.

Python API
----------
    from nrxrdct.slurm_integration import monitor

    monitor(slurm_ids=["12345", "12346"], tmp_dir=Path("output_tmp"))
    monitor(slurm_ids=[...], tmp_dir=Path("output_tmp"), watch=True, interval=30)

CLI
---
    nrxrdct-slurm monitor --slurm-ids 12345,12346 --tmp-dir output_tmp [--watch]
"""

from __future__ import annotations

import json
import subprocess
import time
from datetime import timedelta
from pathlib import Path

_RUNNING_STATES = {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "REQUEUED"}
_DONE_STATES    = {"COMPLETED"}
_FAILED_STATES  = {"FAILED", "OUT_OF_MEMORY", "TIMEOUT", "CANCELLED",
                   "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE"}


def _query_slurm(slurm_ids: list[str]) -> dict[str, str]:
    if not slurm_ids:
        return {}
    try:
        result = subprocess.run(
            ["sacct", "--jobs", ",".join(slurm_ids),
             "--format", "JobID,State", "--noheader", "--parsable2"],
            capture_output=True, text=True, check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {jid: "UNKNOWN" for jid in slurm_ids}

    states: dict[str, str] = {}
    for line in result.stdout.strip().splitlines():
        parts = line.split("|")
        if len(parts) < 2:
            continue
        job_id, state = parts[0].strip(), parts[1].strip().split()[0]
        if "." not in job_id and job_id in slurm_ids:
            states[job_id] = state
    for jid in slurm_ids:
        states.setdefault(jid, "PENDING")
    return states


def _query_progress(tmp_dir: Path) -> tuple[int, int]:
    """Return (n_done, n_total) by counting completed .npy files."""
    try:
        meta_sidecar = tmp_dir / "launch_meta.json"
        if not meta_sidecar.exists():
            return 0, 0
        with open(meta_sidecar) as f:
            launch_meta = json.load(f)
        n_total = len(launch_meta["valid_entries"])
        n_done  = 0
        for p in tmp_dir.glob("scan_????.npy"):
            ii = int(p.stem.split("_")[1])
            if (tmp_dir / f"scan_{ii:04d}.meta.json").exists():
                n_done += 1
        for p in tmp_dir.glob("scan_????.npy.tmp.npy"):
            ii = int(p.name.split("_")[1].split(".")[0])
            metas = (list(tmp_dir.glob(f"scan_{ii:04d}.meta.meta.json.tmp")) +
                     list(tmp_dir.glob(f"scan_{ii:04d}.meta.json.tmp")))
            if metas:
                n_done += 1
        return n_done, n_total
    except Exception:
        return 0, 0


def _fmt_duration(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))


def _progress_bar(done: int, total: int, width: int = 30) -> str:
    if total == 0:
        return f"[{'?' * width}]"
    filled = int(width * done / total)
    return f"[{'█' * filled}{'░' * (width - filled)}]"


def _render_snapshot(
    slurm_ids: list[str],
    states: dict[str, str],
    n_done: int,
    n_total: int,
    elapsed: float,
) -> str:
    lines = []
    n_pending  = sum(1 for s in states.values() if s == "PENDING")
    n_running  = sum(1 for s in states.values() if s == "RUNNING")
    n_done_j   = sum(1 for s in states.values() if s in _DONE_STATES)
    n_failed   = sum(1 for s in states.values() if s in _FAILED_STATES)
    n_unknown  = sum(1 for s in states.values() if s == "UNKNOWN")

    lines.append(f"\n{'─'*56}")
    lines.append(f"  nrxrdct SLURM monitor   elapsed: {_fmt_duration(elapsed)}")
    lines.append(f"{'─'*56}")
    lines.append(f"  Jobs total   : {len(slurm_ids)}")
    lines.append(f"  ⏳ Pending   : {n_pending}")
    lines.append(f"  ▶  Running   : {n_running}")
    lines.append(f"  ✓  Completed : {n_done_j}")
    lines.append(f"  ✗  Failed    : {n_failed}")
    if n_unknown:
        lines.append(f"  ?  Unknown   : {n_unknown}")

    lines.append(f"{'─'*56}")
    pct = 100 * n_done / n_total if n_total else 0
    lines.append(
        f"  Scans  {_progress_bar(n_done, n_total)}  "
        f"{n_done}/{n_total}  ({pct:.1f}%)"
    )

    if n_done > 0 and n_total > 0 and elapsed > 0:
        rate      = n_done / elapsed
        remaining = (n_total - n_done) / rate
        lines.append(f"  Rate   {rate * 3600:.1f} scans/hr")
        lines.append(f"  ETA    {_fmt_duration(remaining)}")
    else:
        lines.append(f"  ETA    —  (waiting for first scan)")

    lines.append(f"{'─'*56}")
    lines.append(f"  {'Job ID':<12} {'State':<16}")
    lines.append(f"  {'──────':<12} {'─────':<16}")
    for jid in slurm_ids:
        state = states.get(jid, "UNKNOWN")
        icon  = (
            "⏳" if state == "PENDING"      else
            "▶ " if state == "RUNNING"      else
            "✓ " if state in _DONE_STATES   else
            "✗ " if state in _FAILED_STATES else
            "? "
        )
        lines.append(f"  {jid:<12} {icon} {state}")
    lines.append(f"{'─'*56}\n")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def monitor(
    slurm_ids: list[str],
    tmp_dir: Path,
    *,
    watch: bool = False,
    interval: int = 30,
    start_time: float | None = None,
) -> dict:
    """
    Monitor SLURM integration jobs.

    Progress is measured by counting completed .npy files in tmp_dir,
    giving an accurate per-scan view independent of SLURM job boundaries.

    Parameters
    ----------
    slurm_ids  : list[str]  — SLURM job IDs returned by launch()
    tmp_dir    : Path       — tmp directory written by workers
    watch      : bool       — block until all jobs finish (default: False)
    interval   : int        — seconds between polls when watch=True
    start_time : float      — unix timestamp of job submission (default: now)

    Returns
    -------
    dict with keys 'states', 'n_done', 'n_total', 'elapsed',
    'all_done', 'any_failed'.
    """
    tmp_dir = Path(tmp_dir)
    t0      = start_time or time.time()

    def _snapshot() -> dict:
        states          = _query_slurm(slurm_ids)
        n_done, n_total = _query_progress(tmp_dir)
        elapsed         = time.time() - t0
        print(_render_snapshot(slurm_ids, states, n_done, n_total, elapsed))
        all_done   = all(s in _DONE_STATES | _FAILED_STATES for s in states.values())
        any_failed = any(s in _FAILED_STATES for s in states.values())
        return {
            "states":     states,
            "n_done":     n_done,
            "n_total":    n_total,
            "elapsed":    elapsed,
            "all_done":   all_done,
            "any_failed": any_failed,
        }

    if not watch:
        return _snapshot()

    print(f"Watching {len(slurm_ids)} jobs — polling every {interval}s. "
          f"Press Ctrl+C to stop.\n")
    result = {}
    try:
        while True:
            result = _snapshot()
            if result["all_done"]:
                if result["any_failed"]:
                    print("⚠  Some jobs failed. Run check() to find missing scans,\n"
                          "   then repair() to resubmit them.")
                else:
                    print(f"✓  All jobs completed in {_fmt_duration(result['elapsed'])}.")
                    print(f"   Run merge() to assemble the output HDF5.")
                break
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nMonitor interrupted.")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser(sub=None):
    import argparse
    desc = "Monitor SLURM powder integration jobs"
    p = (
        sub.add_parser("monitor", help=desc, description=desc)
        if sub else
        argparse.ArgumentParser(description=desc)
    )
    p.add_argument("--slurm-ids",  required=True,
                   help="Comma-separated SLURM job IDs from launch()")
    p.add_argument("--tmp-dir",    required=True, type=Path,
                   help="Tmp directory written by workers")
    p.add_argument("--watch",      action="store_true",
                   help="Poll until all jobs finish (blocking)")
    p.add_argument("--interval",   type=int, default=30)
    return p


def _cli_monitor(args):
    slurm_ids = [s.strip() for s in args.slurm_ids.split(",")]
    monitor(
        slurm_ids = slurm_ids,
        tmp_dir   = args.tmp_dir,
        watch     = args.watch,
        interval  = args.interval,
    )


if __name__ == "__main__":
    p = _build_parser()
    _cli_monitor(p.parse_args())

# """
# nrxrdct.slurm_integration.monitor
# -----------------------------------
# Monitor the progress of SLURM powder integration jobs.

# Two modes
# ---------
# - **Snapshot** (default): print a single status table and return immediately.
# - **Watch** (blocking):   poll SLURM + the output HDF5 at a fixed interval
#                           until all jobs finish or fail.

# Python API
# ----------
#     from nrxrdct.slurm_integration import monitor

#     # One-shot snapshot
#     monitor(slurm_ids=["12345", "12346"], output_file=Path("output.h5"))

#     # Blocking watch until done
#     monitor(
#         slurm_ids   = ["12345", "12346"],
#         output_file = Path("output.h5"),
#         watch       = True,
#         interval    = 30,    # seconds between polls
#     )

# CLI
# ---
#     nrxrdct-slurm monitor --slurm-ids 12345,12346 \\
#         --output-file output.h5 [--watch] [--interval 30]
# """

# from __future__ import annotations

# import subprocess
# import time
# from datetime import timedelta
# from pathlib import Path

# import h5py

# # ─────────────────────────────────────────────────────────────────────────────
# # SLURM query
# # ─────────────────────────────────────────────────────────────────────────────

# # SLURM states that mean the job is still alive
# _RUNNING_STATES = {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "REQUEUED"}
# _DONE_STATES    = {"COMPLETED"}
# _FAILED_STATES  = {"FAILED", "OUT_OF_MEMORY", "TIMEOUT", "CANCELLED", "NODE_FAIL",
#                    "PREEMPTED", "BOOT_FAIL", "DEADLINE"}


# def _query_slurm(slurm_ids: list[str]) -> dict[str, str]:
#     """
#     Return {job_id: state} for each id using `sacct`.
#     Falls back to 'UNKNOWN' if sacct is unavailable or the job is too old.
#     """
#     if not slurm_ids:
#         return {}
#     try:
#         result = subprocess.run(
#             [
#                 "sacct",
#                 "--jobs", ",".join(slurm_ids),
#                 "--format", "JobID,State",
#                 "--noheader",
#                 "--parsable2",   # pipe-delimited, no trailing pipe
#             ],
#             capture_output=True, text=True, check=True,
#         )
#     except (subprocess.CalledProcessError, FileNotFoundError):
#         return {jid: "UNKNOWN" for jid in slurm_ids}

#     # sacct may return sub-step rows like "12345.batch" — keep only the main rows
#     states: dict[str, str] = {}
#     for line in result.stdout.strip().splitlines():
#         parts = line.split("|")
#         if len(parts) < 2:
#             continue
#         job_id, state = parts[0].strip(), parts[1].strip().split()[0]
#         if "." not in job_id and job_id in slurm_ids:
#             states[job_id] = state

#     # Jobs not yet in sacct (very recently submitted) show as PENDING
#     for jid in slurm_ids:
#         states.setdefault(jid, "PENDING")

#     return states


# # ─────────────────────────────────────────────────────────────────────────────
# # HDF5 progress query
# # ─────────────────────────────────────────────────────────────────────────────

# def _query_progress(output_file: Path) -> tuple[int, int]:
#     """
#     Return (n_done, n_total) by counting completed scan datasets in the
#     output HDF5 file.
#     """
#     try:
#         with h5py.File(output_file, "r") as h:
#             if "meta/valid_entries" not in h:
#                 return 0, 0
#             n_total = len(h["meta/valid_entries"])
#             n_done  = sum(
#                 1 for i in range(n_total)
#                 if f"integrated/scan_{i:04d}" in h
#             )
#             return n_done, n_total
#     except Exception:
#         return 0, 0


# # ─────────────────────────────────────────────────────────────────────────────
# # Formatting helpers
# # ─────────────────────────────────────────────────────────────────────────────

# def _fmt_duration(seconds: float) -> str:
#     """Format *seconds* as a human-readable ``HH:MM:SS`` string."""
#     return str(timedelta(seconds=int(seconds)))


# def _progress_bar(done: int, total: int, width: int = 30) -> str:
#     """Return a fixed-width ASCII progress bar string, e.g. ``[████████░░░░]``."""
#     if total == 0:
#         return f"[{'?' * width}]"
#     filled = int(width * done / total)
#     return f"[{'█' * filled}{'░' * (width - filled)}]"


# def _render_snapshot(
#     slurm_ids: list[str],
#     states: dict[str, str],
#     n_done: int,
#     n_total: int,
#     elapsed: float,
# ) -> str:
#     """
#     Render a status snapshot as a multi-line string.

#     Includes a per-job state table, an ASCII scan progress bar, throughput
#     rate, and ETA estimate.

#     Parameters
#     ----------
#     slurm_ids : list of str
#         Ordered list of SLURM job IDs being tracked.
#     states : dict
#         Mapping of job ID to SLURM state string (from :func:`_query_slurm`).
#     n_done : int
#         Number of scan datasets written to the output file so far.
#     n_total : int
#         Total expected number of scans.
#     elapsed : float
#         Wall-clock seconds since jobs were submitted.

#     Returns
#     -------
#     str
#         Formatted status block ready to print.
#     """
#     lines = []

#     # ── Job table ─────────────────────────────────────────────────────────────
#     n_pending  = sum(1 for s in states.values() if s in _RUNNING_STATES and s == "PENDING")
#     n_running  = sum(1 for s in states.values() if s == "RUNNING")
#     n_done_j   = sum(1 for s in states.values() if s in _DONE_STATES)
#     n_failed   = sum(1 for s in states.values() if s in _FAILED_STATES)
#     n_unknown  = sum(1 for s in states.values() if s == "UNKNOWN")

#     lines.append(f"\n{'─'*56}")
#     lines.append(f"  nrxrdct SLURM monitor   elapsed: {_fmt_duration(elapsed)}")
#     lines.append(f"{'─'*56}")
#     lines.append(f"  Jobs total   : {len(slurm_ids)}")
#     lines.append(f"  ⏳ Pending   : {n_pending}")
#     lines.append(f"  ▶  Running   : {n_running}")
#     lines.append(f"  ✓  Completed : {n_done_j}")
#     lines.append(f"  ✗  Failed    : {n_failed}")
#     if n_unknown:
#         lines.append(f"  ?  Unknown   : {n_unknown}")

#     # ── Scan progress ─────────────────────────────────────────────────────────
#     lines.append(f"{'─'*56}")
#     pct = 100 * n_done / n_total if n_total else 0
#     lines.append(
#         f"  Scans  {_progress_bar(n_done, n_total)}  "
#         f"{n_done}/{n_total}  ({pct:.1f}%)"
#     )

#     # ── ETA ───────────────────────────────────────────────────────────────────
#     if n_done > 0 and n_total > 0 and elapsed > 0:
#         rate        = n_done / elapsed          # scans/second
#         remaining   = (n_total - n_done) / rate
#         lines.append(f"  Rate   {rate * 3600:.1f} scans/hr")
#         lines.append(f"  ETA    {_fmt_duration(remaining)}")
#     else:
#         lines.append(f"  ETA    —  (waiting for first scan to complete)")

#     # ── Per-job detail ────────────────────────────────────────────────────────
#     lines.append(f"{'─'*56}")
#     lines.append(f"  {'Job ID':<12} {'State':<16}")
#     lines.append(f"  {'──────':<12} {'─────':<16}")
#     for jid in slurm_ids:
#         state = states.get(jid, "UNKNOWN")
#         icon  = (
#             "⏳" if state == "PENDING"         else
#             "▶ " if state == "RUNNING"         else
#             "✓ " if state in _DONE_STATES      else
#             "✗ " if state in _FAILED_STATES    else
#             "? "
#         )
#         lines.append(f"  {jid:<12} {icon} {state}")

#     lines.append(f"{'─'*56}\n")
#     return "\n".join(lines)


# # ─────────────────────────────────────────────────────────────────────────────
# # Public API
# # ─────────────────────────────────────────────────────────────────────────────

# def monitor(
#     slurm_ids: list[str],
#     output_file: Path,
#     *,
#     watch: bool = False,
#     interval: int = 30,
#     start_time: float | None = None,
# ) -> dict:
#     """
#     Monitor SLURM powder integration jobs.

#     Parameters
#     ----------
#     slurm_ids   : list[str]
#         SLURM job IDs returned by ``launch()``.
#     output_file : Path
#         Path to the output HDF5 file (used to count completed scans).
#     watch       : bool
#         If True, poll repeatedly until all jobs finish (blocking).
#         If False (default), print a single snapshot and return.
#     interval    : int
#         Seconds between polls when watch=True (default: 30).
#     start_time  : float, optional
#         Unix timestamp of when the jobs were submitted.
#         Defaults to now if not provided.

#     Returns
#     -------
#     dict with keys:
#         'states'    – {job_id: slurm_state}
#         'n_done'    – scans written to the output file
#         'n_total'   – total expected scans
#         'elapsed'   – wall-clock seconds since start_time
#         'all_done'  – True if every job reached COMPLETED or a failed state
#         'any_failed'– True if any job is in a failed state
#     """
#     output_file = Path(output_file)
#     t0          = start_time or time.time()

#     def _snapshot() -> dict:
#         states          = _query_slurm(slurm_ids)
#         n_done, n_total = _query_progress(output_file)
#         elapsed         = time.time() - t0
#         print(_render_snapshot(slurm_ids, states, n_done, n_total, elapsed))
#         all_done   = all(s in _DONE_STATES | _FAILED_STATES for s in states.values())
#         any_failed = any(s in _FAILED_STATES for s in states.values())
#         return {
#             "states":     states,
#             "n_done":     n_done,
#             "n_total":    n_total,
#             "elapsed":    elapsed,
#             "all_done":   all_done,
#             "any_failed": any_failed,
#         }

#     if not watch:
#         return _snapshot()

#     # ── Blocking watch loop ───────────────────────────────────────────────────
#     print(f"Watching {len(slurm_ids)} jobs — polling every {interval}s. "
#           f"Press Ctrl+C to stop.\n")
#     result = {}
#     try:
#         while True:
#             result = _snapshot()
#             if result["all_done"]:
#                 if result["any_failed"]:
#                     print("⚠  Some jobs failed. "
#                           "Run check(..., resubmit=True) to see which scans need rerunning,\n"
#                           "or repair(...) to fix automatically.")
#                 else:
#                     print(f"✓  All jobs completed in {_fmt_duration(result['elapsed'])}.")
#                 break
#             time.sleep(interval)
#     except KeyboardInterrupt:
#         print("\nMonitor interrupted.")

#     return result


# # ─────────────────────────────────────────────────────────────────────────────
# # CLI
# # ─────────────────────────────────────────────────────────────────────────────

# def _build_parser(sub=None):
#     """Build the ``monitor`` sub-command argument parser, attaching it to *sub* if provided."""
#     import argparse
#     desc = "Monitor SLURM powder integration jobs"
#     p = (
#         sub.add_parser("monitor", help=desc, description=desc)
#         if sub else
#         argparse.ArgumentParser(description=desc)
#     )
#     p.add_argument("--slurm-ids",    required=True,
#                    help="Comma-separated SLURM job IDs (returned by launch)")
#     p.add_argument("--output-file",  required=True, type=Path,
#                    help="Output HDF5 file (to count completed scans)")
#     p.add_argument("--watch",        action="store_true",
#                    help="Poll repeatedly until all jobs finish (blocking)")
#     p.add_argument("--interval",     type=int, default=30,
#                    help="Seconds between polls when --watch is set (default: 30)")
#     return p


# def _cli_monitor(args):
#     """Parse CLI arguments and delegate to :func:`monitor`."""
#     slurm_ids = [s.strip() for s in args.slurm_ids.split(",")]
#     monitor(
#         slurm_ids   = slurm_ids,
#         output_file = args.output_file,
#         watch       = args.watch,
#         interval    = args.interval,
#     )


# if __name__ == "__main__":
#     p = _build_parser()
#     _cli_monitor(p.parse_args())
