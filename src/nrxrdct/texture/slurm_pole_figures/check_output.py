"""
nrxrdct.texture.slurm_pole_figures.check_output
---------------------------------------------------
Verify progress and completeness of the pole-figure extraction pipeline.

Two stages can be checked independently:

1. **Extraction progress** — counts .npy files in the tmp directory
   (before merge).
2. **Merge completeness** — counts scan datasets in the output HDF5
   (after merge).

Python API
----------
    from nrxrdct.texture.slurm_pole_figures import check

    check(tmp_dir=Path("pole_figures_tmp"))
    check(output_file=Path("pole_figures.h5"), hkl_label="111")
    check(tmp_dir=Path("pole_figures_tmp"), output_file=Path("pole_figures.h5"))

CLI
---
    nrxrdct-slurm-texture check --tmp-dir pole_figures_tmp [--output-file pole_figures.h5]
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import h5py

# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def check(
    tmp_dir: Optional[Path] = None,
    output_file: Optional[Path] = None,
    hkl_label: Optional[str] = None,
) -> dict:
    """
    Verify extraction progress and/or merge completeness.

    Args:
        tmp_dir (Path, optional): Tmp directory written by workers. If provided,
            counts completed .npy files and reports missing indices.
        output_file (Path, optional): Output HDF5 file. If provided, counts
            merged scan datasets.
        hkl_label (str, optional): Ring identifier. Required together with
            *output_file* unless *tmp_dir* is also given (in which case it is
            read from launch_meta.json).

    Returns:
        dict: With keys ``'n_total'``, ``'n_extracted'``, ``'n_merged'``,
            ``'missing_tmp'``, ``'missing_h5'``.
    """
    if tmp_dir is None and output_file is None:
        raise ValueError("Provide at least one of tmp_dir or output_file.")

    result = {
        "n_total": 0,
        "n_extracted": 0,
        "n_merged": 0,
        "missing_tmp": [],
        "missing_h5": [],
    }

    if tmp_dir is not None:
        tmp_dir = Path(tmp_dir)
        meta_sidecar = tmp_dir / "launch_meta.json"
        if not meta_sidecar.exists():
            raise FileNotFoundError(
                f"launch_meta.json not found in {tmp_dir}. Was launch() called?"
            )
        with open(meta_sidecar) as f:
            launch_meta = json.load(f)

        valid_entries = launch_meta["valid_entries"]
        hkl_label = hkl_label or launch_meta["hkl_label"]
        n_total = len(valid_entries)
        result["n_total"] = n_total

        extracted = set()
        for p in tmp_dir.glob("scan_????.npy"):
            ii = int(p.stem.split("_")[1])
            if (tmp_dir / f"scan_{ii:04d}.meta.json").exists():
                extracted.add(ii)

        missing_tmp = sorted(set(range(n_total)) - extracted)
        result["n_extracted"] = len(extracted)
        result["missing_tmp"] = missing_tmp

        print(f"\n{'='*60}")
        print(f"  Extraction progress  ({tmp_dir.name})")
        print(f"{'='*60}")
        print(f"  Expected    : {n_total}")
        print(f"  Extracted   : {len(extracted)}")
        print(f"  Remaining   : {len(missing_tmp)}")
        if missing_tmp:
            print(f"  Missing idx : {missing_tmp}")
        else:
            print(f"  ✓  All scans extracted — ready to merge.")

    if output_file is not None:
        output_file = Path(output_file)
        if hkl_label is None:
            raise ValueError("hkl_label is required when checking output_file directly.")
        group = f"pole_figures/{hkl_label}"

        if not output_file.exists():
            print(f"\n  ⚠  Output file not found: {output_file}")
            print(f"     Run merge() first.")
            return result

        with h5py.File(output_file, "r") as hout:
            if f"motors/dty" not in hout and "motors" not in hout:
                pass  # motors group name varies with translation_motor; not fatal here

            n_total = max(result["n_total"], 0)
            if group in hout:
                merged = {
                    int(key.split("_")[1])
                    for key in hout[group].keys()
                    if key.startswith("scan_")
                }
            else:
                merged = set()

            if n_total == 0:
                n_total = max(merged, default=-1) + 1
            missing_h5 = sorted(set(range(n_total)) - merged)

            result["n_total"] = n_total
            result["n_merged"] = len(merged)
            result["missing_h5"] = missing_h5

            has_azimuthal = f"{group}/azimuthal" in hout
            n_chi = hout[f"{group}/azimuthal"].shape[0] if has_azimuthal else None

        print(f"\n{'='*60}")
        print(f"  Merge completeness    ({output_file.name}) [{group}]")
        print(f"{'='*60}")
        print(f"  Expected  : {n_total}")
        print(f"  Merged    : {len(merged)}")
        print(f"  Missing   : {len(missing_h5)}")
        if missing_h5:
            print(f"  Missing idx : {missing_h5}")
        else:
            print(f"  ✓  All scans merged.")
        if n_chi:
            print(f"  Azimuthal : {n_chi} bins")

    print()
    return result


# ─────────────────────────────────────────────────────────────────────────────
# repair() — resubmit missing scans
# ─────────────────────────────────────────────────────────────────────────────


def repair(
    tmp_dir: Path,
    master_file: Optional[Path] = None,
    poni_file: Optional[Path] = None,
    mask_file: Optional[Path] = None,
    *,
    output_file: Optional[Path] = None,
    n_jobs: int = 1,
    watch: bool = False,
    interval: int = 30,
    **kwargs,
) -> dict:
    """
    Resubmit SLURM jobs for any scans missing from the tmp directory.

    All extraction and SLURM settings are read from launch_meta.json so you
    don't need to repeat them. Pass **kwargs to override any individual
    setting (e.g. partition, mem, n_workers).

    Args:
        tmp_dir (Path): Tmp directory from the original launch().
        master_file (Path, optional): Override master HDF5 (default: from launch_meta).
        poni_file (Path, optional): Override calibration file (default: from launch_meta).
        mask_file (Path, optional): Override mask file (default: from launch_meta).
        output_file (Path, optional): Only used to check merge status if provided.
        n_jobs (int): Number of repair jobs. Defaults to 1.
        watch (bool): Block until repair jobs finish.
        interval (int): Polling interval in seconds when watch=True.
        **kwargs: Override any setting from launch_meta (partition, mem, etc.).
    """
    from .launch_jobs import _split_indices, _submit_job

    tmp_dir = Path(tmp_dir)
    with open(tmp_dir / "launch_meta.json") as f:
        lm = json.load(f)

    result = check(tmp_dir=tmp_dir, output_file=output_file, hkl_label=lm["hkl_label"])

    missing = result["missing_tmp"]
    if not missing:
        print("✓  Nothing to repair — all scans present in tmp dir.")
        return result

    print(f"\n🔧  Repairing {len(missing)} missing scans across {n_jobs} job(s)...")

    _master_file = Path(kwargs.pop("master_file", master_file or lm["master_file"]))
    _poni_file = Path(kwargs.pop("poni_file", poni_file or lm["poni_file"]))
    _mask_file = Path(kwargs.pop("mask_file", mask_file or lm["mask_file"]))
    _env_activate = kwargs.pop("env_activate", lm.get("env_activate"))
    _env_activate = Path(_env_activate) if _env_activate else None

    background_ranges = lm.get("background_ranges")
    background_ranges = (
        (tuple(background_ranges[0]), tuple(background_ranges[1]))
        if background_ranges is not None
        else None
    )

    settings = dict(
        tth_range=tuple(lm["tth_range"]),
        hkl_label=lm["hkl_label"],
        background_ranges=background_ranges,
        npt_rad=lm.get("npt_rad", 1000),
        npt_azim=lm.get("npt_azim", 360),
        n_workers=lm.get("n_workers"),
        batch_size=lm.get("batch_size", 32),
        camera_name=lm.get("camera_name", "eiger"),
        monitor_name=lm.get("monitor_name", "fpico6"),
        rotation_motor=lm.get("rotation_motor", "rot"),
        partition=lm.get("partition", "nice"),
        time=lm.get("time", "04:00:00"),
        mem=lm.get("mem", "32G"),
        cpus=lm.get("cpus", 16),
        gpu=lm.get("gpu", False),
        conda_env=lm.get("conda_env", None),
    )
    settings.update(kwargs)

    log_dir = Path(lm["output_file"]).parent / "slurm_logs_texture"
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
            tth_range=settings["tth_range"],
            hkl_label=settings["hkl_label"],
            background_ranges=settings["background_ranges"],
            npt_rad=settings["npt_rad"],
            npt_azim=settings["npt_azim"],
            n_workers=settings["n_workers"],
            batch_size=settings["batch_size"],
            camera_name=settings["camera_name"],
            monitor_name=settings["monitor_name"],
            rotation_motor=settings["rotation_motor"],
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

    print(f"\n✓  {len(slurm_ids)} repair job(s) submitted — IDs: {', '.join(slurm_ids)}")

    if watch:
        from .monitor import monitor as _monitor

        _monitor(slurm_ids=slurm_ids, tmp_dir=tmp_dir, watch=True, interval=interval)

    result["repair_job_ids"] = slurm_ids
    return result


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def _build_parser(sub=None):
    import argparse

    desc = "Check pole-figure extraction progress and merge completeness"
    p = (
        sub.add_parser("check", help=desc, description=desc)
        if sub
        else argparse.ArgumentParser(description=desc)
    )
    p.add_argument("--tmp-dir", type=Path, default=None,
                    help="Tmp directory from launch() — checks .npy progress")
    p.add_argument("--output-file", type=Path, default=None,
                    help="Output HDF5 — checks merge completeness")
    p.add_argument("--hkl-label", default=None,
                    help="Required with --output-file if --tmp-dir is not also given")
    return p


def _cli_check(args):
    check(tmp_dir=args.tmp_dir, output_file=args.output_file, hkl_label=args.hkl_label)


if __name__ == "__main__":
    p = _build_parser()
    _cli_check(p.parse_args())