"""
nrxrdct.laue.workers.slurm_guided_seg_worker
--------------------------------------------
SLURM worker for simulation-guided segmentation.

Uses a **single fixed orientation** (whatever U matrices are baked into the
serialised stack) to predict spot positions for every frame, then runs
:func:`~nrxrdct.laue.segmentation.simulation_guided_segmentation` on each
raw image.  This is the right strategy when one good orientation estimate
(e.g. from interactive indexing) applies to the whole map — which is
typically true for well-ordered substrate materials.

Invoked by :meth:`LayeredMap.submit_guided_segmentation` via::

    python -m nrxrdct.laue.workers.slurm_guided_seg_worker \\
        --meta-json path/to/guided_seg_meta.json \\
        --frame-indices 0,1,2,...
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import dill as _pickle
import h5py
import numpy as np


def main() -> None:
    p = argparse.ArgumentParser(description="Simulation-guided segmentation worker")
    p.add_argument("--meta-json",     required=True)
    p.add_argument("--frame-indices", required=True)
    args = p.parse_args()

    with open(args.meta_json) as fh:
        meta = json.load(fh)

    frame_indices = [int(x) for x in args.frame_indices.split(",")]

    from nrxrdct.laue.camera import Camera
    from nrxrdct.laue.simulation import simulate_laue_stack
    from nrxrdct.laue.segmentation import simulation_guided_segmentation

    # ── Load stack (U matrices already set to the user's orientation estimate)
    with open(meta["stack_pkl"], "rb") as fh:
        stack = _pickle.load(fh)
    camera = Camera(**meta["camera"])

    # ── Simulate spots once — same prediction for every frame ─────────────────
    _SIM_KEYS = ("E_min_eV", "E_max_eV", "f2_thresh", "geometry_only",
                 "structure_model", "correct_depth")
    sim_kwargs = {k: meta[k] for k in _SIM_KEYS if k in meta}

    t_sim = time.time()
    spots = simulate_laue_stack(stack, camera, **sim_kwargs)
    print(f"  {len(spots)} spots simulated ({time.time() - t_sim:.2f}s)", flush=True)

    # ── Load mask ─────────────────────────────────────────────────────────────
    mask = np.load(meta["mask_path"]).astype(bool)

    # ── Segmentation kwargs ───────────────────────────────────────────────────
    _SEG_KEYS = (
        "psf_sigma", "search_radius", "min_distance",
        "min_snr", "bg_sigma", "d", "r_squared_min",
        "include_unfitted", "fit_spots",
    )
    seg_kwargs = {k: meta[k] for k in _SEG_KEYS if k in meta}

    seg_dir   = meta["seg_dir"]
    overwrite = meta.get("overwrite", False)

    print(
        f"Guided-seg worker — {len(frame_indices)} frames | "
        f"{len(spots)} predicted spots",
        flush=True,
    )

    # ── Load images in one I/O pass ───────────────────────────────────────────
    t_io = time.time()
    frames: dict[int, np.ndarray] = {}
    with h5py.File(meta["h5_path"], "r") as hf:
        ds         = hf[meta["h5_dataset"]]
        monitor_ds = hf.get(meta["monitor"]) if meta.get("monitor") else None
        for fi in frame_indices:
            try:
                img = ds[fi].astype(np.float32)
                if monitor_ds is not None:
                    mon_val = float(monitor_ds[fi])
                    if mon_val > 0:
                        img /= mon_val
                frames[fi] = img
            except Exception as exc:
                print(f"  ✗  frame {fi}: image read: {exc}", flush=True)
    print(f"  {len(frames)} images loaded ({time.time() - t_io:.1f}s)", flush=True)

    # ── Run guided segmentation per frame ─────────────────────────────────────
    t0   = time.time()
    n_ok = 0

    for fi in frame_indices:
        if fi not in frames:
            continue

        out_path = os.path.join(seg_dir, f"frame_{fi:05d}.h5")
        if os.path.exists(out_path) and not overwrite:
            n_ok += 1
            continue

        try:
            simulation_guided_segmentation(
                frames[fi], spots, mask,
                outpath=out_path, overwrite=True,
                **seg_kwargs,
            )
            n_ok += 1
        except Exception as exc:
            print(f"  ✗  frame {fi}: {exc}", flush=True)

    print(
        f"Guided-seg worker done — {n_ok}/{len(frame_indices)} frames "
        f"({time.time() - t0:.1f}s)",
        flush=True,
    )
    sys.exit(0)


if __name__ == "__main__":
    main()
