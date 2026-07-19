"""
nrxrdct.laue.workers.slurm_full_detector_image_worker
--------------------------------------------------------
SLURM worker for one row-chunk of a whole-frame Laue simulation.

Loads the pickled stack + camera from the shared job metadata, calls
:func:`~nrxrdct.laue.simulation.simulate_full_detector_image` for the
assigned `[row_start, row_end)` slice of the binned-row grid, and writes
the result to `chunks_dir/chunk_{chunk_index:04d}.npz`.

Invoked by :func:`~nrxrdct.laue.simulation.submit_full_detector_image` via::

    python -m nrxrdct.laue.workers.slurm_full_detector_image_worker \\
        --meta-json path/to/full_detector_image_meta.json \\
        --chunk-index 0 --row-start 0 --row-end 51

Combine all chunks afterwards with
:func:`~nrxrdct.laue.simulation.collect_full_detector_image`.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import dill as pickle
import numpy as np

from nrxrdct.laue.camera import Camera
from nrxrdct.laue.simulation import simulate_full_detector_image


def _camera_from_dict(d: dict) -> Camera:
    return Camera(
        dd=d["dd"],
        xcen=d["xcen"],
        ycen=d["ycen"],
        xbet=d["xbet"],
        xgam=d["xgam"],
        pixelsize=d["pixelsize"],
        n_pix_h=d["n_pix_h"],
        n_pix_v=d["n_pix_v"],
        kf_direction=d["kf_direction"],
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description="nrxrdct full-detector-image row-chunk worker (one SLURM job)"
    )
    p.add_argument("--meta-json", required=True)
    p.add_argument("--chunk-index", type=int, required=True)
    p.add_argument("--row-start", type=int, required=True)
    p.add_argument("--row-end", type=int, required=True)
    args = p.parse_args()

    with open(args.meta_json) as fh:
        meta = json.load(fh)

    with open(meta["stack_pkl"], "rb") as fh:
        stack = pickle.load(fh)
    camera = _camera_from_dict(meta["camera"])

    print(
        f"chunk {args.chunk_index}: rows [{args.row_start}:{args.row_end}) "
        f"of {meta['n_rows_total']} — starting",
        flush=True,
    )

    t0 = time.time()
    img = simulate_full_detector_image(
        stack, camera,
        bin_px=meta["bin_px"],
        n_energy=meta["n_energy"],
        E_min_eV=meta["E_min_eV"],
        E_max_eV=meta["E_max_eV"],
        source=meta["source"],
        source_kwargs=meta.get("source_kwargs"),
        kb_params=meta.get("kb_params"),
        ki_hat=meta.get("ki_hat"),
        structure_model=meta["structure_model"],
        darwin=meta["darwin"],
        exclude_layers=meta.get("exclude_layers"),
        sigma_h_mrad=meta["sigma_h_mrad"],
        sigma_v_mrad=meta["sigma_v_mrad"],
        n_div=meta["n_div"],
        row_start=args.row_start,
        row_end=args.row_end,
        verbose=True,
    )
    elapsed = time.time() - t0

    out_path = os.path.join(meta["chunks_dir"], f"chunk_{args.chunk_index:04d}.npz")
    tmp_path = out_path[:-4] + ".tmp.npz"
    np.savez(
        tmp_path,
        I=img["I"],
        x0=img["x0"],
        y0=img["y0"],
        row_start=args.row_start,
        row_end=args.row_end,
        elapsed_seconds=elapsed,
    )
    os.rename(tmp_path, out_path)

    print(
        f"chunk {args.chunk_index}: done in {elapsed:.1f}s -> {out_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()