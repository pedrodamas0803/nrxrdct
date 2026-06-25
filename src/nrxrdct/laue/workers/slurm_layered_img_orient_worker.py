"""
nrxrdct.laue.workers.slurm_layered_img_orient_worker
------------------------------------------------------
SLURM worker for image-based orientation refinement of a LayeredCrystal stack.

Invoked by :meth:`LayeredMap.submit_image_orientation` via::

    python -m nrxrdct.laue.workers.slurm_layered_img_orient_worker \\
        --meta-json  path/to/img_orient_meta.json \\
        --frame-indices 0,1,2,...
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import dill as _pickle
import h5py
import numpy as np

from nrxrdct.laue.camera import Camera
from nrxrdct.laue.simulation import E_MAX_eV


# ── per-process globals ────────────────────────────────────────────────────────

_g_stack   = None
_g_camera  = None
_g_allowed = None


def _pool_init(stack_pkl: str, camera_dict: dict, allowed_hkl) -> None:
    global _g_stack, _g_camera, _g_allowed
    with open(stack_pkl, "rb") as fh:
        _g_stack = _pickle.load(fh)
    _g_camera  = Camera(**camera_dict)
    _g_allowed = allowed_hkl


def _process_frame(
    frame_idx: int,
    frame_data: "np.ndarray | None",
    *,
    out_dir: str,
    fit_kwargs: dict,
    overwrite: bool,
) -> tuple:
    from nrxrdct.laue.fitting import refine_orientation_image_stack

    out_path = os.path.join(out_dir, f"frame_{frame_idx:05d}.npz")
    if os.path.exists(out_path) and not overwrite:
        return frame_idx, True
    if frame_data is None:
        return frame_idx, False

    try:
        result = refine_orientation_image_stack(
            _g_stack, _g_camera, frame_data,
            allowed_hkl=_g_allowed,
            **fit_kwargs,
        )
    except Exception as exc:
        print(f"  ✗  frame {frame_idx}: {exc}", flush=True)
        return frame_idx, False

    tmp = out_path + ".tmp"
    np.savez(
        tmp,
        result_type = np.array("img_orientation"),
        R_global    = result.R_global,
        rotvec      = result.rotvec,
        U_layers    = np.stack(result.U_layers),
        score       = np.array(result.score),
        score0      = np.array(result.score0),
        n_sim       = np.array(result.n_sim),
    )
    os.replace(tmp, out_path)
    return frame_idx, True


def main() -> None:
    p = argparse.ArgumentParser(description="LayeredMap image-orientation worker")
    p.add_argument("--meta-json",     required=True)
    p.add_argument("--frame-indices", required=True)
    args = p.parse_args()

    with open(args.meta_json) as fh:
        meta = json.load(fh)
    frame_indices = [int(x) for x in args.frame_indices.split(",")]

    # 1. Precompute allowed HKL.
    from nrxrdct.laue.simulation import precompute_allowed_hkl

    with open(meta["stack_pkl"], "rb") as fh:
        _stack_tmp = _pickle.load(fh)

    allowed_hkl = None
    if meta.get("geometry_only", True):
        _enum_pool = (
            _stack_tmp.buffer_layers + _stack_tmp.layers[:1]
            if (_stack_tmp.buffer_layers or _stack_tmp.layers)
            else _stack_tmp.all_layers
        )
        t0 = time.time()
        allowed_hkl = {
            id(l.crystal): precompute_allowed_hkl(
                l.crystal,
                E_max_eV=meta.get("E_max", E_MAX_eV),
                f2_thresh=meta.get("f2_thresh", 1e-4),
            )
            for l in _enum_pool
        }
        print(f"  allowed_hkl precomputed ({time.time() - t0:.1f}s)", flush=True)
    del _stack_tmp

    # 2. Load frames from HDF5.
    frames: dict[int, np.ndarray] = {}
    with h5py.File(meta["h5_path"], "r") as hf:
        ds = hf[meta["h5_dataset"]]
        for fi in frame_indices:
            try:
                frames[fi] = ds[fi].astype(np.float32)
            except Exception as exc:
                print(f"  ✗  frame {fi}: image read: {exc}", flush=True)

    print(
        f"Layered img-orient worker — {len(frame_indices)} frames | "
        f"{len(frames)} images loaded",
        flush=True,
    )

    # 3. Refine in parallel.
    _FIT_KEYS = (
        "kernel_sigma", "bg_sigma", "E_min", "E_max",
        "max_angle_deg", "correct_depth", "method", "options", "structure_model",
    )
    fit_kwargs = {k: meta[k] for k in _FIT_KEYS if k in meta}

    common = dict(out_dir=meta["out_dir"], fit_kwargs=fit_kwargs,
                  overwrite=meta.get("overwrite", False))
    n_workers = min(len(frames) or 1, os.cpu_count() or 1)
    n_ok = 0
    t0   = time.time()

    with ProcessPoolExecutor(
        max_workers = n_workers,
        initializer = _pool_init,
        initargs    = (meta["stack_pkl"], meta["camera"], allowed_hkl),
    ) as pool:
        futs = {
            pool.submit(_process_frame, fi, frames.get(fi), **common): fi
            for fi in frame_indices
        }
        for fut in as_completed(futs):
            try:
                _, ok = fut.result()
                if ok:
                    n_ok += 1
            except Exception as exc:
                print(f"  ✗  frame {futs[fut]}: {exc}", flush=True)

    print(
        f"Layered img-orient worker done — {n_ok}/{len(frame_indices)} saved  "
        f"({time.time() - t0:.1f}s)",
        flush=True,
    )
    sys.exit(0)


if __name__ == "__main__":
    main()
