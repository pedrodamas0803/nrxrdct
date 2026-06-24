"""
nrxrdct.laue.workers.slurm_layered_orient_worker
-------------------------------------------------
SLURM worker for orientation fitting of a LayeredCrystal stack.

Invoked by :meth:`LayeredMap.submit_orientation` via::

    python -m nrxrdct.laue.workers.slurm_layered_orient_worker \\
        --meta-json  path/to/orient_meta.json \\
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
    obs_xy: "np.ndarray | None",
    *,
    out_dir: str,
    max_match_px,
    min_matched: int,
    min_match_rate: float,
    max_rms_px: "float | None",
    fit_kwargs: dict,
    overwrite: bool,
) -> tuple:
    from nrxrdct.laue.fitting import fit_orientation_stack

    out_path = os.path.join(out_dir, f"frame_{frame_idx:05d}.npz")
    if os.path.exists(out_path) and not overwrite:
        return frame_idx, True
    if obs_xy is None or len(obs_xy) < min_matched:
        return frame_idx, False

    try:
        result = fit_orientation_stack(
            _g_stack, _g_camera, obs_xy,
            max_match_px=list(max_match_px),
            allowed_hkl=_g_allowed,
            update_stack=False,
            **fit_kwargs,
        )
    except Exception as exc:
        print(f"  ✗  frame {frame_idx}: {exc}", flush=True)
        return frame_idx, False

    if result.n_matched < min_matched or result.match_rate < min_match_rate:
        return frame_idx, False
    if max_rms_px is not None and result.rms_px > max_rms_px:
        return frame_idx, False

    tmp = out_path + ".tmp"
    np.savez(
        tmp,
        result_type = np.array("orientation"),
        R_global    = result.R_global,
        rotvec      = result.rotvec,
        U_layers    = np.stack(result.U_layers),
        rms_px      = np.array(result.rms_px),
        mean_px     = np.array(result.mean_px),
        n_matched   = np.array(result.n_matched),
        match_rate  = np.array(result.match_rate),
        cost        = np.array(result.cost),
        n_sim       = np.array(result.n_sim),
    )
    os.replace(tmp, out_path)
    return frame_idx, True


def main() -> None:
    p = argparse.ArgumentParser(description="LayeredMap orientation worker")
    p.add_argument("--meta-json",     required=True)
    p.add_argument("--frame-indices", required=True)
    args = p.parse_args()

    with open(args.meta_json) as fh:
        meta = json.load(fh)
    frame_indices = [int(x) for x in args.frame_indices.split(",")]

    # 1. Precompute allowed HKL once per job.
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
                E_max_eV=meta.get("E_max_eV", E_MAX_eV),
                f2_thresh=meta.get("f2_thresh", 1e-4),
            )
            for l in _enum_pool
        }
        print(f"  allowed_hkl precomputed ({time.time() - t0:.1f}s)", flush=True)
    del _stack_tmp

    # 2. Load peaklists.
    from nrxrdct.laue.segmentation import convert_spotsfile2peaklist

    min_matched = meta.get("min_matched", 5)
    peaklists: dict[int, np.ndarray] = {}
    for fi in frame_indices:
        seg_path = os.path.join(meta["seg_dir"], f"frame_{fi:05d}.h5")
        if not os.path.exists(seg_path):
            continue
        try:
            pl = convert_spotsfile2peaklist(
                seg_path,
                r_squared_min    = meta.get("r_squared_min", 0.9),
                include_unfitted = meta.get("include_unfitted", False),
            )
            if len(pl) >= min_matched:
                peaklists[fi] = pl[:, :2]
        except Exception as exc:
            print(f"  ✗  frame {fi}: peaklist: {exc}", flush=True)

    print(
        f"Layered orient worker — {len(frame_indices)} frames | "
        f"{len(peaklists)} with ≥{min_matched} spots",
        flush=True,
    )

    # 3. Fit in parallel.
    _FIT_KEYS = (
        "E_min_eV", "E_max_eV", "f2_thresh",
        "top_n_sim", "top_n_obs",
        "method", "ftol", "xtol", "gtol", "max_nfev",
        "geometry_only", "source", "source_kwargs", "structure_model",
    )
    fit_kwargs = {k: meta[k] for k in _FIT_KEYS if k in meta}

    common = dict(
        out_dir        = meta["out_dir"],
        max_match_px   = meta.get("max_match_px", [30, 10, 3]),
        min_matched    = min_matched,
        min_match_rate = meta.get("min_match_rate", 0.2),
        max_rms_px     = meta.get("max_rms_px"),
        fit_kwargs     = fit_kwargs,
        overwrite      = meta.get("overwrite", False),
    )
    n_workers = min(len(peaklists) or 1, os.cpu_count() or 1)
    n_ok = 0
    t0   = time.time()

    with ProcessPoolExecutor(
        max_workers = n_workers,
        initializer = _pool_init,
        initargs    = (meta["stack_pkl"], meta["camera"], allowed_hkl),
    ) as pool:
        futs = {
            pool.submit(_process_frame, fi, peaklists.get(fi), **common): fi
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
        f"Layered orient worker done — {n_ok}/{len(frame_indices)} saved  "
        f"({time.time() - t0:.1f}s)",
        flush=True,
    )
    sys.exit(0)


if __name__ == "__main__":
    main()
