"""
nrxrdct.laue.slurm_orient_worker
----------------------------------
SLURM worker for the orientation-fitting step of a micro-Laue map.

For each assigned frame the worker:

1. Loads the observed spot list from ``seg_dir/frame_{idx:05d}.h5``.
2. Tries every ``UB*.npy`` reference matrix in turn.
3. Runs :func:`~nrxrdct.laue.fitting.fit_orientation` with optional staged
   ``max_match_px``.
4. Saves a ``.npz`` result to ``ubs_dir/frame_{idx:05d}_g{gi:02d}.npz``
   **only if** the fit passes the quality thresholds.

Invoked by :meth:`GrainMap.submit_orientation` via::

    python -m nrxrdct.laue.slurm_orient_worker \\
        --meta-json path/to/orient_meta.json  \\
        --frame-indices 0,1,2,...
"""

from __future__ import annotations

import argparse
import json
import os
import dill as pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from nrxrdct.laue.camera import Camera
from nrxrdct.laue.fitting import fit_orientation
from nrxrdct.laue.segmentation import convert_spotsfile2peaklist
from nrxrdct.laue.simulation import precompute_allowed_hkl


# ── per-process globals set by the pool initializer ──────────────────────────
_g_crystal    = None
_g_camera     = None
_g_allowed    = None


def _camera_from_dict(d: dict) -> Camera:
    return Camera(
        dd           = d["dd"],
        xcen         = d["xcen"],
        ycen         = d["ycen"],
        xbet         = d["xbet"],
        xgam         = d["xgam"],
        pixelsize    = d["pixelsize"],
        n_pix_h      = d["n_pix_h"],
        n_pix_v      = d["n_pix_v"],
        kf_direction = d["kf_direction"],
    )


def _pool_init(crystal_pkl: str, camera_dict: dict, allowed_hkl) -> None:
    global _g_crystal, _g_camera, _g_allowed
    with open(crystal_pkl, "rb") as fh:
        _g_crystal = pickle.load(fh)
    _g_camera  = _camera_from_dict(camera_dict)
    _g_allowed = allowed_hkl


def _process_frame(
    frame_idx: int,
    obs_xy: "np.ndarray | None",
    *,
    ubs_dir: str,
    ub_arrays: list,
    max_match_px,
    min_matched: int,
    min_match_rate: float,
    max_rms_px: "float | None",
    fit_kwargs: dict,
) -> int:
    """Process one frame. Returns number of grains successfully fitted."""
    if obs_xy is None or len(obs_xy) < min_matched:
        return 0

    crystal = _g_crystal
    camera  = _g_camera
    n_saved = 0

    for gi, U_ref in enumerate(ub_arrays):
        out_path = os.path.join(ubs_dir, f"frame_{frame_idx:05d}_g{gi:02d}.npz")
        if os.path.exists(out_path):
            n_saved += 1
            continue

        try:
            result = fit_orientation(
                crystal, camera, obs_xy, U_ref,
                max_match_px=list(max_match_px),
                allowed_hkl=_g_allowed,
                **fit_kwargs,
            )
        except Exception as exc:
            print(f"  ✗  frame {frame_idx} g{gi}: fit: {exc}", flush=True)
            continue

        if result.n_matched < min_matched:
            continue
        if result.match_rate < min_match_rate:
            continue
        if max_rms_px is not None and result.rms_px > max_rms_px:
            continue

        tmp = out_path[:-4] + ".tmp.npz"
        np.savez(
            tmp,
            U          = result.U,
            rotvec     = result.rotvec,
            rms_px     = np.array(result.rms_px),
            mean_px    = np.array(result.mean_px),
            n_matched  = np.array(result.n_matched),
            match_rate = np.array(result.match_rate),
            cost       = np.array(result.cost),
        )
        os.rename(tmp, out_path)
        n_saved += 1

    return n_saved


def main() -> None:
    p = argparse.ArgumentParser(
        description="nrxrdct Laue orientation worker (one SLURM job)"
    )
    p.add_argument("--meta-json",     required=True)
    p.add_argument("--frame-indices", required=True)
    args = p.parse_args()

    with open(args.meta_json) as fh:
        meta = json.load(fh)

    frame_indices = [int(x) for x in args.frame_indices.split(",")]
    ub_arrays     = [np.load(f) for f in meta["ub_files"]]

    _FIT_KEYS = (
        "hmax", "f2_thresh", "top_n_sim", "top_n_obs",
        "method", "ftol", "xtol", "gtol", "max_nfev",
        "geometry_only", "source", "source_kwargs",
    )
    fit_kwargs = {k: meta[k] for k in _FIT_KEYS if k in meta}

    # 1. Precompute allowed HKL once for the whole job.
    t_hkl = time.time()
    with open(meta["crystal_pkl"], "rb") as fh:
        _crystal_tmp = pickle.load(fh)
    allowed_hkl = None
    if fit_kwargs.get("geometry_only", True):
        allowed_hkl = precompute_allowed_hkl(
            _crystal_tmp,
            fit_kwargs.get("hmax", 15),
            f2_thresh=fit_kwargs.get("f2_thresh", 1e-4),
        )
        print(
            f"  allowed_hkl: {len(allowed_hkl)} reflections "
            f"({time.time() - t_hkl:.1f}s)",
            flush=True,
        )
    del _crystal_tmp

    # 2. Batch-load all peaklists with a single pass over the seg directory.
    t_io = time.time()
    r_squared_min    = meta.get("r_squared_min", 0.9)
    include_unfitted = meta.get("include_unfitted", False)
    min_matched      = meta.get("min_matched", 5)
    peaklists: dict[int, np.ndarray] = {}
    for fi in frame_indices:
        seg_path = os.path.join(meta["seg_dir"], f"frame_{fi:05d}.h5")
        if not os.path.exists(seg_path):
            continue
        try:
            pl = convert_spotsfile2peaklist(
                seg_path,
                r_squared_min    = r_squared_min,
                include_unfitted = include_unfitted,
            )
            if len(pl) >= min_matched:
                peaklists[fi] = pl[:, :2]
        except Exception as exc:
            print(f"  ✗  frame {fi}: load peaklist: {exc}", flush=True)

    print(
        f"Orient worker — {len(frame_indices)} frames | {len(ub_arrays)} UB(s) | "
        f"{len(peaklists)} with enough spots | "
        f"max_match_px={meta.get('max_match_px')} | "
        f"I/O: {time.time() - t_io:.1f}s",
        flush=True,
    )

    # 3. Process frames in parallel; crystal/camera/allowed_hkl loaded once per
    #    worker process via the initializer (not re-sent with every task).
    common = dict(
        ubs_dir        = meta["ubs_dir"],
        ub_arrays      = ub_arrays,
        max_match_px   = meta.get("max_match_px", [30, 10, 3]),
        min_matched    = min_matched,
        min_match_rate = meta.get("min_match_rate", 0.2),
        max_rms_px     = meta.get("max_rms_px", None),
        fit_kwargs     = fit_kwargs,
    )
    n_workers = min(len(peaklists) or 1, os.cpu_count() or 1)

    t0 = time.time()
    n_total = 0
    with ProcessPoolExecutor(
        max_workers = n_workers,
        initializer = _pool_init,
        initargs    = (meta["crystal_pkl"], meta["camera"], allowed_hkl),
    ) as pool:
        futs = {
            pool.submit(_process_frame, fi, peaklists.get(fi), **common): fi
            for fi in frame_indices
        }
        for fut in as_completed(futs):
            try:
                n_total += fut.result()
            except Exception as exc:
                print(f"  ✗  frame {futs[fut]}: {exc}", flush=True)

    elapsed = time.time() - t0
    print(
        f"Orient worker done — {n_total} grain fits saved, {elapsed:.1f}s",
        flush=True,
    )
    sys.exit(0)


if __name__ == "__main__":
    main()
