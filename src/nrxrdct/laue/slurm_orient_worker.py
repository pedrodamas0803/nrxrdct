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

import numpy as np

from nrxrdct.laue.camera import Camera
from nrxrdct.laue.fitting import fit_orientation
from nrxrdct.laue.segmentation import convert_spotsfile2peaklist


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


def _process_frame(
    frame_idx: int,
    *,
    seg_dir: str,
    ubs_dir: str,
    camera: Camera,
    crystal,
    ub_arrays: list,
    max_match_px,
    min_matched: int,
    min_match_rate: float,
    max_rms_px: float | None,
    fit_kwargs: dict,
    r_squared_min: float,
    include_unfitted: bool,
) -> int:
    """Process one frame. Returns number of grains successfully fitted."""
    seg_path = os.path.join(seg_dir, f"frame_{frame_idx:05d}.h5")
    if not os.path.exists(seg_path):
        return 0

    try:
        peaklist = convert_spotsfile2peaklist(
            seg_path,
            r_squared_min    = r_squared_min,
            include_unfitted = include_unfitted,
        )
        obs_xy = peaklist[:, :2]
    except Exception as exc:
        print(f"  ✗  frame {frame_idx}: load peaklist: {exc}", flush=True)
        return 0

    if len(obs_xy) < min_matched:
        return 0

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
                **fit_kwargs,
            )
        except Exception as exc:
            print(f"  ✗  frame {frame_idx} g{gi}: fit: {exc}", flush=True)
            continue

        # Quality gate
        if result.n_matched < min_matched:
            continue
        if result.match_rate < min_match_rate:
            continue
        if max_rms_px is not None and result.rms_px > max_rms_px:
            continue

        tmp = out_path[:-4] + ".tmp.npz"
        np.savez(
            tmp,
            U           = result.U,
            rotvec      = result.rotvec,
            rms_px      = np.array(result.rms_px),
            mean_px     = np.array(result.mean_px),
            n_matched   = np.array(result.n_matched),
            match_rate  = np.array(result.match_rate),
            cost        = np.array(result.cost),
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

    with open(meta["crystal_pkl"], "rb") as fh:
        crystal = pickle.load(fh)

    camera    = _camera_from_dict(meta["camera"])
    ub_arrays = [np.load(f) for f in meta["ub_files"]]

    # Forward only known fit_orientation kwargs present in the meta
    _FIT_KEYS = (
        "hmax", "f2_thresh", "top_n_sim", "top_n_obs",
        "method", "ftol", "xtol", "gtol", "max_nfev",
        "geometry_only", "source", "source_kwargs",
    )
    fit_kwargs = {k: meta[k] for k in _FIT_KEYS if k in meta}

    print(
        f"Orient worker — {len(frame_indices)} frames | "
        f"{len(ub_arrays)} UB(s) | "
        f"max_match_px={meta.get('max_match_px')}",
        flush=True,
    )

    t0 = time.time()
    n_total = 0
    for fi in frame_indices:
        n_total += _process_frame(
            fi,
            seg_dir       = meta["seg_dir"],
            ubs_dir       = meta["ubs_dir"],
            camera        = camera,
            crystal       = crystal,
            ub_arrays     = ub_arrays,
            max_match_px     = meta.get("max_match_px", [30, 10, 3]),
            min_matched      = meta.get("min_matched", 5),
            min_match_rate   = meta.get("min_match_rate", 0.2),
            max_rms_px       = meta.get("max_rms_px", None),
            fit_kwargs       = fit_kwargs,
            r_squared_min    = meta.get("r_squared_min", 0.9),
            include_unfitted = meta.get("include_unfitted", False),
        )

    elapsed = time.time() - t0
    print(
        f"Orient worker done — {n_total} grain fits saved, {elapsed:.1f}s",
        flush=True,
    )
    sys.exit(0)


if __name__ == "__main__":
    main()
