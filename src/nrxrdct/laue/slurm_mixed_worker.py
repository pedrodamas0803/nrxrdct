"""
nrxrdct.laue.slurm_mixed_worker
---------------------------------
SLURM worker for mixed-phase orientation fitting of a micro-Laue map.

For each assigned frame the worker:

1. Loads the observed spot list from ``seg_dir/frame_{idx:05d}.h5``.
2. Runs :func:`~nrxrdct.laue.fitting.fit_orientation_mixed` with all phases
   simultaneously, using staged ``max_match_px`` refinement.
3. Saves one ``mixed_dir/frame_{idx:05d}.npz`` per frame containing the
   refined U matrix for each phase plus shared quality metrics.

Invoked by :meth:`GrainMap.submit_orientation_mixed` via::

    python -m nrxrdct.laue.slurm_mixed_worker \\
        --meta-json path/to/mixed_meta.json  \\
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
from nrxrdct.laue.fitting import fit_orientation_mixed
from nrxrdct.laue.segmentation import convert_spotsfile2peaklist
from nrxrdct.laue.simulation import precompute_allowed_hkl, E_MAX_eV


# ── per-process globals ───────────────────────────────────────────────────────
_g_crystals     = None
_g_camera       = None
_g_allowed_list = None   # list of allowed_hkl arrays, one per crystal


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


def _pool_init(crystals_pkl: str, camera_dict: dict, allowed_hkl_list) -> None:
    global _g_crystals, _g_camera, _g_allowed_list
    with open(crystals_pkl, "rb") as fh:
        _g_crystals = pickle.load(fh)
    _g_camera       = _camera_from_dict(camera_dict)
    _g_allowed_list = allowed_hkl_list


def _process_frame(
    frame_idx: int,
    obs_xy: "np.ndarray | None",
    *,
    mixed_dir: str,
    ub_arrays: list,
    shared: bool,
    max_match_px,
    min_matched: int,
    min_match_rate: float,
    max_rms_px: "float | None",
    fit_kwargs: dict,
    overwrite: bool = False,
) -> bool:
    """Process one frame. Returns True if a result was written."""
    if obs_xy is None or len(obs_xy) < min_matched:
        return False

    out_path = os.path.join(mixed_dir, f"frame_{frame_idx:05d}.npz")
    if os.path.exists(out_path) and not overwrite:
        return True

    crystals = _g_crystals
    camera   = _g_camera

    allowed = None
    if _g_allowed_list is not None:
        allowed = {id(c): a for c, a in zip(crystals, _g_allowed_list)}

    stages = max_match_px if isinstance(max_match_px, (list, tuple)) else [max_match_px]

    phases = [
        {"crystal": c, "U": np.asarray(U, dtype=float).copy(),
         "volume_fraction": 1.0 / len(crystals)}
        for c, U in zip(crystals, ub_arrays)
    ]

    try:
        result = None
        for px in stages:
            result = fit_orientation_mixed(
                phases, camera, obs_xy,
                shared=shared,
                max_match_px=float(px),
                geometry_only=False,
                allowed_hkl=allowed,
                update_phases=True,
                **fit_kwargs,
            )
    except Exception as exc:
        print(f"  ✗  frame {frame_idx}: mixed fit: {exc}", flush=True)
        return False

    if result.n_matched < min_matched:
        return False
    if result.match_rate < min_match_rate:
        return False
    if max_rms_px is not None and result.rms_px > max_rms_px:
        return False

    save_dict = {
        "rms_px":     np.array(result.rms_px),
        "mean_px":    np.array(result.mean_px),
        "n_matched":  np.array(result.n_matched),
        "match_rate": np.array(result.match_rate),
        "cost":       np.array(result.cost),
    }
    for i, (U, rv) in enumerate(zip(result.U_phases, result.rotvecs)):
        save_dict[f"U_{i}"]      = U
        save_dict[f"rotvec_{i}"] = rv

    tmp = out_path[:-4] + ".tmp.npz"
    np.savez(tmp, **save_dict)
    os.rename(tmp, out_path)
    return True


def main() -> None:
    p = argparse.ArgumentParser(
        description="nrxrdct Laue mixed-phase orientation worker (one SLURM job)"
    )
    p.add_argument("--meta-json",     required=True)
    p.add_argument("--frame-indices", required=True)
    args = p.parse_args()

    with open(args.meta_json) as fh:
        meta = json.load(fh)

    frame_indices = [int(x) for x in args.frame_indices.split(",")]

    _FIT_KEYS = (
        "E_max_eV", "f2_thresh", "top_n_sim", "top_n_obs",
        "method", "ftol", "xtol", "gtol", "max_nfev",
        "source", "source_kwargs",
    )
    fit_kwargs = {k: meta[k] for k in _FIT_KEYS if k in meta}

    # 1. Precompute allowed HKL once per crystal for the whole job.
    t_hkl = time.time()
    with open(meta["crystals_pkl"], "rb") as fh:
        _crystals_tmp = pickle.load(fh)

    allowed_hkl_list = None
    if meta.get("geometry_only", True):
        allowed_hkl_list = [
            precompute_allowed_hkl(
                c,
                E_max_eV=meta.get("E_max_eV", E_MAX_eV),
                f2_thresh=meta.get("f2_thresh", 1e-4),
            )
            for c in _crystals_tmp
        ]
        n_refs = sum(len(a) for a in allowed_hkl_list)
        print(
            f"  allowed_hkl: {n_refs} reflections across {len(_crystals_tmp)} phases "
            f"({time.time() - t_hkl:.1f}s)",
            flush=True,
        )
    del _crystals_tmp

    # 2. Load U reference matrices.
    ub_arrays = [np.load(f) for f in meta["ub_files"]]

    # 3. Batch-load peaklists.
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
                r_squared_min=r_squared_min,
                include_unfitted=include_unfitted,
            )
            if len(pl) >= min_matched:
                peaklists[fi] = pl[:, :2]
        except Exception as exc:
            print(f"  ✗  frame {fi}: load peaklist: {exc}", flush=True)

    print(
        f"Mixed worker — {len(frame_indices)} frames | {len(ub_arrays)} phase(s) | "
        f"{len(peaklists)} with enough spots | "
        f"shared={meta.get('shared', False)} | "
        f"I/O: {time.time() - t_io:.1f}s",
        flush=True,
    )

    # 4. Process frames in parallel.
    common = dict(
        mixed_dir      = meta["mixed_dir"],
        ub_arrays      = ub_arrays,
        shared         = meta.get("shared", False),
        max_match_px   = meta.get("max_match_px", [30, 10, 3]),
        min_matched    = min_matched,
        min_match_rate = meta.get("min_match_rate", 0.2),
        max_rms_px     = meta.get("max_rms_px", None),
        fit_kwargs     = fit_kwargs,
        overwrite      = meta.get("overwrite", False),
    )
    n_workers = min(len(peaklists) or 1, os.cpu_count() or 1)

    t0 = time.time()
    n_total = n_ok = 0
    with ProcessPoolExecutor(
        max_workers = n_workers,
        initializer = _pool_init,
        initargs    = (meta["crystals_pkl"], meta["camera"], allowed_hkl_list),
    ) as pool:
        futs = {
            pool.submit(_process_frame, fi, peaklists.get(fi), **common): fi
            for fi in frame_indices
        }
        for fut in as_completed(futs):
            n_total += 1
            try:
                ok = fut.result()
                n_ok += ok
            except Exception as exc:
                print(f"  ✗  frame {futs[fut]}: {exc}", flush=True)

    elapsed = time.time() - t0
    print(
        f"Mixed worker done — {n_ok}/{n_total} frames fitted, {elapsed:.1f}s",
        flush=True,
    )
    sys.exit(0)


if __name__ == "__main__":
    main()
