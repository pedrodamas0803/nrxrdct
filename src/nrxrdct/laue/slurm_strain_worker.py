"""
nrxrdct.laue.slurm_strain_worker
----------------------------------
SLURM worker for the strain-fitting step of a micro-Laue map.

For each assigned frame the worker:

1. Loads the observed spot list from ``seg_dir/frame_{idx:05d}.h5``.
2. For each grain index ``gi``, checks whether an orientation result exists at
   ``ubs_dir/frame_{idx:05d}_g{gi:02d}.npz``.
3. Runs :func:`~nrxrdct.laue.fitting.fit_strain_orientation` starting from the
   fitted U matrix.
4. Saves the full strain result to
   ``strain_dir/frame_{idx:05d}_g{gi:02d}.npz``.

Invoked by :meth:`GrainMap.submit_strain` via::

    python -m nrxrdct.laue.slurm_strain_worker \\
        --meta-json path/to/strain_meta.json  \\
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
from nrxrdct.laue.fitting import fit_strain_orientation
from nrxrdct.laue.segmentation import convert_spotsfile2peaklist
from nrxrdct.laue.simulation import precompute_allowed_hkl


# ── per-process globals set by the pool initializer ──────────────────────────
_g_crystal = None
_g_camera  = None
_g_allowed = None


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
    strain_dir: str,
    n_grains: int,
    max_match_px,
    fit_strain: tuple,
    fit_kwargs: dict,
) -> int:
    """Process one frame. Returns number of grains successfully fitted."""
    if obs_xy is None or len(obs_xy) < 3:
        return 0

    crystal = _g_crystal
    camera  = _g_camera
    n_saved = 0

    for gi in range(n_grains):
        orient_path = os.path.join(ubs_dir, f"frame_{frame_idx:05d}_g{gi:02d}.npz")
        if not os.path.exists(orient_path):
            continue

        out_path = os.path.join(strain_dir, f"frame_{frame_idx:05d}_g{gi:02d}.npz")
        if os.path.exists(out_path):
            n_saved += 1
            continue

        try:
            U0 = np.load(orient_path)["U"]
            result = fit_strain_orientation(
                crystal, camera, obs_xy, U0,
                max_match_px = list(max_match_px),
                fit_strain   = fit_strain,
                allowed_hkl  = _g_allowed,
                **fit_kwargs,
            )

            tmp = out_path[:-4] + ".tmp.npz"
            np.savez(
                tmp,
                U             = result.U,
                U_eff         = result.U_eff,
                strain_tensor = result.strain_tensor,
                strain_voigt  = result.strain_voigt,
                rotvec        = result.rotvec,
                rms_px        = np.array(result.rms_px),
                mean_px       = np.array(result.mean_px),
                n_matched     = np.array(result.n_matched),
                match_rate    = np.array(result.match_rate),
                cost          = np.array(result.cost),
            )
            os.rename(tmp, out_path)
            n_saved += 1

        except Exception as exc:
            print(f"  ✗  frame {frame_idx} g{gi}: strain fit: {exc}", flush=True)

    return n_saved


def main() -> None:
    p = argparse.ArgumentParser(
        description="nrxrdct Laue strain worker (one SLURM job)"
    )
    p.add_argument("--meta-json",     required=True)
    p.add_argument("--frame-indices", required=True)
    args = p.parse_args()

    with open(args.meta_json) as fh:
        meta = json.load(fh)

    frame_indices = [int(x) for x in args.frame_indices.split(",")]
    n_grains      = int(meta["n_grains"])

    _FIT_KEYS = (
        "hmax", "f2_thresh", "top_n_sim", "top_n_obs",
        "method", "ftol", "xtol", "gtol", "max_nfev",
        "geometry_only", "strain_scale", "source", "source_kwargs",
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
            if len(pl) >= 3:
                peaklists[fi] = pl[:, :2]
        except Exception as exc:
            print(f"  ✗  frame {fi}: load peaklist: {exc}", flush=True)

    print(
        f"Strain worker — {len(frame_indices)} frames | {n_grains} grain(s) | "
        f"{len(peaklists)} with spots | "
        f"fit_strain={meta.get('fit_strain')} | "
        f"I/O: {time.time() - t_io:.1f}s",
        flush=True,
    )

    # 3. Process frames in parallel; crystal/camera/allowed_hkl loaded once per
    #    worker process via the initializer (not re-sent with every task).
    common = dict(
        ubs_dir      = meta["ubs_dir"],
        strain_dir   = meta["strain_dir"],
        n_grains     = n_grains,
        max_match_px = meta.get("max_match_px", [10, 3]),
        fit_strain   = tuple(meta.get("fit_strain",
                             ["e_xx", "e_yy", "e_zz", "e_xy", "e_xz", "e_yz"])),
        fit_kwargs   = fit_kwargs,
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
        f"Strain worker done — {n_total} strain fits saved, {elapsed:.1f}s",
        flush=True,
    )
    sys.exit(0)


if __name__ == "__main__":
    main()
