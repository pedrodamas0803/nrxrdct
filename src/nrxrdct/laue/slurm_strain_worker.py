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

import numpy as np

from nrxrdct.laue.camera import Camera
from nrxrdct.laue.fitting import fit_strain_orientation
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
    strain_dir: str,
    camera: Camera,
    crystal,
    n_grains: int,
    max_match_px,
    fit_strain: tuple,
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

    if len(obs_xy) < 3:
        return 0

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

    with open(meta["crystal_pkl"], "rb") as fh:
        crystal = pickle.load(fh)

    camera   = _camera_from_dict(meta["camera"])
    n_grains = int(meta["n_grains"])

    _FIT_KEYS = (
        "hmax", "f2_thresh", "top_n_sim", "top_n_obs",
        "method", "ftol", "xtol", "gtol", "max_nfev",
        "geometry_only", "strain_scale", "source", "source_kwargs",
    )
    fit_kwargs = {k: meta[k] for k in _FIT_KEYS if k in meta}

    print(
        f"Strain worker — {len(frame_indices)} frames | "
        f"{n_grains} grain(s) | "
        f"fit_strain={meta.get('fit_strain')}",
        flush=True,
    )

    t0 = time.time()
    n_total = 0
    for fi in frame_indices:
        n_total += _process_frame(
            fi,
            seg_dir    = meta["seg_dir"],
            ubs_dir    = meta["ubs_dir"],
            strain_dir = meta["strain_dir"],
            camera     = camera,
            crystal    = crystal,
            n_grains   = n_grains,
            max_match_px     = meta.get("max_match_px", [10, 3]),
            fit_strain       = tuple(meta.get("fit_strain",
                                     ["e_xx","e_yy","e_zz","e_xy","e_xz","e_yz"])),
            fit_kwargs       = fit_kwargs,
            r_squared_min    = meta.get("r_squared_min", 0.9),
            include_unfitted = meta.get("include_unfitted", False),
        )

    elapsed = time.time() - t0
    print(
        f"Strain worker done — {n_total} strain fits saved, {elapsed:.1f}s",
        flush=True,
    )
    sys.exit(0)


if __name__ == "__main__":
    main()
