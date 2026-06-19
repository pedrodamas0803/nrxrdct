"""
nrxrdct.laue.workers.slurm_search_image_refine_worker
------------------------------------------------------
SLURM worker for the image-based orientation *search* step.

For each assigned frame and each grain listed in ``U_refs`` the worker:

1. Loads the raw detector image from the master HDF5 file.
2. Calls :func:`~nrxrdct.laue.fitting.search_orientation_image` to find the
   best orientation within a misorientation ball of radius *search_misor_deg*
   around the supplied reference U, then locally polishes with Powell.
3. Saves the result to ``search_img_refine_dir/frame_{idx:05d}_g{gi:02d}.npz``.

Unlike :mod:`slurm_image_refine_worker`, the starting orientation for each
grain is a **constant** supplied by the caller (``U_refs[gi]``), not read
from a per-pixel ``.npz`` file.  This makes it suitable for grains whose
orientation is only approximately known (e.g. an epitaxial film grain where
you have a rough estimate but want to find the true pixel-by-pixel orientation).

Invoked by :meth:`GrainMap.submit_search_image_refine` via::

    python -m nrxrdct.laue.workers.slurm_search_image_refine_worker \\
        --meta-json path/to/search_img_refine_meta.json              \\
        --frame-indices 0,1,2,...
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import dill as pickle
import h5py
import numpy as np

from nrxrdct.laue.camera import Camera
from nrxrdct.laue.fitting import search_orientation_image
from nrxrdct.laue.simulation import precompute_allowed_hkl, E_MAX_eV, E_MIN_eV


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
    *,
    h5_path: str,
    h5_dataset: str,
    U_refs: dict,
    search_img_refine_dir: str,
    kernel_sigma: float,
    bg_sigma: float,
    E_min: float,
    E_max: float,
    search_misor_deg: float,
    n_search: int,
    max_angle_deg: float,
    method: str,
    overwrite: bool,
) -> int:
    try:
        with h5py.File(h5_path, "r") as fh:
            image = fh[h5_dataset][frame_idx].astype(np.float64)
    except Exception as exc:
        print(f"  ✗  frame {frame_idx}: image load: {exc}", flush=True)
        return 0

    crystal = _g_crystal
    camera  = _g_camera
    n_saved = 0

    for gi_str, U_ref_list in U_refs.items():
        gi    = int(gi_str)
        U_ref = np.asarray(U_ref_list, dtype=float)

        out_path = os.path.join(
            search_img_refine_dir, f"frame_{frame_idx:05d}_g{gi:02d}.npz"
        )
        if os.path.exists(out_path) and not overwrite:
            n_saved += 1
            continue

        try:
            result = search_orientation_image(
                crystal, U_ref, camera, image,
                kernel_sigma     = kernel_sigma,
                bg_sigma         = bg_sigma,
                E_min            = E_min,
                E_max            = E_max,
                allowed_hkl      = _g_allowed,
                search_misor_deg = search_misor_deg,
                n_search         = n_search,
                max_angle_deg    = max_angle_deg,
                method           = method,
            )

            tmp = out_path[:-4] + ".tmp.npz"
            np.savez(
                tmp,
                U      = result.U,
                U0     = result.U0,
                rotvec = result.rotvec,
                score  = np.array(result.score),
                score0 = np.array(result.score0),
                n_sim  = np.array(result.n_sim),
            )
            os.rename(tmp, out_path)
            n_saved += 1

        except Exception as exc:
            print(f"  ✗  frame {frame_idx} g{gi}: search image refine: {exc}", flush=True)

    return n_saved


def main() -> None:
    p = argparse.ArgumentParser(
        description="nrxrdct Laue image-search worker (one SLURM job)"
    )
    p.add_argument("--meta-json",     required=True)
    p.add_argument("--frame-indices", required=True)
    args = p.parse_args()

    with open(args.meta_json) as fh:
        meta = json.load(fh)

    frame_indices = [int(x) for x in args.frame_indices.split(",")]
    E_min         = float(meta.get("E_min_eV", E_MIN_eV))
    E_max         = float(meta.get("E_max_eV", E_MAX_eV))
    U_refs        = meta["U_refs"]   # {str(gi): [[row0],[row1],[row2]]}

    t_hkl = time.time()
    with open(meta["crystal_pkl"], "rb") as fh:
        _crystal_tmp = pickle.load(fh)
    allowed_hkl = precompute_allowed_hkl(_crystal_tmp, E_max_eV=E_max)
    print(
        f"  allowed_hkl: {len(allowed_hkl)} reflections "
        f"({time.time() - t_hkl:.1f}s)",
        flush=True,
    )
    del _crystal_tmp

    print(
        f"Search image-refine worker — {len(frame_indices)} frames | "
        f"{len(U_refs)} grain(s) | "
        f"search_misor={meta['search_misor_deg']}° | "
        f"n_search={meta['n_search']} | "
        f"kernel_sigma={meta['kernel_sigma']} px | "
        f"max_angle={meta['max_angle_deg']}°",
        flush=True,
    )

    common = dict(
        h5_path               = meta["h5_path"],
        h5_dataset            = meta["h5_dataset"],
        U_refs                = U_refs,
        search_img_refine_dir = meta["search_img_refine_dir"],
        kernel_sigma          = float(meta["kernel_sigma"]),
        bg_sigma              = float(meta["bg_sigma"]),
        E_min                 = E_min,
        E_max                 = E_max,
        search_misor_deg      = float(meta["search_misor_deg"]),
        n_search              = int(meta["n_search"]),
        max_angle_deg         = float(meta["max_angle_deg"]),
        method                = meta.get("method", "Powell"),
        overwrite             = meta.get("overwrite", False),
    )

    n_workers = min(len(frame_indices) or 1, os.cpu_count() or 1)
    t0 = time.time()
    n_total = 0

    with ProcessPoolExecutor(
        max_workers = n_workers,
        initializer = _pool_init,
        initargs    = (meta["crystal_pkl"], meta["camera"], allowed_hkl),
    ) as pool:
        futs = {
            pool.submit(_process_frame, fi, **common): fi
            for fi in frame_indices
        }
        done = 0
        ntot = len(futs)
        tick = max(1, ntot // 20)
        for fut in as_completed(futs):
            done += 1
            try:
                n_total += fut.result()
            except Exception as exc:
                print(f"  ✗  frame {futs[fut]}: {exc}", flush=True)
            if done % tick == 0 or done == ntot:
                elapsed = time.time() - t0
                rate    = done / elapsed if elapsed > 0 else float("inf")
                eta     = (ntot - done) / rate if rate > 0 else float("inf")
                print(
                    f"  {done}/{ntot}  {n_total} refined  "
                    f"{elapsed:.0f}s elapsed  ETA {eta:.0f}s",
                    flush=True,
                )

    print(
        f"Search image-refine worker done — {n_total} results saved, "
        f"{time.time() - t0:.1f}s",
        flush=True,
    )
    sys.exit(0)


if __name__ == "__main__":
    main()
