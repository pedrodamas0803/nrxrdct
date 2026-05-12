"""
nrxrdct.laue.slurm_seg_worker
------------------------------
SLURM worker for the segmentation step of a micro-Laue map.

Each job processes an assigned subset of frames and writes one HDF5 spots
file per frame::

    seg_dir/frame_{idx:05d}.h5

Invoked by :meth:`GrainMap.submit_segmentation` via::

    python -m nrxrdct.laue.slurm_seg_worker \\
        --meta-json path/to/seg_meta.json  \\
        --frame-indices 0,1,2,...

The meta JSON must contain the fields documented in
:meth:`GrainMap.submit_segmentation`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import h5py
import numpy as np

from nrxrdct.laue.segmentation import (
    LoG_segmentation,
    WTH_segmentation,
    clean_segmentation,
    filter_and_rescale_images,
    gaussian_background,
    label_segmented_image,
    measure_peaks,
    write_h5_spotsfile,
)


def _process_frame(
    frame_idx: int,
    *,
    h5_path: str,
    h5_dataset: str,
    seg_dir: str,
    detector_mask: np.ndarray,
    method: str,
    method_kwargs: dict,
    min_size: int,
    max_size: int,
    gap_exclude: int,
    bg_sigma: float,
) -> bool:
    out_path = os.path.join(seg_dir, f"frame_{frame_idx:05d}.h5")
    if os.path.exists(out_path):
        return True  # resume: already done

    try:
        with h5py.File(h5_path, "r") as f:
            frame = f[h5_dataset][frame_idx].astype(np.float32)

        valid = detector_mask if detector_mask is not None else np.ones(frame.shape, dtype=bool)
        bg = gaussian_background(frame, valid, sigma=bg_sigma)
        frame = frame - bg
        frame -= frame[valid].min()
        frame[~valid] = 0.0

        if method.upper() == "WTH":
            seg_mask = WTH_segmentation(frame, detector_mask, **method_kwargs)
        else:
            seg_mask = LoG_segmentation(frame, detector_mask, **method_kwargs)

        final_mask, _ = clean_segmentation(
            seg_mask, detector_mask, frame,
            min_size=min_size, max_size=max_size, gap_exclude=gap_exclude,
        )

        filt_im = filter_and_rescale_images(frame, cutoff_freq=0.001)
        label_img, _, _ = label_segmented_image(final_mask, filt_im)
        regionprops = measure_peaks(label_img, filt_im)

        tmp = out_path + ".tmp"
        write_h5_spotsfile(filt_im, regionprops, outpath=tmp, overwrite=True)
        os.rename(tmp, out_path)
        return True

    except Exception as exc:
        print(f"  ✗  frame {frame_idx}: {exc}", flush=True)
        return False


def main() -> None:
    p = argparse.ArgumentParser(
        description="nrxrdct Laue segmentation worker (one SLURM job)"
    )
    p.add_argument("--meta-json",      required=True,
                   help="Path to the seg_meta.json sidecar written by submit_segmentation")
    p.add_argument("--frame-indices",  required=True,
                   help="Comma-separated frame indices assigned to this job")
    args = p.parse_args()

    with open(args.meta_json) as fh:
        meta = json.load(fh)

    frame_indices = [int(x) for x in args.frame_indices.split(",")]

    mask_path = meta.get("mask_path")
    detector_mask = np.load(mask_path).astype(bool) if mask_path else None

    print(
        f"Seg worker — {len(frame_indices)} frames | "
        f"method={meta.get('method', 'LoG')} | "
        f"dataset={meta['h5_dataset']}",
        flush=True,
    )

    t0 = time.time()
    n_ok = n_fail = 0
    for fi in frame_indices:
        ok = _process_frame(
            fi,
            h5_path       = meta["h5_path"],
            h5_dataset    = meta["h5_dataset"],
            seg_dir       = meta["seg_dir"],
            detector_mask = detector_mask,
            method        = meta.get("method", "LoG"),
            method_kwargs = meta.get("method_kwargs", {}),
            min_size      = meta.get("min_size", 3),
            max_size      = meta.get("max_size", 500),
            gap_exclude   = meta.get("gap_exclude", 3),
            bg_sigma      = meta.get("bg_sigma", 251),
        )
        n_ok   += ok
        n_fail += not ok

    elapsed = time.time() - t0
    print(f"Seg worker done — {n_ok} OK, {n_fail} failed, {elapsed:.1f}s", flush=True)
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
