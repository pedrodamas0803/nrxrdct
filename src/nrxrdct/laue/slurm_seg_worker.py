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
import re
import sys
import time

import h5py
import numpy as np
import skimage as sk

from nrxrdct.laue.segmentation import (
    LoG_segmentation,
    WTH_segmentation,
    hybrid_segmentation,
    clean_segmentation,
    filter_and_rescale_images,
    gaussian_background,
    label_segmented_image,
    measure_peaks,
    write_h5_spotsfile,
)


def _process_frame(
    frame_idx: int,
    frame: np.ndarray,
    *,
    seg_dir: str,
    detector_mask: np.ndarray,
    method: str,
    method_kwargs: dict,
    min_size: int,
    max_size: int,
    gap_exclude: int,
    bg_sigma: float,
    max_components: int,
    d: int,
) -> bool:
    out_path = os.path.join(seg_dir, f"frame_{frame_idx:05d}.h5")
    if os.path.exists(out_path):
        return True  # resume: already done

    try:
        # Always work with a concrete boolean mask.  When none is supplied,
        # derive it from the frame: invalid pixels are stored as zero.
        valid = detector_mask if detector_mask is not None else (frame > 0)

        # Background subtraction for spot detection only — keep original intensities.
        bg        = gaussian_background(frame, valid, sigma=bg_sigma)
        frame_sub = frame - bg
        frame_sub -= frame_sub[valid].min()
        frame_sub[~valid] = 0.0

        if method.upper() == "WTH":
            seg_mask = WTH_segmentation(frame_sub, valid, **method_kwargs)
        elif method.upper() == "HYBRID":
            seg_mask = hybrid_segmentation(frame_sub, valid, **method_kwargs)
        else:
            seg_mask = LoG_segmentation(frame_sub, valid, **method_kwargs)

        final_mask, _ = clean_segmentation(
            seg_mask, valid, frame_sub,
            min_size=min_size, max_size=max_size, gap_exclude=gap_exclude,
        )

        # Gaussian fits and intensity measurements use the original (unmodified) frame.
        filt_im = filter_and_rescale_images(frame, cutoff_freq=0.001)
        label_img, _, _ = label_segmented_image(final_mask, filt_im)
        regionprops = measure_peaks(label_img, filt_im)

        tmp = out_path + ".tmp"
        write_h5_spotsfile(filt_im, regionprops, outpath=tmp, overwrite=True,
                           d=d, max_components=max_components)
        os.rename(tmp, out_path)
        return True

    except Exception as exc:
        print(f"  ✗  frame {frame_idx}: {exc}", flush=True)
        return False


def _load_tiff_frames(tiff_dir: str, frame_indices: list) -> dict:
    """
    Load TIFF frames from *tiff_dir*.

    Files must match ``img_<number>.tif`` (case-insensitive).  They are
    sorted by their embedded number and mapped to 0-based frame indices in
    that order (i.e. the file with the smallest number → frame 0).

    Returns
    -------
    dict[int, np.ndarray]  mapping frame_idx → float32 image array.
    """
    pattern = re.compile(r'^img_(\d+)\.tif$', re.IGNORECASE)
    all_files = []
    for fname in os.listdir(tiff_dir):
        m = pattern.match(fname)
        if m:
            all_files.append((int(m.group(1)), os.path.join(tiff_dir, fname)))
    all_files.sort(key=lambda x: x[0])

    idx_to_path = {fi: path for fi, (_, path) in enumerate(all_files)}

    frames = {}
    for fi in frame_indices:
        if fi not in idx_to_path:
            continue
        try:
            frames[fi] = sk.io.imread(idx_to_path[fi]).astype(np.float32)
        except Exception as exc:
            print(f"  ✗  frame {fi}: TIFF read: {exc}", flush=True)
    return frames


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

    tiff_dir = meta.get("tiff_dir")
    source_label = f"tiff:{tiff_dir}" if tiff_dir else f"h5:{meta.get('h5_dataset')}"
    print(
        f"Seg worker — {len(frame_indices)} frames | "
        f"method={meta.get('method', 'LoG')} | "
        f"source={source_label}",
        flush=True,
    )

    # Load all assigned frames in a single I/O pass.
    t_io = time.time()
    if tiff_dir:
        frames = _load_tiff_frames(tiff_dir, frame_indices)
    else:
        with h5py.File(meta["h5_path"], "r") as f:
            ds = f[meta["h5_dataset"]]
            frames = {fi: ds[fi].astype(np.float32) for fi in frame_indices}
    print(f"  image read done ({time.time() - t_io:.1f}s)", flush=True)

    t0 = time.time()
    n_ok = n_fail = 0
    for fi in frame_indices:
        if fi not in frames:
            n_fail += 1
            continue
        ok = _process_frame(
            fi,
            frames[fi],
            seg_dir       = meta["seg_dir"],
            detector_mask = detector_mask,
            method        = meta.get("method", "LoG"),
            method_kwargs = meta.get("method_kwargs", {}),
            min_size       = meta.get("min_size", 3),
            max_size       = meta.get("max_size", 500),
            gap_exclude    = meta.get("gap_exclude", 3),
            bg_sigma       = meta.get("bg_sigma", 251),
            max_components = meta.get("max_components", 1),
            d              = meta.get("d", 10),
        )
        n_ok   += ok
        n_fail += not ok

    elapsed = time.time() - t0
    print(f"Seg worker done — {n_ok} OK, {n_fail} failed, {elapsed:.1f}s", flush=True)
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
