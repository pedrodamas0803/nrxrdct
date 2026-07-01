"""
nrxrdct.laue.layered_map — LayeredMap: per-pixel fitting for LayeredCrystal stacks.
=====================================================================================

Typical workflow::

    lmap = LayeredMap(ny=21, nx=21, stack=stack, h5_path="scan.h5")

    # Fit orientation (parallel, local cores)
    lmap.run_orientation_local(camera, seg_dir="seg/", out_dir="ubs/")
    lmap.collect("ubs/")

    # Polish with strain
    lmap.run_strain_local(camera, seg_dir="seg/", out_dir="strain/",
                          fit_strain=("e_xx", "e_yy", "e_zz"))
    lmap.collect("strain/")

    # Inspect and plot
    lmap.plot_overview(layer=0)
    lmap.plot_strain_component("e_zz", layer=0)
    lmap.inspect_frame(camera, frame_idx=42, h5_path="scan.h5")

    lmap.save("layered_map.h5")
    lmap2 = LayeredMap.load("layered_map.h5", stack=stack)
"""

from __future__ import annotations

import concurrent.futures
import glob
import json
import os
import tempfile
import time

import dill as _dill
import h5py
import matplotlib.pyplot as plt
import numpy as np
import subprocess
from scipy.spatial.transform import Rotation


# ─────────────────────────────────────────────────────────────────────────────
# Per-process globals for the worker pool
# ─────────────────────────────────────────────────────────────────────────────

_g_stack   = None
_g_camera  = None
_g_allowed = None


def _lm_pool_init(stack_pkl_path: str, camera, allowed_hkl) -> None:
    global _g_stack, _g_camera, _g_allowed
    with open(stack_pkl_path, "rb") as fh:
        _g_stack = _dill.load(fh)
    _g_camera  = camera
    _g_allowed = allowed_hkl


# ─────────────────────────────────────────────────────────────────────────────
# Worker functions (module-level so they are picklable)
# ─────────────────────────────────────────────────────────────────────────────

def _lm_orient_frame(
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
    """Orientation-only stack fit for one frame."""
    from .fitting import fit_orientation_stack

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

    if result.n_matched < min_matched:
        return frame_idx, False
    if result.match_rate < min_match_rate:
        return frame_idx, False
    if max_rms_px is not None and result.rms_px > max_rms_px:
        return frame_idx, False

    tmp = out_path[:-4] + ".tmp.npz"
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


def _lm_strain_frame(
    frame_idx: int,
    obs_xy: "np.ndarray | None",
    *,
    out_dir: str,
    fit_strain,
    max_match_px,
    min_matched: int,
    min_match_rate: float,
    max_rms_px: "float | None",
    fit_kwargs: dict,
    overwrite: bool,
    orient_data: "dict | None" = None,
) -> tuple:
    """Orientation + per-layer strain stack fit for one frame."""
    from .fitting import fit_strain_orientation_stack

    out_path = os.path.join(out_dir, f"frame_{frame_idx:05d}.npz")
    if os.path.exists(out_path) and not overwrite:
        return frame_idx, True
    if obs_xy is None or len(obs_xy) < min_matched:
        return frame_idx, False

    # Save template U matrices and optionally warm-start from prior orientation.
    _saved_U  = [l.U.copy() for l in _g_stack.all_layers]
    _orient_U = orient_data.get(frame_idx) if orient_data else None
    if _orient_U is not None:
        for l, U in zip(_g_stack.all_layers, _orient_U):
            l.U = U.copy()

    try:
        result = fit_strain_orientation_stack(
            _g_stack, _g_camera, obs_xy,
            fit_strain=fit_strain,
            max_match_px=list(max_match_px),
            allowed_hkl=_g_allowed,
            update_stack=False,
            **fit_kwargs,
        )
    except Exception as exc:
        print(f"  ✗  frame {frame_idx}: {exc}", flush=True)
        return frame_idx, False
    finally:
        # Always restore the template U so the next frame starts clean.
        for l, U in zip(_g_stack.all_layers, _saved_U):
            l.U = U

    if result.n_matched < min_matched:
        return frame_idx, False
    if result.match_rate < min_match_rate:
        return frame_idx, False
    if max_rms_px is not None and result.rms_px > max_rms_px:
        return frame_idx, False

    tmp = out_path[:-4] + ".tmp.npz"
    np.savez(
        tmp,
        result_type    = np.array("strain"),
        R_global       = result.R_global,
        rotvec         = result.rotvec,
        U_layers       = np.stack(result.U_layers),
        U_eff_layers   = np.stack(result.U_eff_layers),
        strain_tensors = np.stack(result.strain_tensors),
        strain_voigts  = np.stack(result.strain_voigts),
        rms_px         = np.array(result.rms_px),
        mean_px        = np.array(result.mean_px),
        n_matched      = np.array(result.n_matched),
        match_rate     = np.array(result.match_rate),
        cost           = np.array(result.cost),
        n_sim          = np.array(result.n_sim),
    )
    os.replace(tmp, out_path)
    return frame_idx, True


def _lm_img_orient_frame(
    frame_idx: int,
    frame_data: "np.ndarray | None",
    *,
    out_dir: str,
    fit_kwargs: dict,
    overwrite: bool,
) -> tuple:
    """Image-based orientation stack refinement for one frame."""
    from .fitting import refine_orientation_image_stack

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

    tmp = out_path[:-4] + ".tmp.npz"
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


def _lm_img_strain_frame(
    frame_idx: int,
    frame_data: "np.ndarray | None",
    *,
    out_dir: str,
    fit_strain,
    fit_kwargs: dict,
    overwrite: bool,
) -> tuple:
    """Image-based orientation + per-layer strain stack refinement for one frame."""
    from .fitting import refine_strain_image_stack

    out_path = os.path.join(out_dir, f"frame_{frame_idx:05d}.npz")
    if os.path.exists(out_path) and not overwrite:
        return frame_idx, True
    if frame_data is None:
        return frame_idx, False

    try:
        result = refine_strain_image_stack(
            _g_stack, _g_camera, frame_data,
            fit_strain=fit_strain,
            allowed_hkl=_g_allowed,
            **fit_kwargs,
        )
    except Exception as exc:
        print(f"  ✗  frame {frame_idx}: {exc}", flush=True)
        return frame_idx, False

    tmp = out_path[:-4] + ".tmp.npz"
    np.savez(
        tmp,
        result_type    = np.array("img_strain"),
        R_global       = result.R_global,
        rotvec         = result.rotvec,
        U_layers       = np.stack(result.U_layers),
        U_eff_layers   = np.stack(result.U_eff_layers),
        strain_tensors = np.stack(result.strain_tensors),
        strain_voigts  = np.stack(result.strain_voigts),
        score          = np.array(result.score),
        score0         = np.array(result.score0),
        n_sim          = np.array(result.n_sim),
    )
    os.replace(tmp, out_path)
    return frame_idx, True


# ─────────────────────────────────────────────────────────────────────────────
# Motor reader (mirrors map.py)
# ─────────────────────────────────────────────────────────────────────────────

def _read_motor_array(
    h5_file: h5py.File, entry: str, motor: str, n_frames: int
) -> "np.ndarray | None":
    for path in (
        f"{entry}/instrument/positioners/{motor}",
        f"{entry}/measurement/{motor}",
    ):
        if path in h5_file:
            arr = np.asarray(h5_file[path], dtype=float).ravel()
            if arr.size == n_frames:
                return arr
            if arr.size == 1:
                return np.full(n_frames, arr.item())
    return None


# ─────────────────────────────────────────────────────────────────────────────
# IPF colour fallback (no orix required)
# ─────────────────────────────────────────────────────────────────────────────

def _ipf_color_fallback(U_arr: np.ndarray, ref_dir: np.ndarray) -> np.ndarray:
    """
    Cubic IPF-like coloring without orix.

    Maps the crystal direction parallel to *ref_dir* into the [001]-[011]-[111]
    fundamental zone and returns RGB colours:
    [001] → blue, [011] → green, [111] → red.

    Args:
        U_arr:   (M, 3, 3) orientation matrices (crystal → sample).
        ref_dir: (3,) unit reference direction in the sample frame.

    Returns:
        (M, 3) float32 RGB array clipped to [0, 1].
    """
    # Crystal direction parallel to ref: d_crystal = U^T @ ref_sample
    d = np.einsum("mji,j->mi", U_arr, ref_dir)   # (M, 3)

    # Bring to cubic fundamental zone: abs + sort descending → a >= b >= c >= 0
    d = np.abs(d)
    d = np.sort(d, axis=1)[:, ::-1]

    norms = np.linalg.norm(d, axis=1, keepdims=True)
    d /= np.where(norms > 0, norms, 1.0)

    a, b, c = d[:, 0], d[:, 1], d[:, 2]

    # Angular parameterisation of the [001]-[011]-[111] triangle
    theta     = np.arctan2(b, a)                          # [0, pi/4]
    phi       = np.arctan2(c * np.sqrt(2.0), a + b)      # [0, arctan(1/sqrt(2))]
    phi_max   = np.arctan(1.0 / np.sqrt(2.0))

    t = np.clip(theta / (np.pi / 4), 0.0, 1.0)   # 0=[001], 1=[011]
    p = np.clip(phi   / phi_max,      0.0, 1.0)   # 0=[001]-[011] edge, 1=[111]

    # [001]→blue (0,0,1), [011]→green (0,1,0), [111]→red (1,0,0)
    r = p
    g = (1.0 - p) * t
    b = (1.0 - p) * (1.0 - t)

    return np.clip(np.stack([r, g, b], axis=1).astype(np.float32), 0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Voigt helpers
# ─────────────────────────────────────────────────────────────────────────────

_VOIGT_TENSOR = {
    "e_xx": (0, 0), "e_yy": (1, 1), "e_zz": (2, 2),
    "e_xy": (0, 1), "e_xz": (0, 2), "e_yz": (1, 2),
}


# ─────────────────────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────────────────────

class LayeredMap:
    """
    Per-pixel orientation and strain map for a :class:`~nrxrdct.laue.layers.LayeredCrystal`.

    Unlike :class:`~nrxrdct.laue.map.GrainMap`, all layers are predefined in
    the stack template — no grain selection or merging step is needed.  A single
    fit per pixel drives all layers simultaneously through a shared global
    rotation and optional per-layer strain tensor.

    Per-pixel arrays
    ----------------

    Scalar (shape ``(ny, nx)``):
        ``rms_px``, ``mean_px``, ``match_rate``, ``cost``, ``n_sim``,
        ``score``, ``score0`` (image-based).
        ``n_matched`` uses ``-1`` for unfitted pixels.

    Orientation (shape ``(n_layers, ny, nx, 3, 3)`` or ``(ny, nx, 3[, 3])``) :
        ``U``, ``U_eff``, ``R_global``, ``rotvec``.

    Strain (shape ``(n_layers, ny, nx, 3, 3)`` or ``(n_layers, ny, nx, 6)``):
        ``strain_tensor``, ``strain_voigt``, ``strain_tensor_deviatoric``.
    """

    def __init__(
        self,
        ny: int,
        nx: int,
        stack,
        *,
        h5_path: str | None = None,
        entry: str = "1.1",
        motor_x: str | None = None,
        motor_y: str | None = None,
        save_path: str | None = None,
        monitor: str | None = None,
    ) -> None:
        self.ny        = ny
        self.nx        = nx
        self.stack     = stack
        self.h5_path   = h5_path
        self.entry     = entry
        self.save_path = save_path
        self.monitor   = monitor

        n_layers = len(stack.all_layers)
        self.n_layers    = n_layers
        self.layer_labels = [
            getattr(l, "label", f"layer_{i}")
            for i, l in enumerate(stack.all_layers)
        ]

        # ── scalar per-pixel arrays ────────────────────────────────────────────
        self.rms_px     = np.full((ny, nx), np.nan)
        self.mean_px    = np.full((ny, nx), np.nan)
        self.n_matched  = np.full((ny, nx), -1, dtype=int)
        self.match_rate = np.full((ny, nx), np.nan)
        self.cost       = np.full((ny, nx), np.nan)
        self.n_sim      = np.full((ny, nx), -1, dtype=int)
        self.score      = np.full((ny, nx), np.nan)
        self.score0     = np.full((ny, nx), np.nan)

        # ── orientation arrays ─────────────────────────────────────────────────
        self.R_global = np.full((ny, nx, 3, 3), np.nan)
        self.rotvec   = np.full((ny, nx, 3), np.nan)
        self.U        = np.full((n_layers, ny, nx, 3, 3), np.nan)
        self.U_eff    = np.full((n_layers, ny, nx, 3, 3), np.nan)

        # ── per-layer strain arrays ────────────────────────────────────────────
        self.strain_tensor            = np.full((n_layers, ny, nx, 3, 3), np.nan)
        self.strain_voigt             = np.full((n_layers, ny, nx, 6), np.nan)
        self.strain_tensor_deviatoric = np.full((n_layers, ny, nx, 3, 3), np.nan)

        # ── motor positions ────────────────────────────────────────────────────
        self.motors: dict[str, np.ndarray] = {}
        if h5_path is not None:
            self._load_motors(motor_x, motor_y)

        # ── raw result cache (optional) ────────────────────────────────────────
        self._results: list[list] = [[None] * nx for _ in range(ny)]

    # ── Index helpers ──────────────────────────────────────────────────────────

    def frame_index(self, iy: int, ix: int) -> int:
        """Flat frame index for pixel ``(iy, ix)``."""
        return iy * self.nx + ix

    def map_index(self, frame_idx: int) -> tuple[int, int]:
        """Map pixel ``(iy, ix)`` for a flat frame index."""
        return divmod(frame_idx, self.nx)

    # ── Motor loading ──────────────────────────────────────────────────────────

    def _load_motors(self, motor_x: str | None, motor_y: str | None) -> None:
        n_frames = self.ny * self.nx
        with h5py.File(self.h5_path, "r") as hf:
            for name in (motor_x, motor_y):
                if name is None:
                    continue
                arr = _read_motor_array(hf, self.entry, name, n_frames)
                if arr is not None:
                    self.motors[name] = arr.reshape(self.ny, self.nx)

    # ── Result storage ─────────────────────────────────────────────────────────

    def set_result(self, iy: int, ix: int, result) -> None:
        """
        Store a stack fitting result for pixel ``(iy, ix)``.

        Accepts :class:`~nrxrdct.laue.fitting.StackFitResult`,
        :class:`~nrxrdct.laue.fitting.StackStrainFitResult`,
        :class:`~nrxrdct.laue.fitting.StackImageRefinementResult`, and
        :class:`~nrxrdct.laue.fitting.StackStrainImageRefinementResult`.
        """
        from .fitting import (
            StackFitResult, StackStrainFitResult,
            StackImageRefinementResult, StackStrainImageRefinementResult,
        )

        self._results[iy][ix] = result

        self.R_global[iy, ix] = result.R_global
        self.rotvec[iy, ix]   = result.rotvec
        for li, U in enumerate(result.U_layers):
            self.U[li, iy, ix] = U

        if isinstance(result, (StackFitResult, StackStrainFitResult)):
            self.rms_px[iy, ix]     = result.rms_px
            self.mean_px[iy, ix]    = result.mean_px
            self.n_matched[iy, ix]  = result.n_matched
            self.match_rate[iy, ix] = result.match_rate
            self.cost[iy, ix]       = result.cost
            self.n_sim[iy, ix]      = result.n_sim

        if isinstance(result, (StackImageRefinementResult, StackStrainImageRefinementResult)):
            self.score[iy, ix]  = result.score
            self.score0[iy, ix] = result.score0
            self.n_sim[iy, ix]  = result.n_sim

        if isinstance(result, (StackStrainFitResult, StackStrainImageRefinementResult)):
            for li, (U_eff, eps, voigt) in enumerate(zip(
                result.U_eff_layers, result.strain_tensors, result.strain_voigts
            )):
                self.U_eff[li, iy, ix]           = U_eff
                self.strain_tensor[li, iy, ix]   = eps
                self.strain_voigt[li, iy, ix]    = voigt
                self.strain_tensor_deviatoric[li, iy, ix] = (
                    eps - np.trace(eps) / 3.0 * np.eye(3)
                )

    def get_result(self, iy: int, ix: int):
        """Return the cached result object for pixel ``(iy, ix)``, or ``None``."""
        return self._results[iy][ix]

    # ── Collect from .npz files ────────────────────────────────────────────────

    def collect(self, out_dir: str) -> int:
        """
        Load all ``frame_?????.npz`` files written by the ``run_*_local``
        methods into the map arrays.

        Returns the number of frames successfully loaded.
        """
        files = sorted(glob.glob(os.path.join(out_dir, "frame_?????.npz")))
        n_ok  = 0
        for path in files:
            frame_idx = int(os.path.basename(path)[6:11])
            iy, ix = self.map_index(frame_idx)
            if iy >= self.ny or ix >= self.nx:
                continue
            try:
                d = np.load(path, allow_pickle=False)
            except Exception as exc:
                print(f"  ✗  {path}: {exc}", flush=True)
                continue

            self.R_global[iy, ix]  = d["R_global"]
            self.rotvec[iy, ix]    = d["rotvec"]
            self.U[:, iy, ix]      = d["U_layers"]

            if "rms_px" in d:
                self.rms_px[iy, ix]     = float(d["rms_px"])
                self.mean_px[iy, ix]    = float(d["mean_px"])
                self.n_matched[iy, ix]  = int(d["n_matched"])
                self.match_rate[iy, ix] = float(d["match_rate"])
                self.cost[iy, ix]       = float(d["cost"])
            if "score" in d:
                self.score[iy, ix]  = float(d["score"])
                self.score0[iy, ix] = float(d["score0"])
            if "n_sim" in d:
                self.n_sim[iy, ix] = int(d["n_sim"])

            if "strain_tensors" in d:
                eps_arr = d["strain_tensors"]   # (n_layers, 3, 3)
                self.U_eff[:, iy, ix]           = d["U_eff_layers"]
                self.strain_tensor[:, iy, ix]   = eps_arr
                self.strain_voigt[:, iy, ix]    = d["strain_voigts"]
                for li in range(self.n_layers):
                    eps = eps_arr[li]
                    self.strain_tensor_deviatoric[li, iy, ix] = (
                        eps - np.trace(eps) / 3.0 * np.eye(3)
                    )

            n_ok += 1

        print(f"collect: {n_ok}/{len(files)} frames loaded from {out_dir!r}",
              flush=True)
        return n_ok

    # ── Internal pool helpers ──────────────────────────────────────────────────

    def _serialize_stack(self) -> str:
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tf:
            path = tf.name
        with open(path, "wb") as fh:
            _dill.dump(self.stack, fh)
        return path

    def _load_peaklists(
        self,
        seg_dir: str,
        frame_indices: list[int],
        r_squared_min: float,
        include_unfitted: bool,
        min_matched: int,
    ) -> dict:
        from .segmentation import convert_spotsfile2peaklist

        peaklists: dict[int, np.ndarray] = {}
        for fi in frame_indices:
            path = os.path.join(seg_dir, f"frame_{fi:05d}.h5")
            if not os.path.exists(path):
                continue
            try:
                pl = convert_spotsfile2peaklist(
                    path,
                    r_squared_min=r_squared_min,
                    include_unfitted=include_unfitted,
                )
                if len(pl) >= min_matched:
                    peaklists[fi] = pl[:, :2]
            except Exception as exc:
                print(f"  ✗  frame {fi}: peaklist load: {exc}", flush=True)
        return peaklists

    def _load_frames(
        self, h5_path: str, h5_dataset: str, frame_indices: list[int]
    ) -> dict:
        frames: dict[int, np.ndarray] = {}
        with h5py.File(h5_path, "r") as hf:
            ds = hf[h5_dataset]
            for fi in frame_indices:
                try:
                    frames[fi] = ds[fi].astype(np.float32)
                except Exception as exc:
                    print(f"  ✗  frame {fi}: image read: {exc}", flush=True)
        return frames

    def _run_pool(
        self,
        worker_fn,
        frame_indices: list[int],
        task_args: dict,
        common_kwargs: dict,
        stack_pkl: str,
        camera,
        allowed_hkl,
        n_workers: int,
    ) -> int:
        n_total = len(frame_indices)
        tick    = max(1, n_total // 20)
        t0      = time.time()
        n_ok    = 0
        done    = 0

        with concurrent.futures.ProcessPoolExecutor(
            max_workers = n_workers,
            initializer = _lm_pool_init,
            initargs    = (stack_pkl, camera, allowed_hkl),
        ) as pool:
            futs = {
                pool.submit(worker_fn, fi, task_args.get(fi), **common_kwargs): fi
                for fi in frame_indices
            }
            for fut in concurrent.futures.as_completed(futs):
                done += 1
                try:
                    _, ok = fut.result()
                    if ok:
                        n_ok += 1
                except Exception as exc:
                    print(f"  ✗  frame {futs[fut]}: {exc}", flush=True)

                if done % tick == 0 or done == n_total:
                    elapsed = time.time() - t0
                    rate    = done / elapsed if elapsed > 0 else float("inf")
                    eta     = (n_total - done) / rate if rate > 0 else float("inf")
                    print(
                        f"  {done}/{n_total}  {n_ok} ok  "
                        f"{elapsed:.0f}s  ETA {eta:.0f}s",
                        flush=True,
                    )
        return n_ok

    def _precompute_allowed(
        self, E_max_eV: float, f2_thresh: float
    ) -> "dict | None":
        from .simulation import precompute_allowed_hkl

        _enum_pool = (
            self.stack.buffer_layers + self.stack.layers[:1]
            if (self.stack.buffer_layers or self.stack.layers)
            else self.stack.all_layers
        )
        return {
            id(l.crystal): precompute_allowed_hkl(
                l.crystal, E_max_eV=E_max_eV, f2_thresh=f2_thresh
            )
            for l in _enum_pool
        }

    # ── Local parallel fitting ─────────────────────────────────────────────────

    def run_orientation_local(
        self,
        camera,
        seg_dir: str,
        out_dir: str,
        *,
        r_squared_min: float = 0.0,
        include_unfitted: bool = False,
        max_match_px=(30, 10, 3),
        min_matched: int = 5,
        min_match_rate: float = 0.2,
        max_rms_px: "float | None" = None,
        geometry_only: bool = True,
        correct_depth: bool = False,
        f2_thresh: float = 1e-4,
        n_workers: "int | None" = None,
        overwrite: bool = False,
        frame_indices: "list[int] | None" = None,
        **fit_kwargs,
    ) -> int:
        """
        Fit stack orientation for every frame in *seg_dir* using local CPU cores.

        Results are written as ``frame_?????.npz`` in *out_dir*.
        Call :meth:`collect` afterwards to load them into the map arrays.

        Args:
            camera: Detector geometry.
            seg_dir: Directory with ``frame_?????.h5`` peaklist files.
            out_dir: Output directory for ``frame_?????.npz`` result files.
            max_match_px: Pixel match radius; a decreasing list enables staged
                refinement (e.g. ``[30, 10, 3]``).
            fit_kwargs: Forwarded to :func:`~nrxrdct.laue.fitting.fit_orientation_stack`.

        Returns:
            Number of frames successfully fitted.
        """
        os.makedirs(out_dir, exist_ok=True)

        if frame_indices is None:
            h5s = sorted(glob.glob(os.path.join(seg_dir, "frame_?????.h5")))
            frame_indices = [int(os.path.basename(p)[6:11]) for p in h5s]

        allowed_hkl = None
        if geometry_only:
            _E_max = fit_kwargs.get("E_max_eV", 27_000)
            allowed_hkl = self._precompute_allowed(_E_max, f2_thresh)

        peaklists = self._load_peaklists(
            seg_dir, frame_indices, r_squared_min, include_unfitted, min_matched
        )
        print(
            f"run_orientation_local: {len(frame_indices)} frames | "
            f"{len(peaklists)} with ≥{min_matched} spots | "
            f"{self.n_layers} layers",
            flush=True,
        )

        common = dict(
            out_dir        = out_dir,
            max_match_px   = max_match_px,
            min_matched    = min_matched,
            min_match_rate = min_match_rate,
            max_rms_px     = max_rms_px,
            fit_kwargs     = {**fit_kwargs, "geometry_only": geometry_only,
                               "correct_depth": correct_depth},
            overwrite      = overwrite,
        )
        stack_pkl = self._serialize_stack()
        try:
            n_ok = self._run_pool(
                _lm_orient_frame, frame_indices, peaklists, common,
                stack_pkl, camera, allowed_hkl,
                n_workers or os.cpu_count() or 1,
            )
        finally:
            os.unlink(stack_pkl)

        print(f"  → {n_ok}/{len(frame_indices)} frames fitted", flush=True)
        return n_ok

    def run_strain_local(
        self,
        camera,
        seg_dir: str,
        out_dir: str,
        *,
        fit_strain: "tuple | None" = None,
        orient_dir: "str | None" = None,
        r_squared_min: float = 0.0,
        include_unfitted: bool = False,
        max_match_px=(10, 3),
        min_matched: int = 5,
        min_match_rate: float = 0.2,
        max_rms_px: "float | None" = None,
        geometry_only: bool = True,
        correct_depth: bool = False,
        f2_thresh: float = 1e-4,
        n_workers: "int | None" = None,
        overwrite: bool = False,
        frame_indices: "list[int] | None" = None,
        **fit_kwargs,
    ) -> int:
        """
        Fit stack orientation + per-layer strain for every frame using local cores.

        Mirrors :meth:`run_orientation_local` but calls
        :func:`~nrxrdct.laue.fitting.fit_strain_orientation_stack` instead.
        Start from a good orientation (run :meth:`run_orientation_local` first
        and :meth:`collect` the result into the stack's layer U matrices).

        Args:
            fit_strain: Strain components to refine, e.g.
                ``('e_xx', 'e_yy', 'e_zz')``.  ``None`` refines all six.
        """
        from .fitting import _STRAIN_ALL
        _fit_strain = tuple(fit_strain) if fit_strain is not None else _STRAIN_ALL
        os.makedirs(out_dir, exist_ok=True)

        if frame_indices is None:
            h5s = sorted(glob.glob(os.path.join(seg_dir, "frame_?????.h5")))
            frame_indices = [int(os.path.basename(p)[6:11]) for p in h5s]

        allowed_hkl = None
        if geometry_only:
            _E_max = fit_kwargs.get("E_max_eV", 27_000)
            allowed_hkl = self._precompute_allowed(_E_max, f2_thresh)

        peaklists = self._load_peaklists(
            seg_dir, frame_indices, r_squared_min, include_unfitted, min_matched
        )
        print(
            f"run_strain_local: {len(frame_indices)} frames | "
            f"{len(peaklists)} with ≥{min_matched} spots | "
            f"fit_strain={list(_fit_strain)}",
            flush=True,
        )

        # Load per-frame orientation warm-starts if an orient_dir was given.
        orient_data: dict = {}
        if orient_dir is not None:
            for fi in frame_indices:
                path = os.path.join(orient_dir, f"frame_{fi:05d}.npz")
                if os.path.exists(path):
                    try:
                        d = np.load(path, allow_pickle=False)
                        if "U_layers" in d:
                            orient_data[fi] = d["U_layers"]   # (n_layers, 3, 3)
                    except Exception as exc:
                        print(f"  ✗  frame {fi}: orient load: {exc}", flush=True)
            print(
                f"  orient warm-start: {len(orient_data)}/{len(frame_indices)} "
                f"frames from {orient_dir!r}",
                flush=True,
            )

        common = dict(
            out_dir        = out_dir,
            fit_strain     = _fit_strain,
            max_match_px   = max_match_px,
            min_matched    = min_matched,
            min_match_rate = min_match_rate,
            max_rms_px     = max_rms_px,
            fit_kwargs     = {**fit_kwargs, "geometry_only": geometry_only,
                               "correct_depth": correct_depth},
            overwrite      = overwrite,
            orient_data    = orient_data or None,
        )
        stack_pkl = self._serialize_stack()
        try:
            n_ok = self._run_pool(
                _lm_strain_frame, frame_indices, peaklists, common,
                stack_pkl, camera, allowed_hkl,
                n_workers or os.cpu_count() or 1,
            )
        finally:
            os.unlink(stack_pkl)

        print(f"  → {n_ok}/{len(frame_indices)} frames fitted", flush=True)
        return n_ok

    def run_image_orientation_local(
        self,
        camera,
        h5_path: str,
        h5_dataset: str,
        out_dir: str,
        *,
        geometry_only: bool = True,
        correct_depth: bool = False,
        f2_thresh: float = 1e-4,
        n_workers: "int | None" = None,
        overwrite: bool = False,
        frame_indices: "list[int] | None" = None,
        **fit_kwargs,
    ) -> int:
        """
        Image-based orientation refinement for every frame using local cores.

        Calls :func:`~nrxrdct.laue.fitting.refine_orientation_image_stack`.
        The stack's layer U matrices must already be set to a good starting
        orientation (e.g. via a prior :meth:`run_orientation_local` + stack
        update).

        Args:
            h5_path: Path to the HDF5 scan file.
            h5_dataset: Dataset path inside *h5_path*, e.g.
                ``"1.1/measurement/eiger4m"``.
            out_dir: Output directory for result ``.npz`` files.
        """
        os.makedirs(out_dir, exist_ok=True)

        if frame_indices is None:
            frame_indices = list(range(self.ny * self.nx))

        allowed_hkl = None
        if geometry_only:
            _E_max = fit_kwargs.get("E_max", 27_000)
            allowed_hkl = self._precompute_allowed(_E_max, f2_thresh)

        frames = self._load_frames(h5_path, h5_dataset, frame_indices)
        print(
            f"run_image_orientation_local: {len(frame_indices)} frames loaded",
            flush=True,
        )

        common = dict(out_dir=out_dir,
                      fit_kwargs={**fit_kwargs, "correct_depth": correct_depth},
                      overwrite=overwrite)
        stack_pkl = self._serialize_stack()
        try:
            n_ok = self._run_pool(
                _lm_img_orient_frame, frame_indices, frames, common,
                stack_pkl, camera, allowed_hkl,
                n_workers or os.cpu_count() or 1,
            )
        finally:
            os.unlink(stack_pkl)

        return n_ok

    def run_strain_image_local(
        self,
        camera,
        h5_path: str,
        h5_dataset: str,
        out_dir: str,
        *,
        fit_strain: "tuple | None" = None,
        geometry_only: bool = True,
        correct_depth: bool = False,
        f2_thresh: float = 1e-4,
        n_workers: "int | None" = None,
        overwrite: bool = False,
        frame_indices: "list[int] | None" = None,
        **fit_kwargs,
    ) -> int:
        """
        Image-based orientation + per-layer strain refinement using local cores.

        Calls :func:`~nrxrdct.laue.fitting.refine_strain_image_stack`.
        Pass ``strain0_list`` via *fit_kwargs* to warm-start from a prior
        strain fit.

        Args:
            fit_strain: Strain components to refine.  ``None`` refines all six.
        """
        from .fitting import _STRAIN_ALL
        _fit_strain = tuple(fit_strain) if fit_strain is not None else _STRAIN_ALL
        os.makedirs(out_dir, exist_ok=True)

        if frame_indices is None:
            frame_indices = list(range(self.ny * self.nx))

        allowed_hkl = None
        if geometry_only:
            _E_max = fit_kwargs.get("E_max", 27_000)
            allowed_hkl = self._precompute_allowed(_E_max, f2_thresh)

        frames = self._load_frames(h5_path, h5_dataset, frame_indices)
        print(
            f"run_strain_image_local: {len(frame_indices)} frames | "
            f"fit_strain={list(_fit_strain)}",
            flush=True,
        )

        common = dict(
            out_dir=out_dir, fit_strain=_fit_strain,
            fit_kwargs={**fit_kwargs, "correct_depth": correct_depth},
            overwrite=overwrite,
        )
        stack_pkl = self._serialize_stack()
        try:
            n_ok = self._run_pool(
                _lm_img_strain_frame, frame_indices, frames, common,
                stack_pkl, camera, allowed_hkl,
                n_workers or os.cpu_count() or 1,
            )
        finally:
            os.unlink(stack_pkl)

        return n_ok

    # ── Plotting ───────────────────────────────────────────────────────────────

    def _motor_extent(
        self, motor_x: "str | None", motor_y: "str | None"
    ) -> tuple:
        """Return (extent, xlabel, ylabel) for imshow, or (None, 'col', 'row')."""
        mx = self.motors.get(motor_x) if motor_x else None
        my = self.motors.get(motor_y) if motor_y else None
        if mx is not None and my is not None:
            return ([mx.min(), mx.max(), my.max(), my.min()],
                    motor_x, motor_y)
        return None, "column", "row"

    # ── Segmentation helpers ───────────────────────────────────────────────────

    def load_n_obs_map(self, seg_dir: str) -> np.ndarray:
        """Return a ``(ny, nx)`` map of segmented spot counts (-1 = no file)."""
        import re as _re
        n_obs = np.full((self.ny, self.nx), -1, dtype=int)
        pat = _re.compile(r'^frame_(\d+)\.h5$', _re.IGNORECASE)
        for fname in os.listdir(seg_dir):
            m = pat.match(fname)
            if not m:
                continue
            frame_idx = int(m.group(1))
            iy, ix = self.map_index(frame_idx)
            if iy >= self.ny or ix >= self.nx:
                continue
            fpath = os.path.join(seg_dir, fname)
            try:
                with h5py.File(fpath, "r") as fh:
                    if "n_spots" in fh.attrs:
                        n_obs[iy, ix] = int(fh.attrs["n_spots"])
                    else:
                        n_obs[iy, ix] = sum(
                            1 for k in fh if k.startswith("spot_")
                        )
            except Exception:
                pass
        return n_obs

    def segment_frame(
        self,
        base_dir: str,
        detector_mask: "np.ndarray | None" = None,
        *,
        h5_dataset: "str | None" = None,
        tiff_dir: "str | None" = None,
        map_quantity: str = "n_obs",
        motor_x: "str | None" = None,
        motor_y: "str | None" = None,
        motor_units: "dict | None" = None,
        figsize: tuple = (14, 7),
    ) -> None:
        """
        Interactive single-frame segmentation tuner.

        Click a pixel on the left map panel to load the frame, adjust
        segmentation parameters, press **⚙ Segment** to preview, then
        **💾 Save** to write ``<base_dir>/seg/frame_NNNNN.h5``.

        Args:
            base_dir: Processing root; seg files go to ``<base_dir>/seg/``.
            detector_mask: Valid-pixel mask (``True`` = active).
            h5_dataset: Dataset path inside ``self.h5_path``.
                Mutually exclusive with *tiff_dir*.
            tiff_dir: Folder of ``img_*.tif`` files.
                Mutually exclusive with *h5_dataset*.
            map_quantity: Left-panel overview — ``'n_obs'`` (default),
                ``'match_rate'``, ``'rms_px'``, ``'score'``, ``'cost'``.
            motor_x, motor_y: Motor names in ``self.motors`` for physical axes.
            motor_units: Unit labels, e.g. ``{'sy': 'mm', 'sz': 'mm'}``.
            figsize: Figure size in inches.
        """
        import ipywidgets as ipw
        from IPython.display import display as _ipy_display
        from .segmentation import (
            LoG_segmentation, WTH_segmentation, hybrid_segmentation,
            clean_segmentation, label_segmented_image, measure_peaks,
            filter_and_rescale_images, gaussian_background, write_h5_spotsfile,
        )

        seg_dir = os.path.join(base_dir, "seg")
        os.makedirs(seg_dir, exist_ok=True)

        mu = motor_units or {}
        mx = self.motors.get(motor_x) if motor_x else None
        my = self.motors.get(motor_y) if motor_y else None

        if mx is not None and my is not None:
            extent_map = [mx[0, 0], mx[0, -1], my[-1, 0], my[0, 0]]
            xu_ = mu.get(motor_x, "")
            yu_ = mu.get(motor_y, "")
            xlabel_map = f"{motor_x} ({xu_})" if xu_ else motor_x
            ylabel_map = f"{motor_y} ({yu_})" if yu_ else motor_y
        else:
            extent_map = [0, self.nx, self.ny, 0]
            xlabel_map = "column (ix)"
            ylabel_map = "row (iy)"

        def _click_to_iy_ix(xdata: float, ydata: float):
            if mx is not None and my is not None:
                dist = (mx - xdata) ** 2 + (my - ydata) ** 2
                iy, ix = np.unravel_index(int(np.argmin(dist)), dist.shape)
            else:
                ix = int(np.clip(int(xdata), 0, self.nx - 1))
                iy = int(np.clip(int(ydata), 0, self.ny - 1))
            return int(iy), int(ix)

        _tiff_index: list = []

        def _load_image(frame_idx: int) -> "np.ndarray | None":
            if tiff_dir is not None:
                if not _tiff_index:
                    import re as _re
                    pat = _re.compile(r'(\d+)\.tiff?$', _re.IGNORECASE)
                    files = []
                    for fname in os.listdir(tiff_dir):
                        m = pat.search(fname)
                        if m:
                            files.append((int(m.group(1)), os.path.join(tiff_dir, fname)))
                    files.sort(key=lambda t: t[0])
                    _tiff_index.extend(p for _, p in files)
                    if not _tiff_index:
                        print(f"  ✗ No .tif/.tiff files found in {tiff_dir!r}", flush=True)
                        return None
                if frame_idx >= len(_tiff_index):
                    print(f"  ✗ frame {frame_idx} out of range", flush=True)
                    return None
                try:
                    import skimage.io
                    return skimage.io.imread(_tiff_index[frame_idx]).astype(np.float32)
                except Exception as exc:
                    print(f"  ✗ TIFF read error frame {frame_idx}: {exc}", flush=True)
                    return None
            elif h5_dataset is not None:
                _h5 = self.h5_path
                if _h5 is None:
                    print("  ✗ h5_path not set on this LayeredMap.", flush=True)
                    return None
                try:
                    with h5py.File(_h5, "r") as fh:
                        if h5_dataset not in fh:
                            print(f"  ✗ dataset {h5_dataset!r} not in {_h5!r}", flush=True)
                            return None
                        img = fh[h5_dataset][frame_idx].astype(np.float32)
                        if self.monitor and self.monitor in fh:
                            mon_val = float(fh[self.monitor][frame_idx])
                            if mon_val > 0:
                                img /= mon_val
                        return img
                except Exception as exc:
                    print(f"  ✗ HDF5 read error frame {frame_idx}: {exc}", flush=True)
                    return None
            return None

        def _build_n_obs_data() -> np.ndarray:
            raw = self.load_n_obs_map(seg_dir)
            return np.where(raw >= 0, raw.astype(float), np.nan)

        _map_opts = {
            "n_obs":       (_build_n_obs_data,                 "N spots (seg)",  "YlOrRd"),
            "match_rate":  (lambda: self.match_rate,           "Match rate",     "viridis"),
            "rms_px":      (lambda: self.rms_px,               "RMS (px)",       "plasma_r"),
            "mean_px":     (lambda: self.mean_px,              "Mean dev (px)",  "plasma_r"),
            "score":       (lambda: self.score,                "Image score",    "viridis"),
            "cost":        (lambda: self.cost,                 "Cost",           "plasma_r"),
        }
        _data_fn, map_label, map_cmap = _map_opts.get(
            map_quantity, _map_opts["n_obs"]
        )
        map_data = _data_fn()

        # ── figure ────────────────────────────────────────────────────────────
        with plt.ioff():
            fig = plt.figure(figsize=figsize)
        try:
            fig.canvas.manager.set_window_title("Laue — segment frame")
        except Exception:
            pass

        gs = fig.add_gridspec(
            1, 2, width_ratios=[1, 1.8], wspace=0.08,
            left=0.07, right=0.97, bottom=0.09, top=0.91,
        )
        ax_map = fig.add_subplot(gs[0])
        ax_det = fig.add_subplot(gs[1])

        im_map = ax_map.imshow(
            map_data, origin="upper", extent=extent_map,
            cmap=map_cmap, interpolation="nearest", aspect="auto",
        )
        fig.colorbar(im_map, ax=ax_map, fraction=0.04, pad=0.03,
                     shrink=0.8, label=map_label)
        ax_map.set_xlabel(xlabel_map, fontsize=9)
        ax_map.set_ylabel(ylabel_map, fontsize=9)
        ax_map.set_title(f"Click to select — {map_label}", fontsize=9)
        sel_dot, = ax_map.plot([], [], "w+", ms=11, mew=2.0, zorder=10)

        ax_det.set_facecolor("k")
        ax_det.set_aspect("equal")
        ax_det.set_xlabel("x (px)", fontsize=9)
        ax_det.set_ylabel("y (px)", fontsize=9)
        ax_det.set_title("← click map to select a frame", fontsize=9, color="#888")
        fig.suptitle(
            "Segment frame  —  click map → adjust params → Segment → Save",
            fontsize=9, color="#555",
        )

        _state: dict = {
            "frame_idx": None, "iy": None, "ix": None,
            "image": None, "proc_image": None, "props": None, "drawn": False,
            "saved_xy": None,   # (N,2) xy from existing seg file, or None
        }

        def _parse(s: str):
            parts = [x.strip() for x in s.split(",") if x.strip()]
            if not parts:
                raise ValueError(f"Empty field: {s!r}")
            nums = [float(p) if "." in p else int(p) for p in parts]
            return nums[0] if len(nums) == 1 else nums

        def _draw_det(props=None) -> None:
            disp = _state["proc_image"] if _state["proc_image"] is not None else _state["image"]
            saved_xlim = ax_det.get_xlim()
            saved_ylim = ax_det.get_ylim()
            ax_det.cla()
            ax_det.set_facecolor("k")
            if disp is None:
                ax_det.set_title("← click map to select a frame", fontsize=9, color="#888")
                fig.canvas.draw_idle()
                return
            nv, nh = disp.shape
            try:
                user_vmin = float(w_vmin.value) if w_vmin.value.strip() else None
            except ValueError:
                user_vmin = None
            try:
                user_vmax = float(w_vmax.value) if w_vmax.value.strip() else None
            except ValueError:
                user_vmax = None
            pos = disp[disp > 0]
            clip_min = user_vmin if user_vmin is not None else 0.0
            clip_max = user_vmax if user_vmax is not None else (
                float(np.percentile(pos, 99)) if pos.size else 1.0
            )
            if clip_max <= clip_min:
                clip_max = clip_min + 1.0
            disp_clipped = np.clip(disp, clip_min, clip_max)
            disp_norm    = (disp_clipped - clip_min) / (clip_max - clip_min)
            disp_show = np.log1p(disp_norm * 1000) if w_log_scale.value else disp_norm
            ax_det.imshow(disp_show, origin="upper", extent=[0, nh, nv, 0],
                          cmap="gray", aspect="equal", zorder=0)
            _has_legend = False
            saved_xy = _state.get("saved_xy")
            if saved_xy is not None and len(saved_xy) > 0:
                ax_det.scatter(saved_xy[:, 0], saved_xy[:, 1], s=35,
                               c="none", edgecolors="#ffcc00",
                               linewidths=0.9, zorder=3,
                               label=f"saved ({len(saved_xy)})")
                _has_legend = True
            if props is not None and len(props) > 0:
                ys = np.array([p.centroid_weighted[0] for p in props])
                xs = np.array([p.centroid_weighted[1] for p in props])
                ax_det.scatter(xs, ys, s=35, c="none", edgecolors="#44aaff",
                               linewidths=0.9, zorder=4, label=f"new ({len(props)})")
                _has_legend = True
            if _has_legend:
                ax_det.legend(fontsize=7, loc="upper right",
                              facecolor="#111", edgecolor="#444", labelcolor="white",
                              framealpha=0.85)
            ax_det.set_aspect("equal")
            ax_det.set_xlabel("x (px)", fontsize=9)
            ax_det.set_ylabel("y (px)", fontsize=9)
            iy, ix = _state["iy"], _state["ix"]
            n_new   = len(props) if props is not None else None
            n_saved = len(saved_xy) if saved_xy is not None else None
            if n_new is not None and n_saved is not None:
                suffix = f"  — {n_new} new  |  {n_saved} saved"
            elif n_new is not None:
                suffix = f"  — {n_new} spots"
            elif n_saved is not None:
                suffix = f"  — {n_saved} saved spots"
            else:
                suffix = ""
            ax_det.set_title(f"Frame {_state['frame_idx']}  (iy={iy}, ix={ix}){suffix}", fontsize=9)
            if _state["drawn"]:
                ax_det.set_xlim(saved_xlim)
                ax_det.set_ylim(saved_ylim)
            _state["drawn"] = True
            fig.canvas.draw_idle()

        def _on_click(event) -> None:
            if event.inaxes is not ax_map:
                return
            if event.xdata is None or event.ydata is None:
                return
            try:
                if fig.canvas.toolbar.mode != "":
                    return
            except Exception:
                pass
            iy, ix    = _click_to_iy_ix(event.xdata, event.ydata)
            frame_idx = self.frame_index(iy, ix)
            if mx is not None and my is not None:
                sel_dot.set_data([mx[iy, ix]], [my[iy, ix]])
            else:
                sel_dot.set_data([ix + 0.5], [iy + 0.5])
            image = _load_image(frame_idx)
            # Load existing segmentation if present
            saved_xy = None
            seg_path = os.path.join(seg_dir, f"frame_{frame_idx:05d}.h5")
            if os.path.exists(seg_path):
                try:
                    from .segmentation import convert_spotsfile2peaklist
                    pl = convert_spotsfile2peaklist(
                        seg_path, include_unfitted=True, r_squared_min=0.0
                    )
                    if len(pl) > 0:
                        saved_xy = pl[:, :2]   # (N, 2) x, y
                except Exception:
                    pass
            _state.update(frame_idx=frame_idx, iy=iy, ix=ix,
                          image=image, proc_image=None, props=None,
                          saved_xy=saved_xy)
            btn_segment.disabled = image is None
            btn_save.disabled    = True
            n_saved = len(saved_xy) if saved_xy is not None else 0
            _info.value = (
                f"<span style='color:#aaa'>Frame {frame_idx} loaded"
                + (f" — {n_saved} existing spots" if n_saved else "")
                + (" — no image data" if image is None else "") + "</span>"
            )
            _draw_det(props=None)

        fig.canvas.mpl_connect("button_press_event", _on_click)

        # ── widgets ───────────────────────────────────────────────────────────
        _sk  = dict(continuous_update=False, style={"description_width": "110px"},
                    layout=ipw.Layout(width="320px"))
        _isk = dict(style={"description_width": "90px"}, layout=ipw.Layout(width="160px"))

        w_method = ipw.Dropdown(
            options=["LoG", "WTH", "Hybrid"], value="LoG",
            description="Method:", layout=ipw.Layout(width="220px"),
            style={"description_width": "70px"},
        )
        w_log_sigmas = ipw.Text(value="2, 4, 8", description="Sigmas:", **_sk,
                                placeholder="e.g. 2, 4, 8  or  4")
        w_wth_radius = ipw.Text(value="5, 7",    description="Disk radii:", **_sk,
                                placeholder="e.g. 5, 7  or  7")
        w_hyb_log    = ipw.Text(value="2, 4, 8", description="LoG sigmas:", **_sk,
                                placeholder="e.g. 2, 4, 8")
        w_hyb_wth    = ipw.Text(value="5, 7",    description="WTH radii:", **_sk,
                                placeholder="e.g. 5, 7")
        w_thresh     = ipw.FloatText(value=99.9, description="Threshold %:", **_sk)
        w_bg_sigma   = ipw.FloatText(value=5.0,  description="BG sigma:", **_sk)
        w_min_size   = ipw.IntText(value=3,   description="min_size:",    **_isk)
        w_max_size   = ipw.IntText(value=500, description="max_size:",    **_isk)
        w_gap_excl   = ipw.IntText(value=3,   description="gap_exclude:", **_isk)
        w_gap_clos   = ipw.IntText(value=3,   description="gap_closing:", **_isk)
        w_d          = ipw.IntText(value=10,  description="Crop d (px):", **_isk)
        w_r2         = ipw.FloatText(value=0.9, description="R² min:",    **_isk)
        w_fit_spots  = ipw.Checkbox(value=True, description="Fit spots (Gaussian)",
                                    layout=ipw.Layout(width="200px"),
                                    style={"description_width": "initial"})
        w_overwrite  = ipw.Checkbox(value=True, description="Overwrite existing",
                                    layout=ipw.Layout(width="200px"))

        box_log    = ipw.VBox([w_log_sigmas])
        box_wth    = ipw.VBox([w_wth_radius])
        box_hybrid = ipw.VBox([w_hyb_log, w_hyb_wth])
        box_wth.layout.display    = "none"
        box_hybrid.layout.display = "none"

        def _on_method(change) -> None:
            m = change["new"]
            box_log.layout.display    = "" if m == "LoG"    else "none"
            box_wth.layout.display    = "" if m == "WTH"    else "none"
            box_hybrid.layout.display = "" if m == "Hybrid" else "none"

        w_method.observe(_on_method, names="value")

        _bkw = dict(layout=ipw.Layout(width="130px", height="32px"))
        btn_segment = ipw.Button(description="⚙ Segment", button_style="primary", **_bkw)
        btn_save    = ipw.Button(description="💾 Save",   button_style="success", **_bkw)
        btn_segment.disabled = True
        btn_save.disabled    = True

        _info = ipw.HTML(
            "<span style='color:#666;font-style:italic'>click a map pixel to load a frame</span>",
            layout=ipw.Layout(margin="4px 0 0 6px"),
        )

        def _cb_segment(_) -> None:
            import asyncio
            import queue as _qmod
            import threading

            image = _state["image"]
            if image is None:
                return
            if getattr(_cb_segment, "_running", False):
                return
            _cb_segment._running    = True
            btn_segment.disabled    = True
            btn_segment.description = "Running…"

            method = w_method.value
            try:
                if method == "LoG":
                    method_kwargs = {"sigmas": _parse(w_log_sigmas.value),
                                     "threshold_percentile": w_thresh.value}
                elif method == "WTH":
                    method_kwargs = {"disk_radius": _parse(w_wth_radius.value),
                                     "threshold_percentile": w_thresh.value}
                else:
                    method_kwargs = {"log_sigmas": _parse(w_hyb_log.value),
                                     "wth_disk_radius": _parse(w_hyb_wth.value),
                                     "threshold_percentile": w_thresh.value}
            except ValueError as exc:
                _info.value = f"<b style='color:#f44'>Parameter error: {exc}</b>"
                btn_segment.description = "⚙ Segment"
                btn_segment.disabled    = False
                _cb_segment._running    = False
                return

            clean_kw = dict(min_size=w_min_size.value, max_size=w_max_size.value,
                            gap_exclude=w_gap_excl.value, gap_closing=w_gap_clos.value)
            bg_sigma = w_bg_sigma.value
            _det_mask = (detector_mask.astype(bool) if detector_mask is not None
                         else np.ones(image.shape, dtype=bool))

            q: _qmod.Queue = _qmod.Queue()

            def _run() -> None:
                try:
                    valid = _det_mask
                    frame = image.copy().astype(np.float32)
                    if bg_sigma > 0:
                        bg    = gaussian_background(frame, valid, sigma=bg_sigma)
                        frame = frame - bg
                    frame -= frame[valid].min()
                    frame[~valid] = 0.0
                    if method == "WTH":
                        seg = WTH_segmentation(frame, valid, **method_kwargs)
                    elif method == "Hybrid":
                        seg = hybrid_segmentation(frame, valid, **method_kwargs)
                    else:
                        seg = LoG_segmentation(frame, valid, **method_kwargs)
                    final_mask, _ = clean_segmentation(seg, valid, frame, **clean_kw)
                    filt = filter_and_rescale_images(frame, cutoff_freq=0.001)
                    labels, _, _ = label_segmented_image(final_mask, filt)
                    props = measure_peaks(labels, filt)
                    q.put(("ok", props, filt))
                except Exception as exc:
                    q.put(("err", exc))

            async def _wait() -> None:
                threading.Thread(target=_run, daemon=True).start()
                while q.empty():
                    await asyncio.sleep(0.15)
                tag, *payload = q.get_nowait()
                if tag == "err":
                    _info.value = f"<b style='color:#f44'>Error: {payload[0]}</b>"
                else:
                    props, filt = payload
                    _state["props"] = props
                    _state["proc_image"] = filt
                    n = len(props)
                    col = "#44dd66" if n > 0 else "#ffaa33"
                    _info.value = f"<b style='color:{col}'>{n} spots detected</b>"
                    btn_save.disabled = n == 0
                    _draw_det(props=props)
                    if n > 0:
                        _cb_save(None)
                btn_segment.description = "⚙ Segment"
                btn_segment.disabled    = False
                _cb_segment._running    = False

            try:
                asyncio.get_event_loop().create_task(_wait())
            except RuntimeError:
                asyncio.ensure_future(_wait())

        def _cb_save(_) -> None:
            props = _state["props"]
            if not props:
                return
            frame_idx = _state["frame_idx"]
            out_path  = os.path.join(seg_dir, f"frame_{frame_idx:05d}.h5")
            try:
                write_h5_spotsfile(
                    _state["image"], props, outpath=out_path,
                    d=w_d.value, r_squared_min=w_r2.value,
                    overwrite=w_overwrite.value, fit_spots=w_fit_spots.value,
                )
                _info.value = f"<b style='color:#44dd66'>Saved → {out_path}</b>"
                print(f"  💾 Saved → {os.path.abspath(out_path)}")
                _state["saved_xy"] = None   # new result replaces old overlay
                if map_quantity == "n_obs":
                    new_data = _build_n_obs_data()
                    im_map.set_data(new_data)
                    valid = new_data[np.isfinite(new_data)]
                    if valid.size:
                        im_map.set_clim(valid.min(), valid.max())
                    fig.canvas.draw_idle()
            except Exception as exc:
                _info.value = f"<b style='color:#f44'>Save error: {exc}</b>"

        _dsk = dict(style={"description_width": "40px"},
                    layout=ipw.Layout(width="130px"), continuous_update=False)
        w_log_scale = ipw.Checkbox(value=True, description="Log scale",
                                   layout=ipw.Layout(width="110px"),
                                   style={"description_width": "initial"})
        w_vmin = ipw.Text(value="", description="vmin:", placeholder="auto", **_dsk)
        w_vmax = ipw.Text(value="", description="vmax:", placeholder="auto (99th %)", **_dsk)

        def _redraw(_=None) -> None:
            if _state["image"] is not None:
                _draw_det(props=_state["props"])

        def _on_fit_toggle(change) -> None:
            w_r2.disabled = not change["new"]

        w_log_scale.observe(_redraw, names="value")
        w_vmin.observe(_redraw, names="value")
        w_vmax.observe(_redraw, names="value")
        w_fit_spots.observe(_on_fit_toggle, names="value")
        btn_segment.on_click(_cb_segment)
        btn_save.on_click(_cb_save)

        _controls = ipw.VBox([
            ipw.HBox([w_method], layout=ipw.Layout(margin="4px 0 2px 0")),
            ipw.HBox([
                ipw.VBox([box_log, box_wth, box_hybrid, w_thresh, w_bg_sigma],
                         layout=ipw.Layout(padding="0 16px 0 0")),
                ipw.VBox([
                    ipw.HTML("<b>Clean:</b>"),
                    ipw.HBox([w_min_size, w_max_size]),
                    ipw.HBox([w_gap_excl, w_gap_clos]),
                    ipw.HTML("<b>Save:</b>"),
                    ipw.HBox([w_d, w_r2]),
                    w_fit_spots, w_overwrite,
                ]),
            ]),
            ipw.HBox([btn_segment, btn_save],
                     layout=ipw.Layout(gap="8px", margin="6px 0 0 0")),
            ipw.HBox(
                [ipw.HTML("<b style='line-height:26px;margin-right:6px'>Display:</b>"),
                 w_log_scale, w_vmin, w_vmax],
                layout=ipw.Layout(gap="6px", margin="4px 0 0 0", align_items="center"),
            ),
            _info,
        ], layout=ipw.Layout(padding="6px 8px"))

        _ipy_display(ipw.VBox([fig.canvas, _controls]))

    def plot_map(
        self,
        quantity: str,
        *,
        layer: "int | None" = None,
        strain_component: "str | None" = None,
        frame: str = "crystal",
        motor_x: "str | None" = None,
        motor_y: "str | None" = None,
        cmap: str = "viridis",
        vmin=None,
        vmax=None,
        title: "str | None" = None,
        ax=None,
    ):
        """
        Plot a scalar quantity map.

        Args:
            quantity: One of ``'rms_px'``, ``'mean_px'``, ``'match_rate'``,
                ``'n_matched'``, ``'cost'``, ``'score'``, ``'score0'``,
                ``'n_sim'``, ``'rotation_deg'`` (magnitude of global rotation),
                or ``'strain'`` (requires *layer* and *strain_component*).
            layer: Layer index (required for ``'strain'``).
            strain_component: One of ``'e_xx'``, ``'e_yy'``, ``'e_zz'``,
                ``'e_xy'``, ``'e_xz'``, ``'e_yz'``.
            frame: ``'crystal'`` (default), ``'deviatoric'`` (strain only).
        """
        if quantity == "strain":
            if layer is None or strain_component is None:
                raise ValueError(
                    "plot_map('strain') requires layer= and strain_component="
                )
            ti, tj = _VOIGT_TENSOR[strain_component]
            if frame == "deviatoric":
                data = self.strain_tensor_deviatoric[layer, :, :, ti, tj]
            else:
                data = self.strain_tensor[layer, :, :, ti, tj]
            label = f"ε {strain_component}"
            if layer is not None:
                label += f" [{self.layer_labels[layer]}]"

        elif quantity == "rotation_deg":
            norms = np.linalg.norm(self.rotvec, axis=-1)
            data  = np.where(np.isnan(self.rotvec[:, :, 0]), np.nan,
                             np.degrees(norms))
            label = "|δω| (°)"

        else:
            arr = getattr(self, quantity, None)
            if arr is None:
                raise ValueError(f"Unknown quantity {quantity!r}")
            data  = np.where(arr == -1, np.nan, arr).astype(float)
            label = quantity.replace("_", " ")

        extent, xlabel, ylabel = self._motor_extent(motor_x, motor_y)
        if ax is None:
            _, ax = plt.subplots(figsize=(7, 6))

        kw = dict(origin="upper", cmap=cmap, vmin=vmin, vmax=vmax)
        if extent is not None:
            kw.update(extent=extent, aspect="auto")

        im = ax.imshow(data, **kw)
        plt.colorbar(im, ax=ax, label=label)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title or label)
        return ax

    def plot_strain_component(
        self,
        component: str,
        layer: int,
        *,
        frame: str = "crystal",
        motor_x: "str | None" = None,
        motor_y: "str | None" = None,
        cmap: str = "RdBu_r",
        vmin=None,
        vmax=None,
        ax=None,
    ):
        """
        Plot one strain component for *layer*.

        Shortcut for ``plot_map('strain', layer=layer, strain_component=component)``.
        """
        return self.plot_map(
            "strain",
            layer=layer,
            strain_component=component,
            frame=frame,
            motor_x=motor_x,
            motor_y=motor_y,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            ax=ax,
            title=f"ε {component} — {self.layer_labels[layer]}",
        )

    def plot_overview(
        self,
        layer: int = 0,
        *,
        strain_components: tuple = ("e_xx", "e_yy", "e_zz"),
        motor_x: "str | None" = None,
        motor_y: "str | None" = None,
        cmap_scalar: str = "viridis",
        cmap_strain: str = "RdBu_r",
        figsize: "tuple | None" = None,
    ):
        """
        Multi-panel overview: match quality + strain components for *layer*.

        Panels: ``match_rate``, ``rms_px``, ``rotation_deg``, then one panel
        per entry in *strain_components*.
        """
        n_panels = 3 + len(strain_components)
        if figsize is None:
            figsize = (5 * n_panels, 4)
        fig, axes = plt.subplots(1, n_panels, figsize=figsize)

        scalar_kw = dict(motor_x=motor_x, motor_y=motor_y, cmap=cmap_scalar)
        self.plot_map("match_rate",   ax=axes[0], **scalar_kw)
        self.plot_map("rms_px",       ax=axes[1], **scalar_kw)
        self.plot_map("rotation_deg", ax=axes[2], **scalar_kw)

        for ii, comp in enumerate(strain_components):
            self.plot_strain_component(
                comp, layer,
                motor_x=motor_x, motor_y=motor_y,
                cmap=cmap_strain, ax=axes[3 + ii],
            )

        fig.suptitle(
            f"LayeredMap — layer {self.layer_labels[layer]}", fontsize=13
        )
        fig.tight_layout()
        return fig, axes

    def plot_ipf_map(
        self,
        layer: int,
        *,
        direction=None,
        symmetry: str = "cubic",
        motor_x: "str | None" = None,
        motor_y: "str | None" = None,
        ax=None,
    ):
        """
        Inverse pole figure colour map for *layer*.

        Args:
            layer: Layer index.
            direction: Reference direction in the sample frame as a length-3
                sequence (default: ``[0, 0, 1]``, the beam/z axis).
            symmetry: Crystal point-group symmetry for reducing orientations to
                the fundamental zone.  One of ``'cubic'``, ``'hexagonal'``,
                ``'tetragonal'``, ``'orthorhombic'``.  Used by orix when
                available; the fallback coloring always uses cubic reduction.
        """
        _dir = np.asarray(direction if direction is not None else [0, 0, 1],
                          dtype=float)
        _dir = _dir / np.linalg.norm(_dir)

        U_map = self.U[layer]                     # (ny, nx, 3, 3)
        valid = ~np.any(np.isnan(U_map), axis=(-2, -1))

        idx = np.argwhere(valid)
        if len(idx) == 0:
            print("No valid orientations to plot.")
            return ax

        U_valid = U_map[idx[:, 0], idx[:, 1]]    # (M, 3, 3)

        # ── Try modern orix API (>= 0.11) ────────────────────────────────────
        rgb = None
        try:
            from orix.quaternion import Rotation as ORotation
            from orix.vector import Vector3d as OVector3d
            from orix.plot import IPFColorKeyTSL
            from orix.crystal_map import Phase

            _sym_map = {
                "cubic":        "m-3m",
                "hexagonal":    "6/mmm",
                "tetragonal":   "4/mmm",
                "orthorhombic": "mmm",
            }
            pg_str = _sym_map.get(symmetry, "m-3m")
            phase  = Phase(point_group=pg_str)

            # scipy [x,y,z,w] → orix [w,x,y,z]
            q_arr  = Rotation.from_matrix(U_valid).as_quat()          # (M,4)
            orot   = ORotation(q_arr[:, [3, 0, 1, 2]])
            ipf_key = IPFColorKeyTSL(phase.point_group,
                                     direction=OVector3d(_dir))
            rgb = np.clip(ipf_key.orientation2color(orot), 0, 1)     # (M,3)
        except Exception:
            pass

        # ── Fallback: pure-scipy coloring (no orix needed) ───────────────────
        if rgb is None:
            rgb = _ipf_color_fallback(U_valid, _dir)

        img = np.ones((self.ny, self.nx, 3))
        img[idx[:, 0], idx[:, 1]] = rgb

        extent, xlabel, ylabel = self._motor_extent(motor_x, motor_y)
        if ax is None:
            _, ax = plt.subplots(figsize=(6, 5))

        imkw = dict(origin="upper")
        if extent is not None:
            imkw.update(extent=extent, aspect="auto")
        ax.imshow(img, **imkw)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"IPF map — {self.layer_labels[layer]}  [{symmetry}]")
        return ax

    def inspect_frame(
        self,
        camera,
        frame_idx: int,
        *,
        h5_path: "str | None" = None,
        h5_dataset: str = "1.1/measurement/eiger4m",
        tiff_dir: "str | None" = None,
        seg_dir: "str | None" = None,
        r_squared_min: float = 0.0,
        include_unfitted: bool = True,
        E_min_eV: float = 5_000,
        E_max_eV: float = 27_000,
        f2_thresh: float = 1e-4,
        top_n_sim: "int | None" = None,
        max_match_dist: float = 5.0,
        use_eff: bool = True,
        figsize: tuple = (14, 6),
    ):
        """
        Display the diffraction image for *frame_idx* overlaid with simulated
        spots from all stack layers and (optionally) the measured segmented spots.

        The fitted orientations stored in ``U_eff`` (or ``U`` when strain is
        unavailable) are temporarily applied to the stack before simulating.

        Args:
            camera: Detector geometry.
            frame_idx: Flat frame index.
            h5_path: Path to the HDF5 scan file (mutually exclusive with
                *tiff_dir*).
            h5_dataset: Dataset path inside *h5_path*.
            tiff_dir: Directory of TIFF frames (mutually exclusive with
                *h5_path*).
            seg_dir: Directory containing ``frame_?????.h5`` segmentation files.
                When provided, the measured peaklist is loaded and overlaid.
            r_squared_min: Minimum Gaussian fit R² to accept a spot from the
                seg file (default ``0.0`` — accept all).
            include_unfitted: Include spots whose Gaussian fit failed
                (default ``True``).
            use_eff: If ``True`` (default), use ``U_eff`` (strained) when
                available; otherwise use ``U``.
        """
        from .simulation import simulate_laue_stack, precompute_allowed_hkl
        from .laue_plotting import plot_measured_vs_simulated

        iy, ix = self.map_index(frame_idx)

        # Load detector image
        image = None
        if h5_path is not None:
            with h5py.File(h5_path, "r") as hf:
                image = hf[h5_dataset][frame_idx].astype(np.float32)
        elif tiff_dir is not None:
            import skimage.io
            path  = os.path.join(tiff_dir, f"frame_{frame_idx:05d}.tif")
            image = skimage.io.imread(path).astype(np.float32)

        # Load measured peaklist from seg file if available
        peaklist = np.empty((0, 9), dtype=np.float32)
        if seg_dir is not None:
            seg_path = os.path.join(seg_dir, f"frame_{frame_idx:05d}.h5")
            if os.path.exists(seg_path):
                try:
                    from .segmentation import convert_spotsfile2peaklist
                    peaklist = convert_spotsfile2peaklist(
                        seg_path,
                        r_squared_min=r_squared_min,
                        include_unfitted=include_unfitted,
                    )
                except Exception as exc:
                    print(f"  ✗ could not load seg file: {exc}", flush=True)
            else:
                print(f"  no seg file for frame {frame_idx}", flush=True)

        # Precompute allowed HKL per unique crystal using the unit-cell |F|²
        # threshold.  This is the same filter used by the fitting functions and
        # correctly removes weak reflections before the simulation.
        allowed_hkl: dict = {}
        for _layer in self.stack.all_layers:
            _cid = id(_layer.crystal)
            if _cid not in allowed_hkl:
                allowed_hkl[_cid] = precompute_allowed_hkl(
                    _layer.crystal, E_max_eV=E_max_eV, f2_thresh=f2_thresh
                )

        # Choose U or U_eff
        U_src = self.U_eff if (use_eff and not np.all(np.isnan(self.U_eff[:, iy, ix]))) \
                else self.U

        # Apply fitted orientations to the stack (restore after)
        saved_U = [l.U.copy() for l in self.stack.all_layers]
        try:
            for li, layer in enumerate(self.stack.all_layers):
                if not np.any(np.isnan(U_src[li, iy, ix])):
                    layer.U = U_src[li, iy, ix].copy()

            spots = simulate_laue_stack(
                self.stack, camera,
                E_min_eV=E_min_eV, E_max_eV=E_max_eV,
                geometry_only=True,
                allowed_hkl=allowed_hkl,
                verbose=False,
            )
            if top_n_sim is not None:
                spots = spots[:top_n_sim]
        finally:
            for layer, U0 in zip(self.stack.all_layers, saved_U):
                layer.U = U0

        fig = plot_measured_vs_simulated(
            peaklist,
            spots,
            image=image,
            camera=camera,
            max_match_dist=max_match_dist,
            figsize=figsize,
        )
        axes = fig.axes
        fig.suptitle(
            f"Frame {frame_idx}  [{iy}, {ix}]  —  LayeredMap",
            fontsize=11,
        )
        return fig, axes

    def reindex_frame(
        self,
        camera,
        base_dir: str,
        *,
        h5_dataset: "str | None" = None,
        tiff_dir: "str | None" = None,
        seg_dir: "str | None" = None,
        map_quantity: str = "match_rate",
        motor_x: "str | None" = None,
        motor_y: "str | None" = None,
        motor_units: "dict | None" = None,
        E_min_eV: float = 5_000.0,
        E_max_eV: float = 27_000.0,
        f2_thresh: float = 1e-4,
        max_match_px: float = 5.0,
        fit_max_match_px: "float | list[float]" = (30.0, 10.0, 3.0),
        fit_strain: "tuple | None" = None,
        r_squared_min: float = 0.0,
        include_unfitted: bool = True,
        kernel_sigma: float = 0.3,
        bg_sigma: float = 251.0,
        max_angle_deg: float = 0.2,
        max_shift_px: "float | list[float] | None" = None,
        correct_depth: bool = False,
        figsize: tuple = (14, 7),
    ) -> None:
        """
        Interactive single-frame fitter and inspector for a
        :class:`~nrxrdct.laue.layers.LayeredCrystal` stack.

        Click a pixel on the left map panel to load the frame, then use the
        buttons to refit and store the result:

        1. **⚡ Fit** — runs :func:`~nrxrdct.laue.fitting.fit_orientation_stack`
           starting from the current map orientation (or the template stack U
           matrices if no result exists for that pixel).  Per-layer simulated
           spots are drawn in distinct colors; green lines connect matched pairs.
        2. **🔩 Fit strain** — runs
           :func:`~nrxrdct.laue.fitting.fit_strain_orientation_stack` from the
           best available orientation.  Enabled after a successful spot-based
           orientation fit.
        3. **🖼 Img orient** — runs
           :func:`~nrxrdct.laue.fitting.refine_orientation_image_stack` on the
           raw pixel image.  Enabled whenever a starting orientation is
           available (existing map result or fit result).
        4. **🖼 Img strain** — runs
           :func:`~nrxrdct.laue.fitting.refine_strain_image_stack` jointly
           refining orientation and strain against the raw image.  The
           ``max_shift_px`` constraint (if set) is forwarded to the optimizer.
        5. **⬆ Store** — writes the best available result back into the map at
           the selected pixel via :meth:`set_result`.  If ``self.save_path`` is
           set the map is also persisted to disk.
        6. **💾 Save UBs** — saves the per-layer U matrices to auto-numbered
           ``.npy`` files in the current directory.

        Args:
            camera: Detector geometry (:class:`~nrxrdct.laue.camera.Camera`).
            base_dir: Processing root directory.  The seg sub-folder defaults to
                ``<base_dir>/seg`` unless *seg_dir* is given explicitly.
            h5_dataset: HDF5 dataset path for the image stack (e.g.
                ``"1.1/measurement/eiger4m"``).  Mutually exclusive with
                *tiff_dir*.
            tiff_dir: Directory of ``img_*.tif`` files.  Mutually exclusive with
                *h5_dataset*.
            seg_dir: Directory containing ``frame_?????.h5`` segmentation files.
                Defaults to ``<base_dir>/seg``.
            map_quantity: Scalar quantity displayed on the left panel.  One of
                ``'match_rate'``, ``'rms_px'``, ``'n_matched'``, ``'score'``,
                ``'cost'``.
            motor_x, motor_y: Motor names for physical axis labels and
                click-to-pixel conversion.
            motor_units: Unit labels per motor, e.g. ``{'sy': 'mm', 'sz': 'mm'}``.
            E_min_eV, E_max_eV: Energy range for spot simulation.
            f2_thresh: Structure-factor threshold for allowed-HKL precomputation.
            max_match_px: Pixel radius used when drawing match lines between
                observed and simulated spots.
            fit_max_match_px: Match-radius schedule passed to
                :func:`~nrxrdct.laue.fitting.fit_orientation_stack`.  A list
                enables staged refinement.
            fit_strain: Strain components to refine (e.g. ``('e_xx', 'e_yy',
                'e_zz')``).  ``None`` refines all six.
            r_squared_min: Minimum Gaussian-fit R² when loading the seg file.
            include_unfitted: Include raw centroids from the seg file.
            kernel_sigma: Gaussian kernel width (px) for image-based refinement.
            bg_sigma: Background subtraction kernel width (px).
            max_angle_deg: Local-search radius (°) for image-based orientation.
            max_shift_px: Per-layer pixel-displacement budget forwarded to
                :func:`~nrxrdct.laue.fitting.refine_strain_image_stack`.
            figsize: Figure size in inches.
        """
        import re as _re
        import ipywidgets as ipw
        from IPython.display import display as _ipy_display
        from .simulation import simulate_laue, precompute_allowed_hkl
        from .segmentation import convert_spotsfile2peaklist
        from .fitting import (
            fit_orientation_stack,
            fit_strain_orientation_stack,
            refine_orientation_image_stack,
            refine_strain_image_stack,
            _match_spots,
            _STRAIN_ALL,
        )

        _seg_dir    = seg_dir or os.path.join(base_dir, "seg")
        _fit_strain = tuple(fit_strain) if fit_strain is not None else _STRAIN_ALL

        # Precompute allowed HKL for each unique crystal in the stack.
        _allowed_by_crystal: dict = {}
        for _layer in self.stack.all_layers:
            _cid = id(_layer.crystal)
            if _cid not in _allowed_by_crystal:
                _allowed_by_crystal[_cid] = precompute_allowed_hkl(
                    _layer.crystal, E_max_eV=E_max_eV, f2_thresh=f2_thresh
                )

        _LAYER_COLORS = [
            "#ff6b35", "#44aaff", "#44dd66", "#ffcc44", "#ff44aa", "#aa44ff",
        ]

        mu = motor_units or {}
        mx = self.motors.get(motor_x) if motor_x else None
        my = self.motors.get(motor_y) if motor_y else None

        if mx is not None and my is not None:
            extent_map = [mx[0, 0], mx[0, -1], my[-1, 0], my[0, 0]]
            xu_ = mu.get(motor_x, "")
            yu_ = mu.get(motor_y, "")
            xlabel_map = f"{motor_x} ({xu_})" if xu_ else motor_x
            ylabel_map = f"{motor_y} ({yu_})" if yu_ else motor_y
        else:
            extent_map = [0, self.nx, self.ny, 0]
            xlabel_map = "column (ix)"
            ylabel_map = "row (iy)"

        def _click_to_iy_ix(xdata: float, ydata: float):
            if mx is not None and my is not None:
                dist = (mx - xdata) ** 2 + (my - ydata) ** 2
                iy, ix = np.unravel_index(int(np.argmin(dist)), dist.shape)
            else:
                ix = int(np.clip(int(xdata), 0, self.nx - 1))
                iy = int(np.clip(int(ydata), 0, self.ny - 1))
            return int(iy), int(ix)

        _tiff_index: list = []

        def _load_image(frame_idx: int) -> "np.ndarray | None":
            if tiff_dir is not None:
                if not _tiff_index:
                    pat = _re.compile(r'(\d+)\.tiff?$', _re.IGNORECASE)
                    files = []
                    for fname in os.listdir(tiff_dir):
                        m = pat.search(fname)
                        if m:
                            files.append((int(m.group(1)), os.path.join(tiff_dir, fname)))
                    files.sort(key=lambda t: t[0])
                    _tiff_index.extend(p for _, p in files)
                if not _tiff_index or frame_idx >= len(_tiff_index):
                    return None
                try:
                    import skimage.io
                    return skimage.io.imread(_tiff_index[frame_idx]).astype(np.float32)
                except Exception as exc:
                    print(f"  ✗ TIFF read error frame {frame_idx}: {exc}", flush=True)
                    return None
            elif h5_dataset is not None:
                _h5 = self.h5_path
                if _h5 is None:
                    print("  ✗ h5_path not set on this LayeredMap.", flush=True)
                    return None
                try:
                    with h5py.File(_h5, "r") as fh:
                        if h5_dataset not in fh:
                            print(f"  ✗ dataset {h5_dataset!r} not in {_h5!r}", flush=True)
                            return None
                        return fh[h5_dataset][frame_idx].astype(np.float32)
                except Exception as exc:
                    print(f"  ✗ HDF5 read error frame {frame_idx}: {exc}", flush=True)
                    return None
            return None

        _map_opts = {
            "match_rate": (self.match_rate,                                      "Match rate", "viridis"),
            "rms_px":     (self.rms_px,                                          "RMS (px)",   "plasma_r"),
            "n_matched":  (np.where(self.n_matched == -1, np.nan,
                                    self.n_matched.astype(float)),               "N matched",  "YlOrRd"),
            "score":      (self.score,                                           "Img score",  "viridis"),
            "cost":       (self.cost,                                            "Cost",       "plasma_r"),
        }
        map_data, map_label, map_cmap = _map_opts.get(
            map_quantity, _map_opts["match_rate"]
        )

        with plt.ioff():
            fig = plt.figure(figsize=figsize)
        try:
            fig.canvas.manager.set_window_title("Laue — LayeredMap re-index")
        except Exception:
            pass

        gs = fig.add_gridspec(
            1, 2, width_ratios=[1, 1.8], wspace=0.08,
            left=0.07, right=0.97, bottom=0.09, top=0.91,
        )
        ax_map = fig.add_subplot(gs[0])
        ax_det = fig.add_subplot(gs[1])

        im_map = ax_map.imshow(
            map_data, origin="upper", extent=extent_map,
            cmap=map_cmap, interpolation="nearest", aspect="auto",
        )
        fig.colorbar(im_map, ax=ax_map, fraction=0.04, pad=0.03,
                     shrink=0.8, label=map_label)
        ax_map.set_xlabel(xlabel_map, fontsize=9)
        ax_map.set_ylabel(ylabel_map, fontsize=9)
        ax_map.set_title(f"Click to select — {map_label}", fontsize=9)
        sel_dot, = ax_map.plot([], [], "w+", ms=11, mew=2.0, zorder=10)

        ax_det.set_facecolor("k")
        ax_det.set_xlim(0, camera.Nh)
        ax_det.set_ylim(camera.Nv, 0)
        ax_det.set_aspect("equal")
        ax_det.set_xlabel("x (px)", fontsize=9)
        ax_det.set_ylabel("y (px)", fontsize=9)
        ax_det.set_title("← click map to select a frame", fontsize=9, color="#888")
        fig.suptitle(
            "LayeredMap re-index  —  click map → Fit → Store",
            fontsize=9, color="#555",
        )

        _state: dict = {
            "frame_idx": None, "iy": None, "ix": None,
            "obs_xy": np.empty((0, 2)),
            "fit_result": None,
            "drawn": False,
        }

        def _best_U_layers() -> "list | None":
            """Return U_layers from the current fit result or map arrays."""
            r = _state["fit_result"]
            iy, ix = _state["iy"], _state["ix"]
            if r is not None:
                return (r.U_eff_layers if hasattr(r, "U_eff_layers") else r.U_layers)
            if iy is not None:
                U_src = (
                    self.U_eff if not np.all(np.isnan(self.U_eff[:, iy, ix]))
                    else self.U
                )
                if not np.all(np.isnan(U_src[:, iy, ix])):
                    return [U_src[li, iy, ix].copy() for li in range(self.n_layers)]
            return None

        def _overlay_sim(U_layers_list: list, obs_xy: np.ndarray) -> None:
            """Draw per-layer simulated spots (colored) with match lines."""
            for li, (layer, U) in enumerate(
                zip(self.stack.all_layers, U_layers_list)
            ):
                if np.any(np.isnan(U)):
                    continue
                color = _LAYER_COLORS[li % len(_LAYER_COLORS)]
                label = getattr(layer, "label", f"layer {li}")
                try:
                    spots = simulate_laue(
                        layer.crystal, U, camera,
                        E_min=E_min_eV, E_max=E_max_eV,
                        geometry_only=True,
                        allowed_hkl=_allowed_by_crystal.get(id(layer.crystal)),
                    )
                    sim_xy = np.array(
                        [s["pix"] for s in spots if s.get("pix") is not None],
                        dtype=float,
                    ) if spots else np.empty((0, 2))
                    if not len(sim_xy):
                        continue
                    ax_det.scatter(
                        sim_xy[:, 0], sim_xy[:, 1],
                        s=24, c=color, marker="D", linewidths=0,
                        zorder=5, alpha=0.85,
                        label=f"{label} ({len(sim_xy)})",
                    )
                    if len(obs_xy) and len(sim_xy):
                        row_ind, col_ind, dist_px = _match_spots(
                            obs_xy, sim_xy, max_match_px
                        )
                        for r, c, d in zip(row_ind, col_ind, dist_px):
                            if d < max_match_px:
                                ax_det.plot(
                                    [obs_xy[r, 0], sim_xy[c, 0]],
                                    [obs_xy[r, 1], sim_xy[c, 1]],
                                    color="#44dd66", lw=0.7, alpha=0.55, zorder=3,
                                )
                except Exception as exc:
                    print(f"  layer {li} sim error: {exc}", flush=True)

        def _draw_det() -> None:
            frame_idx  = _state["frame_idx"]
            obs_xy     = _state["obs_xy"]
            fit_result = _state["fit_result"]
            iy, ix     = _state["iy"], _state["ix"]

            saved_xlim = ax_det.get_xlim()
            saved_ylim = ax_det.get_ylim()
            ax_det.cla()
            ax_det.set_facecolor("k")

            image = _load_image(frame_idx)
            if image is not None:
                pos  = image[image > 0]
                vmax = float(np.percentile(pos, 99)) if pos.size else 1.0
                nv_im, nh_im = image.shape
                ax_det.imshow(
                    np.log1p(image / vmax * 1000),
                    origin="upper", extent=[0, nh_im, nv_im, 0],
                    cmap="gray", aspect="equal", zorder=0,
                )
            else:
                ax_det.set_xlim(0, camera.Nh)
                ax_det.set_ylim(camera.Nv, 0)
            ax_det.set_aspect("equal")

            if len(obs_xy):
                ax_det.scatter(
                    obs_xy[:, 0], obs_xy[:, 1],
                    s=40, c="none", edgecolors="white", linewidths=0.8,
                    zorder=4, label=f"observed ({len(obs_xy)})",
                )

            U_layers_draw = _best_U_layers()
            if U_layers_draw is not None:
                _overlay_sim(U_layers_draw, obs_xy)

            if len(obs_xy) or U_layers_draw is not None:
                ax_det.legend(
                    fontsize=7, loc="upper right",
                    facecolor="#111", edgecolor="#444", labelcolor="white",
                    framealpha=0.85,
                )

            ax_det.set_xlabel("x (px)", fontsize=9)
            ax_det.set_ylabel("y (px)", fontsize=9)
            suffix = f"  — {fit_result}" if fit_result is not None else ""
            ax_det.set_title(
                f"Frame {frame_idx}  (iy={iy}, ix={ix}){suffix}",
                fontsize=8,
            )
            if _state["drawn"]:
                ax_det.set_xlim(saved_xlim)
                ax_det.set_ylim(saved_ylim)
            _state["drawn"] = True
            fig.canvas.draw_idle()

        def _on_click(event) -> None:
            if event.inaxes is not ax_map:
                return
            if event.xdata is None or event.ydata is None:
                return
            try:
                if fig.canvas.toolbar.mode != "":
                    return
            except Exception:
                pass

            iy, ix    = _click_to_iy_ix(event.xdata, event.ydata)
            frame_idx = self.frame_index(iy, ix)

            if mx is not None and my is not None:
                sel_dot.set_data([mx[iy, ix]], [my[iy, ix]])
            else:
                sel_dot.set_data([ix + 0.5], [iy + 0.5])

            seg_path = os.path.join(_seg_dir, f"frame_{frame_idx:05d}.h5")
            if os.path.exists(seg_path):
                try:
                    obs_xy = convert_spotsfile2peaklist(
                        seg_path,
                        r_squared_min=r_squared_min,
                        include_unfitted=include_unfitted,
                    )[:, :2]
                except Exception:
                    obs_xy = np.empty((0, 2))
            else:
                obs_xy = np.empty((0, 2))

            has_map_result = not np.all(np.isnan(self.U[:, iy, ix, 0, 0]))
            has_image      = (h5_dataset is not None or tiff_dir is not None)
            enough_obs     = len(obs_xy) >= 3

            _state.update(
                frame_idx=frame_idx, iy=iy, ix=ix,
                obs_xy=obs_xy, fit_result=None,
            )

            _info.value = (
                f"<span style='color:#aaa'>Frame {frame_idx} — "
                f"{len(obs_xy)} observed spots"
                + (" — existing map result" if has_map_result else "")
                + "</span>"
            )
            btn_fit_ori.disabled = not (enough_obs and (has_map_result or True))
            btn_fit_str.disabled = True
            btn_img_ori.disabled = not (has_image and (has_map_result or enough_obs))
            btn_img_str.disabled = not (has_image and (has_map_result or enough_obs))
            btn_store.disabled   = True
            btn_save.disabled    = True
            _draw_det()

        fig.canvas.mpl_connect("button_press_event", _on_click)

        # ── widgets ────────────────────────────────────────────────────────────
        _bkw  = dict(layout=ipw.Layout(width="130px", height="32px"))
        _bkw2 = dict(layout=ipw.Layout(width="140px", height="32px"))

        btn_fit_ori = ipw.Button(description="⚡ Fit",       button_style="primary", **_bkw)
        btn_fit_str = ipw.Button(description="🔩 Fit strain", button_style="warning", **_bkw2)
        btn_img_ori = ipw.Button(description="🖼 Img orient", button_style="info",    **_bkw2)
        btn_img_str = ipw.Button(description="🖼 Img strain", button_style="warning", **_bkw2)
        btn_store   = ipw.Button(description="⬆ Store",      button_style="success", **_bkw)
        btn_save    = ipw.Button(description="💾 Save UBs",  button_style="",        **_bkw)

        for _b in (btn_fit_ori, btn_fit_str, btn_img_ori, btn_img_str,
                   btn_store, btn_save):
            _b.disabled = True

        _info = ipw.HTML(
            "<span style='color:#666;font-style:italic'>"
            "click a map pixel to select a frame"
            "</span>",
            layout=ipw.Layout(margin="4px 0 0 6px"),
        )

        # ── async-threaded button callbacks ───────────────────────────────────

        def _async_run(worker_fn, btn, label_running, label_done, on_success):
            import asyncio
            import queue as _qmod
            import threading

            if getattr(btn, "_running", False):
                return
            btn._running    = True
            btn.disabled    = True
            btn.description = label_running

            q: _qmod.Queue = _qmod.Queue()

            def _run():
                try:
                    q.put(("ok", worker_fn()))
                except Exception as exc:
                    q.put(("err", exc))

            async def _wait():
                threading.Thread(target=_run, daemon=True).start()
                while q.empty():
                    await asyncio.sleep(0.15)
                tag, payload = q.get_nowait()
                if tag == "err":
                    _info.value = (
                        f"<b style='color:#f44'>{label_done} error: {payload}</b>"
                    )
                else:
                    on_success(payload)
                btn.description = label_done
                btn.disabled    = False
                btn._running    = False

            try:
                asyncio.get_event_loop().create_task(_wait())
            except RuntimeError:
                asyncio.ensure_future(_wait())

        def _cb_fit_ori(_) -> None:
            obs_xy = _state["obs_xy"]
            if len(obs_xy) < 3:
                return

            start_U = _best_U_layers() or [l.U.copy() for l in self.stack.all_layers]
            saved_U = [l.U.copy() for l in self.stack.all_layers]
            for l, U in zip(self.stack.all_layers, start_U):
                l.U = U.copy()

            def _work():
                try:
                    return fit_orientation_stack(
                        self.stack, camera, obs_xy,
                        E_min_eV=E_min_eV, E_max_eV=E_max_eV,
                        max_match_px=list(fit_max_match_px)
                        if hasattr(fit_max_match_px, "__iter__")
                        else fit_max_match_px,
                        allowed_hkl=_allowed_by_crystal,
                        geometry_only=True,
                        correct_depth=correct_depth,
                        update_stack=False,
                        verbose=True,
                    )
                finally:
                    for l, U in zip(self.stack.all_layers, saved_U):
                        l.U = U

            def _on_ok(res):
                _state["fit_result"] = res
                col = "#44aaff" if res.success else "#ffaa33"
                _info.value = f"<b style='color:{col}'>{res}</b>"
                btn_fit_str.disabled = len(_state["obs_xy"]) < 3
                btn_img_str.disabled = False
                btn_store.disabled   = False
                btn_save.disabled    = False
                _draw_det()

            _async_run(_work, btn_fit_ori, "Fitting…", "⚡ Fit", _on_ok)

        def _cb_fit_str(_) -> None:
            obs_xy = _state["obs_xy"]
            if len(obs_xy) < 3:
                return

            start_U = _best_U_layers() or [l.U.copy() for l in self.stack.all_layers]
            saved_U = [l.U.copy() for l in self.stack.all_layers]
            for l, U in zip(self.stack.all_layers, start_U):
                l.U = U.copy()

            def _work():
                try:
                    return fit_strain_orientation_stack(
                        self.stack, camera, obs_xy,
                        fit_strain=_fit_strain,
                        E_min_eV=E_min_eV, E_max_eV=E_max_eV,
                        max_match_px=[3.0],
                        allowed_hkl=_allowed_by_crystal,
                        geometry_only=True,
                        correct_depth=correct_depth,
                        update_stack=False,
                        verbose=True,
                    )
                finally:
                    for l, U in zip(self.stack.all_layers, saved_U):
                        l.U = U

            def _on_ok(res):
                _state["fit_result"] = res
                col = "#ffcc44" if res.success else "#ffaa33"
                _info.value = f"<b style='color:{col}'>{res}</b>"
                btn_img_str.disabled = False
                btn_store.disabled   = False
                btn_save.disabled    = False
                _draw_det()

            _async_run(_work, btn_fit_str, "Fitting…", "🔩 Fit strain", _on_ok)

        def _cb_img_ori(_) -> None:
            frame_idx = _state["frame_idx"]
            if frame_idx is None:
                return

            start_U = _best_U_layers() or [l.U.copy() for l in self.stack.all_layers]
            if any(np.any(np.isnan(U)) for U in start_U):
                _info.value = "<b style='color:#f44'>No starting orientation available</b>"
                return

            saved_U = [l.U.copy() for l in self.stack.all_layers]
            for l, U in zip(self.stack.all_layers, start_U):
                l.U = U.copy()

            image = _load_image(frame_idx)
            if image is None:
                _info.value = "<b style='color:#f44'>Could not load image</b>"
                return

            def _work():
                try:
                    return refine_orientation_image_stack(
                        self.stack, camera, image.astype(np.float64),
                        allowed_hkl=_allowed_by_crystal,
                        kernel_sigma=kernel_sigma,
                        bg_sigma=bg_sigma,
                        max_angle_deg=max_angle_deg,
                        E_min=E_min_eV, E_max=E_max_eV,
                        correct_depth=correct_depth,
                        verbose=True,
                    )
                finally:
                    for l, U in zip(self.stack.all_layers, saved_U):
                        l.U = U

            def _on_ok(res):
                _state["fit_result"] = res
                _info.value = f"<b style='color:#44aaff'>{res}</b>"
                btn_img_str.disabled = False
                btn_store.disabled   = False
                btn_save.disabled    = False
                _draw_det()

            _async_run(_work, btn_img_ori, "Refining…", "🖼 Img orient", _on_ok)

        def _cb_img_str(_) -> None:
            frame_idx = _state["frame_idx"]
            if frame_idx is None:
                return

            start_U = _best_U_layers() or [l.U.copy() for l in self.stack.all_layers]
            if any(np.any(np.isnan(U)) for U in start_U):
                _info.value = "<b style='color:#f44'>No starting orientation available</b>"
                return

            r = _state["fit_result"]
            strain0_list = (
                r.strain_tensors if hasattr(r, "strain_tensors") else None
            )

            saved_U = [l.U.copy() for l in self.stack.all_layers]
            for l, U in zip(self.stack.all_layers, start_U):
                l.U = U.copy()

            image = _load_image(frame_idx)
            if image is None:
                _info.value = "<b style='color:#f44'>Could not load image</b>"
                return

            def _work():
                try:
                    kw: dict = dict(
                        allowed_hkl   = _allowed_by_crystal,
                        fit_strain    = _fit_strain,
                        kernel_sigma  = kernel_sigma,
                        bg_sigma      = bg_sigma,
                        max_angle_deg = max_angle_deg,
                        E_min         = E_min_eV,
                        E_max         = E_max_eV,
                        correct_depth = correct_depth,
                        verbose       = True,
                    )
                    if strain0_list is not None:
                        kw["strain0_list"] = strain0_list
                    if max_shift_px is not None:
                        kw["max_shift_px"] = max_shift_px
                    return refine_strain_image_stack(
                        self.stack, camera, image.astype(np.float64), **kw
                    )
                finally:
                    for l, U in zip(self.stack.all_layers, saved_U):
                        l.U = U

            def _on_ok(res):
                _state["fit_result"] = res
                _info.value = f"<b style='color:#ffcc44'>{res}</b>"
                btn_store.disabled = False
                btn_save.disabled  = False
                _draw_det()

            _async_run(_work, btn_img_str, "Refining…", "🖼 Img strain", _on_ok)

        def _cb_store(_) -> None:
            r = _state["fit_result"]
            if r is None:
                return
            iy, ix = _state["iy"], _state["ix"]
            self.set_result(iy, ix, r)
            save_note = ""
            if self.save_path:
                self.save(self.save_path)
                save_note = f" — saved to {os.path.basename(self.save_path)}"
            else:
                save_note = " — <b style='color:#ffaa33'>no save_path, result in memory only</b>"
            _info.value = (
                f"<b style='color:#44dd66'>Stored → (iy={iy}, ix={ix})</b>"
                f"&emsp;{r}"
                f"<span style='color:#aaa'>{save_note}</span>"
            )
            print(
                f"  ⬆ Stored result at (iy={iy}, ix={ix})"
                + (f"  → {self.save_path}" if self.save_path else " (in memory only)"),
                flush=True,
            )

        def _cb_save(_) -> None:
            r = _state["fit_result"]
            if r is None:
                return
            U_layers = (
                r.U_eff_layers if hasattr(r, "U_eff_layers") else r.U_layers
            )
            saved = []
            for li, U in enumerate(U_layers):
                existing = glob.glob(os.path.join(os.getcwd(), f"UB_layer{li}_*.npy"))
                max_n = -1
                for fpath in existing:
                    m = _re.search(r"UB_layer\d+_(\d+)\.npy$", os.path.basename(fpath))
                    if m:
                        max_n = max(max_n, int(m.group(1)))
                fname = f"UB_layer{li}_{max_n + 1:02d}.npy"
                np.save(fname, U)
                saved.append(fname)
                print(f"  💾 Saved U layer {li} → {os.path.abspath(fname)}")
            _info.value = (
                f"<b style='color:#44dd66'>Saved {len(saved)} UB files: "
                + ", ".join(saved) + "</b>"
            )

        btn_fit_ori.on_click(_cb_fit_ori)
        btn_fit_str.on_click(_cb_fit_str)
        btn_img_ori.on_click(_cb_img_ori)
        btn_img_str.on_click(_cb_img_str)
        btn_store.on_click(_cb_store)
        btn_save.on_click(_cb_save)

        _controls = ipw.VBox([
            ipw.HBox(
                [btn_fit_ori, btn_fit_str],
                layout=ipw.Layout(gap="6px", margin="4px 0 0 0", align_items="center"),
            ),
            ipw.HBox(
                [btn_img_ori, btn_img_str],
                layout=ipw.Layout(gap="6px", margin="2px 0 0 0", align_items="center"),
            ),
            ipw.HBox(
                [btn_store, btn_save],
                layout=ipw.Layout(gap="6px", margin="2px 0 0 0", align_items="center"),
            ),
            _info,
        ], layout=ipw.Layout(padding="6px 8px"))

        _ipy_display(ipw.VBox([fig.canvas, _controls]))

    # ── Serialization ──────────────────────────────────────────────────────────

    def save(self, path: str, compress: bool = True) -> None:
        """
        Save the map to an HDF5 file.

        All per-pixel arrays are stored under their attribute names.
        Motor positions land in ``motors/<name>``.
        """
        kw = {"compression": "gzip", "compression_opts": 4} if compress else {}

        with h5py.File(path, "w") as hf:
            meta = hf.create_group("meta")
            meta.attrs["ny"]       = self.ny
            meta.attrs["nx"]       = self.nx
            meta.attrs["n_layers"] = self.n_layers
            meta.attrs["h5_path"]  = self.h5_path or ""
            meta.attrs["entry"]    = self.entry
            meta.attrs["labels"]   = json.dumps(self.layer_labels)

            for name in (
                "R_global", "rotvec",
                "U", "U_eff",
                "rms_px", "mean_px", "n_matched", "match_rate",
                "cost", "n_sim", "score", "score0",
                "strain_tensor", "strain_voigt", "strain_tensor_deviatoric",
            ):
                hf.create_dataset(name, data=getattr(self, name), **kw)

            for name, arr in self.motors.items():
                hf.create_dataset(f"motors/{name}", data=arr, **kw)

        print(f"LayeredMap saved → {path!r}", flush=True)

    @classmethod
    def load(cls, path: str, stack) -> "LayeredMap":
        """
        Load a :class:`LayeredMap` from *path*.

        *stack* must be the same
        :class:`~nrxrdct.laue.layers.LayeredCrystal` used during fitting
        (supplies the layer crystal objects).

        Returns:
            LayeredMap with all per-pixel arrays restored.
        """
        with h5py.File(path, "r") as hf:
            meta     = hf["meta"]
            ny       = int(meta.attrs["ny"])
            nx       = int(meta.attrs["nx"])
            h5_path  = str(meta.attrs.get("h5_path", "")) or None
            entry    = str(meta.attrs.get("entry", "1.1"))

            obj = cls(ny=ny, nx=nx, stack=stack, h5_path=h5_path, entry=entry)

            for name in (
                "R_global", "rotvec",
                "U", "U_eff",
                "rms_px", "mean_px", "n_matched", "match_rate",
                "cost", "n_sim", "score", "score0",
                "strain_tensor", "strain_voigt", "strain_tensor_deviatoric",
            ):
                if name in hf:
                    setattr(obj, name, hf[name][()])

            if "motors" in hf:
                for name in hf["motors"]:
                    obj.motors[name] = hf[f"motors/{name}"][()]

        return obj

    # ── Disorientation ────────────────────────────────────────────────────────

    def disorientation_map(
        self,
        layer_a: int,
        layer_b: int,
        *,
        symmetry: str = "cubic",
        symmetry_b: "str | None" = None,
        use_eff: bool = False,
        motor_x: "str | None" = None,
        motor_y: "str | None" = None,
        cmap: str = "inferno",
        vmin: float = 0.0,
        vmax: "float | None" = None,
        plot: bool = True,
        ax=None,
    ) -> np.ndarray:
        """
        Compute the per-pixel disorientation angle between two stack layers.

        The misorientation between the two layers is minimised over all pairs
        of point-group symmetry operators (the same algorithm as
        :func:`~nrxrdct.laue.simulation.disorientation`) and the result is
        the **fundamental zone disorientation angle** — the smallest rotation
        that maps one layer's orientation to the other's, accounting for
        crystal symmetry.

        Both orientation matrices are first projected onto SO(3) (via SVD),
        so ``U_eff`` matrices (which carry a small strain factor) are handled
        correctly.

        Args:
            layer_a: Index of the first layer (e.g. the substrate/buffer).
            layer_b: Index of the second layer (e.g. the film).
            symmetry: Point-group symmetry for *layer_a* (and *layer_b* if
                *symmetry_b* is ``None``).  One of ``'cubic'``, ``'hexagonal'``,
                ``'tetragonal'``, ``'orthorhombic'``.
            symmetry_b: Point-group symmetry for *layer_b*.  If ``None``
                (default), *symmetry* is used for both layers.  Set a
                different value for heterostructures with different crystal
                systems (e.g. cubic substrate + hexagonal film).
            use_eff: If ``True``, use ``U_eff`` (strained orientation) instead
                of the pure-rotation ``U``.  ``U_eff`` is only available after
                a strain fit; falls back silently to ``U`` if not set.
            motor_x, motor_y: Motor names for physical axis labels on the plot.
            cmap: Matplotlib colormap for the angle map.  Default ``'inferno'``.
            vmin: Minimum value for the colorscale (degrees).  Default ``0``.
            vmax: Maximum value for the colorscale.  ``None`` uses the 99th
                percentile of the valid pixel values.
            plot: If ``True`` (default), draw the map.  Set ``False`` to only
                compute and return the array.
            ax: Existing matplotlib ``Axes`` to draw into.  A new figure is
                created if ``None``.

        Returns:
            angle_map ((ny, nx) float64 ndarray): Disorientation angle in
                degrees at each pixel.  Pixels where either layer has no valid
                orientation are ``NaN``.
        """
        from .simulation import _symmetry_ops_np

        _sym_b = symmetry_b if symmetry_b is not None else symmetry

        # Choose U or U_eff
        if use_eff and not np.all(np.isnan(self.U_eff)):
            U_src = self.U_eff
        else:
            U_src = self.U

        Ua = U_src[layer_a]   # (ny, nx, 3, 3)
        Ub = U_src[layer_b]   # (ny, nx, 3, 3)

        # Valid mask: both layers must have a fitted orientation
        valid = (
            ~np.any(np.isnan(Ua), axis=(-2, -1)) &
            ~np.any(np.isnan(Ub), axis=(-2, -1))
        )

        ny, nx = self.ny, self.nx
        angle_map = np.full((ny, nx), np.nan)

        if not valid.any():
            if plot:
                if ax is None:
                    _, ax = plt.subplots(figsize=(7, 6))
                ax.set_title(
                    f"Disorientation — layers {self.layer_labels[layer_a]} / "
                    f"{self.layer_labels[layer_b]} (no data)"
                )
            return angle_map

        # Flatten to (M, 3, 3) for valid pixels only
        idx   = np.argwhere(valid)           # (M, 2)
        iy_v  = idx[:, 0]
        ix_v  = idx[:, 1]
        Ua_v  = Ua[iy_v, ix_v]              # (M, 3, 3)
        Ub_v  = Ub[iy_v, ix_v]              # (M, 3, 3)

        # Project to SO(3) — handles strained U_eff correctly
        Ra = Rotation.from_matrix(Ua_v).as_matrix()   # (M, 3, 3)
        Rb = Rotation.from_matrix(Ub_v).as_matrix()   # (M, 3, 3)

        # Raw misorientation per pixel: R_mis = Rb @ Ra.T
        R_mis = Rb @ Ra.transpose(0, 2, 1)            # (M, 3, 3)

        # Symmetry operators for both layers
        ops_a = _symmetry_ops_np(symmetry)             # (Na, 3, 3)
        ops_b = _symmetry_ops_np(_sym_b)               # (Nb, 3, 3)

        # Apply operators: S_a @ R_mis @ S_b.T for all (S_a, S_b) pairs.
        #   ops_a_R[s, m] = ops_a[s] @ R_mis[m]  →  (Na, M, 3, 3)
        ops_a_R = np.einsum("sij,mjk->smik", ops_a, R_mis)

        #   candidates[sa, sb, m] = ops_a_R[sa, m] @ ops_b[sb].T
        #   broadcast: ops_a_R[:, None]  (Na, 1,  M, 3, 3)
        #            @ ops_b.T[None, :, None]  (1, Nb, 1, 3, 3)
        ops_b_T = ops_b.transpose(0, 2, 1)            # (Nb, 3, 3)
        candidates = (
            ops_a_R[:, None] @ ops_b_T[None, :, None]
        )                                              # (Na, Nb, M, 3, 3)

        # Trace → angle for every candidate
        traces = (
            candidates[:, :, :, 0, 0] +
            candidates[:, :, :, 1, 1] +
            candidates[:, :, :, 2, 2]
        )                                              # (Na, Nb, M)

        # Flatten symmetry pairs → (Na*Nb, M), find minimum angle per pixel
        traces_flat  = traces.reshape(-1, len(iy_v))   # (Na*Nb, M)
        cos_half_ang = np.clip((traces_flat - 1.0) / 2.0, -1.0, 1.0)
        angles       = np.degrees(np.arccos(cos_half_ang))   # (Na*Nb, M)
        min_angles   = angles.min(axis=0)              # (M,)

        angle_map[iy_v, ix_v] = min_angles

        if not plot:
            return angle_map

        # ── Plot ──────────────────────────────────────────────────────────────
        if vmax is None:
            valid_vals = min_angles[np.isfinite(min_angles)]
            vmax = float(np.percentile(valid_vals, 99)) if len(valid_vals) else 1.0

        extent, xlabel, ylabel = self._motor_extent(motor_x, motor_y)
        if ax is None:
            _, ax = plt.subplots(figsize=(7, 6))

        imkw = dict(origin="upper", cmap=cmap, vmin=vmin, vmax=vmax)
        if extent is not None:
            imkw.update(extent=extent, aspect="auto")

        im = ax.imshow(angle_map, **imkw)
        plt.colorbar(im, ax=ax, label="Disorientation (°)")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(
            f"Disorientation — {self.layer_labels[layer_a]}"
            f" / {self.layer_labels[layer_b]}"
        )
        return angle_map

    def plot_kam(
        self,
        layer: int,
        *,
        order: int = 1,
        symmetry: str = "cubic",
        use_eff: bool = False,
        motor_x: "str | None" = None,
        motor_y: "str | None" = None,
        cmap: str = "inferno",
        vmin: float = 0.0,
        vmax: "float | None" = None,
        plot: bool = True,
        ax=None,
    ) -> np.ndarray:
        """
        Kernel Average Misorientation (KAM) map for *layer*.

        For each valid pixel the mean disorientation angle to all valid
        neighbors within Chebyshev distance *order* is computed after
        minimisation over crystal symmetry operators.  The result is the
        standard KAM metric used in EBSD analysis to highlight orientation
        gradients caused by geometrically necessary dislocations or grain
        sub-structure.

        Args:
            layer: Index into ``stack.all_layers`` selecting which layer's
                orientation to analyse.  Each layer has its own independent
                ``U`` / ``U_eff`` array, so the substrate and film will
                produce separate KAM maps reflecting gradients in their
                respective materials.
            order: Chebyshev shell radius in pixels.  ``1`` (default)
                uses the 8 immediate neighbors; ``2`` adds the next ring
                for 24 neighbors total; and so on.  Larger values smooth
                the map and capture longer-range gradients but blur sharp
                boundaries.  Standard EBSD convention is ``order=1``.
            symmetry: Point-group symmetry applied when minimising the
                misorientation angle between neighbors.  Must match the
                crystal structure of *layer*: ``'cubic'``,
                ``'hexagonal'``, ``'tetragonal'``, or
                ``'orthorhombic'``.  Using the wrong symmetry
                (e.g. ``'orthorhombic'`` for a cubic crystal) causes
                symmetry-equivalent orientations to be reported as real
                gradients, inflating KAM everywhere.
            use_eff: If ``True``, use ``U_eff`` (which carries the strain
                factor ``I + ε``) instead of the pure rotation ``U``.
                Before computing misorientations the matrices are projected
                onto SO(3), so large inter-pixel strain differences can
                still bleed into the KAM value.  Keep ``False`` (default)
                to isolate orientation gradients from strain.
            motor_x, motor_y: Names of motor position arrays stored in
                ``self.motors``.  When supplied the axes are labeled in
                physical units and the map extent is set from the motor
                ranges; without them row/column pixel indices are used.
            cmap: Matplotlib colormap.  ``'inferno'`` (default) is the
                standard choice for KAM because near-zero values appear
                black and gradient regions become progressively brighter.
            vmin: Lower colorscale bound in degrees.  Almost always left
                at ``0.0`` since KAM is non-negative by construction.
            vmax: Upper colorscale bound in degrees.  Set an explicit
                value when comparing maps across different scans so all
                images share the same scale.  ``None`` (default)
                auto-scales to the 99th percentile of valid pixels,
                which avoids isolated outlier pixels blowing out the
                colour range.
            plot: Render the map when ``True`` (default).  Set ``False``
                to compute and return the array without any drawing, e.g.
                for batch processing or when embedding in a custom figure.
            ax: Existing ``Axes`` to draw into.  A new figure is created
                when ``None``.

        Returns:
            ``(ny, nx)`` float64 array of KAM angles in degrees.  Pixels
            with no valid orientation or no valid neighbors are ``NaN``.
        """
        from .simulation import _symmetry_ops_np

        # Choose U source
        if use_eff and not np.all(np.isnan(self.U_eff[layer])):
            U_src = self.U_eff[layer]   # (ny, nx, 3, 3)
        else:
            U_src = self.U[layer]

        valid = ~np.any(np.isnan(U_src), axis=(-2, -1))   # (ny, nx) bool

        ny, nx = self.ny, self.nx
        kam_map = np.full((ny, nx), np.nan)

        if not valid.any():
            if plot:
                if ax is None:
                    _, ax = plt.subplots(figsize=(7, 6))
                ax.set_title(f"KAM — {self.layer_labels[layer]} (no data)")
            return kam_map

        # Project all valid orientations to SO(3) once (handles U_eff correctly).
        U_rot = np.full((ny, nx, 3, 3), np.nan)
        U_rot[valid] = Rotation.from_matrix(U_src[valid]).as_matrix()

        ops   = _symmetry_ops_np(symmetry)   # (N_sym, 3, 3)

        # Accumulate per-pixel misorientation sums and neighbor counts.
        angle_sum = np.zeros((ny, nx), dtype=np.float64)
        count     = np.zeros((ny, nx), dtype=np.int32)

        # All neighbor offsets with 1 ≤ Chebyshev distance ≤ order.
        offsets = [
            (diy, dix)
            for diy in range(-order, order + 1)
            for dix in range(-order, order + 1)
            if 1 <= max(abs(diy), abs(dix)) <= order
        ]

        for diy, dix in offsets:
            # Source pixel range: rows/cols that have a valid neighbor at (diy, dix).
            iy0_s = max(0, -diy);  iy1_s = min(ny, ny - diy)
            ix0_s = max(0, -dix);  ix1_s = min(nx, nx - dix)

            IY_s, IX_s = np.meshgrid(
                np.arange(iy0_s, iy1_s),
                np.arange(ix0_s, ix1_s),
                indexing="ij",
            )
            IY_n = IY_s + diy
            IX_n = IX_s + dix

            mask = valid[IY_s, IX_s] & valid[IY_n, IX_n]
            if not mask.any():
                continue

            iy_s = IY_s[mask]; ix_s = IX_s[mask]
            iy_n = IY_n[mask]; ix_n = IX_n[mask]

            Ra = U_rot[iy_s, ix_s]                        # (M, 3, 3)
            Rb = U_rot[iy_n, ix_n]                        # (M, 3, 3)

            # Misorientation: R_mis[m] = Rb[m] @ Ra[m].T
            R_mis = Rb @ Ra.transpose(0, 2, 1)            # (M, 3, 3)

            # ops[s] @ R_mis[m] → (N_sym, M, 3, 3)
            ops_R = np.einsum("sij,mjk->smik", ops, R_mis)

            # trace(ops_R[s,m] @ ops[t].T) for all (s, t, m)
            # = einsum smij, tji -> stm
            traces = np.einsum("smij,tji->stm", ops_R, ops)   # (N_sym, N_sym, M)

            # Minimum angle over all symmetry-equivalent pairs
            cos_ang    = np.clip((traces - 1.0) / 2.0, -1.0, 1.0)
            angles_sym = np.degrees(np.arccos(cos_ang))        # (N_sym, N_sym, M)
            min_ang    = angles_sym.reshape(-1, len(iy_s)).min(axis=0)  # (M,)

            np.add.at(angle_sum, (iy_s, ix_s), min_ang)
            np.add.at(count,     (iy_s, ix_s), 1)

        has_nbrs = count > 0
        kam_map[has_nbrs] = angle_sum[has_nbrs] / count[has_nbrs]

        if not plot:
            return kam_map

        # ── Plot ──────────────────────────────────────────────────────────────
        valid_vals = kam_map[np.isfinite(kam_map)]
        if vmax is None:
            vmax = float(np.percentile(valid_vals, 99)) if len(valid_vals) else 1.0
        vmax = max(float(vmax), float(vmin) + 1e-6)   # prevent blank image when all angles ≈ 0

        extent, xlabel, ylabel = self._motor_extent(motor_x, motor_y)
        if ax is None:
            _, ax = plt.subplots(figsize=(7, 6))

        imkw = dict(origin="upper", cmap=cmap, vmin=vmin, vmax=vmax)
        if extent is not None:
            imkw.update(extent=extent, aspect="auto")

        im = ax.imshow(kam_map, **imkw)
        plt.colorbar(im, ax=ax, label="KAM (°)")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(
            f"KAM — {self.layer_labels[layer]}  "
            f"(order={order}, {symmetry})"
        )
        return kam_map

    # ── SLURM submission ───────────────────────────────────────────────────────

    @staticmethod
    def _camera_to_dict(camera) -> dict:
        return {
            "dd":           float(camera.dd),
            "xcen":         float(camera.xcen),
            "ycen":         float(camera.ycen),
            "xbet":         float(camera.xbet),
            "xgam":         float(camera.xgam),
            "pixelsize":    float(camera.pixel_mm),
            "n_pix_h":      int(camera.Nh),
            "n_pix_v":      int(camera.Nv),
            "kf_direction": str(camera.kf_direction),
        }

    def _setup_slurm_dirs(self, base_dir: str, sub: str) -> dict:
        dirs = {
            "out":       os.path.join(base_dir, sub),
            "job_meta":  os.path.join(base_dir, "job_meta"),
            "slurm_logs": os.path.join(base_dir, "slurm_logs"),
        }
        for d in dirs.values():
            os.makedirs(d, exist_ok=True)
        return dirs

    def _write_stack_pkl(self, base_dir: str) -> str:
        path = os.path.join(base_dir, "job_meta", "stack.pkl")
        with open(path, "wb") as fh:
            _dill.dump(self.stack, fh)
        return path

    def _submit_jobs(
        self,
        job_name: str,
        worker_module: str,
        meta_json_path: str,
        chunks: list,
        slurm_logs_dir: str,
        *,
        partition: str = "all",
        time: str = "01:00:00",
        mem: str = "4G",
        cpus_per_task: int = 1,
        python_bin: str = "python",
        extra_sbatch: "dict | None" = None,
    ) -> list:
        job_ids = []
        for i, chunk in enumerate(chunks):
            indices_str = ",".join(str(fi) for fi in chunk)
            wrap_cmd = (
                f"{python_bin} -m {worker_module} "
                f"--meta-json {meta_json_path} "
                f"--frame-indices {indices_str}"
            )
            sbatch_args = [
                "sbatch",
                f"--job-name={job_name}_{i:04d}",
                f"--partition={partition}",
                f"--time={time}",
                f"--mem={mem}",
                f"--cpus-per-task={cpus_per_task}",
                f"--output={os.path.join(slurm_logs_dir, f'{job_name}_{i:04d}_%j.out')}",
                f"--error={os.path.join(slurm_logs_dir, f'{job_name}_{i:04d}_%j.err')}",
            ]
            if extra_sbatch:
                for k, v in extra_sbatch.items():
                    sbatch_args.append(f"--{k}={v}")
            sbatch_args += ["--wrap", wrap_cmd]

            result = subprocess.run(sbatch_args, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(
                    f"sbatch failed for chunk {i}: {result.stderr.strip()}"
                )
            job_id = result.stdout.strip().split()[-1]
            job_ids.append(job_id)
            print(f"  submitted {job_name}_{i:04d} → job {job_id}", flush=True)

        return job_ids

    # ── Simulation-guided segmentation ────────────────────────────────────────

    def submit_guided_segmentation(
        self,
        base_dir: str,
        camera,
        h5_dataset: str,
        mask_path: str,
        *,
        n_jobs: int = 10,
        partition: str = "all",
        time: str = "02:00:00",
        mem: str = "8G",
        cpus_per_task: int = 1,
        python_bin: str = "python",
        f2_thresh: float = 1e-4,
        correct_depth: bool = False,
        overwrite: bool = False,
        extra_sbatch: "dict | None" = None,
        **seg_kwargs,
    ) -> list:
        """
        Submit simulation-guided segmentation jobs to SLURM.

        The stack's **current U matrices** (set interactively or from a
        previous fit) are serialised to a pickle and shipped to each job.
        Every worker simulates Laue spots once with that fixed orientation
        and uses the predicted positions to drive
        :func:`~nrxrdct.laue.segmentation.simulation_guided_segmentation`
        on its assigned frames.

        This is the right strategy when one orientation estimate is a good
        approximation for the whole map — typically true for well-ordered
        substrate materials.  Results are written to
        ``<base_dir>/seg/frame_?????.h5``.

        Args:
            base_dir: Processing root directory.
            camera: Detector geometry.
            h5_dataset: Dataset path inside ``self.h5_path``.
            mask_path: Path to the ``.npy`` detector mask.
            n_jobs: Number of SLURM array jobs.
            correct_depth: Pass to :func:`~nrxrdct.laue.simulate_laue_stack`.
            **seg_kwargs: Forwarded to
                :func:`~nrxrdct.laue.segmentation.simulation_guided_segmentation`
                (``psf_sigma``, ``search_radius``, ``min_snr``, etc.).

        Returns:
            List of SLURM job IDs.
        """
        if self.h5_path is None:
            raise ValueError("h5_path not set on this LayeredMap.")

        dirs      = self._setup_slurm_dirs(base_dir, "seg")
        stack_pkl = self._write_stack_pkl(base_dir)

        all_frames = list(range(self.ny * self.nx))
        chunks = [
            list(map(int, c))
            for c in np.array_split(all_frames, min(n_jobs, len(all_frames)))
            if len(c) > 0
        ]

        meta = {
            "stack_pkl":     stack_pkl,
            "camera":        self._camera_to_dict(camera),
            "h5_path":       self.h5_path,
            "h5_dataset":    h5_dataset,
            "mask_path":     mask_path,
            "monitor":       self.monitor,
            "seg_dir":       dirs["out"],
            "f2_thresh":     f2_thresh,
            "correct_depth": correct_depth,
            "overwrite":     overwrite,
            **seg_kwargs,
        }
        meta_path = os.path.join(dirs["job_meta"], "guided_seg_meta.json")
        with open(meta_path, "w") as fh:
            json.dump(meta, fh, indent=2)

        job_ids = self._submit_jobs(
            "lm_gseg", "nrxrdct.laue.workers.slurm_guided_seg_worker",
            meta_path, chunks, dirs["slurm_logs"],
            partition=partition, time=time, mem=mem,
            cpus_per_task=cpus_per_task, python_bin=python_bin,
            extra_sbatch=extra_sbatch,
        )
        print(f"Guided segmentation: {len(job_ids)} jobs → {dirs['out']}")
        return job_ids

    def submit_segmentation(
        self,
        base_dir: str,
        h5_path: "str | None" = None,
        h5_dataset: "str | None" = None,
        n_jobs: int = 10,
        *,
        tiff_dir: "str | None" = None,
        partition: str = "all",
        time: str = "01:00:00",
        mem: str = "4G",
        cpus_per_task: int = 1,
        python_bin: str = "python",
        mask_path: "str | None" = None,
        method: str = "LoG",
        method_kwargs: "dict | None" = None,
        min_size: int = 3,
        max_size: int = 500,
        gap_exclude: int = 3,
        gap_closing: int = 3,
        bg_sigma: float = 251,
        max_components: int = 1,
        d: int = 10,
        r_squared_min: float = 0.0,
        include_unfitted: bool = False,
        fit_spots: bool = True,
        extra_sbatch: "dict | None" = None,
    ) -> list:
        """
        Submit segmentation jobs to SLURM.

        Peak-list files are written to ``<base_dir>/seg/frame_?????.h5``.
        Pass *seg_dir* to the ``submit_orientation`` / ``submit_strain`` calls
        (defaults to the same path automatically).

        Args:
            base_dir: Root processing directory.
            h5_path: Path to the HDF5 scan file (must be accessible from
                compute nodes).
            h5_dataset: Dataset path inside *h5_path*, e.g.
                ``'1.1/measurement/eiger4m'``.  Mutually exclusive with
                *tiff_dir*; exactly one must be supplied.
            tiff_dir: Path to a directory of ``img_*.tif`` files.
                Mutually exclusive with *h5_dataset*.
            n_jobs: Number of SLURM array jobs.
            method: ``'LoG'``, ``'WTH'``, or ``'HYBRID'``.
            bg_sigma: Gaussian sigma for background estimation (pixels).
            r_squared_min: Minimum Gaussian-fit R² to accept a spot.

        Returns:
            List of SLURM job IDs.
        """
        h5_path = h5_path or self.h5_path
        if h5_path is None and h5_dataset is not None:
            raise ValueError("h5_path not set on the object and not passed as argument.")
        if h5_dataset is None and tiff_dir is None:
            raise ValueError("Provide either h5_dataset or tiff_dir.")
        if h5_dataset is not None and tiff_dir is not None:
            raise ValueError("Provide h5_dataset or tiff_dir, not both.")

        dirs = self._setup_slurm_dirs(base_dir, "seg")
        all_frames = list(range(self.ny * self.nx))
        chunks = [
            list(map(int, c))
            for c in np.array_split(all_frames, min(n_jobs, len(all_frames)))
            if len(c) > 0
        ]
        meta = {
            "h5_path":        h5_path,
            "h5_dataset":     h5_dataset,
            "tiff_dir":       tiff_dir,
            "seg_dir":        dirs["out"],
            "mask_path":      mask_path,
            "monitor":        self.monitor,
            "method":         method,
            "method_kwargs":  method_kwargs or {},
            "min_size":       min_size,
            "max_size":       max_size,
            "gap_exclude":    gap_exclude,
            "gap_closing":    gap_closing,
            "bg_sigma":       bg_sigma,
            "max_components": max_components,
            "d":              d,
            "r_squared_min":  r_squared_min,
            "include_unfitted": include_unfitted,
            "fit_spots":      fit_spots,
        }
        meta_path = os.path.join(dirs["job_meta"], "seg_meta.json")
        with open(meta_path, "w") as fh:
            json.dump(meta, fh, indent=2)

        job_ids = self._submit_jobs(
            "lm_seg", "nrxrdct.laue.workers.slurm_seg_worker",
            meta_path, chunks, dirs["slurm_logs"],
            partition=partition, time=time, mem=mem,
            cpus_per_task=cpus_per_task, python_bin=python_bin,
            extra_sbatch=extra_sbatch,
        )
        print(f"Segmentation: {len(job_ids)} jobs → {dirs['out']}")
        return job_ids

    def submit_orientation(
        self,
        base_dir: str,
        camera,
        *,
        seg_dir: "str | None" = None,
        n_jobs: int = 10,
        partition: str = "all",
        time: str = "02:00:00",
        mem: str = "4G",
        cpus_per_task: int = 1,
        python_bin: str = "python",
        max_match_px=(30, 10, 3),
        min_matched: int = 5,
        min_match_rate: float = 0.2,
        max_rms_px: "float | None" = None,
        r_squared_min: float = 0.0,
        include_unfitted: bool = False,
        geometry_only: bool = True,
        f2_thresh: float = 1e-6,
        correct_depth: bool = False,
        overwrite: bool = False,
        extra_sbatch: "dict | None" = None,
        **fit_kwargs,
    ) -> list:
        """
        Submit peak-list-based orientation fitting for the full LayeredCrystal
        stack to SLURM.

        Each frame is fitted with :func:`~nrxrdct.laue.fitting.fit_orientation_stack`,
        which finds a single global rotation shared by all layers simultaneously.
        Fitting is staged: ``max_match_px`` is applied as a sequence of decreasing
        matching radii so the optimizer coarsely locks onto the orientation before
        tightening.

        **Workflow**::

            job_ids = lmap.submit_orientation(base_dir, camera, seg_dir="path/to/seg")
            # wait for SLURM jobs to finish
            lmap.collect("path/to/base_dir/layered_ori/")

        Output files are written to ``<base_dir>/layered_ori/frame_?????.npz``.

        Args:
            base_dir: Root directory for all job artefacts.  Sub-directories
                ``layered_ori/``, ``slurm_logs/`` and ``job_meta/`` are created
                automatically.
            camera: Detector geometry (:class:`~nrxrdct.laue.camera.Camera`).
            seg_dir: Directory containing ``frame_?????.h5`` segmentation files
                produced by :meth:`segment_frame` or the segmentation workers.
                Defaults to ``<base_dir>/seg``.
            n_jobs: Number of SLURM array jobs.  Frames are split evenly across
                jobs; each job runs all its frames in a ``ProcessPoolExecutor``
                with up to ``cpus_per_task`` workers.
            partition: SLURM partition name.
            time: Wall-clock time limit per job (``HH:MM:SS``).
            mem: Memory limit per job (e.g. ``"4G"``).
            cpus_per_task: CPU cores per job, also the ``ProcessPoolExecutor``
                worker count.  Set to ``> 1`` to parallelise frames within a job.
            python_bin: Python interpreter to invoke on the cluster (e.g.
                ``"python"`` or ``"/path/to/env/bin/python"``).
            max_match_px: Staged matching threshold(s) in pixels.  A sequence
                ``(30, 10, 3)`` runs three successive refinement stages.  Each
                stage uses the previous stage's solution as the starting point.
            min_matched: Minimum number of matched spots required to accept and
                save a result.
            min_match_rate: Minimum fraction of simulated spots that must be
                matched (``n_matched / n_sim``).
            max_rms_px: If set, frames with RMS pixel residual above this value
                are rejected even if ``min_matched`` is satisfied.
            r_squared_min: Minimum Gaussian-fit R² for peaks loaded from the
                segmentation file.  ``0.0`` (default) keeps all peaks including
                those with poor fits.
            include_unfitted: If ``True``, also use peaks that failed Gaussian
                fitting (stored with ``fitted=False`` in the segmentation file).
            geometry_only: If ``True`` (default), precompute the allowed-HKL
                set from structure factors at rest and skip per-spot F² evaluation
                during the fit.  Much faster; disable only if you need exact
                intensity weighting.
            f2_thresh: Structure-factor threshold used when precomputing the
                allowed-HKL set (``geometry_only=True``) or filtering spots
                (``geometry_only=False``).
            correct_depth: Apply depth correction to the pixel projection to
                account for the displacement of the diffracting volume below
                the sample surface.
            overwrite: Re-process frames that already have a result file.
            extra_sbatch: Additional ``#SBATCH`` directives passed verbatim,
                e.g. ``{"account": "myproject", "constraint": "gpu"}``.
            **fit_kwargs: Extra keyword arguments forwarded to
                :func:`~nrxrdct.laue.fitting.fit_orientation_stack`
                (e.g. ``E_min_eV``, ``E_max_eV``, ``source``, ``ftol``).

        Returns:
            List of SLURM job IDs (strings).
        """
        dirs = self._setup_slurm_dirs(base_dir, "layered_ori")
        stack_pkl = self._write_stack_pkl(base_dir)

        meta = {
            "stack_pkl":       stack_pkl,
            "camera":          self._camera_to_dict(camera),
            "seg_dir":         seg_dir or os.path.join(base_dir, "seg"),
            "out_dir":         dirs["out"],
            "max_match_px":    list(max_match_px) if hasattr(max_match_px, "__iter__") else [float(max_match_px)],
            "min_matched":     min_matched,
            "min_match_rate":  min_match_rate,
            "max_rms_px":      max_rms_px,
            "r_squared_min":   r_squared_min,
            "include_unfitted": include_unfitted,
            "geometry_only":   geometry_only,
            "f2_thresh":       f2_thresh,
            "correct_depth":   correct_depth,
            "overwrite":       overwrite,
            **fit_kwargs,
        }
        meta_path = os.path.join(dirs["job_meta"], "layered_orient_meta.json")
        with open(meta_path, "w") as fh:
            json.dump(meta, fh, indent=2)

        all_frames = list(range(self.ny * self.nx))
        chunks = [
            list(map(int, c))
            for c in np.array_split(all_frames, min(n_jobs, len(all_frames)))
            if len(c) > 0
        ]

        return self._submit_jobs(
            "lm_ori",
            "nrxrdct.laue.workers.slurm_layered_orient_worker",
            meta_path, chunks, dirs["slurm_logs"],
            partition=partition, time=time, mem=mem,
            cpus_per_task=cpus_per_task, python_bin=python_bin,
            extra_sbatch=extra_sbatch,
        )

    def submit_strain(
        self,
        base_dir: str,
        camera,
        *,
        seg_dir: "str | None" = None,
        orient_dir: "str | None" = None,
        fit_strain: "tuple | None" = None,
        n_jobs: int = 10,
        partition: str = "all",
        time: str = "04:00:00",
        mem: str = "4G",
        cpus_per_task: int = 1,
        python_bin: str = "python",
        max_match_px=(10, 3),
        min_matched: int = 5,
        min_match_rate: float = 0.2,
        max_rms_px: "float | None" = None,
        r_squared_min: float = 0.0,
        include_unfitted: bool = False,
        geometry_only: bool = True,
        f2_thresh: float = 1e-6,
        correct_depth: bool = False,
        overwrite: bool = False,
        extra_sbatch: "dict | None" = None,
        **fit_kwargs,
    ) -> list:
        """
        Submit simultaneous orientation + per-layer strain fitting to SLURM.

        Each frame is fitted with
        :func:`~nrxrdct.laue.fitting.fit_strain_orientation_stack`, which
        refines a single shared global rotation (3 parameters) and an
        independent strain tensor for every layer (``N_layers × len(fit_strain)``
        parameters).  Because the parameter count grows with the number of
        layers, this is inherently slower than single-crystal strain fitting;
        reducing ``fit_strain`` to only the physically expected components is
        the main lever to control cost.

        **Workflow**::

            job_ids = lmap.submit_strain(base_dir, camera, seg_dir="path/to/seg",
                                         fit_strain=("e_xx", "e_yy", "e_zz"))
            # wait for SLURM jobs to finish
            lmap.collect("path/to/base_dir/layered_strain/")

        Output files are written to ``<base_dir>/layered_strain/frame_?????.npz``.
        Each file contains ``R_global``, ``rotvec``, ``U_layers``,
        ``U_eff_layers``, ``strain_tensors``, ``strain_voigts``, ``rms_px``,
        ``n_matched``, ``match_rate``, and ``n_sim``.

        Args:
            base_dir: Root directory for all job artefacts.
            camera: Detector geometry (:class:`~nrxrdct.laue.camera.Camera`).
            seg_dir: Directory containing ``frame_?????.h5`` segmentation files.
                Defaults to ``<base_dir>/seg``.
            fit_strain: Strain tensor components to refine, e.g.
                ``("e_xx", "e_yy", "e_zz")`` for diagonal/biaxial strain only.
                ``None`` (default) refines all six independent components
                ``(e_xx, e_yy, e_zz, e_xy, e_xz, e_yz)``.  Reducing this set
                is the most effective way to speed up the fit for systems with
                known symmetry constraints.
            n_jobs: Number of SLURM array jobs.
            partition: SLURM partition name.
            time: Wall-clock time limit per job (``HH:MM:SS``).  Strain fitting
                is significantly slower than orientation-only; increase if jobs
                time out.
            mem: Memory limit per job.
            cpus_per_task: CPU cores per job / ``ProcessPoolExecutor`` workers.
            python_bin: Python interpreter on the cluster.
            max_match_px: Staged matching threshold(s) in pixels.  A sequence
                ``(10, 3)`` runs two stages.
            min_matched: Minimum matched spots to save a result.
            min_match_rate: Minimum fraction of simulated spots matched.
            max_rms_px: Maximum RMS pixel residual to accept.
            r_squared_min: Minimum peak R² for loading from segmentation files.
                ``0.0`` (default) keeps all peaks.
            include_unfitted: Include peaks that failed Gaussian fitting.
            geometry_only: Skip per-spot F² evaluation during fitting by using
                a precomputed allowed-HKL set.  Recommended (default ``True``).
            f2_thresh: Structure-factor threshold for allowed-HKL precomputation.
            correct_depth: Apply depth correction to pixel projections.
            overwrite: Re-process frames that already have a result file.
            extra_sbatch: Additional ``#SBATCH`` directives.
            **fit_kwargs: Extra keyword arguments forwarded to
                :func:`~nrxrdct.laue.fitting.fit_strain_orientation_stack`
                (e.g. ``E_min_eV``, ``E_max_eV``, ``strain_scale``, ``ftol``).

        Returns:
            List of SLURM job IDs (strings).
        """
        from .fitting import _STRAIN_ALL
        _fit_strain = list(fit_strain) if fit_strain is not None else list(_STRAIN_ALL)

        dirs = self._setup_slurm_dirs(base_dir, "layered_strain")
        stack_pkl = self._write_stack_pkl(base_dir)

        meta = {
            "stack_pkl":       stack_pkl,
            "camera":          self._camera_to_dict(camera),
            "seg_dir":         seg_dir or os.path.join(base_dir, "seg"),
            "orient_dir":      orient_dir,
            "out_dir":         dirs["out"],
            "fit_strain":      _fit_strain,
            "max_match_px":    list(max_match_px) if hasattr(max_match_px, "__iter__") else [float(max_match_px)],
            "min_matched":     min_matched,
            "min_match_rate":  min_match_rate,
            "max_rms_px":      max_rms_px,
            "r_squared_min":   r_squared_min,
            "include_unfitted": include_unfitted,
            "geometry_only":   geometry_only,
            "f2_thresh":       f2_thresh,
            "correct_depth":   correct_depth,
            "overwrite":       overwrite,
            **fit_kwargs,
        }
        meta_path = os.path.join(dirs["job_meta"], "layered_strain_meta.json")
        with open(meta_path, "w") as fh:
            json.dump(meta, fh, indent=2)

        all_frames = list(range(self.ny * self.nx))
        chunks = [
            list(map(int, c))
            for c in np.array_split(all_frames, min(n_jobs, len(all_frames)))
            if len(c) > 0
        ]

        return self._submit_jobs(
            "lm_strain",
            "nrxrdct.laue.workers.slurm_layered_strain_worker",
            meta_path, chunks, dirs["slurm_logs"],
            partition=partition, time=time, mem=mem,
            cpus_per_task=cpus_per_task, python_bin=python_bin,
            extra_sbatch=extra_sbatch,
        )

    def submit_image_orientation(
        self,
        base_dir: str,
        camera,
        h5_path: "str | None" = None,
        h5_dataset: "str | None" = None,
        *,
        n_jobs: int = 10,
        partition: str = "all",
        time: str = "02:00:00",
        mem: str = "8G",
        cpus_per_task: int = 1,
        python_bin: str = "python",
        geometry_only: bool = True,
        f2_thresh: float = 1e-6,
        correct_depth: bool = False,
        overwrite: bool = False,
        extra_sbatch: "dict | None" = None,
        **fit_kwargs,
    ) -> list:
        """
        Submit image-based orientation refinement to SLURM.

        Unlike :meth:`submit_orientation`, which works from a pre-extracted
        peak list, this method refines the orientation directly from the raw
        detector image using
        :func:`~nrxrdct.laue.fitting.refine_orientation_image_stack`.
        The fit correlates a simulated spot pattern against the measured
        intensity image, which is useful when segmentation is difficult or
        unreliable.

        Requires ``self.h5_path`` to be set on the object, or an explicit
        ``h5_path`` argument.

        **Workflow**::

            lmap.h5_path = "/data/scan.h5"
            job_ids = lmap.submit_image_orientation(
                base_dir, camera, h5_dataset="1.1/measurement/eiger4m"
            )
            # wait for SLURM jobs to finish
            lmap.collect("path/to/base_dir/layered_img_ori/")

        Output files are written to ``<base_dir>/layered_img_ori/frame_?????.npz``.
        Each file contains ``R_global``, ``rotvec``, ``U_layers``, ``score``,
        ``score0``, and ``n_sim``.

        Args:
            base_dir: Root directory for all job artefacts.
            camera: Detector geometry (:class:`~nrxrdct.laue.camera.Camera`).
            h5_path: Path to the HDF5 scan file.  Defaults to ``self.h5_path``.
            h5_dataset: Dataset path within the HDF5 file, e.g.
                ``"1.1/measurement/eiger4m"``.  Required.
            n_jobs: Number of SLURM array jobs.
            partition: SLURM partition name.
            time: Wall-clock time limit per job (``HH:MM:SS``).
            mem: Memory limit per job.  Image-based fitting loads full detector
                frames; increase if jobs are killed for OOM.
            cpus_per_task: CPU cores per job / ``ProcessPoolExecutor`` workers.
            python_bin: Python interpreter on the cluster.
            geometry_only: Use precomputed allowed-HKL set instead of computing
                structure factors per spot.  Strongly recommended (default ``True``).
            f2_thresh: Structure-factor threshold for allowed-HKL precomputation.
            correct_depth: Apply depth correction to pixel projections.
            overwrite: Re-process frames that already have a result file.
            extra_sbatch: Additional ``#SBATCH`` directives.
            **fit_kwargs: Extra keyword arguments forwarded to
                :func:`~nrxrdct.laue.fitting.refine_orientation_image_stack`
                (e.g. ``E_min_eV``, ``E_max_eV``, ``kernel_sigma``,
                ``bg_sigma``, ``max_angle_deg``).

        Returns:
            List of SLURM job IDs (strings).
        """
        h5_path = h5_path or self.h5_path
        if h5_path is None:
            raise ValueError("h5_path not set on the object and not passed as argument.")
        if h5_dataset is None:
            raise ValueError("h5_dataset is required.")
        dirs = self._setup_slurm_dirs(base_dir, "layered_img_ori")
        stack_pkl = self._write_stack_pkl(base_dir)

        meta = {
            "stack_pkl":    stack_pkl,
            "camera":       self._camera_to_dict(camera),
            "h5_path":      h5_path,
            "h5_dataset":   h5_dataset,
            "out_dir":      dirs["out"],
            "geometry_only":  geometry_only,
            "f2_thresh":      f2_thresh,
            "correct_depth":  correct_depth,
            "overwrite":      overwrite,
            **fit_kwargs,
        }
        meta_path = os.path.join(dirs["job_meta"], "layered_img_orient_meta.json")
        with open(meta_path, "w") as fh:
            json.dump(meta, fh, indent=2)

        all_frames = list(range(self.ny * self.nx))
        chunks = [
            list(map(int, c))
            for c in np.array_split(all_frames, min(n_jobs, len(all_frames)))
            if len(c) > 0
        ]

        return self._submit_jobs(
            "lm_img_ori",
            "nrxrdct.laue.workers.slurm_layered_img_orient_worker",
            meta_path, chunks, dirs["slurm_logs"],
            partition=partition, time=time, mem=mem,
            cpus_per_task=cpus_per_task, python_bin=python_bin,
            extra_sbatch=extra_sbatch,
        )

    def submit_strain_image(
        self,
        base_dir: str,
        camera,
        h5_path: "str | None" = None,
        h5_dataset: "str | None" = None,
        *,
        fit_strain: "tuple | None" = None,
        n_jobs: int = 10,
        partition: str = "all",
        time: str = "04:00:00",
        mem: str = "8G",
        cpus_per_task: int = 1,
        python_bin: str = "python",
        geometry_only: bool = True,
        f2_thresh: float = 1e-6,
        correct_depth: bool = False,
        overwrite: bool = False,
        extra_sbatch: "dict | None" = None,
        **fit_kwargs,
    ) -> list:
        """
        Submit image-based orientation + per-layer strain refinement to SLURM.

        Combines image-based fitting with full per-layer strain refinement using
        :func:`~nrxrdct.laue.fitting.refine_strain_image_stack`.  This is the
        most computationally intensive submit method: it fits a shared global
        rotation plus an independent strain tensor per layer directly against
        the raw detector image.

        Use this method when you want strain maps without a prior segmentation
        step, or as a refinement stage after :meth:`submit_image_orientation`.

        **Workflow**::

            lmap.h5_path = "/data/scan.h5"
            job_ids = lmap.submit_strain_image(
                base_dir, camera,
                h5_dataset="1.1/measurement/eiger4m",
                fit_strain=("e_xx", "e_yy", "e_zz"),
            )
            # wait for SLURM jobs to finish
            lmap.collect("path/to/base_dir/layered_img_strain/")

        Output files are written to ``<base_dir>/layered_img_strain/frame_?????.npz``.
        Each file contains ``R_global``, ``rotvec``, ``U_layers``,
        ``U_eff_layers``, ``strain_tensors``, ``strain_voigts``, ``score``,
        ``score0``, and ``n_sim``.

        Args:
            base_dir: Root directory for all job artefacts.
            camera: Detector geometry (:class:`~nrxrdct.laue.camera.Camera`).
            h5_path: Path to the HDF5 scan file.  Defaults to ``self.h5_path``.
            h5_dataset: Dataset path within the HDF5 file.  Required.
            fit_strain: Strain tensor components to refine, e.g.
                ``("e_xx", "e_yy", "e_zz")``.  ``None`` refines all six.
                Reducing this set is the most effective way to lower compute cost.
            n_jobs: Number of SLURM array jobs.
            partition: SLURM partition name.
            time: Wall-clock time limit per job.  This method is the slowest of
                the four; budget generously.
            mem: Memory limit per job.
            cpus_per_task: CPU cores per job / ``ProcessPoolExecutor`` workers.
            python_bin: Python interpreter on the cluster.
            geometry_only: Use precomputed allowed-HKL set.  Strongly recommended
                (default ``True``).
            f2_thresh: Structure-factor threshold for allowed-HKL precomputation.
            correct_depth: Apply depth correction to pixel projections.
            overwrite: Re-process frames that already have a result file.
            extra_sbatch: Additional ``#SBATCH`` directives.
            **fit_kwargs: Extra keyword arguments forwarded to
                :func:`~nrxrdct.laue.fitting.refine_strain_image_stack`
                (e.g. ``E_min_eV``, ``E_max_eV``, ``kernel_sigma``,
                ``strain_scale``, ``max_angle_deg``).

        Returns:
            List of SLURM job IDs (strings).
        """
        h5_path = h5_path or self.h5_path
        if h5_path is None:
            raise ValueError("h5_path not set on the object and not passed as argument.")
        if h5_dataset is None:
            raise ValueError("h5_dataset is required.")
        from .fitting import _STRAIN_ALL
        _fit_strain = list(fit_strain) if fit_strain is not None else list(_STRAIN_ALL)

        dirs = self._setup_slurm_dirs(base_dir, "layered_img_strain")
        stack_pkl = self._write_stack_pkl(base_dir)

        meta = {
            "stack_pkl":    stack_pkl,
            "camera":       self._camera_to_dict(camera),
            "h5_path":      h5_path,
            "h5_dataset":   h5_dataset,
            "out_dir":      dirs["out"],
            "fit_strain":   _fit_strain,
            "geometry_only":  geometry_only,
            "f2_thresh":      f2_thresh,
            "correct_depth":  correct_depth,
            "overwrite":      overwrite,
            **fit_kwargs,
        }
        meta_path = os.path.join(dirs["job_meta"], "layered_img_strain_meta.json")
        with open(meta_path, "w") as fh:
            json.dump(meta, fh, indent=2)

        all_frames = list(range(self.ny * self.nx))
        chunks = [
            list(map(int, c))
            for c in np.array_split(all_frames, min(n_jobs, len(all_frames)))
            if len(c) > 0
        ]

        return self._submit_jobs(
            "lm_img_strain",
            "nrxrdct.laue.workers.slurm_layered_img_strain_worker",
            meta_path, chunks, dirs["slurm_logs"],
            partition=partition, time=time, mem=mem,
            cpus_per_task=cpus_per_task, python_bin=python_bin,
            extra_sbatch=extra_sbatch,
        )

    # ── Repr ──────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        n_fitted = int(np.sum(self.n_matched >= 0))
        return (
            f"LayeredMap(ny={self.ny}, nx={self.nx}, "
            f"n_layers={self.n_layers}, "
            f"fitted={n_fitted}/{self.ny * self.nx})"
        )
