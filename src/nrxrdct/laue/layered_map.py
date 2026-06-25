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
) -> tuple:
    """Orientation + per-layer strain stack fit for one frame."""
    from .fitting import fit_strain_orientation_stack

    out_path = os.path.join(out_dir, f"frame_{frame_idx:05d}.npz")
    if os.path.exists(out_path) and not overwrite:
        return frame_idx, True
    if obs_xy is None or len(obs_xy) < min_matched:
        return frame_idx, False

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

    if result.n_matched < min_matched:
        return frame_idx, False
    if result.match_rate < min_match_rate:
        return frame_idx, False
    if max_rms_px is not None and result.rms_px > max_rms_px:
        return frame_idx, False

    tmp = out_path + ".tmp"
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

    tmp = out_path + ".tmp"
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

    tmp = out_path + ".tmp"
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
    ) -> None:
        self.ny        = ny
        self.nx        = nx
        self.stack     = stack
        self.h5_path   = h5_path
        self.entry     = entry
        self.save_path = save_path

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
        r_squared_min: float = 0.9,
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
        r_squared_min: float = 0.9,
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
        motor_x: "str | None" = None,
        motor_y: "str | None" = None,
        ax=None,
    ):
        """
        Inverse pole figure colour map for *layer*.

        Requires ``orix`` (``pip install orix``).

        Args:
            layer: Layer index.
            direction: Reference direction as an ``orix.vector.Vector3d``
                (default: z-axis).
        """
        try:
            from orix.quaternion import Rotation as ORotation
            from orix.vector import Vector3d
        except ImportError:
            raise ImportError("plot_ipf_map requires orix: pip install orix")

        if direction is None:
            direction = Vector3d.zvector()

        U_map = self.U[layer]                     # (ny, nx, 3, 3)
        valid = ~np.any(np.isnan(U_map), axis=(-2, -1))

        # Collect valid orientations and positions
        scipy_quats = []
        positions   = []
        for iy in range(self.ny):
            for ix in range(self.nx):
                if valid[iy, ix]:
                    q = Rotation.from_matrix(U_map[iy, ix]).as_quat()
                    scipy_quats.append(q)     # [x, y, z, w]
                    positions.append((iy, ix))

        if not scipy_quats:
            print("No valid orientations to plot.")
            return ax

        # Convert scipy [x,y,z,w] → orix [w,x,y,z]
        q_arr   = np.array(scipy_quats)
        orix_q  = q_arr[:, [3, 0, 1, 2]]
        orot    = ORotation(orix_q)
        rgb     = orot.IPF_color(direction)   # (n, 3) float in [0, 1]

        img = np.ones((self.ny, self.nx, 3))
        for (iy, ix), c in zip(positions, rgb):
            img[iy, ix] = c

        extent, xlabel, ylabel = self._motor_extent(motor_x, motor_y)
        if ax is None:
            _, ax = plt.subplots(figsize=(6, 5))

        imkw = dict(origin="upper")
        if extent is not None:
            imkw.update(extent=extent, aspect="auto")
        ax.imshow(img, **imkw)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"IPF map — {self.layer_labels[layer]}")
        return ax

    def inspect_frame(
        self,
        camera,
        frame_idx: int,
        *,
        h5_path: "str | None" = None,
        h5_dataset: str = "1.1/measurement/eiger4m",
        tiff_dir: "str | None" = None,
        E_min_eV: float = 5_000,
        E_max_eV: float = 27_000,
        max_match_dist: float = 5.0,
        use_eff: bool = True,
        figsize: tuple = (14, 6),
    ):
        """
        Display the diffraction image for *frame_idx* overlaid with simulated
        spots from all stack layers.

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
            use_eff: If ``True`` (default), use ``U_eff`` (strained) when
                available; otherwise use ``U``.
        """
        from .simulation import simulate_laue_stack
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
                verbose=False,
            )
        finally:
            for layer, U0 in zip(self.stack.all_layers, saved_U):
                layer.U = U0

        fig, axes = plot_measured_vs_simulated(
            np.empty((0, 9)),
            spots,
            image=image,
            camera=camera,
            max_match_dist=max_match_dist,
            figsize=figsize,
        )
        fig.suptitle(
            f"Frame {frame_idx}  [{iy}, {ix}]  —  LayeredMap",
            fontsize=11,
        )
        return fig, axes

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
        r_squared_min: float = 0.9,
        include_unfitted: bool = False,
        geometry_only: bool = True,
        f2_thresh: float = 1e-4,
        overwrite: bool = False,
        extra_sbatch: "dict | None" = None,
        **fit_kwargs,
    ) -> list:
        """
        Submit stack orientation fitting to SLURM.

        Frames are split into *n_jobs* chunks.  Each job runs
        ``slurm_layered_orient_worker`` and writes
        ``<base_dir>/layered_ori/frame_?????.npz``.
        Call :meth:`collect` on ``<base_dir>/layered_ori/`` after jobs finish.

        Args:
            base_dir: Root directory for all outputs.
            camera: Detector geometry.
            seg_dir: Directory with ``frame_?????.h5`` peaklist files.
                Defaults to ``<base_dir>/seg``.
            n_jobs: Number of SLURM jobs to submit.

        Returns:
            List of SLURM job IDs.
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
        r_squared_min: float = 0.9,
        include_unfitted: bool = False,
        geometry_only: bool = True,
        f2_thresh: float = 1e-4,
        overwrite: bool = False,
        extra_sbatch: "dict | None" = None,
        **fit_kwargs,
    ) -> list:
        """
        Submit stack orientation + per-layer strain fitting to SLURM.

        Writes to ``<base_dir>/layered_strain/frame_?????.npz``.
        Call :meth:`collect` on that directory after jobs finish.

        Args:
            fit_strain: Strain components to refine, e.g.
                ``('e_xx', 'e_yy', 'e_zz')``.  ``None`` refines all six.
        """
        from .fitting import _STRAIN_ALL
        _fit_strain = list(fit_strain) if fit_strain is not None else list(_STRAIN_ALL)

        dirs = self._setup_slurm_dirs(base_dir, "layered_strain")
        stack_pkl = self._write_stack_pkl(base_dir)

        meta = {
            "stack_pkl":       stack_pkl,
            "camera":          self._camera_to_dict(camera),
            "seg_dir":         seg_dir or os.path.join(base_dir, "seg"),
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
        h5_path: str,
        h5_dataset: str,
        *,
        n_jobs: int = 10,
        partition: str = "all",
        time: str = "02:00:00",
        mem: str = "8G",
        cpus_per_task: int = 1,
        python_bin: str = "python",
        geometry_only: bool = True,
        f2_thresh: float = 1e-4,
        overwrite: bool = False,
        extra_sbatch: "dict | None" = None,
        **fit_kwargs,
    ) -> list:
        """
        Submit image-based orientation refinement to SLURM.

        Writes to ``<base_dir>/layered_img_ori/frame_?????.npz``.

        Args:
            h5_path: Path to the HDF5 scan file (must be accessible from
                compute nodes).
            h5_dataset: Dataset path inside *h5_path*, e.g.
                ``"1.1/measurement/eiger4m"``.
        """
        dirs = self._setup_slurm_dirs(base_dir, "layered_img_ori")
        stack_pkl = self._write_stack_pkl(base_dir)

        meta = {
            "stack_pkl":    stack_pkl,
            "camera":       self._camera_to_dict(camera),
            "h5_path":      h5_path,
            "h5_dataset":   h5_dataset,
            "out_dir":      dirs["out"],
            "geometry_only": geometry_only,
            "f2_thresh":    f2_thresh,
            "overwrite":    overwrite,
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
        h5_path: str,
        h5_dataset: str,
        *,
        fit_strain: "tuple | None" = None,
        n_jobs: int = 10,
        partition: str = "all",
        time: str = "04:00:00",
        mem: str = "8G",
        cpus_per_task: int = 1,
        python_bin: str = "python",
        geometry_only: bool = True,
        f2_thresh: float = 1e-4,
        overwrite: bool = False,
        extra_sbatch: "dict | None" = None,
        **fit_kwargs,
    ) -> list:
        """
        Submit image-based orientation + per-layer strain refinement to SLURM.

        Writes to ``<base_dir>/layered_img_strain/frame_?????.npz``.

        Args:
            fit_strain: Strain components to refine.  ``None`` refines all six.
            h5_path: Path to the HDF5 scan file.
            h5_dataset: Dataset path inside *h5_path*.
        """
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
            "geometry_only": geometry_only,
            "f2_thresh":    f2_thresh,
            "overwrite":    overwrite,
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
