"""
nrxrdct.laue.map — GrainMap: multi-grain results on a 2-D micro-Laue raster.
=============================================================================

Typical workflow::

    gmap = GrainMap(ny=21, nx=21, h5_path="scan.h5", processing_dir="./")

    # Fill results point by point (e.g. inside a processing loop)
    for iy in range(gmap.ny):
        for ix in range(gmap.nx):
            frame_idx = gmap.frame_index(iy, ix)
            obs_xy = load_peaklist(frame_idx)
            for gi in range(gmap.n_grains):
                result = fit_orientation(crystal, camera, obs_xy,
                                         gmap.U_ref[gi],
                                         max_match_px=[30, 10, 3])
                gmap.set_result(iy, ix, gi, result)

    # Inspect
    gmap.plot_map("match_rate", grain=0)
    gmap.save("grainmap.h5")

    # Later
    gmap2 = GrainMap.load("grainmap.h5")
"""

from __future__ import annotations

import concurrent.futures
import glob
import json
import os
import dill as pickle
import re
import subprocess

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation


# ─────────────────────────────────────────────────────────────────────────────
# Scan-title parser (minimal; user can extend)
# ─────────────────────────────────────────────────────────────────────────────

def parse_scan_title(title: str) -> dict:
    """
    Parse an ESRF/SPEC scan-command string and return scan geometry.

    **Supported commands**
    `dmesh` / `mesh`
        `dmesh motor1 start1 stop1 n1 motor2 start2 stop2 n2 [exposure]`
        → `ny = n1+1`,  `nx = n2+1`
    `ascan`
        `ascan motor start stop n [exposure]`
        → `ny = 1`,  `nx = n+1`
    `loopscan`
        `loopscan n [exposure]`
        → `ny = 1`,  `nx = n`

    Returns:
        dict with keys: `cmd`, `ny`, `nx`, `n_frames`,
        and optionally `motor1`, `motor2`, `start1/2`, `stop1/2`, `n1/2`.

    Raises:
        ValueError: If the command is not recognised.
"""
    tokens = title.strip().split()
    cmd = tokens[0].lower().lstrip("#").strip()

    if cmd in ("dmesh", "mesh"):
        motor1 = tokens[1]
        start1, stop1, n1 = float(tokens[2]), float(tokens[3]), int(tokens[4])
        motor2 = tokens[5]
        start2, stop2, n2 = float(tokens[6]), float(tokens[7]), int(tokens[8])
        ny, nx = n1 + 1, n2 + 1
        return dict(cmd=cmd,
                    motor1=motor1, start1=start1, stop1=stop1, n1=n1,
                    motor2=motor2, start2=start2, stop2=stop2, n2=n2,
                    ny=ny, nx=nx, n_frames=ny * nx)

    if cmd in ("ascan", "a2scan"):
        motor1 = tokens[1]
        start1, stop1, n1 = float(tokens[2]), float(tokens[3]), int(tokens[4])
        ny, nx = 1, n1 + 1
        return dict(cmd=cmd,
                    motor1=motor1, start1=start1, stop1=stop1, n1=n1,
                    ny=ny, nx=nx, n_frames=nx)

    if cmd == "loopscan":
        n = int(tokens[1])
        return dict(cmd=cmd, ny=1, nx=n, n_frames=n)

    raise ValueError(f"Unrecognised scan command {cmd!r} in title {title!r}")



def _read_motor_array(h5_file: h5py.File, entry: str,
                      motor: str, n_frames: int) -> np.ndarray | None:
    """Read a motor-position array from common h5 locations."""
    candidates = [
        f"{entry}/instrument/positioners/{motor}",
        f"{entry}/measurement/{motor}",
    ]
    for path in candidates:
        if path in h5_file:
            arr = np.asarray(h5_file[path], dtype=float).ravel()
            if arr.size == n_frames:
                return arr
            if arr.size == 1:
                return np.full(n_frames, arr.item())
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────────────────────

class GrainMap:
    """
    Multi-grain orientation-fit results on a 2-D micro-Laue raster scan.

    Args:
        ny, nx (int): Number of map rows (slow motor) and columns (fast motor).
        h5_path (str): Path to the master HDF5 scan file.  Used to read motor positions.
            May be `None` if you don't need motor coordinates.
        processing_dir (str or None): Directory scanned for `UB[0-9]*.npy` grain reference matrices.
            Defaults to the directory containing *h5_path*, or CWD if both are
            absent.
        entry (str): HDF5 entry key, e.g. `"1.1"`.
        motor_x (str or None): Name of the horizontal (column) motor in the HDF5 file, e.g.
            `"xech"`.  When given, `motors["x"]` is populated as an alias
            so you can always use `gmap.motors["x"]` regardless of the
            beamline-specific motor name.
        motor_y (str or None): Name of the vertical (row) motor, e.g. `"yech"`.  Populates
            `motors["y"]` as an alias.

    Attributes:
        ny, nx (int):
        n_grains (int): Number of UB files found.
        U_ref ((n_grains, 3, 3) ndarray): Reference orientation matrices loaded from `UB*.npy`.
        U ((n_grains, ny, nx, 3, 3) ndarray): Fitted orientation matrices.  `NaN` where not yet fitted.
        rms_px ((n_grains, ny, nx) ndarray):
        mean_px ((n_grains, ny, nx) ndarray):
        n_matched ((n_grains, ny, nx) int ndarray): (-1 = not fitted)
        match_rate ((n_grains, ny, nx) ndarray):
        cost ((n_grains, ny, nx) ndarray):
        strain_tensor_deviatoric ((n_grains, ny, nx, 3, 3) ndarray): Deviatoric part of the strain tensor
            in the crystal frame: ``ε_dev = ε − (Tr(ε)/3) I``.  Derived automatically from
            ``strain_tensor`` whenever strain results are stored.  `NaN` where not yet fitted.
        motors (dict[str, (ny, nx) ndarray]): Motor positions reshaped to the map grid (if h5_path is given and
            motors are found).
"""

    # ── Construction ──────────────────────────────────────────────────────────

    def __init__(
        self,
        ny: int,
        nx: int,
        h5_path: str | None = None,
        processing_dir: str | None = None,
        entry: str = "1.1",
        motor_x: str | None = None,
        motor_y: str | None = None,
        save_path: str | None = None,
    ):
        self.ny = int(ny)
        self.nx = int(nx)
        self.h5_path = h5_path
        self.entry = entry
        self.motor_x = motor_x
        self.motor_y = motor_y

        if processing_dir is None:
            if h5_path is not None:
                processing_dir = os.path.dirname(os.path.abspath(h5_path))
            else:
                processing_dir = os.getcwd()
        self.processing_dir = processing_dir

        if save_path is None and processing_dir is not None:
            self.save_path: str | None = os.path.join(self.processing_dir, "grain_map.h5")
        elif save_path is None:
            self.save_path = os.path.join(os.getcwd(), "grain_map.h5")
        else:
            self.save_path = save_path

        self._load_ub_matrices()
        self._init_arrays()

        self.motors: dict[str, np.ndarray] = {}
        if h5_path is not None:
            self._load_motors()

    # ── UB matrices ───────────────────────────────────────────────────────────

    def _load_ub_matrices(self) -> None:
        pattern = os.path.join(self.processing_dir, "UB[0-9]*.npy")
        files = sorted(
            glob.glob(pattern),
            key=lambda p: int(re.search(r"UB(\d+)\.npy$", p).group(1)),
        )
        self.ub_files: list[str] = files
        self.n_grains: int = len(files)
        if files:
            self.U_ref = np.array([np.load(f) for f in files])
        else:
            self.U_ref = np.empty((0, 3, 3), dtype=float)

    def reload_ub_matrices(self) -> None:
        """Rescan *processing_dir* for UB files and grow arrays if needed."""
        old_n = self.n_grains
        self._load_ub_matrices()
        if self.n_grains > old_n:
            extra = self.n_grains - old_n
            shape2d = (self.ny, self.nx)
            self.U         = np.concatenate([self.U,
                np.full((extra, *shape2d, 3, 3), np.nan)], axis=0)
            self.rms_px    = np.concatenate([self.rms_px,
                np.full((extra, *shape2d), np.nan)], axis=0)
            self.mean_px   = np.concatenate([self.mean_px,
                np.full((extra, *shape2d), np.nan)], axis=0)
            self.n_matched = np.concatenate([self.n_matched,
                np.full((extra, *shape2d), -1, dtype=int)], axis=0)
            self.match_rate = np.concatenate([self.match_rate,
                np.full((extra, *shape2d), np.nan)], axis=0)
            self.cost          = np.concatenate([self.cost,
                np.full((extra, *shape2d), np.nan)], axis=0)
            self.strain_voigt             = np.concatenate([self.strain_voigt,
                np.full((extra, *shape2d, 6), np.nan)], axis=0)
            self.strain_tensor            = np.concatenate([self.strain_tensor,
                np.full((extra, *shape2d, 3, 3), np.nan)], axis=0)
            self.strain_tensor_deviatoric = np.concatenate([self.strain_tensor_deviatoric,
                np.full((extra, *shape2d, 3, 3), np.nan)], axis=0)
            for _ in range(extra):
                self._results.append(
                    [[None] * self.nx for _ in range(self.ny)]
                )

    def drop_grain(self, *indices: int) -> None:
        """
        Remove one or more grain slots from the map in-place.

        All per-grain arrays (`U`, `rms_px`, `match_rate`, etc.),
        the reference UB matrix list, the UB file list, and the stored result
        objects are sliced to exclude the requested indices.  `n_grains` is
        updated accordingly.

        If the merged-grain slot is among the dropped indices it is cleared to
        `None`; if it survives the drop its index is remapped to the new
        position.

        Args:
            *indices (int): One or more grain slot indices to remove.  Duplicates are ignored.
                Pass a single iterable with `*` unpacking if you have a list::

                gmap.drop_grain(0, 2)        # drop grains 0 and 2
                gmap.drop_grain(*[0, 2])     # same, from a list

        Raises:
            ValueError: If any index is out of range or if the call would drop all grains.
"""
        drop_set = set(indices)
        out_of_range = [i for i in drop_set if not (0 <= i < self.n_grains)]
        if out_of_range:
            raise ValueError(
                f"Grain indices out of range (0 – {self.n_grains - 1}): "
                f"{sorted(out_of_range)}"
            )
        keep = [i for i in range(self.n_grains) if i not in drop_set]
        if not keep:
            raise ValueError("drop_grain would remove all grains — at least one must remain.")

        idx = np.array(keep)

        self.U             = self.U[idx]
        self.rms_px        = self.rms_px[idx]
        self.mean_px       = self.mean_px[idx]
        self.n_matched     = self.n_matched[idx]
        self.match_rate    = self.match_rate[idx]
        self.cost          = self.cost[idx]
        self.strain_voigt             = self.strain_voigt[idx]
        self.strain_tensor            = self.strain_tensor[idx]
        self.strain_tensor_deviatoric = self.strain_tensor_deviatoric[idx]
        self.U_ref                    = self.U_ref[idx]
        self.ub_files      = [self.ub_files[i] for i in keep]
        self._results      = [self._results[i] for i in keep]
        self.n_grains      = len(keep)

        # remap _merged_grain: clear if dropped, shift if it survived
        mg = self._merged_grain
        if mg is not None:
            if mg in drop_set:
                self._merged_grain = None
            else:
                self._merged_grain = keep.index(mg)

        dropped_str = ", ".join(str(i) for i in sorted(drop_set))
        print(f"Dropped grain(s) {dropped_str} — {self.n_grains} grain(s) remaining.")

    # ── Array initialisation ──────────────────────────────────────────────────

    def _init_arrays(self) -> None:
        ng = self.n_grains
        shape2d = (self.ny, self.nx)
        self.U             = np.full((ng, *shape2d, 3, 3), np.nan)
        self.rms_px        = np.full((ng, *shape2d), np.nan)
        self.mean_px       = np.full((ng, *shape2d), np.nan)
        self.n_matched     = np.full((ng, *shape2d), -1, dtype=int)
        self.match_rate    = np.full((ng, *shape2d), np.nan)
        self.cost          = np.full((ng, *shape2d), np.nan)
        self.strain_voigt             = np.full((ng, *shape2d, 6), np.nan)
        self.strain_tensor            = np.full((ng, *shape2d, 3, 3), np.nan)
        self.strain_tensor_deviatoric = np.full((ng, *shape2d, 3, 3), np.nan)
        self._results: list[list[list]] = [
            [[None] * self.nx for _ in range(self.ny)]
            for _ in range(ng)
        ]
        self._merged_grain: int | None = None

    # ── Motor positions ───────────────────────────────────────────────────────

    def _load_motors(self) -> None:
        n_frames = self.ny * self.nx
        try:
            with h5py.File(self.h5_path, "r") as f:
                for grp_path in (
                    f"{self.entry}/instrument/positioners",
                    f"{self.entry}/measurement",
                ):
                    if grp_path not in f:
                        continue
                    for motor in f[grp_path].keys():
                        arr = _read_motor_array(f, self.entry, motor, n_frames)
                        if arr is not None:
                            self.motors[motor] = arr.reshape(self.ny, self.nx)
        except Exception:
            pass  # motors are optional; don't break initialisation

        if self.motor_x is not None and self.motor_x in self.motors:
            self.motors["x"] = self.motors[self.motor_x]
        if self.motor_y is not None and self.motor_y in self.motors:
            self.motors["y"] = self.motors[self.motor_y]

    # ── Index helpers ─────────────────────────────────────────────────────────

    def frame_index(self, iy: int, ix: int) -> int:
        """Flat frame index from `(row, col)` — matches h5 frame order."""
        return iy * self.nx + ix

    def map_index(self, frame_idx: int) -> tuple[int, int]:
        """`(row, col)` from a flat frame index."""
        return divmod(frame_idx, self.nx)

    # ── Result storage / retrieval ────────────────────────────────────────────

    def set_result(self, iy: int, ix: int, grain: int, result) -> None:
        """
        Store a fit result at map position `(iy, ix)` for *grain*.

        *result* can be an :class:`~nrxrdct.laue.fitting.OrientationFitResult`,
        :class:`~nrxrdct.laue.fitting.StrainFitResult`, or `None` (marks the
        point as attempted but failed / no convergence).
"""
        self._results[grain][iy][ix] = result
        if result is not None:
            self.U[grain, iy, ix]          = result.U
            self.rms_px[grain, iy, ix]     = result.rms_px
            self.mean_px[grain, iy, ix]    = result.mean_px
            self.n_matched[grain, iy, ix]  = result.n_matched
            self.match_rate[grain, iy, ix] = result.match_rate
            self.cost[grain, iy, ix]       = result.cost
            if hasattr(result, "strain_voigt"):
                self.strain_voigt[grain, iy, ix]  = result.strain_voigt
                eps = result.strain_tensor
                self.strain_tensor[grain, iy, ix] = eps
                self.strain_tensor_deviatoric[grain, iy, ix] = eps - np.trace(eps) / 3.0 * np.eye(3)

    def get_result(self, iy: int, ix: int, grain: int):
        """Return the stored fit result (or `None`) at `(iy, ix, grain)`."""
        return self._results[grain][iy][ix]

    def merge(
        self,
        metric: str = "match_rate",
        min_match_rate: float = 0.0,
        min_n_matched: int = 1,
        max_rms_px: float = np.inf,
        source: str = "auto",
    ) -> tuple[np.ndarray, dict]:
        """
        Select the best-fitting grain at every map position.

        **Background**
        A micro-Laue raster scan is processed independently for each
        reference grain (each `UB*.npy` file).  At every pixel the
        diffraction pattern may be explained well by one grain, poorly by
        another, or not at all.  `merge` collapses the per-grain result
        arrays into a single best-grain map by comparing a chosen quality
        metric across all grain slots and keeping the winner.

        This method does **not** modify the map — it only reads the stored
        metrics and returns the selection.  Call :meth:`apply_merge`
        afterwards to register the result as a new grain slot that all
        existing analysis and plotting methods can use transparently.

        **Quality metric choice**
        `"match_rate"` (default) is the recommended primary criterion.
        It is defined as `n_matched / n_observed` and therefore lies in
        [0, 1] regardless of how many spots are in the pattern.  This makes
        it directly comparable across positions and immune to the artefact
        where a single near-perfectly placed spot gives `rms_px ≈ 0`
        without actually explaining the pattern.

        `"rms_px"` is useful as a *secondary* tiebreaker once a floor on
        `min_match_rate` is already enforced, because a low RMS among
        few matches can be misleading.

        **Quality filters**
        Grains that fail *any* of the three filters
        (`min_match_rate`, `min_n_matched`, `max_rms_px`) are masked
        out before scoring.  A map position where *all* grains fail the
        filters receives `best_grain = -1` and `NaN` metrics — it
        appears white in IPF / scalar maps.

        Args:
            metric (str): Quality metric used to rank grains at each position.  One of:

            * `"match_rate"`  — fraction of observed spots matched
              (higher is better).  Defined as `n_matched / n_observed`.
            * `"n_matched"`   — raw count of matched spots
              (higher is better).  Favours positions with many spots,
              which may not always be desirable.
            * `"rms_px"`      — root-mean-square pixel residual of
              matched pairs (lower is better).
            * `"mean_px"`     — mean pixel residual of matched pairs
              (lower is better).
            * `"cost"`        — optimizer cost function value ½Σr²
              at convergence (lower is better).

            min_match_rate (float): Minimum acceptable match rate for a grain to be considered at
                any position.  Grain fits below this threshold are excluded
                before scoring.  Default `0.0` (no filtering).  A value of
                `0.2`–`0.3` is a reasonable starting point.
            min_n_matched (int): Minimum number of matched spots required.  Positions with
                fewer matches than this (including unfitted positions where
                `n_matched = -1`) are excluded.  Default `1`.
            max_rms_px (float): Maximum RMS pixel residual allowed.  Fits with larger residuals
                are excluded.  Default `inf` (no filtering).
            source (str): Carried forward into the returned `metrics` dict so that
                :meth:`write_merge_links` can inherit the correct result
                directory without the user having to repeat it.  One of
                `"auto"` (default, prefers `strain/` over `ubs/`),
                `"ubs"`, or `"strain"`.

        Returns:
            best_grain ((ny, nx) int ndarray): Index (0-based) of the winning grain slot at each map
                position.  `-1` where no grain passed the quality filters.
            metrics (dict): Quality metrics for the winning grain at each position.
                All arrays have shape `(ny, nx)`:

            * `"match_rate"` — match rate of the winner.
            * `"rms_px"`     — RMS residual of the winner (pixels).
            * `"mean_px"`    — mean residual of the winner (pixels).
            * `"n_matched"`  — matched-spot count of the winner
              (`int`, `-1` where invalid).
            * `"cost"`       — optimizer cost of the winner.
            * `"U"`          — orientation matrix of the winner,
              shape `(ny, nx, 3, 3)`.
            * `"source"`     — the *source* argument (scalar str).

            Values are `NaN` / `-1` at positions where
            `best_grain == -1`.

            **See also**
            :meth:`apply_merge` (register the selection as a new grain slot.):
            :meth:`write_merge_links` (persist the selection as disk symlinks.):
            :meth:`reduce_to_fundamental_zone` (resolve symmetry-equivalent): orientation jumps before merging.
"""
        _higher_better = {"match_rate", "n_matched"}
        _lower_better  = {"rms_px", "mean_px", "cost"}
        _all_metrics   = _higher_better | _lower_better
        if metric not in _all_metrics:
            raise ValueError(
                f"metric must be one of {sorted(_all_metrics)}, got {metric!r}"
            )
        if self.n_grains == 0:
            empty = np.full((self.ny, self.nx), np.nan)
            return (
                np.full((self.ny, self.nx), -1, dtype=int),
                {"match_rate": empty, "rms_px": empty, "mean_px": empty,
                 "n_matched": np.full((self.ny, self.nx), -1, dtype=int),
                 "cost": empty,
                 "U": np.full((self.ny, self.nx, 3, 3), np.nan),
                 "source": source},
            )

        # ── quality mask ─────────────────────────────────────────────────────
        valid = (
            (self.n_matched >= min_n_matched)
            & (self.match_rate >= min_match_rate)
            & (self.rms_px <= max_rms_px)
        )  # (n_grains, ny, nx) bool

        # ── score array (always "higher = better") ───────────────────────────
        raw = {
            "match_rate": self.match_rate,
            "n_matched":  self.n_matched.astype(float),
            "rms_px":     self.rms_px,
            "mean_px":    self.mean_px,
            "cost":       self.cost,
        }[metric]

        score = np.where(valid, raw, np.nan)
        if metric in _lower_better:
            score = -score

        # ── find winning grain ────────────────────────────────────────────────
        any_valid = np.any(valid, axis=0)  # (ny, nx)
        # Replace NaN with -inf so argmax never sees an all-NaN column.
        # np.where evaluates both branches eagerly, so nanargmax would raise
        # "All-NaN slice encountered" for positions where no grain is valid.
        score_filled = np.where(np.isnan(score), -np.inf, score)
        best_grain = np.where(
            any_valid,
            np.argmax(score_filled, axis=0),
            -1,
        ).astype(int)

        # ── extract metric values for the winner ─────────────────────────────
        iy_idx, ix_idx = np.meshgrid(
            np.arange(self.ny), np.arange(self.nx), indexing="ij"
        )
        g_idx = np.clip(best_grain, 0, self.n_grains - 1)

        def _sel(arr: np.ndarray) -> np.ndarray:
            out = arr[g_idx, iy_idx, ix_idx].astype(float)
            out[~any_valid] = np.nan
            return out

        n_sel = self.n_matched[g_idx, iy_idx, ix_idx].copy()
        n_sel[~any_valid] = -1

        U_sel = self.U[g_idx, iy_idx, ix_idx].copy()
        U_sel[~any_valid] = np.nan

        return best_grain, {
            "match_rate": _sel(self.match_rate),
            "rms_px":     _sel(self.rms_px),
            "mean_px":    _sel(self.mean_px),
            "n_matched":  n_sel,
            "cost":       _sel(self.cost),
            "U":          U_sel,
            "source":     source,
        }

    def apply_merge(self, best_grain: np.ndarray, metrics: dict) -> int:
        """
        Write a merge result into a dedicated merged grain slot.

        The first call appends a new slot so that a map with *n* fitted
        grains ends up with *n + 1* grains total.  Every subsequent call
        **replaces** that same slot in-place, so repeated merging with
        different thresholds never adds extra slots.

        Args:
            best_grain ((ny, nx) int ndarray): First return value of :meth:`merge`.
            metrics (dict): Second return value of :meth:`merge`.

        Returns:
            int
                Index of the merged grain slot.  Pass it directly to any method
                that accepts a `grain` argument::

                best_grain, m = gmap.merge(min_match_rate=0.3)
                gi = gmap.apply_merge(best_grain, m)

                gmap.kam_map(gi)
                gmap.plot_ipf_map(gi)
                gmap.plot_map("match_rate", grain=gi)
"""
        shape2d = (self.ny, self.nx)
        valid = (best_grain >= 0)[..., None, None]  # (ny, nx, 1, 1)

        U_new = np.where(valid, metrics["U"], np.nan)            # (ny, nx, 3, 3)

        def _f(key: str) -> np.ndarray:
            return np.asarray(metrics[key], dtype=float)         # (ny, nx)

        n_new = np.asarray(metrics["n_matched"], dtype=int)

        if self._merged_grain is None:
            # ── first merge: append a new slot ───────────────────────────────
            self.U             = np.concatenate([self.U,          U_new[None]],                    axis=0)
            self.rms_px        = np.concatenate([self.rms_px,     _f("rms_px")[None]],             axis=0)
            self.mean_px       = np.concatenate([self.mean_px,    _f("mean_px")[None]],            axis=0)
            self.n_matched     = np.concatenate([self.n_matched,  n_new[None]],                    axis=0)
            self.match_rate    = np.concatenate([self.match_rate, _f("match_rate")[None]],         axis=0)
            self.cost          = np.concatenate([self.cost,       _f("cost")[None]],               axis=0)
            self.strain_voigt  = np.concatenate([self.strain_voigt,
                                                 np.full((1, *shape2d, 6),    np.nan)],            axis=0)
            self.strain_tensor = np.concatenate([self.strain_tensor,
                                                 np.full((1, *shape2d, 3, 3), np.nan)],           axis=0)
            self._results.append([[None] * self.nx for _ in range(self.ny)])
            self.n_grains += 1
            self._merged_grain = self.n_grains - 1
        else:
            # ── subsequent merge: replace the existing slot in-place ──────────
            gi = self._merged_grain
            self.U[gi]          = U_new
            self.rms_px[gi]     = _f("rms_px")
            self.mean_px[gi]    = _f("mean_px")
            self.n_matched[gi]  = n_new
            self.match_rate[gi] = _f("match_rate")
            self.cost[gi]       = _f("cost")
            self.strain_voigt[gi]  = np.nan
            self.strain_tensor[gi] = np.nan
            self._results[gi] = [[None] * self.nx for _ in range(self.ny)]

        return self._merged_grain

    # ── Symmetry reduction ────────────────────────────────────────────────────

    @staticmethod
    def _symmetry_ops(symmetry: str) -> np.ndarray:
        """
        Return the proper rotations of the chosen crystal point group as a
        (N, 3, 3) array.

        **Supported**
        `'cubic'`     24 proper rotations of Oh (m-3m).
        `'hexagonal'` 12 proper rotations of D6h (6/mmm).
        `'tetragonal'` 8 proper rotations of D4h (4/mmm).
        `'orthorhombic'` 4 proper rotations of D2h (mmm).
"""
        from itertools import permutations, product as _product
        if symmetry == "cubic":
            ops = []
            for perm in permutations(range(3)):
                for signs in _product((-1, 1), repeat=3):
                    R = np.zeros((3, 3))
                    for j in range(3):
                        R[perm[j], j] = signs[j]
                    if round(np.linalg.det(R)) == 1:
                        ops.append(R)
            return np.array(ops)   # (24, 3, 3)

        if symmetry == "hexagonal":
            ops = []
            for n in range(6):
                a = n * np.pi / 3
                c, s = np.cos(a), np.sin(a)
                ops.append(np.array([[ c, -s, 0],
                                     [ s,  c, 0],
                                     [ 0,  0, 1]]))
            # C2 rotations about in-plane axes (six of them for D6)
            for n in range(6):
                a = n * np.pi / 6
                c, s = np.cos(2 * a), np.sin(2 * a)
                ops.append(np.array([[ c,  s, 0],
                                     [ s, -c, 0],
                                     [ 0,  0, -1]]))
            return np.array(ops)   # (12, 3, 3)

        if symmetry == "tetragonal":
            ops = []
            for n in range(4):
                a = n * np.pi / 2
                c, s = np.cos(a), np.sin(a)
                ops.append(np.array([[ c, -s, 0],
                                     [ s,  c, 0],
                                     [ 0,  0, 1]]))
            for n in range(4):
                a = n * np.pi / 2
                c, s = np.cos(a), np.sin(a)
                ops.append(np.array([[ c,  s, 0],
                                     [ s, -c, 0],
                                     [ 0,  0, -1]]))
            return np.array(ops)   # (8, 3, 3)

        if symmetry == "orthorhombic":
            return np.array([
                np.eye(3),
                np.diag([1, -1, -1]),
                np.diag([-1, 1, -1]),
                np.diag([-1, -1, 1]),
            ])

        raise ValueError(
            f"Unknown symmetry {symmetry!r}. "
            "Choose from: 'cubic', 'hexagonal', 'tetragonal', 'orthorhombic'."
        )

    def reduce_to_fundamental_zone(
        self,
        grain: int = 0,
        *,
        symmetry: str = "cubic",
        reference: "np.ndarray | None" = None,
    ) -> np.ndarray:
        """
        Re-label each pixel's orientation to the symmetry-equivalent U matrix
        closest to a common reference, resolving spurious isolated pixels that
        converged to a different member of the symmetry-equivalent family.

        **Background: why isolated pixels appear**
        In polychromatic Laue diffraction the peak *positions* on the detector
        are determined jointly by the orientation matrix **U** and the crystal
        lattice **B**, but are independent of the absolute X-ray wavelength.
        A direct consequence is that replacing **U** with any symmetry-
        equivalent `U' = U @ S` (where **S** is any proper rotation in the
        crystal's point group) produces an *identical* set of predicted spot
        positions.  No residual-based optimizer can distinguish between the
        N\\ :sub:`sym` members of this family.

        During a map fit every pixel is refined independently, starting from
        the same pre-indexed reference orientation.  Pixels near a grain
        boundary or in a strained region are more sensitive to the starting
        point, and the optimizer can converge to *any* of the N\\ :sub:`sym`
        symmetry equivalents.  The result is a map that is correct in the
        physical sense (every U is a valid fit) but inconsistent in the
        representation sense: neighbouring pixels that should have the same
        orientation may carry different Euler-angle triples because they
        happen to belong to different members of the symmetry family.  This
        causes isolated pixels with an "orientation" that looks very different
        from its neighbours, purely as an artefact of the representation.

        **Algorithm**
        For each pixel the N\\ :sub:`sym` candidate matrices are formed as
        `U_equiv[s] = U @ ops[s]`, where `ops` is the set of proper
        rotations in the point group (24 for cubic, 12 for hexagonal, 8 for
        tetragonal, 4 for orthorhombic).

        The candidate closest to *reference* **R** is identified by maximising
        the matrix trace:

        $$
        s^*(\\text{iy, ix}) = \\underset{s}{\\operatorname{argmax}}\\;
                              \\operatorname{tr}\\!\\bigl(R^T U^{(s)}\\bigr)
        $$

        This is equivalent to minimising the geodesic misorientation angle
        $\\omega = \\arccos\\!\\left(\\tfrac{\\operatorname{tr}(R^T U^{(s)})-1}{2}\\right)$,
        which lies in $[0°, 180°]$.  The computation is vectorised over
        all map pixels simultaneously.

        **Reference orientation**
        When `reference=None` the target is computed as the quaternion mean
        of all *valid* (non-NaN) pixels in the map:

        1. Convert each U matrix to a unit quaternion.
        2. Flip every quaternion to the same hemisphere as the first
           (`q ← −q` if `q · q₀ < 0`), so the average is not pulled
           toward zero by sign cancellation.
        3. Average the flipped quaternions and re-normalise.
        4. Convert back to a rotation matrix.

        This gives a robust, bias-free estimate of the "dominant" orientation
        in the map, which is almost always sufficient.  A custom reference can
        be supplied when the grain of interest occupies a minority of pixels.

        **Strain tensor rotation**
        When strain data are present (`self.strain_tensor[grain]` contains
        finite values), the strain tensor at corrected pixels is transformed
        consistently with the symmetry operation:

        $$
        \\boldsymbol{\\varepsilon}' = S^T \\boldsymbol{\\varepsilon}\\, S
        $$

        where **S** = `ops[s*]`.  The Voigt representation is rebuilt from
        the updated full tensor and written back into `self.strain_voigt`.

        !!! note
            This rotation makes the *representation* of the strain tensor
            consistent with the new orientation convention but does not
            change the physical strain state.  Pixels that were corrected
            ideally should be re-refined starting from the corrected **U**;
            the rotated tensor is an approximation that is exact only in the
            limit of a purely rotational symmetry operation.

        Args:
            grain (int): Grain slot to correct.  Default `0`.
            symmetry (str): Crystal point-group symmetry.  One of `'cubic'`,
                `'hexagonal'`, `'tetragonal'`, `'orthorhombic'`.
                Default `'cubic'`.
            reference ((3, 3) ndarray or None): Target orientation matrix **R**.  Every pixel will be mapped to
                the symmetry-equivalent closest to this matrix.
                `None` (default) uses the quaternion mean of all valid pixels.

        Returns:
            changed ((ny, nx) bool ndarray): `True` at positions where a different symmetry equivalent was
                selected (i.e. where the operator index `s* ≠ 0`).

            **See also**
            :meth:`merge` (combine fits from multiple reference grains into one): best-grain map.
            :meth:`apply_merge` (register the merged selection as a new grain slot.):
            :meth:`_symmetry_ops` (returns the rotation matrices for a given): crystal point-group symmetry.

        Note:
        `reduce_to_fundamental_zone` modifies `self.U[grain]` and
        `self.strain_tensor[grain]` / `self.strain_voigt[grain]` **in
        place**.  Run it before calling :meth:`merge` or :meth:`apply_merge`
        if you want the merged grain to also benefit from the correction.
"""
        ops = self._symmetry_ops(symmetry)          # (N_sym, 3, 3)
        U   = self.U[grain]                         # (ny, nx, 3, 3)
        valid = ~np.any(np.isnan(U), axis=(-2, -1)) # (ny, nx) bool

        # ── compute reference orientation ─────────────────────────────────────
        if reference is None:
            valid_U = U[valid]                      # (M, 3, 3)
            if len(valid_U) == 0:
                return np.zeros((self.ny, self.nx), dtype=bool)
            # Quaternion mean (sign-flip to consistent hemisphere)
            rots = Rotation.from_matrix(valid_U)
            q    = rots.as_quat()                   # (M, 4)  xyzw
            q    = np.where(((q @ q[0]) < 0)[:, None], -q, q) # flip to same hemisphere
            q_mean = q.mean(axis=0)
            q_mean /= np.linalg.norm(q_mean)
            reference = Rotation.from_quat(q_mean).as_matrix()

        ref = np.asarray(reference, dtype=float)

        # ── for each pixel pick the equivalent closest to reference ───────────
        # U_equiv[i] = U @ ops[i]  → shape (N_sym, ny, nx, 3, 3)
        U_equiv = np.einsum("...ij,kjl->k...il", U, ops)  # (N_sym, ny, nx, 3, 3)

        # Misorientation angle: trace(ref^T @ U') in [-1, 3]
        # angle = arccos((trace - 1) / 2)  → minimise angle = maximise trace
        traces = np.einsum("ij,k...jl->k...li", ref, U_equiv)
        # traces shape is wrong — let me compute correctly:
        # trace(ref^T @ U_equiv[k,iy,ix]) = sum_i (ref^T)_ii' U_equiv[k,iy,ix]_i'i
        # = sum_i ref_i'i U_equiv[k,iy,ix]_i'i  ... simpler:
        # = einsum("ji,k...ji->k...", ref, U_equiv)
        traces = np.einsum("ji,k...ji->k...", ref, U_equiv)  # (N_sym, ny, nx)
        best   = np.argmax(traces, axis=0)                   # (ny, nx)

        # ── apply best symmetry op — vectorised ──────────────────────────────
        changed = valid & (best != 0)

        if changed.any():
            # U_new[iy,ix] = U[iy,ix] @ ops[best[iy,ix]]
            # Gather: ops_sel[iy,ix] = ops[best[iy,ix]]  → (ny,nx,3,3)
            ops_sel = ops[best]                          # (ny, nx, 3, 3)
            U_new   = np.einsum("...ij,...jk->...ik", U, ops_sel)
            self.U[grain] = np.where(changed[..., None, None], U_new, U)

            # Strain: ε' = Sᵀ ε S  (only where strain exists)
            eps     = self.strain_tensor[grain]          # (ny, nx, 3, 3)
            has_eps = changed & ~np.any(np.isnan(eps), axis=(-2, -1))
            if has_eps.any():
                eps_new = np.einsum(
                    "...ki,...kl,...lj->...ij", ops_sel, eps, ops_sel
                )   # Sᵀ ε S  (ops_sel rows are the columns of S, so "ki" = Sᵀ)
                self.strain_tensor[grain] = np.where(
                    has_eps[..., None, None], eps_new, eps
                )
                # Repack strain_voigt
                e = self.strain_tensor[grain]
                sv = np.stack([
                    e[..., 0, 0], e[..., 1, 1], e[..., 2, 2],
                    e[..., 0, 1], e[..., 0, 2], e[..., 1, 2],
                ], axis=-1)
                self.strain_voigt[grain] = np.where(
                    has_eps[..., None], sv, self.strain_voigt[grain]
                )

        n_changed = int(changed.sum())
        print(
            f"reduce_to_fundamental_zone: {n_changed} / {int(valid.sum())} "
            f"pixels reoriented  [grain={grain}, symmetry={symmetry!r}]"
        )
        return changed

    # ── Derived quantities ────────────────────────────────────────────────────

    def euler_map(
        self,
        grain: int,
        convention: str = "ZXZ",
    ) -> np.ndarray:
        """
        Euler angles for every map point.

        Returns:
            angles ((ny, nx, 3) ndarray, degrees.): `NaN` where no fit exists.
"""
        angles = np.full((self.ny, self.nx, 3), np.nan)
        for iy in range(self.ny):
            for ix in range(self.nx):
                U = self.U[grain, iy, ix]
                if not np.any(np.isnan(U)):
                    angles[iy, ix] = Rotation.from_matrix(U).as_euler(
                        convention, degrees=True
                    )
        return angles

    def misorientation_map(
        self,
        grain: int,
        reference: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Misorientation angle (degrees) relative to a reference.

        Args:
            reference ((3, 3) ndarray or None): Reference orientation.  Defaults to `U_ref[grain]` if available,
                otherwise the mean of all fitted points.
"""
        if reference is None:
            if self.n_grains > grain:
                reference = self.U_ref[grain]
            else:
                fitted = self.U[grain][~np.any(
                    np.isnan(self.U[grain]), axis=(-2, -1)
                )]
                if len(fitted) == 0:
                    return np.full((self.ny, self.nx), np.nan)
                reference = fitted[0]

        misor = np.full((self.ny, self.nx), np.nan)
        for iy in range(self.ny):
            for ix in range(self.nx):
                U = self.U[grain, iy, ix]
                if not np.any(np.isnan(U)):
                    dR = U @ reference.T
                    misor[iy, ix] = np.degrees(
                        Rotation.from_matrix(dR).magnitude()
                    )
        return misor

    def kam_map(
        self,
        grain: int = 0,
        kernel: int = 1,
        max_misor_deg: float | None = 5.0,
    ) -> np.ndarray:
        """
        Kernel Average Misorientation (KAM) map.

        For each fitted pixel the misorientation angle to every fitted
        neighbour within a square kernel of half-size *kernel* is computed,
        and the average of those angles is stored.  Pairs whose misorientation
        exceeds *max_misor_deg* are excluded so that grain-boundary pixels do
        not inflate the KAM inside grains.

        Args:
            grain (int): Grain index (0-based).  Default `0`.
            kernel (int): Half-size of the square neighbourhood in pixels.  `1` uses all
                8 immediate neighbours (3×3 kernel excluding the centre); `2`
                uses a 5×5 neighbourhood, and so on.  Default `1`.
            max_misor_deg (float or None): Neighbour pairs with misorientation above this value are ignored.
                Set to `None` to include all neighbours regardless of angle.
                Default `5.0`°.

        Returns:
            kam ((ny, nx) ndarray): KAM values in degrees.  `NaN` at unfitted points or points
                with no valid neighbours.
"""
        U     = self.U[grain]                                    # (ny, nx, 3, 3)
        valid = ~np.any(np.isnan(U), axis=(-2, -1))             # (ny, nx) bool

        offsets = [
            (dy, dx)
            for dy in range(-kernel, kernel + 1)
            for dx in range(-kernel, kernel + 1)
            if (dy, dx) != (0, 0)
        ]

        kam = np.full((self.ny, self.nx), np.nan)
        for iy in range(self.ny):
            for ix in range(self.nx):
                if not valid[iy, ix]:
                    continue
                U0     = U[iy, ix]
                angles = []
                for dy, dx in offsets:
                    jy, jx = iy + dy, ix + dx
                    if (0 <= jy < self.ny and 0 <= jx < self.nx
                            and valid[jy, jx]):
                        dR    = U0 @ U[jy, jx].T
                        angle = np.degrees(
                            Rotation.from_matrix(dR).magnitude()
                        )
                        if max_misor_deg is None or angle <= max_misor_deg:
                            angles.append(angle)
                if angles:
                    kam[iy, ix] = float(np.mean(angles))
        return kam

    # ── Visualisation ─────────────────────────────────────────────────────────

    _SCALAR_QUANTITIES = {
        "rms_px", "mean_px", "match_rate", "cost", "n_matched", "misorientation",
        "euler_phi1", "euler_Phi", "euler_phi2",
    }

    def plot_map(
        self,
        quantity: str = "match_rate",
        grain: int = 0,
        *,
        ax: "plt.Axes | None" = None,
        cmap: str | None = None,
        vmin: float | None = None,
        vmax: float | None = None,
        motor_x: str | None = None,
        motor_y: str | None = None,
        motor_units: "dict | None" = None,
        title: str | None = None,
        figsize: tuple = (6, 5),
        colorbar: bool = True,
    ) -> tuple:
        """
        Plot a scalar map for a given grain.

        Args:
            quantity (str): One of `'match_rate'`, `'rms_px'`, `'cost'`,
                `'n_matched'`, `'misorientation'`,
                `'euler_phi1'`, `'euler_Phi'`, `'euler_phi2'`.
            grain (int): Grain index (0-based).
            motor_x, motor_y (str or None): Motor names to use as axis tick labels (from `self.motors`).
                If `None`, integer pixel indices are shown.
"""
        # ── build data array ──────────────────────────────────────────────────
        if isinstance(quantity, np.ndarray):
            data  = quantity
            label = title or ""
            cmap  = cmap or "viridis"
        elif quantity == "match_rate":
            data = self.match_rate[grain]
            label = "Match rate"
            cmap  = cmap or "viridis"
        elif quantity == "rms_px":
            data = self.rms_px[grain]
            label = "RMS (px)"
            cmap  = cmap or "plasma_r"
        elif quantity == "mean_px":
            data = self.mean_px[grain]
            label = "Mean dev (px)"
            cmap  = cmap or "plasma_r"
        elif quantity == "cost":
            data = self.cost[grain]
            label = "Cost"
            cmap  = cmap or "plasma_r"
        elif quantity == "n_matched":
            raw = self.n_matched[grain].astype(float)
            raw[raw < 0] = np.nan
            data = raw
            label = "N matched"
            cmap  = cmap or "viridis"
        elif quantity == "misorientation":
            data = self.misorientation_map(grain)
            label = "Misorientation (°)"
            cmap  = cmap or "RdYlGn_r"
        elif quantity in ("euler_phi1", "euler_Phi", "euler_phi2"):
            euler = self.euler_map(grain)
            idx   = {"euler_phi1": 0, "euler_Phi": 1, "euler_phi2": 2}[quantity]
            data  = euler[:, :, idx]
            label = {"euler_phi1": "φ₁ (°)", "euler_Phi": "Φ (°)",
                     "euler_phi2": "φ₂ (°)"}[quantity]
            cmap  = cmap or "hsv"
        else:
            raise ValueError(
                f"Unknown quantity {quantity!r}. "
                f"Choose from: {sorted(self._SCALAR_QUANTITIES)}"
            )

        # ── axis extent and labels ────────────────────────────────────────────
        mu = motor_units or {}
        mx = self.motors.get(motor_x) if motor_x else None
        my = self.motors.get(motor_y) if motor_y else None

        if mx is not None and my is not None:
            extent = [mx[0, 0], mx[0, -1], my[-1, 0], my[0, 0]]
            xu = mu.get(motor_x, "")
            yu = mu.get(motor_y, "")
            xlabel = f"{motor_x} ({xu})" if xu else motor_x
            ylabel = f"{motor_y} ({yu})" if yu else motor_y
        else:
            extent = [0, self.nx, self.ny, 0]
            xlabel = "column (ix)"
            ylabel = "row (iy)"

        # ── figure ────────────────────────────────────────────────────────────
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        im = ax.imshow(
            data,
            origin="upper",
            extent=extent,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
            aspect="auto",
        )

        if colorbar:
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cb.set_label(label, fontsize=9)

        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(
            title or f"Grain {grain + 1}  —  {label}",
            fontsize=10,
        )
        fig.tight_layout()
        return fig, ax

    def plot_mean_px(
        self,
        *,
        grains: "list[int] | None" = None,
        ax: "plt.Axes | None" = None,
        cmap: str | None = None,
        vmin: float | None = None,
        vmax: float | None = None,
        motor_x: str | None = None,
        motor_y: str | None = None,
        motor_units: "dict | None" = None,
        title: str | None = None,
        figsize: tuple | None = None,
        colorbar: bool = True,
        share_scale: bool = True,
    ) -> tuple:
        """
        Plot the mean pixel deviation map for one or all grains.

        When more than one grain is shown a figure with one subplot per grain
        is created automatically.  Pass `ax` to place a single-grain plot on
        an existing axes.

        Args:
            grains (list[int] or None): Grain indices to plot.  `None` plots all grains.
            ax (Axes or None): If provided, only the first (or only) grain is plotted here.
            cmap (str or None): Colormap.  Defaults to `'plasma_r'`.
            vmin, vmax (float or None): Color scale limits.  If `share_scale` is `True` and both are
                `None`, the limits are computed jointly from all shown grains.
            motor_x, motor_y (str or None): Motor names for axis labels.
            motor_units (dict or None): Units for motor axes, e.g. `{'pz': 'mm', 'py': 'mm'}`.
            share_scale (bool): If `True` (default), all subplots share the same `vmin`/`vmax`.
"""
        grains = list(range(self.n_grains)) if grains is None else list(grains)
        cmap   = cmap or "plasma_r"

        # ── motor extent ──────────────────────────────────────────────────────
        mu = motor_units or {}
        mx = self.motors.get(motor_x) if motor_x else None
        my = self.motors.get(motor_y) if motor_y else None

        if mx is not None and my is not None:
            extent = [mx[0, 0], mx[0, -1], my[-1, 0], my[0, 0]]
            xu = mu.get(motor_x, "")
            yu = mu.get(motor_y, "")
            xlabel = f"{motor_x} ({xu})" if xu else motor_x
            ylabel = f"{motor_y} ({yu})" if yu else motor_y
        else:
            extent = [0, self.nx, self.ny, 0]
            xlabel = "column (ix)"
            ylabel = "row (iy)"

        # ── shared color limits ───────────────────────────────────────────────
        if share_scale and vmin is None and vmax is None:
            all_vals = np.concatenate([
                self.mean_px[gi][np.isfinite(self.mean_px[gi])].ravel()
                for gi in grains
            ])
            if all_vals.size > 0:
                vmin = float(np.nanmin(all_vals))
                vmax = float(np.nanmax(all_vals))

        # ── single-axes shortcut ──────────────────────────────────────────────
        if ax is not None or len(grains) == 1:
            gi = grains[0]
            if ax is None:
                fig, ax = plt.subplots(figsize=figsize or (6, 5))
            else:
                fig = ax.get_figure()
            im = ax.imshow(
                self.mean_px[gi],
                origin="upper", extent=extent,
                cmap=cmap, vmin=vmin, vmax=vmax,
                interpolation="nearest", aspect="auto",
            )
            if colorbar:
                cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cb.set_label("Mean dev (px)", fontsize=9)
            ax.set_xlabel(xlabel, fontsize=9)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.set_title(title or f"Grain {gi + 1}  —  Mean pixel deviation", fontsize=10)
            fig.tight_layout()
            return fig, ax

        # ── multi-grain figure ────────────────────────────────────────────────
        ncols  = len(grains)
        fsize  = figsize or (5 * ncols, 4.5)
        fig, axes_arr = plt.subplots(1, ncols, figsize=fsize,
                                     squeeze=False)
        axes_arr = axes_arr[0]

        for col, gi in enumerate(grains):
            a  = axes_arr[col]
            im = a.imshow(
                self.mean_px[gi],
                origin="upper", extent=extent,
                cmap=cmap, vmin=vmin, vmax=vmax,
                interpolation="nearest", aspect="auto",
            )
            if colorbar:
                cb = fig.colorbar(im, ax=a, fraction=0.046, pad=0.04)
                if col == len(grains) - 1:
                    cb.set_label("Mean dev (px)", fontsize=9)
            a.set_xlabel(xlabel, fontsize=9)
            a.set_ylabel(ylabel if col == 0 else "", fontsize=9)
            a.set_title(
                title or f"Grain {gi + 1}  —  Mean pixel deviation",
                fontsize=10,
            )

        fig.tight_layout()
        return fig, axes_arr

    def plot_kam(
        self,
        grain: int = 0,
        kernel: int = 1,
        max_misor_deg: float | None = 5.0,
        *,
        ax: "plt.Axes | None" = None,
        cmap: str | None = None,
        vmin: float | None = None,
        vmax: float | None = None,
        motor_x: str | None = None,
        motor_y: str | None = None,
        motor_units: "dict | None" = None,
        title: str | None = None,
        figsize: tuple | None = None,
        colorbar: bool = True,
    ) -> tuple:
        """
        Plot the Kernel Average Misorientation (KAM) map.

        Calls :meth:`kam_map` and displays the result.  The KAM value at
        each pixel is the mean misorientation angle (°) to its neighbours
        within a square kernel of half-size *kernel*, excluding pairs above
        *max_misor_deg* (grain boundaries).

        Args:
            grain (int): Grain index (0-based).  Default `0`.
            kernel (int): Half-size of the square neighbourhood in pixels.  `1` → 8
                immediate neighbours; `2` → 24 neighbours in a 5×5 window.
                Default `1`.
            max_misor_deg (float or None): Neighbour pairs with misorientation above this threshold are
                excluded from the average.  `None` includes all neighbours.
                Default `5.0`°.
            ax (Axes or None): Existing axes to draw on.  If `None` a new figure is created.
            cmap (str or None): Colormap.  Defaults to `'inferno'`.
            vmin, vmax (float or None): Color scale limits.  `None` uses the data range.
            motor_x, motor_y (str or None): Motor names for axis labels (from `self.motors`).
            motor_units (dict or None): Units per motor, e.g. `{'pz': 'mm', 'py': 'mm'}`.
            title (str or None): Axes title.  Auto-generated if `None`.
            figsize (tuple or None): Figure size.  Defaults to `(6, 5)`.
            colorbar (bool): Whether to add a colorbar.  Default `True`.

        Returns:
            fig (Figure):
            ax (Axes):
"""
        data = self.kam_map(grain, kernel=kernel, max_misor_deg=max_misor_deg)
        cmap = cmap or "inferno"

        mu = motor_units or {}
        mx = self.motors.get(motor_x) if motor_x else None
        my = self.motors.get(motor_y) if motor_y else None

        if mx is not None and my is not None:
            extent = [mx[0, 0], mx[0, -1], my[-1, 0], my[0, 0]]
            xu = mu.get(motor_x, "")
            yu = mu.get(motor_y, "")
            xlabel = f"{motor_x} ({xu})" if xu else motor_x
            ylabel = f"{motor_y} ({yu})" if yu else motor_y
        else:
            extent = [0, self.nx, self.ny, 0]
            xlabel = "column (ix)"
            ylabel = "row (iy)"

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize or (6, 5))
        else:
            fig = ax.get_figure()

        im = ax.imshow(
            data,
            origin="upper", extent=extent,
            cmap=cmap, vmin=vmin, vmax=vmax,
            interpolation="nearest", aspect="auto",
        )
        if colorbar:
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cb.set_label("KAM (°)", fontsize=9)

        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        _t = (
            title or
            f"Grain {grain + 1}  —  KAM  "
            f"(kernel={kernel}, max={max_misor_deg}°)"
        )
        ax.set_title(_t, fontsize=10)
        fig.tight_layout()
        return fig, ax

    # ── strain component map ──────────────────────────────────────────────────

    _STRAIN_INDICES = {
        "e_xx": (0, 0), "e_yy": (1, 1), "e_zz": (2, 2),
        "e_xy": (0, 1), "e_xz": (0, 2), "e_yz": (1, 2),
    }
    _STRAIN_LABELS = {
        "e_xx": "ε_xx", "e_yy": "ε_yy", "e_zz": "ε_zz",
        "e_xy": "ε_xy", "e_xz": "ε_xz", "e_yz": "ε_yz",
    }

    def _strain_component_map(
        self,
        component: str,
        grain: int,
        frame: str,
        sample_tilt_deg: float,
        sample_tilt_axis: str,
    ) -> np.ndarray:
        """Return (ny, nx) array of the requested strain component."""
        i, j = self._STRAIN_INDICES[component]
        eps = self.strain_tensor[grain]   # (ny, nx, 3, 3)
        U   = self.U[grain]               # (ny, nx, 3, 3)

        if frame == "crystal":
            data = eps[..., i, j]
        elif frame == "lab":
            # ε_lab = U @ ε @ U^T  (vectorised over map points)
            eps_t = np.einsum("...ik,...kl,...jl->...ij", U, eps, U)
            data  = eps_t[..., i, j]
        elif frame == "sample":
            # ε_lab first, then rotate by R_s about the chosen lab axis
            R_s        = Rotation.from_euler(
                sample_tilt_axis, sample_tilt_deg, degrees=True
            ).as_matrix()
            eps_lab    = np.einsum("...ik,...kl,...jl->...ij", U, eps, U)
            eps_sample = np.einsum("ik,...kl,jl->...ij", R_s, eps_lab, R_s)
            data = eps_sample[..., i, j]
        else:
            raise ValueError(
                f"Unknown frame {frame!r}. Choose 'crystal', 'lab', or 'sample'."
            )
        return data

    _STRAIN_DEV_LABELS = {
        "e_xx": "ε'_xx", "e_yy": "ε'_yy", "e_zz": "ε'_zz",
        "e_xy": "ε'_xy", "e_xz": "ε'_xz", "e_yz": "ε'_yz",
    }

    def _deviatoric_component_map(
        self,
        component: str,
        grain: int,
        frame: str,
        sample_tilt_deg: float,
        sample_tilt_axis: str,
    ) -> np.ndarray:
        """Return (ny, nx) array of the requested deviatoric strain component."""
        i, j = self._STRAIN_INDICES[component]
        eps_dev = self.strain_tensor_deviatoric[grain]   # (ny, nx, 3, 3)
        U       = self.U[grain]                          # (ny, nx, 3, 3)

        if frame == "crystal":
            data = eps_dev[..., i, j]
        elif frame == "lab":
            eps_t = np.einsum("...ik,...kl,...jl->...ij", U, eps_dev, U)
            data  = eps_t[..., i, j]
        elif frame == "sample":
            R_s        = Rotation.from_euler(
                sample_tilt_axis, sample_tilt_deg, degrees=True
            ).as_matrix()
            eps_lab    = np.einsum("...ik,...kl,...jl->...ij", U, eps_dev, U)
            eps_sample = np.einsum("ik,...kl,jl->...ij", R_s, eps_lab, R_s)
            data = eps_sample[..., i, j]
        else:
            raise ValueError(
                f"Unknown frame {frame!r}. Choose 'crystal', 'lab', or 'sample'."
            )
        return data

    def plot_deviatoric(
        self,
        component: str = "e_xx",
        grain: int = 0,
        *,
        frame: str = "crystal",
        sample_tilt_deg: float = -40.0,
        sample_tilt_axis: str = "y",
        ax: "plt.Axes | None" = None,
        cmap: str | None = None,
        vmin: float | None = None,
        vmax: float | None = None,
        motor_x: str | None = None,
        motor_y: str | None = None,
        motor_units: "dict | None" = None,
        title: str | None = None,
        figsize: tuple = (6, 5),
        colorbar: bool = True,
    ) -> tuple:
        """
        Plot a single deviatoric strain component for a given grain.

        The deviatoric strain is the traceless part of the full strain tensor:

        .. math::

            \\boldsymbol{\\varepsilon}' =
            \\boldsymbol{\\varepsilon} - \\frac{\\operatorname{tr}(\\boldsymbol{\\varepsilon})}{3}\\,\\mathbf{I}

        Because white-beam Laue is insensitive to hydrostatic strain, the
        deviatoric components are the physically meaningful quantity for
        inter-method comparisons.

        Args:
            component (str): One of ``'e_xx'``, ``'e_yy'``, ``'e_zz'``,
                ``'e_xy'``, ``'e_xz'``, ``'e_yz'``.
            grain (int): Grain index (0-based).
            frame (str): Reference frame — ``'crystal'``, ``'lab'``, or ``'sample'``.
            sample_tilt_deg (float): Tilt angle (degrees) from lab to sample frame.  Default ``-40``.
            sample_tilt_axis (str): Lab axis of the tilt rotation.  Default ``'y'``.
            ax (Axes or None): Existing axes to draw into.  A new figure is created when ``None``.
            cmap (str or None): Colormap name.  Defaults to ``'RdBu_r'``.
            vmin, vmax (float or None): Colour scale limits.
            motor_x, motor_y (str or None): Motor names for physical-coordinate axis labels.
            motor_units (dict or None): Units per motor name, e.g. ``{'pz': 'mm'}``.
            title (str or None): Axes title.  Auto-generated when ``None``.
            figsize (tuple): Figure size in inches.  Default ``(6, 5)``.
            colorbar (bool): Whether to add a colorbar.  Default ``True``.

        Returns:
            tuple: ``(fig, ax)``
        """
        if component not in self._STRAIN_INDICES:
            raise ValueError(
                f"Unknown component {component!r}. "
                f"Choose from: {sorted(self._STRAIN_INDICES)}"
            )

        data  = self._deviatoric_component_map(
            component, grain, frame, sample_tilt_deg, sample_tilt_axis
        )
        label = self._STRAIN_DEV_LABELS[component]
        cmap  = cmap or "RdBu_r"

        mx = self.motors.get(motor_x) if motor_x else None
        my = self.motors.get(motor_y) if motor_y else None
        mu = motor_units or {}

        if mx is not None and my is not None:
            extent = [mx[0, 0], mx[0, -1], my[-1, 0], my[0, 0]]
            xu = mu.get(motor_x, "")
            yu = mu.get(motor_y, "")
            xlabel = f"{motor_x} ({xu})" if xu else motor_x
            ylabel = f"{motor_y} ({yu})" if yu else motor_y
        else:
            extent = [0, self.nx, self.ny, 0]
            xlabel = "column (ix)"
            ylabel = "row (iy)"

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        im = ax.imshow(
            data,
            origin="upper",
            extent=extent,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
            aspect="auto",
        )

        if colorbar:
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cb.set_label(label, fontsize=9)

        _frame_label = {
            "crystal": "crystal frame",
            "lab":     "lab frame",
            "sample":  f"sample frame ({sample_tilt_deg:+.0f}° about {sample_tilt_axis})",
        }[frame]
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(
            title or f"Grain {grain + 1}  —  {label} (deviatoric)  [{_frame_label}]",
            fontsize=10,
        )
        fig.tight_layout()
        return fig, ax

    def plot_strain_component(
        self,
        component: str = "e_xx",
        grain: int = 0,
        *,
        frame: str = "crystal",
        sample_tilt_deg: float = -40.0,
        sample_tilt_axis: str = "y",
        ax: "plt.Axes | None" = None,
        cmap: str | None = None,
        vmin: float | None = None,
        vmax: float | None = None,
        motor_x: str | None = None,
        motor_y: str | None = None,
        motor_units: "dict | None" = None,
        title: str | None = None,
        figsize: tuple = (6, 5),
        colorbar: bool = True,
    ) -> tuple:
        """
        Plot a single strain-tensor component for a given grain.

        Args:
            component (str): One of `'e_xx'`, `'e_yy'`, `'e_zz'`,
                `'e_xy'`, `'e_xz'`, `'e_yz'`.
            grain (int): Grain index (0-based).
            frame (str): Reference frame for the strain tensor:

            `'crystal'`
                As fitted — components in the crystal coordinate system.
            `'lab'`
                Rotated to the lab frame via `ε_lab = U @ ε @ Uᵀ`.
            `'sample'`
                Lab frame further rotated by *sample_tilt_deg* about
                *sample_tilt_axis* (default −40° about Y).

            sample_tilt_deg (float): Tilt angle (degrees) from lab to sample frame.  Default `-40`.
            sample_tilt_axis (str): Lab axis of the tilt rotation (`'x'`, `'y'`, or `'z'`).
                Default `'y'`.
            motor_x, motor_y (str or None): Motor names to use as axis tick labels.
"""
        if component not in self._STRAIN_INDICES:
            raise ValueError(
                f"Unknown component {component!r}. "
                f"Choose from: {sorted(self._STRAIN_INDICES)}"
            )

        data  = self._strain_component_map(
            component, grain, frame, sample_tilt_deg, sample_tilt_axis
        )
        label = self._STRAIN_LABELS[component]
        cmap  = cmap or "RdBu_r"

        # ── axis extent ───────────────────────────────────────────────────────
        mx = self.motors.get(motor_x) if motor_x else None
        my = self.motors.get(motor_y) if motor_y else None
        mu = motor_units or {}

        if mx is not None and my is not None:
            extent = [mx[0, 0], mx[0, -1], my[-1, 0], my[0, 0]]
            xu = mu.get(motor_x, "")
            yu = mu.get(motor_y, "")
            xlabel = f"{motor_x} ({xu})" if xu else motor_x
            ylabel = f"{motor_y} ({yu})" if yu else motor_y
        else:
            extent = [0, self.nx, self.ny, 0]
            xlabel = "column (ix)"
            ylabel = "row (iy)"

        # ── figure ────────────────────────────────────────────────────────────
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        im = ax.imshow(
            data,
            origin="upper",
            extent=extent,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
            aspect="auto",
        )

        if colorbar:
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cb.set_label(label, fontsize=9)

        _frame_label = {
            "crystal": "crystal frame",
            "lab":     "lab frame",
            "sample":  f"sample frame ({sample_tilt_deg:+.0f}° about {sample_tilt_axis})",
        }[frame]
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(
            title or f"Grain {grain + 1}  —  {label}  [{_frame_label}]",
            fontsize=10,
        )
        fig.tight_layout()
        return fig, ax

    def plot_deviatoric_panel(
        self,
        grain: int = 0,
        *,
        frame: str = "crystal",
        sample_tilt_deg: float = -40.0,
        sample_tilt_axis: str = "y",
        cmap: str | None = None,
        vmin: float | None = None,
        vmax: float | None = None,
        symmetric_clim: bool = True,
        motor_x: str | None = None,
        motor_y: str | None = None,
        motor_units: "dict | None" = None,
        figsize: tuple | None = None,
        title: str | None = None,
        colorbar: bool = True,
    ) -> tuple:
        """
        Plot all six deviatoric strain components in a 2×3 panel.

        Components are arranged as::

            ε'_xx  ε'_yy  ε'_zz
            ε'_xy  ε'_xz  ε'_yz

        By default the colour scale is symmetric around zero and shared
        across all six panels (``symmetric_clim=True``), so that the maps
        are directly comparable.

        Args:
            grain (int): Grain index (0-based).
            frame (str): Reference frame — ``'crystal'``, ``'lab'``, or ``'sample'``.
            sample_tilt_deg (float): Tilt angle (degrees) from lab to sample frame.  Default ``-40``.
            sample_tilt_axis (str): Lab axis of the tilt rotation.  Default ``'y'``.
            cmap (str or None): Colormap.  Defaults to ``'RdBu_r'``.
            vmin, vmax (float or None): Shared colour-scale limits.  When both are
                ``None`` and ``symmetric_clim=True``, limits are set to
                ``±max(|ε'|)`` over all six components.  When either is given
                explicitly, ``symmetric_clim`` is ignored.
            symmetric_clim (bool): Auto-set ``vmin = -vmax`` from the data.
                Default ``True``.
            motor_x, motor_y (str or None): Motor names for physical-coordinate axis labels.
            motor_units (dict or None): Units per motor name, e.g. ``{'pz': 'mm'}``.
            figsize (tuple or None): Figure size in inches.  Defaults to ``(12, 7)``.
            title (str or None): Figure suptitle.  Auto-generated when ``None``.
            colorbar (bool): Add a single shared colorbar on the right.  Default ``True``.

        Returns:
            tuple: ``(fig, axes)`` where *axes* is a ``(2, 3)`` ndarray of Axes.
        """
        _order = ["e_xx", "e_yy", "e_zz", "e_xy", "e_xz", "e_yz"]
        cmap = cmap or "RdBu_r"

        mx = self.motors.get(motor_x) if motor_x else None
        my = self.motors.get(motor_y) if motor_y else None
        mu = motor_units or {}

        if mx is not None and my is not None:
            extent = [mx[0, 0], mx[0, -1], my[-1, 0], my[0, 0]]
            xu = mu.get(motor_x, "")
            yu = mu.get(motor_y, "")
            xlabel = f"{motor_x} ({xu})" if xu else motor_x
            ylabel = f"{motor_y} ({yu})" if yu else motor_y
        else:
            extent = [0, self.nx, self.ny, 0]
            xlabel = "column (ix)"
            ylabel = "row (iy)"

        # ── collect data and determine shared colour limits ────────────────────
        maps = {
            comp: self._deviatoric_component_map(
                comp, grain, frame, sample_tilt_deg, sample_tilt_axis
            )
            for comp in _order
        }

        if vmin is None and vmax is None and symmetric_clim:
            finite = np.concatenate([m[np.isfinite(m)].ravel() for m in maps.values()])
            if finite.size:
                vmax = float(np.abs(finite).max())
            vmin = -vmax if vmax is not None else None
        elif vmin is None and vmax is not None:
            vmin = -vmax if symmetric_clim else None
        elif vmax is None and vmin is not None:
            vmax = -vmin if symmetric_clim else None

        # ── figure ────────────────────────────────────────────────────────────
        fig, axes = plt.subplots(
            2, 3,
            figsize=figsize or (12, 7),
            squeeze=False,
        )

        im = None
        for ax, comp in zip(axes.ravel(), _order):
            im = ax.imshow(
                maps[comp],
                origin="upper",
                extent=extent,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
                aspect="auto",
            )
            ax.set_title(self._STRAIN_DEV_LABELS[comp], fontsize=10)
            ax.set_xlabel(xlabel, fontsize=8)
            ax.set_ylabel(ylabel, fontsize=8)

        if colorbar and im is not None:
            fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)

        _frame_label = {
            "crystal": "crystal frame",
            "lab":     "lab frame",
            "sample":  f"sample frame ({sample_tilt_deg:+.0f}° about {sample_tilt_axis})",
        }.get(frame, frame)
        fig.suptitle(
            title or f"Grain {grain + 1}  —  deviatoric strain  [{_frame_label}]",
            fontsize=11,
        )
        fig.tight_layout()
        return fig, axes

    # ── Stress analysis ───────────────────────────────────────────────────────

    # Code Voigt ordering: [xx=0, yy=1, zz=2, xy=3, xz=4, yz=5]
    # Standard crystallographic Voigt (xrayutilities cij):
    #                      [xx=0, yy=1, zz=2, yz=3, xz=4, xy=5]
    # Permutation between the two (its own inverse): [0,1,2,5,4,3]
    _VOIGT_REORDER = np.array([0, 1, 2, 5, 4, 3])

    _STRESS_INDICES = {
        "s_xx": (0, 0), "s_yy": (1, 1), "s_zz": (2, 2),
        "s_xy": (0, 1), "s_xz": (0, 2), "s_yz": (1, 2),
    }
    _STRESS_LABELS = {
        "s_xx": "σ_xx", "s_yy": "σ_yy", "s_zz": "σ_zz",
        "s_xy": "σ_xy", "s_xz": "σ_xz", "s_yz": "σ_yz",
    }
    _STRESS_ALL = ("s_xx", "s_yy", "s_zz", "s_xy", "s_xz", "s_yz")

    @staticmethod
    def _extract_cij(crystal, cij=None) -> np.ndarray:
        """
        Return the 6×6 Voigt stiffness matrix in GPa.

        **Resolution order**
        1. *cij* parameter if given directly.
        2. `crystal.cij` (xrayutilities stores this in Pa → converted to GPa).
        3. `crystal.cijkl` (4th-rank tensor in Pa) → Mandel/Voigt reduction.

        The returned matrix uses the standard crystallographic Voigt ordering
        `[xx, yy, zz, yz, xz, xy]`.
"""
        if cij is not None:
            return np.asarray(cij, dtype=float)

        # xrayutilities Crystal
        if hasattr(crystal, "cij"):
            raw = np.asarray(crystal.cij, dtype=float)
            if raw.shape == (6, 6):
                # xrayutilities returns Pa; convert to GPa
                scale = 1e-9 if raw.max() > 1e6 else 1.0
                return raw * scale

        # 4th-rank tensor fallback
        if hasattr(crystal, "cijkl"):
            cijkl = np.asarray(crystal.cijkl, dtype=float)
            # Mandel/Voigt contraction: (i,j) → Voigt index
            _v = {(0,0):0,(1,1):1,(2,2):2,(1,2):3,(2,1):3,(0,2):4,(2,0):4,(0,1):5,(1,0):5}
            mat = np.zeros((6, 6))
            for (i,j), a in _v.items():
                for (k,l), b in _v.items():
                    mat[a, b] = cijkl[i, j, k, l]
            scale = 1e-9 if mat.max() > 1e6 else 1.0
            return mat * scale

        raise AttributeError(
            f"Cannot find a 6×6 stiffness matrix on {type(crystal).__name__}. "
            "Pass cij explicitly as a (6,6) array in GPa."
        )

    def stress_voigt(
        self,
        crystal,
        grain: int = 0,
        *,
        cij: "np.ndarray | None" = None,
        frame: str = "crystal",
        sample_tilt_deg: float = -40.0,
        sample_tilt_axis: str = "y",
    ) -> np.ndarray:
        """
        Compute the Cauchy stress tensor in Voigt notation for every map point.

        Uses Hooke's law  **σ = C : ε**  where *C* is the 6×6 stiffness
        matrix extracted from *crystal* and *ε* is the fitted strain tensor
        stored in `self.strain_tensor[grain]`.

        Args:
            crystal (Crystal or LayeredCrystal): Source of elastic constants.  The stiffness matrix is extracted
                automatically (see :meth:`_extract_cij`).
            grain (int): Grain index (0-based).
            cij ((6, 6) array or None): Override the stiffness matrix (GPa, standard Voigt ordering
                `[xx, yy, zz, yz, xz, xy]`).  `None` reads from *crystal*.
            frame (str): Reference frame of the returned stress tensor.

            `'crystal'`
                Components referred to the crystal axes (as fitted).
            `'lab'`
                Rotated to the lab frame via `σ_lab = U σ_crystal Uᵀ`.
            `'sample'`
                Lab frame further rotated by *sample_tilt_deg* about
                *sample_tilt_axis*.

        Returns:
            stress ((ny, nx, 6) ndarray, GPa): Stress in code Voigt ordering `[s_xx, s_yy, s_zz, s_xy, s_xz, s_yz]`.
                `NaN` where strain data are absent.
"""
        C = self._extract_cij(crystal, cij)          # (6,6) GPa, std Voigt
        eps_code = self.strain_tensor[grain]          # (ny, nx, 3, 3)

        # Build (ny, nx, 6) engineering Voigt in standard ordering
        # Standard: [xx, yy, zz, yz, xz, xy]
        eps_std = np.full((*eps_code.shape[:2], 6), np.nan)
        eps_std[..., 0] = eps_code[..., 0, 0]
        eps_std[..., 1] = eps_code[..., 1, 1]
        eps_std[..., 2] = eps_code[..., 2, 2]
        eps_std[..., 3] = 2.0 * eps_code[..., 1, 2]  # engineering γ_yz
        eps_std[..., 4] = 2.0 * eps_code[..., 0, 2]  # engineering γ_xz
        eps_std[..., 5] = 2.0 * eps_code[..., 0, 1]  # engineering γ_xy

        # σ_std (ny,nx,6) = C (6,6) @ ε_std (ny,nx,6) via einsum
        sig_std = np.einsum("ij,...j->...i", C, eps_std)   # (ny, nx, 6)

        # Reorder back to code convention [xx,yy,zz,xy,xz,yz]
        sig_code = sig_std[..., self._VOIGT_REORDER]

        if frame in ("lab", "sample"):
            U = self.U[grain]                             # (ny, nx, 3, 3)
            R = U
            if frame == "sample":
                R_s = Rotation.from_euler(
                    sample_tilt_axis, sample_tilt_deg, degrees=True
                ).as_matrix()
                R = np.einsum("ij,...jk->...ik", R_s, U)

            # Rebuild (ny,nx,3,3) tensor from code Voigt, rotate, repack
            _idx = self._STRESS_INDICES
            sig_t = np.full((*eps_code.shape[:2], 3, 3), np.nan)
            for comp, (i, j) in _idx.items():
                v = sig_code[..., self._STRESS_ALL.index(comp)]
                sig_t[..., i, j] = v
                sig_t[..., j, i] = v

            sig_rot = np.einsum("...ik,...kl,...jl->...ij", R, sig_t, R)

            # Repack rotated tensor to code Voigt
            sig_code = np.full_like(sig_code, np.nan)
            for comp, (i, j) in _idx.items():
                sig_code[..., self._STRESS_ALL.index(comp)] = sig_rot[..., i, j]

        return sig_code

    def plot_stress_component(
        self,
        component: str = "s_xx",
        grain: int = 0,
        crystal = None,
        *,
        cij: "np.ndarray | None" = None,
        frame: str = "crystal",
        sample_tilt_deg: float = -40.0,
        sample_tilt_axis: str = "y",
        ax: "plt.Axes | None" = None,
        cmap: str | None = None,
        vmin: float | None = None,
        vmax: float | None = None,
        motor_x: str | None = None,
        motor_y: str | None = None,
        motor_units: "dict | None" = None,
        title: str | None = None,
        figsize: tuple = (6, 5),
        colorbar: bool = True,
        scale: float = 1e3,
    ) -> tuple:
        """
        Plot a single stress-tensor component for a given grain.

        Args:
            component (str): One of `'s_xx'`, `'s_yy'`, `'s_zz'`,
                `'s_xy'`, `'s_xz'`, `'s_yz'`.
            grain (int): Grain index (0-based).
            crystal (Crystal or LayeredCrystal or None): Source of elastic constants.  Required unless *cij* is given.
            cij ((6, 6) array or None): Override stiffness matrix (GPa, standard Voigt ordering).
            frame (str): `'crystal'`, `'lab'`, or `'sample'`.  Default
                `'crystal'`.
            sample_tilt_deg (float): Tilt angle (°) for sample-frame rotation.  Default `-40`.
            sample_tilt_axis (str): Lab axis of the tilt rotation.  Default `'y'`.
            scale (float): Multiply stress values before plotting.  Default `1e3`
                converts GPa → MPa.
            motor_x, motor_y, motor_units, ax, cmap, vmin, vmax,
            title, figsize, colorbar
                Same as :meth:`plot_strain_component`.
"""
        if component not in self._STRESS_INDICES:
            raise ValueError(
                f"Unknown component {component!r}. "
                f"Choose from: {sorted(self._STRESS_INDICES)}"
            )
        if crystal is None and cij is None:
            raise ValueError("Provide crystal or cij.")

        sig = self.stress_voigt(
            crystal, grain, cij=cij, frame=frame,
            sample_tilt_deg=sample_tilt_deg, sample_tilt_axis=sample_tilt_axis,
        )
        idx  = self._STRESS_ALL.index(component)
        data = sig[..., idx] * scale
        unit = "MPa" if abs(scale - 1e3) < 1 else "GPa" if scale == 1.0 else f"×{scale} GPa"
        label = f"{self._STRESS_LABELS[component]} ({unit})"
        cmap  = cmap or "RdBu_r"

        mx = self.motors.get(motor_x) if motor_x else None
        my = self.motors.get(motor_y) if motor_y else None
        mu = motor_units or {}

        if mx is not None and my is not None:
            extent = [mx[0, 0], mx[0, -1], my[-1, 0], my[0, 0]]
            xu = mu.get(motor_x, "")
            yu = mu.get(motor_y, "")
            xlabel = f"{motor_x} ({xu})" if xu else motor_x
            ylabel = f"{motor_y} ({yu})" if yu else motor_y
        else:
            extent = [0, self.nx, self.ny, 0]
            xlabel = "column (ix)"
            ylabel = "row (iy)"

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        im = ax.imshow(
            data, origin="upper", extent=extent,
            cmap=cmap, vmin=vmin, vmax=vmax,
            interpolation="nearest", aspect="auto",
        )
        if colorbar:
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cb.set_label(label, fontsize=9)

        _frame_label = {
            "crystal": "crystal frame",
            "lab":     "lab frame",
            "sample":  f"sample frame ({sample_tilt_deg:+.0f}° about {sample_tilt_axis})",
        }[frame]
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(
            title or f"Grain {grain + 1}  —  {self._STRESS_LABELS[component]}  [{_frame_label}]",
            fontsize=10,
        )
        fig.tight_layout()
        return fig, ax

    def plot_strain_histogram(
        self,
        components: "list[str] | None" = None,
        grains: "list[int] | None" = None,
        *,
        frame: str = "crystal",
        sample_tilt_deg: float = -40.0,
        sample_tilt_axis: str = "y",
        bins: int = 40,
        density: bool = False,
        scale: float = 1e3,
        alpha: float = 0.7,
        figsize: tuple | None = None,
        title: str | None = None,
    ) -> tuple:
        """
        Histogram of strain-tensor components for one or more grains.

        Each component gets its own subplot; when multiple grains are
        requested their distributions are overlaid with different colours.
        A vertical dashed line marks the mean of each distribution.

        Args:
            components (list of str or None): Strain components to plot.  Valid values: `'e_xx'`, `'e_yy'`,
                `'e_zz'`, `'e_xy'`, `'e_xz'`, `'e_yz'`.  `None` plots
                all six.  Default `None`.
            grains (list of int or None): Grain indices to include.  `None` uses all grains.
                Default `None`.
            frame (str): Reference frame for the strain tensor:

            `'crystal'`
                Components in the crystal coordinate system (as fitted).
            `'lab'`
                Rotated to the lab frame via `ε_lab = U @ ε @ Uᵀ`.
            `'sample'`
                Lab frame further rotated by *sample_tilt_deg* about
                *sample_tilt_axis*.

            sample_tilt_deg (float): Tilt angle (degrees) from lab to sample frame.  Default `-40`.
            sample_tilt_axis (str): Lab axis of the tilt rotation.  Default `'y'`.
            bins (int): Number of histogram bins.  Default `40`.
            density (bool): If `True`, normalise each histogram to unit area.
                Default `False`.
            scale (float): Multiplicative factor applied to all strain values before
                plotting.  The default `1e3` converts dimensionless strain to
                millistrain (×10⁻³), giving axis values near 1 for typical
                elastic strains.
            alpha (float): Bar transparency (0–1).  Default `0.7`.
            figsize (tuple or None): Figure size.  Auto-sized if `None`.
            title (str or None): Overall figure suptitle.  Auto-generated if `None`.

        Returns:
            fig (Figure):
            axes (ndarray of Axes): (shape matches the subplot grid)
"""
        _all_components = list(self._STRAIN_INDICES.keys())
        components = list(components) if components is not None else _all_components

        invalid = [c for c in components if c not in self._STRAIN_INDICES]
        if invalid:
            raise ValueError(
                f"Unknown component(s) {invalid}. "
                f"Choose from: {_all_components}"
            )

        grains = list(grains) if grains is not None else list(range(self.n_grains))

        # ── subplot grid ──────────────────────────────────────────────────────
        n    = len(components)
        ncols = min(n, 3)
        nrows = int(np.ceil(n / ncols))

        default_fs = (4.5 * ncols, 3.5 * nrows)
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=figsize or default_fs,
            squeeze=False,
        )

        prop_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        scale_str  = f"  ×10⁻³" if scale == 1e3 else (
                     f"  ×{scale:.0e}" if scale != 1.0 else "")

        for idx, comp in enumerate(components):
            row, col = divmod(idx, ncols)
            ax       = axes[row, col]
            label    = self._STRAIN_LABELS[comp] + scale_str

            for gi, grain in enumerate(grains):
                data = self._strain_component_map(
                    comp, grain, frame, sample_tilt_deg, sample_tilt_axis
                )
                vals = data[np.isfinite(data)].ravel() * scale
                if vals.size == 0:
                    continue

                color  = prop_cycle[gi % len(prop_cycle)]
                glabel = f"Grain {grain + 1}" if self.n_grains > 1 else None
                ax.hist(vals, bins=bins, density=density,
                        color=color, alpha=alpha, label=glabel)
                ax.axvline(float(np.mean(vals)), color=color,
                           linestyle="--", linewidth=1.2, alpha=0.9)

            ax.set_xlabel(label, fontsize=9)
            ax.set_ylabel("Density" if density else "Count", fontsize=9)
            ax.set_title(self._STRAIN_LABELS[comp], fontsize=10)
            ax.tick_params(labelsize=8)

            if self.n_grains > 1 and idx == 0:
                ax.legend(fontsize=7, framealpha=0.7)

        # Hide any unused axes in the last row
        for idx in range(n, nrows * ncols):
            row, col = divmod(idx, ncols)
            axes[row, col].set_visible(False)

        _frame_label = {
            "crystal": "crystal frame",
            "lab":     "lab frame",
            "sample":  f"sample frame ({sample_tilt_deg:+.0f}° about {sample_tilt_axis})",
        }[frame]
        fig.suptitle(
            title or f"Strain histogram  [{_frame_label}]",
            fontsize=11, y=1.01,
        )
        fig.tight_layout()
        return fig, axes

    def plot_deviatoric_strain_histogram(
        self,
        components: "list[str] | None" = None,
        grains: "list[int] | None" = None,
        *,
        frame: str = "crystal",
        sample_tilt_deg: float = -40.0,
        sample_tilt_axis: str = "y",
        bins: int = 40,
        density: bool = False,
        scale: float = 1e3,
        alpha: float = 0.7,
        figsize: tuple | None = None,
        title: str | None = None,
    ) -> tuple:
        """
        Histogram of deviatoric strain components for one or more grains.

        Each component gets its own subplot; when multiple grains are
        requested their distributions are overlaid with different colours.
        A vertical dashed line marks the mean of each distribution.

        Args:
            components (list of str or None): Deviatoric components to plot.  Valid values:
                ``'e_xx'``, ``'e_yy'``, ``'e_zz'``, ``'e_xy'``, ``'e_xz'``, ``'e_yz'``.
                ``None`` plots all six.  Default ``None``.
            grains (list of int or None): Grain indices to include.  ``None`` uses all grains.
                Default ``None``.
            frame (str): Reference frame — ``'crystal'``, ``'lab'``, or ``'sample'``.
            sample_tilt_deg (float): Tilt angle (degrees) from lab to sample frame.  Default ``-40``.
            sample_tilt_axis (str): Lab axis of the tilt rotation.  Default ``'y'``.
            bins (int): Number of histogram bins.  Default ``40``.
            density (bool): Normalise each histogram to unit area.  Default ``False``.
            scale (float): Multiplicative factor applied before plotting.  Default ``1e3``
                converts to millistrain (×10⁻³).
            alpha (float): Bar transparency (0–1).  Default ``0.7``.
            figsize (tuple or None): Figure size.  Auto-sized when ``None``.
            title (str or None): Figure suptitle.  Auto-generated when ``None``.

        Returns:
            tuple: ``(fig, axes)`` where *axes* is a 2-D ndarray matching the subplot grid.
        """
        _all_components = list(self._STRAIN_INDICES.keys())
        components = list(components) if components is not None else _all_components

        invalid = [c for c in components if c not in self._STRAIN_INDICES]
        if invalid:
            raise ValueError(
                f"Unknown component(s) {invalid}. Choose from: {_all_components}"
            )

        grains = list(grains) if grains is not None else list(range(self.n_grains))

        n     = len(components)
        ncols = min(n, 3)
        nrows = int(np.ceil(n / ncols))

        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=figsize or (4.5 * ncols, 3.5 * nrows),
            squeeze=False,
        )

        prop_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        scale_str  = "  ×10⁻³" if scale == 1e3 else (
                     f"  ×{scale:.0e}" if scale != 1.0 else "")

        for idx, comp in enumerate(components):
            row, col = divmod(idx, ncols)
            ax       = axes[row, col]
            label    = self._STRAIN_DEV_LABELS[comp] + scale_str

            for gi, grain in enumerate(grains):
                data = self._deviatoric_component_map(
                    comp, grain, frame, sample_tilt_deg, sample_tilt_axis
                )
                vals = data[np.isfinite(data)].ravel() * scale
                if vals.size == 0:
                    continue

                color  = prop_cycle[gi % len(prop_cycle)]
                glabel = f"Grain {grain + 1}" if self.n_grains > 1 else None
                ax.hist(vals, bins=bins, density=density,
                        color=color, alpha=alpha, label=glabel)
                ax.axvline(float(np.mean(vals)), color=color,
                           linestyle="--", linewidth=1.2, alpha=0.9)

            ax.set_xlabel(label, fontsize=9)
            ax.set_ylabel("Density" if density else "Count", fontsize=9)
            ax.set_title(self._STRAIN_DEV_LABELS[comp], fontsize=10)
            ax.tick_params(labelsize=8)

            if self.n_grains > 1 and idx == 0:
                ax.legend(fontsize=7, framealpha=0.7)

        for idx in range(n, nrows * ncols):
            row, col = divmod(idx, ncols)
            axes[row, col].set_visible(False)

        _frame_label = {
            "crystal": "crystal frame",
            "lab":     "lab frame",
            "sample":  f"sample frame ({sample_tilt_deg:+.0f}° about {sample_tilt_axis})",
        }.get(frame, frame)
        fig.suptitle(
            title or f"Deviatoric strain histogram  [{_frame_label}]",
            fontsize=11, y=1.01,
        )
        fig.tight_layout()
        return fig, axes

    # ── IPF map ───────────────────────────────────────────────────────────────

    @staticmethod
    def _cubic_ipf_colors(c: np.ndarray) -> np.ndarray:
        """
        Vectorised IPF RGB for cubic (m-3m) symmetry.

        Colour convention: [001] → blue, [011] → green, [111] → red.

        Args:
            c ((…, 3) array): Crystal-frame directions.  Need not be unit vectors; NaN rows
                produce NaN RGB output.

        Returns:
            rgb (same leading shape + (3,), float in [0, 1].):
"""
        c       = np.asarray(c, dtype=float)
        leading = c.shape[:-1]
        flat    = c.reshape(-1, 3).copy()

        rgb  = np.full((len(flat), 3), np.nan)
        norm = np.linalg.norm(flat, axis=1)
        ok   = (norm > 0) & ~np.any(np.isnan(flat), axis=1)

        d = flat[ok] / norm[ok, None]
        d = np.sort(np.abs(d), axis=1)      # h1 ≤ h2 ≤ h3

        theta = np.arctan2(d[:, 1], d[:, 2])
        phi   = np.arctan2(d[:, 0], np.sqrt(d[:, 1]**2 + d[:, 2]**2))

        t = np.clip(theta / (np.pi / 4.0),          0.0, 1.0)
        p = np.clip(phi   / np.arctan(1.0 / np.sqrt(2.0)), 0.0, 1.0)

        r_c = p
        g_c = t * (1.0 - p)
        b_c = (1.0 - t) * (1.0 - p)
        mx  = np.maximum(np.maximum(r_c, g_c), b_c)
        mx  = np.maximum(mx, 1e-10)

        rgb[ok] = np.stack([r_c / mx, g_c / mx, b_c / mx], axis=1)
        return rgb.reshape(*leading, 3)

    @staticmethod
    def _cubic_ipf_colorkey(N: int = 256) -> np.ndarray:
        """
        Render the cubic IPF color key as an (N, N, 4) float32 RGBA image.

        The standard triangle [001]–[011]–[111] is filled; pixels outside
        have alpha=0 (transparent).  Rows correspond to increasing *p*
        (phi, bottom = [001]–[011] edge) and columns to increasing *t*
        (theta, left = [001]).
"""
        t_vals = np.linspace(0.0, 1.0, N)
        p_vals = np.linspace(0.0, 1.0, N)
        T, P   = np.meshgrid(t_vals, p_vals)

        # Boundary curve [001] → [111]: directions [s, s, 1]/norm, s ∈ [0, 1].
        s_b  = np.linspace(0.0, 1.0, 1000)
        t_b  = np.arctan(s_b) / (np.pi / 4.0)
        p_b  = np.arctan(s_b / np.sqrt(s_b**2 + 1.0)) / np.arctan(1.0 / np.sqrt(2.0))
        p_b[0] = 0.0   # exact zero at s = 0

        # For each column index (t value) the maximum allowed p.
        p_max  = np.interp(t_vals, t_b, p_b)   # shape (N,)
        inside = P <= p_max[np.newaxis, :]      # broadcast over rows

        r_c = P
        g_c = T * (1.0 - P)
        b_c = (1.0 - T) * (1.0 - P)
        mx  = np.maximum(np.maximum(r_c, g_c), b_c)
        mx  = np.maximum(mx, 1e-10)

        rgba          = np.zeros((N, N, 4), dtype=np.float32)
        rgba[..., 0]  = r_c / mx
        rgba[..., 1]  = g_c / mx
        rgba[..., 2]  = b_c / mx
        rgba[..., 3]  = inside.astype(np.float32)   # 0 = transparent outside
        return rgba

    @staticmethod
    def _ipf_colorkey_inset(parent_ax, c_mean: "np.ndarray | None" = None) -> None:
        """Add the cubic IPF color-key as a small inset in *parent_ax*.

        Args:
            c_mean ((3,) array or None): Mean crystal direction (already in the fundamental zone, i.e.
                sorted absolute values).  If given, a marker is drawn at the
                corresponding position in the triangle.
"""
        try:
            from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        except ImportError:
            return

        ax_key = inset_axes(parent_ax, width="28%", height="28%",
                            loc="lower right", borderpad=0.5)
        ax_key.set_facecolor("none")

        rgba = GrainMap._cubic_ipf_colorkey(200)
        ax_key.imshow(rgba, origin="lower", extent=[0, 1, 0, 1],
                      aspect="auto", interpolation="bilinear")

        # Corner labels
        kw = dict(fontsize=6, fontweight="bold", color="k")
        ax_key.text(0.03, 0.03, "001", ha="left",  va="bottom", **kw)
        ax_key.text(0.97, 0.03, "011", ha="right", va="bottom", **kw)
        ax_key.text(0.97, 0.97, "111", ha="right", va="top",    **kw)

        # Triangle outline following the exact [001]–[111] boundary
        s_b  = np.linspace(0.0, 1.0, 120)
        t_b  = np.arctan(s_b) / (np.pi / 4.0)
        p_b  = np.arctan(s_b / np.sqrt(s_b**2 + 1.0)) / np.arctan(1.0 / np.sqrt(2.0))
        p_b[0] = 0.0
        verts = np.column_stack([
            np.concatenate([[0, 1, 1], t_b[::-1]]),
            np.concatenate([[0, 0, 1], p_b[::-1]]),
        ])
        from matplotlib.patches import Polygon as _Poly
        ax_key.add_patch(_Poly(verts, fill=False, edgecolor="k", linewidth=0.8))

        # Average orientation marker
        if c_mean is not None:
            n = np.linalg.norm(c_mean)
            if n > 1e-12:
                d     = c_mean / n                        # already sorted abs
                theta = np.arctan2(d[1], d[2])
                phi   = np.arctan2(d[0], np.sqrt(d[1] ** 2 + d[2] ** 2))
                t_avg = float(np.clip(theta / (np.pi / 4.0),             0.0, 1.0))
                p_avg = float(np.clip(phi   / np.arctan(1.0 / np.sqrt(2.0)), 0.0, 1.0))
                ax_key.scatter(
                    [t_avg], [p_avg],
                    s=55, marker="*", zorder=6,
                    c="white", edgecolors="black", linewidths=0.7,
                )

        ax_key.set_xlim(0, 1); ax_key.set_ylim(0, 1)
        ax_key.set_xticks([]); ax_key.set_yticks([])
        for sp in ax_key.spines.values():
            sp.set_visible(False)

    @staticmethod
    def _ipf_colorkey_inset_stretched(
        parent_ax,
        t_vals: np.ndarray,
        p_vals: np.ndarray,
        rgb_stretched: np.ndarray,
    ) -> None:
        """
        Colour-key inset for stretched IPF maps.

        Shows the full standard triangle with the data extent highlighted,
        then a zoomed inset of the data region coloured with the *stretched*
        colours so you can read off absolute orientations.

        Args:
            t_vals, p_vals ((N,) flat arrays): (t, p) coordinates in [0,1]² of all valid map pixels.
            rgb_stretched ((N, 3) float array): Stretched RGB for each valid pixel.
"""
        try:
            from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        except ImportError:
            return

        # ── outer inset: full triangle + data bounding box ────────────────────
        ax_full = inset_axes(parent_ax, width="28%", height="28%",
                             loc="lower right", borderpad=0.5)
        ax_full.set_facecolor("none")

        rgba = GrainMap._cubic_ipf_colorkey(200)
        ax_full.imshow(rgba, origin="lower", extent=[0, 1, 0, 1],
                       aspect="auto", interpolation="bilinear", alpha=0.35)

        # Triangle outline
        s_b = np.linspace(0.0, 1.0, 120)
        t_b = np.arctan(s_b) / (np.pi / 4.0)
        p_b = np.arctan(s_b / np.sqrt(s_b**2 + 1.0)) / np.arctan(1.0 / np.sqrt(2.0))
        p_b[0] = 0.0
        verts = np.column_stack([
            np.concatenate([[0, 1, 1], t_b[::-1]]),
            np.concatenate([[0, 0, 1], p_b[::-1]]),
        ])
        from matplotlib.patches import Polygon as _Poly, Rectangle as _Rect
        ax_full.add_patch(_Poly(verts, fill=False, edgecolor="k", linewidth=0.6))

        # Bounding box of data in (t, p) space
        if len(t_vals):
            t_lo, t_hi = float(t_vals.min()), float(t_vals.max())
            p_lo, p_hi = float(p_vals.min()), float(p_vals.max())
            pad = 0.02
            ax_full.add_patch(_Rect(
                (t_lo - pad, p_lo - pad),
                (t_hi - t_lo) + 2 * pad,
                (p_hi - p_lo) + 2 * pad,
                fill=False, edgecolor="black", linewidth=1.0, linestyle="--",
            ))

        kw = dict(fontsize=5, color="k")
        ax_full.text(0.02, 0.02, "001", ha="left",  va="bottom", **kw)
        ax_full.text(0.98, 0.02, "011", ha="right", va="bottom", **kw)
        ax_full.text(0.98, 0.98, "111", ha="right", va="top",    **kw)
        ax_full.set_xlim(0, 1); ax_full.set_ylim(0, 1)
        ax_full.set_xticks([]); ax_full.set_yticks([])
        for sp in ax_full.spines.values():
            sp.set_visible(False)

        # ── inner inset: zoomed scatter coloured with stretched RGB ───────────
        if len(t_vals) == 0:
            return
        t_lo, t_hi = float(t_vals.min()), float(t_vals.max())
        p_lo, p_hi = float(p_vals.min()), float(p_vals.max())

        ax_zoom = inset_axes(parent_ax, width="28%", height="28%",
                             loc="upper right", borderpad=0.5)
        ax_zoom.set_facecolor("0.15")

        # scatter each valid pixel as a tiny point
        ax_zoom.scatter(t_vals, p_vals, c=rgb_stretched, s=1.5,
                        linewidths=0, rasterized=True)

        pad_t = max((t_hi - t_lo) * 0.15, 1e-4)
        pad_p = max((p_hi - p_lo) * 0.15, 1e-4)
        ax_zoom.set_xlim(t_lo - pad_t, t_hi + pad_t)
        ax_zoom.set_ylim(p_lo - pad_p, p_hi + pad_p)
        ax_zoom.set_xticks([]); ax_zoom.set_yticks([])
        ax_zoom.set_title("stretched", fontsize=5, pad=2)
        for sp in ax_zoom.spines.values():
            sp.set_color("0.5"); sp.set_linewidth(0.5)

    def plot_overview(
        self,
        grain: int = 0,
        *,
        ipf_axes: "list[str] | None" = None,
        quality_metrics: "list[str] | None" = None,
        show_strain: bool = True,
        strain_components: "list[str] | None" = None,
        strain_frame: str = "crystal",
        sample_tilt_deg: float = -40.0,
        sample_tilt_axis: str = "y",
        symmetry: str = "cubic",
        frame: str = "lab",
        motor_x: str | None = None,
        motor_y: str | None = None,
        motor_units: "dict | None" = None,
        figsize_per_panel: tuple = (4.0, 3.5),
        title: str | None = None,
    ) -> tuple:
        """
        Multi-panel overview figure for one grain.

        Assembles up to three rows:

        * **Row 1 — Orientations**: IPF maps for each axis in *ipf_axes*
          (default: X, Y, Z).
        * **Row 2 — Quality**: scalar quality maps listed in
          *quality_metrics* (default: `match_rate`, `rms_px`, `kam`).
        * **Row 3 — Strain** *(optional)*: one panel per component in
          *strain_components* (default: all six).  Only shown when
          *show_strain* is `True` **and** the grain has non-NaN strain data.

        Args:
            grain (int): Grain index (0-based).  Use the index returned by
                :meth:`apply_merge` to plot the merged result.
            ipf_axes (list of str or None): Reference axes for the IPF maps.  Each entry is `'x'`,
                `'y'`, or `'z'`.  Default `['x', 'y', 'z']`.
            quality_metrics (list of str or None): Scalar maps for row 2.  Valid values: any key accepted by
                :meth:`plot_map` (`'match_rate'`, `'rms_px'`, `'mean_px'`,
                `'cost'`, `'n_matched'`, `'misorientation'`) plus
                `'kam'`.  Default `['match_rate', 'rms_px', 'kam']`.
            show_strain (bool): Include the strain row.  Silently skipped if no strain data are
                present for *grain*.  Default `True`.
            strain_components (list of str or None): Strain components to show.  Default: all six
                `['e_xx', 'e_yy', 'e_zz', 'e_xy', 'e_xz', 'e_yz']`.
            strain_frame (str): `'crystal'`, `'lab'`, or `'sample'`.  Default
                `'crystal'`.
            sample_tilt_deg (float): Tilt angle (°) for sample-frame strain rotation.  Default
                `-40`.
            sample_tilt_axis (str): Lab axis of the tilt rotation.  Default `'y'`.
            symmetry (str): Crystal symmetry for IPF colouring.  Default `'cubic'`.
            frame (str): Frame passed to :meth:`plot_ipf_map`.  Default `'lab'`.
            motor_x, motor_y (str or None): Motor names for axis labels (from `self.motors`).
            motor_units (dict or None): Units per motor, e.g. `{'xech': 'mm', 'yech': 'mm'}`.
            figsize_per_panel (tuple): `(width, height)` in inches for each individual panel.
                Default `(4.0, 3.5)`.
            title (str or None): Overall figure title.  Auto-generated if `None`.

        Returns:
            fig (Figure):
            axes ((n_rows, n_cols) ndarray of Axes):
"""
        ipf_axes       = list(ipf_axes or ["x", "y", "z"])
        quality_metrics = list(quality_metrics or ["match_rate", "rms_px", "kam"])
        strain_comps   = list(
            strain_components or ["e_xx", "e_yy", "e_zz", "e_xy", "e_xz", "e_yz"]
        )

        # decide whether to show strain row
        has_strain = (
            show_strain
            and self.n_grains > grain
            and np.any(np.isfinite(self.strain_voigt[grain]))
        )

        # ── build row specs ───────────────────────────────────────────────────
        rows: list[list[str]] = []
        rows.append([f"ipf_{a}" for a in ipf_axes])
        rows.append(quality_metrics)
        if has_strain:
            rows.append(strain_comps)

        ncols = max(len(r) for r in rows)
        nrows = len(rows)
        pw, ph = figsize_per_panel
        fig, axes_all = plt.subplots(
            nrows, ncols,
            figsize=(pw * ncols, ph * nrows),
            squeeze=False,
        )

        _km = {"motor_x": motor_x, "motor_y": motor_y, "motor_units": motor_units}

        for row_idx, row_specs in enumerate(rows):
            for col_idx in range(ncols):
                ax = axes_all[row_idx, col_idx]
                if col_idx >= len(row_specs):
                    ax.set_visible(False)
                    continue

                spec = row_specs[col_idx]

                if spec.startswith("ipf_"):
                    ipf_ax = spec[4:]
                    self.plot_ipf_map(
                        ipf_ax, grain=grain,
                        frame=frame,
                        sample_tilt_deg=sample_tilt_deg,
                        sample_tilt_axis=sample_tilt_axis,
                        symmetry=symmetry,
                        ax=ax,
                        show_colorkey=(col_idx == len(row_specs) - 1),
                        **_km,
                    )

                elif spec == "kam":
                    data = self.kam_map(grain)
                    im = ax.imshow(
                        data, origin="upper", cmap="inferno",
                        extent=self._motor_extent(motor_x, motor_y),
                        interpolation="nearest", aspect="auto",
                    )
                    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    cb.set_label("KAM (°)", fontsize=8)
                    self._apply_motor_labels(ax, motor_x, motor_y, motor_units)
                    ax.set_title(f"Grain {grain + 1}  —  KAM", fontsize=9)

                elif spec in self._STRAIN_INDICES:
                    self.plot_strain_component(
                        spec, grain=grain,
                        frame=strain_frame,
                        sample_tilt_deg=sample_tilt_deg,
                        sample_tilt_axis=sample_tilt_axis,
                        ax=ax,
                        **_km,
                    )

                else:
                    self.plot_map(spec, grain=grain, ax=ax, **_km)

        fig.suptitle(
            title or f"Grain {grain + 1}  —  overview",
            fontsize=11, y=1.01,
        )
        fig.tight_layout()
        return fig, axes_all

    # ── helpers used by plot_overview ─────────────────────────────────────────

    def _motor_extent(
        self,
        motor_x: str | None,
        motor_y: str | None,
    ) -> list:
        mx = self.motors.get(motor_x) if motor_x else None
        my = self.motors.get(motor_y) if motor_y else None
        if mx is not None and my is not None:
            return [mx[0, 0], mx[0, -1], my[-1, 0], my[0, 0]]
        return [0, self.nx, self.ny, 0]

    def _apply_motor_labels(
        self,
        ax: "plt.Axes",
        motor_x: str | None,
        motor_y: str | None,
        motor_units: "dict | None" = None,
    ) -> None:
        mu = motor_units or {}
        mx = self.motors.get(motor_x) if motor_x else None
        my = self.motors.get(motor_y) if motor_y else None
        if mx is not None and my is not None:
            xu = mu.get(motor_x, "")
            yu = mu.get(motor_y, "")
            ax.set_xlabel(f"{motor_x} ({xu})" if xu else motor_x, fontsize=8)
            ax.set_ylabel(f"{motor_y} ({yu})" if yu else motor_y, fontsize=8)
        else:
            ax.set_xlabel("column (ix)", fontsize=8)
            ax.set_ylabel("row (iy)", fontsize=8)

    def plot_ipf_map(
        self,
        axis="z",
        grain: int = 0,
        *,
        frame: str = "lab",
        sample_tilt_deg: float = -40.0,
        sample_tilt_axis: str = "y",
        symmetry: str = "cubic",
        ax: "plt.Axes | None" = None,
        motor_x: str | None = None,
        motor_y: str | None = None,
        motor_units: "dict | None" = None,
        title: str | None = None,
        figsize: tuple = (6, 5),
        show_colorkey: bool = True,
        stretch: "bool | str" = False,
        best_grain: "np.ndarray | None" = None,
    ) -> tuple:
        """
        Inverse pole figure (IPF) map coloured by which crystal direction is
        parallel to a chosen reference axis.

        Args:
            axis (str or (3,) array-like): Reference direction in the chosen *frame*.
                Shortcuts: `'x'`, `'y'`, `'z'`; or a custom 3-vector.
            grain (int): Grain index (0-based).
            frame (str): `'lab'`    — *axis* is in the lab frame.
                `'sample'` — *axis* is in the sample frame, converted to lab
                via the inverse of the sample tilt (see *sample_tilt_deg*).
            sample_tilt_deg (float): Rotation angle (°) about *sample_tilt_axis* that maps lab → sample.
                Default `-40`.
            sample_tilt_axis (str): Lab axis of the tilt rotation.  Default `'y'`.
            symmetry (str): Crystal symmetry for IPF reduction.  Currently only `'cubic'`.
            motor_x, motor_y (str or None): Motor names to use as axis tick labels.
            motor_units (dict or None): Optional units for motor axes, e.g. `{'pz': 'mm', 'py': 'mm'}`.
                Appended to the axis label in parentheses.
            show_colorkey (bool): Overlay a small colour-key triangle in the lower-right corner.
                Automatically disabled when *stretch* is `True`.
            stretch (bool or str): Contrast enhancement mode.  Options:

            `False` (default)
                No stretching — standard IPF colour mapping.
            `True` or `"global"`
                Single linear stretch across all valid pixels.  Good for
                a map with one dominant orientation; with two distinct
                grains the inter-grain distance consumes the full gamut
                and intra-grain variation remains compressed.
            `"local"`
                Independent per-grain stretch: each grain region is
                normalised to [0, 1] separately, revealing intra-grain
                orientation spread for every grain simultaneously.
                Requires *best_grain* (the `(ny, nx)` int array returned
                by :meth:`merge`); raises `ValueError` if omitted.
                Colours are **not** comparable between grains.

            When stretching is active and *show_colorkey* is `True` the
            standard colour-key is replaced by two insets (see
            :meth:`_ipf_colorkey_inset_stretched`).
            best_grain ((ny, nx) int ndarray or None): Grain-label array required for `stretch="local"`.  Positions
                with value `-1` are treated as invalid (white).
"""
        _shortcuts = {
            "x": np.array([1.0, 0.0, 0.0]),
            "y": np.array([0.0, 1.0, 0.0]),
            "z": np.array([0.0, 0.0, 1.0]),
        }
        if isinstance(axis, str):
            ref        = _shortcuts[axis.lower()]
            axis_label = axis.upper()
        else:
            ref        = np.asarray(axis, dtype=float)
            ref        = ref / np.linalg.norm(ref)
            axis_label = f"[{ref[0]:.2f} {ref[1]:.2f} {ref[2]:.2f}]"

        if frame == "sample":
            R_s = Rotation.from_euler(
                sample_tilt_axis, sample_tilt_deg, degrees=True
            ).as_matrix()
            ref = R_s.T @ ref   # sample frame → lab frame

        if symmetry != "cubic":
            raise ValueError(
                f"Unsupported symmetry {symmetry!r}. Only 'cubic' is implemented."
            )

        # Crystal direction parallel to ref: c = U^T @ ref  (vectorised)
        U   = self.U[grain]                             # (ny, nx, 3, 3)
        c   = np.einsum("...ji,j->...i", U, ref)        # (ny, nx, 3)
        rgb = self._cubic_ipf_colors(c)                 # (ny, nx, 3)

        # (t, p) coordinates for every valid pixel — needed for the stretched
        # colour key and for the standard mean-orientation marker.
        valid = ~np.any(np.isnan(c), axis=-1)
        t_flat = np.full(valid.shape, np.nan)
        p_flat = np.full(valid.shape, np.nan)
        if valid.any():
            c_v    = c[valid]
            norms  = np.linalg.norm(c_v, axis=1, keepdims=True)
            c_unit = c_v / np.maximum(norms, 1e-12)
            c_sym  = np.sort(np.abs(c_unit), axis=1)          # h1 ≤ h2 ≤ h3
            t_flat[valid] = np.arctan2(c_sym[:, 1], c_sym[:, 2]) / (np.pi / 4.0)
            p_flat[valid] = (np.arctan2(c_sym[:, 0],
                             np.sqrt(c_sym[:, 1]**2 + c_sym[:, 2]**2))
                             / np.arctan(1.0 / np.sqrt(2.0)))
            c_mean = c_sym.mean(axis=0)
        else:
            c_mean = None

        # Normalise stretch argument
        if stretch is True:
            stretch = "global"
        elif stretch is False:
            stretch = None

        if stretch == "global":
            for ch in range(3):
                ch_vals = rgb[..., ch]
                lo, hi = np.nanmin(ch_vals), np.nanmax(ch_vals)
                if hi > lo:
                    rgb[..., ch] = (ch_vals - lo) / (hi - lo)

        elif stretch == "local":
            if best_grain is None:
                raise ValueError(
                    "stretch='local' requires best_grain — pass the (ny, nx) "
                    "int array returned by merge()."
                )
            bg = np.asarray(best_grain)
            for gi in np.unique(bg[bg >= 0]):
                mask = (bg == gi) & valid
                if not mask.any():
                    continue
                for ch in range(3):
                    ch_vals = rgb[..., ch]
                    lo = float(np.nanmin(ch_vals[mask]))
                    hi = float(np.nanmax(ch_vals[mask]))
                    if hi > lo:
                        rgb[mask, ch] = (ch_vals[mask] - lo) / (hi - lo)

        elif stretch is not None:
            raise ValueError(
                f"stretch must be False, True, 'global', or 'local', got {stretch!r}"
            )

        # NaN → white
        img = np.where(np.isnan(rgb), 1.0, np.clip(rgb, 0.0, 1.0)).astype(np.float32)

        # ── axis extent and labels ────────────────────────────────────────────
        mu  = motor_units or {}
        mx  = self.motors.get(motor_x) if motor_x else None
        my  = self.motors.get(motor_y) if motor_y else None

        if mx is not None and my is not None:
            extent = [mx[0, 0], mx[0, -1], my[-1, 0], my[0, 0]]
            xu = mu.get(motor_x, "")
            yu = mu.get(motor_y, "")
            xlabel = f"{motor_x} ({xu})" if xu else motor_x
            ylabel = f"{motor_y} ({yu})" if yu else motor_y
        else:
            extent = [0, self.nx, self.ny, 0]
            xlabel = "column (ix)"
            ylabel = "row (iy)"

        # ── figure ────────────────────────────────────────────────────────────
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        ax.imshow(img, origin="upper", extent=extent,
                  interpolation="nearest", aspect="auto")

        _frame_str = {
            "lab":    "lab",
            "sample": f"sample ({sample_tilt_deg:+.0f}° {sample_tilt_axis})",
        }.get(frame, frame)

        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(
            title or f"Grain {grain + 1}  —  IPF ∥ {axis_label}  [{_frame_str} frame]",
            fontsize=10,
        )

        if show_colorkey:
            if stretch:
                self._ipf_colorkey_inset_stretched(
                    ax,
                    t_vals=t_flat[valid],
                    p_vals=p_flat[valid],
                    rgb_stretched=img[valid],
                )
            else:
                self._ipf_colorkey_inset(ax, c_mean=c_mean)

        fig.tight_layout()
        return fig, ax

    def plot_ipf_scatter(
        self,
        grain: int = 0,
        *,
        frame: str = "lab",
        sample_tilt_deg: float = -40.0,
        sample_tilt_axis: str = "y",
        symmetry: str = "cubic",
        figsize: tuple = (15, 5),
        s: float = 15.0,
        alpha: float = 0.8,
    ) -> tuple:
        """
        Scatter pole figure — all three crystal axes in the chosen frame.

        Each of the three panels shows where the crystal a-, b- or c-axis
        points relative to the lab/sample coordinate system, for every fitted
        map pixel.  Points are coloured with the same IPF scheme as
        :meth:`plot_ipf_map` (cubic: [001] → blue, [011] → green, [111] → red).

        Args:
            grain (int): Grain index (0-based).
            frame (str): `'lab'`    — directions expressed in the lab frame.
                `'sample'` — directions expressed in the sample frame (rotated
                from lab by *sample_tilt_deg* about *sample_tilt_axis*).
            sample_tilt_deg (float): Lab-to-sample rotation angle (°).  Default `-40`.
            sample_tilt_axis (str): Axis of the lab-to-sample rotation.  Default `'y'`.
            symmetry (str): IPF colour symmetry.  Currently only `'cubic'`.
            s, alpha (float): Scatter marker size and transparency.
"""
        U = self.U[grain]   # (ny, nx, 3, 3)

        if frame == "sample":
            R_s    = Rotation.from_euler(
                sample_tilt_axis, sample_tilt_deg, degrees=True
            ).as_matrix()
            U_plot = np.einsum("ij,...jk->...ik", R_s, U)   # R_s @ U at each point
        else:
            U_plot = U

        # Valid (fitted) pixels
        valid = ~np.any(np.isnan(U_plot.reshape(*U_plot.shape[:2], -1)), axis=-1)

        _axis_names = ["a", "b", "c"]
        _frame_str  = {
            "lab":    "lab frame",
            "sample": f"sample frame ({sample_tilt_deg:+.0f}° {sample_tilt_axis})",
        }.get(frame, frame)

        fig, axes_arr = plt.subplots(1, 3, figsize=figsize)

        for ai, aname in enumerate(_axis_names):
            ax = axes_arr[ai]

            # i-th column of U_plot = i-th crystal axis in the chosen frame
            d = U_plot[valid, :, ai]           # (N, 3)

            if symmetry == "cubic":
                colors = np.clip(self._cubic_ipf_colors(d), 0.0, 1.0)
            else:
                nrm    = np.linalg.norm(d, axis=1, keepdims=True)
                colors = np.abs(d) / np.maximum(nrm, 1e-10)

            ax.scatter(d[:, 0], d[:, 1], s=s, c=colors, alpha=alpha,
                       linewidths=0)

            # Reference circle (unit sphere projected to XY)
            theta_c = np.linspace(0, 2 * np.pi, 200)
            ax.plot(np.cos(theta_c), np.sin(theta_c),
                    color="k", linewidth=0.5, zorder=3)
            ax.axhline(0, color="k", linewidth=0.3, zorder=3)
            ax.axvline(0, color="k", linewidth=0.3, zorder=3)

            ax.set_aspect("equal")
            ax.set_xlim(-1.1, 1.1)
            ax.set_ylim(-1.1, 1.1)
            ax.set_xlabel("X", fontsize=9)
            ax.set_ylabel("Y", fontsize=9)
            ax.set_title(
                f"Grain {grain + 1}  —  crystal {aname}-axis  [{_frame_str}]",
                fontsize=10,
            )

        fig.tight_layout()
        return fig, axes_arr

    def inspect_frame(
        self,
        crystal,
        camera,
        base_dir: str,
        *,
        h5_dataset: "str | None" = None,
        tiff_dir: "str | None" = None,
        grains: "list[int] | None" = None,
        map_quantity: str = "match_rate",
        map_grain: "int | None" = None,
        motor_x: "str | None" = None,
        motor_y: "str | None" = None,
        motor_units: "dict | None" = None,
        E_min_eV: float = 5000.0,
        E_max_eV: float = 27000.0,
        max_match_px: float = 10.0,
        top_n_sim: "int | None" = None,
        r_squared_min: float = 0.0,
        include_unfitted: bool = True,
        figsize: tuple = (14, 7),
    ) -> tuple:
        """
        Interactive frame inspector: click a map pixel to display the
        diffraction image, observed spots, and simulated grain patterns.

        The figure has two panels:

        * **Left** — a scalar map (e.g. match rate) of the raster scan.
          Click any pixel to load that frame.
        * **Right** — the detector image for the selected frame, with
          observed spots (white circles), simulated spots per grain
          (coloured diamonds), and lines connecting matched pairs.
          Standard matplotlib zoom / pan tools work on this panel; the
          zoom level is preserved across clicks.

        Args:
            crystal (Crystal or LayeredCrystal): Crystal structure used for spot simulation.
            camera (Camera): Detector geometry.
            base_dir (str): Processing directory that contains the `seg/` sub-folder
                with segmentation HDF5 spot files.
            h5_dataset (str or None): HDF5 dataset path inside `self.h5_path` for the image stack
                (e.g. `'1.1/measurement/det'`).  Mutually exclusive with
                *tiff_dir*; supply exactly one (or neither to skip image loading).
            tiff_dir (str or None): Path to a folder of `img_<number>.tif` files.  Files are
                sorted by their embedded number and mapped to 0-based frame
                indices.  Mutually exclusive with *h5_dataset*.
            grains (list of int or None): Grain indices to simulate.  `None` uses all grains.
            map_quantity (str): Scalar quantity shown on the left panel.  One of
                `'match_rate'`, `'rms_px'`, `'mean_px'`, `'cost'`.
                Default `'match_rate'`.
            map_grain (int or None): Grain index used to build the left-panel map.  `None` (default)
                uses the merged grain slot when :meth:`apply_merge` has been
                called, otherwise falls back to grain `0`.
            motor_x, motor_y (str or None): Motor names for axis labels and click-to-pixel conversion.
            motor_units (dict or None): Units per motor, e.g. `{'pz': 'mm', 'py': 'mm'}`.
            E_min_eV, E_max_eV (float): Energy range for spot simulation.  Defaults `5000` / `27000` eV.
                The allowed-HKL sphere cutoff is derived automatically from
                `E_max_eV`.
            max_match_px (float): Match radius in pixels for drawing connection lines.
                Default `10`.
            top_n_sim (int or None): Limit the number of simulated spots shown.  `None` keeps all.
            r_squared_min (float): Minimum Gaussian-fit R² for loading observed spots.  Default
                `0.0` (show everything from the spots file).
            include_unfitted (bool): Include spots stored as raw centroids (fit failed).  Default
                `True`.
            figsize (tuple): Figure size.  Default `(14, 7)`.

        Returns:
            fig (Figure):
            axes ((ax_map, ax_det)):
"""
        from .simulation import simulate_laue
        from .fitting import _match_spots
        from .segmentation import convert_spotsfile2peaklist

        seg_dir    = os.path.join(base_dir, "seg")
        grains_use = list(grains) if grains is not None else list(range(self.n_grains))
        if map_grain is None:
            map_grain = self._merged_grain if self._merged_grain is not None else 0

        # ── motor / extent helpers ────────────────────────────────────────────
        mu = motor_units or {}
        mx = self.motors.get(motor_x) if motor_x else None
        my = self.motors.get(motor_y) if motor_y else None

        if mx is not None and my is not None:
            extent_map = [mx[0, 0], mx[0, -1], my[-1, 0], my[0, 0]]
            xu = mu.get(motor_x, "")
            yu = mu.get(motor_y, "")
            xlabel_map = f"{motor_x} ({xu})" if xu else motor_x
            ylabel_map = f"{motor_y} ({yu})" if yu else motor_y
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

        # ── image loader ──────────────────────────────────────────────────────
        _tiff_index: "list | None" = None   # built lazily

        def _load_image(frame_idx: int) -> "np.ndarray | None":
            if tiff_dir is not None:
                nonlocal _tiff_index
                if _tiff_index is None:
                    import re as _re
                    pat   = _re.compile(r'^img_(\d+)\.tif$', _re.IGNORECASE)
                    files = []
                    for fname in os.listdir(tiff_dir):
                        m = pat.match(fname)
                        if m:
                            files.append(
                                (int(m.group(1)), os.path.join(tiff_dir, fname))
                            )
                    files.sort(key=lambda x: x[0])
                    _tiff_index = [p for _, p in files]
                if frame_idx >= len(_tiff_index):
                    return None
                try:
                    import skimage.io
                    return skimage.io.imread(_tiff_index[frame_idx]).astype(np.float32)
                except Exception:
                    return None
            elif h5_dataset is not None:
                try:
                    with h5py.File(self.h5_path, "r") as f:
                        return f[h5_dataset][frame_idx].astype(np.float32)
                except Exception:
                    return None
            return None

        # ── left-panel map data ───────────────────────────────────────────────
        _map_opts = {
            "match_rate": (self.match_rate[map_grain], "Match rate",    "viridis"),
            "rms_px":     (self.rms_px[map_grain],     "RMS (px)",      "plasma_r"),
            "mean_px":    (self.mean_px[map_grain],     "Mean dev (px)", "plasma_r"),
            "cost":       (self.cost[map_grain],        "Cost",          "plasma_r"),
        }
        map_data, map_label, map_cmap = _map_opts.get(
            map_quantity,
            (self.match_rate[map_grain], "Match rate", "viridis"),
        )

        # ── figure ────────────────────────────────────────────────────────────
        fig = plt.figure(figsize=figsize)
        gs  = fig.add_gridspec(
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
        ax_map.set_title(
            f"Click to inspect — {map_label}  (grain {map_grain + 1})",
            fontsize=9,
        )
        sel_dot, = ax_map.plot([], [], "w+", ms=11, mew=2.0, zorder=10)

        ax_det.set_facecolor("k")
        ax_det.set_xlim(0, camera.Nh)
        ax_det.set_ylim(camera.Nv, 0)
        ax_det.set_aspect("equal")
        ax_det.set_xlabel("x (px)", fontsize=9)
        ax_det.set_ylabel("y (px)", fontsize=9)
        ax_det.set_title("← click map to load frame", fontsize=9, color="#888")

        prop_cycle  = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        grain_colors = [prop_cycle[gi % len(prop_cycle)] for gi in range(self.n_grains)]

        fig.suptitle(
            "Frame inspector  —  zoom / pan right panel freely between clicks",
            fontsize=9, color="#555",
        )

        # ── click handler ─────────────────────────────────────────────────────
        _state = {"drawn": False}

        def _on_click(event) -> None:
            if event.inaxes is not ax_map:
                return
            if event.xdata is None or event.ydata is None:
                return

            iy, ix    = _click_to_iy_ix(event.xdata, event.ydata)
            frame_idx = self.frame_index(iy, ix)

            # Preserve user zoom between clicks
            saved_xlim = ax_det.get_xlim()
            saved_ylim = ax_det.get_ylim()

            # Move selection crosshair on map
            if mx is not None and my is not None:
                sel_dot.set_data([mx[iy, ix]], [my[iy, ix]])
            else:
                sel_dot.set_data([ix + 0.5], [iy + 0.5])

            # Load image
            image = _load_image(frame_idx)

            # Load observed spots
            seg_path = os.path.join(seg_dir, f"frame_{frame_idx:05d}.h5")
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

            # Simulate fitted grains
            sim_data = {}   # gi → (sim_xy, row_ind, col_ind, ok_mask)
            for gi in grains_use:
                U = self.U[gi, iy, ix]
                if np.any(np.isnan(U)):
                    continue
                try:
                    spots  = simulate_laue(
                        crystal, U, camera,
                        E_min=E_min_eV, E_max=E_max_eV,
                    )
                    on_det = [s for s in spots if s.get("pix") is not None]
                    if top_n_sim is not None:
                        on_det = on_det[:top_n_sim]
                    sim_xy = (
                        np.array([s["pix"] for s in on_det])
                        if on_det else np.empty((0, 2))
                    )
                    if len(obs_xy) > 0 and len(sim_xy) > 0:
                        row_ind, col_ind, dist_px = _match_spots(
                            obs_xy, sim_xy, max_match_px
                        )
                        ok = dist_px < max_match_px
                    else:
                        row_ind = col_ind = np.array([], dtype=int)
                        ok      = np.array([], dtype=bool)
                    sim_data[gi] = (sim_xy, row_ind, col_ind, ok)
                except Exception as exc:
                    print(f"  grain {gi}: simulation error: {exc}", flush=True)

            # ── redraw detector panel ─────────────────────────────────────────
            ax_det.cla()
            ax_det.set_facecolor("k")

            if image is not None:
                pos = image[image > 0]
                vmax = float(np.percentile(pos, 99)) if pos.size else 1.0
                nv_im, nh_im = image.shape
                ax_det.imshow(
                    np.log1p(image / vmax * 1000),
                    origin="upper",
                    extent=[0, nh_im, nv_im, 0],
                    cmap="gray", aspect="equal", zorder=0,
                )
            else:
                ax_det.set_xlim(0, camera.Nh)
                ax_det.set_ylim(camera.Nv, 0)

            ax_det.set_aspect("equal")

            # Observed spots
            if len(obs_xy):
                ax_det.scatter(
                    obs_xy[:, 0], obs_xy[:, 1],
                    s=40, c="none", edgecolors="white", linewidths=0.8,
                    zorder=4,
                )

            # Simulated spots + match lines
            from matplotlib.lines import Line2D
            legend_handles = [
                Line2D([0], [0], marker="o", color="w",
                       markerfacecolor="none", markeredgecolor="white",
                       markersize=5, linestyle="none",
                       label=f"observed ({len(obs_xy)})")
            ] if len(obs_xy) else []

            for gi, (sim_xy, row_ind, col_ind, ok) in sim_data.items():
                color     = grain_colors[gi]
                n_matched = int(ok.sum()) if len(ok) else 0
                if len(sim_xy):
                    ax_det.scatter(
                        sim_xy[:, 0], sim_xy[:, 1],
                        s=28, c=color, marker="D",
                        linewidths=0, zorder=5, alpha=0.85,
                    )
                for r, c, hit in zip(row_ind, col_ind, ok):
                    if hit:
                        ax_det.plot(
                            [obs_xy[r, 0], sim_xy[c, 0]],
                            [obs_xy[r, 1], sim_xy[c, 1]],
                            color=color, lw=0.7, alpha=0.55, zorder=3,
                        )
                legend_handles.append(Line2D(
                    [0], [0], marker="D", color="w",
                    markerfacecolor=color, markersize=5,
                    linestyle="none",
                    label=f"grain {gi + 1}  ({n_matched} matched)",
                ))

            if legend_handles:
                ax_det.legend(
                    handles=legend_handles, fontsize=7, loc="upper right",
                    facecolor="#111", edgecolor="#444", labelcolor="white",
                    framealpha=0.85,
                )

            ax_det.set_xlabel("x (px)", fontsize=9)
            ax_det.set_ylabel("y (px)", fontsize=9)
            ax_det.set_title(
                f"Frame {frame_idx}  (iy={iy}, ix={ix})  "
                f"— {len(sim_data)} grain(s) simulated",
                fontsize=9,
            )

            # Restore zoom from before the click (skip on first draw)
            if _state["drawn"]:
                ax_det.set_xlim(saved_xlim)
                ax_det.set_ylim(saved_ylim)
            _state["drawn"] = True

            fig.canvas.draw_idle()

        fig.canvas.mpl_connect("button_press_event", _on_click)
        return fig, (ax_map, ax_det)

    def reindex_frame(
        self,
        crystal,
        camera,
        base_dir: str,
        *,
        h5_dataset: "str | None" = None,
        tiff_dir: "str | None" = None,
        map_quantity: str = "match_rate",
        map_grain: "int | None" = None,
        motor_x: "str | None" = None,
        motor_y: "str | None" = None,
        motor_units: "dict | None" = None,
        E_min_eV: float = 5000.0,
        E_max_eV: float = 27000.0,
        angle_tol_deg: float = 0.5,
        min_match_rate: float = 0.25,
        n_obs_use: int = 20,
        max_match_px: float = 30.0,
        fit_max_match_px: "float | list[float]" = [30.0, 10.0, 3.0],
        r_squared_min: float = 0.0,
        include_unfitted: bool = True,
        figsize: tuple = (14, 7),
    ) -> None:
        """
        Interactive single-frame re-indexer and fitter.

        Click a pixel on the left map panel to select a frame, then:

        1. **⚡ Index** — runs :func:`index_orientation` to find a rough
           orientation from the inter-spot angle table.  Orange diamonds show
           the simulated spots; green lines connect matched pairs.
        2. **⚡ Fit** — refines the orientation with :func:`fit_orientation`
           starting from the index result (or from the map's existing U if
           indexing was skipped).  Cyan diamonds replace the index overlay.
        3. **⬆ Store** — writes the fit result back into the map at the
           selected position and grain slot chosen by the grain selector.
        4. **💾 Save UB** — writes the best available U to an auto-numbered
           `UB<n>.npy` file in the current directory.

        Args:
            crystal (Crystal): xrayutilities crystal structure used for indexing and simulation.
            camera (Camera): Detector geometry.
            base_dir (str): Processing directory containing the `seg/` sub-folder with
                segmentation HDF5 spot files.
            h5_dataset (str or None): HDF5 dataset path inside `self.h5_path` for the image stack.
                Mutually exclusive with *tiff_dir*.
            tiff_dir (str or None): Path to a folder of `img_<number>.tif` files.  Mutually
                exclusive with *h5_dataset*.
            map_quantity (str): Scalar quantity shown on the left panel.  One of
                `'match_rate'`, `'rms_px'`, `'mean_px'`, `'cost'`.
            map_grain (int or None): Grain index used to build the left-panel map.  `None` uses
                the merged grain slot when available, otherwise grain `0`.
            motor_x, motor_y (str or None): Motor names for axis labels and click-to-pixel conversion.
            motor_units (dict or None): Units per motor, e.g. `{'pz': 'mm', 'py': 'mm'}`.
            E_min_eV, E_max_eV (float): Energy range for spot simulation.  The allowed-HKL sphere
                cutoff is derived automatically from `E_max_eV`.
            angle_tol_deg (float): Angular tolerance for table lookup and final scoring.  Default `0.5`.
            min_match_rate (float): Minimum match rate for `IndexResult.success`.  Default `0.25`.
            n_obs_use (int): Number of brightest observed spots passed to the indexer.
                Default `20`.
            max_match_px (float): Pixel radius for drawing match lines.  Default `30`.
            fit_max_match_px (float or list of float): Match-radius schedule passed to :func:`fit_orientation`.  A list
                enables staged refinement (each stage tightens the window).
                Default `[30, 10, 3]`.
            r_squared_min (float): Minimum Gaussian-fit R² when loading the spot file.  Default `0`.
            include_unfitted (bool): Include raw centroids (fit failed) from the spot file.
            figsize (tuple): Figure size.  Default `(14, 7)`.
"""
        import ipywidgets as ipw
        from IPython.display import display as _ipy_display
        from .simulation import simulate_laue as _sim_laue
        from .segmentation import convert_spotsfile2peaklist
        from .fitting import (
            index_orientation      as _index,
            fit_orientation        as _fit_ori,
            fit_strain_orientation as _fit_strain,
            _match_spots,
        )

        seg_dir = os.path.join(base_dir, "seg")
        if map_grain is None:
            map_grain = self._merged_grain if self._merged_grain is not None else 0

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
                    pat = _re.compile(r'^img_(\d+)\.tif$', _re.IGNORECASE)
                    files = []
                    for fname in os.listdir(tiff_dir):
                        m = pat.match(fname)
                        if m:
                            files.append((int(m.group(1)), os.path.join(tiff_dir, fname)))
                    files.sort(key=lambda t: t[0])
                    _tiff_index.extend(p for _, p in files)
                if frame_idx >= len(_tiff_index):
                    return None
                try:
                    import skimage.io
                    return skimage.io.imread(_tiff_index[frame_idx]).astype(np.float32)
                except Exception:
                    return None
            elif h5_dataset is not None:
                try:
                    with h5py.File(self.h5_path, "r") as fh:
                        return fh[h5_dataset][frame_idx].astype(np.float32)
                except Exception:
                    return None
            return None

        _map_opts = {
            "match_rate": (self.match_rate[map_grain], "Match rate",    "viridis"),
            "rms_px":     (self.rms_px[map_grain],     "RMS (px)",      "plasma_r"),
            "mean_px":    (self.mean_px[map_grain],     "Mean dev (px)", "plasma_r"),
            "cost":       (self.cost[map_grain],        "Cost",          "plasma_r"),
        }
        map_data, map_label, map_cmap = _map_opts.get(
            map_quantity, (self.match_rate[map_grain], "Match rate", "viridis")
        )

        with plt.ioff():
            fig = plt.figure(figsize=figsize)
        try:

            fig.canvas.manager.set_window_title("Laue — re-index frame")
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
        ax_map.set_title(
            f"Click to select — {map_label}  (grain {map_grain + 1})",
            fontsize=9,
        )
        sel_dot, = ax_map.plot([], [], "w+", ms=11, mew=2.0, zorder=10)

        ax_det.set_facecolor("k")
        ax_det.set_xlim(0, camera.Nh)
        ax_det.set_ylim(camera.Nv, 0)
        ax_det.set_aspect("equal")
        ax_det.set_xlabel("x (px)", fontsize=9)
        ax_det.set_ylabel("y (px)", fontsize=9)
        ax_det.set_title("← click map to select a frame", fontsize=9, color="#888")
        fig.suptitle(
            "Re-index frame  —  click map → Index → Fit → Store",
            fontsize=9, color="#555",
        )

        _state: dict = {
            "frame_idx": None,
            "iy": None, "ix": None,
            "obs_xy": np.empty((0, 2)),
            "result":     None,   # IndexResult
            "fit_result": None,   # OrientationFitResult
            "drawn": False,
        }

        # ── detector panel draw ───────────────────────────────────────────────
        def _overlay_sim(U, color, label) -> None:
            """Draw simulated spots + match lines for a given U."""
            try:
                spots  = _sim_laue(
                    crystal, U, camera,
                    E_min=E_min_eV, E_max=E_max_eV,
                )
                on_det = [s for s in spots if s.get("pix") is not None]
                sim_xy = (
                    np.array([s["pix"] for s in on_det])
                    if on_det else np.empty((0, 2))
                )
                obs_xy = _state["obs_xy"]
                ax_det.scatter(
                    sim_xy[:, 0], sim_xy[:, 1],
                    s=28, c=color, marker="D",
                    linewidths=0, zorder=5, alpha=0.85,
                    label=f"{label} ({len(sim_xy)} sim)",
                )
                if len(obs_xy) and len(sim_xy):
                    row_ind, col_ind, dist_px = _match_spots(
                        obs_xy, sim_xy, max_match_px
                    )
                    ok = dist_px < max_match_px
                    for r, c, hit in zip(row_ind, col_ind, ok):
                        if hit:
                            ax_det.plot(
                                [obs_xy[r, 0], sim_xy[c, 0]],
                                [obs_xy[r, 1], sim_xy[c, 1]],
                                color="#44dd66", lw=0.7, alpha=0.55, zorder=3,
                            )
            except Exception as exc:
                print(f"  simulation error: {exc}", flush=True)

        def _draw_det() -> None:
            frame_idx  = _state["frame_idx"]
            obs_xy     = _state["obs_xy"]
            result     = _state["result"]
            fit_result = _state["fit_result"]

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
                    origin="upper",
                    extent=[0, nh_im, nv_im, 0],
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

            # Fit result (cyan) takes priority; fall back to index result (orange)
            if fit_result is not None:
                _overlay_sim(fit_result.U, "#44aaff", "fitted")
            elif result is not None:
                _overlay_sim(result.U, "#ff6b35", "indexed")

            if len(obs_xy) or result is not None or fit_result is not None:
                ax_det.legend(
                    fontsize=7, loc="upper right",
                    facecolor="#111", edgecolor="#444", labelcolor="white",
                    framealpha=0.85,
                )

            ax_det.set_xlabel("x (px)", fontsize=9)
            ax_det.set_ylabel("y (px)", fontsize=9)
            iy, ix = _state["iy"], _state["ix"]
            best   = fit_result or result
            suffix = f"  — {best}" if best is not None else ""
            ax_det.set_title(
                f"Frame {frame_idx}  (iy={iy}, ix={ix}){suffix}",
                fontsize=8,
            )

            if _state["drawn"]:
                ax_det.set_xlim(saved_xlim)
                ax_det.set_ylim(saved_ylim)
            _state["drawn"] = True
            fig.canvas.draw_idle()

        # ── map click handler ─────────────────────────────────────────────────
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

            seg_path = os.path.join(seg_dir, f"frame_{frame_idx:05d}.h5")
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

            # Check if map already has a fitted U for this position
            existing_U = self.U[map_grain, iy, ix]
            has_existing = not np.any(np.isnan(existing_U))

            _state.update(
                frame_idx=frame_idx, iy=iy, ix=ix,
                obs_xy=obs_xy, result=None, fit_result=None,
            )
            enough = len(obs_xy) >= 3
            _info.value = (
                f"<span style='color:#aaa'>Frame {frame_idx} — "
                f"{len(obs_xy)} observed spots"
                + (" — existing fit in map" if has_existing else "")
                + "</span>"
            )
            btn_index.disabled  = not enough
            btn_fit.disabled    = not (enough and (has_existing or False))
            btn_strain.disabled = True
            btn_store.disabled  = True
            btn_save.disabled   = True
            _draw_det()

        fig.canvas.mpl_connect("button_press_event", _on_click)

        # ── ipywidgets ────────────────────────────────────────────────────────
        _bkw = dict(layout=ipw.Layout(width="120px", height="32px"))
        btn_index  = ipw.Button(description="⚡ Index",      button_style="primary",  **_bkw)
        btn_fit    = ipw.Button(description="⚡ Fit",        button_style="info",     **_bkw)
        btn_strain = ipw.Button(description="🔩 Fit strain", button_style="warning",
                                layout=ipw.Layout(width="130px", height="32px"))
        btn_store  = ipw.Button(description="⬆ Store",      button_style="success",  **_bkw)
        btn_save   = ipw.Button(description="💾 Save UB",   button_style="",         **_bkw)
        w_grain    = ipw.BoundedIntText(
            value=0, min=0, max=max(self.n_grains - 1, 0), step=1,
            layout=ipw.Layout(width="55px", height="32px"),
        )
        btn_index.disabled  = True
        btn_fit.disabled    = True
        btn_strain.disabled = True
        btn_store.disabled  = True
        btn_save.disabled   = True

        _info = ipw.HTML(
            "<span style='color:#666;font-style:italic'>"
            "click a map pixel to select a frame"
            "</span>",
            layout=ipw.Layout(margin="4px 0 0 6px"),
        )

        def _cb_index(_) -> None:
            import asyncio
            import queue as _qmod
            import threading

            if len(_state["obs_xy"]) < 3:
                return
            if getattr(_cb_index, "_running", False):
                return
            _cb_index._running    = True
            btn_index.disabled    = True
            btn_index.description = "Indexing…"

            q: _qmod.Queue = _qmod.Queue()

            def _run() -> None:
                try:
                    res = _index(
                        crystal, camera, _state["obs_xy"],
                        angle_tol_deg=angle_tol_deg,
                        min_match_rate=min_match_rate,
                        n_obs_use=n_obs_use,
                        verbose=True,
                    )
                    q.put(res)
                except Exception as exc:
                    q.put(exc)

            async def _wait() -> None:
                threading.Thread(target=_run, daemon=True).start()
                while q.empty():
                    await asyncio.sleep(0.15)
                item = q.get_nowait()
                if isinstance(item, Exception):
                    _info.value = (
                        f"<b style='color:#f44'>Index error: {item}</b>"
                    )
                else:
                    _state["result"] = item
                    col = "#44dd66" if item.success else "#ffaa33"
                    _info.value = f"<b style='color:{col}'>{item}</b>"
                    btn_fit.disabled  = len(_state["obs_xy"]) < 3
                    btn_save.disabled = False
                    _draw_det()
                btn_index.description = "⚡ Index"
                btn_index.disabled    = False
                _cb_index._running    = False

            try:
                asyncio.get_event_loop().create_task(_wait())
            except RuntimeError:
                asyncio.ensure_future(_wait())

        def _cb_fit(_) -> None:
            import asyncio
            import queue as _qmod
            import threading

            obs_xy = _state["obs_xy"]
            if len(obs_xy) < 3:
                return
            if getattr(_cb_fit, "_running", False):
                return
            _cb_fit._running   = True
            btn_fit.disabled   = True
            btn_fit.description = "Fitting…"

            # Starting U: prefer index result, fall back to existing map U
            if _state["result"] is not None:
                U0 = _state["result"].U
            else:
                U0 = self.U[map_grain, _state["iy"], _state["ix"]]

            q: _qmod.Queue = _qmod.Queue()

            def _run() -> None:
                try:
                    res = _fit_ori(
                        crystal, camera, obs_xy, U0,
                        E_min_eV=E_min_eV, E_max_eV=E_max_eV,
                        max_match_px=fit_max_match_px,
                        verbose=True,
                    )
                    q.put(res)
                except Exception as exc:
                    q.put(exc)

            async def _wait() -> None:
                threading.Thread(target=_run, daemon=True).start()
                while q.empty():
                    await asyncio.sleep(0.15)
                item = q.get_nowait()
                if isinstance(item, Exception):
                    _info.value = f"<b style='color:#f44'>Fit error: {item}</b>"
                else:
                    _state["fit_result"] = item
                    col = "#44aaff" if item.success else "#ffaa33"
                    _info.value = f"<b style='color:{col}'>{item}</b>"
                    btn_strain.disabled = len(_state["obs_xy"]) < 3
                    btn_store.disabled  = False
                    btn_save.disabled   = False
                    _draw_det()
                btn_fit.description = "⚡ Fit"
                btn_fit.disabled    = False
                _cb_fit._running    = False

            try:
                asyncio.get_event_loop().create_task(_wait())
            except RuntimeError:
                asyncio.ensure_future(_wait())

        def _cb_store(_) -> None:
            fit_result = _state["fit_result"]
            if fit_result is None:
                return
            iy   = _state["iy"]
            ix   = _state["ix"]
            gi   = int(w_grain.value)
            if gi < 0 or gi >= self.n_grains:
                _info.value = (
                    f"<b style='color:#f44'>Grain {gi} out of range "
                    f"(0 – {self.n_grains - 1})</b>"
                )
                return
            self.set_result(iy, ix, gi, fit_result)
            if self.save_path:
                self.save(self.save_path)
                save_note = f" — saved to {os.path.basename(self.save_path)}"
            else:
                save_note = " — <b style='color:#ffaa33'>no save_path set, results in memory only</b>"
            _info.value = (
                f"<b style='color:#44dd66'>Stored → grain {gi + 1} "
                f"(iy={iy}, ix={ix})</b>&emsp;{fit_result}"
                f"<span style='color:#aaa'>{save_note}</span>"
            )
            print(
                f"  ⬆ Stored fit result at (iy={iy}, ix={ix}) grain={gi}  "
                f"rms={fit_result.rms_px:.2f} px  "
                f"match={fit_result.match_rate:.0%}"
                + (f"  → saved to {self.save_path}" if self.save_path else "  (in memory only)")
            )

        def _cb_save(_) -> None:
            # Prefer the refined fit; fall back to index result
            best = _state["fit_result"] or _state["result"]
            if best is None:
                return
            existing = glob.glob(os.path.join(os.getcwd(), "UB[0-9]*.npy"))
            max_n = -1
            for fpath in existing:
                m = re.search(r"UB(\d+)\.npy$", os.path.basename(fpath))
                if m:
                    max_n = max(max_n, int(m.group(1)))
            fname    = f"UB{max_n + 1:02d}.npy"
            np.save(fname, best.U)
            abs_path = os.path.abspath(fname)
            print(f"  💾 Saved U → {abs_path}")
            _info.value = (
                f"<b style='color:#44dd66'>Saved → {fname}</b>"
                f"&emsp;{_info.value}"
            )

        def _cb_strain(_) -> None:
            import asyncio
            import queue as _qmod
            import threading

            fit_result = _state["fit_result"]
            obs_xy     = _state["obs_xy"]
            if fit_result is None or len(obs_xy) < 3:
                return
            if getattr(_cb_strain, "_running", False):
                return
            _cb_strain._running    = True
            btn_strain.disabled    = True
            btn_strain.description = "Fitting…"

            U0 = fit_result.U
            q: _qmod.Queue = _qmod.Queue()

            def _run() -> None:
                try:
                    res = _fit_strain(
                        crystal, camera, obs_xy, U0,
                        E_min_eV=E_min_eV, E_max_eV=E_max_eV,
                        max_match_px=fit_max_match_px,
                        verbose=True,
                    )
                    q.put(res)
                except Exception as exc:
                    q.put(exc)

            async def _wait() -> None:
                threading.Thread(target=_run, daemon=True).start()
                while q.empty():
                    await asyncio.sleep(0.15)
                item = q.get_nowait()
                if isinstance(item, Exception):
                    _info.value = f"<b style='color:#f44'>Strain fit error: {item}</b>"
                else:
                    _state["fit_result"] = item
                    col = "#ffcc44" if item.success else "#ffaa33"
                    _info.value = f"<b style='color:{col}'>{item}</b>"
                    btn_store.disabled = False
                    btn_save.disabled  = False
                    _draw_det()
                btn_strain.description = "🔩 Fit strain"
                btn_strain.disabled    = False
                _cb_strain._running    = False

            try:
                asyncio.get_event_loop().create_task(_wait())
            except RuntimeError:
                asyncio.ensure_future(_wait())

        btn_index.on_click(_cb_index)
        btn_fit.on_click(_cb_fit)
        btn_strain.on_click(_cb_strain)
        btn_store.on_click(_cb_store)
        btn_save.on_click(_cb_save)

        _controls = ipw.VBox([
            ipw.HBox(
                [btn_index, btn_fit, btn_strain,
                 ipw.HTML("<span style='color:#aaa;align-self:center'>grain:</span>",
                          layout=ipw.Layout(margin="0 2px 0 10px")),
                 w_grain, btn_store, btn_save],
                layout=ipw.Layout(gap="6px", margin="4px 0 0 0",
                                  align_items="center"),
            ),
            _info,
        ], layout=ipw.Layout(padding="6px 8px"))

        _ipy_display(ipw.VBox([fig.canvas, _controls]))

    def reindex_frame_manual(
        self,
        crystal,
        camera,
        base_dir: str,
        *,
        h5_dataset: "str | None" = None,
        tiff_dir: "str | None" = None,
        map_quantity: str = "match_rate",
        map_grain: "int | None" = None,
        motor_x: "str | None" = None,
        motor_y: "str | None" = None,
        motor_units: "dict | None" = None,
        E_min_eV: float = 5000.0,
        E_max_eV: float = 27000.0,
        fit_max_match_px: "float | list[float]" = [30.0, 10.0, 3.0],
        r_squared_min: float = 0.0,
        include_unfitted: bool = True,
        click_radius_px: float = 25.0,
        angle_tol_deg: float = 0.5,
        min_match_rate: float = 0.25,
        n_obs_use: int = 20,
        remove_match_px: float = 5.0,
        figsize: tuple = (14, 7),
    ) -> None:
        """
        Interactive manual re-indexer using click-to-pair spot assignment.

        When the automatic indexer confuses two grains with similar orientation,
        use this widget to manually tell it which simulated reflection belongs to
        which observed spot:

        1. **Click the map** to select a frame (observed and simulated spots appear).
        2. Optionally click **⚡ Index** to auto-index the observed spots.
        3. **Click a simulated spot** (orange diamond) — it turns yellow and waits.
        4. **Click the matching observed spot** (white circle) — the pair is
           registered and shown as a green line with an hkl label.
        5. Repeat until you have ≥ 3 pairs.
        6. **⚡ Fit pairs** — refines U using only the manually assigned pairs.
        7. **🔬 Refine all** — full refinement against all observed spots,
           starting from the pair-fitted U.
        8. **⬆ Store** — writes the result into the map.
        9. **✂ Remove spots** — removes the matched spots from the current
           observation list and resets the state so you can index the next grain
           from the remaining spots.  Only enabled after a successful *Store*.

        Right-click on the detector panel cancels a pending sim-spot selection.

        Args:
            crystal (Crystal): xrayutilities crystal structure.
            camera (Camera): Detector geometry.
            base_dir (str): Processing directory containing the `seg/` sub-folder.
            h5_dataset (str or None): HDF5 dataset path for the image stack.  Mutually exclusive with
                *tiff_dir*.
            tiff_dir (str or None): Path to `img_<n>.tif` files.  Mutually exclusive with
                *h5_dataset*.
            map_quantity (str): Scalar shown on the left panel.  One of `'match_rate'`,
                `'rms_px'`, `'mean_px'`, `'cost'`.
            map_grain (int or None): Grain slot for the left-panel map.  `None` uses the merged slot
                or grain 0.
            motor_x, motor_y (str or None): Motor names for axis labels and click conversion.
            motor_units (dict or None): Units per motor, e.g. `{'pz': 'mm', 'py': 'mm'}`.
            E_min_eV, E_max_eV (float): Energy range for spot simulation.  The allowed-HKL sphere
                cutoff is derived automatically from `E_max_eV`.
            fit_max_match_px (float or list of float): Match-radius schedule for *Refine all*.  Default `[30, 10, 3]`.
            r_squared_min (float): Minimum Gaussian R² when loading the spot file.  Default `0`.
            include_unfitted (bool): Include centroid-only spots from the spot file.  Default `True`.
            click_radius_px (float): Maximum pixel distance for a click to select a sim or obs spot.
                Default `25`.
            angle_tol_deg (float): Angular tolerance for the auto-indexer.  Default `0.5`.
            min_match_rate (float): Minimum match rate for the auto-indexer to report success.
                Default `0.25`.
            n_obs_use (int): Number of brightest observed spots passed to the auto-indexer.
                Default `20`.
            remove_match_px (float): Pixel radius used to identify matched spots when *Remove spots* is
                clicked.  Spots within this distance of any simulated spot from the
                stored grain are removed.  Default `5`.
            figsize (tuple): Figure size.  Default `(14, 7)`.
"""
        import ipywidgets as ipw
        from IPython.display import display as _ipy_display
        from .simulation import simulate_laue as _sim_laue
        from .segmentation import convert_spotsfile2peaklist
        from .fitting import (
            index_orientation as _index,
            fit_orientation   as _fit_ori,
            _match_spots,
        )

        seg_dir = os.path.join(base_dir, "seg")
        if map_grain is None:
            map_grain = self._merged_grain if self._merged_grain is not None else 0

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
                    pat = _re.compile(r'^img_(\d+)\.tif$', _re.IGNORECASE)
                    files = []
                    for fname in os.listdir(tiff_dir):
                        m = pat.match(fname)
                        if m:
                            files.append((int(m.group(1)), os.path.join(tiff_dir, fname)))
                    files.sort(key=lambda t: t[0])
                    _tiff_index.extend(p for _, p in files)
                if frame_idx >= len(_tiff_index):
                    return None
                try:
                    import skimage.io
                    return skimage.io.imread(_tiff_index[frame_idx]).astype(np.float32)
                except Exception:
                    return None
            elif h5_dataset is not None:
                try:
                    with h5py.File(self.h5_path, "r") as fh:
                        return fh[h5_dataset][frame_idx].astype(np.float32)
                except Exception:
                    return None
            return None

        _map_opts = {
            "match_rate": (self.match_rate[map_grain], "Match rate",    "viridis"),
            "rms_px":     (self.rms_px[map_grain],     "RMS (px)",      "plasma_r"),
            "mean_px":    (self.mean_px[map_grain],     "Mean dev (px)", "plasma_r"),
            "cost":       (self.cost[map_grain],        "Cost",          "plasma_r"),
        }
        map_data, map_label, map_cmap = _map_opts.get(
            map_quantity, (self.match_rate[map_grain], "Match rate", "viridis")
        )

        with plt.ioff():
            fig = plt.figure(figsize=figsize)
        try:
            fig.canvas.manager.set_window_title("Laue — manual re-index")
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
        ax_map.set_title(
            f"Click to select — {map_label}  (grain {map_grain + 1})",
            fontsize=9,
        )
        sel_dot, = ax_map.plot([], [], "w+", ms=11, mew=2.0, zorder=10)

        ax_det.set_facecolor("k")
        ax_det.set_xlim(0, camera.Nh)
        ax_det.set_ylim(camera.Nv, 0)
        ax_det.set_aspect("equal")
        ax_det.set_xlabel("x (px)", fontsize=9)
        ax_det.set_ylabel("y (px)", fontsize=9)
        ax_det.set_title("← click map to select a frame", fontsize=9, color="#888")
        fig.suptitle(
            "Manual re-index  —  click sim spot → click obs spot  (≥3 pairs) → Fit pairs → Refine all → Store",
            fontsize=9, color="#555",
        )

        _state: dict = {
            "frame_idx":      None,
            "iy":             None,
            "ix":             None,
            "obs_xy":         np.empty((0, 2)),
            "sim_spots":      [],    # list of spot dicts (hkl, pix, …)
            "U0":             None,  # current working orientation matrix
            "pairs":          [],    # list of {"hkl", "obs_xy", "sim_xy"}
            "pending_hkl":    None,  # sim hkl awaiting obs assignment
            "pending_sim_xy": None,
            "fit_result":     None,
            "stored_result":  None,  # last result written via Store (for Remove)
            "drawn":          False,
        }

        # ── simulation helper ─────────────────────────────────────────────────
        def _run_simulation(U: np.ndarray) -> None:
            try:
                spots = _sim_laue(
                    crystal, U, camera,
                    E_min=E_min_eV, E_max=E_max_eV,
                )
                _state["sim_spots"] = [s for s in spots if s.get("pix") is not None]
            except Exception as exc:
                print(f"  simulation error: {exc}", flush=True)
                _state["sim_spots"] = []

        # ── detector panel draw ───────────────────────────────────────────────
        def _draw_det() -> None:
            frame_idx      = _state["frame_idx"]
            obs_xy         = _state["obs_xy"]
            sim_spots      = _state["sim_spots"]
            pairs          = _state["pairs"]
            pending_hkl    = _state["pending_hkl"]
            pending_sim_xy = _state["pending_sim_xy"]
            fit_result     = _state["fit_result"]

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
                    origin="upper",
                    extent=[0, nh_im, nv_im, 0],
                    cmap="gray", aspect="equal", zorder=0,
                )
            else:
                ax_det.set_xlim(0, camera.Nh)
                ax_det.set_ylim(camera.Nv, 0)
            ax_det.set_aspect("equal")

            # ── observed spots ────────────────────────────────────────────
            if len(obs_xy):
                ax_det.scatter(
                    obs_xy[:, 0], obs_xy[:, 1],
                    s=40, c="none", edgecolors="white", linewidths=0.8,
                    zorder=4, label=f"observed ({len(obs_xy)})",
                )

            # ── simulated spots ───────────────────────────────────────────
            paired_hkls = {tuple(p["hkl"]) for p in pairs}
            if sim_spots:
                sim_xy_arr = np.array([s["pix"] for s in sim_spots])
                colors, sizes = [], []
                for s in sim_spots:
                    hkl = s["hkl"]
                    if hkl == pending_hkl:
                        colors.append("#ffff00")   # yellow = selected/pending
                        sizes.append(90)
                    elif tuple(hkl) in paired_hkls:
                        colors.append("#44dd66")   # green = already paired
                        sizes.append(55)
                    else:
                        colors.append("#ff6b35")   # orange = free
                        sizes.append(25)
                ax_det.scatter(
                    sim_xy_arr[:, 0], sim_xy_arr[:, 1],
                    s=sizes, c=colors, marker="D",
                    linewidths=0, zorder=5, alpha=0.85,
                    label=f"simulated ({len(sim_spots)})",
                )

            # ── confirmed pairs: line + hkl label ────────────────────────
            for p in pairs:
                ox, oy = p["obs_xy"]
                sx, sy = p["sim_xy"]
                ax_det.plot(
                    [ox, sx], [oy, sy],
                    color="#44dd66", lw=1.2, zorder=6,
                )
                ax_det.annotate(
                    str(p["hkl"]),
                    xy=(ox, oy), fontsize=6, color="#44dd66", zorder=7,
                    xytext=(3, 3), textcoords="offset points",
                )

            # ── pending sim-spot annotation ───────────────────────────────
            if pending_hkl is not None and pending_sim_xy is not None:
                ax_det.annotate(
                    f"{pending_hkl} → ?",
                    xy=pending_sim_xy, fontsize=7, color="#ffff00", zorder=8,
                    xytext=(5, 5), textcoords="offset points",
                )

            if sim_spots or len(obs_xy):
                ax_det.legend(
                    fontsize=7, loc="upper right",
                    facecolor="#111", edgecolor="#444", labelcolor="white",
                    framealpha=0.85,
                )

            ax_det.set_xlabel("x (px)", fontsize=9)
            ax_det.set_ylabel("y (px)", fontsize=9)
            iy, ix = _state["iy"], _state["ix"]
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

        # Declared early so closures below can reference them before the main
        # widget block.
        btn_index  = ipw.Button(description="⚡ Index",        button_style="primary",
                                layout=ipw.Layout(width="130px", height="32px"))
        btn_remove = ipw.Button(description="✂ Remove spots",  button_style="danger",
                                layout=ipw.Layout(width="140px", height="32px"))
        btn_index.disabled  = True
        btn_remove.disabled = True

        # ── pair-list display ─────────────────────────────────────────────────
        def _refresh_pair_list() -> None:
            pairs = _state["pairs"]
            if not pairs:
                w_pair_list.value = (
                    "<span style='color:#666;font-style:italic'>No pairs yet — "
                    "click a simulated spot then an observed spot</span>"
                )
            else:
                rows = "".join(
                    f"<tr>"
                    f"<td style='color:#aaa;padding:0 6px'>{i + 1}</td>"
                    f"<td style='color:#44dd66;padding:0 6px'>{p['hkl']}</td>"
                    f"<td style='color:#aaa;padding:0 6px'>"
                    f"({p['obs_xy'][0]:.0f}, {p['obs_xy'][1]:.0f})</td>"
                    f"</tr>"
                    for i, p in enumerate(pairs)
                )
                w_pair_list.value = (
                    f"<table style='font-size:11px;line-height:1.5;border-spacing:0'>"
                    f"<tr><th style='color:#888'>#</th>"
                    f"<th style='color:#888;padding:0 6px'>hkl</th>"
                    f"<th style='color:#888;padding:0 6px'>obs (px)</th></tr>"
                    f"{rows}</table>"
                )
            btn_fit_pairs.disabled = len(pairs) < 3

        # ── map click ─────────────────────────────────────────────────────────
        def _on_map_click(event) -> None:
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

            seg_path = os.path.join(seg_dir, f"frame_{frame_idx:05d}.h5")
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

            existing_U = self.U[map_grain, iy, ix]
            has_map_U  = not np.any(np.isnan(existing_U))
            U0 = existing_U if has_map_U else _state["U0"]

            _state.update(
                frame_idx=frame_idx, iy=iy, ix=ix,
                obs_xy=obs_xy, U0=U0,
                pairs=[], pending_hkl=None, pending_sim_xy=None,
                fit_result=None, stored_result=None, drawn=False,
            )

            if U0 is not None:
                _run_simulation(U0)
            else:
                _state["sim_spots"] = []

            n_sim = len(_state["sim_spots"])
            _info.value = (
                f"<span style='color:#aaa'>Frame {frame_idx} — "
                f"{len(obs_xy)} obs, {n_sim} sim"
                + (" — U from map" if has_map_U else
                   " — U from file" if U0 is not None else
                   " — <b style='color:#ffaa33'>no U loaded</b>")
                + "</span>"
            )
            btn_index.disabled  = len(obs_xy) < 3
            btn_store.disabled  = True
            btn_save.disabled   = True
            btn_remove.disabled = True
            _refresh_pair_list()
            _draw_det()

        # ── detector click ────────────────────────────────────────────────────
        def _on_det_click(event) -> None:
            if event.inaxes is not ax_det:
                return
            if event.xdata is None or event.ydata is None:
                return
            try:
                if fig.canvas.toolbar.mode != "":
                    return
            except Exception:
                pass
            if _state["frame_idx"] is None:
                return

            # Right-click cancels a pending sim-spot selection
            if event.button == 3:
                if _state["pending_hkl"] is not None:
                    _state["pending_hkl"]    = None
                    _state["pending_sim_xy"] = None
                    _info.value = "<span style='color:#aaa'>Selection cancelled.</span>"
                    _draw_det()
                return

            x, y      = event.xdata, event.ydata
            obs_xy    = _state["obs_xy"]
            sim_spots = _state["sim_spots"]

            if _state["pending_hkl"] is None:
                # ── Step 1: select a simulated spot ──────────────────────────
                if not sim_spots:
                    _info.value = (
                        "<span style='color:#ffaa33'>No simulated spots — "
                        "load a UB file first.</span>"
                    )
                    return
                sim_xy = np.array([s["pix"] for s in sim_spots])
                dists  = np.hypot(sim_xy[:, 0] - x, sim_xy[:, 1] - y)
                idx    = int(np.argmin(dists))
                if dists[idx] <= click_radius_px:
                    sel = sim_spots[idx]
                    _state["pending_hkl"]    = sel["hkl"]
                    _state["pending_sim_xy"] = list(sim_xy[idx])
                    _info.value = (
                        f"<span style='color:#ffff00'>"
                        f"Sim spot {sel['hkl']} selected "
                        f"({sim_xy[idx, 0]:.0f}, {sim_xy[idx, 1]:.0f}) — "
                        f"now click the matching observed spot "
                        f"(right-click to cancel)</span>"
                    )
                    _draw_det()
                else:
                    _info.value = (
                        f"<span style='color:#aaa'>"
                        f"No sim spot within {click_radius_px:.0f} px.</span>"
                    )
            else:
                # ── Step 2: assign to an observed spot ────────────────────────
                if len(obs_xy) == 0:
                    _state["pending_hkl"]    = None
                    _state["pending_sim_xy"] = None
                    _info.value = "<span style='color:#aaa'>No observed spots loaded.</span>"
                    _draw_det()
                    return

                dists = np.hypot(obs_xy[:, 0] - x, obs_xy[:, 1] - y)
                idx   = int(np.argmin(dists))
                if dists[idx] <= click_radius_px:
                    pair = {
                        "hkl":    _state["pending_hkl"],
                        "obs_xy": [float(obs_xy[idx, 0]), float(obs_xy[idx, 1])],
                        "sim_xy": _state["pending_sim_xy"],
                    }
                    _state["pairs"].append(pair)
                    n = len(_state["pairs"])
                    _info.value = (
                        f"<span style='color:#44dd66'>"
                        f"Pair {n}: {pair['hkl']} → "
                        f"({obs_xy[idx, 0]:.0f}, {obs_xy[idx, 1]:.0f})"
                        + (" — ready to fit!" if n >= 3 else
                           f" — need {3 - n} more pair(s)")
                        + "</span>"
                    )
                else:
                    _info.value = (
                        f"<span style='color:#ffaa33'>"
                        f"No obs spot within {click_radius_px:.0f} px — "
                        f"pair cancelled.  Click a sim spot again.</span>"
                    )
                _state["pending_hkl"]    = None
                _state["pending_sim_xy"] = None
                _refresh_pair_list()
                _draw_det()

        fig.canvas.mpl_connect("button_press_event", _on_map_click)
        fig.canvas.mpl_connect("button_press_event", _on_det_click)

        # ── async fit helper ──────────────────────────────────────────────────
        def _async_fit(run_fn, on_done_fn, btn, label_busy, label_idle) -> None:
            import asyncio
            import queue as _qmod
            import threading

            if getattr(btn, "_running", False):
                return
            btn._running    = True
            btn.disabled    = True
            btn.description = label_busy

            q: _qmod.Queue = _qmod.Queue()

            def _run() -> None:
                try:
                    q.put(run_fn())
                except Exception as exc:
                    q.put(exc)

            async def _wait() -> None:
                threading.Thread(target=_run, daemon=True).start()
                while q.empty():
                    await asyncio.sleep(0.15)
                item = q.get_nowait()
                on_done_fn(item)
                btn.description = label_idle
                btn.disabled    = False
                btn._running    = False

            try:
                asyncio.get_event_loop().create_task(_wait())
            except RuntimeError:
                asyncio.ensure_future(_wait())

        # ── widgets ───────────────────────────────────────────────────────────
        _bkw = dict(layout=ipw.Layout(width="130px", height="32px"))
        btn_fit_pairs = ipw.Button(description="⚡ Fit pairs",   button_style="warning", **_bkw)
        btn_refine    = ipw.Button(description="🔬 Refine all",  button_style="info",    **_bkw)
        btn_clear     = ipw.Button(description="🗑 Clear pairs", button_style="danger",
                                   layout=ipw.Layout(width="120px", height="32px"))
        btn_store     = ipw.Button(description="⬆ Store",        button_style="success", **_bkw)
        btn_save      = ipw.Button(description="💾 Save UB",     button_style="",        **_bkw)
        w_grain       = ipw.BoundedIntText(
            value=0, min=0, max=max(self.n_grains - 1, 0), step=1,
            layout=ipw.Layout(width="55px", height="32px"),
        )
        btn_fit_pairs.disabled = True
        btn_refine.disabled    = True
        btn_store.disabled     = True
        btn_save.disabled      = True

        _info = ipw.HTML(
            "<span style='color:#666;font-style:italic'>"
            "click a map pixel to select a frame</span>",
            layout=ipw.Layout(margin="4px 0 0 6px"),
        )
        w_pair_list = ipw.HTML(
            "<span style='color:#666;font-style:italic'>"
            "No pairs yet — click a simulated spot then an observed spot</span>",
            layout=ipw.Layout(margin="2px 0 0 6px"),
        )

        # ── UB file loader ────────────────────────────────────────────────────
        def _scan_ub_files() -> list:
            seen: dict[str, str] = {}
            for d in sorted({self.processing_dir, os.getcwd()}):
                for p in sorted(glob.glob(os.path.join(d, "UB[0-9]*.npy"))):
                    seen.setdefault(os.path.basename(p), p)
            return [("— select —", "")] + [
                (os.path.basename(p), p) for p in seen.values()
            ]

        w_ub_dd = ipw.Dropdown(
            options=_scan_ub_files(),
            value="",
            description="UB matrix:",
            layout=ipw.Layout(width="260px"),
            style={"description_width": "72px"},
        )
        btn_refresh_ub = ipw.Button(
            description="🔄", tooltip="Rescan for UB*.npy files",
            layout=ipw.Layout(width="38px", height="32px"),
        )

        def _load_ub_from_path(path: str) -> None:
            try:
                U = np.load(path)
                if U.shape != (3, 3):
                    raise ValueError(f"expected (3, 3), got {U.shape}")
                _state["U0"] = U
                if _state["frame_idx"] is not None:
                    _run_simulation(U)
                    n_sim = len(_state["sim_spots"])
                    _draw_det()
                    _info.value = (
                        f"<b style='color:#44dd66'>Loaded U from "
                        f"{os.path.basename(path)} — {n_sim} sim spots</b>"
                    )
                else:
                    _info.value = (
                        f"<b style='color:#44dd66'>Loaded U from "
                        f"{os.path.basename(path)}</b>"
                    )
                btn_refine.disabled = len(_state["obs_xy"]) < 3
            except Exception as exc:
                _info.value = f"<b style='color:#f44'>Load error: {exc}</b>"

        def _cb_ub_select(change) -> None:
            path = change["new"]
            if path:
                _load_ub_from_path(path)

        def _cb_refresh_ub(_) -> None:
            current = w_ub_dd.value
            opts = _scan_ub_files()
            w_ub_dd.options = opts
            valid_paths = {v for _, v in opts if v}
            w_ub_dd.value = current if current in valid_paths else ""

        w_ub_dd.observe(_cb_ub_select, names="value")
        btn_refresh_ub.on_click(_cb_refresh_ub)

        # ── button callbacks ──────────────────────────────────────────────────
        def _cb_clear(_) -> None:
            _state["pairs"]          = []
            _state["pending_hkl"]    = None
            _state["pending_sim_xy"] = None
            _refresh_pair_list()
            if _state["frame_idx"] is not None:
                _draw_det()
            _info.value = "<span style='color:#aaa'>Pairs cleared.</span>"

        def _cb_fit_pairs(btn) -> None:
            pairs = _state["pairs"]
            U0    = _state["U0"]
            if len(pairs) < 3 or U0 is None:
                return

            manual_obs = np.array([p["obs_xy"] for p in pairs])
            manual_hkl = {tuple(p["hkl"]) for p in pairs}

            def _run():
                return _fit_ori(
                    crystal, camera, manual_obs, U0,
                    E_min_eV=E_min_eV, E_max_eV=E_max_eV,
                    max_match_px=3000.0,
                    allowed_hkl=manual_hkl,
                    verbose=True,
                )

            def _on_done(item) -> None:
                if isinstance(item, Exception):
                    _info.value = f"<b style='color:#f44'>Fit error: {item}</b>"
                    return
                _state["fit_result"] = item
                _state["U0"]         = item.U
                _run_simulation(item.U)
                col = "#44aaff" if item.success else "#ffaa33"
                _info.value = (
                    f"<b style='color:{col}'>{item}</b>"
                    "<span style='color:#aaa'> — click 🔬 Refine all for full refinement</span>"
                )
                btn_refine.disabled = len(_state["obs_xy"]) < 3
                btn_store.disabled  = False
                btn_save.disabled   = False
                _draw_det()

            _async_fit(_run, _on_done, btn, "Fitting…", "⚡ Fit pairs")

        def _cb_refine(btn) -> None:
            obs_xy = _state["obs_xy"]
            U0     = _state["U0"]
            if len(obs_xy) < 3 or U0 is None:
                return

            def _run():
                return _fit_ori(
                    crystal, camera, obs_xy, U0,
                    E_min_eV=E_min_eV, E_max_eV=E_max_eV,
                    max_match_px=fit_max_match_px,
                    verbose=True,
                )

            def _on_done(item) -> None:
                if isinstance(item, Exception):
                    _info.value = f"<b style='color:#f44'>Refine error: {item}</b>"
                    return
                _state["fit_result"] = item
                _state["U0"]         = item.U
                _run_simulation(item.U)
                col = "#44aaff" if item.success else "#ffaa33"
                _info.value = f"<b style='color:{col}'>{item}</b>"
                btn_store.disabled = False
                btn_save.disabled  = False
                _draw_det()

            _async_fit(_run, _on_done, btn, "Refining…", "🔬 Refine all")

        def _cb_store(_) -> None:
            fit_result = _state["fit_result"]
            if fit_result is None:
                return
            iy = _state["iy"]
            ix = _state["ix"]
            gi = int(w_grain.value)
            if gi < 0 or gi >= self.n_grains:
                _info.value = (
                    f"<b style='color:#f44'>Grain {gi} out of range "
                    f"(0 – {self.n_grains - 1})</b>"
                )
                return
            self.set_result(iy, ix, gi, fit_result)
            _state["stored_result"] = fit_result
            btn_remove.disabled = False
            if self.save_path:
                self.save(self.save_path)
                save_note = f" — saved to {os.path.basename(self.save_path)}"
            else:
                save_note = " — <b style='color:#ffaa33'>no save_path set, results in memory only</b>"
            _info.value = (
                f"<b style='color:#44dd66'>Stored → grain {gi + 1} "
                f"(iy={iy}, ix={ix})</b>&emsp;{fit_result}"
                f"<span style='color:#aaa'>{save_note}"
                " — click ✂ Remove spots to isolate next grain</span>"
            )
            print(
                f"  ⬆ Stored fit result at (iy={iy}, ix={ix}) grain={gi}  "
                f"rms={fit_result.rms_px:.2f} px  "
                f"match={fit_result.match_rate:.0%}"
                + (f"  → saved to {self.h5_path}" if self.h5_path else "  (in memory only)")
            )

        def _cb_save(_) -> None:
            best = _state["fit_result"]
            if best is None:
                return
            existing = glob.glob(os.path.join(os.getcwd(), "UB[0-9]*.npy"))
            max_n = -1
            for fpath in existing:
                m = re.search(r"UB(\d+)\.npy$", os.path.basename(fpath))
                if m:
                    max_n = max(max_n, int(m.group(1)))
            fname    = f"UB{max_n + 1:02d}.npy"
            np.save(fname, best.U)
            abs_path = os.path.abspath(fname)
            print(f"  💾 Saved U → {abs_path}")
            _info.value = (
                f"<b style='color:#44dd66'>Saved → {fname}</b>"
                f"&emsp;{_info.value}"
            )
            # Refresh dropdown and select the newly saved file
            opts = _scan_ub_files()
            w_ub_dd.options = opts
            w_ub_dd.value = abs_path

        def _cb_index(btn) -> None:
            obs_xy = _state["obs_xy"]
            if len(obs_xy) < 3:
                return

            def _run():
                return _index(
                    crystal, camera, obs_xy,
                    angle_tol_deg=angle_tol_deg,
                    min_match_rate=min_match_rate,
                    n_obs_use=n_obs_use,
                    verbose=True,
                )

            def _on_done(item) -> None:
                if isinstance(item, Exception):
                    _info.value = f"<b style='color:#f44'>Index error: {item}</b>"
                    return
                _state["U0"] = item.U
                _run_simulation(item.U)
                col = "#44dd66" if item.success else "#ffaa33"
                _info.value = (
                    f"<b style='color:{col}'>{item}</b>"
                    "<span style='color:#aaa'> — refine with 🔬 or pair manually</span>"
                )
                btn_refine.disabled = len(_state["obs_xy"]) < 3
                btn_save.disabled   = False
                _draw_det()

            _async_fit(_run, _on_done, btn, "Indexing…", "⚡ Index")

        def _cb_remove(_) -> None:
            stored = _state["stored_result"]
            obs_xy = _state["obs_xy"]
            if stored is None or len(obs_xy) == 0:
                return

            try:
                spots = _sim_laue(
                    crystal, stored.U, camera,
                    E_min=E_min_eV, E_max=E_max_eV,
                )
                sim_xy_stored = np.array(
                    [s["pix"] for s in spots if s.get("pix") is not None]
                )
            except Exception as exc:
                _info.value = f"<b style='color:#f44'>Simulation error: {exc}</b>"
                return

            if len(sim_xy_stored) == 0:
                _info.value = "<span style='color:#aaa'>No simulated spots — nothing removed.</span>"
                return

            row_ind, _, dist_px = _match_spots(obs_xy, sim_xy_stored, remove_match_px)
            matched_mask = np.zeros(len(obs_xy), dtype=bool)
            matched_mask[row_ind[dist_px <= remove_match_px]] = True
            remaining = obs_xy[~matched_mask]
            n_removed = int(matched_mask.sum())

            _state.update(
                obs_xy=remaining, U0=None,
                sim_spots=[], pairs=[], pending_hkl=None, pending_sim_xy=None,
                fit_result=None, stored_result=None, drawn=False,
            )
            btn_remove.disabled   = True
            btn_store.disabled    = True
            btn_save.disabled     = True
            btn_refine.disabled   = True
            btn_fit_pairs.disabled = True
            btn_index.disabled    = len(remaining) < 3
            _refresh_pair_list()
            _draw_det()
            _info.value = (
                f"<b style='color:#44dd66'>Removed {n_removed} matched spots — "
                f"{len(remaining)} remaining.</b>"
                "<span style='color:#aaa'> Load a UB or click ⚡ Index for the next grain.</span>"
            )

        btn_index.on_click(_cb_index)
        btn_fit_pairs.on_click(_cb_fit_pairs)
        btn_refine.on_click(_cb_refine)
        btn_clear.on_click(_cb_clear)
        btn_store.on_click(_cb_store)
        btn_save.on_click(_cb_save)
        btn_remove.on_click(_cb_remove)

        _controls = ipw.VBox([
            ipw.HBox(
                [ipw.HTML(
                     "<span style='color:#aaa;align-self:center;"
                     "font-size:11px;margin-right:4px'>Load U:</span>",
                 ),
                 w_ub_dd, btn_refresh_ub, btn_index],
                layout=ipw.Layout(gap="4px", align_items="center",
                                  margin="4px 0 2px 0"),
            ),
            ipw.HBox(
                [btn_fit_pairs, btn_refine, btn_clear,
                 ipw.HTML(
                     "<span style='color:#aaa;align-self:center'>grain:</span>",
                     layout=ipw.Layout(margin="0 2px 0 10px"),
                 ),
                 w_grain, btn_store, btn_save, btn_remove],
                layout=ipw.Layout(gap="6px", align_items="center",
                                  margin="2px 0 2px 0"),
            ),
            _info,
            w_pair_list,
        ], layout=ipw.Layout(padding="6px 8px"))

        _ipy_display(ipw.VBox([fig.canvas, _controls]))

    def segment_frame(
        self,
        base_dir: str,
        detector_mask: "np.ndarray | None" = None,
        *,
        h5_dataset: "str | None" = None,
        tiff_dir: "str | None" = None,
        map_quantity: str = "match_rate",
        map_grain: "int | None" = None,
        motor_x: "str | None" = None,
        motor_y: "str | None" = None,
        motor_units: "dict | None" = None,
        figsize: tuple = (14, 7),
    ) -> None:
        """
        Interactive single-frame segmentation tuner.

        Click a pixel on the left map panel to load the frame into the right
        panel, adjust parameters, then press **⚙ Segment** to preview the
        result.  When satisfied press **💾 Save** to write the spots file to
        `<base_dir>/seg/frame_NNNNN.h5`.

        **Segmentation methods**
        Selected via the **Method** dropdown in the widget:

        * **LoG** — Laplacian-of-Gaussian.  *Sigmas* controls the blob scales;
          supply a single value (e.g. `4`) or a comma-separated list
          (e.g. `2, 4, 8`) for multi-scale detection.
        * **WTH** — White top-hat transform.  *Disk radii* should exceed the
          spot size but stay below the background correlation length.
        * **Hybrid** — LoG and WTH combined (logical OR); use when spot sizes
          vary strongly across the detector.

        Args:
            base_dir (str): Processing directory.  Segmentation files are written to
                `<base_dir>/seg/`.
            detector_mask ((Nv, Nh) bool ndarray or None): Valid-pixel mask (`True` = active pixel).  `None` treats all
                pixels as valid.
            h5_dataset (str or None): HDF5 dataset path inside `self.h5_path` for the raw image stack.
                Mutually exclusive with *tiff_dir*; one of the two must be given.
            tiff_dir (str or None): Path to a folder of `img_<number>.tif` files sorted by frame
                index.  Mutually exclusive with *h5_dataset*.
            map_quantity (str): Scalar quantity shown on the left overview panel.  Options:

            `"match_rate"`
                Fraction of simulated reflections matched after indexing
                (default).  Useful to identify already-indexed pixels.
            `"rms_px"`
                RMS deviation of matched spot positions in pixels.
            `"mean_px"`
                Mean absolute deviation of matched spot positions in pixels.
            `"cost"`
                Indexing cost function value.
            `"n_obs"`
                Number of segmented spots per pixel, read from the `seg/`
                directory.  Pixels with no saved segmentation file appear as
                NaN (background colour).  The map refreshes live each time a
                frame is saved, making it easy to track segmentation coverage
                before running indexing.
            map_grain (int or None): Grain slot index used to build the left-panel map for quantities
                that are per-grain (`match_rate`, `rms_px`, `mean_px`,
                `cost`).  Ignored for `n_obs`.  Defaults to the merged-grain
                slot if one exists, otherwise grain 0.
            motor_x, motor_y (str or None): Motor names looked up in `self.motors`.  When both are supplied
                the map axes use physical motor coordinates and click-to-pixel
                conversion is done by nearest-neighbour search; otherwise pixel
                indices are used.
            motor_units (dict or None): Physical units for axis labels, e.g. `{'pz': 'mm', 'py': 'mm'}`.
            figsize (tuple): Matplotlib figure size `(width, height)` in inches.
                Default `(14, 7)`.
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

        if map_grain is None:
            map_grain = self._merged_grain if self._merged_grain is not None else 0

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
                    pat = _re.compile(r'^img_(\d+)\.tif$', _re.IGNORECASE)
                    files = []
                    for fname in os.listdir(tiff_dir):
                        m = pat.match(fname)
                        if m:
                            files.append((int(m.group(1)), os.path.join(tiff_dir, fname)))
                    files.sort(key=lambda t: t[0])
                    _tiff_index.extend(p for _, p in files)
                if frame_idx >= len(_tiff_index):
                    return None
                try:
                    import skimage.io
                    return skimage.io.imread(_tiff_index[frame_idx]).astype(np.float32)
                except Exception:
                    return None
            elif h5_dataset is not None:
                try:
                    with h5py.File(self.h5_path, "r") as fh:
                        return fh[h5_dataset][frame_idx].astype(np.float32)
                except Exception:
                    return None
            return None

        def _build_n_obs_data() -> np.ndarray:
            raw = self.load_n_obs_map(seg_dir)
            return np.where(raw >= 0, raw.astype(float), np.nan)

        def _grain_map(arr):
            if self.n_grains == 0:
                return np.full((self.ny, self.nx), np.nan)
            return arr[map_grain]

        _map_opts = {
            "match_rate": (lambda: _grain_map(self.match_rate), "Match rate",    "viridis"),
            "rms_px":     (lambda: _grain_map(self.rms_px),     "RMS (px)",      "plasma_r"),
            "mean_px":    (lambda: _grain_map(self.mean_px),    "Mean dev (px)", "plasma_r"),
            "cost":       (lambda: _grain_map(self.cost),       "Cost",          "plasma_r"),
            "n_obs":      (_build_n_obs_data,                   "N spots (seg)", "YlOrRd"),
        }
        _data_fn, map_label, map_cmap = _map_opts.get(
            map_quantity, (_map_opts["match_rate"][0], "Match rate", "viridis")
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
        _map_title = (
            f"Click to select — {map_label}"
            if map_quantity == "n_obs"
            else f"Click to select — {map_label}  (grain {map_grain + 1})"
        )
        ax_map.set_title(_map_title, fontsize=9)
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
            "image": None,      # raw frame
            "proc_image": None, # background-subtracted + filtered (set after segmentation)
            "props": None,
            "drawn": False,
        }

        # ── parse text → scalar or list ───────────────────────────────────────
        def _parse(s: str):
            parts = [x.strip() for x in s.split(",") if x.strip()]
            if not parts:
                raise ValueError(f"Empty parameter field: {s!r}")
            nums = []
            for p in parts:
                nums.append(float(p) if "." in p else int(p))
            return nums[0] if len(nums) == 1 else nums

        # ── draw the detector panel ───────────────────────────────────────────
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

            # Parse user vmin / vmax (blank → auto)
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

            if w_log_scale.value:
                disp_show = np.log1p(disp_norm * 1000)
            else:
                disp_show = disp_norm

            ax_det.imshow(
                disp_show,
                origin="upper", extent=[0, nh, nv, 0],
                cmap="gray", aspect="equal", zorder=0,
            )

            if props is not None and len(props) > 0:
                ys = np.array([p.centroid_weighted[0] for p in props])
                xs = np.array([p.centroid_weighted[1] for p in props])
                ax_det.scatter(
                    xs, ys, s=35, c="none", edgecolors="#44aaff",
                    linewidths=0.9, zorder=4,
                    label=f"spots ({len(props)})",
                )
                ax_det.legend(
                    fontsize=7, loc="upper right",
                    facecolor="#111", edgecolor="#444", labelcolor="white",
                    framealpha=0.85,
                )

            ax_det.set_aspect("equal")
            ax_det.set_xlabel("x (px)", fontsize=9)
            ax_det.set_ylabel("y (px)", fontsize=9)
            iy, ix = _state["iy"], _state["ix"]
            n_spots = len(props) if props else 0
            suffix  = f"  — {n_spots} spots" if props is not None else ""
            ax_det.set_title(
                f"Frame {_state['frame_idx']}  (iy={iy}, ix={ix}){suffix}",
                fontsize=9,
            )

            if _state["drawn"]:
                ax_det.set_xlim(saved_xlim)
                ax_det.set_ylim(saved_ylim)
            _state["drawn"] = True
            fig.canvas.draw_idle()

        # ── map click ─────────────────────────────────────────────────────────
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
            _state.update(frame_idx=frame_idx, iy=iy, ix=ix,
                          image=image, proc_image=None, props=None)
            btn_segment.disabled = image is None
            btn_save.disabled    = True
            _info.value = (
                f"<span style='color:#aaa'>Frame {frame_idx} loaded"
                + (" — no image data" if image is None else "")
                + "</span>"
            )
            _draw_det(props=None)

        fig.canvas.mpl_connect("button_press_event", _on_click)

        # ── ipywidgets — parameter controls ───────────────────────────────────
        _sk = dict(
            continuous_update=False,
            style={"description_width": "110px"},
            layout=ipw.Layout(width="320px"),
        )
        _isk = dict(
            style={"description_width": "90px"},
            layout=ipw.Layout(width="160px"),
        )

        w_method = ipw.Dropdown(
            options=["LoG", "WTH", "Hybrid"],
            value="LoG",
            description="Method:",
            layout=ipw.Layout(width="220px"),
            style={"description_width": "70px"},
        )

        # LoG-specific
        w_log_sigmas = ipw.Text(
            value="2, 4, 8", description="Sigmas:", **_sk,
            placeholder="e.g. 2, 4, 8  or  4",
        )
        # WTH-specific
        w_wth_radius = ipw.Text(
            value="5, 7", description="Disk radii:", **_sk,
            placeholder="e.g. 5, 7  or  7",
        )
        # Hybrid-specific (separate copies so values are independent)
        w_hyb_log = ipw.Text(
            value="2, 4, 8", description="LoG sigmas:", **_sk,
            placeholder="e.g. 2, 4, 8",
        )
        w_hyb_wth = ipw.Text(
            value="5, 7", description="WTH radii:", **_sk,
            placeholder="e.g. 5, 7",
        )

        # Shared
        w_thresh = ipw.FloatText(
            value=99.9, description="Threshold %:", **_sk,
        )
        w_bg_sigma = ipw.FloatText(
            value=5.0, description="BG sigma:", **_sk,
        )

        # Clean segmentation
        w_min_size  = ipw.IntText(value=3,   description="min_size:",   **_isk)
        w_max_size  = ipw.IntText(value=500, description="max_size:",   **_isk)
        w_gap_excl  = ipw.IntText(value=3,   description="gap_exclude:",**_isk)
        w_gap_clos  = ipw.IntText(value=3,   description="gap_closing:",**_isk)

        # Save params
        w_d          = ipw.IntText(value=10,  description="Crop d (px):", **_isk)
        w_r2         = ipw.FloatText(value=0.9, description="R² min:",   **_isk)
        w_fit_spots  = ipw.Checkbox(
            value=True, description="Fit spots (Gaussian)",
            layout=ipw.Layout(width="200px"),
            style={"description_width": "initial"},
        )
        w_overwrite  = ipw.Checkbox(
            value=True, description="Overwrite existing",
            layout=ipw.Layout(width="200px"),
        )

        # Method-specific containers (show/hide)
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

        # ── buttons ───────────────────────────────────────────────────────────
        _bkw = dict(layout=ipw.Layout(width="130px", height="32px"))
        btn_segment = ipw.Button(description="⚙ Segment", button_style="primary", **_bkw)
        btn_save    = ipw.Button(description="💾 Save",   button_style="success", **_bkw)
        btn_segment.disabled = True
        btn_save.disabled    = True

        _info = ipw.HTML(
            "<span style='color:#666;font-style:italic'>"
            "click a map pixel to load a frame"
            "</span>",
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
                    method_kwargs = {
                        "sigmas": _parse(w_log_sigmas.value),
                        "threshold_percentile": w_thresh.value,
                    }
                elif method == "WTH":
                    method_kwargs = {
                        "disk_radius": _parse(w_wth_radius.value),
                        "threshold_percentile": w_thresh.value,
                    }
                else:  # Hybrid
                    method_kwargs = {
                        "log_sigmas": _parse(w_hyb_log.value),
                        "wth_disk_radius": _parse(w_hyb_wth.value),
                        "threshold_percentile": w_thresh.value,
                    }
            except ValueError as exc:
                _info.value = f"<b style='color:#f44'>Parameter error: {exc}</b>"
                btn_segment.description = "⚙ Segment"
                btn_segment.disabled    = False
                _cb_segment._running    = False
                return

            clean_kw = dict(
                min_size=w_min_size.value, max_size=w_max_size.value,
                gap_exclude=w_gap_excl.value, gap_closing=w_gap_clos.value,
            )
            bg_sigma = w_bg_sigma.value
            _det_mask = (
                detector_mask.astype(bool)
                if detector_mask is not None
                else np.ones(image.shape, dtype=bool)
            )

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
                    overwrite=w_overwrite.value,
                    fit_spots=w_fit_spots.value,
                )
                _info.value = (
                    f"<b style='color:#44dd66'>Saved → {out_path}</b>"
                )
                print(f"  💾 Saved → {os.path.abspath(out_path)}")
                if map_quantity == "n_obs":
                    new_data = _build_n_obs_data()
                    im_map.set_data(new_data)
                    valid = new_data[np.isfinite(new_data)]
                    if valid.size:
                        im_map.set_clim(valid.min(), valid.max())
                    fig.canvas.draw_idle()
            except Exception as exc:
                _info.value = f"<b style='color:#f44'>Save error: {exc}</b>"

        # ── display controls ──────────────────────────────────────────────────
        _dsk = dict(
            style={"description_width": "40px"},
            layout=ipw.Layout(width="130px"),
            continuous_update=False,
        )
        w_log_scale = ipw.Checkbox(
            value=True, description="Log scale",
            layout=ipw.Layout(width="110px"),
            style={"description_width": "initial"},
        )
        w_vmin = ipw.Text(
            value="", description="vmin:", placeholder="auto",
            **_dsk,
        )
        w_vmax = ipw.Text(
            value="", description="vmax:", placeholder="auto (99th %)",
            **_dsk,
        )

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

        # ── layout ────────────────────────────────────────────────────────────
        _controls = ipw.VBox([
            ipw.HBox([w_method], layout=ipw.Layout(margin="4px 0 2px 0")),
            ipw.HBox([
                ipw.VBox([
                    box_log, box_wth, box_hybrid,
                    w_thresh, w_bg_sigma,
                ], layout=ipw.Layout(padding="0 16px 0 0")),
                ipw.VBox([
                    ipw.HTML("<b>Clean:</b>"),
                    ipw.HBox([w_min_size, w_max_size]),
                    ipw.HBox([w_gap_excl, w_gap_clos]),
                    ipw.HTML("<b>Save:</b>"),
                    ipw.HBox([w_d, w_r2]),
                    w_fit_spots,
                    w_overwrite,
                ]),
            ]),
            ipw.HBox(
                [btn_segment, btn_save],
                layout=ipw.Layout(gap="8px", margin="6px 0 0 0"),
            ),
            ipw.HBox(
                [ipw.HTML("<b style='line-height:26px;margin-right:6px'>Display:</b>"),
                 w_log_scale, w_vmin, w_vmax],
                layout=ipw.Layout(gap="6px", margin="4px 0 0 0", align_items="center"),
            ),
            _info,
        ], layout=ipw.Layout(padding="6px 8px"))

        _ipy_display(ipw.VBox([fig.canvas, _controls]))

    # ── Segmentation map ──────────────────────────────────────────────────────

    def load_n_obs_map(self, seg_dir: str) -> np.ndarray:
        """
        Build a `(ny, nx)` map of the number of segmented spots per pixel
        by scanning an existing segmentation directory.

        Reads the `n_spots` attribute written by
        :func:`~nrxrdct.laue.segmentation.write_h5_spotsfile`.  For files
        that pre-date this attribute the number of unique spot groups is
        counted directly from the HDF5 keys (slower but backwards-compatible).

        Args:
            seg_dir (str): Directory containing `frame_NNNNN.h5` segmentation files.

        Returns:
            n_obs ((ny, nx) int ndarray): Per-pixel spot count.  Pixels with no seg file are set to `-1`.
                The same array is stored on `self.n_obs` for subsequent use.
"""
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
                        indices = {k.split("_")[1] for k in fh.keys()
                                   if k.startswith("spot_")}
                        n_obs[iy, ix] = len(indices)
            except Exception:
                pass
        self.n_obs = n_obs
        return n_obs

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """
        Serialise the GrainMap to an HDF5 file.

        All numeric arrays are stored under `/grain_{i:02d}/` groups.
        Metadata (ny, nx, ub_files, h5_path, entry) go into `/meta`.
"""
        if self.h5_path and os.path.abspath(path) == os.path.abspath(self.h5_path):
            raise ValueError(
                f"save path {path!r} is the same as the input scan file (h5_path) — "
                "refusing to overwrite"
            )
        with h5py.File(path, "w") as f:
            meta = f.create_group("meta")
            meta.attrs["ny"]            = self.ny
            meta.attrs["nx"]            = self.nx
            meta.attrs["n_grains"]      = self.n_grains
            meta.attrs["h5_path"]       = self.h5_path or ""
            meta.attrs["entry"]         = self.entry
            meta.attrs["processing_dir"] = self.processing_dir
            meta.attrs["merged_grain"]  = self._merged_grain if self._merged_grain is not None else -1
            meta.create_dataset(
                "ub_files",
                data=np.array([os.path.basename(p) for p in self.ub_files],
                               dtype=h5py.string_dtype()),
            )
            if self.n_grains:
                meta.create_dataset("U_ref", data=self.U_ref)

            for motor, arr in self.motors.items():
                f.create_dataset(f"motors/{motor}", data=arr)

            for gi in range(self.n_grains):
                grp = f.create_group(f"grain_{gi:02d}")
                grp.create_dataset("U",             data=self.U[gi],             compression="gzip")
                grp.create_dataset("rms_px",        data=self.rms_px[gi],        compression="gzip")
                grp.create_dataset("mean_px",       data=self.mean_px[gi],       compression="gzip")
                grp.create_dataset("n_matched",     data=self.n_matched[gi],     compression="gzip")
                grp.create_dataset("match_rate",    data=self.match_rate[gi],    compression="gzip")
                grp.create_dataset("cost",          data=self.cost[gi],          compression="gzip")
                grp.create_dataset("strain_voigt",             data=self.strain_voigt[gi],             compression="gzip")
                grp.create_dataset("strain_tensor",            data=self.strain_tensor[gi],            compression="gzip")
                grp.create_dataset("strain_tensor_deviatoric", data=self.strain_tensor_deviatoric[gi], compression="gzip")

        print(f"GrainMap saved → {os.path.abspath(path)}")

    def save_merged_result(
        self,
        path: str,
        grain: "int | None" = None,
        euler_convention: str = "ZXZ",
        compress: bool = True,
    ) -> None:
        """
        Export the final merged-grain result to a self-contained HDF5 file.

        Unlike :meth:`save` (which preserves the full multi-grain object for
        later reloading), this method writes a *flat*, human-readable file
        meant for downstream analysis tools (visualisation, strain maps, data
        exchange).  Every dataset is a plain array — no GrainMap class is
        needed to read it back.

        **Layout**
        `/meta`
            Scalar metadata (`ny`, `nx`, `grain_index`,
            `h5_path`, `processing_dir`).

        `/orientation/`
            `U`          — (ny, nx, 3, 3)  orientation matrices.
            `euler_ZXZ`  — (ny, nx, 3)     Euler angles in degrees
                             (or the chosen `euler_convention`).
            `U_ref`      — (3, 3)          reference orientation for
                             this grain (NaN if unavailable).

        `/fit_quality/`
            `match_rate`, `rms_px`, `mean_px`,
            `n_matched`, `cost` — all (ny, nx).

        `/strain/`
            `voigt`   — (ny, nx, 6)    deviatoric strain in Voigt notation
                          `[e_xx, e_yy, e_zz, e_xy, e_xz, e_yz]`.
            `tensor`  — (ny, nx, 3, 3) full strain tensor.
            Only written when at least one pixel has a fitted strain value.

        `/motors/`
            One dataset per motor name, each (ny, nx).

        Args:
            path (str): Output HDF5 file path.  Overwritten if it already exists.
            grain (int or None): Grain slot to export.  `None` (default) uses the merged slot
                set by :meth:`apply_merge`; falls back to grain 0 if no merge
                has been performed.
            euler_convention (str): Euler-angle convention passed to
                `scipy.spatial.transform.Rotation.as_euler`.  Default
                `"ZXZ"` (Bunge convention commonly used in EBSD).
            compress (bool): Apply gzip compression to numeric datasets.  Default `True`.
"""
        if grain is None:
            grain = self._merged_grain if self._merged_grain is not None else 0
        if not (0 <= grain < self.n_grains):
            raise ValueError(
                f"grain={grain} out of range (0 – {self.n_grains - 1})"
            )

        kw = dict(compression="gzip") if compress else {}

        # ── pre-compute Euler angles (vectorised, NaN-safe) ───────────────────
        U_gi   = self.U[grain]                          # (ny, nx, 3, 3)
        valid  = ~np.any(np.isnan(U_gi), axis=(-2, -1)) # (ny, nx)
        euler  = np.full((self.ny, self.nx, 3), np.nan)
        if valid.any():
            euler[valid] = Rotation.from_matrix(
                U_gi[valid]
            ).as_euler(euler_convention, degrees=True)

        # ── strain presence check ─────────────────────────────────────────────
        sv = self.strain_voigt[grain]
        has_strain = np.any(np.isfinite(sv))

        # ── U_ref for this grain ──────────────────────────────────────────────
        U_ref_gi = (
            self.U_ref[grain]
            if self.n_grains and self.U_ref.shape[0] > grain
            else np.full((3, 3), np.nan)
        )

        with h5py.File(path, "w") as f:
            # ── metadata ─────────────────────────────────────────────────────
            meta = f.create_group("meta")
            meta.attrs["ny"]             = self.ny
            meta.attrs["nx"]             = self.nx
            meta.attrs["grain_index"]    = grain
            meta.attrs["euler_convention"] = euler_convention
            meta.attrs["h5_path"]        = self.h5_path or ""
            meta.attrs["processing_dir"] = self.processing_dir or ""

            # ── orientation ───────────────────────────────────────────────────
            ori = f.create_group("orientation")
            ori.create_dataset("U",      data=U_gi,     **kw)
            ori.create_dataset(
                f"euler_{euler_convention}", data=euler, **kw
            )
            ori.create_dataset("U_ref",  data=U_ref_gi)

            # ── fit quality ───────────────────────────────────────────────────
            fq = f.create_group("fit_quality")
            fq.create_dataset("match_rate", data=self.match_rate[grain], **kw)
            fq.create_dataset("rms_px",     data=self.rms_px[grain],     **kw)
            fq.create_dataset("mean_px",    data=self.mean_px[grain],    **kw)
            fq.create_dataset("n_matched",  data=self.n_matched[grain],  **kw)
            fq.create_dataset("cost",       data=self.cost[grain],       **kw)

            # ── strain (only if present) ──────────────────────────────────────
            if has_strain:
                sg = f.create_group("strain")
                sg.create_dataset("voigt",  data=sv,                         **kw)
                sg.create_dataset("tensor", data=self.strain_tensor[grain],  **kw)

            # ── motor positions ───────────────────────────────────────────────
            if self.motors:
                mg = f.create_group("motors")
                for name, arr in self.motors.items():
                    mg.create_dataset(name, data=arr, **kw)

        n_fitted = int(valid.sum())
        print(
            f"Merged result saved → {os.path.abspath(path)}\n"
            f"  grain slot : {grain}\n"
            f"  fitted px  : {n_fitted} / {self.ny * self.nx} "
            f"({100 * n_fitted / (self.ny * self.nx):.1f} %)\n"
            f"  strain     : {'yes' if has_strain else 'no'}\n"
            f"  motors     : {list(self.motors) if self.motors else 'none'}"
        )

    @classmethod
    def load(cls, path: str) -> "GrainMap":
        """
        Restore a GrainMap from a file previously written by :meth:`save`.

        UB reference matrices and motor positions are re-read from the file;
        the `_results` list (which holds full Python objects) is not
        persisted and will be all-`None` after loading.
"""
        with h5py.File(path, "r") as f:
            meta = f["meta"]
            ny           = int(meta.attrs["ny"])
            nx           = int(meta.attrs["nx"])
            h5_path      = meta.attrs.get("h5_path") or None
            entry        = meta.attrs.get("entry", "1.1")
            processing_dir = meta.attrs.get("processing_dir", "")
            n_grains     = int(meta.attrs["n_grains"])

            obj = cls.__new__(cls)
            obj.ny             = ny
            obj.nx             = nx
            obj.h5_path        = h5_path if h5_path else None
            obj.save_path      = path
            obj.entry          = entry
            obj.processing_dir = processing_dir
            obj.n_grains       = n_grains

            if n_grains and "U_ref" in meta:
                obj.U_ref = np.array(meta["U_ref"])
            else:
                obj.U_ref = np.empty((0, 3, 3), dtype=float)

            raw_files = [s.decode() if isinstance(s, bytes) else s
                         for s in meta["ub_files"][()]]
            obj.ub_files = [
                os.path.join(processing_dir, fn) for fn in raw_files
            ]

            shape2d = (ny, nx)
            obj.U             = np.full((n_grains, *shape2d, 3, 3), np.nan)
            obj.rms_px        = np.full((n_grains, *shape2d), np.nan)
            obj.mean_px       = np.full((n_grains, *shape2d), np.nan)
            obj.n_matched     = np.full((n_grains, *shape2d), -1, dtype=int)
            obj.match_rate    = np.full((n_grains, *shape2d), np.nan)
            obj.cost          = np.full((n_grains, *shape2d), np.nan)
            obj.strain_voigt             = np.full((n_grains, *shape2d, 6), np.nan)
            obj.strain_tensor            = np.full((n_grains, *shape2d, 3, 3), np.nan)
            obj.strain_tensor_deviatoric = np.full((n_grains, *shape2d, 3, 3), np.nan)

            for gi in range(n_grains):
                grp = f[f"grain_{gi:02d}"]
                obj.U[gi]          = grp["U"][()]
                obj.rms_px[gi]     = grp["rms_px"][()]
                obj.mean_px[gi]    = grp["mean_px"][()] if "mean_px" in grp else np.full((ny, nx), np.nan)
                obj.n_matched[gi]  = grp["n_matched"][()]
                obj.match_rate[gi] = grp["match_rate"][()]
                obj.cost[gi]       = grp["cost"][()]
                if "strain_voigt" in grp:
                    obj.strain_voigt[gi]  = grp["strain_voigt"][()]
                if "strain_tensor" in grp:
                    obj.strain_tensor[gi] = grp["strain_tensor"][()]
                if "strain_tensor_deviatoric" in grp:
                    obj.strain_tensor_deviatoric[gi] = grp["strain_tensor_deviatoric"][()]
                elif "strain_tensor" in grp:
                    eps = obj.strain_tensor[gi]
                    tr  = np.trace(eps, axis1=-2, axis2=-1)[..., np.newaxis, np.newaxis]
                    obj.strain_tensor_deviatoric[gi] = eps - tr / 3.0 * np.eye(3)

            obj.motors = {}
            if "motors" in f:
                for motor in f["motors"].keys():
                    obj.motors[motor] = f[f"motors/{motor}"][()]

            obj._results = [
                [[None] * nx for _ in range(ny)]
                for _ in range(n_grains)
            ]
            mg = int(meta.attrs.get("merged_grain", -1))
            obj._merged_grain = mg if mg >= 0 else None

        return obj

    # ── SLURM cluster processing ──────────────────────────────────────────────

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

    def setup_processing_dirs(self, base_dir: str) -> dict:
        """Create and return a dict of processing subdirectory paths."""
        dirs = {
            "seg":        os.path.join(base_dir, "seg"),
            "ubs":        os.path.join(base_dir, "ubs"),
            "strain":     os.path.join(base_dir, "strain"),
            "slurm_logs": os.path.join(base_dir, "slurm_logs"),
            "job_meta":   os.path.join(base_dir, "job_meta"),
        }
        for d in dirs.values():
            os.makedirs(d, exist_ok=True)
        return dirs

    def _submit_jobs(
        self,
        job_name: str,
        worker_module: str,
        meta_json_path: str,
        frame_chunks: list,
        slurm_logs_dir: str,
        *,
        partition: str = "all",
        time: str = "01:00:00",
        mem: str = "4G",
        cpus_per_task: int = 1,
        python_bin: str = "python",
        extra_sbatch: dict | None = None,
    ) -> list:
        """Submit one SLURM job per chunk. Returns list of job IDs."""
        python_cmd = python_bin
        job_ids = []
        for i, chunk in enumerate(frame_chunks):
            indices_str = ",".join(str(fi) for fi in chunk)
            wrap_cmd = (
                f"{python_cmd} -m {worker_module} "
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
                    f"sbatch failed for {job_name} chunk {i}:\n{result.stderr}"
                )
            job_ids.append(result.stdout.strip().split()[-1])
        return job_ids

    @staticmethod
    def cancel_jobs(
        job_ids: "list[str | int]",
        *,
        dry_run: bool = False,
    ) -> None:
        """
        Cancel SLURM jobs by ID.

        Args:
            job_ids (list[str | int]): Job IDs returned by :meth:`submit_segmentation`,
                :meth:`submit_orientation`, or :meth:`submit_strain`.
            dry_run (bool): If `True`, print the `scancel` command without executing it.
"""
        if not job_ids:
            print("cancel_jobs: no job IDs provided.")
            return

        ids = [str(j) for j in job_ids]
        cmd = ["scancel"] + ids

        if dry_run:
            print("cancel_jobs (dry run):", " ".join(cmd))
            return

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"scancel failed:\n{result.stderr.strip()}"
            )
        print(f"cancel_jobs: cancelled {len(ids)} job(s): {', '.join(ids)}")

    @staticmethod
    def _seg_defaults(base_dir: str) -> dict:
        """Read r_squared_min / include_unfitted from seg_meta.json if present."""
        path = os.path.join(base_dir, "job_meta", "seg_meta.json")
        if not os.path.exists(path):
            return {}
        try:
            with open(path) as fh:
                m = json.load(fh)
            out = {}
            if "r_squared_min" in m:
                out["r_squared_min"] = m["r_squared_min"]
            if "include_unfitted" in m:
                out["include_unfitted"] = m["include_unfitted"]
            return out
        except Exception:
            return {}

    def submit_segmentation(
        self,
        base_dir: str,
        h5_dataset: "str | None" = None,
        n_jobs: int = 10,
        *,
        tiff_dir: "str | None" = None,
        partition: str = "all",
        time: str = "01:00:00",
        mem: str = "4G",
        cpus_per_task: int = 1,
        python_bin: str = "python",
        mask_path: str | None = None,
        method: str = "LoG",
        method_kwargs: dict | None = None,
        min_size: int = 3,
        max_size: int = 500,
        gap_exclude: int = 3,
        gap_closing: int = 3,
        bg_sigma: float = 251,
        max_components: int = 1,
        d: int = 10,
        r_squared_min: float = 0.9,
        include_unfitted: bool = False,
        extra_sbatch: dict | None = None,
    ) -> list:
        """
        Submit segmentation jobs to SLURM.

        Each job processes an assigned subset of frames and writes one HDF5
        spots file per frame under `base_dir/seg/`.  The pipeline is:

        1. Estimate and subtract a Gaussian background (sigma `bg_sigma`)
           from the raw frame — **used only for spot detection**.
        2. Detect spots with the chosen segmentation method.
        3. Clean the binary mask (size filter, border removal).
        4. Measure regionprops and fit a 2-D Gaussian to the **original**
           (unmodified) frame intensities inside a `(2d)×(2d)` ROI around
           each centroid.
        5. Write results to `seg_dir/frame_{idx:05d}.h5`.

        Args:
            base_dir (str): Root processing directory.  The sub-directories `seg/`,
                `ubs/`, `strain/`, `slurm_logs/`, and `job_meta/` are
                created automatically if they do not exist.
            h5_dataset (str or None): HDF5 dataset path inside `self.h5_path` that holds the image
                stack, e.g. `'1.1/measurement/det'`.  Mutually exclusive with
                *tiff_dir*; exactly one must be supplied.
            tiff_dir (str or None): Path to a directory containing one TIFF file per frame, named
                `img_<number>.tif` (e.g. `img_1500.tif`).  Files are sorted
                by their embedded number and mapped to 0-based frame indices in
                that order.  Motor positions are still read from `self.h5_path`
                as usual.  Mutually exclusive with *h5_dataset*.
            n_jobs (int): Number of SLURM array jobs.  Frames are split as evenly as
                possible across jobs.  Default `10`.
            partition (str): SLURM partition name.  Default `'all'`.
            time (str): Wall-clock time limit per job in `HH:MM:SS` format.
                Default `'01:00:00'`.
            mem (str): Memory per job, e.g. `'4G'`, `'16G'`.  Default `'4G'`.
            cpus_per_task (int): CPU cores requested per job.  Default `1`.
            python_bin (str): Python executable used in the `--wrap` command.  Default
                `'python'`.
            mask_path (str or None): Path to a `.npy` boolean array marking valid detector pixels
                (`True` = active).  `None` treats the whole frame as valid.
            method (str): Spot-detection algorithm:

            `'LoG'`
                Laplacian-of-Gaussian blob detector.  Good for round,
                diffuse spots.
            `'WTH'`
                White top-hat transform.  More robust on strong or uneven
                background.
            `'HYBRID'`
                LoG and WTH responses combined with a logical OR; best
                when the pattern contains both large and small spots.

            Default `'LoG'`.
            method_kwargs (dict or None): Extra keyword arguments forwarded to the segmentation function.
                Useful keys by method:

            * `'LoG'`: `sigmas`, `threshold_percentile`
            * `'WTH'`: `disk_radius`, `threshold_percentile`
            * `'HYBRID'`: `log_sigmas`, `wth_disk_radius`,
              `threshold_percentile`
            min_size (int): Minimum connected-component area in pixels; smaller blobs are
                discarded.  Default `3`.
            max_size (int): Maximum connected-component area in pixels; larger blobs are
                discarded.  Default `500`.
            gap_exclude (int): Width in pixels of the border region to clear before labelling
                (removes spots cut off by the detector edge).  Default `3`.
            gap_closing (int): Radius (pixels) of the disk used for binary closing of the
                detector mask **before** the gap-exclusion dilation.  Closing
                fills isolated dead pixels so that spots near a single bad pixel
                are not incorrectly excluded by the gap zone.  Set to `0` to
                disable closing (mask is used as-is).  Default `3`.
            bg_sigma (float): Gaussian sigma (pixels) for FFT-based background estimation.
                A large value (≥ several spot spacings) captures the slowly
                varying beam profile.  The subtracted frame is used only for
                segmentation; Gaussian fits are performed on the original
                intensities.  Default `251`.
            max_components (int): Maximum number of Gaussian components tried per spot during
                fitting.  `1` fits a single 2-D Gaussian; higher values
                attempt mixture models for overlapping spots.  Default `1`.
            d (int): Half-size in pixels of the square ROI cropped around each spot
                centroid for Gaussian fitting.  The crop window is
                `(2d) × (2d)` pixels.  Increase for large or diffuse spots;
                decrease to speed up fitting for small, sharp spots.
                Default `10`.
            r_squared_min (float): Minimum R² of the Gaussian fit for a spot to be accepted.
                Spots below this threshold are either skipped or stored without
                fit parameters (see *include_unfitted*).  This value is stored
                in `seg_meta.json` and used as the default for
                :meth:`submit_orientation` and :meth:`submit_strain` unless
                explicitly overridden there.  Default `0.9`.
            include_unfitted (bool): If `True`, spots whose best Gaussian fit has R² < *r_squared_min*
                are still written to the HDF5 file using the raw weighted
                centroid as position (shape parameters set to zero).  If
                `False`, those spots are silently discarded.  Stored in
                `seg_meta.json` and inherited by downstream workers.
                Default `False`.
            extra_sbatch (dict or None): Additional `sbatch` options passed as `--key=value` flags,
                e.g. `{'account': 'myproject', 'constraint': 'gpu'}`.

        Returns:
            list of str
                SLURM job IDs, one per submitted job.
"""
        if h5_dataset is None and tiff_dir is None:
            raise ValueError(
                "Provide either h5_dataset (HDF5 image stack) "
                "or tiff_dir (folder of img_*.tif files)."
            )
        if h5_dataset is not None and tiff_dir is not None:
            raise ValueError("Provide h5_dataset or tiff_dir, not both.")

        dirs = self.setup_processing_dirs(base_dir)
        all_frames = list(range(self.ny * self.nx))
        chunks = [
            list(map(int, c))
            for c in np.array_split(all_frames, min(n_jobs, len(all_frames)))
            if len(c) > 0
        ]
        meta = {
            "h5_path":        self.h5_path,
            "h5_dataset":     h5_dataset,
            "tiff_dir":       tiff_dir,
            "seg_dir":        dirs["seg"],
            "mask_path":      mask_path,
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
        }
        meta_path = os.path.join(dirs["job_meta"], "seg_meta.json")
        with open(meta_path, "w") as fh:
            json.dump(meta, fh, indent=2)

        job_ids = self._submit_jobs(
            "seg", "nrxrdct.laue.slurm_seg_worker", meta_path, chunks,
            dirs["slurm_logs"],
            partition=partition, time=time, mem=mem, cpus_per_task=cpus_per_task,
            python_bin=python_bin, extra_sbatch=extra_sbatch,
        )
        print(f"Segmentation: {len(job_ids)} jobs → {dirs['seg']}")
        return job_ids

    def submit_orientation(
        self,
        base_dir: str,
        crystal,
        camera,
        n_jobs: int = 10,
        *,
        partition: str = "all",
        time: str = "02:00:00",
        mem: str = "4G",
        cpus_per_task: int = 1,
        python_bin: str = "python",
        max_match_px=30.0,
        min_matched: int = 5,
        min_match_rate: float = 0.2,
        max_rms_px: float | None = None,
        r_squared_min: "float | None" = None,
        include_unfitted: "bool | None" = None,
        E_max_eV: float | None = None,
        f2_thresh: float | None = None,
        top_n_sim: int | None = None,
        top_n_obs: int | None = None,
        method: str = "lm",
        ftol: float = 1e-6,
        xtol: float = 1e-6,
        gtol: float = 1e-6,
        max_nfev: int | None = None,
        source: str | None = None,
        source_kwargs: dict | None = None,
        extra_sbatch: dict | None = None,
        overwrite: bool = False,
    ) -> list:
        """
        Submit orientation-fitting jobs to SLURM.

        Each job processes an assigned subset of frames.  For every frame the
        worker loads the observed spot list, then tries each `UB*.npy`
        reference matrix in :attr:`ub_files` independently.  A fit is saved
        only if it passes the quality thresholds (*min_matched*,
        *min_match_rate*, *max_rms_px*).  The pipeline is:

        1. Load observed spot positions from `seg_dir/frame_{idx:05d}.h5`
           (filtered by *r_squared_min* / *include_unfitted*).
        2. Precompute allowed HKL reflections once per SLURM job.
        3. For each `UB*.npy` reference matrix (grain index *gi*):

           a. Run :func:`~nrxrdct.laue.fitting.fit_orientation` with the
              staged *max_match_px* schedule.
           b. Accept the result only if `n_matched ≥ min_matched` **and**
              `match_rate ≥ min_match_rate` **and** (if set)
              `rms_px ≤ max_rms_px`.
           c. Write `ubs_dir/frame_{idx:05d}_g{gi:02d}.npz`.

        Results are collected into the map arrays by
        :meth:`collect_orientation`.

        Args:
            base_dir (str): Root processing directory — the same path used for
                :meth:`submit_segmentation`.  Sub-directories `seg/`,
                `ubs/`, `slurm_logs/`, and `job_meta/` are created if
                absent.
            crystal (Crystal or LayeredCrystal): Crystal structure object (xrayutilities `Crystal` or the
                project's :class:`~nrxrdct.laue.layers.LayeredCrystal`).
                Serialised with `dill` into `job_meta/crystal.pkl` and
                deserialised inside each worker process.
            camera (Camera): Detector geometry used for spot simulation.
            n_jobs (int): Number of SLURM array jobs.  Frames are split as evenly as
                possible.  Default `10`.
            partition (str): SLURM partition name.  Default `'all'`.
            time (str): Wall-clock time limit per job in `HH:MM:SS` format.
                Default `'02:00:00'`.
            mem (str): Memory per job, e.g. `'4G'`, `'16G'`.  Default `'4G'`.
            cpus_per_task (int): CPU cores requested per SLURM job.  Each job spawns a
                `ProcessPoolExecutor` that uses all allocated cores.
                Default `1`.
            python_bin (str): Python executable used in the `--wrap` command.
                Default `'python'`.
            max_match_px (float or list of float): Matching radius (pixels) for the spot-to-simulation assignment.
                Pass a list for staged matching: e.g. `[30, 10, 3]` starts
                with a loose radius to bootstrap the fit and tightens it in
                successive rounds.  A single float is wrapped in a list.
                Default `30.0`.
            min_matched (int): Minimum number of matched spots required to save a result.
                Frames with fewer spots than this value are skipped entirely.
                Default `5`.
            min_match_rate (float): Minimum match rate `n_matched / min(n_obs, n_sim)` required to
                accept a fit.  Default `0.2`.
            max_rms_px (float or None): Maximum allowed RMS residual in pixels.  `None` disables this
                filter.  Default `None`.
            r_squared_min (float or None): Minimum R² of the Gaussian fit for a spot to be loaded from the
                HDF5 spots file.  `None` inherits the value written by
                :meth:`submit_segmentation` in `seg_meta.json`; falls back to
                `0.9` if that file is absent.
            include_unfitted (bool or None): Whether to include spots whose Gaussian fit did not reach
                *r_squared_min* (stored as raw centroid positions).  `None`
                inherits from `seg_meta.json`; falls back to `False`.
            E_max_eV (float or None): High-energy cut-off (eV) for the Laue simulation.  The allowed-HKL
                sphere is derived automatically from this value.  `None` uses the
                fitting default (27 000 eV).
            f2_thresh (float or None): Minimum squared structure factor |F|² for a reflection to be
                included.  `None` uses the default (`1e-4`).
            top_n_sim (int or None): Keep only the *top_n_sim* strongest simulated spots per frame.
                `None` keeps all.
            top_n_obs (int or None): Keep only the *top_n_obs* brightest observed spots per frame.
                `None` keeps all.
            method (str): `scipy.optimize` method passed to
                :func:`~nrxrdct.laue.fitting.fit_orientation`.  `'lm'`
                (Levenberg–Marquardt) is fastest for unconstrained problems.
                Default `'lm'`.
            ftol (float): Relative tolerance on the cost function for convergence.
                Default `1e-6`.
            xtol (float): Relative tolerance on the parameter vector for convergence.
                Default `1e-6`.
            gtol (float): Tolerance on the gradient norm for convergence.  Default `1e-6`.
            max_nfev (int or None): Maximum number of function evaluations per fit.  `None` uses
                the scipy default (`100 * n_params`).
            source (str or None): X-ray source spectrum model forwarded to
                :func:`~nrxrdct.laue.simulation.simulate_laue`.  Common values:
                `'bending_magnet'`, `'wiggler'`.  `None` uses the
                simulation default.
            source_kwargs (dict or None): Extra keyword arguments for the source spectrum model.
            extra_sbatch (dict or None): Additional `sbatch` options passed as `--key=value` flags,
                e.g. `{'account': 'myproject', 'constraint': 'gpu'}`.

        Returns:
            list of str
                SLURM job IDs, one per submitted job.
"""
        dirs = self.setup_processing_dirs(base_dir)

        # Inherit filtering thresholds from the segmentation step if not set.
        _seg = self._seg_defaults(base_dir)
        if r_squared_min is None:
            r_squared_min = _seg.get("r_squared_min", 0.9)
        if include_unfitted is None:
            include_unfitted = _seg.get("include_unfitted", False)

        crystal_pkl = os.path.join(dirs["job_meta"], "crystal.pkl")
        with open(crystal_pkl, "wb") as fh:
            pickle.dump(crystal, fh)

        all_frames = list(range(self.ny * self.nx))
        chunks = [
            list(map(int, c))
            for c in np.array_split(all_frames, min(n_jobs, len(all_frames)))
            if len(c) > 0
        ]
        meta: dict = {
            "seg_dir":        dirs["seg"],
            "ubs_dir":        dirs["ubs"],
            "crystal_pkl":    crystal_pkl,
            "camera":         self._camera_to_dict(camera),
            "ub_files":       self.ub_files,
            "max_match_px":   max_match_px if isinstance(max_match_px, list)
                              else [float(max_match_px)],
            "min_matched":     min_matched,
            "min_match_rate":  min_match_rate,
            "max_rms_px":      max_rms_px,
            "r_squared_min":   r_squared_min,
            "include_unfitted": include_unfitted,
            "overwrite":       overwrite,
            "method":          method,
            "ftol":           ftol,
            "xtol":           xtol,
            "gtol":           gtol,
        }
        for key, val in [
            ("E_max_eV", E_max_eV), ("f2_thresh", f2_thresh),
            ("top_n_sim", top_n_sim), ("top_n_obs", top_n_obs),
            ("max_nfev", max_nfev), ("source", source),
            ("source_kwargs", source_kwargs),
        ]:
            if val is not None:
                meta[key] = val

        meta_path = os.path.join(dirs["job_meta"], "orient_meta.json")
        with open(meta_path, "w") as fh:
            json.dump(meta, fh, indent=2)

        job_ids = self._submit_jobs(
            "orient", "nrxrdct.laue.slurm_orient_worker", meta_path, chunks,
            dirs["slurm_logs"],
            partition=partition, time=time, mem=mem, cpus_per_task=cpus_per_task,
            python_bin=python_bin, extra_sbatch=extra_sbatch,
        )
        print(f"Orientation: {len(job_ids)} jobs → {dirs['ubs']}")
        return job_ids

    def submit_orientation_mixed(
        self,
        base_dir: str,
        crystals: list,
        camera,
        n_jobs: int = 10,
        *,
        shared: bool = False,
        partition: str = "all",
        time: str = "02:00:00",
        mem: str = "4G",
        cpus_per_task: int = 1,
        python_bin: str = "python",
        max_match_px=(30, 10, 3),
        min_matched: int = 5,
        min_match_rate: float = 0.2,
        max_rms_px: float | None = None,
        r_squared_min: "float | None" = None,
        include_unfitted: "bool | None" = None,
        f2_thresh: float | None = None,
        top_n_sim: int | None = None,
        top_n_obs: int | None = None,
        method: str = "lm",
        ftol: float = 1e-6,
        xtol: float = 1e-6,
        gtol: float = 1e-8,
        max_nfev: int | None = None,
        source: str | None = None,
        source_kwargs: dict | None = None,
        extra_sbatch: dict | None = None,
        overwrite: bool = False,
    ) -> list:
        """
        Submit mixed-phase orientation-fitting jobs to SLURM.

        All phases are fitted simultaneously for each frame using
        :func:`~nrxrdct.laue.fitting.fit_orientation_mixed`.  Results are
        written to ``base_dir/mixed/frame_{idx:05d}.npz``, one file per frame,
        containing the refined U matrix for every phase.

        Args:
            crystals (list of Crystal): One crystal per phase, in grain-index order.  Must match
                ``len(self.ub_files)`` and ``self.n_grains``.
            camera (Camera): Detector geometry.
            shared (bool): If ``True``, all phases share a single rotation vector (3 free
                parameters).  If ``False`` (default), each phase has an
                independent rotation (3 × N_phases parameters).
"""
        if len(crystals) != len(self.ub_files):
            raise ValueError(
                f"len(crystals)={len(crystals)} must equal "
                f"len(self.ub_files)={len(self.ub_files)}"
            )
        if len(crystals) != self.n_grains:
            raise ValueError(
                f"len(crystals)={len(crystals)} must equal "
                f"self.n_grains={self.n_grains}"
            )

        dirs = self.setup_processing_dirs(base_dir)
        mixed_dir = os.path.join(base_dir, "mixed")
        os.makedirs(mixed_dir, exist_ok=True)

        _seg = self._seg_defaults(base_dir)
        if r_squared_min is None:
            r_squared_min = _seg.get("r_squared_min", 0.9)
        if include_unfitted is None:
            include_unfitted = _seg.get("include_unfitted", False)

        crystals_pkl = os.path.join(dirs["job_meta"], "crystals.pkl")
        with open(crystals_pkl, "wb") as fh:
            pickle.dump(crystals, fh)

        all_frames = list(range(self.ny * self.nx))
        chunks = [
            list(map(int, c))
            for c in np.array_split(all_frames, min(n_jobs, len(all_frames)))
            if len(c) > 0
        ]
        meta: dict = {
            "seg_dir":        dirs["seg"],
            "mixed_dir":      mixed_dir,
            "crystals_pkl":   crystals_pkl,
            "camera":         self._camera_to_dict(camera),
            "ub_files":       self.ub_files,
            "shared":         shared,
            "max_match_px":   max_match_px if isinstance(max_match_px, list)
                              else list(max_match_px),
            "min_matched":    min_matched,
            "min_match_rate": min_match_rate,
            "max_rms_px":     max_rms_px,
            "r_squared_min":  r_squared_min,
            "include_unfitted": include_unfitted,
            "overwrite":      overwrite,
            "method":         method,
            "ftol":           ftol,
            "xtol":           xtol,
            "gtol":           gtol,
            "geometry_only":  True,
        }
        for key, val in [
            ("f2_thresh", f2_thresh), ("top_n_sim", top_n_sim),
            ("top_n_obs", top_n_obs), ("max_nfev", max_nfev),
            ("source", source), ("source_kwargs", source_kwargs),
        ]:
            if val is not None:
                meta[key] = val

        meta_path = os.path.join(dirs["job_meta"], "mixed_meta.json")
        with open(meta_path, "w") as fh:
            json.dump(meta, fh, indent=2)

        job_ids = self._submit_jobs(
            "mixed", "nrxrdct.laue.slurm_mixed_worker", meta_path, chunks,
            dirs["slurm_logs"],
            partition=partition, time=time, mem=mem, cpus_per_task=cpus_per_task,
            python_bin=python_bin, extra_sbatch=extra_sbatch,
        )
        print(f"Mixed orientation: {len(job_ids)} jobs → {mixed_dir}")
        return job_ids

    def collect_orientation_mixed(self, base_dir: str, n_workers: int | None = None) -> int:
        """
        Load mixed-phase orientation results into the map arrays.

        Reads ``base_dir/mixed/frame_{idx:05d}.npz`` files produced by the
        mixed worker or :func:`~nrxrdct.laue.fitting.run_orientation_mixed_local`.
        Fills ``self.U[gi]``, ``self.rms_px[gi]``, ``self.n_matched[gi]``,
        ``self.match_rate[gi]``, and ``self.cost[gi]`` for every grain index
        *gi* from 0 to N_phases-1.  Quality metrics are shared across all
        phases (joint fit).  Files are loaded in parallel using a thread pool.

        Args:
            base_dir (str): Root processing directory.
            n_workers (int or None): Number of threads.  Defaults to
                ``os.cpu_count()``.

        Returns:
            int: Number of frames loaded.
        """
        mixed_dir = os.path.join(base_dir, "mixed")
        files = sorted(glob.glob(os.path.join(mixed_dir, "frame_?????.npz")))

        def _load(fpath):
            m = re.search(r"frame_(\d{5})\.npz$", os.path.basename(fpath))
            if not m:
                return False
            frame_idx = int(m.group(1))
            iy, ix    = self.map_index(frame_idx)
            if iy >= self.ny or ix >= self.nx:
                return False
            try:
                d          = np.load(fpath)
                rms_px     = float(d["rms_px"])
                mean_px    = float(d["mean_px"]) if "mean_px" in d else np.nan
                n_matched  = int(d["n_matched"])
                match_rate = float(d["match_rate"])
                cost       = float(d["cost"])
                for gi in range(self.n_grains):
                    key = f"U_{gi}"
                    if key not in d:
                        break
                    self.U[gi, iy, ix]          = d[key]
                    self.rms_px[gi, iy, ix]     = rms_px
                    self.mean_px[gi, iy, ix]    = mean_px
                    self.n_matched[gi, iy, ix]  = n_matched
                    self.match_rate[gi, iy, ix] = match_rate
                    self.cost[gi, iy, ix]       = cost
                return True
            except Exception as exc:
                print(f"  ✗  {fpath}: {exc}", flush=True)
                return False

        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
            n_loaded = sum(pool.map(_load, files))
        print(f"collect_orientation_mixed: {n_loaded} frames loaded from {mixed_dir}", flush=True)
        return n_loaded

    def submit_strain(
        self,
        base_dir: str,
        crystal,
        camera,
        n_jobs: int = 10,
        *,
        partition: str = "all",
        time: str = "02:00:00",
        mem: str = "4G",
        cpus_per_task: int = 1,
        python_bin: str = "python",
        max_match_px=10.0,
        fit_strain: list | None = None,
        r_squared_min: "float | None" = None,
        include_unfitted: "bool | None" = None,
        E_max_eV: float | None = None,
        f2_thresh: float | None = None,
        top_n_sim: int | None = None,
        top_n_obs: int | None = None,
        method: str = "lm",
        ftol: float = 1e-6,
        xtol: float = 1e-6,
        gtol: float = 1e-6,
        max_nfev: int | None = None,
        strain_scale: float | None = None,
        source: str | None = None,
        source_kwargs: dict | None = None,
        extra_sbatch: dict | None = None,
        overwrite: bool = False,
    ) -> list:
        """
        Submit strain-fitting jobs to SLURM.

        Requires orientation results produced by :meth:`submit_orientation`
        (`base_dir/ubs/frame_*_g*.npz`).  For each frame and grain the
        worker refines both the orientation matrix **and** the six independent
        strain-tensor components simultaneously.  The pipeline is:

        1. Load observed spot positions from `seg_dir/frame_{idx:05d}.h5`
           (filtered by *r_squared_min* / *include_unfitted*).
        2. Precompute allowed HKL reflections once per SLURM job.
        3. For each grain index *gi*, load the orientation matrix U from
           `ubs_dir/frame_{idx:05d}_g{gi:02d}.npz`.
        4. Run :func:`~nrxrdct.laue.fitting.fit_strain_orientation` with the
           staged *max_match_px* schedule, fitting only the strain components
           listed in *fit_strain*.
        5. Write `strain_dir/frame_{idx:05d}_g{gi:02d}.npz` containing
           the updated U, strain tensor, and fit quality metrics.

        Results are collected into the map arrays by :meth:`collect_strain`.

        Args:
            base_dir (str): Root processing directory — the same path used for
                :meth:`submit_segmentation` and :meth:`submit_orientation`.
            crystal (Crystal or LayeredCrystal): Crystal structure object.  Reuses `job_meta/crystal.pkl` if it
                already exists from the orientation step; otherwise writes it.
            camera (Camera): Detector geometry.
            n_jobs (int): Number of SLURM array jobs.  Default `10`.
            partition (str): SLURM partition name.  Default `'all'`.
            time (str): Wall-clock time limit per job in `HH:MM:SS` format.
                Default `'02:00:00'`.
            mem (str): Memory per job, e.g. `'4G'`, `'16G'`.  Default `'4G'`.
            cpus_per_task (int): CPU cores requested per SLURM job.  Default `1`.
            python_bin (str): Python executable used in the `--wrap` command.
                Default `'python'`.
            max_match_px (float or list of float): Matching radius (pixels) for the spot-to-simulation assignment.
                Strain fitting starts from a good orientation, so a tighter
                default (`10.0`) is appropriate compared to the orientation
                step.  Pass a list for staged refinement, e.g. `[10, 3]`.
                Default `10.0`.
            fit_strain (list of str or None): Strain-tensor components to include in the fit.  Valid component
                names are `'e_xx'`, `'e_yy'`, `'e_zz'`, `'e_xy'`,
                `'e_xz'`, `'e_yz'`.  Components not listed are fixed at
                zero.  `None` fits all six components.
                Default `None` (all six).
            r_squared_min (float or None): Minimum R² of the Gaussian fit for a spot to be loaded.  `None`
                inherits from `seg_meta.json`; falls back to `0.9`.
            include_unfitted (bool or None): Whether to include spots stored as raw centroids (Gaussian fit
                failed).  `None` inherits from `seg_meta.json`; falls back
                to `False`.
            E_max_eV (float or None): High-energy cut-off (eV).  The allowed-HKL sphere is derived
                automatically from this value.  `None` uses the fitting
                default (27 000 eV).
            f2_thresh (float or None): Minimum |F|² for reflection inclusion.  `None` uses the default
                (`1e-4`).
            top_n_sim (int or None): Keep only the *top_n_sim* strongest simulated spots.  `None`
                keeps all.
            top_n_obs (int or None): Keep only the *top_n_obs* brightest observed spots.  `None`
                keeps all.
            method (str): `scipy.optimize` method for :func:`fit_strain_orientation`.
                Default `'lm'`.
            ftol (float): Relative tolerance on the cost function.  Default `1e-6`.
            xtol (float): Relative tolerance on the parameter vector.  Default `1e-6`.
            gtol (float): Gradient-norm tolerance.  Default `1e-6`.
            max_nfev (int or None): Maximum function evaluations per fit.  `None` uses the scipy
                default.
            strain_scale (float or None): Multiplicative scale applied to strain parameters inside the
                optimizer to improve conditioning (strain components are ~10⁻³
                while rotation angles are ~10⁻² rad).  `None` uses the
                :func:`fit_strain_orientation` default.
            source (str or None): X-ray source spectrum model.  `None` uses the simulation
                default.
            source_kwargs (dict or None): Extra keyword arguments for the source spectrum model.
            extra_sbatch (dict or None): Additional `sbatch` options, e.g.
                `{'account': 'myproject'}`.

        Returns:
            list of str
                SLURM job IDs, one per submitted job.
"""
        dirs = self.setup_processing_dirs(base_dir)

        # Inherit filtering thresholds from the segmentation step if not set.
        _seg = self._seg_defaults(base_dir)
        if r_squared_min is None:
            r_squared_min = _seg.get("r_squared_min", 0.9)
        if include_unfitted is None:
            include_unfitted = _seg.get("include_unfitted", False)

        crystal_pkl = os.path.join(dirs["job_meta"], "crystal.pkl")
        if not os.path.exists(crystal_pkl):
            with open(crystal_pkl, "wb") as fh:
                pickle.dump(crystal, fh)

        all_frames = list(range(self.ny * self.nx))
        chunks = [
            list(map(int, c))
            for c in np.array_split(all_frames, min(n_jobs, len(all_frames)))
            if len(c) > 0
        ]
        meta: dict = {
            "seg_dir":    dirs["seg"],
            "ubs_dir":    dirs["ubs"],
            "strain_dir": dirs["strain"],
            "crystal_pkl": crystal_pkl,
            "camera":     self._camera_to_dict(camera),
            "n_grains":   self.n_grains,
            "max_match_px": max_match_px if isinstance(max_match_px, list)
                            else [float(max_match_px)],
            "fit_strain":      fit_strain or
                               ["e_xx", "e_yy", "e_zz", "e_xy", "e_xz", "e_yz"],
            "r_squared_min":   r_squared_min,
            "include_unfitted": include_unfitted,
            "overwrite":       overwrite,
            "method":          method,
            "ftol":       ftol,
            "xtol":       xtol,
            "gtol":       gtol,
        }
        for key, val in [
            ("E_max_eV", E_max_eV), ("f2_thresh", f2_thresh),
            ("top_n_sim", top_n_sim), ("top_n_obs", top_n_obs),
            ("max_nfev", max_nfev), ("strain_scale", strain_scale),
            ("source", source), ("source_kwargs", source_kwargs),
        ]:
            if val is not None:
                meta[key] = val

        meta_path = os.path.join(dirs["job_meta"], "strain_meta.json")
        with open(meta_path, "w") as fh:
            json.dump(meta, fh, indent=2)

        job_ids = self._submit_jobs(
            "strain", "nrxrdct.laue.slurm_strain_worker", meta_path, chunks,
            dirs["slurm_logs"],
            partition=partition, time=time, mem=mem, cpus_per_task=cpus_per_task,
            python_bin=python_bin, extra_sbatch=extra_sbatch,
        )
        print(f"Strain: {len(job_ids)} jobs → {dirs['strain']}")
        return job_ids

    def submit_strain_mixed(
        self,
        base_dir: str,
        crystals: list,
        camera,
        n_jobs: int = 10,
        *,
        partition: str = "all",
        time: str = "02:00:00",
        mem: str = "4G",
        cpus_per_task: int = 1,
        python_bin: str = "python",
        max_match_px=(10, 3),
        fit_strain: list | None = None,
        r_squared_min: "float | None" = None,
        include_unfitted: "bool | None" = None,
        f2_thresh: float | None = None,
        top_n_sim: int | None = None,
        top_n_obs: int | None = None,
        method: str = "lm",
        ftol: float = 1e-6,
        xtol: float = 1e-6,
        gtol: float = 1e-6,
        max_nfev: int | None = None,
        strain_scale: float | None = None,
        source: str | None = None,
        source_kwargs: dict | None = None,
        extra_sbatch: dict | None = None,
        overwrite: bool = False,
    ) -> list:
        """
        Submit per-phase strain-fitting jobs to SLURM, starting from mixed
        orientation results.

        Reads ``base_dir/mixed/frame_{idx:05d}.npz`` for the starting U of
        each phase (produced by :meth:`submit_orientation_mixed`), then runs
        :func:`~nrxrdct.laue.fitting.fit_strain_orientation` independently
        per phase.  Output is written to
        ``strain_dir/frame_{idx:05d}_g{gi:02d}.npz`` — the same format as
        :meth:`submit_strain` — so :meth:`collect_strain` works unchanged.

        Args:
            crystals (list of Crystal): One crystal per phase, in grain-index order.  Must match
                ``self.n_grains``.
"""
        if len(crystals) != self.n_grains:
            raise ValueError(
                f"len(crystals)={len(crystals)} must equal "
                f"self.n_grains={self.n_grains}"
            )

        dirs = self.setup_processing_dirs(base_dir)
        mixed_dir = os.path.join(base_dir, "mixed")

        _seg = self._seg_defaults(base_dir)
        if r_squared_min is None:
            r_squared_min = _seg.get("r_squared_min", 0.9)
        if include_unfitted is None:
            include_unfitted = _seg.get("include_unfitted", False)

        crystals_pkl = os.path.join(dirs["job_meta"], "crystals.pkl")
        if not os.path.exists(crystals_pkl):
            with open(crystals_pkl, "wb") as fh:
                pickle.dump(crystals, fh)

        all_frames = list(range(self.ny * self.nx))
        chunks = [
            list(map(int, c))
            for c in np.array_split(all_frames, min(n_jobs, len(all_frames)))
            if len(c) > 0
        ]
        meta: dict = {
            "seg_dir":        dirs["seg"],
            "mixed_dir":      mixed_dir,
            "strain_dir":     dirs["strain"],
            "crystals_pkl":   crystals_pkl,
            "camera":         self._camera_to_dict(camera),
            "n_grains":       self.n_grains,
            "max_match_px":   max_match_px if isinstance(max_match_px, list)
                              else list(max_match_px),
            "fit_strain":     fit_strain or
                              ["e_xx", "e_yy", "e_zz", "e_xy", "e_xz", "e_yz"],
            "r_squared_min":  r_squared_min,
            "include_unfitted": include_unfitted,
            "overwrite":      overwrite,
            "method":         method,
            "ftol":           ftol,
            "xtol":           xtol,
            "gtol":           gtol,
            "geometry_only":  True,
        }
        for key, val in [
            ("f2_thresh", f2_thresh), ("top_n_sim", top_n_sim),
            ("top_n_obs", top_n_obs), ("max_nfev", max_nfev),
            ("strain_scale", strain_scale),
            ("source", source), ("source_kwargs", source_kwargs),
        ]:
            if val is not None:
                meta[key] = val

        meta_path = os.path.join(dirs["job_meta"], "mixed_strain_meta.json")
        with open(meta_path, "w") as fh:
            json.dump(meta, fh, indent=2)

        job_ids = self._submit_jobs(
            "mixed_strain", "nrxrdct.laue.slurm_mixed_strain_worker",
            meta_path, chunks, dirs["slurm_logs"],
            partition=partition, time=time, mem=mem, cpus_per_task=cpus_per_task,
            python_bin=python_bin, extra_sbatch=extra_sbatch,
        )
        print(f"Mixed strain: {len(job_ids)} jobs → {dirs['strain']}")
        return job_ids

    def collect_orientation(self, base_dir: str, n_workers: int | None = None) -> int:
        """
        Load orientation npz files produced by SLURM workers into the map arrays.

        Files are loaded in parallel using a thread pool.

        Args:
            base_dir (str): Root processing directory.
            n_workers (int or None): Number of threads.  Defaults to
                ``os.cpu_count()``.

        Returns:
            int: Number of results loaded.
        """
        ubs_dir = os.path.join(base_dir, "ubs")
        files   = glob.glob(os.path.join(ubs_dir, "frame_*_g*.npz"))

        def _load(fpath: str) -> bool:
            m = re.search(r"frame_(\d{5})_g(\d{2})\.npz$", os.path.basename(fpath))
            if not m:
                return False
            frame_idx = int(m.group(1))
            gi        = int(m.group(2))
            iy, ix    = self.map_index(frame_idx)
            if gi >= self.n_grains or iy >= self.ny or ix >= self.nx:
                return False
            try:
                d = np.load(fpath)
                self.U[gi, iy, ix]          = d["U"]
                self.rms_px[gi, iy, ix]     = float(d["rms_px"])
                self.mean_px[gi, iy, ix]    = float(d["mean_px"]) if "mean_px" in d else np.nan
                self.n_matched[gi, iy, ix]  = int(d["n_matched"])
                self.match_rate[gi, iy, ix] = float(d["match_rate"])
                self.cost[gi, iy, ix]       = float(d["cost"])
                return True
            except Exception as exc:
                print(f"  ✗  {fpath}: {exc}", flush=True)
                return False

        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
            n_loaded = sum(pool.map(_load, files))
        print(f"collect_orientation: {n_loaded} results loaded from {ubs_dir}", flush=True)
        return n_loaded

    def collect_strain(self, base_dir: str, n_workers: int | None = None) -> int:
        """
        Load strain npz files produced by SLURM workers into the map arrays.

        Files are loaded in parallel using a thread pool.

        Args:
            base_dir (str): Root processing directory.
            n_workers (int or None): Number of threads.  Defaults to
                ``os.cpu_count()``.

        Returns:
            int: Number of results loaded.
        """
        strain_dir = os.path.join(base_dir, "strain")
        files      = glob.glob(os.path.join(strain_dir, "frame_*_g*.npz"))

        def _load(fpath: str) -> bool:
            m = re.search(r"frame_(\d{5})_g(\d{2})\.npz$", os.path.basename(fpath))
            if not m:
                return False
            frame_idx = int(m.group(1))
            gi        = int(m.group(2))
            iy, ix    = self.map_index(frame_idx)
            if gi >= self.n_grains or iy >= self.ny or ix >= self.nx:
                return False
            try:
                d = np.load(fpath)
                self.U[gi, iy, ix]             = d["U"]
                self.rms_px[gi, iy, ix]        = float(d["rms_px"])
                self.mean_px[gi, iy, ix]       = float(d["mean_px"]) if "mean_px" in d else np.nan
                self.n_matched[gi, iy, ix]     = int(d["n_matched"])
                self.match_rate[gi, iy, ix]    = float(d["match_rate"])
                self.cost[gi, iy, ix]          = float(d["cost"])
                self.strain_voigt[gi, iy, ix]  = d["strain_voigt"]
                eps = d["strain_tensor"]
                self.strain_tensor[gi, iy, ix] = eps
                self.strain_tensor_deviatoric[gi, iy, ix] = eps - np.trace(eps) / 3.0 * np.eye(3)
                return True
            except Exception as exc:
                print(f"  ✗  {fpath}: {exc}", flush=True)
                return False

        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
            n_loaded = sum(pool.map(_load, files))
        print(f"collect_strain: {n_loaded} results loaded from {strain_dir}", flush=True)
        return n_loaded

    def write_merge_links(
        self,
        base_dir: str,
        best_grain: np.ndarray,
        metrics: dict | None = None,
        *,
        grain_index: int | None = None,
        source: str | None = None,
        overwrite: bool = False,
    ) -> tuple[str, int]:
        """
        Persist a merge selection as a folder of symlinks.

        Creates `base_dir/merged/` and, for every map position where
        *best_grain* is non-negative, places a symlink
        `frame_{frame_idx:05d}_g{gi_merged:02d}.npz` pointing to the
        corresponding source file in *source* (`ubs/` or `strain/`).

        The folder can later be fed to :meth:`collect_merged` to restore the
        merged grain slot into a freshly-loaded :class:`GrainMap`.

        Args:
            base_dir (str): Root of the processing directory tree (same root passed to
                `submit_orientation` / `collect_orientation`).
            best_grain ((ny, nx) int ndarray): First return value of :meth:`merge`.  Positions with value `-1`
                (no valid fit) are skipped.
            metrics (dict or None): Second return value of :meth:`merge`.  When provided, *source* is
                inherited from `metrics["source"]` unless explicitly overridden.
            grain_index (int or None): Grain slot index embedded in the symlink filename.  `None` uses
                `self.n_grains - 1` (i.e., the slot created by the most recent
                :meth:`apply_merge` call).
            source (str or None): Which result directory to link from.  One of:

            * `"ubs"`    — orientation `.npz` files in `base_dir/ubs/`.
            * `"strain"` — strain `.npz` files in `base_dir/strain/`.
            * `"auto"`   — prefers `strain/` when it contains matching
              files, otherwise falls back to `ubs/`.
            overwrite (bool): If `True`, existing entries in `merged/` are removed before
                writing.  Default `False`.

        Returns:
            merged_dir (str): Absolute path of the created `merged/` directory.
            n_links (int): Number of symlinks written.
"""
        gi_merged = self.n_grains - 1 if grain_index is None else int(grain_index)
        if source is None:
            source = (metrics or {}).get("source", "auto")

        ubs_dir    = os.path.join(base_dir, "ubs")
        strain_dir = os.path.join(base_dir, "strain")

        if source == "auto":
            src_dir  = strain_dir if glob.glob(os.path.join(strain_dir, "frame_*_g*.npz")) else ubs_dir
            src_label = "strain" if src_dir == strain_dir else "ubs"
        elif source == "ubs":
            src_dir, src_label = ubs_dir, "ubs"
        elif source == "strain":
            src_dir, src_label = strain_dir, "strain"
        else:
            raise ValueError(f"source must be 'ubs', 'strain', or 'auto', got {source!r}")

        merged_dir = os.path.join(base_dir, "merged", src_label)
        os.makedirs(merged_dir, exist_ok=True)

        n_links = 0
        n_missing = 0
        for iy in range(self.ny):
            for ix in range(self.nx):
                gi = int(best_grain[iy, ix])
                if gi < 0:
                    continue

                frame_idx = self.frame_index(iy, ix)
                src_file  = os.path.join(src_dir,    f"frame_{frame_idx:05d}_g{gi:02d}.npz")
                dst_file  = os.path.join(merged_dir, f"frame_{frame_idx:05d}_g{gi_merged:02d}.npz")

                if not os.path.exists(src_file):
                    n_missing += 1
                    continue

                if os.path.lexists(dst_file):
                    if overwrite:
                        os.remove(dst_file)
                    else:
                        continue

                rel_src = os.path.relpath(src_file, merged_dir)
                os.symlink(rel_src, dst_file)
                n_links += 1

        if n_missing:
            print(f"write_merge_links: {n_missing} source files not found (skipped)", flush=True)
        print(f"write_merge_links: {n_links} symlinks → {merged_dir}  [source={src_label}]", flush=True)
        return merged_dir, n_links

    def collect_merged(
        self,
        base_dir: str,
        grain_index: int | None = None,
        *,
        source: str = "ubs",
        n_workers: int | None = None,
    ) -> int:
        """
        Load results from `base_dir/merged/{source}/` into a grain slot.

        This is the complement of :meth:`write_merge_links`: it reads the
        `.npz` symlinks produced by that method and populates the chosen
        grain slot.  Strain fields (`strain_voigt`, `strain_tensor`) are
        loaded automatically when present.

        Args:
            base_dir (str): Same root directory passed to :meth:`write_merge_links`.
            grain_index (int or None): Grain slot to populate.  `None` uses `self.n_grains - 1`.
            source ({"ubs", "strain"}): Which merged subfolder to read from.  `"ubs"` loads orientation-
                only results; `"strain"` loads results that include strain tensors.
                Must match the `source` used when calling :meth:`write_merge_links`.
            n_workers (int or None): Number of threads.  Defaults to
                ``os.cpu_count()``.

        Returns:
            int: Number of results loaded.
"""
        if source not in ("ubs", "strain"):
            raise ValueError(f"source must be 'ubs' or 'strain', got {source!r}")
        merged_dir = os.path.join(base_dir, "merged", source)
        if not os.path.isdir(merged_dir):
            raise FileNotFoundError(
                f"merged/{source}/ directory not found: {merged_dir!r}. "
                f"Call write_merge_links(source={source!r}) first."
            )

        gi_target = self.n_grains - 1 if grain_index is None else int(grain_index)
        files     = glob.glob(os.path.join(merged_dir, "frame_*_g*.npz"))

        def _load(fpath: str) -> bool:
            m = re.search(r"frame_(\d{5})_g(\d{2})\.npz$", os.path.basename(fpath))
            if not m:
                return False
            frame_idx = int(m.group(1))
            iy, ix    = self.map_index(frame_idx)
            if iy >= self.ny or ix >= self.nx:
                return False
            try:
                d = np.load(fpath, allow_pickle=False)
                self.U[gi_target, iy, ix]          = d["U"]
                self.rms_px[gi_target, iy, ix]     = float(d["rms_px"])
                self.mean_px[gi_target, iy, ix]    = float(d["mean_px"]) if "mean_px" in d else np.nan
                self.n_matched[gi_target, iy, ix]  = int(d["n_matched"])
                self.match_rate[gi_target, iy, ix] = float(d["match_rate"])
                self.cost[gi_target, iy, ix]       = float(d["cost"])
                if "strain_voigt" in d:
                    self.strain_voigt[gi_target, iy, ix]  = d["strain_voigt"]
                    self.strain_tensor[gi_target, iy, ix] = d["strain_tensor"]
                return True
            except Exception as exc:
                print(f"  ✗  {fpath}: {exc}", flush=True)
                return False

        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
            n_loaded = sum(pool.map(_load, files))
        print(f"collect_merged [{source}]: {n_loaded} results loaded from {merged_dir}", flush=True)
        return n_loaded

    # ── Dunder ────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        fitted = int(np.sum(self.n_matched >= 0)) if self.n_grains else 0
        return (
            f"GrainMap(ny={self.ny}, nx={self.nx}, "
            f"n_grains={self.n_grains}, "
            f"fitted_points={fitted}/{self.ny * self.nx * max(self.n_grains, 1)}, "
            f"h5={os.path.basename(self.h5_path) if self.h5_path else 'None'})"
        )
