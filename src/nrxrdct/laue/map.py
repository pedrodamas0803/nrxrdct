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

    Supported commands
    ------------------
    ``dmesh`` / ``mesh``
        ``dmesh motor1 start1 stop1 n1 motor2 start2 stop2 n2 [exposure]``
        → ``ny = n1+1``,  ``nx = n2+1``
    ``ascan``
        ``ascan motor start stop n [exposure]``
        → ``ny = 1``,  ``nx = n+1``
    ``loopscan``
        ``loopscan n [exposure]``
        → ``ny = 1``,  ``nx = n``

    Returns
    -------
    dict with keys: ``cmd``, ``ny``, ``nx``, ``n_frames``,
    and optionally ``motor1``, ``motor2``, ``start1/2``, ``stop1/2``, ``n1/2``.

    Raises
    ------
    ValueError
        If the command is not recognised.
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

    Parameters
    ----------
    ny, nx : int
        Number of map rows (slow motor) and columns (fast motor).
    h5_path : str
        Path to the master HDF5 scan file.  Used to read motor positions.
        May be ``None`` if you don't need motor coordinates.
    processing_dir : str or None
        Directory scanned for ``UB[0-9]*.npy`` grain reference matrices.
        Defaults to the directory containing *h5_path*, or CWD if both are
        absent.
    entry : str
        HDF5 entry key, e.g. ``"1.1"``.

    Attributes
    ----------
    ny, nx : int
    n_grains : int          Number of UB files found.
    U_ref : (n_grains, 3, 3) ndarray
        Reference orientation matrices loaded from ``UB*.npy``.
    U : (n_grains, ny, nx, 3, 3) ndarray
        Fitted orientation matrices.  ``NaN`` where not yet fitted.
    rms_px : (n_grains, ny, nx) ndarray
    mean_px : (n_grains, ny, nx) ndarray
    n_matched : (n_grains, ny, nx) int ndarray   (-1 = not fitted)
    match_rate : (n_grains, ny, nx) ndarray
    cost : (n_grains, ny, nx) ndarray
    motors : dict[str, (ny, nx) ndarray]
        Motor positions reshaped to the map grid (if h5_path is given and
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
    ):
        self.ny = int(ny)
        self.nx = int(nx)
        self.h5_path = h5_path
        self.entry = entry

        if processing_dir is None:
            if h5_path is not None:
                processing_dir = os.path.dirname(os.path.abspath(h5_path))
            else:
                processing_dir = os.getcwd()
        self.processing_dir = processing_dir

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
            self.strain_voigt  = np.concatenate([self.strain_voigt,
                np.full((extra, *shape2d, 6), np.nan)], axis=0)
            self.strain_tensor = np.concatenate([self.strain_tensor,
                np.full((extra, *shape2d, 3, 3), np.nan)], axis=0)
            for _ in range(extra):
                self._results.append(
                    [[None] * self.nx for _ in range(self.ny)]
                )

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
        self.strain_voigt  = np.full((ng, *shape2d, 6), np.nan)
        self.strain_tensor = np.full((ng, *shape2d, 3, 3), np.nan)
        self._results: list[list[list]] = [
            [[None] * self.nx for _ in range(self.ny)]
            for _ in range(ng)
        ]

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

    # ── Index helpers ─────────────────────────────────────────────────────────

    def frame_index(self, iy: int, ix: int) -> int:
        """Flat frame index from ``(row, col)`` — matches h5 frame order."""
        return iy * self.nx + ix

    def map_index(self, frame_idx: int) -> tuple[int, int]:
        """``(row, col)`` from a flat frame index."""
        return divmod(frame_idx, self.nx)

    # ── Result storage / retrieval ────────────────────────────────────────────

    def set_result(self, iy: int, ix: int, grain: int, result) -> None:
        """
        Store a fit result at map position ``(iy, ix)`` for *grain*.

        *result* can be an :class:`~nrxrdct.laue.fitting.OrientationFitResult`,
        :class:`~nrxrdct.laue.fitting.StrainFitResult`, or ``None`` (marks the
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
                self.strain_tensor[grain, iy, ix] = result.strain_tensor

    def get_result(self, iy: int, ix: int, grain: int):
        """Return the stored fit result (or ``None``) at ``(iy, ix, grain)``."""
        return self._results[grain][iy][ix]

    # ── Derived quantities ────────────────────────────────────────────────────

    def euler_map(
        self,
        grain: int,
        convention: str = "ZXZ",
    ) -> np.ndarray:
        """
        Euler angles for every map point.

        Returns
        -------
        angles : (ny, nx, 3) ndarray, degrees.  ``NaN`` where no fit exists.
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

        Parameters
        ----------
        reference : (3, 3) ndarray or None
            Reference orientation.  Defaults to ``U_ref[grain]`` if available,
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

        Parameters
        ----------
        quantity : str
            One of ``'match_rate'``, ``'rms_px'``, ``'cost'``,
            ``'n_matched'``, ``'misorientation'``,
            ``'euler_phi1'``, ``'euler_Phi'``, ``'euler_phi2'``.
        grain : int
            Grain index (0-based).
        motor_x, motor_y : str or None
            Motor names to use as axis tick labels (from ``self.motors``).
            If ``None``, integer pixel indices are shown.
        """
        # ── build data array ──────────────────────────────────────────────────
        if quantity == "match_rate":
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
        is created automatically.  Pass ``ax`` to place a single-grain plot on
        an existing axes.

        Parameters
        ----------
        grains : list[int] or None
            Grain indices to plot.  ``None`` plots all grains.
        ax : Axes or None
            If provided, only the first (or only) grain is plotted here.
        cmap : str or None
            Colormap.  Defaults to ``'plasma_r'``.
        vmin, vmax : float or None
            Color scale limits.  If ``share_scale`` is ``True`` and both are
            ``None``, the limits are computed jointly from all shown grains.
        motor_x, motor_y : str or None
            Motor names for axis labels.
        motor_units : dict or None
            Units for motor axes, e.g. ``{'pz': 'mm', 'py': 'mm'}``.
        share_scale : bool
            If ``True`` (default), all subplots share the same ``vmin``/``vmax``.
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

        Parameters
        ----------
        component : str
            One of ``'e_xx'``, ``'e_yy'``, ``'e_zz'``,
            ``'e_xy'``, ``'e_xz'``, ``'e_yz'``.
        grain : int
            Grain index (0-based).
        frame : str
            Reference frame for the strain tensor:

            ``'crystal'``
                As fitted — components in the crystal coordinate system.
            ``'lab'``
                Rotated to the lab frame via ``ε_lab = U @ ε @ Uᵀ``.
            ``'sample'``
                Lab frame further rotated by *sample_tilt_deg* about
                *sample_tilt_axis* (default −40° about Y).

        sample_tilt_deg : float
            Tilt angle (degrees) from lab to sample frame.  Default ``-40``.
        sample_tilt_axis : str
            Lab axis of the tilt rotation (``'x'``, ``'y'``, or ``'z'``).
            Default ``'y'``.
        motor_x, motor_y : str or None
            Motor names to use as axis tick labels.
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

    # ── IPF map ───────────────────────────────────────────────────────────────

    @staticmethod
    def _cubic_ipf_colors(c: np.ndarray) -> np.ndarray:
        """
        Vectorised IPF RGB for cubic (m-3m) symmetry.

        Colour convention: [001] → blue, [011] → green, [111] → red.

        Parameters
        ----------
        c : (…, 3) array
            Crystal-frame directions.  Need not be unit vectors; NaN rows
            produce NaN RGB output.

        Returns
        -------
        rgb : same leading shape + (3,), float in [0, 1].
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
        Render the cubic IPF color key as an (N, N, 3) float32 RGB image.

        The standard triangle [001]–[011]–[111] is filled; pixels outside
        are white.  In the returned image rows correspond to increasing *p*
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
        p_max = np.interp(t_vals, t_b, p_b)   # shape (N,)
        inside = P <= p_max[np.newaxis, :]     # broadcast over rows

        r_c = P
        g_c = T * (1.0 - P)
        b_c = (1.0 - T) * (1.0 - P)
        mx  = np.maximum(np.maximum(r_c, g_c), b_c)
        mx  = np.maximum(mx, 1e-10)

        rgb         = np.stack([r_c / mx, g_c / mx, b_c / mx], axis=-1).astype(np.float32)
        rgb[~inside] = 1.0
        return rgb

    @staticmethod
    def _ipf_colorkey_inset(parent_ax) -> None:
        """Add the cubic IPF color-key as a small inset in *parent_ax*."""
        try:
            from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        except ImportError:
            return

        ax_key = inset_axes(parent_ax, width="28%", height="28%",
                            loc="lower right", borderpad=0.5)

        rgb = GrainMap._cubic_ipf_colorkey(200)
        ax_key.imshow(rgb, origin="lower", extent=[0, 1, 0, 1],
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

        ax_key.set_xlim(0, 1); ax_key.set_ylim(0, 1)
        ax_key.set_xticks([]); ax_key.set_yticks([])
        for sp in ax_key.spines.values():
            sp.set_visible(False)

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
    ) -> tuple:
        """
        Inverse pole figure (IPF) map coloured by which crystal direction is
        parallel to a chosen reference axis.

        Parameters
        ----------
        axis : str or (3,) array-like
            Reference direction in the chosen *frame*.
            Shortcuts: ``'x'``, ``'y'``, ``'z'``; or a custom 3-vector.
        grain : int
            Grain index (0-based).
        frame : str
            ``'lab'``    — *axis* is in the lab frame.
            ``'sample'`` — *axis* is in the sample frame, converted to lab
            via the inverse of the sample tilt (see *sample_tilt_deg*).
        sample_tilt_deg : float
            Rotation angle (°) about *sample_tilt_axis* that maps lab → sample.
            Default ``-40``.
        sample_tilt_axis : str
            Lab axis of the tilt rotation.  Default ``'y'``.
        symmetry : str
            Crystal symmetry for IPF reduction.  Currently only ``'cubic'``.
        motor_x, motor_y : str or None
            Motor names to use as axis tick labels.
        motor_units : dict or None
            Optional units for motor axes, e.g. ``{'pz': 'mm', 'py': 'mm'}``.
            Appended to the axis label in parentheses.
        show_colorkey : bool
            Overlay a small colour-key triangle in the lower-right corner.
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
            self._ipf_colorkey_inset(ax)

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

        Parameters
        ----------
        grain : int
            Grain index (0-based).
        frame : str
            ``'lab'``    — directions expressed in the lab frame.
            ``'sample'`` — directions expressed in the sample frame (rotated
            from lab by *sample_tilt_deg* about *sample_tilt_axis*).
        sample_tilt_deg : float
            Lab-to-sample rotation angle (°).  Default ``-40``.
        sample_tilt_axis : str
            Axis of the lab-to-sample rotation.  Default ``'y'``.
        symmetry : str
            IPF colour symmetry.  Currently only ``'cubic'``.
        s, alpha : float
            Scatter marker size and transparency.
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

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """
        Serialise the GrainMap to an HDF5 file.

        All numeric arrays are stored under ``/grain_{i:02d}/`` groups.
        Metadata (ny, nx, ub_files, h5_path, entry) go into ``/meta``.
        """
        with h5py.File(path, "w") as f:
            meta = f.create_group("meta")
            meta.attrs["ny"]           = self.ny
            meta.attrs["nx"]           = self.nx
            meta.attrs["n_grains"]     = self.n_grains
            meta.attrs["h5_path"]      = self.h5_path or ""
            meta.attrs["entry"]        = self.entry
            meta.attrs["processing_dir"] = self.processing_dir
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
                grp.create_dataset("strain_voigt",  data=self.strain_voigt[gi],  compression="gzip")
                grp.create_dataset("strain_tensor", data=self.strain_tensor[gi], compression="gzip")

        print(f"GrainMap saved → {os.path.abspath(path)}")

    @classmethod
    def load(cls, path: str) -> "GrainMap":
        """
        Restore a GrainMap from a file previously written by :meth:`save`.

        UB reference matrices and motor positions are re-read from the file;
        the ``_results`` list (which holds full Python objects) is not
        persisted and will be all-``None`` after loading.
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
            obj.strain_voigt  = np.full((n_grains, *shape2d, 6), np.nan)
            obj.strain_tensor = np.full((n_grains, *shape2d, 3, 3), np.nan)

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

            obj.motors = {}
            if "motors" in f:
                for motor in f["motors"].keys():
                    obj.motors[motor] = f[f"motors/{motor}"][()]

            obj._results = [
                [[None] * nx for _ in range(ny)]
                for _ in range(n_grains)
            ]

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

        Parameters
        ----------
        job_ids : list[str | int]
            Job IDs returned by :meth:`submit_segmentation`,
            :meth:`submit_orientation`, or :meth:`submit_strain`.
        dry_run : bool
            If ``True``, print the ``scancel`` command without executing it.
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
        spots file per frame under ``base_dir/seg/``.  The pipeline is:

        1. Estimate and subtract a Gaussian background (sigma ``bg_sigma``)
           from the raw frame — **used only for spot detection**.
        2. Detect spots with the chosen segmentation method.
        3. Clean the binary mask (size filter, border removal).
        4. Measure regionprops and fit a 2-D Gaussian to the **original**
           (unmodified) frame intensities inside a ``(2d)×(2d)`` ROI around
           each centroid.
        5. Write results to ``seg_dir/frame_{idx:05d}.h5``.

        Parameters
        ----------
        base_dir : str
            Root processing directory.  The sub-directories ``seg/``,
            ``ubs/``, ``strain/``, ``slurm_logs/``, and ``job_meta/`` are
            created automatically if they do not exist.
        h5_dataset : str or None
            HDF5 dataset path inside ``self.h5_path`` that holds the image
            stack, e.g. ``'1.1/measurement/det'``.  Mutually exclusive with
            *tiff_dir*; exactly one must be supplied.
        tiff_dir : str or None
            Path to a directory containing one TIFF file per frame, named
            ``img_<number>.tif`` (e.g. ``img_1500.tif``).  Files are sorted
            by their embedded number and mapped to 0-based frame indices in
            that order.  Motor positions are still read from ``self.h5_path``
            as usual.  Mutually exclusive with *h5_dataset*.
        n_jobs : int
            Number of SLURM array jobs.  Frames are split as evenly as
            possible across jobs.  Default ``10``.
        partition : str
            SLURM partition name.  Default ``'all'``.
        time : str
            Wall-clock time limit per job in ``HH:MM:SS`` format.
            Default ``'01:00:00'``.
        mem : str
            Memory per job, e.g. ``'4G'``, ``'16G'``.  Default ``'4G'``.
        cpus_per_task : int
            CPU cores requested per job.  Default ``1``.
        python_bin : str
            Python executable used in the ``--wrap`` command.  Default
            ``'python'``.
        mask_path : str or None
            Path to a ``.npy`` boolean array marking valid detector pixels
            (``True`` = active).  ``None`` treats the whole frame as valid.
        method : str
            Spot-detection algorithm:

            ``'LoG'``
                Laplacian-of-Gaussian blob detector.  Good for round,
                diffuse spots.
            ``'WTH'``
                White top-hat transform.  More robust on strong or uneven
                background.
            ``'HYBRID'``
                LoG and WTH responses combined with a logical OR; best
                when the pattern contains both large and small spots.

            Default ``'LoG'``.
        method_kwargs : dict or None
            Extra keyword arguments forwarded to the segmentation function.
            Useful keys by method:

            * ``'LoG'``: ``sigmas``, ``threshold_percentile``
            * ``'WTH'``: ``disk_radius``, ``threshold_percentile``
            * ``'HYBRID'``: ``log_sigmas``, ``wth_disk_radius``,
              ``threshold_percentile``
        min_size : int
            Minimum connected-component area in pixels; smaller blobs are
            discarded.  Default ``3``.
        max_size : int
            Maximum connected-component area in pixels; larger blobs are
            discarded.  Default ``500``.
        gap_exclude : int
            Width in pixels of the border region to clear before labelling
            (removes spots cut off by the detector edge).  Default ``3``.
        bg_sigma : float
            Gaussian sigma (pixels) for FFT-based background estimation.
            A large value (≥ several spot spacings) captures the slowly
            varying beam profile.  The subtracted frame is used only for
            segmentation; Gaussian fits are performed on the original
            intensities.  Default ``251``.
        max_components : int
            Maximum number of Gaussian components tried per spot during
            fitting.  ``1`` fits a single 2-D Gaussian; higher values
            attempt mixture models for overlapping spots.  Default ``1``.
        d : int
            Half-size in pixels of the square ROI cropped around each spot
            centroid for Gaussian fitting.  The crop window is
            ``(2d) × (2d)`` pixels.  Increase for large or diffuse spots;
            decrease to speed up fitting for small, sharp spots.
            Default ``10``.
        r_squared_min : float
            Minimum R² of the Gaussian fit for a spot to be accepted.
            Spots below this threshold are either skipped or stored without
            fit parameters (see *include_unfitted*).  This value is stored
            in ``seg_meta.json`` and used as the default for
            :meth:`submit_orientation` and :meth:`submit_strain` unless
            explicitly overridden there.  Default ``0.9``.
        include_unfitted : bool
            If ``True``, spots whose best Gaussian fit has R² < *r_squared_min*
            are still written to the HDF5 file using the raw weighted
            centroid as position (shape parameters set to zero).  If
            ``False``, those spots are silently discarded.  Stored in
            ``seg_meta.json`` and inherited by downstream workers.
            Default ``False``.
        extra_sbatch : dict or None
            Additional ``sbatch`` options passed as ``--key=value`` flags,
            e.g. ``{'account': 'myproject', 'constraint': 'gpu'}``.

        Returns
        -------
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
        hmax: int | None = None,
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
    ) -> list:
        """
        Submit orientation-fitting jobs to SLURM.

        Each UB reference matrix in *processing_dir* is tried independently;
        successful fits are written as ``frame_{idx:05d}_g{gi:02d}.npz``.

        Parameters
        ----------
        r_squared_min : float or None
            Minimum R² for spot inclusion.  ``None`` inherits the value stored
            in ``seg_meta.json`` by :meth:`submit_segmentation`; falls back to
            ``0.9`` if that file is absent.
        include_unfitted : bool or None
            Include spots whose Gaussian fit failed.  ``None`` inherits from
            ``seg_meta.json``; falls back to ``False``.
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
            "method":          method,
            "ftol":           ftol,
            "xtol":           xtol,
            "gtol":           gtol,
        }
        for key, val in [
            ("hmax", hmax), ("f2_thresh", f2_thresh),
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
        hmax: int | None = None,
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
    ) -> list:
        """
        Submit strain-fitting jobs to SLURM.

        Requires orientation results in ``base_dir/ubs/`` (run
        :meth:`submit_orientation` first).

        Parameters
        ----------
        r_squared_min : float or None
            Minimum R² for spot inclusion.  ``None`` inherits the value stored
            in ``seg_meta.json`` by :meth:`submit_segmentation`; falls back to
            ``0.9`` if that file is absent.
        include_unfitted : bool or None
            Include spots whose Gaussian fit failed.  ``None`` inherits from
            ``seg_meta.json``; falls back to ``False``.
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
            "method":          method,
            "ftol":       ftol,
            "xtol":       xtol,
            "gtol":       gtol,
        }
        for key, val in [
            ("hmax", hmax), ("f2_thresh", f2_thresh),
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

    def collect_orientation(self, base_dir: str) -> int:
        """
        Load orientation npz files produced by SLURM workers into the map arrays.

        Returns the number of results loaded.
        """
        ubs_dir = os.path.join(base_dir, "ubs")
        files = glob.glob(os.path.join(ubs_dir, "frame_*_g*.npz"))
        n_loaded = 0
        for fpath in files:
            m = re.search(r"frame_(\d{5})_g(\d{2})\.npz$", os.path.basename(fpath))
            if not m:
                continue
            frame_idx = int(m.group(1))
            gi        = int(m.group(2))
            iy, ix    = self.map_index(frame_idx)
            if gi >= self.n_grains or iy >= self.ny or ix >= self.nx:
                continue
            try:
                d = np.load(fpath)
                self.U[gi, iy, ix]          = d["U"]
                self.rms_px[gi, iy, ix]     = float(d["rms_px"])
                self.mean_px[gi, iy, ix]    = float(d["mean_px"]) if "mean_px" in d else np.nan
                self.n_matched[gi, iy, ix]  = int(d["n_matched"])
                self.match_rate[gi, iy, ix] = float(d["match_rate"])
                self.cost[gi, iy, ix]       = float(d["cost"])
                n_loaded += 1
            except Exception as exc:
                print(f"  ✗  {fpath}: {exc}", flush=True)
        print(f"collect_orientation: {n_loaded} results loaded from {ubs_dir}")
        return n_loaded

    def collect_strain(self, base_dir: str) -> int:
        """
        Load strain npz files produced by SLURM workers into the map arrays.

        Returns the number of results loaded.
        """
        strain_dir = os.path.join(base_dir, "strain")
        files = glob.glob(os.path.join(strain_dir, "frame_*_g*.npz"))
        n_loaded = 0
        for fpath in files:
            m = re.search(r"frame_(\d{5})_g(\d{2})\.npz$", os.path.basename(fpath))
            if not m:
                continue
            frame_idx = int(m.group(1))
            gi        = int(m.group(2))
            iy, ix    = self.map_index(frame_idx)
            if gi >= self.n_grains or iy >= self.ny or ix >= self.nx:
                continue
            try:
                d = np.load(fpath)
                self.U[gi, iy, ix]             = d["U"]
                self.rms_px[gi, iy, ix]        = float(d["rms_px"])
                self.mean_px[gi, iy, ix]       = float(d["mean_px"]) if "mean_px" in d else np.nan
                self.n_matched[gi, iy, ix]     = int(d["n_matched"])
                self.match_rate[gi, iy, ix]    = float(d["match_rate"])
                self.cost[gi, iy, ix]          = float(d["cost"])
                self.strain_voigt[gi, iy, ix]  = d["strain_voigt"]
                self.strain_tensor[gi, iy, ix] = d["strain_tensor"]
                n_loaded += 1
            except Exception as exc:
                print(f"  ✗  {fpath}: {exc}", flush=True)
        print(f"collect_strain: {n_loaded} results loaded from {strain_dir}")
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
