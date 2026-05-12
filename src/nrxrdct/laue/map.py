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
import pickle
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
        "rms_px", "match_rate", "cost", "n_matched", "misorientation",
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

        # ── axis extent ───────────────────────────────────────────────────────
        mx = self.motors.get(motor_x) if motor_x else None
        my = self.motors.get(motor_y) if motor_y else None

        if mx is not None and my is not None:
            extent = [
                mx[0, 0], mx[0, -1],
                my[-1, 0], my[0, 0],
            ]
            xlabel = motor_x
            ylabel = motor_y
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

    def plot_ipf(
        self,
        grain: int = 0,
        *,
        figsize: tuple = (6, 5),
        s: float = 15,
    ) -> tuple:
        """
        Scatter plot of the crystal Z-axis in the lab frame (crude IPF stand-in).

        Returns fig, ax.
        """
        fig, ax = plt.subplots(figsize=figsize)
        for iy in range(self.ny):
            for ix in range(self.nx):
                U = self.U[grain, iy, ix]
                if not np.any(np.isnan(U)):
                    z_lab = U[:, 2]
                    ax.scatter(z_lab[1], z_lab[2], s=s, c=[[abs(z_lab)]], vmin=0, vmax=1)
        ax.set_aspect("equal")
        ax.set_xlabel("lab Y")
        ax.set_ylabel("lab Z")
        ax.set_title(f"Grain {grain + 1}  —  crystal c-axis (lab frame)")
        fig.tight_layout()
        return fig, ax

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
            obj.n_matched     = np.full((n_grains, *shape2d), -1, dtype=int)
            obj.match_rate    = np.full((n_grains, *shape2d), np.nan)
            obj.cost          = np.full((n_grains, *shape2d), np.nan)
            obj.strain_voigt  = np.full((n_grains, *shape2d, 6), np.nan)
            obj.strain_tensor = np.full((n_grains, *shape2d, 3, 3), np.nan)

            for gi in range(n_grains):
                grp = f[f"grain_{gi:02d}"]
                obj.U[gi]          = grp["U"][()]
                obj.rms_px[gi]     = grp["rms_px"][()]
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
            "pixelsize":    float(camera.pixelsize),
            "n_pix_h":      int(camera.n_pix_h),
            "n_pix_v":      int(camera.n_pix_v),
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
        conda_env: str | None = None,
        extra_sbatch: dict | None = None,
    ) -> list:
        """Submit one SLURM job per chunk. Returns list of job IDs."""
        python_cmd = f"conda run -n {conda_env} python" if conda_env else "python"
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
            job_id = result.stdout.strip().split()[-1]
            job_ids.append(job_id)
            print(f"  Submitted {job_name} chunk {i:04d} → job {job_id}", flush=True)
        return job_ids

    def submit_segmentation(
        self,
        base_dir: str,
        h5_dataset: str,
        n_jobs: int = 10,
        *,
        partition: str = "all",
        time: str = "01:00:00",
        mem: str = "4G",
        cpus_per_task: int = 1,
        conda_env: str | None = None,
        mask_path: str | None = None,
        method: str = "LoG",
        method_kwargs: dict | None = None,
        min_size: int = 3,
        max_size: int = 500,
        gap_exclude: int = 3,
        bg_sigma: float = 251,
        extra_sbatch: dict | None = None,
    ) -> list:
        """
        Submit segmentation jobs to SLURM.

        Parameters
        ----------
        base_dir : str
            Root processing directory.  Subdirs are created automatically.
        h5_dataset : str
            HDF5 dataset path inside *h5_path* that holds the image stack.
        n_jobs : int
            Number of SLURM jobs (frames split evenly).
        method : str
            ``'LoG'`` (Laplacian of Gaussian) or ``'WTH'`` (white top-hat).
        method_kwargs : dict or None
            Extra kwargs forwarded to the chosen segmentation function.
        bg_sigma : float
            Gaussian sigma (pixels) for background estimation.  A large-scale
            Gaussian is fitted to the frame, subtracted, and the result is
            shifted to be entirely non-negative before segmentation.
        """
        dirs = self.setup_processing_dirs(base_dir)
        all_frames = list(range(self.ny * self.nx))
        chunks = [
            list(map(int, c))
            for c in np.array_split(all_frames, min(n_jobs, len(all_frames)))
            if len(c) > 0
        ]
        meta = {
            "h5_path":       self.h5_path,
            "h5_dataset":    h5_dataset,
            "seg_dir":       dirs["seg"],
            "mask_path":     mask_path,
            "method":        method,
            "method_kwargs": method_kwargs or {},
            "min_size":      min_size,
            "max_size":      max_size,
            "gap_exclude":   gap_exclude,
            "bg_sigma":      bg_sigma,
        }
        meta_path = os.path.join(dirs["job_meta"], "seg_meta.json")
        with open(meta_path, "w") as fh:
            json.dump(meta, fh, indent=2)

        job_ids = self._submit_jobs(
            "seg", "nrxrdct.laue.slurm_seg_worker", meta_path, chunks,
            dirs["slurm_logs"],
            partition=partition, time=time, mem=mem, cpus_per_task=cpus_per_task,
            conda_env=conda_env, extra_sbatch=extra_sbatch,
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
        conda_env: str | None = None,
        max_match_px=30.0,
        min_matched: int = 5,
        min_match_rate: float = 0.2,
        max_rms_px: float | None = None,
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
        """
        dirs = self.setup_processing_dirs(base_dir)

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
            "min_matched":    min_matched,
            "min_match_rate": min_match_rate,
            "max_rms_px":     max_rms_px,
            "method":         method,
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
            conda_env=conda_env, extra_sbatch=extra_sbatch,
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
        conda_env: str | None = None,
        max_match_px=10.0,
        fit_strain: list | None = None,
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
        """
        dirs = self.setup_processing_dirs(base_dir)

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
            "fit_strain": fit_strain or
                          ["e_xx", "e_yy", "e_zz", "e_xy", "e_xz", "e_yz"],
            "method":     method,
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
            conda_env=conda_env, extra_sbatch=extra_sbatch,
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
