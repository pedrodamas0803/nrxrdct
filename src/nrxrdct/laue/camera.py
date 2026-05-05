from dataclasses import dataclass

import numpy as np

DD = 85.475  # dd    (mm)
XCEN = 1040.26  # xcen  (pixels)
YCEN = 1126.63  # ycen  (pixels)
XBET = 0.447  # xbet  (degrees)
XGAM = 0.333  # xgam  (degrees)
PIXEL_SIZE_MM = 0.0734  # xpixelsize = ypixelsize (mm)
N_PIX_H = 2018  # framedim[0]
N_PIX_V = 2016  # framedim[1]
KF_DIRECTION = "Z>0"  # kf_direction from calibration file
SPOT_SIGMA_PIX = 2


@dataclass
class CalibrationResult:
    """Returned by :meth:`Camera.fit_calibration`."""

    camera: "Camera"
    U: np.ndarray
    rms_px: float
    n_matched: int
    n_obs: int
    n_sim: int
    fit_params: tuple
    success: bool
    message: str

    def __repr__(self):
        return (
            f"CalibrationResult(rms={self.rms_px:.2f} px, "
            f"matched={self.n_matched}/{self.n_obs} obs / {self.n_sim} sim, "
            f"success={self.success})"
        )

    def plot(
        self,
        crystal,
        peaklist: np.ndarray,
        image: np.ndarray | None = None,
        max_match_px: float = 20.0,
        space: str = "detector",
        E_min_eV: float = 5_000.0,
        E_max_eV: float = 25_000.0,
        source: str = "bending_magnet",
        source_kwargs: dict | None = None,
        hmax: int = 15,
        f2_thresh: float = 0.0,
        top_n_sim: int | None = None,
        figsize: tuple = (14, 5),
    ):
        """
        Plot the calibration result: observed vs simulated spot positions.

        Parameters
        ----------
        crystal      : Crystal  — calibration standard used during fitting.
        peaklist     : (N, ≥2) array  — peak list from
                       :func:`~nrxrdct.laue.segmentation.convert_spotsfile2peaklist`.
                       Columns 0 and 1 are used as (xcam, ycam) pixel positions.
        image        : (Nv, Nh) array or None  — optional detector image
                       (``'detector'`` space only; ignored for ``'angular'``).
        max_match_px : float  — pixel-space match radius for spot pairing.
        space        : ``'detector'`` or ``'angular'``  — coordinate frame for
                       the main panel.  ``'angular'`` plots 2θ (x) vs χ (y) in
                       degrees.  Matching is always done in detector (pixel)
                       space; the histogram shows pixel distances in detector
                       mode and Euclidean (2θ, χ) distances in angular mode.
        f2_thresh    : float  — minimum |F|² to include a reflection (0 = all).
        top_n_sim    : int or None  — restrict to the N brightest simulated spots.
        figsize      : (w, h)  — figure size in inches.

        Returns
        -------
        fig, (ax_det, ax_info, ax_hist)
        """
        import matplotlib.colors as mcolors
        import matplotlib.gridspec as gridspec
        import matplotlib.pyplot as plt

        from .simulation import simulate_laue
        from .fitting import _match_spots

        if space not in ("detector", "angular"):
            raise ValueError(f"space must be 'detector' or 'angular', got {space!r}")

        _BG   = "#080c14"
        _BG2  = "#0d1220"
        _FG   = "#ccccee"
        _GRAY = "#4a5070"
        _OBS  = "#ffffff"
        _SIM  = "#ff6b35"

        cam    = self.camera
        obs_xy = np.asarray(peaklist, dtype=float)[:, :2]
        src_kw = source_kwargs or {}

        # ── simulate with the fitted camera + U ──────────────────────────────
        spots = simulate_laue(
            crystal, self.U, cam,
            E_min=E_min_eV, E_max=E_max_eV,
            source=source, source_kwargs=src_kw,
            hmax=hmax, f2_thresh=f2_thresh,
        )

        # Collect on-detector spots keeping pixel, angular, and intensity aligned
        _on_det = [
            (s["pix"], s.get("tth", np.nan), s.get("chi", np.nan), s.get("I_raw", 1.0))
            for s in spots if s.get("pix") is not None
        ]
        if top_n_sim is not None and len(_on_det) > top_n_sim:
            _on_det.sort(key=lambda t: -t[3])
            _on_det = _on_det[:top_n_sim]

        if _on_det:
            sim_xy  = np.array([t[0] for t in _on_det], dtype=float)
            tth_sim = np.array([t[1] for t in _on_det], dtype=float)
            chi_sim = np.array([t[2] for t in _on_det], dtype=float)
        else:
            sim_xy  = np.empty((0, 2), dtype=float)
            tth_sim = np.empty(0, dtype=float)
            chi_sim = np.empty(0, dtype=float)

        # ── match in detector (pixel) space ───────────────────────────────────
        row_ind, col_ind, dist_px = _match_spots(obs_xy, sim_xy, max_match_px)
        ok_mask   = dist_px < max_match_px
        n_matched = int(ok_mask.sum())
        rate      = n_matched / max(len(obs_xy), 1)

        # ── angular coords for observed spots ─────────────────────────────────
        uf_obs  = cam.pixel_to_kf(obs_xy[:, 0], obs_xy[:, 1])
        tth_obs = np.degrees(np.arccos(np.clip(uf_obs[:, 0], -1.0, 1.0)))
        chi_obs = np.degrees(np.arctan2(uf_obs[:, 1], uf_obs[:, 2] + 1e-17))

        # ── distances for histogram ───────────────────────────────────────────
        if space == "detector":
            dists_ok  = dist_px[ok_mask]
            dist_unit = "px"
        else:
            ang_dists = np.sqrt(
                (tth_obs[row_ind] - tth_sim[col_ind]) ** 2
                + (chi_obs[row_ind] - chi_sim[col_ind]) ** 2
            )
            dists_ok  = ang_dists[ok_mask]
            dist_unit = "°"

        mean_d = float(np.mean(dists_ok))               if n_matched > 0 else float("nan")
        rms_d  = float(np.sqrt(np.mean(dists_ok ** 2))) if n_matched > 0 else float("nan")
        mean_s = f"{mean_d:.3f} {dist_unit}" if np.isfinite(mean_d) else "—"
        rms_s  = f"{rms_d:.3f} {dist_unit}"  if np.isfinite(rms_d)  else "—"

        # ── figure layout ─────────────────────────────────────────────────────
        with plt.ioff():
            fig = plt.figure(figsize=figsize, facecolor=_BG)

        gs = gridspec.GridSpec(
            1, 2, figure=fig,
            left=0.06, right=0.98, bottom=0.08, top=0.93,
            wspace=0.06, width_ratios=[2.5, 1.0],
        )
        gs_r = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=gs[1], hspace=0.45, height_ratios=[1, 1],
        )
        ax_det  = fig.add_subplot(gs[0])
        ax_info = fig.add_subplot(gs_r[0])
        ax_hist = fig.add_subplot(gs_r[1])

        for ax in (ax_det, ax_info, ax_hist):
            ax.set_facecolor(_BG2)
            ax.tick_params(colors=_GRAY, labelsize=7)
            for sp in ax.spines.values():
                sp.set_edgecolor(_GRAY)

        # ── choose display coordinates ────────────────────────────────────────
        if space == "detector":
            obs_plot = obs_xy
            sim_plot = sim_xy
            ax_det.set_xlim(0, cam.Nh)
            ax_det.set_ylim(cam.Nv, 0)
            ax_det.set_aspect("equal")
            ax_det.set_xlabel("xcam  (px)", color=_FG, fontsize=8)
            ax_det.set_ylabel("ycam  (px)", color=_FG, fontsize=8)
            if image is not None:
                img = np.asarray(image, dtype=float)
                vmax = np.percentile(img[img > 0], 99) if img.max() > 0 else 1.0
                ax_det.imshow(
                    np.log1p(img / vmax * 1000),
                    origin="upper", extent=[0, cam.Nh, cam.Nv, 0],
                    cmap="inferno", aspect="auto", alpha=0.55, zorder=0,
                )
            else:
                ax_det.add_patch(plt.Rectangle(
                    (0, 0), cam.Nh, cam.Nv,
                    linewidth=0.8, edgecolor=_GRAY, facecolor="none", zorder=0,
                ))
        else:
            obs_plot = np.column_stack([tth_obs, chi_obs])
            sim_plot = (np.column_stack([tth_sim, chi_sim])
                        if len(tth_sim) else np.empty((0, 2)))
            ax_det.set_aspect("auto")
            ax_det.set_xlabel("2θ  (°)", color=_FG, fontsize=8)
            ax_det.set_ylabel("χ  (°)", color=_FG, fontsize=8)
            ax_det.grid(True, ls=":", lw=0.35, color="#181e2e", zorder=0)

        ax_det.set_title(
            f"Calibration result   ○ observed   ◆ fitted   — matched   "
            f"mean={mean_s}   rms={rms_s}",
            color=_FG, fontsize=9, pad=6,
        )
        ax_det.scatter(
            obs_plot[:, 0], obs_plot[:, 1],
            s=45, c="none", edgecolors=_OBS, linewidths=0.9,
            zorder=4, label=f"observed ({len(obs_xy)})",
        )
        if len(sim_plot):
            ax_det.scatter(
                sim_plot[:, 0], sim_plot[:, 1],
                s=28, c=_SIM, marker="D", linewidths=0,
                zorder=5, label=f"simulated ({len(sim_xy)})",
            )

        # Lines coloured green→red by pixel distance (calibration quality)
        cmap_match = plt.get_cmap("RdYlGn_r")
        cnorm = mcolors.Normalize(vmin=0, vmax=max_match_px)
        for r, c, d, ok in zip(row_ind, col_ind, dist_px, ok_mask):
            if ok:
                ax_det.plot(
                    [obs_plot[r, 0], sim_plot[c, 0]],
                    [obs_plot[r, 1], sim_plot[c, 1]],
                    color=cmap_match(cnorm(d)), lw=0.8, alpha=0.7, zorder=3,
                )
        ax_det.legend(loc="upper right", fontsize=7,
                      facecolor=_BG2, edgecolor=_GRAY, labelcolor=_FG)

        # ── info panel ────────────────────────────────────────────────────────
        ax_info.set_axis_off()
        status = "OK" if self.success else "FAILED"
        ax_info.text(
            0.06, 0.97,
            f"Fit  [{status}]\n"
            f"{'─' * 22}\n"
            f"  matched : {n_matched} / {len(obs_xy)}\n"
            f"  rate    : {rate:.0%}\n"
            f"  mean    : {mean_s}\n"
            f"  rms     : {rms_s}\n"
            f"\n"
            f"Camera\n"
            f"{'─' * 22}\n"
            f"  dd   ={cam.dd:9.4f} mm\n"
            f"  xcen ={cam.xcen:9.2f} px\n"
            f"  ycen ={cam.ycen:9.2f} px\n"
            f"  xbet ={cam.xbet:9.4f} °\n"
            f"  xgam ={cam.xgam:9.4f} °\n",
            transform=ax_info.transAxes,
            color=_FG, fontsize=8, va="top", family="monospace",
            linespacing=1.5,
        )

        # ── histogram panel ───────────────────────────────────────────────────
        ax_hist.set_xlabel(f"distance  ({dist_unit})", color=_FG, fontsize=8)
        ax_hist.set_ylabel("count", color=_FG, fontsize=8)
        ax_hist.set_title("Match distances", color=_FG, fontsize=9, pad=4)

        if n_matched > 0:
            bins = np.linspace(0, dists_ok.max() * 1.05, min(30, n_matched + 1))
            ax_hist.hist(dists_ok, bins=bins, color=_SIM, alpha=0.85,
                         edgecolor=_BG, linewidth=0.4)
            ax_hist.axvline(mean_d, color="#44dd66", lw=1.4, ls="--",
                            label=f"mean {mean_s}")
            ax_hist.axvline(rms_d, color="#ffcc44", lw=1.4, ls=":",
                            label=f"rms  {rms_s}")
            ax_hist.legend(fontsize=7, facecolor=_BG2, edgecolor=_GRAY,
                           labelcolor=_FG)

        plt.show()
        return fig, (ax_det, ax_info, ax_hist)


class Camera:
    """
    Pixelated area detector fully compatible with LaueTools calibration files.

    Public frame convention (LT frame)
    -----------------------------------
    All public methods (kf_to_pixel, pixel_to_kf, project) use the canonical
    LaueTools LT frame:

        x  : along incident beam  (ki direction)
        z  : vertical up
        y  : horizontal (= z ^ x, towards the wall)

    Internally the camera geometry is computed in the LT2 frame (y // beam),
    which is what LaueGeometry.py uses.  The LT ↔ LT2 conversion is handled
    transparently inside each method:

        LT → LT2 :  x_LT2 = −y_LT,   y_LT2 =  x_LT,  z_LT2 = z_LT
        LT2 → LT :  x_LT  =  y_LT2,  y_LT  = −x_LT2, z_LT  = z_LT2

    Calibration parameters  (CCDCalibParameters = [dd, xcen, ycen, xbet, xgam])
    ---------------------------------------------------------------------------
    dd   : distance sample → detector reference point O  (mm)
    xcen : pixel X of point O  (normal-incidence / beam-footprint pixel)
    ycen : pixel Y of point O
    xbet : angle (°) between the vector IO and the vertical z axis
             xbet ≈ 0  →  camera directly above sample  (Z>0 geometry, 2θ ~ 90°)
             xbet ≈ 90 →  transmission forward camera
    xgam : in-plane rotation (°) of the CCD pixel axes around the IO direction

    kf_direction : geometry label used by LaueTools
        'Z>0'  top/side reflection  (most common, xbet small)
        'X>0'  transmission (forward)
        'X<0'  back-reflection

    Pixel convention  (identical to LaueTools)
    ------------------------------------------
    (xcam=0, ycam=0) : top-left corner of the array
    xcam increases to the right (columns)
    ycam increases downward    (rows)
    (xcen, ycen) : sub-pixel reference point where the detector normal
                   intersects the pixel array.

    The IO vector and detector normal  (in LT2 frame)
    --------------------------------------------------
    For Z>0 geometry:
        beta  = pi/2 - xbet * pi/180
        IO    = dd * [0,  cos(beta),  sin(beta)]
              = dd * [0,  sin(xbet),  cos(xbet)]
        normal = IO / |IO|

    The two key public functions:
        kf_to_pixel  : uflab (N×3, LT)  →  (xcam, ycam)   [LaueTools: calc_xycam]
        pixel_to_kf  : (xcam, ycam)     →  uflab (N×3, LT) [LaueTools: calc_uflab]
    """

    def __init__(
        self,
        dd=DD,
        xcen=XCEN,
        ycen=YCEN,
        xbet=XBET,
        xgam=XGAM,
        pixelsize=PIXEL_SIZE_MM,
        n_pix_h=N_PIX_H,
        n_pix_v=N_PIX_V,
        kf_direction=KF_DIRECTION,
    ):

        self.dd = float(dd)
        self.xcen = float(xcen)
        self.ycen = float(ycen)
        self.xbet = float(xbet)
        self.xgam = float(xgam)
        self.pixel_mm = float(pixelsize)
        self.Nh = int(n_pix_h)
        self.Nv = int(n_pix_v)
        self.kf_direction = kf_direction

        self._build_geometry()

    # ── internal geometry ──────────────────────────────────────────────────────

    def _build_geometry(self):
        DEG = np.pi / 180.0
        xbet = self.xbet
        xgam = self.xgam
        dd = self.dd

        if self.kf_direction in ("Z>0", "Y>0", "Y<0"):
            # Top / side reflection geometry (default)
            # beta = pi/2 - xbet*DEG  (angle between IO and y axis)
            self._cosbeta = np.cos(np.pi / 2 - xbet * DEG)  # = sin(xbet)
            self._sinbeta = np.sin(np.pi / 2 - xbet * DEG)  # = cos(xbet)
            # IO vector: points from sample I to detector reference point O
            self.IOlab = dd * np.array([0.0, self._cosbeta, self._sinbeta])

        elif self.kf_direction == "X>0":
            # Transmission geometry
            self._cosbeta = np.cos(-xbet * DEG)
            self._sinbeta = np.sin(-xbet * DEG)
            self.IOlab = dd * np.array([0.0, self._cosbeta, self._sinbeta])

        elif self.kf_direction == "X<0":
            # Back-reflection geometry
            self._cosbeta = np.cos(-xbet * DEG)
            self._sinbeta = np.sin(-xbet * DEG)
            self.IOlab = dd * np.array([0.0, -self._cosbeta, self._sinbeta])

        else:
            raise ValueError(f"Unknown kf_direction: {self.kf_direction!r}")

        # Detector unit normal
        self.normal = self.IOlab / np.linalg.norm(self.IOlab)
        # Precompute gam rotation coefficients (used in both directions)
        self._cosgam = np.cos(-xgam * DEG)
        self._singam = np.sin(-xgam * DEG)
        # Physical size
        self.size_h_mm = self.Nh * self.pixel_mm
        self.size_v_mm = self.Nv * self.pixel_mm

    # ── forward projection: kf unit vector → pixel ────────────────────────────

    def kf_to_pixel(self, uflab_arr):
        """
        Map scattered unit vectors to pixel coordinates.
        Implements LaueTools calc_xycam() for Z>0 geometry.

        Parameters
        ----------
        uflab_arr : (N, 3) array of unit scattered vectors in LT frame (x // beam)

        Returns
        -------
        xcam, ycam : (N,) arrays of pixel coordinates (float, sub-pixel precision)
                     Returns NaN for beams that miss the detector or go backward.
        """
        # Convert LT -> LT2:  x_LT2 = -y_LT,  y_LT2 = x_LT,  z_LT2 = z_LT
        uf_lt = np.atleast_2d(np.array(uflab_arr, dtype=float))
        norms = np.linalg.norm(uf_lt, axis=1, keepdims=True)
        uf_lt = uf_lt / norms
        uf = np.column_stack([-uf_lt[:, 1], uf_lt[:, 0], uf_lt[:, 2]])

        scal = uf @ self.normal  # cos(angle between kf and detector normal)
        valid = scal > 1e-8
        normeIM = np.where(valid, self.dd / scal, np.nan)

        IMlab = uf * normeIM[:, None]
        OMlab = IMlab - self.IOlab

        xca0 = OMlab[:, 0]
        if abs(self._sinbeta) > 1e-8:
            yca0 = OMlab[:, 1] / self._sinbeta
        else:
            yca0 = -OMlab[:, 2] / self._cosbeta

        xcam1 = self._cosgam * xca0 + self._singam * yca0
        ycam1 = -self._singam * xca0 + self._cosgam * yca0

        xcam = self.xcen + xcam1 / self.pixel_mm
        ycam = self.ycen + ycam1 / self.pixel_mm

        xcam[~valid] = np.nan
        ycam[~valid] = np.nan
        return xcam, ycam

    # ── inverse projection: pixel → kf unit vector ────────────────────────────

    def pixel_to_kf(self, xcam_arr, ycam_arr):
        """
        Map pixel coordinates to scattered unit vectors.
        Implements LaueTools calc_uflab() for Z>0 geometry.

        Parameters
        ----------
        xcam_arr, ycam_arr : array-like of pixel coordinates

        Returns
        -------
        uflab : (N, 3) unit scattered vectors in LT frame  (x // ki)
        """
        xcam1 = (np.asarray(xcam_arr, float) - self.xcen) * self.pixel_mm
        ycam1 = (np.asarray(ycam_arr, float) - self.ycen) * self.pixel_mm

        xca0 = self._cosgam * xcam1 - self._singam * ycam1
        yca0 = self._singam * xcam1 + self._cosgam * ycam1

        xO, yO, zO = self.IOlab
        xM = xO + xca0
        yM = yO + yca0 * self._sinbeta
        zM = zO - yca0 * self._cosbeta

        nIM = np.sqrt(xM**2 + yM**2 + zM**2)
        # Geometry is in LT2 frame (y // beam); convert to LT frame (x // beam):
        # x_LT = y_LT2,  y_LT = -x_LT2,  z_LT = z_LT2
        uflab_lt2 = np.array([xM, yM, zM]).T / nIM[:, None]
        uflab = np.column_stack([uflab_lt2[:, 1], -uflab_lt2[:, 0], uflab_lt2[:, 2]])
        return uflab

    # ── single-spot projection for simulation ────────────────────────────────

    def project(self, kf_hat):
        """
        Project one scattered beam direction (in LT frame, x // beam) onto
        the detector.  Returns (xcam, ycam) in pixels, or None if beam misses.
        """
        # Convert LT -> LT2 for camera geometry
        kf_lt = np.array(kf_hat, dtype=float)
        kf_lt = kf_lt / np.linalg.norm(kf_lt)
        kf = np.array([-kf_lt[1], kf_lt[0], kf_lt[2]])  # LT2 frame
        scal = float(np.dot(kf, self.normal))
        if scal < 1e-8:
            return None
        normeIM = self.dd / scal
        IM = kf * normeIM
        OM = IM - self.IOlab
        xca0 = OM[0]
        yca0 = (
            OM[1] / self._sinbeta
            if abs(self._sinbeta) > 1e-8
            else -OM[2] / self._cosbeta
        )
        xcam1 = self._cosgam * xca0 + self._singam * yca0
        ycam1 = -self._singam * xca0 + self._cosgam * yca0
        xcam = self.xcen + xcam1 / self.pixel_mm
        ycam = self.ycen + ycam1 / self.pixel_mm
        if 0 <= xcam < self.Nh and 0 <= ycam < self.Nv:
            return float(xcam), float(ycam)
        return None

    # ── 2theta / chi from pixel ───────────────────────────────────────────────

    def pixel_to_2theta_chi(self, xcam, ycam):
        """
        Compute 2theta and chi (degrees) from pixel position.

        Uses pixel_to_kf to obtain the scattered unit vector in the LT frame
        (x // beam), then:

            2theta = arccos(uf_x)
            chi    = arctan2(uf_y, uf_z)
        """
        uf_lt = self.pixel_to_kf([xcam], [ycam])[0]
        tth = np.degrees(np.arccos(np.clip(uf_lt[0], -1, 1)))
        chi = np.degrees(np.arctan2(uf_lt[1], uf_lt[2] + 1e-17))
        return tth, chi

    # ── 2theta grid on detector ───────────────────────────────────────────────

    def tth_grid(self, step=None):
        """
        Return a 2theta map over the whole detector (shape Nv x Nh).
        Useful for contour overlays on detector images.
        """
        if step is None:
            step = max(1, self.Nh // 40)
        cs = np.arange(0, self.Nh, step)
        rs = np.arange(0, self.Nv, step)
        CC, RR = np.meshgrid(cs, rs)
        uf = self.pixel_to_kf(CC.ravel(), RR.ravel())
        TTH = np.degrees(np.arccos(np.clip(uf[:, 0], -1, 1))).reshape(CC.shape)
        return CC, RR, TTH

    # ── describe ──────────────────────────────────────────────────────────────

    def describe(self):
        tth_cen, chi_cen = self.pixel_to_2theta_chi(self.xcen, self.ycen)
        corners = [
            (0, 0),
            (self.Nh - 1, 0),
            (0, self.Nv - 1),
            (self.Nh - 1, self.Nv - 1),
        ]
        tths = [self.pixel_to_2theta_chi(c, r)[0] for c, r in corners]
        print(
            f"  Camera ({self.kf_direction}) : {self.Nh} x {self.Nv} px  "
            f"pixel={self.pixel_mm*1e3:.1f} um"
        )
        print(f"  Physical size : {self.size_h_mm:.1f} x {self.size_v_mm:.1f} mm²")
        print(f"  LaueTools calibration:")
        print(f"    dd={self.dd:.3f} mm   xcen={self.xcen:.2f}   ycen={self.ycen:.2f}")
        print(f"    xbet={self.xbet:.3f} deg   xgam={self.xgam:.3f} deg")
        print(
            f"  2theta at (xcen,ycen) : {tth_cen:.4f} deg  "
            f"(= 90 - xbet = {90-self.xbet:.4f} deg)"
        )
        print(f"  chi   at (xcen,ycen) : {chi_cen:.4f} deg")
        print(
            f"  Angular coverage (corners): "
            f"2theta = {min(tths):.1f} - {max(tths):.1f} deg"
        )
        # direct beam position
        ki_hat = np.array([1.0, 0.0, 0.0])
        db = self.project(ki_hat)
        if db:
            print(
                f"  Direct beam footprint: xcam={db[0]:.1f}  ycam={db[1]:.1f}  "
                f"(pixel; should match xcen,ycen for xbet~0)"
            )
        else:
            print("  Direct beam does not hit this detector")

    # ── load from LaueTools calibration dict or list ──────────────────────────

    @classmethod
    def from_lauetools(cls, calib, pixelsize=None, framedim=None, kf_direction="Z>0"):
        """
        Build a Camera from a LaueTools calibration.

        Parameters
        ----------
        calib : list or array  [dd, xcen, ycen, xbet, xgam]
                (= CCDCalibParameters in LaueTools)
        pixelsize : float, mm  (= xpixelsize in LaueTools dict)
        framedim  : (Nh, Nv)   (= framedim in LaueTools dict)
        kf_direction : str     (= kf_direction in LaueTools dict)
        """
        dd, xcen, ycen, xbet, xgam = calib[:5]
        px = pixelsize if pixelsize is not None else PIXEL_SIZE_MM
        Nh, Nv = framedim if framedim is not None else (N_PIX_H, N_PIX_V)
        return cls(
            dd=dd,
            xcen=xcen,
            ycen=ycen,
            xbet=xbet,
            xgam=xgam,
            pixelsize=px,
            n_pix_h=int(Nh),
            n_pix_v=int(Nv),
            kf_direction=kf_direction,
        )

    # ── synthetic image ────────────────────────────────────────────────────────

    def add_poisson_noise(
        self,
        image: "np.ndarray",
        peak_counts: float = 1000.0,
        rng: "np.random.Generator | int | None" = None,
    ) -> "np.ndarray":
        """
        Simulate Poissonian photon-counting statistics on a detector image.

        The image is treated as a map of *relative* intensities.  It is
        scaled so that the brightest pixel has ``peak_counts`` expected
        photons, then every pixel is sampled independently from a Poisson
        distribution with its own expected count λᵢ:

        .. math::

            \\lambda_i = I_i \\cdot \\frac{\\text{peak\\_counts}}{\\max(I)}
            \\qquad
            n_i \\sim \\operatorname{Poisson}(\\lambda_i)

        This reproduces the correct counting statistics: the noise standard
        deviation at pixel *i* is :math:`\\sqrt{\\lambda_i}`, so bright spots
        have more absolute noise but better signal-to-noise than dim pixels.

        .. note::
            Pass a **linear** (non-log-scaled) image so that relative
            intensities are preserved.  Use
            ``camera.render(spots, log_scale=False)`` before calling this
            method, then optionally apply log-scaling to the returned noisy
            image for display.

        Parameters
        ----------
        image : numpy.ndarray, shape (Nv, Nh)
            Linear intensity image from :meth:`render` (``log_scale=False``).
            All values must be ≥ 0.
        peak_counts : float
            Expected photon count at the brightest pixel.  Controls the
            overall exposure level and therefore the noise level:
            SNR ∝ √peak_counts.
        rng : numpy.random.Generator or int or None
            Random-number source.  Pass an integer seed for reproducibility,
            or ``None`` (default) to use a fresh default RNG.

        Returns
        -------
        noisy : numpy.ndarray, shape (Nv, Nh), dtype float32
            Poisson-sampled photon-count image.

        Examples
        --------
        >>> img_linear = camera.render(spots, log_scale=False)
        >>> img_noisy  = camera.add_poisson_noise(img_linear, peak_counts=500)
        >>> img_display = np.log1p(img_noisy / img_noisy.max() * 1000)
        """
        img = np.asarray(image, dtype=np.float64)
        img_max = img.max()
        if img_max > 0:
            lam = img * (float(peak_counts) / img_max)
        else:
            lam = img.copy()

        if isinstance(rng, np.random.Generator):
            gen = rng
        elif rng is None:
            gen = np.random.default_rng()
        else:
            gen = np.random.default_rng(int(rng))

        noisy = gen.poisson(lam).astype(np.float32)
        return noisy

    def render(self, spots, sigma_pix=SPOT_SIGMA_PIX, log_scale=False, normalize=False):
        """
        Render a synthetic detector image (float32, shape Nv x Nh).

        Each spot is drawn as an isotropic 2-D Gaussian weighted by ``I_raw``
        (the un-normalised kinematical intensity).  By default the image is
        returned in raw intensity units with no scaling applied.

        Parameters
        ----------
        spots : list of dicts
            Spot list from any ``simulate_laue*`` function.
            Required keys: ``'pix'`` (xcam, ycam), ``'I_raw'``.
            The key ``'tth'`` (degrees) is required when *sigma_pix* is a
            callable or a broadening result dict.
        sigma_pix : float | callable | dict
            Controls the Gaussian σ (pixels) for each spot:

            * **float** *(default)*  — fixed width for every spot.
            * **callable** ``f(tth_deg) → float``  — per-spot width as a
              function of 2θ.  Pass the ``'model'`` callable returned by
              :func:`~nrxrdct.laue.estimate_instrument_broadening`.
            * **dict** — the full result dict from
              :func:`~nrxrdct.laue.estimate_instrument_broadening`; the
              ``'model'`` key is extracted automatically.

        log_scale : bool
            Apply ``log1p`` compression before returning (default ``False``).
        normalize : bool
            Divide by the image maximum after all other processing
            (default ``False``).
        """
        # Resolve broadening model
        if isinstance(sigma_pix, dict):
            _model = sigma_pix["model"]
            _fixed_sigma = None
        elif callable(sigma_pix):
            _model = sigma_pix
            _fixed_sigma = None
        else:
            _model = None
            _fixed_sigma = float(sigma_pix)

        img = np.zeros((self.Nv, self.Nh), dtype=np.float64)
        for s in spots:
            if s.get("pix") is None:
                continue

            sigma = _model(s["tth"]) if _model is not None else _fixed_sigma
            if not (sigma > 0):
                continue

            c, r = s["pix"]  # xcam, ycam
            ci, ri = int(round(c)), int(round(r))
            margin = int(5 * sigma) + 1
            c0, c1 = max(0, ci - margin), min(self.Nh, ci + margin + 1)
            r0, r1 = max(0, ri - margin), min(self.Nv, ri + margin + 1)
            if c0 >= c1 or r0 >= r1:
                continue
            yy, xx = np.mgrid[r0:r1, c0:c1]
            gauss = np.exp(-((xx - c) ** 2 + (yy - r) ** 2) / (2 * sigma ** 2))
            img[r0:r1, c0:c1] += s["I_raw"] * gauss

        if log_scale:
            img = np.log1p(img)
        if normalize and img.max() > 0:
            img = img / img.max()
        return img.astype(np.float32)

    # ── .det file IO ──────────────────────────────────────────────────────────

    def to_det(self, path) -> None:
        """
        Write a LaueTools-compatible ``.det`` calibration file.

        Format (4 data lines, preceded by a comment header)::

            # LaueTools camera calibration
            # dd xcen ycen xbet xgam   pixelsize   Nh Nv   kf_direction
            # Written: <ISO date>
            dd xcen ycen xbet xgam
            pixelsize
            Nh Nv
            kf_direction
        """
        import datetime
        from pathlib import Path

        header = (
            "# LaueTools camera calibration\n"
            "# dd(mm)  xcen(px)  ycen(px)  xbet(deg)  xgam(deg)"
            "   pixelsize(mm)   Nh Nv   kf_direction\n"
            f"# Written: {datetime.date.today().isoformat()}\n"
        )
        data = (
            f"{self.dd:.6g} {self.xcen:.6g} {self.ycen:.6g} "
            f"{self.xbet:.6g} {self.xgam:.6g}\n"
            f"{self.pixel_mm:.6g}\n"
            f"{self.Nh} {self.Nv}\n"
            f"{self.kf_direction}\n"
        )
        Path(path).write_text(header + data)

    @classmethod
    def from_det(cls, path, pixelsize=None, framedim=None, kf_direction=None):
        """
        Read a LaueTools-compatible ``.det`` calibration file.

        Parameters
        ----------
        path : path-like
            Path to the ``.det`` file.
        pixelsize : float or None
            Override the pixel size (mm) read from the file.
        framedim : (Nh, Nv) or None
            Override the frame dimensions read from the file.
        kf_direction : str or None
            Override the kf_direction read from the file.
        """
        from pathlib import Path

        lines = [
            ln.strip()
            for ln in Path(path).read_text().splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        ]
        dd, xcen, ycen, xbet, xgam = map(float, lines[0].split())
        px = float(lines[1]) if pixelsize is None else float(pixelsize)
        if framedim is None:
            Nh, Nv = map(int, lines[2].split())
        else:
            Nh, Nv = int(framedim[0]), int(framedim[1])
        kfd = lines[3] if kf_direction is None else kf_direction
        return cls(
            dd=dd, xcen=xcen, ycen=ycen, xbet=xbet, xgam=xgam,
            pixelsize=px, n_pix_h=Nh, n_pix_v=Nv, kf_direction=kfd,
        )

    # ── calibration fitting ───────────────────────────────────────────────────

    def fit_calibration(
        self,
        crystal,
        U,
        obs_xy,
        *,
        fit_params=("dd", "xcen", "ycen", "xbet", "xgam"),
        fit_U=False,
        E_min=5_000.0,
        E_max=25_000.0,
        source="bending_magnet",
        source_kwargs=None,
        hmax=15,
        f2_thresh=0.01,
        max_match_px=20.0,
        top_n_sim=None,
        bounds=None,
        dd_range=None,
        cen_range_px=None,
        angle_range_deg=None,
        U_range_deg=None,
        method=None,
        options=None,
    ):
        """
        Fit camera calibration parameters to an observed Laue pattern.

        The cost function is the mean squared nearest-neighbour distance from
        each observed spot to the closest simulated spot, capped at
        ``max_match_px``.  This soft matching gives a smooth landscape suitable
        for derivative-free optimisation.

        When bounds are specified the optimizer switches to L-BFGS-B (which
        enforces them natively); otherwise Nelder-Mead is used.

        Parameters
        ----------
        crystal : Crystal
            Crystal structure of the calibration standard.
        U : (3, 3) array
            Initial orientation matrix.  Updated in the result if ``fit_U=True``.
        obs_xy : (N, 2) array
            Observed spot pixel positions ``[[xcam, ycam], ...]``.
        fit_params : sequence of str
            Camera parameters to optimise.  Any subset of
            ``{"dd", "xcen", "ycen", "xbet", "xgam"}``.
        fit_U : bool
            If ``True``, also refine the orientation matrix via three small-angle
            Euler rotations around lab x, y, z (degrees).
        E_min, E_max : float
            Energy range (eV) for the Laue simulation.
        source : str
            Source model passed to ``simulate_laue``.
        source_kwargs : dict or None
            Extra kwargs for the spectrum function.
        hmax : int
            Maximum Miller index for HKL precomputation.
        f2_thresh : float
            Structure-factor threshold for HKL precomputation.
        max_match_px : float
            Cap distance (pixels) used in the cost function and for
            reporting the final match rate.
        top_n_sim : int or None
            Restrict cost evaluation to the brightest *N* simulated spots.
        bounds : dict or None
            Explicit parameter bounds as ``{param_name: (lo, hi)}``.  Values
            are absolute (not relative).  Keys are the same as ``fit_params``
            plus ``"U_rx"``, ``"U_ry"``, ``"U_rz"`` for orientation angles.
            Takes priority over the convenience range parameters below.
        dd_range : float or None
            Maximum allowed deviation of ``dd`` from its starting value (mm).
            E.g. ``dd_range=5.0`` → bounds ``(dd0 - 5, dd0 + 5)``.
        cen_range_px : float or None
            Maximum allowed deviation of ``xcen`` / ``ycen`` from their
            starting values (pixels).
        angle_range_deg : float or None
            Maximum allowed deviation of ``xbet`` / ``xgam`` from their
            starting values (degrees).
        U_range_deg : float or None
            Maximum allowed rotation of the orientation angles ``U_rx``,
            ``U_ry``, ``U_rz`` from zero (degrees).  Only relevant when
            ``fit_U=True``.
        method : str or None
            Scipy optimisation method.  Defaults to ``"L-BFGS-B"`` when any
            bounds are active, ``"Nelder-Mead"`` otherwise.
        options : dict or None
            Options forwarded to ``scipy.optimize.minimize``, overriding
            defaults.

        Returns
        -------
        CalibrationResult
        """
        from scipy.optimize import minimize
        from scipy.spatial import cKDTree
        from .simulation import simulate_laue, precompute_allowed_hkl

        obs_xy = np.asarray(obs_xy, dtype=float)
        if obs_xy.ndim != 2 or obs_xy.shape[1] != 2:
            raise ValueError("obs_xy must be shape (N, 2)")
        n_obs = len(obs_xy)
        U0 = np.asarray(U, dtype=float).copy()
        src_kw = source_kwargs or {}

        _valid = {"dd", "xcen", "ycen", "xbet", "xgam"}
        for p in fit_params:
            if p not in _valid:
                raise ValueError(f"Unknown fit_param {p!r}; must be one of {_valid}")
        fit_params = list(fit_params)

        E_ref = 0.5 * (E_min + E_max)
        allowed_hkl = precompute_allowed_hkl(crystal, hmax, E_ref, f2_thresh)

        _all_names = fit_params + (["U_rx", "U_ry", "U_rz"] if fit_U else [])
        _cam_init = {
            "dd": self.dd, "xcen": self.xcen, "ycen": self.ycen,
            "xbet": self.xbet, "xgam": self.xgam,
        }
        x0 = np.array(
            [_cam_init[p] for p in fit_params] + ([0.0, 0.0, 0.0] if fit_U else []),
            dtype=float,
        )

        def _build_cam(x):
            kw = dict(
                dd=self.dd, xcen=self.xcen, ycen=self.ycen,
                xbet=self.xbet, xgam=self.xgam,
                pixelsize=self.pixel_mm, n_pix_h=self.Nh, n_pix_v=self.Nv,
                kf_direction=self.kf_direction,
            )
            for i, p in enumerate(fit_params):
                kw[p] = float(x[i])
            return Camera(**kw)

        def _build_U(x):
            if not fit_U:
                return U0
            from scipy.spatial.transform import Rotation
            rx, ry, rz = x[len(fit_params):]
            dU = Rotation.from_euler("xyz", [rx, ry, rz], degrees=True).as_matrix()
            return U0 @ dU

        def _cost(x):
            try:
                cam = _build_cam(x)
                U_cur = _build_U(x)
                spots = simulate_laue(
                    crystal, U_cur, cam,
                    E_min=E_min, E_max=E_max,
                    source=source, source_kwargs=src_kw,
                    hmax=hmax, f2_thresh=f2_thresh,
                    allowed_hkl=allowed_hkl,
                )
            except Exception:
                return float(max_match_px ** 2)

            sim_xy = np.array(
                [s["pix"] for s in spots if s.get("pix") is not None]
            )
            if len(sim_xy) == 0:
                return float(max_match_px ** 2)

            if top_n_sim is not None and len(sim_xy) > top_n_sim:
                I_vals = np.array(
                    [s["I_raw"] for s in spots if s.get("pix") is not None]
                )
                idx = np.argsort(I_vals)[::-1][:top_n_sim]
                sim_xy = sim_xy[idx]

            tree = cKDTree(sim_xy)
            dists, _ = tree.query(obs_xy, k=1)
            return float(np.mean(np.minimum(dists, max_match_px) ** 2))

        # ── build bounds ─────────────────────────────────────────────────────
        # Start from the explicit bounds dict, then fill missing entries from
        # the convenience range parameters.
        _bounds_dict: dict = dict(bounds) if bounds else {}

        _range_defaults = {
            "dd":   dd_range,
            "xcen": cen_range_px,
            "ycen": cen_range_px,
            "xbet": angle_range_deg,
            "xgam": angle_range_deg,
            "U_rx": U_range_deg,
            "U_ry": U_range_deg,
            "U_rz": U_range_deg,
        }
        for pname in _all_names:
            if pname not in _bounds_dict and _range_defaults.get(pname) is not None:
                r = float(_range_defaults[pname])
                centre = _cam_init.get(pname, 0.0)   # orientation deltas start at 0
                _bounds_dict[pname] = (centre - r, centre + r)

        # Ordered bounds list matching x0 / _all_names
        scipy_bounds = (
            [_bounds_dict[p] for p in _all_names]
            if _bounds_dict else None
        )

        # ── choose method ────────────────────────────────────────────────────
        if method is not None:
            _method = method
        elif scipy_bounds is not None:
            _method = "L-BFGS-B"
        else:
            _method = "Nelder-Mead"

        # ── build options ────────────────────────────────────────────────────
        if _method == "Nelder-Mead":
            _steps = {
                "dd": 2.0, "xcen": 50.0, "ycen": 50.0, "xbet": 0.5, "xgam": 0.5,
                "U_rx": 2.0, "U_ry": 2.0, "U_rz": 2.0,
            }
            n_p = len(x0)
            init_simplex = np.tile(x0, (n_p + 1, 1))
            for i, pname in enumerate(_all_names):
                step = _steps.get(pname, 1.0)
                if scipy_bounds:
                    lo, hi = scipy_bounds[i]
                    step = min(step, (hi - lo) * 0.25)
                init_simplex[i + 1, i] += step
            _opts: dict = {
                "maxiter": 5000, "xatol": 0.05, "fatol": 1e-4,
                "initial_simplex": init_simplex,
            }
        else:
            _opts = {"maxiter": 5000, "ftol": 1e-6, "gtol": 1e-6}

        if options:
            _opts.update(options)

        result = minimize(
            _cost, x0,
            method=_method,
            bounds=scipy_bounds,
            options=_opts,
        )

        cam_final = _build_cam(result.x)
        U_final = _build_U(result.x)

        # Final match statistics with the converged parameters
        spots_final = simulate_laue(
            crystal, U_final, cam_final,
            E_min=E_min, E_max=E_max,
            source=source, source_kwargs=src_kw,
            hmax=hmax, f2_thresh=f2_thresh,
            allowed_hkl=allowed_hkl,
        )
        sim_xy_f = np.array(
            [s["pix"] for s in spots_final if s.get("pix") is not None]
        )

        n_matched, rms_px = 0, float("nan")
        if len(sim_xy_f) > 0 and n_obs > 0:
            tree = cKDTree(sim_xy_f)
            dists, _ = tree.query(obs_xy, k=1)
            ok = dists < max_match_px
            n_matched = int(ok.sum())
            if n_matched > 0:
                rms_px = float(np.sqrt((dists[ok] ** 2).mean()))

        return CalibrationResult(
            camera=cam_final,
            U=U_final,
            rms_px=rms_px,
            n_matched=n_matched,
            n_obs=n_obs,
            n_sim=len(sim_xy_f),
            fit_params=tuple(_all_names),
            success=bool(result.success),
            message=result.message,
        )

    def fit_calibration_staged(
        self,
        crystal,
        U,
        obs_xy,
        *,
        # ── Phase A — stabilise geometry ─────────────────────────────────────
        phaseA_params=("dd", "xcen", "ycen", "xbet", "xgam"),
        phaseA_fit_U=False,
        phaseA_dd_range=5.0,
        phaseA_cen_range_px=100.0,
        phaseA_angle_range_deg=1.0,
        phaseA_U_range_deg=0.5,
        phaseA_options=None,
        # ── Phase B — refine orientation ─────────────────────────────────────
        phaseB_params=(),
        phaseB_fit_U=True,
        phaseB_U_range_deg=10.0,
        phaseB_options=None,
        # ── Phase C — global refinement ──────────────────────────────────────
        phaseC_params=("dd", "xcen", "ycen", "xbet", "xgam"),
        phaseC_fit_U=True,
        phaseC_dd_range=2.0,
        phaseC_cen_range_px=30.0,
        phaseC_angle_range_deg=0.5,
        phaseC_U_range_deg=3.0,
        phaseC_options=None,
        # ── shared ───────────────────────────────────────────────────────────
        E_min=5_000.0,
        E_max=25_000.0,
        source="bending_magnet",
        source_kwargs=None,
        hmax=15,
        f2_thresh=0.01,
        max_match_px=20.0,
        top_n_sim=None,
        verbose=False,
    ) -> "CalibrationResult":
        """
        Three-phase staged calibration that alternates between geometry and
        orientation refinement for improved convergence.

        **Phase A — stabilise geometry**
            Refine camera geometry parameters with U fixed (or nearly fixed).
            Large initial bounds allow coarse corrections from the starting
            estimate; tight U bounds prevent orientation drift from corrupting
            the geometry.

        **Phase B — refine orientation**
            Fix the geometry from Phase A and refine the orientation matrix U.
            Geometry is now good enough that the spot pattern can guide U
            without the two coupled parameters fighting each other.

        **Phase C — global refinement**
            Refine geometry and U simultaneously, starting from the Phase B
            solution with tighter bounds.  Resolves the remaining correlation
            between small geometry shifts and orientation corrections.

        Parameters
        ----------
        crystal : Crystal
            Calibration standard.
        U : (3, 3) array
            Initial orientation matrix.
        obs_xy : (N, 2) array
            Observed spot pixel positions.

        phaseA_params : sequence of str
            Camera parameters to optimise in Phase A.
            Default: all five geometry parameters.
        phaseA_fit_U : bool
            Allow a small orientation correction in Phase A.  Default False.
        phaseA_dd_range : float
            Phase A bound on ``dd`` (mm, ± from starting value).
        phaseA_cen_range_px : float
            Phase A bound on ``xcen`` / ``ycen`` (pixels).
        phaseA_angle_range_deg : float
            Phase A bound on ``xbet`` / ``xgam`` (degrees).
        phaseA_U_range_deg : float
            Phase A bound on orientation angles (degrees), only used when
            ``phaseA_fit_U=True``.
        phaseA_options : dict or None
            Extra options for the Phase A optimizer.

        phaseB_params : sequence of str
            Camera parameters to optimise in Phase B.  Default: none
            (geometry frozen).
        phaseB_fit_U : bool
            Refine U in Phase B.  Default True.
        phaseB_U_range_deg : float
            Phase B bound on orientation angles (degrees).
        phaseB_options : dict or None
            Extra options for the Phase B optimizer.

        phaseC_params : sequence of str
            Camera parameters to optimise in Phase C.  Default: all five.
        phaseC_fit_U : bool
            Refine U in Phase C.  Default True.
        phaseC_dd_range : float
            Phase C bound on ``dd`` (mm).
        phaseC_cen_range_px : float
            Phase C bound on ``xcen`` / ``ycen`` (pixels).
        phaseC_angle_range_deg : float
            Phase C bound on ``xbet`` / ``xgam`` (degrees).
        phaseC_U_range_deg : float
            Phase C bound on orientation angles (degrees).
        phaseC_options : dict or None
            Extra options for the Phase C optimizer.

        E_min, E_max : float
            Energy range (eV) shared by all phases.
        source, source_kwargs, hmax, f2_thresh, max_match_px, top_n_sim
            Forwarded unchanged to each :meth:`fit_calibration` call.
        verbose : bool
            Print a one-line summary after each phase.

        Returns
        -------
        CalibrationResult
            Result of Phase C (the final global refinement).  Intermediate
            results are available via ``verbose=True`` output.
        """
        _shared = dict(
            E_min=E_min, E_max=E_max,
            source=source, source_kwargs=source_kwargs,
            hmax=hmax, f2_thresh=f2_thresh,
            max_match_px=max_match_px, top_n_sim=top_n_sim,
        )

        # ── Phase A ───────────────────────────────────────────────────────────
        res_A = self.fit_calibration(
            crystal, U, obs_xy,
            fit_params=phaseA_params,
            fit_U=phaseA_fit_U,
            dd_range=phaseA_dd_range,
            cen_range_px=phaseA_cen_range_px,
            angle_range_deg=phaseA_angle_range_deg,
            U_range_deg=phaseA_U_range_deg if phaseA_fit_U else None,
            options=phaseA_options,
            **_shared,
        )
        if verbose:
            print(f"Phase A (geometry):     {res_A}")

        # ── Phase B ───────────────────────────────────────────────────────────
        res_B = res_A.camera.fit_calibration(
            crystal, res_A.U, obs_xy,
            fit_params=phaseB_params,
            fit_U=phaseB_fit_U,
            U_range_deg=phaseB_U_range_deg if phaseB_fit_U else None,
            options=phaseB_options,
            **_shared,
        )
        if verbose:
            print(f"Phase B (orientation):  {res_B}")

        # ── Phase C ───────────────────────────────────────────────────────────
        res_C = res_B.camera.fit_calibration(
            crystal, res_B.U, obs_xy,
            fit_params=phaseC_params,
            fit_U=phaseC_fit_U,
            dd_range=phaseC_dd_range,
            cen_range_px=phaseC_cen_range_px,
            angle_range_deg=phaseC_angle_range_deg,
            U_range_deg=phaseC_U_range_deg if phaseC_fit_U else None,
            options=phaseC_options,
            **_shared,
        )
        if verbose:
            print(f"Phase C (global):       {res_C}")

        return res_C
