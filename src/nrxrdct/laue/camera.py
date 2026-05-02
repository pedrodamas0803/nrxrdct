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

        Format (4 lines)::

            dd xcen ycen xbet xgam
            pixelsize
            Nh Nv
            kf_direction
        """
        from pathlib import Path

        text = (
            f"{self.dd:.6g} {self.xcen:.6g} {self.ycen:.6g} "
            f"{self.xbet:.6g} {self.xgam:.6g}\n"
            f"{self.pixel_mm:.6g}\n"
            f"{self.Nh} {self.Nv}\n"
            f"{self.kf_direction}\n"
        )
        Path(path).write_text(text)

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
            if ln.strip()
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
        method="Nelder-Mead",
        options=None,
    ):
        """
        Fit camera calibration parameters to an observed Laue pattern.

        The cost function is the mean squared nearest-neighbour distance from
        each observed spot to the closest simulated spot, capped at
        ``max_match_px``.  This soft matching gives a smooth landscape suitable
        for Nelder-Mead optimisation.

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
        method : str
            Scipy optimisation method (default ``"Nelder-Mead"``).
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

        # Build initial simplex with parameter-appropriate step sizes
        _steps = {
            "dd": 2.0, "xcen": 50.0, "ycen": 50.0, "xbet": 0.5, "xgam": 0.5,
            "U_rx": 2.0, "U_ry": 2.0, "U_rz": 2.0,
        }
        n_p = len(x0)
        init_simplex = np.tile(x0, (n_p + 1, 1))
        for i, pname in enumerate(_all_names):
            init_simplex[i + 1, i] += _steps.get(pname, 1.0)

        _opts = {
            "maxiter": 5000, "xatol": 0.05, "fatol": 1e-4,
            "initial_simplex": init_simplex,
        }
        if options:
            _opts.update(options)

        result = minimize(_cost, x0, method=method, options=_opts)

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
