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
        peak_counts: float = 1000,
        rng: "np.random.Generator | int | None" = None,
    ) -> "np.ndarray":
        """
        Apply Poissonian counting noise to a rendered detector image.

        The image is first scaled so that its maximum pixel equals
        *peak_counts* (the expected photon count at the brightest spot),
        then each pixel is drawn independently from a Poisson distribution
        with that expected value.

        .. note::
            Pass a **linear** (non-log-scaled) image.  Use
            ``camera.render(..., log_scale=False)`` to obtain the raw image
            before calling this method.

        Parameters
        ----------
        image : numpy.ndarray, shape (Nv, Nh)
            Linear intensity image, e.g. from :meth:`render` with
            ``log_scale=False``.  All values must be ≥ 0.
        peak_counts : float
            Expected photon count at the brightest pixel.  Higher values
            give lower relative noise (SNR ∝ √peak_counts).
        rng : numpy.random.Generator or int or None
            Random-number source.  Pass an integer seed for reproducibility
            or ``None`` (default) to use the global NumPy RNG.

        Returns
        -------
        noisy : numpy.ndarray, shape (Nv, Nh), dtype float32
            Poisson-sampled image in units of photon counts.

        Examples
        --------
        >>> img_linear = camera.render(spots, log_scale=False)
        >>> img_noisy  = camera.add_poisson_noise(img_linear, peak_counts=500)
        """
        img = np.asarray(image, dtype=np.float64)
        if img.max() > 0:
            img = img * (peak_counts / img.max())

        if isinstance(rng, np.random.Generator):
            gen = rng
        elif rng is None:
            gen = np.random.default_rng()
        else:
            gen = np.random.default_rng(int(rng))

        noisy = gen.poisson(img).astype(np.float32)
        return noisy

    def render(self, spots, sigma_pix=SPOT_SIGMA_PIX, log_scale=True, normalize=False):
        """
        Render a synthetic detector image (float32, shape Nv x Nh).
        Each spot is a 2D Gaussian of width sigma_pix.
        spot's 'pix' entry must be (xcam, ycam) in LaueTools convention.

        Parameters
        ----------
        normalize : bool
            If True, divide the image by its maximum value so intensities
            are in [0, 1] before any log scaling.
        """
        img = np.zeros((self.Nv, self.Nh), dtype=np.float32)
        margin = int(5 * sigma_pix) + 1
        for s in spots:
            if s.get("pix") is None:
                continue
            c, r = s["pix"]  # xcam, ycam
            ci, ri = int(round(c)), int(round(r))
            c0, c1 = max(0, ci - margin), min(self.Nh, ci + margin + 1)
            r0, r1 = max(0, ri - margin), min(self.Nv, ri + margin + 1)
            if c0 >= c1 or r0 >= r1:
                continue
            yy, xx = np.mgrid[r0:r1, c0:c1]
            gauss = np.exp(-((xx - c) ** 2 + (yy - r) ** 2) / (2 * sigma_pix**2))
            img[r0:r1, c0:c1] += s["intensity"] * gauss
        if normalize and img.max() > 0:
            img = img / img.max()
        if log_scale and img.max() > 0:
            img = np.log1p(img / img.max() * 1000)
        return img
