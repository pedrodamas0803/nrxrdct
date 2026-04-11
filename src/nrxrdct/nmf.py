"""
Non-negative Matrix Factorisation (NMF) for hyperspectral XRD-CT volumes.

Originally partially prepared by Beatriz G. Foschiani (CEA Grenoble).

Wraps scikit-learn's :class:`~sklearn.decomposition.NMF` for volumetric
diffraction data, providing :class:`HyperspectralNMF` as a convenient high-level
interface and private helpers for fitting and visualisation.
"""

# This module was partially prepared by Beatriz G. Foschiani - CEA Grenoble

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import NMF


class HyperspectralNMF:
    """
    High-level interface for NMF decomposition of a hyperspectral XRD-CT volume.

    Reshapes the 3-D volume into a 2-D pixel-by-channel matrix, runs
    scikit-learn NMF, and exposes the component maps, component spectra, and
    per-pixel RMSE through :meth:`fit_data`.  Results can be visualised with
    :meth:`plot`.

    Attributes:
        vol (np.ndarray): Original ``(nx, ny, n_channels)`` volume passed at construction.
        X (np.ndarray): Flattened 2-D representation of shape ``(nx*ny, n_channels)``
            used as NMF input.
        n_comp (int): Number of NMF components.
        x_spectra (np.ndarray): Spectral axis values.
        unit (str): Spectral axis label used in plots.
        W_maps (np.ndarray): Per-component spatial intensity maps, shape ``(nx, ny, K)``.
            Set after calling :meth:`fit_data`.
        H (np.ndarray): Component spectra of shape ``(K, n_channels)``.
            Set after calling :meth:`fit_data`.
        X_rec (np.ndarray): Reconstructed data matrix ``W @ H``, shape
            ``(nx*ny, n_channels)``. Set after calling :meth:`fit_data`.
        E_map (np.ndarray): Per-pixel RMSE of shape ``(nx, ny)``.
            Set after calling :meth:`fit_data`.
        model (sklearn.decomposition.NMF): Fitted scikit-learn NMF object.
            Set after calling :meth:`fit_data`.
    """

    def __init__(
        self,
        volume: np.ndarray,
        n_components: int,
        spectral_axis: np.ndarray,
        unit_name="energy (keV)",
        loss_function="frobenius",  # "frobenius" or "kullback-leibler"
        solver=None,  # None -> pick default based on loss
        init="nndsvdar",
        max_iter=1000,
        random_state=0,
        l1_ratio=0.0,
        alpha_W=0.0,
        alpha_H=0.0,
        clip_negative=True,
    ):
        """
        Args:
            volume (np.ndarray): 3-D array of shape ``(nx, ny, n_channels)``
                containing the hyperspectral volume.
            n_components (int): Number of NMF components (endmembers) to extract.
            spectral_axis (np.ndarray): 1-D array of spectral axis values
                (e.g. 2θ or energy), length ``n_channels``.
            unit_name (str, optional): Label for the spectral axis used in plots
                (default ``"energy (keV)"``).
            loss_function (str, optional): NMF beta-loss: ``"frobenius"`` (default)
                or ``"kullback-leibler"``.
            solver (str or None, optional): NMF solver; ``None`` selects ``"cd"``
                for Frobenius and ``"mu"`` for KL loss automatically.
            init (str, optional): Initialisation strategy passed to
                :class:`~sklearn.decomposition.NMF` (default ``"nndsvdar"``).
            max_iter (int, optional): Maximum number of NMF iterations (default 1000).
            random_state (int, optional): Random seed for reproducibility (default 0).
            l1_ratio (float, optional): Elastic-net mixing parameter; 0 = pure L2,
                1 = pure L1 (default 0.0).
            alpha_W (float, optional): Regularisation strength applied to W
                (default 0.0).
            alpha_H (float, optional): Regularisation strength applied to H
                (default 0.0).
            clip_negative (bool, optional): If ``True``, negative values in the
                input are clipped to zero before fitting (default ``True``).
        """
        self.vol = volume
        self.X = volume.reshape((volume.shape[0] * volume.shape[1], volume.shape[2]))
        self.map_shape = (volume.shape[0], volume.shape[1])
        self.n_comp = n_components
        # self.comp_map = np.empty((self.n_comp, self.map_shape[0], self.map_shape[1]), dtype=float)
        self.x_spectra = spectral_axis
        self.unit = unit_name
        self.loss_function = loss_function
        self.solver = solver
        self.init = init
        self.iter_max = max_iter
        self.rand_state = random_state
        self.l1ratio = l1_ratio
        self.alphaW = alpha_W
        self.alphaH = alpha_H
        self.clip_negative = clip_negative

    def fit_data(self) -> None:
        """
        Fit the NMF model and store results as instance attributes.

        After calling this method the following attributes are available:

        - ``W_maps`` — ``(nx, ny, K)`` component intensity maps
        - ``H``      — ``(K, n_channels)`` component spectra
        - ``X_rec``  — ``(n_pixels, n_channels)`` reconstructed data matrix
        - ``E_map``  — ``(nx, ny)`` per-pixel RMSE map
        - ``model``  — fitted :class:`~sklearn.decomposition.NMF` object
        """
        W_maps, H, X_rec, E_map, model = _nmf_sklearn_hyperspectral(
            X=self.X,  # (n_pixels, n_channels)
            map_shape=self.map_shape,  # (nx, ny) with nx*ny == n_pixels
            n_components=self.n_comp,
            wavelength=self.x_spectra,  # (n_channels,)
            unit_name=self.unit,
            loss=self.loss_function,  # "frobenius" or "kullback-leibler"
            solver=self.solver,  # None -> pick default based on loss
            init=self.init,
            max_iter=self.iter_max,
            random_state=self.rand_state,
            l1_ratio=self.l1ratio,
            alpha_W=self.alphaW,
            alpha_H=self.alphaH,
            clip_negative=self.clip_negative,
            show_progress=True,
        )

        self.W_maps = W_maps
        self.H = H
        self.X_rec = X_rec
        self.E_map = E_map
        self.model = model

    def plot(
        self,
        normalize_spectra: bool = False,
        titles=None,
        figsize=(14, 7),
        extent=None,
        save=True,
    ) -> None:
        """
        Visualise the NMF decomposition with component spectra and spatial maps.

        Args:
            normalize_spectra (bool, optional): If ``True``, normalise each component
                spectrum to unit L2 norm before plotting (default ``False``).
            titles (list or None, optional): Per-component subplot titles; defaults
                to ``["Component 1", ...]``.
            figsize (tuple, optional): Figure size in inches (default ``(14, 7)``).
            extent (tuple or None, optional): ``(xmin, xmax, ymin, ymax)`` passed to
                :func:`~matplotlib.pyplot.imshow` for physical axis scaling
                (default ``None``).
            save (bool, optional): If ``True``, save the figure to
                ``nmf_decomposition_<K>_components.png`` (default ``True``).
        """
        _plot_nmf_panel(
            W_maps=self.W_maps,
            H=self.H,
            E_map=self.E_map,
            wavelength=self.x_spectra,
            unit_name=self.unit,
            normalize_spectra=normalize_spectra,
            titles=titles,
            figsize=figsize,
            extent=extent,
            save=save,
        )


def _nmf_sklearn_hyperspectral(
    X,  # (n_pixels, n_channels)
    map_shape,  # (nx, ny) with nx*ny == n_pixels
    n_components=4,
    wavelength=None,  # (n_channels,)
    unit_name="Wavelength (nm)",
    loss="frobenius",  # "frobenius" or "kullback-leibler"
    solver=None,  # None -> pick default based on loss
    init="nndsvdar",
    max_iter=1000,
    random_state=0,
    l1_ratio=0.0,
    alpha_W=0.0,
    alpha_H=0.0,
    clip_negative=True,
    show_progress=True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, NMF]:
    """
    Fit NMF with scikit-learn on a 2-D hyperspectral array.

    Args:
        X (array-like): Flattened hyperspectral data of shape ``(n_pixels, n_channels)``;
            must be non-negative (or ``clip_negative=True``).
        map_shape (tuple): ``(nx, ny)`` spatial dimensions with ``nx * ny == n_pixels``.
        n_components (int, optional): Number of NMF components (default 4).
        wavelength (np.ndarray or None, optional): Spectral axis values of length
            ``n_channels``; auto-generated if ``None``.
        unit_name (str, optional): Spectral axis label used in the progress display
            (default ``"Wavelength (nm)"``).
        loss (str, optional): Beta-loss: ``"frobenius"`` (default) or
            ``"kullback-leibler"``.
        solver (str or None, optional): NMF solver; ``None`` picks ``"cd"`` for
            Frobenius and ``"mu"`` for KL.
        init (str, optional): Initialisation strategy (default ``"nndsvdar"``).
        max_iter (int, optional): Maximum iterations (default 1000).
        random_state (int, optional): Random seed (default 0).
        l1_ratio (float, optional): Elastic-net mixing parameter (default 0.0).
        alpha_W (float, optional): Regularisation on W (default 0.0).
        alpha_H (float, optional): Regularisation on H (default 0.0).
        clip_negative (bool, optional): Clip negative values to zero before fitting
            (default ``True``).
        show_progress (bool, optional): Show a tqdm progress bar (default ``True``).

    Returns:
        W_maps (np.ndarray): Per-component spatial intensity maps, shape ``(nx, ny, K)``.
        H (np.ndarray): Component spectra of shape ``(K, n_channels)``.
        X_rec (np.ndarray): Reconstructed data matrix ``W @ H``, shape
            ``(n_pixels, n_channels)``.
        E_map (np.ndarray): Per-pixel RMSE of shape ``(nx, ny)``.
        model: Fitted :class:`~sklearn.decomposition.NMF` object.
    """
    if show_progress:
        from tqdm.auto import tqdm

        pbar = tqdm(total=4, desc="NMF (sklearn)", unit="step")

        def step(msg):
            pbar.set_postfix_str(msg)
            pbar.update(1)

    else:
        pbar = None

        def step(msg):  # no-op
            return

    try:
        step("prepare input")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D (n_pixels, n_channels). Got {X.shape}.")

        n_pixels, n_ch = X.shape
        nx, ny = map_shape
        if nx * ny != n_pixels:
            raise ValueError(
                f"map_shape product must match n_pixels. Got {nx*ny} vs {n_pixels}."
            )

        # NMF requires non-negative data
        if clip_negative and X.min() < 0:
            X = X.copy()
            X[X < 0] = 0.0
        elif X.min() < 0:
            raise ValueError(
                "NMF requires non-negative X. Clip negatives or change preprocessing."
            )

        if wavelength is None:
            wavelength = np.arange(n_ch)

        # Choose solver default
        if solver is None:
            solver = "mu" if loss == "kullback-leibler" else "cd"

        model = NMF(
            n_components=n_components,
            init=init,
            solver=solver,
            beta_loss=loss,
            max_iter=max_iter,
            random_state=random_state,
            alpha_W=alpha_W,
            alpha_H=alpha_H,
            l1_ratio=l1_ratio,
        )

        # Fit
        step("fit_transform (this is the long part)")
        W = model.fit_transform(X)  # (n_pixels, K)
        H = model.components_  # (K, n_channels)

        # Reconstruction
        step("reconstruct")
        X_rec = W @ H

        # Residuals / reshape
        step("rmse + reshape")
        rmse = np.sqrt(np.mean((X - X_rec) ** 2, axis=1))
        E_map = rmse.reshape(nx, ny)
        W_maps = W.reshape(nx, ny, n_components)

        return W_maps, H, X_rec, E_map, model

    finally:
        if pbar is not None:
            pbar.close()


def _plot_nmf_panel(
    W_maps,
    H,
    E_map,
    wavelength,
    unit_name="Wavelength (nm)",
    normalize_spectra=True,
    titles=None,
    figsize=(14, 7),
    extent=None,
    save=True,
) -> None:
    """
    Render a publication-style NMF results panel.

    Layout: one row of component spectra (top, full width) and one row of
    spatial maps plus a residual RMSE map (bottom).

    Args:
        W_maps (np.ndarray): Component intensity maps of shape ``(nx, ny, K)``.
        H (np.ndarray): Component spectra of shape ``(K, n_channels)``.
        E_map (np.ndarray): Per-pixel RMSE map of shape ``(nx, ny)``.
        wavelength (np.ndarray): Spectral axis values.
        unit_name (str, optional): Spectral axis label
            (default ``"Wavelength (nm)"``).
        normalize_spectra (bool, optional): Normalise each spectrum to unit L2
            norm before plotting (default ``True``).
        titles (list or None, optional): Component subplot titles; defaults to
            ``["Component 1", ...]``.
        figsize (tuple, optional): Figure size in inches (default ``(14, 7)``).
        extent (tuple or None, optional): ``(xmin, xmax, ymin, ymax)`` for
            spatial map axis scaling (default ``None``).
        save (bool, optional): Save to ``nmf_decomposition_<K>_components.png``
            (default ``True``).
    """
    K = H.shape[0]
    if titles is None:
        titles = [f"Component {k+1}" for k in range(K)]

    fig = plt.figure(figsize=figsize)

    # --- spectra (top, full width) ---
    ax0 = plt.subplot2grid((2, K + 1), (0, 0), colspan=K + 1)
    for k in range(K):
        y = H[k].copy()
        if normalize_spectra:
            y /= np.linalg.norm(y) + 1e-16
        ax0.plot(wavelength, y, label=str(k + 1))
    ax0.set_xlabel(unit_name)
    ax0.set_ylabel("Intensity (a.u.)" if normalize_spectra else "Intensity")
    ax0.legend(ncol=min(K, 6), frameon=False)
    ax0.grid(True, alpha=0.25)

    # --- component maps (bottom row) ---
    for k in range(K):
        ax = plt.subplot2grid((2, K + 1), (1, k))
        im = ax.imshow(W_maps[:, :, k], origin="upper", extent=extent)
        ax.set_title(titles[k])
        ax.invert_yaxis()
        # ax.set_xticks([])
        # ax.set_yticks([])

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # --- residual map (last column) ---
    axR = plt.subplot2grid((2, K + 1), (1, K))
    imR = axR.imshow(E_map, origin="upper", extent=extent)
    axR.set_title("Residual (RMSE)")
    axR.invert_yaxis()
    # axR.set_xticks([])
    # axR.set_yticks([])
    plt.colorbar(imR, ax=axR, fraction=0.046, pad=0.04)

    plt.tight_layout()
    if save:
        plt.savefig(f"nmf_decomposition_{len(H):d}_components.png", dpi=150)
    # plt.show()
