"""
StackViewer — interactive Jupyter widget for browsing 2-D/3-D NumPy arrays.

Displays a single 2-D array, or scrolls through the slices of a 3-D array
along a chosen axis, with live ipywidgets controls for the colormap,
display range (vmin / vmax), and colour normalization (linear, log,
symmetric log, power, or centered/diverging).

Setup (once per environment)
-----------------------------
    pip install ipywidgets matplotlib numpy
    # Classic notebook:
    jupyter nbextension enable --py widgetsnbextension
    # JupyterLab:
    pip install jupyterlab_widgets

Usage in a notebook cell
-------------------------
    %matplotlib widget          # or 'inline'
    from nrxrdct.visualization import StackViewer
    viewer = StackViewer(volume)
"""

from __future__ import annotations

from typing import Optional

import ipywidgets as widgets
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display

# ── colour palette (matches nrxrdct.xrdct.visualization / laue.interactive) ──
_BG = "#0e1117"
_PANEL = "#161b22"
_FG = "#e6edf3"
_MUTED = "#8b949e"
_BORDER = "#30363d"
_ACCENT = "#58a6ff"
_WARN = "#f0883e"

_COLORMAPS = [
    "gray", "viridis", "plasma", "inferno", "magma", "cividis",
    "turbo", "jet", "coolwarm", "RdBu_r", "seismic", "twilight",
    "hot", "bone", "cubehelix", "terrain", "nipy_spectral",
]

_NORMALIZATIONS = ["Linear", "Log", "Symmetric log", "Power", "Centered"]


class StackViewer:
    """
    Interactive slice viewer for 2-D or 3-D NumPy arrays in Jupyter.

    A 2-D array is shown as a single static image. A 3-D array is treated as
    a stack of 2-D images along *axis* (default: axis 0); a slider lets you
    scroll through the slices. Additional controls let you change the
    colormap, the display range (vmin / vmax), and the colour normalization
    on the fly.

    Args:
        array (np.ndarray): A 2-D array ``(Y, X)`` or 3-D array to browse.
            For a 3-D array, the slice axis is *axis* and the other two
            axes are displayed as the image.
        axis (int, optional): Axis of *array* to scroll through when it is
            3-D. Default: 0.
        name (str, optional): Title shown above the figure. Default: "Stack".
        cmap (str, optional): Initial matplotlib colormap name. Default: "gray".
        vmin (float, optional): Initial lower display limit. Defaults to the
            data minimum.
        vmax (float, optional): Initial upper display limit. Defaults to the
            data maximum.
        norm ({"Linear", "Log", "Symmetric log", "Power", "Centered"}, optional):
            Initial colour normalization. Default: "Linear".
        gamma (float, optional): Initial exponent used by the "Power"
            normalization. Default: 1.0.
        figsize (tuple, optional): Figure size in inches. Default: (6.5, 5.5).

    Attributes:
        fig (plt.Figure): The underlying matplotlib figure.
        ax (plt.Axes): The image axes.
        im (matplotlib.image.AxesImage): The image artist being updated.

    Example:
        >>> import numpy as np
        >>> from nrxrdct.visualization import StackViewer
        >>> vol = np.random.rand(20, 128, 128).astype(np.float32)
        >>> viewer = StackViewer(vol, cmap="turbo")
        >>> img = np.random.rand(128, 128)
        >>> viewer2 = StackViewer(img, norm="Log")
    """

    def __init__(
        self,
        array: np.ndarray,
        axis: int = 0,
        name: str = "Stack",
        cmap: str = "gray",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        norm: str = "Linear",
        gamma: float = 1.0,
        figsize: tuple = (6.5, 5.5),
    ) -> None:
        array = np.asarray(array)
        if array.ndim < 2:
            raise ValueError(
                f"Expected an array with at least 2 dimensions, got shape {array.shape}."
            )
        if array.ndim > 3:
            raise ValueError(
                f"Expected a 2-D or 3-D array, got shape {array.shape} "
                f"(ndim={array.ndim})."
            )

        if array.ndim == 3:
            if not (-3 <= axis < 3):
                raise ValueError(f"axis={axis} out of range for a 3-D array.")
            self.data = np.moveaxis(array, axis, 0)
        else:
            self.data = array[np.newaxis, ...]

        self.name = name
        self.n_slices = self.data.shape[0]

        data_min = float(np.nanmin(self.data))
        data_max = float(np.nanmax(self.data))
        if data_min == data_max:
            data_max = data_min + 1.0
        self._data_min = data_min
        self._data_max = data_max

        positive = self.data[self.data > 0]
        self._positive_min = float(positive.min()) if positive.size else 1e-12

        vmin = data_min if vmin is None else float(vmin)
        vmax = data_max if vmax is None else float(vmax)
        cmap = cmap if cmap in _COLORMAPS else cmap
        norm = norm if norm in _NORMALIZATIONS else "Linear"

        self._build_widgets(vmin, vmax, cmap, norm, gamma, figsize)
        self._redraw_image(self.n_slices // 2 if self.n_slices > 1 else 0)
        self._display()

    # ------------------------------------------------------------------
    # Widget construction
    # ------------------------------------------------------------------
    def _build_widgets(
        self,
        vmin: float,
        vmax: float,
        cmap: str,
        norm: str,
        gamma: float,
        figsize: tuple,
    ) -> None:
        with plt.ioff():
            self.fig = plt.figure(figsize=figsize, facecolor=_BG)
        try:
            self.fig.canvas.manager.set_window_title(f"StackViewer — {self.name}")
        except Exception:
            pass

        self.ax = self.fig.add_axes([0.1, 0.08, 0.75, 0.84], facecolor=_PANEL)
        self.ax.tick_params(colors=_MUTED, labelsize=7)
        for spine in self.ax.spines.values():
            spine.set_edgecolor(_BORDER)

        self.im = self.ax.imshow(
            self.data[0],
            cmap=cmap,
            origin="upper",
            interpolation="nearest",
        )
        self.cbar = self.fig.colorbar(self.im, ax=self.ax, fraction=0.046, pad=0.04)
        self.cbar.ax.tick_params(colors=_MUTED, labelsize=7)
        self.cbar.outline.set_edgecolor(_BORDER)

        # ── Slice slider ──────────────────────────────────────────────
        self.slider = widgets.IntSlider(
            value=self.n_slices // 2,
            min=0,
            max=self.n_slices - 1,
            step=1,
            description="Slice",
            continuous_update=True,
            layout=widgets.Layout(width="97%"),
            style={"description_width": "80px"},
            disabled=self.n_slices == 1,
        )
        self.slider.observe(self._on_slice_change, names="value")

        # ── Colormap (free-text with suggestions) ────────────────────
        self.cmap_box = widgets.Combobox(
            value=cmap,
            placeholder="colormap name",
            options=_COLORMAPS,
            description="Colormap",
            ensure_option=False,
            layout=widgets.Layout(width="97%"),
            style={"description_width": "80px"},
        )
        self.cmap_box.observe(self._on_cmap_change, names="value")

        # ── Normalization ──────────────────────────────────────────────
        self.norm_dd = widgets.Dropdown(
            options=_NORMALIZATIONS,
            value=norm,
            description="Norm",
            layout=widgets.Layout(width="97%"),
            style={"description_width": "80px"},
        )
        self.norm_dd.observe(self._on_norm_type_change, names="value")

        # ── Display range (vmin / vmax) ──────────────────────────────
        pad = 0.05 * (self._data_max - self._data_min)
        self.range_slider = widgets.FloatRangeSlider(
            value=[vmin, vmax],
            min=self._data_min - pad,
            max=self._data_max + pad,
            step=(self._data_max - self._data_min) / 500 or 1e-6,
            description="vmin/vmax",
            continuous_update=True,
            readout_format=".4g",
            layout=widgets.Layout(width="97%"),
            style={"description_width": "80px"},
        )
        self.range_slider.observe(self._on_style_change, names="value")

        self.reset_full_btn = widgets.Button(
            description="Full range", layout=widgets.Layout(width="110px")
        )
        self.reset_full_btn.on_click(self._reset_full_range)

        self.reset_slice_btn = widgets.Button(
            description="Slice range", layout=widgets.Layout(width="110px")
        )
        self.reset_slice_btn.on_click(self._reset_slice_range)

        # ── Norm-specific extra parameters ───────────────────────────
        self.gamma_slider = widgets.FloatSlider(
            value=gamma,
            min=0.1,
            max=5.0,
            step=0.1,
            description="Gamma",
            continuous_update=True,
            layout=widgets.Layout(width="97%"),
            style={"description_width": "80px"},
        )
        self.gamma_slider.observe(self._on_style_change, names="value")

        linthresh0 = max(abs(vmax - vmin) * 0.01, 1e-6)
        self.linthresh_slider = widgets.FloatLogSlider(
            value=linthresh0,
            base=10,
            min=-6,
            max=6,
            step=0.1,
            description="Linthresh",
            continuous_update=True,
            layout=widgets.Layout(width="97%"),
            style={"description_width": "80px"},
        )
        self.linthresh_slider.observe(self._on_style_change, names="value")

        self.vcenter_box = widgets.FloatText(
            value=0.0,
            description="Center",
            layout=widgets.Layout(width="97%"),
            style={"description_width": "80px"},
        )
        self.vcenter_box.observe(self._on_style_change, names="value")

        self._set_extra_visibility(norm)

        self.status = widgets.HTML(value="")
        self.info = widgets.Label(
            value=f"shape={self.data.shape[1:]}  dtype={self.data.dtype}"
            + (f"  |  {self.n_slices} slices" if self.n_slices > 1 else ""),
            layout=widgets.Layout(width="auto"),
        )

    def _set_extra_visibility(self, norm: str) -> None:
        self.gamma_slider.layout.display = "flex" if norm == "Power" else "none"
        self.linthresh_slider.layout.display = (
            "flex" if norm == "Symmetric log" else "none"
        )
        self.vcenter_box.layout.display = "flex" if norm == "Centered" else "none"

    def _display(self) -> None:
        controls = widgets.VBox(
            [
                self.slider,
                self.cmap_box,
                self.norm_dd,
                self.range_slider,
                widgets.HBox(
                    [self.reset_full_btn, self.reset_slice_btn],
                    layout=widgets.Layout(gap="6px", margin="2px 0 2px 0"),
                ),
                self.gamma_slider,
                self.linthresh_slider,
                self.vcenter_box,
                self.status,
                self.info,
            ],
            layout=widgets.Layout(width="360px", padding="4px 8px"),
        )
        display(widgets.HBox([self.fig.canvas, controls]))

    # ------------------------------------------------------------------
    # Normalization helper
    # ------------------------------------------------------------------
    def _build_norm(self, norm_name: str, vmin: float, vmax: float) -> mcolors.Normalize:
        if vmin >= vmax:
            vmax = vmin + 1e-9

        if norm_name == "Log":
            safe_vmin = vmin if vmin > 0 else self._positive_min
            if safe_vmin >= vmax:
                safe_vmin = vmax * 1e-3 if vmax > 0 else 1e-12
            return mcolors.LogNorm(vmin=safe_vmin, vmax=vmax)

        if norm_name == "Symmetric log":
            linthresh = max(self.linthresh_slider.value, 1e-12)
            return mcolors.SymLogNorm(linthresh=linthresh, vmin=vmin, vmax=vmax, base=10)

        if norm_name == "Power":
            return mcolors.PowerNorm(gamma=self.gamma_slider.value, vmin=vmin, vmax=vmax)

        if norm_name == "Centered":
            vcenter = self.vcenter_box.value
            lo, hi = vmin, vmax
            if vcenter <= lo:
                lo = vcenter - 1e-9
            if vcenter >= hi:
                hi = vcenter + 1e-9
            return mcolors.TwoSlopeNorm(vmin=lo, vcenter=vcenter, vmax=hi)

        return mcolors.Normalize(vmin=vmin, vmax=vmax)

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    def _on_slice_change(self, change: dict) -> None:
        self._redraw_image(change["new"])

    def _on_norm_type_change(self, change: dict) -> None:
        self._set_extra_visibility(change["new"])
        self._apply_style()

    def _on_cmap_change(self, change: dict) -> None:
        name = change["new"]
        try:
            cmap_obj = plt.get_cmap(name)
        except (ValueError, KeyError):
            self.status.value = (
                f"<span style='color:{_WARN}'>Unknown colormap '{name}'</span>"
            )
            return
        self.status.value = ""
        self.im.set_cmap(cmap_obj)
        self.fig.canvas.draw_idle()

    def _on_style_change(self, change: dict) -> None:
        self._apply_style()

    def _apply_style(self) -> None:
        vmin, vmax = self.range_slider.value
        norm_obj = self._build_norm(self.norm_dd.value, vmin, vmax)
        self.im.set_norm(norm_obj)
        self.cbar.update_normal(self.im)
        self.fig.canvas.draw_idle()

    def _reset_full_range(self, _button: widgets.Button) -> None:
        self.range_slider.value = [self._data_min, self._data_max]

    def _reset_slice_range(self, _button: widgets.Button) -> None:
        current = self.data[self.slider.value]
        self.range_slider.value = [
            float(np.nanmin(current)),
            float(np.nanmax(current)),
        ]

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------
    def _redraw_image(self, idx: int) -> None:
        self.im.set_data(self.data[idx])
        title = f"{self.name}"
        if self.n_slices > 1:
            title += f"  —  slice {idx + 1}/{self.n_slices}"
        self.ax.set_title(title, color=_FG, fontsize=10, pad=8)
        self._apply_style()


# ---------------------------------------------------------------------------
# Quick smoke-test (run as a script — not inside a notebook)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(
        "This module is designed for Jupyter notebooks.\n"
        "Import it in a cell and instantiate StackViewer().\n\n"
        "Minimal example:\n"
        "  %matplotlib widget\n"
        "  import numpy as np\n"
        "  from nrxrdct.visualization import StackViewer\n"
        "  vol = np.random.rand(20, 128, 128).astype(np.float32)\n"
        "  StackViewer(vol, cmap='turbo')\n"
    )