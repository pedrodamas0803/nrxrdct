"""
StackViewer — interactive Jupyter widget for browsing 2-D/3-D NumPy arrays.

Displays a single 2-D array, or scrolls through the slices of a 3-D array
along a chosen axis, with live ipywidgets controls for the colormap,
display range (vmin / vmax), and colour normalization (linear, log,
symmetric log, power, or centered/diverging). When browsing a stack,
clicking any pixel in the image opens a profile panel showing the
intensity along the stack axis at that pixel.

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

from typing import Dict, List, Optional, Union

import ipywidgets as widgets
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

from . import _plot_helpers as _ph
from ._plot_helpers import draw_phase_ticks

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

    When the array is a stack (more than one slice), clicking a pixel in the
    image opens a profile panel plotting the intensity along the stack axis
    at that pixel, with a marker tracking the currently displayed slice.

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
        figsize (tuple, optional): Figure size in inches. Defaults to a size
            that accommodates the profile panel when one is shown.
        z_values (array-like, optional): Physical coordinates for the stack
            axis (e.g. depth, energy, angle). Must have exactly
            ``n_slices`` elements. Defaults to integer slice indices. Only
            used when a profile panel is shown.
        z_label (str, optional): X-axis label of the profile panel.
            Default: "Slice index".
        phases (dict, optional): Either a mapping of name -> list of
            positions (in the same units as *z_values*), or a
            ``dict[str, pd.DataFrame]`` (e.g. the output of
            ``get_powder_xrd_peaks``) whose ``tth`` column is used as the
            positions. Drawn as a row of coloured tick marks below the
            profile panel. Example::

                {"Austenite": [2.07, 2.48, 3.59], "Ferrite": [2.03, 2.87]}
        show_profile (bool, optional): Whether to show the click-to-plot
            profile panel. Defaults to ``True`` when the array has more
            than one slice, ``False`` for a plain 2-D array (a profile
            along a single slice would be trivial).

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
        >>> depths = np.linspace(0, 31.5, 20)
        >>> viewer3 = StackViewer(vol, z_values=depths, z_label="Depth (µm)")
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
        figsize: Optional[tuple] = None,
        z_values: Optional[np.ndarray] = None,
        z_label: str = "Slice index",
        phases: Optional[Union[Dict[str, List[float]], Dict[str, pd.DataFrame]]] = None,
        show_profile: Optional[bool] = None,
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
        self.show_profile = (
            self.n_slices > 1 if show_profile is None else bool(show_profile)
        ) and self.n_slices > 1

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
        norm = norm if norm in _NORMALIZATIONS else "Linear"

        self.phases = phases
        if self.show_profile:
            self.z_axis = (
                np.asarray(z_values, dtype=float)
                if z_values is not None
                else np.arange(self.n_slices, dtype=float)
            )
            if self.z_axis.shape != (self.n_slices,):
                raise ValueError(
                    f"z_values must have length {self.n_slices} (= number of "
                    f"slices), got {self.z_axis.shape}."
                )
        else:
            self.z_axis = None
        self.z_label = z_label

        self._sel = {"y": None, "x": None}

        if figsize is None:
            figsize = (10.5, 5.5) if self.show_profile else (6.5, 5.5)

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
            self.fig = plt.figure(figsize=figsize, facecolor=_ph.BG)
        try:
            self.fig.canvas.manager.set_window_title(f"StackViewer — {self.name}")
        except Exception:
            pass

        if self.show_profile:
            self._build_image_and_profile_axes()
        else:
            self.ax = self.fig.add_axes([0.1, 0.08, 0.75, 0.84], facecolor=_ph.PANEL)
            self.ax.tick_params(colors=_ph.MUTED, labelsize=7)
            for spine in self.ax.spines.values():
                spine.set_edgecolor(_ph.BORDER)
            self.ax_prof = None
            self.ax_ticks = None
            self.vline_prof = None
            self.vline_ticks = None

        self.im = self.ax.imshow(
            self.data[0],
            cmap=cmap,
            origin="upper",
            interpolation="nearest",
        )
        self.cbar = self.fig.colorbar(self.im, ax=self.ax, fraction=0.046, pad=0.04)
        self.cbar.ax.tick_params(colors=_ph.MUTED, labelsize=7)
        self.cbar.outline.set_edgecolor(_ph.BORDER)

        if self.show_profile:
            self._build_profile_artists()
            self.fig.canvas.mpl_connect("button_press_event", self._on_click)

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

        # ── Colormap (dropdown) ──────────────────────────────────────
        cmap_options = _COLORMAPS if cmap in _COLORMAPS else [cmap, *_COLORMAPS]
        self.cmap_box = widgets.Dropdown(
            value=cmap,
            options=cmap_options,
            description="Colormap",
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
        info_text = f"shape={self.data.shape[1:]}  dtype={self.data.dtype}"
        if self.n_slices > 1:
            info_text += f"  |  {self.n_slices} slices"
        if self.show_profile:
            info_text += "  |  click a pixel for its profile"
        self.info = widgets.Label(value=info_text, layout=widgets.Layout(width="auto"))
        self.pixel_info = widgets.Label(value="", layout=widgets.Layout(width="auto"))

    def _build_image_and_profile_axes(self) -> None:
        if self.phases:
            gs = gridspec.GridSpec(
                1, 2, figure=self.fig,
                left=0.07, right=0.97, bottom=0.1, top=0.92,
                wspace=0.32, width_ratios=[1.15, 1.0],
            )
            self.ax = self.fig.add_subplot(gs[0], facecolor=_ph.PANEL)
            n_phases = len(self.phases)
            gs_right = gridspec.GridSpecFromSubplotSpec(
                2, 1, subplot_spec=gs[1],
                height_ratios=[5, max(1, n_phases)], hspace=0.08,
            )
            self.ax_prof = self.fig.add_subplot(gs_right[0], facecolor=_ph.PANEL)
            self.ax_ticks = self.fig.add_subplot(
                gs_right[1], facecolor=_ph.PANEL, sharex=self.ax_prof
            )
            self.vline_ticks = draw_phase_ticks(self.ax_ticks, self.phases, self.z_axis)
            self.ax_ticks.set_xlabel(self.z_label, color=_ph.MUTED, fontsize=8)
            self.ax_prof.tick_params(colors=_ph.MUTED, labelsize=7, axis="x", labelbottom=False)
            self.ax_prof.tick_params(colors=_ph.MUTED, labelsize=7, axis="y")
        else:
            gs = gridspec.GridSpec(
                1, 2, figure=self.fig,
                left=0.07, right=0.97, bottom=0.1, top=0.92,
                wspace=0.32, width_ratios=[1.15, 1.0],
            )
            self.ax = self.fig.add_subplot(gs[0], facecolor=_ph.PANEL)
            self.ax_prof = self.fig.add_subplot(gs[1], facecolor=_ph.PANEL)
            self.ax_ticks = None
            self.vline_ticks = None
            self.ax_prof.tick_params(colors=_ph.MUTED, labelsize=7)
            self.ax_prof.set_xlabel(self.z_label, color=_ph.MUTED, fontsize=8)

        for spine in self.ax.spines.values():
            spine.set_edgecolor(_ph.BORDER)
        for spine in self.ax_prof.spines.values():
            spine.set_edgecolor(_ph.BORDER)
        self.ax.tick_params(colors=_ph.MUTED, labelsize=7)

    def _build_profile_artists(self) -> None:
        self.ax_prof.set_title(
            "Profile  —  click a pixel", color=_ph.MUTED, fontsize=9, pad=6
        )
        self.ax_prof.set_ylabel("Intensity", color=_ph.MUTED, fontsize=8)
        self.ax_prof.set_xlim(self.z_axis[0], self.z_axis[-1])
        self.ax_prof.set_ylim(self._data_min, self._data_max)

        (self.profile_line,) = self.ax_prof.plot(
            [], [], color=_ph.ACCENT, linewidth=1.4, solid_capstyle="round"
        )
        current_x = self.z_axis[self.n_slices // 2]
        self.vline_prof = self.ax_prof.axvline(
            x=current_x, color=_ph.WARN, linewidth=1.0, linestyle="--", alpha=0.85
        )
        if self.vline_ticks is not None:
            self.vline_ticks.set_xdata([current_x, current_x])

        # Crosshair on the image (hidden until first click)
        (self.hline,) = self.ax.plot([], [], color=_ph.WARN, linewidth=0.8, alpha=0.7)
        (self.vline_img,) = self.ax.plot([], [], color=_ph.WARN, linewidth=0.8, alpha=0.7)
        (self.dot,) = self.ax.plot(
            [], [], "o", color="#ff4444", markersize=6,
            markeredgecolor="white", markeredgewidth=0.8,
        )

    def _set_extra_visibility(self, norm: str) -> None:
        self.gamma_slider.layout.display = "flex" if norm == "Power" else "none"
        self.linthresh_slider.layout.display = (
            "flex" if norm == "Symmetric log" else "none"
        )
        self.vcenter_box.layout.display = "flex" if norm == "Centered" else "none"

    def _display(self) -> None:
        children = [
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
        ]
        if self.show_profile:
            children.append(self.pixel_info)
        controls = widgets.VBox(children, layout=widgets.Layout(width="360px", padding="4px 8px"))
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
            self.status.value = f"<span style='color:{_ph.WARN}'>Unknown colormap '{name}'</span>"
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
        self.range_slider.value = [float(np.nanmin(current)), float(np.nanmax(current))]

    def _on_click(self, event) -> None:
        if event.inaxes is not self.ax or event.button != 1:
            return
        if event.xdata is None or event.ydata is None:
            return
        ny, nx = self.data.shape[1:]
        x = int(np.clip(round(event.xdata), 0, nx - 1))
        y = int(np.clip(round(event.ydata), 0, ny - 1))
        self._sel = {"y": y, "x": x}
        self._redraw_profile()

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------
    def _redraw_image(self, idx: int) -> None:
        self.im.set_data(self.data[idx])
        title = f"{self.name}"
        if self.n_slices > 1:
            title += f"  —  slice {idx + 1}/{self.n_slices}"
        self.ax.set_title(title, color=_ph.FG, fontsize=10, pad=8)
        if self.show_profile:
            self._sync_profile_vline(idx)
        self._apply_style()

    def _sync_profile_vline(self, idx: int) -> None:
        xv = self.z_axis[idx]
        self.vline_prof.set_xdata([xv, xv])
        if self.vline_ticks is not None:
            self.vline_ticks.set_xdata([xv, xv])

    def _redraw_profile(self) -> None:
        y, x = self._sel["y"], self._sel["x"]
        if y is None:
            return

        profile = self.data[:, y, x]
        self.profile_line.set_xdata(self.z_axis)
        self.profile_line.set_ydata(profile)
        p_min, p_max = float(np.nanmin(profile)), float(np.nanmax(profile))
        margin = max((p_max - p_min) * 0.05, 1e-9)
        self.ax_prof.set_ylim(p_min - margin, p_max + margin)
        self._sync_profile_vline(self.slider.value)
        self.ax_prof.set_title(f"Profile  |  pixel  y={y}, x={x}", color=_ph.MUTED, fontsize=9, pad=6)

        ny, nx = self.data.shape[1:]
        self.hline.set_xdata([0, nx - 1])
        self.hline.set_ydata([y, y])
        self.vline_img.set_xdata([x, x])
        self.vline_img.set_ydata([0, ny - 1])
        self.dot.set_xdata([x])
        self.dot.set_ydata([y])

        self.pixel_info.value = (
            f"pixel (y={y}, x={x})  min={p_min:.4g}  max={p_max:.4g}  "
            f"mean={float(np.nanmean(profile)):.4g}"
        )
        self.fig.canvas.draw_idle()


def visualize_slices_with_profile_jupyter(
    volume: np.ndarray,
    name: str = "Volume",
    colormap: str = "gray",
    contrast_limits: Optional[tuple] = None,
    initial_slice: Optional[int] = None,
    z_values: Optional[np.ndarray] = None,
    z_label: str = "Z  (axis-0 index)",
    figsize: tuple = (12, 5),
    phases: Optional[Union[Dict[str, List[float]], Dict[str, pd.DataFrame]]] = None,
    return_fig: bool = False,
) -> Optional[plt.Figure]:
    """
    Deprecated: use :class:`StackViewer` directly.

    Thin backward-compatible wrapper around :class:`StackViewer` that
    reproduces the signature of the original
    ``nrxrdct.xrdct.visualization.visualize_slices_with_profile_jupyter``.

    Args:
        See :class:`StackViewer` — *colormap* maps to *cmap*, *contrast_limits*
        maps to *(vmin, vmax)*, and *initial_slice* is ignored (the viewer
        always starts on the middle slice; move the slider to change it).

    Returns:
        matplotlib.figure.Figure or None: The figure, if *return_fig* is True.
    """
    vmin, vmax = contrast_limits if contrast_limits else (None, None)
    viewer = StackViewer(
        volume,
        name=name,
        cmap=colormap,
        vmin=vmin,
        vmax=vmax,
        figsize=figsize,
        z_values=z_values,
        z_label=z_label,
        phases=phases,
        show_profile=True,
    )
    if initial_slice is not None:
        viewer.slider.value = initial_slice
    if return_fig:
        return viewer.fig
    return None


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