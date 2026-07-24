"""
napari-backed 3-D volume and slice viewers, plus a matplotlib Z-profile
companion plot for napari's own slice slider.

Setup (once per environment)
-----------------------------
    pip install napari ipywidgets matplotlib numpy

Usage in a notebook cell
-------------------------
    from nrxrdct.visualization import visualize_slices_with_profile
    viewer = visualize_slices_with_profile(volume)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import _plot_helpers as _ph
from ._plot_helpers import draw_phase_ticks


def visualize_volume(
    volume: np.ndarray,
    name: str = "Volume",
    colormap: str = "gray",
    rendering: str = "mip",
    contrast_limits: Optional[tuple] = None,
    gamma: float = 1.0,
    opacity: float = 1.0,
    scale: Optional[tuple] = None,
    add_axes: bool = True,
    downsample: Optional[int] = None,
    max_gb: float = 1.0,
) -> napari.Viewer:
    """
    Visualize a 3D NumPy array interactively using napari.

    Args:
        volume (np.ndarray): A 3D array of shape (Z, Y, X) to visualize.
        name (str): Label shown in the napari layer list. Default: "Volume".
        colormap (str): Colormap name (e.g. "grays", "turbo", "magma", "green").
            See napari.utils.colormaps.AVAILABLE_COLORMAPS for the full list.
        rendering (str): Volume rendering mode: "mip" – Max Intensity Projection
            (default), "minip" – Min Intensity Projection, "translucent" –
            semi-transparent full volume, "iso" – isosurface, "attenuated_mip" –
            MIP with depth attenuation.
        contrast_limits (tuple, optional): (low, high) display range. Defaults to
            the data min/max.
        gamma (float): Gamma correction applied to the colormap. Default: 1.0.
        opacity (float): Layer opacity between 0 (transparent) and 1 (opaque).
            Default: 1.0.
        scale (tuple, optional): Physical voxel spacing (z_scale, y_scale, x_scale).
            Useful when voxels are anisotropic (e.g. (4.0, 1.0, 1.0)).
            Default: None (isotropic).
        add_axes (bool): Whether to display the 3D axis widget. Default: True.
        downsample (int, optional): Downsample factor applied uniformly along all
            axes before sending data to the GPU (e.g. 2 → every other voxel).
            When None (default) the factor is chosen automatically so that the
            volume fits within `max_gb` of GPU memory.
        max_gb (float): Maximum GPU memory in GB to target when auto-downsampling.
            Default: 1.0.

    Returns:
        napari.Viewer: The napari viewer instance (keep a reference to prevent GC).

    Example:
        >>> import numpy as np
        >>> vol = np.random.rand(64, 128, 128).astype(np.float32)
        >>> viewer = visualize_volume(vol, colormap="turbo", rendering="mip")
        >>> viewer = visualize_volume(
        ...     vol, name="Confocal stack", colormap="green",
        ...     rendering="translucent", scale=(4.0, 1.0, 1.0),
        ...     contrast_limits=(0.1, 0.9),
        ... )
        >>> napari.run()
    """
    import napari  # type: ignore

    if volume.ndim != 3:
        raise ValueError(
            f"Expected a 3-D array, got shape {volume.shape}. "
            "For 4-D (multi-channel) data use napari directly."
        )

    # --- auto-downsample to avoid GPU out-of-memory ---
    bytes_per_voxel = np.dtype(np.float32).itemsize  # napari uploads as float32
    volume_bytes = volume.size * bytes_per_voxel
    max_bytes = max_gb * 1024**3

    if downsample is None:
        if volume_bytes > max_bytes:
            downsample = int(np.ceil((volume_bytes / max_bytes) ** (1 / 3)))
        else:
            downsample = 1

    if downsample > 1:
        print(
            f"[visualize_volume] Volume is {volume_bytes / 1024**3:.2f} GB — "
            f"downsampling by {downsample}x to fit within {max_gb} GB GPU budget."
        )
        volume = volume[::downsample, ::downsample, ::downsample]
        if scale is not None:
            scale = tuple(s * downsample for s in scale)

    if contrast_limits is None:
        contrast_limits = (float(volume.min()), float(volume.max()))

    viewer = napari.Viewer(title=f"napari — {name}", ndisplay=3)

    viewer.add_image(
        volume,
        name=name,
        colormap=colormap,
        rendering=rendering,
        contrast_limits=contrast_limits,
        gamma=gamma,
        opacity=opacity,
        scale=scale,
    )

    if add_axes:
        viewer.axes.visible = True

    # Zoom camera to fit the volume nicely
    viewer.reset_view()

    return viewer


def visualize_slices(
    volume: np.ndarray,
    name: str = "Volume",
    colormap: str = "gray",
    contrast_limits: Optional[tuple] = None,
    gamma: float = 1.0,
    opacity: float = 1.0,
    scale: Optional[tuple] = None,
    initial_slice: Optional[int] = None,
) -> napari.Viewer:
    """
    Visualize 2-D slices of a 3-D volume parallel to axis-0 (the Z axis)
    using napari's interactive slice slider.

    The viewer opens in 2-D mode.  A slider at the bottom of the window lets
    you scroll through every slice along axis-0 in real time.

    Args:
        volume (np.ndarray): A 3-D array of shape (Z, Y, X). Each slice shown
            is volume[z, :, :].
        name (str): Label shown in the napari layer list. Default: "Volume".
        colormap (str): Napari colormap name (e.g. "grays", "turbo", "magma",
            "green").
        contrast_limits (tuple, optional): (low, high) display range. Defaults
            to data min / max.
        gamma (float): Gamma correction on the colormap. Default: 1.0.
        opacity (float): Layer opacity between 0 and 1. Default: 1.0.
        scale (tuple, optional): Physical voxel spacing (z_scale, y_scale,
            x_scale). Example: (4.0, 1.0, 1.0) for anisotropic confocal data.
        initial_slice (int, optional): Index along axis-0 to show on first open.
            Defaults to the middle slice.

    Returns:
        napari.Viewer: The napari viewer instance (keep a reference to prevent
            garbage collection while the window is open).

    Example:
        >>> import numpy as np
        >>> vol = np.random.rand(64, 128, 128).astype(np.float32)
        >>> viewer = visualize_slices(vol, colormap="turbo")
        >>> viewer = visualize_slices(
        ...     vol, name="Confocal stack", colormap="green",
        ...     scale=(4.0, 1.0, 1.0), contrast_limits=(0.05, 0.95),
        ...     initial_slice=10,
        ... )
        >>> napari.run()
    """
    import napari  # type: ignore

    if volume.ndim != 3:
        raise ValueError(f"Expected a 3-D array, got shape {volume.shape}.")

    n_slices = volume.shape[0]

    if contrast_limits is None:
        contrast_limits = (float(volume.min()), float(volume.max()))

    if initial_slice is None:
        initial_slice = n_slices // 2
    elif not (0 <= initial_slice < n_slices):
        raise ValueError(
            f"initial_slice={initial_slice} is out of range [0, {n_slices - 1}]."
        )

    # --- open viewer in 2-D mode (slices) ---
    viewer = napari.Viewer(title=f"napari — {name}  |  axis-0 slices", ndisplay=2)

    viewer.add_image(
        volume,
        name=name,
        colormap=colormap,
        contrast_limits=contrast_limits,
        gamma=gamma,
        opacity=opacity,
        scale=scale,
    )

    # Position the slider on the requested slice.
    # napari represents the "current step" as a tuple (axis0, axis1, axis2).
    viewer.dims.set_current_step(0, initial_slice)

    viewer.reset_view()

    print(
        f"[visualize_slices] '{name}'  shape={volume.shape}  "
        f"dtype={volume.dtype}  slices along axis-0: {n_slices}\n"
        f"  Use the slider (or arrow keys) to scroll through slices."
    )

    return viewer


class ZProfilePlot:
    """
    A persistent matplotlib figure that displays the Z-profile of a
    selected pixel and updates in-place on every new click.
    """

    def __init__(
        self,
        n_slices: int,
        volume_name: str = "Volume",
        z_values: Optional[np.ndarray] = None,
        z_label: str = "Z  (axis-0 index)",
        phases: Optional[Dict[str, List[float]]] = None,
    ):
        """
        Args:
            n_slices (int): Total number of slices along axis-0 of the volume.
            volume_name (str, optional): Used as part of the matplotlib window title
                (default ``"Volume"``).
            z_values (np.ndarray or None, optional): Physical values for the Z axis
                (e.g. depth in mm or 2θ in degrees). Defaults to integer indices
                ``0 … n_slices-1``.
            z_label (str, optional): X-axis label for the profile plot
                (default ``"Z  (axis-0 index)"``).
            phases (dict, optional): Mapping of phase name → list of peak positions
                (in the same units as *z_values*). Each phase is shown as a row of
                colored tick marks in a panel below the profile. Example::

                    {"Austenite": [2.07, 2.48, 3.59], "Ferrite": [2.03, 2.87]}
        """
        self.n_slices = n_slices
        self.z_axis = (
            np.asarray(z_values) if z_values is not None else np.arange(n_slices)
        )
        self.z_label = z_label

        plt.ion()  # non-blocking mode
        n_phases = len(phases) if phases else 0
        fig_height = 3.6 + (0.4 + 0.22 * n_phases if n_phases else 0)
        self.fig = plt.figure(
            figsize=(6, fig_height),
            facecolor=_ph.BG,
            num=f"Z-profile — {volume_name}",
        )

        if phases:
            gs = gridspec.GridSpec(
                2, 1,
                figure=self.fig,
                left=0.12, right=0.97, top=0.88, bottom=0.12,
                height_ratios=[5, max(1, n_phases)],
                hspace=0.06,
            )
            ax = self.fig.add_subplot(gs[0], facecolor=_ph.PANEL)
            ax_ticks = self.fig.add_subplot(gs[1], facecolor=_ph.PANEL, sharex=ax)
            ax.tick_params(colors=_ph.MUTED, labelsize=8, labelbottom=False)
            self.vline_ticks = draw_phase_ticks(ax_ticks, phases, self.z_axis)
            ax_ticks.set_xlabel(z_label, color=_ph.MUTED, fontsize=9)
        else:
            self.fig.subplots_adjust(left=0.12, right=0.97, top=0.88, bottom=0.16)
            ax = self.fig.add_subplot(111, facecolor=_ph.PANEL)
            ax.tick_params(colors=_ph.MUTED, labelsize=8)
            ax.set_xlabel(z_label, color=_ph.MUTED, fontsize=9)
            self.vline_ticks = None

        for spine in ax.spines.values():
            spine.set_edgecolor(_ph.BORDER)
        ax.set_ylabel("Intensity", color=_ph.MUTED, fontsize=9)
        ax.set_xlim(self.z_axis[0], self.z_axis[-1])

        # Placeholder line
        (self.line,) = ax.plot(
            [],
            [],
            color=_ph.ACCENT,
            linewidth=1.4,
            solid_capstyle="round",
        )
        # Vertical marker for the current slice
        self.vline = ax.axvline(
            x=0, color=_ph.WARN, linewidth=1.0, linestyle="--", alpha=0.8
        )

        self.title = self.fig.suptitle(
            "Click a pixel in napari to start",
            color=_ph.FG,
            fontsize=10,
            fontweight="bold",
        )
        self.ax = ax
        self.fig.canvas.draw_idle()
        plt.pause(0.05)

    def update(self, y: int, x: int, profile: np.ndarray, current_z_idx: int) -> None:
        """
        Redraw the profile for a newly selected pixel.

        Args:
            y (int): Row index of the selected pixel.
            x (int): Column index of the selected pixel.
            profile (np.ndarray): 1-D intensity profile along axis-0, length
                ``n_slices``.
            current_z_idx (int): Current slider position (axis-0 index) used to
                place the vertical marker line.
        """
        self.line.set_xdata(self.z_axis)
        self.line.set_ydata(profile)
        self.ax.set_ylim(profile.min() * 0.95 - 1e-9, profile.max() * 1.05 + 1e-9)
        x_now = [self.z_axis[current_z_idx], self.z_axis[current_z_idx]]
        self.vline.set_xdata(x_now)
        if self.vline_ticks is not None:
            self.vline_ticks.set_xdata(x_now)
        self.title.set_text(f"Z-profile  |  pixel  y={y},  x={x}")
        self.fig.canvas.draw_idle()
        plt.pause(0.01)

    def mark_current_z(self, z_idx: int) -> None:
        """
        Move the vertical slice-position marker without redrawing the profile.

        Args:
            z_idx (int): Current axis-0 slice index.
        """
        x_now = [self.z_axis[z_idx], self.z_axis[z_idx]]
        self.vline.set_xdata(x_now)
        if self.vline_ticks is not None:
            self.vline_ticks.set_xdata(x_now)
        self.fig.canvas.draw_idle()
        plt.pause(0.005)


def visualize_slices_with_profile(
    volume: np.ndarray,
    name: str = "Volume",
    colormap: str = "gray",
    contrast_limits: Optional[tuple] = None,
    gamma: float = 1.0,
    opacity: float = 1.0,
    scale: Optional[tuple] = None,
    initial_slice: Optional[int] = None,
    z_values: Optional[np.ndarray] = None,
    z_label: str = "Z  (axis-0 index)",
    phases: Optional[Union[Dict[str, List[float]], Dict[str, pd.DataFrame]]] = None,
) -> napari.Viewer:
    """
    Visualize 2-D slices of a 3-D volume (parallel to axis-0) in napari.

    **Click any pixel** in the napari canvas to open / update a matplotlib
    window showing the intensity profile along axis-0 (Z) at that (y, x)
    location.  A dashed orange vertical line tracks the currently displayed
    slice as you move the slider.

    Args:
        volume (np.ndarray): 3-D array of shape (Z, Y, X).
        name (str): Layer name shown in the napari layer list.
        colormap (str): Napari colormap (e.g. "grays", "turbo", "magma").
        contrast_limits (tuple, optional): Display range. Defaults to data min/max.
        gamma (float): Gamma correction for the colormap.
        opacity (float): Layer opacity [0, 1].
        scale (tuple, optional): Anisotropic voxel spacing (z, y, x).
        initial_slice (int, optional): Axis-0 index to display on open. Defaults
            to middle slice.
        z_values (array-like, optional): Physical values for the X axis of the
            Z-profile plot (e.g. depths in mm, timestamps, wavelengths). Must have
            exactly ``volume.shape[0]`` elements. Defaults to integer slice indices.
        z_label (str, optional): Label for the X axis of the Z-profile plot.
            Default: "Z  (axis-0 index)".
        phases (dict, optional): Either a mapping of phase name → list of peak
            positions (in the same units as *z_values*), or the
            ``dict[str, pd.DataFrame]`` returned by ``get_powder_xrd_peaks``
            (the ``tth`` column is used as tick positions).  Each phase is
            shown as a row of colored tick marks in a panel below the Z-profile
            plot. Example::

                {"Austenite": [2.07, 2.48, 3.59], "Ferrite": [2.03, 2.87]}
                # or:
                peaks = get_powder_xrd_peaks(["au.cif", "fe.cif"])
                phases=peaks

    Returns:
        napari.Viewer: The napari viewer instance.

    Note:
        Works with matplotlib's interactive backend (Qt / Tk / Wx). Call
        ``napari.run()`` at the end of a script to start the event loop.

    Example:
        >>> import numpy as np
        >>> vol = np.random.rand(64, 128, 128).astype(np.float32)
        >>> viewer = visualize_slices_with_profile(vol, colormap="turbo")
        >>> napari.run()
    """
    import napari  # type: ignore

    if volume.ndim != 3:
        raise ValueError(f"Expected a 3-D array, got shape {volume.shape}.")

    n_slices = volume.shape[0]

    if z_values is not None:
        z_values = np.asarray(z_values, dtype=float)
        if z_values.shape != (n_slices,):
            raise ValueError(
                f"z_values must have length {n_slices} (= volume.shape[0]), "
                f"got {z_values.shape}."
            )

    if contrast_limits is None:
        contrast_limits = (float(volume.min()), float(volume.max()))

    if initial_slice is None:
        initial_slice = n_slices // 2
    elif not (0 <= initial_slice < n_slices):
        raise ValueError(
            f"initial_slice={initial_slice} out of range [0, {n_slices - 1}]."
        )

    # --- napari viewer (2-D slice mode) ------------------------------------
    viewer = napari.Viewer(
        title=f"napari — {name}  |  axis-0 slices  (click pixel → Z-profile)",
        ndisplay=2,
    )

    image_layer = viewer.add_image(
        volume,
        name=name,
        colormap=colormap,
        contrast_limits=contrast_limits,
        gamma=gamma,
        opacity=opacity,
        scale=scale,
    )

    viewer.dims.set_current_step(0, initial_slice)
    viewer.reset_view()

    # --- Points layer to mark the selected pixel ---------------------------
    point_layer = viewer.add_points(
        data=np.empty((0, 2)),
        name="Selected pixel",
        size=10,
        face_color="red",
        symbol="cross",
    )

    # --- Z-profile plot -----------------------------------------------------
    profile_plot = ZProfilePlot(
        n_slices=n_slices,
        volume_name=name,
        z_values=z_values,
        z_label=z_label,
        phases=phases,
    )

    # Track last selected pixel so the vline keeps updating on slice changes
    last_pixel = {"y": None, "x": None}

    # --- Callback: click in the napari canvas -------------------------------
    @image_layer.mouse_drag_callbacks.append
    def on_click(layer: Any, event: Any) -> None:
        # Only react on left-button press (not drag)
        if event.type != "mouse_press" or event.button != 1:
            return

        # event.position is in *world* coordinates (respects scale).
        # For a 3-D array displayed in 2-D, position = (z_world, y_world, x_world).
        pos = event.position  # world coords

        # Convert world → data (pixel) indices
        data_coords = layer.world_to_data(pos)

        if len(data_coords) == 3:
            z_idx, y_idx, x_idx = (int(round(c)) for c in data_coords)
        elif len(data_coords) == 2:
            # Some napari versions drop the z dim in 2-D mode
            y_idx, x_idx = (int(round(c)) for c in data_coords)
            z_idx = int(viewer.dims.current_step[0])
        else:
            return

        # Clamp to valid range
        z_idx = np.clip(z_idx, 0, volume.shape[0] - 1)
        y_idx = np.clip(y_idx, 0, volume.shape[1] - 1)
        x_idx = np.clip(x_idx, 0, volume.shape[2] - 1)

        # Update marker on the slice
        point_layer.data = np.array([[y_idx, x_idx]], dtype=float)

        # Extract Z-profile and update plot
        profile = volume[:, y_idx, x_idx]
        current_z = int(viewer.dims.current_step[0])
        profile_plot.update(y_idx, x_idx, profile, current_z)

        last_pixel["y"] = y_idx
        last_pixel["x"] = x_idx

        print(
            f"[Z-profile] pixel (y={y_idx}, x={x_idx})  "
            f"min={profile.min():.4f}  max={profile.max():.4f}  "
            f"mean={profile.mean():.4f}"
        )

    # --- Callback: slider moves → update vertical marker -------------------
    def on_slice_change(event: Any) -> None:
        if last_pixel["y"] is not None:
            current_z = int(viewer.dims.current_step[0])
            profile_plot.mark_current_z(current_z)

    viewer.dims.events.current_step.connect(on_slice_change)

    print(
        f"\n[visualize_slices_with_profile] '{name}'\n"
        f"  Shape  : {volume.shape}\n"
        f"  dtype  : {volume.dtype}\n"
        f"  Slices : {n_slices} along axis-0\n"
        f"  → Click any pixel in the napari window to see its Z-profile.\n"
        f"  → Use the slider (or ← → arrow keys) to scroll through slices.\n"
    )

    return viewer