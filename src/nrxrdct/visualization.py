"""
Interactive 3D volume visualization using napari or matplotlib in jupyter.

Everything runs inside a notebook cell using ipywidgets + matplotlib.
No napari required.

Setup (once per environment)
-----------------------------
    pip install ipywidgets matplotlib numpy
    # Classic notebook:
    jupyter nbextension enable --py widgetsnbextension
    # JupyterLab:
    pip install jupyterlab_widgets

Usage in a notebook cell
-------------------------
    %matplotlib widget          # or 'inline' — see note in docstring
    from visualize_slices_jupyter import visualize_slices_with_profile_jupyter
    visualize_slices_with_profile_jupyter(volume)
"""

from typing import Optional

import ipywidgets as widgets
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import napari  # type: ignore
import numpy as np
from IPython.display import display
from matplotlib.lines import Line2D
from matplotlib.patches import Circle


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
    if volume.ndim != 3:
        raise ValueError(
            f"Expected a 3-D array, got shape {volume.shape}. "
            "For 4-D (multi-channel) data use napari directly."
        )

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
        """
        self.n_slices = n_slices
        self.z_axis = (
            np.asarray(z_values) if z_values is not None else np.arange(n_slices)
        )
        self.z_label = z_label

        plt.ion()  # non-blocking mode
        self.fig = plt.figure(
            figsize=(6, 3.6),
            facecolor="#0e1117",
            num=f"Z-profile — {volume_name}",
        )
        self.fig.subplots_adjust(left=0.12, right=0.97, top=0.88, bottom=0.16)

        ax = self.fig.add_subplot(111, facecolor="#161b22")
        ax.tick_params(colors="#8b949e", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")
        ax.set_xlabel(z_label, color="#8b949e", fontsize=9)
        ax.set_ylabel("Intensity", color="#8b949e", fontsize=9)
        ax.set_xlim(self.z_axis[0], self.z_axis[-1])

        # Placeholder line
        (self.line,) = ax.plot(
            [],
            [],
            color="#58a6ff",
            linewidth=1.4,
            solid_capstyle="round",
        )
        # Vertical marker for the current slice
        self.vline = ax.axvline(
            x=0, color="#f0883e", linewidth=1.0, linestyle="--", alpha=0.8
        )

        self.title = self.fig.suptitle(
            "Click a pixel in napari to start",
            color="#e6edf3",
            fontsize=10,
            fontweight="bold",
        )
        self.ax = ax
        self.fig.canvas.draw_idle()
        plt.pause(0.05)

    def update(self, y: int, x: int, profile: np.ndarray, current_z_idx: int):
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
        self.vline.set_xdata([self.z_axis[current_z_idx], self.z_axis[current_z_idx]])
        self.title.set_text(f"Z-profile  |  pixel  y={y},  x={x}")
        self.fig.canvas.draw_idle()
        plt.pause(0.01)

    def mark_current_z(self, z_idx: int):
        """
        Move the vertical slice-position marker without redrawing the profile.

        Args:
            z_idx (int): Current axis-0 slice index.
        """
        self.vline.set_xdata([self.z_axis[z_idx], self.z_axis[z_idx]])
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
    )

    # Track last selected pixel so the vline keeps updating on slice changes
    last_pixel = {"y": None, "x": None}

    # --- Callback: click in the napari canvas -------------------------------
    @image_layer.mouse_drag_callbacks.append
    def on_click(layer, event):
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
    def on_slice_change(event):
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


def visualize_slices_with_profile_jupyter(
    volume: np.ndarray,
    name: str = "Volume",
    colormap: str = "gray",
    contrast_limits: Optional[tuple] = None,
    initial_slice: Optional[int] = None,
    z_values: Optional[np.ndarray] = None,
    z_label: str = "Z  (axis-0 index)",
    figsize: tuple = (12, 5),
) -> None:
    """
    Display an interactive 2-D slice viewer with a Z-profile panel inside
    a Jupyter notebook.

    Layout: left panel shows 2-D slices (axis-0 = Z scrolled by a slider,
    click any pixel to update); right panel shows the intensity profile along
    axis-0 at the selected (y, x) with a dashed orange line marking the current
    slice.

    Args:
        volume (np.ndarray): 3-D array of shape (Z, Y, X).
        name (str): Title shown above the figure.
        colormap (str): Matplotlib colormap name (e.g. "gray", "turbo", "magma",
            "viridis").
        contrast_limits (tuple, optional): (vmin, vmax) for the image display.
            Defaults to data min / max.
        initial_slice (int, optional): Axis-0 index shown on first render.
            Defaults to middle slice.
        z_values (array-like, optional): Physical coordinates for the X axis of
            the Z-profile plot (e.g. depths in µm, timestamps, wavelengths). Must
            have exactly ``volume.shape[0]`` elements. Defaults to integer indices.
        z_label (str, optional): X-axis label of the Z-profile plot.
            Default: ``"Z  (axis-0 index)"``.
        figsize (tuple, optional): Overall figure size in inches. Default: (12, 5).

    Note:
        Recommended cell magic is ``%matplotlib widget`` (ipympl backend) for a
        truly live canvas. ``%matplotlib inline`` also works but redraws on every
        interaction.

    Example:
        In a notebook cell::

            %matplotlib widget
            import numpy as np
            vol = np.random.rand(64, 128, 128).astype(np.float32)
            visualize_slices_with_profile_jupyter(vol, colormap="turbo")

            depths = np.linspace(0, 31.5, 64)
            visualize_slices_with_profile_jupyter(
                vol, z_values=depths, z_label="Depth (µm)",
            )
    """
    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3-D array, got shape {volume.shape}.")

    n_slices, n_y, n_x = volume.shape

    z_axis = (
        np.asarray(z_values, dtype=float)
        if z_values is not None
        else np.arange(n_slices, dtype=float)
    )
    if z_axis.shape != (n_slices,):
        raise ValueError(
            f"z_values must have length {n_slices} (= volume.shape[0]), "
            f"got {z_axis.shape}."
        )

    vmin, vmax = (
        contrast_limits
        if contrast_limits
        else (float(volume.min()), float(volume.max()))
    )

    if initial_slice is None:
        initial_slice = n_slices // 2
    elif not (0 <= initial_slice < n_slices):
        raise ValueError(
            f"initial_slice={initial_slice} out of range [0, {n_slices - 1}]."
        )

    # State shared between callbacks
    state = {
        "z_idx": initial_slice,
        "y_sel": None,
        "x_sel": None,
    }

    # ------------------------------------------------------------------
    # Figure layout
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=figsize, facecolor="#0e1117")
    fig.suptitle(name, color="#e6edf3", fontsize=12, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(
        1,
        2,
        figure=fig,
        left=0.06,
        right=0.97,
        bottom=0.12,
        top=0.91,
        wspace=0.35,
    )

    # --- Left: slice image ---
    ax_img = fig.add_subplot(gs[0], facecolor="#161b22")
    ax_img.set_title(
        "Slice viewer  —  click to select pixel", color="#8b949e", fontsize=9, pad=6
    )
    ax_img.tick_params(colors="#8b949e", labelsize=7)
    for sp in ax_img.spines.values():
        sp.set_edgecolor("#30363d")
    ax_img.set_xlabel("X", color="#8b949e", fontsize=8)
    ax_img.set_ylabel("Y", color="#8b949e", fontsize=8)

    img_display = ax_img.imshow(
        volume[initial_slice],
        cmap=colormap,
        vmin=vmin,
        vmax=vmax,
        origin="upper",
        interpolation="nearest",
        aspect="auto",
    )

    # Crosshair marker (hidden until first click)
    (hline,) = ax_img.plot([], [], color="#f0883e", linewidth=0.8, alpha=0.7)
    (vline_img,) = ax_img.plot([], [], color="#f0883e", linewidth=0.8, alpha=0.7)
    (dot,) = ax_img.plot(
        [],
        [],
        "o",
        color="#ff4444",
        markersize=6,
        markeredgecolor="white",
        markeredgewidth=0.8,
    )

    slice_label = ax_img.set_title(
        f"Slice viewer  —  click to select pixel",
        color="#8b949e",
        fontsize=9,
        pad=6,
    )

    # --- Right: Z-profile ---
    ax_prof = fig.add_subplot(gs[1], facecolor="#161b22")
    ax_prof.set_title(
        "Z-profile  —  select a pixel", color="#8b949e", fontsize=9, pad=6
    )
    ax_prof.tick_params(colors="#8b949e", labelsize=7)
    for sp in ax_prof.spines.values():
        sp.set_edgecolor("#30363d")
    ax_prof.set_xlabel(z_label, color="#8b949e", fontsize=8)
    ax_prof.set_ylabel("Intensity", color="#8b949e", fontsize=8)
    ax_prof.set_xlim(z_axis[0], z_axis[-1])
    ax_prof.set_ylim(vmin, vmax)

    (profile_line,) = ax_prof.plot(
        [], [], color="#58a6ff", linewidth=1.4, solid_capstyle="round"
    )
    vline_prof = ax_prof.axvline(
        x=z_axis[initial_slice],
        color="#f0883e",
        linewidth=1.0,
        linestyle="--",
        alpha=0.85,
    )
    prof_title = ax_prof.set_title(
        "Z-profile  —  select a pixel", color="#8b949e", fontsize=9, pad=6
    )

    # ------------------------------------------------------------------
    # Slider widget
    # ------------------------------------------------------------------
    slider = widgets.IntSlider(
        value=initial_slice,
        min=0,
        max=n_slices - 1,
        step=1,
        description=f"Slice (Z):",
        continuous_update=True,
        layout=widgets.Layout(width="90%"),
        style={"description_width": "80px"},
    )

    slice_info = widgets.Label(
        value=_slice_info_text(initial_slice, z_axis, z_label),
        layout=widgets.Layout(width="auto"),
    )

    pixel_info = widgets.Label(
        value="",
        layout=widgets.Layout(width="auto"),
    )

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------
    def _redraw_slice(z_idx: int):
        img_display.set_data(volume[z_idx])
        vline_prof.set_xdata([z_axis[z_idx], z_axis[z_idx]])
        # Update crosshair title
        ax_img.set_title(
            f"Slice {z_idx}  |  {z_label} = {z_axis[z_idx]:.4g}",
            color="#8b949e",
            fontsize=9,
            pad=6,
        )
        fig.canvas.draw_idle()

    def _redraw_profile(y: int, x: int, z_idx: int):
        profile = volume[:, y, x]
        profile_line.set_xdata(z_axis)
        profile_line.set_ydata(profile)
        p_min, p_max = profile.min(), profile.max()
        margin = max((p_max - p_min) * 0.05, 1e-9)
        ax_prof.set_ylim(p_min - margin, p_max + margin)
        vline_prof.set_xdata([z_axis[z_idx], z_axis[z_idx]])
        prof_title.set_text(f"Z-profile  |  pixel  y={y},  x={x}")
        # Crosshair on image
        hline.set_xdata([0, n_x - 1])
        hline.set_ydata([y, y])
        vline_img.set_xdata([x, x])
        vline_img.set_ydata([0, n_y - 1])
        dot.set_xdata([x])
        dot.set_ydata([y])
        fig.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    def on_slider_change(change):
        z_idx = change["new"]
        state["z_idx"] = z_idx
        slice_info.value = _slice_info_text(z_idx, z_axis, z_label)
        _redraw_slice(z_idx)
        if state["y_sel"] is not None:
            # Just move the vline; profile data stays the same
            vline_prof.set_xdata([z_axis[z_idx], z_axis[z_idx]])
            fig.canvas.draw_idle()

    slider.observe(on_slider_change, names="value")

    def on_click(event):
        # Only react to clicks inside the image axis
        if event.inaxes is not ax_img:
            return
        if event.button != 1:
            return

        x = int(round(event.xdata))
        y = int(round(event.ydata))
        x = np.clip(x, 0, n_x - 1)
        y = np.clip(y, 0, n_y - 1)

        state["y_sel"] = y
        state["x_sel"] = x

        profile = volume[:, y, x]
        pixel_info.value = (
            f"pixel (y={y}, x={x})  "
            f"min={profile.min():.4f}  "
            f"max={profile.max():.4f}  "
            f"mean={profile.mean():.4f}"
        )
        _redraw_profile(y, x, state["z_idx"])

    fig.canvas.mpl_connect("button_press_event", on_click)

    # ------------------------------------------------------------------
    # Layout & display
    # ------------------------------------------------------------------
    controls = widgets.VBox(
        [
            widgets.HBox([slider, slice_info]),
            pixel_info,
        ],
        layout=widgets.Layout(margin="4px 0 0 0"),
    )

    display(widgets.VBox([fig.canvas, controls]))

    # Initial render
    _redraw_slice(initial_slice)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------
def _slice_info_text(z_idx: int, z_axis: np.ndarray, z_label: str) -> str:
    return f"  {z_label} = {z_axis[z_idx]:.4g}  (index {z_idx})"


# ---------------------------------------------------------------------------
# Quick smoke-test (run as a script — not inside a notebook)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(
        "This module is designed for Jupyter notebooks.\n"
        "Import it in a cell and call visualize_slices_with_profile_jupyter().\n\n"
        "Minimal example:\n"
        "  %matplotlib widget\n"
        "  import numpy as np\n"
        "  from visualize_slices_jupyter import visualize_slices_with_profile_jupyter\n"
        "  vol = np.random.rand(64, 128, 128).astype(np.float32)\n"
        "  visualize_slices_with_profile_jupyter(\n"
        "      vol,\n"
        "      z_values=np.linspace(0, 31.5, 64),\n"
        "      z_label='Depth (µm)',\n"
        "  )\n"
    )
