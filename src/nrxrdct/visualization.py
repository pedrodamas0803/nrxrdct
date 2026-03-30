"""
Interactive 3D volume visualization using napari.
"""

import numpy as np
import napari # type: ignore
from typing import Optional
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from typing import Optional


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

    Parameters
    ----------
    volume : np.ndarray
        A 3D array of shape (Z, Y, X) to visualize.
    name : str
        Label shown in the napari layer list. Default: "Volume".
    colormap : str
        Colormap name (e.g. "grays", "turbo", "magma", "green").
        See napari.utils.colormaps.AVAILABLE_COLORMAPS for the full list.
    rendering : str
        Volume rendering mode. One of:
          - "mip"       Max Intensity Projection (good for bright spots)
          - "minip"     Min Intensity Projection
          - "translucent"  Semi-transparent full volume
          - "iso"       Isosurface
          - "attenuated_mip"  MIP with depth attenuation
        Default: "mip".
    contrast_limits : tuple of (float, float), optional
        (low, high) display range. Defaults to the data min/max.
    gamma : float
        Gamma correction applied to the colormap. Default: 1.0.
    opacity : float
        Layer opacity between 0 (transparent) and 1 (opaque). Default: 1.0.
    scale : tuple of float, optional
        Physical voxel spacing (z_scale, y_scale, x_scale).
        Useful when voxels are anisotropic (e.g. (4.0, 1.0, 1.0)).
        Default: None (isotropic, all ones).
    add_axes : bool
        Whether to display the 3D axis widget. Default: True.

    Returns
    -------
    napari.Viewer
        The napari viewer instance (keep a reference to prevent GC).

    Examples
    --------
    >>> import numpy as np
    >>> from visualize_volume_napari import visualize_volume

    # Synthetic Gaussian blob
    >>> vol = np.random.rand(64, 128, 128).astype(np.float32)
    >>> viewer = visualize_volume(vol, colormap="turbo", rendering="mip")

    # Real microscopy-like data with anisotropic voxels
    >>> viewer = visualize_volume(
    ...     vol,
    ...     name="Confocal stack",
    ...     colormap="green",
    ...     rendering="translucent",
    ...     scale=(4.0, 1.0, 1.0),  # z is 4x coarser than xy
    ...     contrast_limits=(0.1, 0.9),
    ... )

    # Start napari event loop (required in scripts; not needed in Jupyter)
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

    Parameters
    ----------
    volume : np.ndarray
        A 3-D array of shape (Z, Y, X).  Each slice shown is volume[z, :, :].
    name : str
        Label shown in the napari layer list.  Default: "Volume".
    colormap : str
        Napari colormap name (e.g. "grays", "turbo", "magma", "green").
        Full list: napari.utils.colormaps.AVAILABLE_COLORMAPS.
    contrast_limits : tuple of (float, float), optional
        (low, high) display range.  Defaults to data min / max.
    gamma : float
        Gamma correction on the colormap.  Default: 1.0.
    opacity : float
        Layer opacity between 0 and 1.  Default: 1.0.
    scale : tuple of (float, float, float), optional
        Physical voxel spacing (z_scale, y_scale, x_scale).
        Example: (4.0, 1.0, 1.0) for anisotropic confocal data.
    initial_slice : int, optional
        Index along axis-0 to show on first open.
        Defaults to the middle slice.

    Returns
    -------
    napari.Viewer
        The napari viewer instance (keep a reference to prevent garbage
        collection while the window is open).

    Examples
    --------
    >>> import numpy as np
    >>> from visualize_slices_napari import visualize_slices

    # Synthetic test volume
    >>> vol = np.random.rand(64, 128, 128).astype(np.float32)
    >>> viewer = visualize_slices(vol, colormap="turbo")

    # Real microscopy stack with anisotropic voxels
    >>> viewer = visualize_slices(
    ...     vol,
    ...     name="Confocal stack",
    ...     colormap="green",
    ...     scale=(4.0, 1.0, 1.0),
    ...     contrast_limits=(0.05, 0.95),
    ...     initial_slice=10,
    ... )

    # Start the event loop (required in scripts; not needed in Jupyter)
    >>> napari.run()
    """
    if volume.ndim != 3:
        raise ValueError(
            f"Expected a 3-D array, got shape {volume.shape}."
        )

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
        Parameters
        ----------
        n_slices : int
            Total number of slices along axis-0 of the volume.
        volume_name : str, optional
            Used as part of the matplotlib window title (default ``"Volume"``).
        z_values : np.ndarray or None, optional
            Physical values for the Z axis (e.g. depth in mm or 2θ in degrees).
            Defaults to integer indices ``0 … n_slices-1``.
        z_label : str, optional
            X-axis label for the profile plot (default ``"Z  (axis-0 index)"``).
        """
        self.n_slices = n_slices
        self.z_axis = np.asarray(z_values) if z_values is not None else np.arange(n_slices)
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
            [], [],
            color="#58a6ff",
            linewidth=1.4,
            solid_capstyle="round",
        )
        # Vertical marker for the current slice
        self.vline = ax.axvline(x=0, color="#f0883e", linewidth=1.0, linestyle="--", alpha=0.8)
 
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

        Parameters
        ----------
        y : int
            Row index of the selected pixel.
        x : int
            Column index of the selected pixel.
        profile : np.ndarray
            1-D intensity profile along axis-0, length ``n_slices``.
        current_z_idx : int
            Current slider position (axis-0 index) used to place the vertical
            marker line.
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

        Parameters
        ----------
        z_idx : int
            Current axis-0 slice index.
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
 
    Parameters
    ----------
    volume : np.ndarray
        3-D array of shape (Z, Y, X).
    name : str
        Layer name shown in the napari layer list.
    colormap : str
        Napari colormap (e.g. "grays", "turbo", "magma").
    contrast_limits : (float, float), optional
        Display range. Defaults to data min/max.
    gamma : float
        Gamma correction for the colormap.
    opacity : float
        Layer opacity [0, 1].
    scale : (float, float, float), optional
        Anisotropic voxel spacing (z, y, x).
    initial_slice : int, optional
        Axis-0 index to display on open. Defaults to middle slice.
    z_values : array-like, optional
        Physical values to use on the X axis of the Z-profile plot
        (e.g. depths in mm, timestamps, wavelengths).  Must have exactly
        ``volume.shape[0]`` elements.  Defaults to integer slice indices.
    z_label : str, optional
        Label for the X axis of the Z-profile plot.
        Default: "Z  (axis-0 index)".
 
    Returns
    -------
    napari.Viewer
 
    Notes
    -----
    * Works with matplotlib's interactive backend (Qt / Tk / Wx).
    * Call ``napari.run()`` at the end of a script to start the event loop.
 
    Examples
    --------
    >>> import numpy as np
    >>> from visualize_slices_with_profile_napari import visualize_slices_with_profile
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
 
        print(f"[Z-profile] pixel (y={y_idx}, x={x_idx})  "
              f"min={profile.min():.4f}  max={profile.max():.4f}  "
              f"mean={profile.mean():.4f}")
 
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



