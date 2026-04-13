"""
Interactive polygonal ROI selection on a matplotlib image.

Works in both regular Python scripts and Jupyter notebooks.

Jupyter requirements
--------------------
    %matplotlib widget      # must use an interactive backend (ipympl)
    # pip install ipympl

Script requirements
-------------------
    Any interactive matplotlib backend (TkAgg, Qt5Agg, …).

Usage
-----
    from nrxrdct.roi import select_roi, select_roi_nb

    # Scripts / blocking:
    mask, verts = select_roi(image)

    # Jupyter (async — use top-level await or an async cell):
    mask, verts = await select_roi_nb(image)

    # or reuse an existing axes:
    fig, ax = plt.subplots()
    ax.imshow(other_image)
    mask, verts = await select_roi_nb(image, ax=ax)
"""

from __future__ import annotations

import asyncio
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path
from matplotlib.widgets import PolygonSelector


def select_roi(
    image: np.ndarray,
    ax: Optional[plt.Axes] = None,
    cmap: str = "gray",
    title: str = "Draw polygon — click to add vertices, press Enter to confirm",
    **imshow_kwargs,
) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """
    Display *image* and let the user draw a closed polygon interactively.

    Controls:
        - **Left click** — add a vertex.
        - **Right-click / backspace** — remove the last vertex.
        - **Enter** — confirm the polygon and close the figure.
        - **Escape** — cancel (returns an all-False mask).

    Args:
        image (np.ndarray): 2-D (or 3-D RGB/RGBA) array to display.
        ax (plt.Axes, optional): Existing ``Axes`` to draw into. A new figure
            is created when *None* (default).
        cmap (str): Colormap passed to ``imshow``. Ignored for RGB images.
        title (str): Title shown above the axes.
        **imshow_kwargs: Extra keyword arguments forwarded to ``ax.imshow``.

    Returns:
        mask (np.ndarray): Boolean array of shape ``image.shape[:2]``;
            ``True`` inside the selected polygon, ``False`` outside.
        vertices (list): Polygon vertices as (x, y) tuples in image (pixel)
            coordinates. Empty list if the selection was cancelled.
    """
    own_figure = ax is None
    if own_figure:
        fig, ax = plt.subplots(figsize=(16, 9))
    else:
        fig = ax.figure

    ax.imshow(image, cmap=cmap, **imshow_kwargs)
    ax.set_title(title)

    rows, cols = image.shape[:2]

    # State shared with callbacks -----------------------------------------
    _vertices: list[tuple[float, float]] = []
    _done = [False]

    def _onselect(verts: list[tuple[float, float]]) -> None:
        _vertices.clear()
        _vertices.extend(verts)
        _done[0] = True
        # Close the figure so the blocking loop exits (works in scripts).
        # In Jupyter the figure stays open but _done signals completion.
        if own_figure:
            plt.close(fig)

    selector = PolygonSelector(  # noqa: F841  (kept alive via closure)
        ax,
        _onselect,
        props=dict(color="red", linewidth=1.5),
        handle_props=dict(markersize=6),
    )

    # ESC key → cancel
    def _on_key(event) -> None:
        if event.key == "escape":
            _done[0] = True
            if own_figure:
                plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", _on_key)

    # --- Blocking strategy ------------------------------------------------
    # `plt.show(block=True)` works in scripts but not in Jupyter.
    # `plt.pause()` inside a loop works in *both* when using an interactive
    # backend (%matplotlib widget / ipympl in Jupyter).
    try:
        while not _done[0] and plt.fignum_exists(fig.number):
            plt.pause(0.05)
    except Exception:
        pass  # figure was closed externally

    # --- Build mask -------------------------------------------------------
    if not _vertices:
        return np.zeros((rows, cols), dtype=bool), []

    y_idx, x_idx = np.mgrid[:rows, :cols]
    points = np.column_stack([x_idx.ravel(), y_idx.ravel()])
    path = Path(_vertices)
    mask = path.contains_points(points).reshape(rows, cols)

    return mask, list(_vertices)


async def select_roi_nb(
    image: np.ndarray,
    ax: Optional[plt.Axes] = None,
    cmap: str = "gray",
    title: str = "Draw polygon — click to add vertices, press Enter to confirm",
    **imshow_kwargs,
) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """
    Jupyter-native version of :func:`select_roi`.

    Uses ``await asyncio.sleep()`` instead of a blocking ``plt.pause()`` loop,
    so it cooperates with Jupyter's running asyncio event loop.

    Requirements:
        - ``%matplotlib widget`` (ipympl) must be active in the notebook.
        - Call with ``await``: ``mask, verts = await select_roi_nb(image)``.

    Controls:
        - **Left click** — add a vertex.
        - **Right-click / backspace** — remove the last vertex.
        - **Enter** — confirm the polygon.
        - **Escape** — cancel (returns an all-False mask).

    Args:
        image (np.ndarray): 2-D (or 3-D RGB/RGBA) array to display.
        ax (plt.Axes, optional): Existing ``Axes`` to draw into. A new figure
            is created when *None* (default).
        cmap (str): Colormap passed to ``imshow``. Ignored for RGB images.
        title (str): Title shown above the axes.
        **imshow_kwargs: Extra keyword arguments forwarded to ``ax.imshow``.

    Returns:
        mask (np.ndarray): Boolean array of shape ``image.shape[:2]``;
            ``True`` inside the selected polygon, ``False`` outside.
        vertices (list): Polygon vertices as (x, y) tuples in image (pixel)
            coordinates. Empty list if the selection was cancelled.
    """
    own_figure = ax is None
    if own_figure:
        fig, ax = plt.subplots(figsize=(16, 9))
    else:
        fig = ax.figure

    ax.imshow(image, cmap=cmap, **imshow_kwargs)
    ax.set_title(title)

    rows, cols = image.shape[:2]

    _vertices: list[tuple[float, float]] = []
    _done = [False]

    def _onselect(verts: list[tuple[float, float]]) -> None:
        _vertices.clear()
        _vertices.extend(verts)
        _done[0] = True

    selector = PolygonSelector(  # noqa: F841
        ax,
        _onselect,
        props=dict(color="red", linewidth=1.5),
        handle_props=dict(markersize=6),
    )

    def _on_key(event) -> None:
        if event.key == "escape":
            _done[0] = True

    fig.canvas.mpl_connect("key_press_event", _on_key)

    plt.show()

    while not _done[0]:
        await asyncio.sleep(0.05)

    selector.disconnect_events()
    ax.set_title("ROI confirmed" if _vertices else "ROI cancelled")
    fig.canvas.draw_idle()

    if not _vertices:
        return np.zeros((rows, cols), dtype=bool), []

    y_idx, x_idx = np.mgrid[:rows, :cols]
    points = np.column_stack([x_idx.ravel(), y_idx.ravel()])
    path = Path(_vertices)
    mask = path.contains_points(points).reshape(rows, cols)

    return mask, list(_vertices)
