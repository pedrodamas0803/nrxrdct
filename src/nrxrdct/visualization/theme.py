"""
Global light/dark theme switch for nrxrdct's visualization tools.

The interactive widgets in this package (:class:`~nrxrdct.visualization.StackViewer`,
:class:`~nrxrdct.visualization.ZProfilePlot`, the napari/Jupyter profile
viewers) default to a dark theme suited to on-screen exploration. Call
:func:`switch_to_light_mode` before generating figures intended for a
publication, report, or slide deck.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from . import _plot_helpers as _ph

_LIGHT_RC = {
    "figure.facecolor": "#ffffff",
    "axes.facecolor": "#ffffff",
    "savefig.facecolor": "#ffffff",
    "axes.edgecolor": "#d0d7de",
    "axes.labelcolor": "#1f2328",
    "xtick.color": "#57606a",
    "ytick.color": "#57606a",
    "text.color": "#1f2328",
    "grid.color": "#d0d7de",
    "legend.facecolor": "#ffffff",
    "legend.edgecolor": "#d0d7de",
}

_DARK_RC = {
    "figure.facecolor": "#0e1117",
    "axes.facecolor": "#161b22",
    "savefig.facecolor": "#0e1117",
    "axes.edgecolor": "#30363d",
    "axes.labelcolor": "#e6edf3",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "text.color": "#e6edf3",
    "grid.color": "#30363d",
    "legend.facecolor": "#161b22",
    "legend.edgecolor": "#30363d",
}


def switch_to_light_mode() -> None:
    """
    Switch matplotlib and nrxrdct's visualization widgets to a light theme.

    Updates ``matplotlib.pyplot.rcParams`` — so plain ``plt.subplots()``-based
    plots (e.g. :func:`~nrxrdct.visualization.plot_integrated_cake`,
    :func:`~nrxrdct.visualization.plot_labeled_image`, or your own notebook
    cells) render on a white background — and the internal palette used by
    :class:`~nrxrdct.visualization.StackViewer` and the napari/Jupyter
    profile viewers.

    Only figures created *after* calling this pick up the new theme; it does
    not repaint widgets that are already displayed.

    Example:
        >>> from nrxrdct.visualization import switch_to_light_mode, StackViewer
        >>> switch_to_light_mode()
        >>> viewer = StackViewer(volume)  # renders light, ready for a figure export
    """
    plt.rcParams.update(_LIGHT_RC)
    _ph.set_theme("light")


def switch_to_dark_mode() -> None:
    """
    Switch matplotlib and nrxrdct's visualization widgets back to the
    default dark theme used for interactive exploration.

    See :func:`switch_to_light_mode` for details on scope and timing.
    """
    plt.rcParams.update(_DARK_RC)
    _ph.set_theme("dark")