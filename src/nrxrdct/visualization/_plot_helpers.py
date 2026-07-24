"""
Shared theme constants and helpers used across the visualization submodule.

Colour constants (``BG``, ``PANEL``, ``FG``, ``MUTED``, ``BORDER``, ``ACCENT``,
``WARN``, ``PHASE_COLORS``) are mutated in place by :func:`set_theme` — other
modules in this package do ``from . import _plot_helpers as _ph`` and read
``_ph.PANEL`` etc. at draw time (rather than ``from ._plot_helpers import
PANEL``) so that widgets built after a theme switch pick up the new palette.
Use :func:`nrxrdct.visualization.switch_to_light_mode` /
:func:`~nrxrdct.visualization.switch_to_dark_mode` to switch themes.
"""

from __future__ import annotations

from typing import Dict, List, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_DARK: Dict[str, object] = {
    "BG": "#0e1117",
    "PANEL": "#161b22",
    "FG": "#e6edf3",
    "MUTED": "#8b949e",
    "BORDER": "#30363d",
    "ACCENT": "#58a6ff",
    "WARN": "#f0883e",
    "PHASE_COLORS": [
        "#ff7b72", "#7ee787", "#d2a8ff", "#ffa657",
        "#79c0ff", "#f2cc60", "#ff9bce", "#56d364",
    ],
}

_LIGHT: Dict[str, object] = {
    "BG": "#ffffff",
    "PANEL": "#f6f8fa",
    "FG": "#1f2328",
    "MUTED": "#57606a",
    "BORDER": "#d0d7de",
    "ACCENT": "#0969da",
    "WARN": "#bc4c00",
    "PHASE_COLORS": [
        "#d73a49", "#22863a", "#6f42c1", "#e36209",
        "#005cc5", "#b08800", "#ea4aaa", "#0598bc",
    ],
}

_THEME = "dark"
BG, PANEL, FG, MUTED, BORDER, ACCENT, WARN, PHASE_COLORS = (
    _DARK["BG"], _DARK["PANEL"], _DARK["FG"], _DARK["MUTED"],
    _DARK["BORDER"], _DARK["ACCENT"], _DARK["WARN"], _DARK["PHASE_COLORS"],
)


def set_theme(name: str) -> None:
    """
    Switch the palette used by *newly created* visualization widgets.

    Args:
        name ({"light", "dark"}): Theme to activate.
    """
    global _THEME, BG, PANEL, FG, MUTED, BORDER, ACCENT, WARN, PHASE_COLORS
    if name not in ("light", "dark"):
        raise ValueError(f"theme must be 'light' or 'dark', got {name!r}.")
    palette = _LIGHT if name == "light" else _DARK
    _THEME = name
    BG, PANEL, FG, MUTED, BORDER, ACCENT, WARN, PHASE_COLORS = (
        palette["BG"], palette["PANEL"], palette["FG"], palette["MUTED"],
        palette["BORDER"], palette["ACCENT"], palette["WARN"], palette["PHASE_COLORS"],
    )


def current_theme() -> str:
    """Return the currently active theme name (``"light"`` or ``"dark"``)."""
    return _THEME


def draw_phase_ticks(
    ax: plt.Axes,
    phases: Union[Dict[str, List[float]], Dict[str, pd.DataFrame]],
    z_axis: np.ndarray,
):
    """
    Populate a dedicated axes with phase/marker tick marks.

    Each entry occupies one row. Ticks are vertical lines drawn at the
    supplied positions; names appear as y-tick labels.

    Args:
        ax: The matplotlib axes to draw into (should be below the profile axes
            and share its x-axis).
        phases: Either a mapping of name -> list of positions (in the same
            units as *z_axis*), or a ``dict[str, pd.DataFrame]`` (e.g. the
            output of ``get_powder_xrd_peaks``) — in which case the ``tth``
            (or ``energy_keV``) column is used as the positions.
        z_axis: Physical x-coordinates used for the profile plot.

    Returns:
        The vertical marker line (``Line2D``) tracking the current slice,
        so the caller can keep moving it in sync with the slider.
    """
    first = next(iter(phases.values()), None)
    if isinstance(first, pd.DataFrame):
        col = "tth" if "tth" in first.columns else "energy_keV"
        phases = {name: df[col].tolist() for name, df in phases.items()}

    n_phases = len(phases)
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)

    ax.set_ylim(-0.5, n_phases - 0.5)
    ax.set_xlim(z_axis[0], z_axis[-1])
    ax.tick_params(colors=MUTED, labelsize=7)
    ax.tick_params(axis="y", length=0)

    for i, (_, peaks) in enumerate(phases.items()):
        color = PHASE_COLORS[i % len(PHASE_COLORS)]
        ax.vlines(peaks, i - 0.25, i + 0.25, colors=color, linewidth=1.2)

    ax.set_yticks(range(n_phases))
    ax.set_yticklabels(list(phases.keys()), color=MUTED, fontsize=7)

    vline = ax.axvline(x=z_axis[0], color=WARN, linewidth=1.0, linestyle="--", alpha=0.85)
    return vline