"""
Backward-compatible re-exports.

These tools moved to :mod:`nrxrdct.visualization` since they are generic
array/image viewers with no XRD-CT-specific logic. Import from there in new
code — this module is kept so existing ``from nrxrdct.xrdct.visualization
import ...`` statements keep working.
"""

from __future__ import annotations

from ..visualization._plot_helpers import draw_phase_ticks as _draw_phase_ticks  # noqa: F401
from ..visualization import (
    ZProfilePlot,
    plot_integrated_cake,
    select_roi,
    select_roi_nb,
    visualize_slices,
    visualize_slices_with_profile,
    visualize_slices_with_profile_jupyter,
    visualize_volume,
)

__all__ = [
    "visualize_volume",
    "visualize_slices",
    "ZProfilePlot",
    "visualize_slices_with_profile",
    "visualize_slices_with_profile_jupyter",
    "select_roi",
    "select_roi_nb",
    "plot_integrated_cake",
]