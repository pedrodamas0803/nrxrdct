"""
General-purpose interactive visualization widgets for NumPy arrays.

Array-agnostic Jupyter/matplotlib tools usable anywhere in the codebase:
scrollable stack/slice viewers, napari volume viewers, ROI selection, and
simple static image plots. Domain-specific plotting (laue detector
geometry, shadow4 beam optics, per-crystal orientation UIs, etc.) stays
in its owning subpackage.
"""

from .image_plots import plot_integrated_cake, plot_labeled_image
from .napari_views import ZProfilePlot, visualize_slices, visualize_slices_with_profile, visualize_volume
from .roi_selection import select_roi, select_roi_nb
from .stack_viewer import StackViewer, visualize_slices_with_profile_jupyter
from .theme import switch_to_dark_mode, switch_to_light_mode

__all__ = [
    "StackViewer",
    "visualize_volume",
    "visualize_slices",
    "visualize_slices_with_profile",
    "visualize_slices_with_profile_jupyter",
    "ZProfilePlot",
    "select_roi",
    "select_roi_nb",
    "plot_integrated_cake",
    "plot_labeled_image",
    "switch_to_light_mode",
    "switch_to_dark_mode",
]