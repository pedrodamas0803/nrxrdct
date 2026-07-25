"""
Texture tomography.

Reconstructs local crystallographic texture (orientation distribution) as a
function of position from the same XRD-CT diffraction images used by
:mod:`nrxrdct.xrdct`, rather than the scalar density/phase maps that pipeline
produces.

Covers pole-figure extraction (:mod:`nrxrdct.texture.odf`), ODF inversion via
WIMV (:mod:`nrxrdct.texture.odf_inversion`), and pole-figure plotting
(:mod:`nrxrdct.texture.texture_plotting`).
"""

from .odf import (
    assemble_pole_figure_data,
    assemble_pole_figure_sinogram,
    extract_ring_intensity,
    pole_figure_coordinates,
    sinogram_to_pole_figure,
)
from .odf_inversion import (
    compute_odf,
    load_pole_figures,
    misorientation_angle_deg,
    orientation_grid,
    recalculate_pole_figure,
)
from .texture_plotting import plot_pole_figure, plot_pole_figure_comparison

__all__ = [
    "extract_ring_intensity",
    "pole_figure_coordinates",
    "assemble_pole_figure_data",
    "assemble_pole_figure_sinogram",
    "sinogram_to_pole_figure",
    "orientation_grid",
    "misorientation_angle_deg",
    "compute_odf",
    "load_pole_figures",
    "recalculate_pole_figure",
    "plot_pole_figure",
    "plot_pole_figure_comparison",
]