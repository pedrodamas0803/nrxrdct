"""
Texture tomography.

Reconstructs local crystallographic texture (orientation distribution) as a
function of position from the same XRD-CT diffraction images used by
:mod:`nrxrdct.xrdct`, rather than the scalar density/phase maps that pipeline
produces.

Covers pole-figure extraction (:mod:`nrxrdct.texture.odf`), two ODF inversion
strategies — hard-binning WIMV (:mod:`nrxrdct.texture.odf_inversion`) and
smooth kernel-density fitting (:mod:`nrxrdct.texture.kernel_odf`) — and
pole-figure plotting (:mod:`nrxrdct.texture.texture_plotting`).
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
from .kernel_odf import (
    compute_odf_kernel,
    dlvp_kernel_so3,
    dlvp_pole_figure_kernel,
    kappa_from_halfwidth_deg,
    recalculate_pole_figure_kernel,
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
    "compute_odf_kernel",
    "recalculate_pole_figure_kernel",
    "dlvp_kernel_so3",
    "dlvp_pole_figure_kernel",
    "kappa_from_halfwidth_deg",
    "plot_pole_figure",
    "plot_pole_figure_comparison",
]