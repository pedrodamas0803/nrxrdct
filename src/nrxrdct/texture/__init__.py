"""
Texture tomography.

Reconstructs local crystallographic texture (orientation distribution) as a
function of position from the same XRD-CT diffraction images used by
:mod:`nrxrdct.xrdct`, rather than the scalar density/phase maps that pipeline
produces.

Currently covers pole-figure extraction (:mod:`nrxrdct.texture.odf`); ODF
inversion from pole figures is not yet implemented.
"""

from .odf import (
    assemble_pole_figure_data,
    assemble_pole_figure_sinogram,
    extract_ring_intensity,
    pole_figure_coordinates,
)

__all__ = [
    "extract_ring_intensity",
    "pole_figure_coordinates",
    "assemble_pole_figure_data",
    "assemble_pole_figure_sinogram",
]