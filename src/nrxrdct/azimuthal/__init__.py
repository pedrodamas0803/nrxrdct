"""
Azimuthal integration routines for XRD-CT detector images.

Wraps pyFAI's AzimuthalIntegrator for 1-D and 2-D (CAKE) integration with
outlier-rejection strategies and a parallelised batch pipeline.
"""

from .integration import (
    azimuthal_integration_1d,
    azimuthal_integration_1d_filter,
    azimuthal_integration_1d_sigma_clip,
    cake_integration,
    integrate_powder_parallel,
)

__all__ = [
    "azimuthal_integration_1d",
    "azimuthal_integration_1d_filter",
    "azimuthal_integration_1d_sigma_clip",
    "cake_integration",
    "integrate_powder_parallel",
]
