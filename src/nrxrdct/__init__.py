"""
nrxrdct — Near-field / far-field X-ray diffraction computed tomography utilities.

Subpackages
-----------
azimuthal
    pyFAI-backed 1-D and 2-D (CAKE) azimuthal integration with outlier
    rejection and SLURM batch submission helpers.
fitting
    Single-peak fitting (:func:`~nrxrdct.fitting.fit_peak`) and
    NMF decomposition (:class:`~nrxrdct.fitting.HyperspectralNMF`).
fluo
    X-ray fluorescence line lookup, spectral decomposition, and sinogram
    assembly from HDF5 master files.
laue
    Full Laue diffraction simulation, orientation fitting, strain mapping,
    and interactive visualisation tools.
rietveld
    GSAS-II Rietveld refinement wrappers (:class:`~nrxrdct.rietveld.BaseRefinement`,
    :class:`~nrxrdct.rietveld.InstrumentCalibration`) and pre-built refinement
    dictionary templates.
visualization
    Array-agnostic interactive Jupyter widgets, notably
    :class:`~nrxrdct.visualization.StackViewer` for browsing 2-D/3-D NumPy
    arrays with live colormap, vmin/vmax, and normalization controls.
xrdct
    End-to-end powder XRD-CT pipeline: sinogram assembly, ASTRA reconstruction,
    per-voxel GSAS-II refinement via :class:`~nrxrdct.xrdct.ReconstructedVolume`,
    and napari / Jupyter visualisation.
"""

from .xrdct.parameters import Scan
from .xrdct.volume import ReconstructedVolume
from .xrdct.reconstruction import reconstruct_slice, HAS_GPU
from .azimuthal.integration import azimuthal_integration_1d, cake_integration
from .fitting.peakfit import fit_peak
from .rietveld.refinement import BaseRefinement

__all__ = [
    "Scan",
    "ReconstructedVolume",
    "reconstruct_slice",
    "HAS_GPU",
    "azimuthal_integration_1d",
    "cake_integration",
    "fit_peak",
    "BaseRefinement",
]
