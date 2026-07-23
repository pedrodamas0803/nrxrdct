"""
General-purpose interactive visualization widgets for NumPy arrays.

Unlike :mod:`nrxrdct.xrdct.visualization` (napari/XRD-CT-specific tools),
this subpackage holds array-agnostic Jupyter widgets usable anywhere in the
codebase.
"""

from .stack_viewer import StackViewer

__all__ = ["StackViewer"]