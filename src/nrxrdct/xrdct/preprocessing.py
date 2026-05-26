"""
Preprocessing routines for XRD-CT detector images.

Zinger removal has moved to :mod:`nrxrdct.utils`.  This module re-exports the
public API for backwards compatibility.
"""

from nrxrdct.utils import NTHREAD, dezinger, zinger_remove

__all__ = ["NTHREAD", "zinger_remove", "dezinger"]
