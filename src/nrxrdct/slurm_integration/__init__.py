"""
nrxrdct.slurm_integration
--------------------------
Tools for distributing powder integration across SLURM HPC clusters.

Python API
----------
    from nrxrdct.slurm_integration import launch, check, repair

CLI
---
    nrxrdct-slurm launch --help
    nrxrdct-slurm check  --help
"""

from .launch_jobs  import launch          # noqa: F401
from .check_output import check, repair   # noqa: F401

__all__ = ["launch", "check", "repair"]