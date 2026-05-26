"""
nrxrdct.slurm_integration
--------------------------
Tools for distributing powder integration across SLURM HPC clusters.

Typical workflow
----------------
    from nrxrdct.slurm_integration import launch, monitor, merge, check, repair

    # 1. Submit integration jobs
    result = launch(
        master_file = Path("master.h5"),
        output_file = Path("output.h5"),
        poni_file   = Path("calib.poni"),
        mask_file   = Path("mask.edf"),
        n_jobs      = 8,
        conda_env   = "nrxrdct",
    )

    # 2. Watch until done (blocking)
    monitor(result["slurm_ids"], result["tmp_dir"], watch=True)

    # 3. Assemble the output HDF5
    merge(tmp_dir=result["tmp_dir"], output_file=Path("output.h5"))

    # 4. Verify
    check(tmp_dir=result["tmp_dir"], output_file=Path("output.h5"))

    # 5. If anything is missing, resubmit
    repair(tmp_dir=result["tmp_dir"], master_file=..., poni_file=..., mask_file=...)
"""

from .check_output import check, repair  # noqa: F401
from .launch_jobs import launch  # noqa: F401
from .merge import merge  # noqa: F401
from .monitor import monitor  # noqa: F401

__all__ = ["launch", "merge", "check", "repair", "monitor"]

# """
# nrxrdct.slurm_integration
# --------------------------
# Tools for distributing powder integration across SLURM HPC clusters.

# Python API
# ----------
#     from nrxrdct.slurm_integration import launch, check, repair

# CLI
# ---
#     nrxrdct-slurm launch --help
#     nrxrdct-slurm check  --help
# """

# from .check_output import check, rebuild, repair  # noqa: F401
# from .launch_jobs import launch  # noqa: F401
# from .monitor import monitor  # noqa: F401

# __all__ = ["launch", "check", "repair", "rebuild", "monitor"]
