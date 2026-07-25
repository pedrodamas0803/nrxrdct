"""
nrxrdct.texture.slurm_pole_figures
--------------------------------------
Tools for distributing pole-figure extraction (:mod:`nrxrdct.texture.odf`)
across SLURM HPC clusters.

Typical workflow
----------------
    from nrxrdct.texture.slurm_pole_figures import launch, monitor, merge, check, repair

    # 1. Submit extraction jobs
    result = launch(
        master_file = Path("master.h5"),
        output_file = Path("pole_figures.h5"),
        poni_file   = Path("calib.poni"),
        mask_file   = Path("mask.edf"),
        tth_range   = (4.0, 6.0),
        hkl_label   = "111",
        n_jobs      = 8,
        conda_env   = "nrxrdct",
    )

    # 2. Watch until done (blocking)
    monitor(result["slurm_ids"], result["tmp_dir"], watch=True)

    # 3. Assemble the output HDF5
    merge(tmp_dir=result["tmp_dir"], output_file=Path("pole_figures.h5"))

    # 4. Verify
    check(tmp_dir=result["tmp_dir"], output_file=Path("pole_figures.h5"), hkl_label="111")

    # 5. If anything is missing, resubmit
    repair(tmp_dir=result["tmp_dir"])
"""

from .check_output import check, repair  # noqa: F401
from .launch_jobs import launch  # noqa: F401
from .merge import merge  # noqa: F401
from .monitor import monitor  # noqa: F401

__all__ = ["launch", "merge", "check", "repair", "monitor"]