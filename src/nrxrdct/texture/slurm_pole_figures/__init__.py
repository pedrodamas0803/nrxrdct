"""
nrxrdct.texture.slurm_pole_figures
--------------------------------------
Tools for distributing pole-figure extraction (:mod:`nrxrdct.texture.odf`)
across SLURM HPC clusters, for one or more hkl rings at once — each worker
cake-integrates every frame once and extracts all requested rings from that
single cake, so adding more hkls does not multiply the (expensive)
integration cost.

Typical workflow
----------------
    from nrxrdct.texture.slurm_pole_figures import launch, monitor, merge, check, repair

    # 1. Submit extraction jobs for as many hkls as you want in one go
    result = launch(
        master_file = Path("master.h5"),
        output_file = Path("pole_figures.h5"),
        poni_file   = Path("calib.poni"),
        mask_file   = Path("mask.edf"),
        hkls        = {"111": (4.0, 6.0), "200": (7.0, 9.0), "220": (11.0, 13.0)},
        n_jobs      = 8,
        conda_env   = "nrxrdct",
    )

    # 2. Watch until done (blocking)
    monitor(result["slurm_ids"], result["tmp_dir"], watch=True)

    # 3. Assemble the output HDF5 (writes pole_figures/<hkl>/... for every hkl)
    merge(tmp_dir=result["tmp_dir"], output_file=Path("pole_figures.h5"))

    # 4. Verify (checks every hkl found in launch_meta.json by default)
    check(tmp_dir=result["tmp_dir"], output_file=Path("pole_figures.h5"))

    # 5. If anything is missing, resubmit
    repair(tmp_dir=result["tmp_dir"])
"""

from .check_output import check, repair  # noqa: F401
from .launch_jobs import launch  # noqa: F401
from .merge import merge  # noqa: F401
from .monitor import monitor  # noqa: F401

__all__ = ["launch", "merge", "check", "repair", "monitor"]