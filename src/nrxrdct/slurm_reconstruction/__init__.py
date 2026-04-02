"""
nrxrdct.slurm_reconstruction
-----------------------------
Tools for distributing tomographic reconstruction across SLURM HPC clusters.

Python API
----------
    from nrxrdct.slurm_reconstruction import build_sinogram, launch_recon

    # Step 1 (optional) — assemble sinogram from integrated HDF5
    build_sinogram(
        integrated_file = Path("output.h5"),
        sinogram_file   = Path("sinogram.h5"),
        n_rot           = 360,
        n_tth_angles    = 1000,
        n_lines         = 10,
    )

    # Step 2 — submit reconstruction jobs
    launch_recon(
        sinogram_file = Path("sinogram.h5"),
        output_file   = Path("reconstruction.h5"),
        n_jobs        = 8,
        algo          = "SART_CUDA",
        num_iter      = 200,
        gpu           = True,
        conda_env     = "nrxrdct",
    )

CLI
---
    nrxrdct-slurm-recon build  --help
    nrxrdct-slurm-recon launch --help
"""

from .launch_recon import build_sinogram, launch_recon  # noqa: F401

__all__ = ["build_sinogram", "launch_recon"]
