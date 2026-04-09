# SLURM Pipeline Example

This example walks through a complete HPC submission: distributed integration followed by distributed GPU reconstruction, using the `nrxrdct-slurm` and `nrxrdct-slurm-recon` CLIs (or their Python equivalents).

---

## Step 1 — Distributed integration

Split azimuthal integration across 8 SLURM nodes.

=== "Python API"

    ```python
    from pathlib import Path
    from nrxrdct.slurm_integration import launch, monitor

    slurm_ids = launch(
        master_file  = Path("/data/raw/sample_master.h5"),
        output_file  = Path("/data/processed/integrated.h5"),
        poni_file    = Path("/data/calib/detector.poni"),
        mask_file    = Path("/data/calib/mask.edf"),
        n_jobs       = 8,
        n_points     = 1000,
        n_workers    = 16,
        batch_size   = 32,
        method       = "filter",
        percentile   = (10, 90),
        partition    = "nice",
        time         = "04:00:00",
        mem          = "64G",
        cpus         = 16,
        conda_env    = "nrxrdct",
    )

    # Block until all integration jobs finish
    monitor(
        slurm_ids   = slurm_ids,
        output_file = Path("/data/processed/integrated.h5"),
        watch       = True,
        interval    = 60,
    )
    ```

=== "CLI"

    ```bash
    nrxrdct-slurm launch \
        --master-file  /data/raw/sample_master.h5 \
        --output-file  /data/processed/integrated.h5 \
        --poni-file    /data/calib/detector.poni \
        --mask-file    /data/calib/mask.edf \
        --n-jobs       8 \
        --n-points     1000 \
        --n-workers    16 \
        --method       filter \
        --percentile   10,90 \
        --partition    nice \
        --time         04:00:00 \
        --mem          64G \
        --cpus         16 \
        --conda-env    nrxrdct

    # Monitor (blocking)
    nrxrdct-slurm monitor \
        --slurm-ids   $(cat slurm_ids.txt) \
        --output-file /data/processed/integrated.h5 \
        --watch --interval 60
    ```

---

## Step 2 — Build sinogram

Assemble the sinogram from the integrated HDF5 before launching reconstruction.

=== "Python API"

    ```python
    from nrxrdct.slurm_reconstruction import build_sinogram

    build_sinogram(
        integrated_file = Path("/data/processed/integrated.h5"),
        sinogram_file   = Path("/data/processed/sinogram.h5"),
        n_rot           = 901,
        n_tth_angles    = 1000,
        n_lines         = 10,
    )
    ```

=== "CLI"

    ```bash
    nrxrdct-slurm-recon build \
        --integrated-file /data/processed/integrated.h5 \
        --sinogram-file   /data/processed/sinogram.h5 \
        --n-rot           901 \
        --n-tth-angles    1000 \
        --n-lines         10
    ```

---

## Step 3 — Distributed GPU reconstruction

Split the 2θ axis across 8 GPU jobs.

=== "Python API"

    ```python
    from nrxrdct.slurm_reconstruction import launch_recon

    recon_ids = launch_recon(
        sinogram_file = Path("/data/processed/sinogram.h5"),
        output_file   = Path("/data/processed/reconstruction.h5"),
        n_jobs        = 8,
        algo          = "SART_CUDA",
        num_iter      = 200,
        dty_step      = 1.0,
        partition     = "gpu",
        time          = "08:00:00",
        mem           = "64G",
        cpus          = 8,
        gpu           = True,
        conda_env     = "nrxrdct",
    )

    print("Reconstruction job IDs:", recon_ids)
    ```

=== "CLI"

    ```bash
    nrxrdct-slurm-recon launch \
        --sinogram-file  /data/processed/sinogram.h5 \
        --output-file    /data/processed/reconstruction.h5 \
        --n-jobs         8 \
        --algo           SART_CUDA \
        --num-iter       200 \
        --partition      gpu \
        --time           08:00:00 \
        --mem            64G \
        --cpus           8 \
        --gpu \
        --conda-env      nrxrdct
    ```

Sbatch scripts and logs are written to `<output_file_dir>/slurm_logs_recon/`.

---

## Step 4 — (Optional) Check and repair integration output

After integration jobs finish, verify every scan was written correctly before running reconstruction.

=== "Python API"

    ```python
    from nrxrdct.slurm_integration import check, repair

    result = check(output_file=Path("/data/processed/integrated.h5"))
    print("Missing scans:",   result["missing"])
    print("Corrupted scans:", result["corrupted"])

    # Resubmit only the failed scans
    if result["missing"] or result["corrupted"]:
        repair(
            output_file = Path("/data/processed/integrated.h5"),
            master_file = Path("/data/raw/sample_master.h5"),
            poni_file   = Path("/data/calib/detector.poni"),
            mask_file   = Path("/data/calib/mask.edf"),
            n_jobs      = 2,
            watch       = True,
            partition   = "nice",
            conda_env   = "nrxrdct",
        )
    ```

=== "CLI"

    ```bash
    # Check only
    nrxrdct-slurm check --output-file /data/processed/integrated.h5

    # Automatic repair (resubmits missing/corrupted scans)
    nrxrdct-slurm check \
        --output-file  /data/processed/integrated.h5 \
        --repair \
        --master-file  /data/raw/sample_master.h5 \
        --poni-file    /data/calib/detector.poni \
        --mask-file    /data/calib/mask.edf \
        --n-jobs       2 \
        --partition    nice \
        --conda-env    nrxrdct \
        --watch
    ```

---

## Full pipeline script

A minimal end-to-end script combining all steps:

    ```python
    from pathlib import Path
    from nrxrdct.slurm_integration import launch, monitor, check, repair
    from nrxrdct.slurm_reconstruction import build_sinogram, launch_recon

    DATA        = Path("/data/raw/sample_master.h5")
    INTEGRATED  = Path("/data/processed/integrated.h5")
    SINOGRAM    = Path("/data/processed/sinogram.h5")
    RECON       = Path("/data/processed/reconstruction.h5")
    PONI        = Path("/data/calib/detector.poni")
    MASK        = Path("/data/calib/mask.edf")

    # 1. Integrate
    ids = launch(
        master_file=DATA, output_file=INTEGRATED,
        poni_file=PONI, mask_file=MASK,
        n_jobs=8, n_points=1000, n_workers=16,
        method="filter", percentile=(10, 90),
        partition="nice", time="04:00:00", mem="64G", cpus=16,
        conda_env="nrxrdct",
    )
    monitor(slurm_ids=ids, output_file=INTEGRATED, watch=True)

    # 2. Check and repair
    result = check(output_file=INTEGRATED)
    if result["missing"] or result["corrupted"]:
        repair(output_file=INTEGRATED, master_file=DATA,
            poni_file=PONI, mask_file=MASK,
            n_jobs=2, watch=True, conda_env="nrxrdct")

    # 3. Build sinogram
    build_sinogram(integrated_file=INTEGRATED, sinogram_file=SINOGRAM,
                n_rot=901, n_tth_angles=1000, n_lines=10)

    # 4. Reconstruct
    recon_ids = launch_recon(
        sinogram_file=SINOGRAM, output_file=RECON,
        n_jobs=8, algo="SART_CUDA", num_iter=200,
        partition="gpu", time="08:00:00", mem="64G", cpus=8,
        gpu=True, conda_env="nrxrdct",
    )
    print("Reconstruction submitted:", recon_ids)
    ```
