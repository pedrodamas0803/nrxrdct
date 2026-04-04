# SLURM / HPC Integration

For large datasets, `nrxrdct` provides a SLURM pipeline that splits integration work across cluster nodes. All functionality is exposed through a Python API and the `nrxrdct-slurm` CLI.

## Submitting jobs

=== "Python API"

    ```python
    from pathlib import Path
    from nrxrdct.slurm_integration import launch

    slurm_ids = launch(
        master_file  = Path("/data/raw/sample_master.h5"),
        output_file  = Path("/data/processed/integrated.h5"),
        poni_file    = Path("/data/calib/detector.poni"),
        mask_file    = Path("/data/calib/mask.edf"),
        n_jobs       = 8,          # number of SLURM jobs
        n_points     = 1000,       # radial bins
        n_workers    = 16,         # integration threads per job
        batch_size   = 32,         # frames streamed per batch (RAM control)
        method       = "filter",   # "standard" | "filter" | "sigma_clip"
        percentile   = (10, 90),   # used with method="filter"
        partition    = "nice",
        time         = "04:00:00",
        mem          = "64G",
        cpus         = 16,
        conda_env    = "nrxrdct",
    )

    print("Submitted job IDs:", slurm_ids)
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
    ```

Sbatch scripts and logs are written to `<output_file_dir>/slurm_logs/`.

---

## Monitoring progress

=== "Python API"

    ```python
    from nrxrdct.slurm_integration import monitor

    # Single snapshot (non-blocking)
    result = monitor(
        slurm_ids   = slurm_ids,
        output_file = Path("/data/processed/integrated.h5"),
    )

    # Block until all jobs finish, polling every 60 s
    result = monitor(
        slurm_ids   = slurm_ids,
        output_file = Path("/data/processed/integrated.h5"),
        watch       = True,
        interval    = 60,
    )
    ```

=== "CLI"

    ```bash
    # Single snapshot
    nrxrdct-slurm monitor \
        --slurm-ids   12345,12346,12347 \
        --output-file /data/processed/integrated.h5

    # Blocking watch
    nrxrdct-slurm monitor \
        --slurm-ids   12345,12346,12347 \
        --output-file /data/processed/integrated.h5 \
        --watch --interval 60
    ```

The monitor prints a status table:

```text
────────────────────────────────────────────────────────
  nrxrdct SLURM monitor   elapsed: 0:42:17
────────────────────────────────────────────────────────
  Jobs total   : 8
  ⏳ Pending   : 0
  ▶  Running   : 6
  ✓  Completed : 2
  ✗  Failed    : 0
────────────────────────────────────────────────────────
  Scans  [████████████░░░░░░░░░░░░░░░░░░]  421/901  (46.7%)
  Rate   601.4 scans/hr
  ETA    0:47:53
────────────────────────────────────────────────────────
```

---

## Checking and repairing output

After all jobs finish, verify that every scan was written correctly.

=== "Python API"

    ```python
    from nrxrdct.slurm_integration import check, repair

    # Check only — report missing and corrupted scans
    result = check(output_file=Path("/data/processed/integrated.h5"))
    # result["missing"]   → list of scan indices not written
    # result["corrupted"] → list of scan indices with truncated/corrupt datasets

    # Automatic repair — delete corrupted entries and resubmit SLURM jobs
    repair_ids = repair(
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

    # Check with manual resubmit hints
    nrxrdct-slurm check \
        --output-file /data/processed/integrated.h5 \
        --resubmit

    # Automatic repair
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

## Rebuilding a corrupted HDF5

If the output HDF5 is deeply corrupted (damaged B-tree), use `rebuild` to salvage all readable scans into a fresh file:

```python
from nrxrdct.slurm_integration import rebuild

rebuild(
    output_file = Path("/data/processed/integrated.h5"),
    master_file = Path("/data/raw/sample_master.h5"),
    poni_file   = Path("/data/calib/detector.poni"),
    mask_file   = Path("/data/calib/mask.edf"),
    n_jobs      = 4,
    watch       = True,
    conda_env   = "nrxrdct",
)
# The original file is renamed to integrated.bak.h5
# The rebuilt file takes its place at the original path
```

---

## SLURM reconstruction

Large-scale tomographic reconstruction is also supported via `nrxrdct-slurm-recon`. See the [API reference](../api/slurm_reconstruction.md) for details.
