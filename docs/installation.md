# Installation

## Requirements

- Python **3.11** or later
- A working [GSAS-II](https://gsas-ii.readthedocs.io/) installation (`GSASIIscriptable` must be importable)
- [ASTRA Toolbox](https://astra-toolbox.com/) (GPU reconstruction requires an NVIDIA GPU and CUDA)

## With `uv` (recommended)

[`uv`](https://docs.astral.sh/uv/) is a fast Python package and project manager. If you do not have it installed yet:

=== "Linux / macOS"

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

=== "Windows (PowerShell)"

    ```powershell
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```

=== "Via pip (any platform)"

    ```bash
    pip install uv
    ```

After installing, restart your terminal so the `uv` command is on your `PATH`.

Then clone and install the project:

```bash
git clone <repo-url>
cd nrxrdct
uv sync
```

`uv sync` creates an isolated virtual environment in `.venv/`, installs all declared dependencies, and installs `nrxrdct` in editable mode — no separate `pip install -e .` step is needed.

To activate the environment manually:

=== "Linux / macOS"

    ```bash
    source .venv/bin/activate
    ```

=== "Windows (PowerShell)"

    ```powershell
    .venv\Scripts\Activate.ps1
    ```

Alternatively, prefix any command with `uv run` to execute it inside the managed environment without activating it:

```bash
uv run python my_script.py
uv run jupyter lab
```

### Registering a Jupyter kernel

On shared Jupyter servers such as the ESRF Jupyter-SLURM portal, you need to register the environment as a named kernel so it appears in the launcher.

```bash
# Install ipykernel into the uv-managed environment
uv add ipykernel

# Register the kernel — the --name must be unique on the server
uv run python -m ipykernel install --user --name nrxrdct --display-name "nrxrdct"
```

After refreshing the JupyterLab page the kernel **nrxrdct** will appear in the kernel selector.

!!! tip
    If you are on a SLURM-based Jupyter server (e.g. ESRF's [jupyter-slurm](https://jupyter-slurm.esrf.fr)), the kernel registration only needs to be done once per user account — the `--user` flag writes to `~/.local/share/jupyter/kernels/` which is shared across all sessions.

To remove the kernel later:

```bash
jupyter kernelspec remove nrxrdct
```

## With `pip`

```bash
git clone <repo-url>
cd nrxrdct
pip install -e .
```

### Registering a Jupyter kernel

```bash
pip install ipykernel
python -m ipykernel install --user --name nrxrdct --display-name "nrxrdct"
```

After refreshing the JupyterLab page the kernel **nrxrdct** will appear in the kernel selector.

!!! tip
    Run these commands with the virtual environment that contains `nrxrdct` already activated, so the registered kernel points to the correct Python interpreter.

## Building the documentation locally

Install the docs extras and serve:

```bash
uv sync --extra docs
uv run mkdocs serve
```

Then open <http://127.0.0.1:8000>.

---

## Dependencies

### Declared (`pyproject.toml`)

| Package | Purpose |
|---|---|
| `numpy >= 2.4` | Array operations throughout |
| `scipy >= 1.17` | Median filter, Gaussian filter |
| `h5py >= 3.16` | HDF5 file I/O |
| `matplotlib >= 3.10` | Plotting |
| `pyFAI >= 2026.2` | Azimuthal integration |
| `scikit-image >= 0.26` | Block reduce for XRF binning |
| `scikit-learn >= 1.8` | NMF decomposition |
| `pandas >= 3.0` | Peak table output |
| `xraylib >= 4.2` | XRF emission line energies |
| `xrayutilities >= 1.7` | Powder pattern simulation and peak listing |

### Additional runtime dependencies

These are used by specific modules but must be installed separately:

| Package | Module | Purpose |
|---|---|---|
| `astra-toolbox` | `reconstruction` | Tomographic reconstruction (GPU + CPU) |
| `GSAS-II` | `reconstruction`, `refinement` | Rietveld refinement scripting |
| `fabio` | `integration` | Reading mask files |
| `hdf5plugin` | `reconstruction` | Compressed HDF5 dataset support |
| `napari` | `visualization` | Interactive 3-D volume viewer |
| `pybaselines` | `utils` | XRD baseline fitting |
| `tqdm` | `integration`, `fluorescence`, `nmf` | Progress bars |
