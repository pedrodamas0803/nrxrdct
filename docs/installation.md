# Installation

## Requirements

- Python **3.11** or later
- A working [GSAS-II](https://gsas-ii.readthedocs.io/) installation (`GSASIIscriptable` must be importable)
- [ASTRA Toolbox](https://astra-toolbox.com/) (GPU reconstruction requires an NVIDIA GPU and CUDA)

## With `uv` (recommended)

[`uv`](https://docs.astral.sh/uv/) is a fast Python package and project manager. If you do not have it installed yet:

**Linux / macOS**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell)**

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Via pip (any platform)**

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

**Linux / macOS**

```bash
source .venv/bin/activate
```

**Windows (PowerShell)**

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

## Optional dependencies

### Laue extras (`orix`)

The `nrxrdct.laue` subpackage uses [orix](https://orix.readthedocs.io/) for IPF colour-key rendering and orientation symmetry reduction.  It is declared as an optional extra so it is not installed by default.

**With `uv`:**

```bash
uv sync --extra laue
```

**With `pip`:**

```bash
pip install "nrxrdct[laue]"
# or, inside an already-activated environment:
pip install "orix>=0.11"
```

---

### GSAS-II

GSAS-II is not on PyPI and must be installed separately.  The recommended method is **conda**:

```bash
conda install -c conda-forge gsas2pkg
```

This installs GSAS-II and registers `GSASIIscriptable` on the Python path automatically.

If you are using a `uv`-managed environment alongside a conda base, you can instead point Python at the GSAS-II source tree by adding it to `PYTHONPATH`:

```bash
# Clone the GSAS-II source (one-time setup)
git clone https://github.com/AdvancedPhotonSource/GSAS-II.git ~/gsas2

# Add to PYTHONPATH in your shell profile (or prepend it per-session)
export PYTHONPATH="$HOME/gsas2:$PYTHONPATH"
```

Verify the installation:

```python
import GSASIIscriptable as G2sc
print(G2sc.__file__)
```

!!! note
    GSAS-II requires a compiled binary (`constrDict.so` / `.pyd`) for some refinement operations.  The conda package builds this automatically; the git-clone approach may require running `python setup.py build_ext --inplace` inside the GSAS-II directory.

---

### ASTRA Toolbox

ASTRA provides GPU-accelerated (CUDA) and CPU tomographic reconstruction algorithms.

**With conda (recommended — includes CUDA support):**

```bash
conda install -c astra-toolbox astra-toolbox
```

**With pip (CPU-only build):**

```bash
pip install astra-toolbox
```

Verify GPU access:

```python
import astra
print(astra.get_gpu_info())
```

GPU algorithms (`SART_CUDA`, `SIRT3D_CUDA`, `CGLS3D_CUDA`) require an NVIDIA GPU with CUDA drivers installed.  `nrxrdct` falls back to CPU automatically when no GPU is found.

---

### Other optional packages

These are all available on PyPI and can be installed with `uv add` or `pip install`:

| Package | Install command | Used by |
|---|---|---|
| `fabio` | `pip install fabio` | Reading `.edf` / `.cbf` mask files in `azimuthal.integration` |
| `hdf5plugin` | `pip install hdf5plugin` | Reading Bitshuffle/LZ4-compressed HDF5 datasets in `xrdct` |
| `napari` | `pip install "napari[all]"` | Interactive 3-D volume viewer in `xrdct.visualization` |
| `ipywidgets` | `pip install ipywidgets` | Interactive Jupyter widgets in `laue.interactive` |

With `uv`:

```bash
uv add fabio hdf5plugin "napari[all]" ipywidgets
```

---

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
