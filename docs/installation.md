# Installation

## Requirements

- Python **3.11** or later
- A working [GSAS-II](https://gsas-ii.readthedocs.io/) installation (`GSASIIscriptable` must be importable)
- [ASTRA Toolbox](https://astra-toolbox.com/) (GPU reconstruction requires an NVIDIA GPU and CUDA)

## With `uv` (recommended)

```bash
git clone <repo-url>
cd nrxrdct
uv sync
```

## With `pip`

```bash
git clone <repo-url>
cd nrxrdct
pip install -e .
```

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
