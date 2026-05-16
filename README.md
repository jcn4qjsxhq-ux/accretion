# Accretion Disk Fitting

Fitting simple accretion disk models to photometric / spectroscopic data.

This repository contains reusable Python functions, notebooks, and command-line
scripts for fitting accretion-disk models to time-series photometry and spectra.
It does not include private observational data, generated result tables, or
generated images. It does include a small demo photometry file so the workflow
can be run immediately. Add your own data locally using the formats described in
[docs/data_format.md](docs/data_format.md).

## Repository Layout

```text
.
├── src/accretion/              # Importable fitting, physics, loading, and plotting code
├── notebooks/                # Clean Ultimate workflow plus quickstart tutorial
├── scripts/                  # Command-line entry points
├── data/demo_data.txt        # Small public demo photometry table
├── data/raw/                 # Local input data; ignored by Git
├── data/processed/           # Local processed data; ignored by Git
├── results/generated/
│   ├── pictures/             # New generated PDF plots
│   └── datas/                # New generated CSV/TXT result tables
├── docs/                     # Data format and usage notes
└── _private/                 # Local-only material; ignored by Git
```

## Installation

Clone the repository, create an environment, and install the package in editable
mode:

```bash
git clone https://github.com/YOUR_USERNAME/accretion.git
cd accretion
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

On Windows, activate the environment with:

```bash
.venv\Scripts\activate
```

## Quick Start

Open the tutorial notebook:

```bash
jupyter notebook notebooks/quickstart.ipynb
```

Or run a basic photometric fit from the command line:

```bash
python scripts/run_fit.py \
  --data data/demo_data.txt \
  --data-kind legacy-photometry \
  --mode basic \
  --required-filters J,H,K \
  --fit-filters J,H,K \
  --lambda-reg 1 \
  --output results/generated/datas/basic_JHK
```

The command writes a saved result bundle:

```text
results/generated/datas/basic_JHK_daily_params.csv
results/generated/datas/basic_JHK_global_params.csv
results/generated/datas/basic_JHK_fit_info.csv
results/generated/datas/basic_JHK_daily_data.csv
```

Then create summary figures:

```bash
python scripts/make_figures.py \
  --results results/generated/datas/basic_JHK \
  --data data/demo_data.txt \
  --legacy-data \
  --output-dir results/generated/pictures
```

## Fitting Modes

The main high-level modes are available from `accretion`:

```python
from accretion import (
    basic_fitting,
    allpams_fitting,
    AR_fitting,
    rede_fitting,
    spectral_fitting,
    result_saver,
)
```

Common modes:

| Mode | Function | Purpose |
| --- | --- | --- |
| Basic photometric | `basic_fitting` | Fit daily `Mdot` and `Av` with fixed global disk parameters |
| Spectral | `spectral_fitting(..., rede=False)` | Fit spectra directly |
| All-parameters | `allpams_fitting` | Also fit selected global parameters |
| AR mode | `AR_fitting` | Fit with variable inner radius behavior |
| Red-excess | `rede_fitting` | Add a blackbody red-excess component |
| Spectral red-excess | `spectral_fitting(..., rede=True)` | Fit spectra with accretion plus red-excess component |

## Using Your Own Photometry

Your CSV should contain at least:

```text
JD, Date, Filter, Lambda, Mag, Magerr, ZP, Flux, Fluxerr
```

Example:

```python
import pandas as pd
from accretion import basic_fitting, result_saver

df = pd.read_csv("data/raw/my_photometry.csv")
results = basic_fitting(
    df,
    required_filters=("J", "H", "K"),
    fit_filters=("J", "H", "K"),
    lambda_reg=1,
)
result_saver(results, filepath="results/generated/datas/my_basic_fit")
```

See [docs/data_format.md](docs/data_format.md) for photometry and spectroscopy
column definitions. See [docs/capabilities.md](docs/capabilities.md) for a
longer guide to fitting modes, plotting tools, comparison workflows, and common
ways to combine them.

## Using Spectra

For IRTF-style spectra in the text-header format supported by the loaders:

```python
from accretion import load_irtf_sxd_spectra, load_irtf_lxd_spectra, spectral_fitting
import pandas as pd

sxd = load_irtf_sxd_spectra("data/raw/IRTF")
lxd = load_irtf_lxd_spectra("data/raw/IRTF")
spectra = pd.concat([sxd, lxd], ignore_index=True)

results = spectral_fitting(spectra, lambda_reg=1, rede=True)
```

For your own spectra, provide a CSV with `JD` or `JD_day`, `Lambda` or
`Lambda_m`, `Flux`, and `Fluxerr`.

## Reproducibility Notes

- This public repository intentionally excludes private data, old generated
  results, and old generated figures.
- New generated plots are PDFs in `results/generated/pictures/`; generated
  CSV/TXT tables are in `results/generated/datas/`.
- The public notebooks are `notebooks/Ultimate.ipynb` and
  `notebooks/quickstart.ipynb`.
- The importable source of truth is `src/accretion/`.

## Citation

If you use this code, cite the repository using the metadata in
[CITATION.cff](CITATION.cff). Update the GitHub URL, author details, and
publication information before making the repository public.

## License

This project is released under the MIT License. Check that all bundled data can
be redistributed before adding any data to a public fork.
