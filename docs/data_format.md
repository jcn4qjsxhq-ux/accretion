# Data Format

The fitting functions use `pandas.DataFrame` inputs. You can load your own CSV
with `pandas.read_csv`, or use `load_database_new` if your local file follows
the legacy whitespace photometry table format used during development.

The repository includes `data/demo_data.txt`, a small legacy-format photometry
table that can be used to test the loaders and basic fitting workflow.

## Photometry

Photometric fitting expects one row per measurement.

Required columns:

| Column | Meaning |
| --- | --- |
| `JD` | Julian Date |
| `Date` | Calendar date or date-like label |
| `Filter` | Filter name, for example `J`, `H`, `K`, `W1` |
| `Lambda` | Effective wavelength in microns |
| `Mag` | Magnitude |
| `Magerr` | Magnitude uncertainty |
| `ZP` | Flux zero point, in the same flux units used for `Flux` |
| `Flux` | Observed flux |
| `Fluxerr` | Flux uncertainty |

Optional columns:

| Column | Meaning |
| --- | --- |
| `Flag` | Quality flag |
| `Interpolated` | `yes`, `no`, or another provenance label |
| `Mag_re`, `Magerr_re` | Reddening-corrected magnitude and uncertainty |
| `Flux_re`, `Fluxerr_re` | Reddening-corrected flux and uncertainty |

The loader and daily averaging helpers create `JD_day`, which is used to group
measurements from the same observing day.

## Spectroscopy

Spectral fitting expects one row per wavelength sample.

Required columns:

| Column | Meaning |
| --- | --- |
| `JD` or `JD_day` | Julian Date or day grouping key |
| `Lambda` or `Lambda_m` | Wavelength in microns (`Lambda`) or meters (`Lambda_m`) |
| `Flux` | Flux, usually Jy for the current model helpers |
| `Fluxerr` | Flux uncertainty |

Recommended columns:

| Column | Meaning |
| --- | --- |
| `Date` or `DateCode` | Used to combine same-night spectra |
| `Filter` | Spectral mode label such as `SXD` or `LXD` |
| `Frequency` | Frequency in Hz; computed if absent |
| `Instrument`, `Mode`, `Filename` | Provenance fields |

## Built-in Loaders

```python
from accretion import load_database_new, load_irtf_sxd_spectra, load_irtf_lxd_spectra

photometry = load_database_new("data/raw/my_legacy_photometry.txt")
sxd = load_irtf_sxd_spectra("data/raw/IRTF")
lxd = load_irtf_lxd_spectra("data/raw/IRTF")
```

For a normal CSV, use:

```python
import pandas as pd

df = pd.read_csv("data/raw/my_photometry.csv")
```

For the included demo:

```python
from accretion import load_database_new

df = load_database_new("data/demo_data.txt")
```
