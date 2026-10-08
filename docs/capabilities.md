# Capabilities

This project is centered on fitting simple accretion-disk models to photometric
and spectroscopic observations. The notebook `notebooks/Ultimate.ipynb` shows
the full working style, while `notebooks/quickstart.ipynb` is the smallest
runnable example.

## 1. Data Loading And Daily Grouping

The loaders convert observations into `pandas.DataFrame` objects with consistent
column names. Photometric rows are grouped by observing day with `JD_day`, so
multiple filters from the same night can be fitted together.

Useful functions:

- `load_database_new`: reads the legacy whitespace photometry table format.
- `load_irtf_sxd_spectra` and `load_irtf_lxd_spectra`: read IRTF-style spectral text files.
- `average_daily_measurements`: averages repeated same-day, same-filter measurements.
- `get_daily_data`: selects only days containing the required filter set.

## 2. Basic Photometric Fitting

`basic_fitting` is the simplest science workflow. It fits one accretion rate
(`Mdot`) and one extinction (`Av`) per observing day, while keeping the global
stellar/disk parameters fixed to defaults.

Typical use:

```python
results = basic_fitting(
    df,
    required_filters=("J", "H", "K"),
    fit_filters=("J", "H", "K"),
    lambda_reg=1,
)
```

Use this mode when you want a stable first pass through a photometric data set.
The `required_filters` decide which days are included; the `fit_filters` decide
which filters contribute to the objective function.

To evaluate the model at exact parameter values without fitting, pass
`initial_params`. The same fixed `Mdot` and `Av` are used for every selected
day, and the result records zero optimizer evaluations:

```python
from accretion import M_sun, year

results = basic_fitting(
    df,
    initial_params={
        "Mdot": 1e-4 * M_sun / year,  # kg/s
        "Av": 15.0,
    },
)
```

## 3. Regularization

Most fitting modes accept `lambda_reg`. This penalizes rapid day-to-day changes
in selected parameters, usually `logMdot` and `Av`. A value of `0` gives the
least constrained fit. Larger values produce smoother parameter evolution.

This is useful when the object is expected to evolve continuously and when
individual days have sparse filter coverage or noisy measurements.

## 4. All-Parameter Photometric Fitting

`allpams_fitting` extends the basic model by allowing selected global disk or
stellar parameters to vary. In the current wrapper, this includes global values
such as `R_star`, `R_out`, and `distance`.

Use this mode carefully: it has more freedom and can be more degenerate than
the basic mode. It is best used for clear accretion fits and enough data per day
to constrain global parameters.

## 5. AR Mode

`AR_fitting` enables the mode where the fitting machinery can explore variable
inner-radius behavior. This is useful when the standard fixed-geometry disk
model does not capture the data well, or when you want to test whether geometry
changes can mimic changes in accretion/extinction.

The wrapper saves and visualizes results in the same style as the other
photometric modes, so AR outputs can be compared with basic or red-excess runs.
This is in no mean a definite answer, rather a statistcal ecidence for most likely
combination.

## 6. Red-Excess Fitting

`rede_fitting` adds an extra blackbody component on top of the accretion-disk
model. It is designed for cases where long-wavelength filters show excess flux
relative to the basic disk model. Usually when dust is obscuring the observations.

The red-excess component can be parameterized by temperature (`T_bb`) or radius
(`R_bb`) thus creating a Planck model (assuming circular area). The wrapper 
regularizes the chosen red-excess parameter alongside `logMdot` and `Av`.

Typical use:

```python
results = rede_fitting(
    df,
    required_filters=("J", "H", "K", "L"),
    fit_filters=("J", "H", "K", "L"),
    filename="red_excess",
    lambda_reg=1,
    red_excess_fit_param="T_bb",
)
```

## 7. Spectral Fitting

`spectral_fitting` fits spectral samples directly rather than discrete filter
photometry. It can run in a plain accretion mode or a red-excess spectral mode.

Plain spectral mode:

```python
results = spectral_fitting(spectra, lambda_reg=1, rede=False)
```

Accretion plus red-excess spectral mode:

```python
results = spectral_fitting(spectra, lambda_reg=1, rede=True)
```

Spectral fitting expects wavelength and flux columns. If `Frequency` is absent,
the code computes it from wavelength.

## 8. Combining Fitting Mechanisms

The project is useful because modes can be run side by side rather than treated
as isolated workflows. A common pattern is:

1. Run `basic_fitting` as a baseline.
2. Run `rede_fitting` with longer-wavelength filters included.
3. Run `spectral_fitting` for spectra from matching epochs.
4. Save each result bundle with `result_saver`.
5. Compare parameter evolution across all result bundles.

Saved result bundles use the same naming convention:

```text
<name>_daily_params.csv
<name>_global_params.csv
<name>_fit_info.csv
<name>_daily_data.csv
```

That shared structure is what makes comparison and visualization workflows work.

## 9. Visualization Types

The visualization helpers cover several levels of inspection:

- `plot_results_regularized`: shows fitted daily parameter evolution and model behavior.
- `plot_parameter_evolution`: focuses on time evolution of fitted parameters.
- `create_color_plots_with_model`: compares observed color behavior with model predictions.
- `plot_unique_filter_with_model`: visualizes each filter against model predictions.
- `create_residual_plots`: shows where the model misses the data.
- `plot_red_excess_results`: visualizes the accretion plus blackbody red-excess fit.
- `plot_spectral_fit_day`: inspects one spectral epoch.
- `plot_spectral_fit_grid`: makes a multi-epoch spectral overview.
- `visualize_spectral_results`: wraps the common spectral plots.

Most plotting functions accept a save path or output directory. Public plot
outputs are saved as PDFs in `results/generated/pictures/`.

## 10. Comparison Modes

`compare_daily_params_evolution` compares saved result bundles by reading every
`*_daily_params.csv` file in a results directory. This is useful for comparing:

- photometric vs spectroscopic fits
- JH vs JHK filter sets
- basic vs red-excess models
- different regularization strengths
- fitted extinction values against external extinction estimates, if supplied

The final-panel helpers add more publication-style comparison tools, including
color-grid panels, SED panels, and model/data overlays.

## 11. Synthetic Data And Residual Experiments

The synthetic-data helpers can generate model-based synthetic photometry from a
fit and compare it to the original observations. This is useful for checking
whether a fitted model reproduces filters that were not included in a fit, or
for visualizing excesses and residual structure.

Useful functions:

- `generate_synthetic_data`
- `save_synthetic_to_file`
- `compare_original_synthetic`

## 12. Saving, Reopening, And Reusing Results

Use `result_saver` after any successful fit. Use `result_opener` to reload the
bundle later without rerunning the optimizer.

```python
result_saver(results, filepath="results/generated/datas/basic_JHK")
loaded = result_opener(filepath="results/generated/datas/basic_JHK")
```

This allows a workflow where expensive fitting is done once, then visualization
and comparison notebooks can be rerun quickly.


### Configurable Final Round Up workflow

`_private/notebooks/Ultimate_mine.ipynb` follows import → configure → load/fit →
analyse → Final Panels. The separate configuration cell selects radius,
photometric filters, spectral JH windows, red-continuum threshold and masks, parameter
bounds, result stems and output directories. `RUN_FITS=False` loads existing
results; `SAVE_FIGURES=False` displays without replacing published outputs.
No batch runner or folder deletion is required to view the latest results.

Final Panels defaults to the saved 3R JH photometric and spectral bundles.
It preserves raw data in the magnitude panels and interpolated K/L/W1
measurements at directly fitted epochs in the excess panel. Parameter-
interpolated excess is excluded; only L/W1 has the 75% error cut. Residuals
are evaluated at each measurement's actual wavelength with the fitted
Mdot/Av covariance. The widget backend shows live full and zoom canvases;
it does not rasterize them merely for notebook display.

`ultimate_fitting.py` computes covariance errors in the normal workflow.
Its `expand_bounds` option (enabled by basic/red/spectral convenience wrappers)
retries active bounds and records attempts, retaining independent epoch fits.
Boundary-asymptotic, rank-deficient or extremely weak joint BB components are
flagged and their numerical candidates retained separately. Retry exhaustion
raises an error. The notebook's `PHOTOMETRIC_NOISE` controls separate missing
measurement and common systematic flux uncertainties. Missing photometric
errors default to 1%; the unchanged 5% systematic term is added in quadrature
to both reported and substituted errors, giving 5.099% total for missing errors.
Reported positive errors are retained. The spectral fallback remains 10%.
Local covariance still omits correlated calibration/systematic uncertainties.

`final_tables.py` builds Table 2, radius comparisons, red parameter tables,
bound/coverage and covariance diagnostics. It exports supplied mass estimates;
`integrate_accretion_history` remains in `final_panels.py`. The notebook offers
an optional complete table cell for all saved radius and red fits. CSV, LaTeX
and mass JSON exports use a configurable directory without deleting outputs.
Historical provenance/error investigations remain saved in Final Round Up.

`plot_red_excess_evolution` and `plot_red_excess_correlations` in
`ultimate_visualisation.py` accept either a saved result or a photo/spectral
mapping, optional save paths and a show flag. They preserve joint T/R fits,
flagged components and uncertainty arrows. The four correlation panels share
viridis observation-year colours. `plot_spectral_fit_grid` accepts full masked
observations through `observed_df` and `save_dir=None` for display-only use;
fitted regions are shaded from the saved fitted data. Contextual spectral fits
retain the instrument's JH disk window and fit the BB continuum from K onward
(wavelengths >= 2.0 micron), masking the inclusive 2.8--3.3 micron ice-water
interval along with the existing atmospheric masks. The core fitter applies
`spectral_red_mask_regions_micron` to every fitting stage. XSHOOTER and
SXD-only epochs lack L coverage and are excluded from contextual BB fitting.
Eligibility requires at least two distinct unmasked continuum wavelengths in
each of K (2.0--2.5 micron) and L (>3.3--4.2 micron); the standard BB
identifiability checks then apply. Unregistered photometric wavelengths
still stop data loading.
