# Final Round Up

Updated 2026-10-07 using the existing configurable pipeline.

## Use the notebook

In `_private/notebooks/Ultimate_mine.ipynb`, run the imports and following configuration cell, load photometry, and run the two Final Panels cells. The default radius is 3R, with `RUN_FITS=False` and `SAVE_FIGURES=False`, so saved results open as live interactive panels without refitting or overwriting outputs. Contextual red-excess analysis and optional complete tables reuse the saved bundles. Configuration follows imports so it works in a fresh kernel.

The notebook's two flux uncertainty controls are:

```python
PHOTOMETRIC_NOISE = {"missing_photometric_fractional_error": .01,
                     "systematic_fractional_error": .05}
```

Missing, zero or invalid photometric errors use 1% measurement uncertainty. The 5% common systematic term is added in quadrature to every measurement: missing errors therefore have 5.099% total, while the median reported J/H error of 0.921% gives approximately 5.084% total. Positive reported errors retain their values. These are adopted likelihood assumptions, not estimates of missing errors from the data. Spectral noise assumptions remain unchanged; the contextual spectral red-excess fits were subsequently rerun with the corrected wavelength selection below.

## Refreshed results

All photometric fits used by this folder were rerun: 84 JH epochs at each of 2R, 3R and 4R, and 28 contextual JHKL epochs with joint temperature/radius fitting. Bound expansion and identifiability checks run through the standard fitting wrappers. JH central parameters are unchanged; 11 J/H measurements across six epochs use the fallback, and only those six epochs have smaller JH parameter errors. Integrated mass central estimates are unchanged.

Main results, Table 2 and integrated masses use JH fits. Table 2 contains all 11 spectral epochs and the four 2026 photometric epochs at 3R. The contextual JHKL and spectral JH + K-onward fits provide separate red-excess context. The spectral BB continuum begins at 2.0 micron and masks the inclusive 2.8--3.3 micron ice-water interval, preserving the existing atmospheric masks. Eligible spectra must contain at least two distinct unmasked continuum wavelengths in each of K (2.0--2.5 micron) and L (>3.3--4.2 micron). Disk initialization retains the existing IRTF JH selection (1.20--1.60 micron), followed by the existing joint four-parameter refinement and blue-BB soft penalty. The red figures report 24 identifiable photometric BB epochs and six spectral BB epochs; four photometric components remain unidentifiable, with NaN reported T/R and retained optimizer candidates.

The corrected spectral range was refitted for the six IRTF epochs with K and L coverage. Three XSHOOTER and two SXD-only IRTF epochs lack L coverage and are excluded from contextual spectral red-excess fitting. The four dependent PDFs were regenerated and visually inspected: Context_spectral_red_excess_overview, red_excess_context_correlations, rede4L_T_evolution, and spectral_red_excess_parameter_evolution. Their saved spectral bundle, parameter/correlation tables, coverage diagnostics and covariance diagnostics were updated. All main JH bundles, Table 2, integrated masses, photometric red fits and seven other PDFs were verified unchanged. See logs/spectral_red_mask_refit.json for per-epoch coverage and tables/spectral_red_range_change_audit.csv for old/new parameters. The six retained fits reproduce their previous K-onward parameters. See tables/spectral_red_coverage_selection.csv for all 11 epochs and exclusion reasons.

## Plotting and uncertainty conventions

Photometric fits and colour grids exclude interpolated rows. Interpolated K/L/W1 measurements at directly fitted epochs remain in the excess plot with ordinary markers; excess requiring interpolated fitted parameters is excluded. Only L/W1 has the 75% relative-error cut. There is no additional excess uncertainty floor. Residual predictions use each measurement's actual wavelength and the local Mdot/Av covariance.

Errors come from the local weighted model Jacobian with absolute likelihood uncertainties, without fixed parameter-error placeholders or reduced-chi-squared rescaling for the two-point JH inversion. Initial main-panel vertical grids use Gregorian year ticks; interactive zoom remains dynamic. Red correlations retain photometric and spectral symbols with shared viridis observation-year colours. Correlation axes use parameters from the same joint fits and share covariance. Triangles mark uncertainty continuing beyond displayed axes; full errors are in CSV tables.

XSHOOTER NIR coverage reaches 2.479 micron; overviews use full available masked NIR coverage, while main fits use JH only. XSHOOTER and SXD-only IRTF epochs lack L coverage and are excluded from contextual BB fitting, while remaining in the main JH results. IRTF overview display extends to 4.2 micron. The latest IRTF 20260704 divisors remain SXD 0.653625 and LXD 0.659931. Spectral covariances treat samples as independent and do not model correlated calibration errors; the inherited blue BB term is a soft penalty, with actual ratios exported in the coverage diagnostics. BB radii retain the existing emitting-area convention.

## Provenance and verification

The 2015 photometric JH fit exists at JD 2457226.5 (daily date July 23). Earlier databases identify the same J/H/K values as IRTF/SpeX-derived photometry, with source JD 2457226.169 (July 22). Daily binning explains the date offset from the spectral JH fit at JD 2457225.7462704 (July 22); these are not independent measurements. Dates and original source records are preserved in `tables/2015_fit_provenance.csv`.

**37 tests passed.** Synthetic checks recover T/R with strongly absorbed ice-water samples excluded and verify exclusion of K-only, L-only, and ice-window-only coverage through both the core fitter and bound-retry wrapper. Independent covariance calculations using finer derivative steps agree within 1e-4 relative tolerance; all JH standardized residuals are below 1e-6. Saved pickle and CSV parameters agree, with NaN fields omitted on CSV loading treated as missing. Current notebook cells were exercised using real ipympl widget canvases; displayed residuals, Table 2 and red/radius tables match saved exports. Unchanged main JH and photometric fit hashes, seven unaffected figure hashes and input hashes were verified; current contextual spectral hashes are recorded separately.

See `tables/photometric_noise_change_audit.csv` for old/new per-epoch uncertainties, `tables/JH_convergence_uncertainty_audit.csv` for covariance checks, and `logs/uncertainty_investigation.md` for the current provenance/error audit. Historical error-recipe columns in `tables/uncertainty_investigation.csv` are explicitly comparison calculations, rather than adopted uncertainties. `logs/run_manifest.json` and `logs/artifact_validation.json` record current options, source/input hashes and artifact checksums.

## Unidentifiable contextual BB components

| JD | Date | Status |
| --- | --- | --- |
| 2446081.5 | 1985-01-16 | boundary asymptote |
| 2446098.5 | 1985-02-02 | boundary asymptote |
| 2446481.5 | 1986-02-20 | effectively unconstrained (>1000% local T and R uncertainty) |
| 2447286.5 | 1988-05-05 | effectively unconstrained (>1000% local T and R uncertainty) |

## Integrated mass estimates

Mass estimates depend on interpolation across observational gaps. The coverage-limited calculation excludes gaps longer than 365.25 days.

| Family | Radius (Rsun) | Method | Mass (Msun) | Covered years |
| --- | --- | --- | --- | --- |
| photo | 2 | coverage_limited | 0.0005255766 | 12.783025 |
| photo | 2 | full_interval | 0.001678487 | 47.389459 |
| photo | 3 | coverage_limited | 0.0005546242 | 12.783025 |
| photo | 3 | full_interval | 0.001774385 | 47.389459 |
| photo | 4 | coverage_limited | 0.000579885 | 12.783025 |
| photo | 4 | full_interval | 0.001857863 | 47.389459 |
| spectral | 2 | coverage_limited | 0.0002474928 | 2.995664 |
| spectral | 2 | full_interval | 0.0008858929 | 10.995789 |
| spectral | 3 | coverage_limited | 0.0002608952 | 2.995664 |
| spectral | 3 | full_interval | 0.0009309048 | 10.995789 |
| spectral | 4 | coverage_limited | 0.0002724787 | 2.995664 |
| spectral | 4 | full_interval | 0.0009698721 | 10.995789 |

## Files

`figures/` contains the 11 final PDFs; `fits/` contains the saved result bundles; `tables/` contains CSV/LaTeX exports and mass JSON; `logs/` contains current validation and render QA. The output folder remains ignored by Git.
