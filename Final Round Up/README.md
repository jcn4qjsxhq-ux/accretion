# Final Round Up

Reproduce using the Final Round Up section at the end of `_private/notebooks/Ultimate_mine.ipynb`.

Validation: **10 tests passed**; all **9 PDFs visually inspected**. No requested fits were withheld.

## Scientific settings

Table 2 contains the **exact plotted 3R JH results**: all 11 spectral epochs and four 2026 photometric epochs. JHK cross-checks and the separately normalized 2015 fit are excluded. Every parameter and uncertainty was checked against the plot-source bundles.

Updated database_daily_2609.txt; raw-only photometric fits and colours. Interpolated L/W1 points appear as open diamonds in excess panels. Radius comparisons use 3R as baseline. New IRTF divisors: SXD 0.653625; LXD 0.659931.

**Uncertainty limitation:** the inherited fitter uses fixed placeholder parameter errors with debug=False. Reported parameter and propagated mass errors are not statistical confidence intervals. Integrated masses also depend on interpolation across observational gaps.

## Runtime gates

- photo: {'status': 'pass', 'epochs': 28, 'sampled_epochs': [2444098.5, 2460766.5, 2461165.5], 'conservative_seconds': 25.95850600191625, 'hard_timeout_seconds': 600}
- spectral: {'status': 'pass', 'epochs': 11, 'sampled_epochs': [2457209.599, 2460766.1298276, 2461225.8110455], 'conservative_seconds': 38.140563005945296, 'hard_timeout_seconds': 600}

## Fit limitations

All photometric fits exclude interpolated rows.
Existing fitter uncertainties with debug=False are fixed placeholders (0.1 in internal parameter units), not statistically estimated confidence intervals. Propagated mass errors inherit this limitation.
The inherited 10% blue-BB setting is a soft penalty, not a hard upper bound. Actual blue fractions are exported in fit_bound_and_coverage_diagnostics.csv.
XSHOOTER JH-only spectra do not directly constrain the long-wavelength BB component; retain their fits for consistency but inspect wavelength coverage and bound diagnostics.

Parameter bounds and actual blue BB fractions are listed in tables/fit_bound_and_coverage_diagnostics.csv.


## Integrated masses

 radius_Rsun           method  mass_accreted_Msun  covered_years
           2    full_interval            0.001678      47.389459
           2 coverage_limited            0.000526      12.783025
           3    full_interval            0.001774      47.389459
           3 coverage_limited            0.000555      12.783025
           4    full_interval            0.001858      47.389459
           4 coverage_limited            0.000580      12.783025

## Outputs

- [figures/0-4_JH_2R.pdf](figures/0-4_JH_2R.pdf)
- [figures/0-4_JH_3R.pdf](figures/0-4_JH_3R.pdf)
- [figures/0-4_JH_4R.pdf](figures/0-4_JH_4R.pdf)
- [figures/6_colour_grid_JH.pdf](figures/6_colour_grid_JH.pdf)
- [figures/6_colour_grid_JK.pdf](figures/6_colour_grid_JK.pdf)
- [figures/Panels_spectral_fit_overview.pdf](figures/Panels_spectral_fit_overview.pdf)
- [figures/rede4L_T.pdf](figures/rede4L_T.pdf)
- [figures/rede4L_T_evolution.pdf](figures/rede4L_T_evolution.pdf)
- [figures/spectral_red_excess_parameter_evolution.pdf](figures/spectral_red_excess_parameter_evolution.pdf)
- [tables/JH_2R_excess_residuals.csv](tables/JH_2R_excess_residuals.csv)
- [tables/JH_3R_excess_residuals.csv](tables/JH_3R_excess_residuals.csv)
- [tables/JH_4R_excess_residuals.csv](tables/JH_4R_excess_residuals.csv)
- [tables/JH_radius_differences_from_3R.csv](tables/JH_radius_differences_from_3R.csv)
- [tables/JH_radius_fit_comparison.tex](tables/JH_radius_fit_comparison.tex)
- [tables/JH_radius_fit_parameter_comparison.csv](tables/JH_radius_fit_parameter_comparison.csv)
- [tables/JH_radius_fit_summary.csv](tables/JH_radius_fit_summary.csv)
- [tables/JH_radius_integrated_mass.csv](tables/JH_radius_integrated_mass.csv)
- [tables/JH_radius_integrated_mass.json](tables/JH_radius_integrated_mass.json)
- [tables/JH_radius_integrated_mass.tex](tables/JH_radius_integrated_mass.tex)
- [tables/Table_2.csv](tables/Table_2.csv)
- [tables/Table_2.tex](tables/Table_2.tex)
- [tables/colour_JH_I vs I-J.csv](tables/colour_JH_I%20vs%20I-J.csv)
- [tables/colour_JH_J vs J-H.csv](tables/colour_JH_J%20vs%20J-H.csv)
- [tables/colour_JH_J-H vs H-K.csv](tables/colour_JH_J-H%20vs%20H-K.csv)
- [tables/colour_JH_K vs H-K.csv](tables/colour_JH_K%20vs%20H-K.csv)
- [tables/colour_JK_I vs I-J.csv](tables/colour_JK_I%20vs%20I-J.csv)
- [tables/colour_JK_J vs J-K.csv](tables/colour_JK_J%20vs%20J-K.csv)
- [tables/colour_JK_J-H vs H-K.csv](tables/colour_JK_J-H%20vs%20H-K.csv)
- [tables/colour_JK_K vs H-K.csv](tables/colour_JK_K%20vs%20H-K.csv)
- [tables/fit_bound_and_coverage_diagnostics.csv](tables/fit_bound_and_coverage_diagnostics.csv)
- [tables/photo_red_fit_validation.csv](tables/photo_red_fit_validation.csv)
- [tables/rede4L_T_parameters.csv](tables/rede4L_T_parameters.csv)
- [tables/spectral_red_excess_parameters.csv](tables/spectral_red_excess_parameters.csv)
- [tables/spectral_red_fit_validation.csv](tables/spectral_red_fit_validation.csv)
- [fits/basic_JH_2R_newdf_daily_data.csv](fits/basic_JH_2R_newdf_daily_data.csv)
- [fits/basic_JH_2R_newdf_daily_params.csv](fits/basic_JH_2R_newdf_daily_params.csv)
- [fits/basic_JH_2R_newdf_fit_info.csv](fits/basic_JH_2R_newdf_fit_info.csv)
- [fits/basic_JH_2R_newdf_global_params.csv](fits/basic_JH_2R_newdf_global_params.csv)
- [fits/basic_JH_3R_newdf_daily_data.csv](fits/basic_JH_3R_newdf_daily_data.csv)
- [fits/basic_JH_3R_newdf_daily_params.csv](fits/basic_JH_3R_newdf_daily_params.csv)
- [fits/basic_JH_3R_newdf_fit_info.csv](fits/basic_JH_3R_newdf_fit_info.csv)
- [fits/basic_JH_3R_newdf_global_params.csv](fits/basic_JH_3R_newdf_global_params.csv)
- [fits/basic_JH_4R_newdf_daily_data.csv](fits/basic_JH_4R_newdf_daily_data.csv)
- [fits/basic_JH_4R_newdf_daily_params.csv](fits/basic_JH_4R_newdf_daily_params.csv)
- [fits/basic_JH_4R_newdf_fit_info.csv](fits/basic_JH_4R_newdf_fit_info.csv)
- [fits/basic_JH_4R_newdf_global_params.csv](fits/basic_JH_4R_newdf_global_params.csv)
- [fits/irtf_JH_2R_daily_data.csv](fits/irtf_JH_2R_daily_data.csv)
- [fits/irtf_JH_2R_daily_params.csv](fits/irtf_JH_2R_daily_params.csv)
- [fits/irtf_JH_2R_fit_info.csv](fits/irtf_JH_2R_fit_info.csv)
- [fits/irtf_JH_2R_global_params.csv](fits/irtf_JH_2R_global_params.csv)
- [fits/irtf_JH_3R_daily_data.csv](fits/irtf_JH_3R_daily_data.csv)
- [fits/irtf_JH_3R_daily_params.csv](fits/irtf_JH_3R_daily_params.csv)
- [fits/irtf_JH_3R_fit_info.csv](fits/irtf_JH_3R_fit_info.csv)
- [fits/irtf_JH_3R_global_params.csv](fits/irtf_JH_3R_global_params.csv)
- [fits/irtf_JH_4R_daily_data.csv](fits/irtf_JH_4R_daily_data.csv)
- [fits/irtf_JH_4R_daily_params.csv](fits/irtf_JH_4R_daily_params.csv)
- [fits/irtf_JH_4R_fit_info.csv](fits/irtf_JH_4R_fit_info.csv)
- [fits/irtf_JH_4R_global_params.csv](fits/irtf_JH_4R_global_params.csv)
- [fits/rede_4L_T_daily_data.csv](fits/rede_4L_T_daily_data.csv)
- [fits/rede_4L_T_daily_params.csv](fits/rede_4L_T_daily_params.csv)
- [fits/rede_4L_T_fit_info.csv](fits/rede_4L_T_fit_info.csv)
- [fits/rede_4L_T_global_params.csv](fits/rede_4L_T_global_params.csv)
- [fits/spectral_JH_2R_daily_data.csv](fits/spectral_JH_2R_daily_data.csv)
- [fits/spectral_JH_2R_daily_params.csv](fits/spectral_JH_2R_daily_params.csv)
- [fits/spectral_JH_2R_fit_info.csv](fits/spectral_JH_2R_fit_info.csv)
- [fits/spectral_JH_2R_global_params.csv](fits/spectral_JH_2R_global_params.csv)
- [fits/spectral_JH_3R_daily_data.csv](fits/spectral_JH_3R_daily_data.csv)
- [fits/spectral_JH_3R_daily_params.csv](fits/spectral_JH_3R_daily_params.csv)
- [fits/spectral_JH_3R_fit_info.csv](fits/spectral_JH_3R_fit_info.csv)
- [fits/spectral_JH_3R_global_params.csv](fits/spectral_JH_3R_global_params.csv)
- [fits/spectral_JH_4R_daily_data.csv](fits/spectral_JH_4R_daily_data.csv)
- [fits/spectral_JH_4R_daily_params.csv](fits/spectral_JH_4R_daily_params.csv)
- [fits/spectral_JH_4R_fit_info.csv](fits/spectral_JH_4R_fit_info.csv)
- [fits/spectral_JH_4R_global_params.csv](fits/spectral_JH_4R_global_params.csv)
- [fits/spectral_combined_rede_daily_data.csv](fits/spectral_combined_rede_daily_data.csv)
- [fits/spectral_combined_rede_daily_params.csv](fits/spectral_combined_rede_daily_params.csv)
- [fits/spectral_combined_rede_fit_info.csv](fits/spectral_combined_rede_fit_info.csv)
- [fits/spectral_combined_rede_global_params.csv](fits/spectral_combined_rede_global_params.csv)
- [fits/table2_2015_unscaled_SXD_LXD_JHK_daily_data.csv](fits/table2_2015_unscaled_SXD_LXD_JHK_daily_data.csv)
- [fits/table2_2015_unscaled_SXD_LXD_JHK_daily_params.csv](fits/table2_2015_unscaled_SXD_LXD_JHK_daily_params.csv)
- [fits/table2_2015_unscaled_SXD_LXD_JHK_fit_info.csv](fits/table2_2015_unscaled_SXD_LXD_JHK_fit_info.csv)
- [fits/table2_2015_unscaled_SXD_LXD_JHK_global_params.csv](fits/table2_2015_unscaled_SXD_LXD_JHK_global_params.csv)
- [fits/table2_2026_JHK_daily_data.csv](fits/table2_2026_JHK_daily_data.csv)
- [fits/table2_2026_JHK_daily_params.csv](fits/table2_2026_JHK_daily_params.csv)
- [fits/table2_2026_JHK_fit_info.csv](fits/table2_2026_JHK_fit_info.csv)
- [fits/table2_2026_JHK_global_params.csv](fits/table2_2026_JHK_global_params.csv)
- [fits/table2_irtf_JHK_3R_daily_data.csv](fits/table2_irtf_JHK_3R_daily_data.csv)
- [fits/table2_irtf_JHK_3R_daily_params.csv](fits/table2_irtf_JHK_3R_daily_params.csv)
- [fits/table2_irtf_JHK_3R_fit_info.csv](fits/table2_irtf_JHK_3R_fit_info.csv)
- [fits/table2_irtf_JHK_3R_global_params.csv](fits/table2_irtf_JHK_3R_global_params.csv)
- [fits/xshooter_JH_2R_daily_data.csv](fits/xshooter_JH_2R_daily_data.csv)
- [fits/xshooter_JH_2R_daily_params.csv](fits/xshooter_JH_2R_daily_params.csv)
- [fits/xshooter_JH_2R_fit_info.csv](fits/xshooter_JH_2R_fit_info.csv)
- [fits/xshooter_JH_2R_global_params.csv](fits/xshooter_JH_2R_global_params.csv)
- [fits/xshooter_JH_3R_daily_data.csv](fits/xshooter_JH_3R_daily_data.csv)
- [fits/xshooter_JH_3R_daily_params.csv](fits/xshooter_JH_3R_daily_params.csv)
- [fits/xshooter_JH_3R_fit_info.csv](fits/xshooter_JH_3R_fit_info.csv)
- [fits/xshooter_JH_3R_global_params.csv](fits/xshooter_JH_3R_global_params.csv)
- [fits/xshooter_JH_4R_daily_data.csv](fits/xshooter_JH_4R_daily_data.csv)
- [fits/xshooter_JH_4R_daily_params.csv](fits/xshooter_JH_4R_daily_params.csv)
- [fits/xshooter_JH_4R_fit_info.csv](fits/xshooter_JH_4R_fit_info.csv)
- [fits/xshooter_JH_4R_global_params.csv](fits/xshooter_JH_4R_global_params.csv)
