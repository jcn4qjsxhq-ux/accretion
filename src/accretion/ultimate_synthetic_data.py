"""Synthetic data generation and comparison helpers."""

from __future__ import annotations

try:
    from .ultimate_common import *
except ImportError:
    from ultimate_common import *

try:
    from .ultimate_physics import *
except ImportError:
    from ultimate_physics import *

try:
    from .ultimate_file_organisers import assign_jd_day, average_daily_measurements, get_daily_data
except ImportError:
    from ultimate_file_organisers import assign_jd_day, average_daily_measurements, get_daily_data

try:
    from .ultimate_visualisation import add_gregorian_top_axis, _generated_figure_path
    from .ultimate_file_organisers import generated_data_path
except ImportError:
    from ultimate_visualisation import add_gregorian_top_axis, _generated_figure_path
    from ultimate_file_organisers import generated_data_path

try: 
    from .ultimate_fitting import *
except ImportError:
    from ultimate_fitting import *


def generate_synthetic_data(results, original_df,
    noise_level=0.0, random_seed=None, output_file=None):
    if not results or not results['success']:
        raise ValueError("Invalid or unsuccessful fitting results provided")

    if random_seed is not None:
        np.random.seed(random_seed)

    daily_params = results['daily_params']
    global_params = results.get('global_params', {})
    fit_info = results.get('fit_info', {})

    default_params = {
        'M': 0.5 * M_sun,
        'R_star': 3.0 * R_sun,
        'R_in': 3.0 * R_sun,
        'R_out': 2 * constants.au,
        'distance': 700 * constants.parsec,
        'T_bb': RED_EXCESS_DEFAULT_T_BB,
        'R_bb': RED_EXCESS_DEFAULT_R_BB
    }
    for param_name, value in global_params.items():
        if param_name in default_params:
            default_params[param_name] = value

    wavelengths = get_filter_wavelengths()
    filter_frequencies = {
        name: wavelength_to_frequency(wavelength)
        for name, wavelength in wavelengths.items()
    }
    unique_jds = original_df['JD'].unique()
    print(f"Generating synthetic data for {len(unique_jds)} days found in original data...")

    synthetic_rows = []
    for jd_day in unique_jds:
        day_df = original_df[original_df['JD'] == jd_day]
        fitted_jds = list(daily_params.keys())
        closest_jd = min(fitted_jds, key=lambda x: abs(x - jd_day))
        is_calculated = abs(closest_jd - jd_day) < 0.5

        day_params = daily_params[closest_jd]
        mdot = day_params['Mdot'] / year * M_sun
        av = day_params['Av']

        red_excess_mode = fit_info.get('red_excess_mode', False) or any(
            key in day_params for key in ('T_bb', 'R_bb', 'logR_bb')
        )
        T_bb = default_params['T_bb']
        R_bb = default_params['R_bb']
        if red_excess_mode:
            _, T_bb, R_bb = _resolve_red_excess_component(
                day_params,
                fit_info,
                default_T_bb=default_params['T_bb'],
                default_R_bb=default_params['R_bb']
            )

        M_star = default_params['M']
        R_star = default_params['R_star']
        R_out = default_params['R_out']
        distance = default_params['distance']
        R_in = global_params.get(
            'R_in',
            R_star * 2 if fit_info.get('AR_mode', False) else R_star
        )

        for _, row in day_df.iterrows():
            filter_name = row['Filter']
            frequency, _ = _photometric_frequency_from_row(row, filter_name, filter_frequencies)
            if not np.isfinite(frequency):
                print(f"Warning: Filter {filter_name} has no usable wavelength, keeping original values")
                synthetic_rows.append(row.to_dict())
                continue

            try:
                log_flux = accretion_model(
                    np.array([frequency]), mdot, M_star, R_star, R_out, R_in, distance, av
                )[0]
                if log_flux < -45:
                    print(f"Warning: Invalid model result for {filter_name} on JD {jd_day}")
                    synthetic_rows.append(row.to_dict())
                    continue

                synthetic_flux = np.exp(log_flux)
                synthetic_flux_re = synthetic_flux

                if red_excess_mode:
                    try:
                        bb_log_flux = planck_model_custom(
                            np.array([frequency]), T_bb, distance, av, R_bb=R_bb
                        )[0]
                        if np.isfinite(bb_log_flux):
                            synthetic_flux_re = synthetic_flux + np.exp(bb_log_flux)
                        else:
                            print(f"Invalid bb_log_flux={bb_log_flux}, using accretion-only flux")
                    except Exception as exc:
                        print(f"Warning: Planck addition failed for {filter_name} JD {jd_day}: {exc}")

                if noise_level > 0:
                    noise = np.random.normal(0, noise_level * synthetic_flux)
                    synthetic_flux = max(synthetic_flux + noise, 1e-15)
                    if red_excess_mode:
                        synthetic_flux_re = max(synthetic_flux_re + noise, 1e-15)

                zp = row['ZP']
                if synthetic_flux > 0 and zp > 0:
                    synthetic_mag = -2.5 * np.log10(synthetic_flux / zp)
                    if red_excess_mode:
                        synthetic_mag_re = -2.5 * np.log10(synthetic_flux_re / zp)
                else:
                    synthetic_mag = 99.0
                    synthetic_flux = 0.0
                    if red_excess_mode:
                        synthetic_mag_re = 99.0
                        synthetic_flux_re = 0.0

                if noise_level > 0:
                    synthetic_fluxerr = noise_level * synthetic_flux
                    if red_excess_mode:
                        synthetic_fluxerr_re = noise_level * synthetic_flux_re
                else:
                    original_relative_err = row['Fluxerr'] / max(row['Flux'], 1e-15)
                    synthetic_fluxerr = original_relative_err * synthetic_flux
                    if red_excess_mode:
                        synthetic_fluxerr_re = original_relative_err * synthetic_flux_re

                if synthetic_flux > 0 and synthetic_fluxerr > 0:
                    synthetic_magerr = 1.0857 * synthetic_fluxerr / synthetic_flux
                    if red_excess_mode:
                        synthetic_magerr_re = 1.0857 * synthetic_fluxerr_re / synthetic_flux_re
                else:
                    synthetic_magerr = 0.1
                    if red_excess_mode:
                        synthetic_magerr_re = 0.1

                synthetic_row = row.to_dict()
                synthetic_row['Mag'] = synthetic_mag
                synthetic_row['Flux'] = synthetic_flux
                synthetic_row['Magerr'] = synthetic_magerr
                synthetic_row['Fluxerr'] = synthetic_fluxerr
                synthetic_row['Calculated'] = is_calculated

                if red_excess_mode:
                    synthetic_row['Mag_re'] = synthetic_mag_re
                    synthetic_row['Flux_re'] = synthetic_flux_re
                    synthetic_row['Magerr_re'] = synthetic_magerr_re
                    synthetic_row['Fluxerr_re'] = synthetic_fluxerr_re

                synthetic_rows.append(synthetic_row)

            except Exception as exc:
                print(f"Error generating synthetic data for {filter_name} on JD {jd_day}: {exc}")
                synthetic_rows.append(row.to_dict())
                continue

    synthetic_df = pd.DataFrame(synthetic_rows)
    synthetic_df.sort_values('JD', inplace=True)
    synthetic_df.reset_index(drop=True, inplace=True)

    print(f"Successfully generated {len(synthetic_df)} synthetic data points")
    print(f"Filters in synthetic data: {sorted(synthetic_df['Filter'].unique())}")

    if output_file:
        save_synthetic_to_file(synthetic_df, output_file)
        print(f"Synthetic data saved to {output_file}")

    return synthetic_df


def save_synthetic_to_file(synthetic_df, filename):
    """
    Save synthetic DataFrame to text file in the original format.


    Parameters:
    -----------
    synthetic_df : pd.DataFrame
        The synthetic data DataFrame
    filename : str
        Output filename
    """
    output_path = generated_data_path(filename, default_extension='.txt')
    with open(output_path, 'w') as f:
        f.write("JD            Date       Lambda    Mag   Magerr      ZP       Flux    Fluxerr  Interpolated  Flag\n")
        f.write("-----------------------------------------------------------------------------------------------------\n")

        for _, row in synthetic_df.iterrows():
            mag_str = f"{row['Mag']:8.3f}" if row['Mag'] != 99.0 else "   99.000"
            magerr_str = f"{row['Magerr']:7.3f}" if row['Magerr'] > 0 else " -99.000"

            line = (f"{row['JD']:11.3f}    {row['Date']:>10}     "
                   f"{row['Lambda']:5.3f}   {mag_str} {magerr_str}   "
                   f"{row['ZP']:8.3f}    {row['Flux']:8.3f}    {row['Fluxerr']:6.3f}         "
                   f"{row['Interpolated']:>2}       {row['Flag']:1d}\n")
            f.write(line)

    return output_path


def compare_original_synthetic(
    original_df,
    synthetic_df,
    filter_list=None,
    filename="synthetic_vs",
    filter_req=None,
    filter_plot=None,
    difference_mode='excess',
    difference_view='groups',
    crosssection_day=None
):
    """
    Compare original and synthetic data by plotting them together.

    Plots only days where all filters in ``filter_req`` are available, while
    showing every filter listed in ``filter_plot`` on those selected days.

    Parameters:
    -----------
    original_df : pd.DataFrame
        Original data
    synthetic_df : pd.DataFrame or dict
        Synthetic data (optionally with 'Calculated' column) or
        a fitting results dictionary returned by the fitter/result_opener
    filter_list : list, optional
        Backward-compatible alias used when ``filter_req``/``filter_plot`` are
        not provided. When supplied, both are set to this list by default.
    filename : str
        Output filename (without extension)
    filter_req : list, optional
        Filters required to be present for a day to be included.
    filter_plot : list, optional
        Filters to plot on the selected days. Grouped filters can be passed
        using a '+' sign, for example 'L+W1' to plot those two bands with the
        same colour and connected line.
    difference_mode : {'excess', 'residual'}
        ``'excess'`` compares against the base synthetic model, while
        ``'residual'`` compares against the synthetic model including the red
        excess component when available.
    difference_view : {'groups', 'filters'}
        Plot the lower comparison as wavelength groups or as one channel per
        filter.
    crosssection_day : float or None
        If given, produce a spectral cross-section plot for the selected day:
        wavelength on the x-axis, excess magnitude (observed − synthetic) on
        the y-axis, with all available filters shown. Filters in ``filter_req``
        are drawn with a filled circle; filters only in ``filter_plot`` are
        drawn with an open diamond. The closest matching JD_day in the data is
        used if the value does not exactly match.
    """
    import matplotlib.pyplot as plt

    def _as_filter_list(value, default=None):
        if value is None:
            return list(default) if default is not None else []
        if isinstance(value, str):
            return [value]
        return list(value)

    def _parse_filter_groups(values):
        group_keys = []
        group_map = {}
        flat_filters = []
        for value in _as_filter_list(values):
            if value is None:
                continue
            if isinstance(value, str) and '+' in value:
                parts = [part.strip() for part in value.split('+') if part.strip()]
                if len(parts) < 2:
                    parts = [value.strip()]
            else:
                parts = [str(value).strip()]

            parts = [part for part in parts if part]
            if not parts:
                continue

            key = '+'.join(parts)
            if key not in group_map:
                group_keys.append(key)
                group_map[key] = parts
            for part in parts:
                if part not in flat_filters:
                    flat_filters.append(part)
        return group_keys, group_map, flat_filters

    # Backward-compatible input handling:
    # allow passing fit results dict directly (common notebook usage).
    if isinstance(synthetic_df, dict):
        synthetic_df = generate_synthetic_data(
            synthetic_df, original_df, noise_level=0.0, random_seed=None
        )
    elif not hasattr(synthetic_df, 'columns'):
        raise TypeError(
            "synthetic_df must be a pandas DataFrame or fitting results dict"
        )

    legacy_filters = _as_filter_list(filter_list, default=['J', 'H', 'K'])
    filter_req_raw = _as_filter_list(filter_req, default=legacy_filters)
    filter_plot_raw = _as_filter_list(filter_plot, default=filter_req_raw or legacy_filters)

    filter_req_groups, filter_req_map, filter_req = _parse_filter_groups(filter_req_raw)
    filter_plot_groups, filter_plot_map, filter_plot = _parse_filter_groups(filter_plot_raw)

    if not filter_req:
        raise ValueError("filter_req must contain at least one filter")
    if not filter_plot:
        raise ValueError("filter_plot must contain at least one filter")

    difference_mode = difference_mode.lower()
    difference_view = difference_view.lower()
    if difference_mode not in {'excess', 'residual'}:
        raise ValueError("difference_mode must be 'excess' or 'residual'")
    if difference_view not in {'groups', 'filters'}:
        raise ValueError("difference_view must be 'groups' or 'filters'")

    # Use the same day selection logic as the fitting.
    daily_data = get_daily_data(original_df, filter_req)
    allowed_days = sorted(daily_data.keys()) if daily_data else []
    if not allowed_days:
        print(f"No days found with all required filters {filter_req}; nothing to plot.")
        return

    # Average daily measurements for consistent day-level plotting
    original_avg = average_daily_measurements(original_df)

    synth_for_avg = synthetic_df.copy()
    if 'JD_day' not in synth_for_avg.columns:
        synth_for_avg = assign_jd_day(synth_for_avg)

    calc_flags = None
    if 'Calculated' in synth_for_avg.columns:
        calc_flags = (synth_for_avg.groupby(['JD_day', 'Filter'])['Calculated']
                      .any()
                      .reset_index())

    synthetic_avg = average_daily_measurements(synth_for_avg)
    if calc_flags is not None:
        synthetic_avg = synthetic_avg.merge(calc_flags, on=['JD_day', 'Filter'], how='left')
    if 'Calculated' not in synthetic_avg.columns:
        synthetic_avg['Calculated'] = False
    else:
        synthetic_avg['Calculated'] = synthetic_avg['Calculated'].fillna(False)

    def _add_flux_derived_magnitude(frame, flux_col='Flux', zp_col='ZP', out_col='Mag_model_space'):
        if flux_col not in frame.columns or zp_col not in frame.columns:
            return
        flux = pd.to_numeric(frame[flux_col], errors='coerce')
        zp = pd.to_numeric(frame[zp_col], errors='coerce')
        mag = np.full(len(frame), np.nan, dtype=float)
        valid = np.isfinite(flux) & np.isfinite(zp) & (flux > 0) & (zp > 0)
        mag[valid] = -2.5 * np.log10(flux[valid] / zp[valid])
        frame[out_col] = mag

    _add_flux_derived_magnitude(original_avg)
    _add_flux_derived_magnitude(synthetic_avg)
    if 'Flux_re' in synthetic_avg.columns:
        _add_flux_derived_magnitude(synthetic_avg, flux_col='Flux_re', out_col='Mag_re_model_space')

    # Keep full averages for the cross-section plot (all filters, all days).
    original_avg_full = original_avg.copy()
    synthetic_avg_full = synthetic_avg.copy()

    original_avg = original_avg[
        original_avg['JD_day'].isin(allowed_days)
        & original_avg['Filter'].isin(filter_plot)
    ].copy()
    synthetic_avg = synthetic_avg[
        synthetic_avg['JD_day'].isin(allowed_days)
        & synthetic_avg['Filter'].isin(filter_plot)
    ].copy()

    filter_to_group = {
        filter_name: group_key
        for group_key, filters in filter_plot_map.items()
        for filter_name in filters
    }

    cmap = mpl.colormaps['tab20b']
    color_positions = np.linspace(0.0, 1.0, max(len(filter_plot_groups), 2), endpoint=False)
    filter_colors = {
        group_key: cmap(color_positions[index])
        for index, group_key in enumerate(filter_plot_groups)
    }

    fig, axes = plt.subplots(len(filter_plot_groups), 1, figsize=(14, 4 * len(filter_plot_groups)))
    if len(filter_plot_groups) == 1:
        axes = [axes]

    for i, group_key in enumerate(filter_plot_groups):
        ax = axes[i]
        group_filters = filter_plot_map[group_key]
        line_style = '-' if all(f in filter_req for f in group_filters) else '--'
        filter_color = filter_colors[group_key]

        # Original data on selected days only.
        orig_group = original_avg[original_avg['Filter'].isin(group_filters)]
        if len(orig_group) > 0:
            first_label = True
            for filter_name in group_filters:
                orig_filter = orig_group[orig_group['Filter'] == filter_name]
                if len(orig_filter) == 0:
                    continue
                label = f'Original {group_key}' if first_label else None
                ax.errorbar(
                    orig_filter['JD_day'],
                    orig_filter['Mag'],
                    yerr=orig_filter['Magerr'],
                    fmt='o-',
                    linestyle=line_style,
                    label=label,
                    alpha=1,
                    capsize=3,
                    color=filter_color,
                    markersize=6,
                )
                first_label = False

            if len(group_filters) > 1 and len(orig_group) > 1:
                sorted_orig = orig_group.sort_values('JD_day')
                ax.plot(
                    sorted_orig['JD_day'],
                    sorted_orig['Mag'],
                    '-',
                    color=filter_color,
                    alpha=0.4,
                    linewidth=1.2
                )

        # Synthetic data on the same selected days.
        synth_group = synthetic_avg[synthetic_avg['Filter'].isin(group_filters)]
        if len(synth_group) > 0:
            first_label = True
            for filter_name in group_filters:
                synth_filter = synth_group[synth_group['Filter'] == filter_name]
                if len(synth_filter) == 0:
                    continue
                label = f'Synthetic {group_key}' if first_label else None
                ax.errorbar(
                    synth_filter['JD_day'],
                    synth_filter['Mag'],
                    yerr=synth_filter['Magerr'],
                    fmt='s-',
                    linestyle=line_style,
                    label=label,
                    alpha=0.9,
                    capsize=3,
                    color=filter_color,
                    markersize=6,
                    markerfacecolor='white'
                )
                first_label = False

        # Add red excess comparison if available
        if 'Mag_re' in synth_group.columns and len(synth_group) > 0:
            synth_group_re = synth_group[synth_group['Mag_re'].notna()]
            if len(synth_group_re) > 0:
                ax.errorbar(
                    synth_group_re['JD_day'],
                    synth_group_re['Mag_re'],
                    yerr=synth_group_re['Magerr_re'],
                    fmt='^',
                    linestyle=line_style,
                    label=f'Synthetic {group_key} (with red excess)',
                    alpha=0.8,
                    capsize=3,
                    color=filter_color,
                    markersize=5
                )

        # Add Gregorian date axis (dynamic with zoom/pan)
        add_gregorian_top_axis(ax, n_ticks=6, fontsize=12, use_grid_ticks=True, dynamic=True)

        ax.set_xlabel('Julian Date', fontsize=12)
        ax.set_ylabel('Magnitude', fontsize=12)
        if all(f in filter_req for f in group_filters):
            req_label = 'required'
        elif any(f in filter_req for f in group_filters):
            req_label = 'mixed'
        else:
            req_label = 'plot only'
        ax.set_title(f'{group_key} Filter - Original vs Synthetic Comparison ({req_label})', fontsize=13)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.invert_yaxis()  # Brighter magnitudes at top

    plt.tight_layout()
    comparison_filename = _generated_figure_path(filename)
    plt.savefig(comparison_filename, dpi=150, bbox_inches='tight')
    print(f"Comparison plot saved to {comparison_filename}")

    # --- Difference plot: grouped by wavelength class or by filter. ---
    wavelengths = get_filter_wavelengths()
    h_wavelength = wavelengths.get('H', 1.60)

    def classify_excess_filter(f):
        if f in ('J', 'H'):
            return 'neutral'
        wl = wavelengths.get(f)
        if wl is None:
            return 'optical'
        return 'red' if wl > h_wavelength else 'optical'

    diff_mag_col = 'Mag'
    diff_magerr_col = 'Magerr'
    diff_flux_mag_col = 'Mag_model_space'
    diff_ylabel = 'Excess Magnitude (observed − model)'
    diff_title = 'Average Daily Excess Magnitude'
    diff_filename = _generated_figure_path(f'{filename}_excess_{difference_view}')
    if difference_mode == 'residual':
        if 'Mag_re' in synthetic_avg.columns and 'Magerr_re' in synthetic_avg.columns:
            diff_mag_col = 'Mag_re'
            diff_magerr_col = 'Magerr_re'
            if 'Mag_re_model_space' in synthetic_avg.columns:
                diff_flux_mag_col = 'Mag_re_model_space'
        else:
            print('Residual mode requested but no red-excess synthetic columns found; using base synthetic model.')
        diff_ylabel = 'Residual Magnitude (observed − model)'
        diff_title = 'Average Daily Residual Magnitude'
        diff_filename = _generated_figure_path(f'{filename}_residual_{difference_view}')

    # Merge original and synthetic daily averages on JD_day + Filter (include errors)
    orig_sel = original_avg[
        original_avg['Filter'].isin(filter_plot)
    ][['JD_day', 'Filter', 'Mag_model_space', 'Magerr']].copy()
    synth_sel = synthetic_avg[
        synthetic_avg['Filter'].isin(filter_plot)
    ][['JD_day', 'Filter', diff_flux_mag_col, diff_magerr_col]].copy()
    orig_sel = orig_sel.rename(columns={'Mag_model_space': 'Mag'})
    synth_sel = synth_sel.rename(columns={diff_flux_mag_col: 'Mag_synth', diff_magerr_col: 'Magerr_synth'})

    merged = orig_sel.merge(synth_sel, on=['JD_day', 'Filter'], how='inner')
    merged['Group'] = merged['Filter'].map(filter_to_group).fillna(merged['Filter'])
    merged['Difference_mag'] = merged['Mag'] - merged['Mag_synth']
    merged['Difference_err'] = np.sqrt(merged['Magerr']**2 + merged['Magerr_synth']**2)
    merged['Category']   = merged['Filter'].apply(classify_excess_filter)

    if difference_view == 'groups':
        difference_df = (
            merged.groupby(['JD_day', 'Category'])
            .agg(Difference_mag=('Difference_mag', 'mean'), Difference_err=('Difference_err', 'mean'))
            .reset_index()
            .sort_values('JD_day')
        )

        if len(difference_df) > 0:
            fig_difference, ax_difference = plt.subplots(figsize=(14, 5))
            category_styles = {
                'optical': {'color': cmap(0.05), 'label': f'Optical average {difference_mode}'},
                'neutral': {'color': cmap(0.45), 'label': f'Neutral average {difference_mode}'},
                'red': {'color': cmap(0.85), 'label': f'Red average {difference_mode}'},
            }

            plotted_any = False
            for category, style in category_styles.items():
                cat_data = difference_df[difference_df['Category'] == category]
                if len(cat_data) == 0:
                    continue
                ax_difference.errorbar(
                    cat_data['JD_day'],
                    cat_data['Difference_mag'],
                    yerr=cat_data['Difference_err'],
                    fmt='o-',
                    color=style['color'],
                    linewidth=2,
                    markersize=5,
                    capsize=3,
                    label=style['label']
                )
                plotted_any = True

            if plotted_any:
                add_gregorian_top_axis(ax_difference, n_ticks=6, fontsize=12, use_grid_ticks=True, dynamic=True)
                ax_difference.axhline(0, color='grey', linewidth=1, linestyle='--')
                ax_difference.set_xlabel('Julian Date', fontsize=12)
                ax_difference.set_ylabel(diff_ylabel, fontsize=12)
                ax_difference.set_title(f'{diff_title} by Filter Group', fontsize=13)
                ax_difference.legend(loc='best', fontsize=10)
                ax_difference.grid(True, alpha=0.3)
                ax_difference.invert_yaxis()
                plt.tight_layout()
                plt.savefig(diff_filename, dpi=150, bbox_inches='tight')
                plt.show()
                print(f"Grouped {difference_mode} plot saved to {diff_filename}")
        else:
            print(f'No overlapping selected days for grouped {difference_mode} plot.')
    else:
        difference_df = merged.sort_values(['Group', 'JD_day'])
        if len(difference_df) > 0:
            fig_difference, ax_difference = plt.subplots(figsize=(14, 5))
            for group_key in filter_plot_groups:
                group_data = difference_df[difference_df['Group'] == group_key]
                if len(group_data) == 0:
                    continue
                group_filters = filter_plot_map[group_key]
                line_style = '-' if all(f in filter_req for f in group_filters) else '--'
                ax_difference.errorbar(
                    group_data['JD_day'],
                    group_data['Difference_mag'],
                    yerr=group_data['Difference_err'],
                    fmt='o-',
                    linestyle=line_style,
                    color=filter_colors[group_key],
                    linewidth=2,
                    markersize=5,
                    capsize=3,
                    label=group_key
                )

            add_gregorian_top_axis(ax_difference, n_ticks=6, fontsize=12, use_grid_ticks=True, dynamic=True)
            ax_difference.axhline(0, color='grey', linewidth=1, linestyle='--')
            ax_difference.set_xlabel('Julian Date', fontsize=12)
            ax_difference.set_ylabel(diff_ylabel, fontsize=12)
            ax_difference.set_title(f'{diff_title} by Filter Group', fontsize=13)
            ax_difference.legend(loc='best', fontsize=10)
            ax_difference.grid(True, alpha=0.3)
            ax_difference.invert_yaxis()
            plt.tight_layout()
            plt.savefig(diff_filename, dpi=150, bbox_inches='tight')
            plt.show()
            print(f"Per-filter {difference_mode} plot saved to {diff_filename}")
        else:
            print(f'No overlapping selected days for per-filter {difference_mode} plot.')

    # --- Flux difference plot: observed minus model in flux units ---
    flux_diff_col = 'Flux'
    flux_diff_err_col = 'Fluxerr'
    flux_ylabel = 'Excess Flux (observed − model)'
    flux_title = 'Average Daily Excess Flux'
    flux_filename = _generated_figure_path(f'{filename}_flux_{difference_view}')
    if difference_mode == 'residual':
        if 'Flux_re' in synthetic_avg.columns and 'Fluxerr_re' in synthetic_avg.columns:
            flux_diff_col = 'Flux_re'
            flux_diff_err_col = 'Fluxerr_re'
        else:
            print('Residual mode requested but no red-excess synthetic flux columns found; using base synthetic model.')
        flux_title = 'Average Daily Residual Flux'
        flux_filename = _generated_figure_path(f'{filename}_residual_flux_{difference_view}')

    orig_flux_sel = original_avg[
        original_avg['Filter'].isin(filter_plot)
    ][['JD_day', 'Filter', 'Flux', 'Fluxerr']].copy()
    synth_flux_sel = synthetic_avg[
        synthetic_avg['Filter'].isin(filter_plot)
    ][['JD_day', 'Filter', flux_diff_col, flux_diff_err_col]].copy()
    synth_flux_sel = synth_flux_sel.rename(columns={flux_diff_col: 'Flux_synth', flux_diff_err_col: 'Fluxerr_synth'})

    merged_flux = orig_flux_sel.merge(synth_flux_sel, on=['JD_day', 'Filter'], how='inner')
    merged_flux['Group'] = merged_flux['Filter'].map(filter_to_group).fillna(merged_flux['Filter'])
    merged_flux['Difference_flux'] = merged_flux['Flux'] - merged_flux['Flux_synth']
    merged_flux['Difference_err_flux'] = np.sqrt(
        merged_flux['Fluxerr']**2 + merged_flux['Fluxerr_synth']**2
    )
    merged_flux['Category'] = merged_flux['Filter'].apply(classify_excess_filter)

    if difference_view == 'groups':
        flux_df = (
            merged_flux.groupby(['JD_day', 'Category'])
            .agg(Difference_flux=('Difference_flux', 'mean'), Difference_err_flux=('Difference_err_flux', 'mean'))
            .reset_index()
            .sort_values('JD_day')
        )

        if len(flux_df) > 0:
            fig_flux, ax_flux = plt.subplots(figsize=(14, 5))
            plotted_any = False
            for category, style in category_styles.items():
                cat_data = flux_df[flux_df['Category'] == category]
                if len(cat_data) == 0:
                    continue
                ax_flux.errorbar(
                    cat_data['JD_day'],
                    cat_data['Difference_flux'],
                    yerr=cat_data['Difference_err_flux'],
                    fmt='o-',
                    color=style['color'],
                    linewidth=2,
                    markersize=5,
                    capsize=3,
                    label=style['label'].replace('Magnitude', 'Flux')
                )
                plotted_any = True

            if plotted_any:
                add_gregorian_top_axis(ax_flux, n_ticks=6, fontsize=12, use_grid_ticks=True, dynamic=True)
                ax_flux.axhline(0, color='grey', linewidth=1, linestyle='--')
                ax_flux.set_xlabel('Julian Date', fontsize=12)
                ax_flux.set_ylabel(flux_ylabel, fontsize=12)
                ax_flux.set_title(flux_title + ' by Filter Group', fontsize=13)
                ax_flux.legend(loc='best', fontsize=10)
                ax_flux.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(flux_filename, dpi=150, bbox_inches='tight')
                plt.show()
                print(f'Grouped flux {difference_mode} plot saved to {flux_filename}')
        else:
            print(f'No overlapping selected days for grouped flux {difference_mode} plot.')
    else:
        flux_df = merged_flux.sort_values(['Group', 'JD_day'])
        if len(flux_df) > 0:
            fig_flux, ax_flux = plt.subplots(figsize=(14, 5))
            for group_key in filter_plot_groups:
                group_data = flux_df[flux_df['Group'] == group_key]
                if len(group_data) == 0:
                    continue
                line_style = '-' if all(f in filter_req for f in filter_plot_map[group_key]) else '--'
                ax_flux.errorbar(
                    group_data['JD_day'],
                    group_data['Difference_flux'],
                    yerr=group_data['Difference_err_flux'],
                    fmt='o-',
                    linestyle=line_style,
                    color=filter_colors[group_key],
                    linewidth=2,
                    markersize=5,
                    capsize=3,
                    label=group_key
                )

            add_gregorian_top_axis(ax_flux, n_ticks=6, fontsize=12, use_grid_ticks=True, dynamic=True)
            ax_flux.axhline(0, color='grey', linewidth=1, linestyle='--')
            ax_flux.set_xlabel('Julian Date', fontsize=12)
            ax_flux.set_ylabel(flux_ylabel, fontsize=12)
            ax_flux.set_title(flux_title + ' by Filter Group', fontsize=13)
            ax_flux.legend(loc='best', fontsize=10)
            ax_flux.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(flux_filename, dpi=150, bbox_inches='tight')
            plt.show()
            print(f'Per-filter flux {difference_mode} plot saved to {flux_filename}')
        else:
            print(f'No overlapping selected days for per-filter flux {difference_mode} plot.')

    # --- Cross-section plot: excess vs wavelength for a single day ---
    if crosssection_day is not None:
        all_jd_days = sorted(
            set(original_avg_full['JD_day'].unique()) & set(synthetic_avg_full['JD_day'].unique())
        )
        if not all_jd_days:
            print('Cross-section: no overlapping days in original and synthetic data.')
        else:
            cs_jd = min(all_jd_days, key=lambda d: abs(d - crosssection_day))
            print(f'Cross-section: using JD_day={cs_jd} (requested {crosssection_day})')

            orig_cs = original_avg_full[original_avg_full['JD_day'] == cs_jd].copy()
            synth_cs = synthetic_avg_full[synthetic_avg_full['JD_day'] == cs_jd].copy()

            # Choose synthetic column based on difference_mode
            cs_mag_col = 'Mag'
            cs_magerr_col = 'Magerr'
            cs_flux_mag_col = 'Mag_model_space'
            if difference_mode == 'residual':
                if 'Mag_re' in synth_cs.columns:
                    cs_mag_col = 'Mag_re'
                    cs_magerr_col = 'Magerr_re'
                    if 'Mag_re_model_space' in synth_cs.columns:
                        cs_flux_mag_col = 'Mag_re_model_space'
                else:
                    print('Cross-section residual mode: no red-excess columns, using base model.')

            cs_synth_sel = synth_cs[['Filter', cs_flux_mag_col, cs_magerr_col]].rename(
                columns={cs_flux_mag_col: 'Mag_synth', cs_magerr_col: 'Magerr_synth'}
            )
            cs_merged = orig_cs[['Filter', 'Mag_model_space', 'Magerr']].rename(
                columns={'Mag_model_space': 'Mag'}
            ).merge(
                cs_synth_sel, on='Filter', how='inner'
            )
            cs_merged['Excess_mag'] = cs_merged['Mag'] - cs_merged['Mag_synth']
            cs_merged['Excess_err'] = np.sqrt(
                cs_merged['Magerr']**2 + cs_merged['Magerr_synth']**2
            )

            wl_map = get_filter_wavelengths()
            cs_merged['Wavelength'] = cs_merged['Filter'].map(wl_map)
            cs_merged = cs_merged.dropna(subset=['Wavelength']).sort_values('Wavelength')

            if len(cs_merged) == 0:
                print(f'Cross-section: no data with known wavelengths for JD_day={cs_jd}.')
            else:
                # Build per-filter colours from a fresh tab20b cmap slice
                all_cs_filters = cs_merged['Filter'].tolist()
                cs_positions = np.linspace(0.0, 1.0, max(len(all_cs_filters), 2), endpoint=False)
                cs_colors = {
                    f: cmap(cs_positions[idx]) for idx, f in enumerate(all_cs_filters)
                }

                fig_cs, ax_cs = plt.subplots(figsize=(10, 5))
                for _, row in cs_merged.iterrows():
                    f = row['Filter']
                    is_req = f in filter_req
                    marker = 'o' if is_req else 'D'
                    mfc = cs_colors[f]
                    mec = cs_colors[f]
                    mfcw = mfc if is_req else 'white'
                    ax_cs.errorbar(
                        row['Wavelength'],
                        10**(row['Excess_mag']),
                        yerr=10**(row['Excess_err']),
                        fmt=marker,
                        color=mec,
                        markerfacecolor=mfcw,
                        markersize=9,
                        capsize=4,
                        linewidth=1.5,
                        label=f
                    )
                    ax_cs.annotate(
                        f,
                        xy=(row['Wavelength'], 10**(row['Excess_mag'])),
                        xytext=(4, 4),
                        textcoords='offset points',
                        fontsize=9,
                        color=mec
                    )

                ax_cs.axhline(0, color='grey', linewidth=1, linestyle='--')
                ax_cs.set_xlabel('Wavelength (μm)', fontsize=12)
                ax_cs.set_ylabel(
                    f'{difference_mode.capitalize()} Magnitude (observed − model)', fontsize=12
                )
                ax_cs.set_title(
                    f'Spectral Cross-Section — JD {cs_jd}  '
                    f'(● = filter_req, ◆ = plot-only)',
                    fontsize=13
                )
                ax_cs.grid(True, alpha=0.3)
                ax_cs.invert_yaxis()
                plt.tight_layout()
                cs_filename = _generated_figure_path(f'{filename}_crosssection_{cs_jd}')
                plt.savefig(cs_filename, dpi=150, bbox_inches='tight')
                plt.show()
                print(f'Cross-section plot saved to {cs_filename}')

__all__ = ['generate_synthetic_data', 'save_synthetic_to_file', 'compare_original_synthetic']
