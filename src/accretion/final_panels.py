"""Final-panel plotting helpers for the accretion disk fitting figures.

The notebook should configure inputs and call these functions; importing this
module has no plotting or data-loading side effects.
"""

from __future__ import annotations

import io
from pathlib import Path

import matplotlib as mpl

from accretion.ultimate_common import M_sun, R_sun, constants, np, pd, plt, year
from accretion.ultimate_file_organisers import (
    assign_jd_day,
    average_daily_measurements,
    get_daily_data,
)
from accretion.ultimate_visualisation import (
    _generated_figure_path,
    _generated_output_dir,
    add_gregorian_top_axis,
    julian_to_calendar,
)
from accretion.ultimate_synthetic_data import generate_synthetic_data
from accretion.ultimate_fitting import _coerce_wavelength_meters
from accretion.ultimate_physics import accretion_model, get_filter_wavelengths
from accretion.final_panel_models import calculate_filter_accretion_models, normalize_filter_selection


FINAL_PANEL_STYLE = {
    'font.family': 'serif',
    'font.serif': ['Postscript TX Times', 'Times New Roman', 'Times'],
    'mathtext.fontset': 'stix',
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 16,
}

FINAL_PANEL_COLOURS = {
    'mdot_photometric': '#e6ab02',
    'mdot_spectroscopic': '#54278f',
    'mdot_xshooter': '#0072b2',
    'av_photometric': '#d95f02',
    'av_spectroscopic': '#54278f',
    'av_xshooter': '#0072b2',
    'residual_palette': ['#54278f', '#1b9e77', '#66a61e', '#e6ab02', '#d95f02'],
}


def _normalise_sed_jds(sed_jd_input):
    """Return finite SED JDs from a scalar, list, tuple, or NumPy array."""
    if isinstance(sed_jd_input, (str, bytes)) or np.isscalar(sed_jd_input):
        values = np.array([sed_jd_input], dtype=float)
    else:
        values = np.asarray(sed_jd_input, dtype=float).ravel()
    return [float(jd) for jd in values if np.isfinite(jd)]


def _parse_filter_groups(values):
    """Parse grouped filter labels such as ``L+W1`` while preserving order."""
    group_keys = []
    group_map = {}
    flat_filters = []
    for value in values:
        if isinstance(value, str) and '+' in value:
            parts = [part.strip() for part in value.split('+') if part.strip()]
        else:
            parts = [str(value).strip()]
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


def _float_or_none(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number


def _is_plottable_flag(value):
    if pd.isna(value):
        return True
    try:
        return float(value) != 0.0
    except (TypeError, ValueError):
        return str(value).strip().lower() not in {'0', 'false', 'n', 'no'}


def _resolve_selected_filter_data(df, selected_filters):
    """Return photometry rows for selected filter names or wavelength values."""
    selected_values = normalize_filter_selection(selected_filters)
    if not selected_values:
        return df.iloc[0:0].copy(), [], {}

    filter_table = (
        df[['Filter', 'Lambda']]
        .dropna()
        .assign(Filter=lambda frame: frame['Filter'].astype(str))
        .drop_duplicates()
    )
    filter_table['Lambda'] = pd.to_numeric(filter_table['Lambda'], errors='coerce')
    filter_table = filter_table[np.isfinite(filter_table['Lambda'])].copy()

    resolved_filters = []
    selected_masks = []
    legend_label_map = {}
    for value in selected_values:
        wavelength = _float_or_none(value)
        if wavelength is None:
            filter_name = str(value).strip()
            if filter_name and filter_name not in resolved_filters:
                resolved_filters.append(filter_name)
                selected_masks.append(df['Filter'].astype(str).str.lower() == filter_name.lower())
            continue

        if filter_table.empty:
            continue

        exact_mask = np.isclose(filter_table['Lambda'], wavelength, rtol=1e-6, atol=1e-6)
        candidates = filter_table[exact_mask].copy()
        if candidates.empty:
            canonical = get_filter_wavelengths()
            canonical_matches = [
                filter_name
                for filter_name, lambda_um in canonical.items()
                if np.isclose(float(lambda_um), wavelength, rtol=1e-4, atol=1e-4)
            ]
            if canonical_matches:
                candidates = filter_table[filter_table['Filter'].isin(canonical_matches)].copy()

        if candidates.empty:
            distances = (filter_table['Lambda'] - wavelength).abs()
            nearest_index = distances.idxmin()
            nearest_distance = float(distances.loc[nearest_index])
            tolerance = max(0.02, abs(wavelength) * 0.01)
            if nearest_distance <= tolerance:
                candidates = filter_table.loc[[nearest_index]].copy()

        if candidates.empty:
            available = ', '.join(
                f"{row.Filter}={row.Lambda:g}"
                for row in filter_table.sort_values(['Lambda', 'Filter']).itertuples()
            )
            raise ValueError(
                f"No filter found for wavelength {wavelength:g} micron. "
                f"Available filter wavelengths: {available}"
            )

        candidates['Distance'] = (candidates['Lambda'] - wavelength).abs()
        candidates = candidates.sort_values(['Distance', 'Filter'])
        filter_name = str(candidates.iloc[0]['Filter'])
        lambda_value = float(candidates.iloc[0]['Lambda'])
        if filter_name not in resolved_filters:
            resolved_filters.append(filter_name)
        selected_masks.append(
            (df['Filter'].astype(str).str.lower() == filter_name.lower())
            & np.isclose(pd.to_numeric(df['Lambda'], errors='coerce'), lambda_value, rtol=1e-6, atol=1e-6)
        )
        legend_label_map[(filter_name, lambda_value)] = fr'{wavelength:g} $\mu m$ ({filter_name} band filter)'
        print(f"Resolved FILT wavelength {wavelength:g} micron to filter {filter_name}.")

    if selected_masks:
        combined_mask = selected_masks[0].copy()
        for mask in selected_masks[1:]:
            combined_mask = combined_mask | mask
        df_filt = df[combined_mask].copy()
    else:
        df_filt = df.iloc[0:0].copy()
    if 'Flag' in df_filt.columns:
        df_filt = df_filt[df_filt['Flag'].map(_is_plottable_flag)].copy()
    return df_filt, resolved_filters, legend_label_map


def _normalise_date_range(date_range):
    if date_range is None:
        return None
    if len(date_range) != 2:
        raise ValueError("date_range must be [min_jd, max_jd]. Use None for an open end.")
    limits = []
    for value in date_range:
        if value is None:
            limits.append(None)
        else:
            numeric_value = float(value)
            if not np.isfinite(numeric_value):
                raise ValueError("date_range values must be finite JDs or None.")
            limits.append(numeric_value)
    return limits


def _lambda_sort_key(value):
    try:
        lambda_value = float(value)
    except (TypeError, ValueError):
        return float('inf')
    return lambda_value if np.isfinite(lambda_value) else float('inf')


def _mag_from_flux(frame):
    flux = pd.to_numeric(frame['Flux'], errors='coerce')
    zp = pd.to_numeric(frame['ZP'], errors='coerce')
    mag = np.full(len(frame), np.nan, dtype=float)
    valid = np.isfinite(flux) & np.isfinite(zp) & (flux > 0) & (zp > 0)
    mag[valid] = -2.5 * np.log10(flux[valid] / zp[valid])
    return mag


def _resolve_jd_key(mapping, jd_day, tolerance=1e-6):
    if not isinstance(mapping, dict):
        return None
    if jd_day in mapping:
        return jd_day
    try:
        target = float(jd_day)
    except Exception:
        return None

    numeric_keys = []
    for key in mapping.keys():
        try:
            numeric_keys.append((abs(float(key) - target), key))
        except Exception:
            continue
    if not numeric_keys:
        return None

    distance, key = min(numeric_keys, key=lambda item: item[0])
    if distance <= tolerance:
        return key
    # Backward compatibility for result objects created before JD_day stopped rounding.
    if distance <= 0.51:
        return key
    return None


def _repair_flux_from_mag_local(day_df):
    if day_df is None or len(day_df) == 0 or not {'Flux', 'Mag', 'ZP'}.issubset(day_df.columns):
        return day_df
    out = day_df.copy()
    flux = pd.to_numeric(out['Flux'], errors='coerce')
    mag = pd.to_numeric(out['Mag'], errors='coerce')
    zp = pd.to_numeric(out['ZP'], errors='coerce')
    recover_flux = (flux <= 0) & np.isfinite(mag) & np.isfinite(zp) & (zp > 0)
    if recover_flux.any():
        flux.loc[recover_flux] = zp.loc[recover_flux] * 10 ** (-0.4 * mag.loc[recover_flux])
        out['Flux'] = flux

    if {'Fluxerr', 'Magerr'}.issubset(out.columns):
        fluxerr = pd.to_numeric(out['Fluxerr'], errors='coerce')
        magerr = pd.to_numeric(out['Magerr'], errors='coerce')
        recover_fluxerr = (fluxerr <= 0) & (flux > 0) & np.isfinite(magerr) & (magerr > 0)
        if recover_fluxerr.any():
            fluxerr.loc[recover_fluxerr] = flux.loc[recover_fluxerr] * np.log(10.0) * magerr.loc[recover_fluxerr] / 2.5
            out['Fluxerr'] = fluxerr
    return out


def _prepare_day_series(results, jd_day, filters=None):
    daily_data = results.get('daily_data', {}) if isinstance(results, dict) else {}
    jd_key = _resolve_jd_key(daily_data, jd_day)
    if jd_key is None:
        return None

    day_df = _repair_flux_from_mag_local(daily_data[jd_key].copy())
    if filters is not None and 'Filter' in day_df.columns:
        filter_names = [str(f) for f in filters]
        day_df = day_df[day_df['Filter'].astype(str).isin(filter_names)].copy()
        if len(day_df) == 0:
            return None

    lambda_m = _coerce_wavelength_meters(day_df)
    if 'Frequency' in day_df.columns:
        frequencies = pd.to_numeric(day_df['Frequency'], errors='coerce').to_numpy(dtype=float)
    else:
        frequencies = constants.c / lambda_m

    flux = pd.to_numeric(day_df.get('Flux', np.nan), errors='coerce').to_numpy(dtype=float)
    fluxerr = pd.to_numeric(day_df.get('Fluxerr', np.nan), errors='coerce').to_numpy(dtype=float)
    if 'Filter' in day_df.columns:
        filters = day_df['Filter'].astype(str).to_numpy()
    else:
        filters = np.array([''] * len(day_df), dtype=object)

    valid = np.isfinite(lambda_m) & np.isfinite(frequencies) & np.isfinite(flux) & (flux > 0)
    lambda_m = lambda_m[valid]
    frequencies = frequencies[valid]
    flux = flux[valid]
    fluxerr = fluxerr[valid]
    filters = filters[valid]
    if len(lambda_m) == 0:
        return None

    order = np.argsort(lambda_m)
    return {
        'lambda_m': lambda_m[order],
        'lambda_um': lambda_m[order] * 1e6,
        'frequencies': frequencies[order],
        'flux': flux[order],
        'fluxerr': fluxerr[order],
        'filters': filters[order],
    }


def _prepare_day_series_from_frame(frame, jd_day):
    if frame is None or len(frame) == 0:
        return None

    day_df = frame.copy()
    if 'JD_day' in day_df.columns:
        day_df['JD_day'] = pd.to_numeric(day_df['JD_day'], errors='coerce')
    elif 'JD' in day_df.columns:
        day_df = assign_jd_day(day_df)
    else:
        return None

    day_df = day_df[np.isfinite(day_df['JD_day'])]
    target = float(jd_day)
    if 'DateCode' in day_df.columns:
        group_key = day_df['DateCode'].astype(str)
    elif 'Date' in day_df.columns:
        group_key = day_df['Date'].astype(str)
    else:
        group_key = day_df['JD_day'].astype(str)

    group_jd = day_df.groupby(group_key)['JD_day'].median()
    if len(group_jd) == 0:
        return None
    closest_key = (group_jd.astype(float) - target).abs().idxmin()
    if abs(float(group_jd.loc[closest_key]) - target) > 0.51:
        return None

    day_df = day_df[group_key == closest_key].copy()
    if len(day_df) == 0:
        return None
    return _prepare_day_series({'daily_data': {target: day_df}}, target)


def _get_defaults(results):
    global_params = results.get('global_params', {}) if isinstance(results, dict) else {}
    r_star = global_params.get('R_star', 3.0 * R_sun)
    fit_info = results.get('fit_info', {}) if isinstance(results, dict) else {}
    return {
        'M': global_params.get('M', 0.5 * M_sun),
        'R_star': r_star,
        'R_out': global_params.get('R_out', 2 * constants.au),
        'distance': global_params.get('distance', 700 * constants.parsec),
        'R_in': global_params.get('R_in', r_star * 2 if fit_info.get('AR_mode') else r_star),
    }


def _stellar_mass_solar(results, default_params=None):
    """Resolve stellar mass for M_star*Mdot without a 1 Msun fallback."""
    default_params = default_params or {}
    if 'M_star' in default_params:
        value = float(default_params['M_star'])
        return value / M_sun if value > 10 else value

    fit_info = results.get('fit_info', {}) if isinstance(results, dict) else {}
    value = fit_info.get('stellar_mass_solar', np.nan)
    if np.isfinite(value) and value > 0:
        return float(value)
    return float(_get_defaults(results)['M'] / M_sun)


def _spectral_source(results, jd_day):
    """Classify a fitted spectral epoch from its saved row provenance."""
    daily_data = results.get('daily_data', {}) if isinstance(results, dict) else {}
    jd_key = _resolve_jd_key(daily_data, jd_day, tolerance=1e-3)
    if jd_key is None:
        return 'Other spectrum'
    frame = daily_data.get(jd_key)
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return 'Other spectrum'

    provenance = []
    for column in ('Instrument', 'Filename', 'Filter'):
        if column in frame.columns:
            provenance.extend(frame[column].dropna().astype(str).str.lower().tolist())
    joined = ' '.join(provenance)
    if 'xshooter' in joined or 'x-shooter' in joined:
        return 'XSHOOTER'
    if 'spex' in joined or 'irtf' in joined or '_sxd_' in joined or '_lxd_' in joined:
        return 'IRTF/SpeX'
    return 'Other spectrum'


def _spectral_plot_style(source):
    if source == 'XSHOOTER':
        return {
            'marker': 'D',
            'color_mdot': FINAL_PANEL_COLOURS['mdot_xshooter'],
            'color_av': FINAL_PANEL_COLOURS['av_xshooter'],
            'markersize': 6,
        }
    if source == 'IRTF/SpeX':
        return {
            'marker': 'x',
            'color_mdot': FINAL_PANEL_COLOURS['mdot_spectroscopic'],
            'color_av': FINAL_PANEL_COLOURS['av_spectroscopic'],
            'markersize': 7,
        }
    return {
        'marker': '+',
        'color_mdot': '0.35',
        'color_av': '0.35',
        'markersize': 7,
    }


def _mstar_mdot_solar2_per_year(M_star_solar, params):
    mdot = params.get('Mdot', np.nan) if isinstance(params, dict) else np.nan
    if not np.isfinite(mdot):
        log_mdot = params.get('logMdot', np.nan) if isinstance(params, dict) else np.nan
        if np.isfinite(log_mdot):
            mdot = 10 ** float(log_mdot) / M_sun * year
        else:
            return np.nan
    return M_star_solar * float(mdot)


def _mdot_error_solar_per_year(params):
    if not isinstance(params, dict):
        return np.nan
    mdot_err = params.get('Mdot_err', np.nan)
    try:
        if np.isfinite(mdot_err):
            return float(mdot_err)
    except (TypeError, ValueError):
        pass
    mdot = params.get('Mdot', np.nan)
    log_mdot_err = params.get('logMdot_err', np.nan)
    try:
        if np.isfinite(mdot) and np.isfinite(log_mdot_err):
            return float(mdot) * np.log(10.0) * float(log_mdot_err)
    except (TypeError, ValueError):
        pass
    return np.nan


def _av_error(params):
    if not isinstance(params, dict):
        return np.nan
    av_err = params.get('Av_err', np.nan)
    try:
        return float(av_err) if np.isfinite(av_err) else np.nan
    except (TypeError, ValueError):
        return np.nan


def _model_flux_for_freq(results, jd_day, frequencies):
    daily_params = results.get('daily_params', {}) if isinstance(results, dict) else {}
    jd_key = _resolve_jd_key(daily_params, jd_day)
    if jd_key is None:
        return None

    params = daily_params[jd_key]
    if 'logMdot' in params and np.isfinite(params['logMdot']):
        mdot_si = 10 ** float(params['logMdot'])
    elif 'Mdot' in params and np.isfinite(params['Mdot']):
        mdot_si = float(params['Mdot']) * M_sun / year
    else:
        return None

    av = float(params.get('Av', np.nan))
    if not np.isfinite(av):
        return None

    defaults = _get_defaults(results)
    log_flux = accretion_model(
        np.asarray(frequencies, dtype=float),
        mdot_si,
        defaults['M'],
        defaults['R_star'],
        defaults['R_out'],
        defaults['R_in'],
        defaults['distance'],
        av,
    )
    return np.exp(log_flux)


def _apply_final_panel_style():
    mpl.rcParams.update(FINAL_PANEL_STYLE)


def prepare_final_panel_residuals(
    df,
    results,
    fit_filters,
    residual_filter_groups=('I', 'J', 'H', 'K', 'L+W1'),
    exclude_interpolated=False,
):
    """Create grouped magnitude and flux residual tables for panels 3 and 4."""
    group_keys, group_map, residual_filters = _parse_filter_groups(residual_filter_groups)
    filter_to_group = {
        filter_name: group_key
        for group_key, filters in group_map.items()
        for filter_name in filters
    }

    daily_data_sel = get_daily_data(df, fit_filters)
    allowed_days = sorted(daily_data_sel.keys()) if daily_data_sel else []
    residual_df = df
    if exclude_interpolated and 'Interpolated' in df.columns:
        interpolated = df['Interpolated'].astype(str).str.strip().str.lower()
        residual_df = df[interpolated.eq('no')].copy()
    synthetic_df = generate_synthetic_data(results, residual_df, noise_level=0.0, random_seed=None)
    original_avg = average_daily_measurements(residual_df)
    synthetic_avg = average_daily_measurements(synthetic_df)

    original_avg['Mag_model_space'] = _mag_from_flux(original_avg)
    synthetic_avg['Mag_model_space'] = _mag_from_flux(synthetic_avg)

    original_avg = original_avg[
        original_avg['JD_day'].isin(allowed_days)
        & original_avg['Filter'].isin(residual_filters)
    ].copy()
    synthetic_avg = synthetic_avg[
        synthetic_avg['JD_day'].isin(allowed_days)
        & synthetic_avg['Filter'].isin(residual_filters)
    ].copy()

    orig_mag = original_avg[['JD_day', 'Filter', 'Mag_model_space', 'Magerr']].copy()
    synth_mag = synthetic_avg[['JD_day', 'Filter', 'Mag_model_space', 'Magerr']].copy()
    orig_mag = orig_mag.rename(columns={'Mag_model_space': 'Mag'})
    synth_mag = synth_mag.rename(columns={'Mag_model_space': 'Mag_synth', 'Magerr': 'Magerr_synth'})
    merged_mag = orig_mag.merge(synth_mag, on=['JD_day', 'Filter'], how='inner')
    merged_mag['Group'] = merged_mag['Filter'].map(filter_to_group).fillna(merged_mag['Filter'])
    merged_mag['Difference_mag'] = merged_mag['Mag'] - merged_mag['Mag_synth']
    merged_mag['Difference_err'] = np.sqrt(merged_mag['Magerr'] ** 2 + merged_mag['Magerr_synth'] ** 2)

    orig_flux = original_avg[['JD_day', 'Filter', 'Flux', 'Fluxerr']].copy()
    synth_flux = synthetic_avg[['JD_day', 'Filter', 'Flux', 'Fluxerr']].copy()
    synth_flux = synth_flux.rename(columns={'Flux': 'Flux_synth', 'Fluxerr': 'Fluxerr_synth'})
    merged_flux = orig_flux.merge(synth_flux, on=['JD_day', 'Filter'], how='inner')
    merged_flux['Group'] = merged_flux['Filter'].map(filter_to_group).fillna(merged_flux['Filter'])
    merged_flux['Difference_flux'] = merged_flux['Flux'] - merged_flux['Flux_synth']
    merged_flux['Difference_err_flux'] = np.sqrt(merged_flux['Fluxerr'] ** 2 + merged_flux['Fluxerr_synth'] ** 2)

    return {
        'group_keys': group_keys,
        'group_map': group_map,
        'filters': residual_filters,
        'magnitude': merged_mag,
        'flux': merged_flux,
        'flux_max_by_filter': (
            merged_flux.groupby('Filter')['Difference_flux']
            .apply(lambda s: np.max(np.abs(s)))
            .sort_values(ascending=False)
        ),
    }


def plot_final_panels_0_to_4(
    df,
    photometric_results,
    spectral_results=None,
    default_params=None,
    selected_filters=('J', 'H', 'K'),
    photometric_fit_filters=None,
    residual_filter_groups=('I', 'J', 'H', 'K', 'L+W1'),
    exclude_interpolated_residuals=False,
    plot_panel0_fits=True,
    filter_diff=True,
    show_panel3=True,
    date_range=None,
    flux_residual_log_scale=False,
    save_path=None,
    show=True,
    _close_when_hidden=True,
):
    """Create the stacked final panels 0-4 figure."""
    _apply_final_panel_style()
    spectral_results = spectral_results or {}
    default_params = default_params or {}
    photometric_fit_filters = list(
        photometric_fit_filters
        or photometric_results.get('fit_info', {}).get('fit_filters')
        or ['J', 'H', 'K']
    )

    daily_params = photometric_results['daily_params']
    jd_days = sorted(daily_params.keys())
    photo_m_star_solar = _stellar_mass_solar(photometric_results, default_params)
    mdot_values = [photo_m_star_solar * daily_params[jd]['Mdot'] for jd in jd_days]
    mdot_errors = [photo_m_star_solar * daily_params[jd]['Mdot_err'] for jd in jd_days]
    av_values = [daily_params[jd]['Av'] for jd in jd_days]
    av_errors = [daily_params[jd]['Av_err'] for jd in jd_days]

    df_filt, selected_filters, panel0_label_map = _resolve_selected_filter_data(df, selected_filters)
    df_filt = df_filt.copy()
    df_filt['Lambda'] = pd.to_numeric(df_filt['Lambda'], errors='coerce')
    df_filt = df_filt.sort_values(['Lambda', 'Filter', 'JD'])
    date_range = _normalise_date_range(date_range)
    filter_model_by_series = (
        calculate_filter_accretion_models(
            df_filt,
            photometric_results.get('daily_params', {}),
            photometric_results.get('global_params', {}),
        )
        if plot_panel0_fits
        else {}
    )

    residuals = prepare_final_panel_residuals(
        df,
        photometric_results,
        photometric_fit_filters,
        residual_filter_groups=residual_filter_groups,
        exclude_interpolated=exclude_interpolated_residuals,
    )
    print('Max |flux residual| by filter (Jy):')
    print(residuals['flux_max_by_filter'].to_string())

    label_fontsize = 14
    tick_fontsize = 12
    n_panels = 5 if show_panel3 else 4
    fig, axes = plt.subplots(n_panels, 1, figsize=(10, 4 * n_panels), sharex=True, gridspec_kw={'hspace': 0.0})
    axes = np.atleast_1d(axes)
    for index, ax in enumerate(axes):
        ax.tick_params(axis='both', labelsize=tick_fontsize)
        ax.tick_params(axis='x', which='both', bottom=index == len(axes) - 1, labelbottom=index == len(axes) - 1)

    ax = axes[0]
    if filter_diff:
        series_keys = sorted(
            df_filt.groupby(['Filter', 'Lambda'], sort=False).groups.keys(),
            key=lambda item: (_lambda_sort_key(item[1]), str(item[0])),
        )
    else:
        filter_lambdas = df_filt.groupby('Filter', sort=False)['Lambda'].min()
        series_keys = [
            filter_name
            for filter_name, _ in sorted(
                filter_lambdas.items(),
                key=lambda item: (_lambda_sort_key(item[1]), str(item[0])),
            )
        ]
    filter_cmap = mpl.colormaps['Dark2']
    filter_colors = filter_cmap(np.linspace(0, 1, max(len(series_keys), 1)))
    data_handles = []
    data_labels = []
    for index, series_key in enumerate(series_keys):
        if filter_diff:
            filter_name, lambda_value = series_key
            lambda_data = df_filt[
                (df_filt['Filter'] == filter_name)
                & (df_filt['Lambda'] == lambda_value)
            ].sort_values('JD')
            data_label = panel0_label_map.get(
                (str(filter_name), float(lambda_value)),
                fr'{filter_name} $\lambda={lambda_value:g}\,\mu m$',
            )
            model_series_keys = [(str(filter_name), float(lambda_value))]
        else:
            filter_name = series_key
            lambda_data = df_filt[df_filt['Filter'] == filter_name].sort_values(['JD', 'Lambda'])
            data_label = str(filter_name)
            model_series_keys = [
                (str(model_filter), float(model_lambda))
                for model_filter, model_lambda in filter_model_by_series
                if str(model_filter) == str(filter_name)
            ]
        lambda_color = filter_colors[index % len(filter_colors)]
        model_color = tuple((0.45 * np.array(lambda_color[:3])).tolist())
        data_handle = ax.errorbar(
            lambda_data['JD'],
            lambda_data['Mag'],
            yerr=lambda_data['Magerr'],
            fmt='o-',
            color=lambda_color,
            alpha=1.0,
            capsize=3,
            label=data_label,
        )
        data_handles.append(data_handle)
        data_labels.append(data_label)

        for model_series_key in model_series_keys:
            lambda_model = filter_model_by_series.get(model_series_key)
            if lambda_model is not None and len(lambda_model) > 0:
                ax.plot(lambda_model['JD'], lambda_model['Mag_model'], linestyle='--', color=model_color, alpha=0.75)
                direct_model = lambda_model[~lambda_model['Is_interpolated_fit']]
                interpolated_model = lambda_model[lambda_model['Is_interpolated_fit']]
                if len(direct_model) > 0:
                    ax.errorbar(
                        direct_model['JD'],
                        direct_model['Mag_model'],
                        yerr=direct_model.get('Mag_model_err'),
                        linestyle='None',
                        marker='P',
                        color=model_color,
                        alpha=0.75,
                        capsize=2,
                    )
                if len(interpolated_model) > 0:
                    ax.errorbar(
                        interpolated_model['JD'],
                        interpolated_model['Mag_model'],
                        yerr=interpolated_model.get('Mag_model_err'),
                        linestyle='None',
                        marker='x',
                        color=model_color,
                        alpha=0.75,
                        capsize=2,
                    )

    ax.set_ylabel(('Magnitude'), fontsize=label_fontsize)
    ax.invert_yaxis()
    if data_handles:
        data_legend = ax.legend(data_handles, data_labels, title='Data', loc='lower left')
        ax.add_artist(data_legend)
    ax.grid(True, alpha=0.3)
    add_gregorian_top_axis(ax, n_ticks=6, fontsize=12, use_grid_ticks=True, dynamic=True)

    ax = axes[1]
    ax.errorbar(jd_days, mdot_values, yerr=mdot_errors, fmt='o-', color=FINAL_PANEL_COLOURS['mdot_photometric'], capsize=3, label='Photometric fit', zorder=2)
    ax.set_ylabel(r'$M_{\star} \; \dot{M}\ \left(M_{\odot}^2\ \mathrm{yr}^{-1}\right)$', fontsize=label_fontsize)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    spectral_jds = []
    spect_m_star_solar = _stellar_mass_solar(spectral_results, default_params)
    if 'daily_params' in spectral_results:
        spectral_jds = sorted(spectral_results['daily_params'].keys())
        spectral_sources = {jd: _spectral_source(spectral_results, jd) for jd in spectral_jds}
        for source in ('IRTF/SpeX', 'XSHOOTER', 'Other spectrum'):
            source_jds = [jd for jd in spectral_jds if spectral_sources[jd] == source]
            if not source_jds:
                continue
            style = _spectral_plot_style(source)
            source_mdots = [
                spect_m_star_solar * spectral_results['daily_params'][jd]['Mdot']
                for jd in source_jds
            ]
            source_mdot_errors = [
                spect_m_star_solar * _mdot_error_solar_per_year(spectral_results['daily_params'][jd])
                for jd in source_jds
            ]
            ax.errorbar(
                source_jds,
                source_mdots,
                yerr=source_mdot_errors,
                marker=style['marker'],
                linestyle='None',
                color=style['color_mdot'],
                markersize=style['markersize'],
                markeredgewidth=1.8,
                capsize=3,
                label=source,
                zorder=5 if source == 'XSHOOTER' else 4,
            )
        ax.legend()

    ax = axes[2]
    ax.errorbar(jd_days, av_values, yerr=av_errors, fmt='o-', color=FINAL_PANEL_COLOURS['av_photometric'], capsize=3, label='Photometric fit', zorder=2)
    ax.set_ylabel(r'$A_V\ \mathrm{(mag)}$', fontsize=label_fontsize)
    ax.grid(True, alpha=0.3)
    if spectral_jds:
        for source in ('IRTF/SpeX', 'XSHOOTER', 'Other spectrum'):
            source_jds = [jd for jd in spectral_jds if spectral_sources[jd] == source]
            if not source_jds:
                continue
            style = _spectral_plot_style(source)
            source_avs = [spectral_results['daily_params'][jd]['Av'] for jd in source_jds]
            source_av_errors = [_av_error(spectral_results['daily_params'][jd]) for jd in source_jds]
            ax.errorbar(
                source_jds,
                source_avs,
                yerr=source_av_errors,
                marker=style['marker'],
                linestyle='None',
                color=style['color_av'],
                markersize=style['markersize'],
                markeredgewidth=1.8,
                capsize=3,
                label=source,
                zorder=5 if source == 'XSHOOTER' else 4,
            )
        ax.legend()

    residual_palette = FINAL_PANEL_COLOURS['residual_palette']
    filter_colors = {
        group_key: residual_palette[index % len(residual_palette)]
        for index, group_key in enumerate(residuals['group_keys'])
    }

    if show_panel3:
        ax = axes[3]
        for group_key in residuals['group_keys']:
            group_data = residuals['magnitude'][residuals['magnitude']['Group'] == group_key].sort_values('JD_day')
            if len(group_data) == 0:
                continue
            group_filters = residuals['group_map'][group_key]
            is_fit_filter = all(f in photometric_fit_filters for f in group_filters)
            ax.errorbar(
                group_data['JD_day'],
                group_data['Difference_mag'],
                yerr=group_data['Difference_err'],
                fmt='o-',
                linestyle='-' if is_fit_filter else '--',
                color=filter_colors[group_key],
                linewidth=1.8,
                markersize=4,
                alpha=0.75,
                capsize=2,
                label=f'{group_key} ({"fit" if is_fit_filter else "plot-only"})',
            )
        ax.axhline(0, color='black', linestyle='--', alpha=0.7)
        ax.set_ylabel('Magnitude Residual (Mag)', fontsize=label_fontsize)
        ax.invert_yaxis()
        ax.legend(loc='best', fontsize=12)
        ax.grid(True, alpha=0.3)

    ax = axes[4 if show_panel3 else 3]
    for group_key in residuals['group_keys']:
        group_data = residuals['flux'][residuals['flux']['Group'] == group_key].sort_values('JD_day')
        if len(group_data) == 0:
            continue
        group_filters = residuals['group_map'][group_key]
        is_fit_filter = all(f in photometric_fit_filters for f in group_filters)
        raw_flux = pd.to_numeric(group_data['Difference_flux'], errors='coerce').to_numpy(dtype=float)
        raw_jd = pd.to_numeric(group_data['JD_day'], errors='coerce').to_numpy(dtype=float)
        nonzero = np.isfinite(raw_flux) & np.isfinite(raw_jd) & (raw_flux != 0)
        if not np.any(nonzero):
            continue
        plot_jd = raw_jd[nonzero]
        plot_flux = np.abs(raw_flux[nonzero])

        if flux_residual_log_scale:
            plot_err = pd.to_numeric(group_data['Difference_err_flux'], errors='coerce').to_numpy(dtype=float)[nonzero]
            ax.errorbar(
                plot_jd,
                plot_flux,
                yerr=plot_err,
                fmt='o-',
                linestyle='-' if is_fit_filter else '--',
                color=filter_colors[group_key],
                linewidth=1.8,
                markersize=4,
                alpha=0.65,
                capsize=2,
                zorder=2,
                label=f'{group_key} ({"fit" if is_fit_filter else "plot-only"})',
            )
            positive = raw_flux[nonzero] > 0
            negative = raw_flux[nonzero] < 0
            if np.any(positive):
                ax.scatter(
                    plot_jd[positive],
                    plot_flux[positive],
                    facecolors=filter_colors[group_key],
                    edgecolors='black',
                    marker='o',
                    s=40,
                    alpha=0.85,
                    zorder=3,
                )
            if np.any(negative):
                ax.scatter(
                    plot_jd[negative],
                    plot_flux[negative],
                    facecolors='none',
                    edgecolors=filter_colors[group_key],
                    marker='o',
                    s=60,
                    alpha=0.9,
                    zorder=4,
                )
        else:
            ax.errorbar(
                group_data['JD_day'],
                group_data['Difference_flux'],
                yerr=group_data['Difference_err_flux'],
                fmt='o-',
                linestyle='-' if is_fit_filter else '--',
                color=filter_colors[group_key],
                linewidth=1.8,
                markersize=4,
                alpha=0.75,
                capsize=2,
                label=f'{group_key} ({"fit" if is_fit_filter else "plot-only"})',
            )

    if flux_residual_log_scale:
        ax.set_yscale('log')
        ax.set_ylabel('Flux Residual |ΔFlux| (Jy)', fontsize=label_fontsize)
        ax.text(
            0.98,
            0.96,
            'Open markers = negative residuals',
            transform=ax.transAxes,
            ha='right',
            va='top',
            fontsize=10,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=4),
        )
    else:
        ax.axhline(0, color='black', linestyle='--', alpha=0.7)
        ax.set_ylabel('Flux Residual (Jy)', fontsize=label_fontsize)
    ax.set_xlabel('Julian Date', fontsize=label_fontsize)
    ax.legend(loc='best', fontsize=12)
    ax.grid(True, alpha=0.3)

    if date_range is not None:
        current_xlim = axes[-1].get_xlim()
        xmin = current_xlim[0] if date_range[0] is None else date_range[0]
        xmax = current_xlim[1] if date_range[1] is None else date_range[1]
        if xmin >= xmax:
            raise ValueError("date_range lower limit must be less than the upper limit.")
        axes[-1].set_xlim(xmin, xmax)

    fig.subplots_adjust(hspace=0.0, top=0.95, bottom=0.07)
    if save_path:
        fig.savefig(_generated_figure_path(save_path), dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    elif _close_when_hidden:
        plt.close(fig)
    return fig, axes, residuals


def plot_final_panels_0_to_4_layouts(
    df,
    photometric_results,
    spectral_results=None,
    default_params=None,
    selected_filters=('J', 'H', 'K'),
    photometric_fit_filters=None,
    residual_filter_groups=('I', 'J', 'H', 'K', 'L+W1'),
    exclude_interpolated_residuals=False,
    plot_panel0_fits=True,
    filter_diff=True,
    show_panel3=True,
    end_zoom_range=(2459000, 2461300),
    flux_residual_log_scale=False,
    one_column_save_path=None,
    two_column_save_path=None,
    show=True,
    raster_dpi=300,
):
    """Save the full-timeline and full-plus-end-zoom layouts for panels 0-4.

    The one-column version contains only the complete time baseline.  The
    two-column version places that same baseline on the left and a matching
    end-of-timeline zoom on the right.  The saved comparison embeds
    high-resolution renders of the two otherwise identical panel stacks so
    their rows remain perfectly aligned.  On an interactive widget backend,
    the two original canvases are displayed side by side instead, because an
    image-only comparison canvas cannot provide data-aware pan/zoom.
    """
    common_kwargs = {
        'spectral_results': spectral_results,
        'default_params': default_params,
        'selected_filters': selected_filters,
        'photometric_fit_filters': photometric_fit_filters,
        'residual_filter_groups': residual_filter_groups,
        'exclude_interpolated_residuals': exclude_interpolated_residuals,
        'plot_panel0_fits': plot_panel0_fits,
        'filter_diff': filter_diff,
        'show_panel3': show_panel3,
        'flux_residual_log_scale': flux_residual_log_scale,
        'show': False,
        # The widget layout below needs live canvas communication channels.
        # Closing an ipympl figure destroys those channels.
        '_close_when_hidden': False,
    }
    # Suppress the backend's automatic standalone display; these canvases are
    # explicitly arranged below when interactive output is requested.
    with plt.ioff():
        full_fig, full_axes, residuals = plot_final_panels_0_to_4(
            df,
            photometric_results,
            date_range=None,
            save_path=one_column_save_path,
            **common_kwargs,
        )
        zoom_fig, zoom_axes, _ = plot_final_panels_0_to_4(
            df,
            photometric_results,
            date_range=end_zoom_range,
            save_path=None,
            **common_kwargs,
        )

    def _render_figure(figure):
        buffer = io.BytesIO()
        figure.savefig(buffer, format='png', dpi=raster_dpi, bbox_inches='tight')
        buffer.seek(0)
        image = plt.imread(buffer, format='png')
        buffer.close()
        return image

    full_image = _render_figure(full_fig)
    zoom_image = _render_figure(zoom_fig)
    n_panels = 5 if show_panel3 else 4
    comparison_fig, comparison_axes = plt.subplots(1, 2, figsize=(20, 4 * n_panels))
    for axis, image in zip(comparison_axes, (full_image, zoom_image)):
        axis.imshow(image)
        axis.set_axis_off()
    comparison_fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0, wspace=0.015)

    if two_column_save_path:
        comparison_fig.savefig(
            _generated_figure_path(two_column_save_path),
            dpi=raster_dpi,
            bbox_inches='tight',
            pad_inches=0.02,
        )
    if show:
        backend = str(plt.get_backend()).lower()
        is_widget_backend = 'widget' in backend or 'ipympl' in backend or 'nbagg' in backend
        if is_widget_backend:
            try:
                from IPython.display import display
                from ipywidgets import HBox, Layout

                # ``comparison_fig`` contains two PNG artists for exact PDF
                # alignment.  Displaying the live source canvases preserves
                # Matplotlib's toolbar navigation and coordinate transforms.
                plt.close(comparison_fig)
                full_fig.canvas.layout = Layout(width='50%')
                zoom_fig.canvas.layout = Layout(width='50%')
                with plt.ioff():
                    display(HBox([full_fig.canvas, zoom_fig.canvas], layout=Layout(width='100%')))
            except (ImportError, AttributeError, TypeError):
                # Non-widget notebook configurations still get a useful
                # display, albeit without a side-by-side widget container.
                from IPython.display import display

                display(full_fig)
                display(zoom_fig)
        else:
            plt.close(full_fig)
            plt.close(zoom_fig)
            plt.show()
    else:
        plt.close(full_fig)
        plt.close(zoom_fig)
        plt.close(comparison_fig)
    return full_fig, comparison_fig, residuals


def plot_filter_evolution_panel(
    df,
    photometric_results,
    default_params=None,
    selected_filters=('J', 'H', 'K'),
    plot_fits=True,
    filter_diff=True,
    date_range=None,
    save_path=None,
    show=True,
):
    """Plot the flexible final-panel photometry/model evolution panel."""
    _apply_final_panel_style()
    default_params = default_params or {}
    df_filt, selected_filters, panel0_label_map = _resolve_selected_filter_data(df, selected_filters)
    df_filt = df_filt.copy()
    df_filt['Lambda'] = pd.to_numeric(df_filt['Lambda'], errors='coerce')
    df_filt = df_filt.sort_values(['Lambda', 'Filter', 'JD'])
    date_range = _normalise_date_range(date_range)

    filter_model_by_series = (
        calculate_filter_accretion_models(
            df_filt,
            photometric_results.get('daily_params', {}),
            photometric_results.get('global_params', {}),
        )
        if plot_fits
        else {}
    )

    if filter_diff:
        series_keys = sorted(
            df_filt.groupby(['Filter', 'Lambda'], sort=False).groups.keys(),
            key=lambda item: (_lambda_sort_key(item[1]), str(item[0])),
        )
    else:
        filter_lambdas = df_filt.groupby('Filter', sort=False)['Lambda'].min()
        series_keys = [
            filter_name
            for filter_name, _ in sorted(
                filter_lambdas.items(),
                key=lambda item: (_lambda_sort_key(item[1]), str(item[0])),
            )
        ]

    fig, ax = plt.subplots(figsize=(10, 4.8))
    filter_cmap = mpl.colormaps['Dark2']
    filter_colors = filter_cmap(np.linspace(0, 1, max(len(series_keys), 1)))
    data_handles = []
    data_labels = []

    for index, series_key in enumerate(series_keys):
        if filter_diff:
            filter_name, lambda_value = series_key
            lambda_data = df_filt[
                (df_filt['Filter'] == filter_name)
                & (df_filt['Lambda'] == lambda_value)
            ].sort_values('JD')
            data_label = panel0_label_map.get(
                (str(filter_name), float(lambda_value)),
                fr'{filter_name} $\lambda={lambda_value:g}\,\mu m$',
            )
            model_series_keys = [(str(filter_name), float(lambda_value))]
        else:
            filter_name = series_key
            lambda_data = df_filt[df_filt['Filter'] == filter_name].sort_values(['JD', 'Lambda'])
            data_label = str(filter_name)
            model_series_keys = [
                (str(model_filter), float(model_lambda))
                for model_filter, model_lambda in filter_model_by_series
                if str(model_filter) == str(filter_name)
            ]

        lambda_color = filter_colors[index % len(filter_colors)]
        model_color = tuple((0.45 * np.array(lambda_color[:3])).tolist())
        data_handle = ax.errorbar(
            lambda_data['JD'],
            lambda_data['Mag'],
            yerr=lambda_data['Magerr'],
            fmt='o-',
            color=lambda_color,
            alpha=1.0,
            capsize=3,
            label=data_label,
        )
        data_handles.append(data_handle)
        data_labels.append(data_label)

        for model_series_key in model_series_keys:
            lambda_model = filter_model_by_series.get(model_series_key)
            if lambda_model is None or len(lambda_model) == 0:
                continue
            ax.plot(lambda_model['JD'], lambda_model['Mag_model'], linestyle='--', color=model_color, alpha=0.75)
            direct_model = lambda_model[~lambda_model['Is_interpolated_fit']]
            interpolated_model = lambda_model[lambda_model['Is_interpolated_fit']]
            if len(direct_model) > 0:
                ax.errorbar(
                    direct_model['JD'],
                    direct_model['Mag_model'],
                    yerr=direct_model.get('Mag_model_err'),
                    linestyle='None',
                    marker='P',
                    color=model_color,
                    alpha=0.75,
                    capsize=2,
                )
            if len(interpolated_model) > 0:
                ax.errorbar(
                    interpolated_model['JD'],
                    interpolated_model['Mag_model'],
                    yerr=interpolated_model.get('Mag_model_err'),
                    linestyle='None',
                    marker='x',
                    color=model_color,
                    alpha=0.75,
                    capsize=2,
                )

    ax.set_ylabel('Magnitude')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Julian Date')
    if data_handles:
        ax.legend(data_handles, data_labels, title='Data', loc='best')
    add_gregorian_top_axis(ax, n_ticks=6, fontsize=12, use_grid_ticks=True, dynamic=True)

    if date_range is not None:
        current_xlim = ax.get_xlim()
        xmin = current_xlim[0] if date_range[0] is None else date_range[0]
        xmax = current_xlim[1] if date_range[1] is None else date_range[1]
        if xmin >= xmax:
            raise ValueError("date_range lower limit must be less than the upper limit.")
        ax.set_xlim(xmin, xmax)

    fig.tight_layout()
    if save_path:
        fig.savefig(_generated_figure_path(save_path), dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, ax


def plot_final_residuals(
    df,
    photometric_results,
    fit_filters=None,
    residual_filter_groups=('I', 'J', 'H', 'K', 'L+W1'),
    show_magnitude=True,
    show_flux=True,
    flux_residual_log_scale=False,
    date_range=None,
    save_path=None,
    show=True,
):
    """Plot final-panel residuals in one compact figure."""
    _apply_final_panel_style()
    fit_filters = list(
        fit_filters
        or photometric_results.get('fit_info', {}).get('fit_filters')
        or ['J', 'H', 'K']
    )
    residuals = prepare_final_panel_residuals(
        df,
        photometric_results,
        fit_filters,
        residual_filter_groups=residual_filter_groups,
    )
    date_range = _normalise_date_range(date_range)

    panels = []
    if show_magnitude:
        panels.append('magnitude')
    if show_flux:
        panels.append('flux')
    if not panels:
        raise ValueError('At least one of show_magnitude or show_flux must be True.')

    fig, axes = plt.subplots(len(panels), 1, figsize=(10, 4 * len(panels)), sharex=True)
    axes = np.atleast_1d(axes)
    residual_palette = FINAL_PANEL_COLOURS['residual_palette']
    filter_colors = {
        group_key: residual_palette[index % len(residual_palette)]
        for index, group_key in enumerate(residuals['group_keys'])
    }

    for ax, panel in zip(axes, panels):
        for group_key in residuals['group_keys']:
            group_filters = residuals['group_map'][group_key]
            is_fit_filter = all(f in fit_filters for f in group_filters)

            if panel == 'magnitude':
                group_data = residuals['magnitude'][residuals['magnitude']['Group'] == group_key].sort_values('JD_day')
                if len(group_data) == 0:
                    continue
                ax.errorbar(
                    group_data['JD_day'],
                    group_data['Difference_mag'],
                    yerr=group_data['Difference_err'],
                    fmt='o-',
                    linestyle='-' if is_fit_filter else '--',
                    color=filter_colors[group_key],
                    linewidth=1.8,
                    markersize=4,
                    alpha=0.75,
                    capsize=2,
                    label=f'{group_key} ({"fit" if is_fit_filter else "plot-only"})',
                )
            else:
                group_data = residuals['flux'][residuals['flux']['Group'] == group_key].sort_values('JD_day')
                if len(group_data) == 0:
                    continue
                raw_flux = pd.to_numeric(group_data['Difference_flux'], errors='coerce').to_numpy(dtype=float)
                raw_jd = pd.to_numeric(group_data['JD_day'], errors='coerce').to_numpy(dtype=float)
                nonzero = np.isfinite(raw_flux) & np.isfinite(raw_jd) & (raw_flux != 0)
                if not np.any(nonzero):
                    continue

                if flux_residual_log_scale:
                    plot_jd = raw_jd[nonzero]
                    plot_flux = np.abs(raw_flux[nonzero])
                    plot_err = pd.to_numeric(group_data['Difference_err_flux'], errors='coerce').to_numpy(dtype=float)[nonzero]
                    ax.errorbar(
                        plot_jd,
                        plot_flux,
                        yerr=plot_err,
                        fmt='o-',
                        linestyle='-' if is_fit_filter else '--',
                        color=filter_colors[group_key],
                        linewidth=1.8,
                        markersize=4,
                        alpha=0.65,
                        capsize=2,
                        zorder=2,
                        label=f'{group_key} ({"fit" if is_fit_filter else "plot-only"})',
                    )
                    positive = raw_flux[nonzero] > 0
                    negative = raw_flux[nonzero] < 0
                    if np.any(positive):
                        ax.scatter(plot_jd[positive], plot_flux[positive], facecolors=filter_colors[group_key], edgecolors='black', marker='o', s=40, alpha=0.85, zorder=3)
                    if np.any(negative):
                        ax.scatter(plot_jd[negative], plot_flux[negative], facecolors='none', edgecolors=filter_colors[group_key], marker='o', s=60, alpha=0.9, zorder=4)
                else:
                    ax.errorbar(
                        group_data['JD_day'],
                        group_data['Difference_flux'],
                        yerr=group_data['Difference_err_flux'],
                        fmt='o-',
                        linestyle='-' if is_fit_filter else '--',
                        color=filter_colors[group_key],
                        linewidth=1.8,
                        markersize=4,
                        alpha=0.75,
                        capsize=2,
                        label=f'{group_key} ({"fit" if is_fit_filter else "plot-only"})',
                    )

        if panel == 'magnitude':
            ax.axhline(0, color='black', linestyle='--', alpha=0.7)
            ax.set_ylabel('Magnitude Residual (Mag)')
            ax.invert_yaxis()
        elif flux_residual_log_scale:
            ax.set_yscale('log')
            ax.set_ylabel('Flux Residual |Delta Flux| (Jy)')
            ax.text(
                0.98,
                0.96,
                'Open markers = negative residuals',
                transform=ax.transAxes,
                ha='right',
                va='top',
                fontsize=10,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=4),
            )
        else:
            ax.axhline(0, color='black', linestyle='--', alpha=0.7)
            ax.set_ylabel('Flux Residual (Jy)')
        ax.legend(loc='best', fontsize=12)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Julian Date')
    add_gregorian_top_axis(axes[0], n_ticks=6, fontsize=12, use_grid_ticks=True, dynamic=True)

    if date_range is not None:
        current_xlim = axes[-1].get_xlim()
        xmin = current_xlim[0] if date_range[0] is None else date_range[0]
        xmax = current_xlim[1] if date_range[1] is None else date_range[1]
        if xmin >= xmax:
            raise ValueError("date_range lower limit must be less than the upper limit.")
        axes[-1].set_xlim(xmin, xmax)

    fig.tight_layout()
    if save_path:
        fig.savefig(_generated_figure_path(save_path), dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, axes, residuals


def plot_panel5_sed(
    sed_jd,
    photometric_results,
    spectral_results,
    default_params=None,
    spectral_df=None,
    photometric_filters=None,
    show_range=(0.78, 3.5),
    sed_date_str=None,
    save_dir='results/generated/pictures',
    show=True,
):
    """Create one Panel 5 SED plot with spectral and photometric residuals."""
    _apply_final_panel_style()
    sed_jd = float(sed_jd)
    if sed_date_str is None:
        sed_date_str = julian_to_calendar([sed_jd])[0]

    default_params = default_params or {}
    photo_m_star_solar = _stellar_mass_solar(photometric_results, default_params)
    spect_m_star_solar = photo_m_star_solar

    spectral_series = _prepare_day_series_from_frame(spectral_df, sed_jd)
    if spectral_series is None:
        spectral_series = _prepare_day_series(spectral_results, sed_jd)

    if photometric_filters is None:
        photometric_filters = photometric_results.get('fit_info', {}).get('fit_filters') or ['J', 'H', 'K']
    phot_series = _prepare_day_series(photometric_results, sed_jd, filters=photometric_filters)

    if show_range is not None:
        xmin, xmax = sorted([float(show_range[0]), float(show_range[1])])
    else:
        xmin = xmax = None

    for series_name, series in [('spectral', spectral_series), ('phot', phot_series)]:
        if series is None or xmin is None or xmax is None:
            continue
        mask = (series['lambda_um'] >= xmin) & (series['lambda_um'] <= xmax)
        for key in ('lambda_m', 'lambda_um', 'frequencies', 'flux', 'fluxerr', 'filters'):
            series[key] = series[key][mask]
        if len(series['lambda_um']) == 0:
            if series_name == 'spectral':
                spectral_series = None
            else:
                phot_series = None

    spectral_model_data = _model_flux_for_freq(spectral_results, sed_jd, spectral_series['frequencies']) if spectral_series is not None else None
    phot_model_data = _model_flux_for_freq(photometric_results, sed_jd, phot_series['frequencies']) if phot_series is not None else None
    spectral_residual = spectral_series['flux'] - spectral_model_data if spectral_model_data is not None else None
    phot_residual = phot_series['flux'] - phot_model_data if phot_model_data is not None else None
    if phot_residual is not None:
        finite_phot_residual = phot_residual[np.isfinite(phot_residual)]
        if finite_phot_residual.size:
            print(f'Max |SED photometric residual| at fitted bands: {np.max(np.abs(finite_phot_residual)):.3e} Jy')

    spectral_lambda_grid_um, spectral_model_grid = _model_grid_for_day(spectral_results, sed_jd, spectral_series, xmin, xmax, n_grid=1200)
    phot_lambda_grid_um, phot_model_grid = _model_grid_for_day(photometric_results, sed_jd, phot_series or spectral_series, xmin, xmax, n_grid=900)
    if phot_model_grid is not None and phot_series is not None:
        phot_lambda_grid_um = np.unique(np.concatenate([phot_lambda_grid_um, phot_series['lambda_um']]))
        phot_lambda_grid_um.sort()
        phot_model_grid = _model_flux_for_freq(photometric_results, sed_jd, constants.c / (phot_lambda_grid_um * 1e-6))

    has_spectral = spectral_series is not None
    has_phot = phot_series is not None
    if not has_spectral and not has_phot:
        print(f'No spectral or photometric daily data available for JD {sed_jd}')
        return None, None

    cmap_dark2 = mpl.colormaps['Dark2_r']
    spectral_data_color = cmap_dark2(0.10)
    photometric_data_color = cmap_dark2(0.18)
    spectral_fit_color = cmap_dark2(0.58)
    photometric_fit_color = cmap_dark2(0.66)
    label_fontsize = 14
    tick_fontsize = 12

    fig, (ax_sed, ax_res) = plt.subplots(
        2,
        1,
        figsize=(10, 7),
        sharex=True,
        gridspec_kw={'height_ratios': [3.0, 1.4], 'hspace': 0.0},
    )
    ax_sed.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    ax_sed.tick_params(axis='both', labelsize=tick_fontsize)

    if has_spectral:
        ax_sed.plot(spectral_series['lambda_um'], spectral_series['flux'], color=spectral_data_color, linewidth=1.2, alpha=0.85, label='Spectral data')
        _plot_spectral_uncertainty(ax_sed, spectral_series, spectral_data_color)
        if spectral_model_grid is not None:
            ax_sed.plot(spectral_lambda_grid_um, spectral_model_grid, color=spectral_fit_color, linewidth=2.0, alpha=0.95, label='Spectral fit')
        elif spectral_model_data is not None:
            ax_sed.plot(spectral_series['lambda_um'], spectral_model_data, color=spectral_fit_color, linewidth=2.0, alpha=0.95, label='Spectral fit')

    if has_phot:
        _plot_photometry(ax_sed, phot_series, photometric_data_color)
        if phot_model_grid is not None:
            ax_sed.plot(phot_lambda_grid_um, phot_model_grid, color=photometric_fit_color, linewidth=2.0, alpha=0.95, label='Photometric fit')
        elif phot_model_data is not None:
            ax_sed.plot(phot_series['lambda_um'], phot_model_data, color=photometric_fit_color, linewidth=2.0, alpha=0.95, label='Photometric fit')

    y_arrays = []
    for item in (phot_series, spectral_series):
        if item is not None:
            y_arrays.append(item['flux'])
    for item in (phot_model_grid, spectral_model_grid):
        if item is not None:
            y_arrays.append(item)
    y_values = np.concatenate([np.asarray(item, dtype=float) for item in y_arrays])
    y_values = y_values[np.isfinite(y_values) & (y_values > 0)]
    if y_values.size:
        ax_sed.set_ylim(np.nanmin(y_values) * 0.9, np.nanmax(y_values) * 1.1)
    ax_sed.set_ylabel('Flux density (Jy)', fontsize=label_fontsize)
    ax_sed.set_yscale('log')
    ax_sed.grid(True, alpha=0.3)
    ax_sed.legend(loc='best')

    photo_jd_key = _resolve_jd_key(photometric_results.get('daily_params', {}), sed_jd)
    spect_jd_key = _resolve_jd_key(spectral_results.get('daily_params', {}), sed_jd)
    title_bits = [rf'\mathrm{{{str(sed_date_str).replace("-", r"{-}")}}}\ |\ \mathrm{{JD}}={sed_jd:.1f}']
    if photo_jd_key is not None:
        params = photometric_results['daily_params'][photo_jd_key]
        phot_mstar_mdot = _mstar_mdot_solar2_per_year(photo_m_star_solar, params)
        title_bits.append(rf'\mathrm{{Phot}}:\ A_V={params.get("Av", np.nan):.2f},\ M_{{\star}}\dot{{M}}={phot_mstar_mdot:.2e}\ M_\odot^2\,\mathrm{{yr}}^{{-1}}')
    if spect_jd_key is not None:
        params = spectral_results['daily_params'][spect_jd_key]
        spect_mstar_mdot = _mstar_mdot_solar2_per_year(spect_m_star_solar, params)
        title_bits.append(rf'\mathrm{{Spec}}:\ A_V={params.get("Av", np.nan):.2f},\ M_{{\star}}\dot{{M}}={spect_mstar_mdot:.2e}\ M_\odot^2\,\mathrm{{yr}}^{{-1}}')
    ax_sed.set_title('$' + r'\ |\ '.join(title_bits) + '$', fontsize=16)

    if has_spectral and spectral_residual is not None:
        ax_res.plot(spectral_series['lambda_um'], spectral_residual, color=spectral_data_color, linewidth=1.0, alpha=0.7, label='Spectral residual (data - fit)')
    if has_phot and phot_residual is not None:
        _plot_photometry(ax_res, {**phot_series, 'flux': phot_residual}, photometric_data_color, label='Photometric residual (data - fit)', annotate=False, residual=True)

    ax_res.axhline(0.0, color='black', linestyle='--', linewidth=0.9, alpha=0.8)
    ax_res.set_ylabel('Delta Flux density (Jy)', fontsize=label_fontsize)
    ax_res.set_xlabel('Wavelength (um)', fontsize=label_fontsize)
    ax_res.tick_params(axis='both', labelsize=tick_fontsize)
    ax_res.grid(True, alpha=0.3)
    ax_res.legend(loc='best', fontsize=12)
    if xmin is not None and xmax is not None:
        ax_sed.set_xlim(xmin, xmax)
        ax_res.set_xlim(xmin, xmax)

    fig.subplots_adjust(hspace=0.0)
    save_dir = Path(_generated_output_dir() if save_dir == 'results/generated/pictures' else save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    jd_label = f'{sed_jd:.1f}'.replace('.', 'p')
    save_path = save_dir / f'SED_{sed_date_str}_JD_{jd_label}.pdf'
    fig.savefig(save_path, dpi=300, format='pdf', bbox_inches='tight')
    print(f'Panel 5 SED saved to {save_path}')
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, save_path


def _model_grid_for_day(results, jd_day, series, xmin, xmax, n_grid):
    if _resolve_jd_key(results.get('daily_params', {}), jd_day) is None:
        return None, None
    if xmin is not None and xmax is not None:
        grid_min, grid_max = xmin, xmax
    elif series is not None:
        grid_min = float(np.nanmin(series['lambda_um']))
        grid_max = float(np.nanmax(series['lambda_um']))
    else:
        return None, None
    if not np.isfinite(grid_min) or not np.isfinite(grid_max) or grid_max <= grid_min:
        return None, None
    lambda_grid_um = np.linspace(grid_min, grid_max, n_grid)
    freq_grid = constants.c / (lambda_grid_um * 1e-6)
    return lambda_grid_um, _model_flux_for_freq(results, jd_day, freq_grid)


def _plot_spectral_uncertainty(ax, series, color):
    spectral_err = np.asarray(series['fluxerr'], dtype=float)
    spectral_flux = np.asarray(series['flux'], dtype=float)
    good_err = (
        np.isfinite(spectral_err)
        & (spectral_err > 0)
        & np.isfinite(spectral_flux)
        & (spectral_flux > 0)
        & (spectral_err <= spectral_flux)
    )
    if np.any(good_err):
        lower = np.clip(spectral_flux[good_err] - spectral_err[good_err], 1e-30, None)
        upper = spectral_flux[good_err] + spectral_err[good_err]
        ax.fill_between(series['lambda_um'][good_err], lower, upper, color=color, alpha=0.15, linewidth=0)


def _plot_photometry(ax, series, color, label='Photometric data', annotate=True, residual=False):
    yerr = series['fluxerr'] if np.any(np.isfinite(series['fluxerr'])) else None
    if yerr is not None:
        ax.errorbar(
            series['lambda_um'],
            series['flux'],
            yerr=yerr,
            fmt='o',
            color=color,
            markersize=5 if residual else 6,
            capsize=3,
            linestyle='None',
            elinewidth=1.0,
            label=label,
        )
    else:
        ax.plot(series['lambda_um'], series['flux'], linestyle='None', marker='o', color=color, markersize=5 if residual else 6, label=label)

    if annotate:
        for x_um, y_flux, fname in zip(series['lambda_um'], series['flux'], series['filters']):
            if np.isfinite(x_um) and np.isfinite(y_flux):
                ax.annotate(str(fname), (x_um, y_flux), textcoords='offset points', xytext=(4, 4), fontsize=10, color=color, alpha=0.9)


def plot_panel5_seds(sed_jds, photometric_results, spectral_results, **kwargs):
    """Create Panel 5 SED plots for one or more JDs."""
    sed_jds = _normalise_sed_jds(sed_jds)
    sed_date_strs = kwargs.pop('sed_date_strs', None)
    if sed_date_strs is None:
        sed_date_strs = julian_to_calendar(sed_jds)
    print(sed_date_strs)
    return [
        plot_panel5_sed(jd, photometric_results, spectral_results, sed_date_str=date_str, **kwargs)
        for jd, date_str in zip(sed_jds, sed_date_strs)
    ]


def fit_parameter_table(result_sets, m_star_solar=0.5):
    """Return paper-ready Mdot, Mstar*Mdot, and Av rows from saved fits.

    Parameters
    ----------
    result_sets : mapping or fitting-result dict
        A mapping of dataset labels to result dictionaries, or one result dict.
    m_star_solar : float
        Stellar mass in solar masses.  The paper plots ``Mstar*Mdot``; the
        fitted ``Mdot`` values themselves are already in solar masses per year.
    """
    if isinstance(result_sets, dict) and 'daily_params' in result_sets:
        result_sets = {'fit': result_sets}
    if not isinstance(result_sets, dict):
        raise TypeError('result_sets must be a fitting-result dict or label-to-result mapping')

    rows = []
    for dataset, results in result_sets.items():
        if not isinstance(results, dict):
            continue
        daily_params = results.get('daily_params', {})
        date_lookup = {}
        for jd_key, day_df in results.get('daily_data', {}).items():
            if isinstance(day_df, pd.DataFrame) and len(day_df) and 'Date' in day_df.columns:
                date_lookup[float(jd_key)] = str(day_df['Date'].iloc[0])[:10]

        for jd, params in sorted(daily_params.items(), key=lambda item: float(item[0])):
            jd = float(jd)
            mdot = params.get('Mdot')
            if mdot is None and params.get('logMdot') is not None:
                mdot = 10.0 ** float(params['logMdot']) / M_sun * year
            mdot = float(mdot)
            mdot_err = _mdot_error_solar_per_year(params)
            rows.append({
                'Dataset': str(dataset),
                'JD': jd,
                'Date': date_lookup.get(jd, str(julian_to_calendar([jd])[0])),
                'Mdot_Msun_per_yr': mdot,
                'Mdot_err_Msun_per_yr': mdot_err,
                'MstarMdot_Msun2_per_yr': float(m_star_solar) * mdot,
                'MstarMdot_err_Msun2_per_yr': float(m_star_solar) * mdot_err,
                'Av_mag': float(params.get('Av', np.nan)),
                'Av_err_mag': _av_error(params),
            })
    return pd.DataFrame(rows)


def integrate_accretion_history(
    results,
    start_jd=None,
    end_jd=None,
    max_gap_days=None,
    m_star_solar=0.5,
):
    """Integrate a fitted accretion-rate curve with linear interpolation.

    Each included interval is integrated with the trapezoid rule, equivalent to
    assuming that Mdot changes linearly between adjacent fitted epochs.  Setting
    ``max_gap_days`` excludes longer gaps instead of interpolating across them.
    No extrapolation is performed beyond the first or last fitted epoch.
    """
    table = fit_parameter_table(results, m_star_solar=m_star_solar)
    if table.empty:
        raise ValueError('No fitted daily parameters are available to integrate')
    table = table.sort_values('JD').drop_duplicates('JD', keep='last')
    jd = table['JD'].to_numpy(dtype=float)
    mdot = table['Mdot_Msun_per_yr'].to_numpy(dtype=float)
    mdot_err = table['Mdot_err_Msun_per_yr'].to_numpy(dtype=float)
    finite = np.isfinite(jd) & np.isfinite(mdot) & (mdot >= 0)
    jd, mdot, mdot_err = jd[finite], mdot[finite], mdot_err[finite]
    if len(jd) < 2:
        raise ValueError('At least two finite fitted epochs are required for integration')

    lower = jd[0] if start_jd is None else max(float(start_jd), jd[0])
    upper = jd[-1] if end_jd is None else min(float(end_jd), jd[-1])
    if lower >= upper:
        raise ValueError('The requested integration interval does not overlap the fitted epochs')

    interior = (jd > lower) & (jd < upper)
    eval_jd = np.concatenate(([lower], jd[interior], [upper]))
    eval_mdot = np.interp(eval_jd, jd, mdot)
    finite_err = np.isfinite(mdot_err)
    if np.sum(finite_err) >= 2:
        eval_err = np.interp(eval_jd, jd[finite_err], mdot_err[finite_err])
    else:
        eval_err = np.full_like(eval_jd, np.nan)

    gap_days = np.diff(eval_jd)
    include = np.ones(len(gap_days), dtype=bool)
    if max_gap_days is not None:
        include &= gap_days <= float(max_gap_days)
    interval_years = gap_days / 365.25
    interval_mass = 0.5 * (eval_mdot[:-1] + eval_mdot[1:]) * interval_years
    mass_accreted = float(np.sum(interval_mass[include]))

    # Linear propagation for trapezoidal weights.  These are fit-only errors;
    # they do not include the much larger systematic uncertainty in gap filling.
    weights = np.zeros(len(eval_jd), dtype=float)
    for index, (dt_years, use_interval) in enumerate(zip(interval_years, include)):
        if use_interval:
            weights[index] += 0.5 * dt_years
            weights[index + 1] += 0.5 * dt_years
    mass_error = (
        float(np.sqrt(np.sum((weights * eval_err) ** 2)))
        if np.all(np.isfinite(eval_err)) else np.nan
    )

    excluded = [
        {'start_jd': float(eval_jd[i]), 'end_jd': float(eval_jd[i + 1]), 'gap_days': float(gap_days[i])}
        for i in np.flatnonzero(~include)
    ]
    return {
        'start_jd': float(lower),
        'end_jd': float(upper),
        'span_years': float((upper - lower) / 365.25),
        'covered_years': float(np.sum(interval_years[include])),
        'mass_accreted_Msun': mass_accreted,
        'mass_accreted_err_Msun': mass_error,
        'integrated_MstarMdot_Msun2': float(m_star_solar) * mass_accreted,
        'interpolation': 'piecewise linear between adjacent fitted epochs',
        'max_gap_days': None if max_gap_days is None else float(max_gap_days),
        'excluded_gaps': excluded,
        'n_epochs': int(len(eval_jd)),
        'n_intervals_included': int(np.sum(include)),
    }


def print_final_panel_results(
    spectral_result_sets=None,
    photometric_result_sets=None,
    integration_results=None,
    m_star_solar=0.5,
):
    """Print the fit quantities and accreted-mass estimates used by the paper."""
    frames = []
    if spectral_result_sets:
        frame = fit_parameter_table(spectral_result_sets, m_star_solar=m_star_solar)
        if not frame.empty:
            frame.insert(0, 'Kind', 'spectrum')
            frames.append(frame)
    if photometric_result_sets:
        frame = fit_parameter_table(photometric_result_sets, m_star_solar=m_star_solar)
        if not frame.empty:
            frame.insert(0, 'Kind', 'photometry')
            frames.append(frame)

    print('\nFit parameters for Table 2 / final panels')
    if frames:
        report = pd.concat(frames, ignore_index=True)
        columns = ['Kind', 'Dataset', 'JD', 'Date', 'Mdot_Msun_per_yr', 'MstarMdot_Msun2_per_yr', 'Av_mag']
        print(report[columns].to_string(index=False, formatters={
            'JD': lambda value: f'{value:.3f}',
            'Mdot_Msun_per_yr': lambda value: f'{value:.4e}',
            'MstarMdot_Msun2_per_yr': lambda value: f'{value:.4e}',
            'Av_mag': lambda value: f'{value:.3f}',
        }))
    else:
        report = pd.DataFrame()
        print('No fit results supplied.')

    if integration_results:
        if isinstance(integration_results, dict) and 'mass_accreted_Msun' in integration_results:
            integration_results = {'accretion history': integration_results}
        print('\nIntegrated accreted mass')
        for label, summary in integration_results.items():
            gap_note = 'all gaps interpolated' if summary.get('max_gap_days') is None else f"gaps > {summary['max_gap_days']:g} d excluded"
            print(
                f"{label}: {summary['mass_accreted_Msun']:.4e} +/- "
                f"{summary['mass_accreted_err_Msun']:.2e} Msun over "
                f"{summary['covered_years']:.2f} covered yr ({gap_note})"
            )
    return report


__all__ = [
    '_normalise_sed_jds',
    '_parse_filter_groups',
    '_mag_from_flux',
    '_resolve_jd_key',
    '_repair_flux_from_mag_local',
    '_prepare_day_series',
    '_prepare_day_series_from_frame',
    '_get_defaults',
    '_mstar_mdot_solar2_per_year',
    '_model_flux_for_freq',
    'prepare_final_panel_residuals',
    'plot_filter_evolution_panel',
    'plot_final_residuals',
    'plot_final_panels_0_to_4',
    'plot_final_panels_0_to_4_layouts',
    'plot_panel5_sed',
    'plot_panel5_seds',
    'fit_parameter_table',
    'integrate_accretion_history',
    'print_final_panel_results',
]
