"""Core regularized fitting functions."""

from __future__ import annotations

try:
    from .ultimate_common import *
except ImportError:
    from ultimate_common import *

try:
    from .ultimate_file_organisers import assign_jd_day, get_daily_data
except ImportError:
    from ultimate_file_organisers import assign_jd_day, get_daily_data

try:
    from .ultimate_physics import *
except ImportError:
    from ultimate_physics import *

try:
    from .ultimate_helpers import *
except ImportError:
    from ultimate_helpers import *

SPECTRAL_REDE_THRESHOLD_MICRON = globals().get('SPECTRAL_REDE_THRESHOLD_MICRON', 2.0)
SPECTRAL_REDE_MAX_BLUE_BB_FRACTION = globals().get('SPECTRAL_REDE_MAX_BLUE_BB_FRACTION', 0.10)
SPECTRAL_REDE_BLUE_PENALTY_WEIGHT = globals().get('SPECTRAL_REDE_BLUE_PENALTY_WEIGHT', 25.0)


def _normalize_red_excess_fit_param(fit_param):
    value = str(fit_param).strip().lower()
    mapping = {
        't_bb': 'T_bb',
        't': 'T_bb',
        'T': 'T_bb',
        'temperature': 'T_bb',
        'temp': 'T_bb',
        'r_bb': 'R_bb',
        'r': 'R_bb',
        'R': 'R_bb',
        'radius': 'R_bb',
        'geometry': 'R_bb',
        'logr_bb': 'R_bb'
    }
    if value not in mapping:
        raise ValueError("red_excess_fit_param must be one of: 'T_bb', 'R_bb', 'temperature', 'radius', 'geometry'")
    return mapping[value]


def _safe_float(value, default=np.nan):
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _resolve_red_excess_component(day_params, fit_info=None,
                                  default_T_bb=RED_EXCESS_DEFAULT_T_BB,
                                  default_R_bb=RED_EXCESS_DEFAULT_R_BB):
    fit_info = fit_info or {}
    fit_param = fit_info.get('red_excess_fit_param')
    if fit_param is None or str(fit_param).strip().lower() in ('', 'none', 'nan', 'false'):
        if 'R_bb' in day_params or 'logR_bb' in day_params:
            fit_param = 'R_bb'
        else:
            fit_param = 'T_bb'
    try:
        fit_param = _normalize_red_excess_fit_param(fit_param)
    except ValueError:
        fit_param = 'R_bb' if ('R_bb' in day_params or 'logR_bb' in day_params) else 'T_bb'

    if fit_param == 'R_bb':
        T_bb = _safe_float(fit_info.get('fixed_T_bb', default_T_bb), default_T_bb)
        if 'R_bb' in day_params:
            R_bb = _safe_float(day_params.get('R_bb'), default_R_bb)
        elif 'logR_bb' in day_params:
            R_bb = 10 ** _safe_float(day_params.get('logR_bb'), np.log10(default_R_bb))
        else:
            R_bb = _safe_float(fit_info.get('fixed_R_bb', default_R_bb), default_R_bb)
    else:
        T_bb = _safe_float(day_params.get('T_bb', fit_info.get('fixed_T_bb', default_T_bb)), default_T_bb)
        R_bb = _safe_float(fit_info.get('fixed_R_bb', default_R_bb), default_R_bb)

    return fit_param, T_bb, R_bb


def _coerce_wavelength_meters(day_df):
    if 'Lambda_m' in day_df.columns:
        return pd.to_numeric(day_df['Lambda_m'], errors='coerce').to_numpy(dtype=float)

    if 'Lambda' not in day_df.columns:
        raise ValueError("Spectral dataframe must contain 'Lambda_m' or 'Lambda'")

    wavelength_values = pd.to_numeric(day_df['Lambda'], errors='coerce').to_numpy(dtype=float)
    finite_values = wavelength_values[np.isfinite(wavelength_values)]
    if len(finite_values) == 0:
        return wavelength_values
    if np.nanmax(finite_values) > 1e-3:
        return wavelength_values * 1e-6
    return wavelength_values


def _get_spectral_daily_data(df):
    if df is None or len(df) == 0:
        return {}

    spectral_df = df.copy()
    if 'JD' not in spectral_df.columns and 'JD_day' not in spectral_df.columns:
        raise ValueError("Spectral dataframe must contain 'JD' or 'JD_day'")

    # Spectra are fit one model per observing day.  Loader-level JD_day values
    # can be file-level MJDs, so recompute the day key from DateCode/Date when
    # available to combine SXD and LXD spectra from the same calendar date.
    if 'JD' not in spectral_df.columns:
        spectral_df['JD'] = pd.to_numeric(spectral_df['JD_day'], errors='coerce')

    if 'DateCode' in spectral_df.columns:
        date_key = spectral_df['DateCode'].astype(str).str.strip()
        valid_date = date_key.ne('') & date_key.str.lower().ne('nan')
        if valid_date.any():
            spectral_df['_spectral_day_key'] = np.where(
                valid_date,
                'datecode:' + date_key,
                'jd:' + spectral_df['JD'].astype(str),
            )
            jd_day_map = spectral_df.groupby('_spectral_day_key')['JD'].median()
            spectral_df['JD_day'] = spectral_df['_spectral_day_key'].map(jd_day_map).astype(float)
    elif 'Date' in spectral_df.columns:
        date_key = spectral_df['Date'].astype(str).str.strip().str.slice(0, 10)
        valid_date = date_key.ne('') & date_key.str.lower().ne('nan')
        if valid_date.any():
            spectral_df['_spectral_day_key'] = np.where(
                valid_date,
                'date:' + date_key,
                'jd:' + spectral_df['JD'].astype(str),
            )
            jd_day_map = spectral_df.groupby('_spectral_day_key')['JD'].median()
            spectral_df['JD_day'] = spectral_df['_spectral_day_key'].map(jd_day_map).astype(float)
    elif 'JD_day' not in spectral_df.columns:
        spectral_df = assign_jd_day(spectral_df)

    spectral_df['Lambda_m'] = _coerce_wavelength_meters(spectral_df)
    if 'Frequency' not in spectral_df.columns:
        spectral_df['Frequency'] = constants.c / spectral_df['Lambda_m']
    else:
        spectral_df['Frequency'] = pd.to_numeric(spectral_df['Frequency'], errors='coerce')

    spectral_df['Flux'] = pd.to_numeric(spectral_df['Flux'], errors='coerce')
    spectral_df['Fluxerr'] = pd.to_numeric(spectral_df['Fluxerr'], errors='coerce')
    spectral_df = spectral_df.dropna(subset=['JD_day', 'Lambda_m', 'Frequency', 'Flux'])
    spectral_df['JD_day'] = spectral_df['JD_day'].astype(float)
    spectral_df = spectral_df.drop(columns=['_spectral_day_key'], errors='ignore')

    daily_data = {}
    for jd_day, day_data in spectral_df.groupby('JD_day'):
        valid_day = day_data[np.isfinite(day_data['Frequency']) & np.isfinite(day_data['Flux'])]
        if len(valid_day) == 0:
            continue
        daily_data[jd_day] = valid_day.sort_values('Lambda_m').reset_index(drop=True)

    print(f'Number of spectral days available: {len(daily_data)}')
    if daily_data:
        total_samples = sum(len(day_df) for day_df in daily_data.values())
        print(f'Total spectral samples in selected days: {total_samples}')
    return daily_data


def _photometric_frequency_from_row(row, filter_name=None, fallback_frequencies=None):
    """Return the row-level photometric frequency, falling back to filter defaults.

    Photometric rows carry their own ``Lambda`` values.  Those are the
    wavelengths the data point represents, so fitting and plotting must use
    them consistently.  The filter table is only a fallback for older rows
    without a usable wavelength.
    """
    wavelength_value = _safe_float(row.get('Lambda'), np.nan)
    if np.isfinite(wavelength_value) and wavelength_value > 0:
        wavelength_micron = wavelength_value if wavelength_value > 1e-3 else wavelength_value * 1e6
        return wavelength_to_frequency(wavelength_micron), 'raw'

    fallback_frequencies = fallback_frequencies or {}
    if filter_name in fallback_frequencies:
        return fallback_frequencies[filter_name], 'fallback'
    return np.nan, 'missing'


def ultimate_fitting_regularized(required_filters, fit_filters, global_params=(),
    AR_mode=False, red_excess_mode=False, initial_params=None,
    df=None, debug=False, lambda_reg=1.0,
    regularize_params=None,
    red_excess_fit_param='T_bb',
    fixed_T_bb=RED_EXCESS_DEFAULT_T_BB,
    fixed_R_bb=RED_EXCESS_DEFAULT_R_BB,
    data_mode='photometry',
    spectral_rede_joint_refine=True,
    spectral_rede_max_blue_bb_fraction=SPECTRAL_REDE_MAX_BLUE_BB_FRACTION,
    spectral_rede_blue_penalty_weight=SPECTRAL_REDE_BLUE_PENALTY_WEIGHT,
    optimizer_maxiter=100000,
    optimizer_maxfun=100000,
    optimizer_ftol=1e-12,
    optimizer_gtol=1e-6,
    local_solver_max_nfev=10000,
    local_solver_ftol=1e-13,
    local_solver_gtol=1e-13,
    mdot_bounds_msun_per_year=(1e-7, 0.8e-3),
    av_bounds=(0.1, 40.0),
    evaluate_only=False):
    data_mode = str(data_mode).strip().lower()
    if data_mode not in ('photometry', 'spectral'):
        raise ValueError("data_mode must be either 'photometry' or 'spectral'")
    if AR_mode and red_excess_mode:
        raise ValueError('AR_mode and red_excess_mode cannot be enabled together')
    if optimizer_maxiter < 1 or optimizer_maxfun < 1 or local_solver_max_nfev < 1:
        raise ValueError('Optimizer iteration/evaluation limits must be positive')
    for tolerance_name, tolerance_value in (
        ('optimizer_ftol', optimizer_ftol),
        ('optimizer_gtol', optimizer_gtol),
        ('local_solver_ftol', local_solver_ftol),
        ('local_solver_gtol', local_solver_gtol),
    ):
        if not np.isfinite(tolerance_value) or tolerance_value <= 0:
            raise ValueError(f'{tolerance_name} must be a positive finite value')
    if len(av_bounds) != 2 or not np.all(np.isfinite(av_bounds)) or av_bounds[0] >= av_bounds[1]:
        raise ValueError('av_bounds must contain two finite increasing values')
    if (
        len(mdot_bounds_msun_per_year) != 2
        or not np.all(np.isfinite(mdot_bounds_msun_per_year))
        or mdot_bounds_msun_per_year[0] <= 0
        or mdot_bounds_msun_per_year[0] >= mdot_bounds_msun_per_year[1]
    ):
        raise ValueError('mdot_bounds_msun_per_year must contain two positive increasing values')
    fit_param_name = None
    fitted_param_internal = None
    fitted_param_names = ['logMdot', 'Av']
    spectral_rede_mode = data_mode == 'spectral' and red_excess_mode
    if spectral_rede_mode:
        fit_param_name = 'T_bb_R_bb'
        fitted_param_names.extend(['T_bb', 'R_bb'])
    elif red_excess_mode:
        fit_param_name = _normalize_red_excess_fit_param(red_excess_fit_param)
        fitted_param_internal = 'T_bb' if fit_param_name == 'T_bb' else 'logR_bb'
        fitted_param_names.append(fit_param_name)

    if regularize_params is None:
        regularize_params = fitted_param_names.copy()

    internal_regularize_params = []
    for param_name in regularize_params:
        if param_name in ('logMdot', 'Av') and param_name not in internal_regularize_params:
            internal_regularize_params.append(param_name)
        elif spectral_rede_mode and param_name in ('T_bb', 'temperature', 'temp'):
            if 'T_bb' not in internal_regularize_params:
                internal_regularize_params.append('T_bb')
        elif spectral_rede_mode and param_name in ('R_bb', 'R', 'radius', 'geometry', 'logR_bb'):
            if 'logR_bb' not in internal_regularize_params:
                internal_regularize_params.append('logR_bb')
        elif red_excess_mode and fit_param_name == 'T_bb' and param_name in ('T_bb', 'temperature', 'temp'):
            if 'T_bb' not in internal_regularize_params:
                internal_regularize_params.append('T_bb')
        elif red_excess_mode and fit_param_name == 'R_bb' and param_name in ('R_bb', 'R', 'radius', 'geometry', 'logR_bb'):
            if 'logR_bb' not in internal_regularize_params:
                internal_regularize_params.append('logR_bb')

    default_params = {
        'Mdot': 1e-4 * M_sun / year,
        'Av': 15.0,
        'M': 0.5 * M_sun,
        'R_star': 3.0 * R_sun,
        'R_in': 3.0 * R_sun,
        'R_out': 2 * constants.au,
        'distance': 700 * constants.parsec,
        'T_bb': fixed_T_bb,
        'R_bb': fixed_R_bb
    }
    if initial_params:
        default_params.update(initial_params)

    fixed_T_bb = _safe_float(default_params.get('T_bb', fixed_T_bb), fixed_T_bb)
    fixed_R_bb = _safe_float(default_params.get('R_bb', fixed_R_bb), fixed_R_bb)

    if data_mode == 'spectral':
        daily_data = _get_spectral_daily_data(df)
    else:
        daily_data = get_daily_data(df, required_filters)
    if not daily_data:
        print("No days found matching the filter requirements!")
        return None
    print(f"\nFound {len(daily_data)} days with required filters")

    if red_excess_mode and data_mode == 'photometry':
        daily_data = filter_days_for_red_excess(daily_data, fit_filters)
        if not daily_data:
            print(daily_data, fit_filters)
            print("No days found with sufficient measurements for red excess mode!")
            return None
        print(f"Red excess mode: {len(daily_data)} days have sufficient measurements")
        print(f"Red excess fit parameter: {fit_param_name}")

    filter_frequencies = {}
    if data_mode == 'photometry':
        wavelengths = get_filter_wavelengths()
        for filter_name in fit_filters:
            if filter_name in wavelengths:
                filter_frequencies[filter_name] = wavelength_to_frequency(wavelengths[filter_name])

    ar_visualization_days = None
    if AR_mode:
        all_days = list(daily_data.keys())
        if data_mode == 'spectral':
            ar_visualization_days = all_days
        else:
            ar_visualization_days = all_days[-3:]
        print(f"AR mode: Will sample and visualize {len(ar_visualization_days)} days with MCMC")

    all_frequencies = []
    all_log_fluxes = []
    all_errors = []
    day_indices = []
    time_values = []
    filter_indices = []
    wavelength_source_counts = {'raw': 0, 'fallback': 0, 'missing': 0}
    wavelength_source_days = {'raw': set(), 'fallback': set(), 'missing': set()}
    fallback_filter_days = defaultdict(set)
    missing_filter_days = defaultdict(set)

    jd_to_day_idx = {jd_day: idx for idx, jd_day in enumerate(sorted(daily_data.keys()))}
    sorted_jd_days = sorted(daily_data.keys())

    red_excess_data = {}
    if red_excess_mode and data_mode == 'photometry':
        red_excess_data = classify_filters_by_color(daily_data, fit_filters, wavelengths)

    for jd_day, day_df in daily_data.items():
        day_idx = jd_to_day_idx[jd_day]
        if data_mode == 'spectral':
            for _, row in day_df.iterrows():
                flux_val = row['Flux']
                fluxerr_val = row.get('Fluxerr', np.nan)
                frequency_val = row.get('Frequency', np.nan)

                if not np.isfinite(frequency_val) or flux_val <= 0:
                    continue

                log_flux = np.log(flux_val)
                if not np.isfinite(fluxerr_val) or fluxerr_val <= 0:
                    fluxerr_val = max(0.1 * flux_val, 1e-12)

                frac_err = fluxerr_val / flux_val
                frac_err = np.sqrt(frac_err**2 + 0.05**2)
                if np.isnan(log_flux) or np.isinf(log_flux) or np.isnan(frac_err) or np.isinf(frac_err):
                    continue

                all_frequencies.append(frequency_val)
                all_log_fluxes.append(log_flux)
                all_errors.append(frac_err)
                day_indices.append(day_idx)
                time_values.append(jd_day)
                filter_indices.append(row.get('Filter', 'spectral'))
        else:
            for filter_name in fit_filters:
                filter_data = day_df[day_df['Filter'] == filter_name]
                for _, row in filter_data.iterrows():
                    frequency_val, wavelength_source = _photometric_frequency_from_row(
                        row, filter_name, filter_frequencies
                    )
                    wavelength_source_counts[wavelength_source] = wavelength_source_counts.get(wavelength_source, 0) + 1
                    wavelength_source_days.setdefault(wavelength_source, set()).add(jd_day)
                    if wavelength_source == 'fallback':
                        fallback_filter_days[jd_day].add(filter_name)
                    elif wavelength_source == 'missing':
                        missing_filter_days[jd_day].add(filter_name)
                    if not np.isfinite(frequency_val):
                        continue

                    flux_val = row['Flux']
                    mag_val = row['Mag']
                    zp_val = row['ZP']
                    fluxerr_val = row['Fluxerr']

                    if flux_val == 0:
                        if mag_val == 0:
                            print("Flux and magnitude was 0")
                            continue
                        flux_val = 10 ** (-0.4 * mag_val) * zp_val
                    if flux_val <= 0:
                        continue

                    log_flux = np.log(flux_val)
                    if fluxerr_val == 0 or np.isnan(fluxerr_val):
                        fluxerr_val = max(0.1 * flux_val, 1e-12)

                    frac_err = fluxerr_val / flux_val
                    frac_err = np.sqrt(frac_err**2 + 0.05**2)
                    if np.isnan(log_flux) or np.isinf(log_flux) or np.isnan(frac_err) or np.isinf(frac_err):
                        continue

                    all_frequencies.append(frequency_val)
                    all_log_fluxes.append(log_flux)
                    all_errors.append(frac_err)
                    day_indices.append(day_idx)
                    time_values.append(jd_day)
                    filter_indices.append(filter_name)

    if len(all_frequencies) == 0:
        print("No valid data points found for fitting!")
        return None

    if data_mode == 'photometry':
        raw_days = len(wavelength_source_days.get('raw', set()))
        fallback_days = len(wavelength_source_days.get('fallback', set()))
        missing_days = len(wavelength_source_days.get('missing', set()))
        total_days_with_wavelength_source = len(
            set().union(
                wavelength_source_days.get('raw', set()),
                wavelength_source_days.get('fallback', set()),
                wavelength_source_days.get('missing', set())
            )
        )
        print(
            "Photometric wavelength source: "
            f"{raw_days}/{total_days_with_wavelength_source} days use row Lambda "
            f"({wavelength_source_counts.get('raw', 0)} measurements); "
            f"{fallback_days}/{total_days_with_wavelength_source} days use fallback filter Lambda "
            f"({wavelength_source_counts.get('fallback', 0)} measurements)"
        )
        if fallback_filter_days:
            fallback_details = [
                f"{jd_day}: {','.join(sorted(filters))}"
                for jd_day, filters in sorted(fallback_filter_days.items())
            ]
            print("Fallback filter Lambda used for days/filters: " + "; ".join(fallback_details))
        if missing_filter_days:
            missing_details = [
                f"{jd_day}: {','.join(sorted(filters))}"
                for jd_day, filters in sorted(missing_filter_days.items())
            ]
            print("WARNING: No usable row or fallback Lambda for days/filters: " + "; ".join(missing_details))

    all_frequencies = np.array(all_frequencies)
    all_log_fluxes = np.array(all_log_fluxes)
    all_errors = np.array(all_errors)
    day_indices = np.array(day_indices)
    time_values = np.array(time_values)
    filter_indices = np.array(filter_indices)
    all_wavelengths_micron = constants.c / all_frequencies * 1e6

    n_days = len(daily_data)
    n_data_points = len(all_frequencies)
    print(f"Fitting {n_data_points} data points across {n_days} days")

    n_params_per_day = 4 if spectral_rede_mode else (3 if red_excess_mode else 2)
    n_global_params = len(global_params)
    # In evaluation-only mode every model parameter is fixed by the caller, so
    # none of them consume a degree of freedom.
    total_params = 0 if evaluate_only else n_days * n_params_per_day + n_global_params
    dof = n_data_points - total_params
    if dof < 0:
        print("WARNING: Not enough data points for fitting!")
        return None

    reg_param_indices = {'logMdot': [], 'Av': []}
    if spectral_rede_mode:
        reg_param_indices['T_bb'] = []
        reg_param_indices['logR_bb'] = []
    elif red_excess_mode:
        reg_param_indices[fitted_param_internal] = []

    reg_time_diffs = []
    for day_idx in range(n_days - 1):
        dt = sorted_jd_days[day_idx + 1] - sorted_jd_days[day_idx]
        if dt <= 0:
            continue
        reg_time_diffs.append(dt)
        if 'logMdot' in internal_regularize_params:
            reg_param_indices['logMdot'].append((day_idx * n_params_per_day, (day_idx + 1) * n_params_per_day))
        if 'Av' in internal_regularize_params:
            reg_param_indices['Av'].append((day_idx * n_params_per_day + 1, (day_idx + 1) * n_params_per_day + 1))
        if spectral_rede_mode:
            if 'T_bb' in internal_regularize_params:
                reg_param_indices['T_bb'].append((day_idx * n_params_per_day + 2, (day_idx + 1) * n_params_per_day + 2))
            if 'logR_bb' in internal_regularize_params:
                reg_param_indices['logR_bb'].append((day_idx * n_params_per_day + 3, (day_idx + 1) * n_params_per_day + 3))
        elif red_excess_mode and fitted_param_internal in internal_regularize_params:
            reg_param_indices[fitted_param_internal].append((day_idx * n_params_per_day + 2, (day_idx + 1) * n_params_per_day + 2))

    reg_time_diffs = np.array(reg_time_diffs)
    param_bounds = []
    initial_guess = []
    radius_bounds = (0.05 * constants.au, 20 * constants.au)

    for _ in range(n_days):
        initial_guess.extend([np.log10(default_params['Mdot']), default_params['Av']])
        param_bounds.append(tuple(
            np.log10(float(value) * M_sun / year)
            for value in mdot_bounds_msun_per_year
        ))
        param_bounds.append(tuple(float(value) for value in av_bounds))
        if spectral_rede_mode:
            initial_guess.append(default_params['T_bb'])
            param_bounds.append((650.0, 5000.0))
            initial_guess.append(np.log10(default_params['R_bb']))
            param_bounds.append((np.log10(radius_bounds[0]), np.log10(radius_bounds[1])))
        elif red_excess_mode:
            if fit_param_name == 'T_bb':
                initial_guess.append(default_params['T_bb'])
                param_bounds.append((1000.0, 10000.0))
            else:
                initial_guess.append(np.log10(default_params['R_bb']))
                param_bounds.append((np.log10(radius_bounds[0]), np.log10(radius_bounds[1])))

    global_param_map = {}
    for param_name in global_params:
        if param_name in default_params:
            global_param_map[param_name] = len(initial_guess)
            initial_guess.append(default_params[param_name])
            if param_name == 'distance':
                param_bounds.append((100 * constants.parsec, 2000 * constants.parsec))
            elif param_name == 'R_out':
                param_bounds.append((10 * constants.au, 1000 * constants.au))
            elif param_name == 'M':
                param_bounds.append((0.05 * M_sun, 2.0 * M_sun))
            elif param_name == 'R_star':
                param_bounds.append((0.3 * R_sun, 10.0 * R_sun))
            else:
                param_bounds.append((0.1, 100))

    def model_function_vectorized(params):
        predicted_fluxes = np.full(n_data_points, -50.0)
        if spectral_rede_mode:
            logMdot_array = params[::4][:n_days]
            av_array = params[1::4][:n_days]
            t_bb_array = params[2::4][:n_days]
            log_r_bb_array = params[3::4][:n_days]
        elif red_excess_mode:
            logMdot_array = params[::3][:n_days]
            av_array = params[1::3][:n_days]
            extra_param_array = params[2::3][:n_days]
        else:
            logMdot_array = params[::2][:n_days]
            av_array = params[1::2][:n_days]

        M = params[global_param_map['M']] if 'M' in global_param_map else default_params['M']
        R_star = params[global_param_map['R_star']] if 'R_star' in global_param_map else default_params['R_star']
        R_out = params[global_param_map['R_out']] if 'R_out' in global_param_map else default_params['R_out']
        distance = params[global_param_map['distance']] if 'distance' in global_param_map else default_params['distance']
        R_in = R_star * 2 if AR_mode else default_params['R_in']

        for day_idx in np.unique(day_indices):
            day_mask = day_indices == day_idx
            day_frequencies = all_frequencies[day_mask]
            day_filters = filter_indices[day_mask]
            if len(day_frequencies) == 0:
                continue

            logMdot = logMdot_array[day_idx]
            mdot = 10.0 ** logMdot
            av = av_array[day_idx]
            try:
                if spectral_rede_mode:
                    day_wavelengths = all_wavelengths_micron[day_mask]

                    log_fluxes_acc = accretion_model(
                        day_frequencies, mdot, M, R_star, R_out, R_in, distance, av
                    )
                    valid_acc = ~(np.isnan(log_fluxes_acc) | np.isinf(log_fluxes_acc) | (log_fluxes_acc < -45))
                    log_fluxes_acc[~valid_acc] = -50.0

                    T_bb = t_bb_array[day_idx]
                    R_bb = 10.0 ** log_r_bb_array[day_idx]
                    log_fluxes_bb = planck_model_custom(
                        day_frequencies, T_bb, distance, av, R_bb=R_bb
                    )
                    valid_bb = ~(np.isnan(log_fluxes_bb) | np.isinf(log_fluxes_bb) | (log_fluxes_bb < -200))
                    log_fluxes_bb[~valid_bb] = -50.0
                    acc_flux = np.where(np.isfinite(log_fluxes_acc), np.exp(log_fluxes_acc), 0.0)
                    bb_flux = np.where(np.isfinite(log_fluxes_bb), np.exp(log_fluxes_bb), 0.0)
                    total_flux = acc_flux + bb_flux
                    total_flux = np.where((total_flux > 0) & np.isfinite(total_flux), total_flux, np.exp(-50.0))
                    day_predicted = np.log(total_flux)

                    predicted_fluxes[day_mask] = day_predicted
                elif red_excess_mode:
                    jd_day = sorted_jd_days[day_idx]
                    if fit_param_name == 'T_bb':
                        T_bb = extra_param_array[day_idx]
                        R_bb = fixed_R_bb
                    else:
                        T_bb = fixed_T_bb
                        R_bb = 10.0 ** extra_param_array[day_idx]

                    blue_mask = np.array([flt in red_excess_data[jd_day]['blue_filters'] for flt in day_filters])
                    red_mask = np.array([flt in red_excess_data[jd_day]['red_filters'] for flt in day_filters])
                    unclassified_mask = ~(blue_mask | red_mask)
                    if np.any(unclassified_mask):
                        blue_mask = blue_mask | unclassified_mask
                        if debug:
                            print(f"Day {day_idx}: assigned {np.sum(unclassified_mask)} unclassified filters to accretion model")

                    day_predicted = np.full(len(day_frequencies), -50.0)
                    if np.any(blue_mask):
                        try:
                            log_fluxes_blue = accretion_model(
                                day_frequencies[blue_mask], mdot, M, R_star, R_out, R_in, distance, av
                            )
                            valid_blue = ~(np.isnan(log_fluxes_blue) | np.isinf(log_fluxes_blue) | (log_fluxes_blue < -45))
                            log_fluxes_blue[~valid_blue] = -50.0
                            day_predicted[blue_mask] = log_fluxes_blue
                        except Exception as exc:
                            if debug:
                                print(f"Error in accretion model for day {day_idx}: {exc}")
                            day_predicted[blue_mask] = -50.0

                    if np.any(red_mask):
                        try:
                            red_frequencies = day_frequencies[red_mask]
                            log_fluxes_red_acc = accretion_model(
                                red_frequencies, mdot, M, R_star, R_out, R_in, distance, av
                            )
                            log_fluxes_red_bb = planck_model_custom(
                                red_frequencies, T_bb, distance, av, R_bb=R_bb
                            )
                            valid_acc = ~(np.isnan(log_fluxes_red_acc) | np.isinf(log_fluxes_red_acc) | (log_fluxes_red_acc < -45))
                            valid_bb = ~(np.isnan(log_fluxes_red_bb) | np.isinf(log_fluxes_red_bb) | (log_fluxes_red_bb < -200))
                            log_fluxes_red_acc[~valid_acc] = -50.0
                            log_fluxes_red_bb[~valid_bb] = -50.0
                            red_acc_flux = np.where(np.isfinite(log_fluxes_red_acc), np.exp(log_fluxes_red_acc), 0.0)
                            red_bb_flux = np.where(np.isfinite(log_fluxes_red_bb), np.exp(log_fluxes_red_bb), 0.0)
                            red_total_flux = red_acc_flux + red_bb_flux
                            red_total_flux = np.where((red_total_flux > 0) & np.isfinite(red_total_flux), red_total_flux, np.exp(-50.0))
                            day_predicted[red_mask] = np.log(red_total_flux)
                        except Exception as exc:
                            if debug:
                                print(f"Error in red-excess model for day {day_idx}: {exc}")
                            day_predicted[red_mask] = -50.0

                    predicted_fluxes[day_mask] = day_predicted
                else:
                    log_fluxes = accretion_model(
                        day_frequencies, mdot, M, R_star, R_out, R_in, distance, av
                    )
                    valid_mask = ~(np.isnan(log_fluxes) | np.isinf(log_fluxes) | (log_fluxes < -45))
                    log_fluxes[~valid_mask] = -50.0
                    predicted_fluxes[day_mask] = log_fluxes
            except Exception as exc:
                if debug:
                    print(f"Error in model evaluation for day {day_idx}: {exc}")
                predicted_fluxes[day_mask] = -50.0

        return predicted_fluxes

    def regularization_term_vectorized(params):
        reg_penalty = 0.0
        if len(reg_time_diffs) == 0:
            return reg_penalty
        for param_name in internal_regularize_params:
            param_pairs = reg_param_indices.get(param_name, [])
            if not param_pairs:
                continue
            indices_i = np.array([pair[0] for pair in param_pairs])
            indices_j = np.array([pair[1] for pair in param_pairs])
            theta_i = params[indices_i]
            theta_j = params[indices_j]
            reg_penalty += np.sum(((theta_j - theta_i) ** 2) / reg_time_diffs)
        return reg_penalty

    def spectral_rede_blue_penalty(params):
        if not spectral_rede_mode:
            return 0.0

        max_fraction = float(spectral_rede_max_blue_bb_fraction)
        if max_fraction <= 0:
            return 0.0

        penalty = 0.0
        logMdot_array = params[::4][:n_days]
        av_array = params[1::4][:n_days]
        t_bb_array = params[2::4][:n_days]
        log_r_bb_array = params[3::4][:n_days]

        M = params[global_param_map['M']] if 'M' in global_param_map else default_params['M']
        R_star = params[global_param_map['R_star']] if 'R_star' in global_param_map else default_params['R_star']
        R_out = params[global_param_map['R_out']] if 'R_out' in global_param_map else default_params['R_out']
        distance = params[global_param_map['distance']] if 'distance' in global_param_map else default_params['distance']
        R_in = default_params['R_in']

        for day_idx in np.unique(day_indices):
            day_mask = (day_indices == day_idx) & (all_wavelengths_micron < SPECTRAL_REDE_THRESHOLD_MICRON)
            if not np.any(day_mask):
                continue

            day_frequencies = all_frequencies[day_mask]
            try:
                acc_log_flux = accretion_model(
                    day_frequencies,
                    10.0 ** logMdot_array[day_idx],
                    M,
                    R_star,
                    R_out,
                    R_in,
                    distance,
                    av_array[day_idx],
                )
                bb_log_flux = planck_model_custom(
                    day_frequencies,
                    t_bb_array[day_idx],
                    distance,
                    av_array[day_idx],
                    R_bb=10.0 ** log_r_bb_array[day_idx],
                )
            except Exception:
                penalty += 1e6
                continue

            acc_flux = np.exp(np.where(np.isfinite(acc_log_flux), acc_log_flux, -50.0))
            bb_flux = np.exp(np.where(np.isfinite(bb_log_flux), bb_log_flux, -50.0))
            ratio = bb_flux / np.maximum(acc_flux, 1e-30)
            excess_ratio = np.maximum(0.0, ratio / max_fraction - 1.0)
            if np.any(np.isfinite(excess_ratio)):
                penalty += np.sum(excess_ratio[np.isfinite(excess_ratio)] ** 2)

        return spectral_rede_blue_penalty_weight * penalty

    def objective_function_optimized(params):
        predicted = model_function_vectorized(params)
        valid = np.isfinite(predicted) & (predicted > -150)
        if np.sum(valid) == 0:
            print('Huge penalty applied')
            return 1e30
        chi_squared = np.sum(((all_log_fluxes[valid] - predicted[valid]) / all_errors[valid]) ** 2)
        reg_term = regularization_term_vectorized(params)
        blue_penalty = spectral_rede_blue_penalty(params)
        return chi_squared + lambda_reg * reg_term + blue_penalty

    def polish_two_filter_days(params):
        """Solve independent two-band photometric days directly.

        For a trial logMdot the model colour fixes Av analytically, because
        extinction is linear in log-flux.  The two-parameter problem therefore
        collapses to one scalar root in logMdot, which is much more reliable
        than waiting for a generic optimizer to stumble onto a zero.
        """
        if (
            data_mode != 'photometry'
            or red_excess_mode
            or n_global_params > 0
            or n_params_per_day != 2
            or float(lambda_reg) != 0.0
            or optimize is None
        ):
            return params, 0

        polished = np.array(params, dtype=float, copy=True)
        n_polished = 0
        M = default_params['M']
        R_star = default_params['R_star']
        R_out = default_params['R_out']
        distance = default_params['distance']
        R_in = R_star * 2 if AR_mode else default_params['R_in']
        extinction_factor = np.log(10.0) / 2.5

        for day_idx in range(n_days):
            day_mask = day_indices == day_idx
            if np.sum(day_mask) != 2:
                continue

            day_freq = all_frequencies[day_mask]
            day_log_flux = all_log_fluxes[day_mask]
            day_err = all_errors[day_mask]
            wavelengths_micron = constants.c / day_freq * 1e6
            extinction_coeffs = ccm89_extinction(wavelengths_micron)
            colour_coeff = extinction_factor * (extinction_coeffs[0] - extinction_coeffs[1])
            if not np.isfinite(colour_coeff) or abs(colour_coeff) < 1e-12:
                continue

            start_idx = day_idx * n_params_per_day
            lower = np.array([param_bounds[start_idx][0], param_bounds[start_idx + 1][0]], dtype=float)
            upper = np.array([param_bounds[start_idx][1], param_bounds[start_idx + 1][1]], dtype=float)
            x0 = np.clip(polished[start_idx:start_idx + 2], lower, upper)

            def base_log_flux(logMdot):
                try:
                    base = accretion_model(
                        day_freq, 10.0 ** logMdot, M, R_star, R_out, R_in, distance, 0.0
                    )
                except Exception:
                    return None
                if np.any(~np.isfinite(base)) or np.any(base < -45):
                    return None
                return base

            observed_colour = day_log_flux[0] - day_log_flux[1]

            def av_for_logMdot(logMdot):
                base = base_log_flux(logMdot)
                if base is None:
                    return None, None
                model_colour = base[0] - base[1]
                av = (model_colour - observed_colour) / colour_coeff
                if not np.isfinite(av):
                    return None, base
                return av, base

            def absolute_residual(logMdot):
                av, base = av_for_logMdot(logMdot)
                if av is None:
                    return np.nan
                return base[0] - extinction_factor * extinction_coeffs[0] * av - day_log_flux[0]

            def two_param_residual(two_params):
                logMdot, av = two_params
                try:
                    pred = accretion_model(
                        day_freq, 10.0 ** logMdot, M, R_star, R_out, R_in, distance, av
                    )
                except Exception:
                    return np.full(2, 1e12)
                if np.any(~np.isfinite(pred)) or np.any(pred < -45):
                    return np.full(2, 1e12)
                return (day_log_flux - pred) / day_err

            before = np.sum(two_param_residual(x0) ** 2)
            candidate = None
            grid = np.linspace(lower[0], upper[0], 160)
            grid_values = np.array([absolute_residual(value) for value in grid], dtype=float)
            finite_indices = np.where(np.isfinite(grid_values))[0]

            if len(finite_indices) > 0:
                nearest_idx = finite_indices[np.argmin(np.abs(grid_values[finite_indices]))]
                bracket = None
                if abs(grid_values[nearest_idx]) < 1e-12:
                    bracket = (grid[nearest_idx], grid[nearest_idx])
                else:
                    for left_idx, right_idx in zip(finite_indices[:-1], finite_indices[1:]):
                        if right_idx != left_idx + 1:
                            continue
                        left_val = grid_values[left_idx]
                        right_val = grid_values[right_idx]
                        if left_val == 0 or left_val * right_val <= 0:
                            bracket = (grid[left_idx], grid[right_idx])
                            break

                if bracket is not None:
                    try:
                        if bracket[0] == bracket[1]:
                            logMdot_solution = bracket[0]
                        elif hasattr(optimize, 'brentq'):
                            logMdot_solution = optimize.brentq(
                                absolute_residual,
                                bracket[0],
                                bracket[1],
                                xtol=local_solver_ftol,
                                rtol=local_solver_ftol,
                                maxiter=local_solver_max_nfev
                            )
                        elif hasattr(optimize, 'root_scalar'):
                            root_result = optimize.root_scalar(
                                absolute_residual,
                                bracket=bracket,
                                xtol=local_solver_ftol,
                                rtol=local_solver_ftol,
                                maxiter=local_solver_max_nfev
                            )
                            logMdot_solution = root_result.root if root_result.converged else None
                        else:
                            logMdot_solution = None

                        if logMdot_solution is not None:
                            av_solution, _ = av_for_logMdot(logMdot_solution)
                            if av_solution is not None and lower[1] <= av_solution <= upper[1]:
                                candidate = np.array([logMdot_solution, av_solution], dtype=float)
                    except Exception as exc:
                        if debug:
                            print(f"Two-filter root solve failed for day {sorted_jd_days[day_idx]}: {exc}")

            if candidate is None and hasattr(optimize, 'least_squares'):
                try:
                    local_result = optimize.least_squares(
                        two_param_residual,
                        x0=x0,
                        bounds=(lower, upper),
                        ftol=local_solver_ftol,
                        xtol=local_solver_ftol,
                        gtol=local_solver_gtol,
                        x_scale=np.maximum(np.abs(x0), 1.0),
                        max_nfev=local_solver_max_nfev
                    )
                    candidate = local_result.x
                except Exception as exc:
                    if debug:
                        print(f"Two-filter least-squares fallback failed for day {sorted_jd_days[day_idx]}: {exc}")

            if candidate is None:
                continue

            after = np.sum(two_param_residual(candidate) ** 2)
            if np.isfinite(after) and after <= before:
                polished[start_idx:start_idx + 2] = candidate
                if after < before:
                    n_polished += 1

        return polished, n_polished

    def initialize_spectral_rede_stages(params):
        """Stage A disk fit on blue data, then Stage B BB fit on red excess."""
        if not spectral_rede_mode:
            return np.array(params, dtype=float, copy=True), {}

        staged = np.array(params, dtype=float, copy=True)
        stage_info = {
            'stage_A_blue_disk_days': 0,
            'stage_B_red_excess_bb_days': 0,
            'stage_A_failed_days': [],
            'stage_B_failed_days': [],
        }

        M = default_params['M']
        R_star = default_params['R_star']
        R_out = default_params['R_out']
        distance = default_params['distance']
        R_in = default_params['R_in']

        for day_idx in range(n_days):
            start_idx = day_idx * n_params_per_day
            day_mask = day_indices == day_idx
            blue_mask = day_mask & (all_wavelengths_micron < SPECTRAL_REDE_THRESHOLD_MICRON)
            red_mask = day_mask & (all_wavelengths_micron >= SPECTRAL_REDE_THRESHOLD_MICRON)

            day_bounds = np.array(param_bounds[start_idx:start_idx + n_params_per_day], dtype=float)
            lower = day_bounds[:, 0]
            upper = day_bounds[:, 1]

            # Stage A: fit accretion parameters to the blue side only.
            if np.sum(blue_mask) >= 2 and hasattr(optimize, 'least_squares'):
                blue_freq = all_frequencies[blue_mask]
                blue_log_flux = all_log_fluxes[blue_mask]
                blue_err = all_errors[blue_mask]
                disk_x0 = np.clip(staged[start_idx:start_idx + 2], lower[:2], upper[:2])

                def disk_residual(two_params):
                    logMdot, av = two_params
                    try:
                        pred = accretion_model(
                            blue_freq, 10.0 ** logMdot, M, R_star, R_out, R_in, distance, av
                        )
                    except Exception:
                        return np.full(len(blue_freq), 1e6)
                    if np.any(~np.isfinite(pred)) or np.any(pred < -45):
                        return np.full(len(blue_freq), 1e6)
                    return (blue_log_flux - pred) / blue_err

                try:
                    disk_result = optimize.least_squares(
                        disk_residual,
                        x0=disk_x0,
                        bounds=(lower[:2], upper[:2]),
                        ftol=1e-8,
                        xtol=1e-8,
                        gtol=1e-6,
                        max_nfev=2000,
                    )
                    if disk_result.success and np.all(np.isfinite(disk_result.x)):
                        staged[start_idx:start_idx + 2] = disk_result.x
                        stage_info['stage_A_blue_disk_days'] += 1
                    else:
                        stage_info['stage_A_failed_days'].append(sorted_jd_days[day_idx])
                except Exception as exc:
                    stage_info['stage_A_failed_days'].append(sorted_jd_days[day_idx])
                    if debug:
                        print(f"Stage A failed for day {sorted_jd_days[day_idx]}: {exc}")
            else:
                stage_info['stage_A_failed_days'].append(sorted_jd_days[day_idx])

            # Stage B: fit the BB to red excess after subtracting the Stage A disk.
            if np.sum(red_mask) >= 2 and hasattr(optimize, 'least_squares'):
                red_freq = all_frequencies[red_mask]
                red_flux = np.exp(all_log_fluxes[red_mask])
                red_err_flux = red_flux * all_errors[red_mask]
                logMdot_stage = staged[start_idx]
                av_stage = staged[start_idx + 1]

                try:
                    red_acc = np.exp(accretion_model(
                        red_freq, 10.0 ** logMdot_stage, M, R_star, R_out, R_in, distance, av_stage
                    ))
                except Exception:
                    red_acc = np.zeros_like(red_flux)

                red_excess = red_flux - red_acc
                valid_excess = np.isfinite(red_excess) & (red_excess > 0) & np.isfinite(red_err_flux) & (red_err_flux > 0)
                if np.sum(valid_excess) >= 2:
                    bb_freq = red_freq[valid_excess]
                    bb_excess = red_excess[valid_excess]
                    bb_err = red_err_flux[valid_excess]
                    bb_x0 = np.clip(staged[start_idx + 2:start_idx + 4], lower[2:4], upper[2:4])

                    def bb_residual(two_params):
                        T_bb, logR_bb = two_params
                        try:
                            pred_log = planck_model_custom(
                                bb_freq, T_bb, distance, av_stage, R_bb=10.0 ** logR_bb
                            )
                        except Exception:
                            return np.full(len(bb_freq), 1e6)
                        pred_flux = np.exp(np.where(np.isfinite(pred_log), pred_log, -50.0))
                        if np.any(~np.isfinite(pred_flux)):
                            return np.full(len(bb_freq), 1e6)
                        return (bb_excess - pred_flux) / bb_err

                    try:
                        bb_result = optimize.least_squares(
                            bb_residual,
                            x0=bb_x0,
                            bounds=(lower[2:4], upper[2:4]),
                            ftol=1e-8,
                            xtol=1e-8,
                            gtol=1e-6,
                            max_nfev=2000,
                        )
                        if bb_result.success and np.all(np.isfinite(bb_result.x)):
                            staged[start_idx + 2:start_idx + 4] = bb_result.x
                            stage_info['stage_B_red_excess_bb_days'] += 1
                        else:
                            stage_info['stage_B_failed_days'].append(sorted_jd_days[day_idx])
                    except Exception as exc:
                        stage_info['stage_B_failed_days'].append(sorted_jd_days[day_idx])
                        if debug:
                            print(f"Stage B failed for day {sorted_jd_days[day_idx]}: {exc}")
                else:
                    stage_info['stage_B_failed_days'].append(sorted_jd_days[day_idx])
            else:
                stage_info['stage_B_failed_days'].append(sorted_jd_days[day_idx])

        return staged, stage_info

    def refine_spectral_rede_days_independently(params):
        """Jointly fit disk and blackbody parameters one observing day at a time.

        With no temporal regularization or fitted global parameters, the
        spectral red-excess objective is a sum of independent daily
        objectives.  Solving four parameters per day avoids asking a
        numerical-gradient optimizer to reevaluate every other spectrum for
        each parameter perturbation, while retaining the simultaneous
        accretion + blackbody fit over the full wavelength range.
        """
        refined = np.array(params, dtype=float, copy=True)
        diagnostics = []
        M = default_params['M']
        R_star = default_params['R_star']
        R_out = default_params['R_out']
        distance = default_params['distance']
        R_in = default_params['R_in']
        max_fraction = float(spectral_rede_max_blue_bb_fraction)
        penalty_scale = np.sqrt(float(spectral_rede_blue_penalty_weight))

        for day_idx in range(n_days):
            start_idx = day_idx * n_params_per_day
            day_mask = day_indices == day_idx
            day_frequencies = all_frequencies[day_mask]
            day_wavelengths = all_wavelengths_micron[day_mask]
            day_log_fluxes = all_log_fluxes[day_mask]
            day_errors = all_errors[day_mask]
            blue_mask = day_wavelengths < SPECTRAL_REDE_THRESHOLD_MICRON

            day_bounds = np.array(
                param_bounds[start_idx:start_idx + n_params_per_day], dtype=float
            )
            lower = day_bounds[:, 0]
            upper = day_bounds[:, 1]
            x0 = np.clip(refined[start_idx:start_idx + n_params_per_day], lower, upper)

            n_penalty_residuals = int(np.sum(blue_mask)) if max_fraction > 0 else 0
            residual_size = len(day_frequencies) + n_penalty_residuals

            def joint_residual(four_params):
                logMdot, av, T_bb, logR_bb = four_params
                try:
                    acc_log_flux = accretion_model(
                        day_frequencies,
                        10.0 ** logMdot,
                        M,
                        R_star,
                        R_out,
                        R_in,
                        distance,
                        av,
                    )
                    bb_log_flux = planck_model_custom(
                        day_frequencies,
                        T_bb,
                        distance,
                        av,
                        R_bb=10.0 ** logR_bb,
                    )
                except Exception:
                    return np.full(residual_size, 1e6)

                acc_flux = np.exp(np.where(
                    np.isfinite(acc_log_flux) & (acc_log_flux >= -45.0),
                    acc_log_flux,
                    -50.0,
                ))
                bb_flux = np.exp(np.where(
                    np.isfinite(bb_log_flux) & (bb_log_flux >= -200.0),
                    bb_log_flux,
                    -50.0,
                ))
                total_flux = acc_flux + bb_flux
                if np.any(~np.isfinite(total_flux)) or np.any(total_flux <= 0):
                    return np.full(residual_size, 1e6)

                model_residuals = (day_log_fluxes - np.log(total_flux)) / day_errors
                if max_fraction <= 0:
                    return model_residuals

                penalty_acc_flux = np.exp(np.where(
                    np.isfinite(acc_log_flux[blue_mask]), acc_log_flux[blue_mask], -50.0
                ))
                penalty_bb_flux = np.exp(np.where(
                    np.isfinite(bb_log_flux[blue_mask]), bb_log_flux[blue_mask], -50.0
                ))
                blue_ratio = penalty_bb_flux / np.maximum(penalty_acc_flux, 1e-30)
                penalty_residuals = penalty_scale * np.maximum(
                    0.0, blue_ratio / max_fraction - 1.0
                )
                penalty_residuals = np.where(
                    np.isfinite(penalty_residuals), penalty_residuals, 1e6
                )
                return np.concatenate((model_residuals, penalty_residuals))

            jd_day = sorted_jd_days[day_idx]
            try:
                day_result = optimize.least_squares(
                    joint_residual,
                    x0=x0,
                    bounds=(lower, upper),
                    ftol=optimizer_ftol,
                    xtol=optimizer_ftol,
                    gtol=optimizer_gtol,
                    x_scale=np.maximum(np.abs(x0), 1.0),
                    max_nfev=min(local_solver_max_nfev, optimizer_maxfun),
                )
                if np.all(np.isfinite(day_result.x)):
                    refined[start_idx:start_idx + n_params_per_day] = day_result.x
                diagnostics.append({
                    'JD_day': jd_day,
                    'success': bool(day_result.success),
                    'status': int(day_result.status),
                    'message': str(day_result.message),
                    'nfev': int(day_result.nfev),
                    'njev': int(day_result.njev) if day_result.njev is not None else None,
                    'cost': float(2.0 * day_result.cost),
                    'optimality': float(day_result.optimality),
                })
                state = 'converged' if day_result.success else 'stopped with warning'
                print(
                    f"Stage C day {day_idx + 1}/{n_days} ({jd_day}): {state} "
                    f"after {day_result.nfev} evaluations"
                )
            except Exception as exc:
                diagnostics.append({
                    'JD_day': jd_day,
                    'success': False,
                    'status': -1,
                    'message': str(exc),
                    'nfev': 0,
                    'njev': None,
                    'cost': np.inf,
                    'optimality': np.inf,
                })
                print(f"Stage C day {day_idx + 1}/{n_days} ({jd_day}) failed: {exc}")

        return refined, diagnostics

    varied_initial_guess = initial_guess.copy()
    if not evaluate_only:
        np.random.seed(42)
        for idx in range(n_days * n_params_per_day):
            if idx % n_params_per_day == 0:
                varied_initial_guess[idx] += np.random.uniform(-0.2, 0.2)
            elif idx % n_params_per_day == 1:
                varied_initial_guess[idx] += np.random.uniform(-0.3, 0.3)
            elif spectral_rede_mode:
                if idx % n_params_per_day == 2:
                    varied_initial_guess[idx] += np.random.uniform(-150.0, 150.0)
                else:
                    varied_initial_guess[idx] += np.random.uniform(-0.1, 0.1)
            elif red_excess_mode:
                if fit_param_name == 'T_bb':
                    varied_initial_guess[idx] += np.random.uniform(-200.0, 200.0)
                else:
                    varied_initial_guess[idx] += np.random.uniform(-0.1, 0.1)

    spectral_rede_stage_info = {}
    independent_diagnostics = []
    if spectral_rede_mode and not evaluate_only:
        print("Stage A/B initialization: fitting blue accretion disk, then red excess BB...")
        varied_initial_guess, spectral_rede_stage_info = initialize_spectral_rede_stages(initial_guess)
        print(
            "Stage A/B initialization complete: "
            f"disk {spectral_rede_stage_info.get('stage_A_blue_disk_days', 0)}/{n_days} day(s), "
            f"BB {spectral_rede_stage_info.get('stage_B_red_excess_bb_days', 0)}/{n_days} day(s)"
        )

    direct_two_filter_mode = (
        not evaluate_only
        and data_mode == 'photometry'
        and not red_excess_mode
        and n_global_params == 0
        and n_params_per_day == 2
        and float(lambda_reg) == 0.0
        and all(np.sum(day_indices == day_idx) == 2 for day_idx in range(n_days))
    )
    independent_spectral_rede_mode = (
        not evaluate_only
        and spectral_rede_mode
        and spectral_rede_joint_refine
        and n_global_params == 0
        and float(lambda_reg) == 0.0
    )

    if evaluate_only:
        print("Evaluating fixed parameters without optimization...")
    elif direct_two_filter_mode:
        print("Starting direct two-filter root solve...")
    elif independent_spectral_rede_mode:
        print("Starting independent per-day Stage C joint spectral red-excess refinement...")
    elif spectral_rede_mode and spectral_rede_joint_refine:
        print("Starting Stage C joint spectral red-excess refinement...")
    elif spectral_rede_mode:
        print("Skipping Stage C joint refinement; using Stage A/B parameters")
    else:
        print("Starting optimized regularized optimization...")

    def progress_callback(xk):
        if not hasattr(progress_callback, 'iteration'):
            progress_callback.iteration = 0
            progress_callback.start_time = time.time()
        progress_callback.iteration += 1
        if progress_callback.iteration % 5 == 0:
            elapsed = time.time() - progress_callback.start_time
            print(f"Iteration {progress_callback.iteration}, elapsed: {elapsed:.1f}s")

    try:
        start_time = time.time()
        if hasattr(progress_callback, 'iteration'):
            del progress_callback.iteration
        if evaluate_only:
            class FixedParameterResult:
                pass

            result = FixedParameterResult()
            result.x = np.asarray(initial_guess, dtype=float)
            result.fun = objective_function_optimized(result.x)
            result.success = bool(np.isfinite(result.fun))
            result.message = "Fixed parameters evaluated without optimization."
            result.nfev = 0
            result.nit = 0
        elif direct_two_filter_mode:
            class DirectTwoFilterResult:
                pass

            direct_x, n_direct_days = polish_two_filter_days(varied_initial_guess)
            result = DirectTwoFilterResult()
            result.x = direct_x
            result.fun = objective_function_optimized(direct_x)
            result.success = np.isfinite(result.fun) and n_direct_days == n_days
            result.message = (
                f"Direct two-filter root/least-squares solve completed for "
                f"{n_direct_days}/{n_days} day(s)."
            )
            result.nfev = 0
            result.nit = 0
            print(f"Direct two-filter root solve handled {n_direct_days}/{n_days} day(s)")
        elif independent_spectral_rede_mode:
            class IndependentSpectralRedeResult:
                pass

            independent_x, independent_diagnostics = refine_spectral_rede_days_independently(
                varied_initial_guess
            )
            result = IndependentSpectralRedeResult()
            result.x = independent_x
            result.fun = objective_function_optimized(independent_x)
            result.success = all(item['success'] for item in independent_diagnostics)
            successful_days = sum(item['success'] for item in independent_diagnostics)
            result.message = (
                f"Independent four-parameter joint fits converged for "
                f"{successful_days}/{n_days} day(s)."
            )
            result.nfev = sum(item['nfev'] for item in independent_diagnostics)
        elif spectral_rede_mode and not spectral_rede_joint_refine:
            class SpectralRedeStagedResult:
                pass

            result = SpectralRedeStagedResult()
            result.x = varied_initial_guess
            result.fun = objective_function_optimized(varied_initial_guess)
            result.success = np.isfinite(result.fun)
            result.message = "Spectral red-excess Stage A/B initialization completed without joint refinement."
            result.nfev = 0
            result.nit = 0
        else:
            result = minimize(
                objective_function_optimized,
                x0=varied_initial_guess,
                method='L-BFGS-B',
                bounds=param_bounds,
                callback=progress_callback,
                options={
                    'maxiter': optimizer_maxiter,
                    'maxfun': optimizer_maxfun,
                    'ftol': optimizer_ftol,
                    'gtol': optimizer_gtol,
                    'disp': debug,
                    'maxcor': 20,
                    'maxls': 50,
                }
            )
        optimization_time = time.time() - start_time
        print(f"Optimization completed in {optimization_time:.1f}s after {result.nfev} function evaluations")

        if (
            not evaluate_only
            and not direct_two_filter_mode
            and not independent_spectral_rede_mode
            and not result.success
            and result.nfev < 500
        ):
            print("First optimization didn't converge well, trying refined approach...")
            if hasattr(progress_callback, 'iteration'):
                del progress_callback.iteration
            backup_result = minimize(
                objective_function_optimized,
                x0=result.x,
                method='SLSQP',
                bounds=param_bounds,
                callback=progress_callback,
                options={
                    'maxiter': 500,
                    'ftol': 1e-7,
                    'disp': debug
                }
            )
            if backup_result.fun < result.fun:
                result = backup_result
                print("Backup optimization found better solution")
            else:
                print("Original optimization result kept")

        popt = result.x
        n_polished_days = 0
        if not evaluate_only:
            popt, n_polished_days = polish_two_filter_days(popt)
        if n_polished_days and not direct_two_filter_mode:
            result.fun = objective_function_optimized(popt)
            predicted_polished = model_function_vectorized(popt)
            valid_polished = np.isfinite(predicted_polished) & (predicted_polished > -150)
            polished_chi_squared = (
                np.sum(((all_log_fluxes[valid_polished] - predicted_polished[valid_polished]) / all_errors[valid_polished]) ** 2)
                if np.any(valid_polished) else np.inf
            )
            if np.isfinite(polished_chi_squared) and polished_chi_squared < 1e-8:
                result.success = True
                result.message = "Two-filter photometric days solved directly after L-BFGS-B."
            print(f"Two-filter polish improved {n_polished_days} day(s)")
        if result.success:
            print(f"✓ Converged successfully: {result.message}")
        else:
            print(f"⚠ Optimization completed with warning: {result.message}")
        print(f"Final objective value: {result.fun:.3f}")
        print(f"Function evaluations: {result.nfev}")
        if hasattr(result, 'nit'):
            print(f"Iterations: {result.nit}")

        param_errors = np.zeros(len(popt)) if evaluate_only else np.ones(len(popt)) * 0.1
        if debug and not evaluate_only:
            print("Computing parameter uncertainties...")
            epsilon = 1e-6
            chi_squared_center = np.sum(((all_log_fluxes - model_function_vectorized(popt)) / all_errors) ** 2)
            for idx in range(min(20, len(popt))):
                params_plus = popt.copy()
                params_plus[idx] += epsilon
                chi_squared_plus = np.sum(((all_log_fluxes - model_function_vectorized(params_plus)) / all_errors) ** 2)
                curvature = (chi_squared_plus - chi_squared_center) / (epsilon ** 2)
                if curvature > 0:
                    param_errors[idx] = np.sqrt(1.0 / curvature)

        results = {
            'success': bool(result.success),
            'daily_params': {},
            'global_params': {
                'M': default_params['M'],
                'R_star': default_params['R_star'],
                'R_in': default_params['R_in'],
                'R_out': default_params['R_out'],
                'distance': default_params['distance'],
            },
            'param_errors': {},
            'fit_info': {
                'n_days': n_days,
                'n_data_points': n_data_points,
                'dof': dof,
                'lambda_reg': lambda_reg,
                'required_filters': list(required_filters),
                'fit_filters': list(fit_filters),
                'photometric_wavelength_source': 'row Lambda with filter-table fallback',
                # Fixed model parameters are not repeated in global_params.
                # Store the adopted stellar mass explicitly so downstream
                # M_star*Mdot calculations do not silently assume 1 Msun.
                'stellar_mass_solar': float(default_params['M'] / M_sun),
                'param_names': fitted_param_names,
                'regularize_params': regularize_params,
                'data_mode': data_mode,
                'evaluate_only': bool(evaluate_only),
                'AR_mode': AR_mode,
                'AR_method': 'random-walk Metropolis MCMC' if AR_mode else None,
                'AR_mcmc_chains': AR_MCMC_CHAINS if AR_mode else None,
                'AR_mcmc_steps': AR_MCMC_STEPS if AR_mode else None,
                'AR_mcmc_burn_in': AR_MCMC_BURN_IN if AR_mode else None,
                'AR_mcmc_thin': AR_MCMC_THIN if AR_mode else None,
                'two_filter_direct_solver': direct_two_filter_mode,
                'red_excess_mode': red_excess_mode,
                'red_excess_fit_param': fit_param_name,
                'spectral_rede_threshold_micron': SPECTRAL_REDE_THRESHOLD_MICRON if spectral_rede_mode else None,
                'spectral_rede_joint_refine': bool(spectral_rede_joint_refine) if spectral_rede_mode else None,
                'spectral_rede_independent_joint_refine': bool(independent_spectral_rede_mode) if spectral_rede_mode else None,
                'spectral_rede_independent_diagnostics': (
                    _to_builtin(independent_diagnostics)
                    if independent_spectral_rede_mode and '_to_builtin' in globals()
                    else independent_diagnostics if independent_spectral_rede_mode else None
                ),
                'spectral_rede_max_blue_bb_fraction': float(spectral_rede_max_blue_bb_fraction) if spectral_rede_mode else None,
                'spectral_rede_blue_penalty_weight': float(spectral_rede_blue_penalty_weight) if spectral_rede_mode else None,
                'spectral_rede_stage_info': _to_builtin(spectral_rede_stage_info) if spectral_rede_mode and '_to_builtin' in globals() else spectral_rede_stage_info,
                'fixed_T_bb': fixed_T_bb,
                'fixed_R_bb': fixed_R_bb,
                'optimizer_method': (
                    'fixed parameters (no optimization)'
                    if evaluate_only
                    else 'direct two-filter root/least-squares'
                    if direct_two_filter_mode
                    else 'independent per-day four-parameter least-squares'
                    if independent_spectral_rede_mode
                    else 'L-BFGS-B'
                ),
                'optimizer_maxiter': optimizer_maxiter,
                'optimizer_maxfun': optimizer_maxfun,
                'optimizer_ftol': optimizer_ftol,
                'optimizer_gtol': optimizer_gtol,
                'mdot_bounds_msun_per_year': list(mdot_bounds_msun_per_year),
                'av_bounds': list(av_bounds),
                'local_solver_max_nfev': local_solver_max_nfev,
                'local_solver_ftol': local_solver_ftol,
                'local_solver_gtol': local_solver_gtol,
                'optimizer_nfev': int(result.nfev),
                'optimizer_nit': int(result.nit) if hasattr(result, 'nit') else None,
                'optimizer_message': str(result.message),
            }
        }

        for day_idx in range(n_days):
            jd_day = sorted_jd_days[day_idx]
            if spectral_rede_mode:
                logMdot_opt = popt[day_idx * 4]
                Mdot_opt = 10 ** logMdot_opt / M_sun * year
                Av_opt = popt[day_idx * 4 + 1]
                T_bb_opt = popt[day_idx * 4 + 2]
                logR_bb_opt = popt[day_idx * 4 + 3]
                R_bb_opt = 10 ** logR_bb_opt
                logR_err = param_errors[day_idx * 4 + 3] if day_idx * 4 + 3 < len(param_errors) else 0.1
                results['daily_params'][jd_day] = {
                    'logMdot': logMdot_opt,
                    'Mdot': Mdot_opt,
                    'Av': Av_opt,
                    'logMdot_err': param_errors[day_idx * 4] if day_idx * 4 < len(param_errors) else 0.1,
                    'Mdot_err': np.log(10) * Mdot_opt * param_errors[day_idx * 4] if day_idx * 4 < len(param_errors) else 0.1 * Mdot_opt,
                    'Av_err': param_errors[day_idx * 4 + 1] if day_idx * 4 + 1 < len(param_errors) else 0.1,
                    'T_bb': T_bb_opt,
                    'T_bb_err': param_errors[day_idx * 4 + 2] if day_idx * 4 + 2 < len(param_errors) else 100.0,
                    'logR_bb': logR_bb_opt,
                    'logR_bb_err': logR_err,
                    'R_bb': R_bb_opt,
                    'R_bb_err': np.log(10) * R_bb_opt * logR_err,
                }
            elif red_excess_mode:
                logMdot_opt = popt[day_idx * 3]
                Mdot_opt = 10 ** logMdot_opt / M_sun * year
                Av_opt = popt[day_idx * 3 + 1]
                extra_opt = popt[day_idx * 3 + 2]
                day_result = {
                    'logMdot': logMdot_opt,
                    'Mdot': Mdot_opt,
                    'Av': Av_opt,
                    'logMdot_err': param_errors[day_idx * 3] if day_idx * 3 < len(param_errors) else 0.1,
                    'Mdot_err': np.log(10) * Mdot_opt * param_errors[day_idx * 3] if day_idx * 3 < len(param_errors) else 0.1 * Mdot_opt,
                    'Av_err': param_errors[day_idx * 3 + 1] if day_idx * 3 + 1 < len(param_errors) else 0.1,
                    'T_bb': fixed_T_bb,
                    'T_bb_err': 0.0,
                    'R_bb': fixed_R_bb,
                    'R_bb_err': 0.0,
                }
                if fit_param_name == 'T_bb':
                    day_result['T_bb'] = extra_opt
                    day_result['T_bb_err'] = param_errors[day_idx * 3 + 2] if day_idx * 3 + 2 < len(param_errors) else 100.0
                else:
                    R_bb_opt = 10 ** extra_opt
                    logR_err = param_errors[day_idx * 3 + 2] if day_idx * 3 + 2 < len(param_errors) else 0.1
                    day_result['logR_bb'] = extra_opt
                    day_result['logR_bb_err'] = logR_err
                    day_result['R_bb'] = R_bb_opt
                    day_result['R_bb_err'] = np.log(10) * R_bb_opt * logR_err
                results['daily_params'][jd_day] = day_result
            else:
                logMdot_opt = popt[day_idx * 2]
                Mdot_opt = 10 ** logMdot_opt / M_sun * year
                results['daily_params'][jd_day] = {
                    'logMdot': logMdot_opt,
                    'Mdot': Mdot_opt,
                    'Av': popt[day_idx * 2 + 1],
                    'logMdot_err': param_errors[day_idx * 2] if day_idx * 2 < len(param_errors) else 0.1,
                    'Mdot_err': np.log(10) * Mdot_opt * param_errors[day_idx * 2] if day_idx * 2 < len(param_errors) else 0.1 * Mdot_opt,
                    'Av_err': param_errors[day_idx * 2 + 1] if day_idx * 2 + 1 < len(param_errors) else 0.1,
                }

        for param_name, idx in global_param_map.items():
            results['global_params'][param_name] = popt[idx]
            results['param_errors'][param_name] = param_errors[idx] if idx < len(param_errors) else 0.1

        predicted = model_function_vectorized(popt)
        valid_final = np.isfinite(predicted) & (predicted > -150)
        chi_squared = np.sum(((all_log_fluxes[valid_final] - predicted[valid_final]) / all_errors[valid_final]) ** 2) if np.any(valid_final) else np.inf
        reg_term = regularization_term_vectorized(popt)
        blue_penalty = spectral_rede_blue_penalty(popt)
        results['chi_squared'] = chi_squared
        results['regularization_term'] = reg_term
        results['blue_bb_penalty'] = blue_penalty
        results['total_objective'] = chi_squared + lambda_reg * reg_term + blue_penalty
        results['daily_data'] = daily_data

        if AR_mode and ar_visualization_days:
            print("Generating AR mode 3D MCMC posterior...")
            ar_mcmc_data = generate_ar_mcmc_samples(
                ar_visualization_days, daily_data, fit_filters, default_params,
                global_param_map, popt, results['daily_params'], data_mode=data_mode
            )
            results['AR_mcmc_data'] = ar_mcmc_data

        if red_excess_mode:
            results['red_excess_data'] = red_excess_data

        reduced_chi_squared = chi_squared / dof if dof > 0 else np.inf
        results['reduced_chi_squared'] = reduced_chi_squared
        completion_label = "Fixed-parameter evaluation" if evaluate_only else "Optimized fit"
        print(f"{completion_label} completed in {result.nfev} function evaluations!")
        print(f"Chi-squared: {chi_squared:.2f}")
        print(f"Reduced chi-squared: {reduced_chi_squared:.2f}")
        print(f"Regularization term: {reg_term:.2f}")
        if spectral_rede_mode:
            print(f"Blue BB penalty: {blue_penalty:.2f}")
        return results
    except KeyboardInterrupt:
        print("Optimization interrupted by user")
        return {'success': False, 'error': 'Interrupted by user'}
    except Exception as exc:
        print(f"Optimization failed with error: {str(exc)}")
        return {'success': False, 'error': str(exc)}

__all__ = [
    '_normalize_red_excess_fit_param',
    '_safe_float',
    '_resolve_red_excess_component',
    '_coerce_wavelength_meters',
    '_get_spectral_daily_data',
    '_photometric_frequency_from_row',
    'ultimate_fitting_regularized',
]
