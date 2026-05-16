"""Helpers for the final-panel figure model overlays."""

from __future__ import annotations

try:
    from .ultimate_common import M_sun, R_sun, constants, np, pd, year
    from .ultimate_physics import accretion_model, log_flux_to_magnitude, wavelength_to_frequency
except ImportError:
    from ultimate_common import M_sun, R_sun, constants, np, pd, year
    from ultimate_physics import accretion_model, log_flux_to_magnitude, wavelength_to_frequency


def normalize_filter_selection(filters):
    """Return a clean list of filter names from a string or iterable."""
    if filters is None:
        return []
    if isinstance(filters, str):
        raw_filters = filters.replace('+', ',').split(',')
    elif np.isscalar(filters):
        raw_filters = [filters]
    else:
        raw_filters = filters
    return [str(value).strip() for value in raw_filters if str(value).strip()]


def _mdot_from_params(params):
    mdot = params.get('Mdot')
    if mdot is None and params.get('logMdot') is not None:
        mdot = 10.0 ** float(params['logMdot']) / M_sun * year
    return mdot


def _mdot_err_from_params(params, mdot):
    mdot_err = params.get('Mdot_err')
    try:
        has_mdot = np.isfinite(mdot)
    except (TypeError, ValueError):
        has_mdot = False
    if mdot_err is None and params.get('logMdot_err') is not None and has_mdot:
        mdot_err = float(mdot) * np.log(10.0) * float(params['logMdot_err'])
    return mdot_err


def _interp_error(x, xp, errors):
    exact = np.isclose(float(x), xp, rtol=0.0, atol=1e-6)
    if np.any(exact):
        return float(errors[np.argmax(exact)])

    upper = int(np.searchsorted(xp, float(x), side='right'))
    lower = upper - 1
    if lower < 0 or upper >= len(xp):
        return np.nan

    width = xp[upper] - xp[lower]
    if width <= 0:
        return np.nan
    weight_upper = (float(x) - xp[lower]) / width
    weight_lower = 1.0 - weight_upper
    return float(np.hypot(weight_lower * errors[lower], weight_upper * errors[upper]))


def accretion_model_magnitude_error(freq, zp, mdot, av, mdot_err, av_err, model_params):
    """Propagate Mdot and Av errors to one accretion model magnitude."""
    if not (
        np.isfinite(freq)
        and np.isfinite(zp)
        and zp > 0
        and np.isfinite(mdot)
        and mdot > 0
        and np.isfinite(av)
    ):
        return np.nan

    def mag_for(test_mdot, test_av):
        if not np.isfinite(test_mdot) or test_mdot <= 0 or not np.isfinite(test_av):
            return np.nan
        log_flux = accretion_model(
            np.array([freq]),
            test_mdot * M_sun / year,
            model_params['M'],
            model_params['R_star'],
            model_params['R_out'],
            model_params['R_in'],
            model_params['distance'],
            test_av,
        )[0]
        return float(log_flux_to_magnitude(log_flux, zp))

    terms = []
    if np.isfinite(mdot_err) and mdot_err > 0:
        plus = mag_for(mdot + mdot_err, av)
        minus = mag_for(max(mdot - mdot_err, np.finfo(float).tiny), av)
        if np.isfinite(plus) and np.isfinite(minus):
            terms.append(0.5 * abs(plus - minus))
        elif np.isfinite(plus):
            center = mag_for(mdot, av)
            if np.isfinite(center):
                terms.append(abs(plus - center))

    if np.isfinite(av_err) and av_err > 0:
        plus = mag_for(mdot, av + av_err)
        minus = mag_for(mdot, av - av_err)
        if np.isfinite(plus) and np.isfinite(minus):
            terms.append(0.5 * abs(plus - minus))

    return float(np.hypot(*terms)) if terms else np.nan


def calculate_filter_accretion_models(filter_data, daily_params, global_params=None):
    """
    Calculate accretion-disk magnitudes for each wavelength in a filter panel.

    The fitted daily Mdot and Av values are available only on selected days. For
    observed photometry days between fitted days, Mdot and Av are linearly
    interpolated from their neighbouring fitted (Mdot, Av) pair.
    """
    if filter_data is None or len(filter_data) == 0:
        return {}
    if not daily_params:
        return {}

    required_columns = {'JD', 'Filter', 'Lambda', 'ZP'}
    missing = required_columns - set(filter_data.columns)
    if missing:
        raise ValueError(f"filter_data is missing required columns: {sorted(missing)}")

    param_rows = []
    for jd, params in daily_params.items():
        mdot = _mdot_from_params(params)
        av = params.get('Av')
        mdot_err = _mdot_err_from_params(params, mdot)
        av_err = params.get('Av_err')
        try:
            mdot_err_value = float(mdot_err) if np.isfinite(mdot_err) else np.nan
        except (TypeError, ValueError):
            mdot_err_value = np.nan
        try:
            av_err_value = float(av_err) if np.isfinite(av_err) else np.nan
        except (TypeError, ValueError):
            av_err_value = np.nan
        try:
            row_is_finite = np.isfinite(jd) and np.isfinite(mdot) and np.isfinite(av)
        except (TypeError, ValueError):
            row_is_finite = False
        if row_is_finite:
            param_rows.append((
                float(jd),
                float(mdot),
                float(av),
                mdot_err_value,
                av_err_value,
            ))

    if len(param_rows) < 2:
        return {}

    param_rows.sort(key=lambda row: row[0])
    param_jds = np.array([row[0] for row in param_rows], dtype=float)
    mdot_values = np.array([row[1] for row in param_rows], dtype=float)
    av_values = np.array([row[2] for row in param_rows], dtype=float)
    mdot_errors = np.array([row[3] for row in param_rows], dtype=float)
    av_errors = np.array([row[4] for row in param_rows], dtype=float)

    model_params = {
        'M': 0.5 * M_sun,
        'R_star': 3.0 * R_sun,
        'R_out': 2 * constants.au,
        'distance': 700 * constants.parsec,
    }
    if global_params:
        model_params.update(global_params)
    r_in = model_params.get('R_in', model_params['R_star'])

    model_by_series = {}
    for (filter_name, lambda_value), lambda_data in filter_data.groupby(['Filter', 'Lambda'], sort=True):
        filter_name = str(filter_name)
        lambda_value = float(lambda_value)
        lambda_data = lambda_data.sort_values('JD').copy()
        jd_values = pd.to_numeric(lambda_data['JD'], errors='coerce').to_numpy(dtype=float)
        zp_values = pd.to_numeric(lambda_data['ZP'], errors='coerce').to_numpy(dtype=float)
        in_range = (
            np.isfinite(jd_values)
            & np.isfinite(zp_values)
            & (zp_values > 0)
            & (jd_values >= param_jds[0])
            & (jd_values <= param_jds[-1])
        )
        if not np.any(in_range):
            continue

        jd_eval = jd_values[in_range]
        zp_eval = zp_values[in_range]
        mdot_eval = np.interp(jd_eval, param_jds, mdot_values)
        av_eval = np.interp(jd_eval, param_jds, av_values)
        freq = wavelength_to_frequency(lambda_value)
        is_interpolated = ~np.isclose(jd_eval[:, None], param_jds[None, :], rtol=0.0, atol=1e-6).any(axis=1)

        magnitudes = []
        magnitude_errors = []
        for mdot_solar_per_year, av, zp in zip(mdot_eval, av_eval, zp_eval):
            log_flux = accretion_model(
                np.array([freq]),
                mdot_solar_per_year * M_sun / year,
                model_params['M'],
                model_params['R_star'],
                model_params['R_out'],
                r_in,
                model_params['distance'],
                av,
            )[0]
            magnitudes.append(log_flux_to_magnitude(log_flux, zp))
        for jd, mdot_solar_per_year, av, zp, interpolated in zip(jd_eval, mdot_eval, av_eval, zp_eval, is_interpolated):
            mdot_err = _interp_error(jd, param_jds, mdot_errors)
            av_err = _interp_error(jd, param_jds, av_errors)
            mag_err = accretion_model_magnitude_error(
                freq,
                zp,
                mdot_solar_per_year,
                av,
                mdot_err,
                av_err,
                model_params | {'R_in': r_in},
            )
            magnitude_errors.append(mag_err * 1.1 if interpolated and np.isfinite(mag_err) else mag_err)

        model_by_series[(filter_name, lambda_value)] = pd.DataFrame({
            'JD': jd_eval,
            'Filter': filter_name,
            'Lambda': lambda_value,
            'Mdot': mdot_eval,
            'Av': av_eval,
            'Mag_model': magnitudes,
            'Mag_model_err': magnitude_errors,
            'ZP': zp_eval,
            'Is_interpolated_fit': is_interpolated,
        }).sort_values('JD')

    return model_by_series


__all__ = [
    'accretion_model_magnitude_error',
    'calculate_filter_accretion_models',
    'normalize_filter_selection',
]
