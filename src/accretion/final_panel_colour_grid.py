"""Colour-colour and colour-magnitude grid for the final panel figure."""

from __future__ import annotations

try:
    from .ultimate_common import (
        M_sun, R_sun, RED_EXCESS_DEFAULT_R_BB, RED_EXCESS_DEFAULT_T_BB,
        colors, constants, mpl, np, pd, plt, year
    )
    from .ultimate_visualisation import _generated_figure_path, julian_to_calendar
    from .ultimate_physics import accretion_model, flux_to_magnitude, log_flux_to_magnitude, planck_model_custom, wavelength_to_frequency
except ImportError:
    from ultimate_common import (
        M_sun, R_sun, RED_EXCESS_DEFAULT_R_BB, RED_EXCESS_DEFAULT_T_BB,
        colors, constants, mpl, np, pd, plt, year
    )
    from ultimate_visualisation import _generated_figure_path, julian_to_calendar
    from ultimate_physics import accretion_model, flux_to_magnitude, log_flux_to_magnitude, planck_model_custom, wavelength_to_frequency


EXTINCTION_COEFFS = {
    'I': 0.607,
    'J': 0.28760574,
    'H': 0.17830579,
    'K': 0.11701314,
}

PANEL_SPECS = [
    {
        'title': 'J vs J-H',
        'plot_key': 'J,J-H',
        'filters': ('J', 'H'),
        'x': ('color', 'J', 'H'),
        'y': ('mag', 'J'),
        'xlabel': 'J-H',
        'ylabel': 'J',
        'invert_x': False,
        'invert_y': True,
    },
    {
        'title': 'J-H vs H-K',
        'plot_key': 'J-H,H-K',
        'filters': ('J', 'H', 'K'),
        'x': ('color', 'H', 'K'),
        'y': ('color', 'J', 'H'),
        'xlabel': 'H-K',
        'ylabel': 'J-H',
        'invert_x': False,
        'invert_y': False,
    },
    {
        'title': 'I vs I-J',
        'plot_key': 'I,I-J',
        'filters': ('I', 'J'),
        'x': ('color', 'I', 'J'),
        'y': ('mag', 'I'),
        'xlabel': 'I-J',
        'ylabel': 'I',
        'invert_x': False,
        'invert_y': True,
    },
    {
        'title': 'K vs H-K',
        'plot_key': 'K,H-K',
        'filters': ('H', 'K'),
        'x': ('color', 'H', 'K'),
        'y': ('mag', 'K'),
        'xlabel': 'H-K',
        'ylabel': 'K',
        'invert_x': False,
        'invert_y': True,
    },
]


def _normalise_plot_key(value):
    return str(value).replace(' ', '').lower()


def _expr_label(expr):
    if expr[0] == 'mag':
        return expr[1]
    if expr[0] == 'color':
        return f'{expr[1]}-{expr[2]}'
    raise ValueError(f'Unknown expression type: {expr[0]}')


def _parse_plot_expr(value):
    text = str(value).strip().replace(' ', '')
    if not text:
        raise ValueError('Empty Panel 6 expression.')
    parts = [part for part in text.split('-') if part]
    if len(parts) == 1:
        return ('mag', parts[0])
    if len(parts) == 2:
        return ('color', parts[0], parts[1])
    raise ValueError(f"Could not parse Panel 6 expression {value!r}. Use FILTER or FILTER-FILTER.")


def _expr_filters(expr):
    if expr[0] == 'mag':
        return [expr[1]]
    if expr[0] == 'color':
        return [expr[1], expr[2]]
    raise ValueError(f'Unknown expression type: {expr[0]}')


def _build_panel_spec(plot_key):
    parts = [part.strip() for part in str(plot_key).split(',')]
    if len(parts) != 2 or not all(parts):
        raise ValueError(f"Could not parse Panel 6 combination {plot_key!r}. Use 'Y,X', e.g. 'J,J-H'.")

    y_expr = _parse_plot_expr(parts[0])
    x_expr = _parse_plot_expr(parts[1])
    filters = tuple(dict.fromkeys(_expr_filters(y_expr) + _expr_filters(x_expr)))
    y_label = _expr_label(y_expr)
    x_label = _expr_label(x_expr)
    return {
        'title': f'{y_label} vs {x_label}',
        'plot_key': f'{y_label},{x_label}',
        'filters': filters,
        'x': x_expr,
        'y': y_expr,
        'xlabel': x_label,
        'ylabel': y_label,
        'invert_x': x_expr[0] == 'mag',
        'invert_y': y_expr[0] == 'mag',
    }


def _selected_panel_specs(plot):
    if plot is None:
        return list(PANEL_SPECS)
    if isinstance(plot, str):
        plot = (plot,)

    selected = []
    seen = set()
    for value in plot:
        spec = _build_panel_spec(value)
        key = _normalise_plot_key(spec['plot_key'])
        if key not in seen:
            selected.append(spec)
            seen.add(key)

    if not selected:
        raise ValueError("Panel 6 plot selection is empty.")
    return selected


def _is_raw_photometry(value):
    if pd.isna(value):
        return True
    return str(value).strip().lower() in {'', '0', 'false', 'n', 'no', 'raw'}


def _is_plottable_flag(value):
    if pd.isna(value):
        return True
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value).strip().lower() not in {'0', 'false', 'n', 'no'}
    return float(numeric) != 0.0


def _clean_photometry(df, required_filters):
    required_filters = {str(item) for item in required_filters}
    data = df[df['Filter'].astype(str).isin(required_filters)].copy()
    if 'Interpolated' in data.columns:
        data = data[data['Interpolated'].map(_is_raw_photometry)].copy()
    if 'Flag' in data.columns:
        data = data[data['Flag'].map(_is_plottable_flag)].copy()

    for col in ['JD', 'Mag', 'Magerr', 'Lambda', 'ZP']:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    data = data.dropna(subset=['JD', 'Filter', 'Mag', 'Magerr', 'Lambda', 'ZP'])
    data = data[np.isfinite(data['JD']) & np.isfinite(data['Mag'])]
    data = data.reset_index(drop=False).rename(columns={'index': '_source_index'})
    return data.sort_values(['JD', 'Filter', 'Magerr'])


def _best_row(rows, target_jd=None):
    rows = rows.copy()
    if target_jd is not None:
        rows['_jd_distance'] = np.abs(rows['JD'] - target_jd)
        sort_cols = ['_jd_distance', 'Magerr', 'JD']
    else:
        sort_cols = ['Magerr', 'JD']
    return rows.sort_values(sort_cols).iloc[0]


def _add_match(matches, seen, selected, anchor_jd, anchor_filter, match_kind):
    row_ids = tuple(sorted(int(row['_source_index']) for row in selected.values()))
    if row_ids in seen:
        return
    seen.add(row_ids)
    match = {
        'JD': float(anchor_jd),
        'Date': str(selected[anchor_filter].get('Date', '')),
        'AnchorFilter': anchor_filter,
        'MatchKind': match_kind,
    }
    for filter_name, row in selected.items():
        prefix = str(filter_name)
        match[f'{prefix}_JD'] = float(row['JD'])
        match[f'{prefix}_Date'] = str(row.get('Date', ''))
        match[f'{prefix}_Mag'] = float(row['Mag'])
        match[f'{prefix}_Magerr'] = float(row['Magerr'])
        match[f'{prefix}_Lambda'] = float(row['Lambda'])
        match[f'{prefix}_ZP'] = float(row['ZP'])
        if 'Flag' in row.index:
            match[f'{prefix}_Flag'] = row.get('Flag')
        if 'Interpolated' in row.index:
            match[f'{prefix}_Interpolated'] = row.get('Interpolated')
    matches.append(match)


def find_colour_grid_matches(df, filters, tolerance=2.0):
    """
    Match raw photometry for a filter set.

    First use exact same-JD matches, then anchor on each filter in turn and
    accept nearest neighbouring observations within +/- tolerance days.
    """
    filters = tuple(dict.fromkeys(str(item) for item in filters))
    data = _clean_photometry(df, filters)
    if data.empty:
        return pd.DataFrame()

    by_filter = {name: data[data['Filter'].astype(str) == name].copy() for name in filters}
    if any(frame.empty for frame in by_filter.values()):
        return pd.DataFrame()

    matches = []
    seen = set()

    common_jds = sorted(set.intersection(*[set(frame['JD'].to_numpy(float)) for frame in by_filter.values()]))
    for jd in common_jds:
        selected = {name: _best_row(frame[np.isclose(frame['JD'], jd, rtol=0.0, atol=1e-8)]) for name, frame in by_filter.items()}
        _add_match(matches, seen, selected, jd, filters[0], 'same_jd')

    for anchor_filter in filters:
        for _, anchor_row in by_filter[anchor_filter].sort_values('JD').iterrows():
            anchor_jd = float(anchor_row['JD'])
            selected = {anchor_filter: anchor_row}
            for other_filter in filters:
                if other_filter == anchor_filter:
                    continue
                candidates = by_filter[other_filter]
                candidates = candidates[np.abs(candidates['JD'] - anchor_jd) <= tolerance]
                if candidates.empty:
                    selected = None
                    break
                selected[other_filter] = _best_row(candidates, target_jd=anchor_jd)
            if selected is not None:
                _add_match(matches, seen, selected, anchor_jd, anchor_filter, 'neighbour')

    if not matches:
        return pd.DataFrame()
    return pd.DataFrame(matches).sort_values(['JD', 'MatchKind']).reset_index(drop=True)


def _expr_value(row, expr, prefix=''):
    if expr[0] == 'mag':
        filter_name = expr[1]
        return row[f'{filter_name}_{prefix}Mag']
    if expr[0] == 'color':
        left, right = expr[1], expr[2]
        return row[f'{left}_{prefix}Mag'] - row[f'{right}_{prefix}Mag']
    raise ValueError(f'Unknown expression type: {expr[0]}')


def _expr_error(row, expr, prefix=''):
    if expr[0] == 'mag':
        return row[f'{expr[1]}_{prefix}Magerr']
    if expr[0] == 'color':
        left, right = expr[1], expr[2]
        return np.hypot(row[f'{left}_{prefix}Magerr'], row[f'{right}_{prefix}Magerr'])
    raise ValueError(f'Unknown expression type: {expr[0]}')


def _expr_extinction_delta(expr):
    try:
        if expr[0] == 'mag':
            return EXTINCTION_COEFFS[expr[1]]
        if expr[0] == 'color':
            return EXTINCTION_COEFFS[expr[1]] - EXTINCTION_COEFFS[expr[2]]
    except KeyError:
        return None
    raise ValueError(f'Unknown expression type: {expr[0]}')


def _parameter_table(daily_params):
    rows = []
    for jd, params in daily_params.items():
        mdot = params.get('Mdot')
        if mdot is None and params.get('logMdot') is not None:
            mdot = 10.0 ** float(params['logMdot']) / M_sun * year
        mdot_err = params.get('Mdot_err')
        try:
            has_mdot = np.isfinite(mdot)
        except (TypeError, ValueError):
            has_mdot = False
        if mdot_err is None and params.get('logMdot_err') is not None and has_mdot:
            mdot_err = float(mdot) * np.log(10.0) * float(params['logMdot_err'])
        av = params.get('Av')
        av_err = params.get('Av_err')
        try:
            row_is_finite = np.isfinite(jd) and np.isfinite(mdot) and np.isfinite(av)
        except (TypeError, ValueError):
            row_is_finite = False
        if row_is_finite:
            try:
                mdot_err_value = float(mdot_err) if np.isfinite(mdot_err) else np.nan
            except (TypeError, ValueError):
                mdot_err_value = np.nan
            try:
                av_err_value = float(av_err) if np.isfinite(av_err) else np.nan
            except (TypeError, ValueError):
                av_err_value = np.nan
            rows.append((float(jd), float(mdot), float(av), mdot_err_value, av_err_value))
    rows.sort(key=lambda item: item[0])
    return rows


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


def _model_magnitude_error(freq, zp, mdot, av, mdot_err, av_err, model_params, red_excess_mode=False, t_bb=np.nan, r_bb=np.nan):
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
        total_flux = float(np.exp(log_flux))
        if red_excess_mode and np.isfinite(t_bb) and np.isfinite(r_bb):
            bb_log_flux = planck_model_custom(
                np.array([freq]),
                t_bb,
                model_params['distance'],
                test_av,
                R_bb=r_bb,
            )[0]
            if np.isfinite(bb_log_flux):
                total_flux += float(np.exp(bb_log_flux))
        return float(flux_to_magnitude(total_flux, zp))

    terms = []
    if np.isfinite(mdot_err) and mdot_err > 0:
        plus = mag_for(mdot + mdot_err, av)
        minus = mag_for(max(mdot - mdot_err, np.finfo(float).tiny), av)
        if np.isfinite(plus) and np.isfinite(minus):
            terms.append(0.5 * abs(plus - minus))
    if np.isfinite(av_err) and av_err > 0:
        plus = mag_for(mdot, av + av_err)
        minus = mag_for(mdot, av - av_err)
        if np.isfinite(plus) and np.isfinite(minus):
            terms.append(0.5 * abs(plus - minus))
    return float(np.hypot(*terms)) if terms else np.nan


def _red_excess_table(daily_params, fit_info=None):
    fit_info = fit_info or {}
    fixed_t = float(fit_info.get('fixed_T_bb', RED_EXCESS_DEFAULT_T_BB))
    fixed_r = float(fit_info.get('fixed_R_bb', RED_EXCESS_DEFAULT_R_BB))
    rows = []
    for jd, params in daily_params.items():
        if not np.isfinite(jd):
            continue
        t_bb = params.get('T_bb', fixed_t)
        if params.get('R_bb') is not None:
            r_bb = params.get('R_bb')
        elif params.get('logR_bb') is not None:
            r_bb = 10.0 ** float(params['logR_bb'])
        else:
            r_bb = fixed_r
        if np.isfinite(t_bb) and np.isfinite(r_bb):
            rows.append((float(jd), float(t_bb), float(r_bb)))
    rows.sort(key=lambda item: item[0])
    return rows


def _model_params(global_params):
    params = {
        'M': 0.5 * M_sun,
        'R_star': 3.0 * R_sun,
        'R_out': 2.0 * constants.au,
        'distance': 700.0 * constants.parsec,
    }
    if global_params:
        params.update(global_params)
    params['R_in'] = params.get('R_in', params['R_star'])
    return params


def _add_model_columns(matches, filters, daily_params, global_params=None, fit_info=None):
    param_rows = _parameter_table(daily_params)
    if len(param_rows) < 1 or matches.empty:
        return matches.iloc[0:0].copy()

    fit_info = fit_info or {}
    red_excess_mode = bool(fit_info.get('red_excess_mode', False))
    red_rows = _red_excess_table(daily_params, fit_info) if red_excess_mode else []
    param_jds = np.array([row[0] for row in param_rows], dtype=float)
    mdots = np.array([row[1] for row in param_rows], dtype=float)
    avs = np.array([row[2] for row in param_rows], dtype=float)
    mdot_errors = np.array([row[3] for row in param_rows], dtype=float)
    av_errors = np.array([row[4] for row in param_rows], dtype=float)
    red_jds = np.array([row[0] for row in red_rows], dtype=float) if red_rows else np.array([], dtype=float)
    t_values = np.array([row[1] for row in red_rows], dtype=float) if red_rows else np.array([], dtype=float)
    r_values = np.array([row[2] for row in red_rows], dtype=float) if red_rows else np.array([], dtype=float)
    model_params = _model_params(global_params)

    model_rows = []
    for _, row in matches.iterrows():
        jd = float(row['JD'])
        if jd < param_jds[0] or jd > param_jds[-1]:
            continue
        out = row.to_dict()
        out['Model_Mdot'] = float(np.interp(jd, param_jds, mdots))
        out['Model_Av'] = float(np.interp(jd, param_jds, avs))
        interpolated_filter_fit = []
        valid_model_row = True
        for filter_name in filters:
            eval_jd = float(row.get(f'{filter_name}_JD', jd))
            if eval_jd < param_jds[0] or eval_jd > param_jds[-1]:
                valid_model_row = False
                break
            mdot = float(np.interp(eval_jd, param_jds, mdots))
            av = float(np.interp(eval_jd, param_jds, avs))
            mdot_err = _interp_error(eval_jd, param_jds, mdot_errors)
            av_err = _interp_error(eval_jd, param_jds, av_errors)
            freq = wavelength_to_frequency(float(row[f'{filter_name}_Lambda']))
            zp = float(row[f'{filter_name}_ZP'])
            log_flux = accretion_model(
                np.array([freq]),
                mdot * M_sun / year,
                model_params['M'],
                model_params['R_star'],
                model_params['R_out'],
                model_params['R_in'],
                model_params['distance'],
                av,
            )[0]
            model_flux = float(np.exp(log_flux))
            t_bb = np.nan
            r_bb = np.nan
            if red_excess_mode and len(red_rows) > 0:
                t_bb = float(np.interp(eval_jd, red_jds, t_values))
                r_bb = float(np.interp(eval_jd, red_jds, r_values))
                bb_log_flux = planck_model_custom(
                    np.array([freq]),
                    t_bb,
                    model_params['distance'],
                    av,
                    R_bb=r_bb,
                )[0]
                if np.isfinite(bb_log_flux):
                    model_flux += float(np.exp(bb_log_flux))
                out[f'{filter_name}_Model_T_bb'] = t_bb
                out[f'{filter_name}_Model_R_bb'] = r_bb

            out[f'{filter_name}_Model_Mag'] = float(flux_to_magnitude(model_flux, zp))
            mag_err = _model_magnitude_error(
                freq,
                zp,
                mdot,
                av,
                mdot_err,
                av_err,
                model_params,
                red_excess_mode=red_excess_mode,
                t_bb=t_bb,
                r_bb=r_bb,
            )
            is_interpolated_filter_fit = not np.isclose(eval_jd, param_jds, rtol=0.0, atol=1e-6).any()
            out[f'{filter_name}_Model_Magerr'] = (
                mag_err * 1.1 if is_interpolated_filter_fit and np.isfinite(mag_err) else mag_err
            )
            out[f'{filter_name}_Accretion_Mag'] = float(log_flux_to_magnitude(log_flux, zp))
            out[f'{filter_name}_Model_JD'] = eval_jd
            out[f'{filter_name}_Model_Mdot'] = mdot
            out[f'{filter_name}_Model_Av'] = av
            interpolated_filter_fit.append(is_interpolated_filter_fit)
        if not valid_model_row:
            continue
        out['Is_interpolated_fit'] = bool(any(interpolated_filter_fit))
        model_rows.append(out)

    if not model_rows:
        return matches.iloc[0:0].copy()
    return pd.DataFrame(model_rows).sort_values('JD').reset_index(drop=True)


def _year_from_jd(jd):
    date_text = julian_to_calendar([float(jd)])[0]
    return int(date_text.split('-')[0])


def _discrete_cmap_colours(cmap_name):
    cmap = mpl.colormaps[cmap_name]
    if hasattr(cmap, 'colors'):
        return list(cmap.colors)
    return [cmap(index / max(cmap.N - 1, 1)) for index in range(cmap.N)]


def _build_year_colours(jds, cmap_name, pre_range_colour, min_year=2007, colour_order=None):
    years = [_year_from_jd(jd) for jd in jds]
    max_year = max([year_value for year_value in years if year_value >= min_year], default=min_year)
    post_years = list(range(min_year, max_year + 1))
    cmap_colours = _discrete_cmap_colours(cmap_name)
    if colour_order is not None:
        cmap_colours = [cmap_colours[index] for index in colour_order if index < len(cmap_colours)]
    year_colours = {
        year_value: cmap_colours[index % len(cmap_colours)]
        for index, year_value in enumerate(post_years)
    }
    return [pre_range_colour if year_value < min_year else year_colours[year_value] for year_value in years], year_colours


def _calendar_to_jd(year_value, month=1, day=1):
    """Gregorian calendar date to Julian Date at noon."""
    year_int = int(year_value)
    month_int = int(month)
    day_int = int(day)
    if month_int <= 2:
        year_int -= 1
        month_int += 12
    a = year_int // 100
    b = 2 - a + a // 4
    return (
        int(365.25 * (year_int + 4716))
        + int(30.6001 * (month_int + 1))
        + day_int
        + b
        - 1524.5
    )


def _add_av_grid(ax, spec):
    dx = _expr_extinction_delta(spec['x'])
    dy = _expr_extinction_delta(spec['y'])
    if dx is None or dy is None:
        return
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    length = float(np.hypot(dx, dy))
    if not np.isfinite(length) or length == 0:
        return

    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    corners = np.array([[x0, y0], [x0, y1], [x1, y0], [x1, y1]], dtype=float)
    perp = corners[:, 0] * px + corners[:, 1] * py
    main = corners[:, 0] * ux + corners[:, 1] * uy
    perp_positions = np.linspace(perp.min() - 0.1 * np.ptp(perp), perp.max() + 0.1 * np.ptp(perp), 12)
    main_min = main.min() - 0.12 * np.ptp(main)
    main_max = main.max() + 0.12 * np.ptp(main)
    arrow_len = 0.08 * max(abs(x1 - x0), abs(y1 - y0))

    for idx, perp_pos in enumerate(perp_positions):
        center_x = perp_pos * px
        center_y = perp_pos * py
        xs = [center_x + main_min * ux, center_x + main_max * ux]
        ys = [center_y + main_min * uy, center_y + main_max * uy]
        label = r'$A_V$' if idx == 0 else None
        ax.plot(xs, ys, color='0.84', linewidth=0.8, alpha=0.75, zorder=0, label=label)
        for frac in (0.35, 0.65):
            start_main = main_min + frac * (main_max - main_min)
            start_x = center_x + start_main * ux
            start_y = center_y + start_main * uy
            ax.annotate(
                '',
                xy=(start_x + arrow_len * ux, start_y + arrow_len * uy),
                xytext=(start_x, start_y),
                arrowprops={
                    'arrowstyle': '-|>',
                    'color': '0.78',
                    'alpha': 0.65,
                    'linewidth': 0.8,
                    'mutation_scale': 27,
                    'shrinkA': 0,
                    'shrinkB': 0,
                },
                zorder=0,
            )
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)


def _colourbar(fig, axes, jds, data_year_colours, fit_year_colours, show_fit=True, min_year=2007):
    if len(jds) == 0:
        return
    observed_years = [_year_from_jd(jd) for jd in jds]
    max_year = max([year_value for year_value in observed_years if year_value >= min_year], default=min_year)
    year_bins = [f'<{min_year + 1}'] + list(range(min_year, max_year + 1))

    data_row = [colors.to_rgba('black')]
    fit_row = [colors.to_rgba('0.25')]
    for year_value in year_bins[1:]:
        data_row.append(colors.to_rgba(data_year_colours[year_value]))
        fit_row.append(colors.to_rgba(fit_year_colours[year_value]))
    image_rows = [data_row, fit_row] if show_fit else [data_row]
    image = np.transpose(np.array(image_rows, dtype=float), (1, 0, 2))
    column_count = image.shape[1]

    # Keep the time key vertical on the right so it does not compete with the
    # x-axis labels of the bottom panels.  The Gregorian year and abbreviated
    # JD labels live on opposite sides of the strip for legibility.
    cbar_width = 0.060 if show_fit else 0.036
    cax = fig.add_axes([0.89, 0.09, cbar_width, 0.82])
    cax.imshow(
        image,
        aspect='auto',
        interpolation='nearest',
        origin='upper',
        extent=(0, column_count, len(year_bins) - 0.5, -0.5),
    )
    cax.set_ylim(len(year_bins) - 0.5, -0.5)
    if show_fit:
        cax.set_xticks([0.5, 1.5])
        cax.set_xticklabels(['Data', 'Fit'], fontsize=9, rotation=45, ha='left')
    else:
        cax.set_xticks([0.5])
        cax.set_xticklabels(['Data'], fontsize=9, rotation=45, ha='left')
    jd_labels = []
    for year_value in year_bins:
        if isinstance(year_value, str):
            pre_jds = [float(jd) for jd, jd_year in zip(jds, observed_years) if jd_year < min_year]
            representative_jd = min(pre_jds) if pre_jds else min(float(jd) for jd in jds)
        else:
            representative_jd = _calendar_to_jd(year_value, 1, 1)
        jd_labels.append(str(int(representative_jd))[:5])

    cax.set_yticks(range(len(year_bins)))
    cax.set_yticklabels(jd_labels, fontsize=8)
    cax.yaxis.tick_right()
    cax.yaxis.set_label_position('right')
    cax.tick_params(axis='x', bottom=False, top=True, labelbottom=False, labeltop=True, pad=2, length=2)
    cax.tick_params(axis='y', left=False, right=True, labelleft=False, labelright=True, pad=2, length=2)
    for spine in cax.spines.values():
        spine.set_linewidth(0.6)

    year_axis = cax.secondary_yaxis('left')
    year_axis.set_yticks(range(len(year_bins)))
    year_axis.set_yticklabels([str(year_value) for year_value in year_bins], fontsize=8)
    year_axis.tick_params(axis='y', pad=2, length=2)
    cax.set_ylabel('JD', fontsize=10, labelpad=4)
    year_axis.set_ylabel('Gregorian year', fontsize=10, labelpad=4)


def _plot_one(ax, spec, matches, model_matches, data_colours, fit_colours, show_fit=True):
    if matches.empty:
        ax.set_title(spec['title'])
        ax.text(0.5, 0.5, 'No matched raw data', transform=ax.transAxes, ha='center', va='center')
        return

    matches = matches.sort_values('JD').reset_index(drop=True)
    flag_cols = [f'{filter_name}_Flag' for filter_name in spec['filters'] if f'{filter_name}_Flag' in matches.columns]
    interpolated_cols = [
        f'{filter_name}_Interpolated'
        for filter_name in spec['filters']
        if f'{filter_name}_Interpolated' in matches.columns
    ]
    quality_cols = flag_cols + interpolated_cols
    if quality_cols:
        plottable = np.ones(len(matches), dtype=bool)
        if flag_cols:
            plottable &= matches[flag_cols].apply(
                lambda row: all(_is_plottable_flag(value) for value in row),
                axis=1,
            ).to_numpy(bool)
        if interpolated_cols:
            plottable &= matches[interpolated_cols].apply(
                lambda row: all(_is_raw_photometry(value) for value in row),
                axis=1,
            ).to_numpy(bool)
        matches = matches[plottable].reset_index(drop=True)
        data_colours = np.array(data_colours, dtype=object)[plottable].tolist()
        if model_matches is not None and not model_matches.empty and len(model_matches) == len(plottable):
            model_matches = model_matches[plottable].reset_index(drop=True)
            fit_colours = np.array(fit_colours, dtype=object)[plottable].tolist()
        if matches.empty:
            ax.set_title(spec['title'])
            ax.text(0.5, 0.5, 'No plottable matched raw data', transform=ax.transAxes, ha='center', va='center')
            return

    x_data = matches.apply(lambda row: _expr_value(row, spec['x']), axis=1).to_numpy(float)
    y_data = matches.apply(lambda row: _expr_value(row, spec['y']), axis=1).to_numpy(float)
    x_err = matches.apply(lambda row: _expr_error(row, spec['x']), axis=1).to_numpy(float)
    y_err = matches.apply(lambda row: _expr_error(row, spec['y']), axis=1).to_numpy(float)

    ax.plot(x_data, y_data, linestyle='-', color='0.35', linewidth=1.1, alpha=0.65, zorder=1)
    ax.errorbar(x_data, y_data, xerr=x_err, yerr=y_err, fmt='none', ecolor='0.35', elinewidth=0.7, alpha=0.55, zorder=2)
    ax.scatter(x_data, y_data, marker='o', s=42, c=data_colours, edgecolor='black', linewidth=0.35, zorder=3, label='Data')

    if show_fit and model_matches is not None and not model_matches.empty:
        model_matches = model_matches.sort_values('JD').reset_index(drop=True)
        x_fit = model_matches.apply(lambda row: _expr_value(row, spec['x'], prefix='Model_'), axis=1).to_numpy(float)
        y_fit = model_matches.apply(lambda row: _expr_value(row, spec['y'], prefix='Model_'), axis=1).to_numpy(float)
        x_fit_err = model_matches.apply(lambda row: _expr_error(row, spec['x'], prefix='Model_'), axis=1).to_numpy(float)
        y_fit_err = model_matches.apply(lambda row: _expr_error(row, spec['y'], prefix='Model_'), axis=1).to_numpy(float)
        ax.plot(x_fit, y_fit, linestyle='--', color='0.25', linewidth=1.0, alpha=0.45, zorder=1)
        ax.errorbar(x_fit, y_fit, xerr=x_fit_err, yerr=y_fit_err, fmt='none', ecolor='0.25', elinewidth=0.7, alpha=0.5, zorder=3)
        if 'Is_interpolated_fit' in model_matches.columns:
            interpolated = model_matches['Is_interpolated_fit'].astype(bool).to_numpy()
        else:
            interpolated = np.zeros(len(model_matches), dtype=bool)
        fit_colours = np.array(fit_colours, dtype=object)
        direct = ~interpolated
        if np.any(direct):
            ax.scatter(x_fit[direct], y_fit[direct], marker='P', s=48, c=fit_colours[direct].tolist(), alpha=0.75, zorder=4, label='Fit direct')
        if np.any(interpolated):
            ax.scatter(x_fit[interpolated], y_fit[interpolated], marker='x', s=52, c=fit_colours[interpolated].tolist(), alpha=0.8, zorder=4, label='Fit interpolated')

    ax.set_title(spec['title'])
    ax.set_xlabel(spec['xlabel'])
    ax.set_ylabel(spec['ylabel'])
    ax.grid(True, alpha=0.25)
    if spec.get('invert_x'):
        ax.invert_xaxis()
    if spec.get('invert_y'):
        ax.invert_yaxis()
    _add_av_grid(ax, spec)


def plot_final_panel_colour_grid(
    df,
    results,
    tolerance=2.0,
    show_fit=True,
    save_path=None,
    show=True,
    plot=None,
    two_columns=True,
):
    """Create Panel 6 in a two-column grid or a wider single column."""
    daily_params = results.get('daily_params', {}) if results else {}
    global_params = results.get('global_params', {}) if results else {}
    fit_info = results.get('fit_info', {}) if results else {}
    panel_specs = _selected_panel_specs(plot)

    matched_by_title = {}
    model_by_title = {}
    all_jds = []
    for spec in panel_specs:
        matches = find_colour_grid_matches(df, spec['filters'], tolerance=tolerance)
        model_matches = _add_model_columns(matches, spec['filters'], daily_params, global_params, fit_info)
        matched_by_title[spec['title']] = matches
        model_by_title[spec['title']] = model_matches
        if not matches.empty:
            all_jds.extend(matches['JD'].astype(float).tolist())

    fit_colour_order = list(range(16, 20)) + list(range(8, 12)) + list(range(4, 8)) + list(range(0, 4)) + list(range(12, 16))
    data_colours_all, data_year_colours = _build_year_colours(all_jds, 'tab20b', 'black')
    fit_colours_all, fit_year_colours = _build_year_colours(all_jds, 'tab20c', '0.25', colour_order=fit_colour_order)

    n_panels = len(panel_specs)
    ncols = 2 if two_columns and n_panels > 1 else 1
    nrows = int(np.ceil(n_panels / ncols))
    if ncols == 2:
        figure_width, figure_height = 10, 10
    else:
        figure_width, figure_height = 7, 6 * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(figure_width, figure_height))
    axes_flat = np.asarray(axes).ravel()
    plot_axes = axes_flat[:n_panels]
    for extra_ax in axes_flat[n_panels:]:
        fig.delaxes(extra_ax)
    if ncols == 2:
        for ax in plot_axes:
            ax.set_box_aspect(1)

    colour_offset = 0
    for ax, spec in zip(plot_axes, panel_specs):
        matches = matched_by_title[spec['title']]
        n_matches = len(matches)
        data_colours = data_colours_all[colour_offset:colour_offset + n_matches]
        fit_colours = fit_colours_all[colour_offset:colour_offset + n_matches]
        colour_offset += n_matches
        model_matches = model_by_title[spec['title']]
        if len(model_matches) != n_matches and not model_matches.empty:
            fit_colours, _ = _build_year_colours(
                model_matches['JD'].astype(float).tolist(),
                'tab20c',
                '0.25',
                colour_order=fit_colour_order,
            )
        _plot_one(ax, spec, matches, model_matches, data_colours, fit_colours, show_fit=show_fit)

    handles, labels = plot_axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='upper center', ncol=min(len(handles), 4), frameon=False)
    fig.tight_layout(rect=[0.0, 0.02, 0.82, 0.92])
    _colourbar(fig, plot_axes, all_jds, data_year_colours, fit_year_colours, show_fit=show_fit)

    if save_path:
        fig.savefig(_generated_figure_path(save_path), dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    return fig, matched_by_title, model_by_title


__all__ = [
    'find_colour_grid_matches',
    'plot_final_panel_colour_grid',
]
