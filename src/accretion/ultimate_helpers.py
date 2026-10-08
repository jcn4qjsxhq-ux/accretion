"""Helper functions for special fitting modes."""

from __future__ import annotations

try:
    from .ultimate_common import *
except ImportError:
    from ultimate_common import *

try:
    from .ultimate_physics import *
except ImportError:
    from ultimate_physics import *


def filter_days_for_red_excess(daily_data, fit_filters):
    """Filter days that have at least 4 measurements with 2+ blue and 2+ red filters"""

    # Get H filter wavelength for reference
    wavelengths = get_filter_wavelengths()
    h_wavelength = wavelengths.get('H', 1.65e-6)  # Default H-band wavelength

    filtered_daily_data = {}

    for jd_day, day_df in daily_data.items():
        # Count measurements per filter
        filter_counts = {}
        for filter_name in fit_filters:
            count = len(day_df[day_df['Filter'] == filter_name])
            if count > 0:
                filter_counts[filter_name] = count

        # Classify filters as blue or red relative to H
        blue_filters = []
        red_filters = []

        for filter_name in filter_counts.keys():
            if filter_name in wavelengths:
                if wavelengths[filter_name] <= h_wavelength:
                    blue_filters.append(filter_name)
                else:
                    red_filters.append(filter_name)

        # Check if we have at least 2 blue and 2 red filters
        total_measurements = sum(filter_counts.values())
        if (len(blue_filters) >= 2 and len(red_filters) >= 2 and 
            total_measurements >= 4):
            filtered_daily_data[jd_day] = day_df
        
        # print('b:', blue_filters,', r:', red_filters)

    return filtered_daily_data


def classify_filters_by_color(daily_data, fit_filters, wavelengths):
    """Classify filters as blue or red relative to H-band for each day"""

    h_wavelength = wavelengths.get('H', 1.65e-6)
    red_excess_data = {}

    for jd_day, day_df in daily_data.items():
        blue_filters = []
        red_filters = []

        # Get filters that actually have data for this day
        available_filters = day_df['Filter'].unique()

        for filter_name in available_filters:
            if filter_name in wavelengths:
                if wavelengths[filter_name] <= h_wavelength:
                    blue_filters.append(filter_name)
                else:
                    red_filters.append(filter_name)
            else:
                print(f"Warning: Filter {filter_name} not found in wavelengths dictionary")
                # Assign unknown filters to blue as fallback
                blue_filters.append(filter_name)

        red_excess_data[jd_day] = {
            'blue_filters': blue_filters,
            'red_filters': red_filters
        }

    jd = list(red_excess_data.keys())[0]
    print(red_excess_data[jd])
    
    return red_excess_data


def _power_from_reduced_chi_squared(reduced_chi_sq):
    """Convert reduced chi-squared values to a likelihood-like plotting power."""
    return np.exp(-0.5 * reduced_chi_sq)


AR_AV_RANGE = (0.0, 30.0)
AR_RIN_RANGE_RSTAR = (0.0, 100.0)
AR_MDOT_RANGE_SOLAR_PER_YEAR = (0.07e-4, 9.0e-4)
AR_MCMC_CHAINS = 4
AR_MCMC_STEPS = 2000
AR_MCMC_BURN_IN = 500
AR_MCMC_THIN = 2
AR_MCMC_RANDOM_SEED = 314159


def _prepare_ar_day_arrays(day_df, fit_filters, data_mode='photometry', max_spectral_points=250,
                           missing_photometric_fractional_error=.01, systematic_fractional_error=.05):
    """Precompute arrays used by one AR posterior evaluation."""
    data_mode = str(data_mode).strip().lower()
    wavelengths = get_filter_wavelengths()
    frequencies = []
    log_fluxes = []
    errors = []

    if data_mode == 'spectral':
        row_iter = ((None, row) for _, row in day_df.iterrows())
    else:
        row_iter = []
        for filter_name in fit_filters:
            if filter_name not in wavelengths:
                continue
            filter_data = day_df[day_df['Filter'] == filter_name]
            row_iter.extend((filter_name, row) for _, row in filter_data.iterrows())

    for filter_name, row in row_iter:
        if data_mode == 'spectral':
            frequency_val = row.get('Frequency', np.nan)
        else:
            wavelength_value = row.get('Lambda', np.nan)
            try:
                wavelength_value = float(wavelength_value)
            except (TypeError, ValueError):
                wavelength_value = np.nan
            if np.isfinite(wavelength_value) and wavelength_value > 0:
                wavelength_micron = wavelength_value if wavelength_value > 1e-3 else wavelength_value * 1e6
                frequency_val = wavelength_to_frequency(wavelength_micron)
            else:
                frequency_val = wavelength_to_frequency(wavelengths[filter_name])

        flux_val = row['Flux']
        fluxerr_val = row['Fluxerr']
        if flux_val <= 0 or np.isnan(flux_val) or not np.isfinite(frequency_val):
            continue

        if not np.isfinite(fluxerr_val) or fluxerr_val <= 0:
            fluxerr_val = (max(0.1 * flux_val, 1e-12) if data_mode == 'spectral'
                           else missing_photometric_fractional_error * flux_val)

        log_flux = np.log(flux_val)
        frac_err = fluxerr_val / flux_val
        frac_err = np.hypot(frac_err, systematic_fractional_error)
        if np.isnan(log_flux) or np.isnan(frac_err):
            continue

        frequencies.append(frequency_val)
        log_fluxes.append(log_flux)
        errors.append(frac_err)

    frequencies = np.asarray(frequencies, dtype=float)
    log_fluxes = np.asarray(log_fluxes, dtype=float)
    errors = np.asarray(errors, dtype=float)

    if data_mode == 'spectral' and len(frequencies) > max_spectral_points:
        sample_indices = np.unique(
            np.linspace(0, len(frequencies) - 1, max_spectral_points).astype(int)
        )
        frequencies = frequencies[sample_indices]
        log_fluxes = log_fluxes[sample_indices]
        errors = errors[sample_indices]
        print(
            f"AR MCMC: using {len(sample_indices)}/{len(day_df)} spectral samples "
            "for faster posterior sampling"
        )

    return frequencies, log_fluxes, errors


def _ar_chi_squared(theta, prepared_data, default_params):
    """Return total chi-squared for ``(Av, Rin/R_star, log10(Mdot))``."""
    av, rin_rstar, log_mdot = np.asarray(theta, dtype=float)
    frequencies, log_fluxes, errors = prepared_data
    # The thin-disk model is undefined inside the stellar radius.  Keep the
    # requested 0--100 plotting/prior range, but give Rin < R_star zero
    # likelihood without calling the model (which otherwise prints warnings).
    if rin_rstar < 1.0:
        return np.inf
    try:
        predicted = accretion_model(
            frequencies,
            10.0 ** log_mdot,
            default_params['M'],
            default_params['R_star'],
            default_params['R_out'],
            rin_rstar * default_params['R_star'],
            default_params['distance'],
            av,
        )
    except Exception:
        return np.inf

    valid = np.isfinite(predicted) & (predicted > -45)
    if np.sum(valid) != len(frequencies):
        return np.inf
    residuals = (log_fluxes - predicted) / errors
    chi_squared = np.sum(residuals ** 2)
    return float(chi_squared) if np.isfinite(chi_squared) else np.inf


def _run_ar_mcmc(prepared_data, default_params, initial_theta, rng):
    """Run independent adaptive random-walk Metropolis chains for one day."""
    from scipy.optimize import differential_evolution, minimize

    log_mdot_range = (
        np.log10(AR_MDOT_RANGE_SOLAR_PER_YEAR[0] * M_sun / year),
        np.log10(AR_MDOT_RANGE_SOLAR_PER_YEAR[1] * M_sun / year),
    )
    lower = np.array([AR_AV_RANGE[0], AR_RIN_RANGE_RSTAR[0], log_mdot_range[0]])
    upper = np.array([AR_AV_RANGE[1], AR_RIN_RANGE_RSTAR[1], log_mdot_range[1]])
    span = upper - lower
    epsilon = np.maximum(span * 1e-8, 1e-12)
    initial_theta = np.clip(np.asarray(initial_theta, dtype=float), lower + epsilon, upper - epsilon)

    def bounded_chi_squared(theta):
        if np.any(theta < lower) or np.any(theta > upper):
            return 1e100
        return _ar_chi_squared(theta, prepared_data, default_params)

    initial_chi_squared = bounded_chi_squared(initial_theta)
    optimizer_lower = lower.copy()
    optimizer_lower[1] = max(optimizer_lower[1], 1.0)
    optimizer_bounds = list(zip(optimizer_lower, upper))
    global_result = differential_evolution(
        bounded_chi_squared,
        bounds=optimizer_bounds,
        maxiter=35,
        popsize=8,
        tol=1e-4,
        atol=1e-3,
        polish=False,
        seed=rng,
        updating='immediate',
    )
    optimizer_start = (
        global_result.x
        if np.isfinite(global_result.fun) and global_result.fun < initial_chi_squared
        else initial_theta
    )
    map_result = minimize(
        bounded_chi_squared,
        optimizer_start,
        method='Powell',
        bounds=optimizer_bounds,
        options={'maxiter': 500, 'xtol': 1e-4, 'ftol': 1e-5},
    )
    map_theta = np.clip(map_result.x, lower + epsilon, upper - epsilon)
    map_chi_squared = bounded_chi_squared(map_theta)
    if np.isfinite(global_result.fun) and global_result.fun < map_chi_squared:
        map_theta = np.asarray(global_result.x, dtype=float)
        map_chi_squared = float(global_result.fun)
    if initial_chi_squared < map_chi_squared:
        map_theta = initial_theta.copy()
        map_chi_squared = initial_chi_squared
    if not np.isfinite(map_chi_squared) or map_chi_squared >= 1e99:
        map_theta = initial_theta
        map_chi_squared = bounded_chi_squared(map_theta)
    if not np.isfinite(map_chi_squared) or map_chi_squared >= 1e99:
        raise RuntimeError('Could not find a finite starting point for AR MCMC')

    proposal_scale = np.array([0.02, 0.02, 0.025]) * span
    chains = np.empty((AR_MCMC_CHAINS, AR_MCMC_STEPS, 3), dtype=float)
    chain_chi_squared = np.empty((AR_MCMC_CHAINS, AR_MCMC_STEPS), dtype=float)
    acceptance_rates = []

    # Seed chains at separated, high-likelihood members of the global-search
    # population.  This makes disagreement between different parts of a long
    # degeneracy visible to R-hat and helps the combined sample cover it.
    seed_points = [map_theta]
    population = np.asarray(getattr(global_result, 'population', []), dtype=float)
    population_energies = np.asarray(
        getattr(global_result, 'population_energies', []), dtype=float
    )
    candidate_mask = (
        np.all(np.isfinite(population), axis=1)
        & np.isfinite(population_energies)
        & (population_energies <= map_chi_squared + 25.0)
    ) if population.ndim == 2 and len(population) == len(population_energies) else np.array([], dtype=bool)
    candidates = population[candidate_mask] if len(candidate_mask) else np.empty((0, 3))
    while len(seed_points) < AR_MCMC_CHAINS and len(candidates):
        normalized_distance = np.stack([
            np.linalg.norm((candidates - point) / span, axis=1)
            for point in seed_points
        ])
        next_index = int(np.argmax(np.min(normalized_distance, axis=0)))
        seed_points.append(candidates[next_index])
        candidates = np.delete(candidates, next_index, axis=0)

    for chain_idx in range(AR_MCMC_CHAINS):
        if chain_idx < len(seed_points):
            current = np.asarray(seed_points[chain_idx], dtype=float).copy()
        else:
            current = np.clip(
                map_theta + rng.normal(0.0, proposal_scale * 2.0),
                lower + epsilon,
                upper - epsilon,
            )
        current_chi_squared = bounded_chi_squared(current)
        if not np.isfinite(current_chi_squared) or current_chi_squared >= 1e99:
            current = map_theta.copy()
            current_chi_squared = map_chi_squared

        proposal_covariance = np.diag(proposal_scale ** 2)
        accepted = 0
        window_accepted = 0
        for step_idx in range(AR_MCMC_STEPS):
            proposal = current + rng.multivariate_normal(np.zeros(3), proposal_covariance)
            if np.all(proposal >= lower) and np.all(proposal <= upper):
                proposal_chi_squared = bounded_chi_squared(proposal)
                log_acceptance = -0.5 * (proposal_chi_squared - current_chi_squared)
                if np.log(rng.random()) < min(0.0, log_acceptance):
                    current = proposal
                    current_chi_squared = proposal_chi_squared
                    accepted += 1
                    window_accepted += 1

            chains[chain_idx, step_idx] = current
            chain_chi_squared[chain_idx, step_idx] = current_chi_squared

            # Tune only during burn-in, targeting a useful 3-D acceptance rate.
            if step_idx < AR_MCMC_BURN_IN and (step_idx + 1) % 100 == 0:
                window_rate = window_accepted / 100.0
                history = chains[chain_idx, :step_idx + 1]
                empirical_covariance = np.cov(history, rowvar=False)
                if np.all(np.isfinite(empirical_covariance)):
                    proposal_covariance = (
                        empirical_covariance * (2.38 ** 2 / 3.0)
                        + np.diag((span * 1e-5) ** 2)
                    )
                if window_rate < 0.18:
                    proposal_covariance *= 0.7 ** 2
                elif window_rate > 0.45:
                    proposal_covariance *= 1.3 ** 2
                window_accepted = 0

        acceptance_rates.append(accepted / AR_MCMC_STEPS)

    kept_chains = chains[:, AR_MCMC_BURN_IN::AR_MCMC_THIN, :]
    kept_chi_squared_chains = chain_chi_squared[:, AR_MCMC_BURN_IN::AR_MCMC_THIN]
    n_kept = kept_chains.shape[1]
    chain_means = np.mean(kept_chains, axis=1)
    between_chain = n_kept * np.var(chain_means, axis=0, ddof=1)
    within_chain = np.mean(np.var(kept_chains, axis=1, ddof=1), axis=0)
    variance_estimate = ((n_kept - 1) / n_kept) * within_chain + between_chain / n_kept
    r_hat = np.sqrt(np.divide(
        variance_estimate,
        within_chain,
        out=np.full(3, np.inf),
        where=within_chain > 0,
    ))

    kept = kept_chains.reshape(-1, 3)
    kept_chi_squared = kept_chi_squared_chains.reshape(-1)
    return (
        kept,
        kept_chi_squared,
        np.asarray(acceptance_rates),
        r_hat,
        map_theta,
        map_chi_squared,
    )


def generate_ar_mcmc_samples(ar_days, daily_data, fit_filters, default_params,
                             global_param_map, popt, daily_params,
                             data_mode='photometry', missing_photometric_fractional_error=.01,
                             systematic_fractional_error=.05):
    """Sample the 3-D AR posterior for each selected photometric or spectral day."""
    del global_param_map, popt  # Retained in the signature for compatibility with the old generator.
    results = {}
    data_mode = str(data_mode).strip().lower()

    for day_number, jd_day in enumerate(ar_days):
        sample_start = time.time()
        day_df = daily_data[jd_day]
        prepared_data = _prepare_ar_day_arrays(
            day_df, fit_filters, data_mode=data_mode,
            missing_photometric_fractional_error=missing_photometric_fractional_error,
            systematic_fractional_error=systematic_fractional_error)
        n_points = len(prepared_data[0])
        if n_points == 0:
            print(f"Warning: No valid AR MCMC data found for day {jd_day}")
            continue

        day_params = daily_params[jd_day]
        initial_theta = np.array([
            np.clip(day_params.get('Av', 15.0), *AR_AV_RANGE),
            np.clip(2.0, *AR_RIN_RANGE_RSTAR),
            day_params['logMdot'],
        ])
        rng = np.random.default_rng(AR_MCMC_RANDOM_SEED + day_number)
        print(
            f"AR MCMC JD {jd_day}: {AR_MCMC_CHAINS} chains x {AR_MCMC_STEPS} steps "
            f"using {n_points} data points"
        )

        try:
            (
                samples,
                chi_squared,
                acceptance,
                r_hat,
                map_theta,
                map_chi_squared,
            ) = _run_ar_mcmc(prepared_data, default_params, initial_theta, rng)
        except RuntimeError as exc:
            print(f"Warning: AR MCMC failed for day {jd_day}: {exc}")
            continue

        best_index = int(np.nanargmin(chi_squared))
        best_theta = samples[best_index]
        best_chi_squared = float(chi_squared[best_index])
        if map_chi_squared < best_chi_squared:
            best_theta = map_theta
            best_chi_squared = float(map_chi_squared)

        relative_power = np.exp(-0.5 * np.clip(chi_squared - best_chi_squared, 0.0, 1500.0))
        quantiles = np.percentile(samples, [16.0, 50.0, 84.0], axis=0)
        lower_error = quantiles[1] - quantiles[0]
        upper_error = quantiles[2] - quantiles[1]
        mdot_samples_solar = 10.0 ** samples[:, 2] / M_sun * year
        mdot_quantiles = np.percentile(mdot_samples_solar, [16.0, 50.0, 84.0])
        best_mdot = 10.0 ** best_theta[2]
        best_mdot_solar = best_mdot / M_sun * year
        dof = max(1, n_points - 3)

        results[jd_day] = {
            'samples': samples,
            'chi_squared': chi_squared,
            'reduced_chi_squared': chi_squared / dof,
            'relative_power': relative_power,
            'acceptance_rates': acceptance,
            'r_hat': r_hat,
            'converged': bool(np.all(np.isfinite(r_hat)) and np.all(r_hat < 1.1)),
            'parameter_order': ['Av', 'Rin_Rstar', 'log10_Mdot_kg_s'],
            'prior': {
                'Av': AR_AV_RANGE,
                'Rin_Rstar': AR_RIN_RANGE_RSTAR,
                'Rin_model_support': 'Rin/Rstar >= 1',
                'Mdot_solar_per_year': AR_MDOT_RANGE_SOLAR_PER_YEAR,
                'Mdot_prior': 'uniform in log10(Mdot)',
            },
            'posterior': {
                'Av': quantiles[:, 0],
                'Rin': quantiles[:, 1],
                'logMdot': quantiles[:, 2],
                'Mdot_solar': mdot_quantiles,
            },
            'best_params': {
                'Av': best_theta[0],
                'Av_err': 0.5 * (lower_error[0] + upper_error[0]),
                'Rin': best_theta[1],
                'Rin_err': 0.5 * (lower_error[1] + upper_error[1]),
                'logMdot': best_theta[2],
                'Mdot': best_mdot,
                'Mdot_solar': best_mdot_solar,
                'Mdot_solar_err': 0.5 * (mdot_quantiles[2] - mdot_quantiles[0]),
                'chi_squared': best_chi_squared,
                'chi_squared_red': best_chi_squared / dof,
            },
            'n_data_points': n_points,
            'dof': dof,
        }
        elapsed = time.time() - sample_start
        print(
            f"AR MCMC JD {jd_day}: retained {len(samples)} samples in {elapsed:.1f}s; "
            f"acceptance {np.mean(acceptance):.1%}; max R-hat {np.nanmax(r_hat):.3f}"
        )
        if not results[jd_day]['converged']:
            print(
                f"Warning: AR MCMC chains for JD {jd_day} have not converged "
                f"(R-hat={np.array2string(r_hat, precision=3)}). "
                "Increase AR_MCMC_STEPS before interpreting the posterior."
            )

    return results


def generate_ar_parameter_surface(*args, **kwargs):
    """Backward-compatible alias for the MCMC-based AR generator."""
    return generate_ar_mcmc_samples(*args, **kwargs)


def optimize_mdot_for_av_rin(day_df, fit_filters, av_test, rin_test, default_params, initial_mdot,
                             data_mode='photometry', prepared_data=None, return_chi=False,
                             mdot_bounds_solar_per_year=None, missing_photometric_fractional_error=.01,
                             systematic_fractional_error=.05):
    """Quick optimization to find best Mdot for given Av and R_in.

    Args:
        rin_test: R_in in absolute units (not multiples of R_star)
    """

    from scipy.optimize import minimize_scalar

    data_mode = str(data_mode).strip().lower()

    if prepared_data is None:
        frequencies, log_fluxes, errors = _prepare_ar_day_arrays(
            day_df, fit_filters, data_mode=data_mode,
            missing_photometric_fractional_error=missing_photometric_fractional_error,
            systematic_fractional_error=systematic_fractional_error
        )
    else:
        frequencies, log_fluxes, errors = prepared_data

    if len(frequencies) == 0:
        return (initial_mdot, np.nan) if return_chi else initial_mdot

    def chi_squared_mdot(log_mdot):
        """Chi-squared for given log(Mdot)"""
        mdot = 10**log_mdot

        try:
            predicted_log_fluxes = accretion_model(
                frequencies, mdot, 
                default_params['M'], default_params['R_star'], 
                default_params['R_out'], rin_test,  # rin_test is already in absolute units
                default_params['distance'], av_test
            )

            valid_mask = np.isfinite(predicted_log_fluxes) & (predicted_log_fluxes > -45)
            if np.sum(valid_mask) == 0:
                return 1e10

            chi_sq = np.sum(((log_fluxes[valid_mask] - predicted_log_fluxes[valid_mask]) / 
                           errors[valid_mask])**2)
            return chi_sq

        except Exception:
            return 1e10

    # Optimize log(Mdot)
    if mdot_bounds_solar_per_year is None:
        mdot_bounds_solar_per_year = AR_MDOT_RANGE_SOLAR_PER_YEAR
    log_mdot_bounds = (
        np.log10(mdot_bounds_solar_per_year[0] * M_sun / year),
        np.log10(mdot_bounds_solar_per_year[1] * M_sun / year),
    )

    try:
        result = minimize_scalar(
            chi_squared_mdot,
            bounds=log_mdot_bounds,
            method='bounded',
            options={'xatol': 0.02}
        )
        optimized_mdot = 10**result.x
        if not return_chi:
            return optimized_mdot
        chi_sq = result.fun
        dof = max(1, len(frequencies) - 1)
        return optimized_mdot, chi_sq / dof
    except Exception:
        return (initial_mdot, np.nan) if return_chi else initial_mdot

__all__ = [
    'filter_days_for_red_excess',
    'classify_filters_by_color',
    'generate_ar_mcmc_samples',
    'generate_ar_parameter_surface',
    'optimize_mdot_for_av_rin',
    '_power_from_reduced_chi_squared',
    '_prepare_ar_day_arrays',
    'AR_AV_RANGE',
    'AR_RIN_RANGE_RSTAR',
    'AR_MDOT_RANGE_SOLAR_PER_YEAR',
    'AR_MCMC_CHAINS',
    'AR_MCMC_STEPS',
    'AR_MCMC_BURN_IN',
    'AR_MCMC_THIN',
    'AR_MCMC_RANDOM_SEED',
]
