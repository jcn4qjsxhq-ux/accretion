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


def generate_ar_parameter_surface(ar_days, daily_data, fit_filters, default_params,
                                  global_param_map, popt, daily_params):
    """Generate parameter surfaces for AR visualization, including chi-squared values."""

    surfaces = {}
    R_star = default_params['R_star']

    for jd_day in ar_days:
        day_df = daily_data[jd_day]

        # Parameter ranges for the grid
        Av_values = np.linspace(0.1, 30.0, 40)
        Rin_values = np.linspace(1.0, 10.0, 40)  # In units of R_star

        Av_mesh, Rin_mesh = np.meshgrid(Av_values, Rin_values)

        Mdot_surface = np.zeros_like(Av_mesh)
        chi_squared_surface = np.zeros_like(Av_mesh)

        # Retrieve initial guess for Mdot optimization
        best_params = daily_params[jd_day]
        initial_mdot = 10 ** best_params['logMdot']

        # For each (Av, Rin) point, optimize Mdot and compute chi-squared
        for i in range(Av_mesh.shape[0]):
            for j in range(Av_mesh.shape[1]):
                Av_val = Av_mesh[i, j]
                Rin_val = Rin_mesh[i, j]  # In R_star units
                Rin_val_absolute = Rin_val * R_star  # Convert to absolute units

                # Optimize Mdot for this specific (Av, Rin) pair
                optimized_mdot = optimize_mdot_for_av_rin(
                    day_df, fit_filters, Av_val, Rin_val_absolute, 
                    default_params, initial_mdot
                )

                # Store the optimized Mdot
                Mdot_surface[i, j] = optimized_mdot

                # --- Compute chi-squared with the optimized Mdot ---
                obs_fluxes = []
                obs_errors = []
                model_fluxes = []

                for _, row in day_df.iterrows():
                    filt = row['Filter']
                    if filt not in fit_filters:
                        continue

                    flux_val = row['Flux']
                    flux_err = row['Fluxerr']

                    if flux_val <= 0 or np.isnan(flux_val):
                        continue

                    # Safety check for error calculation
                    if flux_err <= 0 or np.isnan(flux_err):
                        flux_err = max(0.1 * flux_val, 1e-12)

                    # Model prediction with optimized Av, Rin, Mdot
                    try:
                        model_log_flux = accretion_model(
                            [wavelength_to_frequency(get_filter_wavelengths()[filt])],
                            optimized_mdot,
                            default_params['M'],
                            default_params['R_star'],
                            default_params['R_out'],
                            Rin_val_absolute,  # Use absolute units
                            default_params['distance'],
                            Av_val
                        )[0]
                    except Exception:
                        model_log_flux = -50.0

                    if np.isnan(model_log_flux) or model_log_flux < -45:
                        continue

                    obs_fluxes.append(np.log(flux_val))
                    # Add systematic floor to errors
                    frac_err = flux_err / flux_val
                    frac_err = np.sqrt(frac_err**2 + 0.05**2)
                    obs_errors.append(frac_err)
                    model_fluxes.append(model_log_flux)

                if len(obs_fluxes) > 0:
                    obs_fluxes = np.array(obs_fluxes)
                    obs_errors = np.array(obs_errors)
                    model_fluxes = np.array(model_fluxes)

                    chi_sq = np.sum(((obs_fluxes - model_fluxes) / obs_errors) ** 2)
                    # Convert to reduced chi-squared
                    n_data_points = len(obs_fluxes)
                    dof = max(1, n_data_points - 1)  # -1 for optimized Mdot
                    reduced_chi_sq = chi_sq / dof
                else:
                    reduced_chi_sq = np.nan

                chi_squared_surface[i, j] = reduced_chi_sq

        # FIND BEST-FIT PARAMETERS FOR THIS DAY
        valid_mask = np.isfinite(chi_squared_surface)
        if np.any(valid_mask):
            best_idx = np.unravel_index(
                np.nanargmin(chi_squared_surface), 
                chi_squared_surface.shape
            )
            best_Av = Av_mesh[best_idx]
            best_Rin = Rin_mesh[best_idx]
            best_Mdot = Mdot_surface[best_idx]
            best_chi_sq = chi_squared_surface[best_idx]

            # Convert Mdot to solar masses per year for readability
            best_Mdot_solar = best_Mdot / M_sun * year

            # --- COMPUTE PARAMETER ERRORS ---
            # Find chi-squared threshold for 1-sigma confidence (delta_chi_sq = 1 for 1 parameter)
            # For 2D confidence regions with 3 parameters, use delta_chi_sq appropriate for your case
            # Common choices: delta_chi_sq = 1 (1-sigma, 1 param), 2.3 (68% 2D), 3.53 (90% 2D)
            delta_chi_sq = 2.3  # 68% confidence for 2 parameters
            chi_sq_threshold = best_chi_sq + delta_chi_sq

            # Create mask for points within confidence region
            within_confidence = chi_squared_surface <= chi_sq_threshold

            if np.sum(within_confidence) > 1:
                # Extract parameter values within confidence region
                Av_conf = Av_mesh[within_confidence]
                Rin_conf = Rin_mesh[within_confidence]
                Mdot_conf = Mdot_surface[within_confidence]

                # Compute errors as standard deviations of confident region
                Av_err = np.std(Av_conf)
                Rin_err = np.std(Rin_conf)
                Mdot_err = np.std(Mdot_conf)
                Mdot_solar_err = Mdot_err / M_sun * year
            else:
                # Fallback: use grid spacing as rough estimate
                Av_err = np.diff(Av_values).mean()
                Rin_err = np.diff(Rin_values).mean()
                # For Mdot, estimate from nearby points
                i_best, j_best = best_idx
                nearby_mdots = []
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        ii, jj = i_best + di, j_best + dj
                        if 0 <= ii < Mdot_surface.shape[0] and 0 <= jj < Mdot_surface.shape[1]:
                            if np.isfinite(Mdot_surface[ii, jj]):
                                nearby_mdots.append(Mdot_surface[ii, jj])
                Mdot_err = np.std(nearby_mdots) if len(nearby_mdots) > 1 else best_Mdot * 0.1
                Mdot_solar_err = Mdot_err / M_sun * year

        else:
            best_Av = best_Rin = best_Mdot = best_chi_sq = np.nan
            Av_err = Rin_err = Mdot_err = Mdot_solar_err = np.nan
            print(f"Warning: No valid chi-squared values found for day {jd_day}")

        # Store results for this day
        surfaces[jd_day] = {
            'Av_mesh': Av_mesh,
            'Rin_mesh': Rin_mesh,
            'Mdot_surface': Mdot_surface,
            'chi_squared_surface': chi_squared_surface,
            'best_params': {  # Store best parameters with errors
                'Av': best_Av,
                'Av_err': Av_err,
                'Rin': best_Rin,
                'Rin_err': Rin_err,
                'Mdot': best_Mdot,
                'Mdot_err': Mdot_err,
                'Mdot_solar': best_Mdot_solar if np.isfinite(best_Mdot) else np.nan,
                'Mdot_solar_err': Mdot_solar_err if np.isfinite(Mdot_err) else np.nan,
                'chi_squared_red': best_chi_sq
            }
        }

    return surfaces


def optimize_mdot_for_av_rin(day_df, fit_filters, av_test, rin_test, default_params, initial_mdot):
    """Quick optimization to find best Mdot for given Av and R_in
    
    Args:
        rin_test: R_in in absolute units (not multiples of R_star)
    """

    from scipy.optimize import minimize_scalar

    # Get wavelengths and frequencies
    wavelengths = get_filter_wavelengths()

    # Prepare day data
    frequencies = []
    log_fluxes = []
    errors = []

    for filter_name in fit_filters:
        if filter_name not in wavelengths:
            continue

        filter_data = day_df[day_df['Filter'] == filter_name]

        for _, row in filter_data.iterrows():
            flux_val = row['Flux']
            fluxerr_val = row['Fluxerr']

            if flux_val <= 0 or np.isnan(flux_val):
                continue

            # Safety check for errors
            if fluxerr_val <= 0 or np.isnan(fluxerr_val):
                fluxerr_val = max(0.1 * flux_val, 1e-12)

            log_flux = np.log(flux_val)
            frac_err = fluxerr_val / flux_val
            frac_err = np.sqrt(frac_err**2 + 0.05**2)  # 5% systematic floor

            if np.isnan(log_flux) or np.isnan(frac_err):
                continue

            frequencies.append(wavelength_to_frequency(wavelengths[filter_name]))
            log_fluxes.append(log_flux)
            errors.append(frac_err)

    if len(frequencies) == 0:
        return initial_mdot

    frequencies = np.array(frequencies)
    log_fluxes = np.array(log_fluxes)
    errors = np.array(errors)

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
    log_mdot_bounds = (np.log10(1e-7*M_sun/year), np.log10(1e-3*M_sun/year))

    try:
        result = minimize_scalar(chi_squared_mdot, bounds=log_mdot_bounds, method='bounded')
        return 10**result.x
    except Exception:
        return initial_mdot

__all__ = ['filter_days_for_red_excess', 'classify_filters_by_color', 'generate_ar_parameter_surface', 'optimize_mdot_for_av_rin']
