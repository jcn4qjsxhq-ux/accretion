"""Plotting and result-visualisation helpers."""

from __future__ import annotations

try:
    from .ultimate_common import *
except ImportError:
    from ultimate_common import *

try:
    from .ultimate_file_organisers import _generated_datas_dir, average_daily_measurements, get_daily_data
except ImportError:
    from ultimate_file_organisers import _generated_datas_dir, average_daily_measurements, get_daily_data

try:
    from .ultimate_physics import *
except ImportError:
    from ultimate_physics import *

try:
    from .ultimate_fitting import *
except ImportError:
    from ultimate_fitting import *

try:
    from .ultimate_helpers import AR_AV_RANGE, AR_MDOT_RANGE_SOLAR_PER_YEAR, AR_RIN_RANGE_RSTAR
except ImportError:
    from ultimate_helpers import AR_AV_RANGE, AR_MDOT_RANGE_SOLAR_PER_YEAR, AR_RIN_RANGE_RSTAR

# CHANGE THRESHOLDS FOR YOUR OWN NEEDS
SPECTRAL_REDE_THRESHOLD_MICRON = globals().get('SPECTRAL_REDE_THRESHOLD_MICRON', 2.0)
SPECTRAL_REDE_BB_PLOT_FLOOR_JY = globals().get('SPECTRAL_REDE_BB_PLOT_FLOOR_JY', 1e-5)


def _stellar_mass_solar(results):
    """Return the fitted/adopted stellar mass in solar-mass units.

    New results record the fixed mass in fit_info.  Older bundles used the
    same 0.5 Msun model default but omitted it, so 0.5 is the compatible
    fallback.  Fitted ``M`` values are stored in SI units.
    """
    results = results if isinstance(results, dict) else {}
    fit_info = results.get('fit_info', {}) or {}
    value = fit_info.get('stellar_mass_solar', np.nan)
    if np.isfinite(value) and value > 0:
        return float(value)

    global_params = results.get('global_params', {}) or {}
    if 'M_star_solar' in global_params:
        value = global_params['M_star_solar']
    elif 'M' in global_params:
        value = float(global_params['M']) / M_sun
    elif 'M_star' in global_params:
        value = global_params['M_star']
        if np.isfinite(value) and value > 10:
            value = float(value) / M_sun
    else:
        value = 0.5
    return float(value) if np.isfinite(value) and value > 0 else 0.5


def _project_root():
    """Return repository root from an installed or local src checkout."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


def _generated_figure_path(filename):
    """Resolve generated plot paths independently of notebook cwd.

    Relative paths are flattened so notebook workflow labels do not create
    public subfolder structures. Plot outputs are always PDFs.
    """
    filename = str(filename)
    root, _ext = os.path.splitext(filename)
    pdf_filename = f'{root}.pdf'
    if os.path.isabs(filename):
        output_path = pdf_filename
    else:
        output_name = os.path.basename(pdf_filename.rstrip('/\\'))
        output_path = os.path.join(_project_root(), 'results', 'generated', 'pictures', output_name)
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    return output_path


def _generated_output_dir():
    output_dir = os.path.join(_project_root(), 'results', 'generated', 'pictures')
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def julian_to_calendar(jd_array):
    calendar_dates = []

    for jd in jd_array:
        jd = jd + 0.5
        z = int(jd)

        if z >= 2299161:
            alpha = int((z - 1867216.25) / 36524.25)
            a = z + 1 + alpha - int(alpha / 4)
        elif z <= 10000:
            a = z + 2450000
        elif 50000 < z < 100000:
            a = z + 2400000
        else:
            a = z

        b = a + 1524
        c_year = int((b - 122.1) / 365.25)
        d = int(365.25 * c_year)
        e = int((b - d) / 30.6001)

        day = b - d - int(30.6001 * e)
        month = e - 1 if e < 14 else e - 13
        year_value = c_year - 4716 if month > 2 else c_year - 4715
        calendar_dates.append(f"{year_value:04d}-{month:02d}-{day:02d}")

    return calendar_dates


def add_gregorian_top_axis(ax, n_ticks=6, fontsize=12, use_grid_ticks=True, dynamic=True):
    """Add a Gregorian date top axis that tracks zoom/pan."""
    ax_top = ax.twiny()

    def update():
        x0, x1 = ax.get_xlim()
        if x1 <= x0:
            return

        if use_grid_ticks:
            ticks = ax.get_xticks()
            ticks = [t for t in ticks if x0 <= t <= x1]
            if len(ticks) < 2:
                ticks = np.linspace(x0, x1, n_ticks)
        else:
            ticks = np.linspace(x0, x1, n_ticks)

        labels = julian_to_calendar(ticks)
        ax_top.set_xlim(x0, x1)
        ax_top.set_xticks(ticks)
        ax_top.set_xticklabels(labels, rotation=-45, ha='right')
        ax_top.set_xlabel('Gregorian Date', fontsize=fontsize)

    update()

    if dynamic:
        ax.callbacks.connect('xlim_changed', lambda _ax: update())

    return ax_top


def get_model_predictions(results, df, accretion_model):
    """Calculate model predictions for each observation."""
    if not results['success']:
        print("Cannot generate predictions - fitting failed")
        return None

    daily_data = results['daily_data']
    daily_params = results['daily_params']
    global_params = results['global_params']

    default_params = {
        'M': 0.5 * M_sun,
        'R_star': 3.0 * R_sun,
        'R_out': 2 * constants.au,
        'distance': 700 * constants.parsec,
        'R_in': 3.0 * R_sun,
    }
    default_params.update(global_params)

    wavelengths = get_filter_wavelengths()
    predictions = {}

    for jd_day, day_data in daily_data.items():
        if jd_day not in daily_params:
            continue

        mdot = daily_params[jd_day]['Mdot'] * M_sun / year
        av = daily_params[jd_day]['Av']
        r_in = global_params.get('R_in', default_params['R_star'])
        day_predictions = {}

        for filter_name in day_data['Filter'].unique():
            if filter_name not in wavelengths:
                continue

            filter_data = day_data[day_data['Filter'] == filter_name]
            if len(filter_data) == 0:
                continue

            freq = wavelength_to_frequency(wavelengths[filter_name])
            try:
                log_flux_pred = accretion_model(
                    np.array([freq]), mdot,
                    default_params['M'], default_params['R_star'],
                    default_params['R_out'], r_in,
                    default_params['distance'], av,
                )[0]
                zp = filter_data['ZP'].iloc[0]
                day_predictions[filter_name] = {
                    'log_flux': log_flux_pred,
                    'magnitude': log_flux_to_magnitude(log_flux_pred, zp),
                    'observed_mag': filter_data['Mag'].iloc[0],
                    'observed_err': filter_data['Magerr'].iloc[0],
                    'zeropoint': zp,
                }
            except Exception as exc:
                print(f"Error calculating prediction for {filter_name} on JD {jd_day}: {exc}")

        predictions[jd_day] = day_predictions

    return predictions


def plot_results_regularized(results, df, filename="Mdot-Av_plot", end=True, peter_data=None):
    '''Enhanced plotting function for regularized results'''
    print("Trying to plot...")

    daily_params = results['daily_params']
    jd_days = list(daily_params.keys())
    m_star = _stellar_mass_solar(results)
    mdot_values = [m_star * (10**(daily_params[jd]['logMdot']) / M_sun * year) for jd in jd_days]
    logmdot_values = [daily_params[jd]['logMdot'] for jd in jd_days]
    mdot_errors = [m_star * daily_params[jd]['Mdot_err'] for jd in jd_days]
    av_values = [daily_params[jd]['Av'] for jd in jd_days]
    av_errors = [daily_params[jd]['Av_err'] for jd in jd_days]

    masses = results['global_params']
    lims = (2.459e6, 2.46097e6)
    mdoty = (1.5e-4, max(mdot_values)*1.5)
    cuts = 6

    if end:
        lims = (2.459e6, 2.46097e6)
        mdoty = (1.5e-4, max(mdot_values)*1.5)
        fig = plt.figure(figsize=(12, 18))
    else:
        lims = (min(jd_days)-500, max(jd_days)+500)
        mdoty = (min(mdot_values)*0.7, max(mdot_values)*1.5)
        fig = plt.figure(figsize=(10, 24))

    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312, sharex=ax1)
    ax3 = fig.add_subplot(313)

    ax1.errorbar(jd_days, mdot_values, yerr=mdot_errors,
                 fmt='o-', capsize=5, capthick=2, label=f'M_star*Mdot (λ={results["fit_info"]["lambda_reg"]})')
    ax1.set_xlabel('Julian Date')
    ax1.set_ylabel('M_star*Mdot (M☉^2/year)')
    ax1.set_yscale('log')
    ax1.set_title(f'Regularized M_star*Mdot vs Time (λ={results["fit_info"]["lambda_reg"]})')
    ax1.tick_params(top=False, labeltop=False, bottom=False, labelbottom=False)
    add_gregorian_top_axis(ax1, n_ticks=6, fontsize=14, use_grid_ticks=True, dynamic=True)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.errorbar(jd_days, av_values, yerr=av_errors,
                 fmt='o-', capsize=5, capthick=2, label=f'Av (λ={results["fit_info"]["lambda_reg"]})', color='red')
    ax2.set_xlabel('Julian Date')
    ax2.set_ylabel('Av (mag)')
    ax2.set_title(f'Regularized Extinction vs Time (λ={results["fit_info"]["lambda_reg"]})')
    ax2.tick_params(top=False, labeltop=False, bottom=True, labelbottom=True)
    add_gregorian_top_axis(ax2, n_ticks=6, fontsize=14, use_grid_ticks=True, dynamic=True)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    ax3.errorbar(av_values, mdot_values, xerr=av_errors, yerr=mdot_errors,
                 fmt='o', capsize=5, capthick=2, color='purple',
                 label=f'M_star*Mdot vs Av (λ={results["fit_info"]["lambda_reg"]})')
    ax3.set_xlabel('Av (mag)')
    ax3.set_ylabel('M_star*Mdot (M☉^2/year)')
    ax3.set_title(f'Regularized M_star*Mdot vs Av (λ={results["fit_info"]["lambda_reg"]})')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # Overlay Peter's model results if provided
    if peter_data is not None:
        _peter_colors = ['tab:orange', 'tab:green', 'tab:brown', 'tab:pink']
        for _i, (_lbl, _df_p) in enumerate(peter_data.items()):
            _c = _peter_colors[_i % len(_peter_colors)]
            ax1.plot(_df_p['JD'], _df_p['Mdot'], 's--', ms=4, lw=1, alpha=0.8,
                     color=_c, label=f'Peter ({_lbl})')
            ax2.plot(_df_p['JD'], _df_p['Av'], 's--', ms=4, lw=1, alpha=0.8,
                     color=_c, label=f'Peter ({_lbl})')
            ax3.plot(_df_p['Av'], _df_p['Mdot'], 's', ms=4, alpha=0.8,
                     color=_c, label=f'Peter ({_lbl})')
        ax1.legend()
        ax2.legend()
        ax3.legend()

    plt.tight_layout()
    output_pdf = _generated_figure_path(filename)
    plt.savefig(output_pdf)
    backend = str(plt.get_backend()).lower()
    is_widget_backend = ('widget' in backend or 'ipympl' in backend or 'nbagg' in backend)
    try:
        fig.canvas.draw()
    except Exception:
        pass
    if is_widget_backend:
        try:
            from IPython.display import display
            plt.ioff()
            display(fig.canvas)
        except Exception:
            plt.show()
    else:
        plt.show()
        plt.close(fig)

    print("\n=== Regularized Fitting Results Summary ===")
    print(f"λ (regularization strength): {results['fit_info']['lambda_reg']}")
    print(f"Regularized parameters: {results['fit_info']['param_names']}")
    print(f"Chi-squared: {results['chi_squared']:.2f}")
    print(f"Regularization term: {results['regularization_term']:.2f}")
    print(f"Total objective: {results['total_objective']:.2f}")
    print(f"Mean M_star*Mdot: {np.mean(mdot_values):.2e} ± {np.std(mdot_values):.2e} M☉^2/year")
    print(f"Mean Av: {np.mean(av_values):.2f} ± {np.std(av_values):.2f} mag")

    mdot_smoothness = np.sum(np.diff(mdot_values)**2)
    av_smoothness = np.sum(np.diff(av_values)**2)
    print(f"M_star*Mdot smoothness (lower = smoother): {mdot_smoothness:.2e}")
    print(f"Av smoothness (lower = smoother): {av_smoothness:.2f}")

    if results['global_params']:
        print("\nGlobal Parameters:")
        for param, value in results['global_params'].items():
            error = results['param_errors'].get(param, 0)
            print(f"  {param}: {value:.3f} ± {error:.3f}")

    return fig


def plot_parameter_evolution(results, df, save_path=None, show=True):
    """
    Plot parameter evolution over time with dual x-axis (Julian/Gregorian dates)
    """
    if not results['success']:
        print("Cannot plot - fitting failed")
        return

    daily_params = results['daily_params']
    global_params = results['global_params']
    jd_days = sorted(daily_params.keys())
    m_star = _stellar_mass_solar(results)
    mdot_values = [m_star * daily_params[jd]['Mdot'] for jd in jd_days]
    mdot_errors = [m_star * daily_params[jd]['Mdot_err'] for jd in jd_days]
    av_values = [daily_params[jd]['Av'] for jd in jd_days]
    av_errors = [daily_params[jd]['Av_err'] for jd in jd_days]

    daily_extra_params = []
    if any('R_bb' in params for params in daily_params.values()):
        daily_extra_params.append(('R_bb', 'Blackbody Radius (AU)', 'tab:purple', constants.au))
    if any('T_bb' in params for params in daily_params.values()):
        daily_extra_params.append(('T_bb', 'Blackbody Temperature (K)', 'tab:orange', 1.0))

    n_params = 2 + len(daily_extra_params) + len(global_params)
    fig, axes = plt.subplots(n_params, 1, figsize=(13, 5*n_params), sharex=True)
    if n_params == 1:
        axes = [axes]

    ax1 = axes[0]
    ax1.errorbar(jd_days, mdot_values, yerr=mdot_errors,
                 fmt='o-', capsize=5, capthick=2, color='blue',
                 markersize=8, linewidth=2)
    ax1.set_ylabel('M_star*Mdot (M☉^2/year)', fontsize=14)
    ax1.set_title('M_star*Mdot vs Time', fontsize=16)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    add_gregorian_top_axis(ax1, n_ticks=6, fontsize=12, use_grid_ticks=True, dynamic=True)
    ax1.tick_params(bottom=False, labelbottom=False)

    ax2 = axes[1]
    ax2.errorbar(jd_days, av_values, yerr=av_errors,
                 fmt='o-', capsize=5, capthick=2, color='red',
                 markersize=8, linewidth=2)
    ax2.set_ylabel('Av (mag)', fontsize=14)
    ax2.set_title('Extinction vs Time', fontsize=16)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel('Julian Date', fontsize=14)
    add_gregorian_top_axis(ax2, n_ticks=6, fontsize=12, use_grid_ticks=True, dynamic=True)
    ax2.tick_params(top=False, labeltop=False)

    param_idx = 2
    for param_name, y_label, color, scale in daily_extra_params:
        if param_idx < len(axes):
            ax = axes[param_idx]
            values = [
                _safe_float(daily_params[jd].get(param_name), np.nan) / scale
                for jd in jd_days
            ]
            errors = [
                _safe_float(daily_params[jd].get(f'{param_name}_err'), np.nan) / scale
                for jd in jd_days
            ]
            ax.errorbar(
                jd_days, values, yerr=errors,
                fmt='o-', capsize=5, capthick=2, color=color,
                markersize=8, linewidth=2
            )
            ax.set_ylabel(y_label, fontsize=14)
            ax.set_title(f'{y_label} vs Time', fontsize=16)
            ax.grid(True, alpha=0.3)
            ax.tick_params(bottom=False, labelbottom=False)
            add_gregorian_top_axis(ax, n_ticks=6, fontsize=12, use_grid_ticks=True, dynamic=True)
            param_idx += 1

    for param_name, param_value in global_params.items():
        if param_idx < len(axes):
            ax = axes[param_idx]
            ax.axhline(y=param_value, color='green', linewidth=3,
                       label=f'{param_name} = {param_value:.3f}')
            ax.set_ylabel(param_name, fontsize=14)
            ax.set_title(f'{param_name} vs Time (Global Parameter)', fontsize=16)
            ax.grid(True, alpha=0.3)
            if param_idx == len(axes) - 1:
                ax.set_xlabel('Julian Date', fontsize=14)
            else:
                ax.set_xlabel('')
                ax.tick_params(bottom=False, labelbottom=False)
            ax.legend()
            add_gregorian_top_axis(ax, n_ticks=6, fontsize=12, use_grid_ticks=True, dynamic=True)
            param_idx += 1

    plt.tight_layout()
    if save_path:
        fig.savefig(_generated_figure_path(save_path), dpi=300, bbox_inches='tight')
    if show:
        plt.show()

    print("\n=== Parameter Evolution Summary ===")
    print(f"M_star*Mdot range: {min(mdot_values):.2e} - {max(mdot_values):.2e} M☉^2/year")
    print(f"M_star*Mdot mean ± std: {np.mean(mdot_values):.2e} ± {np.std(mdot_values):.2e} M☉^2/year")
    print(f"Av range: {min(av_values):.2f} - {max(av_values):.2f} mag")
    print(f"Av mean ± std: {np.mean(av_values):.2f} ± {np.std(av_values):.2f} mag")

    if global_params:
        print("\nGlobal Parameters:")
        for param, value in global_params.items():
            print(f"  {param}: {value:.3f}")

    return fig


def create_color_plots_with_model(df, predictions, save_path=None):
    """
    Create color-magnitude and color-color plots with model predictions overlaid
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('J vs J-K', 'J-H vs H-K', 'I vs I-J', 'K vs H-K'),
        horizontal_spacing=0.15,
        vertical_spacing=0.15
    )

    dI = 0.607
    dJ = 0.28760574
    dH = 0.17830579
    dK = 0.11701314
    n_points_max = len(df)
    plasma_colors = sample_colorscale('plasma', np.linspace(0, 1, n_points_max))


def plot_unique_filter_with_model(df, predictions, save_path=None, end=True):
    """
    Create the unique filter plot with model predictions overlaid
    """
    fig, ax1 = plt.subplots(figsize=(13, 7))

    cmap = mpl.colormaps['tab20b']
    c = cmap(np.linspace(0, 1, len(df['Filter'].unique())))
    c = cmap(np.linspace(0, 1, 5))

    filter_idx = 0
    for filter_name in df['Filter'].unique():
        if filter_name in ('J', 'H', 'K'):
            filter_data = df[df['Filter'] == filter_name]
            JD = filter_data['JD'].values
            mag = filter_data['Mag'].values
            err = filter_data['Magerr'].values
            ax1.plot(JD, mag, '-o', label=f'{filter_name} (obs)',
                    color=c[filter_idx], markersize=8, alpha=0.7, zorder=5)
            ax1.errorbar(JD, mag, yerr=err, fmt='.', capsize=0,
                        color='black', alpha=0.3)
            filter_idx += 1

    filter_idx = 0
    for filter_name in df['Filter'].unique():
        if filter_name in ('J', 'H', 'K'):
            filter_data = df[df['Filter'] == filter_name]
            JD = filter_data['JD'].values
            mag = filter_data['Mag'].values
            err = filter_data['Magerr'].values
            model_jds = []
            model_mags = []

            for jd_day, day_preds in predictions.items():
                if filter_name in day_preds:
                    model_jds.append(jd_day)
                    model_mags.append(day_preds[filter_name]['magnitude'])

            if model_jds:
                ax1.plot(model_jds, model_mags, '--s',
                        label=f'{filter_name} (model)',
                        color=c[filter_idx], markersize=6, alpha=0.9,
                        linewidth=2, zorder=0)
            filter_idx += 1

    spread = max(df['JD']) - min(df['JD'])
    if end:
        lims = (max(df['JD'])-spread/2, max(df['JD'])+spread/(5*10))
    else:
        lims = (min(df['JD'])-spread/10, max(df['JD'])+spread/10)

    cuts = 9
    ax1.set_xlim(lims)
    ax1.set_xlabel('Julian Date', fontsize=14)
    ax1.set_ylabel('Magnitude', fontsize=14)
    ax1.invert_yaxis()
    ax1.set_xticks(np.linspace(lims[0], lims[1], cuts))
    ax1.tick_params(top=False, labeltop=False, bottom=True, labelbottom=True)
    add_gregorian_top_axis(ax1, n_ticks=6, fontsize=14, use_grid_ticks=True, dynamic=True)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=12, ncol=2)

    plt.title('Photometric Data with Accretion Model Predictions', fontsize=16, pad=20)
    plt.tight_layout()
    if save_path:
        plt.savefig(_generated_figure_path(save_path), dpi=300, bbox_inches='tight')
    plt.show()


def create_residual_plots(predictions, df, save_path=None):
    """
    Create residual plots (observed - model) for each filter
    """
    residuals_by_filter = {}

    for jd_day, day_preds in predictions.items():
        for filter_name, pred_data in day_preds.items():
            if filter_name not in residuals_by_filter:
                residuals_by_filter[filter_name] = {
                    'jd': [], 'residual': [], 'error': []
                }

            residual = pred_data['observed_mag'] - pred_data['magnitude']
            residuals_by_filter[filter_name]['jd'].append(jd_day)
            residuals_by_filter[filter_name]['residual'].append(residual)
            residuals_by_filter[filter_name]['error'].append(pred_data['observed_err'])

    n_filters = len(residuals_by_filter)
    n_cols = 3
    n_rows = (n_filters + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1) if n_cols > 1 else [axes]

    filter_names = list(residuals_by_filter.keys())

    for i, filter_name in enumerate(filter_names):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]

        data = residuals_by_filter[filter_name]
        ax.errorbar(data['jd'], data['residual'], yerr=data['error'],
                   fmt='o-', capsize=5, capthick=2, markersize=6)
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        ax.set_xlabel('Julian Date')
        ax.set_ylabel('Residual (obs - model)')
        ax.set_title(f'{filter_name} Filter Residuals')
        ax.grid(True, alpha=0.3)

        rms = np.sqrt(np.mean(np.array(data['residual'])**2))
        ax.text(0.05, 0.95, f'RMS: {rms:.3f}', transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    for i in range(n_filters, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]
        ax.set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(_generated_figure_path(save_path), dpi=300, bbox_inches='tight')
    plt.show()


def comprehensive_results_visualization(results, df, accretion_model, save_dir="results/generated/pictures"):
    """
    Create comprehensive visualization of fitting results

    Parameters:
    - results: output from ultimate_fitting function
    - df: original dataframe
    - accretion_model: the accretion model function
    - save_dir: directory to save plots
    """
    print("=== Creating Comprehensive Visualization ===")

    if not results['success']:
        print("Cannot create visualizations - fitting failed")
        return

    print("1. Creating parameter evolution plots...")
    plot_parameter_evolution(results, df, "parameter_evolution")

    print("2. Generating model predictions...")
    predictions = get_model_predictions(results, df, accretion_model)
    if predictions is None:
        print("Could not generate model predictions")
        return

    print("3. Creating unique filter plot with model overlay...")
    plot_unique_filter_with_model(df, predictions, "unique_filter_with_model")

    print("4. Creating residual plots...")
    create_residual_plots(predictions, df, "residual_plots")

    print("5. Generating summary statistics...")
    total_residuals = []
    total_errors = []

    for jd_day, day_preds in predictions.items():
        for filter_name, pred_data in day_preds.items():
            residual = pred_data['observed_mag'] - pred_data['magnitude']
            total_residuals.append(residual)
            total_errors.append(pred_data['observed_err'])

    total_residuals = np.array(total_residuals)
    total_errors = np.array(total_errors)
    rms_residual = np.sqrt(np.mean(total_residuals**2))
    reduced_chi_squared = np.sum((total_residuals / total_errors)**2) / (len(total_residuals) - len(results['fit_info']['param_names']))

    print(f"\n=== Overall Fit Quality ===")
    print(f"RMS Residual: {rms_residual:.3f} mag")
    print(f"Reduced Chi-squared: {reduced_chi_squared:.3f}")
    print(f"Number of fitted parameters: {len(results['fit_info']['param_names'])}")
    print(f"Number of data points: {len(total_residuals)}")

    print("\n=== Visualization Complete ===")
    print(f"Plots saved to: {save_dir}")

    return {
        'predictions': predictions,
        'rms_residual': rms_residual,
        'reduced_chi_squared': reduced_chi_squared,
        'n_parameters': len(results['fit_info']['param_names']),
        'n_data_points': len(total_residuals)
    }


def plot_ar_surface(results, folder_name):
    """Plot the AR 3-D MCMC posterior, with legacy grid-result support."""

    if 'AR_mcmc_data' in results:
        mcmc_data = results['AR_mcmc_data']
        output_prefix = os.path.splitext(os.path.basename(str(folder_name).rstrip('/\\')))[0] or 'ar_mcmc'
        output_dir = _generated_output_dir()
        print(f"Saving AR MCMC plots to {output_dir}/")

        for jd_day, data in mcmc_data.items():
            samples = np.asarray(data['samples'], dtype=float)
            power = np.asarray(data['relative_power'], dtype=float)
            reduced_chi_squared = np.asarray(data['reduced_chi_squared'], dtype=float)
            valid = (
                np.all(np.isfinite(samples), axis=1)
                & np.isfinite(power)
                & np.isfinite(reduced_chi_squared)
            )
            if not np.any(valid):
                print(f"Warning: No valid AR MCMC samples for day {jd_day}")
                continue

            samples = samples[valid]
            power = power[valid]
            reduced_chi_squared = reduced_chi_squared[valid]
            mdot_solar_scaled = (10.0 ** samples[:, 2] / M_sun * year) * 1e4
            bp = data['best_params']
            posterior = data.get('posterior', {})
            acceptance = np.asarray(data.get('acceptance_rates', []), dtype=float)
            r_hat = np.asarray(data.get('r_hat', []), dtype=float)

            customdata = np.column_stack((power, reduced_chi_squared))
            fig = go.Figure()
            fig.add_trace(go.Scatter3d(
                x=samples[:, 0],
                y=samples[:, 1],
                z=mdot_solar_scaled,
                mode='markers',
                marker=dict(
                    size=3,
                    color=power,
                    colorscale='Inferno_r',
                    cmin=0.0,
                    cmax=1.0,
                    opacity=0.45,
                    colorbar=dict(
                        title=dict(text=r"$\exp(-\Delta\chi^2/2)$", font=dict(size=18), side='right'),
                        len=0.7,
                        thickness=30,
                        tickfont=dict(size=16),
                        x=0.98,
                        xanchor='left',
                        yanchor='middle',
                        y=0.5,
                    ),
                ),
                customdata=customdata,
                hovertemplate=(
                    'Av: %{x:.3f}<br>Rin: %{y:.3f} R*<br>'
                    'Mdot: %{z:.4g} x10^-4 Msun/yr<br>'
                    'Relative power: %{customdata[0]:.3g}<br>'
                    'Reduced chi2: %{customdata[1]:.3g}<extra></extra>'
                ),
                showlegend=False,
            ))

            best_mdot_scaled = bp['Mdot_solar'] * 1e4
            fig.add_trace(go.Scatter3d(
                x=[bp['Av']],
                y=[bp['Rin']],
                z=[best_mdot_scaled],
                mode='markers',
                marker=dict(size=9, color='black', line=dict(color='white', width=2)),
                hovertemplate=(
                    f"<b>Best sample</b><br>Av: {bp['Av']:.3f}<br>"
                    f"Rin: {bp['Rin']:.3f} R*<br>"
                    f"Mdot: {best_mdot_scaled:.4g} x10^-4 Msun/yr<br>"
                    f"Reduced chi2: {bp['chi_squared_red']:.3g}<extra></extra>"
                ),
                showlegend=False,
            ))

            median_text = ''
            if posterior:
                median_text = (
                    f"<br>Posterior medians: Av={posterior['Av'][1]:.3g}, "
                    f"Rin={posterior['Rin'][1]:.3g} R*, "
                    f"Mdot={posterior['Mdot_solar'][1] * 1e4:.3g} x10^-4 Msun/yr"
                )
            acceptance_text = (
                f"Mean acceptance: {np.nanmean(acceptance):.1%}"
                if len(acceptance) else 'Acceptance unavailable'
            )
            if len(r_hat):
                acceptance_text += f"; max R-hat: {np.nanmax(r_hat):.3f}"
                if not data.get('converged', False):
                    acceptance_text += ' (not converged)'

            fig.update_layout(
                title=dict(
                    text=f"AR MCMC posterior — JD {jd_day:.2f}<br><sup>{acceptance_text}{median_text}</sup>",
                    x=0.5,
                ),
                scene=dict(
                    xaxis=dict(
                        title=dict(text="Aᵥ (mag)", font=dict(size=20, family="Latin Modern Math, serif")),
                        range=[AR_AV_RANGE[1], AR_AV_RANGE[0]],
                        backgroundcolor="rgb(240, 240, 240)", gridcolor="white",
                        showbackground=True, tickfont=dict(size=16),
                    ),
                    yaxis=dict(
                        title=dict(text="Rᵢₙ (R⋆)", font=dict(size=20, family="Latin Modern Math, serif")),
                        range=[AR_RIN_RANGE_RSTAR[1], AR_RIN_RANGE_RSTAR[0]],
                        backgroundcolor="rgb(240, 240, 240)", gridcolor="white",
                        showbackground=True, tickfont=dict(size=16),
                    ),
                    zaxis=dict(
                        title=dict(text="Ṁ (10⁻⁴ M⊙/yr)", font=dict(size=20, family="Latin Modern Math, serif")),
                        type='log',
                        range=[np.log10(AR_MDOT_RANGE_SOLAR_PER_YEAR[0] * 1e4),
                               np.log10(AR_MDOT_RANGE_SOLAR_PER_YEAR[1] * 1e4)],
                        backgroundcolor="rgb(240, 240, 240)", gridcolor="white",
                        showbackground=True, tickfont=dict(size=16),
                    ),
                    aspectmode='cube',
                    camera=dict(eye=dict(x=1.5, y=1.5, z=1.2), center=dict(x=0, y=0, z=0)),
                ),
                height=900,
                width=1000,
                showlegend=False,
                template='plotly_white',
                margin=dict(l=50, r=120, t=90, b=50),
            )

            output_stem = f'{output_prefix}_JD_{jd_day:.2f}_mcmc'
            output_preview = os.path.join(output_dir, f'{output_stem}_preview.html')
            preview_start = time.time()
            fig.write_html(output_preview, include_plotlyjs='cdn', auto_open=False)
            print(
                f"  Preview ready: {os.path.basename(output_preview)} "
                f"({time.time() - preview_start:.1f}s)"
            )

            output_pdf = os.path.join(output_dir, f'{output_stem}.pdf')
            pdf_start = time.time()
            fig.write_image(output_pdf, width=1400, height=1260, scale=2)
            print(f"  Saved: {os.path.basename(output_pdf)} ({time.time() - pdf_start:.1f}s)")

        print(f"\nAll AR MCMC plots saved to {output_dir}/")
        return

    if 'AR_surface_data' not in results:
        print("No AR MCMC or legacy surface data found!")
        return

    surface_data = results['AR_surface_data']
    output_prefix = os.path.splitext(os.path.basename(str(folder_name).rstrip('/\\')))[0] or 'ar_surface'
    output_dir = _generated_output_dir()
    print(f"Saving plots to {output_dir}/")

    def _make_ar_surface_figure(data, x_range, y_range, z_range):
        X, Y, Z = data['Av_mesh'], data['Rin_mesh'], data['Mdot_surface']
        C = data.get('power_surface')
        if C is None:
            C = np.exp(-0.5 * data['chi_squared_surface'])
        Z_solar = (Z / M_sun * year) * 1e4

        valid_power = C[np.isfinite(C)]
        if len(valid_power) == 0:
            return None

        power_min, power_max = np.nanmin(valid_power), np.nanmax(valid_power)
        if not np.isfinite(power_min) or not np.isfinite(power_max):
            return None
        if power_min == power_max:
            power_max = power_min + 1e-12
        z_axis_min = 10 ** z_range[0]
        z_axis_max = 10 ** z_range[1]
        default_z_ticks = np.array([0.07, 0.1, 0.2, 0.5, 1, 2, 5, 9], dtype=float)
        z_ticks = default_z_ticks[(default_z_ticks >= z_axis_min) & (default_z_ticks <= z_axis_max)]
        if len(z_ticks) < 2:
            z_ticks = np.array([z_axis_min, np.sqrt(z_axis_min * z_axis_max), z_axis_max])
        z_tick_text = [f"{value:.3g}" for value in z_ticks]

        fig = go.Figure()
        fig.add_trace(go.Surface(
            x=X, y=Y, z=Z_solar,
            surfacecolor=np.where(np.isfinite(C), C, np.nan),
            colorscale='Inferno_r',
            cmin=power_min,
            cmax=power_max,
            colorbar=dict(
                title=dict(
                    text=r"$\exp(-\chi^2_{\mathrm{red}}/2)$",
                    font=dict(size=18),
                    side='right'
                ),
                len=0.7,
                thickness=30,
                tickfont=dict(size=16),
                x=0.98,
                xanchor='left',
                yanchor='middle',
                y=0.5,
            ),
            opacity=0.75,
            hovertemplate='Av: %{x:.2f}<br>Rin: %{y:.2f}<br>Mdot: %{z:.2e}<br>Power: %{surfacecolor:.3e}<extra></extra>'
        ))

        if 'best_params' in data:
            bp = data['best_params']
            if np.isfinite(bp['chi_squared_red']):
                x_min, y_min = np.min(X), np.min(Y)
                z_min = z_axis_min
                bp_z = bp['Mdot_solar'] * 1e4
                print(f'The best fit accretion rate is: {bp_z}:.3g')

                fig.add_trace(go.Scatter3d(
                    x=[bp['Av']], y=[bp['Rin']], z=[bp_z],
                    mode='markers',
                    marker=dict(size=8, color='black', line=dict(color='white', width=2)),
                    showlegend=False,
                    hovertemplate=(f"<b>Best Fit</b><br>Av: {bp['Av']:.2f}<br>"
                                  f"Rin: {bp['Rin']:.2f}<br>chi2: {bp['chi_squared_red']:.1f}<extra></extra>")
                ))

                line_style = dict(color='black', width=4, dash='dash')
                fig.add_trace(go.Scatter3d(
                    x=[bp['Av'], bp['Av']], y=[bp['Rin'], bp['Rin']], z=[z_min, bp_z],
                    mode='lines', line=line_style, showlegend=False, hoverinfo='skip'))
                fig.add_trace(go.Scatter3d(
                    x=[x_min, bp['Av']], y=[bp['Rin'], bp['Rin']], z=[z_min, z_min],
                    mode='lines', line=dict(color='black', width=3, dash='dash'), showlegend=False, hoverinfo='skip'))
                fig.add_trace(go.Scatter3d(
                    x=[bp['Av'], bp['Av']], y=[y_min, bp['Rin']], z=[z_min, z_min],
                    mode='lines', line=dict(color='black', width=3, dash='dash'), showlegend=False, hoverinfo='skip'))

        fig.update_layout(
            scene=dict(
                xaxis=dict(
                    title=dict(text="Aᵥ (mag)", font=dict(size=20, family="Latin Modern Math, serif")),
                    backgroundcolor="rgb(240, 240, 240)", gridcolor="white",
                    showbackground=True, autorange=False, tickfont=dict(size=16), range=x_range
                ),
                yaxis=dict(
                    title=dict(text="Rᵢₙ (R⋆)", font=dict(size=20, family="Latin Modern Math, serif")),
                    backgroundcolor="rgb(240, 240, 240)", gridcolor="white",
                    showbackground=True, autorange=False, tickfont=dict(size=16), range=y_range
                ),
                zaxis=dict(
                    title=dict(text="Ṁ (10⁻⁴ M⊙/yr)", font=dict(size=20, family="Latin Modern Math, serif")),
                    type='log',
                    backgroundcolor="rgb(240, 240, 240)",
                    gridcolor="white",
                    showbackground=True,
                    tickfont=dict(size=16),
                    range=z_range,
                    tickmode='array',
                    tickvals=z_ticks,
                    ticktext=z_tick_text
                ),
                aspectmode='cube',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.2), center=dict(x=0, y=0, z=0))
            ),
            height=900, width=1000,
            showlegend=False,
            template='plotly_white',
            margin=dict(l=50, r=120, t=50, b=50)
        )
        return fig

    def _write_surface_figure(fig, output_stem):
        output_preview = os.path.join(output_dir, f'{output_stem}_preview.html')
        preview_start = time.time()
        fig.write_html(output_preview, include_plotlyjs='cdn', auto_open=False)
        print(
            f"  Preview ready: {os.path.basename(output_preview)} "
            f"({time.time() - preview_start:.1f}s)"
        )

        output_pdf = os.path.join(output_dir, f'{output_stem}.pdf')
        pdf_start = time.time()
        fig.write_image(output_pdf, width=1400, height=1260, scale=2)
        print(f"  Saved: {os.path.basename(output_pdf)} ({time.time() - pdf_start:.1f}s)")

    for jd_day, data in surface_data.items():
        fig = _make_ar_surface_figure(
            data,
            x_range=[30.1, -0.1],
            y_range=[100.1, -0.1],
            z_range=[np.log10(0.07), np.log10(9.0)],
        )
        if fig is None:
            print(f"Warning: No valid power values for day {jd_day}")
            continue
        _write_surface_figure(fig, f'{output_prefix}_JD_{jd_day:.2f}')

        zoom_surface = data.get('zoom_surface')
        if zoom_surface:
            zoom_data = dict(zoom_surface)
            zoom_data['best_params'] = data.get('best_params', {})
            zoom_X = zoom_data['Av_mesh']
            zoom_Y = zoom_data['Rin_mesh']
            mdot_bounds = zoom_data.get('mdot_axis_range_solar_per_year')
            if mdot_bounds is None:
                zoom_Z_solar = zoom_data['Mdot_surface'] / M_sun * year
                mdot_bounds = (np.nanmin(zoom_Z_solar), np.nanmax(zoom_Z_solar))
            zoom_fig = _make_ar_surface_figure(
                zoom_data,
                x_range=[np.nanmax(zoom_X), np.nanmin(zoom_X)],
                y_range=[np.nanmax(zoom_Y), np.nanmin(zoom_Y)],
                z_range=[np.log10(mdot_bounds[0] * 1e4), np.log10(mdot_bounds[1] * 1e4)],
            )
            if zoom_fig is not None:
                _write_surface_figure(zoom_fig, f'{output_prefix}_JD_{jd_day:.2f}_zoom')

    print(f"\nAll plots saved to {output_dir}/")


def plot_red_excess_results(results, filename):
    if not results['fit_info']['red_excess_mode']:
        print("Not in red excess mode!")
        return

    fit_info = results.get('fit_info', {})
    fit_param_name = _normalize_red_excess_fit_param(fit_info.get('red_excess_fit_param', 'T_bb'))

    times = []
    quantity_values = []
    quantity_errors = []
    mdots = []
    mdot_errors = []
    avs = []
    av_errors = []
    m_star = _stellar_mass_solar(results)

    for jd_day, params in results['daily_params'].items():
        times.append(jd_day)
        mdots.append(m_star * params['Mdot'])
        mdot_errors.append(m_star * _safe_float(params.get('Mdot_err'), np.nan))
        avs.append(params['Av'])
        av_errors.append(_safe_float(params.get('Av_err'), np.nan))

        if fit_param_name == 'R_bb':
            quantity_values.append(_safe_float(params.get('R_bb'), np.nan) / constants.au)
            quantity_errors.append(_safe_float(params.get('R_bb_err'), np.nan) / constants.au)
            y_label = 'Dust Radius (AU)'
            title_label = 'Red Excess Radius Evolution'
            corr_label = 'Radius'
            fixed_value_text = f"Fixed T_bb = {fit_info.get('fixed_T_bb', RED_EXCESS_DEFAULT_T_BB):.0f} K"
        else:
            quantity_values.append(_safe_float(params.get('T_bb'), np.nan))
            quantity_errors.append(_safe_float(params.get('T_bb_err'), np.nan))
            y_label = 'Dust Temperature (K)'
            title_label = 'Red Excess Temperature Evolution'
            corr_label = 'Temperature'
            fixed_value_text = f"Fixed R_bb = {fit_info.get('fixed_R_bb', RED_EXCESS_DEFAULT_R_BB) / constants.au:.2f} AU"

    times = np.asarray(times, dtype=float)
    quantity_values = np.asarray(quantity_values, dtype=float)
    quantity_errors = np.asarray(quantity_errors, dtype=float)
    mdots = np.asarray(mdots, dtype=float)
    mdot_errors = np.asarray(mdot_errors, dtype=float)
    avs = np.asarray(avs, dtype=float)
    av_errors = np.asarray(av_errors, dtype=float)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 6))

    ax1.errorbar(times, quantity_values, yerr=quantity_errors, fmt='o-', capsize=3)
    ax1.set_xlabel('Time (JD)')
    ax1.set_ylabel(y_label)
    ax1.set_title(f"{title_label}\n{fixed_value_text}")
    ax1.grid(True)
    add_gregorian_top_axis(ax1, n_ticks=6, fontsize=12, use_grid_ticks=True, dynamic=True)

    sc2 = ax2.scatter(mdots, quantity_values, c=times, cmap='viridis')
    ax2.errorbar(mdots, quantity_values, yerr=quantity_errors, xerr=mdot_errors,
                 fmt='--', ecolor='gray', alpha=0.6, capsize=3)
    ax2.set_xlabel('M_star*Mdot (M_sun^2/yr)')
    ax2.set_ylabel(y_label)
    ax2.set_title(f'{corr_label} vs M_star*Mdot')
    ax2.set_xscale('log')
    ax2.grid(True)
    cbar2 = plt.colorbar(sc2, ax=ax2)
    cbar2.set_label('Time (JD)')

    sc3 = ax3.scatter(avs, quantity_values, c=times, cmap='viridis')
    ax3.errorbar(avs, quantity_values, xerr=av_errors, yerr=quantity_errors,
                 fmt='--', ecolor='gray', alpha=0.6, capsize=3)
    ax3.set_xlabel('A_v (mag)')
    ax3.set_ylabel(y_label)
    ax3.set_title(f'{corr_label} vs A_v')
    ax3.grid(True)
    cbar3 = plt.colorbar(sc3, ax=ax3)
    cbar3.set_label('Time (JD)')

    plt.tight_layout()
    output_path = _generated_figure_path(filename)
    plt.savefig(output_path)
    plt.show()
    print(f"Red-excess plot saved to {output_path}")
    
    """Main visualization function for different modes"""


def _get_accretion_defaults(results):
    global_params = results.get('global_params', {})
    r_star = global_params.get('R_star', 3.0 * R_sun)
    return {
        'M': global_params.get('M', 0.5 * M_sun),
        'R_star': r_star,
        'R_out': global_params.get('R_out', 2 * constants.au),
        'distance': global_params.get('distance', 700 * constants.parsec),
        'R_in': r_star * 2 if results.get('fit_info', {}).get('AR_mode') else r_star,
    }


def _roll_av(array, i):
    """Centered running average helper.

    The window length is forced to an odd integer. Edge values are averaged
    over the available data within the centered window so the output is
    smooth across the full array.
    """
    arr = np.asarray(array, dtype=float)
    n_arr = len(arr)
    if n_arr == 0:
        return arr

    i = int(max(1, i))
    if i % 2 == 0:
        i += 1

    if i == 1 or n_arr == 1:
        return arr.copy()

    mid = i // 2
    av = np.empty(n_arr, dtype=float)

    for n in range(n_arr):
        start = max(0, n - mid)
        end = min(n_arr, n + mid + 1)
        av[n] = np.mean(arr[start:end])

    return av


def _spectral_model_components(results, jd_day, frequencies, bb_plot_tail=False):
    daily_params = results['daily_params'][jd_day]
    defaults = _get_accretion_defaults(results)
    acc_log_flux = accretion_model(
        frequencies,
        10 ** daily_params['logMdot'],
        defaults['M'],
        defaults['R_star'],
        defaults['R_out'],
        defaults['R_in'],
        defaults['distance'],
        daily_params['Av']
    )
    acc_flux = np.exp(acc_log_flux)

    if results.get('fit_info', {}).get('red_excess_mode'):
        bb_log_flux = planck_model_custom(
            frequencies,
            daily_params.get('T_bb', RED_EXCESS_DEFAULT_T_BB),
            defaults['distance'],
            daily_params['Av'],
            R_bb=daily_params.get('R_bb', RED_EXCESS_DEFAULT_R_BB)
        )
        bb_flux = np.exp(np.where(np.isfinite(bb_log_flux), bb_log_flux, -50.0))
        total_flux = acc_flux + bb_flux
    else:
        bb_flux = None
        total_flux = acc_flux

    return acc_flux, bb_flux, total_flux


def plot_spectral_fit_day(results, jd_day, ax=None, ax_residual=None, show=True, residual_window=51, show_range=None):
    if 'daily_data' not in results or jd_day not in results['daily_data']:
        raise KeyError(f'No spectral daily data available for JD {jd_day}')

    day_df = results['daily_data'][jd_day].copy()
    lambda_m = _coerce_wavelength_meters(day_df)
    if 'Frequency' in day_df.columns:
        frequencies = pd.to_numeric(day_df['Frequency'], errors='coerce').to_numpy(dtype=float)
    else:
        frequencies = constants.c / lambda_m

    flux = pd.to_numeric(day_df['Flux'], errors='coerce').to_numpy(dtype=float)
    fluxerr = pd.to_numeric(day_df.get('Fluxerr', np.nan), errors='coerce').to_numpy(dtype=float)

    valid_mask = np.isfinite(lambda_m) & np.isfinite(frequencies) & np.isfinite(flux) & (flux > 0)
    lambda_m = lambda_m[valid_mask]
    frequencies = frequencies[valid_mask]
    flux = flux[valid_mask]
    fluxerr = fluxerr[valid_mask]

    order = np.argsort(lambda_m)
    lambda_m = lambda_m[order]
    frequencies = frequencies[order]
    flux = flux[order]
    fluxerr = fluxerr[order]

    daily_params = results['daily_params'][jd_day]
    acc_flux, bb_flux, model_flux = _spectral_model_components(results, jd_day, frequencies)
    residual_flux = model_flux - flux
    residual_running = _roll_av(residual_flux, residual_window)
    lambda_um = lambda_m * 1e6
    date_label = day_df['Date'].iloc[0] if 'Date' in day_df.columns and len(day_df) else str(jd_day)

    created_fig = False
    if ax is None:
        fig, (ax, ax_residual) = plt.subplots(
            2, 1, figsize=(10, 7), sharex=True,
            gridspec_kw={'height_ratios': [3.0, 1.4]}
        )
        created_fig = True
    else:
        fig = ax.figure
        if ax_residual is None:
            # Backward-compatible: residual panel is optional when caller provides only one axis
            ax_residual = None

    if show_range is not None:
        try:
            xmin, xmax = float(show_range[0]), float(show_range[1])
            if xmin > xmax:
                xmin, xmax = xmax, xmin
            range_mask = (lambda_um >= xmin) & (lambda_um <= xmax)
            lambda_um = lambda_um[range_mask]
            frequencies = frequencies[range_mask]
            flux = flux[range_mask]
            fluxerr = fluxerr[range_mask]
            residual_flux = residual_flux[range_mask]
            residual_running = residual_running[range_mask]

            plot_lambda_um = np.linspace(xmin, xmax, 400)
            plot_lambda_m = plot_lambda_um * 1e-6
            plot_frequencies = constants.c / plot_lambda_m
            plot_acc_flux, plot_bb_flux, plot_model_flux = _spectral_model_components(
                results, jd_day, plot_frequencies, bb_plot_tail=True
            )
            ax.set_xlim(xmin, xmax)
            if ax_residual is not None:
                ax_residual.set_xlim(xmin, xmax)
        except Exception:
            plot_lambda_um = lambda_um
            plot_acc_flux = acc_flux
            plot_bb_flux = bb_flux
            plot_model_flux = model_flux
    else:
        plot_lambda_um = lambda_um
        plot_acc_flux = acc_flux
        _, plot_bb_flux, _ = _spectral_model_components(
            results, jd_day, frequencies, bb_plot_tail=True
        ) if results.get('fit_info', {}).get('red_excess_mode') else (None, bb_flux, None)
        plot_model_flux = model_flux

    # Masked telluric bands create real wavelength gaps.  Insert NaNs at those
    # boundaries so matplotlib does not draw misleading diagonal bridges.
    positive_steps = np.diff(lambda_um)
    positive_steps = positive_steps[np.isfinite(positive_steps) & (positive_steps > 0)]
    typical_step = np.median(positive_steps) if len(positive_steps) else np.nan
    gap_threshold = max(0.01, 20.0 * typical_step) if np.isfinite(typical_step) else 0.01
    gap_indices = np.flatnonzero(np.diff(lambda_um) > gap_threshold) + 1

    def with_gap_breaks(values):
        return np.insert(np.asarray(values, dtype=float), gap_indices, np.nan)

    lambda_um_broken = with_gap_breaks(lambda_um)
    flux_broken = with_gap_breaks(flux)
    ax.plot(lambda_um_broken, flux_broken, color='tab:blue', linewidth=1.2, alpha=0.85, label='Observed spectrum')
    if np.any(np.isfinite(fluxerr)):
        lower = np.clip(flux - np.nan_to_num(fluxerr, nan=0.0), 1e-30, None)
        upper = np.clip(flux + np.nan_to_num(fluxerr, nan=0.0), 1e-30, None)
        ax.fill_between(
            lambda_um_broken,
            with_gap_breaks(lower),
            with_gap_breaks(upper),
            color='tab:blue',
            alpha=0.18,
            linewidth=0,
        )
    if results.get('fit_info', {}).get('red_excess_mode'):
        ax.plot(plot_lambda_um, plot_acc_flux, color='tab:orange', linewidth=1.7, linestyle='--', label='Accretion model')
        bb_plot = np.asarray(plot_bb_flux, dtype=float)
        bb_plot = np.where(bb_plot >= SPECTRAL_REDE_BB_PLOT_FLOOR_JY, bb_plot, np.nan)
        ax.plot(plot_lambda_um, bb_plot, color='tab:green', linewidth=1.7, linestyle=':', label='Planck BB')
        ax.plot(plot_lambda_um, plot_model_flux, color='tab:red', linewidth=2.1, label='Total model')
        title_extra = f', Tbb={daily_params.get("T_bb", np.nan):.0f} K, Rbb={daily_params.get("R_bb", np.nan) / constants.au:.2f} AU'
    else:
        ax.plot(plot_lambda_um, plot_model_flux, color='tab:orange', linewidth=2.0, label='Accretion model')
        title_extra = ''
    ax.set_xlabel('Wavelength (μm)')
    ax.set_ylabel('Flux density (Jy)')
    ax.set_yscale('log')

    # Focus limits around the model envelope (±15%)
    finite_model = np.asarray(plot_model_flux, dtype=float)
    finite_model = finite_model[np.isfinite(finite_model) & (finite_model > 0)]
    if finite_model.size > 0:
        model_min = np.min(finite_model)
        model_max = np.max(finite_model)
        if model_max > model_min:
            ax.set_ylim(model_min * 0.85, model_max * 1.15)
        else:
            ax.set_ylim(model_min * 0.85, model_max * 1.15 + 1e-30)

    ax.set_title(f'{date_label}  |  JD {jd_day}  |  Av={daily_params["Av"]:.2f}, logMdot={daily_params["logMdot"]:.3f}{title_extra}')
    ax.grid(True, alpha=0.3)
    ax.legend()

    if ax_residual is not None:
        ax_residual.plot(
            lambda_um_broken,
            with_gap_breaks(residual_flux),
            color='tab:green',
            linewidth=0.9,
            alpha=0.65,
            label='Residual (model - data)'
        )
        ax_residual.plot(
            lambda_um_broken,
            with_gap_breaks(residual_running),
            color='tab:red',
            linewidth=1.6,
            alpha=0.95,
            label=f'Running average (window={int(max(1, residual_window))})'
        )
        ax_residual.axhline(0.0, color='black', linestyle='--', linewidth=0.9, alpha=0.8)
        ax_residual.set_xlabel('Wavelength (μm)')
        ax_residual.set_ylabel('Δ Flux density (Jy)')
        ax_residual.grid(True, alpha=0.3)
        ax_residual.legend(loc='best', fontsize=9)

    if created_fig:
        plt.tight_layout()
        if show:
            plt.show()
    return fig, ax


def _spectral_day_has_data_in_range(results, jd_day, show_range=None):
    if 'daily_data' not in results or jd_day not in results['daily_data']:
        return False

    day_df = results['daily_data'][jd_day]
    if day_df is None or len(day_df) == 0:
        return False

    try:
        lambda_um = _coerce_wavelength_meters(day_df) * 1e6
    except Exception:
        return False

    valid = np.isfinite(lambda_um)
    if show_range is not None:
        try:
            xmin, xmax = float(show_range[0]), float(show_range[1])
            if xmin > xmax:
                xmin, xmax = xmax, xmin
            valid &= (lambda_um >= xmin) & (lambda_um <= xmax)
        except Exception:
            pass

    return bool(np.any(valid))


def plot_spectral_fit_grid(results, save_dir='results/generated/pictures/spectral_mode', filename='spectral_fit_overview.pdf', show_individual=True, residual_window=51, show_range=None):
    all_jd_days = sorted(results.get('daily_params', {}).keys())
    jd_days = [
        jd_day for jd_day in all_jd_days
        if _spectral_day_has_data_in_range(results, jd_day, show_range=show_range)
    ]
    if len(jd_days) == 0:
        print('No spectral days available to plot')
        return None
    skipped_days = len(all_jd_days) - len(jd_days)
    if skipped_days > 0:
        print(f'Skipping {skipped_days} spectral day(s) with no data in the plotted wavelength range')

    output_prefix = os.path.basename(str(save_dir).rstrip('/\\')) or 'spectral_mode'
    n_rows = len(jd_days)
    fig, axes = plt.subplots(
        n_rows,
        2,
        figsize=(16, 4.0 * n_rows),
        squeeze=False,
        gridspec_kw={'width_ratios': [1.8, 1.2]}
    )

    for index, jd_day in enumerate(jd_days):
        plot_spectral_fit_day(
            results,
            jd_day,
            ax=axes[index, 0],
            ax_residual=axes[index, 1],
            show=False,
            residual_window=residual_window,
            show_range=show_range,
        )

    plt.tight_layout()
    if os.path.isabs(str(save_dir)):
        os.makedirs(save_dir, exist_ok=True)
        output_path = os.path.join(save_dir, filename)
    else:
        output_path = _generated_figure_path(f'{output_prefix}_{filename}')
    fig.savefig(output_path, dpi=200, bbox_inches='tight')

    if show_individual:
        plt.show()
        return None

    return fig


def visualize_spectral_results(results, df=None, save_dir='results/generated/pictures/spectral_mode', show_parameter_evolution=True,
                               show_individual=True, residual_window=51, show_range=None):
    output_prefix = os.path.basename(str(save_dir).rstrip('/\\')) or 'spectral_mode'
    parameter_path = (
        os.path.join(save_dir, 'parameter_evolution.pdf')
        if os.path.isabs(str(save_dir))
        else f'{output_prefix}_parameter_evolution.pdf'
    )
    plot_parameter_evolution(
        results,
        df,
        save_path=parameter_path,
        show=show_parameter_evolution,
    )
    return plot_spectral_fit_grid(
        results,
        save_dir=save_dir,
        show_individual=show_individual,
        residual_window=residual_window,
        show_range=show_range,
    )


def visualize_results(results, df, filename):
    """Main visualization function for different modes"""

    if results.get('fit_info', {}).get('data_mode') == 'spectral':
        if results['fit_info']['AR_mode']:
            plot_ar_surface(results, filename)
        return visualize_spectral_results(results, df, save_dir=filename if filename else 'results/generated/pictures/spectral_mode')

    if results['fit_info']['AR_mode']:
        plot_ar_surface(results, filename)

    if results['fit_info']['red_excess_mode']:
        plot_red_excess_results(results, filename)

    # Standard time series plots
    plot_results_regularized(results, df)


def plot_standard_results(results):
    """Plot standard time series results"""

    # Extract time series data
    times = []
    mdots = []
    mdot_errors = []
    avs = []
    av_errors = []
    m_star = _stellar_mass_solar(results)

    for jd_day, params in results['daily_params'].items():
        times.append(jd_day)
        mdots.append(m_star * params['Mdot'])
        mdot_errors.append(m_star * params.get('Mdot_err', 0.1 * params['Mdot']))
        avs.append(params['Av'])
        av_errors.append(params.get('Av_err', 0.1))

    times = np.array(times)
    mdots = np.array(mdots)
    mdot_errors = np.array(mdot_errors)
    avs = np.array(avs)
    av_errors = np.array(av_errors)

    # Create plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Mdot evolution
    ax1.errorbar(times, mdots, yerr=mdot_errors, fmt='o-', capsize=3)
    ax1.set_ylabel('M_star*Mdot (M_sun^2/yr)')
    ax1.set_title('M_star*Mdot Evolution')
    ax1.set_yscale('log')
    ax1.grid(True)

    # Av evolution
    ax2.errorbar(times, avs, yerr=av_errors, fmt='o-', capsize=3)
    ax2.set_xlabel('Time (JD)')
    ax2.set_ylabel('Visual Extinction (mag)')
    ax2.set_title('Visual Extinction Evolution')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


def example_visualization(df, accretion_model, fitting_results):

    # Create comprehensive visualization
    viz_results = comprehensive_results_visualization(
        fitting_results, df, accretion_model, save_dir="results/generated/pictures"

    )

    return viz_results


def _extract_regularized_series(results):
    daily_params = results['daily_params']
    jd_days = np.array(sorted(daily_params.keys()), dtype=float)
    m_star = _stellar_mass_solar(results)

    mdot_values = np.array([
        m_star * (10 ** (daily_params[jd]['logMdot']) / M_sun * year)
        for jd in jd_days
    ], dtype=float)
    mdot_errors = np.array([
        m_star * daily_params[jd].get('Mdot_err', np.nan)
        for jd in jd_days
    ], dtype=float)
    av_values = np.array([daily_params[jd]['Av'] for jd in jd_days], dtype=float)
    av_errors = np.array([daily_params[jd].get('Av_err', np.nan) for jd in jd_days], dtype=float)

    return jd_days, mdot_values, mdot_errors, av_values, av_errors


def _read_peter_results(path):
    # Header is 5 lines: title, range, blank, columns, separator
    peter_df = pd.read_csv(
        path,
        sep=r'\s+',
        skiprows=5,
        names=['JD', 'Av', 'Mdot'],
        engine='python'
    )
    peter_df = peter_df.apply(pd.to_numeric, errors='coerce').dropna()
    return peter_df.sort_values('JD')


def compare_daily_params_evolution(results_dir='results/generated/datas', parameters=None, save_path=None,
                                   include_errorbars=True, peter_data=None, model_mode='all'):
    """
    Compare parameter evolution across all *_daily_params.csv files in a folder.

    One color is assigned per fit and reused for every parameter subplot.
    Peter's results (peter_data dict) are overlaid as dashed lines on the
    Mdot and Av panels only.

    Parameters
    ----------
    results_dir : str
        Directory containing *_daily_params.csv files.
    parameters : list of str or None
        Which parameters to plot. None → all columns found.
    save_path : str or None
        If given, saves the figure to this path.
    include_errorbars : bool
        Whether to plot error bars from *_err columns.
    peter_data : dict or None
        Dict of {label: DataFrame} where each DataFrame has columns JD, Mdot, Av
        (as returned by _read_peter_results). Plotted on Mdot / Av panels only.
    model_mode : str
        Which data/model subset to compare.
        - 'all': plot all available fits
        - 'jh': plot only JH-family fits
        - 'jhk': plot only JHK-family fits
        - 'photometric': plot only non-spectral fits
        - 'spectrometric': plot only spectral fits
        - 'rede': plot only red excess fits
        - 'peter': plot only Peter data overlays
        - 'Av_crosscheck': plot Av only for JH/JHK photometric fits, JH/JHK
          spectroscopic fits, and data/raw/photometry/av_molecular.txt molecular Av data
    """
    mode = str(model_mode).lower().strip()
    if str(results_dir).replace('\\', '/').rstrip('/') in ('results/generated', '../results/generated'):
        results_dir = _generated_datas_dir()
    mode_aliases = {
        'spectral': 'spectrometric',
        'spectroscopic': 'spectrometric',
        'spectrometry': 'spectrometric',
    }
    mode = mode_aliases.get(mode, mode)

    def _model_group_for_fit(label):
        label_lower = str(label).lower().strip()
        if label_lower in ('basic_jhk', 'spectral_jhk') or 'spectral_jhk' in label_lower:
            return 'jhk'
        if label_lower in ('basic_jh', 'spectral_jh') or 'spectral_jh' in label_lower:
            return 'jh'
        return label_lower

    def _is_spectral_fit(label):
        return 'spectral' in str(label).lower()

    def _is_rede_fit(label):
        return 'rede' in str(label).lower()

    def _model_group_for_peter(label):
        label_lower = str(label).lower().strip()
        if label_lower == 'jh':
            return 'jh'
        if label_lower == 'jhk':
            return 'jhk'
        return f'peter_{label_lower}'

    def _find_av_molecular_path():
        candidates = [
            os.path.join(os.path.dirname(results_dir), 'data', 'raw', 'photometry', 'av_molecular.txt'),
            os.path.join(os.path.dirname(results_dir), 'data', 'av_molecular.txt'),
            os.path.join(os.path.dirname(results_dir), 'Data', 'av_molecular.txt'),
            os.path.join('data', 'raw', 'photometry', 'av_molecular.txt'),
            os.path.join('data', 'av_molecular.txt'),
            os.path.join('Data', 'av_molecular.txt'),
            'av_molecular.txt',
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
        raise FileNotFoundError(
            "model_mode='Av_crosscheck' requested, but av_molecular.txt was not found "
            "in data/raw/photometry/, data/, Data/, or alongside the current working directory"
        )

    pattern = os.path.join(results_dir, '*_daily_params.csv')
    daily_files = sorted(glob.glob(pattern))
    if not daily_files:
        raise FileNotFoundError(f'No daily-parameter files found with pattern: {pattern}')

    fit_frames = {}
    for file_path in daily_files:
        fit_label = os.path.basename(file_path).replace('_daily_params.csv', '')
        try:
            frame = pd.read_csv(file_path)
        except Exception as exc:
            print(f"Skipping {file_path}: could not read file ({exc})")
            continue

        if 'JD_day' not in frame.columns or len(frame) == 0:
            print(f"Skipping {file_path}: missing or empty JD_day")
            continue

        frame = frame.copy()
        frame['JD_day'] = pd.to_numeric(frame['JD_day'], errors='coerce')
        frame = frame.dropna(subset=['JD_day'])
        if len(frame) == 0:
            continue

        frame['JD_day'] = frame['JD_day'].astype(float)
        frame = frame.sort_values('JD_day').reset_index(drop=True)
        fit_frames[fit_label] = frame

    allowed_modes = {'all', 'jh', 'jhk', 'photometric', 'spectrometric', 'rede', 'peter', 'av_crosscheck'}
    if mode not in allowed_modes:
        raise ValueError(
            f"Invalid model_mode '{model_mode}'. Use one of: "
            "all, jh, jhk, photometric, spectrometric, rede, peter, Av_crosscheck"
        )

    if mode in ('jh', 'jhk'):
        fit_frames = {
            label: frame
            for label, frame in fit_frames.items()
            if _model_group_for_fit(label) == mode
        }
        if peter_data is not None:
            peter_data = {
                p_label: p_df
                for p_label, p_df in peter_data.items()
                if _model_group_for_peter(p_label) == mode
            }
    elif mode == 'photometric':
        fit_frames = {
            label: frame
            for label, frame in fit_frames.items()
            if not _is_spectral_fit(label)
        }
        peter_data = None
    elif mode == 'spectrometric':
        fit_frames = {
            label: frame
            for label, frame in fit_frames.items()
            if _is_spectral_fit(label)
        }
        peter_data = None
    elif mode == 'rede':
        fit_frames = {
            label: frame
            for label, frame in fit_frames.items()
            if _is_rede_fit(label)
        }
        peter_data = None
    elif mode == 'peter':
        fit_frames = {}
    elif mode == 'av_crosscheck':
        crosscheck_order = ['basic_JHK_newdf', 'basic_JH_newdf', 'spectral_JHK', 'spectral_JH']
        fit_frames = {
            label: fit_frames[label]
            for label in crosscheck_order
            if label in fit_frames
        }
        peter_data = None

    if mode != 'peter' and not fit_frames:
        raise ValueError(f"No matching fit files found for model_mode='{mode}' in: {results_dir}")
    if mode == 'peter' and (peter_data is None or len(peter_data) == 0):
        raise ValueError("model_mode='peter' requested, but peter_data is empty or missing")

    if mode == 'av_crosscheck':
        try:
            from .ultimate_file_organisers import _read_av_molecular_results
        except ImportError:
            from ultimate_file_organisers import _read_av_molecular_results

        missing_labels = [
            label
            for label in ['basic_JHK_newdf', 'basic_JH_newdf', 'spectral_JHK', 'spectral_JH']
            if label not in fit_frames
        ]
        if missing_labels:
            print(f"Warning: missing Av_crosscheck fit files: {', '.join(missing_labels)}")

        molecular_path = _find_av_molecular_path()
        molecular_av = _read_av_molecular_results(molecular_path)
        if molecular_av.empty:
            raise ValueError(f"No molecular Av rows found in: {molecular_path}")

        selected_params = ['Av']
        fig, ax = plt.subplots(1, 1, figsize=(13, 5), sharex=True)
        dark2 = plt.get_cmap('Dark2')
        color_labels = [
            'basic_JHK_newdf',
            'basic_JH_newdf',
            'spectral_JHK',
            'spectral_JH',
            'silicate',
            'water ice',
        ]
        dark2_colors = {
            label: dark2(i % dark2.N)
            for i, label in enumerate(color_labels)
        }
        display_labels = {
            'basic_JHK_newdf': 'photometric JHK',
            'basic_JH_newdf': 'photometric JH',
            'spectral_JHK': 'spectral JHK',
            'spectral_JH': 'spectral JH',
        }

        for label, frame in fit_frames.items():
            if 'Av' not in frame.columns:
                continue
            cols = ['JD_day', 'Av']
            if include_errorbars and 'Av_err' in frame.columns:
                cols.append('Av_err')
            work = frame[cols].copy()
            work['JD_day'] = pd.to_numeric(work['JD_day'], errors='coerce')
            work['Av'] = pd.to_numeric(work['Av'], errors='coerce')
            work = work.dropna(subset=['JD_day', 'Av'])
            if len(work) == 0:
                continue

            x = work['JD_day'].to_numpy(dtype=float)
            y = work['Av'].to_numpy(dtype=float)
            yerr = None
            if include_errorbars and 'Av_err' in work.columns:
                yerr = pd.to_numeric(work['Av_err'], errors='coerce').to_numpy(dtype=float).copy()
                yerr[~np.isfinite(yerr)] = np.nan
            display_label = display_labels.get(label, label)
            if _is_spectral_fit(label):
                ax.errorbar(
                    x, y, yerr=yerr,
                    fmt='*', linestyle='none', markersize=10, alpha=0.9,
                    capsize=2 if yerr is not None else 0,
                    color=dark2_colors[label],
                    label=display_label,
                    zorder=4,
                )
            else:
                ax.errorbar(
                    x, y, yerr=yerr,
                    fmt='o-', markersize=5, linewidth=1.5, alpha=0.9,
                    capsize=2 if yerr is not None else 0,
                    color=dark2_colors[label],
                    label=display_label,
                    zorder=3,
                )

        for method in ['silicate', 'water ice']:
            method_data = molecular_av[molecular_av['method'] == method]
            if method_data.empty:
                continue
            ax.scatter(
                method_data['JD'].to_numpy(dtype=float),
                method_data['Av'].to_numpy(dtype=float),
                marker='s' if method == 'silicate' else 'D',
                s=65, alpha=0.9,
                color=dark2_colors[method],
                label=f'Molecular Av ({method})',
                zorder=5,
            )

        ax.set_ylabel('Av')
        ax.set_xlabel('Julian Date')
        ax.grid(True, alpha=0.3)
        add_gregorian_top_axis(ax, n_ticks=6, fontsize=10, use_grid_ticks=True, dynamic=True)
        fig.suptitle('Extinction evolution crosscheck', fontsize=16)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            fig.legend(
                handles, labels,
                loc='lower center',
                ncol=min(6, len(labels)),
                frameon=True,
                bbox_to_anchor=(0.5, 0.0),
            )

        plt.tight_layout(rect=[0, 0.12, 1, 0.95])
        if save_path:
            fig.savefig(_generated_figure_path(save_path), dpi=300, bbox_inches='tight')
        plt.show()

        return fig, fit_frames

    available_params = []
    for frame in fit_frames.values():
        for column in frame.columns:
            if column == 'JD_day' or column.endswith('_err'):
                continue
            if column not in available_params:
                available_params.append(column)
    if peter_data is not None:
        peter_has_mdot = any('Mdot' in p_df.columns for p_df in peter_data.values())
        peter_has_av = any('Av' in p_df.columns for p_df in peter_data.values())
        if peter_has_mdot and 'Mdot' not in available_params:
            available_params.append('Mdot')
        if peter_has_av and 'Av' not in available_params:
            available_params.append('Av')

    if parameters is None:
        selected_params = available_params
    elif isinstance(parameters, str):
        selected_params = [parameters]
    else:
        selected_params = list(parameters)

    selected_params = [
        param for param in selected_params
        if any(param in frame.columns for frame in fit_frames.values())
        or (
            peter_data is not None
            and param in ('Mdot', 'Av')
            and any(param in p_df.columns for p_df in peter_data.values())
        )
    ]
    if not selected_params:
        raise ValueError('No matching parameters were found to plot')

    n_params = len(selected_params)
    fig, axes = plt.subplots(n_params, 1, figsize=(15, max(4.0, 5 * n_params)), sharex=True)
    if n_params == 1:
        axes = [axes]

    fit_labels = sorted(fit_frames.keys())

    # Load reduced chi-squared values from result files
    try:
        from .ultimate_file_organisers import result_opener as _result_opener
    except ImportError:
        from ultimate_file_organisers import result_opener as _result_opener
    fit_chi_squared = {}
    for label in fit_labels:
        result_path = os.path.join(results_dir, label)
        try:
            res = _result_opener(filepath=result_path)
            chi2 = res.get('reduced_chi_squared', np.nan)
            fit_chi_squared[label] = chi2
        except Exception as exc:
            print(f"Warning: Could not load reduced_chi_squared for {label}: {exc}")
            fit_chi_squared[label] = np.nan

    def _color_group_for_fit(label):
        return _model_group_for_fit(label)

    def _color_group_for_peter(label):
        return _model_group_for_peter(label)

    explicit_fit_colors = {
        'basic_jh': '#0b3c8c',
        'spectral_jh': '#5fa8ff',
        'basic_jhk': '#1b7f3a',
        'spectral_jhk': '#49a65a',
    }
    group_default_fit_colors = {
        'jh': '#0b3c8c',
        'jhk': '#1b7f3a',
    }
    explicit_peter_colors = {
        'jh': '#2f6fdf',
        'jhk': '#7bc96f',
    }
    other_fit_palette = [
        '#e15759',
        '#b07aa1',
        '#9c755f',
        '#ff9da7',
        '#f28e2b',
        '#bab0ab',
    ]
    fit_groups = [_color_group_for_fit(label) for label in fit_labels]
    extra_groups = []
    for group in fit_groups:
        if group not in ('jh', 'jhk') and group not in extra_groups:
            extra_groups.append(group)

    extra_group_colors = {
        group: other_fit_palette[i % len(other_fit_palette)]
        for i, group in enumerate(extra_groups)
    }
    fit_colors = {}
    for label in fit_labels:
        label_lower = str(label).lower()
        group = _color_group_for_fit(label)
        if label_lower in explicit_fit_colors:
            fit_colors[label] = explicit_fit_colors[label_lower]
        elif group in group_default_fit_colors:
            fit_colors[label] = group_default_fit_colors[group]
        else:
            fit_colors[label] = extra_group_colors.get(
                group,
                other_fit_palette[len(fit_colors) % len(other_fit_palette)],
            )

    def _marker_for_fit(label):
        if _is_spectral_fit(label):
            return '*'
        return 'o'

    def _linestyle_for_fit(label):
        if _is_spectral_fit(label):
            return '-'
        return '-.'

    def _label_with_chi_squared(label):
        """Format label to include reduced chi-squared if available."""
        chi2 = fit_chi_squared.get(label, np.nan)
        if np.isfinite(chi2):
            return f"{label} (χ² = {chi2:.2f})"
        return label

    _peter_colors = ['#edc948', '#af7aa1', '#ff9da7', '#9c755f', '#bab0ab']

    for ax, param in zip(axes, selected_params):
        for label in fit_labels:
            frame = fit_frames[label]
            if param not in frame.columns:
                continue

            marker = _marker_for_fit(label)
            line_style = _linestyle_for_fit(label)

            cols = ['JD_day', param]
            err_col = f'{param}_err'
            if include_errorbars and err_col in frame.columns:
                cols.append(err_col)

            work = frame[cols].copy()
            work['JD_day'] = pd.to_numeric(work['JD_day'], errors='coerce')
            work[param] = pd.to_numeric(work[param], errors='coerce')
            work = work.dropna(subset=['JD_day', param])
            if len(work) == 0:
                continue

            x = work['JD_day'].to_numpy(dtype=float)
            y = work[param].to_numpy(dtype=float)

            display_label = _label_with_chi_squared(label)

            if include_errorbars and err_col in work.columns:
                yerr = pd.to_numeric(work[err_col], errors='coerce').to_numpy(dtype=float).copy()
                yerr[~np.isfinite(yerr)] = np.nan
                ax.errorbar(
                    x, y, yerr=yerr,
                    fmt=marker, linestyle=line_style, markersize=7 if marker == '*' else 5, linewidth=1.5,
                    capsize=2, alpha=0.9,
                    color=fit_colors[label],
                    label=display_label,
                )
            else:
                ax.plot(
                    x, y,
                    marker=marker, linestyle=line_style, markersize=7 if marker == '*' else 5, linewidth=1.5, alpha=0.9,
                    color=fit_colors[label],
                    label=display_label,
                )

        # Overlay Peter's results (Mdot and Av panels only)
        if peter_data is not None and param in ('Mdot', 'Av'):
            peter_col = 'Mdot' if param == 'Mdot' else 'Av'
            for p_idx, (p_label, p_df) in enumerate(peter_data.items()):
                if peter_col not in p_df.columns or 'JD' not in p_df.columns:
                    continue
                p_group = _color_group_for_peter(p_label)
                p_color = explicit_peter_colors.get(p_group, _peter_colors[p_idx % len(_peter_colors)])
                px = pd.to_numeric(p_df['JD'], errors='coerce').to_numpy(dtype=float)
                py = pd.to_numeric(p_df[peter_col], errors='coerce').to_numpy(dtype=float)
                valid = np.isfinite(px) & np.isfinite(py)
                px = px[valid]
                py = py[valid]
                peter_err_col = f'{peter_col}_err'

                if include_errorbars and peter_err_col in p_df.columns:
                    pyerr_full = pd.to_numeric(p_df[peter_err_col], errors='coerce').to_numpy(dtype=float)
                    pyerr = pyerr_full[valid]
                    pyerr[~np.isfinite(pyerr)] = np.nan
                    ax.errorbar(
                        px, py, yerr=pyerr,
                        fmt='^', linestyle='--', markersize=8, linewidth=1.5,
                        capsize=2, alpha=0.85,
                        color=p_color,
                        label=f'Peter ({p_label})',
                    )
                else:
                    ax.plot(
                        px, py,
                        marker='^', linestyle='--', markersize=8, linewidth=1.5, alpha=0.85,
                        color=p_color,
                        label=f'Peter ({p_label})',
                    )

        ax.set_ylabel(param)
        ax.grid(True, alpha=0.3)
        if param in ('Mdot',):
            positive = []
            for line in ax.get_lines():
                line_y = line.get_ydata()
                if line_y is not None:
                    positive.extend([v for v in line_y if np.isfinite(v) and v > 0])
            if len(positive) > 0:
                ax.set_yscale('log')
        add_gregorian_top_axis(ax, n_ticks=6, fontsize=10, use_grid_ticks=True, dynamic=True)

    axes[-1].set_xlabel('Julian Date')

    # Title at the very top, legend below the bottom axis — no overlap
    fig.suptitle('Daily-parameter evolution comparison by fit', fontsize=16)
    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        for hi, li in zip(h, l):
            if li not in labels:
                handles.append(hi)
                labels.append(li)
    if handles:
        n_cols = min(5, len(labels))
        fig.legend(handles, labels,
                   loc='lower center',
                   ncol=n_cols,
                   frameon=True,
                   bbox_to_anchor=(0.5, 0.0))

    plt.tight_layout(rect=[0, 0.06, 1, 0.97])
    if save_path:
        fig.savefig(_generated_figure_path(save_path), dpi=300, bbox_inches='tight')
    plt.show()

    return fig, fit_frames

__all__ = [
    'julian_to_calendar',
    'add_gregorian_top_axis',
    'get_model_predictions',
    'plot_results_regularized',
    'plot_parameter_evolution',
    'create_color_plots_with_model',
    'plot_unique_filter_with_model',
    'create_residual_plots',
    'comprehensive_results_visualization',
    'plot_ar_surface',
    'plot_red_excess_results',
    'plot_spectral_fit_day',
    'plot_spectral_fit_grid',
    'visualize_spectral_results',
    'visualize_results',
    'plot_standard_results',
    'example_visualization',
    '_extract_regularized_series',
    '_read_peter_results',
    'compare_daily_params_evolution',
]
