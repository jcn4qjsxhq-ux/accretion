"""Regression checks for supplying provenance-preserving final-round residuals."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from accretion import final_panels as panels
from accretion import final_panel_models as models


def test_model_interpolates_parameters_but_does_not_extrapolate(monkeypatch):
    seen = []

    def model(freq, mdot, mass, rstar, rout, rin, distance, av):
        seen.append((mdot, av))
        return np.array([0.0])

    monkeypatch.setattr(models, 'accretion_model', model)
    monkeypatch.setattr(models, 'accretion_model_magnitude_error', lambda *args, **kwargs: 0.0)
    data = pd.DataFrame({'JD': [0., 1., 2., 3., 4.], 'Filter': 'L', 'Lambda': 3.5, 'ZP': 280.})
    params = {1.: {'Mdot': 1e-5, 'Av': 10.}, 3.: {'Mdot': 3e-5, 'Av': 20.}}
    result = models.calculate_filter_accretion_models(data, params)[('L', 3.5)]
    np.testing.assert_array_equal(result.JD, [1., 2., 3.])
    np.testing.assert_allclose(result.Mdot, [1e-5, 2e-5, 3e-5])
    np.testing.assert_allclose(result.Av, [10., 15., 20.])
    assert result.Is_interpolated_fit.tolist() == [False, True, False]


def test_explicit_residuals_keep_measurement_interpolation_without_special_marker(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError('Supplied residuals must not be regenerated or filtered')

    monkeypatch.setattr(panels, 'prepare_final_panel_residuals', unexpected)
    dates = [2451000.5, 2451002.5]
    data = pd.DataFrame({'JD': dates, 'Filter': ['J', 'J'], 'Lambda': [1.235]*2,
                         'Mag': [10., 11.], 'Magerr': [.1]*2, 'ZP': [1600.]*2})
    result = {'daily_params': {jd: {'Mdot': 1e-5, 'Mdot_err': 1e-6, 'Av': 10., 'Av_err': .1} for jd in dates},
              'global_params': {}, 'fit_info': {}, 'daily_data': {}}
    flux = pd.DataFrame({'JD_day': [2451001.5], 'Filter': ['L'], 'Group': ['L+W1'],
                         'Difference_flux': [1.], 'Difference_err_flux': [.1], 'Interpolated': ['yes']})
    residuals = {'flux': flux, 'magnitude': pd.DataFrame(), 'group_keys': ['L+W1'],
                 'group_map': {'L+W1': ['L', 'W1']}, 'filters': ['L','W1'],
                 'flux_max_by_filter': pd.Series({'L': 1.})}
    fig, axes, returned = panels.plot_final_panels_0_to_4(
        data, result, selected_filters=('J',), photometric_fit_filters=('J','H'),
        plot_panel0_fits=False, show_panel3=False, show=False, residuals_override=residuals,
    )
    assert returned is residuals
    assert 'Interpolated L/W1' not in axes[-1].get_legend_handles_labels()[1]
    assert 'L+W1 (plot-only)' in axes[-1].get_legend_handles_labels()[1]
    plt.close(fig)


def test_excess_selection_distinguishes_parameter_and_measurement_interpolation():
    frame = pd.DataFrame({
        'Filter': ['L', 'W1', 'K', 'L', 'J', 'L'],
        'Group': ['L+W1', 'L+W1', 'K', 'L+W1', 'J', 'L+W1'],
        'Difference_flux': [1., -1., 1., 1., 1., 0.],
        'Difference_err_flux': [.74, .75, 1., .1, .1, .1],
        'Interpolated': ['yes', 'no', 'yes', 'no', 'no', 'no'],
        'Is_interpolated_fit': [False, False, False, True, True, False],
    })
    assert panels.excess_plot_selection(frame).tolist() == [True, False, True, False, False, False]


def test_calendar_grid_starts_at_january_first_and_tracks_zoom():
    from datetime import datetime
    from accretion.ultimate_visualisation import add_gregorian_top_axis, julian_to_calendar
    fig, ax = plt.subplots()
    ax.set_xlim(2444000, 2461300)
    top = add_gregorian_top_axis(ax, calendar_years=True, show_title=False)
    dates = julian_to_calendar(ax.get_xticks())
    assert all(d.endswith('-01-01') and int(d[:4]) % 5 == 0 for d in dates)
    assert top.get_xlabel() == ''
    assert all(len(t.get_text()) == 4 for t in top.get_xticklabels())
    ax.set_xlim(2460000, 2460100)
    assert top.get_xlim() == ax.get_xlim()
    assert all(2460000 <= t <= 2460100 for t in top.get_xticks())
    plt.close(fig)


def test_model_error_uses_mdot_extinction_covariance(monkeypatch):
    def fake(freq, mdot, mass, rstar, rout, rin, distance, av):
        return np.array([np.log(mdot)-av])
    monkeypatch.setattr(models,'accretion_model',fake)
    parameters={'M':1.,'R_star':1.,'R_out':1.,'R_in':1.,'distance':1.}
    independent=models.accretion_model_magnitude_error(1.,1.,1.,1.,.01,.01,parameters)
    correlated=models.accretion_model_magnitude_error(1.,1.,1.,1.,.01,.01,parameters,log_mdot_av_cov=.0001/np.log(10.))
    assert correlated < independent*.01


def test_model_error_large_correlated_errors_cancel_in_log_coordinates(monkeypatch):
    # Exact cancellation must survive large reported parameter uncertainties.
    monkeypatch.setattr(models, 'accretion_model', lambda freq, mdot, mass, rstar, rout, rin, distance, av: np.array([np.log(mdot)-av]))
    parameters={'M':1.,'R_star':1.,'R_out':1.,'R_in':1.,'distance':1.}
    error=models.accretion_model_magnitude_error(1.,1.,1.,1.,.8,.8,parameters,log_mdot_av_cov=.64/np.log(10.))
    assert error < 1e-5
