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
    monkeypatch.setattr(models, 'accretion_model_magnitude_error', lambda *args: 0.0)
    data = pd.DataFrame({'JD': [0., 1., 2., 3., 4.], 'Filter': 'L', 'Lambda': 3.5, 'ZP': 280.})
    params = {1.: {'Mdot': 1e-5, 'Av': 10.}, 3.: {'Mdot': 3e-5, 'Av': 20.}}
    result = models.calculate_filter_accretion_models(data, params)[('L', 3.5)]
    np.testing.assert_array_equal(result.JD, [1., 2., 3.])
    np.testing.assert_allclose(result.Mdot, [1e-5, 2e-5, 3e-5])
    np.testing.assert_allclose(result.Av, [10., 15., 20.])
    assert result.Is_interpolated_fit.tolist() == [False, True, False]


def test_explicit_residuals_keep_interpolated_points_and_marker(monkeypatch):
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
    assert 'Interpolated L/W1' in axes[-1].get_legend_handles_labels()[1]
    plt.close(fig)
