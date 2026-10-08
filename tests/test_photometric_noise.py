"""Noise choices propagate through the real JH inversion and covariance."""
import numpy as np
import pandas as pd
import pytest

from accretion.ultimate_common import M_sun, R_sun, year, constants
from accretion.ultimate_fitting import ultimate_fitting_regularized
from accretion.ultimate_physics import accretion_model
from accretion.ultimate_helpers import _prepare_ar_day_arrays


def _fit(errors, missing=.01, systematic=.05, spectral=False):
    wave = np.array([1.235, 1.662])
    params = dict(M=.5*M_sun, R_star=3*R_sun, R_in=3*R_sun,
                  R_out=2*constants.au, distance=700*constants.parsec)
    flux = np.exp(accretion_model(constants.c/(wave*1e-6), 1.2e-4*M_sun/year,
                                 params['M'], params['R_star'], params['R_out'],
                                 params['R_in'], params['distance'], 12.))
    frame = pd.DataFrame(dict(JD=2450000.5, Date='1995-10-10', Filter=['J','H'],
                              Lambda=wave, Flux=flux, Fluxerr=flux*np.asarray(errors),
                              Mag=-2.5*np.log10(flux), Magerr=0., ZP=1.))
    result = ultimate_fitting_regularized(
        ('J','H'), ('J','H'), df=frame, lambda_reg=0, initial_params=params,
        data_mode='spectral' if spectral else 'photometry',
        missing_photometric_fractional_error=missing,
        systematic_fractional_error=systematic)
    assert result['success']
    return result, next(iter(result['daily_params'].values()))


def test_missing_errors_change_covariance_without_changing_jh_solution():
    result, new = _fit([0., np.nan])
    _, old = _fit([0., np.nan], missing=.1)
    _, reported = _fit([.01, .01], missing=.1)
    ratio = np.hypot(.01,.05)/np.hypot(.1,.05)
    for key in ('Mdot','Av'):
        np.testing.assert_allclose(new[key], old[key], rtol=1e-7)
        np.testing.assert_allclose(new[key+'_err']/old[key+'_err'], ratio, rtol=1e-6)
        np.testing.assert_allclose(new[key+'_err'], reported[key+'_err'], rtol=1e-6)
    assert result['fit_info']['missing_photometric_fractional_error'] == .01
    assert result['fit_info']['systematic_fractional_error'] == .05


def test_reported_errors_and_spectral_fallback_do_not_depend_on_photometric_fallback():
    for spectral, errors in [(False,[.02,.03]), (True,[0.,np.nan])]:
        _, new = _fit(errors, missing=.01, spectral=spectral)
        _, old = _fit(errors, missing=.1, spectral=spectral)
        for key in ('Mdot','Av','Mdot_err','Av_err'):
            np.testing.assert_allclose(new[key], old[key], rtol=1e-8)


def test_ar_preparation_uses_same_photometric_noise_and_retains_spectral_fallback():
    frame = pd.DataFrame(dict(Filter=['J','H'], Lambda=[1.235,1.662],
                              Frequency=[2e14,1.8e14], Flux=[1.,1.], Fluxerr=[0.,.02]))
    _, _, photo_errors = _prepare_ar_day_arrays(frame, ('J','H'))
    _, _, spectral_errors = _prepare_ar_day_arrays(frame, ('J','H'), data_mode='spectral')
    np.testing.assert_allclose(photo_errors, np.hypot([.01,.02],.05))
    np.testing.assert_allclose(spectral_errors, np.hypot([.1,.02],.05))


@pytest.mark.parametrize('options', [
    {'missing_photometric_fractional_error':0},
    {'missing_photometric_fractional_error':np.nan},
    {'systematic_fractional_error':-.05},
])
def test_invalid_noise_options_fail_before_fitting(options):
    with pytest.raises(ValueError):
        ultimate_fitting_regularized(('J','H'), ('J','H'), **options)
