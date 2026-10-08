import numpy as np
import pandas as pd
import pytest

from accretion.ultimate_common import M_sun, year, constants
from accretion.ultimate_physics import accretion_model, planck_model_custom
from accretion.ultimate_fitting import ultimate_fitting_regularized, _resolve_red_excess_component
from accretion.ultimate_file_organisers import load_database_new


def test_parser_lists_all_unknown_wavelengths(tmp_path):
    p=tmp_path/'unknown.txt'
    p.write_text('header\nseparator\n2450000 1995-01-01 8.123 1 .1 1 1 .1 no 0\n2450001 1995-01-02 9.456 1 .1 1 1 .1 no 0\n')
    with pytest.raises(ValueError, match='8.123, 9.456'):
        load_database_new(p)


def test_joint_component_resolver_preserves_fitted_temperature_and_radius():
    _, t, r = _resolve_red_excess_component({'T_bb':1200.,'R_bb':3*constants.au},{'red_excess_fit_param':'both','fixed_T_bb':1500.,'fixed_R_bb':2*constants.au})
    assert t == 1200.
    assert r == 3*constants.au


def test_jh_only_spectrum_has_no_red_fit():
    frame=pd.DataFrame({'JD':[2450000.5]*3,'Lambda':[1.2,1.4,1.6],'Flux':[1.]*3,'Fluxerr':[.1]*3})
    assert ultimate_fitting_regularized((),(),df=frame,data_mode='spectral',red_excess_mode=True) is None


@pytest.mark.parametrize('long_continuum', [False, True])
def test_spectral_k_continuum_recovers_bb_with_ice_absorption_masked(long_continuum):
    """Require K and L coverage; deep ice absorption cannot bias eligible fits."""
    red = [2.0, 2.1, 2.2, 2.35, 2.48]
    if long_continuum:
        red += [3.31, 3.5, 3.8, 4.0, 4.8]
    wave = np.array([1.2, 1.25, 1.5, 1.6, 1.9, 2.8, 3.0, 3.3] + red)
    freq = constants.c/(wave*1e-6)
    mdot = 1.2e-4*M_sun/year
    radius = 1.5*constants.au
    globals_ = dict(M=.5*M_sun, R_star=3*6.957e8, R_in=3*6.957e8,
                    R_out=2*constants.au, distance=700*constants.parsec)
    flux = np.exp(accretion_model(freq, mdot, globals_['M'], globals_['R_star'],
        globals_['R_out'], globals_['R_in'], globals_['distance'], 12.))
    flux += np.exp(planck_model_custom(freq, 1300., globals_['distance'], 12., R_bb=radius))
    flux[(wave >= 2.8) & (wave <= 3.3)] *= .001
    frame = pd.DataFrame(dict(JD=2450000.5, Lambda=wave, Flux=flux, Fluxerr=.02*flux))
    if long_continuum:
        k_only = frame[frame.Lambda <= 2.48].assign(JD=2450001.5)
        frame = pd.concat([frame, k_only], ignore_index=True)
    result = ultimate_fitting_regularized((), (), df=frame, data_mode='spectral',
        red_excess_mode=True, lambda_reg=0, initial_params=globals_,
        spectral_rede_max_blue_bb_fraction=0, expand_bounds=True)
    if not long_continuum:
        assert result is None
        return
    assert result['success']
    assert len(result['daily_params']) == 1
    assert result['fit_info']['excluded_red_epochs'] == [2450001.5]
    fitted = next(iter(result['daily_params'].values()))
    assert fitted['BB_identifiable']
    np.testing.assert_allclose([fitted['T_bb'], fitted['R_bb']], [1300., radius], rtol=1e-4)
    selected = next(iter(result['daily_data'].values()))['Lambda'].to_numpy()
    assert 2.0 in selected and 2.48 in selected
    assert not np.any((selected >= 2.8) & (selected <= 3.3))
    assert 1.9 not in selected
    assert result['fit_info']['spectral_red_mask_regions_micron'] == [[2.8, 3.3]]


@pytest.mark.parametrize('red', [[2.0, 2.2, 2.8, 3.0, 3.3], [3.5, 3.8, 4.0]])
@pytest.mark.parametrize('expand_bounds', [False, True])
def test_spectral_red_fit_requires_both_k_and_unmasked_l(red, expand_bounds):
    wave = [1.2, 1.5, 1.6] + red
    frame = pd.DataFrame(dict(JD=2450000.5, Lambda=wave, Flux=1., Fluxerr=.1))
    assert ultimate_fitting_regularized((), (), df=frame, data_mode='spectral',
        red_excess_mode=True, lambda_reg=0, expand_bounds=expand_bounds) is None


def test_photometric_joint_fit_recovers_temperature_and_radius():
    wave=np.array([1.235,1.662,2.159,3.5])
    freq=constants.c/(wave*1e-6)
    mdot=1.2e-4*M_sun/year
    mass=.5*M_sun; rstar=3*6.957e8; rout=2*constants.au; distance=700*constants.parsec
    av=12.; temperature=1300.; radius=1.5*constants.au
    acc=np.exp(accretion_model(freq,mdot,mass,rstar,rout,rstar,distance,av))
    bb=np.exp(planck_model_custom(freq,temperature,distance,av,R_bb=radius))
    flux=acc+np.where(wave>1.7,bb,0.)
    frame=pd.DataFrame({'JD':2450000.5,'Date':'1995-01-01','Filter':['J','H','K','L'],'Lambda':wave,'Flux':flux,'Fluxerr':.02*flux,'Mag':-2.5*np.log10(flux),'Magerr':.02,'ZP':1.})
    result=ultimate_fitting_regularized(('J','H','K','L'),('J','H','K','L'),df=frame,red_excess_mode=True,red_excess_fit_param='both',lambda_reg=0)
    assert result['success']
    fitted=next(iter(result['daily_params'].values()))
    np.testing.assert_allclose([fitted['T_bb'],fitted['R_bb']],[temperature,radius],rtol=1e-4)
    assert result['fit_info']['param_names']==['logMdot','Av','T_bb','R_bb']
    assert result['fit_info']['dof']==0
    assert np.isfinite(fitted['T_bb_err']) and fitted['T_bb_err']>0


def test_cold_planck_red_continuum_survives_faint_blue_tail():
    frequencies=constants.c/(np.array([.4,3.5])*1e-6)
    log_flux=planck_model_custom(frequencies,150.,700*constants.parsec,R_bb=2*constants.au)
    assert np.isfinite(log_flux).all()
    assert log_flux[0] < -100
    assert log_flux[1] > -50
    doubled=planck_model_custom(frequencies,150.,700*constants.parsec,R_bb=4*constants.au)
    np.testing.assert_allclose(doubled-log_flux,np.log(4.))
