"""Reusable saved-result plotting and bound policies, without scientific refits."""
import copy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from accretion import ultimate_fitting as fitting
from accretion import ultimate_visualisation as plots
from accretion import final_panels as panels
from accretion.final_tables import build_final_tables
from accretion.ultimate_common import M_sun, constants


def result(jd=2450000.5, identifiable=True):
    return {'success':True,'global_params':{'M':.5*M_sun},
            'daily_params':{jd:{'Mdot':1e-5,'Mdot_err':1e-6,'Av':10.,'Av_err':.2,
                              'T_bb':800.,'T_bb_err':30.,'R_bb':2*constants.au,'R_bb_err':.1*constants.au,
                              'BB_identifiable':identifiable}},
            'daily_data':{jd:pd.DataFrame({'Date':['1995-10-10'],'Lambda':[1.2], 'Instrument':['IRTF/SpeX']})},
            'fit_info':{'param_names':['logMdot','Av','T_bb','R_bb'],'n_days':1,'n_data_points':4,'dof':0},
            'total_objective':1.,'chi_squared':1.,'regularization_term':0.}


def test_bound_retry_expands_only_hit_side_and_keeps_attempts(monkeypatch):
    bounds=[]
    def solve(*args,**kwargs):
        bounds.append(kwargs['bb_temperature_bounds'])
        r=result();next(iter(r['daily_params'].values()))['T_bb']=650. if len(bounds)==1 else 600.
        return r
    monkeypatch.setattr(fitting,'ultimate_fitting_regularized',solve)
    fitted=fitting.fit_with_expanding_bounds(('J','H'),('J','H'),pd.DataFrame(),
        lambda_reg=1,red_excess_mode=True,bb_temperature_bounds=(650.,10000.))
    assert bounds==[(650.,10000.),(325.,10000.)]
    assert len(fitted['fit_info']['epoch_diagnostics'][0]['info']['bound_attempts'])==2
    assert next(iter(fitted['daily_params'].values()))['BB_identifiable']


def test_flat_bb_boundary_preserves_candidate_but_omits_unconstrained_value(monkeypatch):
    def solve(*args,**kwargs):
        r=result();next(iter(r['daily_params'].values()))['T_bb']=kwargs['bb_temperature_bounds'][0]
        return r
    monkeypatch.setattr(fitting,'ultimate_fitting_regularized',solve)
    fitted=fitting.fit_with_expanding_bounds(('J','H'),('J','H'),pd.DataFrame(),
        lambda_reg=1,red_excess_mode=True,bb_temperature_bounds=(650.,10000.))
    params=next(iter(fitted['daily_params'].values()))
    assert params['BB_constraint_status']=='boundary asymptote'
    assert not params['BB_identifiable'] and np.isnan(params['T_bb'])
    assert np.isfinite(params['T_bb_candidate'])


def test_correlations_share_time_scale_preserve_families_and_keep_figures_open():
    a=result();b=result(2460000.5);invalid=result(2450010.5,False)
    a['daily_params'].update(invalid['daily_params'])
    before=copy.deepcopy(a)
    fig,axes,frame=plots.plot_red_excess_correlations({'photo':a,'spectral':b},show=False)
    assert frame.family.tolist()==['photo','spectral']
    assert axes.shape==(2,2) and plt.fignum_exists(fig.number)
    assert a['daily_params']==before['daily_params']
    assert all(ax.get_xscale()=='log' for ax in axes[:,0])
    plt.close(fig)


def test_csv_style_missing_nan_parameters_does_not_break_evolution():
    r=result();next(iter(r['daily_params'].values())).pop('T_bb')
    fig,axes=plots.plot_red_excess_evolution(r,show=False)
    assert plt.fignum_exists(fig.number) and axes.shape==(4,1)
    plt.close(fig)


def test_residuals_keep_actual_wavelengths_and_direct_measurement_interpolation(monkeypatch):
    data=pd.DataFrame({'JD':[1.,1.,2.], 'Filter':['L']*3,'Lambda':[3.5,3.8,3.5],
        'ZP':[100.]*3,'Flux':[2.]*3,'Fluxerr':[.1]*3,'Mag':[1.]*3,'Magerr':[.1]*3,
        'Interpolated':['yes','no','no']})
    monkeypatch.setattr(panels,'calculate_filter_accretion_models',lambda *args:{'L':data[['JD','Filter','Lambda','ZP']].assign(
        Mag_model=[5.,4.,5.],Mag_model_err=.1,Is_interpolated_fit=[False,False,True])})
    out=panels.prepare_final_panel_residuals(data,{'daily_params':{}},('J','H'),('L+W1',))['flux']
    assert out.Lambda.tolist()==[3.5,3.8] and out.Interpolated.tolist()==['yes','no']
    assert out.Flux_synth.iloc[0] != out.Flux_synth.iloc[1]


def test_tables_reuse_supplied_masses_and_filter_only_photometry(tmp_path):
    photo=result(2461001.5);spectral=result(2450000.5)
    tables=build_final_tables(photo,spectral,integrated_masses=[{'mass_accreted_Msun':.123}],save_dir=tmp_path)
    assert tables['Table_2'].Kind.tolist()==['spectrum','photometry']
    assert tables['JH_radius_integrated_mass'].mass_accreted_Msun.iloc[0]==.123
    assert (tmp_path/'Table_2.csv').exists() and (tmp_path/'Table_2.tex').exists()
