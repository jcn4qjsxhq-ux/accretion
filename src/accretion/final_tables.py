"""Paper tables and diagnostics from saved fits; no fitting or plotting side effects."""
from pathlib import Path
import numpy as np
import pandas as pd
from .ultimate_common import M_sun, constants, year
from .final_panels import fit_parameter_table
from .ultimate_fitting import _coerce_wavelength_meters


def _parameter_table(result):
    return fit_parameter_table(result, m_star_solar=result['global_params'].get('M', .5*M_sun)/M_sun)


def build_fit_diagnostics(results):
    """Report active bounds, actual fitted coverage and BB identifiability."""
    rows = []
    for label, result in results.items():
        for jd, params in sorted(result['daily_params'].items()):
            info = result.get('fit_info', {})
            for epoch in info.get('epoch_diagnostics', []):
                if jd in epoch['JD']: info = epoch['info']; break
            day = result.get('daily_data', {}).get(jd)
            wave = _coerce_wavelength_meters(day)*1e6 if day is not None else np.array([np.nan])
            hits = []
            for key, name, default, scale in [
                ('Mdot','mdot_bounds_msun_per_year',(1e-7,8e-4),1),
                ('Av','av_bounds',(.1,40),1),
                ('T_bb','bb_temperature_bounds',(650,10000),1),
                ('R_bb','bb_radius_bounds_au',(.05,20),constants.au),
            ]:
                value = params.get(key, np.nan)/scale
                bounds = info.get(name, default)
                if np.isfinite(value) and (value <= bounds[0]*(1+1e-3) or value >= bounds[1]*(1-1e-3)):
                    hits.append(key)
            blue_fraction = np.nan
            if info.get('red_excess_mode') and day is not None and info.get('data_mode')=='spectral':
                from .ultimate_physics import accretion_model, planck_model_custom
                blue = np.isfinite(wave) & (wave>=1.15) & (wave<=1.75)
                temperature = params.get('T_bb_candidate',params.get('T_bb',np.nan))
                radius = params.get('R_bb_candidate',params.get('R_bb',np.nan))
                if blue.any() and np.isfinite(temperature) and np.isfinite(radius):
                    g = result['global_params']; freq = constants.c/(wave[blue]*1e-6)
                    disk = accretion_model(freq,params['Mdot']*M_sun/year,g['M'],g['R_star'],g['R_out'],g['R_in'],g['distance'],params['Av'])
                    bb = planck_model_custom(freq,temperature,g['distance'],params['Av'],R_bb=radius)
                    blue_fraction = float(np.exp(bb-disk).max())
            rows.append(dict(fit=label, JD=jd, BB_constraint_status=params.get('BB_constraint_status','not applicable'),
                BB_identifiable=params.get('BB_identifiable',np.nan), bound_hits=','.join(hits),
                Av_upper=info.get('av_bounds',(.1,40))[1], wavelength_min_um=np.nanmin(wave),
                wavelength_max_um=np.nanmax(wave), max_blue_BB_over_disk=blue_fraction, no_coverage_beyond_2_2um=not np.any(wave>2.2)))
    return pd.DataFrame(rows)


def build_radius_comparison_tables(results_by_radius, baseline_radius=3):
    """Compare photo/spectral radius variants with the selected baseline."""
    comparisons, summaries = [], []
    for family in ('photo', 'spectral'):
        frames = {radius:_parameter_table(bundle[family]).set_index('JD') for radius,bundle in results_by_radius.items()}
        baseline = frames[baseline_radius]
        for radius, frame in frames.items():
            result = results_by_radius[radius][family]
            dof = result['fit_info'].get('dof',0)
            summaries.append(dict(family=family,radius_Rsun=radius,epochs=len(frame),
                median_MstarMdot=frame.MstarMdot_Msun2_per_yr.median(),min_MstarMdot=frame.MstarMdot_Msun2_per_yr.min(),
                max_MstarMdot=frame.MstarMdot_Msun2_per_yr.max(),median_Av=frame.Av_mag.median(),min_Av=frame.Av_mag.min(),
                max_Av=frame.Av_mag.max(),dof=dof,reduced_chi_squared=result.get('reduced_chi_squared',np.nan) if dof>0 else np.nan))
            if radius==baseline_radius: continue
            merged = frame.join(baseline,lsuffix='_variant',rsuffix='_baseline',how='outer')
            for jd,row in merged.iterrows():
                matched = pd.notna(row.Mdot_Msun_per_yr_variant) and pd.notna(row.Mdot_Msun_per_yr_baseline)
                ratio = row.MstarMdot_Msun2_per_yr_variant/row.MstarMdot_Msun2_per_yr_baseline if matched else np.nan
                comparisons.append(dict(family=family,radius_Rsun=radius,baseline_Rsun=baseline_radius,JD=jd,matched=matched,
                    MstarMdot_variant=row.MstarMdot_Msun2_per_yr_variant,**{f'MstarMdot_{baseline_radius}R':row.MstarMdot_Msun2_per_yr_baseline},
                    **{f'MstarMdot_ratio_to_{baseline_radius}R':ratio},MstarMdot_percent_change=100*(ratio-1),
                    Av_variant=row.Av_mag_variant,**{f'Av_{baseline_radius}R':row.Av_mag_baseline},Av_difference_mag=row.Av_mag_variant-row.Av_mag_baseline))
    comparison = pd.DataFrame(comparisons)
    differences = comparison.groupby(['family','radius_Rsun'],as_index=False).agg(
        matched_epochs=('matched','sum'),median_percent_change=('MstarMdot_percent_change','median'),median_Av_difference_mag=('Av_difference_mag','median')) if not comparison.empty else pd.DataFrame()
    return {'JH_radius_fit_parameter_comparison':comparison,'JH_radius_fit_summary':pd.DataFrame(summaries),
            f'JH_radius_differences_from_{baseline_radius}R':differences}


def build_final_tables(photo, spectral, red_results=None, radius_results=None,
                       integrated_masses=None, photometric_min_jd=2461000,
                       baseline_radius=3, save_dir=None):
    """Build Table 2 plus optional red/radius/mass tables from supplied results.

    ``integrated_masses`` is a table already computed with final_panels; mass
    integration is not duplicated here. ``save_dir=None`` returns tables only.
    Supplying an output directory writes CSV and LaTeX without deleting files.
    """
    spectra = _parameter_table(spectral)
    spectra['Dataset'] = [
        'XSHOOTER calibrated JH' if 'XSHOOTER' in ' '.join(spectral['daily_data'][jd].get('Instrument',pd.Series(dtype=str)).astype(str)).upper() else 'IRTF JH'
        for jd in spectra.JD]
    spectra.insert(0,'Kind','spectrum')
    photometry = _parameter_table(photo)
    if photometric_min_jd is not None: photometry = photometry[photometry.JD>=photometric_min_jd].copy()
    label = ''.join(photo.get('fit_info',{}).get('fit_filters') or ['J','H'])
    photometry['Dataset'] = ('2026 ' if photometric_min_jd==2461000 else 'Photometric ')+label
    photometry.insert(0,'Kind','photometry')
    tables = {'Table_2':pd.concat([spectra,photometry],ignore_index=True)}
    diagnostic_sets = {} if radius_results else {'JH_photo':photo,'JH_spectral':spectral}
    if red_results:
        for family, result in red_results.items():
            stem = 'rede_4L_T_parameters' if family=='photo' else 'spectral_red_excess_parameters'
            tables[stem] = pd.DataFrame.from_dict(result['daily_params'],orient='index').rename_axis('JD').reset_index()
            tables[f'{family}_red_fit_validation'] = tables[stem].copy()
            diagnostic_sets[f'red_{family}'] = result
        from .ultimate_visualisation import red_excess_parameter_table
        frame = red_excess_parameter_table(red_results)
        tables['red_excess_context_correlations'] = frame[frame.BB_identifiable & np.isfinite(frame[['T','R']]).all(axis=1)].drop(columns='BB_identifiable')
    if radius_results:
        tables.update(build_radius_comparison_tables(radius_results,baseline_radius))
        diagnostic_sets.update({f'JH_{r}R_{family}':bundle[family] for r,bundle in radius_results.items() for family in ('photo','spectral')})
    tables['fit_bound_and_coverage_diagnostics'] = build_fit_diagnostics(diagnostic_sets)
    covariance_rows = []
    for label,result in diagnostic_sets.items():
        info = result.get('fit_info',{})
        diagnostics = {row['JD']:row for row in info.get('uncertainty_diagnostics',[])}
        for epoch in info.get('epoch_diagnostics',[]):
            diagnostics.update({row['JD']:row for row in epoch['info'].get('uncertainty_diagnostics',[])})
        for diagnostic in diagnostics.values():
            covariance_rows.append({'fit':label, **diagnostic})
    if covariance_rows: tables['fit_uncertainty_diagnostics'] = pd.DataFrame(covariance_rows)
    if integrated_masses is not None:
        masses = pd.DataFrame(integrated_masses).drop(columns='excluded_gaps',errors='ignore')
        if radius_results and not masses.empty:
            baseline = masses[masses.radius_Rsun==baseline_radius].set_index(['family','method']).mass_accreted_Msun
            ratios = [row.mass_accreted_Msun/baseline.loc[(row.family,row.method)] for row in masses.itertuples()]
            masses[f'mass_ratio_to_{baseline_radius}R'] = ratios
            masses['mass_percent_change_from_'+str(baseline_radius)+'R'] = 100*(np.asarray(ratios)-1)
        tables['JH_radius_integrated_mass'] = masses
    if save_dir is not None:
        destination = Path(save_dir); destination.mkdir(parents=True,exist_ok=True)
        for stem, frame in tables.items():
            frame.to_csv(destination/f'{stem}.csv',index=False)
            (destination/f'{stem}.tex').write_text(frame.to_latex(index=False,float_format='%.6g'),encoding='utf-8')
        if radius_results:
            summary = tables['JH_radius_fit_summary']
            differences = tables[f'JH_radius_differences_from_{baseline_radius}R']
            (destination/'JH_radius_fit_comparison.tex').write_text(summary.to_latex(index=False,float_format='%.5g')+'\n'+differences.to_latex(index=False,float_format='%.5g'),encoding='utf-8')
        if integrated_masses is not None:
            tables['JH_radius_integrated_mass'].to_json(destination/'JH_radius_integrated_mass.json',orient='records',indent=2)
    return tables


__all__ = ['build_final_tables','build_radius_comparison_tables','build_fit_diagnostics']
