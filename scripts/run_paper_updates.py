#!/usr/bin/env python3
"""Reproduce the missing spectral/2026 fits and paper-summary numbers."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from accretion import (  # noqa: E402
    calibrate_spectra_to_photometry,
    load_database_new,
    load_irtf_lxd_spectra,
    load_irtf_sxd_spectra,
    load_xshooter_nir_spectra,
    result_opener,
    result_saver,
    spectral_fitting,
    ultimate_fitting_regularized,
)
from accretion.final_panels import (  # noqa: E402
    fit_parameter_table,
    integrate_accretion_history,
    print_final_panel_results,
)


def _fit_standard_spectrum(frame, output_root, picture_dir, make_plots):
    if make_plots:
        results = spectral_fitting(
            frame,
            lambda_reg=0,
            debug=False,
            save_dir=str(picture_dir),
        )
    else:
        results = ultimate_fitting_regularized(
            (),
            (),
            df=frame,
            lambda_reg=0,
            regularize_params=['logMdot', 'Av'],
            data_mode='spectral',
        )
    result_saver(results, str(output_root))
    return results


def _merge_photometric_results(baseline, update):
    merged = dict(baseline)
    merged['daily_params'] = dict(baseline.get('daily_params', {}))
    merged['daily_params'].update(update.get('daily_params', {}))
    merged['daily_data'] = dict(baseline.get('daily_data', {}))
    merged['daily_data'].update(update.get('daily_data', {}))
    merged['fit_info'] = dict(baseline.get('fit_info', {}))
    merged['fit_info']['paper_update'] = '2026 infrared epochs replaced/appended from database_daily_2603.txt'
    merged['fit_info']['n_days'] = len(merged['daily_params'])
    return merged


def _without_epoch(results, target_jd, tolerance_days=1.0):
    filtered = dict(results)
    filtered['daily_params'] = {
        jd: params
        for jd, params in results.get('daily_params', {}).items()
        if abs(float(jd) - float(target_jd)) > tolerance_days
    }
    filtered['daily_data'] = {
        jd: frame
        for jd, frame in results.get('daily_data', {}).items()
        if abs(float(jd) - float(target_jd)) > tolerance_days
    }
    return filtered


def _merge_result_sets(*result_sets, label='combined result bundles'):
    """Combine non-overlapping fitted epochs for plotting/reporting."""
    merged = {
        'success': True,
        'daily_params': {},
        'daily_data': {},
        'global_params': {},
        'param_errors': {},
        'fit_info': {
            'data_mode': 'spectral',
            'AR_mode': False,
            'red_excess_mode': False,
            'combined_sources': label,
        },
        'chi_squared': float('nan'),
        'reduced_chi_squared': float('nan'),
        'regularization_term': float('nan'),
        'total_objective': float('nan'),
    }
    for results in result_sets:
        merged['daily_params'].update(results.get('daily_params', {}))
        merged['daily_data'].update(results.get('daily_data', {}))
    merged['fit_info']['n_days'] = len(merged['daily_params'])
    return merged


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-plots', action='store_true', help='Skip spectral diagnostic figures.')
    parser.add_argument('--sample-stride', type=int, default=1, help='Keep every Nth binned XSHOOTER sample.')
    args = parser.parse_args()

    private_data = PROJECT_ROOT / '_private' / 'data' / 'raw'
    output_dir = PROJECT_ROOT / '_private' / 'results' / 'paper_updates'
    picture_dir = PROJECT_ROOT / '_private' / 'figures' / 'paper_updates'
    output_dir.mkdir(parents=True, exist_ok=True)
    picture_dir.mkdir(parents=True, exist_ok=True)

    photometry = load_database_new(private_data / 'photometry' / 'database_daily_2603.txt')

    # XSHOOTER J/H continuum: convert F_lambda -> Jy, robustly bin the
    # oversampled pixels, remove the unstable 1.4 micron atmospheric band, and
    # correct the absolute scale/colour with simultaneous J/H photometry.  The
    # raw 2024 spectra are about 35--40 per cent faint and 15--20 per cent too
    # red between J and H, which otherwise drives Av high.
    xshooter_df = load_xshooter_nir_spectra(
        private_data / 'Xshooter',
        wavelength_range=(1.15e-6, 1.75e-6),
        sample_stride=args.sample_stride,
    )
    xshooter_df = calibrate_spectra_to_photometry(
        xshooter_df,
        photometry,
        filters=('J', 'H'),
    )
    xshooter_results = _fit_standard_spectrum(
        xshooter_df,
        output_dir / 'xshooter_nir_1_2um',
        picture_dir / 'xshooter_nir_1_2um',
        not args.no_plots,
    )

    # Missing 2015 IRTF epoch: use both supplied files in JHK (1.2--2.2 um)
    # and explicitly override both normalization factors to unity.
    irtf_dir = private_data / 'IRTF'
    sxd = load_irtf_sxd_spectra(
        irtf_dir,
        normalization_factors={'20150722': 1.0},
    )
    lxd = load_irtf_lxd_spectra(
        irtf_dir,
        mask_regions=[(1.82e-6, 1.95e-6), (2.45e-6, 2.9e-6), (4.15e-6, 4.6e-6)],
        normalization_factors={'20150722': 1.0},
    )
    irtf_2015 = pd.concat([
        sxd[(sxd['DateCode'] == '20150722') & sxd['Lambda'].between(1.2, 2.2)],
        lxd[(lxd['DateCode'] == '20150722') & lxd['Lambda'].between(1.2, 2.2)],
    ], ignore_index=True)
    irtf_2015_results = _fit_standard_spectrum(
        irtf_2015,
        output_dir / 'irtf_20150722_sxd_lxd_jhk',
        picture_dir / 'irtf_20150722_sxd_lxd_jhk',
        not args.no_plots,
    )

    # The final JH panels must use the IRTF JH fits.  Using the JHK bundle here
    # folds the K-band excess into a two-parameter disk fit and systematically
    # raises both Mstar*Mdot and Av relative to the J/H photometric curve.
    existing_irtf = result_opener(
        str(PROJECT_ROOT / '_private' / 'results' / 'legacy' / 'spectral_JH')
    )
    existing_irtf_jhk_table2 = _without_epoch(
        result_opener(
            str(PROJECT_ROOT / '_private' / 'results' / 'legacy' / 'spectral_JHK')
        ),
        2457225.75,
    )
    spectral_all_epochs = _merge_result_sets(
        existing_irtf,
        xshooter_results,
        label='IRTF JH + XSHOOTER JH',
    )
    result_saver(spectral_all_epochs, str(output_dir / 'spectral_all_updated_epochs'))

    # New 2026 infrared epochs.  Restricting the input keeps the update independent
    # of the older high-dimensional fit while using the identical disk model,
    # bounds, uncertainty floor, and lambda=0 objective.
    photometry_2026 = photometry[photometry['JD'] >= 2461000].copy()
    photo_2026_jh_results = ultimate_fitting_regularized(
        ('J', 'H'),
        ('J', 'H'),
        df=photometry_2026,
        lambda_reg=0,
        regularize_params=['logMdot', 'Av'],
        data_mode='photometry',
    )
    result_saver(photo_2026_jh_results, str(output_dir / 'photometry_2026_jh'))

    photo_2026_jhk_results = ultimate_fitting_regularized(
        ('J', 'H', 'K'),
        ('J', 'H', 'K'),
        df=photometry_2026,
        lambda_reg=0,
        regularize_params=['logMdot', 'Av'],
        data_mode='photometry',
    )
    result_saver(photo_2026_jhk_results, str(output_dir / 'photometry_2026_jhk'))

    baseline_jh = result_opener(
        str(PROJECT_ROOT / '_private' / 'results' / 'legacy' / 'basic_JH_newdf')
    )
    combined_photo_jh = _merge_photometric_results(baseline_jh, photo_2026_jh_results)
    result_saver(combined_photo_jh, str(output_dir / 'photometry_jh_with_2026'))

    baseline_jhk = result_opener(
        str(PROJECT_ROOT / '_private' / 'results' / 'legacy' / 'basic_JHK_newdf')
    )
    combined_photo_jhk = _merge_photometric_results(baseline_jhk, photo_2026_jhk_results)
    result_saver(combined_photo_jhk, str(output_dir / 'photometry_jhk_with_2026'))

    # The first estimate bridges every fitted epoch linearly.  The second is a
    # coverage-limited check that does not assign mass to gaps longer than one
    # year, making the role of sparse historical coverage explicit.
    all_gaps = integrate_accretion_history(combined_photo_jh)
    one_year_gaps = integrate_accretion_history(combined_photo_jh, max_gap_days=365.25)
    jhk_all_gaps = integrate_accretion_history(combined_photo_jhk)

    spectral_sets = {
        'IRTF JHK (existing epochs)': existing_irtf_jhk_table2,
        'IRTF 2015 SXD+LXD, no scaling': irtf_2015_results,
        'XSHOOTER': xshooter_results,
    }
    report = print_final_panel_results(
        spectral_result_sets=spectral_sets,
        photometric_result_sets={
            '2026 JH (Fig. 7 method)': photo_2026_jh_results,
            '2026 JHK cross-check': photo_2026_jhk_results,
        },
        integration_results={
            'JH piecewise-linear full interval': all_gaps,
            'JH coverage-limited check': one_year_gaps,
            'JHK piecewise-linear sensitivity': jhk_all_gaps,
        },
    )
    report.to_csv(output_dir / 'table2_and_2026_fit_parameters.csv', index=False)
    pd.DataFrame([
        {'Estimate': 'JH piecewise-linear full interval', **all_gaps},
        {'Estimate': 'JH coverage-limited check', **one_year_gaps},
        {'Estimate': 'JHK piecewise-linear sensitivity', **jhk_all_gaps},
    ]).to_json(output_dir / 'accreted_mass_estimates.json', orient='records', indent=2)

    # A smaller table with the unique 2015 replacement value is useful when
    # copying numbers into Table 2 without accidentally selecting the old row.
    fit_parameter_table(
        {'IRTF 2015 SXD+LXD, no scaling': irtf_2015_results}
    ).to_csv(output_dir / 'irtf_2015_table2_row.csv', index=False)
    print(f'\nPaper-update outputs saved under {output_dir}')


if __name__ == '__main__':
    main()
