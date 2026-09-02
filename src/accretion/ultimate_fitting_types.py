"""Convenience wrappers for the different fitting modes."""

from __future__ import annotations

try:
    from .ultimate_common import *
except ImportError:
    from ultimate_common import *

try:
    from .ultimate_fitting import *
except ImportError:
    from ultimate_fitting import *

try:
    from .ultimate_visualisation import *
except ImportError:
    from ultimate_visualisation import *


def basic_fitting(df, lambda_reg=0, debug=False, required_filters=('J', 'H', 'K'),
        fit_filters=('J', 'H', 'K')):

    print(f'\n=== Optimized Regularized Fitting (λ={lambda_reg}) ===')

    start_time = time.time()

    results = ultimate_fitting_regularized(
        required_filters=required_filters,
        fit_filters=fit_filters,
        AR_mode=False,
        red_excess_mode=False,
        df=df,
        debug=debug,
        lambda_reg=lambda_reg,
        regularize_params=['logMdot', 'Av']
    )

    end_time = time.time()
    execution_time = end_time - start_time

    print(f"\nTotal execution time: {execution_time:.1f} seconds")

    if results and results['success']:
        plot_results_regularized(results, df)
        print('And it was a success')
    return results


def allpams_fitting(df, lambda_reg=0, debug=False):

    start_time = time.time()
    results = ultimate_fitting_regularized(
        required_filters=('J', 'H', 'K'),
        fit_filters=('J', 'H', 'K'),
        AR_mode=False,
        red_excess_mode=False,
        df=df,
        debug=debug,
        lambda_reg=lambda_reg,
        regularize_params=['logMdot', 'Av'],
        global_params=['R_star', 'R_out','distance']
    )

    end_time = time.time()
    execution_time = end_time - start_time

    print(f"\nTotal execution time: {execution_time:.1f} seconds")

    if results and results['success']:
        plot_results_regularized(results, df)
        print('And it was a success')
    return results


def AR_fitting(df, filename, required_filters=('J', 'H', 'K'), fit_filters=('J', 'H', 'K'), debug=False, lambda_reg=0):
    """Run photometric AR fitting with a 3-D MCMC posterior."""

    print("Running AR mode fitting...")

    results = ultimate_fitting_regularized(
        required_filters=required_filters,
        fit_filters=fit_filters,
        AR_mode=True,
        red_excess_mode=False,
        df=df,
        debug=debug,
        lambda_reg=lambda_reg
    )

    if results and results['success']:
        print("AR mode fitting completed successfully!")
        visualize_results(results, df, filename)
        return results
    else:
        print("AR mode fitting failed!")
        return None


def rede_fitting(df, required_filters, fit_filters, filename, debug=False, lambda_reg=0,
                 red_excess_fit_param='T_bb', fixed_T_bb=RED_EXCESS_DEFAULT_T_BB,
                 fixed_R_bb=RED_EXCESS_DEFAULT_R_BB, fit_param=None):
    if fit_param is not None:
        red_excess_fit_param = fit_param

    print("Running red excess mode fitting...")

    results = ultimate_fitting_regularized(
        required_filters=required_filters,
        fit_filters=fit_filters,
        AR_mode=False,
        red_excess_mode=True,
        df=df,
        debug=debug,
        lambda_reg=lambda_reg,
        regularize_params=['logMdot', 'Av', _normalize_red_excess_fit_param(red_excess_fit_param)],
        red_excess_fit_param=red_excess_fit_param,
        fixed_T_bb=fixed_T_bb,
        fixed_R_bb=fixed_R_bb
    )

    if results and results['success']:
        print("Red excess mode fitting completed successfully!")
        visualize_results(results, df, filename)
        return results

    print("Red excess mode fitting failed!")
    return None


def spectral_fitting(df, lambda_reg=0, debug=False, save_dir='results/generated/pictures/spectral_mode', rede=False,
                     joint_refine=True, max_blue_bb_fraction=SPECTRAL_REDE_MAX_BLUE_BB_FRACTION,
                     ar_mode=False):
    mode_label = 'Spectral Red-Excess Accretion + BB Fitting' if rede else 'Spectral Accretion Fitting'
    if ar_mode:
        mode_label += ' + AR MCMC'
    print(f'\n=== {mode_label} (λ={lambda_reg}) ===')

    start_time = time.time()
    results = ultimate_fitting_regularized(
        required_filters=(),
        fit_filters=(),
        AR_mode=ar_mode,
        red_excess_mode=rede,
        df=df,
        debug=debug,
        lambda_reg=lambda_reg,
        regularize_params=['logMdot', 'Av', 'T_bb', 'R_bb'] if rede else ['logMdot', 'Av'],
        data_mode='spectral',
        spectral_rede_joint_refine=joint_refine,
        spectral_rede_max_blue_bb_fraction=max_blue_bb_fraction,
    )

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"\nTotal execution time: {execution_time:.1f} seconds")

    if results and results['success']:
        if ar_mode:
            plot_ar_surface(results, save_dir)
        visualize_spectral_results(results, df, save_dir=save_dir)
        print('And it was a success')
    return results

__all__ = ['basic_fitting', 'allpams_fitting', 'AR_fitting', 'rede_fitting', 'spectral_fitting']
