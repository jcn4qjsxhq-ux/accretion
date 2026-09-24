#!/usr/bin/env python3
"""Fit and plot the JH model for fixed 2 and 4 R_sun stellar radii."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

os.environ.setdefault("MPLBACKEND", "Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from accretion.final_panels import (  # noqa: E402
    integrate_accretion_history,
    plot_final_panels_0_to_4_layouts,
)
from accretion.ultimate_common import M_sun, R_sun, constants  # noqa: E402
from accretion.ultimate_file_organisers import (  # noqa: E402
    calibrate_spectra_to_photometry,
    load_database_new,
    load_irtf_sxd_spectra,
    load_xshooter_nir_spectra,
    result_saver,
)
from accretion.ultimate_fitting import ultimate_fitting_regularized  # noqa: E402


DATABASE_PATH = PROJECT_ROOT / "_private/data/raw/photometry/database_daily_2609.txt"
RAW_DIR = PROJECT_ROOT / "_private/data/raw"
RESULTS_DIR = PROJECT_ROOT / "_private/results/jh_radius_variants"
PUBLIC_DATA_DIR = PROJECT_ROOT / "results/generated/datas"
PICTURES_DIR = PROJECT_ROOT / "results/generated/pictures"
SUMMARY_CSV = PROJECT_ROOT / "results/generated/datas/JH_radius_integrated_mass.csv"
SUMMARY_JSON = PROJECT_ROOT / "results/generated/datas/JH_radius_integrated_mass.json"

STRICT_FIT_OPTIONS = {
    "optimizer_maxiter": 1_000_000,
    "optimizer_maxfun": 1_000_000,
    "optimizer_ftol": 1e-15,
    "optimizer_gtol": 1e-10,
    "local_solver_max_nfev": 100_000,
    "local_solver_ftol": 1e-14,
    "local_solver_gtol": 1e-14,
}


def _radius_params(radius_solar: float) -> dict[str, float]:
    radius = float(radius_solar) * R_sun
    return {
        "M": 0.5 * M_sun,
        "R_star": radius,
        "R_in": radius,
        "R_out": 2.0 * constants.au,
        "distance": 700.0 * constants.parsec,
    }


def _fit(
    frame,
    *,
    mode: str,
    initial_params: dict[str, float],
    av_bounds=(0.1, 40.0),
    # Permit the deep 2010 minimum to be determined by the data rather than
    # the historical 1e-7 Msun/yr optimizer floor.  For M_star=0.5 Msun this
    # corresponds to M_star*Mdot >= 5e-10 Msun^2/yr.
    mdot_bounds_msun_per_year=(1e-9, 2e-3),
):
    required = ("J", "H") if mode == "photometry" else ()
    results = ultimate_fitting_regularized(
        required,
        required,
        df=frame,
        lambda_reg=0,
        regularize_params=["logMdot", "Av"],
        data_mode=mode,
        initial_params=initial_params,
        mdot_bounds_msun_per_year=mdot_bounds_msun_per_year,
        av_bounds=av_bounds,
        **STRICT_FIT_OPTIONS,
    )
    if not results or not results.get("success"):
        raise RuntimeError(f"The {mode} JH fit did not converge: {results}")

    av_values = [float(params["Av"]) for params in results["daily_params"].values()]
    if av_values and max(av_values) >= av_bounds[1] - 1e-5:
        expanded_bounds = (av_bounds[0], max(80.0, 2.0 * av_bounds[1]))
        print(
            f"A_V reached {av_bounds[1]:g} mag in the {mode} fit; "
            f"rerunning with bounds {expanded_bounds}."
        )
        return _fit(
            frame,
            mode=mode,
            initial_params=initial_params,
            av_bounds=expanded_bounds,
            mdot_bounds_msun_per_year=mdot_bounds_msun_per_year,
        )
    return results


def _combine_spectral_results(irtf_results, xshooter_results, radius_solar):
    combined = {
        "success": bool(irtf_results.get("success") and xshooter_results.get("success")),
        "daily_params": {},
        "daily_data": {},
        "global_params": dict(irtf_results.get("global_params", {})),
        "param_errors": {},
        "fit_info": dict(irtf_results.get("fit_info", {})),
        "chi_squared": float(irtf_results.get("chi_squared", 0.0))
        + float(xshooter_results.get("chi_squared", 0.0)),
        "regularization_term": 0.0,
        "total_objective": float(irtf_results.get("total_objective", 0.0))
        + float(xshooter_results.get("total_objective", 0.0)),
    }
    combined["daily_params"].update(irtf_results.get("daily_params", {}))
    combined["daily_params"].update(xshooter_results.get("daily_params", {}))
    combined["daily_data"].update(irtf_results.get("daily_data", {}))
    combined["daily_data"].update(xshooter_results.get("daily_data", {}))
    combined["fit_info"].update(
        {
            "n_days": len(combined["daily_params"]),
            "n_data_points": sum(len(frame) for frame in combined["daily_data"].values()),
            "combined_sources": "IRTF/SpeX JH + calibrated XSHOOTER JH",
            "fixed_radius_solar": float(radius_solar),
        }
    )
    dof = combined["fit_info"]["n_data_points"] - 2 * combined["fit_info"]["n_days"]
    combined["fit_info"]["dof"] = dof
    combined["reduced_chi_squared"] = combined["chi_squared"] / dof if dof > 0 else float("inf")
    return combined


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--radii",
        nargs="+",
        type=float,
        default=(2.0, 4.0),
        help="Stellar/inner radii to fit, in solar radii (default: 2 4).",
    )
    args = parser.parse_args()
    selected_radii = tuple(dict.fromkeys(float(value) for value in args.radii))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PICTURES_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)

    photometry = load_database_new(DATABASE_PATH)
    irtf = load_irtf_sxd_spectra(
        RAW_DIR / "IRTF",
        mask_regions=[
            (0.68e-6, 0.74e-6),
            (1.30e-6, 1.41e-6),
            (1.82e-6, 1.95e-6),
            (2.45e-6, 2.90e-6),
        ],
    )
    irtf_jh = irtf[irtf["Lambda_m"].between(1.20e-6, 1.60e-6)].copy()

    xshooter_jh = load_xshooter_nir_spectra(
        RAW_DIR / "Xshooter",
        wavelength_range=(1.15e-6, 1.75e-6),
        sample_stride=1,
    )
    xshooter_jh = calibrate_spectra_to_photometry(
        xshooter_jh,
        photometry,
        filters=("J", "H"),
    )

    summary_rows = []
    for radius_solar in selected_radii:
        label = f"{int(radius_solar)}R"
        output_dir = RESULTS_DIR / label
        output_dir.mkdir(parents=True, exist_ok=True)
        initial_params = _radius_params(radius_solar)

        print(f"\n=== Running JH fits for R_star = R_in = {radius_solar:g} R_sun ===")
        photometric_results = _fit(
            photometry,
            mode="photometry",
            initial_params=initial_params,
        )
        irtf_results = _fit(irtf_jh, mode="spectral", initial_params=initial_params)
        xshooter_results = _fit(xshooter_jh, mode="spectral", initial_params=initial_params)
        spectral_results = _combine_spectral_results(irtf_results, xshooter_results, radius_solar)

        result_saver(photometric_results, str(output_dir / "photometric_jh"))
        result_saver(irtf_results, str(output_dir / "irtf_jh"))
        result_saver(xshooter_results, str(output_dir / "xshooter_jh"))
        result_saver(spectral_results, str(output_dir / "spectral_jh_combined"))
        # Keep the established public result-stem names used by the notebook.
        result_saver(
            photometric_results,
            str(PUBLIC_DATA_DIR / f"basic_JH_{label}_newdf"),
        )
        result_saver(
            spectral_results,
            str(PUBLIC_DATA_DIR / f"spectral_JH_{label}"),
        )

        figure_path = PICTURES_DIR / f"0-4_JH_{label}.pdf"
        plot_final_panels_0_to_4_layouts(
            photometry,
            photometric_results,
            spectral_results=spectral_results,
            default_params={
                "M_star": initial_params["M"],
                "R_star": initial_params["R_star"],
                "R_in": initial_params["R_in"],
                "R_out": initial_params["R_out"],
                "distance": initial_params["distance"],
            },
            selected_filters=("J", "H", "K"),
            photometric_fit_filters=("J", "H"),
            residual_filter_groups=("J", "H", "K", "L+W1"),
            exclude_interpolated_residuals=True,
            plot_panel0_fits=False,
            filter_diff=False,
            show_panel3=False,
            end_zoom_range=(2459000, 2461300),
            flux_residual_log_scale=False,
            one_column_save_path=None,
            two_column_save_path=figure_path,
            show=False,
        )

        mass = integrate_accretion_history(photometric_results)
        av_values = [float(params["Av"]) for params in photometric_results["daily_params"].values()]
        spectral_av_values = [float(params["Av"]) for params in spectral_results["daily_params"].values()]
        summary_rows.append(
            {
                "radius_Rsun": radius_solar,
                "R_star_Rsun": radius_solar,
                "R_in_Rsun": radius_solar,
                "mass_accreted_Msun": mass["mass_accreted_Msun"],
                "mass_accreted_err_Msun": mass["mass_accreted_err_Msun"],
                "start_jd": mass["start_jd"],
                "end_jd": mass["end_jd"],
                "span_years": mass["span_years"],
                "n_photometric_epochs": mass["n_epochs"],
                "photometric_Av_min": min(av_values),
                "photometric_Av_max": max(av_values),
                "spectroscopic_Av_min": min(spectral_av_values),
                "spectroscopic_Av_max": max(spectral_av_values),
                "photometric_Av_bounds": json.dumps(photometric_results["fit_info"]["av_bounds"]),
                "photometric_Mdot_bounds": json.dumps(
                    photometric_results["fit_info"]["mdot_bounds_msun_per_year"]
                ),
                "irtf_Av_bounds": json.dumps(irtf_results["fit_info"]["av_bounds"]),
                "xshooter_Av_bounds": json.dumps(xshooter_results["fit_info"]["av_bounds"]),
                "panel_0_4": str(figure_path),
            }
        )

    summary = pd.DataFrame(summary_rows)
    if SUMMARY_CSV.exists():
        previous = pd.read_csv(SUMMARY_CSV)
        previous = previous[
            ~pd.to_numeric(previous["radius_Rsun"], errors="coerce").isin(selected_radii)
        ]
        summary = pd.concat([previous, summary], ignore_index=True, sort=False)
    summary = summary.sort_values("radius_Rsun").reset_index(drop=True)
    summary.to_csv(SUMMARY_CSV, index=False)
    SUMMARY_JSON.write_text(summary.to_json(orient="records", indent=2) + "\n", encoding="utf-8")
    print(f"\nFinal 0-4 panels saved under {PICTURES_DIR}")
    print(f"Integrated masses saved to {SUMMARY_CSV}")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
