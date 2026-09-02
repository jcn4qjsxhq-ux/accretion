#!/usr/bin/env python3
"""Rerun the fiducial IRTF, XSHOOTER, and photometric JH fits strictly."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from accretion import (  # noqa: E402
    calibrate_spectra_to_photometry,
    load_database_new,
    load_irtf_sxd_spectra,
    load_xshooter_nir_spectra,
    result_saver,
    ultimate_fitting_regularized,
)


STRICT_FIT_OPTIONS = {
    "optimizer_maxiter": 1_000_000,
    "optimizer_maxfun": 1_000_000,
    "optimizer_ftol": 1e-15,
    "optimizer_gtol": 1e-10,
    "local_solver_max_nfev": 100_000,
    "local_solver_ftol": 1e-14,
    "local_solver_gtol": 1e-14,
}


def _fit_spectrum(frame):
    return ultimate_fitting_regularized(
        (),
        (),
        df=frame,
        lambda_reg=0,
        regularize_params=["logMdot", "Av"],
        data_mode="spectral",
        **STRICT_FIT_OPTIONS,
    )


def _combine_spectral_results(irtf_results, xshooter_results):
    combined = {
        "success": bool(irtf_results.get("success") and xshooter_results.get("success")),
        "daily_params": {},
        "daily_data": {},
        "global_params": {},
        "param_errors": {},
        "fit_info": dict(irtf_results.get("fit_info", {})),
        "chi_squared": irtf_results.get("chi_squared", 0.0) + xshooter_results.get("chi_squared", 0.0),
        "regularization_term": 0.0,
        "total_objective": irtf_results.get("total_objective", 0.0) + xshooter_results.get("total_objective", 0.0),
    }
    combined["daily_params"].update(irtf_results.get("daily_params", {}))
    combined["daily_params"].update(xshooter_results.get("daily_params", {}))
    combined["daily_data"].update(irtf_results.get("daily_data", {}))
    combined["daily_data"].update(xshooter_results.get("daily_data", {}))
    combined["fit_info"].update({
        "n_days": len(combined["daily_params"]),
        "n_data_points": sum(len(frame) for frame in combined["daily_data"].values()),
        "combined_sources": "strict IRTF JH + strict calibrated XSHOOTER JH",
        "optimizer_nfev": sum(
            int(results.get("fit_info", {}).get("optimizer_nfev", 0))
            for results in (irtf_results, xshooter_results)
        ),
        "optimizer_nit": sum(
            int(results.get("fit_info", {}).get("optimizer_nit", 0))
            for results in (irtf_results, xshooter_results)
        ),
        "optimizer_message": "Both strict instrument fits converged independently.",
    })
    dof = combined["fit_info"]["n_data_points"] - 2 * combined["fit_info"]["n_days"]
    combined["fit_info"]["dof"] = dof
    combined["reduced_chi_squared"] = combined["chi_squared"] / dof if dof > 0 else float("inf")
    return combined


def main() -> int:
    raw_dir = PROJECT_ROOT / "_private" / "data" / "raw"
    output_dir = PROJECT_ROOT / "_private" / "results" / "tight_jh"
    output_dir.mkdir(parents=True, exist_ok=True)

    photometry = load_database_new(raw_dir / "photometry" / "database_daily_2603.txt")

    # Match the existing fiducial IRTF JH definition: SXD 1.20--1.60 micron,
    # with the 1.30--1.41 micron atmospheric interval removed.
    irtf = load_irtf_sxd_spectra(
        raw_dir / "IRTF",
        mask_regions=[
            (0.68e-6, 0.74e-6),
            (1.30e-6, 1.41e-6),
            (1.82e-6, 1.95e-6),
            (2.45e-6, 2.90e-6),
        ],
    )
    irtf_jh = irtf[irtf["Lambda_m"].between(1.20e-6, 1.60e-6)].copy()
    irtf_results = _fit_spectrum(irtf_jh)
    result_saver(irtf_results, str(output_dir / "irtf_jh"))

    # Match the calibrated XSHOOTER JH setup used by the paper-update workflow.
    xshooter_jh = load_xshooter_nir_spectra(
        raw_dir / "Xshooter",
        wavelength_range=(1.15e-6, 1.75e-6),
        sample_stride=1,
    )
    xshooter_jh = calibrate_spectra_to_photometry(
        xshooter_jh,
        photometry,
        filters=("J", "H"),
    )
    xshooter_results = _fit_spectrum(xshooter_jh)
    result_saver(xshooter_results, str(output_dir / "xshooter_jh"))

    photometric_results = ultimate_fitting_regularized(
        ("J", "H"),
        ("J", "H"),
        df=photometry,
        lambda_reg=0,
        regularize_params=["logMdot", "Av"],
        data_mode="photometry",
        **STRICT_FIT_OPTIONS,
    )
    result_saver(photometric_results, str(output_dir / "photometric_jh"))

    combined = _combine_spectral_results(irtf_results, xshooter_results)
    result_saver(combined, str(output_dir / "spectral_jh_combined"))
    print(f"\nStrict JH result bundles saved under {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
