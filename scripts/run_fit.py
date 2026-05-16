"""Run one of the public accretion fitting modes from the command line."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import pandas as pd

from accretion import (
    AR_fitting,
    allpams_fitting,
    basic_fitting,
    load_database_new,
    load_irtf_lxd_spectra,
    load_irtf_sxd_spectra,
    rede_fitting,
    result_saver,
    spectral_fitting,
)


def _parse_filters(value: str) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(part.strip() for part in value.split(",") if part.strip())


def load_input(path: Path, data_kind: str) -> pd.DataFrame:
    if data_kind == "legacy-photometry":
        return load_database_new(str(path))
    if data_kind == "csv":
        return pd.read_csv(path)
    if data_kind == "irtf-sxd":
        return load_irtf_sxd_spectra(str(path))
    if data_kind == "irtf-lxd":
        return load_irtf_lxd_spectra(str(path))
    raise ValueError(f"Unknown data kind: {data_kind}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="Input file or directory.")
    parser.add_argument(
        "--data-kind",
        default="csv",
        choices=("csv", "legacy-photometry", "irtf-sxd", "irtf-lxd"),
        help="How to load --data.",
    )
    parser.add_argument(
        "--mode",
        default="basic",
        choices=("basic", "allparams", "ar", "red-excess", "spectral", "spectral-red-excess"),
        help="Fitting mode to run.",
    )
    parser.add_argument("--lambda-reg", type=float, default=0.0, help="Regularization strength.")
    parser.add_argument("--required-filters", default="J,H,K", help="Comma-separated filters required per day.")
    parser.add_argument("--fit-filters", default="J,H,K", help="Comma-separated filters to fit.")
    parser.add_argument("--output", default="results/generated/datas/fit", help="Output path stem for saved results.")
    parser.add_argument("--debug", action="store_true", help="Print extra fitting diagnostics.")
    args = parser.parse_args()

    df = load_input(args.data, args.data_kind)
    required_filters = _parse_filters(args.required_filters)
    fit_filters = _parse_filters(args.fit_filters)

    if args.mode == "basic":
        results = basic_fitting(df, lambda_reg=args.lambda_reg, debug=args.debug,
                                required_filters=required_filters, fit_filters=fit_filters)
    elif args.mode == "allparams":
        results = allpams_fitting(df, lambda_reg=args.lambda_reg, debug=args.debug)
    elif args.mode == "ar":
        results = AR_fitting(df, filename=args.output, required_filters=required_filters,
                             fit_filters=fit_filters, debug=args.debug, lambda_reg=args.lambda_reg)
    elif args.mode == "red-excess":
        results = rede_fitting(df, required_filters=required_filters, fit_filters=fit_filters,
                               filename=args.output, debug=args.debug, lambda_reg=args.lambda_reg)
    elif args.mode == "spectral":
        results = spectral_fitting(df, lambda_reg=args.lambda_reg, debug=args.debug,
                                   save_dir="results/generated/pictures/spectral", rede=False)
    else:
        results = spectral_fitting(df, lambda_reg=args.lambda_reg, debug=args.debug,
                                   save_dir="results/generated/pictures/spectral_red_excess", rede=True)

    if not results or not results.get("success"):
        print("Fit did not finish successfully.")
        return 1

    result_saver(results, filepath=args.output, file_format="csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
