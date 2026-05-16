"""Create summary figures from a saved result bundle and matching data."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import pandas as pd

from accretion import (
    accretion_model,
    comprehensive_results_visualization,
    load_database_new,
    result_opener,
    visualize_spectral_results,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, help="Saved result path stem.")
    parser.add_argument("--data", type=Path, help="Photometry CSV or legacy photometry table.")
    parser.add_argument("--legacy-data", action="store_true", help="Load --data with load_database_new.")
    parser.add_argument("--spectral", action="store_true", help="Use spectral plotting helpers.")
    parser.add_argument("--output-dir", default="results/generated/pictures", help="Figure output directory.")
    args = parser.parse_args()

    results = result_opener(args.results, file_format="csv")

    if args.spectral:
        visualize_spectral_results(results, save_dir=args.output_dir)
        return 0

    if args.data is None:
        raise ValueError("--data is required for photometric summary figures")

    df = load_database_new(str(args.data)) if args.legacy_data else pd.read_csv(args.data)
    comprehensive_results_visualization(results, df, accretion_model, save_dir=args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
