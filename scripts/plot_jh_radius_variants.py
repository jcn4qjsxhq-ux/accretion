#!/usr/bin/env python3
"""Regenerate JH radius final panels using non-interpolated plot data."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from accretion.final_panels import plot_final_panels_0_to_4_layouts  # noqa: E402
from accretion.ultimate_common import M_sun, R_sun, constants  # noqa: E402
from accretion.ultimate_file_organisers import load_database_new, result_opener  # noqa: E402


DATABASE_PATH = PROJECT_ROOT / "_private/data/raw/photometry/database_daily_2609.txt"
RESULTS_DIR = PROJECT_ROOT / "results/generated/datas"
PICTURES_DIR = PROJECT_ROOT / "results/generated/pictures"


def main() -> int:
    photometry = load_database_new(DATABASE_PATH)
    for radius_rsun in (2, 3, 4):
        photometric_results = result_opener(
            str(RESULTS_DIR / f"basic_JH_{radius_rsun}R_newdf")
        )
        spectral_results = result_opener(
            str(RESULTS_DIR / f"spectral_JH_{radius_rsun}R")
        )
        output_path = PICTURES_DIR / f"0-4_JH_{radius_rsun}R.pdf"

        _, _, residuals = plot_final_panels_0_to_4_layouts(
            photometry,
            photometric_results,
            spectral_results=spectral_results,
            default_params={
                "M_star": 0.5 * M_sun,
                "R_star": radius_rsun * R_sun,
                "R_in": radius_rsun * R_sun,
                "R_out": 2 * constants.au,
                "distance": 700 * constants.parsec,
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
            two_column_save_path=output_path,
            show=False,
        )

        residual_counts = residuals["flux"].groupby("Group").size().to_dict()
        print(f"Saved {output_path}")
        print(f"Residual points: {residual_counts}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
