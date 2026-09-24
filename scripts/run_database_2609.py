#!/usr/bin/env python3
"""Run database_daily_2609 fits with and without interpolated photometry."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from accretion.final_panel_colour_grid import plot_final_panel_colour_grid  # noqa: E402
from accretion.final_panels import (  # noqa: E402
    plot_final_panels_0_to_4_layouts,
    plot_panel5_seds,
)
from accretion.ultimate_common import (  # noqa: E402
    M_sun,
    R_sun,
    RED_EXCESS_DEFAULT_R_BB,
    RED_EXCESS_DEFAULT_T_BB,
    constants,
    plt,
)
from accretion.ultimate_file_organisers import (  # noqa: E402
    load_database_new,
    result_opener,
    result_saver,
)
from accretion.ultimate_fitting_types import basic_fitting  # noqa: E402


DATABASE_PATH = (
    PROJECT_ROOT / "_private" / "data" / "raw" / "photometry" / "database_daily_2609.txt"
)
RESULTS_DIR = PROJECT_ROOT / "_private" / "results" / "database_daily_2609"
FIGURES_DIR = PROJECT_ROOT / "_private" / "figures" / "database_daily_2609"
SPECTRAL_RESULTS_PATH = (
    PROJECT_ROOT / "_private" / "results" / "paper_updates" / "spectral_all_updated_epochs"
)


def _make_panels(label, photometry, photo_results, spectral_results, figures_dir):
    panel_dir = figures_dir / label
    panel_dir.mkdir(parents=True, exist_ok=True)

    default_params = {
        "M_star": 0.5 * M_sun,
        "R_star": 3.0 * R_sun,
        "R_in": 3.0 * R_sun,
        "R_out": 2 * constants.au,
        "distance": 700 * constants.parsec,
        "T_bb": RED_EXCESS_DEFAULT_T_BB,
        "R_bb": RED_EXCESS_DEFAULT_R_BB,
    }
    fit_filters = list(photo_results.get("fit_info", {}).get("fit_filters") or label)

    _, _, residuals = plot_final_panels_0_to_4_layouts(
        photometry,
        photo_results,
        spectral_results=spectral_results,
        default_params=default_params,
        selected_filters=("J", "H", "K"),
        photometric_fit_filters=fit_filters,
        residual_filter_groups=("J", "H", "K", "L+W1"),
        plot_panel0_fits=False,
        filter_diff=False,
        show_panel3=False,
        end_zoom_range=(2459000, 2461300),
        flux_residual_log_scale=False,
        one_column_save_path=panel_dir / f"0-4_{label}_one_column.pdf",
        two_column_save_path=panel_dir / f"0-4_{label}_two_columns.pdf",
        show=False,
    )

    panel5_outputs = plot_panel5_seds(
        2460848,
        photo_results,
        spectral_results,
        default_params=default_params,
        photometric_filters=residuals["filters"],
        show_range=(0.78, 3.5),
        save_dir=panel_dir / "panel5_sed",
        show=False,
    )

    panel6_path = panel_dir / f"6_colour_grid_{label}.pdf"
    panel6_fig, panel6_matches, _ = plot_final_panel_colour_grid(
        photometry,
        photo_results,
        tolerance=2.0,
        show_fit=False,
        two_columns=True,
        plot=("J,J-H", "J-H,H-K", "I,I-J", "K,H-K"),
        save_path=panel6_path,
        show=False,
    )
    plt.close(panel6_fig)

    return {
        "panels_0_4_one_column": str(panel_dir / f"0-4_{label}_one_column.pdf"),
        "panels_0_4_two_columns": str(panel_dir / f"0-4_{label}_two_columns.pdf"),
        "panel_5": [str(path) for _, path in panel5_outputs if path is not None],
        "panel_6": str(panel6_path),
        "panel_6_match_counts": {
            name: int(len(matches)) for name, matches in panel6_matches.items()
        },
    }


def _run_variant(name, photometry, spectral_results, include_interpolated):
    results_dir = RESULTS_DIR / name
    figures_dir = FIGURES_DIR / name
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    fits = {}
    for label, filters in (("JH", ("J", "H")), ("JHK", ("J", "H", "K"))):
        print(f"\nRunning {name} {label} photometric fit from {DATABASE_PATH.name}")
        results = basic_fitting(
            photometry,
            required_filters=filters,
            fit_filters=filters,
            debug=0,
        )
        if not results or not results.get("success"):
            raise RuntimeError(f"{name} {label} photometric fit did not complete successfully")
        result_stem = results_dir / f"photometry_{label.lower()}"
        result_saver(results, str(result_stem))
        fits[label] = results

    summary = {
        "database": str(DATABASE_PATH),
        "variant": name,
        "include_interpolated": include_interpolated,
        "input_rows": int(len(photometry)),
        "results_directory": str(results_dir),
        "figures_directory": str(figures_dir),
        "fits": {},
    }
    for label, results in fits.items():
        summary["fits"][label] = {
            "result_stem": str(results_dir / f"photometry_{label.lower()}"),
            "n_days": int(len(results.get("daily_params", {}))),
            "fit_info": results.get("fit_info", {}),
            "panels": _make_panels(
                label,
                photometry,
                results,
                spectral_results,
                figures_dir,
            ),
        }

    manifest_path = results_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"\n{name} pipeline manifest saved to {manifest_path}")
    return summary


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    photometry = load_database_new(DATABASE_PATH)
    raw_photometry = photometry[
        photometry["Interpolated"].astype(str).str.strip().str.lower().eq("no")
    ].copy()
    spectral_results = result_opener(str(SPECTRAL_RESULTS_PATH))

    summaries = {
        "database": str(DATABASE_PATH),
        "database_rows": int(len(photometry)),
        "raw_rows": int(len(raw_photometry)),
        "variants": {},
    }
    summaries["variants"]["with_interpolated"] = _run_variant(
        "with_interpolated", photometry, spectral_results, True
    )
    summaries["variants"]["without_interpolated"] = _run_variant(
        "without_interpolated", raw_photometry, spectral_results, False
    )

    manifest_path = RESULTS_DIR / "run_manifest_comparison.json"
    manifest_path.write_text(json.dumps(summaries, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"\nComparison manifest saved to {manifest_path}")
    return summaries


if __name__ == "__main__":
    main()
