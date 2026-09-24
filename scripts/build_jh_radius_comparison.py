#!/usr/bin/env python3
"""Build comparison tables for the saved 2R, 3R, and 4R JH fits."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "results/generated/datas"
EPOCH_CSV = DATA_DIR / "JH_radius_fit_parameter_comparison.csv"
SUMMARY_CSV = DATA_DIR / "JH_radius_fit_summary.csv"
LATEX_FILE = DATA_DIR / "JH_radius_fit_comparison.tex"

FAMILIES = (
    (
        "photometric",
        "Photometric $JH$",
        "basic_JH_{radius}R_newdf_daily_params.csv",
        "basic_JH_{radius}R_newdf_fit_info.csv",
    ),
    (
        "spectroscopic",
        "Spectroscopic $JH$",
        "spectral_JH_{radius}R_daily_params.csv",
        "spectral_JH_{radius}R_fit_info.csv",
    ),
)


def _sci_latex(value: float) -> str:
    exponent = int(np.floor(np.log10(abs(value))))
    mantissa = value / 10**exponent
    return rf"${mantissa:.3f}\times10^{{{exponent}}}$"


def main() -> int:
    all_data = {}
    reduced_chi_squared = {}
    epoch_rows = []
    summary_rows = []

    for key, _label, parameter_pattern, info_pattern in FAMILIES:
        data = {
            radius: pd.read_csv(
                DATA_DIR / parameter_pattern.format(radius=radius)
            ).set_index("JD_day")
            for radius in (2, 3, 4)
        }
        all_data[key] = data
        reduced_chi_squared[key] = {}

        for radius in (2, 3, 4):
            info = pd.read_csv(
                DATA_DIR / info_pattern.format(radius=radius)
            ).set_index("key")["value"]
            chi_nu = float(info["reduced_chi_squared"])
            reduced_chi_squared[key][radius] = chi_nu
            mstar_mdot = 0.5 * data[radius]["Mdot"]
            summary_rows.append(
                {
                    "fit_family": key,
                    "radius": f"{radius}R",
                    "median_MstarMdot_Msun2_per_yr": mstar_mdot.median(),
                    "min_MstarMdot_Msun2_per_yr": mstar_mdot.min(),
                    "max_MstarMdot_Msun2_per_yr": mstar_mdot.max(),
                    "median_Av_mag": data[radius]["Av"].median(),
                    "min_Av_mag": data[radius]["Av"].min(),
                    "max_Av_mag": data[radius]["Av"].max(),
                    "reduced_chi_squared": chi_nu,
                }
            )

        for jd in data[2].index:
            date = pd.to_datetime(
                (jd - 2440587.5) * 86400,
                unit="s",
                origin="unix",
                utc=True,
            ).strftime("%Y-%m-%d")
            row = {"fit_family": key, "JD": jd, "date": date}
            for radius in (2, 3, 4):
                row[f"MstarMdot_{radius}R_Msun2_per_yr"] = (
                    0.5 * data[radius].loc[jd, "Mdot"]
                )
                row[f"Av_{radius}R_mag"] = data[radius].loc[jd, "Av"]
                row[f"reduced_chi_squared_{radius}R"] = reduced_chi_squared[key][radius]
            row["MstarMdot_4R_over_2R"] = (
                row["MstarMdot_4R_Msun2_per_yr"]
                / row["MstarMdot_2R_Msun2_per_yr"]
            )
            row["Av_4R_minus_2R_mag"] = row["Av_4R_mag"] - row["Av_2R_mag"]
            epoch_rows.append(row)

    pd.DataFrame(epoch_rows).to_csv(EPOCH_CSV, index=False, float_format="%.10g")
    pd.DataFrame(summary_rows).to_csv(SUMMARY_CSV, index=False, float_format="%.10g")

    lines = [
        r"% Requires \usepackage{booktabs}",
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Comparison of fitted parameters for the 2R, 3R, and 4R models.",
        r"Values are summarized by the median and full range across epochs.}",
        r"\label{tab:radius-fit-parameters}",
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"Fit family & Radius & Median $M_\star\dot{M}$ [min, max] & Median $A_V$ [min, max] & $\chi^2_\nu$ \\",
        r"& & ($M_\odot^2\,\mathrm{yr}^{-1}$) & (mag) & \\",
        r"\midrule",
    ]
    for family_index, (key, label, _parameter_pattern, _info_pattern) in enumerate(FAMILIES):
        for radius in (2, 3, 4):
            frame = all_data[key][radius]
            mstar_mdot = 0.5 * frame["Mdot"]
            chi_nu = reduced_chi_squared[key][radius]
            chi_text = "---" if not np.isfinite(chi_nu) else f"{chi_nu:.4f}"
            lines.append(
                f"{label} & ${radius}R$ & {_sci_latex(mstar_mdot.median())} "
                f"[{_sci_latex(mstar_mdot.min())}, {_sci_latex(mstar_mdot.max())}] "
                f"& {frame['Av'].median():.2f} [{frame['Av'].min():.2f}, "
                f"{frame['Av'].max():.2f}] & {chi_text} \\\\"
            )
        if family_index == 0:
            lines.append(r"\midrule")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\begin{flushleft}",
            r"\footnotesize The photometric $JH$ fits have zero degrees of freedom, so their reduced $\chi^2$ is undefined.",
            r"\end{flushleft}",
            r"\end{table}",
            "",
            r"\begin{table}[ht]",
            r"\centering",
            r"\caption{Systematic changes in the fitted parameters resulting from the assumed stellar and inner-disc radius.}",
            r"\label{tab:radius-fit-differences}",
            r"\begin{tabular}{lrrrr}",
            r"\toprule",
            r"Comparison & \multicolumn{2}{c}{Photometric $JH$} & \multicolumn{2}{c}{Spectroscopic $JH$} \\",
            r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
            r"& $\Delta(M_\star\dot{M})$ & $\Delta A_V$ & $\Delta(M_\star\dot{M})$ & $\Delta A_V$ \\",
            r"& (\%) & (mag) & (\%) & (mag) \\",
            r"\midrule",
        ]
    )
    for lower, upper in ((2, 3), (2, 4)):
        values = []
        for key in ("photometric", "spectroscopic"):
            data = all_data[key]
            values.extend(
                [
                    100 * ((data[upper]["Mdot"] / data[lower]["Mdot"]).median() - 1),
                    (data[upper]["Av"] - data[lower]["Av"]).median(),
                ]
            )
        lines.append(
            f"${upper}R$ versus ${lower}R$ & ${values[0]:+.2f}$ & ${values[1]:+.2f}$ "
            f"& ${values[2]:+.2f}$ & ${values[3]:+.2f}$ \\\\"
        )
    spectral_chi = reduced_chi_squared["spectroscopic"]
    lines.extend(
        [
            r"\midrule",
            r"Reduced $\chi^2$ ($2R$, $3R$, $4R$) & \multicolumn{2}{c}{undefined (0 dof)} "
            rf"& \multicolumn{{2}}{{c}}{{{spectral_chi[2]:.4f}, {spectral_chi[3]:.4f}, {spectral_chi[4]:.4f}}} \\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )
    LATEX_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Saved {EPOCH_CSV}")
    print(f"Saved {SUMMARY_CSV}")
    print(f"Saved {LATEX_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
