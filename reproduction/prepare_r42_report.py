"""Join the R4.2 configuration and verified pilot evidence into LaTeX."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from reproduction.prepare_r21_report import read_rows, tex_number

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Generate configuration/pilot tables and macros only from saved data.

    Returns:
        None. Write TeX data and the measured configuration-check figure.
    """
    stage = ROOT / "stage4/r42"
    folder = stage / "results"
    configs = read_rows(folder / "configurations.csv")
    point = {
        r["configuration"]: r
        for r in read_rows(folder / "pilot_summary.csv")
        if r["joint_condition_id"] == "11"
    }
    controls = {
        r["configuration"]: r for r in read_rows(folder / "clean_controls.csv")
    }
    freeze = json.loads(
        (folder / "validation_choices_freeze.json").read_text()
    )
    check = json.loads((folder / "independent_checks.json").read_text())
    decomposition = read_rows(folder / "decomposition.csv")
    generated = ROOT / "report/generated"
    macros = {
        "VRecords": str(freeze["records_per_condition"]),
        "VCheckedPaths": str(check["independent_path_records"]),
        "VIndependentMse": tex_number(
            check["maximum_independent_MSE_difference"]
        ),
        "VCleanNoFilter": tex_number(float(controls["00"]["nmse"])),
        "VCleanFilter": tex_number(float(controls["10"]["nmse"])),
        "VDecompositionResidual": tex_number(
            max(float(r["mse_residual"]) for r in decomposition)
        ),
    }
    (generated / "r42_measurements.tex").write_text(
        "% Generated from R4.2 configuration/pilot data; not phase1 results.\n"
        + "\n".join(r"\newcommand{\%s}{%s}" % item for item in macros.items())
        + "\n",
        encoding="utf-8",
    )
    table = [
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"ID & Configuration & $f$ & $c$ & Sent bits / logical kbit/s \\",
        r"\midrule",
    ]
    for r in configs:
        f_enabled = int(r["filter_enabled"] == "True")
        c_enabled = int(r["ecc_enabled"] == "True")
        sent = int(r["transmitted_bits"])
        rate = int(r["logical_rate_bits_s"]) // 1000
        table.append(
            r["configuration_id"]
            + " & "
            + r["name"]
            + f" & {f_enabled} & {c_enabled} & {sent} / {rate}"
            + r" \\"
        )
    table += [r"\bottomrule", r"\end{tabular}"]
    (generated / "r42_configurations_table.tex").write_text(
        "\n".join(table) + "\n"
    )
    table = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Configuration & Pilot mean NMSE & Pilot mean MAE & Data BER \\",
        r"\midrule",
    ]
    for config in ["00", "10", "01", "11"]:
        r = point[config]
        table.append(
            config
            + f" & {float(r['mean_nmse']):.6f} & "
            + f"{float(r['mean_mae']):.6f} & $"
            + tex_number(float(r["data_ber"]))
            + r"$ \\"
        )
    table += [r"\bottomrule", r"\end{tabular}"]
    (generated / "r42_pilot_table.tex").write_text("\n".join(table) + "\n")
    shutil.copy2(
        stage / "figures/r42_paired_checks.png",
        ROOT / "report/images/r42_paired_checks.png",
    )


if __name__ == "__main__":
    main()
