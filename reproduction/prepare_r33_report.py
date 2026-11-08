"""Join the frozen R3.3 range evidence into report macros and tables."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from reproduction.prepare_r21_report import read_rows, tex_number

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Prepare two measured tables and two figure copies.

    Returns:
        None. Write report assets without selecting new conditions.
    """
    stage = ROOT / "stage3/r33"
    frozen = json.loads((stage / "results/range_freeze.json").read_text())
    parameter = json.loads(
        (stage / "results/personal_parameter.json").read_text()
    )
    summary = read_rows(stage / "results/summary.csv")
    generated = ROOT / "report/generated"
    macros = {
        "PUniform": f"{parameter['uniform_draw']:.17g}",
        "PPersonal": f"{parameter['personal_probability']:.17g}",
        "PCapacityRoot": f"{frozen['capacity_equals_code_rate_at']:.8f}",
        "PBerRoot": f"{frozen['exact_data_ber_equals_uncoded_at']:.8f}",
    }
    (generated / "r33_measurements.tex").write_text(
        "% Generated from saved R3.3 evidence.\n"
        + "\n".join(r"\newcommand{\%s}{%s}" % item for item in macros.items())
        + "\n",
        encoding="utf-8",
    )
    table = [
        r"\begin{tabular}{rlrrr}",
        r"\toprule",
        r"$p$ & Role & $C(p)$ & $\Pr(J>1)$ & Expected failed blocks \\",
        r"\midrule",
    ]
    labels = {
        "identity": "Identity",
        "primary_grid": "Main grid",
        "personal": "Anchor",
        "stress": "Stress",
    }
    for row in summary:
        if row["code"] != "hamming74":
            continue
        probability = (
            r"$p_{\rm ID}$"
            if row["role"] == "personal"
            else f"{float(row['probability']):g}"
        )
        tail = float(row["theory_block_failure_probability"])
        table.append(
            probability
            + " & "
            + labels[row["role"]]
            + f" & {float(row['capacity']):.5f} & $"
            + (tex_number(tail) if tail else "0")
            + "$ & "
            + f"{float(row['expected_failed_blocks']):.3f}"
            + r" \\"
        )
    table += [r"\bottomrule", r"\end{tabular}"]
    (generated / "r33_range_table.tex").write_text(
        "\n".join(table) + "\n", encoding="utf-8"
    )
    table = [
        r"\begin{tabular}{rrrrr}",
        r"\toprule",
        r"$p$ & Failed blocks & Measured BLER & "
        r"Pointwise 95\% CI & Data errors \\",
        r"\midrule",
    ]
    for row in summary:
        if row["code"] != "hamming74" or row["condition_id"] not in [
            "1",
            "8",
            "11",
            "2",
        ]:
            continue
        p = (
            r"$p_{\rm ID}$"
            if row["role"] == "personal"
            else (f"{float(row['probability']):g}")
        )
        table.append(
            f"{p} & {row['failed_blocks']} & $"
            + tex_number(float(row["block_failure_fraction"]))
            + "$ & $["
            + tex_number(float(row["block_ci95_low"]))
            + ", "
            + tex_number(float(row["block_ci95_high"]))
            + f"]$ & {row['wrong_data_bits']}"
            + r" \\"
        )
    table += [r"\bottomrule", r"\end{tabular}"]
    (generated / "r33_rare_event_table.tex").write_text(
        "\n".join(table) + "\n", encoding="utf-8"
    )
    for name in ["r33_range_precision", "r33_capacity_ber_limits"]:
        shutil.copyfile(
            stage / "figures" / (name + ".png"),
            ROOT / "report/images" / (name + ".png"),
        )
    print("R3.3: macros, two tables and two figures prepared.")


if __name__ == "__main__":
    main()
