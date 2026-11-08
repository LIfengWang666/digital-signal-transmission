"""Prepare R3.2 report tables from the frozen, independently checked pilot."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from reproduction.prepare_r21_report import read_rows, tex_number

ROOT = Path(__file__).resolve().parents[1]
LABELS = {
    "uncoded": "Uncoded",
    "hamming74": "Hamming (7,4)",
    "hamming1511": "Hamming (15,11)",
    "repetition51": "Repetition (5,1)",
}


def main() -> None:
    """Generate three evidence tables, macros and image copies.

    Returns:
        None. Write only report assets, with no experimental choices.
    """
    stage = ROOT / "stage3/r32"
    generated = ROOT / "report/generated"
    selection = read_rows(stage / "results/code_selection.csv")
    summary = read_rows(stage / "results/summary.csv")
    paired = read_rows(stage / "results/paired_scores.csv")
    frozen = json.loads((stage / "results/code_freeze.json").read_text())
    assert frozen["selected_code"] == "hamming74"
    chosen = next(r for r in selection if r["code"] == "hamming74")
    macros = {
        "ECCScore": f"{float(chosen['score']):.5f}",
        "ECCScoreLow": f"{float(chosen['score_ci95_low']):.5f}",
        "ECCScoreHigh": f"{float(chosen['score_ci95_high']):.5f}",
    }
    (generated / "r32_measurements.tex").write_text(
        "% Generated from saved R3.2 measurements.\n"
        + "\n".join(r"\newcommand{\%s}{%s}" % item for item in macros.items())
        + "\n",
        encoding="utf-8",
    )
    table = [
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"Code & $n$ & $k$ & $d_{\min}$ & $t$ & Expansion & kbit/s & $J$ \\",
        r"\midrule",
    ]
    for row in selection:
        table.append(
            LABELS[row["code"]]
            + " & "
            + " & ".join(row[key] for key in ["n", "k", "d_min", "t"])
            + f" & {float(row['payload_expansion']):.5f}"
            + f" & {float(row['transmitted_bits_per_second'])/1000:g}"
            + f" & {float(row['score']):.5f}"
            + r" \\"
        )
    table += [r"\bottomrule", r"\end{tabular}"]
    (generated / "r32_selection_table.tex").write_text(
        "\n".join(table) + "\n", encoding="utf-8"
    )
    table = [
        r"\begin{tabular}{rlrrr}",
        r"\toprule",
        r"$p$ & Code & Wrong data bits & Data BER & Full-grid MSE \\",
        r"\midrule",
    ]
    for probability in [0.001, 0.01, 0.03, 0.3]:
        for code in LABELS:
            row = next(
                r
                for r in summary
                if r["code"] == code and float(r["probability"]) == probability
            )
            table.append(
                f"{probability:g} & {LABELS[code]} & "
                + row["wrong_bits"]
                + " & $"
                + tex_number(float(row["data_ber"]))
                + "$ & $"
                + tex_number(float(row["mean_mse_48000"]))
                + r"$ \\"
            )
        table.append(r"\addlinespace")
    table += [r"\bottomrule", r"\end{tabular}"]
    (generated / "r32_quality_table.tex").write_text(
        "\n".join(table) + "\n", encoding="utf-8"
    )
    profile = read_rows(stage / "results/error_profile.csv")
    table = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Code & Blocks & Measured $J_b>t$ & Fraction & Binomial tail \\",
        r"\midrule",
    ]
    for code, radius in [
        ("hamming74", 1),
        ("hamming1511", 1),
        ("repetition51", 2),
    ]:
        rows = [
            r
            for r in profile
            if r["code"] == code and float(r["probability"]) == 0.01
        ]
        outside = [r for r in rows if int(r["weight"]) > radius]
        count = sum(int(r["observed_blocks"]) for r in outside)
        blocks = int(rows[0]["total_blocks"])
        theory = sum(float(r["binomial_probability"]) for r in outside)
        table.append(
            f"{LABELS[code]} & {blocks:,} & {count:,} & "
            f"{count/blocks:.6f} & {theory:.6f}" + r" \\"
        )
    table += [r"\bottomrule", r"\end{tabular}"]
    (generated / "r32_profile_table.tex").write_text(
        "\n".join(table) + "\n", encoding="utf-8"
    )
    assert len(paired) == 3
    for name in [
        "r32_block_profile",
        "r32_ber_reconstruction",
        "r32_cost_example",
    ]:
        shutil.copyfile(
            stage / "figures" / (name + ".png"),
            ROOT / "report/images" / (name + ".png"),
        )
    print("R3.2: three tables, macros and three figures prepared.")


if __name__ == "__main__":
    main()
