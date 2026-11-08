"""Prepare R3.4 tables and values from the independent validation outputs."""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

from reproduction.prepare_r21_report import read_rows, tex_number

ROOT = Path(__file__).resolve().parents[1]


def probability_label(value: float, personal: float) -> str:
    """Label the personal point without rounding it into a grid point.

    Args:
        value: Actual probability.
        personal: Frozen personal parameter.

    Returns:
        A LaTeX math expression for the table.
    """
    return r"$p_{\rm ID}$" if value == personal else "$" + f"{value:g}" + "$"


def main() -> None:
    """Join measured validation evidence into four tables and three figures.

    Returns:
        None. Write report macros, tables and image copies.
    """
    stage = ROOT / "stage3/r34"
    folder = stage / "results"
    design = json.loads((folder / "validation_freeze.json").read_text())
    outcome = json.loads((folder / "gain_summary.json").read_text())
    summary = read_rows(folder / "summary.csv")
    comparisons = read_rows(folder / "comparisons.csv")
    generated = ROOT / "report/generated"
    personal = design["personal_probability"]
    coded = next(
        r
        for r in summary
        if r["code"] == "hamming74" and r["condition_id"] == "11"
    )
    uncoded = next(
        r
        for r in summary
        if r["code"] == "uncoded" and r["condition_id"] == "11"
    )
    paired = next(r for r in comparisons if r["condition_id"] == "11")
    zero = next(
        r
        for r in summary
        if r["code"] == "hamming74" and r["condition_id"] == "0"
    )
    threshold_db = -10 * math.log10(design["primary_nmse_threshold"])
    gain_high = float(paired["paired_gain_ci95_high_db"])
    macros = {
        "CTau": f"{design['primary_nmse_threshold']:.8f}",
        "CTauMse": f"{design['primary_mse_threshold']:.8f}",
        "CDesignBase": f"{design['design_baseline_nmse']:.8f}",
        "CNewBase": f"{float(zero['mean_nmse']):.8f}",
        "CCritical": f"{design['adjusted_t_critical']:.6f}",
        "CThresholdDb": f"{threshold_db:.5f}",
        "CPersonalCoded": f"{float(coded['mean_nmse']):.8f}",
        "CPersonalUncoded": f"{float(uncoded['mean_nmse']):.8f}",
        "CPersonalCodedMse": f"{float(coded['mean_mse']):.8f}",
        "CPersonalUncodedMse": f"{float(uncoded['mean_mse']):.8f}",
        "CPerfectPersonal": coded["perfect_payload_records"],
        "CPerfectPersonalPercent": (
            f"{100*float(coded['perfect_payload_fraction']):.2f}"
        ),
        "CPerfectPersonalLow": (
            f"{100*float(coded['perfect_payload_ci95_low']):.2f}"
        ),
        "CPerfectPersonalHigh": (
            f"{100*float(coded['perfect_payload_ci95_high']):.2f}"
        ),
        "CPerfectUncodedHigh": (
            f"{100*float(uncoded['perfect_payload_ci95_high']):.2f}"
        ),
        "CPerfectPersonalTheory": (
            f"{100*float(coded['theory_perfect_payload_probability']):.2f}"
        ),
        "CPersonalGainDb": (
            f"{float(paired['mean_paired_reconstruction_gain_db']):.5f}"
        ),
        "CPersonalGainLow": f"{float(paired['paired_gain_ci95_low_db']):.5f}",
        "CPersonalGainHigh": f"{gain_high:.5f}",
    }
    (generated / "r34_measurements.tex").write_text(
        "% Generated only from the frozen R3.4 validation outputs.\n"
        + "\n".join(r"\newcommand{\%s}{%s}" % item for item in macros.items())
        + "\n",
        encoding="utf-8",
    )
    table = [
        r"\begin{tabular}{rlll}",
        r"\toprule",
        r"$p$ & Uncoded mean [adjusted CI] & "
        r"Hamming mean [adjusted CI] & Gain \\",
        r"\midrule",
    ]
    conditions = sorted(design["conditions"], key=lambda c: c["probability"])
    for condition in conditions:
        records = {}
        for code in ["uncoded", "hamming74"]:
            row = next(
                r
                for r in summary
                if r["code"] == code
                and int(r["condition_id"]) == (condition["condition_id"])
            )
            records[code] = (
                f"{float(row['mean_nmse']):.4f}"
                + f" [{float(row['adjusted_ci_low']):.4f},"
                + f"{float(row['adjusted_ci_high']):.4f}]"
            )
        result = next(
            r
            for r in comparisons
            if int(r["condition_id"]) == condition["condition_id"]
        )
        table.append(
            probability_label(condition["probability"], personal)
            + " & "
            + records["uncoded"]
            + " & "
            + records["hamming74"]
            + " & "
            + ("yes" if result["coding_gain_established"] == "1" else "no")
            + r" \\"
        )
    table += [r"\bottomrule", r"\end{tabular}"]
    (generated / "r34_quality_table.tex").write_text(
        "\n".join(table) + "\n", encoding="utf-8"
    )
    table = [
        r"\begin{tabular}{rrrrrr}",
        r"\toprule",
        r"$p$ & \multicolumn{2}{c}{Bit-perfect payloads} & "
        r"\multicolumn{2}{c}{Waveform records passing} & Coded BER \\",
        r"& Uncoded & Hamming & Uncoded & Hamming & \\",
        r"\midrule",
    ]
    for cid in [0, 1, 11, 2, 12]:
        code = next(
            r
            for r in summary
            if int(r["condition_id"]) == cid and r["code"] == "hamming74"
        )
        un = next(
            r
            for r in summary
            if int(r["condition_id"]) == cid and r["code"] == "uncoded"
        )
        error = float(code["data_ber"])
        table.append(
            probability_label(float(code["probability"]), personal)
            + " & "
            + un["perfect_payload_records"]
            + "/300 & "
            + code["perfect_payload_records"]
            + "/300 & "
            + un["individual_waveform_passes"]
            + "/300 & "
            + code["individual_waveform_passes"]
            + "/300 & $"
            + (tex_number(error) if error else "0")
            + r"$ \\"
        )
    table += [r"\bottomrule", r"\end{tabular}"]
    (generated / "r34_payload_table.tex").write_text(
        "\n".join(table) + "\n", encoding="utf-8"
    )
    table = [
        r"\begin{tabular}{rrp{0.6\linewidth}}",
        r"\toprule",
        r"Budget factor & NMSE threshold & Gain at tested probabilities \\",
        r"\midrule",
    ]
    for factor, probabilities in outcome["threshold_sensitivity"].items():
        values = [probability_label(v, personal) for v in probabilities]
        table.append(
            f"{float(factor):g} & "
            f"{float(factor)*design['design_baseline_nmse']:.6f} & "
            + ", ".join(values)
            + r" \\"
        )
    table += [r"\bottomrule", r"\end{tabular}"]
    (generated / "r34_sensitivity_table.tex").write_text(
        "\n".join(table) + "\n", encoding="utf-8"
    )
    cost = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Path & Rate & Padding & Bits / record & kbit/s & Expansion \\",
        r"\midrule",
        r"Uncoded & $1$ & 0 & 448000 & 224 & $1$ \\",
        r"Hamming (7,4) & $4/7$ & 0 & 784000 & 392 & $1.75$ \\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    (generated / "r34_cost_table.tex").write_text(
        "\n".join(cost) + "\n", encoding="utf-8"
    )
    for name in [
        "r34_quality_gain",
        "r34_bit_recovery",
        "r34_sensitivity_examples",
    ]:
        shutil.copyfile(
            stage / "figures" / (name + ".png"),
            ROOT / "report/images" / (name + ".png"),
        )
    print("R3.4: report macros, four tables and three figures prepared.")


if __name__ == "__main__":
    main()
