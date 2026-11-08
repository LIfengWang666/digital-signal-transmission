"""Join saved R3.1 measurements into report macros, tables and images."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from reproduction.prepare_r21_report import read_rows, tex_number

ROOT = Path(__file__).resolve().parents[1]


def precise_number(value: float) -> str:
    """Keep enough digits to distinguish the paired tiny MSE change.

    Args:
        value: A measured mean square.

    Returns:
        Eight significant digits in a LaTeX scientific expression.
    """
    mantissa, exponent = f"{value:.7e}".split("e")
    return mantissa + r"\times10^{" + str(int(exponent)) + "}"


def main() -> None:
    """Prepare report evidence without selecting or recomputing experiments.

    Returns:
        None. Write generated LaTeX and copy three verified figures.
    """
    stage = ROOT / "stage3/r31"
    frozen = json.loads((stage / "results/quantiser_freeze.json").read_text())
    verify = json.loads((ROOT / "metadata/r31_verification.json").read_text())
    rows = read_rows(stage / "results/summary.csv")
    selection = read_rows(stage / "results/bit_selection.csv")
    generated = ROOT / "report/generated"
    bits = frozen["bits"]
    personal = next(
        r
        for r in rows
        if r["condition_id"] == "5"
        and r["branch"] == "filtered"
        and int(r["bits"]) == bits
    )
    previous = next(r for r in selection if int(r["bits"]) == bits - 1)
    bounds = frozen["headroom"]["six_sigma_amplitude_bounds"]
    macros = {
        "QBits": str(bits),
        "QHalfRange": f"{frozen['half_range']:g}",
        "QLevels": str(frozen["levels"]),
        "QStep": tex_number(frozen["step"]),
        "QPersonalSqnr": f"{float(personal['mean_sqnr_db']):.3f}",
        "QPersonalModel": f"{float(personal['mean_model_sqnr_db']):.3f}",
        "QPersonalMin": f"{float(personal['min_sqnr_db']):.3f}",
        "QPersonalMax": f"{float(personal['max_sqnr_db']):.3f}",
        "QPersonalMse": tex_number(
            float(personal["mean_quantisation_mse_16000"])
        ),
        "QPersonalBefore": precise_number(
            float(personal["mean_upstream_mse_48000"])
        ),
        "QPersonalAfter": precise_number(
            float(personal["mean_total_mse_48000"])
        ),
        "QWorstPercent": (
            f"{100*frozen['worst_observed_component_ratio']:.4f}"
        ),
        "QPreviousPercent": (
            f"{100*float(previous['worst_component_ratio']):.4f}"
        ),
        "QMaxTotalPercent": (
            f"{100*frozen['maximum_observed_total_mse_change_fraction']:.4f}"
        ),
        "QEpsilonChanged": str(verify["selected_epsilon_changed_labels"]),
        "QRawHeadroom": f"{bounds['unfiltered']:.5f}",
        "QFilterHeadroom": f"{bounds['filtered']:.5f}",
        "QUnionBound": tex_number(
            frozen["headroom"]["two_branch_record_union_bound"]
        ),
    }
    (generated / "r31_measurements.tex").write_text(
        "% Generated only from saved R3.1 CSV/JSON.\n"
        + "\n".join(r"\newcommand{\%s}{%s}" % item for item in macros.items())
        + "\n",
        encoding="utf-8",
    )
    table = [
        r"\begin{tabular}{rrrrr}",
        r"\toprule",
        r"$B$ & Personal filtered SQNR & Worst component ratio & "
        r"Payload & Pass \\",
        r"(bits) & (dB, mean of five) & (\%, 80 branch records) & "
        r"(kbit/s) & 1\% budget \\",
        r"\midrule",
    ]
    for width in [8, 10, 12, 13, 14, 15, 16]:
        record = next(
            r
            for r in rows
            if r["condition_id"] == "5"
            and r["branch"] == "filtered"
            and int(r["bits"]) == width
        )
        chosen = next(r for r in selection if int(r["bits"]) == width)
        table.append(
            f"{width} & {float(record['mean_sqnr_db']):.3f} & "
            f"{100*float(chosen['worst_component_ratio']):.4f} & "
            f"{int(chosen['payload_bits_per_second'])/1000:g} & "
            + ("yes" if chosen["passes"] == "1" else "no")
            + r" \\"
        )
    table += [r"\bottomrule", r"\end{tabular}"]
    (generated / "r31_bit_table.tex").write_text(
        "\n".join(table) + "\n", encoding="utf-8"
    )
    table = [
        r"\begin{tabular}{rrrrr}",
        r"\toprule",
        r"$S$ & SQNR & $P_{e_q}$ at 16 kHz & "
        r"MSE before $Q$ & MSE after $Q$ \\",
        r"(dB) & (dB) & (NPY units$^2$) & "
        r"(NPY units$^2$) & (NPY units$^2$) \\",
        r"\midrule",
    ]
    selected = sorted(
        [
            r
            for r in rows
            if r["branch"] == "filtered" and int(r["bits"]) == bits
        ],
        key=lambda r: float(r["target_snr_db"]),
    )
    for record in selected:
        table.append(
            f"{float(record['target_snr_db']):.3f} & "
            f"{float(record['mean_sqnr_db']):.3f} & $"
            + tex_number(float(record["mean_quantisation_mse_16000"]))
            + "$ & $"
            + precise_number(float(record["mean_upstream_mse_48000"]))
            + "$ & $"
            + precise_number(float(record["mean_total_mse_48000"]))
            + r"$ \\"
        )
    table += [r"\bottomrule", r"\end{tabular}"]
    (generated / "r31_quality_table.tex").write_text(
        "\n".join(table) + "\n", encoding="utf-8"
    )
    for name in [
        "r31_sqnr_selection",
        "r31_quantiser_errors",
        "r31_range_fidelity",
    ]:
        shutil.copyfile(
            stage / "figures" / (name + ".png"),
            ROOT / "report/images" / (name + ".png"),
        )
    print("R3.1: report values, two tables and three figures prepared.")


if __name__ == "__main__":
    main()
