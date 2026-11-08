"""Join the saved R2.3 measurements into LaTeX values and tables."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from reproduction.prepare_r21_report import read_rows, tex_number

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Prepare report assets without recomputing or selecting experiments.

    Returns:
        None. Save generated TeX macros/tables and copy two figures.
    """
    stage = ROOT / "stage2/r23"
    rows = read_rows(stage / "results/summary.csv")
    report = ROOT / "report"
    generated = report / "generated"
    verification = json.loads(
        (ROOT / "metadata/r23_verification.json").read_text()
    )

    def row(cid: int, rate: int, metric: str) -> dict[str, str]:
        """Select one unambiguous saved summary row.

        Args:
            cid: Stable condition ID.
            rate: Evaluation-grid sample rate.
            metric: CSV measurement name.

        Returns:
            The corresponding record.
        """
        return next(
            item
            for item in rows
            if int(item["condition_id"]) == cid
            and int(item["rate_hz"]) == rate
            and item["metric"] == metric
        )

    personal_before = float(row(5, 48000, "no_filter_mse")["mean"])
    personal_after = float(row(5, 48000, "filter_mse")["mean"])
    personal_gain = row(5, 48000, "paired_gain_db")
    high_after = float(row(7, 48000, "filter_mse")["mean"])
    high_edge = float(row(7, 48000, "filter_edge_mse")["mean"])
    edge_fraction = (2 * 768 / 96000) * high_edge / high_after
    reduction_percent = 100 * (1 - personal_after / personal_before)
    macros = {
        "RtvPersonalBefore": tex_number(personal_before),
        "RtvPersonalAfter": tex_number(personal_after),
        "RtvPersonalGain": f"{float(personal_gain['mean']):.3f}",
        "RtvPersonalLow": f"{float(personal_gain['ci95_low']):.3f}",
        "RtvPersonalHigh": f"{float(personal_gain['ci95_high']):.3f}",
        "RtvPersonalReduction": f"{reduction_percent:.2f}",
        "RtvHighEdgePercent": f"{100*edge_fraction:.2f}",
        "RtvCrossover": f"{verification['sample_crossover_db']:.2f}",
        "RtvCleanHigh": tex_number(
            verification["prediction_base"]["48000"][0]
        ),
        "RtvNoiseGainHigh": (
            f"{verification['prediction_base']['48000'][1]:.11f}"
        ),
    }
    (generated / "r23_measurements.tex").write_text(
        "% Generated from the held-out validation CSV/JSON.\n"
        + "\n".join(
            r"\newcommand{\%s}{%s}" % (name, value)
            for name, value in macros.items()
        )
        + "\n",
        encoding="utf-8",
    )
    table = [
        r"\begin{tabular}{rrrrr}",
        r"\toprule",
        r"Target $S$ & Mean MSE, no filter & Mean MSE, filter & "
        r"Mean gain [95\% CI] & Wins \\",
        r"(dB) & (NPY units$^2$) & (NPY units$^2$) & (dB) & (/50) \\",
        r"\midrule",
    ]
    for cid in [0, 1, 5, 2, 3, 4, 6, 7]:
        before = row(cid, 48000, "no_filter_mse")
        after = row(cid, 48000, "filter_mse")
        gain = row(cid, 48000, "paired_gain_db")
        difference = row(cid, 48000, "paired_mse_reduction")
        label = f"{float(before['target_snr_db']):g}"
        if cid == 5:
            label = r"2.661$^\dagger$"
        table.append(
            f"{label} & $"
            + tex_number(float(before["mean"]))
            + "$ & $"
            + tex_number(float(after["mean"]))
            + "$ & "
            + f"{float(gain['mean']):.3f} "
            + f"[{float(gain['ci95_low']):.3f}, "
            + f"{float(gain['ci95_high']):.3f}] & "
            + difference["positive_count"]
            + r" \\"
        )
    table.extend([r"\bottomrule", r"\end{tabular}"])
    (generated / "r23_quality_table.tex").write_text(
        "\n".join(table) + "\n", encoding="utf-8"
    )
    table = [
        r"\begin{tabular}{rrrrr}",
        r"\toprule",
        r"Target $S$ (dB) & 16 kHz total MSE & 16 kHz gain (dB) & "
        r"48 kHz edge MSE & 48 kHz interior MSE \\",
        r"\midrule",
    ]
    for cid in [5, 4, 6, 7]:
        table.append(
            f"{float(row(cid, 16000, 'filter_mse')['target_snr_db']):.3f}"
            + " & $"
            + tex_number(float(row(cid, 16000, "filter_mse")["mean"]))
            + "$ & "
            + f"{float(row(cid, 16000, 'paired_gain_db')['mean']):.3f}"
            + " & $"
            + tex_number(float(row(cid, 48000, "filter_edge_mse")["mean"]))
            + "$ & $"
            + tex_number(float(row(cid, 48000, "filter_interior_mse")["mean"]))
            + r"$ \\"
        )
    table.extend([r"\bottomrule", r"\end{tabular}"])
    (generated / "r23_boundary_table.tex").write_text(
        "\n".join(table) + "\n", encoding="utf-8"
    )
    for name in ["r23_recovery_quality.png", "r23_examples_boundaries.png"]:
        shutil.copyfile(stage / "figures" / name, report / "images" / name)
    print("R2.3: report values and tables prepared from saved measurements.")


if __name__ == "__main__":
    main()
