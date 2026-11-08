"""Generate the R4.1 metric tables, numbers and figure from measured CSV."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from reproduction.prepare_r21_report import read_rows

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Prepare metric evidence while keeping legacy examples labelled.

    Returns:
        None. Write report macros/tables and copy the metric-check PNG.
    """
    stage = ROOT / "stage4/r41"
    semantic = {
        r["case"]: r for r in read_rows(stage / "results/semantic_checks.csv")
    }
    records = read_rows(stage / "results/existing_record_metrics.csv")
    aggregate = read_rows(stage / "results/aggregation_example.csv")[0]
    checks = json.loads(
        (stage / "results/independent_checks.json").read_text()
    )
    generated = ROOT / "report/generated"
    generated.mkdir(exist_ok=True)
    sparse = float(semantic["sparse_2_at_1percent"]["nmse"])
    macros = {
        "MReferencePower": f"{float(records[0]['reference_power']):.2f}",
        "MFromMeanDb": f"{float(aggregate['db_from_mean_nmse']):.5f}",
        "MMeanOfDb": f"{float(aggregate['mean_of_record_db']):.5f}",
        "MIndependentChecks": str(checks["independent_assertions"]),
        "MDiffuseNmse": f"{float(semantic['diffuse_0.1']['nmse']):.8f}",
        "MSparseNmse": f"{sparse:.8f}",
    }
    (generated / "r41_measurements.tex").write_text(
        "% Generated from R4.1 CSV/checks; examples are not new R4 trials.\n"
        + "\n".join(r"\newcommand{\%s}{%s}" % item for item in macros.items())
        + "\n",
        encoding="utf-8",
    )
    labels = {
        "prechannel": "Before BSC (shared)",
        "personal_uncoded": r"$p_{\rm ID}$, uncoded",
        "personal_hamming74": r"$p_{\rm ID}$, Hamming",
        "high_error_uncoded": "$p=0.03$, uncoded",
        "high_error_hamming74": "$p=0.03$, Hamming",
    }
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Existing record & NMSE & MAE & Peak $|e|$ & Wrong data bits \\",
        r"\midrule",
    ]
    for row in records:
        wrong = (
            "--"
            if row["decoded_wrong_bits"] == "not_applied"
            else row["decoded_wrong_bits"]
        )
        lines.append(
            labels[row["case"]]
            + f" & {float(row['nmse']):.6f} & "
            + f"{float(row['mae']):.6f} & "
            + f"{float(row['peak_absolute_error']):.4f} & "
            + wrong
            + r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (generated / "r41_existing_metrics_table.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    images = ROOT / "report/images"
    images.mkdir(exist_ok=True)
    shutil.copy2(stage / "figures/r41_metric_ranking.png", images)


if __name__ == "__main__":
    main()
