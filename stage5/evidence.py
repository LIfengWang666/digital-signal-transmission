"""Join provenance-labelled R1--R4 evidence without mixing study cohorts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from stage2.r21.filtering_approach import write_csv
from stage2.r23.validation import file_hash
from stage4.r43.assessment import read_rows

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent


def component_evidence() -> list[dict]:
    """Aggregate R4 powers on the original main/personal/diagnostic sets.

    Returns:
        Signed full-grid component powers, keeping cohort and units explicit.
        Power ratios are not interpreted as additive responsibility shares.
    """
    freeze = json.loads(
        (ROOT / "stage4/r43/results/validation_freeze.json").read_text()
    )
    main = {
        c["joint_condition_id"]
        for c in freeze["choices"]["conditions"]
        if c["primary"]
    }
    components = read_rows(ROOT / "stage4/r43/results/component_summary.csv")
    result = []
    for scope, ids in [("main12", main), ("personal", {11}), ("SNR40", {21})]:
        selected = [
            r
            for r in components
            if r["configuration"] == "11"
            and int(r["joint_condition_id"]) in ids
        ]
        assert len(selected) == len(ids)
        for name in [
            "sampling",
            "filter",
            "gaussian",
            "quantisation",
            "channel",
            "cross_sum",
            "total",
        ]:
            if name == "cross_sum":
                values = [
                    sum(
                        float(v)
                        for k, v in row.items()
                        if k.startswith("cross_")
                    )
                    for row in selected
                ]
            else:
                field = (
                    "mean_mse_from_components"
                    if name == "total"
                    else f"power_{name}"
                )
                values = [float(row[field]) for row in selected]
            power = float(np.mean(values))
            result.append(
                {
                    "scope": scope,
                    "term": name,
                    "mean_power_48k": power,
                    "divided_by_reference_power": power / 0.7299999999999999,
                    "cohort": "R4 study4 phase1;1400 bundles per condition",
                    "unique_conditions": len(ids),
                    "source": "stage4/r43/results/component_summary.csv",
                }
            )
    return result


def confirmation_accounting() -> list[dict]:
    """Account for prechannel and digital error in the same R5 cohort.

    Returns:
        Main/personal mean prechannel, channel and signed-cross terms for
        each variant. Differences are not mislabelled pure quantiser gains.
    """
    design = json.loads((HERE / "design.json").read_text())
    source = read_rows(HERE / "results/case_summary.csv")
    result = []
    for scope, ids in [
        ("main12", set(design["main_ids"])),
        ("personal", {11}),
    ]:
        for name in [v["id"] for v in design["variants"]]:
            selected = [
                r
                for r in source
                if int(r["joint_condition_id"]) in ids and r["variant"] == name
            ]
            before = float(
                np.mean([float(r["mean_prechannel_mse"]) for r in selected])
            )
            channel = float(
                np.mean([float(r["mean_channel_power"]) for r in selected])
            )
            cross = float(
                np.mean([float(r["mean_channel_cross"]) for r in selected])
            )
            nmse = float(np.mean([float(r["mean_nmse"]) for r in selected]))
            assert (
                abs((before + channel + cross) / 0.7299999999999999 - nmse)
                < 1e-10
            )
            result.append(
                {
                    "scope": scope,
                    "variant": name,
                    "mean_prechannel_mse": before,
                    "mean_channel_power": channel,
                    "mean_channel_cross": cross,
                    "mean_nmse": nmse,
                    "cohort": "R5 study5 phase1; same350 bundles",
                }
            )
    return result


def main() -> None:
    """Save measured decision anchors and their source hashes.

    Returns:
        None. Existing R1--R4 data are read only; new R5 confirmation
        comparisons remain a separate cohort with their own random leaves.
    """
    folder = HERE / "results"
    write_csv(folder / "r4_component_evidence.csv", component_evidence())
    if (folder / "phase1_independent_checks.json").exists():
        write_csv(
            folder / "confirmed_stage_accounting.csv",
            confirmation_accounting(),
        )
    rates = read_rows(ROOT / "stage1/r12/results/rate_comparison.csv")
    quant = read_rows(ROOT / "stage3/r31/results/summary.csv")
    bits = read_rows(ROOT / "stage3/r31/results/bit_selection.csv")
    checks = json.loads(
        (ROOT / "stage4/r43/results/independent_checks.json").read_text()
    )
    rows = []
    for rate in [16000, 24000]:
        row = next(r for r in rates if float(r["sample_rate_hz"]) == rate)
        rows.append(
            {
                "decision": f"sampling{rate}",
                "quantity": "clean reconstruction MSE",
                "value": float(row["mse"]),
                "cohort": "R1 deterministic sufficient-rate comparison",
                "source": "stage1/r12/results/rate_comparison.csv",
            }
        )
    selected = next(
        r
        for r in quant
        if r["condition_id"] == "5"
        and r["branch"] == "filtered"
        and r["bits"] == "14"
    )
    rows.append(
        {
            "decision": "quantisation14",
            "quantity": "quantisation component MSE on16k grid",
            "value": float(selected["mean_quantisation_mse_16000"]),
            "cohort": "R3.1 phase0;5 design records, not R4 validation",
            "source": "stage3/r31/results/summary.csv",
        }
    )
    for width in [11, 14, 16]:
        row = next(r for r in bits if r["bits"] == str(width))
        rows.append(
            {
                "decision": f"quantisation{width}",
                "quantity": "worst original quantisation component ratio",
                "value": float(row["worst_component_ratio"]),
                "cohort": "R3.1 declared full design envelope",
                "source": "stage3/r31/results/bit_selection.csv",
            }
        )
    rows.append(
        {
            "decision": "FFT numerical implementation",
            "quantity": "maximum independent R4 MSE discrepancy",
            "value": checks["maximum_independent_MSE_difference"],
            "cohort": ("R4 phase1; all123200 paths; independent algorithms"),
            "source": "stage4/r43/results/independent_checks.json",
        }
    )
    write_csv(folder / "earlier_decision_evidence.csv", rows)
    sources = sorted(
        {r["source"] for r in rows}
        | {
            "stage4/r43/results/component_summary.csv",
            "stage4/r43/results/validation_freeze.json",
            "stage4/r43/results/objective_summary.csv",
        }
    )
    for name in [
        "phase1_freeze.json",
        "phase1_independent_checks.json",
        "decision_evidence.csv",
    ]:
        if (folder / name).exists():
            sources.append(f"stage5/results/{name}")
    (folder / "evidence_sources.json").write_text(
        json.dumps(
            {
                "date": "2026-10-21",
                "cohorts_combined_as_independent_observations": False,
                "source_sha256": {
                    name: file_hash(ROOT / name) for name in sources
                },
            },
            indent=2,
        )
        + "\n"
    )
    print("R5 component and competing-decision provenance generated.")


if __name__ == "__main__":
    main()
