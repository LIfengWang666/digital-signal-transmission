"""Use complete pilot bundles to plan, not infer, later mean precision."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import t

from stage2.r21.filtering_approach import write_csv


def summarise_pilot(
    trials: list[dict], design: dict, destination: Path
) -> dict:
    """Calculate descriptive means and the proposed validation record budget.

    Args:
        trials: All phase-0 path rows, indexed by condition/configuration/r.
        design: The predeclared finite main set and precision rule.
        destination: Pilot result directory.

    Returns:
        Planning metadata; no pilot observation becomes validation evidence.
    """
    summary, statistics, bundles = [], [], []
    configs = [c["configuration_id"] for c in design["configurations"]]
    for condition in design["conditions"]:
        cid = condition["joint_condition_id"]
        for config in configs:
            rows = [
                r
                for r in trials
                if r["joint_condition_id"] == cid
                and r["configuration"] == config
            ]
            data = np.array([r["nmse"] for r in rows])
            assert len(data) == 20
            mean, sd = float(np.mean(data)), float(np.std(data, ddof=1))
            record = {
                "joint_condition_id": cid,
                "configuration": config,
                "snr_db": condition["snr_db"],
                "probability": condition["probability"],
                "records": len(data),
                "mean_nmse": mean,
                "sd_nmse": sd,
                "mean_mae": float(np.mean([r["mae"] for r in rows])),
                "data_ber": sum(r["decoded_wrong_bits"] for r in rows)
                / (len(data) * 448000),
            }
            summary.append(record)
            statistics.append(
                {
                    "statistic": f"condition{cid}:{config}",
                    "mean": mean,
                    "sd": sd,
                }
            )
    main_ids = {
        c["joint_condition_id"] for c in design["conditions"] if c["primary"]
    }
    for repeat in range(20):
        for config in configs:
            values = [
                r["nmse"]
                for r in trials
                if r["record_id"] == repeat
                and r["configuration"] == config
                and r["joint_condition_id"] in main_ids
            ]
            assert len(values) == 12
            bundles.append(
                {
                    "record_id": repeat,
                    "configuration": config,
                    "objective": float(np.mean(values)),
                }
            )
    for config in configs:
        data = [
            r["objective"] for r in bundles if r["configuration"] == config
        ]
        statistics.append(
            {
                "statistic": f"main_objective:{config}",
                "mean": float(np.mean(data)),
                "sd": float(np.std(data, ddof=1)),
            }
        )
    assert len(statistics) == 92
    records = 200
    while True:
        critical = float(t.ppf(1 - 0.05 / (2 * 92), records - 1))
        if all(
            critical * r["sd"] / math.sqrt(records) <= 0.05 * r["mean"]
            for r in statistics
        ):
            break
        records += 50
    for row in statistics:
        half = critical * row["sd"] / math.sqrt(records)
        row.update(
            {
                "candidate_records": records,
                "adjusted_t_critical": critical,
                "predicted_half_width": half,
                "relative_half_width": half / row["mean"],
                "target_fraction": 0.05,
            }
        )
    write_csv(destination / "pilot_summary.csv", summary)
    write_csv(destination / "pilot_bundle_scores.csv", bundles)
    write_csv(destination / "precision_planning.csv", statistics)
    plan = {
        "status": "Pilot candidate; freeze final protocol before phase1",
        "pilot_complete_bundles": 20,
        "statistics": 92,
        "candidate_validation_records_per_condition": records,
        "rule": (
            "smallest multiple of50 >=200 with predicted adjusted "
            "half-width <=5% of all92 means"
        ),
        "maximum_predicted_relative_half_width": max(
            r["relative_half_width"] for r in statistics
        ),
        "limiting_pilot_statistic": max(
            statistics, key=lambda r: r["relative_half_width"]
        )["statistic"],
        "limitation": (
            "20-bank estimates can underestimate rare-event variance; "
            "precision planning, not a guarantee"
        ),
        "phase1_records_generated": 0,
    }
    (destination / "validation_planning.json").write_text(
        json.dumps(plan, indent=2) + "\n"
    )
    return plan
