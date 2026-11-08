"""Plan and assess R5 decisions using the prespecified19-statistic family."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import t

from stage2.r21.filtering_approach import write_csv
from stage4.r43.assessment import interval, read_rows

HERE = Path(__file__).resolve().parent


def primary_arrays(
    trials: list[dict], design: dict, records: int
) -> tuple[dict, list[dict]]:
    """Form paired objective/personal gains and3 constraint means.

    Args:
        trials: Complete measurement rows, with string or numeric values.
        design: Frozen candidates, main IDs and constraint subset.
        records: Independent complete bundles in this phase.

    Returns:
        Exactly19 primary arrays and descriptive objective score rows.
    """
    cases = {
        (
            int(r["record_id"]),
            int(r["joint_condition_id"]),
            r["variant"],
        ): float(r["nmse"])
        for r in trials
    }
    assert len(cases) == len(trials) == records * 15 * 9
    variants = [v["id"] for v in design["variants"]]
    objective = {
        name: np.array(
            [
                np.mean([cases[(r, cid, name)] for cid in design["main_ids"]])
                for r in range(records)
            ]
        )
        for name in variants
    }
    personal = {
        name: np.array([cases[(r, 11, name)] for r in range(records)])
        for name in variants
    }
    primary = {}
    for name in variants:
        if name == "base":
            continue
        primary[f"gain:J:{name}"] = objective["base"] - objective[name]
        primary[f"gain:personal:{name}"] = personal["base"] - personal[name]
    for name in ["base", "h1511", "uncoded"]:
        primary[f"mean:personal:{name}"] = personal[name]
    scores = [
        {
            "record_id": r,
            "variant": name,
            "objective": float(objective[name][r]),
        }
        for r in range(records)
        for name in variants
    ]
    assert len(primary) == 19
    return primary, scores


def plan_validation(trials: list[dict], design: dict, folder: Path) -> dict:
    """Choose the minimum declared multiple using only phase0 variation.

    Args:
        trials:20 complete pilot bundles, never confirmation observations.
        design: Pre-draw precision, candidate and family declaration.
        folder: Result directory for the freeze-able planning outputs.

    Returns:
        Planned record count and all19 predicted half-width checks.
    """
    arrays, _ = primary_arrays(trials, design, 20)
    count = 200
    while True:
        rows = []
        for name, values in arrays.items():
            mean, sd = float(np.mean(values)), float(np.std(values, ddof=1))
            target = max(0.01, 0.05 * abs(mean))
            half = float(
                t.ppf(1 - 0.05 / (2 * 19), count - 1) * sd / np.sqrt(count)
            )
            rows.append(
                {
                    "statistic": name,
                    "pilot_mean": mean,
                    "pilot_sd": sd,
                    "records": count,
                    "predicted_half_width": half,
                    "target": target,
                }
            )
        if all(r["predicted_half_width"] <= r["target"] for r in rows):
            break
        count += 50
    write_csv(folder / "phase0_primary.csv", rows)
    plan = {
        "records_per_condition": count,
        "primary_family": 19,
        "planning_uses_phase0_only": True,
        "optional_stopping": False,
        "target": (
            "max(.01,.05*abs(pilot mean)); a project precision target "
            "or precision guarantee"
        ),
        "previous_multiple_fails": count > 200,
        "source_scope": (
            "20 complete pilot bundles; no R4 observations imported"
        ),
    }
    (folder / "validation_planning.json").write_text(
        json.dumps(plan, indent=2) + "\n"
    )
    return plan


def assess_confirmation(folder: Path, frozen: dict) -> dict:
    """Summarise confirmed single-decision gains and descriptive diagnostics.

    Args:
        folder: Result directory containing new phase1 trial CSV.
        frozen: Pre-confirmation source/choice freeze and record count.

    Returns:
        Assessment metadata, without selecting the reflection conclusion.
        Nonnormal record means use explicitly approximate t/CLT intervals.
    """
    design, count = frozen["design"], frozen["records_per_condition"]
    trials = read_rows(folder / "phase1_trials.csv")
    primary, scores = primary_arrays(trials, design, count)
    summary = []
    for name, values in primary.items():
        stat = interval(values, family=19)
        pilot = next(
            r
            for r in read_rows(folder / "phase0_primary.csv")
            if r["statistic"] == name
        )
        summary.append(
            {
                "statistic": name,
                "records": count,
                **stat,
                "planning_target": float(pilot["target"]),
                "meets_planning_target": int(
                    stat["half_width"] <= float(pilot["target"])
                ),
            }
        )
    write_csv(folder / "primary_summary.csv", summary)
    write_csv(folder / "bundle_scores.csv", scores)
    case_summary = []
    for condition in design["conditions"]:
        cid = condition["joint_condition_id"]
        for variant in frozen["variants"]:
            name = variant["id"]
            rows = [
                r
                for r in trials
                if int(r["joint_condition_id"]) == cid and r["variant"] == name
            ]
            assert len(rows) == count
            case_summary.append(
                {
                    "joint_condition_id": cid,
                    "snr_db": condition["snr_db"],
                    "probability": condition["probability"],
                    "variant": name,
                    "records": count,
                    "mean_nmse": float(
                        np.mean([float(r["nmse"]) for r in rows])
                    ),
                    "mean_mae": float(
                        np.mean([float(r["mae"]) for r in rows])
                    ),
                    "mean_prechannel_mse": float(
                        np.mean([float(r["prechannel_mse"]) for r in rows])
                    ),
                    "mean_channel_power": float(
                        np.mean(
                            [float(r["channel_error_power"]) for r in rows]
                        )
                    ),
                    "mean_channel_cross": float(
                        np.mean([float(r["channel_cross"]) for r in rows])
                    ),
                    "decoded_wrong_bits": sum(
                        int(r["decoded_wrong_bits"]) for r in rows
                    ),
                    "real_payload_exposure": count * variant["payload_bits"],
                    "information_BER": sum(
                        int(r["decoded_wrong_bits"]) for r in rows
                    )
                    / (count * variant["payload_bits"]),
                    "perfect_messages": sum(
                        int(r["payload_perfect"]) for r in rows
                    ),
                    "message_success_fraction": sum(
                        int(r["payload_perfect"]) for r in rows
                    )
                    / count,
                }
            )
    write_csv(folder / "case_summary.csv", case_summary)
    decisions = []
    for variant in frozen["variants"]:
        name = variant["id"]
        js = [r["objective"] for r in scores if r["variant"] == name]
        gain = next(
            (r for r in summary if r["statistic"] == f"gain:J:{name}"), None
        )
        personal_gain = next(
            (r for r in summary if r["statistic"] == f"gain:personal:{name}"),
            None,
        )
        decisions.append(
            {
                **variant,
                "mean_objective": float(np.mean(js)),
                "main_gain": 0 if gain is None else gain["mean"],
                "main_gain_lower": 0 if gain is None else gain["lower"],
                "main_gain_upper": 0 if gain is None else gain["upper"],
                "personal_gain": (
                    0 if personal_gain is None else personal_gain["mean"]
                ),
                "personal_gain_lower": (
                    0 if personal_gain is None else personal_gain["lower"]
                ),
                "personal_gain_upper": (
                    0 if personal_gain is None else personal_gain["upper"]
                ),
            }
        )
    write_csv(folder / "decision_evidence.csv", decisions)
    result = {
        "status": "complete",
        "records": count,
        "path_rows": len(trials),
        "primary_family": 19,
        "statistics_above_planned_precision": [
            r["statistic"] for r in summary if not r["meets_planning_target"]
        ],
        "maximum_power_identity_residual": max(
            float(r["power_identity_residual"]) for r in trials
        ),
        "limiting_decision_selected_by_this_aggregation": None,
        "R52_constraint_assessment_pending": True,
        "phase0_rows_imported_into_confirmation": 0,
        "coverage": (
            "Approximate19-family t/CLT mean intervals; "
            "diagnostic means/counts descriptive"
        ),
    }
    (folder / "assessment.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    return result
