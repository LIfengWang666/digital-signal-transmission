"""Apply the predeclared R3.4 whole-record quality decision."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import t

from stage2.r21.filtering_approach import write_csv
from stage3.r33.error_range import exact_interval, expectations


def mean_interval(values: list[float], family: int = 1) -> dict[str, float]:
    """Calculate an approximate t interval from independent record values.

    Args:
        values: One measured value per independent record.
        family: Number of predeclared means in the confidence family.

    Returns:
        Mean, sample SD, standard error, critical value and bounds.
    """
    data = np.asarray(values, dtype=float)
    if len(data) < 2 or family < 1 or not np.all(np.isfinite(data)):
        raise ValueError(
            "Use finite independent records and a positive family."
        )
    mean = float(np.mean(data))
    sd = float(np.std(data, ddof=1))
    se = sd / math.sqrt(len(data))
    critical = float(t.ppf(1 - 0.05 / (2 * family), len(data) - 1))
    return {
        "mean": mean,
        "sd": sd,
        "se": se,
        "critical": critical,
        "low": mean - critical * se,
        "high": mean + critical * se,
    }


def classify(low: float, high: float, threshold: float) -> str:
    """Use the same frozen threshold for both paths.

    Args:
        low: Mean interval lower bound.
        high: Mean interval upper bound.
        threshold: Maximum acceptable population mean NMSE.

    Returns:
        pass, fail or inconclusive.
    """
    if high <= threshold:
        return "pass"
    if low > threshold:
        return "fail"
    return "inconclusive"


def summarise(trials: list[dict], design: dict, destination: Path) -> dict:
    """Summarise validation without adjusting its frozen decision rule.

    Args:
        trials: Complete independent-record measurements.
        design: Prevalidation system, threshold and sampling freeze.
        destination: Results folder.

    Returns:
        Discrete coding-gain outcome and probability brackets.
    """
    summaries, comparisons, sensitivity = [], [], []
    threshold = design["primary_nmse_threshold"]
    family = design["confidence_family_size"]
    for condition in design["conditions"]:
        cid, p = condition["condition_id"], condition["probability"]
        for code in ["uncoded", "hamming74"]:
            rows = [
                r
                for r in trials
                if r["condition_id"] == cid and r["code"] == code
            ]
            assert len(rows) == design["records_per_condition"]
            values = [r["nmse_48000"] for r in rows]
            stats = mean_interval(values, family)
            wrong = sum(r["decoded_wrong_bits"] for r in rows)
            flips = sum(r["channel_flips"] for r in rows)
            data = sum(r["payload_bits"] for r in rows)
            tx = sum(r["transmitted_bits"] for r in rows)
            blocks = sum(r["block_count"] for r in rows)
            failed = sum(r["block_failures"] for r in rows)
            perfect = sum(r["decoded_wrong_bits"] == 0 for r in rows)
            perfect_low, perfect_high = exact_interval(perfect, len(rows))
            ber, q = expectations(p, code)
            block_count = rows[0]["block_count"]
            log_perfect = block_count * math.log1p(-q)
            summaries.append(
                {
                    "code": code,
                    "condition_id": cid,
                    "probability": p,
                    "role": condition["role"],
                    "records": len(rows),
                    "mean_nmse": stats["mean"],
                    "sd_nmse": stats["sd"],
                    "se_nmse": stats["se"],
                    "adjusted_ci_low": stats["low"],
                    "adjusted_ci_high": stats["high"],
                    "adjusted_critical": stats["critical"],
                    "status": classify(stats["low"], stats["high"], threshold),
                    "mean_mse": stats["mean"] * design["reference_power"],
                    "mean_prechannel_nmse": float(
                        np.mean([r["prechannel_nmse_48000"] for r in rows])
                    ),
                    "threshold": threshold,
                    "individual_waveform_passes": sum(
                        v <= threshold for v in values
                    ),
                    "payload_bits": data,
                    "transmitted_bits": tx,
                    "channel_flips": flips,
                    "wrong_data_bits": wrong,
                    "channel_ber": flips / tx,
                    "data_ber": wrong / data,
                    "theory_data_ber": ber,
                    "blocks": blocks,
                    "failed_blocks": failed,
                    "block_failure_fraction": failed / blocks,
                    "theory_block_failure_probability": q,
                    "perfect_payload_records": perfect,
                    "perfect_payload_fraction": perfect / len(rows),
                    "perfect_payload_ci95_low": perfect_low,
                    "perfect_payload_ci95_high": perfect_high,
                    "theory_perfect_payload_probability": math.exp(
                        log_perfect
                    ),
                    "theory_log_perfect_payload_probability": log_perfect,
                }
            )
        by_code = {
            name: sorted(
                [
                    r
                    for r in trials
                    if r["condition_id"] == cid and r["code"] == name
                ],
                key=lambda r: r["record_id"],
            )
            for name in ["uncoded", "hamming74"]
        }
        difference, gains = [], []
        for uncoded, coded in zip(by_code["uncoded"], by_code["hamming74"]):
            assert uncoded["record_id"] == coded["record_id"]
            difference.append(uncoded["nmse_48000"] - coded["nmse_48000"])
            gains.append(
                10 * math.log10(uncoded["mse_48000"] / coded["mse_48000"])
            )
        delta, gain = mean_interval(difference), mean_interval(gains)
        unc = next(
            s
            for s in summaries
            if s["condition_id"] == cid and s["code"] == "uncoded"
        )
        cod = next(
            s
            for s in summaries
            if s["condition_id"] == cid and s["code"] == "hamming74"
        )
        comparisons.append(
            {
                "condition_id": cid,
                "probability": p,
                "role": condition["role"],
                "uncoded_status": unc["status"],
                "coded_status": cod["status"],
                "coding_gain_established": int(
                    cod["status"] == "pass" and unc["status"] == "fail"
                ),
                "mean_paired_nmse_improvement": delta["mean"],
                "paired_nmse_difference_ci95_low": delta["low"],
                "paired_nmse_difference_ci95_high": delta["high"],
                "mean_paired_reconstruction_gain_db": gain["mean"],
                "paired_gain_ci95_low_db": gain["low"],
                "paired_gain_ci95_high_db": gain["high"],
            }
        )
        for factor in design["threshold_sensitivity_factors"]:
            tau = factor * design["design_baseline_nmse"]
            uncoded_status = classify(
                unc["adjusted_ci_low"], unc["adjusted_ci_high"], tau
            )
            coded_status = classify(
                cod["adjusted_ci_low"], cod["adjusted_ci_high"], tau
            )
            sensitivity.append(
                {
                    "budget_factor": factor,
                    "threshold": tau,
                    "condition_id": cid,
                    "probability": p,
                    "uncoded_status": uncoded_status,
                    "coded_status": coded_status,
                    "coding_gain_established": int(
                        coded_status == "pass" and uncoded_status == "fail"
                    ),
                }
            )
    summaries.sort(key=lambda s: (s["probability"], s["code"]))
    comparisons.sort(key=lambda s: s["probability"])
    sensitivity.sort(key=lambda s: (s["budget_factor"], s["probability"]))
    write_csv(destination / "summary.csv", summaries)
    write_csv(destination / "comparisons.csv", comparisons)
    write_csv(destination / "threshold_sensitivity.csv", sensitivity)
    success = [
        r["probability"] for r in comparisons if r["coding_gain_established"]
    ]
    brackets = {}
    for code in ["uncoded", "hamming74"]:
        points = sorted(
            [r for r in summaries if r["code"] == code],
            key=lambda r: r["probability"],
        )
        passed = [r["probability"] for r in points if r["status"] == "pass"]
        last_pass = max(passed) if passed else None
        later_failures = [
            r["probability"]
            for r in points
            if r["status"] == "fail"
            and (last_pass is None or r["probability"] > last_pass)
        ]
        next_failure = min(later_failures) if later_failures else None
        brackets[code] = {
            "largest_tested_pass": last_pass,
            "next_tested_fail_above": next_failure,
            "inconclusive_tested_points": [
                r["probability"]
                for r in points
                if r["status"] == "inconclusive"
            ],
            "continuous_crossing_claimed": False,
        }
    outcome = {
        "status": "R3.4 independent validation complete",
        "success_definition": "Mean waveform NMSE with adjusted interval",
        "primary_threshold": threshold,
        "gain_at_tested_probabilities": success,
        "threshold_brackets": brackets,
        "uncertainty": design["interval_scope"],
        "threshold_sensitivity": {
            str(factor): [
                r["probability"]
                for r in sensitivity
                if r["budget_factor"] == factor
                and r["coding_gain_established"]
            ]
            for factor in design["threshold_sensitivity_factors"]
        },
        "not_a_bit_perfect_or_every_record_success_claim": True,
        "not_an_Eb_N0_gain": True,
        "r4_complete": False,
    }
    (destination / "gain_summary.json").write_text(
        json.dumps(outcome, indent=2) + "\n"
    )
    return outcome
