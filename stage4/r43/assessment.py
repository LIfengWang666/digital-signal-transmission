"""Summarise held-out NMSE with the predeclared finite-family intervals."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import t

from stage2.r21.filtering_approach import write_csv
from stage4.r42.pipeline import paired_effects
from stage4.r43.protocol import freeze_measurement

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
CONFIGS = ["00", "10", "01", "11"]
CONTRASTS = {
    "filter_without_code": {"00": 1, "10": -1},
    "filter_with_code": {"01": 1, "11": -1},
    "code_without_filter": {"00": 1, "01": -1},
    "code_with_filter": {"10": 1, "11": -1},
    "interaction": {"00": 1, "10": -1, "01": -1, "11": 1},
}


def read_rows(path: Path) -> list[dict[str, str]]:
    """Read a measurement CSV, preserving configuration IDs with leading zero.

    Args:
        path: UTF-8 measurement file.

    Returns:
        Rows with their original column names and string values.
    """
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def interval(values: np.ndarray, family: int = 92) -> dict[str, float]:
    """Calculate the declared sample mean, SD and adjusted-t limits.

    Args:
        values: Independent record or complete-bundle measurements.
        family: Number of means covered;1 means a pointwise95% interval.

    Returns:
        Mean, ddof1 SD, standard error, half-width and unmodified limits.
        Coverage uses the t/CLT approximation for nonnormal record metrics.
    """
    count = len(values)
    if count < 2:
        raise ValueError(
            "Mean intervals require independent repeated records."
        )
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1))
    se = sd / np.sqrt(count)
    half = float(t.ppf(1 - 0.05 / (2 * family), count - 1) * se)
    return {
        "mean": mean,
        "sd": sd,
        "se": float(se),
        "half_width": half,
        "lower": mean - half,
        "upper": mean + half,
    }


def contrast_rows(
    cid: int, arrays: dict[str, np.ndarray], bounds: dict[str, dict]
) -> list[dict]:
    """Form conditional effects and honest simultaneous/pointwise bounds.

    Args:
        cid: Joint condition ID, or -1 for the12-condition objective.
        arrays: Four matched record-level NMSE or bundle-objective arrays.
        bounds: Their92-family adjusted mean limits.

    Returns:
        Five contrasts; conservative simultaneous bounds use linear
        combinations of primary limits, paired t limits are supplementary.
    """
    rows = []
    for effect, coefficients in CONTRASTS.items():
        values = sum(
            coef * arrays[config] for config, coef in coefficients.items()
        )
        point = interval(values, family=1)
        lower = sum(
            coef * bounds[config]["lower" if coef > 0 else "upper"]
            for config, coef in coefficients.items()
        )
        upper = sum(
            coef * bounds[config]["upper" if coef > 0 else "lower"]
            for config, coef in coefficients.items()
        )
        rows.append(
            {
                "joint_condition_id": cid,
                "effect": effect,
                "records": len(values),
                "mean_reduction": point["mean"],
                "sd_paired": point["sd"],
                "simultaneous_lower": lower,
                "simultaneous_upper": upper,
                "paired_pointwise95_lower": point["lower"],
                "paired_pointwise95_upper": point["upper"],
                "simultaneous_direction": (
                    "positive"
                    if lower > 0
                    else "negative" if upper < 0 else "unresolved"
                ),
            }
        )
    return rows


def assess(folder: Path, choices: dict, records: int) -> dict:
    """Aggregate raw measurements without retuning weights or record count.

    Args:
        folder: R4.3 measurement/result directory.
        choices: Frozen conditions, configurations and main weights.
        records: Predeclared number of independent complete bundles.

    Returns:
        Assessment manifest. All derived tables are written from actual CSV;
        pilot observations are excluded and nonpositive limits are retained.
    """
    trials = read_rows(folder / "trials.csv")
    main_ids = sorted(
        c["joint_condition_id"] for c in choices["conditions"] if c["primary"]
    )
    assert len(main_ids) == 12 and len(trials) == records * 88
    indexed = {
        (
            int(r["record_id"]),
            int(r["joint_condition_id"]),
            r["configuration"],
        ): r
        for r in trials
    }
    assert len(indexed) == len(trials)
    summary, arrays, bounds = [], {}, {}
    for condition in choices["conditions"]:
        cid = condition["joint_condition_id"]
        arrays[cid], bounds[cid] = {}, {}
        for config in CONFIGS:
            rows = [indexed[(r, cid, config)] for r in range(records)]
            values = np.array([float(r["nmse"]) for r in rows])
            stat = interval(values)
            arrays[cid][config], bounds[cid][config] = values, stat
            summary.append(
                {
                    "joint_condition_id": cid,
                    "configuration": config,
                    "snr_db": condition["snr_db"],
                    "probability": condition["probability"],
                    "primary": int(condition["primary"]),
                    "role": condition["role"],
                    "records": records,
                    **{f"{k}_nmse": v for k, v in stat.items()},
                    "relative_half_width": stat["half_width"] / stat["mean"],
                    "mean_mse": float(
                        np.mean([float(r["mse"]) for r in rows])
                    ),
                    "mean_mae": float(
                        np.mean([float(r["mae"]) for r in rows])
                    ),
                    "mean_peak_absolute_error": float(
                        np.mean(
                            [float(r["peak_absolute_error"]) for r in rows]
                        )
                    ),
                    "mean_channel_flips": float(
                        np.mean([int(r["channel_flips"]) for r in rows])
                    ),
                    "mean_decoded_wrong_bits": float(
                        np.mean([int(r["decoded_wrong_bits"]) for r in rows])
                    ),
                    "information_BER": sum(
                        int(r["decoded_wrong_bits"]) for r in rows
                    )
                    / (records * 448000),
                    "bit_perfect_fraction": sum(
                        int(r["decoded_wrong_bits"]) == 0 for r in rows
                    )
                    / records,
                    "mean_wrong_samples": float(
                        np.mean([int(r["wrong_samples"]) for r in rows])
                    ),
                    "mean_wrong_msb_bits": float(
                        np.mean([int(r["wrong_msb_bits"]) for r in rows])
                    ),
                    "mean_prechannel_nmse": float(
                        np.mean([float(r["prechannel_nmse"]) for r in rows])
                    ),
                    "error_ratio_db_from_mean_nmse": float(
                        -10 * np.log10(stat["mean"])
                    ),
                }
            )
    write_csv(folder / "summary.csv", summary)
    objective_arrays = {
        config: np.mean([arrays[cid][config] for cid in main_ids], axis=0)
        for config in CONFIGS
    }
    objective_bounds = {
        config: interval(v) for config, v in objective_arrays.items()
    }
    write_csv(
        folder / "bundle_scores.csv",
        [
            {
                "record_id": r,
                "configuration": config,
                "objective": float(objective_arrays[config][r]),
                "unique_conditions": 12,
            }
            for r in range(records)
            for config in CONFIGS
        ],
    )
    objectives = [
        {
            "configuration": config,
            "records": records,
            "main_unique_conditions": 12,
            **{
                f"{k}_objective": v
                for k, v in objective_bounds[config].items()
            },
            "relative_half_width": objective_bounds[config]["half_width"]
            / objective_bounds[config]["mean"],
        }
        for config in CONFIGS
    ]
    write_csv(folder / "objective_summary.csv", objectives)
    effects = [
        row
        for cid in arrays
        for row in contrast_rows(cid, arrays[cid], bounds[cid])
    ]
    effects += contrast_rows(-1, objective_arrays, objective_bounds)
    write_csv(folder / "effect_summary.csv", effects)
    stored_effects = read_rows(folder / "effects.csv")
    assert len(stored_effects) == records * 22
    for row in stored_effects:
        r, cid = int(row["record_id"]), int(row["joint_condition_id"])
        expected = paired_effects(
            {config: arrays[cid][config][r] for config in CONFIGS}
        )
        assert all(
            np.isclose(float(row[k]), v, rtol=1e-12, atol=1e-13)
            for k, v in expected.items()
        )
    decomposition = read_rows(folder / "decomposition.csv")
    components = [
        k
        for k in decomposition[0]
        if k.startswith("power_") or k.startswith("cross_")
    ]
    component_rows = []
    for condition in choices["conditions"]:
        cid = condition["joint_condition_id"]
        for config in CONFIGS:
            rows = [
                r
                for r in decomposition
                if int(r["joint_condition_id"]) == cid
                and r["configuration"] == config
            ]
            component_rows.append(
                {
                    "joint_condition_id": cid,
                    "configuration": config,
                    **{
                        k: float(np.mean([float(r[k]) for r in rows]))
                        for k in components
                    },
                    "mean_mse_from_components": float(
                        np.mean(
                            [float(r["mse_from_components"]) for r in rows]
                        )
                    ),
                }
            )
    write_csv(folder / "component_summary.csv", component_rows)
    metric_rows = []
    for cid in arrays:
        rows = [r for r in summary if r["joint_condition_id"] == cid]
        for row in rows:
            metric_rows.append(
                {
                    "joint_condition_id": cid,
                    "configuration": row["configuration"],
                    "mean_nmse": row["mean_nmse"],
                    "mean_mae": row["mean_mae"],
                    "rank_nmse": 1
                    + sum(
                        other["mean_nmse"] < row["mean_nmse"] for other in rows
                    ),
                    "rank_mae": 1
                    + sum(
                        other["mean_mae"] < row["mean_mae"] for other in rows
                    ),
                }
            )
    write_csv(folder / "metric_comparison.csv", metric_rows)
    quantisers = read_rows(folder / "quantiser_banks.csv")
    assert len(quantisers) == 16 * records
    write_csv(
        folder / "overload_summary.csv",
        [
            {
                "snr_condition_id": snr["snr_condition_id"],
                "snr_db": snr["snr_db"],
                "filter_enabled": f,
                "unique_records": records,
                "unique_input_samples": records * 32000,
                "overload_samples": sum(
                    int(q["overload_samples"])
                    for q in quantisers
                    if int(q["snr_condition_id"]) == snr["snr_condition_id"]
                    and int(q["filter_enabled"]) == f
                ),
            }
            for snr in choices["snr_conditions"]
            for f in [0, 1]
        ],
    )
    write_csv(
        folder / "costs.csv",
        [
            {
                "configuration": c["configuration_id"],
                "name": c["name"],
                "sample_rate_hz": 16000,
                "bits_per_sample": 14,
                "filter_taps": 385 if c["filter_enabled"] else 0,
                "filter_group_delay_ms": 12 if c["filter_enabled"] else 0,
                "payload_bits": 448000,
                "transmitted_bits": c["transmitted_bits"],
                "logical_rate_bits_s": c["logical_rate_bits_s"],
                "extra_transmitted_fraction": c["transmitted_bits"] / 448000
                - 1,
            }
            for c in choices["configurations"]
        ],
    )
    precision = [
        {
            "statistic": (
                f"condition:{r['joint_condition_id']}:{r['configuration']}"
            ),
            "relative_half_width": r["relative_half_width"],
        }
        for r in summary
    ]
    precision += [
        {
            "statistic": f"objective:{r['configuration']}",
            "relative_half_width": r["relative_half_width"],
        }
        for r in objectives
    ]
    write_csv(
        folder / "precision_audit.csv",
        [
            {**r, "meets_5percent": int(r["relative_half_width"] <= 0.05)}
            for r in precision
        ],
    )
    result = {
        "status": "complete",
        "phase_id": 1,
        "complete_bundles": records,
        "path_rows": len(trials),
        "primary_mean_family": 92,
        "t_critical": float(t.ppf(1 - 0.05 / (2 * 92), records - 1)),
        "all92_actual_relative_half_widths_le_5percent": all(
            r["relative_half_width"] <= 0.05 for r in precision
        ),
        "statistics_above_planning_target": [
            r for r in precision if r["relative_half_width"] > 0.05
        ],
        "maximum_actual_relative_half_width": max(
            r["relative_half_width"] for r in precision
        ),
        "record_count_changed_after_outcomes": False,
        "phase0_records_imported": 0,
        "maximum_mse_decomposition_residual": max(
            float(r["mse_residual"]) for r in decomposition
        ),
        "maximum_waveform_identity_residual": max(
            float(r["waveform_identity_residual"]) for r in decomposition
        ),
        "maximum_nyquist_energy_residual": max(
            float(r["nyquist_energy_residual"]) for r in decomposition
        ),
        "descriptive_MAE_and_bit_counts_not_in92_family": True,
        "supported_negative_conditional_effects": [
            r
            for r in effects
            if r["simultaneous_direction"] == "negative"
            and r["effect"] != "interaction"
        ],
        "coverage_limitation": (
            "t/CLT approximate finite-family mean coverage; no exact "
            "finite-sample or continuous-band guarantee"
        ),
    }
    (folder / "assessment.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    return result


def main() -> None:
    """Verify the source freeze and summarise exactly the held-out records.

    Returns:
        None. Write report-ready tables without consuming any RNG stream.
    """
    frozen = freeze_measurement()
    result = assess(HERE / "results", frozen["choices"], 1400)
    print(
        "R4.3: 92 primary means; maximum relative half-width="
        f"{result['maximum_actual_relative_half_width']:.4%}"
    )


if __name__ == "__main__":
    main()
