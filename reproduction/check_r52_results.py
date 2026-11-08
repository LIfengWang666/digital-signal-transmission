"""Independently audit R5.2 costs, decisions and true-payload probability."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from statistics import fmean, stdev

import numpy as np
from scipy.stats import beta, t

from reproduction.check_r32_results import independent_tables
from stage4.r43.assessment import read_rows

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "stage5/results"


def close(actual: float, expected: float) -> None:
    """Require numerical agreement appropriate for independently summed data.

    Args:
        actual: Saved or computed measurement.
        expected: Independently computed reference.

    Returns:
        None. Raise if absolute/relative floating-point tolerance is exceeded.
    """
    assert math.isclose(actual, expected, rel_tol=2e-11, abs_tol=2e-13), (
        actual,
        expected,
    )


def main() -> None:
    """Check every R5.2 table without generating any new random observations.

    Returns:
        None. Save independent evidence or raise on a wrong cost/decision.
    """
    result = json.loads((FOLDER / "constraint_assessment.json").read_text())
    for name, expected in result["source_sha256"].items():
        assert (
            hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected
        )
    complete = json.loads(
        (ROOT / "stage5/results/phase1_independent_checks.json").read_text()
    )
    assert complete["status"] == "passed"
    assert result["additional_independent_records"] == 0
    assert result["records_reused_from_R5_confirmation"] == 350
    threshold, cap = result["NMSE_threshold"], result["maximum_bits_s"]
    rule = json.loads((ROOT / "stage5/design.json").read_text())[
        "R52_prespecified_constraint"
    ]
    assert threshold == rule["primary_NMSE_threshold"]
    assert cap == rule["maximum_bits_s"] == 320000
    rows = read_rows(FOLDER / "constraint_tradeoff.csv")
    expected_costs = {
        "base": (448000, 0, 784000),
        "h1511": (448000, 8, 610920),
        "uncoded": (448000, 0, 448000),
        "repeat5": (448000, 0, 2240000),
        "bits11_h74": (352000, 0, 616000),
    }
    trials = [
        r
        for r in read_rows(FOLDER / "phase1_trials.csv")
        if r["joint_condition_id"] == "11"
    ]
    selected = []
    for row in rows:
        name = row["variant"]
        cost = tuple(
            int(row[key])
            for key in ["payload_bits", "padding_bits", "transmitted_bits"]
        )
        assert cost == expected_costs[name]
        rate = cost[2] / 2
        close(float(row["logical_rate_bits_s"]), rate)
        assert int(row["rate_feasible"]) == int(rate <= cap)
        assert int(row["quantisation_budget_feasible"]) == int(
            name != "bits11_h74"
        )
        if name not in {"base", "h1511", "uncoded"}:
            assert row["mean_nmse"] == row["mean_upper"] == ""
            continue
        subset = [r for r in trials if r["variant"] == name]
        assert len(subset) == 350
        assert len({r["record_id"] for r in subset}) == 350
        values = [float(r["nmse"]) for r in subset]
        mean = fmean(values)
        half = t.ppf(1 - 0.05 / 38, 349) * stdev(values) / math.sqrt(350)
        for key, value in [
            ("mean_nmse", mean),
            ("mean_half_width", half),
            ("mean_lower", mean - half),
            ("mean_upper", mean + half),
        ]:
            close(float(row[key]), value)
        wrong = sum(int(r["decoded_wrong_bits"]) for r in subset)
        perfect = sum(int(r["payload_perfect"]) for r in subset)
        close(float(row["information_BER"]), wrong / (350 * cost[0]))
        assert int(row["perfect_messages"]) == perfect
        verdict = (
            "pass"
            if mean + half <= threshold
            else "fail" if mean - half > threshold else "unresolved"
        )
        assert row["mean_quality_result"] == verdict
        assert int(row["mean_quality_passes"]) == int(verdict == "pass")
        if verdict == "pass" and rate <= cap:
            selected.append(name)
    assert selected == [result["selected_variant"]] == ["h1511"]
    prefix_rows = read_rows(FOLDER / "last_block_prefix_counts.csv")
    definitions = json.loads(
        (ROOT / "stage3/r32/results/code_definitions.json").read_text()
    )
    specs = {r["code"]: r for r in definitions}
    theoretical = read_rows(FOLDER / "payload_recovery_theory.csv")
    pattern_checks, max_probability_error = 0, 0.0
    for row in theoretical:
        code = row["code"]
        spec = specs[code]
        _, table = independent_tables(spec)
        n, k, radius = spec["n"], spec["k"], spec["t"]
        payload = int(row["payload_bits"])
        blocks, pad = (payload + k - 1) // k, (-payload) % k
        last = k - pad
        assert int(row["blocks"]) == blocks
        assert int(row["padding_bits"]) == pad
        assert int(row["last_real_bits"]) == last
        p = float(row["probability"])
        counts = [0] * (n + 1)
        for word in range(2**n):
            if not np.any(table[word, :last]):
                counts[word.bit_count()] += 1
        saved = [r for r in prefix_rows if r["code"] == code]
        assert len(saved) == n + 1
        assert [
            int(r["prefix_correct_error_patterns"]) for r in saved
        ] == counts
        pattern_checks += 2**n
        # A perfect decoder's complete-message success is its zero sphere.
        q = math.fsum(
            math.comb(n, w) * p**w * (1 - p) ** (n - w)
            for w in range(radius + 1, n + 1)
        )
        last_ok = math.fsum(
            count * p**w * (1 - p) ** (n - w) for w, count in enumerate(counts)
        )
        probability = math.exp((blocks - 1) * math.log1p(-q)) * last_ok
        difference = abs(
            probability - float(row["theoretical_payload_success"])
        )
        max_probability_error = max(max_probability_error, difference)
        close(float(row["block_failure_probability"]), q)
        close(float(row["last_prefix_success_probability"]), last_ok)
        close(float(row["theoretical_payload_success"]), probability)
        success, records = int(row["perfect_messages"]), int(row["records"])
        cost_row = next(r for r in rows if r["variant"] == row["variant"])
        assert success == int(cost_row["perfect_messages"])
        assert records == int(cost_row["records"]) == 350
        lower = 0 if success == 0 else beta.ppf(0.025, success, 351 - success)
        upper = (
            1
            if success == 350
            else beta.ppf(0.975, success + 1, 350 - success)
        )
        close(float(row["measured_fraction"]), success / records)
        close(float(row["pointwise95_lower"]), lower)
        close(float(row["pointwise95_upper"]), upper)
    bit11 = next(
        r
        for r in read_rows(ROOT / "stage3/r31/results/bit_selection.csv")
        if r["bits"] == "11"
    )
    close(
        result["bits11_worst_quantisation_component_ratio"],
        float(bit11["worst_component_ratio"]),
    )
    assert float(bit11["worst_component_ratio"]) > 0.01
    checked = {
        "status": "passed",
        "cost_rows_verified": 5,
        "personal_path_rows_reused_and_verified": 1050,
        "new_random_observations": 0,
        "selected_variant": "h1511",
        "error_patterns_independently_checked": pattern_checks,
        "last_partial_word_verified": "Hamming15:3 real bits,8 pad bits",
        "maximum_probability_difference": max_probability_error,
        "Clopper_Pearson_checked_by": "independent beta quantiles",
        "mean_intervals_checked_by": "scalar fmean/stdev,19-family t",
        "checker_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "constraint_assessment_sha256": hashlib.sha256(
            (FOLDER / "constraint_assessment.json").read_bytes()
        ).hexdigest(),
    }
    (FOLDER / "r52_independent_checks.json").write_text(
        json.dumps(checked, indent=2) + "\n"
    )
    print("R5.2 independent audit passed: costs, means and all32898 patterns.")


if __name__ == "__main__":
    main()
