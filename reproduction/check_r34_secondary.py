"""Independently check ancillary formulas without rerunning signal recovery."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import statistics
from pathlib import Path

from scipy.stats import t


def rows(path: Path) -> list[dict[str, str]]:
    """Read a CSV.

    Args:
        path: Measurement file.

    Returns:
        Header-keyed string rows.
    """
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def main(root: Path) -> None:
    """Check paired dB intervals and complete-payload probabilities.

    Args:
        root: Source package root containing measured R3.4 results.

    Returns:
        None. Write the independent secondary review.
    """
    folder = root / "stage3/r34/results"
    trials = rows(folder / "trials.csv")
    by_case = {
        (r["code"], int(r["condition_id"]), int(r["record_id"])): r
        for r in trials
    }
    maximum = 0.0
    for row in rows(folder / "comparisons.csv"):
        cid = int(row["condition_id"])
        deltas = [
            float(by_case[("uncoded", cid, r)]["nmse_48000"])
            - float(by_case[("hamming74", cid, r)]["nmse_48000"])
            for r in range(300)
        ]
        delta_mean = statistics.fmean(deltas)
        delta_half = (
            float(t.isf(0.025, 299))
            * statistics.stdev(deltas)
            / math.sqrt(300)
        )
        gain = [
            10
            * math.log10(
                float(by_case[("uncoded", cid, r)]["mse_48000"])
                / float(by_case[("hamming74", cid, r)]["mse_48000"])
            )
            for r in range(300)
        ]
        mean, sd = statistics.fmean(gain), statistics.stdev(gain)
        half = float(t.isf(0.025, 299)) * sd / math.sqrt(300)
        for actual, field in [
            (mean, "mean_paired_reconstruction_gain_db"),
            (mean - half, "paired_gain_ci95_low_db"),
            (mean + half, "paired_gain_ci95_high_db"),
            (delta_mean, "mean_paired_nmse_improvement"),
            (delta_mean - delta_half, "paired_nmse_difference_ci95_low"),
            (delta_mean + delta_half, "paired_nmse_difference_ci95_high"),
        ]:
            difference = abs(actual - float(row[field]))
            maximum = max(maximum, difference)
            assert difference < 1e-12
    parity = [(1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1)]
    words = []
    for message in itertools.product([0, 1], repeat=4):
        checks = tuple(
            sum(message[i] * parity[i][j] for i in range(4)) % 2
            for j in range(3)
        )
        words.append((message, message + checks))
    patterns = []
    for received in itertools.product([0, 1], repeat=7):
        distances = [
            sum(a != b for a, b in zip(received, codeword))
            for _, codeword in words
        ]
        assert min(distances) <= 1 and distances.count(min(distances)) == 1
        decoded = words[distances.index(min(distances))][0]
        patterns.append((sum(received), sum(decoded), any(decoded)))
    max_theory_error = 0.0
    for row in rows(folder / "summary.csv"):
        p = float(row["probability"])
        if row["code"] == "uncoded":
            ber, q, blocks = p, p, 448000
        else:
            ber = sum(
                wrong / 4 * p**weight * (1 - p) ** (7 - weight)
                for weight, wrong, _ in patterns
            )
            q = sum(
                p**weight * (1 - p) ** (7 - weight)
                for weight, _, failed in patterns
                if failed
            )
            blocks = 112000
        log_probability = blocks * math.log1p(-q)
        for actual, field in [
            (ber, "theory_data_ber"),
            (q, "theory_block_failure_probability"),
            (math.exp(log_probability), "theory_perfect_payload_probability"),
            (log_probability, "theory_log_perfect_payload_probability"),
        ]:
            expected = float(row[field])
            max_theory_error = max(max_theory_error, abs(actual - expected))
            assert math.isclose(actual, expected, abs_tol=1e-10, rel_tol=1e-11)
    review = {
        "date": "2026-10-21",
        "status": "passed",
        "paired_gain_conditions_checked": 15,
        "theory_conditions_checked": 30,
        "independent_theory_decoder": "Nearest of all 16 valid codewords",
        "maximum_paired_gain_statistic_difference": maximum,
        "maximum_theory_error_including_log_probability": max_theory_error,
        "all_complete_payload_probabilities_match": True,
    }
    (root / "metadata/r34_secondary_review.json").write_text(
        json.dumps(review, indent=2) + "\n"
    )
    print("R3.4 ancillary dB intervals and full-payload theory verified.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    main(parser.parse_args().root)
