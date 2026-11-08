"""Independently verify metric meaning and all saved R4.1 evidence rows."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from stage3.bitstream import bits_to_labels, labels_to_bits
from stage3.r31.quantisation import dequantise
from stage4.r41.metrics import db_from_nmse, reconstruction_metrics

ROOT = Path(__file__).resolve().parents[1]


def scalar_metrics(x: list[float], y: list[float]) -> dict[str, float]:
    """Use scalar accurate sums, independent of the NumPy mean implementation.

    Args:
        x: Complete reference samples.
        y: Complete reconstruction on the same grid.

    Returns:
        Independent values for every numeric metric saved by R4.1.
    """
    n = len(x)
    e = [b - a for a, b in zip(x, y)]
    power = math.fsum(a * a for a in x) / n
    mse = math.fsum(a * a for a in e) / n
    bias = math.fsum(e) / n
    mx, my = math.fsum(x) / n, math.fsum(y) / n
    xc, yc = [a - mx for a in x], [a - my for a in y]
    vx = math.fsum(a * a for a in xc) / n
    vy = math.fsum(a * a for a in yc) / n
    rho = math.nan
    if vx > 0 and vy > 0:
        rho = math.fsum(a * b for a, b in zip(xc, yc)) / n
        rho /= math.sqrt(vx) * math.sqrt(vy)
        rho = min(1.0, max(-1.0, rho))
    return {
        "samples": n,
        "reference_power": power,
        "mse": mse,
        "nmse": mse / power,
        "mae": math.fsum(abs(a) for a in e) / n,
        "rmse": math.sqrt(mse),
        "mean_error": bias,
        "error_variance_ddof0": math.fsum((a - bias) ** 2 for a in e) / n,
        "peak_absolute_error": max(abs(a) for a in e),
        "reconstruction_error_ratio_db": (
            math.inf if mse == 0 else 10 * math.log10(power / mse)
        ),
        "correlation": rho,
    }


def assert_metrics(row: dict[str, str], expected: dict[str, float]) -> int:
    """Check scalar oracle values, including honest infinity/NaN semantics.

    Args:
        row: One numeric CSV result with named metrics.
        expected: Independently computed quantities.

    Returns:
        Number of verified numeric fields, raising on any mismatch.
    """
    for key, value in expected.items():
        actual = float(row[key])
        if math.isnan(value):
            assert math.isnan(actual), (row["case"], key)
        elif math.isinf(value):
            assert actual == value, (row["case"], key)
        else:
            assert math.isclose(actual, value, rel_tol=3e-12, abs_tol=2e-14), (
                row["case"],
                key,
                actual,
                value,
            )
    assert float(row["mae"]) <= float(row["rmse"]) + 2e-14
    assert math.isclose(
        float(row["mse"]),
        float(row["error_variance_ddof0"]) + float(row["mean_error"]) ** 2,
        rel_tol=3e-12,
        abs_tol=2e-14,
    )
    return len(expected) + 2


def main() -> None:
    """Verify all evidence, closed-form cases and anti-broadcasting guards.

    Returns:
        None. Save an independent check summary or raise on a discrepancy.
    """
    folder = ROOT / "stage4/r41/results"
    reference = np.load(ROOT / "reference_signal.npy").tolist()
    n = len(reference)
    checks = 0
    with (folder / "semantic_checks.csv").open(
        newline="", encoding="utf-8"
    ) as f:
        rows = {r["case"]: r for r in csv.DictReader(f)}
    for name, row in rows.items():
        if name == "exact":
            y = reference.copy()
        elif name == "double_gain":
            y = [2 * a for a in reference]
        elif name == "offset_0.1":
            y = [a + 0.1 for a in reference]
        elif name == "diffuse_0.1":
            y = [
                a + (0.1 if i % 2 == 0 else -0.1)
                for i, a in enumerate(reference)
            ]
        elif name == "sparse_2_at_1percent":
            y = [
                a + (2 * (-1) ** (i // 100) if i % 100 == 0 else 0)
                for i, a in enumerate(reference)
            ]
        elif name == "zero_output":
            y = [0.0] * n
        elif name == "circular_delay_1sample":
            y = reference[-1:] + reference[:-1]
        else:
            raise AssertionError(name)
        checks += assert_metrics(row, scalar_metrics(reference, y))
    assert float(rows["exact"]["nmse"]) == 0
    assert float(rows["double_gain"]["nmse"]) == 1
    assert float(rows["double_gain"]["correlation"]) == 1
    assert math.isclose(float(rows["offset_0.1"]["mse"]), 0.01, abs_tol=2e-16)
    assert float(rows["offset_0.1"]["error_variance_ddof0"]) < 1e-28
    assert math.isclose(float(rows["diffuse_0.1"]["mse"]), 0.01, abs_tol=2e-16)
    assert math.isclose(
        float(rows["sparse_2_at_1percent"]["mse"]), 0.04, abs_tol=2e-16
    )
    assert float(rows["sparse_2_at_1percent"]["mae"]) < float(
        rows["diffuse_0.1"]["mae"]
    )
    assert float(rows["sparse_2_at_1percent"]["nmse"]) > float(
        rows["diffuse_0.1"]["nmse"]
    )
    checks += 9
    with (folder / "existing_record_metrics.csv").open(
        newline="", encoding="utf-8"
    ) as f:
        legacy = list(csv.DictReader(f))
    with (ROOT / "stage3/r34/results/trials.csv").open(
        newline="", encoding="utf-8"
    ) as f:
        old_trials = list(csv.DictReader(f))
    cache = {}
    for row in legacy:
        source = row["source_file"]
        if source not in cache:
            cache[source] = np.loadtxt(
                ROOT / source, delimiter=",", skiprows=1
            )
        column = {"before_bsc": 3, "uncoded": 4, "hamming74": 5}[row["code"]]
        y = cache[source][:, column].tolist()
        checks += assert_metrics(row, scalar_metrics(reference, y))
        if row["code"] != "before_bsc":
            old = next(
                r
                for r in old_trials
                if r["code"] == row["code"]
                and r["condition_id"] == row["condition_id"]
                and r["record_id"] == row["record_id"]
            )
            assert row["decoded_wrong_bits"] == old["decoded_wrong_bits"]
            for key in ["mse", "nmse"]:
                assert math.isclose(
                    float(row[key]),
                    float(old[f"{key}_48000"]),
                    rel_tol=3e-12,
                    abs_tol=2e-14,
                )
            checks += 3
    personal = next(r for r in legacy if r["case"] == "personal_hamming74")
    before = next(r for r in legacy if r["case"] == "prechannel")
    assert int(personal["decoded_wrong_bits"]) == 0
    assert float(personal["nmse"]) == float(before["nmse"]) > 0
    checks += 2
    with (folder / "aggregation_example.csv").open(
        newline="", encoding="utf-8"
    ) as f:
        aggregation = next(csv.DictReader(f))
    assert math.isclose(
        float(aggregation["mean_nmse"]), 0.025 / 0.73, rel_tol=2e-14
    )
    assert math.isclose(
        float(aggregation["db_from_mean_nmse"]),
        10 * math.log10(0.73 / 0.025),
        abs_tol=2e-13,
    )
    assert math.isclose(
        float(aggregation["mean_of_record_db"]),
        10 * math.log10(0.73 / 0.02),
        abs_tol=2e-13,
    )
    assert float(aggregation["db_from_mean_nmse"]) < float(
        aggregation["mean_of_record_db"]
    )
    checks += 4
    for label in [0, 2**14 - 1]:
        indices = np.array([label], dtype=np.uint32)
        original = labels_to_bits(indices, 14)
        for u in range(14):
            changed = original.copy()
            changed[u] ^= 1
            decoded = bits_to_labels(changed, 14)
            delta = float(dequantise(decoded, 14, 18)[0])
            delta -= float(dequantise(indices, 14, 18)[0])
            expected = (
                36 / (2**14 - 1) * (1 - 2 * int(original[u])) * 2 ** (13 - u)
            )
            assert math.isclose(delta, expected, rel_tol=1e-11, abs_tol=1e-14)
            checks += 1
    invalid_pairs = [
        (np.array([]), np.array([])),
        (np.ones(3), np.ones(4)),
        (np.ones(3), np.ones((3, 1))),
        (np.zeros(3), np.ones(3)),
        (np.ones(3), np.array([1.0, math.nan, 0.0])),
        (np.array([1.0, math.inf]), np.ones(2)),
        (np.ones(2), np.array([1j, 2j])),
    ]
    for a, b in invalid_pairs:
        try:
            reconstruction_metrics(a, b)
        except ValueError:
            checks += 1
        else:
            raise AssertionError("Invalid metric input was accepted.")
    for value in [-0.1, math.inf, math.nan]:
        try:
            db_from_nmse(value)
        except ValueError:
            checks += 1
        else:
            raise AssertionError("Invalid NMSE was accepted.")
    result = {
        "status": "passed",
        "independent_assertions": checks,
        "semantic_rows": len(rows),
        "existing_full_record_rows": len(legacy),
        "oracle": "scalar math.fsum with separate sample construction/loading",
        "legacy_four_received_examples_match_R3_4": True,
        "single_bit_amplitude_relation_checks": 28,
        "no_new_random_records": True,
    }
    (folder / "independent_checks.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(f"R4.1: {checks} independent assertions passed.")


if __name__ == "__main__":
    main()
