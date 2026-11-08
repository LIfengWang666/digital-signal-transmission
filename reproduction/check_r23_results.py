"""Independently recompute R2.3 fidelity, uncertainty and replay evidence."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.signal import resample
from scipy.stats import sem, t

ROOT = Path(__file__).resolve().parents[1]


def read_rows(path: Path) -> list[dict[str, str]]:
    """Read the published data without importing the measurement module.

    Args:
        path: A saved CSV with headers.

    Returns:
        Header-keyed string records.
    """
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    """Check published measurements through independent numerical paths.

    Returns:
        None. Write a compact audit; raise on inconsistency.
    """
    results = ROOT / "stage2/r23/results"
    manifest = json.loads(
        (results / "seed_manifest.json").read_text(encoding="utf-8")
    )
    trials = read_rows(results / "paired_trials.csv")
    summaries = read_rows(results / "summary.csv")
    h = np.load(ROOT / "stage2/r22/results/selected_coefficients.npy")
    x = np.load(ROOT / "stage1/r12/results/selected_samples.npy")
    original = np.load(ROOT / "reference_signal.npy")
    replayed = 0
    for run in manifest["runs"]:
        sequence = np.random.SeedSequence(
            manifest["root_seed"],
            spawn_key=tuple(run["spawn_key"]),
            pool_size=manifest["pool_size"],
        )
        rng = np.random.Generator(np.random.PCG64(sequence))
        if rng.bit_generator.state != run["initial_state"]:
            raise AssertionError("The recorded initial RNG state changed.")
        # Directly use the specified distribution, without degradation.py.
        e = rng.normal(0.0, np.sqrt(run["variance"]), x.shape)
        y = x + e
        for values, key in [
            (e, "noise_values_sha256"),
            (y, "contaminated_values_sha256"),
        ]:
            actual = hashlib.sha256(
                np.asarray(values, dtype="<f8").tobytes()
            ).hexdigest()
            if actual != run[key]:
                raise AssertionError("A full-record draw did not replay.")
        replayed += 1
    max_summary_difference = 0.0
    max_partition_difference = 0.0
    max_gain_difference = 0.0
    for row in trials:
        rate = int(row["rate_hz"])
        q, count = (256, 32000) if rate == 16000 else (768, 96000)
        combined = (2 * q / count) * float(row["filter_edge_mse"]) + (
            1 - 2 * q / count
        ) * float(row["filter_interior_mse"])
        max_partition_difference = max(
            max_partition_difference, abs(combined - float(row["filter_mse"]))
        )
        gain = 10 * np.log10(
            float(row["no_filter_mse"]) / float(row["filter_mse"])
        )
        max_gain_difference = max(
            max_gain_difference, abs(float(row["paired_gain_db"]) - gain)
        )
    for summary in summaries:
        values = np.array(
            [
                float(row[summary["metric"]])
                for row in trials
                if row["condition_id"] == summary["condition_id"]
                and row["rate_hz"] == summary["rate_hz"]
            ]
        )
        mean = np.mean(values)
        margin = t.ppf(0.975, len(values) - 1) * sem(values, ddof=1)
        actual = np.array(
            [mean, np.std(values, ddof=1), mean - margin, mean + margin]
        )
        saved = np.array(
            [
                float(summary[key])
                for key in ["mean", "sample_sd", "ci95_low", "ci95_high"]
            ]
        )
        max_summary_difference = max(
            max_summary_difference, float(np.max(np.abs(actual - saved)))
        )
    max_example_difference = 0.0
    max_recovery_difference = 0.0
    for cid in [5, 7]:
        for rate, reference in [(16000, x), (48000, original)]:
            path = results / f"example_c{cid}_r0_{rate}hz.csv"
            example = np.loadtxt(path, delimiter=",", skiprows=1)
            if not np.array_equal(example[:, 2], reference):
                raise AssertionError("An example reference has changed.")
            before = np.mean((example[:, 3] - reference) ** 2)
            after = np.mean((example[:, 4] - reference) ** 2)
            recorded = next(
                row
                for row in trials
                if int(row["condition_id"]) == cid
                and int(row["repeat_id"]) == 0
                and int(row["rate_hz"]) == rate
            )
            max_example_difference = max(
                max_example_difference,
                abs(before - float(recorded["no_filter_mse"])),
                abs(after - float(recorded["filter_mse"])),
            )
            sample_example = np.loadtxt(
                results / f"example_c{cid}_r0_16000hz.csv",
                delimiter=",",
                skiprows=1,
            )
            # Independent direct convolution and SciPy Fourier resampling.
            direct = np.convolve(sample_example[:, 3], h, mode="same")
            for values, column in [
                (sample_example[:, 3], 3),
                (direct, 4),
            ]:
                recovered = (
                    values if rate == 16000 else resample(values, 96000)
                )
                max_recovery_difference = max(
                    max_recovery_difference,
                    float(np.max(np.abs(recovered - example[:, column]))),
                )
    checked_sources = {}
    for relative, expected in manifest["input_and_code_sha256"].items():
        actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        checked_sources[relative] = actual == expected
    for name, expected in manifest["saved_example_sha256"].items():
        checked_sources["stage2/r23/results/" + name] = (
            hashlib.sha256((results / name).read_bytes()).hexdigest()
            == expected
        )
    checks = {
        "all_400_noise_and_contaminated_arrays_replay": replayed == 400,
        "all_summary_statistics_and_intervals": max_summary_difference < 1e-12,
        "all_full_edge_interior_partitions": max_partition_difference < 1e-12,
        "all_paired_gain_definitions": max_gain_difference < 1e-12,
        "four_full_example_csv_metrics": max_example_difference < 1e-12,
        "direct_convolution_scipy_recovery_examples": max_recovery_difference
        < 1e-11,
        "all_recorded_source_and_example_hashes": all(
            checked_sources.values()
        ),
    }
    if not all(checks.values()):
        raise AssertionError(checks)
    audit = {
        "status": "Independent R2.3 recomputation passed",
        "checks": {key: bool(value) for key, value in checks.items()},
        "max_summary_difference": max_summary_difference,
        "max_partition_difference": max_partition_difference,
        "max_gain_difference": max_gain_difference,
        "max_example_mse_difference": max_example_difference,
        "max_independent_recovery_difference": max_recovery_difference,
        "source_hash_checks": checked_sources,
    }
    (ROOT / "metadata/r23_independent_check.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    print("R2.3: seven independent recomputation/replay checks passed.")


if __name__ == "__main__":
    main()
