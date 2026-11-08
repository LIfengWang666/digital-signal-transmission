"""Replay R3.1 using threshold search and independent reconstruction."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.signal import resample

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Check all design measurements without calling the quantiser.

    Returns:
        None. Save independent replay evidence or raise on a discrepancy.
    """
    stage = ROOT / "stage3/r31"
    result = stage / "results"
    protocol = json.loads((stage / "design.json").read_text())
    frozen = json.loads((result / "quantiser_freeze.json").read_text())
    manifest = json.loads((result / "seed_manifest.json").read_text())
    with (result / "trials.csv").open(newline="") as stream:
        trials = list(csv.DictReader(stream))
    reference = np.load(ROOT / "reference_signal.npy")
    x = reference[::3]
    h = np.load(ROOT / protocol["filter_file"])
    selected = np.load(ROOT / "stage1/r12/results/selected_samples.npy")
    assert np.array_equal(x, selected)
    for name, digest in frozen["source_sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
    a = frozen["half_range"]
    eps = frozen["normalised_epsilon"]
    n = len(x)
    delay = len(h) // 2
    max_numeric_difference = 0.0
    max_relative_mse_difference = 0.0
    max_label_difference = 0
    recomputed = []
    for record in manifest["trial_paths"]:
        path = record["spawn_key"]
        seed = np.random.SeedSequence(
            manifest["root_seed"],
            spawn_key=tuple(path),
            pool_size=manifest["pool_size"],
        )
        rng = np.random.Generator(np.random.PCG64(seed))
        assert rng.bit_generator.state == record["initial_state"]
        variance = np.mean(x**2) / 10 ** (record["target_snr_db"] / 10)
        noise = rng.normal(0.0, np.sqrt(variance), x.shape)
        digest = hashlib.sha256(
            np.asarray(noise, dtype="<f8").tobytes()
        ).hexdigest()
        assert digest == record["noise_array_sha256"]
        noisy = x + noise
        end = delay + n
        direct = np.convolve(noisy, h, mode="full")[delay:end]
        for branch, z in [("filtered", direct), ("unfiltered", noisy)]:
            baseline = resample(z, len(reference))
            base_error = baseline - reference
            base_mse = float(np.mean(base_error**2))
            rows = [
                row
                for row in trials
                if int(row["condition_id"]) == path[3]
                and int(row["repeat_id"]) == path[4]
                and row["branch"] == branch
            ]
            assert len(rows) == len(protocol["bits_candidates"])
            for row in rows:
                bits = int(row["bits"])
                levels = 2**bits
                step = 2 * a / (levels - 1)
                codebook = -a + step * np.arange(levels)
                thresholds = (codebook[:-1] + codebook[1:]) / 2
                clipped = np.minimum(np.maximum(z, -a), a)
                indices = np.searchsorted(
                    thresholds, clipped - a * eps, side="left"
                )
                output = codebook[indices]
                error = output - z
                recovered = resample(output, len(reference))
                added = recovered - baseline
                pz = float(np.mean(z**2))
                qp = float(np.mean(error**2))
                qh = float(np.mean(added**2))
                zero_indices = np.searchsorted(
                    thresholds, clipped, side="left"
                )
                qzero = float(np.mean((codebook[zero_indices] - z) ** 2))
                computed = {
                    "input_power": pz,
                    "quantisation_mse_16000": qp,
                    "sqnr_db": float(10 * np.log10(pz / qp)),
                    "model_sqnr_db": float(10 * np.log10(12 * pz / step**2)),
                    "upstream_mse_48000": base_mse,
                    "quantisation_mse_48000": qh,
                    "total_mse_48000": float(
                        np.mean((recovered - reference) ** 2)
                    ),
                    "upstream_quantisation_cross_48000": float(
                        2 * np.mean(base_error * added)
                    ),
                    "component_ratio": qh / base_mse,
                    "zero_epsilon_quantisation_mse": qzero,
                }
                for key, actual in computed.items():
                    saved = float(row[key])
                    difference = abs(actual - saved)
                    max_numeric_difference = max(
                        max_numeric_difference, difference
                    )
                    if "mse" in key and saved != 0:
                        max_relative_mse_difference = max(
                            max_relative_mse_difference,
                            difference / abs(saved),
                        )
                    assert np.isclose(actual, saved, rtol=2e-9, atol=2e-14), (
                        path,
                        branch,
                        bits,
                        key,
                        actual,
                        saved,
                    )
                assert np.count_nonzero(np.abs(z) > a) == int(
                    row["clip_count"]
                )
                recomputed.append(
                    {
                        "bits": bits,
                        "component_ratio": computed["component_ratio"],
                        "clip_count": int(row["clip_count"]),
                    }
                )
                if path[3:] == [5, 0] and branch == "filtered":
                    if bits == frozen["bits"]:
                        stored = np.load(result / "example_labels.npy")
                        max_label_difference = int(
                            np.max(np.abs(indices - stored.astype(int)))
                        )
                        assert max_label_difference == 0
    assert len(recomputed) == 1040
    passing = []
    for bits in protocol["bits_candidates"]:
        rows = [r for r in recomputed if r["bits"] == bits]
        if (
            max(r["component_ratio"] for r in rows)
            <= protocol["quantisation_component_budget"]
            and max(r["clip_count"] for r in rows) == 0
        ):
            passing.append(bits)
    assert min(passing) == frozen["bits"]
    packed = np.frombuffer(
        (result / "example_bits_msb.bin").read_bytes(), dtype=np.uint8
    )
    stream = np.unpackbits(packed, bitorder="big")
    width = frozen["bits"]
    assert len(stream) == n * width
    weights = 2 ** np.arange(width - 1, -1, -1)
    unpacked = stream.reshape(n, width) @ weights
    assert np.array_equal(unpacked, np.load(result / "example_labels.npy"))
    registry = json.loads(
        (ROOT / "metadata/randomness_design.json").read_text()
    )
    p_registry_is_null = all(
        registry[key] is None
        for key in [
            "bit_flip_probability_range",
            "bit_flip_mapping",
            "bit_flip_baseline_probability",
        ]
    )
    evidence = {
        "status": "passed",
        "noise_records_replayed": len(manifest["trial_paths"]),
        "measurement_rows_recomputed": len(recomputed),
        "methods": [
            "Direct time-domain np.convolve instead of FFT filtering",
            "Threshold search on a codebook instead of scaled np.round",
            "SciPy Fourier resample instead of project reconstruction",
            "Packed bytes decoded by np.unpackbits and binary weights",
        ],
        "maximum_absolute_numeric_difference": max_numeric_difference,
        "maximum_relative_mse_difference": max_relative_mse_difference,
        "example_label_difference": max_label_difference,
        "selected_bits_independently_confirmed": min(passing),
        "packed_payload_bytes": len(packed),
        "p_registry_is_null_at_audit": p_registry_is_null,
        "r31_draws_bsc_or_parameter_leaf": False,
        "inference_limit": (
            "Design evidence only; no R3 phase-1 coding validation"
        ),
    }
    (ROOT / "metadata/r31_independent_review.json").write_text(
        json.dumps(evidence, indent=2) + "\n"
    )
    print("R3.1: 40 independent replays and 1040 recomputations passed.")


if __name__ == "__main__":
    main()
