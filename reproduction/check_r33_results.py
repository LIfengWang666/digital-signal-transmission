"""Independently replay the chosen R3.3 range and its measured evidence."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import bisect
from scipy.signal import resample
from scipy.stats import binomtest

from reproduction.check_r32_results import independent_tables, read_rows

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Check parameter replay, every case, binomial bounds and theory.

    Returns:
        None. Save the independent review, or raise on a disagreement.
    """
    stage = ROOT / "stage3/r33"
    results = stage / "results"
    protocol = json.loads((stage / "design.json").read_text())
    frozen = json.loads((results / "range_freeze.json").read_text())
    manifest = json.loads((results / "seed_manifest.json").read_text())
    parameter = json.loads((results / "personal_parameter.json").read_text())
    registry = json.loads(
        (ROOT / "metadata/randomness_design.json").read_text()
    )
    for name, digest in frozen["source_sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
    rng = np.random.Generator(
        np.random.PCG64(
            np.random.SeedSequence(2470964, spawn_key=(0, 1), pool_size=4)
        )
    )
    assert rng.bit_generator.state == parameter["initial_state"]
    u = float(rng.random())
    personal = 10.0 ** (-4 + 3 * u)
    assert u == parameter["uniform_draw"]
    assert personal == parameter["personal_probability"]
    assert registry["bit_flip_baseline_probability"] == personal
    assert registry["bit_flip_probability_range"] == [0.0001, 0.1]
    assert protocol["primary_grid"] == [
        0.0001,
        0.0003,
        0.001,
        0.003,
        0.01,
        0.03,
        0.1,
    ]
    definitions = json.loads(
        (ROOT / "stage3/r32/results/code_definitions.json").read_text()
    )
    specs = [s for s in definitions if s["code"] in protocol["code_names"]]
    oracles = {s["code"]: independent_tables(s) for s in specs}
    trials = read_rows(results / "trials.csv")
    by_case = {
        (
            r["code"],
            int(r["condition_id"]),
            int(r["gaussian_repeat_id"]),
            int(r["bit_repeat_id"]),
        ): r
        for r in trials
    }
    old_rows = read_rows(ROOT / "stage3/r32/results/trials.csv")
    imported = 0
    for old in old_rows:
        if old["code"] not in protocol["code_names"]:
            continue
        key = (
            old["code"],
            int(old["condition_id"]),
            int(old["gaussian_repeat_id"]),
            int(old["bit_repeat_id"]),
        )
        saved = by_case[key]
        for field, value in old.items():
            if field == "code":
                assert saved[field] == value
            else:
                assert float(saved[field]) == float(value)
        assert saved["origin"] == "r32_existing"
        imported += 1
    assert imported == 570
    weight_rows = read_rows(results / "block_weights.csv")
    by_weight = {
        (
            r["code"],
            int(r["condition_id"]),
            int(r["gaussian_repeat_id"]),
            int(r["bit_repeat_id"]),
            int(r["weight"]),
        ): int(r["blocks"])
        for r in weight_rows
    }
    assert len(by_weight) == len(weight_rows)
    original = np.load(ROOT / "reference_signal.npy")
    x = original[::3]
    h = np.load(ROOT / "stage2/r22/results/selected_coefficients.npy")
    delay = len(h) // 2
    stop = delay + len(x)
    qspec = json.loads(
        (ROOT / "stage3/r31/results/quantiser_freeze.json").read_text()
    )
    a, bits = qspec["half_range"], qspec["bits"]
    levels = -a + 2 * a / (2**bits - 1) * np.arange(2**bits)
    thresholds = (levels[:-1] + levels[1:]) / 2
    checked = []
    max_mse_difference = 0.0
    new_masks, all_masks = 0, 0
    for bank in manifest["input_banks"]:
        group = bank["gaussian_repeat_id"]
        rng = np.random.Generator(
            np.random.PCG64(
                np.random.SeedSequence(
                    2470964,
                    spawn_key=tuple(bank["gaussian_path"]),
                    pool_size=4,
                )
            )
        )
        variance = np.mean(x**2) * 10 ** (-registry["snr_baseline_db"] / 10)
        noise = rng.normal(0, np.sqrt(variance), x.shape)
        assert (
            hashlib.sha256(
                np.asarray(noise, dtype=np.float64).tobytes()
            ).hexdigest()
            == bank["noise_sha256"]
        )
        z = np.convolve(x + noise, h, mode="full")[delay:stop]
        clipped = np.clip(z, -a, a)
        labels = np.searchsorted(
            thresholds,
            clipped - a * qspec["normalised_epsilon"],
            side="left",
        )
        payload = (
            ((labels[:, None] >> np.arange(bits - 1, -1, -1)) & 1)
            .astype(np.uint8)
            .reshape(-1)
        )
        assert hashlib.sha256(payload.tobytes()).hexdigest() == (
            bank["payload_sha256"]
        )
        before = resample(levels[labels], len(original))
        upstream = before - original
        for spec in specs:
            name, n, k = spec["code"], spec["n"], spec["k"]
            parity, lookup = oracles[name]
            message = payload.reshape(-1, k)
            check = np.zeros((len(message), n - k), dtype=np.uint8)
            for column in range(n - k):
                check[:, column] = np.bitwise_xor.reduce(
                    message[:, parity[:, column].astype(bool)], axis=1
                )
            sent = np.concatenate((message, check), axis=1).reshape(-1)
            integers = 2 ** np.arange(n - 1, -1, -1, dtype=np.uint32)
            cases = [
                c
                for c in manifest["channel_trials"]
                if c["code"] == name and c["gaussian_repeat_id"] == group
            ]
            for case in cases:
                cid, repeat = case["condition_id"], case["bit_repeat_id"]
                saved = by_case[(name, cid, group, repeat)]
                if case["identity_no_rng"]:
                    assert case["spawn_key"] is None
                    mask = np.zeros(len(sent), dtype=bool)
                else:
                    path = case["spawn_key"]
                    assert path[:3] == [2, 3, 0]
                    assert path[4] == 8 * group + repeat
                    seed = np.random.SeedSequence(
                        2470964, spawn_key=tuple(path), pool_size=4
                    )
                    rng = np.random.Generator(np.random.PCG64(seed))
                    assert rng.bit_generator.state == case["initial_state"]
                    mask = rng.random(sent.shape) < case["probability"]
                    all_masks += 1
                    new_masks += case["origin"] == "r33_new"
                received = sent ^ mask.astype(np.uint8)
                assert hashlib.sha256(received.tobytes()).hexdigest() == (
                    case["received_sha256"]
                )
                data = lookup[
                    received.reshape(-1, n).astype(np.uint32) @ integers
                ].reshape(-1)
                assert hashlib.sha256(data.tobytes()).hexdigest() == (
                    case["decoded_real_payload_sha256"]
                )
                wrong = data != payload
                counts = {
                    "channel_flips": int(np.count_nonzero(mask)),
                    "decoded_wrong_bits": int(np.count_nonzero(wrong)),
                    "full_message_block_failures": int(
                        np.count_nonzero(np.any(wrong.reshape(-1, k), axis=1))
                    ),
                }
                for field, value in counts.items():
                    assert value == int(saved[field])
                histogram = np.bincount(
                    np.sum(mask.reshape(-1, n), axis=1).astype(int),
                    minlength=n + 1,
                )
                for weight, count in enumerate(histogram):
                    assert (
                        count == by_weight[(name, cid, group, repeat, weight)]
                    )
                recovered_labels = data.reshape(-1, bits).astype(np.uint32) @ (
                    2 ** np.arange(bits - 1, -1, -1, dtype=np.uint32)
                )
                high = resample(levels[recovered_labels], len(original))
                actual = float(np.mean((high - original) ** 2))
                max_mse_difference = max(
                    max_mse_difference, abs(actual - float(saved["mse_48000"]))
                )
                assert np.isclose(
                    actual, float(saved["mse_48000"]), atol=1e-12, rtol=1e-10
                )
                added = high - before
                assert np.isclose(
                    np.mean(added**2),
                    float(saved["channel_component_mse_48000"]),
                    atol=1e-12,
                    rtol=1e-10,
                )
                assert np.isclose(
                    2 * np.mean(upstream * added),
                    float(saved["upstream_channel_cross"]),
                    atol=1e-12,
                    rtol=1e-9,
                )
                checked.append(saved)
    assert len(checked) == 890 and new_masks == 320 and all_masks == 880
    summaries = read_rows(results / "summary.csv")
    max_interval_difference = 0.0
    for row in summaries:
        code, cid, p = (
            row["code"],
            int(row["condition_id"]),
            float(row["probability"]),
        )
        group = [
            r
            for r in trials
            if r["code"] == code and int(r["condition_id"]) == cid
        ]
        assert len(group) == int(row["records"])
        for result_key, trial_key in [
            ("payload_bits", "payload_bits"),
            ("transmitted_bits", "transmitted_bits"),
            ("wrong_data_bits", "decoded_wrong_bits"),
            ("channel_flips", "channel_flips"),
            ("blocks", "block_count"),
            ("failed_blocks", "full_message_block_failures"),
        ]:
            assert sum(int(r[trial_key]) for r in group) == int(
                row[result_key]
            )
        for count_key, denominator, low_key, high_key in [
            (
                "channel_flips",
                "transmitted_bits",
                "channel_ci95_low",
                "channel_ci95_high",
            ),
            ("failed_blocks", "blocks", "block_ci95_low", "block_ci95_high"),
        ]:
            ci = binomtest(
                int(row[count_key]), int(row[denominator])
            ).proportion_ci(confidence_level=0.95, method="exact")
            error = max(
                abs(ci.low - float(row[low_key])),
                abs(ci.high - float(row[high_key])),
            )
            max_interval_difference = max(max_interval_difference, error)
            assert error < 1e-9
        spec = next(s for s in specs if s["code"] == code)
        n, k = spec["n"], spec["k"]
        words = (
            (np.arange(2**n)[:, None] >> np.arange(n - 1, -1, -1)) & 1
        ).astype(np.uint8)
        weights = np.sum(words, axis=1)
        decoded = oracles[code][1]
        probabilities = p**weights * (1 - p) ** (n - weights)
        theory_ber = float(np.sum(np.sum(decoded, axis=1) * probabilities) / k)
        theory_bler = float(
            np.sum(probabilities[np.any(decoded != 0, axis=1)])
        )
        assert np.isclose(
            theory_ber, float(row["theory_data_ber"]), atol=1e-15
        )
        assert np.isclose(
            theory_bler,
            float(row["theory_block_failure_probability"]),
            atol=1e-15,
        )
        assert np.isclose(
            np.mean([float(r["mse_48000"]) for r in group]),
            float(row["mean_mse_48000"]),
            atol=1e-12,
        )

    def capacity(p: float) -> float:
        """Compute the binary-channel capacity for an interior probability.

        Args:
            p: Probability strictly between zero and one.

        Returns:
            Capacity in bits per channel use.
        """
        return 1 + p * math.log2(p) + (1 - p) * math.log2(1 - p)

    root = bisect(lambda p: capacity(p) - 4 / 7, 1e-12, 0.499)
    assert abs(root - frozen["capacity_equals_code_rate_at"]) < 1e-10
    assert (
        abs(
            (3 - math.sqrt(3)) / 6 - frozen["exact_data_ber_equals_uncoded_at"]
        )
        < 1e-10
    )
    review = {
        "date": "2026-10-21",
        "status": "passed",
        "all_cases_replayed": 890,
        "imported_r32_cases_checked": imported,
        "new_masks_replayed": new_masks,
        "all_unique_masks_replayed": all_masks,
        "independent_decoder": "Correction-sphere lookup, no syndrome",
        "independent_encoder": "Parity-bit XOR, no G multiplication",
        "independent_upstream": (
            "Direct convolution, threshold quantiser, resample"
        ),
        "parameter_scalar_and_state_match": True,
        "personal_probability": personal,
        "maximum_mse_difference": max_mse_difference,
        "maximum_interval_difference_vs_binomtest": max_interval_difference,
        "exact_ber_bler_independently_enumerated": True,
        "no_phase1_or_study4_paths": True,
    }
    (ROOT / "metadata/r33_independent_review.json").write_text(
        json.dumps(review, indent=2) + "\n"
    )
    print("R3.3: parameter, 890 cases, exact theory and intervals verified.")


if __name__ == "__main__":
    main()
