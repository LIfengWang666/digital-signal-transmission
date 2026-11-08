"""Replay R3.2 with independent correction-sphere lookup decoders."""

from __future__ import annotations

import csv
import hashlib
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.signal import resample
from scipy.stats import t

from stage3.r32.codes import decode

ROOT = Path(__file__).resolve().parents[1]


def read_rows(path: Path) -> list[dict[str, str]]:
    """Read measurements for independent checking.

    Args:
        path: Saved CSV file.

    Returns:
        Header-keyed string rows.
    """
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def independent_tables(spec: dict) -> tuple[np.ndarray, np.ndarray]:
    """Build a decoder from nonoverlapping correction spheres, without H.

    Args:
        spec: Recorded code definition, checked against fixed parity rows.

    Returns:
        Parity rows and a table of decoded message bits for every n-bit word.
    """
    expected = {
        "hamming74": ["110", "101", "011", "111"],
        "hamming1511": [
            "1100",
            "1010",
            "1001",
            "0110",
            "0101",
            "0011",
            "1110",
            "1101",
            "1011",
            "0111",
            "1111",
        ],
        "repetition51": ["1111"],
        "uncoded": [""],
    }
    name, n, k, radius = spec["code"], spec["n"], spec["k"], spec["t"]
    rows = ["".join(map(str, row)) for row in spec["P"]]
    assert rows == expected[name]
    parity = np.array(spec["P"], dtype=np.uint8).reshape(k, n - k)
    message_integers = np.arange(2**k, dtype=np.uint32)
    shifts = np.arange(k - 1, -1, -1, dtype=np.uint32)
    messages = ((message_integers[:, None] >> shifts) & 1).astype(np.uint8)
    parity_bits = np.zeros((len(messages), n - k), dtype=np.uint8)
    for column in range(n - k):
        selected = parity[:, column].astype(bool)
        parity_bits[:, column] = np.bitwise_xor.reduce(
            messages[:, selected], axis=1
        )
    codewords = np.concatenate((messages, parity_bits), axis=1)
    assert int(np.min(np.sum(codewords[1:], axis=1))) == spec["d_min"]
    table = np.zeros((2**n, k), dtype=np.uint8)
    coverage = np.zeros(2**n, dtype=np.uint8)
    weights = 2 ** np.arange(n - 1, -1, -1, dtype=np.uint32)
    for count in range(radius + 1):
        for positions in combinations(range(n), count):
            received = codewords.copy()
            for position in positions:
                received[:, position] ^= 1
            keys = received.astype(np.uint32) @ weights
            coverage[keys] += 1
            table[keys] = messages
    assert np.all(coverage == 1)
    raw = np.arange(2**n, dtype=np.uint32)
    raw_bits = (
        (raw[:, None] >> np.arange(n - 1, -1, -1, dtype=np.uint32)) & 1
    ).astype(np.uint8)
    under_test, _ = decode(raw_bits.reshape(-1), name)
    assert np.array_equal(under_test.reshape(-1, k), table)
    return parity, table


def main() -> None:
    """Replay every actual mask and independently recompute the decision.

    Returns:
        None. Save the review or raise on any inconsistent bit or metric.
    """
    stage = ROOT / "stage3/r32"
    results = stage / "results"
    protocol = json.loads((stage / "design.json").read_text())
    frozen = json.loads((results / "code_freeze.json").read_text())
    manifest = json.loads((results / "seed_manifest.json").read_text())
    definitions = json.loads((results / "code_definitions.json").read_text())
    quantiser = json.loads((ROOT / protocol["quantiser_freeze"]).read_text())
    for name, digest in frozen["source_sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
    tables = {spec["code"]: independent_tables(spec) for spec in definitions}
    trial_rows = read_rows(results / "trials.csv")
    by_case = {
        (
            r["code"],
            int(r["condition_id"]),
            int(r["gaussian_repeat_id"]),
            int(r["bit_repeat_id"]),
        ): r
        for r in trial_rows
    }
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
    original = np.load(ROOT / "reference_signal.npy")
    x = original[::3]
    h = np.load(ROOT / "stage2/r22/results/selected_coefficients.npy")
    delay = len(h) // 2
    end = delay + len(x)
    a, bits = quantiser["half_range"], quantiser["bits"]
    epsilon = quantiser["normalised_epsilon"]
    codebook = -a + 2 * a / (2**bits - 1) * np.arange(2**bits)
    thresholds = (codebook[:-1] + codebook[1:]) / 2
    snr = json.loads((ROOT / "metadata/randomness_design.json").read_text())[
        "snr_baseline_db"
    ]
    checked = []
    max_mse_difference = 0.0
    masks = 0
    for bank in manifest["input_banks"]:
        group = bank["gaussian_repeat_id"]
        rng = np.random.Generator(
            np.random.PCG64(
                np.random.SeedSequence(
                    manifest["root_seed"],
                    spawn_key=tuple(bank["gaussian_path"]),
                    pool_size=manifest["pool_size"],
                )
            )
        )
        variance = np.mean(x**2) / 10 ** (snr / 10)
        noise = rng.normal(0, np.sqrt(variance), x.shape)
        z = np.convolve(x + noise, h, mode="full")[delay:end]
        clipped = np.minimum(np.maximum(z, -a), a)
        labels = np.searchsorted(
            thresholds, clipped - a * epsilon, side="left"
        )
        payload = (
            ((labels[:, None] >> np.arange(bits - 1, -1, -1)) & 1)
            .astype(np.uint8)
            .reshape(-1)
        )
        assert (
            hashlib.sha256(payload.tobytes()).hexdigest()
            == bank["payload_sha256"]
        )
        q = codebook[labels]
        pre_high = resample(q, len(original))
        pre_error = pre_high - original
        for spec in definitions:
            name, n, k = spec["code"], spec["n"], spec["k"]
            parity, table = tables[name]
            padding = (-len(payload)) % k
            message = np.pad(payload, (0, padding)).reshape(-1, k)
            check_bits = np.zeros((len(message), n - k), dtype=np.uint8)
            for column in range(n - k):
                check_bits[:, column] = np.bitwise_xor.reduce(
                    message[:, parity[:, column].astype(bool)], axis=1
                )
            sent = np.concatenate((message, check_bits), axis=1).reshape(-1)
            weights = 2 ** np.arange(n - 1, -1, -1, dtype=np.uint32)
            cases = [
                c
                for c in manifest["channel_trials"]
                if c["code"] == name and c["gaussian_repeat_id"] == group
            ]
            for case in cases:
                cid, repeat = case["condition_id"], case["bit_repeat_id"]
                saved = by_case[(name, cid, group, repeat)]
                if case["identity_no_rng"]:
                    mask = np.zeros(len(sent), dtype=bool)
                else:
                    seed = np.random.SeedSequence(
                        manifest["root_seed"],
                        spawn_key=tuple(case["spawn_key"]),
                        pool_size=manifest["pool_size"],
                    )
                    rng = np.random.Generator(np.random.PCG64(seed))
                    assert rng.bit_generator.state == case["initial_state"]
                    mask = rng.random(sent.shape) < case["probability"]
                    masks += 1
                received = sent ^ mask.astype(np.uint8)
                assert (
                    hashlib.sha256(received.tobytes()).hexdigest()
                    == case["received_sha256"]
                )
                keys = received.reshape(-1, n).astype(np.uint32) @ weights
                decoded_full = table[keys].reshape(-1)
                data = decoded_full[: len(payload)]
                assert (
                    hashlib.sha256(data.tobytes()).hexdigest()
                    == case["decoded_real_payload_sha256"]
                )
                wrong = data != payload
                counts = {
                    "channel_flips": int(np.count_nonzero(mask)),
                    "decoded_wrong_bits": int(np.count_nonzero(wrong)),
                    "full_message_block_failures": int(
                        np.count_nonzero(
                            np.any(
                                decoded_full.reshape(-1, k) != message, axis=1
                            )
                        )
                    ),
                    "transmitted_bits": len(sent),
                    "padding_bits": padding,
                }
                for key, value in counts.items():
                    assert value == int(saved[key]), (name, cid, group, key)
                observed_weights = np.sum(mask.reshape(-1, n), axis=1)
                histogram = np.bincount(
                    observed_weights.astype(int), minlength=n + 1
                )
                for weight, count in enumerate(histogram):
                    assert (
                        int(count)
                        == by_weight[(name, cid, group, repeat, weight)]
                    )
                indices = data.reshape(-1, bits).astype(np.uint32) @ (
                    2 ** np.arange(bits - 1, -1, -1, dtype=np.uint32)
                )
                high = resample(codebook[indices], len(original))
                actual = float(np.mean((high - original) ** 2))
                error = abs(actual - float(saved["mse_48000"]))
                max_mse_difference = max(max_mse_difference, error)
                assert np.isclose(
                    actual, float(saved["mse_48000"]), rtol=1e-10, atol=1e-12
                )
                added = high - pre_high
                assert np.isclose(
                    np.mean(added**2),
                    float(saved["channel_component_mse_48000"]),
                    rtol=1e-10,
                    atol=1e-12,
                )
                assert np.isclose(
                    2 * np.mean(pre_error * added),
                    float(saved["upstream_channel_cross"]),
                    rtol=1e-9,
                    atol=1e-12,
                )
                checked.append(
                    {
                        "code": name,
                        "probability": case["probability"],
                        "group": group,
                        "mse": actual,
                        "wrong": counts["decoded_wrong_bits"],
                        "flips": counts["channel_flips"],
                        "data_bits": len(payload),
                        "tx_bits": len(sent),
                    }
                )
    assert len(checked) == len(trial_rows) == 1140
    score_values = {}
    for spec in definitions:
        name = spec["code"]
        values = []
        for group in protocol["gaussian_repeat_ids"]:
            ratios = []
            for probability in protocol["selection_probabilities"]:
                coded = [
                    r["mse"]
                    for r in checked
                    if r["code"] == name
                    and r["group"] == group
                    and r["probability"] == probability
                ]
                base = [
                    r["mse"]
                    for r in checked
                    if r["code"] == "uncoded"
                    and r["group"] == group
                    and r["probability"] == probability
                ]
                ratios.append(np.mean(coded) / np.mean(base))
            values.append(float(np.mean(ratios)))
        score_values[name] = np.array(values)
    selection = read_rows(results / "code_selection.csv")
    for row in selection:
        values = score_values[row["code"]]
        mean = float(np.mean(values))
        half = t.ppf(0.975, 4) * np.std(values, ddof=1) / np.sqrt(5)
        assert np.isclose(mean, float(row["score"]), rtol=1e-11, atol=1e-13)
        assert np.isclose(mean - half, float(row["score_ci95_low"]))
        assert np.isclose(mean + half, float(row["score_ci95_high"]))
    eligible = [
        row
        for row in selection
        if row["code"] != "uncoded"
        and float(row["payload_expansion"])
        <= protocol["max_payload_expansion"]
        and float(row["score"]) < 1
    ]
    chosen = min(eligible, key=lambda r: np.mean(score_values[r["code"]]))[
        "code"
    ]
    assert chosen == frozen["selected_code"]
    summaries = read_rows(results / "summary.csv")
    for row in summaries:
        data = [
            r
            for r in checked
            if r["code"] == row["code"]
            and r["probability"] == float(row["probability"])
        ]
        assert sum(r["wrong"] for r in data) == int(row["wrong_bits"])
        assert np.isclose(
            np.mean([r["mse"] for r in data]),
            float(row["mean_mse_48000"]),
            rtol=1e-10,
            atol=1e-12,
        )
    evidence = {
        "date": "2026-10-21",
        "status": "passed",
        "all_decoder_received_words_checked": sum(
            2 ** spec["n"] for spec in definitions
        ),
        "correction_spheres_cover_once": True,
        "pilot_rows_replayed": len(checked),
        "bsc_masks_replayed": masks,
        "gaussian_input_banks_replayed": 5,
        "independent_decoder": "Nearest correction-sphere table, no syndrome",
        "independent_encoder": "Per-parity XOR, no G matrix multiplication",
        "independent_quantiser": "Threshold search, no scaled round",
        "independent_filter_and_reconstruction": (
            "np.convolve and scipy.signal.resample"
        ),
        "all_decoded_payload_hashes_match": True,
        "maximum_full_grid_mse_difference": max_mse_difference,
        "decision_independently_confirmed": chosen,
        "phase_1_used": False,
    }
    (ROOT / "metadata/r32_independent_review.json").write_text(
        json.dumps(evidence, indent=2) + "\n"
    )
    print("R3.2: all 32930 decoder inputs and 1140 pilot replays verified.")


if __name__ == "__main__":
    main()
