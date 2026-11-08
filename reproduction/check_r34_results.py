"""Replay R3.4 with independent encoding, decoding and quality statistics."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path

import numpy as np
from scipy.signal import resample
from scipy.stats import binomtest, t

from reproduction.check_r32_results import independent_tables, read_rows

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Check every validation case without the production recovery chain.

    Returns:
        None. Write independent review evidence or raise on a discrepancy.
    """
    stage = ROOT / "stage3/r34"
    folder = stage / "results"
    design = json.loads((folder / "validation_freeze.json").read_text())
    manifest = json.loads((folder / "seed_manifest.json").read_text())
    for name, digest in design["source_sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
    assert (
        hashlib.sha256(
            (folder / "validation_freeze.json").read_bytes()
        ).hexdigest()
        == manifest["prevalidation_freeze_sha256"]
    )
    assert design["records_per_condition"] == 300
    assert design["confidence_family_size"] == 30
    original = np.load(ROOT / "reference_signal.npy")
    x = original[::3]
    power = float(np.mean(original**2))
    assert power == design["reference_power"]
    old = read_rows(ROOT / "stage3/r33/results/trials.csv")
    baseline = statistics.fmean(
        [
            float(r["nmse_48000"])
            for r in old
            if r["code"] == "hamming74" and float(r["probability"]) == 0
        ]
    )
    assert abs(design["primary_nmse_threshold"] - 2 * baseline) < 1e-15
    definitions = json.loads(
        (ROOT / "stage3/r32/results/code_definitions.json").read_text()
    )
    specs = [s for s in definitions if s["code"] in ["uncoded", "hamming74"]]
    tables = {s["code"]: independent_tables(s) for s in specs}
    trials = read_rows(folder / "trials.csv")
    cases = {
        (r["code"], int(r["condition_id"]), int(r["record_id"])): r
        for r in trials
    }
    assert len(cases) == len(trials) == 9000
    weights = {
        (
            r["code"],
            int(r["condition_id"]),
            int(r["record_id"]),
            int(r["weight"]),
        ): int(r["blocks"])
        for r in read_rows(folder / "block_weights.csv")
    }
    h = np.load(ROOT / "stage2/r22/results/selected_coefficients.npy")
    delay = len(h) // 2
    stop = delay + len(x)
    qspec = json.loads(
        (ROOT / "stage3/r31/results/quantiser_freeze.json").read_text()
    )
    bits, a = qspec["bits"], qspec["half_range"]
    levels = -a + 2 * a / (2**bits - 1) * np.arange(2**bits)
    midpoints = (levels[:-1] + levels[1:]) / 2
    max_mse_difference, masks_checked, rows_checked = 0.0, 0, 0
    bitplanes = {}
    paths = set()
    for bank in manifest["input_banks"]:
        record = bank["record_id"]
        path = tuple(bank["spawn_key"])
        assert path == (1, 3, 1, 5, record) and path not in paths
        paths.add(path)
        rng = np.random.Generator(
            np.random.PCG64(
                np.random.SeedSequence(2470964, spawn_key=path, pool_size=4)
            )
        )
        assert rng.bit_generator.state == bank["initial_state"]
        variance = np.mean(x**2) / 10 ** (design["gaussian_snr_db"] / 10)
        noise = rng.normal(0, np.sqrt(variance), x.shape)
        assert (
            hashlib.sha256(
                np.asarray(noise, dtype=np.float64).tobytes()
            ).hexdigest()
            == bank["noise_sha256"]
        )
        z = np.convolve(x + noise, h, mode="full")[delay:stop]
        labels = np.searchsorted(
            midpoints,
            np.clip(z, -a, a) - a * qspec["normalised_epsilon"],
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
        assert int(np.count_nonzero(np.abs(z) > a)) == bank["overload_samples"]
        for spec in specs:
            code, n, k = spec["code"], spec["n"], spec["k"]
            parity, lookup = tables[code]
            message = payload.reshape(-1, k)
            check = np.zeros((len(message), n - k), dtype=np.uint8)
            for column in range(n - k):
                check[:, column] = np.bitwise_xor.reduce(
                    message[:, parity[:, column].astype(bool)], axis=1
                )
            sent = np.concatenate((message, check), axis=1).reshape(-1)
            sent_hash = hashlib.sha256(sent.tobytes()).hexdigest()
            keys = 2 ** np.arange(n - 1, -1, -1, dtype=np.uint32)
            channels = [
                c
                for c in manifest["channel_trials"]
                if c["code"] == code and c["record_id"] == record
            ]
            assert len(channels) == 15
            for channel in channels:
                cid, p = channel["condition_id"], channel["probability"]
                saved = cases[(code, cid, record)]
                assert sent_hash == channel["transmitted_sha256"]
                if channel["identity_no_rng"]:
                    assert p == 0 and channel["spawn_key"] is None
                    mask = np.zeros(len(sent), dtype=bool)
                else:
                    path = tuple(channel["spawn_key"])
                    assert path == (
                        2,
                        3,
                        1,
                        cid,
                        record,
                        0 if code == "uncoded" else 1,
                    )
                    assert path not in paths
                    paths.add(path)
                    rng = np.random.Generator(
                        np.random.PCG64(
                            np.random.SeedSequence(
                                2470964, spawn_key=path, pool_size=4
                            )
                        )
                    )
                    assert rng.bit_generator.state == channel["initial_state"]
                    mask = rng.random(sent.shape) < p
                    masks_checked += 1
                received = sent ^ mask.astype(np.uint8)
                assert hashlib.sha256(received.tobytes()).hexdigest() == (
                    channel["received_sha256"]
                )
                data = lookup[
                    received.reshape(-1, n).astype(np.uint32) @ keys
                ].reshape(-1)
                assert hashlib.sha256(data.tobytes()).hexdigest() == (
                    channel["decoded_real_payload_sha256"]
                )
                wrong = data != payload
                integer_counts = {
                    "channel_flips": int(np.count_nonzero(mask)),
                    "decoded_wrong_bits": int(np.count_nonzero(wrong)),
                    "block_failures": int(
                        np.count_nonzero(np.any(wrong.reshape(-1, k), axis=1))
                    ),
                    "wrong_msb_bits": int(
                        np.count_nonzero(wrong.reshape(-1, bits)[:, 0])
                    ),
                    "wrong_samples": int(
                        np.count_nonzero(
                            np.any(wrong.reshape(-1, bits), axis=1)
                        )
                    ),
                }
                for field, count in integer_counts.items():
                    assert count == int(saved[field])
                histogram = np.bincount(
                    np.sum(mask.reshape(-1, n), axis=1).astype(int),
                    minlength=n + 1,
                )
                for weight, count in enumerate(histogram):
                    assert count == weights[(code, cid, record, weight)]
                plane = (code, cid)
                if plane not in bitplanes:
                    bitplanes[plane] = np.zeros(bits, dtype=np.int64)
                bitplanes[plane] += np.sum(
                    wrong.reshape(-1, bits), axis=0
                ).astype(np.int64)
                recovered = data.reshape(-1, bits).astype(np.uint32) @ (
                    2 ** np.arange(bits - 1, -1, -1, dtype=np.uint32)
                )
                high = resample(levels[recovered], len(original))
                mse = float(np.mean((high - original) ** 2))
                difference = abs(mse - float(saved["mse_48000"]))
                max_mse_difference = max(max_mse_difference, difference)
                assert np.isclose(
                    mse, float(saved["mse_48000"]), rtol=1e-10, atol=1e-12
                )
                added = high - before
                assert np.isclose(
                    np.mean(added**2),
                    float(saved["channel_component_mse_48000"]),
                    rtol=1e-10,
                    atol=1e-12,
                )
                assert np.isclose(
                    2 * np.mean(upstream * added),
                    float(saved["upstream_channel_cross"]),
                    rtol=1e-9,
                    atol=1e-12,
                )
                rows_checked += 1
        if (record + 1) % 100 == 0:
            print(
                f"R3.4 independent replay: {record + 1}/300 inputs.",
                flush=True,
            )
    assert masks_checked == 8400 and rows_checked == 9000
    for row in read_rows(folder / "bitplane_counts.csv"):
        assert (
            int(row["wrong_bits"])
            == bitplanes[(row["code"], int(row["condition_id"]))][
                int(row["msb_first_position"])
            ]
        )
    summary = read_rows(folder / "summary.csv")
    critical = float(t.isf(0.05 / 60, 299))
    decisions = {}
    max_statistic_difference = 0.0
    for row in summary:
        cid, code = int(row["condition_id"]), row["code"]
        group = [
            r
            for r in trials
            if r["code"] == code and int(r["condition_id"]) == cid
        ]
        values = [float(r["nmse_48000"]) for r in group]
        mean, sd = statistics.fmean(values), statistics.stdev(values)
        low, high = mean - critical * sd / math.sqrt(300), (
            mean + critical * sd / math.sqrt(300)
        )
        for actual, field in [
            (mean, "mean_nmse"),
            (sd, "sd_nmse"),
            (low, "adjusted_ci_low"),
            (high, "adjusted_ci_high"),
        ]:
            error = abs(actual - float(row[field]))
            max_statistic_difference = max(max_statistic_difference, error)
            assert error < 1e-10
        status = (
            "pass"
            if high <= design["primary_nmse_threshold"]
            else (
                "fail"
                if low > design["primary_nmse_threshold"]
                else "inconclusive"
            )
        )
        assert status == row["status"]
        decisions[(code, cid)] = (low, high, status)
        for summary_field, trial_field in [
            ("wrong_data_bits", "decoded_wrong_bits"),
            ("channel_flips", "channel_flips"),
            ("failed_blocks", "block_failures"),
            ("transmitted_bits", "transmitted_bits"),
            ("payload_bits", "payload_bits"),
        ]:
            assert sum(int(r[trial_field]) for r in group) == int(
                row[summary_field]
            )
        perfect = sum(int(r["decoded_wrong_bits"]) == 0 for r in group)
        assert perfect == int(row["perfect_payload_records"])
        ci = binomtest(perfect, 300).proportion_ci(method="exact")
        assert abs(ci.low - float(row["perfect_payload_ci95_low"])) < 1e-10
        assert abs(ci.high - float(row["perfect_payload_ci95_high"])) < 1e-10
    gain_set = []
    comparisons = read_rows(folder / "comparisons.csv")
    for row in comparisons:
        cid = int(row["condition_id"])
        gain = decisions[("hamming74", cid)][2] == "pass" and (
            decisions[("uncoded", cid)][2] == "fail"
        )
        assert int(gain) == int(row["coding_gain_established"])
        if gain:
            gain_set.append(float(row["probability"]))
        un = [cases[("uncoded", cid, i)] for i in range(300)]
        co = [cases[("hamming74", cid, i)] for i in range(300)]
        paired = [
            float(a["nmse_48000"]) - float(b["nmse_48000"])
            for a, b in zip(un, co)
        ]
        actual = statistics.fmean(paired)
        assert abs(actual - float(row["mean_paired_nmse_improvement"])) < 1e-10
    for row in read_rows(folder / "threshold_sensitivity.csv"):
        threshold = float(row["budget_factor"]) * baseline
        cid = int(row["condition_id"])
        statuses = {}
        for code in ["uncoded", "hamming74"]:
            low, high, _ = decisions[(code, cid)]
            statuses[code] = (
                "pass"
                if high <= threshold
                else ("fail" if low > threshold else "inconclusive")
            )
        assert statuses["uncoded"] == row["uncoded_status"]
        assert statuses["hamming74"] == row["coded_status"]
    outcome = json.loads((folder / "gain_summary.json").read_text())
    assert gain_set == outcome["gain_at_tested_probabilities"]
    report = {
        "date": "2026-10-21",
        "status": "passed",
        "independent_gaussian_inputs_replayed": 300,
        "independent_bsc_masks_replayed": masks_checked,
        "all_validation_rows_replayed": rows_checked,
        "all_payload_hashes_and_integer_counts_match": True,
        "decoder": "Independent correction-sphere lookup, no syndrome",
        "encoder": "Independent parity XOR, no G multiplication",
        "upstream": "Direct convolution, threshold quantiser, scipy resample",
        "maximum_mse_difference": max_mse_difference,
        "maximum_summary_statistic_difference": max_statistic_difference,
        "primary_gain_set_confirmed": gain_set,
        "no_design_observations_in_validation": True,
        "r4_streams_used": False,
    }
    (ROOT / "metadata/r34_independent_review.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print("R3.4: all validation cases and frozen-rule decisions verified.")


if __name__ == "__main__":
    main()
