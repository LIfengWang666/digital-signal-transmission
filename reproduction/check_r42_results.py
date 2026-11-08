"""Replay R4.2 using independent convolution, quantising and decoding."""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import statistics
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.signal import resample
from scipy.stats import t

from reproduction.check_r32_results import independent_tables, read_rows
from stage4.r42.pipeline import receive_stream

ROOT = Path(__file__).resolve().parents[1]


def direct_filter(values: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Use direct finite convolution and the frozen centred output crop.

    Args:
        values: Complete real 16 kHz record, zero outside its interval.
        kernel: Frozen symmetric 385-tap FIR.

    Returns:
        All 32000 aligned samples from the full linear convolution.
    """
    return np.convolve(values, kernel, mode="full")[192:32192]


def binary_hash(values: np.ndarray) -> str:
    """Hash independently assembled channel bits and true error masks.

    Args:
        values: Flat binary or boolean values.

    Returns:
        SHA-256 of uint8 sample bytes.
    """
    return hashlib.sha256(
        np.asarray(values, dtype=np.uint8).tobytes()
    ).hexdigest()


def check_waveform(
    row: dict[str, str],
    reference: np.ndarray,
    high: np.ndarray,
    before: np.ndarray,
    errors: dict[str, np.ndarray],
    component: dict[str, str],
) -> float:
    """Check actual full-record quality and every component/cross term.

    Args:
        row: Saved pilot path metrics.
        reference: Original NPY.
        high: Independently recovered waveform.
        before: Independently reconstructed prechannel waveform.
        errors: Five independent-grid component arrays.
        component: Saved five-power/ten-cross-term row.

    Returns:
        Absolute MSE discrepancy between independent and saved results.
    """
    error = high - reference
    mse = float(np.mean(error**2))
    power = float(np.mean(reference**2))
    expected = {
        "mse": mse,
        "nmse": mse / power,
        "mae": float(np.mean(np.abs(error))),
        "rmse": math.sqrt(mse),
        "mean_error": float(np.mean(error)),
        "error_variance_ddof0": float(np.var(error)),
        "peak_absolute_error": float(np.max(np.abs(error))),
        "prechannel_nmse": float(np.mean((before - reference) ** 2) / power),
    }
    for name, value in expected.items():
        assert math.isclose(
            float(row[name]), value, rel_tol=3e-11, abs_tol=2e-12
        ), (name, row, value)
    for name, error in errors.items():
        assert math.isclose(
            float(component[f"power_{name}"]),
            float(np.mean(error**2)),
            rel_tol=3e-10,
            abs_tol=2e-11,
        )
    for a, b in combinations(errors, 2):
        value = float(2 * np.mean(errors[a] * errors[b]))
        assert math.isclose(
            float(component[f"cross_{a}_{b}"]),
            value,
            rel_tol=3e-10,
            abs_tol=2e-11,
        )
    assert float(component["mse_residual"]) < 1e-9
    assert float(component["waveform_identity_residual"]) < 1e-10
    assert float(component["nyquist_energy_residual"]) < 1e-9
    assert np.max(np.abs(high - reference - sum(errors.values()))) < 1e-10
    return abs(mse - float(row["mse"]))


def main() -> None:
    """Independently replay all 1760 paths and audit the planning summaries.

    Returns:
        None. Save an independent review without importing the pilot runner.
    """
    folder = ROOT / "stage4/r42/results"
    frozen = json.loads((folder / "pilot_freeze.json").read_text())
    design = frozen["choices"]
    manifest = json.loads((folder / "seed_manifest.json").read_text())
    for name, digest in frozen["source_sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
    assert (
        hashlib.sha256((folder / "pilot_freeze.json").read_bytes()).hexdigest()
        == manifest["pilot_freeze_sha256"]
    )
    reference = np.load(ROOT / "reference_signal.npy")
    x = reference[::3]
    h = np.load(ROOT / "stage2/r22/results/selected_coefficients.npy")
    clean = {0: x, 1: direct_filter(x, h)}
    clean_high = {f: resample(v, 96000) for f, v in clean.items()}
    levels = -18 + 36 / (2**14 - 1) * np.arange(2**14)
    midpoints = (levels[:-1] + levels[1:]) / 2
    definitions = json.loads(
        (ROOT / "stage3/r32/results/code_definitions.json").read_text()
    )
    specs = {
        s["code"]: s
        for s in definitions
        if s["code"] in {"uncoded", "hamming74"}
    }
    tables = {key: independent_tables(value) for key, value in specs.items()}
    trials = read_rows(folder / "pilot_trials.csv")
    cases = {
        (
            int(r["record_id"]),
            int(r["joint_condition_id"]),
            r["configuration"],
        ): r
        for r in trials
    }
    components = {
        (
            int(r["record_id"]),
            int(r["joint_condition_id"]),
            r["configuration"],
        ): r
        for r in read_rows(folder / "decomposition.csv")
    }
    channel = {
        (r["record_id"], r["joint_condition_id"], r["layout_id"]): r
        for r in manifest["bsc_mask_groups"]
    }
    quantisers = {
        (
            int(r["record_id"]),
            int(r["snr_condition_id"]),
            int(r["filter_enabled"]),
        ): r
        for r in read_rows(folder / "quantiser_banks.csv")
    }
    assert len(cases) == len(trials) == 1760
    assert (
        len(components) == 1760
        and len(channel) == 880
        and len(quantisers) == 320
    )
    paths = set()
    checked, masks, max_difference = 0, 0, 0.0
    for bank in manifest["gaussian_inputs"]:
        repeat, scid = bank["record_id"], bank["snr_condition_id"]
        path = (1, 4, 0, scid, repeat)
        assert list(path) == bank["spawn_key"] and path not in paths
        paths.add(path)
        rng = np.random.Generator(
            np.random.PCG64(
                np.random.SeedSequence(2470964, spawn_key=path, pool_size=4)
            )
        )
        assert rng.bit_generator.state == bank["initial_state"]
        variance = np.mean(x**2) / 10 ** (bank["snr_db"] / 10)
        noise = rng.normal(0, np.sqrt(variance), x.shape)
        assert (
            hashlib.sha256(noise.astype("<f8").tobytes()).hexdigest()
            == bank["noise_sha256"]
        )
        masks_by_case = {}
        for f in [0, 1]:
            z = direct_filter(x + noise, h) if f else x + noise
            labels = np.searchsorted(
                midpoints, np.clip(z, -18, 18) - 18e-9, side="left"
            )
            payload = (
                ((labels[:, None] >> np.arange(13, -1, -1)) & 1)
                .astype(np.uint8)
                .reshape(-1)
            )
            before = resample(levels[labels], 96000)
            qrow = quantisers[(repeat, scid, f)]
            assert int(qrow["overload_samples"]) == np.count_nonzero(
                np.abs(z) > 18
            )
            assert math.isclose(
                float(qrow["prechannel_nmse"]),
                float(
                    np.mean((before - reference) ** 2) / np.mean(reference**2)
                ),
                rel_tol=3e-11,
                abs_tol=2e-12,
            )
            upstream = {
                "sampling": clean_high[0] - reference,
                "filter": clean_high[f] - clean_high[0],
                "gaussian": resample(
                    direct_filter(noise, h) if f else noise, 96000
                ),
                "quantisation": before - resample(z, 96000),
            }
            for layout, code in [(0, "uncoded"), (1, "hamming74")]:
                n, k = specs[code]["n"], specs[code]["k"]
                parity, lookup = tables[code]
                messages = payload.reshape(-1, k)
                parity_bits = np.zeros((len(messages), n - k), dtype=np.uint8)
                for column in range(n - k):
                    parity_bits[:, column] = np.bitwise_xor.reduce(
                        messages[:, parity[:, column].astype(bool)], axis=1
                    )
                sent = np.concatenate((messages, parity_bits), axis=1).reshape(
                    -1
                )
                for condition in design["conditions"]:
                    if condition["snr_condition_id"] != scid:
                        continue
                    cid, p = (
                        condition["joint_condition_id"],
                        condition["probability"],
                    )
                    key = (cid, layout)
                    ch = channel[(repeat, cid, layout)]
                    if f == 0:
                        if p == 0:
                            assert (
                                ch["spawn_key"] is None
                                and ch["initial_state"] is None
                            )
                            mask = np.zeros(len(sent), dtype=bool)
                        else:
                            path = (2, 4, 0, cid, repeat, layout)
                            assert (
                                path not in paths
                                and list(path) == ch["spawn_key"]
                            )
                            paths.add(path)
                            rng = np.random.Generator(
                                np.random.PCG64(
                                    np.random.SeedSequence(
                                        2470964, spawn_key=path, pool_size=4
                                    )
                                )
                            )
                            assert (
                                rng.bit_generator.state == ch["initial_state"]
                            )
                            mask = rng.random(sent.shape) < p
                            masks += 1
                        masks_by_case[key] = mask
                    mask = masks_by_case[key]
                    assert binary_hash(mask) == ch["mask_sha256"]
                    received = sent ^ mask.astype(np.uint8)
                    weights = 2 ** np.arange(n - 1, -1, -1, dtype=np.uint32)
                    data = lookup[
                        received.reshape(-1, n).astype(np.uint32) @ weights
                    ].reshape(-1)
                    wrong = data != payload
                    row = cases[(repeat, cid, f"{f}{layout}")]
                    assert binary_hash(sent) == row["transmitted_sha256"]
                    assert binary_hash(received) == row["received_sha256"]
                    assert binary_hash(data) == row["decoded_payload_sha256"]
                    assert binary_hash(wrong) == row["wrong_pattern_sha256"]
                    expected = {
                        "channel_flips": np.count_nonzero(mask),
                        "decoded_wrong_bits": np.count_nonzero(wrong),
                        "block_failures": np.count_nonzero(
                            np.any(wrong.reshape(-1, k), axis=1)
                        ),
                        "wrong_msb_bits": np.count_nonzero(
                            wrong.reshape(-1, 14)[:, 0]
                        ),
                        "wrong_samples": np.count_nonzero(
                            np.any(wrong.reshape(-1, 14), axis=1)
                        ),
                        "overload_samples": np.count_nonzero(np.abs(z) > 18),
                    }
                    for name, value in expected.items():
                        assert int(row[name]) == value, name
                    indices = data.reshape(-1, 14).astype(np.uint32) @ (
                        2 ** np.arange(13, -1, -1, dtype=np.uint32)
                    )
                    high = resample(levels[indices], 96000)
                    errors = {**upstream, "channel": high - before}
                    difference = check_waveform(
                        row,
                        reference,
                        high,
                        before,
                        errors,
                        components[(repeat, cid, f"{f}{layout}")],
                    )
                    max_difference = max(max_difference, difference)
                    checked += 1
    assert checked == 1760 and masks == 840 and len(paths) == 1000
    pairs = read_rows(folder / "pair_checks.csv")
    assert len(pairs) == 880
    assert {
        (
            int(r["record_id"]),
            int(r["joint_condition_id"]),
            int(r["layout_id"]),
        )
        for r in pairs
    } == set(channel)
    assert all(
        r["same_mask"]
        == r["same_decoded_error_pattern"]
        == r["same_bit_and_sample_error_counts"]
        == "1"
        for r in pairs
    )
    for repeat in range(20):
        for condition in design["conditions"]:
            cid = condition["joint_condition_id"]
            for layout in [0, 1]:
                a, b = (
                    cases[(repeat, cid, f"0{layout}")],
                    cases[(repeat, cid, f"1{layout}")],
                )
                assert a["mask_sha256"] == b["mask_sha256"]
                assert a["wrong_pattern_sha256"] == b["wrong_pattern_sha256"]
            if condition["probability"] == 0:
                for f in [0, 1]:
                    a, b = (
                        cases[(repeat, cid, f"{f}0")],
                        cases[(repeat, cid, f"{f}1")],
                    )
                    assert (
                        a["decoded_payload_sha256"]
                        == b["decoded_payload_sha256"]
                    )
                    assert a["mse"] == b["mse"]
    effects = read_rows(folder / "paired_effects.csv")
    assert len(effects) == 440
    for row in effects:
        repeat, cid = int(row["record_id"]), int(row["joint_condition_id"])
        a, b, c, d = [
            float(cases[(repeat, cid, config)]["nmse"])
            for config in ["00", "10", "01", "11"]
        ]
        expected = {
            "filter_without_code": a - b,
            "filter_with_code": c - d,
            "code_without_filter": a - c,
            "code_with_filter": b - d,
            "interaction": a - b - c + d,
        }
        for key, value in expected.items():
            assert math.isclose(
                float(row[key]), value, rel_tol=1e-12, abs_tol=1e-13
            )
    controls = read_rows(folder / "clean_controls.csv")
    assert len(controls) == 4
    for row in controls:
        z = clean[int(row["configuration"][0])]
        labels = np.searchsorted(
            midpoints, np.clip(z, -18, 18) - 18e-9, side="left"
        )
        error = resample(levels[labels], 96000) - reference
        assert math.isclose(
            float(row["mse"]),
            float(np.mean(error**2)),
            rel_tol=3e-11,
            abs_tol=2e-14,
        )
        assert int(row["channel_flips"]) == int(row["decoded_wrong_bits"]) == 0
    summary = read_rows(folder / "pilot_summary.csv")
    for row in summary:
        values = [
            float(r["nmse"])
            for r in trials
            if r["joint_condition_id"] == row["joint_condition_id"]
            and r["configuration"] == row["configuration"]
        ]
        assert math.isclose(
            float(row["mean_nmse"]), statistics.fmean(values), rel_tol=1e-13
        )
        assert math.isclose(
            float(row["sd_nmse"]), statistics.stdev(values), rel_tol=1e-12
        )
    bundles = read_rows(folder / "pilot_bundle_scores.csv")
    main = {
        c["joint_condition_id"] for c in design["conditions"] if c["primary"]
    }
    assert len(main) == 12 and len(bundles) == 80
    for row in bundles:
        values = [
            float(
                cases[(int(row["record_id"]), cid, row["configuration"])][
                    "nmse"
                ]
            )
            for cid in main
        ]
        assert math.isclose(
            float(row["objective"]), statistics.fmean(values), rel_tol=1e-13
        )
    planning = read_rows(folder / "precision_planning.csv")
    records = int(planning[0]["candidate_records"])
    assert len(planning) == 92 and records >= 200 and records % 50 == 0
    previous_fails = False
    for row in planning:
        if row["statistic"].startswith("condition"):
            number, config = row["statistic"].split(":")
            cid = int(number.removeprefix("condition"))
            values = [
                float(cases[(r, cid, config)]["nmse"]) for r in range(20)
            ]
        else:
            config = row["statistic"].split(":")[1]
            values = [
                float(r["objective"])
                for r in bundles
                if r["configuration"] == config
            ]
        mean, sd = statistics.fmean(values), statistics.stdev(values)
        assert math.isclose(mean, float(row["mean"]), rel_tol=1e-13)
        assert math.isclose(sd, float(row["sd"]), rel_tol=1e-12)
        half = (
            t.ppf(1 - 0.05 / (2 * 92), records - 1) * sd / math.sqrt(records)
        )
        assert math.isclose(
            half, float(row["predicted_half_width"]), rel_tol=1e-12
        )
        assert half <= 0.05 * mean
        if records > 200:
            previous = (
                t.ppf(1 - 0.05 / (2 * 92), records - 51)
                * sd
                / math.sqrt(records - 50)
            )
            previous_fails |= previous > 0.05 * mean
    assert records == 200 or previous_fails
    assert list(inspect.signature(receive_stream).parameters) == [
        "received",
        "code",
        "payload_bits",
        "bits",
        "half_range",
        "output_samples",
    ]
    result = {
        "status": "passed",
        "independent_path_records": checked,
        "independent_gaussian_inputs": 160,
        "independent_unique_BSC_masks": masks,
        "unique_paths": len(paths),
        "paired_mask_groups_verified": 880,
        "full_decomposition_rows_verified": 1760,
        "summary_means_verified": 88,
        "deterministic_controls_verified": 4,
        "unique_quantiser_banks_verified": 320,
        "paired_effect_rows_verified": 440,
        "bundle_objective_rows_verified": 80,
        "planning_statistics_verified": 92,
        "candidate_validation_records_verified": records,
        "maximum_independent_MSE_difference": max_difference,
        "methods": [
            "direct full convolution",
            "midpoint search quantisation",
            "XOR parity encoding",
            "correction-sphere lookup decoding",
            "SciPy Fourier resampling",
            "independent bundle/SD calculations",
        ],
        "receiver_has_no_truth_or_mask_parameter": True,
        "phase1_used": False,
    }
    (folder / "independent_checks.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print("R4.2: all1760 paths independently replayed; planned R=", records)


if __name__ == "__main__":
    main()
