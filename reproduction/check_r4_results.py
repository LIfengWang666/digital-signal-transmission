"""Independently replay every held-out R4 path and verify its statistics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import math
import statistics
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.signal import resample
from scipy.stats import t

from reproduction.check_r32_results import independent_tables, read_rows
from reproduction.check_r42_results import (
    binary_hash,
    check_waveform,
    direct_filter,
)
from stage4.r42.pipeline import receive_stream

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "stage4/r43"
CONFIGS = ["00", "10", "01", "11"]


def digest(path: Path) -> str:
    """Hash a complete source or measurement file.

    Args:
        path: Existing file.

    Returns:
        SHA-256 of its bytes.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replay_chunk(start: int) -> dict:
    """Recompute all full-length paths in one20-bundle output chunk.

    Args:
        start: First complete-record ID in the chunk.

    Returns:
        Independent counts and maximum MSE discrepancy. This path uses
        direct convolution, midpoint quantisation, XOR parity, a correction
        sphere lookup decoder and SciPy resampling, not the production chain.
    """
    folder = STAGE / ".work" / f"records_{start:04d}"
    frozen = json.loads((STAGE / "results/validation_freeze.json").read_text())
    choices = frozen["choices"]
    cache = STAGE / ".work" / f"independent_{start:04d}.json"
    inputs = {p.name: digest(p) for p in folder.iterdir() if p.is_file()}
    source_hash = digest(Path(__file__))
    if cache.exists():
        prior = json.loads(cache.read_text())
        if (
            prior["input_sha256"] == inputs
            and prior["checker_sha256"] == source_hash
        ):
            return prior
    reference = np.load(ROOT / "reference_signal.npy")
    x = reference[::3]
    h = np.load(ROOT / "stage2/r22/results/selected_coefficients.npy")
    clean = {0: x, 1: direct_filter(x, h)}
    clean_high = {f: resample(v, 96000) for f, v in clean.items()}
    levels = -18 + 36 / (2**14 - 1) * np.arange(2**14)
    midpoints = (levels[:-1] + levels[1:]) / 2
    specs = {
        s["code"]: s
        for s in json.loads(
            (ROOT / "stage3/r32/results/code_definitions.json").read_text()
        )
        if s["code"] in {"uncoded", "hamming74"}
    }
    tables = {code: independent_tables(spec) for code, spec in specs.items()}
    trials = read_rows(folder / "trials.csv")
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
    quantisers = {
        (
            int(r["record_id"]),
            int(r["snr_condition_id"]),
            int(r["filter_enabled"]),
        ): r
        for r in read_rows(folder / "quantiser_banks.csv")
    }
    leaves = json.loads((folder / "leaves.json").read_text())
    channel = {
        (r["record_id"], r["joint_condition_id"], r["layout_id"]): r
        for r in leaves["bsc"]
    }
    paths, checked, masks, max_difference = set(), 0, 0, 0.0
    examples = {11: "personal", 5: "personal_snr_p003", 21: "snr40_personal_p"}
    example_checks = 0
    for gaussian in leaves["gaussian"]:
        repeat, scid = gaussian["record_id"], gaussian["snr_condition_id"]
        path = (1, 4, 1, scid, repeat)
        assert (
            start <= repeat < start + 20
            and list(path) == gaussian["spawn_key"]
        )
        assert path not in paths
        paths.add(path)
        rng = np.random.Generator(
            np.random.PCG64(
                np.random.SeedSequence(2470964, spawn_key=path, pool_size=4)
            )
        )
        assert rng.bit_generator.state == gaussian["initial_state"]
        variance = np.mean(x**2) / 10 ** (gaussian["snr_db"] / 10)
        noise = rng.normal(0, np.sqrt(variance), x.shape)
        assert (
            hashlib.sha256(noise.astype("<f8").tobytes()).hexdigest()
            == gaussian["noise_sha256"]
        )
        assert variance == gaussian["model_variance"]
        assert float(np.mean(noise**2)) == gaussian["realised_noise_power"]
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
                for condition in choices["conditions"]:
                    if condition["snr_condition_id"] != scid:
                        continue
                    cid, p = (
                        condition["joint_condition_id"],
                        condition["probability"],
                    )
                    ch = channel[(repeat, cid, layout)]
                    if f == 0:
                        if p == 0:
                            assert (
                                ch["spawn_key"] is None
                                and ch["initial_state"] is None
                            )
                            mask = np.zeros(len(sent), bool)
                        else:
                            path = (2, 4, 1, cid, repeat, layout)
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
                        masks_by_case[(cid, layout)] = mask
                    mask = masks_by_case[(cid, layout)]
                    assert binary_hash(mask) == ch["mask_sha256"]
                    assert np.count_nonzero(mask) == ch["channel_flips"]
                    received = sent ^ mask.astype(np.uint8)
                    data = lookup[
                        received.reshape(-1, n).astype(np.uint32)
                        @ (2 ** np.arange(n - 1, -1, -1, dtype=np.uint32))
                    ].reshape(-1)
                    wrong = data != payload
                    row = cases[(repeat, cid, f"{f}{layout}")]
                    assert row["phase_id"] == "1"
                    for key, values in [
                        ("transmitted_sha256", sent),
                        ("received_sha256", received),
                        ("decoded_payload_sha256", data),
                        ("wrong_pattern_sha256", wrong),
                        ("mask_sha256", mask),
                    ]:
                        assert binary_hash(values) == row[key], (
                            repeat,
                            cid,
                            f,
                            code,
                            key,
                        )
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
                        "payload_bits": 448000,
                        "transmitted_bits": len(sent),
                        "padding_bits": 0,
                        "samples": 96000,
                    }
                    for key, value in expected.items():
                        assert int(row[key]) == value, key
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
                    ratio = -10 * math.log10(
                        float(
                            np.mean((high - reference) ** 2)
                            / np.mean(reference**2)
                        )
                    )
                    assert math.isclose(
                        float(row["reconstruction_error_ratio_db"]),
                        ratio,
                        rel_tol=3e-11,
                        abs_tol=2e-10,
                    )
                    assert math.isclose(
                        float(row["correlation"]),
                        float(np.corrcoef(reference, high)[0, 1]),
                        rel_tol=3e-10,
                        abs_tol=2e-11,
                    )
                    if repeat == 0 and cid in examples:
                        example = read_rows(
                            STAGE / "results" / f"example_{examples[cid]}.csv"
                        )
                        values = np.array(
                            [
                                float(r[f"reconstruction_{f}{layout}"])
                                for r in example
                            ]
                        )
                        assert (
                            np.max(np.abs(values - high[9600:10560])) < 2e-11
                        )
                        assert np.array_equal(
                            np.array([float(r["reference"]) for r in example]),
                            reference[9600:10560],
                        )
                        example_checks += 1
                    checked += 1
    assert checked == 1760 and masks == 840 and len(paths) == 1000
    for repeat in range(start, start + 20):
        for condition in choices["conditions"]:
            cid = condition["joint_condition_id"]
            for layout in [0, 1]:
                a, b = [cases[(repeat, cid, f"{f}{layout}")] for f in [0, 1]]
                assert a["mask_sha256"] == b["mask_sha256"]
                assert a["wrong_pattern_sha256"] == b["wrong_pattern_sha256"]
            if condition["probability"] == 0:
                for f in [0, 1]:
                    a, b = [cases[(repeat, cid, f"{f}{c}")] for c in [0, 1]]
                    assert (
                        a["decoded_payload_sha256"]
                        == b["decoded_payload_sha256"]
                    )
                    assert a["mse"] == b["mse"]
    result = {
        "status": "passed",
        "start": start,
        "records": checked,
        "gaussian_inputs": 160,
        "unique_positive_BSC_masks": masks,
        "unique_paths": len(paths),
        "example_waveforms_checked": example_checks,
        "maximum_independent_MSE_difference": max_difference,
        "input_sha256": inputs,
        "checker_sha256": source_hash,
    }
    cache.write_text(json.dumps(result, indent=2) + "\n")
    return result


def verify_statistics(choices: dict) -> dict:
    """Recompute the means, SDs, objective and contrasts using scalar stats.

    Args:
        choices: The original pre-validation choice freeze.

    Returns:
        Counts of independently checked primary and supplementary statistics.
    """
    folder = STAGE / "results"
    rows = read_rows(folder / "trials.csv")
    cases = {
        (
            int(r["record_id"]),
            int(r["joint_condition_id"]),
            r["configuration"],
        ): r
        for r in rows
    }
    assert len(cases) == len(rows) == 123200
    values = {
        (cid, config): [
            float(cases[(r, cid, config)]["nmse"]) for r in range(1400)
        ]
        for cid in range(22)
        for config in CONFIGS
    }
    summaries = read_rows(folder / "summary.csv")
    bounds = {}
    critical = float(t.ppf(1 - 0.05 / (2 * 92), 1399))
    for row in summaries:
        key = int(row["joint_condition_id"]), row["configuration"]
        data = values[key]
        mean, sd = statistics.fmean(data), statistics.stdev(data)
        half = critical * sd / math.sqrt(1400)
        for field, value in [
            ("mean_nmse", mean),
            ("sd_nmse", sd),
            ("half_width_nmse", half),
            ("lower_nmse", mean - half),
            ("upper_nmse", mean + half),
        ]:
            assert math.isclose(
                float(row[field]), value, rel_tol=2e-12, abs_tol=2e-14
            ), field
        bounds[key] = (mean - half, mean + half)
        matched = [cases[(r, key[0], key[1])] for r in range(1400)]
        for field, source in [
            ("mean_mae", "mae"),
            ("mean_mse", "mse"),
            ("mean_decoded_wrong_bits", "decoded_wrong_bits"),
            ("mean_wrong_msb_bits", "wrong_msb_bits"),
            ("mean_wrong_samples", "wrong_samples"),
            ("mean_channel_flips", "channel_flips"),
            ("mean_prechannel_nmse", "prechannel_nmse"),
        ]:
            assert math.isclose(
                float(row[field]),
                statistics.fmean(float(r[source]) for r in matched),
                rel_tol=2e-12,
                abs_tol=2e-14,
            )
    main_ids = sorted(
        c["joint_condition_id"] for c in choices["conditions"] if c["primary"]
    )
    objectives = {
        config: [
            statistics.fmean(values[(cid, config)][r] for cid in main_ids)
            for r in range(1400)
        ]
        for config in CONFIGS
    }
    for row in read_rows(folder / "bundle_scores.csv"):
        assert math.isclose(
            float(row["objective"]),
            objectives[row["configuration"]][int(row["record_id"])],
            rel_tol=2e-12,
        )
    for row in read_rows(folder / "objective_summary.csv"):
        config = row["configuration"]
        data = objectives[config]
        mean, sd = statistics.fmean(data), statistics.stdev(data)
        half = critical * sd / math.sqrt(1400)
        bounds[(-1, config)] = (mean - half, mean + half)
        for field, value in [
            ("mean_objective", mean),
            ("sd_objective", sd),
            ("half_width_objective", half),
            ("lower_objective", mean - half),
            ("upper_objective", mean + half),
        ]:
            assert math.isclose(
                float(row[field]), value, rel_tol=2e-12, abs_tol=2e-14
            )
    coefficients = {
        "filter_without_code": [1, -1, 0, 0],
        "filter_with_code": [0, 0, 1, -1],
        "code_without_filter": [1, 0, -1, 0],
        "code_with_filter": [0, 1, 0, -1],
        "interaction": [1, -1, -1, 1],
    }
    for row in read_rows(folder / "effects.csv"):
        r, cid = int(row["record_id"]), int(row["joint_condition_id"])
        for name, weights in coefficients.items():
            expected = sum(
                weight * values[(cid, config)][r]
                for config, weight in zip(CONFIGS, weights)
            )
            assert math.isclose(
                float(row[name]), expected, rel_tol=2e-12, abs_tol=2e-13
            )
    contrasts = read_rows(folder / "effect_summary.csv")
    for row in contrasts:
        cid = int(row["joint_condition_id"])
        weights = coefficients[row["effect"]]
        data = [
            sum(
                weight
                * (
                    objectives[config][r]
                    if cid == -1
                    else values[(cid, config)][r]
                )
                for config, weight in zip(CONFIGS, weights)
            )
            for r in range(1400)
        ]
        mean, sd = statistics.fmean(data), statistics.stdev(data)
        half = t.ppf(0.975, 1399) * sd / math.sqrt(1400)
        lower = sum(
            weight * bounds[(cid, config)][0 if weight > 0 else 1]
            for config, weight in zip(CONFIGS, weights)
            if weight
        )
        upper = sum(
            weight * bounds[(cid, config)][1 if weight > 0 else 0]
            for config, weight in zip(CONFIGS, weights)
            if weight
        )
        for field, value in [
            ("mean_reduction", mean),
            ("sd_paired", sd),
            ("simultaneous_lower", lower),
            ("simultaneous_upper", upper),
            ("paired_pointwise95_lower", mean - half),
            ("paired_pointwise95_upper", mean + half),
        ]:
            assert math.isclose(
                float(row[field]), value, rel_tol=3e-11, abs_tol=3e-13
            ), (field, row)
    return {
        "primary_condition_means": len(summaries),
        "objective_bundle_scores": 5600,
        "primary_objective_means": 4,
        "paired_effect_rows": 30800,
        "effect_mean_and_interval_rows": len(contrasts),
    }


def main() -> None:
    """Replay all1400 bundles and audit final merge, leaves and statistics.

    Returns:
        None. Save independent verification without changing the measurements.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    frozen = json.loads((STAGE / "results/validation_freeze.json").read_text())
    for name, expected in frozen["source_sha256"].items():
        assert digest(ROOT / name) == expected
    manifest = json.loads((STAGE / "results/seed_manifest.json").read_text())
    assert (
        digest(STAGE / "results/validation_freeze.json")
        == manifest["validation_freeze_sha256"]
    )
    assert len(manifest["gaussian_inputs"]) == 11200
    assert len(manifest["bsc_mask_groups"]) == 61600
    paths = [
        tuple(r["spawn_key"])
        for r in manifest["gaussian_inputs"] + manifest["bsc_mask_groups"]
        if r["spawn_key"] is not None
    ]
    assert len(paths) == len(set(paths)) == 70000
    assert all(path[1:3] == (4, 1) for path in paths)
    starts = list(range(0, 1400, 20))
    for name in [
        "trials",
        "decomposition",
        "effects",
        "quantiser_banks",
        "pair_checks",
    ]:
        combined = hashlib.sha256()
        for start in starts:
            with (
                STAGE / ".work" / f"records_{start:04d}" / f"{name}.csv"
            ).open("rb") as stream:
                header = stream.readline()
                if start == 0:
                    combined.update(header)
                while block := stream.read(1024 * 1024):
                    combined.update(block)
        assert combined.hexdigest() == digest(
            STAGE / "results" / f"{name}.csv"
        )
    reviewed = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for index, result in enumerate(pool.map(replay_chunk, starts), 1):
            reviewed.append(result)
            print(
                f"R4.3 independent replay: {index * 20}/1400 bundles",
                flush=True,
            )
    stats = verify_statistics(frozen["choices"])
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
        "independent_path_records": sum(r["records"] for r in reviewed),
        "independent_gaussian_inputs": 11200,
        "independent_unique_positive_BSC_masks": 58800,
        "unique_phase1_paths": 70000,
        "full_decomposition_rows_verified": 123200,
        "example_waveforms_verified": sum(
            r["example_waveforms_checked"] for r in reviewed
        ),
        "maximum_independent_MSE_difference": max(
            r["maximum_independent_MSE_difference"] for r in reviewed
        ),
        **stats,
        "receiver_has_no_truth_or_mask_parameter": True,
        "checker_sha256": digest(Path(__file__)),
        "measurements_sha256": {
            p.name: digest(p)
            for p in sorted((STAGE / "results").glob("*.csv"))
        },
        "methods": [
            "direct full convolution",
            "midpoint search quantisation",
            "XOR parity encoding",
            "correction-sphere lookup decoding",
            "SciPy Fourier resample",
            "scalar statistics.fmean/stdev",
            "independent deterministic merge hashing",
        ],
    }
    (STAGE / "results/independent_checks.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print("R4.3: all123200 paths and92 primary means independently verified.")


if __name__ == "__main__":
    main()
