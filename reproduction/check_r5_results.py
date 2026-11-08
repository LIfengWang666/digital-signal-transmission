"""Independently replay R5 variants, exact framing and paired statistics."""

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
from reproduction.check_r42_results import binary_hash
from stage5.pipeline import receive_variant

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "stage5"


def sha(path: Path) -> str:
    """Hash a stored source or measurement file.

    Args:
        path: Existing file.

    Returns:
        SHA-256 of its unchanged bytes.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def direct_filter(values: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Compute finite full convolution with the variant's predetermined crop.

    Args:
        values: Full32000-sample record, zero outside its interval.
        kernel: Frozen odd-length symmetric variant FIR.

    Returns:
        The32000 aligned output samples without FFT convolution.
    """
    delay = (len(kernel) - 1) // 2
    stop = delay + len(values)
    return np.convolve(values, kernel, mode="full")[delay:stop]


def replay_chunk(task: tuple[int, int]) -> dict:
    """Replay every variant/path in one stored pilot or confirmation chunk.

    Args:
        task: Phase and first complete-record ID.

    Returns:
        Independent counts and maximum full-grid metric discrepancies.
    """
    phase, start = task
    folder = HERE / ".work" / f"phase{phase}" / f"records_{start:04d}"
    completed = json.loads((folder / "complete.json").read_text())
    count = completed["stop"] - start
    inputs = {name: sha(folder / name) for name in completed["sha256"]}
    assert inputs == completed["sha256"]
    cache = HERE / ".work" / f"independent_phase{phase}_{start:04d}.json"
    if cache.exists():
        prior = json.loads(cache.read_text())
        if prior["input_sha256"] == inputs and prior["checker_sha256"] == sha(
            Path(__file__)
        ):
            return prior
    frozen = json.loads(
        (HERE / "results" / f"phase{phase}_freeze.json").read_text()
    )
    reference = np.load(ROOT / "reference_signal.npy")
    x, power = reference[::3], float(np.mean(reference**2))
    kernels = {
        c["id"]: np.load(HERE / "results" / f"kernel_{c['id']}.npy")
        for c in frozen["design"]["kernels"]
    }
    definitions = {
        r["code"]: r
        for r in json.loads(
            (ROOT / "stage3/r32/results/code_definitions.json").read_text()
        )
    }
    tables = {
        name: independent_tables(definitions[name])
        for name in ["uncoded", "hamming74", "hamming1511", "repetition51"]
    }
    cases = {
        (int(r["record_id"]), int(r["joint_condition_id"]), r["variant"]): r
        for r in read_rows(folder / "trials.csv")
    }
    leaves = json.loads((folder / "leaves.json").read_text())
    channels = {
        (r["record_id"], r["joint_condition_id"], r["layout_id"]): r
        for r in leaves["bsc"]
    }
    assert len(cases) == 135 * count
    checked, masks, paths, maximum = 0, 0, set(), 0.0
    for gaussian in leaves["gaussian"]:
        repeat, scid = gaussian["record_id"], gaussian["snr_condition_id"]
        path = (1, 5, phase, scid, repeat)
        assert list(path) == gaussian["spawn_key"] and path not in paths
        paths.add(path)
        rng = np.random.Generator(
            np.random.PCG64(
                np.random.SeedSequence(2470964, spawn_key=path, pool_size=4)
            )
        )
        assert rng.bit_generator.state == gaussian["initial_state"]
        variance = np.mean(x**2) / 10 ** (gaussian["snr_db"] / 10)
        noise = rng.normal(0, np.sqrt(variance), x.shape)
        assert sha_bytes(noise.astype("<f8")) == gaussian["noise_sha256"]
        assert variance == gaussian["model_variance"]
        quantisers, drawn = {}, {}
        for variant in frozen["variants"]:
            name, layout, width, code = (
                variant["id"],
                variant["layout"],
                variant["bits"],
                variant["code"],
            )
            key = variant["filter_id"], width
            if key not in quantisers:
                z = direct_filter(x + noise, kernels[key[0]])
                levels = -18 + 36 / (2**width - 1) * np.arange(2**width)
                midpoints = (levels[1:] + levels[:-1]) / 2
                labels = np.searchsorted(
                    midpoints, np.clip(z, -18, 18) - 18e-9, side="left"
                )
                payload = (
                    ((labels[:, None] >> np.arange(width - 1, -1, -1)) & 1)
                    .astype(np.uint8)
                    .reshape(-1)
                )
                before = resample(levels[labels], 96000)
                quantisers[key] = z, levels, payload, before
            z, levels, payload, before = quantisers[key]
            n, k = definitions[code]["n"], definitions[code]["k"]
            padding = (-len(payload)) % k
            messages = np.pad(payload, (0, padding)).reshape(-1, k)
            parity, lookup = tables[code]
            parity_bits = np.zeros((len(messages), n - k), dtype=np.uint8)
            for column in range(n - k):
                parity_bits[:, column] = np.bitwise_xor.reduce(
                    messages[:, parity[:, column].astype(bool)], axis=1
                )
            sent = np.concatenate((messages, parity_bits), axis=1).reshape(-1)
            for condition in frozen["design"]["conditions"]:
                if condition["snr_condition_id"] != scid:
                    continue
                cid, p = (
                    condition["joint_condition_id"],
                    condition["probability"],
                )
                row = cases[(repeat, cid, name)]
                if (cid, layout) not in drawn:
                    path = (2, 5, phase, cid, repeat, layout)
                    record = channels[(repeat, cid, layout)]
                    assert (
                        list(path) == record["spawn_key"] and path not in paths
                    )
                    paths.add(path)
                    rng = np.random.Generator(
                        np.random.PCG64(
                            np.random.SeedSequence(
                                2470964, spawn_key=path, pool_size=4
                            )
                        )
                    )
                    assert rng.bit_generator.state == record["initial_state"]
                    mask = rng.random(sent.shape) < p
                    assert binary_hash(mask) == record["mask_sha256"]
                    drawn[(cid, layout)] = mask
                    masks += 1
                mask = drawn[(cid, layout)]
                received = sent ^ mask.astype(np.uint8)
                index = received.reshape(-1, n).astype(np.uint32) @ (
                    2 ** np.arange(n - 1, -1, -1, dtype=np.uint32)
                )
                real_data = lookup[index].reshape(-1)[: len(payload)]
                wrong = real_data != payload
                for field, data in [
                    ("transmitted_sha256", sent),
                    ("received_sha256", received),
                    ("payload_sha256", real_data),
                    ("wrong_pattern_sha256", wrong),
                    ("mask_sha256", mask),
                ]:
                    assert binary_hash(data) == row[field], (
                        phase,
                        repeat,
                        cid,
                        name,
                        field,
                    )
                expected_counts = {
                    "channel_flips": np.count_nonzero(mask),
                    "decoded_wrong_bits": np.count_nonzero(wrong),
                    "wrong_samples": np.count_nonzero(
                        np.any(wrong.reshape(-1, width), axis=1)
                    ),
                    "wrong_msb_bits": np.count_nonzero(
                        wrong.reshape(-1, width)[:, 0]
                    ),
                    "payload_perfect": int(not np.any(wrong)),
                    "payload_bits": len(payload),
                    "transmitted_bits": len(sent),
                    "padding_bits": padding,
                    "overload_samples": np.count_nonzero(np.abs(z) > 18),
                }
                assert all(
                    int(row[key]) == value
                    for key, value in expected_counts.items()
                )
                labels = real_data.reshape(-1, width).astype(np.uint32) @ (
                    2 ** np.arange(width - 1, -1, -1, dtype=np.uint32)
                )
                high = resample(levels[labels], 96000)
                error, pre, channel = (
                    high - reference,
                    before - reference,
                    high - before,
                )
                mse = float(np.mean(error**2))
                expected = {
                    "mse": mse,
                    "nmse": mse / power,
                    "mae": float(np.mean(np.abs(error))),
                    "rmse": math.sqrt(mse),
                    "mean_error": float(np.mean(error)),
                    "error_variance_ddof0": float(np.var(error)),
                    "peak_absolute_error": float(np.max(np.abs(error))),
                    "prechannel_mse": float(np.mean(pre**2)),
                    "channel_error_power": float(np.mean(channel**2)),
                    "channel_cross": float(2 * np.mean(pre * channel)),
                }
                for field, value in expected.items():
                    assert math.isclose(
                        float(row[field]), value, rel_tol=5e-10, abs_tol=3e-11
                    ), (field, row, value)
                maximum = max(maximum, abs(mse - float(row["mse"])))
                assert float(row["power_identity_residual"]) < 1e-9
                checked += 1
    assert (
        checked == 135 * count
        and masks == 75 * count
        and len(paths) == 81 * count
    )
    result = {
        "status": "passed",
        "phase": phase,
        "start": start,
        "bundles": count,
        "independent_paths": checked,
        "unique_masks": masks,
        "unique_random_leaves": len(paths),
        "maximum_independent_MSE_difference": maximum,
        "input_sha256": inputs,
        "checker_sha256": sha(Path(__file__)),
    }
    cache.write_text(json.dumps(result, indent=2) + "\n")
    return result


def sha_bytes(values: np.ndarray) -> str:
    """Hash independently generated raw numeric samples.

    Args:
        values: Explicit byte-order numeric array.

    Returns:
        SHA-256 of raw contiguous bytes.
    """
    return hashlib.sha256(values.tobytes()).hexdigest()


def verify_kernels(frozen: dict) -> dict:
    """Check FIR coefficients by the finite IDFT cosine sum and Hann formula.

    Args:
        frozen: Prespecified filter choices and saved feasibility rows.

    Returns:
        Independent construction and gain-guard discrepancies for all5 FIRs.
    """
    design = frozen["design"]
    frequencies = np.fft.fftfreq(32000, 1 / 16000)
    maximum = 0.0
    for choice in design["kernels"]:
        mask = np.zeros(32000, bool)
        for centre in design["centres_hz"]:
            mask |= (
                np.abs(np.abs(frequencies) - centre) <= choice["half_width_hz"]
            )
        length = choice["taps"]
        lags = np.arange(length) - (length - 1) // 2
        impulse = (
            np.mean(
                np.cos(2 * np.pi * lags[:, None] * frequencies[mask] / 16000),
                axis=1,
            )
            * np.count_nonzero(mask)
            / 32000
        )
        hann = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(length) / (length - 1))
        h = np.load(HERE / "results" / f"kernel_{choice['id']}.npy")
        difference = float(np.max(np.abs(h - impulse * hann)))
        assert difference < 5e-13
        maximum = max(maximum, difference)
        guard_f = (
            np.array(design["centres_hz"])[:, None]
            + np.array(design["guard_offsets_hz"])
        ).reshape(-1)
        response = np.cos(2 * np.pi * guard_f[:, None] * lags / 16000) @ h
        deviation = float(np.max(np.abs(np.abs(response) - 1)))
        for variant in frozen["variants"]:
            if variant["filter_id"] == choice["id"]:
                assert math.isclose(
                    deviation, variant["gain_max_deviation"], abs_tol=2e-13
                )
                assert (
                    bool(deviation <= 0.01 and np.all(response > 0))
                    == variant["gain_guard_feasible"]
                )
    return {
        "kernels_verified": 5,
        "maximum_direct_IDFT_coefficient_difference": maximum,
    }


def verify_statistics(phase: int, frozen: dict) -> dict:
    """Recompute all19 declared statistics with scalar independent sums/SDs.

    Args:
        phase:0 for planning or1 for confirmation.
        frozen: Saved design and independent-bundle count.

    Returns:
        Number of checked primary arrays and the planning/validation count.
    """
    folder = HERE / "results"
    count = frozen["records_per_condition"]
    trials = read_rows(folder / f"phase{phase}_trials.csv")
    cases = {
        (
            int(r["record_id"]),
            int(r["joint_condition_id"]),
            r["variant"],
        ): float(r["nmse"])
        for r in trials
    }
    names = [v["id"] for v in frozen["variants"]]
    objectives = {
        name: [
            statistics.fmean(
                cases[(r, cid, name)] for cid in frozen["design"]["main_ids"]
            )
            for r in range(count)
        ]
        for name in names
    }
    data = {}
    for name in names:
        if name == "base":
            continue
        data[f"gain:J:{name}"] = [
            a - b for a, b in zip(objectives["base"], objectives[name])
        ]
        data[f"gain:personal:{name}"] = [
            cases[(r, 11, "base")] - cases[(r, 11, name)] for r in range(count)
        ]
    for name in ["base", "h1511", "uncoded"]:
        data[f"mean:personal:{name}"] = [
            cases[(r, 11, name)] for r in range(count)
        ]
    saved = read_rows(
        folder
        / ("phase0_primary.csv" if phase == 0 else "primary_summary.csv")
    )
    assert len(data) == len(saved) == 19
    previous_fails = False
    planned = 0
    for row in saved:
        values = data[row["statistic"]]
        mean, sd = statistics.fmean(values), statistics.stdev(values)
        if phase == 0:
            planned = int(row["records"])
            half = t.ppf(1 - 0.05 / 38, planned - 1) * sd / math.sqrt(planned)
            target = max(0.01, 0.05 * abs(mean))
            assert math.isclose(
                float(row["pilot_mean"]), mean, rel_tol=2e-11, abs_tol=1e-13
            )
            assert math.isclose(
                float(row["pilot_sd"]), sd, rel_tol=2e-11, abs_tol=1e-13
            )
            assert math.isclose(
                float(row["predicted_half_width"]),
                half,
                rel_tol=2e-11,
                abs_tol=1e-13,
            )
            assert half <= target
            if planned > 200:
                previous_fails |= (
                    t.ppf(1 - 0.05 / 38, planned - 51)
                    * sd
                    / math.sqrt(planned - 50)
                    > target
                )
        else:
            half = t.ppf(1 - 0.05 / 38, count - 1) * sd / math.sqrt(count)
            for field, value in [
                ("mean", mean),
                ("sd", sd),
                ("half_width", half),
                ("lower", mean - half),
                ("upper", mean + half),
            ]:
                assert math.isclose(
                    float(row[field]), value, rel_tol=3e-11, abs_tol=2e-13
                ), (field, row)
    if phase == 0:
        assert (
            planned >= 200
            and planned % 50 == 0
            and (planned == 200 or previous_fails)
        )
    return {
        "primary_statistics_verified": 19,
        "planned_confirmation_count_verified": (
            planned if phase == 0 else count
        ),
    }


def main() -> None:
    """Audit deterministic merge and replay every new pilot/confirmation path.

    Returns:
        None. Save an independent record without changing source data.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", type=int, choices=[0, 1], default=1)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    folder = HERE / "results"
    frozen = json.loads(
        (folder / f"phase{args.phase}_freeze.json").read_text()
    )
    for name, expected in frozen["source_sha256"].items():
        assert sha(ROOT / name) == expected, name
    count = frozen["records_per_condition"]
    kernels = verify_kernels(frozen)
    starts = list(range(0, count, 20))
    for name, final in [
        ("trials.csv", f"phase{args.phase}_trials.csv"),
        ("quantiser_inputs.csv", f"phase{args.phase}_quantiser_inputs.csv"),
    ]:
        combined = hashlib.sha256()
        for start in starts:
            with (
                HERE
                / ".work"
                / f"phase{args.phase}"
                / f"records_{start:04d}"
                / name
            ).open("rb") as stream:
                header = stream.readline()
                if start == 0:
                    combined.update(header)
                while block := stream.read(1024 * 1024):
                    combined.update(block)
        assert combined.hexdigest() == sha(folder / final)
    reviewed = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for index, result in enumerate(
            pool.map(replay_chunk, [(args.phase, start) for start in starts]),
            1,
        ):
            reviewed.append(result)
            print(
                f"R5 phase{args.phase} independent: "
                f"{min(index * 20, count)}/{count} bundles",
                flush=True,
            )
    assert list(inspect.signature(receive_variant).parameters) == [
        "received",
        "code",
        "bits",
        "payload_bits",
    ]
    stats = verify_statistics(args.phase, frozen)
    manifest = json.loads(
        (folder / f"phase{args.phase}_seed_manifest.json").read_text()
    )
    assert manifest["freeze_sha256"] == sha(
        folder / f"phase{args.phase}_freeze.json"
    )
    paths = [
        tuple(r["spawn_key"])
        for r in manifest["gaussian_inputs"] + manifest["bsc_mask_groups"]
    ]
    assert len(paths) == len(set(paths)) == 81 * count
    assert all(path[1:3] == (5, args.phase) for path in paths)
    result = {
        "status": "passed",
        "phase_id": args.phase,
        "complete_bundles": count,
        "independent_path_records": sum(
            r["independent_paths"] for r in reviewed
        ),
        "unique_random_leaves": 81 * count,
        "unique_Gaussian_inputs": 6 * count,
        "unique_BSC_masks": 75 * count,
        "unique_quantiser_inputs": 36 * count,
        "maximum_independent_MSE_difference": max(
            r["maximum_independent_MSE_difference"] for r in reviewed
        ),
        **stats,
        **kernels,
        "truth_free_receiver_signature": True,
        "checker_sha256": sha(Path(__file__)),
        "methods": [
            "direct full convolution",
            "midpoint threshold quantisation",
            "XOR parity encoding",
            "independent correction-sphere lookup",
            "SciPy Fourier resampling",
            "scalar fmean/stdev and fixed-unit planning",
        ],
    }
    (folder / f"phase{args.phase}_independent_checks.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(
        f"R5 phase{args.phase}: all{135 * count} paths independently verified."
    )


if __name__ == "__main__":
    main()
