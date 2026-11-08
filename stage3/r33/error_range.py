"""Choose the R3.3 error range and measure its added design conditions."""

from __future__ import annotations

import csv
import json
import math
import platform
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.stats import beta

from degradation import add_gaussian_noise, transmit_bsc
from stage1.r12.sampling_rate import periodic_reconstruct
from stage2.r21.filtering_approach import centred_filter, make_rng, write_csv
from stage2.r23.validation import array_hash, file_hash
from stage3.bitstream import bits_to_labels, labels_to_bits
from stage3.r31.quantisation import dequantise, quantise
from stage3.r32.codes import decode, encode
from stage3.r32.pilot_comparison import bits_hash

ROOT = Path(__file__).resolve().parent
PACKAGE = ROOT.parents[1]


def read_rows(path: Path) -> list[dict[str, str]]:
    """Read a measurement file.

    Args:
        path: Existing CSV.

    Returns:
        Header-keyed records.
    """
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def binary_entropy(probability: float) -> float:
    """Use the coding example binary-entropy formula, with continuous endpoints.

    Args:
        probability: BSC crossover probability.

    Returns:
        Binary entropy in bits per channel use.
    """
    p = float(probability)
    if not 0 <= p <= 1:
        raise ValueError("A probability must lie in [0,1].")
    if p in (0, 1):
        return 0.0
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)


def expectations(probability: float, code: str) -> tuple[float, float]:
    """Evaluate the exact R3.2 data BER and full-message BLER.

    Args:
        probability: Independent flip probability.
        code: Uncoded or the frozen Hamming code.

    Returns:
        Expected real data BER and complete block failure probability.
    """
    p = float(probability)
    if not 0 <= p <= 1:
        raise ValueError("Invalid probability.")
    if code == "uncoded":
        return p, p
    if code != "hamming74":
        raise ValueError("Only the frozen code and uncoded control are used.")
    coefficients = [0, 0, 36, 76, 64, 48, 28, 4]
    ber = (
        sum(a * p**j * (1 - p) ** (7 - j) for j, a in enumerate(coefficients))
        / 4
    )
    bler = sum(
        math.comb(7, j) * p**j * (1 - p) ** (7 - j) for j in range(2, 8)
    )
    return ber, bler


def exact_interval(successes: int, count: int) -> tuple[float, float]:
    """Calculate a pointwise two-sided 95% Clopper-Pearson interval.

    Args:
        successes: Count of independent events.
        count: Number of independent trials.

    Returns:
        Lower and upper binomial-probability bounds.
    """
    if not 0 <= successes <= count or count <= 0:
        raise ValueError("Invalid event count or denominator.")
    low = (
        0.0
        if successes == 0
        else beta.ppf(0.025, successes, count - successes + 1)
    )
    high = (
        1.0
        if successes == count
        else beta.ppf(0.975, successes + 1, count - successes)
    )
    return float(low), float(high)


def bind_parameter(protocol: dict) -> tuple[dict, dict]:
    """Replay one fixed parameter draw after the mapping has been frozen.

    Args:
        protocol: Predeclared range, mapping, conditions and stream paths.

    Returns:
        Updated registry and fully recorded scalar parameter draw.
    """
    registry_path = PACKAGE / "metadata/randomness_design.json"
    registry = json.loads(registry_path.read_text())
    assert registry["root_seed"] == 2470964
    lower, upper = protocol["primary_positive_range"]
    rng = make_rng(tuple(protocol["parameter_path"]))
    initial = rng.bit_generator.state
    uniform = float(rng.random())
    probability = 10 ** (
        math.log10(lower) + uniform * (math.log10(upper) - math.log10(lower))
    )
    existing = registry["bit_flip_baseline_probability"]
    if existing is not None and existing != probability:
        raise ValueError("The registered personal probability changed.")
    registry["bit_flip_probability_range"] = [lower, upper]
    registry["bit_flip_mapping"] = protocol["parameter_mapping"]
    registry["bit_flip_baseline_probability"] = probability
    registry["bit_flip_parameter_uniform_draw"] = uniform
    registry["date"] = protocol["date"]
    registry["status"] = "R3.3 p range and personal mapping frozen"
    registry["course_scope"] = (
        "R2/R3.1/R3.2 preserved; R3.3 phase-0 range study; "
        "R3.4 independent coding validation pending"
    )
    registry["r33_protocol"] = "../stage3/r33/design.json"
    registry["r33_seed_manifest"] = "../stage3/r33/results/seed_manifest.json"
    conditions = registry["bsc_condition_registry"]
    for definition in protocol["new_bsc_conditions"]:
        item = dict(definition)
        if item["probability"] is None:
            item["probability"] = probability
        item.update(
            {
                "study_id": 3,
                "phase_id": 0,
                "gaussian_condition_id": 5,
                "target_snr_db": registry["snr_baseline_db"],
                "scope": "R3.3 phase-0 range study",
            }
        )
        previous = [
            c
            for c in conditions
            if c["study_id"] == 3
            and c["phase_id"] == 0
            and c["condition_id"] == item["condition_id"]
        ]
        if previous:
            if len(previous) != 1 or previous[0] != item:
                raise ValueError("An existing BSC condition was reassigned.")
        else:
            conditions.append(item)
    registry_path.write_text(json.dumps(registry, indent=2) + "\n")
    parameter = {
        "root_seed": 2470964,
        "spawn_key": protocol["parameter_path"],
        "pool_size": 4,
        "bit_generator": "PCG64",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "distribution_call": "random()",
        "scalar_count": 1,
        "initial_state": initial,
        "uniform_draw": uniform,
        "positive_range": [lower, upper],
        "mapping": protocol["parameter_mapping"],
        "personal_probability": probability,
        "protocol_sha256": file_hash(ROOT / "design.json"),
        "grid_or_code_changed_after_draw": False,
        "is_externally_prescribed_mapping": False,
    }
    return registry, parameter


def main() -> None:
    """Reuse the pilot and add only the four registered range-study points.

    Returns:
        None. Save parameter provenance, measurements and the range freeze.
    """
    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    protocol = json.loads((ROOT / "design.json").read_text())
    registry, parameter = bind_parameter(protocol)
    (results / "personal_parameter.json").write_text(
        json.dumps(parameter, indent=2) + "\n"
    )
    qfreeze = json.loads((PACKAGE / protocol["quantiser_freeze"]).read_text())
    cfreeze = json.loads((PACKAGE / protocol["code_freeze"]).read_text())
    assert cfreeze["selected_code"] == "hamming74"
    for frozen in [qfreeze, cfreeze]:
        for name, digest in frozen["source_sha256"].items():
            if file_hash(PACKAGE / name) != digest:
                raise ValueError("A frozen upstream input changed.")
    r32 = PACKAGE / "stage3/r32/results"
    old_manifest = json.loads((r32 / "seed_manifest.json").read_text())
    float_fields = {
        "probability",
        "prechannel_mse_48000",
        "mse_48000",
        "nmse_48000",
        "channel_component_mse_48000",
        "upstream_channel_cross",
    }
    trials = []
    for raw in read_rows(r32 / "trials.csv"):
        if raw["code"] not in protocol["code_names"]:
            continue
        row = {
            k: (
                v if k == "code" else float(v) if k in float_fields else int(v)
            )
            for k, v in raw.items()
        }
        row["origin"] = "r32_existing"
        trials.append(row)
    weight_rows = [
        {
            k: (
                v
                if k == "code"
                else float(v) if k == "probability" else int(v)
            )
            for k, v in raw.items()
        }
        for raw in read_rows(r32 / "block_weights.csv")
        if raw["code"] in protocol["code_names"]
    ]
    channel_manifest = [
        dict(c, origin="r32_existing")
        for c in old_manifest["channel_trials"]
        if c["code"] in protocol["code_names"]
    ]
    reference = np.load(PACKAGE / "reference_signal.npy")
    x = np.load(PACKAGE / "stage1/r12/results/selected_samples.npy")
    h = np.load(PACKAGE / "stage2/r22/results/selected_coefficients.npy")
    bits, a = qfreeze["bits"], qfreeze["half_range"]
    new_ids = [c["condition_id"] for c in protocol["new_bsc_conditions"]]
    new_conditions = [
        c
        for c in registry["bsc_condition_registry"]
        if c["study_id"] == 3
        and c["phase_id"] == 0
        and c["condition_id"] in new_ids
    ]
    max_decomposition = 0.0
    for group in protocol["gaussian_repeat_ids"]:
        noisy, noise, _ = add_gaussian_noise(
            x, registry["snr_baseline_db"], make_rng((1, 3, 0, 5, group))
        )
        bank = next(
            c
            for c in old_manifest["input_banks"]
            if c["gaussian_repeat_id"] == group
        )
        assert array_hash(noise) == bank["noise_sha256"]
        z = centred_filter(noisy, h)
        labels, q, _ = quantise(z, bits, a, qfreeze["normalised_epsilon"])
        payload = labels_to_bits(labels, bits)
        assert bits_hash(payload) == bank["payload_sha256"]
        before = periodic_reconstruct(q, len(reference))
        upstream = before - reference
        baseline = float(np.mean(upstream**2))
        for name in protocol["code_names"]:
            n, k = (1, 1) if name == "uncoded" else (7, 4)
            radius = 0 if name == "uncoded" else 1
            sent, padding = encode(payload, name)
            assert padding == 0
            for condition in new_conditions:
                cid, probability = (
                    condition["condition_id"],
                    condition["probability"],
                )
                for b in range(protocol["bit_repeats_per_gaussian"]):
                    repeat = 8 * group + b
                    path = (2, 3, 0, cid, repeat, protocol["layout_ids"][name])
                    rng = make_rng(path)
                    initial = rng.bit_generator.state
                    received, mask = transmit_bsc(sent, probability, rng)
                    data, _ = decode(received, name)
                    wrong = data != payload
                    failed = int(
                        np.count_nonzero(np.any(wrong.reshape(-1, k), axis=1))
                    )
                    weights = np.sum(mask.reshape(-1, n), axis=1)
                    outside = int(np.count_nonzero(weights > radius))
                    assert failed == outside
                    recovered = dequantise(bits_to_labels(data, bits), bits, a)
                    high = periodic_reconstruct(recovered, len(reference))
                    added = high - before
                    mse = float(np.mean((high - reference) ** 2))
                    component = float(np.mean(added**2))
                    cross = float(2 * np.mean(upstream * added))
                    max_decomposition = max(
                        max_decomposition,
                        abs(mse - baseline - component - cross),
                    )
                    trials.append(
                        {
                            "code": name,
                            "condition_id": cid,
                            "probability": probability,
                            "gaussian_repeat_id": group,
                            "bit_repeat_id": b,
                            "repeat_id": repeat,
                            "layout_id": protocol["layout_ids"][name],
                            "payload_bits": len(payload),
                            "transmitted_bits": len(sent),
                            "padding_bits": 0,
                            "channel_flips": int(np.count_nonzero(mask)),
                            "decoded_wrong_bits": int(np.count_nonzero(wrong)),
                            "block_count": len(payload) // k,
                            "full_message_block_failures": failed,
                            "payload_block_failures": failed,
                            "outside_correction_radius_blocks": outside,
                            "wrong_samples": int(
                                np.count_nonzero(
                                    np.any(wrong.reshape(-1, bits), axis=1)
                                )
                            ),
                            "wrong_msb_bits": int(
                                np.count_nonzero(wrong.reshape(-1, bits)[:, 0])
                            ),
                            "prechannel_mse_48000": baseline,
                            "mse_48000": mse,
                            "nmse_48000": mse / float(np.mean(reference**2)),
                            "channel_component_mse_48000": component,
                            "upstream_channel_cross": cross,
                            "origin": "r33_new",
                        }
                    )
                    histogram = np.bincount(
                        weights.astype(int), minlength=n + 1
                    )
                    for weight, count in enumerate(histogram):
                        weight_rows.append(
                            {
                                "code": name,
                                "condition_id": cid,
                                "probability": probability,
                                "gaussian_repeat_id": group,
                                "bit_repeat_id": b,
                                "weight": weight,
                                "blocks": int(count),
                            }
                        )
                    channel_manifest.append(
                        {
                            "code": name,
                            "condition_id": cid,
                            "gaussian_repeat_id": group,
                            "bit_repeat_id": b,
                            "spawn_key": list(path),
                            "initial_state": initial,
                            "probability": probability,
                            "uniform_call_shape": [len(sent)],
                            "transmitted_sha256": bits_hash(sent),
                            "received_sha256": bits_hash(received),
                            "decoded_real_payload_sha256": bits_hash(data),
                            "pairing_group": f"r33-g{group}-b{b}",
                            "identity_no_rng": False,
                            "origin": "r33_new",
                        }
                    )
    trials.sort(
        key=lambda r: (
            r["code"],
            r["condition_id"],
            r["gaussian_repeat_id"],
            r["bit_repeat_id"],
        )
    )
    write_csv(results / "trials.csv", trials)
    write_csv(results / "block_weights.csv", weight_rows)
    summary = []
    for name in protocol["code_names"]:
        conditions = sorted(
            {
                (r["condition_id"], r["probability"])
                for r in trials
                if r["code"] == name
            },
            key=lambda pair: pair[1],
        )
        for cid, probability in conditions:
            rows = [
                r
                for r in trials
                if r["code"] == name and r["condition_id"] == cid
            ]
            flips = sum(r["channel_flips"] for r in rows)
            tx = sum(r["transmitted_bits"] for r in rows)
            wrong = sum(r["decoded_wrong_bits"] for r in rows)
            payload = sum(r["payload_bits"] for r in rows)
            failed = sum(r["full_message_block_failures"] for r in rows)
            blocks = sum(r["block_count"] for r in rows)
            channel_low, channel_high = exact_interval(flips, tx)
            block_low, block_high = exact_interval(failed, blocks)
            theory_ber, theory_bler = expectations(probability, name)
            role = (
                "identity"
                if probability == 0
                else (
                    "stress"
                    if probability > 0.1
                    else "personal" if cid == 11 else "primary_grid"
                )
            )
            summary.append(
                {
                    "code": name,
                    "condition_id": cid,
                    "role": role,
                    "probability": probability,
                    "records": len(rows),
                    "payload_bits": payload,
                    "transmitted_bits": tx,
                    "channel_flips": flips,
                    "wrong_data_bits": wrong,
                    "blocks": blocks,
                    "failed_blocks": failed,
                    "channel_ber": flips / tx,
                    "channel_ci95_low": channel_low,
                    "channel_ci95_high": channel_high,
                    "data_ber": wrong / payload,
                    "theory_data_ber": theory_ber,
                    "block_failure_fraction": failed / blocks,
                    "block_ci95_low": block_low,
                    "block_ci95_high": block_high,
                    "theory_block_failure_probability": theory_bler,
                    "expected_failed_blocks": blocks * theory_bler,
                    "probability_zero_failed_blocks": (
                        math.exp(blocks * math.log1p(-theory_bler))
                        if theory_bler < 1
                        else 0.0
                    ),
                    "zero_count_one_sided95_upper": -math.expm1(
                        math.log(0.05) / blocks
                    ),
                    "capacity": 1 - binary_entropy(probability),
                    "mean_mse_48000": float(
                        np.mean([r["mse_48000"] for r in rows])
                    ),
                    "mean_prechannel_mse_48000": float(
                        np.mean([r["prechannel_mse_48000"] for r in rows])
                    ),
                }
            )
    write_csv(results / "summary.csv", summary)
    capacity_crossing = brentq(lambda p: 1 - binary_entropy(p) - 4 / 7, 0, 0.5)
    ber_crossing = brentq(
        lambda p: expectations(p, "hamming74")[0] - p, 0.1, 0.3
    )
    sources = [
        "stage3/r33/design.json",
        "stage3/r33/error_range.py",
        "stage3/r32/results/code_freeze.json",
        "stage3/r32/results/seed_manifest.json",
        "stage3/r32/results/trials.csv",
        "stage3/r32/results/block_weights.csv",
        "stage3/r31/results/quantiser_freeze.json",
        "stage3/r32/codes.py",
        "stage3/bitstream.py",
        "degradation.py",
        "stage1/r12/sampling_rate.py",
        "stage2/r21/filtering_approach.py",
        "reference_signal.npy",
        "stage2/r22/results/selected_coefficients.npy",
    ]
    hashes = {p: file_hash(PACKAGE / p) for p in sources}
    freeze = {
        "date": protocol["date"],
        "status": "R3.3 range frozen",
        "primary_positive_range": protocol["primary_positive_range"],
        "primary_grid": protocol["primary_grid"],
        "identity_probability": 0.0,
        "stress_probabilities": protocol["stress_probabilities"],
        "personal_probability": parameter["personal_probability"],
        "personal_uniform_draw": parameter["uniform_draw"],
        "parameter_path": parameter["spawn_key"],
        "mapping": protocol["parameter_mapping"],
        "capacity_equals_code_rate_at": capacity_crossing,
        "exact_data_ber_equals_uncoded_at": ber_crossing,
        "code": "hamming74",
        "code_changed": False,
        "bits": bits,
        "half_range": a,
        "source_sha256": hashes,
        "r34_success_threshold_frozen": False,
        "phase_1_or_study_4_used": False,
        "range_is_not_a_quality_success_interval": True,
    }
    (results / "range_freeze.json").write_text(
        json.dumps(freeze, indent=2) + "\n"
    )
    manifest = {
        "root_seed": 2470964,
        "pool_size": 4,
        "bit_generator": "PCG64",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scope": "R3.3 phase-0 study, with imported R3.2 evidence",
        "parameter": parameter,
        "input_banks": old_manifest["input_banks"],
        "channel_trials": channel_manifest,
        "source_sha256": hashes,
    }
    (results / "seed_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    paths = [
        tuple(c["spawn_key"])
        for c in channel_manifest
        if not c["identity_no_rng"]
    ]
    fresh = [c for c in channel_manifest if c["origin"] == "r33_new"]
    assert len(paths) == len(set(paths)) == 880
    assert len(trials) == 890 and len(fresh) == 320
    assert all(p[:3] == (2, 3, 0) for p in paths)
    assert max_decomposition < 1e-11
    verification = {
        "date": protocol["date"],
        "status": "passed",
        "total_trial_rows": len(trials),
        "imported_r32_rows": 570,
        "new_bsc_masks": len(fresh),
        "unique_total_bsc_masks": len(paths),
        "gaussian_input_banks_replayed": 5,
        "parameter_scalar_count": 1,
        "personal_probability": parameter["personal_probability"],
        "maximum_new_mse_decomposition_residual": max_decomposition,
        "all_upstream_frozen_hashes_match": True,
        "no_r3_phase1_or_r4_paths": True,
    }
    (PACKAGE / "metadata/r33_verification.json").write_text(
        json.dumps(verification, indent=2) + "\n"
    )
    print(
        f"R3.3: p_ID={parameter['personal_probability']:.12g}; "
        "890 cases (570 imported, 320 new); range frozen."
    )


if __name__ == "__main__":
    main()
