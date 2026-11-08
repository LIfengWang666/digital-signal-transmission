"""Measure R3.2 pilot error profiles and select a precisely defined code."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from math import comb
from pathlib import Path

import numpy as np
from scipy.stats import t

ROOT = Path(__file__).resolve().parent
PACKAGE = ROOT.parents[1]
if str(PACKAGE) not in sys.path:
    sys.path.insert(0, str(PACKAGE))

from degradation import add_gaussian_noise, transmit_bsc  # noqa: E402
from stage1.r12.sampling_rate import periodic_reconstruct  # noqa: E402
from stage2.r21.filtering_approach import (  # noqa: E402
    centred_filter,
    make_rng,
    write_csv,
)
from stage2.r23.validation import array_hash, file_hash  # noqa: E402
from stage3.bitstream import bits_to_labels, labels_to_bits  # noqa: E402
from stage3.r31.quantisation import dequantise, quantise  # noqa: E402
from stage3.r32.codes import decode, encode  # noqa: E402


def bits_hash(bits: np.ndarray) -> str:
    """Hash contiguous binary values without an array header.

    Args:
        bits: Actual binary values.

    Returns:
        SHA-256 of flattened uint8 bytes.
    """
    return hashlib.sha256(
        np.asarray(bits, dtype=np.uint8).tobytes()
    ).hexdigest()


def score_candidates(
    trials: list[dict],
    definitions: list[dict],
    protocol: dict,
) -> tuple[str, list[dict], list[dict], list[dict]]:
    """Apply the predeclared input-bank loss and transmission budget.

    Args:
        trials: Actual full-record pilot measurements.
        definitions: Precisely defined candidate codes.
        protocol: Predeclared score, probability set and cost tolerance.

    Returns:
        Selected code, per-bank scores, code summaries and budget sensitivity.
    """
    scores = []
    for spec in definitions:
        name = spec["code"]
        for group in protocol["gaussian_repeat_ids"]:
            ratios = []
            for probability in protocol["selection_probabilities"]:
                coded = [
                    r["mse_48000"]
                    for r in trials
                    if r["code"] == name
                    and r["probability"] == probability
                    and r["gaussian_repeat_id"] == group
                ]
                uncoded = [
                    r["mse_48000"]
                    for r in trials
                    if r["code"] == "uncoded"
                    and r["probability"] == probability
                    and r["gaussian_repeat_id"] == group
                ]
                ratios.append(float(np.mean(coded) / np.mean(uncoded)))
            scores.append(
                {
                    "code": name,
                    "gaussian_repeat_id": group,
                    "normalised_mse_score": float(np.mean(ratios)),
                }
            )
    summaries = []
    for spec in definitions:
        name = spec["code"]
        values = np.array(
            [r["normalised_mse_score"] for r in scores if r["code"] == name]
        )
        tx = next(r["transmitted_bits"] for r in trials if r["code"] == name)
        data = trials[0]["payload_bits"]
        sd = float(np.std(values, ddof=1))
        half = float(t.ppf(0.975, len(values) - 1)) * sd / np.sqrt(len(values))
        summaries.append(
            {
                "code": name,
                "n": spec["n"],
                "k": spec["k"],
                "d_min": spec["d_min"],
                "t": spec["t"],
                "score": float(np.mean(values)),
                "score_sd": sd,
                "score_ci95_low": float(np.mean(values) - half),
                "score_ci95_high": float(np.mean(values) + half),
                "payload_expansion": tx / data,
                "transmitted_bits": tx,
                "transmitted_bits_per_second": tx / 2,
                "effective_rate": data / tx,
                "eligible": int(
                    name != "uncoded"
                    and tx / data <= protocol["max_payload_expansion"]
                ),
            }
        )
    eligible = [r for r in summaries if r["eligible"] and r["score"] < 1]
    if not eligible:
        raise ArithmeticError("No ECC improves the declared budgeted loss.")
    selected = min(eligible, key=lambda r: r["score"])["code"]
    sensitivity = []
    for budget in protocol["budget_sensitivity"]:
        allowed = [
            r
            for r in summaries
            if r["code"] != "uncoded"
            and r["payload_expansion"] <= budget
            and r["score"] < 1
        ]
        winner = min(allowed, key=lambda r: r["score"]) if allowed else None
        sensitivity.append(
            {
                "max_expansion": budget,
                "selected_code": winner["code"] if winner else "none",
                "selected_score": winner["score"] if winner else "",
            }
        )
    return selected, scores, summaries, sensitivity


def main() -> None:
    """Run the recorded design experiment and freeze the selected code.

    Returns:
        None. Save true counts, reconstruction metrics, sources and RNG paths.
    """
    protocol = json.loads((ROOT / "design.json").read_text())
    registry = json.loads(
        (PACKAGE / "metadata/randomness_design.json").read_text()
    )
    results = ROOT / "results"
    definitions = json.loads((results / "code_definitions.json").read_text())
    coefficients = np.genfromtxt(
        results / "error_pattern_coefficients.csv",
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8",
    )
    qfreeze_path = PACKAGE / protocol["quantiser_freeze"]
    qfreeze = json.loads(qfreeze_path.read_text())
    for name, digest in qfreeze["source_sha256"].items():
        if file_hash(PACKAGE / name) != digest:
            raise ValueError("An upstream frozen R3.1 input changed.")
    h_path = PACKAGE / "stage2/r22/results/selected_coefficients.npy"
    if file_hash(h_path) != protocol["coefficient_sha256"]:
        raise ValueError("The frozen filter changed.")
    for condition in protocol["pilot_conditions"]:
        registered = [
            c
            for c in registry["bsc_condition_registry"]
            if c["study_id"] == 3
            and c["phase_id"] == 0
            and c["condition_id"] == condition["condition_id"]
        ]
        if len(registered) != 1 or (
            registered[0]["probability"] != condition["probability"]
        ):
            raise ValueError("A BSC condition is not registered.")
    h = np.load(h_path)
    reference = np.load(PACKAGE / "reference_signal.npy")
    x = np.load(PACKAGE / "stage1/r12/results/selected_samples.npy")
    bits = qfreeze["bits"]
    a = qfreeze["half_range"]
    epsilon = qfreeze["normalised_epsilon"]
    snr = registry["snr_baseline_db"]
    r31_manifest = json.loads(
        (PACKAGE / "stage3/r31/results/seed_manifest.json").read_text()
    )
    trials, weights_rows, channel_manifest, input_manifest = [], [], [], []
    plane_counts = {}
    example_high, example_low = {}, {}
    max_decomposition = 0.0
    zero_control_error = 0.0
    total_primary_masks = 0
    for group in protocol["gaussian_repeat_ids"]:
        gaussian_path = (1, 3, 0, 5, group)
        rng = make_rng(gaussian_path)
        noisy, noise, _ = add_gaussian_noise(x, snr, rng)
        old = next(
            r
            for r in r31_manifest["trial_paths"]
            if r["spawn_key"] == list(gaussian_path)
        )
        if array_hash(noise) != old["noise_array_sha256"]:
            raise ArithmeticError("The R3.1 input bank did not replay.")
        z = centred_filter(noisy, h)
        labels, q, _ = quantise(z, bits, a, epsilon)
        payload = labels_to_bits(labels, bits)
        pre_high = periodic_reconstruct(q, len(reference))
        pre_error = pre_high - reference
        baseline = float(np.mean(pre_error**2))
        input_manifest.append(
            {
                "gaussian_repeat_id": group,
                "gaussian_path": list(gaussian_path),
                "noise_sha256": array_hash(noise),
                "quantiser_input_sha256": array_hash(z),
                "payload_sha256": bits_hash(payload),
                "payload_bits": len(payload),
                "prechannel_mse_48000": baseline,
            }
        )
        for spec in definitions:
            name = spec["code"]
            n, k, t_correct = spec["n"], spec["k"], spec["t"]
            transmitted, padding = encode(payload, name)
            padded = np.pad(payload, (0, padding)).reshape(-1, k)
            for condition in protocol["pilot_conditions"]:
                probability = condition["probability"]
                cid = condition["condition_id"]
                repeats = protocol[
                    (
                        "identity_bit_repeats_per_gaussian"
                        if probability == 0
                        else "bit_repeats_per_gaussian"
                    )
                ]
                for bit_repeat in range(repeats):
                    repeat = protocol["bit_repeats_per_gaussian"] * group
                    repeat += bit_repeat
                    layout = protocol["layout_ids"][name]
                    path = (2, 3, 0, cid, repeat, layout)
                    if probability == 0:
                        received = transmitted.copy()
                        mask = np.zeros(len(transmitted), dtype=bool)
                        initial = None
                    else:
                        rng = make_rng(path)
                        initial = rng.bit_generator.state
                        received, mask = transmit_bsc(
                            transmitted, probability, rng
                        )
                        total_primary_masks += 1
                    decoded, corrected = decode(received, name)
                    data = decoded[: len(payload)]
                    wrong = data != payload
                    block_wrong = decoded.reshape(-1, k) != padded
                    effective_wrong = block_wrong.copy()
                    if padding:
                        effective_wrong[-1, -padding:] = False
                    weights = np.sum(mask.reshape(-1, n), axis=1)
                    histogram = np.bincount(
                        weights.astype(int), minlength=n + 1
                    )
                    failed = int(np.count_nonzero(np.any(block_wrong, axis=1)))
                    outside = int(np.count_nonzero(weights > t_correct))
                    if failed != outside:
                        raise ArithmeticError(
                            "Correction-radius check failed."
                        )
                    recovered_labels = bits_to_labels(data, bits)
                    recovered = dequantise(recovered_labels, bits, a)
                    high = periodic_reconstruct(recovered, len(reference))
                    added = high - pre_high
                    mse = float(np.mean((high - reference) ** 2))
                    added_mse = float(np.mean(added**2))
                    cross = float(2 * np.mean(pre_error * added))
                    max_decomposition = max(
                        max_decomposition,
                        abs(mse - baseline - added_mse - cross),
                    )
                    if probability == 0:
                        zero_control_error = max(
                            zero_control_error,
                            float(np.max(np.abs(high - pre_high))),
                        )
                        if np.any(wrong):
                            raise ArithmeticError(
                                "A zero-error round trip failed."
                            )
                    key = (name, cid)
                    if key not in plane_counts:
                        plane_counts[key] = np.zeros(bits, dtype=np.int64)
                    plane_counts[key] += np.sum(
                        wrong.reshape(-1, bits), axis=0
                    ).astype(np.int64)
                    row = {
                        "code": name,
                        "condition_id": cid,
                        "probability": probability,
                        "gaussian_repeat_id": group,
                        "bit_repeat_id": bit_repeat,
                        "repeat_id": repeat,
                        "layout_id": layout,
                        "payload_bits": len(payload),
                        "transmitted_bits": len(transmitted),
                        "padding_bits": padding,
                        "channel_flips": int(np.count_nonzero(mask)),
                        "decoded_wrong_bits": int(np.count_nonzero(wrong)),
                        "block_count": len(padded),
                        "full_message_block_failures": failed,
                        "payload_block_failures": int(
                            np.count_nonzero(np.any(effective_wrong, axis=1))
                        ),
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
                        "channel_component_mse_48000": added_mse,
                        "upstream_channel_cross": cross,
                    }
                    trials.append(row)
                    for weight, count in enumerate(histogram):
                        weights_rows.append(
                            {
                                "code": name,
                                "condition_id": cid,
                                "probability": probability,
                                "gaussian_repeat_id": group,
                                "bit_repeat_id": bit_repeat,
                                "weight": weight,
                                "blocks": int(count),
                            }
                        )
                    channel_manifest.append(
                        {
                            "code": name,
                            "condition_id": cid,
                            "gaussian_repeat_id": group,
                            "bit_repeat_id": bit_repeat,
                            "spawn_key": list(path) if initial else None,
                            "initial_state": initial,
                            "probability": probability,
                            "uniform_call_shape": (
                                [len(transmitted)] if initial else []
                            ),
                            "transmitted_sha256": bits_hash(transmitted),
                            "received_sha256": bits_hash(received),
                            "decoded_real_payload_sha256": bits_hash(data),
                            "pairing_group": f"r32-g{group}-b{bit_repeat}",
                            "identity_no_rng": initial is None,
                        }
                    )
                    example = protocol["example"]
                    if (
                        probability == example["probability"]
                        and group == example["gaussian_repeat_id"]
                        and bit_repeat == example["bit_repeat_id"]
                    ):
                        example_high[name] = high.copy()
                        example_low[name] = recovered.copy()
                        example_high["prechannel"] = pre_high
                        example_low["prechannel"] = q
    write_csv(results / "trials.csv", trials)
    write_csv(results / "block_weights.csv", weights_rows)
    summaries, profile = [], []
    for spec in definitions:
        name, n, k, t_correct = (spec["code"], spec["n"], spec["k"], spec["t"])
        coeff = coefficients[coefficients["code"] == name]
        for condition in protocol["pilot_conditions"]:
            cid, probability = (
                condition["condition_id"],
                condition["probability"],
            )
            rows = [
                r
                for r in trials
                if r["code"] == name and r["condition_id"] == cid
            ]
            q_theory = (
                sum(
                    int(r["wrong_message_bits_sum"])
                    * probability ** int(r["weight"])
                    * (1 - probability) ** (n - int(r["weight"]))
                    for r in coeff
                )
                / k
            )
            block_theory = sum(
                comb(n, j) * probability**j * (1 - probability) ** (n - j)
                for j in range(t_correct + 1, n + 1)
            )
            total_data = sum(r["payload_bits"] for r in rows)
            total_tx = sum(r["transmitted_bits"] for r in rows)
            total_blocks = sum(r["block_count"] for r in rows)
            wrong_bits = sum(r["decoded_wrong_bits"] for r in rows)
            summary = {
                "code": name,
                "condition_id": cid,
                "probability": probability,
                "records": len(rows),
                "gaussian_banks": 5,
                "payload_bits": total_data,
                "transmitted_bits": total_tx,
                "wrong_bits": wrong_bits,
                "channel_flips": sum(r["channel_flips"] for r in rows),
                "channel_ber": sum(r["channel_flips"] for r in rows)
                / total_tx,
                "data_ber": wrong_bits / total_data,
                "theory_data_ber": q_theory,
                "full_message_bler": sum(
                    r["full_message_block_failures"] for r in rows
                )
                / total_blocks,
                "theory_full_message_bler": block_theory,
                "mean_mse_48000": float(
                    np.mean([r["mse_48000"] for r in rows])
                ),
                "mean_prechannel_mse_48000": float(
                    np.mean([r["prechannel_mse_48000"] for r in rows])
                ),
            }
            summaries.append(summary)
            for weight in range(n + 1):
                observed = sum(
                    r["blocks"]
                    for r in weights_rows
                    if r["code"] == name
                    and r["condition_id"] == cid
                    and r["weight"] == weight
                )
                profile.append(
                    {
                        "code": name,
                        "condition_id": cid,
                        "probability": probability,
                        "weight": weight,
                        "total_blocks": total_blocks,
                        "observed_blocks": observed,
                        "observed_fraction": observed / total_blocks,
                        "binomial_probability": comb(n, weight)
                        * probability**weight
                        * (1 - probability) ** (n - weight),
                    }
                )
    write_csv(results / "summary.csv", summaries)
    write_csv(results / "error_profile.csv", profile)
    planes = [
        {
            "code": name,
            "condition_id": cid,
            "probability": next(
                c["probability"]
                for c in protocol["pilot_conditions"]
                if c["condition_id"] == cid
            ),
            "bit_position_msb_first": pos,
            "wrong_bits": int(count),
        }
        for (name, cid), counts in plane_counts.items()
        for pos, count in enumerate(counts)
    ]
    write_csv(results / "bitplane_counts.csv", planes)
    chosen, scores, selection, sensitivity = score_candidates(
        trials, definitions, protocol
    )
    write_csv(results / "bank_scores.csv", scores)
    write_csv(results / "code_selection.csv", selection)
    write_csv(results / "budget_sensitivity.csv", sensitivity)
    paired = []
    for candidate in protocol["code_names"]:
        if candidate == chosen:
            continue
        differences = np.array(
            [
                next(
                    r["normalised_mse_score"]
                    for r in scores
                    if r["code"] == chosen and r["gaussian_repeat_id"] == g
                )
                - next(
                    r["normalised_mse_score"]
                    for r in scores
                    if r["code"] == candidate and r["gaussian_repeat_id"] == g
                )
                for g in protocol["gaussian_repeat_ids"]
            ]
        )
        half = (
            float(t.ppf(0.975, 4)) * np.std(differences, ddof=1) / np.sqrt(5)
        )
        paired.append(
            {
                "chosen": chosen,
                "comparison": candidate,
                "mean_score_difference": float(np.mean(differences)),
                "ci95_low": float(np.mean(differences) - half),
                "ci95_high": float(np.mean(differences) + half),
                "independent_input_banks": 5,
            }
        )
    write_csv(results / "paired_scores.csv", paired)
    ordered = ["uncoded", "hamming74", "hamming1511", "repetition51"]
    for rate, waves, clean in [
        (16000, example_low, x),
        (48000, example_high, reference),
    ]:
        np.savetxt(
            results / f"example_{rate}hz.csv",
            np.column_stack(
                (
                    np.arange(len(clean)),
                    np.arange(len(clean)) / rate,
                    clean,
                    waves["prechannel"],
                    *(waves[name] for name in ordered),
                )
            ),
            delimiter=",",
            comments="",
            header="index,time_s,reference,prechannel," + ",".join(ordered),
            fmt=["%d"] + ["%.17g"] * 7,
        )
    source_paths = [
        ROOT / "codes.py",
        ROOT / "code_checks.py",
        ROOT / "pilot_comparison.py",
        ROOT / "design.json",
        qfreeze_path,
        PACKAGE / "stage3/r31/quantisation.py",
        PACKAGE / "stage3/bitstream.py",
        PACKAGE / "degradation.py",
        PACKAGE / "stage1/r12/sampling_rate.py",
        PACKAGE / "stage2/r21/filtering_approach.py",
        PACKAGE / "reference_signal.npy",
        h_path,
        PACKAGE / "stage3/r31/results/seed_manifest.json",
    ]
    source_hashes = {
        p.relative_to(PACKAGE).as_posix(): file_hash(p) for p in source_paths
    }
    chosen_spec = next(r for r in definitions if r["code"] == chosen)
    chosen_score = next(r for r in selection if r["code"] == chosen)
    frozen = {
        "date": "2026-10-21",
        "status": "R3.2 code frozen from pilot design",
        "selected_code": chosen,
        "definition": chosen_spec,
        "selection": chosen_score,
        "payload_expansion_budget": protocol["max_payload_expansion"],
        "selection_probabilities": protocol["selection_probabilities"],
        "quantiser_bits": bits,
        "quantiser_half_range": a,
        "source_sha256": source_hashes,
        "p_range_and_mapping_finalised": False,
        "personal_p_drawn": False,
        "phase_1_used": False,
        "limits": (
            "Pilot code choice at personal Gaussian SNR; "
            "not an R3.4 success threshold or validation claim"
        ),
    }
    (results / "code_freeze.json").write_text(
        json.dumps(frozen, indent=2) + "\n"
    )
    (results / "seed_manifest.json").write_text(
        json.dumps(
            {
                "root_seed": registry["root_seed"],
                "pool_size": 4,
                "bit_generator": "PCG64",
                "numpy": np.__version__,
                "python": platform.python_version(),
                "scope": "R3.2 design phase 0 only",
                "input_banks": input_manifest,
                "channel_trials": channel_manifest,
                "source_sha256": source_hashes,
                "no_personal_parameter_leaf_0_1_draw": True,
            },
            indent=2,
        )
        + "\n"
    )
    if zero_control_error or max_decomposition > 1e-11:
        raise ArithmeticError(
            "Identity or complete-grid decomposition failed."
        )
    paths = [r["spawn_key"] for r in channel_manifest if r["spawn_key"]]
    assert len(paths) == len(set(map(tuple, paths)))
    assert all(p[:3] == [2, 3, 0] for p in paths)
    verification = {
        "status": "passed",
        "trial_rows": len(trials),
        "reused_gaussian_input_banks": len(input_manifest),
        "positive_probability_bsc_masks": total_primary_masks,
        "unique_bsc_paths": len(paths),
        "zero_control_reconstruction_difference": zero_control_error,
        "maximum_mse_decomposition_residual": max_decomposition,
        "all_full_block_failures_equal_radius_tail_counts": True,
        "selected_code": chosen,
        "all_upstream_frozen_hashes_checked": True,
        "p_parameter_fields_remain_null": all(
            registry[k] is None
            for k in [
                "bit_flip_probability_range",
                "bit_flip_mapping",
                "bit_flip_baseline_probability",
            ]
        ),
        "phase_1_used": False,
    }
    (PACKAGE / "metadata/r32_verification.json").write_text(
        json.dumps(verification, indent=2) + "\n"
    )
    print(
        f"R3.2: {len(trials)} pilot rows, {total_primary_masks} BSC masks; "
        f"selected {chosen}."
    )


if __name__ == "__main__":
    main()
