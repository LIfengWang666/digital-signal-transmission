"""Freeze and run R3.4 with new independent complete-record noise draws."""

from __future__ import annotations

import argparse
import json
import math
import platform
from pathlib import Path

import numpy as np
from scipy.stats import t

from degradation import add_gaussian_noise, transmit_bsc
from stage1.r12.sampling_rate import periodic_reconstruct
from stage2.r21.filtering_approach import centred_filter, make_rng, write_csv
from stage2.r23.validation import array_hash, file_hash
from stage3.bitstream import bits_to_labels, labels_to_bits
from stage3.r31.quantisation import dequantise, quantise
from stage3.r32.codes import decode, encode
from stage3.r32.pilot_comparison import bits_hash
from stage3.r33.error_range import read_rows
from stage3.r34.assessment import summarise

ROOT = Path(__file__).resolve().parent
PACKAGE = ROOT.parents[1]


def freeze_protocol() -> dict:
    """Derive the threshold and budget using phase-0 evidence only.

    Returns:
        Frozen design written before any R3 phase-1 generator is used.
    """
    protocol = json.loads((ROOT / "design.json").read_text())
    registry_path = PACKAGE / "metadata/randomness_design.json"
    registry = json.loads(registry_path.read_text())
    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    for key in ["quantiser_freeze", "code_freeze", "range_freeze"]:
        frozen = json.loads((PACKAGE / protocol[key]).read_text())
        for name, digest in frozen["source_sha256"].items():
            if file_hash(PACKAGE / name) != digest:
                raise ValueError(
                    "An upstream frozen scientific input changed."
                )
    summary = read_rows(PACKAGE / "stage3/r33/results/summary.csv")
    base = next(
        r
        for r in summary
        if r["code"] == "hamming74" and float(r["probability"]) == 0
    )
    power = float(np.mean(np.load(PACKAGE / "reference_signal.npy") ** 2))
    baseline = float(base["mean_mse_48000"]) / power
    threshold = protocol["primary_total_mse_budget_factor"] * baseline
    old = [
        dict(c)
        for c in registry["bsc_condition_registry"]
        if c["study_id"] == 3
        and c["phase_id"] == 0
        and c["condition_id"] in protocol["base_condition_ids"]
    ]
    assert len(old) == 12
    conditions = [
        {
            "condition_id": c["condition_id"],
            "probability": c["probability"],
            "role": (
                "identity"
                if c["condition_id"] == 0
                else (
                    "personal"
                    if c["condition_id"] == 11
                    else "stress" if c["probability"] > 0.1 else "main_grid"
                )
            ),
        }
        for c in old
    ] + [dict(c) for c in protocol["additional_conditions"]]
    conditions.sort(key=lambda c: c["probability"])
    assert (
        len({c["condition_id"] for c in conditions}) == len(conditions) == 15
    )
    family = 2 * len(conditions)
    pilot = read_rows(PACKAGE / "stage3/r33/results/trials.csv")
    planning = []
    for case in protocol["sample_size_planning"]["target_cases"]:
        within, bank_means = [], []
        for group in range(5):
            values = [
                float(r["nmse_48000"])
                for r in pilot
                if r["code"] == case["code"]
                and float(r["probability"]) == case["probability"]
                and int(r["gaussian_repeat_id"]) == group
            ]
            assert len(values) == 8
            within.append(float(np.var(values, ddof=1)))
            bank_means.append(float(np.mean(values)))
        conditional = float(np.mean(within))
        between = max(float(np.var(bank_means, ddof=1)) - conditional / 8, 0)
        planning.append(
            {
                "code": case["code"],
                "probability": case["probability"],
                "mean_within_bank_variance": conditional,
                "estimated_between_bank_variance": between,
                "planned_record_sd_nmse": math.sqrt(conditional + between),
            }
        )
    count = 200
    goal = (
        protocol["sample_size_planning"][
            "target_half_width_fraction_of_primary_threshold"
        ]
        * threshold
    )
    while True:
        critical = float(t.ppf(1 - 0.05 / (2 * family), count - 1))
        if all(
            critical * r["planned_record_sd_nmse"] / math.sqrt(count) <= goal
            for r in planning
        ):
            break
        count += 50
    assert count == protocol["records_per_condition"] == 300
    for row in planning:
        row["chosen_records"] = count
        row["planned_adjusted_half_width"] = (
            critical * row["planned_record_sd_nmse"] / math.sqrt(count)
        )
        row["target_half_width"] = goal
    write_csv(results / "sample_size_planning.csv", planning)
    sources = [
        "stage3/r34/design.json",
        "stage3/r34/coding_gain.py",
        "stage3/r34/assessment.py",
        "stage3/r33/results/range_freeze.json",
        "stage3/r33/results/summary.csv",
        "stage3/r33/results/trials.csv",
        "stage3/r33/results/personal_parameter.json",
        "stage3/r31/results/quantiser_freeze.json",
        "stage3/r32/results/code_freeze.json",
        "stage3/r31/quantisation.py",
        "stage3/r32/codes.py",
        "stage3/r33/error_range.py",
        "stage3/bitstream.py",
        "degradation.py",
        "stage1/r12/sampling_rate.py",
        "stage2/r21/filtering_approach.py",
        "reference_signal.npy",
        "reference_signal.wav",
        "stage2/r22/results/selected_coefficients.npy",
        "stage1/r12/results/selected_samples.npy",
    ]
    design = {
        "date": protocol["date"],
        "status": "Frozen before validation",
        "reference_power": power,
        "design_baseline_nmse": baseline,
        "design_baseline_mse": baseline * power,
        "primary_nmse_threshold": threshold,
        "primary_mse_threshold": threshold * power,
        "primary_budget_factor": protocol["primary_total_mse_budget_factor"],
        "budget_db": 10 * math.log10(2),
        "threshold_sensitivity_factors": protocol[
            "threshold_sensitivity_factors"
        ],
        "records_per_condition": count,
        "conditions": conditions,
        "confidence_family_size": family,
        "familywise_alpha": 0.05,
        "adjusted_t_critical": critical,
        "interval_scope": protocol["interval_scope"],
        "planning_uses_phase0_only": True,
        "validation_used_to_choose_threshold_or_budget": False,
        "gaussian_snr_db": registry["snr_baseline_db"],
        "personal_probability": registry["bit_flip_baseline_probability"],
        "filter_taps": 385,
        "bits": 14,
        "half_range": 18,
        "code": "hamming74",
        "sample_rate_hz": 16000,
        "source_sha256": {s: file_hash(PACKAGE / s) for s in sources},
        "r4_streams_used": False,
    }
    (results / "validation_freeze.json").write_text(
        json.dumps(design, indent=2) + "\n"
    )
    gaussian = {
        "study_id": 3,
        "phase_id": 1,
        "condition_id": 5,
        "target_snr_db": registry["snr_baseline_db"],
        "role": "R3.4 independent Gaussian bank shared across p and code",
    }
    entries = [(registry["condition_registry"], gaussian)]
    for c in conditions:
        entries.append(
            (
                registry["bsc_condition_registry"],
                {
                    "study_id": 3,
                    "phase_id": 1,
                    "condition_id": c["condition_id"],
                    "probability": c["probability"],
                    "role": c["role"],
                    "gaussian_condition_id": 5,
                    "target_snr_db": registry["snr_baseline_db"],
                    "scope": "R3.4 predeclared independent coding validation",
                },
            )
        )
    for target, entry in entries:
        previous = [
            c
            for c in target
            if (c["study_id"], c["phase_id"], c["condition_id"])
            == (entry["study_id"], entry["phase_id"], entry["condition_id"])
        ]
        if previous:
            assert len(previous) == 1 and previous[0] == entry
        else:
            target.append(entry)
    registry["r34_protocol"] = "../stage3/r34/design.json"
    registry["r34_validation_freeze"] = (
        "../stage3/r34/results/validation_freeze.json"
    )
    registry["r34_seed_manifest"] = "../stage3/r34/results/seed_manifest.json"
    registry_path.write_text(json.dumps(registry, indent=2) + "\n")
    return design


def main() -> None:
    """Run new independent records using the unchanged fixed chain.

    Returns:
        None. Write all measured evidence and the frozen-rule assessment.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-only", action="store_true")
    args = parser.parse_args()
    design = freeze_protocol()
    if args.freeze_only:
        print(
            "R3.4: threshold, 300 records and 30-mean family frozen; no draws."
        )
        return
    protocol = json.loads((ROOT / "design.json").read_text())
    results = ROOT / "results"
    reference = np.load(PACKAGE / "reference_signal.npy")
    x = np.load(PACKAGE / "stage1/r12/results/selected_samples.npy")
    h = np.load(PACKAGE / "stage2/r22/results/selected_coefficients.npy")
    qfreeze = json.loads((PACKAGE / protocol["quantiser_freeze"]).read_text())
    bits, a = qfreeze["bits"], qfreeze["half_range"]
    trials, profiles, input_banks, channel_trials = [], [], [], []
    planes, examples = {}, {}
    max_decomposition, max_identity_error = 0.0, 0.0
    overloads = 0
    for record in range(design["records_per_condition"]):
        gaussian_path = (1, 3, 1, 5, record)
        rng = make_rng(gaussian_path)
        initial = rng.bit_generator.state
        noisy, noise, _ = add_gaussian_noise(x, design["gaussian_snr_db"], rng)
        z = centred_filter(noisy, h)
        overload_count = int(np.count_nonzero(np.abs(z) > a))
        overloads += overload_count
        labels, q, _ = quantise(z, bits, a, qfreeze["normalised_epsilon"])
        payload = labels_to_bits(labels, bits)
        before = periodic_reconstruct(q, len(reference))
        error_before = before - reference
        baseline = float(np.mean(error_before**2))
        input_banks.append(
            {
                "record_id": record,
                "spawn_key": list(gaussian_path),
                "initial_state": initial,
                "normal_call_shape": list(x.shape),
                "noise_sha256": array_hash(noise),
                "quantiser_input_sha256": array_hash(z),
                "payload_sha256": bits_hash(payload),
                "prechannel_mse_48000": baseline,
                "overload_samples": overload_count,
            }
        )
        for code in protocol["code_names"]:
            n, k = (1, 1) if code == "uncoded" else (7, 4)
            radius = 0 if code == "uncoded" else 1
            sent, padding = encode(payload, code)
            assert padding == 0
            for condition in design["conditions"]:
                cid, p = condition["condition_id"], condition["probability"]
                path = (2, 3, 1, cid, record, protocol["layout_ids"][code])
                if p == 0:
                    received = sent.copy()
                    mask = np.zeros(len(sent), dtype=bool)
                    state = None
                else:
                    rng = make_rng(path)
                    state = rng.bit_generator.state
                    received, mask = transmit_bsc(sent, p, rng)
                data, _ = decode(received, code)
                assert len(data) == len(payload)
                wrong = data != payload
                failed = int(
                    np.count_nonzero(np.any(wrong.reshape(-1, k), axis=1))
                )
                weights = np.sum(mask.reshape(-1, n), axis=1)
                assert failed == int(np.count_nonzero(weights > radius))
                recovered_labels = bits_to_labels(data, bits)
                high = periodic_reconstruct(
                    dequantise(recovered_labels, bits, a), len(reference)
                )
                added = high - before
                mse = float(np.mean((high - reference) ** 2))
                component = float(np.mean(added**2))
                cross = float(2 * np.mean(error_before * added))
                max_decomposition = max(
                    max_decomposition,
                    abs(mse - baseline - component - cross),
                )
                if p == 0:
                    assert not np.any(wrong)
                    max_identity_error = max(
                        max_identity_error,
                        float(np.max(np.abs(high - before))),
                    )
                trials.append(
                    {
                        "code": code,
                        "condition_id": cid,
                        "probability": p,
                        "record_id": record,
                        "pairing_group": f"r34-r{record}",
                        "payload_bits": len(payload),
                        "transmitted_bits": len(sent),
                        "padding_bits": padding,
                        "block_count": len(payload) // k,
                        "channel_flips": int(np.count_nonzero(mask)),
                        "decoded_wrong_bits": int(np.count_nonzero(wrong)),
                        "block_failures": failed,
                        "wrong_msb_bits": int(
                            np.count_nonzero(wrong.reshape(-1, bits)[:, 0])
                        ),
                        "wrong_samples": int(
                            np.count_nonzero(
                                np.any(wrong.reshape(-1, bits), axis=1)
                            )
                        ),
                        "mse_48000": mse,
                        "nmse_48000": mse / design["reference_power"],
                        "prechannel_mse_48000": baseline,
                        "prechannel_nmse_48000": baseline
                        / design["reference_power"],
                        "channel_component_mse_48000": component,
                        "upstream_channel_cross": cross,
                        "overload_samples": overload_count,
                    }
                )
                histogram = np.bincount(weights.astype(int), minlength=n + 1)
                for weight, count in enumerate(histogram):
                    profiles.append(
                        {
                            "code": code,
                            "condition_id": cid,
                            "record_id": record,
                            "probability": p,
                            "weight": weight,
                            "blocks": int(count),
                        }
                    )
                key = (code, cid)
                if key not in planes:
                    planes[key] = np.zeros(bits, dtype=np.int64)
                planes[key] += np.sum(wrong.reshape(-1, bits), axis=0).astype(
                    np.int64
                )
                channel_trials.append(
                    {
                        "code": code,
                        "condition_id": cid,
                        "probability": p,
                        "record_id": record,
                        "spawn_key": list(path) if state else None,
                        "initial_state": state,
                        "uniform_call_shape": [len(sent)] if state else [],
                        "transmitted_sha256": bits_hash(sent),
                        "received_sha256": bits_hash(received),
                        "decoded_real_payload_sha256": bits_hash(data),
                        "identity_no_rng": state is None,
                    }
                )
                for example in protocol["examples"]:
                    if cid == example["condition_id"] and (
                        record == example["record_id"]
                    ):
                        name = example["name"]
                        if name not in examples:
                            examples[name] = {
                                "reference": reference,
                                "prechannel": before,
                            }
                        examples[name][code] = high.copy()
        if (record + 1) % 50 == 0:
            print(
                f"R3.4: {record + 1}/300 new input records measured.",
                flush=True,
            )
    write_csv(results / "trials.csv", trials)
    write_csv(results / "block_weights.csv", profiles)
    plane_rows = []
    for (code, cid), counts in planes.items():
        for position, count in enumerate(counts):
            plane_rows.append(
                {
                    "code": code,
                    "condition_id": cid,
                    "msb_first_position": position,
                    "wrong_bits": int(count),
                    "bitplane_exposures": len(x)
                    * design["records_per_condition"],
                }
            )
    write_csv(results / "bitplane_counts.csv", plane_rows)
    for name, signals in examples.items():
        columns = [
            np.arange(len(reference)),
            np.arange(len(reference)) / 48000,
            signals["reference"],
            signals["prechannel"],
            signals["uncoded"],
            signals["hamming74"],
        ]
        np.savetxt(
            results / f"example_{name}_48000hz.csv",
            np.column_stack(columns),
            delimiter=",",
            header="index,time_s,reference,prechannel,uncoded,hamming74",
            comments="",
        )
    outcome = summarise(trials, design, results)
    manifest = {
        "root_seed": 2470964,
        "pool_size": 4,
        "bit_generator": "PCG64",
        "numpy": np.__version__,
        "python": platform.python_version(),
        "study_id": 3,
        "phase_id": 1,
        "scope": "R3.4 independent validation only; no phase0 observations",
        "input_banks": input_banks,
        "channel_trials": channel_trials,
        "source_sha256": design["source_sha256"],
        "prevalidation_freeze_sha256": file_hash(
            results / "validation_freeze.json"
        ),
        "r4_streams_used": False,
    }
    (results / "seed_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    paths = [
        tuple(c["spawn_key"])
        for c in channel_trials
        if c["spawn_key"] is not None
    ]
    assert len(paths) == len(set(paths)) == 8400
    assert len(trials) == 9000 and len(input_banks) == 300
    assert max_identity_error == 0 and max_decomposition < 1e-10
    for name, digest in design["source_sha256"].items():
        assert file_hash(PACKAGE / name) == digest
    verification = {
        "date": protocol["date"],
        "status": "passed",
        "new_gaussian_input_records": len(input_banks),
        "new_bsc_masks": len(paths),
        "trial_rows": len(trials),
        "conditions": len(design["conditions"]),
        "identity_reconstruction_difference": max_identity_error,
        "maximum_mse_decomposition_residual": max_decomposition,
        "observed_overload_samples_across_300_inputs": overloads,
        "all_inputs_phase1_study3": True,
        "design_rows_imported": 0,
        "frozen_sources_match_before_and_after": True,
        "threshold_or_sample_size_retuned": False,
        "r4_streams_used": False,
    }
    (PACKAGE / "metadata/r34_verification.json").write_text(
        json.dumps(verification, indent=2) + "\n"
    )
    registry_path = PACKAGE / "metadata/randomness_design.json"
    registry = json.loads(registry_path.read_text())
    registry["status"] = "R3.4 independent validation complete; R4 pending"
    registry["r34_status"] = (
        "300 new Gaussian inputs, 8400 new BSC masks, "
        "15 conditions; frozen decision, no R4 draws"
    )
    registry["course_scope"] = "R2 and R3 complete; R4 study-4 unused"
    registry_path.write_text(json.dumps(registry, indent=2) + "\n")
    print(
        "R3.4: independent coding gain at tested p = "
        + repr(outcome["gain_at_tested_probabilities"])
    )


if __name__ == "__main__":
    main()
