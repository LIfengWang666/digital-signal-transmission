"""Run the declared four-configuration phase-0 implementation pilot."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import scipy

from degradation import add_gaussian_noise, transmit_bsc
from stage2.r21.filtering_approach import make_rng, write_csv
from stage2.r23.validation import array_hash, file_hash
from stage4.r42.pipeline import (
    load_system,
    measure_path,
    paired_effects,
    prepare_bank,
    receive_stream,
)
from stage4.r42.planning import summarise_pilot
from stage4.r42.protocol import freeze_pilot, verify_sources

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def clean_controls(system: dict) -> list[dict]:
    """Check all four paths once without Gaussian noise or BSC errors.

    Args:
        system: Frozen transmitter and public receiver metadata.

    Returns:
        Four deterministic metrics, excluded from random counts and CIs.
    """
    banks = prepare_bank(
        system, system["x"].copy(), np.zeros_like(system["x"])
    )
    rows = []
    for code in ["uncoded", "hamming74"]:
        for f in [0, 1]:
            bank = banks[f]
            sent = bank["sent"][code]
            received = sent.copy()
            recovered = receive_stream(received, code, 448000, 14, 18, 96000)
            row, _ = measure_path(
                system,
                bank,
                code,
                received,
                np.zeros(len(sent), dtype=bool),
                recovered,
            )
            rows.append(
                {
                    "configuration": f"{f}{int(code == 'hamming74')}",
                    "gaussian_enabled": 0,
                    "probability": 0,
                    **row,
                }
            )
    for f in [0, 1]:
        a = next(r for r in rows if r["configuration"] == f"{f}0")
        b = next(r for r in rows if r["configuration"] == f"{f}1")
        assert a["decoded_payload_sha256"] == b["decoded_payload_sha256"]
        assert a["mse"] == b["mse"]
        assert a["decoded_wrong_bits"] == b["decoded_wrong_bits"] == 0
    return rows


def main() -> None:
    """Freeze, generate and diagnose new phase-0 complete record bundles.

    Returns:
        None. Save paired measurements, unique leaves and precision planning.
        Never generate a phase-1 record or change upstream scientific choices.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-only", action="store_true")
    args = parser.parse_args()
    frozen = freeze_pilot()
    if args.freeze_only:
        print("R4.2 phase0 protocol frozen; no random draw performed.")
        return
    design = frozen["choices"]
    system = load_system()
    folder = HERE / "results"
    write_csv(folder / "configurations.csv", design["configurations"])
    write_csv(
        folder / "conditions.csv",
        [
            {**c, "slice_membership": "|".join(c["slice_membership"])}
            for c in design["conditions"]
        ],
    )
    write_csv(folder / "clean_controls.csv", clean_controls(system))
    trials, decompositions, effects = [], [], []
    gaussian, quantisers, channels, pair_checks = [], [], [], []
    started = time.perf_counter()
    for repeat in range(20):
        for snr in design["snr_conditions"]:
            scid, target = snr["snr_condition_id"], snr["snr_db"]
            path = (1, 4, 0, scid, repeat)
            rng = make_rng(path)
            state = rng.bit_generator.state
            noisy, noise, variance = add_gaussian_noise(
                system["x"], target, rng
            )
            banks = prepare_bank(system, noisy, noise)
            gaussian.append(
                {
                    "record_id": repeat,
                    "snr_condition_id": scid,
                    "snr_db": target,
                    "spawn_key": list(path),
                    "initial_state": state,
                    "normal_call_shape": [32000],
                    "noise_sha256": array_hash(noise),
                    "model_variance": variance,
                    "realised_noise_power": float(np.mean(noise**2)),
                }
            )
            for f, bank in banks.items():
                quantisers.append(
                    {
                        "record_id": repeat,
                        "snr_condition_id": scid,
                        "filter_enabled": f,
                        "overload_samples": bank["overload_samples"],
                        "quantiser_input_sha256": array_hash(bank["z"]),
                        "prechannel_nmse": float(
                            np.mean(
                                (bank["before"] - system["reference"]) ** 2
                            )
                            / np.mean(system["reference"] ** 2)
                        ),
                    }
                )
            for condition in design["conditions"]:
                if condition["snr_condition_id"] != scid:
                    continue
                cid, p = (
                    condition["joint_condition_id"],
                    condition["probability"],
                )
                case_rows = {}
                for layout, code in [(0, "uncoded"), (1, "hamming74")]:
                    path = (2, 4, 0, cid, repeat, layout)
                    pair = []
                    for f in [0, 1]:
                        bank = banks[f]
                        sent = bank["sent"][code]
                        if p == 0:
                            received = sent.copy()
                            mask, state = np.zeros(len(sent), dtype=bool), None
                        else:
                            rng = make_rng(path)
                            state = rng.bit_generator.state
                            received, mask = transmit_bsc(sent, p, rng)
                        recovered = receive_stream(
                            received, code, 448000, 14, 18, 96000
                        )
                        measured, component = measure_path(
                            system, bank, code, received, mask, recovered
                        )
                        meta = {
                            "phase_id": 0,
                            "record_id": repeat,
                            "joint_condition_id": cid,
                            "snr_condition_id": scid,
                            "snr_db": target,
                            "probability": p,
                            "configuration": f"{f}{layout}",
                            "code": code,
                            "filter_enabled": f,
                        }
                        row = {**meta, **measured}
                        trials.append(row)
                        decompositions.append({**meta, **component})
                        case_rows[f"{f}{layout}"] = row
                        pair.append(row)
                    for key in [
                        "mask_sha256",
                        "wrong_pattern_sha256",
                        "channel_flips",
                        "decoded_wrong_bits",
                        "wrong_samples",
                        "wrong_msb_bits",
                    ]:
                        assert pair[0][key] == pair[1][key], key
                    channels.append(
                        {
                            "record_id": repeat,
                            "joint_condition_id": cid,
                            "layout_id": layout,
                            "probability": p,
                            "spawn_key": list(path) if p else None,
                            "initial_state": state,
                            "uniform_call_shape": [len(mask)] if p else [],
                            "mask_sha256": pair[0]["mask_sha256"],
                            "channel_flips": pair[0]["channel_flips"],
                            "shared_configurations": [
                                f"0{layout}",
                                f"1{layout}",
                            ],
                            "replay_calls": 2 if p else 0,
                            "unique_random_mask_exposures": 1 if p else 0,
                        }
                    )
                    pair_checks.append(
                        {
                            "record_id": repeat,
                            "joint_condition_id": cid,
                            "layout_id": layout,
                            "same_mask": 1,
                            "same_decoded_error_pattern": 1,
                            "same_bit_and_sample_error_counts": 1,
                        }
                    )
                if p == 0:
                    for f in [0, 1]:
                        a, b = case_rows[f"{f}0"], case_rows[f"{f}1"]
                        assert (
                            a["decoded_payload_sha256"]
                            == b["decoded_payload_sha256"]
                        )
                        assert a["mse"] == b["mse"]
                effects.append(
                    {
                        "record_id": repeat,
                        "joint_condition_id": cid,
                        **paired_effects(
                            {
                                key: row["nmse"]
                                for key, row in case_rows.items()
                            }
                        ),
                    }
                )
        if (repeat + 1) % 5 == 0:
            print(f"R4.2 pilot: {repeat+1}/20 complete bundles", flush=True)
    elapsed = time.perf_counter() - started
    write_csv(folder / "pilot_trials.csv", trials)
    write_csv(folder / "decomposition.csv", decompositions)
    write_csv(folder / "paired_effects.csv", effects)
    write_csv(folder / "quantiser_banks.csv", quantisers)
    write_csv(folder / "pair_checks.csv", pair_checks)
    plan = summarise_pilot(trials, design, folder)
    paths = [
        tuple(r["spawn_key"])
        for r in gaussian + channels
        if r["spawn_key"] is not None
    ]
    assert len(paths) == len(set(paths)) == 1000
    assert len(trials) == 1760 and len(gaussian) == 160
    assert len(quantisers) == 320 and len(channels) == 880
    assert max(r["mse_residual"] for r in decompositions) < 1e-9
    assert max(r["waveform_identity_residual"] for r in decompositions) < 1e-10
    assert max(r["nyquist_energy_residual"] for r in decompositions) < 1e-9
    verify_sources(design)
    for name, expected in frozen["source_sha256"].items():
        assert file_hash(ROOT / name) == expected
    manifest = {
        "date": design["date"],
        "scope": "R4.2 phase0 implementation/precision pilot, not validation",
        "root_seed": 2470964,
        "pool_size": 4,
        "bit_generator": "PCG64",
        "software": {
            "Python": platform.python_version(),
            "NumPy": np.__version__,
            "SciPy": scipy.__version__,
        },
        "pilot_freeze_sha256": file_hash(folder / "pilot_freeze.json"),
        "gaussian_inputs": gaussian,
        "bsc_mask_groups": channels,
        "unique_gaussian_inputs": 160,
        "unique_positive_p_bsc_masks": 840,
        "identity_no_rng_groups": 40,
        "unique_quantiser_inputs": 320,
        "random_path_rows": 1760,
        "deterministic_clean_rows": 4,
        "all_pairs_identical_error_patterns": True,
        "phase1_records_generated": 0,
    }
    (folder / "seed_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    (folder / "execution_timing.json").write_text(
        json.dumps(
            {"pilot_seconds": elapsed, "seconds_per_bundle": elapsed / 20},
            indent=2,
        )
        + "\n"
    )
    registry_path = ROOT / "metadata/randomness_design.json"
    registry = json.loads(registry_path.read_text())
    registry["status"] = "R4.2 phase0 pilot complete; R4 phase1 unused"
    registry["r42_status"] = (
        "20 bundles,160 Gaussian,840 unique BSC masks,1760 paths; "
        "no validation records"
    )
    registry["course_scope"] = (
        "R2/R3 and R4.1/R4.2 complete; R4.3 phase1 pending"
    )
    registry_path.write_text(json.dumps(registry, indent=2) + "\n")
    print(
        "R4.2 candidate validation records:",
        plan["candidate_validation_records_per_condition"],
    )


if __name__ == "__main__":
    main()
