"""Run fixed held-out bundles through the unchanged fixed pipeline."""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import time
from concurrent.futures import ProcessPoolExecutor
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
from stage4.r42.pilot import clean_controls
from stage4.r43.protocol import (
    CHOICES,
    freeze_measurement,
    register_validation,
)

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
TABLES = [
    "trials",
    "decomposition",
    "effects",
    "quantiser_banks",
    "pair_checks",
]
_SYSTEM: dict | None = None


def run_chunk(task: tuple[int, int, int, str, str]) -> dict:
    """Evaluate complete bundles, retaining the exact leaf/call conventions.

    Args:
        task: Start, exclusive stop, phase, output directory, freeze digest.

    Returns:
        Chunk completion metadata with hashes. Process scheduling changes
        neither numerical operations nor the random observations.
    """
    global _SYSTEM
    start, stop, phase, directory, freeze_hash = task
    folder = Path(directory)
    done = folder / "complete.json"
    if done.exists():
        prior = json.loads(done.read_text())
        if (
            prior["freeze_sha256"] == freeze_hash
            and prior["start"] == start
            and prior["stop"] == stop
            and prior["phase"] == phase
            and all(
                file_hash(folder / k) == v for k, v in prior["sha256"].items()
            )
        ):
            return prior
        raise ValueError("Stale or corrupt R4.3 chunk; do not mix protocols.")
    folder.mkdir(parents=True, exist_ok=True)
    if _SYSTEM is None:
        _SYSTEM = load_system()
    system = _SYSTEM
    choices = json.loads(CHOICES.read_text())
    tables: dict[str, list[dict]] = {name: [] for name in TABLES}
    gaussian, channels = [], []
    example_names = {
        11: "personal",
        5: "personal_snr_p003",
        21: "snr40_personal_p",
    }
    for repeat in range(start, stop):
        for snr in choices["snr_conditions"]:
            scid, target = snr["snr_condition_id"], snr["snr_db"]
            path = (1, 4, phase, scid, repeat)
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
                tables["quantiser_banks"].append(
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
            for condition in choices["conditions"]:
                if condition["snr_condition_id"] != scid:
                    continue
                cid, p = (
                    condition["joint_condition_id"],
                    condition["probability"],
                )
                case_rows, example = {}, {}
                for layout, code in [(0, "uncoded"), (1, "hamming74")]:
                    path = (2, 4, phase, cid, repeat, layout)
                    pair = []
                    for f in [0, 1]:
                        bank = banks[f]
                        sent = bank["sent"][code]
                        if p == 0:
                            received, mask, state = (
                                sent.copy(),
                                np.zeros(len(sent), bool),
                                None,
                            )
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
                        if (
                            component["mse_residual"] > 1e-9
                            or component["waveform_identity_residual"] > 1e-10
                            or component["nyquist_energy_residual"] > 1e-9
                        ):
                            raise ValueError("Full-grid decomposition failed.")
                        meta = {
                            "phase_id": phase,
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
                        tables["trials"].append(row)
                        tables["decomposition"].append({**meta, **component})
                        case_rows[f"{f}{layout}"] = row
                        pair.append(row)
                        if repeat == 0 and cid in example_names:
                            example[f"reconstruction_{f}{layout}"] = recovered[
                                "reconstruction"
                            ][9600:10560]
                    for key in [
                        "mask_sha256",
                        "wrong_pattern_sha256",
                        "channel_flips",
                        "decoded_wrong_bits",
                        "wrong_samples",
                        "wrong_msb_bits",
                    ]:
                        if pair[0][key] != pair[1][key]:
                            raise ValueError(
                                f"Shared-mask pair mismatch: {key}"
                            )
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
                    tables["pair_checks"].append(
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
                tables["effects"].append(
                    {
                        "record_id": repeat,
                        "joint_condition_id": cid,
                        **paired_effects(
                            {k: v["nmse"] for k, v in case_rows.items()}
                        ),
                    }
                )
                if example:
                    write_csv(
                        folder / f"example_{example_names[cid]}.csv",
                        [
                            {
                                "sample_index": i,
                                "time_s": i / 48000,
                                "reference": float(system["reference"][i]),
                                **{
                                    k: float(v[i - 9600])
                                    for k, v in example.items()
                                },
                            }
                            for i in range(9600, 10560)
                        ],
                    )
    for name, rows in tables.items():
        write_csv(folder / f"{name}.csv", rows)
    (folder / "leaves.json").write_text(
        json.dumps({"gaussian": gaussian, "bsc": channels}, indent=2) + "\n"
    )
    completed = {
        "start": start,
        "stop": stop,
        "phase": phase,
        "freeze_sha256": freeze_hash,
        "sha256": {
            p.name: file_hash(p)
            for p in sorted(folder.iterdir())
            if p.name != "complete.json"
        },
    }
    done.write_text(json.dumps(completed, indent=2) + "\n")
    return completed


def merge_chunks(folders: list[Path], frozen: dict) -> dict:
    """Merge fixed-order chunk measurements and unique random-leaf records.

    Args:
        folders: Completed chunk directories in increasing record order.
        frozen: Saved phase-1 measurement protocol.

    Returns:
        Seed manifest. Merge order is independent of process completion.
    """
    result = HERE / "results"
    counts = {}
    for name in TABLES:
        count = 0
        with (result / f"{name}.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = None
            for folder in folders:
                with (folder / f"{name}.csv").open(
                    newline="", encoding="utf-8"
                ) as source:
                    reader = csv.DictReader(source)
                    if writer is None:
                        writer = csv.DictWriter(
                            stream, fieldnames=reader.fieldnames
                        )
                        writer.writeheader()
                    for row in reader:
                        writer.writerow(row)
                        count += 1
        counts[name] = count
    assert counts == {
        "trials": 123200,
        "decomposition": 123200,
        "effects": 30800,
        "quantiser_banks": 22400,
        "pair_checks": 61600,
    }
    gaussian, channels = [], []
    for folder in folders:
        leaves = json.loads((folder / "leaves.json").read_text())
        gaussian.extend(leaves["gaussian"])
        channels.extend(leaves["bsc"])
    paths = [
        tuple(row["spawn_key"])
        for row in gaussian + channels
        if row["spawn_key"] is not None
    ]
    assert len(paths) == len(set(paths)) == 70000
    assert all(path[1:3] == (4, 1) for path in paths)
    manifest = {
        "date": "2026-10-21",
        "scope": "R4.3 held-out study4 phase1 validation",
        "root_seed": 2470964,
        "pool_size": 4,
        "bit_generator": "PCG64",
        "software": {
            "Python": platform.python_version(),
            "NumPy": np.__version__,
            "SciPy": scipy.__version__,
        },
        "validation_freeze_sha256": file_hash(
            result / "validation_freeze.json"
        ),
        "gaussian_inputs": gaussian,
        "bsc_mask_groups": channels,
        "unique_gaussian_inputs": 11200,
        "unique_positive_p_bsc_masks": 58800,
        "identity_no_rng_groups": 2800,
        "unique_quantiser_inputs": 22400,
        "random_path_rows": 123200,
        "complete_bundles": 1400,
        "deterministic_clean_rows": 4,
        "all_pairs_identical_error_patterns": True,
        "phase0_observations_imported": False,
        "fixed_record_count_observed": True,
    }
    (result / "seed_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    for path in folders[0].glob("example_*.csv"):
        (result / path.name).write_bytes(path.read_bytes())
    choices = frozen["choices"]
    write_csv(
        result / "conditions.csv",
        [
            {**c, "slice_membership": "|".join(c["slice_membership"])}
            for c in choices["conditions"]
        ],
    )
    write_csv(result / "configurations.csv", choices["configurations"])
    write_csv(result / "clean_controls.csv", clean_controls(load_system()))
    return manifest


def self_check() -> None:
    """Replay an existing pilot bundle before freezing or new random draws.

    Returns:
        None. Require identical hashes, counts and floating-point results
        against the previously verified R4.2 record zero.
    """
    folder = ROOT.parent / "planning/r43_implementation_20261021/pilot_replay"
    run_chunk((0, 1, 0, str(folder), "existing-pilot-only"))
    with (ROOT / "stage4/r42/results/pilot_trials.csv").open(
        newline=""
    ) as stream:
        old = {
            (r["joint_condition_id"], r["configuration"]): r
            for r in csv.DictReader(stream)
            if r["record_id"] == "0"
        }
    with (folder / "trials.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 88
    for row in rows:
        expected = old[(row["joint_condition_id"], row["configuration"])]
        for key in row:
            if row[key] == expected[key]:
                continue
            assert math.isclose(
                float(row[key]),
                float(expected[key]),
                rel_tol=1e-14,
                abs_tol=1e-14,
            ), key
    print(
        "R4.3 pre-freeze pilot replay: all88 path rows match R4.2.", flush=True
    )


def main() -> None:
    """Freeze, register, evaluate and merge exactly1400 held-out bundles.

    Returns:
        None. Save deterministic measurements and separate execution timing.
        Completed chunks can resume only with matching source-freeze hashes.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return
    frozen = freeze_measurement()
    if args.freeze_only:
        print("R4.3 code and choices frozen; no phase1 draw.")
        return
    register_validation(frozen["choices"])
    work = HERE / ".work"
    digest = file_hash(HERE / "results/validation_freeze.json")
    tasks = [
        (i, min(i + 20, 1400), 1, str(work / f"records_{i:04d}"), digest)
        for i in range(0, 1400, 20)
    ]
    started = time.perf_counter()
    if args.workers < 1:
        raise ValueError("At least one worker is required.")
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for index, _ in enumerate(pool.map(run_chunk, tasks), 1):
            print(
                f"R4.3 held-out: {index * 20}/1400 complete bundles",
                flush=True,
            )
    manifest = merge_chunks([Path(task[3]) for task in tasks], frozen)
    freeze_measurement()
    (HERE / "results/execution_timing.json").write_text(
        json.dumps(
            {
                "seconds": time.perf_counter() - started,
                "workers": args.workers,
                "records": 1400,
            },
            indent=2,
        )
        + "\n"
    )
    print(
        f"R4.3 measurements complete: {manifest['random_path_rows']} paths.",
        flush=True,
    )


if __name__ == "__main__":
    main()
