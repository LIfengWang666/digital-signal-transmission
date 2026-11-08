"""Measure predeclared one-decision variants on new study5 random streams."""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import scipy

from degradation import add_gaussian_noise, transmit_bsc
from stage2.r21.filtering_approach import make_rng, write_csv
from stage2.r23.validation import array_hash, file_hash
from stage5.assessment import assess_confirmation, plan_validation
from stage5.pipeline import (
    build_system,
    measure_variant,
    prepare_inputs,
    receive_variant,
)
from stage5.protocol import freeze_protocol, register_phase

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
_SYSTEM: dict | None = None


def run_chunk(task: tuple[int, int, int, str, str]) -> dict:
    """Measure complete fixed-ID bundles with honest shared-mask exposures.

    Args:
        task: Start, exclusive stop, phase, directory and freeze hash.

    Returns:
        Verified checkpoint metadata; no process-global RNG is used.
    """
    global _SYSTEM
    start, stop, phase, directory, freeze_hash = task
    folder = Path(directory)
    complete = folder / "complete.json"
    if complete.exists():
        old = json.loads(complete.read_text())
        if old["freeze_sha256"] == freeze_hash and all(
            file_hash(folder / name) == expected
            for name, expected in old["sha256"].items()
        ):
            return old
        raise ValueError("R5 checkpoint differs from the source freeze.")
    folder.mkdir(parents=True, exist_ok=True)
    if _SYSTEM is None:
        _SYSTEM = build_system()
    system, trials, gaussian, channels, quantisers = _SYSTEM, [], [], [], []
    design = system["design"]
    for repeat in range(start, stop):
        for snr in design["snr_conditions"]:
            scid, target = snr["snr_condition_id"], snr["snr_db"]
            path = (1, 5, phase, scid, repeat)
            rng = make_rng(path)
            state = rng.bit_generator.state
            noisy, noise, variance = add_gaussian_noise(
                system["x"], target, rng
            )
            banks = prepare_inputs(system, noisy)
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
                }
            )
            seen_q = set()
            for variant in system["variants"]:
                key = variant["filter_id"], variant["bits"]
                if key in seen_q:
                    continue
                seen_q.add(key)
                bank = banks[variant["id"]]
                quantisers.append(
                    {
                        "record_id": repeat,
                        "snr_condition_id": scid,
                        "filter_id": key[0],
                        "bits": key[1],
                        "overload_samples": bank["overload_samples"],
                        "quantiser_input_sha256": bank[
                            "quantiser_input_sha256"
                        ],
                        "quantisation_mse_16k": bank["quantisation_mse_16k"],
                    }
                )
            for condition in design["conditions"]:
                if condition["snr_condition_id"] != scid:
                    continue
                cid, p = (
                    condition["joint_condition_id"],
                    condition["probability"],
                )
                layout_rows = {}
                for variant in system["variants"]:
                    name, layout = variant["id"], variant["layout"]
                    bank = banks[name]
                    path = (2, 5, phase, cid, repeat, layout)
                    rng = make_rng(path)
                    state = rng.bit_generator.state
                    received, mask = transmit_bsc(bank["sent"], p, rng)
                    recovered = receive_variant(
                        received,
                        variant["code"],
                        variant["bits"],
                        variant["payload_bits"],
                    )
                    measured = measure_variant(
                        system, variant, bank, received, mask, recovered
                    )
                    trials.append(
                        {
                            "phase_id": phase,
                            "record_id": repeat,
                            "joint_condition_id": cid,
                            "snr_condition_id": scid,
                            "snr_db": target,
                            "probability": p,
                            "variant": name,
                            "layout_id": layout,
                            **measured,
                        }
                    )
                    if layout not in layout_rows:
                        channels.append(
                            {
                                "record_id": repeat,
                                "joint_condition_id": cid,
                                "layout_id": layout,
                                "probability": p,
                                "spawn_key": list(path),
                                "initial_state": state,
                                "uniform_call_shape": [len(mask)],
                                "mask_sha256": measured["mask_sha256"],
                                "channel_flips": measured["channel_flips"],
                                "shared_variants": [
                                    v["id"]
                                    for v in system["variants"]
                                    if v["layout"] == layout
                                ],
                            }
                        )
                        layout_rows[layout] = measured
                    else:
                        for key in [
                            "mask_sha256",
                            "wrong_pattern_sha256",
                            "decoded_wrong_bits",
                            "wrong_samples",
                            "wrong_msb_bits",
                        ]:
                            assert (
                                measured[key] == layout_rows[layout][key]
                            ), key
    write_csv(folder / "trials.csv", trials)
    write_csv(folder / "quantiser_inputs.csv", quantisers)
    (folder / "leaves.json").write_text(
        json.dumps({"gaussian": gaussian, "bsc": channels}, indent=2) + "\n"
    )
    result = {
        "start": start,
        "stop": stop,
        "phase": phase,
        "freeze_sha256": freeze_hash,
        "sha256": {
            p.name: file_hash(p)
            for p in folder.iterdir()
            if p.name != "complete.json"
        },
    }
    complete.write_text(json.dumps(result, indent=2) + "\n")
    return result


def merge_chunks(folders: list[Path], phase: int, count: int) -> list[dict]:
    """Merge complete bundles in deterministic order and save unique leaves.

    Args:
        folders: Increasing fixed-record chunk directories.
        phase: Declared pilot or confirmation phase.
        count: Frozen number of complete records.

    Returns:
        All trial rows for planning or assessment.
    """
    result = HERE / "results"
    trials, quantisers, gaussian, bsc = [], [], [], []
    for folder in folders:
        for name, target in [
            ("trials.csv", trials),
            ("quantiser_inputs.csv", quantisers),
        ]:
            with (folder / name).open(newline="") as stream:
                target.extend(csv.DictReader(stream))
        leaves = json.loads((folder / "leaves.json").read_text())
        gaussian.extend(leaves["gaussian"])
        bsc.extend(leaves["bsc"])
    assert len(trials) == 135 * count
    assert len(quantisers) == 36 * count
    assert len(gaussian) == 6 * count and len(bsc) == 75 * count
    paths = [tuple(r["spawn_key"]) for r in gaussian + bsc]
    assert len(paths) == len(set(paths)) == 81 * count
    assert all(path[1:3] == (5, phase) for path in paths)
    write_csv(result / f"phase{phase}_trials.csv", trials)
    write_csv(result / f"phase{phase}_quantiser_inputs.csv", quantisers)
    manifest = {
        "date": "2026-10-21",
        "study_id": 5,
        "phase_id": phase,
        "records": count,
        "root_seed": 2470964,
        "pool_size": 4,
        "bit_generator": "PCG64",
        "software": {
            "Python": platform.python_version(),
            "NumPy": np.__version__,
            "SciPy": scipy.__version__,
        },
        "freeze_sha256": file_hash(result / f"phase{phase}_freeze.json"),
        "gaussian_inputs": gaussian,
        "bsc_mask_groups": bsc,
        "unique_gaussian_inputs": len(gaussian),
        "unique_BSC_masks": len(bsc),
        "unique_quantiser_inputs": len(quantisers),
        "path_records": len(trials),
        "new_independent_units_from_shared_mask_replays": 0,
    }
    (result / f"phase{phase}_seed_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    return trials


def self_check() -> None:
    """Check all noiseless code roundtrips and one existing R4 baseline.

    Returns:
        None. New study5 random leaves are not consumed by these checks.
    """
    system = build_system()
    banks = prepare_inputs(system, system["x"])
    for variant in system["variants"]:
        bank = banks[variant["id"]]
        recovered = receive_variant(
            bank["sent"],
            variant["code"],
            variant["bits"],
            variant["payload_bits"],
        )
        assert np.array_equal(recovered["payload"], bank["payload"])
        assert np.array_equal(recovered["high"], bank["before"])
    target = next(
        c["snr_db"]
        for c in system["design"]["conditions"]
        if c["joint_condition_id"] == 11
    )
    noisy, _, _ = add_gaussian_noise(
        system["x"], target, make_rng((1, 4, 1, 5, 0))
    )
    bank = prepare_inputs(system, noisy)["base"]
    variant = next(v for v in system["variants"] if v["id"] == "base")
    p = next(
        c["probability"]
        for c in system["design"]["conditions"]
        if c["joint_condition_id"] == 11
    )
    received, mask = transmit_bsc(
        bank["sent"], p, make_rng((2, 4, 1, 11, 0, 1))
    )
    recovered = receive_variant(received, "hamming74", 14, 448000)
    measured = measure_variant(
        system, variant, bank, received, mask, recovered
    )
    with (ROOT / "stage4/r43/results/trials.csv").open(newline="") as stream:
        row = next(
            r
            for r in csv.DictReader(stream)
            if r["record_id"] == "0"
            and r["joint_condition_id"] == "11"
            and r["configuration"] == "11"
        )
    for key in [
        "mse",
        "nmse",
        "mae",
        "decoded_wrong_bits",
        "channel_flips",
        "received_sha256",
    ]:
        if isinstance(measured[key], str):
            assert measured[key] == row[key]
        else:
            assert math.isclose(
                measured[key], float(row[key]), rel_tol=1e-14, abs_tol=1e-14
            )
    print("R5 pre-freeze check: all9 roundtrips and R4 baseline match.")


def main() -> None:
    """Freeze and execute the declared pilot or independent confirmation.

    Returns:
        None. Never select a limiting decision during measurement.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", type=int, choices=[0, 1], default=0)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return
    frozen = freeze_protocol(args.phase)
    if args.freeze_only:
        print("R5 protocol frozen; no new random draw.")
        return
    register_phase(frozen)
    count = frozen["records_per_condition"]
    digest = file_hash(HERE / "results" / f"phase{args.phase}_freeze.json")
    tasks = [
        (
            i,
            min(i + 20, count),
            args.phase,
            str(HERE / ".work" / f"phase{args.phase}" / f"records_{i:04d}"),
            digest,
        )
        for i in range(0, count, 20)
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for task, _ in zip(tasks, pool.map(run_chunk, tasks)):
            print(
                f"R5 phase{args.phase}: {task[1]}/{count} complete bundles",
                flush=True,
            )
    trials = merge_chunks([Path(task[3]) for task in tasks], args.phase, count)
    freeze_protocol(args.phase)
    if args.phase == 0:
        plan = plan_validation(trials, frozen["design"], HERE / "results")
        print(
            "R5 planned confirmation records:", plan["records_per_condition"]
        )
    else:
        assess_confirmation(HERE / "results", frozen)
        print("R5 confirmation measurements and19-family assessment complete.")


if __name__ == "__main__":
    main()
