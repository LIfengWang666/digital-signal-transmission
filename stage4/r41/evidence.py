"""Evaluate metric semantics and existing R3.4 examples without RNG draws."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
from pathlib import Path

import numpy as np

from stage2.r21.filtering_approach import write_csv
from stage4.r41.metrics import FloatArray, db_from_nmse, reconstruction_metrics

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def semantic_cases(reference: FloatArray) -> dict[str, FloatArray]:
    """Build declared deterministic examples, not simulated channel records.

    Args:
        reference: Original 96000-point reference, retained in its own units.

    Returns:
        Seven reconstructions for exactness, gain, bias, loss ranking,
        constant-output and one-sample circular-delay checks.
    """
    diffuse = np.full(reference.size, 0.1)
    diffuse[1::2] *= -1
    sparse = np.zeros(reference.size)
    sparse[::100] = 2.0
    sparse[100::200] = -2.0
    return {
        "exact": reference.copy(),
        "double_gain": 2 * reference,
        "offset_0.1": reference + 0.1,
        "diffuse_0.1": reference + diffuse,
        "sparse_2_at_1percent": reference + sparse,
        "zero_output": np.zeros_like(reference),
        "circular_delay_1sample": np.roll(reference, 1),
    }


def legacy_examples(reference: FloatArray) -> list[dict]:
    """Re-evaluate the two existing full-record R3.4 example files.

    Args:
        reference: Original NPY used to verify every saved reference value.

    Returns:
        Five labelled metric rows with R3.4 record and error-count provenance.
        The repeated prechannel waveform is counted only once.
    """
    previous = ROOT / "stage3/r34/results"
    with (previous / "trials.csv").open(newline="", encoding="utf-8") as f:
        trials = list(csv.DictReader(f))
    rows = []
    prechannel = None
    for name, cid in [("personal", 11), ("high_error", 5)]:
        path = previous / f"example_{name}_48000hz.csv"
        data = np.genfromtxt(path, delimiter=",", names=True)
        if not np.array_equal(data["reference"], reference):
            raise ValueError("Legacy example does not use the original NPY.")
        if not np.array_equal(data["index"], np.arange(reference.size)):
            raise ValueError("Legacy example is not the full reference grid.")
        if not np.allclose(
            data["time_s"],
            np.arange(reference.size) / 48000,
            atol=1e-15,
            rtol=0,
        ):
            raise ValueError("Legacy example time coordinates differ.")
        if prechannel is None:
            prechannel = data["prechannel"]
            rows.append(
                {
                    "case": "prechannel",
                    "origin": "existing R3.4 record0; before BSC",
                    "source_file": path.relative_to(ROOT).as_posix(),
                    "condition_id": 11,
                    "record_id": 0,
                    "code": "before_bsc",
                    "probability": "not_applied",
                    "decoded_wrong_bits": "not_applied",
                    **reconstruction_metrics(reference, prechannel),
                }
            )
        elif not np.array_equal(prechannel, data["prechannel"]):
            raise ValueError("The paired legacy input unexpectedly differs.")
        for code in ["uncoded", "hamming74"]:
            old = next(
                r
                for r in trials
                if int(r["condition_id"]) == cid
                and int(r["record_id"]) == 0
                and r["code"] == code
            )
            measured = reconstruction_metrics(reference, data[code])
            for key in ["mse", "nmse"]:
                if not np.isclose(
                    measured[key],
                    float(old[f"{key}_48000"]),
                    rtol=2e-13,
                    atol=1e-15,
                ):
                    raise ValueError("Metric differs from frozen R3.4 output.")
            rows.append(
                {
                    "case": f"{name}_{code}",
                    "origin": "existing R3.4 phase1 example; not R4 data",
                    "source_file": path.relative_to(ROOT).as_posix(),
                    "condition_id": cid,
                    "record_id": 0,
                    "code": code,
                    "probability": old["probability"],
                    "decoded_wrong_bits": int(old["decoded_wrong_bits"]),
                    **measured,
                }
            )
    return rows


def main() -> None:
    """Write deterministic metric evidence and input/source provenance.

    Returns:
        None. Write R4.1 CSV and JSON without changing earlier results or RNG.
    """
    design = json.loads((HERE / "design.json").read_text(encoding="utf-8"))
    freeze_path = HERE / "metric_definition_freeze.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    for name, expected in freeze["source_sha256"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Frozen metric definition changed: {name}")
    for name, expected in design["input_sha256"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"R4.1 input hash changed: {name}")
    reference = np.load(ROOT / "reference_signal.npy", allow_pickle=False)
    if reference.shape != (design["reference_samples"],):
        raise ValueError("Unexpected reference shape.")
    results = HERE / "results"
    results.mkdir(exist_ok=True)
    semantics = [
        {
            "case": name,
            "origin": "deterministic semantic example, not channel data",
            **reconstruction_metrics(reference, signal),
        }
        for name, signal in semantic_cases(reference).items()
    ]
    write_csv(results / "semantic_checks.csv", semantics)
    measured = legacy_examples(reference)
    write_csv(results / "existing_record_metrics.csv", measured)
    losses = next(r for r in semantics if r["case"] == "diffuse_0.1"), next(
        r for r in semantics if r["case"] == "sparse_2_at_1percent"
    )
    nmse = np.array([r["nmse"] for r in losses])
    write_csv(
        results / "aggregation_example.csv",
        [
            {
                "source": "two deterministic ranking examples, equal weight",
                "record_count": len(nmse),
                "mean_nmse": float(np.mean(nmse)),
                "db_from_mean_nmse": db_from_nmse(float(np.mean(nmse))),
                "mean_of_record_db": float(
                    np.mean([db_from_nmse(v) for v in nmse])
                ),
            }
        ],
    )
    manifest = {
        "date": "2026-10-21",
        "status": "R4.1 metric selection implemented",
        "primary_metric": design["primary_metric"],
        "metric_selection_source": (
            "stage4/FRAMEWORK.md, prepared before this metric evaluation"
        ),
        "semantic_cases": len(semantics),
        "legacy_full_record_examples": len(measured),
        "new_R4_channel_records": 0,
        "new_rng_draws": 0,
        "legacy_examples_are_independent_validation_for_R4": False,
        "confidence_intervals_estimated_here": False,
        "software": {
            "Python": platform.python_version(),
            "NumPy": np.__version__,
        },
        "input_sha256": design["input_sha256"],
        "source_sha256": {
            path.relative_to(ROOT)
            .as_posix(): hashlib.sha256(path.read_bytes())
            .hexdigest()
            for path in [
                HERE / "design.json",
                HERE / "metrics.py",
                HERE / "evidence.py",
                freeze_path,
            ]
        },
    }
    (results / "metric_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        "R4.1: 7 semantic checks and 5 existing full-record examples; no RNG."
    )


if __name__ == "__main__":
    main()
