"""Freeze the held-out measurement code before study-4 phase-1 draws."""

from __future__ import annotations

import json
from pathlib import Path

from stage2.r23.validation import file_hash
from stage4.r42.protocol import verify_sources

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
CHOICES = ROOT / "stage4/r42/results/validation_choices_freeze.json"


def freeze_measurement() -> dict:
    """Check prior choices and freeze measurement/statistics source hashes.

    Returns:
        An immutable phase-1 protocol, saved before any held-out draw.
        Re-execution checks this record instead of silently replacing it.
    """
    choices = json.loads(CHOICES.read_text())
    verify_sources(choices)
    for name, expected in choices["source_sha256"].items():
        if file_hash(ROOT / name) != expected:
            raise ValueError(f"A frozen R4.2 choice/source changed: {name}")
    if choices["records_per_condition"] != 1400:
        raise ValueError("Keep the independently checked 1400-record plan.")
    design = json.loads((HERE / "design.json").read_text())
    sources = [
        "stage4/r43/design.json",
        "stage4/r43/protocol.py",
        "stage4/r43/evaluation.py",
        "stage4/r43/assessment.py",
        "stage4/r42/pipeline.py",
        "stage4/r41/metrics.py",
        "degradation.py",
        "stage2/r21/filtering_approach.py",
        "stage1/r12/sampling_rate.py",
        "stage3/bitstream.py",
        "stage3/r31/quantisation.py",
        "stage3/r32/codes.py",
    ]
    frozen = {
        "date": "2026-10-21",
        "status": (
            "Measurement/statistics implementation frozen before phase1 draws"
        ),
        "study_id": 4,
        "phase_id": 1,
        "records_per_condition": 1400,
        "choices_freeze_sha256": file_hash(CHOICES),
        "choices": choices,
        "measurement_design": design,
        "source_sha256": {name: file_hash(ROOT / name) for name in sources},
        "parallelism": (
            "Independent complete-record chunks only; scheduling never "
            "changes leaf, call shape, array formula or aggregation order"
        ),
        "phase0_observations_imported": False,
        "optional_stopping": False,
    }
    folder = HERE / "results"
    folder.mkdir(exist_ok=True)
    path = folder / "validation_freeze.json"
    if path.exists():
        if json.loads(path.read_text()) != frozen:
            raise ValueError(
                "R4.3 frozen measurement code or choices changed."
            )
    else:
        path.write_text(json.dumps(frozen, indent=2) + "\n")
    return frozen


def register_validation(choices: dict) -> None:
    """Append only the declared phase-1 condition IDs to the RNG registry.

    Args:
        choices: Existing R4.2 validation-choice freeze.

    Returns:
        None. Preserve all prior entries and personal parameter draws.
    """
    path = ROOT / "metadata/randomness_design.json"
    registry = json.loads(path.read_text())
    if (
        registry["root_seed"] != 2470964
        or registry["pool_size"] != 4
        or registry["bit_generator"] != "PCG64"
    ):
        raise ValueError("The agreed RNG architecture changed.")
    for key, cases in [
        ("condition_registry", choices["snr_conditions"]),
        ("bsc_condition_registry", choices["conditions"]),
    ]:
        for case in cases:
            gaussian = key == "condition_registry"
            cid = case[
                "snr_condition_id" if gaussian else "joint_condition_id"
            ]
            entry = {
                "study_id": 4,
                "phase_id": 1,
                "condition_id": cid,
                "target_snr_db": case["snr_db"],
                "role": (
                    "R4.3 held-out validation; frozen choices and record count"
                ),
            }
            if not gaussian:
                entry.update(
                    probability=case["probability"],
                    gaussian_condition_id=case["snr_condition_id"],
                    scope="p0 is identity, without a BSC random draw",
                )
            prior = [
                row
                for row in registry[key]
                if (row["study_id"], row["phase_id"], row["condition_id"])
                == (4, 1, cid)
            ]
            if prior and prior != [entry]:
                raise ValueError("A registered phase-1 condition changed.")
            if not prior:
                registry[key].append(entry)
    registry["r43_validation_freeze"] = (
        "../stage4/r43/results/validation_freeze.json"
    )
    registry["status"] = (
        "R4.3 phase1 registered; retain all earlier study/phase records"
    )
    path.write_text(json.dumps(registry, indent=2) + "\n")
