"""Freeze the four-path pilot and register only its study-4 phase-0 leaves."""

from __future__ import annotations

import json
from pathlib import Path

from stage2.r23.validation import file_hash

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def verify_sources(design: dict) -> None:
    """Check original inputs and every upstream frozen scientific source.

    Args:
        design: R4.2 choices and immutable input hashes.

    Returns:
        None. Raise before channel generation if any frozen source differs.
    """
    for name, expected in design["fixed_inputs_sha256"].items():
        if file_hash(ROOT / name) != expected:
            raise ValueError(f"R4 fixed input changed: {name}")
    for name in design["upstream_freezes"]:
        frozen = json.loads((ROOT / name).read_text(encoding="utf-8"))
        entries = frozen.get(
            "source_sha256", frozen.get("source_input_sha256")
        )
        if not entries:
            raise ValueError("Unknown upstream freeze schema.")
        if name == "stage2/r22/results/filter_freeze.json":
            entries = dict(entries)
            for source, key in [
                ("stage2/r22/design.json", "design_sha256"),
                ("stage2/r22/filter_parameters.py", "code_sha256"),
                ("stage2/r21/results/noise_draws.npy", "source_noise_sha256"),
                (
                    "stage2/r21/results/seed_manifest.json",
                    "source_manifest_sha256",
                ),
                (
                    "stage2/r22/results/selected_coefficients.npy",
                    "coefficient_sha256",
                ),
            ]:
                entries[source] = frozen[key]
        for source, expected in entries.items():
            if file_hash(ROOT / source) != expected:
                raise ValueError(f"Upstream frozen source changed: {source}")


def freeze_pilot() -> dict:
    """Fix pilot choices and add consistent phase-0 condition registrations.

    Returns:
        A saved source/choice freeze, created before any new pilot draw.
        No phase-1 leaf is registered or consumed here.
    """
    design = json.loads((HERE / "design.json").read_text(encoding="utf-8"))
    verify_sources(design)
    if (
        len(design["conditions"]) != 22
        or sum(c["primary"] for c in design["conditions"]) != 12
    ):
        raise ValueError(
            "The two predeclared slices must contain 22/12 cases."
        )
    result = HERE / "results"
    result.mkdir(exist_ok=True)
    sources = [
        HERE / name
        for name in [
            "design.json",
            "pipeline.py",
            "protocol.py",
            "pilot.py",
            "planning.py",
        ]
    ]
    frozen = {
        "date": design["date"],
        "status": "Frozen before study4 phase0 pilot",
        "study_id": 4,
        "phase_id": 0,
        "complete_bundles": 20,
        "choices": design,
        "source_sha256": {
            p.relative_to(ROOT).as_posix(): file_hash(p) for p in sources
        },
        "phase1_used": False,
    }
    freeze_path = result / "pilot_freeze.json"
    if freeze_path.exists():
        if json.loads(freeze_path.read_text()) != frozen:
            raise ValueError(
                "Pilot source/choices changed after their freeze."
            )
    else:
        freeze_path.write_text(json.dumps(frozen, indent=2) + "\n")
    registry_path = ROOT / "metadata/randomness_design.json"
    registry = json.loads(registry_path.read_text())
    if (
        registry["root_seed"] != 2470964
        or registry["pool_size"] != 4
        or registry["bit_generator"] != "PCG64"
    ):
        raise ValueError("Keep the agreed root and generator conventions.")
    entries = []
    for snr in design["snr_conditions"]:
        entries.append(
            (
                registry["condition_registry"],
                {
                    "study_id": 4,
                    "phase_id": 0,
                    "condition_id": snr["snr_condition_id"],
                    "target_snr_db": snr["snr_db"],
                    "role": "R4.2 pilot Gaussian shared across all paths/p",
                },
            )
        )
    for condition in design["conditions"]:
        entries.append(
            (
                registry["bsc_condition_registry"],
                {
                    "study_id": 4,
                    "phase_id": 0,
                    "condition_id": condition["joint_condition_id"],
                    "probability": condition["probability"],
                    "role": condition["role"],
                    "gaussian_condition_id": condition["snr_condition_id"],
                    "target_snr_db": condition["snr_db"],
                    "scope": "R4.2 phase0 pilot; p0 draws no random mask",
                },
            )
        )
    for target, entry in entries:
        prior = [
            c
            for c in target
            if all(
                c[key] == entry[key]
                for key in ["study_id", "phase_id", "condition_id"]
            )
        ]
        if prior and prior != [entry]:
            raise ValueError(
                "An existing condition ID has different semantics."
            )
        if not prior:
            target.append(entry)
    registry["r42_protocol"] = "../stage4/r42/design.json"
    registry["r42_pilot_freeze"] = "../stage4/r42/results/pilot_freeze.json"
    registry["r42_seed_manifest"] = "../stage4/r42/results/seed_manifest.json"
    registry["status"] = "R4.2 phase0 pilot registered; R4 phase1 unused"
    registry_path.write_text(json.dumps(registry, indent=2) + "\n")
    return frozen
