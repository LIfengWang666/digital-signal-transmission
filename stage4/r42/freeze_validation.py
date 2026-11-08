"""Freeze the R4 validation choices after independent phase-0 verification."""

from __future__ import annotations

import json
from pathlib import Path

from stage2.r23.validation import file_hash
from stage4.r42.protocol import verify_sources

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def main() -> None:
    """Fix the pilot-planned repeat count and choices without validation draws.

    Returns:
        None. Save an idempotent phase-1 choice freeze for the later R4.3 run.
        R4.3 must also hash its measurement/reporting code before its draws.
    """
    folder = HERE / "results"
    pilot = json.loads((folder / "pilot_freeze.json").read_text())
    plan = json.loads((folder / "validation_planning.json").read_text())
    check = json.loads((folder / "independent_checks.json").read_text())
    if (
        check["status"] != "passed"
        or check["independent_path_records"] != 1760
    ):
        raise ValueError("Independently verify the complete pilot first.")
    verify_sources(pilot["choices"])
    for name, expected in pilot["source_sha256"].items():
        if file_hash(ROOT / name) != expected:
            raise ValueError("The verified pipeline changed after the pilot.")
    records = plan["candidate_validation_records_per_condition"]
    if records != check["candidate_validation_records_verified"]:
        raise ValueError(
            "Repeat planning and independent verification differ."
        )
    sources = dict(pilot["source_sha256"])
    files = [
        HERE / "freeze_validation.py",
        ROOT / "reproduction/check_r42_results.py",
        *[
            folder / name
            for name in [
                "pilot_freeze.json",
                "pilot_trials.csv",
                "pilot_summary.csv",
                "pilot_bundle_scores.csv",
                "precision_planning.csv",
                "validation_planning.json",
                "independent_checks.json",
            ]
        ],
    ]
    sources.update(
        {p.relative_to(ROOT).as_posix(): file_hash(p) for p in files}
    )
    result = {
        "date": "2026-10-21",
        "status": "Choices frozen before R4 phase1",
        "study_id": 4,
        "phase_id": 1,
        "records_per_condition": records,
        "conditions": pilot["choices"]["conditions"],
        "configurations": pilot["choices"]["configurations"],
        "snr_conditions": pilot["choices"]["snr_conditions"],
        "main_objective": "equal mean of12 unique main cases per full bundle",
        "primary_metric": "frozen R4.1 NMSE, original 96000-point reference",
        "confidence_family_size": 92,
        "familywise_alpha": 0.05,
        "mean_interval": "mean +/- t.ppf(1-.05/(2*92),R-1)*sample_sd/sqrt(R)",
        "interval_scope": (
            "approximate finite-family mean intervals; t/CLT approximation, "
            "not exact finite-sample or continuous-parameter coverage"
        ),
        "precision_rule": plan["rule"],
        "rare_event_planning_limitation": plan["limitation"],
        "gaussian_path": [1, 4, 1, "snr_condition_id", "record_id"],
        "bsc_path": [2, 4, 1, "joint_condition_id", "record_id", "layout_id"],
        "configuration_pairing": pilot["choices"]["pairing"],
        "phase0_observations_imported_into_validation": False,
        "phase1_used_to_choose_any_rule": False,
        "phase1_records_generated": 0,
        "source_sha256": sources,
        "fixed_inputs_sha256": pilot["choices"]["fixed_inputs_sha256"],
        "upstream_freezes": pilot["choices"]["upstream_freezes"],
        "validation_code_freeze_still_required_before_R4_3_draws": True,
    }
    path = folder / "validation_choices_freeze.json"
    if path.exists() and json.loads(path.read_text()) != result:
        raise ValueError("Do not silently revise the phase-1 choice freeze.")
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"R4 phase1 choices frozen at {records} bundles; no phase1 draw.")


if __name__ == "__main__":
    main()
