"""Freeze R5 candidate/constraint studies before each new random phase."""

from __future__ import annotations

import json
from pathlib import Path

from stage2.r23.validation import file_hash
from stage5.pipeline import build_system, load_design

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent


def freeze_protocol(phase: int) -> dict:
    """Freeze sources and the pilot or planned confirmation record count.

    Args:
        phase:0 for20 design bundles or1 for the pilot-planned confirmation.

    Returns:
        Immutable source/choice protocol. No random draw occurs here.
    """
    if phase not in {0, 1}:
        raise ValueError("Use the declared pilot or confirmation phase.")
    design = load_design()
    old = json.loads(
        (ROOT / "stage4/r43/results/validation_freeze.json").read_text()
    )
    for name, expected in old["source_sha256"].items():
        if file_hash(ROOT / name) != expected:
            raise ValueError(f"A frozen R4 source changed: {name}")
    system = build_system(save=True)
    sources = [
        "stage5/design.json",
        "stage5/protocol.py",
        "stage5/pipeline.py",
        "stage5/counterfactual.py",
        "stage5/assessment.py",
        "stage2/r22/filter_parameters.py",
        "stage4/r41/metrics.py",
        "stage4/r43/assessment.py",
        "stage3/r32/codes.py",
        "stage3/r31/quantisation.py",
        "stage1/r12/sampling_rate.py",
        "degradation.py",
        *design["fixed_sources"],
        *[f"stage5/results/kernel_{c['id']}.npy" for c in design["kernels"]],
        "stage5/results/variants.csv",
        "stage5/results/kernel_coefficients.csv",
    ]
    count = 20
    plan = None
    if phase == 1:
        plan = json.loads(
            (HERE / "results/validation_planning.json").read_text()
        )
        count = plan["records_per_condition"]
        sources += [
            "stage5/results/phase0_freeze.json",
            "stage5/results/validation_planning.json",
            "stage5/results/phase0_primary.csv",
        ]
    frozen = {
        "date": "2026-10-21",
        "status": "Frozen before R5 new phase draws",
        "study_id": 5,
        "phase_id": phase,
        "records_per_condition": count,
        "design": design,
        "variants": system["variants"],
        "planning": plan,
        "source_sha256": {name: file_hash(ROOT / name) for name in sources},
        "phase0_imported_into_confirmation": False,
        "R52_personal_subset_prespecified": True,
        "R52_subset_is_reused_R5_data_not_extra_independent_records": True,
    }
    path = HERE / "results" / f"phase{phase}_freeze.json"
    if path.exists():
        if json.loads(path.read_text()) != frozen:
            raise ValueError("A frozen R5 source or choice changed.")
    else:
        path.write_text(json.dumps(frozen, indent=2) + "\n")
    return frozen


def register_phase(frozen: dict) -> None:
    """Append only the declared study5 phase/conditions and five layouts.

    Args:
        frozen: Saved pre-draw study protocol.

    Returns:
        None. Existing studies, root seed and personal parameters persist.
    """
    path = ROOT / "metadata/randomness_design.json"
    registry = json.loads(path.read_text())
    assert registry["root_seed"] == 2470964 and registry["pool_size"] == 4
    assert registry["bit_generator"] == "PCG64"
    phase, design = frozen["phase_id"], frozen["design"]
    for key, conditions in [
        ("condition_registry", design["snr_conditions"]),
        ("bsc_condition_registry", design["conditions"]),
    ]:
        for condition in conditions:
            gaussian = key == "condition_registry"
            cid = condition[
                "snr_condition_id" if gaussian else "joint_condition_id"
            ]
            entry = {
                "study_id": 5,
                "phase_id": phase,
                "condition_id": cid,
                "target_snr_db": condition["snr_db"],
                "role": "R5 predeclared single-decision/constraint study",
            }
            if not gaussian:
                entry.update(
                    probability=condition["probability"],
                    gaussian_condition_id=condition["snr_condition_id"],
                )
            prior = [
                r
                for r in registry[key]
                if (r["study_id"], r["phase_id"], r["condition_id"])
                == (5, phase, cid)
            ]
            if prior and prior != [entry]:
                raise ValueError("Registered study5 leaf semantics changed.")
            if not prior:
                registry[key].append(entry)
    layouts = {}
    for v in frozen["variants"]:
        layout = {
            k: v[k]
            for k in [
                "layout",
                "bits",
                "code",
                "payload_bits",
                "padding_bits",
                "transmitted_bits",
            ]
        }
        if v["layout"] in layouts:
            assert layouts[v["layout"]] == layout
        layouts[v["layout"]] = layout
    registry["r5_design"] = "../stage5/design.json"
    registry["r5_layouts"] = list(layouts.values())
    registry["r5_status"] = (
        f"Study5 phase{phase} registered; sources and count fixed before draws"
    )
    path.write_text(json.dumps(registry, indent=2) + "\n")
