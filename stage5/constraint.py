"""Assess the prespecified R5.2 rate constraint after completing R5.1."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from stage2.r21.filtering_approach import write_csv
from stage2.r23.validation import file_hash
from stage3.r32.codes import binary_words, decode, matrices
from stage4.r43.assessment import read_rows

ROOT = Path(__file__).resolve().parents[1]
FOLDER = Path(__file__).resolve().parent / "results"
LABELS = {
    "base": "Hamming(7,4)",
    "h1511": "Hamming(15,11)",
    "uncoded": "Uncoded",
    "repeat5": "Repetition(5,1)",
    "bits11_h74": "11 bit + (7,4)",
}


def payload_probability(
    code: str, payload: int, p: float
) -> tuple[dict, list]:
    """Compute exact iid-BSC true-payload recovery, including partial words.

    Args:
        code: Fixed systematic code/decoder name.
        payload: Number of real information bits before zero padding.
        p: Fixed independent channel flip probability.

    Returns:
        Probability/accounting and counts of last-word error patterns whose
        real data prefix is correct. Padding correctness is not required.
    """
    generator, _ = matrices(code)
    k, n = generator.shape
    blocks = (payload + k - 1) // k
    real_last = payload - (blocks - 1) * k
    radius = 1 if code.startswith("hamming") else 0
    errors = binary_words(n)
    message, _ = decode(errors.reshape(-1), code)
    prefix_ok = ~np.any(message.reshape(-1, k)[:, :real_last], axis=1)
    weights = np.sum(errors, axis=1)
    counts = [int(np.sum(prefix_ok & (weights == w))) for w in range(n + 1)]
    q = math.fsum(
        math.comb(n, w) * p**w * (1 - p) ** (n - w)
        for w in range(radius + 1, n + 1)
    )
    last_success = math.fsum(
        counts[w] * p**w * (1 - p) ** (n - w) for w in range(n + 1)
    )
    probability = math.exp(
        (blocks - 1) * math.log1p(-q) + math.log(last_success)
    )
    rows = [
        {
            "code": code,
            "n": n,
            "k": k,
            "last_real_bits": real_last,
            "error_weight": w,
            "prefix_correct_error_patterns": counts[w],
            "all_patterns_at_weight": math.comb(n, w),
        }
        for w in range(n + 1)
    ]
    return {
        "code": code,
        "payload_bits": payload,
        "n": n,
        "k": k,
        "blocks": blocks,
        "padding_bits": blocks * k - payload,
        "last_real_bits": real_last,
        "block_failure_probability": q,
        "last_prefix_success_probability": last_success,
        "theoretical_payload_success": probability,
    }, rows


def main() -> None:
    """Compare unchanged personal-point quality with exact logical bit costs.

    Returns:
        None. Reuse the prespecified R5 cohort; draw no additional samples.
        Save traceable cost/quality decisions and exact-model probabilities.
    """
    completed = json.loads(
        (ROOT / "stage5/results/phase1_independent_checks.json").read_text()
    )
    independent = json.loads(
        (FOLDER / "phase1_independent_checks.json").read_text()
    )
    assert completed["status"] == independent["status"] == "passed"
    design = json.loads((FOLDER.parent / "design.json").read_text())
    rule = design["R52_prespecified_constraint"]
    threshold, cap = rule["primary_NMSE_threshold"], rule["maximum_bits_s"]
    variants = {r["id"]: r for r in read_rows(FOLDER / "variants.csv")}
    means = {
        r["statistic"].split(":")[-1]: r
        for r in read_rows(FOLDER / "primary_summary.csv")
        if r["statistic"].startswith("mean:personal:")
    }
    personal = {
        r["variant"]: r
        for r in read_rows(FOLDER / "case_summary.csv")
        if r["joint_condition_id"] == "11"
    }
    bits11 = next(
        r
        for r in read_rows(ROOT / "stage3/r31/results/bit_selection.csv")
        if r["bits"] == "11"
    )
    assert float(bits11["worst_component_ratio"]) > 0.01
    costs, probabilities, prefix_rows = [], [], []
    for name in ["base", "h1511", "uncoded", "repeat5", "bits11_h74"]:
        if name == "bits11_h74":
            payload, pad, sent = 32000 * 11, 0, 32000 * 11 // 4 * 7
        else:
            payload, pad, sent = (
                int(variants[name][key])
                for key in ["payload_bits", "padding_bits", "transmitted_bits"]
            )
        rate = sent / rule["duration_s"]
        row = {
            "variant": name,
            "label": LABELS[name],
            "payload_bits": payload,
            "padding_bits": pad,
            "transmitted_bits": sent,
            "logical_rate_bits_s": rate,
            "rate_feasible": int(rate <= cap),
            "quantisation_budget_feasible": int(name != "bits11_h74"),
            "mean_nmse": "",
            "mean_half_width": "",
            "mean_lower": "",
            "mean_upper": "",
            "information_BER": "",
            "perfect_messages": "",
            "records": "",
            "mean_quality_passes": "",
            "mean_quality_result": "not assessed in primary family",
        }
        if name in means:
            mean, case = means[name], personal[name]
            low, high = float(mean["lower"]), float(mean["upper"])
            verdict = (
                "pass"
                if high <= threshold
                else "fail" if low > threshold else "unresolved"
            )
            row.update(
                mean_nmse=float(mean["mean"]),
                mean_half_width=float(mean["half_width"]),
                mean_lower=low,
                mean_upper=high,
                information_BER=float(case["information_BER"]),
                perfect_messages=int(case["perfect_messages"]),
                records=int(mean["records"]),
                mean_quality_passes=int(verdict == "pass"),
                mean_quality_result=verdict,
            )
            p = float(case["probability"])
            theory, patterns = payload_probability(
                variants[name]["code"], payload, p
            )
            prefix_rows.extend(patterns)
            ci = binomtest(
                int(case["perfect_messages"]), int(case["records"])
            ).proportion_ci(confidence_level=0.95, method="exact")
            probabilities.append(
                {
                    "variant": name,
                    "label": LABELS[name],
                    **theory,
                    "probability": p,
                    "perfect_messages": int(case["perfect_messages"]),
                    "records": int(case["records"]),
                    "measured_fraction": float(
                        case["message_success_fraction"]
                    ),
                    "pointwise95_lower": ci.low,
                    "pointwise95_upper": ci.high,
                    "interval_unit": "independent complete payload record",
                }
            )
        costs.append(row)
    selected = [
        r["variant"]
        for r in costs
        if r["rate_feasible"]
        and r["quantisation_budget_feasible"]
        and r["mean_quality_passes"] == 1
    ]
    assert selected == ["h1511"]
    write_csv(FOLDER / "constraint_tradeoff.csv", costs)
    write_csv(FOLDER / "payload_recovery_theory.csv", probabilities)
    write_csv(FOLDER / "last_block_prefix_counts.csv", prefix_rows)
    sources = [
        "stage5/design.json",
        "stage5/constraint.py",
        "stage5/results/phase1_independent_checks.json",
        "stage5/results/phase1_freeze.json",
        "stage5/results/phase1_independent_checks.json",
        "stage5/results/primary_summary.csv",
        "stage5/results/case_summary.csv",
        "stage5/results/variants.csv",
        "stage3/r31/results/bit_selection.csv",
        "stage3/r32/codes.py",
    ]
    result = {
        "date": "2026-10-21",
        "status": "complete; awaiting independent R5.2 derived audit",
        "selected_variant": "h1511",
        "maximum_bits_s": cap,
        "NMSE_threshold": threshold,
        "threshold_origin": rule["threshold_origin"],
        "scope": "personal SNR/p only; fixed14bit,A18,FIR385,16kHz",
        "records_reused_from_R5_confirmation": 350,
        "additional_independent_records": 0,
        "reused_subset_and_rule_prespecified": True,
        "mean_interval_family": 19,
        "message_interval_method": "pointwise95% Clopper-Pearson",
        "bits11_worst_quantisation_component_ratio": float(
            bits11["worst_component_ratio"]
        ),
        "bits11_new_whole_chain_quality_measured": False,
        "perfect_payload_means_real_bits_not_padding": True,
        "logical_rate_excludes_unmodelled_headers_modulation": True,
        "stronger_protection_or_energy_saving_claim": False,
        "source_sha256": {name: file_hash(ROOT / name) for name in sources},
    }
    (FOLDER / "constraint_assessment.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print("R5.2 assessed: Hamming(15,11) fits rate and retained mean-quality.")


if __name__ == "__main__":
    main()
