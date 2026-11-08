"""Generate reflection tables/macros from the verified, labelled evidence."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

from reproduction.prepare_r21_report import tex_number
from reproduction.prepare_r4_report import table_file
from stage4.r43.assessment import read_rows

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "stage5/results"
LABELS = {
    "base": "Original Hamming(7,4)",
    "band60": "Half-width 60 Hz",
    "band80": "Half-width 80 Hz",
    "taps257": "257 taps",
    "taps513": "513 taps",
    "bits16": "16 bit",
    "h1511": "Hamming(15,11)",
    "repeat5": "Repetition(5,1)",
    "uncoded": "Uncoded",
}


def macros_file(name: str, macros: dict[str, str]) -> None:
    """Write numerical TeX commands from saved evidence or stated equations.

    Args:
        name: Generated TeX basename.
        macros: LaTeX-safe letter-only command names and measured values.

    Returns:
        None. No random or fabricated numerical values are introduced.
    """
    (ROOT / "report/generated" / name).write_text(
        "% Generated from verified R5 evidence; do not edit numbers.\n"
        + "\n".join(r"\newcommand{\%s}{%s}" % item for item in macros.items())
        + "\n"
    )


def prepare_r51() -> None:
    """Prepare the scoped limiting-decision argument and its measured probes.

    Returns:
        None. Preserve the R4/R5 cohort distinction and feasibility labels.
    """
    check = json.loads((FOLDER / "phase1_independent_checks.json").read_text())
    assert check["status"] == "passed"
    assert json.loads((FOLDER / "r51_conclusion.json").read_text())[
        "status"
    ].startswith("R5.1")
    variants = read_rows(FOLDER / "decision_evidence.csv")
    values = {r["id"]: r for r in variants}
    component = {
        (r["scope"], r["term"]): r
        for r in read_rows(FOLDER / "r4_component_evidence.csv")
    }
    accounting = {
        (r["scope"], r["variant"]): r
        for r in read_rows(FOLDER / "confirmed_stage_accounting.csv")
    }
    before_gain = (
        float(accounting[("main12", "base")]["mean_prechannel_mse"])
        - float(accounting[("main12", "bits16")]["mean_prechannel_mse"])
    ) / 0.7299999999999999
    q7 = sum(math.comb(7, j) * 0.1**j * 0.9 ** (7 - j) for j in range(2, 8))
    q15 = sum(
        math.comb(15, j) * 0.1**j * 0.9 ** (15 - j) for j in range(2, 16)
    )
    q5 = sum(math.comb(5, j) * 0.1**j * 0.9 ** (5 - j) for j in range(3, 6))
    macros = {
        "RfiveRecords": str(check["complete_bundles"]),
        "RfivePaths": str(check["independent_path_records"]),
        "RfiveIndependentMse": tex_number(
            check["maximum_independent_MSE_difference"]
        ),
        "RfiveMainChannel": tex_number(
            float(component[("main12", "channel")]["mean_power_48k"])
        ),
        "RfiveMainGaussian": tex_number(
            float(component[("main12", "gaussian")]["mean_power_48k"])
        ),
        "RfiveMainQuant": tex_number(
            float(component[("main12", "quantisation")]["mean_power_48k"])
        ),
        "RfiveBaseJ": f"{float(values['base']['mean_objective']):.6f}",
        "RfiveRepeatJ": f"{float(values['repeat5']['mean_objective']):.6f}",
        "RfiveRepeatGain": f"{float(values['repeat5']['main_gain']):.6f}",
        "RfiveRepeatLower": (
            f"{float(values['repeat5']['main_gain_lower']):.6f}"
        ),
        "RfiveRepeatUpper": (
            f"{float(values['repeat5']['main_gain_upper']):.6f}"
        ),
        "RfiveBitsGain": f"{float(values['bits16']['main_gain']):.6f}",
        "RfiveBitsLower": f"{float(values['bits16']['main_gain_lower']):.6f}",
        "RfiveBitsUpper": f"{float(values['bits16']['main_gain_upper']):.6f}",
        "RfiveBitsPreGain": tex_number(before_gain),
        "RfiveCodeOtherJ": f"{float(values['h1511']['mean_objective']):.6f}",
        "RfiveQSeven": f"{q7:.7f}",
        "RfiveQFifteen": f"{q15:.7f}",
        "RfiveQRepeat": f"{q5:.7f}",
    }
    macros_file("r51_measurements.tex", macros)
    terms = {
        "sampling": "Sampling",
        "filter": "Filter distortion",
        "gaussian": "Gaussian",
        "quantisation": "Quantisation",
        "channel": "Channel-induced waveform change",
        "cross_sum": "Signed sum of 10 cross terms",
        "total": "Total MSE",
    }
    table_file(
        "r51_components_table.tex",
        "lrr",
        "R4 configuration 11 term & Main 12 mean power & Personal-point power",
        [
            label
            + " & $"
            + tex_number(float(component[("main12", term)]["mean_power_48k"]))
            + "$ & $"
            + tex_number(
                float(component[("personal", term)]["mean_power_48k"])
            )
            + "$"
            for term, label in terms.items()
        ],
    )
    table_file(
        "r51_decisions_table.tex",
        "llrrr",
        r"One change & Feasible & kbit/s & Mean $J$ "
        r"& Paired $J$ reduction [interval]",
        [
            LABELS[r["id"]]
            + " & "
            + ("Yes" if r["original_constraints_feasible"] == "True" else "No")
            + f" & {float(r['logical_rate_bits_s']) / 1000:.2f}"
            + f" & {float(r['mean_objective']):.5f}"
            + " & $"
            + tex_number(float(r["main_gain"]))
            + r"\ ["
            + tex_number(float(r["main_gain_lower"]))
            + ","
            + tex_number(float(r["main_gain_upper"]))
            + "]$"
            for r in variants
            if r["id"] != "base"
        ],
    )
    shutil.copy2(
        ROOT / "stage5/figures/r51_counterfactual_gains.png",
        ROOT / "report/images/r51_counterfactual_gains.png",
    )
    (FOLDER / "r51_reporting_sources.json").write_text(
        json.dumps(
            {
                "macros": macros,
                "component_cohort": "R4phase1R1400",
                "counterfactual_cohort": "R5phase1R350",
                "derived_block_failure_probabilities_at_p01": {
                    "hamming74": q7,
                    "hamming1511": q15,
                    "repetition51": q5,
                },
                "prechannel_gain_B16_not_equal_full_gain": before_gain,
            },
            indent=2,
        )
        + "\n"
    )
    print("R5.1 verified tables, macros and figure copy prepared.")


def prepare_r52() -> None:
    """Prepare the exact alternative-constraint trade-off after its audit.

    Returns:
        None. Unmeasured candidates keep explicit missing quality values.
    """
    result = json.loads((FOLDER / "constraint_assessment.json").read_text())
    verified = json.loads((FOLDER / "r52_independent_checks.json").read_text())
    assert verified["status"] == "passed"
    rows = read_rows(FOLDER / "constraint_tradeoff.csv")
    measured = {
        r["variant"]: r
        for r in rows
        if r["variant"] in {"base", "h1511", "uncoded"}
    }
    saving = 100 * (
        1
        - int(measured["h1511"]["transmitted_bits"])
        / int(measured["base"]["transmitted_bits"])
    )
    macros = {
        "SfiveThreshold": f"{result['NMSE_threshold']:.17g}",
        "SfiveNewMean": f"{float(measured['h1511']['mean_nmse']):.6f}",
        "SfiveNewLower": f"{float(measured['h1511']['mean_lower']):.6f}",
        "SfiveNewUpper": f"{float(measured['h1511']['mean_upper']):.6f}",
        "SfiveBaseMean": f"{float(measured['base']['mean_nmse']):.6f}",
        "SfiveRawMean": f"{float(measured['uncoded']['mean_nmse']):.6f}",
        "SfiveRateSaving": f"{saving:.2f}",
        "SfivePerfectBase": measured["base"]["perfect_messages"],
        "SfivePerfectNew": measured["h1511"]["perfect_messages"],
        "SfiveBitElevenRatio": (
            f"{result['bits11_worst_quantisation_component_ratio']:.6f}"
        ),
    }
    macros_file("r52_measurements.tex", macros)
    table_file(
        "r52_constraint_table.tex",
        "lrrrrll",
        "Scheme & Real bits & Pad & Sent bits & kbit/s & Rate fit & Q fit",
        [
            r["label"]
            + f" & {int(r['payload_bits'])} & {int(r['padding_bits'])}"
            + f" & {int(r['transmitted_bits'])}"
            + f" & {float(r['logical_rate_bits_s']) / 1000:.2f} & "
            + ("Yes" if r["rate_feasible"] == "1" else "No")
            + " & "
            + ("Yes" if r["quantisation_budget_feasible"] == "1" else "No")
            for r in rows
        ],
    )
    table_file(
        "r52_quality_table.tex",
        "lrrrl",
        r"Scheme & Mean NMSE [interval] & Payload BER "
        r"& Perfect payloads & Mean rule",
        [
            r["label"]
            + f" & {float(r['mean_nmse']):.6f} "
            + f"[{float(r['mean_lower']):.6f},{float(r['mean_upper']):.6f}]"
            + " & $"
            + tex_number(float(r["information_BER"]))
            + "$"
            + f" & {int(r['perfect_messages'])}/{int(r['records'])} & "
            + r["mean_quality_result"].title()
            for r in rows
            if r["variant"] in measured
        ],
    )
    theory = read_rows(FOLDER / "payload_recovery_theory.csv")
    table_file(
        "r52_payload_table.tex",
        "lrrr",
        "Scheme & Perfect fraction & Pointwise 95\\% interval "
        "& Exact-model probability",
        [
            r["label"]
            + f" & {float(r['measured_fraction']):.4f}"
            + f" & [{float(r['pointwise95_lower']):.4f},"
            + f"{float(r['pointwise95_upper']):.4f}]"
            + " & $"
            + tex_number(float(r["theoretical_payload_success"]))
            + "$"
            for r in theory
        ],
    )
    shutil.copy2(
        ROOT / "stage5/figures/r52_quality_rate.png",
        ROOT / "report/images/r52_quality_rate.png",
    )
    (FOLDER / "r52_reporting_sources.json").write_text(
        json.dumps(
            {
                "macros": macros,
                "constraint_source": "design.json R52_prespecified_constraint",
                "quality_source": (
                    "primary_summary.csv19-family personal means"
                ),
                "probability_source": (
                    "payload_recovery_theory.csv; "
                    "last partial block handled separately"
                ),
                "additional_independent_records_claimed": 0,
            },
            indent=2,
        )
        + "\n"
    )
    print(
        "R5.2 verified constraint/quality/payload tables and figure prepared."
    )


def main() -> None:
    """Generate only the reflection question whose evidence is complete.

    Returns:
        None. R5.2 cannot be reported before its separate assessment file.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part", choices=["r51", "r52", "all"], default="all")
    args = parser.parse_args()
    if args.part in {"r51", "all"}:
        prepare_r51()
    if args.part in {"r52", "all"}:
        prepare_r52()


if __name__ == "__main__":
    main()
