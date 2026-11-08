"""Generate R4.3 report numbers and tables strictly from verified CSV."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from reproduction.prepare_r21_report import tex_number
from stage4.r43.assessment import read_rows

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ["00", "10", "01", "11"]


def table_file(name: str, columns: str, header: str, rows: list[str]) -> None:
    """Save a booktabs table assembled from measured numeric rows.

    Args:
        name: Generated TeX basename.
        columns: LaTeX column alignment specification.
        header: Complete column heading row without its line terminator.
        rows: Data rows without their line terminators.

    Returns:
        None. Keep table layout separate from hand-written discussion.
    """
    lines = [
        r"\begin{tabular}{" + columns + "}",
        r"\toprule",
        header + r" \\",
        r"\midrule",
    ]
    lines += [row + r" \\" for row in rows]
    lines += [r"\bottomrule", r"\end{tabular}"]
    (ROOT / "report/generated" / name).write_text("\n".join(lines) + "\n")


def main() -> None:
    """Read verified held-out measurements and populate report assets.

    Returns:
        None. Write tables, macros, a provenance manifest and figure copies.
        No reported number is a manually substituted ideal result.
    """
    folder = ROOT / "stage4/r43/results"
    checked = json.loads((folder / "independent_checks.json").read_text())
    assert checked["status"] == "passed"
    summary = read_rows(folder / "summary.csv")
    objectives = read_rows(folder / "objective_summary.csv")
    effects = read_rows(folder / "effect_summary.csv")
    precision = read_rows(folder / "precision_audit.csv")
    assessment = json.loads((folder / "assessment.json").read_text())
    personal = {
        r["configuration"]: r
        for r in summary
        if r["joint_condition_id"] == "11"
    }
    high = {
        r["configuration"]: r
        for r in summary
        if r["joint_condition_id"] == "21"
    }
    controls = {
        r["configuration"]: r for r in read_rows(folder / "clean_controls.csv")
    }
    components = {
        (r["joint_condition_id"], r["configuration"]): r
        for r in read_rows(folder / "component_summary.csv")
    }
    reference_power = float(controls["00"]["reference_power"])
    uniform_level_variance = 18**2 * (2**14 + 1) / (3 * (2**14 - 1))
    expected_half_nmse = (
        1 + uniform_level_variance * (1 - 1 / (2 * 32000)) / reference_power
    )
    selected_effect = {
        (r["joint_condition_id"], r["effect"]): r for r in effects
    }
    high_filter = selected_effect[("21", "filter_with_code")]
    macros = {
        "WRecords": "1400",
        "WPaths": "123200",
        "WGaussian": "11200",
        "WMasks": "58800",
        "WIndependentMse": tex_number(
            checked["maximum_independent_MSE_difference"]
        ),
        "WDecompositionResidual": tex_number(
            assessment["maximum_mse_decomposition_residual"]
        ),
        "WCritical": f"{assessment['t_critical']:.6f}",
        "WPrecisionFailures": str(
            len(assessment["statistics_above_planning_target"])
        ),
        "WMaxRelativePercent": (
            f"{100 * assessment['maximum_actual_relative_half_width']:.2f}"
        ),
        "WPersonalNeither": f"{float(personal['00']['mean_nmse']):.6f}",
        "WPersonalFilter": f"{float(personal['10']['mean_nmse']):.6f}",
        "WPersonalCode": f"{float(personal['01']['mean_nmse']):.6f}",
        "WPersonalBoth": f"{float(personal['11']['mean_nmse']):.6f}",
        "WHighCode": tex_number(float(high["01"]["mean_nmse"])),
        "WHighBoth": tex_number(float(high["11"]["mean_nmse"])),
        "WHighFilterReduction": tex_number(
            float(high_filter["mean_reduction"])
        ),
        "WHighFilterLower": tex_number(
            float(high_filter["simultaneous_lower"])
        ),
        "WHighFilterUpper": tex_number(
            float(high_filter["simultaneous_upper"])
        ),
        "WCleanRaw": tex_number(float(controls["00"]["nmse"])),
        "WCleanFilter": tex_number(float(controls["10"]["nmse"])),
        "WHalfTheory": f"{expected_half_nmse:.6f}",
    }
    (ROOT / "report/generated/r43_measurements.tex").write_text(
        "% Generated from held-out R4.3 results; no pilot data.\n"
        + "\n".join(r"\newcommand{\%s}{%s}" % item for item in macros.items())
        + "\n"
    )
    table_file(
        "r43_personal_table.tex",
        "lrrrr",
        r"Config. & Mean NMSE & Adjusted mean interval & Mean MAE & Data BER",
        [
            config
            + f" & {float(personal[config]['mean_nmse']):.6f}"
            + f" & [{float(personal[config]['lower_nmse']):.6f}, "
            + f"{float(personal[config]['upper_nmse']):.6f}]"
            + f" & {float(personal[config]['mean_mae']):.6f} & $"
            + tex_number(float(personal[config]["information_BER"]))
            + "$"
            for config in CONFIGS
        ],
    )
    table_file(
        "r43_objective_table.tex",
        "lrrrr",
        r"Config. & $\bar J$ & Adjusted mean interval "
        r"& Half-width (\%) & kbit/s",
        [
            r["configuration"]
            + f" & {float(r['mean_objective']):.5f}"
            + f" & [{float(r['lower_objective']):.5f}, "
            + f"{float(r['upper_objective']):.5f}]"
            + f" & {100 * float(r['relative_half_width']):.2f}"
            + (" & 392" if r["configuration"][1] == "1" else " & 224")
            for r in objectives
        ],
    )
    labels = {
        "filter_without_code": r"$G_{F\mid C=0}$",
        "filter_with_code": r"$G_{F\mid C=1}$",
        "code_without_filter": r"$G_{C\mid F=0}$",
        "code_with_filter": r"$G_{C\mid F=1}$",
        "interaction": "$I$",
    }
    table_file(
        "r43_effects_table.tex",
        "llrr",
        r"Condition & Contrast & Mean reduction & Conservative interval",
        [
            ("Personal" if cid == "11" else "40 dB")
            + " & "
            + labels[name]
            + " & $"
            + tex_number(float(selected_effect[(cid, name)]["mean_reduction"]))
            + "$ & $["
            + tex_number(
                float(selected_effect[(cid, name)]["simultaneous_lower"])
            )
            + ", "
            + tex_number(
                float(selected_effect[(cid, name)]["simultaneous_upper"])
            )
            + "]$"
            for cid in ["11", "21"]
            for name in labels
        ],
    )
    worst = sorted(
        precision, key=lambda r: float(r["relative_half_width"]), reverse=True
    )[:4]
    table_file(
        "r43_precision_table.tex",
        "lrr",
        r"Statistic (joint ID, config.) & Half-width (\%) & Target (\%)",
        [
            r["statistic"]
            .replace("condition:", "")
            .replace("objective:", "J,")
            .replace(":", ", ")
            + f" & {100 * float(r['relative_half_width']):.2f} & 5.00"
            for r in worst
        ],
    )
    component_names = {
        "power_sampling": "Sampling",
        "power_filter": "Filter",
        "power_gaussian": "Gaussian",
        "power_quantisation": "Quantisation",
        "power_channel": "Channel",
    }
    component_values = [
        label
        + " & $"
        + tex_number(float(components[("11", "11")][name]))
        + "$ & $"
        + tex_number(float(components[("21", "11")][name]))
        + "$"
        for name, label in component_names.items()
    ]
    cross_sums = {
        cid: sum(
            float(v)
            for k, v in components[(cid, "11")].items()
            if k.startswith("cross_")
        )
        for cid in ["11", "21"]
    }
    component_values.append(
        "Signed sum of 10 cross terms & $"
        + tex_number(cross_sums["11"])
        + "$ & $"
        + tex_number(cross_sums["21"])
        + "$"
    )
    component_values.append(
        "Total MSE & $"
        + tex_number(
            float(components[("11", "11")]["mean_mse_from_components"])
        )
        + "$ & $"
        + tex_number(
            float(components[("21", "11")]["mean_mse_from_components"])
        )
        + "$"
    )
    table_file(
        "r43_components_table.tex",
        "lrr",
        "Term (configuration 11) & Personal point & 40 dB, personal $p$",
        component_values,
    )
    for name in [
        "r43_central_quality",
        "r43_conditional_effects",
        "r43_error_components",
        "r43_fixed_examples",
    ]:
        shutil.copy2(
            ROOT / "stage4/r43/figures" / f"{name}.png",
            ROOT / "report/images" / f"{name}.png",
        )
    findings = {
        "personal_point": personal,
        "SNR40_point": high,
        "high_SNR_conditional_filter_effect_with_code": high_filter,
        "objective_means": objectives,
        "actual_precision": worst,
        "negative_conditional_effects": assessment[
            "supported_negative_conditional_effects"
        ],
        "macros": macros,
        "p_half_analytical_check": {
            "uniform_label_level_variance": uniform_level_variance,
            "expected_NMSE": expected_half_nmse,
            "all_four_mean_intervals_contain_theory": all(
                float(r["lower_nmse"])
                <= expected_half_nmse
                <= float(r["upper_nmse"])
                for r in summary
                if r["joint_condition_id"] == "10"
            ),
            "role": (
                "Post-measurement analytic consistency check; no RNG, "
                "tuning or changes to the frozen mean family"
            ),
        },
        "scope": (
            "Held-out R4.3 only; no statements about untested2D "
            "conditions or exact finite-sample coverage"
        ),
    }
    (folder / "reporting_sources.json").write_text(
        json.dumps(findings, indent=2) + "\n"
    )
    print("R4.3 report macros, five tables and four figure copies generated.")


if __name__ == "__main__":
    main()
