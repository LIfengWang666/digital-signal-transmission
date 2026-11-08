"""Generate report numbers/tables from CSV and synchronize figures."""

from __future__ import annotations

import csv
import hashlib
import shutil
from math import pi
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def report_source_digest() -> str:
    """Return a digest of TeX, bibliography, images, and supporting PDF
    sources.

    Returns:
        SHA-256 digest of the named report-source inputs.
    """
    report = ROOT / "report"
    digest = hashlib.sha256()
    paths = sorted(
        p
        for p in report.rglob("*")
        if p.suffix in {".tex", ".bib", ".png"}
        or (p.suffix == ".pdf" and p.parent == report / "supporting")
    )
    for path in paths:
        digest.update(path.relative_to(report).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def main() -> None:
    """Read current Stage 1 CSV and populate report assets.

    Returns:
        None. Write the described outputs or raise on invalid data.
    """
    stage = ROOT / "stage1" / "r11"
    report = ROOT / "report"
    generated = report / "generated"
    generated.mkdir(exist_ok=True)
    (report / "images").mkdir(exist_ok=True)
    with (stage / "results" / "summary.csv").open(
        encoding="utf-8", newline=""
    ) as f:
        summary = {
            row["metric"]: float(row["value"]) for row in csv.DictReader(f)
        }
    with (stage / "results" / "components.csv").open(
        encoding="utf-8", newline=""
    ) as f:
        components = list(csv.DictReader(f))
    with (stage / "results" / "checks.csv").open(
        encoding="utf-8", newline=""
    ) as f:
        checks = list(csv.DictReader(f))
    check_errors = {row["check"]: float(row["error"]) for row in checks}
    round_trip_error = check_errors["same-grid FFT inverse round trip"]
    parseval_error = check_errors["Parseval mean-square consistency"]
    hann_difference = summary["max_hann_peak_amplitude_difference"]
    macros = {
        "SampleCount": f"{int(summary['sample_count']):,}",
        "SampleRate": f"{summary['sample_rate_hz']:.0f}",
        "RecordSeconds": f"{summary['record_duration_s']:.0f}",
        "BinSpacing": f"{summary['bin_spacing_hz']:.1f}",
        "SignalPower": f"{summary['mean_square']:.2f}",
        "BandEdgeHz": f"{summary['band_edge_hz']:.0f}",
        "HannGain": f"{summary['hann_coherent_gain']:.10f}",
        "HannDifference": f"{hann_difference:.3g}",
        "ResidualRms": f"{summary['measured_tone_model_residual_rms']:.3g}",
        "OtherBinDb": f"{summary['maximum_other_bin_db']:.2f}",
        "RoundTripError": f"{round_trip_error:.3g}",
        "ParsevalError": f"{parseval_error:.3g}",
    }

    def latex_number(value: str) -> str:
        """Convert a numeric value string into a LaTeX-safe scientific
        expression.

        Args:
            value: Decimal or scientific-notation numeric string from
                measured CSV.

        Returns:
            LaTeX-safe number string with an explicit power of ten if
            needed.
        """
        if "e" not in value:
            return value
        mantissa, exponent = value.split("e")
        return mantissa + r"\times10^{" + str(int(exponent)) + "}"

    (generated / "measurements.tex").write_text(
        "% Generated from CSV; do not edit numerical values manually.\n"
        + "\n".join(
            "\\newcommand{\\" + key + "}{" + latex_number(value) + "}"
            for key, value in macros.items()
        )
        + "\n",
        encoding="utf-8",
    )
    lines = [
        r"\begin{tabular}{rrrrr}",
        r"\toprule",
        r"Bin & Frequency (Hz) & Amplitude & Relative dB & Power (\%) \\",
        r"\midrule",
    ]
    for row in components:
        lines.append(
            f"{int(row['bin'])} & {float(row['frequency_hz']):.0f} & "
            f"{float(row['amplitude_npy_units']):.1f} & "
            f"{float(row['relative_db']):.3f} & "
            f"{100*float(row['power_fraction']):.4f}" + r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (generated / "components_table.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    with (stage / "results/bandwidth_sensitivity.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        sensitivity = list(csv.DictReader(stream))
    lines = [
        r"\begin{tabular}{rrrrr}",
        r"\toprule",
        r"Threshold (dB) & R/H peaks & R/H bins & "
        r"R edge (Hz) & H edge (Hz) \\",
        r"\midrule",
    ]
    thresholds = sorted({float(row["threshold_db"]) for row in sensitivity})
    for threshold in reversed(thresholds):
        pair = {
            row["window"]: row
            for row in sensitivity
            if float(row["threshold_db"]) == threshold
        }
        rect, hann = pair["rectangular"], pair["symmetric_hann"]
        lines.append(
            f"{threshold:.0f} & "
            f"{rect['peak_count']}/{hann['peak_count']} & "
            f"{rect['all_bins_above_threshold']}/"
            f"{hann['all_bins_above_threshold']} & "
            f"{float(rect['highest_peak_centre_hz']):.0f} & "
            f"{float(hann['highest_peak_centre_hz']):.0f}" + r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (generated / "bandwidth_sensitivity_table.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    with (ROOT / "metadata/sampling_identity_checks.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        folding_checks = list(csv.DictReader(stream))
    folding_error = max(
        float(row["maximum_coefficient_error"]) for row in folding_checks
    )
    (generated / "folding_measurements.tex").write_text(
        "% Generated from the independent DFT-folding check CSV.\n"
        + r"\newcommand{\FoldingError}{"
        + latex_number(f"{folding_error:.3g}")
        + "}\n",
        encoding="utf-8",
    )
    for name in ["reference_spectrum", "window_comparison"]:
        shutil.copy2(
            stage / "figures" / f"{name}.png",
            report / "images" / f"{name}.png",
        )
    sampling = ROOT / "stage1" / "r12"
    with (sampling / "results" / "summary.csv").open(
        encoding="utf-8", newline=""
    ) as f:
        sampling_summary = {
            row["metric"]: float(row["value"]) for row in csv.DictReader(f)
        }
    with (sampling / "results" / "rate_comparison.csv").open(
        encoding="utf-8", newline=""
    ) as f:
        rates = list(csv.DictReader(f))
    with (sampling / "results" / "component_retention.csv").open(
        encoding="utf-8", newline=""
    ) as f:
        retained = list(csv.DictReader(f))
    critical_rate = sampling_summary["nyquist_rate_hz"]
    edge = sampling_summary["measured_band_edge_hz"]
    critical = next(
        r for r in rates if float(r["sample_rate_hz"]) == critical_rate
    )
    critical_tone = next(
        r
        for r in retained
        if float(r["sample_rate_hz"]) == critical_rate
        and float(r["source_frequency_hz"]) == edge
    )
    transmission_rate = sampling_summary["selected_sample_rate_hz"]
    transmission_count = int(sampling_summary["selected_sample_count"])
    maximum_error = sampling_summary["selected_maximum_absolute_error"]
    critical_amplitude = float(critical_tone["measured_amplitude"])
    sampling_macros = {
        "TransmissionRate": f"{transmission_rate:.0f}",
        "TransmissionCount": f"{transmission_count:,}",
        "SamplingGuard": f"{sampling_summary['selected_guard_hz']:.0f}",
        "ReplicaGap": f"{sampling_summary['selected_replica_gap_hz']:.0f}",
        "SamplingMse": f"{sampling_summary['selected_mse']:.3g}",
        "SamplingNmse": f"{sampling_summary['selected_nmse']:.3g}",
        "SamplingMaxError": f"{maximum_error:.3g}",
        "CriticalMse": f"{float(critical['mse']):.3g}",
        "CriticalToneAmplitude": f"{critical_amplitude:.3g}",
    }
    (generated / "sampling_measurements.tex").write_text(
        "% Generated from R1.2 measured CSV.\n"
        + "\n".join(
            "\\newcommand{\\" + key + "}{" + latex_number(value) + "}"
            for key, value in sampling_macros.items()
        )
        + "\n",
        encoding="utf-8",
    )
    table = [
        r"\begin{tabular}{rrrrrr}",
        r"\toprule",
        r"$f_s$ (kHz) & $M$ & $f_N$ (kHz) & Samples & Raw ratio & MSE \\",
        r"\midrule",
    ]
    for row in rates:
        table.append(
            f"{float(row['sample_rate_hz']) / 1000:g} & "
            f"{int(row['decimation_factor'])} & "
            f"{float(row['nyquist_frequency_hz']) / 1000:g} & "
            f"{int(row['sample_count']):,} & "
            f"{float(row['payload_ratio_to_selected']):g} & $"
            + latex_number(f"{float(row['mse']):.3g}")
            + r"$ \\"
        )
    table += [r"\bottomrule", r"\end{tabular}"]
    (generated / "sampling_table.tex").write_text(
        "\n".join(table) + "\n", encoding="utf-8"
    )
    for name in ["sampling_rate_decision", "selected_rate_reconstruction"]:
        shutil.copy2(
            sampling / "figures" / f"{name}.png",
            report / "images" / f"{name}.png",
        )
    aliasing = ROOT / "stage1" / "r13"
    with (aliasing / "results/summary.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        alias_summary = {
            row["metric"]: float(row["value"])
            for row in csv.DictReader(stream)
        }
    with (aliasing / "results/alias_mapping.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        aliases = list(csv.DictReader(stream))
    with (aliasing / "results/comparison.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        comparisons = list(csv.DictReader(stream))
    aliases_macros = {
        "UnderMse": alias_summary["under_mse"],
        "UnderRmse": alias_summary["under_rmse"],
        "UnderNmse": alias_summary["under_nmse"],
        "UnderMaxError": alias_summary["under_maximum_error"],
        "AliasBaselineMse": alias_summary["baseline_mse"],
    }
    (generated / "aliasing_measurements.tex").write_text(
        "% Generated from R1.3 measured CSV.\n"
        + "\n".join(
            "\\newcommand{\\" + key + "}{" + latex_number(f"{value:.3g}") + "}"
            for key, value in aliases_macros.items()
        )
        + "\n",
        encoding="utf-8",
    )
    table = [
        r"\begin{tabular}{rrrrrr}",
        r"\toprule",
        r"Source (Hz) & Signed (Hz) & Predicted (Hz) & "
        r"Measured (Hz) & Amplitude & $\phi/\pi$ \\",
        r"\midrule",
    ]
    for row in aliases:
        if float(row["sample_rate_hz"]) != alias_summary["under_fs_hz"]:
            continue
        table.append(
            f"{float(row['source_frequency_hz']):.0f} & "
            f"{float(row['signed_frequency_hz']):.0f} & "
            f"{float(row['predicted_frequency_hz']):.0f} & "
            f"{float(row['measured_frequency_hz']):.0f} & "
            f"{float(row['measured_amplitude']):.1f} & "
            f"{float(row['measured_phase_rad']) / pi:.3f}" + r" \\"
        )
    table += [r"\bottomrule", r"\end{tabular}"]
    (generated / "aliasing_mapping_table.tex").write_text(
        "\n".join(table) + "\n", encoding="utf-8"
    )
    table = [
        r"\begin{tabular}{rrrrr}",
        r"\toprule",
        r"$f_s$ (kHz) & MSE & RMSE & NMSE & Peak error \\",
        r"\midrule",
    ]
    for row in comparisons:
        values = [
            latex_number(f"{float(row[k]):.3g}")
            for k in ["mse", "rmse", "nmse", "maximum_absolute_error"]
        ]
        table.append(
            f"{float(row['sample_rate_hz']) / 1000:g} & $"
            + "$ & $".join(values)
            + r"$ \\"
        )
    table += [r"\bottomrule", r"\end{tabular}"]
    (generated / "aliasing_error_table.tex").write_text(
        "\n".join(table) + "\n", encoding="utf-8"
    )
    for name in ["aliasing_spectra", "aliasing_reconstruction_error"]:
        shutil.copy2(
            aliasing / "figures" / f"{name}.png",
            report / "images" / f"{name}.png",
        )
    shutil.copy2(
        ROOT / "stage1/r14/figures/sampling_decision.png",
        report / "images/sampling_decision.png",
    )
    print("R1.1--R1.4 report numbers, tables and images regenerated.")
