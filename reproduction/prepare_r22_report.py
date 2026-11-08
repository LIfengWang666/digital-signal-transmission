"""Generate R2.2 report values and tables from the saved parameter search."""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

from reproduction.prepare_r21_report import read_rows, tex_number

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Join measured spectra/search data and copy the two R2.2 figures.

    Returns:
        None. Write LaTeX macros/tables and copy report images.
    """
    stage = ROOT / "stage2/r22"
    results = stage / "results"
    report = ROOT / "report"
    generated = report / "generated"
    frozen = json.loads(
        (results / "filter_freeze.json").read_text(encoding="utf-8")
    )
    selected = frozen["selected"]
    fs = frozen["sample_rate_hz"]
    scan = read_rows(results / "parameter_scan.csv")
    prototype = next(
        row for row in scan if row["candidate_id"] == "M513-B80-hann"
    )
    spectral = read_rows(results / "spectral_evidence.csv")
    bands = read_rows(results / "final_bands.csv")
    selected_trials = read_rows(results / "selected_trials.csv")
    prototype_trials = read_rows(results / "prototype_trials.csv")
    high_new = (
        sum(
            float(row["mse"])
            for row in selected_trials
            if int(row["condition_id"]) == 4
        )
        / 3
    )
    high_old = (
        sum(
            float(row["mse"])
            for row in prototype_trials
            if int(row["condition_id"]) == 4
        )
        / 3
    )
    source_summary = {
        row["metric"]: float(row["value"])
        for row in read_rows(ROOT / "stage2/r21/results/summary.csv")
    }
    crossover = 10 * math.log10(
        source_summary["input_power"]
        * (1 - selected["noise_gain_full"])
        / selected["clean_full_mse"]
    )
    improvement = 100 * (1 - selected["score"] / float(prototype["score"]))
    gain_percent = 100 * frozen["gain_guard_dense_max_deviation"]
    weak_db = float(spectral[-1]["peak_to_background_db"])
    width_min = min(float(row["selected_width_3db_hz"]) for row in bands)
    width_max = max(float(row["selected_width_3db_hz"]) for row in bands)
    macros = {
        "RttLength": str(selected["length"]),
        "RttHalfWidth": f"{selected['half_width_hz']:g}",
        "RttWindow": selected["window"].title(),
        "RttDelay": str(selected["delay_samples"]),
        "RttDelayMs": f"{1000 * selected['delay_samples'] / fs:g}",
        "RttFFT": str(selected["fft_length"]),
        "RttCount": str(frozen["candidate_count"]),
        "RttFeasible": str(frozen["feasible_count"]),
        "RttScore": f"{selected['score']:.9f}",
        "RttOldScore": f"{float(prototype['score']):.9f}",
        "RttScoreImprovement": f"{improvement:.2f}",
        "RttGainPercent": f"{gain_percent:.4f}",
        "RttCleanMse": tex_number(selected["clean_full_mse"]),
        "RttNoiseGain": f"{selected['noise_gain_full']:.8f}",
        "RttHighNew": tex_number(high_new),
        "RttHighOld": tex_number(high_old),
        "RttCrossover": f"{crossover:.2f}",
        "RttPsdBackground": tex_number(float(spectral[0]["offband_mean_psd"])),
        "RttWeakPeakBackground": f"{weak_db:.2f}",
        "RttBandWidthMin": f"{width_min:.2f}",
        "RttBandWidthMax": f"{width_max:.2f}",
    }
    (generated / "r22_measurements.tex").write_text(
        "% Generated from stage2/r22/results CSV/JSON.\n"
        + "\n".join(
            r"\newcommand{\%s}{%s}" % (name, value)
            for name, value in macros.items()
        )
        + "\n",
        encoding="utf-8",
    )
    spectral_table = [
        r"\begin{tabular}{rrrrr}",
        r"\toprule",
        r"Centre (Hz) & Peak/background (dB) & Rect. width & Hann width "
        r"& Hamming width \\",
        r"\midrule",
    ]
    for row in spectral:
        spectral_table.append(
            f"{float(row['centre_hz']):.0f} & "
            f"{float(row['peak_to_background_db']):.2f} & "
            f"{float(row['rectangular_width_3db_hz']):.4f} & "
            f"{float(row['hann_width_3db_hz']):.4f} & "
            f"{float(row['hamming_width_3db_hz']):.4f} " + r"\\"
        )
    spectral_table.extend([r"\bottomrule", r"\end{tabular}"])
    (generated / "r22_spectrum_table.tex").write_text(
        "\n".join(spectral_table) + "\n", encoding="utf-8"
    )
    window_table = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Kernel window & $M$ & Half-width (Hz) & $J$ & Clean-input MSE \\",
        r"\midrule",
    ]
    for window in ["rectangular", "hann", "hamming"]:
        rows = [
            row
            for row in scan
            if row["window"] == window and row["feasible"] == "True"
        ]
        row = min(rows, key=lambda item: float(item["score"]))
        window_table.append(
            f"{window.title()} & {row['length']} & "
            f"{float(row['half_width_hz']):g} & "
            f"{float(row['score']):.6f} & $"
            + tex_number(float(row["clean_full_mse"]))
            + r"$ \\"
        )
    window_table.extend([r"\bottomrule", r"\end{tabular}"])
    (generated / "r22_window_table.tex").write_text(
        "\n".join(window_table) + "\n", encoding="utf-8"
    )
    intervals = [
        "$["
        + f"{float(row['nominal_lower_hz']):g},"
        + f"{float(row['nominal_upper_hz']):g}"
        + "]$"
        for row in bands
    ]
    (generated / "r22_band_intervals.tex").write_text(
        ", ".join(intervals[:-1]) + " and " + intervals[-1] + "\n",
        encoding="utf-8",
    )
    for name in [
        "r22_parameter_search.png",
        "r22_response_boundary.png",
        "r22_corrupted_spectral_evidence.png",
    ]:
        shutil.copyfile(stage / "figures" / name, report / "images" / name)
    print("R2.2 report values, tables and figures prepared from saved data.")


if __name__ == "__main__":
    main()
