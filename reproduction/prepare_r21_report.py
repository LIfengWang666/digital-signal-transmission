"""Build the R2.1 report assets from saved method-selection measurements."""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_rows(path: Path) -> list[dict[str, str]]:
    """Read a saved measurement CSV.

    Args:
        path: UTF-8 CSV source with a header.

    Returns:
        String-valued records indexed by column names.
    """
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def tex_number(value: float) -> str:
    """Format a measured number with a readable power of ten.

    Args:
        value: Finite measured quantity.

    Returns:
        A LaTeX expression suitable for math mode.
    """
    mantissa, exponent = f"{value:.4e}".split("e")
    return mantissa + r"\times10^{" + str(int(exponent)) + "}"


def main() -> None:
    """Join R2.1 CSV and copy plots into the reproducible report.

    Returns:
        None. Write generated macros/tables and two report images.
    """
    stage = ROOT / "stage2/r21"
    results = stage / "results"
    report = ROOT / "report"
    generated = report / "generated"
    generated.mkdir(exist_ok=True)
    summary = {
        row["metric"]: float(row["value"])
        for row in read_rows(results / "summary.csv")
    }
    noise = read_rows(results / "noise_trials.csv")
    anchor = next(
        row
        for row in noise
        if int(row["condition_id"]) == 5 and int(row["repeat_id"]) == 0
    )
    clean = {
        row["method"]: row
        for row in read_rows(results / "clean_filter_distortion.csv")
    }
    macros = {
        "RtwoPersonalSnr": f"{summary['personal_snr_db']:.9f}",
        "RtwoUniform": f"{summary['parameter_uniform_draw']:.12f}",
        "RtwoRealisedSnr": f"{float(anchor['realised_snr_db']):.6f}",
        "RtwoFftError": tex_number(summary["max_direct_fft_error"]),
        "RtwoCheckCount": f"{int(summary['check_count'])}",
        "RtwoDrawCount": f"{int(summary['design_noise_records'])}",
        "RtwoMaClean": tex_number(float(clean["ma3"]["clean_full_mse"])),
        "RtwoMultiClean": tex_number(
            float(clean["multiband"]["clean_full_mse"])
        ),
        "RtwoMultiInterior": tex_number(
            float(clean["multiband"]["clean_interior_mse"])
        ),
        "RtwoMultiEdge": tex_number(
            float(clean["multiband"]["clean_edge_mse"])
        ),
    }
    (generated / "r21_measurements.tex").write_text(
        "% Generated from stage2/r21/results CSV.\n"
        + "\n".join(
            r"\newcommand{\%s}{%s}" % (name, value)
            for name, value in macros.items()
        )
        + "\n",
        encoding="utf-8",
    )
    responses = read_rows(results / "prototype_response.csv")
    response_table = [
        r"\begin{tabular}{rrrr}",
        r"\toprule",
        r"Frequency (Hz) & Mean: signed gain & Low-pass: $|H|$ "
        r"& Multiband: $|H|$ \\",
        r"\midrule",
    ]
    for frequency in [440, 1320, 2200, 6000]:
        selected = {
            row["method"]: row
            for row in responses
            if float(row["frequency_hz"]) == frequency
        }
        response_table.append(
            f"{frequency} & {float(selected['ma3']['real']):.6f} & "
            f"{float(selected['wide_lowpass']['magnitude']):.6f} & "
            f"{float(selected['multiband']['magnitude']):.6f} " + r"\\"
        )
    response_table.extend([r"\bottomrule", r"\end{tabular}"])
    (generated / "r21_response_table.tex").write_text(
        "\n".join(response_table) + "\n", encoding="utf-8"
    )
    compared = read_rows(results / "approach_comparison.csv")
    pilot_table = [
        r"\begin{tabular}{rrrrr}",
        r"\toprule",
        r"Input SNR (dB) & No filter & Three-point mean & "
        r"Wide low-pass & Multiband \\",
        r"\midrule",
    ]
    for condition in range(5):
        condition_rows = [
            row for row in compared if int(row["condition_id"]) == condition
        ]
        snr = float(condition_rows[0]["target_snr_db"])
        means = []
        for method in ["none", "ma3", "wide_lowpass", "multiband"]:
            rows = [
                float(row["mse"])
                for row in condition_rows
                if row["method"] == method
            ]
            means.append(sum(rows) / len(rows))
        pilot_table.append(
            f"{snr:.0f} & "
            + " & ".join("$" + tex_number(v) + "$" for v in means)
            + r" \\"
        )
    pilot_table.extend([r"\bottomrule", r"\end{tabular}"])
    (generated / "r21_pilot_table.tex").write_text(
        "\n".join(pilot_table) + "\n", encoding="utf-8"
    )
    for name in [
        "approach_spectrum_response.png",
        "approach_pilot_comparison.png",
    ]:
        shutil.copyfile(stage / "figures" / name, report / "images" / name)
    print("R2.1 report assets generated from CSV.")


if __name__ == "__main__":
    main()
